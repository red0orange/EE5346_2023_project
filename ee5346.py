import sys
import os
import os.path as osp
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import torch

from step_4_our_interface import Interface

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus", 'submodules/LoFTR/src'))
sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus"))
from src.KeypointFreeSfM import coarse_match_worker, fine_match_worker
from src.utils.data_io import grayscale2tensor
from src.utils.metric_utils import ransac_PnP

sys.path.append(project_root_dir)
sys.path.append(os.path.join(project_root_dir, "utils"))
from utils.colmap_utils.update_database_camera import camera_to_database
from vis_utils.image_matching import vis_image_matching


def vis(image1, image2, kpts1, kpts2, scores, save_path=None):
    matches = np.stack([np.arange(len(kpts1)), np.arange(len(kpts1))], axis=1)
    good_matches = matches[scores > 0.5]
    result_img = vis_image_matching(image1, image2, kpts1, kpts2, good_matches, None)
    cv2.imwrite(save_path, result_img)
    pass


class CachedMatcher(object):
    def __init__(self, cached_dir):
        self.cached_dir = cached_dir
        os.makedirs(name=self.cached_dir, exist_ok=True)
        # 构造 matcher
        self.interface = Interface()
        pass

    def generate_identity(self, image_path_1, image_path_2):
        # 生成 identity 的方法随具体数据集而定
        identity_1 = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path_1)))) + "_" + os.path.basename(image_path_1).rsplit(".", maxsplit=1)[0]
        identity_2 = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path_2)))) + "_" + os.path.basename(image_path_2).rsplit(".", maxsplit=1)[0]
        identity = identity_1 + "--" + identity_2
        return identity

    def match(self, image_path_1, image_path_2, image_1=None, image_2=None):
        identity = self.generate_identity(image_path_1, image_path_2)
        cached_path = os.path.join(self.cached_dir, identity + ".npy")

        if os.path.exists(cached_path):
            matches = np.load(cached_path, allow_pickle=True).item()
        else:
            if image_1 is None:
                image_1 = cv2.imread(image_path_1)
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
            if image_2 is None:
                image_2 = cv2.imread(image_path_2)
                image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
            
            data = {
                "image0": grayscale2tensor(image_1)[None],
                "image1": grayscale2tensor(image_2)[None],  # 1*1*H*W because no dataloader operation, if batch: 1*H*W
                "scale0": torch.Tensor([1, 1])[None],
                "scale1": torch.Tensor([1, 1])[None],  # 1*2
                # "f_name0": osp.basename(db_image_path).rsplit('.', 1)[0],
                # "f_name1": osp.basename(query_image_path).rsplit('.', 1)[0],
            }
            data_c = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
            }

            self.interface.coarse_matcher(data_c)
            data_c = {k: v.cpu().detach() if isinstance(v, torch.Tensor) else v for k, v in data_c.items()}
            image1_kps = data_c["mkpts0_c"].cpu().numpy()
            image2_kps = data_c["mkpts1_c"].cpu().numpy()
            scores = data_c["mconf"].cpu().numpy()

            result_dict = {}
            result_dict["image1_kps"] = image1_kps
            result_dict["image2_kps"] = image2_kps
            result_dict["scores"] = scores
            matches = result_dict

            np.save(cached_path, matches)
        return matches


if __name__ == "__main__":
    data_root = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/ee5346_dataset"
    result_root = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/ee5346_dataset/results"
    # test_txt_path = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/ee5346_dataset/robotcar_qAutumn_dbSunCloud_diff_final.txt"
    # test_txt_path = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/ee5346_dataset/robotcar_qAutumn_dbNight_diff_final.txt"
    # test_txt_path = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/ee5346_dataset/robotcar_qAutumn_dbNight_easy_final.txt"
    test_txt_path = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/ee5346_dataset/robotcar_qAutumn_dbSunCloud_easy_final.txt"

    test_data = np.loadtxt(test_txt_path, dtype=object, delimiter=",")
    # preprocess
    query_image_paths = np.array([i.strip() for i in test_data[:, 0]])
    ref_image_paths   = np.array([i.strip() for i in test_data[:, 1]])
    gt                = np.array([i.strip() for i in test_data[:, 2]])
    query_image_paths = [os.path.join(data_root, os.path.dirname(i).split("_")[0] + "_val", "stereo", "centre", os.path.basename(i)) for i in query_image_paths]
    ref_image_paths   = [os.path.join(data_root, os.path.dirname(i).split("_")[0] + "_val", "stereo", "centre", os.path.basename(i)) for i in ref_image_paths]
    gt                = gt.astype(np.int)
    # 964.828979 964.828979 643.788025 484.407990
    # K = np.array([[964.828979, 0, 643.788025], [0, 964.828979, 484.407990], [0, 0, 1]])
    # 图像缩小一半
    K = np.array([[964.828979 / 2, 0, 643.788025 / 2], [0, 964.828979 / 2, 484.407990 / 2], [0, 0, 1]])
    image_hw = [480, 640]

    # ref
    db_image_root = os.path.dirname(ref_image_paths[0])
    db_image_paths = [os.path.join(db_image_root, i) for i in os.listdir(db_image_root)]
    db_image_paths = sorted(db_image_paths)
    db_image_names = [os.path.basename(i) for i in db_image_paths]

    # 构造 matcher
    # interface = Interface()
    matcher = CachedMatcher(cached_dir=os.path.join(result_root, "cached_matching"))

    # 对每一张 query image 进行验证
    direct_F_verify_result = []
    direct_score_verify_result = []
    covis_score_verify_result = []
    covis_score_verify_result_1 = []
    covis_score_verify_result_2 = []

    debug_precision_list = []
    debug_precision_std_covis_num = []
    debug_precision_min_covis_num = []
    debug_precision_max_covis_num = []
    debug_precision_mean_covis_num = []

    db_map_num = 5
    db_map_delta = 5
    for i, (query_image_path, ref_image_path) in tqdm(enumerate(zip(query_image_paths, ref_image_paths)), total=len(query_image_paths)):
        # if i > 50:
        #     break

        query_image_path = query_image_paths[i]
        query_ori_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(query_ori_image, cv2.COLOR_BGR2GRAY)
        ref_image_path = ref_image_paths[i]
        ref_ori_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_ori_image, cv2.COLOR_BGR2GRAY)

        ref_image_name = os.path.basename(ref_image_path)
        db_image_index = db_image_names.index(ref_image_name)
        left_index  = max(db_image_index - db_map_num*db_map_delta, 0)
        right_index = min(db_image_index + db_map_num*db_map_delta, len(db_image_paths)-1)
        sampled_indices = [int(db_image_index + i*db_map_delta) for i in range(-db_map_num, db_map_num+1)]
        ref_db_image_paths = [db_image_paths[i] for i in sampled_indices if left_index <= i <= right_index]

        ######################## 结果变量
        # ref to query
        query_to_ref_dict = None
        # ref_db to query
        ref_db_to_query_dicts = {}
        # ref_db to ref
        ref_db_to_ref_dicts = {}

        ######################## direct match
        matches = matcher.match(query_image_path, ref_image_path, query_image, ref_image)
        vis(query_image, ref_image, matches["image1_kps"], matches["image2_kps"], matches["scores"], osp.join(result_root, "query_to_ref.png"))
        query_to_ref_dict = matches

        ######################## direct verify
        F, mask = cv2.findFundamentalMat(matches["image1_kps"], matches["image2_kps"], cv2.FM_RANSAC, 1.0, 0.999)
        inlier_ratio = np.sum(mask) / mask.shape[0]
        direct_F_verify_result.append(inlier_ratio)
        direct_score_verify_result.append(matches["scores"].max())

        ######################## ref match
        for ref_db_image_path in tqdm(ref_db_image_paths, desc="ref_db", total=len(ref_db_image_paths)):
            ref_db_ori_image = cv2.imread(ref_db_image_path)
            ref_db_image = cv2.cvtColor(ref_db_ori_image, cv2.COLOR_BGR2GRAY)
            # ref_db to query
            matches = matcher.match(ref_db_image_path, query_image_path, ref_db_image, query_image)
            ref_db_to_query_dicts[ref_db_image_path] = matches

            # ref_db to ref
            matches = matcher.match(ref_db_image_path, ref_image_path, ref_db_image, ref_image)
            ref_db_to_ref_dicts[ref_db_image_path] = matches
            pass

        ######################## covis verify
        base_score = query_to_ref_dict["scores"].max()

        # 提高 Precision
        good_match_loftr_thres = 0.3
        covis_pixel_thres      = 8
        good_covis_thres       = 0.5

        good_match = query_to_ref_dict["scores"] > good_match_loftr_thres
        query_kps  = query_to_ref_dict["image1_kps"][good_match]
        ref_kps    = query_to_ref_dict["image2_kps"][good_match]

        covis_kp_nums = []
        for ref_db_image_path in ref_db_image_paths:
            ref_db_to_query_result   = ref_db_to_query_dicts[ref_db_image_path]
            query_to_ref_db_kps_dict = {tuple(i): tuple(j) for i, j in zip(ref_db_to_query_result["image2_kps"], ref_db_to_query_result["image1_kps"])}
            final_query_ref_db_kps   = [query_to_ref_db_kps_dict[tuple(i)] if tuple(i) in query_to_ref_db_kps_dict else -1 for i in query_kps]

            ref_db_to_ref_result   = ref_db_to_ref_dicts[ref_db_image_path]
            ref_to_ref_db_kps_dict = {tuple(i): tuple(j) for i, j in zip(ref_db_to_ref_result["image2_kps"], ref_db_to_ref_result["image1_kps"])}
            final_ref_ref_db_kps   = [ref_to_ref_db_kps_dict[tuple(i)] if tuple(i) in ref_to_ref_db_kps_dict else -1 for i in ref_kps]

            cnt = 0
            for kp_i in range(len(query_kps)):
                if final_query_ref_db_kps[kp_i] == -1 or final_ref_ref_db_kps[kp_i] == -1:
                    continue
                query_to_ref_db_kp = final_query_ref_db_kps[kp_i]
                ref_to_ref_db_kp   = final_ref_ref_db_kps[kp_i]
                if np.linalg.norm(np.array(query_to_ref_db_kp) - np.array(ref_to_ref_db_kp)) < covis_pixel_thres:
                    cnt += 1
                    pass
                pass
            covis_kp_nums.append(cnt)

        kp_num = query_kps.shape[0]
        mean_covis_kp_num = sorted(covis_kp_nums)[len(covis_kp_nums)//2]
        precision_score_ratio = mean_covis_kp_num / kp_num

        # 提高 Recall
        best_score = 0
        for ref_db_image_path in ref_db_image_paths:
            if len(ref_db_to_query_dicts[ref_db_image_path]["scores"]) == 0:
                continue
            ref_db_query_score = ref_db_to_query_dicts[ref_db_image_path]["scores"].max()
            if ref_db_query_score > best_score:
                best_score = ref_db_query_score
                pass
        recall_score_ratio = max(1, best_score / base_score)

        debug_precision_list.append(precision_score_ratio)
        debug_precision_max_covis_num.append(np.max(covis_kp_nums))
        debug_precision_min_covis_num.append(np.min(covis_kp_nums))
        debug_precision_mean_covis_num.append(np.mean(covis_kp_nums))
        debug_precision_std_covis_num.append(np.std(covis_kp_nums))
        precision_score_ratio = 1
        if np.std(covis_kp_nums) < 50:
            precision_score_ratio = 0.8

        print(precision_score_ratio)
        final_score = precision_score_ratio * recall_score_ratio * base_score
        covis_score_verify_result.append((0.8 if np.std(covis_kp_nums) < 40 else 1) * recall_score_ratio * base_score)
        covis_score_verify_result_1.append((0.9 if np.std(covis_kp_nums) < 40 else 1) * recall_score_ratio * base_score)
        covis_score_verify_result_2.append((0.9 if np.std(covis_kp_nums) < 40 else 1) * recall_score_ratio * base_score)
    

    def draw_pr_curve(save_name, precision, recall, average_precision):
        plt.plot(recall, precision, label="AP={:.3f}".format(average_precision))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision-Recall Curves")
        plt.savefig(save_name)
        plt.close()

    def get_precision100_recall(precision, recall):
        precision = precision[::-1]
        recall    = recall[::-1]
        precision_100_length = sum(precision == 1)
        if precision_100_length == 0:
            return 0
        else:
            return recall[precision_100_length-1]
    
    precision, recall, _ = precision_recall_curve(gt[:len(direct_F_verify_result)], direct_F_verify_result)
    average_precision = average_precision_score(gt[:len(direct_F_verify_result)], direct_F_verify_result)
    # print(precision, recall)
    print(average_precision)
    print("Recall@100%Precision: ", get_precision100_recall(precision, recall))
    draw_pr_curve("pr_curve_loftr_F.png", precision, recall, average_precision)


    precision, recall, _ = precision_recall_curve(gt[:len(direct_score_verify_result)], direct_score_verify_result)
    average_precision = average_precision_score(gt[:len(direct_score_verify_result)], direct_score_verify_result)
    # print(precision, recall)
    print(average_precision)
    print("Recall@100%Precision: ", get_precision100_recall(precision, recall))
    draw_pr_curve("pr_curve_loftr_direct_score.png", precision, recall, average_precision)

    precision, recall, _ = precision_recall_curve(gt[:len(covis_score_verify_result)], covis_score_verify_result)
    average_precision = average_precision_score(gt[:len(covis_score_verify_result)], covis_score_verify_result)
    # print(precision, recall)
    print(average_precision)
    print("Recall@100%Precision: ", get_precision100_recall(precision, recall))

    draw_pr_curve("pr_curve_loftr_our_score.png", precision, recall, average_precision)

    average_precision = average_precision_score(gt[:len(covis_score_verify_result_1)], covis_score_verify_result_1)
    print(average_precision)
    print("Recall@100%Precision: ", get_precision100_recall(precision, recall))
    average_precision = average_precision_score(gt[:len(covis_score_verify_result_2)], covis_score_verify_result_2)
    print(average_precision)
    print("Recall@100%Precision: ", get_precision100_recall(precision, recall))

    debug_precision_list = np.array(debug_precision_list)
    debug_precision_std_covis_num = np.array(debug_precision_std_covis_num)
    debug_precision_min_covis_num = np.array(debug_precision_min_covis_num)
    debug_precision_max_covis_num = np.array(debug_precision_max_covis_num)
    debug_precision_mean_covis_num = np.array(debug_precision_mean_covis_num)
    pass
