import sys
import os
import os.path as osp
from collections import Counter
import shutil

from tqdm import tqdm
import cv2
import torch
import numpy as np
np.warnings.filterwarnings('ignore')

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus", 'submodules/LoFTR/src'))
sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus"))
from src.KeypointFreeSfM import coarse_match_worker, fine_match_worker
from src.utils.data_io import grayscale2tensor
from src.utils.metric_utils import ransac_PnP

sys.path.append(os.path.join(project_root_dir, "utils"))
from file_utils.file import *
from common_utils.crop_image import *
from vis_utils.vis_pose import *
from vis_utils.image_matching import vis_image_matching
from colmap_utils.read_write_model import read_model, write_model


def compute_RT_errors(sRT_1, sRT_2):
    """
    Base on NCOS
    Args:
        sRT_1: [4, 4]. homogeneous affine transformation
        sRT_2: [4, 4]. homogeneous affine transformation

    Returns:
        theta: angle difference of R in degree
        shift: l2 difference of T in centimeter
    """
    # make sure the last row is [0, 0, 0, 1]
    if sRT_1 is None or sRT_2 is None:
        return -1
    try:
        assert np.array_equal(sRT_1[3, :], sRT_2[3, :])
        assert np.array_equal(sRT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT_1[3, :], sRT_2[3, :])
        exit()

    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    T1 = sRT_1[:3, 3]
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
    T2 = sRT_2[:3, 3]

    R = R1 @ R2.transpose()
    cos_theta = (np.trace(R) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result


class Interface(object):
    def __init__(self, data_dir=None):
        # 构建 matcher
        # coarse
        print("Building matcher...")
        # coarse matcher
        coarse_matcher_cfg = {
            "weight_path": osp.join(project_root_dir, "OnePose_Plus_Plus", "weight/LoFTR_wsize9.ckpt"),
            "seed": 666,
        }
        self.coarse_matcher = coarse_match_worker.build_model(coarse_matcher_cfg)
        self.coarse_matcher.cuda()
        # fine
        fine_matcher_cfg = {
            "model": {
                "weight_path": osp.join(project_root_dir, "OnePose_Plus_Plus", "weight/LoFTR_wsize9.ckpt"),
                "seed": 666,
            },
            "extract_feature_method": "fine_match_backbone",
            "ray": {
                "slurm": False,
                "n_workers": 4,
                "n_cpus_per_worker": 1,
                "n_gpus_per_worker": 0.25,
                "local_mode": False,
            },
        }
        self.fine_matcher = fine_match_worker.build_model(fine_matcher_cfg["model"])
        self.fine_matcher.cuda()

        if data_dir is None:
            pass
        else:
            self.set_db(data_dir)
        pass

    def set_db(self, data_dir):
        self.data_dir       = data_dir
        self.input_data_dir = osp.join(self.data_dir, "0_input_data")
        self.map_dir        = osp.join(self.data_dir, "1_map")

        self.debug_dir = "exp"

        self.object_ids = [int(i.split("_")[0]) for i in os.listdir(os.path.join(self.input_data_dir, "bbox"))]
        self.db_query_images, coarse_model, fine_model = self.get_map_data(self.data_dir)
        self.db_coarse_cameras, self.db_coarse_images, self.db_coarse_points3D = coarse_model
        self.db_fine_cameras, self.db_fine_images, self.db_fine_points3D = fine_model
        pass

    @staticmethod
    def get_map_data(scene_path):
        points = {}

        coarse_sfm_model_dir = osp.join(scene_path, "1_map", "full_map", "sfm_ws", "model_coarse")
        coarse_cameras, coarse_images, coarse_points3D = read_model(coarse_sfm_model_dir)
        fine_sfm_model_dir = osp.join(scene_path, "1_map", "full_map", "sfm_ws", "model")
        fine_cameras, fine_images, fine_points3D = read_model(fine_sfm_model_dir)

        query_images = []
        scene_name = os.path.basename(scene_path)
        K = np.load(os.path.join(scene_path, "0_input_data", "intrinsics.npy"))
        for image_id in coarse_images.keys():
            image_name = osp.basename(coarse_images[image_id].name)
            image_index = int(image_name.split(".")[0])

            target_img_path = os.path.join(scene_path, "0_input_data/color/{}.png".format(image_index))
            target_img = cv2.imread(target_img_path)
            target_img_pose_path = os.path.join(scene_path, "0_input_data/poses/{}.npy".format(image_index))
            target_img_bboxs_path = os.path.join(scene_path, "0_input_data/2d_bbox/{}.npy".format(image_index))
            target_img_object_poses_path = os.path.join(scene_path, "0_input_data/object_poses/{}.npy".format(image_index))  # @note TODO

            target_img_camera_pose = np.load(target_img_pose_path)
            target_img_bboxs = np.load(target_img_bboxs_path, allow_pickle=True).item()
            target_object_poses = np.load(target_img_object_poses_path, allow_pickle=True).item()

            image_dict = {}
            for object_id, bbox in target_img_bboxs.items():
                target_object_3d_bbox_path = os.path.join(scene_path, "0_input_data/bbox/{}_bbox_corners.npy".format(object_id))
                ori_fake_pose_path = os.path.join(scene_path, "0_input_data", "ori_fake_poses", "object_{}.npy").format(object_id)
                pybullet_pose_path = os.path.join(scene_path, "0_input_data", "pybullet_poses", "{}.npy").format(object_id)

                ori_fake_pose = np.load(ori_fake_pose_path)
                pybullet_pose = np.load(pybullet_pose_path)
                bbox_3d = np.load(target_object_3d_bbox_path)
                bbox = list(map(int, bbox))
                pose = target_object_poses[object_id]

                query_image_dict = {}
                query_image_dict["scene_name"] = scene_name
                query_image_dict["image_id"] = image_id
                query_image_dict["image_path"] = target_img_path
                query_image_dict["camera_pose"] = target_img_camera_pose
                query_image_dict["K"] = K
                query_image_dict["bbox_3d"] = bbox_3d
                query_image_dict["bbox"] = bbox
                query_image_dict["object_id"] = object_id
                query_image_dict["ori_fake_pose"] = ori_fake_pose
                query_image_dict["pybullet_pose"] = pybullet_pose
                query_image_dict["pose"] = pose

                image_dict[object_id] = query_image_dict
                query_images.append(query_image_dict)
            coarse_images[image_id].query_images = image_dict
            
        return query_images, (coarse_cameras, coarse_images, coarse_points3D), (fine_cameras, fine_images, fine_points3D)

    @staticmethod
    def get_query_objects(scene_path):
        query_images = {}

        print("Getting query objects...")
        scene_name = os.path.basename(scene_path)
        object_class_ids = [int(i.split("_")[0]) for i in os.listdir(os.path.join(scene_path, "0_input_data", "bbox"))]
        for object_class_id in object_class_ids:
            if object_class_id not in query_images.keys():
                query_images[object_class_id] = []

        images_path = os.path.join(scene_path, "0_input_data", "color")
        K = np.load(os.path.join(scene_path, "0_input_data", "intrinsics.npy"))
        indexes = [int(i.rsplit(".", maxsplit=1)[0]) for i in os.listdir(images_path)]
        for index in tqdm(indexes, desc="Query Images"):
            target_img_path = os.path.join(scene_path, "0_input_data/color/{}.png".format(index))
            target_img = cv2.imread(target_img_path)
            target_img_pose_path = os.path.join(scene_path, "0_input_data/poses/{}.npy".format(index))
            target_img_bboxs_path = os.path.join(scene_path, "0_input_data/2d_bbox/{}.npy".format(index))
            target_img_object_poses_path = os.path.join(scene_path, "0_input_data/object_poses/{}.npy".format(index))  # @note TODO

            target_img_camera_pose = np.load(target_img_pose_path)
            target_img_bboxs = np.load(target_img_bboxs_path, allow_pickle=True).item()
            target_object_poses = np.load(target_img_object_poses_path, allow_pickle=True).item()

            for object_id in object_class_ids:
                if object_id not in target_img_bboxs.keys():
                    continue

                bbox = list(map(int, target_img_bboxs[object_id]))
                pose = target_object_poses[object_id]

                query_image_dict = {}
                query_image_dict["scene_name"] = scene_name
                query_image_dict["image_path"] = target_img_path
                query_image_dict["camera_pose"] = target_img_camera_pose
                query_image_dict["K"] = K
                query_image_dict["bbox"] = bbox
                query_image_dict["pose"] = pose

                query_images[object_id].append(query_image_dict)
        return query_images

    def infer(self, query_image_dict):
        infer_result_dict = {}

        query_image_path = query_image_dict["image_path"]
        query_K = query_image_dict["K"]
        query_bbox = query_image_dict["bbox"]

        save_dir = osp.join(self.debug_dir, "1")
        if osp.exists(save_dir):  shutil.rmtree(save_dir) 
        os.makedirs(name=save_dir)

        ori_query_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(ori_query_image, cv2.COLOR_BGR2GRAY)
        query_image_hw = query_image.shape[:2]
        query_K = query_K
        mask_query_image = np.zeros_like(query_image)
        mask_query_image[query_bbox[1]:query_bbox[3], query_bbox[0]:query_bbox[2]] = query_image[query_bbox[1]:query_bbox[3], query_bbox[0]:query_bbox[2]]

        ####################################
        #### 1. LoFTR Coarse Matching
        ####################################
        db_matching_result = []
        # TODO 用唯一标识来记录 matching 的结果，避免每次都要重新计算
        for i, db_image_dict in enumerate(self.db_query_images):
            db_image_path = db_image_dict["image_path"]
            db_bbox       = db_image_dict["bbox"]
            db_K          = db_image_dict["K"]
            db_bbox_3d    = db_image_dict["bbox_3d"]
            ori_db_image = cv2.imread(db_image_path)
            db_image = cv2.cvtColor(ori_db_image, cv2.COLOR_BGR2GRAY)
            mask_db_image = np.zeros_like(db_image)
            mask_db_image[db_bbox[1]:db_bbox[3], db_bbox[0]:db_bbox[2]] = db_image[db_bbox[1]:db_bbox[3], db_bbox[0]:db_bbox[2]]

            data = {
                "image0": grayscale2tensor(mask_db_image)[None],
                "image1": grayscale2tensor(mask_query_image)[None],  # 1*1*H*W because no dataloader operation, if batch: 1*H*W
                "scale0": torch.Tensor([1, 1])[None],
                "scale1": torch.Tensor([1, 1])[None],  # 1*2
                "f_name0": osp.basename(db_image_path).rsplit('.', 1)[0],
                "f_name1": osp.basename(query_image_path).rsplit('.', 1)[0],
                # "frameID": idx,
                # "pair_key": (img_path0, img_path1),
            }
            data_c = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
            }
            self.coarse_matcher(data_c)
            # data_c["mkpts0_c"] += 0.5  # no idea
            # data_c["mkpts0_f"] += 0.5  # no idea
            # data_c["mkpts1_c"] += 0.5  # no idea
            # data_c["mkpts1_f"] += 0.5  # no idea

            db_kps = data_c["mkpts0_c"].cpu().numpy()
            query_kps = data_c["mkpts1_c"].cpu().numpy()
            scores = data_c["mconf"].cpu().numpy()
            matches = np.stack([np.arange(len(query_kps)), np.arange(len(query_kps))], axis=1)

            # debug
            good_matches = matches[scores > 0.5]
            # print(len(good_matches))
            # result_img = vis_image_matching(ori_query_image, ori_db_image, query_kps, db_kps, good_matches, None)
            result_img = vis_image_matching(mask_query_image, mask_db_image, query_kps, db_kps, good_matches, None)
            save_path = osp.join(save_dir, "result_{}.png".format(i))
            # cv2.imwrite(save_path, result_img)
            # if len(matches) > 10:
            # if db_image_path == "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-103651/0_input_data/color/50.png":
            #     cv2.imshow("result", result_img)
            #     cv2.waitKey(0)

            # save result
            result_dict = {}
            result_dict["data_c"] = {k: v.cpu().detach() if isinstance(v, torch.Tensor) else v for k, v in data_c.items()}  # 先放进内存，后面要用再放进显存
            result_dict["db_image_dict"] = db_image_dict
            result_dict["query_image_kps"] = query_kps
            result_dict["db_image_kps"] = db_kps
            result_dict["scores"] = scores
            result_dict["bbox_3d"] = db_bbox_3d
            result_dict["debug_img_path"] = save_path  # debug

            db_matching_result.append([[db_image_dict["image_id"], db_image_dict["object_id"]], result_dict])
            pass
        sorted_db_matching_result = sorted(db_matching_result, key=lambda x: np.max(x[1]["scores"]) if len(x[1]["scores"]) != 0 else 0, reverse=True)
        db_matching_result_dict = {}
        for identity, result_dict in db_matching_result:
            image_id, object_id = identity
            if image_id not in db_matching_result_dict:
                db_matching_result_dict[image_id] = {}
            db_matching_result_dict[image_id][object_id] = result_dict

        # for data in score_data:
        #     new_save_path = data[0].replace("result", "result_{}".format(int(data[2] * 1000)))
        #     os.rename(data[0], new_save_path)

        infer_result_dict["1-2D_2D_matching_max_score"] = np.max(sorted_db_matching_result[0][1]["scores"]) if len(sorted_db_matching_result[0][1]["scores"]) != 0 else 0
        infer_result_dict["1-2D_2D_matching_mean_score"] = np.mean(sorted_db_matching_result[0][1]["scores"]) if len(sorted_db_matching_result[0][1]["scores"]) != 0 else 0
        # # 如果最高的 score 都不到阈值，第一步就失败
        # thres_max_score = 0.8
        # if np.max(sorted_db_matching_result[0][1]["scores"]) < thres_max_score:
        #     print("Fail in step 1: image matching!")
        #     return False

        ####################################
        #### 2. Covis graph to get 3D-2D
        ####################################
        # TODO 调参
        filter_num = 5
        filtered_db_matching_result = sorted_db_matching_result[:filter_num]
        covis_num = 6
        db_covis_matching_result = []
        for identity, match_result_dict in filtered_db_matching_result: 
            image_id, object_id = identity

            final_match_point3D_dict = {}
            final_match_image_point2D_dict = {}

            db_image_dict = match_result_dict["db_image_dict"]
            model_image = self.db_coarse_images[image_id]

            # get best n covis
            point3d_ids = model_image.point3D_ids
            all_vis_img_ids = []
            for i in point3d_ids:
                if i != -1:
                    all_vis_img_ids += self.db_coarse_points3D[i].image_ids.tolist()
            covis_results = dict(Counter(all_vis_img_ids))
            covis_results = [(k, v) for k, v in covis_results.items()]
            # covis_results = sorted(covis_results, key=lambda x: x[1], reverse=True)[1:]
            covis_results = sorted(covis_results, key=lambda x: x[1], reverse=True)
            best_covis_image_ids = [covis_result[0] for covis_result in covis_results[:covis_num]]  # 一个候选帧的 covis_num 个最佳共视帧

            # get covis 3D points num
            covis_3d_points_num = 0
            best_covis_matching_results = [db_matching_result_dict[covis_image_id][object_id] for covis_image_id in best_covis_image_ids if object_id in db_matching_result_dict[covis_image_id]]   # 选用相同 object_id 的共视帧，有点绕
            best_covis_images = [self.db_coarse_images[covis_image_id] for covis_image_id in best_covis_image_ids]
            for covis_matching_result, covis_image in zip(best_covis_matching_results, best_covis_images):
                db_kps = covis_matching_result["db_image_kps"]
                query_kps = covis_matching_result["query_image_kps"]
                scores = covis_matching_result["scores"]
                # 暂时先不筛选低得分匹配

                db_image_kp_point2ds = {tuple(j.astype(np.int32)): i for i, j in enumerate(covis_image.xys)}
                db_image_kp_point3d_ids = {tuple(i.astype(np.int32)): j for i, j in zip(covis_image.xys, covis_image.point3D_ids) if j != -1}
                db_point3d_ids = [db_image_kp_point3d_ids[tuple(db_kp.astype(np.int32))] if tuple(db_kp.astype(np.int32)) in db_image_kp_point3d_ids else -1 for db_kp in db_kps]

                # 统计候选帧+它的最佳共视帧 与 query 帧的共视 3D 点对应
                for i, query_kp in enumerate(query_kps):
                    named_kp = tuple(query_kp.astype(np.int32))
                    if named_kp not in final_match_point3D_dict:
                        final_match_point3D_dict[named_kp] = []
                        final_match_image_point2D_dict[named_kp] = []
                    final_match_point3D_dict[named_kp].append(db_point3d_ids[i])
                    if tuple(db_kps[i].astype(np.int32)) in db_image_kp_point2ds:
                        final_match_image_point2D_dict[named_kp].append({"image_id": covis_image.id, "point2D": db_kps[i], "point2D_id": i, "db_image_point2D_id": db_image_kp_point2ds[tuple(db_kps[i].astype(np.int32))]})
                pass
            pass
            result_dict = {}
            result_dict["image_id"] = image_id
            result_dict["object_id"] = object_id
            result_dict["db_matching_result"] = match_result_dict
            result_dict["covis_match_image_point2D_dict"] = final_match_image_point2D_dict
            result_dict["covis_match_point3D_dict"] = final_match_point3D_dict
            db_covis_matching_result.append(result_dict)

        covis_point3d_thres = 2

        best_covis_kp_num = -1
        best_covis_image_id = None
        best_covis_object_id = None
        best_covis_db_dict = None
        best_fine_query_point2D_match_db_image_point2D_dict = None
        for result_dict in db_covis_matching_result:
            fine_query_point2D_match_db_image_point2D_dict = {}

            image_id = result_dict["image_id"]
            object_id = result_dict["object_id"]
            match_point3D_dict = result_dict["covis_match_point3D_dict"]
            match_image_point2D_dict = result_dict["covis_match_image_point2D_dict"]

            match_indexes_dict = {}
            match_p3D_dict = {}
            statics_match_point3D_dict = {k: sorted([[i, j] for i, j in dict(Counter(v)).items()], reverse=True, key=lambda x: x[1]) for k, v in match_point3D_dict.items()}
            covis_kp_num = 0
            for k in match_point3D_dict.keys():
                match_point3D = match_point3D_dict[k]
                statics_match_point3D = statics_match_point3D_dict[k]
                if len(statics_match_point3D) != 0 and statics_match_point3D[0][0] != -1 and statics_match_point3D[0][1] > covis_point3d_thres:
                    covis_kp_num += 1
                    match_p3D_dict[k] = statics_match_point3D[0][0]
                    match_indexes_dict[k] = match_point3D.index(statics_match_point3D[0][0])
                else:
                    match_indexes_dict[k] = -1

            for k in match_image_point2D_dict.keys():
                match_index = match_indexes_dict[k]
                if match_index == -1:
                    continue

                match_point3D_id = match_p3D_dict[k]
                match_dict = match_image_point2D_dict[k]
                match_item = match_dict[match_index]
                image_id = match_item["image_id"]
                db_point2D = match_item["point2D"]
                if image_id not in fine_query_point2D_match_db_image_point2D_dict:
                    fine_query_point2D_match_db_image_point2D_dict[image_id] = []
                fine_query_point2D_match_db_image_point2D_dict[image_id].append({"query_point2D": k, "db_image_point2D_id": match_item["db_image_point2D_id"], "db_image_point3D_id": match_point3D_id, "db_point2D": db_point2D})  # db_point2D for debug


            if covis_kp_num > best_covis_kp_num:
                best_covis_kp_num = covis_kp_num
                best_covis_image_id = image_id
                best_covis_object_id = object_id
                best_covis_db_dict = result_dict
                best_fine_query_point2D_match_db_image_point2D_dict = fine_query_point2D_match_db_image_point2D_dict
            pass

        infer_result_dict["2-3D_2D_covis_matching_num"] = best_covis_kp_num
        # # 如果最高的 num 都不到阈值，第二步就失败
        # thres_covis_num = 10
        # if best_covis_kp_num < thres_covis_num:
        #     print("Fail in step 2: covis!")
        #     return False

        ####################################
        #### 3. Fine-grained matching + Pose estimation
        ####################################
        fine_query_kps = []
        fine_db_point3Ds = []

        bbox_3d = None
        ori_fake_pose = None
        pybullet_pose = None
        for image_id, match_data in best_fine_query_point2D_match_db_image_point2D_dict.items():
            ori_db_match_data = db_matching_result_dict[image_id][best_covis_object_id]
            fine_db_image_data = self.db_fine_images[image_id]

            query_kps = []
            db_kps = []
            db_point3D = []
            for i, match_item in enumerate(match_data):
                query_point2D = match_item["query_point2D"]
                db_image_point2D_id = match_item["db_image_point2D_id"]
                db_image_point3D_id = match_item["db_image_point3D_id"]

                query_kps.append(query_point2D)
                db_kps.append(fine_db_image_data.xys[db_image_point2D_id])
                db_point3D.append(self.db_fine_points3D[db_image_point3D_id].xyz)

            data_c = ori_db_match_data["data_c"]
            bbox_3d = ori_db_match_data["bbox_3d"]
            ori_fake_pose = ori_db_match_data["db_image_dict"]["ori_fake_pose"]
            pybullet_pose = ori_db_match_data["db_image_dict"]["pybullet_pose"]
            bbox_center = np.mean(bbox_3d, axis=0)

            data_c["mkpts0_c"] = torch.tensor(db_kps)
            data_c["mkpts0_f"] = torch.tensor(db_kps)
            data_c["mkpts1_c"] = torch.tensor(query_kps)
            data_c["mkpts1_f"] = torch.tensor(query_kps)
            data_c["mconf"] = torch.ones(len(query_kps))

            data_c = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data_c.items()}
            self.fine_matcher(data_c)

            fine_query_kps += data_c["mkpts1_f"].cpu().numpy().tolist()
            fine_db_point3Ds += (db_point3D - bbox_center).tolist()
            pass

        db_kpts_3d = np.array(fine_db_point3Ds)
        query_kpts_2d = np.array(fine_query_kps)
        if len(query_kpts_2d) < 5:
            print("Fail in step 3: not enough fine-grained matches!")
            return {"fail": True, "fail_step": 2, "fail_reason": "not enough fine-grained matches!"}

        pose_pred, _, inliers, _ = ransac_PnP(query_K, query_kpts_2d, db_kpts_3d, scale=1000, pnp_reprojection_error=7, img_hw=query_image_hw, use_pycolmap_ransac=True)

        infer_result_dict["3-RANSAC_inlier_num"] = len(inliers)
        infer_result_dict["3-RANSAC_inlier_ratio"] = len(inliers) / len(query_kpts_2d)
        # # 如果最终 inliers 数量没达到阈值，第三步失败
        # final_pose_inliers_thres = 20
        # if len(inliers) < final_pose_inliers_thres:
        #     print("Fail in step 3: pose estimation!")
        #     return False

        if pose_pred is None:
            print("Fail in step 3: pose estimation!")
            return {"fail": True, "fail_step": 3, "fail_reason": "pose estimation!"}

        object_rel_T = np.concatenate([pose_pred, np.array([[0, 0, 0, 1]])], axis=0)
        Tog = np.linalg.inv(ori_fake_pose) @ pybullet_pose # Tog = Tow * Twg
        Tcg = object_rel_T @ Tog          # Tcg = Tco * Tog

        # # vis
        # draw_image = cv2.imread(query_image_path)
        # draw_image = vis_pose(draw_image, [np.concatenate([pose_pred, np.array([0, 0, 0, 1])[None, ...]], axis=0)], query_K)
        # cv2.imwrite(osp.join(save_dir, "tmp.png"), draw_image)

        # infer_result_dict
        infer_result_dict["fail"] = False
        infer_result_dict["pose"] = Tcg

        return infer_result_dict

    
def main():
    # data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942"
    # data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-104479"
    data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-103651"
    interface = Interface(data_dir)
    
    # target_data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-102453"
    # target_data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-104479"
    # target_data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942"
    target_data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-103651"
    target_query_images = Interface.get_query_objects(scene_path=target_data_dir)

    results = []
    # for object_id in [4, 8, 11]:
    # for object_id in [6, 9, 10]:
    # for object_id in [7]:
    for object_id in interface.object_ids:
        object_target_query_images = target_query_images[object_id]
        for target_query_image_dict in tqdm(object_target_query_images, desc="obejct_id: {}".format(object_id)):
            result = interface.infer(target_query_image_dict)

            if result["fail"]:
                results.append([-1, -1]) 
            else:
                angle, shift = compute_RT_errors(result["pose"], target_query_image_dict["pose"])
                print("evaluate: angle: {}, shift: {}".format(angle, shift))
                results.append([angle, shift])

    np.savetxt("results.txt", np.array(results))
    pass


if __name__ == "__main__":
    main()
    pass