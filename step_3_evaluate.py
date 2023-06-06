import os
import os.path as osp
import random
from tqdm import tqdm
import sys
import shutil
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

import cv2
import torch
import numpy as np
np.warnings.filterwarnings('ignore')

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus", 'submodules/LoFTR/src'))

sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus"))
# from src.utils import data_utils
from src.utils import vis_utils
from src.utils.metric_utils import ransac_PnP
from src.inference.inference_OnePosePlus import build_model
from src.local_feature_object_detector.local_feature_2D_detector import LocalFeatureObjectDetector

sys.path.append(os.path.join(project_root_dir, "utils"))
from file_utils.file import *
from common_utils.crop_image import *
from vis_utils.vis_pose import *


def read_anno3d(
    shape3d, avg_anno3d_file, pad=True, load_3d_coarse=True
):
    """ Read(and pad) 3d info"""
    avg_data = np.load(avg_anno3d_file)

    keypoints3d = torch.Tensor(avg_data["keypoints3d"])  # [m, 3]
    avg_descriptors3d = torch.Tensor(avg_data["descriptors3d"])  # [dim, m]
    avg_scores = torch.Tensor(avg_data["scores3d"])  # [m, 1]

    num_3d_orig = keypoints3d.shape[0]

    if load_3d_coarse:
        avg_anno3d_coarse_file = (
            osp.splitext(avg_anno3d_file)[0]
            + "_coarse"
            + osp.splitext(avg_anno3d_file)[1]
        )
        avg_coarse_data = np.load(avg_anno3d_coarse_file)
        avg_coarse_descriptors3d = torch.Tensor(
            avg_coarse_data["descriptors3d"]
        )  # [dim, m]
        avg_coarse_scores = torch.Tensor(avg_coarse_data["scores3d"])  # [m, 1]

    else:
        avg_coarse_descriptors3d = None

    if pad:
        (keypoints3d, padding_index,) = data_utils.pad_keypoints3d_random(
            keypoints3d, shape3d
        )
        (avg_descriptors3d, avg_scores,) = data_utils.pad_features3d_random(
            avg_descriptors3d, avg_scores, shape3d, padding_index
        )

        if avg_coarse_descriptors3d is not None:
            (
                avg_coarse_descriptors3d,
                avg_coarse_scores,
            ) = data_utils.pad_features3d_random(
                avg_coarse_descriptors3d,
                avg_coarse_scores,
                shape3d,
                padding_index,
            )

    return (
        keypoints3d,
        avg_descriptors3d,
        avg_coarse_descriptors3d,
        avg_scores,
        num_3d_orig,
    )


def build_inference_cfg():
    cfg = DictConfig({
        "type": "inference",
        # "data_base_dir": "null",
        # "sfm_base_dir": "null"
    })

    cfg.model = DictConfig({
        "pretrained_ckpt": osp.join(os.path.join(project_root_dir, "OnePose_Plus_Plus"), "weight/OnePosePlus_model.ckpt"),
        "OnePosePlus": DictConfig({
            "loftr_backbone": DictConfig({
                "type": "ResNetFPN",
                "resolution": [8, 2],
                "resnetfpn": DictConfig({
                    "block_type": "BasicBlock",
                    "initial_dim": 128,
                    "block_dims": [128, 196, 256],
                    "output_layers": [3, 1]
                }),
                "pretrained": osp.join(os.path.join(project_root_dir, "OnePose_Plus_Plus"), "weight/LoFTR_wsize9.ckpt"),
                "pretrained_fix": False
            }),
            "interpol_type": "bilinear",
            "keypoints_encoding": DictConfig({
                "enable": True,
                "type": "mlp_linear",
                "descriptor_dim": 256,
                "keypoints_encoder": [32, 64, 128],
                "norm_method": "instancenorm"
            }),
            "positional_encoding": DictConfig({
                "enable": True,
                "pos_emb_shape": [256, 256]
            }),
            "loftr_coarse": DictConfig({
                "type": "LoFTR",
                "d_model": 256,
                "d_ffm": 128,
                "nhead": 8,
                "layer_names": ["self", "cross"],
                "layer_iter_n": 3,
                "dropout": 0.0,
                "attention": "linear",
                "norm_method": "layernorm",
                "kernel_fn": "elu + 1",
                "d_kernel": 16,
                "redraw_interval": 2,
                "rezero": None,
                "final_proj": False
            }),
            "coarse_matching": DictConfig({
                "type": "dual-softmax",
                "thr": 0.1,
                "feat_norm_method": "sqrt_feat_dim",
                "border_rm": 2,
                "dual_softmax": DictConfig({
                    "temperature": 0.08
                }),
                "train": DictConfig({
                    "train_padding": True,
                    "train_coarse_percent": 0.3,
                    "train_pad_num_gt_min": 200
                })
            }),
            "loftr_fine": DictConfig({
                "enable": True,
                "window_size": 5,
                "coarse_layer_norm": False,
                "type": "LoFTR",
                "d_model": 128,
                "nhead": 8,
                "layer_names": ["self", "cross"],
                "layer_iter_n": 1,
                "dropout": 0.0,
                "attention": "linear",
                "norm_method": "layernorm",
                "kernel_fn": "elu + 1",
                "d_kernel": 16,
                "redraw_interval": 2,
                "rezero": None,
                "final_proj": False
            }),
            "fine_matching": DictConfig({
                "enable": True,
                "type": 's2d',
                "s2d": DictConfig({
                    "type": 'heatmap'
                })
            })
        })
    })

    cfg.datamodule = DictConfig({
        "shape3d_val": 7000,
        "load_3d_coarse": True,
        "pad3D": False,
        "img_pad": False,
        "img_resize": [512, 512],
        "df": 8,
        "coarse_scale": 0.125
    })

    cfg.network = DictConfig({
        "detection": "loftr",
        "matching": "loftr"
    })

    return cfg

inference_cfg = build_inference_cfg()


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


class Evaluator:
    def __init__(self, scenes_dir, scene_paths=None):
        self.scenes_dir = scenes_dir
        if scene_paths is None:
            self.scene_paths = [os.path.join(scenes_dir, i) for i in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, i))]
            # self.scene_paths = self.scene_paths[:20]
        else:
            self.scene_paths = scene_paths
        pass

    def evaluate(self, each_object_query_num=10, seed=12345, save_path=None, vis_save_dir=None):
        # 构建整个 Query Object Images 的字典，物体类别为 key，每个类别下包含一个列表，装有所有该类别的 Query Object Images
        # 一张完整图路径 + Camera Pose + 一个目标物体的 ID + 一个目标物体的 2D BBox + 一个目标物体相对于相机的 Pose
        query_images = {}
        for scene_path in self.scene_paths:
        # for scene_path in ["/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-102453"]:
        # for scene_path in ["/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942"]:
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

        # 开始以场景为单位进行评估
        random.seed(seed)

        # 记录一个以场景为单位的平均指标
        # 记录一个以类别为单位的平均指标
        data_id = 0
        scene_result_dict = {}
        class_result_dict = {}
        for scene_path in self.scene_paths:
        # for scene_path in ["/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942"]:
            scene_name = os.path.basename(scene_path)
            object_class_ids = [int(i.split("_")[0]) for i in os.listdir(os.path.join(scene_path, "0_input_data", "bbox"))]
            # object_class_ids = [7]
            # object_class_ids = [3]
            # object_class_ids = [3, 7]
            objects_pybullet_pose = {}
            objects_ori_fake_pose = {}
            for object_id in object_class_ids:
                path = os.path.join(scene_path, "0_input_data", "pybullet_poses", str(object_id)+".npy")
                objects_pybullet_pose[object_id] = np.load(path, allow_pickle=True)
                path = os.path.join(scene_path, "0_input_data", "ori_fake_poses", "object_" + str(object_id)+".npy")
                objects_ori_fake_pose[object_id] = np.load(path)

            scene_result_dict[scene_name] = []
            for object_id in object_class_ids:
                if object_id not in class_result_dict.keys():
                    class_result_dict[object_id] = []

            # 构建该场景的 obejct_pose_estimators
            match_2D_3D_model = build_model(inference_cfg['model']["OnePosePlus"], inference_cfg['model']['pretrained_ckpt'])
            match_2D_3D_model.cuda()
            object_3d_infos = {}
            for object_class_id in object_class_ids:
                object_dir = osp.join(scene_path, "1_map", "sub_map", str(object_class_id))
                avg_anno3d_file = osp.join(object_dir, "anno", "anno_3d_average.npz")
                (
                    keypoints3d,
                    avg_descriptors3d,
                    avg_coarse_descriptors3d,
                    avg_scores,
                    num_3d_orig,
                ) = read_anno3d(
                    inference_cfg.datamodule.shape3d_val, 
                    avg_anno3d_file, 
                    pad=inference_cfg.datamodule.pad3D, 
                    load_3d_coarse=inference_cfg.datamodule.load_3d_coarse
                )
                object_3d_infos[object_class_id] = {"keypoints3d": keypoints3d, "avg_descriptors3d": avg_descriptors3d, "avg_coarse_descriptors3d": avg_coarse_descriptors3d, "avg_scores": avg_scores, "num_3d_orig": num_3d_orig}

            # 随机给每个物体选取 each_object_query_num 张 Query Image
            for object_id in object_class_ids:
                object_query_images = random.sample(query_images[object_id], each_object_query_num)
                # object_query_images = query_images[object_id]
                info_dict = object_3d_infos[object_id]
                data = {
                    "keypoints3d": info_dict["keypoints3d"][None].cuda(),
                    "descriptors3d_db": info_dict["avg_descriptors3d"][None].cuda(),
                    "descriptors3d_coarse_db": info_dict["avg_coarse_descriptors3d"][None].cuda(),
                    # "query_image": image_crop_tensor.cuda(),
                    # "query_image_path": query_image_path,
                    # "query_intrinsic": K_crop[None],
                    # "query_intrinsic_origin": K[None],
                }

                for object_query_image in object_query_images:
                    image_path = object_query_image["image_path"]
                    image = cv2.imread(image_path)
                    camera_pose = object_query_image["camera_pose"]  # Tcw
                    bbox = object_query_image["bbox"]
                    K = object_query_image["K"]
                    
                    # predict
                    # img_hw = [512, 512]
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # image_crop, K_crop = crop_img_by_bbox(image, bbox, K, crop_size=512)
                    # image_crop = image_crop.astype(np.float32) / 255
                    # image_crop_tensor = torch.from_numpy(image_crop)[None][None]

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    img_hw = image.shape[:2]
                    K_crop = K
                    mask_image = np.zeros_like(image)
                    mask_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    image_crop = mask_image.astype(np.float32) / 255
                    image_crop_tensor = torch.from_numpy(image_crop)[None][None]

                    data.update({"query_image": image_crop_tensor.cuda(), "query_image_path": image_path})
                    
                    with torch.no_grad():
                        match_2D_3D_model(data)
                    mkpts_3d = data["mkpts_3d_db"].cpu().numpy() # N*3
                    mkpts_query = data["mkpts_query_f"].cpu().numpy() # N*2
                    if len(mkpts_3d) > 5:
                        pose_pred, _, inliers, _ = ransac_PnP(K_crop, mkpts_query, mkpts_3d, scale=1000, pnp_reprojection_error=7, img_hw=img_hw, use_pycolmap_ransac=True)
                        object_rel_T = np.concatenate([pose_pred, np.array([[0, 0, 0, 1]])], axis=0)
                    else:
                        pose_pred = None
                        inliers = None
                        object_rel_T = None

                    scores = data["mconf"].cpu().numpy() # N
                    ##############################

                    if object_rel_T is not None:
                        Tog = np.linalg.inv(objects_ori_fake_pose[object_id]) @ objects_pybullet_pose[object_id] # Tog = Tow * Twg
                        Tcg = object_rel_T @ Tog          # Tcg = Tco * Tog

                        result_dict = {}
                        result_dict["scene_name"] = scene_name
                        result_dict["object_id"] = object_id
                        result_dict["scores"] = scores
                        result_dict["pose"] = Tcg
                        result_dict["gt_pose"] = object_query_image["pose"]
                        result_dict["id"] = data_id

                        # evaluate
                        angle, shift = compute_RT_errors(Tcg, object_query_image["pose"])
                        result_dict["eval_angle"] = angle
                        result_dict["eval_shift"] = shift

                        # vis
                        if vis_save_dir is not None:
                            vis_img = image.copy()
                            poses = [Tcg, object_query_image["pose"]]
                            vis_img = vis_pose(vis_img, [i for i in poses if i is not None], K)
                            cv2.imwrite(os.path.join(vis_save_dir, "{}.png".format(data_id)), vis_img)
                    else:  # 无法解算出来
                        result_dict = {}
                        result_dict["scene_name"] = scene_name
                        result_dict["object_id"] = object_id
                        result_dict["scores"] = scores
                        result_dict["pose"] = None
                        result_dict["gt_pose"] = object_query_image["pose"]
                        result_dict["id"] = data_id
                        result_dict["eval_angle"] = -1
                        result_dict["eval_shift"] = -1

                    scene_result_dict[scene_name].append(result_dict)
                    class_result_dict[object_id].append(result_dict)

                    data_id += 1
        
        # 保存结果
        if save_path is not None:
            np.save(save_path, {"scene_result_dict": scene_result_dict, "class_result_dict": class_result_dict})
        pass


def main():
    scenes_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs"
    save_path = "/home/huangdehao/github_projects/multi_view_rearr/tmp/new_tmp.npy"
    vis_save_dir = "/home/huangdehao/github_projects/multi_view_rearr/tmp"

    # tmp_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/runs"
    # scene_paths = [os.path.join(tmp_dir, i) for i in os.listdir(tmp_dir) if i.startswith("seed-")]
    # scene_paths = [
    #     # "",
    #     "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/runs/seed-272143",
    #     "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/runs/seed-878123",
    # ]
    # scene_paths = [
    #     "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-104479"
    # ]
    scene_paths = None

    evaluator = Evaluator(scenes_dir, scene_paths=scene_paths)
    evaluator.evaluate(each_object_query_num=10, seed=12345, save_path=save_path, vis_save_dir=vis_save_dir)
    pass


def evaluate_file():
    file_path = "/home/huangdehao/github_projects/multi_view_rearr/tmp/new_tmp.npy"
    result = np.load(file_path, allow_pickle=True).item()
    scene_result_dict = result["scene_result_dict"]
    class_result_dict = result["class_result_dict"]

    all_cnt = 0
    tp_fn_cnt = 0
    correct_cnt = 0
    pred = []
    for scene_name, scene_results in scene_result_dict.items():
        # print(scene_name)
        for scene_result in scene_results:
            object_id = scene_result["object_id"]
            scores = scene_result["scores"]
            eval_angle = scene_result["eval_angle"]
            eval_shift = scene_result["eval_shift"]

            pred_value = 0

            all_cnt += 1

            if eval_angle == -1:
                pred.append(pred_value)
                continue

            if np.max(scores) > 0:
                tp_fn_cnt += 1
                # if eval_angle < 10 and eval_shift < 10:
                if eval_angle < 0.5 and eval_shift < 0.5:
                    pred_value = np.max(scores)
                    correct_cnt += 1
            pred.append(pred_value)
    print("Precision: ", correct_cnt / tp_fn_cnt)
    print("Recall: ", correct_cnt / all_cnt)

    # y = np.ones(all_cnt)
    # pred = np.array(pred)
    # print("AUC: ", roc_auc_score(y, pred))
    pass


if __name__ == "__main__":
    # main()
    evaluate_file()
    pass