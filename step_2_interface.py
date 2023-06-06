import os
import os.path as osp
import sys
import shutil
from omegaconf import DictConfig

import cv2
import torch
import numpy as np

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus", 'submodules/LoFTR/src'))

sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus"))
# from src.utils import data_utils
# from src.utils import vis_utils
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


def interface(cfg, map_data_dir, inference_data_dir, output_dir):
    if osp.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(name=output_dir, exist_ok=True)

    sub_maps = osp.join(map_data_dir, "sub_map")
    object_class_ids = [int(x) for x in os.listdir(sub_maps) if osp.isdir(osp.join(sub_maps, x))]

    # 读入每个 object 的 infos
    object_3d_infos = {}
    for object_class_id in object_class_ids:
        object_dir = osp.join(sub_maps, str(object_class_id))
        avg_anno3d_file = osp.join(object_dir, "anno", "anno_3d_average.npz")
        (
            keypoints3d,
            avg_descriptors3d,
            avg_coarse_descriptors3d,
            avg_scores,
            num_3d_orig,
        ) = read_anno3d(
            cfg.datamodule.shape3d_val, 
            avg_anno3d_file, 
            pad=cfg.datamodule.pad3D, 
            load_3d_coarse=cfg.datamodule.load_3d_coarse
        )
        object_3d_infos[object_class_id] = {"keypoints3d": keypoints3d, "avg_descriptors3d": avg_descriptors3d, "avg_coarse_descriptors3d": avg_coarse_descriptors3d, "avg_scores": avg_scores, "num_3d_orig": num_3d_orig}

    # 构建 matchers
    match_2D_3D_model = build_model(cfg['model']["OnePosePlus"], cfg['model']['pretrained_ckpt'])
    match_2D_3D_model.cuda()

    # 读入要预测的 Object ROIs
    query_images = []
    image_dir = osp.join(inference_data_dir, "color")
    roi_bbox_dir = osp.join(inference_data_dir, "2d_bbox")
    K = np.load(osp.join(inference_data_dir, "intrinsics.npy"))
    image_paths = sorted(get_image_files(image_dir), key=lambda x: int(osp.basename(x).split('.')[0]))
    roi_bbox_paths = [osp.join(roi_bbox_dir, osp.basename(x).rsplit(".", maxsplit=1)[0] + ".npy") for x in image_paths]
    for image_path, roi_bbox_path in zip(image_paths, roi_bbox_paths):
        image = cv2.imread(image_path)
        roi_bboxs = np.load(roi_bbox_path, allow_pickle=True).item()
        for class_id, roi_bbox in roi_bboxs.items():
            roi_bbox = pad_bbox_to_square(roi_bbox)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_crop, K_crop = crop_img_by_bbox(image, roi_bbox, K, crop_size=512)
            # image_crop, K_crop = crop_img_by_bbox_no_resize(image, roi_bbox, K)
            cv2.imwrite(osp.join(output_dir, osp.basename(image_path).rsplit(".", maxsplit=1)[0] + "_" + str(class_id) + ".png"), image_crop)
            image_crop = image_crop.astype(np.float32) / 255
            image_crop_tensor = torch.from_numpy(image_crop)[None][None]

            query_images.append({"image_path": image_path, "crop_img": image_crop, "K": K, "crop_inp": image_crop_tensor, "K_crop": K_crop, "class_id": class_id})
    
    for i, data_dict in enumerate(query_images):
        query_image_path = data_dict["image_path"]
        K = data_dict["K"]
        image_crop_tensor = data_dict["crop_inp"]
        K_crop = data_dict["K_crop"]
        class_id = data_dict["class_id"]

        info_dict = object_3d_infos[class_id]
        data = {
            "keypoints3d": info_dict["keypoints3d"][None].cuda(),
            "descriptors3d_db": info_dict["avg_descriptors3d"][None].cuda(),
            "descriptors3d_coarse_db": info_dict["avg_coarse_descriptors3d"][None].cuda(),
            "query_image": image_crop_tensor.cuda(),
            "query_image_path": query_image_path,
            # "query_intrinsic": K_crop[None],
            # "query_intrinsic_origin": K[None],
        }
        
        with torch.no_grad():
            match_2D_3D_model(data)
        mkpts_3d = data["mkpts_3d_db"].cpu().numpy() # N*3
        mkpts_query = data["mkpts_query_f"].cpu().numpy() # N*2
        scores = data["mconf"].cpu().numpy() # N
        # img_hw = data_dict["crop_inp"].shape[2:][::-1]
        img_hw = [512, 512]
        pose_pred, _, inliers, _ = ransac_PnP(K_crop, mkpts_query, mkpts_3d, scale=1000, pnp_reprojection_error=7, img_hw=img_hw, use_pycolmap_ransac=True)

        # draw_image = vis_pose(draw_image, [np.concatenate([pose_pred, np.array([0, 0, 0, 1])[None, ...]], axis=0)], K_crop)
        # cv2.imwrite(osp.join(output_dir, osp.basename(query_image_path)), draw_image)

        draw_image = cv2.imread(query_image_path)
        draw_image = vis_pose(draw_image, [np.concatenate([pose_pred, np.array([0, 0, 0, 1])[None, ...]], axis=0)], K)
        cv2.imwrite(osp.join(output_dir, "final_" + osp.basename(query_image_path).rsplit(".", maxsplit=1)[0] + "_" + str(class_id) + ".png"), draw_image)

        # bbox_path = os.path.join("/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942/0_input_data/bbox", "{}_bbox_corners.npy".format(class_id))
        # bbox3d = np.load(bbox_path)
        # vis_utils.save_demo_image(
        #     pose_pred,
        #     K,
        #     image_path=query_image_path,
        #     box3d=bbox3d,
        #     draw_box=len(inliers) > 10,
        #     save_path=osp.join(output_dir, "final_" + osp.basename(query_image_path).rsplit(".", maxsplit=1)[0] + "_" + str(class_id) + ".png"),
        # )
    pass


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


class Interface(object):
    def __init__(self, map_dir):
        cfg = build_inference_cfg()
        self.cfg = cfg
        # 构建 matchers
        self.match_2D_3D_model = build_model(cfg['model']["OnePosePlus"], cfg['model']['pretrained_ckpt'])
        self.match_2D_3D_model.cuda()
        # avg
        self.object_3d_infos = {}
        for object_class_id in os.listdir(osp.join(map_dir, "sub_map")):
            if not object_class_id.isdigit():
                continue
            object_class_id = int(object_class_id)
            object_dir = osp.join(map_dir, "sub_map", str(object_class_id))
            avg_anno3d_file = osp.join(object_dir, "anno", "anno_3d_average.npz")
            (
                keypoints3d,
                avg_descriptors3d,
                avg_coarse_descriptors3d,
                avg_scores,
                num_3d_orig,
            ) = read_anno3d(
                cfg.datamodule.shape3d_val, 
                avg_anno3d_file, 
                pad=cfg.datamodule.pad3D, 
                load_3d_coarse=cfg.datamodule.load_3d_coarse
            )
            self.object_3d_infos[object_class_id] = {"keypoints3d": keypoints3d, "avg_descriptors3d": avg_descriptors3d, "avg_coarse_descriptors3d": avg_coarse_descriptors3d, "avg_scores": avg_scores, "num_3d_orig": num_3d_orig}
        pass

    def infer(self, object_id, image, bbox, K, camera_pose, tmp_save_path=None):
        # 3d info
        info_dict = self.object_3d_infos[object_id]
        data = {
            "keypoints3d": info_dict["keypoints3d"][None].cuda(),
            "descriptors3d_db": info_dict["avg_descriptors3d"][None].cuda(),
            "descriptors3d_coarse_db": info_dict["avg_coarse_descriptors3d"][None].cuda(),
            # "query_image": image_crop_tensor.cuda(),
            # "query_image_path": query_image_path,
            # "query_intrinsic": K_crop[None],
            # "query_intrinsic_origin": K[None],
        }

        # predict
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_hw = image.shape[:2]
        K_crop = K
        mask_image = np.zeros_like(image)
        mask_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        image_crop = mask_image.astype(np.float32) / 255
        image_crop_tensor = torch.from_numpy(image_crop)[None][None]

        data.update({"query_image": image_crop_tensor.cuda()})
        
        with torch.no_grad():
            self.match_2D_3D_model(data)
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
        print(np.max(scores), np.min(scores), np.mean(scores), np.median(scores))

        return pose_pred


if __name__ == "__main__":
    # map_data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942/1_map"
    # interface_data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942/0_input_data"
    # output_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942/2_interface"
    # cfg = build_inference_cfg()

    # interface(cfg, map_data_dir, interface_data_dir, output_dir)

    map_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942/1_map"
    query_img_path = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-111979/0_input_data/color/0.png"
    query_bbox_path = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-111979/0_input_data/2d_bbox/0.npy"
    camera_pose_path = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-111979/0_input_data/poses/0.npy"
    K_path = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-111979/0_input_data/intrinsics.npy"

    query_img = cv2.imread(query_img_path)
    query_bbox = np.load(query_bbox_path, allow_pickle=True).item()[8]
    query_K = np.load(K_path)
    query_camera_pose = np.load(camera_pose_path)

    interface = Interface(map_dir)
    interface.infer(8, query_img, query_bbox, query_K, query_camera_pose)
    pass