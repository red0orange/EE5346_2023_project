import os
import sys
import shutil
import math
import os.path as osp
import hydra
from tqdm import tqdm
from copy import deepcopy
import natsort
from loguru import logger
from pathlib import Path
from omegaconf import DictConfig

os.environ["TORCH_USE_RTLD_GLOBAL"] = "TRUE"  # important for DeepLM module

import cv2
import numpy as np

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus", 'submodules/LoFTR/src'))

sys.path.append(os.path.join(project_root_dir, "OnePose_Plus_Plus"))
from src.utils.ray_utils import ProgressBar, chunks
sys.path.append(os.path.join(project_root_dir, "utils"))
from file_utils.file import *
from colmap_utils.read_write_model import read_cameras_text, read_images_text, read_cameras_binary, read_images_binary
from colmap_utils.update_database_camera import camera_to_database
from tf_utils.T_7dof import *


def sfm_core(cfg, img_lists, pose_lists, intrin_lists, outputs_dir_root):
    from src.sfm_utils import (
        generate_empty,
        triangulation,
        pairs_exhaustive_all, pairs_from_index, pairs_from_poses
    )
    from src.KeypointFreeSfM import coarse_match, post_optimization

    outputs_dir = outputs_dir_root
    vis3d_pth = osp.join(outputs_dir_root, "vis3d")
    Path(outputs_dir).mkdir(exist_ok=True, parents=True)

    feature_out = osp.join(outputs_dir, f"feats-{cfg.network.detection}.h5")
    covis_num = cfg.sfm.covis_num
    covis_pairs_out = osp.join(outputs_dir, f"pairs-covis{covis_num}.txt")
    matches_out = osp.join(outputs_dir, f"matches-{cfg.network.matching}.h5")
    empty_dir = osp.join(outputs_dir, "sfm_empty")
    deep_sfm_dir = osp.join(outputs_dir, "sfm_ws")

    ###############################
    # 1. Coarse reconstruction  ->  model_coarse
    ###############################
    logger.info("Keypoint-Free SfM coarse reconstruction begin...")

    # 寻找每个图像的共视图像，生成pairs-covis.txt。
    # 如果covis_num=-1，则生成所有的共视图像对，否则生成covis_num个共视图像对。
    if covis_num == -1:
        # 排列组合生成全部
        pairs_exhaustive_all.exhaustive_all_pairs(
            img_lists, covis_pairs_out
        )
    else:
        pairs_from_poses.my_covis_from_pose(
            img_lists,
            pose_lists,
            covis_pairs_out,
            covis_num,
            min_rotation=cfg.sfm.min_rotation
        )

    # 2. LoFTR coarse matching
    coarse_match.detector_free_coarse_matching(
        # input
        img_lists,
        covis_pairs_out,
        # output
        feature_out,
        matches_out,
        # param
        # use_ray=cfg.use_local_ray,
        use_ray=False,
        verbose=cfg.verbose
    )
    # 生成 images.bin，cameras.bin，points3D.bin（空的，下面的 triangulator 才生成）
    generate_empty.my_generate_model(img_lists, pose_lists, intrin_lists, empty_dir)

    triangulation.main(
        deep_sfm_dir,
        empty_dir,
        outputs_dir,
        covis_pairs_out,
        feature_out,
        matches_out,
        match_model=cfg.network.matching,  # loftr
        image_dir=None,
        verbose=cfg.verbose,
    )

    # feat-loftr_coarse.h5
    os.system(
        f"mv {feature_out} {osp.splitext(feature_out)[0] + '_coarse' + osp.splitext(feature_out)[1]}"
    )
    if cfg.enable_post_refine:
        assert osp.exists(osp.join(deep_sfm_dir, "model"))
        # model_coarse
        os.system(
            f"mv {osp.join(deep_sfm_dir, 'model')} {osp.join(deep_sfm_dir, 'model_coarse')}"
        )

    # 3. 优化 model_coarse
    if cfg.enable_post_refine:
        if (
            not osp.exists(osp.join(deep_sfm_dir, "model"))
            or cfg.overwrite_fine
        ):
            assert osp.exists(
                osp.join(deep_sfm_dir, "model_coarse")
            ), f"model_coarse not exist under: {deep_sfm_dir}, please set 'cfg.overwrite_coarse = True'"
            os.system(f"rm -rf {osp.join(deep_sfm_dir, 'model')}")

            # configs for post optimization:
            post_optim_configs = cfg.post_optim if 'post_optim' in cfg else None

            logger.info("Keypoint-Free SfM post refinement begin...")
            # 3a. 调用函数
            state = post_optimization.post_optimization(
                img_lists,
                covis_pairs_out,
                colmap_coarse_dir=osp.join(deep_sfm_dir, "model_coarse"),
                refined_model_save_dir=osp.join(deep_sfm_dir, "model"),
                match_out_pth=matches_out,
                feature_out_pth=feature_out,
                # use_global_ray=cfg.use_global_ray,
                # fine_match_use_ray=cfg.use_local_ray,
                use_global_ray=False,
                fine_match_use_ray=False,
                vis3d_pth=vis3d_pth,
                verbose=cfg.verbose,
                args=post_optim_configs
            )
            if state == False:
                logger.error("Coarse reconstruction failed!")
    else:
        raise NotImplementedError


def postprocess(cfg, img_lists, bbox_dict, full_map_dir, sub_maps_dir):
    """ Filter points and average feature"""
    from src.sfm_utils.postprocess import filter_points, feature_process, filter_tkl

    if osp.exists(sub_maps_dir): shutil.rmtree(sub_maps_dir)

    full_model_path = osp.join(full_map_dir, "sfm_ws", "model")
    for obj_name in bbox_dict.keys():
        bbox = bbox_dict[obj_name]
        
        filted_ws_path = osp.join(sub_maps_dir, str(obj_name))
        # copy descriptors
        outputs_dir = filted_ws_path
        vis3d_pth = osp.join(filted_ws_path, "vis3d")
        feature_out = osp.join(filted_ws_path, f"feats-{cfg.network.detection}.h5")
        feature_coarse_out = osp.join(filted_ws_path, f"feats-{cfg.network.detection}_coarse.h5")
        shutil.copytree(osp.join(full_map_dir, "vis3d"), vis3d_pth)
        shutil.copy(osp.join(full_map_dir, f"feats-{cfg.network.detection}.h5"), feature_out)
        shutil.copy(osp.join(full_map_dir, f"feats-{cfg.network.detection}_coarse.h5"), feature_coarse_out)

        ann_model_path = osp.join(filted_ws_path, "anno")
        os.makedirs(name=ann_model_path, exist_ok=True)
        filted_model_path = osp.join(filted_ws_path, "model")
        os.makedirs(name=filted_model_path, exist_ok=True)

        filter_points.my_filter_bbox(
            full_model_path,
            filted_model_path,
            bbox,
        )

        track_length, points_count_list = filter_tkl.get_tkl(
            filted_model_path, thres=cfg.dataset.max_num_kp3d, show=False
        )
        xyzs, points_ids = filter_points.filter_track_length(
            filted_model_path, track_length
        )  # crop 3d points by 3d box and track length
        merge_xyzs, merge_idxs = filter_points.merge(xyzs, points_ids)  # merge 3d points by distance between points

        # 3. 生成 2D keypoints 和 3D mappoints 的描述子
        # Save loftr coarse keypoints:
        cfg_coarse = deepcopy(cfg)
        cfg_coarse.network.detection = "loftr_coarse"
        feature_coarse_path = (
            osp.splitext(feature_out)[0] + "_coarse" + osp.splitext(feature_out)[1]
        )
        feature_process.my_get_kpt_ann(
            cfg_coarse,
            img_lists,
            feature_coarse_path,
            filted_model_path,
            ann_model_path,
            merge_idxs,
            merge_xyzs,
            save_feature_for_each_image=False,
            feat_3d_name_suffix="_coarse",
            # use_ray=cfg.use_local_ray,
            use_ray=False,
            verbose=cfg.verbose,
        )

        # Save fine level points and features:
        feature_process.my_get_kpt_ann(
            cfg,
            img_lists,
            feature_out,
            filted_model_path,
            ann_model_path,
            merge_idxs,
            merge_xyzs,
            save_feature_for_each_image=False,
            # use_ray=cfg.use_local_ray,
            use_ray=False,
            verbose=cfg.verbose,
        )
    pass


def mapping(data_dir):
    """

    Args:
        data_dir (_type_): 工作空间目录
        cfg (_type_): 参数
    """
    cfg = build_sfm_cfg()

    input_data_dir = osp.join(data_dir, "0_input_data")
    output_dir = osp.join(data_dir, "1_map")
    if osp.exists(output_dir): shutil.rmtree(output_dir)

    # mapping
    color_dir = osp.join(input_data_dir, "color")
    img_lists = get_image_files(path=color_dir)
    img_lists = sorted(img_lists, key=lambda x: int(osp.basename(x).split(".")[0]))
    if len(img_lists) == 0:
        logger.info(f"No png image in {root_dir}")
        exit(1000)
    pose_lists = [None] * len(img_lists)
    for i in range(len(img_lists)):
        name = osp.basename(img_lists[i]).split(".")[0]
        pose = np.load(osp.join(input_data_dir, "poses", f"{name}.npy"))
        inv_pose = np.linalg.inv(pose)
        np.save(osp.join(input_data_dir, "poses", f"{name}_inv.npy"), inv_pose)
        pose_lists[i] = osp.join(input_data_dir, "poses", f"{name}_inv.npy")
    intrin_lists = [osp.join(input_data_dir, "intrinsics.npy")] * len(img_lists)

    bbox_dir = osp.join(input_data_dir, "bbox")
    bbox_dict = {}
    for bbox_path in get_files(bbox_dir):
        name = osp.basename(bbox_path).split("_")[0]
        bbox_dict[int(name)] = np.load(bbox_path)

    full_map_output_dir = osp.join(output_dir, "full_map")
    sfm_core(cfg, img_lists, pose_lists, intrin_lists, full_map_output_dir)
    sub_maps_dir = osp.join(output_dir, "sub_map")
    postprocess(cfg, img_lists, bbox_dict, full_map_output_dir, sub_maps_dir)

    logger.info(f"Finish Processing {data_dir}.")
    pass


def build_sfm_cfg():
    cfg = DictConfig({
        "type": "sfm", 
        "work_dir": osp.join(project_root_dir, "OnePose_Plus_Plus"),
        "match_type": "softmax",
        "enable_post_refine": True,
        "verbose": True,
        "disable_lightning_logs": True
        })

    cfg.dataset = DictConfig({
        "max_num_kp3d": 15000,
        # "data_dir": ["null"],
        # "outputs_dir": "null"
    })

    cfg.network = DictConfig({
        "detection": "loftr",
        "matching": "loftr",
    })

    cfg.sfm = DictConfig({
        "gen_cov_from": "pose",
        "covis_num": 10,
        "min_rotation": 10
    })

    cfg.post_optim = DictConfig({
        "coarse_recon_data": DictConfig({"feature_track_assignment_strategy": "greedy"}),
        "optimizer": DictConfig({"solver_type": "SecondOrder", "residual_mode": "geometry_error", "optimize_lr": DictConfig({"depth": 0.03})}),
    })

    cfg.post_process = DictConfig({
        "filter_bbox_before_filter_track_length": True,
        "skip_bbox_filter": False,
    })
    return cfg


if __name__ == "__main__":
    # output:
    # 1. 3D Map with descriptors
    # 2. 2D Images with 2D descriptors
    onepose_dir = osp.join(project_root_dir, "OnePose_Plus_Plus")

    # # data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100942"
    # # data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-102453"
    # data_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs/seed-100477"
    # mapping(data_dir, cfg)

    data_root = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs"
    for data_path in os.listdir(data_root):
        print("=====================================")
        print("===== {} =====".format(data_path))
        print("=====================================")
        # if data_path == "seed-100942" or data_path == "seed-102453": continue
        if os.path.exists(osp.join(data_root, data_path, "1_map")): continue

        data_path = osp.join(data_root, data_path)
        mapping(data_path)
    pass