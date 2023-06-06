import sys
import os
import random
import time
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

from step_4_our_interface import Interface, compute_RT_errors

debug_index = 0


class Evaluator(object):
    def __init__(self, scenes_dir, scene_paths=None, debug_index=None):
        # 初始化 db
        self.scenes_dir = scenes_dir
        if scene_paths is None:
            self.scene_paths = [os.path.join(scenes_dir, i) for i in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, i))]
            # self.scene_paths = self.scene_paths[:5]
            if debug_index is not None:
                self.scene_paths = self.scene_paths[debug_index:]
        else:
            self.scene_paths = scene_paths

        # 初始化
        self.interface = Interface()
        pass

    def evaluate(self, each_object_query_num=10, seed=12345, save_path=None, vis_save_dir=None):
        global debug_index
        # 构建所有 query images
        query_images = {}
        for scene_path in self.scene_paths:
            query_images_dict = self.interface.get_query_objects(scene_path)
            for object_id, images_dict in query_images_dict.items():
                if object_id not in query_images:
                    query_images[object_id] = []
                query_images[object_id].extend(images_dict)

        # 开始以场景为单位进行评估
        random.seed(seed)

        # 记录一个以场景为单位的平均指标
        # 记录一个以类别为单位的平均指标
        data_id = 0
        scene_result_dict = {}
        class_result_dict = {}
        for scene_i, scene_path in enumerate(self.scene_paths):
            debug_index = scene_i
            scene_name = os.path.basename(scene_path)
            self.interface.set_db(data_dir=scene_path)

            scene_result_dict[scene_name] = []
            for object_id in self.interface.object_ids:
                if object_id not in class_result_dict.keys():
                    class_result_dict[object_id] = []

            for object_id in self.interface.object_ids:
                print("== Current scene: {}, object: {} ==".format(scene_name, object_id))
                object_query_images = random.sample(query_images[object_id], each_object_query_num)
                for object_query_image in object_query_images:
                    infer_result_dict = self.interface.infer(object_query_image)
                    if infer_result_dict["fail"]:
                        result_dict = {}
                        result_dict["scene_name"] = scene_name
                        result_dict["object_id"] = object_id
                        # result_dict["scores"] = scores
                        result_dict["pose"] = None
                        result_dict["gt_pose"] = object_query_image["pose"]
                        result_dict["id"] = data_id
                        result_dict["eval_angle"] = -1
                        result_dict["eval_shift"] = -1
                        pass
                    else:
                        result_dict = {}
                        result_dict["scene_name"] = scene_name
                        result_dict["object_id"] = object_id
                        # result_dict["scores"] = scores
                        result_dict["pose"] = infer_result_dict["pose"]
                        result_dict["gt_pose"] = object_query_image["pose"]
                        result_dict["id"] = data_id

                        result_dict["1-2D_2D_matching_max_score"] = infer_result_dict["1-2D_2D_matching_max_score"]
                        result_dict["1-2D_2D_matching_mean_score"] = infer_result_dict["1-2D_2D_matching_mean_score"]
                        result_dict["2-3D_2D_covis_matching_num"] = infer_result_dict["2-3D_2D_covis_matching_num"]
                        result_dict["3-RANSAC_inlier_num"] = infer_result_dict["3-RANSAC_inlier_num"]
                        result_dict["3-RANSAC_inlier_ratio"] = infer_result_dict["3-RANSAC_inlier_ratio"]

                        # evaluate
                        angle, shift = compute_RT_errors(infer_result_dict["pose"], object_query_image["pose"])
                        result_dict["eval_angle"] = angle
                        result_dict["eval_shift"] = shift
                        pass

                    # 保存结果
                    scene_result_dict[scene_name].append(result_dict)
                    class_result_dict[object_id].append(result_dict)
                    pass

            # 保存结果，为了防止中途出错，每个场景都保存一次
            if save_path is not None:
                np.save(save_path, {"scene_result_dict": scene_result_dict, "class_result_dict": class_result_dict})
            pass
        if save_path is not None:
            self.metrics_by_file(save_path)
            pass

    @staticmethod
    def metrics_by_file(result_path):
        result = np.load(result_path, allow_pickle=True).item()
        scene_result_dict = result["scene_result_dict"]
        class_result_dict = result["class_result_dict"]

        def judge_positive(result_dict):
            result = True
            if result_dict["eval_angle"] == -1:
                result = False
            # if result_dict["3-RANSAC_inlier_ratio"] < 0.5:
            #     result = False
            return result

        all_cnt = 0
        tp_fn_cnt = 0
        correct_cnt = 0
        print(len(scene_result_dict))
        for scene_name, scene_results in scene_result_dict.items():
            # print(scene_name)
            for scene_result in scene_results:
                object_id = scene_result["object_id"]
                # scores = scene_result["scores"]
                eval_angle = scene_result["eval_angle"]
                eval_shift = scene_result["eval_shift"]

                all_cnt += 1

                if eval_angle == -1:
                    continue

                if judge_positive(scene_result):
                    tp_fn_cnt += 1
                    if eval_angle < 0.5 and eval_shift < 0.5:
                        correct_cnt += 1
        print("Precision: ", correct_cnt / tp_fn_cnt)
        print("Recall: ", correct_cnt / all_cnt)
        pass

    
def main():
    scenes_dir = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/new_runs"
    save_path = "/home/huangdehao/github_projects/multi_view_rearr/tmp/our_new_tmp.npy"
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

    global debug_index
    while True:
        try:
        # if 1:
            evaluator = Evaluator(scenes_dir, scene_paths=scene_paths, debug_index=debug_index)
            evaluator.evaluate(each_object_query_num=10, seed=12345, save_path=save_path, vis_save_dir=vis_save_dir)
        except BaseException:
            debug_index += 1
            pass
        time.sleep(1)
        if debug_index > 95:
            break
    # evaluator.metrics_by_file(save_path)
    pass


if __name__ == "__main__":
    # main()
    Evaluator.metrics_by_file("/home/huangdehao/github_projects/multi_view_rearr/tmp/our_new_tmp.npy")
    pass