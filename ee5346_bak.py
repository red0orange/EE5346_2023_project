import sys
import os
import shutil
import numpy as np
from tqdm import tqdm

import cv2

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
sys.path.append(os.path.join(project_root_dir, "utils"))
from utils.colmap_utils.update_database_camera import camera_to_database


if __name__ == "__main__":
    data_root = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/ee5346_dataset"
    result_root = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/ee5346_dataset/results"
    test_txt_path = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/ee5346_dataset/robotcar_qAutumn_dbSunCloud_diff_final.txt"
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

    # 对每一张 query image 进行验证
    db_map_num = 30
    db_map_delta = 40
    for i, (query_image_path, ref_image_path) in tqdm(enumerate(zip(query_image_paths, ref_image_paths)), total=len(query_image_paths)):
        query_image_path = query_image_paths[i]
        ref_image_path = ref_image_paths[i]

        ref_image_name = os.path.basename(ref_image_path)
        db_image_index = db_image_names.index(ref_image_name)
        left_index  = max(db_image_index - db_map_num*db_map_delta, 0)
        right_index = min(db_image_index + db_map_num*db_map_delta, len(db_image_paths)-1)
        sampled_indices = [int(db_image_index + i*db_map_delta) for i in range(-db_map_num, db_map_num+1)]
        ref_db_image_paths = [db_image_paths[i] for i in sampled_indices if left_index <= i <= right_index]

        # if db_image_index < db_map_num:
        #     ref_db_image_paths = db_image_paths[:db_image_index+db_map_num]
        # else:
        #     ref_db_image_paths = db_image_paths[db_image_index-db_map_num:db_image_index+db_map_num]

        cur_save_root = os.path.join(result_root, "query_" + str(i))
        input_data_root = os.path.join(cur_save_root, "0_input_data")
        shutil.rmtree(cur_save_root, ignore_errors=True)
        os.makedirs(name=cur_save_root, exist_ok=False)
        image_root = os.path.join(input_data_root, "color")
        os.makedirs(name=image_root, exist_ok=False)
        for ref_db_image_path in ref_db_image_paths:
            save_path = os.path.join(image_root, os.path.basename(ref_db_image_path))
            ori_image = cv2.imread(ref_db_image_path)
            new_image = cv2.resize(ori_image, (ori_image.shape[1] // 2, ori_image.shape[0] // 2))
            cv2.imwrite(save_path, new_image)
            # shutil.copy(ref_db_image_path, os.path.join(image_root, os.path.basename(ref_db_image_path)))

        ##################### colmap 计算 Pose
        # images
        image_num = len(ref_db_image_paths)
        # camera
        my_camera_txt_path = os.path.join(input_data_root, "my_camera.txt")
        with open(my_camera_txt_path, "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {image_num}\n")
            for i in range(image_num):
                f.write(f"{i+1} PINHOLE {image_hw[1]} {image_hw[0]} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
        # mapping - feature extract
        cmd = f"cd {input_data_root} && colmap feature_extractor --database_path database.db --image_path color"
        _ = os.system(cmd)

        # update camera
        camera_to_database(os.path.join(input_data_root, "database.db"), my_camera_txt_path)

        # mapping - feature match
        cmd = f"cd {input_data_root} && colmap exhaustive_matcher --database_path database.db"
        _ = os.system(cmd)
        # mapping - sparse map
        os.makedirs(os.path.join(input_data_root, "sparse_map"))
        cmd = f"cd {input_data_root} && colmap mapper --database_path database.db --image_path color --output_path sparse_map && mv sparse_map/0/* sparse_map/ && rm -rf sparse_map/0"
        _ = os.system(cmd)

        pass
        

    pass