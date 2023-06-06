# EE5346 Project
- `step_1_mapping.py`: The first step is to map the scene and save the results to the file.
- `step_2_interface.py`: Using the results of the first step mapping, we use OnePose + + as the Baseline for Pose prediction.
- `step_3_evaluate.py`: Using the results of the first step mapping, OnePose + + was used as the Baseline evaluation result.
- `step_4_our_interface.py`: Using the results of the first step mapping, we use our ORB-SLAM + LoFTR-based method for Pose prediction.
- `step_5_our_evaluate.py`: Using the results of the first step mapping, we evaluated the results using our ORB-SLAM + LoFTR-based method.
- `ee5346.py`: Evaluate our ideas on RobotCar. Since Colmap mapping cannot be performed on RobotCar, it is adjusted to rely mainly on common view on 2D. 
