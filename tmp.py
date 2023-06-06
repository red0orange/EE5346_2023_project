import numpy as np

a = np.load("/home/huangdehao/github_projects/multi_view_rearr/tmp/our_new_tmp.npy", allow_pickle=True).item()
print(a)

# path = "/home/huangdehao/github_projects/multi_view_rearr/4_oneposeplus_pipeline/results.txt"
# data = np.loadtxt(path)

# data = data[:79, ...]

# tp = 0
# all_cnt = 0
# for i in data:
#     all_cnt += 1
#     if i[0] < 10 and i[1] < 10:
#         tp += 1
# print(tp, all_cnt, tp/all_cnt)