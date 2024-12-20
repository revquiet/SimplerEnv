import pickle
import numpy as np
# 打开 pkl 文件
with open('./sysid_log/sysid_dataset_single_ep.pkl', 'rb') as f:
# with open('/home/fftai/Downloads/sysid_dataset.pkl', "rb") as f:
    data = pickle.load(f)

if isinstance(data, list):
    print("列表长度:", len(data))
    for i in range(len(data)):
        print(len(data[i]))
    print("首个元素的类型:", type(data[0]))
    if isinstance(data[0], list): # or isinstance(data[0], np.ndarray):
        print("首个元素的维度:", len(data[0])) #if isinstance(data[0], list) else data[0].shape)
        print(data[0][3])
        print(data[0][4])
        print(data[0][5]["action_world_vector"])
        print(data[0][5]["action_rotation_delta"])
        print(np.concatenate((data[0][5]["action_world_vector"],data[0][5]["action_rotation_delta"]),axis=0))


# 查看数据
# print(len(data))
# print(data[1])
# for i in range(len(data)):
#     # print(np.array(data[i]).shape)
#     pass