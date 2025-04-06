
import numpy as np

import json


# data = json.load(open('new_samples2000.json', 'r'))
# data_gen = np.array(data)
# row,_ = data_gen.shape
# dict_gen = {}
# for i in range(row):
#     dict = {}
#     dict["supply_cap_constrs_lagrange"] = data_gen[i,0:24]  # 24个
#     dict["required_demand_constrs_lagrange"] = data_gen[i,24:64] # 40个
#     dict["optimal_solution"] = data_gen[i,64:1024]    # 40*24个
#     dict["capacity"] = data_gen[i,1024:1048] # 24个
#     dict["target_demand"] = data_gen[i,1048:1088] # 40个
#     dict["required_demand"] = data_gen[i,1088:] # 40个
#     dict_gen[str(i)] = dict
# data_processed = []
# for key in data:
#     data_element = data[key]
#     # 假设你想要合并的 key 列表，按顺序排列
#     keys_to_merge = ["supply_cap_constrs_lagrange", "required_demand_constrs_lagrange", "optimal_solution", "capacity", "target_demand", "required_demand"]
#     # 从字典中提取对应的数组并合并
#     arrays_to_merge = [np.array(data_element[key]).flatten() for key in keys_to_merge]
#     merged_array = np.concatenate(arrays_to_merge, axis=0)
#     data_processed.append(merged_array)
# data_processed = np.array(data_processed)
# 加载生成器
generator2000 = torch.load('generator2500.pth')
generator4000 = torch.load('generator4000.pth')
generator5000 = torch.load('generator5000.pth')

# 生成新数据样本
generator2000.eval()
num_new_samples = 100
z = torch.randn((num_new_samples, latent_dim), device=device)
new_samples2000 = generator2000(z).detach().cpu().numpy()

# 将生成数据还原到原始范围
new_samples2000[:,0:24] = scaler_cap_lagrange.inverse_transform(new_samples2000[:,0:24])
new_samples2000[:,24:64] = scaler_dem_lagrange.inverse_transform(new_samples2000[:,24:64])
new_samples2000[:,64:1024] = scaler_optimal_solution.inverse_transform(new_samples2000[:,64:1024])
new_samples2000[:,1024:1048] = scaler_capacity.inverse_transform(new_samples2000[:,1024:1048])
new_samples2000[:,1048:1088] = scaler_target_demand.inverse_transform(new_samples2000[:,1048:1088])
new_samples2000[:,1088:] = scaler_required_demand.inverse_transform(new_samples2000[:,1088:])


#print(new_samples)
with open('new_samples2500.json', 'w') as f:
    json.dump(new_samples2000.tolist(), f)