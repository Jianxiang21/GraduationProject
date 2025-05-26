import torch
import numpy as np
from case118dcopf import *

# data_path = 'train_data/linear_result.pt'
# data = torch.load(data_path).numpy()
# l = data[:,54]  # lambda 平衡约束维度
# mu1plus = data[:,55:241]
# mu1minus = data[:,241:427]
# mu2plus = data[:,427:481]
# mu2minus = data[:,481:535]

# print("l", l[100:110].tolist())

# print("mu1plus", mu1plus)
# print("mu1minus", mu1minus[100,:].tolist())
# print("mu2plus", mu2plus)
# print("mu2minus", mu2minus)
# import numpy as np
# from ResNet import ResNetPredictor  # 从你的ResNet.py文件中导入预测器类

# # 初始化预测器
# predictor = ResNetPredictor(
#     model_path='model/linear_resnet_model.pt',
#     preprocess_path='model_data/linear_resnet_preprocess.npz'
# )

# # 示例：准备新的输入数据
# # 注意：输入数据的维度应该与训练时相同(118维)
# # 这里创建一个随机输入作为示例，实际使用时替换为你的真实数据
# new_input = torch.load('validate_data/linear_result_validate.pt').unsqueeze(0)  # 添加batch维度

# # 进行预测
# predictions = predictor.predict(new_input)

# # 输出预测结果
# print("预测结果形状:", predictions.shape)
# print("第一个样本的预测结果:")
# print(predictions[0])
# from case118dcopf import makeCg
# import pypower.api as pp

# ppc = pp.case118()
# Cg = makeCg(ppc)
# print(Cg)
# print(Cg[0,1])

# data = np.load('/home/lijianxiang/GraduationProject/generated_data.npy')
# print(data.shape)
# data0 = data[0]
# pd0 = data0[0:118]
# # print(pd0)
# pg0 = data0[118:172]
# # print(pg0)
# multiplier0 = data0[172:]
# powerbalance0 = multiplier0[0]
# mulinemax0 = multiplier0[1:187]
# mulinemin0 = multiplier0[187:373]
# mugmin0 = multiplier0[373:427]
# mugmax0 = multiplier0[427:]
# # print("powerbalance0", powerbalance0)




# ppc = init_ppc()
# ppc['bus'][:, 2] = pd0
# results = solve_dcopf(ppc,type='poly')
# lambda0 = results['lambda_power_balance']
# pg = results['Pg_opt']
# mu_linemax = results['lambda_line_max']
# mu_linemin = results['lambda_line_min']
# mugmin = results['lambda_pg_min']
# mugmax = results['lambda_pg_max']

# # print(lambda0)
# # print(powerbalance0)
# # print("mulinemax0", mulinemax0)
# # print("mu_linemax", mu_linemax)

# # print(pg-pg0)
# # print(mu_linemax-mulinemax0)

# print("mugmin0", mugmin0)
# print("mugmin", mugmin)
# print("mugmax0", mugmax0)
# print("mugmax", mugmax)

result = torch.load('/home/lijianxiang/GraduationProject/train_data/poly_result_aug.pt')
print(result.shape)
print(result[1:20, 54])