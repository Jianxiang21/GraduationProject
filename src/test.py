# import torch
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
from case118dcopf import makeCg
import pypower.api as pp

ppc = pp.case118()
Cg = makeCg(ppc)
print(Cg)
print(Cg[0,1])