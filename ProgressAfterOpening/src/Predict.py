import torch
import numpy as np
from test_training import MLP

# 加载模型
model = MLP()
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()  # 切换到评估模式

X = np.load('Pd.npy').T

# 加载数据（假设输入维度是118）
input_data = X[0,:]
input_tensor = torch.tensor(input_data, dtype=torch.float32)

# 进行预测
with torch.no_grad():
    output = model(input_tensor)

print("预测输出：", output.numpy())
