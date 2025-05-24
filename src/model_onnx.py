import torch
from ResNet_poly import StructuredResNet

model = StructuredResNet(input_dim=118, hidden_dim=256, output_dim=54)
# 设置为 eval 模式
model.eval()

# 创建一个 dummy 输入（例如 batch size = 1）
dummy_input = torch.randn(64, 118)  # 注意输入维度是 (batch_size, input_dim)

# 导出为 ONNX 文件
torch.onnx.export(
    model,                          # 模型对象
    dummy_input,                    # 示例输入
    "resnet_model.onnx",           # 输出文件名
    input_names=["input"],          # 输入名（可选）
    output_names=["output"],        # 输出名（可选）
    dynamic_axes={                  # 动态 batch size（可选）
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11                # 推荐使用 11 或更高
)
