import torch
import numpy as np
from data_aug import Generator  # 确保 Generator 类可被导入

# -------------------------
# Parameters
# -------------------------
latent_dim = 100
input_dim = 653
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Generator
# -------------------------
generator = Generator().to(device)
generator.load_state_dict(torch.load("model/generator.pth", map_location=device))
generator.eval()

# -------------------------
# Generate New Data
# -------------------------
with torch.no_grad():
    z = torch.randn(8000, latent_dim).to(device)
    generated_samples = generator(z).cpu().numpy()
    # generated_samples第173列以后的列小于1的元素置0
    # generated_samples[:, 173:] = np.where(generated_samples[:, 173:] < 1, 0, generated_samples[:, 173:])

# 保存生成样本
np.save("generated_data_new1.npy", generated_samples)
# 前118列存为pt文件

# generated_Pd = torch.tensor(generated_samples[:, :118], dtype=torch.float32)
# torch.save(generated_Pd, "train_data/Pd_aug.pt")
# generated_result = torch.tensor(generated_samples[:, 118:], dtype=torch.float32)
# torch.save(generated_result, "train_data/poly_result_aug.pt")
# print("✅ 已生成并保存 8000 个样本到 generated_data_new.npy")
