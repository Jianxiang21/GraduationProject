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
    z = torch.randn(1000, latent_dim).to(device)
    generated_samples = generator(z).cpu().numpy()

# 保存生成样本
np.save("generated_data_new.npy", generated_samples)
print("✅ 已生成并保存 1000 个样本到 generated_data_new.npy")
