"""
该脚本先使用自编码器进行特征提取，再基于低维潜在表示训练 WGAN-GP，
最后生成新数据样本并还原到原始数值范围。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import json
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
data = json.load(open('optimization_results100.json', 'r'))
cap_lagrange = []
dem_lagrange = []
optimal_solution = []
capacity = []
target_demand = []
required_demand = []
# keys_to_merge = ["supply_cap_constrs_lagrange", "required_demand_constrs_lagrange", "optimal_solution", "capacity", "target_demand", "required_demand"]

for key in data:
    data_element = data[key]
    cap_lagrange.append(np.array(data_element["supply_cap_constrs_lagrange"]))
    dem_lagrange.append(np.array(data_element["required_demand_constrs_lagrange"]))
    optimal_solution.append(np.array(data_element["optimal_solution"]).flatten())
    capacity.append(np.array(data_element["capacity"]))
    target_demand.append(np.array(data_element["target_demand"]))
    required_demand.append(np.array(data_element["required_demand"]))

# 对每一部分分别归一化
scaler_cap_lagrange = MinMaxScaler()
scaler_dem_lagrange = MinMaxScaler()
scaler_optimal_solution = MinMaxScaler()
scaler_capacity = MinMaxScaler()
scaler_target_demand = MinMaxScaler()
scaler_required_demand = MinMaxScaler()

cap_lagrange = scaler_cap_lagrange.fit_transform(cap_lagrange)
dem_lagrange = scaler_dem_lagrange.fit_transform(dem_lagrange)
optimal_solution = scaler_optimal_solution.fit_transform(optimal_solution)
capacity = scaler_capacity.fit_transform(capacity)
target_demand = scaler_target_demand.fit_transform(target_demand)
required_demand = scaler_required_demand.fit_transform(required_demand)

# 拼接所有特征，得到最终数据 (所有值均归一化到 [0, 1])
data_processed = np.concatenate([cap_lagrange, dem_lagrange, optimal_solution, capacity, target_demand, required_demand], axis=1)
num_samples, array_dim = data_processed.shape

# 转换为 PyTorch 数据集
tensor_data = torch.tensor(data_processed, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=25, shuffle=True)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# 2. 自编码器定义与训练（特征提取）
# ---------------------------
latent_dim = 64  # 自编码器的潜在空间维度，与后续 GAN 模型保持一致

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # 因为输入数据已经归一化到 [0, 1]
        )
    
    def forward(self, z):
        return self.decoder(z)

# 实例化自编码器
encoder = Encoder(input_dim=array_dim, latent_dim=latent_dim).to(device)
decoder = Decoder(latent_dim=latent_dim, input_dim=array_dim).to(device)
autoencoder = nn.Sequential(encoder, decoder)

optimizer_AE = optim.Adam(autoencoder.parameters(), lr=1e-3)
ae_epochs = 100
criterion_AE = nn.MSELoss()

print("开始训练自编码器进行特征提取...")
for epoch in range(ae_epochs):
    epoch_loss = 0.0
    for batch, in dataloader:
        batch = batch.to(device)
        optimizer_AE.zero_grad()
        reconstructed = autoencoder(batch)
        loss = criterion_AE(reconstructed, batch)
        loss.backward()
        optimizer_AE.step()
        epoch_loss += loss.item()
    print(f"自编码器 Epoch [{epoch+1}/{ae_epochs}], Loss: {epoch_loss/len(dataloader):.6f}")

# 利用训练好的 Encoder 将原始数据映射到低维潜在空间
with torch.no_grad():
    latent_reps = encoder(tensor_data.to(device)).cpu()

# 构造用于 GAN 训练的低维数据集
latent_dataset = TensorDataset(latent_reps)
latent_dataloader = DataLoader(latent_dataset, batch_size=25, shuffle=True)

# ---------------------------
# 3. WGAN-GP 模型定义与训练（在低维潜在空间上）
# ---------------------------
# 定义生成器与判别器，输入输出维度均为 latent_dim
noise_dim = 64  # 噪声向量维度

class Generator(nn.Module):
    def __init__(self, noise_dim, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# 梯度惩罚函数
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interpolates),
                                    create_graph=True, retain_graph=True)[0]
    gradients_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty

generator = Generator(noise_dim, latent_dim).to(device)
discriminator = Discriminator(latent_dim).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

n_epochs = 5002
lambda_gp = 15  # 梯度惩罚权重
d_loss_history = []
g_loss_history = []

print("开始训练 WGAN-GP（低维潜在空间）...")
for epoch in range(n_epochs):
    for i, (real_latent,) in enumerate(latent_dataloader):
        real_latent = real_latent.to(device)
        # -----------------
        # 训练判别器 4 次
        # -----------------
        for _ in range(4):
            optimizer_D.zero_grad()
            # 生成假样本（低维向量）
            z = torch.randn((real_latent.size(0), noise_dim), device=device)
            fake_latent = generator(z)
            real_loss = discriminator(real_latent).mean()
            fake_loss = discriminator(fake_latent.detach()).mean()
            gp = compute_gradient_penalty(discriminator, real_latent, fake_latent, device)
            d_loss = fake_loss - real_loss + lambda_gp * gp
            d_loss.backward()
            optimizer_D.step()
        
        # -----------------
        # 训练生成器 3 次
        # -----------------
        for _ in range(3):
            optimizer_G.zero_grad()
            z = torch.randn((real_latent.size(0), noise_dim), device=device)
            fake_latent = generator(z)
            g_loss = -discriminator(fake_latent).mean()
            g_loss.backward()
            optimizer_G.step()
    
    d_loss_history.append(d_loss.item())
    g_loss_history.append(g_loss.item())
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{n_epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# 绘制生成器和判别器的损失曲线
plt.plot(d_loss_history, label='Discriminator Loss')
plt.plot(g_loss_history, label='Generator Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("WGAN-GP Loss Curve")
plt.show()

# ---------------------------
# 4. 使用训练好的生成器生成新样本，并经 Decoder 还原回原始空间
# ---------------------------
generator.eval()
num_new_samples = 100
z = torch.randn((num_new_samples, noise_dim), device=device)
with torch.no_grad():
    # 生成低维潜在向量
    gen_latent = generator(z)
    # 使用训练好的 Decoder 将低维向量还原成原始数据
    new_samples = decoder(gen_latent).cpu().numpy()

# 对各个部分进行逆归一化处理，还原到原始范围
# 注意：这里根据原始数据拼接顺序进行逆变换
new_samples[:, 0:24] = scaler_cap_lagrange.inverse_transform(new_samples[:, 0:24])
new_samples[:, 24:64] = scaler_dem_lagrange.inverse_transform(new_samples[:, 24:64])
new_samples[:, 64:1024] = scaler_optimal_solution.inverse_transform(new_samples[:, 64:1024])
new_samples[:, 1024:1048] = scaler_capacity.inverse_transform(new_samples[:, 1024:1048])
new_samples[:, 1048:1088] = scaler_target_demand.inverse_transform(new_samples[:, 1048:1088])
new_samples[:, 1088:] = scaler_required_demand.inverse_transform(new_samples[:, 1088:])

# 保存生成的新样本到 JSON 文件
with open('new_samples.json', 'w') as f:
    json.dump(new_samples.tolist(), f)

print("新样本已保存到 'new_samples.json'")
