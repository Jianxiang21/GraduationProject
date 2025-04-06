"""
This file trys to implement GAN for the dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import json
from sklearn.preprocessing import MinMaxScaler


data = json.load(open('optimization_results.json', 'r'))
data_processed = []
for key in data:
    data_element = data[key]
    # 假设你想要合并的 key 列表，按顺序排列
    keys_to_merge = ["supply_cap_constrs_lagrange", "required_demand_constrs_lagrange", "optimal_solution", "capacity", "target_demand", "required_demand"]
    # 从字典中提取对应的数组并合并
    arrays_to_merge = [np.array(data_element[key]).flatten() for key in keys_to_merge]
    merged_array = np.concatenate(arrays_to_merge, axis=0)
    data_processed.append(merged_array)
data_processed = np.array(data_processed)

# 用min max归一化data_processed
scaler = MinMaxScaler()
data_processed = scaler.fit_transform(data_processed)

# 获得data_processed的shape
num_samples, array_dim = data_processed.shape

# 转换为 PyTorch 数据集
tensor_data = torch.tensor(data_processed, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, array_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, array_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, array_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(array_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, x):
        return self.model(x)


latent_dim = 50  # 噪声维度

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(latent_dim, array_dim).to(device)
discriminator = Discriminator(array_dim).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

adversarial_loss = nn.BCELoss()  # 二分类交叉熵损失


n_epochs = 500

for epoch in range(n_epochs):
    for i, (real_samples,) in enumerate(dataloader):
        real_samples = real_samples.to(device)

        # 标签
        valid = torch.ones((real_samples.size(0), 1), device=device)
        fake = torch.zeros((real_samples.size(0), 1), device=device)

        # -----------------
        #  训练生成器
        # -----------------
        optimizer_G.zero_grad()

        # 生成噪声并生成数组
        z = torch.randn((real_samples.size(0), latent_dim), device=device)
        generated_samples = generator(z)

        # 生成器希望判别器认为生成数据是真的
        g_loss = adversarial_loss(discriminator(generated_samples), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  训练判别器
        # ---------------------
        optimizer_D.zero_grad()

        # 判别器分别计算真实和生成数据的损失
        real_loss = adversarial_loss(discriminator(real_samples), valid)
        fake_loss = adversarial_loss(discriminator(generated_samples.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch [{epoch}/{n_epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


generator.eval()

# 生成新数据样本
num_new_samples = 100  # 需要增强的数据样本数
z = torch.randn((num_new_samples, latent_dim), device=device)
new_samples = generator(z).detach().cpu().numpy()

# 将生成数据还原到原始范围
new_samples = new_samples * 0.5 + 0.5
print(new_samples)

