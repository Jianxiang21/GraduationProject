import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
latent_dim = 100
input_dim = 653  # 118 (负荷) + 535 (输出)
batch_size = 64
n_critic = 5
lambda_gp = 10
epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Generator
# -------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, z):
        return self.model(z)

# -------------------------
# Critic
# -------------------------
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
# Gradient Penalty
# -------------------------
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = critic(interpolates)
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

# -------------------------
# Training Data Placeholder
# -------------------------
# 假设你已经将所有数据合并为 numpy 数组 data ∈ [N, 653]
# 示例：data = np.load("your_data.npy")
if __name__ == "__main__":
    data_tensor = torch.load("train_data/dataset_poly.pt")[:60,:]
    data_tensor[:,172:] /= 10000

    # 数据标准化建议（可选）：
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler().fit(data)
    # data_tensor = torch.tensor(scaler.transform(data), dtype=torch.float32)

    data_loader = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

    # -------------------------
    # Instantiate Models
    # -------------------------
    generator = Generator().to(device)
    critic = Critic().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=2e-5, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # -------------------------
    # Training Loop
    # -------------------------
    gen_losses = []
    critic_losses = []
    for epoch in tqdm(range(epochs)):
        for i, real_samples in enumerate(data_loader):
            real_samples = real_samples.to(device)

            # ---------------------
            #  Train Critic
            # ---------------------
            for _ in range(n_critic):
                z = torch.randn(real_samples.size(0), latent_dim).to(device)
                fake_samples = generator(z).detach()
                real_validity = critic(real_samples)
                fake_validity = critic(fake_samples)

                gp = compute_gradient_penalty(critic, real_samples.data, fake_samples.data)
                critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

                optimizer_C.zero_grad()
                critic_loss.backward()
                optimizer_C.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            z = torch.randn(real_samples.size(0), latent_dim).to(device)
            gen_samples = generator(z)
            gen_validity = critic(gen_samples)
            gen_loss = -torch.mean(gen_validity)

            optimizer_G.zero_grad()
            gen_loss.backward()
            optimizer_G.step()
        critic_losses.append(critic_loss.item())
        gen_losses.append(gen_loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Critic loss: {critic_loss.item():.4f}, Generator loss: {gen_loss.item():.4f}")
    torch.save(generator.state_dict(), "model/generator.pth")
    torch.save(critic.state_dict(), "model/critic.pth")

    # === 绘图 ===
    plt.figure(figsize=(10, 5))
    plt.plot(critic_losses, label="Critic Loss")
    plt.plot(gen_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("WGAN-GP Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss_curve_WGANGP.png", dpi=300)
    plt.show()

    # -------------------------
    # Sampling new data
    # -------------------------
    # generator.eval()
    # with torch.no_grad():
    #     z = torch.randn(1000, latent_dim).to(device)
    #     generated_samples = generator(z).cpu().numpy()
    #     # 如果之前标准化过，这里可以 scaler.inverse_transform(generated_samples)
    #     np.save("generated_data.npy", generated_samples)
