import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# ======== Config ========
INPUT_DIM = 118
FULL_OUTPUT_DIM = 1
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 500
SAVE_NAME = 'model/poly_resnet_model_500epoch_lambda1.pt'
PREPROCESS_FILE = 'model_data/poly_resnet_preprocess_lambda1.npz'


# ======== Model ========
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim))

    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.relu(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        out += residual
        return torch.relu(out)


class StructuredResNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=FULL_OUTPUT_DIM):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


# ======== DataLoader with input/lambda normalization ========
def load_data_with_zero_mask(Pd_path, data_path, test_ratio=0.2):
    X = torch.load(Pd_path).numpy()
    y = torch.load(data_path).numpy()[:, 54]  # lambda 平衡约束维度

    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    # X 归一化
    mean, std = X[train_idx].mean(0), X[train_idx].std(0)
    std = np.where(std == 0, 1.0, std)

    # y 归一化
    lambda_mean = y[train_idx].mean()
    lambda_std = y[train_idx].std()

    np.savez(PREPROCESS_FILE, mean=mean, std=std, lambda_mean=lambda_mean, lambda_std=lambda_std)

    def normalize(data): return (data - mean) / std
    def normalize_lambda(lmbd): return (lmbd - lambda_mean) / lambda_std

    train_loader = DataLoader(TensorDataset(
        torch.tensor(normalize(X[train_idx]), dtype=torch.float32),
        torch.tensor(normalize_lambda(y[train_idx]), dtype=torch.float32).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(normalize(X[test_idx]), dtype=torch.float32),
        torch.tensor(normalize_lambda(y[test_idx]), dtype=torch.float32).unsqueeze(1)),
        batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


# ======== Training & Evaluation ========
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_samples = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
    return total_loss / total_samples


def test_model(model, loader, criterion):
    model.eval()
    total_loss, total_samples = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            loss = criterion(model(inputs), targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples


# ======== Training Pipeline ========
def main_train(Pd_path, data_path):
    train_loader, test_loader = load_data_with_zero_mask(Pd_path, data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StructuredResNet().to(device)
    model.device = device

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = nn.MSELoss()

    best_loss, train_losses, test_losses = float('inf'), [], []

    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion)
        test_loss = test_model(model, test_loader, criterion)
        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), SAVE_NAME)

    # Plot loss
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.legend()
    plt.grid()
    plt.title("Loss Curve")
    plt.savefig('poly_loss_curve_lambda.png', dpi=300)
    plt.show()


# ======== Predictor with inverse normalization ========
class ResNetPredictor:
    def __init__(self, model_path=SAVE_NAME, preprocess_path=PREPROCESS_FILE):
        params = np.load(preprocess_path)
        self.mean = params['mean']
        self.std = params['std']
        self.lambda_mean = params['lambda_mean']
        self.lambda_std = params['lambda_std']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StructuredResNet().to(self.device)
        self.model.device = self.device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, x):
        return (x - self.mean) / self.std

    def predict(self, x):
        x_tensor = torch.tensor(self.preprocess(x), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(x_tensor).cpu().numpy()
        return output * self.lambda_std + self.lambda_mean  # 反标准化输出


# ======== Run Entry ========
if __name__ == '__main__':
    Pd_file = 'train_data/Pd_torch.pt'
    data_file = 'train_data/poly_result.pt'

    if not os.path.exists(SAVE_NAME):
        print("模型不存在，开始训练...")
        main_train(Pd_file, data_file)
        print("训练完成，模型已保存到:", SAVE_NAME)
    else:
        print("检测到已有模型:", SAVE_NAME)

    predictor = ResNetPredictor()
    X = torch.load(Pd_file).numpy()
    predictions = predictor.predict(X[1000:1010])
    y = torch.load(data_file).numpy()[:, 54][1000:1010]

    print("真实值（第1条）：", y[1])
    print("预测结果（第1条）：", predictions[1])
