import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# ======== Config ========
INPUT_DIM = 118
FULL_OUTPUT_DIM = 535
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 400
SAVE_NAME = 'model/linear_resnet_model_400epoch.pt'
PREPROCESS_FILE = 'model_data/linear_resnet_preprocess.npz'

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
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=None):
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

# ======== DataLoader with zero-column removal ========
def load_data_with_zero_mask(Pd_path, data_path, test_ratio=0.2):
    X = torch.load(Pd_path).numpy()
    y = torch.load(data_path).numpy()
    
    zero_columns = np.where(np.all(y == 0, axis=0))[0]
    non_zero_columns = np.setdiff1d(np.arange(y.shape[1]), zero_columns)
    y_reduced = y[:, non_zero_columns]

    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    mean, std = X[train_idx].mean(0), X[train_idx].std(0)
    std = np.where(std == 0, 1.0, std)


    np.savez(PREPROCESS_FILE, mean=mean, std=std, non_zero_cols=non_zero_columns)

    def normalize(data): return (data - mean) / std

    train_loader = DataLoader(TensorDataset(
        torch.tensor(normalize(X[train_idx]), dtype=torch.float32),
        torch.tensor(y_reduced[train_idx], dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(normalize(X[test_idx]), dtype=torch.float32),
        torch.tensor(y_reduced[test_idx], dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, non_zero_columns

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
    train_loader, test_loader, non_zero_cols = load_data_with_zero_mask(Pd_path, data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = StructuredResNet(output_dim=len(non_zero_cols)).to(device)
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

        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), SAVE_NAME)

    # Plot loss
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.legend(); plt.grid(); plt.title("Loss Curve")
    plt.savefig('loss_curve.png', dpi=300)
    plt.show()

# ======== Predictor with restoration ========
class ResNetPredictor:
    def __init__(self, model_path=SAVE_NAME, preprocess_path=PREPROCESS_FILE):
        params = np.load(preprocess_path)
        self.mean = params['mean']
        self.std = params['std']
        self.non_zero_cols = params['non_zero_cols']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = StructuredResNet(output_dim=len(self.non_zero_cols)).to(self.device)
        self.model.device = self.device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def preprocess(self, x):
        return (x - self.mean) / self.std

    def predict(self, x):
        x_np = self.preprocess(x)
        if isinstance(x_np, torch.Tensor):
            x_tensor = x_np.clone().detach().float().to(self.device)
        else:
            x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(x_tensor).cpu().numpy()

        full_output = np.zeros((x.shape[0], FULL_OUTPUT_DIM))
        full_output[:, self.non_zero_cols] = output
        return full_output


# ======== Run Entry ========
if __name__ == '__main__':
    Pd_file = 'train_data/Pd_torch.pt'
    data_file = 'train_data/linear_result.pt'

    # 检查模型是否已训练过
    if not os.path.exists(SAVE_NAME):
        print("模型不存在，开始训练...")
        main_train(Pd_file, data_file)
        print("训练完成，模型已保存到:", SAVE_NAME)
    else:
        print("检测到已有模型:", SAVE_NAME)

    # # 加载预测器进行测试
    # predictor = ResNetPredictor(model_path=SAVE_NAME, preprocess_path=PREPROCESS_FILE)
    
    # # 加载一批测试样本进行预测
    # X = torch.load(Pd_file).numpy()
    # predictions = predictor.predict(X[1000:1010])
    
    # print("预测结果示例（第1条）：")
    # print(predictions[0])

