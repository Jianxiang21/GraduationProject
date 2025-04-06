import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 超参数设置
INPUT_DIM = 118
OUTPUT_DIM = 130  # 根据非零列动态调整,poly=130,lin=59
HIDDEN_DIM = 64  # 增加隐藏层维度
LEARNING_RATE = 0.001  # 调整学习率
BATCH_SIZE = 64
EPOCHS = 100

from plot import plot_stacked_bar_chart
# 适用于结构化数据的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim))
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.relu(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        out += residual
        return torch.relu(out)

# 结构化数据ResNet
class StructuredResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU()
        )
        
        # 残差块堆叠
        self.res_blocks = nn.Sequential(
            ResidualBlock(HIDDEN_DIM, HIDDEN_DIM),
            ResidualBlock(HIDDEN_DIM, HIDDEN_DIM),
            ResidualBlock(HIDDEN_DIM, HIDDEN_DIM)
        )
        
        self.output_layer = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

# 训练函数（修改版）
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0.0
    total_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        batch_size = inputs.size(0)
        epoch_loss += loss.item() * batch_size
        total_samples += batch_size
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx:3d}/{len(train_loader)} | Loss: {loss.item():.6f}')
    
    return epoch_loss / total_samples

# 数据加载与预处理（保持原有逻辑）
def load_data(X, y, test_ratio=0.2):
    zero_columns = np.where(np.all(y == 0, axis=0))[0]
    non_zero_columns = np.setdiff1d(np.arange(y.shape[1]), zero_columns)
    y_reduced = y[:, non_zero_columns]

    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_ratio))
    
    train_X = X[indices[:split_idx]]
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)
    std[std == 0] = 1.0

    # 保存预处理参数
    np.savez('resnet_preprocess.npz',
             mean=mean,
             std=std,
             non_zero_cols=non_zero_columns)

    # 数据转换
    train_x = (train_X - mean) / std
    test_x = (X[indices[split_idx:]] - mean) / std

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(y_reduced[indices[:split_idx]], dtype=torch.float32)
        ), batch_size=BATCH_SIZE, shuffle=True)
    
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_x, dtype=torch.float32),
            torch.tensor(y_reduced[indices[split_idx:]], dtype=torch.float32)
        ), batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, zero_columns, non_zero_columns

# 测试函数
def test_model(model, test_loader, criterion):
    model.eval()
    device = next(model.parameters()).device
    epoch_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size
    
    return epoch_loss / total_samples

# 主训练流
def main_train(X, y):
    train_loader, test_loader, zc, non_zc = load_data(X, y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StructuredResNet().to(device)
    criterion = nn.MSELoss()
    epoch_test_loss = test_model(model, test_loader, criterion)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    train_loss = []
    test_loss = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # 训练阶段
        model.train()
        epoch_train_loss = train_model(model, train_loader, criterion, optimizer)
        train_loss.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_test_loss = test_model(model, test_loader, criterion)
        test_loss.append(epoch_test_loss)
        scheduler.step(epoch_test_loss)
        
        print(f"  Train Loss: {epoch_train_loss:.6f} | Test Loss: {epoch_test_loss:.6f}")
        
        # 保存最佳模型
        if epoch_test_loss == min(test_loss):
            torch.save(model.state_dict(), 'best_resnet_model_poly.pth')
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(test_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('ResNet Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('resnet_training_curve_poly.png', dpi=300)
    plt.show()

# 预测器类
class ResNetPredictor:
    def __init__(self, model_path='best_resnet_model_poly.pth'):
        # 加载预处理参数
        params = np.load('resnet_preprocess.npz')
        self.mean = params['mean']
        self.std = params['std']
        self.non_zero_cols = params['non_zero_cols']
        
        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StructuredResNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def preprocess(self, x):
        """输入: numpy数组 (N, 118)"""
        return (x - self.mean) / self.std
    
    def predict(self, x):
        """输出: numpy数组 (N, 535)"""
        with torch.no_grad():
            x_tensor = torch.tensor(self.preprocess(x), 
                                  dtype=torch.float32).to(self.device)
            outputs = self.model(x_tensor).cpu().numpy()
            
        # 重建完整输出维度
        full_output = np.zeros((x.shape[0], 535))
        full_output[:, self.non_zero_cols] = outputs
        return full_output

if __name__ == '__main__':
    # 训练流程
    X_data = np.load('Pd.npy').T
    y_poly = np.load('y_poly.npy').T
    # y_linear = np.load('y.npy').T
    # main_train(X_data, y_data)
    predictor = ResNetPredictor()
    input_data = X_data[2000:2100,:]  # 准备(样本数, 118)的numpy数组
    output = predictor.predict(input_data)
    y_real = y_poly[2000,: 54]
    y_predict = output[0,: 54]
    plot_stacked_bar_chart(y_real, y_predict)
    # delta = output - y_data[:2,:]
    # print(delta[0])  # 打印输出结果