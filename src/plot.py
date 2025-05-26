import numpy as np
from case118dcopf import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import ResNet_poly
import ResNet_poly_lambda
import ResNet_poly_mu
import ResNet
import os




POLY_MODEL_PATH = "model/poly_resnet_model_50epoch54output.pt"
POLY_MODEL_PATH_LAMBDA = "model/poly_resnet_model_500epoch_lambda1.pt"
POLY_MODEL_PATH_MU = "model/poly_resnet_model_mu_50epoch.pt"
POLY_PROCESS_FILE_LAMBDA = "model_data/poly_resnet_preprocess_lambda1.npz"
POLY_PROCESS_FILE = "model_data/poly_resnet_preprocess_54.npz"
POLY_PROCESS_FILE_MU = "model_data/poly_resnet_preprocess_mu.npz"

LINEAR_MODEL_PATH = "model/linear_resnet_model_100epoch.pt"
LINEAR_PROCESS_FILE = "model_data/linear_resnet_preprocess.npz"

# PD_FILE = "train_data/Pd_torch.pt"
PD_FILE = "validate_data/Pd_validate.pt"

predictor_optimal = ResNet_poly.ResNetPredictor(
    model_path=POLY_MODEL_PATH,
    preprocess_path=POLY_PROCESS_FILE
)
predictor_lambda = ResNet_poly_lambda.ResNetPredictor(
    model_path=POLY_MODEL_PATH_LAMBDA,
    preprocess_path=POLY_PROCESS_FILE_LAMBDA
)
predictor_mu = ResNet_poly_mu.ResNetPredictor(
    model_path=POLY_MODEL_PATH_MU,
    preprocess_path=POLY_PROCESS_FILE_MU
)
predictor_linear = ResNet.ResNetPredictor(
    model_path=LINEAR_MODEL_PATH,
    preprocess_path=LINEAR_PROCESS_FILE
)

Pd = torch.load(PD_FILE).numpy()
real_poly = torch.load("validate_data/poly_result_validate.pt").numpy()
real_Pg_poly = real_poly[:, :54]
real_lambda_poly = real_poly[:, 54]/10000
real_mu_poly = real_poly[:, 56:]

pred_Pg_poly = predictor_optimal.predict(Pd)
pred_lambda_poly = predictor_lambda.predict(Pd).squeeze()/10000
pred_mu_poly = predictor_mu.predict(Pd)

# 转换为 NumPy 以便处理
# pred_Pg_np = pred_Pg_poly.cpu().numpy()
# real_Pg_np = real_Pg_poly.cpu().numpy()

# 每个样本的 RMS 误差（只对 Pg）
rms_errors_pg = np.sqrt(np.mean((pred_Pg_poly - real_Pg_poly) ** 2, axis=1))

# 最大和最小 RMS 的样本索引
max_rms_idx = np.argmax(rms_errors_pg)
min_rms_idx = np.argmin(rms_errors_pg)

abs_errors = np.abs(pred_lambda_poly - real_lambda_poly)

# 最大和最小误差的索引
max_err_idx = np.argmax(abs_errors)
min_err_idx = np.argmin(abs_errors)
# print(pred_lambda_poly)
# print(real_lambda_poly)
# print(pred_lambda_poly.shape)
# print(real_lambda_poly.shape)
# print("max_err_idx", max_err_idx)
# print("min_err_idx", min_err_idx)

def plot_pg_bar_comparison(y_true, y_pred, sample_idx, title, filename):
    n_outputs = len(y_true)
    x = np.arange(n_outputs)

    plt.figure(figsize=(12, 4))
    plt.bar(x - 0.2, y_true, width=0.4, label='True', alpha=0.7)
    plt.bar(x + 0.2, y_pred, width=0.4, label='Predicted', alpha=0.7)
    plt.xlabel('Generator Index')
    plt.ylabel('Active Power Output (Pg)')
    plt.title(f"{title} (Sample #{sample_idx})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename,dpi = 500)
    plt.show()

def plot_lambda_comparison(true_val, pred_val, sample_idx, title, filename):
    plt.figure(figsize=(6, 4))
    categories = ['True', 'Predicted']
    values = [true_val, pred_val]
    plt.bar(categories, values, color=['blue', 'orange'], alpha=0.7)
    plt.ylabel('Lagrange Multiplier for Power Balance Constraint')
    plt.title(f"{title} (Sample #{sample_idx})")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=500)
    plt.show()

def plot_pg_comparison():
    # 读取数据
    data = np.load('generated_data.npy')
    data0 = data[0,:]
    Pd = data0[0:118]
    ppc = init_ppc()
    ppc['bus'][:, 2] = Pd
    pg_gen = data0[118:118+54]
    result = solve_dcopf(ppc, 'poly')
    pg = result['Pg_opt']

    # 绘制条形图比较
    x = np.arange(len(pg_gen))  # 发电机编号
    bar_width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - bar_width/2, pg_gen, bar_width, label='Generated Pg')
    plt.bar(x + bar_width/2, pg, bar_width, label='Gurobi Pg')

    plt.xlabel('Generator Index')
    plt.ylabel('Power Output (MW)')
    plt.title('Comparison of Generated vs. Gurobi-Calculated Generator Outputs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def pca_plot(real_data, fake_data):
    # 使用 PCA 降维到二维
    X = np.vstack([real_data, fake_data])
    y = np.array([0]*len(real_data) + [1]*len(fake_data))  # 0: 真实数据, 1: 生成数据
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', label='Real Data', alpha=0.6)
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='orange', label='Generated Data', alpha=0.6)
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title('PCA Projection of Real vs Generated Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def tsne_plot(real_data, fake_data):
    # 使用 t-SNE 降维到二维
    X = np.vstack([real_data, fake_data])
    y = np.array([0]*len(real_data) + [1]*len(fake_data))  # 0: 真实数据, 1: 生成数据

    # 先PCA降到50维，再用TSNE
    X_pca = PCA(n_components=50).fit_transform(X)
    os.environ["OMP_NUM_THREADS"] = "1"
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)

    # 可视化
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], c='blue', label='Real Data', alpha=0.6)
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], c='orange', label='Generated Data', alpha=0.6)
    plt.xlabel('t-SNE-1')
    plt.ylabel('t-SNE-2')
    plt.title('t-SNE Visualization of Real vs Generated Data')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # print("hello world")
    # Plot the bar comparison for max and min RMS error samples
    # plot_pg_bar_comparison(real_Pg_poly[max_rms_idx], pred_Pg_poly[max_rms_idx], max_rms_idx, "Max RMS Error", "D:/Senior/thuthesis-v7.6.0/figures/max_rms_pg_comparison.png")
    # plot_pg_bar_comparison(real_Pg_poly[min_rms_idx], pred_Pg_poly[min_rms_idx], min_rms_idx, "Min RMS Error", "D:/Senior/thuthesis-v7.6.0/figures/min_rms_pg_comparison.png")

    # # Plot the comparison of generated data vs Gurobi results
    # # plot_pg_comparison()
    # plot_lambda_comparison(real_lambda_poly[max_err_idx], pred_lambda_poly[max_err_idx], max_err_idx, "Max Absolute Error", "D:/Senior/thuthesis-v7.6.0/figures/max_rms_lambda_comparison.png")
    # plot_lambda_comparison(real_lambda_poly[min_err_idx], pred_lambda_poly[min_err_idx], min_err_idx, "Min Absolute Error", "D:/Senior/thuthesis-v7.6.0/figures/min_rms_lambda_comparison.png")
    data_tensor = torch.load("train_data/dataset_poly.pt")
    data_tensor[:,172:] /= 10000
    real_data = data_tensor.numpy()
    fake_data = np.load("generated_data_new1.npy")

    pca_plot(real_data, fake_data)
    # tsne_plot(real_data, fake_data)