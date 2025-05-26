import numpy as np
from case118dcopf import get_params,init_ppc, solve_dcopf
import pypower.api as pp
import torch
import ResNet_poly
import ResNet_poly_lambda
import ResNet_poly_mu
import ResNet
import ResNet_poly_aug
from timeit import default_timer as timer
import matplotlib.pyplot as plt


POLY_AUG_PATH = "model/poly_resnet_aug.pt"
POLY_MODEL_PATH = "model/poly_resnet_model_50epoch54output.pt"
POLY_MODEL_PATH_LAMBDA = "model/poly_resnet_model_500epoch_lambda1.pt"
POLY_MODEL_PATH_MU = "model/poly_resnet_model_mu_50epoch.pt"
POLY_PROCESS_FILE_LAMBDA = "model_data/poly_resnet_preprocess_lambda1.npz"
POLY_PROCESS_FILE = "model_data/poly_resnet_preprocess_54.npz"
POLY_PROCESS_FILE_MU = "model_data/poly_resnet_preprocess_mu.npz"
POLY_PROCESS_FILE_AUG = "model_data/poly_resnet_preprocess_aug.npz"

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
predictor_poly_aug = ResNet_poly_aug.ResNetPredictor(
    model_path=POLY_AUG_PATH,
    preprocess_path=POLY_PROCESS_FILE_AUG
)

def setup_ppc(idx, pd_file=PD_FILE):
    """
    设置 ppc 的参数
    :return: 初始化的 ppc
    """
    ppc = init_ppc()
    p_load = torch.load(pd_file)
    if idx >= p_load.shape[0]:
        raise ValueError(f"Index {idx} out of range for p_load data.")
    else:
        p_load = p_load[idx,:].numpy()
        ppc["bus"][:, 2] = p_load  # 负荷功率
    return ppc

def primal_dual_lp(
    ppc,
    alpha=1e-3, alpha_pg = 0.1, max_iter=500, tol=1e-4,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,plot = False
):
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # 负荷功率
    Pd_total = np.sum(p_load)  # 总负荷功率
    m = len(c2)
    l = len(Fmax)

    # 初值设定（支持用户提供预测初值）
    pg = init_pg if init_pg is not None else np.zeros(m)
    lamda = init_lambda if init_lambda is not None else 0.0
    mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
    mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
    mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
    mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

    # pg = predictor_optimal.predict(np.array([p_load])).squeeze(0)
    # lamda = - predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000
    # mu_pred = predictor_mu.predict(np.array([p_load])).squeeze(0)

    # mu1_plus = mu_pred[:l]
    # mu1_minus = mu_pred[l:2*l]
    # mu2_plus = mu_pred[2*l:2*l+m]
    # mu2_minus = mu_pred[2*l+m:]

    # C2_diag = np.diag(2 * c2)  # (1/2c2) inverse diag matrix

    residual_list = []

    for it in range(max_iter):
        primal_res = np.abs(np.sum(pg) - Pd_total)
        residual_list.append(primal_res)
        if primal_res < tol:
            print(f"Converged at iter {it}")
            break
        # 更新 pg（用解析公式）
        grad_term = c1 + lamda * np.ones(m) \
                    + Cg.T @ h.T @ mu1_plus - Cg.T @ h.T @ mu1_minus \
                    + mu2_plus - mu2_minus
        pg -= alpha_pg * grad_term

        # 投影到可行域
        pg = np.clip(pg, gmin, gmax)

        # 更新拉格朗日乘子（对偶梯度上升）
        flow = h @ (Cg @ pg - p_load)

        lamda += alpha * (np.sum(pg) - Pd_total)
        mu1_plus += alpha * (flow - Fmax)
        mu1_minus -= alpha * (flow + Fmax)
        mu2_plus += alpha * (pg - gmax)
        mu2_minus -= alpha * (pg - gmin)

        # 投影到非负
        # lamda = np.maximum(lamda, 0)
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

    if it == max_iter - 1:
        # 如果没有收敛，输出最后的结果
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_res)

    if plot:
        plt.plot(residual_list)
        plt.xlabel("Iteration")
        plt.ylabel("Primal Residual")
        plt.title("Primal Residual Convergence")
        plt.grid(True)
        # plt.show()
        plt.savefig("primal_residual_lp", dpi=500)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus,residual_list

def primal_dual_lp_smooth(
    ppc,
    alpha=1e-3, alpha_pg = 0.1, beta = 0.2, max_iter=500, tol=1e-4,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,plot=False
):
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # 负荷功率
    Pd_total = np.sum(p_load)  # 总负荷功率
    m = len(c2)
    l = len(Fmax)

    # 初值设定（支持用户提供预测初值）
    pg = init_pg if init_pg is not None else np.zeros(m)
    lamda = init_lambda if init_lambda is not None else 0.0
    mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
    mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
    mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
    mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

    residual_list = []  # 用于记录每次迭代的残差

    for it in range(max_iter):
        # 收敛性检测（你也可以增加 Lagrangian 残差或 KKT 条件）
        primal_res = np.abs(np.sum(pg) - Pd_total)
        residual_list.append(primal_res)
        if primal_res < tol:
            print(f"Converged at iter {it}")
            break
        # 更新 pg（用解析公式）
        grad_term = c1 + lamda * np.ones(m) \
                    + Cg.T @ h.T @ mu1_plus - Cg.T @ h.T @ mu1_minus \
                    + mu2_plus - mu2_minus
        # 解析的 pg 更新
        pg_new = pg - alpha_pg * grad_term

        # 加入 beta 平滑
        pg = (1 - beta) * pg + beta * pg_new

        # 投影到可行域
        pg = np.clip(pg, gmin, gmax)

        # 更新拉格朗日乘子（对偶梯度上升）
        flow = h @ (Cg @ pg - p_load)

        lamda += alpha * (np.sum(pg) - Pd_total)
        mu1_plus += alpha * (flow - Fmax)
        mu1_minus -= alpha * (flow + Fmax)
        mu2_plus += alpha * (pg - gmax)
        mu2_minus -= alpha * (pg - gmin)

        # 投影到非负
        # lamda = np.maximum(lamda, 0)
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        
    if it == max_iter - 1:
        # 如果没有收敛，输出最后的结果
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_res)
    if plot:
        plt.plot(residual_list)
        plt.xlabel("Iteration")
        plt.ylabel("Primal Residual")
        plt.title("Primal Residual Convergence")
        plt.grid(True)
        # plt.show()
        plt.savefig("primal_residual_smooth", dpi=500)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus, residual_list

def primal_dual_lp_smooth_pred(
    ppc,predictor,
    alpha=1e-3, alpha_pg = 0.1, beta = 0.2, max_iter=500, tol=1e-4,plot=True,
):
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # 负荷功率
    Pd_total = np.sum(p_load)  # 总负荷功率
    m = len(c2)
    l = len(Fmax)

    predict = predictor.predict(np.array([p_load])).squeeze(0)
    pg = predict[:m]
    lamda = - predict[m]
    mu1_plus = predict[m+1:m+1+l]
    mu1_minus = predict[m+1+l:m+1+2*l]
    mu2_plus = predict[m+1+2*l:m+1+2*l+m]
    mu2_minus = predict[m+1+2*l+m:]

    residual_list = []  # 用于记录每次迭代的残差

    for it in range(max_iter):
        # 收敛性检测（你也可以增加 Lagrangian 残差或 KKT 条件）
        primal_res = np.abs(np.sum(pg) - Pd_total)
        residual_list.append(primal_res)
        if primal_res < tol:
            print(f"Converged at iter {it}")
            break
        # 更新 pg（用解析公式）
        grad_term = c1 + lamda * np.ones(m) \
                    + Cg.T @ h.T @ mu1_plus - Cg.T @ h.T @ mu1_minus \
                    + mu2_plus - mu2_minus
        # 解析的 pg 更新
        pg_new = pg - alpha_pg * grad_term

        # 加入 beta 平滑
        pg = (1 - beta) * pg + beta * pg_new

        # 投影到可行域
        pg = np.clip(pg, gmin, gmax)

        # 更新拉格朗日乘子（对偶梯度上升）
        flow = h @ (Cg @ pg - p_load)

        lamda += alpha * (np.sum(pg) - Pd_total)
        mu1_plus += alpha * (flow - Fmax)
        mu1_minus -= alpha * (flow + Fmax)
        mu2_plus += alpha * (pg - gmax)
        mu2_minus -= alpha * (pg - gmin)

        # 投影到非负
        # lamda = np.maximum(lamda, 0)
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        
    if it == max_iter - 1:
        # 如果没有收敛，输出最后的结果
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_res)
    if plot:
        plt.plot(residual_list)
        plt.xlabel("Iteration")
        plt.ylabel("Primal Residual")
        plt.title("Primal Residual Convergence")
        plt.grid(True)
        # plt.show()
        plt.savefig("primal_residual_lp_smooth_pred", dpi=500)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus, residual_list

def primal_dual_qp(
    ppc,
    alpha=1e-3, max_iter=500, tol=1e-4,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,plot = False
):
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # 负荷功率
    Pd_total = np.sum(p_load)  # 总负荷功率
    m = len(c2)
    l = len(Fmax)

    # 初值设定（支持用户提供预测初值）
    pg = init_pg if init_pg is not None else np.zeros(m)
    lamda = init_lambda if init_lambda is not None else 0.0
    mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
    mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
    mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
    mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

    C2_inv_diag = np.diag(1 / (2 * c2))  # (1/2c2) inverse diag matrix

    residual_list = []  # 用于记录每次迭代的残差

    for it in range(max_iter):
        # 收敛性检测（你也可以增加 Lagrangian 残差或 KKT 条件）
        primal_res = np.abs(np.sum(pg) - Pd_total)
        residual_list.append(primal_res)
        if primal_res < tol:
            print(f"Converged at iter {it}")
            break
        # 更新 pg（用解析公式）
        grad_term = -c1 - lamda * np.ones(m) \
                    - Cg.T @ h.T @ mu1_plus + Cg.T @ h.T @ mu1_minus \
                    - mu2_plus + mu2_minus
        pg = C2_inv_diag @ grad_term

        # 投影到可行域
        pg = np.clip(pg, gmin, gmax)

        # 更新拉格朗日乘子（对偶梯度上升）
        flow = h @ (Cg @ pg - p_load)

        lamda += alpha * (np.sum(pg) - Pd_total)
        mu1_plus += alpha * (flow - Fmax)
        mu1_minus -= alpha * (flow + Fmax)
        mu2_plus += alpha * (pg - gmax)
        mu2_minus -= alpha * (pg - gmin)

        # 投影到非负
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        # print(f"----------Iteration {it}-----------")

        
    if it == max_iter - 1:
        # 如果没有收敛，输出最后的结果
        print("Warning: Maximum iterations reached without convergence.")
    
    if plot:
        plt.plot(residual_list)
        plt.xlabel("Iteration")
        plt.ylabel("Primal Residual")
        plt.title("Primal Residual Convergence")
        plt.grid(True)
        # plt.show()
        plt.savefig("primal_residual_qp", dpi=500)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus, residual_list

def primal_dual_qp_smooth(
    ppc,
    alpha=1e-3, max_iter=500, tol=1e-4, beta=0.2, 
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,plot = False
):
    # 解析问题参数
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # 负荷功率
    Pd_total = np.sum(p_load)  # 总负荷功率
    # print("Pd_total:", Pd_total)
    m = len(c2)
    l = len(Fmax)

    # 初值设定（支持用户提供预测初值）
    pg = init_pg if init_pg is not None else np.zeros(m)
    lamda = init_lambda if init_lambda is not None else 0.0
    mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
    mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
    mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
    mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

    # 预计算逆矩阵
    C2_inv_diag = np.diag(1 / (2 * c2))  # (1/2c2) inverse diag matrix

    residual_list = []  # 用于记录每次迭代的残差

    for it in range(max_iter):
        primal_res = np.abs(np.sum(pg) - Pd_total)
        residual_list.append(primal_res)
        if primal_res < tol:
            print(f"Converged at iteration {it}")
            break
        # --- Step 1: 计算解析的pg更新方向 ---
        grad_term = -c1 - lamda * np.ones(m) \
                    - Cg.T @ h.T @ mu1_plus + Cg.T @ h.T @ mu1_minus \
                    - mu2_plus + mu2_minus
        pg_new = C2_inv_diag @ grad_term

        # --- Step 2: 平滑更新（引入beta） ---
        pg = (1 - beta) * pg + beta * pg_new

        # --- Step 3: 投影到物理可行域 ---
        pg = np.clip(pg, gmin, gmax)

        # --- Step 4: 更新拉格朗日乘子（对偶变量） ---
        flow = h @ (Cg @ pg - p_load)

        lamda += alpha * (np.sum(pg) - Pd_total)
        mu1_plus += alpha * (flow - Fmax)
        mu1_minus -= alpha * (flow + Fmax)
        mu2_plus += alpha * (pg - gmax)
        mu2_minus -= alpha * (pg - gmin)

        # 投影对偶变量到非负域
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        # --- Step 5: 收敛性检测 ---
        

    if it == max_iter - 1:
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_res)

    if plot:
        plt.plot(residual_list)
        # plt.yscale("log")  # 使用对数坐标系以便更好地查看收敛情况
        plt.xlabel("Iteration")
        plt.ylabel("Primal Residual")
        plt.title("Primal Residual Convergence")
        plt.grid(True)
        # plt.show()
        plt.savefig("primal_residual_qp_smooth", dpi=500)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus, residual_list

def primal_dual_qp_smooth_pred(
    ppc, predictor_optimal, predictor_lambda, predictor_mu,
    alpha=1e-3, beta=0.2,  # 新增beta参数
    max_iter=500, tol=1e-4, plot=True
):
    # 解析问题参数
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # 负荷功率
    Pd_total = np.sum(p_load)  # 总负荷功率
    m = len(c2)
    l = len(Fmax)

    # 初值设定（基于预测器）
    pg = predictor_optimal.predict(np.array([p_load])).squeeze(0)
    lamda = - predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000
    mu_pred = predictor_mu.predict(np.array([p_load])).squeeze(0)

    mu1_plus = mu_pred[:l]
    mu1_minus = mu_pred[l:2*l]
    mu2_plus = mu_pred[2*l:2*l+m]
    mu2_minus = mu_pred[2*l+m:]

    # 预计算逆矩阵
    C2_inv_diag = np.diag(1 / (2 * c2))  # (1/2c2) inverse diag matrix

    residual_list = []  # 用于记录每次迭代的残差

    for it in range(max_iter):
        # --- Step 1: 计算解析的pg更新方向 ---
        grad_term = -c1 - lamda * np.ones(m) \
                    - Cg.T @ h.T @ mu1_plus + Cg.T @ h.T @ mu1_minus \
                    - mu2_plus + mu2_minus
        pg_new = C2_inv_diag @ grad_term

        # --- Step 2: 平滑更新（引入beta） ---
        pg = (1 - beta) * pg + beta * pg_new

        # --- Step 3: 投影到物理可行域 ---
        pg = np.clip(pg, gmin, gmax)

        # --- Step 4: 更新拉格朗日乘子（对偶变量） ---
        flow = h @ (Cg @ pg - p_load)

        lamda += alpha * (np.sum(pg) - Pd_total)
        mu1_plus += alpha * (flow - Fmax)
        mu1_minus -= alpha * (flow + Fmax)
        mu2_plus += alpha * (pg - gmax)
        mu2_minus -= alpha * (pg - gmin)

        # 投影对偶变量到非负域
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        # --- Step 5: 收敛性检测 ---
        primal_res = np.abs(np.sum(pg) - Pd_total)
        residual_list.append(primal_res)

        if primal_res < tol:
            print(f"Converged at iteration {it}")
            break

    if it == max_iter - 1:
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_res)

    # 如果需要绘制残差图
    if plot:
        plt.plot(residual_list)
        # plt.yscale("log")  # 使用对数坐标系以便更好地查看收敛情况
        plt.xlabel("Iteration")
        plt.ylabel("Primal Residual")
        plt.title("Primal Residual Convergence")
        plt.grid(True)
        plt.show()

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus, residual_list

def primal_dual_qp_smooth_pred_aug(
    ppc, predictor,
    alpha=1e-3, beta=0.2,  # 新增beta参数
    max_iter=500, tol=1e-4, plot=True
):
    # 解析问题参数
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # 负荷功率
    Pd_total = np.sum(p_load)  # 总负荷功率
    m = len(c2)
    l = len(Fmax)

    # 初值设定（基于预测器）
    result = predictor.predict(np.array([p_load])).squeeze(0)
    pg = result[:m]
    lamda = - result[m]
    print("lamda:", lamda)
    mu1_plus = result[m+1:m+1+l]
    mu1_minus = result[m+1+l:m+1+2*l]
    mu2_plus = result[m+1+2*l:m+1+2*l+m]
    mu2_minus = result[m+1+2*l+m:]

    # lamda = - predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000
    # mu_pred = predictor_mu.predict(np.array([p_load])).squeeze(0)

    # mu1_plus = mu_pred[:l]
    # mu1_minus = mu_pred[l:2*l]
    # mu2_plus = mu_pred[2*l:2*l+m]
    # mu2_minus = mu_pred[2*l+m:]

    # 预计算逆矩阵
    C2_inv_diag = np.diag(1 / (2 * c2))  # (1/2c2) inverse diag matrix

    residual_list = []  # 用于记录每次迭代的残差
    print("Initial primal residual:", np.abs(np.sum(pg) - Pd_total))

    for it in range(max_iter):
        # --- Step 5: 收敛性检测 ---
        primal_res = np.abs(np.sum(pg) - Pd_total)
        residual_list.append(primal_res)
        if primal_res < tol:
            print(f"Converged at iteration {it}")
            break
        # --- Step 1: 计算解析的pg更新方向 ---
        grad_term = -c1 - lamda * np.ones(m) \
                    - Cg.T @ h.T @ mu1_plus + Cg.T @ h.T @ mu1_minus \
                    - mu2_plus + mu2_minus
        pg_new = C2_inv_diag @ grad_term

        # --- Step 2: 平滑更新（引入beta） ---
        pg = (1 - beta) * pg + beta * pg_new

        # --- Step 3: 投影到物理可行域 ---
        pg = np.clip(pg, gmin, gmax)

        # --- Step 4: 更新拉格朗日乘子（对偶变量） ---
        flow = h @ (Cg @ pg - p_load)

        lamda += alpha * (np.sum(pg) - Pd_total)
        mu1_plus += alpha * (flow - Fmax)
        mu1_minus -= alpha * (flow + Fmax)
        mu2_plus += alpha * (pg - gmax)
        mu2_minus -= alpha * (pg - gmin)

        # 投影对偶变量到非负域
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        

        

    if it == max_iter - 1:
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_res)

    # 如果需要绘制残差图
    if plot:
        plt.plot(residual_list)
        # plt.yscale("log")  # 使用对数坐标系以便更好地查看收敛情况
        plt.xlabel("Iteration")
        plt.ylabel("Primal Residual")
        plt.title("Primal Residual Convergence")
        plt.grid(True)
        # plt.savefig("primal_residual_qp_smooth_pred_aug", dpi=500)
        plt.show()

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus, residual_list


if __name__ == "__main__":
    ppc = setup_ppc(53)
    primal_dual_qp_smooth_pred_aug(
        ppc, predictor_poly_aug,
        alpha=0.002, beta=0.16, max_iter=1000, tol=1e-3, plot=True
    )
    # lp_time = []
    # qp_time = []
    # lp_smooth_time = []
    # qp_smooth_time = []
    # lp_smooth_pred_time = []
    # qp_smooth_pred_time = []
    # gurobi_qp_time = []
    # gurobi_lp_time = []
    # for i in range(10):
    #     start = timer()
    #     *_, residual_list_lp = primal_dual_lp(
    #         ppc,
    #         alpha=0.1, max_iter=4000, tol=1e-3, alpha_pg=0.15,plot=False
    #     )
    #     end = timer()
    #     lp_time.append(end - start)

    #     start = timer()
    #     *_, residual_list_lp_smooth = primal_dual_lp_smooth(
    #         ppc,
    #         alpha=0.1, beta = 0.18, max_iter=4000, tol=1e-3, alpha_pg=0.15,plot=False
    #     )
    #     end = timer()
    #     lp_smooth_time.append(end - start)

    #     start = timer()
    #     *_, residual_list_qp = primal_dual_qp(
    #         ppc,
    #         alpha=0.002, max_iter=1000, tol=1e-3,plot=False
    #     )
    #     end = timer()
    #     qp_time.append(end - start)

    #     start = timer()
    *_, residual_list_qp_smooth = primal_dual_qp_smooth(
        ppc,
        alpha=0.002, beta = 0.18, max_iter=1000, tol=1e-3,plot=False
    )
    #     end = timer()
    #     qp_smooth_time.append(end - start)

    #     # 预测器方法
    #     start = timer()
    #     *_, residual_list_lp_smooth_pred = primal_dual_lp_smooth_pred(
    #         ppc,
    #         predictor_linear,
    #         alpha=0.1, beta = 0.18, alpha_pg=0.15, max_iter=1000, tol=1e-3,plot=False
    #     )
    #     end = timer()
    #     lp_smooth_pred_time.append(end - start)

    #     start = timer()
    #     *_, residual_list_qp_smooth_pred = primal_dual_qp_smooth_pred(
    #         ppc,
    #         predictor_optimal, predictor_lambda, predictor_mu,
    #         alpha=0.002, beta = 0.18, max_iter=1000, tol=1e-3,plot=False
    #     )
    #     end = timer()
    #     qp_smooth_pred_time.append(end - start)

    #     _, time_gurobi_lp = solve_dcopf(ppc)
    #     gurobi_lp_time.append(time_gurobi_lp)
    #     _, time_gurobi_qp = solve_dcopf(ppc, type="poly")
    #     gurobi_qp_time.append(time_gurobi_qp)

    # print("LP time:", sum(lp_time) / len(lp_time))
    # print("QP time:", sum(qp_time) / len(qp_time))
    # print("LP Smooth time:", sum(lp_smooth_time) / len(lp_smooth_time))
    # print("QP Smooth time:", sum(qp_smooth_time) / len(qp_smooth_time))
    # print("LP Smooth Pred time:", sum(lp_smooth_pred_time) / len(lp_smooth_pred_time))
    # print("QP Smooth Pred time:", sum(qp_smooth_pred_time) / len(qp_smooth_pred_time))
    # print("Gurobi LP time:", sum(gurobi_lp_time) / len(gurobi_lp_time))
    # print("Gurobi QP time:", sum(gurobi_qp_time) / len(gurobi_qp_time))

    # # 绘制 LP 方法的残差对比图
    # plt.figure(figsize=(6, 4))
    # plt.plot(residual_list_lp_smooth, label="Primal-Dual LP + Smooth", linewidth=1, alpha=0.9)
    # plt.plot(residual_list_lp_smooth_pred, label="Primal-Dual LP + Smooth + Predict", linewidth=1, alpha=0.9)
    # plt.xlabel("Iteration")
    # plt.ylabel("Residual")
    # plt.title("Convergence of LP-based Methods")
    # plt.yscale("log")  # 残差通常对数尺度更清晰
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("convergence_lp_methods_pred.png", dpi=500)
    # plt.close()

    # # 绘制 QP 方法的残差对比图
    # plt.figure(figsize=(6, 4))
    # plt.plot(residual_list_qp_smooth, label="Primal-Dual QP", linewidth=1, alpha=0.9)
    # plt.plot(residual_list_qp_smooth_pred, label="Primal-Dual QP + Smooth + Predict",linewidth=1, alpha=0.9)
    # plt.xlabel("Iteration")
    # plt.ylabel("Residual")
    # plt.title("Convergence of QP-based Methods")
    # plt.yscale("log")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("convergence_qp_methods_pred.png", dpi=500)
    # plt.close()
    # ppc = pp.case118()
    # start = timer()
    # pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus = primal_dual_lp(
    #     ppc,
    #     alpha=0.1, max_iter=4000, tol=1e-3, alpha_pg=0.15
    # )
    # end = timer()
    # print("Time taken for primal dual refinement:", end - start)

    # # print("Primal solution:", pg)
    # # print("Lambda:", lamda)

    # start = timer()
    # primal_dual_lp_smooth_pred(
    #     ppc, predictor_linear,
    #     alpha=0.1, max_iter=4000, tol=1e-3, alpha_pg=0.15,beta = 0.18
    # )
    # end = timer()
    # print("Time taken for primal dual refinement:", end - start)

    
    # start = timer()
    # primal_dual_qp_smooth(
    #     ppc,
    #     alpha=0.003, max_iter=1000, tol=1e-3
    # )
    # end = timer()
    # print("Time taken for dual gradient ascent refinement0:", end - start)

    # start = timer()
    # pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus = primal_dual_qp_smooth_pred(
    #     ppc,
    #     predictor_optimal, predictor_lambda, predictor_mu,
    #     alpha=0.003, max_iter=1000, tol=1e-3,plot=False
    # )
    # end = timer()
    # print("Time taken for dual gradient ascent refinement:", end - start)
    # print("Primal solution:", pg)
    # print("Lambda:", lamda)
    # print("Mu1_plus:", mu1_plus)
    
    # print("Mu2_plus:", mu2_plus)
    # print("Mu2_minus:", mu2_minus)
    

    # start = timer()
    # solve_dcopf(ppc)
    # end = timer()
    # print("Time taken for LP Gurobi:", end - start)

    # start = timer()
    # result = solve_dcopf(ppc, type="poly")
    # end = timer()
    # print("Time taken for poly Gurobi:", end - start)


