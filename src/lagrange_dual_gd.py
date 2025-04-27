import numpy as np
from case118dcopf import get_params,init_ppc, solve_dcopf
import pypower.api as pp
import torch
import ResNet_poly
import ResNet_poly_lambda
from timeit import default_timer as timer
import matplotlib.pyplot as plt



POLY_MODEL_PATH = "model/poly_resnet_model_50epoch54output.pt"
POLY_MODEL_PATH_LAMBDA = "model/poly_resnet_model_500epoch_lambda1.pt"
POLY_PROCESS_FILE_LAMBDA = "model_data/poly_resnet_preprocess_lambda1.npz"
POLY_PROCESS_FILE = "model_data/poly_resnet_preprocess_54.npz"
PD_FILE = "train_data/Pd_torch.pt"

predictor_optimal = ResNet_poly.ResNetPredictor(
    model_path=POLY_MODEL_PATH,
    preprocess_path=POLY_PROCESS_FILE
)
predictor_lambda = ResNet_poly_lambda.ResNetPredictor(
    model_path=POLY_MODEL_PATH_LAMBDA,
    preprocess_path=POLY_PROCESS_FILE_LAMBDA
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


def dual_gradient_ascent_refinement(
    ppc,
    alpha=1e-3, max_iter=500, tol=1e-4,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None
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

    for it in range(max_iter):
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

        # 收敛性检测（你也可以增加 Lagrangian 残差或 KKT 条件）
        primal_res = np.abs(np.sum(pg) - Pd_total)
        if primal_res < tol:
            print(f"Converged at iter {it}")
            break
    if it == max_iter - 1:
        # 如果没有收敛，输出最后的结果
        print("Warning: Maximum iterations reached without convergence.")

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus

# 以下是version 0
# def dual_gradient_ascent_refinement_improved(
#     ppc, predictor_optimal, predictor_lambda,
#     init_pg=None, init_lambda=None,
#     init_mu1_plus=None, init_mu1_minus=None,
#     init_mu2_plus=None, init_mu2_minus=None,
#     alpha=1e-3, max_iter=500, tol=1e-4,
# ):
#     c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
#     p_load = ppc["bus"][:, 2]  # 负荷功率
#     Pd_total = np.sum(p_load)  # 总负荷功率
#     print("Pd_total:", Pd_total)
#     m = len(c2)
#     l = len(Fmax)

#     # 初值设定（支持用户提供预测初值）
#     # load_tensor = torch.tensor(p_load, dtype=torch.float32).unsqueeze(0)
#     pg = predictor_optimal.predict(np.array([p_load])).squeeze(0)
#     lamda = predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000
#     print(lamda)
#     # lamda = -25.159929691274353
#     print("pg:", sum(pg))
    
#     # lamda = prediction[m]
#     # mu1_plus = prediction[m+1:m+1+l]
#     # mu1_minus = prediction[m+1+l:m+1+2*l]
#     # mu2_plus = prediction[m+1+2*l:m+1+2*l+m]
#     # mu2_minus = prediction[m+1+2*l+m:]

#     # lamda = init_lambda if init_lambda is not None else 0.0
#     mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
#     mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
#     mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
#     mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

#     C2_inv_diag = np.diag(1 / (2 * c2))  # (1/2c2) inverse diag matrix

#     for it in range(max_iter):
#         # 更新 pg（用解析公式）
#         grad_term = -c1 - lamda * np.ones(m) \
#                     - Cg.T @ h.T @ mu1_plus + Cg.T @ h.T @ mu1_minus \
#                     - mu2_plus + mu2_minus
#         pg = C2_inv_diag @ grad_term

#         # 投影到可行域
#         pg = np.clip(pg, gmin, gmax)

#         # 更新拉格朗日乘子（对偶梯度上升）
#         flow = h @ (Cg @ pg - p_load)

#         lamda += alpha * (np.sum(pg) - Pd_total)
#         mu1_plus += alpha * (flow - Fmax)
#         mu1_minus -= alpha * (flow + Fmax)
#         mu2_plus += alpha * (pg - gmax)
#         mu2_minus -= alpha * (pg - gmin)

#         # 投影到非负
#         mu1_plus = np.maximum(mu1_plus, 0)
#         mu1_minus = np.maximum(mu1_minus, 0)
#         mu2_plus = np.maximum(mu2_plus, 0)
#         mu2_minus = np.maximum(mu2_minus, 0)

#         # print(f"----------Iteration {it}-----------")

#         # 收敛性检测（你也可以增加 Lagrangian 残差或 KKT 条件）
#         primal_res = np.abs(np.sum(pg) - Pd_total)
#         if primal_res < tol:
#             print(f"Converged at iter {it}")
#             break
#     if it == max_iter - 1:
#         # 如果没有收敛，输出最后的结果
#         print("Warning: Maximum iterations reached without convergence.")
#         print("Final primal residual:", primal_res)

#     return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus
# 以下是version 1
# import numpy as np

def dual_gradient_ascent_refinement_improved(
    ppc, predictor_optimal, predictor_lambda,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,
    alpha=1e-3, beta=0.25,  # 新增beta参数
    max_iter=500, tol=1e-4,
):
    # 解析问题参数
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # 负荷功率
    Pd_total = np.sum(p_load)  # 总负荷功率
    # print("Pd_total:", Pd_total)
    m = len(c2)
    l = len(Fmax)

    # 初值设定（基于预测器）
    pg_pred = predictor_optimal.predict(np.array([p_load])).squeeze(0)
    lambda_pred = - predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000

    print("Predicted lambda:", lambda_pred)
    # print("Predicted sum(pg):", sum(pg_pred))
    
    # 使用预测初值或默认值
    pg = init_pg if init_pg is not None else pg_pred
    lamda = init_lambda if init_lambda is not None else lambda_pred

    mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
    mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
    mu1_minus[95] = 29.59226271
    mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
    mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

    # 预计算逆矩阵
    C2_inv_diag = np.diag(1 / (2 * c2))  # (1/2c2) inverse diag matrix

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
        if primal_res < tol:
            print(f"Converged at iteration {it}")
            break

    if it == max_iter - 1:
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_res)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus


def dual_gradient_ascent_refinement_advanced(
    ppc, predictor_optimal, predictor_lambda,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,
    alpha_init=1e-3, beta=0.25, 
    max_iter=500, tol=1e-4,
    early_tol=200,  # 新增：Early Correction容忍度
    alpha_decay=0.95,  # 新增：alpha每次衰减比例
):
    # 解析问题参数
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]
    Pd_total = np.sum(p_load)
    # print("Pd_total:", Pd_total)
    m = len(c2)
    l = len(Fmax)

    # 初值设定
    pg_pred = predictor_optimal.predict(np.array([p_load])).squeeze(0)
    lambda_pred = predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000

    pg = init_pg if init_pg is not None else pg_pred
    lamda = init_lambda if init_lambda is not None else lambda_pred

    mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
    mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
    mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
    mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

    C2_inv_diag = np.diag(1 / (2 * c2))
    alpha = alpha_init  # 初始化alpha

    for it in range(max_iter):
        # 计算当前残差
        flow = h @ (Cg @ pg - p_load)
        primal_res = np.abs(np.sum(pg) - Pd_total)
        constraint_violation = np.maximum(0, np.abs(flow) - Fmax).sum() \
                             + np.maximum(0, pg - gmax).sum() \
                             + np.maximum(0, gmin - pg).sum()

        # --- Step 1: 决定是否更新pg ---
        if primal_res > early_tol or constraint_violation < early_tol:
            grad_term = -c1 - lamda * np.ones(m) \
                        - Cg.T @ h.T @ mu1_plus + Cg.T @ h.T @ mu1_minus \
                        - mu2_plus + mu2_minus
            pg_new = C2_inv_diag @ grad_term
            pg = (1 - beta) * pg + beta * pg_new
            pg = np.clip(pg, gmin, gmax)

        # --- Step 2: 更新拉格朗日乘子 ---
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

        # --- Step 3: 动态调整alpha ---
        alpha *= alpha_decay

        # --- Step 4: 收敛检测 ---
        if primal_res < tol:
            print(f"Converged at iteration {it}")
            break

    if it == max_iter - 1:
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_res)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus

import numpy as np

def dual_gradient_ascent_refinement_final(
    ppc, predictor_optimal, predictor_lambda,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,
    alpha_init=1e-3, primal_correction_step=1e-2,
    beta_pg=0.25,
    max_iter=1000, tol=1e-4,
    alpha_decay=0.98,
):
    # 参数解析
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]
    Pd_total = np.sum(p_load)
    print("Pd_total:", Pd_total)
    m = len(c2)
    l = len(Fmax)

    # 初值设定
    pg_pred = predictor_optimal.predict(np.array([p_load])).squeeze(0)
    lambda_pred = predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000

    pg = init_pg if init_pg is not None else pg_pred
    lamda = init_lambda if init_lambda is not None else lambda_pred

    mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
    mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
    mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
    mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

    alpha = alpha_init

    for it in range(max_iter):
        # Step 1: 计算残差
        flow = h @ (Cg @ pg - p_load)
        power_balance = np.sum(pg) - Pd_total

        # Step 2: 更新拉格朗日乘子
        lamda += alpha * power_balance
        mu1_plus += alpha * (flow - Fmax)
        mu1_minus -= alpha * (flow + Fmax)
        mu2_plus += alpha * (pg - gmax)
        mu2_minus -= alpha * (pg - gmin)

        # 对偶变量投影到非负
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        # Step 3: 细致地修正 pg
        grad_pg = -lamda * np.ones(m) \
                  - Cg.T @ h.T @ (mu1_plus - mu1_minus) \
                  - (mu2_plus - mu2_minus)
        pg_correction = primal_correction_step * grad_pg  # 小步更新
        pg = pg + beta_pg * pg_correction  # beta_pg控制更平滑
        pg = np.clip(pg, gmin, gmax)  # 保证可行性

        # Step 4: 动态调整 alpha
        alpha *= alpha_decay

        # Step 5: 检查收敛
        primal_residual = np.abs(power_balance)
        if primal_residual < tol:
            print(f"Converged at iteration {it}")
            break

    if it == max_iter - 1:
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_residual)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus

# import numpy as np

def dual_gradient_ascent_with_beta_and_logging(
    ppc, predictor_optimal, predictor_lambda,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,
    alpha=1e-3, beta=0.25, max_iter=500, tol=1e-4,
    plot_residual=True
):
    # 参数提取
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]
    Pd_total = np.sum(p_load)
    m = len(c2)
    l = len(Fmax)

    # 初始值（支持预测器提供）
    pg = predictor_optimal.predict(np.array([p_load])).squeeze(0)
    lamda = predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000

    mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
    mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
    mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
    mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

    C2_inv_diag = np.diag(1 / (2 * c2))

    # 用于记录残差
    primal_residuals = []
    # (np.abs(np.sum(pg) - Pd_total))
    primal_residuals.append(np.abs(np.sum(pg) - Pd_total))
    print("Initial primal residual:", primal_residuals[0])

    for it in range(max_iter):
        # 更新 pg（带 beta 混合）
        grad_term = -c1 - lamda * np.ones(m) \
                    - Cg.T @ h.T @ mu1_plus + Cg.T @ h.T @ mu1_minus \
                    - mu2_plus + mu2_minus
        pg_new = C2_inv_diag @ grad_term
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
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        # 记录当前主问题残差
        primal_res = np.abs(np.sum(pg) - Pd_total)
        primal_residuals.append(primal_res)

        # 收敛性检测
        if primal_res < tol:
            print(f"Converged at iteration {it}")
            break

    if it == max_iter - 1:
        print("Warning: Maximum iterations reached without convergence.")
        print(f"Final primal residual: {primal_res:.4e}")

    # 绘制残差曲线
    if plot_residual:
        plt.figure(figsize=(8,5))
        plt.plot(primal_residuals, label='Primal Residual')
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('Primal Residual over Iterations')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus

def dual_gradient_ascent_refinement_stable(
    ppc, predictor_optimal, predictor_lambda,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,
    alpha=2e-3, beta=0.05, max_iter=1000, tol=1e-4,
    warm_dual_steps=10, clip_ratio=0.01,
):
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # Load power
    Pd_total = np.sum(p_load)
    m = len(c2)
    l = len(Fmax)

    # Initial values
    pg = predictor_optimal.predict(np.array([p_load])).squeeze(0)
    lamda = predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000

    mu1_plus = init_mu1_plus if init_mu1_plus is not None else np.zeros(l)
    mu1_minus = init_mu1_minus if init_mu1_minus is not None else np.zeros(l)
    mu2_plus = init_mu2_plus if init_mu2_plus is not None else np.zeros(m)
    mu2_minus = init_mu2_minus if init_mu2_minus is not None else np.zeros(m)

    C2_inv_diag = np.diag(1 / (2 * c2))

    residual_list = []

    for it in range(max_iter):
        # Compute pg update direction
        grad_term = -c1 - lamda * np.ones(m) \
                    - Cg.T @ h.T @ mu1_plus + Cg.T @ h.T @ mu1_minus \
                    - mu2_plus + mu2_minus
        pg_new = C2_inv_diag @ grad_term

        # Project onto box constraints
        pg_new = np.clip(pg_new, gmin, gmax)

        if it >= warm_dual_steps:
            # Gradient clipping on pg update
            pg_update = pg_new - pg
            pg_update_norm = np.linalg.norm(pg_update)
            max_update_norm = clip_ratio * np.linalg.norm(pg)

            if pg_update_norm > max_update_norm and pg_update_norm > 0:
                pg_update = pg_update * (max_update_norm / pg_update_norm)

            pg = pg + beta * pg_update
        else:
            # During warm-up, keep pg fixed
            pass

        # Update dual variables (with damping)
        flow = h @ (Cg @ pg - p_load)

        lambda_update = np.sum(pg) - Pd_total
        lamda += 0.1 * alpha * lambda_update  # Damped lambda update

        mu1_plus += 0.1 * alpha * (flow - Fmax)
        mu1_minus -= 0.1 * alpha * (flow + Fmax)
        mu2_plus += 0.1 * alpha * (pg - gmax)
        mu2_minus -= 0.1 * alpha * (pg - gmin)

        # Projection onto non-negative orthant
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        # Primal residual
        primal_res = np.abs(np.sum(pg) - Pd_total)
        residual_list.append(primal_res)

        if primal_res < tol:
            print(f"Converged at iteration {it}")
            break

    if it == max_iter - 1:
        print("Warning: Maximum iterations reached without convergence.")
        print("Final primal residual:", primal_res)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus, residual_list

# 注意：get_params 函数需要您根据您的 ppc 数据结构定义；
# predictor_optimal 和 predictor_lambda 也是您已有的预测模型接口。



if __name__ == "__main__":
    ppc = setup_ppc(100)
    # ppc118 = pp.case118()
    # pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus = 
    dual_gradient_ascent_refinement(
        ppc,
        alpha=0.002, max_iter=1000, tol=1e-4,
    )

    # print("pg:", pg)
    # print("lamda:", lamda)
    # print("mu1_plus:", mu1_plus)
    # print("mu1_minus:", mu1_minus)
    # print("mu2_plus:", mu2_plus)
    # print("mu2_minus:", mu2_minus)

    start = timer()
    pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus = dual_gradient_ascent_refinement_improved(
        ppc,
        predictor_optimal, predictor_lambda,
        alpha=0.002, max_iter=1000, tol=1e-4,
    )
    end = timer()
    print("Time taken for dual gradient ascent refinement:", end - start)
    print("pg:", pg)
    print("lamda:", lamda)
    print("mu1_plus:", mu1_plus)
    print("mu1_minus:", mu1_minus)
    print("mu2_plus:", mu2_plus)
    print("mu2_minus:", mu2_minus)


    # pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus, residual_list = dual_gradient_ascent_refinement_stable(
    #     ppc,
    #     predictor_optimal, predictor_lambda
    # )
    # print("pg:", pg)
    # print("lamda:", lamda)
    # print("mu1_plus:", mu1_plus)
    # print("mu1_minus:", mu1_minus)
    # print("mu2_plus:", mu2_plus)
    # print("mu2_minus:", mu2_minus)

    # start = timer()
    # solve_dcopf(ppc, type = 'poly')
    # end = timer()
    # print("Time taken for Gurobi solver:", end - start)



    # result = torch.load("train_data/poly_result.pt")
    # result100 = result[100, :54].numpy().tolist()
    # print("result100:", result100)

    # result = torch.load("train_data/poly_result.pt")
    # result0 = result[0, :]
    # print("result0:", result0)
