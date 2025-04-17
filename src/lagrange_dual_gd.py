import numpy as np

def dual_gradient_ascent_refinement(
    c2, c1, Pd, H, Cg, pd, Fmax, gmin, gmax,
    alpha=1e-3, max_iter=500, tol=1e-4,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None
):
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
                    - Cg.T @ H.T @ mu1_plus + Cg.T @ H.T @ mu1_minus \
                    - mu2_plus + mu2_minus
        pg = C2_inv_diag @ grad_term

        # 投影到可行域
        pg = np.clip(pg, gmin, gmax)

        # 更新拉格朗日乘子（对偶梯度上升）
        flow = H @ (Cg @ pg - pd)

        lamda += alpha * (np.sum(pg) - Pd)
        mu1_plus += alpha * (flow - Fmax)
        mu1_minus -= alpha * (flow + Fmax)
        mu2_plus += alpha * (pg - gmax)
        mu2_minus -= alpha * (pg - gmin)

        # 投影到非负
        mu1_plus = np.maximum(mu1_plus, 0)
        mu1_minus = np.maximum(mu1_minus, 0)
        mu2_plus = np.maximum(mu2_plus, 0)
        mu2_minus = np.maximum(mu2_minus, 0)

        # 收敛性检测（你也可以增加 Lagrangian 残差或 KKT 条件）
        primal_res = np.abs(np.sum(pg) - Pd)
        if primal_res < tol:
            print(f"Converged at iter {it}")
            break

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus
