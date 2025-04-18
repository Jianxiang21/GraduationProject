import numpy as np
from case118dcopf import get_params,init_ppc, solve_dcopf
import pypower.api as pp
import torch
import ResNet_poly
import ResNet_poly_lambda
from timeit import default_timer as timer


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

def dual_gradient_ascent_refinement_improved(
    ppc, predictor_optimal, predictor_lambda,
    init_pg=None, init_lambda=None,
    init_mu1_plus=None, init_mu1_minus=None,
    init_mu2_plus=None, init_mu2_minus=None,
    alpha=1e-3, max_iter=500, tol=1e-4,
):
    c2, c1, Cg, h, Fmax, gmin, gmax = get_params(ppc)
    p_load = ppc["bus"][:, 2]  # 负荷功率
    Pd_total = np.sum(p_load)  # 总负荷功率
    print("Pd_total:", Pd_total)
    m = len(c2)
    l = len(Fmax)

    # 初值设定（支持用户提供预测初值）
    # load_tensor = torch.tensor(p_load, dtype=torch.float32).unsqueeze(0)
    pg = predictor_optimal.predict(np.array([p_load])).squeeze(0)
    lamda = predictor_lambda.predict(np.array([p_load])).squeeze(0) / 10000
    print(lamda)
    # lamda = -25.159929691274353
    print("pg:", sum(pg))
    
    # lamda = prediction[m]
    # mu1_plus = prediction[m+1:m+1+l]
    # mu1_minus = prediction[m+1+l:m+1+2*l]
    # mu2_plus = prediction[m+1+2*l:m+1+2*l+m]
    # mu2_minus = prediction[m+1+2*l+m:]

    # lamda = init_lambda if init_lambda is not None else 0.0
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
        print("Final primal residual:", primal_res)

    return pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus


if __name__ == "__main__":
    ppc = setup_ppc(100)
    # ppc118 = pp.case118()
    # pg, lamda, mu1_plus, mu1_minus, mu2_plus, mu2_minus = 
    dual_gradient_ascent_refinement(
        ppc,
        alpha=0.002, max_iter=1000, tol=1e-2,
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
        alpha=0.002, max_iter=1000, tol=1e-2,
    )
    end = timer()
    print("Time taken for dual gradient ascent refinement:", end - start)
    # print("pg:", pg)
    # print("lamda:", lamda)
    # print("mu1_plus:", mu1_plus)
    # print("mu1_minus:", mu1_minus)
    # print("mu2_plus:", mu2_plus)
    # print("mu2_minus:", mu2_minus)

    start = timer()
    solve_dcopf(ppc, type = 'poly')
    end = timer()
    print("Time taken for Gurobi solver:", end - start)



    # result = torch.load("train_data/poly_result.pt")
    # result100 = result[100, :54].numpy().tolist()
    # print("result100:", result100)

    # result = torch.load("train_data/poly_result.pt")
    # result0 = result[0, :]
    # print("result0:", result0)
