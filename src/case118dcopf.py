import pandas as pd
import numpy as np
import pypower.api as pypower
from pypower.idx_bus import PD
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime
from pypower.makePTDF import makePTDF
from pypower.idx_brch import RATE_A
import torch
from scipy import sparse
from timeit import default_timer as timer

START_TIME = "3/1/2024 5:00:00 AM"
END_TIME = "3/1/2025 4:00:00 AM"
def load_data(data, utc, data_type) -> np.ndarray:
    """
    从 csv 文件中加载数据
    :param data: csv 文件路径
    :param utc: UTC 时间戳
    :param data_type: 数据类型 ('load', 'wind', 'solar')
    :return: 数据数组
    """
    # data = pd.read_csv(file_path)
    data = data[data['datetime_beginning_utc'] == utc]
    if data_type == 'load':
        data = data[['mw']]
    elif data_type == 'wind':
        data = data[['wind_generation_mw']]
    elif data_type == 'solar':
        data = data[['solar_generation_mw']]
    data = data.values.flatten()
    #data = [val/1000 for val in data]

    # 如果数据为空，抛出错误
    if len(data) == 0:
        raise ValueError(f"No data found for {utc}")

    return data/400

def set_load(ppc, data_load, load_buses):
    """
    设置负载数据
    :param ppc: PyPower格式的电网数据
    :param data_load: 负载数据
    :param load_buses: 负载节点列表
    :return: 更新后的 ppc
    """
    # 确保 data_load 是 numpy 数组
    data_load = np.array(data_load)

    # 处理数据长度不匹配的情况
    if len(data_load) < len(load_buses):
        repeat_times = -(-len(load_buses) // len(data_load))  # 向上取整
        data_load = np.tile(data_load, repeat_times)[:len(load_buses)]
    elif len(data_load) > len(load_buses):
        data_load = data_load[:len(load_buses)]

    # 设置负载
    for i, bus in enumerate(load_buses):
        ppc['bus'][bus - 1, PD] = data_load[i]

    return ppc

def init_ppc():
    """
    设置 ppc 的参数
    :return: 初始化的 ppc
    """
    ppc = pypower.case118()
    # ppc['gencost'][:, 4] = 0  # 将发电机成本设置为 0
    ppc['branch'][:, RATE_A] = 200  # 线路潮流约束
    ppc['branch'][[95,103,106], RATE_A] = 50  # 线路潮流约束
    ppc['gen'][:20, 8] = 50  # 发电机最大出力
    ppc['gen'][47:53, 8] = 300  # 发电机最大出力

    return ppc

def set_case118(utc, load, wind, solar):
    """
    设置 118 节点系统的负载数据
    :param utc: UTC 时间戳
    :param load: 负载数据
    :param wind: 风电数据
    :param solar: 太阳能数据
    :return: PyPower格式的电网数据
    """
    data_load = load_data(load, utc, 'load')
    data_wind = load_data(wind, utc, 'wind')
    data_solar = load_data(solar, utc, 'solar')

    ppc = init_ppc()

    solar_buses = [15, 33, 49, 66, 80, 95]
    wind_buses = [50, 90, 100, 45, 68, 110]
    load_buses = [bus for bus in range(1, 119) if bus not in solar_buses + wind_buses + [68]]

    # 设置负载
    ppc = set_load(ppc, -data_wind, wind_buses)
    ppc = set_load(ppc, -data_solar, solar_buses)
    ppc = set_load(ppc, data_load, load_buses)

    return ppc

def PTDF(ppc):
    """
    计算 PTDF 矩阵
    :param ppc: PyPower格式的电网数据
    :return: PTDF 矩阵
    """
    ppc = pypower.ext2int(ppc)
    baseMVA = ppc['baseMVA']
    bus = ppc['bus']
    branch = ppc['branch']
    PTDF_full = pypower.makePTDF(baseMVA, bus, branch)
    return PTDF_full

def format_timestamp(ts):
    """
    将时间戳格式化为字符串
    :param ts: 时间戳
    :return: 格式化后的字符串
    """
    # 处理小时（转换为12小时制并去除前导零）
    hour_12 = ts.hour % 12
    hour_12 = 12 if hour_12 == 0 else hour_12  # 处理0小时的情况（如12 AM）
    am_pm = ts.strftime('%p')  # 获取AM/PM
    # 组合为字符串，月、日、小时直接转为整数去除前导零
    return f"{ts.month}/{ts.day}/{ts.year} {hour_12}:{ts.minute:02}:{ts.second:02} {am_pm}"

def format_timelist(start_time, end_time):
    # 生成start_time end_time之间每隔 1 小时的时间戳
    timestamps = pd.date_range(start_time, end_time, freq='h')
    formatted_timestamps = [format_timestamp(ts) for ts in timestamps]
    return formatted_timestamps

def makeCg(ppc):
    """
    计算发电机连接矩阵Cg
    :param ppc: PyPower格式的电网数据
    :return: 发电机连接矩阵Cg
    """
    ppc = pypower.ext2int(ppc)
    nb = ppc['bus'].shape[0]  # 节点数
    ng = ppc['gen'].shape[0]  # 发电机数

    cg = sparse.coo_matrix((np.ones(ng), (ppc['gen'][:, 0].astype(int), range(ng))), shape=(nb, ng), dtype=int).tocsr()
    
    return cg

def get_params(ppc):
    """
    提取优化问题所需参数
    :param ppc: PyPower格式的电网数据
    :return: 发电机成本系数、线路最大功率流约束、发电机最小功率、发电机最大功率
    """
    cg = makeCg(ppc)
    h = PTDF(ppc)

    ppc = pypower.ext2int(ppc)

    c1 = ppc["gencost"][:, 5]  # 线性成本系数
    c2 = ppc["gencost"][:, 4]  # 二次成本系数
    F_max = ppc["branch"][:, 5]  # 线路最大功率流约束
    g_max = ppc["gen"][:, 8]  # 发电机最大功率
    g_min = ppc["gen"][:, 9]  # 发电机最小功率
    return c2, c1, cg, h, F_max, g_min, g_max

def solve_dcopf(ppc,type='linear'):
    """
    使用Gurobi求解一次/二次目标函数的DCOPF问题，返回包含所有拉格朗日乘子的结果
    
    输入:
        - ppc (dict): PyPower格式的电网数据
        - type (str): 目标函数类型，'linear'或'poly'，默认为'linear'
        
    输出:
        result (dict): 包含以下键值对:
            - 'Pg_opt': 发电机最优出力 (MW)
            - 'lambda_power_balance': 功率平衡约束的乘子 ($/MWh)
            - 'lambda_line_max': 线路正向潮流约束的乘子 ($/MWh)
            - 'lambda_line_min': 线路反向潮流约束的乘子 ($/MWh)
            - 'lambda_pg_min': 发电机出力下限约束的乘子 ($/MWh)
            - 'lambda_pg_max': 发电机出力上限约束的乘子 ($/MWh)
    """
    # 数据预处理
    # 如果type不为linear或poly，抛出错误
    if type not in ['linear', 'poly']:
        raise ValueError("type must be 'linear' or 'poly'")
    
    ppc = pypower.ext2int(ppc)
    baseMVA = ppc['baseMVA']
    bus = ppc['bus']
    gen = ppc['gen']
    branch = ppc['branch']
    gencost = ppc['gencost']
    
    # 计算PTDF矩阵
    ptdf = makePTDF(baseMVA, bus, branch)
    
    # 发电机参数
    gen_bus = gen[:, 0].astype(int)
    ng = len(gen_bus)

    c1 = gencost[:, 5]  # 线性成本系数
    c2 = gencost[:, 4]  # 二次成本系数
    Pg_min = gen[:, 9] / baseMVA  # 转换为标幺值
    Pg_max = gen[:, 8] / baseMVA
    
    # 负荷参数
    Pd = bus[:, 2] / baseMVA
    
    # 创建模型
    start = timer()
    model = gp.Model("DCOPF")
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)
    model.setParam('LogToConsole', 0)
    
    # 定义变量（不直接设置lb/ub，而是通过显式约束）
    Pg = model.addVars(ng, name="Pg")  # 自由变量
    
    # 目标函数
    if type == 'linear':
        model.setObjective(gp.quicksum(c1[i] * Pg[i] * baseMVA for i in range(ng)), GRB.MINIMIZE)
    elif type == 'poly':
        model.setObjective(gp.quicksum(c2[i] * Pg[i]**2 * baseMVA * baseMVA + c1[i] * Pg[i] * baseMVA for i in range(ng)), GRB.MINIMIZE)
    
    # 功率平衡约束
    power_balance = model.addConstr(
        sum(Pg[i] for i in range(ng)) == sum(Pd),
        name="PowerBalance"
    )
    
    # 显式添加发电机出力上下限约束
    pg_min_constr = []
    pg_max_constr = []
    for i in range(ng):
        c_min = model.addConstr(Pg[i] >= Pg_min[i], name=f"PgMin_{i}")
        c_max = model.addConstr(Pg[i] <= Pg_max[i], name=f"PgMax_{i}")
        pg_min_constr.append(c_min)
        pg_max_constr.append(c_max)
    
    # 预计算节点注入表达式
    node_injection = {b: -Pd[b] for b in range(bus.shape[0])}
    for i in range(ng):
        b = gen_bus[i]
        node_injection[b] += Pg[i]
    
    # 线路潮流约束
    line_max_constr = []
    line_min_constr = []
    for l in range(branch.shape[0]):
        line_limit = branch[l, 5] / baseMVA
        flow = gp.quicksum(ptdf[l, b] * node_injection[b] for b in range(bus.shape[0]))
        c_max = model.addConstr(flow <= line_limit, name=f"LineMax_{l}")
        c_min = model.addConstr(flow >= -line_limit, name=f"LineMin_{l}")
        line_max_constr.append(c_max)
        line_min_constr.append(c_min)
    
    # 求解模型
    model.optimize()
    end = timer()
    solve_time = end - start
    # print(f"求解时间: {end - start:.5f}秒")
    # 结果处理
    result = {}
    if model.status == GRB.OPTIMAL:
        # 发电机最优出力 (MW)
        result['Pg_opt'] = [Pg[i].X * baseMVA for i in range(ng)]
        
        # 拉格朗日乘子
        result['lambda_power_balance'] = power_balance.Pi / baseMVA  # $/MWh
        # 这里可能有错误，不应该是乘以 baseMVA，而是除以 baseMVA
        # 线路约束乘子
        result['lambda_line_max'] = [c.Pi / baseMVA for c in line_max_constr]
        result['lambda_line_min'] = [c.Pi / baseMVA for c in line_min_constr]
        
        # 发电机出力上下限约束乘子
        result['lambda_pg_min'] = [c.Pi / baseMVA for c in pg_min_constr]
        result['lambda_pg_max'] = [c.Pi / baseMVA for c in pg_max_constr]
    else:
        raise RuntimeError(f"优化失败，状态码: {model.status}")
    
    return result, solve_time


# def concat_torch(data_dict):
#     """
#     将包含多个 key 的字典中的值拼接成一个张量。
#     期望每个 key 对应一个二维数组（例如 shape: [N, d_i]），按列拼接。
#     """
#     keys = ('Pg_opt', 'lambda_power_balance', 'lambda_line_max',
#             'lambda_line_min', 'lambda_pg_min', 'lambda_pg_max')
    
#     tensors = []
#     for key in keys:
#         if key not in data_dict:
#             raise KeyError(f"Missing key: {key} in data_dict.")
        
#         val = data_dict[key]
#         tensor = torch.tensor(val, dtype=torch.float32)

#         # 如果是 0 维（标量），升维为 (1,)
#         if tensor.ndim == 0:
#             tensor = tensor.unsqueeze(0)

#         # 如果是一维向量，升维为 (1, d)
#         if tensor.ndim == 1:
#             tensor = tensor.unsqueeze(0)

#         tensors.append(tensor)

#     return torch.cat(tensors, dim=1)  # shape: (1, total_dim)



# if __name__ == '__main__':
#     load_file_path = 'load_data/hrl_load_metered_20242025.csv'
#     wind_file_path = 'load_data/wind_gen_20242025.csv'
#     solar_file_path = 'load_data/solar_gen_20242025.csv'
#     data_load = pd.read_csv(load_file_path)
#     data_wind = pd.read_csv(wind_file_path)
#     data_solar = pd.read_csv(solar_file_path)
#     # 起始时间
#     start_time = datetime.strptime("3/1/2024 5:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
#     # end_time = datetime.strptime("3/1/2024 7:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
#     end_time = datetime.strptime("3/1/2025 4:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
#     # 生成start_time end_time之间每隔 1 小时的时间戳
#     timestamps = pd.date_range(start_time, end_time, freq='h')
#     formatted_timestamps = [format_timestamp(ts) for ts in timestamps]
#     data = torch.tensor([])
#     # Pd_total = torch.tensor([])
#     i = 0
    # for time in formatted_timestamps:
    #     ppc = set_case118(time, data_load, data_wind, data_solar)
    #     # Pd = torch.tensor(ppc['bus'][:, 2], dtype=torch.float32)
    #     # Pd_total = torch.cat((Pd_total, Pd), dim=0) if Pd_total.numel() > 0 else Pd
    #     result = solve_dcopf(ppc, type='poly')
    #     # 将结果转换为张量
    #     result_tensor = concat_torch(result)
    #     # 将结果添加到数据集中
    #     data = torch.cat((data, result_tensor), dim=0) if data.numel() > 0 else result_tensor
    #     i += 1
    #     print(f"------------------第{i}次计算完成----------------------")
    
#     # 将data存成pt文件
#     torch.save(data, 'train_data/poly_result.pt')
#     # 将Pd_total存成pt文件
#     # torch.save(Pd_total, 'train_data/Pd.pt')

def concat_flat(result_dict):
    """
    将结果字典直接展开为一个一维 list(不转成 tensor)
    """
    keys = ('Pg_opt', 'lambda_power_balance', 'lambda_line_max',
            'lambda_line_min', 'lambda_pg_min', 'lambda_pg_max')
    
    flat_result = []
    for key in keys:
        val = result_dict[key]
        if np.isscalar(val):
            flat_result.append(float(val))
        else:
            flat_result.extend(val)
    return flat_result

if __name__ == '__main__':
    # load_file_path = 'load_data/hrl_load_metered_20242025.csv'
    # wind_file_path = 'load_data/wind_gen_20242025.csv'
    # solar_file_path = 'load_data/solar_gen_20242025.csv'
    load_file_path = 'load_data/hrl_load_metered_20232024.csv'
    wind_file_path = 'load_data/wind_gen_20232024.csv'
    solar_file_path = 'load_data/solar_gen_20232024.csv'
    data_load = pd.read_csv(load_file_path)
    data_wind = pd.read_csv(wind_file_path)
    data_solar = pd.read_csv(solar_file_path)

    start_time = datetime.strptime("1/1/2023 5:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
    # end_time = datetime.strptime("1/1/2023 7:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
    end_time = datetime.strptime("1/2/2024 4:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
    timestamps = pd.date_range(start_time, end_time, freq='h')
    formatted_timestamps = [format_timestamp(ts) for ts in timestamps]

    all_results = []  # 用于收集每个样本的一维 list
    Pd_list = []  # 用于收集每个样本的 Pd 数据
    for i, time in enumerate(formatted_timestamps):
        ppc = set_case118(time, data_load, data_wind, data_solar)
        Pd_list.append(ppc['bus'][:, 2])
        result = solve_dcopf(ppc, type='poly')
        flat = concat_flat(result)  # 返回 list
        all_results.append(flat)

        print(f"------------------第{i+1}次计算完成----------------------")

    # 一次性转为 tensor
    data_tensor = torch.tensor(all_results, dtype=torch.float32)  # shape = [N, 535]
    Pd_tensor = torch.tensor(Pd_list, dtype=torch.float32)  # shape = [N, 118]

    # 保存
    torch.save(data_tensor, 'train_data/poly_result_validate.pt')
    torch.save(Pd_tensor, 'train_data/Pd_validate.pt')
    print(f"✅ 所有样本保存完成，共计 {len(all_results)} 条，保存至 train_data/poly_resul_validatet.pt")