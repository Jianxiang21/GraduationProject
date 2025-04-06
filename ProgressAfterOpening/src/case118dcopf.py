import pandas as pd
import numpy as np
import pypower.api as pypower
from pypower.idx_bus import PD
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csr_matrix
from datetime import datetime, timedelta
import json
from pypower.makePTDF import makePTDF



# from pypower.idx_gen import *
from pypower.idx_brch import RATE_A

START_TIME = "3/1/2024 5:00:00 AM"
END_TIME = "3/1/2025 4:00:00 AM"
def load_data(data, utc, data_type) -> np.ndarray:
    """从 csv 文件中加载数据"""
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
    """设置负载数据"""
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

def set_case118(utc, load, wind, solar):
    """设置 118 节点系统的负载数据"""
    data_load = load_data(load, utc, 'load')
    data_wind = load_data(wind, utc, 'wind')
    data_solar = load_data(solar, utc, 'solar')

    ppc = pypower.case118()
    # ppc['gencost'][:, 4] = 0  # 将发电机成本设置为 0
    ppc['branch'][:, RATE_A] = 200  # 线路潮流约束
    ppc['branch'][[95,103,106], RATE_A] = 50  # 线路潮流约束
    ppc['gen'][:20, 8] = 50  # 发电机最大出力
    ppc['gen'][47:53, 8] = 300  # 发电机最大出力

    solar_buses = [15, 33, 49, 66, 80, 95]
    wind_buses = [50, 90, 100, 45, 68, 110]
    load_buses = [bus for bus in range(1, 119) if bus not in solar_buses + wind_buses + [68]]

    # 设置负载
    ppc = set_load(ppc, -data_wind, wind_buses)
    ppc = set_load(ppc, -data_solar, solar_buses)
    ppc = set_load(ppc, data_load, load_buses)

    # 计算 DCOPF
    #result = pypower.rundcopf(ppc)

    return ppc

def dcopf_case118(ppc):
    """计算 118 节点系统的 DCOPF"""
    # 计算 DCOPF
    ppopt = pypower.ppoption(VERBOSE=1, OUT_ALL=1)
    # ppopt = pypower.ppoption(VERBOSE=1, OUT_ALL=1)
    
    # 设置求解器为 Gurobi
    ppopt['OPF_ALG_DC'] = 200  # 使用 Gurobi 求解器
    # ppopt['OPF_VOLTAGE_LIMITS'] = 1  # 启用电压限值
    # ppopt['OPF_FLOW_LIM'] = 1        # 启用流量限值
    # ppopt['OPF_MAX_IT'] = 100        # 增加最大迭代次数

    # 运行 DCOPF
    result = pypower.rundcopf(ppc, ppopt)

    Pg = result['var']['val']['Pg'].tolist()# p.u.
    Va = result['var']['val']['Va'].tolist()
    mu_Pg_lb = result['var']['mu']['l']['Pg'].tolist()
    mu_Pg_ub = result['var']['mu']['u']['Pg'].tolist()
    mu_Va_lb = result['var']['mu']['l']['Va'].tolist()
    mu_Va_ub = result['var']['mu']['u']['Va'].tolist()
    mu_Pmis_lb = result['lin']['mu']['l']['Pmis'].tolist()
    mu_Pmis_ub = result['lin']['mu']['u']['Pmis'].tolist()
    mu_Pf_lb = result['lin']['mu']['l']['Pf'].tolist()
    mu_Pf_ub = result['lin']['mu']['u']['Pf'].tolist()
    mu_Pt_lb = result['lin']['mu']['l']['Pt'].tolist()
    mu_Pt_ub = result['lin']['mu']['u']['Pt'].tolist()
    result_dict = {}
    result_dict['Pg'] = Pg
    result_dict['Va'] = Va
    result_dict['mu_Pg_lb'] = mu_Pg_lb
    result_dict['mu_Pg_ub'] = mu_Pg_ub
    result_dict['mu_Va_lb'] = mu_Va_lb
    result_dict['mu_Va_ub'] = mu_Va_ub
    result_dict['mu_Pmis_lb'] = mu_Pmis_lb
    result_dict['mu_Pmis_ub'] = mu_Pmis_ub
    result_dict['mu_Pf_lb'] = mu_Pf_lb
    result_dict['mu_Pf_ub'] = mu_Pf_ub
    result_dict['mu_Pt_lb'] = mu_Pt_lb
    result_dict['mu_Pt_ub'] = mu_Pt_ub
    return result_dict,result

def PTDF(ppc):
    """计算 PTDF 矩阵"""
    ppc = pypower.ext2int(ppc)
    baseMVA = ppc['baseMVA']
    bus = ppc['bus']
    branch = ppc['branch']
    # gen = ppc['gen']
    
    PTDF_full = pypower.makePTDF(baseMVA, bus, branch)
    # gen_buses = np.unique(gen[:, 0]).astype(int)  # 只取发电机母线 ID

    # # 5. 提取仅针对发电机母线的 PTDF 子矩阵 (branch_num × gen_bus_num)
    # PTDF_gen = PTDF_full[:, gen_buses].T

    return PTDF_full

def get_optimization_params(ppc):
    """提取优化问题所需参数"""
    c = ppc["gencost"][:, 5]  # 线性成本系数
    F_max = ppc["branch"][:, 5]  # 线路最大功率流约束
    g_max = ppc["gen"][:, 8]  # 发电机最大功率
    g_min = ppc["gen"][:, 9]  # 发电机最小功率
    return c, F_max, g_min, g_max


def solve_dcopf_with_gurobi(ppc):
    """
    使用Gurobi求解DCOPF问题，返回包含所有拉格朗日乘子的结果
    
    输入:
        ppc (dict): PyPower格式的电网数据
        
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
    ppc = pypower.ext2int(ppc)
    baseMVA = ppc['baseMVA']
    bus = ppc['bus']
    gen = ppc['gen']
    branch = ppc['branch']
    
    # 计算PTDF矩阵
    ptdf = makePTDF(baseMVA, bus, branch)
    
    # 发电机参数
    gen_bus = gen[:, 0].astype(int)
    ng = len(gen_bus)
    c = gen[:, 5]  # 线性成本系数
    Pg_min = gen[:, 9] / baseMVA  # 转换为标幺值
    Pg_max = gen[:, 8] / baseMVA
    
    # 负荷参数
    Pd = bus[:, 2] / baseMVA
    
    # 创建模型
    model = gp.Model("DCOPF")
    
    # 定义变量（不直接设置lb/ub，而是通过显式约束）
    Pg = model.addVars(ng, name="Pg")  # 自由变量
    
    # 目标函数
    model.setObjective(gp.quicksum(c[i] * Pg[i] for i in range(ng)), GRB.MINIMIZE)
    
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
    
    # 结果处理
    result = {}
    if model.status == GRB.OPTIMAL:
        # 发电机最优出力 (MW)
        result['Pg_opt'] = [Pg[i].X * baseMVA for i in range(ng)]
        
        # 拉格朗日乘子
        result['lambda_power_balance'] = power_balance.Pi * baseMVA  # $/MWh
        
        # 线路约束乘子
        result['lambda_line_max'] = [c.Pi * baseMVA for c in line_max_constr]
        result['lambda_line_min'] = [c.Pi * baseMVA for c in line_min_constr]
        
        # 发电机出力上下限约束乘子
        result['lambda_pg_min'] = [c.Pi * baseMVA for c in pg_min_constr]
        result['lambda_pg_max'] = [c.Pi * baseMVA for c in pg_max_constr]
        
        # 调试输出：检查哪些约束被触发
        # print("\n发电机约束触发情况:")
        # for i in range(ng):
        #     status_min = "触发" if abs(Pg[i].X - Pg_min[i]) < 1e-6 else "未触发"
        #     status_max = "触发" if abs(Pg[i].X - Pg_max[i]) < 1e-6 else "未触发"
        #     print(f"发电机{i}: 下限约束 {status_min}, 上限约束 {status_max}")
    else:
        raise RuntimeError(f"优化失败，状态码: {model.status}")
    
    return result

def format_timestamp(ts):
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

def solve_dcopf_with_polycost(ppc):
    """
    使用Gurobi求解DCOPF问题，返回包含所有拉格朗日乘子的结果
    
    输入:
        ppc (dict): PyPower格式的电网数据
        
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
    model = gp.Model("DCOPF")
    
    # 定义变量（不直接设置lb/ub，而是通过显式约束）
    Pg = model.addVars(ng, name="Pg")  # 自由变量
    
    # 目标函数
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
    
    # 结果处理
    result = {}
    if model.status == GRB.OPTIMAL:
        # 发电机最优出力 (MW)
        result['Pg_opt'] = [Pg[i].X * baseMVA for i in range(ng)]
        
        # 拉格朗日乘子
        result['lambda_power_balance'] = power_balance.Pi * baseMVA  # $/MWh
        
        # 线路约束乘子
        result['lambda_line_max'] = [c.Pi * baseMVA for c in line_max_constr]
        result['lambda_line_min'] = [c.Pi * baseMVA for c in line_min_constr]
        
        # 发电机出力上下限约束乘子
        result['lambda_pg_min'] = [c.Pi * baseMVA for c in pg_min_constr]
        result['lambda_pg_max'] = [c.Pi * baseMVA for c in pg_max_constr]
        
        # 调试输出：检查哪些约束被触发
        # print("\n发电机约束触发情况:")
        # for i in range(ng):
        #     status_min = "触发" if abs(Pg[i].X - Pg_min[i]) < 1e-6 else "未触发"
        #     status_max = "触发" if abs(Pg[i].X - Pg_max[i]) < 1e-6 else "未触发"
        #     print(f"发电机{i}: 下限约束 {status_min}, 上限约束 {status_max}")
    else:
        raise RuntimeError(f"优化失败，状态码: {model.status}")
    
    return result


if __name__ == '__main__':
    load_file_path = 'hrl_load_metered_20242025.csv'
    wind_file_path = 'wind_gen_20242025.csv'
    solar_file_path = 'solar_gen_20242025.csv'
    data_load = pd.read_csv(load_file_path)
    data_wind = pd.read_csv(wind_file_path)
    data_solar = pd.read_csv(solar_file_path)
    # # 起始时间
    start_time = datetime.strptime("3/1/2024 5:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
    # end_time = datetime.strptime("3/1/2024 7:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
    end_time = datetime.strptime("3/1/2025 4:00:00 AM", "%m/%d/%Y %I:%M:%S %p")
    # 生成start_time end_time之间每隔 1 小时的时间戳
    timestamps = pd.date_range(start_time, end_time, freq='h')
    formatted_timestamps = [format_timestamp(ts) for ts in timestamps]
    data = {}
    for time in formatted_timestamps:
        ppc = set_case118(time, data_load, data_wind, data_solar)
        result = solve_dcopf_with_polycost(ppc)
        data[time] = result
    with open('result_poly.json', 'w') as f:
        json.dump(data, f, indent=4)
    # ppc = set_case118(formatted_timestamps[0], data_load, data_wind, data_solar)
    # ppc = pypower.case118()
    # result = solve_dcopf_with_polycost(ppc)
    # print(formatted_timestamps)
    # data = {}
    # Pd_list = []
    # for time in formatted_timestamps:
    #     ppc = set_case118(time, data_load, data_wind, data_solar)
    #     Pd = np.array(ppc['bus'][:, PD]).reshape(-1, 1)
    #     Pd_list.append(Pd)
    # Pd_list = np.hstack(Pd_list)
    # np.save("Pd.npy", Pd_list)


    # result = solve_dcopf_with_gurobi(ppc)
    # data[time] = result
    # with open('result.json', 'w') as f:
    #     json.dump(data, f, indent=4)
        
    

    #utc1 = '10/8/2024 6:00:00 AM'
    # utc2 = '10/8/2024 7:00:00 AM'
    # ppc = pypower.case39()
    # ppc = pypower.case118()
    # ppc = set_case118(utc2, load_file_path, wind_file_path, solar_file_path)
    # result_dict , result = dcopf_case118(ppc)
    # print(result)

    # result1 = case118dcopf(utc1, load_file_path, wind_file_path, solar_file_path)
    # result2 = dcopf_case118(utc2, load_file_path, wind_file_path, solar_file_path)
    # c, F_max, g_min, g_max = get_optimization_params(ppc)
    # PTDF_gen = PTDF(ppc)
    # g_opt, P_opt,  mu_F_upper, mu_F_lower, nu_plus, nu_minus = solve_opf(ppc)
    # ppc = pypower.case118()lambda_P,
    # H = PTDF(ppc)
    # print(np.shape(H))

    # c, F_max, g_min, g_max = get_optimization_params(ppc)


