"""
Created on Mon Jan 6 2025 by Jianxiang
This file is the main file to solve the problem with gurobi
We form the dataset of lagrange multipliers and the corresponding optimal solution
"""
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
# import pickle
# class CustomJSONEncoder(json.JSONEncoder):
#     def encode(self, obj):
#         if isinstance(obj, list):
#             return '[' + ', '.join(self.encode(el) for el in obj) + ']'
#         return json.JSONEncoder.encode(self, obj)

if __name__ == "__main__":
    # 初始化参数
    N = 40 # 用户数量（小规模示例）
    T = 24  # 时间段数量
    x_lower = 1   # 下限
    x_upper = 10  # 上限
    
    target_demand = np.zeros(N)
    required_demand = np.zeros(N)  # r_i
    iteration = 0
    result = {}

    while iteration < 100:
        capacity = np.random.uniform(130.6, 221.8, T)  # 每个时间段的电网容量
        for i in range(N):
            r_i = np.random.uniform(31.35, 53.23)  # 每个用户的日累计需求
            required_demand[i] = r_i
            y = np.random.uniform(1.3, 2.2)  # 用户目标需求
            target_demand[i] = y
            
        
        # 创建模型
        model = gp.Model("demand_response")
        
        # 定义变量
        x = model.addVars(N, T, lb=x_lower, ub=x_upper, vtype=GRB.CONTINUOUS, name="x")
                
        # 定义目标函数
        model.setObjective(gp.quicksum(- target_demand[i]**2 + target_demand[i] * x[i,t] for i in range(N) for t in range(T)), GRB.MAXIMIZE)
        
        # 添加时段总需求上限约束
        supply_cap_constrs = []
        for t in range(T):
            constr = model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= capacity[t], f"supply_cap_{t}")
            supply_cap_constrs.append(constr)
        
        # 添加每个用户的需求阈值约束
        required_demand_constrs = []
        for i in range(N):
            constr = model.addConstr(gp.quicksum(x[i, t] for t in range(T)) >= required_demand[i], f"required_demand_{i}")
            required_demand_constrs.append(constr)
        
        # 求解模型
        model.optimize()
        
        # 获取拉格朗日乘子
        lagrange_supply_cap = np.zeros(T)
        for t in range(T):
            lagrange_supply_cap[t] = supply_cap_constrs[t].Pi
        # print(lagrange_supply_cap)
        lagrange_required_demand = np.zeros(N)
        for i in range(N):
            lagrange_required_demand[i] = required_demand_constrs[i].Pi
        # print(lagrange_required_demand)
        # 获取最优解
        xgurobi = np.zeros([N, T])
        if model.status == GRB.OPTIMAL:
            print(f"Optimal objective value: {model.objVal}")
            for i in range(N):
                for t in range(T):
                    xgurobi[i, t] = x[i, t].x
            # print(xgurobi)

            data_to_save = {
                'optimal_solution': xgurobi.tolist(),
                'supply_cap_constrs_lagrange': lagrange_supply_cap.tolist(),
                'required_demand_constrs_lagrange': lagrange_required_demand.tolist(),
                'capacity': capacity.tolist(),
                'target_demand': target_demand.tolist(),
                'required_demand': required_demand.tolist()
            }
            key = "data" + str(iteration)
            result[key] = data_to_save
            
            # with open('optimization_results.pkl', 'wb') as f:
            #     pickle.dump(data_to_save, f)
        else:
            print("No optimal solution found.")
        
        iteration += 1
        
    with open('optimization_results100.json', 'w') as f:
        json.dump(result, f, indent=4)

   
