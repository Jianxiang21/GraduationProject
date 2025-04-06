"""
This script is used to check if the data generated suit the optimization problem.
"""

import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class CheckLagrange:
    def __init__(self, json_file):
        data = json.load(open(json_file, 'r'))
        data_gen = np.array(data)
        row,_ = data_gen.shape
        dict_gen = {}
        for i in range(row):
            dict = {}
            dict["supply_cap_constrs_lagrange"] = data_gen[i,0:24]  # 24个
            dict["required_demand_constrs_lagrange"] = data_gen[i,24:64] # 40个
            dict["optimal_solution"] = data_gen[i,64:1024]    # 40*24个
            dict["capacity"] = data_gen[i,1024:1048] # 24个
            dict["target_demand"] = data_gen[i,1048:1088] # 40个
            dict["required_demand"] = data_gen[i,1088:] # 40个
            dict_gen[str(i)] = dict
        self.data = dict_gen
            

    def check(self,index):
        data_element = self.data[str(index)]
        supply_cap_constrs_lagrange_gen = data_element["supply_cap_constrs_lagrange"]
        required_demand_constrs_lagrange_gen = data_element["required_demand_constrs_lagrange"]
        optimal_solution_gen = data_element["optimal_solution"]
        capacity = data_element["capacity"]
        target_demand = data_element["target_demand"]
        required_demand = data_element["required_demand"]
        
        N = 40 # 用户数量（小规模示例）
        T = 24  # 时间段数量
        x_lower = 1   # 下限
        x_upper = 10  # 上限
            
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
        else:
            print("No optimal solution found.")
        
        # 将 optimal_solution_gen 调整为与 xgurobi 相同的形状
        optimal_solution_gen = optimal_solution_gen.reshape(N, T)

        diff1 = supply_cap_constrs_lagrange_gen - lagrange_supply_cap
        diff2 = required_demand_constrs_lagrange_gen - lagrange_required_demand
        diff3 = optimal_solution_gen - xgurobi
        
        # 计算误差
        error_supply_cap = np.linalg.norm(diff1)
        error_required_demand = np.linalg.norm(diff2)
        error_optimal_solution = np.linalg.norm(diff3)
        print(f"误差：时段总需求上限约束拉格朗日乘子：{error_supply_cap}")
        print(f"误差：用户需求阈值约束拉格朗日乘子：{error_required_demand}")
        print(f"误差：最优解：{error_optimal_solution}")

        # 计算百分误差
        percent_error_supply_cap = (error_supply_cap / np.linalg.norm(supply_cap_constrs_lagrange_gen)) * 100
        percent_error_required_demand = (error_required_demand / np.linalg.norm(required_demand_constrs_lagrange_gen)) * 100
        percent_error_optimal_solution = (error_optimal_solution / np.linalg.norm(optimal_solution_gen)) * 100
        # 打印百分误差
        print(f"百分误差：时段总需求上限约束拉格朗日乘子：{percent_error_supply_cap:.2f}%")
        print(f"百分误差：用户需求阈值约束拉格朗日乘子：{percent_error_required_demand:.2f}%")
        print(f"百分误差：最优解：{percent_error_optimal_solution:.2f}%")

        return diff1,diff2,diff3,xgurobi,optimal_solution_gen

if __name__ == "__main__":
    check = CheckLagrange('new_samples2500.json')
    diff1, diff2, diff3,xgurobi,optimal_solution_gen = check.check(1)
    # print(diff1)
    # print(diff2)   
    # print(diff3)
    print(xgurobi)
    print(optimal_solution_gen)
    