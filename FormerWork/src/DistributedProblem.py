# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:36:02 2024

@author: LJX1210


"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# 用户类：负责管理每个用户的需求计算和效用优化
class User:
    def __init__(self, user_id, r_i, y, demand_bounds, price,T):
        self.user_id = user_id
        self.r_i = r_i  # 用户的日累计需求
        self.y = y  # 用户目标需求
        self.x = np.zeros(T)  # 用户需求矩阵 x_i^t
        self.mu = 0  # 协调参数 μ
        self.demand_bounds = demand_bounds  # (x_min, x_max) 用电上下限
        self.price = price
        self.T = T
        self.mu0 = 0

    # 用户的效用函数
    def utility(self):
        u_t = []
        for x_t in self.x:
            if x_t >= self.y:
                u_t.append(0)
            elif x_t >= 0:
                u_t.append(-(x_t - self.y) * (x_t - self.y)) 
            else:
                print("x cannot be negative")
        return u_t 
    
    def Wit(self,i,t):
        u = self.utility()
        return u - self.price * x
    
    def W_derivative(self):
        dw = np.zeros(self.T)  # 使用numpy数组来初始化而非列表
        for t, x_t in enumerate(self.x):
            if x_t >= self.y:
                dw[t] = -self.price[t]
            else:
                dw[t] = -2 * (x_t - self.y) - self.price[t]
        return dw
    
    def W_derivative_inv(self, lambda_t, mui=None):
        Wprime = np.zeros(self.T)  # 初始化为24维一维数组
        for t, lamb in enumerate(lambda_t):
            # 如果提供了 mui 参数，则使用它；否则使用 self.mu
            current_mu = mui if mui is not None else self.mu
            
            if current_mu == lamb:
                Wprime[t] = self.y  # x > y 的情况
            else:
                Wprime[t] = 0.5 * (current_mu - lamb - self.price[t]) + self.y
        return Wprime

    
    def demand_sum(self, m, lambda_t): #g_i(mu_i,lambda)
        x_min, x_max = self.demand_bounds
        #Wprime_inv = self.W_derivative_inv(lambda_t)  
        xit = np.zeros(self.T)# 用来存储 (W^{t'}_i)^{-1} 的计算结果
        for t in range(self.T):
            # 计算 μ_i^t 和 μ̄_i^t
            mu_t_lower = lambda_t[t] - (-2 * (x_min - self.y) - self.price[t])
            mu_t_upper = lambda_t[t] + self.price[t]
            
            # 根据条件选择 x_i^t
            if m <= mu_t_lower:
                xit[t] = x_min
            elif m >= mu_t_upper:
                xit[t] = x_max
            else:
                # 计算 (W^{t'}_i)^{-1}(\lambda_t - μ_i)
                #Wprime_inv[t] = 0.5 * (self.mu - lambda_t[t] + self.price[t]) + self.y
                xit[t] = self.W_derivative_inv(lambda_t, m)[t]
            xit[t] = np.clip(xit[t], x_min, x_max)
        # 更新需求 x 为 Wprime_inv 的结果，确保需求在上下限内
        return sum(xit)
    
    def get_pos_breakpts(self, lambda_t):
        x_min, x_max = self.demand_bounds
        pts = np.zeros([self.T, 2])
        for t in range(self.T):
            pts[t,0] = lambda_t[t] + 2 * (x_min - self.y) + self.price[t]
            pts[t,1] = lambda_t[t] + self.price[t]
        pts_sort = np.sort(pts.flatten())
        return pts_sort[pts_sort > 0]
        

    # 快速求解 mu_i，使用二分搜索
    def fast_solve_mu(self, lambda_t, epsilon):
        # Step 1: 初始条件检查
        initial_demand_sum = self.demand_sum(0, lambda_t)
        if initial_demand_sum > self.r_i:
            self.mu = 0
            return
    
        # Step 2: 二分搜索
        breakpts = self.get_pos_breakpts(lambda_t)
        left, right = 0, len(breakpts) - 1
        
        while right - left > 1:
            middle = (left + right) // 2
            M = self.demand_sum(breakpts[middle], lambda_t)
            
            if M == self.r_i:
                self.mu = breakpts[middle]
                return breakpts[middle]
            elif M < self.r_i:
                left = middle
            else:
                right = middle
    
        # Step 3: 划分时间段集合 T1, T2, T3
        T1 = [t for t in range(len(lambda_t)) if lambda_t[t] + self.price[t] <= breakpts[left]]
        T2 = [t for t in range(len(lambda_t)) if lambda_t[t] - (-2 * (self.demand_bounds[0] - self.y) - self.price[t]) >= breakpts[right]]
        T3 = [t for t in range(len(lambda_t)) if t not in T1 and t not in T2]
    
        # Step 4: 计算满足总需求的 mu 值
        demand_sum_T1 = self.demand_bounds[1] * len(T1)
        demand_sum_T2 = self.demand_bounds[0] * len(T2)
        
        self.mu0 = self.mu
        self.mu = (sum((lambda_t[t] + self.price[t]) for t in T3) + 
                   2 * (self.r_i - demand_sum_T1 - demand_sum_T2)) / len(T3) - 2 * self.y
        

    # 更新用户的需求 x_i^t
    def update_demand(self, lambda_t):
        x_min, x_max = self.demand_bounds
        #Wprime_inv = self.W_derivative_inv(lambda_t)  
        xit = np.zeros(self.T)# 用来存储 (W^{t'}_i)^{-1} 的计算结果
        for t in range(self.T):
            # 计算 μ_i^t 和 μ̄_i^t
            mu_t_lower = lambda_t[t] - (-2 * (x_min - self.y) - self.price[t])
            mu_t_upper = lambda_t[t] + self.price[t]
            
            # 根据条件选择 x_i^t
            if self.mu <= mu_t_lower:
                xit[t] = x_min
            elif self.mu >= mu_t_upper:
                xit[t] = x_max
            else:
                # 计算 (W^{t'}_i)^{-1}(\lambda_t - μ_i)
                #Wprime_inv[t] = 0.5 * (self.mu - lambda_t[t] + self.price[t]) + self.y
                xit[t] = self.W_derivative_inv(lambda_t)[t]
        
        # 更新需求 x 为 Wprime_inv 的结果，确保需求在上下限内
        self.x = np.clip(xit, x_min, x_max)
        #return self.x


# LSE类：管理电网服务和协调价格 λ
class LSE:
    def __init__(self, num_users, num_timeslots, capacity, price, step_size, epsilon):
        self.num_users = num_users
        self.num_timeslots = num_timeslots
        self.capacity = capacity  # 电网容量 c_t
        self.price = price  # 电价 p_t
        self.lambda_t = np.zeros(num_timeslots)  # 拥堵价格 λ_t
        self.step_size = step_size  # 步长 gamma
        self.users = []
        self.epsilon = epsilon
        self.lambda_t0 = 0

    # 添加用户
    def add_user(self, user):
        if len(self.users) < self.num_users:
            self.users.append(user)
        else:
            raise ValueError("Error: Cannot add more users than the maximum allowed.")

    # 更新 λ_t 基于当前的总需求
    def update_lambda(self):
        total_demand = np.zeros(self.num_timeslots)
        for t in range(self.num_timeslots):
            total_demand[t] = sum(user.x[t] for user in self.users)
        delta = total_demand - self.capacity
        self.lambda_t0 = self.lambda_t
        self.lambda_t = np.maximum(self.lambda_t + self.step_size * delta, 0)

    # 计算收敛条件
    def check_convergence(self):
        lambda_converged = all(abs(self.lambda_t - self.lambda_t0) < self.epsilon)
        mu_converged = all(abs(user.mu - user.mu0) < self.epsilon for user in self.users)
        return lambda_converged and mu_converged
        

if __name__ == "__main__":
    
    # 初始化参数
    N = 40 # 用户数量（小规模示例）
    T = 24  # 时间段数量
    capacity = np.random.uniform(130.6, 221.8, T)  # 每个时间段的电网容量
    price = np.random.uniform(0.015, 0.03, T)  # 电价
    step_size = 0.03
    epsilon = 1e-6
    target_demand = np.zeros(N)
    # 初始化 LSE
    lse = LSE(num_users=N, num_timeslots=T, capacity=capacity, price=price, step_size=step_size, epsilon=epsilon)
    
    required_demand = np.zeros(N)  # r_i
    # 添加用户
    for i in range(N):
        r_i = np.random.uniform(31.35, 53.23)  # 每个用户的日累计需求
        required_demand[i] = r_i
        y = np.random.uniform(1.3, 2.2)  # 用户目标需求
        target_demand[i] = y
        demand_bounds = (1, 2.5)  # 用电上下限
        user = User(user_id=i, r_i=r_i, y=y, demand_bounds=demand_bounds, price=price, T = T)
        lse.add_user(user)
    
    #主迭代过程
    it = 1000
    for iteration in range(it):
        # Step 1: 更新每个用户的 mu 值
        for user in lse.users:
            user.fast_solve_mu(lse.lambda_t, epsilon)
    
        # Step 2: 更新 LSE 的 λ_t 值
        lse.update_lambda()
    
        # Step 3: 更新每个用户的需求 x_i^t
        for user in lse.users:
            user.update_demand(lse.lambda_t)
    
        # 检查收敛条件
        if lse.check_convergence():
            print(f"收敛在第 {iteration} 次迭代")
            break
        
    X = []
    # 输出结果
    if iteration == it - 1:
        print("Not converged")
    else:
        print("Converged!")
        for user in lse.users:
            
            print(f"用户 {user.user_id} 的需求矩阵 x:")
            print(user.x)
            X.append(user.x)
    X_user = np.vstack(X)    
    
    
    # 生成样例数据
    x_lower = 1   # 下限
    x_upper = 2.5  # 上限
   
    
    # 创建模型
    model = gp.Model("demand_response")
    
    # 定义变量
    x = model.addVars(N, T, lb=x_lower, ub=x_upper, vtype=GRB.CONTINUOUS, name="x")
    u = model.addVars(N, T, vtype=GRB.CONTINUOUS, name="u")  # 分段目标值
    z = model.addVars(N, T, vtype=GRB.BINARY, name="z")  # 分段指示变量
    
    for i in range(N):
        for t in range(T):
            target = target_demand[i]
            model.addConstr(u[i,t] <= 0)
            # diff = model.addVar(vtype=GRB.CONTINUOUS, name=f"diff_{i}_{t}")
            # #model.addConstr(diff == - (x[i, t] - target) ** 2, name=f"diff_def_{i}_{t}")
            # model.addConstr(diff == (x[i, t] - target), name=f"diff_def_{i}_{t}")
            model.addConstr(u[i,t] >= - (x[i, t] - target) ** 2)
            model.addConstr(u[i,t] >= 1e6 * (z[i,t] - 1))
            model.addConstr(u[i,t] <= - (x[i, t] - target) ** 2 + 1e6 * z[i,t])
            
    # 定义目标函数
    model.setObjective(gp.quicksum(u[i,t] - price[t] * x[i, t] for i in range(N) for t in range(T)), GRB.MAXIMIZE)
    
    # 添加时段总需求上限约束
    for t in range(T):
        model.addConstr(gp.quicksum(x[i, t] for i in range(N)) <= capacity[t], f"supply_cap_{t}")
    
    # 添加每个用户的需求阈值约束
    for i in range(N):
        model.addConstr(gp.quicksum(x[i, t] for t in range(T)) >= required_demand[i], f"required_demand_{i}")
    
    # 求解模型
    model.optimize()
    
    xgurobi = np.zeros([N,T])
    # 输出结果
    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective value: {model.objVal}")
        for i in range(N):
            for t in range(T):
                #print(f"x[{i}, {t}] = {x[i, t].x}")
                xgurobi[i,t] = x[i,t].x
    else:
        print("No optimal solution found.")
        




    # user0 = lse.users[0]
    # id0 = user0.user_id
    # r0 = user0.r_i
    # y0 = user0.y
    # x0 = user0.x
    # mu0 = user0.mu
    # bound0 = user0.demand_bounds
    # price0 = user0.price
    
    # ut = user0.utility()
    # dw = user0.W_derivative()
    # lambda_t = np.zeros(T)
    # dwr = user0.W_derivative_inv(lambda_t)
    # dwr0 = user0.W_derivative_inv(lambda_t,0)
    # pts = user0.get_pos_breakpts(lambda_t)
    # demand = user0.update_demand(lambda_t)
    # s = user0.demand_sum(0,lambda_t)
    
    







































