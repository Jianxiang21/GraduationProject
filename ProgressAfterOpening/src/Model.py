import json
import numpy as np
from case118dcopf import format_timelist, START_TIME, END_TIME
"""
lambda_line_max
lambda_line_min
lambda_pg_max
lambda_pg_min
lambda_power_balance
Pg_opt
186
186
54
54
1
54
"""
class DCOPFModel:
    def __init__(self, json_file):
        self.path = json_file

    def load_y(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
        timelist = format_timelist(START_TIME, END_TIME)
        data_y = []
        # 遍历获得vector，每个vector是一个时间点的数据，然后将其拼接成一个矩阵
        for time in timelist:
            item = data[time]
            vector = np.array(item['Pg_opt'] + [item['lambda_power_balance']] + item['lambda_line_max'] + 
                              item['lambda_line_min'] + item['lambda_pg_min'] + item['lambda_pg_max']).reshape(-1, 1)
            data_y.append(vector)
        data_y = np.hstack(data_y)
        return data_y
    
    def load_X(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
        timelist = format_timelist(START_TIME, END_TIME)
        data_X = []
        for time in timelist:
            item = data[time]
            vector = np.array(item['d']).reshape(-1, 1)
            data_X.append(vector)
        data_X = np.hstack(data_X)
        return data_X

if __name__ == "__main__":
    model = DCOPFModel('result_poly.json')
    y = model.load_y()
    np.save('y_poly.npy', y)
    # y = np.load('y.npy')
    # X = np.load('Pd.npy')
    # print(data)


