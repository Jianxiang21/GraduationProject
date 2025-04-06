import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_load(file_path, month, load_type = 'load'):
    df = pd.read_csv(file_path)
    if load_type == 'load':
        area = 'mkt_region'
        load = 'mw'
    elif load_type == 'solar':
        area = 'area'
        load = 'solar_generation_mw'
    elif load_type == 'wind':
        area = 'area'
        load = 'wind_generation_mw'
    else:
        raise ValueError('Invalid load type!')
    # 解析时间列
    df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'], format='%m/%d/%Y %I:%M:%S %p')

    # 确定年份
    year = 2025 if month in [1, 2] else 2024

    # 构造起始和结束时间
    start_time = pd.Timestamp(f'{year}-{month:02d}-01 05:00:00')
    end_time = pd.Timestamp(f'{year}-{month:02d}-02 04:00:00')

    # 筛选数据
    load_data = df.loc[
        (df[area] == 'SOUTH') & 
        (df['datetime_beginning_utc'] >= start_time) & 
        (df['datetime_beginning_utc'] <= end_time),
        load
    ]

    return load_data.tolist()

def plot_load(file_path, months, load_type='load'):


    # df = pd.read_csv(file_path)
    
    # # 解析时间，并转换为本地时间（UTC+0 转 UTC+5）
    # df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'], format='%m/%d/%Y %I:%M:%S %p')
    # df['local_time'] = df['datetime_beginning_utc'] + pd.Timedelta(hours=5)  # UTC+5
    
    # # 提取小时
    # df['hour'] = df['local_time'].dt.hour
    
    plt.figure(figsize=(10, 6))  # 设定图像大小
    cmap = plt.cm.get_cmap('tab20', len(months))

    for month in months:
        # # 筛选指定月份的数据
        # month_data = df[(df['local_time'].dt.year == 2024) & (df['local_time'].dt.month == month) & (df['mkt_region'] == 'SOUTH')]
        
        # 计算每小时的平均负荷
        hourly_load = get_load(file_path, month, load_type)

        # 绘制曲线
        plt.plot(range(0,24), hourly_load, label=f'{month}', marker='o', color=cmap(month - 1))

    # 图表美化
    plt.xlabel('Hour(h)', fontsize=12)
    plt.ylabel('P(MW)', fontsize=12)
    plt.title(load_type + ' curves for different months', fontsize=14)
    plt.xticks(range(24))  # 横坐标设置 0-23
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
    plt.legend(title="Months")  # 添加图例
    plt.show()

def plot_LDC(file_path):
    # 读取数据
    df = pd.read_csv(file_path)

    month_data = df[
        (df['mkt_region'] == 'SOUTH')
    ]

    # 负荷数据排序（从高到低）
    sorted_load = np.sort(month_data['mw'])[::-1]

    # 计算时间占比
    time_percent = np.linspace(0, 100, len(sorted_load))  # 归一化时间

    # 绘制负荷持续时间曲线
    plt.figure(figsize=(10, 6))
    plt.plot(time_percent, sorted_load, color='b', linewidth=2)

    # 图表美化
    plt.xlabel('Time percentage (%)', fontsize=12)
    plt.ylabel('Load (MW)', fontsize=12)
    plt.title(f'Load Duration Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# def plot_stacked_bar_chart(y_real, y_predict):
#     # 计算偏差
#     y_diff = y_predict - y_real

#     # 生成柱状图
#     x = np.arange(len(y_real))  # 机组编号

#     fig, ax = plt.subplots(figsize=(12, 6))

#     # 真实值柱子
#     ax.bar(x, y_real, color='blue', label='Real Value')

#     # 偏差柱子（正负颜色区分）
#     ax.bar(x, y_diff, color=['red' if d > 0 else 'green' for d in y_diff], 
#            bottom=y_real, label='Residual', alpha=0.7)

#     # 添加标签
#     ax.set_xlabel("Generator ID")
#     ax.set_ylabel("P(MW)")
#     ax.set_title("Generator Output Prediction vs Real Value")
#     ax.legend()

#     plt.show()
def plot_stacked_bar_chart(y_real, y_predict):
    # 计算偏差
    y_diff = y_predict - y_real

    # 生成柱状图
    x = np.arange(len(y_real))  # 机组编号

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制真实值柱子
    ax.bar(x, y_real, color='blue', label='Real Value')

    # 分别绘制正负偏差（用不同颜色和标签）
    # 正偏差（预测值 > 真实值）
    ax.bar(x, np.clip(y_diff, 0, None), 
           bottom=y_real, 
           color='red', 
           alpha=0.7,
           label='Positive Residual (Overestimate)')
    
    # 负偏差（预测值 < 真实值）
    ax.bar(x, np.clip(y_diff, None, 0), 
           bottom=y_real, 
           color='green', 
           alpha=0.7,
           label='Negative Residual (Underestimate)')

    # 添加标签和标题
    ax.set_xlabel("Generator ID")
    ax.set_ylabel("P(MW)")
    ax.set_title("Generator Output Prediction vs Real Value")
    ax.legend()

    plt.show()

# a = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# b = np.array([3.5, 4.5, 4.5, 6.5, 7.5, 7.5, 9.5, 9.5, 11.5, 12.5])
# plot_stacked_bar_chart(a, b)
# 生成示例数据
# np.random.seed(42)
# n = 54  # 机组数
# y_real = np.random.uniform(50, 150, n)  # 真实值
# y_predict = y_real + np.random.uniform(-20, 20, n)  # 预测值，带有一定偏差

# # 计算偏差
# y_diff = y_predict - y_real

# # 生成柱状图
# x = np.arange(n)  # 机组编号

# fig, ax = plt.subplots(figsize=(12, 6))

# # 真实值柱子
# ax.bar(x, y_real, color='blue', label='真实值')

# # 偏差柱子（正负颜色区分）
# ax.bar(x, y_diff, color=['red' if d > 0 else 'green' for d in y_diff], 
#        bottom=y_real, label='偏差', alpha=0.7)

# # 添加标签
# ax.set_xlabel("机组编号")
# ax.set_ylabel("出力")
# ax.set_title("机组出力预测 vs 真实值")
# ax.legend()

# plt.show()

# 示例调用
# plot_load('hrl_load_metered_20242025.csv', range(1,13))  # 绘制 1月、3月、5月的负荷曲线
# plot_load('solar_gen_20242025.csv', range(1,13), 'solar')  # 绘制 1月、3月、5月的负荷曲线
# plot_load('wind_gen_20242025.csv', range(1,13), 'wind')  # 绘制 1月、3月、5月的负荷曲线
# plot_LDC('hrl_load_metered_20242025.csv')  # 绘制负荷持续时间曲线




