import matplotlib.pyplot as plt
from datetime import datetime

# 读取数据
def read_data(file_path):
    times = []
    rates = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('INFO'):
                parts = line.strip().split()
                time_str = parts[1] + ' ' + parts[2]
                rate = float(parts[-1])
                time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S,%f')
                times.append(time)
                rates.append(rate)
    return times, rates

# 读取数据并作图
def plot_data(file_paths, labels):
    plt.figure(figsize=(10, 6))

    for i, file_path in enumerate(file_paths):
        times, rates = read_data(file_path)
        plt.plot(times, rates, label=labels[i])

    plt.xlabel('Time')
    plt.ylabel('Rate')
    plt.title('Rate over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

# 文件路径和标签
file_paths = ['logs_rate/model_1_rate.log', 'logs_rate/model_2_rate.log', 'logs_rate/model_3_rate.log', 'logs_rate/model_4_rate.log', 'logs_rate/model_5_rate.log']
labels = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']

# 绘制图像
plot_data(file_paths, labels)
