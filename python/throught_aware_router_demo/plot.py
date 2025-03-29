import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data_list = []
    data = {}
    token1 = 0
    token2 = 0
    for line in lines:
        # 跳过以 "Long requests" 或 "Short requests" 开头的行
        if line.strip().startswith("Long requests") or line.strip().startswith("Short requests"):
            continue

        if "Worker: http://localhost:8001," in line:
            worker, tokens = line.split(", Tokens:")
            worker = worker.strip().split(" ")[-1]
            token1 = float(tokens.strip())
        elif "Worker: http://localhost:8002," in line:
            worker, tokens = line.split(", Tokens:")
            worker = worker.strip().split(" ")[-1]
            token2 = float(tokens.strip())
            data["Difference in Tokens"] = abs(token1 - token2)  # 计算两个 worker 的 token 数差值
        elif "Successful requests:" in line:
            data['Successful requests'] = int(line.split(":")[1].strip())
        elif "Benchmark duration (s):" in line:
            data['Benchmark duration (s)'] = float(line.split(":")[1].strip())
        elif "Total input tokens:" in line:
            data['Total input tokens'] = int(line.split(":")[1].strip())
        elif "Total generated tokens:" in line:
            data['Total generated tokens'] = int(line.split(":")[1].strip())
        elif "Request throughput (req/s):" in line:
            data['Request throughput (req/s)'] = float(line.split(":")[1].strip())
        elif "Input token throughput (tok/s):" in line:
            data['Input token throughput (tok/s)'] = float(line.split(":")[1].strip())
        elif "Output token throughput (tok/s):" in line:
            data['Output token throughput (tok/s)'] = float(line.split(":")[1].strip())
        elif "Mean E2E Latency (ms):" in line:
            data['Mean E2E Latency (ms)'] = float(line.split(":")[1].strip())
        elif "Median E2E Latency (ms):" in line:
            data['Median E2E Latency (ms)'] = float(line.split(":")[1].strip())
        elif "P99 E2E Latency (ms):" in line:
            data['P99 E2E Latency (ms)'] = float(line.split(":")[1].strip())
        elif "Mean TTFT (ms):" in line:
            data['Mean TTFT (ms)'] = float(line.split(":")[1].strip())
        elif "Median TTFT (ms):" in line:
            data['Median TTFT (ms)'] = float(line.split(":")[1].strip())
        elif "P95 TTFT (ms):" in line:
            data['P95 TTFT (ms)'] = float(line.split(":")[1].strip())
        elif "P99 TTFT (ms):" in line:
            data['P99 TTFT (ms)'] = float(line.split(":")[1].strip())
        
        # 每次读取到 "P99 ITL (ms):" 行时，表示一份数据结束
        if "P99 ITL (ms)" in line:
            data_list.append(data)
            data = {}

    # 计算平均值
    avg_data = {}
    keys = set(key for data in data_list for key in data.keys())
    for key in keys:
        avg_data[key] = np.mean([data[key] for data in data_list if key in data])
    
    return avg_data

# 读取文件A、文件B和文件C
data_A = read_file('tokens.txt')
data_B = read_file('round_robin.txt')
data_C = read_file('pow_2.txt')

# 修复 calculate_worker_token_difference 函数
def calculate_worker_token_difference(data):
    return data.get("Difference in Tokens", 0)

token_diff_A = calculate_worker_token_difference(data_A)
token_diff_B = calculate_worker_token_difference(data_B)
token_diff_C = calculate_worker_token_difference(data_C)

# 绘制Worker处理的Token数差值对比图
strategies = ['Token based Router', 'Round robin Router', 'Pow 2 Router']
token_differences = [token_diff_A, token_diff_B, token_diff_C]
x = np.arange(len(strategies))  # x轴位置

plt.figure(figsize=(10, 6))
bars = plt.bar(x, token_differences, color=['blue', 'orange', 'green'], alpha=0.7)
plt.xlabel('Routing Strategies')
plt.ylabel('Token Difference')
plt.title('Token Difference Between Two Workers')
plt.xticks(x, strategies)

# 在柱状图上标出具体数值
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.savefig('worker_token_difference.png')
plt.close()

# 绘制Throughput对比图（分组柱状图）
throughput_metrics = ['Request throughput (req/s)', 'Input token throughput (tok/s)', 'Output token throughput (tok/s)']
throughput_A = [data_A[metric] for metric in throughput_metrics]
throughput_B = [data_B[metric] for metric in throughput_metrics]
throughput_C = [data_C[metric] for metric in throughput_metrics]

x = np.arange(len(throughput_metrics))  # x轴位置
width = 0.25  # 柱状图宽度

plt.figure(figsize=(10, 6))
bars_A = plt.bar(x - width, throughput_A, width, label='Token based Router', alpha=0.7)
bars_B = plt.bar(x, throughput_B, width, label='Round robin Router', alpha=0.7)
bars_C = plt.bar(x + width, throughput_C, width, label='Pow 2 Router', alpha=0.7)
plt.xlabel('Throughput Metrics')
plt.ylabel('Throughput')
plt.title('Throughput')
plt.xticks(x, throughput_metrics)
plt.legend()
# 在柱状图上标出具体数值
for bar in bars_A:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
for bar in bars_B:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
for bar in bars_C:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
plt.savefig('throughput_comparison.png')
plt.close()

# 绘制E2E Latency对比图（分组柱状图）
e2e_latency_metrics = ['Mean E2E Latency (ms)', 'Median E2E Latency (ms)', 'P99 E2E Latency (ms)']
e2e_latency_A = [data_A[metric] for metric in e2e_latency_metrics]
e2e_latency_B = [data_B[metric] for metric in e2e_latency_metrics]
e2e_latency_C = [data_C[metric] for metric in e2e_latency_metrics]

x = np.arange(len(e2e_latency_metrics))  # x轴位置
width = 0.25  # 柱状图宽度

plt.figure(figsize=(10, 6))
plt.bar(x - width, e2e_latency_A, width, label='Token based Router', alpha=0.7)
plt.bar(x, e2e_latency_B, width, label='Round robin Router', alpha=0.7)
plt.bar(x + width, e2e_latency_C, width, label='Pow 2 Router', alpha=0.7)
plt.xlabel('E2E Latency Metrics')
plt.ylabel('Latency (ms)')
plt.title('E2E Latency')
plt.xticks(x, e2e_latency_metrics)
plt.legend()
plt.savefig('e2e_latency_comparison.png')
plt.close()

# 绘制TTFT对比图（分组柱状图）
ttft_metrics = ['Mean TTFT (ms)', 'Median TTFT (ms)', 'P95 TTFT (ms)', 'P99 TTFT (ms)']
ttft_A = [data_A[metric] for metric in ttft_metrics]
ttft_B = [data_B[metric] for metric in ttft_metrics]
ttft_C = [data_C[metric] for metric in ttft_metrics]

x = np.arange(len(ttft_metrics))  # x轴位置
width = 0.25  # 柱状图宽度

plt.figure(figsize=(10, 6))
plt.bar(x - width, ttft_A, width, label='Token based Router', alpha=0.7)
plt.bar(x, ttft_B, width, label='Round robin Router', alpha=0.7)
plt.bar(x + width, ttft_C, width, label='Pow 2 Router', alpha=0.7)
plt.xlabel('TTFT Metrics')
plt.ylabel('Latency (ms)')
plt.title('TTFT Comparison')
plt.xticks(x, ttft_metrics)
plt.legend()

# # 在柱状图上标出具体数值
# for bar_group in [ttft_A, ttft_B, ttft_C]:
#     for bar in bar_group:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

plt.savefig('ttft_comparison_with_p95.png')
plt.close()