import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data_list = []
    data = {}
    for line in lines:
        if "Worker:" in line:
            worker, tokens = line.split(", Tokens:")
            worker = worker.strip().split(" ")[-1]
            tokens = float(tokens.strip())
            data[worker] = tokens
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
        elif "P99 TTFT (ms):" in line:
            data['P99 TTFT (ms)'] = float(line.split(":")[1].strip())
        
        # 每次读取到 "P99 TTFT (ms):" 行时，表示一份数据结束
        if "P99 TTFT (ms):" in line:
            data_list.append(data)
            data = {}

    # 计算平均值
    avg_data = {}
    print(len(data_list))
    for key in data_list[0].keys():
        avg_data[key] = np.mean([data[key] for data in data_list])
    
    return avg_data

# 读取文件A和文件B
data_A = read_file('token-based.txt')
data_B = read_file('round-robin.txt')

# 绘制Worker处理的Token数对比图
workers = ['Worker1', 'Worker2']
workers_A = list(data_A.keys())[:2]
tokens_A = [data_A[worker] for worker in workers_A]
workers_B = list(data_B.keys())[:2]
tokens_B = [data_B[worker] for worker in workers_B]
x = np.arange(len(workers))  # x轴位置
width = 0.35  # 柱状图宽度

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, tokens_A, width, label='Token based Router', alpha=0.7)
plt.bar(x + width/2, tokens_B, width, label='Round robin Router', alpha=0.7)
plt.xlabel('Worker')
plt.ylabel('Tokens')
plt.title('Processed tokens of each worker')
plt.xticks(x, ['Worker1', 'Worker2'])
plt.legend()
plt.savefig('worker_tokens_comparison.png')
plt.close()

# 绘制Throughput对比图（分组柱状图）
throughput_metrics = ['Request throughput (req/s)', 'Input token throughput (tok/s)', 'Output token throughput (tok/s)']
throughput_A = [data_A[metric] for metric in throughput_metrics]
throughput_B = [data_B[metric] for metric in throughput_metrics]

x = np.arange(len(throughput_metrics))  # x轴位置
width = 0.35  # 柱状图宽度

plt.figure(figsize=(10, 6))
bars_A = plt.bar(x - width/2, throughput_A, width, label='Token based Router', alpha=0.7)
bars_B = plt.bar(x + width/2, throughput_B, width, label='Round robin Router', alpha=0.7)
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
plt.savefig('throughput_comparison.png')
plt.close()

# 绘制E2E Latency对比图（分组柱状图）
e2e_latency_metrics = ['Mean E2E Latency (ms)', 'Median E2E Latency (ms)', 'P99 E2E Latency (ms)']
e2e_latency_A = [data_A[metric] for metric in e2e_latency_metrics]
e2e_latency_B = [data_B[metric] for metric in e2e_latency_metrics]

x = np.arange(len(e2e_latency_metrics))  # x轴位置
width = 0.35  # 柱状图宽度

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, e2e_latency_A, width, label='Token based Router', alpha=0.7)
plt.bar(x + width/2, e2e_latency_B, width, label='Round robin Router', alpha=0.7)
plt.xlabel('E2E Latency Metrics')
plt.ylabel('Latency (ms)')
plt.title('E2E Latency')
plt.xticks(x, e2e_latency_metrics)
plt.legend()
plt.savefig('e2e_latency_comparison.png')
plt.close()

# 绘制TTFT对比图（分组柱状图）
ttft_metrics = ['Mean TTFT (ms)', 'Median TTFT (ms)', 'P99 TTFT (ms)']
ttft_A = [data_A[metric] for metric in ttft_metrics]
ttft_B = [data_B[metric] for metric in ttft_metrics]

x = np.arange(len(ttft_metrics))  # x轴位置
width = 0.35  # 柱状图宽度

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, ttft_A, width, label='Token based Router', alpha=0.7)
plt.bar(x + width/2, ttft_B, width, label='Round robin Router', alpha=0.7)
plt.xlabel('TTFT Metrics')
plt.ylabel('Latency (ms)')
plt.title('TTFT')
plt.xticks(x, ttft_metrics)
plt.legend()
plt.savefig('ttft_comparison.png')
plt.close()