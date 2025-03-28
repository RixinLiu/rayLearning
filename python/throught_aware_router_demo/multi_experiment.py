import subprocess
import threading
import time
import argparse
import socket
import signal
import os
import sys
from datetime import datetime
from datetime import datetime, timedelta

global args
router_started = False  # 全局变量，标记路由器是否已启动
lock = threading.Lock()  # 用于线程同步的锁
processes = []  # 用于存储所有子进程的列表

def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def run_vllm_server():
    cmd = """python run_replicas.py \
        --host 127.0.0.1 --worker-ports 8001,8002 \
        --gpu-indices 0,1 --model-name 'Qwen/Qwen2.5-1.5B-Instruct' \
        --no-enable-prefix-caching > run_replicas.log"""
    
    # 启动子进程并捕获输出
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    processes.append(process)  # 将子进程添加到全局列表中
    
    # 监控输出
    for line in process.stdout:
        print(line, end="")  # 打印控制台输出
        if "Application startup complete" in line:
            print("VLLM server started successfully! Detected 'Application startup complete' in log.")
            time.sleep(2)
            break

def run_router(strategy, experiment_dir):
    global router_started
    with lock:  # 确保线程安全
        if router_started:
            print("Router is already running. Skipping...")
            return

        # 检查端口是否被占用
        if is_port_in_use(8000):
            print("Error: Port 8000 is already in use. Exiting...")
            return

        # 标记路由器已启动
        router_started = True

        cmd = f"""python router.py \
            --worker-ports 8001,8002 \
            --port 8000 \
            --strategy {strategy}"""
        
        # 启动路由器并捕获输出
        router_log_path = os.path.join(experiment_dir, f"router_{strategy}.log")
        with open(router_log_path, "w") as log_file:
            process = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file)
            processes.append(process)  # 将子进程添加到全局列表中
        
        print(f"Router started with strategy: {strategy}")

def run_benchmark(num_prompts, concurrency, experiment_dir):
    cmd = f"""python3 -m bench_serving --backend vllm --host 127.0.0.1 --port 8000 \
    --dataset-name sharegpt --num-prompts {num_prompts} --sharegpt-output-len 256 --max-concurrency {concurrency} \
    --dataset-path ./short_80_long_20_sharegpt_requests.json"""
    
    # 启动基准测试并捕获输出
    benchmark_log_path = os.path.join(experiment_dir, "benchmark.log")
    with open(benchmark_log_path, "w") as log_file:
        process = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file)
        processes.append(process)  # 将子进程添加到全局列表中
    
    print(f"Benchmark started with {num_prompts} prompts and concurrency {concurrency}.")

    # 监控 benchmark.log 文件
    monitor_benchmark_log(benchmark_log_path)

def monitor_benchmark_log(log_file_path):
    """监控 benchmark.log 文件的输出"""
    print("Monitoring benchmark log for completion...")
    with open(log_file_path, "r") as log_file:
        while True:
            line = log_file.readline()
            if not line:
                time.sleep(0.1)  # 如果没有新内容，稍作等待
                continue
            print(line, end="")  # 打印日志内容
            if "Serving Benchmark Result" in line:
                print("Benchmark completed! Detected 'Serving Benchmark Result' in log.")
                break

def terminate_all_processes():
    """终止所有子进程"""
    print("\nTerminating all processes...")
    for process in processes:
        if process.poll() is None:  # 如果进程仍在运行
            process.terminate()  # 发送终止信号
            try:
                process.wait(timeout=5)  # 等待进程终止
            except subprocess.TimeoutExpired:
                process.kill()  # 强制杀死进程
    processes.clear()  # 清空进程列表
    global router_started
    router_started = False  # 重置路由器启动标志
    print("All processes terminated.")

    # 确保端口释放
    wait_for_port_release(8000)

def wait_for_port_release(port, timeout=10):
    """等待端口释放"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_port_in_use(port):
            print(f"Port {port} is now free.")
            return
        print(f"Waiting for port {port} to be released...")
        time.sleep(1)
    print(f"Warning: Port {port} is still in use after {timeout} seconds.")

def signal_handler(sig, frame):
    """捕获 Ctrl+C 信号并终止所有子进程"""
    terminate_all_processes()
    sys.exit(0)

if __name__ == "__main__":
    # 捕获 Ctrl+C 信号
    signal.signal(signal.SIGINT, signal_handler)

    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Run VLLM server, router, and benchmark.")
    parser.add_argument("--num-prompts", type=int, default=1024, help="Number of prompts for benchmarking.")
    parser.add_argument("--concurrency", type=int, default=32, help="Maximum concurrency for benchmarking.")
    args = parser.parse_args()

    # 获取当前中国时间并创建实验目录
    china_time = datetime.utcnow() + timedelta(hours=8)
    experiment_root_dir = china_time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(experiment_root_dir, exist_ok=True)

    # 路由策略和实验次数
    strategies = ["pow_2", "tokens", "round_robin"]
    num_experiments = 3

    for strategy in strategies:
        strategy_dir = os.path.join(experiment_root_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)

        for i in range(1, num_experiments + 1):
            experiment_dir = os.path.join(strategy_dir, f"experiment_{i}")
            os.makedirs(experiment_dir, exist_ok=True)

            print(f"Running experiment {i} for strategy {strategy}...")
            run_vllm_server()
            run_router(strategy, experiment_dir)
            run_benchmark(args.num_prompts, args.concurrency, experiment_dir)

            # 终止所有进程并等待 5 秒
            terminate_all_processes()
            time.sleep(30)