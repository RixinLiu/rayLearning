import subprocess
import threading
import time
import argparse
import socket

global args
router_started = False  # 全局变量，标记路由器是否已启动
lock = threading.Lock()  # 用于线程同步的锁

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
    
    # 监控输出
    for line in process.stdout:
        print(line, end="")  # 打印控制台输出
        if "Application startup complete" in line:
            time.sleep(2)
            run_router(args.strategy)
            break

def run_router(strategy):
    global router_started
    with lock:  # 确保线程安全
        if router_started:
            print("Router is already running. Skipping...")
            return

        # 检查端口是否被占用
        if is_port_in_use(8000):
            print("Error: Port 8000 is already in use. Exiting...")
            return

        cmd = f"""python router.py \
            --worker-ports 8001,8002 \
            --port 8000 \
            --strategy {strategy} \
            > router_{strategy}.log"""
        
        # 启动路由器
        subprocess.Popen(cmd, shell=True)
        router_started = True
        print(f"Router started with strategy: {strategy}")

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Run VLLM server, router, and benchmark.")
    parser.add_argument("--strategy", type=str, default="round_robin", help="Routing strategy (e.g., round_robin, tokens, pow_2.)")
    parser.add_argument("--num-prompts", type=int, default=1024, help="Number of prompts for benchmarking.")
    parser.add_argument("--concurrency", type=int, default=32, help="Maximum concurrency for benchmarking.")
    args = parser.parse_args()

    # 使用线程运行 VLLM 服务器
    server_thread = threading.Thread(target=run_vllm_server)
    server_thread.start()
    server_thread.join()  # 等待 VLLM 服务器线程完成