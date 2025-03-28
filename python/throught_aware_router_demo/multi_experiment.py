import subprocess
import threading
import time
import argparse

def run_vllm_server():
    cmd = """python run_replicas.py \
        --host 127.0.0.1 --worker-ports 8001,8002 \
        --gpu-indices 0,1 --model-name 'Qwen/Qwen2.5-1.5B-Instruct' \
        --no-enable-prefix-caching"""
    
    # 启动子进程并捕获输出
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # 监控输出
    for line in process.stdout:
        print(line, end="")  # 打印控制台输出
        if "Application startup complete" in line:
            time.sleep(2)
            run_router()
            time.sleep(2)
            break

def run_router(strategy):
    cmd = f"""python router.py \
        --worker-ports 8001,8002 \
        --port 8000 \
        --strategy {strategy} \
        > router_{strategy}.log"""
    
    # 启动路由器
    subprocess.Popen(cmd, shell=True)
    print(f"Router started with strategy: {strategy}")

def run_benchmark(num_prompts, concurrency):
    cmd = f"""python3 -m bench_serving --backend vllm --host 127.0.0.1 --port 8000 \
    --dataset-name sharegpt --num-prompts {num_prompts} --sharegpt-output-len 256 --max-concurrency {concurrency} \
    --dataset-path ./short_80_long_20_sharegpt_requests.json \
        > benchmark.log"""
    
    # 启动基准测试
    subprocess.Popen(cmd, shell=True)
    print(f"Benchmark started with num_prompts={num_prompts} and concurrency={concurrency}")

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

    # 等待 VLLM 服务器启动完成后运行路由器和基准测试
    server_thread.join()
    run_router(args.strategy)
    run_benchmark(args.num_prompts, args.concurrency)