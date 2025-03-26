import subprocess
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Tuple

def create_experiment_dir() -> str:
    now = datetime.utcnow() + timedelta(hours=8)
    formatted_now = now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(formatted_now, exist_ok=True)
    return formatted_now

def run_command(command: str, 
                success_message: Optional[str] = None, 
                log_file: Optional[str] = None, 
                wait_for_success: bool = False) -> Tuple[subprocess.Popen, str]:
    """优化后的命令运行函数"""
    print(f"Running command: {command}")
    start_time = time.time()
    
    if log_file:
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  # 行缓冲
            )

            while True:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                
                log.write(line)
            
            if wait_for_success and success_message:
                print(f"Waiting for success message: '{success_message}'")
                    
                if success_message in line:
                    print("Success message found. Proceeding to next step.")
                    return process, ""
            else:
                # 对于需要持续运行的后台进程（如router.py）
                return process, ""
    
    # 非日志模式的处理（同步执行）
    process = subprocess.Popen(command, shell=True)
    process.wait()
    return process, ""

def run_experiment(base_dir: str, strategy: str, run_id: int):
    """运行一次完整实验"""
    print(f"\n=== Starting Experiment {run_id} with strategy {strategy} ===")
    
    # 创建策略目录和运行目录
    strategy_dir = os.path.join(base_dir, strategy)
    os.makedirs(strategy_dir, exist_ok=True)
    run_dir = os.path.join(strategy_dir, str(run_id))
    os.makedirs(run_dir, exist_ok=True)
    
    processes = []  # 用于保存所有子进程
    
    try:
        # 第一步
        step1_log = os.path.join(run_dir, "run_replicas.log")
        step1_cmd = (
            "python run_replicas.py --host 127.0.0.1 --worker-ports 8001,8002 "
            "--gpu-indices 0,1 --model-name 'Qwen/Qwen2.5-1.5B-Instruct' "
            "--no-enable-prefix-caching"
        )
        step1_process, _ = run_command(
            step1_cmd, 
            success_message="INFO:     Application startup complete.", 
            log_file=step1_log, 
            wait_for_success=True
        )
        processes.append(step1_process)
        
        # 第二步 - 作为后台进程运行
        step2_log = os.path.join(run_dir, "router.log")
        step2_cmd = (
            f"python router.py --worker-ports 8001,8002 "
            f"--port 8000 --strategy {strategy}"
        )
        step2_process, _ = run_command(
            step2_cmd, 
            log_file=step2_log
        )
        processes.append(step2_process)
        
        # 短暂等待确保router启动
        time.sleep(3)
        
        # 第三步 - 同步执行
        step3_log = os.path.join(run_dir, "bench_serving.log")
        step3_cmd = (
            "python3 -m bench_serving --backend vllm --host 127.0.0.1 --port 8000 "
            "--dataset-name sharegpt --num-prompts 1024 --sharegpt-output-len 256 "
            "--max-concurrency 32 --dataset-path ./long_short_pattern_sharegpt_requests.json"
        )
        step3_process, _ = run_command(
            step3_cmd, 
            log_file=step3_log
        )
        step3_process.wait()
        
    finally:
        # 确保所有进程都被终止
        for p in processes:
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        
    print(f"=== Experiment {run_id} with strategy {strategy} completed ===")

def main():
    strategies = ["round_robin", "tokens", "pow_2"]
    num_runs = 3
    
    # 创建实验目录
    experiment_dir = create_experiment_dir()
    print(f"All results will be saved in directory: {experiment_dir}")
    
    try:
        for strategy in strategies:
            for run_id in range(1, num_runs + 1):
                run_experiment(experiment_dir, strategy, run_id)
                time.sleep(5)  # 实验间间隔
        
        print(f"\nAll experiments completed. Results saved in {experiment_dir}")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()