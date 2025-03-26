import subprocess
import time
import sys
import os
import re
from datetime import datetime
from typing import Optional, Tuple

def create_experiment_dir() -> str:
    """创建以当前日期时间命名的实验目录"""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(now, exist_ok=True)
    return now

# def run_command(command: str, 
#                 success_message: Optional[str] = None, 
#                 log_file: Optional[str] = None, 
#                 wait_for_success: bool = False) -> Tuple[subprocess.Popen, str]:
#     """运行命令并处理输出"""
#     print(f"Running command: {command}")
#     output_buffer = []
    
#     if log_file:
#         with open(log_file, 'w') as log:
#             process = subprocess.Popen(
#                 command, 
#                 shell=True, 
#                 stdout=subprocess.PIPE, 
#                 stderr=subprocess.STDOUT,
#                 universal_newlines=True
#             )
            
#             while True:
#                 line = process.stdout.readline()
#                 if not line and process.poll() is not None:
#                     break
#                 if line:
#                     log.write(line)
#                     output_buffer.append(line)
#     else:
#         process = subprocess.Popen(command, shell=True)
#         process.wait()
    
#     full_output = ''.join(output_buffer)
    
#     if wait_for_success and success_message:
#         print(f"Waiting for success message: '{success_message}'")
#         while True:
#             if success_message in full_output:
#                 print("Success message found. Proceeding to next step.")
#                 break
#             time.sleep(1)
#             line = process.stdout.readline()
#             if line:
#                 if log_file:
#                     with open(log_file, 'a') as log:
#                         log.write(line)
#                 full_output += line
    
#     return process, full_output

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
            
            if wait_for_success and success_message:
                print(f"Waiting for success message: '{success_message}'")
                while True:
                    
                    line = process.stdout.readline()
                    if not line:
                        if process.poll() is not None:
                            break
                        continue
                    
                    log.write(line)
                    
                    # 关键优化：逐行检查
                    if success_message in line:
                        print("Success message found. Proceeding to next step.")
                        return process, ""
            else:
                # 非等待模式的处理
                output = []
                for line in process.stdout:
                    log.write(line)
                    output.append(line)
                return process, "".join(output)
    
    # 非日志模式的处理
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
    
    # 第二步
    step2_log = os.path.join(run_dir, "router.log")
    step2_cmd = (
        f"python router.py --worker-ports 8001,8002 "
        f"--port 8000 --strategy {strategy}"
    )
    step2_process, _ = run_command(
        step2_cmd, 
        log_file=step2_log
    )
    
    # 第三步
    step3_log = os.path.join(run_dir, "bench_serving.log")
    step3_cmd = (
        "python3 -m bench_serving --backend vllm --host 127.0.0.1 --port 8000 "
        "--dataset-name sharegpt --num-prompts 1024 --sharegpt-output-len 256 "
        "--max-concurrency 32 --dataset-path ./long_short_pattern_sharegpt_requests.json"
    )
    step3_process, step3_output = run_command(
        step3_cmd, 
        log_file=step3_log
    )
    
    # 等待第三步完成
    step3_process.wait()
    
    # 终止第一步和第二步进程
    step1_process.terminate()
    step2_process.terminate()
    
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
                
                # 短暂暂停，确保端口释放
                time.sleep(5)
        
        print(f"\nAll experiments completed. Results saved in {experiment_dir}")
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()