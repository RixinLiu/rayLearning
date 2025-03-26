import os
import subprocess
import time
from datetime import datetime

# Configuration
STRATEGIES = ['tokens', 'round_robin', 'pow_2']
NUM_RUNS = 3
STEP1_SUCCESS_MSG = 'INFO:     Application startup complete.'
STEP3_SUCCESS_MSG = '============ Serving Benchmark Result ============'

def run_experiment(root_dir):
    for strategy in STRATEGIES:
        strategy_dir = os.path.join(root_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)
        
        for run_num in range(1, NUM_RUNS+1):
            run_dir = os.path.join(strategy_dir, str(run_num))
            os.makedirs(run_dir, exist_ok=True)
            
            print(f"Running experiment: {strategy} - Run {run_num}")
            
            # Step 1: Start replicas
            step1_log = os.path.join(run_dir, 'run_replicas.log')
            proc_step1 = subprocess.Popen(
                [
                    'python', 'run_replicas.py',
                    '--host', '127.0.0.1',
                    '--worker-ports', '8001,8002',
                    '--gpu-indices', '0,1',
                    '--model-name', 'Qwen/Qwen2.5-1.5B-Instruct',
                    '--no-enable-prefix-caching'
                ],
                stdout=open(step1_log, 'w'),
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for step1 to complete startup
            if not monitor_log(step1_log, STEP1_SUCCESS_MSG):
                print(f"Step 1 failed for {strategy} run {run_num}")
                terminate_process(proc_step1)
                continue
            
            # Step 2: Start router
            step2_log = os.path.join(run_dir, 'router.log')
            proc_step2 = subprocess.Popen(
                [
                    'python', 'router.py',
                    '--worker-ports', '8001,8002',
                    '--port', '8000',
                    '--strategy', strategy
                ],
                stdout=open(step2_log, 'w'),
                stderr=subprocess.STDOUT,
                text=True
            )
            time.sleep(5)  # Wait for router to start
            
            # Step 3: Run benchmark
            step3_log = os.path.join(run_dir, 'bench.log')
            with open(step3_log, 'w') as f:
                proc_step3 = subprocess.Popen(
                    [
                        'python3', '-m', 'bench_serving',
                        '--backend', 'vllm',
                        '--host', '127.0.0.1',
                        '--port', '8000',
                        '--dataset-name', 'sharegpt',
                        '--num-prompts', '1024',
                        '--sharegpt-output-len', '256',
                        '--max-concurrency', '32',
                        '--dataset-path', './long_short_pattern_sharegpt_requests.json'
                    ],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                proc_step3.wait()
            
            # Verify step3 success
            with open(step3_log, 'r') as f:
                if STEP3_SUCCESS_MSG not in f.read():
                    print(f"Step 3 failed for {strategy} run {run_num}")
            
            # Cleanup processes
            terminate_process(proc_step2)
            terminate_process(proc_step1)
            time.sleep(2)  # Allow ports to release

def monitor_log(log_path, success_msg, timeout=60, check_interval=1):
    """Monitor log file for success message"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                if success_msg in content:
                    return True
        except FileNotFoundError:
            pass
        time.sleep(check_interval)
    return False

def terminate_process(proc):
    """Terminate a process gracefully"""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == "__main__":
    root_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(root_dir, exist_ok=True)
    run_experiment(root_dir)
    print("All experiments completed!")