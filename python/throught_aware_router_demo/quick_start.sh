#!/bin/bash

# 检查是否提供了策略参数
if [ -z "$1" ]; then
    echo "Usage: $0 <strategy>"
    echo "Available strategies: tokens, latency, cache_hit"
    exit 1
fi

strategy=$1

# 启动vLLM服务副本
echo "Starting vLLM workers..."
python run_replicas.py --host 127.0.0.1 --worker-ports 8001,8002 --gpu-indices 0,1 --model-name "Qwen/Qwen2.5-1.5B-Instruct" --enable-prefix-caching 

python run_replicas.py \
    --host 127.0.0.1 \
    --worker-ports 8001,8002 \
    --gpu-indices 0,1 \
    --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
    --enable-prefix-caching \
    > run_replicas.log 2>&1 &

# 等待workers启动
echo "Waiting 60 seconds for workers to start..."
sleep 60

# 启动三个不同策略的路由器
strategies=("tokens" "latency" "cache_hit")

# 启动指定策略的路由器
echo "Starting ${strategy} strategy router on port 8000..."
python router.py --worker-ports 8001,8002 --port 8000 --strategy ${strategy}
python router.py \
    --worker-ports 8001,8002 \
    --port 8000 \
    --strategy ${strategy} \
    > router_${strategy}.log 2>&1 &

# 等待路由器启动
echo "Waiting 20 seconds for router to start..."
sleep 20

echo "Starting benchmark for ${strategy} strategy..."
python3 -m bench_serving --backend vllm --host 127.0.0.1 --port 8000 --dataset-name sharegpt --num-prompts 1024 --sharegpt-output-len 256 --max-concurrency 32 --dataset-path /root/rayLearning/python/throught_aware_router_demo/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json

python3 -m bench_serving --backend vllm --host 127.0.0.1 --port 8000 \
    --dataset-name sharegpt --num-prompts 1024 --sharegpt-output-len 256 --max-concurrency 32

echo "Benchmark for ${strategy} strategy done, check the log file for details."

# 保持脚本运行
wait