import json
from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse
import httpx
import asyncio
import argparse
import time
from typing import List, Dict, Optional
from collections import deque
import signal
import random

app = FastAPI()

# Global config
worker_urls = []
metrics_cache: Dict[str, Dict] = {}
latency_history: Dict[str, deque] = {}
worker_long_request_count = {"http://localhost:8001": 0, "http://localhost:8002": 0}
worker_short_request_count = {"http://localhost:8001": 0, "http://localhost:8002": 0}
strategy_config = {
    "current_strategy": "tokens",  # default strategy
    "history_size": 100,          # num of recent latency values to keep
    "metric_expiry": 10           # metric expiration time(s)
}
last_access_worker = 0

# Add manual tokens tracking
manual_tokens = {}  # {worker_url: {"tokens": int, "timestamp": float}}

def initialize_config(ports: List[int], strategy: str):
    global worker_urls, latency_history, manual_tokens
    worker_urls = [f"http://localhost:{port}" for port in ports]
    latency_history = {url: deque(maxlen=strategy_config["history_size"]) for url in worker_urls}
    strategy_config["current_strategy"] = strategy
    manual_tokens = {url: {"tokens": 0, "last_100ms_tokens": 0, "throughput": 0, "timestamp": time.time()} for url in worker_urls}
    print(f"Initialized with strategy: {strategy}")

def update_manual_tokens(worker_url: str, delta: int):
    """Update manual tokens count and calculate throughput"""
    current_time = time.time()

    manual_tokens[worker_url]["tokens"] += delta
    manual_tokens[worker_url]["timestamp"] = current_time

async def monitor_manual_tokens():
    """Monitor and print manual_tokens every 100ms"""
    while True:
        await asyncio.sleep(1.5)  # 100ms interval
        for worker_url, data in manual_tokens.items():
            # Calculate tokens processed in last 100ms window
            tokens_in_window = data["tokens"] - data["last_100ms_tokens"]
            manual_tokens[worker_url]["throughput"] = tokens_in_window
            manual_tokens[worker_url]["last_100ms_tokens"] = data["tokens"]

previous_tokens = {}
async def metrics_updater():
    global previous_tokens
    while True:
        for worker_url in worker_urls:
            try:
                async with httpx.AsyncClient() as client:
                    # Get metrics from each worker, and cache them
                    # Metrics are maintained by vLLM
                    response = await client.get(f"{worker_url}/metrics", timeout=2)
                    if response.status_code == 200:
                        metrics = parse_metrics(response.text)
                        metrics_cache[worker_url] = {
                            "metrics": metrics,
                            "timestamp": time.time()
                        }
            except Exception as e:
                print(f"Metrics update failed for {worker_url}: {str(e)}")
        # await asyncio.sleep(1)

def parse_metrics(text: str) -> Dict[str, float]:
    """Parse Prometheus format metrics"""
    metrics = {}
    for line in text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return metrics

async def track_latency(worker_url: str, start_time: float):
    """Track request latency"""
    latency = time.time() - start_time
    latency_history[worker_url].append(latency)

def select_worker() -> Optional[str]:
    """Main routing logic"""
    global last_access_worker
    valid_workers = get_valid_workers()
    if not valid_workers:
        return None
    
    strategy = strategy_config["current_strategy"]

    # 添加请求计数器
    if not hasattr(select_worker, "request_count"):
        select_worker.request_count = 0
    select_worker.request_count += 1
    
    if strategy == "tokens" and select_worker.request_count <= 100:
        worker = valid_workers[last_access_worker % len(valid_workers)]
        last_access_worker += 1
        
        # 打印warm start信息
        print(f"Warm start ({select_worker.request_count}/100): Round-robin selected worker: {worker}")
        return worker
    elif strategy == "tokens":
        # Modified to use manual token throughput
        min_throughput = float('inf')
        selected_worker = None
        for w in valid_workers:
            throughput = manual_tokens[w]["throughput"]
            tokens = manual_tokens[w]["tokens"]
            print(f"Worker: {w}, Tokens: {tokens}")
            print(f"Worker: {w}, Throughput: {throughput:.2f} tokens/s")
            if throughput < min_throughput:
                min_throughput = throughput
                selected_worker = w
        print(f"Selected worker: {selected_worker}")
        return selected_worker
    elif strategy == "pow_2":
        min_qlen = float('inf')
        for w in valid_workers:
            qlen = get_queue_len(w)
            print(f"Worker: {w}, qlen: {qlen}")
            if qlen == min_qlen:
                worker = random.choice(valid_workers)
            elif qlen < min_qlen:
                min_qlen = qlen
                worker = w
            tokens = manual_tokens[w]["tokens"]
            print(f"Worker: {w}, Tokens: {tokens}")
        print(f"Selected worker: {worker}")
    elif strategy == "random":
        worker = random.choice(valid_workers)
        for w in valid_workers:
            tokens = manual_tokens[w]["tokens"]
            print(f"Worker: {w}, Tokens: {tokens}")
        print(f"Selected worker: {worker}")
    elif strategy == "latency":
        latencies = [get_avg_latency(w) for w in valid_workers]
        min_latency = min(latencies)
        candidates = [w for w in valid_workers if get_avg_latency(w) <= min_latency * 1.1]  # Allow 10% tolerance
        # Use round-robin among candidates
        worker = candidates[last_access_worker % len(candidates)]
        last_access_worker += 1
    elif strategy == "cache_hit":
        max_hit = max(get_cache_hit_rate(w) for w in valid_workers)
        candidates = [w for w in valid_workers if get_cache_hit_rate(w) == max_hit]
        worker = random.choice(candidates)
    elif strategy == "round_robin":
        for w in valid_workers:
            tokens = manual_tokens[w]["tokens"]
            print(f"Worker: {w}, Tokens: {tokens}")
        worker = valid_workers[last_access_worker % len(valid_workers)]
        print(f"Selected worker: {worker}")
        last_access_worker += 1
    elif strategy == "only_one":
        worker = valid_workers[0] 
        print(f"Selected worker: {worker}")

    return worker

def get_valid_workers() -> List[str]:
    """Return a list of valid worker URLs based on the freshness of their metric"""
    now = time.time()
    return [
        url for url in worker_urls 
        if (now - metrics_cache.get(url, {}).get("timestamp", 0)) < strategy_config["metric_expiry"]
    ]

def get_total_tokens(worker_url: str) -> float:
    """Total tokens processed by the worker"""
    metrics = metrics_cache.get(worker_url, {}).get("metrics", {})
    return metrics.get("vllm:prompt_tokens_total{model_name=\"Qwen/Qwen2.5-1.5B-Instruct\"}", 0) + metrics.get("vllm:generation_tokens_total{model_name=\"Qwen/Qwen2.5-1.5B-Instruct\"}", 0)

def get_queue_len (worker_url: str) -> int:
    """Return the current queue length of the worker"""
    metrics = metrics_cache.get(worker_url, {}).get("metrics", {})
    num_running = metrics.get("vllm:num_requests_running{model_name=\"Qwen/Qwen2.5-1.5B-Instruct\"}", 0)
    num_waiting = metrics.get("vllm:num_requests_waiting{model_name=\"Qwen/Qwen2.5-1.5B-Instruct\"}", 0)
    print(f"Worker: {worker_url}, Running: {num_running}, Waiting: {num_waiting}")
    return num_running

def get_avg_latency(worker_url: str) -> float:
    """Return average request latency in the recent history"""
    history = latency_history.get(worker_url, deque())
    return sum(history)/len(history) if history else float('inf')

def get_cache_hit_rate(worker_url: str) -> float:
    metrics = metrics_cache.get(worker_url, {}).get("metrics", {})
    return metrics.get("vllm:gauge_gpu_prefix_cache_hit_rate", -1)

is_first_try = True
async def forward_request(request: Request, endpoint: str):
    # print("Forwarding request...")
    start_time = time.time()
    # print(f"start_time = {start_time}")

    # Get word count first
    word_count = 0
    try:
        body = await request.body()
        request_body = json.loads(body) if body else None
        if request_body and 'prompt' in request_body:
            prompt = request_body['prompt']
            word_count = len(prompt.split())
    except:
        pass

    worker_url = select_worker()
    
    if not worker_url:
        return Response(
            content=json.dumps({"error": "No available workers"}),
            status_code=503,
            media_type="application/json"
        )

    if word_count > 1000:
        word_count += 400
    else:
        word_count += 50
    
    # Update manual tokens before processing
    update_manual_tokens(worker_url, word_count)

    try:
        body = await request.body()
        request_body = json.loads(body) if body else None
        
        # Call different function based on request method
        async with httpx.AsyncClient() as client:
            if request.method == "POST":
                response = await client.post(
                    f"{worker_url}{endpoint}",
                    json=request_body,
                    timeout=60
                )
            else:
                response = await client.get(
                    f"{worker_url}{endpoint}",
                    timeout=60
                )
            
            # Track latency
            await track_latency(worker_url, start_time)
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type", "application/json")
            )
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )

async def forward_streaming_request(request: Request, endpoint: str):
    # print("Forwarding streaming request...")
    # estimated_prompt_len = 0
    
    """Handle streaming request, where the response is sent back in chunks"""
    body = await request.body()
    request_body = json.loads(body) if body else None
    word_count = 0


    # Calculate word count using split
    if request_body and 'prompt' in request_body:
        prompt = request_body['prompt']
        word_count = len(prompt.split())
        # print(f"Word count in request: {word_count}")

    worker_url = select_worker()

    if word_count > 1000:
        word_count += 1000
    else:
        word_count += 50

    # Update manual tokens before processing
    update_manual_tokens(worker_url, word_count)
    
    if not worker_url:
        return Response(
            content=json.dumps({"error": "No available workers"}),
            status_code=503,
            media_type="application/json"
        )

    try:
        
        async def generate_stream():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{worker_url}{endpoint}",
                    json=request_body,
                    timeout=120
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
        
        # await track_latency(worker_url, start_time)
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )

@app.post("/v1/completions")
async def handle_completions(request: Request):
    try:
        body = await request.json()
        # print(body)
        if body.get("stream", False):
            return await forward_streaming_request(request, "/v1/completions")
    except:
        pass
    return await forward_request(request, "/v1/completions")

@app.on_event("shutdown")
async def shutdown():
    print("\nShutting down... Final worker request counts:")
    print(json.dumps(worker_long_request_count, indent=4))
    print(json.dumps(worker_short_request_count, indent=4))

# Handle `ctrl+c` gracefully
def handle_sigint(sig, frame):
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(shutdown())
    loop.stop()

signal.signal(signal.SIGINT, handle_sigint)

@app.get("/v1/models")
async def handle_models(request: Request):
    return await forward_request(request, "/v1/models")

@app.on_event("startup")
async def startup():
    asyncio.create_task(metrics_updater())
    asyncio.create_task(monitor_manual_tokens())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-ports", type=str, required=True,
                       help="Comma-separated list of worker ports")
    parser.add_argument("--port", type=int, required=True,
                       help="Router listening port")
    parser.add_argument("--strategy", type=str, required=True,
            choices=["tokens", "pow_2", "latency", "cache_hit", "round_robin", "random", "only_one"],
                       help="Routing strategy")
    args = parser.parse_args()
    
    ports = list(map(int, args.worker_ports.split(',')))
    initialize_config(ports, args.strategy)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
