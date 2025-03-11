import json
import os
from fastapi import FastAPI, Request, Response, HTTPException
from starlette.responses import StreamingResponse
import httpx
import asyncio
import argparse
import time
from typing import List, Dict

app = FastAPI()

worker_urls = []
metrics_cache: Dict[str, Dict] = {}  # {worker_url: {"total_tokens": int, "last_updated": float}}

def initialize_worker_urls(ports: List[int]):
    """Initialize the worker URLs from the provided ports."""
    global worker_urls
    worker_urls = [f"http://localhost:{port}" for port in ports]
    print(f"Initialized worker URLs: {worker_urls}")

async def update_metrics_cache():
    """Periodically update metrics cache from all workers."""
    while True:
        global worker_urls, metrics_cache
        for worker_url in worker_urls:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{worker_url}/metrics", timeout=2.0)
                    if response.status_code == 200:
                        prompt_tokens = 0.0
                        generation_tokens = 0.0
                        for line in response.text.split('\n'):
                            line = line.strip()
                            if line.startswith('#') or not line:
                                continue
                            if line.startswith('vllm:prompt_tokens_total'):
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        prompt_tokens += float(parts[-1])
                                    except ValueError:
                                        pass
                            elif line.startswith('vllm:generation_tokens_total'):
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        generation_tokens += float(parts[-1])
                                    except ValueError:
                                        pass
                        total = int(prompt_tokens + generation_tokens)
                        metrics_cache[worker_url] = {
                            "total_tokens": total,
                            "last_updated": time.time()
                        }
            except Exception as e:
                print(f"Error updating metrics for {worker_url}: {str(e)}")
                metrics_cache[worker_url] = {
                    "total_tokens": float('inf'),
                    "last_updated": time.time()
                }
        await asyncio.sleep(1)  # Update every 1 second

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_metrics_cache())

def select_worker() -> str:
    """Select worker with the least processed tokens."""
    global worker_urls, metrics_cache
    current_time = time.time()
    min_tokens = float('inf')
    selected_worker = None

    for worker_url in worker_urls:
        metrics = metrics_cache.get(worker_url, {})
        # Skip metrics older than 10 seconds
        if current_time - metrics.get('last_updated', 0) > 10:
            continue
        if metrics.get('total_tokens', float('inf')) < min_tokens:
            min_tokens = metrics['total_tokens']
            selected_worker = worker_url

    return selected_worker or worker_urls[0] if worker_urls else None

async def forward_request(request: Request, endpoint: str):
    """Forward request to the worker with least tokens."""
    worker_url = select_worker()
    if not worker_url:
        return Response(
            content=json.dumps({"error": "No available workers"}),
            status_code=503,
            media_type="application/json"
        )

    try:
        request_body = await request.json() if await request.body() else None
        async with httpx.AsyncClient() as client:
            if request_body:
                response = await client.post(f"{worker_url}{endpoint}", json=request_body, timeout=60.0)
            else:
                response = await client.get(f"{worker_url}{endpoint}", timeout=60.0)

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
    """Forward streaming request to the worker with least tokens."""
    worker_url = select_worker()
    if not worker_url:
        return Response(
            content=json.dumps({"error": "No available workers"}),
            status_code=503,
            media_type="application/json"
        )

    try:
        request_body = await request.json() if await request.body() else None
        if request_body and "stream" not in request_body:
            request_body["stream"] = True

        async def stream_generator():
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{worker_url}{endpoint}", json=request_body, timeout=120.0) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )

@app.post("/v1/completions")
async def completions(request: Request):
    try:
        body = await request.json()
        if body.get("stream", False):
            return await forward_streaming_request(request, "/v1/completions")
    except:
        pass
    return await forward_request(request, "/v1/completions")

@app.get("/v1/models")
async def models(request: Request):
    all_models = []
    async with httpx.AsyncClient() as client:
        for worker_url in worker_urls:
            try:
                response = await client.get(f"{worker_url}/v1/models", timeout=10.0)
                if response.status_code == 200:
                    worker_models = response.json()
                    if "data" in worker_models:
                        all_models.extend(worker_models["data"])
            except:
                continue
    return {"object": "list", "data": all_models}

@app.get("/health")
async def health():
    available = []
    async with httpx.AsyncClient() as client:
        for worker_url in worker_urls:
            try:
                response = await client.get(f"{worker_url}/health", timeout=2.0)
                if response.status_code == 200:
                    available.append(worker_url)
            except:
                continue
    return {
        "status": "healthy" if available else "unhealthy",
        "available_workers": available,
        "worker_count": len(available)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-ports", nargs="+", type=int, required=True,
                       help="List of worker ports")
    args = parser.parse_args()
    initialize_worker_urls(args.worker_ports)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)