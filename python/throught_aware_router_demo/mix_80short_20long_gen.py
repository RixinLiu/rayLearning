import json
import random
import os
from tqdm import tqdm
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
import requests

# 定义 ShareGPT 数据集的 URL
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"

def download_and_cache_file(url: str, filename: Optional[str] = None):
    """从 URL 下载并缓存文件。"""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # 检查缓存文件是否已经存在
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # 流式下载并显示进度条
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 检查请求错误

    # 文件的总大小（字节）
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # 每次下载 1KB

    # 使用 tqdm 显示进度条
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename

def sample_sharegpt_requests_with_ratio(
    dataset_path: str,
    short_ratio: float = 0.8,
    fixed_output_len: Optional[int] = None,
) -> List[Dict]:
    """从 ShareGPT 数据集中生成 short 占比 short_ratio，long 占比 (1 - short_ratio) 的数据集。"""
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # 下载 ShareGPT 数据集（如果必要）
    if not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # 加载数据集
    with open(dataset_path) as f:
        dataset = json.load(f)

    # 过滤掉对话轮次少于 2 的对话
    old_dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # 只保留每个对话的前两轮
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in old_dataset
    ]

    # 过滤掉过长或过短的序列
    short_requests = []
    long_requests = []
    print("length of dataset: ", len(dataset))
    for i in range(len(dataset)):
        # 获取提示和补全
        prompt = dataset[i][0]
        completion = dataset[i][1]

        # 计算输入和输出的长度
        prompt_len = len(prompt.split())

        if prompt_len < 50:
            short_requests.append(old_dataset[i])
        elif prompt_len > 1000:
            long_requests.append(old_dataset[i])

    print(f"Found {len(short_requests)} short requests and {len(long_requests)} long requests.")

    # 根据比例计算需要的 short 和 long 的数量
    num_short = len(short_requests)
    num_long = len(long_requests)
    total_requests = num_short + num_long

    if (num_long < int(total_requests * (1 - short_ratio))):
        print(f"Not enough long requests to meet the ratio. Adjusting the number of short requests.")
        num_long = len(long_requests)
        num_short = num_long * int(short_ratio / (1 - short_ratio))  # Adjust num_short to maintain the ratio

    # 随机抽取 short 和 long 请求
    sampled_short_requests = random.sample(short_requests, num_short)
    sampled_long_requests = random.sample(long_requests, num_long)

    # 合并并打乱数据集
    filtered_dataset = sampled_short_requests + sampled_long_requests
    random.shuffle(filtered_dataset)

    print(f"Sampled {len(filtered_dataset)} requests: {len(sampled_short_requests)} short and {len(sampled_long_requests)} long.")

    output_file = "short_80_long_20_sharegpt_requests.json"
    with open(output_file, "w") as f:
        json.dump(filtered_dataset, f, indent=4)
    print(f"Sampled requests saved to {output_file}")

    return filtered_dataset

if __name__ == "__main__":
    sampled_requests = sample_sharegpt_requests_with_ratio(
        dataset_path="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
        short_ratio=0.8,
        fixed_output_len=256
    )