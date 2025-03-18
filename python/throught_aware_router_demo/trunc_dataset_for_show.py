import json
import random
import os
from tqdm import tqdm
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

# 定义 ShareGPT 数据集的 URL
SHAREGPT_URL = "https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"

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

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer=None,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    """从 ShareGPT 数据集中随机抽取一定数量的请求。"""
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # 下载 ShareGPT 数据集（如果必要）
    if not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # 加载数据集
    with open(dataset_path) as f:
        dataset = json.load(f)

    # 过滤掉对话轮次少于 2 的对话
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # 只保留每个对话的前两轮
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # 打乱数据集
    random.shuffle(dataset)

    # 过滤掉过长或过短的序列
    filtered_dataset = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # 获取提示和补全
        prompt = dataset[i][0]
        completion = dataset[i][1]

        # 计算输入和输出的长度
        prompt_len = len(prompt.split())
        output_len = len(completion.split()) if fixed_output_len is None else fixed_output_len

        if prompt_len < 4 or output_len < 4:
            # 过滤掉过短的序列
            continue
        if prompt_len > 1024 or (prompt_len + output_len > 2048 and fixed_output_len is None):
            # 过滤掉过长的序列
            continue

        filtered_dataset.append({
            "prompt": prompt,
            "completion": completion,
            "prompt_len": prompt_len,
            "output_len": output_len
        })

    print(f"Sampled {len(filtered_dataset)} requests.")
    return filtered_dataset

def save_sampled_requests_to_json(sampled_requests, output_file):
    """将采样后的请求保存到 JSON 文件中。"""
    with open(output_file, "w") as f:
        json.dump(sampled_requests, f, indent=4)
    print(f"Sampled requests saved to {output_file}")

if __name__ == "__main__":
    # 设置输出文件名
    output_file = "sampled_sharegpt_requests.json"

    # 从 ShareGPT 数据集中采样 100 个请求
    sampled_requests = sample_sharegpt_requests(
        dataset_path="ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
        num_requests=100
    )

    # 将采样后的请求保存到 JSON 文件中
    save_sampled_requests_to_json(sampled_requests, output_file)