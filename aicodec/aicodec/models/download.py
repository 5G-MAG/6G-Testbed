#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 HuggingFace 模型权重
依赖: pip install -U huggingface_hub hf_transfer

用法:
    python download.py                                          # 默认模型
    python download.py --repo moonshotai/Kimi-K2.5              # 指定仓库
    python download.py --repo Qwen/Qwen3-8B --name Qwen3-8B     # 指定仓库名和保存名
    python download.py --repo Qwen/Qwen3-8B --min-free-gb 20    # 小模型调低磁盘阈值
"""

import argparse
import os
import sys
import shutil

# ========== 默认配置（可被命令行参数覆盖） ==========
DEFAULT_REPO = "Qwen/Qwen3.6-35B-A3B"                          # 仓库 ID
DEFAULT_LOCAL_DIR = os.path.expanduser("./Qwen3.6-35B-A3B")  # 保存路径

# 国内机器走镜像加速（海外机器注释掉）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 启用高速分块下载（新版 huggingface_hub 用 Xet；旧版可改回
# HF_HUB_ENABLE_HF_TRANSFER=1，需 pip install hf_transfer）
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

# 如需鉴权则填 token（https://huggingface.co/settings/tokens）
# os.environ["HF_TOKEN"] = "hf_xxxxxxxx"
# ============================

from huggingface_hub import snapshot_download


def check_disk(local_dir: str, min_free_gb: int = 1500):
    """K2 系列权重约 1TB+，务必预留磁盘"""
    target = os.path.dirname(local_dir) or "."
    os.makedirs(target, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / 1024 ** 3
    print(f"[磁盘检查] 剩余可用空间: {free_gb:.0f} GB")
    if free_gb < min_free_gb:
        print(f"[警告] 空间不足 {min_free_gb} GB")
        sys.exit(1)


def parse_args():
    ap = argparse.ArgumentParser(
        description="从 HuggingFace（镜像）下载模型权重")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="模型仓库 ID（默认: %(default)s）")
    ap.add_argument("--name", default=None,
                    help="保存的模型目录名；缺省：取 repo 末段")
    ap.add_argument("--min-free-gb", type=int, default=1500,
                    help="磁盘剩余空间检查阈值 GB（默认: %(default)s）")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_id = args.repo
    if args.name:
        local_dir = os.path.expanduser(
            os.path.join("./models", args.name))
    elif args.repo != DEFAULT_REPO:
        local_dir = os.path.expanduser(
            os.path.join("./models", repo_id.rsplit("/", 1)[-1]))
    else:
        local_dir = DEFAULT_LOCAL_DIR

    check_disk(local_dir, args.min_free_gb)
    print(f"开始下载: {repo_id}")
    print(f"保存目录: {local_dir}\n")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[
            "*.json", "*.safetensors", "*.txt",
            "tokenizer*", "*.model",
        ],
        max_workers=8,   # 并发线程数，可按带宽调整
    )
    print(f"\n下载完成: {path}")


if __name__ == "__main__":
    main()