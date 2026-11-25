#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLVU 数据集自动下载脚本（Hugging Face Hub）

用法:
    python download_mlvu.py                              # 默认下载 MLVU/MVLU 全部
    python download_mlvu.py MLVU/MVLU                    # 指定数据集名称
    python download_mlvu.py MLVU/MVLU --subset needle plotQA
    python download_mlvu.py MLVU/MVLU --endpoint https://hf-mirror.com
    python download_mlvu.py MLVU/MVLU --list            # 只查看可用子集
"""

import argparse
import os
import sys
import tarfile
import zipfile
from pathlib import Path

ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz")


def parse_args():
    p = argparse.ArgumentParser(description="自动下载 MLVU 数据集（支持子集、断点续传、自动解压）")
    p.add_argument("dataset", nargs="?", default="MLVU/MVLU",
                   help="数据集名称（HF 仓库名），默认 MLVU/MVLU")
    p.add_argument("--subset", nargs="*", default=None,
                   help="要下载的子任务名，如 needle plotQA；不填则全部下载")
    p.add_argument("--root", default="./mlvu", help="保存目录（默认 ./mlvu）")
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    p.add_argument("--revision", default=None, help="指定仓库版本/分支")
    p.add_argument("--endpoint", default=None, help="镜像地址，如 https://hf-mirror.com")
    p.add_argument("--token", default=None, help="HF 令牌，也可用环境变量 HF_TOKEN")
    p.add_argument("--workers", type=int, default=4, help="并发下载线程数")
    p.add_argument("--meta-only", action="store_true", help="只下载标注/说明文件，不下载视频压缩包")
    p.add_argument("--no-extract", action="store_true", help="下载后不解压")
    p.add_argument("--keep-zip", action="store_true", help="解压后保留压缩包")
    p.add_argument("--list", action="store_true", help="仅列出可用子集，不下载")
    return p.parse_args()


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:,.1f} {u}"
        n /= 1024
    return f"{n:,.1f} PB"


def main():
    args = parse_args()

    # HF_ENDPOINT 必须在 import huggingface_hub 之前设置
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit("缺少依赖，请先执行: pip install -U huggingface_hub")

    token = args.token or os.environ.get("HF_TOKEN")

    # 1) 列出仓库文件（含大小）
    try:
        api = HfApi(token=token)
        info = api.repo_info(args.dataset, repo_type=args.repo_type,
                             revision=args.revision, files_metadata=True)
    except Exception as e:
        sys.exit(f"无法访问仓库 {args.dataset}: {e}\n"
                 "若网络受限，可加 --endpoint https://hf-mirror.com 重试")

    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    archives = [(n, sz) for n, sz in files if n.lower().endswith(ARCHIVE_EXTS)]
    metas = [(n, sz) for n, sz in files if not n.lower().endswith(ARCHIVE_EXTS)]

    # 2) --list：仅展示
    if args.list:
        print(f"仓库 {args.dataset} 可用子集（压缩包）:")
        for n, sz in archives:
            print(f"  {Path(n).stem:<20} {human(sz):>12}   ({n})")
        if metas:
            print("其他文件（标注/说明，默认一并下载）:")
            for n, sz in metas:
                print(f"  {n:<40} {human(sz)}")
        return

    # 3) 选择子集
    if args.meta_only:
        selected = []
    elif args.subset:
        selected, missing = [], []
        for w in [s.lower() for s in args.subset]:
            hits = [(n, sz) for n, sz in archives
                    if Path(n).stem.lower() == w or w in Path(n).stem.lower()]
            if hits:
                selected += hits
            else:
                missing.append(w)
        if missing:
            avail = ", ".join(Path(n).stem for n, _ in archives)
            sys.exit(f"未找到子集: {', '.join(missing)}\n可用子集: {avail}\n（--list 可查看详情）")
    else:
        selected = archives

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    total = sum(sz for _, sz in selected) + sum(sz for _, sz in metas)
    print(f"开始下载: {args.dataset} ({args.repo_type})")
    print(f"保存目录: {root}")
    print(f"共 {len(selected)} 个压缩包 + {len(metas)} 个标注/说明文件，约 {human(total)}")
    for n, sz in selected:
        print(f"  - {n}  {human(sz)}")

    # 4) 下载（snapshot_download 自带断点续传）
    patterns = [n for n, _ in selected] + [n for n, _ in metas]
    snapshot_download(
        repo_id=args.dataset,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=str(root),
        allow_patterns=patterns,
        max_workers=args.workers,
        token=token,
    )
    print("下载完成。")

    # 5) 解压（统一解到 root，保持 zip 内部目录结构，与官方标注路径一致）
    if args.no_extract or not selected:
        return
    print("开始解压 ...")
    for n, _ in selected:
        arc = root / n
        if not arc.exists():
            print(f"  [跳过] 未找到 {arc}")
            continue
        try:
            if arc.suffix == ".zip":
                with zipfile.ZipFile(arc) as zf:
                    zf.extractall(root)
            else:
                with tarfile.open(arc) as tf:
                    try:
                        tf.extractall(root, filter="data")
                    except TypeError:
                        tf.extractall(root)
            print(f"  已解压 {arc.name}")
            if not args.keep_zip:
                arc.unlink()
        except Exception as e:
            print(f"  [错误] 解压 {arc.name} 失败: {e}（压缩包已保留）")

    print(f"全部完成 ✅ 数据位于: {root}")


if __name__ == "__main__":
    main()