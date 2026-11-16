"""Unified MMBench MCQ evaluation for Qwen3-VL over a self-contained data folder.

A data folder must hold the images plus the question manifest, e.g.

    <data_dir>/
        manifest.jsonl      # index / question / hint / choices / answer
        241.jpg
        252.jpg  ...

Baseline and every One-D-Piece reconstruction condition are therefore the same
kind of input: just a different folder.

    python mmbench_eval.py --data-dir .../images/MMBench_DEV_EN \
        --output-dir results/baseline
    # --model defaults to <repo_root>/aicodec/aicodec/models/Qwen3_VL_30B_A3B_Instruct,
    # resolved relative to this script; pass --model to override.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CHOICE_KEYS = ("A", "B", "C", "D")
# <repo_root>/aicodec/aicodec/models/Qwen3_VL_30B_A3B_Instruct,
# resolved relative to this script (aitestbed/tools -> repo root) so it works from any cwd.
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "aicodec" / "aicodec" / "models" / "Qwen3_VL_30B_A3B_Instruct"
)
ANSWER_PATTERN = re.compile(r"(?<![A-Z])([A-D])(?![A-Z])", re.IGNORECASE)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def resolve_manifest(data_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            sys.exit(f"[FATAL] manifest not found: {path}")
        return path
    for name in ("manifest.jsonl", "questions.jsonl", "manifest.json"):
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    sys.exit(f"[FATAL] no manifest.jsonl inside {data_dir}; pass --manifest")


def find_image(data_dir: Path, index: str) -> Path | None:
    # Standard naming: 241.jpg, 241.png, ...
    for suffix in IMAGE_SUFFIXES:
        candidate = data_dir / f"{index}{suffix}"
        if candidate.is_file():
            return candidate
    # One-D-Piece recon naming: img_00241_241.png
    idx_padded = f"{int(index):05d}"
    for suffix in IMAGE_SUFFIXES:
        candidate = data_dir / f"img_{idx_padded}_{index}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_records(data_dir: Path, manifest: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    # utf-8-sig tolerates a BOM from manifests touched by Windows editors.
    with manifest.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            index = str(raw["index"]).strip()
            # MMBench rows may expose only A/B; drop blanks so the prompt never
            # shows empty options and the letter hint matches real choices.
            choices = {
                key: str(raw.get("choices", {}).get(key, "")).strip()
                for key in CHOICE_KEYS
            }
            choices = {key: value for key, value in choices.items() if value}
            answer = str(raw.get("answer", "")).strip().upper()
            if not choices or answer not in choices:
                sys.exit(f"[FATAL] manifest line {line_number}: bad answer {answer!r}")
            image = find_image(data_dir, index)  ## match the image file
            if image is None:
                missing.append(index)
                continue
            records.append({
                "index": index,
                "image_path": str(image),
                "question": str(raw.get("question", "")).strip(),
                "hint": str(raw.get("hint", "")).strip(),
                "choices": choices,
                "answer": answer,
                "category": str(raw.get("category", "")).strip(),
                "l2_category": str(raw.get("l2_category", "")).strip(),
            })
            if limit and len(records) >= limit:
                break
    if missing:
        sys.exit(f"[FATAL] {len(missing)} manifest rows have no image in {data_dir}; "
                 f"first: {missing[:10]}")
    if not records:
        sys.exit(f"[FATAL] no usable records from {manifest}")
    return records


# --------------------------------------------------------------------------
# prompting / scoring
# --------------------------------------------------------------------------
def format_question(record: dict[str, Any]) -> str:
    parts = []
    if record["hint"]:
        parts.append(f"Context: {record['hint']}")
    parts.append(f"Question: {record['question']}")
    labels = [key for key in CHOICE_KEYS if key in record["choices"]]
    for label in labels:
        parts.append(f"{label}. {record['choices'][label]}")
    if len(labels) > 2:
        allowed = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    elif len(labels) == 2:
        allowed = f"{labels[0]} or {labels[1]}"
    else:
        allowed = labels[0]
    parts.append(f"Answer with only one letter: {allowed}.")
    return "\n".join(parts)


def extract_choice(model_output: str, labels: list[str]) -> str | None:
    """First standalone A-D token that is actually offered by the question."""
    for match in ANSWER_PATTERN.finditer((model_output or "").strip()):
        letter = match.group(1).upper()
        if letter in labels:
            return letter
    return None


def summarize(records: list[dict[str, Any]], baseline: dict[str, bool] | None) -> dict[str, Any]:
    total = len(records)
    correct = sum(item["correct"] for item in records)
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in records:
        bucket = by_category[item.get("category") or "unknown"]
        bucket["total"] += 1
        bucket["correct"] += int(item["correct"])
    summary = {
        "total": total,
        "correct": correct,
        "raw_accuracy": correct / total if total else 0.0,
        "invalid_predictions": sum(item["prediction"] is None for item in records),
        "by_category": {
            key: {**value, "accuracy": value["correct"] / value["total"]}
            for key, value in sorted(by_category.items())
        },
    }
    if baseline:
        subset = [item for item in records if baseline.get(item["index"])]
        summary["filtered_accuracy"] = {
            "baseline_correct_total": len(subset),
            "still_correct": sum(item["correct"] for item in subset),
            "accuracy": (sum(item["correct"] for item in subset) / len(subset)) if subset else 0.0,
        }
    return summary


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def build_vllm_input(record: dict[str, Any], processor) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{record['image_path']}"},
            {"type": "text", "text": format_question(record)},
        ],
    }]
    ## Convert the messages list into the plain-text sequence the model expects
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ signature, mirroring the RealWorldQA pipeline.
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def run_inference(records: list[dict[str, Any]], args) -> list[str]:
    # vLLM V1 launches EngineCore in a subprocess. With the default "fork"
    # method it dies if the parent already holds a CUDA context, so force
    # "spawn" before vllm is imported.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from tqdm import tqdm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    # Build the engine FIRST: image preprocessing below can touch CUDA, and a
    # CUDA context in this process would break the engine's subprocess launch.
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,  ## load the model
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
    )
    inputs = [
        build_vllm_input(record, processor)
        for record in tqdm(records, desc="Building prompts")
    ]
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, seed=args.seed,
    )
    outputs = llm.generate(inputs, sampling_params=sampling)
    return [output.outputs[0].text.strip() for output in outputs]


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def load_baseline_flags(path: str | None) -> dict[str, bool] | None:
    if not path:
        return None
    source = Path(path)
    if source.is_dir():
        source = source / "predictions.jsonl"
    if not source.is_file():
        sys.exit(f"[FATAL] baseline predictions not found: {source}")
    flags: dict[str, bool] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                flags[str(row["index"])] = bool(row["correct"])
    return flags


def save_results(records, summary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    fields = ["index", "image_path", "answer", "prediction", "correct", "model_output", "category"]
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True,
                        help="folder holding images AND the question manifest")
    parser.add_argument("--manifest", default=None,
                        help="override manifest path (default: <data-dir>/manifest.jsonl)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR),
                        help="model dir (default: <repo_root>/aicodec/aicodec/models/Qwen3_VL_30B_A3B_Instruct)")
    parser.add_argument("--baseline", default=None,
                        help="baseline result dir/predictions.jsonl for filtered_accuracy")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test on N samples")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate folder + prompts without loading the model")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"[FATAL] data dir not found: {data_dir}")
    manifest = resolve_manifest(data_dir, args.manifest)
    records = load_records(data_dir, manifest, args.limit)
    print(f"[DATA] {len(records)} questions | images+manifest from {data_dir}")

    if args.dry_run:
        print("[DRY-RUN] sample prompt:\n" + format_question(records[0]))
        return

    outputs = run_inference(records, args)
    for record, text in zip(records, outputs):
        labels = [key for key in CHOICE_KEYS if key in record["choices"]]
        prediction = extract_choice(text, labels)
        record["model_output"] = text
        record["prediction"] = prediction
        record["correct"] = prediction == record["answer"]

    summary = summarize(records, load_baseline_flags(args.baseline))
    summary["run_config"] = {
        "data_dir": str(data_dir), "manifest": str(manifest), "model": args.model,
        "limit": args.limit, "max_new_tokens": args.max_new_tokens, "seed": args.seed,
    }
    save_results(records, summary, Path(args.output_dir))
    print(f"[DONE] raw_accuracy={summary['raw_accuracy']:.4f} "
          f"({summary['correct']}/{summary['total']}) "
          f"invalid={summary['invalid_predictions']} -> {args.output_dir}")


if __name__ == "__main__":
    main()
