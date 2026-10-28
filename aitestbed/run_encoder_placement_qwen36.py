#!/usr/bin/env python3
"""
run_encoder_placement_qwen36.py — standalone runner for the Qwen3.6
encoder-placement scenario.

Why standalone: the Qwen3-VL experiment's files (orchestrator.py,
configs/scenarios.yaml, clients/__init__.py, scenarios/__init__.py)
are being actively debugged; this runner wires the Qwen3.6 scenario
(client + scenario + logger) together WITHOUT touching any of them.
When the Qwen3-VL work stabilizes, the three one-line registrations
documented in the scenario's module docstring can move the scenario
into the main orchestrator.

Usage (from the 6G-Testbed root, inside the epd conda env):

  # sidecar first (separate terminal):
  #   bash start_sidecar_qwen36.sh
  # then:
  python aitestbed/run_encoder_placement_qwen36.py \
      --runs 3 --profile no_emulation

  # edge mode (env wins over the yaml `mode` key):
  ENC_MODE_QWEN36=edge python aitestbed/run_encoder_placement_qwen36.py \
      --runs 3 --profile no_emulation --db logs/traffic_logs_qwen36_edge.db

  # no netem at all (default): raw loopback. Use a different output
  # DB/session per mode for A/B comparison.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Allow running both as `python aitestbed/run_...py` (repo root on
# sys.path via cwd) and from anywhere (add the aitestbed dir).
HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent, HERE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Bootstrap the sibling aicodec package BEFORE importing clients (or
# anything that imports clients): the clients package __init__ pulls
# realtime_webrtc_vlm_client, which does `import aicodec`. With the
# repo root on sys.path (the loop above), the OUTER aicodec/ dir (a
# pyproject-only wrapper, no __init__.py) would otherwise resolve as
# an empty PEP 420 namespace package on hosts without an editable
# install (observed on dgspl3310: aicodec.__file__ is None, no
# register()). Same bootstrap orchestrator.py runs before its own
# clients import; idempotent and CWD-independent.
_AICODEC_PATH = HERE.parent / "aicodec"
if _AICODEC_PATH.is_dir() and str(_AICODEC_PATH) not in sys.path:
    sys.path.insert(0, str(_AICODEC_PATH))

from analysis.logger import TrafficLogger  # noqa: E402
from clients.encoder_placement_client_qwen36 import (  # noqa: E402
    EncoderPlacementQwen36Client,
)
from scenarios.encoder_placement_qwen36 import (  # noqa: E402
    EncoderPlacementQwen36Scenario,
)

DEFAULT_CONFIG = HERE / "configs" / "scenarios_qwen36.yaml"


def _load_scenario_config(config_path: Path) -> dict:
    import yaml

    # encoding="utf-8": the yaml has em-dashes; the Windows default
    # codec (gbk/cp936) cannot decode them and crashes at startup.
    # Same fix as tools/encpl_bench_eval_concurrent.py:165.
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenarios = raw.get("scenarios") or {}
    cfg = scenarios.get("encoder_placement_qwen36")
    if not cfg:
        raise SystemExit(
            f"scenario 'encoder_placement_qwen36' not found in {config_path}")
    cfg["scenario_id"] = "encoder_placement_qwen36"
    # The yaml's video_base_dir "." means "the aitestbed dir (where
    # examples/ lives)" — anchor it to the CONFIG's parent so the
    # runner works from any cwd (repo root or aitestbed/).
    if cfg.get("video_base_dir") == ".":
        cfg["video_base_dir"] = str(config_path.parent.parent)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Qwen3.6 encoder-placement scenario "
                    "(standalone; no orchestrator changes). Precedence: "
                    "CLI flag > env (mode only) > yaml key > built-in "
                    "default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ---- experiment arm --------------------------------------------------
    parser.add_argument("--mode", default=None, choices=["server", "edge"],
                        help="server: terminal uploads the original video, "
                             "server runs decode+processor+ViT+prefill. "
                             "edge: terminal runs decode+processor+ViT "
                             "locally and uploads the "
                             "precomputed_embedding bundle. Env "
                             "ENC_MODE_QWEN36 (fallback ENC_MODE) wins "
                             "over the yaml `mode` key but NOT over this "
                             "flag")
    parser.add_argument("--model", default=None,
                        help="visual model path (terminal-side ViT in edge "
                             "mode; yaml default is the FP8 35B repo)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="video frame cap passed to the pipeline "
                             "(native sglang key, caps BOTH sampling modes; "
                             "no patch needed)")
    parser.add_argument("--frame-ratio", type=float, default=None,
                        metavar="R",
                        help="percentage frame sampling: keep "
                             "floor(total_frames * R) frames (multiple of "
                             "2, min 4, capped by --max-frames) instead of "
                             "sglang's default 2fps x duration. In (0, 1]. "
                             "EDGE mode: sampled locally via the patched "
                             "smart_nframes. SERVER mode: the ENGINE samples "
                             "with its STARTUP-time config — start the "
                             "sidecar with the SAME --frame-ratio (the "
                             "sidecar warns on request mismatch)")
    parser.add_argument("--decode-threads", type=int, default=None,
                        metavar="N",
                        help="cap decord's/torchcodec's internal decode "
                             "thread pool to N threads (decord sizes it by "
                             "HOST core count and is NOT covered by torch "
                             "thread knobs; the env name is historical — "
                             "caps BOTH backends). EDGE mode only (server "
                             " mode has no terminal-side decode; to cap "
                             "the ENGINE's decoder restart the sidecar "
                             "with the same --decode-threads; default 0 = "
                             "native)")
    parser.add_argument("--preprocess-threads", type=int, default=None,
                        metavar="N",
                        help="cap the terminal's torch/OMP intra-op thread "
                             "count for CPU-side preprocessing AFTER decode "
                             "(torchvision resize / HF processor): exports "
                             "OMP/MKL/OPENBLAS_NUM_THREADS + torch."
                             "set_num_threads, clamped to the edge-cpu pin's "
                             "core count (min(N, K); values BELOW the pin "
                             "simulate a weaker terminal, values above are "
                             "clamped). EDGE mode only — server mode warns "
                             "and ignores (the ENGINE-side cap is the "
                             "sidecar's own --preprocess-threads flag; "
                             "restart it with the same value for symmetric "
                             "comparisons). Omit = derive from the "
                             "edge_cpus pin (current behaviour)")
    parser.add_argument("--compressor", default=None,
                        choices=["", "vidcom2"],
                        help="VidCom2 token compression of the edge bundle "
                             "(edge mode only; '' uploads all video tokens)")
    parser.add_argument("--retention-ratio", type=float, default=None,
                        help="fraction of video tokens kept by vidcom2 "
                             "(0, 1]; only meaningful with "
                             "--compressor vidcom2 in edge mode")
    parser.add_argument("--bundle-quant", default=None, choices=["", "fp8"],
                        help="fp8 (e4m3, per-tensor scale) quantization of "
                             "the uploaded edge-bundle feature tensor; "
                             "requires --compressor vidcom2 + edge mode")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="generation token cap per request")
    parser.add_argument("--video-path", default=None,
                        help="input video (relative paths resolve against "
                             "--video-base-dir, else cwd)")
    parser.add_argument("--video-base-dir", default=None,
                        help="base directory for relative --video-path")
    parser.add_argument("--vision-only", action="store_true",
                        help="load only the vision tower (~0.5B) instead "
                             "of the full 35B fp8 model for the "
                             "terminal-side ViT (edge mode only). This is "
                             "the yaml DEFAULT on Qwen3.6 — the flag is "
                             "only needed to override a yaml that set it "
                             "false")
    parser.add_argument("--no-vision-only", action="store_true",
                        help="force the FULL 35B fp8 edge load (edge mode "
                             "only; ~35 GB VRAM — only sane on multi-GPU "
                             "hosts; overrides the yaml vision_only)")
    parser.add_argument("--attn-impl", default=None,
                        choices=["sdpa", "flash_attention_4", "eager"],
                        help="attention backend for the terminal-side ViT "
                             "(edge mode only; sdpa default. fa4/eager are "
                             "NOT bit-identical to sdpa — compare with a "
                             "tolerance, and use separate --db files per "
                             "backend)")
    # ---- dataset-bench mode ------------------------------------------------
    parser.add_argument("--dataset", default=None,
                        choices=["videomme", "mlvu"],
                        help="dataset-bench mode: iterate a QA test set "
                             "instead of the single configured video — one "
                             "turn per question; per-turn records carry "
                             "metadata.bench (question/answer/model_answer/"
                             "correct). Generation tasks (mlvu 8/9) are "
                             "recorded unscored. A failed question does NOT "
                             "abort the run. Absent = the historical "
                             "single-video path (yaml/env entry also works: "
                             "dataset/dataset_dir keys, env "
                             "ENCPL_DATASET_DIR)")
    parser.add_argument("--dataset-dir", default=None, metavar="PATH",
                        help="dataset root: data/ + videos/ for videomme, "
                             "json/ + video/ for mlvu. Env ENCPL_DATASET_DIR "
                             "wins over this flag when set")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="cap on questions evaluated (0/absent = full "
                             "set); with --dataset only")
    parser.add_argument("--tasks", default=None, nargs="+", metavar="STEM",
                        help="MLVU task selection: absent = MCQ tasks 1-7; "
                             "'all' = every task (8/9 generation tasks are "
                             "recorded unscored); or a list of json stems "
                             "(e.g. 1_plotQA 8_sub_scene); with --dataset "
                             "mlvu only")
    parser.add_argument("--sample", default=None,
                        choices=["stratified", "head"],
                        help="sampling for --limit: stratified (default; "
                             "per-task/duration quota so a limit covers the "
                             "workload mix) or head (first N rows in file "
                             "order); with --dataset only")
    # ---- runner ----------------------------------------------------------
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="scenarios yaml (default: configs/scenarios_qwen36.yaml)")
    parser.add_argument("--db", default="logs/traffic_logs_q36.db",
                        help="output SQLite DB (use a different one per mode "
                             "for A/B comparison)")
    parser.add_argument("--runs", type=int, default=3,
                        help="number of runs")
    parser.add_argument("--profile", default="no_emulation",
                        help="network profile name recorded in the DB "
                             "(netem itself is NOT applied by this runner; "
                             "use the orchestrator or tc manually)")
    parser.add_argument("--base-url", default=None,
                        help="sidecar base URL override "
                             "(default: client default http://localhost:8004)")
    # ---- stream safety timeouts (forwarded verbatim to the client) ----
    parser.add_argument("--stream-connect-timeout", type=float, default=None,
                        metavar="SEC",
                        help="TCP connect timeout for sidecar HTTP requests "
                             "(default: client class default 20; 0 disables)")
    parser.add_argument("--stream-first-event-timeout", type=float,
                        default=None, metavar="SEC",
                        help="budget from chat POST to the first SSE event "
                             "(default: client class default 200; 0 disables)")
    parser.add_argument("--stream-idle-timeout", type=float, default=None,
                        metavar="SEC",
                        help="max silent gap between SSE events mid-stream "
                             "(default: client class default 20; 0 disables)")
    parser.add_argument("--stream-upload-read-timeout", type=float,
                        default=None, metavar="SEC",
                        help="read timeout for the bundle-upload PUT response "
                             "(default: client class default 1200; 0 disables)")
    args = parser.parse_args()

    cfg = _load_scenario_config(Path(args.config))

    # CLI > yaml: overlay every flag that was actually given (same
    # overlay pattern as the concurrent Qwen3.6 runner). store_true
    # flags have no None default — overlay only when set.
    flag_to_key = {
        "mode": "mode",
        "model": "model",
        "max_frames": "max_frames",
        "frame_ratio": "frame_ratio",
        "decode_threads": "decode_threads",
        "preprocess_threads": "preprocess_threads",
        "compressor": "compressor",
        "retention_ratio": "retention_ratio",
        "bundle_quant": "bundle_quant",
        "max_tokens": "max_tokens",
        "video_path": "video_path",
        "video_base_dir": "video_base_dir",
        "attn_impl": "attn_impl",
        # dataset-bench mode (yaml keys the scenario reads; env
        # ENCPL_DATASET_DIR still wins INSIDE the scenario, so a set
        # env overrides this flag at run time)
        "dataset": "dataset",
        "dataset_dir": "dataset_dir",
        "limit": "limit",
        "tasks": "tasks",
        "sample": "sample",
    }
    for flag, key in flag_to_key.items():
        value = getattr(args, flag)
        if value is not None:
            cfg[key] = value
    # --tasks normalization: nargs="+" always yields a list, but the
    # scenario's task spec accepts 'all'/'mcq' only as a STRING (a
    # list is treated as literal json stems — 'all' would then look
    # for json/all.json). Collapse single special values to str.
    if isinstance(cfg.get("tasks"), list) and len(cfg["tasks"]) == 1 \
            and str(cfg["tasks"][0]).strip().lower() in ("all", "mcq"):
        cfg["tasks"] = str(cfg["tasks"][0]).strip().lower()
    if args.vision_only:
        cfg["vision_only"] = True
    if args.no_vision_only:
        cfg["vision_only"] = False

    client = EncoderPlacementQwen36Client(
        base_url=args.base_url,
        stream_connect_timeout=args.stream_connect_timeout,
        stream_first_event_timeout=args.stream_first_event_timeout,
        stream_idle_timeout=args.stream_idle_timeout,
        stream_upload_read_timeout=args.stream_upload_read_timeout)
    logger = TrafficLogger(args.db)
    scenario = EncoderPlacementQwen36Scenario(client, logger, cfg)

    mode = (os.environ.get("ENC_MODE_QWEN36")
            or os.environ.get("ENC_MODE")
            or cfg.get("mode", "server")).strip().lower()
    print(f"[EncPl36-runner] scenario=encoder_placement_qwen36 mode={mode} "
          f"model={cfg.get('model')} profile={args.profile} runs={args.runs} "
          + (f"frame_ratio={cfg.get('frame_ratio')} "
              if cfg.get("frame_ratio") else "")
          + (f"compressor={cfg.get('compressor') or 'none'} "
              f"bundle_quant={cfg.get('bundle_quant') or 'none'} "
              if mode == "edge" else "")
          + (f"preprocess_threads={cfg.get('preprocess_threads')} "
              if cfg.get("preprocess_threads") else "")
          + f"db={args.db}")

    n_ok = 0
    for run_index in range(args.runs):
        t0 = time.time()
        result = scenario.run(network_profile=args.profile,
                              run_index=run_index)
        n_ok += 1 if result.success else 0
        ttft = (f"{result.ttft_sec:.2f}s" if result.ttft_sec is not None else "n/a")
        print(f"[EncPl36-runner] run {run_index + 1}/{args.runs}: "
              f"success={result.success} turns={result.turn_count} "
              f"ttft={ttft} latency={result.total_latency_sec:.2f}s "
              f"tokens_in={result.total_tokens_in} "
              f"tokens_out={result.total_tokens_out} "
              f"({time.time() - t0:.1f}s wall)"
              + (f" error={result.error_message}" if result.error_message else ""))

    print(f"[EncPl36-runner] done: {n_ok}/{args.runs} runs succeeded; "
          f"records in {args.db}")


if __name__ == "__main__":
    main()
