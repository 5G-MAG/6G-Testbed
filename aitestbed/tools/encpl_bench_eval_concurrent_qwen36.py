#!/usr/bin/env python3
"""
Concurrent Video-MME / MLVU benchmark for the encoder_placement
experiment, Qwen3.6 variant.

Standalone sibling of tools/encpl_bench_eval_concurrent.py for the
Qwen3.6 (qwen3_5 / qwen3_5_moe) experiment — zero changes to that file,
so both experiments run/debug in parallel. Same playlist wave
semantics, dataset/scoring layer, DB LogRecord schema, pcap metadata
convention, CPU budget matrix and resume policy; every model-specific
surface follows the Qwen3.6 chain
(clients/encoder_placement_client_qwen36.py,
server/sglang_engine_server_concurrent_qwen36.py port 8014):

  - ONE wave of N barrier-aligned concurrent users; each user then
    processes V = --videos-per-user DISTINCT videos SERIALLY (its
    playlist), users running concurrently with NO inter-video
    barrier — the barrier aligns only the first video of each user.
  - Video assignment is INTERLEAVED and DETERMINISTIC: the distinct
    video pool (entries order) is dealt round-robin, user u gets
    pool positions u, u+N, u+2N, ... — no shuffle, no RNG, no seed,
    so switching --mode (server <-> edge) with the same dataset/
    limit/users replays the exact same per-user video lists.
  - Per-video answers land in a questions_results_{profile}.json file
    with the serial bench's entry schema (one entry PER VIDEO, tagged
    with user_index/seq), so the existing
    MetricsCalculator.load_task_results / calculate_task_accuracy
    ("real" task class) scoring works unmodified.
  - Per-video LogRecords go into the SAME SQLite schema the
    encoder_placement_concurrent_qwen36 scenario writes (turn_index =
    user_index * 1000 + seq — unique, >= 0 so metrics consumers'
    turn_index < 0 pcap-row filter is untouched; run_index = wave
    repetition; pcap metadata row keeps turn_index=-2), so
    ran2_metrics / pcap_analyzer work unmodified.
  --runs repeats the WHOLE wave (default 1 = single wave).

Topology / knobs are inherited from BOTH parents (precedence:
CLI flag > yaml key > built-in default, same
configs/scenarios_encoder_placement_concurrent_qwen36.yaml; the yaml's
video/prompts keys are IGNORED — the dataset supplies videos and
questions):

  runner: --mode --users --videos-per-user --compressor
          --retention-ratio --bundle-quant
          --vision-only --no-vision-only --attn-impl --model
          --max-frames --frame-ratio --decode-threads --max-tokens
          --config --db --runs --profile --profile-record-only
          --netem-iface/--interface --base-url --no-flush-cache
          --enable-multi-processes --edge-cpus-per-worker
          --edge-cpu-ids --edge-cpus --capture-pcap
          --capture-interface --capture-filter --capture-dir
  bench:  --dataset --dataset-dir --tasks --sample --limit --output
          --metrics-only --netem-direction

CPU budget matrix (inherited semantics, kept disjoint):
  thread mode + edge:   --edge-cpus pins THIS whole process
                        (bench semantics; default auto-half-other,
                        ENCPL_EDGE_CPUS env wins as default).
  process mode (either arm): --edge-cpus-per-worker K (default 1) pins
                        each worker round-robin from --edge-cpu-ids
                        (default: parent's affinity).
  Mismatched combos warn and are skipped (a stray env var must not
  change the resource model silently).

Server pairing (REQUIRED — launch via the bootstrap script, it loads
the sglang host patches the edge mode needs):
  bash start_sidecar_concurrent_qwen36.sh --port 8014 \
      --max-running-requests 4
(raise --max-running-requests >= N users; server mode also wants the
sidecar started with the SAME --frame-ratio / --decode-threads values,
and honors the sidecar's own --parallel-video-preprocess patch-4 knob)

vision_only: unlike the Qwen3-VL concurrent tool (which defaults the
full 4B load), Qwen3.6 defaults TRUE here — the fp8 35B full load
costs ~35 GB VRAM just to obtain the ~0.5 GB tower
(--no-vision-only forces the full load, mirroring the serial qwen36
bench).

Examples:
  # server arm, 4 users, one videomme question each (legacy V=1)
  python aitestbed/tools/encpl_bench_eval_concurrent_qwen36.py \
      --dataset videomme --dataset-dir /data01/Video-MME \
      --mode server --users 4 --limit 64 \
      --base-url http://localhost:8014

  # playlist mode: 4 users x 16 videos (V = limit // N when
  # --videos-per-user is omitted; barrier aligns only each user's
  # FIRST video, afterwards the users stream independently)
  python aitestbed/tools/encpl_bench_eval_concurrent_qwen36.py \
      --dataset videomme --dataset-dir /data01/Video-MME \
      --mode server --users 4 --videos-per-user 16 --limit 128 \
      --base-url http://localhost:8014

  # edge arm + vidcom2 + fp8, mlvu MCQ tasks
  python aitestbed/tools/encpl_bench_eval_concurrent_qwen36.py \
      --dataset mlvu --dataset-dir /data01/MLVU \
      --mode edge --users 4 --compressor vidcom2 \
      --retention-ratio 0.25 --bundle-quant fp8 \
      --base-url http://localhost:8014

  # N truly independent terminals (OS processes per user)
  python aitestbed/tools/encpl_bench_eval_concurrent_qwen36.py \
      --dataset videomme --dataset-dir /data01/Video-MME \
      --mode edge --users 4 --enable-multi-processes \
      --base-url http://localhost:8014

  # recompute accuracy/latency from an existing results file
  python aitestbed/tools/encpl_bench_eval_concurrent_qwen36.py \
      --metrics-only --dataset videomme --mode server \
      --dataset-dir /data01/Video-MME \
      --base-url http://localhost:8014

Accuracy caveat: with N <= 4 users the MCQ accuracy is noise (a few
questions per wave); this tool's real output is the per-user
CONCURRENT latency distribution (TTFT/latency per user per video,
wave p50/p90, first vs steady split) plus the per-user traffic
footprint.

Environment:
    SGLANG_ENGINE_QWEN36C_BASE_URL  sidecar base URL (default
                                    http://localhost:8014; the
                                    QWEN36C suffix marks the
                                    CONCURRENT qwen36 sidecar)
    ENC_MODE_QWEN36                 fallback for --mode (ENC_MODE as
                                    second fallback, as in the
                                    scenario; env wins over yaml)
    ENCPL_EDGE_CPUS                 fallback for --edge-cpus (env wins)
    ENCPL_DECODE_THREADS            fallback for --decode-threads
    DATASET_DIR                     fallback for --dataset-dir
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AITESTBED_ROOT = REPO_ROOT / "aitestbed"
TOOLS_DIR = Path(__file__).resolve().parent
# Same bootstrap as encpl_bench_eval_qwen36.py: tools/ for the sibling
# bench modules (videomme_eval / encpl_bench_eval_qwen36), aitestbed/
# for clients.* / analysis.* / capture.*, plus the aicodec / netemu
# sibling packages.
for _p in (str(TOOLS_DIR), str(AITESTBED_ROOT),
           str(REPO_ROOT / "aicodec"), str(REPO_ROOT / "netemu" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Both siblings' module levels are import-safe (stdlib + requests).
import videomme_eval as _vmme  # noqa: E402
import encpl_bench_eval_qwen36 as _base  # noqa: E402

from clients.encoder_placement_client_qwen36 import (  # noqa: E402
    EncoderPlacementQwen36Client,
    _resolve_attn_impl,
)
from clients.encpl_edge_worker_qwen36 import _worker_entry  # noqa: E402
from analysis.logger import TrafficLogger, LogRecord  # noqa: E402
from analysis.anonymization import get_anonymizer  # noqa: E402
from analysis.trace_logger import TraceLogger  # noqa: E402
from capture.controller import CaptureController  # noqa: E402

DEFAULT_CONFIG = (AITESTBED_ROOT / "configs"
                  / "scenarios_encoder_placement_concurrent_qwen36.yaml")

DEFAULT_MODEL = "aicodec/aicodec/models/Qwen3.6-35B-A3B"

# Worker protocol timeouts (same defaults as the scenario's config
# keys barrier_timeout / worker_ready_timeout / worker_join_timeout).
_BARRIER_TIMEOUT = 600
_WORKER_READY_TIMEOUT = 900
_WORKER_JOIN_TIMEOUT = 1200

LOG_PREFIX = "[EncPlBenchC36]"

# The concurrent sidecar's launch script (port 8014) — differs from the
# serial bench's 8004 script, so _base._preflight_sidecar's timeout hint
# must name THIS one (sidecar_hint parameter).
CONCURRENT_SIDECAR_SCRIPT = "start_sidecar_concurrent_qwen36.sh"


# ---------------------------------------------------------------------------
# config overlay (runner semantics: CLI flag > yaml key > built-in default)
# ---------------------------------------------------------------------------

def _load_scenario_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenarios = raw.get("scenarios") or {}
    cfg = scenarios.get("encoder_placement_concurrent_qwen36")
    if not cfg:
        sys.exit(f"scenario 'encoder_placement_concurrent_qwen36' not found "
                 f"in {config_path}")
    cfg["scenario_id"] = "encoder_placement_concurrent_qwen36"
    # The dataset supplies videos AND questions: the yaml's video /
    # prompts keys are deliberately ignored by this tool.
    cfg.pop("video_path", None)
    cfg.pop("video_paths", None)
    cfg.pop("video_base_dir", None)
    cfg.pop("prompts", None)
    return cfg


def _yaml_bool(value, default=False) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value) if value is not None else default


# ---------------------------------------------------------------------------
# netem teardown (copied from run_encoder_placement_concurrent_qwen36.py:
# the serial bench's _vmme.clear_netem hardcodes interface='auto' for
# dual-host, which ignores an explicit --netem-iface)
# ---------------------------------------------------------------------------

def _clear_netem(base_url: str, interface: str) -> None:
    try:
        from netemu.emulator import NetworkEmulator
        if _vmme._is_local_base_url(base_url):
            NetworkEmulator(interface="lo").clear_loopback()
        else:
            NetworkEmulator(interface=interface).clear()
    except Exception as e:  # noqa: BLE001 — best-effort cleanup
        print(f"{LOG_PREFIX} warning: netem cleanup failed ({e})")


# ---------------------------------------------------------------------------
# cpu-id parsing (copied from scenarios/encoder_placement_concurrent.py —
# importing the scenarios package would pull every scenario in)
# ---------------------------------------------------------------------------

def _parse_cpu_ids(spec, fallback=None) -> list[int]:
    if spec is None:
        return sorted(int(c) for c in (fallback or []))
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"negative cpu id {spec}")
        return [spec]
    if isinstance(spec, (list, tuple, set)):
        out = set()
        for item in spec:
            n = int(item)
            if n < 0:
                raise ValueError(f"negative cpu id {n}")
            out.add(n)
        return sorted(out)
    text = str(spec).strip()
    if not text:
        return sorted(int(c) for c in (fallback or []))
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"bad range {part!r} in cpu spec "
                                 f"{spec!r}") from exc
            if lo < 0 or hi < lo:
                raise ValueError(f"bad range {part!r} in cpu spec {spec!r}")
            out.update(range(lo, hi + 1))
        else:
            try:
                n = int(part)
            except ValueError as exc:
                raise ValueError(f"bad cpu id {part!r} in cpu spec "
                                 f"{spec!r}") from exc
            if n < 0:
                raise ValueError(f"negative cpu id {n}")
            out.add(n)
    return sorted(out)


# ---------------------------------------------------------------------------
# client cache warmup (thread mode only — copy of the scenario's method;
# process workers warm their own caches before signaling 'ready')
# ---------------------------------------------------------------------------

def _warm_client_caches(client, model: str, mode: str,
                        vision_only: bool = False,
                        attn_impl: str = "sdpa") -> None:
    """Populate the client's process-global caches single-threaded.

    Under a concurrent wave, N user threads racing the lazy
    transformers imports collide in sys.modules and surface spurious
    ImportErrors; warming once here (before the thread pool exists)
    means user threads only ever hit the populated caches. The
    (vision_only, attn_impl) key must match the user requests exactly.
    """
    t0 = time.time()
    try:
        client._get_processor(model)
    except Exception as exc:
        print(f"{LOG_PREFIX} warning: processor warmup failed ({exc}); "
              f"each user request will retry the load")
    if mode == "edge":
        try:
            client._get_edge_model(model, vision_only=vision_only,
                                   attn_impl=attn_impl)
        except Exception as exc:
            print(f"{LOG_PREFIX} warning: edge-model warmup failed "
                  f"({exc}); each user request will retry the load")
    print(f"{LOG_PREFIX} client caches warmed in {time.time() - t0:.1f} s "
          f"(processor" + (" + edge visual tower" if mode == "edge" else "")
          + f"; mode={mode})")


# ---------------------------------------------------------------------------
# dataset loading + DISTINCT-video selection (one user = one video = one
# question; no video_id repeats inside the wave)
# ---------------------------------------------------------------------------

def _build_playlists(entries: list[dict], users: int,
                     videos_per_user: int) -> list[list[dict]]:
    """Deal the first users*V distinct-video entries round-robin.

    Missing video files are skipped (counted) — a playlist item must
    have a real file to upload/decode. Fewer distinct videos than
    users*V is a hard error: re-sharing a video would silently break
    the per-user concurrency semantics this tool exists to measure.

    Assignment is DETERMINISTIC and mode-independent (entries order,
    no shuffle/RNG/seed): user u gets pool positions u, u+N, u+2N...
    so switching --mode with the same dataset/limit/users replays
    the exact same per-user video lists.
    """
    seen: set[str] = set()
    pool: list[dict] = []
    n_missing = 0
    need = users * videos_per_user
    for e in entries:
        if e["video_id"] in seen:
            continue
        if e["video_path"] is None:
            n_missing += 1
            continue
        seen.add(e["video_id"])
        pool.append(e)
        if len(pool) == need:
            break
    if n_missing:
        print(f"{LOG_PREFIX} note: {n_missing} candidate entries skipped "
              f"(video file missing)")
    if len(pool) < need:
        sys.exit(f"only {len(pool)} distinct-video entries available "
                 f"but --users {users} x {videos_per_user} videos/user = "
                 f"{need} requested — lower --users/--videos-per-user or "
                 f"raise --limit")
    return [pool[u::users] for u in range(users)]


# ---------------------------------------------------------------------------
# wave execution — thread mode (copy of the scenario's _run_wave, plus
# total_content for the accuracy scoring)
# ---------------------------------------------------------------------------

def _run_wave_threads(client, user_playlists: list[list[dict]], model: str,
                       mode: str, max_frames: int, max_tokens: int,
                       compressor: str, retention_ratio: float,
                       bundle_quant: str, vision_only: bool,
                       attn_impl: str, frame_ratio,
                       optional_ratio: float = 0.0,
                       opt_tokens_per_packet: int = 64,
                       opt_send_gap_ms: float = 0.05,
                       opt_chat_delay_ms: float = 0.0
                       ) -> tuple[list[dict], bool]:
    """One barrier-aligned wave of N user threads, each on a playlist.

    Returns (per-VIDEO records ordered by (user_index, seq),
    interrupted). Each record carries the full per-request
    measurement set INCLUDING total_content (the model's raw answer,
    needed for MCQ scoring) plus user_index/seq.
    """
    users = len(user_playlists)
    barrier = threading.Barrier(users)
    records: list[dict] = []
    records_lock = threading.Lock()

    def _user(user_index: int):
        playlist = user_playlists[user_index]
        user_tag = f"[u{user_index}] "
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return  # a sibling died pre-start; nothing to salvage here
        for seq, e in enumerate(playlist):
            rec = {
                "user_index": user_index,
                "seq": seq,
                "ok": False,
                "t_request_start": None,
                "ttft": None,
                "latency": 0.0,
                "video_path": e["video_path"],
                "video_id": e["video_id"],
            }
            try:
                t_request_start = time.time()
                rec["t_request_start"] = t_request_start
                if e["options"]:
                    prompt = _vmme.build_prompt(e["question"], e["options"])
                else:
                    # mlvu generation tasks (8/9): free-form,
                    # recorded unscored.
                    prompt = e["question"]
                response = client.generate_content_with_video(
                    prompt=prompt,
                    video_path=e["video_path"],
                    model=model,
                    mode=mode,
                    max_frames=max_frames,
                    max_tokens=max_tokens,
                    compressor=compressor,
                    retention_ratio=retention_ratio,
                    bundle_quant=bundle_quant,
                    vision_only=vision_only if mode == "edge" else False,
                    attn_impl=attn_impl if mode == "edge" else "sdpa",
                    user_tag=f"{user_tag}[v{seq}] ",
                    frame_ratio=frame_ratio,
                    optional_ratio=(optional_ratio
                                    if mode == "edge" else 0.0),
                    opt_tokens_per_packet=opt_tokens_per_packet,
                    opt_send_gap_ms=opt_send_gap_ms,
                    opt_chat_delay_ms=opt_chat_delay_ms,
                )
                tokens_in = response.tokens_in
                tokens_out = response.tokens_out
                if tokens_in is None:
                    tokens_in = client.estimate_tokens(prompt, model)
                if tokens_out is None:
                    tokens_out = client.estimate_tokens(
                        response.total_content, model)
                t_last_chunk = response.t_last_chunk
                rec.update({
                    "ok": True,
                    "t_first_chunk": response.t_first_chunk,
                    "t_last_chunk": t_last_chunk,
                    "ttft": (response.t_first_chunk - t_request_start
                             if response.t_first_chunk is not None
                             else None),
                    "latency": (t_last_chunk - t_request_start
                                if t_last_chunk is not None else 0.0),
                    "request_bytes": response.request_bytes,
                    "response_bytes": response.response_bytes,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "chunk_count": len(response.chunks),
                    "inter_chunk_times": response.inter_chunk_times,
                    "request_payload": response.request_payload,
                    "response_payload": response.response_payload,
                    "response_events": response.response_events,
                    "engine_stats": (getattr(response, "engine_stats", {})
                                     or {}),
                    "total_content": response.total_content or "",
                })
            except BaseException as exc:  # noqa: BLE001 — record, continue
                t_fail = time.time()
                t_request_start = rec.get("t_request_start") or t_fail
                rec.update({
                    "ok": False,
                    "error": str(exc),
                    "t_request_start": t_request_start,
                    "latency": t_fail - t_request_start,
                })
            with records_lock:
                records.append(rec)

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=users) as executor:
            futures = [executor.submit(_user, i) for i in range(users)]
            for future in as_completed(futures):
                # _user never raises (it records its own failures).
                future.result()
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n{LOG_PREFIX} interrupted — salvaging completed videos")
    records.sort(key=lambda r: (r["user_index"], r["seq"]))
    return records, interrupted


# ---------------------------------------------------------------------------
# wave execution — process mode (copy of the scenario's
# _run_wave_processes protocol; reuses clients/encpl_edge_worker_qwen36.py)
# ---------------------------------------------------------------------------

def _content_from_events(rec: dict) -> str:
    """Rebuild the model's answer text from a worker record.

    The worker protocol's record deliberately mirrors the thread-wave
    schema, which has no total_content field — but response_events
    carries every SSE chunk payload, so the answer text is fully
    reconstructable (same join the client itself does while
    streaming). Mirrors the client's own chunk parsing.
    """
    parts: list[str] = []
    for ev in rec.get("response_events") or []:
        if not isinstance(ev, dict):
            continue
        payload = ev.get("payload") or {}
        choices = payload.get("choices") or []
        if choices:
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                parts.append(content)
    return "".join(parts)


def _run_wave_processes(user_playlists: list[list[dict]], model: str,
                        mode: str, max_frames: int, max_tokens: int,
                        compressor: str, retention_ratio: float,
                        bundle_quant: str, vision_only: bool,
                        attn_impl: str, frame_ratio,
                        decode_threads, edge_cpu_plan, base_url: str,
                        optional_ratio: float = 0.0,
                        opt_tokens_per_packet: int = 64,
                        opt_send_gap_ms: float = 0.05,
                        opt_chat_delay_ms: float = 0.0
                        ) -> tuple[list[dict], bool]:
    """One barrier-aligned wave of N PROCESS-isolated playlist users.

    Same per-video record schema as the thread wave (plus
    edge_processes markers), via clients/encpl_edge_worker_qwen36.py's
    _worker_entry: own interpreter, own GIL, own connection, own edge
    ViT (loaded before 'ready', outside every timing window). The
    worker gets cfg['playlist'] (V per-request items) and emits ONE
    'done' record PER ITEM; a failed item is recorded (ok=False) and
    the loop continues. The answer text is reconstructed from
    response_events (see _content_from_events). Returns
    (per-video records sorted by (user_index, seq), interrupted).
    """
    import multiprocessing as mp

    users = len(user_playlists)
    aitestbed_root = str(AITESTBED_ROOT)

    # Terminal CPU assignment (round-robin `per_worker` cpus from the
    # pool per user_index; overlapping when the pool is oversubscribed
    # — the sharing itself is a real terminal-fleet effect).
    cpu_assignments: dict[int, list[int]] = {}
    if edge_cpu_plan:
        pool = edge_cpu_plan["pool"]
        per = edge_cpu_plan["per_worker"]
        if pool and users * per > len(pool):
            print(f"{LOG_PREFIX} warning: edge cpu pool ({len(pool)} cpus) "
                  f"< demand ({users} users x {per} cpu); workers will "
                  f"share cpus round-robin")
        for user_index in range(users):
            cpu_assignments[user_index] = [
                pool[(user_index * per + k) % len(pool)]
                for k in range(per)
            ] if pool else []

    ctx = mp.get_context("spawn")  # CUDA-safe
    # Parties = N workers + this parent: model loads / CUDA contexts
    # never land inside a timing window.
    barrier = ctx.Barrier(users + 1, timeout=_BARRIER_TIMEOUT)

    records: dict[int, list[dict]] = {}
    errors: dict[int, str] = {}
    ready_count = [0]
    error_seen = threading.Event()
    ready_cv = threading.Condition()

    def _reader(pipe, user_index: int):
        last_error = None
        try:
            while True:
                try:
                    msg = pipe.recv()
                except EOFError:
                    break  # worker closed its end (normal exit)
                if msg is None:
                    break  # explicit shutdown marker
                mtype = msg.get("type")
                if mtype == "ready":
                    with ready_cv:
                        ready_count[0] += 1
                        ready_cv.notify_all()
                elif mtype == "done":
                    rec = msg.get("record") or {}
                    with ready_cv:
                        records.setdefault(user_index, []).append(rec)
                elif mtype == "error":
                    last_error = msg.get("error") or "worker error"
                    with ready_cv:
                        errors[user_index] = last_error
                        error_seen.set()
                        ready_cv.notify_all()
        except Exception as exc:
            last_error = f"reader: {exc}"
            with ready_cv:
                errors[user_index] = last_error
                error_seen.set()
                ready_cv.notify_all()
        finally:
            if last_error is not None:
                with ready_cv:
                    errors.setdefault(user_index, last_error)

    procs = []
    readers = []
    interrupted = False
    try:
        for user_index in range(users):
            playlist = []
            for seq, e in enumerate(user_playlists[user_index]):
                if e["options"]:
                    prompt = _vmme.build_prompt(e["question"], e["options"])
                else:
                    prompt = e["question"]
                playlist.append({
                    "prompt": prompt,
                    "user_tag": f"[u{user_index}] [v{seq}] ",
                    "video_path": str(e["video_path"]),
                    "video_id": e["video_id"],
                    "seq": seq,
                })
            parent_end, child_end = ctx.Pipe(duplex=True)
            cfg = {
                "user_index": user_index,
                "playlist": playlist,
                "model": model,
                "mode": mode,
                "max_frames": max_frames,
                "max_tokens": max_tokens,
                "compressor": compressor,
                "retention_ratio": retention_ratio,
                # four two-tier UDP knobs (worker defaults match
                # generate_content_with_video's when absent)
                "optional_ratio": optional_ratio,
                "opt_tokens_per_packet": opt_tokens_per_packet,
                "opt_send_gap_ms": opt_send_gap_ms,
                "opt_chat_delay_ms": opt_chat_delay_ms,
                "bundle_quant": bundle_quant,
                "vision_only": vision_only,
                "attn_impl": attn_impl,
                "frame_ratio": frame_ratio,
                "decode_threads": decode_threads,
                "base_url": base_url,
                "aitestbed_root": aitestbed_root,
                "edge_cpus": cpu_assignments.get(user_index),
            }
            proc = ctx.Process(
                target=_worker_entry,
                args=(cfg, barrier, child_end),
                name=f"encpl-bench-worker-q36-{user_index}",
                daemon=True,
            )
            proc.start()
            child_end.close()  # parent keeps only its side
            procs.append(proc)
            reader = threading.Thread(
                target=_reader, args=(parent_end, user_index),
                name=f"encpl-bench-reader-q36-{user_index}", daemon=True)
            reader.start()
            readers.append((reader, parent_end))

        # Wait for every worker's 'ready' (model loaded, worker at the
        # barrier). A worker that reports an 'error' message or dies
        # before 'ready' fails this wait FAST with the real cause.
        deadline = time.time() + _WORKER_READY_TIMEOUT
        while True:
            with ready_cv:
                if ready_count[0] >= users:
                    break
                if error_seen.is_set():
                    first_err = (list(errors.values())[0]
                                 if errors else "unknown worker error")
                    raise RuntimeError(
                        f"edge worker failed during startup: {first_err}")
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"only {ready_count[0]}/{users} edge workers became "
                        f"ready within {_WORKER_READY_TIMEOUT} s")
                ready_cv.wait(timeout=min(remaining, 5.0))
            if any(not p.is_alive() and p.exitcode != 0 for p in procs):
                dead = [p.name for p in procs
                        if not p.is_alive() and p.exitcode != 0]
                raise RuntimeError(
                    f"edge worker(s) died during startup: {dead}; check "
                    f"the console output above")

        # Release the wave (parent is the last barrier party).
        barrier.wait()
        t_barrier_release = time.time()

        # Wait for all workers to finish their playlist and exit.
        for proc in procs:
            proc.join(timeout=_WORKER_JOIN_TIMEOUT)
        for reader, _ in readers:
            reader.join(timeout=30)
        t_workers_done = time.time()
        # Barrier-to-drain wall: excludes spawn/import/model-load and
        # process teardown — comparable with the thread wave's
        # wave_wall_sec (see METRICS.md §2.3).
        wave_barrier_wall = t_workers_done - t_barrier_release
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n{LOG_PREFIX} interrupted — salvaging completed videos")
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
        for _, pipe in readers:
            try:
                pipe.close()
            except Exception:
                pass

    # Assemble per-video records in (user_index, seq) order —
    # identical schema to the thread wave, plus the process-mode
    # marker. Workers with no records at all get one synthetic
    # failure row so every user stays represented.
    out: list[dict] = []
    for user_index in range(users):
        user_recs = records.get(user_index) or []
        playlist = user_playlists[user_index]
        by_seq: dict[int, dict] = {}
        for rec in user_recs:
            seq = rec.get("seq")
            if not isinstance(seq, int) or seq < 0:
                seq = len(by_seq)
            by_seq[seq] = rec
        for seq in range(len(playlist)):
            e = playlist[seq]
            if seq in by_seq:
                rec = dict(by_seq[seq])
                rec.setdefault("user_index", user_index)
                rec.setdefault("seq", seq)
                rec.setdefault("video_path", str(e["video_path"]))
                rec.setdefault("video_id", e["video_id"])
                if rec.get("ok") and "total_content" not in rec:
                    rec["total_content"] = _content_from_events(rec)
            else:
                err = errors.get(
                    user_index,
                    "worker produced no record for this video "
                    "(crashed or timed out)")
                rec = {
                    "user_index": user_index,
                    "seq": seq,
                    "ok": False,
                    "error": err,
                    "t_request_start": None,
                    "ttft": None,
                    "latency": 0.0,
                    "video_path": str(e["video_path"]),
                    "video_id": e["video_id"],
                }
            rec["edge_processes"] = True
            # Barrier-release -> all-workers-joined wall: the
            # spawn/load/teardown-free twin of wave_wall_sec.
            rec["wave_barrier_wall_sec"] = wave_barrier_wall
            if rec.get("edge_cpus") is None:
                planned = cpu_assignments.get(user_index)
                if planned:
                    rec["edge_cpus"] = ",".join(str(c) for c in planned)
            out.append(rec)
    return out, interrupted


# ---------------------------------------------------------------------------
# wave aggregation (scenario semantics)
# ---------------------------------------------------------------------------

def _pct(values, p):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    # Linear-interpolated percentile (same semantics as
    # analysis/metrics.py _percentile); the old round-to-index version
    # was order-statistics-only (n=2 p50 = minimum).
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def _stat_block(values: list[float]) -> dict:
    """mean/p50/p90/p95/max over raw per-request values (None-safe)."""
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _pct(values, 50),
        "p90": _pct(values, 90),
        "p95": _pct(values, 95),
        "max": max(values) if values else None,
    }


def _fmt_block(prefix: str, b: dict) -> str:
    if not b or not b.get("n"):
        return f"{prefix} n=0"
    return (f"{prefix} mean={b['mean']:.1f} p50={b['p50']:.1f} "
            f"p90={b['p90']:.1f} p95={b['p95']:.1f} "
            f"max={b['max']:.1f} s (n={b['n']})")


def _wave_stats(video_records: list[dict], users: int, wall: float,
                videos_per_user: int) -> dict:
    ok = [r for r in video_records if r.get("ok")]
    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    # Truthiness filter (not "is not None"): an empty completion records
    # latency=0.0, which must stay OUT of the wave stats just like
    # analysis/metrics.py keeps it out of latency_mean.
    lats = [r["latency"] for r in ok if r.get("latency")]
    first = [r["ttft"] for r in ok
             if r.get("seq") == 0 and r.get("ttft") is not None]
    steady = [r["ttft"] for r in ok
              if (r.get("seq") or 0) > 0 and r.get("ttft") is not None]
    ttft_by_user: dict[int, list[float]] = {}
    lat_by_user: dict[int, list[float]] = {}
    for r in ok:
        u = r.get("user_index")
        if u is None:
            continue
        if r.get("ttft") is not None:
            ttft_by_user.setdefault(u, []).append(r["ttft"])
        # Same truthiness filter as `lats` above (and metrics.py's
        # latency_mean): empty completions record latency=0.0, which
        # must stay OUT of the per-user/pooled stats too — an
        # "is not None" check here would pull per-user and overall
        # latency means toward zero on empty-completion runs.
        if r.get("latency"):
            lat_by_user.setdefault(u, []).append(r["latency"])
    return {
        "wave_size": users,
        "videos_per_user": videos_per_user,
        "wave_wall_sec": wall,
        # barrier-release -> workers-joined wall; present only in
        # edge_processes mode where wave_wall_sec includes spawn/
        # model-load/teardown (see METRICS.md §2.3)
        **({"wave_barrier_wall_sec":
            max((r.get("wave_barrier_wall_sec") or 0.0)
                for r in video_records)}
           if any(r.get("wave_barrier_wall_sec") is not None
                  for r in video_records) else {}),
        "wave_ok": len(ok),
        "wave_total": len(video_records),
        "wave_failed": len(video_records) - len(ok),
        # ok=True but ttft=None: zero content tokens (greedy first-token
        # EOS / retract) — surfaced so p50/p90 don't silently include
        # decode-only non-answers.
        "wave_empty": sum(1 for r in ok if r.get("ttft") is None),
        "ttft": _stat_block(ttfts),
        "latency": _stat_block(lats),
        "ttft_first": _stat_block(first),
        "ttft_steady": _stat_block(steady),
        "ttft_by_user": {u: [round(v, 4) for v in sorted(vs)]
                         for u, vs in sorted(ttft_by_user.items())},
        "latency_by_user": {u: [round(v, 4) for v in sorted(vs)]
                            for u, vs in sorted(lat_by_user.items())},
        "ttft_first_by_user": {
            u: round(r["ttft"], 4) for r in ok
            if r.get("seq") == 0 and r.get("ttft") is not None
            and r.get("user_index") is not None
            for u in [r["user_index"]]
        },
        "ttft_steady_by_user": {
            u: [round(r["ttft"], 4) for r in ok
                if r.get("user_index") == u
                and (r.get("seq") or 0) > 0
                and r.get("ttft") is not None]
            for u in sorted({r.get("user_index") for r in ok
                             if r.get("user_index") is not None})
        },
        "ok_by_user": {
            u: sum(1 for r in video_records
                   if r.get("user_index") == u and r.get("ok"))
            for u in range(users)
        },
        "conns_by_user": {
            r.get("user_index"): (r.get("engine_stats") or {}).get("conns")
            for r in ok if r.get("user_index") is not None
        },
        "video_ids_by_user": {
            r.get("user_index"): r.get("video_id")
            for r in ok if r.get("seq") == 0
            and r.get("user_index") is not None
        },
    }


def _print_wave_summary(wave: dict) -> None:
    print(f"{LOG_PREFIX} wave done in {wave['wave_wall_sec']:.1f} s | "
          f"ok={wave['wave_ok']}/{wave['wave_total']}"
          + (f" (empty={wave['wave_empty']})"
             if wave["wave_empty"] else "")
          + " | " + _fmt_block("TTFT", wave["ttft"]))
    print(f"{' ' * len(LOG_PREFIX)}  "
          + _fmt_block("lat", wave["latency"]))


def _print_per_user_summary(waves: list[dict]) -> None:
    """Per-user TTFT/latency stats over that user's V videos.

    Each user row aggregates the user's own requests across waves:
    ttft/lat mean/p50/p90/p95/max, first vs steady split, video mix.
    Per-user p95 with few samples (<10) is indicative only.
    """
    if not waves:
        return
    users = waves[0]["wave_size"]
    ttft_all: dict[int, list[float]] = {}
    lat_all: dict[int, list[float]] = {}
    first_all: dict[int, list[float]] = {}
    steady_all: dict[int, list[float]] = {}
    ok_all: dict[int, int] = {}
    conns: dict[int, dict] = {}
    vpu = waves[0].get("videos_per_user") or 1
    for w in waves:
        for u, vs in (w.get("ttft_by_user") or {}).items():
            ttft_all.setdefault(u, []).extend(vs)
        for u, vs in (w.get("latency_by_user") or {}).items():
            lat_all.setdefault(u, []).extend(vs)
        for u, v in (w.get("ttft_first_by_user") or {}).items():
            first_all.setdefault(u, []).append(v)
        for u, vs in (w.get("ttft_steady_by_user") or {}).items():
            steady_all.setdefault(u, []).extend(vs)
        for u, n in (w.get("ok_by_user") or {}).items():
            ok_all[u] = ok_all.get(u, 0) + n
        for u, c in (w.get("conns_by_user") or {}).items():
            if c:
                conns.setdefault(u, c)
    print(f"{LOG_PREFIX} per-user summary "
          f"({users} users x {vpu} videos/user):")
    for u in range(users):
        tv = ttft_all.get(u) or []
        lv = lat_all.get(u) or []
        fv = first_all.get(u) or []
        sv = steady_all.get(u) or []
        n_ok = ok_all.get(u, 0)
        first_desc = (f"first={fv[0]:.2f}" if fv else "first=-")
        steady_desc = (f"steady_mean={statistics.fmean(sv):.2f}"
                       if sv else "steady_mean=-")
        print(f"  u{u}: n_ok={n_ok}/{vpu * len(waves)} | "
              + _fmt_block("ttft", _stat_block(tv)))
        print(f"{' ' * (6 + len(f'u{u}'))} "
              + _fmt_block("lat", _stat_block(lv))
              + f" | {first_desc} {steady_desc}")


def _fmt_5tuple(t) -> str:
    """`src_ip:src_port -> dst_ip:dst_port` from a conn_5tuple dict."""
    if not t or not isinstance(t, dict):
        return "-"
    src = f"{t.get('src_ip')}:{t.get('src_port')}"
    dst = f"{t.get('dst_ip')}:{t.get('dst_port')}"
    return f"{src} -> {dst}"


def _fmt_bytes(n) -> str:
    """Human MiB with raw bytes (None -> '-')."""
    if n is None:
        return "-"
    mib = n / (1024 * 1024)
    return f"{mib:.2f} MiB ({n} B)"


def _print_per_user_detail(video_records: list[dict],
                           run_label: str) -> None:
    """FULL per-user listing (one block per user, one line per video).

    Per video line: seq, video_id + FULL video_path, upload & chat
    5-tuples (src_ip:src_port -> dst_ip:dst_port), uploaded payload /
    wire bytes, upload_time, ttft / lat. Failure rows carry the error
    string instead of the numeric tail. Records missing engine_stats
    (mock clients) degrade to path + ttft/lat only.
    """
    if not video_records:
        return
    print(f"{LOG_PREFIX} per-user detail ({run_label}):")
    by_user: dict[int, list[dict]] = {}
    for r in video_records:
        u = r.get("user_index")
        if u is None:
            continue
        by_user.setdefault(u, []).append(r)
    for u in sorted(by_user):
        recs = sorted(by_user[u], key=lambda r: r.get("seq") or 0)
        n_ok = sum(1 for r in recs if r.get("ok"))
        print(f"  u{u}: {n_ok}/{len(recs)} ok")
        for r in recs:
            stats = r.get("engine_stats") or {}
            conns = stats.get("conns") or {}
            vid = r.get("video_id") or "-"
            path = r.get("video_path") or "-"
            seq = r.get("seq")
            ttft = r.get("ttft")
            lat = r.get("latency")
            ttft_s = f"{ttft:.2f}" if ttft is not None else (
                "EMPTY" if r.get("ok") else "FAIL")
            lat_s = (f"{lat:.2f}" if (lat is not None and r.get("ok"))
                     else "-")
            head = (f"    [v{seq}] {vid} ttft={ttft_s} lat={lat_s} s")
            if r.get("ok"):
                up_b = (stats.get("uploaded_payload_bytes")
                        if stats.get("uploaded_payload_bytes") is not None
                        else stats.get("video_payload_bytes"))
                wire_b = stats.get("uploaded_wire_bytes")
                up_t = stats.get("upload_time")
                up_t_s = f"{up_t:.2f}" if up_t is not None else "-"
                head += (f" | upload={_fmt_bytes(up_b)}"
                         + (f" wire={_fmt_bytes(wire_b)}"
                            if wire_b is not None else "")
                         + f" up_time={up_t_s} s")
            else:
                head += f" | error={r.get('error') or '?'}"
            print(head)
            print(f"        video: {path}")
            if conns.get("upload"):
                print(f"        up:  "
                      f"{_fmt_5tuple(conns.get('upload'))}")
            if conns.get("chat"):
                print(f"        chat:"
                      f"{_fmt_5tuple(conns.get('chat'))}")


def _print_final_summary(waves: list[dict]) -> None:
    """Pooled final stats over ALL raw requests across waves.

    TTFT and lat each get mean/p50/p90/p95/max over the pooled
    per-request values (not per-wave aggregates), plus the
    first-video (seq==0) vs steady-state (seq>0) TTFT split and
    ok/fail/empty counts.
    """
    ttft_all: list[float] = []
    lat_all: list[float] = []
    first_all: list[float] = []
    steady_all: list[float] = []
    total_ok = total_fail = total_empty = 0
    for w in waves:
        for u, vs in (w.get("ttft_by_user") or {}).items():
            ttft_all.extend(vs)
        for u, vs in (w.get("latency_by_user") or {}).items():
            lat_all.extend(vs)
        for u, v in (w.get("ttft_first_by_user") or {}).items():
            first_all.append(v)
        for u, vs in (w.get("ttft_steady_by_user") or {}).items():
            steady_all.extend(vs)
        total_ok += w.get("wave_ok") or 0
        total_fail += w.get("wave_failed") or 0
        total_empty += w.get("wave_empty") or 0
    if not ttft_all and not lat_all:
        return
    n_desc = (f"{total_ok} ok / {total_fail} fail / {total_empty} empty")
    print(f"{LOG_PREFIX} overall TTFT ({n_desc}): "
          f"mean={statistics.fmean(ttft_all):.2f} "
          f"p50={_pct(ttft_all, 50):.2f} "
          f"p90={_pct(ttft_all, 90):.2f} "
          f"p95={_pct(ttft_all, 95):.2f} "
          f"max={max(ttft_all):.2f} s"
          if ttft_all else
          f"{LOG_PREFIX} overall TTFT: n=0 ({n_desc})")
    if lat_all:
        print(f"{LOG_PREFIX} overall lat  ({n_desc}): "
              f"mean={statistics.fmean(lat_all):.2f} "
              f"p50={_pct(lat_all, 50):.2f} "
              f"p90={_pct(lat_all, 90):.2f} "
              f"p95={_pct(lat_all, 95):.2f} "
              f"max={max(lat_all):.2f} s")
    if first_all and steady_all:
        print(f"{LOG_PREFIX} TTFT first-video: "
              f"mean={statistics.fmean(first_all):.2f} "
              f"n={len(first_all)} | steady (seq>0): "
              f"mean={statistics.fmean(steady_all):.2f} "
              f"p50={_pct(steady_all, 50):.2f} "
              f"p95={_pct(steady_all, 95):.2f} "
              f"n={len(steady_all)}")


# ---------------------------------------------------------------------------
# LogRecord assembly (mirror of BaseScenario._create_log_record: trace
# files via TraceLogger, provider/model anonymization, metadata JSON)
# ---------------------------------------------------------------------------

def _make_log_record(scenario_id: str, session_id: str, turn_index: int,
                     run_index: int, network_profile: str, provider: str,
                     model: str, trace_logger, **kwargs) -> LogRecord:
    trace_request = kwargs.pop("trace_request", None)
    trace_response = kwargs.pop("trace_response", None)
    trace_events = kwargs.pop("trace_events", None)
    trace_note = kwargs.pop("trace_note", None)
    metadata = kwargs.pop("metadata", None)

    trace_path = trace_logger.write_trace(
        scenario_id=scenario_id,
        session_id=session_id,
        turn_index=turn_index,
        run_index=run_index,
        network_profile=network_profile,
        provider=provider,
        model=model,
        request_payload=trace_request,
        response_payload=trace_response,
        response_events=trace_events,
        note=trace_note,
        timing={
            "t_request_start": kwargs.get("t_request_start"),
            "t_first_token": kwargs.get("t_first_token"),
            "t_last_token": kwargs.get("t_last_token"),
            "latency_sec": kwargs.get("latency_sec"),
        },
        metrics={
            "request_bytes": kwargs.get("request_bytes"),
            "response_bytes": kwargs.get("response_bytes"),
            "tokens_in": kwargs.get("tokens_in"),
            "tokens_out": kwargs.get("tokens_out"),
            "http_status": kwargs.get("http_status"),
            "success": kwargs.get("success"),
            "is_streaming": kwargs.get("is_streaming"),
        },
    )
    if trace_path:
        if isinstance(metadata, dict):
            meta_dict = dict(metadata)
        elif isinstance(metadata, str) and metadata:
            try:
                meta_dict = json.loads(metadata)
            except Exception:
                meta_dict = {"raw_metadata": metadata}
        else:
            meta_dict = {}
        meta_dict["trace_path"] = trace_path
        metadata = json.dumps(meta_dict)
    elif isinstance(metadata, dict):
        metadata = json.dumps(metadata)

    anonymizer = get_anonymizer()
    return LogRecord(
        timestamp=time.time(),
        scenario_id=scenario_id,
        session_id=session_id,
        turn_index=turn_index,
        run_index=run_index,
        provider=anonymizer.provider_alias(provider),
        model=anonymizer.model_alias(model),
        network_profile=network_profile,
        metadata=metadata or "",
        **kwargs
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concurrent Video-MME / MLVU benchmark for the "
                    "Qwen3.6 encoder_placement experiment: ONE wave of N "
                    "barrier-aligned users, one distinct video + one "
                    "question each, against the lock-free SGLang sidecar "
                    "(sglang_engine_server_concurrent_qwen36.py, port "
                    "8014). Precedence: CLI flag > yaml key > built-in "
                    "default.",
    )
    # ---- bench dataset surface (serial bench inheritance) --------------
    parser.add_argument("--dataset", required=True,
                        choices=["videomme", "mlvu"])
    parser.add_argument("--dataset-dir",
                        default=os.environ.get("DATASET_DIR"),
                        help="dataset root: Video-MME HF dir (data/ + "
                             "videos/) or MLVU dir (json/ + video/, MLVU/ "
                             "wrapper ok)")
    parser.add_argument("--tasks", default=None,
                        help="mlvu only: 'mcq' (default, tasks 1-7), 'all' "
                             "(adds unscored generation tasks 8/9), or "
                             "comma-separated stems; ignored for videomme")
    parser.add_argument("--limit", type=int, default=0,
                        help="candidate pool cap BEFORE the distinct-video "
                             "selection (0 = all questions); with "
                             "--videos-per-user omitted, V = limit // N")
    parser.add_argument("--sample", default="stratified",
                        choices=["stratified", "head"],
                        help="limit>0 sampling of the candidate pool: "
                             "stratified (videomme: by duration, mlvu: by "
                             "task) or head (first N)")
    # ---- runner surface (concurrent inheritance) ------------------------

    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="scenarios yaml supplying every key not given "
                             "as a flag (video/prompts keys are ignored — "
                             "the dataset supplies them)")
    parser.add_argument("--mode", default=(os.environ.get("ENC_MODE_QWEN36")
                                           or os.environ.get("ENC_MODE")),
                        choices=["server", "edge"],
                        help="encoder placement: server (upload raw mp4) or "
                             "edge (local decode+processor+ViT, upload "
                             "bundle); ENC_MODE_QWEN36 env wins over yaml "
                             "(ENC_MODE as second fallback, as in the "
                             "scenario)")
    parser.add_argument("--users", type=int, default=None, metavar="N",
                        help="concurrent users per wave, each on its own "
                             "SERIAL playlist of V videos (yaml "
                             "concurrent_users default 2; keep <= the "
                             "sidecar's --max-running-requests)")
    parser.add_argument("--videos-per-user", type=int, default=None,
                        metavar="V",
                        help="distinct videos per user's playlist (V x N = "
                             "total requests; default: limit // N when "
                             "--limit > 0, else 1 = the legacy "
                             "one-video-per-user wave)")
    parser.add_argument("--model", default=None,
                        help="visual model path (yaml default wins when "
                             "omitted; qwen36 default: "
                             "aicodec/aicodec/models/Qwen3.6-35B-A3B)")
    parser.add_argument("--compressor", default=None,
                        choices=["", "vidcom2"],
                        help="edge mode only: 'vidcom2' prunes the edge "
                             "bundle to --retention-ratio of the video "
                             "tokens; '' uploads all tokens")
    parser.add_argument("--retention-ratio", type=float, default=None,
                        help="VidCom2 token keep ratio in (0, 1] (only "
                             "used with --compressor vidcom2 in edge mode)")
    parser.add_argument("--optional-ratio", type=float, default=None,
                        metavar="R",
                        help="two-tier retention split (edge + vidcom2 "
                             "only): carves the least-critical tail out "
                             "of the RETAINED set; the must tier "
                             "(retention - optional) rides the reliable "
                             "first wave, the optional tier streams "
                             "best-effort over UDP (E2U datagrams) "
                             "BETWEEN the must PUT and the chat POST "
                             "(late parts dropped, never retransmitted). "
                             "0 = single-tier legacy path. Must be in "
                             "[0, retention_ratio). REQUIRES the "
                             "concurrent sidecar started with "
                             "--opt-udp-port > 0 (default 8015). yaml "
                             "key: optional_ratio")
    parser.add_argument("--opt-tokens-per-packet", type=int, default=None,
                        metavar="N",
                        help="best-effort part granularity for the "
                             "optional tier: tokens packed into one UDP "
                             "part (packaging-only knob; the server "
                             "merges per-token). Default 64. yaml key: "
                             "opt_tokens_per_packet")
    parser.add_argument("--opt-send-gap-ms", type=float, default=None,
                        metavar="MS",
                        help="sleep between ADJACENT optional-tier UDP "
                             "datagrams (0.05 default paces the wave "
                             "below the receiver's reassembly rate; "
                             "0 = blast, only safe with a raised "
                             "receiver SO_RCVBUF). yaml key: "
                             "opt_send_gap_ms")
    parser.add_argument("--opt-chat-delay-ms", type=float, default=None,
                        metavar="MS",
                        help="sleep AFTER the UDP wave and BEFORE the "
                             "chat POST (trades opt-token arrival rate "
                             "against TTFT — the server snapshots at "
                             "chat-arrival). Default 0. yaml key: "
                             "opt_chat_delay_ms")
    parser.add_argument("--bundle-quant", default=None,
                        choices=["", "fp8"],
                        help="edge mode + vidcom2 only: quantize the pruned "
                             "feature tensor to fp8 e4m3 before upload")
    parser.add_argument("--vision-only", action="store_true",
                        help="edge mode only: load just the vision tower "
                             "(default TRUE on Qwen3.6 — the fp8 35B full "
                             "load costs ~35 GB VRAM just to obtain the "
                             "~0.5 GB tower)")
    parser.add_argument("--no-vision-only", action="store_true",
                        help="force the FULL model load in edge mode "
                             "(historical behaviour; the LAST flag of the "
                             "--vision-only/--no-vision-only pair wins)")
    parser.add_argument("--attn-impl", default=None,
                        choices=["sdpa", "flash_attention_4", "eager"],
                        help="edge mode only: attention backend for the "
                             "edge ViT (sdpa default; fa4/eager are NOT "
                             "bit-identical to sdpa)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="cap on sampled frames for BOTH sampling modes "
                             "(native sglang key; default 768)")
    parser.add_argument("--frame-ratio", type=float, default=None,
                        metavar="R",
                        help="ratio-of-native frame sampling (v3), in "
                             "(0, 1]; 0/omit = off. EDGE mode: sampled "
                             "locally (patched smart_nframes). SERVER mode: "
                             "start the sidecar with the SAME --frame-ratio")
    parser.add_argument("--decode-threads", type=int, default=None,
                        metavar="N",
                        help="cap decord's INTERNAL decode thread pool "
                             "(SGLANG_DECORD_NUM_THREADS). Edge mode: "
                             "thread mode exports it in the parent, "
                             "process mode passes it per worker. Server "
                             "mode: ignored — restart the sidecar with the "
                             "SAME --decode-threads. ENCPL_DECODE_THREADS "
                             "env wins as default")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="generation token cap per request")
    parser.add_argument("--db",
                        default="logs/traffic_logs_encpl_bench_conc_q36.db",
                        help="output SQLite DB (per-user LogRecords in the "
                             "encoder_placement_concurrent_qwen36 schema; "
                             "use one db per mode/N combination for A/B)")
    parser.add_argument("--runs", type=int, default=1,
                        help="repeat the WHOLE wave R times (default 1 = "
                             "single wave per invocation; run_index in the "
                             "DB distinguishes repetitions)")
    # ---- netem (bench direction-aware + runner iface) -------------------
    parser.add_argument("--base-url",
                        default=os.environ.get(
                            "SGLANG_ENGINE_QWEN36C_BASE_URL",
                            "http://localhost:8014"),
                        help="SGLang engine sidecar base URL (the Qwen3.6 "
                             "concurrent sidecar convention: port 8014); "
                             "localhost means single-host (loopback netem "
                             "on the base-url port), a remote host means "
                             "dual-host (whole-interface netem on this "
                             "machine's egress)")
    parser.add_argument("--profile", default="no_emulation",
                        help="network profile from configs/profiles.yaml, "
                             "applied as real topology-aware tc/netem "
                             "shaping (no_emulation = bypass)")
    parser.add_argument("--profile-record-only", action="store_true",
                        help="do NOT apply netem; only record --profile in "
                             "the DB (shape manually with tc instead)")
    parser.add_argument("--netem-iface", "--interface", dest="netem_iface",
                        default="auto",
                        help="egress interface for dual-host netem shaping "
                             "(alias: --interface; ignored single-host)")
    parser.add_argument("--netem-direction", default="both",
                        choices=["both", "uplink-only"],
                        help="both: shape upload and SSE download; "
                             "uplink-only: shape only client->server upload")
    # ---- wave topology / terminal resources ------------------------------
    parser.add_argument("--no-flush-cache", action="store_true",
                        help="skip the once-per-wave /flush_cache (measures "
                             "steady-state cache-hit behaviour instead of "
                             "cold prefixes)")
    parser.add_argument("--wait-sidecar", type=float, default=900,
                        metavar="SEC",
                        help="preflight GET /v1/models retry budget while "
                             "the sidecar builds its engine before opening "
                             "the port (900 default; 0 = try once and exit "
                             "on failure)")
    parser.add_argument("--no-model-check", action="store_true",
                        help="skip the preflight model-agreement check "
                             "(sidecar ids that are placeholders/anonymized "
                             "cannot be compared; a mismatch then only "
                             "warns)")
    parser.add_argument("--enable-multi-processes", action="store_true",
                        help="run each of the N users as an OS PROCESS "
                             "(own interpreter/GIL/connection; edge mode "
                             "also own ViT) instead of threads sharing one "
                             "process — truly independent terminals. Works "
                             "in BOTH modes")
    parser.add_argument("--edge-cpus", default=None, metavar="SPEC",
                        help="THREAD mode + edge mode only: pin THIS whole "
                             "process (decode + processor + ViT for all N "
                             "threads) to a cpu set, mirroring the serial "
                             "bench's budget. Default 'auto-half-other' "
                             "(second half of the cores, disjoint from the "
                             "sidecar's default --server-cpus auto-half); "
                             "ENCPL_EDGE_CPUS env wins as default. Process "
                             "mode: warn + skip (use "
                             "--edge-cpus-per-worker)")
    parser.add_argument("--edge-cpus-per-worker", type=int, default=None,
                        metavar="K",
                        help="process mode only (either arm): pin each "
                             "worker to K cpus (round-robin from the "
                             "--edge-cpu-ids pool, default K=1, pool=parent "
                             "affinity). 0 disables pinning. Thread mode "
                             "warns and skips")
    parser.add_argument("--edge-cpu-ids", default=None, metavar="SPEC",
                        help="cpu pool for --edge-cpus-per-worker, e.g. "
                             "\"0-7,32\" (default: every cpu this runner "
                             "may use); point it OUTSIDE the sidecar's "
                             "--server-cpus range so the arms do not "
                             "contend")
    # ---- bench output ----------------------------------------------------
    parser.add_argument("--output", default=None,
                        help="output dir (default "
                             "logs/encpl_bench_conc_q36/"
                             "<dataset>/"
                             "<mode>[-vidcom2-rR][-fp8][-fa4|-eager]"
                             "[-frR][-mfM]-n<N>)")
    parser.add_argument("--metrics-only", action="store_true",
                        help="only recompute accuracy/latency from an "
                             "existing results file (pass the same "
                             "--dataset/--mode/--users/--profile/--output "
                             "as the original run)")
    # ---- packet capture ---------------------------------------------------
    parser.add_argument("--capture-pcap", action="store_true",
                        help="L3/L4 tcpdump capture of the sidecar traffic "
                             "spanning the WHOLE wave loop (needs "
                             "passwordless sudo tcpdump; the tool EXITS if "
                             "the capture cannot start). Writes a "
                             "turn_index=-2 metadata row into the same "
                             "--db AND a sidecar JSON next to the results "
                             "file")
    parser.add_argument("--capture-interface", default=None,
                        help="interface to capture on (default: 'lo' when "
                             "--base-url is local, else the default-route "
                             "interface from 'ip route')")
    parser.add_argument("--capture-filter", default=None,
                        help="BPF filter (default: 'tcp port <sidecar port "
                             "from --base-url>')")
    parser.add_argument("--capture-dir",
                        default=str(AITESTBED_ROOT / "results" / "captures"),
                        help="directory for pcap files (default: "
                             "<repo>/aitestbed/results/captures, cwd-"
                             "independent)")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not args.dataset_dir:
        sys.exit("--dataset-dir (or the DATASET_DIR env var) is required")
    dataset_dir = Path(args.dataset_dir).expanduser()

    cfg = _load_scenario_config(Path(args.config))

    # CLI > yaml overlay (runner's flag_to_key, minus the dataset-supplied
    # video keys). store_true flags overlay only when set.
    flag_to_key = {
        "mode": "mode",
        "users": "concurrent_users",
        "compressor": "compressor",
        "retention_ratio": "retention_ratio",
        "optional_ratio": "optional_ratio",
        "opt_tokens_per_packet": "opt_tokens_per_packet",
        "opt_send_gap_ms": "opt_send_gap_ms",
        "opt_chat_delay_ms": "opt_chat_delay_ms",
        "bundle_quant": "bundle_quant",
        "attn_impl": "attn_impl",
        "model": "model",
        "max_frames": "max_frames",
        "frame_ratio": "frame_ratio",
        "decode_threads": "decode_threads",
        "max_tokens": "max_tokens",
        "edge_cpus_per_worker": "edge_cpus_per_worker",
        "edge_cpu_ids": "edge_cpu_ids",
    }
    for flag, key in flag_to_key.items():
        value = getattr(args, flag)
        if value is not None:
            cfg[key] = value
    if args.vision_only:
        cfg["vision_only"] = True
    if args.no_vision_only:
        cfg["vision_only"] = False
    if args.enable_multi_processes:
        cfg["edge_processes"] = True
    if args.no_flush_cache:
        cfg["flush_cache"] = False

    # ---- effective values (yaml > built-in; CLI already overlaid) -------
    mode = str(cfg.get("mode", "server")).strip().lower()
    if mode not in ("server", "edge"):
        sys.exit(f"invalid mode {mode!r}: must be 'server' or 'edge'")
    try:
        users = int(cfg.get("concurrent_users", 2))
    except (TypeError, ValueError):
        sys.exit("invalid concurrent_users: must be a positive integer")
    if users < 1:
        sys.exit(f"invalid concurrent_users {users}: must be >= 1")
    model = str(cfg.get("model") or DEFAULT_MODEL)
    compressor = str(cfg.get("compressor", "")).strip().lower()
    try:
        retention_ratio = float(cfg.get("retention_ratio", 0.25))
    except (TypeError, ValueError):
        sys.exit("invalid retention_ratio: must be a number in (0, 1]")
    if not (0.0 < retention_ratio <= 1.0):
        sys.exit("--retention-ratio must be in (0, 1]")
    bundle_quant = str(cfg.get("bundle_quant", "")).strip().lower()
    # Qwen3.6: vision_only defaults TRUE (yaml default true; the full
    # 35B fp8 load costs ~35 GB VRAM just to obtain the ~0.5 GB tower).
    vision_only = _yaml_bool(cfg.get("vision_only", True))
    attn_impl = cfg.get("attn_impl", "sdpa")
    try:
        attn_impl = _resolve_attn_impl(attn_impl)
    except ValueError as exc:
        sys.exit(f"invalid attn_impl: {exc}")
    max_frames = int(cfg.get("max_frames", 768))
    max_tokens = int(cfg.get("max_tokens", 256))
    frame_ratio = cfg.get("frame_ratio")
    try:
        frame_ratio = (float(frame_ratio)
                       if frame_ratio not in (None, "") else None)
    except (TypeError, ValueError):
        sys.exit(f"invalid frame_ratio {frame_ratio!r}: must be in (0, 1]")
    if frame_ratio == 0:
        frame_ratio = None
    if frame_ratio is not None and not (0.0 < frame_ratio <= 1.0):
        sys.exit("--frame-ratio must be in (0, 1] (0 or omit = off)")
    flush_cache = _yaml_bool(cfg.get("flush_cache", True))
    edge_processes = _yaml_bool(cfg.get("edge_processes", False))
    inter_run_delay = float(cfg.get("inter_prompt_delay_sec", 0) or 0)

    # ---- cross-knob validation (serial bench semantics) ------------------
    if compressor and mode != "edge":
        sys.exit("--compressor vidcom2 is only meaningful in edge mode "
                 "(the server path has no edge bundle to prune)")
    if bundle_quant:
        if mode != "edge":
            sys.exit("--bundle-quant fp8 is only meaningful in edge mode "
                     "(the server path has no edge bundle to quantize)")
        if compressor != "vidcom2":
            sys.exit("--bundle-quant fp8 requires --compressor vidcom2 "
                     "(quantization applies to the pruned bundle)")
    if attn_impl != "sdpa" and mode != "edge":
        sys.exit("--attn-impl is only meaningful in edge mode (server mode "
                 "lets SGLang pick its own backend)")
    # ---- two-tier (optional retention) knobs: same validation semantics
    # as the concurrent scenario (scenarios/
    # encoder_placement_concurrent_qwen36.py — config-fed, so the RANGE
    # check mirrors its hard raise); the edge+vidcom2 requirement is a
    # hard error (a silently-ignored optional_ratio would produce
    # results files that LOOK like a tiered measurement but are
    # single-tier).
    try:
        optional_ratio = float(cfg.get("optional_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        sys.exit("invalid optional_ratio: must be a number in "
                 "[0, retention_ratio)")
    if not (0.0 <= optional_ratio < retention_ratio):
        sys.exit(f"invalid optional_ratio {optional_ratio}: must be in "
                 f"[0, retention_ratio={retention_ratio})")
    if optional_ratio > 0 and (mode != "edge" or compressor != "vidcom2"):
        sys.exit("optional_ratio > 0 requires edge mode with compressor "
                 "vidcom2 (the optional tier is carved out of the pruned "
                 "edge bundle)")
    try:
        # NOTE: no `or 64` fold here — an EXPLICIT --opt-tokens-per-packet 0
        # must FAIL validation below, not silently become the default (the
        # scenario's config-fed `or` fold is fine because a config 0 is
        # meaningless there; a CLI 0 here is a user typo).
        opt_tokens_per_packet = int(
            cfg.get("opt_tokens_per_packet", 64))
    except (TypeError, ValueError):
        sys.exit("invalid opt_tokens_per_packet: must be a positive integer")
    if opt_tokens_per_packet < 1:
        sys.exit(f"invalid opt_tokens_per_packet {opt_tokens_per_packet}: "
                 f"must be >= 1")
    try:
        opt_send_gap_ms = float(cfg.get("opt_send_gap_ms", 0.05) or 0.0)
    except (TypeError, ValueError):
        sys.exit("invalid opt_send_gap_ms: must be a number >= 0")
    if opt_send_gap_ms < 0:
        sys.exit(f"invalid opt_send_gap_ms {opt_send_gap_ms}: must be >= 0")
    try:
        opt_chat_delay_ms = float(cfg.get("opt_chat_delay_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        sys.exit("invalid opt_chat_delay_ms: must be a number >= 0")
    if opt_chat_delay_ms < 0:
        sys.exit(f"invalid opt_chat_delay_ms {opt_chat_delay_ms}: "
                 f"must be >= 0")

    # ---- terminal CPU budget matrix --------------------------------------
    # --edge-cpus: thread+edge only (serial-bench whole-process pin).
    # --edge-cpus-per-worker / --edge-cpu-ids: process mode only (either
    # arm). Mismatched combos warn and skip so a stray env var cannot
    # silently change the resource model.
    edge_cpu_spec = (args.edge_cpus
                     if args.edge_cpus is not None
                     else os.environ.get("ENCPL_EDGE_CPUS",
                                         "auto-half-other"))
    edge_cpus_explicit = (args.edge_cpus is not None
                           or "ENCPL_EDGE_CPUS" in os.environ)
    try:
        own_affinity = os.sched_getaffinity(0)
    except AttributeError:
        own_affinity = None
    try:
        edge_cpu_list = _base._parse_edge_cpu_spec(edge_cpu_spec,
                                                   own_affinity)
    except ValueError as exc:
        sys.exit(f"--edge-cpus: {exc}")

    edge_cpus_per_worker = int(cfg.get("edge_cpus_per_worker", 1) or 0)
    edge_cpu_ids_raw = cfg.get("edge_cpu_ids")
    edge_cpu_plan = None
    if edge_processes:
        if edge_cpus_explicit:
            print(f"{LOG_PREFIX} warning: --edge-cpus={edge_cpu_spec!r} is "
                  f"set but process mode pins PER WORKER; use "
                  f"--edge-cpus-per-worker / --edge-cpu-ids instead "
                  f"(whole-process pin skipped)")
        edge_cpu_list = None
        if edge_cpus_per_worker > 0:
            if not hasattr(os, "sched_setaffinity"):
                print(f"{LOG_PREFIX} warning: os.sched_setaffinity "
                      f"unavailable on this platform; edge CPU pinning "
                      f"skipped")
            else:
                try:
                    pool = _parse_cpu_ids(edge_cpu_ids_raw,
                                          fallback=own_affinity)
                except ValueError as exc:
                    sys.exit(f"--edge-cpu-ids: {exc}")
                edge_cpu_plan = {"per_worker": edge_cpus_per_worker,
                                 "pool": pool}
        elif edge_cpu_ids_raw:
            print(f"{LOG_PREFIX} warning: --edge-cpu-ids given but "
                  f"--edge-cpus-per-worker<=0; edge CPU pinning disabled")
    else:
        if edge_cpus_per_worker > 0:
            print(f"{LOG_PREFIX} warning: --edge-cpus-per-worker="
                  f"{edge_cpus_per_worker} requires process mode "
                  f"(--enable-multi-processes); thread mode shares one "
                  f"process and cannot pin per-user — skipping CPU pinning")
        if edge_cpu_list and mode != "edge":
            print(f"{LOG_PREFIX} warning: --edge-cpus={edge_cpu_spec!r} is "
                  f"set but mode={mode!r} (no terminal-side model work); "
                  f"edge CPU pinning is ignored")
            edge_cpu_list = None
        if edge_cpu_spec and edge_cpu_list is None and own_affinity is None:
            print(f"{LOG_PREFIX} warning: --edge-cpus={edge_cpu_spec!r} "
                  f"given but os.sched_getaffinity is unavailable on this "
                  f"platform; edge CPU pinning is NOT applied")

    # ---- decord decode-thread cap ----------------------------------------
    decode_threads = (args.decode_threads
                      if args.decode_threads is not None
                      else os.environ.get("ENCPL_DECODE_THREADS"))
    if decode_threads is not None:
        try:
            decode_threads = int(decode_threads)
        except (TypeError, ValueError):
            sys.exit(f"--decode-threads must be a positive integer, got "
                     f"{decode_threads!r}")
        if decode_threads <= 0:
            sys.exit(f"--decode-threads must be > 0 (omit for decord "
                     f"native default), got {decode_threads}")
    if decode_threads is not None and mode != "edge":
        print(f"{LOG_PREFIX} warning: --decode-threads={decode_threads} is "
              f"set but mode={mode!r}; the ENGINE-side cap is the "
              f"sidecar's --decode-threads flag (restart it with the same "
              f"value), this flag is edge-only")
        decode_threads = None
    applied_decode_threads = None
    if decode_threads is not None and not edge_processes:
        os.environ.setdefault("SGLANG_DECORD_NUM_THREADS",
                              str(decode_threads))
        applied_decode_threads = int(os.environ["SGLANG_DECORD_NUM_THREADS"])
        print(f"{LOG_PREFIX} decode_threads: capped to "
              f"{applied_decode_threads} (SGLANG_DECORD_NUM_THREADS; env "
              f"name historical — caps BOTH backends; thread mode — "
              f"inherited by all {users} users)")
    elif decode_threads is not None:
        # process mode: handed to each worker via cfg (workers export
        # their own env before their first VideoReader).
        applied_decode_threads = decode_threads

    # A portless base URL would make the loopback netem filter + capture
    # filter default to another tool's port while the client connects
    # elsewhere; pin the Qwen3.6 CONCURRENT sidecar port explicitly
    # (8014, the qwen36 runner's convention) instead of the serial
    # sidecar's 8004.
    parsed = urlparse(args.base_url)
    if parsed.scheme and parsed.port is None:
        args.base_url = args.base_url.rstrip("/") + ":8014"
        print(f"{LOG_PREFIX} note: portless --base-url pinned to :8014 "
              f"(Qwen3.6 concurrent sidecar convention)")

    mode_dir = (mode + ("-vidcom2" if compressor else "")
                + (f"-r{retention_ratio:g}" if compressor else "")
                + (f"-opt{optional_ratio:g}"
                   if (compressor and optional_ratio > 0) else "")
                + ("-fp8" if bundle_quant else "")
                + ({"sdpa": "", "flash_attention_4": "-fa4",
                    "eager": "-eager"}[attn_impl])
                + (f"-fr{frame_ratio:g}" if frame_ratio else "")
                + ((f"-mf{max_frames}" if max_frames != 768 else ""))
                + f"-n{users}")
    # V>1 changes the workload, not the mode knobs; keep it in the dir
    # name so A/B comparisons never overwrite each other's results.
    videos_per_user_cli = args.videos_per_user
    if videos_per_user_cli is not None and videos_per_user_cli > 1:
        mode_dir += f"-v{videos_per_user_cli}"
    output_dir = Path(args.output or AITESTBED_ROOT / "logs"
                      / "encpl_bench_conc_q36" / args.dataset
                      / mode_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"questions_results_{args.profile}.json"

    if args.metrics_only:
        _base.compute_and_print(results_path, args.profile, model,
                                args.dataset)
        return

    # ---- dataset load + DISTINCT-video playlist building ------------------
    if args.dataset == "videomme":
        parquet_path = _vmme.find_parquet(dataset_dir)
        if parquet_path is None:
            sys.exit(f"no parquet found under {dataset_dir}")
        video_root = _vmme.find_video_root(dataset_dir)
        if video_root is None:
            sys.exit(f"no video dir (videozip/ or videos/) found under "
                     f"{dataset_dir}")
        entries = _vmme.load_entries(parquet_path, video_root,
                                      args.limit, args.sample)
    else:
        mlvu_root = _base.find_mlvu_root(dataset_dir)
        if mlvu_root is None:
            sys.exit(f"no MLVU layout (json/ + video/) found under "
                     f"{dataset_dir}")
        tasks = _base.resolve_mlvu_tasks(args.tasks)
        entries = _base.load_mlvu_entries(mlvu_root, tasks,
                                          args.limit, args.sample)

    if not entries:
        sys.exit("no questions loaded")

    # V = --videos-per-user, or limit // N when omitted (0/limitless -> 1
    # = the legacy one-video-per-user wave).
    if videos_per_user_cli is not None:
        if videos_per_user_cli < 1:
            sys.exit(f"--videos-per-user must be >= 1, got "
                     f"{videos_per_user_cli}")
        videos_per_user = videos_per_user_cli
    elif args.limit > 0:
        videos_per_user = max(1, args.limit // users)
    else:
        videos_per_user = 1

    user_playlists = _build_playlists(entries, users, videos_per_user)

    print(f"{LOG_PREFIX} dataset={args.dataset} mode={mode} "
          f"model={model}")
    print(f"{LOG_PREFIX} playlist: {users} users x {videos_per_user} "
          f"distinct videos = {users * videos_per_user} requests "
          f"(interleaved assign, deterministic, no inter-video barrier)")
    pool_entries = [e for pl in user_playlists for e in pl]
    if args.dataset == "mlvu":
        mix = Counter(e["domain"] for e in pool_entries)
        print(f"{LOG_PREFIX} task mix: "
              + ", ".join(f"{k}={v}" for k, v in sorted(mix.items())))
    else:
        dur_counts = Counter(e["duration"] for e in pool_entries)
        print(f"{LOG_PREFIX} duration mix: "
              + ", ".join(f"{k}={v}" for k, v in sorted(dur_counts.items())))

    client = EncoderPlacementQwen36Client(base_url=args.base_url)
    logger = TrafficLogger(args.db)
    trace_logger = TraceLogger.from_env()
    effective_base_url = client.base_url
    provider = client.provider
    model_key = Path(model).name

    # Sidecar preflight BEFORE anything destructive or expensive:
    # (a) connectivity with retry — the sidecar binds its port only
    # after the multi-minute engine build, so "bench started first"
    # waits here instead of producing a wave of connection-refused
    # failure rows; (b) model agreement — catches sidecar/client model
    # drift BEFORE the run (and before the results archive below), so a
    # failed preflight leaves the previous results file untouched.
    _base._preflight_sidecar(effective_base_url, model,
                             wait_sec=args.wait_sidecar,
                             check_model=not args.no_model_check,
                             log_prefix=LOG_PREFIX,
                             sidecar_hint=CONCURRENT_SIDECAR_SCRIPT)
    # Environment context for the run log (decode backend, thread caps,
    # patch-4 state) — best-effort, immediately after preflight and
    # BEFORE the results archive below so it precedes all run output.
    _base._print_sidecar_stats(effective_base_url,
                                log_prefix=LOG_PREFIX)
    # Tiered-run preflight: the optional-tier UDP receiver must be live
    # BEFORE the wave (a tiered edge run against a sidecar without
    # --opt-udp-port fails on EVERY request at the must PUT). The
    # returned port feeds the loopback netem udp filter + pcap filter.
    # Reuses the serial bench's helper (same /stats contract on both
    # sidecars).
    opt_udp_port = None
    if optional_ratio > 0:
        opt_udp_port = _base._preflight_opt_udp(effective_base_url,
                                                log_prefix=LOG_PREFIX)

    # Fresh report per invocation (single-wave tool, no --resume): archive
    # any previous results file so stale entries never pollute accuracy.
    if results_path.exists():
        archive = results_path.with_name(
            results_path.stem + "_" + time.strftime("%Y%m%d_%H%M%S")
            + ".json")
        os.replace(results_path, archive)
        print(f"{LOG_PREFIX} fresh report: archived previous results to "
              f"{archive.name}")

    # Edge CPU budget BEFORE the first request (thread+edge mode only —
    # process mode pins per worker inside the wave): lazy torch import +
    # model load land inside the budget; restored in the finally below.
    _cpu_restore = _base._apply_edge_cpu_budget(edge_cpu_list)
    applied_edge_cpus = (
        ",".join(str(c) for c in edge_cpu_list)
        if (_cpu_restore is not None) else None)

    # netem BEFORE the first request (direction-aware, topology-aware;
    # single-host asymmetric profiles are rejected inside apply_netem).
    # Tiered runs pass the sidecar's opt-udp port so the second-wave UDP
    # datagrams join the same shaping band (they would otherwise bypass
    # netem entirely on single-host loopback runs).
    if not args.profile_record_only:
        _vmme.apply_netem(effective_base_url, args.profile,
                          args.netem_direction, interface=args.netem_iface,
                          udp_port=opt_udp_port)

    # pcap capture spans the WHOLE wave loop (fail-fast preflight: a
    # requested capture that cannot start is a hard error).
    pcap_controller = None
    pcap_start_time = None
    pcap_iface = None
    capture_filter = None
    if args.capture_pcap:
        pcap_iface = (args.capture_interface
                      or _base._resolve_capture_interface(effective_base_url))
        _base._preflight_capture(pcap_iface, Path(args.capture_dir))
        sidecar_port = _vmme._base_url_port(effective_base_url, 8014)
        # Tiered runs: capture the UDP second wave too (same convention
        # as the matrix driver's bpf_filter).
        capture_filter = (args.capture_filter
                          or (f"tcp port {sidecar_port}"
                              + (f" or udp port {opt_udp_port}"
                                 if opt_udp_port else "")))
        try:
            pcap_controller = CaptureController(
                interface=pcap_iface,
                capture_dir=args.capture_dir,
            )
            ts = time.strftime("%Y%m%d_%H%M%S")
            arm = f"{args.dataset}_{mode}_n{users}"
            pcap_start_time = time.time()
            pcap_file = pcap_controller.start(
                filename=f"encpl_bench_conc_q36_{arm}_{args.profile}_{ts}.pcap",
                filter_expr=capture_filter,
            )
            if not pcap_file:
                sys.exit(f"--capture-pcap: tcpdump failed to start on "
                         f"'{pcap_iface}' (check the interface name and "
                         f"BPF filter '{capture_filter}')")
            print(f"{LOG_PREFIX} pcap: {pcap_file} "
                  f"(iface={pcap_iface} filter='{capture_filter}')")
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(f"--capture-pcap: failed to start tcpdump capture "
                     f"({e})")

    print(f"{LOG_PREFIX} scenario=encoder_placement_concurrent_qwen36 "
          f"dataset={args.dataset} mode={mode} users={users}"
          + (f" frame_ratio={frame_ratio}" if frame_ratio else "")
          + (f" optional_ratio={optional_ratio:g}"
             + f" opt_tokens_per_packet={opt_tokens_per_packet}"
             + f" opt_send_gap_ms={opt_send_gap_ms:g}"
             + f" opt_chat_delay_ms={opt_chat_delay_ms:g}"
             if optional_ratio > 0 else "")
          + (f" decode_threads={decode_threads}"
             if decode_threads else "")
          + (f" multi_processes={users}p" if edge_processes else "")
          + (f" edge_cpus={edge_cpu_plan['per_worker']}c/worker"
             f"@{len(edge_cpu_plan['pool'])}c-pool" if edge_cpu_plan else "")
          + f" model={model} profile={args.profile}"
          + (" (record-only)" if args.profile_record_only else "")
          + f" runs={args.runs} db={Path(args.db).resolve()}")

    # Thread mode: warm the client's process-global caches BEFORE any
    # user thread exists (N racing lazy transformers imports collide in
    # sys.modules). Process mode: workers warm their own before 'ready'.
    if not edge_processes:
        _warm_client_caches(client, model=model, mode=mode,
                            vision_only=vision_only, attn_impl=attn_impl)
        # Edge thread mode decodes HERE in THIS process: name the local
        # decode backend next to the sidecar /stats line (engine-side).
        # Process mode is covered per worker: each worker prints its own
        # backend after applying decode_threads (inherited env could
        # differ per worker spawn).
        if mode == "edge":
            _base._print_client_decode_backend(log_prefix=LOG_PREFIX)

    results: list[dict] = []
    waves: list[dict] = []
    interrupted = False
    try:
        for run_index in range(args.runs):
            if run_index > 0 and inter_run_delay > 0:
                time.sleep(inter_run_delay)
            # One flush per wave, BEFORE the users start: everyone in the
            # wave measures a cold prefix; nothing in-window pays for it.
            if flush_cache:
                _base.flush_server_cache(effective_base_url)

            wave_t0 = time.time()
            if edge_processes:
                video_records, interrupted = _run_wave_processes(
                    user_playlists=user_playlists,
                    model=model,
                    mode=mode,
                    max_frames=max_frames,
                    max_tokens=max_tokens,
                    compressor=compressor,
                    retention_ratio=retention_ratio,
                    bundle_quant=bundle_quant,
                    vision_only=vision_only,
                    attn_impl=attn_impl,
                    frame_ratio=frame_ratio,
                    decode_threads=decode_threads,
                    edge_cpu_plan=edge_cpu_plan,
                    base_url=effective_base_url,
                    optional_ratio=optional_ratio,
                    opt_tokens_per_packet=opt_tokens_per_packet,
                    opt_send_gap_ms=opt_send_gap_ms,
                    opt_chat_delay_ms=opt_chat_delay_ms,
                )
            else:
                video_records, interrupted = _run_wave_threads(
                    client=client,
                    user_playlists=user_playlists,
                    model=model,
                    mode=mode,
                    max_frames=max_frames,
                    max_tokens=max_tokens,
                    compressor=compressor,
                    retention_ratio=retention_ratio,
                    bundle_quant=bundle_quant,
                    vision_only=vision_only,
                    attn_impl=attn_impl,
                    frame_ratio=frame_ratio,
                    optional_ratio=optional_ratio,
                    opt_tokens_per_packet=opt_tokens_per_packet,
                    opt_send_gap_ms=opt_send_gap_ms,
                    opt_chat_delay_ms=opt_chat_delay_ms,
                )
            wall = time.time() - wave_t0

            wave = _wave_stats(video_records, users, wall,
                               videos_per_user)
            wave["run_index"] = run_index
            waves.append(wave)
            _print_wave_summary(wave)
            _print_per_user_detail(video_records,
                                   f"run {run_index}")

            # ---- per-video bench JSON entries + DB LogRecords ----------
            session_id = str(uuid.uuid4())
            for r in video_records:
                e = user_playlists[r["user_index"]][r["seq"]]
                q0 = {
                    "task_type": e["task_type"],
                    "question": e["question"],
                    "answer": e["answer"],
                    "options": [
                        o if ". " in o[:4] else f"{'ABCD'[i]}. {o}"
                        for i, o in enumerate(e["options"])],
                    model_key: "",
                }
                for extra in ("answer_text", "scoring_points"):
                    if e.get(extra):
                        q0[extra] = e[extra]
                entry = {
                    "time": "",
                    "captions": "",
                    "questions": [q0],
                    "video_categories": e["domain"],
                    "video_path": e["video_path"],
                    "video_id": e["video_id"],
                    "question": e["question"],
                    "duration": e["duration"],
                    "mode": mode,
                    "dataset": args.dataset,
                    "compressor": compressor,
                    "retention_ratio": (
                        retention_ratio if compressor else None),
                    "optional_ratio": (
                        float(optional_ratio)
                        if (compressor and optional_ratio) else None),
                    "bundle_quant": bundle_quant or None,
                    "vision_only": bool(vision_only),
                    "attn_impl": attn_impl,
                    "frame_ratio": (
                        float(frame_ratio) if frame_ratio else None),
                    "max_frames": int(max_frames) if max_frames else None,
                    "user_index": r["user_index"],
                    "seq": r["seq"],
                    "run_index": run_index,
                }

                if r.get("ok"):
                    entry["questions"][0][model_key] = (
                        r.get("total_content") or "")
                    stats = r.get("engine_stats")
                    if isinstance(stats, dict):
                        entry["encpl_stats"] = {
                            k: v for k, v in stats.items()
                            if k != "video_path"}
                    entry.setdefault("encpl_stats", {})
                    entry["encpl_stats"].update({
                        "edge_cpus": (r.get("edge_cpus")
                                      if edge_processes
                                      else applied_edge_cpus),
                        "decode_threads": applied_decode_threads,
                        "user_index": r["user_index"],
                        "seq": r["seq"],
                        "wave_size": users,
                        "videos_per_user": videos_per_user,
                        "wave_wall_sec": wall,
                        **({"wave_barrier_wall_sec":
                            r["wave_barrier_wall_sec"]}
                           if r.get("wave_barrier_wall_sec")
                           is not None else {}),
                        "ttft_sec": (None if r.get("ttft") is None
                                     else round(r["ttft"], 4)),
                        "latency_sec": (None if r.get("latency") is None
                                        else round(r["latency"], 4)),
                        "run_index": run_index,
                        "edge_processes": bool(r.get("edge_processes")),
                    })
                else:
                    entry["questions"][0][model_key] = ""
                    entry["error"] = r.get("error") or "request failed"

                results.append(entry)

                # ---- DB LogRecord (scenario schema + bench metadata) ----
                # turn_index = user_index * 1000 + seq: unique per video,
                # always >= 0 (metrics consumers filter turn_index < 0 as
                # pcap metadata rows), and decodable back to (user, seq).
                turn_index = r["user_index"] * 1000 + r["seq"]
                meta_common = {
                    "mode": mode,
                    "video_path": r.get("video_path"),
                    "wave_size": users,
                    "videos_per_user": videos_per_user,
                    "user_index": r["user_index"],
                    "seq": r["seq"],
                    "wave_wall_sec": wall,
                    **({"wave_barrier_wall_sec":
                        r["wave_barrier_wall_sec"]}
                       if r.get("wave_barrier_wall_sec")
                       is not None else {}),
                    "edge_processes": bool(r.get("edge_processes")),
                    "edge_cpus": (r.get("edge_cpus")
                                  if edge_processes
                                  else applied_edge_cpus),
                    "dataset": args.dataset,
                    "video_id": e["video_id"],
                    "question": e["question"],
                    "run_index": run_index,
                    "optional_ratio": (
                        float(optional_ratio) if optional_ratio else 0.0),
                    "opt_tokens_per_packet": int(opt_tokens_per_packet),
                    "opt_send_gap_ms": float(opt_send_gap_ms),
                    "opt_chat_delay_ms": float(opt_chat_delay_ms),
                }
                if r.get("ok"):
                    record = _make_log_record(
                        scenario_id="encoder_placement_concurrent_qwen36",
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=args.profile,
                        provider=provider,
                        model=model,
                        trace_logger=trace_logger,
                        request_bytes=r.get("request_bytes") or 0,
                        response_bytes=r.get("response_bytes") or 0,
                        tokens_in=r.get("tokens_in"),
                        tokens_out=r.get("tokens_out"),
                        t_request_start=r.get("t_request_start") or 0.0,
                        t_first_token=r.get("t_first_chunk"),
                        t_last_token=r.get("t_last_chunk"),
                        latency_sec=r.get("latency") or 0.0,
                        http_status=200,
                        success=True,
                        is_streaming=True,
                        chunk_count=r.get("chunk_count") or 0,
                        inter_chunk_times=json.dumps(
                            r.get("inter_chunk_times") or []),
                        trace_request=r.get("request_payload"),
                        trace_response=r.get("response_payload"),
                        trace_events=r.get("response_events"),
                        trace_note="encpl_bench_eval_concurrent_qwen36",
                        metadata=json.dumps({
                            **meta_common,
                            "ttft_sec": (None if r.get("ttft") is None
                                         else round(r["ttft"], 4)),
                            "latency_sec": (
                                None if r.get("latency") is None
                                else round(r["latency"], 4)),
                            **(r.get("engine_stats") or {}),
                        }),
                    )
                else:
                    record = _make_log_record(
                        scenario_id="encoder_placement_concurrent_qwen36",
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=args.profile,
                        provider=provider,
                        model=model,
                        trace_logger=trace_logger,
                        t_request_start=r.get("t_request_start") or 0.0,
                        latency_sec=r.get("latency") or 0.0,
                        http_status=0,
                        error_type=r.get("error") or "request failed",
                        success=False,
                        is_streaming=True,
                        metadata=json.dumps(meta_common),
                    )
                logger.log(record)

            _base._save_results(results_path, results)
            if interrupted:
                break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n{LOG_PREFIX} interrupted, saving partial results")
        _base._save_results(results_path, results)
    finally:
        # pcap stop BEFORE netem teardown so the capture window stays
        # strictly inside the shaped window; metadata lands BOTH in the
        # DB (turn_index=-2 row, pcap_analyzer/ran2_metrics pick it up
        # natively) AND in a sidecar JSON next to the results file.
        if pcap_controller is not None:
            try:
                pcap_stop_time = time.time()
                pcap_path = pcap_controller.stop()
                if pcap_path:
                    capture_duration = (
                        pcap_stop_time - pcap_start_time
                        if pcap_start_time is not None else None)
                    pcap_size = 0
                    try:
                        pcap_size = pcap_path.stat().st_size
                    except Exception:
                        pass
                    capture_metadata = {
                        "type": "pcap_capture",
                        "pcap_file": str(pcap_path),
                        "capture_dir": args.capture_dir,
                        "capture_filter": capture_filter,
                        "capture_duration_sec": capture_duration,
                        "interface": pcap_iface,
                        "success": pcap_size > 0,
                        "results_file": str(results_path),
                        "timestamp": time.time(),
                    }
                    # DB row first (turn_index=-2, the scenario's pcap
                    # metadata convention), then the sidecar JSON.
                    try:
                        capture_record = LogRecord(
                            timestamp=time.time(),
                            scenario_id=(
                                "encoder_placement_concurrent_qwen36"),
                            session_id=(
                                f"pcap_{int(pcap_start_time or time.time())}"),
                            turn_index=-2,
                            run_index=-1,
                            provider="tcpdump",
                            model="",
                            request_bytes=0,
                            response_bytes=pcap_size,
                            t_request_start=pcap_start_time or 0.0,
                            latency_sec=capture_duration or 0.0,
                            network_profile=args.profile,
                            http_status=200,
                            metadata=json.dumps(capture_metadata),
                            success=pcap_size > 0,
                        )
                        logger.log(capture_record)
                    except Exception as exc:
                        print(f"{LOG_PREFIX} warning: failed to log the "
                              f"pcap metadata row ({exc})")
                    sidecar = results_path.with_name(
                        results_path.stem + "_capture_meta.json")
                    try:
                        with open(sidecar, "w", encoding="utf-8") as f:
                            json.dump(capture_metadata, f, indent=2)
                    except Exception as exc:
                        print(f"{LOG_PREFIX} warning: failed to write the "
                              f"pcap metadata sidecar ({exc})")
                    print(f"{LOG_PREFIX} pcap stopped: {pcap_path} "
                          f"({pcap_size} bytes, meta -> {sidecar.name})"
                          + ("" if pcap_size else
                             " WARNING: 0 bytes captured — no packets "
                             "matched the filter/interface"))
            except Exception as e:
                print(f"{LOG_PREFIX} warning: failed to stop/log tcpdump "
                      f"capture cleanly ({e})")
        if not args.profile_record_only:
            _clear_netem(effective_base_url, args.netem_iface)
        _base._restore_edge_cpu_budget(_cpu_restore)

    _print_per_user_summary(waves)
    _print_final_summary(waves)
    _base.compute_and_print(results_path, args.profile, model,
                            args.dataset)


if __name__ == "__main__":
    main()
