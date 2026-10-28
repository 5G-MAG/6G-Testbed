#!/usr/bin/env python3
"""
Main Orchestrator for the 6G AI Traffic Testbed.

Coordinates scenario execution, network emulation, and data collection.
"""

import argparse
import logging
import colorlog
import time
import yaml
import json
import signal
import sys
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
from datetime import datetime
from dataclasses import asdict

# Bootstrap sibling packages (aicodec, netemu) relative to this file's
# location, so the testbed runs without pip-installing them and without
# setting PYTHONPATH. Works from any CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (_REPO_ROOT / "aicodec", _REPO_ROOT / "netemu" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # Loads from .env in current directory
load_dotenv(Path(__file__).parent / ".env")  # Also try aitestbed/.env
load_dotenv(Path(__file__).parent.parent / ".env")  # Also try repo root .env

from clients import OpenAIClient, GeminiClient, DeepSeekClient, VLLMClient, AzureOpenAIClient, AzureInferenceClient, RealtimeWebRTCVLMClient, OpenAITokenClient, EncoderPlacementClient, EncoderPlacementQwen36Client
from analysis import TrafficLogger, MetricsCalculator, LogRecord
from netemu import NetworkEmulator
from scenarios import (
    ChatScenario,
    ShoppingAgentScenario,
    WebSearchAgentScenario,
    GeneralAgentScenario,
    ImageGenerationScenario,
    MultimodalScenario,
    VideoUnderstandingScenario,
    ScenarioResult,
    DirectWebSearchScenario,
    ParallelSearchBenchmarkScenario,
    RealtimeConversationScenario,
    RealtimeWebRTCConversationScenario,
    RealtimeAudioScenario,
    RealtimeAudioWebRTCScenario,
    ComputerUseScenario,
    MusicAgentScenario,
    MusicResearchAgentScenario,
    PlaywrightAgentScenario,
    TradingAgentScenario,
    RealtimeVideoUnderstandingScenario,
    ChatTokenScenario,
    EncoderPlacementScenario,
    EncoderPlacementConcurrentScenario,
    EncoderPlacementQwen36Scenario,
    EncoderPlacementConcurrentQwen36Scenario
)

# Configure logging. scenarios/*, clients/*, analysis/* all log via
# logging.getLogger(__name__), which propagates to the root logger, not to
# a logger named "orchestrator" -- so the root logger still needs its own
# level+handler or every non-orchestrator INFO/DEBUG message is silently
# dropped (only WARNING+ survives, via Python's last-resort handler, and it
# never reaches logs/orchestrator.log either).
logging.basicConfig(level=logging.INFO)

# orchestrator's OWN logger: console (colored) + file handler, exactly one
# each (an earlier copy-paste duplication attached every handler twice,
# double-printing console lines and double-writing the log file).
logger = colorlog.getLogger("orchestrator")
console_handler = colorlog.StreamHandler()
console_handler.setFormatter(colorlog.ColoredFormatter(
    "%(asctime)s - %(name)s - %(log_color)s%(levelname)-8s%(reset)s - %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger.addHandler(console_handler)
Path('logs').mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler('logs/orchestrator.log', mode='w', encoding='utf-8')
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
# Don't also feed the root handler installed above -- it would double-print
# every message this logger already sends to its own console+file handlers.
logger.propagate = False



# ---------------------------------------------------------------------------
# Encoder-placement dataset-bench rich summary (bench-tool style)
#
# The encpl scenarios log per-video rows with metadata.bench + ttft_sec/
# latency_sec and per-wave stats in ScenarioResult.metadata["waves"].
# These module-level helpers turn a run's ScenarioResult list into the
# same rich end-of-run report the standalone bench tools print
# (tools/encpl_bench_eval*.py / tools/encpl_bench_db_report.py), so the
# orchestrator output is no longer limited to the 3-line real_* block.
# ---------------------------------------------------------------------------

def _bench_pct(values, p):
    """Linear-interpolated percentile (same semantics as metrics.py)."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def _collect_bench_rows(results: list) -> list[dict]:
    """Flat per-video bench rows from a run's ScenarioResult list.

    Each row: session_id, bench (parsed metadata.bench dict), ttft_sec,
    latency_sec, mode, user_index, seq, ok. Records without metadata
    (failed uploads) or without a bench object (single-video mode) are
    skipped, mirroring _collect_bench_entries.
    """
    rows = []
    for result in results:
        for record in getattr(result, "log_records", []):
            meta_raw = getattr(record, "metadata", None)
            if not meta_raw:
                continue
            try:
                meta = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            bench = meta.get("bench")
            if not isinstance(bench, dict):
                continue
            rows.append({
                "session_id": record.session_id,
                "bench": bench,
                "ttft_sec": meta.get("ttft_sec"),
                "latency_sec": meta.get("latency_sec"),
                "mode": meta.get("mode"),
                "user_index": meta.get("user_index"),
                "seq": meta.get("seq"),
                "ok": bool(getattr(record, "success", True)),
            })
    return rows


def _print_encpl_bench_summary(results: list, scenario_name: str,
                               profile_name: str) -> None:
    """Print the bench-tool-style rich end-of-run bench summary.

    Mirrors tools/encpl_bench_db_report.py report_combo() (OVERALL
    run-mean + pooled, accuracy by task_type/duration, per-run detail,
    E2E TTFT/LAT) plus the concurrent tool's first-video vs steady TTFT
    split and a compact per-user block from metadata["waves"]. Prints
    nothing when the run has neither bench rows nor wave stats (e.g.
    serial single-video mode), falling back to a db_report pointer.
    """
    rows = _collect_bench_rows(results)

    # per-user / first-vs-steady data from the waves metadata
    user_stats: dict[int, dict] = {}
    ttft_first: list[float] = []
    ttft_steady: list[float] = []
    for result in results:
        meta = getattr(result, "metadata", None) or {}
        for wave in meta.get("waves", []) or []:
            ttft_pu = wave.get("ttft_per_user") or []
            ok_pu = wave.get("ok_per_user") or []
            up_bytes = wave.get("upload_bytes_per_user") or []
            for u in range(len(ttft_pu)):
                st = user_stats.setdefault(u, {
                    "n_ok": 0, "n_total": 0, "ttfts": [],
                    "upload_bytes": 0,
                })
                v = ttft_pu[u]
                # playlist mode: nested list (one ttft per video, seq
                # order); single-video: scalar per user.
                vals = list(v) if isinstance(v, list) else [v]
                oks = None
                if u < len(ok_pu):
                    ov = ok_pu[u]
                    oks = list(ov) if isinstance(ov, list) else [ov]
                for j, t in enumerate(vals):
                    st["n_total"] += 1
                    # ok count follows the wave's ok_per_user flags so
                    # an EMPTY completion (ok=True, ttft=None) still
                    # counts as ok — only a real FAIL shows 0/N. ttfts
                    # / first-steady, like the scenario's own wave
                    # aggregation, collect from OK rows only (a failed
                    # row's residual ttft is not a decode measurement).
                    row_ok = (oks[j] if oks is not None and
                              j < len(oks) else t is not None)
                    if row_ok:
                        st["n_ok"] += 1
                        if t is not None:
                            st["ttfts"].append(t)
                            (ttft_first if j == 0 else
                             ttft_steady).append(t)
                    elif t is not None:
                        st["ttfts"].append(t)
                if u < len(up_bytes):
                    b = up_bytes[u]
                    if isinstance(b, (int, float)):
                        st["upload_bytes"] += b

    has_bench = bool(rows)
    has_waves = bool(user_stats)
    if not has_bench and not has_waves:
        print(f"\n[EncPlBench] {scenario_name}/{profile_name}: no bench "
              f"rows and no wave stats in this run's results "
              f"(single-video mode, all runs skipped via --resume, or "
              f"all failed).")
        print("[EncPlBench] historical bench data can be reported from "
              "the DB with:")
        print("[EncPlBench]   python tools/encpl_bench_db_report.py "
              f"--db logs/traffic_logs.db --scenario {scenario_name} "
              f"--profile {profile_name}")
        return

    print(f"\n[EncPlBench] ===== {scenario_name} / {profile_name} =====")

    if not has_bench:
        print(f"  0 scored bench entries in this run "
              f"(waves present; see note above for DB history)")

    if has_bench:
        scored = [r for r in rows if not r["bench"].get("padded")]
        padded = len(rows) - len(scored)
        # dataset from the first row that has one (all-padded edge case:
        # scored is empty but the dataset is still knowable from rows)
        dataset = next((r["bench"].get("dataset") for r in rows
                        if r["bench"].get("dataset")), "?")
        print(f"  dataset: {dataset}, {len(scored)} scored"
              + (f", {padded} padded skipped" if padded else ""))

        mode_counts = {}
        for r in scored:
            mode_counts[r["mode"] or "?"] = (
                mode_counts.get(r["mode"] or "?", 0) + 1)
        for m, n in sorted(mode_counts.items()):
            print(f"  mode {m}: n={n}")

        agg = MetricsCalculator.aggregate_bench_accuracy(
            [(r["session_id"], r["bench"]) for r in scored])
        if agg is None:
            print(f"  {len(scored)} bench row(s) but 0 scored MCQ "
                  f"entries (generation tasks / non-A-D answers are "
                  f"recorded unscored)")
        else:
            print(f"\n  OVERALL (run-mean over {agg['runs']} run(s)): "
                  f"{agg['accuracy']:.3f}   "
                  f"[pooled: {agg['correct']}/{agg['total']}]"
                  + (f"   parse_fail={agg['parse_fail']}"
                     if agg["parse_fail"] else ""))
            if agg["empty_runs"]:
                print(f"  ({agg['empty_runs']} empty run(s) excluded "
                      f"from the mean)")

            # accuracy by task_type (MLVU) / duration (Video-MME)
            by_bucket = {}
            label = "duration" if dataset == "videomme" else "task_type"
            for r in scored:
                b = r["bench"]
                answer = (b.get("answer") or "").strip().upper()
                if answer not in ("A", "B", "C", "D"):
                    continue
                key = (str(b.get("duration") or "?") if dataset == "videomme"
                       else str(b.get("task_type") or "?"))
                v = by_bucket.setdefault(key, {"total": 0, "correct": 0})
                v["total"] += 1
                if b.get("model_answer") and (
                        b["correct"] if b["correct"] is not None
                        else b["model_answer"] == answer):
                    v["correct"] += 1
            if by_bucket:
                print(f"\n  accuracy by {label}:")
                for k, v in sorted(by_bucket.items()):
                    acc = (v["correct"] / v["total"]
                           if v["total"] else 0.0)
                    print(f"    {k:35s} total={v['total']:5d} "
                          f"correct={v['correct']:5d} accuracy={acc:.3f}")

            # per-run (session) detail
            per_run = {}
            for r in scored:
                b = r["bench"]
                answer = (b.get("answer") or "").strip().upper()
                if answer not in ("A", "B", "C", "D"):
                    continue
                v = per_run.setdefault(
                    r["session_id"], {"total": 0, "correct": 0})
                v["total"] += 1
                if b.get("model_answer") and (
                        b["correct"] if b["correct"] is not None
                        else b["model_answer"] == answer):
                    v["correct"] += 1
            print(f"\n  per-run (session) detail:")
            for sid, v in sorted(per_run.items()):
                acc = v["correct"] / v["total"] if v["total"] else 0.0
                print(f"    {sid[:40]:40s} total={v['total']:4d} "
                      f"correct={v['correct']:4d} accuracy={acc:.3f}")

        # E2E timing summary (scored, non-padded rows only).
        # Truthiness filter (>0), not "is not None": a zero-token
        # completion records latency_sec=0.0, which must stay OUT of the
        # E2E stats exactly like the scenario's own wave aggregation
        # keeps it out of latency_mean (EMPTY, not FAIL — see the wave
        # stats' wave_empty).
        ttfts = [r["ttft_sec"] for r in scored
                 if isinstance(r["ttft_sec"], (int, float))
                 and r["ttft_sec"] > 0]
        lats = [r["latency_sec"] for r in scored
                if isinstance(r["latency_sec"], (int, float))
                and r["latency_sec"] > 0]
        if ttfts:
            print(f"\n  E2E TTFT  (n={len(ttfts)}): "
                  f"mean={sum(ttfts)/len(ttfts):.2f}s "
                  f"p50={_bench_pct(ttfts, 50):.2f}s "
                  f"p90={_bench_pct(ttfts, 90):.2f}s "
                  f"max={max(ttfts):.2f}s")
        if lats:
            print(f"  E2E LAT   (n={len(lats)}): "
                  f"mean={sum(lats)/len(lats):.2f}s "
                  f"p50={_bench_pct(lats, 50):.2f}s "
                  f"p90={_bench_pct(lats, 90):.2f}s "
                  f"max={max(lats):.2f}s")

    # first-video vs steady TTFT (playlist mode's core split; in
    # single-video mode every request is first, so no steady line)
    if ttft_first:
        print(f"\n  TTFT first-video (seq==0, n={len(ttft_first)}): "
              f"mean={sum(ttft_first)/len(ttft_first):.2f}s "
              f"p50={_bench_pct(ttft_first, 50):.2f}s "
              f"p95={_bench_pct(ttft_first, 95):.2f}s "
              f"max={max(ttft_first):.2f}s")
    if ttft_steady:
        print(f"  TTFT steady      (seq>0,  n={len(ttft_steady)}): "
              f"mean={sum(ttft_steady)/len(ttft_steady):.2f}s "
              f"p50={_bench_pct(ttft_steady, 50):.2f}s "
              f"p95={_bench_pct(ttft_steady, 95):.2f}s "
              f"max={max(ttft_steady):.2f}s")

    # compact per-user block (from waves; user_index order)
    if user_stats:
        print(f"\n  per-user summary:")
        for u in sorted(user_stats):
            st = user_stats[u]
            ts = st["ttfts"]
            if ts:
                t_line = (f"ttft mean={sum(ts)/len(ts):.2f} "
                          f"p50={_bench_pct(ts, 50):.2f} "
                          f"p90={_bench_pct(ts, 90):.2f}")
            else:
                t_line = "ttft n/a"
            upload_mib = st["upload_bytes"] / (1024 * 1024)
            print(f"    u{u}: ok={st['n_ok']}/{st['n_total']} | {t_line} "
                  f"| upload={upload_mib:.2f} MiB")


def _print_encpl_per_user_detail(results: list) -> None:
    """Per-run x per-wave x per-user TTFT/latency detail (runner parity).

    Ported verbatim from run_encoder_placement_concurrent{,_qwen36}.py
    _print_per_user_summary + _print_final_summary so the orchestrator
    path prints the same per-user detail the standalone runners do.

    Reads each ScenarioResult.metadata['waves'] (filled by the
    concurrent scenarios):
      [{wave_index, wave_size, ok_per_user: [...], ttft_per_user: [...],
        latency_per_user: [...], wave_empty, ttft_mean/p50/p90/max,
        wave_wall_sec, wave_ok, ...}, ...]

    Cell semantics: ttft/latency pair per user —
      "82.31/88.19"  ok, normal numbers;
      "EMPTY/-"      ok=True but the engine produced ZERO content tokens
                     (greedy first-token EOS / retract; sidecar logged a
                     zero-token-completion warning with finish_reason);
      "FAIL/-"       ok=False (real request failure — error in the DB).

    Playlist mode (bench dataset): ttft/latency/ok/video arrive as a
    PER-VIDEO list per user (seq order) — rendered as one indented line
    per video under the user, upload is the whole-playlist total, and
    the shown conns are the FIRST video's (barrier-aligned start;
    later videos open fresh connections — full detail in the DB rows).

    Layout: one user PER LINE with its upload size/time, FULL 5-tuples
    (src_ip:src_port -> dst_ip:dst_port for the upload PUT and the chat
    SSE legs) and FULL video path on indented follow-up lines
    (heterogeneous waves: which video caused which TTFT is readable at
    a glance); the wave header carries only the aggregate p50/p90/max.
    src_port is the pcap<->user join key (all flows share dst ip/port).
    Fields absent in wave metadata (older scenario / mock clients)
    degrade silently — lines are omitted, never crash the printer.
    Serial scenarios (no waves in metadata) print nothing.
    """
    has_waves = False
    for result in results:
        waves = (getattr(result, "metadata", None) or {}).get("waves") or []
        if not waves:
            continue
        has_waves = True
        run_index = getattr(result, "run_index", 0)
        print(f"\n[EncPlBench] per-user summary (run {run_index}):")
        for w in waves:
            size = w.get("wave_size") or 0
            ttfts = w.get("ttft_per_user") or [None] * size
            lats = w.get("latency_per_user") or [None] * size
            oks = w.get("ok_per_user") or [False] * size
            conns = w.get("conns_per_user") or [None] * size
            vids = w.get("video_paths_per_user") or [None] * size
            up_bytes = w.get("upload_bytes_per_user") or [None] * size
            up_times = w.get("upload_time_per_user") or [None] * size
            p50 = w.get("ttft_p50")
            p90 = w.get("ttft_p90")
            mx = w.get("ttft_max")
            empty = w.get("wave_empty")
            empty_s = f" empty={empty}" if empty else ""
            # wave header: the aggregate stats only; every user's detail
            # moves to its own line below (full video path per user).
            # Playlist mode: ok counts VIDEOS (wave_videos), not users.
            n_videos = w.get("wave_videos")
            denom = n_videos if n_videos is not None else size
            print(f"  wave {w.get('wave_index')} (ok={w.get('wave_ok')}/"
                  f"{denom}{empty_s}) || ttft p50={p50 and round(p50, 2)} "
                  f"p90={p90 and round(p90, 2)} "
                  f"max={mx and round(mx, 2)} s")
            for u in range(size):
                t = ttfts[u] if u < len(ttfts) else None
                l = lats[u] if u < len(lats) else None
                ok_u = oks[u] if u < len(oks) else False
                v = vids[u] if u < len(vids) else None
                if isinstance(t, list):
                    # playlist mode: per-video lines under the user
                    l_list = l if isinstance(l, list) else [None] * len(t)
                    ok_list = (ok_u if isinstance(ok_u, list)
                               else [False] * len(t))
                    v_list = v if isinstance(v, list) else [None] * len(t)
                    n_ok = sum(1 for o in ok_list if o)
                    print(f"    u{u}: {n_ok}/{len(t)} videos ok")
                    for s in range(len(t)):
                        tv = t[s] if s < len(t) else None
                        lv = l_list[s] if s < len(l_list) else None
                        ok_v = ok_list[s] if s < len(ok_list) else False
                        if not ok_v:
                            t_s = "FAIL"
                        elif tv is None:
                            t_s = "EMPTY"
                        else:
                            t_s = f"{tv:.2f}"
                        l_s = f"{lv:.2f}" if lv is not None else "-"
                        print(f"      v{s}: ttft={t_s} lat={l_s} s")
                        vv = v_list[s] if s < len(v_list) else None
                        if vv:
                            print(f"        video: {vv}")
                else:
                    if not ok_u:
                        t_s = "FAIL"
                    elif t is None:
                        t_s = "EMPTY"
                    else:
                        t_s = f"{t:.2f}"
                    l_s = f"{l:.2f}" if l is not None else "-"
                    print(f"    u{u}: ttft={t_s} lat={l_s} s")
                    if v:
                        print(f"      video: {v}")
                ub = up_bytes[u] if u < len(up_bytes) else None
                ut = up_times[u] if u < len(up_times) else None
                if ub is not None or ut is not None:
                    ub_s = (f"{ub / (1024 * 1024):.2f} MiB ({ub} B)"
                            if ub is not None else "-")
                    ut_s = f"{ut:.2f}" if ut is not None else "-"
                    print(f"      upload: {ub_s} in {ut_s} s")
                c = conns[u] if u < len(conns) else None
                if c:
                    up = c.get("upload") or {}
                    ch = c.get("chat") or {}
                    if up:
                        print(f"      up:  {up.get('src_ip')}:"
                              f"{up.get('src_port')} -> "
                              f"{up.get('dst_ip')}:{up.get('dst_port')}")
                    if ch:
                        print(f"      chat:{ch.get('src_ip')}:"
                              f"{ch.get('src_port')} -> "
                              f"{ch.get('dst_ip')}:{ch.get('dst_port')}")
    if not has_waves:
        return
    # cross-run TTFT stats over every wave of every result (runner's
    # _print_final_summary). These are PER-WAVE aggregates (each wave
    # contributes its own p50/p90/max), NOT pooled per-request
    # percentiles — matching the standalone runners' line verbatim;
    # the pooled per-request percentiles live in
    # _print_encpl_bench_per_user_stats's "overall TTFT/lat" lines.
    all_p50 = []
    all_p90 = []
    all_max = []
    total_empty = 0
    for r in results:
        for w in (getattr(r, "metadata", None) or {}).get("waves") or []:
            if w.get("ttft_p50") is not None:
                all_p50.append(w["ttft_p50"])
            if w.get("ttft_p90") is not None:
                all_p90.append(w["ttft_p90"])
            if w.get("ttft_max") is not None:
                all_max.append(w["ttft_max"])
            total_empty += w.get("wave_empty") or 0
    if all_p50:
        print(f"\n[EncPlBench] overall TTFT across {len(all_p50)} waves "
              f"(per-wave p50/p90, median-of-waves): "
              f"p50={_bench_pct(all_p50, 50):.2f} "
              f"p90={_bench_pct(all_p90, 50):.2f} "
              f"max={max(all_max):.2f} s"
              + (f" (empty={total_empty})" if total_empty else ""))


def _encpl_collect_video_records(results: list) -> list[dict]:
    """Rebuild the bench tool's per-video record dicts from LogRecords.

    Mirrors the in-memory video_records the concurrent bench tool
    (tools/encpl_bench_eval_concurrent*.py) feeds its printers: one
    dict per logged video with ok / user_index / seq / video_path /
    video_id / ttft / latency / engine_stats / error. Records without
    user_index (serial rows, pcap metadata) are skipped.
    """
    recs = []
    for result in results:
        for record in getattr(result, "log_records", []):
            meta_raw = getattr(record, "metadata", None)
            if not meta_raw:
                continue
            try:
                meta = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if meta.get("user_index") is None:
                continue
            recs.append({
                "user_index": meta.get("user_index"),
                "seq": meta.get("seq") or 0,
                "ok": bool(getattr(record, "success", True)),
                "video_path": meta.get("video_path"),
                "video_id": (meta.get("bench") or {}).get("video_id"),
                "ttft": meta.get("ttft_sec"),
                "latency": meta.get("latency_sec"),
                "engine_stats": meta,  # flattened (engine_stats keys
                # merged into metadata at log time — conns/upload_*
                # read off it directly)
                "error": getattr(record, "error_type", None),
            })
    return recs


def _encpl_stat_block(values: list) -> dict:
    """mean/p50/p90/p95/max over raw per-request values (None-safe)."""
    return {
        "n": len(values),
        "mean": (sum(values) / len(values)) if values else None,
        "p50": _bench_pct(values, 50),
        "p90": _bench_pct(values, 90),
        "p95": _bench_pct(values, 95),
        "max": max(values) if values else None,
    }


def _encpl_fmt_block(prefix: str, b: dict) -> str:
    if not b or not b.get("n"):
        return f"{prefix} n=0"
    return (f"{prefix} mean={b['mean']:.1f} p50={b['p50']:.1f} "
            f"p90={b['p90']:.1f} p95={b['p95']:.1f} "
            f"max={b['max']:.1f} s (n={b['n']})")


def _encpl_fmt_5tuple(t) -> str:
    """`src_ip:src_port -> dst_ip:dst_port` from a conn_5tuple dict."""
    if not t or not isinstance(t, dict):
        return "-"
    src = f"{t.get('src_ip')}:{t.get('src_port')}"
    dst = f"{t.get('dst_ip')}:{t.get('dst_port')}"
    return f"{src} -> {dst}"


def _encpl_fmt_bytes(n) -> str:
    """Human MiB with raw bytes (None -> '-')."""
    if n is None:
        return "-"
    mib = n / (1024 * 1024)
    return f"{mib:.2f} MiB ({n} B)"


def _print_encpl_bench_per_user_stats(results: list) -> None:
    """Bench-tool-style per-user + pooled stats from this run's records.

    Ported from tools/encpl_bench_eval_concurrent*.py
    _print_per_user_summary + _print_per_user_detail +
    _print_final_summary, rebuilt from the run's LogRecords (the bench
    tool reads its in-memory video_records; the orchestrator path only
    has the DB-bound rows, so the record dicts are reconstructed from
    metadata JSON first — see _encpl_collect_video_records).

    Aggregation semantics identical to the bench tool's _wave_stats:
    ttfts collect ok rows with ttft not None; latency uses the
    truthiness filter (>0) so empty completions (latency 0.0) stay out;
    first = seq==0, steady = seq>0 (ok rows only). Records without
    user_index (serial scenarios, pcap rows) print nothing.
    """
    video_records = _encpl_collect_video_records(results)
    if not video_records:
        return

    ok = [r for r in video_records if r.get("ok")]
    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    # Truthiness filter (not "is not None"): an empty completion
    # records latency=0.0, which must stay OUT of the stats just like
    # the scenarios' own wave aggregation.
    lats = [r["latency"] for r in ok if r["latency"]]
    first = [r["ttft"] for r in ok
             if r.get("seq") == 0 and r.get("ttft") is not None]
    steady = [r["ttft"] for r in ok
              if (r.get("seq") or 0) > 0 and r.get("ttft") is not None]
    ttft_by_user: dict[int, list] = {}
    lat_by_user: dict[int, list] = {}
    for r in ok:
        u = r.get("user_index")
        if u is None:
            continue
        if r.get("ttft") is not None:
            ttft_by_user.setdefault(u, []).append(r["ttft"])
        # Same truthiness filter as `lats` above.
        if r.get("latency"):
            lat_by_user.setdefault(u, []).append(r["latency"])

    users = sorted({r["user_index"] for r in video_records})
    total_ok = len(ok)
    total_fail = len(video_records) - total_ok
    total_empty = sum(1 for r in ok if r.get("ttft") is None)

    # ---- per-user summary (bench tool's _print_per_user_summary) ----
    # Each user row aggregates the user's own requests across runs:
    # ttft/lat mean/p50/p90/p95/max, first vs steady split. n_ok carries
    # a denominator (the user's total logged records incl. failed /
    # placeholder rows) so multi-run mixing is readable at a glance —
    # the bench tool's n_ok/{videos_per_user * waves} equivalent.
    print(f"\n[EncPlBench] per-user stats ({len(users)} user(s)):")
    for u in users:
        tv = ttft_by_user.get(u) or []
        lv = lat_by_user.get(u) or []
        fv = [r["ttft"] for r in ok
              if r.get("user_index") == u and r.get("seq") == 0
              and r.get("ttft") is not None]
        sv = [r["ttft"] for r in ok
              if r.get("user_index") == u and (r.get("seq") or 0) > 0
              and r.get("ttft") is not None]
        n_ok = sum(1 for r in video_records
                   if r.get("user_index") == u and r.get("ok"))
        n_total = sum(1 for r in video_records
                      if r.get("user_index") == u)
        first_desc = (f"first={fv[0]:.2f}" if fv else "first=-")
        steady_desc = (f"steady_mean={sum(sv) / len(sv):.2f}"
                       if sv else "steady_mean=-")
        print(f"  u{u}: n_ok={n_ok}/{n_total} | "
              + _encpl_fmt_block("ttft", _encpl_stat_block(tv)))
        print(f"{' ' * (6 + len(f'u{u}'))} "
              + _encpl_fmt_block("lat", _encpl_stat_block(lv))
              + f" | {first_desc} {steady_desc}")

    # ---- per-user detail (bench tool's _print_per_user_detail) ------
    # FULL listing (one block per user, one line per video): seq,
    # video_id + FULL video_path, upload & chat 5-tuples, uploaded
    # payload / wire bytes, upload_time, ttft / lat. Failure rows
    # carry the error string instead of the numeric tail.
    print(f"[EncPlBench] per-user detail ({len(video_records)} video(s)):")
    by_user: dict[int, list] = {}
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
            head = f"    [v{seq}] {vid} ttft={ttft_s} lat={lat_s} s"
            if r.get("ok"):
                up_b = (stats.get("uploaded_payload_bytes")
                        if stats.get("uploaded_payload_bytes") is not None
                        else stats.get("video_payload_bytes"))
                wire_b = stats.get("uploaded_wire_bytes")
                up_t = stats.get("upload_time")
                up_t_s = f"{up_t:.2f}" if up_t is not None else "-"
                head += (f" | upload={_encpl_fmt_bytes(up_b)}"
                         + (f" wire={_encpl_fmt_bytes(wire_b)}"
                            if wire_b is not None else "")
                         + f" up_time={up_t_s} s")
            else:
                head += f" | error={r.get('error') or '?'}"
            print(head)
            print(f"        video: {path}")
            if conns.get("upload"):
                print(f"        up:  "
                      f"{_encpl_fmt_5tuple(conns.get('upload'))}")
            if conns.get("chat"):
                print(f"        chat:"
                      f"{_encpl_fmt_5tuple(conns.get('chat'))}")

    # ---- pooled final stats (bench tool's _print_final_summary) -----
    # TTFT and lat each get mean/p50/p90/p95/max over the pooled
    # per-request values, plus the first-video (seq==0) vs
    # steady-state (seq>0) TTFT split and ok/fail/empty counts.
    if ttfts or lats:
        n_desc = (f"{total_ok} ok / {total_fail} fail / "
                  f"{total_empty} empty")
        if ttfts:
            print(f"\n[EncPlBench] overall TTFT ({n_desc}): "
                  + _encpl_fmt_block("", _encpl_stat_block(ttfts)).strip())
        if lats:
            print(f"[EncPlBench] overall lat  ({n_desc}): "
                  + _encpl_fmt_block("", _encpl_stat_block(lats)).strip())
        if first and steady:
            print(f"[EncPlBench] TTFT first-video: "
                  f"mean={sum(first) / len(first):.2f} n={len(first)} | "
                  f"steady (seq>0): "
                  f"mean={sum(steady) / len(steady):.2f} "
                  f"p50={_bench_pct(steady, 50):.2f} "
                  f"p95={_bench_pct(steady, 95):.2f} "
                  f"n={len(steady)}")


class RunFailedError(Exception):
    """Raised when a run fails and --stop-on-error is active."""
    pass


class TestbedOrchestrator:
    """
    Main orchestrator for running traffic characterization experiments.
    """

    def __init__(
        self,
        config_path: str = "configs/scenarios.yaml",
        profiles_path: str = "configs/profiles.yaml",
        db_path: str = "logs/traffic_logs.db",
        network_interface: str = "auto",
        egress_only: bool = False,
        mcp_transport: str = "http",
    ):
        """
        Initialize the orchestrator.

        Args:
            config_path: Path to scenarios configuration
            profiles_path: Path to network profiles configuration
            db_path: Path to SQLite database for logs
            network_interface: Network interface for emulation
            egress_only: If True, only shape egress traffic (default: False, shapes both)
        """
        self.config_path = Path(config_path)
        self.profiles_path = Path(profiles_path)

        # Load configurations
        with open(self.config_path) as f:
            self.scenarios_config = yaml.safe_load(f)
        # Re-parse profiles.yaml as a raw dict so we retain access to optional
        # `uplink:` blocks (netemu's NetworkProfile dataclass strips them).
        with open(self.profiles_path) as f:
            self.profiles_config = yaml.safe_load(f) or {}

        # Initialize components
        self.logger = TrafficLogger(db_path)
        self.emulator = NetworkEmulator(
            interface=network_interface,
            profiles_path=str(self.profiles_path),
            bidirectional=not egress_only
        )
        # Expose the resolved interface (in case "auto" was passed)
        self.network_interface = self.emulator.interface
        # keep the raw CLI value: "auto" must NOT activate the ICE filter
        # (WebRTC scenarios filter candidates only on an explicit interface)
        self.raw_network_interface = network_interface
        self.mcp_transport = mcp_transport

        # Validate any asymmetric `uplink:` blocks now so bad YAML fails fast.
        self._validate_asymmetric_profiles()

        # Initialize clients (lazy loaded based on provider)
        self._clients = {}

        # Scenario type mapping
        self.scenario_classes = {
            "chat": ChatScenario,
            "agent": ShoppingAgentScenario,
            "shopping_agent": ShoppingAgentScenario,
            "web_search_agent": WebSearchAgentScenario,
            "general_agent": GeneralAgentScenario,
            "image": ImageGenerationScenario,
            "image_generation": ImageGenerationScenario,
            "multimodal": MultimodalScenario,
            "video_understanding": VideoUnderstandingScenario,
            "video": VideoUnderstandingScenario,
            "computer_use": ComputerUseScenario,
            # Music agent scenarios (Spotify MCP)
            "music_agent": MusicAgentScenario,
            "music_research_agent": MusicResearchAgentScenario,
            # Playwright browser automation agent
            "playwright_agent": PlaywrightAgentScenario,
            # Trading / market data agent (Alpaca MCP)
            "trading_agent": TradingAgentScenario,
            # Direct search scenarios (no MCP)
            "direct_search": DirectWebSearchScenario,
            "direct_web_search": DirectWebSearchScenario,
            "parallel_search_benchmark": ParallelSearchBenchmarkScenario,
            # Real-time conversational AI scenarios
            "realtime_conversation": RealtimeConversationScenario,
            "realtime_conversation_webrtc": RealtimeWebRTCConversationScenario,
            "realtime_audio": RealtimeAudioScenario,
            "realtime_audio_webrtc": RealtimeAudioWebRTCScenario,
            # Real-time video understanding scenario
            "realtime_video_understanding": RealtimeVideoUnderstandingScenario,
            # chat with token ID
            "chat_token": ChatTokenScenario,
            # Encoder placement (server-side vs terminal-side visual encoder)
            "encoder_placement": EncoderPlacementScenario,
            # Encoder placement under concurrency — Qwen3-VL concurrent
            # sidecar 8012 (N barrier-aligned users per wave; orchestrator
            # yaml keys mirror configs/scenarios_encoder_placement_
            # concurrent.yaml; point it at the 8012 sidecar via env
            # SGLANG_ENGINE_BASE_URL=http://localhost:8012 — the same
            # provider/client as the serial 8002 scenario, so the env is
            # set per orchestrator invocation)
            "encoder_placement_concurrent": EncoderPlacementConcurrentScenario,
            # Encoder placement, Qwen3.6 sibling — serial sidecar 8004
            "encoder_placement_qwen36": EncoderPlacementQwen36Scenario,
            # Encoder placement, Qwen3.6 sibling — concurrent sidecar 8014
            # (N barrier-aligned users per wave; orchestrator yaml keys
            # mirror configs/scenarios_encoder_placement_concurrent_
            # qwen36.yaml; point it at the 8014 sidecar via env
            # SGLANG_ENGINE_QWEN36_BASE_URL=http://localhost:8014)
            "encoder_placement_concurrent_qwen36":
                EncoderPlacementConcurrentQwen36Scenario

        }

    # ------------------------------------------------------------------
    # Asymmetric profile support (S4-260848 Table C.Z-1)
    # ------------------------------------------------------------------
    # Asymmetric profile support (S4-260848 Table C.Z-1)
    # ------------------------------------------------------------------

    # Fields that may legally appear at the top level of a profile or
    # inside an `uplink:` block. Mirrors NetworkProfile dataclass + the
    # kwargs accepted by NetworkEmulator.apply_settings(). `description`
    # and `uplink` are intentionally excluded.
    _PROFILE_IMPAIRMENT_FIELDS = frozenset({
        "delay_ms", "jitter_ms",
        "delay_distribution", "delay_correlation_pct",
        "loss_pct", "loss_correlation_pct", "loss_model",
        "rate_mbit", "rate_ceil_mbit",
        "rate_burst_kbit", "rate_cburst_kbit",
        "corruption_pct", "corruption_correlation_pct",
        "reorder_pct", "reorder_correlation_pct",
        "duplicate_pct", "duplicate_correlation_pct",
        "limit_packets",
    })

    def _validate_asymmetric_profiles(self) -> None:
        """Reject malformed `uplink:` blocks at startup.

        Catches: non-dict uplink, nested uplink, unknown fields. Empty
        uplink (`uplink: {}`) is allowed and logged — treated as symmetric.
        """
        profiles = (self.profiles_config or {}).get("profiles", {}) or {}
        for name, raw in profiles.items():
            if not isinstance(raw, dict):
                continue
            ul = raw.get("uplink")
            if ul is None:
                continue
            if not isinstance(ul, dict):
                raise ValueError(
                    f"profile '{name}': 'uplink:' must be a mapping, "
                    f"got {type(ul).__name__}"
                )
            if not ul:
                logger.warning(
                    f"profile '{name}': empty uplink: block — treating as symmetric"
                )
                continue
            if "uplink" in ul:
                raise ValueError(
                    f"profile '{name}': nested 'uplink:' is forbidden"
                )
            unknown = set(ul) - self._PROFILE_IMPAIRMENT_FIELDS
            if unknown:
                raise ValueError(
                    f"profile '{name}': unknown key(s) under uplink: "
                    f"{sorted(unknown)}"
                )

    def _apply_emulator_profile(
        self, profile_name: str, ingress_profile: Optional[str]
    ) -> bool:
        """Apply *profile_name*, honoring the optional `uplink:` override.

        Symmetric profiles fall through to ``emulator.apply_profile`` (existing
        behavior). Asymmetric profiles are applied via ``apply_settings``,
        which already supports an ``ingress_settings`` dict — no synthetic
        netemu profiles are registered.

        CLI ``--ingress-profile Y`` wins over a built-in ``uplink:`` block:
        egress still uses (base ⊕ uplink), ingress uses Y's base fields.

        ``no_emulation`` is a true bypass: any prior tc state is cleared and
        no qdisc is added, so the device is observed exactly as the kernel
        sees it (no netem queueing, no IFB).
        """
        # No-emulation reference baseline: skip tc/netem entirely.
        if profile_name == "no_emulation":
            logger.info(
                "Profile no_emulation: clearing prior tc state and bypassing netem"
            )
            self.emulator.clear()
            return True

        profiles = (self.profiles_config or {}).get("profiles", {}) or {}
        raw = profiles.get(profile_name) if isinstance(profiles, dict) else None
        uplink = raw.get("uplink") if isinstance(raw, dict) else None

        # Symmetric fast path
        if not uplink:
            return self.emulator.apply_profile(
                profile_name, ingress_profile=ingress_profile
            )

        # Asymmetric — split base + uplink overrides into egress/ingress dicts
        base = {
            k: v for k, v in raw.items()
            if k in self._PROFILE_IMPAIRMENT_FIELDS
        }
        egress = {**base, **uplink}

        if ingress_profile == "none":
            ingress_dict = None
            disable_ingress = True
        elif ingress_profile:
            cli_raw = profiles.get(ingress_profile)
            if not isinstance(cli_raw, dict):
                raise ValueError(
                    f"--ingress-profile '{ingress_profile}' not found in profiles.yaml"
                )
            ingress_dict = {
                k: v for k, v in cli_raw.items()
                if k in self._PROFILE_IMPAIRMENT_FIELDS
            }
            disable_ingress = False
        else:
            # Default: ingress uses the base (downlink) fields unchanged
            ingress_dict = base
            disable_ingress = False

        logger.info(
            f"Profile {profile_name} is asymmetric. "
            f"uplink overrides: {sorted(uplink.keys())}"
        )
        if not self.emulator.bidirectional:
            logger.warning(
                f"Profile {profile_name} has an uplink: block but --egress-only "
                f"is set — downlink (base) side will not be applied."
            )
        logger.debug(f"  egress (UL):  {egress}")
        logger.debug(
            f"  ingress (DL): "
            f"{ingress_dict if ingress_dict is not None else '<egress-only>'}"
        )

        return self.emulator.apply_settings(
            **egress,
            profile_name=profile_name,
            ingress_settings=ingress_dict,
            disable_ingress=disable_ingress,
        )

    # Scenario types whose dataset-bench scoring lives in the DB as
    # per-record `metadata.bench` (see scenarios/encoder_placement*.py).
    # Unlike realtime_video_understanding (which re-reads a
    # questions_results_*.json file), these scenarios never write a
    # results json — the report chain must aggregate from log_records.
    _ENCPL_SCENARIO_TYPES = frozenset({
        "encoder_placement",
        "encoder_placement_concurrent",
        "encoder_placement_qwen36",
        "encoder_placement_concurrent_qwen36",
    })

    def _collect_bench_entries(self, results: list) -> list[tuple[str, dict]]:
        """Collect (session_id, bench_dict) pairs from ScenarioResults.

        Parsed from each LogRecord's metadata JSON; records without a
        bench object (single-video mode, failed turns — and padded
        filler on pre-playlist wave-chunked rows) are skipped.
        """
        entries = []
        for result in results:
            for record in getattr(result, "log_records", []):
                meta_raw = getattr(record, "metadata", None)
                if not meta_raw:
                    continue
                try:
                    meta = json.loads(meta_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                bench = meta.get("bench")
                if isinstance(bench, dict):
                    entries.append((record.session_id, bench))
        return entries

    def _update_bench_metrics(
        self, metrics, results: list, scenario_name: str, profile_name: str
    ) -> None:
        """Aggregate dataset-bench accuracy into ScenarioMetrics (real_*).

        Run-mean semantics: accuracy = mean of per-run accuracies
        (one run = one session_id); pooled counts populate
        real_total/real_correct. Logs the same summary fields the
        realtime_video path logs. Also fills the bench_* quality flags
        (bench_available/scored/parse_ok/parse_fail/runs/empty_runs) so
        downstream report readers can tell "no bench data" apart from
        a genuine 0.0 accuracy. Emits a WARNING (not silence) when
        there is nothing to aggregate.
        """
        entries = self._collect_bench_entries(results)
        if not entries:
            n_records = sum(
                len(getattr(result, "log_records", [])) for result in results)
            if n_records == 0:
                logger.warning(
                    f"bench accuracy ({scenario_name}/{profile_name}): "
                    f"no log records in results (all runs skipped via "
                    f"--resume, or scenario returned nothing) — "
                    f"real_accuracy stays 0.0 with no bench data; "
                    f"historical rows can be reported with "
                    f"tools/encpl_bench_db_report.py"
                )
            else:
                logger.warning(
                    f"bench accuracy ({scenario_name}/{profile_name}): "
                    f"{n_records} log record(s) but none carries "
                    f"metadata.bench (single-video mode) — "
                    f"real_accuracy stays 0.0 with no bench data"
                )
            return
        agg = MetricsCalculator.aggregate_bench_accuracy(entries)
        if agg is None:
            logger.warning(
                f"bench accuracy ({scenario_name}/{profile_name}): "
                f"{len(entries)} bench entries but 0 scored MCQ "
                f"(generation tasks / non-A-D answers are recorded "
                f"unscored) — real_accuracy stays 0.0"
            )
            return
        MetricsCalculator.update_task_metrics(
            metrics, {"total": {
                "total": agg["total"],
                "correct": agg["correct"],
                "accuracy": agg["accuracy"],
            }}
        )
        metrics.bench_available = True
        metrics.bench_scored = agg["total"]
        metrics.bench_parse_ok = agg.get("parse_ok", 0)
        metrics.bench_parse_fail = agg.get("parse_fail", 0)
        metrics.bench_runs = agg["runs"]
        metrics.bench_empty_runs = agg.get("empty_runs", 0)
        logger.info(
            f"bench accuracy ({scenario_name}/{profile_name}): "
            f"real_total = {metrics.real_total}, "
            f"real_correct = {metrics.real_correct}, "
            f"real_accuracy = {metrics.real_accuracy:.3f} "
            f"(run-mean over {agg['runs']} run(s)"
            + (f", {agg['empty_runs']} empty run(s) excluded" if
               agg["empty_runs"] else "")
            + f", parse_fail = {agg['parse_fail']}"
        )

    def get_client(self, provider: str):
        """Get or create an LLM client for the given provider."""
        if provider not in self._clients:
            if provider == "openai":
                self._clients[provider] = OpenAIClient()
            elif provider == "gemini":
                self._clients[provider] = GeminiClient()
            elif provider == "deepseek":
                self._clients[provider] = DeepSeekClient()
            elif provider == "vllm":
                self._clients[provider] = VLLMClient()
            elif provider == "azure_openai":
                self._clients[provider] = AzureOpenAIClient()
            elif provider == "azure_inference":
                self._clients[provider] = AzureInferenceClient()
            elif provider == "vlm-local":
                self._clients[provider] = RealtimeWebRTCVLMClient()
            elif provider == "openai_token":
                self._clients[provider] = OpenAITokenClient()
            elif provider == "sglang-engine":
                self._clients[provider] = EncoderPlacementClient()
            elif provider == "sglang-engine-qwen36":
                self._clients[provider] = EncoderPlacementQwen36Client()
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._clients[provider]

    def get_completed_runs(self, scenario_name: str, profile_name: str) -> int:
        """Query the DB for the number of completed runs (successful or timed out).

        A "completed run" is a distinct session_id for the given
        scenario+profile that either succeeded on every record OR is a
        timeout placeholder.  Timed-out runs are legitimate data points
        (the network conditions *are* the cause) and must not be retried
        on resume.
        """
        import sqlite3
        try:
            db_path = self.logger.db_path
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT session_id) FROM traffic_logs
                    WHERE scenario_id = ?
                      AND network_profile = ?
                      AND session_id NOT LIKE 'pcap_%'
                      AND (
                          session_id LIKE 'timeout_%'
                          OR session_id NOT IN (
                              SELECT DISTINCT session_id FROM traffic_logs
                              WHERE scenario_id = ?
                                AND network_profile = ?
                                AND success = 0
                          )
                      )
                """, (scenario_name, profile_name, scenario_name, profile_name))
                count = cursor.fetchone()[0]
                return count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Encoder-placement tiered (optional retention) support
    # ------------------------------------------------------------------
    # Two-tier retention = must tier on TCP + optional tier on best-effort
    # UDP datagrams (E2U second wave). The scenario classes read the four
    # knobs (optional_ratio / opt_tokens_per_packet / opt_send_gap_ms /
    # opt_chat_delay_ms) from their config dict (env overrides win inside
    # the scenario); the orchestrator's job is ONLY the wiring around it:
    # netem must shape the UDP pair too (a port-filtered lo shape that
    # misses it lets the optional tier bypass impairment entirely on
    # single-host runs), a tiered run against a sidecar without its UDP
    # receiver must fail ONCE before any run starts, and the summary
    # should carry the sidecar-level opt_udp delta (fragments/parts
    # counters). Mirrors run_qwen36_matrix.py's apply_netem/preflight
    # semantics so orchestrator and standalone cells stay comparable.

    @staticmethod
    def _is_local_url(url: str) -> bool:
        """True when a sidecar base URL targets this host (loopback).

        Same host set as tools/run_qwen36_matrix.py is_local(): the
        port-filtered lo shape is the correct netem mode there; a remote
        sidecar rides a real interface where the normal whole-interface
        profile (optionally --interface lo aside) keeps applying.
        """
        host = (urlparse(url or "").hostname or "").lower()
        return host in ("localhost", "127.0.0.1", "::1", "")

    def _encpl_sidecar_url(self, scenario_config: dict) -> "str | None":
        """Base URL of the ENCPL sidecar a scenario will talk to.

        Order: env override (SGLANG_ENGINE_QWEN36_BASE_URL — the
        documented way to point the scenario at the 8014 concurrent
        sidecar) → the shared client's DEFAULT_BASE_URL (8004 serial).
        Only reads: never constructs the client (side effect-free, and
        env inspection must happen BEFORE get_client memoizes, so call
        sites run pre-run, not post).
        """
        env_url = os.environ.get("SGLANG_ENGINE_QWEN36_BASE_URL")
        if env_url:
            return env_url.rstrip("/")
        return EncoderPlacementQwen36Client.DEFAULT_BASE_URL

    def _encpl_tiered_info(self, scenario_config: dict) -> dict:
        """Resolve the tiered knobs for ONE experiment (read-only).

        Respects the same env precedence the serial scenario applies
        (ENCPL_OPTIONAL_RATIO etc. win over yaml): the orchestrator's
        netem filter and preflight must match what the scenario will
        ACTUALLY run, not what the yaml alone says. The concurrent
        scenario resolves config-only today; env reads here would make
        the orchestrator expect a tiered run the scenario then ignores —
        so for concurrent types the yaml value alone is authoritative.
        """
        info = {
            "tiered": False, "ratio": 0.0, "udp_port": 0,
            "mode": str(scenario_config.get("mode", "server")).strip().lower(),
            "concurrent": scenario_config.get("type")
            == "encoder_placement_concurrent_qwen36",
        }
        try:
            if info["concurrent"]:
                ratio = float(
                    scenario_config.get("optional_ratio", 0.0) or 0.0)
            else:
                ratio = float(
                    os.environ.get("ENCPL_OPTIONAL_RATIO")
                    or scenario_config.get("optional_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            return info  # the scenario raises its own precise error later
        info["ratio"] = ratio
        # Tiered is edge+vidcom2 only; a ratio set without compression is
        # warned-and-ignored by the scenario — mirror that (no UDP filter,
        # no preflight) instead of failing the experiment here.
        compressor = str(
            scenario_config.get("compressor") or "").strip().lower()
        env_compressor = os.environ.get("COMPRESSOR")
        if env_compressor:
            compressor = env_compressor.strip().lower()
        info["tiered"] = bool(
            ratio > 0.0 and info["mode"] == "edge"
            and compressor == "vidcom2")
        if info["tiered"]:
            try:
                info["udp_port"] = int(
                    scenario_config.get("opt_udp_port", 8005) or 0)
            except (TypeError, ValueError):
                info["udp_port"] = 8005
        return info

    def _encpl_sidecar_stats(self, base_url: str) -> dict:
        """GET {base_url}/stats — {} on any failure (sidecar down, old
        build without the endpoint). Callers treat 'unreadable' and
        'absent' as the same preflight failure with distinct messages."""
        import requests
        try:
            resp = requests.get(base_url.rstrip("/") + "/stats",
                                timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, dict) else {}
        except KeyboardInterrupt:
            raise
        except (OSError, ValueError) as exc:
            logger.debug(f"sidecar /stats unreadable at {base_url}: "
                         f"{type(exc).__name__}: {exc}")
        return {}

    def _encpl_opt_udp_delta(self, before: dict, after: dict) -> "dict | None":
        """Sidecar opt_udp counter delta across an experiment.

        Same keys as tools/run_qwen36_matrix.py _opt_udp_delta; None when
        either snapshot lacks an opt_udp block (sidecar without the
        receiver, or /stats unreadable mid-run) — absent counters must
        not be reported as zeros.
        """
        b = (before or {}).get("opt_udp") or {}
        a = (after or {}).get("opt_udp") or {}
        if not a:
            return None
        keys = ("fragments_rx", "fragments_bad", "parts_reassembled",
                "parts_incomplete_dropped", "parts_overflow_dropped",
                "parts_stale_dropped")
        return {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in keys}

    def _encpl_preflight_opt_udp(self, tiered: dict,
                                 base_url: str) -> int:
        """Tiered-run preflight: verify the sidecar's optional-tier UDP
        receiver is live BEFORE the first run (a tiered edge run against
        a sidecar without --opt-udp-port would otherwise fail on EVERY
        question at the must PUT response — client raises on
        opt_udp_port=0). Hard error: the run's measurements would be
        structurally impossible, not merely degraded. Returns the bound
        UDP port (yaml value is only a hint; /stats is the truth).
        """
        stats = self._encpl_sidecar_stats(base_url)
        if not stats:
            raise RunFailedError(
                f"tiered encoder_placement_qwen36 run but the sidecar "
                f"/stats is unreadable at {base_url}; cannot verify the "
                f"optional-tier UDP receiver — restart the sidecar with "
                f"--opt-udp-port > 0 (e.g. 8005)")
        opt = stats.get("opt_udp") or {}
        port = int(opt.get("port") or 0)
        if port <= 0:
            raise RunFailedError(
                f"tiered encoder_placement_qwen36 run (optional_ratio="
                f"{tiered['ratio']:g}) but the sidecar reports no "
                f"optional-tier UDP receiver (opt_udp.port={port}); "
                f"restart the sidecar with --opt-udp-port > 0 "
                f"(e.g. 8005)")
        if not opt.get("alive"):
            raise RunFailedError(
                f"tiered encoder_placement_qwen36 run but the sidecar's "
                f"optional-tier UDP receiver thread is dead "
                f"(opt_udp.port={port}); restart the sidecar with "
                f"--opt-udp-port > 0")
        logger.info(
            f"  Sidecar opt_udp receiver: port={port} "
            f"rcvbuf={opt.get('rcvbuf')}")
        return port

    def _encpl_apply_loopback_netem(self, tiered: dict, tcp_port: int,
                                    profile_name: str) -> bool:
        """Port-filtered loopback shape for local ENCPL sidecars.

        Replaces the whole-interface profile for local-sidecar ENCPL
        experiments: only the sidecar TCP pair (plus the tiered UDP pair
        when tiered) is impaired, so MCP/SSH/other loopback traffic is
        untouched — the same semantics tools/run_qwen36_matrix.py
        apply_netem() uses, keeping orchestrator cells comparable with
        the standalone/matrix/calibration cells. no_emulation clears lo
        (a stale filter from a killed prior run would otherwise silently
        shape the 'reference' profile). Asymmetric uplink: blocks are
        rejected: the port filters cannot express per-direction
        parameters (matrix driver semantics).
        """
        profiles = (self.profiles_config or {}).get("profiles", {}) or {}
        raw = profiles.get(profile_name)
        if isinstance(raw, dict) and raw.get("uplink"):
            raise RunFailedError(
                f"profile '{profile_name}' has an asymmetric 'uplink:' "
                f"block — not supported for local-sidecar "
                f"encoder_placement runs (port-filtered loopback cannot "
                f"express per-direction parameters)")
        udp_port = tiered["udp_port"] if tiered["tiered"] else None
        if profile_name == "no_emulation":
            # Reference profile: leave lo completely unshaped.
            self.emulator.clear_loopback()
            return True
        ok = self.emulator.apply_profile_to_loopback(
            profile_name, tcp_port, udp_port=udp_port)
        if ok:
            logger.info(
                f"  Loopback netem {profile_name}: tcp port {tcp_port}"
                + (f" + udp port {udp_port}" if udp_port else ""))
        return ok

    def run_experiment(
        self,
        scenario_name: str,
        profile_name: str,
        runs: int = 10,
        inter_run_delay: float = 1.0,
        ingress_profile: Optional[str] = None,
        run_timeout: Optional[float] = None,
        stop_on_error: bool = False,
        resume: bool = False,
    ) -> list[ScenarioResult]:
        """
        Run a single experiment (scenario + profile combination).

        Args:
            scenario_name: Name of the scenario to run
            profile_name: Network profile to apply (egress, and ingress if not specified separately)
            runs: Number of runs
            inter_run_delay: Delay between runs in seconds
            ingress_profile: Optional separate profile for ingress traffic.
                            None = use same as egress, "none" = no ingress shaping
            run_timeout: Per-run timeout in seconds. None means no timeout.
            stop_on_error: If True, raise RunFailedError on the first failed run
                          (after retries are exhausted).
            resume: If True, query the DB for already-completed successful runs
                   and skip them.

        Returns:
            List of ScenarioResult objects

        Raises:
            RunFailedError: If stop_on_error is True and a run fails.
        """
        # Get scenario configuration
        scenario_config = self.scenarios_config["scenarios"].get(scenario_name)
        if not scenario_config:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        if scenario_config.get("disabled", False):
            logger.info(f"Skipping disabled scenario: {scenario_name}")
            return []

        # Determine scenario type and class
        scenario_type = scenario_config.get("type", "chat")
        scenario_class = self.scenario_classes.get(scenario_type)
        if not scenario_class:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        defaults = self.scenarios_config.get("defaults", {})
        retry_count = int(defaults.get("retry_count", 0))
        retry_backoff_sec = float(defaults.get("retry_backoff_sec", 1.0))
        retry_backoff_multiplier = float(defaults.get("retry_backoff_multiplier", 2.0))

        # Resume: check before any expensive setup (client init, network
        # profile application) so fully-completed combos are skipped cheaply.
        start_run = 0
        if resume:
            completed = self.get_completed_runs(scenario_name, profile_name)
            if completed >= runs:
                logger.info(
                    f"Skipping {scenario_name}/{profile_name}: "
                    f"all {runs} runs already completed ({completed} in DB)"
                )
                return []
            if completed > 0:
                start_run = completed
                logger.info(
                    f"Resuming {scenario_name}/{profile_name} from run "
                    f"{start_run + 1}/{runs} ({completed} already completed)"
                )

        # Get provider and client
        provider = scenario_config.get("provider", "openai")
        client = self.get_client(provider)

        # Create scenario instance
        scenario_config["scenario_id"] = scenario_name
        # Inject MCP transport setting for agent scenarios
        if self.mcp_transport != "stdio" and "mcp_transport" not in scenario_config:
            scenario_config["mcp_transport"] = self.mcp_transport
        # Inject the raw network interface (from --interface) so WebRTC-based
        # scenarios can restrict ICE candidates to it; the raw value is used
        # so that "auto" does NOT activate the filter (stock ICE behaviour)
        if "network_interface" not in scenario_config:
            scenario_config["network_interface"] = self.raw_network_interface
        scenario = scenario_class(client, self.logger, scenario_config)

        # Pass emulator to agent scenarios so they can apply netem to loopback
        if hasattr(scenario, "emulator"):
            scenario.emulator = self.emulator
            scenario._current_network_profile = profile_name

        # Encoder-placement tiered (optional retention) wiring:
        # local-sidecar runs get the port-filtered loopback shape
        # (TCP pair + tiered UDP pair) instead of the whole-interface
        # profile; tiered runs preflight the sidecar's UDP receiver once,
        # before any run starts. Sidecar /stats snapshot taken here for
        # the opt_udp delta in the summary (local only: on a remote
        # sidecar the counters would span OTHER hosts' traffic).
        encpl_tiered = {}
        encpl_stats_before = {}
        encpl_lo = False
        if scenario_config.get("type") in self._ENCPL_SCENARIO_TYPES:
            encpl_tiered = self._encpl_tiered_info(scenario_config)
            sidecar_url = self._encpl_sidecar_url(scenario_config)
            if self._is_local_url(sidecar_url):
                encpl_lo = True
                tcp_port = int(
                    (urlparse(sidecar_url).port or 0) or 8004)
                if encpl_tiered["tiered"]:
                    encpl_tiered["udp_port"] = \
                        self._encpl_preflight_opt_udp(
                            encpl_tiered, sidecar_url)
                    # stats snapshot only for tiered runs — the delta
                    # summary is a tiered feature; a legacy run has
                    # nothing to count and must not pay the HTTP call
                    # (a down sidecar behind a legacy single-tier run
                    # would log noise otherwise).
                    encpl_stats_before = self._encpl_sidecar_stats(
                        sidecar_url)
                logger.info(
                    f"Applying network profile: {profile_name} "
                    f"(port-filtered loopback, sidecar {sidecar_url})")
                if not self._encpl_apply_loopback_netem(
                        encpl_tiered, tcp_port, profile_name):
                    logger.warning(
                        "Failed to apply loopback netem (may require sudo)")
            elif encpl_tiered["tiered"]:
                logger.info(
                    f"  Tiered run against remote sidecar {sidecar_url}: "
                    f"whole-interface netem applies (E2U datagrams ride "
                    f"the real interface)")

        # Apply network profile (skipped for local-sidecar ENCPL runs —
        # the port-filtered lo shape above IS their netem)
        if not encpl_lo:
            if ingress_profile:
                logger.info(f"Applying network profile: egress={profile_name}, ingress={ingress_profile}")
                if not self._apply_emulator_profile(profile_name, ingress_profile):
                    logger.warning(f"Failed to apply network profile (may require sudo)")
            else:
                logger.info(f"Applying network profile: {profile_name} (bidirectional)")
                if not self._apply_emulator_profile(profile_name, ingress_profile):
                    logger.warning(f"Failed to apply network profile (may require sudo)")

        results = []

        try:
            for run_index in range(start_run, runs):
                logger.info(f"Running {scenario_name} [{run_index + 1}/{runs}] with profile {profile_name}")

                attempt = 0
                retry_reason = None
                while True:
                    t_start = time.time()

                    if run_timeout is not None:
                        result = self._run_with_timeout(
                            scenario, profile_name, run_index, run_timeout,
                            scenario_name,
                        )
                    else:
                        result = scenario.run(network_profile=profile_name, run_index=run_index)

                    t_elapsed = time.time() - t_start

                    current_reason = self._get_retry_reason(result)
                    if current_reason:
                        retry_reason = current_reason

                    if result.success or attempt >= retry_count or not current_reason:
                        break

                    backoff = retry_backoff_sec * (retry_backoff_multiplier ** attempt)
                    logger.warning(
                        f"  Retry {attempt + 1}/{retry_count} after {backoff:.1f}s "
                        f"({current_reason})"
                    )
                    time.sleep(backoff)
                    attempt += 1

                results.append(result)

                # Persist timeout results to the DB so that --resume
                # counts them as completed and does not retry them.
                if not result.log_records and result.session_id.startswith("timeout_"):
                    timeout_record = LogRecord(
                        timestamp=time.time(),
                        scenario_id=scenario_name,
                        session_id=result.session_id,
                        turn_index=0,
                        run_index=run_index,
                        provider=scenario_config.get("provider", ""),
                        model=scenario_config.get("model", ""),
                        network_profile=profile_name,
                        latency_sec=result.total_latency_sec,
                        success=False,
                        error_type="timeout",
                    )
                    self.logger.log(timeout_record)

                logger.info(
                    f"  Completed: success={result.success}, "
                    f"latency={result.total_latency_sec:.2f}s, "
                    f"api_calls={result.api_call_count}, "
                    f"tools={result.tool_calls_count}"
                )
                if not result.success and result.error_message:
                    logger.error(f"  Error: {result.error_message}")
                if attempt > 0:
                    result.metadata["retry_attempts"] = attempt
                    if retry_reason:
                        result.metadata["retry_reason"] = retry_reason

                if run_index < runs - 1:
                    time.sleep(inter_run_delay)

        finally:
            # Clear network profile. Local-sidecar ENCPL runs shaped lo
            # via port filters — emulator.clear() would target the
            # configured interface and miss them (or, with --interface
            # lo, fight with the IFB teardown), so clear_loopback() is
            # the correct reset for that branch.
            if encpl_lo:
                self.emulator.clear_loopback()
            else:
                self.emulator.clear()

        # Encoder-placement tiered summary: sidecar opt_udp counters
        # accumulated over this experiment (fragments received, parts
        # reassembled/dropped — the loss the optional tier ACTUALLY
        # suffered). Logged, not persisted: per-record engine_stats
        # remain the durable trace (same division of labor as the
        # matrix driver).
        if encpl_lo and encpl_tiered.get("tiered"):
            sidecar_url = self._encpl_sidecar_url(scenario_config)
            delta = self._encpl_opt_udp_delta(
                encpl_stats_before, self._encpl_sidecar_stats(sidecar_url))
            if delta is not None:
                logger.info(
                    f"  opt_udp delta over {len(results)} runs: "
                    f"fragments_rx={delta['fragments_rx']} "
                    f"fragments_bad={delta['fragments_bad']} "
                    f"parts_reassembled={delta['parts_reassembled']} "
                    f"parts_incomplete_dropped="
                    f"{delta['parts_incomplete_dropped']} "
                    f"parts_overflow_dropped="
                    f"{delta['parts_overflow_dropped']} "
                    f"parts_stale_dropped={delta['parts_stale_dropped']}")

        # stop-on-error: only raise if every run failed on a baseline profile
        # (no_emulation), indicating a real infrastructure problem. Failures on
        # degraded profiles (satellite, congested, cell_edge, etc.) are expected
        # data points — the network conditions *are* the cause.
        baseline_profiles = {"no_emulation", "ideal_6g"}
        if (
            stop_on_error
            and results
            and not any(r.success for r in results)
            and profile_name in baseline_profiles
        ):
            raise RunFailedError(
                f"{scenario_name}/{profile_name}: all {len(results)} runs failed. "
                f"Last error: {results[-1].error_message}"
            )

        return results

    def _run_with_timeout(
        self,
        scenario,
        profile_name: str,
        run_index: int,
        timeout_sec: float,
        scenario_name: str,
    ) -> ScenarioResult:
        """Run a single scenario with a per-run timeout.

        Uses a *daemon* thread + queue so that (a) the main thread can
        enforce the deadline and (b) if the run blocks in uninterruptible
        I/O, the zombie worker cannot keep the Python interpreter alive
        after the main process decides to exit. ThreadPoolExecutor is
        intentionally not used here because its workers are non-daemon —
        on a timeout they would outlive both the run and any subsequent
        RunFailedError, polluting later metrics and preventing clean
        shutdown.
        """
        import queue as _queue
        import threading

        result_q: "_queue.Queue[tuple[str, object]]" = _queue.Queue(maxsize=1)

        def _run_in_thread():
            import asyncio
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            try:
                r = scenario.run(
                    network_profile=profile_name, run_index=run_index
                )
                result_q.put(("ok", r))
            except BaseException as exc:  # noqa: BLE001 — forward everything
                result_q.put(("err", exc))

        worker = threading.Thread(
            target=_run_in_thread,
            name=f"scenario-{scenario_name}-run{run_index}",
            daemon=True,
        )
        worker.start()

        try:
            kind, payload = result_q.get(timeout=timeout_sec)
        except _queue.Empty:
            logger.warning(
                f"  Run {run_index + 1} of {scenario_name} timed out "
                f"after {timeout_sec:.0f}s (worker thread abandoned as daemon)"
            )
            # Don't join the worker: if the scenario is blocked in a
            # no-timeout HTTP call on a high-latency profile, join would
            # hang the suite. daemon=True ensures it dies with the process.
            return ScenarioResult(
                scenario_id=scenario_name,
                session_id=f"timeout_{int(time.time())}",
                network_profile=profile_name,
                run_index=run_index,
                success=False,
                total_latency_sec=timeout_sec,
                error_message=f"Run timed out after {timeout_sec:.0f}s",
            )

        if kind == "err":
            # Re-raise so the normal run_experiment error path handles it
            raise payload  # type: ignore[misc]
        return payload  # type: ignore[return-value]

    def _get_retry_reason(self, result: ScenarioResult) -> Optional[str]:
        """Return a retry reason for transient failures.

        Note: run-level timeouts (enforced by _run_with_timeout) are NOT
        retried. On slow network profiles (satellite, congested) a timeout
        almost always means the scenario/profile combination legitimately
        exceeds the budget — retrying just stacks zombie threads from the
        previous run and guarantees more timeouts. Per-request HTTP timeouts
        surfaced as LogRecord error_type are also treated as terminal here.
        """
        if not result.log_records:
            error_text = (result.error_message or "").lower()
            if "rate limit" in error_text or "429" in error_text:
                return "rate_limited"
            return None

        for record in result.log_records:
            if record.success:
                continue
            if self._is_tool_record(record):
                continue

            if record.http_status == 429:
                return "rate_limited"
            if record.http_status and 500 <= record.http_status < 600 and record.http_status != 501:
                return "server_error"

        return None

    def _is_tool_record(self, record: LogRecord) -> bool:
        """Detect MCP tool call records via metadata."""
        if not record.metadata:
            return False
        try:
            metadata = json.loads(record.metadata)
        except json.JSONDecodeError:
            return False
        return metadata.get("type") == "mcp_tool_call"

    def run_test_matrix(
        self,
        matrix: Optional[list[dict]] = None,
        runs_per_experiment: int = 10,
        run_timeout: Optional[float] = None,
        stop_on_error: bool = False,
        resume: bool = False,
    ) -> dict:
        """
        Run a full test matrix.

        Args:
            matrix: Test matrix definition (uses config if not provided)
            runs_per_experiment: Default runs per experiment
            run_timeout: Per-run timeout in seconds (None = no timeout)
            stop_on_error: If True, abort the entire matrix on the first failed run.
            resume: If True, skip scenario/profile combos already completed in the DB.

        Returns:
            Dictionary with all results and computed metrics
        """
        if matrix is None:
            matrix = self.scenarios_config.get("test_matrix", [])

        all_results = {}
        all_metrics = []

        start_time = datetime.now()
        logger.info(f"Starting test matrix with {len(matrix)} experiments")

        for entry in matrix:
            scenario_name = entry["scenario"]
            scenario_def = self.scenarios_config["scenarios"].get(scenario_name, {})
            if scenario_def.get("disabled", False):
                logger.info(f"Skipping disabled scenario: {scenario_name}")
                continue
            profiles = entry.get("profiles", ["ideal_6g"])
            runs = entry.get("runs", runs_per_experiment)

            for profile_name in profiles:
                experiment_key = f"{scenario_name}_{profile_name}"
                logger.info(f"\n{'='*60}")
                logger.info(f"Experiment: {experiment_key}")
                logger.info(f"{'='*60}")

                try:
                    results = self.run_experiment(
                        scenario_name=scenario_name,
                        profile_name=profile_name,
                        runs=runs,
                        run_timeout=run_timeout,
                        stop_on_error=stop_on_error,
                        resume=resume,
                    )

                    all_results[experiment_key] = results

                    # Compute metrics
                    records = [
                        asdict(record)
                        for result in results
                        for record in result.log_records
                    ]
                    if not records:
                        records = [
                            {
                                "latency_sec": r.total_latency_sec,
                                "request_bytes": r.total_request_bytes,
                                "response_bytes": r.total_response_bytes,
                                "tokens_in": r.total_tokens_in,
                                "tokens_out": r.total_tokens_out,
                                "success": r.success,
                                "tool_calls_count": r.tool_calls_count,
                                "tool_latency_sec": (
                                    r.tool_total_latency_sec / r.tool_calls_count
                                    if r.tool_calls_count else 0.0
                                ),
                                "session_id": r.session_id,
                            }
                            for r in results
                        ]

                    metrics_defaults = self.scenarios_config.get("defaults", {})
                    metrics = MetricsCalculator.calculate(
                        records,
                        scenario_name,
                        profile_name,
                        stall_gap_sec=metrics_defaults.get("stall_gap_sec"),
                        burst_gap_sec=metrics_defaults.get("burst_gap_sec"),
                    )
                    all_metrics.append(metrics)

                    # check if task accuracy be calculated
                    scenario_config = self.scenarios_config["scenarios"].get(scenario_name, {})
                    if scenario_config.get("type") == "realtime_video_understanding":
                        try:
                            task_config = scenario_config.get("task_config", {})
                            model_name = scenario_config.get("model", "liquid")
                            po_thd = task_config.get("po_thd", 3)
                            results_dir = task_config.get("results_dir", "logs")
                            task_class = task_config.get("task_class", "real")
                            data = MetricsCalculator.load_task_results(results_dir, profile_name)
                            if data:
                                task_stats = MetricsCalculator.calculate_task_accuracy(data, model_name, po_thd, task_class)
                                MetricsCalculator.update_task_metrics(metrics, task_stats)
                                logger.info(
                                    f"task total = {metrics.task_total}, "
                                    f"time_accuracy = {metrics.task_time_accuracy}, "
                                    f"answer_accuracy = {metrics.task_answer_accuracy}, "
                                    f"real_total = {metrics.real_total}, "
                                    f"real_correct = {metrics.real_correct}, "
                                    f"real_accuracy = {metrics.real_accuracy}, "
                                )
                        except Exception as e:
                            logger.warning(f"Failed to calculate task accuracy: {e}")
                    elif scenario_config.get("type") in self._ENCPL_SCENARIO_TYPES:
                        # Encoder-placement dataset mode: bench scoring is
                        # per-record metadata.bench in the DB — aggregate
                        # from this run's log_records (no results json).
                        try:
                            self._update_bench_metrics(
                                metrics, results, scenario_name, profile_name
                            )
                        except Exception as e:
                            logger.warning(f"Failed to calculate bench accuracy: {e}")
                        # rich bench-tool-style summary for this experiment
                        # (kept here so it prints even on empty --resume
                        # results, with the db_report pointer)
                        try:
                            _print_encpl_bench_summary(
                                results, scenario_name, profile_name
                            )
                        except Exception as e:  # noqa: BLE001 — reporting only
                            logger.warning(
                                f"Failed to print encpl bench summary: {e}")
                        # per-run per-user detail (runner parity: conns
                        # 5-tuples, video paths, upload size/time)
                        try:
                            _print_encpl_per_user_detail(results)
                        except Exception as e:  # noqa: BLE001 — reporting only
                            logger.warning(
                                f"Failed to print encpl per-user detail: {e}")
                        # bench-tool-style per-user stats + pooled
                        # per-request percentiles (from log records)
                        try:
                            _print_encpl_bench_per_user_stats(results)
                        except Exception as e:  # noqa: BLE001 — reporting only
                            logger.warning(
                                f"Failed to print encpl bench per-user "
                                f"stats: {e}")

                    logger.info(f"Metrics: latency_mean={metrics.latency_mean:.3f}s, "
                               f"success_rate={metrics.success_rate:.1f}%")

                except RunFailedError:
                    raise
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    all_results[experiment_key] = {"error": str(e)}

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"\n{'='*60}")
        logger.info(f"Test matrix completed in {duration:.1f}s")
        logger.info(f"{'='*60}")

        # ---- cross-experiment bench summary (encpl scenarios) ------
        # One row per experiment from the ScenarioMetrics bench_* flags
        # (filled by _update_bench_metrics); bench_available=False rows
        # still show up so a --resume-skipped or single-video experiment
        # is visible instead of silently missing.
        bench_rows = [m for m in all_metrics
                      if m.scenario_id and self.scenarios_config.get(
                          "scenarios", {}).get(m.scenario_id, {}
                          ).get("type") in self._ENCPL_SCENARIO_TYPES]
        if bench_rows:
            print(f"\n[EncPlBench] matrix bench summary "
                  f"({len(bench_rows)} experiment(s)):")
            print(f"  {'scenario':45s} {'profile':18s} "
                  f"{'acc':>6s} {'correct':>8s} {'scored':>7s} "
                  f"{'p_fail':>7s} {'runs':>5s}")
            for m in bench_rows:
                if m.bench_available:
                    acc = f"{m.real_accuracy:.3f}"
                    correct = str(m.real_correct)
                    scored = str(m.bench_scored)
                    p_fail = str(m.bench_parse_fail)
                    runs = str(m.bench_runs)
                else:
                    acc = "n/a"
                    correct = "-"
                    scored = "-"
                    p_fail = "-"
                    runs = "-"
                print(f"  {m.scenario_id[:45]:45s} "
                      f"{m.network_profile[:18]:18s} "
                      f"{acc:>6s} {correct:>8s} {scored:>7s} "
                      f"{p_fail:>7s} {runs:>5s}")
            if any(not m.bench_available for m in bench_rows):
                print("  (n/a = no bench rows in this run's results: "
                      "single-video mode, --resume skipped everything, "
                      "or all failed — see per-experiment warnings; "
                      "history: tools/encpl_bench_db_report.py)")

        return {
            "results": all_results,
            "metrics": all_metrics,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_sec": duration
        }

    def generate_report(
        self,
        metrics: list,
        output_path: str = "results/reports/experiment_report.json"
    ) -> None:
        """Generate a JSON report from metrics."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now().isoformat(),
            "scenarios": {}
        }

        for m in metrics:
            if m.scenario_id not in report["scenarios"]:
                report["scenarios"][m.scenario_id] = {}

            report["scenarios"][m.scenario_id][m.network_profile] = \
                MetricsCalculator.to_3gpp_format(m)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="6G AI Traffic Characterization Testbed"
    )
    parser.add_argument(
        "--scenario", "-s",
        help="Scenario to run (or 'all' for test matrix)"
    )
    parser.add_argument(
        "--profile", "-p",
        default="ideal_6g",
        help="Network profile to use"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=10,
        help="Number of runs per experiment"
    )
    parser.add_argument(
        "--config",
        default="configs/scenarios.yaml",
        help="Path to scenarios config"
    )
    parser.add_argument(
        "--profiles",
        default="configs/profiles.yaml",
        help="Path to network profiles config"
    )
    parser.add_argument(
        "--db",
        default="logs/traffic_logs.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--interface",
        default="auto",
        help="Network interface for emulation (auto = detect from default route)"
    )
    parser.add_argument(
        "--egress-only",
        action="store_true",
        help="Disable ingress shaping (only shape egress/outbound traffic)"
    )
    parser.add_argument(
        "--ingress-profile",
        default=None,
        help="Use a different network profile for ingress traffic (default: same as --profile)"
    )
    parser.add_argument(
        "--report",
        default="results/reports/experiment_report.json",
        help="Output path for report"
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available network profiles"
    )
    parser.add_argument(
        "--capture-pcap",
        action="store_true",
        help="Enable L3/L4 capture using tcpdump"
    )
    parser.add_argument(
        "--capture-filter",
        default=None,
        help="Optional tcpdump filter expression (e.g., 'port 443')"
    )
    parser.add_argument(
        "--capture-dir",
        default="results/captures",
        help="Directory for pcap captures"
    )
    parser.add_argument(
        "--capture-loopback",
        action="store_true",
        default=True,
        help="When MCP transport is http, also capture loopback (lo) so MCP "
             "JSON-RPC frames are visible in pcap. Enabled by default. "
             "Use --no-capture-loopback to disable."
    )
    parser.add_argument(
        "--no-capture-loopback",
        dest="capture_loopback",
        action="store_false",
        help="Disable the secondary loopback capture (mirror flag)."
    )
    parser.add_argument(
        "--capture-loopback-filter",
        default=None,
        help="BPF filter for the loopback capture. Defaults to "
             "'tcp and not port 22 and not port 53'."
    )
    parser.add_argument(
        "--capture-l7",
        action="store_true",
        help="Enable L7 capture using mitmproxy"
    )
    parser.add_argument(
        "--capture-l7-dir",
        default="results/l7_captures",
        help="Directory for L7 captures"
    )
    parser.add_argument(
        "--capture-l7-hosts",
        default=None,
        help="Comma-separated hostnames to filter L7 capture"
    )
    parser.add_argument(
        "--capture-l7-proxy-port",
        type=int,
        default=8080,
        help="Port for mitmproxy"
    )
    parser.add_argument(
        "--capture-l7-web-port",
        type=int,
        default=8081,
        help="Port for mitmproxy web UI (0 to disable)"
    )
    parser.add_argument(
        "--mcp-transport",
        choices=["stdio", "http"],
        default="http",
        help="MCP server transport: stdio (pipe, no netem) or http (TCP, netem-shaped)"
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=None,
        help="Per-run timeout in seconds (default: no timeout). "
             "Individual runs that exceed this are marked as failed and skipped."
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on the first failed run (after retries)."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from where a previous run left off. Skips scenario/profile "
             "combinations that already have enough successful runs in the database."
    )

    args = parser.parse_args()

    orchestrator = TestbedOrchestrator(
        config_path=args.config,
        profiles_path=args.profiles,
        db_path=args.db,
        network_interface=args.interface,
        egress_only=args.egress_only,
        mcp_transport=args.mcp_transport,
    )

    if args.list_scenarios:
        print("\nAvailable scenarios:")
        for name, config in orchestrator.scenarios_config["scenarios"].items():
            disabled = " [DISABLED]" if config.get("disabled", False) else ""
            print(f"  - {name}: {config.get('description', 'No description')}{disabled}")
        return

    if args.list_profiles:
        print("\nAvailable network profiles:")
        for name in orchestrator.emulator.list_profiles():
            profile = orchestrator.emulator.get_profile(name)
            print(f"  - {name}: {profile.description}")
        return

    pcap_controller = None
    pcap_start_time = None
    pcap_file = None
    l7_controller = None
    pcap_lo_controller = None
    pcap_lo_file = None
    # run_id ties primary + loopback pcaps to the same orchestrator invocation,
    # so post-hoc joins do not have to rely on overlapping timestamps.
    run_id = f"run_{int(time.time())}_{abs(hash(repr(vars(args)))) % 100000:05d}"
    try:
        if args.capture_pcap:
            try:
                from capture import CaptureController
                pcap_controller = CaptureController(
                    interface=orchestrator.network_interface,
                    capture_dir=args.capture_dir
                )
                capture_filter = args.capture_filter
                # Tiered (optional-retention) encoder-placement runs put
                # the optional tier on best-effort UDP (E2U); a TCP-only
                # capture filter would silently drop the whole second
                # wave. Auto-append the sidecar's UDP port when the
                # single scenario being captured is tiered (single-
                # scenario mode only: with --scenario all the filter is
                # shared across scenarios and must stay as given).
                if args.scenario and args.scenario != "all":
                    scfg = orchestrator.scenarios_config.get(
                        "scenarios", {}).get(args.scenario) or {}
                    if scfg.get("type") in orchestrator._ENCPL_SCENARIO_TYPES:
                        tiered = orchestrator._encpl_tiered_info(scfg)
                        if tiered["tiered"]:
                            udp_port = tiered["udp_port"]
                            if udp_port and (not capture_filter
                                             or f"udp port {udp_port}"
                                             not in (capture_filter or "")):
                                capture_filter = (
                                    (capture_filter or "tcp")
                                    + f" or udp port {udp_port}")
                                logger.info(
                                    f"Capture filter auto-extended for "
                                    f"tiered run: {capture_filter}")
                pcap_start_time = time.time()
                pcap_file = pcap_controller.start(
                    filter_expr=capture_filter)
                if not pcap_file:
                    logger.warning("Failed to start tcpdump capture")
                    pcap_start_time = None
            except Exception as e:
                logger.warning(f"Failed to start tcpdump capture: {e}")
                pcap_controller = None
                pcap_start_time = None

            # Secondary loopback capture for MCP-over-HTTP traffic. The MCP
            # client (clients/mcp_client.py:MCPHttpConnection) POSTs JSON-RPC
            # to 127.0.0.1:<auto-port>; that traffic only appears on `lo` and
            # is invisible to the primary capture when the primary interface
            # is anything other than `lo` itself.
            primary_iface = orchestrator.network_interface or ""
            if (
                args.capture_pcap
                and args.capture_loopback
                and primary_iface != "lo"
                and orchestrator.mcp_transport == "http"
            ):
                try:
                    from capture import CaptureController
                    pcap_lo_controller = CaptureController(
                        interface="lo",
                        capture_dir=args.capture_dir,
                    )
                    # Filter out SSH/DNS noise; keep everything else on lo
                    # (MCP HTTP servers use ephemeral ports we cannot enumerate
                    # ahead of time, so a port-range filter is not possible).
                    lo_filter = args.capture_loopback_filter or (
                        "tcp and not port 22 and not port 53"
                    )
                    lo_filename = (
                        f"capture_lo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap"
                    )
                    pcap_lo_file = pcap_lo_controller.start(
                        filename=lo_filename, filter_expr=lo_filter
                    )
                    if not pcap_lo_file:
                        logger.warning("Failed to start loopback tcpdump capture")
                        pcap_lo_controller = None
                except Exception as e:
                    logger.warning(f"Failed to start loopback tcpdump capture: {e}")
                    pcap_lo_controller = None

        if args.capture_l7:
            try:
                from capture import L7CaptureController, configure_client_proxy
                l7_controller = L7CaptureController(
                    capture_dir=args.capture_l7_dir,
                    proxy_port=args.capture_l7_proxy_port,
                    web_port=args.capture_l7_web_port
                )
                filter_hosts = None
                if args.capture_l7_hosts:
                    filter_hosts = [
                        h.strip() for h in args.capture_l7_hosts.split(",") if h.strip()
                    ]
                l7_controller.start(filter_hosts=filter_hosts)
                configure_client_proxy(f"http://localhost:{args.capture_l7_proxy_port}")
            except Exception as e:
                logger.warning(f"Failed to start L7 capture: {e}")
                l7_controller = None

        if args.scenario == "all":
            # Run full test matrix
            results = orchestrator.run_test_matrix(
                runs_per_experiment=args.runs,
                run_timeout=args.run_timeout,
                stop_on_error=args.stop_on_error,
                resume=args.resume,
            )
            orchestrator.generate_report(results["metrics"], args.report)
        elif args.scenario:
            # Run single scenario
            results = orchestrator.run_experiment(
                scenario_name=args.scenario,
                profile_name=args.profile,
                runs=args.runs,
                ingress_profile=args.ingress_profile,
                run_timeout=args.run_timeout,
                stop_on_error=args.stop_on_error,
                resume=args.resume,
            )

            # Compute and display metrics
            records = [
                asdict(record)
                for result in results
                for record in result.log_records
            ]
            if not records:
                records = [
                    {
                        "latency_sec": r.total_latency_sec,
                        "request_bytes": r.total_request_bytes,
                        "response_bytes": r.total_response_bytes,
                        "tokens_in": r.total_tokens_in,
                        "tokens_out": r.total_tokens_out,
                        "success": r.success,
                        "tool_calls_count": r.tool_calls_count,
                        "tool_latency_sec": (
                            r.tool_total_latency_sec / r.tool_calls_count
                            if r.tool_calls_count else 0.0
                        ),
                        "session_id": r.session_id,
                    }
                    for r in results
                ]
            metrics_defaults = orchestrator.scenarios_config.get("defaults", {})
            metrics = MetricsCalculator.calculate(
                records,
                args.scenario,
                args.profile,
                stall_gap_sec=metrics_defaults.get("stall_gap_sec"),
                burst_gap_sec=metrics_defaults.get("burst_gap_sec"),
            )

            # check if task accuracy be calculated
            scenario_config = orchestrator.scenarios_config["scenarios"].get(args.scenario)
            scenario_type = scenario_config.get("type", "chat")
            if scenario_type == "realtime_video_understanding":
                try:
                    task_config = scenario_config.get("task_config", {})
                    model_name = scenario_config.get("model", "liquid")
                    po_thd = task_config.get("po_thd", 3)
                    results_dir = task_config.get("results_dir", "logs")
                    task_class = task_config.get("task_class", "real")

                    data = MetricsCalculator.load_task_results(results_dir, args.profile)
                    if data:
                        task_stats = MetricsCalculator.calculate_task_accuracy(data, model_name, po_thd, task_class)
                        MetricsCalculator.update_task_metrics(metrics, task_stats)
                        logger.info(
                            f"task total = {metrics.task_total}, "
                            f"time_accuracy = {metrics.task_time_accuracy}, "
                            f"answer_accuracy = {metrics.task_answer_accuracy}, "
                            f"real_total = {metrics.real_total}, "
                            f"real_correct = {metrics.real_correct}, "
                            f"real_accuracy = {metrics.real_accuracy}, "
                        )
                except Exception as e:
                    logger.warning(f"Failed to calculate task accuracy: {e}")
            elif scenario_type in orchestrator._ENCPL_SCENARIO_TYPES:
                # Encoder-placement dataset mode: bench scoring is
                # per-record metadata.bench in the DB — aggregate from
                # this run's log_records (no results json).
                try:
                    orchestrator._update_bench_metrics(
                        metrics, results, args.scenario, args.profile
                    )
                except Exception as e:
                    logger.warning(f"Failed to calculate bench accuracy: {e}")

            orchestrator.generate_report([metrics], args.report)

            print("\n" + "="*60)
            print(f"Results for {args.scenario} with {args.profile}")
            print("="*60)
            print(f"  Samples: {metrics.sample_count}")
            print(f"  Latency (mean): {metrics.latency_mean:.3f}s")
            print(f"  Latency (p95):  {metrics.latency_p95:.3f}s")
            print(f"  Success rate:   {metrics.success_rate:.1f}%")
            print(f"  Request bytes:  {metrics.request_bytes_mean:.0f}")
            print(f"  Response bytes: {metrics.response_bytes_mean:.0f}")
            print(f"  UL/DL ratio:    {metrics.ul_dl_ratio_mean:.3f}")
            if metrics.tokens_in_mean is not None and metrics.tokens_out_mean is not None:
                print(f"  Tokens in:      {metrics.tokens_in_mean:.0f}")
                print(f"  Tokens out:     {metrics.tokens_out_mean:.0f}")
            if metrics.ttft_p95 is not None:
                print(f"  TTFT (p95):     {metrics.ttft_p95:.3f}s")
            if metrics.ttlt_p95 is not None:
                print(f"  TTLT (p95):     {metrics.ttlt_p95:.3f}s")
            if metrics.stall_rate is not None:
                print(f"  Stall rate:     {metrics.stall_rate:.3f}")
            if metrics.burst_peak_to_mean is not None:
                print(f"  Burst P/M:      {metrics.burst_peak_to_mean:.2f}")
            if metrics.burst_cv is not None:
                print(f"  Burst CV:       {metrics.burst_cv:.2f}")
            if metrics.tool_calls_mean > 0:
                print(f"  Tool calls:     {metrics.tool_calls_mean:.1f}")
                print(f"  Loop factor:    {metrics.loop_factor:.1f}")
            if metrics.error_breakdown and any(metrics.error_breakdown.values()):
                print(f"  Errors:         {metrics.error_breakdown}")
            # Task block: po-class tasks only — a real-class run leaves
            # task_* zeroed (update_task_metrics never touches them), so
            # gating on task_total>0 keeps other scenarios silent instead
            # of printing "Task total: 0" noise (the old `is not None`
            # gates were always-true: the fields are non-Optional ints).
            if metrics.task_total:
                print(f"  Task total:     {metrics.task_total}")
                print(f"  Time accuracy:  {metrics.task_time_accuracy}")
                print(f"  Answer accuracy:{metrics.task_answer_accuracy}")
            # Real block: populated by BOTH the realtime path
            # (update_task_metrics from task_stats) and the encpl bench
            # path (_update_bench_metrics) — they share the real_*
            # fields, so bench_available alone would hide realtime
            # results. real_total>0 covers realtime (scored questions
            # exist); bench_available covers encpl (flags "no bench
            # data" vs genuine 0.0, since real_accuracy is 0.0 in both).
            if metrics.bench_available or metrics.real_total:
                print(f"  Real total:     {metrics.real_total}")
                print(f"  Real correct:   {metrics.real_correct}")
                print(f"  Real accuracy:  {metrics.real_accuracy:.3f}")
            # bench-tool-style rich summary (per-task/per-run/E2E/
            # first-vs-steady/per-user) — prints only when the run has
            # bench rows or wave stats
            if scenario_type in orchestrator._ENCPL_SCENARIO_TYPES:
                try:
                    _print_encpl_bench_summary(
                        results, args.scenario, args.profile
                    )
                except Exception as e:  # noqa: BLE001 — reporting only
                    logger.warning(
                        f"Failed to print encpl bench summary: {e}")
                # per-run per-user detail (runner parity: conns
                # 5-tuples, video paths, upload size/time)
                try:
                    _print_encpl_per_user_detail(results)
                except Exception as e:  # noqa: BLE001 — reporting only
                    logger.warning(
                        f"Failed to print encpl per-user detail: {e}")
                # bench-tool-style per-user stats + pooled
                # per-request percentiles (from log records)
                try:
                    _print_encpl_bench_per_user_stats(results)
                except Exception as e:  # noqa: BLE001 — reporting only
                    logger.warning(
                        f"Failed to print encpl bench per-user stats: {e}")
        else:
            parser.print_help()
    finally:
        if l7_controller:
            try:
                from capture import clear_client_proxy
                l7_controller.stop()
                clear_client_proxy()
            except Exception as e:
                logger.warning(f"Failed to stop L7 capture cleanly: {e}")

        if pcap_controller:
            try:
                pcap_stop_time = time.time()
                pcap_path = pcap_controller.stop()
                if pcap_path:
                    capture_stats = pcap_controller.get_capture_stats(pcap_path)
                    capture_duration = None
                    if pcap_start_time is not None:
                        capture_duration = pcap_stop_time - pcap_start_time

                    scenario_id = args.scenario if args.scenario else "unknown"
                    if args.scenario == "all":
                        scenario_id = "test_matrix"

                    network_profile = args.profile if args.scenario and args.scenario != "all" else "multiple"

                    pcap_size = 0
                    try:
                        pcap_size = pcap_path.stat().st_size
                    except Exception:
                        pass

                    capture_metadata = {
                        "type": "pcap_capture",
                        "pcap_file": str(pcap_path),
                        "capture_dir": args.capture_dir,
                        "capture_filter": args.capture_filter,
                        "capture_duration_sec": capture_duration,
                        "capture_stats": capture_stats,
                        "interface": orchestrator.network_interface,
                        "run_id": run_id,
                    }

                    capture_record = LogRecord(
                        timestamp=time.time(),
                        scenario_id=scenario_id,
                        session_id=f"pcap_{int(pcap_start_time or time.time())}",
                        turn_index=-2,
                        run_index=-1,
                        provider="tcpdump",
                        model="",
                        request_bytes=0,
                        response_bytes=pcap_size,
                        t_request_start=pcap_start_time or 0.0,
                        latency_sec=capture_duration or 0.0,
                        network_profile=network_profile,
                        http_status=200,
                        success=True,
                        metadata=json.dumps(capture_metadata),
                    )
                    orchestrator.logger.log(capture_record)
            except Exception as e:
                logger.warning(f"Failed to stop tcpdump capture cleanly: {e}")

        if pcap_lo_controller:
            try:
                pcap_lo_stop_time = time.time()
                pcap_lo_path = pcap_lo_controller.stop()
                if pcap_lo_path:
                    lo_stats = pcap_lo_controller.get_capture_stats(pcap_lo_path)
                    lo_duration = None
                    if pcap_start_time is not None:
                        lo_duration = pcap_lo_stop_time - pcap_start_time

                    scenario_id_lo = args.scenario if args.scenario else "unknown"
                    if args.scenario == "all":
                        scenario_id_lo = "test_matrix"

                    network_profile_lo = (
                        args.profile if args.scenario and args.scenario != "all" else "multiple"
                    )

                    pcap_lo_size = 0
                    try:
                        pcap_lo_size = pcap_lo_path.stat().st_size
                    except Exception:
                        pass

                    lo_metadata = {
                        "type": "pcap_capture",
                        "pcap_file": str(pcap_lo_path),
                        "capture_dir": args.capture_dir,
                        "capture_filter": args.capture_loopback_filter or "tcp and not port 22 and not port 53",
                        "capture_duration_sec": lo_duration,
                        "capture_stats": lo_stats,
                        "interface": "lo",
                        "run_id": run_id,
                    }

                    lo_record = LogRecord(
                        timestamp=time.time(),
                        scenario_id=scenario_id_lo,
                        session_id=f"pcap_lo_{int(pcap_start_time or time.time())}",
                        turn_index=-2,
                        run_index=-1,
                        provider="tcpdump",
                        model="",
                        request_bytes=0,
                        response_bytes=pcap_lo_size,
                        t_request_start=pcap_start_time or 0.0,
                        latency_sec=lo_duration or 0.0,
                        network_profile=network_profile_lo,
                        http_status=200,
                        success=True,
                        metadata=json.dumps(lo_metadata),
                    )
                    orchestrator.logger.log(lo_record)
            except Exception as e:
                logger.warning(f"Failed to stop loopback tcpdump capture cleanly: {e}")


if __name__ == "__main__":
    main()
