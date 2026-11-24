#!/usr/bin/env python3
"""Offline dataset-bench accuracy report from the traffic_logs DB.

Reads the per-record `metadata.bench` objects written by the
encoder_placement* scenarios (serial + concurrent, qwen3 + qwen36) in
dataset mode and prints the same style of accuracy report as the
standalone bench tools (tools/encpl_bench_eval*.py):

  - OVERALL accuracy (run-mean: each session_id = one run; accuracy is
    the mean of per-run accuracies; pooled counts also shown)
  - accuracy by task_type (MLVU) / duration bucket (Video-MME)
  - parse failures (model answer not extractable as A-D)
  - per-run detail table
  - E2E TTFT / latency summary (from metadata.ttft_sec / latency_sec)

The orchestrator prints the same rich summary at the end of a run (its
module-level _print_encpl_bench_summary) and computes the run-mean
aggregate into experiment_report.json's task_accuracy block (real_* +
bench_* fields) from freshly-executed runs only; this tool works on
any historical DB (e.g. logs written before that integration, or with
--resume where the orchestrator only aggregates freshly-executed runs).

Usage (from aitestbed/ or repo root):
  python tools/encpl_bench_db_report.py \
      --db logs/traffic_logs.db \
      --scenario encoder_placement_qwen36 --profile no_emulation

  # all profiles of a scenario:
  python tools/encpl_bench_db_report.py --db logs/traffic_logs.db \
      --scenario encoder_placement_concurrent_qwen36 --profile all

  # dump the full per-question table as JSON:
  python tools/encpl_bench_db_report.py ... --json out.json

Exit codes: 0 = report printed (even if zero scored entries found for
some combos); 2 = no matching bench rows at all (wrong db/scenario/
profile, or the runs were single-video mode).
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Same bootstrap as encpl_bench_eval.py: aitestbed itself for
# `analysis.*` (this tool needs no aicodec/netemu).
REPO_ROOT = Path(__file__).resolve().parents[2]
AITESTBED_ROOT = REPO_ROOT / "aitestbed"
TOOLS_DIR = Path(__file__).resolve().parent
for _p in (str(TOOLS_DIR), str(AITESTBED_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.metrics import MetricsCalculator  # noqa: E402


def _pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def load_bench_rows(db_path: str, scenario: str, profile: str) -> list[dict]:
    """Return DB rows that carry a metadata.bench object.
    Each row dict: session_id, run_index, timestamp, latency_sec,
    ttft_sec, tokens_out, bench (parsed dict), mode, model. Padded
    entries are KEPT here (flagged bench['padded']); the aggregation
    skips them.
    """
    rows = []
    where = ["scenario_id = ?", "metadata IS NOT NULL"]
    params: list = [scenario]
    if profile != "all":
        where.append("network_profile = ?")
        params.append(profile)
    # Exclude pcap/L7-capture placeholder sessions.
    where.append("session_id NOT LIKE 'pcap_%'")

    sql = (
        "SELECT session_id, run_index, timestamp, network_profile, "
        "       latency_sec, tokens_out, metadata "
        "FROM traffic_logs WHERE " + " AND ".join(where) +
        " ORDER BY timestamp"
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            sys.exit(f"DB error: {e} (db={db_path})")
        for r in cur:
            try:
                meta = json.loads(r["metadata"])
            except (json.JSONDecodeError, TypeError):
                continue
            bench = (meta or {}).get("bench")
            if not isinstance(bench, dict):
                continue
            rows.append({
                "session_id": r["session_id"],
                "run_index": r["run_index"],
                "timestamp": r["timestamp"],
                "network_profile": r["network_profile"],
                "latency_sec": r["latency_sec"],
                # tokens_out distinguishes a zero-token completion
                # (EMPTY: tokens_out == 0, latency 0.0, excluded from
                # E2E stats) from a real decode
                "tokens_out": r["tokens_out"],
                "ttft_sec": (meta or {}).get("ttft_sec"),
                "mode": (meta or {}).get("mode"),
                "model": (meta or {}).get("model"),
                "bench": bench,
            })
    return rows


def report_combo(
    db_path: str, scenario: str, profile: str, out: list[dict]
) -> bool:
    """Print the report for one (scenario, profile) combo.

    Appends per-question detail dicts to `out` (for --json). Returns
    True if any scored entry existed.
    """
    rows = load_bench_rows(db_path, scenario, profile)
    if not rows:
        print(f"\n[EncPlBenchDB] {scenario}/{profile}: no bench rows in DB "
              f"(single-video mode, or wrong scenario/profile/db)")
        return False

    dataset = rows[0]["bench"].get("dataset") or "?"
    scored = [r for r in rows if not r["bench"].get("padded")]
    padded = len(rows) - len(scored)

    entries = [(r["session_id"], r["bench"]) for r in scored]
    agg = MetricsCalculator.aggregate_bench_accuracy(entries)
    if agg is None:
        print(f"\n[EncPlBenchDB] {scenario}/{profile}: {len(rows)} bench row(s) "
              f"but 0 scored MCQ entries "
              f"(generation tasks / non-A-D answers are recorded unscored)")
        return False

    print(f"\n[EncPlBenchDB] ===== {scenario} / {profile} "
          f"({dataset}, {len(scored)} scored"
          + (f", {padded} padded skipped" if padded else "") + ") =====")

    # mode / model split (server vs edge arms may share a scenario name)
    mode_counts = defaultdict(int)
    for r in scored:
        mode_counts[r["mode"] or "?"] += 1
    for m, n in sorted(mode_counts.items()):
        print(f"  mode {m}: n={n}")

    # ---- overall (run-mean + pooled) ----
    print(f"\n  OVERALL (run-mean over {agg['runs']} run(s)): "
          f"{agg['accuracy']:.3f}   "
          f"[pooled: {agg['correct']}/{agg['total']}]"
          + (f"   parse_fail={agg['parse_fail']}" if agg["parse_fail"] else ""))
    if agg["empty_runs"]:
        print(f"  ({agg['empty_runs']} empty run(s) excluded from the mean)")

    # ---- per-task / per-duration accuracy ----
    by_bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0})
    for r in scored:
        b = r["bench"]
        answer = (b.get("answer") or "").strip().upper()
        if answer not in ("A", "B", "C", "D"):
            continue
        if dataset == "videomme":
            key = str(b.get("duration") or "?")
        else:
            key = str(b.get("task_type") or "?")
        by_bucket[key]["total"] += 1
        if b.get("model_answer") and (
                b["correct"] if b["correct"] is not None
                else b["model_answer"] == answer):
            by_bucket[key]["correct"] += 1
    if by_bucket:
        label = "duration" if dataset == "videomme" else "task_type"
        print(f"\n  accuracy by {label}:")
        for k, v in sorted(by_bucket.items()):
            acc = v["correct"] / v["total"] if v["total"] else 0.0
            print(f"    {k:35s} total={v['total']:5d} "
                  f"correct={v['correct']:5d} accuracy={acc:.3f}")

    # ---- per-run detail ----
    per_run: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0})
    for r in scored:
        b = r["bench"]
        answer = (b.get("answer") or "").strip().upper()
        if answer not in ("A", "B", "C", "D"):
            continue
        per_run[r["session_id"]]["total"] += 1
        if b.get("model_answer") and (
                b["correct"] if b["correct"] is not None
                else b["model_answer"] == answer):
            per_run[r["session_id"]]["correct"] += 1
    print(f"\n  per-run (session) detail:")
    for sid, v in sorted(per_run.items()):
        acc = v["correct"] / v["total"] if v["total"] else 0.0
        print(f"    {sid[:40]:40s} total={v['total']:4d} "
              f"correct={v['correct']:4d} accuracy={acc:.3f}")

    # ---- E2E timing summary ----
    # Truthiness filter (>0): a zero-token completion records
    # latency_sec=0.0, which must stay OUT of the stats (EMPTY, not
    # FAIL — same semantics as the scenarios' own wave aggregation).
    ttfts = [r["ttft_sec"] for r in scored
             if isinstance(r["ttft_sec"], (int, float))
             and r["ttft_sec"] > 0]
    lats = [r["latency_sec"] for r in scored
            if isinstance(r["latency_sec"], (int, float))
            and r["latency_sec"] > 0]
    if ttfts:
        print(f"\n  E2E TTFT  (n={len(ttfts)}): "
              f"mean={sum(ttfts)/len(ttfts):.2f}s p50={_pct(ttfts,50):.2f}s "
              f"p90={_pct(ttfts,90):.2f}s max={max(ttfts):.2f}s")
    if lats:
        print(f"  E2E LAT   (n={len(lats)}): "
              f"mean={sum(lats)/len(lats):.2f}s p50={_pct(lats,50):.2f}s "
              f"p90={_pct(lats,90):.2f}s max={max(lats):.2f}s")

    # per-question detail for --json
    for r in scored:
        out.append({
            "scenario": scenario,
            "profile": r["network_profile"],
            "session_id": r["session_id"],
            "run_index": r["run_index"],
            "timestamp": r["timestamp"],
            "mode": r["mode"],
            "model": r["model"],
            "ttft_sec": r["ttft_sec"],
            "latency_sec": r["latency_sec"],
            "tokens_out": r["tokens_out"],
            **r["bench"],
        })
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-bench accuracy report from traffic_logs DB "
                    "(encoder_placement* scenarios, dataset mode)")
    parser.add_argument("--db", default="logs/traffic_logs.db",
                        help="Path to SQLite DB (default: logs/traffic_logs.db)")
    parser.add_argument("--scenario", required=True,
                        help="scenario_id in the DB (e.g. "
                             "encoder_placement_qwen36, "
                             "encoder_placement_concurrent)")
    parser.add_argument("--profile", default="all",
                        help="network_profile filter, or 'all' (default)")
    parser.add_argument("--json", default=None,
                        help="Optional path to dump the per-question detail "
                             "as JSON (appends .json rows for every combo)")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        sys.exit(f"no such db: {db}")

    with sqlite3.connect(str(db)) as conn:
        try:
            profiles = [r[0] for r in conn.execute(
                "SELECT DISTINCT network_profile FROM traffic_logs "
                "WHERE scenario_id = ? ORDER BY network_profile",
                (args.scenario,))]
        except sqlite3.OperationalError as e:
            sys.exit(f"DB error: {e} (db={db})")
    if not profiles:
        sys.exit(f"scenario '{args.scenario}' has no rows in {db}")

    if args.profile != "all":
        if args.profile not in profiles:
            sys.exit(f"profile '{args.profile}' not found for scenario "
                     f"'{args.scenario}' (available: {', '.join(profiles)})")
        profiles = [args.profile]

    detail: list[dict] = []
    any_scored = False
    for p in profiles:
        if report_combo(str(db), args.scenario, p, detail):
            any_scored = True

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, ensure_ascii=False)
        print(f"\n[EncPlBenchDB] per-question detail ({len(detail)} rows) "
              f"saved to {out}")

    if not any_scored:
        sys.exit(2)


if __name__ == "__main__":
    main()
