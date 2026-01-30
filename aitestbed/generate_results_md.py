#!/usr/bin/env python3
"""
Generate RESULTS.md from the SQLite database with anonymized identifiers.
"""

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from analysis.anonymization import get_anonymizer


NETWORK_PROFILES = [
    ("ideal_6g", "1ms delay, 0% loss, unlimited BW (baseline)"),
    ("5g_urban", "20ms delay, 0.1% loss, 100Mbit"),
    ("wifi_good", "30ms delay, 0.1% loss, 50Mbit"),
    ("cell_edge", "120ms delay, 1% loss, 5Mbit"),
    ("satellite", "600ms delay, 0.5% loss, 10Mbit"),
    ("congested", "200ms delay, 3% loss, 1Mbit"),
    ("5qi_7", "100ms delay, 0.1% loss (Voice/Live Streaming)"),
    ("5qi_80", "10ms delay, 0.0001% loss (Low-latency eMBB/AR)"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RESULTS.md from database")
    parser.add_argument("--db", default="logs/traffic_logs.db", help="Path to SQLite database")
    parser.add_argument("--since-timestamp", type=float, default=0.0, help="Only include data after this Unix timestamp")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional duration override in seconds")
    parser.add_argument("--all-runs", action="store_true", help="Include all runs instead of latest per scenario")
    parser.add_argument("--run-gap-sec", type=float, default=300.0, help="Gap in seconds to split runs per scenario")
    parser.add_argument("--output", default="RESULTS.md", help="Output path for RESULTS.md")
    args = parser.parse_args()

    db_path = Path(args.db)
    output_path = Path(args.output)
    anonymizer = get_anonymizer()
    since_ts = float(args.since_timestamp or 0.0)

    def load_records(db_file: Path, since_timestamp: float) -> list[dict]:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if since_timestamp:
            cursor.execute(
                "SELECT * FROM traffic_logs WHERE timestamp > ? ORDER BY timestamp",
                (since_timestamp,),
            )
        else:
            cursor.execute("SELECT * FROM traffic_logs ORDER BY timestamp")
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return records

    def fuse_latest_runs(records: list[dict], gap_sec: float) -> list[dict]:
        if not records:
            return []
        by_scenario: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            scenario = record.get("scenario_id") or "unknown"
            by_scenario[scenario].append(record)

        fused: list[dict] = []
        for recs in by_scenario.values():
            recs.sort(key=lambda r: r.get("timestamp", 0.0))
            start_idx = 0
            for i in range(len(recs) - 1, 0, -1):
                if recs[i]["timestamp"] - recs[i - 1]["timestamp"] > gap_sec:
                    start_idx = i
                    break
            fused.extend(recs[start_idx:])
        return fused

    def scenario_provider_map(records: list[dict]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for record in records:
            scenario = record.get("scenario_id") or "unknown"
            provider = record.get("provider")
            if scenario not in mapping and provider:
                mapping[scenario] = provider
        return mapping

    records = load_records(db_path, since_ts)
    if not args.all_runs:
        records = fuse_latest_runs(records, gap_sec=float(args.run_gap_sec))

    provider_map = scenario_provider_map(records)

    def scenario_label(scenario_id: str) -> str:
        base = anonymizer.scenario_alias(scenario_id) or scenario_id
        provider = provider_map.get(scenario_id)
        provider_alias = anonymizer.provider_alias(provider) if provider else ""
        if provider_alias:
            return f"{base} - {provider_alias}"
        return base

    total_records = len(records)
    scenario_ids = sorted({r.get("scenario_id") or "unknown" for r in records})
    total_scenarios = len(scenario_ids)
    total_profiles = len({r.get("network_profile") for r in records if r.get("network_profile")})
    success_count = sum(1 for r in records if r.get("success"))
    success_rate = round(100.0 * success_count / total_records, 1) if total_records else 0.0

    if args.duration_sec is not None:
        duration_sec = int(args.duration_sec)
    else:
        timestamps = [r.get("timestamp", 0.0) for r in records if r.get("timestamp")]
        if timestamps:
            duration_sec = int(max(timestamps) - min(timestamps))
        else:
            duration_sec = 0

    lines = []
    lines.append("# 6G AI Traffic Characterization Testbed - Test Results")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Test Duration: {duration_sec // 60} minutes")
    lines.append("")
    lines.append("## Test Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Records | {total_records} |")
    lines.append(f"| Scenarios Tested | {total_scenarios} |")
    lines.append(f"| Network Profiles | {total_profiles} |")
    lines.append(f"| Success Rate | {success_rate}% |")
    run_selection = "all runs" if args.all_runs else f"latest per scenario (gap > {int(args.run_gap_sec)}s)"
    lines.append(f"| Run Selection | {run_selection} |")
    lines.append("")
    lines.append("## Scenarios Tested")
    lines.append("")
    lines.append("| Scenario | Runs | Success Rate | Avg Latency (s) |")
    lines.append("|----------|------|--------------|-----------------|")

    by_scenario: dict[str, list[dict]] = defaultdict(list)
    by_scenario_profile: dict[tuple[str, str], list[dict]] = defaultdict(list)
    ttft_by_scenario_profile: dict[tuple[str, str], list[float]] = defaultdict(list)

    for record in records:
        scenario_id = record.get("scenario_id") or "unknown"
        profile = record.get("network_profile") or "unknown"
        by_scenario[scenario_id].append(record)
        by_scenario_profile[(scenario_id, profile)].append(record)

        t_start = record.get("t_request_start")
        t_first = record.get("t_first_token")
        if t_start and t_first and t_first > t_start:
            ttft_by_scenario_profile[(scenario_id, profile)].append(t_first - t_start)

    for scenario_id in sorted(by_scenario.keys()):
        recs = by_scenario[scenario_id]
        runs = len(recs)
        success = sum(1 for r in recs if r.get("success"))
        rate = round(100.0 * success / runs, 1) if runs else 0.0
        latencies = [r.get("latency_sec", 0.0) for r in recs if r.get("latency_sec") is not None]
        avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        label = scenario_label(scenario_id)
        lines.append(f"| {label} | {runs} | {rate} | {avg_lat} |")

    lines.append("")
    lines.append("## Network Profiles Used")
    lines.append("")
    lines.append("| Profile | Description |")
    lines.append("|---------|-------------|")
    for profile, description in NETWORK_PROFILES:
        lines.append(f"| {profile} | {description} |")

    lines.append("")
    lines.append("## Detailed Results by Scenario and Profile")
    lines.append("")
    lines.append("| Scenario | Profile | Runs | Success | Avg Latency (s) | Min (s) | Max (s) |")
    lines.append("|----------|---------|------|---------|-----------------|---------|---------|")

    for (scenario_id, profile) in sorted(by_scenario_profile.keys()):
        recs = by_scenario_profile[(scenario_id, profile)]
        runs = len(recs)
        success = sum(1 for r in recs if r.get("success"))
        success_rate = f"{round(100.0 * success / runs, 0)}%" if runs else "0%"
        latencies = [r.get("latency_sec", 0.0) for r in recs if r.get("latency_sec") is not None]
        avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        min_lat = round(min(latencies), 3) if latencies else 0.0
        max_lat = round(max(latencies), 3) if latencies else 0.0
        label = scenario_label(scenario_id)
        lines.append(f"| {label} | {profile} | {runs} | {success_rate} | {avg_lat} | {min_lat} | {max_lat} |")

    lines.append("")
    lines.append("## Time to First Token (TTFT)")
    lines.append("")
    lines.append("| Scenario | Profile | Avg TTFT (s) | Min (s) | Max (s) |")
    lines.append("|----------|---------|--------------|---------|---------|")

    for (scenario_id, profile) in sorted(ttft_by_scenario_profile.keys()):
        ttfts = ttft_by_scenario_profile[(scenario_id, profile)]
        if not ttfts:
            continue
        avg_ttft = round(sum(ttfts) / len(ttfts), 3)
        min_ttft = round(min(ttfts), 3)
        max_ttft = round(max(ttfts), 3)
        label = scenario_label(scenario_id)
        lines.append(f"| {label} | {profile} | {avg_ttft} | {min_ttft} | {max_ttft} |")

    lines.append("")
    lines.append("## Bandwidth Usage")
    lines.append("")
    lines.append("| Scenario | Avg Request (bytes) | Avg Response (bytes) | Asymmetry Ratio |")
    lines.append("|----------|---------------------|----------------------|-----------------|")

    for scenario_id in sorted(by_scenario.keys()):
        recs = by_scenario[scenario_id]
        reqs = [r.get("request_bytes", 0) or 0 for r in recs]
        resps = [r.get("response_bytes", 0) or 0 for r in recs]
        avg_req = int(round(sum(reqs) / len(reqs))) if reqs else 0
        avg_resp = int(round(sum(resps) / len(resps))) if resps else 0
        if avg_req > 0:
            ratio = f"{round(avg_resp / avg_req, 0):.0f}:1"
        else:
            ratio = "0:1"
        label = scenario_label(scenario_id)
        lines.append(f"| {label} | {avg_req} | {avg_resp} | {ratio} |")

    lines.append("")
    lines.append("## SDP Offer/Answer Samples (WebRTC)")
    lines.append("")
    preferred_scenario = "realtime_audio_webrtc"
    preferred_label = scenario_label(preferred_scenario)

    sdp_dir = Path("logs/sdp")
    latest_offer = None
    latest_answer = None
    scenario_sdp_found = False

    def latest_sdp_hash_for_scenario(scenario_id: str) -> str | None:
        candidates = [
            r for r in records
            if r.get("scenario_id") == scenario_id and r.get("metadata")
        ]
        for rec in sorted(candidates, key=lambda r: r.get("timestamp", 0.0), reverse=True):
            try:
                meta = json.loads(rec.get("metadata") or "{}")
            except Exception:
                continue
            if isinstance(meta, dict) and meta.get("sdp_offer_hash"):
                return meta.get("sdp_offer_hash")
        return None

    def find_sdp_pair_by_hash(prefix: str) -> tuple[Path | None, Path | None]:
        offers = sorted(
            sdp_dir.glob(f"*_{prefix}_offer.sdp"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for offer_path in offers:
            answer_path = Path(str(offer_path).replace("_offer.sdp", "_answer.sdp"))
            if answer_path.exists():
                return offer_path, answer_path
        return None, None

    if sdp_dir.exists():
        scenario_hash = latest_sdp_hash_for_scenario(preferred_scenario)
        if scenario_hash:
            latest_offer, latest_answer = find_sdp_pair_by_hash(scenario_hash[:8])
            scenario_sdp_found = latest_offer is not None

        if latest_offer is None:
            offers = sorted(sdp_dir.glob("*_offer.sdp"), key=lambda p: p.stat().st_mtime, reverse=True)
            for offer_path in offers:
                answer_path = Path(str(offer_path).replace("_offer.sdp", "_answer.sdp"))
                if not answer_path.exists():
                    continue
                try:
                    offer_text = offer_path.read_text()
                    answer_text = answer_path.read_text()
                except Exception:
                    offer_text = ""
                    answer_text = ""
                if "m=audio" in offer_text and "m=audio" in answer_text:
                    latest_offer = offer_path
                    latest_answer = answer_path
                    break
            if latest_offer is None and offers:
                latest_offer = offers[0]
                latest_answer = Path(str(latest_offer).replace("_offer.sdp", "_answer.sdp"))

    if latest_offer:
        if scenario_sdp_found:
            lines.append(
                f"Latest offer/answer pair captured during realtime WebRTC sessions for "
                f"{preferred_label} from `logs/sdp/`."
            )
        else:
            lines.append(
                f"No SDP samples found for {preferred_label}; showing the latest "
                f"available WebRTC offer/answer with audio from `logs/sdp/`."
            )
        offer_file = latest_offer.name
        answer_file = latest_answer.name if latest_answer else latest_offer.name.replace("_offer.sdp", "_answer.sdp")
        offer_bytes = latest_offer.stat().st_size if latest_offer.exists() else 0
        answer_bytes = latest_answer.stat().st_size if latest_answer and latest_answer.exists() else 0

        lines.append("")
        lines.append(f"**Offer:** `logs/sdp/{offer_file}` ({offer_bytes} bytes)")
        lines.append("```sdp")
        try:
            lines.append(latest_offer.read_text().rstrip())
        except Exception:
            lines.append("(failed to read offer file)")
        lines.append("```")

        lines.append("")
        lines.append(f"**Answer:** `logs/sdp/{answer_file}` ({answer_bytes} bytes)")
        lines.append("```sdp")
        if latest_answer and latest_answer.exists():
            try:
                lines.append(latest_answer.read_text().rstrip())
            except Exception:
                lines.append("(failed to read answer file)")
        else:
            lines.append("(missing answer file)")
        lines.append("```")
    else:
        lines.append("")
        lines.append("No SDP files found in logs/sdp.")

    lines.append("")
    lines.append("## Charts")
    lines.append("")
    lines.append("### Latency Distribution")
    lines.append("![Latency by Scenario](reports/figures/latency_by_scenario.png)")
    lines.append("")
    lines.append("### Time to First Token (TTFT)")
    lines.append("![TTFT by Scenario](reports/figures/ttft_by_scenario.png)")
    lines.append("")
    lines.append("### Latency Breakdown (TTFT vs Generation)")
    lines.append("![Latency Breakdown](reports/figures/latency_breakdown.png)")
    lines.append("")
    lines.append("### Bandwidth Asymmetry")
    lines.append("![Bandwidth Asymmetry](reports/figures/bandwidth_asymmetry.png)")
    lines.append("")
    lines.append("### Throughput")
    lines.append("![Throughput](reports/figures/throughput_by_scenario.png)")
    lines.append("")
    lines.append("### Token Counts")
    lines.append("![Token Counts](reports/figures/token_counts.png)")
    lines.append("")
    lines.append("### Success Rate")
    lines.append("![Success Rate](reports/figures/success_rate.png)")
    lines.append("")
    lines.append("### Latency by Network Profile")
    lines.append("![Latency by Profile](reports/figures/latency_by_profile.png)")
    lines.append("")
    lines.append("### Latency Heatmap (Scenario × Profile)")
    lines.append("![Latency Heatmap](reports/figures/latency_heatmap.png)")
    lines.append("")
    lines.append("### TTFT Heatmap (Scenario × Profile)")
    lines.append("![TTFT Heatmap](reports/figures/ttft_heatmap.png)")
    lines.append("")
    lines.append("### Streaming Metrics")
    lines.append("![Streaming Metrics](reports/figures/streaming_metrics.png)")
    lines.append("")
    lines.append("### Protocol Comparison")
    lines.append("![Protocol Comparison](reports/figures/protocol_comparison.png)")
    lines.append("")
    lines.append("### Data Volume")
    lines.append("![Data Volume](reports/figures/data_volume.png)")
    lines.append("")
    lines.append("## Data Export")
    lines.append("")
    lines.append("All chart data is available in Excel format: `reports/chart_data.xlsx`")
    lines.append("")
    lines.append("Sheets included:")
    lines.append("- Latency_by_Scenario")
    lines.append("- TTFT_by_Scenario")
    lines.append("- Latency_Breakdown")
    lines.append("- Throughput")
    lines.append("- Bandwidth_Asymmetry")
    lines.append("- Success_Rate")
    lines.append("- Streaming_Metrics")
    lines.append("- Token_Counts")
    lines.append("- Latency_by_Profile")
    lines.append("- Latency_Heatmap")
    lines.append("- TTFT_Heatmap")
    lines.append("- Raw_Data")

    output_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
