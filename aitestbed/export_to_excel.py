#!/usr/bin/env python3
"""
Export all chart data to Excel for external plotting.
"""

import sqlite3
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Optional

from analysis.anonymization import get_anonymizer

ANONYMIZER = get_anonymizer()


def scenario_alias(scenario: Optional[str]) -> str:
    alias = ANONYMIZER.scenario_alias(scenario or "")
    return alias or ANONYMIZER.scenario_alias("unknown") or "Scenario Unknown"


def provider_alias(provider: Optional[str]) -> str:
    alias = ANONYMIZER.provider_alias(provider) if provider else None
    return alias or ANONYMIZER.provider_alias("unknown") or "Provider X"


def model_alias(model: Optional[str]) -> str:
    alias = ANONYMIZER.model_alias(model) if model else None
    return alias or ANONYMIZER.model_alias("unknown") or "Model X"


def scenario_provider_map(records: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        if scenario_id not in mapping and r.get("provider"):
            mapping[scenario_id] = provider_alias(r.get("provider"))
    return mapping


def scenario_label(scenario_id: Optional[str], provider_label: Optional[str]) -> str:
    base = scenario_alias(scenario_id)
    if provider_label:
        return f"{base} - {provider_label}"
    return base


def load_records(db_path: str, since_timestamp: float = None) -> list[dict]:
    """Load records from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if since_timestamp:
        cursor.execute("SELECT * FROM traffic_logs WHERE timestamp > ? ORDER BY timestamp", (since_timestamp,))
    else:
        cursor.execute("SELECT * FROM traffic_logs ORDER BY timestamp")

    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records


def fuse_latest_runs(records: list[dict], gap_sec: float = 300.0) -> list[dict]:
    """Keep only the latest run per scenario based on timestamp gaps."""
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


def export_latency_by_scenario(records: list[dict]) -> pd.DataFrame:
    """Export latency data by scenario."""
    by_scenario = defaultdict(list)
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        latency = r.get("latency_sec")
        if latency and latency > 0:
            by_scenario[scenario_id].append(latency)

    rows = []
    for scenario_id in sorted(by_scenario.keys()):
        latencies = by_scenario[scenario_id]
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        provider = scenario_providers.get(scenario_id, provider_alias(None))
        rows.append({
            "Scenario": scenario_label(scenario_id, provider),
            "Provider": provider,
            "Count": len(latencies),
            "Mean_Latency_sec": sum(latencies) / len(latencies),
            "P50_Latency_sec": sorted_lat[len(sorted_lat)//2],
            "P95_Latency_sec": sorted_lat[min(p95_idx, len(sorted_lat)-1)],
            "Min_Latency_sec": min(latencies),
            "Max_Latency_sec": max(latencies),
        })

    return pd.DataFrame(rows)


def export_ttft_by_scenario(records: list[dict]) -> pd.DataFrame:
    """Export TTFT data by scenario."""
    by_scenario = defaultdict(list)
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        t_start = r.get("t_request_start")
        t_first = r.get("t_first_token")
        if t_start and t_first and t_first > t_start:
            by_scenario[scenario_id].append(t_first - t_start)

    rows = []
    for scenario_id in sorted(by_scenario.keys()):
        ttfts = by_scenario[scenario_id]
        sorted_ttft = sorted(ttfts)
        p95_idx = int(len(sorted_ttft) * 0.95)
        provider = scenario_providers.get(scenario_id, provider_alias(None))
        rows.append({
            "Scenario": scenario_label(scenario_id, provider),
            "Provider": provider,
            "Count": len(ttfts),
            "Mean_TTFT_sec": sum(ttfts) / len(ttfts),
            "P50_TTFT_sec": sorted_ttft[len(sorted_ttft)//2],
            "P95_TTFT_sec": sorted_ttft[min(p95_idx, len(sorted_ttft)-1)],
            "Min_TTFT_sec": min(ttfts),
            "Max_TTFT_sec": max(ttfts),
        })

    return pd.DataFrame(rows)


def export_latency_breakdown(records: list[dict]) -> pd.DataFrame:
    """Export TTFT vs generation time breakdown."""
    by_scenario = defaultdict(lambda: {"ttft": [], "gen": []})
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        t_start = r.get("t_request_start")
        t_first = r.get("t_first_token")
        t_last = r.get("t_last_token")
        if t_start and t_first and t_last and t_first > t_start and t_last >= t_first:
            by_scenario[scenario_id]["ttft"].append(t_first - t_start)
            by_scenario[scenario_id]["gen"].append(t_last - t_first)

    rows = []
    for scenario_id in sorted(by_scenario.keys()):
        ttfts = by_scenario[scenario_id]["ttft"]
        gens = by_scenario[scenario_id]["gen"]
        if ttfts and gens:
            provider = scenario_providers.get(scenario_id, provider_alias(None))
            rows.append({
                "Scenario": scenario_label(scenario_id, provider),
                "Provider": provider,
                "Count": len(ttfts),
                "Mean_TTFT_sec": sum(ttfts) / len(ttfts),
                "Mean_Generation_Time_sec": sum(gens) / len(gens),
                "Total_Latency_sec": sum(ttfts) / len(ttfts) + sum(gens) / len(gens),
            })

    return pd.DataFrame(rows)


def export_throughput(records: list[dict]) -> pd.DataFrame:
    """Export throughput data."""
    by_scenario = defaultdict(lambda: {"ul": [], "dl": []})
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        latency = r.get("latency_sec")
        req_bytes = r.get("request_bytes", 0) or 0
        resp_bytes = r.get("response_bytes", 0) or 0
        if latency and latency > 0:
            by_scenario[scenario_id]["ul"].append(req_bytes / latency)
            by_scenario[scenario_id]["dl"].append(resp_bytes / latency)

    rows = []
    for scenario_id in sorted(by_scenario.keys()):
        ul = by_scenario[scenario_id]["ul"]
        dl = by_scenario[scenario_id]["dl"]
        if ul:
            provider = scenario_providers.get(scenario_id, provider_alias(None))
            rows.append({
                "Scenario": scenario_label(scenario_id, provider),
                "Provider": provider,
                "Count": len(ul),
                "Mean_UL_Throughput_KBps": (sum(ul) / len(ul)) / 1024,
                "Mean_DL_Throughput_KBps": (sum(dl) / len(dl)) / 1024,
            })

    return pd.DataFrame(rows)


def export_bandwidth_asymmetry(records: list[dict]) -> pd.DataFrame:
    """Export bandwidth asymmetry data."""
    by_scenario = defaultdict(lambda: {"ul": 0, "dl": 0})
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        by_scenario[scenario_id]["ul"] += r.get("request_bytes", 0) or 0
        by_scenario[scenario_id]["dl"] += r.get("response_bytes", 0) or 0

    rows = []
    for scenario_id in sorted(by_scenario.keys()):
        ul = by_scenario[scenario_id]["ul"]
        dl = by_scenario[scenario_id]["dl"]
        if dl > 0:
            provider = scenario_providers.get(scenario_id, provider_alias(None))
            rows.append({
                "Scenario": scenario_label(scenario_id, provider),
                "Provider": provider,
                "Total_UL_Bytes": ul,
                "Total_DL_Bytes": dl,
                "DL_to_UL_Ratio": dl / max(ul, 1),
            })

    return pd.DataFrame(rows)


def export_success_rate(records: list[dict]) -> pd.DataFrame:
    """Export success rate data."""
    by_scenario = defaultdict(lambda: {"success": 0, "total": 0})
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        by_scenario[scenario_id]["total"] += 1
        if r.get("success"):
            by_scenario[scenario_id]["success"] += 1

    rows = []
    for scenario_id in sorted(by_scenario.keys()):
        total = by_scenario[scenario_id]["total"]
        success = by_scenario[scenario_id]["success"]
        provider = scenario_providers.get(scenario_id, provider_alias(None))
        rows.append({
            "Scenario": scenario_label(scenario_id, provider),
            "Provider": provider,
            "Total_Runs": total,
            "Successful_Runs": success,
            "Failed_Runs": total - success,
            "Success_Rate_Pct": (success / total) * 100 if total > 0 else 0,
        })

    return pd.DataFrame(rows)


def export_streaming_metrics(records: list[dict]) -> pd.DataFrame:
    """Export streaming metrics data."""
    by_scenario = defaultdict(lambda: {"chunks": [], "rates": []})
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        chunk_count = r.get("chunk_count")
        t_first = r.get("t_first_token")
        t_last = r.get("t_last_token")
        if chunk_count and chunk_count > 0:
            by_scenario[scenario_id]["chunks"].append(chunk_count)
            if t_first and t_last and t_last > t_first:
                by_scenario[scenario_id]["rates"].append(chunk_count / (t_last - t_first))

    rows = []
    for scenario_id in sorted(by_scenario.keys()):
        chunks = by_scenario[scenario_id]["chunks"]
        rates = by_scenario[scenario_id]["rates"]
        if chunks:
            provider = scenario_providers.get(scenario_id, provider_alias(None))
            rows.append({
                "Scenario": scenario_label(scenario_id, provider),
                "Provider": provider,
                "Count": len(chunks),
                "Mean_Chunk_Count": sum(chunks) / len(chunks),
                "Mean_Chunk_Rate_per_sec": sum(rates) / len(rates) if rates else 0,
            })

    return pd.DataFrame(rows)


def export_token_counts(records: list[dict]) -> pd.DataFrame:
    """Export token count data."""
    by_scenario = defaultdict(lambda: {"in": [], "out": []})
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        tokens_in = r.get("tokens_in")
        tokens_out = r.get("tokens_out")
        if tokens_in:
            by_scenario[scenario_id]["in"].append(tokens_in)
        if tokens_out:
            by_scenario[scenario_id]["out"].append(tokens_out)

    rows = []
    for scenario_id in sorted(by_scenario.keys()):
        tin = by_scenario[scenario_id]["in"]
        tout = by_scenario[scenario_id]["out"]
        if tin or tout:
            provider = scenario_providers.get(scenario_id, provider_alias(None))
            rows.append({
                "Scenario": scenario_label(scenario_id, provider),
                "Provider": provider,
                "Count": max(len(tin), len(tout)),
                "Mean_Tokens_In": sum(tin) / len(tin) if tin else 0,
                "Mean_Tokens_Out": sum(tout) / len(tout) if tout else 0,
            })

    return pd.DataFrame(rows)


def export_latency_by_profile(records: list[dict]) -> pd.DataFrame:
    """Export latency by network profile."""
    by_profile = defaultdict(list)
    for r in records:
        profile = r.get("network_profile", "unknown")
        latency = r.get("latency_sec")
        if latency and latency > 0:
            by_profile[profile].append(latency)

    rows = []
    for profile in sorted(by_profile.keys()):
        latencies = by_profile[profile]
        mean = sum(latencies) / len(latencies)
        std = (sum((v - mean) ** 2 for v in latencies) / len(latencies)) ** 0.5
        rows.append({
            "Network_Profile": profile,
            "Count": len(latencies),
            "Mean_Latency_sec": mean,
            "Std_Dev_sec": std,
            "Min_Latency_sec": min(latencies),
            "Max_Latency_sec": max(latencies),
        })

    return pd.DataFrame(rows)


def export_latency_heatmap(records: list[dict]) -> pd.DataFrame:
    """Export latency heatmap data (scenario x profile matrix)."""
    data = defaultdict(lambda: defaultdict(list))
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        profile = r.get("network_profile", "unknown")
        latency = r.get("latency_sec")
        if latency and latency > 0:
            data[scenario_id][profile].append(latency)

    scenarios = sorted(data.keys())
    profiles = sorted(set(p for s in data.values() for p in s.keys()))

    rows = []
    for scenario_id in scenarios:
        provider = scenario_providers.get(scenario_id, provider_alias(None))
        row = {"Scenario": scenario_label(scenario_id, provider), "Provider": provider}
        for profile in profiles:
            latencies = data[scenario_id].get(profile, [])
            row[profile] = sum(latencies) / len(latencies) if latencies else None
        rows.append(row)

    return pd.DataFrame(rows)


def export_ttft_heatmap(records: list[dict]) -> pd.DataFrame:
    """Export TTFT heatmap data (scenario x profile matrix)."""
    data = defaultdict(lambda: defaultdict(list))
    scenario_providers = scenario_provider_map(records)
    for r in records:
        scenario_id = r.get("scenario_id") or "unknown"
        profile = r.get("network_profile", "unknown")
        t_start = r.get("t_request_start")
        t_first = r.get("t_first_token")
        if t_start and t_first and t_first > t_start:
            data[scenario_id][profile].append(t_first - t_start)

    scenarios = sorted(data.keys())
    profiles = sorted(set(p for s in data.values() for p in s.keys()))

    rows = []
    for scenario_id in scenarios:
        provider = scenario_providers.get(scenario_id, provider_alias(None))
        row = {"Scenario": scenario_label(scenario_id, provider), "Provider": provider}
        for profile in profiles:
            ttfts = data[scenario_id].get(profile, [])
            row[profile] = sum(ttfts) / len(ttfts) if ttfts else None
        rows.append(row)

    return pd.DataFrame(rows)


def export_protocol_comparison(records: list[dict]) -> pd.DataFrame:
    """Export protocol comparison data."""
    protocols = {
        "REST": [],
        "REST_Streaming": [],
        "WebSocket": [],
        "WebRTC": [],
    }

    for r in records:
        scenario = r.get("scenario_id", "")
        latency = r.get("latency_sec")
        if not latency or latency <= 0:
            continue

        if "webrtc" in scenario.lower():
            protocols["WebRTC"].append(latency)
        elif "realtime" in scenario.lower():
            protocols["WebSocket"].append(latency)
        elif "streaming" in scenario.lower():
            protocols["REST_Streaming"].append(latency)
        else:
            protocols["REST"].append(latency)

    rows = []
    for protocol, latencies in protocols.items():
        if latencies:
            sorted_lat = sorted(latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            rows.append({
                "Protocol": protocol,
                "Count": len(latencies),
                "Mean_Latency_sec": sum(latencies) / len(latencies),
                "P50_Latency_sec": sorted_lat[len(sorted_lat)//2],
                "P95_Latency_sec": sorted_lat[min(p95_idx, len(sorted_lat)-1)],
                "Min_Latency_sec": min(latencies),
                "Max_Latency_sec": max(latencies),
            })

    return pd.DataFrame(rows)


def export_error_analysis(records: list[dict]) -> pd.DataFrame:
    """Export error analysis data."""
    error_counts = defaultdict(int)
    success_count = 0

    for r in records:
        if r.get("success"):
            success_count += 1
        else:
            error_type = r.get("error_type", "unknown") or "unknown"
            error_counts[error_type] += 1

    rows = [{"Category": "Success", "Count": success_count}]
    for error_type, count in sorted(error_counts.items()):
        rows.append({"Category": f"Error: {error_type}", "Count": count})

    return pd.DataFrame(rows)


def export_raw_data(records: list[dict]) -> pd.DataFrame:
    """Export raw data with selected columns."""
    columns = [
        "scenario_id", "network_profile", "provider", "model",
        "latency_sec", "request_bytes", "response_bytes",
        "tokens_in", "tokens_out", "t_request_start", "t_first_token", "t_last_token",
        "success", "error_type", "tool_calls_count", "chunk_count", "is_streaming"
    ]

    rows = []
    for r in records:
        row = {col: r.get(col) for col in columns if col in r}
        provider = provider_alias(row.get("provider"))
        row["scenario_id"] = scenario_label(row.get("scenario_id"), provider)
        row["provider"] = provider
        row["model"] = model_alias(row.get("model"))
        row["provider_detected"] = provider
        # Calculate TTFT
        t_start = r.get("t_request_start")
        t_first = r.get("t_first_token")
        row["ttft_sec"] = (t_first - t_start) if t_start and t_first and t_first > t_start else None
        # Calculate generation time
        t_last = r.get("t_last_token")
        row["generation_time_sec"] = (t_last - t_first) if t_first and t_last and t_last > t_first else None
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export chart data to Excel")
    parser.add_argument("--db", default="logs/traffic_logs.db", help="Database path")
    parser.add_argument("--since-timestamp", type=float, help="Filter by timestamp")
    parser.add_argument("--all-runs", action="store_true", help="Include all runs instead of latest per scenario")
    parser.add_argument("--run-gap-sec", type=float, default=300.0, help="Gap in seconds to split runs per scenario")
    parser.add_argument("--output", default="reports/chart_data.xlsx", help="Output Excel file")
    args = parser.parse_args()

    print(f"Loading data from {args.db}...")
    records = load_records(args.db, args.since_timestamp)
    print(f"Loaded {len(records)} records")

    if not args.all_runs:
        records = fuse_latest_runs(records, gap_sec=float(args.run_gap_sec))
        print(f"Using {len(records)} records from latest run per scenario")

    if not records:
        print("No data found!")
        return

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to {args.output}...")

    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        # Export each dataset to a separate sheet
        sheets = [
            ("Latency_by_Scenario", export_latency_by_scenario(records)),
            ("TTFT_by_Scenario", export_ttft_by_scenario(records)),
            ("Latency_Breakdown", export_latency_breakdown(records)),
            ("Throughput", export_throughput(records)),
            ("Bandwidth_Asymmetry", export_bandwidth_asymmetry(records)),
            ("Success_Rate", export_success_rate(records)),
            ("Streaming_Metrics", export_streaming_metrics(records)),
            ("Token_Counts", export_token_counts(records)),
            ("Latency_by_Profile", export_latency_by_profile(records)),
            ("Protocol_Comparison", export_protocol_comparison(records)),
            ("Error_Analysis", export_error_analysis(records)),
            ("Latency_Heatmap", export_latency_heatmap(records)),
            ("TTFT_Heatmap", export_ttft_heatmap(records)),
            ("Raw_Data", export_raw_data(records)),
        ]

        for sheet_name, df in sheets:
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  ✓ {sheet_name}: {len(df)} rows")
            else:
                print(f"  - {sheet_name}: (no data)")

    print(f"\n✓ Excel file saved to: {args.output}")


if __name__ == "__main__":
    main()
