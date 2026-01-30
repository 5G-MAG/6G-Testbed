#!/usr/bin/env python3
"""
Generate visualization charts from traffic test data.
"""

import sqlite3
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

from analysis.anonymization import get_anonymizer

# Check dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional pcap analysis
try:
    from analysis import HAS_PCAP_ANALYZER, analyze_multiple_pcaps, merge_pcap_metrics
except ImportError:
    HAS_PCAP_ANALYZER = False
    analyze_multiple_pcaps = None
    merge_pcap_metrics = None

ANONYMIZER = get_anonymizer()

def load_data_from_db(db_path: str = "logs/traffic_logs.db", since_timestamp: float = None) -> list[dict]:
    """Load records from the SQLite database, optionally filtered by timestamp."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if since_timestamp:
        cursor.execute("SELECT * FROM traffic_logs WHERE timestamp > ? ORDER BY timestamp", (since_timestamp,))
    else:
        cursor.execute("SELECT * FROM traffic_logs ORDER BY timestamp")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_latest_run_timestamp(db_path: str = "logs/traffic_logs.db", run_duration_minutes: int = None) -> float:
    """Calculate timestamp for the start of the latest run.

    If run_duration_minutes is provided, subtract that from the max timestamp.
    Otherwise, detect runs by looking for gaps > 60 seconds between records.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(timestamp) FROM traffic_logs")
    max_ts = cursor.fetchone()[0]

    if max_ts is None:
        conn.close()
        return 0

    if run_duration_minutes:
        conn.close()
        return max_ts - (run_duration_minutes * 60)

    # Auto-detect: find the last gap > 5 minutes (indicating separate runs)
    cursor.execute("""
        SELECT timestamp FROM traffic_logs
        ORDER BY timestamp DESC
    """)
    timestamps = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(timestamps) < 2:
        return 0

    # Find the first gap > 300 seconds (5 min) going backwards
    for i in range(len(timestamps) - 1):
        gap = timestamps[i] - timestamps[i + 1]
        if gap > 300:  # 5 minute gap indicates separate run
            return timestamps[i + 1] - 60  # Include records from just before

    return 0  # No gap found, return all data


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


def aggregate_by_scenario(records: list[dict]) -> dict:
    """Aggregate records by scenario ID."""
    by_scenario = defaultdict(list)
    for r in records:
        scenario = r.get("scenario_id", "unknown")
        by_scenario[scenario].append(r)
    return dict(by_scenario)


def _provider_alias_from_records(records: list[dict]) -> str:
    for record in records:
        provider = record.get("provider")
        if provider:
            return ANONYMIZER.provider_alias(provider)
    return ""


def _scenario_alias(scenario: str) -> str:
    alias = ANONYMIZER.scenario_alias(scenario)
    return alias or "Scenario Unknown"


def format_scenario_label(
    scenario: str,
    records: Optional[list[dict]] = None,
    include_provider: bool = True,
    provider: Optional[str] = None,
) -> str:
    """Format scenario name for chart labels."""
    name = _scenario_alias(scenario)
    provider_alias = ""
    if include_provider:
        if provider:
            provider_alias = ANONYMIZER.provider_alias(provider)
        elif records:
            provider_alias = _provider_alias_from_records(records)
    if provider_alias:
        return f"{name} - {provider_alias}"
    return name


def _build_provider_palette(records: list[dict]) -> dict[str, str]:
    providers = sorted({
        ANONYMIZER.provider_alias(r.get("provider"))
        for r in records
        if r.get("provider")
    })
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return {provider: palette[idx % len(palette)] for idx, provider in enumerate(providers)}


def _scenario_provider_alias_map(records: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for record in records:
        scenario = record.get("scenario_id")
        provider = record.get("provider")
        if scenario and provider and scenario not in mapping:
            mapping[scenario] = ANONYMIZER.provider_alias(provider)
    return mapping


def generate_latency_by_scenario(records: list[dict], output_dir: Path) -> str:
    """Generate latency comparison chart by scenario."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    # Filter scenarios with actual latency data
    labels = []
    means = []
    p95s = []

    for scenario, recs in sorted(by_scenario.items()):
        latencies = [
            r["latency_sec"]
            for r in recs
            if r.get("success") and r.get("latency_sec") and r["latency_sec"] > 0
        ]
        if latencies:
            labels.append(format_scenario_label(scenario, recs))
            means.append(sum(latencies) / len(latencies))
            sorted_lat = sorted(latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            p95s.append(sorted_lat[min(p95_idx, len(sorted_lat)-1)])

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(18, 8))

    x = range(len(labels))
    width = 0.35

    # Use consistent colors for Mean vs P95
    bars1 = ax.bar([i - width/2 for i in x], means, width, label='Mean Latency', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], p95s, width, label='P95 Latency', color='coral')

    ax.set_xlabel('Scenario (Provider)')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Latency by Scenario Type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "latency_by_scenario.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_throughput_chart(records: list[dict], output_dir: Path) -> str:
    """Generate throughput (bytes/sec) chart by scenario."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    scenarios = []
    ul_rates = []
    dl_rates = []

    for scenario, recs in sorted(by_scenario.items()):
        valid_recs = [r for r in recs if r.get("latency_sec") and r["latency_sec"] > 0]
        if valid_recs:
            ul_throughput = []
            dl_throughput = []
            for r in valid_recs:
                lat = r["latency_sec"]
                req_bytes = r.get("request_bytes", 0) or 0
                resp_bytes = r.get("response_bytes", 0) or 0
                ul_throughput.append(req_bytes / lat)
                dl_throughput.append(resp_bytes / lat)

            scenarios.append(format_scenario_label(scenario, recs))
            ul_rates.append(sum(ul_throughput) / len(ul_throughput) / 1024)  # KB/s
            dl_rates.append(sum(dl_throughput) / len(dl_throughput) / 1024)  # KB/s

    if not scenarios:
        return None

    fig, ax = plt.subplots(figsize=(18, 7))

    x = range(len(scenarios))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], ul_rates, width, label='Uplink (KB/s)', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], dl_rates, width, label='Downlink (KB/s)', color='#e74c3c')

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Throughput (KB/s)')
    ax.set_title('Average Throughput by Scenario Type')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=7, rotation=55, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    output_path = output_dir / "throughput_by_scenario.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_throughput_over_time(records: list[dict], output_dir: Path, bucket_sec: int = 60) -> str:
    """Generate throughput over time (KB/s) using per-record latency."""
    if not HAS_MATPLOTLIB:
        return None

    samples = []
    for record in records:
        ts = record.get("timestamp")
        latency = record.get("latency_sec")
        if not ts or not latency or latency <= 0:
            continue
        req_bytes = record.get("request_bytes") or 0
        resp_bytes = record.get("response_bytes") or 0
        ul_kbps = (req_bytes * 8) / latency / 1000
        dl_kbps = (resp_bytes * 8) / latency / 1000
        samples.append((ts, ul_kbps, dl_kbps))

    if not samples:
        return None

    samples.sort(key=lambda x: x[0])
    min_ts = samples[0][0]

    buckets: dict[int, dict[str, float]] = {}
    for ts, ul_kb, dl_kb in samples:
        bucket = int(ts // bucket_sec) * bucket_sec
        entry = buckets.setdefault(bucket, {"ul_sum": 0.0, "dl_sum": 0.0, "count": 0})
        entry["ul_sum"] += ul_kb
        entry["dl_sum"] += dl_kb
        entry["count"] += 1

    bucket_keys = sorted(buckets.keys())
    x_minutes = []
    ul_means = []
    dl_means = []
    for bucket in bucket_keys:
        entry = buckets[bucket]
        count = entry["count"] or 1
        x_minutes.append((bucket - min_ts) / 60.0)
        ul_means.append(entry["ul_sum"] / count)
        dl_means.append(entry["dl_sum"] / count)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(x_minutes, ul_means, label="UL Throughput", color="#1f77b4", linewidth=1.5)
    ax.plot(x_minutes, dl_means, label="DL Throughput", color="#ff7f0e", linewidth=1.5)

    ax.set_xlabel("Time since first sample (min)")
    ax.set_ylabel("Throughput (Kbps)")
    ax.set_title("Throughput Over Time")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = output_dir / "throughput_over_time.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_bandwidth_asymmetry_chart(records: list[dict], output_dir: Path) -> str:
    """Generate UL/DL ratio chart showing bandwidth asymmetry."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    scenarios = []
    ratios = []
    colors = []

    for scenario, recs in sorted(by_scenario.items()):
        total_ul = sum(r.get("request_bytes", 0) or 0 for r in recs)
        total_dl = sum(r.get("response_bytes", 0) or 0 for r in recs)

        if total_dl > 0:
            ratio = total_dl / max(total_ul, 1)  # DL:UL ratio
            scenarios.append(format_scenario_label(scenario, recs))
            ratios.append(ratio)
            # Color based on asymmetry
            if ratio > 100:
                colors.append('#e74c3c')  # Very asymmetric (red)
            elif ratio > 10:
                colors.append('#f39c12')  # Moderate (orange)
            else:
                colors.append('#27ae60')  # Near symmetric (green)

    if not scenarios:
        return None

    fig, ax = plt.subplots(figsize=(20, 8))

    bars = ax.bar(range(len(scenarios)), ratios, color=colors)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('DL:UL Ratio (log scale)')
    ax.set_title('Bandwidth Asymmetry by Scenario (Downlink / Uplink)')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=8, rotation=35, ha='right')
    ax.set_yscale('log')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Symmetric (1:1)')
    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, label='10:1')
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='100:1')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.annotate(f'{ratio:.0f}:1',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()
    output_path = output_dir / "bandwidth_asymmetry.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_protocol_comparison_chart(records: list[dict], output_dir: Path) -> str:
    """Generate protocol comparison chart (REST vs WebSocket vs WebRTC)."""
    if not HAS_MATPLOTLIB:
        return None

    # Categorize by protocol
    protocols = {
        "REST": [],
        "REST Streaming": [],
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
            protocols["REST Streaming"].append(latency)
        else:
            protocols["REST"].append(latency)

    # Filter to protocols with data
    valid_protocols = {k: v for k, v in protocols.items() if v}

    if not valid_protocols:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot of latencies
    data = list(valid_protocols.values())
    labels = list(valid_protocols.keys())

    bp = ax1.boxplot(data, labels=labels, patch_artist=True)
    colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Latency (seconds)')
    ax1.set_title('Latency Distribution by Protocol')
    ax1.grid(axis='y', alpha=0.3)

    # Bar chart of mean latencies
    means = [sum(v)/len(v) for v in valid_protocols.values()]
    bars = ax2.bar(labels, means, color=colors[:len(labels)])

    ax2.set_ylabel('Mean Latency (seconds)')
    ax2.set_title('Mean Latency by Protocol')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.annotate(f'{mean:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "protocol_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_success_rate_chart(records: list[dict], output_dir: Path) -> str:
    """Generate success rate chart by scenario."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    labels = []
    success_rates = []
    bar_colors = []

    for scenario, recs in sorted(by_scenario.items()):
        if recs:
            success_count = sum(1 for r in recs if r.get("success", True))
            rate = (success_count / len(recs)) * 100
            labels.append(format_scenario_label(scenario, recs))
            success_rates.append(rate)
            # Color based on success rate
            if rate >= 95:
                bar_colors.append('#27ae60')  # Green
            elif rate >= 50:
                bar_colors.append('#f39c12')  # Orange
            else:
                bar_colors.append('#e74c3c')  # Red

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(16, 7))

    bars = ax.bar(range(len(labels)), success_rates, color=bar_colors)

    ax.set_xlabel('Scenario (Provider)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Scenario Type')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.3, label='95% threshold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.0f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / "success_rate.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_data_volume_chart(records: list[dict], output_dir: Path) -> str:
    """Generate total data volume chart by scenario."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    scenarios = []
    ul_totals = []
    dl_totals = []

    for scenario, recs in sorted(by_scenario.items()):
        ul = sum(r.get("request_bytes", 0) or 0 for r in recs) / 1024  # KB
        dl = sum(r.get("response_bytes", 0) or 0 for r in recs) / 1024  # KB
        if ul > 0 or dl > 0:
            scenarios.append(format_scenario_label(scenario, recs))
            ul_totals.append(ul)
            dl_totals.append(dl)

    if not scenarios:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(scenarios))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], ul_totals, width, label='Uplink (KB)', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], dl_totals, width, label='Downlink (KB)', color='#e74c3c')

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Total Data Volume (KB, log scale)')
    ax.set_title('Total Data Transferred by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=8, rotation=35, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "data_volume.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_latency_heatmap(records: list[dict], output_dir: Path) -> str:
    """Generate latency heatmap (scenario vs network profile)."""
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return None

    # Group by scenario and profile
    data = defaultdict(lambda: defaultdict(list))
    for r in records:
        scenario = r.get("scenario_id", "unknown")
        profile = r.get("network_profile", "unknown")
        latency = r.get("latency_sec")
        if latency and latency > 0:
            data[scenario][profile].append(latency)

    if not data:
        return None

    scenario_providers = _scenario_provider_alias_map(records)

    # Build matrix
    scenarios = sorted(data.keys())
    scenario_labels = [
        format_scenario_label(s, provider=scenario_providers.get(s))
        for s in scenarios
    ]
    profiles = sorted(set(p for s in data.values() for p in s.keys()))

    matrix = []
    for scenario in scenarios:
        row = []
        for profile in profiles:
            latencies = data[scenario].get(profile, [])
            if latencies:
                row.append(sum(latencies) / len(latencies))
            else:
                row.append(0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    fig.colorbar(im, ax=ax, label='Mean Latency (s)')

    ax.set_xticks(range(len(profiles)))
    ax.set_yticks(range(len(scenarios)))
    ax.set_xticklabels(profiles, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(scenario_labels, fontsize=8)
    ax.set_xlabel('Network Profile')
    ax.set_ylabel('Scenario')
    ax.set_title('Latency Heatmap (Scenario x Profile)')

    # Add value annotations
    for i in range(len(scenarios)):
        for j in range(len(profiles)):
            val = matrix[i][j]
            if val > 0:
                text = f'{val:.1f}' if val >= 1 else f'{val:.2f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=7,
                       color='white' if val > max(max(row) for row in matrix)/2 else 'black')

    plt.tight_layout()
    output_path = output_dir / "latency_heatmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_ttft_chart(records: list[dict], output_dir: Path) -> str:
    """Generate Time to First Token (TTFT) chart by scenario."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    labels = []
    scenario_names = []
    ttft_means = []
    ttft_p95s = []

    for scenario, recs in sorted(by_scenario.items()):
        # Calculate TTFT from t_first_token - t_request_start
        ttfts = []
        for r in recs:
            t_start = r.get("t_request_start")
            t_first = r.get("t_first_token")
            if t_start and t_first and t_first > t_start:
                ttfts.append(t_first - t_start)

        if ttfts:
            scenario_names.append(format_scenario_label(scenario, recs))
            labels.append(format_scenario_label(scenario, recs))
            ttft_means.append(sum(ttfts) / len(ttfts))
            sorted_ttft = sorted(ttfts)
            p95_idx = int(len(sorted_ttft) * 0.95)
            ttft_p95s.append(sorted_ttft[min(p95_idx, len(sorted_ttft)-1)])

    if not labels:
        return None

    # Debug output
    print(f"    TTFT chart: {len(labels)} scenarios")
    for lbl, mean in zip(scenario_names, ttft_means):
        print(f"      - {lbl}: {mean:.3f}s")

    fig, ax = plt.subplots(figsize=(18, 8))

    x = range(len(labels))
    width = 0.35

    # Use consistent colors for Mean vs P95
    bars1 = ax.bar([i - width/2 for i in x], ttft_means, width, label='Mean TTFT', color='#2ecc71')
    bars2 = ax.bar([i + width/2 for i in x], ttft_p95s, width, label='P95 TTFT', color='#e74c3c')

    ax.set_xlabel('Scenario (Provider)')
    ax.set_ylabel('Time to First Token (seconds) - Log Scale')
    ax.set_title('Time to First Token (TTFT) by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # Log scale to show both small and large values

    # Add value labels
    for bar, mean in zip(bars1, ttft_means):
        height = bar.get_height()
        ax.annotate(f'{mean:.2f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output_path = output_dir / "ttft_by_scenario.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_latency_breakdown_chart(records: list[dict], output_dir: Path) -> str:
    """Generate stacked bar chart showing TTFT vs generation time breakdown."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    labels = []
    ttft_times = []
    gen_times = []

    for scenario, recs in sorted(by_scenario.items()):
        ttfts = []
        gens = []
        for r in recs:
            t_start = r.get("t_request_start")
            t_first = r.get("t_first_token")
            t_last = r.get("t_last_token")
            if t_start and t_first and t_last and t_first > t_start and t_last >= t_first:
                ttfts.append(t_first - t_start)
                gens.append(t_last - t_first)

        if ttfts and gens:
            labels.append(format_scenario_label(scenario, recs))
            ttft_times.append(sum(ttfts) / len(ttfts))
            gen_times.append(sum(gens) / len(gens))

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(16, 7))

    x = range(len(labels))

    bars1 = ax.bar(x, ttft_times, label='Time to First Token', color='#3498db')
    bars2 = ax.bar(x, gen_times, bottom=ttft_times, label='Generation Time', color='#9b59b6')

    ax.set_xlabel('Scenario (Provider)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Latency Breakdown: TTFT vs Generation Time')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "latency_breakdown.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_token_throughput_chart(records: list[dict], output_dir: Path) -> str:
    """Generate token throughput (tokens/sec) chart."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    scenarios = []
    throughputs = []

    for scenario, recs in sorted(by_scenario.items()):
        token_rates = []
        for r in recs:
            tokens_out = r.get("tokens_out")
            t_first = r.get("t_first_token")
            t_last = r.get("t_last_token")
            if tokens_out and t_first and t_last and t_last > t_first:
                duration = t_last - t_first
                if duration > 0:
                    token_rates.append(tokens_out / duration)

        if token_rates:
            scenarios.append(format_scenario_label(scenario, recs))
            throughputs.append(sum(token_rates) / len(token_rates))

    if not scenarios:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.viridis([i/len(scenarios) for i in range(len(scenarios))])
    bars = ax.bar(range(len(scenarios)), throughputs, color=colors)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Token Generation Throughput by Scenario')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=8, rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, tp in zip(bars, throughputs):
        height = bar.get_height()
        ax.annotate(f'{tp:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / "token_throughput.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_streaming_metrics_chart(records: list[dict], output_dir: Path) -> str:
    """Generate streaming metrics chart (chunk count and inter-chunk timing)."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)
    provider_colors = _build_provider_palette(records)

    labels = []
    chunk_counts = []
    chunk_rates = []
    colors = []

    for scenario, recs in sorted(by_scenario.items()):
        chunks = []
        rates = []
        for r in recs:
            chunk_count = r.get("chunk_count")
            t_first = r.get("t_first_token")
            t_last = r.get("t_last_token")
            if chunk_count and chunk_count > 0:
                chunks.append(chunk_count)
                if t_first and t_last and t_last > t_first:
                    rates.append(chunk_count / (t_last - t_first))

        if chunks:
            labels.append(format_scenario_label(scenario, recs))
            chunk_counts.append(sum(chunks) / len(chunks))
            chunk_rates.append(sum(rates) / len(rates) if rates else 0)
            colors.append(provider_colors.get(_provider_alias_from_records(recs), "#666666"))

    if not labels:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Chunk counts
    bars1 = ax1.bar(range(len(labels)), chunk_counts, color=colors)
    ax1.set_xlabel('Scenario (Provider)')
    ax1.set_ylabel('Average Chunk Count')
    ax1.set_title('Streaming: Average Chunks per Response')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Chunk rates
    bars2 = ax2.bar(range(len(labels)), chunk_rates, color=colors)
    ax2.set_xlabel('Scenario (Provider)')
    ax2.set_ylabel('Chunks per Second')
    ax2.set_title('Streaming: Chunk Delivery Rate')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "streaming_metrics.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_tool_usage_chart(records: list[dict], output_dir: Path) -> str:
    """Generate tool usage statistics chart."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    scenarios = []
    tool_counts = []
    tool_latencies = []

    for scenario, recs in sorted(by_scenario.items()):
        counts = []
        latencies = []
        for r in recs:
            tc = r.get("tool_calls_count", 0) or 0
            tl = r.get("tool_latency_sec", 0) or 0
            if tc > 0:
                counts.append(tc)
                latencies.append(tl)

        if counts:
            scenarios.append(format_scenario_label(scenario, recs))
            tool_counts.append(sum(counts) / len(counts))
            tool_latencies.append(sum(latencies) / len(latencies))

    if not scenarios:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Tool call counts
    bars1 = ax1.bar(range(len(scenarios)), tool_counts, color='#e67e22')
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Average Tool Calls')
    ax1.set_title('Tool Usage: Calls per Request')
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels(scenarios, fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Tool latencies
    bars2 = ax2.bar(range(len(scenarios)), tool_latencies, color='#9b59b6')
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Tool Latency (seconds)')
    ax2.set_title('Tool Usage: Average Latency')
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels(scenarios, fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "tool_usage.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_error_analysis_chart(records: list[dict], output_dir: Path) -> str:
    """Generate error analysis chart by error type."""
    if not HAS_MATPLOTLIB:
        return None

    # Count errors by type
    error_counts = defaultdict(int)
    success_count = 0

    for r in records:
        if r.get("success"):
            success_count += 1
        else:
            error_type = r.get("error_type", "unknown") or "unknown"
            error_counts[error_type] += 1

    if not error_counts:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart of success vs failure
    total = success_count + sum(error_counts.values())
    fail_count = sum(error_counts.values())
    sizes = [success_count, fail_count]
    labels = [f'Success\n({success_count})', f'Failed\n({fail_count})']
    colors = ['#27ae60', '#e74c3c']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Overall Success Rate (n={total})')

    # Bar chart of error types
    error_types = list(error_counts.keys())
    counts = list(error_counts.values())

    bars = ax2.barh(range(len(error_types)), counts, color='#e74c3c')
    ax2.set_xlabel('Count')
    ax2.set_ylabel('Error Type')
    ax2.set_title('Failures by Error Type')
    ax2.set_yticks(range(len(error_types)))
    ax2.set_yticklabels(error_types)
    ax2.grid(axis='x', alpha=0.3)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax2.annotate(f'{count}',
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "error_analysis.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_latency_by_profile_chart(records: list[dict], output_dir: Path) -> str:
    """Generate latency comparison by network profile."""
    if not HAS_MATPLOTLIB:
        return None

    by_profile = defaultdict(list)
    for r in records:
        profile = r.get("network_profile", "unknown")
        latency = r.get("latency_sec")
        if latency and latency > 0:
            by_profile[profile].append(latency)

    if not by_profile:
        return None

    profiles = sorted(by_profile.keys())
    means = [sum(by_profile[p])/len(by_profile[p]) for p in profiles]
    stds = []
    for p in profiles:
        vals = by_profile[p]
        mean = sum(vals)/len(vals)
        std = (sum((v-mean)**2 for v in vals)/len(vals))**0.5
        stds.append(std)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(profiles))
    yerr_lower = [min(std, mean) for std, mean in zip(stds, means)]
    yerr_upper = stds
    bars = ax.bar(
        x,
        means,
        yerr=[yerr_lower, yerr_upper],
        capsize=5,
        color='#3498db',
        alpha=0.8,
    )

    ax.set_xlabel('Network Profile')
    ax.set_ylabel('Mean Latency (seconds)')
    ax.set_title('Latency by Network Profile (with Std Dev)')
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f'{mean:.1f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "latency_by_profile.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_tokens_chart(records: list[dict], output_dir: Path) -> str:
    """Generate token counts (input vs output) chart."""
    if not HAS_MATPLOTLIB:
        return None

    by_scenario = aggregate_by_scenario(records)

    scenarios = []
    tokens_in = []
    tokens_out = []

    for scenario, recs in sorted(by_scenario.items()):
        tin = [r.get("tokens_in", 0) or 0 for r in recs if r.get("tokens_in")]
        tout = [r.get("tokens_out", 0) or 0 for r in recs if r.get("tokens_out")]

        if tin or tout:
            scenarios.append(format_scenario_label(scenario, recs))
            tokens_in.append(sum(tin)/len(tin) if tin else 0)
            tokens_out.append(sum(tout)/len(tout) if tout else 0)

    if not scenarios:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(scenarios))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], tokens_in, width, label='Input Tokens', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], tokens_out, width, label='Output Tokens', color='#e74c3c')

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Token Count')
    ax.set_title('Average Token Counts by Scenario')
    ax.set_xticks(x)
    label_text = [label.replace(" - ", "\n") for label in scenarios]
    ax.set_xticklabels(label_text, fontsize=7, rotation=55, ha='right')
    ax.tick_params(axis='x', labelrotation=55)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    output_path = output_dir / "token_counts.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_ttft_heatmap(records: list[dict], output_dir: Path) -> str:
    """Generate TTFT heatmap (scenario vs network profile)."""
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return None

    # Group by scenario and profile
    data = defaultdict(lambda: defaultdict(list))
    for r in records:
        scenario = r.get("scenario_id", "unknown")
        profile = r.get("network_profile", "unknown")
        t_start = r.get("t_request_start")
        t_first = r.get("t_first_token")
        if t_start and t_first and t_first > t_start:
            data[scenario][profile].append(t_first - t_start)

    if not data:
        return None

    scenario_providers = _scenario_provider_alias_map(records)

    # Build matrix
    scenarios = sorted(data.keys())
    scenario_labels = [
        format_scenario_label(s, provider=scenario_providers.get(s))
        for s in scenarios
    ]
    profiles = sorted(set(p for s in data.values() for p in s.keys()))

    matrix = []
    for scenario in scenarios:
        row = []
        for profile in profiles:
            ttfts = data[scenario].get(profile, [])
            if ttfts:
                row.append(sum(ttfts) / len(ttfts))
            else:
                row.append(0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
    fig.colorbar(im, ax=ax, label='Mean TTFT (s)')

    ax.set_xticks(range(len(profiles)))
    ax.set_yticks(range(len(scenarios)))
    ax.set_xticklabels(profiles, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(scenario_labels, fontsize=8)
    ax.set_xlabel('Network Profile')
    ax.set_ylabel('Scenario')
    ax.set_title('TTFT Heatmap (Scenario x Profile)')

    # Add value annotations
    for i in range(len(scenarios)):
        for j in range(len(profiles)):
            val = matrix[i][j]
            if val > 0:
                text = f'{val:.2f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=7,
                       color='white' if val > max(max(row) for row in matrix)/2 else 'black')

    plt.tight_layout()
    output_path = output_dir / "ttft_heatmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_pcap_rtt_chart(pcap_metrics: list, output_dir: Path) -> str:
    """Generate RTT distribution chart from pcap TCP handshake analysis."""
    if not HAS_MATPLOTLIB:
        return None

    # Collect all RTT samples
    all_rtt = []
    for m in pcap_metrics:
        all_rtt.extend(m.rtt_samples)

    if not all_rtt:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of RTT values
    ax1.hist(all_rtt, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
    ax1.axvline(sum(all_rtt)/len(all_rtt), color='red', linestyle='--', label=f'Mean: {sum(all_rtt)/len(all_rtt):.1f}ms')
    ax1.set_xlabel('RTT (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('TCP Handshake RTT Distribution (from pcap)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Box plot per capture file
    if len(pcap_metrics) > 1:
        rtt_by_file = []
        labels = []
        for m in pcap_metrics:
            if m.rtt_samples:
                rtt_by_file.append(m.rtt_samples)
                labels.append(Path(m.pcap_file).stem[:20])
        if rtt_by_file:
            bp = ax2.boxplot(rtt_by_file, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#3498db')
                patch.set_alpha(0.7)
            ax2.set_xlabel('Capture File')
            ax2.set_ylabel('RTT (ms)')
            ax2.set_title('RTT by Capture')
            ax2.tick_params(axis='x', rotation=45)
    else:
        # Single file - show percentiles
        sorted_rtt = sorted(all_rtt)
        percentiles = [50, 75, 90, 95, 99]
        pct_values = []
        for p in percentiles:
            idx = int(len(sorted_rtt) * p / 100)
            pct_values.append(sorted_rtt[min(idx, len(sorted_rtt)-1)])
        ax2.bar([f'P{p}' for p in percentiles], pct_values, color='#e74c3c', alpha=0.8)
        ax2.set_xlabel('Percentile')
        ax2.set_ylabel('RTT (ms)')
        ax2.set_title('RTT Percentiles')
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "pcap_rtt_analysis.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_pcap_throughput_chart(pcap_metrics: list, output_dir: Path) -> str:
    """Generate throughput over time chart from pcap analysis."""
    if not HAS_MATPLOTLIB:
        return None

    # Merge all throughput time series
    all_throughput = []
    for m in pcap_metrics:
        all_throughput.extend(m.throughput_timeseries)

    if not all_throughput:
        return None

    # Sort by time and bucket
    all_throughput.sort(key=lambda x: x[0])

    times = [t[0] for t in all_throughput]
    ul_kbps = [t[1] for t in all_throughput]
    dl_kbps = [t[2] for t in all_throughput]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Throughput over time
    ax1.plot(times, ul_kbps, label='Uplink', color='#3498db', linewidth=1, alpha=0.8)
    ax1.plot(times, dl_kbps, label='Downlink', color='#e74c3c', linewidth=1, alpha=0.8)
    ax1.set_ylabel('Throughput (Kbps)')
    ax1.set_title('Network Throughput from Packet Capture')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Cumulative throughput
    cumul_ul = []
    cumul_dl = []
    running_ul = 0
    running_dl = 0
    for ul, dl in zip(ul_kbps, dl_kbps):
        running_ul += ul
        running_dl += dl
        cumul_ul.append(running_ul / 1000)  # Convert to Mbps
        cumul_dl.append(running_dl / 1000)

    ax2.fill_between(times, cumul_ul, alpha=0.5, label='Uplink', color='#3498db')
    ax2.fill_between(times, cumul_dl, alpha=0.5, label='Downlink', color='#e74c3c')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Cumulative (Mbps)')
    ax2.set_title('Cumulative Throughput')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "pcap_throughput.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_pcap_retransmission_chart(pcap_metrics: list, output_dir: Path) -> str:
    """Generate retransmission analysis chart from pcap data."""
    if not HAS_MATPLOTLIB:
        return None

    # Collect retransmission data
    labels = []
    retrans_rates = []
    total_packets = []
    retrans_counts = []

    for m in pcap_metrics:
        if m.tcp_packets > 0:
            labels.append(Path(m.pcap_file).stem[:20])
            retrans_rates.append(m.retransmission_rate * 100)  # Convert to percentage
            total_packets.append(m.tcp_packets)
            retrans_counts.append(m.total_retransmissions)

    if not labels:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Retransmission rate
    bars = ax1.bar(range(len(labels)), retrans_rates, color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Capture')
    ax1.set_ylabel('Retransmission Rate (%)')
    ax1.set_title('TCP Retransmission Rate by Capture')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, retrans_rates):
        if rate > 0:
            ax1.annotate(f'{rate:.2f}%',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Packet counts with retransmissions highlighted
    x = range(len(labels))
    width = 0.35
    ax2.bar([i - width/2 for i in x], total_packets, width, label='Total TCP Packets', color='#3498db', alpha=0.8)
    ax2.bar([i + width/2 for i in x], retrans_counts, width, label='Retransmissions', color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('Capture')
    ax2.set_ylabel('Packet Count')
    ax2.set_title('TCP Packets and Retransmissions')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    output_path = output_dir / "pcap_retransmissions.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def generate_pcap_summary_chart(pcap_metrics: list, output_dir: Path) -> str:
    """Generate summary statistics chart from pcap analysis."""
    if not HAS_MATPLOTLIB:
        return None

    if not pcap_metrics:
        return None

    # Aggregate statistics
    merged = merge_pcap_metrics(pcap_metrics)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Packet protocol breakdown
    ax1 = axes[0, 0]
    tcp_total = sum(m.tcp_packets for m in pcap_metrics)
    udp_total = sum(m.udp_packets for m in pcap_metrics)
    other_total = sum(m.other_packets for m in pcap_metrics)
    sizes = [tcp_total, udp_total, other_total]
    labels = [f'TCP\n({tcp_total:,})', f'UDP\n({udp_total:,})', f'Other\n({other_total:,})']
    colors = ['#3498db', '#2ecc71', '#95a5a6']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Packet Protocol Distribution')

    # 2. Throughput comparison (app-layer vs pcap)
    ax2 = axes[0, 1]
    avg_throughput = [m.avg_throughput_mbps for m in pcap_metrics if m.avg_throughput_mbps > 0]
    peak_throughput = [m.peak_throughput_mbps for m in pcap_metrics if m.peak_throughput_mbps > 0]
    if avg_throughput:
        x = range(len(avg_throughput))
        width = 0.35
        ax2.bar([i - width/2 for i in x], avg_throughput, width, label='Average', color='#3498db')
        ax2.bar([i + width/2 for i in x], peak_throughput, width, label='Peak', color='#e74c3c')
        ax2.set_xlabel('Capture')
        ax2.set_ylabel('Throughput (Mbps)')
        ax2.set_title('Throughput: Average vs Peak')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

    # 3. RTT Summary
    ax3 = axes[1, 0]
    if merged.get('rtt_mean_ms'):
        rtt_stats = ['Min', 'Mean', 'P95', 'Max']
        rtt_values = [
            merged.get('rtt_min_ms', 0),
            merged.get('rtt_mean_ms', 0),
            merged.get('rtt_p95_ms', 0),
            merged.get('rtt_max_ms', 0)
        ]
        bars = ax3.bar(rtt_stats, rtt_values, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
        ax3.set_ylabel('RTT (ms)')
        ax3.set_title('RTT Statistics from TCP Handshakes')
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, rtt_values):
            ax3.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No RTT data available', ha='center', va='center', fontsize=12)
        ax3.set_title('RTT Statistics')

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Network Capture Summary
    ═══════════════════════════════

    Captures Analyzed:  {merged.get('total_captures', 0)}
    Total Duration:     {merged.get('total_duration_sec', 0):.1f} seconds

    Packets:
      Total:            {merged.get('total_packets', 0):,}
      TCP Flows:        {merged.get('total_tcp_flows', 0):,}

    Data Volume:
      Total Bytes:      {merged.get('total_bytes', 0):,}
      Avg Throughput:   {merged.get('avg_throughput_mbps', 0):.2f} Mbps

    Quality Metrics:
      Retransmissions:  {merged.get('total_retransmissions', 0):,}
      Retrans Rate:     {merged.get('retransmission_rate', 0)*100:.3f}%
      RTT Mean:         {merged.get('rtt_mean_ms', 0):.1f} ms
      RTT P95:          {merged.get('rtt_p95_ms', 0):.1f} ms
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    output_path = output_dir / "pcap_summary.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return str(output_path)


def main():
    """Generate all charts and print results."""
    parser = argparse.ArgumentParser(description="Generate visualization charts from traffic test data")
    parser.add_argument("--latest", action="store_true", help="Only include data from the latest test run")
    parser.add_argument("--since-minutes", type=int, help="Only include data from the last N minutes")
    parser.add_argument("--since-timestamp", type=float, help="Only include data after this Unix timestamp")
    parser.add_argument("--db", default="logs/traffic_logs.db", help="Path to SQLite database")
    parser.add_argument("--all-runs", action="store_true", help="Include all runs instead of latest per scenario")
    parser.add_argument("--run-gap-sec", type=float, default=300.0, help="Gap in seconds to split runs per scenario")
    parser.add_argument("--output-dir", default="reports/figures", help="Output directory for charts")
    parser.add_argument("--pcap-dir", default=None, help="Directory containing pcap files for network-layer analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine timestamp filter
    since_timestamp = None
    if args.since_timestamp:
        since_timestamp = args.since_timestamp
        print(f"Filtering to records after timestamp {since_timestamp}")
    elif args.since_minutes:
        import time
        since_timestamp = time.time() - (args.since_minutes * 60)
        print(f"Filtering to records from the last {args.since_minutes} minutes")
    elif args.latest:
        since_timestamp = get_latest_run_timestamp(args.db)
        print(f"Filtering to latest run (since timestamp {since_timestamp:.0f})")

    print("Loading data from SQLite database...")
    records = load_data_from_db(args.db, since_timestamp)
    print(f"Loaded {len(records)} records")

    if not args.all_runs:
        records = fuse_latest_runs(records, gap_sec=float(args.run_gap_sec))
        print(f"Using {len(records)} records from latest run per scenario")

    if not records:
        print("No data found in database!")
        return

    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib seaborn")
        return

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    if HAS_SEABORN:
        sns.set_palette("husl")

    generated = {}

    print("\nGenerating charts...")

    # Generate each chart
    path = generate_latency_by_scenario(records, output_dir)
    if path:
        generated["latency_by_scenario"] = path
        print(f"  ✓ Latency by scenario: {path}")

    path = generate_throughput_chart(records, output_dir)
    if path:
        generated["throughput"] = path
        print(f"  ✓ Throughput: {path}")

    path = generate_throughput_over_time(records, output_dir)
    if path:
        generated["throughput_over_time"] = path
        print(f"  ✓ Throughput over time: {path}")

    path = generate_bandwidth_asymmetry_chart(records, output_dir)
    if path:
        generated["bandwidth_asymmetry"] = path
        print(f"  ✓ Bandwidth asymmetry: {path}")

    path = generate_protocol_comparison_chart(records, output_dir)
    if path:
        generated["protocol_comparison"] = path
        print(f"  ✓ Protocol comparison: {path}")

    path = generate_success_rate_chart(records, output_dir)
    if path:
        generated["success_rate"] = path
        print(f"  ✓ Success rate: {path}")

    path = generate_data_volume_chart(records, output_dir)
    if path:
        generated["data_volume"] = path
        print(f"  ✓ Data volume: {path}")

    path = generate_latency_heatmap(records, output_dir)
    if path:
        generated["latency_heatmap"] = path
        print(f"  ✓ Latency heatmap: {path}")

    # New charts
    path = generate_ttft_chart(records, output_dir)
    if path:
        generated["ttft_by_scenario"] = path
        print(f"  ✓ TTFT by scenario: {path}")

    path = generate_latency_breakdown_chart(records, output_dir)
    if path:
        generated["latency_breakdown"] = path
        print(f"  ✓ Latency breakdown: {path}")

    path = generate_token_throughput_chart(records, output_dir)
    if path:
        generated["token_throughput"] = path
        print(f"  ✓ Token throughput: {path}")

    path = generate_streaming_metrics_chart(records, output_dir)
    if path:
        generated["streaming_metrics"] = path
        print(f"  ✓ Streaming metrics: {path}")

    path = generate_tool_usage_chart(records, output_dir)
    if path:
        generated["tool_usage"] = path
        print(f"  ✓ Tool usage: {path}")

    path = generate_error_analysis_chart(records, output_dir)
    if path:
        generated["error_analysis"] = path
        print(f"  ✓ Error analysis: {path}")

    path = generate_latency_by_profile_chart(records, output_dir)
    if path:
        generated["latency_by_profile"] = path
        print(f"  ✓ Latency by profile: {path}")

    path = generate_tokens_chart(records, output_dir)
    if path:
        generated["token_counts"] = path
        print(f"  ✓ Token counts: {path}")

    path = generate_ttft_heatmap(records, output_dir)
    if path:
        generated["ttft_heatmap"] = path
        print(f"  ✓ TTFT heatmap: {path}")

    # Generate pcap-based network layer charts if pcap directory provided
    pcap_metrics = None
    if args.pcap_dir:
        if HAS_PCAP_ANALYZER:
            print(f"\nAnalyzing pcap files from {args.pcap_dir}...")
            try:
                pcap_metrics = analyze_multiple_pcaps(
                    args.pcap_dir,
                    pattern="*.pcap",
                    target_ports=[443, 80]  # HTTPS and HTTP
                )
                print(f"  Analyzed {len(pcap_metrics)} pcap files")

                if pcap_metrics:
                    print("\nGenerating network-layer charts from pcap...")

                    path = generate_pcap_rtt_chart(pcap_metrics, output_dir)
                    if path:
                        generated["pcap_rtt"] = path
                        print(f"  ✓ Pcap RTT analysis: {path}")

                    path = generate_pcap_throughput_chart(pcap_metrics, output_dir)
                    if path:
                        generated["pcap_throughput"] = path
                        print(f"  ✓ Pcap throughput: {path}")

                    path = generate_pcap_retransmission_chart(pcap_metrics, output_dir)
                    if path:
                        generated["pcap_retransmissions"] = path
                        print(f"  ✓ Pcap retransmissions: {path}")

                    path = generate_pcap_summary_chart(pcap_metrics, output_dir)
                    if path:
                        generated["pcap_summary"] = path
                        print(f"  ✓ Pcap summary: {path}")
            except Exception as e:
                print(f"  ⚠ Pcap analysis failed: {e}")
        else:
            print(f"\n⚠ Pcap analysis requested but dpkt not installed.")
            print("  Install with: pip install dpkt")

    print(f"\n✓ Generated {len(generated)} charts in {output_dir}/")

    # Output JSON summary
    summary_path = output_dir / "charts_summary.json"
    with open(summary_path, "w") as f:
        json.dump(generated, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")

    return generated


if __name__ == "__main__":
    main()
