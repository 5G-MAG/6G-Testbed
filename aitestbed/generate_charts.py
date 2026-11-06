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

# RAN2 methodology metrics (S4-260859)
try:
    from analysis.ran2_metrics import compute_ran2_metrics
    HAS_RAN2 = True
except ImportError:
    HAS_RAN2 = False
    compute_ran2_metrics = None

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

    # Filter out pcap capture summary records and timeout placeholders
    return [
        dict(row) for row in rows
        if not (dict(row).get("session_id") or "").startswith(("pcap_", "timeout_"))
    ]


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
    """Keep only the latest run per scenario+profile based on timestamp gaps.

    Groups by (scenario_id, network_profile) so that profiles run in
    different --resume sessions are all retained.
    """
    if not records:
        return []
    by_key: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        scenario = record.get("scenario_id") or "unknown"
        profile = record.get("network_profile") or "unknown"
        by_key[f"{scenario}/{profile}"].append(record)

    fused: list[dict] = []
    for recs in by_key.values():
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_path)


def generate_throughput_burstiness_chart(records: list[dict], output_dir: Path) -> str:
    """Candlestick-style boxplot of per-request throughput (bytes/s) by scenario.

    Wide boxes indicate bursty traffic; narrow boxes indicate steady throughput.
    """
    output_path = Path(output_dir) / "throughput_burstiness.png"
    tput_by_scenario: dict[str, list[float]] = defaultdict(list)
    recs_by_scenario: dict[str, list[dict]] = defaultdict(list)

    for r in records:
        lat = r.get("latency_sec") or 0.0
        resp = r.get("response_bytes") or 0
        req = r.get("request_bytes") or 0
        if lat > 0:
            tput = (req + resp) / lat  # total bytes/sec
            scenario = r.get("scenario_id", "unknown")
            tput_by_scenario[scenario].append(tput)
            recs_by_scenario[scenario].append(r)

    if not tput_by_scenario:
        return ""

    import numpy as np

    # Sort by median throughput
    sorted_scenarios = sorted(
        tput_by_scenario.items(),
        key=lambda x: np.median(x[1]),
        reverse=True,
    )

    labels = [format_scenario_label(s, recs_by_scenario[s]) for s, _ in sorted_scenarios]
    data = [vals for _, vals in sorted_scenarios]

    fig, ax = plt.subplots(figsize=(16, 8))

    bp = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.4, markerfacecolor='#e74c3c'),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    # Color by burstiness ratio (max/median)
    for i, (patch, vals) in enumerate(zip(bp['boxes'], data)):
        median = np.median(vals)
        p95 = np.percentile(vals, 95)
        ratio = p95 / max(median, 1)
        if ratio > 20:
            color = '#e74c3c'   # Very bursty (red)
        elif ratio > 5:
            color = '#f39c12'   # Moderate burst (orange)
        else:
            color = '#27ae60'   # Steady (green)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yscale('log')
    ax.set_ylabel('Throughput (bytes/sec, log scale)')
    ax.set_title('Traffic Burstiness by Scenario — Per-Request Throughput Distribution')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7, rotation=55, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add legend for color coding
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor='#e74c3c', alpha=0.7, label='Very bursty (P95/median > 20x)'),
            Patch(facecolor='#f39c12', alpha=0.7, label='Moderate burst (5–20x)'),
            Patch(facecolor='#27ae60', alpha=0.7, label='Steady (< 5x)'),
        ],
        loc='upper right',
        fontsize=8,
    )

    # Annotate with burstiness ratio
    for i, vals in enumerate(data):
        median = np.median(vals)
        p95 = np.percentile(vals, 95)
        p5 = np.percentile(vals, 5)
        ratio = p95 / max(p5, 1)
        ax.annotate(
            f'{ratio:.0f}x',
            xy=(i + 1, p95),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=6, color='gray',
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
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

        if total_ul > 0 or total_dl > 0:
            ratio = max(total_ul, 1) / max(total_dl, 1)  # UL:DL ratio
            scenarios.append(format_scenario_label(scenario, recs))
            ratios.append(ratio)
            # Color based on direction
            if ratio > 1:
                colors.append('#e74c3c')  # UL-heavy (red)
            elif ratio > 0.1:
                colors.append('#27ae60')  # Near symmetric (green)
            else:
                colors.append('#3498db')  # DL-heavy (blue)

    if not scenarios:
        return None

    fig, ax = plt.subplots(figsize=(20, 8))

    bars = ax.bar(range(len(scenarios)), ratios, color=colors)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('UL/DL Ratio (log scale)')
    ax.set_title('Bandwidth Asymmetry by Scenario (Uplink / Downlink)')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=8, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Symmetric (1:1)')
    ax.axhline(y=0.1, color='gray', linestyle=':', alpha=0.3, label='DL-heavy (10:1)')
    ax.axhline(y=0.01, color='gray', linestyle=':', alpha=0.3, label='DL-heavy (100:1)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        if ratio >= 1:
            label = f'{ratio:.1f}:1 UL'
        elif ratio > 0:
            label = f'1:{1/ratio:.0f} DL'
        else:
            label = 'DL only'
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()
    output_path = output_dir / "bandwidth_asymmetry.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
        "MCP Agent": [],
    }

    for r in records:
        scenario = r.get("scenario_id", "")
        latency = r.get("latency_sec")
        if not latency or latency <= 0:
            continue

        if any(kw in scenario.lower() for kw in (
            "music", "trading", "playwright", "shopping_agent",
            "web_search_agent", "general_agent", "computer_control",
        )):
            protocols["MCP Agent"].append(latency)
        elif "webrtc" in scenario.lower():
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
    colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c']
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    ax.set_xticklabels(scenarios, fontsize=8, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "data_volume.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    ax.set_title('Observed Token Arrival Rate by Scenario (Client-Side)')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=8, rotation=45, ha='right')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

    # Build legend handles for providers
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color, label=provider)
        for provider, color in provider_colors.items()
    ]

    # Chunk counts
    bars1 = ax1.bar(range(len(labels)), chunk_counts, color=colors)
    ax1.set_xlabel('Scenario (Provider)')
    ax1.set_ylabel('Average Chunk Count')
    ax1.set_title('Streaming: Average Chunks per Response')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(handles=legend_handles, fontsize=7, loc='upper right')

    # Chunk rates
    bars2 = ax2.bar(range(len(labels)), chunk_rates, color=colors)
    ax2.set_xlabel('Scenario (Provider)')
    ax2.set_ylabel('Chunks per Second')
    ax2.set_title('Streaming: Chunk Delivery Rate')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(handles=legend_handles, fontsize=7, loc='upper right')

    plt.tight_layout()
    output_path = output_dir / "streaming_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    ax1.set_xticklabels(scenarios, fontsize=8, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Tool latencies
    bars2 = ax2.bar(range(len(scenarios)), tool_latencies, color='#9b59b6')
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Tool Latency (seconds)')
    ax2.set_title('Tool Usage: Average Latency')
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels(scenarios, fontsize=8, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "tool_usage.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_path)


def _parse_metadata(record: dict) -> dict:
    """Parse metadata JSON string from a record."""
    meta = record.get("metadata")
    if not meta:
        return {}
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            return {}
    return meta


def generate_mcp_efficiency_chart(records: list[dict], output_dir: Path) -> str:
    """Generate MCP efficiency analysis: loop factor, latency overhead, byte overhead.

    Shows per-scenario and per-profile how much overhead the MCP agent loop
    adds compared to the useful final response.
    """
    if not HAS_MATPLOTLIB:
        return None

    # Group records by session to compute per-session efficiency
    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("session_id")
        if sid:
            sessions[sid].append(r)

    # Compute per-session metrics
    scenario_data: dict[str, list[dict]] = defaultdict(list)
    for sid, recs in sessions.items():
        llm_records = []
        tool_records = []
        for r in recs:
            meta = _parse_metadata(r)
            if meta.get("type") == "mcp_tool_call":
                tool_records.append(r)
            elif meta.get("type") == "llm_api_call" or (r.get("tokens_out") or 0) > 0:
                llm_records.append(r)

        if not llm_records:
            continue

        scenario = recs[0].get("scenario_id", "unknown")
        profile = recs[0].get("network_profile", "unknown")

        total_llm_latency = sum(r.get("latency_sec", 0) or 0 for r in llm_records)
        total_tool_latency = sum(r.get("tool_latency_sec", 0) or 0 for r in tool_records)
        total_latency = total_llm_latency + total_tool_latency

        total_llm_bytes = sum(
            (r.get("request_bytes", 0) or 0) + (r.get("response_bytes", 0) or 0)
            for r in llm_records
        )
        total_tool_bytes = sum(
            (r.get("request_bytes", 0) or 0) + (r.get("response_bytes", 0) or 0)
            for r in tool_records
        )
        total_bytes = total_llm_bytes + total_tool_bytes

        # Final response = last LLM call (the one that produces the answer)
        final_llm = llm_records[-1]
        final_latency = final_llm.get("latency_sec", 0) or 0
        final_bytes = (final_llm.get("response_bytes", 0) or 0)

        api_calls = len(llm_records)
        tool_calls = len(tool_records)

        scenario_data[scenario].append({
            "profile": profile,
            "loop_factor": api_calls + tool_calls,
            "api_calls": api_calls,
            "tool_calls": tool_calls,
            "total_latency": total_latency,
            "final_latency": final_latency,
            "latency_overhead_pct": ((total_latency - final_latency) / total_latency * 100)
                if total_latency > 0 else 0,
            "total_bytes": total_bytes,
            "final_bytes": final_bytes,
            "byte_overhead_pct": ((total_bytes - final_bytes) / total_bytes * 100)
                if total_bytes > 0 else 0,
            "tool_latency_pct": (total_tool_latency / total_latency * 100)
                if total_latency > 0 else 0,
            "tool_bytes_pct": (total_tool_bytes / total_bytes * 100)
                if total_bytes > 0 else 0,
        })

    if not scenario_data:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("MCP Agent Efficiency Analysis", fontsize=14, fontweight="bold")

    scenarios = sorted(scenario_data.keys())
    labels = [format_scenario_label(s, scenario_data[s]) for s in scenarios]
    x = range(len(scenarios))

    # --- Chart 1: Loop Factor (API calls + tool calls per session) ---
    ax = axes[0, 0]
    api_means = []
    tool_means = []
    for s in scenarios:
        data = scenario_data[s]
        api_means.append(sum(d["api_calls"] for d in data) / len(data))
        tool_means.append(sum(d["tool_calls"] for d in data) / len(data))

    bars1 = ax.bar(x, api_means, label="LLM API Calls", color="#3498db")
    bars2 = ax.bar(x, tool_means, bottom=api_means, label="MCP Tool Calls", color="#e67e22")
    ax.set_ylabel("Calls per Session")
    ax.set_title("Loop Factor: Calls per User Prompt")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, (a, t) in enumerate(zip(api_means, tool_means)):
        ax.text(i, a + t + 0.1, f"{a + t:.1f}", ha="center", fontsize=8)

    # --- Chart 2: Latency Overhead (%) ---
    ax = axes[0, 1]
    overhead_means = []
    tool_lat_means = []
    for s in scenarios:
        data = scenario_data[s]
        overhead_means.append(sum(d["latency_overhead_pct"] for d in data) / len(data))
        tool_lat_means.append(sum(d["tool_latency_pct"] for d in data) / len(data))

    bar_width = 0.35
    x_arr = list(x)
    ax.bar([i - bar_width / 2 for i in x_arr], overhead_means,
           bar_width, label="Total Overhead", color="#e74c3c")
    ax.bar([i + bar_width / 2 for i in x_arr], tool_lat_means,
           bar_width, label="Tool Latency Share", color="#9b59b6")
    ax.set_ylabel("Percentage of Total Latency (%)")
    ax.set_title("Latency Overhead vs Final Response")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # --- Chart 3: Byte Overhead (%) ---
    ax = axes[1, 0]
    byte_overhead_means = []
    tool_byte_means = []
    for s in scenarios:
        data = scenario_data[s]
        byte_overhead_means.append(sum(d["byte_overhead_pct"] for d in data) / len(data))
        tool_byte_means.append(sum(d["tool_bytes_pct"] for d in data) / len(data))

    ax.bar([i - bar_width / 2 for i in x_arr], byte_overhead_means,
           bar_width, label="Total Overhead", color="#e74c3c")
    ax.bar([i + bar_width / 2 for i in x_arr], tool_byte_means,
           bar_width, label="MCP Tool Traffic Share", color="#9b59b6")
    ax.set_ylabel("Percentage of Total Bytes (%)")
    ax.set_title("Traffic Overhead vs Final Response")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # --- Chart 4: Overhead by Network Profile ---
    ax = axes[1, 1]
    # Aggregate all agent sessions by profile
    profile_data: dict[str, list[dict]] = defaultdict(list)
    for s in scenarios:
        for d in scenario_data[s]:
            profile_data[d["profile"]].append(d)

    if profile_data:
        profiles = sorted(profile_data.keys())
        p_labels = profiles
        p_x = range(len(profiles))
        p_overhead = [
            sum(d["latency_overhead_pct"] for d in profile_data[p]) / len(profile_data[p])
            for p in profiles
        ]
        p_loop = [
            sum(d["loop_factor"] for d in profile_data[p]) / len(profile_data[p])
            for p in profiles
        ]

        ax2 = ax.twinx()
        bars_o = ax.bar([i - bar_width / 2 for i in range(len(profiles))],
                        p_overhead, bar_width, label="Latency Overhead %", color="#e74c3c", alpha=0.8)
        bars_l = ax2.bar([i + bar_width / 2 for i in range(len(profiles))],
                         p_loop, bar_width, label="Loop Factor", color="#3498db", alpha=0.8)
        ax.set_ylabel("Latency Overhead (%)", color="#e74c3c")
        ax2.set_ylabel("Loop Factor (calls/session)", color="#3498db")
        ax.set_title("MCP Overhead by Network Profile")
        ax.set_xticks(range(len(profiles)))
        ax.set_xticklabels(p_labels, fontsize=7, rotation=45, ha="right")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "mcp_efficiency.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_path)


def generate_mcp_latency_breakdown_chart(records: list[dict], output_dir: Path) -> str:
    """Generate stacked bar chart breaking down where time is spent in agent sessions.

    Shows per-profile: LLM inference time vs MCP tool time vs MCP protocol overhead.
    """
    if not HAS_MATPLOTLIB:
        return None

    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("session_id")
        if sid:
            sessions[sid].append(r)

    profile_breakdown: dict[str, dict] = defaultdict(lambda: {
        "llm_latency": [], "tool_latency": [], "total_latency": []
    })

    for sid, recs in sessions.items():
        profile = recs[0].get("network_profile", "unknown")
        total_llm = 0.0
        total_tool = 0.0

        for r in recs:
            meta = _parse_metadata(r)
            if meta.get("type") == "mcp_tool_call":
                total_tool += r.get("tool_latency_sec", 0) or 0
            else:
                total_llm += r.get("latency_sec", 0) or 0

        total = total_llm + total_tool
        if total > 0:
            profile_breakdown[profile]["llm_latency"].append(total_llm)
            profile_breakdown[profile]["tool_latency"].append(total_tool)
            profile_breakdown[profile]["total_latency"].append(total)

    if not profile_breakdown:
        return None

    profiles = sorted(profile_breakdown.keys())
    llm_means = [sum(profile_breakdown[p]["llm_latency"]) / len(profile_breakdown[p]["llm_latency"])
                 for p in profiles]
    tool_means = [sum(profile_breakdown[p]["tool_latency"]) / len(profile_breakdown[p]["tool_latency"])
                  for p in profiles]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(profiles))
    ax.bar(x, llm_means, label="LLM Inference", color="#3498db")
    ax.bar(x, tool_means, bottom=llm_means, label="MCP Tool Execution", color="#e67e22")

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Average Session Latency (seconds)")
    ax.set_title("MCP Agent Latency Breakdown by Network Profile")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, fontsize=8, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (l, t) in enumerate(zip(llm_means, tool_means)):
        total = l + t
        ax.text(i, total + 0.1, f"{total:.1f}s", ha="center", fontsize=8)

    plt.tight_layout()
    output_path = output_dir / "mcp_latency_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_path)


def generate_mcp_loop_factor_by_profile_chart(records: list[dict], output_dir: Path) -> str:
    """Generate loop factor distribution across network profiles.

    Shows how network conditions affect the number of agent iterations
    (retries, additional tool calls due to timeouts, etc.).
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return None

    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("session_id")
        if sid:
            sessions[sid].append(r)

    # Collect loop factors per profile per scenario
    data: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for sid, recs in sessions.items():
        profile = recs[0].get("network_profile", "unknown")
        scenario = recs[0].get("scenario_id", "unknown")
        n_calls = len(recs)
        data[scenario][profile].append(n_calls)

    if not data:
        return None

    scenarios = sorted(data.keys())
    all_profiles = sorted({p for s in data.values() for p in s.keys()})

    fig, ax = plt.subplots(figsize=(14, 6))

    n_scenarios = len(scenarios)
    n_profiles = len(all_profiles)
    bar_width = 0.8 / n_scenarios
    colors = plt.cm.Set2(np.linspace(0, 1, n_scenarios))

    for i, scenario in enumerate(scenarios):
        means = []
        stds = []
        for p in all_profiles:
            vals = data[scenario].get(p, [])
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        positions = [j + i * bar_width for j in range(n_profiles)]
        label = format_scenario_label(scenario, records)
        ax.bar(positions, means, bar_width, yerr=stds, label=label,
               color=colors[i], capsize=3, alpha=0.85)

    ax.set_xlabel("Network Profile")
    ax.set_ylabel("Loop Factor (calls per session)")
    ax.set_title("MCP Loop Factor by Network Profile and Scenario")
    center_offset = (n_scenarios - 1) * bar_width / 2
    ax.set_xticks([j + center_offset for j in range(n_profiles)])
    ax.set_xticklabels(all_profiles, fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "mcp_loop_factor_by_profile.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_path)


# =========================================================================
# MCP & LLM Performance Charts
# =========================================================================


def generate_tool_latency_cdf_chart(records: list[dict], output_dir: Path) -> str:
    """Generate CDF of tool call latency per tool name."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return None

    tool_latencies: dict[str, list[float]] = defaultdict(list)
    for r in records:
        meta = _parse_metadata(r)
        if meta.get("type") == "mcp_tool_call":
            lat = r.get("tool_latency_sec", 0) or 0
            name = meta.get("tool_name", "unknown")
            if lat > 0:
                tool_latencies[name].append(lat)

    if not tool_latencies:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(tool_latencies), 10)))

    for i, (name, lats) in enumerate(sorted(tool_latencies.items())):
        sorted_lats = np.sort(lats)
        cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
        ax.plot(sorted_lats * 1000, cdf, label=f"{name} (n={len(lats)})",
                color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Tool Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("MCP Tool Call Latency CDF by Tool")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="P95")
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    output_path = output_dir / "tool_latency_cdf.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_mcp_protocol_overhead_chart(records: list[dict], output_dir: Path) -> str:
    """Compare MCP stdio bytes vs backend HTTP bytes per tool.

    Shows how much the MCP JSON-RPC framing adds on top of the actual
    backend HTTP traffic to Spotify/Brave/etc.
    """
    if not HAS_MATPLOTLIB:
        return None

    tool_data: dict[str, dict] = defaultdict(lambda: {
        "stdio_bytes": [], "backend_bytes": []
    })

    for r in records:
        meta = _parse_metadata(r)
        if meta.get("type") != "mcp_tool_call":
            continue
        name = meta.get("tool_name", "unknown")
        stdio = (meta.get("mcp_stdio_request_bytes", 0) or 0) + \
                (meta.get("mcp_stdio_response_bytes", 0) or 0)
        backend = (meta.get("backend_total_request_bytes", 0) or 0) + \
                  (meta.get("backend_total_response_bytes", 0) or 0)
        if stdio > 0 or backend > 0:
            tool_data[name]["stdio_bytes"].append(stdio)
            tool_data[name]["backend_bytes"].append(backend)

    if not tool_data:
        return None

    tools = sorted(tool_data.keys())
    stdio_means = [np.mean(tool_data[t]["stdio_bytes"]) / 1024 for t in tools]
    backend_means = [np.mean(tool_data[t]["backend_bytes"]) / 1024 for t in tools]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = range(len(tools))
    bar_width = 0.35
    ax1.bar([i - bar_width / 2 for i in x], stdio_means, bar_width,
            label="MCP stdio (JSON-RPC)", color="#3498db")
    ax1.bar([i + bar_width / 2 for i in x], backend_means, bar_width,
            label="Backend HTTP", color="#e67e22")
    ax1.set_xlabel("Tool")
    ax1.set_ylabel("Average Bytes (KB)")
    ax1.set_title("MCP Protocol vs Backend Traffic per Tool")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tools, fontsize=7, rotation=45, ha="right")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Overhead ratio
    overhead_pct = []
    for s, b in zip(stdio_means, backend_means):
        if b > 0:
            overhead_pct.append(((s - b) / b) * 100)
        else:
            overhead_pct.append(0)

    bars = ax2.bar(x, overhead_pct, color=["#e74c3c" if v > 0 else "#2ecc71" for v in overhead_pct])
    ax2.set_xlabel("Tool")
    ax2.set_ylabel("MCP Overhead (%)")
    ax2.set_title("MCP Protocol Overhead vs Backend")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tools, fontsize=7, rotation=45, ha="right")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "mcp_protocol_overhead.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_tool_success_rate_chart(records: list[dict], output_dir: Path) -> str:
    """Generate success/failure rate per MCP tool name."""
    if not HAS_MATPLOTLIB:
        return None

    tool_stats: dict[str, dict] = defaultdict(lambda: {
        "success": 0, "fail": 0, "rate_limited": 0, "timeout": 0
    })

    for r in records:
        meta = _parse_metadata(r)
        if meta.get("type") != "mcp_tool_call":
            continue
        name = meta.get("tool_name", "unknown")
        if r.get("success"):
            tool_stats[name]["success"] += 1
        else:
            tool_stats[name]["fail"] += 1
            err = r.get("error_type", "")
            if "rate" in str(err).lower():
                tool_stats[name]["rate_limited"] += 1
            elif "timeout" in str(err).lower():
                tool_stats[name]["timeout"] += 1

    if not tool_stats:
        return None

    tools = sorted(tool_stats.keys())
    successes = [tool_stats[t]["success"] for t in tools]
    fails = [tool_stats[t]["fail"] for t in tools]
    totals = [s + f for s, f in zip(successes, fails)]
    success_pcts = [(s / t * 100) if t > 0 else 0 for s, t in zip(successes, totals)]
    rate_limited = [tool_stats[t]["rate_limited"] for t in tools]
    timeouts = [tool_stats[t]["timeout"] for t in tools]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71" if p >= 95 else "#f39c12" if p >= 80 else "#e74c3c" for p in success_pcts]
    ax1.bar(range(len(tools)), success_pcts, color=colors)
    ax1.set_xlabel("Tool")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("MCP Tool Success Rate")
    ax1.set_xticks(range(len(tools)))
    ax1.set_xticklabels(tools, fontsize=7, rotation=45, ha="right")
    ax1.set_ylim(0, 105)
    ax1.axhline(y=95, color="green", linestyle="--", alpha=0.3)
    ax1.grid(axis="y", alpha=0.3)
    for i, (p, t) in enumerate(zip(success_pcts, totals)):
        ax1.text(i, p + 1, f"{p:.0f}%\n(n={t})", ha="center", fontsize=7)

    x = range(len(tools))
    other_fails = [f - rl - to for f, rl, to in zip(fails, rate_limited, timeouts)]
    ax2.bar(x, rate_limited, label="Rate Limited", color="#f39c12")
    ax2.bar(x, timeouts, bottom=rate_limited, label="Timeout", color="#e74c3c")
    ax2.bar(x, other_fails, bottom=[rl + to for rl, to in zip(rate_limited, timeouts)],
            label="Other", color="#95a5a6")
    ax2.set_xlabel("Tool")
    ax2.set_ylabel("Failure Count")
    ax2.set_title("MCP Tool Failure Breakdown")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tools, fontsize=7, rotation=45, ha="right")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "tool_success_rate.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_agent_session_waterfall_chart(records: list[dict], output_dir: Path) -> str:
    """Generate waterfall timeline of a sample agent session.

    Picks a representative session and plots each LLM call and tool call
    as horizontal bars on a timeline.
    """
    if not HAS_MATPLOTLIB:
        return None

    # Group by session, pick sessions with tool calls
    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("session_id")
        if sid:
            sessions[sid].append(r)

    # Find a session with mixed LLM + tool calls
    candidate = None
    for sid, recs in sessions.items():
        has_tool = any(_parse_metadata(r).get("type") == "mcp_tool_call" for r in recs)
        has_llm = any(_parse_metadata(r).get("type") == "llm_api_call" for r in recs)
        if has_tool and has_llm and len(recs) >= 3:
            candidate = (sid, recs)
            break

    if not candidate:
        return None

    sid, recs = candidate
    recs.sort(key=lambda r: r.get("t_request_start", 0) or r.get("timestamp", 0))

    # Find session start time
    t0 = min(r.get("t_request_start", 0) or r.get("timestamp", 0) for r in recs)

    fig, ax = plt.subplots(figsize=(14, max(4, len(recs) * 0.5)))

    colors_map = {"llm_api_call": "#3498db", "mcp_tool_call": "#e67e22"}
    y_labels = []

    for i, r in enumerate(recs):
        meta = _parse_metadata(r)
        rec_type = meta.get("type", "llm_api_call")
        start = (r.get("t_request_start", 0) or r.get("timestamp", 0)) - t0
        duration = r.get("latency_sec", 0) or r.get("tool_latency_sec", 0) or 0

        color = colors_map.get(rec_type, "#95a5a6")
        ax.barh(i, duration, left=start, height=0.6, color=color, alpha=0.85, edgecolor="white")

        if rec_type == "mcp_tool_call":
            tool_name = meta.get("tool_name", "tool")
            y_labels.append(f"Tool: {tool_name}")
            ax.text(start + duration + 0.05, i, f"{duration:.2f}s", va="center", fontsize=7)
        else:
            iteration = meta.get("iteration", "?")
            y_labels.append(f"LLM call #{iteration}")
            tokens = r.get("tokens_out", 0) or 0
            ax.text(start + duration + 0.05, i, f"{duration:.2f}s ({tokens} tok)",
                    va="center", fontsize=7)

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Time (seconds from session start)")
    scenario = recs[0].get("scenario_id", "unknown")
    profile = recs[0].get("network_profile", "unknown")
    ax.set_title(f"Agent Session Waterfall — {scenario} / {profile}")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="LLM API Call"),
        Patch(facecolor="#e67e22", label="MCP Tool Call"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    output_path = output_dir / "agent_session_waterfall.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_all_agent_waterfall_charts(records: list[dict], output_dir: Path) -> dict[str, str]:
    """Generate one waterfall chart per agent scenario.

    Returns a dict mapping scenario_id to the chart file path.
    """
    if not HAS_MATPLOTLIB:
        return {}

    from matplotlib.patches import Patch

    # Group all records by session
    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("session_id")
        if sid:
            sessions[sid].append(r)

    # Find best session per scenario (has tool calls, prefer ideal_6g, latest)
    best_per_scenario: dict[str, tuple[str, list[dict]]] = {}
    for sid, recs in sessions.items():
        has_tool = any(_parse_metadata(r).get("type") == "mcp_tool_call" for r in recs)
        has_llm = any(_parse_metadata(r).get("type") == "llm_api_call" for r in recs)
        if not (has_tool and has_llm and len(recs) >= 3):
            continue

        scenario = recs[0].get("scenario_id", "unknown")
        profile = recs[0].get("network_profile", "")
        max_ts = max(r.get("timestamp", 0) for r in recs)

        prev = best_per_scenario.get(scenario)
        if prev is None:
            best_per_scenario[scenario] = (sid, recs)
        else:
            prev_profile = prev[1][0].get("network_profile", "")
            prev_ts = max(r.get("timestamp", 0) for r in prev[1])
            # Prefer ideal_6g, then latest
            if profile == "ideal_6g" and prev_profile != "ideal_6g":
                best_per_scenario[scenario] = (sid, recs)
            elif profile == prev_profile and max_ts > prev_ts:
                best_per_scenario[scenario] = (sid, recs)

    waterfall_dir = output_dir / "waterfalls"
    waterfall_dir.mkdir(parents=True, exist_ok=True)

    generated = {}
    colors_map = {"llm_api_call": "#3498db", "mcp_tool_call": "#e67e22"}

    for scenario, (sid, recs) in sorted(best_per_scenario.items()):
        recs.sort(key=lambda r: r.get("t_request_start", 0) or r.get("timestamp", 0))
        t0 = min(r.get("t_request_start", 0) or r.get("timestamp", 0) for r in recs)
        profile = recs[0].get("network_profile", "unknown")

        fig, ax = plt.subplots(figsize=(14, max(4, len(recs) * 0.45)))

        y_labels = []
        for i, r in enumerate(recs):
            meta = _parse_metadata(r)
            rec_type = meta.get("type", "llm_api_call")
            start = (r.get("t_request_start", 0) or r.get("timestamp", 0)) - t0
            duration = r.get("latency_sec", 0) or r.get("tool_latency_sec", 0) or 0

            color = colors_map.get(rec_type, "#95a5a6")
            ax.barh(i, duration, left=start, height=0.6, color=color, alpha=0.85, edgecolor="white")

            if rec_type == "mcp_tool_call":
                tool_name = meta.get("tool_name", "tool")
                y_labels.append(f"Tool: {tool_name}")
                ax.text(start + duration + 0.05, i, f"{duration:.2f}s", va="center", fontsize=7)
            else:
                iteration = meta.get("iteration", "?")
                y_labels.append(f"LLM #{iteration}")
                tokens = r.get("tokens_out", 0) or 0
                ax.text(start + duration + 0.05, i, f"{duration:.2f}s ({tokens} tok)",
                        va="center", fontsize=7)

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Time (seconds from session start)")

        scenario_label = format_scenario_label(scenario, recs, include_provider=True)
        ax.set_title(f"Agent Session Waterfall — {scenario_label} / {profile}")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        legend_elements = [
            Patch(facecolor="#3498db", label="LLM API Call"),
            Patch(facecolor="#e67e22", label="MCP Tool Call"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

        plt.tight_layout()
        output_path = waterfall_dir / f"waterfall_{scenario}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated[scenario] = str(output_path)

    return generated


def generate_token_rate_by_profile_chart(records: list[dict], output_dir: Path) -> str:
    """Generate token generation rate (tokens/sec) across network profiles."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return None

    profile_rates: dict[str, list[float]] = defaultdict(list)
    for r in records:
        meta = _parse_metadata(r)
        if meta.get("type") == "mcp_tool_call":
            continue
        tokens_out = r.get("tokens_out", 0) or 0
        latency = r.get("latency_sec", 0) or 0
        if tokens_out > 0 and latency > 0:
            profile = r.get("network_profile", "unknown")
            profile_rates[profile].append(tokens_out / latency)

    if not profile_rates:
        return None

    profiles = sorted(profile_rates.keys())
    means = [np.mean(profile_rates[p]) for p in profiles]
    stds = [np.std(profile_rates[p]) for p in profiles]
    p95s = [np.percentile(profile_rates[p], 95) for p in profiles]
    p5s = [np.percentile(profile_rates[p], 5) for p in profiles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(profiles))
    ax1.bar(x, means, yerr=stds, color="#3498db", capsize=4, alpha=0.85)
    ax1.set_xlabel("Network Profile")
    ax1.set_ylabel("Observed Token Rate (tokens/sec)")
    ax1.set_title("Observed Token Arrival Rate by Network Profile (Client-Side)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(profiles, fontsize=8, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    # Box plot
    box_data = [profile_rates[p] for p in profiles]
    bp = ax2.boxplot(box_data, labels=profiles, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)
    ax2.set_xlabel("Network Profile")
    ax2.set_ylabel("Observed Token Rate (tokens/sec)")
    ax2.set_title("Token Arrival Rate Distribution")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "token_rate_by_profile.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_context_growth_chart(records: list[dict], output_dir: Path) -> str:
    """Show how context (tokens_in) grows across agent turns within sessions."""
    if not HAS_MATPLOTLIB:
        return None

    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("session_id")
        if sid:
            sessions[sid].append(r)

    scenario_turns: dict[str, list[list[int]]] = defaultdict(list)
    for sid, recs in sessions.items():
        llm_recs = [r for r in recs if _parse_metadata(r).get("type") != "mcp_tool_call"]
        llm_recs.sort(key=lambda r: r.get("t_request_start", 0) or r.get("timestamp", 0))
        if len(llm_recs) < 2:
            continue
        tokens_per_turn = [r.get("tokens_in", 0) or 0 for r in llm_recs]
        scenario = recs[0].get("scenario_id", "unknown")
        scenario_turns[scenario].append(tokens_per_turn)

    if not scenario_turns:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(scenario_turns), 10)))

    for i, (scenario, all_turns) in enumerate(sorted(scenario_turns.items())):
        max_len = max(len(t) for t in all_turns)
        avg_per_turn = []
        for turn_idx in range(max_len):
            vals = [t[turn_idx] for t in all_turns if turn_idx < len(t)]
            avg_per_turn.append(np.mean(vals) if vals else 0)

        label = format_scenario_label(scenario)
        ax.plot(range(1, len(avg_per_turn) + 1), avg_per_turn,
                marker="o", label=label, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("LLM Call # (within session)")
    ax.set_ylabel("Input Tokens (context size)")
    ax.set_title("Context Window Growth Across Agent Turns")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "context_growth.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_request_response_scatter_chart(records: list[dict], output_dir: Path) -> str:
    """Scatter plot of request bytes vs response bytes per record."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return None

    llm_req, llm_resp = [], []
    tool_req, tool_resp = [], []

    for r in records:
        req = r.get("request_bytes", 0) or 0
        resp = r.get("response_bytes", 0) or 0
        if req <= 0 and resp <= 0:
            continue
        meta = _parse_metadata(r)
        if meta.get("type") == "mcp_tool_call":
            tool_req.append(req / 1024)
            tool_resp.append(resp / 1024)
        else:
            llm_req.append(req / 1024)
            llm_resp.append(resp / 1024)

    if not llm_req and not tool_req:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    if llm_req:
        ax.scatter(llm_req, llm_resp, alpha=0.5, s=30, color="#3498db", label="LLM API", zorder=2)
    if tool_req:
        ax.scatter(tool_req, tool_resp, alpha=0.5, s=30, color="#e67e22", label="MCP Tool", zorder=2)

    all_vals = llm_req + llm_resp + tool_req + tool_resp
    if all_vals:
        max_val = max(all_vals) * 1.1
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.2, label="1:1 ratio")

    ax.set_xlabel("Request Size (KB)")
    ax.set_ylabel("Response Size (KB)")
    ax.set_title("Request vs Response Size (UL/DL Asymmetry)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "request_response_scatter.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_ttft_vs_latency_chart(records: list[dict], output_dir: Path) -> str:
    """Scatter plot of TTFT vs total latency to see correlation."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return None

    ttfts, latencies, profiles = [], [], []
    for r in records:
        t_start = r.get("t_request_start", 0) or 0
        t_first = r.get("t_first_token", 0) or 0
        lat = r.get("latency_sec", 0) or 0
        if t_start > 0 and t_first > 0 and lat > 0:
            ttft = t_first - t_start
            if 0 < ttft < lat:
                ttfts.append(ttft)
                latencies.append(lat)
                profiles.append(r.get("network_profile", "unknown"))

    if len(ttfts) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    unique_profiles = sorted(set(profiles))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_profiles), 10)))
    profile_colors = {p: colors[i % len(colors)] for i, p in enumerate(unique_profiles)}

    for p in unique_profiles:
        p_ttfts = [t for t, pr in zip(ttfts, profiles) if pr == p]
        p_lats = [l for l, pr in zip(latencies, profiles) if pr == p]
        ax.scatter(p_ttfts, p_lats, alpha=0.6, s=30, color=profile_colors[p],
                   label=p, zorder=2)

    # Correlation line
    ttft_arr = np.array(ttfts)
    lat_arr = np.array(latencies)
    if len(ttft_arr) > 2:
        z = np.polyfit(ttft_arr, lat_arr, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min(ttfts), max(ttfts), 100)
        ax.plot(x_line, p_line(x_line), "r--", alpha=0.5, linewidth=2)
        corr = np.corrcoef(ttft_arr, lat_arr)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Time to First Token (seconds)")
    ax.set_ylabel("Total Latency (seconds)")
    ax.set_title("TTFT vs Total Latency Correlation")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "ttft_vs_latency.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_inter_turn_idle_chart(records: list[dict], output_dir: Path) -> str:
    """Show idle time between consecutive calls within agent sessions.

    Reveals client-side processing overhead between LLM response and next request.
    """
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return None

    sessions: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        sid = r.get("session_id")
        if sid:
            sessions[sid].append(r)

    profile_gaps: dict[str, list[float]] = defaultdict(list)
    for sid, recs in sessions.items():
        recs.sort(key=lambda r: r.get("t_request_start", 0) or r.get("timestamp", 0))
        profile = recs[0].get("network_profile", "unknown")

        for i in range(1, len(recs)):
            prev_end = (recs[i - 1].get("t_request_start", 0) or 0) + \
                       (recs[i - 1].get("latency_sec", 0) or recs[i - 1].get("tool_latency_sec", 0) or 0)
            curr_start = recs[i].get("t_request_start", 0) or recs[i].get("timestamp", 0)
            gap = curr_start - prev_end
            if 0 < gap < 30:  # Filter out unreasonable gaps
                profile_gaps[profile].append(gap * 1000)  # Convert to ms

    if not profile_gaps:
        return None

    profiles = sorted(profile_gaps.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of mean idle time
    means = [np.mean(profile_gaps[p]) for p in profiles]
    p95s = [np.percentile(profile_gaps[p], 95) for p in profiles]
    x = range(len(profiles))
    bar_width = 0.35
    ax1.bar([i - bar_width / 2 for i in x], means, bar_width, label="Mean", color="#3498db")
    ax1.bar([i + bar_width / 2 for i in x], p95s, bar_width, label="P95", color="#e74c3c")
    ax1.set_xlabel("Network Profile")
    ax1.set_ylabel("Idle Time (ms)")
    ax1.set_title("Inter-Turn Idle Time (Client Processing)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(profiles, fontsize=8, rotation=45, ha="right")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Box plot
    box_data = [profile_gaps[p] for p in profiles]
    bp = ax2.boxplot(box_data, labels=profiles, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)
    ax2.set_xlabel("Network Profile")
    ax2.set_ylabel("Idle Time (ms)")
    ax2.set_title("Inter-Turn Idle Time Distribution")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "inter_turn_idle.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_degradation_heatmap_chart(records: list[dict], output_dir: Path) -> str:
    """Heatmap: scenarios x profiles, color = % degradation from best profile."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        return None

    # Compute mean latency per scenario+profile
    sp_latencies: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        lat = r.get("latency_sec", 0) or 0
        if lat > 0:
            scenario = r.get("scenario_id", "unknown")
            profile = r.get("network_profile", "unknown")
            sp_latencies[scenario][profile].append(lat)

    if not sp_latencies:
        return None

    scenarios = sorted(sp_latencies.keys())
    all_profiles = sorted({p for s in sp_latencies.values() for p in s.keys()})

    if len(scenarios) < 1 or len(all_profiles) < 2:
        return None

    # Build matrix: % increase from baseline (min latency profile)
    matrix = np.zeros((len(scenarios), len(all_profiles)))
    for i, scenario in enumerate(scenarios):
        profile_means = {}
        for j, profile in enumerate(all_profiles):
            lats = sp_latencies[scenario].get(profile, [])
            profile_means[profile] = np.mean(lats) if lats else np.nan
        baseline = np.nanmin(list(profile_means.values()))
        for j, profile in enumerate(all_profiles):
            val = profile_means.get(profile, np.nan)
            if baseline > 0 and not np.isnan(val):
                matrix[i, j] = ((val - baseline) / baseline) * 100
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(max(10, len(all_profiles) * 1.5),
                                    max(5, len(scenarios) * 0.6)))

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color="lightgray")
    im = ax.imshow(masked, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(all_profiles)))
    ax.set_xticklabels(all_profiles, fontsize=8, rotation=45, ha="right")
    scenario_labels = [format_scenario_label(s) for s in scenarios]
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenario_labels, fontsize=8)
    ax.set_title("Latency Degradation from Best Profile (%)")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Degradation (%)")

    for i in range(len(scenarios)):
        for j in range(len(all_profiles)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, color=color)

    plt.tight_layout()
    output_path = output_dir / "degradation_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_success_vs_latency_chart(records: list[dict], output_dir: Path) -> str:
    """Scatter of success rate vs mean latency per scenario+profile bucket."""
    if not HAS_MATPLOTLIB:
        return None

    buckets: dict[str, dict] = defaultdict(lambda: {"success": 0, "total": 0, "latencies": []})
    for r in records:
        scenario = r.get("scenario_id", "unknown")
        profile = r.get("network_profile", "unknown")
        key = f"{scenario}|{profile}"
        buckets[key]["total"] += 1
        if r.get("success"):
            buckets[key]["success"] += 1
        lat = r.get("latency_sec", 0) or 0
        if lat > 0:
            buckets[key]["latencies"].append(lat)
        buckets[key]["profile"] = profile
        buckets[key]["scenario"] = scenario

    if not buckets:
        return None

    fig, ax = plt.subplots(figsize=(12, 7))

    unique_profiles = sorted({b["profile"] for b in buckets.values()})
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_profiles), 10)))
    profile_colors = {p: colors[i % len(colors)] for i, p in enumerate(unique_profiles)}

    for key, b in buckets.items():
        if not b["latencies"]:
            continue
        mean_lat = np.mean(b["latencies"])
        success_rate = (b["success"] / b["total"] * 100) if b["total"] > 0 else 0
        ax.scatter(mean_lat, success_rate, s=b["total"] * 3,
                   color=profile_colors[b["profile"]], alpha=0.7, edgecolors="white",
                   linewidth=0.5, zorder=2)

    # Legend for profiles
    for p in unique_profiles:
        ax.scatter([], [], color=profile_colors[p], label=p, s=50)

    ax.set_xlabel("Mean Latency (seconds)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs Latency (bubble size = sample count)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.axhline(y=95, color="green", linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "success_vs_latency.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_mcp_transport_comparison_chart(records: list[dict], output_dir: Path) -> str:
    """Compare MCP stdio vs HTTP transport if both are present in the data."""
    if not HAS_MATPLOTLIB:
        return None

    transport_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: {
        "latency": [], "bytes": []
    })

    for r in records:
        meta = _parse_metadata(r)
        if meta.get("type") != "mcp_tool_call":
            continue
        transport = meta.get("transport", "stdio")
        lat = r.get("tool_latency_sec", 0) or 0
        total_bytes = (r.get("request_bytes", 0) or 0) + (r.get("response_bytes", 0) or 0)
        if lat > 0:
            transport_data[transport]["latency"].append(lat * 1000)  # ms
            transport_data[transport]["bytes"].append(total_bytes / 1024)  # KB

    if len(transport_data) < 2:
        return None

    transports = sorted(transport_data.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Latency comparison box plot
    lat_data = [transport_data[t]["latency"] for t in transports]
    bp1 = ax1.boxplot(lat_data, labels=transports, patch_artist=True)
    colors_t = ["#3498db", "#e67e22"]
    for patch, color in zip(bp1["boxes"], colors_t):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xlabel("MCP Transport")
    ax1.set_ylabel("Tool Latency (ms)")
    ax1.set_title("MCP Transport: Latency Comparison")
    ax1.grid(axis="y", alpha=0.3)

    # Bytes comparison box plot
    byte_data = [transport_data[t]["bytes"] for t in transports]
    bp2 = ax2.boxplot(byte_data, labels=transports, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors_t):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_xlabel("MCP Transport")
    ax2.set_ylabel("Traffic per Call (KB)")
    ax2.set_title("MCP Transport: Traffic Comparison")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "mcp_transport_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def _classify_error(error_type: str) -> str:
    """Classify a raw error string into a short category."""
    if not error_type:
        return "Unknown"
    e = error_type.lower()
    if "timeout" in e or "timed out" in e:
        return "Timeout"
    if "429" in e or ("rate" in e and "limit" in e) or "too many requests" in e:
        return "Rate Limited"
    if "400" in e or "bad request" in e or "invalid" in e:
        return "Bad Request"
    if "connect" in e or "connection" in e:
        return "Connection"
    if "auth" in e or "401" in e or "403" in e or "api key" in e:
        return "Auth"
    if "500" in e or "502" in e or "503" in e or "server error" in e or "internal_error" in e:
        return "Server Error"
    if "keyboard" in e or "press" in e or "selector" in e or "element" in e:
        return "Browser/UI"
    if "mcp" in e or "tool" in e:
        return "Tool Failure"
    return "Other"


# Fixed palette for error categories so colours are stable across charts
_ERROR_PALETTE = {
    "Timeout":      "#e74c3c",
    "Rate Limited": "#e67e22",
    "Bad Request":  "#f1c40f",
    "Connection":   "#f39c12",
    "Auth":         "#9b59b6",
    "Server Error": "#c0392b",
    "Browser/UI":   "#2980b9",
    "Tool Failure": "#d35400",
    "Unknown":      "#7f8c8d",
    "Other":        "#95a5a6",
}
_ERROR_MARKERS = {
    "Timeout":      "X",
    "Rate Limited": "D",
    "Bad Request":  "d",
    "Connection":   "s",
    "Auth":         "^",
    "Server Error": "v",
    "Browser/UI":   ">",
    "Tool Failure": "P",
    "Unknown":      "o",
    "Other":        "h",
}


def _error_color(category: str) -> str:
    return _ERROR_PALETTE.get(category, "#95a5a6")


def _error_marker(category: str) -> str:
    return _ERROR_MARKERS.get(category, "o")


def generate_error_analysis_chart(records: list[dict], output_dir: Path) -> str:
    """Generate error analysis chart with colour-coded categories and legend."""
    if not HAS_MATPLOTLIB:
        return None

    # Classify every failure
    success_count = 0
    error_by_category: dict[str, int] = defaultdict(int)
    error_by_scenario: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        if r.get("success"):
            success_count += 1
        else:
            cat = _classify_error(r.get("error_type", ""))
            error_by_category[cat] += 1
            scenario = r.get("scenario_id", "unknown")
            error_by_scenario[scenario][cat] += 1

    if not error_by_category:
        return None

    # Sort categories by count descending
    sorted_cats = sorted(error_by_category.keys(), key=lambda c: error_by_category[c], reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                             gridspec_kw={"width_ratios": [1, 1.4, 1.6]})

    # --- Panel 1: Donut chart (success vs failure) ---
    ax1 = axes[0]
    total = success_count + sum(error_by_category.values())
    fail_count = sum(error_by_category.values())
    wedges, texts, autotexts = ax1.pie(
        [success_count, fail_count],
        labels=None,
        colors=["#27ae60", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.4),
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax1.legend(
        [f"Success ({success_count:,})", f"Failed ({fail_count:,})"],
        loc="lower center", fontsize=9, frameon=False,
    )
    ax1.set_title(f"Overall (n={total:,})", fontsize=11, fontweight="bold")

    # --- Panel 2: Horizontal bars by error category with colour + marker ---
    ax2 = axes[1]
    y_pos = range(len(sorted_cats))
    counts = [error_by_category[c] for c in sorted_cats]
    bar_colors = [_error_color(c) for c in sorted_cats]

    bars = ax2.barh(y_pos, counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_cats, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Count")
    ax2.set_title("Failures by Category", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # Add markers + count labels
    for i, (bar, cat, count) in enumerate(zip(bars, sorted_cats, counts)):
        marker = _error_marker(cat)
        ax2.scatter(
            count + max(counts) * 0.02, i,
            marker=marker, s=60, color=_error_color(cat),
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        ax2.annotate(
            f"{count}", xy=(count, i),
            xytext=(max(counts) * 0.06, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9,
        )

    # --- Panel 3: Per-scenario stacked bars ---
    ax3 = axes[2]

    # Only show scenarios that have errors, sorted by total failures
    scenario_totals = {s: sum(cats.values()) for s, cats in error_by_scenario.items()}
    sorted_scenarios = sorted(scenario_totals.keys(), key=lambda s: scenario_totals[s], reverse=True)
    # Limit to top 12 to avoid crowding
    sorted_scenarios = sorted_scenarios[:12]

    scenario_labels = [format_scenario_label(s, include_provider=False) for s in sorted_scenarios]
    y_pos_s = range(len(sorted_scenarios))

    left = [0] * len(sorted_scenarios)
    legend_handles = []

    for cat in sorted_cats:
        widths = [error_by_scenario[s].get(cat, 0) for s in sorted_scenarios]
        if sum(widths) == 0:
            continue
        color = _error_color(cat)
        marker = _error_marker(cat)
        ax3.barh(y_pos_s, widths, left=left, color=color,
                 edgecolor="white", linewidth=0.5, label=cat)
        legend_handles.append(
            plt.Line2D([0], [0], marker=marker, color="w",
                       markerfacecolor=color, markeredgecolor="black",
                       markersize=8, label=cat)
        )
        left = [l + w for l, w in zip(left, widths)]

    ax3.set_yticks(y_pos_s)
    ax3.set_yticklabels(scenario_labels, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("Failure Count")
    ax3.set_title("Failures by Scenario", fontsize=11, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)
    ax3.legend(handles=legend_handles, loc="lower right", fontsize=8,
               framealpha=0.9, title="Error Type", title_fontsize=9)

    fig.suptitle("Error Analysis", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    output_path = output_dir / "error_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    ax.set_ylabel('Token Count (log scale)')
    ax.set_title('Average Token Counts by Scenario')
    ax.set_xticks(x)
    label_text = [label.replace(" - ", "\n") for label in scenarios]
    ax.set_xticklabels(label_text, fontsize=7, rotation=55, ha='right')
    ax.tick_params(axis='x', labelrotation=55)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    output_path = output_dir / "token_counts.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_path)


# =============================================================================
# RAN2 methodology charts (S4-260859 Annex D)
# =============================================================================
#
# Each function consumes the nested dict returned by
# analysis.ran2_metrics.compute_ran2_metrics(). Charts degrade gracefully to
# an empty-image / None when input data is missing.

_WINDOW_ORDER = ["1ms", "10ms", "100ms", "1s", "10s"]


def _fmt_scenario(s: str) -> str:
    return ANONYMIZER.scenario_alias(s) or s


def generate_ran2_per_direction_packets_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q1.3 — UL vs DL packet counts + mean packet sizes per pcap."""
    rows = ran2.get("Q1", {}).get("pcap_per_direction") or []
    if not rows:
        return None
    labels = [Path(r["pcap_file"]).name[:28] for r in rows]
    ul_pkts = [r.get("ul_packets", 0) for r in rows]
    dl_pkts = [r.get("dl_packets", 0) for r in rows]
    ul_size = [r.get("ul_mean_pkt_size") or 0 for r in rows]
    dl_size = [r.get("dl_mean_pkt_size") or 0 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    x = range(len(labels))
    width = 0.38
    ax1.bar([i - width/2 for i in x], ul_pkts, width, label="UL", color="#1f77b4")
    ax1.bar([i + width/2 for i in x], dl_pkts, width, label="DL", color="#ff7f0e")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Packet count"); ax1.set_title("Q1.3 — Per-direction packet count (per pcap)")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    ax2.bar([i - width/2 for i in x], ul_size, width, label="UL", color="#1f77b4")
    ax2.bar([i + width/2 for i in x], dl_size, width, label="DL", color="#ff7f0e")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean packet size (bytes)")
    ax2.set_title("Q1.3 — Per-direction mean packet size")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = output_dir / "ran2_q1_per_direction_packets.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_multiwindow_throughput_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q1.4 — peak Mbps per window (1/10/100ms/1s/10s), per pcap."""
    rows = ran2.get("Q1", {}).get("pcap_per_direction") or []
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    x = list(range(len(_WINDOW_ORDER)))
    for r in rows:
        peaks = r.get("peak_mbps_by_window") or {}
        y = [peaks.get(w, 0) for w in _WINDOW_ORDER]
        ax.plot(x, y, marker="o", label=Path(r["pcap_file"]).name[:32])
    ax.set_xticks(x); ax.set_xticklabels(_WINDOW_ORDER)
    ax.set_xlabel("Averaging window")
    ax.set_ylabel("Peak throughput (Mbps)")
    ax.set_title("Q1.4 — Per-direction peak throughput across 1ms…10s windows\n"
                 "Short windows reveal sub-second bursts; long windows show sustained rate")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "ran2_q1_multiwindow_throughput.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_burstiness_by_window_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q2.3 — peak/mean burstiness index at 1ms/10ms/100ms/1s/10s windows."""
    rows = ran2.get("Q2", {}).get("per_pcap") or []
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(14, 6))
    x = list(range(len(_WINDOW_ORDER)))
    for r in rows:
        vals = r.get("burstiness_by_window") or {}
        y = [vals.get(w) or 0 for w in _WINDOW_ORDER]
        ax.plot(x, y, marker="s", label=Path(r["pcap_file"]).name[:32])
    ax.set_xticks(x); ax.set_xticklabels(_WINDOW_ORDER)
    ax.set_xlabel("Averaging window")
    ax.set_ylabel("Burstiness index (peak / mean)")
    ax.set_title("Q2.3 — Burstiness across averaging windows\n"
                 "Higher at short windows = bursty; flat curve = smooth")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    # Annotate OS-scheduler-jitter caveat for <10ms
    ax.axvspan(-0.4, 0.4, alpha=0.15, color="red")
    ax.text(0, ax.get_ylim()[1] * 0.95,
            "OS jitter region", ha="center", fontsize=8, color="darkred")
    plt.tight_layout()
    out = output_dir / "ran2_q2_burstiness_by_window.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_interburst_idle_cdf_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q2.2 / Q4.5 — inter-burst idle-gap CDF per direction at 10ms + 100ms gap."""
    # Reach into the raw pcap_metrics? No — ran2_metrics emits distributions.
    # We draw one bar group per (gap, direction) showing p50/p95/p99 of idle gaps.
    rows = ran2.get("Q2", {}).get("per_pcap") or []
    if not rows:
        return None

    categories: list[str] = []
    p50: list[float] = []
    p95: list[float] = []
    p99: list[float] = []
    for r in rows:
        name = Path(r["pcap_file"]).name[:20]
        for gap in ("10ms", "100ms"):
            for direction in ("ul", "dl"):
                d = (((r.get("interburst_idle_by_gap") or {}).get(gap) or {}).get(direction) or {}).get("cdf_sec") or {}
                if d.get("n"):
                    categories.append(f"{name}\n{direction.upper()}@{gap}")
                    p50.append((d.get("p50") or 0) * 1000)
                    p95.append((d.get("p95") or 0) * 1000)
                    p99.append((d.get("p99") or 0) * 1000)
    if not categories:
        return None

    x = range(len(categories))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(14, len(categories) * 0.8), 6))
    ax.bar([i - width for i in x], p50, width, label="p50", color="#4daf4a")
    ax.bar(list(x), p95, width, label="p95", color="#ff7f00")
    ax.bar([i + width for i in x], p99, width, label="p99", color="#e41a1c")
    ax.set_xticks(list(x)); ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Inter-burst idle gap (ms)")
    ax.set_yscale("log")
    ax.set_title("Q2.2 / Q4.5 — Inter-burst idle-gap distribution (per direction, gap threshold)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "ran2_q2_interburst_idle_cdf.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_rtt_components_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q3.1/Q3.2 — TCP handshake RTT vs TLS handshake vs HTTP setup RTT distributions."""
    q3 = ran2.get("Q3") or {}
    items = [
        ("TCP handshake RTT", q3.get("tcp_rtt") or {}),
        ("TLS handshake", q3.get("tls_handshake") or {}),
        ("HTTP setup RTT", q3.get("http_setup_rtt") or {}),
    ]
    items = [(name, d) for name, d in items if d.get("n")]
    if not items:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(items))
    width = 0.22
    mins = [d.get("min") or 0 for _, d in items]
    p50s = [d.get("p50") or 0 for _, d in items]
    p95s = [d.get("p95") or 0 for _, d in items]
    maxs = [d.get("max") or 0 for _, d in items]
    ax.bar([i - 1.5*width for i in x], mins, width, label="min", color="#4daf4a")
    ax.bar([i - 0.5*width for i in x], p50s, width, label="p50", color="#377eb8")
    ax.bar([i + 0.5*width for i in x], p95s, width, label="p95", color="#ff7f00")
    ax.bar([i + 1.5*width for i in x], maxs, width, label="max", color="#e41a1c")
    ax.set_xticks(list(x)); ax.set_xticklabels([n for n, _ in items])
    ax.set_ylabel("Duration (ms)")
    ax.set_title("Q3.1 / Q3.2 — RTT components")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "ran2_q3_rtt_components.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_e2e_latency_over_rtt_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q3.4 — (E2E latency / TCP RTT) per scenario/profile."""
    data = ran2.get("Q3", {}).get("e2e_latency_vs_rtt") or {}
    if not data:
        return None

    labels = sorted(data.keys())
    ratios = [(data[k].get("p50") or 0) for k in labels]
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.55), 6))
    ax.bar(range(len(labels)), ratios, color="#984ea3")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("E2E latency / RTT p50 (x)")
    ax.set_title("Q3.4 — End-to-end latency as a multiple of TCP RTT (non-streaming)")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, label="1× RTT")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "ran2_q3_e2e_latency_over_rtt.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_reliability_vs_loss_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q4.4 — success rate vs nominal profile loss_pct (one dot per scenario/profile)."""
    data = ran2.get("Q4", {}).get("reliability_by_loss_pct") or {}
    if not data:
        return None

    xs, ys, labels = [], [], []
    for key, v in data.items():
        lp = v.get("profile_loss_pct")
        sr = v.get("success_rate")
        if lp is None or sr is None:
            continue
        xs.append(lp)
        ys.append(sr * 100.0)
        labels.append(key)
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(xs, ys, s=60, alpha=0.7, edgecolors="black")
    for lx, ly, ll in zip(xs, ys, labels):
        ax.annotate(ll, (lx, ly), xytext=(4, 4), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Nominal profile loss (%)")
    ax.set_ylabel("Observed success rate (%)")
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_title("Q4.4 — Reliability of service vs netem loss rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "ran2_q4_reliability_vs_loss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_flow_duration_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q4.6 — flow-duration distribution + connection-reuse ratio + flows-per-pcap."""
    cd = ran2.get("Q4", {}).get("connection_duration") or {}
    fd = cd.get("flow_duration_sec") or {}
    if not fd.get("n"):
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = ["min", "p50", "p95", "p99", "max"]
    vals = [fd.get(k) or 0 for k in labels]
    ax1.bar(labels, vals, color="#2ca02c")
    ax1.set_ylabel("Flow duration (s)")
    ax1.set_yscale("log")
    ax1.set_title("Q4.6 — TCP flow duration distribution")
    ax1.grid(axis="y", alpha=0.3)
    for l, v in zip(labels, vals):
        ax1.text(l, v * 1.05, f"{v:.2f}", ha="center", fontsize=8)

    reuse = cd.get("connection_reuse_ratio")
    flows_per = cd.get("flows_per_pcap") or {}
    ax2.axis("off")
    lines = [
        f"Connection-reuse ratio: {reuse:.1%}" if reuse is not None else "Connection-reuse ratio: —",
        "",
        "Flows per pcap:",
        f"  min = {flows_per.get('min') or 0:.0f}",
        f"  p50 = {flows_per.get('p50') or 0:.0f}",
        f"  p95 = {flows_per.get('p95') or 0:.0f}",
        f"  max = {flows_per.get('max') or 0:.0f}",
        "",
        "High reuse = HTTP/2 or keep-alive.",
        "Low reuse = new connection per turn.",
    ]
    ax2.text(0.02, 0.95, "\n".join(lines), transform=ax2.transAxes,
             fontsize=11, va="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5"))
    plt.tight_layout()
    out = output_dir / "ran2_q4_flow_duration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_per_tool_bytes_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q4.7 — per-tool request+response bytes (agentic scenarios)."""
    tools = (ran2.get("Q4", {}).get("agentic_flows") or {}).get("per_tool_bytes") or {}
    if not tools:
        return None
    # Sort by total bytes desc
    items = sorted(tools.items(), key=lambda kv: kv[1].get("request_bytes", 0) + kv[1].get("response_bytes", 0), reverse=True)
    items = items[:25]  # top 25
    names = [k for k, _ in items]
    req = [v.get("request_bytes", 0) for _, v in items]
    resp = [v.get("response_bytes", 0) for _, v in items]

    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.28)))
    y = range(len(names))
    ax.barh(y, req, color="#1f77b4", label="Request bytes")
    ax.barh(y, resp, left=req, color="#ff7f0e", label="Response bytes")
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Bytes")
    ax.set_xscale("log")
    ax.set_title("Q4.7 — Per-tool sub-flow volume (top 25 tools, request + response)")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "ran2_q4_per_tool_bytes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_inter_token_gap_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q5.4 — inter-token gap distribution per network profile."""
    data = ran2.get("Q5", {}).get("inter_token_gap_by_profile") or {}
    if not data:
        return None

    profiles = sorted(data.keys())
    p50 = [(data[p].get("p50") or 0) * 1000 for p in profiles]
    p95 = [(data[p].get("p95") or 0) * 1000 for p in profiles]
    p99 = [(data[p].get("p99") or 0) * 1000 for p in profiles]

    x = range(len(profiles))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(11, len(profiles) * 0.9), 6))
    ax.bar([i - width for i in x], p50, width, label="p50", color="#4daf4a")
    ax.bar(list(x), p95, width, label="p95", color="#ff7f00")
    ax.bar([i + width for i in x], p99, width, label="p99", color="#e41a1c")
    ax.set_xticks(list(x)); ax.set_xticklabels(profiles, rotation=30, ha="right")
    ax.set_ylabel("Inter-token gap (ms)")
    ax.set_yscale("log")
    ax.set_title("Q5.4 — Inter-token gap distribution per network profile")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "ran2_q5_inter_token_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_tokens_to_bytes_regression_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q5.5 — slope (bytes per token) per scenario, UL vs DL, with r²."""
    data = ran2.get("Q5", {}).get("token_to_bytes_regression_by_scenario") or {}
    if not data:
        return None

    scenarios: list[str] = []
    ul_slopes: list[float] = []
    dl_slopes: list[float] = []
    ul_r2s: list[float] = []
    dl_r2s: list[float] = []
    for s, r in data.items():
        ul = r.get("ul") or {}
        dl = r.get("dl") or {}
        if not ul and not dl:
            continue
        scenarios.append(_fmt_scenario(s))
        ul_slopes.append(ul.get("slope") or 0)
        dl_slopes.append(dl.get("slope") or 0)
        ul_r2s.append(ul.get("r2") or 0)
        dl_r2s.append(dl.get("r2") or 0)
    if not scenarios:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(5, len(scenarios) * 0.3)))
    y = range(len(scenarios))
    ax1.barh(y, ul_slopes, color="#1f77b4", label="UL slope (bytes/token_in)")
    ax1.barh(y, dl_slopes, color="#ff7f0e", alpha=0.6, label="DL slope (bytes/token_out)")
    ax1.set_yticks(list(y)); ax1.set_yticklabels(scenarios, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Bytes per token (regression slope)")
    ax1.set_title("Q5.5 — tokens→bytes regression slope per scenario")
    ax1.legend(); ax1.grid(axis="x", alpha=0.3)

    # r² chart
    width = 0.4
    ax2.barh([i - width/2 for i in y], ul_r2s, width, label="UL r²", color="#1f77b4")
    ax2.barh([i + width/2 for i in y], dl_r2s, width, label="DL r²", color="#ff7f0e")
    ax2.set_yticks(list(y)); ax2.set_yticklabels(scenarios, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Coefficient of determination r² (0 → 1)")
    ax2.set_xlim(0, 1.05)
    ax2.axvline(0.8, color="green", linestyle="--", alpha=0.5, label="0.8 (good fit)")
    ax2.set_title("Q5.5 — Regression quality (r²)")
    ax2.legend(); ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "ran2_q5_tokens_to_bytes_regression.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def generate_ran2_token_vs_pkt_rate_chart(ran2: dict, output_dir: Path) -> Optional[str]:
    """Q5.3 — token-arrival rate (per profile) vs DL-packet-arrival rate (from pcap)."""
    q5 = ran2.get("Q5", {}).get("token_arrival_vs_pkt_arrival") or {}
    tok_rate = q5.get("token_rate_per_profile_hz") or {}
    dl_rate_dist = q5.get("dl_pkt_rate_hz") or {}
    if not tok_rate and not dl_rate_dist.get("n"):
        return None

    profiles = sorted(tok_rate.keys())
    trates = [tok_rate[p] or 0 for p in profiles]
    dl_med = dl_rate_dist.get("p50") or 0

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(profiles))
    ax.bar(x, trates, label="Token arrival rate (Hz, per profile)", color="#1f77b4")
    if dl_med > 0:
        ax.axhline(dl_med, color="#d62728", linestyle="--",
                   label=f"DL packet arrival rate p50 = {dl_med:.1f} Hz")
    ax.set_xticks(list(x)); ax.set_xticklabels(profiles, rotation=30, ha="right")
    ax.set_ylabel("Arrival rate (Hz)")
    ax.set_title("Q5.3 — Token arrival rate vs DL packet arrival rate")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "ran2_q5_token_vs_pkt_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    return str(out)


def _generate_all_ran2_charts(ran2: dict, output_dir: Path) -> dict:
    """Dispatcher — run all RAN2 chart generators, return {key: path}."""
    generators = [
        ("ran2_q1_per_direction_packets", generate_ran2_per_direction_packets_chart),
        ("ran2_q1_multiwindow_throughput", generate_ran2_multiwindow_throughput_chart),
        ("ran2_q2_burstiness_by_window", generate_ran2_burstiness_by_window_chart),
        ("ran2_q2_interburst_idle_cdf", generate_ran2_interburst_idle_cdf_chart),
        ("ran2_q3_rtt_components", generate_ran2_rtt_components_chart),
        ("ran2_q3_e2e_latency_over_rtt", generate_ran2_e2e_latency_over_rtt_chart),
        ("ran2_q4_reliability_vs_loss", generate_ran2_reliability_vs_loss_chart),
        ("ran2_q4_flow_duration", generate_ran2_flow_duration_chart),
        ("ran2_q4_per_tool_bytes", generate_ran2_per_tool_bytes_chart),
        ("ran2_q5_inter_token_gap", generate_ran2_inter_token_gap_chart),
        ("ran2_q5_tokens_to_bytes_regression", generate_ran2_tokens_to_bytes_regression_chart),
        ("ran2_q5_token_vs_pkt_rate", generate_ran2_token_vs_pkt_rate_chart),
    ]
    out: dict[str, str] = {}
    for key, fn in generators:
        try:
            path = fn(ran2, output_dir)
            if path:
                out[key] = path
                print(f"  ✓ RAN2 {key}: {path}")
        except Exception as e:
            print(f"  ⚠ RAN2 chart {key} failed: {e}")
    return out


def main():
    """Generate all charts and print results."""
    parser = argparse.ArgumentParser(description="Generate visualization charts from traffic test data")
    parser.add_argument("--latest", action="store_true", help="Only include data from the latest test run")
    parser.add_argument("--since-minutes", type=int, help="Only include data from the last N minutes")
    parser.add_argument("--since-timestamp", type=float, help="Only include data after this Unix timestamp")
    parser.add_argument("--db", default="logs/traffic_logs.db", help="Path to SQLite database")
    parser.add_argument("--all-runs", action="store_true", help="Include all runs instead of latest per scenario")
    parser.add_argument("--run-gap-sec", type=float, default=300.0, help="Gap in seconds to split runs per scenario")
    parser.add_argument("--output-dir", default="results/reports/figures", help="Output directory for charts")
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

    # Split into success-only (for latency/throughput/token charts) and all records
    # (for success rate/error charts). This prevents failed requests with fast error
    # times from skewing latency averages downward.
    success_records = [r for r in records if r.get("success")]
    print(f"  Success-only records: {len(success_records)}/{len(records)}")

    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib seaborn")
        return

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    if HAS_SEABORN:
        sns.set_palette("husl")

    generated = {}

    print("\nGenerating charts...")

    # Generate each chart.
    # Most charts use success_records to avoid failed requests skewing latency
    # averages. Charts that need failure data use the full records list.
    sr = success_records  # shorthand

    path = generate_latency_by_scenario(sr, output_dir)
    if path:
        generated["latency_by_scenario"] = path
        print(f"  ✓ Latency by scenario: {path}")

    path = generate_throughput_chart(sr, output_dir)
    if path:
        generated["throughput"] = path
        print(f"  ✓ Throughput: {path}")

    path = generate_throughput_over_time(sr, output_dir)
    if path:
        generated["throughput_over_time"] = path
        print(f"  ✓ Throughput over time: {path}")

    path = generate_throughput_burstiness_chart(sr, output_dir)
    if path:
        generated["throughput_burstiness"] = path
        print(f"  ✓ Throughput burstiness: {path}")

    path = generate_bandwidth_asymmetry_chart(sr, output_dir)
    if path:
        generated["bandwidth_asymmetry"] = path
        print(f"  ✓ Bandwidth asymmetry: {path}")

    path = generate_protocol_comparison_chart(sr, output_dir)
    if path:
        generated["protocol_comparison"] = path
        print(f"  ✓ Protocol comparison: {path}")

    path = generate_success_rate_chart(records, output_dir)  # needs all records
    if path:
        generated["success_rate"] = path
        print(f"  ✓ Success rate: {path}")

    path = generate_data_volume_chart(records, output_dir)  # needs all records
    if path:
        generated["data_volume"] = path
        print(f"  ✓ Data volume: {path}")

    path = generate_latency_heatmap(sr, output_dir)
    if path:
        generated["latency_heatmap"] = path
        print(f"  ✓ Latency heatmap: {path}")

    path = generate_ttft_chart(sr, output_dir)
    if path:
        generated["ttft_by_scenario"] = path
        print(f"  ✓ TTFT by scenario: {path}")

    path = generate_latency_breakdown_chart(sr, output_dir)
    if path:
        generated["latency_breakdown"] = path
        print(f"  ✓ Latency breakdown: {path}")

    path = generate_token_throughput_chart(sr, output_dir)
    if path:
        generated["token_throughput"] = path
        print(f"  ✓ Token throughput: {path}")

    path = generate_streaming_metrics_chart(sr, output_dir)
    if path:
        generated["streaming_metrics"] = path
        print(f"  ✓ Streaming metrics: {path}")

    path = generate_tool_usage_chart(sr, output_dir)
    if path:
        generated["tool_usage"] = path
        print(f"  ✓ Tool usage: {path}")

    path = generate_mcp_efficiency_chart(sr, output_dir)
    if path:
        generated["mcp_efficiency"] = path
        print(f"  ✓ MCP efficiency: {path}")

    path = generate_mcp_latency_breakdown_chart(sr, output_dir)
    if path:
        generated["mcp_latency_breakdown"] = path
        print(f"  ✓ MCP latency breakdown: {path}")

    path = generate_mcp_loop_factor_by_profile_chart(sr, output_dir)
    if path:
        generated["mcp_loop_factor_by_profile"] = path
        print(f"  ✓ MCP loop factor by profile: {path}")

    path = generate_tool_latency_cdf_chart(sr, output_dir)
    if path:
        generated["tool_latency_cdf"] = path
        print(f"  ✓ Tool latency CDF: {path}")

    path = generate_mcp_protocol_overhead_chart(sr, output_dir)
    if path:
        generated["mcp_protocol_overhead"] = path
        print(f"  ✓ MCP protocol overhead: {path}")

    path = generate_tool_success_rate_chart(records, output_dir)  # needs all records
    if path:
        generated["tool_success_rate"] = path
        print(f"  ✓ Tool success rate: {path}")

    path = generate_agent_session_waterfall_chart(sr, output_dir)
    if path:
        generated["agent_session_waterfall"] = path
        print(f"  ✓ Agent session waterfall: {path}")

    waterfall_paths = generate_all_agent_waterfall_charts(sr, output_dir)
    for scenario, wpath in waterfall_paths.items():
        generated[f"waterfall_{scenario}"] = wpath
        print(f"  ✓ Waterfall ({scenario}): {wpath}")

    path = generate_mcp_transport_comparison_chart(sr, output_dir)
    if path:
        generated["mcp_transport_comparison"] = path
        print(f"  ✓ MCP transport comparison: {path}")

    path = generate_token_rate_by_profile_chart(sr, output_dir)
    if path:
        generated["token_rate_by_profile"] = path
        print(f"  ✓ Token rate by profile: {path}")

    path = generate_context_growth_chart(sr, output_dir)
    if path:
        generated["context_growth"] = path
        print(f"  ✓ Context growth: {path}")

    path = generate_request_response_scatter_chart(sr, output_dir)
    if path:
        generated["request_response_scatter"] = path
        print(f"  ✓ Request/response scatter: {path}")

    path = generate_ttft_vs_latency_chart(sr, output_dir)
    if path:
        generated["ttft_vs_latency"] = path
        print(f"  ✓ TTFT vs latency: {path}")

    path = generate_inter_turn_idle_chart(sr, output_dir)
    if path:
        generated["inter_turn_idle"] = path
        print(f"  ✓ Inter-turn idle time: {path}")

    path = generate_degradation_heatmap_chart(sr, output_dir)
    if path:
        generated["degradation_heatmap"] = path
        print(f"  ✓ Degradation heatmap: {path}")

    path = generate_success_vs_latency_chart(records, output_dir)  # needs all records
    if path:
        generated["success_vs_latency"] = path
        print(f"  ✓ Success vs latency: {path}")

    path = generate_error_analysis_chart(records, output_dir)  # needs all records
    if path:
        generated["error_analysis"] = path
        print(f"  ✓ Error analysis: {path}")

    path = generate_latency_by_profile_chart(sr, output_dir)
    if path:
        generated["latency_by_profile"] = path
        print(f"  ✓ Latency by profile: {path}")

    path = generate_tokens_chart(sr, output_dir)
    if path:
        generated["token_counts"] = path
        print(f"  ✓ Token counts: {path}")

    path = generate_ttft_heatmap(sr, output_dir)
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

    # =================================================================
    # RAN2 methodology metrics + charts (S4-260859 Annex D)
    # Runs regardless of pcap: DB-only metrics still emit; pcap-backed
    # metrics degrade to n=0 if no pcap data was captured.
    # =================================================================
    if HAS_RAN2:
        print("\nComputing RAN2 methodology metrics (S4-260859)...")
        try:
            ran2 = compute_ran2_metrics(
                records=records,
                pcap_metrics=pcap_metrics or [],
                profiles_yaml="configs/profiles.yaml",
            )
            # Persist the full dict so the report generator can pick it up
            ran2_path = output_dir / "ran2_metrics.json"
            with open(ran2_path, "w") as f:
                json.dump(ran2, f, indent=2, default=str)
            print(f"  ✓ RAN2 metrics JSON: {ran2_path}")

            print("Generating RAN2 methodology charts...")
            ran2_charts = _generate_all_ran2_charts(ran2, output_dir)
            generated.update(ran2_charts)
        except Exception as e:
            print(f"  ⚠ RAN2 metrics computation failed: {e}")

    print(f"\n✓ Generated {len(generated)} charts in {output_dir}/")

    # Output JSON summary
    summary_path = output_dir / "charts_summary.json"
    with open(summary_path, "w") as f:
        json.dump(generated, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")

    return generated


if __name__ == "__main__":
    main()
