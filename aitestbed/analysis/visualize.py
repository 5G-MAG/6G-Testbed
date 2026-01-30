"""
Visualization module for the 6G AI Traffic Testbed.

Generates plots and figures for 3GPP-style reporting.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import asdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .metrics import ScenarioMetrics, MetricsCalculator


class TrafficVisualizer:
    """
    Generate visualizations for traffic analysis results.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        if HAS_MATPLOTLIB:
            plt.style.use('seaborn-v0_8-whitegrid')
        if HAS_SEABORN:
            sns.set_palette("husl")

    def plot_latency_comparison(
        self,
        metrics_list: list[ScenarioMetrics],
        title: str = "Latency Comparison Across Network Profiles",
        filename: str = "latency_comparison.png"
    ) -> Optional[Path]:
        """
        Create a bar chart comparing latency across profiles.
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        profiles = [m.network_profile for m in metrics_list]
        means = [m.latency_mean * 1000 for m in metrics_list]  # Convert to ms
        p95s = [m.latency_p95 * 1000 for m in metrics_list]

        x = range(len(profiles))
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], means, width, label='Mean', color='steelblue')
        bars2 = ax.bar([i + width/2 for i in x], p95s, width, label='P95', color='coral')

        ax.set_xlabel('Network Profile')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=45, ha='right')
        ax.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_latency_cdf(
        self,
        latencies: list[float],
        profile: str,
        scenario: str,
        filename: str = "latency_cdf.png"
    ) -> Optional[Path]:
        """
        Create a CDF plot for latency distribution.
        """
        if not HAS_MATPLOTLIB or not HAS_PANDAS:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        sorted_latencies = sorted(latencies)
        cdf = [i / len(sorted_latencies) for i in range(1, len(sorted_latencies) + 1)]

        ax.plot([l * 1000 for l in sorted_latencies], cdf, linewidth=2)
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('CDF')
        ax.set_title(f'Latency CDF - {scenario} ({profile})')
        ax.grid(True, alpha=0.3)

        # Add percentile markers
        for p in [50, 90, 95, 99]:
            idx = int(len(sorted_latencies) * p / 100)
            if idx < len(sorted_latencies):
                val = sorted_latencies[idx] * 1000
                ax.axhline(y=p/100, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5)
                ax.annotate(f'P{p}: {val:.0f}ms',
                           xy=(val, p/100), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_ul_dl_ratio(
        self,
        metrics_list: list[ScenarioMetrics],
        title: str = "Uplink/Downlink Ratio by Scenario",
        filename: str = "ul_dl_ratio.png"
    ) -> Optional[Path]:
        """
        Create a chart showing UL/DL ratios.
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        scenarios = list(set(m.scenario_id for m in metrics_list))
        profiles = list(set(m.network_profile for m in metrics_list))

        # Group by scenario
        data = {}
        for m in metrics_list:
            if m.scenario_id not in data:
                data[m.scenario_id] = {}
            data[m.scenario_id][m.network_profile] = m.ul_dl_ratio_mean

        x = range(len(scenarios))
        width = 0.8 / len(profiles)

        for i, profile in enumerate(profiles):
            values = [data.get(s, {}).get(profile, 0) for s in scenarios]
            offset = (i - len(profiles)/2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], values, width, label=profile)

        ax.set_xlabel('Scenario')
        ax.set_ylabel('UL/DL Ratio')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend(title='Network Profile')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Symmetric')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_agent_metrics(
        self,
        metrics_list: list[ScenarioMetrics],
        title: str = "Agent Loop Factor and Tool Calls",
        filename: str = "agent_metrics.png"
    ) -> Optional[Path]:
        """
        Create a chart showing agent-specific metrics.
        """
        if not HAS_MATPLOTLIB:
            return None

        # Filter to agent scenarios only
        agent_metrics = [m for m in metrics_list if m.tool_calls_mean > 0]
        if not agent_metrics:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loop factor chart
        profiles = [m.network_profile for m in agent_metrics]
        loop_factors = [m.loop_factor for m in agent_metrics]

        ax1.bar(profiles, loop_factors, color='teal')
        ax1.set_xlabel('Network Profile')
        ax1.set_ylabel('Loop Factor (API calls per prompt)')
        ax1.set_title('Agent Loop Factor')
        ax1.tick_params(axis='x', rotation=45)

        # Tool calls chart
        tool_calls = [m.tool_calls_mean for m in agent_metrics]

        ax2.bar(profiles, tool_calls, color='orange')
        ax2.set_xlabel('Network Profile')
        ax2.set_ylabel('Average Tool Calls')
        ax2.set_title('Tool Calls per Session')
        ax2.tick_params(axis='x', rotation=45)

        plt.suptitle(title)
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_success_rate(
        self,
        metrics_list: list[ScenarioMetrics],
        title: str = "Success Rate Under Network Degradation",
        filename: str = "success_rate.png"
    ) -> Optional[Path]:
        """
        Create a chart showing success rates across profiles.
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        scenarios = list(set(m.scenario_id for m in metrics_list))
        profiles = list(set(m.network_profile for m in metrics_list))

        # Group by scenario
        data = {}
        for m in metrics_list:
            if m.scenario_id not in data:
                data[m.scenario_id] = {}
            data[m.scenario_id][m.network_profile] = m.success_rate

        x = range(len(scenarios))
        width = 0.8 / len(profiles)

        for i, profile in enumerate(profiles):
            values = [data.get(s, {}).get(profile, 0) for s in scenarios]
            offset = (i - len(profiles)/2 + 0.5) * width
            bars = ax.bar([xi + offset for xi in x], values, width, label=profile)

        ax.set_xlabel('Scenario')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend(title='Network Profile')
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def generate_3gpp_table(
        self,
        metrics_list: list[ScenarioMetrics],
        output_path: str = "reports/3gpp_metrics_table.md"
    ) -> Path:
        """
        Generate a markdown table formatted for 3GPP documentation.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# 6G AI Traffic Characterization Results",
            "",
            "## QoE Metrics Summary",
            "",
            "| Scenario | Profile | TTFT (ms) | Latency Mean (ms) | P95 (ms) | Success (%) |",
            "|:---------|:--------|----------:|------------------:|---------:|------------:|",
        ]

        for m in metrics_list:
            ttft = f"{m.ttft_mean * 1000:.0f}" if m.ttft_mean else "N/A"
            lines.append(
                f"| {m.scenario_id} | {m.network_profile} | {ttft} | "
                f"{m.latency_mean * 1000:.0f} | {m.latency_p95 * 1000:.0f} | "
                f"{m.success_rate:.1f} |"
            )

        lines.extend([
            "",
            "## Traffic Characteristics",
            "",
            "| Scenario | Profile | UL (bytes) | DL (bytes) | UL/DL Ratio | Token Rate |",
            "|:---------|:--------|----------:|-----------:|------------:|-----------:|",
        ])

        for m in metrics_list:
            token_rate = f"{m.token_rate_mean:.1f}" if m.token_rate_mean else "N/A"
            lines.append(
                f"| {m.scenario_id} | {m.network_profile} | "
                f"{m.request_bytes_mean:.0f} | {m.response_bytes_mean:.0f} | "
                f"{m.ul_dl_ratio_mean:.3f} | {token_rate} |"
            )

        # Agent metrics section
        agent_metrics = [m for m in metrics_list if m.tool_calls_mean > 0]
        if agent_metrics:
            lines.extend([
                "",
                "## Agent/Tool Metrics",
                "",
                "| Scenario | Profile | Loop Factor | Tool Calls | Tool Latency (ms) |",
                "|:---------|:--------|------------:|-----------:|------------------:|",
            ])

            for m in agent_metrics:
                lines.append(
                    f"| {m.scenario_id} | {m.network_profile} | "
                    f"{m.loop_factor:.2f} | {m.tool_calls_mean:.1f} | "
                    f"{m.tool_latency_mean * 1000:.0f} |"
                )

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        return output_path

    def plot_latency_boxplot(
        self,
        latencies_by_profile: dict[str, list[float]],
        scenario: str,
        title: str = "Latency Distribution by Profile",
        filename: str = "latency_boxplot.png"
    ) -> Optional[Path]:
        """
        Create box plots showing latency distributions for each profile.
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        profiles = list(latencies_by_profile.keys())
        data = [[l * 1000 for l in latencies_by_profile[p]] for p in profiles]

        bp = ax.boxplot(data, labels=profiles, patch_artist=True)

        # Color the boxes
        colors = plt.cm.Set3(range(len(profiles)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Network Profile')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'{title} - {scenario}')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_ttft_comparison(
        self,
        metrics_list: list[ScenarioMetrics],
        title: str = "Time-to-First-Token Comparison",
        filename: str = "ttft_comparison.png"
    ) -> Optional[Path]:
        """
        Create a chart specifically comparing TTFT across profiles.
        """
        if not HAS_MATPLOTLIB:
            return None

        # Filter to metrics with TTFT data
        ttft_metrics = [m for m in metrics_list if m.ttft_mean is not None]
        if not ttft_metrics:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        scenarios = list(set(m.scenario_id for m in ttft_metrics))
        profiles = list(set(m.network_profile for m in ttft_metrics))

        # Group by scenario
        data = {}
        for m in ttft_metrics:
            if m.scenario_id not in data:
                data[m.scenario_id] = {}
            data[m.scenario_id][m.network_profile] = m.ttft_mean * 1000  # Convert to ms

        x = range(len(scenarios))
        width = 0.8 / len(profiles)

        for i, profile in enumerate(profiles):
            values = [data.get(s, {}).get(profile, 0) for s in scenarios]
            offset = (i - len(profiles)/2 + 0.5) * width
            ax.bar([xi + offset for xi in x], values, width, label=profile)

        ax.set_xlabel('Scenario')
        ax.set_ylabel('Time-to-First-Token (ms)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.legend(title='Network Profile')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_metrics_heatmap(
        self,
        metrics_list: list[ScenarioMetrics],
        metric: str = "latency_mean",
        title: str = "Latency Heatmap (Scenario x Profile)",
        filename: str = "metrics_heatmap.png"
    ) -> Optional[Path]:
        """
        Create a heatmap showing a metric across scenarios and profiles.

        Args:
            metrics_list: List of ScenarioMetrics
            metric: Attribute name to plot (latency_mean, success_rate, etc.)
            title: Plot title
            filename: Output filename
        """
        if not HAS_MATPLOTLIB or not HAS_SEABORN:
            return None

        scenarios = sorted(set(m.scenario_id for m in metrics_list))
        profiles = sorted(set(m.network_profile for m in metrics_list))

        # Build matrix
        matrix = []
        for scenario in scenarios:
            row = []
            for profile in profiles:
                value = 0
                for m in metrics_list:
                    if m.scenario_id == scenario and m.network_profile == profile:
                        val = getattr(m, metric, 0)
                        # Convert latency to ms if applicable
                        if 'latency' in metric or 'ttft' in metric or 'ttlt' in metric:
                            val = val * 1000 if val else 0
                        value = val
                        break
                row.append(value)
            matrix.append(row)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(len(profiles)))
        ax.set_yticks(range(len(scenarios)))
        ax.set_xticklabels(profiles, rotation=45, ha='right')
        ax.set_yticklabels(scenarios)
        ax.set_xlabel('Network Profile')
        ax.set_ylabel('Scenario')
        ax.set_title(title)

        # Add value annotations
        for i in range(len(scenarios)):
            for j in range(len(profiles)):
                val = matrix[i][j]
                text = f'{val:.0f}' if val > 1 else f'{val:.2f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_streaming_profile(
        self,
        chunk_times: list[float],
        scenario: str,
        profile: str,
        title: str = "Streaming Token Arrival",
        filename: str = "streaming_profile.png"
    ) -> Optional[Path]:
        """
        Plot token arrival times showing streaming behavior.

        Args:
            chunk_times: List of timestamps (in seconds) when each chunk arrived
            scenario: Scenario name
            profile: Network profile name
        """
        if not HAS_MATPLOTLIB or not chunk_times:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Cumulative chunks over time
        times = chunk_times
        cumulative = list(range(1, len(times) + 1))

        ax1.plot(times, cumulative, marker='.', markersize=3, linewidth=1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cumulative Chunks')
        ax1.set_title(f'Cumulative Token Arrival - {scenario} ({profile})')
        ax1.grid(True, alpha=0.3)

        # Inter-chunk intervals
        if len(times) > 1:
            intervals = [times[i] - times[i-1] for i in range(1, len(times))]
            ax2.bar(range(len(intervals)), [i * 1000 for i in intervals], width=0.8, alpha=0.7)
            ax2.axhline(y=sum(intervals)/len(intervals) * 1000, color='red',
                       linestyle='--', label=f'Mean: {sum(intervals)/len(intervals)*1000:.1f}ms')
            ax2.set_xlabel('Chunk Index')
            ax2.set_ylabel('Inter-chunk Interval (ms)')
            ax2.set_title('Token Arrival Intervals')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def generate_full_report(
        self,
        metrics_list: list[ScenarioMetrics],
        report_name: str = "experiment_report"
    ) -> dict:
        """
        Generate a complete report with all visualizations.

        Returns:
            Dictionary with paths to generated files
        """
        generated = {}

        # Generate plots
        if HAS_MATPLOTLIB:
            path = self.plot_latency_comparison(metrics_list, filename=f"{report_name}_latency.png")
            if path:
                generated["latency_comparison"] = str(path)

            path = self.plot_ul_dl_ratio(metrics_list, filename=f"{report_name}_ul_dl.png")
            if path:
                generated["ul_dl_ratio"] = str(path)

            path = self.plot_success_rate(metrics_list, filename=f"{report_name}_success.png")
            if path:
                generated["success_rate"] = str(path)

            path = self.plot_agent_metrics(metrics_list, filename=f"{report_name}_agent.png")
            if path:
                generated["agent_metrics"] = str(path)

            path = self.plot_ttft_comparison(metrics_list, filename=f"{report_name}_ttft.png")
            if path:
                generated["ttft_comparison"] = str(path)

            if HAS_SEABORN:
                path = self.plot_metrics_heatmap(
                    metrics_list,
                    metric="latency_mean",
                    title="Latency Heatmap (ms)",
                    filename=f"{report_name}_heatmap.png"
                )
                if path:
                    generated["metrics_heatmap"] = str(path)

        # Generate tables
        path = self.generate_3gpp_table(metrics_list, f"reports/{report_name}_tables.md")
        generated["markdown_tables"] = str(path)

        # Generate JSON summary
        json_path = self.output_dir.parent / f"{report_name}_summary.json"
        summary = {
            "metrics": [MetricsCalculator.to_3gpp_format(m) for m in metrics_list],
            "generated_files": generated
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        generated["json_summary"] = str(json_path)

        return generated
