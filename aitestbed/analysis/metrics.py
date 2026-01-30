"""
Metrics Calculator for the 6G AI Traffic Testbed.

Computes 3GPP-aligned metrics from traffic logs.
"""

import json
import statistics
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ScenarioMetrics:
    """
    Computed metrics for a scenario/profile combination.

    Aligned with 3GPP TR 26.998 metrics requirements.
    """
    scenario_id: str
    network_profile: str
    sample_count: int = 0

    # Latency metrics (seconds)
    latency_mean: float = 0.0
    latency_median: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_std: float = 0.0

    # Time to First Token (TTFT) - seconds
    ttft_mean: Optional[float] = None
    ttft_median: Optional[float] = None
    ttft_p95: Optional[float] = None
    ttft_p99: Optional[float] = None
    ttft_p50: Optional[float] = None

    # Time to Last Token (TTLT) - seconds
    ttlt_mean: Optional[float] = None
    ttlt_median: Optional[float] = None
    ttlt_p95: Optional[float] = None
    ttlt_p99: Optional[float] = None
    ttlt_p50: Optional[float] = None

    # Traffic volume (bytes)
    request_bytes_mean: float = 0.0
    response_bytes_mean: float = 0.0
    total_bytes_mean: float = 0.0

    # UL/DL ratio (3GPP Section b)
    ul_dl_ratio_mean: float = 0.0

    # Token metrics
    tokens_in_mean: Optional[float] = None
    tokens_out_mean: Optional[float] = None
    token_rate_mean: Optional[float] = None  # tokens/sec

    # Success metrics (3GPP Section a - QoE)
    success_rate: float = 100.0
    error_count: int = 0

    # Agent metrics (3GPP Section d)
    tool_calls_mean: float = 0.0
    loop_factor: float = 1.0  # API calls per user prompt
    tool_latency_mean: float = 0.0
    tool_latency_p50: Optional[float] = None
    tool_latency_p95: Optional[float] = None
    tool_latency_p99: Optional[float] = None

    # Computer use metrics
    computer_actions_mean: float = 0.0
    computer_action_latency_mean: float = 0.0
    computer_action_latency_p95: Optional[float] = None
    computer_screenshot_bytes_mean: Optional[float] = None
    computer_steps_mean: Optional[float] = None
    computer_action_error_rate: Optional[float] = None

    # Streaming metrics
    streaming_rate_mean: Optional[float] = None  # bytes/sec
    chunk_count_mean: Optional[float] = None

    # Streaming stall metrics
    stall_gap_threshold_sec: float = 0.0
    stall_rate: Optional[float] = None
    stall_event_count: int = 0
    stall_duration_mean: Optional[float] = None
    stall_duration_p95: Optional[float] = None
    stall_duration_p99: Optional[float] = None

    # Error taxonomy
    error_breakdown: dict[str, int] = field(default_factory=dict)

    # Burstiness descriptors
    burst_peak_to_mean: Optional[float] = None
    burst_cv: Optional[float] = None
    burst_gap_threshold_sec: float = 0.0
    burst_on_gap_mean_sec: Optional[float] = None
    burst_off_gap_mean_sec: Optional[float] = None
    burst_on_gap_count: int = 0
    burst_off_gap_count: int = 0


class MetricsCalculator:
    """
    Calculate metrics from traffic log records.
    """
    DEFAULT_STALL_GAP_SEC = 1.0
    DEFAULT_BURST_GAP_SEC = 1.0

    @staticmethod
    def calculate(
        records: list[dict],
        scenario_id: str,
        network_profile: str,
        stall_gap_sec: Optional[float] = None,
        burst_gap_sec: Optional[float] = None,
    ) -> ScenarioMetrics:
        """
        Calculate metrics from a list of log records.

        Args:
            records: List of log records (as dicts)
            scenario_id: Scenario identifier
            network_profile: Network profile used

        Returns:
            ScenarioMetrics with all computed values
        """
        stall_gap_sec = stall_gap_sec if stall_gap_sec is not None else MetricsCalculator.DEFAULT_STALL_GAP_SEC
        burst_gap_sec = burst_gap_sec if burst_gap_sec is not None else MetricsCalculator.DEFAULT_BURST_GAP_SEC

        if not records:
            return ScenarioMetrics(
                scenario_id=scenario_id,
                network_profile=network_profile,
                stall_gap_threshold_sec=stall_gap_sec,
                burst_gap_threshold_sec=burst_gap_sec,
            )

        def _parse_json_field(value, default):
            if value is None or value == "":
                return default
            if isinstance(value, (list, dict)):
                return value
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return default

        def _metadata(record: dict) -> dict:
            return _parse_json_field(record.get("metadata"), {})

        def _is_tool_record(record: dict) -> bool:
            metadata = _metadata(record)
            return metadata.get("type") == "mcp_tool_call"

        def _is_computer_action_record(record: dict) -> bool:
            metadata = _metadata(record)
            return metadata.get("type") == "computer_use_action"

        def _is_computer_response_record(record: dict) -> bool:
            metadata = _metadata(record)
            return metadata.get("type") == "computer_use_response"

        def _is_connection_record(record: dict) -> bool:
            if record.get("turn_index", 0) < 0:
                return True
            metadata = _metadata(record)
            event_type = metadata.get("event_type", "")
            return event_type.endswith("_connect")

        primary_records = [
            r for r in records
            if not _is_tool_record(r)
            and not _is_computer_action_record(r)
            and not _is_connection_record(r)
        ]
        tool_records = [r for r in records if _is_tool_record(r)]
        computer_action_records = [r for r in records if _is_computer_action_record(r)]
        computer_response_records = [r for r in records if _is_computer_response_record(r)]

        metrics = ScenarioMetrics(
            scenario_id=scenario_id,
            network_profile=network_profile,
            sample_count=len(primary_records),
            stall_gap_threshold_sec=stall_gap_sec,
            burst_gap_threshold_sec=burst_gap_sec,
        )

        # Extract values
        latencies = [r["latency_sec"] for r in primary_records if r.get("latency_sec")]
        request_bytes = [r["request_bytes"] for r in primary_records if r.get("request_bytes")]
        response_bytes = [r["response_bytes"] for r in primary_records if r.get("response_bytes")]
        tokens_in = [r["tokens_in"] for r in primary_records if r.get("tokens_in")]
        tokens_out = [r["tokens_out"] for r in primary_records if r.get("tokens_out")]
        tool_calls = [r["tool_calls_count"] for r in primary_records if r.get("tool_calls_count") is not None]

        # Calculate TTFT (Time to First Token)
        ttft_values = []
        for r in primary_records:
            if r.get("t_first_token") and r.get("t_request_start"):
                ttft = r["t_first_token"] - r["t_request_start"]
                if ttft >= 0:
                    ttft_values.append(ttft)

        # Calculate TTLT (Time to Last Token)
        ttlt_values = []
        for r in primary_records:
            if r.get("t_last_token") and r.get("t_request_start"):
                ttlt = r["t_last_token"] - r["t_request_start"]
                if ttlt >= 0:
                    ttlt_values.append(ttlt)

        # Latency statistics
        if latencies:
            metrics.latency_mean = statistics.mean(latencies)
            metrics.latency_median = statistics.median(latencies)
            metrics.latency_min = min(latencies)
            metrics.latency_max = max(latencies)
            if len(latencies) > 1:
                metrics.latency_std = statistics.stdev(latencies)
            metrics.latency_p95 = MetricsCalculator._percentile(latencies, 95)
            metrics.latency_p99 = MetricsCalculator._percentile(latencies, 99)

        # TTFT statistics
        if ttft_values:
            metrics.ttft_mean = statistics.mean(ttft_values)
            metrics.ttft_median = statistics.median(ttft_values)
            metrics.ttft_p95 = MetricsCalculator._percentile(ttft_values, 95)
            metrics.ttft_p99 = MetricsCalculator._percentile(ttft_values, 99)
            metrics.ttft_p50 = MetricsCalculator._percentile(ttft_values, 50)

        # TTLT statistics
        if ttlt_values:
            metrics.ttlt_mean = statistics.mean(ttlt_values)
            metrics.ttlt_median = statistics.median(ttlt_values)
            metrics.ttlt_p95 = MetricsCalculator._percentile(ttlt_values, 95)
            metrics.ttlt_p99 = MetricsCalculator._percentile(ttlt_values, 99)
            metrics.ttlt_p50 = MetricsCalculator._percentile(ttlt_values, 50)

        # Traffic volume statistics
        if request_bytes:
            metrics.request_bytes_mean = statistics.mean(request_bytes)
        if response_bytes:
            metrics.response_bytes_mean = statistics.mean(response_bytes)
        metrics.total_bytes_mean = metrics.request_bytes_mean + metrics.response_bytes_mean

        # UL/DL ratio
        if metrics.response_bytes_mean > 0:
            metrics.ul_dl_ratio_mean = metrics.request_bytes_mean / metrics.response_bytes_mean

        # Token statistics
        if tokens_in:
            metrics.tokens_in_mean = statistics.mean(tokens_in)
        if tokens_out:
            metrics.tokens_out_mean = statistics.mean(tokens_out)

        # Token rate (tokens per second)
        token_rates = []
        for r in primary_records:
            if r.get("tokens_out") and r.get("latency_sec") and r["latency_sec"] > 0:
                token_rates.append(r["tokens_out"] / r["latency_sec"])
        if token_rates:
            metrics.token_rate_mean = statistics.mean(token_rates)

        # Success rate
        success_count = sum(1 for r in primary_records if r.get("success", True))
        metrics.success_rate = (success_count / len(primary_records)) * 100 if primary_records else 100.0
        metrics.error_count = len(primary_records) - success_count

        # Agent/Tool metrics
        if tool_calls:
            metrics.tool_calls_mean = statistics.mean(tool_calls)

        # Loop factor (total API calls / user prompts)
        # Approximate by counting records per session
        sessions = {}
        for r in primary_records:
            sid = r.get("session_id", "")
            if sid not in sessions:
                sessions[sid] = 0
            sessions[sid] += 1

        if sessions:
            metrics.loop_factor = statistics.mean(sessions.values())

        tool_latencies = [
            r["tool_latency_sec"] for r in tool_records
            if r.get("tool_latency_sec", 0) > 0
        ]
        if tool_latencies:
            metrics.tool_latency_mean = statistics.mean(tool_latencies)
            metrics.tool_latency_p50 = MetricsCalculator._percentile(tool_latencies, 50)
            metrics.tool_latency_p95 = MetricsCalculator._percentile(tool_latencies, 95)
            metrics.tool_latency_p99 = MetricsCalculator._percentile(tool_latencies, 99)

        if computer_action_records:
            action_latencies = [
                r.get("latency_sec", 0.0) for r in computer_action_records
                if r.get("latency_sec", 0.0) > 0
            ]
            if action_latencies:
                metrics.computer_action_latency_mean = statistics.mean(action_latencies)
                metrics.computer_action_latency_p95 = MetricsCalculator._percentile(action_latencies, 95)

            screenshot_bytes = []
            for r in computer_action_records:
                metadata = _metadata(r)
                bytes_value = metadata.get("screenshot_bytes")
                if bytes_value is None:
                    bytes_value = r.get("response_bytes")
                if bytes_value:
                    screenshot_bytes.append(bytes_value)
            if screenshot_bytes:
                metrics.computer_screenshot_bytes_mean = statistics.mean(screenshot_bytes)

            action_counts = {}
            for r in computer_action_records:
                sid = r.get("session_id", "")
                action_counts[sid] = action_counts.get(sid, 0) + 1
            if action_counts:
                metrics.computer_actions_mean = statistics.mean(action_counts.values())

            error_count = sum(1 for r in computer_action_records if not r.get("success", True))
            metrics.computer_action_error_rate = error_count / len(computer_action_records)

        if computer_response_records:
            step_counts = {}
            for r in computer_response_records:
                sid = r.get("session_id", "")
                step_counts[sid] = step_counts.get(sid, 0) + 1
            if step_counts:
                metrics.computer_steps_mean = statistics.mean(step_counts.values())

        # Streaming metrics
        streaming_records = [r for r in primary_records if r.get("is_streaming")]
        if streaming_records:
            chunk_counts = [r["chunk_count"] for r in streaming_records if r.get("chunk_count")]
            if chunk_counts:
                metrics.chunk_count_mean = statistics.mean(chunk_counts)

            # Calculate streaming rate
            streaming_rates = []
            for r in streaming_records:
                if r.get("response_bytes") and r.get("t_last_token") and r.get("t_first_token"):
                    duration = r["t_last_token"] - r["t_first_token"]
                    if duration > 0:
                        streaming_rates.append(r["response_bytes"] / duration)
            if streaming_rates:
                metrics.streaming_rate_mean = statistics.mean(streaming_rates)

        # Streaming stall metrics
        gap_count = 0
        stall_durations = []
        for r in streaming_records:
            gaps = _parse_json_field(r.get("inter_chunk_times"), [])
            if not gaps:
                continue
            gap_count += len(gaps)
            for gap in gaps:
                if gap >= stall_gap_sec:
                    stall_durations.append(gap)
        metrics.stall_event_count = len(stall_durations)
        if gap_count > 0:
            metrics.stall_rate = len(stall_durations) / gap_count
        if stall_durations:
            metrics.stall_duration_mean = statistics.mean(stall_durations)
            metrics.stall_duration_p95 = MetricsCalculator._percentile(stall_durations, 95)
            metrics.stall_duration_p99 = MetricsCalculator._percentile(stall_durations, 99)

        # Error taxonomy
        error_breakdown = {
            "timeout": 0,
            "rate_limited": 0,
            "server_error": 0,
            "tool_failure": 0,
            "other": 0,
        }
        for r in records:
            if r.get("success", True):
                continue
            if _is_tool_record(r):
                error_breakdown["tool_failure"] += 1
                continue
            if _is_computer_action_record(r):
                error_breakdown["tool_failure"] += 1
                continue

            status = r.get("http_status")
            error_type = (r.get("error_type") or "").lower()

            if "timeout" in error_type or "timed out" in error_type:
                error_breakdown["timeout"] += 1
            elif status == 429 or "rate limit" in error_type or "429" in error_type:
                error_breakdown["rate_limited"] += 1
            elif isinstance(status, int) and status >= 500:
                error_breakdown["server_error"] += 1
            else:
                error_breakdown["other"] += 1

        metrics.error_breakdown = error_breakdown

        # Burstiness descriptors
        total_bytes = [
            (r.get("request_bytes") or 0) + (r.get("response_bytes") or 0)
            for r in primary_records
        ]
        if total_bytes:
            mean_bytes = statistics.mean(total_bytes)
            if mean_bytes > 0:
                metrics.burst_peak_to_mean = max(total_bytes) / mean_bytes
                if len(total_bytes) > 1:
                    metrics.burst_cv = statistics.stdev(total_bytes) / mean_bytes

        request_times = sorted(
            r["t_request_start"] for r in primary_records
            if r.get("t_request_start")
        )
        if len(request_times) > 1:
            gaps = [
                request_times[i] - request_times[i - 1]
                for i in range(1, len(request_times))
                if request_times[i] >= request_times[i - 1]
            ]
            on_gaps = [g for g in gaps if g <= burst_gap_sec]
            off_gaps = [g for g in gaps if g > burst_gap_sec]
            metrics.burst_on_gap_count = len(on_gaps)
            metrics.burst_off_gap_count = len(off_gaps)
            if on_gaps:
                metrics.burst_on_gap_mean_sec = statistics.mean(on_gaps)
            if off_gaps:
                metrics.burst_off_gap_mean_sec = statistics.mean(off_gaps)

        return metrics

    @staticmethod
    def _percentile(values: list[float], p: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f

        if f == c:
            return sorted_values[f]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    @staticmethod
    def compare_profiles(
        metrics_list: list[ScenarioMetrics]
    ) -> dict:
        """
        Compare metrics across different network profiles.

        Returns:
            Dictionary with comparison data suitable for reporting
        """
        if not metrics_list:
            return {}

        comparison = {
            "scenario_id": metrics_list[0].scenario_id,
            "profiles": {},
            "summary": {}
        }

        for m in metrics_list:
            comparison["profiles"][m.network_profile] = {
                "latency_mean": m.latency_mean,
                "latency_p95": m.latency_p95,
                "ttft_mean": m.ttft_mean,
                "success_rate": m.success_rate,
                "ul_dl_ratio": m.ul_dl_ratio_mean,
                "loop_factor": m.loop_factor,
            }

        # Calculate deltas from baseline (first profile)
        baseline = metrics_list[0]
        for m in metrics_list[1:]:
            if baseline.latency_mean > 0:
                latency_delta = ((m.latency_mean - baseline.latency_mean) / baseline.latency_mean) * 100
            else:
                latency_delta = 0

            comparison["profiles"][m.network_profile]["latency_delta_pct"] = latency_delta
            comparison["profiles"][m.network_profile]["success_delta"] = m.success_rate - baseline.success_rate

        return comparison

    @staticmethod
    def to_3gpp_format(metrics: ScenarioMetrics) -> dict:
        """
        Format metrics for 3GPP TR-style reporting.

        Returns:
            Dictionary formatted for 3GPP documentation
        """
        return {
            "scenario": metrics.scenario_id,
            "network_conditions": metrics.network_profile,
            "samples": metrics.sample_count,
            "qoe_metrics": {
                "time_to_first_token_ms": (metrics.ttft_mean or 0) * 1000,
                "time_to_first_token_p50_ms": (metrics.ttft_p50 or 0) * 1000,
                "time_to_last_token_ms": (metrics.ttlt_mean or 0) * 1000,
                "time_to_last_token_p50_ms": (metrics.ttlt_p50 or 0) * 1000,
                "response_latency_ms": metrics.latency_mean * 1000,
                "response_latency_p95_ms": metrics.latency_p95 * 1000,
                "time_to_first_token_p95_ms": (metrics.ttft_p95 or 0) * 1000,
                "time_to_first_token_p99_ms": (metrics.ttft_p99 or 0) * 1000,
                "time_to_last_token_p95_ms": (metrics.ttlt_p95 or 0) * 1000,
                "time_to_last_token_p99_ms": (metrics.ttlt_p99 or 0) * 1000,
                "success_rate_pct": metrics.success_rate,
            },
            "traffic_characteristics": {
                "uplink_bytes_avg": metrics.request_bytes_mean,
                "downlink_bytes_avg": metrics.response_bytes_mean,
                "ul_dl_ratio": metrics.ul_dl_ratio_mean,
                "token_rate_per_sec": metrics.token_rate_mean,
            },
            "ai_service_metrics": {
                "agent_loop_factor": metrics.loop_factor,
                "tool_calls_avg": metrics.tool_calls_mean,
                "tool_latency_ms": metrics.tool_latency_mean * 1000,
                "tool_latency_p50_ms": (metrics.tool_latency_p50 or 0) * 1000,
                "tool_latency_p95_ms": (metrics.tool_latency_p95 or 0) * 1000,
                "tool_latency_p99_ms": (metrics.tool_latency_p99 or 0) * 1000,
            },
            "computer_use_metrics": {
                "actions_avg": metrics.computer_actions_mean,
                "action_latency_ms": metrics.computer_action_latency_mean * 1000,
                "action_latency_p95_ms": (metrics.computer_action_latency_p95 or 0) * 1000,
                "screenshot_bytes_avg": metrics.computer_screenshot_bytes_mean,
                "steps_avg": metrics.computer_steps_mean,
                "action_error_rate": metrics.computer_action_error_rate,
            },
            "streaming_metrics": {
                "stall_gap_threshold_ms": metrics.stall_gap_threshold_sec * 1000,
                "stall_rate": metrics.stall_rate,
                "stall_event_count": metrics.stall_event_count,
                "stall_duration_mean_ms": (metrics.stall_duration_mean or 0) * 1000,
                "stall_duration_p95_ms": (metrics.stall_duration_p95 or 0) * 1000,
                "stall_duration_p99_ms": (metrics.stall_duration_p99 or 0) * 1000,
            },
            "error_taxonomy": metrics.error_breakdown,
            "burstiness": {
                "peak_to_mean": metrics.burst_peak_to_mean,
                "coefficient_of_variation": metrics.burst_cv,
                "gap_threshold_ms": metrics.burst_gap_threshold_sec * 1000,
                "on_gap_mean_ms": (metrics.burst_on_gap_mean_sec or 0) * 1000,
                "off_gap_mean_ms": (metrics.burst_off_gap_mean_sec or 0) * 1000,
                "on_gap_count": metrics.burst_on_gap_count,
                "off_gap_count": metrics.burst_off_gap_count,
            }
        }
