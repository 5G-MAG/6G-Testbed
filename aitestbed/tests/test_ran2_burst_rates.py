from datetime import datetime

import pytest
from matplotlib.axes import Axes

from analysis.ran2_metrics import compute_ran2_metrics
from generate_charts import (
    _generate_per_scenario_ran2_subcharts,
    generate_ran2_q2_burst_peak_rate_distribution_chart,
)
from netemu.pcap import PacketRecord, PcapAnalyzer, PcapMetrics


@pytest.fixture
def capture():
    sample = PcapMetrics("capture_eth0_20261124_120000.pcap", capture_duration=2.005)
    sample.packets = [
        PacketRecord(0.0, 1000, "ul"),
        PacketRecord(0.0, 1000, "dl"),
        PacketRecord(1.0, 1000, "ul"),
        PacketRecord(1.005, 1000, "ul"),
        PacketRecord(2.0, 500, "ul"),
        PacketRecord(2.005, 500, "ul"),
        PacketRecord(2.005, 250, "dl"),
        PacketRecord(2.005, 250, "dl"),
    ]
    PcapAnalyzer._compute_per_direction_and_multi_window(sample)
    return sample


def report(capture):
    record = {
        "timestamp": datetime(2026, 11, 24, 12).timestamp(),
        "scenario_id": "chat_basic",
        "network_profile": "5g_urban",
    }
    return compute_ran2_metrics([record], [capture])


@pytest.mark.parametrize("legacy", [False, True], ids=["current", "legacy"])
def test_rate_distributions_exclude_zero_duration_bursts(capture, legacy):
    if legacy:
        for directions in capture.bursts_by_gap.values():
            for bursts in directions.values():
                for burst in bursts:
                    del burst["mean_rate_bps"]

    q2 = report(capture)["Q2"]
    per_capture = q2["per_pcap"][0]["burst_stats_by_gap"]
    pooled = q2["per_scenario_profile_bursts"]["chat_basic/5g_urban"]["by_gap"]
    for by_gap in (per_capture, pooled):
        for gap in ("10ms", "100ms"):
            ul, dl = by_gap[gap]["ul"], by_gap[gap]["dl"]
            assert ul["count"] == ul["size_bytes"]["n"] == ul["duration_sec"]["n"] == 3
            assert ul["mean_rate_mbps"]["n"] == 2
            assert ul["mean_rate_mbps"]["p95"] == pytest.approx(3.12)
            assert dl["count"] == dl["size_bytes"]["n"] == dl["duration_sec"]["n"] == 2
            assert dl["mean_rate_mbps"]["n"] == 0
            assert dl["mean_rate_mbps"]["p95"] is None


def test_distribution_chart_reads_mean_rate_at_the_existing_path(capture, tmp_path):
    result = report(capture)
    path = generate_ran2_q2_burst_peak_rate_distribution_chart(result, tmp_path)
    expected = tmp_path / "ran2_q2_burst_peak_rate_distribution.png"
    assert path == str(expected)
    assert expected.read_bytes().startswith(b"\x89PNG")


def test_cached_rate_distribution_requires_positive_durations(capture, tmp_path):
    result = report(capture)
    sides = result["Q2"]["per_scenario_profile_bursts"]["chat_basic/5g_urban"]["by_gap"]["100ms"]
    for side in sides.values():
        side["peak_rate_mbps"] = side.pop("mean_rate_mbps")

    chart = generate_ran2_q2_burst_peak_rate_distribution_chart
    assert chart(result, tmp_path) is None
    sides["ul"]["duration_sec"]["min"] = 0.005
    assert chart(result, tmp_path) is not None


def test_per_scenario_plot_uses_the_report_p95_and_gaps(capture, tmp_path, monkeypatch):
    monkeypatch.setattr(PcapAnalyzer, "analyze", lambda self, path, bucket_sec: capture)
    plotted = {"UL mean-rate p95": [], "DL mean-rate p95": []}
    plot = Axes.plot

    def record_plot(ax, x, y, **kwargs):
        if kwargs.get("label") in plotted:
            plotted[kwargs["label"]].extend(y)
        return plot(ax, x, y, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)
    _generate_per_scenario_ran2_subcharts(
        "chat_basic", [(capture.pcap_file, "5g_urban")], tmp_path
    )
    assert plotted["UL mean-rate p95"] == pytest.approx([3.12, 3.12])
    assert plotted["DL mean-rate p95"] == [None, None]
