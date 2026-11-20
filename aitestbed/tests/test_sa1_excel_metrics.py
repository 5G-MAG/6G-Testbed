import numpy as np
import pytest
from sa1_excel_metrics import (
    PACKET_DTYPE,
    burst_slices,
    directional_metrics,
    first_delivery,
    read_packets,
    window_rates,
)


def packet(t, seq, size, flags=16):
    return (t, size + 20, size, 0, seq, flags)


def test_strict_gap_and_singleton_rates():
    times = np.array([0, 10000, 20001])
    assert burst_slices(times, 10) == [(0, 2), (2, 3)]
    packets = np.array([packet(t, 1, 100) for t in times], dtype=PACKET_DTYPE)
    metrics = directional_metrics(packets, 0, 20001)
    bursts = [
        r
        for r in metrics["burst_details"]
        if r["population"] == "tcp_data" and r["idle_gap_ms"] == 10
    ]
    assert len(bursts) == 2
    assert bursts[0]["duration_ms"] == 10
    assert bursts[0]["mean_rate_mbps"] == pytest.approx(240 * 0.008 / 10)
    assert bursts[1]["duration_ms"] == 0 and bursts[1]["mean_rate_mbps"] is None
    assert bursts[1]["peak_1ms_mbps"] == pytest.approx(120 * 0.008)


def test_window_idle_fraction_includes_zero_bins_and_pads_last():
    result = window_rates(np.array([0, 2500]), np.array([1000, 1000]), 1, 0, 2500)
    assert result["window_count"] == 3
    assert result["grid_span_ms"] == 3
    assert result["active_fraction"] == pytest.approx(2 / 3)
    assert result["mean_all_windows_mbps"] == pytest.approx(16 / 3)
    assert result["mean_active_windows_mbps"] == 8
    assert result["peak_to_mean_all"] == pytest.approx(1.5)


def test_first_delivery_includes_recovery_and_resegmented_receipt():
    src = np.array(
        [
            packet(0, 100, 0, 2),
            packet(500, 101, 8),
            packet(1000, 109, 10),
            packet(10000, 109, 10),
        ],
        dtype=PACKET_DTYPE,
    )
    dst = np.array(
        [packet(2000, 109, 5), packet(11000, 114, 5), packet(20000, 109, 10)],
        dtype=PACKET_DTYPE,
    )
    d = first_delivery(src, dst, 10)
    assert len(d["pieces"]) == 1
    assert d["pieces"][0]["latency_ms"] == 10
    assert all(
        r["unique_body_bytes"] == 10 and r["transfer_latency_ms"] == 10
        for r in d["clusters"]
    )
    d = first_delivery(src, dst[:1], 10)
    assert not d["pieces"][0]["complete"] and d["pieces"][0]["latency_ms"] is None
    assert all(
        not r["complete"] and r["transfer_latency_ms"] is None for r in d["clusters"]
    )


def test_source_clustering_remains_fixed_when_receiver_gaps_differ():
    src = np.array(
        [
            packet(0, 100, 0, 2),
            packet(1, 101, 8),
            packet(1000, 109, 5),
            packet(21000, 114, 5),
        ],
        dtype=PACKET_DTYPE,
    )
    dst = np.array([packet(31000, 109, 5), packet(32000, 114, 5)], dtype=PACKET_DTYPE)
    clusters = first_delivery(src, dst, 10)["clusters"]
    at10 = [r for r in clusters if r["idle_gap_ms"] == 10]
    assert len(at10) == 2
    assert [r["transfer_latency_ms"] for r in at10] == [30, 11]
    at50 = [r for r in clusters if r["idle_gap_ms"] == 50]
    assert len(at50) == 1 and at50[0]["transfer_latency_ms"] == 31


def test_same_timestamp_packets_keep_capture_order(tmp_path):
    import dpkt
    import socket

    path = tmp_path / "same-timestamp.pcap"
    with path.open("wb") as stream:
        writer = dpkt.pcap.Writer(stream)
        # The smaller last segment must not sort ahead of the earlier segment.
        for seq, size in [(100, 1400), (1500, 80)]:
            tcp = dpkt.tcp.TCP(
                sport=40000, dport=8098, seq=seq, flags=16, data=b"x" * size
            )
            tcp.off = 5
            ip = dpkt.ip.IP(
                src=socket.inet_aton("10.0.0.1"),
                dst=socket.inet_aton("10.0.0.2"),
                p=6,
                data=tcp,
            )
            ip.len = len(ip)
            writer.writepkt(bytes(dpkt.ethernet.Ethernet(type=2048, data=ip)), ts=1)
    packets, diagnostics = read_packets(path)
    assert list(packets["seq"]) == [100, 1500]
    assert diagnostics["malformed_packet_records"] == 0
