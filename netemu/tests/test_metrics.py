"""Tests for netemu.metrics (aggregation of packet-level metrics).

Captures are synthesized with dpkt so the tests run the real parse path and
then the real aggregation path, without shipping binary fixtures.
"""

import math

import pytest

dpkt = pytest.importorskip("dpkt")

from netemu.metrics import (  # noqa: E402
    Distribution,
    PacketMetricsReport,
    compute_packet_metrics,
    summarize,
)
from netemu.pcap import PcapAnalyzer, PcapMetrics  # noqa: E402

CLIENT_IP = "10.0.0.2"
SERVER_IP = "93.184.216.34"
SERVER_IP2 = "93.184.216.35"
CLIENT_PORT = 51000
SERVER_PORT = 443


def _ip_bytes(addr):
    return bytes(int(o) for o in addr.split("."))


def _frame(src, dst, sport, dport, flags=0, seq=0, ack=0, payload=b""):
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, seq=seq, ack=ack,
                       flags=flags, win=65535, data=payload)
    tcp.off = 5
    ip = dpkt.ip.IP(src=_ip_bytes(src), dst=_ip_bytes(dst),
                    p=dpkt.ip.IP_PROTO_TCP, data=tcp)
    ip.len = len(bytes(ip))
    eth = dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x00" * 6,
                                 type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def _write(path, packets):
    with open(path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        for ts, buf in packets:
            writer.writepkt(buf, ts=ts)
    return str(path)


def _exchange_capture(path, server=SERVER_IP, port=CLIENT_PORT, exchanges=2, t0=1000.0):
    """Handshake, then N request/response payload pairs, then FIN."""
    syn, syn_ack, ack, psh_ack, fin_ack = (
        dpkt.tcp.TH_SYN,
        dpkt.tcp.TH_SYN | dpkt.tcp.TH_ACK,
        dpkt.tcp.TH_ACK,
        dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
        dpkt.tcp.TH_FIN | dpkt.tcp.TH_ACK,
    )
    pkts = [
        (t0 + 0.000, _frame(CLIENT_IP, server, port, SERVER_PORT, syn, 0, 0)),
        (t0 + 0.020, _frame(server, CLIENT_IP, SERVER_PORT, port, syn_ack, 5000, 1)),
        (t0 + 0.021, _frame(CLIENT_IP, server, port, SERVER_PORT, ack, 1, 5001)),
    ]
    t = t0 + 0.030
    cseq, sseq = 1, 5001
    for _ in range(exchanges):
        pkts.append((t, _frame(CLIENT_IP, server, port, SERVER_PORT, psh_ack,
                               cseq, sseq, b"q" * 100)))
        cseq += 100
        t += 0.010
        pkts.append((t, _frame(server, CLIENT_IP, SERVER_PORT, port, psh_ack,
                               sseq, cseq, b"r" * 500)))
        sseq += 500
        t += 0.010
    pkts.append((t, _frame(CLIENT_IP, server, port, SERVER_PORT, fin_ack, cseq, sseq)))
    return _write(path, pkts)


@pytest.fixture
def two_captures(tmp_path):
    a = _exchange_capture(tmp_path / "a.pcap", exchanges=2, t0=1000.0)
    b = _exchange_capture(tmp_path / "b.pcap", server=SERVER_IP2,
                          port=CLIENT_PORT + 1, exchanges=1, t0=2000.0)
    analyzer = PcapAnalyzer(target_ports=[SERVER_PORT])
    return [analyzer.analyze(a), analyzer.analyze(b)]


# ---------------------------------------------------------------- summarize
class TestSummarize:
    def test_empty_sample_is_reported_as_empty_not_zero(self):
        d = summarize([])
        assert d.n == 0
        assert d.p50 is None and d.mean is None

    def test_percentiles_are_observed_values(self):
        d = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        assert d.p50 == 3.0
        assert d.minimum == 1.0 and d.maximum == 5.0
        assert d.n == 5

    def test_non_finite_values_are_dropped(self):
        d = summarize([1.0, float("nan"), 3.0, float("inf")])
        assert d.n == 2
        assert d.mean == 2.0

    def test_cv_is_stdev_over_mean(self):
        d = summarize([10.0, 10.0, 10.0])
        assert d.stdev == 0.0
        assert d.cv == 0.0

    def test_single_value_has_zero_spread(self):
        d = summarize([7.5])
        assert d.n == 1 and d.p50 == 7.5 and d.stdev == 0.0

    def test_to_dict_is_json_shaped(self):
        keys = set(summarize([1.0]).to_dict())
        assert keys == {"n", "mean", "min", "max", "p50", "p95", "p99", "stdev", "cv"}


# ------------------------------------------------------------------- shape
class TestReportShape:
    def test_accepts_a_single_capture(self, two_captures):
        report = compute_packet_metrics(two_captures[0])
        assert report.captures == 1

    def test_accepts_an_iterable(self, two_captures):
        assert compute_packet_metrics(two_captures).captures == 2

    def test_accepts_a_generator(self, two_captures):
        assert compute_packet_metrics(iter(two_captures)).captures == 2

    def test_empty_input_returns_empty_report_not_error(self):
        report = compute_packet_metrics([])
        assert isinstance(report, PacketMetricsReport)
        assert report.captures == 0
        assert report.connection.tcp_handshake_rtt_ms.n == 0

    def test_families_present_even_without_data(self):
        """Report shape must not depend on what a capture happened to contain."""
        report = compute_packet_metrics([PcapMetrics(pcap_file="empty.pcap")])
        d = report.to_dict()
        assert set(d) >= {"connection", "direction", "bursts", "throughput",
                          "reliability", "capture_duration_s"}

    def test_to_dict_is_json_serializable(self, two_captures):
        import json
        json.dumps(compute_packet_metrics(two_captures).to_dict())


# -------------------------------------------------------------- connection
class TestConnectionMetrics:
    def test_handshake_rtt_recovered_in_milliseconds(self, two_captures):
        d = compute_packet_metrics(two_captures).connection.tcp_handshake_rtt_ms
        assert d.n == 2
        assert math.isclose(d.p50, 20.0, abs_tol=1.0)

    def test_connection_setup_spans_syn_to_first_payload(self, two_captures):
        d = compute_packet_metrics(two_captures).connection.connection_setup_ms
        # SYN at t0, first client payload at t0+30 ms.
        assert math.isclose(d.p50, 30.0, abs_tol=2.0)

    def test_setup_is_at_least_handshake_rtt(self, two_captures):
        conn = compute_packet_metrics(two_captures).connection
        assert conn.connection_setup_ms.p50 >= conn.tcp_handshake_rtt_ms.p50

    def test_flows_counted_across_captures(self, two_captures):
        conn = compute_packet_metrics(two_captures).connection
        assert conn.flows == 2
        assert conn.flows_per_capture.n == 2

    def test_exchanges_counted_from_payload_direction_changes(self, two_captures):
        """Two request/response pairs in one capture, one in the other."""
        d = compute_packet_metrics(two_captures).connection.exchanges_per_flow
        assert sorted([d.minimum, d.maximum]) == [1.0, 2.0]

    def test_reuse_ratio_counts_multi_exchange_flows(self, two_captures):
        conn = compute_packet_metrics(two_captures).connection
        assert conn.reuse_ratio == pytest.approx(0.5)

    def test_distinct_destinations_per_capture(self, two_captures):
        d = compute_packet_metrics(two_captures).connection
        assert d.distinct_destinations_per_capture.maximum == 1.0

    def test_flow_duration_is_positive(self, two_captures):
        assert compute_packet_metrics(two_captures).connection.flow_duration_s.p50 > 0


# --------------------------------------------------------------- direction
class TestDirectionMetrics:
    def test_downlink_heavier_than_uplink_for_this_workload(self, two_captures):
        d = compute_packet_metrics(two_captures).direction
        assert d.dl_bytes > d.ul_bytes
        assert 0 < d.ul_dl_byte_ratio < 1

    def test_packet_size_distributions_per_direction(self, two_captures):
        d = compute_packet_metrics(two_captures).direction
        assert d.ul_packet_size.n > 0 and d.dl_packet_size.n > 0
        assert d.dl_packet_size.maximum > d.ul_packet_size.maximum

    def test_inter_packet_gaps_are_milliseconds(self, two_captures):
        d = compute_packet_metrics(two_captures).direction
        assert d.dl_inter_packet_gap_ms.n > 0
        assert d.dl_inter_packet_gap_ms.minimum >= 0

    def test_ratio_is_none_when_no_downlink(self, tmp_path):
        syn = dpkt.tcp.TH_SYN
        path = _write(tmp_path / "ul.pcap",
                      [(1.0, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT,
                                    SERVER_PORT, syn))])
        m = PcapAnalyzer(target_ports=[SERVER_PORT]).analyze(path)
        assert compute_packet_metrics([m]).direction.ul_dl_byte_ratio is None


# ---------------------------------------------------- throughput and bursts
class TestThroughputAndBursts:
    def test_windowed_peaks_are_summarized_per_label(self, two_captures):
        thr = compute_packet_metrics(two_captures).throughput
        assert thr.peak_mbps_by_window, "expected per-window peaks"
        for label, dist in thr.peak_mbps_by_window.items():
            assert dist["n"] >= 1

    def test_burstiness_is_at_least_one(self, two_captures):
        thr = compute_packet_metrics(two_captures).throughput
        for dist in thr.burstiness_by_window.values():
            if dist["min"] is not None:
                assert dist["min"] >= 1.0 - 1e-9

    def test_bursts_keyed_by_gap_then_direction(self, two_captures):
        by_gap = compute_packet_metrics(two_captures).bursts.by_gap
        assert by_gap, "expected burst segmentation"
        label = next(iter(by_gap))
        for direction, stats in by_gap[label].items():
            assert direction in {"ul", "dl"}
            assert {"count", "size_bytes", "duration_s"} <= set(stats)


# ------------------------------------------------------------- reliability
class TestReliabilityMetrics:
    def test_clean_capture_has_no_retransmissions(self, two_captures):
        rel = compute_packet_metrics(two_captures).reliability
        assert rel.total_retransmissions == 0
        assert rel.flow_retransmission_ratio == 0.0

    def test_retransmission_is_counted_and_rated(self, tmp_path):
        psh_ack = dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK
        payload = b"z" * 200
        pkts = [
            (1.0, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                         psh_ack, 1, 1, payload)),
            (1.5, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                         psh_ack, 1, 1, payload)),   # same seq range again
        ]
        m = PcapAnalyzer(target_ports=[SERVER_PORT]).analyze(
            _write(tmp_path / "rx.pcap", pkts))
        rel = compute_packet_metrics([m]).reliability
        assert rel.total_retransmissions >= 1
        assert rel.flows_with_retransmissions == 1
        assert rel.flow_retransmission_ratio == 1.0


# ------------------------------------------------------------- aggregation
class TestAggregationSemantics:
    def test_totals_sum_across_captures(self, two_captures):
        report = compute_packet_metrics(two_captures)
        assert report.total_packets == sum(c.total_packets for c in two_captures)
        assert report.total_bytes == sum(c.total_bytes for c in two_captures)

    def test_per_capture_distribution_has_one_sample_per_capture(self, two_captures):
        report = compute_packet_metrics(two_captures)
        assert report.capture_duration_s.n == 2
        assert report.connection.flows_per_capture.n == 2

    def test_flow_pooled_distribution_pools_all_flows(self, two_captures):
        report = compute_packet_metrics(two_captures)
        assert report.connection.tcp_handshake_rtt_ms.n == report.connection.flows

    def test_module_does_not_reference_external_taxonomies(self):
        """netemu stays standard-neutral: no working-group vocabulary."""
        import re
        from pathlib import Path
        import netemu.metrics as mod
        text = Path(mod.__file__).read_text()
        assert not re.search(r"RAN2|S4-\d+|\bQ\d\.\d", text)


class TestSampleCollector:
    """collect_connection_samples is the single extraction path."""

    def test_summarized_samples_match_the_report(self, two_captures):
        from netemu.metrics import collect_connection_samples
        samples = collect_connection_samples(two_captures)
        report = compute_packet_metrics(two_captures)
        for key in ("tcp_handshake_rtt_ms", "tls_handshake_ms",
                    "connection_setup_ms", "flow_duration_s",
                    "exchanges_per_flow", "flows_per_capture"):
            assert summarize(samples[key]).to_dict() == \
                getattr(report.connection, key).to_dict(), key

    def test_returns_all_keys_for_empty_input(self):
        from netemu.metrics import collect_connection_samples
        samples = collect_connection_samples([])
        assert samples and all(v == [] for v in samples.values())

    def test_accepts_a_single_capture(self, two_captures):
        from netemu.metrics import collect_connection_samples
        s = collect_connection_samples(two_captures[0])
        assert s["flows_per_capture"] == [1.0]
