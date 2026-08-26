"""Tests for netemu.pcap (pcap parsing and metric extraction).

The fixtures synthesize small libpcap files with dpkt so the tests exercise
the real parse path without shipping binary captures.
"""

import struct

import pytest

dpkt = pytest.importorskip("dpkt")

from netemu.pcap import (  # noqa: E402
    PcapAnalyzer,
    PcapMetrics,
    TCPFlow,
    analyze_multiple_pcaps,
    analyze_pcap,
    merge_pcap_metrics,
)

CLIENT_IP = "10.0.0.2"
SERVER_IP = "93.184.216.34"
CLIENT_PORT = 51000
SERVER_PORT = 443


def _ip_bytes(addr: str) -> bytes:
    return bytes(int(o) for o in addr.split("."))


def _frame(src, dst, sport, dport, flags=0, seq=0, ack=0, payload=b"") -> bytes:
    """Build an Ethernet/IPv4/TCP frame."""
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, seq=seq, ack=ack,
                       flags=flags, win=65535, data=payload)
    tcp.off = 5
    ip = dpkt.ip.IP(src=_ip_bytes(src), dst=_ip_bytes(dst), p=dpkt.ip.IP_PROTO_TCP,
                    data=tcp)
    ip.len = len(bytes(ip))
    eth = dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x00" * 6,
                                 type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def _udp_frame(src, dst, sport, dport, payload=b"x" * 100) -> bytes:
    udp = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    udp.ulen = len(bytes(udp))
    ip = dpkt.ip.IP(src=_ip_bytes(src), dst=_ip_bytes(dst),
                    p=dpkt.ip.IP_PROTO_UDP, data=udp)
    ip.len = len(bytes(ip))
    eth = dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x00" * 6,
                                 type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def _write_pcap(path, packets):
    """packets: list of (timestamp, frame_bytes)."""
    with open(path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        for ts, buf in packets:
            writer.writepkt(buf, ts=ts)


@pytest.fixture
def handshake_pcap(tmp_path):
    """A TCP handshake (20 ms RTT) followed by a TLS exchange and FIN."""
    syn = dpkt.tcp.TH_SYN
    syn_ack = dpkt.tcp.TH_SYN | dpkt.tcp.TH_ACK
    ack = dpkt.tcp.TH_ACK
    fin = dpkt.tcp.TH_FIN | dpkt.tcp.TH_ACK

    client_hello = b"\x16\x03\x01" + b"\x00" * 200      # TLS handshake record
    server_app = b"\x17\x03\x03" + b"\x00" * 500        # TLS application data

    pkts = [
        (1000.000, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, syn, seq=1)),
        (1000.010, _frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, syn_ack, seq=1, ack=2)),
        (1000.020, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, ack, seq=2, ack=2)),
        (1000.021, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, ack, seq=2, ack=2,
                          payload=client_hello)),
        (1000.080, _frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT, ack, seq=2, ack=205,
                          payload=server_app)),
        (1000.100, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT, fin, seq=205, ack=505)),
    ]
    path = tmp_path / "capture_eth0.pcap"
    _write_pcap(path, pkts)
    return path


class TestBasicParsing:
    def test_counts_packets_and_flows(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap))

        assert isinstance(m, PcapMetrics)
        assert m.total_packets == 6
        assert m.tcp_packets == 6
        assert m.tcp_flows == 1
        assert len(m.packets) == 6

    def test_capture_duration(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap))

        assert m.capture_duration == pytest.approx(0.100, abs=1e-6)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_pcap(str(tmp_path / "nope.pcap"))


class TestHandshakeMetrics:
    def test_handshake_rtt(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap))
        flow = m.flows[0]

        # SYN 1000.000 -> ACK 1000.020
        assert flow.handshake_rtt == pytest.approx(0.020, abs=1e-6)
        assert flow.syn_ack_rtt == pytest.approx(0.010, abs=1e-6)

    def test_rtt_aggregates_in_ms(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap))

        assert m.rtt_mean_ms == pytest.approx(20.0, abs=1e-3)
        assert m.rtt_min_ms == pytest.approx(20.0, abs=1e-3)
        assert m.rtt_max_ms == pytest.approx(20.0, abs=1e-3)

    def test_tls_handshake_duration(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap))
        flow = m.flows[0]

        # ClientHello 1000.021 -> first ApplicationData 1000.080
        assert flow.tls_handshake_duration == pytest.approx(0.059, abs=1e-6)

    def test_lifecycle_timestamps(self, handshake_pcap):
        flow = analyze_pcap(str(handshake_pcap)).flows[0]

        assert flow.syn_time is not None
        assert flow.first_data_time == pytest.approx(1000.021, abs=1e-6)
        assert flow.fin_time == pytest.approx(1000.100, abs=1e-6)
        assert flow.rst_time is None


class TestDirection:
    def test_target_port_decides_direction(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap), target_ports=[443])

        # Client -> :443 is uplink, :443 -> client is downlink.
        assert m.ul_packets == 4
        assert m.dl_packets == 2
        assert m.ul_bytes_total > 0
        assert m.dl_bytes_total > m.ul_bytes_total  # server sent the bigger record

    def test_mean_packet_sizes(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap), target_ports=[443])

        assert m.ul_mean_pkt_size == pytest.approx(m.ul_bytes_total / m.ul_packets)
        assert m.dl_mean_pkt_size == pytest.approx(m.dl_bytes_total / m.dl_packets)


class TestPortFilter:
    def test_filter_excludes_other_ports(self, tmp_path):
        pkts = [
            (2000.0, _frame(CLIENT_IP, SERVER_IP, 51000, 443, dpkt.tcp.TH_SYN)),
            (2000.1, _frame(CLIENT_IP, SERVER_IP, 51001, 9999, dpkt.tcp.TH_SYN)),
        ]
        path = tmp_path / "capture_eth0.pcap"
        _write_pcap(path, pkts)

        assert analyze_pcap(str(path)).tcp_packets == 2
        assert analyze_pcap(str(path), target_ports=[443]).tcp_packets == 1

    def test_unfiltered_name_pattern_bypasses_filter(self, tmp_path):
        """Loopback captures keep every port even with a target-port filter."""
        pkts = [
            (2000.0, _frame("127.0.0.1", "127.0.0.1", 51000, 18789, dpkt.tcp.TH_SYN)),
        ]
        path = tmp_path / "capture_lo_run1.pcap"
        _write_pcap(path, pkts)

        assert analyze_pcap(str(path), target_ports=[443]).tcp_packets == 0
        assert analyze_pcap(
            str(path), target_ports=[443],
            unfiltered_name_patterns=("capture_lo_",),
        ).tcp_packets == 1


class TestRetransmissions:
    def test_repeated_payload_counts_once(self, tmp_path):
        payload = b"A" * 100
        pkts = [
            (3000.0, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                            dpkt.tcp.TH_ACK, seq=1000, payload=payload)),
            (3000.5, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                            dpkt.tcp.TH_ACK, seq=1000, payload=payload)),
        ]
        path = tmp_path / "capture_eth0.pcap"
        _write_pcap(path, pkts)

        m = analyze_pcap(str(path))
        assert m.total_retransmissions == 1
        assert m.retransmission_rate == pytest.approx(0.5)

    def test_pure_acks_are_not_retransmissions(self, tmp_path):
        pkts = [
            (3000.0, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                            dpkt.tcp.TH_ACK, seq=1000, ack=1)),
            (3000.1, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                            dpkt.tcp.TH_ACK, seq=1000, ack=1)),
        ]
        path = tmp_path / "capture_eth0.pcap"
        _write_pcap(path, pkts)

        assert analyze_pcap(str(path)).total_retransmissions == 0

    def test_directions_have_independent_sequence_spaces(self, tmp_path):
        """Equal seq numbers in opposite directions are not retransmissions."""
        payload = b"B" * 50
        pkts = [
            (3000.0, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                            dpkt.tcp.TH_ACK, seq=500, payload=payload)),
            (3000.1, _frame(SERVER_IP, CLIENT_IP, SERVER_PORT, CLIENT_PORT,
                            dpkt.tcp.TH_ACK, seq=500, payload=payload)),
        ]
        path = tmp_path / "capture_eth0.pcap"
        _write_pcap(path, pkts)

        assert analyze_pcap(str(path)).total_retransmissions == 0


class TestBurstSegmentation:
    def test_gap_threshold_splits_bursts(self, tmp_path):
        """Two packet groups 500 ms apart are two bursts at a 100 ms gap."""
        pkts = []
        for i in range(5):
            pkts.append((4000.0 + i * 0.001,
                         _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                                dpkt.tcp.TH_ACK, seq=1 + i * 100, payload=b"C" * 100)))
        for i in range(5):
            pkts.append((4000.5 + i * 0.001,
                         _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                                dpkt.tcp.TH_ACK, seq=2000 + i * 100, payload=b"C" * 100)))
        path = tmp_path / "capture_eth0.pcap"
        _write_pcap(path, pkts)

        m = analyze_pcap(str(path), target_ports=[443])
        ul_bursts = m.bursts_by_gap["100ms"]["ul"]

        assert len(ul_bursts) == 2
        assert all(b["packet_count"] == 5 for b in ul_bursts)

    def test_interburst_idle_gap_recorded(self, tmp_path):
        pkts = [
            (4000.0, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                            dpkt.tcp.TH_ACK, seq=1, payload=b"D" * 10)),
            (4000.5, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT, SERVER_PORT,
                            dpkt.tcp.TH_ACK, seq=100, payload=b"D" * 10)),
        ]
        path = tmp_path / "capture_eth0.pcap"
        _write_pcap(path, pkts)

        m = analyze_pcap(str(path), target_ports=[443])
        idle = m.interburst_idle_by_gap["100ms"]["ul"]

        assert idle == pytest.approx([0.5], abs=1e-6)


class TestWindowedThroughput:
    def test_all_windows_present(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap))

        for label in ("1ms", "10ms", "100ms", "1s", "10s"):
            assert label in m.throughput_by_window
            assert label in m.peak_mbps_by_window
            assert label in m.burstiness_by_window

    def test_burstiness_is_peak_over_mean(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap))
        series = m.throughput_by_window["10ms"]
        totals = [ul + dl for _, ul, dl in series]

        expected = max(totals) / (sum(totals) / len(totals))
        assert m.burstiness_by_window["10ms"] == pytest.approx(expected)

    def test_burstiness_at_least_one(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap))

        for label, value in m.burstiness_by_window.items():
            assert value >= 1.0, label


class TestUDPAndIPv6:
    def test_udp_packets_counted(self, tmp_path):
        pkts = [(5000.0 + i * 0.01, _udp_frame(CLIENT_IP, SERVER_IP, 40000, 3478))
                for i in range(4)]
        path = tmp_path / "capture_eth0.pcap"
        _write_pcap(path, pkts)

        m = analyze_pcap(str(path))
        assert m.udp_packets == 4
        assert m.tcp_packets == 0

    def test_ipv6_tcp_parsed(self, tmp_path):
        tcp = dpkt.tcp.TCP(sport=CLIENT_PORT, dport=SERVER_PORT,
                           flags=dpkt.tcp.TH_SYN, win=65535)
        tcp.off = 5
        ip6 = dpkt.ip6.IP6(src=b"\x20\x01" + b"\x00" * 14, dst=b"\x20\x01" + b"\x00" * 13 + b"\x01",
                           nxt=dpkt.ip.IP_PROTO_TCP, hlim=64, data=tcp)
        ip6.plen = len(bytes(tcp))
        eth = dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x00" * 6,
                                     type=dpkt.ethernet.ETH_TYPE_IP6, data=ip6)
        path = tmp_path / "capture_eth0.pcap"
        _write_pcap(path, [(6000.0, bytes(eth))])

        m = analyze_pcap(str(path))
        assert m.tcp_packets == 1
        assert len(m.packets) == 1


class TestBatchAndMerge:
    def test_analyze_multiple_pcaps(self, tmp_path, handshake_pcap):
        second = tmp_path / "capture_eth0_b.pcap"
        _write_pcap(second, [(7000.0, _frame(CLIENT_IP, SERVER_IP, CLIENT_PORT,
                                             SERVER_PORT, dpkt.tcp.TH_SYN))])

        results = analyze_multiple_pcaps(str(tmp_path), pattern="*.pcap")
        assert len(results) == 2

    def test_unreadable_file_is_skipped(self, tmp_path, handshake_pcap):
        (tmp_path / "broken.pcap").write_bytes(b"not a pcap at all")

        results = analyze_multiple_pcaps(str(tmp_path), pattern="*.pcap")
        assert len(results) == 1  # the good one; the broken one is logged and skipped

    def test_merge_totals(self, handshake_pcap):
        m = analyze_pcap(str(handshake_pcap))
        merged = merge_pcap_metrics([m, m])

        assert merged["total_captures"] == 2
        assert merged["total_packets"] == 2 * m.total_packets
        assert merged["total_bytes"] == 2 * m.total_bytes
        assert merged["rtt_mean_ms"] == pytest.approx(m.rtt_mean_ms)

    def test_merge_empty_list(self):
        assert merge_pcap_metrics([]) == {}


class TestFlowProperties:
    def test_derived_properties(self):
        flow = TCPFlow(src_ip="10.0.0.1", dst_ip="10.0.0.2",
                       src_port=1234, dst_port=443,
                       start_time=100.0, end_time=102.0,
                       bytes_sent=1000, bytes_recv=3000,
                       packets_sent=10, packets_recv=20,
                       retransmissions=3)

        assert flow.duration == pytest.approx(2.0)
        assert flow.throughput_bps == pytest.approx(4000 * 8 / 2.0)
        assert flow.ul_throughput_bps == pytest.approx(1000 * 8 / 2.0)
        assert flow.dl_throughput_bps == pytest.approx(3000 * 8 / 2.0)
        assert flow.retransmission_rate == pytest.approx(3 / 30)
        assert flow.flow_key == "10.0.0.1:1234-10.0.0.2:443"

    def test_zero_duration_is_safe(self):
        flow = TCPFlow(src_ip="a", dst_ip="b", src_port=1, dst_port=2)

        assert flow.duration == 0.0
        assert flow.throughput_bps == 0.0
        assert flow.handshake_rtt is None


class TestSerialization:
    def test_to_dict_is_json_safe(self, handshake_pcap):
        import json

        d = analyze_pcap(str(handshake_pcap)).to_dict()
        json.dumps(d)  # must not raise

        assert d["total_packets"] == 6
        assert d["tcp_flows"] == 1


class TestSequenceIntervalMerging:
    @pytest.mark.parametrize("intervals,expected_retrans", [
        ([(0, 100), (100, 200)], 0),        # adjacent, no overlap
        ([(0, 100), (50, 150)], 1),         # partial overlap
        ([(0, 100), (0, 100)], 1),          # exact repeat
        ([(0, 100), (200, 300), (50, 250)], 1),  # spans a hole
    ])
    def test_overlap_detection(self, intervals, expected_retrans):
        ranges = []
        hits = sum(PcapAnalyzer._record_sequence_interval(ranges, s, e)
                   for s, e in intervals)

        assert hits == expected_retrans

    def test_empty_interval_ignored(self):
        ranges = []
        assert PcapAnalyzer._record_sequence_interval(ranges, 100, 100) is False
        assert ranges == []
