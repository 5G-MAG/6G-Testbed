import json
from pathlib import Path
import socket

import dpkt
import pytest

from sa1_capture_filter import filter_capture


def frame(src, dst, sport, dport, payload=b"test"):
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, seq=1, flags=16, data=payload)
    tcp.off = 5
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=6, data=tcp)
    ip.len = len(ip)
    return bytes(dpkt.ethernet.Ethernet(type=2048, data=ip))


def test_only_measured_connection_survives_even_with_same_remote_ip(tmp_path):
    metadata = {
        "phases": [
            {
                "local_endpoint": ["10.0.0.1", 40000],
                "remote_endpoint": ["10.0.0.2", 443],
            }
        ]
    }
    (tmp_path / "result.json").write_text(json.dumps({"metadata": metadata}))
    (tmp_path / "capture.pcap.metadata.json").write_text(
        json.dumps({"capture_stats": {"packets_dropped_by_kernel": 2}})
    )
    with (tmp_path / "capture.pcap").open("wb") as stream:
        writer = dpkt.pcap.Writer(stream)
        writer.writepkt(frame("10.0.0.1", "10.0.0.2", 40000, 443), ts=1)
        writer.writepkt(frame("10.0.0.1", "10.0.0.2", 40001, 443), ts=1.5)
        writer.writepkt(frame("10.0.0.2", "10.0.0.1", 443, 40000), ts=2)
        writer.writepkt(frame("10.0.0.3", "10.0.0.1", 443, 40000), ts=2.5)
    report = filter_capture(tmp_path)
    assert report["retained_packet_records"] == 2
    assert report["excluded_packet_records"] == 2
    assert report["unrelated_records_after_verification"] == 0
    assert report["capture_stats"]["packets_dropped_by_kernel"] == 2
    with (tmp_path / "capture.scenario.pcap").open("rb") as stream:
        assert [ts for ts, buf in dpkt.pcap.Reader(stream)] == [1, 2]
    assert filter_capture(tmp_path) == report
    # The campaign's default report must reference the restricted artifact too.
    from sa1_campaign import analyze_capture

    analyze_capture(tmp_path / "capture.pcap", metadata)
    analysis = json.loads((tmp_path / "capture-analysis.json").read_text())
    assert Path(analysis["capture_file"]).name == "capture.scenario.pcap"
    assert analysis["unrelated_records_in_export"] == 0
    assert analysis["excluded_unrelated_records"] == 2
    assert not analysis["capture_complete"]  # Original capture drops stay visible.


def test_missing_attribution_does_not_export_host_traffic(tmp_path):
    (tmp_path / "result.json").write_text('{"metadata": {}}')
    with pytest.raises(ValueError, match="refusing broad attribution"):
        filter_capture(tmp_path)
    assert not (tmp_path / "capture.scenario.pcap").exists()
