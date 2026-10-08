"""The pcap analysis port set must keep what the capture filter records.

``DEFAULT_CAPTURE_FILTER`` records the WebRTC media five-tuple on the STUN/TURN
service ports (OpenAI's Realtime relays answer on UDP 3478). The campaign
analysis in ``generate_charts`` re-filters every capture through
``_collect_capture_target_ports()``; if that set lacks the same UDP ports, the
analyzer discards every RTP packet and a WebRTC run is reduced to its TCP
signaling in every direction, burst and throughput metric.
"""

from pathlib import Path

import dpkt
import pytest

from configs import DEFAULT_CAPTURE_FILTER, WEBRTC_UDP_PORTS
from generate_charts import _collect_capture_target_ports

pcap_mod = pytest.importorskip("netemu.pcap")

CLIENT_IP = "192.168.99.58"
RELAY_IP = "20.0.0.1"
CLIENT_PORT = 45076
RELAY_PORT = 3478  # UDP port advertised in OpenAI's SDP answer


def _udp_frame(src: str, dst: str, sport: int, dport: int) -> bytes:
    payload = b"\x80\x60" + b"\x00" * 158  # RTP v2, PT 96, 160 B total
    udp = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
    udp.ulen = len(bytes(udp))
    ip = dpkt.ip.IP(
        src=bytes(int(o) for o in src.split(".")),
        dst=bytes(int(o) for o in dst.split(".")),
        p=dpkt.ip.IP_PROTO_UDP,
        data=udp,
    )
    ip.len = len(bytes(ip))
    return bytes(dpkt.ethernet.Ethernet(type=dpkt.ethernet.ETH_TYPE_IP, data=ip))


def _write_rtp_pcap(path: Path, packets_per_direction: int = 50) -> None:
    """A 20 ms cadence bidirectional audio stream between an ephemeral client
    port and the relay's UDP 3478, as the primary capture records it."""
    with open(path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        for i in range(packets_per_direction):
            t = 1000.0 + i * 0.020
            writer.writepkt(_udp_frame(CLIENT_IP, RELAY_IP, CLIENT_PORT, RELAY_PORT), t)
            writer.writepkt(_udp_frame(RELAY_IP, CLIENT_IP, RELAY_PORT, CLIENT_PORT), t + 0.005)


def test_analysis_port_set_covers_capture_filter_udp_ports():
    ports = set(_collect_capture_target_ports())
    for port in WEBRTC_UDP_PORTS:
        assert f"udp port {port}" in DEFAULT_CAPTURE_FILTER
        assert port in ports, f"captured UDP port {port} is dropped at analysis"
    assert {443, 80} <= ports


def test_campaign_analysis_keeps_webrtc_rtp(tmp_path):
    pcap = tmp_path / "capture_20260921_120000.pcap"
    _write_rtp_pcap(pcap)

    results = pcap_mod.analyze_multiple_pcaps(
        str(tmp_path),
        pattern="*.pcap",
        target_ports=_collect_capture_target_ports(),
        unfiltered_name_patterns=("capture_lo_",),
    )
    assert len(results) == 1
    m = results[0]

    assert m.total_packets == 100
    assert m.udp_packets == 100, "RTP packets were filtered out of the analysis"
    assert len(m.packets) == 100
    # 3478 is in the target set, so direction is attributed by the relay port,
    # not by the lower-port heuristic.
    assert m.ul_packets == 50
    assert m.dl_packets == 50
    assert m.ul_bytes_total > 0 and m.dl_bytes_total > 0
    assert m.peak_throughput_mbps > 0


def test_analysis_without_webrtc_ports_would_drop_rtp(tmp_path):
    """Documents the failure mode the port set now prevents."""
    pcap = tmp_path / "capture_20260921_120000.pcap"
    _write_rtp_pcap(pcap)

    m = pcap_mod.analyze_pcap(str(pcap), target_ports=[443, 80])
    assert m.total_packets == 100
    assert m.udp_packets == 0
    assert m.packets == []


# ---------------------------------------------------------------------------
# Loopback: a local WebRTC peer. ICE never offers 127.0.0.1, so both ends use
# the primary interface address and both ports are ephemeral; the kernel
# carries the media over lo.
# ---------------------------------------------------------------------------
HOST_IP = "192.168.10.65"
LO_CLIENT_PORT = 45076
LO_SERVER_PORT = 51000  # higher than the client's: lower-port rule would invert


def _write_lo_rtp_pcap(path: Path, packets_per_direction: int = 50) -> None:
    with open(path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        for i in range(packets_per_direction):
            t = 2000.0 + i * 0.020
            writer.writepkt(_udp_frame(HOST_IP, HOST_IP, LO_CLIENT_PORT, LO_SERVER_PORT), t)
            writer.writepkt(_udp_frame(HOST_IP, HOST_IP, LO_SERVER_PORT, LO_CLIENT_PORT), t + 0.005)


def test_video_scenario_declares_loopback():
    """Starts the lo capture (orchestrator.scenarios_use_loopback) and lets
    the scenario shape lo; without it the RTP stream is invisible."""
    import yaml

    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "scenarios.yaml"))
    sc = cfg["scenarios"]["realtime_video_understanding"]
    assert sc.get("uses_loopback") is True


def test_campaign_analysis_keeps_loopback_webrtc_rtp(tmp_path):
    pcap = tmp_path / "capture_lo_20260921_120000.pcap"
    _write_lo_rtp_pcap(pcap)

    results = pcap_mod.analyze_multiple_pcaps(
        str(tmp_path),
        pattern="*.pcap",
        target_ports=_collect_capture_target_ports(),
        unfiltered_name_patterns=("capture_lo_",),
    )
    assert len(results) == 1
    m = results[0]

    assert m.udp_packets == 100, "loopback RTP dropped despite the lo exemption"
    assert m.udp_flows == 1
    # First datagram is the client's: direction follows it, not the port order.
    assert m.ul_packets == 50
    assert m.dl_packets == 50
    assert m.packets[0].direction == "ul"


def test_video_scenario_shapes_all_udp_and_signaling_tcp():
    """The scenario asks netemu for all loopback UDP plus the signaling port,
    and clears lo afterwards, without needing the VLM client to import."""
    import sys
    import types
    from unittest.mock import MagicMock

    # realtime_video imports the VLM client, which needs the aiortc tokenId
    # fork; stub it so the scenario module itself can be imported.
    stub = types.ModuleType("clients.realtime_webrtc_vlm_client")
    stub.RealtimeWebRTCVLMClient = object
    stub.VLMSessionMetrics = object
    stub.VLMTurnMetrics = object
    saved = sys.modules.get("clients.realtime_webrtc_vlm_client")
    sys.modules["clients.realtime_webrtc_vlm_client"] = stub
    try:
        from scenarios.realtime_video import RealtimeVideoUnderstandingScenario
    finally:
        if saved is None:
            sys.modules.pop("clients.realtime_webrtc_vlm_client", None)
        else:
            sys.modules["clients.realtime_webrtc_vlm_client"] = saved

    scenario = object.__new__(RealtimeVideoUnderstandingScenario)
    scenario.emulator = MagicMock()
    scenario.emulator.apply_profile_to_loopback.return_value = True
    scenario._current_network_profile = "poor_cellular"
    scenario._netem_on_lo = False

    scenario._apply_loopback_shaping("ignored", 1234)
    scenario.emulator.apply_profile_to_loopback.assert_called_once_with(
        "poor_cellular", selectors=[("udp", None), ("tcp", 1234)]
    )
    assert scenario._netem_on_lo is True

    scenario._clear_loopback_shaping()
    scenario.emulator.clear_loopback.assert_called_once()
    assert scenario._netem_on_lo is False

    # The reference profile leaves lo bare.
    scenario.emulator.reset_mock()
    scenario._current_network_profile = "no_emulation"
    scenario._apply_loopback_shaping("no_emulation", 1234)
    scenario.emulator.apply_profile_to_loopback.assert_not_called()
