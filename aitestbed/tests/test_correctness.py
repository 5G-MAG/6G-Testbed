import json
import os
import time
from pathlib import Path

import dpkt

from clients.realtime_client import RealtimeClient
from clients.realtime_webrtc_client import RealtimeWebRTCClient
from configs import DEFAULT_CAPTURE_FILTER, WEBRTC_UDP_PORTS
from netemu.pcap import PcapAnalyzer
from analysis.ran2_metrics import _is_primary_turn, _per_tool_bytes
from analysis.trace_logger import TraceLogger
from capture.l7_capture import _redact_headers, _redact_url
from orchestrator import TestbedOrchestrator as _TestbedOrchestrator


def test_default_capture_filter_includes_webrtc_media_ports():
    """The full runner must not reduce WebRTC sessions to TCP signaling."""
    for port in WEBRTC_UDP_PORTS:
        assert f"udp port {port}" in DEFAULT_CAPTURE_FILTER


def test_webrtc_sdp_exchange_uses_ga_calls_endpoint(monkeypatch):
    """The retired Beta application/sdp endpoint must not return."""
    captured = {}

    class _Response:
        status_code = 201
        text = "answer-sdp"

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr("clients.realtime_webrtc_client.requests.post", fake_post)
    client = object.__new__(RealtimeWebRTCClient)
    client.api_key = "test-key"
    session = {"type": "realtime", "model": "gpt-realtime-mini"}

    assert client._post_sdp_offer("offer-sdp", session) == "answer-sdp"
    assert captured["url"].endswith("/v1/realtime/calls")
    assert "OpenAI-Beta" not in captured["headers"]
    assert captured["files"]["sdp"][1] == "offer-sdp"
    assert json.loads(captured["files"]["session"][1]) == session


def test_webrtc_session_config_uses_ga_schema():
    client = object.__new__(RealtimeWebRTCClient)
    client.model = "gpt-realtime-mini"

    session = client._build_session_config(
        modalities=["text", "audio"],
        voice="sage",
        instructions="Be concise.",
        turn_detection={"type": "server_vad", "create_response": False},
        input_audio_transcription={"model": "whisper-1"},
        max_response_output_tokens=256,
    )

    assert session["type"] == "realtime"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["output"]["voice"] == "sage"
    assert session["audio"]["input"]["transcription"]["model"] == "whisper-1"
    assert session["max_output_tokens"] == 256
    assert "modalities" not in session
    assert "temperature" not in session


def test_ws_realtime_uses_ga_handshake_and_schema():
    """The retired Beta WS handshake and session schema must not return."""
    client = object.__new__(RealtimeClient)
    client.api_key = "test-key"
    client.model = "gpt-realtime-mini"

    headers = client._connection_headers()
    assert "OpenAI-Beta" not in headers
    assert headers["Authorization"] == "Bearer test-key"

    update = client._build_session_update(
        modalities=["text", "audio"],
        voice="sage",
        instructions="Be concise.",
        turn_detection={"type": "server_vad", "create_response": False},
        input_audio_transcription={"model": "whisper-1"},
        max_response_output_tokens=256,
    )
    assert update["type"] == "session.update"
    session = update["session"]
    assert session["type"] == "realtime"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["output"]["voice"] == "sage"
    assert session["audio"]["input"]["transcription"]["model"] == "whisper-1"
    assert session["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert session["max_output_tokens"] == 256
    assert "modalities" not in session
    assert "temperature" not in session
    assert "input_audio_format" not in session

    text_only = client._build_session_update(
        modalities=["text"], voice="sage", instructions=None,
        turn_detection=None, input_audio_transcription=None,
        max_response_output_tokens=None,
    )
    assert text_only["session"]["output_modalities"] == ["text"]
    assert "audio" not in text_only["session"]


def test_retransmissions_use_payload_intervals_per_direction():
    analyzer = PcapAnalyzer()
    flows = {}
    ack = dpkt.tcp.TCP(sport=50000, dport=443, seq=100, flags=dpkt.tcp.TH_ACK)
    analyzer._process_tcp_packet(flows, 1.0, "10.0.0.1", "10.0.0.2", ack, len(ack))
    analyzer._process_tcp_packet(flows, 1.1, "10.0.0.1", "10.0.0.2", ack, len(ack))
    flow = next(iter(flows.values()))
    assert flow.retransmissions == 0

    data = dpkt.tcp.TCP(sport=50000, dport=443, seq=200, data=b"payload")
    analyzer._process_tcp_packet(flows, 2.0, "10.0.0.1", "10.0.0.2", data, len(data))
    analyzer._process_tcp_packet(flows, 2.1, "10.0.0.1", "10.0.0.2", data, len(data))
    assert flow.retransmissions == 1

    reverse = dpkt.tcp.TCP(sport=443, dport=50000, seq=200, data=b"payload")
    analyzer._process_tcp_packet(flows, 2.2, "10.0.0.2", "10.0.0.1", reverse, len(reverse))
    assert flow.retransmissions == 1


def test_ran2_understands_emitted_metadata_schema():
    tool = {
        "session_id": "s",
        "turn_index": 0,
        "request_bytes": 10,
        "response_bytes": 20,
        "latency_sec": 0.5,
        "metadata": json.dumps({"type": "mcp_tool_call", "tool_name": "fetch"}),
    }
    action = {**tool, "metadata": json.dumps({"type": "computer_use_action"})}
    assert not _is_primary_turn(tool)
    assert not _is_primary_turn(action)
    assert _per_tool_bytes([tool])["fetch"]["response_bytes"] == 20


def test_trace_redacts_credentials_and_uses_private_permissions(tmp_path):
    logger = TraceLogger(tmp_path / "traces", enabled=True)
    path = Path(logger.write_trace(
        scenario_id="s", session_id="abc", turn_index=0, run_index=0,
        network_profile="none", provider="p", model="m",
        request_payload={"Authorization": "Bearer secret", "api_key": "private-key-value"},
    ))
    text = path.read_text()
    assert "secret" not in text and "private-key-value" not in text
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_l7_redacts_headers_and_query_credentials():
    assert _redact_headers({"Authorization": "Bearer x"})["Authorization"] == "[REDACTED]"
    url = _redact_url("https://example.test/v1?api_key=secret&q=weather#fragment")
    assert "secret" not in url and "weather" in url and "fragment" not in url


class _SlowScenario:
    def __init__(self, marker):
        self.marker = marker

    def run(self, network_profile, run_index):
        time.sleep(0.4)
        Path(self.marker).write_text("late write")


def test_timeout_kills_worker_before_late_side_effect(tmp_path):
    marker = tmp_path / "late.txt"
    result = _TestbedOrchestrator._run_with_timeout(
        None, _SlowScenario(marker), "no_emulation", 0, 0.05, "slow"
    )
    assert not result.success
    time.sleep(0.5)
    assert not marker.exists()
