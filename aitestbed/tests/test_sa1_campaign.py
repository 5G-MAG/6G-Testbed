"""SA1 failure accounting, stream completion and resumable run identity."""

import base64
import json
from dataclasses import asdict

import httpx
import pytest
from analysis.logger import TrafficLogger, LogRecord
from clients.base import NullLLMClient
from scenarios.base import ScenarioResult
from scenarios.media import MeasuredChatScenario, VisualAssistantScenario
from sa1_campaign import report, proportion_interval
from sa1_controlled import matched_metrics


def test_incomplete_stream_is_failure_with_body_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    original = httpx.Client
    body = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, stream=httpx.ByteStream(body))
            ),
            **kw,
        ),
    )
    logger = TrafficLogger(tmp_path / "traffic.db")
    result = MeasuredChatScenario(
        NullLLMClient("test"),
        logger,
        {"type": "sa1_chatbot", "model": "test", "prompt": "hello", "stream": True},
    ).run("no_emulation")
    assert not result.success
    assert result.total_response_bytes == len(body)
    assert result.metadata["outputs"] == []


def test_visual_assistant_uploads_bytes_not_a_local_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video payload")

    def handle(request):
        payload = json.loads(request.content)
        assert (
            base64.b64decode(payload["contents"][0]["parts"][1]["inline_data"]["data"])
            == video.read_bytes()
        )
        return httpx.Response(
            200,
            stream=httpx.ByteStream(
                json.dumps(
                    {"candidates": [{"content": {"parts": [{"text": "A scene."}]}}]}
                ).encode()
            ),
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    result = VisualAssistantScenario(
        NullLLMClient("test"),
        TrafficLogger(tmp_path / "traffic.db"),
        {
            "type": "sa1_visual_assistant",
            "model": "test",
            "prompt": "describe",
            "video_path": str(video),
        },
    ).run("no_emulation")
    assert result.success
    assert result.metadata["input_media_bytes"] == len(video.read_bytes())
    assert result.total_request_bytes > result.metadata["input_media_bytes"]
    assert result.metadata["provider_processing_sec"] is None


def test_terminal_ledger_preserves_failed_indices_and_results(tmp_path):
    from orchestrator import TestbedOrchestrator

    runner = object.__new__(TestbedOrchestrator)
    runner.logger = TrafficLogger(tmp_path / "traffic.db")
    for index, success in [(0, False), (2, True)]:
        result = ScenarioResult(
            "photo", f"session_{index}", "no_emulation", index, success=success
        )
        result.log_records = [
            LogRecord(
                timestamp=1,
                scenario_id="photo",
                session_id=result.session_id,
                turn_index=0,
                run_index=index,
                success=success,
            )
        ]
        runner._save_run_result(result)
    restored = runner._completed_results("photo", "no_emulation")
    assert [r.run_index for r in restored] == [0, 2]
    assert [r.success for r in restored] == [False, True]
    assert sum(len(r.log_records) for r in restored) == 2


def test_report_counts_failure_and_unresolved_attempt(tmp_path):
    jobs = [{"id": str(i)} for i in range(3)]
    for i in range(3):
        (tmp_path / str(i)).mkdir()
    for i, success in [(0, True), (1, False)]:
        result = ScenarioResult(
            "photo", str(i), "no_emulation", i, success=success, total_latency_sec=5
        )
        (tmp_path / str(i) / "result.json").write_text(json.dumps(asdict(result)))
    (tmp_path / "2/started.json").write_text("{}")
    summary = report(
        tmp_path, {"jobs": jobs, "scope": "test", "scenarios": {"photo": {}}}
    )
    assert summary["scenarios"]["photo"]["attempts"] == 2
    assert summary["scenarios"]["photo"]["failed"] == 1
    assert summary["unresolved_jobs"] == ["2"]
    assert proportion_interval(0, 3)[1] > 0.5


def test_duplicate_arrival_does_not_extend_unit_latency(tmp_path):
    import socket, struct
    import dpkt

    def frame(seq, flags, data=b""):
        tcp = dpkt.tcp.TCP(sport=42000, dport=8098, seq=seq, flags=flags, data=data)
        tcp.off = 5
        ip = dpkt.ip.IP(
            src=socket.inet_aton("10.231.1.2"),
            dst=socket.inet_aton("10.231.2.2"),
            p=6,
            id=seq,
            data=tcp,
        )
        ip.len = len(ip)
        return bytes(
            dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x00" * 6, type=2048, data=ip)
        )

    syn = frame(100, dpkt.tcp.TH_SYN)
    body = frame(101, dpkt.tcp.TH_ACK, struct.pack("!Q", 8) + b"abcdefgh")
    for name, packets in [
        ("sender", [(1, syn), (1.01, body)]),
        ("receiver", [(1.005, syn), (1.02, body), (1.03, body)]),
    ]:
        with (tmp_path / f"{name}.pcap").open("wb") as f:
            w = dpkt.pcap.Writer(f)
            for t, p in packets:
                w.writepkt(p, t)
    result = matched_metrics(
        tmp_path / "sender.pcap", tmp_path / "receiver.pcap", "ul", 8
    )
    assert result["complete"]
    assert result["burst_transfer_ms"] == pytest.approx(10)
    assert result["duplicate_received_data_packets"] == 1
    assert result["unmatched_received_data_packets"] == 0
    assert result["matched_data_packet_delays_ms"] == pytest.approx([10])


def test_resume_runs_only_holes_and_keeps_failure_in_aggregate(tmp_path):
    from orchestrator import TestbedOrchestrator
    from analysis.metrics import MetricsCalculator

    runner = object.__new__(TestbedOrchestrator)
    runner.logger = TrafficLogger(tmp_path / "traffic.db")
    runner.scenarios_config = {
        "scenarios": {"test": {"type": "test", "provider": "null"}},
        "defaults": {},
    }
    runner.mcp_transport = "stdio"
    runner.get_client = lambda provider: NullLLMClient(provider)
    runner._apply_emulator_profile = lambda *args: True
    runner.emulator = type("Emulator", (), {"clear": lambda self: True})()
    called = []

    class Scenario:
        def __init__(self, client, logger, config):
            pass

        def run(self, network_profile, run_index):
            called.append(run_index)
            return ScenarioResult(
                "test",
                f"run-{run_index}",
                network_profile,
                run_index,
                success=run_index != 0,
                total_latency_sec=2,
            )

    runner.scenario_classes = {"test": Scenario}
    runner.run_experiment("test", "no_emulation", runs=3, inter_run_delay=0)
    assert called == [0, 1, 2]
    with runner._run_ledger() as connection:
        connection.execute("DELETE FROM experiment_runs WHERE run_index=1")
    called.clear()
    results = runner.run_experiment(
        "test", "no_emulation", runs=3, inter_run_delay=0, resume=True
    )
    assert called == [1]
    assert sorted(r.run_index for r in results) == [0, 1, 2]
    metrics = MetricsCalculator.calculate(
        [asdict(x) for r in results for x in r.log_records], "test", "no_emulation"
    )
    assert metrics.success_rate == pytest.approx(200 / 3)
    called.clear()
    assert len(runner.run_experiment("test", "no_emulation", runs=3, resume=True)) == 3
    assert called == []


def test_unresolved_attempt_is_not_resubmitted(tmp_path):
    from orchestrator import TestbedOrchestrator

    runner = object.__new__(TestbedOrchestrator)
    runner.logger = TrafficLogger(tmp_path / "traffic.db")
    runner._mark_run_started("video", "no_emulation", 0)
    with pytest.raises(ValueError, match="unresolved"):
        runner._completed_results("video", "no_emulation")


def test_incomplete_unit_remains_in_deadline_denominator():
    from sa1_controlled import summarize

    unit = {
        "complete": False,
        "source_unit_started": True,
        "source_data_packets": 1,
        "burst_transfer_ms": None,
        "observation_elapsed_ms": 20000,
        "matched_data_packet_delays_ms": [20],
        "source_all_packet_sizes_ip_payload": [1400],
        "source_data_packet_sizes_ip_payload": [1400],
        "body_bytes": 20000000,
    }
    response = {**unit, "source_unit_started": False, "source_data_packets": 0}
    rows = [
        {
            "workload": "photo_spec_payload",
            "profile": "satellite_leo",
            "success": False,
            "measurement_valid": True,
            "thresholds": {"ul": 1500, "dl": 1500},
            "ul": unit,
            "dl": response,
        }
    ]
    result = summarize(rows)[0]
    assert result["successes"] == 0
    assert result["directions"]["ul"]["deadline_misses"] == 1
    assert result["directions"]["ul"]["deadline_assessable_units"] == 1
    assert result["directions"]["ul"]["burst_transfer_ms"]["n"] == 0
    assert result["directions"]["dl"]["units"] == 0
    rows[0]["dl"] = unit
    downlink = summarize(rows)[0]["directions"]["dl"]
    assert downlink["photo_dl_1000ms_misses"] == 1
    assert downlink["photo_dl_1000ms_assessable_units"] == 1


def test_invalid_capture_does_not_establish_deadline_compliance():
    from sa1_controlled import summarize

    row = {
        "workload": "visual_assistant",
        "profile": "no_emulation",
        "success": True,
        "measurement_valid": False,
        "thresholds": {"ul": 400},
        "ul": {"complete": True},
    }
    result = summarize([row])[0]
    assert result["measurement_invalid_runs"] == 1
    assert result["directions"]["ul"]["deadline_assessable_units"] == 0
    assert result["directions"]["ul"]["deadline_miss_proportion_ci95"] is None


def test_permanent_provider_error_blocks_only_same_scenario(monkeypatch, tmp_path):
    import sa1_campaign as campaign
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 5,
                "scenarios": {
                    "blocked": {"type": "sa1_chatbot", "model": "test"},
                    "available": {"type": "sa1_chatbot", "model": "test"},
                },
            }
        )
    )
    assets = tmp_path / "assets.json"
    assets.write_text("{}")
    calls = []

    class Capture:
        def __init__(self, **kwargs):
            self.path = tmp_path / "dummy.pcap"

        def start(self, **kwargs):
            return self.path

        def stop(self):
            pass

    class Scenario:
        def __init__(self, client, logger, config):
            self.config = config

        def run(self, profile, index):
            name = self.config["scenario_id"]
            calls.append((name, index))
            return ScenarioResult(
                name,
                f"{name}_{index}",
                profile,
                index,
                success=name == "available",
                metadata={
                    "phases": [{"http_status": 402 if name == "blocked" else 200}]
                },
            )

    monkeypatch.setattr(campaign, "validate_media_config", lambda *a, **kw: None)
    monkeypatch.setattr(campaign, "CaptureController", Capture)
    monkeypatch.setattr(campaign, "analyze_capture", lambda *a, **kw: None)
    monkeypatch.setitem(campaign.CLASSES, "sa1_chatbot", Scenario)
    output = tmp_path / "campaign"
    summary = campaign.run(
        output,
        config_path,
        assets,
        3,
        True,
        None,
        "test",
        stop_on_permanent_http_error=True,
    )
    assert summary["scenarios"]["blocked"]["attempts"] == 1
    assert summary["scenarios"]["blocked"]["failed"] == 1
    assert summary["scenarios"]["available"]["successes"] == 3
    assert len(summary["blocked_jobs"]) == 2
    assert not summary["unresolved_jobs"]
    assert len(calls) == 4
    campaign.run(
        output,
        config_path,
        assets,
        3,
        True,
        None,
        "test",
        stop_on_permanent_http_error=True,
    )
    assert len(calls) == 4  # Resume does not submit blocked or completed jobs.
