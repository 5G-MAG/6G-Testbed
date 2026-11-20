#!/usr/bin/env python3
"""Run fresh, resumable SA1 provider campaigns; all outcomes remain visible.

Provider runs use the existing network unchanged. Network impairment and
one-way latency are measured separately by sa1_controlled.py.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent / "netemu/src"))
sys.path.insert(0, str(APP.parent / "aiortc-main-clean/src"))
from dotenv import load_dotenv

load_dotenv(APP / ".env")
load_dotenv(APP.parent / ".env")
import yaml
from analysis.logger import TrafficLogger
from clients.base import NullLLMClient
from netemu.capture import CaptureController
from netemu.pcap import PcapAnalyzer
from scenarios.media import (
    VideoGenerationScenario,
    ImageEnhancementScenario,
    MeasuredImageGenerationScenario,
    MeasuredChatScenario,
    VisualAssistantScenario,
    validate_media_config,
)
from sa1_controlled import stats, write_json
from sa1_capture_filter import filter_capture

CLASSES = {
    "video_generation": VideoGenerationScenario,
    "image_enhancement": ImageEnhancementScenario,
    "sa1_image_generation": MeasuredImageGenerationScenario,
    "sa1_chatbot": MeasuredChatScenario,
    "sa1_visual_assistant": VisualAssistantScenario,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proportion_interval(misses, n):
    if not n:
        return None
    z = 1.96
    p = misses / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, center - half), min(1, center + half)]


def capture_metrics(path, meta):
    parsed = PcapAnalyzer(target_ports=None).analyze(str(path))
    endpoints = {
        (tuple(p["local_endpoint"]), tuple(p["remote_endpoint"]))
        for p in meta.get("phases", [])
        if p.get("local_endpoint") and p.get("remote_endpoint")
    }
    flows = {
        f.flow_key
        for f in parsed.flows
        if ((f.src_ip, f.src_port), (f.dst_ip, f.dst_port)) in endpoints
    }
    selected = [p for p in parsed.packets if p.flow_key in flows]
    result = {
        "flow_attribution": "HTTP transport socket endpoints; dedicated client per run",
        "matched_flows": len(flows),
        "unattributed_flows": len(parsed.flows) - len(flows),
        "one_way_transfer_latency_available": False,
        "directions": {},
    }
    for direction in ["ul", "dl"]:
        packets = sorted(
            (p for p in selected if p.direction == direction), key=lambda p: p.timestamp
        )
        bursts = {}
        for gap in [10, 100]:
            groups = []
            for p in packets:
                if not groups or p.timestamp - groups[-1]["end"] > gap / 1000:
                    groups.append(
                        {
                            "start": p.timestamp,
                            "end": p.timestamp,
                            "bytes": p.size,
                            "packets": 1,
                        }
                    )
                else:
                    g = groups[-1]
                    g["end"] = p.timestamp
                    g["bytes"] += p.size
                    g["packets"] += 1
            for g in groups:
                g["duration_sec"] = g["end"] - g["start"]
                g["mean_rate_bps"] = (
                    g["bytes"] * 8 / g["duration_sec"]
                    if g["duration_sec"] > 0
                    else None
                )
            bursts[str(gap)] = groups
        result["directions"][direction] = {
            "captured_ip_payload_bytes": sum(p.size for p in packets),
            "host_packet_record_size_bytes": stats([p.size for p in packets]),
            "oversized_records": sum(p.size > 1480 for p in packets),
            "wire_packet_size_valid": bool(packets)
            and not any(p.size > 1480 for p in packets),
            "ack_only_packets": sum(
                p.payload_len == 0 and p.tcp_flags == 16 for p in packets
            ),
            "bursts_by_idle_gap_ms": bursts,
        }
    return result


def analyze_capture(pcap, metadata):
    """Analysis failure must not discard an outcome or resubmit a provider job."""
    pcap = Path(pcap)
    try:
        acquisition_hash = digest(pcap)
        filtered = filter_capture(pcap.parent)
        pcap = pcap.parent / filtered["pcap_file"]
        analysis = capture_metrics(pcap, metadata)
        analysis["capture_file"] = str(pcap)
        analysis["source_acquisition_sha256"] = acquisition_hash
        analysis["excluded_unrelated_records"] = filtered["excluded_packet_records"]
        analysis["unrelated_records_in_export"] = filtered[
            "unrelated_records_after_verification"
        ]
        analysis["capture_sha256"] = digest(pcap)
        analysis["capture_stats"] = json.loads(
            pcap.with_suffix(pcap.suffix + ".metadata.json").read_text()
        ).get("capture_stats")
        analysis["capture_complete"] = (analysis["capture_stats"] or {}).get(
            "packets_dropped_by_kernel"
        ) == 0
        analysis["wire_packet_metrics_valid"] = (
            analysis["capture_complete"]
            and analysis["matched_flows"] > 0
            and all(
                d["wire_packet_size_valid"] for d in analysis["directions"].values()
            )
        )
        write_json(pcap.parent / "capture-analysis.json", analysis)
    except Exception as exc:
        write_json(
            pcap.parent / "capture-analysis-error.json",
            {"error_type": type(exc).__name__},
        )


def report(output, plan):
    results = []
    for job in plan["jobs"]:
        path = output / job["id"] / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    groups = {}
    for name in plan["scenarios"]:
        selected = [r for r in results if r["scenario_id"] == name]
        success = sum(r["success"] for r in selected)
        groups[name] = {
            "attempts": len(selected),
            "successes": success,
            "failed": len(selected) - success,
            "failure_proportion_ci95": proportion_interval(
                len(selected) - success, len(selected)
            ),
            "client_completion_sec": stats(
                [r["total_latency_sec"] for r in selected if r["success"]]
            ),
            "request_body_bytes": stats([r["total_request_bytes"] for r in selected]),
            "response_body_bytes": stats([r["total_response_bytes"] for r in selected]),
            "output_artifact_bytes": stats(
                [r.get("metadata", {}).get("output_media_bytes", 0) for r in selected]
            ),
            "one_way_latency": "not measured for public providers",
        }
    summary = {
        "scope": plan["scope"],
        "excluded": ["gaming_enhancement"],
        "scenarios": groups,
        "complete_jobs": len(results),
        "planned_jobs": len(plan["jobs"]),
        "blocked_jobs": [
            {
                "job": j["id"],
                **json.loads((output / j["id"] / "blocked.json").read_text()),
            }
            for j in plan["jobs"]
            if (output / j["id"] / "blocked.json").exists()
        ],
        "unresolved_jobs": [
            j["id"]
            for j in plan["jobs"]
            if (output / j["id"] / "started.json").exists()
            and not (output / j["id"] / "result.json").exists()
        ],
    }
    write_json(output / "summary.json", summary)
    return summary


def run(
    output,
    config_path,
    assets_path,
    repetitions,
    execute,
    selected,
    interface,
    video_path=None,
    stop_on_permanent_http_error=False,
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = Path(config_path).read_bytes()
    config = yaml.safe_load(raw)
    assets = json.loads(Path(assets_path).read_text())
    scenarios = {
        name: dict(value)
        for name, value in config["scenarios"].items()
        if not selected or name in selected
    }
    if selected and set(selected) - scenarios.keys():
        raise ValueError("Unknown scenario selection")
    for name, c in scenarios.items():
        c.update(
            scenario_id=name,
            provider="sa1_http",
            retry_count=0,
            max_duration_sec=540,
            output_dir=str(output / "media"),
        )
        if c.get("image_asset"):
            value = assets[c.pop("image_asset")]
            c["image_paths"] = value if isinstance(value, list) else [value]
        if c.get("prompt_bytes"):
            n = c.pop("prompt_bytes")
            text = c["prompt"]
            c["prompt"] = text + " " * (n - len(text.encode()))
        if video_path and c["type"] == "sa1_visual_assistant":
            c["video_path"] = str(Path(video_path).resolve())
            c["workload_kind"] = "recorded_generated_video_input"
        if c.get("video_path") and not Path(c["video_path"]).is_absolute():
            c["video_path"] = str(APP / c["video_path"])
        validate_media_config(c, credentials=execute)
    hashes = {}
    for c in scenarios.values():
        for path in c.get("image_paths", []) + (
            [c["video_path"]] if c.get("video_path") else []
        ):
            hashes[path] = digest(path)
    sources = {
        str(p.relative_to(APP.parent)): digest(p)
        for p in [
            Path(__file__),
            APP / "scenarios/media.py",
            APP / "sa1_controlled.py",
            APP / "sa1_capture_filter.py",
            APP.parent / "netemu/src/netemu/capture.py",
            APP.parent / "netemu/src/netemu/pcap.py",
        ]
    }
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if (
            plan["config_sha256"] != hashlib.sha256(raw).hexdigest()
            or plan["input_hashes"] != hashes
            or plan["repetitions"] != repetitions
            or plan["scenarios"] != scenarios
            or plan.get("source_files_sha256") != sources
            or plan.get("stop_on_permanent_http_error", False)
            != stop_on_permanent_http_error
        ):
            raise ValueError(
                "Plan/config/inputs changed; choose a fresh output directory"
            )
    else:
        jobs = [
            {"id": f"{name}_{i:03d}", "scenario": name, "run_index": i}
            for name in scenarios
            for i in range(repetitions)
        ]
        random.Random(config["seed"]).shuffle(jobs)
        plan = {
            "schema_version": 1,
            "stop_on_permanent_http_error": stop_on_permanent_http_error,
            "scope": "real-provider baseline; network unchanged; no one-way latency claims",
            "config_sha256": hashlib.sha256(raw).hexdigest(),
            "input_hashes": hashes,
            "scenarios": scenarios,
            "repetitions": repetitions,
            "jobs": jobs,
            "seed": config["seed"],
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=APP, text=True
            ).strip(),
            "source_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff"], cwd=APP)
            ).hexdigest(),
            "source_files_sha256": sources,
        }
        for relative in sources:
            snapshot = output / "source" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((APP.parent / relative).read_bytes())
        write_json(plan_path, plan)
    if not execute:
        return {"prepared_jobs": len(plan["jobs"]), "output": str(output)}
    logger = TrafficLogger(str(output / "traffic.db"))
    for job in plan["jobs"]:
        directory = output / job["id"]
        directory.mkdir(exist_ok=True)
        if (directory / "result.json").exists():
            if not (directory / "capture-analysis.json").exists():
                saved = json.loads((directory / "result.json").read_text())
                analyze_capture(directory / "capture.pcap", saved.get("metadata", {}))
            continue
        if (directory / "blocked.json").exists():
            continue
        if (directory / "started.json").exists():
            print(
                f"Unresolved attempt {job['id']}; not resubmitting a potentially accepted provider job",
                flush=True,
            )
            continue
        c = scenarios[job["scenario"]]
        for path in c.get("image_paths", []) + (
            [c["video_path"]] if c.get("video_path") else []
        ):
            if digest(path) != hashes[path]:
                raise ValueError("Input changed before run")
        capture = CaptureController(interface=interface, capture_dir=str(directory))
        pcap = capture.start(
            filename="capture.pcap",
            filter_expr="tcp port 443",
            metadata={"scenario": job["scenario"], "run_index": job["run_index"]},
        )
        if pcap is None:
            raise RuntimeError("Capture failed before provider submission")
        write_json(
            directory / "started.json",
            {
                "time": time.time(),
                "retry_policy": "never automatically resubmit an unresolved attempt",
            },
        )
        try:
            result = CLASSES[c["type"]](NullLLMClient("sa1_http"), logger, c).run(
                "no_emulation", job["run_index"]
            )
            write_json(directory / "result.json", asdict(result))
        finally:
            capture.stop()
        analyze_capture(pcap, result.metadata)
        statuses = [p.get("http_status") for p in result.metadata.get("phases", [])]
        permanent = next((s for s in statuses if s in (401, 402, 403, 404)), None)
        if stop_on_permanent_http_error and not result.success and permanent:
            for pending in plan["jobs"]:
                dest = output / pending["id"]
                if (
                    pending["scenario"] == job["scenario"]
                    and not (dest / "started.json").exists()
                ):
                    dest.mkdir(exist_ok=True)
                    write_json(
                        dest / "blocked.json",
                        {
                            "state": "not_attempted_provider_blocked",
                            "caused_by_job": job["id"],
                            "http_status": permanent,
                            "reason": "Permanent provider/account/access error; no automatic repeated submission",
                        },
                    )
        report(output, plan)
        print(
            json.dumps(
                {
                    "job": job["id"],
                    "success": result.success,
                    "seconds": result.total_latency_sec,
                    "error": result.error_message,
                }
            ),
            flush=True,
        )
    return report(output, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(APP / "configs/sa1.yaml"))
    parser.add_argument("--assets", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--interface", default="enP7s7")
    parser.add_argument("--video-path")
    parser.add_argument(
        "--stop-on-permanent-http-error",
        action="store_true",
        help="Leave remaining repetitions blocked after HTTP 401/402/403/404 for that scenario",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit the prepared provider jobs; without this flag only a plan is written",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    print(
        json.dumps(
            run(
                args.output,
                args.config,
                args.assets,
                args.repetitions,
                args.execute,
                args.scenarios,
                args.interface,
                args.video_path,
                args.stop_on_permanent_http_error,
            ),
            indent=2,
        )
    )
