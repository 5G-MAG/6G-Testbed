#!/usr/bin/env python3
"""Run a reproducible scenario × profile × repetition provider campaign.

Every job uses fresh private client/router namespaces. Results and PCAPs live
outside the checkout. Failed and interrupted jobs are never silently retried.
"""

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import pwd
import random
import signal
import subprocess
import sys
import time

from sa1_campaign import (
    APP,
    CLASSES,
    CaptureController,
    NullLLMClient,
    TrafficLogger,
    analyze_capture,
    digest,
    validate_media_config,
    write_json,
)
from netemu.profile import NetworkProfile, load_profile_catalog
from sa1_network import IsolatedPath
import yaml

DEFAULT_SCENARIOS = [
    "sa1_chat_small",
    "sa1_image_text",
    "sa1_video_generation",
    "sa1_photo_enhancement",
]


def prepare(
    output,
    config_path,
    assets_path,
    profiles_path,
    profile_names,
    selected,
    repetitions,
    interface,
    timeout=540,
):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(Path(config_path).read_text())
    assets = json.loads(Path(assets_path).read_text())
    catalog = load_profile_catalog(profiles_path)["profiles"]
    if len(set(selected)) != len(selected) or len(set(profile_names)) != len(
        profile_names
    ):
        raise ValueError("Duplicate scenario or profile selection")
    if repetitions < 1 or timeout <= 0:
        raise ValueError("Repetitions and timeout must be positive")
    scenarios, inputs = {}, {}
    for name in selected:
        c = dict(raw["scenarios"][name])
        c.update(
            scenario_id=name,
            provider="sa1_http",
            retry_count=0,
            max_duration_sec=timeout,
            output_dir=str(output / "media"),
        )
        if c.get("image_asset"):
            value = assets[c.pop("image_asset")]
            c["image_paths"] = value if isinstance(value, list) else [value]
        if c.get("prompt_bytes"):
            size = c.pop("prompt_bytes")
            c["prompt"] += " " * max(0, size - len(c["prompt"].encode()))
        if c.get("video_path"):
            c["video_path"] = str((APP / c["video_path"]).resolve())
        validate_media_config(c, credentials=False)
        for path in c.get("image_paths", []) + (
            [c["video_path"]] if c.get("video_path") else []
        ):
            inputs[path] = digest(path)
        scenarios[name] = c
    profiles = {name: catalog[name] for name in profile_names}
    for name, p in profiles.items():
        NetworkProfile.from_dict(name, p).validate()
    jobs = [
        {
            "id": f"{name}__{profile}__{index:03d}",
            "scenario": name,
            "profile": profile,
            "run_index": index,
        }
        for index in range(repetitions)
        for name in selected
        for profile in profile_names
    ]
    random.Random(raw["seed"]).shuffle(jobs)
    source_paths = [
        Path(__file__),
        APP / "sa1_network.py",
        APP / "sa1_campaign.py",
        APP / "sa1_controlled.py",
        APP / "sa1_capture_filter.py",
        APP / "scenarios/media.py",
        APP / "scenarios/base.py",
        APP.parent / "netemu/src/netemu/capture.py",
        APP.parent / "netemu/src/netemu/pcap.py",
        APP.parent / "netemu/src/netemu/profile.py",
        APP.parent / "netemu/src/netemu/tc.py",
    ]
    plan = {
        "schema_version": 1,
        "scope": "real providers over isolated bidirectional network emulation",
        "excluded": ["gaming_enhancement", "visual_assistant"],
        "scenarios": scenarios,
        "profiles": profiles,
        "jobs": jobs,
        "repetitions": repetitions,
        "seed": raw["seed"],
        "interface": interface,
        "timeout_sec": timeout,
        "retry_count": 0,
        "concurrency": 1,
        "input_hashes": inputs,
        "config_sha256": digest(config_path),
        "profiles_sha256": digest(profiles_path),
        "source_files_sha256": {
            str(p.relative_to(APP.parent)): digest(p) for p in source_paths
        },
        "one_way_latency_scope": "Only the local emulated segment is observable; full provider one-way delay is unavailable",
        "packet_size_scope": "TCP header plus payload (IP payload); offloads disabled on every private veth",
        "capture_scope": "Private interfaces only, then exact scenario HTTP TCP endpoint verification",
    }
    path = output / "plan.json"
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != v for k, v in plan.items()):
            raise ValueError(
                "Plan, inputs or source changed; use a new campaign directory"
            )
        return old
    plan["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=APP, text=True
    ).strip()
    plan["created_at"] = time.time()
    for relative in plan["source_files_sha256"]:
        snapshot = output / "source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((APP.parent / relative).read_bytes())
    write_json(path, plan)
    return plan


def summarize(output, plan):
    groups = []
    for scenario in plan["scenarios"]:
        for profile in plan["profiles"]:
            results, unresolved, blocked = [], 0, 0
            for job in plan["jobs"]:
                if (job["scenario"], job["profile"]) != (scenario, profile):
                    continue
                directory = output / job["id"]
                if (directory / "result.json").exists():
                    results.append(json.loads((directory / "result.json").read_text()))
                elif (directory / "started.json").exists():
                    unresolved += 1
                elif (directory / "blocked.json").exists():
                    blocked += 1
            groups.append(
                {
                    "scenario": scenario,
                    "profile": profile,
                    "planned": plan["repetitions"],
                    "attempted": len(results) + unresolved,
                    "completed": len(results),
                    "successes": sum(r["success"] for r in results),
                    "failures": sum(not r["success"] for r in results),
                    "unresolved": unresolved,
                    "blocked": blocked,
                }
            )
    summary = {
        "planned": len(plan["jobs"]),
        "completed": sum(g["completed"] for g in groups),
        "successes": sum(g["successes"] for g in groups),
        "failures": sum(g["failures"] for g in groups),
        "unresolved": sum(g["unresolved"] for g in groups),
        "blocked": sum(g["blocked"] for g in groups),
        "groups": groups,
        "updated_at": time.time(),
    }
    write_json(output / "summary.json", summary)
    return summary


def worker(job_file):
    spec = json.loads(Path(job_file).read_text())
    job, c = spec["job"], spec["config"]
    directory = Path(job_file).parent
    logger = TrafficLogger(str(directory / "traffic.db"))
    capture = CaptureController("eth0", str(directory))
    if (
        capture.start(
            "capture.pcap",
            "tcp port 443",
            {"job": job, "isolation": "dedicated client namespace"},
        )
        is None
    ):
        raise RuntimeError("Capture failed before submission")
    try:
        write_json(
            directory / "started.json",
            {
                "time": time.time(),
                "retry_policy": "never automatically repeat an attempt",
            },
        )
        result = CLASSES[c["type"]](NullLLMClient("sa1_http"), logger, c).run(
            job["profile"], job["run_index"]
        )
        write_json(directory / "result.json", asdict(result))
    finally:
        capture.stop()


def execute(output, plan, limit=None):
    output = Path(output)
    # A second invocation must not race with a paid provider submission.
    lock = (output / ".campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for c in plan["scenarios"].values():
        validate_media_config(c, credentials=True)
    count = 0
    for job in plan["jobs"]:
        directory = output / job["id"]
        if any(
            (directory / name).exists()
            for name in ("result.json", "started.json", "blocked.json")
        ):
            continue
        if limit is not None and count >= limit:
            break
        for path, expected in plan["input_hashes"].items():
            if digest(path) != expected:
                raise ValueError("Campaign input changed")
        directory.mkdir(exist_ok=True)
        config = plan["scenarios"][job["scenario"]]
        write_json(directory / "job.json", {"job": job, "config": config})
        profile = NetworkProfile.from_dict(
            job["profile"], plan["profiles"][job["profile"]]
        )
        path = IsolatedPath(profile, plan["interface"])
        edge = directory / "edge"
        edge.mkdir(exist_ok=True)
        print(json.dumps({"starting": job["id"], "time": time.time()}), flush=True)
        try:
            with path:
                write_json(directory / "network.json", path.evidence)
                capture = CaptureController(path.host_device, str(edge))
                if (
                    capture.start(
                        "capture.pcap",
                        "tcp port 443",
                        {"job": job, "isolation": "dedicated host veth"},
                    )
                    is None
                ):
                    raise RuntimeError("Edge capture failed before submission")
                try:
                    user = pwd.getpwuid(os.getuid()).pw_name
                    cmd = [
                        "sudo",
                        "-n",
                        "ip",
                        "netns",
                        "exec",
                        path.client,
                        "sudo",
                        "-n",
                        "-H",
                        "-u",
                        user,
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(directory / "job.json"),
                    ]
                    with (directory / "worker.log").open("w") as log:
                        proc = subprocess.Popen(
                            cmd, stdout=log, stderr=log, start_new_session=True
                        )
                        try:
                            code = proc.wait(timeout=plan["timeout_sec"] + 90)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGTERM)
                            proc.wait(timeout=15)
                            write_json(
                                directory / "interrupted.json",
                                {
                                    "reason": "Worker watchdog timeout; no resubmission",
                                    "time": time.time(),
                                },
                            )
                            code = -15
                    if code and not (directory / "started.json").exists():
                        raise RuntimeError(
                            f"Worker failed before provider attempt; inspect {directory / 'worker.log'}"
                        )
                finally:
                    capture.stop()
                    path.evidence["qdisc_after"] = path.qdiscs()
        finally:
            write_json(directory / "network.json", path.evidence)
        if (directory / "result.json").exists():
            result = json.loads((directory / "result.json").read_text())
            write_json(edge / "result.json", result)
            for point in (directory, edge):
                analyze_capture(point / "capture.pcap", result["metadata"])
            statuses = [
                p.get("http_status") for p in result["metadata"].get("phases", [])
            ]
            permanent = next((s for s in statuses if s in (401, 402, 403, 404)), None)
            if permanent and not result["success"]:
                for pending in plan["jobs"]:
                    dest = output / pending["id"]
                    if (
                        pending["scenario"] == job["scenario"]
                        and not (dest / "started.json").exists()
                    ):
                        dest.mkdir(exist_ok=True)
                        write_json(
                            dest / "blocked.json",
                            {"http_status": permanent, "caused_by": job["id"]},
                        )
        count += 1
        summary = summarize(output, plan)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "groups"}), flush=True
        )
    return summarize(output, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--output")
    parser.add_argument("--assets")
    parser.add_argument("--config", default=str(APP / "configs/sa1.yaml"))
    parser.add_argument("--profiles-config", default=str(APP / "configs/profiles.yaml"))
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=540)
    parser.add_argument("--interface", default="enP7s7")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        if not args.output or not args.assets:
            parser.error("--output and --assets are required")
        names = (
            args.profiles
            or yaml.safe_load((APP / "configs/scenarios.yaml").read_text())[
                "runner_profile_sets"
            ]["full_suite_profiles"]
        )
        plan = prepare(
            args.output,
            args.config,
            args.assets,
            args.profiles_config,
            names,
            args.scenarios,
            args.repetitions,
            args.interface,
            args.timeout,
        )
        if args.execute:
            print(
                json.dumps(execute(Path(args.output).resolve(), plan, args.limit)),
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {
                        "prepared": len(plan["jobs"]),
                        "profiles": names,
                        "scenarios": args.scenarios,
                    }
                )
            )
