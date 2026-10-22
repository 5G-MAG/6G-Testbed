#!/usr/bin/env python3
"""Controlled SA1 TCP transfers over three isolated Linux network namespaces.

Both endpoint captures use the same kernel clock. Shaping is on the router,
so the sender capture precedes impairment and the receiver capture follows it.
This measures a controlled network, not a public provider or model quality.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "netemu/src"))
from netemu.profile import load_profile_catalog, NetworkProfile
from netemu.tc import shaping_commands
import dpkt


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def receive(sock, n):
    parts = bytearray()
    while len(parts) < n:
        chunk = sock.recv(min(n - len(parts), 65536))
        if not chunk:
            raise EOFError("Incomplete data unit")
        parts.extend(chunk)
    return bytes(parts)


def worker(role, job_path):
    job = json.loads(Path(job_path).read_text())
    if role == "server":
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("10.231.2.2", 8098))
        server.listen(1)
        server.settimeout(90)
        Path(job["ready"]).touch()
        sock, address = server.accept()
        with sock:
            sock.settimeout(90)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            length = struct.unpack("!Q", receive(sock, 8))[0]
            if length != job["upload_bytes"]:
                raise ValueError("Unexpected input size")
            data = receive(sock, length)
            if hashlib.sha256(data).hexdigest() != job["upload_sha256"]:
                raise ValueError("Upload digest mismatch")
            output = Path(job["download_file"]).read_bytes()
            sock.sendall(struct.pack("!Q", len(output)))
            sock.sendall(output)
            receive(sock, 1)  # client confirms complete reception
        server.close()
    else:
        data = Path(job["upload_file"]).read_bytes()
        start = time.monotonic()
        with socket.create_connection(("10.231.2.2", 8098), timeout=90) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_port = sock.getsockname()[1]
            sock.sendall(struct.pack("!Q", len(data)))
            sock.sendall(data)
            length = struct.unpack("!Q", receive(sock, 8))[0]
            if length != job["download_bytes"]:
                raise ValueError("Unexpected output size")
            output = receive(sock, length)
            if hashlib.sha256(output).hexdigest() != job["download_sha256"]:
                raise ValueError("Download digest mismatch")
            sock.sendall(b"K")
        write_json(
            job["client_result"],
            {
                "success": True,
                "client_port": client_port,
                "duration_sec": time.monotonic() - start,
            },
        )


def packets(path):
    with open(path, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for ts, buf in reader:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP) or not isinstance(ip.data, dpkt.tcp.TCP):
                continue
            tcp = ip.data
            if tcp.sport != 8098 and tcp.dport != 8098:
                continue
            yield {
                "t": float(ts),
                "direction": "ul" if tcp.dport == 8098 else "dl",
                "seq": tcp.seq,
                "ack": tcp.ack,
                "flags": tcp.flags,
                "ip_id": ip.id,
                "payload": bytes(tcp.data),
                "ip_payload_bytes": len(ip.data),
            }


def matched_metrics(sender_path, receiver_path, direction, body_bytes):
    source = [p for p in packets(sender_path) if p["direction"] == direction]
    dest = [p for p in packets(receiver_path) if p["direction"] == direction]
    # Unit is the length-prefixed body on this run's dedicated connection.
    syn = next((p for p in source if p["flags"] & dpkt.tcp.TH_SYN), None)
    if syn is None:
        raise ValueError("Missing handshake; cannot identify byte interval")
    start_seq = (syn["seq"] + 1 + 8) % (2**32)

    def offsets(p):
        start = (p["seq"] - start_seq) % (2**32)
        if start > 2**31:
            start -= 2**32
        return max(start, 0), min(start + len(p["payload"]), body_bytes)

    def in_unit(p):
        a, b = offsets(p)
        return bool(p["payload"]) and a < b

    src = [p for p in source if in_unit(p)]
    dst = [p for p in dest if in_unit(p)]
    coverage = []
    for p in dst:
        a, b = offsets(p)
        coverage.append((a, b))
    end = 0
    for a, b in sorted(coverage):
        if a > end:
            break
        end = max(end, b)
    complete = end >= body_bytes

    # Include IP ID to distinguish retransmission attempts with identical TCP bytes.
    def key(p):
        return (p["seq"], p["ack"], p["flags"], p["ip_id"], p["payload"])

    sent = defaultdict(list)
    for p in src:
        sent[key(p)].append(p["t"])
    delays = []
    unmatched = 0
    duplicates = 0
    matched_times = {}
    for p in dst:
        times = sent.get(key(p), [])
        candidate = next((i for i, t in enumerate(times) if t <= p["t"]), None)
        if candidate is None:
            if key(p) in matched_times and p["t"] >= matched_times[key(p)]:
                duplicates += 1
            else:
                unmatched += 1
            continue
        transmitted = times.pop(candidate)
        matched_times[key(p)] = transmitted
        delays.append((p["t"] - transmitted) * 1000)
    # First delivery of each required byte: duplicates after completion must not extend latency.
    intervals = []
    finish = None
    for p in sorted(dst, key=lambda x: x["t"]):
        intervals.append(offsets(p))
        merged = []
        for a, b in sorted(intervals):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        intervals = merged
        if intervals and intervals[0][0] == 0 and intervals[0][1] >= body_bytes:
            finish = p["t"]
            break
    duration_ms = (
        (finish - min(p["t"] for p in src)) * 1000
        if finish is not None and src
        else None
    )
    unique_source_bytes = 0
    last_end = 0
    for a, b in sorted(map(offsets, src)):
        unique_source_bytes += max(0, b - max(a, last_end))
        last_end = max(last_end, b)
    return {
        "direction": direction,
        "body_bytes": body_bytes,
        "complete": complete,
        "source_unit_started": bool(src),
        "observation_elapsed_ms": (
            (max(p["t"] for p in source + dest) - min(p["t"] for p in src)) * 1000
            if src
            else None
        ),
        "burst_transfer_ms": duration_ms,
        "matched_data_packet_delays_ms": delays,
        "unmatched_received_data_packets": unmatched,
        "duplicate_received_data_packets": duplicates,
        "source_data_packets": len(src),
        "undelivered_source_attempts": sum(len(v) for v in sent.values()),
        "source_all_packet_sizes_ip_payload": [p["ip_payload_bytes"] for p in source],
        "source_data_packet_sizes_ip_payload": [p["ip_payload_bytes"] for p in src],
        "receiver_all_packet_sizes_ip_payload": [p["ip_payload_bytes"] for p in dest],
        "source_retransmitted_payload_bytes": max(
            0, sum(b - a for a, b in map(offsets, src)) - unique_source_bytes
        ),
        "observed_max_ip_payload": max(
            (p["ip_payload_bytes"] for p in source + dest), default=0
        ),
        "clock_scope": "same kernel CLOCK_REALTIME, pcap microsecond resolution; software timestamp placement uncertainty not calibrated",
        "packet_delay_scope": "successfully matched data packet attempts; duplicates counted separately; lost attempts have no delivered delay",
    }


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    import numpy as np

    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "observed_max": values[-1],
    }


def wilson(misses, n):
    if not n:
        return None
    import math

    z = 1.96
    p = misses / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, center - half), min(1, center + half)]


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["workload"], r["profile"])].append(r)
    output = []
    for (workload, profile), runs in sorted(groups.items()):
        item = {
            "workload": workload,
            "profile": profile,
            "runs": len(runs),
            "successes": sum(r["success"] for r in runs),
            "measurement_invalid_runs": sum(
                not r.get("measurement_valid") for r in runs
            ),
            "directions": {},
        }
        for d in ["ul", "dl"]:
            units = [
                r[d]
                for r in runs
                if d in r
                and r.get("measurement_valid")
                and r[d].get("source_unit_started", r[d]["source_data_packets"] > 0)
            ]
            packet = [x for u in units for x in u["matched_data_packet_delays_ms"]]
            deadline = runs[0]["thresholds"].get(d)
            completed = [u["burst_transfer_ms"] for u in units if u["complete"]]
            deadline_units = [
                u
                for u in units
                if u["complete"]
                or (
                    deadline is not None
                    and (u.get("observation_elapsed_ms") or 0) >= deadline
                )
            ]
            misses = (
                sum(
                    not u["complete"] or u["burst_transfer_ms"] > deadline
                    for u in deadline_units
                )
                if deadline is not None
                else None
            )
            photo_dl_units = (
                [
                    u
                    for u in units
                    if u["complete"] or (u.get("observation_elapsed_ms") or 0) >= 1000
                ]
                if workload.startswith("photo") and d == "dl"
                else []
            )
            item["directions"][d] = {
                "burst_transfer_ms": stats(completed),
                "delivered_packet_delay_ms": stats(packet),
                "deadline_ms": deadline,
                "deadline_misses": misses,
                "units": len(units),
                "incomplete_units": sum(not u["complete"] for u in units),
                "deadline_assessable_units": len(deadline_units)
                if deadline is not None
                else 0,
                "deadline_miss_proportion_ci95": wilson(misses, len(deadline_units))
                if deadline is not None
                else None,
                "packet_deadline_ms": 30
                if workload.startswith("chatbot")
                else 20
                if workload.startswith(("image", "visual"))
                else None,
                "packet_deadline_misses": sum(
                    x > (30 if workload.startswith("chatbot") else 20) for x in packet
                )
                if workload.startswith(("chatbot", "image", "visual"))
                else None,
                "photo_dl_1000ms_misses": sum(
                    not u["complete"] or u["burst_transfer_ms"] > 1000
                    for u in photo_dl_units
                )
                if workload.startswith("photo") and d == "dl"
                else None,
                "photo_dl_1000ms_assessable_units": len(photo_dl_units),
                "photo_dl_1000ms_miss_proportion_ci95": wilson(
                    sum(
                        not u["complete"] or u["burst_transfer_ms"] > 1000
                        for u in photo_dl_units
                    ),
                    len(photo_dl_units),
                )
                if photo_dl_units
                else None,
                "achieved_body_mbps": stats(
                    [
                        u["body_bytes"] * 0.008 / u["burst_transfer_ms"]
                        for u in units
                        if u["burst_transfer_ms"] and u["burst_transfer_ms"] > 0
                    ]
                ),
                "packet_size_ip_payload": stats(
                    [v for u in units for v in u["source_all_packet_sizes_ip_payload"]]
                ),
                "data_packet_size_ip_payload": stats(
                    [
                        v
                        for u in units
                        for v in u.get("source_data_packet_sizes_ip_payload", [])
                    ]
                ),
                "missing_or_failed_runs": len(runs) - len(units),
            }
        output.append(item)
    return output


def campaign(
    output, profiles_path, repetitions, selected, raw_payload=None, timeout_sec=20
):
    if os.geteuid() != 0:
        raise PermissionError("Run with sudo; only private namespaces are modified")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise ValueError("Choose a fresh campaign directory")
    catalog = load_profile_catalog(profiles_path)["profiles"]
    # Explicit finite target-capacity cases, separate from calibrated field profiles.
    for rate in [100, 200, 400, 600]:
        catalog[f"capacity_{rate}"] = {
            "rate_mbit": rate,
            "rate_mode": "htb",
            "delay_ms": 1,
            "limit_packets": 10000,
        }
    names = selected or [
        "no_emulation",
        "5g_urban",
        "satellite_leo",
        "capacity_100",
        "capacity_400",
        "capacity_600",
    ]
    for name in names:
        if name not in catalog:
            raise ValueError("Unknown profile " + name)
        NetworkProfile.from_dict(name, catalog[name])
    workloads = [
        ("chatbot_small", 500, 500, {"ul": 20, "dl": 20}),
        ("chatbot_large", 6000, 500, {"ul": 20, "dl": 20}),
        ("image_prompt", 500, 1000000, {"ul": 50}),
        ("image_conditioned", 400000, 1000000, {"ul": 50}),
        ("visual_assistant", 20000000, 500, {"ul": 400}),
        ("photo_spec_payload", 20000000, 20000000, {"ul": 1500, "dl": 1500}),
    ]
    if raw_payload:
        workloads.append(
            (
                "photo_actual_compressed_raw",
                Path(raw_payload).stat().st_size,
                20000000,
                {"ul": 1500, "dl": 1500},
            )
        )
    rng = random.Random(26870)
    jobs = [(w, p, i) for w in workloads for p in names for i in range(repetitions)]
    rng.shuffle(jobs)
    write_json(
        output / "plan.json",
        {
            "seed": 26870,
            "profiles": {p: catalog[p] for p in names},
            "repetitions": repetitions,
            "workloads": workloads,
            "timeout_sec": timeout_sec,
            "scope": "controlled TCP data units; no inference; gaming excluded",
            "profiles_sha256": hashlib.sha256(
                Path(profiles_path).read_bytes()
            ).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_commit": command(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1]
            ).stdout.strip(),
        },
    )
    payloads = output / "payloads"
    payloads.mkdir()
    sizes = {n for _, a, b, _ in workloads for n in [a, b]}
    for n in sizes:
        p = payloads / f"{n}.bin"
        if raw_payload and n == Path(raw_payload).stat().st_size:
            p.write_bytes(Path(raw_payload).read_bytes())
        else:
            p.write_bytes(random.Random(n).randbytes(n))
    token = "s1" + uuid.uuid4().hex[:8]
    client, router, server = [token + s for s in ("c", "r", "s")]
    rows = []
    processes = []

    def ns(n, args, **kwargs):
        return command(["ip", "netns", "exec", n, *args], **kwargs)

    try:
        for n in [client, router, server]:
            command(["ip", "netns", "add", n])
            ns(n, ["ip", "link", "set", "lo", "up"])
        for a, b, left, right, subnet in [
            ("c0", "r0", client, router, 1),
            ("r1", "s0", router, server, 2),
        ]:
            ns(left, ["ip", "link", "add", a, "type", "veth", "peer", "name", b])
            ns(left, ["ip", "link", "set", b, "netns", right])
            for n, dev, last in [
                (left, a, 2 if left == client else 1),
                (right, b, 1 if right == router else 2),
            ]:
                ns(n, ["ip", "addr", "add", f"10.231.{subnet}.{last}/24", "dev", dev])
                ns(n, ["ip", "link", "set", dev, "up"])
                ns(
                    n,
                    [
                        "ethtool",
                        "-K",
                        dev,
                        "tso",
                        "off",
                        "gso",
                        "off",
                        "gro",
                        "off",
                        "lro",
                        "off",
                    ],
                )
        ns(client, ["ip", "route", "add", "default", "via", "10.231.1.1"])
        ns(server, ["ip", "route", "add", "default", "via", "10.231.2.1"])
        ns(router, ["sysctl", "-w", "net.ipv4.ip_forward=1"])
        write_json(
            output / "offload-settings.json",
            {
                n: ns(n, ["ethtool", "-k", dev]).stdout
                for n, dev in [(client, "c0"), (server, "s0")]
            },
        )
        script = str(Path(__file__).resolve())
        for j, (workload, profile, index) in enumerate(jobs):
            name, ul, dl, thresholds = workload
            run = output / f"{j:04d}_{name}_{profile}_{index}"
            run.mkdir()
            spec = catalog[profile]
            for dev, direction in [("r1", "uplink"), ("r0", "downlink")]:
                subprocess.run(
                    [
                        "ip",
                        "netns",
                        "exec",
                        router,
                        "tc",
                        "qdisc",
                        "del",
                        "dev",
                        dev,
                        "root",
                    ],
                    capture_output=True,
                )
                resolved = NetworkProfile.from_dict(profile, spec)
                shape = (
                    (resolved.uplink or resolved) if direction == "uplink" else resolved
                )
                for cmd in shaping_commands(shape, dev):
                    words = shlex.split(cmd)
                    ns(router, words[1:] if words[0] == "sudo" else words)
            job = {
                "upload_file": str(payloads / f"{ul}.bin"),
                "download_file": str(payloads / f"{dl}.bin"),
                "upload_bytes": ul,
                "download_bytes": dl,
                "timeout_sec": timeout_sec,
                "ready": str(run / "ready"),
                "client_result": str(run / "client.json"),
            }
            for d in ["upload", "download"]:
                job[d + "_sha256"] = hashlib.sha256(
                    Path(job[d + "_file"]).read_bytes()
                ).hexdigest()
            write_json(run / "job.json", job)
            row = {
                "workload": name,
                "profile": profile,
                "run_index": index,
                "thresholds": thresholds,
                "success": False,
            }
            captures = []
            proc = None
            try:
                for n, dev, label in [
                    (client, "c0", "client"),
                    (server, "s0", "server"),
                ]:
                    log = (run / f"{label}-tcpdump.log").open("wb")
                    cap = subprocess.Popen(
                        [
                            "ip",
                            "netns",
                            "exec",
                            n,
                            "tcpdump",
                            "-i",
                            dev,
                            "-n",
                            "-s",
                            "0",
                            "-U",
                            "--immediate-mode",
                            "-w",
                            str(run / f"{label}.pcap"),
                            "tcp",
                            "port",
                            "8098",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=log,
                    )
                    captures.append((cap, log, label))
                    processes.append(cap)
                time.sleep(0.15)
                for cap, _, _ in captures:
                    if cap.poll() is not None:
                        raise RuntimeError("Endpoint capture failed")
                proc = subprocess.Popen(
                    [
                        "ip",
                        "netns",
                        "exec",
                        server,
                        sys.executable,
                        script,
                        "worker",
                        "server",
                        str(run / "job.json"),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                processes.append(proc)
                deadline = time.monotonic() + 5
                while not Path(job["ready"]).exists():
                    if proc.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Server not ready")
                    time.sleep(0.01)
                ns(
                    client,
                    [sys.executable, script, "worker", "client", str(run / "job.json")],
                    timeout=timeout_sec,
                )
                proc.wait(timeout=5)
                if proc.returncode:
                    raise RuntimeError("Server transfer failed")
                row["success"] = True
            except Exception as exc:
                row["error"] = str(exc)[:1000]
                row["error_type"] = type(exc).__name__
                if isinstance(exc, subprocess.CalledProcessError):
                    row["worker_stderr"] = exc.stderr[-1500:] if exc.stderr else None
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
                time.sleep(0.1)
                for cap, log, label in captures:
                    if cap.poll() is None:
                        cap.send_signal(signal.SIGINT)
                    cap.wait(timeout=5)
                    log.close()
                    stderr = (run / f"{label}-tcpdump.log").read_text()
                    match = re.search(r"(\d+) packets dropped by kernel", stderr)
                    row[label + "_capture_drops"] = (
                        int(match.group(1)) if match else None
                    )
                for direction, a, b, size in [
                    ("ul", "client", "server", ul),
                    ("dl", "server", "client", dl),
                ]:
                    try:
                        row[direction] = matched_metrics(
                            run / f"{a}.pcap", run / f"{b}.pcap", direction, size
                        )
                    except Exception as exc:
                        row[direction + "_measurement_error"] = str(exc)
                row["measurement_valid"] = all(
                    row.get(x + "_capture_drops") == 0 for x in ["client", "server"]
                ) and all(
                    direction in row
                    and row[direction]["observed_max_ip_payload"] <= 1480
                    and row[direction]["unmatched_received_data_packets"] == 0
                    for direction in ["ul", "dl"]
                )
                row["qdisc_stats"] = {
                    dev: ns(router, ["tc", "-s", "qdisc", "show", "dev", dev]).stdout
                    for dev in ["r0", "r1"]
                }
                row["pcap_sha256"] = {
                    x: hashlib.sha256((run / f"{x}.pcap").read_bytes()).hexdigest()
                    for x in ["client", "server"]
                    if (run / f"{x}.pcap").exists()
                }
                write_json(run / "result.json", row)
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "done": j + 1,
                            "total": len(jobs),
                            "workload": name,
                            "profile": profile,
                            "success": row["success"],
                            "measurement_valid": row["measurement_valid"],
                        }
                    ),
                    flush=True,
                )
        write_json(output / "summary.json", summarize(rows))
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        for n in [client, router, server]:
            subprocess.run(["ip", "netns", "del", n], capture_output=True)
        if os.environ.get("SUDO_UID"):
            for p in [output, *output.rglob("*")]:
                os.chown(p, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", required=True)
        parser.add_argument(
            "--profiles-path",
            default=str(Path(__file__).parent / "configs/profiles.yaml"),
        )
        parser.add_argument("--repetitions", type=int, default=5)
        parser.add_argument("--profiles", nargs="+")
        parser.add_argument("--raw-payload")
        parser.add_argument("--timeout-sec", type=float, default=20)
        args = parser.parse_args()
        if args.repetitions < 1:
            parser.error("repetitions must be positive")
        campaign(
            args.output,
            args.profiles_path,
            args.repetitions,
            args.profiles,
            args.raw_payload,
            args.timeout_sec,
        )
