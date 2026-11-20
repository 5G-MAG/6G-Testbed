"""PCAP evidence for SA1 workbooks. All durations have an explicit scope."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import socket

import dpkt
import numpy as np

GAPS_MS = (5, 10, 20, 50, 100, 200, 500, 1000)
WINDOWS_MS = (1, 5, 10, 50, 100, 1000, 10000)
PACKET_DTYPE = [
    ("t", "i8"),
    ("size", "i8"),
    ("payload", "i8"),
    ("direction", "i1"),
    ("seq", "i8"),
    ("flags", "i8"),
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {
            "n": 0,
            **{
                k: None
                for k in ("mean", "min", "p50", "p95", "p99", "max", "std", "cv")
            },
        }
    mean, std = float(a.mean()), float(a.std())
    q = np.percentile(a, [0, 50, 95, 99, 100])
    return dict(
        n=len(a),
        mean=mean,
        min=float(q[0]),
        p50=float(q[1]),
        p95=float(q[2]),
        p99=float(q[3]),
        max=float(q[4]),
        std=std,
        cv=std / mean if mean else None,
    )


def read_packets(path, endpoints=None):
    """Direction 0=UL, 1=DL. Endpoint filtering excludes unrelated host flows."""
    result = []
    excluded = malformed = 0
    with Path(path).open("rb") as stream:
        reader = dpkt.pcap.Reader(stream)
        if reader.datalink() != dpkt.pcap.DLT_EN10MB:
            raise ValueError("Only Ethernet PCAP captures are supported")
        for ts, buf in reader:
            try:
                ip = dpkt.ethernet.Ethernet(buf).data
                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)) or not isinstance(
                    ip.data, dpkt.tcp.TCP
                ):
                    excluded += 1
                    continue
                tcp = ip.data
                if endpoints is not None:
                    family = (
                        socket.AF_INET
                        if isinstance(ip, dpkt.ip.IP)
                        else socket.AF_INET6
                    )
                    src = (socket.inet_ntop(family, ip.src), tcp.sport)
                    dst = (socket.inet_ntop(family, ip.dst), tcp.dport)
                    if (src, dst) in endpoints:
                        direction = 0
                    elif (dst, src) in endpoints:
                        direction = 1
                    else:
                        excluded += 1
                        continue
                elif tcp.dport == 8098:
                    direction = 0
                elif tcp.sport == 8098:
                    direction = 1
                else:
                    excluded += 1
                    continue
                result.append(
                    (
                        round(float(ts) * 1e6),
                        len(ip.data),
                        len(tcp.data),
                        direction,
                        tcp.seq,
                        tcp.flags,
                    )
                )
            except (dpkt.UnpackError, ValueError, IndexError):
                malformed += 1
    array = np.array(result, dtype=PACKET_DTYPE)
    if len(array):
        array = array[np.argsort(array["t"], kind="stable")]
    return array, {
        "excluded_packet_records": excluded,
        "malformed_packet_records": malformed,
    }


def burst_slices(times, gap_ms):
    """Strict > gap, using integer microseconds to preserve boundary equality."""
    if not len(times):
        return []
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > gap_ms * 1000) + 1, len(times)]
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]


def window_rates(times, sizes, width_ms, start, end):
    """Fixed bins, common connection origin; final bin is padded to full width."""
    width = width_ms * 1000
    count = int((end - start) // width) + 1
    bins = np.zeros(count)
    if len(times):
        np.add.at(bins, ((times - start) // width).astype(int), sizes)
    rates = bins * 8 / (width_ms / 1000)
    active = rates[rates > 0]
    d = distribution(rates)
    return {
        "window_ms": width_ms,
        "window_count": count,
        "grid_span_ms": count * width_ms,
        "active_windows": len(active),
        "active_fraction": len(active) / count,
        "mean_all_windows_mbps": d["mean"] / 1e6,
        "peak_mbps": d["max"] / 1e6,
        "p95_mbps": d["p95"] / 1e6,
        "p99_mbps": d["p99"] / 1e6,
        "cv_all_windows": d["cv"],
        "mean_active_windows_mbps": float(active.mean()) / 1e6 if len(active) else None,
        "peak_to_mean_all": d["max"] / d["mean"] if d["mean"] else None,
        "peak_to_mean_active": float(active.max() / active.mean())
        if len(active)
        else None,
    }


def directional_metrics(packets, start, end):
    """A capture-point distribution, not an interface-to-interface delay."""
    results = {"packet": [], "burst": [], "burst_details": [], "window": []}
    for population in ("all_tcp", "tcp_data"):
        selected = (
            packets if population == "all_tcp" else packets[packets["payload"] > 0]
        )
        times, sizes = selected["t"], selected["size"]
        base = {
            "population": population,
            "packet_count": len(selected),
            "ip_payload_bytes": int(sizes.sum()),
            "observation_span_ms": (end - start) / 1000,
        }
        results["packet"].append(
            {
                **base,
                "size_bytes": distribution(sizes),
                "iat_ms": distribution(np.diff(times) / 1000),
                "oversized_records": int((sizes > 1480).sum()),
                "ack_only_count": int(
                    ((selected["payload"] == 0) & (selected["flags"] == 16)).sum()
                ),
            }
        )
        for width in WINDOWS_MS:
            results["window"].append(
                {**base, **window_rates(times, sizes, width, start, end)}
            )
        for gap in GAPS_MS:
            details = []
            previous = None
            for index, (a, b) in enumerate(burst_slices(times, gap)):
                span = (int(times[b - 1]) - int(times[a])) / 1000
                total = int(sizes[a:b].sum())
                row = {
                    "population": population,
                    "idle_gap_ms": gap,
                    "burst_index": index,
                    "start_offset_ms": (int(times[a]) - start) / 1000,
                    "packet_count": b - a,
                    "ip_payload_bytes": total,
                    "duration_ms": span,
                    "preceding_idle_ms": (int(times[a]) - previous) / 1000
                    if previous is not None
                    else None,
                    "mean_rate_mbps": total * 0.008 / span if span > 0 else None,
                }
                for width in (1, 5, 10, 50, 100):
                    bins = np.bincount(
                        ((times[a:b] - times[a]) // (width * 1000)).astype(int),
                        weights=sizes[a:b],
                    )
                    row[f"peak_{width}ms_mbps"] = float(bins.max()) * 0.008 / width
                details.append(row)
                previous = int(times[b - 1])
            results["burst_details"].extend(details)
            results["burst"].append(
                {
                    **base,
                    "idle_gap_ms": gap,
                    "burst_count": len(details),
                    "size_bytes": distribution(
                        [x["ip_payload_bytes"] for x in details]
                    ),
                    "duration_ms": distribution([x["duration_ms"] for x in details]),
                    "idle_ms": distribution(
                        [
                            x["preceding_idle_ms"]
                            for x in details
                            if x["preceding_idle_ms"] is not None
                        ]
                    ),
                    "mean_rate_mbps": distribution(
                        [
                            x["mean_rate_mbps"]
                            for x in details
                            if x["mean_rate_mbps"] is not None
                        ]
                    ),
                    "active_duration_fraction": sum(x["duration_ms"] for x in details)
                    / ((end - start) / 1000)
                    if end > start
                    else None,
                    "zero_duration_bursts": sum(x["duration_ms"] == 0 for x in details),
                }
            )
    return results


def first_delivery(source, destination, body_bytes):
    """First transmission of new unit bytes to their first successful delivery.

    Retransmissions can deliver an original piece later. Source clustering is
    fixed before inspecting receiver times. Header bytes and duplicate source
    retransmissions are excluded from the new-byte pieces.
    """
    syn = source[(source["flags"] & dpkt.tcp.TH_SYN) != 0]
    if not len(syn):
        raise ValueError("Source SYN missing")
    origin = (int(syn[0]["seq"]) + 9) % (2**32)

    def ranges(array):
        data = array[array["payload"] > 0]
        offset = (data["seq"] - origin) % (2**32)
        offset[offset > 2**31] -= 2**32
        left, right = (
            np.maximum(offset, 0),
            np.minimum(offset + data["payload"], body_bytes),
        )
        keep = left < right
        return data["t"][keep], left[keep], right[keep]

    st, sl, sr = ranges(source)
    dt, dl, dr = ranges(destination)
    first = []
    seen_end = 0
    for t, a, b in zip(st, sl, sr):
        if a > seen_end:
            raise ValueError("Unobserved source byte interval")
        if b > seen_end:
            first.append((int(t), seen_end, int(b)))
            seen_end = int(b)
    if not first:
        return {"pieces": [], "clusters": []}
    cuts = np.unique(np.r_[sl, sr, dl, dr])
    arrival = np.full(len(cuts) - 1, np.inf)
    for t, a, b in zip(dt, dl, dr):
        left, right = np.searchsorted(cuts, [a, b])
        arrival[left:right] = np.minimum(arrival[left:right], t)
    pieces = []
    for t, a, b in first:
        left, right = np.searchsorted(cuts, [a, b])
        finish = float(arrival[left:right].max())
        pieces.append(
            {
                "first_tx_us": t,
                "start_byte": a,
                "end_byte": b,
                "bytes": b - a,
                "complete": math.isfinite(finish),
                "finish_us": int(finish) if math.isfinite(finish) else None,
                "latency_ms": (finish - t) / 1000 if math.isfinite(finish) else None,
            }
        )
    clusters = []
    ts = np.array([x["first_tx_us"] for x in pieces], dtype="i8")
    for gap in GAPS_MS:
        for index, (a, b) in enumerate(burst_slices(ts, gap)):
            part = pieces[a:b]
            complete = all(x["complete"] for x in part)
            latency = (
                (max(x["finish_us"] for x in part) - part[0]["first_tx_us"]) / 1000
                if complete
                else None
            )
            clusters.append(
                {
                    "idle_gap_ms": gap,
                    "burst_index": index,
                    "unique_body_bytes": sum(x["bytes"] for x in part),
                    "source_piece_count": len(part),
                    "source_duration_ms": (
                        part[-1]["first_tx_us"] - part[0]["first_tx_us"]
                    )
                    / 1000,
                    "complete": complete,
                    "transfer_latency_ms": latency,
                }
            )
    return {"pieces": pieces, "clusters": clusters}


def build_controlled(run, analysis_file, output):
    run, output = Path(run), Path(output)
    source_hash = digest(__file__)
    raw = json.loads((run / "result.json").read_text())
    analysis = json.loads(Path(analysis_file).read_text())
    job = json.loads((run / "job.json").read_text())
    signatures = {
        name: digest(run / name)
        for name in ("client.pcap", "server.pcap", "job.json", "result.json")
    }
    signatures["analysis"] = digest(analysis_file)
    if output.exists():
        with gzip.open(output, "rt") as f:
            old = json.load(f)
        if (
            old.get("source_hash") == source_hash
            and old.get("input_hashes") == signatures
        ):
            return old
    for side in ("client", "server"):
        if signatures[side + ".pcap"] != raw["pcap_sha256"][side]:
            raise ValueError("PCAP has changed since capture")
    captures = {}
    diagnostics = {}
    for side in ("client", "server"):
        captures[side], diagnostics[side] = read_packets(run / f"{side}.pcap")
    output_data = {
        "source_hash": source_hash,
        "input_hashes": signatures,
        "run_id": run.name,
        "campaign": "controlled_existing_10",
        "kind": "controlled",
        "scenario": raw["workload"],
        "profile": raw["profile"],
        "run_index": raw["run_index"],
        "success": raw["success"],
        "measurement_valid": analysis["measurement_valid"],
        "source_result": str(run / "result.json"),
        "diagnostics": diagnostics,
        "views": [],
        "first_delivery": {},
    }
    all_t = np.r_[captures["client"]["t"], captures["server"]["t"]]
    start, end = int(all_t.min()), int(all_t.max())
    for direction, number, source, receiver, size in [
        ("ul", 0, "client", "server", job["upload_bytes"]),
        ("dl", 1, "server", "client", job["download_bytes"]),
    ]:
        src = captures[source][captures[source]["direction"] == number]
        dst = captures[receiver][captures[receiver]["direction"] == number]
        for point, array in [("sender", src), ("receiver", dst)]:
            output_data["views"].append(
                {
                    "direction": direction,
                    "point": point,
                    **directional_metrics(array, start, end),
                }
            )
        delivered = first_delivery(src, dst, size)
        deadline = (
            30
            if raw["workload"].startswith("chatbot")
            else 20
            if raw["workload"].startswith(("image", "visual"))
            else None
        )
        pieces = delivered.pop("pieces")
        assessable = [
            p
            for p in pieces
            if p["complete"]
            or (deadline is not None and (end - p["first_tx_us"]) / 1000 >= deadline)
        ]
        output_data["first_delivery"][direction] = {
            **delivered,
            "piece_count": len(pieces),
            "incomplete_pieces": sum(not p["complete"] for p in pieces),
            "piece_delivery_ms": distribution(
                [p["latency_ms"] for p in pieces if p["complete"]]
            ),
            "piece_deadline_ms": deadline,
            "piece_deadline_assessed": len(assessable) if deadline else None,
            "piece_deadline_misses": sum(
                not p["complete"] or p["latency_ms"] > deadline for p in assessable
            )
            if deadline
            else None,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as f:
        json.dump(output_data, f, allow_nan=False)
    return output_data


def build_provider(run, campaign, config, output):
    run, output = Path(run), Path(output)
    raw = json.loads((run / "result.json").read_text())
    meta = raw["metadata"]
    signatures = {
        n: digest(run / n)
        for n in ("result.json", "capture.pcap", "capture.pcap.metadata.json")
    }
    if output.exists():
        with gzip.open(output, "rt") as f:
            old = json.load(f)
        if (
            old.get("source_hash") == digest(__file__)
            and old.get("input_hashes") == signatures
        ):
            return old
    endpoints = {
        (tuple(p["local_endpoint"]), tuple(p["remote_endpoint"]))
        for p in meta.get("phases", [])
        if p.get("local_endpoint") and p.get("remote_endpoint")
    }
    packets, diagnostics = read_packets(run / "capture.pcap", endpoints)
    stats = json.loads((run / "capture.pcap.metadata.json").read_text()).get(
        "capture_stats", {}
    )
    valid = (
        bool(len(packets))
        and stats.get("packets_dropped_by_kernel") == 0
        and not diagnostics["malformed_packet_records"]
    )
    wire_valid = valid and not (packets["size"] > 1480).any()
    result = {
        "source_hash": digest(__file__),
        "input_hashes": signatures,
        "run_id": run.name,
        "campaign": campaign,
        "kind": "provider",
        "scenario": raw["scenario_id"],
        "profile": raw["network_profile"],
        "run_index": raw["run_index"],
        "success": raw["success"],
        "measurement_valid": bool(wire_valid),
        "capture_complete": valid,
        "source_result": str(run / "result.json"),
        "diagnostics": diagnostics,
        "capture_stats": stats,
        "views": [],
    }
    if len(packets):
        start, end = int(packets["t"].min()), int(packets["t"].max())
        for direction, number in [("ul", 0), ("dl", 1)]:
            result["views"].append(
                {
                    "direction": direction,
                    "point": "client_host",
                    **directional_metrics(
                        packets[packets["direction"] == number], start, end
                    ),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as f:
        json.dump(result, f, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled")
    parser.add_argument("--controlled-analysis")
    parser.add_argument("--provider", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if args.controlled:
        for index, run in enumerate(
            sorted(Path(args.controlled).glob("*/result.json"))
        ):
            build_controlled(
                run.parent,
                Path(args.controlled_analysis) / (run.parent.name + ".json"),
                out / "controlled" / (run.parent.name + ".json.gz"),
            )
            if (index + 1) % 20 == 0:
                print(f"Controlled traces analyzed: {index + 1}", flush=True)
    for directory in args.provider:
        directory = Path(directory)
        plan = json.loads((directory / "plan.json").read_text())
        for path in sorted(directory.glob("*/result.json")):
            raw = json.loads(path.read_text())
            build_provider(
                path.parent,
                directory.name,
                plan["scenarios"][raw["scenario_id"]],
                out / directory.name / (path.parent.name + ".json.gz"),
            )
        print(f"Provider traces analyzed: {directory}", flush=True)
