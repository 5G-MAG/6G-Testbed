"""Delay across the two captures bracketing the private emulated segment.

Encrypted TCP bytes are observable; application data-unit boundaries and the
public provider's receive timestamp are not. Never label these full-path delay.
"""

from collections import defaultdict, deque
import gzip
import hashlib
import json
import math
from pathlib import Path
import socket

import dpkt
import numpy as np

from sa1_excel_metrics import GAPS_MS, burst_slices, digest, distribution


def read_trace(path, endpoints):
    records = []
    with Path(path).open("rb") as stream:
        for stamp, buf in dpkt.pcap.Reader(stream):
            ip = dpkt.ethernet.Ethernet(buf).data
            if not isinstance(ip, dpkt.ip.IP) or not isinstance(ip.data, dpkt.tcp.TCP):
                raise ValueError("Expected verified IPv4 TCP scenario capture")
            tcp = ip.data
            src = (socket.inet_ntoa(ip.src), tcp.sport)
            dst = (socket.inet_ntoa(ip.dst), tcp.dport)
            if (src, dst) in endpoints:
                flow, direction = (src, dst), "ul"
            elif (dst, src) in endpoints:
                flow, direction = (dst, src), "dl"
            else:
                raise ValueError("Unrelated traffic in segment trace")
            key = (
                flow,
                direction,
                tcp.seq,
                tcp.ack,
                tcp.flags,
                ip.id,
                hashlib.sha256(bytes(tcp.data)).digest(),
            )
            records.append(
                {
                    "t": int(round(float(stamp) * 1e6)),
                    "flow": flow,
                    "direction": direction,
                    "seq": tcp.seq,
                    "payload": len(tcp.data),
                    "size": len(ip.data),
                    "key": key,
                }
            )
    return sorted(records, key=lambda p: p["t"])


def unique_delivery(source, destination):
    """First delivery of observed TCP bytes, including recovery by resegmentation.

    Coordinate compression permits reordered first transmissions and sequence
    holes; only bytes actually observed at the source become required bytes.
    One connection must fit within the TCP serial-number half-space.
    """
    source = [p for p in source if p["payload"]]
    destination = [p for p in destination if p["payload"]]
    if not source:
        return []
    origin = source[0]["seq"]

    def ranges(packets):
        result = []
        for p in packets:
            left = (p["seq"] - origin) % (2**32)
            if left > 2**31:
                left -= 2**32
            result.append((p["t"], left, left + p["payload"]))
        return result

    sent, received = ranges(source), ranges(destination)
    cuts = np.unique(
        [value for _, left, right in sent + received for value in (left, right)]
    )
    first_tx = np.full(len(cuts) - 1, np.inf)
    first_rx = np.full(len(cuts) - 1, np.inf)
    for packets, times in ((sent, first_tx), (received, first_rx)):
        for t, left, right in packets:
            a, b = np.searchsorted(cuts, [left, right])
            times[a:b] = np.minimum(times[a:b], t)
    pieces = []
    # Combine adjacent byte intervals from the same first source transmission.
    for index, t in enumerate(first_tx):
        if not math.isfinite(t):
            continue
        end = first_rx[index]
        if math.isfinite(end) and end < t:
            raise ValueError("Receiver timestamp precedes first observed transmission")
        size = int(cuts[index + 1] - cuts[index])
        if (
            pieces
            and pieces[-1]["t"] == int(t)
            and pieces[-1]["end"] == int(cuts[index])
        ):
            piece = pieces[-1]
            piece["bytes"] += size
            piece["end"] = int(cuts[index + 1])
            piece["finish"] = max(piece["finish"], float(end))
        else:
            pieces.append(
                {
                    "t": int(t),
                    "bytes": size,
                    "end": int(cuts[index + 1]),
                    "finish": float(end),
                }
            )
    return sorted(pieces, key=lambda p: p["t"])


def direction_metrics(source, destination):
    # Exact packet attempts: IP ID distinguishes most retransmissions; count
    # ambiguous repeated identities explicitly instead of concealing them.
    queues = defaultdict(deque)
    identities = defaultdict(int)
    for p in source:
        queues[p["key"]].append(p)
        identities[p["key"]] += 1
    delays, data_delays = [], []
    unmatched = 0
    for p in destination:
        candidates = queues[p["key"]]
        if not candidates or candidates[0]["t"] > p["t"]:
            unmatched += 1
            continue
        sent = candidates.popleft()
        delay = (p["t"] - sent["t"]) / 1000
        delays.append(delay)
        if p["payload"]:
            data_delays.append(delay)
    pieces = []
    for flow in {p["flow"] for p in source}:
        pieces.extend(
            unique_delivery(
                [p for p in source if p["flow"] == flow],
                [p for p in destination if p["flow"] == flow],
            )
        )
    pieces.sort(key=lambda p: p["t"])
    times = np.asarray([p["t"] for p in pieces], dtype="i8")
    bursts = []
    for gap in GAPS_MS:
        sizes, spans, latencies = [], [], []
        incomplete = 0
        for a, b in burst_slices(times, gap):
            selected = pieces[a:b]
            sizes.append(sum(p["bytes"] for p in selected))
            spans.append((selected[-1]["t"] - selected[0]["t"]) / 1000)
            finish = max(p["finish"] for p in selected)
            if math.isfinite(finish):
                latencies.append((finish - selected[0]["t"]) / 1000)
            else:
                incomplete += 1
        bursts.append(
            {
                "idle_gap_ms": gap,
                "bursts": len(sizes),
                "incomplete": incomplete,
                "unique_tcp_payload_bytes": distribution(sizes),
                "source_duration_ms": distribution(spans),
                "segment_delivery_ms": distribution(latencies),
            }
        )
    return {
        "source_packets": len(source),
        "destination_packets": len(destination),
        "matched_packet_attempts": len(delays),
        "unmatched_or_duplicate_destination_packets": unmatched,
        "source_attempts_not_matched": sum(map(len, queues.values())),
        "ambiguous_source_identities": sum(n > 1 for n in identities.values()),
        "matched_all_tcp_delay_ms": distribution(delays),
        "matched_data_packet_delay_ms": distribution(data_delays),
        "first_byte_piece_delivery_ms": distribution(
            [
                (p["finish"] - p["t"]) / 1000
                for p in pieces
                if math.isfinite(p["finish"])
            ]
        ),
        "source_unique_payload_bytes": sum(p["bytes"] for p in pieces),
        "incomplete_source_pieces": sum(not math.isfinite(p["finish"]) for p in pieces),
        "packet_thresholds": [
            {
                "threshold_ms": threshold,
                "delivered_data_packets": len(data_delays),
                "over_threshold": sum(d > threshold for d in data_delays),
            }
            for threshold in (20, 30, 50, 100, 400, 1000, 1500)
        ],
        "source_bursts": bursts,
    }


def build_segment(run, output):
    run, output = Path(run), Path(output)
    inputs = {
        name: digest(run / name)
        for name in (
            "result.json",
            "capture.scenario.pcap",
            "capture.scenario.pcap.metadata.json",
            "edge/capture.scenario.pcap",
            "edge/capture.scenario.pcap.metadata.json",
            "network.json",
        )
    }
    if output.exists():
        with gzip.open(output, "rt") as stream:
            old = json.load(stream)
        if (
            old.get("source_hash") == digest(__file__)
            and old.get("input_hashes") == inputs
        ):
            return old
    result = json.loads((run / "result.json").read_text())
    endpoints = {
        (tuple(p["local_endpoint"]), tuple(p["remote_endpoint"]))
        for p in result["metadata"]["phases"]
        if p.get("local_endpoint") and p.get("remote_endpoint")
    }
    client = read_trace(run / "capture.scenario.pcap", endpoints)
    edge = read_trace(run / "edge/capture.scenario.pcap", endpoints)
    complete = all(
        json.loads((run / name).read_text())["capture_stats"].get(
            "packets_dropped_by_kernel"
        )
        == 0
        for name in (
            "capture.scenario.pcap.metadata.json",
            "edge/capture.scenario.pcap.metadata.json",
        )
    )
    oversized = sum(p["size"] > 1480 for p in client + edge)
    analysis = {
        "run_id": run.name,
        "source_hash": digest(__file__),
        "input_hashes": inputs,
        "measurement_valid": complete and oversized == 0,
        "oversized_records": oversized,
        "capture_complete": complete,
        "scope": "Local emulated segment only, same kernel clock, microsecond timestamps; timestamp placement not calibrated",
        "source_burst_scope": "Unique encrypted TCP bytes grouped at their first source transmission, strict idle-gap threshold; includes retransmission recovery, excludes duplicate bytes; not an application data unit",
        "directions": {},
    }
    for direction, source, destination in (("ul", client, edge), ("dl", edge, client)):
        analysis["directions"][direction] = direction_metrics(
            [p for p in source if p["direction"] == direction],
            [p for p in destination if p["direction"] == direction],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as stream:
        json.dump(analysis, stream, allow_nan=False)
    return analysis
