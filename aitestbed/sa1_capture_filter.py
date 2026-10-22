"""Export only the scenario's measured HTTP socket connections from a PCAP.

The broad acquisition file stays local for provenance. The scenario PCAP is
the reportable artifact: no domain-wide, IP-wide or port-wide attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket

import dpkt


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tcp_endpoints(buf):
    try:
        ip = dpkt.ethernet.Ethernet(buf).data
        if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)) or not isinstance(
            ip.data, dpkt.tcp.TCP
        ):
            return None
        family = socket.AF_INET if isinstance(ip, dpkt.ip.IP) else socket.AF_INET6
        return (
            (socket.inet_ntop(family, ip.src), ip.data.sport),
            (socket.inet_ntop(family, ip.dst), ip.data.dport),
        )
    except (dpkt.UnpackError, ValueError, IndexError):
        return None


def filter_capture(run):
    run = Path(run)
    result_file = run / "result.json"
    raw = run / "capture.pcap"
    raw_metadata = run / "capture.pcap.metadata.json"
    output = run / "capture.scenario.pcap"
    manifest_file = run / "capture.scenario.pcap.metadata.json"
    result = json.loads(result_file.read_text())
    endpoints = {
        (tuple(phase["local_endpoint"]), tuple(phase["remote_endpoint"]))
        for phase in result.get("metadata", {}).get("phases", [])
        if phase.get("local_endpoint") and phase.get("remote_endpoint")
    }
    if not endpoints:
        raise ValueError(
            f"No measured socket endpoints for {run.name}; refusing broad attribution"
        )
    signatures = {str(p.name): sha256(p) for p in (raw, raw_metadata, result_file)}
    signatures["filter_source"] = sha256(__file__)
    if output.exists() and manifest_file.exists():
        old = json.loads(manifest_file.read_text())
        if old.get("input_hashes") == signatures and old.get("sha256") == sha256(
            output
        ):
            return old
    allowed = endpoints | {(dst, src) for src, dst in endpoints}
    temporary = output.with_suffix(".pcap.tmp")
    total = retained = 0
    with raw.open("rb") as source, temporary.open("wb") as destination:
        reader = dpkt.pcap.Reader(source)
        if reader.datalink() != dpkt.pcap.DLT_EN10MB:
            raise ValueError("Expected Ethernet PCAP")
        writer = dpkt.pcap.Writer(
            destination,
            snaplen=reader.snaplen,
            linktype=reader.datalink(),
            nano=reader._divisor == 10**9,
        )
        for timestamp, buf in reader:
            total += 1
            if tcp_endpoints(buf) in allowed:
                writer.writepkt(buf, ts=timestamp)
                retained += 1
    if not retained:
        temporary.unlink()
        raise ValueError("No packets matched the measured scenario connections")
    # Independently re-read the artifact and fail closed on any unrelated record.
    checked = 0
    with temporary.open("rb") as source:
        for timestamp, buf in dpkt.pcap.Reader(source):
            if tcp_endpoints(buf) not in allowed:
                raise ValueError("Unrelated traffic in scenario capture")
            checked += 1
    if checked != retained:
        raise ValueError("Scenario capture record count mismatch")
    temporary.replace(output)
    acquisition = json.loads(raw_metadata.read_text())
    manifest = {
        "schema_version": 1,
        "pcap_file": output.name,
        "sha256": sha256(output),
        "input_hashes": signatures,
        "flow_attribution": "Exact bidirectional TCP socket tuples recorded by this scenario's HTTP transport",
        "scope": "Post-acquisition filtering; handshakes, ACKs and retransmissions of scenario connections retained",
        "endpoints": [
            {"local": list(src), "remote": list(dst)} for src, dst in sorted(endpoints)
        ],
        "original_packet_records": total,
        "retained_packet_records": retained,
        "excluded_packet_records": total - retained,
        "unrelated_records_after_verification": 0,
        "capture_stats": acquisition.get("capture_stats", {}),
        "capture_stats_scope": "Original acquisition; original drops conservatively apply to this subset",
        "raw_acquisition": str(raw),
        "raw_acquisition_policy": "Local audit evidence only; report and exports use capture.scenario.pcap",
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", nargs="+")
    args = parser.parse_args()
    for directory in args.campaign:
        count = excluded = 0
        for result in sorted(Path(directory).glob("*/result.json")):
            # result.json is written before tcpdump stops. Wait for finalized metadata.
            metadata = result.parent / "capture.pcap.metadata.json"
            if not metadata.exists() or not json.loads(metadata.read_text()).get(
                "t_end"
            ):
                continue
            manifest = filter_capture(result.parent)
            count += 1
            excluded += manifest["excluded_packet_records"]
        print(
            json.dumps(
                {
                    "campaign": directory,
                    "filtered_runs": count,
                    "excluded_unrelated_records": excluded,
                }
            ),
            flush=True,
        )
