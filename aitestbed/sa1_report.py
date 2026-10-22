#!/usr/bin/env python3
"""Recompute controlled measurements and produce a report without running jobs."""

import argparse
import hashlib
import json
from pathlib import Path

from sa1_controlled import matched_metrics, summarize, write_json


def analyze(controlled, output):
    controlled, output = Path(controlled), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    analyzer_sha = hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("sa1_controlled.py").read_bytes()
    ).hexdigest()
    for path in sorted(controlled.glob("*/result.json")):
        row = json.loads(path.read_text())
        directory = path.parent
        job = json.loads((directory / "job.json").read_text())
        evidence = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ["job.json", "result.json", "client.pcap", "server.pcap"]
        }
        cache_path = output / (directory.name + ".json")
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if (
                cached.get("analysis_source_sha256") == analyzer_sha
                and cached.get("evidence_sha256") == evidence
            ):
                rows.append(cached)
                continue
        for direction, a, b, size in [
            ("ul", "client", "server", job["upload_bytes"]),
            ("dl", "server", "client", job["download_bytes"]),
        ]:
            try:
                row[direction] = matched_metrics(
                    directory / f"{a}.pcap", directory / f"{b}.pcap", direction, size
                )
            except Exception as exc:
                row[direction + "_measurement_error"] = str(exc)
        row["measurement_valid"] = all(
            row.get(x + "_capture_drops") == 0 for x in ["client", "server"]
        ) and all(
            d in row
            and row[d]["observed_max_ip_payload"] <= 1480
            and row[d]["unmatched_received_data_packets"] == 0
            for d in ["ul", "dl"]
        )
        row["source_result"] = str(path.resolve())
        row["evidence_sha256"] = evidence
        row["analysis_source_sha256"] = analyzer_sha
        write_json(output / (directory.name + ".json"), row)
        rows.append(row)
    summary = summarize(rows)
    write_json(output / "summary.json", summary)
    write_json(
        output / "provenance.json",
        {
            "source_campaign": str(controlled.resolve()),
            "runs": len(rows),
            "analysis_source_sha256": analyzer_sha,
            "interpretation": "Incomplete units remain in deadline accounting where observation exceeds the deadline. Unstarted response units and invalid captures are identified separately.",
        },
    )
    return summary


def render(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    profiles = [
        "no_emulation",
        "capacity_100",
        "capacity_400",
        "capacity_600",
        "5g_urban",
        "satellite_leo",
    ]
    workloads = [
        "visual_assistant",
        "photo_spec_payload",
        "photo_actual_compressed_raw",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, workload in zip(axes, workloads):
        rows = {x["profile"]: x for x in summary if x["workload"] == workload}
        values = []
        labels = []
        for profile in profiles:
            r = rows.get(profile)
            if not r:
                continue
            d = r["directions"]["ul"]
            m = d["burst_transfer_ms"]
            labels.append(profile.replace("_", "\n"))
            values.append(m.get("median", float("nan")))
        ax.bar(range(len(values)), values, color="#3478a6")
        ax.axhline(
            400 if workload == "visual_assistant" else 1500,
            color="#b53b40",
            linestyle="--",
            label="Draft UL deadline",
        )
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(workload.replace("_", " "))
        ax.set_yscale("log")
        ax.set_ylabel("Completed UL transfer median (ms, log scale)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Controlled TCP measurements — incomplete transfers excluded from medians, retained in failure counts"
    )
    fig.savefig(output / "controlled-transfers.png", dpi=170)
    fig.savefig(output / "controlled-transfers.svg")
    plt.close(fig)
    lines = [
        "# Controlled SA1 network evaluation",
        "",
        "These are isolated Linux namespace TCP transfers with router shaping and captures at both endpoints. They measure software-network behavior, not radio performance or model inference. Gaming was not evaluated.",
        "",
        "All packet sizes below are IP payload, including TCP headers. ACK-inclusive and data-only sizes are separate. One-way timestamps share a kernel clock, with microsecond PCAP resolution; timestamp-placement uncertainty is not calibrated.",
        "",
        "| Workload | Profile | Successful transactions / runs | UL median, ms | UL deadline misses / assessed | Incomplete UL units |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in summary:
        d = r["directions"]["ul"]
        median = d["burst_transfer_ms"].get("median")
        med = f"{median:.3f}" if median is not None else "not completed"
        lines.append(
            f"| {r['workload']} | {r['profile']} | {r['successes']}/{r['runs']} | {med} | {d['deadline_misses']}/{d['deadline_assessable_units']} | {d['incomplete_units']} |"
        )
    lines += [
        "",
        "![Controlled transfer medians](controlled-transfers.png)",
        "",
        "Ten repetitions per condition give preliminary distributions, not a reliable estimate of extreme tails. Observed maxima are not guaranteed bounds. The JSON includes confidence intervals, capture validity, delivered packet delays, lost transmission attempts, duplicates, actual payload throughput and both photo downlink thresholds.",
        "",
        "An incomplete response that never began transmitting is not assigned a fabricated downlink transfer latency. Timeouts and failed transactions remain visible in the transaction counts.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = analyze(args.controlled, args.output)
    render(summary, args.output)
    print(
        json.dumps(
            {
                "conditions": len(summary),
                "runs": sum(r["runs"] for r in summary),
                "output": args.output,
            }
        )
    )
