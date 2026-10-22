#!/usr/bin/env python3
"""Export a live or completed isolated provider matrix to Excel, without API calls."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from sa1_excel import Book, flat_stats
from sa1_excel_metrics import build_provider, digest, distribution, GAPS_MS, WINDOWS_MS
from sa1_segment_metrics import build_segment


def read(path):
    return json.loads(Path(path).read_text())


def flatten(row):
    result = {}
    for key, value in row.items():
        if isinstance(value, dict) and "n" in value:
            result.update(flat_stats(key, value))
        else:
            result[key] = value
    return result


def collect(root):
    plan = read(root / "plan.json")
    tables = defaultdict(list)
    for order, job in enumerate(plan["jobs"]):
        run = root / job["id"]
        base = {
            "scenario": job["scenario"],
            "profile": job["profile"],
            "run_index": job["run_index"],
            "run_id": job["id"],
        }
        state = "pending"
        if (run / "result.json").exists():
            raw = read(run / "result.json")
            finalized = (
                (run / "network.json").exists()
                and "cleanup_errors" in read(run / "network.json")
                and all(
                    any(
                        (point / name).exists()
                        for name in (
                            "capture-analysis.json",
                            "capture-analysis-error.json",
                        )
                    )
                    for point in (run, run / "edge")
                )
            )
            state = (
                ("success" if raw["success"] else "failed")
                if finalized
                else "finalizing_capture"
            )
        elif (run / "started.json").exists():
            state = "started_without_terminal_result"
        elif (run / "blocked.json").exists():
            blocked = read(run / "blocked.json")
            state = (
                "deferred_for_balanced_coverage"
                if blocked.get("state") == "deferred_for_balanced_coverage"
                else "blocked_not_attempted"
            )
        row = {
            **base,
            "planned_order": order,
            "state": state,
            "model": plan["scenarios"][job["scenario"]]["model"],
            "client_completion_sec": None,
            "request_body_bytes": None,
            "response_body_bytes": None,
            "output_media_bytes": None,
            "error": None,
            "source_result": None,
        }
        tables["Run_Ledger"].append(row)
        if state not in ("success", "failed"):
            continue
        meta = raw["metadata"]
        row.update(
            client_completion_sec=raw["total_latency_sec"],
            request_body_bytes=raw["total_request_bytes"],
            response_body_bytes=raw["total_response_bytes"],
            output_media_bytes=meta.get("output_media_bytes"),
            error=raw.get("error_message"),
            source_result=str(run / "result.json"),
        )
        for phase in meta["phases"]:
            tables["HTTP_Phases"].append({**base, **phase})
        for kind in ("inputs", "outputs"):
            for artifact in meta.get(kind, []):
                tables["Media_Artifacts"].append({**base, "kind": kind, **artifact})
        network_file = run / "network.json"
        # Result precedes capture stop and cleanup. Analyze only finalized pairs.
        if not network_file.exists() or "cleanup_errors" not in read(network_file):
            continue
        network = read(network_file)
        tables["Network_Audit"].append(
            {
                **base,
                "physical_qdisc_unchanged": network["physical_qdisc_unchanged"],
                "cleanup_errors": network["cleanup_errors"],
                "offloads_disabled": all(
                    "generic-receive-offload: off" in v
                    for v in network["offloads"].values()
                ),
                "network_evidence": str(network_file),
                "sha256": digest(network_file),
                "uplink_qdisc_after": network["qdisc_after"]["wan"],
                "downlink_qdisc_after": network["qdisc_after"]["lan"],
            }
        )
        for point, directory in (("client", run), ("edge", run / "edge")):
            if not (directory / "capture.scenario.pcap.metadata.json").exists():
                tables["Analysis_Errors"].append(
                    {**base, "point": point, "reason": "Verified capture unavailable"}
                )
                continue
            evidence = read(directory / "capture.scenario.pcap.metadata.json")
            tables["Capture_Audit"].append(
                {
                    **base,
                    "point": point,
                    "capture_path": str(directory / evidence["pcap_file"]),
                    "sha256": evidence["sha256"],
                    "retained_packets": evidence["retained_packet_records"],
                    "excluded_records": evidence["excluded_packet_records"],
                    "unrelated_records_in_export": evidence[
                        "unrelated_records_after_verification"
                    ],
                    **evidence["capture_stats"],
                }
            )
            try:
                metrics = build_provider(
                    directory,
                    root.name,
                    plan["scenarios"][job["scenario"]],
                    root / "analysis-cache" / point / (job["id"] + ".json.gz"),
                )
                for view in metrics["views"]:
                    label = {
                        **base,
                        "point": point,
                        "direction": view["direction"],
                        "measurement_valid": metrics["measurement_valid"],
                    }
                    for field, sheet in (
                        ("packet", "Packet_Runs"),
                        ("burst", "Burst_Runs"),
                        ("window", "Rate_Windows"),
                    ):
                        for item in view[field]:
                            record = {**label, **flatten(item)}
                            if field == "burst":
                                details = [
                                    b
                                    for b in view["burst_details"]
                                    if b["population"] == item["population"]
                                    and b["idle_gap_ms"] == item["idle_gap_ms"]
                                ]
                                for width in (1, 5, 10, 50, 100):
                                    key = f"peak_{width}ms_mbps"
                                    record.update(
                                        flat_stats(
                                            key, distribution([b[key] for b in details])
                                        )
                                    )
                            tables[sheet].append(record)
            except Exception as exc:
                tables["Analysis_Errors"].append(
                    {
                        **base,
                        "point": point,
                        "reason": type(exc).__name__ + ": " + str(exc),
                    }
                )
        try:
            segment = build_segment(
                run, root / "analysis-cache/segment" / (job["id"] + ".json.gz")
            )
            for direction, values in segment["directions"].items():
                label = {
                    **base,
                    "direction": direction,
                    "measurement_valid": segment["measurement_valid"],
                }
                tables["Segment_Delay_Runs"].append(
                    {
                        **label,
                        **flatten(
                            {
                                k: v
                                for k, v in values.items()
                                if k not in ("source_bursts", "packet_thresholds")
                            }
                        ),
                    }
                )
                tables["Source_Burst_Runs"].extend(
                    {**label, **flatten(b)} for b in values["source_bursts"]
                )
                tables["Packet_Thresholds"].extend(
                    {**label, **t} for t in values["packet_thresholds"]
                )
        except Exception as exc:
            tables["Analysis_Errors"].append(
                {
                    **base,
                    "point": "segment",
                    "reason": type(exc).__name__ + ": " + str(exc),
                }
            )
    for scenario in plan["scenarios"]:
        for profile in plan["profiles"]:
            rows = [
                r
                for r in tables["Run_Ledger"]
                if (r["scenario"], r["profile"]) == (scenario, profile)
            ]
            successes = [r for r in rows if r["state"] == "success"]
            terminal = [r for r in rows if r["state"] in ("success", "failed")]
            tables["Matrix_Summary"].append(
                {
                    "scenario": scenario,
                    "profile": profile,
                    "scenario_profile": scenario.removeprefix("sa1_") + " / " + profile,
                    "planned": plan["repetitions"],
                    "completed": len(terminal),
                    "successes": len(successes),
                    "failures": len(terminal) - len(successes),
                    "pending_or_unresolved": len(rows) - len(terminal),
                    **flat_stats(
                        "successful_completion_sec",
                        distribution([r["client_completion_sec"] for r in successes]),
                    ),
                    **flat_stats(
                        "request_body_bytes",
                        distribution([r["request_body_bytes"] for r in terminal]),
                    ),
                    **flat_stats(
                        "successful_output_bytes",
                        distribution([r["output_media_bytes"] for r in successes]),
                    ),
                }
            )
    return plan, tables


def grouped_summary(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        if row.get("measurement_valid", True):
            groups[tuple(row[k] for k in keys)].append(row)
    for key, selected in groups.items():
        out = {**dict(zip(keys, key)), "valid_runs": len(selected)}
        for name, count, mean, maximum in fields:
            n = sum(r.get(count, 0) or 0 for r in selected)
            out[name + "_count"] = n
            out[name + "_weighted_mean"] = (
                sum((r.get(mean) or 0) * (r.get(count) or 0) for r in selected) / n
                if n
                else None
            )
            maxima = [r[maximum] for r in selected if r.get(maximum) is not None]
            out[name + "_observed_max"] = max(maxima) if maxima else None
            out.update(
                flat_stats(
                    name + "_run_means",
                    distribution(
                        [r[mean] for r in selected if r.get(mean) is not None]
                    ),
                )
            )
        yield out


def export(root, output):
    root, output = Path(root), Path(output)
    plan, tables = collect(root)
    keys = ["scenario", "profile", "point", "direction", "population"]
    tables["Packet_Summary"] = list(
        grouped_summary(
            tables["Packet_Runs"],
            keys,
            [("size_bytes", "size_bytes_n", "size_bytes_mean", "size_bytes_max")],
        )
    )
    tables["Burst_Summary"] = list(
        grouped_summary(
            tables["Burst_Runs"],
            [*keys, "idle_gap_ms"],
            [
                ("size_bytes", "size_bytes_n", "size_bytes_mean", "size_bytes_max"),
                ("duration_ms", "duration_ms_n", "duration_ms_mean", "duration_ms_max"),
                ("idle_ms", "idle_ms_n", "idle_ms_mean", "idle_ms_max"),
            ],
        )
    )
    tables["Segment_Delay_Summary"] = list(
        grouped_summary(
            tables["Segment_Delay_Runs"],
            ["scenario", "profile", "direction"],
            [
                (
                    "data_packet_ms",
                    "matched_data_packet_delay_ms_n",
                    "matched_data_packet_delay_ms_mean",
                    "matched_data_packet_delay_ms_max",
                ),
                (
                    "recovered_piece_ms",
                    "first_byte_piece_delivery_ms_n",
                    "first_byte_piece_delivery_ms_mean",
                    "first_byte_piece_delivery_ms_max",
                ),
            ],
        )
    )
    tables["Source_Burst_Summary"] = list(
        grouped_summary(
            tables["Source_Burst_Runs"],
            ["scenario", "profile", "direction", "idle_gap_ms"],
            [
                (
                    "unique_tcp_bytes",
                    "unique_tcp_payload_bytes_n",
                    "unique_tcp_payload_bytes_mean",
                    "unique_tcp_payload_bytes_max",
                ),
                (
                    "segment_delivery_ms",
                    "segment_delivery_ms_n",
                    "segment_delivery_ms_mean",
                    "segment_delivery_ms_max",
                ),
            ],
        )
    )
    profiles = []
    from netemu.profile import NetworkProfile

    for name, value in plan["profiles"].items():
        profile = NetworkProfile.from_dict(name, value)
        for direction, p in (("ul", profile.uplink or profile), ("dl", profile)):
            profiles.append({"profile": name, "direction": direction, **p.settings()})
    tables["Profiles"] = profiles
    tables["Scenario_Config"] = [
        {"scenario": name, **value} for name, value in plan["scenarios"].items()
    ]
    tables["Inputs"] = [
        {"path": path, "sha256": value, "bytes": Path(path).stat().st_size}
        for path, value in plan["input_hashes"].items()
    ]
    tables["SA1_Questions"] = [
        {
            "question": "Average uplink/downlink burst size",
            "answer": "Packet and Burst summaries/runs provide both directions, two capture points, all-TCP and data-only populations at eight gap criteria. Source-burst tables additionally count unique encrypted TCP bytes.",
            "limitation": "A traffic cluster is not necessarily one application data unit.",
        },
        {
            "question": "Average uplink/downlink packet size",
            "answer": "Packet_Summary provides packet-weighted means; Packet_Runs retains per-run distributions. IP payload = TCP header plus TCP payload.",
            "limitation": "Only verified captures with disabled offloads and zero capture drops enter summaries.",
        },
        {
            "question": "Maximum uplink/downlink packet latency",
            "answer": "Segment_Delay sheets provide observed maxima for matched packet attempts across the emulator. Retransmission recovery is shown separately.",
            "limitation": "Public-provider destination timestamps are unavailable; full-path one-way delay and a guaranteed maximum are not measured.",
        },
        {
            "question": "Maximum uplink/downlink burst latency",
            "answer": "Source_Burst sheets retain grouping at first source transmission and measure delivery across the emulator, including retransmission recovery.",
            "limitation": "Encrypted TCP clusters are not full application data units; incomplete clusters do not become zero-latency successes.",
        },
        {
            "question": "Different burst duration criteria",
            "answer": f"Strict inter-packet idle gaps {list(GAPS_MS)} ms; throughput windows {list(WINDOWS_MS)} ms. Burst duration is last minus first packet time; one-packet bursts have zero duration and no bytes/duration rate.",
            "limitation": "10/100 ms are source-report criteria; other gaps are sensitivity analysis.",
        },
        {
            "question": "Gaming",
            "answer": "Not performed; excluded from this campaign.",
            "limitation": "No gaming results are included.",
        },
        {
            "question": "Video generation versus visual assistant",
            "answer": "This matrix uses Google text-to-video generation, as requested. Visual-assistant baseline results remain in the earlier workbook.",
            "limitation": "The SA1 visual-assistant upload targets are not assigned to text-to-video generation.",
        },
    ]
    tables["Draft_Requirements"] = [
        {
            "scenario": "chatbot",
            "ul_unit_MB": "0.0005–0.006",
            "dl_unit_MB": "0.0005",
            "ul_unit_deadline_ms": "20",
            "dl_unit_deadline_ms": "20",
            "packet_delay_ms": "30",
            "packet_size": "<800 bytes",
        },
        {
            "scenario": "image_genai",
            "ul_unit_MB": "0.0005–0.4",
            "dl_unit_MB": "unspecified",
            "ul_unit_deadline_ms": "50",
            "dl_unit_deadline_ms": "unspecified",
            "packet_delay_ms": "20",
            "packet_size": ">800 bytes",
        },
        {
            "scenario": "photo_enhancement",
            "ul_unit_MB": "20",
            "dl_unit_MB": "20",
            "ul_unit_deadline_ms": "1500",
            "dl_unit_deadline_ms": "1000–1500",
            "packet_delay_ms": "N/A",
            "packet_size": ">800 bytes",
        },
        {
            "scenario": "additional_video_generation",
            "ul_unit_MB": "unspecified",
            "dl_unit_MB": "unspecified",
            "ul_unit_deadline_ms": "unspecified",
            "dl_unit_deadline_ms": "unspecified",
            "packet_delay_ms": "unspecified",
            "packet_size": "unspecified",
        },
    ]
    tables["Methodology"] = [
        {"topic": topic, "definition": definition}
        for topic, definition in [
            (
                "Design",
                f"{len(plan['jobs'])} planned attempts: {len(plan['scenarios'])} scenarios × {len(plan['profiles'])} profiles × {plan['repetitions']} repetitions. Original randomized order, seed {plan['seed']}; sequential execution, fresh TCP namespaces per job. Any coverage-priority scheduling is documented in the campaign amendment.",
            ),
            (
                "Failure accounting",
                "No automatic retries. Failed attempts count as runs, never as successes. Permanent account/API errors block remaining same-scenario submissions. Started jobs without results are unresolved and never automatically resubmitted.",
            ),
            (
                "Isolation",
                "All shaping is on dedicated router namespace veths; captures only on dedicated client/edge interfaces. Exact bidirectional HTTP socket tuples are independently verified. Other applications' packets are excluded; the physical Internet link remains shared with the host.",
            ),
            (
                "Direction",
                "UL: client to provider; DL: provider to client. Client capture precedes UL impairment and follows DL impairment. Edge capture follows UL impairment and precedes DL impairment.",
            ),
            (
                "Capture validity",
                "All private interfaces MTU 1500, TSO/GSO/GRO/LRO disabled. Summary eligibility requires zero kernel capture drops and no oversized IP payload records. Analysis errors are explicit.",
            ),
            (
                "Latency",
                "API completion includes provider processing and Internet transport; not network one-way latency. Two local captures share the kernel clock with microsecond resolution; timestamp-placement uncertainty is not calibrated.",
            ),
            (
                "Source clusters",
                "Unique TCP payload intervals are assigned to their first source transmission, then clustered by strict idle gap. First destination byte delivery handles retransmission and resegmentation. These encrypted transport clusters do not identify application data units.",
            ),
            (
                "Packet attempts",
                "Exact flow/sequence/ACK/flags/IP-ID/payload fingerprints pair attempts. Lost attempts have no delivered delay. Duplicate/unmatched destination records and ambiguous repeated identities are counted.",
            ),
            (
                "Aggregation",
                "Weighted means use individual packet/burst counts. Summary percentiles explicitly describe per-run means, not pooled packet percentiles. Observed maximum is the largest measured value, not a guaranteed bound. Raw run distributions retain p50/p95/p99.",
            ),
            (
                "Windows",
                "Fixed non-overlapping bins share the run connection origin; final bin padded to full width. Zero bins included in all-window means; active-window means reported separately. Burst peak rates use actual fixed bins, not bytes divided by burst duration.",
            ),
            (
                "Photo input",
                "Six HDR+ public raw-dataset images rendered as camera-oriented JPEGs from 4K raw crops; one scene. This cloud JPEG proxy is not direct raw enhancement, Jetraw compression, or a model-quality assessment.",
            ),
            (
                "Media validation",
                "Returned images decoded and videos fully frame-decoded. Artifact hashes and dimensions retained. Semantic/visual quality is not evaluated.",
            ),
            (
                "Requirements",
                "TS 22.270 v0.3.0 Table 6.1.1-1 contains bracketed draft values, decimal MB. Local TCP-cluster metrics do not establish compliance with full-path application data-unit requirements.",
            ),
            (
                "Earlier evidence",
                "Earlier baseline/provider and controlled fixture campaigns are separate cohorts and do not count toward this 400-attempt matrix.",
            ),
        ]
    ]
    tables["Evidence"] = [
        {
            "kind": "plan",
            "path": str(root / "plan.json"),
            "sha256": digest(root / "plan.json"),
        },
        {
            "kind": "report source",
            "path": str(Path(__file__).resolve()),
            "sha256": digest(__file__),
        },
        {
            "kind": "segment analysis source",
            "path": str(Path(__file__).with_name("sa1_segment_metrics.py")),
            "sha256": digest(Path(__file__).with_name("sa1_segment_metrics.py")),
        },
    ]
    temporary = output.with_name(output.stem + ".partial.xlsx")
    book = Book(temporary, compact=True)
    descriptions = {
        "Matrix_Summary": "Planned attempts for each scenario/profile pair. Completion statistics use successful jobs; failures and pending jobs remain visible.",
        "Run_Ledger": "Every planned attempt, including pending, failed, blocked and unresolved jobs. API duration is client completion time, not one-way network delay.",
        "Packet_Summary": "Eligible captures only. Packet-weighted mean and observed maximum; percentiles describe per-run means.",
        "Burst_Summary": "Eligible captures, strict idle-gap grouping; both all-TCP and data-only populations. Statistics summarize observed traffic clusters.",
        "Segment_Delay_Summary": "Local emulated segment only. Successfully matched packet attempts and first-delivered byte pieces have distinct delay semantics.",
        "Source_Burst_Summary": "Unique encrypted TCP bytes grouped at first transmission; local segment delivery includes retransmission recovery. Not application-unit compliance.",
        "Draft_Requirements": "Bracketed draft values from TS 22.270 v0.3.0 Table 6.1.1-1, S1-263335. Decimal MB. Text-to-video has no separate numeric row.",
        "Rate_Windows": "Mbps over fixed bins; includes active/all-window means, p95/p99, peak, CV and peak/mean. Final bin padded to full width.",
        "Packet_Thresholds": "Counts of matched data packets exceeding reference thresholds across the local emulator only. Passing this segment does not imply full-path compliance.",
    }
    order = [
        "Matrix_Summary",
        "Run_Ledger",
        "SA1_Questions",
        "Draft_Requirements",
        "Packet_Summary",
        "Burst_Summary",
        "Segment_Delay_Summary",
        "Source_Burst_Summary",
        "Packet_Runs",
        "Burst_Runs",
        "Rate_Windows",
        "Segment_Delay_Runs",
        "Source_Burst_Runs",
        "Packet_Thresholds",
        "HTTP_Phases",
        "Media_Artifacts",
        "Capture_Audit",
        "Network_Audit",
        "Analysis_Errors",
        "Profiles",
        "Scenario_Config",
        "Inputs",
        "Methodology",
        "Evidence",
    ]
    for name in order:
        # Include keys from every row, e.g. image/video artifact metadata differ.
        columns = list(dict.fromkeys(k for r in tables[name] for k in r))
        book.sheet(
            name,
            tables[name],
            descriptions.get(name, "See Methodology for units, scope and limitations."),
            columns=columns or ["status"],
        )
    ws = book.overview
    complete = sum(r["completed"] for r in tables["Matrix_Summary"])
    success = sum(r["successes"] for r in tables["Matrix_Summary"])
    ws.merge_range(
        "A1:H2", f"SA1 — {len(plan['jobs'])}-run network-profile matrix", book.title
    )
    ws.merge_range(
        "A4:H5",
        "COMPLETE"
        if complete == len(plan["jobs"])
        else "IN PROGRESS — pending jobs are not results",
        book.note,
    )
    for i, (label, value) in enumerate(
        [
            ("Planned attempts", len(plan["jobs"])),
            ("Terminal results", complete),
            ("Successful", success),
            ("Failed", complete - success),
            ("Remaining", len(plan["jobs"]) - complete),
            ("Scenario/profile pairs", len(plan["scenarios"]) * len(plan["profiles"])),
            ("Analysis errors", len(tables["Analysis_Errors"])),
            ("Exported (UTC)", datetime.now(timezone.utc).isoformat()),
        ],
        6,
    ):
        ws.write(i, 0, label)
        ws.write(i, 1, value)
    ws.merge_range(
        "A16:H18",
        "Chatbot · image generation · Google video generation · six-photo enhancement. Gaming excluded. Sequential private-network runs; local emulator delay is separate from full provider completion time.",
        book.note,
    )
    chart = book.wb.add_chart({"type": "bar", "subtype": "stacked"})
    matrix_ws, cols, count = book.sheets["Matrix_Summary"]
    for key, color in (
        ("successes", "#168AAD"),
        ("failures", "#D55E00"),
        ("pending_or_unresolved", "#CCD5DF"),
    ):
        chart.add_series(
            {
                "name": key,
                "categories": [
                    "Matrix_Summary",
                    5,
                    cols.index("scenario_profile"),
                    5 + count - 1,
                    cols.index("scenario_profile"),
                ],
                "values": [
                    "Matrix_Summary",
                    5,
                    cols.index(key),
                    5 + count - 1,
                    cols.index(key),
                ],
                "fill": {"color": color},
                "border": {"none": True},
                "categories_data": [
                    r["scenario_profile"] for r in tables["Matrix_Summary"]
                ],
                "values_data": [r[key] for r in tables["Matrix_Summary"]],
            }
        )
    chart.set_title(
        {"name": f"{plan['repetitions']} attempts per scenario/profile pair"}
    )
    chart.set_x_axis({"name": "Attempts", "min": 0, "max": plan["repetitions"]})
    chart.set_y_axis({"reverse": True})
    chart.set_size({"width": 1050, "height": 1000})
    ws.insert_chart("A21", chart)
    book.wb.close()
    temporary.replace(output)
    manifest = {
        "workbook": str(output),
        "sha256": digest(output),
        "planned": len(plan["jobs"]),
        "completed": complete,
        "successes": success,
        "failures": complete - success,
        "analysis_errors": len(tables["Analysis_Errors"]),
        "sheets": book.index,
        "plan_sha256": digest(root / "plan.json"),
        "exported_at": time.time(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--output")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    root = Path(args.campaign).resolve()
    target = Path(args.output) if args.output else root / "SA1-matrix-results.xlsx"
    previous = -1
    while True:
        # Complete capture finalization and cleanup before publishing that run.
        summary = root / "summary.json"
        count = read(summary)["completed"] if summary.exists() else 0
        if count != previous:
            manifest = export(root, target)
            print(
                json.dumps({k: v for k, v in manifest.items() if k != "sheets"}),
                flush=True,
            )
            previous = count
            if manifest["completed"] == manifest["planned"]:
                break
        if not args.watch:
            break
        time.sleep(60)
