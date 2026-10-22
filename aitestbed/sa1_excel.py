#!/usr/bin/env python3
"""Build an auditable SA1 Excel workbook from completed campaigns (no API calls)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

import numpy as np
import xlsxwriter
from sa1_excel_metrics import distribution, digest
from sa1_capture_filter import filter_capture

STATS = ("n", "mean", "min", "p50", "p95", "p99", "max", "std", "cv")
LABELS = {
    "sa1_chat_small": "Chatbot: 500-byte prompt, nonstreaming",
    "sa1_chat_streaming": "Chatbot: 6000-byte prompt, streaming",
    "sa1_image_text": "Image GenAI: text prompt",
    "sa1_image_conditioned": "Image GenAI: image + text",
    "sa1_photo_enhancement": "Photo enhancement: six HDR+ JPEG renders",
    "sa1_visual_assistant_openai_frames": "Visual assistant: sampled-frame OpenAI proxy",
    "sa1_visual_assistant": "Visual assistant: native video upload",
    "sa1_video_generation": "Additional text-to-video generation",
}


def read(path):
    return json.loads(Path(path).read_text())


def flat_stats(name, stats):
    return {name + "_" + k: stats.get(k) for k in STATS}


def delta(phase, end, start, scale=1000):
    a, b = phase.get(end), phase.get(start)
    return (a - b) * scale if a is not None and b is not None else None


def wilson(k, n):
    if not n:
        return (None, None)
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


class Book:
    def __init__(self, path, compact=False):
        self.path = Path(path)
        self.summary = (
            None
            if compact
            else Book(
                self.path.with_name(self.path.stem + "-summary.xlsx"), compact=True
            )
        )
        self.wb = xlsxwriter.Workbook(
            str(path),
            {
                "constant_memory": True,
                "strings_to_urls": False,
                "strings_to_formulas": False,
            },
        )
        self.wb.set_properties(
            {
                "title": "SA1 traffic and latency evaluation",
                "subject": "Ten-run campaigns, burst criteria and source evidence",
                "author": "6G-Testbed evaluation",
                "comments": "See Methodology and Evidence sheets.",
            }
        )
        self.title = self.wb.add_format(
            {"bold": True, "font_size": 18, "font_color": "#16324F"}
        )
        self.note = self.wb.add_format(
            {"text_wrap": True, "font_color": "#45556C", "valign": "top"}
        )
        self.header = self.wb.add_format(
            {
                "bold": True,
                "font_color": "white",
                "bg_color": "#16324F",
                "text_wrap": True,
                "valign": "vcenter",
            }
        )
        self.number = self.wb.add_format({"num_format": "#,##0.000"})
        self.integer = self.wb.add_format({"num_format": "#,##0"})
        self.wrap = self.wb.add_format({"text_wrap": True, "valign": "top"})
        self.link = self.wb.add_format({"font_color": "#1565C0", "underline": True})
        self.bad = self.wb.add_format({"bg_color": "#FCE4D6", "font_color": "#9C0006"})
        self.good = self.wb.add_format({"bg_color": "#E2F0D9", "font_color": "#375623"})
        self.sheets = {}
        self.index = []
        self.overview = self.wb.add_worksheet("Overview")
        self.overview.hide_gridlines(2)
        self.overview.set_column(0, 0, 43)
        self.overview.set_column(1, 7, 18)

    def sheet(self, name, rows, description, columns=None):
        if self.summary is not None and name not in (
            "Burst_Details",
            "Source_Burst_Latency",
            "Burst_Runs",
        ):
            rows = list(rows)
            self.summary.sheet(name, rows, description, columns)
        iterator = iter(rows)
        first = next(iterator, None)
        if columns is None:
            columns = list(first) if first is not None else ["status"]
        ws = self.wb.add_worksheet(name)
        ws.hide_gridlines(2)
        ws.set_tab_color("#168AAD" if "Summary" in name else "#718096")
        ws.write(0, 0, name.replace("_", " "), self.title)
        ws.merge_range(
            1, 0, 2, min(max(len(columns) - 1, 1), 10), description, self.note
        )
        ws.set_row(1, 32)
        ws.set_row(2, 25)
        ws.write_row(4, 0, columns, self.header)
        ws.set_row(4, 46)
        ws.freeze_panes(5, min(2, len(columns)))
        for i, col in enumerate(columns):
            width = 22
            if col in ("scenario", "workload", "run_id", "campaign"):
                width = 33
            if any(
                k in col
                for k in (
                    "path",
                    "source_result",
                    "sha256",
                    "reason",
                    "limitations",
                    "explanation",
                    "answer",
                    "definition",
                    "scope",
                    "question",
                    "notes",
                    "url",
                )
            ):
                width = 50
            if col in ("direction", "n", "run_index", "population", "point", "state"):
                width = 17
            ws.set_column(i, i, width)
        n = 0
        for row in [first] if first is not None else []:
            self._row(ws, 5 + n, columns, row)
            n += 1
        for row in iterator:
            if n >= 1048570:
                raise ValueError(
                    f"Excel row limit exceeded in {name}; split this sheet"
                )
            self._row(ws, 5 + n, columns, row)
            n += 1
        if n:
            ws.autofilter(4, 0, 4 + n, len(columns) - 1)
        ws.set_landscape()
        ws.fit_to_pages(1, 0)
        ws.repeat_rows(0, 4)
        self.sheets[name] = (ws, columns, n)
        self.index.append({"sheet": name, "rows": n, "description": description})
        if self.summary is not None:
            print(f"{name}: {n:,} rows", flush=True)
        return ws

    def _row(self, ws, r, columns, row):
        for c, key in enumerate(columns):
            v = row.get(key)
            if v is None:
                continue
            if isinstance(v, (dict, list, tuple)):
                v = json.dumps(v, ensure_ascii=False)
            if isinstance(v, (bool, np.bool_)):
                ws.write_boolean(r, c, bool(v))
            elif isinstance(v, (int, np.integer)):
                ws.write_number(r, c, int(v), self.integer)
            elif isinstance(v, (float, np.floating)):
                if not math.isfinite(float(v)):
                    raise ValueError(f"Non-finite value {key}")
                ws.write_number(r, c, float(v), self.number)
            else:
                ws.write_string(r, c, str(v))

    def finish(self, highlights, counts):
        ws = self.overview
        ws.merge_range("A1:H2", "SA1 evaluation — results and evidence", self.title)
        ws.merge_range(
            "A3:H4",
            "Fresh provider repetitions + existing controlled 10-run matrix. Gaming excluded. Measurements, failures and unavailable quantities are kept separate.",
            self.note,
        )
        r = 5
        for label, value in counts:
            ws.write(r, 0, label, self.header)
            ws.merge_range(r, 1, r, 7, value, self.wrap)
            r += 1
        r += 1
        for item in highlights:
            ws.merge_range(r, 0, r + 1, 7, item, self.wrap)
            ws.set_row(r, 23)
            r += 2
        r += 1
        ws.write_row(r, 0, ["Sheet", "Rows", "Purpose"], self.header)
        r += 1
        for item in self.index:
            ws.write_url(
                r, 0, "internal:'" + item["sheet"] + "'!A1", self.link, item["sheet"]
            )
            ws.write(r, 1, item["rows"], self.integer)
            ws.merge_range(r, 2, r, 7, item["description"], self.wrap)
            ws.set_row(r, 33)
            r += 1
        ws.freeze_panes(5, 0)
        self.wb.close()
        if self.summary is not None:
            self.summary.finish(
                [
                    f"Full per-run and per-burst detail is in {self.path.name}, alongside this summary workbook. References to Burst_Runs, Burst_Details and Source_Burst_Latency refer to that complete workbook."
                ]
                + highlights,
                counts,
            )


def provider_rows(root, previous):
    rows = []
    phases = []
    artifacts = []
    audit = []
    summaries = []
    cov = []
    selected = []
    campaign_names = ["provider-openai", "provider-google"]
    if (root / "provider-google-funded/plan.json").exists():
        campaign_names.append("provider-google-funded")
    for name in campaign_names:
        directory = root / name
        plan = read(directory / "plan.json")
        for scenario, config in plan["scenarios"].items():
            states = Counter()
            for job in sorted(
                (j for j in plan["jobs"] if j["scenario"] == scenario),
                key=lambda x: x["run_index"],
            ):
                run = directory / job["id"]
                p = run / "result.json"
                blocked = run / "blocked.json"
                row = {
                    "campaign": name,
                    "scenario": scenario,
                    "description": LABELS.get(scenario, scenario),
                    "run_index": job["run_index"],
                    "run_id": job["id"],
                    "model": config.get("model"),
                    "network_profile": "no_emulation",
                    "state": None,
                    "success": None,
                    "http_status": None,
                    "client_completion_sec": None,
                    "http_request_body_bytes": None,
                    "http_response_body_bytes": None,
                    "input_media_bytes": None,
                    "output_media_bytes": None,
                    "sampled_frame_count": None,
                    "uploaded_sampled_JPEG_bytes": None,
                    "native_video_uploaded": None,
                    "audio_uploaded": None,
                    "other_campaign_overlap_count": None,
                    "capture_complete": None,
                    "wire_packet_metrics_valid": None,
                    "error_or_block_reason": None,
                    "result_path": None,
                    "result_sha256": None,
                    "scenario_capture_path": None,
                    "scenario_capture_sha256": None,
                }
                if p.exists():
                    raw = read(p)
                    meta = raw.get("metadata", {})
                    selected.append((name, run, raw, config))
                    row.update(
                        state="success" if raw["success"] else "failed",
                        success=raw["success"],
                        client_completion_sec=raw["total_latency_sec"],
                        http_request_body_bytes=raw["total_request_bytes"],
                        http_response_body_bytes=raw["total_response_bytes"],
                        input_media_bytes=meta.get("input_media_bytes"),
                        output_media_bytes=meta.get("output_media_bytes"),
                        sampled_frame_count=len(meta["sampled_frames"])
                        if "sampled_frames" in meta
                        else None,
                        uploaded_sampled_JPEG_bytes=meta.get("uploaded_image_bytes"),
                        native_video_uploaded=meta.get("native_video_uploaded"),
                        audio_uploaded=meta.get("audio_uploaded"),
                        error_or_block_reason=raw.get("error_message"),
                        result_path=str(p),
                        result_sha256=digest(p),
                    )
                    statuses = [x.get("http_status") for x in meta.get("phases", [])]
                    row["http_status"] = ",".join(map(str, statuses))
                    fm = filter_capture(run)
                    ca = read(run / "capture-analysis.json")
                    row.update(
                        capture_complete=ca.get("capture_complete"),
                        wire_packet_metrics_valid=ca.get("wire_packet_metrics_valid"),
                        scenario_capture_path=str(run / "capture.scenario.pcap"),
                        scenario_capture_sha256=fm["sha256"],
                    )
                    audit.append(
                        {
                            "campaign": name,
                            "run_id": run.name,
                            "scenario": scenario,
                            "original_records": fm["original_packet_records"],
                            "scenario_records": fm["retained_packet_records"],
                            "excluded_unrelated_records": fm["excluded_packet_records"],
                            "unrelated_records_in_export": fm[
                                "unrelated_records_after_verification"
                            ],
                            "matched_connections": len(fm["endpoints"]),
                            "kernel_capture_drops": fm["capture_stats"].get(
                                "packets_dropped_by_kernel"
                            ),
                            "capture_complete": row["capture_complete"],
                            "wire_packet_metrics_valid": row[
                                "wire_packet_metrics_valid"
                            ],
                            "scenario_capture_path": row["scenario_capture_path"],
                            "scenario_capture_sha256": fm["sha256"],
                            "filter_manifest_path": str(
                                run / "capture.scenario.pcap.metadata.json"
                            ),
                        }
                    )
                elif blocked.exists():
                    b = read(blocked)
                    row.update(
                        state="blocked_not_attempted",
                        http_status=b["http_status"],
                        error_or_block_reason=b["reason"]
                        + "; caused by "
                        + b["caused_by_job"],
                    )
                else:
                    row["state"] = (
                        "unresolved"
                        if (run / "started.json").exists()
                        else "not_started"
                    )
                states[row["state"]] += 1
                rows.append(row)
            cov.append(
                {
                    "cohort": name,
                    "scenario": scenario,
                    "description": LABELS[scenario],
                    "planned": sum(states.values()),
                    "attempted": states["success"] + states["failed"],
                    "succeeded": states["success"],
                    "failed": states["failed"],
                    "blocked_not_attempted": states["blocked_not_attempted"],
                    "unresolved": states["unresolved"],
                    "not_started": states["not_started"],
                    "model": config.get("model"),
                    "scope": config.get("workload_kind", config["type"]),
                }
            )
    prior = []
    for record in read(previous / "evaluation-summary.json")["selected_provider_runs"]:
        p = Path(record["result"])
        if digest(p) != record["sha256"]:
            raise ValueError("Prior result hash mismatch")
        raw = read(p)
        config = read(p.parent.parent / "plan.json")["scenarios"][raw["scenario_id"]]
        selected.append(("prior_selected", p.parent, raw, config))
        prior.append(
            {
                "cohort": "prior_selected",
                "scenario": raw["scenario_id"],
                "run_id": p.parent.name,
                "success": raw["success"],
                "model": config.get("model"),
                "client_completion_sec": raw["total_latency_sec"],
                "http_request_body_bytes": raw["total_request_bytes"],
                "http_response_body_bytes": raw["total_response_bytes"],
                "input_media_bytes": raw["metadata"].get("input_media_bytes"),
                "output_media_bytes": raw["metadata"].get("output_media_bytes"),
                "result_path": str(p),
                "sha256": record["sha256"],
                "pooling_policy": "Separate earlier 3-repeat cohort; not included in fresh 10-run statistics",
            }
        )
    groups = defaultdict(list)
    for campaign, run, raw, config in selected:
        meta = raw.get("metadata", {})
        base = {
            "campaign": campaign,
            "scenario": raw["scenario_id"],
            "run_id": run.name,
            "run_index": raw["run_index"],
            "success": raw["success"],
        }
        groups[(campaign, raw["scenario_id"])].append(raw)
        for i, p in enumerate(meta.get("phases", [])):
            phases.append(
                {
                    **base,
                    "phase_index": i,
                    "phase": p.get("phase"),
                    "method": p.get("method"),
                    "http_status": p.get("http_status"),
                    "complete": p.get("complete"),
                    "start_utc": datetime.fromtimestamp(
                        p["t_start"], timezone.utc
                    ).isoformat()
                    if p.get("t_start")
                    else None,
                    "phase_duration_sec": p.get("duration_sec"),
                    "request_body_bytes": p.get("request_body_bytes"),
                    "response_body_bytes": p.get("response_body_bytes"),
                    "body_send_callback_span_ms": delta(
                        p, "t_body_send_last", "t_body_send_first"
                    ),
                    "body_receive_callback_span_ms": delta(
                        p, "t_body_receive_last", "t_body_receive_first"
                    ),
                    "last_send_to_response_headers_ms": delta(
                        p, "t_response_headers", "t_body_send_last"
                    ),
                    "first_request_to_first_response_body_ms": delta(
                        p, "t_body_receive_first", "t_body_send_first"
                    ),
                    "last_request_to_first_response_body_ms": delta(
                        p, "t_body_receive_first", "t_body_send_last"
                    ),
                    "local_endpoint": p.get("local_endpoint"),
                    "remote_endpoint": p.get("remote_endpoint"),
                    "scope": "Client HTTP I/O callbacks; includes buffering/provider processing; not one-way delay",
                }
            )
        for direction, key in [("input", "inputs"), ("output", "outputs")]:
            for i, a in enumerate(meta.get(key, [])):
                asset_path = a.get("path")
                if not asset_path and direction == "input":
                    candidates = config.get("image_paths", []) + (
                        [config["video_path"]] if config.get("video_path") else []
                    )
                    asset_path = next(
                        (p for p in candidates if Path(p).name == a.get("name")), None
                    )
                if asset_path and a.get("sha256") and digest(asset_path) != a["sha256"]:
                    raise ValueError("Artifact hash mismatch: " + asset_path)
                artifacts.append(
                    {
                        **base,
                        "artifact_direction": direction,
                        "artifact_index": i,
                        "bytes": a.get("bytes"),
                        "name": a.get("name"),
                        "width": a.get("width"),
                        "height": a.get("height"),
                        "format": a.get("format", a.get("mime_type")),
                        "duration_sec": a.get("duration_sec"),
                        "codec": a.get("codec"),
                        "decoded_frames": a.get("decoded_frames"),
                        "validation": a.get("validation"),
                        "path": asset_path,
                        "sha256": a.get("sha256"),
                        "quality_validation": meta.get("quality_validation"),
                        "workload_kind": meta.get("workload_kind"),
                    }
                )
    for (campaign, scenario), runs in sorted(groups.items()):
        good = [r for r in runs if r["success"]]
        for metric, unit, values in [
            ("client_completion", "s", [r["total_latency_sec"] for r in good]),
            ("http_request_body", "B", [r["total_request_bytes"] for r in good]),
            ("http_response_body", "B", [r["total_response_bytes"] for r in good]),
            (
                "input_media",
                "B",
                [r["metadata"].get("input_media_bytes", 0) for r in good],
            ),
            (
                "output_media",
                "B",
                [r["metadata"].get("output_media_bytes", 0) for r in good],
            ),
        ]:
            summaries.append(
                {
                    "campaign": campaign,
                    "scenario": scenario,
                    "metric": metric,
                    "unit": unit,
                    "attempts": len(runs),
                    "successes": len(good),
                    "failures": len(runs) - len(good),
                    **distribution(values),
                    "population": "Successful runs only; failures retained in Provider_Runs",
                }
            )
    intervals = []
    for campaign, run, raw, config in selected:
        if campaign == "prior_selected":
            continue
        observed = raw.get("metadata", {}).get("phases", [])
        starts = [p["t_start"] for p in observed if p.get("t_start")]
        ends = [p["t_end"] for p in observed if p.get("t_end")]
        if starts and ends:
            intervals.append((campaign, run.name, min(starts), max(ends)))
    for r in rows:
        own = next(
            (x for x in intervals if x[:2] == (r["campaign"], r["run_id"])), None
        )
        if own:
            r["other_campaign_overlap_count"] = sum(
                other[0] != own[0] and other[2] < own[3] and own[2] < other[3]
                for other in intervals
            )
    return rows, phases, artifacts, audit, summaries, cov, prior


def controlled_rows(previous, caches):
    units = []
    delays = []
    summary = []
    coverage = []
    delay_groups = defaultdict(list)
    for p in sorted((previous / "controlled-analysis").glob("*.json")):
        if p.name in ("summary.json", "provenance.json"):
            continue
        d = read(p)
        run_id = p.stem
        extra = caches.get(run_id)
        if extra is None:
            raise ValueError("Controlled cache missing: " + run_id)
        for direction in ("ul", "dl"):
            u = d[direction]
            deadline = d["thresholds"].get(direction)
            elapsed = u.get("observation_elapsed_ms")
            assessed = u["complete"] or (
                u.get("source_unit_started")
                and deadline is not None
                and (elapsed or 0) >= deadline
            )
            miss = (
                (not u["complete"] or u["burst_transfer_ms"] > deadline)
                if deadline is not None and assessed
                else None
            )
            base = {
                "workload": d["workload"],
                "profile": d["profile"],
                "run_index": d["run_index"],
                "run_id": run_id,
                "direction": direction,
            }
            units.append(
                {
                    **base,
                    "transaction_success": d["success"],
                    "transaction_error_type": d.get("error_type"),
                    "measurement_valid": d["measurement_valid"],
                    "body_bytes": u["body_bytes"],
                    "body_MB": u["body_bytes"] / 1e6,
                    "unit_started": u.get("source_unit_started"),
                    "unit_complete": u["complete"],
                    "transfer_ms": u["burst_transfer_ms"],
                    "deadline_ms": deadline,
                    "deadline_assessable": bool(assessed) if deadline else None,
                    "deadline_miss": miss,
                    "observation_elapsed_ms": elapsed,
                    "achieved_body_mbps": u["body_bytes"]
                    * 0.008
                    / u["burst_transfer_ms"]
                    if u["complete"] and u["burst_transfer_ms"]
                    else None,
                    "packet_deadline_ms": 30
                    if d["workload"].startswith("chatbot")
                    else 20
                    if d["workload"].startswith(("image", "visual"))
                    else None,
                    "source_retransmitted_payload_bytes": u[
                        "source_retransmitted_payload_bytes"
                    ],
                    "source_result": d["source_result"],
                    "analysis_path": str(p),
                }
            )
            delay = u["matched_data_packet_delays_ms"]
            packet_deadline = units[-1]["packet_deadline_ms"]
            if d["measurement_valid"]:
                delay_groups[(d["workload"], d["profile"], direction)].extend(delay)
            delays.append(
                {
                    **base,
                    "measurement_valid": d["measurement_valid"],
                    "source_data_packet_attempts": u["source_data_packets"],
                    "matched_arrived_attempts": len(delay),
                    "undelivered_source_attempts": u["undelivered_source_attempts"],
                    "duplicate_received_packets": u["duplicate_received_data_packets"],
                    "unmatched_received_packets": u["unmatched_received_data_packets"],
                    "packet_deadline_ms": packet_deadline,
                    "arrived_attempt_deadline_misses": sum(
                        x > packet_deadline for x in delay
                    )
                    if packet_deadline
                    else None,
                    **flat_stats("arrived_attempt_delay_ms", distribution(delay)),
                    **flat_stats(
                        "first_piece_delivery_ms",
                        extra["first_delivery"][direction]["piece_delivery_ms"],
                    ),
                    **{
                        k: v
                        for k, v in extra["first_delivery"][direction].items()
                        if k not in ("clusters", "piece_delivery_ms")
                    },
                    "scope": "Attempt delays exclude lost attempts; first-piece delivery includes recovery and excludes duplicate byte transmissions",
                }
            )
    grouped = defaultdict(list)
    for r in units:
        grouped[(r["workload"], r["profile"], r["direction"])].append(r)
    for (workload, profile, direction), rs in sorted(grouped.items()):
        complete = [r for r in rs if r["unit_complete"] and r["measurement_valid"]]
        assessed = [
            r for r in rs if r["deadline_assessable"] and r["measurement_valid"]
        ]
        k = sum(r["deadline_miss"] for r in assessed)
        lo, hi = wilson(k, len(assessed))
        lower = [
            r
            for r in rs
            if direction == "dl"
            and workload.startswith("photo")
            and r["measurement_valid"]
            and (
                r["unit_complete"]
                or r["unit_started"]
                and (r["observation_elapsed_ms"] or 0) >= 1000
            )
        ]
        lower_miss = sum(
            not r["unit_complete"] or r["transfer_ms"] > 1000 for r in lower
        )
        summary.append(
            {
                "workload": workload,
                "profile": profile,
                "direction": direction,
                "runs": len(rs),
                "transaction_successes": sum(r["transaction_success"] for r in rs),
                "valid_captures": sum(r["measurement_valid"] for r in rs),
                "started_units": sum(bool(r["unit_started"]) for r in rs),
                "complete_units": len(complete),
                "incomplete_units": sum(not r["unit_complete"] for r in rs),
                "body_bytes": rs[0]["body_bytes"],
                "deadline_ms": rs[0]["deadline_ms"],
                "assessable_units": len(assessed) if rs[0]["deadline_ms"] else None,
                "deadline_misses": k if rs[0]["deadline_ms"] else None,
                "miss_fraction": k / len(assessed) if assessed else None,
                "wilson95_low": lo,
                "wilson95_high": hi,
                "photo_DL_1000ms_assessable": len(lower) if lower else None,
                "photo_DL_1000ms_misses": lower_miss if lower else None,
                **flat_stats(
                    "transfer_ms", distribution([r["transfer_ms"] for r in complete])
                ),
                **flat_stats(
                    "achieved_body_mbps",
                    distribution([r["achieved_body_mbps"] for r in complete]),
                ),
                "scope": "Full source-defined application body, first source byte to complete receiver body; empirical observations",
            }
        )
        if direction == "ul":
            coverage.append(
                {
                    "cohort": "controlled_existing_10",
                    "workload": workload,
                    "profile": profile,
                    "planned": len(rs),
                    "attempted": len(rs),
                    "succeeded": sum(r["transaction_success"] for r in rs),
                    "failed": sum(not r["transaction_success"] for r in rs),
                    "valid_captures": sum(r["measurement_valid"] for r in rs),
                }
            )
    delay_summary = []
    for key, values in sorted(delay_groups.items()):
        selected = [
            d
            for d in delays
            if (d["workload"], d["profile"], d["direction"]) == key
            and d["measurement_valid"]
        ]
        pieces = sum(d["first_piece_delivery_ms_n"] for d in selected)
        assessed = sum(d["piece_deadline_assessed"] or 0 for d in selected)
        threshold = selected[0]["packet_deadline_ms"]
        delay_summary.append(
            {
                "workload": key[0],
                "profile": key[1],
                "direction": key[2],
                "valid_runs": len(selected),
                "packet_deadline_ms": threshold,
                "source_attempts": sum(
                    d["source_data_packet_attempts"] for d in selected
                ),
                "undelivered_source_attempts": sum(
                    d["undelivered_source_attempts"] for d in selected
                ),
                "arrived_attempt_deadline_misses": sum(x > threshold for x in values)
                if threshold
                else None,
                **flat_stats("arrived_attempt_delay_ms", distribution(values)),
                "completed_first_pieces": pieces,
                "first_piece_delivery_mean_ms": sum(
                    d["first_piece_delivery_ms_mean"] * d["first_piece_delivery_ms_n"]
                    for d in selected
                    if d["first_piece_delivery_ms_n"]
                )
                / pieces
                if pieces
                else None,
                "first_piece_delivery_max_ms": max(
                    (
                        d["first_piece_delivery_ms_max"]
                        for d in selected
                        if d["first_piece_delivery_ms_n"]
                    ),
                    default=None,
                ),
                "incomplete_first_pieces": sum(
                    d["incomplete_pieces"] for d in selected
                ),
                "first_piece_deadline_assessed": assessed if threshold else None,
                "first_piece_deadline_misses": sum(
                    d["piece_deadline_misses"] or 0 for d in selected
                )
                if threshold
                else None,
                "scope": "Pooled packet attempts for quantiles; first-piece mean weighted by completed pieces, maximum across runs; recovery included only in first-piece metric",
            }
        )
    return units, delays, summary, coverage, delay_summary


def cache_rows(root):
    for p in sorted((root / "analysis-cache").glob("*/*.json.gz")):
        with gzip.open(p, "rt") as stream:
            d = json.load(stream)
        yield p, d


def capture_tables(book, root):
    cache = list(cache_rows(root))
    basekeys = [
        "campaign",
        "scenario",
        "profile",
        "run_id",
        "run_index",
        "success",
        "measurement_valid",
        "direction",
        "point",
        "population",
    ]

    def base(d, v, pop):
        return {
            **{k: d[k] for k in basekeys[:7]},
            "direction": v["direction"],
            "point": v["point"],
            "population": pop,
        }

    def packets():
        for p, d in cache:
            for v in d["views"]:
                for r in v["packet"]:
                    yield {
                        **base(d, v, r["population"]),
                        **{
                            k: val
                            for k, val in r.items()
                            if k not in ("population", "size_bytes", "iat_ms")
                        },
                        **flat_stats("IP_payload_B", r["size_bytes"]),
                        **flat_stats("IAT_ms", r["iat_ms"]),
                    }

    book.sheet(
        "Packet_Stats",
        packets(),
        "Each run / direction / capture point / population. Size = IP payload bytes (TCP header + TCP payload). Host offload records are not wire packets; see measurement_valid.",
    )
    packet_groups = defaultdict(list)
    for p, d in cache:
        for v in d["views"]:
            for r in v["packet"]:
                key = (
                    d["campaign"],
                    d["scenario"],
                    d["profile"],
                    v["direction"],
                    v["point"],
                    r["population"],
                    d["success"],
                    d["measurement_valid"],
                )
                packet_groups[key].append(r)
    packet_summary = []
    for key, rs in sorted(packet_groups.items()):
        b = dict(
            zip(
                [
                    "campaign",
                    "scenario",
                    "profile",
                    "direction",
                    "point",
                    "population",
                    "success",
                    "measurement_valid",
                ],
                key,
            )
        )
        n = sum(r["packet_count"] for r in rs)
        total = sum(r["ip_payload_bytes"] for r in rs)
        average = total / n if n else None
        criterion = (
            "<800" if b["scenario"].startswith(("chatbot", "sa1_chat")) else ">800"
        )
        if b["scenario"] == "sa1_video_generation":
            criterion = (
                None  # Additional generated video has no separate SA1 numeric target.
            )
        packet_summary.append(
            {
                **b,
                "runs": len(rs),
                "packets": n,
                "IP_payload_bytes": total,
                "packet_weighted_mean_B": average,
                "minimum_B": min(
                    (r["size_bytes"]["min"] for r in rs if r["packet_count"]),
                    default=None,
                ),
                "maximum_B": max(
                    (r["size_bytes"]["max"] for r in rs if r["packet_count"]),
                    default=None,
                ),
                "ACK_only_records": sum(r["ack_only_count"] for r in rs),
                "oversized_records": sum(r["oversized_records"] for r in rs),
                "draft_mean_packet_criterion_B": criterion,
                "mean_meets_numeric_criterion": (
                    average < 800 if criterion == "<800" else average > 800
                )
                if average is not None
                and criterion is not None
                and b["measurement_valid"]
                else None,
                **flat_stats(
                    "run_mean_packet_B",
                    distribution(
                        [r["size_bytes"]["mean"] for r in rs if r["packet_count"]]
                    ),
                ),
                "scope": "Numeric comparison only; population/transport choice affects applicability. Invalid host records are not assessed.",
            }
        )
    book.sheet(
        "Packet_Summary",
        packet_summary,
        "Answers mean UL/DL packet size with packet-weighted totals and separately distributed run means. ACK-inclusive and data-only populations remain separate. Numeric draft comparison is not a compliance claim.",
    )

    def bursts():
        for p, d in cache:
            for v in d["views"]:
                for r in v["burst"]:
                    yield {
                        **base(d, v, r["population"]),
                        **{
                            k: val
                            for k, val in r.items()
                            if k
                            not in (
                                "population",
                                "size_bytes",
                                "duration_ms",
                                "idle_ms",
                                "mean_rate_mbps",
                            )
                        },
                        **flat_stats("size_B", r["size_bytes"]),
                        **flat_stats("duration_ms", r["duration_ms"]),
                        **flat_stats("idle_ms", r["idle_ms"]),
                        **flat_stats("mean_rate_mbps", r["mean_rate_mbps"]),
                    }

    book.sheet(
        "Burst_Runs",
        bursts(),
        "Strict > gap splits a burst; gaps 5/10/20/50/100/200/500/1000 ms. Duration is last-minus-first timestamp at one capture point, not one-way latency. Singleton mean rate is blank.",
    )
    groups = defaultdict(list)

    def details():
        for p, d in cache:
            for v in d["views"]:
                for r in v["burst_details"]:
                    key = (
                        d["campaign"],
                        d["scenario"],
                        d["profile"],
                        v["direction"],
                        v["point"],
                        r["population"],
                        r["idle_gap_ms"],
                        d["success"],
                        d["measurement_valid"],
                    )
                    groups[key].append(r)
                    yield {**base(d, v, r["population"]), **r}

    book.sheet(
        "Burst_Details",
        details(),
        "Every observed packet cluster, including per-cluster peak rates in 1/5/10/50/100 ms windows anchored at its first packet. Data population includes framing/TLS/control payload and retransmissions.",
    )
    aggregate = []
    for key, rs in sorted(groups.items()):
        b = dict(
            zip(
                [
                    "campaign",
                    "scenario",
                    "profile",
                    "direction",
                    "point",
                    "population",
                    "idle_gap_ms",
                    "success",
                    "measurement_valid",
                ],
                key,
            )
        )
        for metric in (
            "ip_payload_bytes",
            "duration_ms",
            "preceding_idle_ms",
            "mean_rate_mbps",
            "peak_1ms_mbps",
            "peak_5ms_mbps",
            "peak_10ms_mbps",
            "peak_50ms_mbps",
            "peak_100ms_mbps",
        ):
            vals = [r[metric] for r in rs if r[metric] is not None]
            aggregate.append(
                {
                    **b,
                    "metric": metric,
                    **distribution(vals),
                    "weighting": "Pooled bursts, one observation per burst; not run-weighted",
                }
            )
    book.sheet(
        "Burst_Summary",
        aggregate,
        "Pooled empirical burst distributions. Filter by cohort, success, validity, capture point and population. p50/p95/p99 refer to individual bursts; per-run summaries are in Burst_Runs.",
    )
    groups.clear()

    def windows():
        for p, d in cache:
            for v in d["views"]:
                for r in v["window"]:
                    yield {**base(d, v, r["population"]), **r}

    book.sheet(
        "Rate_Windows",
        windows(),
        "Fixed windows 1/5/10/50/100/1000/10000 ms, common connection origin. Includes zero windows and pads final bin. Reports both all-window and active-window means and peak/mean ratios.",
    )

    def clusters():
        for p, d in cache:
            for direction, delivery in d.get("first_delivery", {}).items():
                for r in delivery["clusters"]:
                    yield {
                        **{k: d[k] for k in basekeys[:7]},
                        "direction": direction,
                        **r,
                    }

    book.sheet(
        "Source_Burst_Latency",
        clusters(),
        "Controlled unique body bytes clustered at the source. First transmission to first successful delivery of all cluster bytes, including loss recovery. Source grouping fixed before receiver inspection. Full-unit deadlines do not apply to arbitrary subclusters.",
    )
    return {d["run_id"]: d for p, d in cache if d["kind"] == "controlled"}, aggregate


def historical_rows(path):
    d = read(path)
    results = []
    burst = []
    for pair in d["pairs"]:
        if pair["scenario"] not in ("video_generation", "image_enhancement"):
            continue
        s = pair["summary"]
        base = {
            "cohort": "archived_20260918",
            "scenario": pair["scenario"],
            "profile": pair["profile"],
            "runs": s["runs"],
            "successes": s["successes"],
        }
        for metric in (
            "latency_sec",
            "request_body_bytes",
            "response_body_bytes",
            "input_media_bytes",
            "output_media_bytes",
        ):
            results.append(
                {
                    **base,
                    "metric": metric,
                    **s[metric],
                    "archive_summary_path": str(path),
                    "capture_sha256": pair["capture_sha256"],
                    "limitations": "Historical provider/model and input cohort; photo input was synthetic 512x288, not six 4K raw frames; no capture-drop counters; do not pool with fresh results",
                }
            )
        for direction, v in s["directional"].items():
            for gap, m in v["bursts"].items():
                for metric, values in m.items():
                    burst.append(
                        {
                            **base,
                            "direction": direction,
                            "idle_gap_ms": int(gap),
                            "metric": metric,
                            **values,
                            "oversized_host_records": v["records_over_1480_bytes"],
                            "wire_packet_metrics_valid": False,
                            "scope": "Archived host-record aggregate, old connection attribution; not used for current wire-size conclusions",
                        }
                    )
    return results, burst


def methodology():
    return [
        {"topic": a, "definition_and_limitations": b}
        for a, b in [
            (
                "Units",
                "B = bytes; MB = 1,000,000 bytes; Mb/s = 1,000,000 bits/s; ms = milliseconds. Blank = unavailable/not applicable; numeric zero = observed zero.",
            ),
            (
                "Cohorts",
                "Fresh provider runs, selected prior provider cohort, existing controlled trials, and archived provider evidence. Actual counts and dates are in the run sheets. Never pooled across cohorts.",
            ),
            (
                "What SA1 asks",
                "S4-261873 / S1-263628 requests average UL/DL burst and packet sizes, maximum UL/DL burst latency and packet latency. The five listed cases include gaming, explicitly excluded by user.",
            ),
            (
                "Draft values",
                "TS 22.270 v0.3.0 table 6.1.1-1 values are bracketed draft requirements. Empirical maxima are observations, not universal maximum bounds or certification.",
            ),
            (
                "Flow isolation",
                "Reportable PCAPs are filtered to exact local+remote TCP socket tuples recorded by the scenario HTTP transport, in both directions. Other processes using the same service IP/443 but different source ports are excluded. Original broad acquisition remains local audit evidence only.",
            ),
            (
                "Controlled isolation",
                "Separate client/router/server Linux network namespaces, dedicated port 8098; PCAPs on scenario veth interfaces. No host-side background traffic enters controlled data.",
            ),
            (
                "Shared provider link",
                "Filtering removes unrelated records from captures and statistics; it does not prevent other host traffic from competing on the physical link. Provider_Runs identifies overlapping measured campaigns. Cloud baselines use the existing host network; controlled namespaces provide the separate impairment experiment.",
            ),
            (
                "Capture quality",
                "Wire packet inference requires zero captured drops, no malformed data, appropriate packetization and no observed IP payload >1480 B at IPv4 MTU 1500. Provider NIC offload/coalescing means host records cannot establish wire packet sizes or sub-ms burst structure even with zero drops.",
            ),
            (
                "All TCP population",
                "Includes handshake, ACK-only, FIN/control, application data, retransmissions, length framing and acknowledgements in the controlled fixture. Captured IP payload includes TCP headers. Direction is initiator UL / server DL.",
            ),
            (
                "TCP data population",
                "Only records with non-empty TCP payload; still includes TLS/HTTP/framing, retransmissions and fixture control payload. Not identical to application content packets. Source_Burst_Latency clips to unique application body bytes.",
            ),
            (
                "Application data unit",
                "Full configured body size. Controlled unit latency = first source body byte to complete successful receiver body. Complete UL may be valid even if a later DL timeout causes transaction failure.",
            ),
            (
                "Idle-gap burst",
                "Consecutive packets in a direction/population; new cluster when interarrival gap strictly exceeds 5,10,20,50,100,200,500,1000 ms. Equality stays in the same burst. Groups are recomputed separately at each capture point.",
            ),
            (
                "Burst duration",
                "Last packet timestamp minus first timestamp at a single point. A one-packet burst has duration zero and undefined byte/duration rate; fixed-window peak remains defined. Inter-burst idle = next first minus prior last.",
            ),
            (
                "Per-burst peak",
                "Each burst has 1/5/10/50/100 ms bins anchored at its first packet; peak is maximum bytes*8/window. Burst mean byte/duration rate is not called peak.",
            ),
            (
                "Rate windows",
                "Connection-span fixed bins use common origin across directions, include empty bins, and pad the final partial bin to full width. W=1,5,10,50,100,1000,10000 ms. Mean over all bins and nonzero bins, active fraction, CV, peak/mean reported separately.",
            ),
            (
                "Attempt packet delay",
                "Matched delivered packet attempt from source capture to destination capture; duplicate arrivals counted separately. Lost attempts have no finite arrival delay. This distribution alone is survivor-biased when losses occur.",
            ),
            (
                "First-piece delivery",
                "Partition original body bytes by their first observed source transmission. For each piece, measure until all its bytes have first arrived, allowing retransmission/resegmentation recovery. Incomplete pieces retained. This is a transport-recovery metric, distinct from one delivered packet attempt.",
            ),
            (
                "Source burst latency",
                "Group first-transmission pieces by source idle gaps, then measure until all group bytes are successfully delivered. Groups are not recomputed at receiver. Do not apply the full 20-MB unit deadline to these arbitrary smaller clusters.",
            ),
            (
                "Deadlines and timeout",
                "Controlled transaction timeout 20 s. Incomplete unit is a deadline miss only if unit started and observed duration reaches deadline. Not-started/unassessable units are not passes. Photo DL assessed at both 1000 and 1500 ms.",
            ),
            (
                "Cloud timing",
                "Client submit-to-result/validated artifact and HTTP callback timing include provider computation, connection setup, buffering and serialization. No remote capture; cannot infer one-way packet/unit latency or provider-only compute. Streaming callbacks are not TTFT token events.",
            ),
            (
                "Statistics",
                "NumPy linear empirical quantiles, population standard deviation, CV=std/mean. n is shown. Burst summary pools bursts; per-run statistics stay separate. p99 with 10 trials is near observed max and not a reliable tail bound.",
            ),
            (
                "Dependence",
                "Randomized workload order and fresh TCP connection per controlled trial; kernel CUBIC path metrics were retained. Trials are not cold-start independent TCP states. Wilson 95% intervals are descriptive and do not prove reliability.",
            ),
            (
                "Clock uncertainty",
                "Controlled endpoints share kernel CLOCK_REALTIME, PCAP microsecond resolution. Software timestamp placement uncertainty is not calibrated; tiny no-emulation delays are fixture measurements, not external network latency.",
            ),
            (
                "Photo assets",
                "Six public HDR+ raw frames are cropped and compressed with NPZ/DEFLATE, not Jetraw. The selected burst, dimensions and measured sizes are in Raw_Assets. Camera-oriented JPEG renders sent to OpenAI are a content proxy, not raw transport.",
            ),
            (
                "Photo result quality",
                "Cloud outputs are decoded successfully, but output size/aspect differ from the 4K input. No objective restoration quality or geometry-preservation assessment. One scene only; no claim of full raw enhancement compliance.",
            ),
            (
                "Visual assistant proxy",
                "OpenAI receives 10 sampled JPEG frames at 1 s intervals from a bundled 10 s 640x360 video, not native video or audio. Results do not establish continuous-motion or audio understanding.",
            ),
            (
                "Google billing",
                "Initial campaign has one HTTP 402 attempt each for native-video assistant and video generation; remaining initial repetitions explicitly blocked. Any funded rerun is a separate cohort with its own plan and outcomes. OpenAI Videos API is retired.",
            ),
            (
                "Gaming",
                "No gaming evaluation was performed. Training moved out of this repository and is not part of this campaign.",
            ),
        ]
    ]


def requirements():
    rows = []
    for case, direction, size, unit_ms, packet_ms, rate, sizecriterion in [
        ("Chatbot", "UL", "0.0005–0.006", 20, 30, 0.2, "<800"),
        ("Chatbot", "DL", "0.0005", 20, 30, 0.2, "<800"),
        ("Image GenAI", "UL", "0.0005–0.4", 50, 20, 64, ">800"),
        ("Image GenAI", "DL", None, None, 20, 16, ">800"),
        ("Video GenAI / visual assistant", "UL", "20", 400, 20, 400, ">800"),
        ("Video GenAI / visual assistant", "DL", None, None, 20, None, ">800"),
        ("Synergised photo enhancement", "UL", "20", 1500, None, 100, ">800"),
        ("Synergised photo enhancement", "DL", "20", "1000–1500", None, 100, ">800"),
    ]:
        rows.append(
            {
                "use_case": case,
                "direction": direction,
                "draft_average_burst_MB": size,
                "draft_maximum_burst_latency_ms": unit_ms,
                "draft_maximum_packet_latency_ms": packet_ms,
                "draft_data_rate_Mbps": rate,
                "draft_average_packet_B": sizecriterion,
                "source": "TS 22.270 v0.3.0 / S1-263335, table 6.1.1-1",
                "status": "Bracketed draft values; blanks unspecified",
            }
        )
    return rows


def build(args):
    root = Path(args.campaign).resolve()
    previous = Path(args.previous).resolve()
    sources = Path(args.sources).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assets_path = Path(read(previous / "evaluation-summary.json")["assets_manifest"])
    rows, phases, artifacts, audit, psummary, coverage, prior = provider_rows(
        root, previous
    )
    if any(r["state"] in ("unresolved", "not_started") for r in rows):
        raise ValueError(
            "Campaign still has unfinished jobs; do not label workbook final"
        )
    for r in rows:
        if r["result_path"]:
            cache_path = (
                root / "analysis-cache" / r["campaign"] / (r["run_id"] + ".json.gz")
            )
            with gzip.open(cache_path, "rt") as stream:
                cached = json.load(stream)
            if cached["input_hashes"]["result.json"] != r["result_sha256"]:
                raise ValueError("Stale provider analysis: " + r["run_id"])
            kept = sum(
                p["packet_count"]
                for v in cached["views"]
                for p in v["packet"]
                if p["population"] == "all_tcp"
            )
            filtered = read(Path(r["scenario_capture_path"] + ".metadata.json"))
            if kept != filtered["retained_packet_records"]:
                raise ValueError(
                    "Analysis/export flow selection differs: " + r["run_id"]
                )
    book = Book(out)
    book.sheet(
        "SA1_Questions",
        [
            {
                "question": "Average UL/DL burst sizes",
                "evidence_sheets": "Controlled_Units; Provider_Summary; Burst_Summary; Burst_Details",
                "answer": "Report application body sizes separately from idle-gap cluster sizes; gap sensitivity depends strongly on direction, packet population and loss recovery.",
                "limits": "Cloud HTTP bodies include encoding, not raw media bytes. Provider host-record burst timing is offload-affected.",
            },
            {
                "question": "Average UL/DL packet sizes",
                "evidence_sheets": "Packet_Stats; Packet_Delays; Historical_Bursts",
                "answer": "Controlled captures provide wire-sized records. Report both all TCP and data-only means; ACKs substantially change the result.",
                "limits": "Cloud host records contain coalesced packets; cannot confirm SA1 wire-packet thresholds from provider PCAPs.",
            },
            {
                "question": "Maximum UL/DL burst end-to-end latency",
                "evidence_sheets": "Controlled_Summary; Controlled_Units; Source_Burst_Latency",
                "answer": "Full-unit first-transmission-to-complete-receipt latency, observed maxima and deadline misses; photo DL tested against 1000 and 1500 ms.",
                "limits": "Source-defined units differ from idle-gap clusters. Public API completion is not one-way transfer latency.",
            },
            {
                "question": "Maximum UL/DL packet end-to-end latency",
                "evidence_sheets": "Packet_Delays",
                "answer": "Delivered-attempt delays with missing/lost attempts counted; additional first-piece delivery statistics include TCP loss recovery.",
                "limits": "Empirical maxima only. No synchronized remote cloud endpoint capture. Photo packet target unspecified.",
            },
            {
                "question": "Burst duration criteria and burstiness (related SA4 methodology)",
                "evidence_sheets": "Burst_Runs; Burst_Details; Burst_Summary; Rate_Windows; Source_Burst_Latency",
                "answer": "Idle thresholds 5/10/20/50/100/200/500/1000 ms; rate windows 1/5/10/50/100/1000/10000 ms. Mean, p50, p95, p99, maximum, variability and duty/active fraction.",
                "limits": "TR 26.870 C.5.3 explicitly uses 10 and 100 ms idle gaps; additional thresholds are sensitivity analysis, not SA1 requirements.",
            },
            {
                "question": "Rate assumptions and scenario scope",
                "evidence_sheets": "Draft_Requirements; Controlled_Summary; Network_Profiles; Methodology",
                "answer": "100/400/600 Mb/s capacity sensitivity plus calibrated 5G urban, LEO and no-emulation. Five SA1 cases listed; gaming excluded. Generated video is an additional workload distinct from visual assistant.",
                "limits": "Capacity sweeps are controlled fixture settings, not calibrated radio profiles or independent cloud network experiments.",
            },
        ],
        "Maps each requested SA1 quantity to evidence and states what can and cannot be concluded.",
    )
    book.sheet(
        "Draft_Requirements",
        requirements(),
        "Source values from TS 22.270 v0.3.0, table 6.1.1-1. Bracketed draft requirements; decimal MB. No new numeric targets invented for additional generated video.",
    )
    book.sheet(
        "Coverage",
        coverage,
        "Fresh campaign ledger: 10 planned repetitions per variant. Blocked/not attempted is distinct from an attempted failure. OpenAI frame assistant is a proxy.",
    )
    book.sheet(
        "Provider_Summary",
        psummary,
        "Statistics on successful runs. Earlier selected 3-repeat cohort is separate from fresh ten-run cohorts. Completion includes processing; does not answer one-way SA1 latency.",
    )
    book.sheet(
        "Provider_Runs",
        rows,
        "Every planned fresh job, including failures and blocked repetitions. Source result and filtered scenario-PCAP hashes allow audit. No unrelated host traffic in linked scenario PCAPs.",
    )
    book.sheet(
        "Capture_Isolation",
        audit,
        "Exact measured TCP socket tuple allowlist; independent artifact re-read checks zero unrelated records. Original capture drops apply conservatively. See Methodology for host offload limitations.",
    )
    book.sheet(
        "HTTP_Phases",
        phases,
        "Per-phase measured body bytes and client callback spans. Request-to-response times include provider processing and buffering. They are not one-way interface transfer latency.",
    )
    book.sheet(
        "Media_Artifacts",
        artifacts,
        "Input/output artifact sizes, dimensions, validation and hashes, including prior cohort. Complete decoding verifies file structure, not perceptual quality or source preservation.",
    )
    asset_manifest = read(assets_path)
    book.sheet(
        "Raw_Assets",
        ({"field": k, "value": v} for k, v in asset_manifest.items()),
        "Public raw-photo asset provenance, transformations, dimensions, actual compression sizes and derived JPEG inputs. Values loaded from the preserved input manifest.",
    )
    controlled, aggregates = capture_tables(book, root)
    units, delays, csum, ccov, delay_summary = controlled_rows(previous, controlled)
    book.sheet(
        "Controlled_Summary",
        csum,
        "Per workload/profile. Full application unit latency and deadline outcomes. Completed units only for finite latency distributions; started/incomplete/unassessable counts retained.",
    )
    book.sheet(
        "Controlled_Units",
        units,
        "One source-defined body per direction per run. Whole transaction may time out after UL already completed. Photo has both 20-MB baseline and actual 35.101264-MB compressed raw input.",
    )
    book.sheet(
        "Controlled_Coverage",
        ccov,
        "Existing controlled matrix, newly analyzed at all requested burst thresholds; not rerun unchanged. Counts are derived from authoritative per-run analysis.",
    )
    book.sheet(
        "Packet_Delays",
        delays,
        "Two distinct timing scopes: matched delivered attempts vs first source-byte-piece successful delivery including recovery. Miss denominators explicit; no delay invented for undelivered data.",
    )
    book.sheet(
        "Packet_Delay_Summary",
        delay_summary,
        "Per workload/profile/direction: pooled successful-attempt delays, all lost-attempt counts, and recovery-inclusive first-piece mean/max. Losses cannot be treated as zero latency.",
    )
    profiles = []
    plan = read(previous / "controlled-final/plan.json")
    for name, c in plan["profiles"].items():
        for direction in ("ul", "dl"):
            p = {**c, **c.get("uplink" if direction == "ul" else "downlink", {})}
            profiles.append(
                {
                    "profile": name,
                    "direction": direction,
                    "rate_Mbps": p.get("rate_mbit"),
                    "delay_ms": p.get("delay_ms"),
                    "jitter_ms": p.get("jitter_ms"),
                    "loss_pct": p.get("loss_pct"),
                    "duplicate_pct": p.get("duplicate_pct"),
                    "queue_limit_packets": p.get("limit_packets"),
                    "delay_distribution": p.get("delay_distribution"),
                    "rate_mode": p.get("rate_mode"),
                    "bypass": p.get("bypass", False),
                    "classification": "Capacity sensitivity; not field calibrated"
                    if name.startswith("capacity")
                    else "Catalog profile"
                    if name != "no_emulation"
                    else "Namespace baseline",
                    "catalog_sha256": plan["profiles_sha256"],
                    "effective_parameters": p,
                }
            )
    book.sheet(
        "Network_Profiles",
        profiles,
        "Parameters pinned by controlled campaign plan. Uplink override merged explicitly (LEO UL 15 Mb/s). Both directions shaped separately; capacity cases add 1 ms one-way delay.",
    )
    book.sheet(
        "Prior_Provider_Runs",
        prior,
        "Selected 18 earlier valid provider outcomes; three per variant, kept separate. Bad-crop and sideways input pilots excluded from primary results.",
    )
    if args.historical:
        hist, hburst = historical_rows(args.historical)
        book.sheet(
            "Historical_Provider",
            hist,
            "Archived 2026-09-18 video generation and image enhancement: three runs per scenario/profile. Historical evidence, not fresh ten-run completion. Photo used small synthetic input.",
        )
        book.sheet(
            "Historical_Bursts",
            hburst,
            "Archived 10/100 ms host-record cluster aggregates. No capture-drop counters, offload coalescing and older attribution. Do not use these to establish wire-packet sizes.",
        )
    book.sheet(
        "Methodology",
        methodology(),
        "Read before interpreting SA1 compliance. Units, scope, definitions, weighting, missing-data handling and practical limitations.",
    )
    exclusions = [
        {
            "evidence": "Gaming",
            "reason": "Explicitly excluded by user; no gaming work performed.",
        },
        {
            "evidence": "Training",
            "reason": "Moved to sibling 6G-Testbed-training; outside this evaluation.",
        },
        {
            "evidence": "provider-pilot / hdrplus early assets",
            "reason": "Development pilots including padded bad crop; not primary measurements.",
        },
        {
            "evidence": "provider-campaign image_conditioned/photo runs",
            "reason": "Earlier rendered JPEG orientation incorrect; replaced by selected provider-photo-final cohort and fresh runs.",
        },
        {
            "evidence": "controlled-pilot / controlled-campaign / controlled-campaign-v2",
            "reason": "Interrupted or development fixture runs; final controlled-analysis is authoritative.",
        },
        {
            "evidence": "Google blocked jobs",
            "reason": "Not attempted; explicitly retained in fresh ledger. Do not count as failed measurements or successful zero-byte outputs.",
        },
        {
            "evidence": "Background host traffic",
            "reason": "Removed from reportable provider PCAPs using exact measured connections; broad originals kept only as local audit evidence.",
        },
        {
            "evidence": "Archived September provider data",
            "reason": "Separate historical context; not pooled with current provider/asset/model cohorts.",
        },
    ]
    book.sheet(
        "Exclusions",
        exclusions,
        "Why older/development artifacts do not enter primary measurements; failures within accepted campaigns remain visible.",
    )
    source_rows = []
    for s in read(sources / "sources.json"):
        source_rows.append(
            {
                "document": s["number"],
                "meeting": s["meeting"],
                "url_or_path": s["source_url"],
                "sha256": s["sha256"],
                "purpose": "Source liaison / contribution / draft requirement",
            }
        )
    for path, purpose in [
        (sources / "text/S4-261521.txt", "TR 26.870 related SA4 burst methodology"),
        (
            previous / "evaluation-summary.json",
            "Selection of valid earlier provider cohort",
        ),
        (
            previous / "controlled-analysis/summary.json",
            "Authoritative earlier controlled analysis",
        ),
        (
            previous / "controlled-final/plan.json",
            "Controlled profile/workload/input plan",
        ),
        (
            previous / "controlled-runtime-observed.json",
            "Observed CUBIC/path-cache state",
        ),
        (
            previous / "cleanup-verification.json",
            "Network namespace cleanup verification",
        ),
        (
            root / "provider-openai/plan.json",
            "Fresh OpenAI inputs/config/source hashes",
        ),
        (root / "provider-google/plan.json", "Fresh Google attempt and blocked plan"),
        (
            root / "provider-google-funded/plan.json",
            "Google funded rerun plan, when present",
        ),
        (
            assets_path,
            "Public HDR+ asset provenance",
        ),
        (Path(__file__), "Workbook generation source"),
        (
            Path(__file__).with_name("sa1_excel_metrics.py"),
            "Burst and packet analysis source",
        ),
        (
            Path(__file__).with_name("sa1_capture_filter.py"),
            "Exact scenario capture filter source",
        ),
    ]:
        if path.exists():
            source_rows.append(
                {
                    "document": path.name,
                    "meeting": None,
                    "url_or_path": str(path),
                    "sha256": digest(path),
                    "purpose": purpose,
                }
            )
    source_rows.append(
        {
            "document": "HDR+ dataset",
            "meeting": None,
            "url_or_path": "https://hdrplusdata.org/dataset.html",
            "sha256": None,
            "purpose": "Hasinoff et al. (2016), CC BY-SA 4.0; burst 0006_20160722_115157_431. Derived artifacts retain attribution.",
        }
    )
    book.sheet(
        "Evidence",
        source_rows,
        "Document provenance and campaign/source hashes. Per-run result and scenario PCAP hashes are on the run sheets; full raw PCAPs remain outside Excel.",
    )
    # Compact chart data keeps the dashboard readable; detailed sheets retain distributions.
    chartrows = []
    for s in psummary:
        if (
            s["campaign"] in ("provider-openai", "provider-google-funded")
            and s["metric"] == "client_completion"
        ):
            chartrows.append(
                {
                    "label": LABELS[s["scenario"]],
                    "mean_completion_sec": s["mean"],
                    "p95_completion_sec": s["p95"],
                    "successful_runs": s["n"],
                }
            )
    book.sheet(
        "Chart_Data",
        chartrows,
        "Fresh successful-provider completion times. These include provider processing and must not be compared directly with SA1 network latency deadlines.",
    )
    for target in (book, book.summary):
        chart = target.wb.add_chart({"type": "bar"})
        for col, label in [(1, "Mean"), (2, "p95")]:
            chart.add_series(
                {
                    "name": label,
                    "categories": ["Chart_Data", 5, 0, 4 + len(chartrows), 0],
                    "values": ["Chart_Data", 5, col, 4 + len(chartrows), col],
                    "categories_data": [r["label"] for r in chartrows],
                    "values_data": [
                        r["mean_completion_sec" if col == 1 else "p95_completion_sec"]
                        for r in chartrows
                    ],
                }
            )
        chart.set_title({"name": "Client completion — successful runs"})
        chart.set_x_axis({"name": "Seconds (includes provider processing)"})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 850, "height": 380})
        target.overview.insert_chart("J2", chart)
    sensitivity = []
    for gap in (5, 10, 20, 50, 100, 200, 500, 1000):
        row = {"idle_gap_ms": gap}
        for profile in ("no_emulation", "5g_urban", "capacity_100", "capacity_400"):
            matches = [
                r
                for r in aggregates
                if r["campaign"] == "controlled_existing_10"
                and r["scenario"] == "photo_spec_payload"
                and r["direction"] == "ul"
                and r["point"] == "sender"
                and r["population"] == "tcp_data"
                and r["profile"] == profile
                and r["idle_gap_ms"] == gap
                and r["measurement_valid"]
                and r["metric"] == "ip_payload_bytes"
            ]
            n = sum(r["n"] for r in matches)
            row[profile + "_mean_MB"] = (
                sum(r["mean"] * r["n"] for r in matches) / n / 1e6 if n else None
            )
        sensitivity.append(row)
    book.sheet(
        "Burst_Gap_Chart",
        sensitivity,
        "Controlled 20-MB photo UL: source TCP-data packet clusters, pooled across all ten attempts per profile. IP payload includes TCP/framing/retransmissions. Full application body stays fixed while gap grouping changes.",
    )
    for target in (book, book.summary):
        gapchart = target.wb.add_chart({"type": "line"})
        for col, profile in enumerate(
            ("no_emulation", "5g_urban", "capacity_100", "capacity_400"), 1
        ):
            gapchart.add_series(
                {
                    "name": profile,
                    "categories": ["Burst_Gap_Chart", 5, 0, 12, 0],
                    "values": ["Burst_Gap_Chart", 5, col, 12, col],
                    "categories_data": [r["idle_gap_ms"] for r in sensitivity],
                    "values_data": [r[profile + "_mean_MB"] for r in sensitivity],
                    "marker": {"type": "circle", "size": 4},
                }
            )
        gapchart.set_title({"name": "Burst definition changes measured size"})
        gapchart.set_x_axis({"name": "Idle gap threshold (ms)"})
        gapchart.set_y_axis(
            {"name": "Mean observed burst (decimal MB)", "log_base": 10}
        )
        gapchart.set_legend({"position": "bottom"})
        gapchart.set_size({"width": 850, "height": 380})
        target.overview.insert_chart("J23", gapchart)
    fresh_success = sum(r["state"] == "success" for r in rows)
    failed = sum(r["state"] == "failed" for r in rows)
    blocked = sum(r["state"] == "blocked_not_attempted" for r in rows)
    findings = []
    for workload, profile in [
        ("visual_assistant", "capacity_400"),
        ("visual_assistant", "capacity_600"),
        ("photo_spec_payload", "capacity_100"),
    ]:
        r = next(
            (
                x
                for x in csum
                if x["workload"] == workload
                and x["profile"] == profile
                and x["direction"] == "ul"
            ),
            None,
        )
        if r:
            findings.append(
                f"{workload} UL / {profile}: {r['deadline_misses']} of {r['assessable_units']} assessable units exceeded {r['deadline_ms']} ms"
            )
    capacity_finding = (
        "; ".join(findings) + ". See Controlled_Summary for all conditions."
    )
    highlights = [
        "Burst criteria: strict idle gaps of 5, 10, 20, 50, 100, 200, 500 and 1000 ms; fixed rate windows of 1, 5, 10, 50, 100, 1000 and 10000 ms. Full-unit latency is kept separate from packet-cluster duration.",
        capacity_finding,
        "Provider capture exports contain only measured scenario connections. Cloud host offload prevents valid wire-packet-size claims; controlled captures provide both endpoint measurements.",
        "OpenAI photo input is a six-JPEG proxy derived from public raw photos; its visual assistant uses sampled frames. Google native-video and generation outcomes are listed separately in Coverage, including billing failures and any funded rerun. Archived video evidence is separate.",
        "Observed maxima and small-sample percentiles are evidence for these runs only. No quality-validation claim, no public-cloud one-way latency claim, and no gaming evaluation.",
    ]
    counts = [
        (
            "Fresh provider jobs",
            f"{len(rows)} planned; {fresh_success} succeeded; {failed} failed attempts; {blocked} blocked/not attempted",
        ),
        (
            "Controlled transfers",
            f"{len(controlled)} existing runs across {len(ccov)} workload/profile conditions; {sum(x['succeeded'] for x in ccov)} successful transactions, {sum(x['failed'] for x in ccov)} failures",
        ),
        (
            "Other evidence",
            f"{len(prior)} selected earlier provider runs; archived evidence separately labelled when supplied",
        ),
        (
            "Capture exclusion",
            f"{sum(x['excluded_unrelated_records'] for x in audit):,} unrelated records removed from fresh exports; zero unrelated records in verified scenario PCAPs",
        ),
        ("Generated UTC", datetime.now(timezone.utc).isoformat()),
    ]
    book.finish(highlights, counts)
    manifest = {
        "workbook": str(out),
        "sha256": digest(out),
        "summary_workbook": str(book.summary.path),
        "summary_sha256": digest(book.summary.path),
        "fresh_planned": len(rows),
        "fresh_successes": fresh_success,
        "fresh_failed_attempts": failed,
        "fresh_blocked": blocked,
        "controlled_runs": len(controlled),
        "filtered_capture_unrelated_records": sum(
            x["unrelated_records_in_export"] for x in audit
        ),
        "excluded_unrelated_records": sum(
            x["excluded_unrelated_records"] for x in audit
        ),
        "sheets": book.index,
        "source_sha256": digest(__file__),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sheets"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--previous", required=True)
    p.add_argument("--sources", required=True)
    p.add_argument("--historical")
    p.add_argument("--output", required=True)
    build(p.parse_args())
