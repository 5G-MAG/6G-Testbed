#!/usr/bin/env python3
"""
Generate TRACES.md with detailed request/response payload samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import time
from pathlib import Path

import yaml

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from analysis.anonymization import get_anonymizer


def load_scenario_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text()) or {}
    return config.get("scenarios", {})


def load_records(db_path: Path, scenario_id: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM traffic_logs WHERE scenario_id = ? ORDER BY timestamp",
        (scenario_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def pick_session(records: list[dict]) -> tuple[str | None, list[dict]]:
    sessions: dict[str, list[dict]] = {}
    for r in records:
        sessions.setdefault(r.get("session_id", ""), []).append(r)

    if not sessions:
        return None, []

    def session_key(items: list[dict]) -> tuple[int, float]:
        all_success = 1 if all(r.get("success") for r in items) else 0
        max_ts = max(r.get("timestamp", 0.0) for r in items)
        return (all_success, max_ts)

    session_id, recs = max(sessions.items(), key=lambda kv: session_key(kv[1]))
    recs_sorted = sorted(recs, key=lambda r: r.get("turn_index", 0))
    return session_id, recs_sorted


def fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def fmt_num(val, digits=3) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def load_trace_from_metadata(metadata: str | None) -> tuple[str | None, dict | None]:
    if not metadata:
        return None, None
    try:
        meta = json.loads(metadata)
    except Exception:
        return None, None
    if not isinstance(meta, dict):
        return None, None
    trace_path = meta.get("trace_path")
    if not trace_path:
        return None, None
    path = Path(trace_path)
    if not path.exists():
        return trace_path, None
    try:
        return trace_path, json.loads(path.read_text())
    except Exception:
        return trace_path, None


def latest_sdp_hash_for_scenario(records: list[dict]) -> str | None:
    candidates = [r for r in records if r.get("metadata")]
    for rec in sorted(candidates, key=lambda r: r.get("timestamp", 0.0), reverse=True):
        try:
            meta = json.loads(rec.get("metadata") or "{}")
        except Exception:
            continue
        if isinstance(meta, dict) and meta.get("sdp_offer_hash"):
            return meta.get("sdp_offer_hash")
    return None


def find_sdp_pair_by_hash(sdp_dir: Path, prefix: str) -> tuple[Path | None, Path | None]:
    offers = sorted(
        sdp_dir.glob(f"*_{prefix}_offer.sdp"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for offer_path in offers:
        answer_path = Path(str(offer_path).replace("_offer.sdp", "_answer.sdp"))
        if answer_path.exists():
            return offer_path, answer_path
    return None, None


def build_throughput_data(
    records: list[dict],
    scenario_id: str,
    session_id: str,
    network_profile: str,
) -> dict:
    bucket_sec = 1
    buckets: dict[int, dict[str, float]] = {}
    session_start = None
    session_end = None

    for rec in records:
        start = rec.get("t_request_start") or rec.get("timestamp")
        if not start:
            continue
        latency = rec.get("latency_sec") or 0.0
        end = start + latency if latency and latency > 0 else start
        if session_start is None or start < session_start:
            session_start = start
        if session_end is None or end > session_end:
            session_end = end

        req_bytes = rec.get("request_bytes") or 0
        resp_bytes = rec.get("response_bytes") or 0

        sec = int(start)
        entry = buckets.setdefault(sec, {"req_bytes": 0.0, "resp_bytes": 0.0})
        entry["req_bytes"] += req_bytes

        trace_used = False
        trace_path, trace_data = load_trace_from_metadata(rec.get("metadata"))
        response_events = trace_data.get("response_events") if trace_data else []
        for event in response_events:
            ts = event.get("timestamp")
            size = event.get("bytes")
            if not ts or size is None:
                continue
            event_sec = int(ts)
            entry = buckets.setdefault(event_sec, {"req_bytes": 0.0, "resp_bytes": 0.0})
            entry["resp_bytes"] += float(size)
            trace_used = True
            if session_end is None or ts > session_end:
                session_end = ts

        if not trace_used:
            if latency and latency > 0:
                start_sec = int(start)
                end_sec = int(end)
                for sec in range(start_sec, end_sec + 1):
                    bucket_start = sec
                    bucket_end = sec + bucket_sec
                    overlap = min(end, bucket_end) - max(start, bucket_start)
                    if overlap <= 0:
                        continue
                    fraction = overlap / latency
                    entry = buckets.setdefault(sec, {"req_bytes": 0.0, "resp_bytes": 0.0})
                    entry["resp_bytes"] += resp_bytes * fraction
            else:
                entry = buckets.setdefault(sec, {"req_bytes": 0.0, "resp_bytes": 0.0})
                entry["resp_bytes"] += resp_bytes

    seconds = []
    ul_values = []
    dl_values = []
    seconds_with_data = 0
    if buckets:
        min_sec = min(buckets)
        max_sec = max(buckets)
        for sec in range(min_sec, max_sec + 1):
            entry = buckets.get(sec, {"req_bytes": 0.0, "resp_bytes": 0.0})
            req_bytes = entry["req_bytes"]
            resp_bytes = entry["resp_bytes"]
            ul_kbps = (req_bytes * 8) / bucket_sec / 1000
            dl_kbps = (resp_bytes * 8) / bucket_sec / 1000
            if req_bytes > 0 or resp_bytes > 0:
                seconds_with_data += 1
            ul_values.append(ul_kbps)
            dl_values.append(dl_kbps)
            seconds.append({
                "second": sec - min_sec,
                "timestamp": float(sec),
                "offset_sec": sec - min_sec,
                "request_bytes": req_bytes,
                "response_bytes": resp_bytes,
                "ul_kbps": ul_kbps,
                "dl_kbps": dl_kbps,
            })

    summary = {
        "seconds_total": len(seconds),
        "seconds_with_data": seconds_with_data,
        "avg_ul_kbps": statistics.mean(ul_values) if ul_values else None,
        "avg_dl_kbps": statistics.mean(dl_values) if dl_values else None,
        "min_ul_kbps": min(ul_values) if ul_values else None,
        "max_ul_kbps": max(ul_values) if ul_values else None,
        "min_dl_kbps": min(dl_values) if dl_values else None,
        "max_dl_kbps": max(dl_values) if dl_values else None,
    }

    return {
        "schema_version": "1.1",
        "scenario_id": scenario_id,
        "session_id": session_id,
        "network_profile": network_profile,
        "unit": "Kbps",
        "bucket_sec": bucket_sec,
        "session_start": session_start,
        "session_end": session_end,
        "summary": summary,
        "seconds": seconds,
    }


def write_throughput_file(
    throughput_data: dict,
    output_dir: Path,
    scenario_id: str,
    session_id: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    session_tag = session_id.replace("-", "")[:8]
    scenario_tag = scenario_id.replace(" ", "_")
    path = output_dir / f"throughput_{scenario_tag}_{session_tag}.json"
    if not path.exists():
        path.write_text(json.dumps(throughput_data, indent=2, ensure_ascii=True, default=str))
    return str(path)


def write_throughput_plot(
    throughput_data: dict,
    output_dir: Path,
    scenario_id: str,
    session_id: str,
) -> str | None:
    if not HAS_MATPLOTLIB:
        return None

    seconds = throughput_data.get("seconds", [])
    if not seconds:
        return None

    x_vals = []
    ul_vals = []
    dl_vals = []
    for sec in seconds:
        offset = sec.get("offset_sec")
        if offset is None:
            offset = float(sec.get("second") or 0)
        ul = sec.get("ul_kbps")
        dl = sec.get("dl_kbps")
        if ul is None and dl is None:
            continue
        x_vals.append(offset)
        ul_vals.append(ul)
        dl_vals.append(dl)

    if not x_vals:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    session_tag = session_id.replace("-", "")[:8]
    scenario_tag = scenario_id.replace(" ", "_")
    path = output_dir / f"throughput_{scenario_tag}_{session_tag}.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_vals, ul_vals, marker="o", linewidth=1.2, label="UL (Kbps)")
    ax.plot(x_vals, dl_vals, marker="o", linewidth=1.2, label="DL (Kbps)")
    ax.set_xlabel("Time since session start (s)")
    ax.set_ylabel("Throughput (Kbps)")
    ax.set_title("Per-Second Throughput")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TRACES.md with request/response payloads")
    parser.add_argument("--db", default="logs/traffic_logs.db", help="Path to SQLite database")
    parser.add_argument("--config", default="configs/scenarios.yaml", help="Scenario config file")
    parser.add_argument("--output", default="TRACES.md", help="Output markdown path")
    parser.add_argument(
        "--scenarios",
        default="chat_basic,chat_streaming",
        help="Comma-separated scenario ids",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=20,
        help="Max streaming events to include per sample (0 to skip)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    config_path = Path(args.config)
    output_path = Path(args.output)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    anonymizer = get_anonymizer()
    scenarios_config = load_scenario_config(config_path)
    trace_dir = Path(os.environ.get("TRACE_LOG_DIR", "logs/traces"))
    trace_fig_dir = trace_dir / "figures"

    lines: list[str] = []
    lines.append("# TRACES")
    lines.append("")
    lines.append("Run-level traces and sample request/response payloads.")
    lines.append("")
    lines.append("## Selection Notes")
    lines.append("")
    lines.append(f"- Data source: `{db_path}`")
    lines.append("- Selection: latest fully successful session per scenario (fallback to latest session)")
    lines.append("- Trace files are generated only when `TRACE_PAYLOADS=1` is set during runs.")
    lines.append("")

    lines.append("## SDP Offer/Answer Samples (WebRTC)")
    lines.append("")
    preferred_scenario = "realtime_audio_webrtc"
    preferred_label = anonymizer.scenario_alias(preferred_scenario) or preferred_scenario
    sdp_dir = Path("logs/sdp")
    latest_offer = None
    latest_answer = None
    scenario_sdp_found = False

    if sdp_dir.exists():
        scenario_records = load_records(db_path, preferred_scenario)
        scenario_hash = latest_sdp_hash_for_scenario(scenario_records)
        if scenario_hash:
            latest_offer, latest_answer = find_sdp_pair_by_hash(sdp_dir, scenario_hash[:8])
            scenario_sdp_found = latest_offer is not None

        if latest_offer is None:
            offers = sorted(sdp_dir.glob("*_offer.sdp"), key=lambda p: p.stat().st_mtime, reverse=True)
            for offer_path in offers:
                answer_path = Path(str(offer_path).replace("_offer.sdp", "_answer.sdp"))
                if not answer_path.exists():
                    continue
                try:
                    offer_text = offer_path.read_text()
                    answer_text = answer_path.read_text()
                except Exception:
                    offer_text = ""
                    answer_text = ""
                if "m=audio" in offer_text and "m=audio" in answer_text:
                    latest_offer = offer_path
                    latest_answer = answer_path
                    break
            if latest_offer is None and offers:
                latest_offer = offers[0]
                latest_answer = Path(str(latest_offer).replace("_offer.sdp", "_answer.sdp"))

    if latest_offer:
        if scenario_sdp_found:
            lines.append(
                f"Latest offer/answer pair captured during realtime WebRTC sessions for "
                f"{preferred_label} from `logs/sdp/`."
            )
        else:
            lines.append(
                f"No SDP samples found for {preferred_label}; showing the latest "
                f"available WebRTC offer/answer with audio from `logs/sdp/`."
            )

        offer_file = latest_offer.name
        answer_file = latest_answer.name if latest_answer else latest_offer.name.replace("_offer.sdp", "_answer.sdp")
        offer_bytes = latest_offer.stat().st_size if latest_offer.exists() else 0
        answer_bytes = latest_answer.stat().st_size if latest_answer and latest_answer.exists() else 0

        lines.append("")
        lines.append(f"**Offer:** `logs/sdp/{offer_file}` ({offer_bytes} bytes)")
        lines.append("```sdp")
        try:
            lines.append(latest_offer.read_text().rstrip())
        except Exception:
            lines.append("(failed to read offer file)")
        lines.append("```")

        lines.append("")
        lines.append(f"**Answer:** `logs/sdp/{answer_file}` ({answer_bytes} bytes)")
        lines.append("```sdp")
        if latest_answer and latest_answer.exists():
            try:
                lines.append(latest_answer.read_text().rstrip())
            except Exception:
                lines.append("(failed to read answer file)")
        else:
            lines.append("(missing answer file)")
        lines.append("```")
    else:
        lines.append("No SDP files found in logs/sdp.")

    lines.append("")

    for scenario_id in scenarios:
        records = load_records(db_path, scenario_id)
        session_id, session_records = pick_session(records)

        scenario_label = anonymizer.scenario_alias(scenario_id) or scenario_id
        provider = session_records[0].get("provider") if session_records else None
        model = session_records[0].get("model") if session_records else None
        provider_alias = anonymizer.provider_alias(provider) if provider else ""
        model_alias = anonymizer.model_alias(model) if model else ""
        display_label = scenario_label
        if provider_alias:
            display_label = f"{display_label} - {provider_alias}"

        lines.append(f"## {display_label}")
        lines.append("")

        if not session_records:
            lines.append("No records found for this scenario.")
            lines.append("")
            continue

        min_ts = min(r.get("timestamp", 0.0) for r in session_records)
        max_ts = max(r.get("timestamp", 0.0) for r in session_records)
        success_rate = sum(1 for r in session_records if r.get("success")) / len(session_records) * 100.0
        network_profile = session_records[0].get("network_profile", "-")
        run_index = session_records[0].get("run_index", 0)
        is_streaming = bool(session_records[0].get("is_streaming"))

        lines.append("### Run Metadata")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| Scenario ID | `{scenario_id}` |")
        lines.append(f"| Session ID | `{session_id}` |")
        lines.append(f"| Provider | {provider_alias or provider or '-'} |")
        lines.append(f"| Model | {model_alias or model or '-'} |")
        lines.append(f"| Network Profile | `{network_profile}` |")
        lines.append(f"| Run Index | {run_index} |")
        lines.append(f"| Start | {fmt_ts(min_ts)} |")
        lines.append(f"| End | {fmt_ts(max_ts)} |")
        lines.append(f"| Turns | {len(session_records)} |")
        lines.append(f"| Streaming | {str(is_streaming).lower()} |")
        lines.append(f"| Success Rate | {success_rate:.1f}% |")
        lines.append("")

        prompts = (scenarios_config.get(scenario_id) or {}).get("prompts", [])
        system_prompt = (scenarios_config.get(scenario_id) or {}).get("system_prompt")
        lines.append("### Prompt Set")
        lines.append("")
        if system_prompt:
            lines.append(f"- System prompt: {system_prompt}")
        if prompts:
            for idx, prompt in enumerate(prompts, 1):
                lines.append(f"- Prompt {idx}: {prompt}")
        else:
            lines.append("- Prompts not found in configs.")
        lines.append("")

        sample_record = min(session_records, key=lambda r: r.get("turn_index", 0))
        trace_path, trace_data = load_trace_from_metadata(sample_record.get("metadata"))
        if trace_data:
            lines.append("### Sample Request (exact payload)")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(trace_data.get("request"), indent=2, ensure_ascii=True, default=str))
            lines.append("```")
            lines.append("")
            lines.append("### Sample Response (exact payload)")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(trace_data.get("response"), indent=2, ensure_ascii=True, default=str))
            lines.append("```")
            lines.append("")
            response_events = trace_data.get("response_events") or []
            if response_events and args.max_events != 0:
                sample_events = response_events[: args.max_events] if args.max_events > 0 else []
                lines.append("### Sample Response Events (streaming)")
                lines.append("")
                lines.append(f"- Showing {len(sample_events)} of {len(response_events)} events")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(sample_events, indent=2, ensure_ascii=True, default=str))
                lines.append("```")
                lines.append("")
        else:
            lines.append("### Sample Request/Response")
            lines.append("")
            if trace_path:
                lines.append(f"- Trace file not readable: `{trace_path}`")
            else:
                lines.append("- No trace file recorded for this session.")
            lines.append("")

        throughput_data = build_throughput_data(
            session_records,
            scenario_id=scenario_id,
            session_id=session_id,
            network_profile=network_profile,
        )
        throughput_path = write_throughput_file(
            throughput_data,
            trace_dir,
            scenario_id=scenario_id,
            session_id=session_id,
        )
        throughput_plot = write_throughput_plot(
            throughput_data,
            trace_fig_dir,
            scenario_id=scenario_id,
            session_id=session_id,
        )
        sample_seconds = throughput_data.get("seconds", [])[:5]
        lines.append("### Throughput Sample (per second)")
        lines.append("")
        lines.append(f"- Throughput file: `{throughput_path}`")
        if throughput_plot:
            lines.append(f"- Throughput plot: `{throughput_plot}`")
        lines.append(f"- Unit: {throughput_data.get('unit', 'Kbps')}")
        if sample_seconds:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(sample_seconds, indent=2, ensure_ascii=True, default=str))
            lines.append("```")
        lines.append("")
        if throughput_plot:
            lines.append("### Throughput Plot (per run)")
            lines.append("")
            lines.append(f"![Per-Second Throughput]({throughput_plot})")
            lines.append("")

        lines.append("### Turn Trace")
        lines.append("")
        lines.append("| Turn | Success | Latency (s) | Tokens In | Tokens Out | Request Bytes | Response Bytes | TTFT (s) | TTLT (s) | Chunks | Trace File |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for rec in session_records:
            t_first = rec.get("t_first_token")
            t_last = rec.get("t_last_token")
            t_req = rec.get("t_request_start")
            ttft = t_first - t_req if t_first and t_req else None
            ttlt = t_last - t_req if t_last and t_req else None
            trace_path, _ = load_trace_from_metadata(rec.get("metadata"))
            trace_cell = f"`{trace_path}`" if trace_path else "-"
            lines.append(
                "| {turn} | {success} | {lat} | {tin} | {tout} | {req} | {resp} | {ttft} | {ttlt} | {chunks} | {trace} |".format(
                    turn=rec.get("turn_index", 0),
                    success="yes" if rec.get("success") else "no",
                    lat=fmt_num(rec.get("latency_sec")),
                    tin=rec.get("tokens_in") if rec.get("tokens_in") is not None else "-",
                    tout=rec.get("tokens_out") if rec.get("tokens_out") is not None else "-",
                    req=rec.get("request_bytes") or 0,
                    resp=rec.get("response_bytes") or 0,
                    ttft=fmt_num(ttft),
                    ttlt=fmt_num(ttlt),
                    chunks=rec.get("chunk_count") if rec.get("chunk_count") is not None else "-",
                    trace=trace_cell,
                )
            )
        lines.append("")

    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
