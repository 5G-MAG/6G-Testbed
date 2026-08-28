#!/usr/bin/env python3
"""
Generate RESULTS.md from the SQLite database with anonymized identifiers.
"""

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from analysis.anonymization import get_anonymizer


NETWORK_PROFILES = [
    ("ideal_6g", "1ms delay, 0% loss, unlimited BW (baseline)"),
    ("5g_urban", "20ms delay, 0.1% loss, 100Mbit"),
    ("wifi_good", "30ms delay, 0.1% loss, 50Mbit"),
    ("cell_edge", "120ms delay, 1% loss, 5Mbit"),
    ("satellite", "600ms delay, 0.5% loss, 10Mbit"),
    ("congested", "200ms delay, 3% loss, 1Mbit"),
    ("5qi_7", "100ms delay, 0.1% loss (Voice/Live Streaming)"),
    ("5qi_80", "10ms delay, 0.0001% loss (Low-latency eMBB/AR)"),
]


def _load_profile_descriptions(path: str = "configs/profiles.yaml") -> dict:
    """Profile name -> description, with the shaping parameters appended."""
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}
    out = {}
    for name, spec in (cfg.get("profiles") or {}).items():
        if not isinstance(spec, dict):
            continue
        desc = (spec.get("description") or "").strip()
        params = []
        if spec.get("delay_ms") is not None:
            params.append(f"{spec['delay_ms']}ms delay")
        if spec.get("loss_pct") is not None:
            params.append(f"{spec['loss_pct']}% loss")
        if spec.get("rate_mbit"):
            params.append(f"{spec['rate_mbit']}Mbit")
        if params:
            desc = f"{desc} ({', '.join(params)})" if desc else ", ".join(params)
        # Profile descriptions come from user-authored YAML; normalise the
        # em/en dashes some of them contain to plain hyphens for the report.
        out[name] = desc.replace("—", "-").replace("–", "-")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RESULTS.md from database")
    parser.add_argument("--db", default="logs/traffic_logs.db", help="Path to SQLite database")
    parser.add_argument("--since-timestamp", type=float, default=0.0, help="Only include data after this Unix timestamp")
    parser.add_argument("--duration-sec", type=float, default=None, help="Optional duration override in seconds")
    parser.add_argument("--all-runs", action="store_true", help="Include all runs instead of latest per scenario")
    parser.add_argument("--run-gap-sec", type=float, default=300.0, help="Gap in seconds to split runs per scenario")
    parser.add_argument("--output", default="RESULTS.md", help="Output path for RESULTS.md")
    parser.add_argument("--no-findings", action="store_true",
                        help="Skip the FINDINGS_6G.md include (use for reports scoped "
                             "to a single scenario family)")
    args = parser.parse_args()

    db_path = Path(args.db)
    output_path = Path(args.output)
    anonymizer = get_anonymizer()
    since_ts = float(args.since_timestamp or 0.0)

    def load_records(db_file: Path, since_timestamp: float) -> list[dict]:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if since_timestamp:
            cursor.execute(
                "SELECT * FROM traffic_logs WHERE timestamp > ? ORDER BY timestamp",
                (since_timestamp,),
            )
        else:
            cursor.execute("SELECT * FROM traffic_logs ORDER BY timestamp")
        records = [
            dict(row) for row in cursor.fetchall()
            if not (dict(row).get("session_id") or "").startswith("pcap_")
        ]
        conn.close()
        return records

    def fuse_latest_runs(records: list[dict], gap_sec: float) -> list[dict]:
        if not records:
            return []
        by_key: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            scenario = record.get("scenario_id") or "unknown"
            profile = record.get("network_profile") or "unknown"
            by_key[f"{scenario}/{profile}"].append(record)

        fused: list[dict] = []
        for recs in by_key.values():
            recs.sort(key=lambda r: r.get("timestamp", 0.0))
            start_idx = 0
            for i in range(len(recs) - 1, 0, -1):
                if recs[i]["timestamp"] - recs[i - 1]["timestamp"] > gap_sec:
                    start_idx = i
                    break
            fused.extend(recs[start_idx:])
        return fused

    def scenario_provider_map(records: list[dict]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for record in records:
            scenario = record.get("scenario_id") or "unknown"
            provider = record.get("provider")
            if scenario not in mapping and provider:
                mapping[scenario] = provider
        return mapping

    records = load_records(db_path, since_ts)
    if not args.all_runs:
        records = fuse_latest_runs(records, gap_sec=float(args.run_gap_sec))

    provider_map = scenario_provider_map(records)

    def scenario_label(scenario_id: str) -> str:
        base = anonymizer.scenario_alias(scenario_id) or scenario_id
        provider = provider_map.get(scenario_id)
        provider_alias = anonymizer.provider_alias(provider) if provider else ""
        if provider_alias:
            return f"{base} - {provider_alias}"
        return base

    total_records = len(records)
    scenario_ids = sorted({r.get("scenario_id") or "unknown" for r in records})
    total_scenarios = len(scenario_ids)
    total_profiles = len({r.get("network_profile") for r in records if r.get("network_profile")})
    success_count = sum(1 for r in records if r.get("success"))
    success_rate = round(100.0 * success_count / total_records, 1) if total_records else 0.0

    if args.duration_sec is not None:
        duration_sec = int(args.duration_sec)
    else:
        timestamps = [r.get("timestamp", 0.0) for r in records if r.get("timestamp")]
        if timestamps:
            duration_sec = int(max(timestamps) - min(timestamps))
        else:
            duration_sec = 0

    lines = []
    lines.append("# 6G AI Traffic Characterization Testbed - Test Results")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Test Duration: {duration_sec // 60} minutes")
    lines.append("")
    lines.append("## Test Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Records | {total_records} |")
    lines.append(f"| Scenarios Tested | {total_scenarios} |")
    lines.append(f"| Network Profiles | {total_profiles} |")
    lines.append(f"| Success Rate | {success_rate}% |")
    run_selection = "all runs" if args.all_runs else f"latest per scenario (gap > {int(args.run_gap_sec)}s)"
    lines.append(f"| Run Selection | {run_selection} |")
    lines.append("")
    lines.append("## Scenarios Tested")
    lines.append("")
    lines.append("| Scenario | Runs | Success Rate | Avg Latency (s) |")
    lines.append("|----------|------|--------------|-----------------|")

    by_scenario: dict[str, list[dict]] = defaultdict(list)
    by_scenario_profile: dict[tuple[str, str], list[dict]] = defaultdict(list)
    ttft_by_scenario_profile: dict[tuple[str, str], list[float]] = defaultdict(list)

    for record in records:
        scenario_id = record.get("scenario_id") or "unknown"
        profile = record.get("network_profile") or "unknown"
        by_scenario[scenario_id].append(record)
        by_scenario_profile[(scenario_id, profile)].append(record)

        t_start = record.get("t_request_start")
        t_first = record.get("t_first_token")
        if t_start and t_first and t_first > t_start:
            ttft_by_scenario_profile[(scenario_id, profile)].append(t_first - t_start)

    for scenario_id in sorted(by_scenario.keys()):
        recs = by_scenario[scenario_id]
        runs = len(recs)
        success = sum(1 for r in recs if r.get("success"))
        rate = round(100.0 * success / runs, 1) if runs else 0.0
        latencies = [r.get("latency_sec", 0.0) for r in recs if r.get("latency_sec") is not None and r.get("success")]
        avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        label = scenario_label(scenario_id)
        lines.append(f"| {label} | {runs} | {rate} | {avg_lat} |")

    lines.append("")
    lines.append("## Network Profiles Used")
    lines.append("")
    lines.append("| Profile | Description |")
    lines.append("|---------|-------------|")
    # List the profiles this run actually used, described from configs/profiles.yaml.
    # A static list here silently claims profiles that were never exercised.
    used_profiles = sorted({
        r.get("network_profile") for r in records if r.get("network_profile")
    })
    descriptions = _load_profile_descriptions()
    fallback = dict(NETWORK_PROFILES)
    for profile in used_profiles:
        lines.append(f"| {profile} | {descriptions.get(profile) or fallback.get(profile, '')} |")

    lines.append("")
    lines.append("## Detailed Results by Scenario and Profile")
    lines.append("")
    lines.append("| Scenario | Profile | Runs | Success | Avg Latency (s) | Min (s) | Max (s) |")
    lines.append("|----------|---------|------|---------|-----------------|---------|---------|")

    for (scenario_id, profile) in sorted(by_scenario_profile.keys()):
        recs = by_scenario_profile[(scenario_id, profile)]
        runs = len(recs)
        success = sum(1 for r in recs if r.get("success"))
        success_rate = f"{round(100.0 * success / runs, 0)}%" if runs else "0%"
        latencies = [r.get("latency_sec", 0.0) for r in recs if r.get("latency_sec") is not None and r.get("success")]
        avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
        min_lat = round(min(latencies), 3) if latencies else 0.0
        max_lat = round(max(latencies), 3) if latencies else 0.0
        label = scenario_label(scenario_id)
        lines.append(f"| {label} | {profile} | {runs} | {success_rate} | {avg_lat} | {min_lat} | {max_lat} |")

    lines.append("")
    lines.append("## Time to First Token (TTFT)")
    lines.append("")
    lines.append("| Scenario | Profile | Avg TTFT (s) | Min (s) | Max (s) |")
    lines.append("|----------|---------|--------------|---------|---------|")

    for (scenario_id, profile) in sorted(ttft_by_scenario_profile.keys()):
        ttfts = ttft_by_scenario_profile[(scenario_id, profile)]
        if not ttfts:
            continue
        avg_ttft = round(sum(ttfts) / len(ttfts), 3)
        min_ttft = round(min(ttfts), 3)
        max_ttft = round(max(ttfts), 3)
        label = scenario_label(scenario_id)
        lines.append(f"| {label} | {profile} | {avg_ttft} | {min_ttft} | {max_ttft} |")

    lines.append("")
    lines.append("## Bandwidth Usage")
    lines.append("")
    lines.append("| Scenario | Avg Request (bytes) | Avg Response (bytes) | Asymmetry Ratio |")
    lines.append("|----------|---------------------|----------------------|-----------------|")

    for scenario_id in sorted(by_scenario.keys()):
        recs = by_scenario[scenario_id]
        reqs = [r.get("request_bytes", 0) or 0 for r in recs]
        resps = [r.get("response_bytes", 0) or 0 for r in recs]
        avg_req = int(round(sum(reqs) / len(reqs))) if reqs else 0
        avg_resp = int(round(sum(resps) / len(resps))) if resps else 0
        if avg_req > 0:
            ratio = f"{round(avg_resp / avg_req, 0):.0f}:1"
        else:
            ratio = "0:1"
        label = scenario_label(scenario_id)
        lines.append(f"| {label} | {avg_req} | {avg_resp} | {ratio} |")

    # ── Local Inference section ────────────────────────────────────────────
    LOCAL_INFERENCE_SCENARIOS = {"chat_vllm", "video_understanding_vllm"}
    local_scenarios = [s for s in scenario_ids if s in LOCAL_INFERENCE_SCENARIOS]
    if local_scenarios:
        lines.append("")
        lines.append("## Local Inference Scenarios")
        lines.append("")
        lines.append("> **Note on measurement methodology:** The scenarios below run against a "
                     "locally-hosted LLM inference server. Unlike cloud LLM scenarios "
                     "where the measured latency conflates network round-trip time with unknown "
                     "server-side inference time, these local scenarios allow us to **directly "
                     "measure model inference time** because the server runs on the same machine. "
                     "The network impairment applied via tc/netem on the loopback interface "
                     "isolates the transport-layer effect from the compute-layer effect, enabling "
                     "a clean decomposition of end-to-end latency into:")
        lines.append(">")
        lines.append("> - **Network component:** Added by tc/netem (delay, loss, rate limiting)")
        lines.append("> - **Inference component:** Actual GPU compute time for the model")
        lines.append(">")
        lines.append("> For the video understanding scenario, the request payload includes the "
                     "video file base64-encoded inline (~1.3 MB per request), making it "
                     "upload-heavy and sensitive to bandwidth constraints.")
        lines.append("")
        lines.append("| Scenario | Profile | Runs | Avg Latency (s) | Avg TTFT (s) | Avg Request (KB) | Avg Response (KB) | UL/DL Ratio |")
        lines.append("|----------|---------|------|-----------------|--------------|------------------|-------------------|-------------|")

        for scenario_id in local_scenarios:
            for profile_name, _ in NETWORK_PROFILES:
                key = (scenario_id, profile_name)
                recs = by_scenario_profile.get(key, [])
                if not recs:
                    continue
                runs = len(recs)
                latencies = [r.get("latency_sec", 0.0) for r in recs if r.get("latency_sec") is not None and r.get("success")]
                avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
                ttfts = ttft_by_scenario_profile.get(key, [])
                avg_ttft = round(sum(ttfts) / len(ttfts), 3) if ttfts else None
                ttft_str = f"{avg_ttft}" if avg_ttft is not None else "-"
                reqs = [r.get("request_bytes", 0) or 0 for r in recs]
                resps = [r.get("response_bytes", 0) or 0 for r in recs]
                avg_req_kb = round(sum(reqs) / len(reqs) / 1024, 1) if reqs else 0
                avg_resp_kb = round(sum(resps) / len(resps) / 1024, 1) if resps else 0
                ratio = f"{round(sum(reqs) / max(sum(resps), 1), 0):.0f}:1" if sum(resps) > 0 else "-"
                # For video, UL > DL so show UL:DL; for chat, DL > UL
                if sum(reqs) > sum(resps):
                    ratio = f"{round(sum(reqs) / max(sum(resps), 1), 0):.0f}:1 (UL)"
                else:
                    ratio = f"{round(sum(resps) / max(sum(reqs), 1), 0):.0f}:1 (DL)"
                label = scenario_label(scenario_id)
                lines.append(f"| {label} | {profile_name} | {runs} | {avg_lat} | {ttft_str} | {avg_req_kb} | {avg_resp_kb} | {ratio} |")

    # Only report SDP when this run actually contains WebRTC scenarios. Emitting
    # it unconditionally pulled an unrelated offer/answer pair from an older
    # cycle into reports that have nothing to do with WebRTC.
    webrtc_in_run = any(
        "webrtc" in (r.get("scenario_id") or "").lower() for r in records
    )
    if webrtc_in_run:
        lines.append("")
        lines.append("## SDP Offer/Answer Samples (WebRTC)")
        lines.append("")

    preferred_scenario = "realtime_audio_webrtc"
    preferred_label = scenario_label(preferred_scenario)

    sdp_dir = Path("logs/sdp") if webrtc_in_run else Path("logs/sdp/__absent__")
    latest_offer = None
    latest_answer = None
    scenario_sdp_found = False

    def latest_sdp_hash_for_scenario(scenario_id: str) -> str | None:
        candidates = [
            r for r in records
            if r.get("scenario_id") == scenario_id and r.get("metadata")
        ]
        for rec in sorted(candidates, key=lambda r: r.get("timestamp", 0.0), reverse=True):
            try:
                meta = json.loads(rec.get("metadata") or "{}")
            except Exception:
                continue
            if isinstance(meta, dict) and meta.get("sdp_offer_hash"):
                return meta.get("sdp_offer_hash")
        return None

    def find_sdp_pair_by_hash(prefix: str) -> tuple[Path | None, Path | None]:
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

    if sdp_dir.exists():
        scenario_hash = latest_sdp_hash_for_scenario(preferred_scenario)
        if scenario_hash:
            latest_offer, latest_answer = find_sdp_pair_by_hash(scenario_hash[:8])
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
    elif webrtc_in_run:
        lines.append("")
        lines.append("No SDP files found in logs/sdp.")

    # ── RAN2 Methodology Metrics (S4-260859) ──────────────────────────────
    ran2_path = Path("results/reports/figures/ran2_metrics.json")
    ran2 = None
    if ran2_path.exists():
        try:
            ran2 = json.loads(ran2_path.read_text())
        except Exception:
            ran2 = None

    def _anon_pair(key: str) -> str:
        if "/" in key:
            sc, _, pr = key.partition("/")
            return f"{scenario_label(sc)} / {pr}"
        return key

    if ran2:
        lines.append("")
        lines.append("## RAN2 Methodology Metrics (S4-260859)")
        lines.append("")
        lines.append(
            "Metrics computed by `analysis.ran2_metrics.compute_ran2_metrics()` per "
            "S4-260703 / S4-260859 Annex D. Distributions are summarised as "
            "p50 / p95 / p99 / mean (n)."
        )
        lines.append("")
        lines.append(
            f"_Source: `results/reports/figures/ran2_metrics.json` - "
            f"{ran2.get('n_records', 0)} traffic_log records, "
            f"{ran2.get('n_pcap_files', 0)} pcap files analysed._"
        )
        lines.append("")

        # Q1 - UL-heavy traffic
        q1 = (ran2.get("Q1") or {}).get("per_scenario_profile") or {}
        if q1:
            lines.append("### Q1 - UL-heavy traffic (UL/DL byte breakdown)")
            lines.append("")
            lines.append("| Scenario / Profile | Turns | UL Total (B) | DL Total (B) | UL/DL | UL p50 (B) | DL p50 (B) |")
            lines.append("|---|---|---|---|---|---|---|")
            for key in sorted(q1.keys()):
                row = q1[key] or {}
                turns = row.get("turns", 0)
                ulb = row.get("ul_bytes_total", 0)
                dlb = row.get("dl_bytes_total", 0)
                ratio = row.get("ul_dl_ratio")
                ul_dist = row.get("ul_bytes_per_turn") or {}
                dl_dist = row.get("dl_bytes_per_turn") or {}
                ratio_s = f"{ratio:.3f}" if isinstance(ratio, (int, float)) else "-"
                lines.append(
                    f"| {_anon_pair(key)} | {turns} | {ulb} | {dlb} | {ratio_s} | "
                    f"{int(ul_dist.get('p50') or 0)} | {int(dl_dist.get('p50') or 0)} |"
                )
            lines.append("")

        # Q2 - Burst delay metrics (TTFT / TTLT)
        q2 = (ran2.get("Q2") or {}).get("per_scenario_profile_delay") or {}
        if q2:
            lines.append("### Q2 - Burst delay metrics (TTFT / TTLT)")
            lines.append("")
            lines.append("| Scenario / Profile | TTFT n | TTFT p50 (s) | TTFT p95 (s) | TTLT p50 (s) | TTLT p95 (s) |")
            lines.append("|---|---|---|---|---|---|")
            for key in sorted(q2.keys()):
                row = q2[key] or {}
                ttft = row.get("ttft_sec") or {}
                ttlt = row.get("ttlt_sec") or {}
                lines.append(
                    f"| {_anon_pair(key)} | {ttft.get('n', 0)} | "
                    f"{(ttft.get('p50') or 0):.3f} | "
                    f"{(ttft.get('p95') or 0):.3f} | "
                    f"{(ttlt.get('p50') or 0):.3f} | "
                    f"{(ttlt.get('p95') or 0):.3f} |"
                )
            lines.append("")

        # Q3 - RTT components
        q3 = ran2.get("Q3") or {}
        rtt_rows = [
            ("TCP RTT (ms)", q3.get("tcp_rtt") or {}),
            ("TLS handshake (ms)", q3.get("tls_handshake") or {}),
            ("Connection setup, SYN to App-Data (ms)", q3.get("connection_setup_ms") or {}),
        ]
        if any(d for _, d in rtt_rows):
            lines.append("### Q3 - RTT components")
            lines.append("")
            lines.append("| Component | n | p50 | p95 | p99 | mean |")
            lines.append("|---|---|---|---|---|---|")
            for name, d in rtt_rows:
                if not d:
                    continue
                lines.append(
                    f"| {name} | {d.get('n', 0)} | {(d.get('p50') or 0):.2f} | "
                    f"{(d.get('p95') or 0):.2f} | {(d.get('p99') or 0):.2f} | "
                    f"{(d.get('mean') or 0):.2f} |"
                )
            lines.append("")

            ic = q3.get("inter_chunk_vs_rtt") or {}
            ic_dist = ic.get("inter_chunk_sec") or {}
            if ic_dist:
                lines.append(
                    f"**Inter-chunk gap vs RTT:** chunk gap p50 "
                    f"{(ic_dist.get('p50') or 0)*1000:.3f} ms; TCP RTT p50 "
                    f"{(ic.get('tcp_rtt_p50_ms') or 0):.2f} ms; ratio p50 "
                    f"{(ic.get('ratio_p50') or 0):.4f}."
                )
                lines.append("")

            e2e = q3.get("e2e_latency_vs_rtt") or {}
            if e2e:
                lines.append("**End-to-end latency / RTT (per scenario × profile, ms):**")
                lines.append("")
                lines.append("| Scenario / Profile | n | p50 | p95 | p99 |")
                lines.append("|---|---|---|---|---|")
                for key in sorted(e2e.keys()):
                    d = e2e[key] or {}
                    lines.append(
                        f"| {_anon_pair(key)} | {d.get('n', 0)} | {(d.get('p50') or 0):.2f} | "
                        f"{(d.get('p95') or 0):.2f} | {(d.get('p99') or 0):.2f} |"
                    )
                lines.append("")

        # Q4 - Variability and reliability
        q4 = ran2.get("Q4") or {}
        vol = q4.get("volume_distribution") or {}
        rel = q4.get("reliability_by_loss_pct") or {}
        cdur = q4.get("connection_duration") or {}
        agentic = q4.get("agentic_flows") or {}
        if vol or rel or cdur or agentic:
            lines.append("### Q4 - Variability and reliability")
            lines.append("")

            if vol:
                lines.append("**Volume distribution per scenario:**")
                lines.append("")
                lines.append("| Scenario | Req p50 (B) | Req p95 (B) | Resp p50 (B) | Resp p95 (B) | Pkts/flow p50 |")
                lines.append("|---|---|---|---|---|---|")
                for sc in sorted(vol.keys()):
                    v = vol[sc] or {}
                    rb = v.get("request_bytes") or {}
                    rp = v.get("response_bytes") or {}
                    pf = v.get("packet_count_per_flow") or {}
                    lines.append(
                        f"| {scenario_label(sc)} | {int(rb.get('p50') or 0)} | {int(rb.get('p95') or 0)} | "
                        f"{int(rp.get('p50') or 0)} | {int(rp.get('p95') or 0)} | "
                        f"{int(pf.get('p50') or 0)} |"
                    )
                lines.append("")

            if rel:
                buckets: dict[float, dict] = defaultdict(lambda: {"runs": 0, "succ": 0, "scenarios": 0})
                for _, row in rel.items():
                    loss = row.get("profile_loss_pct")
                    bucket = buckets[round(float(loss or 0), 4)]
                    bucket["runs"] += row.get("turns", 0) or 0
                    bucket["succ"] += row.get("success", 0) or 0
                    bucket["scenarios"] += 1
                lines.append("**Reliability by configured loss percentage:**")
                lines.append("")
                lines.append("| Profile Loss % | Cells | Total Turns | Success Rate |")
                lines.append("|---|---|---|---|")
                for loss_key in sorted(buckets.keys()):
                    b = buckets[loss_key]
                    sr = (100.0 * b["succ"] / b["runs"]) if b["runs"] else 0.0
                    lines.append(
                        f"| {loss_key} | {b['scenarios']} | {b['runs']} | {sr:.1f}% |"
                    )
                lines.append("")

            if cdur:
                fd = cdur.get("flow_duration_sec") or {}
                fp = cdur.get("flows_per_pcap") or {}
                cr = cdur.get("connection_reuse_ratio")
                lines.append("**Connection / flow durability:**")
                lines.append("")
                lines.append("| Metric | n | p50 | p95 | p99 | mean |")
                lines.append("|---|---|---|---|---|---|")
                if fd:
                    lines.append(
                        f"| Flow duration (s) | {fd.get('n', 0)} | "
                        f"{(fd.get('p50') or 0):.2f} | {(fd.get('p95') or 0):.2f} | "
                        f"{(fd.get('p99') or 0):.2f} | {(fd.get('mean') or 0):.2f} |"
                    )
                if fp:
                    lines.append(
                        f"| Flows per pcap | {fp.get('n', 0)} | "
                        f"{int(fp.get('p50') or 0)} | {int(fp.get('p95') or 0)} | "
                        f"{int(fp.get('p99') or 0)} | {(fp.get('mean') or 0):.2f} |"
                    )
                if cr is not None:
                    lines.append("")
                    lines.append(
                        f"**Connection reuse ratio:** {cr*100:.2f}% of TCP flows "
                        f"are reused for multiple turns."
                    )
                lines.append("")

            distinct = agentic.get("distinct_dests_per_pcap") or {}
            if distinct:
                lines.append(
                    f"**Distinct destinations per session (agentic flow fan-out):** "
                    f"p50 {int(distinct.get('p50') or 0)}, p95 {int(distinct.get('p95') or 0)}, "
                    f"max {int(distinct.get('max') or 0)} (n={distinct.get('n', 0)} pcaps)."
                )
                lines.append("")

        # Q5 - Tokenized traffic
        q5 = ran2.get("Q5") or {}
        regr = q5.get("token_to_bytes_regression_by_scenario") or {}
        gap = q5.get("inter_token_gap_by_profile") or {}
        tok = q5.get("token_counts_by_scenario") or {}
        arr = q5.get("token_arrival_vs_pkt_arrival") or {}
        if regr or gap or tok or arr:
            lines.append("### Q5 - Tokenized traffic")
            lines.append("")

            if tok:
                lines.append("**Token counts and rate per scenario:**")
                lines.append("")
                lines.append("| Scenario | Tokens In p50 | Tokens Out p50 | Tokens/s p50 | Tokens/s p95 |")
                lines.append("|---|---|---|---|---|")
                for sc in sorted(tok.keys()):
                    t = tok[sc] or {}
                    ti = t.get("tokens_in") or {}
                    to = t.get("tokens_out") or {}
                    tps = t.get("tokens_per_sec") or {}
                    lines.append(
                        f"| {scenario_label(sc)} | {int(ti.get('p50') or 0)} | {int(to.get('p50') or 0)} | "
                        f"{(tps.get('p50') or 0):.1f} | {(tps.get('p95') or 0):.1f} |"
                    )
                lines.append("")

            if regr:
                lines.append("**Bytes-per-token regression (Q5.5 - RAN2 token→byte mapping):**")
                lines.append("")
                lines.append("| Scenario | UL slope (B/tok) | UL intercept (B) | UL r² | DL slope (B/tok) | DL intercept (B) | DL r² |")
                lines.append("|---|---|---|---|---|---|---|")
                for sc in sorted(regr.keys()):
                    r = regr[sc] or {}
                    u = r.get("ul") or {}
                    d = r.get("dl") or {}
                    lines.append(
                        f"| {scenario_label(sc)} | "
                        f"{(u.get('slope') or 0):.2f} | {(u.get('intercept') or 0):.1f} | {(u.get('r2') or 0):.3f} | "
                        f"{(d.get('slope') or 0):.2f} | {(d.get('intercept') or 0):.1f} | {(d.get('r2') or 0):.3f} |"
                    )
                lines.append("")

            if gap:
                lines.append("**Inter-token gap by profile (ms):**")
                lines.append("")
                lines.append("| Profile | n | p50 | p95 | p99 | mean |")
                lines.append("|---|---|---|---|---|---|")
                for prof in sorted(gap.keys()):
                    g = gap[prof] or {}
                    lines.append(
                        f"| {prof} | {g.get('n', 0)} | "
                        f"{(g.get('p50') or 0)*1000:.3f} | "
                        f"{(g.get('p95') or 0)*1000:.3f} | "
                        f"{(g.get('p99') or 0)*1000:.3f} | "
                        f"{(g.get('mean') or 0)*1000:.3f} |"
                    )
                lines.append("")

            tok_rate = arr.get("token_rate_per_profile_hz") or {}
            if tok_rate:
                lines.append("**Token arrival rate per profile (Hz):**")
                lines.append("")
                lines.append("| Profile | Tokens/sec |")
                lines.append("|---|---|")
                for prof in sorted(tok_rate.keys()):
                    lines.append(f"| {prof} | {tok_rate[prof]:.1f} |")
                lines.append("")

    # ── Static analysis include (FINDINGS_6G.md) ──────────────────────────
    # Hand-written findings that must survive regeneration. Maintained in
    # FINDINGS_6G.md at the repo root; injected verbatim before the charts.
    # The findings span several scenario families, so a report scoped to one
    # family should be generated with --no-findings and link to the file
    # instead of embedding another family's analysis.
    findings_path = Path("FINDINGS_6G.md")
    if findings_path.exists() and not args.no_findings:
        lines.append("")
        lines.append(findings_path.read_text().rstrip())

    lines.append("")
    lines.append("## Charts")
    lines.append("")

    # Ordered list of charts with section titles. Charts not found on disk are skipped.
    chart_entries = [
        ("Latency Distribution", "latency_by_scenario.png"),
        ("Time to First Token (TTFT)", "ttft_by_scenario.png"),
        ("Latency Breakdown (TTFT vs Generation)", "latency_breakdown.png"),
        ("Bandwidth Asymmetry (UL/DL)", "bandwidth_asymmetry.png"),
        ("Throughput", "throughput_by_scenario.png"),
        ("Throughput Over Time", "throughput_over_time.png"),
        ("Traffic Burstiness (Per-Request Throughput)", "throughput_burstiness.png"),
        ("Token Counts", "token_counts.png"),
        ("Token Throughput", "token_throughput.png"),
        ("Token Rate by Profile", "token_rate_by_profile.png"),
        ("Success Rate", "success_rate.png"),
        ("Success vs Latency", "success_vs_latency.png"),
        ("Error Analysis", "error_analysis.png"),
        ("Latency by Network Profile", "latency_by_profile.png"),
        ("Latency Heatmap (Scenario × Profile)", "latency_heatmap.png"),
        ("TTFT Heatmap (Scenario × Profile)", "ttft_heatmap.png"),
        ("TTFT vs Latency", "ttft_vs_latency.png"),
        ("Degradation Heatmap", "degradation_heatmap.png"),
        ("Streaming Metrics", "streaming_metrics.png"),
        ("Protocol Comparison", "protocol_comparison.png"),
        ("Data Volume", "data_volume.png"),
        ("Request/Response Scatter", "request_response_scatter.png"),
        ("Context Growth", "context_growth.png"),
        ("Inter-Turn Idle Time", "inter_turn_idle.png"),
        ("Tool Usage", "tool_usage.png"),
        ("Tool Success Rate", "tool_success_rate.png"),
        ("Tool Latency CDF", "tool_latency_cdf.png"),
        ("MCP Efficiency", "mcp_efficiency.png"),
        ("MCP Latency Breakdown", "mcp_latency_breakdown.png"),
        ("MCP Loop Factor by Profile", "mcp_loop_factor_by_profile.png"),
        ("MCP Protocol Overhead", "mcp_protocol_overhead.png"),
        ("Agent Session Waterfall", "agent_session_waterfall.png"),
        ("Pcap RTT Analysis", "pcap_rtt_analysis.png"),
        ("Pcap Throughput", "pcap_throughput.png"),
        ("Pcap Retransmissions", "pcap_retransmissions.png"),
    ]

    from pathlib import Path as _Path
    figures_dir = _Path("results/reports/figures")
    for title, filename in chart_entries:
        if (figures_dir / filename).exists():
            lines.append(f"### {title}")
            lines.append(f"![{title}](results/reports/figures/{filename})")
            lines.append("")

    # RAN2 methodology charts (S4-260859 Annex D)
    ran2_chart_entries = [
        ("RAN2 Q1 - Per-Direction Packets", "ran2_q1_per_direction_packets.png"),
        ("RAN2 Q1 - Multi-Window Throughput", "ran2_q1_multiwindow_throughput.png"),
        ("RAN2 Q2 - Burstiness by Window", "ran2_q2_burstiness_by_window.png"),
        ("RAN2 Q2 - Inter-Burst Idle CDF", "ran2_q2_interburst_idle_cdf.png"),
        ("RAN2 Q3 - RTT Components", "ran2_q3_rtt_components.png"),
        ("RAN2 Q3 - E2E Latency / RTT", "ran2_q3_e2e_latency_over_rtt.png"),
        ("RAN2 Q4 - Reliability vs Loss", "ran2_q4_reliability_vs_loss.png"),
        ("RAN2 Q4 - Flow Duration", "ran2_q4_flow_duration.png"),
        ("RAN2 Q5 - Inter-Token Gap", "ran2_q5_inter_token_gap.png"),
        ("RAN2 Q5 - Tokens→Bytes Regression", "ran2_q5_tokens_to_bytes_regression.png"),
        ("RAN2 Q5 - Token vs Packet Rate", "ran2_q5_token_vs_pkt_rate.png"),
    ]
    ran2_present = [(t, f) for t, f in ran2_chart_entries if (figures_dir / f).exists()]
    if ran2_present:
        lines.append("### RAN2 Methodology Charts (S4-260859)")
        lines.append("")
        for title, filename in ran2_present:
            lines.append(f"#### {title}")
            lines.append(f"![{title}](results/reports/figures/{filename})")
            lines.append("")

    # Include any per-scenario waterfall charts
    waterfalls_dir = figures_dir / "waterfalls"
    if waterfalls_dir.exists():
        waterfall_files = sorted(waterfalls_dir.glob("waterfall_*.png"))
        if waterfall_files:
            lines.append("### Agent Session Waterfalls (Per-Scenario)")
            lines.append("")
            for wf in waterfall_files:
                scenario_name = wf.stem.replace("waterfall_", "").replace("_", " ").title()
                lines.append(f"**{scenario_name}**")
                lines.append(f"![Waterfall {scenario_name}](results/reports/figures/waterfalls/{wf.name})")
                lines.append("")
    lines.append("")
    lines.append("## Data Export")
    lines.append("")
    lines.append("All chart data is available in Excel format: `results/reports/chart_data.xlsx`")
    lines.append("")
    lines.append("Sheets included:")
    lines.append("- Latency_by_Scenario")
    lines.append("- TTFT_by_Scenario")
    lines.append("- Latency_Breakdown")
    lines.append("- Throughput")
    lines.append("- Bandwidth_Asymmetry")
    lines.append("- Success_Rate")
    lines.append("- Streaming_Metrics")
    lines.append("- Token_Counts")
    lines.append("- Latency_by_Profile")
    lines.append("- Latency_Heatmap")
    lines.append("- TTFT_Heatmap")
    lines.append("- Raw_Data")

    output_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
