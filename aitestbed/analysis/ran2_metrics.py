"""
RAN2 methodology metrics (S4-260859 Annex D).

This module computes the metrics SA4 listed in response to RAN2's four
working assumptions on 6G AI traffic characteristics plus the tokenized-
traffic topic. Each metric family is organized under the corresponding
RAN2 question (Q1..Q5). The output is a nested dict that downstream
report/chart generators read verbatim.

Inputs:
    records:        list[dict] from traffic_logs (one per turn)
    pcap_metrics:   optional list[PcapMetrics] from analysis.pcap_analyzer
    profiles_yaml:  optional path to configs/profiles.yaml (for loss_pct lookup in Q4.4)

Outputs (top-level dict shape):
    {
      "Q1": {"per_scenario": {...}, "per_scenario_profile": {...}, ...},
      "Q2": {...},
      "Q3": {...},
      "Q4": {...},
      "Q5": {...},
      "generated_at": <epoch>,
    }
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)


def _distribution(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "n": len(values),
        "min": min(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def _cv(values: list[float]) -> Optional[float]:
    """Coefficient of variation — stdev / mean. Returns None on < 2 samples
    or zero mean."""
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    return statistics.stdev(values) / mean


def _metadata(record: dict) -> dict:
    raw = record.get("metadata") or ""
    if not raw:
        return {}
    try:
        meta = json.loads(raw)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _is_primary_turn(record: dict) -> bool:
    """Exclude pcap/tool/computer_action/capture rows; keep real LLM turns only."""
    session_id = record.get("session_id") or ""
    if session_id.startswith("pcap_") or session_id.startswith("timeout_"):
        return False
    if record.get("turn_index") is not None and record.get("turn_index") < 0:
        return False
    meta = _metadata(record)
    if meta.get("record_type") in ("tool_call", "computer_action", "pcap_capture"):
        return False
    return True


def _load_profile_loss_pct(profiles_yaml: Optional[str]) -> dict[str, float]:
    """Map profile name -> nominal loss_pct from configs/profiles.yaml.
    Returns empty dict on any failure."""
    if not profiles_yaml:
        profiles_yaml = "configs/profiles.yaml"
    try:
        import yaml
        data = yaml.safe_load(Path(profiles_yaml).read_text())
        out = {}
        for name, cfg in ((data or {}).get("profiles") or {}).items():
            if isinstance(cfg, dict) and "loss_pct" in cfg:
                out[name] = float(cfg["loss_pct"])
        return out
    except Exception:
        return {}


def _pcap_for_scenario_profile(
    pcap_metrics: Iterable,
    scenario: str,
    profile: str,
) -> list:
    """Filter pcap metrics to the (scenario, profile) pair. Uses `pcap_file`
    name convention (orchestrator writes `capture_<iface>_<timestamp>.pcap`
    inside capture dirs; the association is by run grouping in the DB
    `interface` + `run_id` metadata). Without tight binding we return all
    pcap_metrics when the caller doesn't pre-filter; downstream code tolerates."""
    return list(pcap_metrics)


# ---------------------------------------------------------------------------
# Q1 — UL-heavy
# ---------------------------------------------------------------------------

def _q1_ul_heavy(records: list[dict], pcap_metrics: list) -> dict:
    """Q1: UL/DL volumes, ratios, per-direction packet counts/sizes.
    Multi-window per-direction throughput comes from pcap_metrics."""
    out = {"per_scenario_profile": {}, "aggregate": {}}

    # Byte-level per (scenario, profile), from DB
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        if not _is_primary_turn(r):
            continue
        if not r.get("success"):
            continue
        by_key[(r.get("scenario_id") or "?", r.get("network_profile") or "?")].append(r)

    for (scenario, profile), recs in by_key.items():
        reqs = [r.get("request_bytes") or 0 for r in recs]
        resps = [r.get("response_bytes") or 0 for r in recs]
        total_ul = sum(reqs)
        total_dl = sum(resps)
        row = {
            "turns": len(recs),
            "ul_bytes_total": total_ul,
            "dl_bytes_total": total_dl,
            "ul_bytes_per_turn": _distribution(reqs),
            "dl_bytes_per_turn": _distribution(resps),
            "ul_dl_ratio": (total_ul / total_dl) if total_dl else None,
        }
        out["per_scenario_profile"][f"{scenario}/{profile}"] = row

    # Aggregate pcap-derived: per-direction packet counts + sizes +
    # per-window peak throughput (Q1.3, Q1.4).
    pcap_rows = []
    for m in pcap_metrics:
        pcap_rows.append({
            "pcap_file": getattr(m, "pcap_file", ""),
            "ul_packets": getattr(m, "ul_packets", 0),
            "dl_packets": getattr(m, "dl_packets", 0),
            "ul_mean_pkt_size": getattr(m, "ul_mean_pkt_size", None),
            "dl_mean_pkt_size": getattr(m, "dl_mean_pkt_size", None),
            "ul_bytes_total": getattr(m, "ul_bytes_total", 0),
            "dl_bytes_total": getattr(m, "dl_bytes_total", 0),
            "peak_mbps_by_window": dict(getattr(m, "peak_mbps_by_window", {}) or {}),
        })
    out["pcap_per_direction"] = pcap_rows
    return out


# ---------------------------------------------------------------------------
# Q2 — Bursts & delay-bound
# ---------------------------------------------------------------------------

def _q2_bursts(records: list[dict], pcap_metrics: list) -> dict:
    """Q2: per-direction bursts at 10/100ms gap; burstiness per window;
    TTFB/TTLB (already supported)."""
    out = {"per_pcap": [], "per_scenario_profile_delay": {}}

    for m in pcap_metrics:
        bursts_by_gap = dict(getattr(m, "bursts_by_gap", {}) or {})
        idle_by_gap = dict(getattr(m, "interburst_idle_by_gap", {}) or {})
        entry: dict[str, Any] = {
            "pcap_file": getattr(m, "pcap_file", ""),
            "burstiness_by_window": dict(getattr(m, "burstiness_by_window", {}) or {}),
            "burst_stats_by_gap": {},
            "interburst_idle_by_gap": {},
        }
        for label, per_dir in bursts_by_gap.items():
            entry["burst_stats_by_gap"][label] = {}
            for direction, bursts in (per_dir or {}).items():
                sizes = [b["total_bytes"] for b in bursts]
                durs = [b["duration_sec"] for b in bursts]
                peaks = [b["peak_rate_bps"] / 1_000_000 for b in bursts]  # Mbps
                entry["burst_stats_by_gap"][label][direction] = {
                    "count": len(bursts),
                    "size_bytes": _distribution(sizes),
                    "duration_sec": _distribution(durs),
                    "peak_rate_mbps": _distribution(peaks),
                }
        for label, per_dir in idle_by_gap.items():
            entry["interburst_idle_by_gap"][label] = {}
            for direction, gaps in (per_dir or {}).items():
                entry["interburst_idle_by_gap"][label][direction] = {
                    "cdf_sec": _distribution(gaps),
                    "cv": _cv(gaps),
                }
        out["per_pcap"].append(entry)

    # TTFB/TTLB already present in per-record fields — re-emit per (scenario, profile)
    by_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"ttft": [], "ttlt": []}
    )
    for r in records:
        if not _is_primary_turn(r) or not r.get("success"):
            continue
        key = (r.get("scenario_id") or "?", r.get("network_profile") or "?")
        if r.get("t_request_start") and r.get("t_first_token"):
            ttft = r["t_first_token"] - r["t_request_start"]
            if ttft >= 0:
                by_key[key]["ttft"].append(ttft)
        if r.get("t_request_start") and r.get("t_last_token"):
            ttlt = r["t_last_token"] - r["t_request_start"]
            if ttlt >= 0:
                by_key[key]["ttlt"].append(ttlt)
    for (s, p), d in by_key.items():
        out["per_scenario_profile_delay"][f"{s}/{p}"] = {
            "ttft_sec": _distribution(d["ttft"]),
            "ttlt_sec": _distribution(d["ttlt"]),
        }
    return out


# ---------------------------------------------------------------------------
# Q3 — Round-trip delay
# ---------------------------------------------------------------------------

def _q3_rtt(records: list[dict], pcap_metrics: list) -> dict:
    """Q3: TCP RTT (supported), TLS handshake + HTTP setup RTT (partial),
    inter-chunk gap vs RTT (new), E2E latency vs RTT (partial)."""
    out = {"tcp_rtt": {}, "tls_handshake": {}, "http_setup_rtt": {},
           "inter_chunk_vs_rtt": {}, "e2e_latency_vs_rtt": {}}

    # TCP handshake RTT + HTTP setup RTT from pcap flows
    flow_rtts_ms: list[float] = []
    http_setup_ms: list[float] = []
    for m in pcap_metrics:
        for flow in getattr(m, "flows", []) or []:
            if flow.handshake_rtt:
                flow_rtts_ms.append(flow.handshake_rtt * 1000.0)
            # HTTP setup RTT = time_to_first_data - handshake_duration
            t2fd = flow.time_to_first_data
            hs_dur = flow.handshake_duration
            if t2fd is not None and hs_dur is not None:
                http_setup_ms.append(max(0.0, (t2fd - hs_dur) * 1000.0))
    out["tcp_rtt"] = _distribution(flow_rtts_ms)
    out["http_setup_rtt"] = _distribution(http_setup_ms)

    # TLS handshake time — stored in per-record metadata.tls when available
    tls_ms: list[float] = []
    for r in records:
        if not _is_primary_turn(r):
            continue
        meta = _metadata(r)
        tls = meta.get("tls") or {}
        t = tls.get("handshake_ms") or tls.get("handshake_sec")
        if isinstance(t, (int, float)):
            tls_ms.append(float(t) * (1000.0 if t < 1.0 else 1.0))
    out["tls_handshake"] = _distribution(tls_ms)

    # Inter-chunk gap vs RTT for streaming turns
    rtt_p50_ms = out["tcp_rtt"].get("p50") or 0.0
    inter_chunk_gaps_sec: list[float] = []
    for r in records:
        if not _is_primary_turn(r) or not r.get("is_streaming"):
            continue
        raw = r.get("inter_chunk_times")
        try:
            times = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            times = []
        for t in times:
            if isinstance(t, (int, float)) and t >= 0:
                inter_chunk_gaps_sec.append(float(t))
    out["inter_chunk_vs_rtt"] = {
        "inter_chunk_sec": _distribution(inter_chunk_gaps_sec),
        "tcp_rtt_p50_ms": rtt_p50_ms,
        "ratio_p50": (
            (statistics.median(inter_chunk_gaps_sec) * 1000.0 / rtt_p50_ms)
            if inter_chunk_gaps_sec and rtt_p50_ms else None
        ),
    }

    # E2E latency vs RTT for non-streaming turns, per scenario/profile
    per_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        if not _is_primary_turn(r) or not r.get("success"):
            continue
        if r.get("is_streaming"):
            continue
        lat = r.get("latency_sec")
        if isinstance(lat, (int, float)) and lat >= 0 and rtt_p50_ms > 0:
            per_key[(r.get("scenario_id") or "?", r.get("network_profile") or "?")].append(
                (lat * 1000.0) / rtt_p50_ms
            )
    for (s, p), ratios in per_key.items():
        out["e2e_latency_vs_rtt"][f"{s}/{p}"] = _distribution(ratios)
    return out


# ---------------------------------------------------------------------------
# Q4 — Intra-application variability
# ---------------------------------------------------------------------------

def _q4_variability(
    records: list[dict],
    pcap_metrics: list,
    profile_loss_pct: dict[str, float],
) -> dict:
    """Q4: volume/packet-count distributions, per-burst distributions (Q2 reuse),
    reliability vs loss, inter-burst idle CV, flow duration, connection reuse,
    distinct destinations, per-tool sub-flow volumes."""
    out = {
        "volume_distribution": {},
        "reliability_by_loss_pct": {},
        "inter_arrival_cv": {},
        "connection_duration": {},
        "agentic_flows": {},
    }

    # Volume & packet-count distributions per scenario
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if not _is_primary_turn(r):
            continue
        by_scenario[r.get("scenario_id") or "?"].append(r)

    for scenario, recs in by_scenario.items():
        req_bytes = [r.get("request_bytes") or 0 for r in recs if r.get("success")]
        resp_bytes = [r.get("response_bytes") or 0 for r in recs if r.get("success")]
        # Packet counts per turn are only available via a pcap<->session join.
        # We approximate as: for each pcap file, per-flow packets_sent+packets_recv
        # counted as a single "turn", aggregated across all pcaps for this scenario.
        # This is coarse but avoids requiring exact time-window joins here.
        pkt_counts: list[int] = []
        for m in pcap_metrics:
            for flow in getattr(m, "flows", []) or []:
                total = (getattr(flow, "packets_sent", 0) or 0) + (getattr(flow, "packets_recv", 0) or 0)
                if total > 0:
                    pkt_counts.append(total)
        out["volume_distribution"][scenario] = {
            "request_bytes": _distribution(req_bytes),
            "response_bytes": _distribution(resp_bytes),
            "packet_count_per_flow": _distribution(pkt_counts),
        }

    # Reliability vs loss_pct — success rate per (scenario, profile) + profile loss_pct
    by_sp: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        if not _is_primary_turn(r):
            continue
        by_sp[(r.get("scenario_id") or "?", r.get("network_profile") or "?")].append(r)
    for (scenario, profile), recs in by_sp.items():
        n = len(recs)
        ok = sum(1 for r in recs if r.get("success"))
        out["reliability_by_loss_pct"][f"{scenario}/{profile}"] = {
            "turns": n,
            "success": ok,
            "success_rate": (ok / n) if n else None,
            "profile_loss_pct": profile_loss_pct.get(profile),
        }

    # Inter-arrival CV (inter-burst idle time) — take from pcap per direction
    for m in pcap_metrics:
        name = getattr(m, "pcap_file", "")
        entry: dict[str, dict] = {}
        for label, per_dir in (getattr(m, "interburst_idle_by_gap", {}) or {}).items():
            entry[label] = {}
            for direction, gaps in (per_dir or {}).items():
                entry[label][direction] = _cv(gaps)
        out["inter_arrival_cv"][name] = entry

    # Connection duration + reuse ratio (Q4.6) + agentic sub-flows (Q4.7)
    all_flow_durations: list[float] = []
    flow_keys_seen: set = set()
    reuse_hits = 0
    reuse_total = 0
    distinct_dests_per_pcap: list[int] = []
    flows_per_pcap: list[int] = []
    for m in pcap_metrics:
        pcap_flow_keys: set = set()
        pcap_dests: set = set()
        for flow in getattr(m, "flows", []) or []:
            dur = getattr(flow, "duration", 0.0)
            if dur > 0:
                all_flow_durations.append(dur)
            fk = getattr(flow, "flow_key", "")
            if fk:
                reuse_total += 1
                if fk in flow_keys_seen:
                    reuse_hits += 1
                flow_keys_seen.add(fk)
                pcap_flow_keys.add(fk)
            dst_ip = getattr(flow, "dst_ip", "")
            dst_port = getattr(flow, "dst_port", 0)
            if dst_ip:
                pcap_dests.add((dst_ip, dst_port))
        flows_per_pcap.append(len(pcap_flow_keys))
        distinct_dests_per_pcap.append(len(pcap_dests))

    out["connection_duration"] = {
        "flow_duration_sec": _distribution(all_flow_durations),
        "flows_per_pcap": _distribution([float(x) for x in flows_per_pcap]),
        "connection_reuse_ratio": (reuse_hits / reuse_total) if reuse_total else None,
    }
    out["agentic_flows"] = {
        "distinct_dests_per_pcap": _distribution([float(x) for x in distinct_dests_per_pcap]),
        "per_tool_bytes": _per_tool_bytes(records),  # from DB metadata, no pcap join needed
    }
    return out


def _per_tool_bytes(records: list[dict]) -> dict[str, dict]:
    """Aggregate request/response bytes per MCP tool name, from
    tool-call records in the DB (metadata.record_type == 'tool_call')."""
    tool_bytes: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "request_bytes": 0, "response_bytes": 0, "tool_latency_sec": 0.0}
    )
    for r in records:
        meta = _metadata(r)
        if meta.get("record_type") != "tool_call":
            continue
        tool = meta.get("tool_name") or meta.get("tool") or "<unknown>"
        entry = tool_bytes[tool]
        entry["calls"] += 1
        entry["request_bytes"] += r.get("request_bytes") or 0
        entry["response_bytes"] += r.get("response_bytes") or 0
        lat = r.get("tool_latency_sec") or r.get("latency_sec") or 0.0
        entry["tool_latency_sec"] += float(lat)
    return dict(tool_bytes)


# ---------------------------------------------------------------------------
# Q5 — Tokenized traffic
# ---------------------------------------------------------------------------

def _q5_tokenized(records: list[dict], pcap_metrics: list) -> dict:
    """Q5: token counts (supported), token rate (supported),
    token↔DL-pkt rate, inter-token gap distribution per profile,
    tokens→bytes regression."""
    out = {
        "token_counts_by_scenario": {},
        "inter_token_gap_by_profile": {},
        "token_to_bytes_regression_by_scenario": {},
        "token_arrival_vs_pkt_arrival": {},
    }

    # Token counts per scenario (already tracked but re-emit as distributions)
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if not _is_primary_turn(r) or not r.get("success"):
            continue
        by_scenario[r.get("scenario_id") or "?"].append(r)
    for scenario, recs in by_scenario.items():
        tin = [r.get("tokens_in") for r in recs if r.get("tokens_in")]
        tout = [r.get("tokens_out") for r in recs if r.get("tokens_out")]
        rates = [
            (r["tokens_out"] / r["latency_sec"])
            for r in recs
            if r.get("tokens_out") and r.get("latency_sec") and r["latency_sec"] > 0
        ]
        out["token_counts_by_scenario"][scenario] = {
            "tokens_in": _distribution(tin),
            "tokens_out": _distribution(tout),
            "tokens_per_sec": _distribution(rates),
        }

    # Inter-token gap per profile — pulled from inter_chunk_times, filtered to streaming
    by_profile: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if not _is_primary_turn(r) or not r.get("is_streaming"):
            continue
        raw = r.get("inter_chunk_times")
        try:
            times = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            times = []
        for t in times:
            if isinstance(t, (int, float)) and t >= 0:
                by_profile[r.get("network_profile") or "?"].append(float(t))
    for profile, gaps in by_profile.items():
        out["inter_token_gap_by_profile"][profile] = _distribution(gaps)

    # Tokens → bytes regression (Q5.5) — simple least-squares per scenario.
    # Enables RAN2 to model UL bytes ≈ α·tokens_in + β and DL bytes ≈ γ·tokens_out + δ.
    for scenario, recs in by_scenario.items():
        pairs_in = [
            (float(r["tokens_in"]), float(r["request_bytes"]))
            for r in recs
            if r.get("tokens_in") and r.get("request_bytes") is not None
        ]
        pairs_out = [
            (float(r["tokens_out"]), float(r["response_bytes"]))
            for r in recs
            if r.get("tokens_out") and r.get("response_bytes") is not None
        ]
        out["token_to_bytes_regression_by_scenario"][scenario] = {
            "ul": _least_squares(pairs_in),
            "dl": _least_squares(pairs_out),
        }

    # Token-arrival rate vs DL-pkt arrival rate (Q5.3): reuse inter-token gaps
    # as token arrivals; pcap DL packet gaps as packet arrivals. Aggregated.
    token_rate_per_profile = {
        p: (1.0 / (stats["p50"] or 1e-9)) if (stats.get("p50") or 0) > 0 else None
        for p, stats in out["inter_token_gap_by_profile"].items()
    }
    dl_pkt_rates = []
    for m in pcap_metrics:
        dl_pkts = [p for p in (getattr(m, "packets", []) or []) if p.direction == "dl"]
        if len(dl_pkts) >= 2:
            span = dl_pkts[-1].timestamp - dl_pkts[0].timestamp
            if span > 0:
                dl_pkt_rates.append(len(dl_pkts) / span)
    out["token_arrival_vs_pkt_arrival"] = {
        "token_rate_per_profile_hz": token_rate_per_profile,
        "dl_pkt_rate_hz": _distribution(dl_pkt_rates),
    }
    return out


def _least_squares(pairs: list[tuple[float, float]]) -> Optional[dict]:
    """Return slope/intercept/r2 for y = m*x + b. None if < 2 points."""
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(pairs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return None
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    var_y = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = (1.0 - ss_res / var_y) if var_y > 0 else None
    return {"n": n, "slope": slope, "intercept": intercept, "r2": r2}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_ran2_metrics(
    records: list[dict],
    pcap_metrics: Optional[list] = None,
    profiles_yaml: Optional[str] = None,
) -> dict:
    """Compute the full RAN2 methodology metric set (S4-260859 Annex D)."""
    pcap_metrics = list(pcap_metrics or [])
    profile_loss = _load_profile_loss_pct(profiles_yaml)
    return {
        "generated_at": time.time(),
        "n_records": len(records),
        "n_pcap_files": len(pcap_metrics),
        "Q1": _q1_ul_heavy(records, pcap_metrics),
        "Q2": _q2_bursts(records, pcap_metrics),
        "Q3": _q3_rtt(records, pcap_metrics),
        "Q4": _q4_variability(records, pcap_metrics, profile_loss),
        "Q5": _q5_tokenized(records, pcap_metrics),
    }
