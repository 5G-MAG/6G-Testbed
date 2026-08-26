"""
RAN2 methodology metrics (S4-260859 Annex D).

This module computes the metrics SA4 listed in response to RAN2's four
working assumptions on 6G AI traffic characteristics plus the tokenized-
traffic topic. Each metric family is organized under the corresponding
RAN2 question (Q1..Q5). The output is a nested dict that downstream
report/chart generators read verbatim.

Inputs:
    records:        list[dict] from traffic_logs (one per turn)
    pcap_metrics:   optional list[PcapMetrics] from netemu.pcap

Packet-level extraction is delegated to netemu.metrics, which owns the
definitions of handshake RTT, TLS setup, connection setup and flow duration.
This module keeps the join to scenario and profile labels, the application-layer
volumes and token counts from the database, and its own interpolating percentile
so reported values stay comparable with earlier campaigns.
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
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from netemu.metrics import collect_connection_samples


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


def _distribution_extended(values: list[float]) -> dict[str, Optional[float]]:
    """Like _distribution() but also includes p10/p25/p75/p90, plus sum.
    Used for metrics that need to be plotted as box-plot / CDF (Q2 burst
    sizes, durations, peak rates, idle gaps)."""
    if not values:
        return {"n": 0, "min": None, "p10": None, "p25": None, "p50": None,
                "p75": None, "p90": None, "p95": None, "p99": None,
                "max": None, "mean": None, "sum": 0.0}
    return {
        "n": len(values),
        "min": min(values),
        "p10": _percentile(values, 10),
        "p25": _percentile(values, 25),
        "p50": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
        "mean": statistics.mean(values),
        "sum": sum(values),
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
    if isinstance(raw, dict):
        return raw
    try:
        meta = json.loads(raw)
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _record_type(record: dict) -> str:
    """Normalize the two metadata schemas emitted by scenario implementations."""
    meta = _metadata(record)
    return str(meta.get("record_type") or meta.get("type") or "")


def _is_primary_turn(record: dict) -> bool:
    """Exclude pcap/tool/computer_action/capture rows; keep real LLM turns only."""
    session_id = record.get("session_id") or ""
    if session_id.startswith("pcap_") or session_id.startswith("timeout_"):
        return False
    if record.get("turn_index") is not None and record.get("turn_index") < 0:
        return False
    if _record_type(record) in (
        "tool_call", "mcp_tool_call", "computer_action",
        "computer_use_action", "pcap_capture",
    ):
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


_PCAP_TS_RE = re.compile(r"capture(?:_[a-z0-9]+)?_(\d{8})_(\d{6})\.pcap$")


def _pcap_start_unix(pcap_file: str) -> Optional[float]:
    """Parse `capture[_iface]_YYYYMMDD_HHMMSS.pcap` -> unix timestamp (local tz).

    The orchestrator writes pcap filenames using `datetime.now().strftime(...)`,
    which is local time. `datetime.timestamp()` interprets a naive datetime as
    local time, so this round-trips correctly."""
    if not pcap_file:
        return None
    m = _PCAP_TS_RE.search(Path(pcap_file).name)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def _build_pcap_sp_map(pcap_metrics: Iterable, records: list[dict]) -> dict[str, tuple[str, str]]:
    """Map each pcap_file path to (scenario_id, network_profile).

    The orchestrator is invoked once per (scenario, profile); for every run it
    writes a *pair* of pcaps (main interface + loopback) with the same timestamp
    in the filename. We:
      1. group pcap files by their start timestamp (so each pair shares a vote);
      2. bucket records by the timestamp window [start[i], start[i+1])
         and let the dominant (scenario, profile) win;
      3. expand the per-timestamp result back to every pcap file in that group."""
    by_ts: dict[float, list[str]] = defaultdict(list)
    for m in pcap_metrics:
        pf = getattr(m, "pcap_file", "") or ""
        ts = _pcap_start_unix(pf)
        if ts is None:
            continue
        by_ts[ts].append(pf)
    if not by_ts:
        return {}

    boundaries = sorted(by_ts.keys())

    bucket_votes: dict[float, Counter] = defaultdict(Counter)
    for r in records:
        rt = r.get("timestamp") or r.get("t_request_start")
        if rt is None or rt < boundaries[0]:
            continue
        # Binary search for the largest boundary <= rt
        lo, hi = 0, len(boundaries) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if boundaries[mid] <= rt:
                lo = mid
            else:
                hi = mid - 1
        sc = r.get("scenario_id") or "?"
        pr = r.get("network_profile") or "?"
        bucket_votes[boundaries[lo]][(sc, pr)] += 1

    result: dict[str, tuple[str, str]] = {}
    for ts, counter in bucket_votes.items():
        if not counter:
            continue
        (sc, pr), _ = counter.most_common(1)[0]
        for pf in by_ts.get(ts, []):
            result[pf] = (sc, pr)
    return result


# ---------------------------------------------------------------------------
# Q1 — UL-heavy
# ---------------------------------------------------------------------------

def _q1_ul_heavy(records: list[dict], pcap_metrics: list, pcap_sp_map: dict[str, tuple[str, str]]) -> dict:
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
        pf = getattr(m, "pcap_file", "")
        sc, pr = pcap_sp_map.get(pf, ("?", "?"))
        pcap_rows.append({
            "pcap_file": pf,
            "scenario_id": sc,
            "network_profile": pr,
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

def _q2_bursts(records: list[dict], pcap_metrics: list, pcap_sp_map: dict[str, tuple[str, str]]) -> dict:
    """Q2: per-direction bursts at 10/100ms gap; burstiness per window;
    TTFB/TTLB (already supported)."""
    out = {
        "per_pcap": [],
        "per_scenario_profile_delay": {},
        "per_scenario_profile_bursts": {},
    }

    # Raw-value buckets per (scenario, profile) × gap × direction so the
    # downstream chart generator can render extended distributions and
    # arrival-rate / duty-cycle without a second pcap pass.
    raw: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "capture_duration_sum": 0.0,
            "by_gap": defaultdict(lambda: {
                "ul": {"sizes": [], "durs": [], "peaks_mbps": []},
                "dl": {"sizes": [], "durs": [], "peaks_mbps": []},
            }),
        }
    )

    for m in pcap_metrics:
        bursts_by_gap = dict(getattr(m, "bursts_by_gap", {}) or {})
        idle_by_gap = dict(getattr(m, "interburst_idle_by_gap", {}) or {})
        pf = getattr(m, "pcap_file", "")
        sc, pr = pcap_sp_map.get(pf, ("?", "?"))
        cap_dur = float(getattr(m, "capture_duration", 0.0) or 0.0)
        entry: dict[str, Any] = {
            "pcap_file": pf,
            "scenario_id": sc,
            "network_profile": pr,
            "capture_duration_sec": cap_dur,
            "burstiness_by_window": dict(getattr(m, "burstiness_by_window", {}) or {}),
            "burst_stats_by_gap": {},
            "interburst_idle_by_gap": {},
        }
        sp_key = (sc, pr)
        if sc != "?" and pr != "?":
            raw[sp_key]["capture_duration_sum"] += cap_dur
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
                if sc != "?" and pr != "?" and direction in ("ul", "dl"):
                    bucket = raw[sp_key]["by_gap"][label][direction]
                    bucket["sizes"].extend(sizes)
                    bucket["durs"].extend(durs)
                    bucket["peaks_mbps"].extend(peaks)
        for label, per_dir in idle_by_gap.items():
            entry["interburst_idle_by_gap"][label] = {}
            for direction, gaps in (per_dir or {}).items():
                entry["interburst_idle_by_gap"][label][direction] = {
                    "cdf_sec": _distribution(gaps),
                    "cv": _cv(gaps),
                }
        out["per_pcap"].append(entry)

    # Per-(scenario, profile) aggregates for the new burst charts.
    for (sc, pr), data in raw.items():
        cap_total = data["capture_duration_sum"]
        sp_entry: dict[str, Any] = {
            "capture_duration_sec_total": cap_total,
            "by_gap": {},
        }
        for label, per_dir in data["by_gap"].items():
            sp_entry["by_gap"][label] = {}
            for direction in ("ul", "dl"):
                buf = per_dir[direction]
                sizes = buf["sizes"]
                durs = buf["durs"]
                peaks = buf["peaks_mbps"]
                count = len(sizes)
                sum_dur = sum(durs)
                sp_entry["by_gap"][label][direction] = {
                    "count": count,
                    "arrival_rate_per_sec": (count / cap_total) if cap_total > 0 else None,
                    "duty_cycle": (sum_dur / cap_total) if cap_total > 0 else None,
                    "size_bytes": _distribution_extended(sizes),
                    "duration_sec": _distribution_extended(durs),
                    "peak_rate_mbps": _distribution_extended(peaks),
                }
        out["per_scenario_profile_bursts"][f"{sc}/{pr}"] = sp_entry

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
    """Q3: TCP RTT, TLS handshake, full connection setup (SYN → first TLS
    ApplicationData), inter-chunk gap vs RTT, E2E latency vs RTT."""
    out = {"tcp_rtt": {}, "tls_handshake": {}, "connection_setup_ms": {},
           "inter_chunk_vs_rtt": {}, "e2e_latency_vs_rtt": {}}

    # Packet-level samples come from netemu.metrics, which owns the extraction
    # semantics (handshake RTT, TLS ClientHello to first ApplicationData, and
    # SYN to first application data for the full connection setup, falling back
    # to first TCP payload on non-TLS flows). They are summarized here with this
    # module's interpolating percentile so reported values stay comparable with
    # earlier campaigns.
    samples = collect_connection_samples(pcap_metrics)
    out["tcp_rtt"] = _distribution(samples["tcp_handshake_rtt_ms"])
    out["connection_setup_ms"] = _distribution(samples["connection_setup_ms"])

    # TLS handshake duration falls back to per-record metadata.tls for clients
    # that report handshake_ms directly and leave no flow-level value.
    tls_ms: list[float] = [d for d in samples["tls_handshake_ms"] if d > 0]
    if not tls_ms:
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
    pcap_sp_map: dict[str, tuple[str, str]],
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

    packet_counts_by_scenario: dict[str, list[int]] = defaultdict(list)
    for metric in pcap_metrics:
        mapped_scenario, _ = pcap_sp_map.get(
            getattr(metric, "pcap_file", ""), ("?", "?")
        )
        if mapped_scenario == "?":
            continue
        for flow in getattr(metric, "flows", []) or []:
            total = (
                (getattr(flow, "packets_sent", 0) or 0)
                + (getattr(flow, "packets_recv", 0) or 0)
            )
            if total > 0:
                packet_counts_by_scenario[mapped_scenario].append(total)

    for scenario, recs in by_scenario.items():
        req_bytes = [r.get("request_bytes") or 0 for r in recs if r.get("success")]
        resp_bytes = [r.get("response_bytes") or 0 for r in recs if r.get("success")]
        out["volume_distribution"][scenario] = {
            "request_bytes": _distribution(req_bytes),
            "response_bytes": _distribution(resp_bytes),
            "packet_count_per_flow": _distribution(packet_counts_by_scenario[scenario]),
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

    # Flow duration and flows-per-capture come from netemu.metrics, which owns
    # the extraction. Verified identical to the previous local implementation on
    # the reference capture set.
    conn_samples = collect_connection_samples(pcap_metrics)
    all_flow_durations = conn_samples["flow_duration_s"]
    flows_per_pcap = conn_samples["flows_per_capture"]

    # These two are deliberately NOT delegated, because netemu defines them
    # differently and swapping the definitions would move published numbers:
    #   * reuse here means the same 5-tuple recurring across captures, whereas
    #     netemu.metrics counts flows carrying two or more application exchanges;
    #   * a destination here is (ip, port), whereas netemu counts distinct ip.
    flow_keys_seen: set = set()
    reuse_hits = 0
    reuse_total = 0
    distinct_dests_per_pcap: list[int] = []
    for m in pcap_metrics:
        pcap_dests: set = set()
        for flow in getattr(m, "flows", []) or []:
            fk = getattr(flow, "flow_key", "")
            if fk:
                reuse_total += 1
                if fk in flow_keys_seen:
                    reuse_hits += 1
                flow_keys_seen.add(fk)
            dst_ip = getattr(flow, "dst_ip", "")
            if dst_ip:
                pcap_dests.add((dst_ip, getattr(flow, "dst_port", 0)))
        distinct_dests_per_pcap.append(len(pcap_dests))

    out["connection_duration"] = {
        # Extended distribution so the chart can show p10 (and other in-between
        # percentiles) without re-scanning pcaps every time.
        "flow_duration_sec": _distribution_extended(all_flow_durations),
        "flows_per_pcap": _distribution(flows_per_pcap),
        "connection_reuse_ratio": (reuse_hits / reuse_total) if reuse_total else None,
    }
    out["agentic_flows"] = {
        "distinct_dests_per_pcap": _distribution([float(x) for x in distinct_dests_per_pcap]),
        "per_tool_bytes": _per_tool_bytes(records),  # from DB metadata, no pcap join needed
    }
    return out


def _per_tool_bytes(records: list[dict]) -> dict[str, dict]:
    """Aggregate request/response bytes per MCP tool name, from
    tool-call records in either supported metadata schema."""
    tool_bytes: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "request_bytes": 0, "response_bytes": 0, "tool_latency_sec": 0.0}
    )
    for r in records:
        meta = _metadata(r)
        if _record_type(r) not in ("tool_call", "mcp_tool_call"):
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
    pcap_sp_map = _build_pcap_sp_map(pcap_metrics, records)
    return {
        "generated_at": time.time(),
        "n_records": len(records),
        "n_pcap_files": len(pcap_metrics),
        "n_pcap_sp_mapped": len(pcap_sp_map),
        "Q1": _q1_ul_heavy(records, pcap_metrics, pcap_sp_map),
        "Q2": _q2_bursts(records, pcap_metrics, pcap_sp_map),
        "Q3": _q3_rtt(records, pcap_metrics),
        "Q4": _q4_variability(records, pcap_metrics, profile_loss, pcap_sp_map),
        "Q5": _q5_tokenized(records, pcap_metrics),
    }
