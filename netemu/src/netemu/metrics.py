"""Aggregate packet-level metrics across one or more captures.

:mod:`netemu.pcap` parses a single pcap into a :class:`~netemu.pcap.PcapMetrics`
object holding per-packet records, reassembled TCP flows, and the per-capture
aggregates derived from them. This module takes one or more of those objects and
summarizes them as distributions, which is what a measurement campaign
ultimately reports.

Everything here is derived from packet headers alone: sizes, timestamps,
directions, TCP flags and sequence numbers. Nothing in this module knows about
application payloads, test scenarios, traffic labels, or any external metric
taxonomy. That keeps the layer reusable for any shaped-network measurement.

Five metric families are produced:

* **Connection** - handshake RTT, TLS setup, full connection setup, flow
  duration, per-flow exchange counts, and destination fan-out.
* **Direction** - uplink and downlink byte and packet volumes, their ratio, and
  per-direction packet size and inter-packet gap distributions.
* **Burst** - burst count, size, duration, peak rate, and inter-burst idle time,
  per gap threshold and direction.
* **Throughput** - mean and peak throughput, plus peak and burstiness per
  averaging window.
* **Reliability** - retransmission counts and rates.

Typical use::

    >>> from netemu import analyze_multiple_pcaps, compute_packet_metrics
    >>> captures = analyze_multiple_pcaps("captures/")        # doctest: +SKIP
    >>> report = compute_packet_metrics(captures)             # doctest: +SKIP
    >>> report.connection.connection_setup_ms.p95             # doctest: +SKIP
    4248.5
    >>> import json
    >>> json.dumps(report.to_dict())                          # doctest: +SKIP

A single capture works the same way::

    >>> from netemu import analyze_pcap, compute_packet_metrics
    >>> report = compute_packet_metrics(analyze_pcap("run.pcap"))   # doctest: +SKIP
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union

from .pcap import PcapMetrics, TCPFlow

__all__ = [
    "ConnectionSamples",
    "Distribution",
    "ConnectionMetrics",
    "DirectionMetrics",
    "BurstMetrics",
    "ThroughputMetrics",
    "ReliabilityMetrics",
    "PacketMetricsReport",
    "summarize",
    "compute_packet_metrics",
    "collect_connection_samples",
]

# A flow is counted as carrying more than one exchange when at least this many
# uplink-to-downlink payload transitions are observed on it.
_REUSE_MIN_EXCHANGES = 2


# ---------------------------------------------------------------------------
# Distribution summary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Distribution:
    """Summary of a sample: count, central tendency, spread, and percentiles.

    Percentiles use nearest-rank on the sorted sample, so every reported value
    is an observed value rather than an interpolation. All fields except ``n``
    are ``None`` for an empty sample.
    """

    n: int = 0
    mean: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    stdev: Optional[float] = None
    cv: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "stdev": self.stdev,
            "cv": self.cv,
        }


def _percentile(ordered: Sequence[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile of an already sorted sample."""
    if not ordered:
        return None
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def summarize(values: Iterable[float]) -> Distribution:
    """Summarize a sample, ignoring non-finite entries.

    >>> summarize([1.0, 2.0, 3.0, 4.0]).p50
    3.0
    >>> summarize([]).n
    0
    """
    clean = [float(v) for v in values
             if v is not None and float(v) == float(v) and abs(float(v)) != float("inf")]
    if not clean:
        return Distribution()
    ordered = sorted(clean)
    mean = statistics.fmean(ordered)
    stdev = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return Distribution(
        n=len(ordered),
        mean=mean,
        minimum=ordered[0],
        maximum=ordered[-1],
        p50=_percentile(ordered, 50),
        p95=_percentile(ordered, 95),
        p99=_percentile(ordered, 99),
        stdev=stdev,
        cv=(stdev / mean) if mean else None,
    )


# ---------------------------------------------------------------------------
# Metric families
# ---------------------------------------------------------------------------
@dataclass
class ConnectionMetrics:
    """Connection establishment and lifetime, measured from TCP flows."""

    flows: int = 0
    tcp_handshake_rtt_ms: Distribution = field(default_factory=Distribution)
    tls_handshake_ms: Distribution = field(default_factory=Distribution)
    connection_setup_ms: Distribution = field(default_factory=Distribution)
    time_to_first_data_ms: Distribution = field(default_factory=Distribution)
    flow_duration_s: Distribution = field(default_factory=Distribution)
    flows_per_capture: Distribution = field(default_factory=Distribution)
    exchanges_per_flow: Distribution = field(default_factory=Distribution)
    #: Fraction of flows carrying at least two application exchanges, that is,
    #: connections actually reused rather than opened for a single turn.
    reuse_ratio: Optional[float] = None
    distinct_destinations_per_capture: Distribution = field(default_factory=Distribution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flows": self.flows,
            "tcp_handshake_rtt_ms": self.tcp_handshake_rtt_ms.to_dict(),
            "tls_handshake_ms": self.tls_handshake_ms.to_dict(),
            "connection_setup_ms": self.connection_setup_ms.to_dict(),
            "time_to_first_data_ms": self.time_to_first_data_ms.to_dict(),
            "flow_duration_s": self.flow_duration_s.to_dict(),
            "flows_per_capture": self.flows_per_capture.to_dict(),
            "exchanges_per_flow": self.exchanges_per_flow.to_dict(),
            "reuse_ratio": self.reuse_ratio,
            "distinct_destinations_per_capture":
                self.distinct_destinations_per_capture.to_dict(),
        }


@dataclass
class DirectionMetrics:
    """Uplink and downlink volumes and per-direction packet dynamics."""

    ul_packets: int = 0
    dl_packets: int = 0
    ul_bytes: int = 0
    dl_bytes: int = 0
    ul_dl_byte_ratio: Optional[float] = None
    ul_dl_packet_ratio: Optional[float] = None
    ul_packet_size: Distribution = field(default_factory=Distribution)
    dl_packet_size: Distribution = field(default_factory=Distribution)
    ul_inter_packet_gap_ms: Distribution = field(default_factory=Distribution)
    dl_inter_packet_gap_ms: Distribution = field(default_factory=Distribution)
    ul_dl_byte_ratio_per_capture: Distribution = field(default_factory=Distribution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ul_packets": self.ul_packets,
            "dl_packets": self.dl_packets,
            "ul_bytes": self.ul_bytes,
            "dl_bytes": self.dl_bytes,
            "ul_dl_byte_ratio": self.ul_dl_byte_ratio,
            "ul_dl_packet_ratio": self.ul_dl_packet_ratio,
            "ul_packet_size": self.ul_packet_size.to_dict(),
            "dl_packet_size": self.dl_packet_size.to_dict(),
            "ul_inter_packet_gap_ms": self.ul_inter_packet_gap_ms.to_dict(),
            "dl_inter_packet_gap_ms": self.dl_inter_packet_gap_ms.to_dict(),
            "ul_dl_byte_ratio_per_capture": self.ul_dl_byte_ratio_per_capture.to_dict(),
        }


@dataclass
class BurstMetrics:
    """Burst structure, keyed by the gap threshold that defines a burst.

    ``by_gap[label][direction]`` holds ``count``, ``size_bytes``,
    ``duration_s``, ``peak_rate_bps`` and ``idle_gap_s`` for that combination.
    Gap labels come from whatever thresholds :class:`~netemu.pcap.PcapAnalyzer`
    was configured with.
    """

    by_gap: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"by_gap": self.by_gap}


@dataclass
class ThroughputMetrics:
    """Throughput across captures and across averaging windows."""

    mean_mbps: Distribution = field(default_factory=Distribution)
    peak_mbps: Distribution = field(default_factory=Distribution)
    peak_mbps_by_window: dict[str, dict[str, Any]] = field(default_factory=dict)
    burstiness_by_window: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_mbps": self.mean_mbps.to_dict(),
            "peak_mbps": self.peak_mbps.to_dict(),
            "peak_mbps_by_window": self.peak_mbps_by_window,
            "burstiness_by_window": self.burstiness_by_window,
        }


@dataclass
class ReliabilityMetrics:
    """Retransmission behavior across captures and flows."""

    total_retransmissions: int = 0
    retransmission_rate_per_capture: Distribution = field(default_factory=Distribution)
    retransmission_rate_per_flow: Distribution = field(default_factory=Distribution)
    flows_with_retransmissions: int = 0
    flow_retransmission_ratio: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_retransmissions": self.total_retransmissions,
            "retransmission_rate_per_capture":
                self.retransmission_rate_per_capture.to_dict(),
            "retransmission_rate_per_flow":
                self.retransmission_rate_per_flow.to_dict(),
            "flows_with_retransmissions": self.flows_with_retransmissions,
            "flow_retransmission_ratio": self.flow_retransmission_ratio,
        }


@dataclass
class PacketMetricsReport:
    """Everything :func:`compute_packet_metrics` derives from a capture set."""

    captures: int = 0
    total_packets: int = 0
    total_bytes: int = 0
    tcp_packets: int = 0
    udp_packets: int = 0
    capture_duration_s: Distribution = field(default_factory=Distribution)
    connection: ConnectionMetrics = field(default_factory=ConnectionMetrics)
    direction: DirectionMetrics = field(default_factory=DirectionMetrics)
    bursts: BurstMetrics = field(default_factory=BurstMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    reliability: ReliabilityMetrics = field(default_factory=ReliabilityMetrics)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form of the whole report."""
        return {
            "captures": self.captures,
            "total_packets": self.total_packets,
            "total_bytes": self.total_bytes,
            "tcp_packets": self.tcp_packets,
            "udp_packets": self.udp_packets,
            "capture_duration_s": self.capture_duration_s.to_dict(),
            "connection": self.connection.to_dict(),
            "direction": self.direction.to_dict(),
            "bursts": self.bursts.to_dict(),
            "throughput": self.throughput.to_dict(),
            "reliability": self.reliability.to_dict(),
        }


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------
def _count_exchanges(records: Sequence[Any], since: Optional[float] = None) -> int:
    """Uplink-to-downlink payload alternations in one flow.

    Each alternation is one request-response turn as far as the packet headers
    can tell: a payload-bearing uplink packet followed by a payload-bearing
    downlink packet. Pure ACKs and other empty segments are skipped, since they
    carry no exchange.

    ``since`` excludes everything before the flow's first application data. This
    matters on TLS flows, whose handshake is itself a payload-bearing exchange
    in both directions and would otherwise be counted as application turns. On
    a sampled capture that inflated the count by roughly 20 packets per flow.
    """
    exchanges = 0
    pending_ul = False
    for rec in records:
        if getattr(rec, "payload_len", 0) <= 0:
            continue
        if since is not None and rec.timestamp < since:
            continue
        if rec.direction == "ul":
            pending_ul = True
        elif rec.direction == "dl" and pending_ul:
            exchanges += 1
            pending_ul = False
    return exchanges


def _flow_exchange_counts(capture: PcapMetrics) -> list[int]:
    """Per-flow exchange counts, from per-packet records when available.

    Counting starts at each flow's first application data, so a TLS handshake
    is not mistaken for application traffic.
    """
    if not capture.packets:
        return []
    app_start: dict[str, Optional[float]] = {
        flow.flow_key: (flow.tls_first_app_data_time or flow.first_data_time)
        for flow in capture.flows
    }
    by_flow: dict[str, list[Any]] = defaultdict(list)
    for rec in capture.packets:
        by_flow[getattr(rec, "flow_key", "")].append(rec)
    return [_count_exchanges(recs, app_start.get(key))
            for key, recs in by_flow.items()]


def _inter_packet_gaps_ms(capture: PcapMetrics, direction: str) -> list[float]:
    """Inter-packet gaps within one direction, in milliseconds."""
    stamps = [r.timestamp for r in capture.packets if r.direction == direction]
    if len(stamps) < 2:
        return []
    stamps.sort()
    return [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:]) if b >= a]


def _flow_setup_samples(flow: TCPFlow) -> dict[str, Optional[float]]:
    """Per-flow setup timings in milliseconds.

    ``connection_setup`` spans the SYN to the first application data, so it
    includes the handshake, any local scheduling delay, and the TLS exchange
    when one is present.
    """
    def ms(value: Optional[float]) -> Optional[float]:
        return None if value is None else value * 1000.0

    setup: Optional[float] = None
    if flow.syn_time is not None:
        end = flow.tls_first_app_data_time or flow.first_data_time
        if end is not None and end >= flow.syn_time:
            setup = end - flow.syn_time
    return {
        "handshake_rtt": ms(flow.handshake_rtt),
        "tls_handshake": ms(flow.tls_handshake_duration),
        "connection_setup": ms(setup),
        "time_to_first_data": ms(flow.time_to_first_data),
    }


#: Raw per-flow and per-capture samples underlying :class:`ConnectionMetrics`.
#: Values are plain lists so a caller can summarize them with its own
#: conventions (percentile method, rounding, units) while still sharing this
#: module's extraction semantics.
ConnectionSamples = dict


def collect_connection_samples(
    captures: Union[PcapMetrics, Iterable[PcapMetrics]],
) -> dict[str, list[float]]:
    """Extract the raw connection samples without summarizing them.

    :func:`compute_packet_metrics` is the normal entry point. This function
    exists for callers that must apply their own summary convention, for example
    an interpolating percentile, and would otherwise reimplement the flow
    walking and risk drifting from it.

    Returns a dict of samples keyed by metric name. Units match the
    corresponding :class:`ConnectionMetrics` field: milliseconds for the setup
    timings, seconds for durations, counts otherwise.

    >>> samples = collect_connection_samples([])
    >>> sorted(samples)[:3]
    ['connection_setup_ms', 'distinct_destinations_per_capture', 'exchanges_per_flow']
    """
    items: list[PcapMetrics] = (
        [captures] if isinstance(captures, PcapMetrics) else list(captures)
    )
    out: dict[str, list[float]] = {
        "tcp_handshake_rtt_ms": [],
        "tls_handshake_ms": [],
        "connection_setup_ms": [],
        "time_to_first_data_ms": [],
        "flow_duration_s": [],
        "flows_per_capture": [],
        "exchanges_per_flow": [],
        "distinct_destinations_per_capture": [],
        "retransmission_rate_per_flow": [],
    }
    for capture in items:
        flows = list(capture.flows)
        out["flows_per_capture"].append(float(len(flows)))
        out["distinct_destinations_per_capture"].append(
            float(len({f.dst_ip for f in flows}))
        )
        for flow in flows:
            samples = _flow_setup_samples(flow)
            if samples["handshake_rtt"] is not None:
                out["tcp_handshake_rtt_ms"].append(samples["handshake_rtt"])
            if samples["tls_handshake"] is not None:
                out["tls_handshake_ms"].append(samples["tls_handshake"])
            if samples["connection_setup"] is not None:
                out["connection_setup_ms"].append(samples["connection_setup"])
            if samples["time_to_first_data"] is not None:
                out["time_to_first_data_ms"].append(samples["time_to_first_data"])
            if flow.duration > 0:
                out["flow_duration_s"].append(flow.duration)
            out["retransmission_rate_per_flow"].append(flow.retransmission_rate)
        out["exchanges_per_flow"].extend(
            float(c) for c in _flow_exchange_counts(capture)
        )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def compute_packet_metrics(
    captures: Union[PcapMetrics, Iterable[PcapMetrics]],
) -> PacketMetricsReport:
    """Aggregate packet-level metrics over one capture or many.

    Args:
        captures: A single :class:`~netemu.pcap.PcapMetrics`, or any iterable of
            them, as returned by :func:`~netemu.pcap.analyze_pcap` and
            :func:`~netemu.pcap.analyze_multiple_pcaps`.

    Returns:
        A :class:`PacketMetricsReport`. Metric families with no supporting data
        come back empty rather than absent, so the shape of the report does not
        depend on what a particular capture happened to contain.

    Distributions over flows pool every flow in the set. Distributions named
    ``*_per_capture`` treat each capture as one observation, which is the right
    unit when captures differ in length.
    """
    items: list[PcapMetrics] = (
        [captures] if isinstance(captures, PcapMetrics) else list(captures)
    )

    report = PacketMetricsReport(captures=len(items))
    if not items:
        return report

    ul_sizes: list[float] = []
    dl_sizes: list[float] = []
    ul_gaps: list[float] = []
    dl_gaps: list[float] = []
    per_capture_ratio: list[float] = []
    capture_durations: list[float] = []
    mean_mbps: list[float] = []
    peak_mbps: list[float] = []
    retrans_capture: list[float] = []
    peak_by_window: dict[str, list[float]] = defaultdict(list)
    burst_by_window: dict[str, list[float]] = defaultdict(list)
    bursts: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    total_flows = 0
    flows_with_retrans = 0

    for capture in items:
        report.total_packets += capture.total_packets
        report.total_bytes += capture.total_bytes
        report.tcp_packets += capture.tcp_packets
        report.udp_packets += capture.udp_packets
        report.direction.ul_packets += capture.ul_packets
        report.direction.dl_packets += capture.dl_packets
        report.direction.ul_bytes += capture.ul_bytes_total
        report.direction.dl_bytes += capture.dl_bytes_total
        report.reliability.total_retransmissions += capture.total_retransmissions

        capture_durations.append(capture.capture_duration)
        if capture.avg_throughput_mbps:
            mean_mbps.append(capture.avg_throughput_mbps)
        if capture.peak_throughput_mbps:
            peak_mbps.append(capture.peak_throughput_mbps)
        retrans_capture.append(capture.retransmission_rate)

        total_flows += len(capture.flows)
        flows_with_retrans += sum(1 for f in capture.flows if f.retransmissions)

        ul_sizes.extend(float(r.size) for r in capture.packets if r.direction == "ul")
        dl_sizes.extend(float(r.size) for r in capture.packets if r.direction == "dl")
        ul_gaps.extend(_inter_packet_gaps_ms(capture, "ul"))
        dl_gaps.extend(_inter_packet_gaps_ms(capture, "dl"))
        if capture.dl_bytes_total:
            per_capture_ratio.append(capture.ul_bytes_total / capture.dl_bytes_total)

        for label, value in (capture.peak_mbps_by_window or {}).items():
            if value is not None:
                peak_by_window[label].append(float(value))
        for label, value in (capture.burstiness_by_window or {}).items():
            if value is not None:
                burst_by_window[label].append(float(value))

        for label, per_dir in (capture.bursts_by_gap or {}).items():
            for direction, burst_list in (per_dir or {}).items():
                slot = bursts[label][direction]
                for burst in burst_list or []:
                    slot["size_bytes"].append(float(burst.get("total_bytes", 0)))
                    slot["duration_s"].append(float(burst.get("duration_sec", 0.0)))
                    slot["peak_rate_bps"].append(float(burst.get("peak_rate_bps", 0.0)))
                slot["count"].append(float(len(burst_list or [])))
        for label, per_dir in (capture.interburst_idle_by_gap or {}).items():
            for direction, gaps in (per_dir or {}).items():
                bursts[label][direction]["idle_gap_s"].extend(
                    float(g) for g in gaps or []
                )

    # --- connection ---
    # Extraction lives in collect_connection_samples so this module has exactly
    # one definition of each sample, shared with callers that summarize their own way.
    samples = collect_connection_samples(items)
    conn = report.connection
    conn.flows = total_flows
    conn.tcp_handshake_rtt_ms = summarize(samples["tcp_handshake_rtt_ms"])
    conn.tls_handshake_ms = summarize(samples["tls_handshake_ms"])
    conn.connection_setup_ms = summarize(samples["connection_setup_ms"])
    conn.time_to_first_data_ms = summarize(samples["time_to_first_data_ms"])
    conn.flow_duration_s = summarize(samples["flow_duration_s"])
    conn.flows_per_capture = summarize(samples["flows_per_capture"])
    conn.exchanges_per_flow = summarize(samples["exchanges_per_flow"])
    conn.distinct_destinations_per_capture = summarize(
        samples["distinct_destinations_per_capture"])
    exchanges = samples["exchanges_per_flow"]
    if exchanges:
        reused = sum(1 for c in exchanges if c >= _REUSE_MIN_EXCHANGES)
        conn.reuse_ratio = reused / len(exchanges)

    # --- direction ---
    direction = report.direction
    direction.ul_packet_size = summarize(ul_sizes)
    direction.dl_packet_size = summarize(dl_sizes)
    direction.ul_inter_packet_gap_ms = summarize(ul_gaps)
    direction.dl_inter_packet_gap_ms = summarize(dl_gaps)
    direction.ul_dl_byte_ratio_per_capture = summarize(per_capture_ratio)
    if direction.dl_bytes:
        direction.ul_dl_byte_ratio = direction.ul_bytes / direction.dl_bytes
    if direction.dl_packets:
        direction.ul_dl_packet_ratio = direction.ul_packets / direction.dl_packets

    # --- bursts ---
    report.bursts.by_gap = {
        label: {
            dirn: {name: summarize(values).to_dict() for name, values in stats.items()}
            for dirn, stats in per_dir.items()
        }
        for label, per_dir in bursts.items()
    }

    # --- throughput ---
    thr = report.throughput
    thr.mean_mbps = summarize(mean_mbps)
    thr.peak_mbps = summarize(peak_mbps)
    thr.peak_mbps_by_window = {
        label: summarize(values).to_dict() for label, values in peak_by_window.items()
    }
    thr.burstiness_by_window = {
        label: summarize(values).to_dict() for label, values in burst_by_window.items()
    }

    # --- reliability ---
    rel = report.reliability
    rel.retransmission_rate_per_capture = summarize(retrans_capture)
    rel.retransmission_rate_per_flow = summarize(
        samples["retransmission_rate_per_flow"])
    rel.flows_with_retransmissions = flows_with_retrans
    if total_flows:
        rel.flow_retransmission_ratio = flows_with_retrans / total_flows

    report.capture_duration_s = summarize(capture_durations)
    return report
