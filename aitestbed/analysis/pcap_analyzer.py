"""
PCAP Analyzer for the 6G AI Traffic Testbed.

Extracts network-layer metrics from pcap files for accurate throughput,
latency, and packet statistics that complement application-layer metrics.
"""

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import dpkt for pcap parsing
try:
    import dpkt
    HAS_DPKT = True
except ImportError:
    HAS_DPKT = False
    logger.warning("dpkt not installed. Install with: pip install dpkt")


@dataclass
class TCPFlow:
    """Represents a TCP flow (connection) with metrics."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int

    # Packet counts
    packets_sent: int = 0
    packets_recv: int = 0

    # Byte counts
    bytes_sent: int = 0
    bytes_recv: int = 0

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # TCP-specific metrics
    syn_time: Optional[float] = None
    syn_ack_time: Optional[float] = None
    ack_time: Optional[float] = None
    retransmissions: int = 0

    # Sequence tracking for retransmission detection
    seen_seqs: set = field(default_factory=set)

    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        if self.end_time > self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def handshake_rtt(self) -> Optional[float]:
        """TCP handshake RTT (SYN to ACK) in seconds."""
        if self.syn_time and self.ack_time:
            return self.ack_time - self.syn_time
        return None

    @property
    def syn_ack_rtt(self) -> Optional[float]:
        """SYN to SYN-ACK RTT in seconds (server response time)."""
        if self.syn_time and self.syn_ack_time:
            return self.syn_ack_time - self.syn_time
        return None

    @property
    def throughput_bps(self) -> float:
        """Average throughput in bits per second."""
        if self.duration > 0:
            return (self.bytes_sent + self.bytes_recv) * 8 / self.duration
        return 0.0

    @property
    def ul_throughput_bps(self) -> float:
        """Uplink throughput in bits per second."""
        if self.duration > 0:
            return self.bytes_sent * 8 / self.duration
        return 0.0

    @property
    def dl_throughput_bps(self) -> float:
        """Downlink throughput in bits per second."""
        if self.duration > 0:
            return self.bytes_recv * 8 / self.duration
        return 0.0

    @property
    def retransmission_rate(self) -> float:
        """Retransmission rate as fraction of total packets."""
        total = self.packets_sent + self.packets_recv
        if total > 0:
            return self.retransmissions / total
        return 0.0

    @property
    def flow_key(self) -> str:
        """Unique flow identifier."""
        return f"{self.src_ip}:{self.src_port}-{self.dst_ip}:{self.dst_port}"


@dataclass
class PcapMetrics:
    """Aggregate metrics from a pcap file."""
    pcap_file: str
    capture_duration: float = 0.0

    # Packet statistics
    total_packets: int = 0
    tcp_packets: int = 0
    udp_packets: int = 0
    other_packets: int = 0

    # Byte statistics
    total_bytes: int = 0
    tcp_bytes: int = 0
    udp_bytes: int = 0

    # Flow statistics
    tcp_flows: int = 0
    udp_flows: int = 0

    # Throughput (aggregate)
    avg_throughput_mbps: float = 0.0
    peak_throughput_mbps: float = 0.0

    # RTT statistics (from TCP handshakes)
    rtt_samples: list[float] = field(default_factory=list)
    rtt_mean_ms: Optional[float] = None
    rtt_min_ms: Optional[float] = None
    rtt_max_ms: Optional[float] = None
    rtt_p95_ms: Optional[float] = None

    # Retransmission statistics
    total_retransmissions: int = 0
    retransmission_rate: float = 0.0

    # Time series data for plotting
    throughput_timeseries: list[tuple[float, float, float]] = field(default_factory=list)
    # List of (timestamp, ul_kbps, dl_kbps)

    # Per-flow data
    flows: list[TCPFlow] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pcap_file": self.pcap_file,
            "capture_duration": self.capture_duration,
            "total_packets": self.total_packets,
            "tcp_packets": self.tcp_packets,
            "udp_packets": self.udp_packets,
            "total_bytes": self.total_bytes,
            "tcp_flows": self.tcp_flows,
            "avg_throughput_mbps": self.avg_throughput_mbps,
            "peak_throughput_mbps": self.peak_throughput_mbps,
            "rtt_mean_ms": self.rtt_mean_ms,
            "rtt_min_ms": self.rtt_min_ms,
            "rtt_max_ms": self.rtt_max_ms,
            "rtt_p95_ms": self.rtt_p95_ms,
            "total_retransmissions": self.total_retransmissions,
            "retransmission_rate": self.retransmission_rate,
        }


class PcapAnalyzer:
    """
    Analyzer for extracting network-layer metrics from pcap files.
    """

    def __init__(self, target_ports: Optional[list[int]] = None):
        """
        Initialize the analyzer.

        Args:
            target_ports: List of ports to filter (e.g., [443, 80]).
                         If None, all ports are analyzed.
        """
        if not HAS_DPKT:
            raise ImportError("dpkt is required for pcap analysis. Install with: pip install dpkt")

        self.target_ports = set(target_ports) if target_ports else None

    def analyze(self, pcap_path: str, bucket_sec: float = 1.0) -> PcapMetrics:
        """
        Analyze a pcap file and extract metrics.

        Args:
            pcap_path: Path to the pcap file.
            bucket_sec: Time bucket size for throughput time series.

        Returns:
            PcapMetrics with extracted data.
        """
        pcap_path = Path(pcap_path)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        metrics = PcapMetrics(pcap_file=str(pcap_path))
        flows: dict[str, TCPFlow] = {}

        # Time series tracking
        throughput_buckets: dict[int, dict] = defaultdict(
            lambda: {"ul_bytes": 0, "dl_bytes": 0}
        )

        first_ts = None
        last_ts = None

        try:
            with open(pcap_path, 'rb') as f:
                try:
                    pcap = dpkt.pcap.Reader(f)
                except ValueError:
                    # Try pcapng format
                    f.seek(0)
                    pcap = dpkt.pcapng.Reader(f)

                for ts, buf in pcap:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

                    metrics.total_packets += 1
                    metrics.total_bytes += len(buf)

                    # Parse Ethernet frame
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                        continue

                    # Handle IP packets
                    if isinstance(eth.data, dpkt.ip.IP):
                        ip = eth.data
                        src_ip = self._ip_to_str(ip.src)
                        dst_ip = self._ip_to_str(ip.dst)

                        if isinstance(ip.data, dpkt.tcp.TCP):
                            tcp = ip.data
                            metrics.tcp_packets += 1
                            metrics.tcp_bytes += len(buf)

                            # Filter by port if specified
                            if self.target_ports:
                                if tcp.sport not in self.target_ports and tcp.dport not in self.target_ports:
                                    continue

                            # Process TCP flow
                            flow = self._process_tcp_packet(
                                flows, ts, src_ip, dst_ip, tcp, len(ip.data)
                            )

                            # Track throughput by time bucket
                            bucket = int(ts / bucket_sec)
                            # Determine direction (simple heuristic: lower port is server)
                            if tcp.sport < tcp.dport:
                                throughput_buckets[bucket]["dl_bytes"] += len(ip.data)
                            else:
                                throughput_buckets[bucket]["ul_bytes"] += len(ip.data)

                        elif isinstance(ip.data, dpkt.udp.UDP):
                            metrics.udp_packets += 1
                            metrics.udp_bytes += len(buf)

                            udp = ip.data
                            # Track UDP throughput
                            if self.target_ports is None or udp.sport in self.target_ports or udp.dport in self.target_ports:
                                bucket = int(ts / bucket_sec)
                                if udp.sport < udp.dport:
                                    throughput_buckets[bucket]["dl_bytes"] += len(ip.data)
                                else:
                                    throughput_buckets[bucket]["ul_bytes"] += len(ip.data)
                        else:
                            metrics.other_packets += 1

                    elif isinstance(eth.data, dpkt.ip6.IP6):
                        # IPv6 support
                        ip6 = eth.data
                        if isinstance(ip6.data, dpkt.tcp.TCP):
                            metrics.tcp_packets += 1
                            metrics.tcp_bytes += len(buf)
                        elif isinstance(ip6.data, dpkt.udp.UDP):
                            metrics.udp_packets += 1
                            metrics.udp_bytes += len(buf)

        except Exception as e:
            logger.error(f"Error parsing pcap file {pcap_path}: {e}")
            raise

        # Calculate duration
        if first_ts and last_ts:
            metrics.capture_duration = last_ts - first_ts

        # Process flows
        metrics.flows = list(flows.values())
        metrics.tcp_flows = len([f for f in metrics.flows if f.packets_sent > 0 or f.packets_recv > 0])

        # Calculate RTT statistics from handshakes
        rtt_samples = []
        for flow in metrics.flows:
            if flow.handshake_rtt:
                rtt_samples.append(flow.handshake_rtt * 1000)  # Convert to ms

        if rtt_samples:
            metrics.rtt_samples = rtt_samples
            metrics.rtt_mean_ms = sum(rtt_samples) / len(rtt_samples)
            metrics.rtt_min_ms = min(rtt_samples)
            metrics.rtt_max_ms = max(rtt_samples)
            sorted_rtt = sorted(rtt_samples)
            p95_idx = int(len(sorted_rtt) * 0.95)
            metrics.rtt_p95_ms = sorted_rtt[min(p95_idx, len(sorted_rtt) - 1)]

        # Calculate retransmission statistics
        total_retrans = sum(f.retransmissions for f in metrics.flows)
        metrics.total_retransmissions = total_retrans
        if metrics.tcp_packets > 0:
            metrics.retransmission_rate = total_retrans / metrics.tcp_packets

        # Calculate throughput statistics
        if metrics.capture_duration > 0:
            metrics.avg_throughput_mbps = (metrics.total_bytes * 8) / metrics.capture_duration / 1_000_000

        # Build throughput time series
        if throughput_buckets and first_ts:
            sorted_buckets = sorted(throughput_buckets.keys())
            peak_throughput = 0.0

            for bucket in sorted_buckets:
                data = throughput_buckets[bucket]
                rel_time = (bucket * bucket_sec) - first_ts
                ul_kbps = (data["ul_bytes"] * 8) / bucket_sec / 1000
                dl_kbps = (data["dl_bytes"] * 8) / bucket_sec / 1000
                metrics.throughput_timeseries.append((rel_time, ul_kbps, dl_kbps))
                peak_throughput = max(peak_throughput, ul_kbps + dl_kbps)

            metrics.peak_throughput_mbps = peak_throughput / 1000

        return metrics

    def _process_tcp_packet(
        self,
        flows: dict[str, TCPFlow],
        ts: float,
        src_ip: str,
        dst_ip: str,
        tcp: 'dpkt.tcp.TCP',
        payload_len: int
    ) -> TCPFlow:
        """Process a TCP packet and update flow state."""
        # Create canonical flow key (sorted by IP:port to handle bidirectional)
        forward_key = f"{src_ip}:{tcp.sport}-{dst_ip}:{tcp.dport}"
        reverse_key = f"{dst_ip}:{tcp.dport}-{src_ip}:{tcp.sport}"

        # Check if flow exists in either direction
        if forward_key in flows:
            flow = flows[forward_key]
            is_forward = True
        elif reverse_key in flows:
            flow = flows[reverse_key]
            is_forward = False
        else:
            # New flow
            flow = TCPFlow(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=tcp.sport,
                dst_port=tcp.dport,
                start_time=ts
            )
            flows[forward_key] = flow
            is_forward = True

        # Update timing
        flow.end_time = ts

        # Update packet/byte counts
        if is_forward:
            flow.packets_sent += 1
            flow.bytes_sent += payload_len
        else:
            flow.packets_recv += 1
            flow.bytes_recv += payload_len

        # Track TCP flags for handshake RTT
        flags = tcp.flags

        # SYN (no ACK) - connection initiation
        if (flags & dpkt.tcp.TH_SYN) and not (flags & dpkt.tcp.TH_ACK):
            if flow.syn_time is None:
                flow.syn_time = ts

        # SYN-ACK - server response
        elif (flags & dpkt.tcp.TH_SYN) and (flags & dpkt.tcp.TH_ACK):
            if flow.syn_ack_time is None:
                flow.syn_ack_time = ts

        # ACK (completing handshake)
        elif (flags & dpkt.tcp.TH_ACK) and flow.syn_ack_time and not flow.ack_time:
            flow.ack_time = ts

        # Detect retransmissions (simplified: same seq number seen before)
        seq = tcp.seq
        if seq in flow.seen_seqs and payload_len > 0:
            flow.retransmissions += 1
        flow.seen_seqs.add(seq)

        return flow

    @staticmethod
    def _ip_to_str(ip_bytes: bytes) -> str:
        """Convert IP address bytes to string."""
        if len(ip_bytes) == 4:
            return '.'.join(str(b) for b in ip_bytes)
        elif len(ip_bytes) == 16:
            # IPv6
            return ':'.join(f'{ip_bytes[i]:02x}{ip_bytes[i+1]:02x}' for i in range(0, 16, 2))
        return str(ip_bytes)


def analyze_pcap(pcap_path: str, target_ports: Optional[list[int]] = None) -> PcapMetrics:
    """
    Convenience function to analyze a pcap file.

    Args:
        pcap_path: Path to pcap file.
        target_ports: Optional list of ports to filter.

    Returns:
        PcapMetrics with network-layer statistics.
    """
    analyzer = PcapAnalyzer(target_ports=target_ports)
    return analyzer.analyze(pcap_path)


def analyze_multiple_pcaps(
    pcap_dir: str,
    pattern: str = "*.pcap",
    target_ports: Optional[list[int]] = None
) -> list[PcapMetrics]:
    """
    Analyze multiple pcap files in a directory.

    Args:
        pcap_dir: Directory containing pcap files.
        pattern: Glob pattern for pcap files.
        target_ports: Optional list of ports to filter.

    Returns:
        List of PcapMetrics for each file.
    """
    pcap_dir = Path(pcap_dir)
    results = []

    for pcap_file in pcap_dir.glob(pattern):
        try:
            metrics = analyze_pcap(str(pcap_file), target_ports)
            results.append(metrics)
            logger.info(f"Analyzed {pcap_file.name}: {metrics.total_packets} packets, "
                       f"{metrics.tcp_flows} TCP flows")
        except Exception as e:
            logger.error(f"Failed to analyze {pcap_file}: {e}")

    return results


def merge_pcap_metrics(metrics_list: list[PcapMetrics]) -> dict:
    """
    Merge multiple PcapMetrics into aggregate statistics.

    Returns:
        Dictionary with aggregate statistics.
    """
    if not metrics_list:
        return {}

    total_packets = sum(m.total_packets for m in metrics_list)
    total_bytes = sum(m.total_bytes for m in metrics_list)
    total_duration = sum(m.capture_duration for m in metrics_list)

    # Merge RTT samples
    all_rtt = []
    for m in metrics_list:
        all_rtt.extend(m.rtt_samples)

    # Merge throughput time series
    all_throughput = []
    for m in metrics_list:
        all_throughput.extend(m.throughput_timeseries)

    result = {
        "total_captures": len(metrics_list),
        "total_packets": total_packets,
        "total_bytes": total_bytes,
        "total_duration_sec": total_duration,
        "total_tcp_flows": sum(m.tcp_flows for m in metrics_list),
        "total_retransmissions": sum(m.total_retransmissions for m in metrics_list),
    }

    if total_duration > 0:
        result["avg_throughput_mbps"] = (total_bytes * 8) / total_duration / 1_000_000

    if all_rtt:
        result["rtt_mean_ms"] = sum(all_rtt) / len(all_rtt)
        result["rtt_min_ms"] = min(all_rtt)
        result["rtt_max_ms"] = max(all_rtt)
        sorted_rtt = sorted(all_rtt)
        p95_idx = int(len(sorted_rtt) * 0.95)
        result["rtt_p95_ms"] = sorted_rtt[min(p95_idx, len(sorted_rtt) - 1)]

    if total_packets > 0:
        result["retransmission_rate"] = result["total_retransmissions"] / total_packets

    result["throughput_timeseries"] = sorted(all_throughput, key=lambda x: x[0])

    return result
