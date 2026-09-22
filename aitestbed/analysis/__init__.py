"""
Analysis module for the 6G AI Traffic Testbed.

Provides logging, application-layer metrics computation, the RAN2 (S4-260859)
metric families, and visualization.

Network-layer pcap parsing lives in the sibling `netemu` package
(`netemu.pcap`); the names below are re-exported for convenience so testbed
code can keep importing them from `analysis`.
"""

from .logger import TrafficLogger, LogRecord
from .metrics import MetricsCalculator, ScenarioMetrics
from .visualize import TrafficVisualizer

# Optional pcap analysis (netemu[pcap], i.e. requires dpkt)
try:
    from netemu.pcap import (
        PcapAnalyzer,
        PcapMetrics,
        PacketRecord,
        TCPFlow,
        analyze_pcap,
        analyze_multiple_pcaps,
        merge_pcap_metrics,
    )
    HAS_PCAP_ANALYZER = True
except ImportError:
    HAS_PCAP_ANALYZER = False
    PcapAnalyzer = None
    PcapMetrics = None
    PacketRecord = None
    TCPFlow = None
    analyze_pcap = None
    analyze_multiple_pcaps = None
    merge_pcap_metrics = None

# Captures written by capture/ mark loopback pcaps with this filename prefix.
# The port filter that keeps WAN captures clean must not be applied to them,
# because local MCP/A2A/gateway servers bind ports that are not known when the
# filter is configured. Pass this to netemu's analyzer entry points.
LOOPBACK_PCAP_NAME_PATTERNS = ("capture_lo_",)

__all__ = [
    "TrafficLogger",
    "LogRecord",
    "MetricsCalculator",
    "ScenarioMetrics",
    "TrafficVisualizer",
    # Pcap analysis (optional, re-exported from netemu.pcap)
    "HAS_PCAP_ANALYZER",
    "LOOPBACK_PCAP_NAME_PATTERNS",
    "PcapAnalyzer",
    "PcapMetrics",
    "PacketRecord",
    "TCPFlow",
    "analyze_pcap",
    "analyze_multiple_pcaps",
    "merge_pcap_metrics",
]
