"""
Analysis module for the 6G AI Traffic Testbed.

Provides logging, metrics computation, and visualization tools.
"""

from .logger import TrafficLogger, LogRecord
from .metrics import MetricsCalculator, ScenarioMetrics
from .visualize import TrafficVisualizer

# Optional pcap analysis (requires dpkt)
try:
    from .pcap_analyzer import (
        PcapAnalyzer,
        PcapMetrics,
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
    TCPFlow = None
    analyze_pcap = None
    analyze_multiple_pcaps = None
    merge_pcap_metrics = None

__all__ = [
    "TrafficLogger",
    "LogRecord",
    "MetricsCalculator",
    "ScenarioMetrics",
    "TrafficVisualizer",
    # Pcap analysis (optional)
    "HAS_PCAP_ANALYZER",
    "PcapAnalyzer",
    "PcapMetrics",
    "TCPFlow",
    "analyze_pcap",
    "analyze_multiple_pcaps",
    "merge_pcap_metrics",
]
