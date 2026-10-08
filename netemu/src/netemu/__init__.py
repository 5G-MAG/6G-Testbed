"""
netemu - Linux tc/netem network emulation, packet capture, and pcap metrics.

The package covers the full measurement loop for a shaped network:

* **Emulate** - :class:`NetworkEmulator` drives tc/netem/HTB to impose delay,
  jitter, loss, rate limits, corruption, reordering, and duplication, on
  egress and (via IFB) on ingress.
* **Capture** - :class:`CaptureController` runs tcpdump on the shaped
  interface and writes a pcap plus a metadata sidecar.
* **Analyze** - :class:`PcapAnalyzer` parses that pcap back into
  network-layer metrics: throughput, RTT, retransmissions, TLS setup,
  per-direction volumes, burstiness, and burst segmentation.
* **Measure** - :func:`compute_packet_metrics` aggregates one or many parsed
  captures into distributions: connection setup, flow lifetime, per-direction
  volume, burst structure, windowed throughput, and retransmissions.

Emulating:
    >>> from netemu import NetworkEmulator
    >>> emulator = NetworkEmulator(interface="eth0")
    >>> emulator.apply_settings(delay_ms=100, loss_pct=1.0)
    True
    >>> emulator.clear()
    True

Using profiles:
    >>> emulator = NetworkEmulator(profiles_path="profiles.yaml")
    >>> emulator.apply_profile("poor_cellular")
    True

Capturing and analyzing:
    >>> from netemu import capture_to, analyze_pcap
    >>> with NetworkEmulator(interface="eth0") as emu:      # doctest: +SKIP
    ...     emu.apply_profile("cell_edge")
    ...     with capture_to("run.pcap", interface="eth0", filter_expr="port 443"):
    ...         run_workload()
    >>> metrics = analyze_pcap("run.pcap")                   # doctest: +SKIP
    >>> metrics.rtt_mean_ms                                  # doctest: +SKIP
    123.4

Aggregating a campaign:
    >>> from netemu import analyze_multiple_pcaps, compute_packet_metrics
    >>> report = compute_packet_metrics(                      # doctest: +SKIP
    ...     analyze_multiple_pcaps("captures/"))
    >>> report.connection.connection_setup_ms.p95             # doctest: +SKIP
    4248.5

Parsing a pcap needs the optional ``dpkt`` dependency
(``pip install "netemu[pcap]"``). Importing this package without it succeeds;
:data:`HAS_DPKT` is then False and constructing a :class:`PcapAnalyzer` raises
:class:`DpktNotAvailableError`. :func:`compute_packet_metrics` consumes captures
that are already parsed, so it is available either way.
"""

from .capture import CaptureController, capture_to, start_capture, stop_capture
from .emulator import NetworkEmulator, apply_profile, clear_profile
from .exceptions import (
    CommandFailedError,
    DpktNotAvailableError,
    NetEmuError,
    ProfileLoadError,
    ProfileNotFoundError,
    SudoNotAvailableError,
)
from .metrics import (
    BurstMetrics,
    ConnectionMetrics,
    DirectionMetrics,
    Distribution,
    PacketMetricsReport,
    ReliabilityMetrics,
    ThroughputMetrics,
    collect_connection_samples,
    compute_packet_metrics,
    summarize,
)
from .profile import NetworkProfile

# Optional pcap analysis (requires dpkt).
try:
    from .pcap import (
        HAS_DPKT,
        PacketRecord,
        PcapAnalyzer,
        PcapMetrics,
        TCPFlow,
        analyze_multiple_pcaps,
        analyze_pcap,
        merge_pcap_metrics,
    )
except ImportError:  # pragma: no cover - defensive; netemu.pcap imports cleanly
    HAS_DPKT = False
    PacketRecord = None
    PcapAnalyzer = None
    PcapMetrics = None
    TCPFlow = None
    analyze_multiple_pcaps = None
    analyze_pcap = None
    merge_pcap_metrics = None

__version__ = "0.3.0"

__all__ = [
    # Emulation
    "NetworkEmulator",
    "NetworkProfile",
    "apply_profile",
    "clear_profile",
    # Capture
    "CaptureController",
    "capture_to",
    "start_capture",
    "stop_capture",
    # Pcap analysis (optional, requires dpkt)
    "HAS_DPKT",
    "PcapAnalyzer",
    "PcapMetrics",
    "PacketRecord",
    "TCPFlow",
    "analyze_pcap",
    "analyze_multiple_pcaps",
    "merge_pcap_metrics",
    # Aggregate packet metrics (no dpkt needed: consumes parsed captures)
    "compute_packet_metrics",
    "collect_connection_samples",
    "summarize",
    "PacketMetricsReport",
    "ConnectionMetrics",
    "DirectionMetrics",
    "BurstMetrics",
    "ThroughputMetrics",
    "ReliabilityMetrics",
    "Distribution",
    # Exceptions
    "NetEmuError",
    "SudoNotAvailableError",
    "ProfileNotFoundError",
    "CommandFailedError",
    "ProfileLoadError",
    "DpktNotAvailableError",
    # Version
    "__version__",
]
