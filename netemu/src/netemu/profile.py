"""
Network profile data class for netemu.

Defines the NetworkProfile dataclass that represents network condition parameters
for emulation using Linux tc/netem.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NetworkProfile:
    """
    Network profile configuration for tc/netem emulation.

    Attributes:
        name: Unique identifier for the profile.
        description: Human-readable description of network conditions.
        delay_ms: One-way latency in milliseconds (default: 0).
        jitter_ms: Delay variation in milliseconds (default: 0).
        delay_distribution: Statistical distribution for delay variation.
            Options: "normal", "pareto", "paretonormal", "uniform".
            Requires jitter_ms > 0 to take effect.
        delay_correlation_pct: Correlation percentage for successive delays (0-100).
            Higher values make delays more similar to previous values.
        loss_pct: Packet loss percentage (0-100, default: 0.0).
        loss_correlation_pct: Correlation percentage for successive losses (0-100).
            Higher values create bursty loss patterns.
        loss_model: Advanced loss model. Options:
            - "gemodel": Gilbert-Elliot model with 4-state Markov chain
            - "state": State-based loss model
            When set, loss_pct serves as the base probability.
        rate_mbit: Bandwidth limit in Mbps. None means unlimited.
        rate_ceil_mbit: Maximum burst rate in Mbps (HTB ceil parameter).
            Defaults to rate_mbit if not specified.
        rate_burst_kbit: Burst buffer size in kbits for rate limiting.
        rate_cburst_kbit: Ceil burst buffer size in kbits.
        corruption_pct: Single-bit error percentage (0-100, default: 0.0).
        corruption_correlation_pct: Correlation for successive corruptions.
        reorder_pct: Packet reordering percentage (0-100, default: 0.0).
            Requires delay_ms > 0 to observe reordering effects.
        reorder_correlation_pct: Correlation for successive reorderings.
        duplicate_pct: Packet duplication percentage (0-100, default: 0.0).
        duplicate_correlation_pct: Correlation for successive duplications.
        limit_packets: Queue limit in packets. Controls how many packets
            can be queued before drops occur. None uses system default.
    """

    name: str
    description: str = ""
    delay_ms: int = 0
    jitter_ms: int = 0
    delay_distribution: Optional[str] = None
    delay_correlation_pct: Optional[float] = None
    loss_pct: float = 0.0
    loss_correlation_pct: Optional[float] = None
    loss_model: Optional[str] = None
    rate_mbit: Optional[int] = None
    rate_ceil_mbit: Optional[int] = None
    rate_burst_kbit: Optional[int] = None
    rate_cburst_kbit: Optional[int] = None
    corruption_pct: float = 0.0
    corruption_correlation_pct: Optional[float] = None
    reorder_pct: float = 0.0
    reorder_correlation_pct: Optional[float] = None
    duplicate_pct: float = 0.0
    duplicate_correlation_pct: Optional[float] = None
    limit_packets: Optional[int] = None

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "NetworkProfile":
        """
        Create a NetworkProfile from a dictionary.

        Args:
            name: Profile name/identifier.
            data: Dictionary containing profile parameters.

        Returns:
            NetworkProfile instance with the specified parameters.

        Example:
            >>> profile = NetworkProfile.from_dict("slow", {"delay_ms": 100, "loss_pct": 1.0})
            >>> profile.delay_ms
            100
        """
        return cls(
            name=name,
            description=data.get("description", ""),
            delay_ms=data.get("delay_ms", 0),
            jitter_ms=data.get("jitter_ms", 0),
            delay_distribution=data.get("delay_distribution"),
            delay_correlation_pct=data.get("delay_correlation_pct"),
            loss_pct=data.get("loss_pct", 0.0),
            loss_correlation_pct=data.get("loss_correlation_pct"),
            loss_model=data.get("loss_model"),
            rate_mbit=data.get("rate_mbit"),
            rate_ceil_mbit=data.get("rate_ceil_mbit"),
            rate_burst_kbit=data.get("rate_burst_kbit"),
            rate_cburst_kbit=data.get("rate_cburst_kbit"),
            corruption_pct=data.get("corruption_pct", 0.0),
            corruption_correlation_pct=data.get("corruption_correlation_pct"),
            reorder_pct=data.get("reorder_pct", 0.0),
            reorder_correlation_pct=data.get("reorder_correlation_pct"),
            duplicate_pct=data.get("duplicate_pct", 0.0),
            duplicate_correlation_pct=data.get("duplicate_correlation_pct"),
            limit_packets=data.get("limit_packets"),
        )
