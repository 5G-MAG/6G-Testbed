"""
Network profile data class for netemu.

Defines the NetworkProfile dataclass that represents network condition parameters
for emulation using Linux tc/netem.
"""

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Optional

import yaml

from .exceptions import ProfileLoadError


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
            - "gemodel": two-state Gilbert-Elliott model
            - "state": State-based loss model
            With loss_mean_burst_packets, loss_pct is the stationary mean.
            Without it, legacy gemodel/state positional probabilities apply.
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
    delay_ms: float = 0
    jitter_ms: float = 0
    delay_distribution: Optional[str] = None
    delay_correlation_pct: Optional[float] = None
    loss_pct: float = 0.0
    loss_correlation_pct: Optional[float] = None
    loss_model: Optional[str] = None
    rate_mbit: Optional[float] = None
    rate_ceil_mbit: Optional[float] = None
    rate_burst_kbit: Optional[float] = None
    rate_cburst_kbit: Optional[float] = None
    corruption_pct: float = 0.0
    corruption_correlation_pct: Optional[float] = None
    reorder_pct: float = 0.0
    reorder_correlation_pct: Optional[float] = None
    duplicate_pct: float = 0.0
    duplicate_correlation_pct: Optional[float] = None
    limit_packets: Optional[int] = None
    # Explicit calibration service model. None preserves legacy HTB behavior.
    rate_mode: Optional[str] = None
    netem_rate_mbit: Optional[float] = None
    burst_bytes: Optional[int] = None
    cburst_bytes: Optional[int] = None
    loss_mean_burst_packets: Optional[float] = None
    slot_distribution: Optional[str] = None
    slot_delay_ms: Optional[float] = None
    slot_jitter_ms: Optional[float] = None
    slot_max_packets: Optional[int] = None
    slot_max_bytes: Optional[int] = None
    bypass: bool = False
    # Base fields describe downlink. This optional complete profile is uplink.
    uplink: Optional["NetworkProfile"] = None

    def settings(self) -> dict:
        """Flat runtime parameters, excluding identity and directional metadata."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in {"name", "description", "uplink"}}

    def ge_probabilities(self) -> tuple[float, float]:
        """GOOD→BAD / BAD→GOOD percentages for 100% loss in the BAD state."""
        burst = self.loss_mean_burst_packets
        if burst is None or not math.isfinite(burst) or burst < 1:
            raise ValueError("loss_mean_burst_packets must be finite and >= 1")
        if not 0 < self.loss_pct < 100:
            raise ValueError("GE mean loss_pct must be between 0 and 100")
        leave = 1 / burst
        enter = (self.loss_pct / 100) * leave / (1 - self.loss_pct / 100)
        if enter > 1:
            raise ValueError("GE mean and burst imply a probability greater than 100%")
        return 100 * enter, 100 * leave

    def validate(self) -> None:
        """Reject invalid shaping inputs before changing any interfaces."""
        for key, value in self.settings().items():
            if value is None and key in {"delay_ms", "jitter_ms", "loss_pct", "corruption_pct",
                                         "reorder_pct", "duplicate_pct"}:
                raise ValueError(f"{key} must be a number")
            if value is None or key in {"rate_mode", "loss_model", "delay_distribution",
                                       "slot_distribution", "bypass"}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
            if key.endswith("_pct") and value > 100:
                raise ValueError(f"{key} must be <= 100")
            if key in {"rate_mbit", "rate_ceil_mbit", "netem_rate_mbit",
                       "rate_burst_kbit", "rate_cburst_kbit"} and value <= 0:
                raise ValueError(f"{key} must be positive")
            if key in {"limit_packets", "burst_bytes", "cburst_bytes",
                       "slot_max_packets", "slot_max_bytes"} and (
                    value < 1 or value != int(value) or value > 2**31 - 1):
                raise ValueError(f"{key} must be a positive 32-bit integer")
        if not isinstance(self.bypass, bool):
            raise ValueError("bypass must be boolean")
        if self.rate_mode not in {None, "netem", "htb", "htb_netem"}:
            raise ValueError("rate_mode must be netem, htb, or htb_netem")
        if self.rate_mode in {"htb", "htb_netem"} and self.rate_mbit is None:
            raise ValueError("HTB modes require rate_mbit")
        if self.rate_ceil_mbit is not None and (
                self.rate_mbit is None or self.rate_ceil_mbit < self.rate_mbit):
            raise ValueError("ceil_mbit must be >= rate_mbit")
        if self.netem_rate_mbit is not None and self.rate_mode != "htb_netem":
            raise ValueError("netem_rate_mbit requires htb_netem")
        if self.rate_mode == "netem" and any(v is not None for v in (
                self.rate_ceil_mbit, self.burst_bytes, self.cburst_bytes,
                self.rate_burst_kbit, self.rate_cburst_kbit)):
            raise ValueError("ceil and burst settings require HTB")
        if self.burst_bytes is not None and self.rate_burst_kbit is not None:
            raise ValueError("Specify burst_bytes or rate_burst_kbit, not both")
        if self.cburst_bytes is not None and self.rate_cburst_kbit is not None:
            raise ValueError("Specify cburst_bytes or rate_cburst_kbit, not both")
        if self.delay_distribution not in {None, "none", "fixed", "normal", "uniform",
                                          "pareto", "paretonormal"}:
            raise ValueError("Unsupported delay_distribution")
        if self.jitter_ms and (not self.delay_ms or self.delay_distribution == "fixed"):
            raise ValueError("Jitter requires positive delay and a non-fixed distribution")
        if self.reorder_pct and not self.delay_ms:
            raise ValueError("Reordering requires positive delay_ms")
        if self.loss_model not in {None, "random", "correlated", "gemodel", "state"}:
            raise ValueError("Unsupported loss_model")
        if self.loss_mean_burst_packets is not None:
            if self.loss_model != "gemodel" or self.loss_correlation_pct:
                raise ValueError("GE mean burst requires gemodel without loss correlation")
            self.ge_probabilities()
        if self.slot_distribution is not None:
            if self.slot_distribution not in {"normal", "uniform", "pareto", "paretonormal"}:
                raise ValueError("Unsupported slot_distribution")
            if not self.slot_delay_ms or not self.slot_jitter_ms:
                raise ValueError("Slot distribution requires positive delay and jitter")
        elif any(v is not None for v in (self.slot_delay_ms, self.slot_jitter_ms,
                                        self.slot_max_packets, self.slot_max_bytes)):
            raise ValueError("Slot fields require slot_distribution")
        if self.bypass and any((self.delay_ms, self.jitter_ms, self.loss_pct,
                               self.duplicate_pct, self.reorder_pct, self.corruption_pct,
                               self.rate_mbit, self.slot_distribution)):
            raise ValueError("bypass requires no impairments or rate cap")

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
        if not isinstance(data, dict):
            raise ValueError(f"profile '{name}' must be a mapping")
        data = dict(data)
        for alias, canonical in {"ceil_mbit": "rate_ceil_mbit",
                                 "limit_pkts": "limit_packets"}.items():
            if alias in data:
                if canonical in data and data[alias] != data[canonical]:
                    raise ValueError(f"Conflicting {alias} and {canonical}")
                data[canonical] = data.pop(alias)
        unknown = set(data) - {f.name for f in fields(cls) if f.name != "name"}
        if unknown:
            raise ValueError(f"profile '{name}': unknown fields {sorted(unknown)}")
        loss_model = data.get("loss_model")
        if isinstance(loss_model, str) and loss_model.lower() == "none":
            data["loss_model"] = None
        uplink = data.pop("uplink", None)
        profile = cls(name=name, **data)
        profile.validate()
        if uplink is not None:
            if not isinstance(uplink, dict) or "uplink" in uplink:
                raise ValueError(f"profile '{name}': uplink must be a flat mapping")
            if uplink:
                profile.uplink = cls.from_dict(name, {**data, **uplink})
        return profile


def load_profile_catalog(path: str | Path) -> dict:
    """Read legacy profiles or calibration v1 cases into a runtime catalog.

    Directional cases are paired under their existing CLI name. Base fields
    represent downlink; differing uplink fields are kept in an override block.
    Targets, workloads and fitting bounds are evidence, never runtime inputs.
    """
    try:
        with open(path) as handle:
            data = yaml.safe_load(handle)
        if not data:
            raise ValueError("empty file")
        if not isinstance(data, dict):
            raise ValueError("catalog must be a mapping")
        if "cases" in data:
            if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
                raise ValueError("unsupported calibration schema_version (expected 1)")
            if "profiles" in data:
                raise ValueError("catalog cannot contain both cases and profiles")
            cases = data["cases"]
            if not isinstance(cases, dict) or not cases:
                raise ValueError("no calibration cases defined")
            groups = {}
            for key, case in cases.items():
                if not isinstance(key, str) or not isinstance(case, dict):
                    raise ValueError("calibration cases must be named mappings")
                name, separator, direction = key.rpartition("/")
                if not separator or not name or direction not in {"uplink", "downlink"}:
                    raise ValueError(f"case '{key}' must end in /uplink or /downlink")
                parameters = case.get("parameters")
                if not isinstance(parameters, dict):
                    raise ValueError(f"case '{key}' requires parameters")
                if set(parameters) & {"name", "description", "uplink"}:
                    raise ValueError(f"case '{key}' parameters must contain only flat shaping fields")
                parameters = {"rate_mode": "netem", **parameters}
                if parameters.get("rate_mode") not in {"netem", "htb", "htb_netem"}:
                    raise ValueError(f"case '{key}': invalid rate_mode")
                if parameters.get("loss_model") == "gemodel" and parameters.get("loss_mean_burst_packets") is None:
                    raise ValueError(f"case '{key}': gemodel requires loss_mean_burst_packets")
                profile = NetworkProfile.from_dict(key, parameters)
                groups.setdefault(name, {})[direction] = profile.settings()
            profiles = {}
            for name, pair in groups.items():
                if set(pair) != {"uplink", "downlink"}:
                    raise ValueError(f"profile '{name}' requires both uplink and downlink cases")
                base = {"description": name, **pair["downlink"]}
                overrides = {k: v for k, v in pair["uplink"].items()
                             if v != pair["downlink"][k]}
                if overrides:
                    base["uplink"] = overrides
                profiles[name] = base
        else:
            profiles = data.get("profiles")
            if not isinstance(profiles, dict) or not profiles:
                raise ValueError("no profiles defined")
        # Validate the whole file before making any profile visible to callers.
        for name, spec in profiles.items():
            if not isinstance(name, str) or not name:
                raise ValueError("profile names must be nonempty strings")
            NetworkProfile.from_dict(name, spec)
        return {**data, "profiles": profiles}
    except (OSError, yaml.YAMLError, ValueError, TypeError) as exc:
        reason = "file not found" if isinstance(exc, FileNotFoundError) else str(exc)
        if isinstance(exc, yaml.YAMLError):
            reason = f"invalid YAML: {exc}"
        raise ProfileLoadError(str(path), reason) from exc
