"""
Network Emulator using Linux tc/netem.

Provides the NetworkEmulator class for applying network condition emulation
(latency, jitter, packet loss, bandwidth limits, etc.) to network interfaces.
"""

import logging
import subprocess
from typing import Optional, Sequence

from .exceptions import ProfileNotFoundError
from .profile import NetworkProfile, load_profile_catalog
from .tc import netem_parameters, shaping_commands

logger = logging.getLogger(__name__)


def _detect_default_interface() -> str:
    """Detect the default network interface from the routing table.

    Parses ``ip route show default`` to find the device used for the
    default route.  Falls back to ``eth0`` if detection fails.
    """
    try:
        result = subprocess.run(
            ["ip", "-o", "route", "show", "default"],
            capture_output=True, text=True, timeout=5,
        )
        # Example output: "default via 172.25.240.1 dev eth1 proto ..."
        for token_idx, token in enumerate(result.stdout.split()):
            if token == "dev" and token_idx + 1 < len(result.stdout.split()):
                iface = result.stdout.split()[token_idx + 1]
                logger.info(f"Auto-detected default interface: {iface}")
                return iface
    except Exception as e:
        logger.warning(f"Failed to detect default interface: {e}")
    return "eth0"


class NetworkEmulator:
    """
    Network condition emulator using Linux tc/netem.

    Requires sudo access for tc commands. Supports bidirectional shaping
    using IFB (Intermediate Functional Block) device for ingress traffic.

    Example:
        >>> emulator = NetworkEmulator(interface="eth0", profiles_path="profiles.yaml")
        >>> emulator.apply_profile("poor_cellular")
        True
        >>> # ... run tests ...
        >>> emulator.clear()
        True

    Context manager usage:
        >>> with NetworkEmulator(interface="eth0") as emu:
        ...     emu.apply_settings(delay_ms=100, loss_pct=1.0)
        ...     # Rules are automatically cleared on exit
    """

    def __init__(
        self,
        interface: str = "eth0",
        profiles_path: Optional[str] = None,
        bidirectional: bool = True,
        ifb_device: str = "ifb0",
    ):
        """
        Initialize the network emulator.

        Args:
            interface: Network interface to apply rules to. Use "auto" to
                detect the default interface automatically.
            profiles_path: Path to profiles YAML file. If provided, profiles
                are loaded automatically.
            bidirectional: If True, shape both egress and ingress traffic.
            ifb_device: IFB device name for ingress shaping.
        """
        self._interface_from_config = interface == "auto"
        if interface == "auto":
            interface = _detect_default_interface()
        self.interface = interface
        self.profiles: dict[str, NetworkProfile] = {}
        self.current_profile: Optional[str] = None
        self.current_ingress_profile: Optional[str] = None
        self._sudo_available: Optional[bool] = None
        self.bidirectional = bidirectional
        self.ifb_device = ifb_device
        self._ifb_initialized = False
        self._lo_port: Optional[int] = None
        self._lo_selectors: tuple[tuple[str, Optional[int]], ...] = ()

        if profiles_path:
            self.load_profiles(profiles_path)

    def __enter__(self) -> "NetworkEmulator":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, clearing all rules."""
        self.clear()

    def load_profiles(self, path: str) -> None:
        """
        Load network profiles from YAML file.

        Args:
            path: Path to YAML file containing profile definitions.

        Raises:
            ProfileLoadError: If file cannot be read or parsed.
        """
        data = load_profile_catalog(path)
        profiles = {name: NetworkProfile.from_dict(name, config)
                    for name, config in data["profiles"].items()}
        if "default_interface" in data and self._interface_from_config:
            iface = data["default_interface"]
            self.interface = _detect_default_interface() if iface == "auto" else iface
        self.profiles = profiles
        logger.info(f"Loaded {len(self.profiles)} network profiles from {path}")

    def check_sudo(self) -> bool:
        """
        Check if sudo is available without a password for the commands
        netemu actually runs.

        Probes ``sudo -n true`` first (blanket passwordless sudo), then
        falls back to ``sudo -n tc qdisc show`` so that hosts with
        command-scoped NOPASSWD rules (e.g. an /etc/sudoers.d entry listing
        only tc/ip/tcpdump) are detected correctly. Probing only ``true``
        reports False on such hosts even though every command netemu issues
        would succeed.

        Returns:
            True if passwordless sudo covers netemu's commands.
        """
        if self._sudo_available is not None:
            return self._sudo_available

        self._sudo_available = False
        for probe in (["sudo", "-n", "true"],
                      ["sudo", "-n", "tc", "qdisc", "show"]):
            try:
                result = subprocess.run(probe, capture_output=True, timeout=5)
                if result.returncode == 0:
                    self._sudo_available = True
                    break
            except Exception:
                continue

        return self._sudo_available

    def _setup_ifb(self) -> bool:
        """
        Set up IFB device for ingress shaping.

        Creates the IFB device and sets up ingress redirection from
        the main interface to the IFB device.

        Returns:
            True if successful, False otherwise.
        """
        if self._ifb_initialized:
            return True

        # Load the IFB kernel module
        cmd_modprobe = "sudo modprobe ifb numifbs=1"
        if not self._run_tc_command(cmd_modprobe, ignore_errors=True):
            logger.warning("Failed to load ifb module (may already be loaded)")

        # Bring up the IFB device
        cmd_link_up = f"sudo ip link set dev {self.ifb_device} up"
        if not self._run_tc_command(cmd_link_up):
            logger.error(f"Failed to bring up {self.ifb_device}")
            return False

        # Add ingress qdisc to the main interface
        cmd_ingress = f"sudo tc qdisc add dev {self.interface} ingress"
        if not self._run_tc_command(cmd_ingress):
            logger.error(f"Failed to add ingress qdisc to {self.interface}")
            return False

        # Redirect all ingress traffic to the IFB device
        cmd_redirect = (
            f"sudo tc filter add dev {self.interface} parent ffff: "
            f"protocol all u32 match u32 0 0 "
            f"action mirred egress redirect dev {self.ifb_device}"
        )
        if not self._run_tc_command(cmd_redirect):
            logger.error("Failed to set up ingress redirect to IFB")
            return False

        self._ifb_initialized = True
        logger.info(f"IFB device {self.ifb_device} initialized for ingress shaping")
        return True

    def _teardown_ifb(self) -> bool:
        """
        Tear down IFB device and ingress redirection.

        Returns:
            True if successful.
        """
        if not self._ifb_initialized:
            return True

        # Remove ingress qdisc (this also removes the filter)
        cmd_del_ingress = f"sudo tc qdisc del dev {self.interface} ingress"
        self._run_tc_command(cmd_del_ingress, ignore_errors=True)

        # Clear IFB device rules
        cmd_del_ifb = f"sudo tc qdisc del dev {self.ifb_device} root"
        self._run_tc_command(cmd_del_ifb, ignore_errors=True)

        # Bring down IFB device
        cmd_link_down = f"sudo ip link set dev {self.ifb_device} down"
        self._run_tc_command(cmd_link_down, ignore_errors=True)

        self._ifb_initialized = False
        logger.info(f"IFB device {self.ifb_device} torn down")
        return True

    def apply_profile(
        self, profile_name: str, ingress_profile: Optional[str] = None
    ) -> bool:
        """Apply uplink on egress and downlink on ingress for a named profile.

        Legacy symmetric profiles are mirrored. An explicit ingress profile
        selects its downlink fields; "none" disables ingress shaping.
        """
        profile = self.get_profile(profile_name)
        if profile is None:
            raise ProfileNotFoundError(profile_name)
        ingress = None
        if ingress_profile not in (None, "none"):
            ingress = self.get_profile(ingress_profile)
            if ingress is None:
                raise ProfileNotFoundError(ingress_profile)
        elif ingress_profile is None:
            ingress = profile
        if not self.bidirectional or ingress_profile == "none":
            ingress = None
        return self._apply_profiles(profile.uplink or profile, ingress, profile_name)

    def _apply_profiles(self, egress: NetworkProfile, ingress: Optional[NetworkProfile],
                        profile_name: str) -> bool:
        # Validate both sides before removing any existing shaping.
        egress.validate()
        if ingress is not None:
            ingress.validate()
        self.clear()
        success = False
        try:
            if not self._apply_profile_to_device(self.interface, egress):
                return False
            if ingress is not None and not ingress.bypass:
                if not self._setup_ifb():
                    return False
                if not self._apply_profile_to_device(self.ifb_device, ingress):
                    return False
                self.current_ingress_profile = ingress.name
            self.current_profile = profile_name
            success = True
            logger.info("Applied network profile: %s", profile_name)
            return True
        except Exception as exc:
            logger.error("Failed to apply network profile: %s", exc)
            return False
        finally:
            if not success:
                self.clear()

    def _apply_profile_to_device(self, device: str, profile: NetworkProfile) -> bool:
        return all(self._run_tc_command(cmd) for cmd in shaping_commands(profile, device))

    def apply_settings(
        self,
        delay_ms: int = 0,
        jitter_ms: int = 0,
        delay_distribution: Optional[str] = None,
        delay_correlation_pct: Optional[float] = None,
        loss_pct: float = 0.0,
        loss_correlation_pct: Optional[float] = None,
        loss_model: Optional[str] = None,
        rate_mbit: Optional[int] = None,
        rate_ceil_mbit: Optional[int] = None,
        rate_burst_kbit: Optional[int] = None,
        rate_cburst_kbit: Optional[int] = None,
        corruption_pct: float = 0.0,
        corruption_correlation_pct: Optional[float] = None,
        reorder_pct: float = 0.0,
        reorder_correlation_pct: Optional[float] = None,
        duplicate_pct: float = 0.0,
        duplicate_correlation_pct: Optional[float] = None,
        limit_packets: Optional[int] = None,
        profile_name: str = "custom",
        ingress_settings: Optional[dict] = None,
        disable_ingress: bool = False,
        *,
        rate_mode: Optional[str] = None,
        netem_rate_mbit: Optional[float] = None,
        burst_bytes: Optional[int] = None,
        cburst_bytes: Optional[int] = None,
        loss_mean_burst_packets: Optional[float] = None,
        slot_distribution: Optional[str] = None,
        slot_delay_ms: Optional[float] = None,
        slot_jitter_ms: Optional[float] = None,
        slot_max_packets: Optional[int] = None,
        slot_max_bytes: Optional[int] = None,
        bypass: bool = False,
    ) -> bool:
        """
        Apply custom network settings.

        Args:
            delay_ms: One-way delay in milliseconds (egress).
            jitter_ms: Delay jitter in milliseconds (egress).
            delay_distribution: Delay distribution (normal, pareto, etc.).
            delay_correlation_pct: Delay correlation percentage.
            loss_pct: Packet loss percentage (egress).
            loss_correlation_pct: Loss correlation percentage.
            loss_model: Advanced loss model (gemodel, state).
            rate_mbit: Bandwidth limit in Mbps, None for unlimited (egress).
            rate_ceil_mbit: Maximum burst rate in Mbps.
            rate_burst_kbit: Burst buffer size in kbits.
            rate_cburst_kbit: Ceil burst buffer size in kbits.
            corruption_pct: Packet corruption percentage (egress).
            corruption_correlation_pct: Corruption correlation percentage.
            reorder_pct: Packet reordering percentage (egress).
            reorder_correlation_pct: Reorder correlation percentage.
            duplicate_pct: Packet duplication percentage.
            duplicate_correlation_pct: Duplication correlation percentage.
            limit_packets: Queue limit in packets.
            profile_name: Name for logging purposes.
            ingress_settings: Optional dict with separate settings for ingress.
                If None and bidirectional=True, uses same settings as egress.
            disable_ingress: If True, disable ingress shaping even if bidirectional=True.

        Returns:
            True if successful, False otherwise.
        """
        settings = {key: value for key, value in locals().items()
                    if key not in {"self", "profile_name", "ingress_settings", "disable_ingress"}}
        egress = NetworkProfile.from_dict(profile_name, settings)
        ingress = None
        if self.bidirectional and not disable_ingress:
            name = f"{profile_name}_ingress" if ingress_settings else profile_name
            ingress = NetworkProfile.from_dict(name, {**settings, **(ingress_settings or {})})
        return self._apply_profiles(egress, ingress, profile_name)

    def _build_netem_params(
        self,
        delay_ms: int,
        jitter_ms: int,
        delay_distribution: Optional[str],
        delay_correlation_pct: Optional[float],
        loss_pct: float,
        loss_correlation_pct: Optional[float],
        loss_model: Optional[str],
        corruption_pct: float,
        corruption_correlation_pct: Optional[float],
        reorder_pct: float,
        reorder_correlation_pct: Optional[float],
        duplicate_pct: float,
        duplicate_correlation_pct: Optional[float],
        limit_packets: Optional[int],
    ) -> list[str]:
        """Build netem parameter list from legacy flat settings."""
        settings = {key: value for key, value in locals().items() if key != "self"}
        return netem_parameters(NetworkProfile.from_dict("custom", settings))

    def clear(self) -> bool:
        """
        Clear all tc rules from the interface.

        Also tears down IFB device if bidirectional shaping was enabled.
        Always attempts to remove the ingress qdisc to handle stale rules
        left by a previous process.

        Returns:
            True if successful (or no rules to clear), False on error.
        """
        # Clear egress rules on main interface
        cmd = f"sudo tc qdisc del dev {self.interface} root"
        self._run_tc_command(cmd, ignore_errors=True)

        # Tear down IFB if it was initialized
        if self._ifb_initialized:
            self._teardown_ifb()
        else:
            # Always attempt to remove a potentially stale ingress qdisc
            # left by a previous emulator instance that wasn't cleaned up.
            cmd_del_ingress = f"sudo tc qdisc del dev {self.interface} ingress"
            self._run_tc_command(cmd_del_ingress, ignore_errors=True)

        self.current_profile = None
        self.current_ingress_profile = None
        return True  # Ignore "RTNETLINK answers: No such file or directory" error

    def _run_tc_command(self, cmd: str, ignore_errors: bool = False) -> bool:
        """Execute a tc command."""
        logger.debug(f"Running: {cmd}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0 and not ignore_errors:
                logger.error(f"tc command failed: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"tc command timed out: {cmd}")
            return False
        except Exception as e:
            logger.error(f"tc command error: {e}")
            return False

    _LOOPBACK_PROTOCOLS = {"tcp": 6, "udp": 17}

    def apply_profile_to_loopback(
        self,
        profile_name: str,
        dest_port: Optional[int] = None,
        *,
        selectors: Sequence[tuple[str, Optional[int]]] = (),
    ) -> bool:
        """Shape selected TCP/UDP loopback traffic for both IP versions.

        ``dest_port`` selects TCP to/from a server port; ``selectors`` adds
        (protocol, server_port) pairs. A None port selects the entire protocol,
        and no selectors selects all traffic. Each call replaces prior rules.

        Calibration profiles use their complete service model. Server-port
        selectors separate uplink requests and downlink responses. Protocol-wide
        and all-traffic selectors cannot identify the client: they use a shared
        downlink queue, as the original loopback API did. Legacy symmetric
        profiles retain the original single Netem queue and native rate cap.
        """
        profile = self.get_profile(profile_name)
        if profile is None:
            raise ProfileNotFoundError(profile_name)
        rules = []
        if dest_port is not None:
            rules.append(("tcp", dest_port))
        rules.extend(selectors)
        normalized = []
        for proto, port in rules:
            if not isinstance(proto, str) or proto.lower() not in self._LOOPBACK_PROTOCOLS:
                raise ValueError(f"Unsupported loopback selector protocol {proto!r}")
            if port is not None and (
                    isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535):
                raise ValueError("Loopback port must be an integer between 1 and 65535")
            rule = (proto.lower(), port)
            if rule not in normalized:
                normalized.append(rule)
        rules = normalized
        uplink = profile.uplink or profile
        profile.validate()
        uplink.validate()
        calibrated = any(value is not None for value in (
            profile.rate_mode, profile.loss_mean_burst_packets, profile.slot_distribution,
            profile.burst_bytes, profile.cburst_bytes, profile.uplink,
        )) or profile.bypass
        root = ("sudo tc qdisc add dev lo root handle 1: prio bands 3 priomap "
                + " ".join(["0"] * 16))
        if calibrated:
            commands = self._calibrated_loopback_commands(profile, rules, root)
        else:
            # Preserve the PR's legacy loopback behavior, including rate-only
            # profiles and the selector API used by the WebRTC scenario.
            params = netem_parameters(profile)
            if profile.rate_mbit is not None:
                params.append(f"rate {profile.rate_mbit}mbit")
            commands = ([root,
                         "sudo tc qdisc add dev lo parent 1:3 handle 30: netem "
                         + " ".join(params),
                         *self._loopback_filter_commands(rules)] if params else [])
        self.clear_loopback()
        for command in commands:
            if not self._run_tc_command(command):
                self.clear_loopback()
                return False
        if commands:
            self._lo_port = dest_port
            self._lo_selectors = tuple(rules)
            logger.info("Applied %s to lo (%s)", profile_name,
                        self._describe_loopback_rules(rules))
        return True

    @staticmethod
    def _calibrated_loopback_commands(
        profile: NetworkProfile, rules: Sequence[tuple[str, Optional[int]]], root: str,
    ) -> list[str]:
        """Use directional service models where a server port identifies them."""
        uplink = profile.uplink or profile
        port_rules = [(proto, port) for proto, port in rules if port is not None]
        broad_rules = [(proto, port) for proto, port in rules if port is None]
        up = (shaping_commands(uplink, "lo", parent="1:2", htb_handle=20, netem_handle=21)
              if port_rules else [])
        down = shaping_commands(profile, "lo", parent="1:3", htb_handle=30, netem_handle=31)
        if not up and not down:
            return []
        if profile.uplink is not None and (broad_rules or not rules):
            logger.warning("Loopback selectors without a server port use the downlink "
                           "service model in both directions for profile %s", profile.name)
        commands = [root, *up, *down]
        if not rules:
            commands += NetworkEmulator._loopback_filter_commands(())
            return commands
        # Port-specific matches precede protocol-wide fallbacks. Unique filter
        # priorities also avoid IPv4/IPv6 classifier protocol conflicts.
        priority = 0
        for family in ("ip", "ipv6"):
            for proto, port in port_rules:
                for field, band in (("dst_port", 2), ("src_port", 3)):
                    priority += 1
                    commands.append(
                        f"sudo tc filter add dev lo parent 1:0 protocol {family} "
                        f"prio {priority} flower ip_proto {proto} {field} {port} flowid 1:{band}")
            for proto, _ in broad_rules:
                priority += 1
                commands.append(
                    f"sudo tc filter add dev lo parent 1:0 protocol {family} "
                    f"prio {priority} flower ip_proto {proto} flowid 1:3")
        return commands

    @classmethod
    def _loopback_filter_commands(
        cls, rules: Sequence[tuple[str, Optional[int]]]
    ) -> list[str]:
        """tc filter commands steering *rules* into the netem band on lo.

        Port matches use u32's ``ip dport``/``ip sport`` selectors, which
        assume an IP header without options; the same holds for ``ip6``.
        """
        base = "sudo tc filter add dev lo parent 1:0"
        if not rules:
            return [f"{base} protocol all prio 1 u32 match u32 0 0 flowid 1:3"]
        cmds: list[str] = []
        for proto, port in rules:
            number = cls._LOOPBACK_PROTOCOLS[proto]
            # tc rejects filters of different protocols under one priority
            # ("Protocol mismatch for filter with specified priority"), so
            # each address family gets its own.
            for family, prefix, prio in (("ip", "ip", 1), ("ipv6", "ip6", 2)):
                head = (
                    f"{base} protocol {family} prio {prio} u32 "
                    f"match {prefix} protocol {number} 0xff"
                )
                if port is None:
                    cmds.append(f"{head} flowid 1:3")
                    continue
                for side in ("dport", "sport"):
                    cmds.append(
                        f"{head} match {prefix} {side} {port} 0xffff flowid 1:3"
                    )
        return cmds

    @staticmethod
    def _describe_loopback_rules(
        rules: Sequence[tuple[str, Optional[int]]]
    ) -> str:
        if not rules:
            return "all traffic"
        return ", ".join(
            proto if port is None else f"{proto}:{port}" for proto, port in rules
        )

    def clear_loopback(self) -> bool:
        """Remove netem rules from the loopback interface."""
        # Remove tc qdisc (and all child filters) from lo
        self._run_tc_command(
            "sudo tc qdisc del dev lo root", ignore_errors=True
        )

        self._lo_port = None
        self._lo_selectors = ()
        return True

    def get_status(self) -> dict:
        """
        Get current tc/netem status.

        Returns:
            Dictionary with interface, profile info, and tc output.
        """
        try:
            # Get egress status
            result = subprocess.run(
                f"tc qdisc show dev {self.interface}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )

            status = {
                "interface": self.interface,
                "current_profile": self.current_profile,
                "bidirectional": self.bidirectional,
                "egress_tc_output": result.stdout,
                "egress_active": "netem" in result.stdout or "htb" in result.stdout,
            }

            # Get ingress/IFB status if bidirectional
            if self.bidirectional and self._ifb_initialized:
                ifb_result = subprocess.run(
                    f"tc qdisc show dev {self.ifb_device}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                status["ifb_device"] = self.ifb_device
                status["ingress_tc_output"] = ifb_result.stdout
                status["ingress_active"] = (
                    "netem" in ifb_result.stdout or "htb" in ifb_result.stdout
                )

            return status

        except Exception as e:
            return {
                "interface": self.interface,
                "current_profile": self.current_profile,
                "bidirectional": self.bidirectional,
                "error": str(e),
            }

    def get_profile(self, name: str) -> Optional[NetworkProfile]:
        """
        Get a profile by name.

        Args:
            name: Profile name to look up.

        Returns:
            NetworkProfile if found, None otherwise.
        """
        if name in self.profiles:
            return self.profiles[name]
        base, separator, direction = name.rpartition("/")
        if separator and base in self.profiles and direction in {"uplink", "downlink"}:
            profile = self.profiles[base]
            selected = (profile.uplink or profile) if direction == "uplink" else profile
            # An explicit directional name represents a flat, symmetric selection.
            return NetworkProfile.from_dict(name, {"description": selected.description,
                                                  **selected.settings()})
        return None

    def list_profiles(self) -> list[str]:
        """
        List all available profile names.

        Returns:
            List of profile names.
        """
        return list(self.profiles.keys())


# Convenience functions for scripting
def apply_profile(profile_config: dict, interface: str = "eth0") -> bool:
    """
    Apply a network profile from a config dict.

    Args:
        profile_config: Dictionary with profile parameters.
        interface: Network interface to apply to.

    Returns:
        True if successful.
    """
    emulator = NetworkEmulator(interface=interface)
    emulator.profiles["custom"] = NetworkProfile.from_dict("custom", profile_config)
    return emulator.apply_profile("custom")


def clear_profile(interface: str = "eth0") -> bool:
    """
    Clear all network emulation rules.

    Args:
        interface: Network interface to clear rules from.

    Returns:
        True if successful.
    """
    emulator = NetworkEmulator(interface=interface)
    return emulator.clear()
