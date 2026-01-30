"""
Network Emulator using Linux tc/netem.

Provides the NetworkEmulator class for applying network condition emulation
(latency, jitter, packet loss, bandwidth limits, etc.) to network interfaces.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import yaml

from .exceptions import (
    CommandFailedError,
    ProfileLoadError,
    ProfileNotFoundError,
    SudoNotAvailableError,
)
from .profile import NetworkProfile

logger = logging.getLogger(__name__)


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
            interface: Network interface to apply rules to.
            profiles_path: Path to profiles YAML file. If provided, profiles
                are loaded automatically.
            bidirectional: If True, shape both egress and ingress traffic.
            ifb_device: IFB device name for ingress shaping.
        """
        self.interface = interface
        self.profiles: dict[str, NetworkProfile] = {}
        self.current_profile: Optional[str] = None
        self.current_ingress_profile: Optional[str] = None
        self._sudo_available: Optional[bool] = None
        self.bidirectional = bidirectional
        self.ifb_device = ifb_device
        self._ifb_initialized = False

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
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ProfileLoadError(path, "file not found")
        except yaml.YAMLError as e:
            raise ProfileLoadError(path, f"invalid YAML: {e}")

        if not data:
            raise ProfileLoadError(path, "empty file")

        if "default_interface" in data:
            self.interface = data["default_interface"]

        profiles_data = data.get("profiles", {})
        if not profiles_data:
            raise ProfileLoadError(path, "no profiles defined")

        for name, config in profiles_data.items():
            self.profiles[name] = NetworkProfile.from_dict(name, config)

        logger.info(f"Loaded {len(self.profiles)} network profiles from {path}")

    def check_sudo(self) -> bool:
        """
        Check if sudo is available without password.

        Returns:
            True if passwordless sudo is available.
        """
        if self._sudo_available is not None:
            return self._sudo_available

        try:
            result = subprocess.run(
                ["sudo", "-n", "true"], capture_output=True, timeout=5
            )
            self._sudo_available = result.returncode == 0
        except Exception:
            self._sudo_available = False

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
        """
        Apply a network profile for egress and optionally a different profile for ingress.

        Args:
            profile_name: Name of the profile to apply to egress traffic.
            ingress_profile: Name of profile for ingress traffic. Options:
                - None (default): Use same profile as egress when bidirectional=True.
                - "none": Disable ingress shaping entirely.
                - "<profile_name>": Use a different profile for ingress.

        Returns:
            True if successful, False otherwise.

        Raises:
            ProfileNotFoundError: If the specified profile doesn't exist.
        """
        if profile_name not in self.profiles:
            raise ProfileNotFoundError(profile_name)

        # Determine ingress profile
        ingress_profile_obj: Optional[NetworkProfile] = None
        apply_ingress = self.bidirectional

        if ingress_profile == "none":
            apply_ingress = False
        elif ingress_profile is not None:
            if ingress_profile not in self.profiles:
                raise ProfileNotFoundError(ingress_profile)
            ingress_profile_obj = self.profiles[ingress_profile]
        else:
            # Default: use same profile for ingress
            ingress_profile_obj = (
                self.profiles[profile_name] if apply_ingress else None
            )

        egress_profile = self.profiles[profile_name]

        # Clear existing rules first
        self.clear()

        try:
            # Apply egress shaping
            success = self._apply_profile_to_device(
                device=self.interface, profile=egress_profile
            )
            if not success:
                return False

            # Apply ingress shaping if enabled
            if apply_ingress and ingress_profile_obj is not None:
                if not self._setup_ifb():
                    logger.error("Failed to set up IFB for ingress shaping")
                    return False

                success = self._apply_profile_to_device(
                    device=self.ifb_device, profile=ingress_profile_obj
                )
                if not success:
                    logger.error("Failed to apply ingress shaping to IFB")
                    return False

                self.current_ingress_profile = ingress_profile_obj.name
                if ingress_profile_obj.name == egress_profile.name:
                    logger.info(f"Applied bidirectional profile: {profile_name}")
                else:
                    logger.info(
                        f"Applied asymmetric shaping: egress={profile_name}, "
                        f"ingress={ingress_profile_obj.name}"
                    )
            else:
                logger.info(f"Applied egress-only profile: {profile_name}")

            self.current_profile = profile_name
            return True

        except Exception as e:
            logger.error(f"Failed to apply network profile: {e}")
            return False

    def _apply_profile_to_device(
        self, device: str, profile: NetworkProfile
    ) -> bool:
        """Apply a NetworkProfile to a specific device."""
        return self._apply_shaping_to_device(
            device=device,
            delay_ms=profile.delay_ms,
            jitter_ms=profile.jitter_ms,
            delay_distribution=profile.delay_distribution,
            delay_correlation_pct=profile.delay_correlation_pct,
            loss_pct=profile.loss_pct,
            loss_correlation_pct=profile.loss_correlation_pct,
            loss_model=profile.loss_model,
            rate_mbit=profile.rate_mbit,
            rate_ceil_mbit=profile.rate_ceil_mbit,
            rate_burst_kbit=profile.rate_burst_kbit,
            rate_cburst_kbit=profile.rate_cburst_kbit,
            corruption_pct=profile.corruption_pct,
            corruption_correlation_pct=profile.corruption_correlation_pct,
            reorder_pct=profile.reorder_pct,
            reorder_correlation_pct=profile.reorder_correlation_pct,
            duplicate_pct=profile.duplicate_pct,
            duplicate_correlation_pct=profile.duplicate_correlation_pct,
            limit_packets=profile.limit_packets,
        )

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
        # Clear existing rules first
        self.clear()

        try:
            # Apply egress shaping to main interface
            success = self._apply_shaping_to_device(
                device=self.interface,
                delay_ms=delay_ms,
                jitter_ms=jitter_ms,
                delay_distribution=delay_distribution,
                delay_correlation_pct=delay_correlation_pct,
                loss_pct=loss_pct,
                loss_correlation_pct=loss_correlation_pct,
                loss_model=loss_model,
                rate_mbit=rate_mbit,
                rate_ceil_mbit=rate_ceil_mbit,
                rate_burst_kbit=rate_burst_kbit,
                rate_cburst_kbit=rate_cburst_kbit,
                corruption_pct=corruption_pct,
                corruption_correlation_pct=corruption_correlation_pct,
                reorder_pct=reorder_pct,
                reorder_correlation_pct=reorder_correlation_pct,
                duplicate_pct=duplicate_pct,
                duplicate_correlation_pct=duplicate_correlation_pct,
                limit_packets=limit_packets,
            )

            if not success:
                return False

            # Apply ingress shaping via IFB if bidirectional is enabled
            if self.bidirectional and not disable_ingress:
                if not self._setup_ifb():
                    logger.error("Failed to set up IFB for bidirectional shaping")
                    return False

                # Use separate ingress settings if provided, otherwise mirror egress
                ing = ingress_settings or {}
                success = self._apply_shaping_to_device(
                    device=self.ifb_device,
                    delay_ms=ing.get("delay_ms", delay_ms),
                    jitter_ms=ing.get("jitter_ms", jitter_ms),
                    delay_distribution=ing.get("delay_distribution", delay_distribution),
                    delay_correlation_pct=ing.get(
                        "delay_correlation_pct", delay_correlation_pct
                    ),
                    loss_pct=ing.get("loss_pct", loss_pct),
                    loss_correlation_pct=ing.get(
                        "loss_correlation_pct", loss_correlation_pct
                    ),
                    loss_model=ing.get("loss_model", loss_model),
                    rate_mbit=ing.get("rate_mbit", rate_mbit),
                    rate_ceil_mbit=ing.get("rate_ceil_mbit", rate_ceil_mbit),
                    rate_burst_kbit=ing.get("rate_burst_kbit", rate_burst_kbit),
                    rate_cburst_kbit=ing.get("rate_cburst_kbit", rate_cburst_kbit),
                    corruption_pct=ing.get("corruption_pct", corruption_pct),
                    corruption_correlation_pct=ing.get(
                        "corruption_correlation_pct", corruption_correlation_pct
                    ),
                    reorder_pct=ing.get("reorder_pct", reorder_pct),
                    reorder_correlation_pct=ing.get(
                        "reorder_correlation_pct", reorder_correlation_pct
                    ),
                    duplicate_pct=ing.get("duplicate_pct", duplicate_pct),
                    duplicate_correlation_pct=ing.get(
                        "duplicate_correlation_pct", duplicate_correlation_pct
                    ),
                    limit_packets=ing.get("limit_packets", limit_packets),
                )

                if not success:
                    logger.error("Failed to apply ingress shaping to IFB")
                    return False

                self.current_ingress_profile = (
                    f"{profile_name}_ingress" if ingress_settings else profile_name
                )
                if ingress_settings:
                    logger.info(
                        f"Applied asymmetric shaping: {profile_name} (egress) / custom (ingress)"
                    )
                else:
                    logger.info(f"Applied bidirectional shaping: {profile_name}")
            else:
                logger.info(f"Applied egress-only shaping: {profile_name}")

            self.current_profile = profile_name
            return True

        except Exception as e:
            logger.error(f"Failed to apply network settings: {e}")
            return False

    def _apply_shaping_to_device(
        self,
        device: str,
        delay_ms: int,
        jitter_ms: int,
        delay_distribution: Optional[str],
        delay_correlation_pct: Optional[float],
        loss_pct: float,
        loss_correlation_pct: Optional[float],
        loss_model: Optional[str],
        rate_mbit: Optional[int],
        rate_ceil_mbit: Optional[int],
        rate_burst_kbit: Optional[int],
        rate_cburst_kbit: Optional[int],
        corruption_pct: float,
        corruption_correlation_pct: Optional[float],
        reorder_pct: float,
        reorder_correlation_pct: Optional[float],
        duplicate_pct: float,
        duplicate_correlation_pct: Optional[float],
        limit_packets: Optional[int],
    ) -> bool:
        """
        Apply shaping rules to a specific device.

        Args:
            device: Network device to apply rules to (interface or IFB).
            ... (other params same as apply_settings)

        Returns:
            True if successful, False otherwise.
        """
        if rate_mbit is not None:
            # Use HTB + netem for rate limiting with impairments
            return self._apply_htb_netem(
                device,
                delay_ms,
                jitter_ms,
                delay_distribution,
                delay_correlation_pct,
                loss_pct,
                loss_correlation_pct,
                loss_model,
                rate_mbit,
                rate_ceil_mbit,
                rate_burst_kbit,
                rate_cburst_kbit,
                corruption_pct,
                corruption_correlation_pct,
                reorder_pct,
                reorder_correlation_pct,
                duplicate_pct,
                duplicate_correlation_pct,
                limit_packets,
            )
        else:
            # Use netem only (no rate limiting)
            return self._apply_netem_only(
                device,
                delay_ms,
                jitter_ms,
                delay_distribution,
                delay_correlation_pct,
                loss_pct,
                loss_correlation_pct,
                loss_model,
                corruption_pct,
                corruption_correlation_pct,
                reorder_pct,
                reorder_correlation_pct,
                duplicate_pct,
                duplicate_correlation_pct,
                limit_packets,
            )

    def _apply_netem_only(
        self,
        device: str,
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
    ) -> bool:
        """Apply netem rules without rate limiting."""
        params = self._build_netem_params(
            delay_ms=delay_ms,
            jitter_ms=jitter_ms,
            delay_distribution=delay_distribution,
            delay_correlation_pct=delay_correlation_pct,
            loss_pct=loss_pct,
            loss_correlation_pct=loss_correlation_pct,
            loss_model=loss_model,
            corruption_pct=corruption_pct,
            corruption_correlation_pct=corruption_correlation_pct,
            reorder_pct=reorder_pct,
            reorder_correlation_pct=reorder_correlation_pct,
            duplicate_pct=duplicate_pct,
            duplicate_correlation_pct=duplicate_correlation_pct,
            limit_packets=limit_packets,
        )

        if not params:
            return True

        cmd = f"sudo tc qdisc add dev {device} root netem {' '.join(params)}"
        return self._run_tc_command(cmd)

    def _apply_htb_netem(
        self,
        device: str,
        delay_ms: int,
        jitter_ms: int,
        delay_distribution: Optional[str],
        delay_correlation_pct: Optional[float],
        loss_pct: float,
        loss_correlation_pct: Optional[float],
        loss_model: Optional[str],
        rate_mbit: int,
        rate_ceil_mbit: Optional[int],
        rate_burst_kbit: Optional[int],
        rate_cburst_kbit: Optional[int],
        corruption_pct: float,
        corruption_correlation_pct: Optional[float],
        reorder_pct: float,
        reorder_correlation_pct: Optional[float],
        duplicate_pct: float,
        duplicate_correlation_pct: Optional[float],
        limit_packets: Optional[int],
    ) -> bool:
        """Apply HTB rate limiting with netem impairments."""
        # Create HTB root qdisc
        cmd1 = f"sudo tc qdisc add dev {device} root handle 1: htb default 11"
        if not self._run_tc_command(cmd1):
            return False

        ceil_mbit = rate_ceil_mbit or rate_mbit
        burst = f" burst {rate_burst_kbit}kbit" if rate_burst_kbit else ""
        cburst = f" cburst {rate_cburst_kbit}kbit" if rate_cburst_kbit else ""

        # Create HTB class with rate limit
        cmd2 = (
            f"sudo tc class add dev {device} parent 1: "
            f"classid 1:1 htb rate {rate_mbit}mbit ceil {ceil_mbit}mbit{burst}{cburst}"
        )
        if not self._run_tc_command(cmd2):
            return False

        cmd3 = (
            f"sudo tc class add dev {device} parent 1:1 "
            f"classid 1:11 htb rate {rate_mbit}mbit ceil {ceil_mbit}mbit{burst}{cburst}"
        )
        if not self._run_tc_command(cmd3):
            return False

        # Add netem as leaf qdisc
        params = self._build_netem_params(
            delay_ms=delay_ms,
            jitter_ms=jitter_ms,
            delay_distribution=delay_distribution,
            delay_correlation_pct=delay_correlation_pct,
            loss_pct=loss_pct,
            loss_correlation_pct=loss_correlation_pct,
            loss_model=loss_model,
            corruption_pct=corruption_pct,
            corruption_correlation_pct=corruption_correlation_pct,
            reorder_pct=reorder_pct,
            reorder_correlation_pct=reorder_correlation_pct,
            duplicate_pct=duplicate_pct,
            duplicate_correlation_pct=duplicate_correlation_pct,
            limit_packets=limit_packets,
        )

        if params:
            cmd4 = f"sudo tc qdisc add dev {device} parent 1:11 handle 10: netem {' '.join(params)}"
            if not self._run_tc_command(cmd4):
                return False

        return True

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
        """Build netem parameter list from settings."""
        params: list[str] = []

        if limit_packets:
            params.append(f"limit {limit_packets}")

        if delay_ms > 0:
            delay_parts = [f"delay {delay_ms}ms"]
            if jitter_ms > 0:
                delay_parts.append(f"{jitter_ms}ms")
                if delay_correlation_pct is not None:
                    delay_parts.append(f"{delay_correlation_pct}%")
                if delay_distribution:
                    delay_parts.append(f"distribution {delay_distribution}")
            params.append(" ".join(delay_parts))

        if loss_model:
            if loss_pct > 0:
                loss_parts = [f"loss {loss_pct}%"]
                if loss_correlation_pct is not None:
                    loss_parts.append(f"{loss_correlation_pct}%")
                loss_parts.append(loss_model)
                params.append(" ".join(loss_parts))
            else:
                params.append(f"loss {loss_model}")
        elif loss_pct > 0:
            loss_parts = [f"loss {loss_pct}%"]
            if loss_correlation_pct is not None:
                loss_parts.append(f"{loss_correlation_pct}%")
            params.append(" ".join(loss_parts))

        if corruption_pct > 0:
            corrupt_parts = [f"corrupt {corruption_pct}%"]
            if corruption_correlation_pct is not None:
                corrupt_parts.append(f"{corruption_correlation_pct}%")
            params.append(" ".join(corrupt_parts))

        if duplicate_pct > 0:
            duplicate_parts = [f"duplicate {duplicate_pct}%"]
            if duplicate_correlation_pct is not None:
                duplicate_parts.append(f"{duplicate_correlation_pct}%")
            params.append(" ".join(duplicate_parts))

        if reorder_pct > 0:
            corr = reorder_correlation_pct if reorder_correlation_pct is not None else 25
            params.append(f"reorder {reorder_pct}% {corr}%")

        return params

    def clear(self) -> bool:
        """
        Clear all tc rules from the interface.

        Also tears down IFB device if bidirectional shaping was enabled.

        Returns:
            True if successful (or no rules to clear), False on error.
        """
        # Clear egress rules on main interface
        cmd = f"sudo tc qdisc del dev {self.interface} root"
        self._run_tc_command(cmd, ignore_errors=True)

        # Tear down IFB if it was initialized
        if self._ifb_initialized:
            self._teardown_ifb()

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
        return self.profiles.get(name)

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
    return emulator.apply_settings(
        delay_ms=profile_config.get("delay_ms", 0),
        jitter_ms=profile_config.get("jitter_ms", 0),
        loss_pct=profile_config.get("loss_pct", 0.0),
        rate_mbit=profile_config.get("rate_mbit"),
        delay_distribution=profile_config.get("delay_distribution"),
        delay_correlation_pct=profile_config.get("delay_correlation_pct"),
        loss_correlation_pct=profile_config.get("loss_correlation_pct"),
        loss_model=profile_config.get("loss_model"),
        rate_ceil_mbit=profile_config.get("rate_ceil_mbit"),
        rate_burst_kbit=profile_config.get("rate_burst_kbit"),
        rate_cburst_kbit=profile_config.get("rate_cburst_kbit"),
        corruption_pct=profile_config.get("corruption_pct", 0.0),
        corruption_correlation_pct=profile_config.get("corruption_correlation_pct"),
        reorder_pct=profile_config.get("reorder_pct", 0.0),
        reorder_correlation_pct=profile_config.get("reorder_correlation_pct"),
        duplicate_pct=profile_config.get("duplicate_pct", 0.0),
        duplicate_correlation_pct=profile_config.get("duplicate_correlation_pct"),
        limit_packets=profile_config.get("limit_packets"),
    )


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
