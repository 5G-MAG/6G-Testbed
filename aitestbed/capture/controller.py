"""
Capture Controller for the 6G AI Traffic Testbed.

Manages packet capture using tcpdump.
"""

import re
import subprocess
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CaptureController:
    """
    Controls packet capture for traffic analysis.

    Uses tcpdump for L3/L4 capture. For L7 analysis,
    consider using mitmproxy separately.
    """

    def __init__(
        self,
        interface: str = "eth0",
        capture_dir: str = "results/captures"
    ):
        """
        Initialize the capture controller.

        Args:
            interface: Network interface to capture on
            capture_dir: Directory to store capture files
        """
        self.interface = interface
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._process: Optional[subprocess.Popen] = None
        self._current_file: Optional[Path] = None
        # tcpdump's own summary, parsed from stderr on stop():
        # {"captured", "received_by_filter", "received_by_kernel",
        #  "dropped_by_kernel", "dropped_by_tcpdump"}
        self.stats: dict = {}

    def start(
        self,
        filename: Optional[str] = None,
        filter_expr: Optional[str] = None,
        buffer_kib: Optional[int] = None
    ) -> Optional[Path]:
        """
        Start packet capture.

        Args:
            filename: Output filename (auto-generated if not provided)
            filter_expr: BPF filter expression (e.g., "port 443")
            buffer_kib: AF_PACKET ring buffer size in KiB (-B). Loopback
                GSO bursts (64KB super-packets at full no-emulation
                speed) overflow tcpdump's default ~2MB ring and the
                kernel silently drops 30-80% of uplink data frames —
                observed live: pcap held 111MB of a 322MB upload while
                server ACKs proved full delivery. 262144 (256MB) is the
                matrix driver's default; None keeps tcpdump's default.

        Returns:
            Path to the capture file, or None if failed
        """
        if self._process is not None:
            logger.warning("Capture already in progress")
            return self._current_file

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.pcap"

        self._current_file = self.capture_dir / filename
        self.stats = {}

        cmd = [
            # -n: never prompt — a password prompt would hang the run
            # (or die instantly under Popen with no tty); the runner's
            # preflight surfaces this with a clear message instead.
            "sudo", "-n", "tcpdump",
            "-i", self.interface,
            "-w", str(self._current_file),
            "-U",  # Packet-buffered output
        ]
        if buffer_kib:
            cmd.extend(["-B", str(int(buffer_kib))])

        if filter_expr:
            cmd.extend(filter_expr.split())

        try:
            logger.info(f"Starting capture: {' '.join(cmd)}")
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            self._process = None
            self._current_file = None
            return None

        # Liveness check: Popen succeeding only means the process was
        # SPAWNED — sudo password prompts, "tcpdump: auto: No such
        # device" and missing binaries all die within milliseconds
        # AFTER spawn, which used to leave a phantom capture (0-byte
        # pcap promised, never written; observed live: interface='auto'
        # rows with response_bytes=0). Wait briefly and re-poll.
        import time as _time
        _time.sleep(0.5)
        if self._process.poll() is not None:
            stderr_tail = ""
            try:
                err = self._process.stderr.read() if self._process.stderr else b""
                stderr_tail = (err or b"").decode(errors="replace")[-400:].strip()
            except Exception:
                pass
            logger.error(
                f"tcpdump exited immediately (rc={self._process.returncode})"
                + (f": {stderr_tail}" if stderr_tail else ""))
            self._process = None
            self._current_file = None
            return None

        return self._current_file

    def stop(self) -> Optional[Path]:
        """
        Stop packet capture.

        Reads tcpdump's exit summary off stderr ("N packets captured /
        M packets received by filter / K dropped by kernel / J dropped
        by tcpdump") into self.stats; callers surface the drop counts so
        a capture that silently lost frames is visible instead of
        poisoning downstream byte accounting.

        Returns:
            Path to the capture file, or None if no capture was running
        """
        if self._process is None:
            logger.warning("No capture in progress")
            return None

        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        except Exception as e:
            logger.error(f"Error stopping capture: {e}")

        # Drain stderr AFTER the process is reaped: tcpdump prints its
        # packet/drop summary to stderr on SIGTERM exit. Must consume
        # it now (PIPE otherwise) — and it is the drop-count evidence.
        stderr_tail = ""
        try:
            raw = self._process.stderr.read() if self._process.stderr \
                else b""
            stderr_tail = (raw or b"").decode(errors="replace")
        except Exception:
            pass
        self.stats = self._parse_tcpdump_summary(stderr_tail)

        result = self._current_file
        self._process = None
        self._current_file = None

        # Diagnose the two silent-failure outcomes instead of returning
        # a path the caller then reports as a successful 0-byte capture:
        # (a) file never created (tcpdump rejected the interface/filter
        #     and died after the 0.5s liveness window, e.g. mid-run
        #     permissions change), (b) created but 0 packets captured.
        if result is not None:
            try:
                if not result.exists():
                    logger.error(
                        f"Capture file was never written: {result} "
                        f"(tcpdump died before writing any packet)")
                elif result.stat().st_size == 0:
                    logger.warning(
                        f"Capture file is 0 bytes: {result} "
                        f"(no packets matched the filter / interface)")
            except Exception as e:
                logger.warning(f"Could not stat capture file {result}: {e}")

        # Capture-path loss is the ONE failure mode that corrupts pcap
        # byte accounting invisibly (the transport itself stayed clean);
        # make it loud. Bigger ring: -B (start()); kernel cap:
        # sysctl net.core.rmem_max.
        dropped = (self.stats.get("dropped_by_kernel") or 0) \
            + (self.stats.get("dropped_by_tcpdump") or 0)
        if dropped:
            logger.error(
                f"tcpdump dropped {dropped} packets during capture "
                f"(kernel={self.stats.get('dropped_by_kernel')}, "
                f"tcpdump={self.stats.get('dropped_by_tcpdump')}); "
                f"pcap byte counts are UNDERCOUNTS — raise -B "
                f"(start(buffer_kib=...)) or sysctl net.core.rmem_max")

        logger.info(f"Capture stopped: {result} "
                    f"(stats: {self.stats or 'summary unavailable'})")
        return result

    @staticmethod
    def _parse_tcpdump_summary(stderr_text: str) -> dict:
        """Parse tcpdump's stderr exit summary into a stats dict.

        Tolerates both wordings across libpcap versions:
          "1234 packets captured"
          "5678 packets received by filter"
          "9 packets dropped by kernel"
          "12 packets dropped by tcpdump" (0.9.x, optional)
        Missing lines simply stay absent (get with a 0 default).
        """
        stats: dict = {}
        if not stderr_text:
            return stats
        pats = {
            "captured": r"(\d+)\s+packets\s+captured",
            "received_by_filter": r"(\d+)\s+packets\s+received\s+by\s+filter",
            "received_by_kernel": r"(\d+)\s+packets\s+received\s+by\s+kernel",
            "dropped_by_kernel": r"(\d+)\s+packets\s+dropped\s+by\s+kernel",
            "dropped_by_tcpdump": r"(\d+)\s+packets\s+dropped\s+by\s+tcpdump",
        }
        for key, pat in pats.items():
            m = re.search(pat, stderr_text)
            if m:
                stats[key] = int(m.group(1))
        return stats

    def is_running(self) -> bool:
        """Check if capture is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def get_capture_stats(self, pcap_file: Path) -> dict:
        """
        Get basic statistics from a capture file.

        Args:
            pcap_file: Path to pcap file

        Returns:
            Dictionary with capture statistics
        """
        try:
            # Use capinfos if available
            result = subprocess.run(
                ["capinfos", "-c", "-s", "-u", str(pcap_file)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                stats = {}
                for line in result.stdout.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        stats[key.strip()] = value.strip()
                return stats

        except FileNotFoundError:
            logger.debug("capinfos not available")
        except Exception as e:
            logger.error(f"Error getting capture stats: {e}")

        # Fallback: just return file size
        if pcap_file.exists():
            return {
                "file": str(pcap_file),
                "size_bytes": pcap_file.stat().st_size
            }

        return {}


# Convenience functions
def start_capture(
    pcap_file: str,
    interface: str = "eth0",
    filter_expr: Optional[str] = None
) -> Optional[subprocess.Popen]:
    """Start a simple tcpdump capture."""
    cmd = ["sudo", "tcpdump", "-i", interface, "-w", pcap_file, "-U"]
    if filter_expr:
        cmd.extend(filter_expr.split())

    try:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        logger.error(f"Failed to start capture: {e}")
        return None


def stop_capture(proc: subprocess.Popen) -> None:
    """Stop a tcpdump capture."""
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
