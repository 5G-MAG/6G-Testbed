"""
Capture Controller for the 6G AI Traffic Testbed.

Manages packet capture using tcpdump.
"""

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
        capture_dir: str = "capture/captures"
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

    def start(
        self,
        filename: Optional[str] = None,
        filter_expr: Optional[str] = None
    ) -> Optional[Path]:
        """
        Start packet capture.

        Args:
            filename: Output filename (auto-generated if not provided)
            filter_expr: BPF filter expression (e.g., "port 443")

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

        cmd = [
            "sudo", "tcpdump",
            "-i", self.interface,
            "-w", str(self._current_file),
            "-U",  # Packet-buffered output
        ]

        if filter_expr:
            cmd.extend(filter_expr.split())

        try:
            logger.info(f"Starting capture: {' '.join(cmd)}")
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return self._current_file

        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            self._process = None
            self._current_file = None
            return None

    def stop(self) -> Optional[Path]:
        """
        Stop packet capture.

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

        result = self._current_file
        self._process = None
        self._current_file = None

        logger.info(f"Capture stopped: {result}")
        return result

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
