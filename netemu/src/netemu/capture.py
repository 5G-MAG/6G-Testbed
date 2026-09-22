"""
L3/L4 packet capture via tcpdump.

Writes libpcap files alongside a JSON metadata sidecar recording the
interface, BPF filter, and start/end wall-clock times, so a capture can
later be correlated with whatever the caller was doing while it ran.

This is the write side of :mod:`netemu.pcap`, which reads the resulting
files back and derives metrics. Capturing on the same interface that
:class:`~netemu.emulator.NetworkEmulator` shapes yields packet traces of
traffic as it appears under the emulated profile.

tcpdump is invoked through ``sudo``; see the sudoers guidance in the README.
For L7 (decrypted HTTP) capture, use a TLS-terminating proxy such as
mitmproxy instead, that is outside netemu's scope.
"""

import json
import logging
import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CaptureController:
    """
    Controls packet capture for traffic analysis.

    Uses tcpdump for L3/L4 capture. Can be used directly::

        cap = CaptureController(interface="eth0", capture_dir="captures")
        cap.start(filename="run.pcap", filter_expr="port 443")
        ...
        pcap_path = cap.stop()

    or as a context manager via :func:`capture_to`, which stops the capture
    even if the body raises.
    """

    def __init__(
        self,
        interface: str = "eth0",
        capture_dir: str = "captures"
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
        self._metadata: dict = {}

    def start(
        self,
        filename: Optional[str] = None,
        filter_expr: Optional[str] = None,
        metadata: Optional[dict] = None,
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
            cmd.extend(shlex.split(filter_expr))

        try:
            logger.info(f"Starting capture: {' '.join(cmd)}")
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self._metadata = {
                "schema_version": 1,
                "pcap_file": self._current_file.name,
                "interface": self.interface,
                "filter": filter_expr or "",
                "t_start": time.time(),
                **(metadata or {}),
            }
            # Popen success only means sudo/tcpdump was exec'd. Detect bad
            # interfaces, permissions, and invalid BPF before scenarios run.
            time.sleep(0.25)
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace")
                logger.error("tcpdump exited during startup: %s", stderr.strip())
                self._process = None
                self._current_file = None
                self._metadata = {}
                return None
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
        return_code = self._process.returncode
        self._process = None
        self._current_file = None
        if return_code not in (0, -15):
            logger.error("tcpdump stopped with exit code %s", return_code)
            return None
        if result is None or not result.exists() or result.stat().st_size < 24:
            logger.error("Capture did not produce a valid pcap header: %s", result)
            return None
        self._metadata["t_end"] = time.time()
        self._metadata["size_bytes"] = result.stat().st_size
        sidecar = result.with_suffix(result.suffix + ".metadata.json")
        sidecar.write_text(json.dumps(self._metadata, indent=2))
        try:
            os.chmod(sidecar, 0o600)
        except OSError:
            pass
        self._metadata = {}

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


@contextmanager
def capture_to(
    pcap_file: str,
    interface: str = "eth0",
    filter_expr: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """
    Capture to ``pcap_file`` for the duration of the ``with`` block.

    Yields the :class:`CaptureController` so the body can query it, and
    stops the capture on exit even if the body raises::

        from netemu import NetworkEmulator, capture_to, analyze_pcap

        with NetworkEmulator(interface="eth0") as emu:
            emu.apply_profile("cell_edge")
            with capture_to("run.pcap", interface="eth0", filter_expr="port 443"):
                run_workload()

        metrics = analyze_pcap("run.pcap")

    Args:
        pcap_file: Output path for the capture.
        interface: Interface to capture on.
        filter_expr: BPF filter expression (e.g. ``"port 443"``).
        metadata: Extra fields to merge into the ``.metadata.json`` sidecar.
    """
    path = Path(pcap_file)
    controller = CaptureController(
        interface=interface, capture_dir=str(path.parent or ".")
    )
    controller.start(filename=path.name, filter_expr=filter_expr, metadata=metadata)
    try:
        yield controller
    finally:
        controller.stop()


# Convenience functions
def start_capture(
    pcap_file: str,
    interface: str = "eth0",
    filter_expr: Optional[str] = None
) -> Optional[subprocess.Popen]:
    """Start a simple tcpdump capture."""
    cmd = ["sudo", "tcpdump", "-i", interface, "-w", pcap_file, "-U"]
    if filter_expr:
        cmd.extend(shlex.split(filter_expr))

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
