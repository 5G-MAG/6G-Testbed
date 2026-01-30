"""
L7 (Application Layer) Capture for the 6G AI Traffic Testbed.

Uses mitmproxy to capture full HTTP/HTTPS request/response details
including headers, bodies, and timing metrics.
"""

import json
import time
import subprocess
import signal
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class L7Record:
    """
    A single L7 (HTTP) traffic record.
    """
    timestamp: float
    flow_id: str

    # Request details
    request_method: str = ""
    request_url: str = ""
    request_host: str = ""
    request_path: str = ""
    request_headers: dict = field(default_factory=dict)
    request_body_size: int = 0
    request_content_type: str = ""

    # Response details
    response_status: int = 0
    response_headers: dict = field(default_factory=dict)
    response_body_size: int = 0
    response_content_type: str = ""

    # Timing metrics (seconds)
    t_request_start: float = 0.0
    t_request_end: float = 0.0
    t_response_start: float = 0.0
    t_response_end: float = 0.0

    # Computed metrics
    ttfb: float = 0.0  # Time to first byte
    total_time: float = 0.0

    # TLS info
    tls_version: str = ""
    tls_cipher: str = ""

    # Metadata
    scenario_id: str = ""
    session_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class MitmproxyAddon:
    """
    Mitmproxy addon script for capturing L7 traffic.

    This is used as a standalone script by mitmproxy.
    """

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.records = []

    def request(self, flow):
        """Called when a request is received."""
        flow.metadata["t_request_start"] = time.time()

    def response(self, flow):
        """Called when a response is received."""
        t_response_end = time.time()
        t_request_start = flow.metadata.get("t_request_start", t_response_end)

        record = {
            "timestamp": t_response_end,
            "flow_id": flow.id,
            "request_method": flow.request.method,
            "request_url": flow.request.url,
            "request_host": flow.request.host,
            "request_path": flow.request.path,
            "request_headers": dict(flow.request.headers),
            "request_body_size": len(flow.request.content) if flow.request.content else 0,
            "request_content_type": flow.request.headers.get("content-type", ""),
            "response_status": flow.response.status_code,
            "response_headers": dict(flow.response.headers),
            "response_body_size": len(flow.response.content) if flow.response.content else 0,
            "response_content_type": flow.response.headers.get("content-type", ""),
            "t_request_start": t_request_start,
            "t_response_end": t_response_end,
            "total_time": t_response_end - t_request_start,
        }

        # Add TLS info if available
        if flow.server_conn and flow.server_conn.tls_version:
            record["tls_version"] = flow.server_conn.tls_version
            record["tls_cipher"] = flow.server_conn.cipher or ""

        self.records.append(record)

        # Write incrementally
        with open(self.output_file, "a") as f:
            f.write(json.dumps(record) + "\n")


# Generate addon script for mitmproxy
ADDON_SCRIPT_TEMPLATE = '''
"""Auto-generated mitmproxy addon script."""
import json
import time

OUTPUT_FILE = "{output_file}"

class L7CaptureAddon:
    def request(self, flow):
        flow.metadata["t_request_start"] = time.time()

    def response(self, flow):
        t_end = time.time()
        t_start = flow.metadata.get("t_request_start", t_end)

        record = {{
            "timestamp": t_end,
            "flow_id": flow.id,
            "request_method": flow.request.method,
            "request_url": flow.request.url,
            "request_host": flow.request.host,
            "request_path": flow.request.path,
            "request_headers": dict(flow.request.headers),
            "request_body_size": len(flow.request.content) if flow.request.content else 0,
            "request_content_type": flow.request.headers.get("content-type", ""),
            "response_status": flow.response.status_code,
            "response_headers": dict(flow.response.headers),
            "response_body_size": len(flow.response.content) if flow.response.content else 0,
            "response_content_type": flow.response.headers.get("content-type", ""),
            "t_request_start": t_start,
            "t_response_end": t_end,
            "total_time": t_end - t_start,
        }}

        if flow.server_conn and hasattr(flow.server_conn, 'tls_version') and flow.server_conn.tls_version:
            record["tls_version"] = flow.server_conn.tls_version
            record["tls_cipher"] = getattr(flow.server_conn, 'cipher', '') or ""

        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(record) + "\\n")

addons = [L7CaptureAddon()]
'''


class L7CaptureController:
    """
    Controller for L7 traffic capture using mitmproxy.

    Manages mitmproxy process lifecycle and provides access to captured data.
    """

    def __init__(
        self,
        capture_dir: str = "capture/l7_captures",
        proxy_port: int = 8080,
        web_port: int = 8081
    ):
        """
        Initialize L7 capture controller.

        Args:
            capture_dir: Directory for capture files
            proxy_port: Port for mitmproxy to listen on
            web_port: Port for mitmproxy web interface (0 to disable)
        """
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.proxy_port = proxy_port
        self.web_port = web_port
        self._process: Optional[subprocess.Popen] = None
        self._current_file: Optional[Path] = None
        self._addon_script: Optional[Path] = None

    def start(
        self,
        filename: Optional[str] = None,
        filter_hosts: Optional[list[str]] = None
    ) -> Optional[Path]:
        """
        Start L7 capture.

        Args:
            filename: Output filename (auto-generated if not provided)
            filter_hosts: List of hosts to capture (None for all)

        Returns:
            Path to the capture file, or None if failed
        """
        if self._process is not None:
            logger.warning("L7 capture already running")
            return self._current_file

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"l7_capture_{timestamp}.jsonl"

        self._current_file = self.capture_dir / filename

        # Create addon script
        self._addon_script = self.capture_dir / f"_addon_{timestamp}.py"
        addon_content = ADDON_SCRIPT_TEMPLATE.format(
            output_file=str(self._current_file.absolute())
        )
        self._addon_script.write_text(addon_content)

        # Build mitmproxy command
        cmd = [
            "mitmdump",
            "--listen-port", str(self.proxy_port),
            "--set", "stream_large_bodies=1m",
            "-s", str(self._addon_script),
        ]

        # Add host filter if specified
        if filter_hosts:
            filter_expr = " | ".join(f"~d {host}" for host in filter_hosts)
            cmd.extend(["--filter", filter_expr])

        # Add web interface if enabled
        if self.web_port > 0:
            cmd.extend(["--web-port", str(self.web_port)])

        try:
            logger.info(f"Starting mitmproxy: {' '.join(cmd)}")

            # Clear output file
            self._current_file.write_text("")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait a moment for startup
            time.sleep(1)

            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode()
                logger.error(f"mitmproxy failed to start: {stderr}")
                self._cleanup()
                return None

            logger.info(f"L7 capture started on port {self.proxy_port}")
            return self._current_file

        except FileNotFoundError:
            logger.error("mitmproxy not found. Install with: pip install mitmproxy")
            self._cleanup()
            return None
        except Exception as e:
            logger.error(f"Failed to start L7 capture: {e}")
            self._cleanup()
            return None

    def stop(self) -> Optional[Path]:
        """
        Stop L7 capture.

        Returns:
            Path to the capture file
        """
        if self._process is None:
            logger.warning("No L7 capture running")
            return None

        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

        result = self._current_file
        self._cleanup()

        logger.info(f"L7 capture stopped: {result}")
        return result

    def _cleanup(self):
        """Clean up resources."""
        self._process = None
        self._current_file = None
        if self._addon_script and self._addon_script.exists():
            try:
                self._addon_script.unlink()
            except Exception:
                pass
        self._addon_script = None

    def is_running(self) -> bool:
        """Check if capture is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def get_proxy_url(self) -> str:
        """Get the proxy URL for client configuration."""
        return f"http://localhost:{self.proxy_port}"

    def read_records(self, capture_file: Optional[Path] = None) -> list[L7Record]:
        """
        Read captured records from file.

        Args:
            capture_file: Path to capture file (uses current if not specified)

        Returns:
            List of L7Record objects
        """
        file_path = capture_file or self._current_file
        if file_path is None or not file_path.exists():
            return []

        records = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        records.append(L7Record(**data))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse record: {e}")

        return records

    def get_summary(self, capture_file: Optional[Path] = None) -> dict:
        """
        Get summary statistics from captured data.

        Returns:
            Dictionary with summary statistics
        """
        records = self.read_records(capture_file)

        if not records:
            return {"count": 0}

        total_request_bytes = sum(r.request_body_size for r in records)
        total_response_bytes = sum(r.response_body_size for r in records)
        latencies = [r.total_time for r in records if r.total_time > 0]

        hosts = {}
        for r in records:
            hosts[r.request_host] = hosts.get(r.request_host, 0) + 1

        return {
            "count": len(records),
            "total_request_bytes": total_request_bytes,
            "total_response_bytes": total_response_bytes,
            "avg_latency_sec": sum(latencies) / len(latencies) if latencies else 0,
            "hosts": hosts,
            "methods": {
                method: sum(1 for r in records if r.request_method == method)
                for method in set(r.request_method for r in records)
            }
        }


def configure_client_proxy(proxy_url: str = "http://localhost:8080"):
    """
    Configure environment for proxy usage.

    Sets HTTP_PROXY and HTTPS_PROXY environment variables.
    """
    import os
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url
    logger.info(f"Configured proxy: {proxy_url}")


def clear_client_proxy():
    """Clear proxy environment variables."""
    import os
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ.pop(var, None)
    logger.info("Cleared proxy configuration")
