"""
Capture module for the 6G AI Traffic Testbed.

L3/L4 packet capture (tcpdump) lives in the sibling `netemu` package
(`netemu.capture`), so that shaping, capture, and pcap parsing stay
together; `CaptureController` is re-exported here for convenience.

L7 capture (mitmproxy-based HTTP/HTTPS interception with payload redaction)
is testbed-specific and stays here.
"""

from netemu.capture import CaptureController, start_capture, stop_capture

from .l7_capture import L7CaptureController, L7Record, configure_client_proxy, clear_client_proxy

__all__ = [
    "CaptureController",
    "start_capture",
    "stop_capture",
    "L7CaptureController",
    "L7Record",
    "configure_client_proxy",
    "clear_client_proxy",
]
