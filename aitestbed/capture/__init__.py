"""
Capture module for the 6G AI Traffic Testbed.

Provides packet capture and L7 logging capabilities.
"""

from .controller import CaptureController, start_capture, stop_capture
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
