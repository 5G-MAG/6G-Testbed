"""Packaged configuration assets for the AI traffic testbed."""

# Capture HTTPS/HTTP control traffic plus the UDP transports used by ICE,
# STUN, TURN, DTLS, SRTP/RTP, and RTCP. WebRTC local candidate ports are
# ephemeral, but the service endpoint uses one of these well-known ports, so
# BPF's bidirectional ``port`` match captures the complete media five-tuple.
WEBRTC_UDP_PORTS = (3478, 3479, 5349, 5350, 19302)
DEFAULT_CAPTURE_FILTER = (
    "port 443 or port 80 or port 8080 or port 8000 or "
    + " or ".join(f"udp port {port}" for port in WEBRTC_UDP_PORTS)
)

__all__ = ["DEFAULT_CAPTURE_FILTER", "WEBRTC_UDP_PORTS"]
