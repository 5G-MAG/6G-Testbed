#!/usr/bin/env python3
"""
Example: Using L7 Capture with the 6G AI Traffic Testbed

This example demonstrates how to capture full HTTP/HTTPS traffic
details when running LLM experiments.

Prerequisites:
    pip install mitmproxy

For HTTPS interception, you need to trust mitmproxy's CA certificate:
    1. Start mitmproxy once: mitmdump
    2. Find CA cert at: ~/.mitmproxy/mitmproxy-ca-cert.pem
    3. Install system-wide or configure your HTTP client to trust it

For OpenAI/httpx clients, set SSL_CERT_FILE or use verify=False (not recommended for production)
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture import L7CaptureController, configure_client_proxy, clear_client_proxy
from clients import OpenAIClient, ChatMessage, MessageRole


def example_basic_capture():
    """Basic L7 capture example."""
    print("=" * 60)
    print("Example 1: Basic L7 Capture")
    print("=" * 60)

    # Initialize L7 capture controller
    l7_capture = L7CaptureController(
        capture_dir="capture/l7_captures",
        proxy_port=8080,
        web_port=8081  # Web UI at http://localhost:8081
    )

    # Start capture
    capture_file = l7_capture.start(
        filename="openai_chat_capture.jsonl",
        filter_hosts=["api.openai.com"]  # Only capture OpenAI traffic
    )

    if capture_file is None:
        print("Failed to start L7 capture. Is mitmproxy installed?")
        return

    print(f"L7 capture started: {capture_file}")
    print(f"Proxy URL: {l7_capture.get_proxy_url()}")
    print(f"Web UI: http://localhost:8081")

    # Configure client to use proxy
    configure_client_proxy(l7_capture.get_proxy_url())

    # Note: For HTTPS, you need to either:
    # 1. Trust mitmproxy's CA certificate, or
    # 2. Disable SSL verification (not recommended)

    try:
        # Make some API calls
        # Note: This example won't work without proper SSL setup
        # See the SSL setup section below
        print("\nTo test, run API calls in another terminal with proxy configured:")
        print(f"  export HTTPS_PROXY={l7_capture.get_proxy_url()}")
        print("  curl -x http://localhost:8080 https://api.openai.com/v1/models")
        print("\nPress Ctrl+C to stop capture...")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass

    finally:
        # Stop capture and get results
        clear_client_proxy()
        capture_file = l7_capture.stop()

        # Read and display summary
        summary = l7_capture.get_summary(capture_file)
        print("\nCapture Summary:")
        print(f"  Total requests: {summary['count']}")
        print(f"  Request bytes: {summary.get('total_request_bytes', 0)}")
        print(f"  Response bytes: {summary.get('total_response_bytes', 0)}")
        if summary.get('hosts'):
            print(f"  Hosts: {summary['hosts']}")


def example_with_openai_client():
    """
    Example using L7 capture with OpenAI client.

    Requires mitmproxy CA certificate to be trusted.
    """
    print("=" * 60)
    print("Example 2: L7 Capture with OpenAI Client")
    print("=" * 60)

    # For this to work with HTTPS, you need to:
    # 1. Install mitmproxy CA cert, OR
    # 2. Set REQUESTS_CA_BUNDLE/SSL_CERT_FILE to mitmproxy cert, OR
    # 3. Use httpx with verify=False (development only)

    l7_capture = L7CaptureController(proxy_port=8080)

    # Start capture
    capture_file = l7_capture.start(filter_hosts=["api.openai.com"])
    if not capture_file:
        print("Failed to start capture")
        return

    # Configure proxy
    configure_client_proxy(l7_capture.get_proxy_url())

    # Option: Point to mitmproxy CA cert
    mitmproxy_ca = os.path.expanduser("~/.mitmproxy/mitmproxy-ca-cert.pem")
    if os.path.exists(mitmproxy_ca):
        os.environ["SSL_CERT_FILE"] = mitmproxy_ca
        os.environ["REQUESTS_CA_BUNDLE"] = mitmproxy_ca
        print(f"Using mitmproxy CA: {mitmproxy_ca}")
    else:
        print(f"Warning: mitmproxy CA not found at {mitmproxy_ca}")
        print("HTTPS interception may fail. Run 'mitmdump' once to generate it.")

    try:
        # Initialize OpenAI client
        client = OpenAIClient()

        # Make a chat request
        messages = [
            ChatMessage(role=MessageRole.USER, content="Say hello in exactly 5 words.")
        ]

        print("\nMaking API call...")
        response = client.chat(messages, model="gpt-5-mini", stream=False)

        print(f"Response: {response.content}")
        print(f"Latency: {response.latency_sec:.2f}s")
        print(f"Tokens: {response.tokens_in} in, {response.tokens_out} out")

    except Exception as e:
        print(f"Error: {e}")
        print("\nIf SSL error, ensure mitmproxy CA is trusted.")

    finally:
        clear_client_proxy()
        l7_capture.stop()

        # Show captured data
        records = l7_capture.read_records(capture_file)
        print(f"\nCaptured {len(records)} HTTP requests:")
        for r in records:
            print(f"  {r.request_method} {r.request_url[:60]}...")
            print(f"    Status: {r.response_status}")
            print(f"    Request size: {r.request_body_size} bytes")
            print(f"    Response size: {r.response_body_size} bytes")
            print(f"    Total time: {r.total_time:.3f}s")


def example_analyze_capture():
    """Example analyzing previously captured data."""
    print("=" * 60)
    print("Example 3: Analyze Captured Data")
    print("=" * 60)

    from pathlib import Path

    l7_capture = L7CaptureController()

    # Find capture files
    capture_dir = Path("capture/l7_captures")
    if not capture_dir.exists():
        print("No capture directory found")
        return

    capture_files = list(capture_dir.glob("*.jsonl"))
    if not capture_files:
        print("No capture files found")
        return

    print(f"Found {len(capture_files)} capture files:")

    for capture_file in capture_files:
        print(f"\n{capture_file.name}:")

        records = l7_capture.read_records(capture_file)
        if not records:
            print("  (empty)")
            continue

        # Analyze by host
        hosts = {}
        for r in records:
            if r.request_host not in hosts:
                hosts[r.request_host] = {
                    "count": 0,
                    "request_bytes": 0,
                    "response_bytes": 0,
                    "total_time": 0,
                }
            hosts[r.request_host]["count"] += 1
            hosts[r.request_host]["request_bytes"] += r.request_body_size
            hosts[r.request_host]["response_bytes"] += r.response_body_size
            hosts[r.request_host]["total_time"] += r.total_time

        for host, stats in hosts.items():
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            print(f"  {host}:")
            print(f"    Requests: {stats['count']}")
            print(f"    UL: {stats['request_bytes']} bytes")
            print(f"    DL: {stats['response_bytes']} bytes")
            print(f"    Avg time: {avg_time:.3f}s")


def setup_mitmproxy_ca():
    """Instructions for setting up mitmproxy CA certificate."""
    print("=" * 60)
    print("mitmproxy CA Certificate Setup")
    print("=" * 60)

    print("""
To capture HTTPS traffic, you need to trust mitmproxy's CA certificate.

1. Generate the CA certificate (run once):
   $ mitmdump
   (then press Ctrl+C)

2. The CA cert is created at:
   ~/.mitmproxy/mitmproxy-ca-cert.pem

3. Option A - System-wide trust (Linux):
   $ sudo cp ~/.mitmproxy/mitmproxy-ca-cert.pem /usr/local/share/ca-certificates/mitmproxy.crt
   $ sudo update-ca-certificates

4. Option B - Python/requests only:
   $ export SSL_CERT_FILE=~/.mitmproxy/mitmproxy-ca-cert.pem
   $ export REQUESTS_CA_BUNDLE=~/.mitmproxy/mitmproxy-ca-cert.pem

5. Option C - Per-client configuration:
   # OpenAI with httpx
   import httpx
   client = httpx.Client(verify="~/.mitmproxy/mitmproxy-ca-cert.pem")

6. Run the testbed with proxy:
   $ export HTTPS_PROXY=http://localhost:8080
   $ python orchestrator.py --scenario chat_basic

For development/testing only (insecure):
   $ export CURL_CA_BUNDLE=""
   $ export PYTHONHTTPSVERIFY=0
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L7 Capture Examples")
    parser.add_argument(
        "--example",
        choices=["basic", "openai", "analyze", "setup"],
        default="basic",
        help="Which example to run"
    )

    args = parser.parse_args()

    if args.example == "basic":
        example_basic_capture()
    elif args.example == "openai":
        example_with_openai_client()
    elif args.example == "analyze":
        example_analyze_capture()
    elif args.example == "setup":
        setup_mitmproxy_ca()
