# 6G AI Traffic Characterization Testbed

A framework for measuring and analyzing AI/LLM service traffic patterns under various network conditions, designed to support 3GPP SA4 6G Media Study contributions.

## Components

| Component | Description |
|-----------|-------------|
| [aitestbed/](./aitestbed/) | Main testing framework for running AI traffic experiments |
| [netemu/](./netemu/) | Network emulation library wrapping Linux tc/netem |

## aitestbed

The core testing framework (~15K lines Python) that orchestrates experiments across multiple AI providers and scenarios.

**Features:**
- **11 scenario types**: Chat, agentic AI with MCP tools, image generation, multimodal, video understanding, realtime WebSocket/WebRTC
- **8 LLM providers**: OpenAI, Gemini, DeepSeek, vLLM, and realtime variants
- **60+ 3GPP-aligned metrics**: TTFT/TTLT, latency percentiles, UL/DL ratios, token rates, agent loop factors
- **Multi-layer traffic capture**: L3/L4 via tcpdump, L7 via mitmproxy
- **SQLite logging** with structured metrics schema

```bash
cd aitestbed
pip install -r requirements.txt
python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10
```

## netemu

A lightweight network emulation library (~500 lines Python) providing a clean interface to Linux traffic control.

**Features:**
- Wraps `tc` and `netem` for delay, jitter, packet loss, and rate limiting
- Bidirectional shaping via IFB devices
- 27 predefined profiles including 3GPP 5QI mappings
- Context manager support for automatic cleanup

```python
from netemu import NetworkEmulator

with NetworkEmulator(interface="eth0") as emu:
    emu.apply_profile("5g_urban")  # 20ms delay, 0.1% loss, 100 Mbps
    # Run your tests here
# Rules automatically cleared
```

## Quick Start

```bash
# Clone and setup
cd testbed
python -m venv venv
source venv/bin/activate
pip install -r aitestbed/requirements.txt

# Set API keys
export OPENAI_API_KEY="your-key"

# Run a basic experiment
python aitestbed/orchestrator.py --scenario chat_basic --profile ideal_6g --runs 5
```

## Docker

```bash
docker build -t 6g-ai-testbed -f aitestbed/Dockerfile .
docker run --cap-add=NET_ADMIN -e OPENAI_API_KEY="..." \
  6g-ai-testbed python orchestrator.py --scenario all --runs 10
```

## Network Profiles

| Profile | Delay | Loss | Rate | Use Case |
|---------|-------|------|------|----------|
| `ideal_6g` | 1ms | 0% | unlimited | Baseline |
| `5g_urban` | 20ms | 0.1% | 100 Mbps | Urban 5G |
| `wifi_good` | 30ms | 0.1% | 50 Mbps | Home WiFi |
| `cell_edge` | 120ms | 1% | 5 Mbps | Poor coverage |
| `satellite` | 600ms | 0.5% | 10 Mbps | LEO satellite |

## Requirements

- Python 3.10+
- Linux with `iproute2` (for network emulation)
- Sudo access or Docker with `NET_ADMIN` capability

## License

Internal use for 3GPP SA4 contributions.
