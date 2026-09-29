<p align="center">
  <img src=".github/banner.svg" width="100%" alt="Testbeds · 6G AI Traffic Characterization Testbed: 6G AI Traffic Characterization Testbed">
</p>

<p align="center">
  Measures and analyses AI/LLM service traffic under emulated network conditions, to support
  3GPP SA4 6G Media Study contributions.
</p>

<p align="center">
  <img alt="Status: under development"
    src="https://img.shields.io/badge/Status-Under%20Development-e67e22">
  <a href="https://github.com/5G-MAG/6G-Testbed/releases"><img alt="Version"
    src="https://img.shields.io/github/v/release/5G-MAG/6G-Testbed?label=Version"></a>
  <a href="LICENSE.md"><img alt="License: 5G-MAG Public License v1.0"
    src="https://img.shields.io/badge/License-5G--MAG%20PL%20v1.0-blue"></a>
</p>

<p align="center">
  <a href="https://www.5g-mag.com/testbeds/6g-testbed/">Project page</a> &nbsp;&middot;&nbsp;
  <a href="https://github.com/5G-MAG/6G-Testbed/issues">Issues</a> &nbsp;&middot;&nbsp;
  <a href="https://www.5g-mag.com/contributing">Contributing</a>
</p>

---

## At a glance

|  |  |
|---|---|
| **Supports** | 3GPP SA4 6G Media Study contributions; network profiles aligned with SA4 contribution S4-260848, Table C.Z-1 |
| **Part of** | [6G AI Traffic Characterization Testbed](https://www.5g-mag.com/testbeds/6g-testbed/) |

## Introduction

The testbed runs AI service scenarios against LLM providers while shaping the network with Linux
`tc`/`netem`, captures the resulting traffic, and logs metrics for SA4 contributions. It has two
parts: `aitestbed`, the experiment framework, and `netemu`, the network emulation library it uses.

| Component | Description |
|-----------|-------------|
| [aitestbed/](./aitestbed/) | Main testing framework for running AI traffic experiments |
| [netemu/](./netemu/) | Network emulation library wrapping Linux tc/netem |

### aitestbed

Orchestrates experiments across AI providers and scenarios:

- Scenarios: chat (including chat with token IDs), agentic AI with MCP tools, browser automation,
  image generation, multimodal, video understanding, realtime audio and conversation over WebSocket
  and WebRTC, and realtime video understanding with a local VLM. They are defined in
  `configs/scenarios.yaml`.
- Providers: OpenAI, Azure OpenAI, Azure AI Inference, Gemini, DeepSeek, vLLM, OpenAI Realtime (WebSocket and WebRTC),
  and OpenAI-compatible servers.
- Metrics: TTFT/TTLT, latency percentiles, UL/DL ratios, token rates and agent loop factors,
  documented in [METRICS.md](aitestbed/METRICS.md).
- Traffic capture at L3/L4 (tcpdump) and L7 (mitmproxy), with metrics logged to SQLite.

```bash
# From the repo root:
pip install -e netemu
pip install -r aitestbed/requirements.txt
cd aitestbed
python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10
```

The full SA4 cross-check run (all scenarios and profiles, with PCAP capture and report generation)
is described in [aitestbed/README.md](aitestbed/README.md), section "Cross-Checking for SA4 AI Traffic
Characterization".

### netemu

A Python library over Linux traffic control:

- Wraps `tc` and `netem` for delay, jitter, packet loss and rate limiting
- Bidirectional shaping through IFB devices
- Predefined profiles, including 3GPP 5QI mappings and the SA4 S4-260848 reference conditions
- Context-manager use, which clears the rules on exit

```python
from netemu import NetworkEmulator

with NetworkEmulator(interface="eth0") as emu:
    emu.apply_profile("5g_urban")  # 20ms delay, 0.1% loss, 100 Mbps
    # Run your tests here
# Rules automatically cleared
```

## Install dependencies

- Python 3.10 or later
- Linux with `iproute2`, for network emulation
- Sudo access, or Docker with the `NET_ADMIN` capability

## Running

### Quick start

```bash
# Clone and setup
git clone https://github.com/5G-MAG/6G-Testbed.git
cd 6G-Testbed
python -m venv venv
source venv/bin/activate

# Install netemu first (separate package), then testbed dependencies
pip install -e netemu
pip install -r aitestbed/requirements.txt

# Set API keys
export OPENAI_API_KEY="your-key"

# Run a basic experiment
cd aitestbed
python orchestrator.py --scenario chat_basic --profile 6g_itu_hrllc --runs 5
```

### Docker

Build the image from the repository root, then run experiments in it:

```bash
docker build -t 6g-ai-testbed -f aitestbed/Dockerfile .
docker run --cap-add=NET_ADMIN -e OPENAI_API_KEY="..." \
  6g-ai-testbed python orchestrator.py --scenario all --runs 10
```

## Configuration

### Network profiles

The test matrix, from `aitestbed/configs/profiles.yaml`, aligned with SA4 contribution S4-260848,
Table C.Z-1:

| Profile | Delay | Jitter | Loss | Loss Distribution | Rate | Use Case |
|---------|-------|--------|------|-------------------|------|----------|
| `no_emulation` | 0 ms | 0 ms | 0% | -- | -- | Reference (no tc/netem) |
| `6g_itu_hrllc` | 1 ms | 0.2 ms | 0.001% | correlated (10%) | 300 Mbps | 6G HRLLC (ITU IMT-2030 / M.2160) |
| `5g_urban` | 20 ms | 5 ms | 0.1% | correlated (25%) | 100 Mbps | Mainstream urban cellular |
| `wifi_good` | 30 ms | 10 ms | 0.1% | correlated (30%) | 50 Mbps | Home/office WiFi |
| `cell_edge` | 120 ms | 30 ms | 1% | Gilbert-Elliot | 5 Mbps | Weak radio, heavy-tail jitter |
| `satellite_leo` | 22 ms | 7 ms DL / 8 ms UL | 0.5% DL / 0.8% UL | correlated (40% DL / 45% UL) | 100 DL / 15 UL Mbps | LEO satellite (asymmetric) |
| `satellite_geo` | 340 ms | 15 ms DL / 18 ms UL | 0.1% DL / 0.2% UL | correlated (20% DL / 25% UL) | 50 DL / 3 UL Mbps | GEO satellite (asymmetric) |
| `congested` | 200 ms | 50 ms | 3% | Gilbert-Elliot | 1 Mbps | Bufferbloat / heavy congestion |
| `5qi_7` | 80 ms | 10 ms | 0.1% | correlated (20%) | -- | 5QI 7: voice / live streaming |
| `5qi_80` | 8 ms | 1 ms | 0.0001% | correlated (5%) | -- | 5QI 80: low-latency eMBB / AR |

The asymmetric profiles (`satellite_leo`, `satellite_geo`) use an optional `uplink:` block that
overrides the egress-side fields. The full table, with jitter, loss models and advanced `netem`
parameters, is in [aitestbed/README.md](aitestbed/README.md).

## Contributing

Contributions are welcome. How to raise an issue, fork the repository and open a pull request, and
the Contributor License Agreement required before code can be merged, are described at
<https://www.5g-mag.com/contributing>.

## License

Distributed under the 5G-MAG Public License v1.0. See [LICENSE.md](LICENSE.md). Third-party
software, models and datasets the testbed uses are listed in [ATTRIBUTION_NOTICE](ATTRIBUTION_NOTICE).
