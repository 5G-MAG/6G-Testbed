<p align="center">
  <img src=".github/banner.svg" width="100%" alt="Testbeds · 6G AI Traffic Characterization Testbed: 6G AI Traffic Characterization Testbed">
</p>

<p align="center">
  Measures, analyses and models AI/LLM service traffic under emulated network conditions, to
  support 3GPP SA4 6G Media Study contributions.
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
`tc`/`netem`, captures the resulting traffic, and logs metrics for SA4 contributions. It has three
parts: `netemu`, the network emulation and capture library; `aitestbed`, the experiment framework
built on it; and `training`, which learns traffic-pattern models from the captures `aitestbed`
produces.

### Components

| Component | Role | README |
|:----------|:-----|:-------|
| [netemu/](./netemu/) | Network emulation, packet capture, and pcap metric extraction. Standalone package, no dependency on the testbed | [netemu/README.md](./netemu/README.md) |
| [aitestbed/](./aitestbed/) | Experiment orchestration: scenarios, LLM/agent clients, application-layer metrics, reports. Depends on `netemu` | [aitestbed/README.md](./aitestbed/README.md) |
| [training/](./training/) | Traffic-pattern dataset builder and Markov traffic generators trained on the captures `aitestbed` produces | [training/README.md](./training/README.md) |

The dependencies run one way:

```
   training/            reads captures + labels produced by aitestbed
       │
       ▼
   aitestbed/           orchestrates experiments, computes application-layer metrics
       │
       ▼
   netemu/              shapes the network, captures packets, parses pcaps
```

`netemu` does not depend on the testbed and can be used on its own. `aitestbed` imports `netemu`
for shaping, capture and pcap parsing. `training` reads `aitestbed` output files but imports no
testbed code.

### What lives where

The measurement stack is split by layer:

| Concern | Where | Why there |
|:--------|:------|:----------|
| tc/netem shaping | `netemu.emulator` | Network layer, task-agnostic |
| tcpdump capture | `netemu.capture` | Produces pcaps; belongs with the thing that reads them |
| PCAP parsing, flow reassembly, packet/flow/window metrics | `netemu.pcap` | Pure network-layer analysis, no testbed coupling |
| L7 (decrypted HTTP) capture | `aitestbed/capture/l7_capture.py` | Needs TLS interception and payload redaction policy |
| Application-layer metrics (TTFT, TTLT, tokens, agent loops) | `aitestbed/analysis/metrics.py` | Defined against the testbed's log schema |
| RAN2 methodology metrics (S4-260859 Q1-Q5) | `aitestbed/analysis/ran2_metrics.py` | Combines pcap metrics with SQLite session records |
| Reports, charts, Excel export | `aitestbed/` | Contribution-shaped output |
| Feature extraction, Markov traffic generators | `training/` | Consumes captures; independent lifecycle |

### netemu

Linux network emulation, with the packet capture and analysis that go with it:

- Wraps `tc`/`netem`/HTB for delay, jitter, loss, rate limiting, corruption, reordering and
  duplication
- Bidirectional shaping through IFB devices, with asymmetric uplink and downlink profiles
- `tcpdump` capture with startup validation and a JSON metadata sidecar for each pcap
- PCAP parsing (libpcap and pcapng, IPv4 and IPv6, TCP and UDP) with TCP flow reassembly
- Network-layer metrics: handshake RTT, TLS setup duration, retransmissions, per-direction
  volumes, multi-window throughput and burstiness, burst segmentation
- Network profiles loaded from YAML; the testbed's profiles, including the 3GPP 5QI mappings and
  the SA4 S4-260848 reference conditions, are in `aitestbed/configs/profiles.yaml`
- Context managers that clear the rules and stop the capture on exit

```python
from netemu import NetworkEmulator, capture_to, analyze_pcap

with NetworkEmulator(interface="eth0") as emu:
    emu.apply_profile("5g_urban")           # 20 ms delay, 0.1% loss, 100 Mbps
    with capture_to("run.pcap", interface="eth0", filter_expr="port 443"):
        run_workload()
# tc rules cleared, capture stopped, sidecar written

m = analyze_pcap("run.pcap", target_ports=[443])
print(m.rtt_mean_ms, m.ul_bytes_total / m.dl_bytes_total, m.burstiness_by_window["100ms"])
```

The pcap processing and metric calculation are documented in
[netemu/README.md](./netemu/README.md).

### aitestbed

Orchestrates experiments across AI providers, scenarios and network profiles:

- Scenarios: chat (including chat with token IDs), agentic AI over MCP, browser automation,
  image generation, multimodal, video understanding, realtime audio and conversation over
  WebSocket and WebRTC, realtime video understanding with a local VLM, the OpenClaw
  personal-assistant agent, and agent-to-agent (A2A) scenarios. They are defined in
  `configs/scenarios.yaml`.
- Providers: OpenAI, Azure OpenAI, Azure AI Inference, Gemini, DeepSeek, Anthropic, self-hosted
  vLLM, OpenAI Realtime (WebSocket and WebRTC), and OpenAI-compatible servers.
- Metrics: TTFT/TTLT, latency percentiles, UL/DL ratios, token rates, agent loop factors and
  stall detection, documented in [METRICS.md](aitestbed/METRICS.md), plus the RAN2 methodology
  metrics for questions Q1 to Q5 of S4-260859.
- Capture at L3/L4 through `netemu.capture` and at L7 through mitmproxy, with metrics logged to
  SQLite in a structured schema and anonymisation of the logs for submission.

```bash
# From the repo root:
pip install -e "netemu[pcap]"
pip install -r aitestbed/requirements.txt
cd aitestbed
python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10
```

The full SA4 cross-check run (all scenarios and profiles, with PCAP capture and report generation)
is described in [aitestbed/README.md](./aitestbed/README.md), section "Cross-Checking for SA4 AI
Traffic Characterization".

### training

Traffic-pattern models learned from the captures the testbed produces:

- Feature extraction from pcaps and SQLite session labels, with flows attributed to a transport
  surface and segmented at session boundaries
- Nine traffic-pattern categories, including agent-to-agent signalling and local agent control
  channels
- A shared quantisation codec: log-spaced size and inter-arrival bins, with empirical
  within-bin dequantisation
- Zero- and first-order Markov generators conditioned on the category, with per-category
  sampling and a Kolmogorov-Smirnov (KS) evaluation on held-out data

```bash
cd training
pip install -r requirements.txt
python -m dataset --captures-dir ../aitestbed/results/captures \
    --db-path ../aitestbed/logs/traffic_logs.db --output-dir data --max-packets 100
python -m train_markov --data-dir data
```

More detail is in [training/README.md](./training/README.md).

## Install dependencies

- Python 3.10 or later
- Linux with `iproute2`, for network emulation
- `tcpdump`, for packet capture
- Sudo access, or Docker with the `NET_ADMIN` capability
- Node.js 18 or later for the npm-based MCP servers; 22 or later for the OpenClaw scenario

## Running

### Quick start

```bash
python -m venv venv
source venv/bin/activate

# Install netemu first (separate package, with the pcap extra), then the testbed
pip install -e "netemu[pcap]"
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
parameters, is in [aitestbed/README.md](./aitestbed/README.md).

## Development

### Testing

```bash
python -m pytest netemu/tests      # emulation, capture, pcap analysis
python -m pytest aitestbed/tests   # testbed correctness checks
python -m pytest training/tests    # dataset and model unit tests
```

## Contributing

Contributions are welcome. How to raise an issue, fork the repository and open a pull request, and
the Contributor License Agreement required before code can be merged, are described at
<https://www.5g-mag.com/contributing>.

## License

Distributed under the 5G-MAG Public License v1.0. See [LICENSE.md](LICENSE.md). Third-party
software, models and datasets the testbed uses are listed in [ATTRIBUTION_NOTICE](ATTRIBUTION_NOTICE).
