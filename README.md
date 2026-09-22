# 6G AI Traffic Characterization Testbed

A framework for measuring, analyzing, and modelling AI/LLM service traffic under emulated network conditions, built to support 3GPP SA4 6G Media Study contributions.

## Components

| Component | Role | README |
|:----------|:-----|:-------|
| [netemu/](./netemu/) | Network emulation, packet capture, and pcap metric extraction. Standalone package, no dependency on the testbed | [netemu/README.md](./netemu/README.md) |
| [aitestbed/](./aitestbed/) | Experiment orchestration: scenarios, LLM/agent clients, application-layer metrics, reports. Depends on `netemu` | [aitestbed/README.md](./aitestbed/README.md) |
| [training/](./training/) | Traffic-pattern dataset builder and Markov traffic generators trained on the captures `aitestbed` produces | [training/README.md](./training/README.md) |

The dependency direction is one-way:

```
   training/            reads captures + labels produced by aitestbed
       │
       ▼
   aitestbed/           orchestrates experiments, computes application-layer metrics
       │
       ▼
   netemu/              shapes the network, captures packets, parses pcaps
```

`netemu` knows nothing about the testbed and is usable on its own. `aitestbed`
imports `netemu` for shaping, capture, and pcap parsing. `training` consumes
`aitestbed` output files but imports no testbed code.

## What lives where

The measurement stack is split by layer, not by convenience:

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

## netemu

Linux network emulation plus the capture and analysis path that goes with it.

**Features:**
- Wraps `tc`/`netem`/HTB for delay, jitter, loss, rate limiting, corruption, reordering, duplication
- Bidirectional shaping via IFB devices; asymmetric uplink/downlink profiles
- `tcpdump` capture with startup validation and a JSON metadata sidecar per pcap
- PCAP parsing (libpcap and pcapng, IPv4/IPv6, TCP/UDP) with TCP flow reassembly
- Network-layer metrics: handshake RTT, TLS setup duration, retransmissions, per-direction volumes, multi-window throughput and burstiness, burst segmentation
- Predefined profiles including 3GPP 5QI mappings and SA4 S4-260848 reference conditions
- Context managers for automatic cleanup

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

See [netemu/README.md](./netemu/README.md) for the full pcap-processing and metric-calculation documentation.

## aitestbed

The testing framework that orchestrates experiments across AI providers, scenarios, and network profiles.

**Features:**
- **Scenario families**: chat, agentic AI over MCP, image generation, multimodal, video understanding, realtime WebSocket/WebRTC, browser automation, OpenClaw personal-assistant agent, A2A agent-to-agent protocol
- **LLM providers**: OpenAI, Gemini, DeepSeek, Anthropic, Azure, self-hosted vLLM, plus OpenAI Realtime (WebSocket/WebRTC)
- **60+ metrics**: TTFT/TTLT, latency percentiles, UL/DL ratios, token rates, agent loop factors, stall detection, plus the RAN2 S4-260859 Q1-Q5 families
- **Multi-layer capture**: L3/L4 via `netemu.capture`, L7 via mitmproxy
- **SQLite logging** with a structured metrics schema, and anonymization for submission

```bash
# From the repo root:
pip install -e "netemu[pcap]"
pip install -r aitestbed/requirements.txt
cd aitestbed
python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10
```

For the full SA4 cross-check run (all scenarios x profiles, with PCAP capture and report generation), see the **Cross-Checking for SA4 AI Traffic Characterization** section in [aitestbed/README.md](./aitestbed/README.md).

## training

Traffic-pattern models learned from the captures the testbed produces.

**Features:**
- Feature extraction from pcaps plus SQLite session labels, with flows attributed to a transport surface and segmented at session boundaries
- Nine traffic-pattern categories, including agent-to-agent signaling and local agent control channels
- A shared quantization codec: log-spaced size and inter-arrival bins with empirical within-bin dequantization
- Zero and first-order category-conditioned Markov generators, with per-category sampling and held-out KS evaluation

```bash
cd training
pip install -r requirements.txt
python -m dataset --captures-dir ../aitestbed/results/captures \
    --db-path ../aitestbed/logs/traffic_logs.db --output-dir data --max-packets 100
python -m train_markov --data-dir data
```

See [training/README.md](./training/README.md).

## Quick Start

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

## Docker

```bash
docker build -t 6g-ai-testbed -f aitestbed/Dockerfile .
docker run --cap-add=NET_ADMIN -e OPENAI_API_KEY="..." \
  6g-ai-testbed python orchestrator.py --scenario all --runs 10
```

## Network Profiles

Current test matrix (from `aitestbed/configs/profiles.yaml`, aligned with 3GPP SA4 S4-260848 Table C.Z-1):

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
| `5qi_80` | 8 ms | 1 ms | 1e-6 | correlated (5%) | -- | 5QI 80: low-latency eMBB / AR |

Asymmetric profiles (`satellite_leo`, `satellite_geo`) use an optional `uplink:` block that overrides egress-side fields. See [aitestbed/README.md](./aitestbed/README.md) for the full table with jitter, loss models, and advanced netem parameters.

## Requirements

- Python 3.10+
- Linux with `iproute2` (for network emulation)
- `tcpdump` (for packet capture)
- Sudo access or Docker with `NET_ADMIN` capability
- Node.js 18+ (for npm-based MCP servers; 22+ for the OpenClaw scenario)

## Testing

```bash
python -m pytest netemu/tests      # emulation, capture, pcap analysis
python -m pytest aitestbed/tests   # testbed correctness checks
python -m pytest training/tests    # dataset and model unit tests
```

## License

See [LICENSE.md](./LICENSE.md).
