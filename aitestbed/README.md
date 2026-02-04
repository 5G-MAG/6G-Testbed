# 6G AI Traffic Characterization Testbed 

A comprehensive testbed for evaluating and characterizing AI traffic patterns under various network conditions, aligned with **3GPP SA4 6G Media Study** objectives.

## Overview

This testbed enables 3GPP experts to:

- **Measure** traffic characteristics of generative AI services (LLMs, image generation, etc.)
- **Analyze** agentic AI patterns (multi-step tool calling, agent-to-agent communication)
- **Evaluate** QoE metrics under emulated network conditions (latency, loss, bandwidth)
- **Generate** reports for standardization contributions

The testbed supports multiple LLM providers (OpenAI, Google Gemini, DeepSeek, self-hosted LLMs) and implements various traffic scenarios that represent real-world AI service usage patterns.

## Features

### Traffic Scenarios

| Scenario | Description | Applicable Metrics |
|:---------|:------------|:-------------------|
| Chat Basic (`chat_basic`) | Basic single-turn chat interaction | Latency, TTFT/TTLT, tokens, bytes |
| Chat Streaming (`chat_streaming`) | Multi-turn chat with streaming responses | Latency, TTFT/TTLT, streaming chunks, tokens, bytes |
| Chat Basic (Gemini) (`chat_gemini`) | Chat interaction using Google Gemini | Latency, TTFT/TTLT, tokens, bytes |
| Chat Basic (DeepSeek) (`chat_deepseek`) | Basic chat interaction using DeepSeek | Latency, TTFT/TTLT, tokens, bytes |
| Chat Streaming (DeepSeek) (`chat_deepseek_streaming`) | Streaming chat interaction using DeepSeek | Latency, TTFT/TTLT, streaming chunks, tokens, bytes |
| Chat Coding (DeepSeek) (`chat_deepseek_coder`) | Code-focused chat using DeepSeek Coder | Latency, TTFT/TTLT, tokens, bytes |
| Chat Reasoning (DeepSeek) (`chat_deepseek_reasoner`) | Deep reasoning chat using DeepSeek Reasoner | Latency, TTFT/TTLT, tokens, bytes |
| Shopping Agent (`shopping_agent`) | Shopping assistant with tool calling | Latency, tokens, tool calls/latency, bytes |
| Shopping Agent (DeepSeek) (`shopping_agent_deepseek`) | Shopping assistant with tool calling using DeepSeek | Latency, tokens, tool calls/latency, bytes |
| Web Search Agent (`web_search_agent`) | Research agent with web search capability | Latency, tokens, tool calls/latency, bytes |
| Web Search Agent (DeepSeek) (`web_search_agent_deepseek`) | Research agent with web search using DeepSeek | Latency, tokens, tool calls/latency, bytes |
| General Agent (`general_agent`) | General-purpose agent with full MCP access | Latency, tokens, tool calls/latency, bytes |
| Computer Control Agent (`computer_control_agent`) | Computer use agent via OpenAI computer tool | Latency, tool calls/latency, bytes (screenshots) |
| Image Generation (`image_generation`) | Image generation | Latency, bytes, throughput |
| Multimodal Analysis (`multimodal_analysis`) | Multimodal input analysis (image + text) | Latency, tokens, bytes |
| Direct Web Search (`direct_web_search`) | Multi-threaded web search without MCP | Latency, bytes, throughput |
| Direct Web Search (Google) (`direct_web_search_google`) | Web search using Google Custom Search API | Latency, bytes, throughput |
| Direct Web Search Burst (`direct_web_search_burst`) | High-parallelism burst search stress test | Latency, bytes, throughput, concurrency |
| Parallel Search Benchmark (`parallel_search_benchmark`) | Benchmark parallel search with varying threads | Latency, throughput, bytes, concurrency |
| Realtime Text (`realtime_text`) | Real-time conversational AI via WebSocket (text) | Latency, TTFT, streaming chunks, bytes |
| Realtime Text WebRTC (`realtime_text_webrtc`) | Real-time conversational AI via WebRTC (text) | Latency, TTFT, streaming chunks, bytes, SDP sizes |
| Realtime Interactive (`realtime_interactive`) | Interactive real-time conversation (text + audio) | Latency, TTFT, streaming chunks, bytes, audio bytes |
| Realtime Technical (`realtime_technical`) | Technical support real-time conversation | Latency, TTFT, streaming chunks, bytes, audio bytes |
| Realtime Multilingual (`realtime_multilingual`) | Multilingual real-time conversation | Latency, TTFT, streaming chunks, bytes, audio bytes |
| Realtime Audio (`realtime_audio`) | Audio-based real-time conversation (voice in/out) | Latency, TTFT, audio bytes, audio duration, streaming chunks |
| Realtime Audio WebRTC (`realtime_audio_webrtc`) | Audio-based real-time conversation over WebRTC | Latency, TTFT, audio bytes, audio duration, streaming chunks, SDP sizes |


### Network Profiles

| Profile | Bandwidth | Latency | Loss | Use Case |
|:--------|:----------|:--------|:-----|:---------|
| `ideal_6g` | Unlimited | 1 ms | 0% | Baseline measurements |
| `5g_urban` | 100 Mbps | 20 ms | 0.1% | Typical 5G deployment |
| `wifi_good` | 50 Mbps | 30 ms | 0.1% | Home/office WiFi |
| `cell_edge` | 5 Mbps | 120 ms | 1% | Poor cellular coverage |
| `edge_rural` | 10 Mbps | 80 ms | 1% | Rural edge deployment |
| `satellite` | 10 Mbps | 600 ms | 0.5% | LEO satellite access |
| `congested` | 1 Mbps | 200 ms | 3% | Network congestion |

Additional 5QI-derived profiles are available as `5qi_1`, `5qi_2`, ... `5qi_85`.
These map `delay_ms` to the 5QI packet delay budget and `loss_pct` to PER * 100.

#### Bidirectional Shaping

By default, network profiles are applied **bidirectionally** (both egress and ingress) using Linux IFB (Intermediate Functional Block) devices. This ensures realistic network conditions for both upload (requests) and download (responses).

```bash
# Default: same profile for egress and ingress
python orchestrator.py --scenario chat_basic --profile cell_edge

# Egress-only shaping (ingress unaffected)
python orchestrator.py --scenario chat_basic --profile cell_edge --egress-only

# Asymmetric: different profiles for egress vs ingress
python orchestrator.py --scenario chat_basic --profile 5g_urban --ingress-profile satellite

# Asymmetric: shaped egress, no ingress shaping
python orchestrator.py --scenario chat_basic --profile cell_edge --ingress-profile none
```

| Use Case | Egress | Ingress | Command |
|:---------|:-------|:--------|:--------|
| Symmetric bad network | cell_edge | cell_edge | `--profile cell_edge` |
| Good upload, bad download | ideal_6g | cell_edge | `--profile ideal_6g --ingress-profile cell_edge` |
| Test egress only | cell_edge | none | `--profile cell_edge --egress-only` |
| Satellite download, 5G upload | 5g_urban | satellite | `--profile 5g_urban --ingress-profile satellite` |

#### 5QI Profiles (Mapping Summary)

The following profiles are pre-defined in `configs/profiles.yaml` using the
5QI packet delay budget (PDB) and packet error rate (PER) values.

| Profile | Resource | PDB | PER | Example Services |
|:--------|:---------|:----|:----|:-----------------|
| `5qi_1` | GBR | 100 ms | 1e-2 | Conversational Voice |
| `5qi_2` | GBR | 150 ms | 1e-3 | Conversational Video (Live Streaming) |
| `5qi_3` | GBR | 50 ms | 1e-3 | Real Time Gaming, V2X messages |
| `5qi_4` | GBR | 300 ms | 1e-6 | Non-Conversational Video (Buffered Streaming) |
| `5qi_5` | Non-GBR | 100 ms | 1e-6 | IMS Signalling |
| `5qi_6` | Non-GBR | 300 ms | 1e-6 | Buffered Video (TCP-based) |
| `5qi_7` | Non-GBR | 100 ms | 1e-3 | Voice, Live Streaming, Interactive Gaming |
| `5qi_8` | Non-GBR | 300 ms | 1e-6 | Buffered Video (TCP-based) |
| `5qi_9` | Non-GBR | 300 ms | 1e-6 | Buffered Video (TCP-based) |
| `5qi_65` | GBR | 75 ms | 1e-2 | Mission Critical PTT voice |
| `5qi_66` | GBR | 100 ms | 1e-2 | Non-Mission-Critical PTT voice |
| `5qi_67` | GBR | 100 ms | 1e-3 | Mission Critical Video |
| `5qi_69` | Non-GBR | 60 ms | 1e-6 | Mission Critical signalling |
| `5qi_70` | Non-GBR | 200 ms | 1e-6 | Mission Critical data |
| `5qi_75` | GBR | 50 ms | 1e-2 | V2X messages |
| `5qi_79` | Non-GBR | 50 ms | 1e-2 | V2X messages |
| `5qi_80` | Non-GBR | 10 ms | 1e-6 | Low-latency eMBB, AR |
| `5qi_82` | Delay Critical GBR | 10 ms | 1e-4 | Discrete Automation |
| `5qi_83` | Delay Critical GBR | 10 ms | 1e-4 | Discrete Automation |
| `5qi_84` | Delay Critical GBR | 30 ms | 1e-5 | Intelligent Transport Systems |
| `5qi_85` | Delay Critical GBR | 5 ms | 1e-5 | Electricity Distribution (HV) |

#### Advanced Netem Controls

Advanced netem controls are supported in `configs/profiles.yaml`. Fields are
passed directly to `tc netem`/HTB where applicable:

- `delay_distribution`, `delay_correlation_pct`
- `loss_correlation_pct`, `loss_model`
- `corruption_correlation_pct`
- `reorder_correlation_pct`
- `duplicate_pct`, `duplicate_correlation_pct`
- `rate_ceil_mbit`, `rate_burst_kbit`, `rate_cburst_kbit`
- `limit_packets`

Example profile with distribution and correlation:

```yaml
profiles:
  custom_profile:
    description: "Delay distribution + correlated loss"
    delay_ms: 40
    jitter_ms: 10
    delay_distribution: "normal"
    delay_correlation_pct: 25
    loss_pct: 0.2
    loss_correlation_pct: 10
    loss_model: "gemodel 0.1 0.2 0.3 0.4"
    rate_mbit: 20
    rate_ceil_mbit: 40
    rate_burst_kbit: 64
    rate_cburst_kbit: 64
    limit_packets: 1000
```

### Metrics

The testbed captures metrics aligned with TR 22.870:

- **QoE Metrics**
  - Time-to-First-Token (TTFT)
  - Time-to-Last-Token (TTLT)
  - TTFT/TTLT tail percentiles (P50/P95/P99)
  - Session completion rate
  - Perceived responsiveness

- **Traffic Characteristics**
  - Uplink/Downlink byte volumes
  - UL/DL ratio
  - Packet burst patterns
  - Token streaming rate
  - Burstiness descriptors (peak-to-mean, coefficient of variation, ON/OFF gaps)
  - Streaming stall rate and stall duration

- **AI Service Metrics**
  - Agent loop factor (API calls per user prompt)
  - Tool call latency
  - Multi-step task completion time
  - Error taxonomy (timeout vs 429 vs 5xx vs tool failure)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Scenario Orchestrator                        │
│                      (orchestrator.py)                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Chat         │ │  Agent        │ │  Image        │
│  Scenario     │ │  Scenarios    │ │  Scenario     │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        │                 │ MCP Protocol    │
        │                 ▼                 │
        │    ┌────────────────────────┐    │
        │    │    MCP Tool Servers    │    │
        │    │ ┌────────┬───────────┐ │    │
        │    │ │ Cloud  │   Fetch   │ │    │
        │    │ │ Search │  Server   │ │    │
        │    │ └────────┴───────────┘ │    │
        │    │ ┌────────┬───────────┐ │    │
        │    │ │  File  │  Memory   │ │    │
        │    │ │ System │  Server   │ │    │
        │    │ └────────┴───────────┘ │    │
        │    └───────────┬────────────┘    │
        │                │                 │
        └────────────────┼─────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Client Layer                           │
│     ┌────────────────┬────────────────┬────────────────┐──────┐ │
│     │ OpenAIClient   │ GeminiClient   │ DeepSeekClient │ vLLM | │
│     └────────────────┴────────────────┴────────────────┘──────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTPS
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              L7 Capture (mitmproxy) [Optional]                   │
│     Full HTTP/HTTPS interception │ Headers │ Bodies │ Timing     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Network Emulator (tc/netem)                    │
│        Bandwidth limiting │ Latency │ Jitter │ Packet loss       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │  Internet │ ──► LLM APIs + Tool Server APIs
                    └───────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Capture & Analysis                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   L3/L4      │  │     L7       │  │ TrafficLogger│           │
│  │  (tcpdump)   │  │ (mitmproxy)  │  │   (SQLite)   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐  ┌──────────────┐                              │
│  │   Metrics    │  │  Visualizer  │                              │
│  │  Calculator  │  │   (Plots)    │                              │
│  └──────────────┘  └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (for MCP servers)
- Linux with `tc` and `netem` (for network emulation)
- `sudo` access (for network emulation) - see [Sudoers Configuration](#sudoers-configuration-for-network-emulation) for passwordless setup

### Setup

```bash
# Clone or navigate to the testbed directory
cd aitestbed

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install MCP servers (for agent scenarios)
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-fetch
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-memory
```

### API Keys

Set your API keys as environment variables:

```bash
# Required for OpenAI scenarios
export OPENAI_API_KEY="sk-..."

# Required for Gemini scenarios
export GOOGLE_API_KEY="..."

# Required for DeepSeek scenarios
export DEEPSEEK_API_KEY="..."

# Required for agent scenarios (Brave Search MCP server)
# Get your API key at: https://brave.com/search/api/
export BRAVE_API_KEY="..."
```

### Models and Pricing

Each scenario uses a specific model. The table below shows the mapping and estimated costs.

#### Models by Provider

**OpenAI**

| Scenario | Model | Type | Pricing (per 1M tokens) |
|:---------|:------|:-----|:------------------------|
| chat_basic | gpt-5-mini | chat | $0.25 in / $2.00 out |
| chat_streaming | gpt-5-mini | chat | $0.25 in / $2.00 out |
| direct_web_search | gpt-5-mini | search | $0.25 in / $2.00 out |
| shopping_agent | gpt-5-mini | agent | $0.25 in / $2.00 out |
| web_search_agent | gpt-5-mini | agent | $0.25 in / $2.00 out |
| general_agent | gpt-5.2 | agent | $1.75 in / $14.00 out |
| computer_control_agent | gpt-4o-mini | computer_use | $0.25 in / $2.00 out |
| image_generation | gpt-image-1.5 | image | ~$0.04/image (medium) |
| realtime_text | gpt-realtime-mini | realtime (text) | $0.60 in / $2.40 out |
| realtime_text_webrtc | gpt-realtime-mini | realtime (text) | $0.60 in / $2.40 out |
| realtime_interactive | gpt-realtime-mini | realtime (text) | $0.60 in / $2.40 out |
| realtime_technical | gpt-realtime-mini | realtime (text) | $0.60 in / $2.40 out |
| realtime_audio | gpt-realtime-mini | realtime (audio) | $10.00 in / $20.00 out |

**DeepSeek** 

| Scenario | Model | Type | Pricing (per 1M tokens) |
|:---------|:------|:-----|:------------------------|
| chat_deepseek | deepseek-chat | chat | $0.14 in / $0.28 out |
| chat_deepseek_streaming | deepseek-chat | chat | $0.14 in / $0.28 out |
| chat_deepseek_coder | deepseek-coder | chat | $0.14 in / $0.28 out |
| chat_deepseek_reasoner | deepseek-reasoner | chat | $0.55 in / $2.19 out |

**Gemini** 

| Scenario | Model | Type | Pricing (per 1M tokens) |
|:---------|:------|:-----|:------------------------|
| chat_gemini | gemini-3-flash-preview | chat | Free tier / $0.075 in |
| multimodal_analysis | gemini-3-flash-preview | multimodal | Free tier / $0.075 in |

#### Cost Estimates by Test Mode

The `run_full_tests.sh` script supports different modes with varying costs:

| Mode | Regular Scenarios | Realtime Scenarios | Estimated Total Cost |
|:-----|:------------------|:-------------------|:---------------------|
| `--quick` | 2 runs each | 2 runs each | **~$1.50** |
| (default) | 5 runs each | 2 runs each | **~$2.50** |
| `--full` | 10 runs each | 2 runs each | **~$4.00** |


> **Note:** Realtime scenarios with Audio mode (`realtime_audio`) is more expensive ($10-20/1M tokens vs $0.60-2.40/1M for text).

### MCP Server Configuration

Agent scenarios use real MCP (Model Context Protocol) servers for tool execution. The configuration is in `configs/mcp_servers.yaml`:

```yaml
mcpServers:
  brave-search:      # Web search via Brave API
  fetch:             # URL content fetching
  filesystem:        # Sandboxed file operations
  memory:            # Persistent key-value storage

serverGroups:
  shopping:          # brave-search, fetch
  web_research:      # brave-search, fetch, memory
  general:           # All servers
```

The MCP client automatically starts/stops servers as needed during scenario execution.

## Usage

### Command Line Interface

```bash
# List available scenarios
python orchestrator.py --list-scenarios

# List available network profiles
python orchestrator.py --list-profiles

# Run a single scenario with a specific profile
python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10

# Run a scenario with streaming enabled
python orchestrator.py --scenario chat_streaming --profile cell_edge --runs 5

# Run the shopping agent scenario
python orchestrator.py --scenario shopping_agent --profile ideal_6g --runs 10

# Run a DeepSeek chat scenario
python orchestrator.py --scenario chat_deepseek --profile 5g_urban --runs 10

# Run the complete test matrix
python orchestrator.py --scenario all --runs 10

# Specify custom configuration paths
python orchestrator.py \
    --scenario chat_basic \
    --profile 5g_urban \
    --config configs/scenarios.yaml \
    --profiles configs/profiles.yaml \
    --db logs/traffic_logs.db \
    --report reports/my_experiment.json
```

### Programmatic Usage

```python
from orchestrator import TestbedOrchestrator

# Initialize orchestrator
orchestrator = TestbedOrchestrator(
    config_path="configs/scenarios.yaml",
    profiles_path="configs/profiles.yaml",
    db_path="logs/traffic_logs.db"
)

# Run a single experiment
results = orchestrator.run_experiment(
    scenario_name="chat_streaming",
    profile_name="5g_urban",
    runs=10
)

# Access results
for result in results:
    print(f"Session: {result.session_id}")
    print(f"  Latency: {result.total_latency_sec:.2f}s")
    print(f"  Success: {result.success}")
    print(f"  API calls: {result.api_call_count}")

# Run full test matrix
matrix_results = orchestrator.run_test_matrix(runs_per_experiment=10)

# Generate report
orchestrator.generate_report(
    matrix_results["metrics"],
    "reports/experiment_report.json"
)
```

### Using Individual Components

```python
# Using the OpenAI client directly
from clients import OpenAIClient, ChatMessage, MessageRole

client = OpenAIClient()

# Non-streaming chat
messages = [ChatMessage(role=MessageRole.USER, content="Hello!")]
response = client.chat(messages, model="gpt-5-mini", stream=False)
print(f"Response: {response.content}")
print(f"Latency: {response.latency_sec:.2f}s")
print(f"Tokens: {response.tokens_in} in, {response.tokens_out} out")

# Streaming with timing metrics
streaming_response = client.chat_streaming(messages, model="gpt-5-mini")
print(f"TTFT: {streaming_response.ttft:.3f}s")
print(f"TTLT: {streaming_response.ttlt:.3f}s")
print(f"Chunks: {len(streaming_response.chunks)}")

# Image generation
image_response = client.generate_image(
    prompt="A futuristic 6G network visualization",
    model="gpt-image-1.5",
    size="1024x1024"
)
print(f"Image size: {len(image_response.image_data)} bytes")
print(f"Generation time: {image_response.latency_sec:.2f}s")
```

```python
# Using the network emulator
from netem import NetworkEmulator

# Default: bidirectional shaping (egress + ingress)
emulator = NetworkEmulator(
    interface="eth0",
    profiles_path="configs/profiles.yaml"
)

# Apply same profile to both directions (default)
emulator.apply_profile("cell_edge")

# Apply different profiles for egress vs ingress
emulator.apply_profile("5g_urban", ingress_profile="satellite")

# Apply egress only (no ingress shaping)
emulator.apply_profile("cell_edge", ingress_profile="none")

# Check status
status = emulator.get_status()
print(f"Egress active: {status['egress_active']}")
print(f"Ingress active: {status.get('ingress_active', False)}")
print(f"Profile: {status['current_profile']}")

# Clear when done
emulator.clear()

# Egress-only mode (disable IFB entirely)
emulator_egress_only = NetworkEmulator(
    interface="eth0",
    profiles_path="configs/profiles.yaml",
    bidirectional=False  # Only shape egress
)
```

```python
# Using the traffic logger and metrics
from analysis import TrafficLogger, MetricsCalculator

logger = TrafficLogger("logs/traffic_logs.db")

# Query logs
records = logger.query(scenario_id="chat_streaming", network_profile="5g_urban")

# Calculate metrics
metrics = MetricsCalculator.calculate(records, "chat_streaming", "5g_urban")
print(f"Latency mean: {metrics.latency_mean * 1000:.0f}ms")
print(f"Latency P95: {metrics.latency_p95 * 1000:.0f}ms")
print(f"TTFT mean: {metrics.ttft_mean * 1000:.0f}ms")
print(f"Success rate: {metrics.success_rate:.1f}%")

# Export to 3GPP format
report = MetricsCalculator.to_3gpp_format(metrics)
print(report)
```

### L7 (HTTP/HTTPS) Capture

The testbed includes L7 capture capability using **mitmproxy** to capture full HTTP/HTTPS request/response details including headers, bodies, and precise timing.

#### Setup

```bash
# Install mitmproxy (included in requirements.txt)
pip install mitmproxy

# Generate CA certificate (run once)
mitmdump
# Press Ctrl+C after startup

# Trust the CA certificate (Linux)
sudo cp ~/.mitmproxy/mitmproxy-ca-cert.pem /usr/local/share/ca-certificates/mitmproxy.crt
sudo update-ca-certificates

# Or for Python only
export SSL_CERT_FILE=~/.mitmproxy/mitmproxy-ca-cert.pem
export REQUESTS_CA_BUNDLE=~/.mitmproxy/mitmproxy-ca-cert.pem
```

#### Usage

```python
from capture import L7CaptureController, configure_client_proxy, clear_client_proxy
from clients import OpenAIClient, ChatMessage, MessageRole

# Start L7 capture
l7_capture = L7CaptureController(
    capture_dir="capture/l7_captures",
    proxy_port=8080,
    web_port=8081  # Web UI at http://localhost:8081
)

capture_file = l7_capture.start(
    filename="my_capture.jsonl",
    filter_hosts=["api.openai.com"]  # Only capture OpenAI traffic
)

# Configure client to use proxy
configure_client_proxy("http://localhost:8080")

# Make API calls - all traffic is captured with full details
client = OpenAIClient()
response = client.chat(
    [ChatMessage(role=MessageRole.USER, content="Hello!")],
    model="gpt-5-mini"
)

# Stop capture and clean up
clear_client_proxy()
capture_file = l7_capture.stop()

# Read captured records
records = l7_capture.read_records(capture_file)
for r in records:
    print(f"{r.request_method} {r.request_url}")
    print(f"  Request: {r.request_body_size} bytes")
    print(f"  Response: {r.response_body_size} bytes (HTTP {r.response_status})")
    print(f"  Latency: {r.total_time:.3f}s")
    print(f"  TLS: {r.tls_version}")

# Get summary statistics
summary = l7_capture.get_summary(capture_file)
print(f"Total requests: {summary['count']}")
print(f"Total UL: {summary['total_request_bytes']} bytes")
print(f"Total DL: {summary['total_response_bytes']} bytes")
```

#### L7 Record Fields

| Field | Description |
|:------|:------------|
| `request_method` | HTTP method (GET, POST, etc.) |
| `request_url` | Full request URL |
| `request_host` | Target hostname |
| `request_headers` | All request headers (dict) |
| `request_body_size` | Request payload size in bytes |
| `response_status` | HTTP status code |
| `response_headers` | All response headers (dict) |
| `response_body_size` | Response payload size in bytes |
| `total_time` | End-to-end request time (seconds) |
| `tls_version` | TLS version used (e.g., "TLSv1.3") |
| `tls_cipher` | TLS cipher suite |

## Configuration

### Scenario Configuration (`configs/scenarios.yaml`)

```yaml
scenarios:
  my_custom_scenario:
    type: "chat"                    # chat, agent, image, multimodal
    description: "Custom chat test"
    provider: "openai"              # openai, gemini
    model: "gpt-5-mini"
    stream: true                    # Enable streaming
    max_turns: 5                    # For multi-turn conversations
    prompts:
      - "First question..."
      - "Follow-up question..."

  my_agent_scenario:
    type: "agent"
    provider: "openai"
    model: "gpt-5-mini"
    tools:
      - "check_price"
      - "search_products"
    max_tool_calls: 5
    prompts:
      - "Find the best laptop under $1000"

# Test matrix definition
test_matrix:
  - scenario: "my_custom_scenario"
    profiles: ["ideal_6g", "5g_urban", "cell_edge"]
    runs: 10
    priority: "high"
```

Defaults in `configs/scenarios.yaml` also control retry behavior and metrics thresholds:

```yaml
defaults:
  retry_count: 3
  retry_backoff_sec: 1.0
  retry_backoff_multiplier: 2.0
  stall_gap_sec: 1.0        # Streaming stall gap threshold
  burst_gap_sec: 1.0        # Burst ON/OFF gap threshold
```

Retries apply to transient failures (timeouts, HTTP 429, and 5xx excluding 501) on non-tool calls.

### Network Profile Configuration (`configs/profiles.yaml`)

```yaml
profiles:
  my_custom_profile:
    description: "Custom network conditions"
    delay_ms: 50        # One-way delay
    jitter_ms: 10       # Delay variation
    loss_pct: 0.5       # Packet loss percentage
    rate_mbit: 20       # Bandwidth limit (null for unlimited)
    corruption_pct: 0   # Packet corruption
    reorder_pct: 0      # Packet reordering

default_interface: "eth0"
```

## Output

### Log Database (`logs/traffic_logs.db`)

SQLite database with the following schema:

| Column | Type | Description |
|:-------|:-----|:------------|
| `timestamp` | REAL | Unix timestamp |
| `scenario_id` | TEXT | Scenario identifier |
| `session_id` | TEXT | Unique session ID |
| `turn_index` | INT | Turn number in conversation |
| `provider` | TEXT | LLM provider (openai/gemini) |
| `model` | TEXT | Model identifier |
| `request_bytes` | INT | Request payload size |
| `response_bytes` | INT | Response payload size |
| `tokens_in` | INT | Input token count |
| `tokens_out` | INT | Output token count |
| `t_request_start` | REAL | Request start timestamp |
| `t_first_token` | REAL | First token timestamp |
| `t_last_token` | REAL | Last token timestamp |
| `latency_sec` | REAL | Total latency |
| `network_profile` | TEXT | Network profile used |
| `success` | INT | Success flag (0/1) |
| `tool_calls_count` | INT | Number of tool calls |
| `is_streaming` | INT | Streaming flag (0/1) |

### Reports

The testbed generates:

- **JSON Summary** (`reports/experiment_report.json`)
  - Raw metrics in 3GPP-compatible format
  - Per-scenario, per-profile breakdown

- **Markdown Tables** (`reports/experiment_tables.md`)
  - QoE metrics summary
  - Traffic characteristics
  - Agent/tool metrics

- **Visualization Plots** (`reports/figures/`)
  - Latency comparison bar charts
  - Latency CDF plots
  - UL/DL ratio comparisons
  - Success rate charts
  - Agent loop factor visualization

## Direct Web Search (No MCP)

The testbed includes an alternative web search implementation that uses **direct HTTP requests** instead of MCP servers. This is useful for:

- Testing without MCP server dependencies
- Measuring pure HTTP traffic patterns
- Benchmarking parallel request throughput
- Comparing MCP overhead vs direct HTTP

### Supported Search Engines

| Engine | API Required | Rate Limits | Notes |
|:-------|:-------------|:------------|:------|
| **DuckDuckGo** | No | Informal | Uses HTML scraping, no API key needed |
| **Google** | Yes | 100/day (free) | Requires Custom Search API key + CX |

### Configuration

```yaml
# configs/scenarios.yaml
direct_web_search:
  type: "direct_search"
  search_engine: "duckduckgo"  # google or duckduckgo
  thread_count: 5              # Parallel search threads
  search_timeout: 30.0         # Per-search timeout (seconds)
  max_results: 10              # Results per search
  synthesize_with_llm: true    # Use LLM to summarize results
  queries:
    - "6G wireless technology"
    - "AI traffic patterns"
```

### Environment Variables

```bash
# For Google Custom Search
export GOOGLE_SEARCH_API_KEY="your-api-key"
export GOOGLE_SEARCH_CX="your-search-engine-id"
```

### Usage

```bash
# Run direct search with DuckDuckGo (no API key needed)
python orchestrator.py --scenario direct_web_search --profile 5g_urban --runs 10

# Run burst search (20 parallel threads, no LLM synthesis)
python orchestrator.py --scenario direct_web_search_burst --profile ideal_6g --runs 5

# Run parallelism benchmark
python orchestrator.py --scenario parallel_search_benchmark --profile ideal_6g --runs 3
```

### Programmatic Usage

```python
from scenarios import (
    DirectSearchClient,
    ThreadedSearchExecutor,
    SearchEngine,
)

# Single search
client = DirectSearchClient(engine=SearchEngine.DUCKDUCKGO)
result = client.search("6G wireless technology")
print(f"Found {len(result.results)} results in {result.latency_sec:.2f}s")
print(f"Traffic: {result.request_bytes} UL, {result.response_bytes} DL")

# Parallel searches
executor = ThreadedSearchExecutor(
    engine=SearchEngine.DUCKDUCKGO,
    max_workers=10
)
queries = ["AI", "ML", "deep learning", "neural networks"]
results = executor.search_parallel(queries)
print(f"Wall clock: {results.wall_clock_time_sec:.2f}s")
print(f"Sum latency: {results.total_latency_sec:.2f}s")
print(f"Parallelism factor: {results.total_latency_sec / results.wall_clock_time_sec:.1f}x")
```

## Extending the Testbed

### Adding a New Scenario

1. Create a new file in `scenarios/`:

```python
# scenarios/my_scenario.py
from .base import BaseScenario, ScenarioResult

class MyScenario(BaseScenario):
    @property
    def scenario_type(self) -> str:
        return "my_scenario"

    def run(self, network_profile: str, run_index: int = 0) -> ScenarioResult:
        # Implement your scenario logic
        pass
```

2. Register in `scenarios/__init__.py`
3. Add configuration in `configs/scenarios.yaml`

### Adding a New LLM Provider

1. Create a new file in `clients/`:

```python
# clients/my_provider_client.py
from .base import LLMClient

class MyProviderClient(LLMClient):
    @property
    def provider(self) -> str:
        return "my_provider"

    def chat(self, messages, model, stream=False, **kwargs):
        # Implement chat completion
        pass

    def chat_streaming(self, messages, model, **kwargs):
        # Implement streaming
        pass
```

2. Register in `clients/__init__.py`
3. Add to orchestrator's client factory

## Docker Deployment

The testbed can be run as a Docker container for consistent, reproducible experiments.

### Quick Start

```bash
# Build the image
make build

# Set up environment
cp .env.example .env
# Edit .env and add your API keys

# Run a test scenario
make test

# Or run interactively
make shell
```

### Docker Commands

```bash
# Build the Docker image
docker build -t 6g-ai-testbed:latest .

# Run with environment variables
docker run --rm \
    -e OPENAI_API_KEY="sk-..." \
    -e BRAVE_API_KEY="..." \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/reports:/app/reports \
    6g-ai-testbed:latest \
    python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10

# Run with network emulation (requires NET_ADMIN capability)
docker run --rm \
    --cap-add=NET_ADMIN \
    -e ENABLE_NETEM=true \
    -e OPENAI_API_KEY="sk-..." \
    -v $(pwd)/logs:/app/logs \
    6g-ai-testbed:latest \
    python orchestrator.py --scenario shopping_agent --profile cell_edge --runs 10

# Interactive shell
docker run -it --rm \
    -e OPENAI_API_KEY="sk-..." \
    6g-ai-testbed:latest --shell
```

### Docker Compose

```bash
# Start the testbed service
docker compose up -d testbed

# Run scenarios
docker compose run testbed python orchestrator.py --scenario all --runs 10

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

### Makefile Commands

```bash
make help           # Show all available commands
make build          # Build Docker image
make run            # Show help
make shell          # Interactive shell
make test           # Run quick test
make chat           # Run chat scenario
make agent          # Run shopping agent scenario
make all            # Run all scenarios
make compose-up     # Start with docker-compose
make compose-down   # Stop services
make compose-logs   # View service logs
make clean          # Remove containers
```

### Docker Compose Profiles

The compose file ships a single `testbed` service for core runs.

### Persistent Data

Mount these volumes to preserve data between runs:

```bash
-v $(pwd)/logs:/app/logs              # SQLite logs
-v $(pwd)/reports:/app/reports        # Generated reports
-v $(pwd)/capture/captures:/app/capture/captures    # PCAP files
-v $(pwd)/capture/l7_captures:/app/capture/l7_captures  # L7 logs
```

## References

- [3GPP TR 26.998](https://www.3gpp.org/): 6G Media Study
- [3GPP TR 22.870](https://www.3gpp.org/): Service requirements for 6G
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Linux Traffic Control (tc)](https://man7.org/linux/man-pages/man8/tc.8.html)
