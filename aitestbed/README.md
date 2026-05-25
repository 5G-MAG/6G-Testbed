# 6G AI Traffic Characterization Testbed

A testbed for measuring AI/LLM service traffic patterns under emulated network conditions, aligned with **3GPP SA4 6G Media Study** objectives.

## Overview

The testbed enables:

- **Measurement** of traffic characteristics across generative AI services (chat, image, video, realtime voice)
- **Analysis** of agentic AI patterns (multi-step tool calling via MCP, browser automation, market data)
- **Evaluation** of QoE metrics under emulated network conditions (latency, loss, bandwidth)
- **Reporting** in formats suitable for 3GPP standardization contributions

## Cross-Checking for SA4 AI Traffic Characterization

For 3GPP SA4 cross-checking (reproducing or validating contributed results), the canonical entry point is **`run_full_tests.sh`**. It drives the full test matrix in `configs/scenarios.yaml` across the 10 SA4 S4-260848 network profiles, captures L3/L4 PCAPs, and runs the complete post-processing pipeline (charts, Excel, `RESULTS.md`, `TRACES.md`, DB anonymization).

### Quick start

```bash
cd aitestbed
cp .env.example .env          # fill in at least OPENAI_API_KEY

# Smoke test — 3 runs/scenario, ~30 min depending on scenarios enabled
bash run_full_tests.sh --quick

# Cross-check run — 30 runs/scenario (default --full), hours to a day
bash run_full_tests.sh

# Narrow to a single phase to reproduce a specific contribution
bash run_full_tests.sh --enable chat --runs 30
bash run_full_tests.sh --enable realtime --runs 30
bash run_full_tests.sh --enable vllm --runs 30
```

### Parameters and when to use them

| Flag | Effect | When to use |
|:-----|:-------|:------------|
| `--quick` | 3 runs/scenario, short delays | Smoke test before a long run; verifying config/env |
| `--full` | 30 runs/scenario (default) | Publishable statistics — recommended for SA4 cross-check |
| `--runs N` | Exact number of runs | Match the run count used in the contribution being cross-checked |
| `--enable LIST` | Only run listed phases (comma-sep.) | Reproducing a specific scenario family |
| `--disable LIST` | Skip listed phases | Skip what you cannot run (no API key, no GPU, etc.) |
| `--stress` | Enable burst/parallel stress phase | Only for stress-testing contributions (opt-in) |
| `--no-capture` | Disable L3/L4 PCAP | Debugging only — **leave ON for cross-check runs** |
| `--no-anonymize` | Keep real provider/model names | Internal triage; SA4 submissions use anonymized DB |
| `--no-clean` | Keep prior DB and pcaps | Accumulate across runs (off by default) |
| **`--resume`** | **Skip already-completed combos, append to existing DB** | **Critical: use whenever a previous run was interrupted or a scenario failed. Implies `--no-clean`. See dedicated section below.** |
| `--verbose, -v` | Show full log instead of progress bar | Debugging; default is a single progress bar |

Phase names (for `--enable` / `--disable`): `chat, realtime, image, search, deepseek, gemini, music, trading, computer_use, playwright, multimodal, google_search, stress, vllm, realtime_video_understanding, chat_token`.

### `--resume`: recovering an interrupted run

SA4 cross-check runs can easily take 8–24 hours. **Assume at least one will be interrupted** — a quota error, a SIGKILL, a laptop lid, an OS reboot, or a single flaky scenario can derail a long matrix. `--resume` is how you pick up where you left off without losing the hours of data already captured.

**What it does.** Re-run with the same parameters and add `--resume`:

```bash
bash run_full_tests.sh --resume                   # keep original defaults
bash run_full_tests.sh --resume --runs 30         # match original run count
bash run_full_tests.sh --resume --enable vllm     # resume just one phase
```

On startup the script queries `logs/traffic_logs.db` once and drops every `(scenario, profile)` combo that already has `≥ RUNS_PER_SCENARIO` completed sessions from the test matrix. Remaining combos get their missing runs filled in; new runs append to the DB under fresh `session_id`s — no existing records are rewritten.

**Implies `--no-clean`.** `--resume` does **not** archive or wipe `logs/traffic_logs.db`, `results/captures/`, or `results/reports/`. That is the whole point — pass it **exactly when** you want to preserve prior data.

**What counts as "completed"** (orchestrator.py `get_completed_runs`):

| Session kind | Resume treats it as | Reason |
|:-------------|:--------------------|:-------|
| All records `success=1` for that session | ✓ completed — **skipped** | Data point is valid |
| `session_id LIKE 'timeout_%'` | ✓ completed — **skipped** | Timeout under harsh profiles is a legitimate measurement, not an error to retry |
| Any record `success=0` in the session | ✗ not completed — **retried** | Real failure, e.g. API error / crash — must produce a clean run |
| `session_id LIKE 'pcap_%'` | ignored (not a run) | Capture-only placeholder, excluded from counting |

Because successful sessions and timeout placeholders both count, the target run count eventually fills even under lossy profiles like `cell_edge` or `satellite_geo`.

**When to use `--resume`:**

- The run was interrupted for any reason (`Ctrl-C`, SIGKILL, OOM, reboot, power loss).
- `STOP_ON_ERROR=true` (default) tripped on a single failing scenario — fix the cause (quota, network, API key) and resume. The message `Test suite stopped due to failure. Re-run with --resume to continue.` is the prompt to do this.
- You ran `--quick` first to sanity-check and now want to top up to `--full` (pass `--resume --full` — existing 3 runs count against the 30-run target).
- A long phase (e.g. `vllm`, `realtime`) failed due to transient infra; fix the infra and `bash run_full_tests.sh --resume --enable <phase>`.

**When NOT to use `--resume`:**

- You changed `configs/profiles.yaml` or a scenario's model/provider/prompt. The prior runs are no longer comparable — start fresh (drop `--resume`, let `CLEAN_START=true` archive the old DB).
- You are starting a brand-new cross-check. First run should not have `--resume` — it wipes-and-archives correctly on its own.
- You want a deterministic, single-batch dataset where every record was produced by one invocation. `--resume` stitches across invocations; this is usually fine for SA4 cross-check but explicit to call out.

**Safety properties.** Every `--resume` invocation (a) archives nothing, (b) mutates no existing rows, (c) only appends new sessions, (d) re-runs post-processing (`RESULTS.md`, charts, Excel) over the full DB at the end, so the final artifacts reflect all accumulated data — not just the resumed slice. Running `--resume` against a fully-complete DB is a no-op for data collection; it will just regenerate the reports.

**Tip:** pair long runs with `nohup` or `tmux` so an SSH drop does not kill the script. Even so, leave `--resume` in your pocket — you will need it.

Key environment variables (see `--help` for the full list):

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `RUNS_PER_SCENARIO` | 30 | Runs per scenario/profile combo |
| `RUN_TIMEOUT_SEC` | 600 | Per-run timeout; raise for slow reasoning models |
| `NETWORK_INTERFACE` | `auto` | Pin a specific egress interface instead of auto-detect |
| `MCP_TRANSPORT` | `http` | MCP over HTTP so tool traffic is shaped by tc/netem. Set `stdio` only to bypass shaping for MCP |
| `CAPTURE_FILTER` | `port 443 or 80 or 8080 or 8000` | BPF filter; extend if a scenario uses other ports |
| `MANAGE_VLLM` | `true` | Auto-start/stop the vLLM container. Set `false` when you already run vLLM yourself |
| `VLLM_BACKEND` | `docker` | `docker` (recommended) or `host` (needs `vllm` on PATH) |
| `STOP_ON_ERROR` | `true` | Stop on first failed run (use `--resume` to continue) |

### Typical pitfalls

- **Passwordless `sudo` for `tc`/`tcpdump` is required.** Without it the script warns and skips network emulation, giving you `no_emulation`-only results. Fix:
  ```bash
  TCPDUMP_PATH=$(which tcpdump)
  echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/tc, $TCPDUMP_PATH, /usr/sbin/modprobe, /usr/sbin/ip" \
    | sudo tee /etc/sudoers.d/testbed && sudo chmod 440 /etc/sudoers.d/testbed
  ```
- **Stale netem qdiscs** from a crashed prior run cause bizarre first-scenario numbers. The script clears them on start and on `EXIT/INT/TERM`, but if you killed it with `SIGKILL`, run once with any profile to reset, or clear manually:
  ```bash
  sudo tc qdisc del dev <iface> root; sudo tc qdisc del dev <iface> ingress
  sudo tc qdisc del dev lo root;     sudo tc qdisc del dev lo ingress
  sudo tc qdisc del dev ifb0 root;   sudo ip link set dev ifb0 down
  ```
- **vLLM scenarios need a GPU** and ~30 GB VRAM for the default `Qwen3-VL-30B-A3B-Instruct`. If you do not have one, `--disable vllm`. If you already manage vLLM yourself, run with `MANAGE_VLLM=false`.
- **Docker vLLM**: the user must be in the `docker` group (`sudo usermod -aG docker $USER && newgrp docker`) and `nvidia-container-toolkit` installed. Verify with `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi` first.
- **Missing API keys silently skip phases.** The prereq check logs `-` lines for each missing key. Inspect the preamble before a long run — a greyed-out phase produces no data.
- **Playwright phase needs Chromium installed:** `pip install playwright && playwright install chromium`. Same for the `computer_use` phase.
- **Run timeouts** on DeepSeek Reasoner / long agent chains: raise `RUN_TIMEOUT_SEC=1200` (or higher) if you see `timeout_*` session IDs in the DB.
- **First PCAP on loopback is empty**: MCP-over-HTTP traffic capture needs `CAPTURE_LOOPBACK=true` (default). If you set `MCP_TRANSPORT=stdio`, tool traffic is not shaped and not captured — this is intentional but not what you want for SA4 cross-check.
- **API rate limits / quota errors** show up as red `ERROR` rows in the DB with non-zero `http_status`. Re-run with `--resume` after the quota resets; `--resume` skips combos that already have `RUNS_PER_SCENARIO` successful sessions.
- **Disk and time budget**: a full run (30 runs × full matrix × 10 profiles with PCAP) produces a few GB of pcaps and can run 8–24 h depending on enabled providers. Use `--enable <phase>` to cross-check one contribution at a time.

### What you get

After a successful run, the following artifacts are produced (paths relative to `aitestbed/`):

| Artifact | Path | Description |
|:---------|:-----|:------------|
| SQLite DB | `logs/traffic_logs.db` | Per-request records: bytes, tokens, TTFT, TTLT, success, http_status, scenario, profile |
| JSON report | `results/reports/experiment_report.json` | Aggregated metrics in 3GPP-compatible schema |
| Evaluation report | `RESULTS.md` | Full write-up with tables per scenario/profile |
| Traces | `TRACES.md` | Sample request/response traces per scenario |
| Charts | `results/reports/figures/` | Latency CDFs, TTFT/TTLT, throughput, heatmaps, pcap-derived plots |
| Excel workbook | `results/reports/chart_data.xlsx` | 15-sheet export of all metrics for spreadsheet review |
| PCAPs | `results/captures/` | L3/L4 tcpdump per scenario/profile (default filter: HTTPS/HTTP/8080/8000) |
| L7 logs | `results/l7_captures/` | mitmproxy HTTP frame logs (when enabled) |
| Backup | `logs/backups/<timestamp>/` | Snapshot of prior DB/PCAPs when running with default `--clean` |

The run ends with a summary banner showing duration, total records, success rate, and any scenario failures with per-scenario log paths for triage. For SA4 cross-checking, attach `RESULTS.md`, the figures directory, and (if requested) the anonymized `logs/traffic_logs.db`.

## Scenarios

### Chat

| Scenario | Provider | Model | Streaming | Description |
|:---------|:---------|:------|:----------|:------------|
| `chat_basic` | OpenAI | gpt-5-mini | No | Single-turn chat |
| `chat_streaming` | OpenAI | gpt-5-mini | Yes | Multi-turn streaming chat |
| `chat_gemini` | Gemini | gemini-3-flash-preview | Yes | Gemini chat |
| `chat_deepseek` | DeepSeek | deepseek-chat | No | DeepSeek chat |
| `chat_deepseek_streaming` | DeepSeek | deepseek-chat | Yes | DeepSeek streaming chat |
| `chat_deepseek_coder` | DeepSeek | deepseek-coder | Yes | Code-focused chat |
| `chat_deepseek_reasoner` | DeepSeek | deepseek-reasoner | Yes | Deep reasoning chat (R1) |
| `chat_vllm` | vLLM | Qwen3-VL-30B-A3B | Yes | Self-hosted model (loopback) |
| `chat_token` | OpenAI-Compatible | Custom | Yes/No | Chat with Token ID transmission |

### Agentic AI (MCP Tool Calling)

| Scenario | Provider | Model | MCP Server Group | Description |
|:---------|:---------|:------|:-----------------|:------------|
| `shopping_agent` | OpenAI | gpt-5-mini | shopping | Shopping assistant with product search |
| `shopping_agent_deepseek` | DeepSeek | deepseek-chat | shopping | Shopping assistant (DeepSeek) |
| `web_search_agent` | OpenAI | gpt-5-mini | web_research | Research agent with web search |
| `web_search_agent_deepseek` | DeepSeek | deepseek-chat | web_research | Research agent (DeepSeek) |
| `general_agent` | OpenAI | gpt-5.2 | general | General-purpose agent (all tools) |
| `trading_market_data` | OpenAI | gpt-5-mini | trading | Market data analysis (Alpaca) |
| `trading_options_scan` | OpenAI | gpt-5-mini | trading | Options market scan (Alpaca) |
| `music_search` | OpenAI | gpt-5-mini | music | Spotify music search |
| `music_playlist` | OpenAI | gpt-5-mini | music | Playlist composition |
| `music_research` | OpenAI | gpt-5-mini | music_research | Spotify + web search |
| `music_search_deepseek` | DeepSeek | deepseek-chat | music | Music search (DeepSeek) |

### Browser Automation

| Scenario | Provider | Model | Description |
|:---------|:---------|:------|:------------|
| `computer_control_agent` | OpenAI | computer-use-preview | OpenAI computer use tool |
| `playwright_web_test` | OpenAI | gpt-5-mini | Playwright MCP (with screenshots) |
| `playwright_web_test_text` | OpenAI | gpt-5-mini | Playwright MCP (text extraction only) |

### Image & Multimodal

| Scenario | Provider | Model | Description |
|:---------|:---------|:------|:------------|
| `image_generation` | OpenAI | gpt-image-1.5 | DALL-E image generation |
| `multimodal_analysis` | Gemini | gemini-3-flash-preview | Image + text analysis |
| `video_understanding_vllm` | vLLM | Qwen3-VL-30B-A3B | Video understanding (loopback) |
| `realtime_video_understanding` | VLM | Liquid_V1_7B | Realtime video analysis with token transmission |

### Realtime Conversational AI (OpenAI Realtime API)

| Scenario | Transport | Modalities | Voice | Description |
|:---------|:----------|:-----------|:------|:------------|
| `realtime_text` | WebSocket | text + audio | alloy | Text-mode realtime |
| `realtime_text_webrtc` | WebRTC | text | alloy | Text-mode via WebRTC |
| `realtime_interactive` | WebSocket | text + audio | shimmer | Voice assistant simulation |
| `realtime_technical` | WebSocket | text + audio | echo | Technical support conversation |
| `realtime_multilingual` | WebSocket | text + audio | coral | Multilingual conversation |
| `realtime_audio` | WebSocket | text + audio | sage | Voice in/out with TTS |
| `realtime_audio_webrtc` | WebRTC | text + audio | sage | Voice in/out via WebRTC |

### Direct Web Search (No MCP)

| Scenario | Engine | Threads | LLM Synthesis | Description |
|:---------|:-------|:--------|:--------------|:------------|
| `direct_web_search` | DuckDuckGo | 5 | Yes (OpenAI) | Multi-threaded search |
| `direct_web_search_deepseek` | DuckDuckGo | 5 | Yes (DeepSeek) | Multi-threaded search |
| `direct_web_search_google` | Google | 5 | Yes | Google Custom Search API |
| `direct_web_search_burst` | DuckDuckGo | 20 | No | Burst stress test |
| `parallel_search_benchmark` | DuckDuckGo | 1-20 | No | Parallelism benchmark |

## Network Profiles

The test matrix uses the 10 selected profiles from `configs/profiles.yaml`, aligned with 3GPP SA4 contribution **S4-260848 (Table C.Z-1)**:

| Profile | Delay | Jitter | Delay Dist. | Loss | Loss Distribution | Rate | Description |
|:--------|:------|:-------|:------------|:-----|:------------------|:-----|:------------|
| `no_emulation` | 0 ms | 0 ms | fixed | 0% | -- | -- | Reference (no tc/netem applied) |
| `6g_itu_hrllc` | 1 ms | 0.2 ms | normal | 0.001% | correlated (10%) | 300 Mbps | 6G HRLLC (ITU IMT-2030 / M.2160) |
| `5g_urban` | 20 ms | 5 ms | normal | 0.1% | correlated (25%) | 100 Mbps | Mainstream urban terrestrial cellular |
| `wifi_good` | 30 ms | 10 ms | normal | 0.1% | correlated (30%) | 50 Mbps | Non-3GPP local access (WiFi avg) |
| `cell_edge` | 120 ms | 30 ms | paretonormal | 1% | Gilbert-Elliot (35%) | 5 Mbps | Weak radio, heavy-tail jitter |
| `satellite_leo` ⇄ | 22 ms / 22 ms | 7 ms / 8 ms | normal | 0.5% / 0.8% | correlated (40% / 45%) | 100 / 15 Mbps | LEO satellite (asymmetric UL) |
| `satellite_geo` ⇄ | 340 ms / 340 ms | 15 ms / 18 ms | normal | 0.1% / 0.2% | correlated (20% / 25%) | 50 / 3 Mbps | GEO satellite (long RTT, asymmetric UL) |
| `congested` | 200 ms | 50 ms | pareto | 3% | Gilbert-Elliot (40%) | 1 Mbps | Bufferbloat / heavy congestion |
| `5qi_7` | 80 ms | 10 ms | normal | 0.1% | correlated (20%) | -- | 5QI 7: Voice / Live Streaming (PDB 100 ms, PER 1e-3) |
| `5qi_80` | 8 ms | 1 ms | normal | 1e-6 | correlated (5%) | -- | 5QI 80: Low-latency eMBB / AR (PDB 10 ms, PER 1e-6) |
| `lossy` | 0 ms | 0 ms | fixed | 10% | -- | unlimited | High loss test |

⇄ Asymmetric profiles (`satellite_leo`, `satellite_geo`) use an optional `uplink:` sub-block that overrides egress-side fields only. The columns above show **downlink / uplink**; fields not listed under `uplink:` are inherited from the downlink block. 5QI anchors follow the S4-260848 rule `delay_ms = PDB − 2.054 × jitter_ms` when `jitter_ms > 0`.

Profiles also carry advanced netem controls: `loss_correlation_pct`, `reorder_pct`, `reorder_correlation_pct`, `duplicate_pct`, and `limit_packets`. See `configs/profiles.yaml` for full definitions.

Bidirectional shaping is applied by default using IFB devices:

```bash
# Symmetric (default)
python orchestrator.py --scenario chat_basic --profile cell_edge

# Egress-only
python orchestrator.py --scenario chat_basic --profile cell_edge --egress-only

# Asymmetric (e.g. terrestrial DL with LEO UL)
python orchestrator.py --scenario chat_basic --profile 5g_urban --ingress-profile satellite_leo
```

## Test Matrix

The full test matrix runs each scenario against all 9 selected profiles. Defined in `configs/scenarios.yaml` under `test_matrix:`.

| Phase | Scenarios | Priority | Prerequisites |
|:------|:----------|:---------|:--------------|
| chat | `chat_basic`, `chat_streaming` | high | OPENAI_API_KEY |
| realtime | `realtime_text`, `realtime_interactive`, `realtime_technical`, `realtime_multilingual`, `realtime_audio`, `realtime_audio_webrtc`, `realtime_text_webrtc` | high/medium | OPENAI_API_KEY |
| image | `image_generation` | medium | OPENAI_API_KEY |
| search | `direct_web_search` | high | OPENAI_API_KEY |
| deepseek | `chat_deepseek`, `chat_deepseek_streaming`, `chat_deepseek_coder`, `chat_deepseek_reasoner`, `direct_web_search_deepseek` | high/medium | DEEPSEEK_API_KEY |
| gemini | `chat_gemini` | medium | GOOGLE_API_KEY |
| trading | `trading_market_data`, `trading_options_scan` | medium | ALPACA_API_KEY, ALPACA_SECRET_KEY |
| computer_use | `computer_control_agent` | medium | Playwright + Chromium |
| playwright | `playwright_web_test` | medium | Playwright + Chromium |
| multimodal | `multimodal_analysis` | medium | GOOGLE_API_KEY + image assets |
| vllm | `chat_vllm`, `video_understanding_vllm` | high | vLLM server on localhost:8000 |
| stress | `direct_web_search_burst`, `parallel_search_benchmark` | medium/low | Disabled by default |

```bash
# Run the full matrix
python orchestrator.py --scenario all --runs 10

# Run a single scenario across all profiles
python orchestrator.py --scenario chat_basic --profile all --runs 10

# Quick test
python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 5
```

## Metrics

See [METRICS.md](METRICS.md) for full documentation. Summary:

- **QoE**: TTFT, TTLT, tail percentiles (P50/P95/P99), session completion rate
- **Traffic**: UL/DL byte volumes, UL/DL ratio, token streaming rate, burstiness descriptors, stall rate
- **AI Service**: Agent loop factor, tool call latency, multi-step completion time, error taxonomy

## MCP Server Groups

Agent scenarios use MCP servers defined in `configs/mcp_servers.yaml`:

| Group | Servers | Used By |
|:------|:--------|:--------|
| `shopping` | brave-search, fetch | Shopping agent scenarios |
| `web_research` | brave-search, fetch, memory | Web search agent scenarios |
| `music` | spotify | Music search/playlist scenarios |
| `music_research` | spotify, brave-search, fetch | Music research scenarios |
| `trading` | alpaca | Trading/market data scenarios |
| `playwright` | playwright | Browser automation scenarios |
| `general` | All servers | General agent scenario |

All MCP servers support HTTP transport for traffic measurement via loopback (subject to tc/netem shaping).

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
│  Chat         │ │  Agent        │ │  Realtime     │
│  Scenarios    │ │  Scenarios    │ │  Scenarios    │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        │                 │ MCP Protocol    │ WebSocket/WebRTC
        │                 ▼                 │
        │    ┌────────────────────────┐     │
        │    │    MCP Tool Servers    │     │
        │    │  brave-search, fetch,  │     │
        │    │  filesystem, memory,   │     │
        │    │  spotify, playwright,  │     │
        │    │  alpaca                │     │
        │    └───────────┬────────────┘     │
        │                │                  │
        └────────────────┼──────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Client Layer                           │
│       ┌────────┬────────┬──────────┬──────┐                     │
│       │ OpenAI │ Gemini │ DeepSeek │ vLLM │                     │
│       └────────┴────────┴──────────┴──────┘                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTPS / WebSocket / WebRTC
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Network Emulator (tc/netem)                    │
│       Bandwidth │ Latency │ Jitter │ Loss │ Reorder │ Corrupt   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │  Internet │ → LLM APIs + Tool APIs
                    └───────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Capture & Analysis                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  L3/L4   │ │   L7     │ │  SQLite  │ │  Metrics │           │
│  │ tcpdump  │ │ mitmproxy│ │  Logger  │ │  + Plots │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (for MCP servers)
- Linux with `iproute2` (for network emulation)
- `tcpdump` (for PCAP capture)
- Sudo access or Docker with `NET_ADMIN`
- NVIDIA GPU with ~30GB VRAM (~2.5GB for VLM client, ~27GB for VLM server)
- CUDA Toolkit >= 12.1

### Setup

```bash
cd <repo-root>
python -m venv venv
source venv/bin/activate

# Install netemu (sibling package)
pip install -e netemu

# Install aiortc extension (for VLM/RTP scenarios)
pip install -e aiortc-main-clean

# Install testbed dependencies
pip install -r aitestbed/requirements.txt

# Install npm-based MCP servers
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-memory
```

### API Keys

```bash
export OPENAI_API_KEY="sk-..."          # OpenAI scenarios
export GOOGLE_API_KEY="..."             # Gemini scenarios
export DEEPSEEK_API_KEY="..."           # DeepSeek scenarios
export BRAVE_API_KEY="..."              # Agent scenarios (web search)
export SPOTIFY_CLIENT_ID="..."          # Music agent scenarios
export SPOTIFY_CLIENT_SECRET="..."      # Music agent scenarios
export ALPACA_API_KEY="..."             # Trading scenarios
export ALPACA_SECRET_KEY="..."          # Trading scenarios

# For Chat with Token ID scenario
export OPENAI_BASE_URL="http://..."     # OpenAI-compatible server URL
export MODEL_NAME="..."                 # Model name
export TOK_PATH="..." 			# Path to tokenizer files
```

Or copy `.env.example` to `.env` and fill in values.

### VLM & Token ID Scenario Assets

For **Realtime Video Analysis** and **Chat with Token ID** scenarios, additional assets are required.

**1. Realtime Video Analysis (`realtime_video_understanding`)**

- **VLM Tokenizer:** Download `vqgan.ckpt` and `vqgan.yaml`, put them in `aiortc-main-clean/src/aiortc/liquid/checkpoints/chameleon`
  - [vqgan.ckpt](https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.ckpt)
  - [vqgan.yaml](https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.yaml)
- **VLM Model:** Download all model files, put them in `aiortc-main-clean/src/aiortc/liquid/checkpoints/model`
  - [Liquid_V1_7B](https://huggingface.co/Junfeng5/Liquid_V1_7B/tree/main)
- **Dataset:** Download dataset, unzip to `aitestbed/examples/assets/dataset`
  - [StreamingBench Dataset](https://huggingface.co/datasets/mjuicem/StreamingBench/blob/main/Proactive%20Output_1-25.zip)

**2. Chat with Token ID (`chat_token`)**

- Set up an OpenAI-compatible server with an open-source LLM.
- Copy tokenizer related files to the path specified in `TOK_PATH`.
- Configure `.env` file with `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL_NAME`, and `TOK_PATH`.

### Sudoers (for Network Emulation)

```bash
TCPDUMP_PATH=$(which tcpdump)
echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/tc, $TCPDUMP_PATH, /usr/sbin/modprobe, /usr/sbin/ip" \
  | sudo tee /etc/sudoers.d/testbed
sudo chmod 440 /etc/sudoers.d/testbed
```

### vLLM (Self-Hosted Models)

Two ways to run the vLLM server. **Docker is recommended** — the
`vllm/vllm-openai` image bundles the right CUDA runtime and Python, so it
sidesteps `libcudart.so.12` and Python-version compatibility issues typical
of a host-pip install.

#### Docker (recommended)

Requires Docker, the NVIDIA driver, and `nvidia-container-toolkit`. Add
yourself to the `docker` group so you can manage containers without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker                     # or log out / log back in

# Sanity-check GPU passthrough before running scenarios:
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

Launch the vLLM server:

```bash
docker run -d --name vllm-testbed --gpus all --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 127.0.0.1:8000:8000 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --max-model-len 32768 --gpu-memory-utilization 0.95 \
    --trust-remote-code --tensor-parallel-size 1

# Stop / remove when done:
docker stop vllm-testbed && docker rm vllm-testbed
```

Then run the testbed with `MANAGE_VLLM=false` so the scripts probe the
already-running container instead of trying to spawn another:

```bash
MANAGE_VLLM=false ./run_full_tests.sh
MANAGE_VLLM=false ./test_vllm.sh
```

#### Host pip install (fallback — auto-managed by the scripts)

```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 --max-model-len 32768 \
    --gpu-memory-utilization 0.95 --trust-remote-code
```

`run_full_tests.sh` and `test_vllm.sh` default to `MANAGE_VLLM=true`, which
auto-starts and stops a host `vllm serve` process for you.

vLLM scenarios use `network_interface: lo` to shape loopback traffic.

## Usage

```bash
# List scenarios and profiles
python orchestrator.py --list-scenarios
python orchestrator.py --list-profiles

# Single scenario
python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10

# Full test matrix
python orchestrator.py --scenario all --runs 10

# With PCAP capture
python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10 --capture-pcap

# Custom paths
python orchestrator.py \
    --scenario chat_basic \
    --profile 5g_urban \
    --config configs/scenarios.yaml \
    --profiles configs/profiles.yaml \
    --db logs/traffic_logs.db \
    --report results/reports/experiment_report.json
```

### Running VLM & Token ID Scenarios

**Realtime Video Analysis**

```bash
cd aitestbed
# Start server first
python ./server/realtime_vlm_server.py

# Run test
python orchestrator.py --scenario realtime_video_understanding --profile 5g_urban --runs 10

# Run test with trace saved
PROFILE=lossy
python orchestrator.py \
 --scenario realtime_video_understanding \
 --profile $PROFILE \
 --runs 20 \
 --interface lo \
 --capture-pcap \
 --capture-dir results/captures/vlm/${PROFILE} \
 --report results/reports/experiment_report_realtime_video_understanding_${PROFILE}.json \
 --db logs/traffic_logs_vlm_${PROFILE}.db \
 --egress-only
```

**Chat with Token ID**

```bash
# Start up OpenAI-compatible server (locally or remotely)
# Edit .env file with OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME, TOK_PATH

# Run test
python orchestrator.py --scenario chat_token --profile 5g_urban --runs 10

# Run test with trace saved
PROFILE=lossy
python orchestrator.py \
 --scenario chat_token \
 --profile $PROFILE \
 --runs 10 \
 --interface eno1 \ # Replace 'eno1' with actual interface
 --capture-pcap \
 --capture-dir results/captures/${PROFILE} \
 --report results/reports/experiment_report_chat_token_${PROFILE}.json \
 --db logs/traffic_logs${PROFILE}.db
```

### Programmatic

```python
from orchestrator import TestbedOrchestrator

orchestrator = TestbedOrchestrator()
results = orchestrator.run_experiment("chat_streaming", "5g_urban", runs=10)

for r in results:
    print(f"Latency: {r.total_latency_sec:.2f}s  TTFT: {r.ttft_sec:.3f}s")
```

## Docker

```bash
# Build (from repo root)
docker build -t 6g-ai-testbed:latest -f aitestbed/Dockerfile .

# Run
docker run --rm --cap-add=NET_ADMIN \
    -e OPENAI_API_KEY="sk-..." \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/results/reports:/app/results/reports \
    6g-ai-testbed:latest \
    python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10

# Interactive shell
docker run -it --rm --cap-add=NET_ADMIN \
    -e OPENAI_API_KEY="sk-..." \
    6g-ai-testbed:latest --shell

# Docker Compose
docker compose up -d testbed
docker compose run testbed python orchestrator.py --scenario all --runs 10

# Makefile shortcuts
make build          # Build image
make test           # Quick test
make shell          # Interactive shell
make all            # Run all scenarios
```

### Persistent Data Volumes

```
logs:/app/logs                              # SQLite logs
results/reports:/app/results/reports        # JSON/Markdown reports
results/captures:/app/results/captures      # PCAP files
results/l7_captures:/app/results/l7_captures  # L7 HTTP logs
```

## Output

- **SQLite** (`logs/traffic_logs.db`) — per-request records with timestamps, bytes, tokens, TTFT/TTLT
- **JSON report** (`results/reports/experiment_report.json`) — aggregated metrics in 3GPP-compatible format
- **Markdown tables** (`results/reports/experiment_tables.md`) — QoE and traffic summaries
- **Plots** (`results/reports/figures/`) — latency CDFs, UL/DL ratios, success rates

## Extending

### New Scenario

1. Subclass `BaseScenario` in `scenarios/`
2. Register in `scenarios/__init__.py`
3. Add config in `configs/scenarios.yaml`

### New LLM Provider

1. Subclass `LLMClient` in `clients/`
2. Register in `clients/__init__.py`
3. Add to orchestrator's client factory

## References

- [3GPP TR 26.870](https://www.3gpp.org/): 6G Media Study
- [3GPP TR 22.870](https://www.3gpp.org/): Service requirements for 6G
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [Linux tc(8)](https://man7.org/linux/man-pages/man8/tc.8.html)
- [aiortc](https://github.com/aiortc/aiortc/tree/main)
- [Liquid Model](https://github.com/FoundationVision/Liquid)
- [Streaming Bench](https://github.com/THUNLP-MT/StreamingBench/tree/main)
