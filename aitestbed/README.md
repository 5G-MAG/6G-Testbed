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
| `encoder_placement` (+ `_qwen36`, `_concurrent` variants) | sglang-engine | Qwen3-VL-30B-A3B / Qwen3.6-35B-A3B | Visual encoder down-placement: server-side vs terminal-side ViT (precomputed embedding bundle), zero-training compression (see below) |

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

# Install WebRTC + AI codec plugin (for VLM/RTP scenarios)
# aiortc is a stock upstream dependency; aicodec registers the token codecs at runtime
pip install aiortc>=1.14.0,<1.16.0
pip install -e aicodec --config-settings editable_mode=compat

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

For **Realtime Video Analysis**, **Chat with Token ID**, and **Encoder Placement** scenarios, additional assets are required.

**1. Realtime Video Analysis (`realtime_video_understanding`)**

- **VLM Tokenizer:** Download `vqgan.ckpt` and `vqgan.yaml`, put them in `aicodec/aicodec/models/liquid/checkpoints/chameleon`
  - [vqgan.ckpt](https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.ckpt)
  - [vqgan.yaml](https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.yaml)
- **VLM Model:** Download all model files, put them in `aicodec/aicodec/models/liquid/checkpoints/model`
  - [Liquid_V1_7B](https://huggingface.co/Junfeng5/Liquid_V1_7B/tree/main)
- **Dataset:** the scenario supports two task classes, each needing its own dataset and its own `configs/scenarios.yaml` entry:

  | Task class | Dataset | Unzip to |
  |:-----------|:--------|:---------|
  | `proactive_output` | [Proactive Output_1-25.zip](https://huggingface.co/datasets/mjuicem/StreamingBench/blob/main/Proactive%20Output_1-25.zip) | `aitestbed/examples/assets/dataset` |
  | `real` | [Real-Time Visual Understanding_1-50.zip](https://huggingface.co/datasets/mjuicem/StreamingBench/resolve/main/Real-Time%20Visual%20Understanding_1-50.zip) | `aitestbed/examples/assets/dataset_real` |

  `proactive_output` scenario config:
  ```yaml
  realtime_video_understanding:
    type: "realtime_video_understanding"
    description: "Real-time visual understanding via VLM model"
    provider: "vlm-local"
    model: "liquid"
    target_fps: 2
    stream: true
    task_config:
      task_class: "proactive_output"
      po_thd: 3
      results_dir: "examples/assets/dataset/"
      video_paths: "examples/assets/dataset/"
    prompts:
      - "What is in the video? Answer directly!"

**Design notes for `realtime_video_understanding`:**

- **Frame selection.** Each inference call uses only the most recently received video frame, not an accumulated window of frames. This is deliberate: the scenario measures the traffic pattern of a continuous WebRTC uplink video stream against the resulting downlink inference-response pattern over time — it is not a benchmark of the model's multi-frame video-understanding accuracy. A scenario that reasons over multiple frames would need a different design, not a configuration change to this one.
- **Question timing.** The prompt questions are sent as part of the client's request structure and are not re-sent at timestamps aligned to video playback position. The scenario characterizes AI traffic patterns under a fixed question set, not a real-world interaction-timing model — it does not attempt to reproduce when a live user would actually ask each question relative to what's on screen.

Both choices affect how to read results from this scenario: it measures traffic shape and volume for a single-frame, fixed-question workload, not end-to-end video-understanding quality or realistic interaction pacing. A future scenario variant covering multi-frame reasoning or timestamp-aligned questions would be a separate addition, not a change to this one.

**2. Chat with Token ID (`chat_token`)**

- Set up an OpenAI-compatible server with an open-source LLM.
- Copy tokenizer related files to the path specified in `TOK_PATH`.
- Configure `.env` file with `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL_NAME`, and `TOK_PATH`.

**3. Encoder Placement (`encoder_placement*`) — visual encoder down-placement**

The encoder-placement family measures the traffic/QoE impact of moving the **vision encoder (ViT) of a VLM from the server into the terminal** for video understanding. Each family is a fully standalone set of files (scenario, client, runner, yaml config, sidecar server, bootstrap script) — zero changes to `orchestrator.py` or the shared configs, so several experiments can run and be debugged in parallel.

| Family | Model | Sidecar (serial) | Sidecar (concurrent) | Scenario type |
|:-------|:------|:-----------------|:---------------------|:--------------|
| Qwen3-VL | `Qwen3-VL-30B-A3B-Instruct` (4B for edge) | `sglang_engine_server.py`, port 8002 | `sglang_engine_server_concurrent.py`, port 8012 | `encoder_placement` / `encoder_placement_concurrent` |
| Qwen3.6 | `Qwen3.6-35B-A3B(-FP8)` (qwen3_5_moe) | `sglang_engine_server_qwen36.py`, port 8004 | `sglang_engine_server_concurrent_qwen36.py`, port 8014 | `encoder_placement_qwen36` / `encoder_placement_concurrent_qwen36` |

**Modes (both arms, A/B comparison — use a separate `--db` per mode):**

- `server` — the terminal uploads the original video; the sidecar's SGLang engine runs decode + processor + ViT + prefill + decode.
- `edge` — the terminal runs decode + processor + ViT locally and uploads a serialized `precomputed_embedding` bundle (MBs) instead of the raw mp4 (hundreds of MBs); the server consumes the precomputed latent features (`_maybe_dequant_bundle_feature` reconstructs bf16 from fp8 when quantized) and only performs prefill + decode.

Mode precedence: CLI `--mode` > env `ENC_MODE_QWEN36` (fallback `ENC_MODE`) > yaml `mode` key (default `server`). The Qwen3-VL arms use `ENC_MODE`.

**Zero-training compression stack (edge mode only):**

| Knob | Runner flag | Effect |
|:-----|:------------|:-------|
| VidCom2 token pruning | `--compressor vidcom2 --retention-ratio R` | Prune the bundle's video tokens to fraction R (0, 1] |
| fp8 feature quantization | `--bundle-quant fp8` | e4m3, per-tensor scale (`feature_scale` rides in the bundle); halves uploaded feature bytes; requires `--compressor vidcom2` |
| Ratio frame sampling | `--frame-ratio R` (sidecar too) | Keep `floor(total_frames × R)` frames (multiple of 2, min 4, capped by `--max-frames`) instead of sglang's default 2 fps × duration |
| Frame cap | `--max-frames N` | Native sglang whitelist key (default 768), no patch needed |

**Environment installation (epd env):**

- GPU: a single 80GB-class GPU for the sidecar (Qwen3.6 ~35 GB fp8 / ~70 GB bf16 weights; Qwen3-VL 30B bf16 similar); lower `--mem-fraction-static` when the edge tower shares the GPU. Edge terminals default to a `vision_only` load (~1 GB VRAM each on Qwen3.6; `--no-vision-only` forces the full model, ~35 GB per terminal).
- Python: the frozen `epd` conda env — `requirements-epd.txt` at the repo root (Python 3.11, torch 2.13.0, transformers 5.12.1, sglang 0.5.20). Install it as:

  ```bash
  conda create -n epd python=3.11 -y
  conda activate epd
  pip install -e aicodec --config-settings editable_mode=compat   # repo-local
  pip install -e netemu                                            # repo-local
  pip install --no-deps -r requirements-epd.txt
  # flashinfer JIT needs nvcc inside the env (align with torch.version.cuda):
  conda install -y -n epd -c nvidia cuda-nvcc cuda-cudart-dev
  ```

  The `--no-deps` flag is required: the snapshot already pins every transitive dependency and ships a known, harmless protobuf 5.x metadata conflict that pip's resolver would reject.
- Models: place weights under `aicodec/aicodec/models/` (e.g. `Qwen3.6-35B-A3B/`, `Qwen3-VL-30B-A3B-Instruct/`, `qwen3_vl_4B/`); the yaml default `model` paths point there. The VidCom2 `token_compressor` is vendored in-repo (`aicodec/aicodec/models/VidCom2/`) — no install needed.
- The bootstrap scripts (`start_sidecar*.sh` from the repo root, inside the conda env) resolve CUDA_HOME (external toolkit → conda nvcc → pip cu13 prefix), require `nvcc` for flashinfer JIT, and run the sglang host patcher (`patch_host_sglang.py` — anchors the `PRECOMPUTED_EMBEDDING` guard in `qwen3_vl.py::_get_visual_feature`, which `qwen3_5_moe` inherits, plus the frame-ratio sampling branch) before launching the sidecar. First launch JIT-compiles flashinfer kernels (minutes, then cached). Idempotent; verify with `curl http://<host>:<port>/stats`.

**Tested commands — Qwen3.6 serial:**

```bash
# sidecar (from repo root, inside the epd conda env)
bash start_sidecar_qwen36.sh \
    --model aicodec/aicodec/models/Qwen3.6-35B-A3B \
    --mem-fraction-static 0.85

# benchmark (from aitestbed/)
python tools/encpl_bench_eval_qwen36.py \
    --dataset mlvu --dataset-dir ./MLVU \
    --base-url http://localhost:8004 \
    --limit 10 --mode edge --vision-only \
    --compressor vidcom2 --retention-ratio 0.3 --bundle-quant fp8 \
    --edge-cpus "32-63" \
    --model ../aicodec/aicodec/models/Qwen3.6-35B-A3B
    
python tools/encpl_bench_eval_qwen36.py \
     --dataset mlvu --dataset-dir ./MLVU \
     --base-url http://localhost:8004 
     --limit 10 --mode server  
     --model ../aicodec/aicodec/models/Qwen3.6-35B-A3B
```

**Tested commands — Qwen3.6 concurrent:**

```bash
# sidecar (from repo root)
bash start_sidecar_concurrent_qwen36.sh 
    --max-running-requests 2 --mm-processor-worker-num 4 \
    --decode-threads 2 --preprocess-threads 4 \
    --model aicodec/aicodec/models/Qwen3.6-35B-A3B \
    --server-cpus "0-31" --enable-mixed-chunk \
    --mm-io-worker-num 4 --mem-fraction-static 0.85 \

# benchmark (from aitestbed/)
python tools/encpl_bench_eval_concurrent_qwen36.py \
	--mode server --users 8 \
    --base-url http://localhost:8014 \
    --runs 1 --db logs/encpl_conc_server_n4.db \
    --edge-cpu-ids "32-58" --edge-cpus-per-worker 2 \
    --enable-multi-processes --model ../aicodec/aicodec/models/Qwen3.6-35B-A3B \
	--dataset mlvu --dataset-dir /data01/mlvu/MLVU --limit 32

python tools/encpl_bench_eval_concurrent_qwen36.py \
	--mode edge --users 8 --vision-only \
    --compressor vidcom2 --retention-ratio 0.3 --bundle-quant fp8 \
    --base-url http://localhost:8014 \
    --runs 1 --db logs/encpl_conc_edge_n4.db \
    --model ../aicodec/aicodec/models/Qwen3.6-35B-A3B \
	--enable-multi-processes --edge-cpu-ids "32-58" \
	--edge-cpus-per-worker 3 --decode-threads 2 \
	--dataset mlvu --dataset-dir /data01/mlvu/MLVU	--limit 32
```

**Tested commands — Qwen3-VL serial:**

```bash
bash start_sidecar.sh \
    --model aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct \
    --mem-fraction-static 0.85

# benchmark (from aitestbed/)
python tools/encpl_bench_eval.py \
    --dataset mlvu --dataset-dir ./MLVU \
    --base-url http://localhost:8002 \
    --limit 10 --mode edge --vision-only \
    --compressor vidcom2 --retention-ratio 0.3 --bundle-quant fp8
    
python tools/encpl_bench_eval.py \ 
    --dataset mlvu --dataset-dir ./MLVU \
    --mode server --limit 10 \
    --base-url http://localhost:8002 \ 
    --model ../aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct/
```

**Tested commands — Qwen3-VL concurrent:**

```bash
bash start_sidecar_concurrent.sh \
    --max-running-requests 2 \
    --model ./aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct/ \
    --server-cpus "0-31" --enable-mixed-chunk --mm-io-worker-num 4 \
    --mm-processor-worker-num 4  --decode-threads 2 --preprocess-threads 4

# benchmark (from aitestbed/)
python tools/encpl_bench_eval_concurrent.py \
	--mode edge --users 8 --vision-only \
    --compressor vidcom2 --retention-ratio 0.3 --bundle-quant fp8 \
    --base-url http://localhost:8012 \
    --runs 1 --db logs/encpl_conc_edge_n4.db --model ../aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct \
	--enable-multi-processes --edge-cpu-ids "32-63" --edge-cpus-per-worker 3 --decode-threads 2 \
	--dataset mlvu --dataset-dir /data01/mlvu/MLVU	--limit 32	
    
python tools/encpl_bench_eval_concurrent.py \
	--mode server --users 8 \
    --base-url http://localhost:8012 \
    --runs 1 --db logs/encpl_conc_server_n4.db \
    --edge-cpu-ids "32-63" --edge-cpus-per-worker 2 \
    --enable-multi-processes --model ../aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct \
	--dataset mlvu --dataset-dir /data01/mlvu/MLVU --limit 32
```

**Benchmark parameter reference** (both serial and concurrent bench tools; the concurrent ones add the N-user/process rows):

| Flag | Meaning |
|:-----|:--------|
| `--dataset {videomme,mlvu}` | Benchmark set: Video-MME (HF parquet) or MLVU (official json/video layout) |
| `--dataset-dir PATH` | Dataset root (`data/`+`videos/` for videomme; `json/`+`video/` for mlvu) |
| `--base-url URL` | Sidecar endpoint — 8002/8012 Qwen3-VL serial/concurrent, 8004/8014 Qwen3.6 |
| `--limit N` | Cap on questions evaluated (0 = full set) |
| `--mode {server,edge}` | A/B arm: raw-video upload vs terminal-side ViT + bundle upload |
| `--model PATH` | Terminal-side ViT weights (edge mode) — defaults to the family's repo-local path |
| `--vision-only` | Load only the vision tower on the terminal (~1 GB VRAM; identical embeddings) |
| `--compressor vidcom2` | VidCom2 dynamic token pruning of the uploaded bundle (edge mode) |
| `--retention-ratio R` | Fraction of video tokens kept after pruning (0, 1] |
| `--bundle-quant fp8` | fp8 e4m3 bundle-feature quantization (needs `--compressor vidcom2`) |
| `--edge-cpus SPEC` | Serial tool: pin the whole terminal process to a CPU list (e.g. `32-63`), disjoint from the sidecar's cores; default `auto-half-other` = second half of the host's cores (disjoint from the sidecar's `auto-half` first half) |
| `--users N` | Concurrent wave size (raise the sidecar's `--max-running-requests` ≥ N) |
| `--enable-multi-processes` | One OS process per user — own ViT/GIL/connection, truly independent terminals |
| `--edge-cpu-ids SPEC` | Concurrent process mode: CPU pool workers are pinned to, round-robin. Default `auto-half-other` = second half of the host's cores — disjoint from the sidecar's `--server-cpus auto-half` first half; the runner fails fast when an explicit pool overlaps the sidecar pin it sees in `/stats` |
| `--edge-cpus-per-worker K` | Concurrent process mode: K CPUs per worker from the pool |
| `--decode-threads N` | Cap terminal decode threads (decord/torchcodec otherwise size by host cores) |
| `--runs N` | Repetitions recorded to the DB (waves × prompts per run) |
| `--db PATH` | Output SQLite traffic DB (separate file per arm for A/B comparison) |
| `--stream-connect-timeout SEC` | TCP connect timeout for sidecar HTTP requests (default 20; 0 disables) |
| `--stream-first-event-timeout SEC` | Budget from chat POST to the first SSE event — covers server-side edge prepare + queueing + prefill + TTFT. Serial tools default 200; **concurrent tools default 900** (barrier waves of N users against a small running-request cap queue tail users behind minutes of prefill) |
| `--stream-idle-timeout SEC` | Max silent gap between SSE events mid-stream (default 20; applied to the socket right after the first event) |
| `--stream-upload-read-timeout SEC` | Read timeout for the bundle-upload PUT response (default 1200; covers the server's store+ack after the upload-direction transfer) |

**Stream safety timeouts.** All four `--stream-*` flags above exist on every encoder-placement bench tool and scenario runner (both model families). They convert a wedged sidecar stream — historically an infinite hang of the reading loop (`iter_lines` with no timeout; observed as a 29.5 h frozen matrix) — into a per-request exception. `0` disables any guard. Equivalent env overrides for ad-hoc runs: `ENCPL_STREAM_CONNECT_TIMEOUT`, `ENCPL_STREAM_FIRST_EVENT_TIMEOUT`, `ENCPL_STREAM_IDLE_TIMEOUT`, `ENCPL_STREAM_UPLOAD_READ_TIMEOUT` (explicit CLI flags win; the matrix driver's `sanitized_env` strips these from cell subprocesses, so outer values cannot leak in).

**Scenario runners (traffic logging)** — per-request DB records (TTFT, token timing, upload bytes) instead of benchmark scoring; from the repo root:

```bash
python aitestbed/run_encoder_placement_qwen36.py --mode edge --runs 3 \
    --db logs/traffic_logs_qwen36_edge.db

# concurrent runner with per-user videos from MLVU (tested command)
python aitestbed/run_encoder_placement_concurrent.py \
    --mode edge --users 4 \
    --base-url http://172.16.148.166:8012 \
    --runs 1 --db logs/encpl_conc_edge_n4.db \
    --model ../aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct \
    --enable-multi-processes --edge-cpu-ids "32-58" --edge-cpus-per-worker 3 \
    --decode-threads 2 \
    --dataset mlvu --dataset-dir /data01/mlvu/MLVU --limit 32
```

| Runner flag (beyond the bench set above) | Meaning |
|:-----------------------------------------|:--------|
| `--runs N` | Repetitions recorded to the DB (waves × prompts per run) |
| `--db PATH` | Output SQLite traffic DB (separate file per arm for A/B comparison) |
| `--attn-impl sdpa\|flash_attention_4\|eager` | Edge ViT attention backend — not bit-identical; separate DBs per backend |
| `--preprocess-threads N` | Terminal CPU-side (resize/processor) thread cap; mirror on the sidecar too |
| `--frame-ratio R` | Edge: sampled locally via patched `smart_nframes`. Server: the ENGINE samples with its STARTUP config — start the sidecar with the SAME value |
| `--profile NAME` | Profile name recorded in the DB only; netem is NOT applied by these runners (apply `tc` yourself on the terminal's egress interface for dual-host shaping) |
| `--no-flush-cache` | Skip the per-request/per-wave `/flush_cache` (cold-prefix is the default) |

**Orchestrator path (dataset mode + bench report)** — the same five dataset keys work through `orchestrator.py` for all four `encoder_placement*` scenario types (env `ENCPL_DATASET_DIR` wins over `dataset_dir`; `dataset: null` = single-video mode). Enable them in `configs/scenarios.yaml` (or the standalone sibling yamls) — e.g. the qwen36 serial block defaults to `dataset: mlvu, dataset_dir: /data01/mlvu/MLVU, limit: 10`:

```bash
# point the client at the right sidecar first (qwen36 serial example)
export SGLANG_ENGINE_QWEN36_BASE_URL=http://localhost:8004
python orchestrator.py --scenario encoder_placement_qwen36 \
    --profile no_emulation --runs 2 --db logs/traffic_logs_encpl_q36.db
```

Bench scoring lands in the DB as per-record `metadata.bench` (answer, model_answer, correct, padded, task_type/duration). After each experiment the orchestrator aggregates it into `ScenarioMetrics.real_*` — `real_accuracy` uses **run-mean semantics** (mean of per-run accuracies, one session = one run; padded duplicates skipped, parse failures counted separately) — and it flows into the stdout `Real total/correct/accuracy` lines and `experiment_report.json`'s `task_accuracy` block, exactly like `realtime_video_understanding`. It also prints a bench-tool-style rich summary after each experiment (`[EncPlBench]` blocks: mode counts, OVERALL run-mean + pooled, accuracy by task_type/duration, per-run detail, E2E TTFT/LAT percentiles with zero-token EMPTY completions excluded, first-video vs steady TTFT and a per-user summary when waves exist), followed by the standalone runners' per-user detail — per-wave headers, per-video ttft/lat lines with EMPTY/FAIL cells, full video paths, upload bytes + time, and the up/chat connection 5-tuples (src_port is the pcap<->user join key) — the cross-wave overall TTFT line (per-wave aggregates), and the concurrent bench tool's per-user stat blocks (ttft/lat mean/p50/p90/p95/max per user with first/steady split), full per-video detail lines (video_id, upload/wire bytes, up_time, error rows) and pooled per-request overall TTFT/lat percentiles with ok/fail/empty counts. Note `--resume` only aggregates freshly-executed runs; use the DB report tool below for full history.

**Offline DB bench report** — per-task / per-run / per-question detail from any traffic DB (works on standalone-runner DBs too, and on runs pre-dating the orchestrator integration):

```bash
python tools/encpl_bench_db_report.py --db logs/traffic_logs_encpl_q36.db \
    --scenario encoder_placement_qwen36 --profile no_emulation

# all profiles of a scenario + per-question JSON dump
python tools/encpl_bench_db_report.py --db logs/encpl_conc_edge_n4.db \
    --scenario encoder_placement_concurrent --profile all \
    --json results/reports/encpl_conc_edge_questions.json
```

Prints OVERALL accuracy (run-mean + pooled counts), accuracy by task_type (MLVU) / duration bucket (Video-MME), parse failures, per-session detail, and E2E TTFT/latency percentiles (zero-token EMPTY completions excluded). The `--json` per-question dump carries `ttft_sec`/`latency_sec`/`tokens_out` so EMPTY completions (`tokens_out == 0`) are identifiable offline. Exit code 2 = no scored bench rows (single-video mode or wrong scenario/profile).

**Full-matrix driver (Qwen3.6)** — `tools/run_qwen36_matrix.py` runs the whole experiment unattended: serial + concurrent phases, server/edge arms, retention/quant ladders, all network profiles, per-cell pcap + SQLite DBs, auto sidecar management, and a final aggregated report. Designed for multi-day nohup/tmux runs:

```bash
# plan only — render every cell's exact command / netem / capture filter
python aitestbed/tools/run_qwen36_matrix.py --dry-run

# full run (defaults: serial 50 videos/cell, concurrent 3 videos/user)
python aitestbed/tools/run_qwen36_matrix.py \
    --server-gpu 0 --client-gpu 1 \
    --sidecar-mem-frac 0.92 --conc-mem-frac 0.85

# resume: completed cells are skipped, leftover db/pcap of unfinished
# cells are cleaned automatically and those cells re-run
python aitestbed/tools/run_qwen36_matrix.py \
    --resume logs/q36m/manifest_<ts>.json

# regenerate the report from an existing manifest
python aitestbed/tools/run_qwen36_matrix.py --report-only logs/q36m/manifest_<ts>.json
```

Driver flags:

| Flag | Default | Meaning |
|:-----|:--------|:--------|
| `--phases` | both | `serial` / `concurrent` / `both` |
| `--serial-limit N` | 50 | Videos per serial cell |
| `--conc-videos-per-user V[,V...]` | 3 | Comma list of videos per concurrent user; every (users, V) pair becomes its own concurrent cell with limit = V × users (cell count = users × V values × other ladders). V rides the arm tag as `-v{V}` (e.g. `q36m_conc_server-n8-v3__5g_urban`), so different V values never collide on cell_id/db/pcap/log and `--resume` cannot mistake one V for another |
| `--profiles`, `--retentions`, `--quants`, `--users` | no_emulation,5g_urban,wifi_good,cell_edge / 0.8,0.7,0.6,0.4,0.3 / fp8,bf16 / 2,4,8,16 | Matrix ladders. `--retentions` unset (default) keeps the historical split: serial phase runs the full default ladder while concurrent edge cells stay fixed at r=0.6. An explicit `--retentions 0.6,0.7` is honored by BOTH phases (concurrent edge arm tags gain `r0.7` etc., e.g. `q36m_conc_edge-r0.7-fp8-n8-v3__no_emulation`); `--retentions ''` disables edge cells in both phases |
| `--optional-retention` | (empty) | Comma list of ABSOLUTE optional-retention (o) values for the two-tier ladder, e.g. `0.06,0.12,0.18` — every (r, o) pair becomes a tiered cell in BOTH phases (invalid pairs with `0 < o < r` are warned and skipped). Empty keeps the legacy derived value `o = r*0.30`. Naming (both phases, no tier tag): two-tier `...-r0.6-fp8-opt0.18-n8-v3__` (serial `...-r0.6-fp8-opt0.18__`), single-tier `...-r0.6-fp8-n8-v3__` — the `-opt{g}` tag's presence IS the two-tier marker and carries the o value, keeping multiple o values per r collision-free |
| `--server-gpu` / `--client-gpu` | unset | Pin the sidecar engine's card / every cell subprocess's card (`CUDA_VISIBLE_DEVICES`) |
| `--sidecar-mem-frac` | unset | Serial sidecar `--mem-fraction-static` (0.92 recommended on a dedicated 80 GB card) |
| `--conc-mem-frac` | 0.85 | Concurrent sidecar `--mem-fraction-static` — **0.92 OOMs** on a shared 80 GB card (CUDA-graph pools + auxiliary engine processes eat the headroom; observed prefill-time crash) |
| `--stream-connect-timeout` | 20 | Forwarded to every cell's runner (see the stream timeout table above) |
| `--stream-first-event-timeout` | 200 | Serial cells' first-SSE-event budget |
| `--conc-stream-first-event-timeout` | 900 | Concurrent cells' first-SSE-event budget (separate knob — barrier waves queue tail users) |
| `--stream-idle-timeout` | 20 | Max silent gap between SSE events mid-stream |
| `--stream-upload-read-timeout` | 1200 | Bundle-upload PUT response read timeout |
| `--serial-cell-timeout SEC` | 36000 (10 h) | Per-cell wall-clock cap, serial phase |
| `--conc-cell-timeout SEC` | 14400 (4 h) | Per-cell wall-clock cap, concurrent phase |

**Per-cell wall-clock cap (hang protection).** Each cell's runner runs as its own process group; on timeout the driver `killpg`s it (runner + spawned edge workers), tears down netem and the pcap capture, records `error: "cell timeout"` in the manifest, and continues with the next cell — a wedged runner can no longer freeze a multi-day matrix (origin of the 2026-10-18 29.5 h freeze: SSE stream read without timeout AND no outer watchdog). `0` disables a cap.

**Artifacts & preflight.** Per cell: `logs/q36m/<cell_id>.db`, `aitestbed/results/captures/q36m/<cell_id>.pcap`, `logs/q36m/cell_logs/<cell_id>.log`; the run manifest `logs/q36m/manifest_<ts>.json` is rewritten after every cell. The preflight aborts when a fresh run would collide with existing artifacts — `--resume <manifest>` skips completed cells and auto-cleans leftovers; for a from-scratch re-run, delete/archive `logs/q36m/` and `aitestbed/results/captures/q36m/` first (pcaps run to hundreds of GB at full scale — check disk before launching). NOTE: concurrent cell ids gained the `-v{V}` suffix when `--conc-videos-per-user` became a list — an old manifest's concurrent cells (no `-v`) no longer match new ids, so `--resume` against one re-runs the whole concurrent phase (serial cells still carry over; archive the old concurrent db/pcap to reclaim disk).

**Report JSON = full DB mirror.** `results/q36m/report_<ts>.json` carries, per cell, a `requests` array with EVERY `traffic_logs` row verbatim: all scalar columns (`request_bytes`/`response_bytes`, `t_request_start`/`t_first_token`/`t_last_token`, `http_status`, `is_streaming`, …) plus the fully-parsed `metadata` dict (per-video facts, `conns` 5-tuples with per-video src ports, `bench` scoring, engine_stats) and parsed `inter_chunk_times` — everything needed for offline analysis without touching SQLite. Per-user detail lives in `metrics.by_user[u].videos` (one row per playlist entry: full video path, video/bundle/up MB, per-video up/chat ports, ttft, ok) plus `video_mb_sum`/`src_ports`; the CSV detail gains `bundle_mb`/`up_src_port`/`chat_src_port` columns; the markdown report adds a per-user video-detail subtable. pcap cache is versioned (`_pcap_parser`); captures with Ethernet framing (tcpdump on `lo`, linktype 1) parse correctly since parser v2 (older `flows:0` caches auto-invalidate).

**Sidecar parameter reference** (concurrent sidecars expose the full set; serial sidecars expose the subset without the concurrent-only rows):

| Flag | Meaning |
|:-----|:--------|
| `--model PATH` | Engine weights (repo-local default under `aicodec/aicodec/models/`) |
| `--host` / `--port` | Bind address (default `0.0.0.0`) and port (8002/8004 serial, 8012/8014 concurrent) |
| `--mem-fraction-static F` | GPU KV-cache fraction reserved by SGLang (default 0.85) |
| `--server-cpus SPEC` | Pin the sidecar + scheduler subprocess to a CPU set (default `auto-half` = first half of cores; keep disjoint from the terminals' `--edge-cpu-ids` pool — the concurrent runner preflight hard-fails on an overlap it sees in `/stats`, and the matrix driver preflight cross-checks `--conc-server-cpus` vs `--conc-edge-cpu-ids`) |
| `--max-running-requests N` | Scheduler cap on GPU-batched requests; above it requests queue (raise ≥ `--users`) |
| `--max-bundles N` | BundleStore capacity for uploaded bundles before LRU eviction (concurrent-only; default 64, must ≥ wave size) |
| `--chunked-prefill-size N` | Max tokens in one prefill chunk (sglang default ~8192; `-1` disables chunked prefill) |
| `--enable-mixed-chunk` | Mix decode requests into prefill chunks so in-flight decodes don't stall behind long prefills |
| `--enable-radix-cache` | Re-enable prefix caching (off by default in this benchmark — a warmed prefix would collapse cold-prefill concurrency) |
| `--mm-io-worker-num N` | Width of the server's video I/O pool (decode + sample + resize); N videos decode in parallel |
| `--mm-processor-worker-num N` | Width of the multimodal processor pool (normalize / tensorize); 1 = synchronous |
| `--min-new-tokens N` | Suppress EOS for the first N generated tokens (default 1; guards zero-token completions under concurrent prefills) |
| `--decode-threads N` | Cap the ENGINE-side decode thread pool (0 = native default) |
| `--preprocess-threads N` | Cap the ENGINE-side CPU preprocessing threads (resize / HF processor) |
| `--frame-ratio R` / `--max-frames N` | Engine-wide frame sampling; startup-time config — restart the sidecar to change, and mirror the runner's value for symmetric arms |
| `--parallel-video-preprocess` | Concurrent-only knob enabling parallel video preprocessing inside the engine |

**Environment variables (fallbacks; CLI flags win):** `ENC_MODE_QWEN36`/`ENC_MODE`, `SGLANG_ENGINE_QWEN36_MODEL`/`_HOST`/`_PORT`, `SGLANG_ENGINE_QWEN36_BASE_URL` (bench tool), `ENCPL_EDGE_CPUS`, `ENCPL_DECODE_THREADS`(+`_QWEN36`), `ENCPL_PREPROCESS_THREADS_QWEN36`, `ENCPL_BUNDLE_QUANT`, `ENCPL_VISION_ONLY`, `ENCPL_FRAME_RATIO`, `COMPRESSOR`/`R_RATIO`, and the stream-timeout overrides `ENCPL_STREAM_CONNECT_TIMEOUT` / `ENCPL_STREAM_FIRST_EVENT_TIMEOUT` / `ENCPL_STREAM_IDLE_TIMEOUT` / `ENCPL_STREAM_UPLOAD_READ_TIMEOUT` (ad-hoc manual runs only — the matrix driver forwards explicit values and `sanitized_env` strips these from cell subprocesses).

**Model-specific notes:**

- Qwen3.6 (`qwen3_5_moe`, hybrid GDN linear-attention MoE): the vision tower emits a single merged `[N, 2048]` latent segment — **no deepstack** (unlike Qwen3-VL 30B), 4 KB/token at bf16. `vision_only: true` is the yaml default; the same on-disk sglang patch covers it via inheritance.
- Qwen3-VL: deepstack multi-segment latents; the concurrent bench tool defaults the full 4B edge load.

**Correctness gates & benchmarks:**

- `tools/encpl_vision_only_check_qwen36.py` — equivalence gate for the standalone-tower edge load (loading-report, checkpoint census, dry-run remap, bit-identical weight and forward-equivalence checks vs the full reference arm). Exit 0 = safe to bench with `vision_only=True`.
- `tools/encpl_bench_eval_qwen36.py` (serial) / `tools/encpl_bench_eval_concurrent_qwen36.py` — Video-MME + MLVU accuracy + latency benchmark drivers (both arms, both dataset layouts; MCQ tasks scored, generation tasks recorded for offline GPT scoring; cache flushed per question by default). Same tools exist un-suffixed for Qwen3-VL.
- `tools/encpl_bench_db_report.py` — offline accuracy report for orchestrator/runner DBs (reads `metadata.bench` rows; see the orchestrator path section above).

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
