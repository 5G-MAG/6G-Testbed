# 6G AI Traffic Characterization Testbed

A testbed for measuring AI/LLM service traffic patterns under emulated network conditions, aligned with **3GPP SA4 6G Media Study** objectives.

## Overview

The testbed enables:

- **Measurement** of traffic characteristics across generative AI services (chat, image, video, realtime voice)
- **Analysis** of agentic AI patterns (multi-step tool calling via MCP, browser automation, market data, OpenClaw, A2A)
- **Evaluation** of QoE metrics under emulated network conditions (latency, loss, bandwidth)
- **Reporting** in formats suitable for 3GPP standardization contributions

### Relationship to netemu and training

This directory is one of three components in the repository.

| Component | Provides | Direction |
|:----------|:---------|:----------|
| [`netemu/`](../netemu/) | tc/netem shaping, tcpdump capture, pcap parsing and network-layer metrics | aitestbed **depends on** netemu |
| `aitestbed/` (here) | Scenarios, LLM/agent clients, application-layer metrics, RAN2 metrics, reports | |
| [`training/`](../training/) | Traffic-pattern classifier and generative traffic model | training **consumes** aitestbed output files |

The network layer lives entirely in `netemu`, which has no dependency on this
testbed and is usable on its own:

| What | Module |
|:-----|:-------|
| Shaping (`tc`/`netem`/HTB/IFB) | `netemu.emulator` |
| L3/L4 capture (`tcpdump`) | `netemu.capture` (re-exported here as `capture.CaptureController`) |
| PCAP parsing, TCP flow reassembly, packet/flow/window metrics | `netemu.pcap` (re-exported here as `analysis.PcapAnalyzer` and friends) |

What stays in this directory is everything that depends on the testbed's own
semantics: the L7 mitmproxy capture with its redaction policy
(`capture/l7_capture.py`), the application-layer metrics defined against the
SQLite log schema (`analysis/metrics.py`), the RAN2 S4-260859 families that
join pcap metrics to session records (`analysis/ran2_metrics.py`), and the
report/chart/Excel generators.

For how network-layer metrics are computed (direction attribution,
retransmission detection, TLS handshake bounds, burstiness windows, burst
segmentation), see the **PCAP Processing** section of
[netemu/README.md](../netemu/README.md). For the application-layer and RAN2
metrics, see [METRICS.md](METRICS.md).

## Cross-Checking for SA4 AI Traffic Characterization

For 3GPP SA4 cross-checking (reproducing or validating contributed results), the canonical entry point is **`run_full_tests.sh`**. It drives the full test matrix in `configs/scenarios.yaml` across the 10 SA4 S4-260848 network profiles, captures L3/L4 PCAPs, and runs the complete post-processing pipeline (charts, Excel, `RESULTS.md`, `TRACES.md`, DB anonymization).

### Quick start

```bash
cd aitestbed
cp .env.example .env          # fill in at least OPENAI_API_KEY

# Smoke test, 3 runs/scenario, ~30 min depending on scenarios enabled
bash run_full_tests.sh --quick

# Cross-check run, 10 runs/scenario by default, hours to a day
bash run_full_tests.sh

# Publication run, 30 runs/scenario
bash run_full_tests.sh --full

# Narrow to a single phase to reproduce a specific contribution
bash run_full_tests.sh --enable chat --runs 30
bash run_full_tests.sh --enable realtime --runs 30
bash run_full_tests.sh --enable vllm --runs 30
```

### Parameters and when to use them

| Flag | Effect | When to use |
|:-----|:-------|:------------|
| `--quick` | 3 runs/scenario, short delays | Smoke test before a long run; verifying config/env |
| `--full` | 30 runs/scenario | Publishable statistics, recommended for SA4 cross-check |
| `--runs N` | Exact number of runs | Match the run count used in the contribution being cross-checked |
| `--enable LIST` | Only run listed phases (comma-sep.) | Reproducing a specific scenario family |
| `--disable LIST` | Skip listed phases | Skip what you cannot run (no API key, no GPU, etc.) |
| `--stress` | Enable burst/parallel stress phase | Only for stress-testing contributions (opt-in) |
| `--no-capture` | Disable L3/L4 PCAP | Debugging only, **leave ON for cross-check runs** |
| `--no-anonymize` | Keep real provider/model names | Internal triage; SA4 submissions use anonymized DB |
| `--no-clean` | Keep prior DB and pcaps | Accumulate across runs (off by default) |
| **`--resume`** | **Skip already-completed combos, append to existing DB** | **Critical: use whenever a previous run was interrupted or a scenario failed. Implies `--no-clean`. See dedicated section below.** |
| `--verbose, -v` | Show full log instead of progress bar | Debugging; default is a single progress bar |

Phase names (for `--enable` / `--disable`): `chat, realtime, image, search, deepseek, gemini, music, trading, computer_use, playwright, multimodal, google_search, stress, vllm, openclaw, a2a`.

### `--resume`: recovering an interrupted run

SA4 cross-check runs can easily take 8 to 24 hours. **Assume at least one will be interrupted**, a quota error, a SIGKILL, a laptop lid, an OS reboot, or a single flaky scenario can derail a long matrix. `--resume` is how you pick up where you left off without losing the hours of data already captured.

**What it does.** Re-run with the same parameters and add `--resume`:

```bash
bash run_full_tests.sh --resume                   # keep original defaults
bash run_full_tests.sh --resume --runs 30         # match original run count
bash run_full_tests.sh --resume --enable vllm     # resume just one phase
```

On startup the script queries `logs/traffic_logs.db` once and drops every `(scenario, profile)` combo that already has `≥ RUNS_PER_SCENARIO` completed sessions from the test matrix. Remaining combos get their missing runs filled in; new runs append to the DB under fresh `session_id`s, no existing records are rewritten.

**Implies `--no-clean`.** `--resume` does **not** archive or wipe `logs/traffic_logs.db`, `results/captures/`, or `results/reports/`. That is the whole point, pass it **exactly when** you want to preserve prior data.

**What counts as "completed"** (orchestrator.py `get_completed_runs`):

| Session kind | Resume treats it as | Reason |
|:-------------|:--------------------|:-------|
| All records `success=1` for that session | ✓ completed, **skipped** | Data point is valid |
| `session_id LIKE 'timeout_%'` | ✓ completed, **skipped** | Timeout under harsh profiles is a legitimate measurement, not an error to retry |
| Any record `success=0` in the session | ✗ not completed, **retried** | Real failure, e.g. API error / crash, must produce a clean run |
| `session_id LIKE 'pcap_%'` | ignored (not a run) | Capture-only placeholder, excluded from counting |

Because successful sessions and timeout placeholders both count, the target run count eventually fills even under lossy profiles like `cell_edge` or `satellite_geo`.

**When to use `--resume`:**

- The run was interrupted for any reason (`Ctrl-C`, SIGKILL, OOM, reboot, power loss).
- `STOP_ON_ERROR=true` (default) tripped on a single failing scenario, fix the cause (quota, network, API key) and resume. The message `Test suite stopped due to failure. Re-run with --resume to continue.` is the prompt to do this.
- You ran `--quick` first to sanity-check and now want to top up to `--full` (pass `--resume --full`, existing 3 runs count against the 30-run target).
- A long phase (e.g. `vllm`, `realtime`) failed due to transient infra; fix the infra and `bash run_full_tests.sh --resume --enable <phase>`.

**When NOT to use `--resume`:**

- You changed `configs/profiles.yaml` or a scenario's model/provider/prompt. The prior runs are no longer comparable, start fresh (drop `--resume`, let `CLEAN_START=true` archive the old DB).
- You are starting a brand-new cross-check. First run should not have `--resume`, it wipes-and-archives correctly on its own.
- You want a deterministic, single-batch dataset where every record was produced by one invocation. `--resume` stitches across invocations; this is usually fine for SA4 cross-check but explicit to call out.

**Safety properties.** Every `--resume` invocation (a) archives nothing, (b) mutates no existing rows, (c) only appends new sessions, (d) re-runs post-processing (`RESULTS.md`, charts, Excel) over the full DB at the end, so the final artifacts reflect all accumulated data, not just the resumed slice. Running `--resume` against a fully-complete DB is a no-op for data collection; it will just regenerate the reports.

**Tip:** pair long runs with `nohup` or `tmux` so an SSH drop does not kill the script. Even so, leave `--resume` in your pocket, you will need it.

Key environment variables (see `--help` for the full list):

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `RUNS_PER_SCENARIO` | 10 | Runs per scenario/profile combo (`--full` sets 30) |
| `RUN_TIMEOUT_SEC` | 600 | Per-run timeout; raise for slow reasoning models |
| `NETWORK_INTERFACE` | `auto` | Pin a specific egress interface instead of auto-detect |
| `MCP_TRANSPORT` | `http` | MCP over HTTP so tool traffic is shaped by tc/netem. Set `stdio` only to bypass shaping for MCP |
| `CAPTURE_FILTER` | HTTP(S) plus WebRTC UDP ports | BPF filter covering ports 443/80/8080/8000 and ICE/STUN/TURN/SRTP/RTP/RTCP UDP ports 3478/3479/5349/5350/19302 |
| `MANAGE_VLLM` | `true` | Auto-start/stop the vLLM container. Set `false` when you already run vLLM yourself |
| `VLLM_BACKEND` | `docker` | `docker` (recommended) or `host` (needs `vllm` on PATH) |
| `MANAGE_OPENCLAW` | `true` | When the `openclaw` phase is enabled: auto-install (if missing), start, and stop the OpenClaw gateway. Set `false` to manage it yourself |
| `OPENCLAW_AUTO_INSTALL` | `true` | Install the version pinned by `run_full_tests.sh` if the CLI is missing |
| `TRACE_PAYLOADS` | `0` | Opt in to private, recursively redacted payload traces |
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
- **Missing API keys silently skip phases.** The prereq check logs `-` lines for each missing key. Inspect the preamble before a long run, a greyed-out phase produces no data.
- **Playwright phase needs Chromium installed:** `pip install playwright && playwright install chromium`. Same for the `computer_use` phase.
- **Run timeouts** on DeepSeek Reasoner / long agent chains: raise `RUN_TIMEOUT_SEC=1200` (or higher) if you see `timeout_*` session IDs in the DB.
- **First PCAP on loopback is empty**: MCP-over-HTTP traffic capture needs `CAPTURE_LOOPBACK=true` (default). If you set `MCP_TRANSPORT=stdio`, tool traffic is not shaped and not captured, this is intentional but not what you want for SA4 cross-check.
- **API rate limits / quota errors** show up as red `ERROR` rows in the DB with non-zero `http_status`. Re-run with `--resume` after the quota resets; `--resume` skips combos that already have `RUNS_PER_SCENARIO` successful sessions.
- **Disk and time budget**: a full run (30 runs × full matrix × 10 profiles with PCAP) produces a few GB of pcaps and can run 8 to 24 h depending on enabled providers. Use `--enable <phase>` to cross-check one contribution at a time.

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

The WebRTC scenarios use the GA unified Realtime handshake: the SDP offer and
session configuration are sent as multipart fields to `/v1/realtime/calls`,
then events use the GA `session` and `response.output_*` schema over the data
channel. The retired Realtime Beta `application/sdp` request and
`OpenAI-Beta: realtime=v1` header are not used.

For a focused RTP capture smoke test:

```bash
python orchestrator.py --scenario realtime_audio_webrtc \
  --profile no_emulation --runs 1 --interface auto \
  --capture-pcap --capture-dir results/captures
```

The primary pcap should contain the negotiated bidirectional UDP flow to one
of the default WebRTC service ports. The paired loopback pcap contains only
local testbed traffic and is not the WebRTC media source.

### Direct Web Search (No MCP)

| Scenario | Engine | Threads | LLM Synthesis | Description |
|:---------|:-------|:--------|:--------------|:------------|
| `direct_web_search` | DuckDuckGo | 5 | Yes (OpenAI) | Multi-threaded search |
| `direct_web_search_deepseek` | DuckDuckGo | 5 | Yes (DeepSeek) | Multi-threaded search |
| `direct_web_search_google` | Google | 5 | Yes | Google Custom Search API |
| `direct_web_search_burst` | DuckDuckGo | 20 | No | Burst stress test |
| `parallel_search_benchmark` | DuckDuckGo | 1-20 | No | Parallelism benchmark |

### Agentic Runtimes: OpenClaw & A2A

Two agentic-traffic families. Quick reference below; full documentation in
[Agentic Runtimes in Detail](#agentic-runtimes-in-detail-openclaw--a2a) and the
design doc [AGENTIC.md](AGENTIC.md). Both are **opt-in** phases.

| Scenario | Type | Transport | Description |
|:---------|:-----|:----------|:------------|
| `openclaw_personal_assistant` | OpenClaw | Local gateway (lo:18789) + LLM egress | Drives the local OpenClaw agent; control channel shaped/captured on loopback, LLM/tool egress shaped on the main interface |
| `a2a_single_task_local` | A2A | JSON-RPC `message/send` on loopback | Non-streaming task against a local echo agent |
| `a2a_streaming_local` | A2A | JSON-RPC `message/stream` (SSE) on loopback | Streaming artifact updates against a local echo agent |
| `a2a_multi_agent_local` | A2A | Multi-hop on loopback | Orchestrator agent delegates to two downstream agents |
| `a2a_single_task_remote` / `a2a_streaming_remote` | A2A | JSON-RPC over egress | Against an external agent (`A2A_REMOTE_AGENT_URL`) |

## Agentic Runtimes in Detail: OpenClaw & A2A

These two families measure **agentic** traffic, a local personal-assistant agent
(OpenClaw) and agent-to-agent protocol traffic (A2A). They differ from the MCP
agent scenarios: OpenClaw is an external runtime that calls its *own* model, and
A2A agents are deterministic (no LLM) so they isolate transport/protocol cost.
The full design and rationale are in [AGENTIC.md](AGENTIC.md).

Both register a no-LLM `provider` (`openclaw` / `a2a` → an internal
`NullLLMClient`) so they construct without an API key, and both are gated behind
explicit `--enable` because they need external setup.

---

### OpenClaw (local personal-assistant agent)

**What it is.** [OpenClaw](https://openclaw.ai) is a local-first, self-hosted
personal AI agent (Node.js) that connects an LLM to browser/file/shell tools and
chat channels. It runs a local **gateway on `127.0.0.1:18789`** and calls a cloud
or local LLM *inside its own process*, so it does **not** go through the
testbed's `LLMClient`.

**How the testbed drives it.** The scenario shells out to the OpenClaw CLI, one
agent turn per prompt:

```
openclaw agent --json --session-id <random> -m "<prompt>"
```

`agent` runs the turn *via the gateway* (not `--local`), so the request travels
the loopback control channel; `--json` returns a parseable result
(`result.payloads[].text`, `meta.agentMeta.usage`); a fresh `--session-id` per
turn prevents context accumulation from skewing token/byte measurements.

**Architecture & traffic model.**

```
OpenClawScenario (scenarios/openclaw_agent.py)
   └─ OpenClawExecutor ── spawns ──►  openclaw agent --json -m "<prompt>"
                                          │  JSON-RPC/WS over loopback
                                          ▼
                                   OpenClaw gateway (127.0.0.1:18789)   ← shaped+captured on lo
                                          │  HTTPS
                                          ▼
                                   LLM provider + tools (OpenAI, web, …) ← shaped by egress netem,
                                                                            captured by primary pcap
```

Two surfaces are measured: the **loopback control channel** (lo:18789, shaped via
`loopback_ports` + captured by the secondary `lo` pcap) and the **LLM/tool
egress** (main interface, shaped by egress netem + primary pcap).

**Lifecycle (`run_full_tests.sh` auto-manages it; `MANAGE_OPENCLAW=true` default).**
When the `openclaw` phase is explicitly enabled, the runner:
1. activates **nvm** if the system Node is older than 22 (`OPENCLAW_MIN_NODE_MAJOR`);
2. installs the CLI if missing (`OPENCLAW_AUTO_INSTALL=true`) into a user-writable
   prefix (`OPENCLAW_NPM_PREFIX`, default `~/.npm-global`, so `npm -g` needs no root);
3. sets `gateway.auth.mode=none` so the local `openclaw agent` CLI can open the
   gateway WebSocket without credentials;
4. starts the gateway: `openclaw gateway run --allow-unconfigured --auth none --port 18789 --force`;
5. runs the scenario (the model uses your `OPENAI_API_KEY`, default `openai/gpt-5.5`);
6. stops the gateway on exit via a **port-based** teardown (it forks/detaches).

If Node < 22 and nvm is absent, it aborts with an upgrade hint. Set
`MANAGE_OPENCLAW=false` to manage OpenClaw yourself; override commands via
`OPENCLAW_START_CMD` / `OPENCLAW_STOP_CMD`.

**Scenario config** (`openclaw_personal_assistant` in `configs/scenarios.yaml`):

| Field | Meaning |
|:------|:--------|
| `drive` | `cli` (run `openclaw agent …`) or `http` (POST to the gateway) |
| `cli_args` | CLI template; `{session}`/`{prompt}` placeholders substituted per turn |
| `gateway_url` | Gateway base URL (`http://127.0.0.1:18789`) |
| `loopback_ports` | `[18789]` → triggers loopback capture + shaping of the control channel |
| `task_timeout_sec` | Per-turn timeout |

**Metrics recorded** (per turn, SQLite `traffic_logs`): `latency_sec`,
`request_bytes`/`response_bytes`, token estimates, and `metadata` with
`record_type: openclaw_task`, `channel`, `steps`, `tools_used`, `output_chars`.

**Caveats (important when reading results):**
- `response_bytes` is the **CLI `--json` blob size** (~20 KB, metadata-heavy), *not*
  network bytes. Use the **pcaps** for real volume.
- The CLI call is blocking/non-streaming, so **TTFT ≈ TTLT ≈ total latency**, the
  streaming fields aren't separately meaningful for this drive.
- The runner's end-of-run summary `avg_lat` is **inflated** by pcap placeholder
  rows (whose "latency" is the capture *duration*); `RESULTS.md` (task turns only)
  is the correct latency.

**Key finding.** OpenClaw task latency is **LLM-inference-bound** (~10 to 13 s baseline
on ideal networks), with the network a secondary penalty, degraded profiles add
roughly +45 % (satellite_geo), +75 % (cell_edge), +110 % (congested).

**Run it.**

```bash
bash run_full_tests.sh --enable openclaw --runs 5            # auto-managed, all profiles
# Manual (gateway already running):
python orchestrator.py --scenario openclaw_personal_assistant --profile 5g_urban --runs 1 --capture-pcap
```

---

### A2A (Agent2Agent protocol)

**What A2A is.** An open standard for agent-to-agent interop (Google → Linux
Foundation). Wire mechanics the testbed exercises:
- **Transport:** JSON-RPC 2.0 over HTTP(S).
- **Discovery:** each agent publishes an **Agent Card** at
  `/.well-known/agent-card.json` (name, `capabilities.streaming`, skills, and the
  `supportedInterfaces` URL clients should dial).
- **Two call shapes:** `message/send` (non-streaming, one aggregated result) and
  `message/stream` (SSE, incremental events).
- **Task lifecycle:** a message creates a *Task* (`submitted → working →
  completed`) that emits *artifacts* (the reply). Client messages are
  `ROLE_USER`, agent replies `ROLE_AGENT`.

The testbed uses the official `a2a-sdk` (1.1.0) for both client and server. The
local agents are **deterministic (no LLM)**, this isolates the protocol/transport
cost of agent-to-agent traffic for a clean per-profile baseline.

**The scenarios** (`scenarios/a2a_agent.py`, sharing `_A2ABaseScenario`):

| Scenario `type` | Call shape | Topology | Measures |
|:----------------|:-----------|:---------|:---------|
| `a2a_single_task` | `message/send` | client → 1 agent | Baseline request/response |
| `a2a_streaming` | `message/stream` (SSE) | client → 1 agent | TTFT vs TTLT, inter-chunk gaps, stalls |
| `a2a_multi_agent` | `message/send` | client → orchestrator → 2 agents | Multi-hop delegation / fan-out |

Each has a **local** variant (agents launched on loopback, no keys) and a
**remote** variant (external agent over egress).

**Architecture (three layers).**

```
orchestrator.py  →  scenarios/a2a_agent.py  (asyncio driver)
                       ├─ launches ─► a2a_agents/  (real a2a-sdk servers, subprocesses)
                       │                 echo_agent.py        leaf agent (echo/upper/reverse)
                       │                 orchestrator_agent.py delegates to downstream agents
                       │                 common.py            AgentCard + Starlette/uvicorn + routes
                       │                 launcher.py          spawn + wait-ready + stop
                       └─ drives ────► clients/a2a_client.py  (A2AClientSession)
                                          wraps a2a-sdk, instruments bytes/timing per turn
```

- **`a2a_agents/echo_agent.py`**, emits the reply as N **appended artifact chunks**
  (`--stream-chunks`, `--stream-delay-ms`), which is what gives the *streaming*
  scenario multiple observable SSE events.
- **`a2a_agents/orchestrator_agent.py`**, is *itself an A2A client*; fans the input
  out to downstream agents and aggregates, producing the multi-hop traffic.
- **`clients/a2a_client.py`**, `A2AClientSession.send()` builds a
  `SendMessageRequest` and iterates `send_message` events, accumulating
  `A2ATurnMetrics` (TTFT/TTLT, `chunk_count`, `inter_chunk_times`, request/response
  bytes via httpx hooks, assembled text).

**Traffic model & shaping.** Local agents run on **loopback**; the scenario's
`loopback_ports` makes the orchestrator capture a secondary `lo` pcap and shape the
**target port** via netem (same mechanism as MCP-over-HTTP). Remote agents are
reached over the **main interface** (egress netem + primary pcap). *Limitation:*
netem's single `lo` root means only the target port is shaped, in multi-agent,
the client→orchestrator hop is shaped; downstream hops are captured but not
individually shaped.

**Metrics recorded** (per turn): `request_bytes`/`response_bytes`,
`t_first_token`/`t_last_token` → TTFT/TTLT, `latency_sec`, `is_streaming`,
`chunk_count`, `inter_chunk_times`, token estimates, and `metadata` with
`record_type` (`a2a_task` or `a2a_delegation`), `agent_url`, `event_count`, and
(multi-agent) `downstream_agents`.

**Scenario config fields** (`configs/scenarios.yaml`):

| Field | Meaning |
|:------|:--------|
| `streaming` | `false` → `message/send`; `true` → `message/stream` |
| `agents` | List of local agents to launch (`module`, `port`, `args`) |
| `target_port` | Which launched agent the client talks to (also the shaped port) |
| `loopback_ports` | Ports to capture + (target) shape |
| `agent_card_url` / `agent_card_url_env` | Remote agent base URL (literal or from env) |
| `agent_card_path` | Override card path (e.g. legacy `/.well-known/agent.json`) |
| `request_timeout_sec` | Client httpx timeout (default 60 s; for cold starts / high RTT) |

**Remote evaluation (self-host + tunnel).** There are effectively **no usable free
public A2A agents**, community registries are mostly dead/gated/paid, and the one
open public echo agent is protocol-incompatible with current `a2a-sdk`. So
self-host and expose via a tunnel:

```bash
cloudflared tunnel --url http://localhost:9001            # -> https://<rand>.trycloudflare.com
python -m a2a_agents.echo_agent --port 9001 --public-url https://<rand>.trycloudflare.com
export A2A_REMOTE_AGENT_URL=https://<rand>.trycloudflare.com
python orchestrator.py --scenario a2a_single_task_remote --profile 5g_urban --runs 5
```

`--public-url` makes the agent's card advertise the **tunnel** URL (so the client
dials the tunnel, not localhost). For a stable URL, host on a cheap VM and point
`A2A_REMOTE_AGENT_URL` at it. See
[AGENTIC.md](AGENTIC.md#remote-a2a-evaluation-self-host--tunnel).

**Run it.**

```bash
bash run_full_tests.sh --enable a2a --runs 10               # all local scenarios × profiles
python orchestrator.py --scenario a2a_single_task_local --profile cell_edge --runs 5 --capture-pcap
python orchestrator.py --scenario a2a_streaming_local   --profile 5g_urban  --runs 5
python orchestrator.py --scenario a2a_multi_agent_local --profile congested --runs 5
```

**What the measurements show.** Because the agents are deterministic, latency is
pure transport/protocol and scales cleanly with profile severity (representative
medians):

| Scenario | `no_emulation` | `congested` |
|:---------|:---------------|:------------|
| `a2a_single_task` | ~1.3 s | ~4.9 s |
| `a2a_streaming` | ~2.8 s | ~7.8 s |
| `a2a_multi_agent` | ~3.7 s | ~8.3 s |

`multi_agent` is highest (two extra downstream hops); `streaming` sits between
(artifact-chunk pacing adds round-trips). This complements OpenClaw's
inference-bound profile by isolating the agent-to-agent control plane.

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
| openclaw | `openclaw_personal_assistant` | medium | `openclaw` runtime + daemon (opt-in) |
| a2a | `a2a_single_task_local`, `a2a_streaming_local`, `a2a_multi_agent_local`, `a2a_single_task_remote` | high/low | `a2a-sdk` (opt-in; remote needs `A2A_REMOTE_AGENT_URL`) |
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

Metrics come from two layers, documented separately:

| Layer | Computed by | Documentation |
|:------|:------------|:--------------|
| **Application** — TTFT, TTLT, percentiles, tokens, agent loops, stalls, burstiness of requests | `analysis/metrics.py` (from the SQLite log) | [METRICS.md](METRICS.md) |
| **RAN2 methodology (S4-260859 Q1-Q5)** | `analysis/ran2_metrics.py` (joins pcap metrics to session records) | [METRICS.md](METRICS.md#ran2-methodology-metrics-s4-260859) |
| **Network** — handshake RTT, TLS setup, retransmissions, per-direction volumes, windowed throughput, burst segmentation | `netemu.pcap` (from the pcaps) | [netemu/README.md](../netemu/README.md#pcap-processing) |

Summary of the application layer:

- **QoE**: TTFT, TTLT, tail percentiles (P50/P95/P99), session completion rate
- **Traffic**: UL/DL byte volumes, UL/DL ratio, token streaming rate, burstiness descriptors, stall rate
- **AI Service**: Agent loop factor, tool call latency, multi-step completion time, error taxonomy

The RAN2 families depend on pcap input: run with `--capture-pcap` (and
`netemu[pcap]` installed) or Q1.3/Q1.4, Q2.x, Q3.1/Q3.2, and Q4.5-Q4.7 stay
empty.

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
│              netemu.emulator — Network Emulator (tc/netem)       │
│       Bandwidth │ Latency │ Jitter │ Loss │ Reorder │ Corrupt   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │  Internet │ → LLM APIs + Tool APIs
                    └───────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Capture & Analysis                           │
│  ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │    L3/L4     │ │   L7     │ │  SQLite  │ │  App-layer + │   │
│  │  tcpdump     │ │mitmproxy │ │  Logger  │ │ RAN2 metrics │   │
│  │netemu.capture│ │ (local)  │ │ (local)  │ │   + Plots    │   │
│  └──────┬───────┘ └──────────┘ └──────────┘ └──────▲───────┘   │
│         │  pcap                                     │           │
│         ▼                                           │           │
│  ┌──────────────────────────────────────┐           │           │
│  │ netemu.pcap — flow reassembly, RTT,  │───────────┘           │
│  │ retransmits, per-direction volumes,  │  PcapMetrics          │
│  │ windowed throughput, burstiness      │                       │
│  └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

Boxes marked `netemu.*` come from the sibling package; everything else is
local to this directory.

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (for MCP servers)
- Linux with `iproute2` (for network emulation)
- `tcpdump` (for PCAP capture)
- Sudo access or Docker with `NET_ADMIN`

### Setup

```bash
cd <repo-root>
python -m venv venv
source venv/bin/activate

# Install netemu (sibling package) with the pcap extra. The extra pulls in
# dpkt, which netemu.pcap needs; without it, capture still works but pcap
# analysis and all pcap-derived charts are skipped.
pip install -e "netemu[pcap]"

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
```

Or copy `.env.example` to `.env` and fill in values.

### Sudoers (for Network Emulation)

```bash
TCPDUMP_PATH=$(which tcpdump)
echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/tc, $TCPDUMP_PATH, /usr/sbin/modprobe, /usr/sbin/ip" \
  | sudo tee /etc/sudoers.d/testbed
sudo chmod 440 /etc/sudoers.d/testbed
```

### vLLM (Self-Hosted Models)

Two ways to run the vLLM server. **Docker is recommended**, the
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

#### Host pip install (fallback, auto-managed by the scripts)

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

- **SQLite** (`logs/traffic_logs.db`), per-request records with timestamps, bytes, tokens, TTFT/TTLT
- **JSON report** (`results/reports/experiment_report.json`), aggregated metrics in 3GPP-compatible format
- **Markdown tables** (`results/reports/experiment_tables.md`), QoE and traffic summaries
- **Plots** (`results/reports/figures/`), latency CDFs, UL/DL ratios, success rates

## Extending

### New Scenario

1. Subclass `BaseScenario` in `scenarios/`
2. Register in `scenarios/__init__.py`
3. Add config in `configs/scenarios.yaml`

### New LLM Provider

1. Subclass `LLMClient` in `clients/`
2. Register in `clients/__init__.py`
3. Add to orchestrator's client factory

### New network profile or network-layer metric

Profiles are parsed by `netemu.profile.NetworkProfile` and applied by
`netemu.emulator`; add the profile to `configs/profiles.yaml` and, if it needs
a netem parameter that is not modelled yet, extend the dataclass in `netemu`.
New packet-, flow-, or window-level metrics belong in `netemu.pcap`; new
metrics that combine pcap output with session records belong in
`analysis/ran2_metrics.py` here.

## Testing

```bash
python -m pytest tests            # testbed correctness checks
python -m pytest ../netemu/tests  # shaping, capture, and pcap analysis
```

## References

- [3GPP TR 26.870](https://www.3gpp.org/): 6G Media Study
- [3GPP TR 22.870](https://www.3gpp.org/): Service requirements for 6G
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [Linux tc(8)](https://man7.org/linux/man-pages/man8/tc.8.html)
