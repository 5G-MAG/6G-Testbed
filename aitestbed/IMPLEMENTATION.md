# IMPLEMENTATION

This document describes the implementation details of the 6G AI Traffic Characterization Testbed, focusing on runtime flow, key classes, and where metrics are collected.

## Boot and Configuration
- Environment is loaded via `dotenv` in `orchestrator.py` (current working directory `.env` and repo root `.env`).
- Scenario definitions and test matrix come from `configs/scenarios.yaml`.
- Network profiles are defined in `configs/profiles.yaml` and loaded by `netem/controller.py`.
- MCP server definitions and tool aliases are in `configs/mcp_servers.yaml` and consumed by agent scenarios.

## Core Orchestration Flow
`TestbedOrchestrator` in `orchestrator.py` coordinates execution:
- Loads scenarios config once at init and constructs a `TrafficLogger` and `NetworkEmulator`.
- Maps `scenario_type` to classes via `self.scenario_classes`.
- Selects provider clients lazily via `get_client()`.
- Applies network emulation before runs and clears it afterward.
- Retries transient failures (timeouts/429/5xx excluding 501) with exponential backoff using `defaults.retry_*`.

Main run loop (simplified):

```
scenario_config = scenarios_config["scenarios"][name]
scenario_class = scenario_classes[scenario_config.type]
client = get_client(scenario_config.provider)
scenario_config["scenario_id"] = name
scenario = scenario_class(client, logger, scenario_config)
emulator.apply_profile(profile)
for run in runs:
  result = scenario.run(profile, run_index)
emulator.clear()
```

Test matrix execution (`run_test_matrix`) iterates scenario/profile pairs and computes aggregated metrics with `MetricsCalculator`.

## Core Data Structures
- `clients/base.py`
  - `ChatMessage` with `to_dict()` for API payloads.
  - `StreamingResponse` for chunk timing (ttft/ttlt, inter-chunk times).
  - `ChatResponse` and `ImageResponse` for provider outputs and byte estimates.
- `scenarios/base.py`
  - `ScenarioResult` aggregates per-run totals and holds `log_records`.
- `analysis/logger.py`
  - `LogRecord` captures per-turn metrics including timing, bytes, and tool usage.
- `analysis/metrics.py`
  - `ScenarioMetrics` is the aggregated, 3GPP-aligned summary.

## Scenario Implementations
### Chat
`scenarios/chat.py`:
- Maintains `conversation_history` of `ChatMessage` objects.
- Non-streaming path calls `client.chat(...)`.
- Streaming path calls `client.chat_streaming(...)` and captures ttft/ttlt and inter-chunk times.
- Request/response sizes in streaming mode are estimated from message content length.

### Image Generation
`scenarios/image.py`:
- Calls `client.generate_image(...)`.
- Records response bytes from decoded image data when available.
- Captures latency and marks 501 for providers that do not implement images.

### Agent (MCP Tooling)
`scenarios/agent.py`:
- `MCPToolExecutor` reads `configs/mcp_servers.yaml`, resolves `${ENV_VAR}` placeholders, and connects to MCP servers.
- Tool definitions are exposed in OpenAI function-calling schema via `get_tools_for_openai()`.
- `_run_agent_turn()` loops up to `max_tool_calls`:
  - Calls `client.chat(..., tools=...)`.
  - Executes tool calls via MCP (stdio JSON-RPC).
  - Appends tool results back into the message list.
  - Logs each API call and tool execution as separate `LogRecord` entries.

### Direct Web Search (No MCP)
`scenarios/direct_search.py`:
- `DirectSearchClient` uses `requests` and supports Google and DuckDuckGo (HTML scraping).
- `ThreadedSearchExecutor` spins a `ThreadPoolExecutor` and gives each thread its own client for session safety.
- `DirectWebSearchScenario` logs a "search phase" with aggregated metrics, then optionally synthesizes results via an LLM.
- `ParallelSearchBenchmarkScenario` runs repeated searches across thread counts to measure throughput vs parallelism.

### Realtime (WebSocket)
`scenarios/realtime.py` uses `clients/realtime_client.py`:
- Establishes a WebSocket session (`session.created` -> `session.update` -> `session.updated`).
- `send_text()` and `send_audio()` measure raw bytes and per-chunk timing.
- Response events are processed until `response.done`; deltas update `RealtimeTurnMetrics`.
- Logs a connection record (`turn_index = -1`) and per-turn records with chunk timing.

## Provider Clients
### OpenAI
`clients/openai_client.py`:
- Uses `openai` SDK for chat and image generation.
- Non-streaming responses parse `usage` for token counts and tool calls.
- Streaming returns a generator or a `StreamingResponse`.

### DeepSeek
`clients/deepseek_client.py`:
- Uses OpenAI-compatible endpoint (`https://api.deepseek.com`).
- Same parsing logic as OpenAI for tool calls and usage.

### Gemini
`clients/gemini_client.py`:
- Converts `ChatMessage` into Gemini chat format.
- Handles system prompts by extracting the first system message.
- Streaming and non-streaming paths use `google-generativeai`.

### Realtime (OpenAI)
`clients/realtime_client.py`:
- Uses `websockets` to send JSON events and parse streaming deltas.
- Tracks bytes at the WebSocket message level and maintains per-turn and session metrics.

### MCP Client
`clients/mcp_client.py`:
- Spawns MCP servers via subprocess and speaks JSON-RPC over stdin/stdout.
- `tools/list` discovery populates tool metadata for function calling.
- `tools/call` results include request/response byte sizes per RPC.

## Network Emulation
`netem/controller.py`:
- Applies profiles via `tc` with `netem` and optional `htb` rate limiting.
- For rate-limited profiles, uses `htb` root qdisc with a `netem` leaf.
- `clear()` removes root qdisc and ignores missing-qdisc errors.

## Capture
`capture/controller.py`:
- L3/L4 capture via `tcpdump` (requires sudo).
`capture/l7_capture.py`:
- Generates a mitmproxy addon script that writes JSONL records per HTTP flow.
- Records headers, body sizes, and timing for request/response.

## Metrics and Reporting
`analysis/logger.py`:
- SQLite schema lives in `traffic_logs` table with indexes on scenario/profile, session, timestamp.
`analysis/metrics.py`:
- Computes latency stats, TTFT/TTLT tails (P50/P95/P99), UL/DL ratios, token rates, loop factor, stall rate/duration, error taxonomy, and burstiness descriptors.
- Percentiles are computed from sorted samples (custom percentile helper).
`analysis/visualize.py`:
- Optional plots if matplotlib/seaborn/pandas are installed.

## Implementation Notes
- Some byte counts are estimates based on JSON serialization or content length rather than raw wire size.
- Agent tool results are appended as user messages to continue tool loops; this is a pragmatic format for multi-step chains.
- `BaseAgentScenario.run()` uses the current event loop (`asyncio.get_event_loop()`), which can conflict if a loop is already running (e.g., notebooks).
- `NetworkEmulator` will attempt `sudo tc` and logs a warning on failure, but does not fall back to a non-netem mode.
- L7 capture relies on proxy environment variables; traffic that does not honor proxy settings will not be captured.

## Implementation Coverage
### Implemented
- End-to-end orchestration for running scenarios with network profiles, logging, and reporting via `orchestrator.py`.
- Chat scenarios with both streaming and non-streaming paths in `scenarios/chat.py`.
- Image generation via OpenAI DALL-E in `scenarios/image.py` + `clients/openai_client.py`.
- Agent scenarios with real MCP server execution in `scenarios/agent.py` and `clients/mcp_client.py` (OpenAI/DeepSeek providers).
- General agent scenario registered and runnable via `configs/scenarios.yaml`.
- Direct search scenarios using threaded HTTP requests and optional LLM synthesis in `scenarios/direct_search.py`.
- Multimodal image+text scenario in `scenarios/multimodal.py` using provider-specific APIs (Gemini).
- Realtime text/audio conversations over WebSocket in `scenarios/realtime.py` and `clients/realtime_client.py`.
- SQLite logging and 3GPP-aligned metric aggregation in `analysis/logger.py` and `analysis/metrics.py`.
- Optional plotting in `analysis/visualize.py` (when matplotlib/seaborn/pandas are installed).
- Network emulation with tc/netem and optional HTB rate limiting in `netem/controller.py`.
- MCP rate limiting and concurrency throttling via `scenarios/agent.py`.
- Capture start/stop hooks in `orchestrator.py` (pcap and L7, optional CLI flags).
- Streaming response byte tracking via `StreamingResponse` and provider clients.
- Token counting uses `tiktoken` when installed, with a fallback heuristic.

### Partial 
- Gemini tool calling is parsed, but tool schema conversion from OpenAI format is not implemented; callers must pass Gemini-compatible tools explicitly.
- Streaming token counts are typically unavailable because providers do not emit usage data in streaming mode by default.
- Multimodal scenarios require valid image paths; no sample assets are bundled in the repository.
- Realtime function-call events are captured as raw events but not assembled into structured calls or executed.
- Tool metrics are attached to MCP tool log records; aggregated tool bytes are not separated from overall request/response totals.
