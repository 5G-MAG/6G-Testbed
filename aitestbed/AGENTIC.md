# Plan: Adding OpenClaw and A2A Test Scenarios

This document describes how to add two new agentic traffic surfaces to the testbed:
an **OpenClaw** personal-assistant scenario and three **A2A (Agent2Agent protocol)**
scenarios (single task, streaming, multi-agent delegation).

## Context

The testbed today characterizes AI traffic across three scenario families: standard LLM
chat, MCP tool-calling agents, and custom-transport agents (realtime WebSocket/WebRTC,
computer-use). Two emerging agentic surfaces are not yet represented and are relevant to
the SA4 6G Media Study traffic characterization:

1. **OpenClaw** (https://openclaw.ai) is a local-first, self-hosted personal AI agent
   (npm/Node.js) that connects an LLM to browser/file/shell tools and chat channels. It
   runs a local **gateway on `127.0.0.1:18789`** and calls out to a cloud or local LLM. Its
   traffic shape is distinctive: a long-lived local control channel plus bursty LLM and
   tool egress over many agent-loop iterations. It captures the "personal agent on device"
   pattern.
2. **A2A (Agent2Agent protocol)** is the open JSON-RPC-2.0-over-HTTP(S) protocol for
   agent-to-agent interop (Agent Cards at `/.well-known/agent-card.json`, `message/send`,
   `message/stream` SSE, multi-agent delegation). It captures inter-agent traffic, a new
   class of 6G media/agentic traffic.

Both produce HTTP/JSON-RPC traffic that the existing tc/netem + tcpdump + SQLite + RAN2
pipeline can shape, capture, and analyze, so they fit the testbed cleanly. The goal is to
add **one OpenClaw scenario** and **three A2A scenarios**, each runnable across the SA4
S4-260848 profile matrix, with local (loopback-shaped) and remote variants for A2A.

## Background facts (from research)

**OpenClaw**

- Install: `npm install -g openclaw@latest`; daemon: `openclaw onboard --install-daemon`;
  status: `openclaw gateway status`.
- Gateway HTTP surface: `http://127.0.0.1:18789/` (dashboard + control). Config:
  `~/.openclaw/openclaw.json`. Env: `OPENCLAW_HOME`, `OPENCLAW_STATE_DIR`,
  `OPENCLAW_CONFIG_PATH`.
- The LLM is called *inside* the Node process (Claude/GPT/local), so it does **not** go
  through the testbed's `LLMClient`. The provider/key config fields and the exact headless
  "send one task" route are not in the public docs and **must be confirmed at setup time**
  (see Open Items).

**A2A**

- pip package: **`a2a-sdk`** (imports under `a2a.*`). Client surface (note version drift;
  pin a version):
  `from a2a.client import A2ACardResolver, ClientConfig, create_client`; resolve the card
  via `A2ACardResolver(httpx_client, base_url).get_agent_card()` (default path
  `/.well-known/agent-card.json`); `client = await create_client(agent=card,
  client_config=ClientConfig(streaming=False|True))`;
  `async for chunk in client.send_message(SendMessageRequest(message=new_text_message(...)))`.
- Server surface for the local agent: `A2AStarletteApplication`, `DefaultRequestHandler`,
  `AgentExecutor`, served via uvicorn on a loopback port.

## Extension patterns being reused (do not reinvent)

- **New-scenario wiring** (the common path): `scenarios/<name>.py` → export in
  `scenarios/__init__.py` (`__all__`) → import + register in `orchestrator.py`
  `scenario_classes` dict (`orchestrator.py:~28-54` imports, `~122-162` dict) → config entry
  in `configs/scenarios.yaml` + `test_matrix` → optional `configs/anonymization_map.json`
  provider/model/scenario aliases.
- `scenarios/base.py`, subclass `BaseScenario`; reuse `_create_session_id()` and
  `_create_log_record()` (`base.py:147-219`) for LogRecord creation, anonymization, and
  tracing.
- **OpenClaw template** = `scenarios/computer_use.py`: an in-process executor class
  (`ComputerUseExecutor`) that manages an external runtime and emits dual LogRecords (driver
  call + per-step). The LLM call is not via `LLMClient`; metrics come from the executor and
  pcap.
- **A2A template** = `scenarios/realtime.py` + `clients/realtime_client.py`: a custom async
  client (not `LLMClient`) with its own metrics dataclass
  (`RealtimeTurnMetrics`/`RealtimeSessionMetrics`), driven by `asyncio.run()`, emitting
  connection + per-turn LogRecords.
- Loopback shaping primitive already exists:
  `netemu.apply_profile_to_loopback(profile_name, dest_port)`
  (`netemu/src/netemu/emulator.py:~809-891`), used today for MCP-over-HTTP.

## Shared infrastructure change (one cross-cutting edit)

Today the orchestrator only starts the **secondary loopback pcap** and applies **loopback
netem** when `mcp_transport == "http"` (`orchestrator.py:~1006-1035`; loopback shaping in
`scenarios/agent.py setup()`). OpenClaw (port 18789) and local A2A agents (e.g. ports
9001-9003) need the same treatment without being MCP scenarios.

**Change:** generalize the gate to a scenario-config-driven hook.

- Add an optional `loopback_ports: [<int>...]` (and/or `uses_loopback: true`) field read
  from the scenario config in `orchestrator.py`.
- When present, (a) start the loopback tcpdump (reuse the existing secondary-capture code
  path) and (b) call `emulator.apply_profile_to_loopback(profile, port)` for each port. Keep
  the existing MCP `mcp_transport == "http"` behavior as a special case that resolves to the
  same hook (it already knows its dynamic MCP ports).
- This keeps capture/shaping behavior identical for MCP and adds it generically for
  OpenClaw/A2A-local. Egress traffic (cloud LLM / remote A2A) continues to be shaped by
  main-interface egress netem and captured by the primary tcpdump, no change.

## Part A, OpenClaw scenario (1 scenario)

**New files**

- `scenarios/openclaw_agent.py`:
  - `OpenClawExecutor` (modeled on `ComputerUseExecutor`): ensure the gateway daemon is
    running (`openclaw gateway status`; start via the configured command if not), then submit
    a single deterministic task to the local gateway (`127.0.0.1:18789`) via `httpx` POST
    (confirm the exact route at setup; CLI one-shot as fallback), and consume the
    response/stream. Tracks first-event time (TTFT), last-event time (TTLT), step/iteration
    count, tools/skills invoked, and control-channel request/response bytes.
  - `OpenClawScenario(BaseScenario)`: `scenario_type == "openclaw_agent"`; `run()` wraps an
    async drive (or sync subprocess), runs the configured `prompts` as tasks, aggregates into
    `ScenarioResult` (turn_count, tool_calls_count, total bytes, ttft/ttlt, latency), and
    emits LogRecords via `_create_log_record()` with metadata
    `{record_type, step_index, tools_used, channel, gateway_port}`.
- LLM provider/key wired through env (reuse existing `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`)
  by writing/merging `~/.openclaw/openclaw.json` or env at setup.

**Traffic model:** loopback control channel on `18789` (shaped + captured via the shared
hook) plus cloud LLM/tool egress on the main interface (existing egress netem + primary
pcap). `network_interface: auto`, `loopback_ports: [18789]`.

**Config:** add `openclaw_personal_assistant` to `configs/scenarios.yaml` with
`type: openclaw_agent`, provider/model, a few deterministic task `prompts` (e.g. summarize a
local file, draft a calendar note), `max_steps`, and `loopback_ports: [18789]`. Add to
`test_matrix` under a new `openclaw` phase. Add aliases to `anonymization_map.json` (provider
"OpenClaw", scenario label).

**Lifecycle:** `run_full_tests.sh` auto-manages OpenClaw like vLLM. When the `openclaw`
phase is explicitly enabled (`--enable openclaw`) and `MANAGE_OPENCLAW=true` (default), the
runner installs the CLI if missing (`OPENCLAW_AUTO_INSTALL=true`), starts the gateway, polls
it for readiness, and stops it on exit via the `cleanup_on_exit` trap (process-group kill +
`OPENCLAW_STOP_CMD`). An already-running gateway is reused and left untouched. Start/stop
commands are overridable via `OPENCLAW_START_CMD` / `OPENCLAW_STOP_CMD` since they are
version dependent (see Open items).

## Part B, A2A scenarios (3 scenarios + local agents + client)

**New files**

- `clients/a2a_client.py`, a thin async wrapper over `a2a-sdk` with byte/timing
  instrumentation:
  - `A2ATurnMetrics` / `A2ASessionMetrics` dataclasses (mirroring `RealtimeTurnMetrics`):
    `t_request_start`, `t_first_chunk`, `t_last_chunk`, `request_bytes`, `response_bytes`,
    `chunk_count`, `inter_chunk_times`, and `ttft`/`ttlt` properties.
  - Methods: `resolve_card(base_url)`, `send(message, streaming=False)` yielding chunks while
    accumulating metrics. Instrument bytes via an `httpx` event hook / response-size
    accounting so per-turn UL/DL is captured even when egress traffic is TLS.
- `a2a_agents/` (new package, sibling to `mcp_servers/`):
  - `echo_agent.py` (or `summarize_agent.py`): a minimal `AgentExecutor` +
    `DefaultRequestHandler` + `A2AStarletteApplication`, served by uvicorn on a configurable
    loopback port; supports both `message/send` and `message/stream`.
  - `orchestrator_agent.py`: an A2A agent that is itself an A2A **client** to two or more
    downstream `echo_agent` instances (for the multi-agent delegation scenario).
  - `__main__`/launch helper so the scenario can spawn them as subprocesses (modeled on the
    `mcp_servers/http_bridge.py` / `weather_server.py` subprocess + port-discovery pattern).
- `scenarios/a2a_agent.py`, three classes subclassing `BaseScenario`, each driven by
  `asyncio.run()` like `realtime.py`:
  - `A2ASingleTaskScenario` (`a2a_single_task`): non-streaming `message/send`; one LogRecord
    per task turn.
  - `A2AStreamingScenario` (`a2a_streaming`): `message/stream` SSE; `is_streaming=True`,
    `chunk_count`, `inter_chunk_times`, stall metrics like streaming chat.
  - `A2AMultiAgentScenario` (`a2a_multi_agent`): drives the orchestrator agent, which
    delegates to downstream agents; emits per-hop LogRecords
    (`metadata.record_type = "a2a_delegation"`, `agent_url`) so RAN2 sub-flow byte accounting
    works.

**Local vs remote variants (both).**

- Local: the scenario config launches the `a2a_agents` servers on loopback ports (e.g.
  `9001`, `9002`, `9003`), sets `loopback_ports: [9001,9002,9003]` → the shared hook shapes
  and captures them. Fully reproducible, no keys.
- Remote: separate scenario config entries with `agent_card_url: <public A2A agent>` and no
  local launch; shaped by main-interface egress netem, captured by the primary pcap. Mark
  remote entries as opt-in (skipped if the endpoint is unset), mirroring how key-gated phases
  are skipped.

**Config:** add the three local entries (plus remote variants) to `configs/scenarios.yaml`,
and a new `a2a` phase in `test_matrix`. Add `a2a-sdk` to `requirements.txt`. Add aliases to
`anonymization_map.json`.

### Remote A2A evaluation (self-host + tunnel)

There are effectively **no usable free public A2A agents**: community registries
(`a2aregistry.org`, `a2aagentlist.com`) list mostly dead/auth-gated/pay-per-call endpoints,
the official samples are local-run only, and the one open public agent
(`https://hello-world-gxfr.onrender.com`) is a flaky free-tier echo that serves the legacy
`/.well-known/agent.json` path and **rejects current `a2a-sdk` 1.1.0 payloads**
(`InvalidRequestError`). So remote evaluation means **self-hosting our own agent** and
exposing it over the internet.

Recipe (cloudflared quick tunnel, no account):

```bash
# 1. Expose a local port; note the printed https URL (random per run)
cloudflared tunnel --url http://localhost:9001     # -> https://<rand>.trycloudflare.com
#    (or: ngrok http 9001)

# 2. Start our agent so its Agent Card advertises the *tunnel* URL (critical:
#    otherwise the card advertises localhost and the remote client can't dial it)
python -m a2a_agents.echo_agent --port 9001 --public-url https://<rand>.trycloudflare.com

# 3. Point the remote scenario at it and run (real internet RTT + TLS + egress netem)
export A2A_REMOTE_AGENT_URL=https://<rand>.trycloudflare.com
python orchestrator.py --scenario a2a_single_task_remote --profile 5g_urban --runs 5
#    or: bash run_full_tests.sh --enable a2a            # remote entries gated on the env var
```

Supporting features:
- **`--public-url`** on `a2a_agents/echo_agent.py` and `orchestrator_agent.py` →
  `build_agent_card(public_url=...)` makes the card advertise the externally-reachable URL
  (defaults to `http://host:port/`).
- **`agent_card_path`** (scenario config) → `A2AClientSession(agent_card_path=...)` →
  `create_client(relative_card_path=...)` resolves a non-default card path (e.g. legacy
  `/.well-known/agent.json`) for client-conformance smoke tests.
- **`request_timeout_sec`** (scenario config, default 60s) → the client's httpx timeout, so
  cold-starting free-tier hosts and high-latency/lossy profiles (satellite_geo, congested)
  don't blow past httpx's 5s default during card fetch / message turns.

For a stable URL (instead of a per-run quick tunnel), host the agent on a cheap VM
(Fly.io / Hetzner / Cloud Run) and set `A2A_REMOTE_AGENT_URL` to its public address.

## Common wiring checklist (both parts)

For each new scenario class:

1. `scenarios/__init__.py`, import the class, add it to `__all__`.
2. `orchestrator.py`, add to the import block (`~28-54`) and the `scenario_classes` dict
   (`~122-162`); the dict key MUST equal the `scenario_type` string and the `type:` in YAML.
3. `configs/scenarios.yaml`, scenario entry + `test_matrix` phase (`openclaw`, `a2a`).
4. `configs/anonymization_map.json`, provider/model/scenario aliases (so the anonymized SA4
   DB is correct).
5. `run_full_tests.sh`, add `openclaw` and `a2a` phases with prereq checks (Node +
   `openclaw` on PATH; `python -c "import a2a"`; relevant provider key), wired into
   `--enable/--disable`. Update the phase-name list.
6. `README.md`, add the scenarios to the scenario tables, the phase list, and prereqs
   (Node/openclaw, `a2a-sdk`). The `netemu`/root README only changes if profile/interface
   behavior changes (it does not).

## Verification

1. **Import smoke test:** `python -c "import scenarios; import clients.a2a_client; import a2a"`;
   `python orchestrator.py --list-scenarios` shows `openclaw_personal_assistant`,
   `a2a_single_task`, `a2a_streaming`, `a2a_multi_agent`.
2. **A2A local, no network emulation:** start the local agents, run
   `python orchestrator.py --scenario a2a_single_task --profile no_emulation --runs 2`;
   confirm rows in `logs/traffic_logs.db` with non-zero request/response bytes, ttft/ttlt
   set, and `success=1`. Repeat for `a2a_streaming` (chunk_count > 1, inter_chunk_times
   populated) and `a2a_multi_agent` (delegation sub-flow records present).
3. **Loopback shaping/capture hook:** run `a2a_single_task` under `--profile cell_edge`;
   confirm latency rises per profile and a loopback pcap exists in `results/captures/` for
   the agent ports; verify with the pcap analysis (`netemu.pcap`) / the RAN2 output that
   per-direction bytes are attributed.
4. **OpenClaw:** with `openclaw` installed and a provider key set,
   `python orchestrator.py --scenario openclaw_personal_assistant --profile 5g_urban --runs 1`;
   confirm the gateway is reached on 18789 (loopback pcap non-empty), egress LLM traffic is
   captured on the main interface, and step/tool counts are populated in metadata.
5. **End-to-end matrix slice:** `bash run_full_tests.sh --quick --enable a2a` and
   `--enable openclaw`; confirm `RESULTS.md`/charts/Excel regenerate with the new scenarios
   and the run summary shows no failures.

## Open items to confirm during implementation

- **OpenClaw headless drive (RESOLVED for openclaw 2026.6.x):** drive is the CLI
  `openclaw agent --json --session-id <id> -m "<prompt>"`, which runs one agent turn via the
  gateway and returns JSON (`result.payloads[].text`, `meta.agentMeta.usage`). The gateway is
  started with `openclaw gateway run --allow-unconfigured --auth none --port 18789 --force`,
  and `gateway.auth.mode=none` is persisted via `openclaw config set` so the agent CLI can
  open the gateway WebSocket. The model uses the provider configured in OpenClaw (default
  `openai/gpt-5.5`, so `OPENAI_API_KEY` must be in the environment). `run_full_tests.sh`
  performs this whole setup automatically (`--enable openclaw`). Requires Node ≥ 22 (the
  runner auto-activates nvm if present). Override the drive via the scenario's
  `drive`/`cli_args` and the lifecycle via `OPENCLAW_START_CMD`/`OPENCLAW_STOP_CMD`.
- **a2a-sdk version pinning:** the client symbols differ across versions
  (`create_client`/`A2ACardResolver` vs older `A2AClient.get_client_from_agent_card_url`) and
  the card path (`/.well-known/agent-card.json` vs `/.well-known/agent.json`). Pin an explicit
  `a2a-sdk==<version>` in `requirements.txt` and write code to that surface.
- **Per-turn byte accounting under TLS egress (remote A2A):** prefer pcap-derived bytes for
  remote; local/loopback can use both client-side counters and pcap.
