"""A2A (Agent2Agent protocol) scenarios for the 6G AI Traffic Testbed.

Three interaction patterns over the A2A JSON-RPC protocol:

  - ``A2ASingleTaskScenario``  non-streaming ``message/send``
  - ``A2AStreamingScenario``   streaming ``message/stream`` (SSE)
  - ``A2AMultiAgentScenario``  an orchestrator agent that delegates to ≥2
                               downstream agents (multi-hop agent-to-agent)

Each scenario can run against **local** agents (launched as subprocesses on
loopback ports and shaped/captured like MCP-over-HTTP) or a **remote** agent
(``agent_card_url`` / ``agent_card_url_env`` in config; shaped by main-iface
egress netem). Driven by ``asyncio.run`` and the instrumented
``clients.a2a_client.A2AClientSession``; the a2a-sdk import is deferred so
importing this module never requires the SDK.

Config keys (configs/scenarios.yaml):
  provider: "a2a"            # uses NullLLMClient, no API key needed for local
  model: "<label>"          # anonymised label only
  streaming: true|false
  prompts: [ ... ]
  # Local mode:
  loopback_ports: [9001...]   # ports to capture + shape (target is shaped)
  target_port: 9001             # the agent the client talks to
  agents:                       # subprocesses to launch
    - module: "a2a_agents.echo_agent"
      port: 9001
      args: ["--behavior", "echo"]
  # Remote mode (overrides local):
  agent_card_url: "https://..."     # or
  agent_card_url_env: "A2A_REMOTE_AGENT_URL"
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Optional

from .base import BaseScenario, ScenarioResult
from clients.base import LLMClient
from analysis.logger import TrafficLogger, LogRecord
from clients.a2a_client import A2AClientSession, A2ATurnMetrics


class _A2ABaseScenario(BaseScenario):
    """Shared driver for the A2A scenarios."""

    # Subclasses override these two.
    _scenario_type = "a2a_single_task"
    _default_streaming = False

    def __init__(self, client: LLMClient, logger: TrafficLogger, config: dict):
        super().__init__(client, logger, config)
        self.streaming = bool(config.get("streaming", self._default_streaming))
        # Injected by the orchestrator before run() (see run_experiment) so we
        # can shape the loopback hop for local agents.
        self.emulator = None
        self._current_network_profile: Optional[str] = None
        self._netem_on_lo = False
        self._fleet = None

    @property
    def scenario_type(self) -> str:
        return self._scenario_type

    # ------------------------------------------------------------------
    # Target resolution: remote URL or local launched fleet
    # ------------------------------------------------------------------
    def _remote_url(self) -> Optional[str]:
        url = self.config.get("agent_card_url")
        if not url:
            env = self.config.get("agent_card_url_env")
            if env:
                url = os.environ.get(env)
        return url or None

    def _start_local_fleet(self) -> str:
        """Launch configured local agents and return the target base URL."""
        from a2a_agents.launcher import A2AAgentProcess, A2AAgentFleet

        specs = self.config.get("agents", [])
        if not specs:
            raise ValueError(
                f"{self.scenario_id}: local A2A scenario needs an 'agents' list "
                f"or an 'agent_card_url'"
            )
        procs = [
            A2AAgentProcess(
                module=s["module"],
                args=[str(a) for a in s.get("args", [])],
                port=int(s["port"]),
                host=s.get("host", "127.0.0.1"))
            for s in specs
        ]
        self._fleet = A2AAgentFleet(procs)
        self._fleet.start_all(timeout=float(self.config.get("agent_ready_timeout", 20.0)))

        target_port = self.config.get("target_port") or specs[-1]["port"]
        return f"http://127.0.0.1:{int(target_port)}"

    def _apply_loopback_shaping(self) -> None:
        """Shape the client-facing target port on lo (mirrors agent.py).

        netemu's apply_profile_to_loopback resets the lo root each call, so we
        shape only the target (client-facing) port. Downstream inter-agent hops
        in the multi-agent scenario are still captured in the loopback pcap.
        """
        port = self.config.get("target_port")
        if not port:
            ports = self.config.get("loopback_ports") or []
            port = ports[0] if ports else None
        if (
            port
            and self.emulator is not None
            and self._current_network_profile
            and self._current_network_profile != "no_emulation"
        ):
            try:
                self.emulator.apply_profile_to_loopback(
                    self._current_network_profile, int(port)
                )
                self._netem_on_lo = True
            except Exception as e:  # noqa: BLE001
                print(f"  Warning: failed to shape lo:{port} for A2A: {e}")

    def _teardown(self) -> None:
        if self._netem_on_lo and self.emulator is not None:
            try:
                self.emulator.clear_loopback()
            except Exception:
                pass
            self._netem_on_lo = False
        if self._fleet is not None:
            self._fleet.stop_all()
            self._fleet = None

    # ------------------------------------------------------------------
    def run(self, network_profile: str, run_index: int = 0) -> ScenarioResult:
        return asyncio.run(self._run_async(network_profile, run_index))

    async def _run_async(self, network_profile: str, run_index: int = 0) -> ScenarioResult:
        session_id = self._create_session_id()
        prompts = self.config.get("prompts", ["Hello from the testbed."])
        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index)

        remote_url = self._remote_url()
        try:
            if remote_url:
                base_url = remote_url
            else:
                base_url = self._start_local_fleet()
                self._apply_loopback_shaping()

            result.metadata = {
                "protocol": "a2a",
                "mode": "remote" if remote_url else "local",
                "streaming": self.streaming,
                "target": base_url,
            }
            if not remote_url:
                result.metadata["agents"] = [
                    {"module": s["module"], "port": s["port"]}
                    for s in self.config.get("agents", [])
                ]

            async with A2AClientSession(
                base_url,
                streaming=self.streaming,
                agent_card_path=self.config.get("agent_card_path"),
                timeout=float(self.config.get("request_timeout_sec", 60.0))) as sess:
                for turn_index, prompt in enumerate(prompts):
                    await self._wait_between_prompts_async(turn_index)
                    turn = await sess.send(prompt)
                    record = self._turn_record(
                        session_id, turn_index, run_index, network_profile,
                        prompt, turn, base_url)
                    self.logger.log(record)
                    result.log_records.append(record)
                    self._accumulate(result, turn, record)
                    if not turn.success:
                        result.success = False
                        result.error_message = turn.error_message
                        break

        except Exception as e:  # noqa: BLE001
            result.success = False
            result.error_message = f"{type(e).__name__}: {e}"
        finally:
            self._teardown()

        return result

    def _accumulate(
        self, result: ScenarioResult, turn: A2ATurnMetrics, record: LogRecord
    ) -> None:
        result.turn_count += 1
        result.api_call_count += 1
        result.total_latency_sec += turn.total_latency
        result.total_request_bytes += turn.request_bytes
        result.total_response_bytes += turn.response_bytes
        result.total_tokens_in += record.tokens_in or 0
        result.total_tokens_out += record.tokens_out or 0
        if result.ttft_sec is None and turn.ttft is not None:
            result.ttft_sec = turn.ttft
        if turn.ttlt is not None:
            result.ttlt_sec = turn.ttlt

    def _turn_record(
        self,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        prompt: str,
        turn: A2ATurnMetrics,
        base_url: str) -> LogRecord:
        model = self.config.get("model", "")
        tokens_in = self.client.estimate_tokens(prompt, model) if prompt else None
        tokens_out = (
            self.client.estimate_tokens(turn.output_text, model)
            if turn.output_text
            else None
        )
        metadata = {
            "protocol": "a2a",
            "record_type": "a2a_delegation" if self._is_multi_agent() else "a2a_task",
            "agent_url": base_url,
            "streaming": turn.streaming,
            "event_count": turn.event_count,
            "output_chars": len(turn.output_text),
        }
        if self._is_multi_agent():
            metadata["downstream_agents"] = [
                f"http://127.0.0.1:{s['port']}"
                for s in self.config.get("agents", [])
                if "orchestrator" not in s.get("module", "")
            ]
        return self._create_log_record(
            session_id=session_id,
            turn_index=turn_index,
            run_index=run_index,
            network_profile=network_profile,
            request_bytes=turn.request_bytes,
            response_bytes=turn.response_bytes,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            t_request_start=turn.t_request_start,
            t_first_token=turn.t_first_chunk,
            t_last_token=turn.t_last_chunk,
            latency_sec=turn.total_latency,
            http_status=200 if turn.success else 0,
            error_type=turn.error_message,
            success=turn.success,
            is_streaming=turn.streaming,
            chunk_count=turn.chunk_count,
            inter_chunk_times=json.dumps(turn.inter_chunk_times),
            metadata=json.dumps(metadata))

    def _is_multi_agent(self) -> bool:
        return False


class A2ASingleTaskScenario(_A2ABaseScenario):
    """Non-streaming single task: one message/send, one aggregated result."""

    _scenario_type = "a2a_single_task"
    _default_streaming = False


class A2AStreamingScenario(_A2ABaseScenario):
    """Streaming task: message/stream (SSE) with incremental artifact events."""

    _scenario_type = "a2a_streaming"
    _default_streaming = True


class A2AMultiAgentScenario(_A2ABaseScenario):
    """Multi-agent delegation: client → orchestrator agent → downstream agents."""

    _scenario_type = "a2a_multi_agent"
    _default_streaming = False

    def _is_multi_agent(self) -> bool:
        return True
