"""OpenClaw personal-assistant scenario for the 6G AI Traffic Testbed.

OpenClaw (https://openclaw.ai) is a local-first, self-hosted personal AI agent
(Node.js) that connects an LLM to browser/file/shell tools and chat channels.
It runs a local gateway on ``127.0.0.1:18789`` and calls a cloud or local LLM
*inside its own process*, so it does not go through the testbed's LLMClient.

This scenario drives a single deterministic task through OpenClaw and measures
the resulting traffic. Two surfaces are captured/shaped:

  - the loopback control channel to the gateway (port 18789), shaped + captured
    like MCP-over-HTTP via the shared loopback hook (``loopback_ports``);
  - the LLM/tool egress OpenClaw itself makes, shaped by the main-interface
    egress netem and captured by the primary pcap.

The exact gateway HTTP route and provider/key config keys are version
dependent (see AGENTIC.md "Open items"). The drive mechanism is therefore
configurable: ``drive: http`` POSTs to the gateway, ``drive: cli`` shells out
to the ``openclaw`` binary. When the gateway/binary is unavailable the run is
recorded as a failed measurement rather than crashing the suite.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .base import BaseScenario, ScenarioResult
from clients.base import LLMClient
from analysis.logger import TrafficLogger, LogRecord

DEFAULT_GATEWAY = "http://127.0.0.1:18789"
# Candidate gateway message routes tried in order when not pinned in config.
DEFAULT_ENDPOINT_CANDIDATES = ["/api/message", "/api/v1/messages", "/message", "/chat"]


@dataclass
class OpenClawTaskResult:
    """Metrics from driving one OpenClaw task."""

    t_request_start: float
    t_first_byte: Optional[float] = None
    t_last_byte: Optional[float] = None
    request_bytes: int = 0
    response_bytes: int = 0
    steps: int = 0
    tools_used: list = field(default_factory=list)
    output_text: str = ""
    channel: str = "http"
    endpoint: str = ""
    success: bool = True
    error_message: Optional[str] = None

    @property
    def latency(self) -> float:
        end = self.t_last_byte or time.time()
        return end - self.t_request_start


def _parse_agent_steps(payload) -> tuple[int, list]:
    """Best-effort extraction of step count and tool names from a response."""
    steps = 0
    tools: list = []
    if isinstance(payload, dict):
        for key in ("steps", "iterations", "turns"):
            v = payload.get(key)
            if isinstance(v, int):
                steps = max(steps, v)
            elif isinstance(v, list):
                steps = max(steps, len(v))
        for key in ("tool_calls", "tools", "toolCalls", "actions"):
            v = payload.get(key)
            if isinstance(v, list):
                for item in v:
                    name = (
                        item.get("name") or item.get("tool") or item.get("type")
                        if isinstance(item, dict) else str(item)
                    )
                    if name:
                        tools.append(name)
        msgs = payload.get("messages")
        if isinstance(msgs, list):
            steps = max(steps, len(msgs))
    return steps, tools


def _extract_text(payload) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        # openclaw `agent --json`: result.payloads[].text
        result = payload.get("result")
        if isinstance(result, dict):
            payloads = result.get("payloads")
            if isinstance(payloads, list):
                texts = [
                    p.get("text")
                    for p in payloads
                    if isinstance(p, dict) and p.get("text")
                ]
                if texts:
                    return " ".join(texts)
        for key in ("reply", "response", "text", "content", "message", "output"):
            v = payload.get(key)
            if isinstance(v, str):
                return v
        return json.dumps(payload)[:2000]
    return str(payload)[:2000]


class OpenClawExecutor:
    """Drive a single OpenClaw task over the gateway HTTP API or the CLI."""

    def __init__(self, config: dict):
        self.gateway_url = config.get("gateway_url", DEFAULT_GATEWAY).rstrip("/")
        self.drive = config.get("drive", "http")
        self.endpoint = config.get("gateway_endpoint")
        self.endpoints = (
            [self.endpoint] if self.endpoint else list(DEFAULT_ENDPOINT_CANDIDATES)
        )
        self.request_field = config.get("request_field", "message")
        self.extra_payload = config.get("request_extra", {}) or {}
        self.cli_bin = config.get("cli_bin", "openclaw")
        self.cli_args = [str(a) for a in config.get("cli_args", ["run"])]
        self.timeout = float(config.get("task_timeout_sec", 120.0))

    # -- readiness -------------------------------------------------------
    def ensure_ready(self) -> None:
        # The CLI drive needs the binary; both drives talk to the gateway.
        if self.drive == "cli" and shutil.which(self.cli_bin) is None:
            raise RuntimeError(
                f"OpenClaw CLI '{self.cli_bin}' not found on PATH. "
                f"Install with: npm install -g openclaw@latest (needs Node >= 22)"
            )
        try:
            httpx.get(self.gateway_url + "/", timeout=3.0)
        except Exception as exc:
            raise RuntimeError(
                f"OpenClaw gateway not reachable at {self.gateway_url} "
                f"({type(exc).__name__}). Start it with: openclaw gateway run "
                f"--allow-unconfigured --port 18789  (run_full_tests.sh does this "
                f"automatically with --enable openclaw)"
            ) from exc

    # -- drive -----------------------------------------------------------
    def run_task(self, prompt: str) -> OpenClawTaskResult:
        if self.drive == "cli":
            return self._run_cli(prompt)
        return self._run_http(prompt)

    def _run_http(self, prompt: str) -> OpenClawTaskResult:
        payload = {self.request_field: prompt, **self.extra_payload}
        body = json.dumps(payload).encode("utf-8")
        res = OpenClawTaskResult(t_request_start=time.time(), channel="http")
        last_error = None
        for ep in self.endpoints:
            url = self.gateway_url + ep
            res.endpoint = ep
            res.t_request_start = time.time()
            try:
                r = httpx.post(
                    url,
                    content=body,
                    headers={"content-type": "application/json"},
                    timeout=self.timeout)
                res.t_first_byte = time.time()
                res.t_last_byte = time.time()
                res.request_bytes = len(body) + 64  # body + rough header estimate
                res.response_bytes = len(r.content)
                if r.status_code >= 400:
                    last_error = f"HTTP {r.status_code} at {ep}"
                    continue
                try:
                    parsed = r.json()
                except Exception:
                    parsed = r.text
                res.output_text = _extract_text(parsed)
                res.steps, res.tools_used = _parse_agent_steps(parsed)
                res.success = True
                return res
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                continue
        res.success = False
        res.error_message = (
            f"No OpenClaw gateway endpoint accepted the task "
            f"(tried {self.endpoints}): {last_error}. "
            f"Pin 'gateway_endpoint' in scenario config."
        )
        return res

    def _run_cli(self, prompt: str) -> OpenClawTaskResult:
        res = OpenClawTaskResult(t_request_start=time.time(), channel="cli")
        # Fresh session per task so history/context doesn't accumulate across
        # runs (which would inflate token/byte measurements). cli_args may use
        # {session} and {prompt} placeholders; if {prompt} is absent the prompt
        # is appended as a trailing positional (back-compat).
        session = f"testbed-{uuid.uuid4().hex[:8]}"
        args = []
        has_prompt = False
        for a in self.cli_args:
            a = a.replace("{session}", session)
            if "{prompt}" in a:
                a = a.replace("{prompt}", prompt)
                has_prompt = True
            args.append(a)
        cmd = [self.cli_bin, *args]
        if not has_prompt:
            cmd.append(prompt)
        res.request_bytes = len(" ".join(cmd).encode("utf-8"))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )
            res.t_first_byte = time.time()
            res.t_last_byte = time.time()
            out = proc.stdout or ""
            res.response_bytes = len(out.encode("utf-8"))
            res.output_text = out[:4000]
            res.success = proc.returncode == 0
            # Parse openclaw `agent --json` output when present.
            try:
                parsed = json.loads(out)
                res.output_text = _extract_text(parsed)
                res.steps, res.tools_used = _parse_agent_steps(parsed)
                status = parsed.get("status") if isinstance(parsed, dict) else None
                if status and status != "ok":
                    res.success = False
                    res.error_message = _extract_text(parsed)[:500]
            except Exception:
                pass
            if not res.success and not res.error_message:
                res.error_message = (proc.stderr or out or "openclaw CLI failed")[:500]
        except Exception as exc:  # noqa: BLE001
            res.success = False
            res.error_message = f"{type(exc).__name__}: {exc}"
        return res


class OpenClawScenario(BaseScenario):
    """Drive OpenClaw tasks and measure gateway + egress traffic."""

    def __init__(self, client: LLMClient, logger: TrafficLogger, config: dict):
        super().__init__(client, logger, config)
        self.executor = OpenClawExecutor(config)
        # Injected by the orchestrator so we can shape the gateway loopback port.
        self.emulator = None
        self._current_network_profile: Optional[str] = None
        self._netem_on_lo = False

    @property
    def scenario_type(self) -> str:
        return "openclaw_agent"

    def _gateway_port(self) -> Optional[int]:
        ports = self.config.get("loopback_ports") or []
        if ports:
            return int(ports[0])
        try:
            return int(self.executor.gateway_url.rsplit(":", 1)[1].split("/")[0])
        except Exception:
            return 18789

    def _apply_loopback_shaping(self) -> None:
        # Both drives reach the gateway over loopback (the CLI `openclaw agent`
        # connects to the gateway too), so shape the gateway port regardless.
        port = self._gateway_port()
        if (
            port
            and self.emulator is not None
            and self._current_network_profile
            and self._current_network_profile != "no_emulation"
        ):
            try:
                self.emulator.apply_profile_to_loopback(
                    self._current_network_profile, port
                )
                self._netem_on_lo = True
            except Exception as e:  # noqa: BLE001
                print(f"  Warning: failed to shape lo:{port} for OpenClaw: {e}")

    def _teardown(self) -> None:
        if self._netem_on_lo and self.emulator is not None:
            try:
                self.emulator.clear_loopback()
            except Exception:
                pass
            self._netem_on_lo = False

    def run(self, network_profile: str, run_index: int = 0) -> ScenarioResult:
        session_id = self._create_session_id()
        prompts = self.config.get("prompts", ["Summarize today's tasks."])
        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index)
        result.metadata = {"agent": "openclaw", "drive": self.executor.drive}

        try:
            self.executor.ensure_ready()
            self._apply_loopback_shaping()

            for turn_index, prompt in enumerate(prompts):
                self._wait_between_prompts(turn_index)
                task = self.executor.run_task(prompt)
                record = self._task_record(
                    session_id, turn_index, run_index, network_profile, prompt, task
                )
                self.logger.log(record)
                result.log_records.append(record)

                result.turn_count += 1
                result.api_call_count += 1
                result.tool_calls_count += len(task.tools_used)
                result.total_latency_sec += task.latency
                result.total_request_bytes += task.request_bytes
                result.total_response_bytes += task.response_bytes
                result.total_tokens_in += record.tokens_in or 0
                result.total_tokens_out += record.tokens_out or 0
                if result.ttft_sec is None and task.t_first_byte is not None:
                    result.ttft_sec = task.t_first_byte - task.t_request_start
                if task.t_last_byte is not None:
                    result.ttlt_sec = task.t_last_byte - task.t_request_start

                if not task.success:
                    result.success = False
                    result.error_message = task.error_message
                    break

        except Exception as e:  # noqa: BLE001
            result.success = False
            result.error_message = f"{type(e).__name__}: {e}"
        finally:
            self._teardown()

        return result

    def _task_record(
        self,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        prompt: str,
        task: OpenClawTaskResult) -> LogRecord:
        model = self.config.get("model", "")
        tokens_in = self.client.estimate_tokens(prompt, model) if prompt else None
        tokens_out = (
            self.client.estimate_tokens(task.output_text, model)
            if task.output_text
            else None
        )
        metadata = {
            "agent": "openclaw",
            "record_type": "openclaw_task",
            "channel": task.channel,
            "endpoint": task.endpoint,
            "gateway_port": self._gateway_port(),
            "steps": task.steps,
            "tools_used": task.tools_used,
            "output_chars": len(task.output_text),
        }
        return self._create_log_record(
            session_id=session_id,
            turn_index=turn_index,
            run_index=run_index,
            network_profile=network_profile,
            request_bytes=task.request_bytes,
            response_bytes=task.response_bytes,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            t_request_start=task.t_request_start,
            t_first_token=task.t_first_byte,
            t_last_token=task.t_last_byte,
            latency_sec=task.latency,
            http_status=200 if task.success else 0,
            error_type=task.error_message,
            success=task.success,
            tool_calls_count=len(task.tools_used),
            metadata=json.dumps(metadata))
