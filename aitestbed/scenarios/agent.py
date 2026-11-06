"""
Agent Scenarios for the 6G AI Traffic Testbed.

Implements agentic AI patterns with real MCP tool execution.
"""

import asyncio
import json
import logging
import os
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

import yaml

from .base import BaseScenario, ScenarioResult
from clients.base import ChatMessage, MessageRole, ChatResponse
from clients.mcp_client import (
    MCPClient,
    MCPServerConfig,
    MCPToolResult,
)
from analysis.logger import LogRecord


def load_mcp_servers_config(config_path: str = None) -> dict:
    """Load MCP server configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "mcp_servers.yaml"

    with open(config_path) as f:
        return yaml.safe_load(f)


# Module-level shared rate state: persists across MCPToolExecutor instances
# within the same Python process (i.e., one orchestrator invocation).
# This ensures Spotify's global rate limit budget is tracked across experiments.
_shared_rate_state: dict[str, deque] = {}
_shared_rate_state_day: dict[str, deque] = {}
_shared_rate_locks: dict[str, asyncio.Lock] = {}


class MCPToolExecutor:
    """
    Real MCP tool executor using connected MCP servers.

    Manages MCP server lifecycle and provides tool execution.
    """

    def __init__(self, server_group: str = "general", transport: str = "stdio"):
        self.mcp_client = MCPClient()
        self.server_group = server_group
        self.transport = transport
        self.config: dict = {}
        self.connected = False
        self.http_ports: dict[str, int] = {}  # server_name → port (HTTP mode)
        self._tool_stats: dict[str, dict] = {}
        self._rate_limits: dict[str, dict] = {}
        self._rate_groups: dict[str, dict] = {}
        # Use module-level shared state so rate budgets survive across experiments
        self._rate_state = _shared_rate_state
        self._rate_state_day = _shared_rate_state_day
        self._rate_locks = _shared_rate_locks
        self._concurrency_limits: dict[str, asyncio.Semaphore] = {}

    async def connect(self, config_path: str = None):
        """Connect to MCP servers defined in configuration."""
        self.config = load_mcp_servers_config(config_path)
        self._rate_limits = self.config.get("rateLimits", {})
        self._rate_groups = self.config.get("rateGroups", {})

        # Initialize rate state for groups (shared budget across tools)
        for group_name in self._rate_groups:
            self._rate_state.setdefault(group_name, deque())
            self._rate_state_day.setdefault(group_name, deque())
            # Always create fresh locks (asyncio.Lock is event-loop bound)
            self._rate_locks[group_name] = asyncio.Lock()

        for tool_name, limits in self._rate_limits.items():
            # Only init per-tool state for tools without a rate_group
            if "rate_group" not in limits:
                self._rate_state.setdefault(tool_name, deque())
                self._rate_state_day.setdefault(tool_name, deque())
                self._rate_locks[tool_name] = asyncio.Lock()
            concurrent_max = limits.get("concurrent_max")
            if concurrent_max:
                self._concurrency_limits[tool_name] = asyncio.Semaphore(concurrent_max)

        # Get servers for this group
        server_names = self.config.get("serverGroups", {}).get(
            self.server_group,
            list(self.config.get("mcpServers", {}).keys())
        )

        # Connect to each server
        for server_name in server_names:
            server_config = self.config.get("mcpServers", {}).get(server_name)
            if not server_config:
                continue

            # Resolve environment variables
            env = {}
            for key, value in server_config.get("env", {}).items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env[key] = os.environ.get(env_var, "")
                else:
                    env[key] = value

            # Only use HTTP transport for servers that support it
            server_transport = self.transport
            if self.transport == "http" and not server_config.get("supports_http", False):
                server_transport = "stdio"

            mcp_config = MCPServerConfig(
                name=server_name,
                command=server_config["command"],
                args=server_config.get("args", []),
                env=env,
                transport=server_transport,
            )

            success = await self.mcp_client.add_server(mcp_config)
            if success:
                conn = self.mcp_client.servers.get(server_name)
                if self.transport == "http" and hasattr(conn, "_base_url"):
                    # Extract port for netem loopback setup
                    port = int(conn._base_url.rsplit(":", 1)[1])
                    self.http_ports[server_name] = port
                    print(f"  Connected to MCP server: {server_name} (HTTP port {port})")
                else:
                    print(f"  Connected to MCP server: {server_name} (stdio)")
            else:
                print(f"  Failed to connect to MCP server: {server_name}")

        self.connected = len(self.mcp_client.servers) > 0
        return self.connected

    async def disconnect(self):
        """Disconnect all MCP servers."""
        await self.mcp_client.disconnect_all()
        self.connected = False

    def get_tools_for_openai(self) -> list[dict]:
        """Get available tools in OpenAI function calling format."""
        return self.mcp_client.get_tools_for_openai()

    async def execute(self, tool_name: str, arguments: dict) -> MCPToolResult:
        """Execute a tool by name."""
        # Check for tool aliases
        aliases = self.config.get("toolAliases", {})
        actual_tool_name = aliases.get(tool_name, tool_name)

        await self._apply_rate_limit(actual_tool_name)

        # Execute the tool
        semaphore = self._concurrency_limits.get(actual_tool_name)
        if semaphore:
            async with semaphore:
                result = await self.mcp_client.call_tool(actual_tool_name, arguments)
        else:
            result = await self.mcp_client.call_tool(actual_tool_name, arguments)

        # Track stats
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {
                "calls": 0,
                "success": 0,
                "total_latency": 0,
                "total_bytes": 0
            }

        stats = self._tool_stats[tool_name]
        stats["calls"] += 1
        stats["total_latency"] += result.latency_sec
        stats["total_bytes"] += result.request_bytes + result.response_bytes
        if result.success:
            stats["success"] += 1

        return result

    def get_stats(self) -> dict:
        """Get tool execution statistics."""
        return self._tool_stats

    async def _apply_rate_limit(self, tool_name: str) -> None:
        """Apply rate limiting based on config.

        If the tool belongs to a ``rate_group``, the group's shared sliding
        window is used instead of a per-tool window.  This ensures that
        tools sharing a backend budget (e.g. all Spotify endpoints) are
        throttled correctly against a single global limit.
        """
        limits = self._rate_limits.get(tool_name)
        if not limits:
            return

        # Resolve effective key and limits: group overrides per-tool
        group_name = limits.get("rate_group")
        if group_name and group_name in self._rate_groups:
            effective_key = group_name
            effective_limits = self._rate_groups[group_name]
        else:
            effective_key = tool_name
            effective_limits = limits

        lock = self._rate_locks.setdefault(effective_key, asyncio.Lock())
        async with lock:
            now = time.time()

            rpm = effective_limits.get("requests_per_minute")
            if rpm:
                window = self._rate_state.setdefault(effective_key, deque())
                while window and now - window[0] >= 60:
                    window.popleft()
                if len(window) >= rpm:
                    sleep_for = 60 - (now - window[0])
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                    now = time.time()
                    while window and now - window[0] >= 60:
                        window.popleft()
                window.append(now)

            rpd = effective_limits.get("requests_per_day")
            if rpd:
                day_window = self._rate_state_day.setdefault(effective_key, deque())
                while day_window and now - day_window[0] >= 86400:
                    day_window.popleft()
                if len(day_window) >= rpd:
                    sleep_for = 86400 - (now - day_window[0])
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                    now = time.time()
                    while day_window and now - day_window[0] >= 86400:
                        day_window.popleft()
                day_window.append(now)


class BaseAgentScenario(BaseScenario):
    """
    Base class for agent scenarios using real MCP tools.

    Set ``mcp_transport: http`` in scenario config (or pass ``--mcp-transport http``
    on the CLI) to route MCP traffic over TCP so that tc/netem on the loopback
    interface can shape it.
    """

    def __init__(self, client, logger, config):
        super().__init__(client, logger, config)
        self.server_group = config.get("server_group", "general")
        self.mcp_transport = config.get("mcp_transport", "http")
        self.tool_executor = MCPToolExecutor(
            self.server_group, transport=self.mcp_transport
        )
        self.max_iterations = config.get("max_tool_calls", 10)
        self.max_tool_result_chars = config.get("max_tool_result_chars", 200000)
        self.max_message_chars = config.get("max_message_chars", 800000)
        self._netem_on_lo = False
        # Set by the orchestrator before run() so setup/teardown can use it
        self.emulator = None
        self._current_network_profile: Optional[str] = None

    async def setup(self):
        """Connect to MCP servers before running scenarios.

        When transport is ``http`` and self.emulator is set, the same netem
        profile is applied to the loopback interface (port-filtered) so that
        MCP JSON-RPC traffic is shaped identically to external API traffic.
        """
        transport_label = self.mcp_transport
        print(f"Connecting to MCP servers (group: {self.server_group}, transport: {transport_label})...")
        success = await self.tool_executor.connect()
        if not success:
            raise RuntimeError("Failed to connect to any MCP servers")
        print(f"Connected. Available tools: {[t.name for t in self.tool_executor.mcp_client.get_tools()]}")

        # Apply netem to loopback for HTTP-mode MCP servers.
        # Skip for the no_emulation reference profile: that profile is meant
        # to leave lo completely unshaped (no netem queueing on MCP traffic).
        if (
            self.mcp_transport == "http"
            and self.emulator is not None
            and self._current_network_profile
            and self._current_network_profile != "no_emulation"
            and self.tool_executor.http_ports
        ):
            for name, port in self.tool_executor.http_ports.items():
                try:
                    self.emulator.apply_profile_to_loopback(
                        self._current_network_profile, port
                    )
                    self._netem_on_lo = True
                    print(f"  Applied netem ({self._current_network_profile}) to lo:{port} for {name}")
                except Exception as e:
                    print(f"  Warning: failed to apply netem to lo for {name}: {e}")

    async def teardown(self):
        """Disconnect MCP servers after scenarios complete."""
        if self._netem_on_lo and self.emulator is not None:
            self.emulator.clear_loopback()
            self._netem_on_lo = False
        await self.tool_executor.disconnect()

    def run(self, network_profile: str, run_index: int = 0) -> ScenarioResult:
        """Synchronous wrapper for async run."""
        return asyncio.get_event_loop().run_until_complete(
            self.run_async(network_profile, run_index)
        )

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """Execute the agent scenario asynchronously."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def _snapshot_request(
        self,
        model: str,
        messages: list,
        tools: Optional[list],
        iteration: int,
    ) -> dict:
        """Build a JSON-safe snapshot of an outgoing LLM request.

        Used by the exception path in _run_agent_turn so that when a provider
        rejects the request (e.g. HTTP 400) we still have a concrete record
        of what was sent. Content is previewed rather than dumped verbatim
        to keep the log DB compact.
        """
        preview_chars = 400
        msg_snapshots = []
        last_tool_call = None
        for idx, m in enumerate(messages):
            content = getattr(m, "content", None)
            role = getattr(m, "role", None)
            role_str = role.value if hasattr(role, "value") else str(role)
            content_str = content if isinstance(content, str) else repr(content)
            if content_str and content_str.startswith("Tool result from "):
                # e.g. "Tool result from get_current_weather:\n{...}"
                try:
                    last_tool_call = content_str.split(":", 1)[0].replace(
                        "Tool result from ", ""
                    )
                except Exception:
                    pass
            msg_snapshots.append({
                "index": idx,
                "role": role_str,
                "content_type": type(content).__name__,
                "content_len": len(content_str) if content_str else 0,
                "content_preview": (content_str or "")[:preview_chars],
                "tool_call_id": getattr(m, "tool_call_id", None),
            })

        return {
            "iteration": iteration,
            "model": model,
            "message_count": len(messages),
            "tool_count": len(tools) if tools else 0,
            "tool_names": [
                t.get("function", {}).get("name") for t in (tools or [])
            ],
            "messages": msg_snapshots,
            "last_tool_call": last_tool_call,
            "total_content_chars": sum(
                len(s["content_preview"]) for s in msg_snapshots
            ),
        }

    def _validate_json_safety(
        self,
        obj: Any,
        context: str,
        log: bool = True,
    ) -> list:
        """Walk *obj* and flag values that would produce invalid JSON.

        Specifically looks for float NaN/Infinity (Python's json.dumps emits
        these as bare `NaN`/`Infinity` tokens with allow_nan=True, which are
        not legal JSON and are rejected by OpenAI with HTTP 400 "could not
        parse the JSON body"), plus bytes and other non-JSON-native types.
        Returns a list of issue descriptions; emits a WARNING for each when
        *log* is True.
        """
        import math

        issues: list = []

        def _walk(node: Any, path: str) -> None:
            if isinstance(node, float):
                if math.isnan(node):
                    issues.append(f"NaN at {path}")
                elif math.isinf(node):
                    issues.append(f"Infinity at {path}")
            elif isinstance(node, (bytes, bytearray)):
                issues.append(f"bytes ({len(node)}B) at {path}")
            elif isinstance(node, dict):
                for k, v in node.items():
                    _walk(v, f"{path}.{k}")
            elif isinstance(node, (list, tuple)):
                for i, v in enumerate(node):
                    _walk(v, f"{path}[{i}]")
            elif node is None or isinstance(node, (str, int, bool)):
                return
            else:
                issues.append(
                    f"non-JSON-native {type(node).__name__} at {path}"
                )

        try:
            _walk(obj, "$")
        except Exception as e:
            issues.append(f"walker crashed: {type(e).__name__}: {e}")

        if issues and log:
            logger.warning(
                "JSON-safety issues in %s: %s",
                context,
                issues[:20],
            )
        return issues

    async def _run_agent_turn(
        self,
        user_prompt: str,
        system_prompt: str,
        model: str,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
    ) -> dict:
        """Run a single agent turn with potential tool calls."""
        turn_result = {
            "success": True,
            "api_calls": 0,
            "tool_calls": 0,
            "tool_latency": 0.0,
            "total_latency": 0.0,
            "request_bytes": 0,
            "response_bytes": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "log_records": [],
        }

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt)
        ]

        # Get available tools
        tools = self.tool_executor.get_tools_for_openai()

        # Flag suspect tool schemas up front (NaN defaults, non-serializable
        # values) — these would corrupt every subsequent request.
        self._validate_json_safety(
            {"tools": tools},
            context=f"{self.scenario_id} tools schema",
        )

        for iteration in range(self.max_iterations):
            t_start = time.time()

            # Snapshot the outgoing request in a JSON-safe form so the
            # exception handler below has something concrete to log when
            # the server rejects the body.
            pending_request = self._snapshot_request(
                model=model,
                messages=messages,
                tools=tools,
                iteration=iteration,
            )

            try:
                response: ChatResponse = self.client.chat(
                    messages=messages,
                    model=model,
                    stream=False,
                    tools=tools if tools else None
                )

                tokens_in = response.tokens_in
                tokens_out = response.tokens_out
                if tokens_in is None:
                    tokens_in = self.client.estimate_message_tokens(messages, model)
                if tokens_out is None:
                    tokens_out = self.client.estimate_tokens(response.content or "", model)

                turn_result["api_calls"] += 1
                turn_result["request_bytes"] += response.request_bytes
                turn_result["response_bytes"] += response.response_bytes
                turn_result["tokens_in"] += tokens_in or 0
                turn_result["tokens_out"] += tokens_out or 0
                turn_result["total_latency"] += response.latency_sec

                # Log the API call with full traffic detail
                llm_meta = {
                    "type": "llm_api_call",
                    "iteration": iteration,
                    "has_tools": len(response.tool_calls) > 0,
                    "tool_names": [tc.name for tc in response.tool_calls],
                    "message_count": len(messages),
                    "llm_request_bytes": response.request_bytes,
                    "llm_response_bytes": response.response_bytes,
                }

                # Capture tool call arguments the LLM is requesting
                if response.tool_calls:
                    llm_meta["tool_calls_detail"] = [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ]

                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    request_bytes=response.request_bytes,
                    response_bytes=response.response_bytes,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    t_request_start=t_start,
                    latency_sec=response.latency_sec,
                    http_status=200,
                    success=True,
                    tool_calls_count=len(response.tool_calls),
                    trace_request=response.request_payload,
                    trace_response=response.response_payload,
                    trace_note="agent_chat",
                    metadata=json.dumps(llm_meta)
                )
                self.logger.log(record)
                turn_result["log_records"].append(record)

                # Check if there are tool calls to execute
                if not response.tool_calls:
                    # No more tool calls, agent is done
                    break

                # Execute tool calls via MCP
                for tool_call in response.tool_calls:
                    t_tool_start = time.time()

                    # Execute via MCP
                    tool_result = await self.tool_executor.execute(
                        tool_call.name,
                        tool_call.arguments
                    )

                    turn_result["tool_calls"] += 1
                    turn_result["tool_latency"] += tool_result.latency_sec
                    turn_result["request_bytes"] += tool_result.request_bytes
                    turn_result["response_bytes"] += tool_result.response_bytes

                    # Add assistant message with tool call indication
                    messages.append(ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response.content or f"Calling {tool_call.name}..."
                    ))

                    # Check the raw tool result for values that would
                    # corrupt downstream JSON serialization (e.g. NaN from
                    # weather APIs). We warn but do not abort: the agent
                    # loop will still substitute the string representation
                    # into the next prompt.
                    self._validate_json_safety(
                        tool_result.result,
                        context=f"tool_result from {tool_call.name}",
                    )

                    # Add tool result (truncated to avoid context overflow)
                    if tool_result.success:
                        tool_result_str = json.dumps(tool_result.result) if isinstance(
                            tool_result.result, (dict, list)
                        ) else str(tool_result.result)
                    else:
                        tool_result_str = json.dumps({
                            "error": tool_result.error,
                            "success": False
                        })

                    if len(tool_result_str) > self.max_tool_result_chars:
                        tool_result_str = (
                            tool_result_str[:self.max_tool_result_chars]
                            + f"\n... [truncated from {len(tool_result_str)} chars]"
                        )

                    messages.append(ChatMessage(
                        role=MessageRole.USER,
                        content=f"Tool result from {tool_call.name}:\n{tool_result_str}"
                    ))

                    # Trim older tool exchanges if context grows too large
                    total_chars = sum(len(m.content) for m in messages)
                    if total_chars > self.max_message_chars:
                        # Keep system prompt (index 0), original user prompt (1),
                        # and the most recent messages. Summarize the middle.
                        preserved_head = 2
                        preserved_tail = 6  # recent tool call/result pairs
                        if len(messages) > preserved_head + preserved_tail:
                            removed = len(messages) - preserved_head - preserved_tail
                            messages = (
                                messages[:preserved_head]
                                + [ChatMessage(
                                    role=MessageRole.USER,
                                    content=f"[{removed} earlier tool exchanges omitted for brevity]",
                                )]
                                + messages[-preserved_tail:]
                            )

                    # Log tool execution with full MCP traffic detail
                    tool_meta = {
                        "type": "mcp_tool_call",
                        "tool_name": tool_call.name,
                        "iteration": iteration,
                        "tool_arguments": tool_call.arguments,
                        "mcp_stdio_request_bytes": tool_result.request_bytes,
                        "mcp_stdio_response_bytes": tool_result.response_bytes,
                    }

                    # Include result summary (truncated to avoid bloating metadata)
                    if tool_result.result:
                        result_str = str(tool_result.result)
                        tool_meta["result_summary"] = result_str[:500] + ("..." if len(result_str) > 500 else "")
                        tool_meta["result_bytes"] = len(result_str)

                    # Merge backend telemetry from MCP server
                    if tool_result.backend_meta:
                        bm = tool_result.backend_meta
                        tool_meta["backend_total_request_bytes"] = bm.get("backend_total_request_bytes", 0)
                        tool_meta["backend_total_response_bytes"] = bm.get("backend_total_response_bytes", 0)
                        tool_meta["backend_total_latency_sec"] = bm.get("backend_total_latency_sec", 0)
                        tool_meta["backend_retries"] = bm.get("backend_retries", 0)
                        tool_meta["backend_rate_limited"] = bm.get("backend_rate_limited", False)
                        if bm.get("backend_calls"):
                            tool_meta["backend_calls"] = bm["backend_calls"]
                        if bm.get("token_call"):
                            tool_meta["token_call"] = bm["token_call"]

                    # Build trace payloads for MCP tool call
                    tool_trace_req = {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    tool_trace_resp = {
                        "success": tool_result.success,
                        "result": tool_result.result,
                        "error": tool_result.error,
                        "backend_meta": tool_result.backend_meta,
                    }

                    tool_record = self._create_log_record(
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=network_profile,
                        request_bytes=tool_result.request_bytes,
                        response_bytes=tool_result.response_bytes,
                        t_request_start=t_tool_start,
                        latency_sec=tool_result.latency_sec,
                        http_status=200 if tool_result.success else 500,
                        success=tool_result.success,
                        error_type=tool_result.error if not tool_result.success else None,
                        tool_calls_count=1,
                        total_tool_bytes=tool_result.request_bytes + tool_result.response_bytes,
                        tool_latency_sec=tool_result.latency_sec,
                        trace_request=tool_trace_req,
                        trace_response=tool_trace_resp,
                        trace_note="mcp_tool_call",
                        metadata=json.dumps(tool_meta)
                    )
                    self.logger.log(tool_record)
                    turn_result["log_records"].append(tool_record)

            except Exception as e:
                turn_result["success"] = False
                turn_result["error"] = str(e)

                # Extract HTTP status from providers that surface it on
                # the exception (OpenAI SDK -> APIStatusError has .status_code).
                http_status = getattr(e, "status_code", None) or 0

                # Produce a diagnostic dump: the request snapshot + any
                # JSON-safety violations we can detect after the fact.
                diagnostic = {
                    "iteration": iteration,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "http_status": http_status,
                    "pending_request": pending_request,
                    "json_safety_issues": self._validate_json_safety(
                        pending_request,
                        context=f"{self.scenario_id} failed request",
                        log=False,
                    ),
                }
                # Some SDKs expose the raw response body on the exception
                for attr in ("response", "body", "message"):
                    val = getattr(e, attr, None)
                    if val is not None and attr != "message":
                        try:
                            diagnostic[f"exc_{attr}"] = (
                                val.text if hasattr(val, "text") else repr(val)[:2000]
                            )
                        except Exception:
                            pass

                logger.error(
                    "LLM call failed in %s turn=%d iter=%d: %s: %s "
                    "(http=%s, messages=%d, tools=%d, last_tool=%s, json_issues=%s)",
                    self.scenario_id,
                    turn_index,
                    iteration,
                    type(e).__name__,
                    str(e)[:300],
                    http_status,
                    len(messages),
                    len(tools) if tools else 0,
                    pending_request.get("last_tool_call"),
                    diagnostic["json_safety_issues"] or "none",
                )
                logger.debug(
                    "Full failing request snapshot:\n%s",
                    json.dumps(pending_request, indent=2, default=str)[:8000],
                )

                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    t_request_start=t_start,
                    latency_sec=time.time() - t_start,
                    http_status=http_status,
                    error_type=f"{type(e).__name__}: {str(e)[:500]}",
                    success=False,
                    trace_request=pending_request,
                    trace_response=diagnostic,
                    trace_note="agent_chat_error",
                )
                self.logger.log(record)
                turn_result["log_records"].append(record)
                break

        return turn_result


class ShoppingAgentScenario(BaseAgentScenario):
    """
    Shopping agent scenario with real MCP tool calling.

    Uses web search to:
    - Search for products
    - Compare prices
    - Find reviews
    - Make recommendations
    """

    def __init__(self, client, logger, config):
        # Override server group for shopping
        config.setdefault("server_group", "shopping")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "shopping_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """Execute a shopping agent session."""
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Find me the best laptop under $1000 for programming. Compare at least 3 options with prices and reviews."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = """You are a helpful shopping assistant. Use the available tools to:
1. Search the web for products matching the user's requirements
2. Fetch detailed product pages for pricing and specifications
3. Compare options and provide recommendations

When searching for products:
- Use brave_web_search to find product listings and reviews
- Use fetch to get detailed information from promising URLs
- Always provide specific prices, ratings, and where to buy

Be thorough but efficient. Provide actionable recommendations with real data."""

        try:
            # Setup MCP connections
            await self.setup()

            for prompt_index, user_prompt in enumerate(prompts):
                await self._wait_between_prompts_async(prompt_index)
                turn_result = await self._run_agent_turn(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    session_id=session_id,
                    turn_index=prompt_index,
                    run_index=run_index,
                    network_profile=network_profile,
                )

                # Aggregate results
                result.turn_count += 1
                result.api_call_count += turn_result["api_calls"]
                result.tool_calls_count += turn_result["tool_calls"]
                result.tool_total_latency_sec += turn_result["tool_latency"]
                result.total_latency_sec += turn_result["total_latency"]
                result.total_request_bytes += turn_result["request_bytes"]
                result.total_response_bytes += turn_result["response_bytes"]
                result.total_tokens_in += turn_result.get("tokens_in", 0)
                result.total_tokens_out += turn_result.get("tokens_out", 0)
                result.log_records.extend(turn_result["log_records"])

                if not turn_result["success"]:
                    result.success = False
                    result.error_message = turn_result.get("error")
                    break

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        finally:
            await self.teardown()

        return result


class WebSearchAgentScenario(BaseAgentScenario):
    """
    Web search/research agent scenario with real MCP tools.

    Uses MCP servers to:
    - Search the web for information
    - Fetch and analyze content from URLs
    - Synthesize findings into reports
    """

    def __init__(self, client, logger, config):
        # Override server group for web research
        config.setdefault("server_group", "web_research")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "web_search_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """Execute a web search agent session."""
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Research the latest developments in 6G wireless technology. Find at least 3 recent sources and summarize the key trends."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        system_prompt = """You are a research assistant. Use the available tools to:
1. Search the web for relevant, recent information
2. Fetch content from authoritative sources
3. Synthesize findings into a coherent summary

Research guidelines:
- Use brave_web_search to find relevant sources
- Use fetch to retrieve full content from key URLs
- Cite your sources with URLs
- Focus on recent, authoritative information

Be thorough - fetch multiple sources before synthesizing."""

        try:
            # Setup MCP connections
            await self.setup()

            for prompt_index, user_prompt in enumerate(prompts):
                await self._wait_between_prompts_async(prompt_index)
                turn_result = await self._run_agent_turn(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    session_id=session_id,
                    turn_index=prompt_index,
                    run_index=run_index,
                    network_profile=network_profile,
                )

                result.turn_count += 1
                result.api_call_count += turn_result["api_calls"]
                result.tool_calls_count += turn_result["tool_calls"]
                result.tool_total_latency_sec += turn_result["tool_latency"]
                result.total_latency_sec += turn_result["total_latency"]
                result.total_request_bytes += turn_result["request_bytes"]
                result.total_response_bytes += turn_result["response_bytes"]
                result.total_tokens_in += turn_result.get("tokens_in", 0)
                result.total_tokens_out += turn_result.get("tokens_out", 0)
                result.log_records.extend(turn_result["log_records"])

                if not turn_result["success"]:
                    result.success = False
                    result.error_message = turn_result.get("error")
                    break

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        finally:
            await self.teardown()

        return result


class GeneralAgentScenario(BaseAgentScenario):
    """
    General-purpose agent scenario with full MCP tool access.

    Uses all available MCP servers for complex multi-step tasks.
    """

    def __init__(self, client, logger, config):
        config.setdefault("server_group", "general")
        super().__init__(client, logger, config)

    @property
    def scenario_type(self) -> str:
        return "general_agent"

    async def run_async(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """Execute a general agent session."""
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        prompts = self.config.get("prompts", [
            "Research and write a brief report on quantum computing advances in 2024. Save the report to a file."
        ])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        default_system_prompt = """You are a capable AI assistant with access to various tools.

Available capabilities:
- Web search (brave_web_search) - Search the internet for information
- URL fetching (fetch) - Retrieve content from web pages
- File operations (read_file, write_file, etc.) - Work with files
- Memory (store, retrieve) - Remember information across interactions

Use these tools effectively to complete the user's request. Be thorough and verify your work."""
        system_prompt = self.config.get("system_prompt", default_system_prompt)

        try:
            await self.setup()

            for prompt_index, user_prompt in enumerate(prompts):
                await self._wait_between_prompts_async(prompt_index)
                turn_result = await self._run_agent_turn(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    session_id=session_id,
                    turn_index=prompt_index,
                    run_index=run_index,
                    network_profile=network_profile,
                )

                result.turn_count += 1
                result.api_call_count += turn_result["api_calls"]
                result.tool_calls_count += turn_result["tool_calls"]
                result.tool_total_latency_sec += turn_result["tool_latency"]
                result.total_latency_sec += turn_result["total_latency"]
                result.total_request_bytes += turn_result["request_bytes"]
                result.total_response_bytes += turn_result["response_bytes"]
                result.total_tokens_in += turn_result.get("tokens_in", 0)
                result.total_tokens_out += turn_result.get("tokens_out", 0)
                result.log_records.extend(turn_result["log_records"])

                if not turn_result["success"]:
                    result.success = False
                    result.error_message = turn_result.get("error")
                    break

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        finally:
            await self.teardown()

        return result
