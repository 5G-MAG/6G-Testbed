"""
Agent Scenarios for the 6G AI Traffic Testbed.

Implements agentic AI patterns with real MCP tool execution.
"""

import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional, Any

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


class MCPToolExecutor:
    """
    Real MCP tool executor using connected MCP servers.

    Manages MCP server lifecycle and provides tool execution.
    """

    def __init__(self, server_group: str = "general"):
        self.mcp_client = MCPClient()
        self.server_group = server_group
        self.config: dict = {}
        self.connected = False
        self._tool_stats: dict[str, dict] = {}
        self._rate_limits: dict[str, dict] = {}
        self._rate_state: dict[str, deque] = {}
        self._rate_state_day: dict[str, deque] = {}
        self._rate_locks: dict[str, asyncio.Lock] = {}
        self._concurrency_limits: dict[str, asyncio.Semaphore] = {}

    async def connect(self, config_path: str = None):
        """Connect to MCP servers defined in configuration."""
        self.config = load_mcp_servers_config(config_path)
        self._rate_limits = self.config.get("rateLimits", {})

        for tool_name, limits in self._rate_limits.items():
            self._rate_state.setdefault(tool_name, deque())
            self._rate_state_day.setdefault(tool_name, deque())
            self._rate_locks.setdefault(tool_name, asyncio.Lock())
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

            mcp_config = MCPServerConfig(
                name=server_name,
                command=server_config["command"],
                args=server_config.get("args", []),
                env=env
            )

            success = await self.mcp_client.add_server(mcp_config)
            if success:
                print(f"  Connected to MCP server: {server_name}")
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
        """Apply per-tool rate limiting based on config."""
        limits = self._rate_limits.get(tool_name)
        if not limits:
            return

        lock = self._rate_locks.setdefault(tool_name, asyncio.Lock())
        async with lock:
            now = time.time()

            rpm = limits.get("requests_per_minute")
            if rpm:
                window = self._rate_state.setdefault(tool_name, deque())
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

            rpd = limits.get("requests_per_day")
            if rpd:
                day_window = self._rate_state_day.setdefault(tool_name, deque())
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
    """

    def __init__(self, client, logger, config):
        super().__init__(client, logger, config)
        self.server_group = config.get("server_group", "general")
        self.tool_executor = MCPToolExecutor(self.server_group)
        self.max_iterations = config.get("max_tool_calls", 10)

    async def setup(self):
        """Connect to MCP servers before running scenarios."""
        print(f"Connecting to MCP servers (group: {self.server_group})...")
        success = await self.tool_executor.connect()
        if not success:
            raise RuntimeError("Failed to connect to any MCP servers")
        print(f"Connected. Available tools: {[t.name for t in self.tool_executor.mcp_client.get_tools()]}")

    async def teardown(self):
        """Disconnect MCP servers after scenarios complete."""
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

        for iteration in range(self.max_iterations):
            t_start = time.time()

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

                # Log the API call
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
                    metadata=json.dumps({
                        "iteration": iteration,
                        "has_tools": len(response.tool_calls) > 0,
                        "tool_names": [tc.name for tc in response.tool_calls]
                    })
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

                    # Add tool result
                    if tool_result.success:
                        tool_result_str = json.dumps(tool_result.result) if isinstance(
                            tool_result.result, (dict, list)
                        ) else str(tool_result.result)
                    else:
                        tool_result_str = json.dumps({
                            "error": tool_result.error,
                            "success": False
                        })

                    messages.append(ChatMessage(
                        role=MessageRole.USER,
                        content=f"Tool result from {tool_call.name}:\n{tool_result_str}"
                    ))

                    # Log tool execution
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
                        metadata=json.dumps({
                            "type": "mcp_tool_call",
                            "tool_name": tool_call.name,
                            "iteration": iteration
                        })
                    )
                    self.logger.log(tool_record)
                    turn_result["log_records"].append(tool_record)

            except Exception as e:
                turn_result["success"] = False
                turn_result["error"] = str(e)

                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    t_request_start=t_start,
                    latency_sec=time.time() - t_start,
                    http_status=0,
                    error_type=str(e),
                    success=False,
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
