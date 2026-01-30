"""
MCP (Model Context Protocol) Client for the 6G AI Traffic Testbed.

Provides connectivity to real MCP servers for tool execution.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: dict
    server_name: str


@dataclass
class MCPToolResult:
    """Result from an MCP tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    latency_sec: float = 0.0
    request_bytes: int = 0
    response_bytes: int = 0


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None


class MCPServerConnection:
    """
    Manages connection to a single MCP server via stdio.

    Uses JSON-RPC 2.0 protocol over stdin/stdout.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.tools: dict[str, MCPTool] = {}
        self._request_id = 0
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Start the MCP server process and initialize connection."""
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.env)

            # Start the server process
            self.process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.config.cwd,
                bufsize=0
            )

            # Initialize the connection
            await self._initialize()

            # Discover available tools
            await self._list_tools()

            return True

        except Exception as e:
            print(f"Failed to connect to MCP server {self.config.name}: {e}")
            return False

    async def disconnect(self):
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    async def _send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request and wait for response."""
        if not self.process or self.process.poll() is not None:
            raise RuntimeError(f"MCP server {self.config.name} is not running")

        async with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
            }
            if params:
                request["params"] = params

            request_str = json.dumps(request) + "\n"
            request_bytes = request_str.encode()

            # Send request
            self.process.stdin.write(request_bytes)
            self.process.stdin.flush()

            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                raise RuntimeError(f"No response from MCP server {self.config.name}")

            response = json.loads(response_line.decode())

            if "error" in response:
                raise RuntimeError(f"MCP error: {response['error']}")

            return response.get("result", {}), len(request_bytes), len(response_line)

    async def _initialize(self):
        """Send initialize request to MCP server."""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "6g-ai-traffic-testbed",
                "version": "1.0.0"
            }
        }
        await self._send_request("initialize", params)
        await self._send_request("notifications/initialized")

    async def _list_tools(self):
        """Discover available tools from the server."""
        result, _, _ = await self._send_request("tools/list")

        self.tools = {}
        for tool_data in result.get("tools", []):
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self.config.name
            )
            self.tools[tool.name] = tool

    async def call_tool(self, tool_name: str, arguments: dict) -> MCPToolResult:
        """Execute a tool and return the result."""
        t_start = time.time()

        try:
            result, req_bytes, resp_bytes = await self._send_request(
                "tools/call",
                {
                    "name": tool_name,
                    "arguments": arguments
                }
            )

            latency = time.time() - t_start

            # Extract content from result
            content = result.get("content", [])
            if content and isinstance(content, list):
                # Combine text content
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                result_data = "\n".join(text_parts) if text_parts else content
            else:
                result_data = result

            return MCPToolResult(
                tool_name=tool_name,
                success=not result.get("isError", False),
                result=result_data,
                latency_sec=latency,
                request_bytes=req_bytes,
                response_bytes=resp_bytes
            )

        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                latency_sec=time.time() - t_start
            )


class MCPClient:
    """
    MCP Client managing multiple server connections.

    Provides unified interface for tool discovery and execution
    across multiple MCP servers.
    """

    def __init__(self):
        self.servers: dict[str, MCPServerConnection] = {}
        self.tools: dict[str, MCPTool] = {}

    async def add_server(self, config: MCPServerConfig) -> bool:
        """Add and connect to an MCP server."""
        connection = MCPServerConnection(config)
        if await connection.connect():
            self.servers[config.name] = connection
            # Register tools from this server
            for tool_name, tool in connection.tools.items():
                self.tools[tool_name] = tool
            return True
        return False

    async def remove_server(self, name: str):
        """Disconnect and remove an MCP server."""
        if name in self.servers:
            await self.servers[name].disconnect()
            # Remove tools from this server
            self.tools = {k: v for k, v in self.tools.items() if v.server_name != name}
            del self.servers[name]

    async def disconnect_all(self):
        """Disconnect all servers."""
        for name in list(self.servers.keys()):
            await self.remove_server(name)

    def get_tools(self) -> list[MCPTool]:
        """Get all available tools across all servers."""
        return list(self.tools.values())

    def get_tools_for_openai(self) -> list[dict]:
        """Get tools in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            }
            for tool in self.tools.values()
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> MCPToolResult:
        """Execute a tool by name."""
        if tool_name not in self.tools:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}"
            )

        tool = self.tools[tool_name]
        server = self.servers.get(tool.server_name)

        if not server:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Server not connected: {tool.server_name}"
            )

        return await server.call_tool(tool_name, arguments)


# Pre-configured MCP servers for common use cases
BRAVE_SEARCH_SERVER = MCPServerConfig(
    name="brave-search",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-brave-search"],
    env={"BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", "")}
)

FETCH_SERVER = MCPServerConfig(
    name="fetch",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-fetch"]
)

FILESYSTEM_SERVER = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
)

# For Python-based MCP servers
PYTHON_MCP_SERVER = lambda name, module: MCPServerConfig(
    name=name,
    command=sys.executable,
    args=["-m", module]
)


def load_mcp_config_from_file(config_path: str) -> list[MCPServerConfig]:
    """Load MCP server configurations from a JSON/YAML file."""
    path = Path(config_path)

    if not path.exists():
        return []

    with open(path) as f:
        if path.suffix in [".yaml", ".yml"]:
            import yaml
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    configs = []
    for server_data in data.get("mcpServers", {}).values():
        configs.append(MCPServerConfig(
            name=server_data.get("name", "unnamed"),
            command=server_data["command"],
            args=server_data.get("args", []),
            env=server_data.get("env", {})
        ))

    return configs
