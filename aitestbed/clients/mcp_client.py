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
    backend_meta: Optional[dict] = None


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    transport: str = "stdio"  # "stdio" or "http"
    http_host: str = "127.0.0.1"
    http_port: int = 0  # 0 = auto-assign


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

    def _send_notification(self, method: str, params: dict = None) -> int:
        """Send a JSON-RPC notification (no id, no response expected)."""
        if not self.process or self.process.poll() is not None:
            return 0
        notification = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params
        data = json.dumps(notification) + "\n"
        encoded = data.encode()
        self.process.stdin.write(encoded)
        self.process.stdin.flush()
        return len(encoded)

    async def _initialize(self):
        """Send initialize request to MCP server."""
        try:
            from mcp.types import LATEST_PROTOCOL_VERSION
            protocol_version = LATEST_PROTOCOL_VERSION
        except ImportError:
            protocol_version = "2025-11-25"

        params = {
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": {
                "name": "6g-ai-traffic-testbed",
                "version": "1.0.0"
            }
        }
        await self._send_request("initialize", params)
        self._send_notification("notifications/initialized")

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

            # Extract backend telemetry (non-standard _meta field)
            backend_meta = result.pop("_meta", None)

            # Extract content from result
            content = result.get("content", [])
            if content and isinstance(content, list):
                # Combine text content
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                result_data = "\n".join(text_parts) if text_parts else content
            else:
                result_data = result

            is_error = result.get("isError", False)
            error_msg = None
            if is_error:
                error_msg = result_data if isinstance(result_data, str) else str(result_data)

            return MCPToolResult(
                tool_name=tool_name,
                success=not is_error,
                result=result_data,
                error=error_msg,
                latency_sec=latency,
                request_bytes=req_bytes,
                response_bytes=resp_bytes,
                backend_meta=backend_meta,
            )

        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                latency_sec=time.time() - t_start
            )


class MCPHttpConnection:
    """
    Manages connection to an MCP server over HTTP.

    Launches the MCP server subprocess with ``--http`` and communicates
    via HTTP POST so that traffic traverses the network stack and is
    affected by tc/netem rules on the loopback interface.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.tools: dict[str, MCPTool] = {}
        self._request_id = 0
        self._base_url: str = ""

    async def connect(self) -> bool:
        try:
            import requests as _requests
            self._requests = _requests

            env = os.environ.copy()
            env.update(self.config.env)

            # Launch the server with --http flag
            args = [self.config.command] + self.config.args + ["--http"]
            if self.config.http_port:
                args.append(f"--port={self.config.http_port}")
            if self.config.http_host != "127.0.0.1":
                args.append(f"--host={self.config.http_host}")

            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.config.cwd,
            )

            # Read the PORT=<n> line the server prints on startup
            port_line = self.process.stdout.readline().decode().strip()
            if not port_line.startswith("PORT="):
                raise RuntimeError(
                    f"MCP HTTP server did not report port: {port_line}"
                )
            port = int(port_line.split("=", 1)[1])
            self._base_url = f"http://{self.config.http_host}:{port}"

            # Initialize and list tools
            await self._initialize()
            await self._list_tools()
            return True

        except Exception as e:
            print(f"Failed to connect to MCP HTTP server {self.config.name}: {e}")
            return False

    async def disconnect(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def _send_http(self, method: str, params: dict = None) -> tuple[dict, int, int]:
        """Send a JSON-RPC request over HTTP and return (result, req_bytes, resp_bytes)."""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        request_body = json.dumps(request).encode()
        resp = self._requests.post(
            self._base_url,
            data=request_body,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        response = resp.json()

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")

        return response.get("result", {}), len(request_body), len(resp.content)

    def _send_http_notification(self, method: str, params: dict = None) -> int:
        """Send a JSON-RPC notification over HTTP (no id, no response expected)."""
        notification = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params
        body = json.dumps(notification).encode()
        self._requests.post(
            self._base_url,
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return len(body)

    async def _initialize(self):
        try:
            from mcp.types import LATEST_PROTOCOL_VERSION
            protocol_version = LATEST_PROTOCOL_VERSION
        except ImportError:
            protocol_version = "2025-11-25"

        params = {
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": {"name": "6g-ai-traffic-testbed", "version": "1.0.0"},
        }
        self._send_http("initialize", params)
        self._send_http_notification("notifications/initialized")

    async def _list_tools(self):
        result, _, _ = self._send_http("tools/list")
        self.tools = {}
        for tool_data in result.get("tools", []):
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
                server_name=self.config.name,
            )
            self.tools[tool.name] = tool

    async def call_tool(self, tool_name: str, arguments: dict) -> MCPToolResult:
        t_start = time.time()
        try:
            result, req_bytes, resp_bytes = self._send_http(
                "tools/call", {"name": tool_name, "arguments": arguments}
            )
            latency = time.time() - t_start

            backend_meta = result.pop("_meta", None)

            content = result.get("content", [])
            if content and isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                result_data = "\n".join(text_parts) if text_parts else content
            else:
                result_data = result

            is_error = result.get("isError", False)
            error_msg = None
            if is_error:
                error_msg = result_data if isinstance(result_data, str) else str(result_data)

            return MCPToolResult(
                tool_name=tool_name,
                success=not is_error,
                result=result_data,
                error=error_msg,
                latency_sec=latency,
                request_bytes=req_bytes,
                response_bytes=resp_bytes,
                backend_meta=backend_meta,
            )
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                latency_sec=time.time() - t_start,
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
        if config.transport == "http":
            connection = MCPHttpConnection(config)
        else:
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
    command=sys.executable,
    args=["-m", "mcp_server_fetch"]
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
