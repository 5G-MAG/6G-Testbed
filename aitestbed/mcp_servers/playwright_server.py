#!/usr/bin/env python3
"""
Playwright MCP Server for the 6G AI Traffic Testbed.

Provides MCP tools for browser automation via Playwright,
enabling agentic web UI testing scenarios. Communicates over
stdio using JSON-RPC 2.0.

Prerequisites:
    pip install playwright
    playwright install chromium

Usage:
    python -m mcp_servers.playwright_server
"""

import asyncio
import base64
import json
import sys
import time
from typing import Optional

from playwright.async_api import async_playwright, Browser, Page


# ---------------------------------------------------------------------------
# Playwright browser helpers
# ---------------------------------------------------------------------------


def _estimate_bytes(data) -> int:
    """Estimate byte size of a Python object as JSON."""
    return len(json.dumps(data).encode()) if data else 0


class PlaywrightBrowser:
    """Manages a headless Chromium browser via Playwright."""

    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._last_call: Optional[dict] = None

    async def launch(self):
        """Launch headless Chromium."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            chromium_sandbox=False,
            args=["--no-sandbox", "--disable-gpu"],
        )
        context = await self._browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        self._page = await context.new_page()

    async def close(self):
        """Close browser and Playwright."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._page = None

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._page

    def _record_call(self, *, url: str, request_bytes: int, response_bytes: int, latency_sec: float):
        self._last_call = {
            "url": url,
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
            "latency_sec": round(latency_sec, 4),
        }

    # -- public tool implementations --

    async def navigate(self, url: str) -> dict:
        t0 = time.time()
        response = await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        elapsed = time.time() - t0

        title = await self.page.title()
        # Get a text snippet (first 500 chars)
        text_content = await self.page.evaluate("() => document.body?.innerText?.substring(0, 500) || ''")

        status_code = response.status if response else 0
        req_bytes = len(url.encode()) + 200  # estimate request overhead
        resp_bytes = len(title.encode()) + len(text_content.encode())

        self._record_call(url=url, request_bytes=req_bytes, response_bytes=resp_bytes, latency_sec=elapsed)

        return {
            "url": url,
            "status_code": status_code,
            "title": title,
            "text_snippet": text_content,
        }

    async def screenshot(self, selector: Optional[str] = None) -> dict:
        t0 = time.time()
        if selector:
            element = await self.page.query_selector(selector)
            if not element:
                elapsed = time.time() - t0
                self._record_call(url=self.page.url, request_bytes=len(selector.encode()), response_bytes=0, latency_sec=elapsed)
                return {"error": f"Element not found: {selector}"}
            screenshot_bytes = await element.screenshot()
        else:
            screenshot_bytes = await self.page.screenshot(full_page=True)
        elapsed = time.time() - t0

        b64_data = base64.b64encode(screenshot_bytes).decode()

        req_bytes = len((selector or "full_page").encode())
        resp_bytes = len(screenshot_bytes)

        self._record_call(url=self.page.url, request_bytes=req_bytes, response_bytes=resp_bytes, latency_sec=elapsed)

        return {
            "screenshot_base64": b64_data,
            "size_bytes": len(screenshot_bytes),
            "selector": selector or "full_page",
        }

    async def click(self, selector: str) -> dict:
        t0 = time.time()
        await self.page.click(selector, timeout=10000)
        # Wait briefly for navigation or dynamic content
        await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        elapsed = time.time() - t0

        title = await self.page.title()
        req_bytes = len(selector.encode())
        resp_bytes = len(title.encode())

        self._record_call(url=self.page.url, request_bytes=req_bytes, response_bytes=resp_bytes, latency_sec=elapsed)

        return {
            "clicked": selector,
            "page_title": title,
            "current_url": self.page.url,
        }

    async def fill(self, selector: str, value: str) -> dict:
        t0 = time.time()
        await self.page.fill(selector, value, timeout=10000)
        elapsed = time.time() - t0

        req_bytes = len(selector.encode()) + len(value.encode())

        self._record_call(url=self.page.url, request_bytes=req_bytes, response_bytes=0, latency_sec=elapsed)

        return {
            "filled": selector,
            "value": value,
        }

    async def get_text(self, selector: Optional[str] = None) -> dict:
        t0 = time.time()
        if selector:
            element = await self.page.query_selector(selector)
            if not element:
                elapsed = time.time() - t0
                self._record_call(url=self.page.url, request_bytes=len(selector.encode()), response_bytes=0, latency_sec=elapsed)
                return {"error": f"Element not found: {selector}"}
            text = await element.inner_text()
        else:
            text = await self.page.evaluate("() => document.body?.innerText || ''")
        elapsed = time.time() - t0

        req_bytes = len((selector or "body").encode())
        resp_bytes = len(text.encode())

        self._record_call(url=self.page.url, request_bytes=req_bytes, response_bytes=resp_bytes, latency_sec=elapsed)

        return {
            "selector": selector or "body",
            "text": text,
            "length": len(text),
        }

    async def evaluate(self, script: str) -> dict:
        t0 = time.time()
        result = await self.page.evaluate(script)
        elapsed = time.time() - t0

        result_str = json.dumps(result) if result is not None else "null"
        req_bytes = len(script.encode())
        resp_bytes = len(result_str.encode())

        self._record_call(url=self.page.url, request_bytes=req_bytes, response_bytes=resp_bytes, latency_sec=elapsed)

        return {
            "result": result,
            "script_length": len(script),
        }


# ---------------------------------------------------------------------------
# MCP tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "playwright_navigate",
        "description": "Navigate to a URL and return the page title, HTTP status, and a text snippet.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "playwright_screenshot",
        "description": "Take a screenshot of the current page or a specific element. Returns base64-encoded PNG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector to screenshot a specific element. Omit for full page.",
                },
            },
        },
    },
    {
        "name": "playwright_click",
        "description": "Click an element on the page by CSS selector.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to click",
                },
            },
            "required": ["selector"],
        },
    },
    {
        "name": "playwright_fill",
        "description": "Fill a form field with a value.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the input field",
                },
                "value": {
                    "type": "string",
                    "description": "Value to fill into the field",
                },
            },
            "required": ["selector", "value"],
        },
    },
    {
        "name": "playwright_get_text",
        "description": "Extract visible text content from the page or a specific element.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector. Omit to get all visible text from the page body.",
                },
            },
        },
    },
    {
        "name": "playwright_evaluate",
        "description": "Run JavaScript in the page context and return the result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "JavaScript code to evaluate in the browser page context",
                },
            },
            "required": ["script"],
        },
    },
]


# ---------------------------------------------------------------------------
# MCP stdio server (JSON-RPC 2.0)
# ---------------------------------------------------------------------------

browser = PlaywrightBrowser()


def _build_meta(tool_name: str, arguments: dict, error: str = None) -> dict:
    """Build telemetry metadata from the last browser call."""
    meta: dict = {
        "tool_name": tool_name,
        "tool_arguments": arguments,
    }

    if browser._last_call:
        call = browser._last_call
        meta["backend_calls"] = [call]
        meta["backend_total_request_bytes"] = call.get("request_bytes", 0)
        meta["backend_total_response_bytes"] = call.get("response_bytes", 0)
        meta["backend_total_latency_sec"] = call.get("latency_sec", 0)
        meta["backend_retries"] = 0
        meta["backend_rate_limited"] = False
        browser._last_call = None

    if error:
        meta["error"] = error

    return meta


async def handle_request(request: dict) -> dict:
    """Route a JSON-RPC request to the appropriate handler."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        # Launch browser on initialization
        await browser.launch()

        try:
            from mcp.types import LATEST_PROTOCOL_VERSION
            protocol_version = LATEST_PROTOCOL_VERSION
        except ImportError:
            protocol_version = "2025-11-25"
        result = {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "playwright-browser",
                "version": "1.0.0",
            },
        }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        return await _handle_tool_call(req_id, params)

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


async def _handle_tool_call(req_id, params: dict) -> dict:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    try:
        if tool_name == "playwright_navigate":
            data = await browser.navigate(url=arguments["url"])
        elif tool_name == "playwright_screenshot":
            data = await browser.screenshot(selector=arguments.get("selector"))
        elif tool_name == "playwright_click":
            data = await browser.click(selector=arguments["selector"])
        elif tool_name == "playwright_fill":
            data = await browser.fill(
                selector=arguments["selector"],
                value=arguments["value"],
            )
        elif tool_name == "playwright_get_text":
            data = await browser.get_text(selector=arguments.get("selector"))
        elif tool_name == "playwright_evaluate":
            data = await browser.evaluate(script=arguments["script"])
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                },
            }

        text = json.dumps(data, indent=2)
        meta = _build_meta(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "_meta": meta,
            },
        }

    except Exception as e:
        meta = _build_meta(tool_name, arguments, error=str(e))
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "_meta": meta,
            },
        }


async def main_stdio():
    """Run the MCP server over stdio (async for Playwright)."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    transport, _ = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)

    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            line = line.decode().strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = await handle_request(request)
                if response is not None:
                    writer.write((json.dumps(response) + "\n").encode())
                    await writer.drain()
            except json.JSONDecodeError:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                writer.write((json.dumps(error_resp) + "\n").encode())
                await writer.drain()
    finally:
        await browser.close()


async def main_http(host: str = "127.0.0.1", port: int = 0):
    """Run the MCP server over HTTP so traffic traverses the network stack.

    Uses aiohttp for async HTTP handling compatible with Playwright's
    async API.  Falls back to a threaded sync server if aiohttp is not
    installed.

    When *port* is 0 the OS picks a free port.  The chosen port is printed
    to stdout as ``PORT=<n>`` so the parent process can discover it.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
    import threading

    # We need to run the async handle_request from sync handler threads.
    # Create a dedicated event loop running in a background thread.
    _loop = asyncio.new_event_loop()
    _thread = threading.Thread(target=_loop.run_forever, daemon=True)
    _thread.start()

    # Launch browser on the background loop
    asyncio.run_coroutine_threadsafe(browser.launch(), _loop).result(timeout=30)

    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    class MCPHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                request = json.loads(body)
                future = asyncio.run_coroutine_threadsafe(
                    handle_request(request), _loop
                )
                response = future.result(timeout=60)

                if response is None:
                    self.send_response(204)
                    self.end_headers()
                    return

                payload = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except json.JSONDecodeError:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                payload = json.dumps(error_resp).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass  # suppress per-request logging

    server = ThreadingHTTPServer((host, port), MCPHandler)
    actual_port = server.server_address[1]
    sys.stdout.write(f"PORT={actual_port}\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    finally:
        asyncio.run_coroutine_threadsafe(browser.close(), _loop).result(timeout=10)
        _loop.call_soon_threadsafe(_loop.stop)


def main():
    """Entry point -- stdio by default, --http for network transport."""
    if "--http" in sys.argv:
        host = "127.0.0.1"
        port = 0
        for arg in sys.argv[1:]:
            if arg.startswith("--host="):
                host = arg.split("=", 1)[1]
            elif arg.startswith("--port="):
                port = int(arg.split("=", 1)[1])
        asyncio.run(main_http(host, port))
    else:
        asyncio.run(main_stdio())


if __name__ == "__main__":
    main()
