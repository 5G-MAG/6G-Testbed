"""Instrumented A2A (Agent2Agent protocol) client for the testbed.

Thin async wrapper around the official ``a2a-sdk`` that captures the
per-turn timing and byte metrics the testbed records for every other
transport (TTFT/TTLT, inter-chunk gaps, request/response bytes).

The ``a2a-sdk`` import is deferred to call time so that ``import
clients.a2a_client`` (and therefore ``import scenarios``) works even when the
SDK is not installed; only actually opening a session requires it.

Byte counters here are a client-side approximation derived from httpx event
hooks. For A2A the authoritative byte volumes come from the pcap (loopback for
local agents, egress for remote), see AGENTIC.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


def _headers_len(headers) -> int:
    """Rough on-wire size of an httpx Headers mapping."""
    try:
        return sum(len(str(k)) + len(str(v)) + 4 for k, v in headers.items())
    except Exception:
        return 0


@dataclass
class A2ATurnMetrics:
    """Per-turn (one message/send or message/stream) metrics."""

    t_request_start: float
    t_first_chunk: Optional[float] = None
    t_last_chunk: Optional[float] = None
    request_bytes: int = 0
    response_bytes: int = 0
    chunk_count: int = 0
    event_count: int = 0
    inter_chunk_times: list[float] = field(default_factory=list)
    output_text: str = ""
    streaming: bool = False
    success: bool = True
    error_message: Optional[str] = None

    def _record_event(self, now: float, has_text: bool) -> None:
        self.event_count += 1
        if self.t_first_chunk is None:
            self.t_first_chunk = now
        elif self.t_last_chunk is not None:
            self.inter_chunk_times.append(now - self.t_last_chunk)
        self.t_last_chunk = now
        if has_text:
            self.chunk_count += 1

    @property
    def ttft(self) -> Optional[float]:
        if self.t_first_chunk is None:
            return None
        return self.t_first_chunk - self.t_request_start

    @property
    def ttlt(self) -> Optional[float]:
        if self.t_last_chunk is None:
            return None
        return self.t_last_chunk - self.t_request_start

    @property
    def total_latency(self) -> float:
        end = self.t_last_chunk or time.time()
        return end - self.t_request_start


@dataclass
class A2ASessionMetrics:
    """Aggregate metrics across a session's turns."""

    base_url: str
    turns: int = 0
    total_request_bytes: int = 0
    total_response_bytes: int = 0
    agent_name: str = ""
    transport: str = ""


class A2AClientSession:
    """An async A2A client session against a single agent base URL.

    Usage::

        async with A2AClientSession(url, streaming=True) as sess:
            metrics = await sess.send("hello")
    """

    def __init__(
        self,
        base_url: str,
        streaming: bool = False,
        agent_card_path: Optional[str] = None,
        timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.streaming = streaming
        # Override the Agent Card path for agents that serve the legacy
        # '/.well-known/agent.json' instead of the SDK default
        # '/.well-known/agent-card.json' (e.g. the public hello-world agent).
        self.agent_card_path = agent_card_path
        # Generous default: remote agents may cold-start (free-tier hosts) and
        # high-latency/lossy profiles (satellite_geo, congested) blow past
        # httpx's 5s default during card fetch and message turns.
        self.timeout = timeout
        self.metrics = A2ASessionMetrics(base_url=self.base_url)
        self._httpx = None
        self._client = None
        # Filled with the most recent turn's byte counters via httpx hooks.
        self._cur: Optional[A2ATurnMetrics] = None

    async def __aenter__(self) -> "A2AClientSession":
        try:
            import httpx
            from a2a.client import create_client, ClientConfig
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "A2A scenarios require the 'a2a-sdk' package. "
                "Install it with: pip install a2a-sdk"
            ) from exc

        async def _on_request(request):
            if self._cur is not None:
                body = request.content or b""
                self._cur.request_bytes += len(body) + _headers_len(request.headers)

        async def _on_response(response):
            if self._cur is not None:
                self._cur.response_bytes += _headers_len(response.headers)
                cl = response.headers.get("content-length")
                if cl and str(cl).isdigit():
                    self._cur.response_bytes += int(cl)

        self._httpx = httpx.AsyncClient(
            timeout=self.timeout,
            event_hooks={"request": [_on_request], "response": [_on_response]})
        self._client = await create_client(
            agent=self.base_url,
            client_config=ClientConfig(
                streaming=self.streaming, httpx_client=self._httpx
            ),
            relative_card_path=self.agent_card_path)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
        if self._httpx is not None:
            try:
                await self._httpx.aclose()
            except Exception:
                pass
            self._httpx = None

    async def send(self, text: str) -> A2ATurnMetrics:
        """Send one message and collect timing/byte metrics over its events."""
        from a2a.helpers import new_text_message, get_stream_response_text
        from a2a.types import Role, SendMessageRequest

        turn = A2ATurnMetrics(t_request_start=time.time(), streaming=self.streaming)
        self._cur = turn
        try:
            req = SendMessageRequest(
                message=new_text_message(text, role=Role.ROLE_USER)
            )
            async for resp in self._client.send_message(req):
                now = time.time()
                chunk = get_stream_response_text(resp) or ""
                turn._record_event(now, has_text=bool(chunk))
                if chunk:
                    turn.output_text += chunk
            # Floor the response-byte estimate with the assembled text size
            # when the server streamed without content-length headers.
            turn.response_bytes = max(
                turn.response_bytes, len(turn.output_text.encode("utf-8"))
            )
        except Exception as exc:  # noqa: BLE001
            turn.success = False
            turn.error_message = f"{type(exc).__name__}: {exc}"
        finally:
            self._cur = None

        self.metrics.turns += 1
        self.metrics.total_request_bytes += turn.request_bytes
        self.metrics.total_response_bytes += turn.response_bytes
        return turn
