"""Shared helpers for the local A2A agent servers (a2a-sdk 1.1.0).

Builds a JSON-RPC Agent Card, wires a Starlette app from the SDK route
builders, and runs it under uvicorn. Imported only by the agent entry-point
modules (which run as subprocesses), so the a2a-sdk import here never blocks
``import scenarios``.
"""

from __future__ import annotations

from typing import Callable

import uvicorn
from starlette.applications import Starlette

from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    AgentInterface)
from a2a.utils import TransportProtocol
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.routes import create_jsonrpc_routes, create_agent_card_routes


def build_agent_card(
    name: str,
    description: str,
    port: int,
    skill_id: str,
    host: str = "127.0.0.1",
    version: str = "0.1.0",
    public_url: str | None = None) -> AgentCard:
    """Construct a minimal JSON-RPC AgentCard.

    The card advertises the URL clients should connect to. By default that is
    ``http://{host}:{port}/`` (loopback). When the agent is reachable through a
    different externally-visible address (a tunnel or a VM's public host), pass
    ``public_url`` so the card advertises that instead, otherwise a remote
    client would fetch the card and then try to dial the local address.
    """
    advertised = (public_url or f"http://{host}:{port}/").rstrip("/") + "/"
    return AgentCard(
        name=name,
        description=description,
        version=version,
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id=skill_id,
                name=skill_id,
                description=description,
                tags=[skill_id])
        ],
        supported_interfaces=[
            AgentInterface(
                url=advertised,
                protocol_binding=TransportProtocol.JSONRPC)
        ])


def build_app(executor: AgentExecutor, card: AgentCard) -> Starlette:
    """Build a Starlette app exposing the agent card + JSON-RPC routes."""
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card)
    routes = create_agent_card_routes(card) + create_jsonrpc_routes(
        handler, rpc_url="/"
    )
    return Starlette(routes=routes)


def run_agent_server(
    executor: AgentExecutor,
    card: AgentCard,
    host: str = "127.0.0.1",
    port: int = 9001,
    log_level: str = "warning") -> None:
    """Block serving *executor* under uvicorn. Intended as a subprocess main."""
    app = build_app(executor, card)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def chunk_words(text: str, max_chunks: int) -> list[str]:
    """Split *text* into up to *max_chunks* whitespace-aligned pieces.

    Used to stream a reply as several artifact updates so streaming clients
    observe multiple events (and therefore meaningful inter-chunk timing).
    """
    words = text.split()
    if not words or max_chunks <= 1:
        return [text]
    n = min(max_chunks, len(words))
    size = (len(words) + n - 1) // n
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]
