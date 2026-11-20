"""An A2A agent that delegates to downstream A2A agents.

This agent is *also* an A2A client: on each incoming task it fans the user
input out to one or more downstream agents over A2A, aggregates their replies,
and returns the combined result. It exists to generate multi-hop
agent-to-agent traffic on the loopback interface for measurement.

Run as a subprocess::

    python -m a2a_agents.orchestrator_agent --port 9003 \
        --downstream http://127.0.0.1:9001,http://127.0.0.1:9002
"""

from __future__ import annotations

import argparse

import httpx

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.helpers import (
    new_text_part,
    new_text_message,
    new_task_from_user_message,
    get_stream_response_text)
from a2a.types import Role, SendMessageRequest
from a2a.client import create_client, ClientConfig

from .common import build_agent_card, run_agent_server


async def _ask_downstream(url: str, text: str) -> str:
    """Send *text* to a downstream A2A agent and return its concatenated text."""
    async with httpx.AsyncClient() as hx:
        client = await create_client(
            agent=url,
            client_config=ClientConfig(streaming=False, httpx_client=hx))
        try:
            req = SendMessageRequest(
                message=new_text_message(text, role=Role.ROLE_USER)
            )
            parts: list[str] = []
            async for resp in client.send_message(req):
                chunk = get_stream_response_text(resp)
                if chunk:
                    parts.append(chunk)
            return " ".join(parts).strip()
        finally:
            await client.close()


class OrchestratorAgentExecutor(AgentExecutor):
    """Delegate the task to each downstream agent and aggregate replies."""

    def __init__(self, downstream_urls: list[str]):
        self.downstream_urls = downstream_urls

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        results = []
        for url in self.downstream_urls:
            try:
                reply = await _ask_downstream(url, text)
            except Exception as exc:  # noqa: BLE001, surface as text, keep going
                reply = f"<error: {type(exc).__name__}: {exc}>"
            results.append(f"[{url}] {reply}")

        aggregate = " || ".join(results)
        await updater.add_artifact(
            [new_text_part(aggregate)], artifact_id="aggregate", name="aggregate"
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Testbed orchestrator A2A agent")
    parser.add_argument("--port", type=int, default=9003)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--name", default="Orchestrator Agent")
    parser.add_argument(
        "--downstream",
        required=True,
        help="Comma-separated downstream A2A base URLs")
    parser.add_argument(
        "--public-url",
        default=None,
        help="Externally-reachable base URL to advertise in the Agent Card "
             "(e.g. a cloudflared/ngrok tunnel). Defaults to http://host:port/.")
    args = parser.parse_args()

    downstream = [u.strip() for u in args.downstream.split(",") if u.strip()]
    card = build_agent_card(
        name=args.name,
        description="Delegates tasks to downstream A2A agents and aggregates replies",
        port=args.port,
        host=args.host,
        skill_id="delegate",
        public_url=args.public_url)
    executor = OrchestratorAgentExecutor(downstream_urls=downstream)
    run_agent_server(executor, card, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
