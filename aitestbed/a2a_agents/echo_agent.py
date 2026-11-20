"""A deterministic leaf A2A agent for the testbed.

Run as a subprocess::

    python -m a2a_agents.echo_agent --port 9001 --name "Echo Agent" \
        --behavior echo --stream-chunks 4 --stream-delay-ms 40

Supports both ``message/send`` (non-streaming) and ``message/stream`` (SSE):
the reply is emitted as several appended artifact chunks so streaming clients
observe multiple events with measurable inter-chunk timing.
"""

from __future__ import annotations

import argparse
import asyncio

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.helpers import new_text_part, new_task_from_user_message

from .common import build_agent_card, run_agent_server, chunk_words


def _transform(text: str, behavior: str) -> str:
    if behavior == "upper":
        return text.upper()
    if behavior == "reverse":
        return text[::-1]
    # default: echo, prefixed so the caller can tell the reply apart
    return f"echo: {text}"


class EchoAgentExecutor(AgentExecutor):
    """Echo/transform the user input, optionally streamed in chunks."""

    def __init__(
        self,
        behavior: str = "echo",
        stream_chunks: int = 4,
        stream_delay_ms: int = 40,
    ):
        self.behavior = behavior
        self.stream_chunks = max(1, stream_chunks)
        self.stream_delay_ms = max(0, stream_delay_ms)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        task = context.current_task
        if task is None:
            # A Task must be enqueued before any status/artifact update.
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        reply = _transform(text, self.behavior)
        pieces = chunk_words(reply, self.stream_chunks)
        for i, piece in enumerate(pieces):
            part = (" " if i > 0 else "") + piece
            await updater.add_artifact(
                [new_text_part(part)],
                artifact_id="reply",
                name="reply",
                append=i > 0,
                last_chunk=i == len(pieces) - 1,
            )
            if self.stream_delay_ms and i < len(pieces) - 1:
                await asyncio.sleep(self.stream_delay_ms / 1000.0)

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Nothing to cancel for this synchronous, fast agent.
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Testbed echo A2A agent")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--name", default="Echo Agent")
    parser.add_argument(
        "--behavior", default="echo", choices=["echo", "upper", "reverse"]
    )
    parser.add_argument("--stream-chunks", type=int, default=4)
    parser.add_argument("--stream-delay-ms", type=int, default=40)
    parser.add_argument(
        "--public-url",
        default=None,
        help="Externally-reachable base URL to advertise in the Agent Card "
             "(e.g. a cloudflared/ngrok tunnel). Defaults to http://host:port/.",
    )
    args = parser.parse_args()

    card = build_agent_card(
        name=args.name,
        description=f"Deterministic {args.behavior} agent for testbed measurement",
        port=args.port,
        host=args.host,
        skill_id=args.behavior,
        public_url=args.public_url,
    )
    executor = EchoAgentExecutor(
        behavior=args.behavior,
        stream_chunks=args.stream_chunks,
        stream_delay_ms=args.stream_delay_ms,
    )
    run_agent_server(executor, card, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
