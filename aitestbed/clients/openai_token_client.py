"""
OpenAI token client adapter for the 6G AI Traffic Testbed.
"""

import os
import time
import json
from typing import Iterator, Optional

from openai import OpenAI

from .base import (
    LLMClient,
    ChatMessage,
    ChatResponse,
    StreamingResponse,
    estimate_payload_bytes,
)


class OpenAITokenClient(LLMClient):
    """
    OpenAI API client using token ID
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        """
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._last_request_bytes = 0
        self._last_response_bytes = 0

    @property
    def provider(self) -> str:
        return "openai_token"

    def chat(
        self,
        messages: list[int],
        model: str,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> ChatResponse | Iterator[str]:
        """
        Send a completion request to OpenAI.
        """
        if stream:
            return self._chat_stream_generator(messages, model, **kwargs)
        else:
            return self._chat_sync(messages, model, tools, **kwargs)

    def _chat_sync(
        self,
        messages: list[int],
        model: str,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Synchronous (non-streaming) completion."""
        t_start = time.time()

        # Build request parameters
        params = {
            "model": model,
            "prompt": messages,
            **kwargs
        }
        # Completions API does not support tools
        if "tools" in params:
            del params["tools"]
        request_bytes = estimate_payload_bytes(params)
        request_payload = {
            "format": "openai.completions.create",
            "payload": params,
        }

        response = self.client.completions.create(**params)

        t_end = time.time()

        # Extract response data
        content = response.choices[0].text or ""
        usage = response.usage

        # Tool calls are not available
        tool_calls = []

        # Estimate response size at scenario
        tokens_in = usage.prompt_tokens if usage else None
        tokens_out = usage.completion_tokens if usage else None
        if tokens_in is None:
            tokens_in = self.estimate_token_count(messages)
        if tokens_out is None:
            tokens_out = self.estimate_token_count(content, model)

        return ChatResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=usage.total_tokens if usage else None,
            latency_sec=t_end - t_start,
            model=response.model,
            tool_calls=tool_calls,
            raw_response=response,
            request_bytes=request_bytes,
            request_payload=request_payload,
            response_payload=response, # return all response for scenario to calculate response size
        )

    def _chat_stream_generator(
        self,
        messages: list[int],
        model: str,
        **kwargs
    ) -> Iterator[str]:
        """Generator that yields content chunks from a streaming response."""
        response = self.client.completions.create(
            model=model,
            prompt=messages,
            stream=True,
            **kwargs
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text

    def chat_streaming(
        self,
        messages: list[int],
        model: str,
        **kwargs
    ) -> StreamingResponse:
        """
        Send a streaming completion request and collect all chunks with timing metrics.
        """

        params = {
            "model": model,
            "prompt": messages,
            "stream": True,
            **kwargs
        }
        if "stream_options" not in params:
            params["stream_options"] = {"include_usage": True}
        request_bytes = estimate_payload_bytes(params) # todo: token IDs should not be treated as str
        request_payload = {
            "format": "openai.completions.create",
            "payload": params,
        }

        streaming_response = StreamingResponse()
        streaming_response.t_request_start = time.time()
        streaming_response.request_bytes = request_bytes
        streaming_response.model = model
        streaming_response.request_payload = request_payload

        response = self.client.completions.create(**params)

        for chunk in response:
            now = time.time()
            chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            chunk_bytes = estimate_payload_bytes(chunk_dict)
            streaming_response.response_events.append({
                "format": "openai.completions.chunk",
                "timestamp": now,
                "bytes": chunk_bytes,
                "payload": chunk_dict,
            })

            if getattr(chunk, "usage", None):
                usage = chunk.usage
                streaming_response.tokens_in = getattr(usage, "prompt_tokens", None)
                streaming_response.tokens_out = getattr(usage, "completion_tokens", None)
                streaming_response.add_event_bytes(chunk_bytes)
                continue

            if chunk.choices and chunk.choices[0].text:
                content = chunk.choices[0].text
                streaming_response.add_chunk(content, chunk_bytes=chunk_bytes, timestamp=now)
            else:
                streaming_response.add_event_bytes(chunk_bytes)

        if streaming_response.tokens_in is None:
            streaming_response.tokens_in = self.estimate_token_count(messages)
        if streaming_response.tokens_out is None:
            streaming_response.tokens_out = self.estimate_token_count(streaming_response.total_content, model)
        return streaming_response

    def chat_with_tools(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: list[dict],
        tool_executor: callable,
        max_iterations: int = 5,
        **kwargs
    ) -> tuple[ChatResponse, list[dict]]:
        """
        tool calling not supported, only response messages.
        """
        tool_records = []
        current_messages = list(messages)
        response = self._chat_sync(current_messages, model, tools=None, **kwargs)
        return response, tool_records

    def estimate_token_count(self, messages, model = None):
        """Estimate token count """
        if model:
            token_count = len(messages) //4
        else:
            token_count = len(messages)
        return token_count


# Import MessageRole for the chat_with_tools method
from .base import MessageRole
