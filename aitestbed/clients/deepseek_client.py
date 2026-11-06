"""
DeepSeek client adapter for the 6G AI Traffic Testbed.

DeepSeek uses an OpenAI-compatible API, so this client leverages
the OpenAI SDK with a custom base URL.
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
    ToolCall,
    MessageRole,
    estimate_payload_bytes,
)


class DeepSeekClient(LLMClient):
    """
    DeepSeek API client adapter with traffic metrics collection.
    
    Uses the OpenAI-compatible API endpoint at api.deepseek.com.
    """

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the DeepSeek client.

        Args:
            api_key: DeepSeek API key. If not provided, uses DEEPSEEK_API_KEY env var.
            base_url: Optional custom base URL (defaults to api.deepseek.com).
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url or self.DEEPSEEK_BASE_URL
        )
        self._last_request_bytes = 0
        self._last_response_bytes = 0

    @property
    def provider(self) -> str:
        return "deepseek"

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> ChatResponse | Iterator[str]:
        """
        Send a chat completion request to DeepSeek.
        """
        if stream:
            return self._chat_stream_generator(messages, model, **kwargs)
        else:
            return self._chat_sync(messages, model, tools, **kwargs)

    def _chat_sync(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Synchronous (non-streaming) chat completion."""
        # Convert messages to API format
        api_messages = [m.to_dict() for m in messages]

        t_start = time.time()

        # Build request parameters
        params = {
            "model": model,
            "messages": api_messages,
            **kwargs
        }
        if tools:
            params["tools"] = tools
        request_bytes = estimate_payload_bytes(params)
        request_payload = {
            "format": "deepseek.chat.completions.create",
            "payload": params,
        }

        response = self.client.chat.completions.create(**params)

        t_end = time.time()

        # Extract response data
        content = response.choices[0].message.content or ""
        usage = response.usage

        # Parse tool calls if present
        tool_calls = []
        if response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args
                ))

        # Estimate response size
        response_dump = response.model_dump() if hasattr(response, "model_dump") else response
        response_bytes = estimate_payload_bytes(response_dump)
        response_payload = {
            "format": "deepseek.chat.completions.response",
            "payload": response_dump,
        }

        tokens_in = usage.prompt_tokens if usage else None
        tokens_out = usage.completion_tokens if usage else None
        if tokens_in is None:
            tokens_in = self.estimate_message_tokens(messages, model)
        if tokens_out is None:
            tokens_out = self.estimate_tokens(content, model)

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
            response_bytes=response_bytes,
            request_payload=request_payload,
            response_payload=response_payload,
        )

    def _chat_stream_generator(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs
    ) -> Iterator[str]:
        """Generator that yields content chunks from a streaming response."""
        api_messages = [m.to_dict() for m in messages]

        response = self.client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
            **kwargs
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_streaming(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs
    ) -> StreamingResponse:
        """
        Send a streaming chat request and collect all chunks with timing metrics.
        """
        api_messages = [m.to_dict() for m in messages]

        params = {
            "model": model,
            "messages": api_messages,
            "stream": True,
            **kwargs
        }
        request_bytes = estimate_payload_bytes(params)
        request_payload = {
            "format": "deepseek.chat.completions.create",
            "payload": params,
        }

        streaming_response = StreamingResponse()
        streaming_response.t_request_start = time.time()
        streaming_response.request_bytes = request_bytes
        streaming_response.model = model
        streaming_response.request_payload = request_payload

        response = self.client.chat.completions.create(**params)

        for chunk in response:
            now = time.time()
            chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            chunk_bytes = estimate_payload_bytes(chunk_dict)
            streaming_response.response_events.append({
                "format": "deepseek.chat.completions.chunk",
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

            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                streaming_response.add_chunk(content, chunk_bytes=chunk_bytes, timestamp=now)
            else:
                streaming_response.add_event_bytes(chunk_bytes)

        if streaming_response.tokens_in is None:
            streaming_response.tokens_in = self.estimate_message_tokens(messages, model)
        if streaming_response.tokens_out is None:
            streaming_response.tokens_out = self.estimate_tokens(streaming_response.total_content, model)
        streaming_response.response_payload = {
            "format": "deepseek.chat.completions.response_summary",
            "payload": {
                "content": streaming_response.total_content,
                "tokens_in": streaming_response.tokens_in,
                "tokens_out": streaming_response.tokens_out,
            },
        }

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
        Execute a chat with tool calling loop.

        Args:
            messages: Initial conversation messages
            model: Model identifier
            tools: Tool definitions in OpenAI format
            tool_executor: Function that executes tools: (name, args) -> result
            max_iterations: Maximum tool calling iterations

        Returns:
            Tuple of (final ChatResponse, list of tool execution records)
        """
        tool_records = []
        current_messages = list(messages)

        for iteration in range(max_iterations):
            response = self._chat_sync(current_messages, model, tools, **kwargs)

            if not response.tool_calls:
                # No more tool calls, return final response
                return response, tool_records

            # Execute each tool call
            for tool_call in response.tool_calls:
                t_tool_start = time.time()
                result = tool_executor(tool_call.name, tool_call.arguments)
                t_tool_end = time.time()

                tool_records.append({
                    "iteration": iteration,
                    "tool_name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": result,
                    "latency_sec": t_tool_end - t_tool_start,
                })

                # Add assistant message with tool call
                current_messages.append(ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                ))

                # Add tool result
                current_messages.append(ChatMessage(
                    role=MessageRole.TOOL,
                    content=json.dumps(result) if isinstance(result, dict) else str(result),
                    tool_call_id=tool_call.id,
                ))

        # Max iterations reached
        return response, tool_records
