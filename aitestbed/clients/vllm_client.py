"""
vLLM client adapter for the 6G AI Traffic Testbed.

vLLM exposes an OpenAI-compatible API. This client leverages the OpenAI SDK
with a configurable base URL to target a vLLM server.
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
    estimate_payload_bytes,
)


class VLLMClient(LLMClient):
    """
    vLLM API client adapter with traffic metrics collection.

    Uses the OpenAI-compatible API endpoint exposed by vLLM.
    """

    DEFAULT_BASE_URL = "http://localhost:8000/v1"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the vLLM client.

        Args:
            api_key: Optional API key (if your vLLM deployment requires it).
            base_url: Base URL for the vLLM OpenAI-compatible API.
        """
        api_key = api_key or os.environ.get("VLLM_API_KEY") or "vllm"
        base_url = base_url or os.environ.get("VLLM_BASE_URL") or self.DEFAULT_BASE_URL
        base_url = self._normalize_base_url(base_url)

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def provider(self) -> str:
        return "vllm"

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> ChatResponse | Iterator[str]:
        if stream:
            return self._chat_stream_generator(messages, model, **kwargs)
        return self._chat_sync(messages, model, tools, **kwargs)

    def _chat_sync(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> ChatResponse:
        api_messages = [m.to_dict() for m in messages]

        params = {
            "model": model,
            "messages": api_messages,
            **kwargs,
        }
        if tools:
            params["tools"] = tools

        request_bytes = estimate_payload_bytes(params)
        request_payload = {
            "format": "vllm.chat.completions.create",
            "payload": params,
        }

        t_start = time.time()
        response = self.client.chat.completions.create(**params)
        t_end = time.time()

        content = self._extract_message_content(response.choices[0].message.content)
        usage = response.usage

        tool_calls: list[ToolCall] = []
        if response.choices and response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        response_dump = response.model_dump() if hasattr(response, "model_dump") else response
        response_bytes = estimate_payload_bytes(response_dump)
        response_payload = {
            "format": "vllm.chat.completions.response",
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
            model=response.model if hasattr(response, "model") else model,
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
        api_messages = [m.to_dict() for m in messages]

        response = self.client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
            **kwargs,
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
        api_messages = [m.to_dict() for m in messages]

        params = {
            "model": model,
            "messages": api_messages,
            "stream": True,
            **kwargs,
        }
        request_bytes = estimate_payload_bytes(params)
        request_payload = {
            "format": "vllm.chat.completions.create",
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
                "format": "vllm.chat.completions.chunk",
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
            "format": "vllm.chat.completions.response_summary",
            "payload": {
                "content": streaming_response.total_content,
                "tokens_in": streaming_response.tokens_in,
                "tokens_out": streaming_response.tokens_out,
            },
        }

        return streaming_response

    def generate_content_with_video(
        self,
        prompt: str,
        video_url: str,
        model: str,
        **kwargs
    ) -> ChatResponse:
        """
        Generate content from a video + text prompt using OpenAI-compatible
        message content with video_url.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        params = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        request_bytes = estimate_payload_bytes(params)
        request_payload = {
            "format": "vllm.chat.completions.create",
            "payload": params,
        }

        t_start = time.time()
        response = self.client.chat.completions.create(**params)
        t_end = time.time()

        content = self._extract_message_content(response.choices[0].message.content)
        usage = response.usage

        response_dump = response.model_dump() if hasattr(response, "model_dump") else response
        response_bytes = estimate_payload_bytes(response_dump)
        response_payload = {
            "format": "vllm.chat.completions.response",
            "payload": response_dump,
        }

        tokens_in = usage.prompt_tokens if usage else None
        tokens_out = usage.completion_tokens if usage else None
        if tokens_in is None:
            tokens_in = self.estimate_tokens(prompt, model)
        if tokens_out is None:
            tokens_out = self.estimate_tokens(content, model)

        return ChatResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=usage.total_tokens if usage else None,
            latency_sec=t_end - t_start,
            model=response.model if hasattr(response, "model") else model,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            request_payload=request_payload,
            response_payload=response_payload,
        )

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return base_url

    @staticmethod
    def _extract_message_content(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content)
