"""
Azure AI Inference client adapter for the 6G AI Traffic Testbed.

Supports models deployed via Azure AI Studio / Model Catalog (Phi, Llama,
Mistral, etc.) using the ``azure-ai-inference`` SDK.
"""

import os
import time
import json
from typing import Iterator, Optional

from .base import (
    LLMClient,
    ChatMessage,
    ChatResponse,
    StreamingResponse,
    ToolCall,
    MessageRole,
    estimate_payload_bytes,
)


def _get_inference_classes():
    """Lazy-import azure.ai.inference classes with a helpful error message."""
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import (
            SystemMessage,
            UserMessage,
            AssistantMessage,
            ToolMessage,
            StreamingChatCompletionsUpdate,
        )
        from azure.core.credentials import AzureKeyCredential
    except ImportError as exc:
        raise ImportError(
            "azure-ai-inference is required for Azure AI Inference scenarios. "
            "Install it with:  pip install azure-ai-inference>=1.0.0b1"
        ) from exc
    return (
        ChatCompletionsClient,
        AzureKeyCredential,
        SystemMessage,
        UserMessage,
        AssistantMessage,
        ToolMessage,
        StreamingChatCompletionsUpdate,
    )


class AzureInferenceClient(LLMClient):
    """
    Azure AI Inference client adapter with traffic metrics collection.

    Works with serverless endpoints (model baked into the URL) and
    managed-compute deployments.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
    ):
        (
            ChatCompletionsClient,
            AzureKeyCredential,
            self._SystemMessage,
            self._UserMessage,
            self._AssistantMessage,
            self._ToolMessage,
            self._StreamingUpdate,
        ) = _get_inference_classes()

        endpoint = endpoint or os.environ.get("AZURE_INFERENCE_ENDPOINT")
        key = key or os.environ.get("AZURE_INFERENCE_KEY")
        if not endpoint:
            raise ValueError(
                "Azure AI Inference endpoint required. "
                "Set AZURE_INFERENCE_ENDPOINT or pass endpoint parameter."
            )

        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key or ""),
        )
        self._last_request_bytes = 0
        self._last_response_bytes = 0

    @property
    def provider(self) -> str:
        return "azure_inference"

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[ChatMessage]) -> list:
        """Map ChatMessage list to Azure Inference message types."""
        out = []
        for m in messages:
            role = m.role.value if isinstance(m.role, MessageRole) else m.role
            if role == "system":
                out.append(self._SystemMessage(content=m.content))
            elif role == "user":
                out.append(self._UserMessage(content=m.content))
            elif role == "assistant":
                out.append(self._AssistantMessage(content=m.content))
            elif role == "tool":
                out.append(self._ToolMessage(
                    content=m.content,
                    tool_call_id=m.tool_call_id or "",
                ))
            else:
                out.append(self._UserMessage(content=m.content))
        return out

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> ChatResponse | Iterator[str]:
        if stream:
            return self._chat_stream_generator(messages, model, **kwargs)
        return self._chat_sync(messages, model, tools, **kwargs)

    def _chat_sync(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ) -> ChatResponse:
        azure_messages = self._convert_messages(messages)

        params: dict = {"messages": azure_messages, **kwargs}
        if model:
            params["model"] = model
        if tools:
            params["tools"] = tools

        request_bytes = estimate_payload_bytes(
            {"model": model, "messages": [m.to_dict() for m in messages]}
        )
        request_payload = {
            "format": "azure.ai.inference.complete",
            "payload": {"model": model, "messages": [m.to_dict() for m in messages]},
        }

        t_start = time.time()
        response = self.client.complete(**params)
        t_end = time.time()

        content = response.choices[0].message.content or ""
        usage = response.usage

        # Tool calls
        tool_calls = []
        if getattr(response.choices[0].message, "tool_calls", None):
            for tc in response.choices[0].message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        response_dict = {
            "id": response.id,
            "model": getattr(response, "model", model),
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            } if usage else None,
        }
        response_bytes = estimate_payload_bytes(response_dict)
        response_payload = {
            "format": "azure.ai.inference.response",
            "payload": response_dict,
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
            model=getattr(response, "model", model) or model,
            tool_calls=tool_calls,
            raw_response=response,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            request_payload=request_payload,
            response_payload=response_payload,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _chat_stream_generator(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs,
    ) -> Iterator[str]:
        azure_messages = self._convert_messages(messages)
        params: dict = {"messages": azure_messages, "stream": True, **kwargs}
        if model:
            params["model"] = model

        response = self.client.complete(**params)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_streaming(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs,
    ) -> StreamingResponse:
        azure_messages = self._convert_messages(messages)

        params: dict = {"messages": azure_messages, "stream": True, **kwargs}
        if model:
            params["model"] = model

        request_bytes = estimate_payload_bytes(
            {"model": model, "messages": [m.to_dict() for m in messages]}
        )
        request_payload = {
            "format": "azure.ai.inference.complete",
            "payload": {"model": model, "messages": [m.to_dict() for m in messages]},
        }

        streaming_response = StreamingResponse()
        streaming_response.t_request_start = time.time()
        streaming_response.request_bytes = request_bytes
        streaming_response.model = model
        streaming_response.request_payload = request_payload

        response = self.client.complete(**params)

        for chunk in response:
            now = time.time()
            chunk_dict = {
                "id": getattr(chunk, "id", None),
                "choices": [
                    {"delta": {"content": c.delta.content if c.delta else None}}
                    for c in (chunk.choices or [])
                ],
            }
            chunk_bytes = estimate_payload_bytes(chunk_dict)
            streaming_response.response_events.append({
                "format": "azure.ai.inference.chunk",
                "timestamp": now,
                "bytes": chunk_bytes,
                "payload": chunk_dict,
            })

            usage = getattr(chunk, "usage", None)
            if usage:
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
            streaming_response.tokens_out = self.estimate_tokens(
                streaming_response.total_content, model
            )
        streaming_response.response_payload = {
            "format": "azure.ai.inference.response_summary",
            "payload": {
                "content": streaming_response.total_content,
                "tokens_in": streaming_response.tokens_in,
                "tokens_out": streaming_response.tokens_out,
            },
        }

        return streaming_response

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: list[ChatMessage],
        model: str,
        tools: list[dict],
        tool_executor: callable,
        max_iterations: int = 5,
        **kwargs,
    ) -> tuple[ChatResponse, list[dict]]:
        tool_records = []
        current_messages = list(messages)

        for iteration in range(max_iterations):
            response = self._chat_sync(current_messages, model, tools, **kwargs)

            if not response.tool_calls:
                return response, tool_records

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

                current_messages.append(ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.content or "",
                ))
                current_messages.append(ChatMessage(
                    role=MessageRole.TOOL,
                    content=json.dumps(result) if isinstance(result, dict) else str(result),
                    tool_call_id=tool_call.id,
                ))

        return response, tool_records
