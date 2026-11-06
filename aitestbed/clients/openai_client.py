"""
OpenAI client adapter for the 6G AI Traffic Testbed.
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
    ImageResponse,
    ToolCall,
    estimate_payload_bytes,
)


class OpenAIClient(LLMClient):
    """
    OpenAI API client adapter with traffic metrics collection.
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
        return "openai"

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> ChatResponse | Iterator[str]:
        """
        Send a chat completion request to OpenAI.
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
            "format": "openai.chat.completions.create",
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
            "format": "openai.chat.completions.response",
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
        if "stream_options" not in params:
            params["stream_options"] = {"include_usage": True}
        request_bytes = estimate_payload_bytes(params)
        request_payload = {
            "format": "openai.chat.completions.create",
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
                "format": "openai.chat.completions.chunk",
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
            "format": "openai.chat.completions.response_summary",
            "payload": {
                "content": streaming_response.total_content,
                "tokens_in": streaming_response.tokens_in,
                "tokens_out": streaming_response.tokens_out,
            },
        }

        return streaming_response

    def generate_image(
        self,
        prompt: str,
        model: str = "gpt-image-1.5",
        size: str = "1024x1024",
        quality: str = "standard",
        response_format: str = "b64_json",
        **kwargs
    ) -> ImageResponse:
        """
        Generate an image using DALL-E or GPT image models.

        Args:
            prompt: Text description of the image
            model: Model to use (dall-e-2, dall-e-3, gpt-image-1, gpt-image-1.5)
            size: Image size (1024x1024, 1024x1792, 1792x1024)
            quality: "standard", "low", "medium", or "high"
            response_format: "url" or "b64_json" (only for dall-e models)
        """
        # Determine if model supports response_format parameter
        # gpt-image-* models don't support response_format, they return URLs only
        supports_response_format = model.startswith("dall-e")

        # Build request payload for size estimation
        request_data = {
            "model": model,
            "prompt": prompt,
            "size": size,
        }

        # Build API parameters
        api_params = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": 1,
        }

        # Add quality parameter (different values for different models)
        if model.startswith("gpt-image"):
            # gpt-image models use: low, medium, high
            quality_mapping = {"standard": "medium", "hd": "high"}
            api_params["quality"] = quality_mapping.get(quality, quality)
            request_data["quality"] = api_params["quality"]
        else:
            # dall-e models use: standard, hd
            api_params["quality"] = quality
            request_data["quality"] = quality

        # Add response_format only for models that support it
        if supports_response_format:
            api_params["response_format"] = response_format
            request_data["response_format"] = response_format

        request_payload = {
            "format": "openai.images.generate",
            "payload": api_params,
        }
        request_bytes = len(json.dumps(request_data).encode("utf-8"))

        t_start = time.time()

        response = self.client.images.generate(**api_params, **kwargs)

        t_end = time.time()

        image_data = response.data[0]
        revised_prompt = getattr(image_data, "revised_prompt", None)

        # Get image data based on what's available
        import base64
        if hasattr(image_data, 'b64_json') and image_data.b64_json:
            image_bytes = base64.b64decode(image_data.b64_json)
            response_bytes = len(image_data.b64_json)
            image_url = None
        elif hasattr(image_data, 'url') and image_data.url:
            image_bytes = None
            image_url = image_data.url
            response_bytes = len(image_data.url)
        else:
            image_bytes = None
            image_url = None
            response_bytes = 0

        response_dump = response.model_dump() if hasattr(response, "model_dump") else response

        return ImageResponse(
            image_data=image_bytes,
            image_url=image_url,
            revised_prompt=revised_prompt,
            latency_sec=t_end - t_start,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            request_payload=request_payload,
            response_payload=response_dump,
        )

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
                    role=MessageRole.ASSISTANT if hasattr(MessageRole, 'ASSISTANT') else "assistant",
                    content=response.content or "",
                ))

                # Add tool result
                current_messages.append(ChatMessage(
                    role=MessageRole.TOOL if hasattr(MessageRole, 'TOOL') else "tool",
                    content=json.dumps(result) if isinstance(result, dict) else str(result),
                    tool_call_id=tool_call.id,
                ))

        # Max iterations reached
        return response, tool_records


# Import MessageRole for the chat_with_tools method
from .base import MessageRole
