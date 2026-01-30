"""
Google Gemini client adapter for the 6G AI Traffic Testbed.
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
    ImageResponse,
    ToolCall,
    MessageRole,
    estimate_payload_bytes,
)


class GeminiClient(LLMClient):
    """
    Google Gemini API client adapter with traffic metrics collection.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
        """
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai"
            )

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

        self._models = {}  # Cache for model instances

    @property
    def provider(self) -> str:
        return "gemini"

    def _get_model(self, model_name: str):
        """Get or create a GenerativeModel instance."""
        if model_name not in self._models:
            self._models[model_name] = self._genai.GenerativeModel(model_name)
        return self._models[model_name]

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage list to Gemini format."""
        gemini_messages = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Gemini handles system prompts differently
                # Prepend to first user message or add as context
                continue
            elif msg.role == MessageRole.USER:
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == MessageRole.ASSISTANT:
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == MessageRole.TOOL:
                # Tool responses in Gemini format
                gemini_messages.append({
                    "role": "function",
                    "parts": [{"text": msg.content}]
                })
        return gemini_messages

    def _get_system_instruction(self, messages: list[ChatMessage]) -> Optional[str]:
        """Extract system instruction from messages."""
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                return msg.content
        return None

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> ChatResponse | Iterator[str]:
        """
        Send a chat completion request to Gemini.
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
        gemini_model = self._get_model(model)

        # Convert messages
        gemini_messages = self._convert_messages(messages)

        t_start = time.time()

        system_instruction = self._get_system_instruction(messages)

        # Create chat session or generate directly
        if len(gemini_messages) == 1:
            # Single message - use generate_content
            prompt_text = gemini_messages[0]["parts"][0]["text"]
            request_bytes = estimate_payload_bytes({
                "model": model,
                "input": prompt_text,
                **kwargs,
            })
            request_payload = {
                "format": "gemini.generate_content",
                "payload": {
                    "model": model,
                    "input": prompt_text,
                    "system_instruction": system_instruction,
                    **kwargs,
                },
            }
            response = gemini_model.generate_content(prompt_text, **kwargs)
        else:
            # Multi-turn - use chat
            chat = gemini_model.start_chat(history=gemini_messages[:-1])
            last_msg = gemini_messages[-1]["parts"][0]["text"]
            request_bytes = estimate_payload_bytes({
                "model": model,
                "history": gemini_messages[:-1],
                "message": last_msg,
                **kwargs,
            })
            request_payload = {
                "format": "gemini.chat.send_message",
                "payload": {
                    "model": model,
                    "history": gemini_messages[:-1],
                    "message": last_msg,
                    "system_instruction": system_instruction,
                    **kwargs,
                },
            }
            response = chat.send_message(last_msg, **kwargs)

        t_end = time.time()

        # Extract content
        content = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text"):
                    content += part.text

        # Parse tool calls if present
        tool_calls: list[ToolCall] = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                func_call = getattr(part, "function_call", None)
                if not func_call:
                    continue
                name = getattr(func_call, "name", "") or "unknown"
                args = getattr(func_call, "args", {}) or {}
                if hasattr(args, "items"):
                    args = dict(args)
                tool_calls.append(ToolCall(
                    id=f"gemini_call_{len(tool_calls)}",
                    name=name,
                    arguments=args
                ))

        # Get usage metadata if available
        tokens_in = None
        tokens_out = None
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            tokens_in = getattr(usage, "prompt_token_count", None)
            tokens_out = getattr(usage, "candidates_token_count", None)

        # Estimate response size
        response_dump = response.to_dict() if hasattr(response, "to_dict") else None
        if response_dump is not None:
            response_bytes = estimate_payload_bytes(response_dump)
        else:
            response_bytes = len(content.encode("utf-8"))
            response_dump = {"content": content}

        if tokens_in is None:
            tokens_in = self.estimate_message_tokens(messages, model)
        if tokens_out is None:
            tokens_out = self.estimate_tokens(content, model)

        return ChatResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=(tokens_in or 0) + (tokens_out or 0) if tokens_in or tokens_out else None,
            latency_sec=t_end - t_start,
            model=model,
            tool_calls=tool_calls,
            raw_response=response,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            request_payload=request_payload,
            response_payload={
                "format": "gemini.response",
                "payload": response_dump,
            },
        )

    def _chat_stream_generator(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs
    ) -> Iterator[str]:
        """Generator that yields content chunks from a streaming response."""
        gemini_model = self._get_model(model)
        gemini_messages = self._convert_messages(messages)

        if len(gemini_messages) == 1:
            response = gemini_model.generate_content(
                gemini_messages[0]["parts"][0]["text"],
                stream=True,
                **kwargs
            )
        else:
            chat = gemini_model.start_chat(history=gemini_messages[:-1])
            last_msg = gemini_messages[-1]["parts"][0]["text"]
            response = chat.send_message(last_msg, stream=True, **kwargs)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def chat_streaming(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs
    ) -> StreamingResponse:
        """
        Send a streaming chat request and collect all chunks with timing metrics.
        """
        gemini_model = self._get_model(model)
        gemini_messages = self._convert_messages(messages)

        if len(gemini_messages) == 1:
            prompt_text = gemini_messages[0]["parts"][0]["text"]
            request_bytes = estimate_payload_bytes({
                "model": model,
                "input": prompt_text,
                "stream": True,
                **kwargs,
            })
            request_payload = {
                "format": "gemini.generate_content",
                "payload": {
                    "model": model,
                    "input": prompt_text,
                    "stream": True,
                    **kwargs,
                },
            }
        else:
            last_msg = gemini_messages[-1]["parts"][0]["text"]
            request_bytes = estimate_payload_bytes({
                "model": model,
                "history": gemini_messages[:-1],
                "message": last_msg,
                "stream": True,
                **kwargs,
            })
            request_payload = {
                "format": "gemini.chat.send_message",
                "payload": {
                    "model": model,
                    "history": gemini_messages[:-1],
                    "message": last_msg,
                    "stream": True,
                    **kwargs,
                },
            }

        streaming_response = StreamingResponse()
        streaming_response.t_request_start = time.time()
        streaming_response.request_bytes = request_bytes
        streaming_response.model = model
        streaming_response.request_payload = request_payload

        if len(gemini_messages) == 1:
            response = gemini_model.generate_content(
                gemini_messages[0]["parts"][0]["text"],
                stream=True,
                **kwargs
            )
        else:
            chat = gemini_model.start_chat(history=gemini_messages[:-1])
            last_msg = gemini_messages[-1]["parts"][0]["text"]
            response = chat.send_message(last_msg, stream=True, **kwargs)

        for chunk in response:
            now = time.time()
            if hasattr(chunk, "to_dict"):
                chunk_dump = chunk.to_dict()
                chunk_bytes = estimate_payload_bytes(chunk_dump)
            elif hasattr(chunk, "model_dump"):
                chunk_dump = chunk.model_dump()
                chunk_bytes = estimate_payload_bytes(chunk_dump)
            else:
                chunk_dump = str(chunk)
                chunk_bytes = estimate_payload_bytes(str(chunk))
            streaming_response.response_events.append({
                "format": "gemini.stream.chunk",
                "timestamp": now,
                "bytes": chunk_bytes,
                "payload": chunk_dump,
            })

            if chunk.text:
                streaming_response.add_chunk(chunk.text, chunk_bytes=chunk_bytes, timestamp=now)
            else:
                streaming_response.add_event_bytes(chunk_bytes)

        streaming_response.tokens_in = self.estimate_message_tokens(messages, model)
        streaming_response.tokens_out = self.estimate_tokens(streaming_response.total_content, model)
        streaming_response.response_payload = {
            "format": "gemini.response_summary",
            "payload": {
                "content": streaming_response.total_content,
                "tokens_in": streaming_response.tokens_in,
                "tokens_out": streaming_response.tokens_out,
            },
        }

        return streaming_response

    def generate_content_with_image(
        self,
        prompt: str,
        image_path: str,
        model: str = "gemini-3-flash-preview",
        **kwargs
    ) -> ChatResponse:
        """
        Generate content from an image and text prompt (multimodal).

        Args:
            prompt: Text prompt
            image_path: Path to the image file
            model: Model identifier
        """
        import PIL.Image

        gemini_model = self._get_model(model)

        # Load image
        image = PIL.Image.open(image_path)

        # Estimate request size (rough approximation)
        with open(image_path, "rb") as f:
            image_bytes = len(f.read())
        request_bytes = len(prompt.encode("utf-8")) + image_bytes

        t_start = time.time()

        response = gemini_model.generate_content([prompt, image], **kwargs)

        t_end = time.time()

        content = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text"):
                    content += part.text

        tokens_in = None
        tokens_out = None
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            tokens_in = getattr(usage, "prompt_token_count", None)
            tokens_out = getattr(usage, "candidates_token_count", None)
        if tokens_in is None:
            tokens_in = self.estimate_tokens(prompt, model)
        if tokens_out is None:
            tokens_out = self.estimate_tokens(content, model)

        response_dump = response.to_dict() if hasattr(response, "to_dict") else None
        response_bytes = len(content.encode("utf-8"))
        if response_dump is not None:
            response_bytes = estimate_payload_bytes(response_dump)

        return ChatResponse(
            content=content,
            latency_sec=t_end - t_start,
            model=model,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            request_payload={
                "format": "gemini.generate_content",
                "payload": {
                    "model": model,
                    "input": [prompt, f"image:{image_path}"],
                    **kwargs,
                },
            },
            response_payload={
                "format": "gemini.response",
                "payload": response_dump or {"content": content},
            },
        )
