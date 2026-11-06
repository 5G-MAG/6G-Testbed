"""
Google Gemini client adapter for the 6G AI Traffic Testbed.

Uses the unified google-genai SDK (replacement for the deprecated
google-generativeai package).
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

    Uses the google-genai SDK (google.genai).
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY
                     or GEMINI_API_KEY env var.
        """
        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
        except ImportError:
            raise ImportError(
                "google-genai package is required. "
                "Install with: pip install google-genai"
            )

        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)

    @property
    def provider(self) -> str:
        return "gemini"

    def _convert_contents(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage list to Gemini contents format."""
        contents = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System instructions are passed separately via config
                continue
            elif msg.role == MessageRole.USER:
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == MessageRole.ASSISTANT:
                contents.append({
                    "role": "model",
                    "parts": [{"text": msg.content}]
                })
            elif msg.role == MessageRole.TOOL:
                contents.append({
                    "role": "function",
                    "parts": [{"text": msg.content}]
                })
        return contents

    def _get_system_instruction(self, messages: list[ChatMessage]) -> Optional[str]:
        """Extract system instruction from messages."""
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                return msg.content
        return None

    def _build_config(self, system_instruction: Optional[str] = None, **kwargs):
        """Build a GenerateContentConfig with optional system instruction."""
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        # Pass through any supported config params
        for key in ("temperature", "top_p", "top_k", "max_output_tokens",
                     "stop_sequences", "candidate_count"):
            if key in kwargs:
                config_kwargs[key] = kwargs.pop(key)
        if config_kwargs:
            return self._types.GenerateContentConfig(**config_kwargs)
        return None

    def _response_to_dict(self, response) -> Optional[dict]:
        """Safely convert a response to a dict for payload logging."""
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump()
            except Exception:
                pass
        if hasattr(response, "to_dict"):
            try:
                return response.to_dict()
            except Exception:
                pass
        if hasattr(response, "__dict__"):
            try:
                return {k: str(v) for k, v in response.__dict__.items()
                        if not k.startswith("_")}
            except Exception:
                pass
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
        contents = self._convert_contents(messages)
        system_instruction = self._get_system_instruction(messages)
        config = self._build_config(system_instruction=system_instruction, **kwargs)

        request_payload = {
            "format": "genai.models.generate_content",
            "payload": {
                "model": model,
                "contents": contents,
                "system_instruction": system_instruction,
            },
        }
        request_bytes = estimate_payload_bytes(request_payload["payload"])

        t_start = time.time()

        generate_kwargs = {"model": model, "contents": contents}
        if config:
            generate_kwargs["config"] = config

        response = self._client.models.generate_content(**generate_kwargs)

        t_end = time.time()

        # Extract content
        content = response.text or ""

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

        # Get usage metadata
        tokens_in = None
        tokens_out = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            tokens_in = getattr(usage, "prompt_token_count", None)
            tokens_out = getattr(usage, "candidates_token_count", None)

        # Estimate response size
        response_dump = self._response_to_dict(response)
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
                "format": "genai.response",
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
        contents = self._convert_contents(messages)
        system_instruction = self._get_system_instruction(messages)
        config = self._build_config(system_instruction=system_instruction, **kwargs)

        generate_kwargs = {"model": model, "contents": contents}
        if config:
            generate_kwargs["config"] = config

        for chunk in self._client.models.generate_content_stream(**generate_kwargs):
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
        contents = self._convert_contents(messages)
        system_instruction = self._get_system_instruction(messages)
        config = self._build_config(system_instruction=system_instruction, **kwargs)

        request_payload = {
            "format": "genai.models.generate_content_stream",
            "payload": {
                "model": model,
                "contents": contents,
                "stream": True,
                "system_instruction": system_instruction,
            },
        }
        request_bytes = estimate_payload_bytes(request_payload["payload"])

        streaming_response = StreamingResponse()
        streaming_response.t_request_start = time.time()
        streaming_response.request_bytes = request_bytes
        streaming_response.model = model
        streaming_response.request_payload = request_payload

        generate_kwargs = {"model": model, "contents": contents}
        if config:
            generate_kwargs["config"] = config

        for chunk in self._client.models.generate_content_stream(**generate_kwargs):
            now = time.time()
            chunk_dump = self._response_to_dict(chunk)
            if chunk_dump is not None:
                chunk_bytes = estimate_payload_bytes(chunk_dump)
            else:
                chunk_dump = str(chunk)
                chunk_bytes = estimate_payload_bytes(chunk_dump)
            streaming_response.response_events.append({
                "format": "genai.stream.chunk",
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
            "format": "genai.response_summary",
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

        # Load image
        image = PIL.Image.open(image_path)

        # Estimate request size (rough approximation)
        with open(image_path, "rb") as f:
            image_bytes = len(f.read())
        request_bytes = len(prompt.encode("utf-8")) + image_bytes

        system_instruction = kwargs.pop("system_instruction", None)
        config = self._build_config(system_instruction=system_instruction, **kwargs)

        t_start = time.time()

        generate_kwargs = {
            "model": model,
            "contents": [prompt, image],
        }
        if config:
            generate_kwargs["config"] = config

        response = self._client.models.generate_content(**generate_kwargs)

        t_end = time.time()

        content = response.text or ""

        tokens_in = None
        tokens_out = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            tokens_in = getattr(usage, "prompt_token_count", None)
            tokens_out = getattr(usage, "candidates_token_count", None)
        if tokens_in is None:
            tokens_in = self.estimate_tokens(prompt, model)
        if tokens_out is None:
            tokens_out = self.estimate_tokens(content, model)

        response_dump = self._response_to_dict(response)
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
                "format": "genai.models.generate_content",
                "payload": {
                    "model": model,
                    "contents": [prompt, f"image:{image_path}"],
                },
            },
            response_payload={
                "format": "genai.response",
                "payload": response_dump or {"content": content},
            },
        )

    def generate_content_with_document(
        self,
        prompt: str,
        document_path: str,
        model: str = "gemini-3-flash-preview",
        **kwargs,
    ) -> ChatResponse:
        """
        Ask a question about a PDF using Gemini's inline_data path.

        The PDF bytes are sent as an inline Part with mime_type application/pdf.
        For files >~20 MB the File API (upload then reference) is preferred, but
        inline produces the uplink traffic signature we want to characterize.
        """
        import mimetypes

        with open(document_path, "rb") as f:
            raw = f.read()
        raw_bytes = len(raw)
        mime = mimetypes.guess_type(document_path)[0] or "application/pdf"

        doc_part = self._types.Part.from_bytes(data=raw, mime_type=mime)

        system_instruction = kwargs.pop("system_instruction", None)
        config = self._build_config(system_instruction=system_instruction, **kwargs)

        generate_kwargs = {
            "model": model,
            "contents": [prompt, doc_part],
        }
        if config:
            generate_kwargs["config"] = config

        request_bytes = len(prompt.encode("utf-8")) + raw_bytes

        t_start = time.time()
        response = self._client.models.generate_content(**generate_kwargs)
        t_end = time.time()

        content = response.text or ""

        tokens_in = None
        tokens_out = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            tokens_in = getattr(usage, "prompt_token_count", None)
            tokens_out = getattr(usage, "candidates_token_count", None)
        if tokens_in is None:
            tokens_in = self.estimate_tokens(prompt, model)
        if tokens_out is None:
            tokens_out = self.estimate_tokens(content, model)

        response_dump = self._response_to_dict(response)
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
                "format": "genai.models.generate_content",
                "payload": {
                    "model": model,
                    "contents": [
                        prompt,
                        {
                            "inline_data": {
                                "mime_type": mime,
                                "document_path": document_path,
                                "raw_bytes": raw_bytes,
                            }
                        },
                    ],
                },
            },
            response_payload={
                "format": "genai.response",
                "payload": response_dump or {"content": content},
            },
        )
