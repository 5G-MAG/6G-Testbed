"""
Anthropic (Claude) client adapter for the 6G AI Traffic Testbed.

Implements the LLMClient interface plus the multimodal helpers
(``generate_content_with_image``, ``generate_content_with_document``)
that the multimodal scenario expects.
"""

import base64
import mimetypes
import os
import time
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


class AnthropicClient(LLMClient):
    """Anthropic Claude client with traffic-metric collection."""

    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_key: Optional[str] = None):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY env var or pass api_key parameter."
            )
        self._client = Anthropic(api_key=api_key)

    @property
    def provider(self) -> str:
        return "anthropic"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[Optional[str], list[dict]]:
        """Anthropic takes ``system`` as a top-level argument, not as a role."""
        system = None
        out: list[dict] = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system = (system + "\n" + m.content) if system else m.content
            else:
                role = "assistant" if m.role == MessageRole.ASSISTANT else "user"
                out.append({"role": role, "content": m.content})
        return system, out

    @staticmethod
    def _response_to_dict(response) -> dict:
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump()
            except Exception:
                pass
        return {"content": getattr(response, "content", None), "model": getattr(response, "model", None)}

    @staticmethod
    def _extract_text(response) -> str:
        parts = getattr(response, "content", None) or []
        out = []
        for p in parts:
            if getattr(p, "type", None) == "text":
                out.append(getattr(p, "text", "") or "")
        return "".join(out)

    @staticmethod
    def _usage(response) -> tuple[Optional[int], Optional[int]]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None, None
        return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)

    # ------------------------------------------------------------------
    # Required base API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs,
    ):
        if stream:
            return self.chat_streaming(messages, model, **kwargs)

        system, api_messages = self._split_system(messages)
        params = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", self.DEFAULT_MAX_TOKENS),
            "messages": api_messages,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools
        params.update(kwargs)

        request_bytes = estimate_payload_bytes(params)
        request_payload = {"format": "anthropic.messages.create", "payload": params}

        t_start = time.time()
        response = self._client.messages.create(**params)
        t_end = time.time()

        content = self._extract_text(response)
        tokens_in, tokens_out = self._usage(response)
        response_dump = self._response_to_dict(response)
        response_bytes = estimate_payload_bytes(response_dump)

        return ChatResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=(tokens_in or 0) + (tokens_out or 0) if (tokens_in or tokens_out) else None,
            latency_sec=t_end - t_start,
            model=getattr(response, "model", model),
            raw_response=response,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            request_payload=request_payload,
            response_payload={"format": "anthropic.messages.response", "payload": response_dump},
        )

    def chat_streaming(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs,
    ) -> StreamingResponse:
        system, api_messages = self._split_system(messages)
        params = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", self.DEFAULT_MAX_TOKENS),
            "messages": api_messages,
        }
        if system:
            params["system"] = system
        params.update(kwargs)

        request_bytes = estimate_payload_bytes(params)
        sr = StreamingResponse()
        sr.t_request_start = time.time()
        sr.request_bytes = request_bytes
        sr.model = model
        sr.request_payload = {"format": "anthropic.messages.stream", "payload": params}

        with self._client.messages.stream(**params) as stream:
            for event in stream:
                etype = getattr(event, "type", "")
                event_dump = event.model_dump() if hasattr(event, "model_dump") else {"type": etype}
                event_bytes = estimate_payload_bytes(event_dump)
                sr.response_events.append({
                    "format": f"anthropic.{etype}",
                    "timestamp": time.time(),
                    "bytes": event_bytes,
                    "payload": event_dump,
                })
                if etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    text = getattr(delta, "text", None) if delta else None
                    if text:
                        sr.add_chunk(text, chunk_bytes=event_bytes)
                        continue
                sr.add_event_bytes(event_bytes)

            final = stream.get_final_message()
            tokens_in, tokens_out = self._usage(final)
            sr.tokens_in = tokens_in
            sr.tokens_out = tokens_out
            sr.response_payload = {
                "format": "anthropic.messages.final",
                "payload": self._response_to_dict(final),
            }

        return sr

    # ------------------------------------------------------------------
    # Multimodal helpers expected by MultimodalScenario
    # ------------------------------------------------------------------

    def _multimodal_create(
        self,
        prompt: str,
        media_path: str,
        media_type: str,            # "image" | "document"
        model: str,
        **kwargs,
    ) -> ChatResponse:
        with open(media_path, "rb") as f:
            raw = f.read()
        raw_bytes = len(raw)
        mime = mimetypes.guess_type(media_path)[0]
        if media_type == "image" and not mime:
            mime = "image/png"
        if media_type == "document" and not mime:
            mime = "application/pdf"

        b64 = base64.standard_b64encode(raw).decode("ascii")
        content_block = {
            "type": media_type,
            "source": {"type": "base64", "media_type": mime, "data": b64},
        }

        api_messages = [{
            "role": "user",
            "content": [content_block, {"type": "text", "text": prompt}],
        }]

        params = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", self.DEFAULT_MAX_TOKENS),
            "messages": api_messages,
        }
        params.update(kwargs)

        # Account for the base64-encoded body without bloating the trace
        # payload (we keep only a reference to media_path in the trace).
        request_bytes = (
            len(prompt.encode("utf-8"))
            + len(b64.encode("ascii"))
            + 256  # JSON envelope estimate
        )

        t_start = time.time()
        response = self._client.messages.create(**params)
        t_end = time.time()

        content = self._extract_text(response)
        tokens_in, tokens_out = self._usage(response)
        response_dump = self._response_to_dict(response)
        response_bytes = estimate_payload_bytes(response_dump)

        trace_payload = {
            "model": model,
            "max_tokens": params["max_tokens"],
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": media_type,
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            f"{media_type}_path": media_path,
                            "raw_bytes": raw_bytes,
                            "base64_bytes": len(b64),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        }

        return ChatResponse(
            content=content,
            latency_sec=t_end - t_start,
            model=getattr(response, "model", model),
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            request_payload={
                "format": "anthropic.messages.create",
                "payload": trace_payload,
            },
            response_payload={
                "format": "anthropic.messages.response",
                "payload": response_dump,
            },
        )

    def generate_content_with_image(
        self,
        prompt: str,
        image_path: str,
        model: str = "claude-sonnet-4-6",
        **kwargs,
    ) -> ChatResponse:
        return self._multimodal_create(prompt, image_path, "image", model, **kwargs)

    def generate_content_with_document(
        self,
        prompt: str,
        document_path: str,
        model: str = "claude-sonnet-4-6",
        **kwargs,
    ) -> ChatResponse:
        return self._multimodal_create(prompt, document_path, "document", model, **kwargs)
