"""
Base interfaces and data structures for LLM clients.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional, Any
from enum import Enum
import json
import time


class MessageRole(Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to API-compatible dictionary."""
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class ToolCall:
    """Represents a tool/function call requested by the model."""
    id: str
    name: str
    arguments: dict


@dataclass
class StreamingChunk:
    """A single chunk from a streaming response."""
    content: str
    timestamp: float
    index: int
    bytes_received: Optional[int] = None


@dataclass
class StreamingResponse:
    """
    Wrapper for streaming LLM responses with timing metrics.
    """
    chunks: list[StreamingChunk] = field(default_factory=list)
    t_request_start: float = 0.0
    t_first_chunk: Optional[float] = None
    t_last_chunk: Optional[float] = None
    total_content: str = ""
    request_bytes: int = 0
    response_bytes: int = 0
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    model: str = ""
    request_payload: Optional[Any] = None
    response_payload: Optional[Any] = None
    response_events: list[Any] = field(default_factory=list)

    def add_chunk(
        self,
        content: str,
        chunk_bytes: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add a chunk and update timing metrics."""
        now = timestamp or time.time()
        if self.t_first_chunk is None:
            self.t_first_chunk = now
        self.t_last_chunk = now

        chunk = StreamingChunk(
            content=content,
            timestamp=now,
            index=len(self.chunks),
            bytes_received=chunk_bytes,
        )
        self.chunks.append(chunk)
        self.total_content += content
        if chunk_bytes is None:
            chunk_bytes = len(content.encode("utf-8"))
        self.response_bytes += chunk_bytes

    def add_event_bytes(self, byte_count: int) -> None:
        """Track non-content response bytes (e.g., metadata-only chunks)."""
        if byte_count > 0:
            self.response_bytes += byte_count

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token (chunk) in seconds."""
        if self.t_first_chunk is None:
            return None
        return self.t_first_chunk - self.t_request_start

    @property
    def ttlt(self) -> Optional[float]:
        """Time to last token (chunk) in seconds."""
        if self.t_last_chunk is None:
            return None
        return self.t_last_chunk - self.t_request_start

    @property
    def inter_chunk_times(self) -> list[float]:
        """Time between consecutive chunks."""
        if len(self.chunks) < 2:
            return []
        times = []
        for i in range(1, len(self.chunks)):
            times.append(self.chunks[i].timestamp - self.chunks[i-1].timestamp)
        return times


@dataclass
class ChatResponse:
    """Response from a non-streaming chat completion."""
    content: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_sec: float = 0.0
    model: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Any = None
    request_payload: Optional[Any] = None
    response_payload: Optional[Any] = None

    # Byte sizes for traffic analysis
    request_bytes: int = 0
    response_bytes: int = 0


@dataclass
class ImageResponse:
    """Response from image generation."""
    image_data: Optional[bytes] = None
    image_url: Optional[str] = None
    revised_prompt: Optional[str] = None
    latency_sec: float = 0.0
    request_bytes: int = 0
    response_bytes: int = 0
    request_payload: Optional[Any] = None
    response_payload: Optional[Any] = None


class LLMClient(ABC):
    """
    Abstract base class for LLM client adapters.

    All implementations must provide methods for:
    - Chat completions (streaming and non-streaming)
    - Tool/function calling
    - Image generation (where supported)
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name (e.g., 'openai', 'gemini')."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> ChatResponse | Iterator[str]:
        """
        Send a chat completion request.

        Args:
            messages: Conversation history
            model: Model identifier
            stream: Whether to stream the response
            tools: Optional tool definitions for function calling
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatResponse for non-streaming, or Iterator[str] for streaming
        """
        pass

    @abstractmethod
    def chat_streaming(
        self,
        messages: list[ChatMessage],
        model: str,
        **kwargs
    ) -> StreamingResponse:
        """
        Send a streaming chat request and collect all chunks with timing.

        Returns:
            StreamingResponse with all chunks and timing metrics
        """
        pass

    def generate_image(
        self,
        prompt: str,
        model: str = "gpt-image-1.5",
        size: str = "1024x1024",
        **kwargs
    ) -> ImageResponse:
        """
        Generate an image from a text prompt.

        Default implementation raises NotImplementedError.
        Override in subclasses that support image generation.
        """
        raise NotImplementedError(f"{self.provider} does not support image generation")

    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate token count for a text string.

        Simple approximation: ~4 characters per token for English.
        Override for more accurate provider-specific counting.
        """
        try:
            import tiktoken
            if model:
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return len(text) // 4

    def estimate_message_tokens(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None
    ) -> int:
        """Estimate token count for a list of chat messages."""
        parts = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            name = msg.name or ""
            tool_call_id = msg.tool_call_id or ""
            parts.append(f"{role}:{name}:{tool_call_id}:{msg.content}")
        return self.estimate_tokens("\n".join(parts), model=model)


def estimate_payload_bytes(payload: Any) -> int:
    """Estimate serialized payload size in bytes."""
    try:
        return len(json.dumps(payload, default=str).encode("utf-8"))
    except Exception:
        return len(str(payload).encode("utf-8"))
