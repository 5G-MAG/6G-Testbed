"""
LLM Client adapters for the 6G AI Traffic Testbed.

Provides unified interface for interacting with various LLM providers
(OpenAI, Gemini, etc.) while capturing traffic metrics.
"""

from .base import LLMClient, StreamingResponse, ChatMessage, ToolCall, MessageRole, ChatResponse
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .deepseek_client import DeepSeekClient
from .vllm_client import VLLMClient
from .azure_openai_client import AzureOpenAIClient
from .azure_inference_client import AzureInferenceClient
from .anthropic_client import AnthropicClient
from .mcp_client import (
    MCPClient,
    MCPServerConfig,
    MCPServerConnection,
    MCPTool,
    MCPToolResult,
    BRAVE_SEARCH_SERVER,
    FETCH_SERVER,
    FILESYSTEM_SERVER,
    load_mcp_config_from_file,
)
from .realtime_client import (
    RealtimeClient,
    RealtimeEvent,
    RealtimeChunk,
    RealtimeTurnMetrics,
    RealtimeSessionMetrics,
    RealtimeEventType,
    Modality,
    Voice,
)
from .realtime_webrtc_client import RealtimeWebRTCClient

__all__ = [
    "LLMClient",
    "StreamingResponse",
    "ChatMessage",
    "ChatResponse",
    "MessageRole",
    "ToolCall",
    "OpenAIClient",
    "GeminiClient",
    "DeepSeekClient",
    "VLLMClient",
    "AzureOpenAIClient",
    "AzureInferenceClient",
    "AnthropicClient",
    "MCPClient",
    "MCPServerConfig",
    "MCPServerConnection",
    "MCPTool",
    "MCPToolResult",
    "BRAVE_SEARCH_SERVER",
    "FETCH_SERVER",
    "FILESYSTEM_SERVER",
    "load_mcp_config_from_file",
    # Real-time API
    "RealtimeClient",
    "RealtimeWebRTCClient",
    "RealtimeEvent",
    "RealtimeChunk",
    "RealtimeTurnMetrics",
    "RealtimeSessionMetrics",
    "RealtimeEventType",
    "Modality",
    "Voice",
]
