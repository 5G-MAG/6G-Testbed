"""
Traffic Scenarios for the 6G AI Traffic Testbed.

Each scenario represents a distinct AI service interaction pattern.
"""

from .base import BaseScenario, ScenarioResult
from .chat import ChatScenario
from .agent import (
    BaseAgentScenario,
    MCPToolExecutor,
    ShoppingAgentScenario,
    WebSearchAgentScenario,
    GeneralAgentScenario,
)
from .image import ImageGenerationScenario
from .multimodal import MultimodalScenario
from .video import VideoUnderstandingScenario
from .computer_use import ComputerUseScenario
from .direct_search import (
    DirectSearchClient,
    DirectWebSearchScenario,
    ParallelSearchBenchmarkScenario,
    SearchEngine,
    SearchResult,
    ThreadedSearchExecutor,
    ThreadedSearchResult,
)
from .realtime import (
    RealtimeConversationScenario,
    RealtimeWebRTCConversationScenario,
    RealtimeAudioScenario,
    RealtimeAudioWebRTCScenario,
)

__all__ = [
    "BaseScenario",
    "ScenarioResult",
    "ChatScenario",
    "BaseAgentScenario",
    "MCPToolExecutor",
    "ShoppingAgentScenario",
    "WebSearchAgentScenario",
    "GeneralAgentScenario",
    "ImageGenerationScenario",
    "MultimodalScenario",
    "VideoUnderstandingScenario",
    "ComputerUseScenario",
    # Direct search (no MCP)
    "DirectSearchClient",
    "DirectWebSearchScenario",
    "ParallelSearchBenchmarkScenario",
    "SearchEngine",
    "SearchResult",
    "ThreadedSearchExecutor",
    "ThreadedSearchResult",
    # Real-time conversational AI
    "RealtimeConversationScenario",
    "RealtimeWebRTCConversationScenario",
    "RealtimeAudioScenario",
    "RealtimeAudioWebRTCScenario",
]
