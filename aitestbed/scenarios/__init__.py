"""
Traffic Scenarios for the 6G AI Traffic Testbed.

Each scenario represents a distinct AI service interaction pattern.
"""

from .base import BaseScenario, ScenarioResult
from .chat import ChatScenario
from .chat_token import ChatTokenScenario
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
from .music_agent import MusicAgentScenario, MusicResearchAgentScenario
from .playwright_agent import PlaywrightAgentScenario
from .trading_agent import TradingAgentScenario
from .realtime import (
    RealtimeConversationScenario,
    RealtimeWebRTCConversationScenario,
    RealtimeAudioScenario,
    RealtimeAudioWebRTCScenario,
)
from .realtime_video import RealtimeVideoUnderstandingScenario
from .encoder_placement import EncoderPlacementScenario
from .encoder_placement_concurrent import EncoderPlacementConcurrentScenario
from .encoder_placement_qwen36 import EncoderPlacementQwen36Scenario
from .encoder_placement_concurrent_qwen36 import (
    EncoderPlacementConcurrentQwen36Scenario,
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
    # Music agent (Spotify MCP)
    "MusicAgentScenario",
    "MusicResearchAgentScenario",
    # Playwright browser automation agent
    "PlaywrightAgentScenario",
    # Trading / market data agent (Alpaca MCP)
    "TradingAgentScenario",
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
    # Real-time video unstanding
    "RealtimeVideoUnderstandingScenario",
    # chat with token ID
    "ChatTokenScenario",
    # Encoder placement (server-side vs terminal-side visual encoder)
    "EncoderPlacementScenario",
    # Encoder placement under concurrency (sidecar 8012)
    "EncoderPlacementConcurrentScenario",
    # Encoder placement, Qwen3.6 sibling (serial sidecar 8004)
    "EncoderPlacementQwen36Scenario",
    # Encoder placement, Qwen3.6 sibling (concurrent sidecar 8014)
    "EncoderPlacementConcurrentQwen36Scenario",
]
