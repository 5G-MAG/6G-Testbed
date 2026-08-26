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
from .music_agent import MusicAgentScenario, MusicResearchAgentScenario
from .playwright_agent import PlaywrightAgentScenario
from .trading_agent import TradingAgentScenario
from .weather_agent import WeatherAgentScenario, NavigationWeatherAgentScenario
from .maps_agent import MapsAgentScenario
from .exa_search_agent import ExaSearchAgentScenario
from .twilio_agent import TwilioCommunicationAgentScenario
from .smart_home_agent import SmartHomeAgentScenario
from .realtime import (
    RealtimeConversationScenario,
    RealtimeWebRTCConversationScenario,
    RealtimeAudioScenario,
    RealtimeAudioWebRTCScenario,
)
from .openclaw_agent import OpenClawScenario
from .a2a_agent import (
    A2ASingleTaskScenario,
    A2AStreamingScenario,
    A2AMultiAgentScenario,
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
    # Weather / environment agent (Open-Meteo, no API key)
    "WeatherAgentScenario",
    # Combined navigation + weather agent (Google Maps + Open-Meteo)
    "NavigationWeatherAgentScenario",
    # Maps / navigation agent (Google Maps)
    "MapsAgentScenario",
    # Exa.ai search agent
    "ExaSearchAgentScenario",
    # Twilio communications agent (SMS / WhatsApp)
    "TwilioCommunicationAgentScenario",
    # Smart home agent (Home Assistant)
    "SmartHomeAgentScenario",
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
    # OpenClaw local personal-assistant agent
    "OpenClawScenario",
    # A2A (Agent2Agent protocol)
    "A2ASingleTaskScenario",
    "A2AStreamingScenario",
    "A2AMultiAgentScenario",
]
