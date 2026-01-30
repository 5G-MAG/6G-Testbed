"""
Base Scenario class for the 6G AI Traffic Testbed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import uuid
import time
import json

from clients.base import LLMClient, ChatMessage, MessageRole
from analysis.logger import TrafficLogger, LogRecord
from analysis.anonymization import get_anonymizer
from analysis.trace_logger import TraceLogger


@dataclass
class ScenarioResult:
    """Result from running a scenario."""
    scenario_id: str
    session_id: str
    network_profile: str
    run_index: int

    # Overall metrics
    success: bool = True
    total_latency_sec: float = 0.0
    total_request_bytes: int = 0
    total_response_bytes: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    # Turn-level details
    turn_count: int = 0
    api_call_count: int = 0

    # Agent-specific
    tool_calls_count: int = 0
    tool_total_latency_sec: float = 0.0

    # Streaming metrics
    ttft_sec: Optional[float] = None
    ttlt_sec: Optional[float] = None

    # Errors
    error_message: Optional[str] = None

    # Raw data for analysis
    log_records: list[LogRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseScenario(ABC):
    """
    Abstract base class for traffic scenarios.

    Each scenario represents a specific AI service interaction pattern
    that can be run under different network conditions.
    """

    def __init__(
        self,
        client: LLMClient,
        logger: TrafficLogger,
        config: dict
    ):
        """
        Initialize the scenario.

        Args:
            client: LLM client adapter to use
            logger: Traffic logger for recording metrics
            config: Scenario configuration from YAML
        """
        self.client = client
        self.logger = logger
        self.config = config
        self.scenario_id = config.get("scenario_id", self.__class__.__name__)
        self._anonymizer = get_anonymizer()
        self._trace_logger = TraceLogger.from_env()

    @property
    @abstractmethod
    def scenario_type(self) -> str:
        """Return the scenario type identifier."""
        pass

    @abstractmethod
    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """
        Execute the scenario once.

        Args:
            network_profile: Current network profile name
            run_index: Index of this run in a batch

        Returns:
            ScenarioResult with all metrics
        """
        pass

    def _create_session_id(self) -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4())

    def _create_log_record(
        self,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        **kwargs
    ) -> LogRecord:
        """Create a log record with common fields filled in."""
        trace_request = kwargs.pop("trace_request", None)
        trace_response = kwargs.pop("trace_response", None)
        trace_events = kwargs.pop("trace_events", None)
        trace_note = kwargs.pop("trace_note", None)
        metadata = kwargs.pop("metadata", None)

        trace_path = self._trace_logger.write_trace(
            scenario_id=self.scenario_id,
            session_id=session_id,
            turn_index=turn_index,
            run_index=run_index,
            network_profile=network_profile,
            provider=self.client.provider,
            model=self.config.get("model", ""),
            request_payload=trace_request,
            response_payload=trace_response,
            response_events=trace_events,
            note=trace_note,
            timing={
                "t_request_start": kwargs.get("t_request_start"),
                "t_first_token": kwargs.get("t_first_token"),
                "t_last_token": kwargs.get("t_last_token"),
                "latency_sec": kwargs.get("latency_sec"),
            },
            metrics={
                "request_bytes": kwargs.get("request_bytes"),
                "response_bytes": kwargs.get("response_bytes"),
                "tokens_in": kwargs.get("tokens_in"),
                "tokens_out": kwargs.get("tokens_out"),
                "http_status": kwargs.get("http_status"),
                "success": kwargs.get("success"),
                "is_streaming": kwargs.get("is_streaming"),
            },
        )

        if trace_path:
            if isinstance(metadata, dict):
                meta_dict = dict(metadata)
            elif isinstance(metadata, str) and metadata:
                try:
                    meta_dict = json.loads(metadata)
                except Exception:
                    meta_dict = {"raw_metadata": metadata}
            else:
                meta_dict = {}
            meta_dict["trace_path"] = trace_path
            metadata = json.dumps(meta_dict)
        elif isinstance(metadata, dict):
            metadata = json.dumps(metadata)

        provider = self._anonymizer.provider_alias(self.client.provider)
        model = self._anonymizer.model_alias(self.config.get("model", ""))
        return LogRecord(
            timestamp=time.time(),
            scenario_id=self.scenario_id,
            session_id=session_id,
            turn_index=turn_index,
            run_index=run_index,
            provider=provider,
            model=model,
            network_profile=network_profile,
            metadata=metadata or "",
            **kwargs
        )

    def _build_messages(self, prompts: list[str], history: list[dict] = None) -> list[ChatMessage]:
        """Build message list from prompts and optional history."""
        messages = []

        # Add system message if configured
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt
            ))

        # Add history if provided
        if history:
            for h in history:
                messages.append(ChatMessage(
                    role=MessageRole[h["role"].upper()],
                    content=h["content"]
                ))

        # Add current prompts
        for prompt in prompts:
            messages.append(ChatMessage(
                role=MessageRole.USER,
                content=prompt
            ))

        return messages
