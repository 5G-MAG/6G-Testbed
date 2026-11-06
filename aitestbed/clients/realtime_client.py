"""
OpenAI Realtime API client for the 6G AI Traffic Testbed.

Implements WebSocket-based real-time conversational AI with bidirectional
audio/text streaming for characterizing real-time AI traffic patterns.
"""

import os
import time
import json
import base64
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator, Callable
from enum import Enum

import websockets
from websockets.asyncio.client import connect as ws_connect

logger = logging.getLogger(__name__)


class RealtimeEventType(Enum):
    """OpenAI Realtime API event types."""
    # Client events
    SESSION_UPDATE = "session.update"
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"

    # Server events
    ERROR = "error"
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED = "conversation.item.input_audio_transcription.failed"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
    CONVERSATION_ITEM_DELETED = "conversation.item.deleted"
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    RATE_LIMITS_UPDATED = "rate_limits.updated"


class Modality(Enum):
    """Input/output modalities for realtime sessions."""
    TEXT = "text"
    AUDIO = "audio"


class Voice(Enum):
    """Available voices for audio output."""
    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"


@dataclass
class RealtimeEvent:
    """A single event from the Realtime API."""
    event_type: str
    event_id: Optional[str]
    data: dict
    timestamp: float
    raw_bytes: int


@dataclass
class RealtimeChunk:
    """A single chunk of audio or text data."""
    content: str  # Text content or base64-encoded audio
    content_type: str  # "text" or "audio"
    timestamp: float
    index: int
    bytes_received: int


@dataclass
class RealtimeTurnMetrics:
    """Metrics for a single conversation turn."""
    turn_index: int

    # Timing metrics
    t_request_start: float = 0.0
    t_first_response: Optional[float] = None
    t_last_response: Optional[float] = None
    t_response_done: Optional[float] = None

    # Data transfer metrics
    request_bytes: int = 0
    response_bytes: int = 0
    audio_bytes_sent: int = 0
    audio_bytes_received: int = 0
    text_bytes_sent: int = 0
    text_bytes_received: int = 0

    # Chunk metrics
    text_chunks: list[RealtimeChunk] = field(default_factory=list)
    audio_chunks: list[RealtimeChunk] = field(default_factory=list)
    events: list[RealtimeEvent] = field(default_factory=list)
    function_call_events: list[dict] = field(default_factory=list)

    # Content
    input_text: str = ""
    output_text: str = ""
    input_transcript: str = ""
    output_transcript: str = ""

    # Status
    success: bool = True
    error_message: Optional[str] = None

    # Optional audio artifacts
    audio_input_path: Optional[str] = None
    audio_output_path: Optional[str] = None
    audio_meta_path: Optional[str] = None

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token/audio chunk."""
        if self.t_first_response is None:
            return None
        return self.t_first_response - self.t_request_start

    @property
    def ttlt(self) -> Optional[float]:
        """Time to last token/audio chunk."""
        if self.t_last_response is None:
            return None
        return self.t_last_response - self.t_request_start

    @property
    def total_latency(self) -> Optional[float]:
        """Total turn latency (until response.done)."""
        if self.t_response_done is None:
            return None
        return self.t_response_done - self.t_request_start

    @property
    def inter_chunk_times(self) -> list[float]:
        """Time between consecutive chunks (text + audio combined)."""
        all_chunks = sorted(
            self.text_chunks + self.audio_chunks,
            key=lambda c: c.timestamp
        )
        if len(all_chunks) < 2:
            return []
        return [
            all_chunks[i].timestamp - all_chunks[i-1].timestamp
            for i in range(1, len(all_chunks))
        ]

    @property
    def chunk_count(self) -> int:
        """Total number of chunks received."""
        return len(self.text_chunks) + len(self.audio_chunks)


@dataclass
class RealtimeSessionMetrics:
    """Metrics for an entire realtime session."""
    session_id: str
    model: str

    # Session timing
    t_session_start: float = 0.0
    t_session_end: Optional[float] = None
    t_first_connected: Optional[float] = None

    # WebRTC SDP negotiation (optional)
    sdp_offer: Optional[str] = None
    sdp_answer: Optional[str] = None
    sdp_offer_bytes: int = 0
    sdp_answer_bytes: int = 0
    sdp_offer_hash: Optional[str] = None
    sdp_answer_hash: Optional[str] = None
    sdp_negotiation_sec: Optional[float] = None

    # Aggregated metrics
    total_turns: int = 0
    total_request_bytes: int = 0
    total_response_bytes: int = 0
    total_audio_bytes_sent: int = 0
    total_audio_bytes_received: int = 0
    total_text_bytes_sent: int = 0
    total_text_bytes_received: int = 0

    # Turn details
    turns: list[RealtimeTurnMetrics] = field(default_factory=list)

    # Status
    success: bool = True
    error_message: Optional[str] = None

    @property
    def session_duration(self) -> Optional[float]:
        """Total session duration."""
        if self.t_session_end is None:
            return None
        return self.t_session_end - self.t_session_start

    @property
    def avg_ttft(self) -> Optional[float]:
        """Average time to first token across turns."""
        ttfts = [t.ttft for t in self.turns if t.ttft is not None]
        return sum(ttfts) / len(ttfts) if ttfts else None

    @property
    def avg_turn_latency(self) -> Optional[float]:
        """Average turn latency."""
        latencies = [t.total_latency for t in self.turns if t.total_latency is not None]
        return sum(latencies) / len(latencies) if latencies else None


class RealtimeClient:
    """
    OpenAI Realtime API client with traffic metrics collection.

    Supports both text and audio modalities for real-time conversational AI.
    """

    REALTIME_API_URL = "wss://api.openai.com/v1/realtime"
    DEFAULT_MODEL = "gpt-realtime-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL
    ):
        """
        Initialize the Realtime API client.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            model: Model to use for realtime sessions.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for Realtime API")

        self.model = model
        self._ws = None
        self._current_session_metrics: Optional[RealtimeSessionMetrics] = None
        self._current_turn_metrics: Optional[RealtimeTurnMetrics] = None
        self._event_handlers: dict[str, list[Callable]] = {}

    @property
    def provider(self) -> str:
        return "openai_realtime"

    async def connect(
        self,
        modalities: list[str] = None,
        voice: str = "alloy",
        instructions: Optional[str] = None,
        turn_detection: Optional[dict] = None,
        input_audio_format: str = "pcm16",
        output_audio_format: str = "pcm16",
        input_audio_transcription: Optional[dict] = None,
        temperature: float = 0.8,
        max_response_output_tokens: Optional[int] = None,
    ) -> RealtimeSessionMetrics:
        """
        Connect to the Realtime API and configure the session.

        Args:
            modalities: List of modalities ["text", "audio"]. Defaults to ["text", "audio"].
            voice: Voice for audio output.
            instructions: System instructions for the session.
            turn_detection: Turn detection config (e.g., {"type": "server_vad"}).
            input_audio_format: Format for input audio (pcm16, g711_ulaw, g711_alaw).
            output_audio_format: Format for output audio.
            input_audio_transcription: Config for input transcription.
            temperature: Response randomness (0.6-1.2).
            max_response_output_tokens: Max tokens per response.

        Returns:
            RealtimeSessionMetrics for the session.
        """
        if modalities is None:
            modalities = ["text", "audio"]

        t_start = time.time()

        # Initialize session metrics
        self._current_session_metrics = RealtimeSessionMetrics(
            session_id="",
            model=self.model,
            t_session_start=t_start
        )

        # Build WebSocket URL with model
        url = f"{self.REALTIME_API_URL}?model={self.model}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }

        try:
            self._ws = await ws_connect(url, additional_headers=headers)
            self._current_session_metrics.t_first_connected = time.time()

            # Wait for session.created event
            response = await self._ws.recv()
            t_recv = time.time()
            event = json.loads(response)

            if event.get("type") != "session.created":
                raise RuntimeError(f"Expected session.created, got {event.get('type')}")

            self._current_session_metrics.session_id = event.get("session", {}).get("id", "")
            self._current_session_metrics.total_response_bytes += len(response.encode() if isinstance(response, str) else response)

            # Configure session
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": modalities,
                    "voice": voice,
                    "input_audio_format": input_audio_format,
                    "output_audio_format": output_audio_format,
                    "temperature": temperature,
                }
            }

            if instructions:
                session_config["session"]["instructions"] = instructions

            if turn_detection is not None:
                session_config["session"]["turn_detection"] = turn_detection

            if input_audio_transcription:
                session_config["session"]["input_audio_transcription"] = input_audio_transcription

            if max_response_output_tokens:
                session_config["session"]["max_response_output_tokens"] = max_response_output_tokens

            config_json = json.dumps(session_config)
            self._current_session_metrics.total_request_bytes += len(config_json.encode())
            await self._ws.send(config_json)

            # Wait for session.updated confirmation (or error)
            while True:
                response = await self._ws.recv()
                self._current_session_metrics.total_response_bytes += len(
                    response.encode() if isinstance(response, str) else response
                )
                try:
                    event = json.loads(response)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Failed to parse session update response: {exc}")
                    continue

                event_type = event.get("type")
                if event_type == "session.updated":
                    break
                if event_type == "error":
                    error = event.get("error", {})
                    message = error.get("message", str(error))
                    self._current_session_metrics.success = False
                    self._current_session_metrics.error_message = message
                    logger.error(f"Realtime session update error: {message}")
                    raise RuntimeError(message)

                logger.debug(f"Ignoring session update event: {event_type}")

            logger.info(f"Realtime session connected: {self._current_session_metrics.session_id}")
            return self._current_session_metrics

        except Exception as e:
            self._current_session_metrics.success = False
            self._current_session_metrics.error_message = str(e)
            raise

    async def send_text(self, text: str) -> RealtimeTurnMetrics:
        """
        Send a text message and wait for the complete response.

        Args:
            text: Text message to send.

        Returns:
            RealtimeTurnMetrics for this turn.
        """
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        turn_index = len(self._current_session_metrics.turns)
        t_request_start = time.time()

        self._current_turn_metrics = RealtimeTurnMetrics(
            turn_index=turn_index,
            t_request_start=t_request_start,
            input_text=text
        )

        # Create conversation item with user message
        item_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        }

        item_json = json.dumps(item_event)
        text_bytes = len(item_json.encode())
        self._current_turn_metrics.request_bytes += text_bytes
        self._current_turn_metrics.text_bytes_sent += text_bytes
        await self._ws.send(item_json)

        # Request a response
        response_event = {
            "type": "response.create"
        }
        response_json = json.dumps(response_event)
        self._current_turn_metrics.request_bytes += len(response_json.encode())
        await self._ws.send(response_json)

        # Process events until response.done
        await self._process_response_events()

        # Update session metrics
        self._current_session_metrics.turns.append(self._current_turn_metrics)
        self._current_session_metrics.total_turns += 1
        self._current_session_metrics.total_request_bytes += self._current_turn_metrics.request_bytes
        self._current_session_metrics.total_response_bytes += self._current_turn_metrics.response_bytes
        self._current_session_metrics.total_text_bytes_sent += self._current_turn_metrics.text_bytes_sent
        self._current_session_metrics.total_text_bytes_received += self._current_turn_metrics.text_bytes_received
        self._current_session_metrics.total_audio_bytes_received += self._current_turn_metrics.audio_bytes_received

        return self._current_turn_metrics

    async def send_audio(
        self,
        audio_data: bytes,
        commit: bool = True,
        chunk_ms: int = 100,
        sample_rate: int = 24000,
        sample_width_bytes: int = 2,
        clear_buffer: bool = True,
        wait_for_commit: bool = True,
        pace_audio: bool = False,
    ) -> RealtimeTurnMetrics:
        """
        Send audio data and wait for the complete response.

        Args:
            audio_data: Raw audio bytes (PCM16 by default).
            commit: Whether to commit the audio buffer after sending.
            chunk_ms: Chunk size in milliseconds (0 to send as one chunk).
            sample_rate: Sample rate in Hz for PCM audio.
            sample_width_bytes: Bytes per sample (2 for pcm16).
            clear_buffer: Whether to clear the input buffer before sending.
            wait_for_commit: Whether to wait for input_audio_buffer.committed.
            pace_audio: Whether to pace chunk sends in real-time.

        Returns:
            RealtimeTurnMetrics for this turn.
        """
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        turn_index = len(self._current_session_metrics.turns)
        t_request_start = time.time()

        self._current_turn_metrics = RealtimeTurnMetrics(
            turn_index=turn_index,
            t_request_start=t_request_start
        )

        # Optionally clear buffer before appending new audio
        if clear_buffer:
            clear_event = {"type": "input_audio_buffer.clear"}
            clear_json = json.dumps(clear_event)
            self._current_turn_metrics.request_bytes += len(clear_json.encode())
            await self._ws.send(clear_json)
            cleared = await self._wait_for_audio_clear()
            if not cleared:
                return self._current_turn_metrics

        # Send audio data as base64 (optionally chunked)
        if chunk_ms and sample_rate > 0 and sample_width_bytes > 0:
            chunk_bytes = int(sample_rate * (chunk_ms / 1000.0) * sample_width_bytes)
        else:
            chunk_bytes = 0

        if chunk_bytes <= 0:
            chunks = [audio_data]
        else:
            chunks = [
                audio_data[i:i + chunk_bytes]
                for i in range(0, len(audio_data), chunk_bytes)
                if audio_data[i:i + chunk_bytes]
            ]

        for idx, chunk in enumerate(chunks):
            audio_b64 = base64.b64encode(chunk).decode("ascii")
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }

            audio_json = json.dumps(audio_event)
            audio_bytes = len(audio_json.encode())
            self._current_turn_metrics.request_bytes += audio_bytes
            await self._ws.send(audio_json)
            if pace_audio and chunk_ms > 0 and idx < len(chunks) - 1:
                await asyncio.sleep(chunk_ms / 1000.0)

        self._current_turn_metrics.audio_bytes_sent += len(audio_data)

        if commit:
            # Give the server a brief moment to process appended audio
            if chunk_bytes > 0:
                await asyncio.sleep(0.05)
            # Commit the audio buffer
            commit_event = {"type": "input_audio_buffer.commit"}
            commit_json = json.dumps(commit_event)
            self._current_turn_metrics.request_bytes += len(commit_json.encode())
            await self._ws.send(commit_json)

        if wait_for_commit:
            committed = await self._wait_for_audio_commit()
            if not committed:
                return self._current_turn_metrics

        if commit or wait_for_commit:
            # Request a response
            response_event = {"type": "response.create"}
            response_json = json.dumps(response_event)
            self._current_turn_metrics.request_bytes += len(response_json.encode())
            await self._ws.send(response_json)

            # Process events until response.done
            await self._process_response_events()

        # Update session metrics
        self._current_session_metrics.turns.append(self._current_turn_metrics)
        self._current_session_metrics.total_turns += 1
        self._current_session_metrics.total_request_bytes += self._current_turn_metrics.request_bytes
        self._current_session_metrics.total_response_bytes += self._current_turn_metrics.response_bytes
        self._current_session_metrics.total_audio_bytes_sent += self._current_turn_metrics.audio_bytes_sent
        self._current_session_metrics.total_audio_bytes_received += self._current_turn_metrics.audio_bytes_received
        self._current_session_metrics.total_text_bytes_received += self._current_turn_metrics.text_bytes_received

        return self._current_turn_metrics

    async def _wait_for_audio_commit(self, timeout: float = 5.0) -> bool:
        """Wait for input_audio_buffer.committed or error."""
        if not self._ws:
            return False

        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                response = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            t_recv = time.time()

            raw_bytes = len(response.encode() if isinstance(response, str) else response)
            self._current_turn_metrics.response_bytes += raw_bytes

            try:
                event = json.loads(response)
            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to parse audio commit event: {exc}")
                continue

            event_type = event.get("type", "")
            self._current_turn_metrics.events.append(RealtimeEvent(
                event_type=event_type,
                event_id=event.get("event_id"),
                data=event,
                timestamp=t_recv,
                raw_bytes=raw_bytes,
            ))

            if event_type == "input_audio_buffer.committed":
                return True
            if event_type == "error":
                error = event.get("error", {})
                message = error.get("message", str(error))
                self._current_turn_metrics.success = False
                self._current_turn_metrics.error_message = message
                logger.error(f"Realtime audio buffer error: {message}")
                return False

        logger.warning("Timed out waiting for input_audio_buffer.committed")
        return True

    async def _wait_for_audio_clear(self, timeout: float = 5.0) -> bool:
        """Wait for input_audio_buffer.cleared or error."""
        if not self._ws:
            return False

        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                response = await asyncio.wait_for(self._ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            t_recv = time.time()

            raw_bytes = len(response.encode() if isinstance(response, str) else response)
            self._current_turn_metrics.response_bytes += raw_bytes

            try:
                event = json.loads(response)
            except json.JSONDecodeError as exc:
                logger.warning(f"Failed to parse audio clear event: {exc}")
                continue

            event_type = event.get("type", "")
            self._current_turn_metrics.events.append(RealtimeEvent(
                event_type=event_type,
                event_id=event.get("event_id"),
                data=event,
                timestamp=t_recv,
                raw_bytes=raw_bytes,
            ))

            if event_type == "input_audio_buffer.cleared":
                return True
            if event_type == "error":
                error = event.get("error", {})
                message = error.get("message", str(error))
                self._current_turn_metrics.success = False
                self._current_turn_metrics.error_message = message
                logger.error(f"Realtime audio buffer error: {message}")
                return False

        logger.warning("Timed out waiting for input_audio_buffer.cleared")
        return True

    async def _process_response_events(self):
        """Process events until response.done is received."""
        while True:
            try:
                response = await self._ws.recv()
                t_recv = time.time()

                raw_bytes = len(response.encode() if isinstance(response, str) else response)
                self._current_turn_metrics.response_bytes += raw_bytes

                event = json.loads(response)
                event_type = event.get("type", "")

                # Record event
                realtime_event = RealtimeEvent(
                    event_type=event_type,
                    event_id=event.get("event_id"),
                    data=event,
                    timestamp=t_recv,
                    raw_bytes=raw_bytes
                )
                self._current_turn_metrics.events.append(realtime_event)

                # Track first response time
                if self._current_turn_metrics.t_first_response is None:
                    if event_type in ["response.text.delta", "response.audio.delta",
                                     "response.audio_transcript.delta"]:
                        self._current_turn_metrics.t_first_response = t_recv

                # Handle different event types
                if event_type == "response.text.delta":
                    delta = event.get("delta", "")
                    self._current_turn_metrics.output_text += delta
                    self._current_turn_metrics.text_bytes_received += len(delta.encode())

                    chunk = RealtimeChunk(
                        content=delta,
                        content_type="text",
                        timestamp=t_recv,
                        index=len(self._current_turn_metrics.text_chunks),
                        bytes_received=raw_bytes
                    )
                    self._current_turn_metrics.text_chunks.append(chunk)
                    self._current_turn_metrics.t_last_response = t_recv

                elif event_type == "response.audio.delta":
                    audio_b64 = event.get("delta", "")
                    audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
                    self._current_turn_metrics.audio_bytes_received += len(audio_bytes)

                    chunk = RealtimeChunk(
                        content=audio_b64,
                        content_type="audio",
                        timestamp=t_recv,
                        index=len(self._current_turn_metrics.audio_chunks),
                        bytes_received=len(audio_bytes)
                    )
                    self._current_turn_metrics.audio_chunks.append(chunk)
                    self._current_turn_metrics.t_last_response = t_recv

                elif event_type == "response.audio_transcript.delta":
                    delta = event.get("delta", "")
                    self._current_turn_metrics.output_transcript += delta

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    self._current_turn_metrics.input_transcript = event.get("transcript", "")
                elif event_type in [
                    "response.function_call_arguments.delta",
                    "response.function_call_arguments.done",
                ]:
                    self._current_turn_metrics.function_call_events.append(event)

                elif event_type == "response.done":
                    self._current_turn_metrics.t_response_done = t_recv

                    # Check for errors in response
                    response_data = event.get("response", {})
                    if response_data.get("status") == "failed":
                        self._current_turn_metrics.success = False
                        status_details = response_data.get("status_details", {})
                        self._current_turn_metrics.error_message = status_details.get("error", {}).get("message", "Unknown error")

                    break

                elif event_type == "error":
                    error = event.get("error", {})
                    message = error.get("message", str(error))
                    lowered = message.lower()
                    if "buffer too small" in lowered and "input audio buffer" in lowered:
                        has_commit = any(
                            evt.event_type == "input_audio_buffer.committed"
                            for evt in self._current_turn_metrics.events
                        )
                        if has_commit:
                            logger.warning(
                                "Ignoring redundant input audio buffer error after commit."
                            )
                            continue

                    self._current_turn_metrics.success = False
                    self._current_turn_metrics.error_message = message
                    logger.error(f"Realtime API error: {message}")
                    break

            except Exception as e:
                self._current_turn_metrics.success = False
                self._current_turn_metrics.error_message = str(e)
                logger.error(f"Error processing realtime events: {e}")
                break

    async def disconnect(self) -> RealtimeSessionMetrics:
        """
        Disconnect from the Realtime API.

        Returns:
            Final RealtimeSessionMetrics for the session.
        """
        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._current_session_metrics:
            self._current_session_metrics.t_session_end = time.time()

        return self._current_session_metrics

    async def run_text_conversation(
        self,
        messages: list[str],
        instructions: Optional[str] = None,
        voice: str = "alloy",
        modalities: list[str] = None,
        temperature: float = 0.8
    ) -> RealtimeSessionMetrics:
        """
        Run a complete text-based conversation.

        This is a convenience method that connects, sends all messages,
        and disconnects, returning complete session metrics.

        Args:
            messages: List of user messages to send.
            instructions: System instructions.
            voice: Voice for audio output.
            modalities: List of modalities to enable.
            temperature: Response temperature.

        Returns:
            Complete RealtimeSessionMetrics.
        """
        if modalities is None:
            modalities = ["text", "audio"]

        try:
            await self.connect(
                modalities=modalities,
                voice=voice,
                instructions=instructions,
                temperature=temperature
            )

            for message in messages:
                await self.send_text(message)

            return await self.disconnect()

        except Exception as e:
            if self._current_session_metrics:
                self._current_session_metrics.success = False
                self._current_session_metrics.error_message = str(e)
            await self.disconnect()
            raise

    def run_text_conversation_sync(
        self,
        messages: list[str],
        instructions: Optional[str] = None,
        voice: str = "alloy",
        modalities: list[str] = None,
        temperature: float = 0.8
    ) -> RealtimeSessionMetrics:
        """
        Synchronous wrapper for run_text_conversation.

        Runs the async conversation in a new event loop.
        """
        return asyncio.run(
            self.run_text_conversation(
                messages=messages,
                instructions=instructions,
                voice=voice,
                modalities=modalities,
                temperature=temperature
            )
        )
