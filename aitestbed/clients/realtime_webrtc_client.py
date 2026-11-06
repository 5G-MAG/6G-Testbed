"""
OpenAI Realtime API client using WebRTC transport.

Implements a WebRTC flow with a data channel for realtime events and
optional full-duplex audio streaming.
"""

from __future__ import annotations

import asyncio
import base64
import fractions
import hashlib
import json
import logging
import os
import time
from typing import Optional

import requests

try:
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
except Exception:
    MediaStreamTrack = object
    MediaStreamError = Exception

try:
    from av import AudioFrame
    from av.audio.resampler import AudioResampler
except Exception:
    AudioFrame = None
    AudioResampler = None

from .realtime_client import (
    RealtimeEvent,
    RealtimeChunk,
    RealtimeTurnMetrics,
    RealtimeSessionMetrics,
)

logger = logging.getLogger(__name__)


class AudioInputTrack(MediaStreamTrack):
    """Queue-based PCM16 audio track for WebRTC input."""

    kind = "audio"

    def __init__(self, sample_rate: int, sample_width_bytes: int = 2, channels: int = 1):
        super().__init__()
        if AudioFrame is None:
            raise RuntimeError("PyAV is required for WebRTC audio input")
        self.sample_rate = sample_rate
        self.sample_width_bytes = sample_width_bytes
        self.channels = channels
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._start_time: Optional[float] = None
        self._next_pts = 0
        self._time_base = fractions.Fraction(1, sample_rate)

    def enqueue(self, pcm_bytes: bytes) -> None:
        if pcm_bytes:
            self._queue.put_nowait(pcm_bytes)

    def clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def recv(self) -> "AudioFrame":
        if self.readyState != "live":
            raise MediaStreamError

        pcm_bytes = b""
        while not pcm_bytes:
            pcm_bytes = await self._queue.get()

        samples = len(pcm_bytes) // (self.sample_width_bytes * self.channels)
        frame = AudioFrame(
            format="s16",
            layout="mono" if self.channels == 1 else "stereo",
            samples=samples,
        )
        frame.sample_rate = self.sample_rate
        frame.planes[0].update(pcm_bytes)
        frame.pts = self._next_pts
        frame.time_base = self._time_base
        self._next_pts += samples

        if self._start_time is None:
            self._start_time = time.time()
        expected_time = self._start_time + (frame.pts / self.sample_rate)
        delay = expected_time - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        return frame


class RealtimeWebRTCClient:
    """
    OpenAI Realtime API client using WebRTC data channels.

    Notes:
    - Uses a data channel for realtime events (JSON).
    - Audio streaming uses a sendrecv audio transceiver and PCM16 frames.
    """

    REALTIME_API_URL = "https://api.openai.com/v1/realtime"
    DEFAULT_MODEL = "gpt-realtime-mini"
    DEFAULT_CHANNEL = "oai-events"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        event_channel: str = DEFAULT_CHANNEL,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for Realtime WebRTC")

        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
        except ImportError as exc:
            raise ImportError(
                "aiortc is required for WebRTC realtime. Install with: pip install aiortc"
            ) from exc

        self._RTCPeerConnection = RTCPeerConnection
        self._RTCSessionDescription = RTCSessionDescription

        self.model = model
        self.event_channel = event_channel

        self._pc = None
        self._data_channel = None
        self._event_queue: asyncio.Queue[str] = asyncio.Queue()
        self._channel_open = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._current_session_metrics: Optional[RealtimeSessionMetrics] = None
        self._current_turn_metrics: Optional[RealtimeTurnMetrics] = None
        self._audio_transceiver = None
        self._audio_input_track: Optional[AudioInputTrack] = None
        self._audio_task_stop = asyncio.Event()
        self._audio_track_task: Optional[asyncio.Task] = None
        self._audio_capture_active = False
        self._input_sample_rate = 24000
        self._output_sample_rate = 24000
        self._sample_width_bytes = 2
        self._channels = 1

    @property
    def provider(self) -> str:
        return "openai_realtime_webrtc"

    def set_audio_config(
        self,
        sample_rate: int,
        sample_width_bytes: int = 2,
        channels: int = 1,
    ) -> None:
        """Configure audio encoding parameters for input/output."""
        self._input_sample_rate = sample_rate
        self._output_sample_rate = sample_rate
        self._sample_width_bytes = sample_width_bytes
        self._channels = channels

    async def connect(
        self,
        modalities: list[str] | None = None,
        voice: str = "alloy",
        instructions: Optional[str] = None,
        turn_detection: Optional[dict] = None,
        input_audio_format: str = "pcm16",
        output_audio_format: str = "pcm16",
        input_audio_transcription: Optional[dict] = None,
        temperature: float = 0.8,
        max_response_output_tokens: Optional[int] = None,
    ) -> RealtimeSessionMetrics:
        if modalities is None:
            modalities = ["text"]

        self._loop = asyncio.get_running_loop()
        t_start = time.time()

        self._current_session_metrics = RealtimeSessionMetrics(
            session_id="",
            model=self.model,
            t_session_start=t_start,
        )

        self._pc = self._RTCPeerConnection()
        self._audio_task_stop.clear()

        # OpenAI Realtime API requires an audio transceiver in the SDP offer.
        # Use sendrecv so we can attach a local input track when needed.
        self._audio_transceiver = self._pc.addTransceiver("audio", direction="sendrecv")

        self._data_channel = self._pc.createDataChannel(self.event_channel)

        @self._data_channel.on("open")
        def _on_open():
            self._channel_open.set()

        @self._data_channel.on("message")
        def _on_message(message):
            if isinstance(message, bytes):
                text = message.decode("utf-8", errors="replace")
            else:
                text = str(message)
            if self._loop:
                self._loop.call_soon_threadsafe(self._event_queue.put_nowait, text)
            else:
                self._event_queue.put_nowait(text)

        @self._pc.on("track")
        def _on_track(track):
            if track.kind != "audio":
                return
            if self._audio_track_task:
                self._audio_track_task.cancel()
            if self._loop:
                self._audio_task_stop.clear()
                self._audio_track_task = self._loop.create_task(
                    self._consume_audio_track(track)
                )

        # Create offer and send SDP to OpenAI
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        sdp_offer = self._pc.localDescription.sdp
        sdp_offer_bytes = len(sdp_offer.encode("utf-8"))
        sdp_offer_hash = hashlib.sha256(sdp_offer.encode("utf-8")).hexdigest()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/sdp",
            "OpenAI-Beta": "realtime=v1",
        }
        url = f"{self.REALTIME_API_URL}?model={self.model}"

        sdp_start = time.time()
        answer_sdp = await asyncio.to_thread(
            self._post_sdp_offer,
            url,
            headers,
            sdp_offer,
        )
        sdp_negotiation_sec = time.time() - sdp_start
        sdp_answer_bytes = len(answer_sdp.encode("utf-8"))
        sdp_answer_hash = hashlib.sha256(answer_sdp.encode("utf-8")).hexdigest()

        self._current_session_metrics.sdp_offer = sdp_offer
        self._current_session_metrics.sdp_answer = answer_sdp
        self._current_session_metrics.sdp_offer_bytes = sdp_offer_bytes
        self._current_session_metrics.sdp_answer_bytes = sdp_answer_bytes
        self._current_session_metrics.sdp_offer_hash = sdp_offer_hash
        self._current_session_metrics.sdp_answer_hash = sdp_answer_hash
        self._current_session_metrics.sdp_negotiation_sec = sdp_negotiation_sec
        self._current_session_metrics.total_request_bytes += sdp_offer_bytes
        self._current_session_metrics.total_response_bytes += sdp_answer_bytes

        # Save SDP files to disk
        self._save_sdp_files(sdp_offer, answer_sdp, sdp_offer_hash[:8])

        answer = self._RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self._pc.setRemoteDescription(answer)

        # Wait for channel open and session.created
        await asyncio.wait_for(self._channel_open.wait(), timeout=10)
        self._current_session_metrics.t_first_connected = time.time()

        await self._wait_for_event("session.created")

        # Configure session
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": modalities,
                "voice": voice,
                "input_audio_format": input_audio_format,
                "output_audio_format": output_audio_format,
                "temperature": temperature,
            },
        }
        if instructions:
            session_config["session"]["instructions"] = instructions
        if turn_detection is not None:
            session_config["session"]["turn_detection"] = turn_detection
        if input_audio_transcription:
            session_config["session"]["input_audio_transcription"] = input_audio_transcription
        if max_response_output_tokens:
            session_config["session"]["max_response_output_tokens"] = max_response_output_tokens

        await self._send_event(session_config)
        await self._wait_for_event("session.updated")

        logger.info("Realtime WebRTC session connected")
        return self._current_session_metrics

    async def send_text(self, text: str) -> RealtimeTurnMetrics:
        if not self._data_channel:
            raise RuntimeError("Not connected. Call connect() first.")

        turn_index = len(self._current_session_metrics.turns)
        t_request_start = time.time()

        self._current_turn_metrics = RealtimeTurnMetrics(
            turn_index=turn_index,
            t_request_start=t_request_start,
            input_text=text,
        )

        item_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            },
        }
        await self._send_event(item_event, track_text_bytes=True)

        response_event = {"type": "response.create"}
        self._audio_capture_active = True
        await self._send_event(response_event)

        await self._process_response_events()
        await self._stop_audio_capture()

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
        """Send audio over the WebRTC audio track and wait for response events."""
        if not self._data_channel or not self._pc:
            raise RuntimeError("Not connected. Call connect() first.")

        turn_index = len(self._current_session_metrics.turns)
        t_request_start = time.time()

        self._current_turn_metrics = RealtimeTurnMetrics(
            turn_index=turn_index,
            t_request_start=t_request_start,
        )

        self._ensure_audio_sender(sample_rate, sample_width_bytes, self._channels)

        if clear_buffer and self._audio_input_track:
            self._audio_input_track.clear()

        chunks = self._chunk_audio(audio_data, sample_rate, sample_width_bytes, self._channels, chunk_ms)
        for idx, chunk in enumerate(chunks):
            if self._audio_input_track:
                self._audio_input_track.enqueue(chunk)
            if pace_audio and chunk_ms > 0 and idx < len(chunks) - 1:
                await asyncio.sleep(chunk_ms / 1000.0)

        self._current_turn_metrics.audio_bytes_sent += len(audio_data)
        self._current_turn_metrics.request_bytes += len(audio_data)

        if commit or wait_for_commit:
            response_event = {"type": "response.create"}
            self._audio_capture_active = True
            await self._send_event(response_event)
            await self._process_response_events()
            await self._stop_audio_capture()

        self._current_session_metrics.turns.append(self._current_turn_metrics)
        self._current_session_metrics.total_turns += 1
        self._current_session_metrics.total_request_bytes += self._current_turn_metrics.request_bytes
        self._current_session_metrics.total_response_bytes += self._current_turn_metrics.response_bytes
        self._current_session_metrics.total_audio_bytes_sent += self._current_turn_metrics.audio_bytes_sent
        self._current_session_metrics.total_audio_bytes_received += self._current_turn_metrics.audio_bytes_received
        self._current_session_metrics.total_text_bytes_received += self._current_turn_metrics.text_bytes_received

        return self._current_turn_metrics

    def _ensure_audio_sender(
        self,
        sample_rate: int,
        sample_width_bytes: int,
        channels: int,
    ) -> None:
        if not self._audio_transceiver:
            return
        if (
            self._audio_input_track is None
            or self._input_sample_rate != sample_rate
            or self._sample_width_bytes != sample_width_bytes
            or self._channels != channels
        ):
            self._input_sample_rate = sample_rate
            self._sample_width_bytes = sample_width_bytes
            self._channels = channels
            self._audio_input_track = AudioInputTrack(
                sample_rate=sample_rate,
                sample_width_bytes=sample_width_bytes,
                channels=channels,
            )
            try:
                self._audio_transceiver.sender.replaceTrack(self._audio_input_track)
            except Exception as exc:
                logger.warning("Failed to attach audio input track: %s", exc)

    @staticmethod
    def _chunk_audio(
        audio_data: bytes,
        sample_rate: int,
        sample_width_bytes: int,
        channels: int,
        chunk_ms: int,
    ) -> list[bytes]:
        if not audio_data:
            return []
        if chunk_ms and sample_rate > 0 and sample_width_bytes > 0:
            chunk_bytes = int(sample_rate * (chunk_ms / 1000.0) * sample_width_bytes * channels)
        else:
            chunk_bytes = 0
        if chunk_bytes <= 0:
            return [audio_data]
        return [
            audio_data[i:i + chunk_bytes]
            for i in range(0, len(audio_data), chunk_bytes)
            if audio_data[i:i + chunk_bytes]
        ]

    async def _stop_audio_capture(self, delay_sec: float = 0.05) -> None:
        if delay_sec > 0:
            await asyncio.sleep(delay_sec)
        self._audio_capture_active = False

    async def disconnect(self) -> RealtimeSessionMetrics:
        if self._audio_track_task:
            self._audio_task_stop.set()
            self._audio_track_task.cancel()
            self._audio_track_task = None

        if self._audio_input_track:
            try:
                self._audio_input_track.stop()
            except Exception:
                pass
            self._audio_input_track = None

        if self._data_channel:
            try:
                self._data_channel.close()
            except Exception:
                pass
            self._data_channel = None

        if self._pc:
            await self._pc.close()
            self._pc = None

        if self._current_session_metrics:
            self._current_session_metrics.t_session_end = time.time()

        return self._current_session_metrics

    def _post_sdp_offer(self, url: str, headers: dict, sdp: str) -> str:
        response = requests.post(url, headers=headers, data=sdp, timeout=30)
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"SDP exchange failed ({response.status_code}): {response.text[:200]}"
            )
        return response.text

    def _save_sdp_files(self, sdp_offer: str, sdp_answer: str, session_prefix: str) -> None:
        """Save SDP offer and answer to disk for analysis."""
        sdp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "sdp")
        os.makedirs(sdp_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        offer_path = os.path.join(sdp_dir, f"{timestamp}_{session_prefix}_offer.sdp")
        answer_path = os.path.join(sdp_dir, f"{timestamp}_{session_prefix}_answer.sdp")

        try:
            with open(offer_path, "w", encoding="utf-8") as f:
                f.write(sdp_offer)
            with open(answer_path, "w", encoding="utf-8") as f:
                f.write(sdp_answer)
            logger.info("SDP files saved: %s, %s", offer_path, answer_path)
        except OSError as e:
            logger.warning("Failed to save SDP files: %s", e)

    async def _consume_audio_track(self, track) -> None:
        if AudioResampler is None:
            logger.warning("PyAV AudioResampler not available; audio capture disabled")
            return
        resampler = AudioResampler(
            format="s16",
            layout="mono",
            rate=self._output_sample_rate,
        )

        try:
            while not self._audio_task_stop.is_set():
                frame = await track.recv()
                frames = resampler.resample(frame)
                for resampled in frames:
                    pcm_bytes = self._frame_to_pcm_bytes(resampled)
                    if not pcm_bytes:
                        continue
                    if not self._audio_capture_active or not self._current_turn_metrics:
                        continue
                    t_recv = time.time()
                    if self._current_turn_metrics.t_first_response is None:
                        self._current_turn_metrics.t_first_response = t_recv
                    self._current_turn_metrics.t_last_response = t_recv
                    self._current_turn_metrics.audio_bytes_received += len(pcm_bytes)
                    self._current_turn_metrics.response_bytes += len(pcm_bytes)

                    audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
                    chunk = RealtimeChunk(
                        content=audio_b64,
                        content_type="audio",
                        timestamp=t_recv,
                        index=len(self._current_turn_metrics.audio_chunks),
                        bytes_received=len(pcm_bytes),
                    )
                    self._current_turn_metrics.audio_chunks.append(chunk)
        except MediaStreamError:
            logger.info("WebRTC audio track ended")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("WebRTC audio capture error: %s", exc)

    @staticmethod
    def _frame_to_pcm_bytes(frame) -> bytes:
        if frame is None:
            return b""
        if hasattr(frame, "to_ndarray"):
            try:
                array = frame.to_ndarray()
                return array.tobytes()
            except Exception:
                pass
        try:
            if frame.planes:
                try:
                    return frame.planes[0].to_bytes()
                except Exception:
                    return bytes(frame.planes[0])
        except Exception:
            return b""
        return b""

    async def _send_event(self, event: dict, track_text_bytes: bool = False) -> None:
        payload = json.dumps(event)
        payload_bytes = len(payload.encode("utf-8"))
        self._data_channel.send(payload)

        if self._current_turn_metrics:
            self._current_turn_metrics.request_bytes += payload_bytes
            if track_text_bytes:
                self._current_turn_metrics.text_bytes_sent += payload_bytes
        if self._current_session_metrics and not self._current_turn_metrics:
            self._current_session_metrics.total_request_bytes += payload_bytes

    async def _wait_for_event(self, event_type: str, timeout: float = 10) -> dict:
        while True:
            raw, raw_bytes, event = await self._recv_event(timeout=timeout)
            self._current_session_metrics.total_response_bytes += raw_bytes
            if event.get("type") == event_type:
                if event_type == "session.created":
                    self._current_session_metrics.session_id = event.get("session", {}).get("id", "")
                return event

    async def _recv_event(self, timeout: float = 10) -> tuple[str, int, dict]:
        raw = await asyncio.wait_for(self._event_queue.get(), timeout=timeout)
        raw_bytes = len(raw.encode("utf-8"))
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            event = {"type": "error", "error": {"message": "Invalid JSON event"}}
        return raw, raw_bytes, event

    async def _process_response_events(self) -> None:
        while True:
            try:
                raw, raw_bytes, event = await self._recv_event(timeout=30)
                t_recv = time.time()

                self._current_turn_metrics.response_bytes += raw_bytes

                event_type = event.get("type", "")

                realtime_event = RealtimeEvent(
                    event_type=event_type,
                    event_id=event.get("event_id"),
                    data=event,
                    timestamp=t_recv,
                    raw_bytes=raw_bytes,
                )
                self._current_turn_metrics.events.append(realtime_event)

                if self._current_turn_metrics.t_first_response is None:
                    if event_type in [
                        "response.text.delta",
                        "response.audio.delta",
                        "response.audio_transcript.delta",
                    ]:
                        self._current_turn_metrics.t_first_response = t_recv

                if event_type == "response.text.delta":
                    delta = event.get("delta", "")
                    self._current_turn_metrics.output_text += delta
                    self._current_turn_metrics.text_bytes_received += len(delta.encode())

                    chunk = RealtimeChunk(
                        content=delta,
                        content_type="text",
                        timestamp=t_recv,
                        index=len(self._current_turn_metrics.text_chunks),
                        bytes_received=raw_bytes,
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
                        bytes_received=raw_bytes,
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
                    response_data = event.get("response", {})
                    if response_data.get("status") == "failed":
                        self._current_turn_metrics.success = False
                        status_details = response_data.get("status_details", {})
                        error = status_details.get("error", {})
                        self._current_turn_metrics.error_message = error.get("message", "Unknown error")
                    break

                elif event_type == "error":
                    self._current_turn_metrics.success = False
                    error = event.get("error", {})
                    self._current_turn_metrics.error_message = error.get("message", str(error))
                    logger.error("Realtime WebRTC error: %s", self._current_turn_metrics.error_message)
                    break

            except Exception as exc:
                self._current_turn_metrics.success = False
                self._current_turn_metrics.error_message = str(exc)
                logger.error("Realtime WebRTC event processing error: %s", exc)
                break
