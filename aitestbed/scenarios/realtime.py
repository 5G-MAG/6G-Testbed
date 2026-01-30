"""
Realtime Conversation Scenario for the 6G AI Traffic Testbed.

Implements real-time conversational AI using OpenAI's Realtime API
with WebSocket-based bidirectional streaming for traffic characterization.
"""

import time
import json
import asyncio
import os
import base64
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from .base import BaseScenario, ScenarioResult
from clients.base import LLMClient
from clients.realtime_client import RealtimeClient, RealtimeSessionMetrics, RealtimeTurnMetrics
from clients.realtime_webrtc_client import RealtimeWebRTCClient
from analysis.logger import TrafficLogger, LogRecord
from openai import OpenAI


@dataclass
class TTSAudioResult:
    """Result of a TTS synthesis request."""
    text: str
    audio_bytes: bytes
    request_bytes: int
    response_bytes: int
    latency_sec: float
    t_request_start: float
    success: bool
    error_message: Optional[str] = None
    model: str = ""
    voice: str = ""
    response_format: str = "pcm"


class RealtimeConversationScenario(BaseScenario):
    """
    Real-time conversational AI scenario for measuring WebSocket-based
    bidirectional streaming traffic patterns.

    Supports:
    - Text-based real-time conversations
    - Audio input/output (when configured)
    - Time-to-first-token (TTFT) and time-to-last-token (TTLT) metrics
    - Inter-chunk timing analysis
    - Bidirectional traffic characterization

    Key differences from standard chat:
    - Uses WebSocket instead of HTTP REST
    - Supports persistent bidirectional connection
    - Lower latency for interactive use cases
    - Audio streaming capabilities
    """

    def __init__(
        self,
        client: LLMClient,
        logger: TrafficLogger,
        config: dict
    ):
        """
        Initialize the realtime scenario.

        Note: The `client` parameter is kept for API compatibility but
        this scenario uses its own RealtimeClient internally.
        """
        super().__init__(client, logger, config)
        self._realtime_client: Optional[RealtimeClient] = None
        self._client_factory = RealtimeClient
        self._protocol = "websocket"

    @property
    def scenario_type(self) -> str:
        return "realtime_conversation"

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """
        Execute a realtime conversation session.

        Runs the async conversation in a synchronous wrapper.
        """
        return asyncio.run(self._run_async(network_profile, run_index))

    async def _run_async(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """
        Async implementation of the realtime conversation.
        """
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-realtime-mini")
        prompts = self.config.get("prompts", ["Hello, how are you?"])
        instructions = self.config.get("system_prompt") or self.config.get("instructions")
        voice = self.config.get("voice", "alloy")
        modalities = self.config.get("modalities", ["text", "audio"])
        temperature = self.config.get("temperature", 0.8)

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        # Create realtime client
        self._realtime_client = self._client_factory(model=model)

        try:
            # Connect and configure session
            t_connect_start = time.time()
            await self._realtime_client.connect(
                modalities=modalities,
                voice=voice,
                instructions=instructions,
                temperature=temperature
            )
            t_connected = time.time()

            # Record connection overhead
            session_metrics = self._realtime_client._current_session_metrics
            connection_metadata = {
                "event_type": f"{self._protocol}_connect",
                "modalities": modalities,
                "voice": voice,
                "protocol": self._protocol,
            }
            if session_metrics and session_metrics.sdp_offer:
                connection_metadata.update({
                    "sdp_offer": session_metrics.sdp_offer,
                    "sdp_answer": session_metrics.sdp_answer,
                    "sdp_offer_bytes": session_metrics.sdp_offer_bytes,
                    "sdp_answer_bytes": session_metrics.sdp_answer_bytes,
                    "sdp_offer_hash": session_metrics.sdp_offer_hash,
                    "sdp_answer_hash": session_metrics.sdp_answer_hash,
                    "sdp_negotiation_sec": session_metrics.sdp_negotiation_sec,
                })

            connection_record = self._create_log_record(
                session_id=session_id,
                turn_index=-1,  # Special index for connection
                run_index=run_index,
                network_profile=network_profile,
                request_bytes=session_metrics.total_request_bytes,
                response_bytes=session_metrics.total_response_bytes,
                t_request_start=t_connect_start,
                latency_sec=t_connected - t_connect_start,
                http_status=200,
                success=True,
                is_streaming=True,
                metadata=json.dumps(connection_metadata)
            )
            self.logger.log(connection_record)
            result.log_records.append(connection_record)

            # Send each prompt and collect metrics
            for turn_index, prompt in enumerate(prompts):
                turn_metrics = await self._realtime_client.send_text(prompt)

                # Create log record for this turn
                record = self._create_turn_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    turn_metrics=turn_metrics
                )

                self.logger.log(record)
                result.log_records.append(record)

                # Update result totals
                result.turn_count += 1
                result.api_call_count += 1
                result.total_latency_sec += turn_metrics.total_latency or 0.0
                result.total_request_bytes += turn_metrics.request_bytes
                result.total_response_bytes += turn_metrics.response_bytes

                # Track TTFT from first turn
                if result.ttft_sec is None and turn_metrics.ttft is not None:
                    result.ttft_sec = turn_metrics.ttft

                # Track TTLT
                if turn_metrics.ttlt is not None:
                    result.ttlt_sec = turn_metrics.ttlt

                if not turn_metrics.success:
                    result.success = False
                    result.error_message = turn_metrics.error_message
                    break

            # Disconnect and finalize
            session_metrics = await self._realtime_client.disconnect()

            # Store session-level metadata
            result.metadata = {
                "session_id": session_metrics.session_id,
                "model": self._anonymizer.model_alias(session_metrics.model),
                "session_duration_sec": session_metrics.session_duration,
                "total_audio_bytes_sent": session_metrics.total_audio_bytes_sent,
                "total_audio_bytes_received": session_metrics.total_audio_bytes_received,
                "total_text_bytes_sent": session_metrics.total_text_bytes_sent,
                "total_text_bytes_received": session_metrics.total_text_bytes_received,
                "avg_ttft_sec": session_metrics.avg_ttft,
                "avg_turn_latency_sec": session_metrics.avg_turn_latency,
                "modalities": modalities,
                "voice": voice,
                "protocol": self._protocol
            }
            if session_metrics.sdp_offer:
                result.metadata.update({
                    "sdp_offer_bytes": session_metrics.sdp_offer_bytes,
                    "sdp_answer_bytes": session_metrics.sdp_answer_bytes,
                    "sdp_offer_hash": session_metrics.sdp_offer_hash,
                    "sdp_answer_hash": session_metrics.sdp_answer_hash,
                    "sdp_negotiation_sec": session_metrics.sdp_negotiation_sec,
                })

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Realtime scenario error: {type(e).__name__}: {e}")
            result.success = False
            result.error_message = f"{type(e).__name__}: {e}"

            # Try to disconnect cleanly
            if self._realtime_client:
                try:
                    await self._realtime_client.disconnect()
                except Exception:
                    pass

        return result

    def _create_turn_log_record(
        self,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        turn_metrics: RealtimeTurnMetrics
    ) -> LogRecord:
        """Create a log record from turn metrics."""
        # Build inter-chunk times JSON
        inter_chunk_json = json.dumps(turn_metrics.inter_chunk_times)
        tokens_in, tokens_out = self._estimate_realtime_tokens(turn_metrics)

        # Build metadata for realtime-specific info
        metadata = {
            "input_text": turn_metrics.input_text,
            "output_text": turn_metrics.output_text,
            "input_transcript": turn_metrics.input_transcript,
            "output_transcript": turn_metrics.output_transcript,
            "text_chunks_count": len(turn_metrics.text_chunks),
            "audio_chunks_count": len(turn_metrics.audio_chunks),
            "audio_bytes_sent": turn_metrics.audio_bytes_sent,
            "audio_bytes_received": turn_metrics.audio_bytes_received,
            "text_bytes_sent": turn_metrics.text_bytes_sent,
            "text_bytes_received": turn_metrics.text_bytes_received,
            "event_count": len(turn_metrics.events),
            "function_call_event_count": len(turn_metrics.function_call_events),
            "protocol": self._protocol
        }

        return self._create_log_record(
            session_id=session_id,
            turn_index=turn_index,
            run_index=run_index,
            network_profile=network_profile,
            request_bytes=turn_metrics.request_bytes,
            response_bytes=turn_metrics.response_bytes,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            t_request_start=turn_metrics.t_request_start,
            t_first_token=turn_metrics.t_first_response,
            t_last_token=turn_metrics.t_last_response,
            latency_sec=turn_metrics.total_latency or 0.0,
            http_status=200 if turn_metrics.success else 0,
            error_type=turn_metrics.error_message,
            success=turn_metrics.success,
            is_streaming=True,
            chunk_count=turn_metrics.chunk_count,
            inter_chunk_times=inter_chunk_json,
            metadata=json.dumps(metadata)
        )


class RealtimeWebRTCConversationScenario(RealtimeConversationScenario):
    """
    Real-time conversation using WebRTC transport.

    Uses a WebRTC data channel for realtime events.
    """

    def __init__(self, client: LLMClient, logger: TrafficLogger, config: dict):
        super().__init__(client, logger, config)
        self._client_factory = RealtimeWebRTCClient
        self._protocol = "webrtc"

    @property
    def scenario_type(self) -> str:
        return "realtime_conversation_webrtc"


class RealtimeAudioScenario(BaseScenario):
    """
    Real-time audio conversation scenario.

    Specifically designed for audio-based interactions,
    measuring voice-in/voice-out latencies and audio traffic patterns.
    """

    def __init__(
        self,
        client: LLMClient,
        logger: TrafficLogger,
        config: dict
    ):
        super().__init__(client, logger, config)
        self._realtime_client: Optional[RealtimeClient] = None
        self._tts_client: Optional[OpenAI] = None
        self._client_factory = RealtimeClient
        self._protocol = "websocket"

    @property
    def scenario_type(self) -> str:
        return "realtime_audio"

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """
        Execute a realtime audio conversation.

        This scenario uses pre-recorded audio samples or generates
        synthetic audio for testing audio streaming capabilities.
        """
        return asyncio.run(self._run_async(network_profile, run_index))

    async def _run_async(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """Async implementation for audio scenario."""
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-realtime-mini")
        instructions = self.config.get("system_prompt") or self.config.get("instructions")
        voice = self.config.get("voice", "alloy")
        temperature = self.config.get("temperature", 0.8)
        turn_detection = self.config.get("turn_detection")
        turn_detection_config = turn_detection or {"type": "server_vad"}
        if isinstance(turn_detection_config, dict) and "create_response" not in turn_detection_config:
            turn_detection_config["create_response"] = False
        commit_audio = self.config.get("commit_audio")
        if commit_audio is None:
            commit_audio = True

        # Audio configuration
        audio_format = self.config.get("audio_format", "pcm16")
        sample_rate = self.config.get("sample_rate", 24000)
        audio_samples = self.config.get("audio_samples", [])  # List of audio file paths or base64
        tts_config = self.config.get("tts", {})
        tts_enabled = bool(tts_config.get("enabled"))
        tts_fallback_to_text = bool(tts_config.get("fallback_to_text", True))
        save_audio = bool(self.config.get("save_audio", True))
        tts_model_alias = self._anonymizer.model_alias(tts_config.get("model", "tts-1"))

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        self._realtime_client = self._client_factory(model=model)
        if hasattr(self._realtime_client, "set_audio_config"):
            self._realtime_client.set_audio_config(
                sample_rate=sample_rate,
                sample_width_bytes=2,
                channels=1,
            )

        try:
            # Connect with audio modality
            t_connect_start = time.time()
            await self._realtime_client.connect(
                modalities=["text", "audio"],
                voice=voice,
                instructions=instructions,
                input_audio_format=audio_format,
                output_audio_format=audio_format,
                input_audio_transcription={"model": "whisper-1"},
                turn_detection=turn_detection_config,
                temperature=temperature
            )
            t_connected = time.time()

            # Record connection
            session_metrics = self._realtime_client._current_session_metrics
            connection_metadata = {
                "event_type": f"{self._protocol}_connect",
                "modalities": ["text", "audio"],
                "voice": voice,
                "protocol": self._protocol,
            }
            if session_metrics and session_metrics.sdp_offer:
                connection_metadata.update({
                    "sdp_offer_bytes": session_metrics.sdp_offer_bytes,
                    "sdp_answer_bytes": session_metrics.sdp_answer_bytes,
                    "sdp_offer_hash": session_metrics.sdp_offer_hash,
                    "sdp_answer_hash": session_metrics.sdp_answer_hash,
                    "sdp_negotiation_sec": session_metrics.sdp_negotiation_sec,
                })

            connection_record = self._create_log_record(
                session_id=session_id,
                turn_index=-1,
                run_index=run_index,
                network_profile=network_profile,
                request_bytes=session_metrics.total_request_bytes,
                response_bytes=session_metrics.total_response_bytes,
                t_request_start=t_connect_start,
                latency_sec=t_connected - t_connect_start,
                http_status=200,
                success=True,
                is_streaming=True,
                metadata=json.dumps(connection_metadata),
            )
            self.logger.log(connection_record)
            result.log_records.append(connection_record)

            # If no audio samples provided, optionally generate audio via TTS
            if not audio_samples and tts_enabled:
                prompts = self.config.get("prompts", ["Hello, can you hear me?"])
                for turn_index, prompt in enumerate(prompts):
                    input_audio_bytes = None
                    tts_result = self._synthesize_tts_audio(prompt, tts_config)
                    tts_turn_index = -(turn_index + 2)

                    tts_record = self._create_log_record(
                        session_id=session_id,
                        turn_index=tts_turn_index,
                        run_index=run_index,
                        network_profile=network_profile,
                        request_bytes=tts_result.request_bytes,
                        response_bytes=tts_result.response_bytes,
                        t_request_start=tts_result.t_request_start,
                        latency_sec=tts_result.latency_sec,
                        http_status=200 if tts_result.success else 0,
                        success=tts_result.success,
                        error_type=tts_result.error_message,
                        metadata=json.dumps({
                            "type": "tts_generation",
                            "tts_model": tts_model_alias,
                            "tts_voice": tts_result.voice,
                            "response_format": tts_result.response_format,
                            "tts_speed": tts_config.get("speed"),
                            "prompt_chars": len(prompt),
                            "audio_bytes": len(tts_result.audio_bytes),
                        })
                    )
                    self.logger.log(tts_record)
                    result.log_records.append(tts_record)

                    if not tts_result.success:
                        if not tts_fallback_to_text:
                            result.success = False
                            result.error_message = tts_result.error_message
                            break

                        turn_metrics = await self._realtime_client.send_text(prompt)
                    else:
                        input_audio_bytes = tts_result.audio_bytes
                        turn_metrics = await self._realtime_client.send_audio(
                            tts_result.audio_bytes,
                            commit=commit_audio,
                            chunk_ms=100,
                            sample_rate=sample_rate,
                            sample_width_bytes=2,
                            pace_audio=True,
                        )
                        turn_metrics.input_text = prompt

                    if save_audio:
                        self._attach_audio_logs(
                            session_id=session_id,
                            turn_index=turn_index,
                            run_index=run_index,
                            network_profile=network_profile,
                            input_audio_bytes=input_audio_bytes,
                            turn_metrics=turn_metrics,
                            audio_format=audio_format,
                            sample_rate=sample_rate,
                        )

                    record = self._create_audio_turn_log_record(
                        session_id, turn_index, run_index, network_profile, turn_metrics
                    )
                    self.logger.log(record)
                    result.log_records.append(record)
                    self._update_result_from_turn(result, turn_metrics)

                    if not turn_metrics.success:
                        result.success = False
                        result.error_message = turn_metrics.error_message
                        break

            # If no audio samples provided, fall back to text prompts
            elif not audio_samples:
                prompts = self.config.get("prompts", ["Hello, can you hear me?"])
                for turn_index, prompt in enumerate(prompts):
                    turn_metrics = await self._realtime_client.send_text(prompt)

                    if save_audio:
                        self._attach_audio_logs(
                            session_id=session_id,
                            turn_index=turn_index,
                            run_index=run_index,
                            network_profile=network_profile,
                            input_audio_bytes=None,
                            turn_metrics=turn_metrics,
                            audio_format=audio_format,
                            sample_rate=sample_rate,
                        )

                    record = self._create_audio_turn_log_record(
                        session_id, turn_index, run_index, network_profile, turn_metrics
                    )
                    self.logger.log(record)
                    result.log_records.append(record)
                    self._update_result_from_turn(result, turn_metrics)

                    if not turn_metrics.success:
                        result.success = False
                        result.error_message = turn_metrics.error_message
                        break
            else:
                # Process audio samples
                for turn_index, audio_sample in enumerate(audio_samples):
                    audio_data = self._load_audio_sample(audio_sample)
                    if audio_data is None:
                        continue

                    turn_metrics = await self._realtime_client.send_audio(
                        audio_data,
                        commit=commit_audio,
                        chunk_ms=100,
                        sample_rate=sample_rate,
                        sample_width_bytes=2,
                        pace_audio=True,
                    )

                    if save_audio:
                        self._attach_audio_logs(
                            session_id=session_id,
                            turn_index=turn_index,
                            run_index=run_index,
                            network_profile=network_profile,
                            input_audio_bytes=audio_data,
                            turn_metrics=turn_metrics,
                            audio_format=audio_format,
                            sample_rate=sample_rate,
                        )

                    record = self._create_audio_turn_log_record(
                        session_id, turn_index, run_index, network_profile, turn_metrics
                    )
                    self.logger.log(record)
                    result.log_records.append(record)
                    self._update_result_from_turn(result, turn_metrics)

                    if not turn_metrics.success:
                        result.success = False
                        result.error_message = turn_metrics.error_message
                        break

            # Disconnect
            session_metrics = await self._realtime_client.disconnect()

            result.metadata = {
                "session_id": session_metrics.session_id,
                "model": self._anonymizer.model_alias(session_metrics.model),
                "session_duration_sec": session_metrics.session_duration,
                "total_audio_bytes_sent": session_metrics.total_audio_bytes_sent,
                "total_audio_bytes_received": session_metrics.total_audio_bytes_received,
                "audio_format": audio_format,
                "sample_rate": sample_rate,
                "voice": voice,
                "protocol": self._protocol,
                "tts_enabled": tts_enabled,
            }
            if tts_enabled:
                result.metadata.update({
                    "tts_model": tts_model_alias,
                    "tts_voice": tts_config.get("voice", "alloy"),
                    "tts_response_format": tts_config.get("response_format", "pcm"),
                })

        except Exception as e:
            result.success = False
            result.error_message = str(e)

            if self._realtime_client:
                try:
                    await self._realtime_client.disconnect()
                except Exception:
                    pass

        return result

    def _attach_audio_logs(
        self,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        input_audio_bytes: Optional[bytes],
        turn_metrics: RealtimeTurnMetrics,
        audio_format: str,
        sample_rate: int,
    ) -> None:
        """Persist input/output audio to disk and attach paths to turn metrics."""
        output_audio_bytes = self._collect_output_audio(turn_metrics)
        if not input_audio_bytes and not output_audio_bytes:
            return

        log_dir = Path(self.config.get("audio_log_dir", "logs/audio"))
        log_dir.mkdir(parents=True, exist_ok=True)

        session_tag = session_id.replace("-", "")[:8]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{session_tag}_turn{turn_index:03d}"
        ext = "pcm" if audio_format == "pcm16" else audio_format

        input_path = None
        output_path = None
        try:
            if input_audio_bytes:
                input_path = log_dir / f"{base_name}_input.{ext}"
                input_path.write_bytes(input_audio_bytes)
            if output_audio_bytes:
                output_path = log_dir / f"{base_name}_output.{ext}"
                output_path.write_bytes(output_audio_bytes)

            meta = {
                "scenario_id": self.scenario_id,
                "session_id": session_id,
                "run_index": run_index,
                "turn_index": turn_index,
                "network_profile": network_profile,
                "audio_format": audio_format,
                "sample_rate": sample_rate,
                "input_bytes": len(input_audio_bytes or b""),
                "output_bytes": len(output_audio_bytes or b""),
                "input_path": str(input_path) if input_path else None,
                "output_path": str(output_path) if output_path else None,
                "audio_chunks_count": len(turn_metrics.audio_chunks),
            }
            meta_path = log_dir / f"{base_name}_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2))

            turn_metrics.audio_input_path = str(input_path) if input_path else None
            turn_metrics.audio_output_path = str(output_path) if output_path else None
            turn_metrics.audio_meta_path = str(meta_path)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to save audio logs for turn %s: %s",
                turn_index,
                exc,
            )

    def _collect_output_audio(self, turn_metrics: RealtimeTurnMetrics) -> bytes:
        """Join base64 audio chunks into raw audio bytes."""
        if not turn_metrics.audio_chunks:
            return b""
        output = bytearray()
        for chunk in turn_metrics.audio_chunks:
            if not chunk.content:
                continue
            try:
                output.extend(base64.b64decode(chunk.content))
            except Exception:
                continue
        return bytes(output)

    def _get_tts_client(self) -> OpenAI:
        """Lazily initialize the OpenAI client for TTS."""
        if self._tts_client is None:
            self._tts_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._tts_client

    def _synthesize_tts_audio(self, text: str, tts_config: dict) -> TTSAudioResult:
        """Generate PCM audio bytes from text via OpenAI TTS."""
        model = tts_config.get("model", "tts-1")
        voice = tts_config.get("voice", "alloy")
        response_format = tts_config.get("response_format", "pcm")
        speed = tts_config.get("speed")

        payload = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
        }
        if speed is not None:
            payload["speed"] = speed

        request_bytes = len(json.dumps(payload).encode("utf-8"))
        t_start = time.time()

        try:
            response = self._get_tts_client().audio.speech.create(**payload)
            raw_audio = getattr(response, "content", None)
            if not raw_audio:
                if hasattr(response, "read"):
                    raw_audio = response.read()
                elif hasattr(response, "iter_bytes"):
                    raw_audio = b"".join(response.iter_bytes())
                else:
                    raw_audio = bytes(response)

            if not raw_audio:
                raise ValueError("TTS response contained no audio bytes")

            audio_bytes = raw_audio
            if response_format == "wav":
                audio_bytes = self._extract_pcm_from_wav(raw_audio)
            elif response_format != "pcm":
                raise ValueError(
                    f"Unsupported TTS response_format for realtime input: {response_format}"
                )

            latency = time.time() - t_start
            return TTSAudioResult(
                text=text,
                audio_bytes=audio_bytes,
                request_bytes=request_bytes,
                response_bytes=len(raw_audio),
                latency_sec=latency,
                t_request_start=t_start,
                success=True,
                model=model,
                voice=voice,
                response_format=response_format,
            )
        except Exception as exc:
            return TTSAudioResult(
                text=text,
                audio_bytes=b"",
                request_bytes=request_bytes,
                response_bytes=0,
                latency_sec=time.time() - t_start,
                t_request_start=t_start,
                success=False,
                error_message=str(exc),
                model=model,
                voice=voice,
                response_format=response_format,
            )

    def _extract_pcm_from_wav(self, wav_bytes: bytes) -> bytes:
        """Extract raw PCM16 bytes from a WAV container."""
        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getsampwidth() != 2:
                raise ValueError("Expected 16-bit PCM audio in WAV")
            return wav_file.readframes(wav_file.getnframes())

    def _load_audio_sample(self, audio_sample) -> Optional[bytes]:
        """Load audio sample from file path or base64 string."""
        import base64
        from pathlib import Path

        if isinstance(audio_sample, bytes):
            return audio_sample
        elif isinstance(audio_sample, str):
            if audio_sample.startswith("data:audio"):
                # Base64 data URL
                _, data = audio_sample.split(",", 1)
                return base64.b64decode(data)
            elif Path(audio_sample).exists():
                # File path
                return Path(audio_sample).read_bytes()
            else:
                # Assume raw base64
                try:
                    return base64.b64decode(audio_sample)
                except Exception:
                    return None
        return None

    def _create_audio_turn_log_record(
        self,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        turn_metrics: RealtimeTurnMetrics
    ) -> LogRecord:
        """Create log record for audio turn."""
        inter_chunk_json = json.dumps(turn_metrics.inter_chunk_times)
        tokens_in, tokens_out = self._estimate_realtime_tokens(turn_metrics)

        metadata = {
            "input_transcript": turn_metrics.input_transcript,
            "output_transcript": turn_metrics.output_transcript,
            "audio_bytes_sent": turn_metrics.audio_bytes_sent,
            "audio_bytes_received": turn_metrics.audio_bytes_received,
            "audio_chunks_count": len(turn_metrics.audio_chunks),
            "text_chunks_count": len(turn_metrics.text_chunks),
            "function_call_event_count": len(turn_metrics.function_call_events),
            "event_count": len(turn_metrics.events),
            "protocol": self._protocol,
            "modality": "audio"
        }
        if turn_metrics.audio_input_path:
            metadata["audio_input_path"] = turn_metrics.audio_input_path
        if turn_metrics.audio_output_path:
            metadata["audio_output_path"] = turn_metrics.audio_output_path
        if turn_metrics.audio_meta_path:
            metadata["audio_meta_path"] = turn_metrics.audio_meta_path
        audio_format = self.config.get("audio_format", "pcm16")
        sample_rate = self.config.get("sample_rate", 24000)
        if audio_format == "pcm16" and sample_rate:
            metadata["audio_duration_ms"] = round(
                (turn_metrics.audio_bytes_sent / (sample_rate * 2)) * 1000,
                2,
            )

        if not turn_metrics.success and turn_metrics.events:
            from collections import Counter
            event_types = [event.event_type for event in turn_metrics.events]
            metadata["event_types_head"] = event_types[:15]
            metadata["event_counts"] = dict(Counter(event_types))

        return self._create_log_record(
            session_id=session_id,
            turn_index=turn_index,
            run_index=run_index,
            network_profile=network_profile,
            request_bytes=turn_metrics.request_bytes,
            response_bytes=turn_metrics.response_bytes,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            t_request_start=turn_metrics.t_request_start,
            t_first_token=turn_metrics.t_first_response,
            t_last_token=turn_metrics.t_last_response,
            latency_sec=turn_metrics.total_latency or 0.0,
            http_status=200 if turn_metrics.success else 0,
            error_type=turn_metrics.error_message,
            success=turn_metrics.success,
            is_streaming=True,
            chunk_count=turn_metrics.chunk_count,
            inter_chunk_times=inter_chunk_json,
            metadata=json.dumps(metadata)
        )

    def _update_result_from_turn(
        self,
        result: ScenarioResult,
        turn_metrics: RealtimeTurnMetrics
    ):
        """Update scenario result with turn metrics."""
        result.turn_count += 1
        result.api_call_count += 1
        result.total_latency_sec += turn_metrics.total_latency or 0.0
        result.total_request_bytes += turn_metrics.request_bytes
        result.total_response_bytes += turn_metrics.response_bytes

        if result.ttft_sec is None and turn_metrics.ttft is not None:
            result.ttft_sec = turn_metrics.ttft

        if turn_metrics.ttlt is not None:
            result.ttlt_sec = turn_metrics.ttlt

    def _estimate_realtime_tokens(
        self,
        turn_metrics: RealtimeTurnMetrics
    ) -> tuple[Optional[int], Optional[int]]:
        """Estimate tokens for realtime turns using available text/transcripts."""
        model = self.config.get("model", "gpt-realtime-mini")
        input_text = turn_metrics.input_text or turn_metrics.input_transcript or ""
        output_text = turn_metrics.output_text or turn_metrics.output_transcript or ""

        tokens_in = None
        tokens_out = None
        if input_text:
            tokens_in = self.client.estimate_tokens(input_text, model)
        if output_text:
            tokens_out = self.client.estimate_tokens(output_text, model)

        return tokens_in, tokens_out


class RealtimeAudioWebRTCScenario(RealtimeAudioScenario):
    """Realtime audio scenario using WebRTC transport."""

    def __init__(self, client: LLMClient, logger: TrafficLogger, config: dict):
        super().__init__(client, logger, config)
        self._client_factory = RealtimeWebRTCClient
        self._protocol = "webrtc"

    @property
    def scenario_type(self) -> str:
        return "realtime_audio_webrtc"
