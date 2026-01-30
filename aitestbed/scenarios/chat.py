"""
Chat Scenario for the 6G AI Traffic Testbed.

Implements interactive chat interactions with streaming support.
"""

import time
import json
from typing import Optional

from .base import BaseScenario, ScenarioResult
from clients.base import ChatMessage, MessageRole, ChatResponse, StreamingResponse
from analysis.logger import LogRecord


class ChatScenario(BaseScenario):
    """
    Chat scenario for measuring conversational AI traffic patterns.

    Supports:
    - Single-turn and multi-turn conversations
    - Streaming and non-streaming responses
    - Multiple prompts per session
    """

    @property
    def scenario_type(self) -> str:
        return "chat"

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """
        Execute a chat session.

        The session processes all configured prompts, building up
        conversation history for multi-turn interactions.
        """
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-5-mini")
        stream = self.config.get("stream", False)
        prompts = self.config.get("prompts", ["Hello, how are you?"])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        conversation_history: list[ChatMessage] = []

        # Add system message if configured
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            conversation_history.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt
            ))

        try:
            for turn_index, prompt in enumerate(prompts):
                # Add user message
                conversation_history.append(ChatMessage(
                    role=MessageRole.USER,
                    content=prompt
                ))

                t_request_start = time.time()

                if stream:
                    # Streaming response
                    record, response_content = self._run_streaming_turn(
                        conversation_history,
                        model,
                        session_id,
                        turn_index,
                        run_index,
                        network_profile,
                        t_request_start
                    )
                else:
                    # Non-streaming response
                    record, response_content = self._run_sync_turn(
                        conversation_history,
                        model,
                        session_id,
                        turn_index,
                        run_index,
                        network_profile,
                        t_request_start
                    )

                # Log the record
                self.logger.log(record)
                result.log_records.append(record)

                # Update result totals
                result.turn_count += 1
                result.api_call_count += 1
                result.total_latency_sec += record.latency_sec
                result.total_request_bytes += record.request_bytes
                result.total_response_bytes += record.response_bytes
                if record.tokens_in:
                    result.total_tokens_in += record.tokens_in
                if record.tokens_out:
                    result.total_tokens_out += record.tokens_out

                # Track TTFT from first streaming turn
                if stream and result.ttft_sec is None and record.t_first_token:
                    result.ttft_sec = record.t_first_token - record.t_request_start

                # Add assistant response to history
                if response_content:
                    conversation_history.append(ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=response_content
                    ))

                if not record.success:
                    result.success = False
                    result.error_message = record.error_type
                    break

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def _run_sync_turn(
        self,
        messages: list[ChatMessage],
        model: str,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        t_request_start: float
    ) -> tuple[LogRecord, str]:
        """Execute a single non-streaming turn."""
        try:
            response: ChatResponse = self.client.chat(
                messages=messages,
                model=model,
                stream=False
            )

            tokens_in = response.tokens_in
            tokens_out = response.tokens_out
            if tokens_in is None:
                tokens_in = self.client.estimate_message_tokens(messages, model)
            if tokens_out is None:
                tokens_out = self.client.estimate_tokens(response.content or "", model)

            record = self._create_log_record(
                session_id=session_id,
                turn_index=turn_index,
                run_index=run_index,
                network_profile=network_profile,
                request_bytes=response.request_bytes,
                response_bytes=response.response_bytes,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                t_request_start=t_request_start,
                latency_sec=response.latency_sec,
                http_status=200,
                success=True,
                is_streaming=False,
                trace_request=response.request_payload,
                trace_response=response.response_payload,
                trace_note="chat_sync",
            )

            return record, response.content

        except Exception as e:
            record = self._create_log_record(
                session_id=session_id,
                turn_index=turn_index,
                run_index=run_index,
                network_profile=network_profile,
                t_request_start=t_request_start,
                latency_sec=time.time() - t_request_start,
                http_status=0,
                error_type=str(e),
                success=False,
                is_streaming=False,
            )
            return record, ""

    def _run_streaming_turn(
        self,
        messages: list[ChatMessage],
        model: str,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        t_request_start: float
    ) -> tuple[LogRecord, str]:
        """Execute a single streaming turn with chunk timing."""
        try:
            streaming_response: StreamingResponse = self.client.chat_streaming(
                messages=messages,
                model=model
            )

            tokens_in = streaming_response.tokens_in
            tokens_out = streaming_response.tokens_out
            if tokens_in is None:
                tokens_in = self.client.estimate_message_tokens(messages, model)
            if tokens_out is None:
                tokens_out = self.client.estimate_tokens(streaming_response.total_content, model)

            # Calculate inter-chunk times JSON
            inter_chunk_json = json.dumps(streaming_response.inter_chunk_times)

            # Estimate byte sizes if not provided by client
            request_bytes = streaming_response.request_bytes or (
                sum(len(m.content.encode()) for m in messages) + 100
            )
            response_bytes = streaming_response.response_bytes or len(
                streaming_response.total_content.encode()
            )

            record = self._create_log_record(
                session_id=session_id,
                turn_index=turn_index,
                run_index=run_index,
                network_profile=network_profile,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                t_request_start=streaming_response.t_request_start,
                t_first_token=streaming_response.t_first_chunk,
                t_last_token=streaming_response.t_last_chunk,
                latency_sec=streaming_response.ttlt or 0.0,
                http_status=200,
                success=True,
                is_streaming=True,
                chunk_count=len(streaming_response.chunks),
                inter_chunk_times=inter_chunk_json,
                trace_request=streaming_response.request_payload,
                trace_response=streaming_response.response_payload,
                trace_events=streaming_response.response_events,
                trace_note="chat_streaming",
            )

            return record, streaming_response.total_content

        except Exception as e:
            record = self._create_log_record(
                session_id=session_id,
                turn_index=turn_index,
                run_index=run_index,
                network_profile=network_profile,
                t_request_start=t_request_start,
                latency_sec=time.time() - t_request_start,
                http_status=0,
                error_type=str(e),
                success=False,
                is_streaming=True,
            )
            return record, ""
