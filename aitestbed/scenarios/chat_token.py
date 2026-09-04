"""
ChatToken Scenario for the 6G AI Traffic Testbed.

Known issues:
1. response with token ID is not supported in openAI API
2. token ID is treated as str for now
"""

import time
import json
import os
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from .base import BaseScenario, ScenarioResult
from clients.base import ChatMessage, MessageRole, ChatResponse, StreamingResponse
from analysis.logger import LogRecord
from clients.base import estimate_payload_bytes

if TYPE_CHECKING:
    # Only needed for the type hint below; the real import is deferred to
    # _init_tokenizer() so scenarios other than chat_token don't pay the
    # transformers import cost (or need it installed at all).
    from transformers import AutoTokenizer


class ChatTokenScenario(BaseScenario):
    """
    Chat with token ID scenario for measuring conversational AI traffic patterns.

    Supports:
    - Single-turn and multi-turn conversations
    - Streaming and non-streaming responses
    - Multiple prompts per session
    - Input messages converted to token IDs via tokenizers in transformers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer: Optional[AutoTokenizer] = None
        self._tokenizer_initialized = False

    @property
    def scenario_type(self) -> str:
        return "chat_token"

    def _init_tokenizer(self):
        """Initialize tokenizer from config."""
        if self._tokenizer_initialized:
            return
        model_path = self.config.get("tok_path", os.environ.get("TOK_PATH"))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            self._tokenizer_initialized=True
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {model_path}: {str(e)}")

    def _convert_messages_to_dict(self, messages: list[ChatMessage]) -> List[Dict[str, str]]:
        """Convert ChatMessage objects to dict format for tokenizer"""
        dict_messages = []
        for msg in messages:
            role_str = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            if role_str == "MessageRole.SYSTEM":
                role_str = "system"
            elif role_str == "MessageRole.USER":
                role_str = "user"
            elif role_str == "MessageRole.ASSISTANT":
                role_str = "assistant"

            dict_messages.append({
                "role": role_str,
                "content": msg.content
            })
        return dict_messages

    def _get_token_ids(self, messages: list[ChatMessage]) -> List:
        """Convert messages to token IDs"""
        if not self.tokenizer:
            self._init_tokenizer()
        dict_messages = self._convert_messages_to_dict(messages)
        token_ids = self.tokenizer.apply_chat_template(
            dict_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return token_ids

    def _get_eff_response(self, raw_response):
        # replace response text with token ID
        if not self.tokenizer:
            self._init_tokenizer()
        contents = raw_response.choices[0].text
        token_ids = self.tokenizer.encode(contents)
        raw_response.choices[0].text = ", ".join(map(str, token_ids))
        return raw_response.model_dump()


    def _get_eff_stream_response(self, raw_response):
        # replace response text with token ID
        if not self.tokenizer:
            self._init_tokenizer()
        response_bytes = 0
        for chunk_response in raw_response.response_events:
            if len(chunk_response['payload']['choices']) > 0:
                contents = chunk_response['payload']['choices'][0]['text']
                token_ids = self.tokenizer.encode(contents)
                chunk_response['payload']['choices'][0]['text'] = ", ".join(map(str, token_ids))
                chunk_dict = chunk_response['payload'].model_dump() if hasattr(chunk_response['payload'], "model_dump") else chunk_response['payload']
                chunk_bytes = estimate_payload_bytes(chunk_dict)
            else:
                chunk_bytes = estimate_payload_bytes(chunk_response['payload'])
            response_bytes += chunk_bytes
        # Keep the token-ID string for the trace/log payload only. total_content
        # must stay the real decoded assistant text: _run_streaming_turn returns
        # it verbatim and run() stores it in conversation_history, so overwriting
        # it here would feed a token-ID string back into the model as the next
        # turn's chat history instead of what the assistant actually said.
        total_token_ids = self.tokenizer.encode(raw_response.total_content)
        token_id_repr = ", ".join(map(str, total_token_ids))
        raw_response.response_payload = {
            "format": "openai.completions.response_summary",
            "payload": {
                "content": token_id_repr,
                "tokens_in": raw_response.tokens_in,
                "tokens_out": raw_response.tokens_out,
            },
        }
        raw_response.response_bytes = response_bytes
        return raw_response

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """
        Execute a chat session with Token ID conversion.
        """
        try:
            self._init_tokenizer()
        except Exception as e:
            result = ScenarioResult(
                scenario_id=self.scenario_id,
                session_id=self._create_session_id(),
                network_profile=network_profile,
                run_index=run_index,
                success=False,
                error_message=f"Tokenizer init failed: {str(e)}"
            )
            return result


        session_id = self._create_session_id()
        model = self.config.get("model", os.environ.get("MODEL_NAME"))
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
        """Execute a single non-streaming turn with token IDs"""
        try:
            input_ids = self._get_token_ids(messages)
            tokens_in_count = len(input_ids)
            response: ChatResponse = self.client.chat(
                messages=input_ids,
                model=model,
                stream=False,
                max_tokens = 2048,
                extra_body = {
                    "chat_template_kwargs": {"enable_thinking": False}, # for Qwen
                },
            )
            tokens_out = response.tokens_out
            if tokens_out is None:
                tokens_out = self.client.estimate_token_count(response.content or "", model)

            raw_response = response.response_payload # return raw response actually
            eff_response = self._get_eff_response(raw_response)
            response_bytes = estimate_payload_bytes(eff_response)
            response_payload = {
                "format": "openai.completions.response",
                "payload": eff_response,
            }
            request_bytes = response.request_bytes or (tokens_in_count * 4)

            record = self._create_log_record(
                session_id=session_id,
                turn_index=turn_index,
                run_index=run_index,
                network_profile=network_profile,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                tokens_in=tokens_in_count,
                tokens_out=tokens_out,
                t_request_start=t_request_start,
                latency_sec=response.latency_sec,
                http_status=200,
                success=True,
                is_streaming=False,
                trace_request=response.request_payload,
                trace_response=response_payload,
                trace_note="chat_token_sync",
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
        """Execute a single streaming turn with Token IDs"""
        try:
            input_ids = self._get_token_ids(messages)
            tokens_in_count = len(input_ids)
            streaming_response: StreamingResponse = self.client.chat_streaming(
                messages=input_ids,
                model=model,
                max_tokens=2048,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            tokens_out = streaming_response.tokens_out
            if tokens_out is None:
                tokens_out = self.client.estimate_token_count(streaming_response.total_content, model)

            # Calculate inter-chunk times JSON
            inter_chunk_json = json.dumps(streaming_response.inter_chunk_times)

            # Estimate byte sizes
            request_bytes = streaming_response.request_bytes or (tokens_in_count*4)
            eff_stream_response = self._get_eff_stream_response(streaming_response)

            record = self._create_log_record(
                session_id=session_id,
                turn_index=turn_index,
                run_index=run_index,
                network_profile=network_profile,
                request_bytes=request_bytes,
                response_bytes=eff_stream_response.response_bytes,
                tokens_in=tokens_in_count,
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
                trace_response=eff_stream_response.response_payload,
                trace_events=streaming_response.response_events,
                trace_note="chat_token_streaming",
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
