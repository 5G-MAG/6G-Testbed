"""
Video Understanding Scenario for the 6G AI Traffic Testbed.

Executes video + text prompts using providers that support video input
(e.g., vLLM hosting Qwen3-VL).
"""

import json
import time
from pathlib import Path

from .base import BaseScenario, ScenarioResult


class VideoUnderstandingScenario(BaseScenario):
    """
    Video understanding scenario for video + text analysis.
    """

    @property
    def scenario_type(self) -> str:
        return "video_understanding"

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get("model", "Qwen/Qwen3-VL-30B-A3B-Instruct")
        model_alias = self._anonymizer.model_alias(model)
        prompts = self.config.get("prompts", ["Describe the video."])

        video_sources = self._load_video_sources()

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        if not video_sources:
            result.success = False
            result.error_message = "No video_url or video_path configured for video scenario"
            return result

        for turn_index, prompt in enumerate(prompts):
            source = video_sources[min(turn_index, len(video_sources) - 1)]
            video_url = source.get("video_url")
            video_path = source.get("video_path")

            t_request_start = time.time()

            if video_path:
                resolved = self._resolve_video_path(video_path)
                if not resolved.exists():
                    record = self._create_log_record(
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=network_profile,
                        t_request_start=t_request_start,
                        latency_sec=time.time() - t_request_start,
                        http_status=0,
                        error_type=f"Video not found: {resolved}",
                        success=False,
                    )
                    self.logger.log(record)
                    result.log_records.append(record)
                    result.success = False
                    result.error_message = f"Video not found: {resolved}"
                    break
                video_url = f"file://{resolved}"

            try:
                if not hasattr(self.client, "generate_content_with_video"):
                    raise NotImplementedError("Provider does not support video input")

                response = self.client.generate_content_with_video(
                    prompt=prompt,
                    video_url=video_url,
                    model=model,
                )

                tokens_in = response.tokens_in
                tokens_out = response.tokens_out
                if tokens_in is None:
                    tokens_in = self.client.estimate_tokens(prompt, model)
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
                    trace_request=getattr(response, "request_payload", None),
                    trace_response=getattr(response, "response_payload", None),
                    trace_note="video_understanding",
                    metadata=json.dumps({
                        "model": model_alias,
                        "video_url": video_url,
                        "video_path": str(video_path) if video_path else "",
                    })
                )
                self.logger.log(record)
                result.log_records.append(record)

                result.turn_count += 1
                result.api_call_count += 1
                result.total_latency_sec += response.latency_sec
                result.total_request_bytes += response.request_bytes
                result.total_response_bytes += response.response_bytes
                if response.tokens_in:
                    result.total_tokens_in += response.tokens_in
                if response.tokens_out:
                    result.total_tokens_out += response.tokens_out

            except NotImplementedError as exc:
                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    t_request_start=t_request_start,
                    latency_sec=time.time() - t_request_start,
                    http_status=501,
                    error_type=str(exc),
                    success=False,
                )
                self.logger.log(record)
                result.log_records.append(record)
                result.success = False
                result.error_message = str(exc)
                break
            except Exception as exc:
                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    t_request_start=t_request_start,
                    latency_sec=time.time() - t_request_start,
                    http_status=0,
                    error_type=str(exc),
                    success=False,
                )
                self.logger.log(record)
                result.log_records.append(record)
                result.success = False
                result.error_message = str(exc)
                break

        return result

    def _load_video_sources(self) -> list[dict]:
        sources: list[dict] = []

        video_urls = self.config.get("video_urls") or self.config.get("videos") or []
        video_url = self.config.get("video_url")
        if video_url:
            video_urls = [video_url]
        if isinstance(video_urls, str):
            video_urls = [video_urls]
        for url in video_urls:
            sources.append({"video_url": url})

        video_paths = self.config.get("video_paths") or []
        video_path = self.config.get("video_path")
        if video_path:
            video_paths = [video_path]
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        for path in video_paths:
            sources.append({"video_path": path})

        return sources

    def _resolve_video_path(self, video_path: str) -> Path:
        base_dir = self.config.get("video_base_dir")
        path = Path(video_path).expanduser()
        if not path.is_absolute() and base_dir:
            path = Path(base_dir).expanduser() / path
        return path.resolve()
