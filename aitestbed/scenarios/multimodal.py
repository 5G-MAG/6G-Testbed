"""
Multimodal Scenario for the 6G AI Traffic Testbed.

Executes image + text prompts using providers that support multimodal input.
"""

import json
import time
from pathlib import Path
from .base import BaseScenario, ScenarioResult
class MultimodalScenario(BaseScenario):
    """
    Multimodal scenario for image + text analysis.

    Uses provider-specific image-capable APIs (e.g., Gemini).
    """

    @property
    def scenario_type(self) -> str:
        return "multimodal"

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """Execute multimodal requests."""
        session_id = self._create_session_id()
        model = self.config.get("model", "gemini-3-flash-preview")
        model_alias = self._anonymizer.model_alias(model)
        prompts = self.config.get("prompts", ["Describe the image."])

        image_paths = self._load_image_paths()

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        if not image_paths:
            result.success = False
            result.error_message = "No image_paths configured for multimodal scenario"
            return result

        for turn_index, prompt in enumerate(prompts):
            image_path = image_paths[min(turn_index, len(image_paths) - 1)]
            resolved_path = self._resolve_image_path(image_path)

            t_request_start = time.time()

            if not resolved_path.exists():
                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    t_request_start=t_request_start,
                    latency_sec=time.time() - t_request_start,
                    http_status=0,
                    error_type=f"Image not found: {resolved_path}",
                    success=False,
                )
                self.logger.log(record)
                result.log_records.append(record)
                result.success = False
                result.error_message = f"Image not found: {resolved_path}"
                break

            try:
                if not hasattr(self.client, "generate_content_with_image"):
                    raise NotImplementedError("Provider does not support multimodal input")

                response = self.client.generate_content_with_image(
                    prompt=prompt,
                    image_path=str(resolved_path),
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
                    trace_note="multimodal",
                    metadata=json.dumps({
                        "model": model_alias,
                        "image_path": str(resolved_path),
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

            except NotImplementedError as e:
                record = self._create_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    t_request_start=t_request_start,
                    latency_sec=time.time() - t_request_start,
                    http_status=501,
                    error_type=str(e),
                    success=False,
                )
                self.logger.log(record)
                result.log_records.append(record)
                result.success = False
                result.error_message = str(e)
                break
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
                )
                self.logger.log(record)
                result.log_records.append(record)
                result.success = False
                result.error_message = str(e)
                break

        return result

    def _load_image_paths(self) -> list[str]:
        """Load image paths from config."""
        image_paths = self.config.get("image_paths") or self.config.get("images") or []
        image_path = self.config.get("image_path")
        if image_path:
            image_paths = [image_path]
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        return image_paths

    def _resolve_image_path(self, image_path: str) -> Path:
        """Resolve image path with optional base directory."""
        base_dir = self.config.get("image_base_dir")
        path = Path(image_path).expanduser()
        if not path.is_absolute() and base_dir:
            path = Path(base_dir).expanduser() / path
        return path
