"""
Image Generation Scenario for the 6G AI Traffic Testbed.

Implements image generation traffic patterns.
"""

import time
import json
from typing import Optional

from .base import BaseScenario, ScenarioResult
from clients.base import ImageResponse
from analysis.logger import LogRecord


class ImageGenerationScenario(BaseScenario):
    """
    Image generation scenario for measuring generative AI traffic patterns.

    Characteristics:
    - Small uplink (text prompts)
    - Large downlink (image data)
    - Higher latency (generation time)
    - Non-streaming response
    """

    @property
    def scenario_type(self) -> str:
        return "image_generation"

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        """Execute image generation requests."""
        session_id = self._create_session_id()
        model = self.config.get("model", "gpt-image-1.5")
        model_alias = self._anonymizer.model_alias(model)
        size = self.config.get("size", "1024x1024")
        quality = self.config.get("quality", "standard")
        prompts = self.config.get("prompts", ["A beautiful sunset over mountains"])

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        try:
            for turn_index, prompt in enumerate(prompts):
                t_request_start = time.time()

                try:
                    response: ImageResponse = self.client.generate_image(
                        prompt=prompt,
                        model=model,
                        size=size,
                        quality=quality
                    )

                    # Calculate image size
                    image_bytes = len(response.image_data) if response.image_data else 0
                    tokens_in = self.client.estimate_tokens(prompt, model)

                    record = self._create_log_record(
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=network_profile,
                        request_bytes=response.request_bytes,
                        response_bytes=image_bytes or response.response_bytes,
                        tokens_in=tokens_in,
                        t_request_start=t_request_start,
                        latency_sec=response.latency_sec,
                        http_status=200,
                        success=True,
                        is_streaming=False,
                        trace_request=getattr(response, "request_payload", None),
                        trace_response=getattr(response, "response_payload", None),
                        trace_note="image_generation",
                        metadata=json.dumps({
                            "model": model_alias,
                            "size": size,
                            "quality": quality,
                            "revised_prompt": response.revised_prompt
                        })
                    )

                    self.logger.log(record)
                    result.log_records.append(record)

                    result.turn_count += 1
                    result.api_call_count += 1
                    result.total_latency_sec += response.latency_sec
                    result.total_request_bytes += response.request_bytes
                    result.total_response_bytes += image_bytes or response.response_bytes

                except NotImplementedError:
                    # Provider doesn't support image generation
                    record = self._create_log_record(
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=network_profile,
                        t_request_start=t_request_start,
                        latency_sec=time.time() - t_request_start,
                        http_status=501,
                        error_type="NotImplemented",
                        success=False,
                    )
                    self.logger.log(record)
                    result.log_records.append(record)
                    result.success = False
                    result.error_message = "Provider does not support image generation"
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

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result
