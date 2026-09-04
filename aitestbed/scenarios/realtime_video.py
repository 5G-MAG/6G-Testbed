"""
Realtime Video Analysis Scenario for 6G AI Traffic Testbed.

"""

import time
import json
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

from clients.base import LLMClient
from .base import BaseScenario, ScenarioResult
from clients.realtime_webrtc_vlm_client import (
    RealtimeWebRTCVLMClient,
    VLMSessionMetrics,
    VLMTurnMetrics,
)
from analysis.logger import TrafficLogger, LogRecord

logger = logging.getLogger(__name__)


class RealtimeVideoUnderstandingScenario(BaseScenario):
    """
    Real-time video understanding scenario.
    """

    def __init__(
        self,
        client: LLMClient,
        logger: TrafficLogger,
        config: dict,
    ):
        super().__init__(client, logger, config)
        self._vlm_client = None
        self._protocol = "webrtc"

    @property
    def scenario_type(self) -> str:
        return "realtime_video_understanding"

    def run(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        """Execute realtime video understanding session."""
        return asyncio.run(self._run_async(network_profile, run_index))

    async def _run_async(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        """Async implementation with metrics collection."""
        session_id = self._create_session_id()

        # Config
        signaling_host = self.config.get("signaling_host", "127.0.0.1")
        signaling_port = self.config.get("signaling_port", 1234)
        model = self.config.get("model", "vlm-local")
        target_fps = self.config.get("target_fps", 10)
        max_frames = self.config.get("max_frames_per_video")
        task_config = self.config.get("task_config", {})
        dataset_dir = task_config.get("video_paths", "data/videos")
        video_paths = []
        if dataset_dir and Path(dataset_dir).exists():
            dataset_path = Path(dataset_dir)
            subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            valid_subdirs = [
                d for d in subdirs
                if d.name.startswith('sample_') and d.name.split('_')[-1].isdigit()
            ]
            for subdir in sorted(valid_subdirs, key=lambda x: int(x.name.split('_')[-1])):
                for video_file in subdir.glob("*.mp4"):
                    video_paths.append(str(video_file))
                    break
        else:
            video_paths = self.config.get("video_paths", [])
        # print(f"video paths: {video_paths}")
        video_path = [video_paths[run_index % len(video_paths)]]
        print(f"run index: {run_index}, video path: {video_path}")

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        # Create VLM client
        self._vlm_client = RealtimeWebRTCVLMClient(
            signaling_host=signaling_host,
            signaling_port=signaling_port,
            target_fps=target_fps,
            model=model,
            network_profile=network_profile, # for result saving
        )

        try:
            # Connect
            t_connect_start = time.time()
            await self._vlm_client.connect()
            t_connected = time.time()
            print("[Scenario] connect completed")
            # Record connection metrics
            session_metrics = self._vlm_client.session_metrics
            
            connection_metadata = {
                "event_type": f"{self._protocol}_connect",
                "model": model,
                "target_fps": target_fps,
                "protocol": self._protocol,
                "sdp_offer": session_metrics.sdp_offer,
                "sdp_answer": session_metrics.sdp_answer,
                "sdp_offer_bytes": session_metrics.sdp_offer_bytes,
                "sdp_answer_bytes": session_metrics.sdp_answer_bytes,
                "sdp_negotiation_sec": session_metrics.sdp_negotiation_sec,
                "connection_setup_time": session_metrics.connection_setup_time,
                "ice_candidates_received": session_metrics.ice_candidates_received,  
            }

            connection_record = self._create_log_record(
                session_id=session_id,
                turn_index=-1,
                run_index=run_index,
                network_profile=network_profile,
                request_bytes=session_metrics.sdp_offer_bytes,
                response_bytes=session_metrics.sdp_answer_bytes,
                t_request_start=t_connect_start,
                latency_sec=t_connected - t_connect_start,
                http_status=200,
                success=True,
                is_streaming=True,
                metadata=json.dumps(connection_metadata),
            )
            self.logger.log(connection_record)
            result.log_records.append(connection_record)

            # Analyze each video
            # print(f"video path: {video_path}")
            # for turn_index, video_path in enumerate(video_path):
            for turn_index, video_path in enumerate(video_path):
                if not Path(video_path).exists():
                    logger.warning(f"Video file not found: {video_path}")
                    continue
                print(f"video understanding starts, path: {video_path}")
                turn_metrics = await self._vlm_client.analyze_video(
                    video_path=video_path,
                    max_frames=max_frames,
                )

                # Create log record
                record = self._create_turn_log_record(
                    session_id=session_id,
                    turn_index=turn_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    turn_metrics=turn_metrics,
                )

                self.logger.log(record)
                result.log_records.append(record)

                # Update result totals
                tokens_in, tokens_out = self._estimate_vlm_tokens(turn_metrics)
                result.turn_count += 1
                result.api_call_count += 1
                result.total_latency_sec += turn_metrics.total_latency or 0.0
                result.total_request_bytes += turn_metrics.video_bytes_sent
                result.total_response_bytes += turn_metrics.text_bytes_received
                result.total_tokens_in += tokens_in or 0
                result.total_tokens_out += tokens_out or 0

                # Track TTFT
                if result.ttft_sec is None and turn_metrics.ttft is not None:
                    result.ttft_sec = turn_metrics.ttft

                # Track TTLT
                if turn_metrics.ttlt is not None:
                    result.ttlt_sec = turn_metrics.ttlt

                if not turn_metrics.success:
                    result.success = False
                    result.error_message = turn_metrics.error_message
                    break

            # Disconnect
            session_metrics = await self._vlm_client.disconnect()

            # Store session-level metadata
            result.metadata = {
                "session_id": session_metrics.session_id,
                "model": self._anonymizer.model_alias(session_metrics.model),
                "session_duration_sec": session_metrics.session_duration,
                "connection_setup_time": session_metrics.connection_setup_time,
                "sdp_negotiation_sec": session_metrics.sdp_negotiation_sec,
                "total_video_frames_sent": session_metrics.total_video_frames_sent,
                "total_video_bytes_sent": session_metrics.total_video_bytes_sent,
                "total_text_bytes_received": session_metrics.total_text_bytes_received,
                "avg_ttft_sec": session_metrics.avg_ttft,
                "avg_turn_latency_sec": session_metrics.avg_turn_latency,
                "target_fps": target_fps,
                "protocol": self._protocol,
                "ice_candidates_received": session_metrics.ice_candidates_received, 
            }

            try:
                report = self._vlm_client.get_session_report()
                if report:
                    report_path = Path(f"results/reports/tmp_{session_id}_report.json")
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(report_path, "w") as f:
                        json.dump(report, f, indent=2)
                    logger.info(f"Session report saved to {report_path}")
            except Exception as e:
                logger.warning(f"Failed to save session report: {e}")

            except Exception as e:
                logger.warning(f"Failed to save session report: {e}")

            # Do not touch result.success here: ScenarioResult defaults to True
            # (scenarios/base.py), and a failed turn already set it False above
            # (with a break). Unconditionally setting it True here erased that.

        except Exception as e:
            logger.error(f"Scenario error: {type(e).__name__}: {e}")
            result.success = False
            result.error_message = f"{type(e).__name__}: {e}"

            if self._vlm_client:
                try:
                    await self._vlm_client.disconnect()
                except:
                    pass

        return result

    def _create_turn_log_record(
        self,
        session_id: str,
        turn_index: int,
        run_index: int,
        network_profile: str,
        turn_metrics: VLMTurnMetrics,
    ) -> LogRecord:
        """Create log record from turn metrics."""
        # Inter-chunk times
        inter_chunk_json = json.dumps(turn_metrics.inter_chunk_times)

        # Estimate tokens
        tokens_in, tokens_out = self._estimate_vlm_tokens(turn_metrics)

        # Build metadata
        metadata = {
            "video_source": turn_metrics.video_source,
            "output_text_length": len(turn_metrics.output_text),
            "video_frames_sent": turn_metrics.video_frames_sent,
            "video_bytes_sent": turn_metrics.video_bytes_sent,
            "video_resolution": turn_metrics.video_resolution,
            "video_fps": turn_metrics.video_fps,
            "text_chunks_count": turn_metrics.chunk_count,
            "text_bytes_received": turn_metrics.text_bytes_received,
            "ttft": turn_metrics.ttft,
            "ttlt": turn_metrics.ttlt,
            "total_latency": turn_metrics.total_latency,
            "video_upload_rate_mbps": turn_metrics.video_upload_rate_mbps,
            "protocol": self._protocol,
            "modality": "video_to_text",
        }

        return self._create_log_record(
            session_id=session_id,
            turn_index=turn_index,
            run_index=run_index,
            network_profile=network_profile,
            request_bytes=turn_metrics.video_bytes_sent,
            response_bytes=turn_metrics.text_bytes_received,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            t_request_start=turn_metrics.t_turn_start,
            t_first_token=turn_metrics.t_first_token,
            t_last_token=turn_metrics.t_last_token,
            latency_sec=turn_metrics.total_latency or 0.0,
            http_status=200 if turn_metrics.success else 0,
            error_type=turn_metrics.error_message,
            success=turn_metrics.success,
            is_streaming=True,
            chunk_count=turn_metrics.chunk_count,
            inter_chunk_times=inter_chunk_json,
            metadata=json.dumps(metadata),
        )

    def _estimate_vlm_tokens(
        self,
        turn_metrics: VLMTurnMetrics,
    ) -> tuple:
        """Estimate tokens for video understanding turns."""
        frames = turn_metrics.video_frames_sent

        # Rough estimate: 1 frame = 1024 tokens
        tokens_in = frames * 1024 if frames > 0 else None
        # tokens_out = len(turn_metrics.output_text) // 4 if turn_metrics.output_text else None
        tokens_out = turn_metrics.token_count
        
        return tokens_in, tokens_out
