"""Media generation and enhancement with explicit HTTP body accounting.

Client HTTP timings include provider processing. They are not one-way packet
or burst transfer delays. Input/output artifacts and HTTP envelopes are counted
separately. Poll responses are control traffic, never generated media.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import time
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image

from .base import BaseScenario, ScenarioResult

APP_DIR = Path(__file__).resolve().parents[1]
MEDIA_TYPES = {"video_generation", "image_enhancement"}


def asset_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else APP_DIR / path


def video_api_key(config):
    name = config.get("api_key_env", "GEMINI_API_KEY")
    key = os.environ.get(name)
    if not key and name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        # Match the existing Gemini client and reuse the configured DGX key.
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return key


def validate_media_config(config: dict, *, credentials: bool = True) -> None:
    """Local preflight only. Does not submit jobs or contact a provider."""
    kind = config["type"]
    duration = float(config.get("max_duration_sec", 540))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("max_duration_sec must be positive and finite")
    limit = int(config.get("max_input_bytes", 128 * 1024 * 1024))
    if limit <= 0 or int(config.get("max_response_bytes", 256 * 1024 * 1024)) <= 0:
        raise ValueError("Media byte limits must be positive")
    if kind in ("image_enhancement", "sa1_image_generation"):
        paths = config.get("image_paths", [])
        if not (0 if kind == "sa1_image_generation" else 1) <= len(paths) <= 16:
            raise ValueError("image_enhancement requires 1 to 16 image_paths")
        key = config.get("api_key_env", "OPENAI_API_KEY")
    elif kind in ("sa1_chatbot", "sa1_visual_assistant"):
        paths = [config["video_path"]] if kind == "sa1_visual_assistant" else []
        key = config.get("api_key_env", "GOOGLE_API_KEY" if paths else "OPENAI_API_KEY")
    elif kind == "video_generation":
        paths = (
            [config["reference_image_path"]]
            if config.get("reference_image_path")
            else []
        )
        key = config.get("api_key_env", "GEMINI_API_KEY")
        poll = float(config.get("poll_interval_sec", 5))
        if not math.isfinite(poll) or poll <= 0:
            raise ValueError("poll_interval_sec must be positive and finite")
        import importlib.util

        if importlib.util.find_spec("av") is None:
            raise ValueError(
                "Video artifact validation requires av from requirements.txt"
            )
    else:
        raise ValueError(f"Unknown media type: {kind}")
    total = 0
    for value in paths:
        path = asset_path(value)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty media input: {path}")
        total += path.stat().st_size
    if total > limit:
        raise ValueError("Media inputs exceed max_input_bytes")
    present = (
        video_api_key(config)
        if kind == "video_generation"
        else os.environ.get(key)
        if key
        else None
    )
    if credentials and key and not present:
        raise ValueError(f"Set {key} before running {kind}")


class MediaError(RuntimeError):
    pass


class _Body(httpx.SyncByteStream):
    def __init__(self, data, event, remaining):
        self.data, self.event, self.remaining = data, event, remaining

    def __iter__(self):
        for offset in range(0, len(self.data), 65536):
            self.remaining()
            block = self.data[offset : offset + 65536]
            self.event.setdefault("t_body_send_first", time.time())
            self.event["request_body_bytes"] += len(block)
            yield block
            self.event["t_body_send_last"] = time.time()


class _Transfer:
    def __init__(self, config):
        self.config = config
        self.deadline = time.monotonic() + float(config.get("max_duration_sec", 540))
        self.phases = []
        self.first_io_monotonic = None
        self.client = httpx.Client(follow_redirects=False, trust_env=False)
        # HTTPX logs complete download URLs at INFO, including signed queries.
        # Phase metadata supplies diagnostics without publishing those URLs.
        self._http_logger = logging.getLogger("httpx")
        self._previous_log_level = self._http_logger.level
        self._http_logger.setLevel(max(logging.WARNING, self._previous_log_level))

    def close(self):
        try:
            self.client.close()
        finally:
            self._http_logger.setLevel(self._previous_log_level)

    def remaining(self):
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Media job exceeded max_duration_sec")
        return seconds

    def exchange(
        self, phase, method, url, *, headers=None, body=b"", content_type=None
    ):
        event = {
            "phase": phase,
            "method": method,
            "t_start": time.time(),
            "request_body_bytes": 0,
            "response_body_bytes": 0,
            "http_status": 0,
            "complete": False,
            "timing_scope": "client_http_body_io",
        }
        self.phases.append(event)
        start = time.monotonic()
        if self.first_io_monotonic is None:
            self.first_io_monotonic = start
        request_headers = {"Accept-Encoding": "identity", **(headers or {})}
        if content_type:
            request_headers["Content-Type"] = content_type
        if body:
            request_headers["Content-Length"] = str(len(body))
        response_body = bytearray()

        def endpoints(stream):
            if stream is None:
                return
            sock = stream.get_extra_info("socket")
            if sock is not None:
                event["local_endpoint"] = list(sock.getsockname()[:2])
                event["remote_endpoint"] = list(sock.getpeername()[:2])

        def trace(name, info):
            if name in (
                "connection.connect_tcp.complete",
                "connection.start_tls.complete",
            ):
                endpoints(info.get("return_value"))

        try:
            with self.client.stream(
                method,
                url,
                headers=request_headers,
                content=_Body(body, event, self.remaining),
                extensions={"trace": trace},
                timeout=self.remaining(),
            ) as response:
                endpoints(response.extensions.get("network_stream"))
                event["http_status"] = response.status_code
                event["t_response_headers"] = time.time()
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise MediaError("Server ignored identity content encoding")
                for chunk in response.iter_raw():
                    self.remaining()
                    if chunk:
                        event.setdefault("t_body_receive_first", time.time())
                        event["response_body_bytes"] += len(chunk)
                        event["t_body_receive_last"] = time.time()
                        if len(response_body) + len(chunk) > int(
                            self.config.get("max_response_bytes", 256 * 1024 * 1024)
                        ):
                            raise MediaError("Response exceeds max_response_bytes")
                        response_body.extend(chunk)
                event["complete"] = True
                # No provider response bodies, signed URLs or credentials in logs.
                if response.status_code >= 400:
                    raise MediaError(f"{phase}: HTTP {response.status_code}")
                for header in ("x-render-processing-sec", "x-input-sha256"):
                    if header in response.headers:
                        event[header] = response.headers[header]
                return bytes(response_body), response.headers, response.status_code
        finally:
            event["t_end"] = time.time()
            event["duration_sec"] = time.monotonic() - start

    def json(self, phase, method, url, *, headers=None, payload=None):
        body = (
            json.dumps(payload, separators=(",", ":")).encode()
            if payload is not None
            else b""
        )
        data, _, status = self.exchange(
            phase,
            method,
            url,
            headers=headers,
            body=body,
            content_type="application/json" if body else None,
        )
        if status >= 300:
            raise MediaError(f"Unexpected redirect in {phase}")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise MediaError(f"Expected an object in {phase}")
        return result

    def download(self, url, api_origin, headers):
        """Follow download redirects without forwarding API keys across origins."""
        for _ in range(6):
            parts = urlsplit(url)
            if parts.scheme != "https" or parts.username or parts.password:
                raise MediaError(
                    "Video download requires an HTTPS URL without userinfo"
                )
            origin = (parts.scheme, parts.netloc)
            body, response_headers, status = self.exchange(
                "download", "GET", url, headers=headers if origin == api_origin else {}
            )
            if status in (301, 302, 303, 307, 308):
                location = response_headers.get("location")
                if not location:
                    raise MediaError("Download redirect has no location")
                url = urljoin(url, location)
                continue
            if status != 200:
                raise MediaError(f"Unexpected download status: {status}")
            return body
        raise MediaError("Too many video download redirects")


class _MediaScenario(BaseScenario):
    @property
    def scenario_type(self):
        return self.config["type"]

    def _input(self, value, metadata):
        path = asset_path(value)
        data = path.read_bytes()
        metadata["inputs"].append(
            {
                "name": path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        return data

    def _output(self, data, suffix, metadata):
        if not data:
            raise MediaError("Provider returned an empty media artifact")
        info = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "format": suffix,
        }
        if suffix in ("png", "jpg", "webp"):
            with Image.open(io.BytesIO(data)) as image:
                info.update(width=image.width, height=image.height, format=image.format)
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
            info["validation"] = "complete_image_decode"
        elif suffix == "txt":
            if not data.decode("utf-8").strip():
                raise MediaError("Empty text response")
            info["validation"] = "nonempty_utf8_text"
        elif suffix == "mp4":
            if len(data) < 12 or data[4:8] != b"ftyp":
                raise MediaError("Downloaded video has no MP4 file signature")
            import av

            try:
                with av.open(io.BytesIO(data)) as container:
                    stream = container.streams.video[0]
                    frame = None
                    decoded_frames = 0
                    for frame in container.decode(stream):
                        decoded_frames += 1
                    if frame is None:
                        raise MediaError("Downloaded MP4 has no video frames")
                    if stream.frames and decoded_frames != stream.frames:
                        raise MediaError("Decoded frame count differs from MP4 index")
                    info.update(
                        width=frame.width,
                        height=frame.height,
                        codec=stream.codec_context.name,
                        duration_sec=container.duration / av.time_base
                        if container.duration
                        else None,
                        decoded_frames=decoded_frames,
                        validation="complete_video_decode",
                    )
            except Exception as exc:
                raise MediaError("Downloaded MP4 cannot be fully decoded") from exc
        if self.config.get("output_dir"):
            directory = asset_path(self.config["output_dir"]) / metadata["operation_id"]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"output_{len(metadata['outputs'])}.{suffix}"
            path.write_bytes(data)
            info["path"] = str(path)
        metadata["outputs"].append(info)
        metadata["t_artifact_validated"] = time.time()
        metadata["_last_artifact_monotonic"] = time.monotonic()

    def run(self, network_profile, run_index=0):
        session = self._create_session_id()
        result = ScenarioResult(self.scenario_id, session, network_profile, run_index)
        meta = {
            "schema": "media_job_v1",
            "operation_id": session,
            "sa1_use_case": self.config.get("sa1_use_case"),
            "workload_kind": self.config.get("workload_kind", "configured_media"),
            "latency_scope": "client_submit_to_validated_artifact",
            "byte_scope": "HTTP bodies, excluding headers/TLS/IP; includes encoding and polling",
            "one_way_transfer_latency_available": False,
            "quality_validation": "not_evaluated",
            "inputs": [],
            "outputs": [],
        }
        started, tick = time.time(), time.monotonic()
        transfer = None
        error = None
        try:
            validate_media_config(self.config)
            transfer = _Transfer(self.config)
            self._execute(transfer, meta)
            if not meta["outputs"]:
                raise MediaError("Job completed without a media artifact")
        except Exception as exc:
            result.success = False
            # HTTP exceptions may include URLs with signed parameters. Keep them out.
            error = (
                str(exc)
                if isinstance(exc, (MediaError, TimeoutError, ValueError))
                else type(exc).__name__
            )
            result.error_message = error
        finally:
            if transfer:
                transfer.close()
        phases = transfer.phases if transfer else []
        meta.update(
            phases=phases,
            input_media_bytes=sum(x["bytes"] for x in meta["inputs"]),
            output_media_bytes=sum(x["bytes"] for x in meta["outputs"]),
        )
        complete = meta.pop("_last_artifact_monotonic", time.monotonic())
        measured_start = transfer.first_io_monotonic if transfer and phases else None
        meta["client_preparation_sec"] = (
            (measured_start - tick) if measured_start else time.monotonic() - tick
        )
        # Exclude fixture reads and encoding.
        # Failed runs retain elapsed time up to the observed failure.
        end = complete if result.success else time.monotonic()
        result.total_latency_sec = end - (measured_start if measured_start else tick)
        result.total_request_bytes = sum(x["request_body_bytes"] for x in phases)
        result.total_response_bytes = sum(x["response_body_bytes"] for x in phases)
        result.api_call_count = len(phases)
        result.turn_count = 1
        result.metadata = meta
        record = self._create_log_record(
            session,
            0,
            run_index,
            network_profile,
            request_bytes=result.total_request_bytes,
            response_bytes=result.total_response_bytes,
            request_bytes_source="http_body_submitted",
            response_bytes_source="http_body_received",
            t_request_start=phases[0]["t_start"] if phases else started,
            latency_sec=result.total_latency_sec,
            # Media arrival is not a token timestamp.
            success=result.success,
            http_status=phases[-1]["http_status"] if phases else 0,
            error_type=error,
            metadata=meta,
        )
        self.logger.log(record)
        result.log_records.append(record)
        return result


class VideoGenerationScenario(_MediaScenario):
    """Gemini Omni and legacy Veo generation with decoded video artifacts."""

    def _execute(self, transfer, meta):
        if self.config.get("api_style") == "interactions":
            return self._interactions(transfer, meta)
        base = self.config.get(
            "base_url", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        headers = {"x-goog-api-key": video_api_key(self.config)}
        instance = {"prompt": self.config["prompt"]}
        if self.config.get("reference_image_path"):
            raw = self._input(self.config["reference_image_path"], meta)
            with Image.open(io.BytesIO(raw)) as img:
                mime = Image.MIME[img.format]
            instance["image"] = {
                "bytesBase64Encoded": base64.b64encode(raw).decode(),
                "mimeType": mime,
            }
        payload = {
            "instances": [instance],
            "parameters": self.config.get("parameters", {}),
        }
        meta["generation_parameters"] = self.config.get("parameters", {})
        meta["poll_interval_sec"] = self.config.get("poll_interval_sec", 5)
        operation = transfer.json(
            "submit",
            "POST",
            f"{base}/models/{self.config['model']}:predictLongRunning",
            headers=headers,
            payload=payload,
        )
        name = operation.get("name", "")
        if not name.startswith("models/") and not name.startswith("operations/"):
            raise MediaError("Provider returned an invalid operation name")
        if any(x in name for x in (":", "?", "#", "..")) or "//" in name:
            raise MediaError("Provider returned an invalid operation name")
        meta["provider_operation_name"] = name
        while not operation.get("done"):
            delay = min(
                float(self.config.get("poll_interval_sec", 5)), transfer.remaining()
            )
            time.sleep(delay)
            operation = transfer.json("poll", "GET", f"{base}/{name}", headers=headers)
        if operation.get("error"):
            raise MediaError("Video generation job failed")
        meta["t_job_ready_observed"] = time.time()
        samples = (
            operation.get("response", {})
            .get("generateVideoResponse", {})
            .get("generatedSamples", [])
        )
        if not samples:
            raise MediaError("Video generation returned no samples (possibly filtered)")
        parts = urlsplit(base)
        for sample in samples:
            uri = sample.get("video", {}).get("uri")
            if not uri:
                raise MediaError("Video generation returned no download URI")
            data = transfer.download(uri, (parts.scheme, parts.netloc), headers)
            self._output(data, "mp4", meta)

    def _interactions(self, transfer, meta):
        """Current Gemini Omni REST API; preserve its different traffic shape."""
        base = self.config.get(
            "base_url", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        content = [{"type": "text", "text": self.config["prompt"]}]
        if self.config.get("reference_image_path"):
            raw = self._input(self.config["reference_image_path"], meta)
            with Image.open(io.BytesIO(raw)) as img:
                mime = Image.MIME[img.format]
            content.append(
                {
                    "type": "image",
                    "mime_type": mime,
                    "data": base64.b64encode(raw).decode(),
                }
            )
        payload = {
            "model": self.config["model"],
            "input": content,
            "response_format": self.config.get("response_format", {"type": "video"}),
        }
        meta["api_style"] = "interactions"
        meta["generation_parameters"] = payload["response_format"]
        data = transfer.json(
            "submit_and_generate",
            "POST",
            base + "/interactions",
            headers={"x-goog-api-key": video_api_key(self.config)},
            payload=payload,
        )
        meta["provider_operation_name"] = data.get("id")
        if data.get("status") != "completed":
            raise MediaError(
                "Video interaction did not complete; do not resubmit automatically"
            )
        for step in data.get("steps", []):
            if step.get("type") == "model_output":
                for item in step.get("content", []):
                    if (
                        item.get("type") == "video"
                        and item.get("mime_type") == "video/mp4"
                    ):
                        self._output(
                            base64.b64decode(item["data"], validate=True), "mp4", meta
                        )


class ImageEnhancementScenario(_MediaScenario):
    """Upload one or more source frames to OpenAI image edits, receive real images."""

    def _execute(self, transfer, meta):
        files = []
        for value in self.config["image_paths"]:
            data = self._input(value, meta)
            with Image.open(io.BytesIO(data)) as img:
                mime = Image.MIME[img.format]
                meta["inputs"][-1].update(
                    width=img.width, height=img.height, mime_type=mime
                )
            files.append(("image[]", (asset_path(value).name, data, mime)))
        fields = {
            "model": self.config.get("model", "gpt-image-1.5"),
            "prompt": self.config["prompt"],
            "size": self.config.get("size", "1024x1024"),
            "quality": self.config.get("quality", "medium"),
            "n": "1",
            "output_format": "png",
        }
        meta["image_parameters"] = {
            key: fields[key] for key in ("size", "quality", "n", "output_format")
        }
        base = self.config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        # Materialize the exact multipart body so its encoding overhead is counted.
        request = httpx.Request(
            "POST", f"{base}/images/edits", data=fields, files=files
        )
        body = request.read()
        headers = {
            "Authorization": "Bearer "
            + os.environ[self.config.get("api_key_env", "OPENAI_API_KEY")]
        }
        raw, _, status = transfer.exchange(
            "upload_and_enhance",
            "POST",
            str(request.url),
            headers=headers,
            body=body,
            content_type=request.headers["content-type"],
        )
        if status != 200:
            raise MediaError(f"Unexpected image edit status: {status}")
        data = json.loads(raw).get("data", [])
        if not data:
            raise MediaError("Image edit returned no image")
        for item in data:
            if not item.get("b64_json"):
                raise MediaError("Image edit did not return inline image data")
            self._output(base64.b64decode(item["b64_json"], validate=True), "png", meta)


class MeasuredImageGenerationScenario(ImageEnhancementScenario):
    """Text-to-image or image-conditioned generation, with identical byte scopes."""

    def _execute(self, transfer, meta):
        if self.config.get("image_paths"):
            return super()._execute(transfer, meta)
        base = self.config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        payload = {
            "model": self.config["model"],
            "prompt": self.config["prompt"],
            "size": self.config.get("size", "1024x1024"),
            "quality": self.config.get("quality", "medium"),
            "n": 1,
            "output_format": "png",
        }
        meta["image_parameters"] = {k: v for k, v in payload.items() if k != "prompt"}
        data = transfer.json(
            "generate_image",
            "POST",
            base + "/images/generations",
            headers={
                "Authorization": "Bearer "
                + os.environ[self.config.get("api_key_env", "OPENAI_API_KEY")]
            },
            payload=payload,
        )
        for item in data.get("data", []):
            self._output(base64.b64decode(item["b64_json"], validate=True), "png", meta)


class MeasuredChatScenario(_MediaScenario):
    """One controlled prompt with streaming or non-streaming HTTP accounting."""

    def _execute(self, transfer, meta):
        base = self.config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        stream = bool(self.config.get("stream", False))
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": self.config["prompt"]}],
            "stream": stream,
            "max_completion_tokens": self.config.get("max_completion_tokens", 512),
        }
        if self.config.get("reasoning_effort"):
            payload["reasoning_effort"] = self.config["reasoning_effort"]
        meta["prompt_utf8_bytes"] = len(self.config["prompt"].encode())
        raw, _, status = transfer.exchange(
            "chat",
            "POST",
            base + "/chat/completions",
            headers={
                "Authorization": "Bearer "
                + os.environ[self.config.get("api_key_env", "OPENAI_API_KEY")]
            },
            body=json.dumps(payload, separators=(",", ":")).encode(),
            content_type="application/json",
        )
        if status != 200:
            raise MediaError(f"Unexpected chat status: {status}")
        text = []
        if stream:
            complete = False
            for line in raw.decode().splitlines():
                if line == "data: [DONE]":
                    complete = True
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    text.extend(
                        c.get("delta", {}).get("content") or ""
                        for c in data.get("choices", [])
                    )
            if not complete:
                raise MediaError("Incomplete SSE response")
        else:
            data = json.loads(raw)
            text = [
                c.get("message", {}).get("content") or ""
                for c in data.get("choices", [])
            ]
        meta["streaming"] = stream
        meta["token_arrival_timing"] = "not_measured; HTTP body receive timing only"
        self._output("".join(text).encode(), "txt", meta)


class VisualAssistantScenario(_MediaScenario):
    """Upload the actual video bytes to Gemini, separately from video generation."""

    def _execute(self, transfer, meta):
        raw = self._input(self.config["video_path"], meta)
        if self.config.get("api_style") == "openai_frames":
            return self._openai_frames(raw, transfer, meta)
        base = self.config.get(
            "base_url", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self.config["prompt"]},
                        {
                            "inline_data": {
                                "mime_type": "video/mp4",
                                "data": base64.b64encode(raw).decode(),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.config.get("max_output_tokens", 256)
            },
        }
        data = transfer.json(
            "upload_and_understand",
            "POST",
            f"{base}/models/{self.config['model']}:generateContent",
            headers={
                "x-goog-api-key": os.environ[
                    self.config.get("api_key_env", "GOOGLE_API_KEY")
                ]
            },
            payload=payload,
        )
        text = "".join(
            p.get("text", "")
            for c in data.get("candidates", [])
            for p in c.get("content", {}).get("parts", [])
            if not p.get("thought")
        )
        meta["provider_processing_sec"] = (
            None  # Not observable at the remote interface.
        )
        self._output(text.encode(), "txt", meta)

    def _openai_frames(self, raw, transfer, meta):
        """An explicit frame-sampling variant, not a native video upload."""
        import av

        interval = float(self.config.get("frame_interval_sec", 1))
        maximum = int(self.config.get("max_frames", 60))
        if not math.isfinite(interval) or interval <= 0 or not 1 <= maximum <= 120:
            raise ValueError("Invalid frame sampling settings")
        content = [{"type": "text", "text": self.config["prompt"]}]
        samples = []
        next_time = 0
        with av.open(io.BytesIO(raw)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                transfer.remaining()
                timestamp = (
                    float(frame.time)
                    if frame.time is not None
                    else index / float(stream.average_rate or 30)
                )
                if timestamp + 1e-6 < next_time:
                    continue
                if len(samples) >= maximum:
                    raise MediaError(
                        "Video exceeds max_frames; adjust the explicit sampling settings"
                    )
                image = frame.to_image()
                image.thumbnail((1280, 1280))
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=90)
                data = encoded.getvalue()
                samples.append(
                    {
                        "timestamp_sec": timestamp,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "width": image.width,
                        "height": image.height,
                    }
                )
                content.extend(
                    [
                        {"type": "text", "text": f"Frame at {timestamp:.3f} seconds:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(data).decode(),
                                "detail": "high",
                            },
                        },
                    ]
                )
                next_time += interval
        if not samples:
            raise MediaError("Video contains no decodable frames")
        meta.update(
            api_style="openai_frames",
            sampled_frames=samples,
            uploaded_image_bytes=sum(x["bytes"] for x in samples),
            native_video_uploaded=False,
            audio_uploaded=False,
            frame_interval_sec=interval,
            workload_kind="sampled_video_frames_proxy",
        )
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": self.config.get("max_output_tokens", 512),
        }
        if "reasoning_effort" in self.config:
            payload["reasoning_effort"] = self.config["reasoning_effort"]
        base = self.config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        data = transfer.json(
            "upload_frames_and_understand",
            "POST",
            base + "/chat/completions",
            headers={
                "Authorization": "Bearer "
                + os.environ[self.config.get("api_key_env", "OPENAI_API_KEY")]
            },
            payload=payload,
        )
        choices = data.get("choices", [])
        if not choices or any(c.get("finish_reason") != "stop" for c in choices):
            raise MediaError("Visual assistant response did not complete normally")
        text = "".join(c.get("message", {}).get("content") or "" for c in choices)
        meta["provider_processing_sec"] = None
        self._output(text.encode(), "txt", meta)
