"""Media generation and enhancement transfer contracts. No paid provider calls."""

import base64
from email.parser import BytesParser
from email.policy import default
import io
import json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
import pytest

from clients.base import NullLLMClient
from scenarios import media


class Logger:
    def __init__(self):
        self.records = []

    def log(self, record):
        self.records.append(record)


def png(color="red"):
    data = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(data, format="PNG")
    return data.getvalue()


def response(value, status=200, headers=None):
    data = json.dumps(value).encode() if isinstance(value, dict) else value
    return httpx.Response(status, stream=httpx.ByteStream(data), headers=headers)


def mock_http(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(
        media.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_video_reuses_existing_google_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "existing-test-key")
    media.validate_media_config(video_config())
    assert media.video_api_key(video_config()) == "existing-test-key"


def run(cls, config):
    logger = Logger()
    result = cls(NullLLMClient("media-test"), logger, config).run("no_emulation", 2)
    assert len(logger.records) == 1
    assert result.log_records == logger.records
    return result


def video_config():
    return {
        "type": "video_generation",
        "model": "veo-3.1-generate-preview",
        "prompt": "Ocean waves",
        "poll_interval_sec": 0.001,
    }


def test_video_polls_downloads_and_never_forwards_key_to_storage(monkeypatch, tmp_path):
    requests = []
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.full((24, 32, 3), 120, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    mp4 = buffer.getvalue()

    def handle(request):
        requests.append(request)
        if request.method == "POST":
            assert (
                json.loads(request.content)["instances"][0]["prompt"] == "Ocean waves"
            )
            return response({"name": "models/veo/operations/job-1"})
        if request.url.path.endswith("job-1"):
            if len(requests) == 2:
                return response({"done": False})
            return response(
                {
                    "done": True,
                    "response": {
                        "generateVideoResponse": {
                            "generatedSamples": [
                                {
                                    "video": {
                                        "uri": "https://generativelanguage.googleapis.com/download/video"
                                    }
                                }
                            ]
                        }
                    },
                }
            )
        if request.url.host == "generativelanguage.googleapis.com":
            assert request.headers["x-goog-api-key"] == "test-gemini-secret"
            return response(
                b"",
                302,
                {"location": "https://storage.example/video?signature=private"},
            )
        assert "x-goog-api-key" not in request.headers
        return response(mp4)

    mock_http(monkeypatch, handle)
    result = run(
        media.VideoGenerationScenario, {**video_config(), "output_dir": str(tmp_path)}
    )
    assert result.success and result.api_call_count == 5
    assert result.metadata["output_media_bytes"] == len(mp4)
    assert result.metadata["outputs"][0]["width"] == 32
    assert result.total_response_bytes > len(mp4)
    assert result.metadata["provider_operation_name"].endswith("job-1")
    assert Path(result.metadata["outputs"][0]["path"]).read_bytes() == mp4
    record = result.log_records[0]
    assert record.t_first_token is None and record.t_last_token is None
    assert record.response_bytes_source == "http_body_received"
    assert "private" not in record.metadata and "secret" not in record.metadata
    assert not result.metadata["one_way_transfer_latency_available"]


@pytest.mark.parametrize(
    "final",
    [
        {"done": True, "error": {"message": "secret provider text"}},
        {"done": True, "response": {}},
    ],
)
def test_video_failure_or_filtered_output_is_not_a_success(monkeypatch, final):
    mock_http(
        monkeypatch,
        lambda request: (
            response({"name": "operations/1"})
            if request.method == "POST"
            else response(final)
        ),
    )
    result = run(media.VideoGenerationScenario, video_config())
    assert not result.success and result.metadata["output_media_bytes"] == 0
    assert result.total_response_bytes > 0 and result.api_call_count == 2
    assert "secret provider" not in result.log_records[0].metadata


def test_video_download_failure_preserves_submit_and_poll_traffic(monkeypatch):
    def handle(request):
        if request.method == "POST":
            return response({"name": "operations/1"})
        if request.url.path.endswith("operations/1"):
            return response(
                {
                    "done": True,
                    "response": {
                        "generateVideoResponse": {
                            "generatedSamples": [
                                {"video": {"uri": "https://storage.example/video"}}
                            ]
                        }
                    },
                }
            )
        return response(b"service unavailable", 503)

    mock_http(monkeypatch, handle)
    result = run(media.VideoGenerationScenario, video_config())
    assert not result.success and result.api_call_count == 3
    assert result.log_records[0].http_status == 503
    assert result.total_response_bytes > len(b"service unavailable")


def test_video_polling_has_a_total_deadline(monkeypatch):
    mock_http(
        monkeypatch, lambda request: response({"name": "operations/1", "done": False})
    )
    result = run(
        media.VideoGenerationScenario, {**video_config(), "max_duration_sec": 0.05}
    )
    assert not result.success and "max_duration_sec" in result.error_message
    assert result.total_latency_sec < 1
    assert result.api_call_count > 0 and result.total_response_bytes > 0


def test_image_uploads_all_frames_and_counts_encoding_overhead(monkeypatch, tmp_path):
    inputs = []
    for index, color in enumerate(["red", "green", "blue", "yellow", "black", "white"]):
        path = tmp_path / f"photo_{index}.png"
        path.write_bytes(png(color))
        inputs.append(str(path))
    output = png("orange")

    def handle(request):
        assert request.url.path == "/v1/images/edits"
        assert request.headers["Authorization"] == "Bearer test-openai-secret"
        message = BytesParser(policy=default).parsebytes(
            ("Content-Type: " + request.headers["content-type"] + "\r\n\r\n").encode()
            + request.content
        )
        frames = [
            p
            for p in message.iter_parts()
            if p.get_param("name", header="content-disposition") == "image[]"
        ]
        assert [p.get_payload(decode=True) for p in frames] == [
            Path(p).read_bytes() for p in inputs
        ]
        return response({"data": [{"b64_json": base64.b64encode(output).decode()}]})

    mock_http(monkeypatch, handle)
    result = run(
        media.ImageEnhancementScenario,
        {"type": "image_enhancement", "prompt": "Enhance", "image_paths": inputs},
    )
    assert result.success
    assert result.metadata["input_media_bytes"] == sum(
        Path(p).stat().st_size for p in inputs
    )
    assert result.total_request_bytes > result.metadata["input_media_bytes"]
    assert (
        result.total_response_bytes
        > result.metadata["output_media_bytes"]
        == len(output)
    )
    assert len(result.metadata["inputs"]) == 6
    assert result.metadata["outputs"][0]["width"] == 32
    (phase,) = result.metadata["phases"]
    assert phase["complete"] and phase["t_body_send_first"] <= phase["t_body_send_last"]
    assert phase["t_body_receive_first"] <= phase["t_body_receive_last"]


@pytest.mark.parametrize(
    "result_body",
    [
        {"data": []},
        {"data": [{"b64_json": "invalid"}]},
        {"data": [{"b64_json": base64.b64encode(b"not an image").decode()}]},
    ],
)
def test_image_requires_a_decodable_artifact(monkeypatch, tmp_path, result_body):
    path = tmp_path / "input.png"
    path.write_bytes(png())
    mock_http(monkeypatch, lambda request: response(result_body))
    result = run(
        media.ImageEnhancementScenario,
        {"type": "image_enhancement", "prompt": "Enhance", "image_paths": [str(path)]},
    )
    assert not result.success and result.metadata["outputs"] == []
    assert result.total_request_bytes > 0


def test_missing_input_fails_without_issuing_an_http_request(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("No HTTP client should be created")

    monkeypatch.setattr(media.httpx, "Client", unexpected)
    result = run(
        media.ImageEnhancementScenario,
        {"type": "image_enhancement", "image_paths": ["missing-input.png"]},
    )
    assert not result.success and "Missing" in result.error_message
    assert result.api_call_count == 0


def test_response_size_limit_is_enforced(monkeypatch):
    mock_http(monkeypatch, lambda request: response(b"x" * 1024))
    result = run(
        media.VideoGenerationScenario, {**video_config(), "max_response_bytes": 512}
    )
    assert not result.success and "max_response_bytes" in result.error_message
    assert result.total_response_bytes == 1024


def test_scenario_registration_and_config_are_consistent(monkeypatch, tmp_path):
    import orchestrator

    class Emulator:
        def __init__(self, **kwargs):
            self.interface = "lo"

    monkeypatch.setattr(orchestrator, "NetworkEmulator", Emulator)
    runner = orchestrator.TestbedOrchestrator(
        config_path=media.APP_DIR / "configs/scenarios.yaml",
        profiles_path=media.APP_DIR / "configs/profiles.yaml",
        db_path=str(tmp_path / "test.db"),
    )
    config = runner.scenarios_config["scenarios"]
    for name in media.MEDIA_TYPES:
        instance = runner.scenario_classes[config[name]["type"]](
            runner.get_client(config[name]["provider"]), runner.logger, config[name]
        )
        assert instance.scenario_type == name
        assert config[name]["retry_count"] == 0


def test_openai_visual_assistant_sends_sampled_frames_not_native_video(
    monkeypatch, tmp_path
):
    import av

    video = tmp_path / "clip.mp4"
    with av.open(video, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for i in range(6):
            frame = av.VideoFrame.from_ndarray(
                np.full((24, 32, 3), i * 30, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    uploaded = []

    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        content = json.loads(request.content)["messages"][0]["content"]
        frames = [x["image_url"]["url"] for x in content if x["type"] == "image_url"]
        assert len(frames) == 3
        for uri in frames:
            assert uri.startswith("data:image/jpeg;base64,")
            raw = base64.b64decode(uri.split(",", 1)[1])
            with Image.open(io.BytesIO(raw)) as frame:
                assert frame.size == (32, 24)
            uploaded.append(raw)
        return response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "The frames become brighter."},
                    }
                ]
            }
        )

    mock_http(monkeypatch, handle)
    config = {
        "type": "sa1_visual_assistant",
        "api_style": "openai_frames",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5-mini",
        "video_path": str(video),
        "prompt": "Describe changes",
        "frame_interval_sec": 1,
    }
    result = run(media.VisualAssistantScenario, config)
    assert result.success
    meta = result.metadata
    assert meta["input_media_bytes"] == video.stat().st_size
    assert not meta["native_video_uploaded"] and not meta["audio_uploaded"]
    assert meta["uploaded_image_bytes"] == sum(map(len, uploaded))
    assert [x["timestamp_sec"] for x in meta["sampled_frames"]] == [0, 1, 2]
    assert result.total_request_bytes > meta["uploaded_image_bytes"]
    result = run(media.VisualAssistantScenario, {**config, "max_frames": 1})
    assert not result.success and "max_frames" in result.error_message
    assert result.api_call_count == 0
