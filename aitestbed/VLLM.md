# Plan: Add Support for vLLM (Qwen3-VL-30B with Native Video Input)

## Goal
Run the testbed against a vLLM server hosting **Qwen3-VL-30B-A3B-Instruct**, with native video input via the OpenAI-compatible API (`video_url` content), and full logging/metrics parity.

## Scope
- Add a `VLLMClient` provider and wire it into `orchestrator.py` and `configs/scenarios.yaml`.
- Support chat + streaming chat through the vLLM OpenAI-compatible API.
- Add **native video input** support for Qwen3-VL-30B (no frame-extraction fallback in the primary path).
- Keep logging/metrics consistent with existing providers (bytes, latency, TTFT/TTLT, tokens).

---

## vLLM Setup (Qwen3-VL-30B, Video Input)

### 1) Prerequisites
- NVIDIA GPU(s) with enough VRAM for Qwen3-VL-30B-A3B-Instruct.
- Working NVIDIA drivers (`nvidia-smi`).
- CUDA-compatible PyTorch environment.

### 2) Create a Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

### 3) Install vLLM
```bash
pip install vllm
```

### 4) Start the vLLM OpenAI-compatible server
The Qwen3-VL-30B tutorial shows a working setup for **video inputs** using the OpenAI-compatible server. The key flag for local video files is `--allowed-local-media-path` (use a directory like `/media`).

```bash
export MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
export VLLM_HOST=0.0.0.0
export VLLM_PORT=8000

vllm serve "$MODEL" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --max-model-len 128000 \
  --allowed-local-media-path /media
```

Optional flags for scale/perf:
- `--tensor-parallel-size N` for multi-GPU.
- `--enable-expert-parallel` for MoE models (if supported by your build).
- `--gpu-memory-utilization 0.9` to fit larger models.

### 5) Video request sanity check
The OpenAI-compatible API accepts **video input** using `video_url` in the message content:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "file:///media/test.mp4"}},
          {"type": "text", "text": "What is in this video?"}
        ]
      }
    ],
    "max_tokens": 100
  }'
```

### 6) Video fetch timeout
vLLM defaults to a **30s video fetch timeout** for remote URLs. You can override it:
```bash
export VLLM_VIDEO_FETCH_TIMEOUT=60
```

---

## Integration Plan (Testbed)

### Phase 0 — Interface Alignment
1) Review `clients/base.py` and align on required methods/data classes.
2) Decide whether to implement a dedicated `VLLMClient` or configure the existing OpenAI client with `base_url` + `api_key`.

### Phase 1 — Provider Implementation (MVP)
3) Implement `clients/vllm_client.py`:
   - `chat()` and `chat_streaming()` using the OpenAI-compatible API.
   - Configurable base URL, model, timeouts.
   - Best-effort token accounting (use `usage` if present; otherwise estimate).
4) Register provider in `clients/__init__.py` and `TestbedOrchestrator.get_client()`.
5) Add a vLLM scenario in `configs/scenarios.yaml` (e.g., `chat_vllm`).

### Phase 2 — Native Video Support (Qwen3-VL-30B)
6) Add a **video scenario** (new or extend `multimodal`) that builds the OpenAI-compatible payload with `video_url` content.
7) Support both:
   - `video_url` for HTTP/HTTPS URLs.
   - `file:///...` for local files (requires server started with `--allowed-local-media-path`).
8) Log video metadata for analysis:
   - video URL/path
   - total input bytes
   - optional frame/sample hints if provided

### Phase 3 — Streaming & Metrics Parity
9) Ensure TTFT/TTLT + inter-chunk timing work for streaming responses.
10) Normalize error handling for retries (timeouts, 5xx, etc.).
11) Document vLLM usage in README/CLAUDE.md.

---

## Config Design (Proposed)

Environment variables:
- `VLLM_BASE_URL` (default: `http://localhost:8000`)
- `VLLM_API_KEY` (optional)

Scenario config examples:
```yaml
chat_vllm:
  type: "chat"
  description: "Chat using vLLM-hosted Qwen3-VL-30B"
  provider: "vllm"
  model: "Qwen/Qwen3-VL-30B-A3B-Instruct"
  stream: true
  prompts:
    - "Summarize recent advances in AI traffic analysis."
```

```yaml
video_understanding_vllm:
  type: "multimodal"
  description: "Video understanding with Qwen3-VL-30B (native video_url)"
  provider: "vllm"
  model: "Qwen/Qwen3-VL-30B-A3B-Instruct"
  stream: false
  video_url: "file:///media/test.mp4"
  prompts:
    - "Describe the main actions in the video."
```

---

## Acceptance Criteria
- `python orchestrator.py --scenario chat_vllm --profile ideal_6g --runs 2` succeeds.
- A video scenario using `video_url` completes successfully on Qwen3-VL-30B.
- Logs in `logs/traffic_logs.db` include latency/bytes/tokens for vLLM runs.
- `RESULTS.md` and `TRACES.md` include vLLM video runs without errors.

---

## Notes / Risks
- Token usage may be missing depending on vLLM version; fallback to local estimation.
- Video inputs increase memory usage; tune `--max-model-len` and GPU utilization as needed.
- Local file access requires `--allowed-local-media-path` and `file:///` URLs.
