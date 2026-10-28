#!/usr/bin/env python3
"""
SGLang Engine Server for the 6G AI Traffic Testbed (encoder placement
scenario).

Sidecar wrapping an in-process SGLang Engine (Qwen3-VL). Both modes
upload first and reference by URL, keeping the chat request itself
tiny and directly comparable:

  server (mode "server"): the terminal uploads the original mp4 via
                           PUT /v1/bundles/{id}?kind=video. The chat
                           request references bundle://video/{id}; the
                           server hands the raw bytes to SGLang through
                           video_data with a video_token placeholder in
                           the prompt, so decode + processor + ViT +
                           prefill all run server-side.
  edge   (mode "edge"):   the terminal runs decode + processor + ViT
                          locally, serializes the SGLang-native
                          precomputed_embedding dict (processor output
                          plus the visual tower's final embeddings) and
                          uploads it via PUT /v1/bundles/{id}?kind=edge.
                           The chat request references
                           bundle://edge/{id}; the server deserializes
                            it and passes it to Engine.generate() as
                            input_ids + video_data=[{"format":
                            "precomputed_embedding", ...}], skipping the
                            server-side visual encoder entirely.

The TTFT measurement window is owned by the client (upload/edge work
happens inside it). This server adds a per-request timing breakdown
(t_prepare, t_generate) in the final SSE event so server-side stages
can be attributed without trusting client-side clocks.

Endpoints:
  GET  /v1/models                — OpenAI-style model list
  PUT  /v1/bundles/{bundle_id}   — binary upload (raw mp4 or
                                   torch.save'd precomputed bundle)
  POST /v1/chat/completions      — OpenAI chat completions (SSE
                                   streaming only); messages reference
                                   uploads through bundle:// URLs
  POST /flush_cache              — flush SGLang radix + mm caches
                                     (called by the benchmark harness
                                     between rounds)
  GET  /stats                    — applied --server-cpus budget and the
                                     effective video-sampling config
                                     (lets the harness verify both
                                     landed before trusting a
                                     comparison run)

Video frame sampling (--frame-ratio / --max-frames) is configured at
sidecar STARTUP via the official mm_process_config ServerArg (see
encpl_video_sampling.py); /stats surfaces the effective values.
"""

import argparse
import io
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encpl_video_sampling import (
    build_mm_process_config,
    describe_mm_process_config,
    ensure_sglang_patches,
    format_log_line,
    mm_process_config_help,
    parse_body_sampling,
    verify_frame_ratio_patch,
)


# deep_ep (MoE expert-parallel lib, unneeded for dense Qwen3-VL) crashes at
# import time via JIT init when CUDA_HOME is unset; the AssertionError escapes
# sglang's `except ImportError` guard (token_dispatcher/deepep.py:61) and kills
# `import sglang` entirely. Stub it before any sglang import so the import
# degrades to ImportError -> sglang sets use_deepep=False (dense model unaffected).
sys.modules.setdefault("deep_ep", None)

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SGLangEngineServer")

MSS = 1460


def on_wire(payload_bytes: int) -> tuple[int, int]:
    """TCP overhead model from split_logical.py: payload + 40B/packet + 4."""
    packets = max(1, math.ceil(payload_bytes / MSS))
    return payload_bytes + packets * 40 + 4, packets


class BundleStore:
    """In-memory store for uploaded binaries (raw videos or edge bundles)."""

    def __init__(self, max_bundles: int = 16):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._max_bundles = max_bundles

    def put(self, kind: str, data: bytes, meta: dict | None = None,
            bundle_id: str | None = None) -> str:
        bundle_id = bundle_id or uuid.uuid4().hex
        entry = {
            "kind": kind,
            "size": len(data),
            "data": data,
            "meta": meta or {},
            "created": time.time(),
        }
        with self._lock:
            if len(self._store) >= self._max_bundles:
                oldest = min(self._store, key=lambda k: self._store[k]["created"])
                del self._store[oldest]
            self._store[bundle_id] = entry
        return bundle_id

    def pop(self, bundle_id: str) -> dict:
        with self._lock:
            entry = self._store.pop(bundle_id, None)
        if entry is None:
            raise KeyError(bundle_id)
        return entry


def _patch_qwen3_vl_precomputed_feature():
    """Make Qwen3-VL honor PRECOMPUTED_EMBEDDING items (edge mode).

    Upstream gap (sglang 0.5.20): base_processor.py routes a
    "precomputed_embedding" bundle into MultimodalDataItem(feature=<ViT
    output>, format=PRECOMPUTED_EMBEDDING). Qwen2.5-VL / MiniCPM-V /
    InternVL / Gemma3 all detect that case inside get_*_feature and
    return the feature unchanged (qwen2_5_vl.py:792-797, minicpmv.py:944,
    internvl.py:642), but qwen3_vl.py feeds it straight into the ViT
    (pos_embed shape mismatch crash / silent corruption).

    Patch: wrap get_image_feature/get_video_feature on the Qwen3-VL model
    classes so PRECOMPUTED_EMBEDDING items bypass the visual tower. The
    normal server path (raw pixel features) is untouched. Mirrors the
    deepstack-aware width the terminal's ViT produces (hidden*(1+N)).
    """
    import torch

    from sglang.srt.managers.schedule_batch import MultimodalInputFormat
    from sglang.srt import models as srt_models

    def _precomputed_only(items):
        return bool(items) and all(
            item.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING
            for item in items
        )

    def _pass_through(items):
        # feature rows are the final ViT output (incl. deepstack concat);
        # normalize to [tokens, hidden] exactly like the qwen2_5_vl guard.
        feature = torch.cat([item.feature for item in items], dim=0)
        return feature.reshape(-1, feature.shape[-1])

    def _wrap(cls, name):
        original = getattr(cls, name)

        def patched(self, items):
            if _precomputed_only(items):
                return _pass_through(items)
            return original(self, items)

        setattr(cls, name, patched)

    targets = []
    for attr in dir(srt_models):
        obj = getattr(srt_models, attr)
        if not isinstance(obj, type):
            continue
        if obj.__name__ in (
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen4ExpForConditionalGeneration",
        ):
            targets.append(obj)
    for cls in targets:
        for method in ("get_image_feature", "get_video_feature"):
            if hasattr(cls, method):
                _wrap(cls, method)
    if targets:
        logger.info(
            "qwen3_vl precomputed-embedding patch applied to: %s",
            [c.__name__ for c in targets],
        )
    else:
        logger.warning(
            "qwen3_vl precomputed-embedding patch found no model classes")


def _build_engine(model_path: str, device: str,
                  mm_io_worker_num: int = 0,
                  mm_processor_worker_num: int = 0,
                  mm_process_config: dict | None = None):
    """Create the in-process SGLang Engine (Qwen3-VL, multimodal on).

    mm_io_worker_num / mm_processor_worker_num are OFFICIAL ServerArgs
    knobs (sglang 0.5.20, srt/arg_groups/fields/mm.py:77-88) bounding
    the CPU budget of multimodal preprocessing, which runs inside the
    tokenizer-manager process:
      - io pool (sglang-mm-io threads): video decode + sample + resize
        (VideoDecoderWrapper / preprocess_video); one slot per
        concurrent mm item, so it caps how many videos decode in
        parallel across requests.
      - processor pool: HF processor calls (normalize / tensorize);
        also the auto-parallelism width.
    0 = model-specific default (qwen3_vl: io=16, processor=2 —
    qwen_vl.py:372-373). When the io count is 0, the SGLANG_IO_WORKERS
    env var still overrides it (base_processor.py:318-328), so the
    env-based experiment path keeps working unchanged.
    """
    from sglang.srt.entrypoints.engine import Engine

    _patch_qwen3_vl_precomputed_feature()

    engine_kwargs = dict(
        model_path=model_path,
        device=device,
        enable_multimodal=True,
        trust_remote_code=True,
        # Deterministic benchmark: greedy decoding is set per-request; keep
        # engine-level flags minimal so both modes share one configuration.
        disable_cuda_graph=False,
        mem_fraction_static=0.85,
        log_level="info",
    )
    if mm_io_worker_num:
        engine_kwargs["mm_io_worker_num"] = mm_io_worker_num
    if mm_processor_worker_num:
        engine_kwargs["mm_processor_worker_num"] = mm_processor_worker_num
    if mm_process_config:
        engine_kwargs["mm_process_config"] = mm_process_config

    engine = Engine(**engine_kwargs)
    return engine


app = FastAPI(title="6G SGLang Engine Server")
engine = None
engine_lock = threading.Lock()
store = BundleStore(max_bundles=int(os.environ.get("SGLANG_SERVER_MAX_BUNDLES", "16")))
_model_path: str = ""
# Default min_new_tokens for every request (see chat_completions for
# the zero-token-completion rationale). main() rebuilds it from
# --min-new-tokens; requests can override via body min_new_tokens.
_default_min_new_tokens: int = 1
# --server-cpus spec actually applied by main() ("" = unpinned);
# /stats surfaces it so the harness can verify the server CPU budget
# landed before trusting a comparison run.
_applied_server_cpus: str = ""
# Effective video-sampling knobs (None / 0 = off). Kept for the
# per-request advisory check (parse_body_sampling) and /stats.
_engine_frame_ratio: float | None = None
_engine_max_frames: int | None = None


def _video_decode_backend() -> str:
    """Which video decode backend the ACTIVE sglang picked ("torchcodec"
    / "decord"), read from the video_decoder module's _BACKEND global.

    Lazy import, degrade to "unknown" instead of raising: /stats must
    stay cheap and safe (an engine env with a different layout must
    still get a stats response, not a 500). Complements decode_threads:
    the decode-threads label is historical — the cap applies to BOTH
    backends, and WHICH one is active changes where decode CPU lands
    (torchcodec parallel-instances vs decord's internal pool).
    """
    try:
        from sglang.srt.utils import video_decoder as vd
    except Exception:  # pragma: no cover - engine env broken elsewhere
        logger.exception("video_decoder import failed; backend unknown")
        return "unknown"
    return str(getattr(vd, "_BACKEND", "unknown"))


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_path or "sglang-engine",
            "object": "model",
            "owned_by": "sglang-engine",
        }],
    }


@app.put("/v1/bundles/{bundle_id}")
async def put_bundle(bundle_id: str, request: Request):
    kind = request.query_params.get("kind", "video")
    if kind not in ("video", "edge"):
        raise HTTPException(
            status_code=400,
            detail=f"invalid kind {kind!r}: must be 'video' or 'edge'")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    actual_id = store.put(kind=kind, data=data, bundle_id=bundle_id or None)
    wire_bytes, packets = on_wire(len(data))
    logger.info(
        f"PUT /v1/bundles/{actual_id} kind={kind} size={len(data)} wire={wire_bytes}")
    return {
        "id": actual_id,
        "kind": kind,
        "payload_bytes": len(data),
        "wire_bytes": wire_bytes,
        "packets": packets,
    }


def _parse_bundle_ref(messages: list) -> tuple[str, str]:
    """Extract (kind, id) from OpenAI-style messages containing a bundle:// ref."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            url = None
            for key in ("video_url", "image_url", "bundle_url"):
                block = part.get(key)
                if isinstance(block, dict) and block.get("url"):
                    url = block["url"]
                    break
            if url and url.startswith("bundle://"):
                rest = url[len("bundle://"):]
                kind, _, bundle_id = rest.partition("/")
                return kind, bundle_id
    raise HTTPException(status_code=400, detail="no bundle:// reference in messages")


def _extract_prompt(messages: list) -> str:
    prompt = ""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    prompt = part.get("text", "")
        elif isinstance(content, str):
            prompt = content
    return prompt


# qwen2-vl conv template video_token (conversation.py: qwen2-vl template):
# the engine's chat parser expands a video_url content part to exactly this
# placeholder; video bytes are supplied separately through video_data.
VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"


def _prepare_video_mode(entry: dict, prompt: str):
    """server mode: raw mp4 bytes + video_token placeholder for the engine.

    video_data carries the raw bytes (common.py _normalize_video_input
    accepts bytes directly, no base64 round-trip); the text prompt carries
    the <|vision_start|><|video_pad|><|vision_end|> placeholder the qwen
    processor expands against the decoded video.
    """
    text = (
        VIDEO_TOKEN + prompt if "<|vision_start|>" not in prompt else prompt
    )
    return text, {"video_data": [entry["data"]]}, None


def _maybe_dequant_bundle_feature(bundle: dict) -> float:
    """Dequantize an fp8 edge bundle's feature tensor back to bf16.

    The terminal (clients/encoder_placement_client.py _quantize_feature)
    may upload the (VidCom2-pruned) feature as float8_e4m3fn plus a
    per-tensor `feature_scale` float to halve the upload. Right after
    torch.load we reconstruct feature = scale * to(bf16)(feature_fp8) and
    pop the scale key, so everything downstream — engine-side hashing,
    shm transfer, embed_multimodal_inputs dtype checks — sees exactly the
    bf16 tensors an unquantized run would have produced.

    Returns the dequant wall time (0.0 when the bundle is not
    quantized); raises ValueError when the bundle carries fp8 feature
    bytes this torch cannot cast back (old build without fp8 support).
    """
    import torch

    t_start = time.time()
    scale = bundle.pop("feature_scale", None)
    if scale is None:
        return 0.0
    feature = bundle.get("feature")
    if feature is None or not hasattr(feature, "dtype"):
        raise ValueError(
            "edge bundle has feature_scale but no feature tensor")
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None or str(feature.dtype) not in (
            "torch.float8_e4m3fn", "torch.float8_e4m3"):
        raise ValueError(
            f"edge bundle feature dtype {feature.dtype} with feature_scale "
            f"is not a supported fp8 tensor; rebuild the client or upgrade "
            f"torch")
    try:
        dequant = scale * feature.to(torch.bfloat16)
    except (NotImplementedError, RuntimeError, TypeError):
        # CPU fp8 casts are unsupported on some builds; hop to CUDA
        # (this sidecar always holds the GPU) and cast there.
        if not torch.cuda.is_available():
            raise ValueError(
                "cannot dequantize fp8 edge feature (CPU cast unsupported "
                "and no CUDA device available)") from None
        dequant = (scale * feature.to("cuda").to(torch.bfloat16)).cpu()
    bundle["feature"] = dequant.contiguous()
    return time.time() - t_start


def _prepare_edge_mode(entry: dict, prompt: str):
    """edge mode: torch.save'd precomputed_embedding bundle -> engine kwargs.

    The bundle is a dict produced by the terminal:
      {
        "format": "precomputed_embedding",
        "feature": <visual tower output embeddings>,
        ... full HF processor output (input_ids, video_grid_thw,
            second_per_grid_ts, mrope_positions, mrope_position_delta, ...)
      }
    Per the official pattern (test_vlm_input_format.py
    test_accepts_precomputed_embeddings), the server passes BOTH the
    bundle's input_ids and the dict itself:
      engine.generate(input_ids=..., video_data=[bundle])
    so the terminal's tokenization is authoritative and the server-side
    processor takes the preprocessed fast path (base_processor.py).

    NOTE: the bundle MUST go through video_data, not image_data. The
    preprocessed fast path derives item modality from the receiving list
    (base_processor.py organize_results / load_mm_data fast path); a video
    bundle routed via image_data is tagged Modality.IMAGE, its offsets are
    then scanned against image_token_id, find no <|image_pad|> spans in a
    video prompt, and the precomputed embedding is silently dropped (LLM
    sees only constant video_pad text embeddings -> hallucination).

    Optionally the terminal may upload the feature tensor as fp8 e4m3
    with a per-tensor `feature_scale` (bundle_quant='fp8' + vidcom2);
    _maybe_dequant_bundle_feature restores bf16 here so the engine path
    is byte-identical to an unquantized run.
    """
    import torch

    buf = io.BytesIO(entry["data"])
    bundle = torch.load(buf, map_location="cpu", weights_only=False)
    if (not isinstance(bundle, dict)
            or bundle.get("format") != "precomputed_embedding"):
        raise HTTPException(
            status_code=400, detail="edge bundle is not a precomputed_embedding dict")

    try:
        t_dequant = _maybe_dequant_bundle_feature(bundle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if t_dequant:
        logger.info("edge bundle dequant (fp8 -> bf16): %.3f s", t_dequant)

    input_ids = bundle.get("input_ids")
    if input_ids is not None:
        if hasattr(input_ids, "reshape"):
            input_ids = input_ids.reshape(-1).tolist()
        elif input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        else:
            input_ids = list(input_ids)
    if not input_ids:
        raise HTTPException(
            status_code=400, detail="edge bundle has no usable input_ids")
    return None, {"video_data": [bundle]}, input_ids


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")

    body = await request.json()
    if not body.get("stream", False):
        raise HTTPException(status_code=400, detail="only stream=true is supported")
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.0))
    mode = body.get("mode", "server")
    if mode not in ("server", "edge"):
        raise HTTPException(
            status_code=400,
            detail=f"invalid mode {mode!r}: must be 'server' or 'edge'")

    prompt = _extract_prompt(messages)
    kind, bundle_id = _parse_bundle_ref(messages)
    if kind != ("video" if mode == "server" else "edge"):
        raise HTTPException(
            status_code=400,
            detail=f"mode {mode!r} does not match uploaded bundle kind {kind!r}")
    try:
        entry = store.pop(bundle_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"bundle {bundle_id} not found")

    logger.info(
        f"chat/completions: mode={mode} bundle={bundle_id} "
        f"size={entry['size']} max_tokens={max_tokens}")
    # Advisory: the engine samples with its STARTUP-time config; a body
    # carrying different values would silently mislabel the run.
    parse_body_sampling(body, _engine_frame_ratio, _engine_max_frames)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_name = body.get("model", _model_path)
    created = int(time.time())

    def sse_chunk(delta_content: str | None = None, finish: str | None = None,
                  usage: dict | None = None) -> str:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {} if delta_content is None else {"content": delta_content},
                "finish_reason": finish,
            }],
        }
        if usage is not None:
            chunk["usage"] = usage
        return f"data: {json.dumps(chunk)}\n\n"

    def event_stream():
        stats: dict = {"mode": mode}
        try:
            with engine_lock:
                t_prepare_start = time.time()
                if mode == "server":
                    text, extra, input_ids_arg = _prepare_video_mode(entry, prompt)
                else:
                    text, extra, input_ids_arg = _prepare_edge_mode(entry, prompt)
                t_prepare = time.time() - t_prepare_start
                stats["t_prepare"] = t_prepare

                sampling_params = {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                }
                # Guard against zero-token completions (greedy first-token
                # EOS; see the concurrent sidecar's chat_completions for
                # the full rationale). min_new_tokens>=1 suppresses EOS
                # for the first token(s) only; single-letter MCQ answers
                # ("A") still terminate at token 2. 0 disables the guard.
                min_new_tokens = int(body.get(
                    "min_new_tokens", _default_min_new_tokens))
                if min_new_tokens > 0:
                    sampling_params["min_new_tokens"] = min_new_tokens
                gen_kwargs = dict(sampling_params=sampling_params, **extra)
                if input_ids_arg is not None:
                    # edge mode: terminal tokenization is authoritative
                    # (test_vlm_input_format.py: generate(input_ids=...,
                    # video_data=[precomputed dict])); no text prompt.
                    gen_kwargs["input_ids"] = input_ids_arg
                else:
                    gen_kwargs["prompt"] = text

                t_gen_start = time.time()
                # Engine.generate(stream=True) without
                # incremental_streaming_output yields the CUMULATIVE text
                # snapshot on every step (tokenizer_manager resolves each
                # intermediate chunk to state.get_text() before yielding).
                # The OpenAI SSE contract expects deltas, so convert each
                # snapshot to the newly appended suffix. The first yielded
                # item still marks TTFT (server-side view).
                generator = engine.generate(
                    **gen_kwargs, stream=True,
                )
                first = True
                prev_text = ""
                n_tok = 0
                usage = {"prompt_tokens": None, "completion_tokens": 0}
                finish_reason: str | None = None
                for out in generator:
                    text_piece = out.get("text") or ""
                    if first:
                        stats["server_ttft"] = time.time() - t_gen_start
                        first = False
                    if text_piece.startswith(prev_text):
                        delta = text_piece[len(prev_text):]
                    else:
                        # Defensive: non-prefix text (e.g. if the engine is
                        # ever switched to incremental_streaming_output) is
                        # already a delta; forward as-is.
                        delta = text_piece
                    prev_text = text_piece
                    if delta:
                        n_tok += 1
                        yield sse_chunk(delta_content=delta)
                    meta = out.get("meta_info") or {}
                    pt = meta.get("prompt_tokens")
                    if pt is not None:
                        usage["prompt_tokens"] = pt
                    # completion_tokens is cumulative across chunks; the
                    # last observed value is the authoritative total.
                    ct = meta.get("completion_tokens")
                    if ct is not None:
                        usage["completion_tokens"] = ct
                    # finish_reason likewise arrives on the final output;
                    # keep the last non-None value for diagnostics
                    # ("stop" vs "abort"/"length" distinguishes greedy-
                    # EOS zero-token finishes from scheduler retracts).
                    fr = meta.get("finish_reason")
                    if fr is not None:
                        finish_reason = fr
                if first:
                    stats["server_ttft"] = None
                if usage["completion_tokens"] == 0:
                    # Fallback when the engine never reported a count.
                    usage["completion_tokens"] = n_tok
                stats["n_tokens"] = usage["completion_tokens"]
                stats["server_total"] = time.time() - t_gen_start
                stats["finish_reason"] = finish_reason
                if usage["completion_tokens"] == 0:
                    # Zero-token completion: the stream produced NO content
                    # chunk, so the client records ttft=None (NULL
                    # t_first_token in the DB despite success=1).
                    # finish_reason tells WHY: "stop" = greedy first-token
                    # EOS (mitigate via min_new_tokens), "abort" =
                    # scheduler retract/restart, "length" = min>max edge.
                    logger.warning(
                        f"zero-token completion: request={request_id} "
                        f"mode={mode} finish_reason={finish_reason!r} "
                        f"prompt_tokens={usage['prompt_tokens']}")

            yield sse_chunk(finish="stop", usage=usage)
            yield f"data: {json.dumps({'id': request_id, 'engine_stats': stats})}\n\n"
        except Exception as exc:
            logger.exception("generation failed")
            err = json.dumps({"error": {"message": str(exc)}})
            yield f"data: {err}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/flush_cache")
async def flush_cache():
    """Flush radix cache + mm preprocess cache between benchmark rounds."""
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    import asyncio

    with engine_lock:
        await asyncio.to_thread(engine.flush_cache)
    logger.info("flush_cache: radix + mm caches flushed")
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """CPU-budget + video-sampling observability for the harness."""
    return {
        "server_cpus": _applied_server_cpus or "unpinned",
        "decode_threads": os.environ.get("SGLANG_DECORD_NUM_THREADS")
                          or "decord-native",
        # effective intra-op thread cap for THIS process's CPU-side mm
        # preprocessing (None = native default / flag omitted). env var
        # readback mirrors decode_threads: shows what ACTUALLY applies
        # (a pre-set OMP_NUM_THREADS env would win over the flag).
        "preprocess_threads": os.environ.get("OMP_NUM_THREADS"),
        "video_sampling": {
            "frame_ratio": _engine_frame_ratio,
            "max_frames": _engine_max_frames or "sglang-default",
            "effective": describe_mm_process_config(
                build_mm_process_config(
                    _engine_frame_ratio, _engine_max_frames)),
        },
        # which decode backend the ACTIVE sglang selected at import
        # ("torchcodec" / "decord" / "unknown") — decode_threads above
        # caps BOTH, so naming the backend says where decode CPU lands.
        "video_decode_backend": _video_decode_backend(),
    }


def _apply_server_cpu_limit(spec: str):
    """Pin THIS process (and every child it spawns: the in-process
    Engine, its scheduler subprocess, uvicorn workers) to a shared
    CPU set — the server-arm CPU budget.

    Video decode + preprocessing is CPU-heavy and would otherwise grab
    every core of a many-core host, muting the server-vs-edge contrast
    the encoder-placement scenario measures. Called FIRST in main(),
    before the engine builds, so the scheduler subprocess (forked
    during Engine construction) inherits the affinity and the whole
    server lifetime honors the budget. JIT/warmup also runs inside it
    (a one-off cost, accepted by design).

    spec grammar: comma cpu list with ranges ("0-15", "0-7,32,40-41");
    the special value "auto-half" picks the FIRST HALF of the parent's
    affinity (K = ceil(n/2)) — a sane default that leaves the other
    half for the edge arm when both run on one host. Returns the
    applied spec string ("" when unpinned / unsupported platform).
    """
    if not spec or not str(spec).strip():
        return ""
    text = str(spec).strip().lower()
    try:
        own = os.sched_getaffinity(0)
    except AttributeError:
        logger.warning(
            "--server-cpus given but os.sched_getaffinity is unavailable "
            "on this platform; server CPU budget NOT applied")
        return ""
    if text == "auto-half":
        cpus = sorted(own)
        k = max(1, math.ceil(len(cpus) / 2))
        cpu_list = cpus[:k]
    else:
        cpu_list = set()
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, _, hi_s = part.partition("-")
                try:
                    lo, hi = int(lo_s), int(hi_s)
                except ValueError:
                    raise ValueError(
                        f"bad range {part!r} in --server-cpus {spec!r}")
                if lo < 0 or hi < lo:
                    raise ValueError(
                        f"bad range {part!r} in --server-cpus {spec!r}")
                cpu_list.update(range(lo, hi + 1))
            else:
                try:
                    n = int(part)
                except ValueError:
                    raise ValueError(
                        f"bad cpu id {part!r} in --server-cpus {spec!r}")
                if n < 0:
                    raise ValueError(
                        f"negative cpu id {n} in --server-cpus {spec!r}")
                cpu_list.add(n)
        cpu_list = sorted(cpu_list)
    if not cpu_list:
        return ""
    applied = ",".join(str(c) for c in cpu_list)
    try:
        os.sched_setaffinity(0, set(cpu_list))
    except OSError as exc:
        logger.warning(
            f"cannot pin server to cpus [{applied}] ({exc}); running "
            f"unpinned")
        return ""
    logger.info(f"server CPU budget: pinned to cpus [{applied}] "
                f"(scheduler subprocess inherits this affinity)")
    return applied


def main():
    global engine, _model_path, _applied_server_cpus
    parser = argparse.ArgumentParser(description="6G SGLang Engine Server")
    parser.add_argument("--model", default=os.environ.get(
        "SGLANG_ENGINE_MODEL",
        str(REPO_ROOT / "aicodec" / "aicodec" / "models" / "qwen3_vl_4B")))
    parser.add_argument("--host", default=os.environ.get(
        "SGLANG_ENGINE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(
        os.environ.get("SGLANG_ENGINE_PORT", "8002")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-new-tokens", type=int, default=1, metavar="N",
                        help="suppress EOS for the first N generated tokens "
                             "of every request (default 1). Guards against "
                             "zero-token completions: greedy decoding is "
                             "not batch-invariant, so a request's first "
                             "token can argmax to EOS — the request then "
                             "finishes ok with NO content chunk and the "
                             "client records ttft=NULL in the DB despite "
                             "success=1. 0 disables the guard. Requests "
                             "can override per-call via body "
                             "min_new_tokens.")
    parser.add_argument("--mm-io-worker-num", type=int, default=0, metavar="N",
                        help="threads for multimodal data loading and "
                             "decoding (official ServerArgs knob, Engine "
                             "kwargs passthrough). 0 = model default "
                             "(qwen3_vl: 16). Caps how many videos decode "
                             "in parallel across requests; SGLANG_IO_WORKERS "
                             "env still applies when this is 0.")
    parser.add_argument("--mm-processor-worker-num", type=int, default=0,
                        metavar="N",
                        help="threads for multimodal processor calls "
                             "(official ServerArgs knob). 0 = model default "
                             "(qwen3_vl: 2).")
    parser.add_argument("--server-cpus", default="auto-half", metavar="SPEC",
                        help="server-arm CPU budget: pin this server and its "
                             "scheduler subprocess to a shared cpu set so "
                             "CPU-bound video decode cannot grab the whole "
                             "host (pairs with the edge worker's cpu pin; "
                             "without BOTH caps the server-vs-edge contrast "
                             "is muted on a many-core host). SPEC: comma "
                             "list with ranges ('0-15,32') or 'auto-half' = "
                             "first half of the available cores (default). "
                             "'' disables. Applied BEFORE the engine builds, "
                             "so JIT/warmup pays the budget too (one-off "
                             "cost)")
    parser.add_argument("--frame-ratio", type=float, default=0.0,
                        metavar="R",
                        help=mm_process_config_help(
                            "--frame-ratio R (0.25 = keep 25% of frames)"))
    parser.add_argument("--max-frames", type=int, default=0, metavar="N",
                        help="cap on sampled frames for BOTH sampling modes "
                             "(see --frame-ratio). 0 = sglang default 768 "
                             "(FPS_MAX_FRAMES); an explicit 768 equals the "
                             "default. Native sglang 0.5.20 whitelist key — "
                             "works WITHOUT any sglang patch, alone or "
                             "combined with --frame-ratio. ENGINE-wide via "
                             "mm_process_config: restart to change; /stats "
                             "shows the effective values")
    parser.add_argument("--decode-threads", type=int, default=0, metavar="N",
                        help="cap on decord's INTERNAL decode thread pool "
                             "(applied via the 6G-Testbed sglang patch; "
                             "requires re-running the patcher when the "
                             "installed sglang lacks it). decord sizes its "
                             "pool by HOST core count regardless of the "
                             "--server-cpus affinity pin's core BUDGET and "
                             "of OMP/torch/mm-io thread knobs — this is the "
                             "only true per-decode thread cap. Set it to "
                             "match the edge terminal's decode threads for "
                             "symmetric server-vs-edge CPU comparisons. "
                             "0 = decord native default (no cap). Exported "
                             "as SGLANG_DECORD_NUM_THREADS for the engine "
                             "process and inherited by scheduler children.")
    parser.add_argument("--preprocess-threads", type=int, default=0,
                        metavar="N",
                        help="cap the torch/OMP intra-op thread count for "
                             "CPU-side mm preprocessing in THIS process "
                             "(resize / HF processor / edge-mode torch.load+"
                             "dequant): exports OMP_NUM_THREADS / MKL_NUM_"
                             "THREADS / OPENBLAS_NUM_THREADS (setdefault — a "
                             "pre-set env var wins, mirroring --decode-"
                             "threads) BEFORE the engine (and thus torch) is "
                             "imported, plus torch.set_num_threads(N). "
                             "Without it the main process sizes its pools by "
                             "the --server-cpus core BUDGET (e.g. 32 threads "
                             "on a 32-cpu pin), while the edge terminal caps "
                             "its own pools by its pin — the resize-stage "
                             "thread asymmetry behind 'server decodes "
                             "faster'. Set it to the edge terminal's "
                             "edge_cpus thread count for symmetric server-"
                             "vs-edge CPU comparisons. Scheduler children "
                             "are unaffected (model_runner.py pins them to 1 "
                             "thread regardless). 0 = native default (no "
                             "cap; backwards-compatible).")
    args = parser.parse_args()

    _applied_server_cpus = _apply_server_cpu_limit(args.server_cpus)

    global _default_min_new_tokens
    _default_min_new_tokens = max(0, args.min_new_tokens)

    global _engine_frame_ratio, _engine_max_frames
    _engine_frame_ratio = args.frame_ratio or None
    _engine_max_frames = args.max_frames or None
    # decord thread cap: must be set BEFORE the first VideoReader is built
    # (env read at construction time by the patched video_decoder.py).
    # A pre-existing SGLANG_DECORD_NUM_THREADS wins (explicit env override).
    if args.decode_threads and args.decode_threads > 0:
        os.environ.setdefault("SGLANG_DECORD_NUM_THREADS",
                              str(args.decode_threads))
    # CPU-side preprocess thread cap (resize/processor/dequant intra-op
    # pools): must be applied BEFORE the engine import pulls torch in —
    # the env vars are read at native-lib init (once per process) and
    # serve as torch's set_num_threads default. setdefault per var: a
    # pre-set env value wins (explicit override, mirroring --decode-
    # threads). torch.set_num_threads is the belt-and-braces in-process
    # cap for the case where torch was already initialized (it never is
    # at this point in this module, but the direct-python start path
    # deserves the same safety as the shell one).
    if args.preprocess_threads and args.preprocess_threads > 0:
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(_var, str(args.preprocess_threads))
        import torch

        torch.set_num_threads(args.preprocess_threads)
        logger.info(
            "preprocess threads: capped to %d "
            "(OMP/MKL/OPENBLAS_NUM_THREADS + torch.set_num_threads; "
            "covers resize/processor/edge-dequant in this process)",
            args.preprocess_threads)
    try:
        mm_cfg = build_mm_process_config(
            args.frame_ratio, args.max_frames)
    except ValueError as exc:
        parser.error(str(exc))

    # Auto-load the sglang disk patch (same patcher the start_sidecar*.sh
    # bootstrap runs) BEFORE any sglang import / the canary below — so a
    # direct ``python sglang_engine_server.py`` start is as protected as
    # the shell-wrapped one (fail = abort, like ``patcher || die``).
    ensure_sglang_patches(reason="sglang_engine_server startup")

    # Fail fast (BEFORE the multi-minute engine build) when a frame
    # ratio was requested but the active sglang lacks the patch —
    # otherwise the benchmark would silently sample at 2fps.
    verify_frame_ratio_patch(active=bool(args.frame_ratio))

    _model_path = args.model
    logger.info(f"Starting SGLang Engine (model={args.model} "
                f"mm_io_worker_num={args.mm_io_worker_num} "
                f"mm_processor_worker_num={args.mm_processor_worker_num} "
                f"decode_threads={os.environ.get('SGLANG_DECORD_NUM_THREADS') or 'native'} "
                f"preprocess_threads={os.environ.get('OMP_NUM_THREADS') or 'native'}) ...")
    engine = _build_engine(
        args.model, args.device,
        mm_io_worker_num=args.mm_io_worker_num,
        mm_processor_worker_num=args.mm_processor_worker_num,
        mm_process_config=mm_cfg,
    )
    logger.info(f"SGLang Engine ready (min_new_tokens={_default_min_new_tokens}, "
                f"server_cpus={_applied_server_cpus or 'unpinned'})")
    logger.info(format_log_line(
        args.frame_ratio or None, args.max_frames or None,
        json.dumps(mm_cfg) if mm_cfg else None))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
