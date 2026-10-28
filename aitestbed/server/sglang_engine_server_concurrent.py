#!/usr/bin/env python3
"""
SGLang Engine Server (concurrent variant) for the 6G AI Traffic Testbed.

New sibling of server/sglang_engine_server.py for the
encoder_placement_concurrent scenario — ZERO changes to the original
sidecar; both can run side by side on different ports.

Differences vs the original (serial) sidecar:

  1. NO whole-generation engine_lock. The original serializes
     _prepare_* + engine.generate() under a global threading.Lock for
     deterministic single-user benchmarks. This variant lets requests
     enter the in-process SGLang Engine concurrently, so its scheduler
     (continuous batching) — not a Python lock — decides the queueing
     behaviour. That queueing is itself the measurement object of the
     concurrent scenario.

     Implemented as an ASYNCIO BRIDGE over the engine's OWN loop. sglang
     0.5.20's in-process Engine creates a private asyncio loop
     (engine.loop, uvloop) that its tokenizer_manager zmq channels are
     bound to; driving it from ANY other running loop (including
     uvicorn's) raises "Cannot run the event loop while another loop is
     running" — measured live on the box. The bridge therefore runs
     engine.loop FOREVER on a dedicated daemon thread
     (EngineLoopBridge) and submits every engine interaction as a
     coroutine via asyncio.run_coroutine_threadsafe(..., engine.loop):

        - each chat request's stream consumption is ONE coroutine on
          the engine loop (`await engine.async_generate(stream=True)`
          — a plain coroutine in 0.5.20 that RETURNS the async
          generator — then `async for` over it); chunks are relayed
          through a queue.Queue to the uvicorn-side async generator.
          N requests = N coroutines on the engine loop, submitted
          microseconds apart — they enter the scheduler
          (tokenizer_manager -> scheduler subprocess) back-to-back and
          continuous batching decides the rest. This is structurally
          the same concurrency model as the standard sglang
          launch_server: one loop, N request coroutines, one scheduler.

     The historical per-step _engine_iter_lock (sync generators
     calling run_until_complete on the engine loop from request
     threads) is RETIRED: it serialized each request's FIRST step
     across submit + full prefill, turning N concurrent prefills into
     a serial staircase.

  2. _prepare_video_mode / _prepare_edge_mode run in the loop's
     thread pool via asyncio.to_thread (edge: torch.load + fp8
     dequant; server: bytes hand-off). They are pure per-request work
     and safe to overlap; only Engine.generate() is the shared GPU
     resource.

  3. /flush_cache is gated by an active-request counter instead of
     the global lock: a flush waits for in-flight generations to drain
     (the drain itself runs on a worker thread so the uvicorn loop
     never blocks) and closes the gate so NEW generations abort with
     a clear SSE error instead of interleaving with the flush. The
     flush coroutine (tokenizer_manager.flush_cache) is submitted to
     the ENGINE loop via the same bridge — the sync Engine.flush_cache
     wrapper is unusable here (run_until_complete on an
     already-running loop). The concurrent scenario flushes once per
     wave, before the barrier release, so the gate is expected to be
     open; the gate exists so an accidentally concurrent flush cannot
     interleave with active generations (flushing while requests are
     in the scheduler is undefined behaviour in sglang 0.5.20).

Everything else — endpoints, wire-byte accounting, SSE chunk format,
engine_stats fields, the qwen3_vl PRECOMPUTED_EMBEDDING patch, fp8
dequant — is byte-for-byte the same contract as the original sidecar
so EncoderPlacementClient works against either server unchanged.

Endpoints (identical to the original sidecar):
  GET  /v1/models                — OpenAI-style model list
  PUT  /v1/bundles/{bundle_id}   — binary upload (raw mp4 or
                                    torch.save'd precomputed bundle)
  POST /v1/chat/completions      — OpenAI chat completions (SSE
                                    streaming only); messages reference
                                    uploads through bundle:// URLs
  POST /flush_cache              — flush SGLang radix + mm caches
                                    (drain-and-gate; called by the
                                    concurrent scenario once per wave)
  GET  /stats                    — active generations + flush gate
                                     state (concurrency observability)
                                     + the effective video-sampling
                                     config

Video frame sampling (--frame-ratio / --max-frames) is configured at
sidecar STARTUP via the official mm_process_config ServerArg (see
encpl_video_sampling.py); /stats surfaces the effective values.
"""

import argparse
import asyncio
import io
import json
import logging
import math
import os
import queue
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
from fastapi.responses import StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SGLangEngineServerConcurrent")

MSS = 1460


def on_wire(payload_bytes: int) -> tuple[int, int]:
    """TCP overhead model from split_logical.py: payload + 40B/packet + 4."""
    packets = max(1, math.ceil(payload_bytes / MSS))
    return payload_bytes + packets * 40 + 4, packets


class BundleStore:
    """In-memory store for uploaded binaries (raw videos or edge bundles).

    Unchanged from the original sidecar: internally locked, so N
    concurrent PUTs (one per terminal user) interleave safely. The
    eviction max must cover one full wave (N bundles) — see
    SGLANG_SERVER_MAX_BUNDLES below.
    """

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


class GenerationGate:
    """Drain-and-gate coordination between generations and flush_cache.

    Replaces the original sidecar's coarse global engine_lock:

      - enter(): registers an active generation. Returns False (caller
        must abort with 503) when a flush is holding the gate.
      - exit(): deregisters; wakes waiting flushes when the last
        generation drains.
      - flush_close(): waits for active generations to drain, then
        closes the gate so NEW generations cannot start; releases in
        flush_open().

    Fairness note: not strictly FIFO between waiting flushes and new
    enter() attempts — a racing enter() may slip in before the gate
    closes. That is benign here (the flush simply waits one more
    generation); strict fairness would add latency for no measurement
    value.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._active = 0
        self._flushing = False
        self._flush_waiters = 0

    def enter(self) -> bool:
        with self._cond:
            if self._flushing:
                return False
            self._active += 1
            return True

    def exit(self):
        with self._cond:
            self._active -= 1
            if self._active <= 0 and self._flush_waiters > 0:
                self._cond.notify_all()

    @property
    def active(self) -> int:
        with self._cond:
            return self._active

    def flush_close(self):
        with self._cond:
            self._flush_waiters += 1
            while self._active > 0:
                self._cond.wait()
            self._flush_waiters -= 1
            self._flushing = True

    def flush_open(self):
        with self._cond:
            self._flushing = False
            self._cond.notify_all()


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

    Identical to the original sidecar's patch (same anchors, same
    behaviour) so edge-mode bundles are accepted by either server.
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


def _build_engine(model_path: str, device: str, mem_fraction_static: float,
                  max_running_requests: int | None,
                  chunked_prefill_size: int | None = None,
                  enable_mixed_chunk: bool | None = None,
                  prefill_max_requests: int | None = None,
                  max_prefill_tokens: int | None = None,
                   enable_radix_cache: bool = False,
                   mm_io_worker_num: int | None = None,
                   mm_processor_worker_num: int | None = None,
                   mm_process_config: dict | None = None):
    """Create the in-process SGLang Engine (Qwen3-VL, multimodal on).

    Same as the original sidecar except the scheduler-shaping knobs are
    CLI-tunable: under concurrency the KV-cache budget
    (mem_fraction_static), the running-request cap
    (max_running_requests), and the prefill batching geometry below
    determine batching behaviour, so a benchmark sweep needs them
    adjustable without editing code.

    All optional knobs are OFFICIAL sglang 0.5.20 ServerArgs fields
    (arg_groups/fields/schedule.py) consumed via the Schedule
    namespace — no sglang source patching involved; passing them as
    Engine kwargs follows the same path the in-process Engine uses
    for every ServerArgs field (engine.py: server_args_class
    (**kwargs)). None (CLI default) OMITS the key entirely so sglang
    applies its own default; a typo would raise TypeError at startup
    (fail fast, not silently ignored).

    knob semantics (sglang defaults when omitted):
      - max_running_requests: cap on requests admitted to the decode
        running batch; None = sglang default (computed from KV-cache
        capacity). 0 would be a falsy value with version-dependent
        semantics, so None is the only safe "unlimited" spelling.
      - chunked_prefill_size: max tokens in ONE prefill chunk
        (v0.5.20 default resolves to 8192 on this box). -1 disables
        chunked prefill entirely (whole prompt in one forward); the
        scheduler normalizes <=0 to None by itself (scheduler.py
        resolves it), so the value is passed through verbatim.
      - enable_mixed_chunk: mix decode reqs into prefill chunks so
        decode does not stall while a long chunked prefill runs.
      - prefill_max_requests: max requests in ONE prefill batch;
        None = unlimited. Bigger values let multiple queued requests
        prefill together (#new-seq > 1) when token budgets allow.
      - max_prefill_tokens: TOTAL token budget of ONE prefill batch
        (v0.5.20 default 16384; the runtime scheduler also caps it at
        >= model context length). With this workload's ~12k-token
        prompts, TWO requests only share one prefill batch when BOTH
        budgets allow it: max_prefill_tokens >= ~2x prompt size AND
        chunked_prefill_size >= ~2x prompt size (or -1). Raising it
        does NOT reduce total GPU work (prefill is compute-bound);
        it only changes the queueing texture — keep comparisons
        within one fixed geometry.
      - mm_io_worker_num: width of the mm I/O thread pool
        (sglang-mm-io) that runs video decode + sample + resize
        (VideoDecoderWrapper / preprocess_video) inside the
        tokenizer-manager process. One pool slot per concurrent mm
        item, so N = at most N of the wave's videos decode in
        PARALLEL across requests; the rest queue before even
        reaching the scheduler — the server-arm preprocessing
        budget knob under concurrency. qwen3_vl default: 16
        (qwen_vl.py:372-373). When omitted/0, the SGLANG_IO_WORKERS
        env var still overrides the default (base_processor.py:318-328).
      - mm_processor_worker_num: width of the multimodal processor
        pool (HF processor calls: normalize / tensorize; qwen3_vl
        default 2). 1 = synchronous processing (no pool at all).
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
        mem_fraction_static=mem_fraction_static,
        log_level="info",
        # Cold-prefill benchmark semantics: the wave's N users send the
        # same video + prompt, so with radix cache ON the first user to
        # finish prefill inserts the whole 14.5k-token prefix and every
        # later user hits ~100% cached-token (measured: #cached-token
        # 14533 / #new-token 1) — their TTFT collapses to decode-only
        # and the "N concurrent cold prefills queue on the GPU" object
        # of this scenario disappears. OFF unless --enable-radix-cache.
        # (mm preprocess cache is unaffected; flush_cache between waves
        # stays for that and stays correct either way.)
        disable_radix_cache=not enable_radix_cache,
    )
    if max_running_requests is not None:
        engine_kwargs["max_running_requests"] = max_running_requests
    if chunked_prefill_size is not None:
        engine_kwargs["chunked_prefill_size"] = chunked_prefill_size
    if enable_mixed_chunk is not None:
        engine_kwargs["enable_mixed_chunk"] = enable_mixed_chunk
    if prefill_max_requests is not None:
        engine_kwargs["prefill_max_requests"] = prefill_max_requests
    if max_prefill_tokens is not None:
        engine_kwargs["max_prefill_tokens"] = max_prefill_tokens
    if mm_io_worker_num is not None:
        engine_kwargs["mm_io_worker_num"] = mm_io_worker_num
    if mm_processor_worker_num is not None:
        engine_kwargs["mm_processor_worker_num"] = mm_processor_worker_num
    if mm_process_config:
        engine_kwargs["mm_process_config"] = mm_process_config
    engine = Engine(**engine_kwargs)
    return engine


app = FastAPI(title="6G SGLang Engine Server (concurrent)")
engine = None
gate = GenerationGate()
# BundleStore capacity is CLI-tunable (--max-bundles, default 64: one full
# wave of N users uploads N bundles before any chat request pops them, so
# the store must hold a wave without evicting still-referenced entries;
# the original serial sidecar hardcodes 16). main() rebuilds this global
# from the parsed args; the module-level default keeps import-time use
# (tests, /stats before main) consistent.
# Default raised vs the original (16): one full wave of N users uploads
# N bundles before any chat request pops them, so the store must hold a
# wave without evicting still-referenced entries.
store = BundleStore(max_bundles=64)
_model_path: str = ""
# --server-cpus spec actually applied by main() ("" = unpinned);
# /stats surfaces it so the harness can verify the server CPU budget
# landed before trusting a comparison run.
_applied_server_cpus: str = ""
# Effective video-sampling knobs (None / 0 = off). Kept for the
# per-request advisory check (parse_body_sampling) and /stats.
_engine_frame_ratio: float | None = None
_engine_max_frames: int | None = None
# Default min_new_tokens for every request (see chat_completions for
# the zero-token-completion rationale). main() rebuilds it from
# --min-new-tokens; requests can override via body min_new_tokens.
_default_min_new_tokens: int = 1

# Parallel video preprocess (6G-Testbed sglang patch 4, default OFF =
# upstream serial behavior). The authoritative state is the module
# global inside sglang.srt.multimodal.processors.qwen_vl (read at EVERY
# preprocess call, so flipping it needs no engine restart); this string
# is only what /stats echoes and the endpoint flips. Values: "on"/"off"
# (None = sglang lacks patch 4 — endpoint refuses to flip).
_parallel_video_preprocess: str | None = None


def _qwen_vl_preprocess_module():
    """Import the (patched) qwen_vl processor module lazily.

    Returns None when the ACTIVE sglang lacks the 6G-Testbed patch 4
    (no _PARALLEL_VIDEO_PREPROCESS attribute) — every caller degrades
    to "unavailable" instead of raising (a serial sidecar must still
    boot and serve; the A/B experiment simply cannot toggle).
    """
    try:
        from sglang.srt.multimodal.processors import qwen_vl as qv
    except Exception:  # pragma: no cover - engine env broken elsewhere
        logger.exception("qwen_vl import failed; patch-4 switch unavailable")
        return None
    if not hasattr(qv, "_PARALLEL_VIDEO_PREPROCESS"):
        return None  # pristine sglang — patch 4 not applied
    return qv


def _sync_parallel_video_preprocess_flag() -> None:
    """One-shot at startup: read the patch-4 switch state into
    _parallel_video_preprocess ("on"/"off"/None=patch missing)."""
    global _parallel_video_preprocess
    qv = _qwen_vl_preprocess_module()
    _parallel_video_preprocess = (
        None if qv is None else ("on" if qv._PARALLEL_VIDEO_PREPROCESS else "off")
    )


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

# B1 — asyncio bridge over the ENGINE's own loop.
#
# History (three attempts, one lesson):
#   1. sync generators + per-step _engine_iter_lock: serialized each
#      request's FIRST step across submit + full prefill -> N
#      concurrent prefills queued in a serial staircase.
#   2. `async for` on UVICORN's loop assuming engine.loop == uvicorn
#      loop: WRONG. sglang 0.5.20's in-process Engine creates its OWN
#      uvloop (engine.loop) with tokenizer_manager's zmq channels
#      bound to it. Driving the engine from another running loop
#      raises "Cannot run the event loop while another loop is
#      running" (measured: /flush_cache 500, engine.py:1313
#      run_until_complete inside the uvicorn loop).
#   3. CURRENT: run engine.loop FOREVER on a dedicated daemon thread;
#      submit every engine interaction as a coroutine via
#      run_coroutine_threadsafe(..., engine.loop). This mirrors what
#      the standard sglang launch_server does structurally: one loop,
#      N request coroutines, one scheduler subprocess — the exact
#      concurrency semantics a concurrent benchmark needs.
class EngineLoopBridge:
    """Own the engine's private asyncio loop for cross-thread use.

    Started once from main() AFTER the Engine is constructed:
      bridge = EngineLoopBridge(engine); bridge.start()

    - start(): launches engine.loop.run_forever() on a daemon thread
      and parks the loop reference for run_coroutine_threadsafe.
    - submit(coro): schedule `coro` on the engine loop; returns a
      CONCURRENT.futures.Future (NOT asyncio) usable from any thread
      via .result(timeout) / add_done_callback / cancel.
    - stop(): best-effort loop shutdown (call only at process exit;
      the scheduler subprocess teardown is owned by sglang).

    Everything that touches engine internals (generate consumption,
    flush_cache coroutine) MUST go through submit() — calling the
    engine's sync wrappers from a thread that owns another running
    loop (uvicorn workers) is what crashed attempt 2.
    """

    def __init__(self, engine):
        self._engine = engine
        self._loop = getattr(engine, "loop", None)
        self._thread = None
        if self._loop is None:
            raise RuntimeError(
                "sglang Engine exposes no .loop — layout changed? "
                "The asyncio bridge needs the engine's private event "
                "loop to run it on a dedicated thread.")

    def start(self) -> None:
        if self._thread is not None:
            return
        import threading

        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="sglang-engine-loop",
            daemon=True,
        )
        self._thread.start()

    def submit(self, coro):
        """Schedule `coro` on the engine loop; concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self) -> None:
        if self._loop is not None and self._thread is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)


# Module-level handle set by main() once the engine exists. Request
# handlers reach it through this global (same pattern as `engine` /
# `store` above). None until startup completes; every handler that
# needs it checks first and 503s.
bridge: "EngineLoopBridge | None" = None


def _set_store_capacity(max_bundles: int) -> None:
    """Rebuild the module-level BundleStore with a new capacity.

    Called from main() after argparse (--max-bundles). Safe before any
    request arrives; assert-guards against shrinking below a size that
    would evict live entries (never true at startup).
    """
    global store
    if max_bundles < 1:
        raise ValueError(f"--max-bundles must be >= 1, got {max_bundles}")
    store = BundleStore(max_bundles=max_bundles)


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

    Concurrency note: unlike the original sidecar this runs OUTSIDE any
    engine lock — torch.load + dequant is per-request CPU work and the
    whole point of the concurrent variant is letting N of these overlap
    while the engine scheduler batches the GPU side.
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
        f"size={entry['size']} max_tokens={max_tokens} "
        f"active_generations={gate.active}")
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

    async def event_stream():
        stats: dict = {"mode": mode}
        # Register as an active generation BEFORE any engine work. If a
        # flush holds the gate, abort with an SSE error instead of racing
        # the flush coroutine inside the scheduler.
        if not gate.enter():
            err = json.dumps({"error": {
                "message": "flush_cache in progress; retry the request"}})
            yield f"data: {err}\n\n"
            return
        try:
            # t_prepare is per-request work (edge: torch.load + fp8
            # dequant; server: bytes hand-off). The HEAVY part of the
            # edge path (torch.load of an uploaded bundle) would block
            # the uvicorn loop for hundreds of ms per request and, under
            # N concurrent requests, serialize them — exactly what the
            # bridge is supposed to remove. Run it in the loop's thread
            # pool via asyncio.to_thread (release the GIL during IO).
            t_prepare_start = time.time()
            if mode == "server":
                text, extra, input_ids_arg = await asyncio.to_thread(
                    _prepare_video_mode, entry, prompt)
            else:
                text, extra, input_ids_arg = await asyncio.to_thread(
                    _prepare_edge_mode, entry, prompt)
            t_prepare = time.time() - t_prepare_start
            stats["t_prepare"] = t_prepare

            sampling_params = {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
            }
            # Guard against zero-token completions: greedy (temperature
            # 0) decoding is not batch-invariant — under concurrent
            # prefills a user's first token can argmax to EOS, ending
            # the request with completion_tokens=0, no content chunk,
            # and a client-side ttft=None (shown as FAIL in the runner
            # despite ok=True). min_new_tokens>=1 suppresses EOS for
            # the first token(s) only; single-letter MCQ answers ("A")
            # still terminate at token 2. 0 disables the guard.
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
            #
            # B1 asyncio bridge, final form: the consumption below runs
            # as ONE coroutine ON THE ENGINE'S OWN LOOP (submitted via
            # bridge.submit / run_coroutine_threadsafe) and relays each
            # engine output dict through a queue.Queue; this uvicorn-
            # side async generator pulls with await asyncio.to_thread(
            # q.get). Two loops, one queue, no locks:
            #   - N requests = N coroutines on the engine loop,
            #     submitted microseconds apart -> all reach the
            #     scheduler (tokenizer_manager zmq -> scheduler
            #     subprocess) before the first prefill finishes.
            #   - The uvicorn loop never blocks on engine work; the
            #     per-chunk thread hop is ~tens of microseconds.
            #   - The generate() call itself must be AWAITED here:
            #     sglang 0.5.20's Engine.async_generate(stream=True) is
            #     a plain coroutine that RETURNS the async generator
            #     (engine.py: `return tokenizer_manager
            #     .generate_request(obj, None)`) — the await unpacks it
            #     ON THE ENGINE LOOP, which is also where zmq
            #     subscriptions bind. Awaiting it on the uvicorn loop
            #     (or skipping the await, as measured on the box:
            #     "'async for' requires an object with __aiter__
            #     method, got coroutine") misbinds or misdrives the
            #     stream. Everything engine-side lives inside
            #     _consume_on_engine_loop below.
            out_q: queue.Queue = queue.Queue()

            async def _consume_on_engine_loop():
                # Runs ON the engine loop. Drives the stream to
                # completion and relays every output dict; the queue
                # carries ("out", dict) / ("end", None) / ("err", exc).
                try:
                    generator = await engine.async_generate(
                        **gen_kwargs, stream=True,
                    )
                    async for out in generator:
                        out_q.put(("out", out))
                    out_q.put(("end", None))
                except BaseException as exc:  # relay, don't crash the loop
                    out_q.put(("err", exc))

            consumer = bridge.submit(_consume_on_engine_loop())

            def _cancel_consumer():
                if not consumer.done():
                    consumer.cancel()

            first = True
            prev_text = ""
            n_tok = 0
            usage = {"prompt_tokens": None, "completion_tokens": 0}
            finish_reason: str | None = None
            try:
                while True:
                    # Block on the engine relay queue from a worker
                    # thread so this (uvicorn) coroutine stays reactive
                    # to client disconnects between chunks.
                    kind, payload = await asyncio.to_thread(
                        out_q.get, True, 3600.0)
                    if kind == "end":
                        break
                    if kind == "err":
                        raise RuntimeError(
                            f"engine stream failed: {payload}")
                    out = payload
                    text_piece = out.get("text") or ""
                    if first:
                        stats["server_ttft"] = time.time() - t_gen_start
                        first = False
                    if text_piece.startswith(prev_text):
                        delta = text_piece[len(prev_text):]
                    else:
                        # Defensive: non-prefix text (e.g. if the engine
                        # is ever switched to
                        # incremental_streaming_output) is already a
                        # delta; forward as-is.
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
            finally:
                _cancel_consumer()
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
                # chunk, so the client records ttft=None (runner shows
                # FAIL despite ok=True). finish_reason tells WHY:
                # "stop" = greedy first-token EOS (batch-invariance
                # noise; mitigate via min_new_tokens), "abort" =
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
        finally:
            gate.exit()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/flush_cache")
async def flush_cache():
    """Flush radix cache + mm preprocess cache between benchmark waves.

    Drain-and-gate semantics (replaces the original sidecar's global
    engine_lock): wait for in-flight generations to drain, close the
    gate so new generations abort with a clear SSE error instead of
    interleaving with the flush, submit the flush COROUTINE to the
    engine loop via the bridge, then reopen.

    Why tokenizer_manager.flush_cache and not engine.flush_cache: the
    sync wrapper drives the engine loop via run_until_complete
    (engine.py:1313) — illegal from inside the running uvicorn loop
    ("Cannot run the event loop while another loop is running", the
    500 measured on the box). The underlying coroutine
    (tokenizer_manager.flush_cache) is submitted with
    run_coroutine_threadsafe and awaited here as a wrapped future —
    the gate guarantees no generation coroutine is concurrently
    mid-flight on the engine loop when the flush runs.

    The drain (gate.flush_close) runs on a worker thread via
    asyncio.to_thread: gate.exit() fires on THIS (uvicorn) loop, so
    blocking here would deadlock the very generations we are waiting
    for.

    The concurrent scenario calls this once per wave BEFORE releasing
    the client barrier, so in practice the gate is open and the call
    reduces to a plain flush. The gating only matters for accidental
    flushes during a wave — flushing while requests sit in the
    scheduler is undefined behaviour in sglang 0.5.20.
    """
    if engine is None or bridge is None:
        raise HTTPException(status_code=503, detail="engine not initialized")

    t0 = time.time()
    await asyncio.to_thread(gate.flush_close)
    try:
        await asyncio.wrap_future(
            bridge.submit(engine.tokenizer_manager.flush_cache()))
    finally:
        gate.flush_open()
    logger.info("flush_cache: radix + mm caches flushed "
                f"(gate drained in {time.time() - t0:.3f} s)")
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Concurrency + video-sampling observability for the harness."""
    return {
        "active_generations": gate.active,
        "bundles_in_store": len(store._store),
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
        # patch-4 switch state: "on"/"off" (None = ACTIVE sglang lacks
        # the parallel-video-preprocess patch — serial upstream only).
        "parallel_video_preprocess": _parallel_video_preprocess,
        # which decode backend the ACTIVE sglang selected at import
        # ("torchcodec" / "decord" / "unknown") — decode_threads above
        # caps BOTH, so naming the backend says where decode CPU lands.
        "video_decode_backend": _video_decode_backend(),
    }


@app.post("/video_preprocess_mode")
async def video_preprocess_mode(request: Request):
    """Flip the patch-4 parallel video preprocess switch AT RUNTIME.

    Why an endpoint and not a restart: the engine build takes minutes,
    and the point of the knob is SAME-PROCESS A/B comparison of the
    upstream-serial vs fan-out decode paths under identical model /
    cache / affinity state. The switch lives in the qwen_vl module
    global _PARALLEL_VIDEO_PREPROCESS and is read at EVERY preprocess
    call (tokenizer-manager, i.e. THIS process — no scheduler child
    involvement), so a flip lands on the next request.

    Body: {"mode": "on"|"off"} (also accepts {"parallel": true/false}
    for convenience). Returns the resulting state; 409 when the active
    sglang lacks patch 4 (refuses rather than pretending), 400 on a
    malformed body.
    """
    body = await request.json()
    mode = body.get("mode")
    if mode is None and "parallel" in body:
        mode = "on" if body.get("parallel") else "off"
    if mode not in ("on", "off"):
        raise HTTPException(
            status_code=400,
            detail=f'body must be {"mode": "on"|"off"} '
                   f'(or {"parallel": bool}); got {body!r}')

    qv = _qwen_vl_preprocess_module()
    if qv is None:
        raise HTTPException(
            status_code=409,
            detail="ACTIVE sglang lacks the 6G-Testbed patch 4 "
                   "(parallel video preprocess) — run "
                   "python aitestbed/server/patch_host_sglang.py in the "
                   "sidecar's env and RESTART (the module global is "
                   "absent, so there is nothing to flip)")

    qv._PARALLEL_VIDEO_PREPROCESS = (mode == "on")
    global _parallel_video_preprocess
    _parallel_video_preprocess = mode
    logger.info(
        f"video_preprocess_mode: parallel video preprocess -> {mode} "
        f"(takes effect on the next request; width when on = "
        f"mm_io_worker_num)")
    return {"status": "ok", "parallel_video_preprocess": mode}


def _apply_server_cpu_limit(spec: str):
    """Pin THIS process (and every child it spawns: the in-process
    Engine, its scheduler subprocess, uvicorn workers) to a shared
    CPU set — the server-arm CPU budget.

    Video decode + preprocessing is CPU-heavy and would otherwise grab
    every core of a many-core host, muting the server-vs-edge contrast
    the concurrent scenario measures. Called FIRST in main(), before
    the engine builds, so the scheduler subprocess (forked during
    Engine construction) inherits the affinity and the whole server
    lifetime honors the budget. JIT/warmup also runs inside it (a
    one-off cost, accepted by design).

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
    global engine, _model_path, bridge, _applied_server_cpus
    parser = argparse.ArgumentParser(
        description="6G SGLang Engine Server (concurrent variant: no "
                    "engine_lock; continuous batching decides queueing). "
                    "Launch via start_sidecar_concurrent.sh (loads the "
                    "sglang host patches + CUDA bootstrap this bare "
                    "launcher skips).")
    parser.add_argument("--model", default=str(
        REPO_ROOT / "aicodec" / "aicodec" / "models" / "qwen3_vl_4B"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012,
                        help="default 8012 (original serial sidecar uses "
                             "8002; run both side by side on different ports)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85,
                        help="KV-cache static memory fraction (original "
                             "sidecar hardcodes 0.85; concurrent batching "
                             "may need headroom tuning)")
    parser.add_argument("--max-running-requests", type=int, default=None,
                        help="scheduler running-request cap (max requests "
                             "batched on the GPU at once). Omit/None = "
                             "sglang default (computed from KV-cache "
                             "capacity); set e.g. 4 to force queueing "
                             "above 4 in-flight requests")
    parser.add_argument("--chunked-prefill-size", type=int, default=None,
                        metavar="N",
                        help="max tokens in ONE prefill chunk (official "
                             "sglang 0.5.20 ServerArgs field, no source "
                             "patching). Omit/None = sglang default "
                             "(resolves to 8192 on this box). -1 disables "
                             "chunked prefill (whole prompt in one "
                             "forward). Larger values shorten TTFT of the "
                             "first request but stall in-flight decodes "
                             "longer during each prefill chunk (unless "
                             "--enable-mixed-chunk); with this workload's "
                             "~12k-token prompts, raising it above the "
                             "prompt size can let queued requests share "
                             "one prefill batch (#new-seq > 1)")
    parser.add_argument("--enable-mixed-chunk", action="store_true",
                        default=None,
                        help="mix decode requests into prefill chunks so "
                             "in-flight decodes do not stall while a long "
                             "chunked prefill runs (official sglang 0.5.20 "
                             "ServerArgs field). Default off (sglang "
                             "default); note backends are not bit-identical "
                             "across scheduler geometry — keep comparisons "
                             "within one geometry")
    parser.add_argument("--prefill-max-requests", type=int, default=None,
                        metavar="N",
                        help="max requests admitted into ONE prefill batch "
                             "(official sglang 0.5.20 ServerArgs field). "
                             "Omit/None = unlimited (sglang default); the "
                             "token budget (chunked_prefill_size / "
                             "max_prefill_tokens) usually binds first with "
                             "video prompts this large")
    parser.add_argument("--max-prefill-tokens", type=int, default=None,
                        metavar="N",
                        help="TOTAL token budget of ONE prefill batch "
                             "(official sglang 0.5.20 ServerArgs field, "
                             "default 16384, runtime-capped at >= model "
                             "context length). Together with "
                             "--chunked-prefill-size it decides whether "
                             "MULTIPLE queued requests share one prefill "
                             "forward (#new-seq > 1): with ~12k-token "
                             "prompts, two requests co-batch only when "
                             "BOTH budgets are >= ~2x the prompt size. "
                             "Raising it does not reduce total GPU work "
                             "(prefill is compute-bound) — it changes the "
                             "queueing texture; compare runs within one "
                             "fixed geometry only")
    parser.add_argument("--enable-radix-cache", action="store_true",
                        help="re-enable sglang radix cache (OFF by default "
                             "in this benchmark: N concurrent users send "
                             "the same video+prompt, so with the cache ON "
                             "the first user's prefill warms the whole "
                             "~14.5k-token prefix and later users hit "
                             "~100%% cached tokens — TTFT collapses to "
                             "decode-only and the cold-prefill concurrency "
                             "under test disappears)")
    parser.add_argument("--mm-io-worker-num", type=int, default=None,
                        metavar="N",
                        help="width of the mm I/O thread pool running video "
                             "decode + sample + resize (official sglang "
                             "0.5.20 ServerArgs field, Mm namespace, Engine "
                             "kwargs passthrough). N = at most N of the "
                             "wave's videos decode in parallel across "
                             "requests; the rest queue in preprocessing "
                             "before reaching the scheduler — the server-"
                             "arm decode budget under concurrency. "
                             "Omit/None = model default (qwen3_vl: 16). "
                             "SGLANG_IO_WORKERS env still applies when this "
                             "is omitted/0. Pair with --server-cpus for a "
                             "hard affinity cap")
    parser.add_argument("--mm-processor-worker-num", type=int, default=None,
                        metavar="N",
                        help="width of the multimodal processor pool (HF "
                             "processor calls: normalize / tensorize; "
                             "official sglang 0.5.20 ServerArgs field). "
                             "1 = synchronous processing (no pool). "
                             "Omit/None = model default (qwen3_vl: 2)")
    parser.add_argument("--min-new-tokens", type=int, default=1, metavar="N",
                        help="suppress EOS for the first N generated tokens "
                             "of every request (default 1). Guards against "
                             "zero-token completions: greedy decoding is "
                             "not batch-invariant, so under concurrent "
                             "prefills a user's first token can argmax to "
                             "EOS — the request then finishes ok with NO "
                             "content chunk and the client records "
                             "ttft=None (shown as FAIL despite ok=True). "
                             "0 disables the guard. Requests can override "
                             "per-call via body min_new_tokens.")
    parser.add_argument("--max-bundles", type=int, default=64,
                        help="BundleStore capacity: uploaded bundles held "
                             "before LRU eviction. MUST be >= the wave size "
                             "N (a wave uploads N bundles before the chat "
                             "requests pop them; smaller N would evict live "
                             "bundles -> 404). Default 64.")
    parser.add_argument("--server-cpus", default="auto-half", metavar="SPEC",
                        help="server-arm CPU budget: pin this server and its "
                             "scheduler subprocess to a shared cpu set so "
                             "CPU-bound video decode cannot grab the whole "
                             "host (pairs with the runner's --edge-cpus-per-"
                             "worker; without BOTH caps the concurrent "
                             "server-vs-edge contrast is muted on a many-"
                             "core host). SPEC: comma list with ranges "
                             "('0-15,32') or 'auto-half' = first half of the "
                             "available cores (default). '' disables. "
                             "Applied BEFORE the engine builds, so JIT/"
                             "warmup pays the budget too (one-off cost)")
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
                             "only true per-decode thread cap. Especially "
                             "relevant under CONCURRENCY: N users decoding "
                             "in parallel means N such pools. Set it to "
                             "match the edge terminals' decode threads for "
                             "symmetric server-vs-edge CPU comparisons. "
                              "0 = decord native default (no cap). Exported "
                              "as SGLANG_DECORD_NUM_THREADS for the engine "
                              "process and inherited by scheduler children.")
    parser.add_argument("--parallel-video-preprocess",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="STARTUP default for the 6G-Testbed patch-4 "
                             "switch: fan concurrent videos' preprocessing "
                             "(decode + resize + pin) out over the mm-io "
                             "pool instead of upstream's serial "
                             "event-loop staircase (the measured cause of "
                             "equal TTFT across heterogeneous videos). "
                             "DEFAULT ON since the v2 fix (single-video "
                             "requests hop too — the executor hop, not "
                             "gather(), creates the cross-request "
                             "overlap). Use --no-parallel-video-preprocess "
                             "for the upstream serial baseline (A/B "
                             "OFF side; bit-identical to unpatched "
                             "sglang). Flippable AT RUNTIME via "
                             "POST /video_preprocess_mode (no engine "
                             "restart); width when ON = "
                             "--mm-io-worker-num; per-video codec threads "
                             "still capped by --decode-threads. Requires "
                              "the sglang patch (applied automatically at "
                              "startup; /stats shows "
                              "parallel_video_preprocess=null when the "
                              "active sglang lacks it).")
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
                             "its own pools by its per-worker pin — the "
                             "resize-stage thread asymmetry behind 'server "
                             "decodes faster'. Set it to the edge terminals' "
                             "--edge-cpus-per-worker for symmetric server-vs-"
                             "edge CPU comparisons. Scheduler children are "
                             "unaffected (model_runner.py pins them to 1 "
                             "thread regardless). 0 = native default (no "
                             "cap; backwards-compatible).")
    args = parser.parse_args()

    _applied_server_cpus = _apply_server_cpu_limit(args.server_cpus)

    _set_store_capacity(args.max_bundles)

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
    # Patch-4 startup default: set the env BEFORE the engine imports
    # qwen_vl (the module global initializes from it). A pre-existing
    # env value wins (explicit override, mirroring --decode-threads);
    # the RUNTIME flip (POST /video_preprocess_mode) does not use the
    # env at all — it writes the module global directly.
    if args.parallel_video_preprocess:
        os.environ.setdefault("SGLANG_QWEN_PARALLEL_VIDEO_PREPROCESS", "1")
    try:
        mm_cfg = build_mm_process_config(
            args.frame_ratio, args.max_frames)
    except ValueError as exc:
        parser.error(str(exc))

    # Auto-load the sglang disk patch (same patcher the start_sidecar*.sh
    # bootstrap runs) BEFORE any sglang import / the canary below — so a
    # direct ``python sglang_engine_server_concurrent.py`` start is as
    # protected as the shell-wrapped one (fail = abort, like
    # ``patcher || die``).
    ensure_sglang_patches(reason="sglang_engine_server_concurrent startup")

    # Fail fast (BEFORE the multi-minute engine build) when a frame
    # ratio was requested but the active sglang lacks the patch —
    # otherwise the benchmark would silently sample at 2fps.
    verify_frame_ratio_patch(active=bool(args.frame_ratio))

    _model_path = args.model
    logger.info(f"Starting SGLang Engine (concurrent, model={args.model} "
                f"mm_io_worker_num={args.mm_io_worker_num} "
                f"mm_processor_worker_num={args.mm_processor_worker_num} "
                f"decode_threads={os.environ.get('SGLANG_DECORD_NUM_THREADS') or 'native'} "
                f"preprocess_threads={os.environ.get('OMP_NUM_THREADS') or 'native'}) ...")
    engine = _build_engine(
        args.model, args.device,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        chunked_prefill_size=args.chunked_prefill_size,
        enable_mixed_chunk=args.enable_mixed_chunk,
        prefill_max_requests=args.prefill_max_requests,
        max_prefill_tokens=args.max_prefill_tokens,
        enable_radix_cache=args.enable_radix_cache,
        mm_io_worker_num=args.mm_io_worker_num,
        mm_processor_worker_num=args.mm_processor_worker_num,
        mm_process_config=mm_cfg,
    )

    # B1 bridge: assert the engine surface this server needs BEFORE
    # accepting traffic, then park the engine's private loop on its
    # dedicated thread. Fail fast with a precise message instead of
    # 500ing on the first request (the V1 spike turned into a startup
    # check).
    if not hasattr(engine, "async_generate"):
        raise RuntimeError(
            "sglang Engine has no async_generate — 0.5.20 expected; "
            "the asyncio bridge drives streams via "
            "engine.async_generate(stream=True). If this sglang "
            "renamed it, adapt _consume_on_engine_loop in "
            "sglang_engine_server_concurrent.py.")
    if not hasattr(engine, "tokenizer_manager"):
        raise RuntimeError(
            "sglang Engine has no tokenizer_manager — flush_cache "
            "submits engine.tokenizer_manager.flush_cache() to the "
            "engine loop. Layout changed?")
    bridge = EngineLoopBridge(engine)
    bridge.start()

    # Patch-4 switch: sync the echo state now that the engine build has
    # imported the (patched) qwen_vl — the module global initialized
    # from the env we set above. None => active sglang lacks patch 4
    # (the flip endpoint will refuse with 409; the flag default OFF
    # keeps behavior bit-identical either way).
    _sync_parallel_video_preprocess_flag()
    if args.parallel_video_preprocess and _parallel_video_preprocess is None:
        logger.warning(
            "--parallel-video-preprocess given but the ACTIVE sglang "
            "lacks the 6G-Testbed patch 4 — running with upstream "
            "SERIAL video preprocessing (check patch_host_sglang.py "
            "output above; POST /video_preprocess_mode will 409)")

    logger.info("SGLang Engine ready (concurrent variant, no engine_lock; "
                f"max_bundles={args.max_bundles}; max_running_requests="
                f"{args.max_running_requests if args.max_running_requests is not None else 'sglang-default'}; "
                f"chunked_prefill_size="
                f"{args.chunked_prefill_size if args.chunked_prefill_size is not None else 'sglang-default'}; "
                f"enable_mixed_chunk="
                f"{args.enable_mixed_chunk if args.enable_mixed_chunk is not None else 'sglang-default'}; "
                f"prefill_max_requests="
                f"{args.prefill_max_requests if args.prefill_max_requests is not None else 'sglang-default'}; "
                f"max_prefill_tokens="
                f"{args.max_prefill_tokens if args.max_prefill_tokens is not None else 'sglang-default'}; "
                f"mm_io_worker_num="
                f"{args.mm_io_worker_num if args.mm_io_worker_num is not None else 'sglang-default'}; "
                f"mm_processor_worker_num="
                f"{args.mm_processor_worker_num if args.mm_processor_worker_num is not None else 'sglang-default'}; "
                f"radix_cache={'ON' if args.enable_radix_cache else 'OFF'}; "
                f"min_new_tokens={_default_min_new_tokens}; "
                f"parallel_video_preprocess="
                f"{_parallel_video_preprocess or 'patch-missing'}; "
                f"engine loop parked on dedicated thread; server_cpus="
                f"{_applied_server_cpus or 'unpinned'})")
    logger.info(format_log_line(
        args.frame_ratio or None, args.max_frames or None,
        json.dumps(mm_cfg) if mm_cfg else None))
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
