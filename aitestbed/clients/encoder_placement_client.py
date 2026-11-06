"""
Encoder Placement client adapter for the 6G AI Traffic Testbed.

Terminal side of the encoder-placement experiment (visual encoder on
the server vs on the terminal), talking to the SGLang engine sidecar
(server/sglang_engine_server.py).

mode "server":  uploads the original mp4 binary (PUT /v1/bundles
                 ?kind=video) and issues a streaming chat completion
                 referencing it via bundle://video/{id}. Decode,
                 processor, ViT and prefill all run on the server.

mode "edge":    runs video decode + sampling + processor + the visual
                 tower locally, serializes the SGLang-native
                 precomputed_embedding dict (processor output minus the
                 raw pixel tensors, plus the visual tower's final
                 embeddings), uploads it
                 (PUT /v1/bundles?kind=edge) and references it via
                 bundle://edge/{id}. The server injects it into
                 prefill through Engine.generate(video_data=[...]).

The TTFT window spans the full video-input -> first-token path in
both modes (upload/edge work happens inside the window, strictly
synchronous).
"""

import io
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

from .base import LLMClient, StreamingResponse, estimate_payload_bytes

# Shared frame-sampling wiring with the sidecars (build_mm_process_config
# semantics mirrored in _prepare_edge_bundle's video_config assembly;
# verify_frame_ratio_patch canary-checks the ACTIVE sglang's
# smart_nframes for the frame_ratio branch before edge sampling runs;
# ensure_sglang_patches — called once in __init__, OUTSIDE all timing
# windows — auto-applies the sglang disk patch: the client has NO shell
# wrapper, so the Python call is its only auto-load path).
sys.path.insert(0, str(Path(os.path.dirname(__file__)).parent / "server"))
from encpl_video_sampling import (
    ensure_sglang_patches,
    verify_frame_ratio_patch,
)

# Same guard as the sidecar (server/sglang_engine_server.py): deep_ep's
# import-time JIT init crashes without CUDA_HOME and the AssertionError
# escapes sglang's `except ImportError`, killing `import sglang` on
# machines without the toolkit. Dense Qwen3-VL never needs deep_ep.
sys.modules.setdefault("deep_ep", None)


def _resolve_model_path(model: str) -> str:
    """Resolve relative model paths against the repo root (like split client)."""
    if model and os.path.isabs(model):
        return model
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    if model:
        candidate = os.path.join(repo_root, model)
        if os.path.exists(candidate):
            return candidate
        return model
    return repo_root


def _ensure_vidcom2_path() -> str:
    """Add the in-repo VidCom2 checkout to sys.path (no install needed).

    The token_compressor package lives in
    aicodec/aicodec/models/VidCom2 and is imported by the edge
    compression path in _prepare_edge_bundle.
    """
    vidcom2 = os.environ.get("VIDCOM2_PATH") or str(
        Path(os.path.dirname(__file__)).parent.parent
        / "aicodec" / "aicodec" / "models" / "VidCom2"
    )
    if vidcom2 not in sys.path:
        sys.path.insert(0, vidcom2)
    return vidcom2


def _quantize_feature(embeddings, bundle_quant: str):
    """Quantize the edge bundle's `feature` tensor to fp8 (e4m3) for upload.

    Per-tensor symmetric scale: scale = amax / 448.0 (448 = max
    representable e4m3 value), applied as feature_fp8 =
    to(float8_e4m3fn)(embeddings / scale). The scale rides along in the
    bundle as `feature_scale` (a plain float); the sidecar's
    _maybe_dequant_bundle_feature pops it and reconstructs
    bf16 = scale * to(bfloat16)(feature_fp8) right after torch.load, so
    everything downstream of deserialization (engine hash, shm
    transfer, embed_multimodal_inputs dtype checks) sees bf16 tensors
    equivalent to an unquantized run.

    Only reached with bundle_quant='fp8' AND compressor='vidcom2' (the
    caller enforces the coupling), i.e. after VidCom2 row pruning — the
    fp8 tensor is what actually gets uploaded.

    Returns (quantized_embeddings, quant_stats); quant_stats keys:
    quant_dtype, feature_scale, quant_time.
    """
    if bundle_quant != "fp8":
        raise ValueError(
            f"invalid bundle_quant {bundle_quant!r}: must be 'fp8'")
    import time as _time

    import torch

    t_start = _time.time()
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError(
            "bundle_quant=fp8 requires torch.float8_e4m3fn (torch >= 2.1 "
            "with fp8 support); this torch build has no fp8 dtypes")
    work = embeddings.detach()
    amax = work.abs().max()
    amax = float(amax.item() if hasattr(amax, "item") else amax)
    scale = amax / 448.0
    if scale == 0.0 or not math.isfinite(scale):
        # degenerate all-zero / non-finite tensor: keep scale=1 so the
        # division is well-defined and dequant is a plain multiply.
        scale = 1.0
    try:
        quantized = (work / scale).to(fp8_dtype)
    except (NotImplementedError, RuntimeError, TypeError):
        # fp8 casts are CPU-unsupported on some builds; hop to CUDA for
        # the cast (the edge box always holds CUDA) and come back.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "cannot cast the edge feature to fp8 (CPU cast "
                "unsupported and no CUDA device available)") from None
        quantized = ((work / scale).to("cuda").to(fp8_dtype)).cpu()
    quant_stats = {
        "quant_dtype": "float8_e4m3",
        "feature_scale": scale,
        "quant_time": _time.time() - t_start,
    }
    return quantized, quant_stats


_VALID_ATTN_IMPLS = ("sdpa", "flash_attention_4", "eager")


def _resolve_attn_impl(attn_impl: str) -> str:
    """Normalize/validate the edge ViT attention backend selector.

    Accepts 'sdpa' (default), 'flash_attention_4' (the flash-attn-4
    package / kernels-community backend — the ONLY flash backend this
    stack supports; see the note below) and 'eager'. Returns the
    normalized value; raises ValueError on anything else.

    NOTE: flash_attention_2 is intentionally NOT offered. The epd env
    ships flash-attn-4 (beta), whose `flash_attn` shim module passes a
    bare `import flash_attn` but fails transformers'
    is_flash_attn_2_available() gate, and transformers v5 maps
    attn_implementation='flash_attention_2' to a kernels-community
    flash-attn2 kernel that is not registered in this build — every fa2
    request would raise. fa4 is the supported flash path here.
    """
    impl = (attn_impl or "sdpa").strip().lower()
    if impl in ("", "sdpa"):
        return "sdpa"
    if impl in ("fa4", "flash", "flash_attention_4", "flash-attn-4"):
        return "flash_attention_4"
    if impl in ("fa2", "flash_attention_2", "flash-attn-2"):
        raise ValueError(
            f"attn_impl {attn_impl!r} (flash_attention_2) is not "
            f"supported on this stack: the flash-attn-4 beta in this env "
            f"fails transformers' fa2 gate and the kernels-community "
            f"flash-attn2 kernel is unregistered. Use 'flash_attention_4' "
            f"for flash kernels or 'sdpa' (default)")
    if impl == "eager":
        return "eager"
    raise ValueError(
        f"invalid attn_impl {attn_impl!r}: must be one of "
        f"{_VALID_ATTN_IMPLS}")


def _qwen3_vl_model_type(model_path: str) -> str:
    """Read the checkpoint's top-level model_type from its config.json.

    Shared by _qwen3_vl_model_class / _qwen3_vl_vision_class. Raises
    RuntimeError when config.json is unreadable (fail fast instead of
    a wrong-architecture load).
    """
    import json

    cfg_path = os.path.join(model_path, "config.json")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            model_type = json.load(f).get("model_type", "")
    except OSError as exc:
        raise RuntimeError(
            f"cannot read {cfg_path} to pick the model class "
            f"({exc}); pass a local Qwen3-VL model directory"
        ) from exc
    return model_type


def _qwen3_vl_model_class(model_path: str):
    """Pick the transformers class matching the checkpoint's architecture.

    Qwen3-VL ships dense (4B/8B: model_type "qwen3_vl",
    Qwen3VLForConditionalGeneration) and MoE (30B-A3B: model_type
    "qwen3_vl_moe", Qwen3VLMoeForConditionalGeneration) checkpoints.
    Loading a MoE checkpoint with the dense class silently drops every
    mlp.experts.* / mlp.gate weight (UNEXPECTED) and freshly random-
    initializes the dense mlp.* projections (MISSING) — the LOAD REPORT
    the client prints for the 30B model. Dispatch on config.json so the
    right class is used; both expose model.model.visual identically.

    Returns the transformers class. Raises RuntimeError for unknown
    model_types (fail fast instead of a wrong-arch load).
    """
    model_type = _qwen3_vl_model_type(model_path)

    import transformers

    if model_type == "qwen3_vl_moe":
        cls = getattr(transformers, "Qwen3VLMoeForConditionalGeneration", None)
        if cls is None:
            raise RuntimeError(
                "this transformers has no Qwen3VLMoeForConditionalGeneration "
                f"(version {transformers.__version__}); MoE checkpoints like "
                "Qwen3-VL-30B-A3B need a transformers with qwen3_vl_moe "
                "support")
        return cls
    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    raise RuntimeError(
        f"unsupported model_type {model_type!r} at {model_path}: expected "
        "'qwen3_vl' (dense) or 'qwen3_vl_moe' (MoE)")


def _qwen3_vl_vision_class(model_path: str):
    """Pick the standalone vision-tower class matching the checkpoint.

    The dense tower class (Qwen3VLVisionModel) only maps the composite
    checkpoint's model.visual.* keys when the checkpoint is dense
    (vision sub-config model_type "qwen3_vl_vision"). On a MoE
    checkpoint like Qwen3-VL-30B-A3B (vision sub-config normalized to
    "qwen3_vl_moe_vision" by Qwen3VLMoeConfig) it maps nothing: every
    checkpoint key lands UNEXPECTED and the whole tower is randomly
    initialized (MISSING) — the LOAD REPORT the vision_only path
    printed for the 30B model — so the edge embeddings would be
    garbage. Dispatch to Qwen3VLMoeVisionModel for MoE checkpoints;
    both towers' forward returns the same
    BaseModelOutputWithDeepstackFeatures (pooler_output +
    deepstack_features), so downstream code is unchanged.

    Returns the transformers class. Raises RuntimeError for unknown
    model_types or a transformers without the class.
    """
    model_type = _qwen3_vl_model_type(model_path)

    import transformers

    if model_type == "qwen3_vl_moe":
        cls = getattr(transformers, "Qwen3VLMoeVisionModel", None)
        if cls is None:
            raise RuntimeError(
                "this transformers has no Qwen3VLMoeVisionModel "
                f"(version {transformers.__version__}); MoE checkpoints like "
                "Qwen3-VL-30B-A3B need a transformers with qwen3_vl_moe "
                "support")
        return cls
    if model_type == "qwen3_vl":
        from transformers import Qwen3VLVisionModel

        return Qwen3VLVisionModel
    raise RuntimeError(
        f"unsupported model_type {model_type!r} at {model_path}: expected "
        "'qwen3_vl' (dense) or 'qwen3_vl_moe' (MoE)")


def _vision_load_report(visual) -> tuple[list, list]:
    """Missing/unexpected weight keys after a from_pretrained load.

    DEPRECATED for new code — stdout parsing proved unreliable
    (transformers 5.x emits one-line Python-list warnings like
    "Some weights of ... were not initialized" that bullet-style
    parsing misses entirely, yielding false-clean reports; this is
    exactly how a randomly-initialized tower slipped through).
    Kept only as a best-effort fallback for modules loaded without
    output_loading_info.

    Prefer from_pretrained(..., output_loading_info=True), which
    returns the real (missing_keys, unexpected_keys) programmatically.
    """
    info = getattr(visual, "load_info", None)
    if isinstance(info, dict):
        return (list(info.get("missing_keys") or []),
                list(info.get("unexpected_keys") or []))
    missing = getattr(visual, "missing_keys", None)
    unexpected = getattr(visual, "unexpected_keys", None)
    return (list(missing or []), list(unexpected or []))


# Prefixes a composite Qwen3-VL checkpoint may ship its vision tower
# under. Priority matters: model.visual.* is the layout the composite
# model classes actually read; a bare visual.* block (when present) is
# usually a stale/duplicated export and must NEVER override the
# model.visual.* tensors for the same tower key.
_VISUAL_PREFIXES = ("model.visual.", "visual.")


def _remap_visual_state_dict(checkpoint_sd: dict, tower_sd_keys) -> dict:
    """Build a tower-keyed state dict from the composite checkpoint.

    2a — key remap: checkpoints store the tower under model.visual.*
    (and, for some safetensors layouts, visual.*). A standalone tower
    expects the suffix after that prefix. Returns {} when no key
    matches (caller decides that's fatal).

    Prefix priority: model.visual.* wins; visual.* fills only keys not
    already mapped (stale duplicate-block guard).
    """
    tower_keys = set(tower_sd_keys)
    remapped: dict = {}
    seen: dict = {}
    for key, value in checkpoint_sd.items():
        for prefix in _VISUAL_PREFIXES:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                if suffix in tower_keys:
                    if suffix not in seen or seen[suffix] == "model.visual.":
                        remapped[suffix] = value
                        seen[suffix] = prefix
                break
    return remapped


def _load_checkpoint_state_dict(model_path: str) -> dict:
    """Load the full state dict from a (possibly sharded) checkpoint.

    Handles both single-file model.safetensors and sharded layouts
    (model.safetensors.index.json + N shards) — the 30B MoE checkpoint
    is sharded, so reading only model.safetensors would miss most of
    the tower. Returns the merged {key: tensor} dict (CPU tensors).
    """
    import torch
    from safetensors.torch import load_file

    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"no model.safetensors or model.safetensors.index.json under "
            f"{model_path}; cannot remap visual weights")
    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    shards = sorted(set(index.get("weight_map", {}).values()))
    merged: dict = {}
    for shard in shards:
        merged.update(load_file(os.path.join(model_path, shard)))
    return merged


def _load_edge_model(model_path: str, vision_only: bool = False,
                     attn_impl: str = "sdpa"):
    """Load the visual tower for edge-side encoding (returns the module).

    Mirrors test_vlm_input_format.py: transformers' Qwen3-VL vision tower
    with deepstack outputs, producing the [N, hidden*(1+deepstack)] final
    embeddings SGLang expects as precomputed_embeddings. The processor
    is shared via EncoderPlacementClient._processor_cache.

    vision_only=False (default): load the FULL conditional-generation
    model — dense or MoE, dispatched via _qwen3_vl_model_class so the
    30B-A3B MoE checkpoint loads with the right architecture — and
    return model.model.visual, the historical behaviour (the LLM half
    is loaded and discarded).

    vision_only=True: load ONLY the vision tower (ViT + patch merger +
    deepstack projections) via the checkpoint-matched standalone class
    (Qwen3VLVisionModel for dense, Qwen3VLMoeVisionModel for MoE — see
    _qwen3_vl_vision_class), same weights (model.visual.* keys map
    automatically once the class matches), ~0.4B params vs the full
    model, cutting edge VRAM and load time by the LLM weight read.
    Falls back to the full load (with a warning) when the standalone
    class is unavailable in this transformers.

    The standalone load is GUARDED: the transformers load report is
    parsed and any missing/unexpected vision keys trigger a state-dict
    remap from the checkpoint's model.visual.* prefix (2a) — covering
    transformers versions whose standalone tower expects a different
    key layout than the composite checkpoint ships — and, if keys
    still remain unmatched, a hard RuntimeError (2b: fail fast; a
    benchmark must never silently measure a randomly-initialized
    tower, which surfaces downstream as empty/echoing answers in edge
    mode while server mode stays clean).

    attn_impl: attention backend for the tower ('sdpa' default,
    'flash_attention_4', 'eager'). flash_attention_4 requires the
    flash-attn-4 package (present in this env as 4.0.0b19); a load
    failure with it raises (no silent sdpa fallback — a benchmark must
    not silently measure the wrong backend). flash_attention_2 is NOT
    supported on this stack (see _resolve_attn_impl).
    NOTE: different backends are NOT bit-identical (kernel accumulation
    order); comparisons across attn_impl need a tolerance, and the
    vision-only equivalence check must pin both loads to the SAME impl.
    """
    import torch

    impl = _resolve_attn_impl(attn_impl)
    model_path = _resolve_model_path(model_path)
    if vision_only:
        try:
            vision_cls = _qwen3_vl_vision_class(model_path)
        except (RuntimeError, ImportError) as exc:
            print(f"[EncPl] warning: vision_only requested but no "
                  f"checkpoint-matched standalone vision class "
                  f"({exc}); falling back to the full "
                  f"conditional-generation load")
        else:
            try:
                load_out = vision_cls.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    device_map="cuda",
                    attn_implementation=impl,
                    output_loading_info=True,
                )
            except TypeError:
                # Very old transformers without output_loading_info:
                # fall back to the best-effort attribute report.
                visual = vision_cls.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                    device_map="cuda",
                    attn_implementation=impl,
                )
                load_out = None
            except Exception as exc:
                print(f"[EncPl] warning: {vision_cls.__name__} load failed "
                      f"({exc}); falling back to the full "
                      f"conditional-generation load")
                load_out = "failed"
            if load_out != "failed":
                if isinstance(load_out, tuple):
                    # transformers >= 4.x with output_loading_info=True:
                    # (model, load_info dict).
                    visual, info = load_out[0], load_out[1]
                    missing = list((info or {}).get("missing_keys") or [])
                    unexpected = list(
                        (info or {}).get("unexpected_keys") or [])
                elif load_out is None:
                    # TypeError branch already loaded `visual`; old
                    # transformers has no programmatic report.
                    missing, unexpected = _vision_load_report(visual)
                else:
                    # transformers 5.x may return the bare model with a
                    # .load_info attribute attached.
                    visual = load_out
                    info = getattr(load_out, "load_info", {}) or {}
                    missing = list(info.get("missing_keys") or [])
                    unexpected = list(info.get("unexpected_keys") or [])
                if missing or unexpected:
                    # 2a: remap model.visual.* / visual.* checkpoint keys
                    # into the tower's own layout and load_state_dict
                    # (strict) — covers transformers layouts where the
                    # standalone tower's expected keys don't match the
                    # composite checkpoint's shipped prefix (e.g. a
                    # mismatched standalone class silently leaving every
                    # tower weight MISSING → random init → garbage edge
                    # embeddings).
                    print(f"[EncPl] vision_only standalone load mismatch "
                          f"(missing={len(missing)}, "
                          f"unexpected={len(unexpected)}); attempting "
                          f"state-dict remap")
                    checkpoint_sd = _load_checkpoint_state_dict(model_path)
                    remapped = _remap_visual_state_dict(
                        checkpoint_sd, visual.state_dict().keys())
                    if remapped:
                        visual.load_state_dict(remapped, strict=True)
                        print(f"[EncPl] remap loaded {len(remapped)} "
                              "visual tensors")
                        missing, unexpected = [], []
                    if missing or unexpected:
                        # 2b: fail fast — silently benching a
                        # randomly-initialized tower produces garbage
                        # edge embeddings (empty/echoing answers).
                        raise RuntimeError(
                            f"vision_only standalone load has "
                            f"{len(missing)} missing / "
                            f"{len(unexpected)} unexpected keys and the "
                            f"checkpoint remap matched none of them "
                            f"(first missing: "
                            f"{missing[0] if missing else '-'}; first "
                            f"unexpected: "
                            f"{unexpected[0] if unexpected else '-'}); "
                            f"refusing to run edge encoding with a "
                            f"partially-random vision tower. Use "
                            f"vision_only=False or fix the key layout.")
                visual.eval()
                return visual

    model_cls = _qwen3_vl_model_class(model_path)
    if model_cls.__name__ == "Qwen3VLMoeForConditionalGeneration":
        print(f"[EncPl] loading {model_path} as "
              f"Qwen3VLMoeForConditionalGeneration (MoE checkpoint)")
    model = model_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=impl,
    )
    visual = model.model.visual
    visual.eval()
    return visual


class EncoderPlacementClient(LLMClient):
    """
    Terminal client for the SGLang engine sidecar.
    """

    DEFAULT_BASE_URL = "http://localhost:8002"
    _edge_model_cache: dict = {}
    _processor_cache: dict = {}

    # Edge-bundle whitelist: the only processor-output keys the server
    # actually consumes (sidecar _prepare_edge_mode reads input_ids;
    # base_processor.py:1898 pops "feature"; qwen_vl.py reads
    # video_grid_thw / mrope_positions / mrope_position_delta /
    # second_per_grid_ts). pixel_values* is pure dead weight once the
    # ViT has run locally — it rides along in the full processor dict
    # and roughly doubles the bundle. keep_full_bundle=True restores the
    # original fat bundle (reproducing pre-whitelist numbers).
    _EDGE_BUNDLE_KEYS = (
        "input_ids",
        "video_grid_thw",
        "second_per_grid_ts",
        "mrope_positions",
        "mrope_position_delta",
    )

    # Stream safety timeouts (seconds; 0 disables the respective
    # guard, restoring the historical unlimited wait). Mirrors the
    # Qwen3.6 client's knobs: local loopback TCP connect 20, first SSE
    # event 200 (serial arms; concurrent callers pass a larger value),
    # idle gap between SSE events 20, bundle-upload response read
    # 1200 (cell_edge at 5 Mbit/s moves ~300 MB in ~7 min; the READ
    # side only covers the server's store+ack).
    DEFAULT_STREAM_CONNECT_TIMEOUT = 20.0
    DEFAULT_STREAM_FIRST_EVENT_TIMEOUT = 200.0
    DEFAULT_STREAM_IDLE_TIMEOUT = 20.0
    DEFAULT_STREAM_UPLOAD_READ_TIMEOUT = 1200.0

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 stream_connect_timeout: Optional[float] = None,
                 stream_first_event_timeout: Optional[float] = None,
                 stream_idle_timeout: Optional[float] = None,
                 stream_upload_read_timeout: Optional[float] = None):
        self.api_key = api_key or os.environ.get("SGLANG_ENGINE_API_KEY") or "sglang"
        self.base_url = (
            base_url
            or os.environ.get("SGLANG_ENGINE_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        # Precedence: explicit kwarg > env override > class default.
        # Env vars exist for ad-hoc manual runs; the matrix driver
        # forwards explicit values via kwargs (sanitized_env strips
        # these vars from cell subprocesses, so a stale outer value
        # cannot silently flip a cell's arm).
        for attr, env, dflt, kwarg in (
            ("stream_connect_timeout", "ENCPL_STREAM_CONNECT_TIMEOUT",
             self.DEFAULT_STREAM_CONNECT_TIMEOUT, stream_connect_timeout),
            ("stream_first_event_timeout", "ENCPL_STREAM_FIRST_EVENT_TIMEOUT",
             self.DEFAULT_STREAM_FIRST_EVENT_TIMEOUT, stream_first_event_timeout),
            ("stream_idle_timeout", "ENCPL_STREAM_IDLE_TIMEOUT",
             self.DEFAULT_STREAM_IDLE_TIMEOUT, stream_idle_timeout),
            ("stream_upload_read_timeout", "ENCPL_STREAM_UPLOAD_READ_TIMEOUT",
             self.DEFAULT_STREAM_UPLOAD_READ_TIMEOUT, stream_upload_read_timeout),
        ):
            raw = os.environ.get(env)
            val = kwarg if kwarg is not None else (
                dflt if (raw is None or raw == "") else float(raw))
            setattr(self, attr, val)
        # Auto-load the sglang disk patches at CONSTRUCTION — outside
        # every timing window (bench entry loop, wave 'ready' phase,
        # scenario setup all construct the client BEFORE t_request_start;
        # a per-REQUEST call inside generate_content_with_video() would
        # bill the patcher's ~20-100ms to the FIRST measured request of
        # every fresh wave-worker process, and edge-only at that — a
        # systematic TTFT bias vs the server arm). One call point also
        # covers every entry: workers (spawn processes), the bench
        # driver, scenarios. Unconditional: patch 1's edge-mode
        # PRECOMPUTED_EMBEDDING guard is needed even without
        # frame_ratio. Idempotent per process; quiet no-op when the
        # sidecar bootstrap already patched; tolerated as a silent
        # no-op in sglang-less mock environments.
        ensure_sglang_patches(reason="EncoderPlacementClient.__init__")

    @property
    def provider(self) -> str:
        return "sglang-engine"

    # ------------------------------------------------------------------
    # Shared prompt construction (both modes)
    # ------------------------------------------------------------------
    def _get_processor(self, model: str):
        """CPU-only AutoProcessor (tokenizer + image processor configs).

        No max_pixels/min_pixels overrides: the server path uses the
        model-repo defaults (its video_config is empty), so the edge must
        too, otherwise smart_resize produces different frame sizes.
        """
        key = _resolve_model_path(model)
        cached = EncoderPlacementClient._processor_cache.get(key)
        if cached is None:
            from transformers import AutoProcessor

            cached = AutoProcessor.from_pretrained(key)
            EncoderPlacementClient._processor_cache[key] = cached
        return cached

    @staticmethod
    def _build_templated_text(processor, video_path: str, prompt: str,
                              user_tag: str = "") -> str:
        """Render the Qwen3-VL chat template around the video placeholder.

        apply_chat_template is pure jinja here (no video decode): the
        video content part expands to
        <|vision_start|><|video_pad|><|vision_end|>. Both modes use this
        same call so the engine sees byte-identical prompt markup
        whether the video is referenced as raw bytes (server mode) or
        precomputed embeddings (edge mode).

        user_tag (concurrent waves): prepended to the TEXT part. When
        set, the text part is emitted FIRST (before the video part),
        so the tag lands in the token stream ahead of the
        <|video_pad|> expansion; a distinct tag per user then makes
        every user's whole prefix distinct (radix cache cannot share
        anything before the video tokens either) while the video
        tokens themselves stay identical. When empty, the historical
        video-first content order is kept byte-for-byte (serial
        scenario / bench reproducibility).
        """
        text_part = {"type": "text", "text": (user_tag + prompt)}
        # bare video reference: the template only expands the type into
        # the vision placeholder; extra keys like max_pixels would be
        # silently ignored but are removed to avoid any doubt about
        # template-vs-processor config divergence.
        video_part = {"type": "video", "video": os.path.abspath(video_path)}
        message = [{
            "role": "user",
            "content": (
                # tagged waves: text first so the per-user tag forks the
                # prefix BEFORE the video tokens (see docstring above).
                [text_part, video_part] if user_tag
                # historical order (serial scenario, user_tag="").
                else [video_part, text_part]),
        }]
        return processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    # ------------------------------------------------------------------
    # Edge-side computation (edge mode)
    # ------------------------------------------------------------------
    def _get_edge_model(self, model: str, vision_only: bool = False,
                        attn_impl: str = "sdpa"):
        impl = _resolve_attn_impl(attn_impl)
        key = (_resolve_model_path(model), bool(vision_only), impl)
        cached = EncoderPlacementClient._edge_model_cache.get(key)
        if cached is None:
            cached = _load_edge_model(key[0], vision_only=vision_only,
                                      attn_impl=impl)
            EncoderPlacementClient._edge_model_cache[key] = cached
        return cached

    def _prepare_edge_bundle(self, visual, processor, video_path: str,
                              templated_text: str, max_frames: int,
                              keep_full_bundle: bool = False,
                              compressor: str = "",
                              retention_ratio: float = 0.25,
                              bundle_quant: str = "",
                              frame_ratio: float | None = None):
        """
        Run decode + sampling + processor + visual tower locally and
        build the SGLang precomputed_embedding dict.

        Uses SGLang's own decode/sample/resize pipeline (VideoDecoderWrapper
        + preprocess_video) and passes video_metadata + do_sample_frames=False
        to the HF processor — byte-for-byte the same frames and processor
        kwargs the server-side path (qwen_vl.py process_mm_data_async)
        produces, so both modes see identical inputs.

        frame_ratio (0 < r <= 1): ratio-of-native frame sampling for the
        edge arm — kept in video_config, consumed by smart_nframes'
        frame_ratio branch (v3 semantics: sglang's own base =
        total_frames/video_fps*fps, i.e. 2fps x duration before
        clamping, multiplied by the ratio, then the native clamp chain;
        requires the sglang frame_ratio patch; canary-verified per
        request when active). None = off (sglang default 2fps x
        duration). max_frames caps BOTH sampling modes (native sglang
        key, no patch needed); the two knobs are assembled
        INDEPENDENTLY (fixing the historical coupling bug where
        max_frames=0 dropped out of the config).

        compressor="vidcom2" additionally runs the VidCom2 token selection
        (token_compressor.vidcom2, in-repo checkout under
        aicodec/aicodec/models/VidCom2) on the ViT output BEFORE bundle
        assembly, keeping
        retention_ratio of the video tokens. All row-aligned tensors in the
        bundle (feature, input_ids, mrope_positions) are pruned with the
        same kept indices so the server's precomputed fast path sees a
        self-consistent, shorter sequence. video_grid_thw / mrope delta /
        second_per_grid_ts ride along unchanged (see _compress_edge_bundle).

        bundle_quant="fp8" (only valid together with compressor="vidcom2")
        additionally quantizes the PRUNED feature tensor to fp8 e4m3 with
        a per-tensor scale (see _quantize_feature), halving the uploaded
        feature bytes; the server dequantizes right after torch.load.

        Returns (bundle_bytes, edge_stats).
        """
        # Validate the quantization coupling BEFORE any heavy import /
        # decode work: fp8 is only defined on the pruned bundle.
        if bundle_quant and bundle_quant != "fp8":
            raise ValueError(
                f"invalid bundle_quant {bundle_quant!r}: must be '' or 'fp8'")
        if bundle_quant and compressor != "vidcom2":
            raise ValueError(
                "bundle_quant='fp8' requires compressor='vidcom2' "
                "(quantization applies to the pruned bundle; run "
                "vidcom2 first)")

        import asyncio

        import torch
        from sglang.srt.multimodal.processors.qwen_vl import preprocess_video
        from sglang.srt.utils.video_decoder import VideoDecoderWrapper

        t_e2e_start = time.time()

        # Decode + smart_nframes sampling + smart_resize, exactly as the
        # server does (qwen_vl.py preprocess_video). video_config is
        # assembled INDEPENDENTLY per knob (frame_ratio / max_frames),
        # mirroring the sidecar's build_mm_process_config so both arms
        # sample identically: max_frames is a NATIVE sglang whitelist key
        # (fps branch caps by it even without the frame_ratio patch);
        # frame_ratio selects ratio-of-native sampling (v3 patch
        # semantics: sglang's own base = total/video_fps*fps, i.e. 2fps
        # x duration before clamping, multiplied by the ratio, then the
        # native clamp chain; logger.info traces every transition).
        # {} -> model defaults (VIDEO_MIN/TOTAL/MAX_PIXELS + 2fps),
        # matching a sidecar started with no sampling flags.
        # Timeline: open (container probe) is separated from decode_sample
        # (decode + sampling + resize, fused inside preprocess_video).
        video_config: dict = {}
        if frame_ratio:
            if not (0.0 < frame_ratio <= 1.0):
                raise ValueError(
                    f"frame_ratio must be in (0, 1], got {frame_ratio}")
            video_config["frame_ratio"] = float(frame_ratio)
        if max_frames and int(max_frames) != 768:
            video_config["max_frames"] = int(max_frames)
        t_open_start = time.time()
        decoder = VideoDecoderWrapper(video_path)
        t_open = time.time() - t_open_start
        # Container-level frame info (same source sglang reads: len(vr),
        # vr.avg_fps) — captured BEFORE sampling so total_frames/video_fps
        # are available even if the sampled tensor's layout ever changes.
        total_frames = len(decoder)
        video_fps = getattr(decoder, "avg_fps", None)
        t_decode_start = time.time()
        video, video_metadata = asyncio.run(preprocess_video(
            decoder, video_config=video_config))
        t_decode_sample = time.time() - t_decode_start
        # Authoritative final frame count: the sampled tensor's leading
        # dim AFTER the full clamp chain + np.unique dedup (may be lower
        # than smart_nframes' return value).
        sampled_nframes = int(video.shape[0]) if hasattr(
            video, "shape") else len(video)

        # Processor with the same kwargs the server passes
        # (qwen_vl.py:1052-1055): sampled frames + video_metadata +
        # do_sample_frames=False -> pixel_values + video_grid_thw +
        # second_per_grid_ts in one call.
        t_processor_start = time.time()
        inputs = processor(
            text=[templated_text],
            videos=[video],
            video_metadata=[video_metadata],
            do_sample_frames=False,
            processor_video_config={},
            return_tensors="pt",
        )
        t_processor = time.time() - t_processor_start
        t_decode = t_open + t_decode_sample

        # transformers v5 emits the unified `pixel_values` for videos; this
        # env's older transformers emits the Qwen2-VL-style
        # `pixel_values_videos` (sglang maps both: base_processor.py:411).
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            pixel_values = inputs.get("pixel_values_videos")
        if pixel_values is None:
            raise RuntimeError(
                "processor output has neither pixel_values nor "
                "pixel_values_videos; keys="
                f"{list(inputs.keys())} — video likely took the wrong "
                "processor branch")

        # ------------------------------------------------------------------
        # Visual tower: run the local ViT on the processor output.
        # Mirrors sglang's own call (models/qwen3_vl.py):
        # self.visual(pixel_values, grid_thw=grid_thw), whose sglang
        # implementation returns torch.cat([merger(x)] +
        # deepstack_feature_lists, dim=1) -> [merged_tokens,
        # out_hidden_size * (1 + n_deepstack)].
        #
        # transformers' tower does NOT return that final tensor
        # directly; depending on the version its forward returns:
        #   v5:      BaseModelOutputWithDeepstackFeatures with
        #            last_hidden_state = pre-merger block output
        #            [N_patches, hidden_size] (WRONG tensor),
        #            pooler_output = merged [N/4, out_hidden_size],
        #            deepstack_features = n x [N/4, out_hidden_size]
        #   4.57.x:  tuple (merged, [deepstack features...])
        # So we assemble the sglang-contract tensor ourselves.
        # ------------------------------------------------------------------
        # Timeline: h2d (CPU->CUDA copies) / vit_fwd (tower forward) /
        # assemble (deepstack unpack + torch.cat). The trailing
        # cuda.synchronize stays exactly where the old t_vit capture was,
        # so stage2_vit keeps its historical composition bit-for-bit.
        t_vit_start = time.time()
        pixel_values = pixel_values.to("cuda", dtype=torch.bfloat16)
        video_grid_thw = inputs["video_grid_thw"].to("cuda")
        t_h2d = time.time() - t_vit_start
        t_fwd_start = time.time()
        with torch.no_grad():
            vit_out = visual(pixel_values, grid_thw=video_grid_thw)
        t_vit_fwd = time.time() - t_fwd_start
        t_asm_start = time.time()
        if hasattr(vit_out, "pooler_output") and vit_out.pooler_output is not None:
            main_embeds = vit_out.pooler_output
            deepstack = list(vit_out.deepstack_features or [])
        elif isinstance(vit_out, tuple):
            main_embeds = vit_out[0]
            deepstack = list(vit_out[1]) if len(vit_out) > 1 else []
        else:
            main_embeds = vit_out
            deepstack = []
        if deepstack:
            embeddings = torch.cat([main_embeds] + deepstack, dim=1)
        else:
            embeddings = main_embeds
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_vit = time.time() - t_vit_start
        t_assemble = t_vit - t_h2d - t_vit_fwd

        # Fast-fail contract checks (a transformers API change must fail
        # here with a clear message, not as a scheduler-side crash):
        vcfg = getattr(visual, "config", None)
        if vcfg is None:
            # HF towers usually carry the vision config as self.config
            vcfg = getattr(getattr(processor, "image_processor", None),
                           "config", None)
        if vcfg is not None:
            merge = getattr(vcfg, "spatial_merge_size", 2)
            expected_width = vcfg.out_hidden_size * (
                1 + len(vcfg.deepstack_visual_indexes))
            expected_rows = sum(
                t_dim * h_dim * w_dim
                for t_dim, h_dim, w_dim in video_grid_thw.tolist()
            ) // (merge * merge)
            if embeddings.shape[-1] != expected_width:
                raise RuntimeError(
                    f"edge ViT output width {embeddings.shape[-1]} != "
                    f"expected {expected_width} (out_hidden_size * "
                    f"(1+deepstack)); transformers output layout changed?")
            if embeddings.shape[0] != expected_rows:
                raise RuntimeError(
                    f"edge ViT output rows {embeddings.shape[0]} != "
                    f"placeholder token count {expected_rows} from "
                    f"video_grid_thw; grid mismatch?")

        # ------------------------------------------------------------------
        # Optional VidCom2 token compression: select which video-token rows
        # survive, then prune every row-aligned tensor consistently.
        # ------------------------------------------------------------------
        compress_stats: dict = {}
        if compressor:
            if compressor != "vidcom2":
                raise ValueError(
                    f"invalid compressor {compressor!r}: must be '' or 'vidcom2'")
            if not (0.0 < retention_ratio <= 1.0):
                raise ValueError(
                    f"invalid retention_ratio {retention_ratio}: must be "
                    f"in (0, 1]")
            embeddings, inputs, compress_stats = self._compress_edge_bundle(
                embeddings, inputs, video_grid_thw, vcfg,
                retention_ratio, processor,
            )

        # ------------------------------------------------------------------
        # Optional fp8 quantization of the UPLOADED feature tensor.
        # Deliberately downstream of (and gated on) VidCom2: the selector
        # scores the bf16 embeddings, then the pruned tensor is quantized
        # so the fp8 bytes are exactly what goes on the wire. (The
        # bundle_quant/compressor coupling was validated up front.)
        # ------------------------------------------------------------------
        quant_stats: dict = {}
        if bundle_quant:
            embeddings, quant_stats = _quantize_feature(embeddings, bundle_quant)

        # ------------------------------------------------------------------
        # Build the SGLang-native precomputed_embedding dict.
        # Default: whitelist only the keys the server consumes (see
        # _EDGE_BUNDLE_KEYS) — pixel_values* is dead weight after the
        # local ViT run and roughly halves the upload. keep_full_bundle
        # restores the original whole-processor-output bundle.
        # ------------------------------------------------------------------
        dropped_keys: list[str] = []
        dropped_bytes = 0
        if keep_full_bundle:
            bundle = {k: v for k, v in inputs.items()}
        else:
            bundle = {k: v for k, v in inputs.items()
                      if k in self._EDGE_BUNDLE_KEYS}
            dropped_keys = sorted(
                k for k in inputs.keys() if k not in self._EDGE_BUNDLE_KEYS)
            if dropped_keys:
                t_drop = time.time()
                dropped_buf = io.BytesIO()
                torch.save({k: inputs[k] for k in dropped_keys}, dropped_buf)
                dropped_bytes = dropped_buf.tell()
                del dropped_buf
                self._last_drop_est_time = time.time() - t_drop
        bundle["format"] = "precomputed_embedding"
        # D2H copy of the (possibly pruned+quantized) feature tensor —
        # previously unmeasured; it sits between quantization and
        # serialization on the critical path.
        t_d2h_start = time.time()
        feature_cpu = embeddings.cpu()
        t_d2h = time.time() - t_d2h_start
        if quant_stats:
            bundle["feature"] = feature_cpu
            bundle["feature_scale"] = float(quant_stats["feature_scale"])
        else:
            bundle["feature"] = feature_cpu

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ser = time.time()
        buf = io.BytesIO()
        torch.save(bundle, buf)
        ser_time = time.time() - t_ser

        edge_stats = {
            # ---- aggregate stage keys (unchanged composition, history-
            # comparable): stage1_edge = open + decode_sample + processor,
            # stage2_vit = h2d + vit_fwd + assemble. ----
            "stage1_edge": t_processor + t_decode,
            "stage1_breakdown": {
                "open": t_open,
                "decode_sample": t_decode_sample,
                "processor": t_processor,
                # legacy alias (old schema had decode/sample/processor
                # with sample always 0.0); kept one release for readers
                # of historical JSONs — new writers emit the 3 keys above
                "decode": t_decode, "sample": 0.0,
            },
            "stage2_vit": t_vit,
            "stage2_breakdown": {
                "h2d": t_h2d,
                "vit_fwd": t_vit_fwd,
                "assemble": t_assemble,
            },
            # full edge critical path: video input -> serialized bundle
            # ready for upload (everything the terminal does before the
            # PUT). vidcom2_time / quant_time / drop_est_time ride along
            # in compress_stats / quant_stats / the dropped-keys block.
            "edge_prep_time": time.time() - t_e2e_start,
            "d2h_time": t_d2h,
            "ser_time": ser_time,
            "bundle_bytes": buf.tell(),
            "embedding_shape": list(embeddings.shape),
            # frame sampling: container facts + final sampled count (post
            # clamp chain + unique dedup) — the client-side counterpart of
            # the patched sglang preprocess_video logger.info trace.
            "total_frames": total_frames,
            "video_fps": float(video_fps) if video_fps else None,
            "sampled_nframes": sampled_nframes,
        }
        if dropped_keys:
            edge_stats["dropped_keys"] = dropped_keys
            edge_stats["bundle_bytes_dropped"] = dropped_bytes
            edge_stats["drop_est_time"] = getattr(
                self, "_last_drop_est_time", 0.0)
        edge_stats.update(compress_stats)
        edge_stats.update(quant_stats)
        return buf.getvalue(), edge_stats

    # ------------------------------------------------------------------
    # Optional VidCom2 token compression (edge mode)
    # ------------------------------------------------------------------
    @staticmethod
    def _compress_edge_bundle(embeddings, inputs, video_grid_thw, vcfg,
                              retention_ratio: float, processor=None):
        """VidCom2-select surviving video tokens and prune row-aligned tensors.

        Scoring mirrors token_compressor/vidcom2/models/qwen3_vl.py
        _compute_keep_indices (select_low_var_channels ->
        compute_gaussian_scores -> compute_scales ->
        select_outlier_indices) but runs against the final edge ViT output
        (main + deepstack concat, one row per video token) instead of
        patching the LLM forward.

        Layout note (v5 processor): video tokens are NOT one contiguous
        <|video_pad|> run — the processor emits a PER-FRAME layout
        [ts?, <|vision_start|>, video_pad x frame_tokens,
        <|vision_end|>] x T inside the vision block. So pruning is done
        with a whole-sequence keep-COLUMN mask: text/timestamp/boundary
        columns are kept, and inside each frame's pad run the columns
        whose token rows survive are kept.

        mrope: this env's transformers (4.57.x) processor does NOT emit
        mrope_positions, so the server falls back to get_rope_index
        (qwen_vl.py:1128) — which iterates FULL frames (mrope_rope_index
        st advance assumes frame_tokens pads per block) and crashes on
        token-pruned ids. Fix: compute the FULL-sequence mrope HERE with
        sglang's own get_rope_index (bit-identical: same env, same
        function the server would call), then prune it with the same
        column mask and ship it in the bundle as mrope_positions
        [3, 1, n] — the server's _get_precomputed_mrope_from_output
        fast path then skips get_rope_index entirely (qwen_vl.py:1114).
        Surviving tokens keep their ORIGINAL position values, matching
        upstream vidcom2's visual_pos_masks semantics.

        video_grid_thw / second_per_grid_ts are left UNTOUCHED: the
        server's precomputed fast path derives item offsets by scanning
        the (pruned) input_ids for video-pad runs (base_processor.py
        get_mm_items_offset), and get_rope_index (the only grid
        consumer for this path) is skipped when mrope_positions is
        provided — the grid then merely describes the (already consumed)
        source layout. Every layout contract is validated up front and
        violated layouts raise (fail-fast client-side, never a
        scheduler-side crash).

        Returns (embeddings, inputs, compress_stats).
        """
        import time as _time

        import torch

        _ensure_vidcom2_path()
        from token_compressor.vidcom2 import (
            compute_gaussian_scores,
            compute_scales,
            select_low_var_channels,
            select_outlier_indices,
            _map_linear_offset,
        )

        t_start = _time.time()
        n_full = int(embeddings.shape[0])
        if n_full == 0:
            return embeddings, inputs, {"video_tokens_full": 0,
                                        "video_tokens_kept": 0}

        # --- score & select (mirrors _compute_keep_indices) ---
        merge = getattr(vcfg, "spatial_merge_size", 2) if vcfg else 2
        t_dim, h_dim, w_dim = video_grid_thw.tolist()[0]
        frame_tokens = (h_dim * w_dim) // (merge * merge)
        sel_feat = select_low_var_channels(embeddings)
        vid_score, frame_score = compute_gaussian_scores(sel_feat, frame_tokens)
        scales = compute_scales(-vid_score.mean(dim=-1), retention_ratio)
        indices = select_outlier_indices(
            vid_score + frame_score, scales, frame_tokens)
        keep_idx = _map_linear_offset(indices, frame_tokens)
        keep_idx, _ = torch.sort(keep_idx)
        n_kept = int(keep_idx.shape[0])
        if n_kept == 0 or n_kept >= n_full:
            # nothing to prune (ratio ~1 or degenerate selection)
            return embeddings, inputs, {
                "video_tokens_full": n_full, "video_tokens_kept": n_full,
                "vidcom2_active": False, "vidcom2_time": _time.time() - t_start,
            }

        # --- per-frame pad runs in the expanded ids (fail-fast) ---
        video_token_id = EncoderPlacementClient._video_pad_token_id(
            inputs, processor)
        runs = EncoderPlacementClient._video_pad_runs(inputs, video_token_id)
        if (len(runs) != t_dim
                or any(hi - lo != frame_tokens for lo, hi in runs)
                or sum(hi - lo for lo, hi in runs) != n_full):
            raise RuntimeError(
                f"vidcom2: input_ids video-pad layout does not match "
                f"video_grid_thw (runs={len(runs)}, t={t_dim}, "
                f"frame_tokens={frame_tokens}, "
                f"pad_total={sum(hi - lo for lo, hi in runs)}, "
                f"embedding_rows={n_full}); refusing to prune a bundle "
                f"the server would reject")

        # --- whole-sequence keep mask: True = keep the column ---
        input_ids = inputs.get("input_ids")
        ids_1d = input_ids.reshape(-1)
        seq_len = int(ids_1d.shape[0])
        keep_cols = torch.ones(seq_len, dtype=torch.bool,
                               device=ids_1d.device)
        keep_rows = torch.zeros(n_full, dtype=torch.bool,
                                device=keep_idx.device)
        keep_rows[keep_idx] = True
        # map global token rows -> per-run local columns
        row_cursor = 0
        for lo, hi in runs:
            local = keep_rows[row_cursor:row_cursor + (hi - lo)]
            keep_cols[lo:hi] = local.to(keep_cols.device)
            row_cursor += hi - lo

        # --- prune input_ids (preserve 1-D vs 2-D shape) ---
        col_idx = torch.nonzero(keep_cols).reshape(-1)
        if input_ids.ndim == 1:
            new_ids = ids_1d[col_idx]
        else:
            new_ids = input_ids[..., col_idx]

        # --- mrope: compute FULL, then prune with the same column mask ---
        # Prefer a processor-provided mrope (v5); otherwise (4.57.x) compute
        # it with sglang's own get_rope_index — the exact function the
        # server would call as fallback — so values are bit-identical.
        mrope = inputs.get("mrope_positions")
        if mrope is None:
            mrope = EncoderPlacementClient._compute_full_mrope(
                inputs, video_grid_thw, processor)
        mp = mrope.squeeze(1) if mrope.ndim == 3 else mrope  # [3, seq]
        if int(mp.shape[-1]) != seq_len:
            raise RuntimeError(
                f"vidcom2: mrope width {mp.shape[-1]} != input_ids length "
                f"{seq_len}; unknown mrope layout, refusing to prune")
        new_mrope = mp[..., col_idx.to(mp.device)]  # [3, n]

        # --- all contracts held: commit the pruned tensors ---
        # Ship mrope as [3, 1, n] (qwen_vl.py _get_precomputed_mrope_from_output
        # accepts ndim==3 with shape[1]==1) + recomputed [1,1] delta
        # (qwen formula: max + 1 - seq_len).
        inputs = dict(inputs, input_ids=new_ids,
                      mrope_positions=new_mrope.unsqueeze(1))
        new_delta = torch.tensor(
            [[int(new_mrope.max()) + 1 - int(new_mrope.shape[-1])]],
            dtype=mp.dtype, device=new_mrope.device)
        inputs = dict(inputs, mrope_position_delta=new_delta)

        embeddings = embeddings[keep_idx.to(embeddings.device)]

        compress_stats = {
            "video_tokens_full": n_full,
            "video_tokens_kept": n_kept,
            "retention_ratio": float(retention_ratio),
            "retention_ratio_actual": n_kept / n_full,
            "vidcom2_active": True,
            "vidcom2_time": _time.time() - t_start,
            "frames": int(t_dim),
        }
        return embeddings, inputs, compress_stats

    @staticmethod
    def _video_pad_token_id(inputs, processor=None) -> int:
        """Resolve the <|video_pad|> token id for run scanning.

        Preference order: processor.video_token_id (v5 processor attr),
        the bundle's own token id from an existing pad run, then
        processor.tokenizer lookups. Raises if nothing resolves —
        pruning without the pad id would silently corrupt the bundle.
        """
        vid = getattr(processor, "video_token_id", None)
        if vid is not None:
            return int(vid)
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            for name in ("video_token_id", "video_token"):
                vid = getattr(tok, name, None)
                if vid is None:
                    vid = tok.convert_tokens_to_ids("<|video_pad|>")
                    if vid is not None and vid >= 0:
                        return int(vid)
                    vid = None
                else:
                    return int(vid)
        input_ids = (inputs or {}).get("input_ids")
        if input_ids is not None:
            # last resort: single most-frequent interior token heuristic
            # is NOT safe — fail instead.
            pass
        raise RuntimeError(
            "vidcom2: cannot resolve <|video_pad|> token id from the "
            "processor (processor.video_token_id / tokenizer); refusing "
            "to prune")

    @staticmethod
    def _compute_full_mrope(inputs, video_grid_thw, processor):
        """Full-sequence mrope for the UNPRUNED ids via sglang's own
        get_rope_index — the same function the server calls as fallback
        (qwen_vl.py:1128), so the values are bit-identical. Token ids
        come from the processor's tokenizer; model_type / merge size /
        tokens_per_second from the AutoConfig vision section, mirroring
        QwenVLProcessor.__init__ (qwen_vl.py:380-394).
        Returns [3, seq]; raises if anything is unresolvable.
        """
        import torch

        tok = getattr(processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError(
                "vidcom2: processor has no tokenizer; cannot precompute "
                "mrope_positions for the pruned bundle")

        def _tid(name, text):
            v = getattr(tok, name, None)
            if v is None:
                v = tok.convert_tokens_to_ids(text)
            if v is None or int(v) < 0:
                raise RuntimeError(
                    f"vidcom2: cannot resolve {text} token id for mrope "
                    f"precomputation")
            return int(v)

        image_token_id = _tid("image_token_id", "<|image_pad|>")
        video_token_id = _tid("video_token_id", "<|video_pad|>")
        vision_start_id = _tid("vision_start_token_id", "<|vision_start|>")

        model_key = _resolve_model_path(
            getattr(tok, "name_or_path", "") or "")
        try:
            from transformers import AutoConfig
            hf_config = AutoConfig.from_pretrained(model_key)
        except Exception as e:
            raise RuntimeError(
                f"vidcom2: cannot load AutoConfig from {model_key} for "
                f"mrope precomputation: {e}") from e
        model_type = hf_config.model_type
        vision_cfg = getattr(hf_config, "vision_config", None)
        merge = (getattr(vision_cfg, "spatial_merge_size", 2)
                 if vision_cfg is not None else 2)
        tps = (getattr(vision_cfg, "tokens_per_second", None)
               if vision_cfg is not None else None)

        from sglang.srt.layers.rotary_embedding.mrope_rope_index import (
            get_rope_index,
        )

        ids = inputs.get("input_ids")
        ids_2d = ids.reshape(1, -1) if ids.ndim == 1 else ids
        mrope, _delta = get_rope_index(
            spatial_merge_size=merge,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_id,
            model_type=model_type,
            tokens_per_second=tps,
            input_ids=ids_2d,
            image_grid_thw=None,
            video_grid_thw=video_grid_thw.cpu(),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
        )
        if mrope.ndim == 3:      # [3, batch, seq] -> [3, seq]
            mrope = mrope.squeeze(1)
        if int(mrope.shape[-1]) != int(ids_2d.shape[-1]):
            raise RuntimeError(
                f"vidcom2: get_rope_index width {mrope.shape[-1]} != "
                f"input_ids length {ids_2d.shape[-1]}")
        return mrope

    @staticmethod
    def _video_pad_runs(inputs, video_token_id: int):
        """[lo, hi) spans of EVERY <|video_pad|> run in the expanded ids.

        v5 processor layout is per-frame: the vision block contains T
        runs of frame_tokens pads each (with timestamps and
        <|vision_start|>/<|vision_end|> between them). Returns them in
        sequence order; callers validate counts against video_grid_thw.
        """
        import torch

        input_ids = (inputs or {}).get("input_ids")
        if input_ids is None:
            return []
        ids = input_ids.reshape(-1)
        if ids.numel() == 0:
            return []
        is_pad = ids == video_token_id
        # boundaries between pad/non-pad
        change = torch.cat([
            torch.tensor([True], device=ids.device),
            is_pad[1:] != is_pad[:-1],
            torch.tensor([True], device=ids.device),
        ])
        edges = torch.nonzero(change).reshape(-1).tolist()
        runs = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if bool(is_pad[lo]):  # this segment is a pad run
                runs.append((lo, hi))
        return runs

    # ------------------------------------------------------------------
    # Upload + OpenAI-format streaming completion
    # ------------------------------------------------------------------
    def _conn_5tuple(self, response) -> dict | None:
        """Extract the TCP 5-tuple from a requests response.

        Works because `requests` (urllib3) keeps the connection socket
        reachable at response.raw._connection.sock after the call. No
        local 5-tuple is logged when the socket cannot be reached (e.g.
        a mock transport in tests) — callers treat None as absent.

        port must be exported: the pcap of a concurrent wave shows N
        interleaved flows to the SAME (dst_ip, dst_port); the source
        port is the ONLY key that separates user i's upload/SSE flows
        for per-user packet-level analysis (pcap_analyzer flow join).
        """
        try:
            sock = response.raw._connection.sock
            local = sock.getsockname()   # (src_ip, src_port)
            peer = sock.getpeername()    # (dst_ip, dst_port)
            return {
                "src_ip": local[0],
                "src_port": local[1],
                "dst_ip": peer[0],
                "dst_port": peer[1],
                "proto": "tcp",
            }
        except Exception:
            return None

    def _upload_bundle(self, data: bytes, kind: str) -> dict:
        url = f"{self.base_url}/v1/bundles/{uuid.uuid4().hex}?kind={kind}"
        headers = {"Content-Type": "application/octet-stream"}
        t_start = time.time()
        # stream=True ONLY defers reading the RESPONSE body: without it,
        # requests consumes the body inside send() and releases the
        # connection back to the pool, after which the socket is no
        # longer reachable at response.raw._connection.sock and the
        # 5-tuple would read as None (observed live: up:None chat:PORT).
        # The uploaded REQUEST body is unaffected.
        # Timeout: urllib3 keeps the socket timeout at the CONNECT slot
        # value for the WHOLE request-body send phase (conn.request()
        # does sock.settimeout(connect_timeout) before writing the
        # body, and only resets it to the read slot after the last byte
        # is sent — urllib3 connection.py request()). On a netem'd
        # upload path (cell_edge 5 Mbit/s, 47-337 MB videos) the body
        # send alone takes minutes, so the connect slot must NOT be
        # stream_connect_timeout (20 s default) or every large upload
        # dies at ~20 s with "Connection aborted." + TimeoutError
        # (2026-10-20 04:24 matrix, cell_edge 14/14 users on the
        # Qwen3.6 twin). Use the upload read budget for BOTH slots: it
        # is derived from the slowest profile's transfer time in the
        # first place, and a loopback TCP handshake completes well
        # within it. Fallback to the connect slot when the upload
        # budget is disabled; 0/None in both slots disables the
        # timeout entirely.
        send_budget = (self.stream_upload_read_timeout
                       or self.stream_connect_timeout or None)
        up_timeout = None
        if send_budget:
            up_timeout = (send_budget,
                          self.stream_upload_read_timeout or None)
        response = requests.put(url, data=data, headers=headers, stream=True,
                                timeout=up_timeout)
        if response.status_code >= 400:
            raise RuntimeError(
                f"HTTP {response.status_code} from {url}: {response.text[:500]}")
        # Capture the 5-tuple BEFORE .json() consumes the body and the
        # connection is released.
        conn_tuple = self._conn_5tuple(response)
        upload_time = time.time() - t_start
        info = response.json()
        info["upload_time"] = upload_time
        info["t_upload_start"] = t_start
        info["conn_5tuple"] = conn_tuple
        print(f"[EncPl] upload: {upload_time:.3f} s | transferred: "
              f"{len(data) / 1e6:.2f} MB payload, "
              f"{(info.get('wire_bytes') or 0) / 1e6:.2f} MB on wire")
        return info

    # ------------------------------------------------------------------
    # Stream idle-timeout plumbing (see _stream_completion)
    # ------------------------------------------------------------------
    def _raw_stream_socket(self, response):
        """Best-effort extraction of the live socket behind a streamed
        requests response (response.raw._connection.sock across the
        urllib3 versions in use). None when the layout differs."""
        try:
            return response.raw._connection.sock
        except AttributeError:
            return None

    def _apply_idle_timeout(self, response) -> bool:
        """Tighten the SSE socket's recv budget to stream_idle_timeout
        AFTER the first event has landed. Returns True when the socket
        was reachable and patched; False signals the caller to use the
        watchdog fallback instead."""
        sock = self._raw_stream_socket(response)
        if sock is None:
            return False
        try:
            sock.settimeout(self.stream_idle_timeout)
            return True
        except OSError:
            return False

    def _iter_lines_idle_guarded(self, response):
        """iter_lines with an idle-gap guard that arms only AFTER the
        first line has landed. The first line's recv keeps the
        first-event read timeout the POST was issued with (see
        _stream_completion); once bytes arrive, the socket's recv
        budget is tightened to stream_idle_timeout (direct settimeout
        when the raw socket is reachable, else a daemon watchdog that
        hard-closes the response after stream_idle_timeout seconds
        WITHOUT new bytes, making iter_lines raise). Every yielded
        line resets the watchdog deadline. Pure passthrough when the
        idle timeout is disabled."""
        if not self.stream_idle_timeout:
            for line in response.iter_lines(decode_unicode=True):
                yield line
            return
        import threading

        it = response.iter_lines(decode_unicode=True)
        # First line: still under the POST's first-event budget. Only
        # after it lands is the tighter idle budget applied.
        first = next(it, None)
        patched = self._apply_idle_timeout(response) if first is not None \
            else False
        setattr(response, "_encpl_idle_patched", patched)

        deadline = {"t": time.time() + self.stream_idle_timeout}
        closed = {"flag": False}
        lock = threading.Lock()

        def _watchdog():
            while True:
                with lock:
                    wait = deadline["t"] - time.time()
                if wait <= 0:
                    closed["flag"] = True
                    try:
                        response.close()
                    except Exception:
                        pass
                    return
                time.sleep(min(wait, 0.5))

        if first is not None and not patched:
            dog = threading.Thread(target=_watchdog, daemon=True)
            dog.start()
        if first is not None:
            yield first
            with lock:
                deadline["t"] = time.time() + self.stream_idle_timeout
        try:
            for line in it:
                with lock:
                    deadline["t"] = time.time() + self.stream_idle_timeout
                yield line
        finally:
            with lock:
                deadline["t"] = float("inf")
            if closed["flag"]:
                raise requests.exceptions.ConnectionError(
                    f"SSE stream idle for over {self.stream_idle_timeout}s "
                    f"(watchdog closed the response)")

    def _stream_completion(self, messages: list[dict], model: str,
                           mode: str, max_frames: int,
                           max_tokens: int,
                           frame_ratio: float | None = None) -> StreamingResponse:
        params = {
            "model": model,
            "messages": messages,
            "stream": True,
            "mode": mode,
            "max_tokens": max_tokens,
        }
        # Sampling values ride in the body for RECORD-KEEPING (the
        # engine samples with its STARTUP-time mm_process_config; the
        # sidecar warns on mismatch — see parse_body_sampling).
        if frame_ratio:
            params["frame_ratio"] = float(frame_ratio)
        if max_frames:
            params["max_frames"] = int(max_frames)
        request_bytes = estimate_payload_bytes(params)

        streaming_response = StreamingResponse()
        streaming_response.t_request_start = time.time()
        streaming_response.request_bytes = request_bytes
        streaming_response.model = model
        streaming_response.request_payload = {
            "format": "encoder_placement.chat.completions.create",
            "payload": params,
        }

        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        # Timeout tuple: connect = TCP handshake (netem'd loopback SYN
        # retransmits), read = FIRST SSE event budget (covers server-
        # side edge prepare + queueing + prefill + TTFT). Once the
        # first event lands, the per-recv budget switches to the much
        # tighter idle timeout below. 0/None in either slot disables
        # that guard (None means "no timeout" to requests).
        post_timeout = None
        if self.stream_connect_timeout or self.stream_first_event_timeout:
            post_timeout = (self.stream_connect_timeout or None,
                            self.stream_first_event_timeout or None)
        response = requests.post(url, json=params, headers=headers, stream=True,
                                 timeout=post_timeout)
        if response.status_code >= 400:
            raise RuntimeError(
                f"HTTP {response.status_code} from {url}: {response.text[:500]}")
        # 5-tuple of the SSE connection, captured BEFORE the body is
        # consumed (the socket stays reachable while streaming).
        setattr(streaming_response, "conn_5tuple",
                self._conn_5tuple(response))

        # NOTE: do NOT tighten the socket to stream_idle_timeout here.
        # requests.post returns as soon as the response HEADERS arrive
        # (uvicorn sends them before the first SSE body chunk), so a
        # settimeout(idle) at this point makes the FIRST event's recv
        # obey the idle budget instead of the first-event budget —
        # server-mode requests whose first event legitimately arrives
        # after minutes of decode+prefill+queueing die at ~idle seconds
        # with "Read timed out." (2026-10-20 04:24 matrix on the
        # Qwen3.6 twin: no_emulation and 5g_urban partial failures).
        # The idle budget is applied by _iter_lines_idle_guarded only
        # AFTER the first line has landed (see below); until then the
        # socket keeps the first-event read timeout set above.
        engine_stats = None
        for line in self._iter_lines_idle_guarded(response):
            if not line or not line.startswith("data: "):
                continue
            now = time.time()
            payload_str = line[len("data: "):]
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            if "error" in payload:
                raise RuntimeError(
                    f"server error: {payload['error'].get('message', payload['error'])}")

            if "engine_stats" in payload:
                engine_stats = payload["engine_stats"]
                stats_bytes = len(line.encode("utf-8")) + 2
                streaming_response.response_events.append({
                    "format": "encoder_placement.chat.completions.engine_stats",
                    "timestamp": now,
                    "bytes": stats_bytes,
                    "payload": payload,
                })
                streaming_response.add_event_bytes(stats_bytes)
                continue

            # Full SSE line ("data: " prefix + payload) plus the "\n\n"
            # event separator stripped by iter_lines.
            chunk_bytes = len(line.encode("utf-8")) + 2
            streaming_response.response_events.append({
                "format": "encoder_placement.chat.completions.chunk",
                "timestamp": now,
                "bytes": chunk_bytes,
                "payload": payload,
            })

            usage = payload.get("usage")
            if usage:
                if usage.get("prompt_tokens") is not None:
                    streaming_response.tokens_in = usage["prompt_tokens"]
                if usage.get("completion_tokens") is not None:
                    streaming_response.tokens_out = usage["completion_tokens"]
                streaming_response.add_event_bytes(chunk_bytes)
                continue

            choices = payload.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    streaming_response.add_chunk(
                        content, chunk_bytes=chunk_bytes, timestamp=now)
                else:
                    # Role-only first delta and the final finish_reason
                    # delta carry no content but still count toward
                    # response_bytes (keeps sum(events.bytes) ==
                    # response_bytes over the whole stream).
                    streaming_response.add_event_bytes(chunk_bytes)

        if engine_stats is not None:
            streaming_response.response_payload = {
                "format": "encoder_placement.chat.completions.engine_stats",
                "payload": engine_stats,
            }
            setattr(streaming_response, "engine_stats", engine_stats)

        return streaming_response

    # ------------------------------------------------------------------
    # Public API used by the scenario
    # ------------------------------------------------------------------
    def generate_content_with_video(
        self,
        prompt: str,
        video_path: str,
        model: str,
        mode: str = "server",
        max_frames: int = 768,
        max_tokens: int = 256,
        keep_full_bundle: bool = False,
        compressor: str = "",
        retention_ratio: float = 0.25,
        bundle_quant: str = "",
        vision_only: bool = False,
        attn_impl: str = "sdpa",
        user_tag: str = "",
        frame_ratio: float | None = None,
    ) -> StreamingResponse:
        """
        Run one streaming turn in the requested mode.

        server: upload mp4 bytes; server does everything.
        edge:   run edge pipeline (decode+processor+ViT), upload the
                precomputed_embedding bundle. keep_full_bundle=False
                (default) uploads only the server-consumed keys (see
                _EDGE_BUNDLE_KEYS), dropping pixel_values* dead weight;
                True reproduces the original fat bundle.
        frame_ratio (0 < r <= 1, default None=off): ratio-of-native
                frame sampling (v3: sglang's 2fps-x-duration base x
                ratio, then native clamps). EDGE mode: sampled locally
                in _prepare_edge_bundle via the same patched
                smart_nframes (canary-verified per request when
                active). SERVER mode: the ENGINE samples with its
                STARTUP-time mm_process_config — the value rides in
                the body for record-keeping and the sidecar warns on
                mismatch (start the sidecar with the matching
                --frame-ratio). max_frames caps BOTH sampling modes
                independently (native sglang key, no patch);
                default 768 = sglang's FPS_MAX_FRAMES.
        compressor="vidcom2" (edge only) additionally prunes the bundle to
                retention_ratio of the video tokens via the VidCom2
                selector; '' (default) uploads all tokens.
        bundle_quant="fp8" (edge only, requires compressor="vidcom2")
                quantizes the pruned feature tensor to fp8 e4m3 with a
                per-tensor scale before upload (halves feature bytes);
                '' (default) uploads the raw bf16 feature.
        vision_only=True (edge only) loads ONLY the vision tower
                (Qwen3VLVisionModel: ViT + merger + deepstack, ~0.4B
                params) instead of the full 4B model just to grab
                model.model.visual — same class, same weights, identical
                embeddings; cuts edge VRAM ~8GB -> ~1GB. False
                (default) keeps the historical full load. Ignored in
                server mode.
        attn_impl (edge only): attention backend for the edge ViT —
                'sdpa' (default), 'flash_attention_4' (flash-attn-4
                package; load failure raises, no silent fallback), or
                'eager'. flash_attention_2 is NOT supported on this
                stack. Backends are NOT bit-identical; cross-backend
                comparisons need a tolerance. Ignored in server mode.
        user_tag: short per-user marker (e.g. '[u3] ') prepended to
                the TEXT part, which is emitted BEFORE the video
                placeholder when a tag is given (empty tag keeps the
                historical video-first order byte-for-byte). Distinct
                tags give every concurrent user a distinct token
                prefix, which defeats engine radix-cache prefix
                sharing BETWEEN users of a wave (identical prompts
                would otherwise let user #2+ hit user #1's cached
                ~14.5k-token prefill and report a decode-only TTFT).
                Empty = historical shared-prompt behaviour (all users
                byte-identical).
        """
        from typing import Optional as _Optional  # noqa: F401

        resolved = os.path.abspath(video_path)
        if mode not in ("server", "edge"):
            raise ValueError(
                f"invalid mode {mode!r}: must be 'server' or 'edge'")
        attn_impl = _resolve_attn_impl(attn_impl)
        if user_tag and "\n" in user_tag:
            raise ValueError("user_tag must be a single-line marker")
        if frame_ratio and not (0.0 < frame_ratio <= 1.0):
            raise ValueError(
                f"frame_ratio must be in (0, 1], got {frame_ratio}")
        t_e2e_start = time.time()
        with open(resolved, "rb") as f:
            video_bytes = f.read()

        print(f"[EncPl] mode={mode}"
              + (f" frame_ratio={frame_ratio}" if frame_ratio else "")
              + (f" max_frames={max_frames}" if max_frames != 768 else "")
              + (f" compressor={compressor}"
                 f" retention_ratio={retention_ratio}"
                 + (f" bundle_quant={bundle_quant}"
                    if bundle_quant else "")
                 if (compressor and mode == "edge") else "")
              + (f" vision_only=True" if (vision_only
                                          and mode == "edge") else "")
              + (f" attn_impl={attn_impl}" if (attn_impl != "sdpa"
                                                and mode == "edge") else ""))

        # Both modes render the same chat-templated text (identical
        # markup + video placeholder), keeping prefill symmetric. The
        # per-user tag rides inside the text part (see
        # _build_templated_text): distinct per user, so concurrent
        # users' prefixes cannot share radix-cache entries, yet the
        # video token span and everything after it stay byte-identical.
        processor = self._get_processor(model)
        templated_text = self._build_templated_text(
            processor, resolved, prompt, user_tag=user_tag)

        upload_meta: dict = {}
        edge_stats: dict = {}
        if mode == "edge":
            # sglang patches were ensured at __init__ (construction is
            # outside every timing window); edge-mode sampling happens
            # HERE (the same patched smart_nframes the engine would
            # use); canary-verify before the first edge request
            # actually needs it.
            verify_frame_ratio_patch(active=bool(frame_ratio))
            visual = self._get_edge_model(model, vision_only=vision_only,
                                          attn_impl=attn_impl)
            bundle_bytes, edge_stats = self._prepare_edge_bundle(
                visual, processor, resolved, templated_text, max_frames,
                keep_full_bundle=keep_full_bundle,
                compressor=compressor if mode == "edge" else "",
                retention_ratio=retention_ratio,
                bundle_quant=bundle_quant if mode == "edge" else "",
                frame_ratio=frame_ratio,
            )
            upload_meta = self._upload_bundle(bundle_bytes, kind="edge")
            bundle_ref = f"bundle://edge/{upload_meta['id']}"
        else:
            upload_meta = self._upload_bundle(video_bytes, kind="video")
            bundle_ref = f"bundle://video/{upload_meta['id']}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": bundle_ref}},
                {"type": "text", "text": templated_text},
            ],
        }]

        streaming_response = self._stream_completion(
            messages, model=model, mode=mode,
            max_frames=max_frames, max_tokens=max_tokens,
            frame_ratio=frame_ratio,
        )

        # Attach upload/edge metadata for the scenario's LogRecord.
        server_stats = getattr(streaming_response, "engine_stats", {}) or {}
        if isinstance(streaming_response.response_payload, dict):
            server_stats = streaming_response.response_payload.get(
                "payload", server_stats)
        stats = dict(server_stats)
        stats.update({
            "mode": mode,
            "video_path": resolved,
            "video_payload_bytes": len(video_bytes),
            "video_wire_bytes": _on_wire_len(len(video_bytes)),
            "uploaded_payload_bytes": upload_meta.get("payload_bytes"),
            "uploaded_wire_bytes": upload_meta.get("wire_bytes"),
            "upload_time": upload_meta.get("upload_time"),
            "vision_only": bool(vision_only) if mode == "edge" else False,
            "attn_impl": attn_impl if mode == "edge" else "sdpa",
            "frame_ratio": float(frame_ratio) if frame_ratio else None,
            "max_frames": int(max_frames) if max_frames else None,
            # Frame-sampling semantics version: lets the bench's resume
            # guard refuse results files written under a different
            # sampling semantics (same ratio value, different frames).
            # Kept in sync with tools/encpl_bench_eval.py
            # FRAME_SAMPLING_SEMANTICS.
            "frame_sampling_semantics": "v3-ratio-of-native",
        })
        stats.update(edge_stats)
        # Per-request connection 5-tuples (upload + SSE). The concurrent
        # scenario/wave and the edge worker both propagate engine_stats
        # into the per-user LogRecord metadata and the wave summary, so
        # a pcap of an N-user wave can be split per user by src_port.
        stats["conns"] = {
            "upload": upload_meta.get("conn_5tuple"),
            "chat": getattr(streaming_response, "conn_5tuple", None),
        }
        streaming_response.request_bytes = (
            (streaming_response.request_bytes or 0)
            + (upload_meta.get("payload_bytes") or 0)
        )
        setattr(streaming_response, "engine_stats", stats)

        if streaming_response.t_first_chunk is not None:
            e2e_ttft = streaming_response.t_first_chunk - t_e2e_start
            stats["e2e_ttft"] = e2e_ttft
            if streaming_response.t_last_chunk is not None:
                stats["e2e_total"] = streaming_response.t_last_chunk - t_e2e_start
            print(f"[EncPl] E2E TTFT (video input -> first token): {e2e_ttft:.2f} s")

        if mode == "edge":
            stage1 = edge_stats.get("stage1_edge") or 0.0
            stage2 = edge_stats.get("stage2_vit") or 0.0
            upload_time = upload_meta.get("upload_time") or 0.0
            server_ttft = server_stats.get("server_ttft") or 0.0
            print(f"[EncPl] mode=edge | stage1: {stage1:.2f} s | "
                  f"stage2 (ViT): {stage2:.2f} s | upload: {upload_time:.2f} s | "
                  f"server_ttft: {server_ttft:.2f} s | "
                  f"sum: {stage1 + stage2 + upload_time + server_ttft:.2f} s")
            stage1_bd = edge_stats.get("stage1_breakdown") or {}
            if stage1_bd:
                print(f"[EncPl]   stage1 breakdown: "
                      f"open={stage1_bd.get('open', 0):.3f}s | "
                      f"decode_sample={stage1_bd.get('decode_sample', 0):.3f}s | "
                      f"processor={stage1_bd.get('processor', 0):.3f}s")
            stage2_bd = edge_stats.get("stage2_breakdown") or {}
            if stage2_bd:
                print(f"[EncPl]   stage2 breakdown: "
                      f"h2d={stage2_bd.get('h2d', 0):.3f}s | "
                      f"vit_fwd={stage2_bd.get('vit_fwd', 0):.3f}s | "
                      f"assemble={stage2_bd.get('assemble', 0):.3f}s")
            if edge_stats.get("sampled_nframes") is not None:
                fps_part = (f" fps={edge_stats['video_fps']:.2f}"
                            if edge_stats.get("video_fps") else "")
                print(f"[EncPl]   frames: "
                      f"total={edge_stats.get('total_frames')}{fps_part} -> "
                      f"sampled={edge_stats['sampled_nframes']} "
                      f"(frame_ratio={frame_ratio or 'off'}, "
                      f"max_frames={max_frames or 'default'})")
            e2e_ttft_val = stats.get("e2e_ttft")
            if e2e_ttft_val is not None:
                gap = (e2e_ttft_val
                       - stage1 - stage2 - upload_time - server_ttft)
                # edge_prep_time covers _prepare_edge_bundle only
                # (decode..torch.save); file read / templated text /
                # model load sit outside it and land in `other`.
                edge_riders = ((edge_stats.get("edge_prep_time") or 0.0)
                               - stage1 - stage2)
                t_prepare = server_stats.get("t_prepare") or 0.0
                other = gap - edge_riders - t_prepare
                print(f"[EncPl]   reconcile: e2e_ttft={e2e_ttft_val:.2f}s "
                      f"sum={stage1 + stage2 + upload_time + server_ttft:.2f}s "
                      f"gap={gap:.2f}s | "
                      f"edge_riders={edge_riders:.2f}s "
                      f"(vidcom2/quant/drop_est/d2h/ser) | "
                      f"t_prepare={t_prepare:.2f}s (server torch.load+dequant) | "
                      f"other={other:.2f}s "
                      f"(file read+templated text+model load+POST RTT+SSE down)")
        else:
            stage1 = server_stats.get("t_prepare") or 0.0
            server_ttft = server_stats.get("server_ttft") or 0.0
            upload_time = upload_meta.get("upload_time") or 0.0
            print(f"[EncPl] mode=server | upload: {upload_time:.2f} s | "
                  f"stage1 (server prepare): {stage1:.2f} s | "
                  f"server_ttft: {server_ttft:.2f} s | "
                  f"sum: {upload_time + stage1 + server_ttft:.2f} s")

        return streaming_response

    # ------------------------------------------------------------------
    # LLMClient interface (kept minimal; the scenario uses the placement API)
    # ------------------------------------------------------------------
    def chat(self, messages, model, stream=False, tools=None, **kwargs):
        raise NotImplementedError("use generate_content_with_video()")

    def chat_streaming(self, messages, model, **kwargs):
        raise NotImplementedError("use generate_content_with_video()")


def _on_wire_len(payload_bytes: int) -> int:
    packets = max(1, math.ceil(payload_bytes / 1460))
    return payload_bytes + packets * 40 + 4
