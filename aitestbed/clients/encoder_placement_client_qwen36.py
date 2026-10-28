"""
Encoder Placement client adapter (Qwen3.6 variant) for the 6G AI
Traffic Testbed.

Standalone sibling of encoder_placement_client.py for the Qwen3.6
(qwen3_5 / qwen3_5_moe) experiment — zero changes to the Qwen3-VL
files, so both experiments can run/debug in parallel. Talks to the
Qwen3.6 sidecar (server/sglang_engine_server_qwen36.py, default port
8004).

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

Qwen3.6 specifics (vs the Qwen3-VL client — see
aicodec/aicodec/models/Qwen3.6-35B-A3B-FP8/config.json):

  - model_type "qwen3_5_moe" (35B-A3B MoE, arch
    Qwen3_5MoeForConditionalGeneration) or "qwen3_5" (27B dense, arch
    Qwen3_5ForConditionalGeneration); both are hybrid GDN
    linear-attention MoE/dense stacks in transformers 5.x. The class
    dispatch mirrors the Qwen3-VL client's (a wrong class silently
    drops experts / random-inits projections).
  - vision tower: SAME Qwen3-VL tower family (1152 hidden, 27 depth,
    patch 16 / merge 2 / temporal 2, out_hidden_size 2048) but
    deepstack_visual_indexes == [] — the tower returns a SINGLE
    merged [tokens, out_hidden_size] segment. The edge bundle's
    feature tensor is therefore [N, 2048] (4 KB/token at bf16), 1/4
    of Qwen3-VL-30B's [N, 8192] (main + 3 deepstack segments). The
    latent-assembly code handles the empty-deepstack case directly
    (torch.cat of a single tensor is a no-op path).
  - token ids moved (248320 vocab): vision_start 248053, vision_end
    248054, image_pad 248056, video_pad 248057. They are resolved
    from the processor/tokenizer at runtime — never hardcoded — and
    the video-pad runs in the bundle are validated against
    video_grid_thw (same fail-fast contract as the Qwen3-VL client).
  - processor: Qwen3VLProcessor (preprocessor_config.json /
    video_preprocessor_config.json say so); the processor call mirrors
    sglang's qwen3-family branch exactly (videos + video_metadata +
    do_sample_frames=False, qwen_vl.py model_type tuple includes
    "qwen3_5" / "qwen3_5_moe") — NOT the 2.5 branch.
  - video pixel budget identical to Qwen3-VL (longest_edge 25165824 /
    shortest_edge 4096): same max video token count (12288), so the
    bundle-size A/B against Qwen3-VL isolates the latent-width
    change alone (4 KB vs 16 KB per token for 30B).
  - mrope: text side uses partial_rotary 0.25 + mrope_section
    [11,11,10]; the processor emits mrope_positions [3, N] exactly as
    before (whitelist key, 24 B/token, negligible). No bundle change.

VidCom2 compression / fp8 bundle quantization / frame_ratio sampling
are ported from the Qwen3-VL client (same knobs, same semantics):
compressor="vidcom2" prunes the edge bundle to retention_ratio of the
video tokens (in-repo token_compressor checkout under
aicodec/aicodec/models/VidCom2);
bundle_quant="fp8" quantizes the pruned feature to fp8 e4m3 with a
per-tensor scale (halves the uploaded feature bytes; the sidecar
dequantizes right after torch.load); frame_ratio (0 < r <= 1) selects
ratio-of-native frame sampling via the patched smart_nframes (v3
semantics; canary-verified per request when active; the sidecar must
be started with the SAME --frame-ratio for server-mode record
keeping).

The TTFT window spans the full video-input -> first-token path in
both modes (upload/edge work happens inside the window, strictly
synchronous).
"""

import io
import json
import math
import os
import socket
import struct
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

from .base import LLMClient, StreamingResponse, estimate_payload_bytes

# Shared frame-sampling wiring with the sidecars (same import + auto
# patch semantics as the Qwen3-VL client: ensure_sglang_patches is
# called once in __init__, OUTSIDE all timing windows, so the edge
# worker's fresh processes auto-load the sglang disk patch;
# verify_frame_ratio_patch canary-checks the ACTIVE sglang's
# smart_nframes for the frame_ratio branch before edge sampling runs).
sys.path.insert(0, str(Path(os.path.dirname(__file__)).parent / "server"))
from encpl_video_sampling import (
    ensure_sglang_patches,
    verify_frame_ratio_patch,
)

# Same guard as the sidecar (server/sglang_engine_server_qwen36.py):
# deep_ep's import-time JIT init crashes without CUDA_HOME and the
# AssertionError escapes sglang's `except ImportError`, killing
# `import sglang` on machines without the toolkit. The Qwen3.6 edge
# path (decode + processor + ViT) never needs deep_ep; the MoE engine
# side runs standard expert-parallel-free routing.
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

    Same helper as the Qwen3-VL client: the token_compressor package
    lives in aicodec/aicodec/models/VidCom2 and is imported by the edge
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

    Same per-tensor symmetric scale contract as the Qwen3-VL client:
    scale = amax / 448.0 (448 = max representable e4m3 value), applied
    as feature_fp8 = to(float8_e4m3fn)(embeddings / scale). The scale
    rides along in the bundle as `feature_scale` (a plain float); the
    sidecar's _maybe_dequant_bundle_feature pops it and reconstructs
    bf16 = scale * to(bfloat16)(feature_fp8) right after torch.load, so
    everything downstream of deserialization (engine hash, shm
    transfer, embed_multimodal_inputs dtype checks) sees bf16 tensors
    equivalent to an unquantized run.

    Only reached with bundle_quant='fp8' AND compressor='vidcom2' (the
    caller enforces the coupling), i.e. after VidCom2 row pruning — the
    fp8 tensor is what actually gets uploaded. On Qwen3.6 the feature
    is the SINGLE [N, 2048] merged segment (no deepstack), so the
    quantized bytes are exactly N*2048 (vs N*8192 on Qwen3-VL-30B).

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


# ----------------------------------------------------------------------
# E2U: UDP wire format for the optional-tier second wave
# ----------------------------------------------------------------------
# Datagram = 30-byte header + up to _E2U_MAX_PAYLOAD bytes of a
# torch.save'd part (see _build_opt_parts). One part is fragmented
# across as many datagrams as needed; the receiver reassembles and
# only complete parts reach the BundleStore (a part with ANY missing
# fragment is dropped whole — true UDP semantics, no retransmit).
# NO CRC: on a controlled bench LAN the loss mode under test is
# congestion/late-drop, not corruption, and a corrupted datagram only
# ever fails the magic/length validation or the part's torch.load.
_E2U_MAGIC = b"E2U1"
_E2U_HEADER = struct.Struct("<4s16sIHHH")   # magic, bundle_id, part_seq,
                                            # frag_idx, frag_cnt, len
_E2U_HEADER_SIZE = _E2U_HEADER.size        # 30
_E2U_MAX_PAYLOAD = 1400 - _E2U_HEADER_SIZE  # 1370
_E2U_DATAGRAM_SIZE = 1400


def _e2u_pack(bundle_id: str, part_seq: int, frag_idx: int, frag_cnt: int,
              payload: bytes) -> bytes:
    """Encode one E2U datagram (header + payload slice)."""
    return _E2U_HEADER.pack(
        _E2U_MAGIC, uuid.UUID(bundle_id).bytes,
        int(part_seq), int(frag_idx), int(frag_cnt), len(payload),
    ) + payload


def _e2u_unpack(datagram: bytes) -> tuple[bytes, bytes, int, int, int, int]:
    """Decode one E2U datagram; raises ValueError on bad magic/length."""
    if len(datagram) < _E2U_HEADER_SIZE:
        raise ValueError(f"E2U datagram too short ({len(datagram)}B)")
    magic, bid, part_seq, frag_idx, frag_cnt, plen = _E2U_HEADER.unpack(
        datagram[:_E2U_HEADER_SIZE])
    if magic != _E2U_MAGIC:
        raise ValueError(f"E2U bad magic {magic!r}")
    if plen != len(datagram) - _E2U_HEADER_SIZE:
        raise ValueError(
            f"E2U header len {plen} != payload "
            f"{len(datagram) - _E2U_HEADER_SIZE}")
    return bid, datagram[_E2U_HEADER_SIZE:], part_seq, frag_idx, frag_cnt, plen


def _e2u_fragments(bundle_id: str, part_seq: int, part: bytes) -> list[bytes]:
    """Slice one part into E2U datagrams (>= 1 even for empty parts)."""
    frags = []
    frag_cnt = max(1, math.ceil(len(part) / _E2U_MAX_PAYLOAD))
    for frag_idx in range(frag_cnt):
        lo = frag_idx * _E2U_MAX_PAYLOAD
        payload = part[lo:lo + _E2U_MAX_PAYLOAD]
        frags.append(_e2u_pack(
            bundle_id, part_seq, frag_idx, frag_cnt, payload))
    return frags


_VALID_ATTN_IMPLS = ("sdpa", "flash_attention_4", "eager")


def _resolve_attn_impl(attn_impl: str) -> str:
    """Normalize/validate the edge ViT attention backend selector.

    Accepts 'sdpa' (default), 'flash_attention_4' (the flash-attn-4
    package / kernels-community backend — the ONLY flash backend this
    stack supports) and 'eager'. Returns the normalized value; raises
    ValueError on anything else.

    NOTE: flash_attention_2 is intentionally NOT offered (same stack
    rationale as the Qwen3-VL client: the epd env ships flash-attn-4
    beta, whose shim fails transformers' fa2 gate).
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
            f"fails transformers' fa2 gate. Use 'flash_attention_4' "
            f"for flash kernels or 'sdpa' (default)")
    if impl == "eager":
        return "eager"
    raise ValueError(
        f"invalid attn_impl {attn_impl!r}: must be one of "
        f"{_VALID_ATTN_IMPLS}")


def _qwen3_6_model_type(model_path: str) -> str:
    """Read the checkpoint's top-level model_type from its config.json.

    Shared by _qwen3_6_model_class / _qwen3_6_vision_class. Raises
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
            f"({exc}); pass a local Qwen3.6 model directory"
        ) from exc
    return model_type


_QWEN3_6_MODEL_TYPES = ("qwen3_5", "qwen3_5_moe")


def _qwen3_6_model_class(model_path: str):
    """Pick the transformers class matching the checkpoint's architecture.

    Qwen3.6 ships dense (27B: model_type "qwen3_5",
    Qwen3_5ForConditionalGeneration) and MoE (35B-A3B: model_type
    "qwen3_5_moe", Qwen3_5MoeForConditionalGeneration) checkpoints.
    Loading a MoE checkpoint with the dense class silently drops every
    mlp.experts.* / mlp.gate weight (UNEXPECTED) and freshly
    random-initializes the dense mlp.* projections (MISSING) — the same
    load-report failure mode the Qwen3-VL client guards against.
    Dispatch on config.json so the right class is used; both expose
    model.model.visual identically.

    transformers gate: the epd env pins transformers 5.12.1, which
    ships the qwen3_5 family (the checkpoint header's
    transformers_version 4.57.1 is the EXPORT-time version, not a
    runtime requirement). Raises RuntimeError naming the version when
    the class is absent.

    Returns the transformers class. Raises RuntimeError for unknown
    model_types (fail fast instead of a wrong-arch load).
    """
    model_type = _qwen3_6_model_type(model_path)

    import transformers

    if model_type == "qwen3_5_moe":
        cls = getattr(transformers, "Qwen3_5MoeForConditionalGeneration", None)
        if cls is None:
            raise RuntimeError(
                "this transformers has no Qwen3_5MoeForConditionalGeneration "
                f"(version {transformers.__version__}); MoE checkpoints like "
                "Qwen3.6-35B-A3B need a transformers with qwen3_5_moe "
                "support (the epd env pins 5.12.1, which has it)")
        return cls
    if model_type == "qwen3_5":
        cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if cls is None:
            raise RuntimeError(
                "this transformers has no Qwen3_5ForConditionalGeneration "
                f"(version {transformers.__version__}); dense checkpoints "
                "like Qwen3.6-27B need a transformers with qwen3_5 "
                "support (the epd env pins 5.12.1, which has it)")
        return cls
    raise RuntimeError(
        f"unsupported model_type {model_type!r} at {model_path}: expected "
        f"one of {_QWEN3_6_MODEL_TYPES} (Qwen3.6 dense 27B / MoE 35B-A3B); "
        f"for qwen3_vl / qwen3_vl_moe use the Qwen3-VL client")


def _qwen3_6_vision_class(model_path: str):
    """Pick the standalone vision-tower class matching the checkpoint.

    Qwen3.6 keeps the Qwen3-VL tower family (transformers
    Qwen3VLVisionModel / Qwen3VLMoeVisionModel for dense / MoE
    checkpoints). A dense-vs-MoE class mismatch maps nothing: every
    checkpoint key lands UNEXPECTED and the whole tower is randomly
    initialized (MISSING) — the exact silent-garbage mechanism the
    Qwen3-VL vision_only path hit on 30B. Dispatch on config.json's
    top-level model_type (the vision sub-config model_type is
    normalized differently between transformers versions and is NOT a
    reliable discriminator).

    deepstack_visual_indexes == [] on Qwen3.6: the tower's forward
    returns the single merged [N, out_hidden_size] tensor (pooler_output
    / tuple[0]), no deepstack segment concat — the caller assembles
    the sglang-contract tensor with the empty-list cat path.

    Returns the transformers class. Raises RuntimeError for unknown
    model_types or a transformers without the class.
    """
    model_type = _qwen3_6_model_type(model_path)

    import transformers

    if model_type == "qwen3_5_moe":
        cls = getattr(transformers, "Qwen3VLMoeVisionModel", None)
        if cls is None:
            raise RuntimeError(
                "this transformers has no Qwen3VLMoeVisionModel "
                f"(version {transformers.__version__}); MoE checkpoints like "
                "Qwen3.6-35B-A3B need a transformers with qwen3_vl_moe "
                "vision support")
        return cls
    if model_type == "qwen3_5":
        cls = getattr(transformers, "Qwen3VLVisionModel", None)
        if cls is None:
            raise RuntimeError(
                "this transformers has no Qwen3VLVisionModel "
                f"(version {transformers.__version__}); dense checkpoints "
                "like Qwen3.6-27B need a transformers with qwen3_vl "
                "vision support")
        return cls
    raise RuntimeError(
        f"unsupported model_type {model_type!r} at {model_path}: expected "
        f"one of {_QWEN3_6_MODEL_TYPES}")


# Prefixes a composite Qwen3.6 checkpoint may ship its vision tower
# under. Priority matters: model.visual.* is the layout the composite
# model classes actually read (the Qwen3.6-35B weight_map ships
# model.visual.* in outside.safetensors); a bare visual.* block (when
# present) is usually a stale/duplicated export and must NEVER
# override the model.visual.* tensors for the same tower key.
_VISUAL_PREFIXES = ("model.visual.", "visual.")


def _remap_visual_state_dict(checkpoint_sd: dict, tower_sd_keys) -> dict:
    """Build a tower-keyed state dict from the composite checkpoint.

    Key remap: checkpoints store the tower under model.visual.* (and,
    for some safetensors layouts, visual.*). A standalone tower
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
    (model.safetensors.index.json + N shards) — the Qwen3.6-35B-A3B
    checkpoint is sharded (layers-*.safetensors + outside.safetensors
    + mtp.safetensors), so reading only model.safetensors would miss
    most of the tower. Returns the merged {key: tensor} dict (CPU
    tensors).

    VRAM note: this is a FULL read (the MoE experts ride along, ~35 GB
    fp8 on disk). The vision_only production path only calls it when
    the standalone load report shows mismatches (the remap guard);
    the census in the checker tool calls it on CPU.
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


def _vision_load_report(visual) -> tuple[list, list]:
    """Missing/unexpected weight keys after a from_pretrained load.

    Best-effort fallback for modules loaded without
    output_loading_info (the loader prefers the programmatic path).
    """
    info = getattr(visual, "load_info", None)
    if isinstance(info, dict):
        return (list(info.get("missing_keys") or []),
                list(info.get("unexpected_keys") or []))
    missing = getattr(visual, "missing_keys", None)
    unexpected = getattr(visual, "unexpected_keys", None)
    return (list(missing or []), list(unexpected or []))


def _load_edge_model(model_path: str, vision_only: bool = False,
                     attn_impl: str = "sdpa"):
    """Load the visual tower for edge-side encoding (returns the module).

    transformers' Qwen3-VL-family vision tower matching the Qwen3.6
    checkpoint, producing the [N, out_hidden_size] final embeddings
    (SINGLE merged segment — deepstack_visual_indexes is empty on
    Qwen3.6) SGLang expects as precomputed_embeddings. The processor
    is shared via EncoderPlacementQwen36Client._processor_cache.

    vision_only=False (default): load the FULL conditional-generation
    model — dense (qwen3_5) or MoE (qwen3_5_moe), dispatched via
    _qwen3_6_model_class so the 35B-A3B checkpoint loads with the
    right architecture — and return model.model.visual (the LLM half
    is loaded and discarded). On the fp8 35B checkpoint this needs
    ~35 GB VRAM just to obtain the ~0.5 GB tower; prefer
    vision_only=True.

    vision_only=True: load ONLY the vision tower (ViT + patch merger)
    via the checkpoint-matched standalone class (Qwen3VLVisionModel
    for dense, Qwen3VLMoeVisionModel for MoE — see
    _qwen3_6_vision_class), same weights (model.visual.* keys map
    automatically once the class matches), ~0.5B params vs the full
    model, cutting edge VRAM and load time by the LLM weight read.
    The vision tower is in the checkpoint's fp8
    modules_to_not_convert list, so the standalone tower loads in
    bf16 exactly like the composite model's own tower.

    The standalone load is GUARDED: the transformers load report is
    parsed and any missing/unexpected vision keys trigger a state-dict
    remap from the checkpoint's model.visual.* prefix — and, if keys
    still remain unmatched, a hard RuntimeError (fail fast; a
    benchmark must never silently measure a randomly-initialized
    tower, which surfaces downstream as empty/echoing answers in edge
    mode while server mode stays clean).

    attn_impl: attention backend for the tower ('sdpa' default,
    'flash_attention_4', 'eager'); see _resolve_attn_impl. Backends
    are NOT bit-identical (kernel accumulation order); comparisons
    across attn_impl need a tolerance, and the vision-only
    equivalence check pins both loads to the SAME impl.
    """
    import torch

    impl = _resolve_attn_impl(attn_impl)
    model_path = _resolve_model_path(model_path)
    if vision_only:
        try:
            vision_cls = _qwen3_6_vision_class(model_path)
        except (RuntimeError, ImportError) as exc:
            print(f"[EncPl36] warning: vision_only requested but no "
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
                print(f"[EncPl36] warning: {vision_cls.__name__} load failed "
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
                    # remap model.visual.* / visual.* checkpoint keys
                    # into the tower's own layout and load_state_dict
                    # (strict) — covers transformers layouts where the
                    # standalone tower's expected keys don't match the
                    # composite checkpoint's shipped prefix (e.g. a
                    # mismatched standalone class silently leaving every
                    # tower weight MISSING → random init → garbage edge
                    # embeddings).
                    print(f"[EncPl36] vision_only standalone load mismatch "
                          f"(missing={len(missing)}, "
                          f"unexpected={len(unexpected)}); attempting "
                          f"state-dict remap")
                    checkpoint_sd = _load_checkpoint_state_dict(model_path)
                    remapped = _remap_visual_state_dict(
                        checkpoint_sd, visual.state_dict().keys())
                    if remapped:
                        visual.load_state_dict(remapped, strict=True)
                        print(f"[EncPl36] remap loaded {len(remapped)} "
                              "visual tensors")
                        missing, unexpected = [], []
                    if missing or unexpected:
                        # fail fast — silently benching a
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

    model_cls = _qwen3_6_model_class(model_path)
    if model_cls.__name__ == "Qwen3_5MoeForConditionalGeneration":
        print(f"[EncPl36] loading {model_path} as "
              f"Qwen3_5MoeForConditionalGeneration (MoE checkpoint)")
    model = model_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=impl,
    )
    visual = model.model.visual
    visual.eval()
    return visual


class EncoderPlacementQwen36Client(LLMClient):
    """
    Terminal client for the Qwen3.6 SGLang engine sidecar.
    """

    DEFAULT_BASE_URL = "http://localhost:8004"
    _edge_model_cache: dict = {}
    _processor_cache: dict = {}

    # Edge-bundle whitelist: the only processor-output keys the server
    # actually consumes (sidecar _prepare_edge_mode reads input_ids;
    # base_processor.py pops "feature"; qwen_vl.py reads
    # video_grid_thw / mrope_positions / mrope_position_delta /
    # second_per_grid_ts). pixel_values* is pure dead weight once the
    # ViT has run locally — it rides along in the full processor dict
    # and roughly doubles the bundle. Same whitelist as the Qwen3-VL
    # client: the bundle CONTRACT is unchanged by Qwen3.6 (only the
    # feature width narrows to out_hidden_size, no deepstack
    # segments).
    _EDGE_BUNDLE_KEYS = (
        "input_ids",
        "video_grid_thw",
        "second_per_grid_ts",
        "mrope_positions",
        "mrope_position_delta",
    )

    # Stream safety timeouts (seconds; 0 disables the respective
    # guard, restoring the historical unlimited wait). Defaults are
    # tuned for the matrix topology: local loopback TCP connect 20,
    # first SSE event 200 (serial arms; the concurrent runner passes
    # its own larger value), idle gap between SSE events 20, and the
    # bundle-upload response read 1200 (cell_edge at 5 Mbit/s moves
    # ~300 MB in ~7 min, so the READ side only needs to cover the
    # server's store+ack, but 1200 tolerates a straggler profile).
    DEFAULT_STREAM_CONNECT_TIMEOUT = 20.0
    DEFAULT_STREAM_FIRST_EVENT_TIMEOUT = 200.0
    DEFAULT_STREAM_IDLE_TIMEOUT = 20.0
    DEFAULT_STREAM_UPLOAD_READ_TIMEOUT = 1200.0

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 stream_connect_timeout: Optional[float] = None,
                 stream_first_event_timeout: Optional[float] = None,
                 stream_idle_timeout: Optional[float] = None,
                 stream_upload_read_timeout: Optional[float] = None):
        self.api_key = api_key or os.environ.get("SGLANG_ENGINE_QWEN36_API_KEY") or "sglang"
        self.base_url = (
            base_url
            or os.environ.get("SGLANG_ENGINE_QWEN36_BASE_URL")
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
        # every timing window (same rationale as the Qwen3-VL client:
        # workers (spawn processes), the bench driver, scenarios all
        # construct the client BEFORE t_request_start; a per-REQUEST
        # call would bill the patcher's ~20-100ms to the FIRST measured
        # request of every fresh wave-worker process, a systematic TTFT
        # bias vs the server arm). Unconditional: patch 1's edge-mode
        # PRECOMPUTED_EMBEDDING guard is needed even without
        # frame_ratio. Idempotent per process; quiet no-op when the
        # sidecar bootstrap already patched; tolerated as a silent
        # no-op in sglang-less mock environments.
        ensure_sglang_patches(reason="EncoderPlacementQwen36Client.__init__")

    @property
    def provider(self) -> str:
        return "sglang-engine-qwen36"

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
        cached = EncoderPlacementQwen36Client._processor_cache.get(key)
        if cached is None:
            from transformers import AutoProcessor

            cached = AutoProcessor.from_pretrained(key)
            EncoderPlacementQwen36Client._processor_cache[key] = cached
        return cached

    @staticmethod
    def _build_templated_text(processor, video_path: str, prompt: str,
                              user_tag: str = "") -> str:
        """Render the Qwen3.6 chat template around the video placeholder.

        apply_chat_template is pure jinja here (no video decode): the
        video content part expands to
        <|vision_start|><|video_pad|><|vision_end|> with the Qwen3.6
        tokenizer's ids (248053/248057/248054). Both modes use this
        same call so the engine sees byte-identical prompt markup
        whether the video is referenced as raw bytes (server mode) or
        precomputed embeddings (edge mode).

        enable_thinking: the Qwen3.6 template (tokenizer_config.json
        chat_template) is the Qwen3-family renderer, which accepts it;
        pass False for non-thinking outputs, same as the Qwen3-VL
        client.

        user_tag (optional): prepended to the TEXT part, which is
        emitted FIRST (before the video part) when set — kept for
        parity with the Qwen3-VL client's concurrent-wave prefix
        forking; the serial scenario passes "" and gets the
        historical video-first order byte-for-byte.

        Template resolution: the Qwen3.6 repos ship the chat template
        ONLY at tokenizer level (tokenizer_config.json), with no
        processor-level chat_template.json (unlike qwen3_vl_4B).
        ProcessorMixin.apply_chat_template raises "this processor
        does not have a chat template" on such repos, so fall back
        to the TOKENIZER's apply_chat_template (same jinja, same
        video-part rendering — the Qwen3-family renderer natively
        handles {'type': 'video'} content parts).
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
        if not getattr(processor, "chat_template", None):
            # No processor-level template (Qwen3.6 repo shape): the
            # tokenizer holds it. tokenizer.apply_chat_template shares
            # the processor's kwargs contract for the options used
            # here (tokenize / add_generation_prompt / extra jinja
            # vars like enable_thinking).
            tok = getattr(processor, "tokenizer", None)
            if tok is None or not tok.chat_template:
                raise ValueError(
                    "no chat template found at either processor or "
                    "tokenizer level; the model repo is missing "
                    "tokenizer_config.json's chat_template")
            return tok.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
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
        cached = EncoderPlacementQwen36Client._edge_model_cache.get(key)
        if cached is None:
            cached = _load_edge_model(key[0], vision_only=vision_only,
                                      attn_impl=impl)
            EncoderPlacementQwen36Client._edge_model_cache[key] = cached
        return cached

    def _prepare_edge_bundle(self, visual, processor, video_path: str,
                              templated_text: str, max_frames: int,
                              keep_full_bundle: bool = False,
                              compressor: str = "",
                              retention_ratio: float = 0.25,
                              bundle_quant: str = "",
                              frame_ratio: float | None = None,
                              optional_ratio: float = 0.0):
        """
        Run decode + sampling + processor + visual tower locally and
        build the SGLang precomputed_embedding dict.

        Uses SGLang's own decode/sample/resize pipeline (VideoDecoderWrapper
        + preprocess_video) and passes video_metadata +
        do_sample_frames=False to the HF processor — byte-for-byte the
        same frames and processor kwargs the server-side path
        (qwen_vl.py process_mm_data_async) produces for the
        qwen3-family branch (model_type qwen3_5 / qwen3_5_moe is in
        that same tuple, qwen_vl.py:1042-1051), so both modes see
        identical inputs.

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

        optional_ratio (0 <= optional < retention, edge + vidcom2 only)
        splits the retention set into a must tier (this bundle, reliable
        first wave) and an optional tier (least-critical tail of the
        retained set). The must bundle alone is a complete, prefillable
        precomputed_embedding dict — identical to a single-tier run at
        base (retention - optional). The optional rows ride back in
        edge_stats (opt_row_ids / opt_slots / opt_mrope_cols +
        full_row_ids for the slot space) so the caller can stream them
        as self-describing best-effort parts over UDP (E2U datagrams)
        BETWEEN the must PUT and the chat POST (see _send_opt_parts);
        parts that arrive after the server's merge snapshot are
        dropped and never retransmitted.
        0 (default) = single-tier, byte-identical legacy path.

        bundle_quant="fp8" (only valid together with compressor="vidcom2")
        additionally quantizes the PRUNED feature tensor to fp8 e4m3 with
        a per-tensor scale (see _quantize_feature), halving the uploaded
        feature bytes; the server dequantizes right after torch.load.

        Qwen3.6 latent assembly: the tower returns ONE merged
        [N, out_hidden_size] segment (deepstack_visual_indexes == []),
        so the sglang-contract feature tensor is that segment as-is —
        the cat over an empty deepstack list is exercised explicitly
        to keep the code shape identical to the Qwen3-VL client (and
        to fail loudly if a future checkpoint reintroduces deepstack:
        the width contract check below pins width to out_hidden_size).

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
        # semantics). {} -> model defaults (VIDEO_MIN/TOTAL/MAX_PIXELS +
        # 2fps), matching a sidecar started with no sampling flags.
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

        # Processor with the same kwargs the server passes for the
        # qwen3-family branch (qwen_vl.py:1052-1055): sampled frames +
        # video_metadata + do_sample_frames=False -> pixel_values +
        # video_grid_thw + second_per_grid_ts (+ mrope) in one call.
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

        # transformers v5 emits the unified `pixel_values` for videos;
        # older transformers emits the Qwen2-VL-style
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
        # Mirrors sglang's own call (models/qwen3_vl.py, which
        # Qwen3_5*ForConditionalGeneration inherits):
        # self.visual(pixel_values, grid_thw=grid_thw).
        #
        # transformers' tower forward returns (depending on version):
        #   v5:      BaseModelOutputWithDeepstackFeatures with
        #            last_hidden_state = pre-merger block output
        #            [N_patches, hidden_size] (WRONG tensor),
        #            pooler_output = merged [N/4, out_hidden_size],
        #            deepstack_features = [] (empty on Qwen3.6)
        #   4.57.x:  tuple (merged, [deepstack features...]) with an
        #            empty second element
        # The sglang-contract tensor is pooler_output (+ deepstack
        # concat, empty here) — assembled below.
        # ------------------------------------------------------------------
        # Timeline: h2d (CPU->CUDA copies) / vit_fwd (tower forward) /
        # assemble (deepstack unpack + torch.cat), matching the
        # Qwen3-VL client's stage composition.
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
            # Not expected on Qwen3.6 (deepstack_visual_indexes == []),
            # but keep the generic concat so a future checkpoint with
            # deepstack fails the WIDTH contract check below instead of
            # silently shipping a wrong-shaped bundle.
            embeddings = torch.cat([main_embeds] + deepstack, dim=1)
        else:
            embeddings = main_embeds
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_vit = time.time() - t_vit_start
        t_assemble = t_vit - t_h2d - t_vit_fwd

        # Fast-fail contract checks (a transformers API change must fail
        # here with a clear message, not as a scheduler-side crash):
        # Qwen3.6: width == out_hidden_size * (1 + len(deepstack_idx))
        # with deepstack_idx == [] -> width == out_hidden_size (2048).
        vcfg = getattr(visual, "config", None)
        if vcfg is None:
            # HF towers usually carry the vision config as self.config
            vcfg = getattr(getattr(processor, "image_processor", None),
                           "config", None)
        if vcfg is not None:
            merge = getattr(vcfg, "spatial_merge_size", 2)
            deep_idx = list(getattr(vcfg, "deepstack_visual_indexes", None) or [])
            expected_width = vcfg.out_hidden_size * (1 + len(deep_idx))
            expected_rows = sum(
                t_dim * h_dim * w_dim
                for t_dim, h_dim, w_dim in video_grid_thw.tolist()
            ) // (merge * merge)
            if embeddings.shape[-1] != expected_width:
                raise RuntimeError(
                    f"edge ViT output width {embeddings.shape[-1]} != "
                    f"expected {expected_width} (out_hidden_size * "
                    f"(1+deepstack) = {vcfg.out_hidden_size} * "
                    f"(1+{len(deep_idx)})); transformers output layout "
                    f"changed or this checkpoint reintroduced deepstack?")
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
            if not (0.0 <= optional_ratio < retention_ratio):
                raise ValueError(
                    f"invalid optional_ratio {optional_ratio}: must be in "
                    f"[0, retention_ratio)")
            embeddings, inputs, compress_stats = self._compress_edge_bundle(
                embeddings, inputs, video_grid_thw, vcfg,
                retention_ratio, processor,
                optional_ratio=optional_ratio,
            )

        # ------------------------------------------------------------------
        # Tiered mode: the OPT feature rows (bf16, UNPRUNED, send order)
        # were snapshotted INSIDE _compress_edge_bundle before pruning
        # (indexing the must-pruned `embeddings` here would read wrong
        # rows — Bug A). In fp8 runs they are quantized with their OWN
        # per-tensor scale below (the must tier's scale does not apply
        # to these rows — Bug E); the scale rides in every part and the
        # server dequantizes per part.
        # ------------------------------------------------------------------
        opt_feature_rows = compress_stats.pop("opt_feature_rows", None)
        opt_quant_scale: float | None = None
        if opt_feature_rows is not None and bundle_quant:
            opt_feature_rows, _opt_quant_stats = _quantize_feature(
                opt_feature_rows, bundle_quant)
            opt_quant_scale = float(_opt_quant_stats["feature_scale"])

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
        # Tiered bundles: attach the must tier's slot space metadata so
        # the server can merge arriving opt parts by SLOT ("I am the
        # s-th token of the retention-level sequence") without any
        # schema dependency on packet boundaries. must_slots are the
        # positions of the must rows within the retention-level row
        # order; the retention total rides along so the server can
        # sanity-check merges. Non-tiered bundles are untouched.
        if compress_stats.get("tiered"):
            full_row_ids = compress_stats["full_row_ids"]
            must_row_ids = compress_stats["must_row_ids"]
            slot_of = torch.zeros(int(compress_stats["video_tokens_full"]),
                                  dtype=torch.long)
            slot_of[full_row_ids] = torch.arange(full_row_ids.shape[0])
            bundle["format"] = "precomputed_embedding_tiered"
            bundle["must_slots"] = slot_of[must_row_ids]
            bundle["retention_tokens"] = int(full_row_ids.shape[0])
            # BUG S: the authoritative pad id rides with the bundle —
            # the server merge prefers it over the longest-run
            # heuristic (which can misresolve on adversarial prompts).
            if "video_pad_id" in compress_stats:
                bundle["video_pad_id"] = int(compress_stats["video_pad_id"])
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
            # ---- aggregate stage keys (same composition as the
            # Qwen3-VL client, history-comparable):
            # stage1_edge = open + decode_sample + processor,
            # stage2_vit = h2d + vit_fwd + assemble. ----
            "stage1_edge": t_processor + t_decode,
            "stage1_breakdown": {
                "open": t_open,
                "decode_sample": t_decode_sample,
                "processor": t_processor,
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
        # Tiered mode: extract the opt-tier tensors the caller needs to
        # stream parts AFTER the chat request (they must NOT ride in
        # the must bundle — that would defeat the two-wave split).
        # opt_feature_rows was snapshotted from the UNPRUNED bf16
        # embeddings right after selection (before quantization).
        # The 6 tensor keys are popped from EDGE_STATS (their actual
        # home after the update above — popping compress_stats was a
        # no-op on a dict about to be dropped, leaving the tensor
        # copies in edge_stats/engine_stats and blowing up every JSON
        # consumer: bench results files, DB metadata json.dumps, ...).
        # Popping here keeps opt_payload identical while making
        # engine_stats JSON-safe. opt_payload itself rides back in
        # edge_stats under ONE key and the caller pops it (:2178-style)
        # before stats serialization — the historical protocol.
        if compress_stats.get("tiered"):
            opt_payload = {
                "opt_row_ids": edge_stats.pop("opt_row_ids"),
                "opt_slots": edge_stats.pop("opt_slots"),
                "opt_mrope_cols": edge_stats.pop("opt_mrope_cols"),
                "opt_ins_pos": edge_stats.pop("opt_ins_pos"),
                "must_row_ids": edge_stats.pop("must_row_ids"),
                "full_row_ids": edge_stats.pop("full_row_ids"),
                "opt_feature_rows": opt_feature_rows,
                # the OPT tier's own fp8 scale (None = rows are bf16);
                # NOT the must tier's scale (Bug E: server dequantizes
                # each part with the scale it carries)
                "feature_scale": opt_quant_scale,
            }
            edge_stats["opt_payload"] = opt_payload
        return buf.getvalue(), edge_stats

    # ------------------------------------------------------------------
    # Optional VidCom2 token compression (edge mode)
    # ------------------------------------------------------------------
    @staticmethod
    def _compress_edge_bundle(embeddings, inputs, video_grid_thw, vcfg,
                              retention_ratio: float, processor=None,
                              optional_ratio: float = 0.0):
        """VidCom2-select surviving video tokens and prune row-aligned tensors.

        optional_ratio > 0 enables TWO-TIER selection: the retention
        set is split into a must tier (reliable first wave) and an
        optional tier (the least-critical tail of the retained set,
        sent best-effort in a second wave; see _build_opt_parts /
        _send_opt_parts). Per-frame optional counts are driven by the
        FRAME's own dynamic scale gap (compute_scales run twice: once
        at base=retention_ratio, once at base=retention-optional), so
        the split mirrors vidcom2's per-frame budget allocation.
        optional_ratio == 0 reproduces the single-tier path
        byte-for-byte.

        Scoring mirrors token_compressor/vidcom2/models/qwen3_vl.py
        _compute_keep_indices (select_low_var_channels ->
        compute_gaussian_scores -> compute_scales ->
        select_outlier_indices) but runs against the final edge ViT output
        (the single merged [N, 2048] segment on Qwen3.6 — one row per
        video token) instead of patching the LLM forward. The selector
        is width-agnostic (it scores rows), so the Qwen3-VL code
        carries over unchanged; only the deepstack concat (absent
        here) differs upstream.

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
        upstream vidcom2's visual_pos_masks semantics. (When the v5
        processor DOES emit mrope_positions, that tensor is pruned
        directly — the epd env's transformers 5.12.1 emits it.)

        video_grid_thw / second_per_grid_ts are left UNTOUCHED: the
        server's precomputed fast path derives item offsets by scanning
        the (pruned) input_ids for video-pad runs (base_processor.py
        get_mm_items_offset), and get_rope_index (the only grid
        consumer for this path) is skipped when mrope_positions is
        provided — the grid then merely describes the (already consumed)
        source layout. Every layout contract is validated up front and
        violated layouts raise (fail-fast client-side, never a
        scheduler-side crash).

        Returns (embeddings, inputs, compress_stats). In tiered mode
        embeddings/inputs describe the MUST tier only, and
        compress_stats additionally carries:
          must_row_ids      global row ids of the must tier (for the
                            server-side merge fallback path)
          full_row_ids      global row ids of the FULL retention set
                            (slot space base for opt parts)
          opt_row_ids       global row ids of the optional tier,
                            ORDERED most-important-first (send order)
          opt_mrope_cols    [3, n_opt] mrope columns for the opt rows,
                            aligned with opt_row_ids
          opt_score_order   permutation applied (sanity/debug)
        """
        import time as _time

        import torch

        _ensure_vidcom2_path()
        from token_compressor.vidcom2 import (
            compute_gaussian_scores,
            compute_scales,
            select_low_var_channels,
            select_outlier_indices,
            select_tiered_indices,
            _map_linear_offset,
        )

        t_start = _time.time()
        n_full = int(embeddings.shape[0])
        if n_full == 0:
            return embeddings, inputs, {"video_tokens_full": 0,
                                        "video_tokens_kept": 0}

        tiered = 0.0 < optional_ratio < retention_ratio

        # --- score & select (mirrors _compute_keep_indices) ---
        merge = getattr(vcfg, "spatial_merge_size", 2) if vcfg else 2
        t_dim, h_dim, w_dim = video_grid_thw.tolist()[0]
        frame_tokens = (h_dim * w_dim) // (merge * merge)
        sel_feat = select_low_var_channels(embeddings)
        vid_score, frame_score = compute_gaussian_scores(sel_feat, frame_tokens)
        fused_score = vid_score + frame_score
        frame_imp = -vid_score.mean(dim=-1)

        opt_row_ids = None
        opt_mrope_cols = None
        must_row_ids_extra = None
        full_row_ids_extra = None
        opt_feature_rows_extra = None
        if tiered:
            # Per-frame two-tier selection: must = a true must-level
            # vidcom2 run; opt = the retained-but-least-critical tail,
            # with per-frame counts driven by that frame's own scale
            # gap (select_tiered_indices runs compute_scales twice).
            # opt arrives in GLOBAL SEND ORDER (ascending fused score
            # = most important first), flattened across frames.
            base_must = retention_ratio - optional_ratio
            must_local, opt_flat, opt_scores = select_tiered_indices(
                fused_score, frame_imp, frame_tokens,
                retention_ratio, base_must)
            opt_frame, opt_local_flat = opt_flat[0], opt_flat[1]
            # scatter the send-ordered flat indices back to per-frame
            # lists so full_local can be built frame-sorted
            opt_by_frame: list[list[int]] = [
                [] for _ in range(fused_score.shape[0])]
            for j in range(int(opt_frame.shape[0])):
                opt_by_frame[int(opt_frame[j])].append(int(opt_local_flat[j]))
            full_local = [torch.sort(torch.cat(
                [m] + ([torch.tensor(o, dtype=m.dtype,
                                     device=m.device)] if o else []))
            ).values for m, o in zip(must_local, opt_by_frame)]
            must_idx = _map_linear_offset(must_local, frame_tokens)
            full_idx = _map_linear_offset(full_local, frame_tokens)
            # send-ordered global row ids: frame * tpf + local
            if opt_frame.numel() > 0:
                opt_row_ids = (opt_frame.to(torch.long) * frame_tokens
                               + opt_local_flat.to(torch.long))
            else:
                opt_row_ids = torch.zeros(
                    0, dtype=torch.long, device=must_idx.device)
            must_idx, _ = torch.sort(must_idx)
            full_idx, _ = torch.sort(full_idx)
            must_row_ids_extra = must_idx
            full_row_ids_extra = full_idx
            keep_idx = must_idx
            # Snapshot the OPT feature rows from the UNPRUNED bf16
            # embeddings NOW — after this point `embeddings` is
            # must-pruned (keep_idx), so indexing it with opt_row_ids
            # would read wrong rows (Bug A). These rows are the
            # second-wave payload; they never enter the must bundle.
            opt_feature_rows_extra = embeddings[
                opt_row_ids.to(embeddings.device)].detach().clone()
        else:
            scales = compute_scales(frame_imp, retention_ratio)
            indices = select_outlier_indices(fused_score, scales, frame_tokens)
            keep_idx = _map_linear_offset(indices, frame_tokens)
            keep_idx, _ = torch.sort(keep_idx)
        n_kept = int(keep_idx.shape[0])
        if n_kept == 0 or n_kept >= n_full:
            # nothing to prune (ratio ~1 or degenerate selection).
            # tiered MUST be marked False here: this early-exit skips
            # the opt_payload assembly entirely, so "tiered" in stats
            # would otherwise key a two-tier flow that never happens
            # (must bundle only, no UDP wave) — the run is single-tier
            # even though optional_ratio > 0 was requested.
            return embeddings, inputs, {
                "video_tokens_full": n_full, "video_tokens_kept": n_full,
                "vidcom2_active": False, "tiered": False,
                "vidcom2_time": _time.time() - t_start,
            }

        # --- per-frame pad runs in the expanded ids (fail-fast) ---
        video_token_id = EncoderPlacementQwen36Client._video_pad_token_id(
            inputs, processor)
        runs = EncoderPlacementQwen36Client._video_pad_runs(inputs, video_token_id)
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
            mrope = EncoderPlacementQwen36Client._compute_full_mrope(
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
            # BUG S: ship the resolved <|video_pad|> id with the tiered
            # bundle — the server's merge needs it to emit opt-token
            # columns, and its longest-run heuristic can misresolve it
            # on adversarial/edge-case prompts (e.g. a long repeated
            # text token). The client KNOWS the id; don't make the
            # server guess.
            "video_pad_id": int(video_token_id),
        }
        if tiered:
            # NOTE: opt rows were snapshotted from the UNPRUNED bf16
            # embeddings BEFORE keep_idx pruning (opt_feature_rows at
            # the call site) — indexing the pruned tensor here would
            # read MUST-tier rows (wrong values / IndexError).
            # opt rows' mrope columns, aligned with opt_row_ids order.
            # Slot space: the retention-level row id (position within
            # full_row_ids) — parts are self-describing ("I am slot s
            # of the retention sequence"), so the server merges by slot
            # without any schema dependency.
            slot_of = torch.zeros(n_full, dtype=torch.long,
                                  device=full_idx.device)
            slot_of[full_idx] = torch.arange(
                full_idx.shape[0], device=full_idx.device)
            opt_slots = slot_of[opt_row_ids.to(full_idx.device)]
            # mrope column for row r = the column inside r's frame pad
            # run at offset (r - run_start). runs[] is validated above.
            # Build a row->column LUT from the runs (robust to
            # send-order opt_row_ids; the old forward-scan silently
            # skipped out-of-order rows).
            row2col = torch.empty(n_full, dtype=torch.long)
            lut_row = 0
            for lo, hi in runs:
                run_len = hi - lo
                row2col[lut_row:lut_row + run_len] = torch.arange(
                    lo, hi, dtype=torch.long)
                lut_row += run_len
            opt_col_idx = row2col[
                opt_row_ids.to(row2col.device)].to(mp.device)
            if int(opt_col_idx.shape[0]) != int(opt_row_ids.shape[0]):
                raise RuntimeError(
                    f"vidcom2 tiered: row->col mapping missed tokens "
                    f"({opt_col_idx.shape[0]} != {opt_row_ids.shape[0]})")
            opt_mrope = mp[..., opt_col_idx]              # [3, n_opt]
            # Insertion positions in the MUST-PRUNED sequence: for the
            # opt token at full-sequence column c, the number of kept
            # (must) columns strictly before c. The server inserts
            # arriving tokens at exactly these positions — unambiguous,
            # order-independent, and re-sorting by ins_pos reproduces
            # the retention-level sequence byte-for-byte.
            # keep_cols[:c].sum() == cumsum shifted by one: vectorized,
            # and computed per column via the row->col LUT.
            keep_cols_cpu = keep_cols.cpu()
            kept_before = torch.zeros(seq_len + 1, dtype=torch.long)
            kept_before[1:] = torch.cumsum(keep_cols_cpu, dim=0)
            opt_ins_pos = kept_before[
                opt_col_idx.cpu()].to(mp.device)
            compress_stats.update({
                "tiered": True,
                "optional_ratio": float(optional_ratio),
                "video_tokens_must": int(n_kept),
                "video_tokens_optional": int(opt_row_ids.shape[0]),
                "video_tokens_retained": int(full_idx.shape[0]),
                "must_row_ids": must_row_ids_extra.cpu(),
                "full_row_ids": full_row_ids_extra.cpu(),
                "opt_row_ids": opt_row_ids.cpu(),
                "opt_slots": opt_slots.cpu(),
                "opt_mrope_cols": opt_mrope.cpu(),
                "opt_ins_pos": opt_ins_pos.cpu(),
                "opt_feature_rows": opt_feature_rows_extra.cpu(),
            })
        return embeddings, inputs, compress_stats

    @staticmethod
    def _video_pad_token_id(inputs, processor=None) -> int:
        """Resolve the <|video_pad|> token id for run scanning.

        Preference order: processor.video_token_id (v5 processor attr),
        the bundle's own token id from an existing pad run, then
        processor.tokenizer lookups. Raises if nothing resolves —
        pruning without the pad id would silently corrupt the bundle.
        (Qwen3.6's video_pad id 248057 differs from Qwen3-VL's, which
        is exactly why the id is resolved and never hardcoded.)
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
        # (2026-10-20 04:24 matrix, cell_edge 14/14 users). Use the
        # upload read budget for BOTH slots: it is derived from the
        # slowest profile's transfer time in the first place, and a
        # loopback TCP handshake completes well within it. Fallback to
        # the connect slot when the upload budget is disabled; 0/None
        # in both slots disables the timeout entirely.
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
        print(f"[EncPl36] upload: {upload_time:.3f} s | transferred: "
              f"{len(data) / 1e6:.2f} MB payload, "
              f"{(info.get('wire_bytes') or 0) / 1e6:.2f} MB on wire")
        return info

    # ------------------------------------------------------------------
    # Tiered mode: optional-tier best-effort parts (second wave)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_opt_parts(opt_payload: dict, bundle_id: str,
                         tokens_per_packet: int = 64) -> list[bytes]:
        """Pack the optional tier into self-describing best-effort parts.

        Part = torch.save'd dict {ref, seq, slots, ins_pos, feature,
        mrope_cols} where each element is one token's worth of data:
          slots      [n]  retention-sequence slot ids ("I am token s")
          ins_pos    [n]  insertion position in the MUST-PRUNED
                          input_ids (server inserts columns there)
          feature    [n, C] rows — fp8 + feature_scale in EVERY part
                          when the run used bundle_quant=fp8 (the OPT
                          tier's OWN scale, not the must tier's), else
                          plain bf16 rows
          mrope_cols [3, n] the token's mrope columns
        input_ids columns are NOT shipped — every video-token column is
        the same video_pad id the server already knows from the must
        bundle; the server fills the constant on merge.

        Parts are ordered most-important-first (opt_payload tensors
        are already in send order). tokens_per_packet only changes the
        PACKAGING granularity — the server merges by slot and never
        sees packet boundaries, so any value works with the same merge
        code. The final short part carries the remainder.
        """
        import io as _io

        import torch

        n_opt = int(opt_payload["opt_slots"].shape[0])
        if n_opt == 0:
            return []
        tokens_per_packet = max(1, int(tokens_per_packet))
        slots = opt_payload["opt_slots"]
        ins_pos = opt_payload["opt_ins_pos"]
        feature = opt_payload["opt_feature_rows"]
        mrope = opt_payload["opt_mrope_cols"]
        scale = opt_payload.get("feature_scale")
        parts = []
        for seq, lo in enumerate(range(0, n_opt, tokens_per_packet)):
            hi = min(lo + tokens_per_packet, n_opt)
            part = {
                "format": "precomputed_embedding_part",
                "ref": bundle_id,
                "seq": seq,
                "slots": slots[lo:hi].clone(),
                "ins_pos": ins_pos[lo:hi].clone(),
                "feature": feature[lo:hi].clone(),
                "mrope_cols": mrope[:, lo:hi].clone(),
            }
            if scale is not None:
                part["feature_scale"] = float(scale)
            buf = _io.BytesIO()
            torch.save(part, buf)
            parts.append(buf.getvalue())
        return parts

    def _send_opt_parts(self, parts: list[bytes], bundle_id: str,
                        udp_port: int, send_gap_ms: float = 0.05) -> dict:
        """Send the optional-tier parts over UDP (E2U datagrams).

        SERIAL, synchronous, single-threaded — called between the must
        PUT and the chat POST:
            must PUT (TCP) -> this UDP wave -> sleep(opt_chat_delay_ms)
            -> chat POST -> SSE stream.

        Best-effort by contract on the WIRE (datagrams may be lost /
        reordered; a part missing any fragment is dropped server-side,
        never retransmitted), but a LOCAL sendto failure is a loud
        hard error: it means the OS refused the datagram (e.g. socket
        gone, EMSGSIZE) and the run is broken, so it raises (the turn
        aborts BEFORE the chat POST; a silent local failure would look
        like network loss in the measurements — a measurement bug).
        A 2s socket timeout guards the same contract: a sendto that
        would block forever (e.g. full ARP/neighbor table on a dead
        route) is a broken run, not network loss.

        Socket buffer: SO_SNDBUF is raised to 4 MiB so the kernel can
        queue the whole wave; the ACTUAL granted size (after sysctl
        clamping) is reported in the stats as opt_sndbuf — a value far
        below the request means the OS silently clamped it and blast
        runs may drop datagrams locally.

        send_gap_ms spaces ADJACENT datagrams (not after the last one,
        so it never stacks with opt_chat_delay_ms); parts are sent
        most-important-first, so the arrived set at the server's merge
        snapshot is the best possible prefix. Default 0.05 ms (~20k
        datagrams/s) paces the wave below the receiver's reassembly
        rate; 0 = blast (only safe with a raised receiver SO_RCVBUF
        and for small waves).

        Returns stats: parts_sent / parts_bytes / fragments_sent /
        opt_send_time / opt_send_gap_ms / opt_udp_port / opt_sndbuf.
        """
        host = self._udp_host()
        addr = (host, int(udp_port))
        fragments_sent = 0
        total_bytes = 0
        t_start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            try:
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            except OSError:
                pass  # non-fatal: report the actual buffer below
            try:
                sndbuf = int(sock.getsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF))
            except OSError:
                sndbuf = 0
            sock.settimeout(2.0)
            n_parts = len(parts)
            for seq, data in enumerate(parts):
                frags = _e2u_fragments(bundle_id, seq, data)
                n_frags = len(frags)
                for frag_idx, datagram in enumerate(frags):
                    try:
                        sock.sendto(datagram, addr)
                    except OSError as exc:
                        raise RuntimeError(
                            f"E2U sendto failed (part_seq={seq} "
                            f"frag={frag_idx}/{n_frags} "
                            f"dst={host}:{udp_port}): {exc}") from exc
                    fragments_sent += 1
                    # BUG T (round-7 regression V): pace every ADJACENT
                    # pair of datagrams — INCLUDING across part
                    # boundaries — skipping only the wave's very LAST
                    # fragment (its sleep would stack with
                    # opt_chat_delay_ms, the next section's sleep, and
                    # silently inflate TTFT by one gap). Single-fragment
                    # parts MUST still be paced: a wave of them is a
                    # blast otherwise, violating the send_gap_ms
                    # contract ("0 = blast" is the ONLY blast switch).
                    is_wave_tail = (seq == n_parts - 1
                                    and frag_idx == n_frags - 1)
                    if send_gap_ms > 0 and not is_wave_tail:
                        time.sleep(send_gap_ms / 1000.0)
                total_bytes += len(data)
        finally:
            sock.close()
        return {
            "parts_sent": len(parts),
            "parts_bytes": total_bytes,
            "opt_fragments_sent": fragments_sent,
            "opt_send_time": time.time() - t_start,
            "opt_send_gap_ms": float(send_gap_ms),
            "opt_udp_port": int(udp_port),
            "opt_sndbuf": sndbuf,
        }

    def _udp_host(self) -> str:
        """UDP target host = the HTTP base_url's hostname."""
        parsed = self.base_url.split("://", 1)[-1]
        return parsed.split("/", 1)[0].split(":", 1)[0]

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
            "format": "encoder_placement_qwen36.chat.completions.create",
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
        # with "Read timed out." (2026-10-20 04:24 matrix, no_emulation
        # and 5g_urban partial failures). The idle budget is applied by
        # _iter_lines_idle_guarded only AFTER the first line has landed
        # (see below); until then the socket keeps the first-event
        # read timeout set above.
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
                    "format": "encoder_placement_qwen36.chat.completions.engine_stats",
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
                "format": "encoder_placement_qwen36.chat.completions.chunk",
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
                "format": "encoder_placement_qwen36.chat.completions.engine_stats",
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
        optional_ratio: float = 0.0,
        opt_tokens_per_packet: int = 64,
        opt_send_gap_ms: float = 0.05,
        opt_chat_delay_ms: float = 0.0,
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
        optional_ratio (edge + vidcom2 only, 0 <= optional < retention):
                splits the retention set into must (this request's
                bundle, reliable) and optional (least-critical tail,
                streamed best-effort over UDP in
                opt_tokens_per_packet-token parts BETWEEN the must PUT
                and the chat POST — serial timing; parts that miss the
                server's merge snapshot are dropped and never
                retransmitted). 0 (default) = legacy single-tier path.
        opt_tokens_per_packet (edge + tiered only, default 64): how
                many optional tokens are packed into one best-effort
                part. Pure packaging knob — the server merges by slot
                and never sees packet boundaries.
        opt_send_gap_ms (edge + tiered only, default 0.05): sleep between
                ADJACENT UDP datagrams of the optional wave (paces the
                second wave below the receiver's reassembly rate; 0 =
                blast, only safe with a raised receiver SO_RCVBUF and
                small waves).
        opt_chat_delay_ms (edge + tiered only, default 0): sleep AFTER
                the UDP wave completes and BEFORE the chat POST —
                the dial that trades opt-token arrival rate against
                TTFT (the server snapshots at chat-arrival, so a longer
                delay lets more parts land but delays prefill).
        bundle_quant="fp8" (edge only, requires compressor="vidcom2")
                quantizes the pruned feature tensor to fp8 e4m3 with a
                per-tensor scale before upload (halves feature bytes);
                '' (default) uploads the raw bf16 feature.
        vision_only=True (edge only) loads ONLY the vision tower
                (~0.5B params: ViT + merger, no deepstack on Qwen3.6)
                instead of the full 35B fp8 model just to grab
                model.model.visual — same class, same weights,
                identical embeddings; cuts edge VRAM from ~35 GB to
                ~1 GB. Strongly recommended on the 35B-A3B checkpoint.
                False keeps the historical full load. Ignored in
                server mode.
        attn_impl (edge only): attention backend for the edge ViT —
                'sdpa' (default), 'flash_attention_4', or 'eager'.
                Ignored in server mode.
        user_tag: optional per-user marker (see _build_templated_text);
                "" keeps the historical video-first order.
        """
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

        print(f"[EncPl36] mode={mode}"
              + (f" frame_ratio={frame_ratio}" if frame_ratio else "")
              + (f" max_frames={max_frames}" if max_frames != 768 else "")
              + (f" compressor={compressor}"
                 f" retention_ratio={retention_ratio}"
                 + (f" bundle_quant={bundle_quant}"
                    if bundle_quant else "")
                 if (compressor and mode == "edge") else "")
              + (f" optional_ratio={optional_ratio}"
                 if (optional_ratio and mode == "edge") else "")
              + (f" vision_only=True" if (vision_only
                                          and mode == "edge") else "")
              + (f" attn_impl={attn_impl}" if (attn_impl != "sdpa"
                                                and mode == "edge") else ""))

        # Both modes render the same chat-templated text (identical
        # markup + video placeholder), keeping prefill symmetric. The
        # per-user tag rides inside the text part (see
        # _build_templated_text).
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
                optional_ratio=optional_ratio if mode == "edge" else 0.0,
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

        # Tiered mode: SERIAL second wave over UDP, between the must
        # PUT and the chat POST (single-threaded straight-line):
        #   must PUT (TCP) -> UDP datagram wave (send_gap_ms-paced)
        #   -> sleep(opt_chat_delay_ms) -> chat POST -> SSE.
        # The server snapshots arrived parts when the chat request's
        # merge runs; parts that have not landed by then are dropped
        # and never retransmitted (best-effort contract). A local
        # sendto failure RAISES (fail-fast: the turn aborts before the
        # chat POST rather than silently mislabeling network loss); a
        # server that did not open its UDP port reports opt_udp_port=0
        # in the must PUT response, which also raises here.
        opt_send_stats: dict = {}
        opt_payload = edge_stats.pop("opt_payload", None) if mode == "edge" else None
        # wall time of the whole UDP-wave section (pack + send + chat
        # delay), for the reconcile breakdown: e2e_ttft should equal
        # stage1 + stage2 + upload + udp_wave + server_ttft + riders +
        # t_prepare + other.
        t_udp_wave_start = time.time()
        udp_wave_time = 0.0
        if opt_payload is not None:
            bundle_id = upload_meta["id"]
            parts = self._build_opt_parts(
                opt_payload, bundle_id,
                tokens_per_packet=opt_tokens_per_packet)
            if parts:
                udp_port = int(upload_meta.get("opt_udp_port") or 0)
                if udp_port <= 0:
                    raise RuntimeError(
                        f"tiered edge run but the server reported no "
                        f"optional-tier UDP port (opt_udp_port="
                        f"{udp_port}); restart the sidecar with "
                        f"--opt-udp-port > 0 (e.g. 8005) before "
                        f"running optional_ratio > 0")
                opt_send_stats = self._send_opt_parts(
                    parts, bundle_id, udp_port,
                    send_gap_ms=opt_send_gap_ms)
                if opt_chat_delay_ms > 0:
                    time.sleep(opt_chat_delay_ms / 1000.0)
            udp_wave_time = time.time() - t_udp_wave_start

        streaming_response = self._stream_completion(
            messages, model=model, mode=mode,
            max_frames=max_frames, max_tokens=max_tokens,
            frame_ratio=frame_ratio,
        )

        # Serial timing: the UDP wave is already complete before the
        # chat POST, so there is nothing to join — fold the stats in.
        if opt_send_stats:
            opt_send_stats.update({
                "opt_tokens": int(opt_payload["opt_slots"].shape[0]),
                "opt_tokens_per_packet": int(opt_tokens_per_packet),
                "opt_chat_delay_ms": float(opt_chat_delay_ms),
            })
            edge_stats.update(opt_send_stats)

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
            # UDP second-wave section wall time (tiered only; 0.0 for
            # single-tier): part packing + paced sendto wave + optional
            # pre-chat sleep — the reconcile's udp_wave component.
            "udp_wave_time": udp_wave_time if mode == "edge" else 0.0,
            "vision_only": bool(vision_only) if mode == "edge" else False,
            "attn_impl": attn_impl if mode == "edge" else "sdpa",
            "frame_ratio": float(frame_ratio) if frame_ratio else None,
            "max_frames": int(max_frames) if max_frames else None,
            # Frame-sampling semantics version: lets the bench's resume
            # guard refuse results files written under a different
            # sampling semantics (same ratio value, different frames).
            # Kept in sync with tools/encpl_bench_eval_qwen36.py
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
            print(f"[EncPl36] E2E TTFT (video input -> first token): {e2e_ttft:.2f} s")

        if mode == "edge":
            stage1 = edge_stats.get("stage1_edge") or 0.0
            stage2 = edge_stats.get("stage2_vit") or 0.0
            upload_time = upload_meta.get("upload_time") or 0.0
            server_ttft = server_stats.get("server_ttft") or 0.0
            # UDP second-wave section (tiered only): part packing +
            # paced sendto + optional pre-chat sleep. ~0 for single-tier.
            udp_wave = udp_wave_time
            print(f"[EncPl36] mode=edge | stage1: {stage1:.2f} s | "
                  f"stage2 (ViT): {stage2:.2f} s | upload: {upload_time:.2f} s | "
                  f"udp_wave: {udp_wave:.2f} s | "
                  f"server_ttft: {server_ttft:.2f} s | "
                  f"sum: {stage1 + stage2 + upload_time + udp_wave + server_ttft:.2f} s")
            stage1_bd = edge_stats.get("stage1_breakdown") or {}
            if stage1_bd:
                print(f"[EncPl36]   stage1 breakdown: "
                      f"open={stage1_bd.get('open', 0):.3f}s | "
                      f"decode_sample={stage1_bd.get('decode_sample', 0):.3f}s | "
                      f"processor={stage1_bd.get('processor', 0):.3f}s")
            stage2_bd = edge_stats.get("stage2_breakdown") or {}
            if stage2_bd:
                print(f"[EncPl36]   stage2 breakdown: "
                      f"h2d={stage2_bd.get('h2d', 0):.3f}s | "
                      f"vit_fwd={stage2_bd.get('vit_fwd', 0):.3f}s | "
                      f"assemble={stage2_bd.get('assemble', 0):.3f}s")
            if edge_stats.get("sampled_nframes") is not None:
                fps_part = (f" fps={edge_stats['video_fps']:.2f}"
                            if edge_stats.get("video_fps") else "")
                print(f"[EncPl36]   frames: "
                      f"total={edge_stats.get('total_frames')}{fps_part} -> "
                      f"sampled={edge_stats['sampled_nframes']} "
                      f"(frame_ratio={frame_ratio or 'off'}, "
                      f"max_frames={max_frames or 'default'})")
            if edge_stats.get("vidcom2_active"):
                print(f"[EncPl36]   vidcom2: kept "
                      f"{edge_stats.get('video_tokens_kept')}/"
                      f"{edge_stats.get('video_tokens_full')} video tokens "
                      f"(retention={edge_stats.get('retention_ratio_actual', 0):.3f}, "
                      f"{edge_stats.get('vidcom2_time', 0):.3f}s)")
            if edge_stats.get("quant_dtype"):
                print(f"[EncPl36]   bundle_quant: {edge_stats['quant_dtype']} "
                      f"(scale={edge_stats.get('feature_scale')}, "
                      f"{edge_stats.get('quant_time', 0):.3f}s)")
            if edge_stats.get("tiered"):
                # Tiered selection breakdown (client-side ground truth):
                # must tokens ship in the TCP bundle, optional tokens
                # stream over the UDP wave. Degenerate early-exits set
                # tiered=False, so this block stays silent for them and
                # for single-tier runs.
                print(f"[EncPl36]   tiered: must="
                      f"{edge_stats.get('video_tokens_must')} opt="
                      f"{edge_stats.get('video_tokens_optional')} tokens "
                      f"(retained={edge_stats.get('video_tokens_retained')}, "
                      f"optional_ratio="
                      f"{edge_stats.get('optional_ratio', 0):.3f})")
                if edge_stats.get("parts_sent") is not None:
                    print(f"[EncPl36]   tiered wave: parts="
                          f"{edge_stats.get('parts_sent')} frags="
                          f"{edge_stats.get('opt_fragments_sent')} "
                          f"{edge_stats.get('parts_bytes', 0)}B in "
                          f"{edge_stats.get('opt_send_time', 0):.3f}s "
                          f"(gap={edge_stats.get('opt_send_gap_ms', 0):.2f}ms "
                          f"tpp={edge_stats.get('opt_tokens_per_packet')} "
                          f"delay={edge_stats.get('opt_chat_delay_ms', 0):.1f}ms, "
                          f"port={edge_stats.get('opt_udp_port')})")
                # Server-side merge snapshot (SSE trailer engine_stats;
                # absent on older servers -> skip the line entirely).
                merged = server_stats.get("opt_tokens_merged")
                if merged is not None:
                    expected = server_stats.get("opt_tokens_expected")
                    expected_str = (str(expected) if expected is not None
                                    else "?")
                    dup_str = (
                        f", dup_dropped="
                        f"{server_stats['parts_slot_dup_dropped']}"
                        if "parts_slot_dup_dropped" in server_stats else "")
                    print(f"[EncPl36]   tiered merge: {merged}/{expected_str} "
                          f"opt tokens (late_dropped="
                          f"{server_stats.get('parts_late_dropped', 0)}, "
                          f"parts_merged="
                          f"{server_stats.get('parts_merged', 0)}{dup_str}, "
                          f"merge={server_stats.get('merge_time', 0):.3f}s)")
            e2e_ttft_val = stats.get("e2e_ttft")
            if e2e_ttft_val is not None:
                gap = (e2e_ttft_val
                        - stage1 - stage2 - upload_time - udp_wave
                        - server_ttft)
                # edge_prep_time covers _prepare_edge_bundle only
                # (decode..torch.save); file read / templated text /
                # model load sit outside it and land in `other`.
                edge_riders = ((edge_stats.get("edge_prep_time") or 0.0)
                                - stage1 - stage2)
                t_prepare = server_stats.get("t_prepare") or 0.0
                other = gap - edge_riders - t_prepare
                print(f"[EncPl36]   reconcile: e2e_ttft={e2e_ttft_val:.2f}s "
                      f"sum={stage1 + stage2 + upload_time + udp_wave + server_ttft:.2f}s "
                      f"gap={gap:.2f}s | "
                      f"udp_wave={udp_wave:.2f}s (pack+send+chat_delay) | "
                      f"edge_riders={edge_riders:.2f}s "
                      f"(vidcom2/quant/drop_est/d2h/ser) | "
                      f"t_prepare={t_prepare:.2f}s (server torch.load+dequant+drain+merge) | "
                      f"other={other:.2f}s "
                      f"(file read+templated text+model load+POST RTT+SSE down)")
        else:
            stage1 = server_stats.get("t_prepare") or 0.0
            server_ttft = server_stats.get("server_ttft") or 0.0
            upload_time = upload_meta.get("upload_time") or 0.0
            print(f"[EncPl36] mode=server | upload: {upload_time:.2f} s | "
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
