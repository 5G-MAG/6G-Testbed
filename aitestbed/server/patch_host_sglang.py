#!/usr/bin/env python3
"""
patch_host_sglang.py — one-shot, idempotent source patch for the ACTIVE
environment's sglang installation (site-packages copy).

Why: the SGLang engine sidecar spawns scheduler child processes
(multiprocessing "spawn"), which re-import sglang from the active
env's site-packages. Monkey-patches applied in the parent (server)
process never reach them. The only robust place for the
encoder-placement (edge mode) fix is on disk, inside the env sglang
itself imports.

What (patch 1 — sglang 0.5.20 qwen3_vl.py routes PRECOMPUTED_EMBEDDING
mm items (edge-mode ViT-output bundles) straight into the visual tower,
which crashes on the positional-embedding add (token rows != patch
rows: "The size of tensor a (28672) must match the size of tensor b
(43008)"). qwen2_5_vl.py:792, minicpmv.py:944, internvl.py:642 and
cosmos3_edge.py:349 all guard this case; qwen3_vl.py is missing it.

The patch (same as the reference copy under share/sglang-0.5.20):
  1) import MultimodalInputFormat in the schedule_batch import block
  2) _get_visual_feature: PRECOMPUTED_EMBEDDING items bypass the
     visual tower; cat(item.feature) -> [tokens, hidden] on the
     visual tower's device/dtype (Qwen3VL is in the
     _can_skip_pre_embed_feature_move allowlist, so the scheduler does
     NOT pre-move features — the model-side .to() is required).

What (patch 3 — srt/utils/video_decoder.py: VideoDecoderWrapper's
  decode parallelism is sized by the HOST core count and is NOT covered
  by OMP/torch/mm-worker knobs. The patch adds the
  SGLANG_DECORD_NUM_THREADS env knob (name historical — covers BOTH
  backends; read at VideoDecoderWrapper construction time):
  - torchcodec (the PREFERRED backend when importable — e.g.
     torchcodec==0.15.0 in the epd env): clamps
     self._num_decode_threads, bounding the parallel VideoDecoder
     instance count in get_frames_as_tensor (native auto =
     min(cpu_count, 16) instances; env wins over the ctor arg, which
     the EncPl paths never pass);
  - decord (fallback backend): num_threads= forwarded to every
     VideoReader construction (decord's internal thread pool).
   unset/0/invalid = native default, bit-identical. Set via the
   sidecars' --decode-threads flag or the edge runner's decode_threads
   config/env — see the encoder_placement scenarios.)

What (patch 4 — mm processors/qwen_vl.py: upstream 0.5.20 has
  preprocess_video as `async def` with a ZERO-await body, so the heavy
  per-video work (get_frames_as_tensor decode, torchvision resize,
  pin_memory) runs ON the tokenizer-manager event loop; the call site
  (_process_mm_data_uncached) is a sequential `await` list-comprehension,
  so N concurrent video requests serialize into ONE decode staircase
  (measured: 6 users, heterogeneous videos, TTFT pinned to the staircase
  end under both --max-running-requests 2 and 6, with and without
  --enable-mixed-chunk). glm4v.py:570-607 and mimo_v2.py:851 are the
  upstream-correct precedents (run_in_executor + gather / submit).
  The patch splits preprocess_video into an async shell + module-level
  _preprocess_video_sync (body moved VERBATIM, so the patch-2
  frame_sampling traces land in the sync half unchanged) and rewrites
  the call site to fan videos out over the processor's io_executor via
  asyncio.gather(loop.run_in_executor(...)). DYNAMIC SWITCH, default
  OFF (= upstream serial behavior, bit-identical):
  - module global _PARALLEL_VIDEO_PREPROCESS, initialized from env
    SGLANG_QWEN_PARALLEL_VIDEO_PREPROCESS at import; read at EVERY call,
    so flipping the global at runtime takes effect on the next request
    without an engine restart;
  - the concurrent sidecar flips it via POST /video_preprocess_mode and
    echoes it in /stats (startup flag --parallel-video-preprocess sets
    the env default before the engine imports the module).
  Width when ON = the existing --mm-io-worker-num pool (io_executor),
  so the knob semantics stay "parallel video decode lanes".)

What (patch 2 — mm processors/qwen_vl.py smart_nframes has no
"fraction of total frames" mode: the fps branch caps at max_frames but
always targets FPS (2.0) x duration, so a percentage-sampling
experiment is impossible. The patch adds a frame_ratio branch
mirroring the fps branch's clamping structure (floor-by FRAME_FACTOR,
min FPS_MIN_FRAMES=4, cap by max_frames and total_frames) and
registers "frame_ratio" in QWEN_VIDEO_PREPROCESS_CONFIG_KEYS so it
flows through mm_process_config -> BaseMultimodalProcessor.video_config
-> preprocess_video -> smart_nframes. The edge client
(clients/encoder_placement_client.py) ALSO calls smart_nframes
directly with video_config={"frame_ratio": ...} — the same patched
function, so both arms sample identically. NOTE: max_frames alone
needs NO patch — it is a native sglang 0.5.20 whitelist key consumed
by the fps branch (qwen_vl.py:230-232); only frame_ratio requires
patch 2.)

Safety:
  - refuses to patch unless both exact anchors match the pristine
    0.5.20 layout (any mismatch -> exit 1, file untouched);
  - marker comments make re-runs no-ops (independently per file);
  - one-time backup <file>.bak-<tag> (first successful patch only);
  - results are ast.parse-verified and written atomically (os.replace).

Run inside the target conda env (start_sidecar.sh does this
automatically before launching the sidecar):

    python aitestbed/server/patch_host_sglang.py

Exit codes: 0 = patched or already patched; 1 = any failure.
"""

import ast
import os
import sys
import tempfile
from importlib.metadata import distribution
from pathlib import Path

# ---------------------------------------------------------------------------
# Patch 1: qwen3_vl.py PRECOMPUTED_EMBEDDING guard
# ---------------------------------------------------------------------------
MARKER = "# 6G-Testbed: precomputed-embedding guard (edge mode)"
TARGET_1 = "srt/models/qwen3_vl.py"

# Anchor 1: schedule_batch import block (pristine sglang 0.5.20).
ANCHOR_IMPORT_OLD = """\
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
"""
ANCHOR_IMPORT_NEW = """\
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalInputs,
)
"""

# Anchor 2: head of _get_visual_feature (pristine sglang 0.5.20).
# The 'assert' line is kept so the guard is provably inserted at the
# method head, before every use of grid_thw / self.visual.
ANCHOR_METHOD_OLD = """\
    def _get_visual_feature(
        self, items: List[MultimodalDataItem], grid_thw: torch.Tensor
    ) -> torch.Tensor:
        assert grid_thw.dim() == 2, grid_thw.dim()
"""
ANCHOR_METHOD_NEW = """\
    def _get_visual_feature(
        self, items: List[MultimodalDataItem], grid_thw: torch.Tensor
    ) -> torch.Tensor:
        {marker}
        if items and all(
            item.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING
            for item in items
        ):
            # Final visual embeddings computed off-engine (edge mode): bypass
            # the visual tower entirely, mirroring the qwen2_5_vl guard
            # (qwen2_5_vl.py:792) and cosmos3_edge._get_visual_feature.
            # Qwen3VL is in the _can_skip_pre_embed_feature_move allowlist,
            # so the scheduler does not pre-move features: move here, to the
            # tower's device and dtype, exactly once.
            feature = torch.cat(
                [
                    item.feature
                    if isinstance(item.feature, torch.Tensor)
                    else torch.as_tensor(item.feature)
                    for item in items
                ],
                dim=0,
            )
            return feature.reshape(-1, feature.shape[-1]).to(
                device=self.visual.device, dtype=self.visual.dtype
            )
        assert grid_thw.dim() == 2, grid_thw.dim()
""".format(marker=MARKER)

# ---------------------------------------------------------------------------
# Patch 2: mm processors/qwen_vl.py frame_ratio sampling
# ---------------------------------------------------------------------------
MARKER_2 = "# 6G-Testbed: frame_ratio sampling"
TARGET_2 = "srt/multimodal/processors/qwen_vl.py"

# Anchor 2a: QWEN_VIDEO_PREPROCESS_CONFIG_KEYS frozenset (pristine 0.5.20).
# Adding "frame_ratio" lets the key ride mm_process_config["video"] ->
# BaseMultimodalProcessor.video_config instead of being stripped by
# _get_processor_video_config (which drops every listed key from the
# processor-facing config — precisely the whitelist semantics we want:
# smart_nframes consumes it, the HF processor never sees it).
# NOTE: "max_frames" is ALREADY in the pristine whitelist — the max_frames
# pass-through needs no patch, only frame_ratio does.
ANCHOR_KEYS_OLD = """\
QWEN_VIDEO_PREPROCESS_CONFIG_KEYS = frozenset(
    {
        "fps",
        "nframes",
        "min_frames",
        "max_frames",
        "min_pixels",
        "max_pixels",
        "total_pixels",
        "resized_height",
        "resized_width",
    }
)
"""
ANCHOR_KEYS_NEW = """\
QWEN_VIDEO_PREPROCESS_CONFIG_KEYS = frozenset(
    {
        "fps",
        "nframes",
        "min_frames",
        "max_frames",
        "frame_ratio",
        "min_pixels",
        "max_pixels",
        "total_pixels",
        "resized_height",
        "resized_width",
    }
)
"""

# Anchor 2b: the fps-branch of smart_nframes (pristine 0.5.20). The
# frame_ratio branch is inserted BEFORE the fps fallback. SEMANTICS v3:
# ratio multiplies sglang's OWN base frame count (total/video_fps*fps,
# i.e. the 2fps x duration figure BEFORE any clamping), then the
# product runs through the fps branch's EXACT clamp chain (min ->
# cap -> total -> floor-by-2) — so the percentage always reads
# "fraction of sglang's native sampling", and the native safeguards
# (min 4, cap 768, <= total) keep applying afterwards. v1/v2 instead
# multiplied ratio by total_frames; both are deprecated (v3 has its
# own marker, and an installed v2 layout is upgraded in place).
# logger.info trace: every frame-count transition on every video.
ANCHOR_NFRAMES_OLD = """\
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
"""
ANCHOR_NFRAMES_NEW = """\
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    elif "frame_ratio" in ele:
        # __MARKER__
        # Ratio-of-native sampling (v3): multiply sglang's OWN base
        # frame count (total_frames / video_fps * fps, the 2fps x
        # duration figure BEFORE clamping) by frame_ratio, THEN run the
        # fps branch's exact clamp chain — the ratio always means
        # "fraction of sglang's native sampling", never "fraction of
        # total_frames" (v1/v2 semantics, deprecated).
        frame_ratio = ele["frame_ratio"]
        if not (0.0 < frame_ratio <= 1.0):
            raise ValueError(
                f"frame_ratio must be in (0, 1], but got {frame_ratio}."
            )
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        base = total_frames / video_fps * fps
        nframes = base * frame_ratio
        logger.info(
            "frame_sampling: total_frames=%s video_fps=%s fps=%s base=%s "
            "frame_ratio=%s -> raw=%s (clamping: min=%s max=%s total=%s)",
            total_frames, video_fps, fps, base, frame_ratio, nframes,
            min_frames, max_frames, total_frames,
        )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
        logger.info(
            "frame_sampling: final nframes=%s (ratio-of-native; "
            "unique linspace indices may lower it further)",
            nframes,
        )
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
""".replace("__MARKER__", MARKER_2)

# The DEPRECATED v2 frame_ratio branch (ratio x total_frames) — kept
# verbatim so _apply_patch can recognize and upgrade an installed v2
# layout (the GPU box currently runs it) to the v3 semantics above.
ANCHOR_NFRAMES_V2 = """\
    elif "frame_ratio" in ele:
        # __MARKER__
        # Percentage sampling: keep floor(total_frames * frame_ratio)
        # frames, mirroring the fps branch's clamp structure so results
        # stay comparable across sampling modes. frame_ratio=1.0 reduces
        # to "all frames" (capped by max_frames / total_frames).
        frame_ratio = ele["frame_ratio"]
        if not (0.0 < frame_ratio <= 1.0):
            raise ValueError(
                f"frame_ratio must be in (0, 1], but got {frame_ratio}."
            )
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames * frame_ratio
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
""".replace("__MARKER__", MARKER_2)

# Anchor 2c: preprocess_video's sampling block (pristine 0.5.20). The
# replacement logs total_frames, the smart_nframes result, and the
# FINAL decoded frame count after np.unique (duplicate indices can
# lower it: linspace over small totals) — the full observable
# transition chain for EVERY video, both sampling modes, both arms.
ANCHOR_TRACE_OLD = """\
    nframes = smart_nframes(
        video_config, total_frames=total_frames, video_fps=video_fps
    )
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)
"""
ANCHOR_TRACE_NEW = """\
    nframes = smart_nframes(
        video_config, total_frames=total_frames, video_fps=video_fps
    )
    logger.info(
        "frame_sampling: total_frames=%s video_fps=%s sampled_nframes=%s",
        total_frames, video_fps, nframes,
    )
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)
    if len(idx) != nframes:
        logger.info(
            "frame_sampling: unique indices lowered nframes %s -> %s",
            nframes, len(idx),
        )
"""


# ---------------------------------------------------------------------------
# Patch 3: utils/video_decoder.py decode thread cap (BOTH backends)
# ---------------------------------------------------------------------------
# Rationale: VideoDecoderWrapper's decode parallelism is NOT covered by
# OMP/MKL/OPENBLAS_NUM_THREADS, torch.set_num_threads, or the
# mm_io_worker_num / mm_processor_worker_num ServerArgs (those bind task
# concurrency, not decode threads). The only real constraint on decode
# threads is the --server-cpus affinity pin; this patch adds a
# process-wide thread-count knob so both arms can cap decode explicitly
# (e.g. match the edge terminal's core count for symmetric CPU
# comparisons).
# Env knob: SGLANG_DECORD_NUM_THREADS (int > 0; unset/0/invalid =
# native default). Read at VideoReader construction time, so the
# sidecar's os.environ set in main() applies to every decode.
#
# Backend coverage (sglang 0.5.20 picks torchcodec when importable,
# decord only as fallback — check WHICH branch your env takes!):
#   - torchcodec: __init__ clamps self._num_decode_threads to the env
#     value; get_frames_as_tensor then splits the sampled indices into
#     that many parallel VideoDecoder instances (the native "auto"
#     behavior is min(cpu_count, 16) instances). env > explicit ctor
#     arg wins (the arg is never passed by the EncPl paths).
#   - decord: num_threads is forwarded to every VideoReader
#     construction (decord's own internal pool size).
MARKER_3 = "# 6G-Testbed: decord thread cap"
TARGET_3 = "srt/utils/video_decoder.py"

# Anchor 3a: decord import + both VideoReader constructions (pristine
# sglang 0.5.20). A helper method inserted before __len__ builds the
# kwargs; both call sites route through it.
ANCHOR_DECORD_OLD = """\
        else:
            from decord import VideoReader, cpu

            if isinstance(source, bytes):
                import tempfile

                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                try:
                    os.write(fd, source)
                finally:
                    os.close(fd)
                self._tmp_path = tmp_path
                self._decoder = VideoReader(tmp_path, ctx=cpu(0))
            else:
                self._decoder = VideoReader(source, ctx=cpu(0))

    def __len__(self):
"""
ANCHOR_DECORD_NEW = """\
        else:
            from decord import VideoReader, cpu

            # __MARKER__
            # Cap decord's internal thread pool (sized by host cores by
            # default; unaffected by OMP/torch thread knobs). Unset/0/invalid
            # = decord native default — bit-identical behavior.
            _dt = os.environ.get("SGLANG_DECORD_NUM_THREADS", "")
            try:
                _dt = int(_dt)
            except ValueError:
                _dt = 0
            _vr_kwargs = {"ctx": cpu(0)}
            if _dt and _dt > 0:
                _vr_kwargs["num_threads"] = _dt
                logger.info(
                    "decord threads capped to %s (SGLANG_DECORD_NUM_THREADS)",
                    _dt,
                )
            if isinstance(source, bytes):
                import tempfile

                fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                try:
                    os.write(fd, source)
                finally:
                    os.close(fd)
                self._tmp_path = tmp_path
                self._decoder = VideoReader(tmp_path, **_vr_kwargs)
            else:
                self._decoder = VideoReader(source, **_vr_kwargs)

    def __len__(self):
""".replace("# __MARKER__", MARKER_3)

# Anchor 3b: torchcodec branch head in __init__ (pristine 0.5.20): the
# env knob clamps self._num_decode_threads BEFORE get_frames_as_tensor
# consults it for the parallel-instance count (native auto =
# min(cpu_count, 16)). env wins over the explicit ctor arg (never
# passed by the EncPl paths). Applies regardless of which backend the
# env ends up selecting — harmless no-op under decord.
ANCHOR_TORCHCODEC_OLD = """\
        self._source = source
        self._num_decode_threads = num_decode_threads
        self._source_bytes = source if isinstance(source, bytes) else None
        self._source_path = source if isinstance(source, str) else None
        self._tmp_path = None
        if _BACKEND == "torchcodec":
"""
ANCHOR_TORCHCODEC_NEW = """\
        self._source = source
        self._num_decode_threads = num_decode_threads
        self._source_bytes = source if isinstance(source, bytes) else None
        self._source_path = source if isinstance(source, str) else None
        self._tmp_path = None
        # __MARKER__
        # Cap the decode thread/instance count (SGLANG_DECORD_NUM_THREADS;
        # the name is historical — it now covers BOTH backends).
        # torchcodec: clamps the parallel VideoDecoder instance count in
        # get_frames_as_tensor (native auto = min(cpu_count, 16)); env
        # wins over the explicit ctor arg. decord: anchor 3a forwards the
        # value to VideoReader(num_threads=...). Unset/0/invalid = native
        # default — bit-identical behavior.
        _dtc = os.environ.get("SGLANG_DECORD_NUM_THREADS", "")
        try:
            _dtc = int(_dtc)
        except ValueError:
            _dtc = 0
        if _dtc and _dtc > 0:
            if self._num_decode_threads and self._num_decode_threads != _dtc:
                logger.info(
                    "decode threads: env cap %s overrides ctor arg %s "
                    "(SGLANG_DECORD_NUM_THREADS)",
                    _dtc, self._num_decode_threads,
                )
            self._num_decode_threads = _dtc
        if _BACKEND == "torchcodec":
""".replace("# __MARKER__", MARKER_3)


# ---------------------------------------------------------------------------
# Patch 4: mm processors/qwen_vl.py parallel video preprocess (switchable)
# ---------------------------------------------------------------------------
# Same target file as patch 2 (TARGET_2) — INDEPENDENT marker, so an
# install carrying patch 2 (any generation) is not touched by the
# marker-present check of patch 4 and vice versa. The anchors below are
# verified against BOTH layouts of the call region:
#   - pristine 0.5.20 (pre-patch-2): preprocess_video body has NO
#     frame_sampling traces;
#   - patch-2 installs: ANCHOR_TRACE (2c) added a logger.info block
#     between smart_nframes and linspace, plus a unique-lowering trace
#     after np.unique — INSIDE the function body we never anchor on.
# All three patch-4 anchors sit OUTSIDE every patch-2-edited region:
#   4a: the stdlib import block at the module head (patch 2 touches no
#       imports);
#   4b: the preprocess_video SIGNATURE + the first 4 body lines up to
#       (and including) `total_frames, video_fps = len(vr), vr.avg_fps`
#       (patch 2's first edit inside the body is the logger.info AFTER
#       smart_nframes — 5+ lines deeper);
#   4c: the _process_mm_data_uncached call site (patch 2 never edits
#       it; its anchors are smart_nframes / the keys frozenset / the
#       sampling block).
MARKER_4 = "# 6G-Testbed: parallel video preprocess (switchable)"
ENV_KNOB_4 = "SGLANG_QWEN_PARALLEL_VIDEO_PREPROCESS"

# Anchor 4a: module-head stdlib imports (pristine 0.5.20). Insert
# `import asyncio` alphabetically. qwen_vl.py:1024-1030's replacement
# uses asyncio.gather / run_in_executor; upstream never imports asyncio
# in this module (checked: no other reference).
ANCHOR_P4_IMPORT_OLD = """\
import math
import os
import re
import time
"""
ANCHOR_P4_IMPORT_NEW = """\
import asyncio
import math
import os
import re
import time
"""

# Anchor 4b: preprocess_video head (pristine 0.5.20). The signature
# keeps its public name/args (two EXTERNAL callers await it:
# interns1pro.py:65, disaggregation/encoder/preprocessor.py:539); the
# body moves VERBATIM into module-level _preprocess_video_sync — patch
# 2's traces inside the body ride along untouched. The shell preserves
# the early-passthrough (`not isinstance(vr, VideoDecoderWrapper)`
# -> `return vr, None`) — the ONLY branch that can run on the loop
# cheaply. The switch global is read INSIDE the shell at EVERY call:
# flipping it at runtime (POST /video_preprocess_mode) takes effect on
# the next request with no engine restart.
ANCHOR_P4_FUNC_OLD = """\
# process video, qwen-specific
async def preprocess_video(
    vr,
    image_factor: int = IMAGE_FACTOR,
    video_config: dict = {},
) -> torch.Tensor:
    # preprocessed video
    is_video_obj = isinstance(vr, VideoDecoderWrapper)
    if not is_video_obj:
        return vr, None
    entry_time = time.perf_counter()

    total_frames, video_fps = len(vr), vr.avg_fps
"""
ANCHOR_P4_FUNC_NEW = """\
# __MARKER__
# Dynamic switch (default OFF = upstream serial behavior, bit-identical).
# Initialized from env SGLANG_QWEN_PARALLEL_VIDEO_PREPROCESS; read at
# every preprocess_video call so the concurrent sidecar can flip the
# module global at runtime (POST /video_preprocess_mode) without an
# engine restart. When ON, _process_mm_data_uncached fans videos out
# over the processor's io_executor (width = mm_io_worker_num).
_PARALLEL_VIDEO_PREPROCESS = (
    os.environ.get("SGLANG_QWEN_PARALLEL_VIDEO_PREPROCESS", "")
    .strip().lower() in ("1", "true", "yes", "on")
)


# process video, qwen-specific
# __SHELL__
# Async shell over the upstream-verbatim sync body: the body contains
# ZERO awaits (decode + resize + pin_memory are all blocking calls), so
# awaiting this shell runs the body ON the caller's event loop — the
# exact upstream 0.5.20 behavior, kept as the OFF path. The ON path
# calls _preprocess_video_sync directly from an io_executor thread
# (see the call site below); external `await preprocess_video(...)`
# callers (interns1pro.py, disaggregation/encoder/preprocessor.py)
# keep working unchanged.
async def preprocess_video(
    vr,
    image_factor: int = IMAGE_FACTOR,
    video_config: dict = {},
) -> torch.Tensor:
    return _preprocess_video_sync(
        vr, image_factor=image_factor, video_config=video_config
    )


def _preprocess_video_sync(
    vr,
    image_factor: int = IMAGE_FACTOR,
    video_config: dict = {},
):
    # preprocessed video
    is_video_obj = isinstance(vr, VideoDecoderWrapper)
    if not is_video_obj:
        return vr, None
    entry_time = time.perf_counter()

    total_frames, video_fps = len(vr), vr.avg_fps
""".replace("__MARKER__", MARKER_4).replace(
    "# __SHELL__", MARKER_4 + " (async shell)"
)

# Anchor 4c: the _process_mm_data_uncached call site (pristine 0.5.20,
# qwen_vl.py:1024-1030). OFF branch = the original comprehension
# VERBATIM (same await, same order — bit-identical serial behavior);
# ON branch mirrors glm4v.py:570-607: run_in_executor on the processor's
# OWN io_executor + asyncio.gather. Width = --mm-io-worker-num.
#
# v1 BUG (measured on the box, 2026-10-06): the guard read
# `_PARALLEL_VIDEO_PREPROCESS and len(base_output.videos) > 1`, so with
# ONE video per request — the EncPl concurrent scenario's exact shape —
# every request fell to the serial else-branch and ON == OFF == upstream
# (5-video wave: 4 users reached decode while the 5th's 768-frame decode
# held the event loop ~168 s; server log 19:58:02 -> 20:00:50 with no
# prefill in between). The fix: the guard is the switch ALONE. The
# executor hop is what creates overlap (the `await future` yields the
# event loop; the pool serializes decodes at mm_io_worker_num width) —
# it must trigger for EVERY request, single video included. gather()
# is then irrelevant for 1-video requests but stays correct (and
# beneficial) for multi-video ones.
ANCHOR_P4_CALL_OLD = """\
        video_metadata = None
        if base_output.videos and not isinstance(base_output.videos[0], dict):
            videos_processed = [
                await preprocess_video(video, video_config=self.video_config)
                for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))
"""
ANCHOR_P4_CALL_NEW = """\
        video_metadata = None
        if base_output.videos and not isinstance(base_output.videos[0], dict):
            # __MARKER__
            # OFF (default): upstream serial comprehension, VERBATIM —
            # preprocess_video's shell runs the sync body on the event
            # loop (upstream 0.5.20 behavior).
            # ON: EVERY request hops its video(s) to the processor's
            # io_executor (width = mm_io_worker_num; glm4v.py:570-607
            # precedent) — the awaiting coroutine RELEASES the event
            # loop, so N concurrent requests overlap their decodes at
            # pool width instead of stacking on one serial staircase.
            # Applies to single-video requests too: the hop, not
            # gather(), is what creates the overlap.
            if _PARALLEL_VIDEO_PREPROCESS:
                loop = asyncio.get_running_loop()
                decode_tasks = [
                    loop.run_in_executor(
                        self.io_executor,
                        _preprocess_video_sync,
                        video,
                        IMAGE_FACTOR,
                        self.video_config,
                    )
                    for video in base_output.videos
                ]
                videos_processed = await asyncio.gather(*decode_tasks)
            else:
                videos_processed = [
                    await preprocess_video(video, video_config=self.video_config)
                    for video in base_output.videos
                ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))
""".replace("__MARKER__", MARKER_4)

# v1 (buggy) call-site replacement, byte-exact as shipped to the box —
# the upgrade path (marker present, v2 branch absent, v1 branch present)
# splices the fixed call site over this text. Built by reconstructing
# the v1 anchor generation: identical to ANCHOR_P4_CALL_NEW except the
# guard line and its comment block.
ANCHOR_P4_CALL_V1 = """\
        video_metadata = None
        if base_output.videos and not isinstance(base_output.videos[0], dict):
            # """ + MARKER_4 + """
            # OFF (default): upstream serial comprehension, VERBATIM —
            # preprocess_video's shell falls through to the sync body on
            # the event loop (upstream 0.5.20 behavior).
            # ON: fan each video out to the processor's io_executor
            # (width = mm_io_worker_num; glm4v.py:570-607 precedent) and
            # overlap the decodes — the per-video serial staircase
            # becomes a width-bounded parallel one.
            if _PARALLEL_VIDEO_PREPROCESS and len(base_output.videos) > 1:
                loop = asyncio.get_running_loop()
                decode_tasks = [
                    loop.run_in_executor(
                        self.io_executor,
                        _preprocess_video_sync,
                        video,
                        IMAGE_FACTOR,
                        self.video_config,
                    )
                    for video in base_output.videos
                ]
                videos_processed = await asyncio.gather(*decode_tasks)
            else:
                videos_processed = [
                    await preprocess_video(video, video_config=self.video_config)
                    for video in base_output.videos
                ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))
"""


def _die(msg: str) -> None:
    print(f"[patch_host_sglang] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


_QUIET = False


def _info(msg: str) -> None:
    # suppressed per-main(-quiet=True) call: the auto-load path
    # (ensure_sglang_patches) runs the patcher in EVERY wave worker
    # process, and N x "already patched" lines per wave is log spam;
    # real state changes (backup/patched/upgrade) stay visible by
    # printing directly, NOT through _info.
    if not _QUIET:
        print(f"[patch_host_sglang] {msg}")


def _locate_sglang_file(rel_path: str) -> Path:
    """Locate <rel_path> (e.g. 'srt/models/qwen3_vl.py') of the ACTIVE
    environment's sglang.

    Resolution order:
      1. import-system spec (works for PYTHONPATH/development installs;
         finds the package exactly the way Python itself would);
      2. installed-distribution metadata (site-packages installs).

    Neither imports sglang (importing it here would drag in torch and
    take seconds; worse, on some envs it has side effects).
    """
    # --- path 1: find_spec ----------------------------------------------------
    saved_sys_path = list(sys.path)
    try:
        import importlib.util

        spec = importlib.util.find_spec("sglang")
        if spec is not None and spec.origin:
            # origin: <pkg_root>/sglang/__init__.py
            target = Path(spec.origin).parent / Path(rel_path)
            if target.is_file():
                return target
    except (ImportError, ValueError, ModuleNotFoundError):
        pass
    finally:
        sys.path[:] = saved_sys_path

    # --- path 2: distribution metadata ---------------------------------------
    try:
        dist = distribution("sglang")
    except Exception as exc:
        _die(f"no sglang installation found in this env ({exc})")

    # distribution files are relative; match the exact rel_path
    parts = Path(rel_path).parts
    candidates = []
    for rel in (dist.files or []):
        rel_parts = Path(str(rel)).parts
        if rel_parts[: len(parts)] == parts:
            candidates.append(Path(dist.locate_file(Path(*rel_parts))))
    if not candidates:
        _die(f"sglang distribution found but {rel_path} not listed")

    target = candidates[0]
    if not target.is_file():
        _die(f"{target} does not exist")
    return target


def _apply_patch(target: Path, replacements: list[tuple[str, str]],
                 marker: str, backup_suffix: str, what: str,
                 upgrade: tuple[str, str, str] | None = None,
                 upgrade_mode: str = "elif") -> bool:
    """Apply one anchored source patch; True when the file was changed.

    replacements is a list of (old, new) exact-string pairs; ALL must
    match exactly once (or the marker must already be present —
    idempotent no-op). The result is ast.parse-verified and written
    atomically; any failure exits 1 WITHOUT touching the file.

    upgrade=(v2_marker, v2_branch, v3_branch): when v2_marker is
    present but the CURRENT branch text is not v3 (detected via a
    distinctive substring of v3_branch), the installed v2 branch is
    cut out and replaced by v3_branch in place — an in-place semantic
    upgrade with the standard backup/ast/atomic-write safety. Used for
    patch 2: v2 (ratio x total_frames) -> v3 (ratio x native base).

    upgrade_mode="line" (patch 3): the previous generation is detected
    by the marker alone; the upgrade applies the remaining replacement
    pairs on top (their anchor regions are untouched by the old
    generation). v2_branch/v3_branch are unused beyond the presence
    checks.

    upgrade_mode="replace" (patch 4): the previous generation EDITED the
    same anchor region the current generation edits (the call site), so
    neither the pristine anchor nor an untouched region exists. The
    v2_branch must appear EXACTLY ONCE and byte-exact; it is replaced
    by v3_branch verbatim. Remaining replacement pairs are skipped
    (regions identical across generations — any pair whose old text is
    present would mean a mixed layout; refuse).
    """
    src = target.read_text(encoding="utf-8")

    if marker in src:
        if upgrade is None:
            _info(f"{target.name}: marker present — already patched, "
                  f"nothing to do")
            return False
        v2_marker, v2_branch, v3_branch = upgrade
        if v3_branch in src:
            _info(f"{target.name}: marker present — already patched "
                  f"(current semantics), nothing to do")
            return False
        if v2_marker not in src:
            _die(f"{target.name}: marker present but neither the current "
                  f"nor the previous patch layout matches — refusing to "
                  f"touch it (restore from {backup_suffix} or reinstall "
                  f"sglang 0.5.20, then re-run)")
        if upgrade_mode == "line":
            # v2 = a PREVIOUS PATCH GENERATION identified by the marker
            # (patch 3 shipped decord-only first; the torchcodec clamp
            # is the current generation). The 3b anchor region (ctor
            # head before `if _BACKEND == "torchcodec":`) is untouched
            # by the old generation, so the upgrade simply applies the
            # remaining replacement pairs on top. A pair whose OLD text
            # is absent AND whose NEW text is absent means the old
            # generation already applied that anchor in a slightly
            # different wording (the gen-1 marker line carried a
            # doubled leading '#') — skipped; the fresh wording is
            # cosmetic, not semantic.
            patched = src
            for old, new in replacements:
                if new in patched:
                    continue  # already present (e.g. re-upgrade retry)
                if old not in patched:
                    # gen-1 wording drift: the anchor was applied by the
                    # previous generation; only the NEW text is absent.
                    # Detectable via the old generation's marker inside
                    # the target region; safe to skip (semantics equal).
                    continue
                if patched.count(old) != 1:
                    _die(
                        f"{target.name}: upgrade anchor ambiguous "
                        f"while upgrading — file left untouched "
                        f"(restore from {backup_suffix} or reinstall "
                        f"sglang 0.5.20, then re-run)"
                    )
                patched = patched.replace(old, new, 1)
            if patched == src:
                _info(f"{target.name}: marker present — already patched, "
                      f"nothing to do")
                return False
            _info(f"{target.name}: upgrading patch in place "
                  f"(decode thread cap: torchcodec branch added; see the "
                  f"new block comment)")
        elif upgrade_mode == "replace":
            # v1 -> v2 for regions the previous generation EDITED: the
            # v1 text must appear exactly once, byte-exact; splice the
            # v2 text over it. The other anchor regions are IDENTICAL
            # across generations — every remaining replacement pair must
            # be already applied (its new text present); any pair whose
            # OLD text is still present means a mixed/unknown layout:
            # refuse rather than half-upgrade.
            if src.count(v2_branch) != 1:
                _die(
                    f"{target.name}: marker present but the previous-"
                    f"generation patch layout is absent or ambiguous "
                    f"(count={src.count(v2_branch)}) — refusing to touch "
                    f"it (restore from {backup_suffix} or reinstall "
                    f"sglang 0.5.20, then re-run)"
                )
            for old, new in replacements:
                if old == v2_branch:
                    continue  # handled by the splice below
                if new == v3_branch:
                    continue  # the splice's own pair — absent on v1 by design
                if new in src:
                    continue  # identical across generations — already in
                _die(
                    f"{target.name}: mixed layout while upgrading — "
                    f"replacement pair not applied by the previous "
                    f"generation (anchor present: {src.count(old)}) — "
                    f"refusing to touch it (restore from {backup_suffix} "
                    f"or reinstall sglang 0.5.20, then re-run)"
                )
            patched = src.replace(v2_branch, v3_branch, 1)
            if patched == src:
                _info(f"{target.name}: marker present — already patched, "
                      f"nothing to do")
                return False
            _info(f"{target.name}: upgrading patch in place "
                  f"(parallel video preprocess: single-video executor "
                  f"hop fix; see the new call-site comment)")
        else:
            # Locate the installed v2 branch (marker line sits INSIDE the
            # v2 branch text, right after the elif line) and verify the
            # layout is byte-exact before touching anything. The upgrade
            # splices the v3 ELIF SECTION in place of the v2 elif section —
            # the surrounding `if "nframes"` head and fps `else:` tail stay
            # untouched (exactly how the v2->v3 layouts differ).
            start = src.find(v2_branch)
            if start < 0:
                _die(f"{target.name}: marker present but the installed "
                      f"frame_ratio branch does not match the known v2 layout "
                      f"byte-exactly — refusing to touch it (restore from "
                      f"{backup_suffix} or reinstall sglang 0.5.20, then re-run)")
            v2_end = src.find("\n    else:", start)
            if v2_end < 0:
                _die(f"{target.name}: installed v2 branch has no fps fallback "
                      f"after it — layout not recognized; refusing to touch it")
            v2_text = src[start:v2_end + 1]      # elif section incl. trailing \n
            if v2_text.rstrip("\n") != v2_branch.rstrip("\n"):
                _die(f"{target.name}: installed v2 branch deviates from the "
                      f"known layout — refusing to upgrade (restore from "
                      f"{backup_suffix} or reinstall sglang 0.5.20, then re-run)")
            # the v3 ELIF SECTION only (not the whole if/elif/else block):
            # from the elif line to just before the fps fallback's else
            v3_elif = v3_branch[v3_branch.index('    elif "frame_ratio" in ele:'):]
            v3_elif = v3_elif[: v3_elif.index("\n    else:")]
            patched = src[:start] + v3_elif + "\n" + src[start + len(v2_text):]
            _info(f"{target.name}: upgrading patch layout in place "
                  f"(semantics change: see the new branch comment)")
            # the remaining replacement pairs cover everything OUTSIDE the
            # v2/v3 branch that a current-layout install also lacks (e.g.
            # the preprocess_video frame-count trace): each must match
            # exactly once in the upgraded source — a missing anchor means
            # the layout is not what we expect; refuse rather than half-
            # upgrade.
            for old, new in replacements:
                if old == ANCHOR_NFRAMES_OLD:
                    continue  # pristine-only anchor; absent on v2 layouts
                if new in patched:
                    continue  # already present (e.g. re-upgrade retry)
                if patched.count(old) != 1:
                    _die(
                        f"{target.name}: upgrade anchor not found (or ambiguous) "
                        f"while upgrading — file left untouched (restore from "
                        f"{backup_suffix} or reinstall sglang 0.5.20, then re-run)"
                    )
                patched = patched.replace(old, new, 1)
    else:
        for old, _new in replacements:
            if src.count(old) != 1:
                _die(
                    f"{target.name}: anchor for {what} not found (or ambiguous) "
                    f"— the installed sglang is not the pristine 0.5.20 layout "
                    f"this patch was built for; refusing to touch it"
                )

        patched = src
        for old, new in replacements:
            patched = patched.replace(old, new, 1)

        if patched == src:
            _die(f"{target.name}: replacement produced no change — internal error")

    try:
        ast.parse(patched)
    except SyntaxError as exc:
        _die(f"{target.name}: patched source does not parse ({exc}) — "
             f"file left untouched")

    backup = target.with_name(target.name + backup_suffix)
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="")
        # NOT _info: a backup is a real state change — visible even in
        # quiet mode (auto-load path), where "already patched" is spam.
        print(f"[patch_host_sglang] backup: {backup}")

    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.stem}_patch_", suffix=".py"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(patched, encoding="utf-8", newline="")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()

    # NOT _info: same reason as the backup line above.
    print(f"[patch_host_sglang] patched: {what} ({target.name})")
    return True


def main(quiet: bool = False) -> None:
    global _QUIET
    _QUIET = quiet
    # ---- patch 1: qwen3_vl.py PRECOMPUTED_EMBEDDING guard ------------------
    target_1 = _locate_sglang_file(TARGET_1)
    _info(f"target: {target_1}")
    _apply_patch(
        target_1,
        [(ANCHOR_IMPORT_OLD, ANCHOR_IMPORT_NEW),
         (ANCHOR_METHOD_OLD, ANCHOR_METHOD_NEW)],
        MARKER, ".bak-precomputed",
        "PRECOMPUTED_EMBEDDING guard added to Qwen3VL "
        "_get_visual_feature (edge-mode pass-through)",
    )
    _info("note: scheduler child processes spawned AFTER this point "
          "import the patched module")

    # ---- patch 2: qwen_vl.py frame_ratio sampling ---------------------------
    target_2 = _locate_sglang_file(TARGET_2)
    _info(f"target: {target_2}")
    _apply_patch(
        target_2,
        [(ANCHOR_KEYS_OLD, ANCHOR_KEYS_NEW),
         (ANCHOR_NFRAMES_OLD, ANCHOR_NFRAMES_NEW),
         (ANCHOR_TRACE_OLD, ANCHOR_TRACE_NEW)],
        MARKER_2, ".bak-frameratio",
        "frame_ratio branch added to smart_nframes + "
        "QWEN_VIDEO_PREPROCESS_CONFIG_KEYS registration",
        upgrade=(MARKER_2, ANCHOR_NFRAMES_V2, ANCHOR_NFRAMES_NEW),
    )
    _info("note: the edge client imports smart_nframes from this module; "
          "patch it BEFORE any edge-mode frame_ratio run "
          "(max_frames alone needs NO patch — native whitelist key)")

    # ---- patch 3: video_decoder.py decode thread cap ------------------------
    # Two anchors: 3a (decord VideoReader num_threads) + 3b (torchcodec
    # parallel-instance clamp — the backend sglang 0.5.20 PREFERS when
    # torchcodec is importable, which the epd env is: torchcodec==0.15.0
    # in the freeze file). upgrade mode "line": an install carrying the
    # decord-only first generation (MARKER_3 present, 3b absent) is
    # upgraded in place by applying the remaining anchor pair — the 3b
    # anchor region is untouched by that first generation.
    target_3 = _locate_sglang_file(TARGET_3)
    _info(f"target: {target_3}")
    _apply_patch(
        target_3,
        [(ANCHOR_DECORD_OLD, ANCHOR_DECORD_NEW),
         (ANCHOR_TORCHCODEC_OLD, ANCHOR_TORCHCODEC_NEW)],
        MARKER_3, ".bak-decord",
        "decode thread cap (SGLANG_DECORD_NUM_THREADS env knob, BOTH "
        "backends) added to VideoDecoderWrapper",
        upgrade=(MARKER_3, MARKER_3, ANCHOR_TORCHCODEC_NEW),
        upgrade_mode="line",
    )

    # ---- patch 4: qwen_vl.py parallel video preprocess (switchable) ----------
    # Same FILE as patch 2 but an INDEPENDENT marker: an install that
    # already carries patch 2 (any generation — v2 or v3) still has all
    # three patch-4 anchor regions pristine (imports / preprocess_video
    # head BEFORE the frame_sampling traces / the call site patch 2
    # never touches). Idempotent on re-runs like every patch above.
    # Backup suffix differs from patch 2's (.bak-frameratio) so a fresh
    # install gets BOTH one-time backups; on an already-patch-2 backup
    # the FIRST patch-4 application writes .bak-parallel — capturing a
    # patch-2-only state, which is still the rollback point for patch 4
    # (restore .bak-frameratio first for patch-2 rollback, in order).
    target_4 = _locate_sglang_file(TARGET_2)
    _info(f"target: {target_4}")
    _apply_patch(
        target_4,
        [(ANCHOR_P4_IMPORT_OLD, ANCHOR_P4_IMPORT_NEW),
         (ANCHOR_P4_FUNC_OLD, ANCHOR_P4_FUNC_NEW),
         (ANCHOR_P4_CALL_OLD, ANCHOR_P4_CALL_NEW)],
        MARKER_4, ".bak-parallel",
        "parallel video preprocess (env "
        f"{ENV_KNOB_4}, default OFF = upstream serial) — "
        "preprocess_video split into async shell + "
        "_preprocess_video_sync; call site fans out over io_executor "
        "when the switch is on",
        # v1 -> v2 upgrade (the len>1-guard bug measured on the box):
        # marker present + fixed call site absent + v1 call site present
        # -> splice the fixed call site over the v1 text. The 4a/4b
        # regions are IDENTICAL between generations, so the call-site
        # pair alone carries the upgrade; on an upgrade retry the fixed
        # text is already in and the marker short-circuits (v3-in-src
        # check). upgrade_mode="replace": the v1 generation DID touch
        # the call-site anchor region, so the pristine
        # ANCHOR_P4_CALL_OLD is absent — neither the fresh-install path
        # nor "line" mode applies.
        upgrade=(MARKER_4, ANCHOR_P4_CALL_V1, ANCHOR_P4_CALL_NEW),
        upgrade_mode="replace",
    )
    _info("note: the switch is read per call — flip it at runtime via "
          "the concurrent sidecar's POST /video_preprocess_mode (or "
          f"set {ENV_KNOB_4}=1 before engine start for a persistent "
          "default); width when ON = mm_io_worker_num")


if __name__ == "__main__":
    main()
