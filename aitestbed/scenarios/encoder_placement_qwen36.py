"""
Encoder Placement Scenario (Qwen3.6 variant) for the 6G AI Traffic
Testbed.

Standalone sibling of encoder_placement.py for the Qwen3.6
(qwen3_5 / qwen3_5_moe) experiment — zero changes to the Qwen3-VL
scenario, so both experiments can run/debug in parallel. Talks to the
Qwen3.6 sidecar (server/sglang_engine_server_qwen36.py, port 8004)
via EncoderPlacementQwen36Client.

  mode "server": the terminal uploads the original mp4 (binary PUT)
                 and the server runs the full pipeline (decode +
                 processor + ViT + prefill + decode).
  mode "edge":   the terminal runs decode + processor + ViT locally
                 (single-segment [N, 2048] latent — Qwen3.6 has no
                 deepstack), uploads the serialized
                 precomputed_embedding bundle, and the server injects
                 it into prefill directly (skipping the server-side
                 visual encoder).

Both modes share the "upload first, reference by URL" flow, so the
chat request itself is tiny and directly comparable. The client's
TTFT window covers the full video-input -> first-token path (upload
and edge work are strictly synchronous inside the window).

Mode is selected via config key `mode` or env ENC_MODE_QWEN36 (env
wins; falls back to ENC_MODE for convenience).

Edge-mode knobs honored here (same surface as the Qwen3-VL scenario,
now ported): compressor/retention_ratio (VidCom2 token pruning),
bundle_quant (fp8 feature quantization), frame_ratio (ratio-of-native
sampling), vision_only (default TRUE — the 35B fp8 full load costs
~35 GB VRAM just to obtain the ~0.5 GB tower), attn_impl, and the
terminal CPU budget (config key `edge_cpus` or env ENCPL_EDGE_CPUS,
default "auto-half-other" = the SECOND half of the cores, disjoint
from the sidecar's default --server-cpus auto-half).

decode_threads: config key `decode_threads` or env
ENCPL_DECODE_THREADS_QWEN36 (env wins), default '' (decord native).
Caps the terminal VideoDecoderWrapper's internal decode thread pool
via SGLANG_DECORD_NUM_THREADS (env name historical — caps BOTH
backends); same knob as the sidecar's --decode-threads flag.

Dataset-bench mode (optional): config keys `dataset` ('mlvu' or
'videomme') + `dataset_dir` (env ENCPL_DATASET_DIR wins) + `limit`
(0 = all) replace the single configured video with a QA test set —
MLVU MCQ tasks (1-7) or the full Video-MME set, loaded via the
standalone tools' loaders. Each question is one logged turn; the
letter answer is scored with MetricsCalculator.extract_answer and
per-question detail (video_id/question/answer/correct) lands in the
log record's metadata.bench; a summary line prints at the end.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

from .base import BaseScenario, ScenarioResult

# Dataset-bench mode: reuse the standalone tools' loaders / MCQ prompt
# (module levels are stdlib-only — the netem/pandas imports inside
# videomme_eval's functions stay lazy). Same bootstrap the tools
# themselves use: aitestbed for `analysis.*`, aicodec/netemu for the
# transitively-imported packages.
_TOOLS_BOOTSTRAPPED = False


def _ensure_tools_importable():
    global _TOOLS_BOOTSTRAPPED
    if _TOOLS_BOOTSTRAPPED:
        return
    here = Path(__file__).resolve().parent
    aitestbed_root = here.parent
    repo_root = aitestbed_root.parent
    tools_dir = aitestbed_root / "tools"
    for _p in (str(tools_dir), str(aitestbed_root),
               str(repo_root / "aicodec"),
               str(repo_root / "netemu" / "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    _TOOLS_BOOTSTRAPPED = True


def _load_bench_items(dataset, dataset_dir, limit, tasks=None, sample=None):
    """Dataset question list -> [(video_path, prompt, entry), ...].

    MLVU: MCQ tasks (1-7) via encpl_bench_eval_qwen36.load_mlvu_entries;
    Video-MME: full set via videomme_eval.load_entries. Both loaders
    produce the same videomme-style entry dict; entries with a missing
    video file are skipped (counted in the warning).

    tasks: MLVU task selection — null/'' (default) = MCQ tasks (1-7),
    'all' = every task (8/9 generation tasks are recorded unscored),
    or a list of json stems (e.g. ['1_plotQA', '8_sub_scene']).
    sample: null/'' (default) = 'stratified' (per-task/duration quota
    sampling so a limit covers the workload mix), 'head' = first N
    rows in file order.
    """
    from analysis.metrics import MetricsCalculator  # noqa: F401 — same extract_answer downstream

    _ensure_tools_importable()
    import videomme_eval as _vmme
    import encpl_bench_eval_qwen36 as _bench

    if dataset not in ("mlvu", "videomme"):
        raise ValueError(
            f"invalid dataset {dataset!r} for "
            f"encoder_placement_qwen36: must be 'mlvu' or 'videomme'")
    d = Path(dataset_dir).expanduser()
    if not d.is_dir():
        raise ValueError(
            f"dataset_dir {dataset_dir!r} for encoder_placement_qwen36 "
            f"does not exist")
    limit = int(limit or 0)
    sample = (str(sample or "").strip().lower() or "stratified")
    if sample not in ("stratified", "head"):
        raise ValueError(
            f"invalid sample {sample!r} for encoder_placement_qwen36: "
            f"must be 'stratified' or 'head'")
    if dataset == "videomme":
        parquet_path = _vmme.find_parquet(d)
        video_root = _vmme.find_video_root(d)
        if parquet_path is None or video_root is None:
            raise ValueError(
                f"no Video-MME layout (test-*.parquet + videozip/) found "
                f"under {dataset_dir!r}")
        entries = _vmme.load_entries(parquet_path, video_root, limit,
                                     sample)
    else:  # mlvu
        mlvu_root = _bench.find_mlvu_root(d)
        if mlvu_root is None:
            raise ValueError(
                f"no MLVU layout (json/ + video/) found under "
                f"{dataset_dir!r}")
        task_spec = tasks if tasks not in (None, "", []) else "mcq"
        if isinstance(task_spec, str):
            try:
                tasks_resolved = _bench.resolve_mlvu_tasks(task_spec)
            except SystemExit as exc:
                # resolve_mlvu_tasks exits the process on a bad spec
                # (a CLI-tool convention); in the scenario convert it
                # to the scenario-contract ValueError instead of
                # killing the orchestrator.
                raise ValueError(
                    f"invalid tasks {task_spec!r} for "
                    f"encoder_placement_qwen36: {exc}") from exc
        else:
            tasks_resolved = [str(t) for t in task_spec]
            unknown = [t for t in tasks_resolved
                       if t not in _bench.MLVU_ALL_STEMS]
            if unknown:
                raise ValueError(
                    f"unknown MLVU tasks {unknown} for "
                    f"encoder_placement_qwen36; valid stems: "
                    f"{sorted(_bench.MLVU_ALL_STEMS)} (or 'mcq'/'all')")
        entries = _bench.load_mlvu_entries(mlvu_root, tasks_resolved,
                                           limit, sample)

    items = []
    n_missing = 0
    for e in entries:
        video_path = e.get("video_path")
        if not video_path or not Path(video_path).exists():
            n_missing += 1
            continue
        if e.get("options"):
            prompt = _vmme.build_prompt(e["question"], e["options"])
        else:
            # Generation task (mlvu 8/9): no options -> free-form
            # answer, recorded unscored (same guard as the bench tools).
            prompt = e["question"]
        items.append((Path(video_path), prompt, e))
    if n_missing:
        print(f"[EncPl36] warning: {n_missing} dataset entries skipped "
              f"(video file missing)")
    if not items:
        raise ValueError(
            f"dataset {dataset!r} under {dataset_dir!r} yielded no "
              f"runnable questions")
    return items


def _parse_edge_cpu_spec(spec, own_affinity):
    """Parse the serial edge-arm CPU spec into a sorted cpu list.

    Accepts: '' / None (-> None, pinning disabled), an int, a
    list/tuple/set of ints, or a string like "48-95,100" (spaces and
    a trailing comma tolerated). Special values: "auto-half" = first
    ceil(n/2) cores of own_affinity (mirrors the sidecar's
    --server-cpus auto-half), "auto-half-other" = the remaining cores
    (the default: disjoint from the sidecar's half). Raises
    ValueError for garbage; negative or duplicate ids are rejected /
    deduped. Returns None when pinning is off.
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"negative cpu id {spec}")
        return [spec]
    if isinstance(spec, (list, tuple, set)):
        out = set()
        for item in spec:
            n = int(item)
            if n < 0:
                raise ValueError(f"negative cpu id {n}")
            out.add(n)
        return sorted(out) or None
    text = str(spec).strip().lower()
    if not text:
        return None
    cpus = sorted(own_affinity or [])
    if text == "auto-half":
        if not cpus:
            return None
        k = max(1, -(-len(cpus) // 2))
        return cpus[:k]
    if text == "auto-half-other":
        if not cpus:
            return None
        k = max(1, -(-len(cpus) // 2))
        return cpus[k:] or cpus[:k]
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(
                    f"bad range {part!r} in edge_cpus spec {spec!r}") from exc
            if lo < 0 or hi < lo:
                raise ValueError(
                    f"bad range {part!r} in edge_cpus spec {spec!r}")
            out.update(range(lo, hi + 1))
        else:
            try:
                n = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"bad cpu id {part!r} in edge_cpus spec {spec!r}") from exc
            if n < 0:
                raise ValueError(
                    f"negative cpu id {n} in edge_cpus spec {spec!r}")
            out.add(n)
    return sorted(out) or None


class EncoderPlacementQwen36Scenario(BaseScenario):
    """
    Encoder placement scenario (Qwen3.6): server-side vs terminal-side ViT.
    """

    @property
    def scenario_type(self) -> str:
        return "encoder_placement_qwen36"

    def _flush_engine_cache(self) -> None:
        """Drop the SGLang radix + mm preprocess caches before a request.

        Across runs/turns the same video is re-sent, and without a flush
        the engine would serve run 2+ from a warm prefix, understating
        prefill/TTFT for every record after the first. Best-effort: a
        missing/erroring /flush_cache endpoint only warns (older sidecar
        deployments) — the run proceeds with the server's cache state.
        The call happens before t_request_start is captured, so it is
        outside every timing window.
        """
        base_url = getattr(self.client, "base_url", None)
        if not base_url:
            return
        url = base_url.rstrip("/") + "/flush_cache"
        try:
            resp = requests.post(url, timeout=60)
            if resp.status_code >= 400:
                print(f"[EncPl36] warning: flush_cache returned HTTP "
                      f"{resp.status_code}; engine caches were NOT reset "
                      f"(expect warm-prefix TTFT from run 2 on)")
        except Exception as exc:
            print(f"[EncPl36] warning: flush_cache failed ({exc}); engine "
                  f"caches were NOT reset (expect warm-prefix TTFT from "
                  f"run 2 on)")

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get(
            "model", "../aicodec/aicodec/models/Qwen3.6-35B-A3B")
        # Anchor relative model paths to the aitestbed/ dir (the scenario
        # dir's parent — "../aicodec/..." is a sibling of aitestbed/) so
        # the runner works from any cwd; the client's repo-root resolver
        # misses this shape and falls back to CWD-relative.
        if model and not os.path.isabs(model):
            here = os.path.dirname(os.path.abspath(__file__))
            anchored = os.path.normpath(
                os.path.join(os.path.dirname(here), model))
            if os.path.exists(anchored):
                model = anchored
        model_alias = self._anonymizer.model_alias(model)
        prompts = self.config.get("prompts", ["Describe this video in detail."])
        mode = (os.environ.get("ENC_MODE_QWEN36")
                or os.environ.get("ENC_MODE")
                or self.config.get("mode", "server")).strip().lower()
        if mode not in ("server", "edge"):
            raise ValueError(
                f"invalid mode {mode!r} for encoder_placement_qwen36: "
                f"must be 'server' or 'edge'")
        max_frames = int(self.config.get("max_frames", 768))
        max_tokens = int(self.config.get("max_tokens", 256))
        # Ratio-of-native frame sampling (v3): sglang's own base
        # (total_frames/video_fps*fps, i.e. 2fps x duration BEFORE
        # clamping) multiplied by R, then the native clamp chain (min 4,
        # cap max_frames/768, <= total, floor to a multiple of 2).
        # In (0, 1]; absent/0 = off. Edge mode samples locally (patched
        # smart_nframes v3, canary-verified by the client); server mode
        # requires the sidecar started with the SAME --frame-ratio (the
        # value rides in the body for record-keeping; the sidecar warns
        # on mismatch). Env ENCPL_FRAME_RATIO wins.
        frame_ratio_raw = (os.environ.get("ENCPL_FRAME_RATIO")
                           or self.config.get("frame_ratio"))
        try:
            frame_ratio = (float(frame_ratio_raw)
                           if frame_ratio_raw not in (None, "") else None)
        except ValueError as exc:
            raise ValueError(
                f"invalid frame_ratio {frame_ratio_raw!r} for "
                f"encoder_placement_qwen36: must be a number in (0, 1]"
            ) from exc
        # 0 folds to off (same as the sidecar's build_mm_process_config):
        # yaml `frame_ratio: 0` / ENCPL_FRAME_RATIO=0 disable percentage
        # sampling instead of raising, matching the documented contract.
        if frame_ratio == 0.0:
            frame_ratio = None
        if frame_ratio is not None and not (0.0 < frame_ratio <= 1.0):
            raise ValueError(
                f"invalid frame_ratio {frame_ratio} for "
                f"encoder_placement_qwen36: must be in (0, 1]")
        # flush_cache: reset SGLang radix/mm caches before each request so
        # every run is a cold-prefix measurement (default true). Set false
        # to measure steady-state cache-hit behaviour instead.
        flush_cache = bool(self.config.get("flush_cache", True))

        # Optional VidCom2 token compression on the edge bundle: compressor '' (default) or
        # 'vidcom2' (env COMPRESSOR wins), retention_ratio in (0, 1]
        # (env R_RATIO wins) only applied in edge mode.
        compressor = (os.environ.get("COMPRESSOR")
                      or self.config.get("compressor") or "").strip().lower()
        if compressor and compressor != "vidcom2":
            raise ValueError(
                f"invalid compressor {compressor!r} for "
                f"encoder_placement_qwen36: must be '' or 'vidcom2'")
        try:
            retention_ratio = float(
                os.environ.get("R_RATIO")
                or self.config.get("retention_ratio", 0.3))
        except ValueError as exc:
            raise ValueError(
                f"invalid retention_ratio for encoder_placement_qwen36: "
                f"must be a number in (0, 1]") from exc
        if not (0.0 < retention_ratio <= 1.0):
            raise ValueError(
                f"invalid retention_ratio {retention_ratio} for "
                f"encoder_placement_qwen36: must be in (0, 1]")
        compression_active = compressor == "vidcom2" and mode == "edge"
        if retention_ratio != 0.25 and not compression_active:
            print(f"[EncPl36] warning: retention_ratio={retention_ratio} "
                  f"is set but compression is inactive (needs edge mode + "
                  f"compressor=vidcom2); it will be ignored")

        # Two-tier retention split (edge + vidcom2 only): optional_ratio
        # carves the LEAST-CRITICAL tail out of the retained set. The
        # must tier (retention - optional) rides the reliable first
        # wave; the optional tier streams best-effort AFTER the chat
        # POST (parts that miss the server's merge snapshot are dropped
        # and never retransmitted). Per-frame optional counts follow
        # each frame's own dynamic scale gap (compute_scales run at
        # both bases). 0 (default) = legacy single-tier path, byte-
        # identical. Env ENCPL_OPTIONAL_RATIO wins.
        try:
            optional_ratio = float(
                (os.environ.get("ENCPL_OPTIONAL_RATIO")
                 or self.config.get("optional_ratio", 0.0) or 0.0))
        except ValueError as exc:
            raise ValueError(
                f"invalid optional_ratio for encoder_placement_qwen36: "
                f"must be a number in [0, retention_ratio)") from exc
        if optional_ratio < 0.0 or optional_ratio >= retention_ratio:
            raise ValueError(
                f"invalid optional_ratio {optional_ratio} for "
                f"encoder_placement_qwen36: must be in "
                f"[0, retention_ratio={retention_ratio})")
        if optional_ratio > 0.0 and not compression_active:
            optional_ratio = 0.0
            print(f"[EncPl36] warning: optional_ratio is set but "
                  f"compression is inactive (needs edge mode + "
                  f"compressor=vidcom2); it will be ignored")

        # Best-effort part granularity for the optional tier: how many
        # tokens are packed into one UDP part. Pure packaging
        # knob — the server merges by per-token insertion position and
        # never sees packet boundaries. Default 64 (~132KB/part fp8).
        # Env ENCPL_OPT_TOKENS_PER_PACKET wins.
        try:
            opt_tokens_per_packet = int(
                (os.environ.get("ENCPL_OPT_TOKENS_PER_PACKET")
                 or self.config.get("opt_tokens_per_packet", 64) or 64))
        except ValueError as exc:
            raise ValueError(
                f"invalid opt_tokens_per_packet for "
                f"encoder_placement_qwen36: must be a positive integer"
            ) from exc
        if opt_tokens_per_packet < 1:
            raise ValueError(
                f"invalid opt_tokens_per_packet {opt_tokens_per_packet} "
                f"for encoder_placement_qwen36: must be >= 1")

        # UDP second-wave timing knobs (edge + tiered only):
        # opt_send_gap_ms sleeps between ADJACENT datagrams (default
        # 0.05 ms ~ 20k datagrams/s paces the wave below the
        # receiver's reassembly rate; 0 = blast, only safe with a
        # raised receiver SO_RCVBUF and small waves);
        # opt_chat_delay_ms sleeps AFTER the wave and BEFORE the chat
        # POST (the dial trading opt-token arrival rate against TTFT
        # — the server snapshots at chat-arrival). Env
        # ENCPL_OPT_SEND_GAP_MS / ENCPL_OPT_CHAT_DELAY_MS win.
        try:
            opt_send_gap_ms = float(
                (os.environ.get("ENCPL_OPT_SEND_GAP_MS")
                 or self.config.get("opt_send_gap_ms", 0.05) or 0.0))
        except ValueError as exc:
            raise ValueError(
                f"invalid opt_send_gap_ms for encoder_placement_qwen36: "
                f"must be a number >= 0") from exc
        if opt_send_gap_ms < 0:
            raise ValueError(
                f"invalid opt_send_gap_ms {opt_send_gap_ms} for "
                f"encoder_placement_qwen36: must be >= 0")
        try:
            opt_chat_delay_ms = float(
                (os.environ.get("ENCPL_OPT_CHAT_DELAY_MS")
                 or self.config.get("opt_chat_delay_ms", 0.0) or 0.0))
        except ValueError as exc:
            raise ValueError(
                f"invalid opt_chat_delay_ms for encoder_placement_qwen36: "
                f"must be a number >= 0") from exc
        if opt_chat_delay_ms < 0:
            raise ValueError(
                f"invalid opt_chat_delay_ms {opt_chat_delay_ms} for "
                f"encoder_placement_qwen36: must be >= 0")

        # Optional fp8 quantization of the uploaded edge-bundle feature
        # tensor. Only valid TOGETHER with vidcom2: quantization applies
        # to the pruned bundle (compress first, quantize the survivors).
        # Env ENCPL_BUNDLE_QUANT wins; ''/'fp8'.
        bundle_quant = (os.environ.get("ENCPL_BUNDLE_QUANT")
                        or self.config.get("bundle_quant") or "").strip().lower()
        if bundle_quant and bundle_quant != "fp8":
            raise ValueError(
                f"invalid bundle_quant {bundle_quant!r} for "
                f"encoder_placement_qwen36: must be '' or 'fp8'")
        if bundle_quant and not (compressor == "vidcom2" and mode == "edge"):
            raise ValueError(
                f"bundle_quant='fp8' requires edge mode with "
                f"compressor=vidcom2 (quantization targets the pruned "
                f"bundle); got mode={mode!r}, compressor={compressor!r}")

        # Vision-tower-only edge load — DEFAULT TRUE on Qwen3.6 (the
        # Qwen3-VL scenario defaults False because its 4B full load is
        # cheap; here the fp8 35B full load costs ~35 GB VRAM just to
        # obtain the ~0.5 GB tower, so the standalone tower is the sane
        # bench default). ~1 GB edge VRAM. Edge mode only. Env
        # ENCPL_VISION_ONLY wins.
        vision_only_raw = (os.environ.get("ENCPL_VISION_ONLY")
                           or self.config.get("vision_only", True))
        if isinstance(vision_only_raw, str):
            vision_only = vision_only_raw.strip().lower() in (
                "1", "true", "yes", "on")
        else:
            vision_only = bool(vision_only_raw)

        # Edge ViT attention backend: 'sdpa' (default),
        # 'flash_attention_4', or 'eager' (see the client's
        # _resolve_attn_impl). Edge mode only. Env ENCPL_ATTN_IMPL wins.
        attn_impl = (os.environ.get("ENCPL_ATTN_IMPL")
                     or self.config.get("attn_impl", "sdpa"))
        try:
            from clients.encoder_placement_client_qwen36 import (
                _resolve_attn_impl as _norm_attn)
            attn_impl = _norm_attn(attn_impl)
        except ValueError as exc:
            raise ValueError(
                f"invalid attn_impl for encoder_placement_qwen36: "
                f"{exc}") from exc

        # Edge-mode terminal CPU budget: pin THIS process (decode +
        # processor + ViT run here in edge mode) to a cpu set. Default
        # "auto-half-other" = the SECOND half of the cores, disjoint
        # from the sidecar's default --server-cpus auto-half (first
        # half) on a single host. '' disables. Server mode warns and
        # ignores (no local model work there). Env ENCPL_EDGE_CPUS wins.
        edge_cpu_spec = (os.environ.get("ENCPL_EDGE_CPUS")
                         if os.environ.get("ENCPL_EDGE_CPUS") is not None
                         else self.config.get("edge_cpus",
                                              "auto-half-other"))
        try:
            own_affinity = os.sched_getaffinity(0)
        except AttributeError:
            own_affinity = None
        try:
            edge_cpu_list = _parse_edge_cpu_spec(edge_cpu_spec, own_affinity)
        except ValueError as exc:
            raise ValueError(
                f"invalid edge_cpus for encoder_placement_qwen36: "
                f"{exc}") from exc
        if edge_cpu_list and mode != "edge":
            print(f"[EncPl36] warning: edge_cpus={edge_cpu_spec!r} is set "
                  f"but mode={mode!r} (no terminal-side model work); "
                  f"edge CPU pinning is ignored")
            edge_cpu_list = None
        if edge_cpu_spec and edge_cpu_list is None and own_affinity is None:
            print(f"[EncPl36] warning: edge_cpus={edge_cpu_spec!r} given but "
                  f"os.sched_getaffinity is unavailable on this platform; "
                  f"edge CPU pinning is NOT applied")

        # decord decode-thread cap for the TERMINAL's VideoDecoderWrapper
        # (same SGLANG_DECORD_NUM_THREADS knob as the sidecar's
        # --decode-threads). Must be set BEFORE the first VideoReader is
        # constructed; read at construction time by the patched
        # video_decoder.py. A pre-set env var wins (explicit override).
        # Env ENCPL_DECODE_THREADS_QWEN36 wins over config decode_threads.
        decode_threads_spec = (
            os.environ.get("ENCPL_DECODE_THREADS_QWEN36")
            if os.environ.get("ENCPL_DECODE_THREADS_QWEN36") is not None
            else self.config.get("decode_threads"))
        if decode_threads_spec not in (None, ""):
            try:
                n_dt = int(decode_threads_spec)
            except (TypeError, ValueError):
                raise ValueError(
                    f"invalid decode_threads for encoder_placement_qwen36: "
                    f"{decode_threads_spec!r} (must be an integer)")
            if n_dt > 0:
                os.environ.setdefault("SGLANG_DECORD_NUM_THREADS", str(n_dt))
                print(f"[EncPl36] decode_threads: capped to {n_dt} "
                      f"(SGLANG_DECORD_NUM_THREADS; env name historical — "
                      f"caps BOTH backends)")
            else:
                print(f"[EncPl36] warning: decode_threads={n_dt} ignored "
                      f"(must be > 0)")

        # torch/OMP intra-op thread cap for the terminal's CPU-side
        # preprocessing (resize / HF processor; edge mode only — server
        # mode has no local model work, warn so an edge-tuned yaml does
        # not mislead). Effective value min(N, edge_cpu pin core count):
        # N above the pin would only thrash the scheduler inside the
        # budget; N below simulates a weaker terminal. Applied INSIDE
        # _apply_edge_cpu_budget so the pin / env vars / torch pool cap
        # land atomically and the restore token stays one unit. Env
        # ENCPL_PREPROCESS_THREADS_QWEN36 wins over the yaml key.
        # Omitted = the pin's core count (historical behaviour).
        preprocess_threads_spec = (
            os.environ.get("ENCPL_PREPROCESS_THREADS_QWEN36")
            if os.environ.get("ENCPL_PREPROCESS_THREADS_QWEN36") is not None
            else self.config.get("preprocess_threads"))
        preprocess_threads = None
        if preprocess_threads_spec not in (None, ""):
            try:
                preprocess_threads = int(preprocess_threads_spec)
            except (TypeError, ValueError):
                raise ValueError(
                    f"invalid preprocess_threads for "
                    f"encoder_placement_qwen36: "
                    f"{preprocess_threads_spec!r} (must be an integer)")
            if preprocess_threads <= 0:
                print(f"[EncPl36] warning: preprocess_threads="
                      f"{preprocess_threads} ignored (must be > 0)")
                preprocess_threads = None
        if preprocess_threads is not None and mode != "edge":
            print(f"[EncPl36] warning: preprocess_threads="
                  f"{preprocess_threads} is set but mode={mode!r} (no "
                  f"terminal-side preprocessing); NOT applied")
            preprocess_threads = None

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        # Optional dataset-bench mode: iterate a QA test set (MLVU MCQ /
        # Video-MME) instead of the single configured video. Absent
        # dataset/dataset_dir = the historical single-video path (zero
        # behaviour change). Env ENCPL_DATASET_DIR wins over the yaml
        # key (same precedence convention as the other knobs).
        dataset = str(self.config.get("dataset", "") or "").strip().lower()
        dataset_dir = (os.environ.get("ENCPL_DATASET_DIR")
                       or self.config.get("dataset_dir"))
        if dataset and dataset not in ("mlvu", "videomme"):
            raise ValueError(
                f"invalid dataset {dataset!r} for "
                f"encoder_placement_qwen36: must be 'mlvu' or 'videomme'")
        if dataset and not dataset_dir:
            raise ValueError(
                f"dataset {dataset!r} requires dataset_dir for "
                f"encoder_placement_qwen36")
        limit = int(self.config.get("limit", 0) or 0)
        # MLVU task selection (null/'' = MCQ 1-7; 'all'; or a stem
        # list) and sampling strategy (null/'' = stratified; 'head').
        # Both feed _load_bench_items — see its docstring.
        bench_tasks = self.config.get("tasks")
        bench_sample = self.config.get("sample")

        video_path = self._resolve_video_path()
        if dataset:
            bench_items = _load_bench_items(dataset, dataset_dir, limit,
                                            bench_tasks, bench_sample)
            print(f"[EncPl36] dataset bench: {dataset} "
                  f"({len(bench_items)} questions"
                  f"{f', limit={limit}' if limit else ''})")
        else:
            bench_items = None
            if video_path is None:
                result.success = False
                result.error_message = (
                    "No video_path configured for "
                    "encoder_placement_qwen36 scenario")
                return result

        # Apply the terminal CPU budget BEFORE the first request: the
        # client's heavy libs (torch, decord, transformers) are lazily
        # imported inside the request path, so first-touch import AND
        # model load happen inside the budget. Restored in finally —
        # this orchestrator process is shared with other scenarios.
        _cpu_restore = self._apply_edge_cpu_budget(
            edge_cpu_list, preprocess_threads)
        # token None = pin NOT applied (disabled/unavailable/failed);
        # truthy token = pinned, restore data inside.
        applied_edge_cpus = (
            ",".join(str(c) for c in edge_cpu_list)
            if (_cpu_restore is not None) else None)
        # Dataset mode iterates (video, prompt, entry) work items; the
        # single-video mode degenerates to the historical per-prompt
        # list over the one configured video.
        if bench_items is not None:
            work_items = bench_items
        else:
            work_items = [(video_path, prompt, None) for prompt in prompts]

        # In-memory correctness counters (dataset mode): only 4-option
        # MCQ entries with a ground-truth letter score; answers that
        # fail to parse stay out of the denominator (same semantics as
        # MetricsCalculator.calculate_task_accuracy task_class "real").
        bench = {"scored": 0, "parse_ok": 0, "parse_fail": 0,
                 "correct": 0, "total": len(work_items)}
        # one-shot warning latch (see the BUG N note in the loop)
        warned_single_tier = False
        try:
            for turn_index, (turn_video, prompt, entry) in enumerate(
                    work_items):
                turn_video_str = str(turn_video)
                self._wait_between_prompts(turn_index)
                if flush_cache:
                    self._flush_engine_cache()
                t_request_start = time.time()

                try:
                    response = self.client.generate_content_with_video(
                        prompt=prompt,
                        video_path=turn_video_str,
                        model=model,
                        mode=mode,
                        max_frames=max_frames,
                        max_tokens=max_tokens,
                        compressor=compressor if mode == "edge" else "",
                        retention_ratio=retention_ratio,
                        bundle_quant=bundle_quant if mode == "edge" else "",
                        vision_only=vision_only if mode == "edge" else False,
                        attn_impl=attn_impl if mode == "edge" else "sdpa",
                        frame_ratio=frame_ratio,
                        optional_ratio=(optional_ratio
                                        if mode == "edge" else 0.0),
                        opt_tokens_per_packet=opt_tokens_per_packet,
                        opt_send_gap_ms=opt_send_gap_ms,
                        opt_chat_delay_ms=opt_chat_delay_ms,
                    )

                    tokens_in = response.tokens_in
                    tokens_out = response.tokens_out
                    if tokens_in is None:
                        tokens_in = self.client.estimate_tokens(prompt, model)
                    if tokens_out is None:
                        tokens_out = self.client.estimate_tokens(
                            response.total_content, model)

                    engine_stats = getattr(response, "engine_stats", {}) or {}

                    # BUG N visibility: a tiered-configured run
                    # (optional_ratio > 0) whose compression early-exited
                    # is SINGLE-TIER in reality — the must bundle rides
                    # the first wave only and no UDP wave ever happens.
                    # Warn once per run instead of letting the row's
                    # opt_* fields mislabel the experiment.
                    if (mode == "edge" and optional_ratio > 0
                            and not engine_stats.get("tiered")):
                        if not warned_single_tier:
                            warned_single_tier = True
                            print(f"[EncPl36] warning: optional_ratio="
                                  f"{optional_ratio} requested but this run "
                                  f"is effectively SINGLE-TIER (vidcom2 "
                                  f"selection early-exited / compression "
                                  f"inactive); no optional UDP wave is "
                                  f"happening — accuracy comparisons against "
                                  f"a real tiered run would be wrong")

                    # E2E timing anchors: t_request_start was captured BEFORE
                    # the client's edge/upload work, so both TTFT
                    # (t_first_token - t_request_start) and latency_sec
                    # (t_last_chunk - t_request_start) span video input ->
                    # first/last token, keeping latency_sec >= TTFT in both
                    # modes.
                    t_last_chunk = response.t_last_chunk
                    latency_sec = (
                        t_last_chunk - t_request_start
                        if t_last_chunk is not None else 0.0
                    )

                    # Dataset mode: score this turn's answer (same
                    # extract_answer as the tools/final report).
                    turn_meta = {
                        # engine stats FIRST so the explicit keys below
                        # always win: a same-named key added to the
                        # client's engine_stats in the future must never
                        # silently overwrite the scenario's authoritative
                        # row fields (today's engine_stats keys — mode,
                        # video_path, upload_*, stage*, conns, e2e_* —
                        # collide only on mode/video_path, where both
                        # writers carry identical values, so this
                        # reordering is a no-op for current data).
                        **engine_stats,
                        "model": model_alias,
                        "mode": mode,
                        "video_path": turn_video_str,
                        # explicit per-turn metrics (SQL-queryable without
                        # decoding timing anchors; same field names/rounding
                        # as the concurrent scenario's per-video rows, so
                        # the orchestrator's rich bench summary and
                        # tools/encpl_bench_db_report.py get E2E TTFT/LAT
                        # stats from serial runs too)
                        "ttft_sec": (
                            None if response.t_first_chunk is None
                            else round(
                                response.t_first_chunk - t_request_start, 4)),
                        "latency_sec": (None if latency_sec is None
                                        else round(latency_sec, 4)),
                        # terminal CPU budget actually applied (comma
                        # cpu list; null = unpinned) — self-describing
                        # runs
                        "edge_cpus": applied_edge_cpus,
                    }
                    if entry is not None:
                        from analysis.metrics import MetricsCalculator
                        answer_letter = (entry.get("answer") or
                                         "").strip().upper()
                        model_letter = ""
                        bench["scored"] += 1
                        if len(entry.get("options") or []) == 4 and \
                                answer_letter in ("A", "B", "C", "D"):
                            model_letter = (
                                MetricsCalculator.extract_answer(
                                    response.total_content or "") or "")
                            if model_letter:
                                bench["parse_ok"] += 1
                                if model_letter == answer_letter:
                                    bench["correct"] += 1
                            else:
                                bench["parse_fail"] += 1
                        turn_meta["bench"] = {
                            "dataset": dataset,
                            "video_id": entry.get("video_id"),
                            "question": entry.get("question"),
                            "task_type": entry.get("task_type"),
                            "domain": entry.get("domain"),
                            "duration": entry.get("duration"),
                            "answer": answer_letter,
                            "model_answer": model_letter,
                            "correct": (
                                model_letter == answer_letter
                                if model_letter else None),
                            # always False here (serial turns are never
                            # barrier filler); kept for schema
                            # compatibility with the concurrent
                            # scenario's rows (aggregators skip
                            # padded=true)
                            "padded": False,
                        }

                    record = self._create_log_record(
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=network_profile,
                        request_bytes=response.request_bytes,
                        response_bytes=response.response_bytes,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        t_request_start=t_request_start,
                        t_first_token=response.t_first_chunk,
                        t_last_token=t_last_chunk,
                        latency_sec=latency_sec,
                        http_status=200,
                        success=True,
                        is_streaming=True,
                        chunk_count=len(response.chunks),
                        inter_chunk_times=json.dumps(
                            response.inter_chunk_times),
                        trace_request=response.request_payload,
                        trace_response=response.response_payload,
                        trace_events=response.response_events,
                        trace_note="encoder_placement_qwen36",
                        metadata=json.dumps(turn_meta),
                    )
                    self.logger.log(record)
                    result.log_records.append(record)

                    result.turn_count += 1
                    result.api_call_count += 1
                    result.total_latency_sec += latency_sec
                    result.total_request_bytes += response.request_bytes
                    result.total_response_bytes += response.response_bytes
                    if tokens_in:
                        result.total_tokens_in += tokens_in
                    if tokens_out:
                        result.total_tokens_out += tokens_out
                    if result.ttft_sec is None and response.t_first_chunk:
                        result.ttft_sec = (
                            response.t_first_chunk - t_request_start)

                except Exception as exc:
                    record = self._create_log_record(
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=network_profile,
                        t_request_start=t_request_start,
                        latency_sec=time.time() - t_request_start,
                        http_status=0,
                        error_type=str(exc),
                        success=False,
                        is_streaming=True,
                    )
                    self.logger.log(record)
                    result.log_records.append(record)
                    # Dataset mode: one failed question does not abort
                    # the bench (recorded, counted, keep going); the
                    # single-video mode keeps the historical abort.
                    if bench_items is None:
                        result.success = False
                        result.error_message = str(exc)
                        break
                    print(f"[EncPl36] question {turn_index + 1}/"
                          f"{bench['total']} failed ({exc}); continuing")

        finally:
            self._restore_edge_cpu_budget(_cpu_restore)

        # Dataset mode summary (in-memory; per-question detail lives in
        # the DB log records' metadata.bench).
        if bench_items is not None:
            n_ok, n_c = bench["parse_ok"], bench["correct"]
            acc = n_c / n_ok if n_ok else 0.0
            print(f"[EncPl36] bench summary: dataset={dataset} "
                  f"total={bench['total']} scored={bench['scored']} "
                  f"parse_ok={n_ok} parse_fail={bench['parse_fail']} "
                  f"correct={n_c} accuracy={acc:.3f}")
            # All-failed guard: a bench where EVERY question failed
            # (e.g. sidecar down) is a failed run, not a success with
            # zero answers — the partial-failure contract (some ok ->
            # success=True) only covers real benches.
            if bench["scored"] == 0 and result.success:
                result.success = False
                result.error_message = (
                    f"dataset bench: all {bench['total']} questions "
                    f"failed (0 scored)")

        return result

    _NUM_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS")

    def _apply_edge_cpu_budget(self, cpu_list, preprocess_threads=None):
        """Pin this process to cpu_list for the duration of the run.

        Returns an opaque restore token (None when nothing was pinned,
        meaning "budget NOT applied" — the caller records edge_cpus
        accordingly). Non-Linux / setaffinity failure: warn and return
        None (measurement still valid, just with a softer terminal-
        resource model — same contract as the concurrent worker).
        Thread-pool caps (OMP/MKL/OPENBLAS_NUM_THREADS +
        torch.set_num_threads) keep intra-op pools inside the budget;
        preprocess_threads REPLACES the pin-derived count, clamped to
        the budget (min(N, len(cpu_list)) — N above the pin only
        thrashes the scheduler inside it; N below simulates a weaker
        terminal). The env vars are SET (not setdefault) only when
        absent, and both they and the pre-existing torch thread count
        are part of the restore token.
        """
        if not cpu_list:
            return None
        spec = ",".join(str(c) for c in cpu_list)
        try:
            prev_affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0, set(cpu_list))
        except (AttributeError, OSError) as exc:
            print(f"[EncPl36] warning: cannot pin to cpus [{spec}] ({exc}); "
                  f"running unpinned")
            return None
        effective = len(cpu_list)
        if preprocess_threads is not None:
            if preprocess_threads > len(cpu_list):
                print(f"[EncPl36] note: preprocess_threads="
                      f"{preprocess_threads} clamped to the "
                      f"{len(cpu_list)}-cpu budget")
            else:
                effective = preprocess_threads
        prev_env = {}
        for var in self._NUM_THREAD_VARS:
            if var not in os.environ:
                os.environ[var] = str(effective)
                prev_env[var] = None
        torch_threads = None
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            torch_threads = torch_mod.get_num_threads()
            torch_mod.set_num_threads(effective)
        print(f"[EncPl36] edge cpus: pinned to [{spec}] "
              f"(intra-op threads: {effective}; restored after the run)")
        return (prev_affinity, prev_env, torch_threads)

    def _restore_edge_cpu_budget(self, token):
        """Undo _apply_edge_cpu_budget (token from that method)."""
        if token is None:
            return
        prev_affinity, prev_env, torch_threads = token
        try:
            os.sched_setaffinity(0, prev_affinity)
        except (AttributeError, OSError) as exc:
            print(f"[EncPl36] warning: cannot restore affinity ({exc})")
        for var in prev_env:
            if var in os.environ:
                del os.environ[var]
        if torch_threads is not None:
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None:
                torch_mod.set_num_threads(torch_threads)

    def _resolve_video_path(self) -> Path | None:
        video_path = self.config.get("video_path")
        if not video_path:
            return None
        base_dir = self.config.get("video_base_dir")
        path = Path(video_path).expanduser()
        if not path.is_absolute() and base_dir:
            path = Path(base_dir).expanduser() / path
        path = path.resolve()
        if not path.exists():
            print(f"[EncPl36] warning: video path does not exist: {path}")
            return None
        return path
