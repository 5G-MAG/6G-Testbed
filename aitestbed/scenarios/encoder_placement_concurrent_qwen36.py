"""
Encoder Placement Concurrent Scenario (Qwen3.6 variant) for the 6G AI
Traffic Testbed.

Standalone sibling of scenarios/encoder_placement_concurrent.py (the
Qwen3-VL concurrent scenario) for the Qwen3.6 (qwen3_5 / qwen3_5_moe)
experiment — zero changes to that file, so both experiments run/debug
in parallel. Same two modes, same wave measurement contract, but every
model-specific surface (client class, worker module, vision_only
default, path anchoring) follows the SERIAL Qwen3.6 scenario's
conventions (scenarios/encoder_placement_qwen36.py):

  N terminal users start together (barrier-aligned t_request_start),
  each issuing the full synchronous per-user flow against
  server/sglang_engine_server_concurrent_qwen36.py (the lock-free
  Qwen3.6 sidecar, port 8014):

    mode "server": upload mp4 -> server decode+proc+ViT+prefill
    mode "edge":   local decode+proc+ViT -> upload bundle ->
                   server torch.load+prefill

  Under concurrency the server-side resource contention differs per
  mode (server: N×ViT + N×decode on the inference GPU/CPU; edge:
  N×torch.load only), which is precisely the effect the per-user TTFT
  distribution + engine_stats stage attribution should reveal.

Qwen3.6-vs-Qwen3-VL config-surface deltas (deliberate, mirroring the
serial Qwen3.6 scenario):
  - compressor / retention_ratio / bundle_quant / frame_ratio keys:
    PORTED from the Qwen3-VL concurrent scenario (the Qwen3.6 client's
    generate_content_with_video now carries the same four knobs); same
    validation, same worker-cfg passthrough, same semantics. Unknown
    keys in the yaml are simply ignored (BaseScenario reads config
    keys explicitly).
  - vision_only defaults TRUE (the Qwen3-VL concurrent scenario
    defaults False because its 4B full load is cheap; here the fp8 35B
    full load costs ~35 GB VRAM per edge process just to obtain the
    ~0.5 GB tower, so the standalone tower is the sane default; in
    PROCESS mode it also means ~1 GB VRAM per worker instead of
    ~35 GB).
  - mode honors env ENC_MODE_QWEN36 (falls back to ENC_MODE) over the
    yaml key, matching the serial Qwen3.6 scenario.

Everything else — wave semantics, thread-vs-process arm topologies,
per-wave flush, per-user prompt uniquification, LogRecord schema,
aggregates — is the Qwen3-VL concurrent scenario's design verbatim:

  - Two arm topologies, chosen per mode:
      * EITHER mode: threads by default (ThreadPoolExecutor, one
        thread per user), or N OS PROCESSES via edge_processes=true
        (runner --enable-multi-processes) — one truly independent
        terminal per user: own interpreter, own GIL, own connection,
        and in edge mode own ViT instance. Only the GPU itself
        remains shared (kernel-level interleave, a single-host
        limitation). Synchronized by an mp.Barrier (N workers +
        parent): workers do their heavy setup (edge: model load +
        CUDA context; server: interpreter spawn only), signal
        'ready', and only then does the parent join the barrier —
        all setup stays outside every timing window. See
        _run_wave_processes and clients/encpl_edge_worker_qwen36.py
        for the message protocol.
      * Why processes in server mode too: per-user work is pure
        HTTP, but the thread wave's N concurrent uploads serialize on
        the shared GIL + one file handle (Qwen3-VL measurement: 4
        users uploading the same 309 MB video staircased 1.9 -> 3.2 s);
        process mode gives each user an independent upload path (disk
        read + socket write overlap). Spawn overhead lands before
        'ready', outside the timing window.

  - flush_cache once per WAVE (barrier + flush before t_request_start),
    not per request: the serial scenario's per-request flush is
    impossible under concurrency (N users race one flush endpoint) and
    a wave-aligned flush keeps every run a cold-prefix measurement with
    zero in-window cost.

  - Per-user prompt uniquification: every user's text part is prefixed
    with a short "[u<i>] " tag BEFORE the video placeholder (the
    Qwen3.6 client's _build_templated_text reorders text-first when
    user_tag is set), so the N users of a wave share the video tokens
    but never the text prefix. Without it, users #2..N of a wave could
    ride user #1's radix-cached ~14.5k-token prefill. The sidecar
    additionally defaults disable_radix_cache=True (see its
    --enable-radix-cache), making the engine itself prefix-share-proof;
    the tag keeps semantics correct even against a radix-enabled
    sidecar, and costs ~4 tokens per user.

  - Per-user LogRecords keep the EXACT schema of the serial Qwen3.6
    scenario (turn_index encodes wave*1000+user so per-user rows stay
    distinguishable; metadata gains wave_size / user_index /
    edge_processes / edge_cpus / ttft_ms fields). Aggregates (TTFT
    mean/P50/P90/max per wave) are printed and attached to the
    ScenarioResult metadata.

  - Wave semantics: one wave per configured prompt (like one turn per
    prompt in the serial scenario), inter_prompt_delay_sec between
    waves as before.

Config surface (config keys ONLY — no environment variables except
ENC_MODE_QWEN36/ENC_MODE for mode, matching the serial Qwen3.6
scenario; the standalone runner maps every CLI flag onto these keys,
CLI > yaml):
  mode / concurrent_users (wave size N) / compressor / retention_ratio
  / bundle_quant / frame_ratio / vision_only / attn_impl /
  edge_processes / flush_cache / max_frames / max_tokens / video_path
  / video_paths / video_base_dir — same key names and semantics as
  the Qwen3-VL concurrent yaml (vision_only / attn_impl /
  compressor / retention_ratio / bundle_quant are edge-mode only;
  ignored with a warning in server mode).
  video_paths — OPTIONAL list of per-user videos: user i uses
  video_paths[i] in BOTH modes (strict: len(video_paths) MUST equal
  concurrent_users — a miscount aborts the run; distinct files are
  the point, N copies of one path is the caller's mistake to make
  explicitly). Absent/empty falls back to video_path shared by all
  users (the original behavior). Works with threads AND processes,
  server AND edge. Runner flag: --video-paths.
  edge_cpus_per_worker / edge_cpu_ids — terminal-side CPU budget for
  edge PROCESS mode (defaults: 1 / all-of-parent's-affinity): each
  worker process is pinned round-robin to `edge_cpus_per_worker` cpus
  from the `edge_cpu_ids` pool ("0-3,8" style or explicit list),
  modelling N users on real N-core terminals. Thread mode / non-Linux
  warn and skip; records and DB metadata carry the applied edge_cpus.
  Pair with the sidecar's --server-cpus so the server arm gets a
  comparable CPU budget (video decode is CPU-bound) — without BOTH
  caps the server-vs-edge contrast on a many-core host is muted.
"""

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .base import BaseScenario, ScenarioResult
from .encoder_placement_qwen36 import _load_bench_items

# Bootstrap the sibling aicodec package before any (lazy) `from
# clients...` import executes: importing the clients package runs its
# __init__, whose realtime_webrtc_vlm_client does `import aicodec` —
# without the bootstrap that resolves to the OUTER pyproject-only
# aicodec/ dir as an empty PEP 420 namespace package on hosts without
# an editable install (observed on dgspl3310). The standalone runner
# already bootstraps before importing this module; the guard here
# keeps the scenario safe through any other entry, and keeps the
# parent's sys.path correct for the spawn children's own clients
# import (they inherit sys.path at spawn time). Idempotent.
_AICODEC_PATH = Path(__file__).resolve().parents[2] / "aicodec"
if _AICODEC_PATH.is_dir() and str(_AICODEC_PATH) not in sys.path:
    sys.path.insert(0, str(_AICODEC_PATH))


def _parse_cpu_ids(spec, fallback=None) -> list[int]:
    """Parse a CPU-id spec into a sorted list of ints.

    Accepts: None (-> fallback, or [] when no fallback), an int, a
    list/tuple/set of ints, or a string like "0-3,7,12-13" (also
    tolerating single values / spaces / a trailing comma). Special
    value "auto-half-other" = the SECOND half of the fallback pool
    (ceil-split), mirroring the serial scenario's edge_cpus default:
    the sidecar's --server-cpus auto-half takes the FIRST half, so
    the two arms stay disjoint with zero manual spec. Raises
    ValueError for garbage; negative or duplicate ids are rejected /
    deduped. The fallback is the worker's own affinity (default pool
    = every CPU the parent may use).
    """
    if spec is None:
        return sorted(int(c) for c in (fallback or []))
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
        return sorted(out)
    text = str(spec).strip()
    if not text:
        return sorted(int(c) for c in (fallback or []))
    if text.lower() == "auto-half-other":
        cpus = sorted(int(c) for c in (fallback or []))
        if not cpus:
            return []
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
                raise ValueError(f"bad range {part!r} in cpu spec "
                                 f"{spec!r}") from exc
            if lo < 0 or hi < lo:
                raise ValueError(f"bad range {part!r} in cpu spec "
                                 f"{spec!r}")
            out.update(range(lo, hi + 1))
        else:
            try:
                n = int(part)
            except ValueError as exc:
                raise ValueError(f"bad cpu id {part!r} in cpu spec "
                                 f"{spec!r}") from exc
            if n < 0:
                raise ValueError(f"negative cpu id {n} in cpu spec "
                                 f"{spec!r}")
            out.add(n)
    return sorted(out)


class EncoderPlacementConcurrentQwen36Scenario(BaseScenario):
    """
    Encoder placement scenario (Qwen3.6): N concurrent terminal users
    per wave.
    """

    @property
    def scenario_type(self) -> str:
        return "encoder_placement_concurrent_qwen36"

    def _flush_engine_cache(self) -> None:
        """Drop the SGLang radix + mm preprocess caches before a wave.

        Same best-effort contract as the Qwen3-VL concurrent scenario,
        called ONCE per wave (before the barrier releases the users),
        not once per request: N concurrent per-request flushes would
        race each other and the drain-and-gate in the concurrent
        sidecar. Happens strictly before any t_request_start, so it is
        outside every timing window.
        """
        base_url = getattr(self.client, "base_url", None)
        if not base_url:
            return
        url = base_url.rstrip("/") + "/flush_cache"
        try:
            resp = requests.post(url, timeout=120)
            if resp.status_code >= 400:
                print(f"[EncPlC36] warning: flush_cache returned HTTP "
                      f"{resp.status_code}; engine caches were NOT reset "
                      f"(expect warm-prefix TTFT from wave 2 on)")
        except Exception as exc:
            print(f"[EncPlC36] warning: flush_cache failed ({exc}); engine "
                  f"caches were NOT reset (expect warm-prefix TTFT from "
                  f"wave 2 on)")

    def _warm_client_caches(self, model: str, mode: str,
                            vision_only: bool = True,
                            attn_impl: str = "sdpa") -> None:
        """Populate the client's process-global caches single-threaded.

        EncoderPlacementQwen36Client lazily imports transformers inside
        its cache-miss paths (_get_processor: AutoProcessor;
        _get_edge_model: the Qwen3_5* model classes). Under a
        concurrent wave, N user threads racing those first imports
        collide in sys.modules and surface spurious ImportErrors — an
        import-order artifact, not a request failure. Warming once here
        (main scenario thread, before the wave thread pool exists)
        means user threads only ever hit the populated caches.

        vision_only / attn_impl must match the user requests exactly:
        the edge-model cache is keyed on (path, vision_only, impl), so
        a mismatched warmup would warm the WRONG key and leave the
        users' key cold (import race back inside the wave). Server mode
        warms the processor only — same as before (vision_only /
        attn_impl are edge-only knobs and already warned about in
        run()).

        Best-effort by design: a warmup failure (e.g. broken model
        path) is printed as a warning and left to surface again, with
        full context, inside the actual user requests — the wave is
        NOT aborted, matching the scenario's failure-isolation
        semantics.
        """
        t0 = time.time()
        try:
            self.client._get_processor(model)
        except Exception as exc:
            print(f"[EncPlC36] warning: processor warmup failed ({exc}); "
                  f"each user request will retry the load")
        if mode == "edge":
            try:
                self.client._get_edge_model(
                    model, vision_only=vision_only, attn_impl=attn_impl)
            except Exception as exc:
                print(f"[EncPlC36] warning: edge-model warmup failed "
                      f"({exc}); each user request will retry the load")
        print(f"[EncPlC36] client caches warmed in {time.time() - t0:.1f} s "
              f"(processor"
              + (f" + edge visual tower"
                 if mode == "edge" else "")
              + f"; mode={mode})")

    def run(
        self,
        network_profile: str,
        run_index: int = 0
    ) -> ScenarioResult:
        session_id = self._create_session_id()
        model = self.config.get(
            "model", "../aicodec/aicodec/models/Qwen3.6-35B-A3B")
        # Anchor relative model paths to the aitestbed/ dir (the
        # scenario dir's parent — "../aicodec/..." is a sibling of
        # aitestbed/) so the runner works from any cwd; the client's
        # repo-root resolver misses this shape and falls back to
        # CWD-relative. Same anchoring as the serial Qwen3.6 scenario.
        if model and not os.path.isabs(model):
            here = os.path.dirname(os.path.abspath(__file__))
            anchored = os.path.normpath(
                os.path.join(os.path.dirname(here), model))
            if os.path.exists(anchored):
                model = anchored
        model_alias = self._anonymizer.model_alias(model)
        prompts = self.config.get("prompts", ["Describe this video in detail."])
        # Mode: env ENC_MODE_QWEN36 (falls back to ENC_MODE) wins over
        # the yaml key — same precedence as the serial Qwen3.6
        # scenario (the Qwen3-VL concurrent scenario reads config
        # only; Qwen3.6 keeps the env override for workflow parity
        # with its serial arm).
        mode = (os.environ.get("ENC_MODE_QWEN36")
                or os.environ.get("ENC_MODE")
                or self.config.get("mode", "server")).strip().lower()
        if mode not in ("server", "edge"):
            raise ValueError(
                f"invalid mode {mode!r} for "
                f"encoder_placement_concurrent_qwen36: "
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
        # on mismatch). Config key only; the runner maps --frame-ratio.
        frame_ratio_raw = self.config.get("frame_ratio")
        try:
            frame_ratio = (float(frame_ratio_raw)
                           if frame_ratio_raw not in (None, "") else None)
        except ValueError as exc:
            raise ValueError(
                f"invalid frame_ratio {frame_ratio_raw!r} for "
                f"encoder_placement_concurrent_qwen36: must be a number "
                f"in (0, 1]") from exc
        # yaml `frame_ratio: 0` / runner `--frame-ratio 0` disable
        # percentage sampling instead of raising (same folding as the
        # serial scenario / sidecar build_mm_process_config).
        if frame_ratio == 0.0:
            frame_ratio = None
        if frame_ratio is not None and not (0.0 < frame_ratio <= 1.0):
            raise ValueError(
                f"invalid frame_ratio {frame_ratio} for "
                f"encoder_placement_concurrent_qwen36: must be in (0, 1]")

        # decord/torchcodec decode-thread cap, BOTH modes (same knob as
        # the sidecar's --decode-threads). Edge: passed into each
        # worker's cfg (process mode) or exported here in the parent
        # before the wave (thread mode — workers inherit the parent's
        # env). A pre-set SGLANG_DECORD_NUM_THREADS env var wins
        # (explicit override), matching the serial scenario's contract.
        # Server mode ignores the key locally — the ENGINE-side cap is
        # the sidecar's own flag (warn once so a yaml set for edge runs
        # does not mislead server runs).
        decode_threads_raw = self.config.get("decode_threads")
        try:
            decode_threads = (int(decode_threads_raw)
                              if decode_threads_raw not in (None, "")
                              else None)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid decode_threads {decode_threads_raw!r} for "
                f"encoder_placement_concurrent_qwen36: must be a positive "
                f"integer") from exc
        if decode_threads is not None and decode_threads <= 0:
            raise ValueError(
                f"invalid decode_threads {decode_threads} for "
                f"encoder_placement_concurrent_qwen36: must be > 0 "
                f"(omit/null for native default)")
        # torch/OMP intra-op thread cap for the terminal's CPU-side
        # preprocessing (resize / HF processor). Same contract as
        # decode_threads: edge-only (server mode warns and ignores);
        # thread mode exports in the parent, process mode rides the
        # worker cfg (workers clamp it to their pin's core count).
        preprocess_threads_raw = self.config.get("preprocess_threads")
        try:
            preprocess_threads = (int(preprocess_threads_raw)
                                  if preprocess_threads_raw not in (None, "")
                                  else None)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid preprocess_threads {preprocess_threads_raw!r} "
                f"for encoder_placement_concurrent_qwen36: must be a "
                f"positive integer") from exc
        if preprocess_threads is not None and preprocess_threads <= 0:
            raise ValueError(
                f"invalid preprocess_threads {preprocess_threads} for "
                f"encoder_placement_concurrent_qwen36: must be > 0 "
                f"(omit/null to derive from the worker's edge-cpu pin)")

        # Wave size N: config concurrent_users, then 2. (No env vars —
        # the runner maps --users onto this config key; CLI > yaml.)
        try:
            concurrent_users = int(self.config.get("concurrent_users", 2))
        except ValueError as exc:
            raise ValueError(
                f"invalid concurrent_users for "
                f"encoder_placement_concurrent_qwen36: "
                f"must be a positive integer") from exc
        if concurrent_users < 1:
            raise ValueError(
                f"invalid concurrent_users {concurrent_users} for "
                f"encoder_placement_concurrent_qwen36: must be >= 1")
        # flush_cache: once per wave (cold-prefix semantics preserved;
        # see _flush_engine_cache). Set false to measure steady-state
        # cache-hit behaviour instead.
        flush_cache = bool(self.config.get("flush_cache", True))
        # Optional VidCom2 token compression on the edge bundle. Mirrors
        # the serial Qwen3.6 scenario's config surface exactly (config
        # keys only; the runner maps --compressor/--retention-ratio onto
        # them).
        compressor = str(self.config.get("compressor") or "").strip().lower()
        if compressor and compressor != "vidcom2":
            raise ValueError(
                f"invalid compressor {compressor!r} for "
                f"encoder_placement_concurrent_qwen36: must be '' or "
                f"'vidcom2'")
        try:
            retention_ratio = float(self.config.get("retention_ratio", 0.25))
        except ValueError as exc:
            raise ValueError(
                f"invalid retention_ratio for "
                f"encoder_placement_concurrent_qwen36: must be a number "
                f"in (0, 1]") from exc
        if not (0.0 < retention_ratio <= 1.0):
            raise ValueError(
                f"invalid retention_ratio {retention_ratio} for "
                f"encoder_placement_concurrent_qwen36: must be in (0, 1]")
        compression_active = compressor == "vidcom2" and mode == "edge"
        if retention_ratio != 0.25 and not compression_active:
            print(f"[EncPlC36] warning: retention_ratio={retention_ratio} "
                  f"is set but compression is inactive (needs edge mode + "
                  f"compressor=vidcom2); it will be ignored")
        # Two-tier retention needs vidcom2 exactly like bundle_quant —
        # first VALIDATE the env-or-config resolved range (env
        # ENCPL_OPTIONAL_RATIO wins, same as the serial scenario), then
        # warn (not raise) when compression is inactive so the serial
        # and concurrent scenarios reject the same bad values.
        try:
            _opt_ratio_cfg = float(
                (os.environ.get("ENCPL_OPTIONAL_RATIO")
                 or self.config.get("optional_ratio", 0.0) or 0.0))
        except ValueError as exc:
            raise ValueError(
                f"invalid optional_ratio for "
                f"encoder_placement_concurrent_qwen36: must be a number "
                f"in [0, retention_ratio)") from exc
        if _opt_ratio_cfg < 0.0 or _opt_ratio_cfg >= retention_ratio:
            raise ValueError(
                f"invalid optional_ratio {_opt_ratio_cfg} for "
                f"encoder_placement_concurrent_qwen36: must be in "
                f"[0, retention_ratio={retention_ratio})")
        if _opt_ratio_cfg > 0.0 and not compression_active:
            print(f"[EncPlC36] warning: optional_ratio={_opt_ratio_cfg} "
                  f"is set but compression is inactive (needs edge mode + "
                  f"compressor=vidcom2); it will be ignored")
        # Publish the VALIDATED value for _run_wave/_user (separate
        # methods — run()'s locals are not in scope there). Written on
        # every run() call so config/env changes between runs are
        # honored; the process-mode worker dict re-resolves env on its
        # own (spawn children do not inherit this attribute).
        self._resolved_optional_ratio = _opt_ratio_cfg

        # Optional fp8 quantization of the uploaded edge-bundle feature
        # tensor. Only valid TOGETHER with vidcom2: quantization applies
        # to the pruned bundle (compress first, quantize the survivors).
        bundle_quant = str(self.config.get("bundle_quant") or "").strip().lower()
        if bundle_quant and bundle_quant != "fp8":
            raise ValueError(
                f"invalid bundle_quant {bundle_quant!r} for "
                f"encoder_placement_concurrent_qwen36: must be '' or 'fp8'")
        if bundle_quant and not (compressor == "vidcom2" and mode == "edge"):
            raise ValueError(
                f"bundle_quant='fp8' requires edge mode with "
                f"compressor=vidcom2 (quantization targets the pruned "
                f"bundle); got mode={mode!r}, compressor={compressor!r}")

        # Vision-tower-only edge load — DEFAULT TRUE on Qwen3.6 (the
        # Qwen3-VL concurrent scenario defaults False because its 4B
        # full load is cheap; here the fp8 35B full load costs ~35 GB
        # VRAM per edge terminal just to obtain the ~0.5 GB tower, and
        # in PROCESS mode N full loads would not even fit — the
        # standalone tower is the only sane bench default). ~1 GB edge
        # VRAM per terminal. Edge mode only; ignored with a warning in
        # server mode.
        vision_only_raw = self.config.get("vision_only", True)
        if isinstance(vision_only_raw, str):
            vision_only = vision_only_raw.strip().lower() in (
                "1", "true", "yes", "on")
        else:
            vision_only = bool(vision_only_raw)
        if vision_only and mode != "edge":
            print(f"[EncPlC36] warning: vision_only={vision_only} is set "
                  f"but mode={mode!r} is not 'edge'; it will be ignored")

        # Edge ViT attention backend: 'sdpa' (default),
        # 'flash_attention_4', or 'eager' (see the client's
        # _resolve_attn_impl). Edge mode only (server mode lets
        # SGLang pick its own backend). Backends are not bit-identical;
        # compare across them with a tolerance and keep them in
        # separate DBs.
        attn_impl = self.config.get("attn_impl", "sdpa")
        try:
            from clients.encoder_placement_client_qwen36 import (
                _resolve_attn_impl as _norm_attn)
            attn_impl = _norm_attn(attn_impl)
        except ValueError as exc:
            raise ValueError(
                f"invalid attn_impl for "
                f"encoder_placement_concurrent_qwen36: {exc}") from exc

        # Process mode (either mode): run each user as an OS PROCESS
        # (own interpreter + own connection +, in edge mode, own ViT)
        # instead of a thread — N truly independent terminals.
        #   - edge mode: each worker loads its own ViT before 'ready'
        #     (no GIL/model sharing between terminals);
        #   - server mode: workers are pure HTTP upload+stream clients
        #     (no local model — warmup guard in
        #     encpl_edge_worker_qwen36.py skips it), so a process per
        #     user buys independent upload paths.
        # Config key only; the runner maps --enable-multi-processes.
        # DB note: metadata.edge_processes=true + mode field together
        # distinguish the semantics (edge=isolated ViT terminals,
        # server=isolated upload terminals); no separate column.
        # Process-isolated terminals by default (BOTH modes — edge:
        # isolated ViT per user; server: independent upload paths);
        # thread mode remains available via edge_processes=false.
        edge_processes_raw = self.config.get("edge_processes", True)
        if isinstance(edge_processes_raw, str):
            edge_processes = edge_processes_raw.strip().lower() in (
                "1", "true", "yes", "on")
        else:
            edge_processes = bool(edge_processes_raw)

        # decord decode-thread cap APPLICATION (validation happened
        # next to frame_ratio above): edge process mode hands the value
        # to each worker via cfg (workers set their own env — see
        # _apply_decode_threads in encpl_edge_worker_qwen36.py); thread
        # mode exports it HERE, in the parent, before the wave (all N
        # user threads share this process's env). setdefault: a pre-set
        # SGLANG_DECORD_NUM_THREADS wins (explicit override). Server
        # mode has no local decord — the ENGINE-side cap is the
        # sidecar's own --decode-threads flag (warn once so a yaml
        # written for edge runs does not mislead server runs).
        if decode_threads is not None and mode == "server":
            print(f"[EncPlC36] warning: decode_threads={decode_threads} "
                  f"is set but mode=server; the ENGINE-side cap is the "
                  f"sidecar's --decode-threads flag (restart it with the "
                  f"same value), this key is edge-only")
        elif decode_threads is not None and not edge_processes:
            os.environ.setdefault("SGLANG_DECORD_NUM_THREADS",
                                  str(decode_threads))
            print(f"[EncPlC36] decode_threads: capped to "
                  f"{decode_threads} (SGLANG_DECORD_NUM_THREADS; env name "
                  f"historical — caps BOTH backends; thread mode — "
                  f"inherited by all {concurrent_users} users)")

        # preprocess-threads cap APPLICATION (validation happened next to
        # decode_threads above): edge process mode hands the value to each
        # worker via cfg (workers apply min(N, pin cores) themselves — see
        # encpl_edge_worker_qwen36.py); thread mode exports the three env
        # vars HERE, in the parent, before the wave (all N user threads
        # share this process's env and its lazily-imported torch).
        # setdefault per var: a pre-set value wins (explicit override).
        # Server mode has no local preprocessing — the ENGINE-side cap is
        # the sidecar's own --preprocess-threads flag (warn once so a
        # yaml written for edge runs does not mislead server runs).
        if preprocess_threads is not None and mode == "server":
            print(f"[EncPlC36] warning: preprocess_threads="
                  f"{preprocess_threads} is set but mode=server; the "
                  f"ENGINE-side cap is the sidecar's --preprocess-threads "
                  f"flag (restart it with the same value), this key is "
                  f"edge-only")
        elif preprocess_threads is not None and not edge_processes:
            for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS"):
                os.environ.setdefault(_var, str(preprocess_threads))
            print(f"[EncPlC36] preprocess_threads: capped to "
                  f"{preprocess_threads} (OMP/MKL/OPENBLAS_NUM_THREADS; "
                  f"thread mode — inherited by all {concurrent_users} "
                  f"users; torch's own pool picks the env default at "
                  f"import)")

        # Terminal-side CPU budget (process mode only, either arm):
        # each worker process is pinned to `edge_cpus_per_worker` CPUs
        # from the `edge_cpu_ids` pool (round-robin), modelling N
        # users on real N-core terminals instead of N processes
        # roaming a many-core server host freely. Defaults ON (1 core
        # per worker) because the wave is exactly the comparison the
        # experiment wants: constrained terminals vs the (also
        # CPU-capped, see --server-cpus on the sidecar) server.
        # Config keys only; the runner maps --edge-cpus-per-worker /
        # --edge-cpu-ids. Thread mode cannot pin per-user: warn and
        # skip (the thread wave shares one process by design).
        edge_cpus_per_worker = int(
            self.config.get("edge_cpus_per_worker", 1) or 0)
        edge_cpu_ids_raw = self.config.get("edge_cpu_ids")
        edge_cpu_plan = None
        if edge_cpus_per_worker > 0:
            if not edge_processes:
                print(f"[EncPlC36] warning: edge_cpus_per_worker="
                      f"{edge_cpus_per_worker} requires process mode "
                      f"(--enable-multi-processes); thread mode "
                      f"shares one process and cannot pin per-user — "
                      f"skipping CPU pinning")
            elif not hasattr(os, "sched_setaffinity"):
                print(f"[EncPlC36] warning: os.sched_setaffinity "
                      f"unavailable on this platform; edge CPU pinning "
                      f"skipped")
            else:
                try:
                    # Default pool = SECOND half of own affinity
                    # ("auto-half-other"): the sidecar's --server-cpus
                    # auto-half pins the FIRST half, so the terminal
                    # fleet and the engine stay disjoint by default.
                    # An explicit spec behaves exactly as before.
                    pool = _parse_cpu_ids(
                        edge_cpu_ids_raw
                        if edge_cpu_ids_raw not in (None, "")
                        else "auto-half-other",
                        fallback=os.sched_getaffinity(0))
                except ValueError as exc:
                    raise ValueError(
                        f"invalid edge_cpu_ids for "
                        f"encoder_placement_concurrent_qwen36: "
                        f"{exc}") from exc
                edge_cpu_plan = {
                    "per_worker": edge_cpus_per_worker,
                    "pool": pool,
                }
        elif edge_cpu_ids_raw:
            print(f"[EncPlC36] warning: edge_cpu_ids given but "
                  f"edge_cpus_per_worker<=0; edge CPU pinning disabled")

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        try:
            video_paths = self._resolve_video_paths(concurrent_users)
        except ValueError as exc:
            raise ValueError(
                f"invalid video_paths for "
                f"encoder_placement_concurrent_qwen36: {exc}") from exc
        if video_paths is None:
            video_paths = None  # dataset mode may not need it

        # Optional dataset-bench mode (same contract as the qwen3
        # concurrent scenario): iterate a QA test set (MLVU MCQ /
        # Video-MME) as N PLAYLISTS — one barrier-aligned wave; the
        # barrier aligns only each user's FIRST video, then every user
        # runs its playlist serially (user u gets question u::N
        # round-robin) with no inter-video barrier. Per-user MCQ
        # prompts replace the shared prompt, each VIDEO's record
        # carries metadata.bench, and a failed video is one ok=False
        # entry whose playlist continues. Absent dataset/dataset_dir =
        # the historical shared-prompt waves (zero behaviour change).
        # Env ENCPL_DATASET_DIR wins over the yaml key.
        dataset = str(self.config.get("dataset", "") or "").strip().lower()
        dataset_dir = (os.environ.get("ENCPL_DATASET_DIR")
                       or self.config.get("dataset_dir"))
        if dataset and dataset not in ("mlvu", "videomme"):
            raise ValueError(
                f"invalid dataset {dataset!r} for "
                f"encoder_placement_concurrent_qwen36: must be 'mlvu' or "
                f"'videomme'")
        if dataset and not dataset_dir:
            raise ValueError(
                f"dataset {dataset!r} requires dataset_dir for "
                f"encoder_placement_concurrent_qwen36")
        limit = int(self.config.get("limit", 0) or 0)
        bench_tasks = self.config.get("tasks")
        bench_sample = self.config.get("sample")

        bench_items = None
        if dataset:
            bench_items = _load_bench_items(dataset, dataset_dir, limit,
                                            bench_tasks, bench_sample)
            if len(bench_items) < concurrent_users:
                raise ValueError(
                    f"dataset bench: {len(bench_items)} questions < "
                    f"concurrent_users={concurrent_users} — every user "
                    f"needs at least one video; lower concurrent_users "
                    f"or raise limit")
            print(f"[EncPlC36] dataset bench: {dataset} "
                  f"({len(bench_items)} questions as {concurrent_users} "
                  f"concurrent playlists"
                  f"{f', limit={limit}' if limit else ''})")
        elif video_paths is None:
            result.success = False
            result.error_message = (
                "No video_path configured for "
                "encoder_placement_concurrent_qwen36 scenario")
            return result

        # Distinct-video bookkeeping for the banner: K = distinct files
        # (shared fallback -> K=1); "shared" marks the video_path fallback.
        # Dataset mode: per-wave videos come from the bench items, so
        # the note is replaced by the dataset banner printed above.
        if video_paths is not None:
            n_distinct = len(set(video_paths))
            shared_note = (f" videos={n_distinct}"
                           + (" (shared)" if n_distinct < concurrent_users
                              else ""))
        else:
            shared_note = ""

        print(f"[EncPlC36] mode={mode} concurrent_users={concurrent_users}"
              f"{shared_note} "
              + (f" frame_ratio={frame_ratio}" if frame_ratio else "")
              + (f" compressor={compressor or 'none'} "
                 f"retention_ratio={retention_ratio}"
                 + (f" bundle_quant={bundle_quant}"
                    if bundle_quant else "")
                 if (mode == "edge") else "")
              + (f" vision_only={vision_only}"
                 if (vision_only and mode == "edge") else "")
              + (f" attn_impl={attn_impl}"
                 if (attn_impl != "sdpa" and mode == "edge") else "")
              + (f" multi_processes={concurrent_users}p"
                 if edge_processes else "")
              + (f" edge_cpus={edge_cpu_plan['per_worker']}c/worker"
                 f"@{len(edge_cpu_plan['pool'])}c-pool"
                 if edge_cpu_plan else ""))

        # Warm the client's process-global caches SINGLE-THREADED,
        # before any user thread exists (import-race guard; see
        # _warm_client_caches). The edge warmup passes the SAME
        # (vision_only, attn_impl) the user requests will — the model
        # cache is keyed on (path, vision_only, impl).
        #
        # PROCESS mode skips this warmup entirely: each worker process
        # loads its own model BEFORE signaling 'ready' (loading is
        # outside the timing window by protocol), so there is no
        # shared cache to pre-populate and no import race to avoid.
        if not edge_processes:
            self._warm_client_caches(
                model=model, mode=mode,
                vision_only=vision_only, attn_impl=attn_impl,
            )

        # In-memory correctness counters (dataset mode): only 4-option
        # MCQ entries with a ground-truth letter score; answers that
        # fail to parse stay out of the denominator (same semantics as
        # the serial scenario's bench summary).
        bench = {"scored": 0, "parse_ok": 0, "parse_fail": 0,
                 "correct": 0,
                 "total": len(bench_items) if bench_items else 0}

        # PLAYLIST plan: dataset mode = ONE barrier-aligned wave; the
        # barrier aligns only the FIRST video of each user, then every
        # user loops its playlist (user u gets bench_items[u::N],
        # deterministic round-robin — same assignment the old
        # wave-chunking produced: user u answered question w*N+u) with
        # NO inter-video barrier — a fast user NEVER waits for its
        # siblings (matches tools/encpl_bench_eval_concurrent*.py's
        # playlist semantics). A failed video is recorded (ok=False)
        # and the user CONTINUES with its next video.
        # Single-video mode = one wave per configured prompt (the
        # historical multi-wave loop, unchanged).
        bench_playlists = None
        if bench_items is not None:
            bench_playlists = [bench_items[u::concurrent_users]
                               for u in range(concurrent_users)]
            wave_plan = [None]  # single wave
        else:
            wave_plan = [None] * len(prompts)

        for wave_index in range(len(wave_plan)):
            if bench_playlists is not None:
                wave_prompts = [[item[1] for item in playlist]
                                for playlist in bench_playlists]
                wave_video_paths = [[str(item[0]) for item in playlist]
                                    for playlist in bench_playlists]
            else:
                wave_prompts = [prompts[wave_index]] * concurrent_users
                wave_video_paths = [str(p) for p in video_paths]
            self._wait_between_prompts(wave_index)
            # One flush per wave, BEFORE the users start: everyone in
            # the wave measures a cold prefix; nothing in-window pays
            # for it.
            if flush_cache:
                self._flush_engine_cache()

            wave_t0 = time.time()
            if edge_processes:
                user_records = self._run_wave_processes(
                    prompt=wave_prompts,
                    video_paths=wave_video_paths,
                    model=model,
                    mode=mode,
                    max_frames=max_frames,
                    max_tokens=max_tokens,
                    concurrent_users=concurrent_users,
                    compressor=compressor,
                    retention_ratio=retention_ratio,
                    bundle_quant=bundle_quant,
                    vision_only=vision_only,
                    attn_impl=attn_impl,
                    decode_threads=decode_threads,
                    preprocess_threads=preprocess_threads,
                    edge_cpu_plan=edge_cpu_plan,
                    session_id=session_id,
                    wave_index=wave_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    model_alias=model_alias,
                    frame_ratio=frame_ratio,
                )
            else:
                user_records = self._run_wave(
                    prompt=wave_prompts,
                    video_paths=wave_video_paths,
                    model=model,
                    mode=mode,
                    max_frames=max_frames,
                    max_tokens=max_tokens,
                    concurrent_users=concurrent_users,
                    compressor=compressor,
                    retention_ratio=retention_ratio,
                    bundle_quant=bundle_quant,
                    vision_only=vision_only,
                    attn_impl=attn_impl,
                    session_id=session_id,
                    wave_index=wave_index,
                    run_index=run_index,
                    network_profile=network_profile,
                    model_alias=model_alias,
                    frame_ratio=frame_ratio,
                )
            wall = time.time() - wave_t0

            # ---- aggregate the wave -----------------------------------
            # Playlist mode: user_records is a FLAT list of per-video
            # records (user u contributes len(playlist[u]) entries, each
            # tagged user_index + seq); a user's failed video is one
            # entry with ok=False, not a whole dead user.
            ok = [r for r in user_records if r.get("ok")]
            failed = [r for r in user_records if not r.get("ok")]
            ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
            # Truthiness filter (not "is not None"): an empty completion
            # records latency=0.0, which must stay OUT of the wave stats
            # just like analysis/metrics.py keeps it out of latency_mean.
            latencies = [r["latency"] for r in ok if r.get("latency")]

            def _pct(values, p):
                if not values:
                    return None
                values = sorted(values)
                if len(values) == 1:
                    return values[0]
                # Linear-interpolated percentile (same semantics as
                # analysis/metrics.py _percentile); the old round-to-
                # index version was order-statistics-only (n=2 p50 =
                # minimum).
                k = (len(values) - 1) * (p / 100.0)
                f = int(k)
                c = min(f + 1, len(values) - 1)
                if f == c:
                    return values[f]
                return values[f] + (values[c] - values[f]) * (k - f)

            wave_stats = {
                "wave_index": wave_index,
                "wave_size": concurrent_users,
                "wave_wall_sec": wall,
                "wave_ok": len(ok),
                "wave_failed": len(failed),
                # ok=True but ttft=None: the engine finished the request
                # with ZERO content tokens (greedy first-token EOS /
                # retract; see the sidecar's zero-token-completion
                # warning + finish_reason in each user's engine_stats).
                # Surfaced separately so the runner can show EMPTY
                # instead of FAIL and p50/p90 don't silently include
                # decode-only non-answers.
                "wave_empty": sum(
                    1 for r in ok if r.get("ttft") is None),
                # playlist mode: videos logged this wave
                "wave_videos": len(user_records),
                # per-user ok flags (user_index order): the runner's
                # summary table needs them to distinguish a real FAIL
                # (ok=False) from an EMPTY (ok=True, ttft=None).
                # Playlist mode: a user's list — None when the user
                # produced no record at all (error before any video).
                "ok_per_user": [
                    ([r.get("ok") for r in user_records
                      if r.get("user_index") == u]
                     if bench_playlists is not None
                     else bool(next(
                         (r.get("ok") for r in user_records
                          if r.get("user_index") == u), False)))
                    for u in range(concurrent_users)
                ],
                "ttft_mean": (statistics.fmean(ttfts) if ttfts else None),
                "ttft_p50": _pct(ttfts, 50),
                "ttft_p90": _pct(ttfts, 90),
                "ttft_max": max(ttfts) if ttfts else None,
                "latency_mean": (statistics.fmean(latencies)
                                 if latencies else None),
                # per-user TTFT/latency, user_index order — surfaced so
                # the runner can print them without re-reading the DB.
                # Playlist mode: a nested list (one per video, seq
                # order) so the runner's per-user lines stay readable.
                "ttft_per_user": [
                    ([r.get("ttft") for r in user_records
                      if r.get("user_index") == u]
                     if bench_playlists is not None
                     else next((r.get("ttft") for r in user_records
                                if r.get("user_index") == u), None))
                    for u in range(concurrent_users)
                ],
                "latency_per_user": [
                    ([r.get("latency") for r in user_records
                      if r.get("user_index") == u]
                     if bench_playlists is not None
                     else next((r.get("latency") for r in user_records
                                if r.get("user_index") == u), None))
                    for u in range(concurrent_users)
                ],
                # per-user connection 5-tuples (upload + chat/SSE), from
                # each user's engine_stats["conns"]; lets the runner
                # print them and a pcap be split per user by src_port.
                # Playlist mode: the FIRST video's conns (the
                # barrier-aligned start; later videos open new conns —
                # full detail stays in the DB rows).
                # None entries: user failed before the client captured
                # the tuple (error path) — runner prints "-".
                "conns_per_user": [
                    (next(((r.get("engine_stats") or {}).get("conns")
                           for r in user_records
                           if r.get("user_index") == u
                           and (r.get("engine_stats") or {}).get("conns")),
                          None))
                    for u in range(concurrent_users)
                ],
                # per-user FULL video path (user_index order — user i
                # watches video_paths[i]): surfaced so the runner's
                # per-user summary line pairs each TTFT with the video
                # that caused it (heterogeneous waves: the long-decode
                # user is identifiable at a glance). Playlist mode:
                # the user's playlist paths (list). None entries: user
                # record was built before a path was assigned (error
                # path) — runner omits the video line.
                "video_paths_per_user": (
                    [[r.get("video_path") for r in user_records
                      if r.get("user_index") == u]
                     for u in range(concurrent_users)]
                    if bench_playlists is not None else
                    [r.get("video_path") for r in user_records]
                ),
                # per-user upload size + time (from engine_stats, the
                # client's _upload_bundle meta): the runner's per-user
                # detail prints them so "how many bytes did user i
                # actually push" is readable without opening the DB.
                # Playlist mode: totals over the user's whole playlist.
                # upload_bytes prefers the true uploaded payload,
                # falling back to the raw video size (server mode).
                "upload_bytes_per_user": [
                    (sum(((r.get("engine_stats") or {}).get(
                              "uploaded_payload_bytes")
                          if (r.get("engine_stats") or {}).get(
                              "uploaded_payload_bytes") is not None
                          else (r.get("engine_stats") or {}).get(
                              "video_payload_bytes")) or 0
                         for r in user_records
                         if r.get("user_index") == u)
                     if bench_playlists is not None else
                     next((((r.get("engine_stats") or {}).get(
                                "uploaded_payload_bytes")
                            if (r.get("engine_stats") or {}).get(
                                "uploaded_payload_bytes") is not None
                            else (r.get("engine_stats") or {}).get(
                                "video_payload_bytes"))
                           for r in user_records
                           if r.get("user_index") == u), None))
                    for u in range(concurrent_users)
                ],
                "upload_time_per_user": [
                    (sum((r.get("engine_stats") or {}).get("upload_time")
                         or 0 for r in user_records
                         if r.get("user_index") == u)
                     if bench_playlists is not None else
                     next(((r.get("engine_stats") or {}).get("upload_time")
                           for r in user_records
                           if r.get("user_index") == u), None))
                    for u in range(concurrent_users)
                ],
            }
            result.metadata.setdefault("waves", []).append(wave_stats)
            ok_label = (f"ok={len(ok)}/{len(user_records)}"
                        if bench_playlists is not None
                        else f"ok={len(ok)}/{concurrent_users}")
            print(f"[EncPlC36] wave {wave_index} done in {wall:.1f} s | "
                  f"{ok_label}"
                  + (f" (empty={wave_stats['wave_empty']})"
                     if wave_stats["wave_empty"] else "")
                  + f" | "
                  f"TTFT mean={wave_stats['ttft_mean'] and round(wave_stats['ttft_mean'], 3)} s "
                  f"p50={wave_stats['ttft_p50'] and round(wave_stats['ttft_p50'], 3)} s "
                  f"p90={wave_stats['ttft_p90'] and round(wave_stats['ttft_p90'], 3)} s "
                  f"max={wave_stats['ttft_max'] and round(wave_stats['ttft_max'], 3)} s")

            # ---- persist per-video LogRecords (serial Qwen3.6 schema)
            # Playlist mode: one record PER VIDEO; turn_index =
            # user_index * 1000 + seq (unique, >= 0 — same convention
            # the concurrent bench tool uses; the wave*1000 layout
            # would collide when one user runs >1000 videos, and the
            # pcap-row filter turn_index < 0 stays untouched).
            for r in user_records:
                if bench_playlists is not None:
                    turn_index = (r["user_index"] * 1000
                                  + int(r.get("seq") or 0))
                else:
                    turn_index = wave_index * 1000 + r["user_index"]
                # Dataset mode: score this video's answer (same
                # extract_answer cascade as the serial scenario).
                # FAILED videos are not scored (serial semantics: a
                # failed turn carries no bench metadata). No padded
                # duplicates exist in playlist mode (the old final-wave
                # filler is gone); the bench.padded key stays false for
                # schema compatibility with older rows/consumers.
                bench_meta = None
                if bench_playlists is not None and r.get("ok"):
                    playlist = bench_playlists[r["user_index"]]
                    seq = int(r.get("seq") or 0)
                    if 0 <= seq < len(playlist):
                        entry = playlist[seq][2]
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
                                        r.get("total_content") or "") or "")
                                if model_letter:
                                    bench["parse_ok"] += 1
                                    if model_letter == answer_letter:
                                        bench["correct"] += 1
                                else:
                                    bench["parse_fail"] += 1
                            bench_meta = {
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
                                # always False in playlist mode; kept for
                                # schema compatibility with wave-chunked
                                # rows (aggregators skip padded=true)
                                "padded": False,
                            }
                    else:
                        print(f"[EncPlC36] warning: user {r['user_index']} "
                              f"reported seq={seq} outside its playlist "
                              f"(len={len(playlist)}); logged unscored")
                if r.get("ok"):
                    record = self._create_log_record(
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=network_profile,
                        request_bytes=r["request_bytes"],
                        response_bytes=r["response_bytes"],
                        tokens_in=r["tokens_in"],
                        tokens_out=r["tokens_out"],
                        t_request_start=r["t_request_start"],
                        t_first_token=r["t_first_chunk"],
                        t_last_token=r["t_last_chunk"],
                        latency_sec=r["latency"],
                        http_status=200,
                        success=True,
                        is_streaming=True,
                        chunk_count=r["chunk_count"],
                        inter_chunk_times=json.dumps(r["inter_chunk_times"]),
                        trace_request=r["request_payload"],
                        trace_response=r["response_payload"],
                        trace_events=r["response_events"],
                        trace_note="encoder_placement_concurrent_qwen36",
                         metadata=json.dumps({
                            # engine stats FIRST so the explicit keys
                            # below always win: a same-named key added
                            # to the client's engine_stats in the future
                            # must never silently overwrite the
                            # scenario's authoritative row fields
                            # (today's engine_stats keys — mode,
                            # video_path, upload_*, stage*, conns,
                            # e2e_* — collide only on mode/video_path,
                            # where both writers carry identical
                            # values, so this reordering is a no-op for
                            # current data).
                             **(r.get("engine_stats") or {}),
                             "model": model_alias,
                             "mode": mode,
                             # per-user video (video_paths[i]; identical
                             # across users only in the shared fallback)
                             "video_path": r.get("video_path"),
                             "wave_size": concurrent_users,
                             "user_index": r["user_index"],
                             # playlist mode: this video's position in the
                             # user's playlist (turn_index = user*1000+seq)
                             **({"seq": int(r.get("seq") or 0)}
                                if bench_playlists is not None else {}),
                             "wave_wall_sec": wall,
                             # barrier-release -> workers-joined wall;
                             # present only in edge_processes mode where
                             # wave_wall_sec includes spawn/model-load/
                             # teardown (see METRICS.md §2.3)
                             **({"wave_barrier_wall_sec":
                                 r["wave_barrier_wall_sec"]}
                                if r.get("wave_barrier_wall_sec")
                                is not None else {}),
                             # arm topology: threads vs OS processes per
                             # user (edge mode can run process-isolated
                             # terminals; server mode is always threads)
                             "edge_processes": bool(r.get("edge_processes")),
                             # terminal CPU budget actually applied by the
                             # worker (comma cpu list; null = unpinned)
                             "edge_cpus": r.get("edge_cpus"),
                             # explicit per-user metrics: directly SQL-
                             # queryable without decoding turn_index
                             # (ttft = t_first_token - t_request_start)
                             "ttft_sec": (None if r.get("ttft") is None
                                          else round(r["ttft"], 4)),
                             "latency_sec": (None if r.get("latency") is None
                                             else round(r["latency"], 4)),
                             # dataset mode: this user's scored MCQ answer
                             **({"bench": bench_meta}
                                if bench_meta is not None else {}),
                         }),
                    )
                else:
                    record = self._create_log_record(
                        session_id=session_id,
                        turn_index=turn_index,
                        run_index=run_index,
                        network_profile=network_profile,
                        t_request_start=r["t_request_start"],
                        latency_sec=r["latency"],
                        http_status=0,
                        error_type=r["error"],
                        success=False,
                        is_streaming=True,
                        metadata=json.dumps({
                            "model": model_alias,
                            "mode": mode,
                            # per-user video (video_paths[i]; identical
                            # across users only in the shared fallback)
                            "video_path": r.get("video_path"),
                            "wave_size": concurrent_users,
                            "user_index": r["user_index"],
                            **({"seq": int(r.get("seq") or 0)}
                               if bench_playlists is not None else {}),
                            "wave_wall_sec": wall,
                            "edge_processes": bool(r.get("edge_processes")),
                            "edge_cpus": r.get("edge_cpus"),
                        }),
                    )
                self.logger.log(record)
                result.log_records.append(record)

            result.turn_count += 1
            result.api_call_count += len(ok)
            result.total_latency_sec += wall
            result.total_request_bytes += sum(
                r.get("request_bytes") or 0 for r in ok)
            result.total_response_bytes += sum(
                r.get("response_bytes") or 0 for r in ok)
            tokens_in_total = sum(r.get("tokens_in") or 0 for r in ok)
            tokens_out_total = sum(r.get("tokens_out") or 0 for r in ok)
            result.total_tokens_in += tokens_in_total
            result.total_tokens_out += tokens_out_total
            # ttft_sec semantics for the concurrent scenario: the FIRST
            # wave's TTFT p50 = "typical user" TTFT under this wave size.
            # (min() would describe only the luckiest user; documented in
            # result.metadata so downstream consumers don't misread it.)
            if result.ttft_sec is None and wave_stats["ttft_p50"] is not None:
                result.ttft_sec = wave_stats["ttft_p50"]
                result.metadata["ttft_semantics"] = (
                    "first wave ttft_p50 (typical user under concurrency)")
            if failed:
                if bench_items is None:
                    result.success = False
                    result.error_message = (
                        f"wave {wave_index}: {len(failed)}/{concurrent_users} "
                        f"users failed; first error: {failed[0]['error']}")
                else:
                    # Dataset mode: one failed video does not abort the
                    # bench (serial semantics) — the failed question is
                    # recorded unscored and the user's playlist keeps
                    # going; the summary line carries the toll.
                    print(f"[EncPlC36] wave {wave_index}: "
                          f"{len(failed)}/{len(user_records)} videos failed "
                          f"({failed[0]['error']}); continuing")

        # Dataset mode summary (in-memory; per-question detail lives in
        # the DB log records' metadata.bench — same field name as the
        # serial scenario).
        if bench_items is not None:
            n_ok, n_c = bench["parse_ok"], bench["correct"]
            acc = n_c / n_ok if n_ok else 0.0
            print(f"[EncPlC36] bench summary: dataset={dataset} "
                  f"total={bench['total']} scored={bench['scored']} "
                  f"parse_ok={n_ok} parse_fail={bench['parse_fail']} "
                  f"correct={n_c} accuracy={acc:.3f}")
            # All-failed guard (same as the serial scenario): every
            # question's every user failed -> failed run, not a
            # zero-answer success.
            if bench["scored"] == 0 and result.success:
                result.success = False
                result.error_message = (
                    f"dataset bench: all {bench['total']} questions "
                    f"failed (0 scored)")

        return result

    def _run_wave(
        self,
        prompt: str,
        video_paths: list[str],
        model: str,
        mode: str,
        max_frames: int,
        max_tokens: int,
        concurrent_users: int,
        compressor: str,
        retention_ratio: float,
        bundle_quant: str,
        vision_only: bool,
        attn_impl: str,
        session_id: str,
        wave_index: int,
        run_index: int,
        network_profile: str,
        model_alias: str,
        frame_ratio: float | None = None,
    ) -> list[dict]:
        """Launch one barrier-aligned wave of N concurrent user THREADS.

        Each user thread runs the exact per-user flow the serial
        Qwen3.6 scenario runs once
        (client.generate_content_with_video is fully synchronous and
        self-contained) except for two per-user uniquifiers: user i
        streams video_paths[i] (distinct files when the video_paths
        config is used; the shared video_path fallback makes them
        identical) and a "[u<i>] " prompt tag, so no user can ride
        another's radix-cached prefill even against a radix-enabled
        sidecar (see the module docstring's prompt-uniquification
        note). The barrier aligns t_request_start across users so the
        wave hits the server together; per-user timing anchors are
        captured inside each thread (client-side clock, same
        convention as the serial scenario).

        Returns one dict per user (order = user_index) with ok/error,
        timing anchors, byte counts, engine_stats, video_path —
        everything the LogRecord assembly in run() needs.
        """
        import threading

        barrier = threading.Barrier(concurrent_users)
        results: list[dict | None] = [None] * concurrent_users
        # Playlist mode: prompt/video_paths are PER-USER LISTS OF LISTS
        # (user u runs playlist[u] serially after ONE barrier release);
        # single-video mode keeps the flat per-user lists (one video per
        # user, the historical shape).
        playlist_mode = (isinstance(prompt, (list, tuple))
                         and bool(prompt)
                         and isinstance(prompt[0], (list, tuple)))

        def _per_user_plan(user_index: int) -> list[dict]:
            """This user's request plan: one dict per video."""
            if playlist_mode:
                return [
                    {
                        "prompt": str(p),
                        "video_path": str(vp),
                        "seq": seq,
                    }
                    for seq, (p, vp) in enumerate(
                        zip(prompt[user_index], video_paths[user_index]))
                ]
            # Single-video mode: per-user prompt list or shared prompt.
            if isinstance(prompt, (list, tuple)):
                prompts_per_user = [str(p) for p in prompt]
                if len(prompts_per_user) != concurrent_users:
                    raise ValueError(
                        f"per-user prompt list has {len(prompts_per_user)} "
                        f"entries but concurrent_users={concurrent_users}")
            else:
                prompts_per_user = [prompt] * concurrent_users
            return [{
                "prompt": prompts_per_user[user_index],
                "video_path": video_paths[user_index],
                "seq": 0,
            }]

        plans = [_per_user_plan(u) for u in range(concurrent_users)]
        if playlist_mode:
            if len(prompt) != concurrent_users or \
                    len(video_paths) != concurrent_users:
                raise ValueError(
                    f"playlist plan has {len(prompt)} users x "
                    f"{len(video_paths)} video lists but "
                    f"concurrent_users={concurrent_users}")

        def _user(user_index: int):
            # Playlist mode returns ONE flat list of per-video records;
            # single-video mode keeps the historical single-record shape
            # (results[user_index] = one dict).
            if playlist_mode:
                user_recs: list[dict] = []
            else:
                user_recs = None  # type: ignore[assignment]
            rec: dict = {}
            # resolved tiered knobs (env-or-config, validated once in
            # run() above — the same values every user passes below) +
            # the one-shot BUG N single-tier warning latch for this
            # user thread
            optional_ratio_cfg = getattr(
                self, "_resolved_optional_ratio", 0.0)
            warned_single_tier = False

            def _base_rec(seq: int, video_path) -> dict:
                return {
                    "user_index": user_index,
                    "seq": seq,
                    "ok": False,
                    "t_request_start": None,
                    "ttft": None,
                    "latency": 0.0,
                    # per-user video (distinct with video_paths config;
                    # identical in the shared video_path fallback)
                    "video_path": video_path,
                }

            # All users share one t_request_start semantic for their
            # FIRST video: capture it AFTER the barrier so upload/edge
            # work stays strictly inside the window (same convention as
            # the serial scenario, where t_request_start precedes all
            # client work). Later playlist videos capture their own
            # anchors AFTER the previous response lands (no
            # inter-video barrier — a fast user never waits for its
            # siblings).
            barrier.wait()
            for item in plans[user_index]:
                rec = _base_rec(item["seq"], item["video_path"])
                if playlist_mode:
                    user_recs.append(rec)  # type: ignore[union-attr]
                try:
                    t_request_start = time.time()
                    # Publish the anchor immediately so the except path
                    # can measure a true failure latency (elapsed since
                    # the request start), not a zero-length window.
                    rec["t_request_start"] = t_request_start

                    response = self.client.generate_content_with_video(
                        prompt=item["prompt"],
                        video_path=item["video_path"],
                        model=model,
                        mode=mode,
                        max_frames=max_frames,
                        max_tokens=max_tokens,
                        compressor=compressor if mode == "edge" else "",
                        retention_ratio=retention_ratio,
                        bundle_quant=bundle_quant if mode == "edge" else "",
                        vision_only=vision_only if mode == "edge" else False,
                        attn_impl=attn_impl if mode == "edge" else "sdpa",
                        user_tag=(
                            f"[u{user_index}] [v{item['seq']}] "
                            if playlist_mode else f"[u{user_index}] "),
                        frame_ratio=frame_ratio,
                        optional_ratio=(optional_ratio_cfg
                                        if mode == "edge" else 0.0),
                        opt_tokens_per_packet=int(
                            (os.environ.get("ENCPL_OPT_TOKENS_PER_PACKET")
                             or self.config.get("opt_tokens_per_packet", 64)
                             or 64)),
                        opt_send_gap_ms=(
                            float((os.environ.get("ENCPL_OPT_SEND_GAP_MS")
                                   or self.config.get("opt_send_gap_ms", 0.05)
                                   or 0.0))),
                        opt_chat_delay_ms=(
                            float((os.environ.get("ENCPL_OPT_CHAT_DELAY_MS")
                                   or self.config.get("opt_chat_delay_ms", 0.0)
                                   or 0.0))),
                    )

                    # StreamingResponse token fields are tokens_in/out (the
                    # OpenAI-style prompt/completion_tokens names exist only
                    # on the wire chunks, never on the response object).
                    tokens_in = response.tokens_in
                    tokens_out = response.tokens_out
                    if tokens_in is None:
                        tokens_in = self.client.estimate_tokens(
                            item["prompt"], model)
                    if tokens_out is None:
                        tokens_out = self.client.estimate_tokens(
                            response.total_content, model)

                    engine_stats = (getattr(response, "engine_stats", {})
                                    or {})

                    # BUG N visibility: a tiered-configured run
                    # (optional_ratio > 0) whose compression early-exited
                    # is SINGLE-TIER in reality — the must bundle rides
                    # the first wave only and no UDP wave ever happens.
                    # Warn once per process/thread wave so a mislabeled
                    # experiment is visible in the console.
                    if (mode == "edge" and optional_ratio_cfg > 0
                            and not engine_stats.get("tiered")):
                        if not warned_single_tier:
                            warned_single_tier = True
                            print(f"[EncPlC36] warning: optional_ratio="
                                  f"{optional_ratio_cfg} requested but this "
                                  f"wave is effectively SINGLE-TIER (vidcom2 "
                                  f"selection early-exited / compression "
                                  f"inactive); no optional UDP wave is "
                                  f"happening — accuracy comparisons against "
                                  f"a real tiered run would be wrong")

                    # E2E timing anchors identical to the serial scenario:
                    # t_request_start was captured BEFORE the client's
                    # edge/upload work, so TTFT and latency_sec span video
                    # input -> first/last token in both modes.
                    t_last_chunk = response.t_last_chunk
                    latency_sec = (
                        t_last_chunk - t_request_start
                        if t_last_chunk is not None else 0.0)

                    rec.update({
                        "ok": True,
                        "t_request_start": t_request_start,
                        "t_first_chunk": response.t_first_chunk,
                        "t_last_chunk": t_last_chunk,
                        "ttft": (response.t_first_chunk - t_request_start
                                 if response.t_first_chunk is not None
                                 else None),
                        "latency": latency_sec,
                        "request_bytes": response.request_bytes,
                        "response_bytes": response.response_bytes,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "chunk_count": len(response.chunks),
                        "inter_chunk_times": response.inter_chunk_times,
                        "request_payload": response.request_payload,
                        "response_payload": response.response_payload,
                        "response_events": response.response_events,
                        "engine_stats": engine_stats,
                        # full generated text — the dataset-mode scoring
                        # (metadata.bench) reads it after the wave
                        "total_content": response.total_content,
                    })
                except BaseException as exc:  # noqa: BLE001 — record, keep going
                    t_fail = time.time()
                    t_request_start = rec.get("t_request_start") or t_fail
                    rec.update({
                        "ok": False,
                        "error": str(exc),
                        "t_request_start": t_request_start,
                        "latency": t_fail - t_request_start,
                    })
                    # Playlist mode: one failed video does not kill the
                    # user's remaining playlist (same semantics the
                    # process worker has) — the loop continues.
                    # Single-video mode: the loop had exactly one item,
                    # so this is the historical fail-and-stop.
                finally:
                    if playlist_mode:
                        results[user_index] = user_recs  # type: ignore[assignment]
                    else:
                        results[user_index] = rec

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(_user, i)
                       for i in range(concurrent_users)]
            for future in as_completed(futures):
                # _user never raises (it records its own failures), but
                # keep the wait deterministic: every future completes
                # before the wave is aggregated.
                future.result()

        # Playlist mode: flatten per-user lists (user_index then seq
        # order); single-video mode keeps the historical per-user list.
        if playlist_mode:
            return [r for user_recs in results if user_recs
                    for r in user_recs]
        return [r for r in results if r is not None]

    def _run_wave_processes(
        self,
        prompt: str | list[str],
        video_paths: list[str],
        model: str,
        mode: str,
        max_frames: int,
        max_tokens: int,
        concurrent_users: int,
        compressor: str,
        retention_ratio: float,
        bundle_quant: str,
        vision_only: bool,
        attn_impl: str,
        edge_cpu_plan: dict | None,
        session_id: str,
        wave_index: int,
        run_index: int,
        network_profile: str,
        model_alias: str,
        frame_ratio: float | None = None,
        decode_threads: int | None = None,
        preprocess_threads: int | None = None,
    ) -> list[dict]:
        """Launch one barrier-aligned wave of N PROCESS-isolated users.

        Same contract as _run_wave (one record dict per user, exact
        same schema) but each user is an OS process running
        clients/encpl_edge_worker_qwen36.py: own interpreter, own GIL,
        own edge ViT instance, own HTTP connection — N truly
        independent terminals. What stays shared is the GPU itself
        (kernel interleave; a single-host limitation, noted in the
        module docstring and the DB metadata).

        decode_threads rides the worker cfg (each worker exports its
        own SGLANG_DECORD_NUM_THREADS before its first VideoReader —
        see _apply_decode_threads in the worker module);
        preprocess_threads rides it too (workers clamp it to their
        edge-cpu pin's core count — see _apply_cpu_affinity in the
        worker module); thread mode instead exports both once in the
        parent (see run()).

        Synchronization: an mp.Barrier (parties = N workers + 1
        parent) is created here and passed to each worker via its
        config JSON as a multiprocessing handle (BarrierProxy through
        a small shared ctx.Manager — picklable, portable across
        spawn). Workers send 'ready' AFTER loading their model, then
        block in barrier.wait(); the parent waits for N 'ready'
        messages, joins the barrier (releasing the wave), then drains
        each worker's messages to its final 'done'/'error' record and
        waits for process exit. Model loading and CUDA context
        creation are OUTSIDE the timing window; t_request_start is
        captured after barrier.wait() inside the worker — the same
        convention the thread wave uses.

        Reading: each worker's message pipe is drained on a dedicated
        reader THREAD (a worker emits several protocol messages while
        the parent tends its siblings; an undrained OS pipe would fill
        and deadlock the worker mid-wave). Plain stdout is inherited
        untouched (library prints reach the parent's console).
        """
        import multiprocessing as mp
        import threading as _threading

        from clients.encpl_edge_worker_qwen36 import _worker_entry

        # Dataset PLAYLIST mode passes per-user LISTS OF LISTS (user u
        # runs playlist[u] serially — same convention as _run_wave).
        # Single-video mode keeps the flat per-user prompt list /
        # shared prompt. The worker cfg gets a 'playlist' key either
        # way (the worker's native protocol; a 1-item playlist is
        # bit-compatible with the legacy single-request path).
        playlist_mode = (isinstance(prompt, (list, tuple))
                         and bool(prompt)
                         and isinstance(prompt[0], (list, tuple)))
        if playlist_mode:
            if len(prompt) != concurrent_users or \
                    len(video_paths) != concurrent_users:
                raise ValueError(
                    f"playlist plan has {len(prompt)} users x "
                    f"{len(video_paths)} video lists but "
                    f"concurrent_users={concurrent_users}")
        else:
            if isinstance(prompt, (list, tuple)):
                prompts_per_user = [str(p) for p in prompt]
                if len(prompts_per_user) != concurrent_users:
                    raise ValueError(
                        f"per-user prompt list has {len(prompts_per_user)} "
                        f"entries but concurrent_users={concurrent_users}")
            else:
                prompts_per_user = [prompt] * concurrent_users

        # Resolve the repo root the worker needs on sys.path: this
        # scenario file lives in <root>/aitestbed/scenarios/, and the
        # worker imports clients.* (sibling package).
        aitestbed_root = str(Path(__file__).resolve().parent.parent)

        # Terminal CPU assignment (edge process mode): round-robin
        # `per_worker` cpus from the pool per user_index. When the
        # pool is smaller than the demand the assignments OVERLAP —
        # a warning (N users share a smaller terminal fleet) but the
        # measurement stays meaningful (that sharing is exactly what
        # a real over-subscribed terminal fleet does to the wave).
        cpu_assignments = {}
        if edge_cpu_plan:
            pool = edge_cpu_plan["pool"]
            per = edge_cpu_plan["per_worker"]
            if pool and concurrent_users * per > len(pool):
                print(f"[EncPlC36] warning: edge cpu pool ({len(pool)} cpus) "
                      f"< demand ({concurrent_users} users x {per} cpu); "
                      f"workers will share cpus round-robin")
            for user_index in range(concurrent_users):
                cpu_assignments[user_index] = [
                    pool[(user_index * per + k) % len(pool)]
                    for k in range(per)
                ] if pool else []

        ctx = mp.get_context("spawn")  # CUDA-safe; POSIX default differs
        # Parties = N workers + this parent: the parent joins only
        # after every worker signaled 'ready', so model loads and CUDA
        # context creation never land inside a timing window.
        barrier = ctx.Barrier(concurrent_users + 1,
                              timeout=self.config.get(
                                  "barrier_timeout", 600))

        # Message channel per worker: a duplex Pipe (picklable across
        # spawn). The worker sends protocol messages through it; its
        # plain stdout (library prints) is inherited untouched and
        # appears on the parent's console — good enough for debugging
        # and avoids per-process stdout plumbing entirely.
        # records[user_index] is a LIST: playlist mode emits one 'done'
        # per video (the reader appends in arrival order; run() sorts
        # by nothing — seq is on each record); single-video mode has
        # exactly one 'done' per worker.
        records: dict[int, list[dict]] = {}
        errors: dict[int, str] = {}
        ready_count = [0]
        error_seen = _threading.Event()
        ready_cv = _threading.Condition()

        def _reader(pipe, user_index: int):
            """Drain ONE worker's message pipe; count 'ready', keep
            every 'done' record (playlist mode: one per video) and the
            final 'error' message. Runs on a dedicated thread: the
            worker sends several messages while the parent tends its
            siblings, and an undrained pipe would backpressure (and
            block) the worker mid-wave.

            Race note: 'error'/'done' entries are written into the
            shared dicts IMMEDIATELY (under ready_cv), not in the
            finally block — the startup wait checks errors{} as soon
            as error_seen is set, and the pipe may stay open for
            minutes after the error (worker teardown), which would
            otherwise leave errors{} empty and the failure cause
            replaced by 'unknown worker error'."""
            n_done = 0
            last_error = None
            try:
                while True:
                    try:
                        msg = pipe.recv()
                    except EOFError:
                        break  # worker closed its end (normal exit)
                    if msg is None:
                        break  # explicit shutdown marker
                    mtype = msg.get("type")
                    if mtype == "ready":
                        with ready_cv:
                            ready_count[0] += 1
                            ready_cv.notify_all()
                    elif mtype == "done":
                        rec = msg.get("record") or {}
                        n_done += 1
                        with ready_cv:
                            records.setdefault(user_index, []).append(rec)
                    elif mtype == "error":
                        last_error = msg.get("error") or "worker error"
                        with ready_cv:
                            errors[user_index] = last_error
                            error_seen.set()
                            ready_cv.notify_all()
            except Exception as exc:
                last_error = f"reader: {exc}"
                with ready_cv:
                    errors[user_index] = last_error
                    error_seen.set()
                    ready_cv.notify_all()
            finally:
                # A worker that finished playlist items but crashed
                # before EOF keeps its already-appended records; a
                # worker that produced NO done records and NO error is
                # the crash case handled by the caller.
                if n_done == 0 and last_error is None:
                    with ready_cv:
                        errors.setdefault(
                            user_index,
                            "worker produced no records (crashed or "
                            "timed out)")
                if last_error is not None:
                    errors.setdefault(user_index, last_error)

        procs = []
        readers = []
        try:
            for user_index in range(concurrent_users):
                parent_end, child_end = ctx.Pipe(duplex=True)
                cfg = {
                    "user_index": user_index,
                    "user_tag": f"[u{user_index}] ",
                    "model": model,
                    "mode": mode,
                    "max_frames": max_frames,
                    "max_tokens": max_tokens,
                    "compressor": compressor,
                    "retention_ratio": retention_ratio,
                    "bundle_quant": bundle_quant,
                    "vision_only": vision_only,
                    "attn_impl": attn_impl,
                    "frame_ratio": frame_ratio,
                    # Two-tier retention (edge + vidcom2): must tier
                    # rides the reliable first wave; optional tier
                    # streams best-effort UDP parts between the must
                    # upload and the chat POST (the server snapshots
                    # them at chat-arrival merge).
                    "optional_ratio": (
                        (os.environ.get("ENCPL_OPTIONAL_RATIO")
                         or self.config.get("optional_ratio", 0.0) or 0.0)),
                    "opt_tokens_per_packet": int(
                        (os.environ.get("ENCPL_OPT_TOKENS_PER_PACKET")
                         or self.config.get("opt_tokens_per_packet", 64)
                         or 64)),
                    "opt_send_gap_ms": float(
                        (os.environ.get("ENCPL_OPT_SEND_GAP_MS")
                         or self.config.get("opt_send_gap_ms", 0.05)
                         or 0.0)),
                    "opt_chat_delay_ms": float(
                        (os.environ.get("ENCPL_OPT_CHAT_DELAY_MS")
                         or self.config.get("opt_chat_delay_ms", 0.0)
                         or 0.0)),
                    "base_url": getattr(self.client, "base_url", None),
                    # Stream safety timeouts, forwarded from the PARENT
                    # client so the runner's flags (e.g. concurrent
                    # first-event 900 vs the serial class default 200)
                    # survive the spawn boundary. Without these keys the
                    # worker's fresh client silently falls back to the
                    # class defaults — the 2026-10-20 14:17 run lost
                    # 2/24 requests to exactly that (read-timeout at
                    # 200.6 s for tail-of-wave users whose legitimate
                    # first event was still queued behind prefills).
                    "stream_connect_timeout": getattr(
                        self.client, "stream_connect_timeout", None),
                    "stream_first_event_timeout": getattr(
                        self.client, "stream_first_event_timeout", None),
                    "stream_idle_timeout": getattr(
                        self.client, "stream_idle_timeout", None),
                    "stream_upload_read_timeout": getattr(
                        self.client, "stream_upload_read_timeout", None),
                    "aitestbed_root": aitestbed_root,
                    # test-harness passthrough: mocks without the heavy
                    # deps skip the per-worker model warmup and/or
                    # replace the worker's client class with a shim
                    # module file (spawn children re-import in a fresh
                    # interpreter). Both empty in production.
                    "skip_warmup": bool(self.config.get("skip_warmup")),
                    "client_shim_path": self.config.get("client_shim_path"),
                    # Terminal CPU budget (round-robin from the pool;
                    # empty/absent = unpinned, decided by the parent).
                    "edge_cpus": cpu_assignments.get(user_index),
                    # decord/torchcodec decode-thread cap for THIS
                    # worker (exported as SGLANG_DECORD_NUM_THREADS
                    # before the first VideoReader; see
                    # _apply_decode_threads in the worker module).
                    "decode_threads": decode_threads,
                    # torch/OMP intra-op thread cap for this worker's
                    # CPU-side preprocessing (resize / HF processor);
                    # the worker clamps it to its edge-cpu pin's core
                    # count (min(N, K)) — see _apply_cpu_affinity in
                    # encpl_edge_worker_qwen36.py. None = derive from
                    # the pin (current behaviour).
                    "preprocess_threads": preprocess_threads,
                }
                # Request plan: playlist mode hands the worker its
                # per-video list (the worker's native playlist protocol
                # — one barrier release, then serial per-video requests
                # with per-item records); single-video mode keeps the
                # legacy single-request keys bit-for-bit.
                if playlist_mode:
                    cfg["playlist"] = [
                        {
                            "prompt": str(p),
                            "video_path": str(vp),
                            "user_tag": f"[u{user_index}] [v{seq}] ",
                            "seq": seq,
                        }
                        for seq, (p, vp) in enumerate(
                            zip(prompt[user_index],
                                video_paths[user_index]))
                    ]
                else:
                    cfg["prompt"] = prompts_per_user[user_index]
                    # per-user video (distinct with the video_paths
                    # config; identical in the shared fallback)
                    cfg["video_path"] = video_paths[user_index]
                proc = ctx.Process(
                    target=_worker_entry,
                    args=(cfg, barrier, child_end),
                    name=f"encpl-edge-worker-qwen36-{user_index}",
                    daemon=True,
                )
                proc.start()
                child_end.close()  # parent keeps only its side
                procs.append(proc)
                reader = _threading.Thread(
                    target=_reader, args=(parent_end, user_index),
                    name=f"encpl-edge-reader-qwen36-{user_index}", daemon=True)
                reader.start()
                readers.append((reader, parent_end))

            # Wait for every worker's 'ready' (model loaded, worker at
            # the barrier). A worker that reports an 'error' message
            # or dies before 'ready' fails this wait FAST with the
            # real cause instead of hanging until the timeout.
            deadline = time.time() + self.config.get(
                "worker_ready_timeout", 900)
            while True:
                with ready_cv:
                    if ready_count[0] >= concurrent_users:
                        break
                    if error_seen.is_set():
                        first_err = (list(errors.values())[0]
                                     if errors else "unknown worker error")
                        raise RuntimeError(
                            f"edge worker failed during startup: "
                            f"{first_err}")
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"only {ready_count[0]}/{concurrent_users} "
                            f"edge workers became ready within "
                            f"{self.config.get('worker_ready_timeout', 900)} s")
                    ready_cv.wait(timeout=min(remaining, 5.0))
                if any(not p.is_alive() and p.exitcode != 0
                       for p in procs):
                    dead = [p.name for p in procs
                            if not p.is_alive() and p.exitcode != 0]
                    raise RuntimeError(
                        f"edge worker(s) died during startup: {dead}; "
                        f"check the console output above")

            # Release the wave (parent is the last barrier party).
            barrier.wait()
            t_barrier_release = time.time()

            # Wait for all workers to finish their request and exit.
            for proc in procs:
                proc.join(timeout=self.config.get(
                    "worker_join_timeout", 1200))
            for reader, _ in readers:
                reader.join(timeout=30)
            t_workers_done = time.time()
            # Barrier-to-drain wall: excludes spawn/import/model-load
            # and process teardown — comparable with the thread wave's
            # wave_wall_sec (see METRICS.md §2.3).
            wave_barrier_wall = t_workers_done - t_barrier_release
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=10)
            for _, pipe in readers:
                try:
                    pipe.close()
                except Exception:
                    pass

        # Assemble per-VIDEO records in user_index/seq order — identical
        # schema to the thread wave, plus the process-mode marker.
        # Playlist mode: records[user_index] is a LIST of per-video
        # records; single-video mode: a 1-item list (normalized here).
        out: list[dict] = []
        for user_index in range(concurrent_users):
            user_records = records.get(user_index) or []
            if user_records:
                for rec in user_records:
                    rec = dict(rec)
                    rec.setdefault("user_index", user_index)
                    rec.setdefault("seq", 0)
                    rec["edge_processes"] = True
                    # Barrier-release -> all-workers-joined wall: the
                    # spawn/load/teardown-free twin of wave_wall_sec.
                    rec["wave_barrier_wall_sec"] = wave_barrier_wall
                    # Per-user video: the worker's record may lack it
                    # (shim/mock records); the parent knows the
                    # assignment (playlist mode: that video's path), so
                    # back-fill for a self-describing row either way.
                    if playlist_mode:
                        if 0 <= rec["seq"] < len(video_paths[user_index]):
                            rec.setdefault(
                                "video_path",
                                video_paths[user_index][rec["seq"]])
                    else:
                        rec.setdefault("video_path",
                                       video_paths[user_index])
                    # Normalized cpu spec (the worker stamps its own on
                    # success; crashed-before-record workers still get the
                    # parent's planned assignment so every row is
                    # self-describing).
                    if rec.get("edge_cpus") is None:
                        planned = cpu_assignments.get(user_index)
                        if planned:
                            rec["edge_cpus"] = ",".join(
                                str(c) for c in planned)
                    out.append(rec)
            else:
                # Worker produced NO records: crash, startup error, or
                # timeout. Playlist mode: one failed placeholder PER
                # PLANNED VIDEO (the questions exist and must be logged
                # unscored, mirroring the thread path's per-video
                # records); single-video mode: the historical one.
                err = errors.get(
                    user_index,
                    "worker produced no record (crashed or timed out)")
                planned = cpu_assignments.get(user_index)
                if playlist_mode:
                    for seq in range(len(video_paths[user_index])):
                        out.append({
                            "user_index": user_index,
                            "seq": seq,
                            "ok": False,
                            "error": err,
                            "t_request_start": None,
                            "ttft": None,
                            "latency": 0.0,
                            "video_path":
                                video_paths[user_index][seq],
                            "edge_processes": True,
                            "wave_barrier_wall_sec": wave_barrier_wall,
                            "edge_cpus": (
                                ",".join(str(c) for c in planned)
                                if planned else None),
                        })
                else:
                    out.append({
                        "user_index": user_index,
                        "seq": 0,
                        "ok": False,
                        "error": err,
                        "t_request_start": None,
                        "ttft": None,
                        "latency": 0.0,
                        "video_path": video_paths[user_index],
                        "edge_processes": True,
                        "wave_barrier_wall_sec": wave_barrier_wall,
                        "edge_cpus": (",".join(str(c) for c in planned)
                                      if planned else None),
                    })
        return out

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
            print(f"[EncPlC36] warning: video path does not exist: {path}")
            return None
        return path

    def _resolve_video_paths(self, concurrent_users: int) -> list[Path] | None:
        """Resolve the per-user video list for a wave of N users.

        Priority: config key `video_paths` (list) > legacy `video_path`
        (single file shared by all N users, the original behavior).

        video_paths is STRICT: len must equal concurrent_users exactly
        — a miscount aborts the run (raise ValueError to the caller)
        instead of silently re-sharing or dropping files. Distinct
        files are the point of the key; N copies of one path is the
        caller's mistake to make explicitly. Each entry resolves via
        the same expanduser / video_base_dir / exists rules as
        _resolve_video_path; ANY missing file fails the whole run
        (None return -> "No video_path configured" abort in run()),
        because a wave that silently re-shares the survivors would
        corrupt the per-user comparison.

        Returns a list of N resolved Paths, or None when no usable
        video source is configured.
        """
        raw_list = self.config.get("video_paths")
        if not raw_list:
            # Legacy fallback: one shared video for all N users.
            shared = self._resolve_video_path()
            if shared is None:
                return None
            return [shared] * concurrent_users
        if not isinstance(raw_list, (list, tuple)):
            raise ValueError(
                f"video_paths must be a list, got {type(raw_list).__name__}")
        if len(raw_list) != concurrent_users:
            raise ValueError(
                f"video_paths has {len(raw_list)} entries but "
                f"concurrent_users={concurrent_users} — strict "
                f"one-video-per-user requires exactly N entries")
        base_dir = self.config.get("video_base_dir")
        out: list[Path] = []
        for i, raw in enumerate(raw_list):
            path = Path(raw).expanduser()
            if not path.is_absolute() and base_dir:
                path = Path(base_dir).expanduser() / path
            path = path.resolve()
            if not path.exists():
                print(f"[EncPlC36] warning: video path does not exist: "
                      f"{path}")
                return None
            out.append(path)
        return out
