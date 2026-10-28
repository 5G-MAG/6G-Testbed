#!/usr/bin/env python3
"""
encpl_edge_worker_qwen36.py — ONE terminal user of a Qwen3.6
encoder-placement concurrent WAVE, as an OS PROCESS (not a thread).

Standalone sibling of clients/encpl_edge_worker.py (the Qwen3-VL
worker) for the Qwen3.6 experiment — zero changes to that file, so
both experiments can run/debug in parallel. The ONLY differences from
the Qwen3-VL worker: it instantiates EncoderPlacementQwen36Client
(clients/encoder_placement_client_qwen36.py) and therefore passes the
Qwen3.6 client's generate_content_with_video signature — which now
carries the SAME compressor / retention_ratio / bundle_quant /
frame_ratio knobs as the Qwen3-VL client (ported; see that client's
module docstring). The ready/barrier/done/error protocol, the CPU
pinning, and the record schema are IDENTICAL, so the Qwen3.6
concurrent scenario's _run_wave_processes can treat either worker the
same way.

Why processes: the thread-based wave shares one Python process among
all N "terminals", so they share the GIL AND one edge ViT model object
(process-global cache in EncoderPlacementQwen36Client) — terminal-side
ViT work serializes even though the server side is concurrent. N
processes = N truly independent terminals: own interpreter, own GIL,
own model instance, own HTTP connection. What remains shared is only
the GPU itself (kernel-level interleave, unavoidable single-host).

VRAM note (Qwen3.6 specifics): with vision_only=True (the Qwen3.6
concurrent default — the 35B fp8 full load costs ~35 GB per process
just to obtain the ~0.5 GB tower), each worker holds ~1 GB of edge
VRAM, so N workers fit alongside the engine; vision_only=False costs
~35 GB PER worker and is only for multi-GPU hosts or tiny N.

Spawned by scenarios/encoder_placement_concurrent_qwen36.py's (and
tools/encpl_bench_eval_concurrent_qwen36.py's) _run_wave_processes
via mp.get_context("spawn").Process with:

    target=_worker_entry
    args=(cfg, barrier, msg_pipe)

  cfg       dict — every generate_content_with_video knob plus
                   user_index / base_url / aitestbed_root /
                   edge_cpus (terminal CPU pin) / decode_threads
                   (decord/torchcodec thread cap; exported as
                   SGLANG_DECORD_NUM_THREADS by _apply_decode_threads
                   before the worker's first VideoReader), and EITHER
                   the single-request keys (prompt / video_path /
                   user_tag) OR a 'playlist' key: list of per-request
                   dicts {prompt, video_path, user_tag, seq, video_id}
                   the worker processes SERIALLY after one barrier
                   release (used by the concurrent bench's playlist
                   mode; absent everywhere else -> the historical
                   single-request path, bit-for-bit).
  barrier   multiprocessing Barrier, parties = N workers + parent.
                   Workers signal 'ready' (model loaded) through the
                   pipe BEFORE entering barrier.wait(); the parent
                   joins the barrier only after N 'ready's, so model
                   loads / CUDA contexts never land in a timing
                   window. t_request_start is captured right AFTER
                   barrier.wait() — same convention as the thread
                   wave. With a playlist the barrier aligns only the
                   FIRST video of each user; afterwards every user
                   loops freely (no inter-video barrier).
  msg_pipe  the child end of a duplex Pipe. Protocol messages are
                   SENT through it (pickled dicts), never printed:

    {"type": "ready", "user_index", "pid"}
    {"type": "barrier_wait", "user_index"}
    {"type": "done", "user_index", "record": {...}}   one PER
        playlist item (single-request path: exactly one), final
        schema the thread path produces (ok/ttft/latency/
        engine_stats/... plus seq — everything LogRecord assembly
        needs). A failed item still emits done with ok=False and
        the loop CONTINUES — one dead video must not kill the
        user's remaining playlist.
    {"type": "error", "user_index", "error": "..."}   crash report

  Plain stdout is inherited: library prints (transformers warnings,
  tqdm) reach the parent's console unchanged, which is exactly what
  a human debugging a wave wants.

  The worker NEVER raises out of _worker_main: failures become an
  'error' message (or, if the record itself is unserializable, a
  best-effort 'error' line) so the wave keeps its failure-isolation
  semantics — one dead user must not kill its siblings.
"""

import os
import sys
import time

import json


def _cpus_spec(cfg: dict):
    """Canonical 'edge_cpus' string for ready-messages and records:
    a comma-joined cpu list, or None when no assignment exists."""
    cpus = cfg.get("edge_cpus")
    if not cpus:
        return None
    cpu_list = sorted({int(c) for c in
                       ([cpus] if isinstance(cpus, int) else cpus)})
    return ",".join(str(c) for c in cpu_list) if cpu_list else None


def _apply_cpu_affinity(cfg: dict) -> None:
    """Pin THIS worker process to its assigned CPU set (Linux only).

    Called FIRST in _worker_main, before any heavy import, so the
    whole worker lifetime (import, model load, decode+ViT, HTTP)
    runs inside the assignment. cfg["edge_cpus"] is either a list of
    ints or an int (single core) from the parent's CPU-assignment
    pass; absent/empty/falsy -> no pinning (parent decided this
    worker gets no budget, e.g. thread-mode or affinity unavailable).

    The cpu-ids spec is echoed back in the 'ready' message and the
    final record (edge_cpus field) so runs are self-describing on
    the DB side. Non-Linux / missing sched_setaffinity: warn once on
    stderr and continue unpinned (measurement still valid, just with
    a softer terminal-resource model).

    cfg["preprocess_threads"] (optional int) caps the torch/OMP
    intra-op pools for CPU-side preprocessing (resize / HF processor)
    instead of deriving them from the pin: effective value is
    min(N, pin core count) when pinned (N above the pin would only
    thrash the scheduler inside the budget), N as-is when unpinned,
    and the pin's core count when absent (historical behaviour).
    """
    cpus = cfg.get("edge_cpus")
    cpu_list = sorted({int(c) for c in
                       ([cpus] if isinstance(cpus, int) else cpus)}) if cpus else []
    pt_raw = cfg.get("preprocess_threads")
    try:
        pt = int(pt_raw) if pt_raw not in (None, "") else None
    except (TypeError, ValueError):
        print(f"[encpl-edge-worker-qwen36:{os.getpid()}] warning: "
              f"preprocess_threads={pt_raw!r} is not an integer; "
              f"deriving the intra-op thread count from the pin "
              f"(native default when unpinned)",
              file=sys.stderr)
        pt = None
    if pt is not None and pt <= 0:
        pt = None
    if not cpu_list:
        # No pin: an explicit preprocess_threads still applies (the
        # terminal-resource model is thread-count-only); without one,
        # leave the pools at their native defaults.
        if pt is not None:
            _apply_intraop_thread_cap(pt)
        return
    spec = ",".join(str(c) for c in cpu_list)
    try:
        os.sched_setaffinity(0, set(cpu_list))
    except (AttributeError, OSError) as exc:
        print(f"[encpl-edge-worker-qwen36:{os.getpid()}] warning: cannot "
              f"pin to cpus [{spec}] ({exc}); running unpinned",
              file=sys.stderr)
        if pt is not None:
            _apply_intraop_thread_cap(pt)
        return
    # Keep intra-process thread pools inside the budget too: without
    # this, torch/OMP would spawn one thread per MACHINE core on a
    # one-core assignment, thrashing the scheduler. Explicit
    # preprocess_threads REPLACES the pin-derived count (clamped to
    # the budget: min(N, K)).
    effective = len(cpu_list)
    if pt is not None:
        if pt > len(cpu_list):
            print(f"[encpl-edge-worker-qwen36:{os.getpid()}] note: "
                  f"preprocess_threads={pt} clamped to the "
                  f"{len(cpu_list)}-cpu pin",
                  file=sys.stderr)
        else:
            effective = pt
    _apply_intraop_thread_cap(effective)
    print(f"[encpl-edge-worker-qwen36:{os.getpid()}] pinned to cpus "
          f"[{spec}] (intra-op threads: {effective})",
          file=sys.stderr)


def _apply_intraop_thread_cap(n: int) -> None:
    """Export the OMP/MKL/OPENBLAS caps + clamp torch's own pool to n.

    Env vars via setdefault (a pre-set value wins — explicit shell
    override, same contract as the sidecar's --preprocess-threads);
    torch.set_num_threads is the in-process belt-and-braces (torch is
    imported lazily by the request path, so the env vars also cover
    the native-lib init of the eventual import).
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    import torch as _torch

    _torch.set_num_threads(n)


def _apply_decode_threads(cfg: dict) -> None:
    """Cap decord's/torchcodec's internal decode thread pool for this
    worker (SGLANG_DECORD_NUM_THREADS; the env name is historical —
    the patched video_decoder.py reads it for BOTH backends).

    decord's VideoReader sizes its pool by HOST core count and is NOT
    covered by the OMP/MKL/torch thread knobs set in
    _apply_cpu_affinity. The parent's decode_threads config (same
    knob as the sidecar's --decode-threads flag) exports the env var,
    read at VideoReader construction time by the patched
    video_decoder.py. Must run BEFORE the first VideoReader is built.
    A pre-set env var wins (explicit override);
    absent/0/invalid = native default.
    """
    raw = cfg.get("decode_threads")
    if raw in (None, ""):
        return
    try:
        n = int(raw)
    except (TypeError, ValueError):
        print(f"[encpl-edge-worker-qwen36:{os.getpid()}] warning: "
              f"decode_threads={raw!r} is not an integer; decode "
              f"threads uncapped (native default)",
              file=sys.stderr)
        return
    if n <= 0:
        return
    if os.environ.setdefault("SGLANG_DECORD_NUM_THREADS", str(n)) != str(n):
        print(f"[encpl-edge-worker-qwen36:{os.getpid()}] note: SGLANG_"
              f"DECORD_NUM_THREADS already set to "
              f"{os.environ['SGLANG_DECORD_NUM_THREADS']}; decode_threads="
              f"{n} ignored (env wins)",
              file=sys.stderr)
        return
    print(f"[encpl-edge-worker-qwen36:{os.getpid()}] decode threads capped "
          f"to {n} (SGLANG_DECORD_NUM_THREADS; env name historical — caps "
          f"BOTH backends)",
          file=sys.stderr)


def _worker_main(cfg: dict, barrier, msg_pipe) -> None:

    def _send(msg: dict) -> None:
        try:
            msg_pipe.send(msg)
        except (BrokenPipeError, OSError):
            # Parent side closed (wave aborted / parent crashed):
            # nothing to report to; die quietly.
            pass

    # Pin BEFORE the heavy imports: torch inherits the thread env and
    # the whole lifetime (model load included) honors the budget.
    _apply_cpu_affinity(cfg)
    # Cap decord/torchcodec decode threads BEFORE the first
    # VideoReader: the env var is read at reader construction time
    # (must precede any decode, including the edge-mode local decode
    # inside generate_content_with_video).
    _apply_decode_threads(cfg)
    # Name THIS worker's decode backend (torchcodec vs decord — the
    # module-level _BACKEND global of sglang's video_decoder, resolved
    # per process at import time). The env the worker inherits could
    # select a different backend than the parent's, so each worker
    # reports its own. Degrades to "unknown" on import failure — the
    # first decode would surface a real error anyway. Mirrors the
    # thread-mode parent's _print_client_decode_backend.
    try:
        from sglang.srt.utils import video_decoder as _vd

        print(f"[encpl-edge-worker-qwen36:{os.getpid()}] decode backend: "
              f"{getattr(_vd, '_BACKEND', 'unknown')}",
              file=sys.stderr)
    except Exception as _exc:
        print(f"[encpl-edge-worker-qwen36:{os.getpid()}] decode backend: "
              f"unknown (video_decoder import failed: "
              f"{type(_exc).__name__})",
              file=sys.stderr)

    try:
        # Import INSIDE the worker: each process pays its own
        # transformers/torch import + model load BEFORE 'ready' —
        # nothing heavy remains for the timed window.
        root = cfg.get("aitestbed_root")
        if root and root not in sys.path:
            sys.path.insert(0, root)
        # Bootstrap the sibling aicodec package for the same reason
        # the runners do: the `from clients...` import below runs the
        # clients package __init__, which transitively does
        # `import aicodec` (realtime_webrtc_vlm_client) — without the
        # bootstrap that resolves to the OUTER pyproject-only aicodec/
        # dir as an empty PEP 420 namespace package. Spawn children
        # inherit the parent's sys.path (already fixed by the runner),
        # so this mainly guards --selftest direct runs and any parent
        # that never bootstrapped. Idempotent.
        _aicodec = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "aicodec")
        if os.path.isdir(_aicodec) and _aicodec not in sys.path:
            sys.path.insert(0, _aicodec)
        # Test-harness shim: replaces the client CLASS the worker
        # uses (spawn children re-import in a fresh interpreter, so
        # parent-side monkeypatching does not carry; and the real
        # clients package is often already import-locked by the
        # spawn bootstrap importing THIS module, so a sys.path shim
        # cannot reliably shadow it). The shim module path is given
        # directly and loaded by file location. Empty in production.
        shim_path = cfg.get("client_shim_path")
        if shim_path:
            import importlib.util as _ilu

            _spec = _ilu.spec_from_file_location(
                "_encpl_shim_client_qwen36", shim_path)
            _shim_mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_shim_mod)
            EncoderPlacementQwen36Client = _shim_mod.EncoderPlacementQwen36Client
        else:
            from clients.encoder_placement_client_qwen36 import (
                EncoderPlacementQwen36Client,
            )

        # Stream safety timeouts: MUST ride the cfg, not the class
        # defaults — the parent resolved them from the runner's flags
        # (concurrent waves need first-event 900, the class default 200
        # is the SERIAL budget; a 2026-10-20 14:17 run lost 2/24
        # requests to the default leaking through this exact gap when
        # the cfg keys were absent). Only non-None values are passed so
        # old cfgs (and the test shim's (base_url)-only signature)
        # keep the class-default fallback. The sanitized_env of the
        # matrix driver strips ENCPL_STREAM_* env vars, so without
        # these keys NOTHING carries the runner's timeout flags into
        # the spawn workers.
        _timeout_kwargs = {}
        for _k in ("stream_connect_timeout",
                   "stream_first_event_timeout",
                   "stream_idle_timeout",
                   "stream_upload_read_timeout"):
            _v = cfg.get(_k)
            if _v is not None:
                _timeout_kwargs[_k] = _v
        client = EncoderPlacementQwen36Client(
            base_url=cfg.get("base_url"), **_timeout_kwargs)
        # Per-process warmup — the import-race problem the thread wave
        # had to pre-warm around does not exist here: each worker warms
        # ITS OWN caches before signaling 'ready'. The Qwen3.6 edge
        # warmup honors vision_only (default True on Qwen3.6: the
        # standalone ~0.5 GB tower instead of the 35 GB fp8 full load)
        # and attn_impl, matching what the user request will ask for —
        # the model cache is keyed on (path, vision_only, impl), so a
        # mismatched warmup would leave the real key cold.
        # (cfg["skip_warmup"] is a test-harness knob: mocks without
        # the heavy deps skip the model load and go straight to
        # 'ready'.)
        mode = cfg["mode"]
        if mode == "edge" and not cfg.get("skip_warmup"):
            client._get_processor(cfg["model"])
            client._get_edge_model(
                cfg["model"],
                vision_only=cfg.get("vision_only", True),
                attn_impl=cfg.get("attn_impl", "sdpa"))

        _send({
            "type": "ready",
            "user_index": cfg["user_index"],
            "pid": os.getpid(),
            "edge_cpus": _cpus_spec(cfg),
        })

        barrier.wait()
        _send({
            "type": "barrier_wait",
            "user_index": cfg["user_index"],
        })

        # Request plan: cfg['playlist'] (bench playlist mode: V
        # per-request dicts processed SERIALLY, per-item records)
        # or the single-request legacy keys as a 1-item plan.
        if cfg.get("playlist"):
            plan = cfg["playlist"]
        else:
            plan = [{
                "prompt": cfg["prompt"],
                "video_path": cfg["video_path"],
                "user_tag": cfg.get("user_tag", ""),
                "seq": 0,
                "video_id": None,
            }]
        for item in plan:
            t_request_start = time.time()
            item_rec = {
                "user_index": cfg["user_index"],
                "seq": int(item.get("seq") or 0),
                "ok": False,
                "t_request_start": t_request_start,
                "ttft": None,
                "latency": 0.0,
                "edge_cpus": _cpus_spec(cfg),
            }
            if item.get("video_id") is not None:
                item_rec["video_id"] = item["video_id"]
            if item.get("video_path") is not None:
                item_rec["video_path"] = item["video_path"]
            try:
                # Qwen3.6 client signature: the same
                # compression/sampling knobs as the Qwen3-VL worker
                # plus the four two-tier UDP knobs (cfg keys arrive
                # from the scenario's / bench's _run_wave_processes
                # config dict — defaults match
                # generate_content_with_video's, incl. the 0.05 ms
                # inter-datagram gap).
                response = client.generate_content_with_video(
                    prompt=item["prompt"],
                    video_path=item["video_path"],
                    model=cfg["model"],
                    mode=mode,
                    max_frames=cfg["max_frames"],
                    max_tokens=cfg["max_tokens"],
                    compressor=cfg.get("compressor", ""),
                    retention_ratio=cfg.get("retention_ratio", 0.25),
                    bundle_quant=cfg.get("bundle_quant", ""),
                    vision_only=(cfg.get("vision_only", True)
                                 if mode == "edge" else False),
                    attn_impl=(cfg.get("attn_impl", "sdpa")
                               if mode == "edge" else "sdpa"),
                    user_tag=item.get("user_tag", ""),
                    frame_ratio=cfg.get("frame_ratio"),
                    optional_ratio=(cfg.get("optional_ratio", 0.0)
                                    if mode == "edge" else 0.0),
                    opt_tokens_per_packet=cfg.get(
                        "opt_tokens_per_packet", 64),
                    opt_send_gap_ms=float(cfg.get("opt_send_gap_ms", 0.05)
                                          or 0.0),
                    opt_chat_delay_ms=float(cfg.get("opt_chat_delay_ms", 0.0)
                                            or 0.0),
                )

                # StreamingResponse token fields are tokens_in/out (the
                # OpenAI-style prompt/completion_tokens names exist only
                # on the wire chunks, never on the response object).
                tokens_in = response.tokens_in
                tokens_out = response.tokens_out
                if tokens_in is None:
                    tokens_in = client.estimate_tokens(
                        item["prompt"], cfg["model"])
                if tokens_out is None:
                    tokens_out = client.estimate_tokens(
                        response.total_content, cfg["model"])

                t_last_chunk = response.t_last_chunk

                item_rec.update({
                    "ok": True,
                    "t_request_start": t_request_start,
                    "t_first_chunk": response.t_first_chunk,
                    "t_last_chunk": t_last_chunk,
                    "ttft": (response.t_first_chunk - t_request_start
                             if response.t_first_chunk is not None
                             else None),
                    "latency": (t_last_chunk - t_request_start
                                if t_last_chunk is not None else 0.0),
                    "request_bytes": response.request_bytes,
                    "response_bytes": response.response_bytes,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "chunk_count": len(response.chunks),
                    "inter_chunk_times": response.inter_chunk_times,
                    "request_payload": response.request_payload,
                    "response_payload": response.response_payload,
                    "response_events": response.response_events,
                    "engine_stats": (getattr(response, "engine_stats", {})
                                     or {}),
                    # full generated text — the dataset-mode scoring
                    # (metadata.bench) reads it in the parent
                    "total_content": response.total_content,
                })
            except BaseException as exc:  # noqa: BLE001 — record, continue
                t_fail = time.time()
                t_req = item_rec.get("t_request_start") or t_fail
                item_rec.update({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "t_request_start": t_req,
                    "latency": t_fail - t_req,
                })
            _send({
                "type": "done",
                "user_index": cfg["user_index"],
                "record": item_rec,
            })
    except BaseException as exc:  # noqa: BLE001 — record, don't crash
        # Startup failures (import/model-load, i.e. BEFORE 'ready')
        # must ALSO emit an explicit 'error' message: the parent's
        # ready-wait reacts to it immediately instead of timing out
        # with the real cause buried in a 'done' record it never
        # reads. Harmless duplication when the failure happened
        # in-window (the reader keeps only the LAST 'done'/'error').
        _send({
            "type": "error",
            "user_index": cfg["user_index"],
            "error": f"{type(exc).__name__}: {exc}",
        })
        # Crashed before the plan loop produced any record: fall
        # back to the legacy single 'done' record so the parent's
        # assembly still sees this user (ok=False, real cause).
        _send({
            "type": "done",
            "user_index": cfg["user_index"],
            "record": {
                "user_index": cfg["user_index"],
                "seq": 0,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "t_request_start": None,
                "ttft": None,
                "latency": 0.0,
                "edge_cpus": _cpus_spec(cfg),
            },
        })
    finally:
        try:
            msg_pipe.close()
        except Exception:
            pass


def _worker_entry(cfg: dict, barrier, msg_pipe) -> None:
    """Spawn target: plain indirection so the module stays importable
    (and picklable by reference) from the parent's spawn context."""
    _worker_main(cfg, barrier, msg_pipe)


def main() -> None:
    """Manual/debug entry: '--selftest' + JSON cfg runs the flow with
    a pass-through barrier and a pipe that echoes to stdout — lets a
    human or the mock harness exercise the protocol without a real
    spawn."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        cfg = json.loads(sys.argv[2])

        class _EchoPipe:
            def send(self, msg):
                print(json.dumps(msg), flush=True)

            def close(self):
                pass

        class _PassBarrier:
            def wait(self, timeout=None):
                return 0

        _worker_main(cfg, _PassBarrier(), _EchoPipe())
        return

    print("usage: encpl_edge_worker_qwen36.py --selftest '<json cfg>'",
          file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
