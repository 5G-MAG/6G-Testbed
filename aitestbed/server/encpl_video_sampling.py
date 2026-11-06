"""
encpl_video_sampling.py — shared frame-sampling wiring for the
encoder-placement sidecars (sglang_engine_server.py and
sglang_engine_server_concurrent.py).

frame_ratio / max_frames reach the ENGINE-side sampling via the
OFFICIAL ServerArgs knob mm_process_config (arg_groups/fields/mm.py:70,
validated in serving_hook.py handle_multimodal, consumed in
base_processor.py:267-270):

    mm_process_config = {"video": {"frame_ratio": ..., "max_frames": ...}}
      -> BaseMultimodalProcessor.video_config
      -> preprocess_video(video_config=...)          (qwen_vl.py:1027)
      -> smart_nframes(ele=video_config, ...)        (qwen_vl.py:261)

The two knobs are independent:
  - max_frames caps BOTH sampling modes and is a NATIVE sglang 0.5.20
    whitelist key (QWEN_VIDEO_PREPROCESS_CONFIG_KEYS, fps branch at
    qwen_vl.py:230-232) — NO sglang patch needed to use it;
  - frame_ratio selects ratio-of-native sampling (v3 semantics:
    sglang's own base = total_frames/video_fps*fps, i.e. 2fps x
    duration BEFORE clamping, multiplied by the ratio, then the fps
    branch's clamp chain — min 4, cap max_frames/768, <= total,
    floor to a multiple of 2; 0/None = off) and needs patch 2 of
    server/patch_host_sglang.py (smart_nframes frame_ratio branch).
    Only frame_ratio pays the canary cost (max_frames-only users have
    zero patch dependency). The patch carries per-video logger.info
    traces of every frame-count transition (total, base, raw,
    clamped, final).

Because mm_process_config is an ENGINE-level ServerArg, the values are
set at sidecar STARTUP — changing them requires a restart.

`ensure_sglang_patches()` — the Python-level auto-load of the sglang
disk patcher, closing the "no shell wrapper" gap for BOTH sides: the
sidecars' main() calls it at startup (fail -> abort, matching the
start_sidecar*.sh bootstrap), and the edge client calls it
unconditionally on its first edge-mode request (the wave workers are
fresh OS processes, so each one re-ensures; within a process the
lock+flag makes it a one-shot). It must run BEFORE the first
``import sglang`` in the same process — that is exactly where both
callers sit it.

`verify_frame_ratio_patch()` — called by the sidecars at startup when a
frame ratio is configured — probes the ACTIVE sglang with a canary
    call: smart_nframes({"frame_ratio": 0.25}, total_frames=1000,
    video_fps=25) must return 20 on a v3-patched install (base 80 x 0.25).
    On a pristine install the "frame_ratio" key is simply ignored by
    smart_nframes (unknown keys don't raise) and the fps branch returns
    80; the DEPRECATED v2 patch (ratio x total_frames) returns 250 — all
    three outcomes are distinct, so the canary detects both a missing
    patch and a stale-semantics one, turning either into a loud,
    actionable error BEFORE the engine accepts traffic instead of a
    silent wrong-frame-count benchmark.

    NOTE: this canary also validates ensure_sglang_patches()' work —
    it reads the patched smart_nframes through the import system, so it
    can only pass after the disk patch landed (or the shell bootstrap
    applied it earlier).

Edge mode does NOT come through here: the terminal runs the sampling
itself (clients/encoder_placement_client.py _prepare_edge_bundle calls
preprocess_video directly with its own video_config), so both arms
sample identically by construction (same function, same keys).
"""


def build_mm_process_config(frame_ratio: float | None,
                            max_frames: int | None) -> dict | None:
    """Assemble the Engine kwarg from the CLI knobs.

    Returns None (omit the ServerArg entirely — sglang applies its own
    defaults) unless at least one knob carries a non-default value.
    ``frame_ratio=0`` / ``max_frames=0`` are the CLI spellings of "off"
    (argparse cannot default a flag to None portably across the two
    sidecars); 768 is max_frames' sglang default (FPS_MAX_FRAMES), so
    an explicit 768 is also folded away — byte-equivalent to omitting.
    The two knobs are assembled INDEPENDENTLY: max_frames alone (no
    ratio) is a valid, patch-free configuration.
    """
    video_cfg: dict = {}
    # 0 / None are the CLI spellings of "off" (argparse cannot default
    # a flag to None portably across the two sidecars), so a falsy
    # ratio folds away instead of raising — build_mm_process_config is
    # shared with the sidecars' main(), where --frame-ratio 0 must
    # simply mean "default sampling". The sidecars validate their CLI
    # values BEFORE calling here; direct callers passing a real
    # out-of-range value (e.g. 1.5) still get the loud error below.
    if frame_ratio and not (0.0 < frame_ratio <= 1.0):
        raise ValueError(
            f"--frame-ratio must be in (0, 1], got {frame_ratio}")
    if frame_ratio:
        video_cfg["frame_ratio"] = float(frame_ratio)
    if max_frames and int(max_frames) != 768:
        video_cfg["max_frames"] = int(max_frames)
    if not video_cfg:
        return None
    return {"video": video_cfg}


def describe_mm_process_config(cfg: dict | None) -> str:
    """Human-readable rendering for startup logs."""
    if not cfg:
        return "sglang-default (2fps x duration, cap 768)"
    video = cfg.get("video", {})
    ratio = video.get("frame_ratio")
    maxf = video.get("max_frames")
    if ratio:
        return (f"frame_ratio={ratio}"
                + (f" cap={maxf}" if maxf is not None else ""))
    return f"max_frames={maxf}"


import threading

_ensured_lock = threading.Lock()
_ensured_done = False


def ensure_sglang_patches(reason: str = "startup",
                          patcher_name: str = "patch_host_sglang") -> None:
    """Run the sglang disk patcher from INSIDE a Python entry point,
    covering starts that bypass the start_sidecar*.sh bootstrap (direct
    ``python sglang_engine_server*.py``, benchmark drivers, edge
    workers — none of which go through a shell wrapper).

    patcher_name selects WHICH sibling patcher to run:
      - "patch_host_sglang" (default): the Qwen3-VL patcher (patch A:
        qwen3_vl.py guard; patches 2/3/4 on the shared qwen_vl.py /
        video_decoder.py).

    The shared-file patches are byte-identical across patcher siblings,
    so the lock+flag one-shot semantics stay correct even if different
    processes pick different patchers; within one process the first
    call wins and later calls (any patcher) are no-ops.

    Idempotent within one process (a lock + flag, so the concurrent
    sidecar's threads and an edge worker calling it per-wave do not
    re-patch). The patcher itself is ALSO idempotent (marker no-op,
    v2->v3 in-place upgrade) and does not import sglang.

    Any failure raises RuntimeError -> the caller dies before
    accepting traffic / running the edge arm (server main(): fail =
    abort, matching the shell bootstrap's ``patcher || die``).

    IMPORTANT: call BEFORE the first ``import sglang`` — patches land
    on disk source, so they only take effect for modules imported
    after this returns (in the sidecars the first sglang import
    happens deep inside _build_engine, well after main() calls this).
    """
    global _ensured_done
    import sys

    with _ensured_lock:
        if _ensured_done:
            return
        # Reuse the caller's already-inserted server/ dir on sys.path
        # (both sidecars and the edge client insert it to import THIS
        # module); resolve the patcher beside this file either way.
        import importlib

        patcher = importlib.import_module(patcher_name)

        # Mock/test environments without sglang installed: nothing to
        # patch (and _locate_sglang_file would die). The patch only
        # matters for processes that actually import sglang — those
        # fail loudly at THEIR import with the original error.
        import importlib.util

        if importlib.util.find_spec("sglang") is None:
            _ensured_done = True
            return

        try:
            # quiet: suppress the per-_info "already patched" chatter —
            # the auto-load path runs once per wave-worker PROCESS
            # (batch tests spawn many), and N x info lines per wave is
            # log spam. Real state changes (backup/patched/upgrade)
            # stay visible: _apply_patch prints those directly.
            patcher.main(quiet=True)
        except SystemExit as exc:
            raise RuntimeError(
                f"ensure_sglang_patches({reason}): {patcher_name} "
                f"exited with {exc.code} — see its stderr above; the "
                f"sglang source patch required for encoder-placement "
                f"(edge mode / frame_ratio sampling) failed to apply"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"ensure_sglang_patches({reason}): {patcher_name} "
                f"raised {type(exc).__name__}: {exc}") from exc
        _ensured_done = True


def verify_frame_ratio_patch(active: bool) -> None:
    """Canary-check the ACTIVE sglang's smart_nframes for the v3
    frame_ratio branch (ratio-of-native semantics). Raises RuntimeError
    when `active` (a frame ratio was configured) but the patch is
    missing or carries the DEPRECATED v2 semantics.

    Discriminating input: total=1000 @ 25fps, ratio=0.25 ->
      v3 (installed): base = 1000/25*2 = 80, raw = 80*0.25 = 20,
                      clamp chain keeps 20 (above min 4, below all
                      caps) -> 20
      unpatched:      frame_ratio key ignored -> fps branch ->
                      1000/25*2 = 80 (clamps keep 80) -> 80
      v2 (deprecated): total_frames * ratio = 250 -> clamped by
                      min(768, 1000) = 768 -> 250
    All three outcomes are distinct; the raw value stays clear of the
    clamp floor (min_frames=4) so the check keeps its discriminating
    power.

    Import cost: sglang.srt.multimodal.processors.qwen_vl pulls torch
    etc. — the sidecar imports all of that anyway when building the
    Engine, and the call is skipped entirely for max_frames-only runs.
    """
    if not active:
        return
    from sglang.srt.multimodal.processors.qwen_vl import smart_nframes

    n = smart_nframes({"frame_ratio": 0.25}, total_frames=1000, video_fps=25)
    if n != 20:
        detail = (f"canary returned {n}, expected 20 (unpatched sglang "
                  f"returns 80, the deprecated v2 patch returns 250)")
        raise RuntimeError(
            f"smart_nframes did not apply the v3 ratio-of-native "
            f"semantics ({detail}): the ACTIVE sglang lacks the current "
            f"6G-Testbed frame_ratio patch. Run "
            f"python aitestbed/server/patch_host_sglang.py inside the "
            f"sidecar's conda env (the start_sidecar*.sh bootstrap does "
            f"this automatically; it also UPGRADES an installed v2 "
            f"patch in place) and restart the sidecar. (max_frames "
            f"alone needs no patch — only frame_ratio does.)")


def parse_body_sampling(body: dict, engine_ratio: float | None,
                        engine_max_frames: int | None) -> None:
    """Advisory-only check of per-request body values against the
    engine-level sampling configuration.

    The OpenAI chat-completions body carries max_frames / frame_ratio
    from the client for record-keeping; the ENGINE samples with its
    startup-time mm_process_config (per-request video_config would need
    deeper engine surgery — deliberately out of scope). A mismatch
    between what the client asked and what the engine will do would
    silently mislabel a benchmark run, so warn loudly instead.

    Never raises; server-mode requests whose values differ from the
    sidecar's simply get the warning below (edge mode is unaffected:
    the terminal's own sampling is authoritative there).

    Comparison rule: warn only when BOTH sides are explicitly set and
    differ; an unset side (None / 0 / 768-absent) never triggers —
    max_frames=768 equals the sglang default, so a client sending the
    historical default against an engine without the key is a match,
    not a misconfiguration.
    """
    body_ratio = body.get("frame_ratio")
    body_max_frames = body.get("max_frames")
    problems = []

    def _num(value):
        """Coerce a JSON body value to float; None on any non-numeric
        type (advisory check must never raise on malformed input —
        `int("abc")` / `int([1])` would 500 the request AFTER the
        bundle was popped from the store, losing it)."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    body_ratio_num = _num(body_ratio)
    if (body_ratio_num is not None and body_ratio_num != 0
            and body_ratio_num != (engine_ratio or 0)):
        problems.append(
            f"body frame_ratio={body_ratio!r} != engine "
            f"frame_ratio={engine_ratio or 0} (set at sidecar startup)")
    body_mf_num = _num(body_max_frames)
    if (body_mf_num is not None
            and body_mf_num != 768
            and engine_max_frames
            and body_mf_num != int(engine_max_frames)):
        problems.append(
            f"body max_frames={body_max_frames!r} != engine "
            f"max_frames={engine_max_frames} (set at sidecar startup)")
    if problems:
        print("[EncPl-server] warning: request sampling values do not "
              "match the engine configuration; the ENGINE values win "
              "(restart the sidecar with matching --frame-ratio / "
              "--max-frames to change them): " + "; ".join(problems))


def mm_process_config_help(flag: str) -> str:
    """Shared argparse help text for the sampling knobs.

    NOTE: the returned string must contain NO bare '%' — argparse runs
    help strings through %-formatting, and a stray single '%' raises
    "badly formed help string" at add_argument time. The flag fragment
    is pre-escaped here so callers can pass literal '%'.
    """
    flag = (flag or "").replace("%", "%%")
    return (
        "video frame-sampling knobs, applied ENGINE-wide via the "
        "official mm_process_config ServerArg (restart to change); the "
        "two knobs are independent. "
        f"{flag}: ratio-of-native sampling — sglang's own base "
        "(total_frames/video_fps*fps, i.e. 2 fps x duration BEFORE "
        "clamping) multiplied by the ratio, then the native clamp "
        "chain (min 4, cap max_frames/768, <= total, floor to a "
        "multiple of 2); 0 = off (default); REQUIRES the sglang "
        "frame_ratio patch v3 (patch_host_sglang.py; start_sidecar*.sh "
        "applies it and upgrades an old patch in place; per-video "
        "frame counts are logger.info-traced). max_frames caps BOTH "
        "sampling modes; 0 = sglang default 768 (FPS_MAX_FRAMES) — "
        "explicit 768 equals the default, and max_frames alone needs "
        "NO sglang patch (native whitelist key). Server-mode sampling "
        "happens engine-side with these values; edge-mode sampling "
        "runs on the terminal with the client's values (the same "
        "smart_nframes function)")


def format_log_line(engine_ratio: float | None, engine_max_frames: int | None,
                    cfg_json: str | None) -> str:
    """One-line summary for the 'Engine ready' log."""
    return (f"video_sampling: frame_ratio="
            f"{engine_ratio if engine_ratio else 'off'}"
            f" max_frames={engine_max_frames or 'sglang-default'}"
            f" mm_process_config={cfg_json or 'omitted'}")
