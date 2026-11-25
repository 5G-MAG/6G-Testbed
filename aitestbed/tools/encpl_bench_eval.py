#!/usr/bin/env python3
"""
Encoder-placement benchmark evaluation (Video-MME + MLVU).

Standalone driver (no orchestrator, no new scenario): the encoder_placement
sibling of tools/videomme_eval.py. Loads a video QA test set, sends each
video+question through the existing EncoderPlacementClient (server or edge
mode) against the SGLang engine sidecar (port 8002), collects the model's
raw answer into a questions_results_{profile}.json file that the existing
MetricsCalculator.load_task_results / calculate_task_accuracy ("real" task
class) can consume, and prints accuracy + latency summaries.

Datasets:
    videomme  (HF official release)
        $DATASET_DIR/data/test-00000-of-00001.parquet
        $DATASET_DIR/videos/  (unzipped Video-MME videos, mp4)
        4-option MCQ; parquet answers are already option letters.

    mlvu      (official MLVU release)
        $DATASET_DIR/json/{1..9}_*.json
        $DATASET_DIR/video/<task_dir>/*.mp4
        Tasks 1-7 are 4-option MCQ (candidate texts mapped to A-D letters;
        ground-truth letter = index of the answer text among candidates).
        8_sub_scene and 9_summary are generation tasks scored by GPT over
        scoring_points in the official kit — excluded by default
        (--tasks mcq); --tasks all records them unscored (raw answer +
        scoring_points stored for offline GPT scoring later).

Usage:
    python tools/encpl_bench_eval.py \
        --dataset videomme --dataset-dir /data01/Video-MME \
        --mode server --profile no_emulation --limit 12

    ENC_MODE=edge python tools/encpl_bench_eval.py \
        --dataset mlvu --dataset-dir /data01/MLVU \
        --limit 12 --resume

    python tools/encpl_bench_eval.py --metrics-only \
        --dataset mlvu --mode edge --dataset-dir /data01/MLVU

Netem topology handling (single-host loopback port shaping derived from
the base URL; dual-host whole-interface shaping; asymmetric-profile
rejection on loopback) is reused verbatim from tools/videomme_eval.py.

CPU budgets (both arms): edge mode's decode + processor + ViT run in THIS
process and would otherwise grab every core of a many-core host, muting
the server-vs-edge contrast. --edge-cpus pins this process for the whole
run (lazy torch import + model load included), defaulting to
"auto-half-other" (SECOND half of the cores — disjoint from the sidecar's
default --server-cpus auto-half, first half, so a single host gets both
arms budgeted with zero manual spec). Other values: "auto-half", a cpu
list with ranges ("48-95,100"), or '' to disable. Server mode warns and
ignores it (no local model work there). The applied spec lands in every
result entry's encpl_stats.edge_cpus and the compute_and_print summary,
so --metrics-only can reconstruct the CPU budget of a past run. The pin
is removed when the run ends (finally, next to the netem teardown).

Cache hygiene: SGLang's radix cache and multimodal preprocess cache are
flushed via the sidecar's POST /flush_cache before every question by
default, because the same video appears across multiple questions
(Video-MME ~3 questions/video; MLVU task dirs share files) and a warm
prefix would understate prefill/TTFT for repeat questions. Disable with
--no-flush-cache (e.g. to measure steady-state cache-hit behaviour).
Upload binaries themselves are never reused: each question gets a fresh
uuid bundle id and the server pops it right after the chat request.

Environment:
    SGLANG_ENGINE_BASE_URL   sidecar base URL (default http://localhost:8002)
    ENC_MODE                 fallback for --mode (env wins, as in the scenario)
    ENCPL_EDGE_CPUS          fallback for --edge-cpus (env wins)
    ENCPL_DECODE_THREADS     fallback for --decode-threads (env wins)
    DATASET_DIR              fallback for --dataset-dir
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
AITESTBED_ROOT = REPO_ROOT / "aitestbed"
TOOLS_DIR = Path(__file__).resolve().parent
# Same bootstrap as videomme_eval.py / orchestrator.py: aitestbed itself for
# `clients.*` / `analysis.*`, plus the sibling aicodec / netemu packages.
for _p in (str(TOOLS_DIR), str(AITESTBED_ROOT),
           str(REPO_ROOT / "aicodec"), str(REPO_ROOT / "netemu" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# videomme_eval's module level is stdlib-only, so importing it here is safe;
# netem/pandas imports inside its functions stay lazy.
import videomme_eval as _vmme  # noqa: E402

# Frame-sampling semantics version stamped into every results file this
# tool writes (top-level "frame_sampling_semantics" key). v3 = the
# ratio-of-native patch (base = total_frames/video_fps*fps, i.e. 2fps x
# duration before clamping, x ratio, then the native clamp chain);
# older files (v2: ratio x total_frames; or unstamped = pre-v3) are
# REFUSED for --resume — same ratio VALUE, different sampled frames, so
# mixing would silently skew the accuracy. Old files stay on disk as
# old-semantics baselines; start a fresh results file per semantics.
FRAME_SAMPLING_SEMANTICS = "v3-ratio-of-native"

MLVU_MCQ_STEMS = (
    "1_plotQA", "2_needle", "3_ego", "4_count",
    "5_order", "6_anomaly_reco", "7_topic_reasoning",
)
MLVU_GEN_STEMS = ("8_sub_scene", "9_summary")
MLVU_ALL_STEMS = MLVU_MCQ_STEMS + MLVU_GEN_STEMS


def _save_results(path: Path, results: list[dict]) -> None:
    """_vmme.save with the frame-sampling semantics stamped per-entry.

    Every entry's encpl_stats carries frame_sampling_semantics (the
    client already mirrors other sampling knobs there), so a later
    --resume can REFUSE files produced under different sampling
    semantics (same ratio value, different frames) without changing
    the on-disk format: MetricsCalculator.load_task_results consumes
    the same bare list as before.
    """
    for r in results:
        stats = r.get("encpl_stats")
        if isinstance(stats, dict):
            stats.setdefault("frame_sampling_semantics",
                             FRAME_SAMPLING_SEMANTICS)
    _vmme.save(path, results)


def flush_server_cache(base_url: str, verbose: bool = True) -> bool:
    """POST the sidecar's /flush_cache (radix + mm preprocess caches).

    Best-effort: returns False (and warns once) when the endpoint is
    missing or errors — the run then proceeds with whatever cache state
    the server has, which is no worse than not flushing at all. A warning
    is printed rather than an exception because the endpoint is a
    benchmark-purity control, not a functional dependency.
    """
    url = base_url.rstrip("/") + "/flush_cache"
    try:
        resp = requests.post(url, timeout=60)
        if resp.status_code < 400:
            return True
        msg = f"HTTP {resp.status_code}"
    except Exception as exc:
        msg = str(exc)
    if verbose:
        print(f"[EncPlBench] warning: flush_cache failed ({msg}); server-side "
              f"radix/mm caches are NOT reset between questions — repeat-video "
              f"questions may show understated TTFT/prefill")
    return False


# ---------------------------------------------------------------------------
# sidecar preflight: connectivity (retry-until-ready) + model agreement
# (ported from encpl_bench_eval_concurrent.py, same semantics)
# ---------------------------------------------------------------------------

def _norm_model_id(s: str | None) -> str:
    """Normalize a model id for comparison: strip ./ prefix and trailing
    slashes, normalize case and separators (Windows/POSIX forms)."""
    if not s:
        return ""
    s = s.strip()
    while s.startswith("./"):
        s = s[2:]
    s = s.rstrip("/\\")
    return os.path.normcase(os.path.normpath(s))


def _model_paths_match(client_model: str, server_id: str) -> bool:
    """True when the bench --model and the sidecar /v1/models id agree.

    Path forms differ legitimately (sidecar started with ./aicodec/...
    from the repo root, tool given a repo-relative or absolute path):
    compare normalized forms; when one side is RELATIVE, also try both
    resolved against the repo root and CWD (same-host deployments).

    Last resort: RELATIVE prefixes are anchor-dependent — the sidecar
    started from the repo root reads ./aicodec/... against it while the
    bench runs from aitestbed/ and passes ../aicodec/... against THAT
    cwd, so no single anchor can ever align both. The final path
    component (the model dir name) then decides: same dir name from any
    root is the same model in practice. A DIFFERENT basename
    (qwen3_vl_4B vs Qwen3-VL-30B) still does NOT count.
    """
    a, b = _norm_model_id(client_model), _norm_model_id(server_id)
    if not a or not b:
        return False
    if a == b:
        return True
    bases = {os.path.normcase(os.path.abspath(p))
             for p in (str(REPO_ROOT), os.getcwd())}
    for anchor in bases:
        ca = os.path.normcase(os.path.abspath(
            os.path.join(str(anchor), client_model.strip().rstrip("/\\"))))
        cb = os.path.normcase(os.path.abspath(
            os.path.join(str(anchor), server_id.strip().rstrip("/\\"))))
        if ca == cb:
            return True
    name_a = os.path.basename(a)
    name_b = os.path.basename(b)
    if name_a and name_a == name_b:
        return True
    return False


def _preflight_sidecar(base_url: str, model: str,
                       wait_sec: float, check_model: bool = True,
                       poll_sec: float = 2.0,
                       log_prefix: str = "[EncPlBench]") -> str:
    """GET {base_url}/v1/models until the sidecar is up, then verify the
    served model id matches the bench --model.

    Why both checks here: the sidecar only binds its port AFTER the
    multi-minute engine build (uvicorn.run is the last main() step), so
    a reachable /v1/models means fully ready — retrying on
    Connection-refused/timeout turns "bench started too early" from a
    crash on question 1 into a bounded wait. The model check catches
    sidecar/client model drift: the chat endpoint only ECHOES
    body.model (no validation), and the edge arm would locally load the
    client's processor+ViT while the server runs a different LLM —
    silently corrupt results — so a mismatch is a hard exit unless
    --no-model-check was given (exotic/anonymized server ids).

    log_prefix lets sibling tools (e.g. run_encoder_placement_concurrent)
    reuse this helper with their own log tag; default = this bench's.

    Returns the served model id. Exits (sys.exit) on timeout or
    mismatch; KeyboardInterrupt during the wait propagates.
    """
    url = base_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + max(0.0, wait_sec)
    t0 = time.monotonic()
    waited = False
    last_err: Exception | None = None
    server_id: str | None = None
    while True:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = (resp.json() or {}).get("data") or []
            ids = [str(d.get("id")) for d in data if d.get("id")]
            if ids:
                server_id = ids[0]
                break
            last_err = RuntimeError(
                f"/v1/models returned no model ids: {resp.text[:200]}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_err = exc
        if time.monotonic() >= deadline:
            hint = ("; is the sidecar running? (start_sidecar.sh; "
                    "engine build takes minutes before the port "
                    "opens)") if waited else ""
            sys.exit(f"sidecar preflight failed at {url}: "
                     f"{type(last_err).__name__}: {last_err}"
                     f" (waited {time.monotonic() - t0:.0f} s of "
                     f"{wait_sec:.0f} s budget{hint}; raise --wait-sidecar"
                     f" or start the sidecar first)")
        if not waited:
            print(f"{log_prefix} sidecar not ready yet at {url} "
                  f"(retrying every {poll_sec:.0f} s, "
                  f"budget {wait_sec:.0f} s)")
            waited = True
        time.sleep(poll_sec)

    if not check_model:
        if server_id != model and not _model_paths_match(model, server_id):
            print(f"{log_prefix} warning: --no-model-check: sidecar serves "
                  f"{server_id!r}, bench uses {model!r} — proceeding anyway")
        return server_id

    if not _model_paths_match(model, server_id):
        sys.exit(
            f"model mismatch: sidecar at {base_url} serves "
            f"{server_id!r} but the bench uses --model {model!r}. "
            f"The chat endpoint does not validate body.model, and in edge "
            f"mode the client would locally load {model!r}'s processor/ViT "
            f"while the server runs a different LLM — results would be "
            f"silently wrong. Restart the sidecar with the same --model, "
            f"or point the bench at the served model (or --no-model-check "
            f"to override).")
    print(f"{log_prefix} sidecar ready: {base_url} serves {server_id!r} "
          f"(matches --model)")
    return server_id


def _print_sidecar_stats(base_url: str,
                         log_prefix: str = "[EncPlBench]") -> None:
    """GET {base_url}/stats and print ONE environment-context line
    (video_decode_backend, decode_threads, parallel_video_preprocess,
    server_cpus).

    Best-effort ONLY — never sys.exit: preflight already gated
    connectivity/model, and an older or serial sidecar may lack the
    newer keys (e.g. video_decode_backend); the run must proceed. A
    missing/unreadable /stats degrades to a single warning line.

    log_prefix lets sibling tools reuse this helper with their own tag.
    """
    url = base_url.rstrip("/") + "/stats"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        stats = resp.json()
        if not isinstance(stats, dict):
            raise ValueError(f"not a JSON object: {resp.text[:200]}")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"{log_prefix} warning: sidecar stats unreadable at {url} "
              f"({type(exc).__name__}: {exc}) — continuing")
        return
    keys = ("video_decode_backend", "decode_threads",
            "parallel_video_preprocess", "server_cpus")
    parts = [f"{k}={stats[k]}" for k in keys if k in stats]
    if parts:
        print(f"{log_prefix} sidecar stats: " + " ".join(parts))
    else:
        print(f"{log_prefix} warning: sidecar stats at {url} carry none "
              f"of {', '.join(keys)} — continuing")


def _auto_half_other(cpus: list[int]) -> list[int]:
    """SECOND half (ceil-split) of a sorted cpu pool — the default
    terminal pool when the sidecar runs --server-cpus auto-half (FIRST
    half). Same arithmetic as the scenarios' _parse_cpu_ids
    auto-half-other and the serial encoder_placement precedent."""
    if not cpus:
        return []
    k = max(1, -(-len(cpus) // 2))
    return cpus[k:] or cpus[:k]


def _check_sidecar_edge_cpu_overlap(
        base_url: str,
        edge_pool_spec,
        log_prefix: str = "[EncPlBench]",
        sidecar_hint: str = "start_sidecar.sh",
) -> None:
    """Fail fast (sys.exit) when the sidecar's --server-cpus pin and
    the edge workers' cpu pool OVERLAP.

    Why hard: in the concurrent edge-vs-server experiment the two CPU
    budgets are the independent variables — an overlap silently turns
    'N terminals on a fixed fleet' into 'terminals competing with the
    engine for the same cores', and every timing the wave measures
    (decode, ViT prefill, TTFT) is contaminated. The default pools
    (server auto-half = FIRST half, edge auto-half-other = SECOND
    half) are disjoint by construction; this check catches MANUAL
    pins that defeat it (e.g. server 0-31 + edge 1-65).

    Reads the sidecar's /stats server_cpus (comma cpu list, or
    "unpinned"). Degrades to a WARNING (not exit) when /stats is
    unreadable or server_cpus is missing/"unpinned"/garbage: the
    preflight already gated connectivity, and an unpinned server is
    a legal (if noisy) configuration, not a config CONTRADICTION.

    edge_pool_spec: the runner's --edge-cpu-ids value (None/"" =
    default auto-half-other; parsed against THIS process's affinity,
    same rule the scenario's _parse_cpu_ids applies).
    """
    url = base_url.rstrip("/") + "/stats"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        stats = resp.json()
        if not isinstance(stats, dict):
            raise ValueError(f"not a JSON object: {resp.text[:200]}")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"{log_prefix} warning: sidecar /stats unreadable at "
              f"{url} ({type(exc).__name__}: {exc}) — cannot verify "
              f"server/edge cpu disjointness, continuing")
        return
    server_cpus_raw = stats.get("server_cpus")
    if not server_cpus_raw or not str(server_cpus_raw).strip():
        print(f"{log_prefix} warning: sidecar /stats carries no "
              f"server_cpus — cannot verify server/edge cpu "
              f"disjointness, continuing")
        return
    server_cpus_text = str(server_cpus_raw).strip()
    server_cpu_list: list[int] = []
    if server_cpus_text.lower() != "unpinned":
        for part in server_cpus_text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                server_cpu_list.append(int(part))
            except ValueError:
                print(f"{log_prefix} warning: sidecar server_cpus="
                      f"{server_cpus_text!r} is not a cpu list — "
                      f"cannot verify server/edge cpu disjointness, "
                      f"continuing")
                return
    if not server_cpu_list:
        print(f"{log_prefix} warning: sidecar runs UNPINNED "
              f"(server_cpus={server_cpus_text!r}); its decode/"
              f"preprocess threads roam every core — edge timings "
              f"will be noisy but not contradictory, continuing")
        return

    # Edge pool: same parse + same default as the scenario layer's
    # _parse_cpu_ids — including int/list specs (yaml documents
    # edge_cpu_ids: [0, 1, 2, 3, 8]). A spec this helper cannot parse
    # WARNS and continues (never crashes the runner): the scenario's
    # own _parse_cpu_ids stays the authority for hard errors.
    try:
        own_affinity = os.sched_getaffinity(0)
    except AttributeError:
        own_affinity = None
    cpus = sorted(int(c) for c in (own_affinity or []))
    if edge_pool_spec is None or (isinstance(edge_pool_spec, str)
                                  and not edge_pool_spec.strip()):
        edge_cpu_list = _auto_half_other(cpus)
    elif isinstance(edge_pool_spec, int):
        edge_cpu_list = [edge_pool_spec] if edge_pool_spec >= 0 else []
    elif isinstance(edge_pool_spec, (list, tuple, set)):
        edge_cpu_list = sorted({int(c) for c in edge_pool_spec
                                if int(c) >= 0})
    else:
        text = str(edge_pool_spec).strip()
        if text.lower() == "auto-half-other":
            edge_cpu_list = _auto_half_other(cpus)
        else:
            edge_cpu_list = []
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    lo_s, _, hi_s = part.partition("-")
                    try:
                        lo, hi = int(lo_s), int(hi_s)
                    except ValueError:
                        lo, hi = -1, -2  # falls into the reject below
                    if lo < 0 or hi < lo:
                        print(f"{log_prefix} warning: edge cpu spec "
                              f"{edge_pool_spec!r} is not parseable — "
                              f"cannot verify server/edge cpu "
                              f"disjointness, continuing")
                        return
                    edge_cpu_list.extend(range(lo, hi + 1))
                else:
                    try:
                        n = int(part)
                    except ValueError:
                        print(f"{log_prefix} warning: edge cpu spec "
                              f"{edge_pool_spec!r} is not parseable — "
                              f"cannot verify server/edge cpu "
                              f"disjointness, continuing")
                        return
                    if n < 0:
                        print(f"{log_prefix} warning: edge cpu spec "
                              f"{edge_pool_spec!r} has negative cpu "
                              f"{n} — cannot verify server/edge cpu "
                              f"disjointness, continuing")
                        return
                    edge_cpu_list.append(n)
    if not edge_cpu_list:
        return  # pinning effectively off — nothing to contradict

    overlap = sorted(set(server_cpu_list) & set(edge_cpu_list))
    if overlap:
        def _fmt(ids: list[int]) -> str:
            return (",".join(str(c) for c in ids) if len(ids) <= 16
                    else f"{len(ids)} cpus ({ids[0]}..{ids[-1]})")
        sys.exit(
            f"{log_prefix} FATAL: sidecar --server-cpus pin "
            f"[{_fmt(server_cpu_list)}] overlaps the terminal cpu pool "
            f"[{_fmt(edge_cpu_list)}] on [{_fmt(overlap)}] — the "
            f"server and the edge workers would compete for the same "
            f"cores and every wave timing is contaminated. Restart "
            f"the sidecar with a disjoint --server-cpus (e.g. first "
            f"half / auto-half) or point the runners at a disjoint "
            f"--edge-cpu-ids pool (e.g. second half / "
            f"auto-half-other). ({sidecar_hint} runs the sidecar.)")


def _print_client_decode_backend(log_prefix: str = "[EncPlBench]") -> None:
    """Print which decode backend THIS process (edge arm: decode runs
    client-side) picked: torchcodec vs decord, from the
    sglang.srt.utils.video_decoder module's _BACKEND global.

    In edge mode the sidecar /stats line describes the ENGINE's decode
    — irrelevant, the terminal decodes locally — so the client arm
    needs its own line. Lazy import, degrade to "unknown" instead of
    raising: this must run BEFORE any video request (hence before the
    first VideoDecoderWrapper construction), where sglang may still be
    importable even if the engine env is partially broken; the run
    itself will surface real import failures at first use. Mirrors the
    sidecar's _video_decode_backend() (same global, other process).
    """
    try:
        from sglang.srt.utils import video_decoder as vd
    except Exception as exc:
        print(f"{log_prefix} client decode backend: unknown "
              f"(sglang.srt.utils.video_decoder import failed: "
              f"{type(exc).__name__}; edge mode will surface it at "
              f"first decode)")
        return
    print(f"{log_prefix} client decode backend: "
          f"{getattr(vd, '_BACKEND', 'unknown')}")


def _resolve_capture_interface(base_url: str) -> str:
    """Interface for the pcap capture, never a bare 'auto'.

    'auto' is a netemu-side convention (resolved internally via ip
    route); tcpdump -i auto is NOT auto-detection and dies with "No
    such device" — the phantom 0-byte-capture root cause fixed for the
    concurrent runner and the orchestrator; same rule here.
    """
    import subprocess
    if _vmme._is_local_base_url(base_url):
        return "lo"
    try:
        out = subprocess.run(
            ["ip", "-o", "route", "show", "default"],
            capture_output=True, text=True, timeout=5).stdout.split()
        for i, tok in enumerate(out):
            if tok == "dev" and i + 1 < len(out):
                return out[i + 1]
    except Exception:
        pass
    sys.exit("cannot resolve the capture interface for dual-host "
             f"(base_url={base_url}): 'ip route show default' failed — "
             "pass --capture-interface explicitly")


def _preflight_capture(interface: str, capture_dir: Path) -> None:
    """Fail FAST (sys.exit) when the capture cannot possibly work.

    Same contract as the concurrent runner: tcpdump installed,
    passwordless sudo, the interface exists. Silent degradation is how
    phantom captures happen (0-byte pcaps 'promised' in metadata).
    """
    import subprocess
    if subprocess.run(["which", "tcpdump"],
                      capture_output=True).returncode != 0:
        sys.exit("--capture-pcap: tcpdump is not installed "
                 "(apt install tcpdump)")

    if subprocess.run(["sudo", "-n", "true"],
                      capture_output=True).returncode != 0:
        sys.exit("--capture-pcap: passwordless sudo is required "
                 "(sudo -n true failed). Add a NOPASSWD sudoers rule "
                 "for tcpdump, e.g.: "
                 "$USER ALL=(root) NOPASSWD: /usr/bin/tcpdump")

    if interface not in os.listdir("/sys/class/net"):
        sys.exit(f"--capture-pcap: interface '{interface}' does not exist "
                 f"(available: "
                 f"{', '.join(sorted(os.listdir('/sys/class/net')))})")

    capture_dir.mkdir(parents=True, exist_ok=True)


def _parse_edge_cpu_spec(spec, own_affinity):
    """Parse the edge-arm CPU spec into a sorted cpu list.

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
                    f"bad range {part!r} in edge-cpus spec {spec!r}") from exc
            if lo < 0 or hi < lo:
                raise ValueError(
                    f"bad range {part!r} in edge-cpus spec {spec!r}")
            out.update(range(lo, hi + 1))
        else:
            try:
                n = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"bad cpu id {part!r} in edge-cpus spec {spec!r}") from exc
            if n < 0:
                raise ValueError(
                    f"negative cpu id {n} in edge-cpus spec {spec!r}")
            out.add(n)
    return sorted(out) or None


_NUM_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS")


def _apply_edge_cpu_budget(cpu_list):
    """Pin this process to cpu_list for the duration of the run.

    Returns an opaque restore token (None when nothing was pinned,
    meaning "budget NOT applied"). Non-Linux / setaffinity failure:
    warn and return None (measurement still valid, just with a softer
    terminal-resource model — same contract as the concurrent worker
    and the serial scenario). Thread-pool caps (OMP/MKL/OPENBLAS
    _NUM_THREADS + torch.set_num_threads) keep intra-op pools inside
    the budget; the env vars are only set when absent, and both they
    and the pre-existing torch thread count are part of the restore
    token.
    """
    if not cpu_list:
        return None
    spec = ",".join(str(c) for c in cpu_list)
    try:
        prev_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, set(cpu_list))
    except (AttributeError, OSError) as exc:
        print(f"[EncPlBench] warning: cannot pin to cpus [{spec}] ({exc}); "
              f"running unpinned")
        return None
    prev_env = {}
    for var in _NUM_THREAD_VARS:
        if var not in os.environ:
            os.environ[var] = str(len(cpu_list))
            prev_env[var] = None
    torch_threads = None
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        torch_threads = torch_mod.get_num_threads()
        torch_mod.set_num_threads(len(cpu_list))
    print(f"[EncPlBench] edge cpus: pinned to [{spec}] "
          f"(restored after the run)")
    return (prev_affinity, prev_env, torch_threads)


def _restore_edge_cpu_budget(token):
    """Undo _apply_edge_cpu_budget (token from that method)."""
    if token is None:
        return
    prev_affinity, prev_env, torch_threads = token
    try:
        os.sched_setaffinity(0, prev_affinity)
    except (AttributeError, OSError) as exc:
        print(f"[EncPlBench] warning: cannot restore affinity ({exc})")
    for var in prev_env:
        if var in os.environ:
            del os.environ[var]
    if torch_threads is not None:
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            torch_mod.set_num_threads(torch_threads)


def find_mlvu_root(dataset_dir: Path) -> Path | None:
    """Locate the MLVU root (json/ + video/), tolerating an MLVU/ wrapper dir."""
    for cand in (dataset_dir, dataset_dir / "MLVU"):
        if (cand / "json").is_dir() and (cand / "video").is_dir():
            return cand
    return None


def resolve_mlvu_tasks(spec: str | None) -> list[str]:
    """Resolve --tasks to MLVU json stems: 'mcq' (default), 'all', or a list."""
    s = (spec or "").strip().lower()
    if not s or s == "mcq":
        return list(MLVU_MCQ_STEMS)
    if s == "all":
        return list(MLVU_ALL_STEMS)
    stems = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in stems if t not in MLVU_ALL_STEMS]
    if unknown:
        sys.exit(f"unknown MLVU tasks {unknown}; valid stems: {MLVU_ALL_STEMS} "
                 f"(or 'mcq' / 'all')")
    return stems


def load_mlvu_entries(mlvu_root: Path, tasks: list[str], limit: int,
                      sample: str = "stratified") -> list[dict]:
    """
    Load MLVU questions as videomme-style entries.

    Video filenames are NOT unique across task dirs (e.g. movie101_66.mp4
    exists in both 1_plotQA and 9_summary), so video_id carries the task
    stem: "<stem>/<file>.mp4" — also what makes resume keys collision-free.

    MCQ rows: candidates -> A-D letters; rows whose answer text is not among
    the candidates (or with a non-4 candidate count) are skipped and counted.
    Generation rows (no candidates, tasks 8/9) are kept with answer="" and
    flagged unscored downstream.
    """
    json_dir = mlvu_root / "json"
    video_root = mlvu_root / "video"
    entries: list[dict] = []
    n_bad_answer = 0
    for stem in tasks:
        json_path = json_dir / f"{stem}.json"
        if not json_path.exists():
            print(f"[EncPlBench] warning: task json missing: {json_path}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            video = str(row.get("video") or "")
            if not video:
                continue
            video_path = video_root / stem / video
            candidates = [str(c) for c in (row.get("candidates") or [])]
            answer_text = str(row.get("answer") or "")
            answer = ""
            if len(candidates) == 4:
                if answer_text not in candidates:
                    n_bad_answer += 1
                    continue
                answer = "ABCD"[candidates.index(answer_text)]
            elif candidates:
                continue
            entries.append({
                "video_id": f"{stem}/{video}",
                "video_path": str(video_path) if video_path.exists() else None,
                "question": str(row.get("question") or ""),
                "options": candidates,
                "answer": answer,
                "answer_text": answer_text,
                "scoring_points": row.get("scoring_points") or [],
                "task_type": str(row.get("question_type") or stem),
                "domain": stem,
                "duration": float(row.get("duration") or 0.0),
            })
    if n_bad_answer:
        print(f"[EncPlBench] warning: {n_bad_answer} MLVU rows skipped "
              f"(answer text not among candidates)")

    if limit > 0:
        if sample == "stratified":
            # Stratified by task (json stem): round-robin across buckets so an
            # interrupted partial run keeps a balanced task mix (unlike the
            # duration-bucket concatenation in the videomme loader).
            buckets: dict[str, list[dict]] = {}
            for e in entries:
                buckets.setdefault(e["domain"], []).append(e)
            n_buckets = len(buckets)
            if n_buckets:
                per_bucket = limit // n_buckets
                remainder = limit - per_bucket * n_buckets
                quotas = {stem: per_bucket for stem in buckets}
                for stem in sorted(buckets, key=lambda s: -len(buckets[s])):
                    if remainder <= 0:
                        break
                    if len(buckets[stem]) > quotas[stem]:
                        quotas[stem] += 1
                        remainder -= 1
                queues = [buckets[stem][:quotas[stem]] for stem in sorted(buckets)]
                entries = [e for tup in zip_longest(*queues)
                           for e in tup if e is not None][:limit]
        else:  # "head": first N rows in json-file order
            entries = entries[:limit]
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Video-MME / MLVU benchmark for the encoder_placement "
                    "experiment (SGLang engine sidecar)")
    parser.add_argument("--dataset", required=True, choices=["videomme", "mlvu"])
    parser.add_argument("--dataset-dir", default=os.environ.get("DATASET_DIR"),
                        help="dataset root: Video-MME HF dir (data/ + videos/) "
                             "or MLVU dir (json/ + video/, MLVU/ wrapper ok)")
    parser.add_argument("--base-url",
                        default=os.environ.get("SGLANG_ENGINE_BASE_URL",
                                               "http://localhost:8002"),
                        help="SGLang engine sidecar base URL; localhost means "
                             "single-host (loopback netem on the base-url port), "
                             "a remote host means dual-host (whole-interface "
                             "netem on this machine's egress)")
    parser.add_argument("--interface", default="auto",
                        help="egress interface for dual-host netem shaping "
                             "(default: auto-detect; ignored single-host)")
    parser.add_argument("--profile", default="no_emulation",
                        help="network profile from configs/profiles.yaml, "
                             "applied as real tc/netem shaping "
                             "(no_emulation = bypass)")
    parser.add_argument("--netem-direction", default="both",
                        choices=["both", "uplink-only"],
                        help="both: shape upload and SSE download; "
                             "uplink-only: shape only client->server upload")
    # ---- stream safety timeouts (forwarded verbatim to the client) ----
    parser.add_argument("--stream-connect-timeout", type=float, default=None,
                        metavar="SEC",
                        help="TCP connect timeout for sidecar HTTP requests "
                             "(default: client class default 20; 0 disables)")
    parser.add_argument("--stream-first-event-timeout", type=float,
                        default=None, metavar="SEC",
                        help="budget from chat POST to the first SSE event "
                             "(default: client class default 200; 0 disables)")
    parser.add_argument("--stream-idle-timeout", type=float, default=None,
                        metavar="SEC",
                        help="max silent gap between SSE events mid-stream "
                             "(default: client class default 20; 0 disables)")
    parser.add_argument("--stream-upload-read-timeout", type=float,
                        default=None, metavar="SEC",
                        help="read timeout for the bundle-upload PUT response "
                             "(default: client class default 1200; 0 disables)")
    parser.add_argument("--mode", default=os.environ.get("ENC_MODE", "server"),
                        choices=["server", "edge"],
                        help="encoder placement: server (upload raw mp4) or "
                             "edge (local decode+processor+ViT, upload bundle); "
                             "ENC_MODE env wins as default")
    parser.add_argument("--model", default="aicodec/aicodec/models/qwen3_vl_4B")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all questions")
    parser.add_argument("--sample", default="stratified",
                        choices=["stratified", "head"],
                        help="limit>0 sampling: stratified (videomme: by "
                             "duration, mlvu: by task) or head (first N)")
    parser.add_argument("--tasks", default=None,
                        help="mlvu only: 'mcq' (default, tasks 1-7), 'all' "
                             "(adds unscored generation tasks 8/9), or "
                             "comma-separated stems (e.g. 2_needle,3_ego); "
                             "ignored for videomme")
    parser.add_argument("--max-frames", type=int, default=768,
                        help="cap on sampled frames for BOTH sampling "
                             "modes (native sglang 0.5.20 whitelist key — "
                             "no patch needed). Default 768 = sglang's "
                             "FPS_MAX_FRAMES (i.e. effectively off)")
    parser.add_argument("--frame-ratio", type=float, default=None,
                        metavar="R",
                        help="ratio-of-native frame sampling (v3): sglang's "
                             "own base (total_frames/video_fps*fps, i.e. "
                             "2fps x duration BEFORE clamping) multiplied "
                             "by R, then the native clamp chain (min 4, "
                             "cap --max-frames/768, <= total, floor to a "
                             "multiple of 2). In (0, 1]; 0/omit = off. "
                             "EDGE mode: sampled locally via the patched "
                             "smart_nframes (canary-verified per request "
                             "when active). SERVER mode: the ENGINE "
                             "samples with its STARTUP-time "
                             "mm_process_config — START THE SIDECAR WITH "
                             "THE SAME --frame-ratio (the value rides in "
                             "the request body for record-keeping; the "
                             "sidecar warns on mismatch). REQUIRES the "
                             "sglang frame_ratio patch v3 "
                             "(patch_host_sglang.py; it upgrades an old "
                             "patch in place) on the machine that "
                             "samples. Runs are keyed apart in the "
                             "resume tag and get a distinct output dir. "
                             "NOTE: results files carry a frame-sampling "
                             "semantics version; a file written under "
                             "the OLD semantics (ratio x total_frames) "
                             "is REFUSED for --resume — start a fresh "
                             "results file (the old one remains on disk "
                             "as the old-semantics baseline)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--compressor", default="",
                        choices=["", "vidcom2"],
                        help="edge mode only: 'vidcom2' prunes the edge "
                             "bundle to --retention-ratio of the video "
                             "tokens using the VidCom2 selector (in-repo "
                             "checkout under aicodec/aicodec/models/VidCom2); "
                             "'' uploads all tokens")
    parser.add_argument("--retention-ratio", type=float, default=0.25,
                        help="VidCom2 token keep ratio in (0, 1] (only "
                             "used with --compressor vidcom2 in edge mode)")
    parser.add_argument("--bundle-quant", default="",
                        choices=["", "fp8"],
                        help="edge mode + vidcom2 only: quantize the "
                             "pruned feature tensor to fp8 e4m3 (per-tensor "
                             "scale) before upload, halving the uploaded "
                             "feature bytes; the server dequantizes after "
                             "deserialization; '' uploads raw bf16")
    parser.add_argument("--vision-only", action="store_true",
                        help="edge mode only: load just the vision tower "
                             "(Qwen3VLVisionModel, ~0.4B) instead of the "
                             "full 4B model — same weights, identical "
                             "embeddings, ~8GB -> ~1GB edge VRAM; does NOT "
                             "change outputs (no separate results dir)")
    parser.add_argument("--attn-impl", default="sdpa",
                        choices=["sdpa", "flash_attention_4", "eager"],
                        help="edge mode only: attention backend for the "
                             "edge ViT. sdpa (default, bit-reproducible); "
                             "flash_attention_4 (flash-attn-4 kernels, "
                             "NOT bit-identical to sdpa); eager (reference). "
                             "flash_attention_2 is unsupported on this "
                             "stack (flash-attn-4 beta fails the fa2 gate). "
                             "Separate results dir -fa4 / -eager so runs "
                             "are never mixed")
    parser.add_argument("--edge-cpus", default=None, metavar="SPEC",
                        help="edge-mode CPU budget: pin THIS process "
                             "(decode + processor + ViT) to a cpu set for "
                             "the whole run, mirroring the scenario's "
                             "edge_cpus. Default 'auto-half-other' "
                             "(SECOND half of the cores — disjoint from "
                             "the sidecar's default --server-cpus "
                             "auto-half); also 'auto-half', a cpu list "
                             "with ranges ('48-95,100'), or '' to "
                             "disable. ENCPL_EDGE_CPUS env wins as "
                             "default. Server mode warns and ignores. "
                             "The applied spec lands in every entry's "
                             "encpl_stats.edge_cpus")
    parser.add_argument("--decode-threads", type=int, default=None,
                        metavar="N",
                        help="cap decord's INTERNAL decode thread pool "
                             "(SGLANG_DECORD_NUM_THREADS, read at "
                             "VideoReader construction — same knob as the "
                             "sidecar's --decode-threads). decord sizes "
                             "its pool by HOST core count regardless of "
                             "--edge-cpus / OMP / torch knobs. Edge mode: "
                             "exported here BEFORE the first request. "
                             "Server mode: warns and ignores — restart the "
                             "sidecar with the SAME --decode-threads for "
                             "the engine-side cap. ENCPL_DECODE_THREADS "
                             "env wins as default. The applied value lands "
                             "in every entry's encpl_stats.decode_threads")
    parser.add_argument("--output", default=None,
                        help="output dir (default logs/encpl_bench/"
                             "<dataset>/<mode>[-vidcom2-rR][-fp8]")
    parser.add_argument("--resume", action="store_true",
                        help="keep the existing results file and skip questions "
                             "already answered in it; without this flag each run "
                             "starts a fresh report (previous file archived). "
                             "REFUSED when the results file was written under "
                             "a different frame-sampling semantics version "
                             "(see FRAME_SAMPLING_SEMANTICS) — mixing "
                             "old- and new-semantics entries would silently "
                             "skew the accuracy; start a fresh file instead")
    parser.add_argument("--prune", action="store_true",
                        help="drop result entries not in the current sample "
                             "(stale entries from earlier runs/samples)")
    parser.add_argument("--metrics-only", action="store_true",
                        help="only recompute accuracy/latency from an existing "
                             "results file (pass the same --dataset/--mode/"
                             "--profile/--output as the original run)")
    parser.add_argument("--no-flush-cache", action="store_true",
                        help="skip the POST /flush_cache before every question "
                             "(default: flush SGLang radix + mm preprocess "
                             "caches so repeat-video questions are not served "
                             "from a warm prefix)")
    parser.add_argument("--wait-sidecar", type=float, default=900,
                        metavar="SEC",
                        help="preflight GET /v1/models retry budget while "
                             "the sidecar builds its engine before opening "
                             "the port (900 default; 0 = try once and exit "
                             "on failure)")
    parser.add_argument("--no-model-check", action="store_true",
                        help="skip the preflight model-agreement check "
                             "(sidecar ids that are placeholders/anonymized "
                             "cannot be compared; a mismatch then only "
                             "warns)")
    parser.add_argument("--capture-pcap", action="store_true",
                        help="L3/L4 tcpdump capture of the sidecar traffic "
                             "spanning the WHOLE question loop (needs "
                             "passwordless sudo tcpdump; the tool EXITS if "
                             "the capture cannot start). Capture metadata "
                             "(pcap path/size/duration/filter) is written "
                             "to a sidecar JSON next to the results file "
                             "(<stem>_capture_meta.json)")
    parser.add_argument("--capture-interface", default=None,
                        help="interface to capture on (default: 'lo' when "
                             "--base-url is local, else the default-route "
                             "interface from 'ip route')")
    parser.add_argument("--capture-filter", default=None,
                        help="BPF filter (default: 'tcp port <sidecar port "
                             "from --base-url>')")
    parser.add_argument("--capture-dir",
                        default=str(AITESTBED_ROOT / "results" / "captures"),
                        help="directory for pcap files (default: "
                             "<repo>/aitestbed/results/captures, cwd-"
                             "independent)")
    args = parser.parse_args()

    if not args.dataset_dir:
        sys.exit("--dataset-dir (or the DATASET_DIR env var) is required")
    dataset_dir = Path(args.dataset_dir).expanduser()

    if args.compressor and args.mode != "edge":
        sys.exit("--compressor vidcom2 is only meaningful in edge mode "
                 "(the server path has no edge bundle to prune)")
    if not (0.0 < args.retention_ratio <= 1.0):
        sys.exit("--retention-ratio must be in (0, 1]")
    if args.bundle_quant:
        if args.mode != "edge":
            sys.exit("--bundle-quant fp8 is only meaningful in edge mode "
                     "(the server path has no edge bundle to quantize)")
        if args.compressor != "vidcom2":
            # Hard error, not a warning: silently skipping would produce
            # results files that LOOK like a quantization measurement but
            # contain raw bf16 uploads.
            sys.exit("--bundle-quant fp8 requires --compressor vidcom2 "
                     "(quantization applies to the pruned bundle)")
    if args.attn_impl != "sdpa":
        if args.mode != "edge":
            sys.exit("--attn-impl is only meaningful in edge mode "
                     "(server mode lets SGLang pick its own backend)")
    # 0 folds to off (same as the sidecar's build_mm_process_config):
    # --frame-ratio 0 disables percentage sampling instead of exiting.
    if args.frame_ratio == 0:
        args.frame_ratio = None
    if args.frame_ratio is not None and not (0.0 < args.frame_ratio <= 1.0):
        sys.exit("--frame-ratio must be in (0, 1] (0 or omit = off)")

    # Edge CPU budget: CLI > ENCPL_EDGE_CPUS env > auto-half-other (the
    # second half of the cores, disjoint from the sidecar's default
    # --server-cpus auto-half first half).
    edge_cpu_spec = (args.edge_cpus
                     if args.edge_cpus is not None
                     else os.environ.get("ENCPL_EDGE_CPUS", "auto-half-other"))
    try:
        own_affinity = os.sched_getaffinity(0)
    except AttributeError:
        own_affinity = None
    try:
        edge_cpu_list = _parse_edge_cpu_spec(edge_cpu_spec, own_affinity)
    except ValueError as exc:
        sys.exit(f"--edge-cpus: {exc}")
    if edge_cpu_list and args.mode != "edge":
        # Warning + ignore (not sys.exit): a stray env var from an edge
        # session should not kill a server-mode run; the budget affects
        # only the resource model, not correctness.
        print(f"[EncPlBench] warning: --edge-cpus={edge_cpu_spec!r} is set "
              f"but mode={args.mode!r} (no terminal-side model work); "
              f"edge CPU pinning is ignored")
        edge_cpu_list = None
    if edge_cpu_spec and edge_cpu_list is None and own_affinity is None:
        print(f"[EncPlBench] warning: --edge-cpus={edge_cpu_spec!r} given "
              f"but os.sched_getaffinity is unavailable on this platform; "
              f"edge CPU pinning is NOT applied")

    # decord decode-thread cap: CLI > ENCPL_DECODE_THREADS env (same
    # precedence pattern as --edge-cpus). Resource-model knob, not a
    # correctness knob — validated but NEVER a resume key / output-dir
    # component (matching edge_cpus' treatment).
    decode_threads = (args.decode_threads
                      if args.decode_threads is not None
                      else os.environ.get("ENCPL_DECODE_THREADS"))
    if decode_threads is not None:
        try:
            decode_threads = int(decode_threads)
        except (TypeError, ValueError):
            sys.exit(f"--decode-threads must be a positive integer, got "
                     f"{decode_threads!r}")
        if decode_threads <= 0:
            sys.exit(f"--decode-threads must be > 0 "
                     f"(omit for decord native default), got {decode_threads}")
        if args.mode != "edge":
            # Warning + ignore (same tolerance as --edge-cpus above): the
            # cap applies to THIS process's decord, which only decodes in
            # edge mode; the ENGINE-side cap is the sidecar's own flag.
            print(f"[EncPlBench] warning: --decode-threads={decode_threads} "
                  f"is set but mode={args.mode!r}; the ENGINE-side cap is "
                  f"the sidecar's --decode-threads flag (restart it with "
                  f"the same value), this flag is edge-only")
            decode_threads = None

    # A portless base URL would make the loopback netem filter default to the
    # videomme port (8001) while the client still connects elsewhere; pin the
    # sidecar port explicitly instead.
    parsed = urlparse(args.base_url)
    if parsed.scheme and parsed.port is None:
        args.base_url = args.base_url.rstrip("/") + ":8002"

    mode_dir = (args.mode + ("-vidcom2" if args.compressor else "")
                + (f"-r{args.retention_ratio:g}" if args.compressor else "")
                + ("-fp8" if args.bundle_quant else "")
                + ({"sdpa": "", "flash_attention_4": "-fa4",
                    "eager": "-eager"}[args.attn_impl])
                + (f"-fr{args.frame_ratio:g}" if args.frame_ratio else "")
                + ((f"-mf{args.max_frames}" if args.max_frames != 768 else "")))
    output_dir = Path(args.output or AITESTBED_ROOT / "logs" / "encpl_bench"
                      / args.dataset / mode_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"questions_results_{args.profile}.json"

    if args.metrics_only:
        compute_and_print(results_path, args.profile, args.model, args.dataset)
        return

    if args.dataset == "videomme":
        parquet_path = _vmme.find_parquet(dataset_dir)
        if parquet_path is None:
            sys.exit(f"no parquet found under {dataset_dir}")
        video_root = _vmme.find_video_root(dataset_dir)
        if video_root is None:
            sys.exit(f"no video dir (videozip/ or videos/) found under {dataset_dir}")
        entries = _vmme.load_entries(parquet_path, video_root,
                                     args.limit, args.sample)
        tasks = None
    else:
        mlvu_root = find_mlvu_root(dataset_dir)
        if mlvu_root is None:
            sys.exit(f"no MLVU layout (json/ + video/) found under {dataset_dir}")
        tasks = resolve_mlvu_tasks(args.tasks)
        entries = load_mlvu_entries(mlvu_root, tasks, args.limit, args.sample)

    if not entries:
        sys.exit("no questions loaded")
    missing = [e for e in entries if e["video_path"] is None]
    print(f"[EncPlBench] dataset={args.dataset} mode={args.mode} "
          f"model={args.model}")
    print(f"[EncPlBench] {len(entries)} questions loaded, "
          f"{len(missing)} missing video files")
    if args.dataset == "mlvu":
        mix = Counter(e["domain"] for e in entries)
        print("[EncPlBench] task mix: "
              + ", ".join(f"{k}={v}" for k, v in sorted(mix.items())))
    elif args.limit > 0 and args.sample == "stratified":
        dur_counts = Counter(e["duration"] for e in entries)
        print("[EncPlBench] duration mix: "
              + ", ".join(f"{k}={v}" for k, v in sorted(dur_counts.items())))

    # Sidecar preflight BEFORE anything destructive or expensive:
    # (a) connectivity with retry — the sidecar binds its port only
    # after the multi-minute engine build, so "bench started first"
    # waits here instead of crashing on question 1; (b) model
    # agreement — catches sidecar/client model drift BEFORE the run
    # (and before the resume/archive block below), so a failed
    # preflight leaves the previous results file untouched. Skipped
    # entirely under --metrics-only (returns above before this point).
    _preflight_sidecar(args.base_url, args.model,
                       wait_sec=args.wait_sidecar,
                       check_model=not args.no_model_check)
    # Environment context for the run log (decode backend, thread caps,
    # server CPUs) — best-effort, immediately after preflight and
    # BEFORE the resume/archive block below so it precedes all run
    # output.
    _print_sidecar_stats(args.base_url)

    # Same fresh-report/archive vs resume/prune policy as videomme_eval.py,
    # keyed on (video_id, question, mode). A results file from a DIFFERENT
    # frame-sampling semantics is refused outright (see
    # FRAME_SAMPLING_SEMANTICS): the resume tags only carry the ratio
    # VALUE, so old-semantics entries would silently mix in.
    results: list[dict] = []
    if args.resume and results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            results = loaded if isinstance(loaded, list) else []
        except Exception as e:
            print(f"[EncPlBench] warning: could not load existing results: {e}")
        if results:
            stamps = {
                (r.get("encpl_stats") or {}).get(
                    "frame_sampling_semantics") for r in results}
            stamps.discard(None)
            if stamps and stamps != {FRAME_SAMPLING_SEMANTICS}:
                sys.exit(
                    f"results file {results_path} contains entries written "
                    f"under frame-sampling semantics {sorted(stamps)} but "
                    f"this run uses {FRAME_SAMPLING_SEMANTICS} — the same "
                    f"--frame-ratio VALUE samples DIFFERENT frames under "
                    f"the two semantics, so resuming would silently mix "
                    f"them and skew the accuracy. Start a fresh results "
                    f"file (drop --resume; the old file will be archived "
                    f"on disk as the old-semantics baseline)")
            if not stamps:
                # pre-stamp file: written before the semantics version
                # existed — i.e. the OLD (ratio x total_frames) patch or
                # no frame_ratio at all. Refuse only when this run
                # actually carries a ratio (a no-ratio --resume over a
                # no-ratio old file is still apples-to-apples).
                if args.frame_ratio:
                    sys.exit(
                        f"results file {results_path} predates the "
                        f"frame-sampling semantics stamp (old ratio x "
                        f"total_frames semantics or no frame_ratio) but "
                        f"this run uses --frame-ratio under "
                        f"{FRAME_SAMPLING_SEMANTICS} — mixing would "
                        f"silently skew the accuracy. Start a fresh "
                        f"results file (drop --resume)")
    elif results_path.exists():
        archive = results_path.with_name(
            results_path.stem + "_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
        os.replace(results_path, archive)
        print(f"[EncPlBench] fresh report: archived previous results to {archive.name}")

    if results:
        # Resume key carries the compression + quantization + attention
        # + sampling settings: a vidcom2 run must NOT reuse uncompressed
        # entries, an fp8 run must NOT reuse bf16 entries, an fa2/eager
        # run must NOT reuse sdpa entries, and a frame_ratio /
        # max_frames run must NOT reuse default-sampling entries
        # (different sampled frames -> different answers expected).
        def _run_tag(mode: str, compressor: str, ratio: float,
                     quant: str = "", attn: str = "sdpa",
                     frame_ratio: float | None = None,
                     max_frames: int | None = None) -> tuple:
            return (mode, compressor or None,
                    round(ratio, 4) if compressor else None,
                    quant or None,
                    attn or None,
                    round(frame_ratio, 4) if frame_ratio else None,
                    int(max_frames) if max_frames else None)

        def _stats_run_tag(r: dict) -> tuple:
            stats = r.get("encpl_stats") or {}
            return _run_tag(
                r.get("mode", args.mode),
                (stats.get("retention_ratio") is not None
                 and "vidcom2" or ""),
                stats.get("retention_ratio", args.retention_ratio),
                "fp8" if stats.get("quant_dtype") else None,
                stats.get("attn_impl") or "sdpa"
                if r.get("mode") == "edge" else None,
                stats.get("frame_ratio"),
                stats.get("max_frames"),
            )

        run_tag = _run_tag(args.mode, args.compressor,
                           args.retention_ratio, args.bundle_quant,
                           args.attn_impl if args.mode == "edge" else None,
                           args.frame_ratio,
                           args.max_frames)
        current_keys = {
            (e["video_id"], e["question"], run_tag) for e in entries
        }
        stale = [r for r in results
                 if (r.get("video_id"), r.get("question"),
                     _stats_run_tag(r))
                 not in current_keys]
        if stale:
            if args.prune:
                results = [r for r in results
                           if (r.get("video_id"), r.get("question"),
                               _stats_run_tag(r))
                           in current_keys]
                _save_results(results_path, results)
                print(f"[EncPlBench] pruned {len(stale)} stale entries "
                      f"(not in current sample)")
            else:
                print(f"[EncPlBench] warning: {len(stale)} stale entries in "
                      f"results file are not in the current sample and WILL "
                      f"pollute the final accuracy. Use --prune to drop them, "
                      f"or drop --resume for a fresh report")

    def _entry_run_tag(r: dict) -> tuple:
        stats = r.get("encpl_stats") or {}
        mode = r.get("mode", args.mode)
        # Edge-mode knobs only: server-mode entries carry the DEFAULT
        # attn_impl in their stats (the client always reports one), but
        # the server arm never loads the edge ViT — its attn value is
        # measurement-irrelevant and must NOT split the resume key, or
        # a server-mode --resume would never match (the entry says
        # "sdpa", the current tag said "sdpa" but past entries with
        # OTHER values would look like different runs) and duplicate
        # entries would silently double-count the accuracy.
        is_edge = mode == "edge"
        if is_edge and stats.get("vidcom2_active"):
            return (mode, "vidcom2",
                    round(float(stats.get("retention_ratio",
                                           args.retention_ratio)), 4),
                    "fp8" if stats.get("quant_dtype") else None,
                    stats.get("attn_impl") or "sdpa",
                    round(float(stats["frame_ratio"]), 4)
                    if stats.get("frame_ratio") else None,
                    int(stats["max_frames"])
                    if stats.get("max_frames") else None)
        # Non-vidcom2 entries (either mode): compressor slot None even
        # when stats carry a retention_ratio (a plain edge run without
        # compression records retention_ratio=1.0/args default in some
        # client versions — it is NOT a vidcom2 run).
        return (mode, None, None,
                "fp8" if stats.get("quant_dtype") else None,
                stats.get("attn_impl") or "sdpa" if is_edge else None,
                round(float(stats["frame_ratio"]), 4)
                if stats.get("frame_ratio") else None,
                int(stats["max_frames"])
                if stats.get("max_frames") else None)

    done_keys = {
        (r.get("video_id"), r.get("question"), _entry_run_tag(r))
        for r in results if r.get("mode") == args.mode
    }
    # Must be built with the SAME normalization as _entry_run_tag (the
    # historical cur_tag kept args.attn_impl unconditionally and the
    # quant slot as ""-or-None inconsistently, so server-mode --resume
    # never matched and re-ran — and re-APPENDED — every entry).
    cur_tag = _entry_run_tag({
        "mode": args.mode,
        "encpl_stats": {
            "vidcom2_active": bool(
                args.compressor and args.mode == "edge"),
            "retention_ratio": args.retention_ratio,
            "quant_dtype": "fp8" if args.bundle_quant else "",
            "attn_impl": args.attn_impl,
            "frame_ratio": args.frame_ratio,
            "max_frames": args.max_frames,
        },
    })

    from clients.encoder_placement_client import EncoderPlacementClient
    client = EncoderPlacementClient(
        base_url=args.base_url,
        stream_connect_timeout=args.stream_connect_timeout,
        stream_first_event_timeout=args.stream_first_event_timeout,
        stream_idle_timeout=args.stream_idle_timeout,
        stream_upload_read_timeout=args.stream_upload_read_timeout)

    # Client cache warmup (serial bench counterpart of the concurrent
    # bench's _warm_client_caches): populates the processor + edge
    # visual tower BEFORE the first question so the FIRST measured
    # entry does not pay transformers import + model load inside its
    # e2e_ttft/e2e_total window. Best-effort (a failure is warned and
    # left to surface inside the first request with full context).
    # Runs AFTER the CPU pin below so the load honors the budget —
    # but BEFORE the question loop; nothing timed happens here.
    def _warm_client_caches():
        t0 = time.time()
        try:
            client._get_processor(args.model)
        except Exception as exc:
            print(f"[EncPlBench] warning: processor warmup failed "
                  f"({exc}); the first question will retry the load")
        if args.mode == "edge":
            try:
                client._get_edge_model(
                    args.model,
                    vision_only=bool(args.vision_only),
                    attn_impl=args.attn_impl)
            except Exception as exc:
                print(f"[EncPlBench] warning: edge-model warmup failed "
                      f"({exc}); the first question will retry the load")
            # The edge bundle path lazily imports sglang's qwen_vl
            # preprocess + VideoDecoderWrapper INSIDE _prepare_edge_bundle
            # (i.e. inside the e2e window). Pre-import them here so the
            # first measured entry does not pay the multi-second sglang
            # import. Best-effort: a missing sglang (mock envs) warns.
            try:
                from sglang.srt.multimodal.processors.qwen_vl import (
                    preprocess_video as _pv)
                from sglang.srt.utils.video_decoder import (
                    VideoDecoderWrapper as _vdw)
                assert _pv and _vdw
            except Exception as exc:
                print(f"[EncPlBench] warning: sglang edge-path warmup "
                      f"failed ({exc}); the first question will pay the "
                      f"import inside its window")
        print(f"[EncPlBench] client caches warmed in "
              f"{time.time() - t0:.1f} s (processor"
              + (" + edge visual tower + sglang edge path"
                 if args.mode == "edge" else "")
              + f"; mode={args.mode})")

    # Edge CPU budget BEFORE the first request (same window as netem):
    # the client's heavy libs (torch, decord, transformers) are lazily
    # imported inside the request path, so first-touch import AND model
    # load happen inside the budget. Torn down in the finally below.
    _cpu_restore = _apply_edge_cpu_budget(edge_cpu_list)
    # token None = pin NOT applied (disabled/unavailable/failed);
    # truthy token = pinned, restore data inside.
    applied_edge_cpus = (
        ",".join(str(c) for c in edge_cpu_list)
        if (_cpu_restore is not None) else None)

    # decord thread cap BEFORE the first request (same window as the CPU
    # pin above): decord is lazily imported inside the request path, and
    # the patched video_decoder.py reads SGLANG_DECORD_NUM_THREADS at
    # VideoReader construction time. setdefault: a pre-set env var wins
    # (explicit override); the applied value is read back for the stats.
    applied_decode_threads = None
    if decode_threads is not None:
        os.environ.setdefault("SGLANG_DECORD_NUM_THREADS",
                              str(decode_threads))
        applied_decode_threads = int(
            os.environ["SGLANG_DECORD_NUM_THREADS"])
        print(f"[EncPlBench] decode_threads: capped to "
              f"{applied_decode_threads} (SGLANG_DECORD_NUM_THREADS; env "
              f"name historical — caps BOTH backends)")

    # Edge mode decodes HERE (client-side): name the local decode
    # backend next to the sidecar /stats line's engine-side backend so
    # an edge run log carries BOTH arms' decode environments. Server
    # mode skips it — decode happens in the engine, already covered by
    # _print_sidecar_stats above.
    if args.mode == "edge":
        _print_client_decode_backend()

    # Warm the client caches AFTER the CPU pin + decode-thread env are
    # in place (the load itself then honors both budgets) and BEFORE
    # the question loop: keeps the heavy first-touch import/model load
    # out of every timing window.
    _warm_client_caches()

    # Apply the network profile AFTER loading entries and BEFORE the first
    # request; torn down in the finally below (same contract as videomme_eval).
    _vmme.apply_netem(args.base_url, args.profile, args.netem_direction,
                      interface=args.interface)

    # pcap capture spans the WHOLE question loop (started after netem so
    # the pcap shows the shaped traffic; stopped in the finally below,
    # before netem teardown, so the capture window is inside the shaped
    # window). Fail-fast preflight: a requested capture that cannot
    # start is a hard error, never a silent 0-byte phantom.
    pcap_controller = None
    pcap_start_time = None
    pcap_iface = None
    capture_filter = None
    if args.capture_pcap:
        from capture.controller import CaptureController
        pcap_iface = args.capture_interface or _resolve_capture_interface(
            args.base_url)
        _preflight_capture(pcap_iface, Path(args.capture_dir))
        sidecar_port = _vmme._base_url_port(args.base_url)
        capture_filter = args.capture_filter or f"tcp port {sidecar_port}"
        try:
            pcap_controller = CaptureController(
                interface=pcap_iface,
                capture_dir=args.capture_dir,
            )
            ts = time.strftime("%Y%m%d_%H%M%S")
            arm = f"{args.dataset}_{args.mode}"
            pcap_start_time = time.time()
            pcap_file = pcap_controller.start(
                filename=f"encpl_bench_{arm}_{args.profile}_{ts}.pcap",
                filter_expr=capture_filter,
            )
            if not pcap_file:
                sys.exit(f"--capture-pcap: tcpdump failed to start on "
                         f"'{pcap_iface}' (check the interface name and "
                         f"BPF filter '{capture_filter}')")
            print(f"[EncPlBench] pcap: {pcap_file} "
                  f"(iface={pcap_iface} filter='{capture_filter}')")
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(f"--capture-pcap: failed to start tcpdump capture "
                     f"({e})")

    model_key = Path(args.model).name
    n_done = 0
    t_start = time.time()
    total = sum(1 for e in entries
                if e["video_path"]
                and (e["video_id"], e["question"], cur_tag) not in done_keys)

    try:
        for idx, e in enumerate(entries):
            key = (e["video_id"], e["question"], cur_tag)
            if key in done_keys or e["video_path"] is None:
                continue

            if e["options"]:
                prompt = _vmme.build_prompt(e["question"], e["options"])
            else:
                # Generation task (mlvu 8/9): no options -> free-form answer,
                # recorded unscored.
                prompt = e["question"]
            q0 = {
                "task_type": e["task_type"],
                "question": e["question"],
                "answer": e["answer"],
                "options": [o if ". " in o[:4] else f"{'ABCD'[i]}. {o}"
                            for i, o in enumerate(e["options"])],
                model_key: "",
            }
            for extra in ("answer_text", "scoring_points"):
                if e.get(extra):
                    q0[extra] = e[extra]
            entry = {
                "time": "",
                "captions": "",
                "questions": [q0],
                "video_categories": e["domain"],
                "video_path": e["video_path"],
                "video_id": e["video_id"],
                # Top-level question mirrors questions[0] so the resume key
                # (video_id, question, mode) matches without reaching into
                # the nested record (videomme_eval.py now also writes the
                # top-level question; the mirror is kept so entries produced
                # by either tool round-trip through the same key).
                "question": e["question"],
                "duration": e["duration"],
                "mode": args.mode,
                "dataset": args.dataset,
                "compressor": args.compressor,
                "retention_ratio": (
                    args.retention_ratio if args.compressor else None),
                "bundle_quant": args.bundle_quant or None,
                "vision_only": bool(args.vision_only),
                "attn_impl": args.attn_impl,
                "frame_ratio": (
                    float(args.frame_ratio) if args.frame_ratio else None),
                "max_frames": (
                    int(args.max_frames) if args.max_frames else None),
            }

            # Cold-prefix guarantee: drop radix/mm caches BEFORE the request
            # so repeat-video questions are not served from a warm prefix
            # (outside all timing windows; failure is non-fatal, warned).
            if not args.no_flush_cache:
                flush_server_cache(args.base_url)

            try:
                response = client.generate_content_with_video(
                    prompt=prompt,
                    video_path=e["video_path"],
                    model=args.model,
                    mode=args.mode,
                    max_frames=args.max_frames,
                    max_tokens=args.max_tokens,
                    compressor=args.compressor if args.mode == "edge" else "",
                    retention_ratio=args.retention_ratio,
                    bundle_quant=args.bundle_quant if args.mode == "edge" else "",
                    vision_only=bool(args.vision_only),
                    attn_impl=args.attn_impl,
                    # serial bench: no per-request tag (historical
                    # behaviour; the per-user tag is a concurrent-wave
                    # knob in encoder_placement_concurrent.py)
                    user_tag="",
                    frame_ratio=args.frame_ratio,
                )
                entry["questions"][0][model_key] = response.total_content or ""
                stats = getattr(response, "engine_stats", None)
                if stats:
                    # Full client engine_stats: mode, upload bytes/time,
                    # e2e_ttft/e2e_total, server t_prepare/server_ttft, and
                    # (edge) stage1_edge/stage2_vit/ser_time/bundle_bytes.
                    # video_path is dropped (already on the entry).
                    entry["encpl_stats"] = {
                        k: v for k, v in stats.items() if k != "video_path"}
                # Terminal CPU budget actually applied (comma cpu list;
                # null = unpinned) — recorded on EVERY entry (not only
                # when stats exist) so the compute_and_print summary and
                # --metrics-only can reconstruct the run's budget.
                if entry.get("encpl_stats") is None:
                    entry["encpl_stats"] = {}
                entry["encpl_stats"]["edge_cpus"] = applied_edge_cpus
                # decord cap actually in force for THIS entry (int, or
                # null = decord native) — same unconditional recording
                # as edge_cpus above so --metrics-only can reconstruct
                # the run's decode-thread budget.
                entry["encpl_stats"]["decode_threads"] = (
                    applied_decode_threads)
                n_done += 1
            except KeyboardInterrupt:
                print("\n[EncPlBench] interrupted, saving partial results")
                results.append(entry)
                break
            except Exception as exc:
                print(f"[EncPlBench] Q{idx} ({e['video_id']}) failed: {exc}")
                entry["questions"][0][model_key] = ""
                entry["error"] = str(exc)

            results.append(entry)
            if n_done % 5 == 0 or n_done == total:
                _save_results(results_path, results)
                elapsed = time.time() - t_start
                eta = elapsed / max(n_done, 1) * max(total - n_done, 0)
                print(f"[EncPlBench] progress {n_done}/{total} "
                      f"({elapsed/60:.1f}m elapsed, ~{eta/60:.1f}m left)")

        _save_results(results_path, results)
    finally:
        # pcap stop BEFORE netem teardown so the capture window stays
        # strictly inside the shaped window.
        if pcap_controller is not None:
            try:
                pcap_stop_time = time.time()
                pcap_path = pcap_controller.stop()
                if pcap_path:
                    capture_duration = (
                        pcap_stop_time - pcap_start_time
                        if pcap_start_time is not None else None)
                    pcap_size = 0
                    try:
                        pcap_size = pcap_path.stat().st_size
                    except Exception:
                        pass
                    capture_metadata = {
                        "type": "pcap_capture",
                        "pcap_file": str(pcap_path),
                        "capture_dir": args.capture_dir,
                        "capture_filter": capture_filter,
                        "capture_duration_sec": capture_duration,
                        "interface": pcap_iface,
                        "success": pcap_size > 0,
                        "results_file": str(results_path),
                        "timestamp": time.time(),
                    }
                    # The tool has no SQLite DB and the results file is a
                    # bare list (MetricsCalculator.load_task_results
                    # consumes exactly that); capture metadata goes to a
                    # sidecar JSON next to it — same dir, same stem, the
                    # "-capture_meta" suffix. Overwritten on every
                    # capturing run; --resume keeps the results but a
                    # fresh capture rewrites this sidecar.
                    sidecar = results_path.with_name(
                        results_path.stem + "_capture_meta.json")
                    with open(sidecar, "w", encoding="utf-8") as f:
                        json.dump(capture_metadata, f, indent=2)
                    print(f"[EncPlBench] pcap stopped: {pcap_path} "
                          f"({pcap_size} bytes, meta -> {sidecar.name})"
                          + ("" if pcap_size else
                             " WARNING: 0 bytes captured — no packets "
                             "matched the filter/interface"))
            except Exception as e:
                print(f"[EncPlBench] warning: failed to stop/log tcpdump "
                      f"capture cleanly ({e})")
        _vmme.clear_netem(args.base_url)
        _restore_edge_cpu_budget(_cpu_restore)
    compute_and_print(results_path, args.profile, args.model, args.dataset)


def compute_and_print(results_path: Path, profile: str, model: str,
                      dataset: str) -> None:
    from analysis.metrics import MetricsCalculator

    if not results_path.exists():
        sys.exit(f"no results file at {results_path}; run the evaluation first")
    data = MetricsCalculator.load_task_results(str(results_path.parent), profile)
    if not data:
        sys.exit(f"no results loaded from {results_path}")
    model_key = Path(model).name

    # Scored = 4-option MCQ entries; mlvu generation tasks (no options) are
    # recorded but excluded from the "real"-class accuracy.
    scored: list[dict] = []
    unscored: list[dict] = []
    for e in data:
        q = (e.get("questions") or [{}])[0] if e.get("questions") else {}
        if len(q.get("options") or []) == 4:
            scored.append(e)
        else:
            unscored.append(e)

    stats = MetricsCalculator.calculate_task_accuracy(
        scored, model_key, po_thd=3, task_class="real")

    # Terminal CPU budget actually applied per entry (from
    # encpl_stats.edge_cpus; null = unpinned): surfaces the run's
    # edge-arm resource model, incl. for --metrics-only re-reads of an
    # old results file. Mixed values mean a resumed run changed the
    # budget mid-way — each is shown with its count.
    cpu_counts = Counter(
        (e.get("encpl_stats") or {}).get("edge_cpus") or "unpinned"
        for e in data)
    if cpu_counts:
        print(f"\n[EncPlBench] ===== edge cpu budget ({len(data)} entries) =====")
        for cpus, n in sorted(cpu_counts.items(), key=lambda kv: -kv[1]):
            print(f"  edge_cpus: {cpus} (n={n})")

    # decord decode-thread cap actually applied per entry (from
    # encpl_stats.decode_threads; null = decord native default).
    # Mixed values mean a resumed run changed the cap mid-way — each is
    # shown with its count (mirrors the edge cpu budget block above).
    dt_counts = Counter(
        (e.get("encpl_stats") or {}).get("decode_threads")
        if (e.get("encpl_stats") or {}).get("decode_threads") is not None
        else "decord-native"
        for e in data)
    if dt_counts:
        print(f"\n[EncPlBench] ===== decode threads ({len(data)} entries) =====")
        for dt, n in sorted(dt_counts.items(), key=lambda kv: -kv[1]):
            print(f"  decode_threads: {dt} (n={n})")

    print(f"\n[EncPlBench] ===== accuracy ({dataset}, {len(scored)} scored) =====")
    for task_type, counts in sorted(stats.items()):
        if task_type == "total":
            continue
        print(f"  {task_type:35s} total={counts.get('total', 0):5d} "
              f"correct={counts.get('correct', 0):5d} "
              f"accuracy={counts.get('accuracy', 0.0):.3f}")
    overall = stats.get("total", {})
    if overall:
        print(f"  OVERALL: {overall.get('correct', 0)}/{overall.get('total', 0)} "
              f"= {overall.get('accuracy', 0.0):.3f}")
    n_failed = 0
    for e in scored:
        q = (e.get("questions") or [{}])[0]
        if not MetricsCalculator.extract_answer(q.get(model_key, "") or ""):
            n_failed += 1
    if n_failed:
        print(f"  ({n_failed} answers failed to parse / empty)")
    if unscored:
        n_err = sum(1 for e in unscored if e.get("error"))
        print(f"  ({len(unscored)} generation-task entries stored unscored, "
              f"{n_err} with errors)")

    def bucket_of(entry: dict) -> str:
        # videomme: short/medium/long duration buckets; mlvu: task_type.
        if dataset == "videomme":
            return str(entry.get("duration") or "unknown")
        q = (entry.get("questions") or [{}])[0] if entry.get("questions") else {}
        return str(q.get("task_type") or entry.get("video_categories") or "unknown")

    bucket_stats: dict[str, dict[str, int]] = {}
    for e in scored:
        q = (e.get("questions") or [{}])[0]
        ans = MetricsCalculator.extract_answer(q.get(model_key, "") or "")
        if not ans:
            continue
        b = bucket_of(e)
        s = bucket_stats.setdefault(b, {"total": 0, "correct": 0})
        s["total"] += 1
        if ans == (q.get("answer") or "").strip().upper():
            s["correct"] += 1
    if bucket_stats:
        label = "duration" if dataset == "videomme" else "task_type"
        print(f"\n[EncPlBench] ===== accuracy by {label} =====")
        for b, s in sorted(bucket_stats.items()):
            acc = s["correct"] / s["total"] if s["total"] else 0.0
            print(f"  {b:35s} total={s['total']:5d} correct={s['correct']:5d} "
                  f"accuracy={acc:.3f}")

    def stats_of(entry: dict) -> dict:
        return entry.get("encpl_stats") or {}

    ttft_vals = [stats_of(e)["e2e_ttft"] for e in data
                 if stats_of(e).get("e2e_ttft") is not None]
    e2e_vals = [stats_of(e)["e2e_total"] for e in data
                if stats_of(e).get("e2e_total") is not None]
    n_no_stats = sum(1 for e in data if not stats_of(e))

    def _lat_summary(vals: list[float]) -> str:
        if not vals:
            return "n=0"
        vs = sorted(vals)
        p50 = MetricsCalculator._percentile(vs, 50)
        p95 = MetricsCalculator._percentile(vs, 95)
        return (f"n={len(vs)} | mean={sum(vs)/len(vs):.1f} s | "
                f"p50={p50:.1f} s | p95={p95:.1f} s | "
                f"min={vs[0]:.1f} s | max={vs[-1]:.1f} s")

    if ttft_vals or e2e_vals:
        print("\n[EncPlBench] ===== E2E latency (video input -> token) =====")
        if ttft_vals:
            print(f"  TTFT  (first token): {_lat_summary(ttft_vals)}")
        if e2e_vals:
            print(f"  TOTAL (full answer): {_lat_summary(e2e_vals)}")
        if n_no_stats:
            print(f"  ({n_no_stats} entries without encpl_stats)")

        bucket_lat: dict[str, list[float]] = {}
        for e in data:
            v = stats_of(e).get("e2e_ttft")
            if v is None:
                continue
            bucket_lat.setdefault(bucket_of(e), []).append(v)
        if bucket_lat:
            label = "duration" if dataset == "videomme" else "task_type"
            print(f"\n  E2E TTFT by {label}:")
            for b, vs in sorted(bucket_lat.items()):
                sv = sorted(vs)
                print(f"    {b:20s} n={len(vs):3d} mean={sum(vs)/len(vs):6.1f} s "
                      f"p50={MetricsCalculator._percentile(sv, 50):6.1f} s "
                      f"p95={MetricsCalculator._percentile(sv, 95):6.1f} s")

    # Upload footprint: server mode uploads the raw mp4, edge mode the
    # serialized precomputed_embedding bundle — the headline traffic delta.
    up_payload = [stats_of(e)["uploaded_payload_bytes"] for e in data
                  if stats_of(e).get("uploaded_payload_bytes")]
    up_wire = [stats_of(e)["uploaded_wire_bytes"] for e in data
               if stats_of(e).get("uploaded_wire_bytes")]
    if up_payload:
        print(f"\n[EncPlBench] ===== upload ({len(up_payload)} requests) =====")
        print(f"  payload: mean={sum(up_payload)/len(up_payload)/1e6:.2f} MB | "
              f"min={min(up_payload)/1e6:.2f} MB | "
              f"max={max(up_payload)/1e6:.2f} MB")
        if up_wire:
            print(f"  wire:    mean={sum(up_wire)/len(up_wire)/1e6:.2f} MB")

    stage_keys = ("upload_time", "t_prepare", "stage1_edge", "stage2_vit",
                  "d2h_time", "ser_time", "server_ttft", "edge_prep_time")
    stage_vals = {k: [stats_of(e)[k] for e in data
                      if stats_of(e).get(k) is not None]
                  for k in stage_keys}
    present = {k: v for k, v in stage_vals.items() if v}
    if present:
        print("[EncPlBench] ===== stage means (s) =====")
        print("  " + " | ".join(f"{k}={sum(v)/len(v):.2f}"
                                for k, v in present.items()))

    # Fine-grained edge timeline (new-schema entries only; old JSONs
    # simply lack these keys and are skipped):
    #   stage1 = open + decode_sample + processor
    #   stage2 = h2d + vit_fwd + assemble
    #   opt    = vidcom2_time + quant_time + drop_est_time (riders)
    #   wire   = d2h + ser + upload
    def _mean(key, sub=None):
        vals = []
        for e in data:
            s = stats_of(e)
            if sub is not None:
                br = s.get(key)
                if isinstance(br, dict) and br.get(sub) is not None:
                    vals.append(br[sub])
            elif s.get(key) is not None:
                vals.append(s[key])
        return (sum(vals) / len(vals)) if vals else None

    b1 = {k: _mean("stage1_breakdown", k)
          for k in ("open", "decode_sample", "processor")}
    b2 = {k: _mean("stage2_breakdown", k)
          for k in ("h2d", "vit_fwd", "assemble")}
    if any(v is not None for v in b1.values()):
        print("  stage1: " + " | ".join(
            f"{k}={v:.2f}" for k, v in b1.items() if v is not None))
    if any(v is not None for v in b2.values()):
        print("  stage2: " + " | ".join(
            f"{k}={v:.2f}" for k, v in b2.items() if v is not None))
    riders = {k: _mean(k) for k in ("vidcom2_time", "quant_time",
                                    "drop_est_time")}
    riders = {k: v for k, v in riders.items() if v is not None}
    if riders:
        print("  opt:    " + " | ".join(
            f"{k}={v:.2f}" for k, v in riders.items()))
    wire = {k: _mean(k) for k in ("d2h_time", "ser_time", "upload_time")}
    wire = {k: v for k, v in wire.items() if v is not None}
    if wire:
        print("  wire:   " + " | ".join(
            f"{k}={v:.2f}" for k, v in wire.items()))
    bundles = [stats_of(e)["bundle_bytes"] for e in data
               if stats_of(e).get("bundle_bytes")]
    if bundles:
        print(f"  bundle: mean={sum(bundles)/len(bundles)/1e6:.2f} MB | "
              f"min={min(bundles)/1e6:.2f} MB | "
              f"max={max(bundles)/1e6:.2f} MB")
        dropped = [stats_of(e)["bundle_bytes_dropped"] for e in data
                   if stats_of(e).get("bundle_bytes_dropped")]
        if dropped:
            pct = (sum(dropped) / (sum(bundles) + sum(dropped)) * 100
                   if (sum(bundles) + sum(dropped)) else 0.0)
            keys = sorted({k for e in data
                           for k in (stats_of(e).get("dropped_keys") or [])})
            print(f"  dropped: mean={sum(dropped)/len(dropped)/1e6:.2f} MB "
                  f"({pct:.0f}% of the untrimmed bundle would have been "
                  f"uploaded; keys: {', '.join(keys) or 'n/a'})")

    # VidCom2 token compression summary (vidcom2 runs only): tokens
    # kept vs full + the compression stage's wall time.
    n_compressed = sum(1 for e in data
                       if (stats_of(e).get("vidcom2_active")
                           or e.get("compressor") == "vidcom2"))
    if n_compressed:
        full = [stats_of(e)["video_tokens_full"] for e in data
                if stats_of(e).get("video_tokens_full") is not None]
        kept = [stats_of(e)["video_tokens_kept"] for e in data
                if stats_of(e).get("video_tokens_kept") is not None]
        vt = [stats_of(e)["vidcom2_time"] for e in data
              if stats_of(e).get("vidcom2_time") is not None]
        ratios = [stats_of(e)["retention_ratio_actual"] for e in data
                  if stats_of(e).get("retention_ratio_actual") is not None]
        print(f"\n[EncPlBench] ===== vidcom2 ({n_compressed} entries) =====")
        if full and kept:
            # retention_ratio_actual is only written on the pruning path
            # (the no-prune degenerate path writes tokens full/kept only),
            # so the retention segment is appended per-need: the ternary
            # must NOT span the whole concatenation or an empty ratios
            # list would swallow the full/kept means too.
            line = (f"  video tokens: full mean={sum(full)/len(full):.0f} | "
                    f"kept mean={sum(kept)/len(kept):.0f}")
            if ratios:
                line += f" | actual retention mean={sum(ratios)/len(ratios):.3f}"
            print(line)
        if vt:
            print(f"  selector time: mean={sum(vt)/len(vt):.3f} s")

    # fp8 bundle quantization summary (vidcom2+fp8 runs only): uploaded
    # bundle size vs the feature's bf16-equivalent bytes + quant time.
    n_quant = sum(1 for e in data
                  if stats_of(e).get("quant_dtype"))
    if n_quant:
        qscales = [float(stats_of(e)["feature_scale"]) for e in data
                   if stats_of(e).get("feature_scale") is not None]
        qtimes = [stats_of(e)["quant_time"] for e in data
                  if stats_of(e).get("quant_time") is not None]
        print(f"\n[EncPlBench] ===== bundle fp8 ({n_quant} entries) =====")
        if qscales:
            print(f"  feature_scale: mean={sum(qscales)/len(qscales):.4f} | "
                  f"min={min(qscales):.4f} | max={max(qscales):.4f}")
        if qtimes:
            print(f"  quant time: mean={sum(qtimes)/len(qtimes):.3f} s")
        # feature bytes as uploaded (fp8, 1 byte/elem) vs hypothetical
        # bf16 (2 bytes/elem) from the embedding shape.
        f8 = [s["embedding_shape"][0] * s["embedding_shape"][1]
              for e in data
              if isinstance((s := stats_of(e)).get("embedding_shape"),
                            list) and len(s["embedding_shape"]) >= 2
              and s.get("quant_dtype")]
        if f8:
            print(f"  feature bytes: fp8 mean={sum(f8)/len(f8)/1e6:.2f} MB | "
                  f"bf16-equiv mean={2*sum(f8)/len(f8)/1e6:.2f} MB "
                  f"(~2x uplink reduction on the feature tensor)")

    # Connection 5-tuples (client >= the conns fix): each question has
    # TWO TCP connections — upload (PUT /v1/bundles) and chat (SSE
    # POST /v1/chat/completions) — recorded in encpl_stats.conns. The
    # pcap of a run contains the SAME (dst_ip, dst_port) for every
    # question; src_port is the join key for packet-level analysis
    # (pcap flow <-> question). Old results files simply lack the key.
    n_conns = sum(1 for e in data if stats_of(e).get("conns"))
    if n_conns:
        dsts = set()
        n_up = n_chat = 0
        for e in data:
            c = stats_of(e).get("conns") or {}
            for leg in ("upload", "chat"):
                t = c.get(leg)
                if not t:
                    continue
                dsts.add((t.get("dst_ip"), t.get("dst_port")))
                if leg == "upload":
                    n_up += 1
                else:
                    n_chat += 1
        print(f"\n[EncPlBench] ===== connections (5-tuple, {n_conns} entries) "
              f"=====")
        for ip, port in sorted(dsts, key=lambda x: (x[0] or "", x[1] or 0)):
            print(f"  dst {ip}:{port} proto=tcp")
        print(f"  upload legs: {n_up} | chat legs: {n_chat} "
              f"(src_port per entry in encpl_stats.conns.*.src_port — "
              f"the pcap<->question join key)")
        # A few src_port examples so the pairing is visible without
        # opening the JSON; full per-entry data stays in the results file.
        shown = 0
        for e in data:
            c = stats_of(e).get("conns") or {}
            up_p = (c.get("upload") or {}).get("src_port")
            ch_p = (c.get("chat") or {}).get("src_port")
            if up_p or ch_p:
                print(f"    e.g. {e.get('video_id', '?')}: "
                      f"up:{up_p} chat:{ch_p}")
                shown += 1
                if shown >= 3:
                    break

    n_err = sum(1 for e in data if e.get("error"))
    if n_err:
        print(f"\n[EncPlBench] {n_err} entries with errors")


if __name__ == "__main__":
    main()
