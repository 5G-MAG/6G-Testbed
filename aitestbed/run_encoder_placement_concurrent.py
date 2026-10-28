#!/usr/bin/env python3
"""
run_encoder_placement_concurrent.py — standalone runner for the
encoder_placement_concurrent scenario (N barrier-aligned terminal
users per wave).

Why standalone: the orchestrator / configs/scenarios.yaml /
clients/__init__.py / scenarios/__init__.py are actively used by the
serial experiment; this runner wires the new concurrent scenario with
ZERO changes to any existing file.

All knobs are CLI flags (no environment variables). Precedence:
  CLI flag  >  yaml key  >  built-in default
Every flag below maps onto the scenario's config key of the same
name, so a custom yaml + selective flag overrides compose cleanly.

Server pairing (REQUIRED — launch via the bootstrap script, it loads
the sglang host patches the edge mode needs):
  bash start_sidecar_concurrent.sh --port 8012

  The original serial sidecar also works as a queueing-attribution
  baseline (concurrent users serialize on its engine_lock):
  bash start_sidecar.sh --port 8002

Examples:
  # baseline arm: server mode, 4 concurrent users
  python aitestbed/run_encoder_placement_concurrent.py \\
      --mode server --users 4 \\
      --base-url http://<server-host>:8012 \\
      --runs 3 --db logs/encpl_conc_server_n4.db

  # comparison arm: edge mode + vidcom2 + fp8 (network-levelled)
  python aitestbed/run_encoder_placement_concurrent.py \\
      --mode edge --users 4 \\
      --compressor vidcom2 --retention-ratio 0.25 --bundle-quant fp8 \\
      --base-url http://<server-host>:8012 \\
      --runs 3 --db logs/encpl_conc_edge_n4.db

  # edge arm with vision-tower-only load + fa4 ViT backend (separate
  # db: backends are not bit-identical, don't pool them)
  python aitestbed/run_encoder_placement_concurrent.py \\
      --mode edge --users 4 --vision-only --attn-impl flash_attention_4 \\
      --base-url http://<server-host>:8012 \\
      --runs 3 --db logs/encpl_conc_edge_vo_fa4_n4.db

  # edge arm as N truly independent terminals (OS processes, own
  # ViT/GIL/connection each; model loads before the timing window)
  python aitestbed/run_encoder_placement_concurrent.py \\
      --mode edge --users 4 --enable-multi-processes --vision-only \\
      --base-url http://<server-host>:8012 \\
      --runs 3 --db logs/encpl_conc_edge_proc_n4.db

  # server arm as N independent upload terminals (OS processes, own
  # GIL/connection each; overlaps the raw-video uploads that serialize
  # on the shared GIL in thread mode)
  python aitestbed/run_encoder_placement_concurrent.py \\
      --mode server --users 4 --enable-multi-processes \\
      --base-url http://<server-host>:8012 \\
      --runs 3 --db logs/encpl_conc_server_proc_n4.db

  # per-user DISTINCT videos (either mode, threads or processes):
  # user i watches video i — strict N entries, miscount aborts
  python aitestbed/run_encoder_placement_concurrent.py \\
      --mode server --users 3 \\
      --video-paths sample_1/video.mp4 sample_2/video.mp4 sample_3/video.mp4 \\
      --base-url http://<server-host>:8012 \\
      --runs 3 --db logs/encpl_conc_server_n3_3vid.db

  # netem: --profile is APPLIED (topology-aware), not just recorded.
  #   single host (localhost base-url): lo:<port> filtered shaping,
  #     both directions (upload AND SSE token stream)
  #   dual host (remote base-url): whole-interface shaping on this
  #     machine's egress — WARNING: ssh etc. is impaired too; run
  #     inside tmux. Asymmetric (uplink:) profiles need dual host.
  #   --profile-record-only restores the old record-only behaviour
  #     (manual tc on the terminal machine).
  python aitestbed/run_encoder_placement_concurrent.py \\
      --mode server --users 4 --profile 5g_urban \\
      --base-url http://<server-host>:8012 \\
      --runs 3 --db logs/encpl_conc_server_n4_5g.db

  # pcap: L3/L4 tcpdump capture of the sidecar traffic, plus a
  # turn_index=-2 metadata row in the same --db (pcap path/size/
  # duration/filter; ran2_metrics/pcap_analyzer pick it up natively)
  python aitestbed/run_encoder_placement_concurrent.py \\
      --mode edge --users 4 --compressor vidcom2 --retention-ratio 0.25 \\
      --base-url http://<server-host>:8012 \\
      --runs 3 --db logs/encpl_conc_edge_n4.db \\
      --capture-pcap
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Make sibling packages importable both as `python
# aitestbed/run_encoder_placement_concurrent.py` (cwd = repo root) and
# from anywhere: resolve aitestbed/ from this file's location, and the
# repo root for the aicodec sibling (clients/__init__.py imports
# aicodec.codec via realtime_webrtc_vlm_client — same path setup as
# tools/encpl_bench_eval.py). netemu/src is added for the tc shaping
# (--profile) support, mirroring orchestrator.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
AITESTBED_ROOT = REPO_ROOT / "aitestbed"
for _p in (str(AITESTBED_ROOT), str(REPO_ROOT),
            str(REPO_ROOT / "aicodec"),
            str(REPO_ROOT / "netemu" / "src"),
            str(AITESTBED_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from clients.encoder_placement_client import EncoderPlacementClient  # noqa: E402
from analysis.logger import TrafficLogger, LogRecord  # noqa: E402
from capture.controller import CaptureController  # noqa: E402
from scenarios.base import ScenarioResult  # noqa: E402
from scenarios.encoder_placement_concurrent import (  # noqa: E402
    EncoderPlacementConcurrentScenario,
)
# Sidecar preflight/stats helpers shared with tools/encpl_bench_eval.py
# (module level there is stdlib+requests only — import-safe here).
from encpl_bench_eval import (  # noqa: E402
    _preflight_sidecar,
    _print_sidecar_stats,
    _print_client_decode_backend,
    _check_sidecar_edge_cpu_overlap,
)

_RUNNER_LOG_PREFIX = "[EncPlC-runner]"

DEFAULT_CONFIG = AITESTBED_ROOT / "configs" / "scenarios_encoder_placement_concurrent.yaml"
DEFAULT_PROFILES = AITESTBED_ROOT / "configs" / "profiles.yaml"

# Profile impairment fields (mirror of orchestrator._PROFILE_IMPAIRMENT_FIELDS;
# netemu's NetworkProfile dataclass strips `uplink:` blocks, so the raw YAML
# dict is re-read here to support asymmetric profiles in the dual-host case).
_PROFILE_IMPAIRMENT_FIELDS = frozenset({
    "delay_ms", "jitter_ms",
    "delay_distribution", "delay_correlation_pct",
    "loss_pct", "loss_correlation_pct", "loss_model",
    "rate_mbit", "rate_ceil_mbit",
    "rate_burst_kbit", "rate_cburst_kbit",
    "corruption_pct", "corruption_correlation_pct",
    "reorder_pct", "reorder_correlation_pct",
    "duplicate_pct", "duplicate_correlation_pct",
    "limit_packets",
})


def _load_scenario_config(config_path: Path) -> dict:
    # encoding="utf-8": the yaml has em-dashes; the Windows default
    # codec (gbk/cp936) cannot decode them and crashes at startup.
    # Same fix as tools/encpl_bench_eval_concurrent.py:165.
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    scenarios = raw.get("scenarios") or {}
    cfg = scenarios.get("encoder_placement_concurrent")
    if not cfg:
        raise SystemExit(
            f"scenario 'encoder_placement_concurrent' not found in {config_path}")
    cfg["scenario_id"] = "encoder_placement_concurrent"
    return cfg


# ---------------------------------------------------------------------------
# netem shaping (--profile). Logic mirrors tools/videomme_eval.py
# apply_netem()/clear_netem(): topology-aware, port-filtered on loopback
# for single-host runs, whole-interface egress shaping for dual-host.
# ---------------------------------------------------------------------------

def _is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "")


def _base_url_port(base_url: str, default: int = 8012) -> int:
    port = urlparse(base_url).port
    return port if port is not None else default


def _uplink_merged_profile(raw: dict) -> dict | None:
    """base ⊕ uplink shallow merge (egress view of an asymmetric profile).

    Returns None for symmetric profiles (no `uplink:` block).
    Mirrors orchestrator._apply_emulator_profile's asymmetric handling.
    """
    uplink = raw.get("uplink") if isinstance(raw, dict) else None
    if not uplink:
        return None
    base = {k: v for k, v in raw.items() if k in _PROFILE_IMPAIRMENT_FIELDS}
    return {**base, **{k: v for k, v in uplink.items()
                       if k in _PROFILE_IMPAIRMENT_FIELDS}}


def _apply_netem(base_url: str, profile_name: str, interface: str) -> None:
    """Apply the named profiles.yaml profile, topology-aware.

    Single host (base URL localhost/127.0.0.1):
        port-filtered shaping on loopback lo:<port from base URL>,
        BOTH directions (dport AND sport u32 filters — on lo both
        directions traverse the egress qdisc), via netemu's
        apply_profile_to_loopback. The upload (bundle PUT) and the SSE
        token stream are both impaired.
        Asymmetric profiles (with `uplink:`) are rejected: the loopback
        single-port filters cannot express per-direction parameters.

    Dual host (base URL points at a remote server):
        whole-interface shaping on this (terminal) machine's egress
        interface, entirely via native netemu calls. WARNING: the whole
        interface is shaped — other traffic (e.g. ssh) is impaired too;
        run long experiments inside tmux or via an out-of-band console.
        Asymmetric profiles apply egress = base ⊕ uplink (the upload,
        which is the dominant direction for both modes).

    Raises SystemExit on any failure: silently running on a bare network
    while the DB claims a profile would invalidate the experiment.
    """
    from netemu.emulator import NetworkEmulator

    profiles_path = DEFAULT_PROFILES
    single_host = _is_local_base_url(base_url)
    lo_port = _base_url_port(base_url)

    if single_host:
        emulator = NetworkEmulator(interface="lo",
                                   profiles_path=str(profiles_path))
    else:
        emulator = NetworkEmulator(interface=interface,
                                   profiles_path=str(profiles_path))

    # Raw profile dict (keeps `uplink:` blocks; netemu strips them).
    # encoding="utf-8" for the same Windows-gbk reason as
    # _load_scenario_config above.
    with open(profiles_path, encoding="utf-8") as f:
        raw_profiles = (yaml.safe_load(f) or {}).get("profiles", {})
    raw = raw_profiles.get(profile_name)
    is_asymmetric = isinstance(raw, dict) and bool(raw.get("uplink"))

    if profile_name == "no_emulation":
        if single_host:
            emulator.clear_loopback()
        else:
            emulator.clear()
        print(f"[EncPlC-runner] netem: no_emulation (cleared "
              f"{'lo' if single_host else emulator.interface})")
        return

    if profile_name not in emulator.profiles:
        raise SystemExit(f"unknown profile '{profile_name}' "
                         f"(not in {profiles_path})")

    if single_host:
        if is_asymmetric:
            raise SystemExit(
                f"profile '{profile_name}' has an `uplink:` block; asymmetric "
                f"profiles are only supported in dual-host mode (remote "
                f"--base-url), where egress and ingress are separate interfaces"
            )
        if not emulator.apply_profile_to_loopback(profile_name, lo_port):
            raise SystemExit(f"failed to apply netem profile '{profile_name}' "
                             f"to lo:{lo_port} (both directions)")
        print(f"[EncPlC-runner] netem: {profile_name} on lo:{lo_port} "
              f"(single host, both directions)")
        return

    # ---- Dual host: whole-interface shaping, native netemu only ----
    iface = emulator.interface
    egress_dict = _uplink_merged_profile(raw) if is_asymmetric else None

    if is_asymmetric:
        # egress = base ⊕ uplink (the upload direction), no ingress shaping
        egress_dict = egress_dict or {}
        ok = emulator.apply_settings(
            **egress_dict, disable_ingress=True,
            profile_name=f"{profile_name}:uplink")
        if not ok:
            raise SystemExit(f"failed to apply netem profile '{profile_name}' "
                             f"to {iface} (dual host, asymmetric egress)")
        print(f"[EncPlC-runner] netem: {profile_name} on {iface} "
              f"(dual host, asymmetric: egress=base+uplink)")
        return

    if not emulator.apply_profile(profile_name):
        raise SystemExit(f"failed to apply netem profile '{profile_name}' to "
                         f"{iface} (dual host, both directions, IFB ingress)")
    print(f"[EncPlC-runner] netem: {profile_name} on {iface} "
          f"(dual host, both directions, IFB ingress)")


def _clear_netem(base_url: str, interface: str) -> None:
    """Remove any netem rules (idempotent, safe on exit).

    Single host: loopback qdisc; dual host: interface + IFB teardown.
    """
    try:
        from netemu.emulator import NetworkEmulator
        if _is_local_base_url(base_url):
            NetworkEmulator(interface="lo").clear_loopback()
        else:
            NetworkEmulator(interface=interface).clear()
    except Exception as e:  # noqa: BLE001 — best-effort cleanup
        print(f"[EncPlC-runner] warning: netem cleanup failed: {e}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the encoder-placement CONCURRENT scenario "
                    "(standalone; no orchestrator changes; launch the "
                    "server via start_sidecar_concurrent.sh). Precedence: "
                    "CLI flag > yaml key > built-in default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ---- experiment arm --------------------------------------------------
    parser.add_argument("--mode", default=None, choices=["server", "edge"],
                        help="server: terminal uploads the original video, "
                             "server runs decode+processor+ViT+prefill. "
                             "edge: terminal runs decode+processor+ViT "
                             "locally and uploads the "
                             "precomputed_embedding bundle")
    parser.add_argument("--users", type=int, default=None, metavar="N",
                        help="concurrent users per wave (keep <= the "
                             "server's --max-bundles, default 64)")
    parser.add_argument("--compressor", default=None,
                        choices=["", "vidcom2"],
                        help="VidCom2 token compression of the edge bundle "
                             "(edge mode only; '' uploads all video tokens)")
    parser.add_argument("--retention-ratio", type=float, default=None,
                        help="fraction of video tokens kept by vidcom2 "
                             "(0, 1]; only meaningful with "
                             "--compressor vidcom2 in edge mode")
    parser.add_argument("--bundle-quant", default=None, choices=["", "fp8"],
                        help="fp8 (e4m3, per-tensor scale) quantization of "
                             "the uploaded edge-bundle feature tensor; "
                             "requires --compressor vidcom2 + edge mode")
    parser.add_argument("--vision-only", action="store_true",
                        help="load only the vision tower (~0.4B) instead of "
                             "the full model for the terminal-side ViT "
                             "(edge mode only; ~8GB -> ~1GB edge VRAM, "
                             "identical embeddings)")
    parser.add_argument("--attn-impl", default=None,
                        choices=["sdpa", "flash_attention_4", "eager"],
                        help="attention backend for the terminal-side ViT "
                             "(edge mode only; sdpa default. fa4/eager are "
                             "NOT bit-identical to sdpa — compare with a "
                             "tolerance, and use separate --db files per "
                             "backend). Aliases like 'fa4' are accepted via "
                             "the yaml key")
    # ---- model / video ---------------------------------------------------
    parser.add_argument("--model", default=None,
                        help="visual model path (terminal-side ViT in edge "
                             "mode)")
    parser.add_argument("--video-path", default=None,
                        help="input video (relative paths resolve against "
                             "--video-base-dir, else cwd)")
    parser.add_argument("--video-paths", default=None, nargs="+",
                        metavar="P",
                        help="per-user videos: user i uses video i, in BOTH "
                             "modes (threads and processes). STRICT: exactly "
                             "N entries (--users), miscount aborts the run. "
                             "Distinct files are the point; N copies of one "
                             "path is your explicit choice. Relative paths "
                             "resolve against --video-base-dir, else cwd. "
                             "Omitted -> --video-path shared by all users "
                             "(original behavior)")
    parser.add_argument("--video-base-dir", default=None,
                        help="base directory for relative --video-path")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="video frame cap passed to the pipeline "
                             "(native sglang key, caps BOTH sampling modes; "
                             "no patch needed)")
    parser.add_argument("--frame-ratio", type=float, default=None,
                        metavar="R",
                        help="percentage frame sampling: keep "
                             "floor(total_frames * R) frames (multiple of "
                             "2, min 4, capped by --max-frames) instead of "
                             "sglang's default 2fps x duration. In (0, 1]. "
                             "EDGE mode: sampled locally via the patched "
                             "smart_nframes. SERVER mode: the ENGINE samples "
                             "with its STARTUP-time config — start the "
                             "sidecar with the SAME --frame-ratio (the "
                             "sidecar warns on request mismatch)")
    parser.add_argument("--decode-threads", type=int, default=None,
                        metavar="N",
                        help="cap decord's INTERNAL decode thread pool "
                             "(SGLANG_DECORD_NUM_THREADS, read at "
                             "VideoReader construction). decord sizes its "
                             "pool by HOST core count regardless of CPU "
                             "affinity / OMP / torch knobs; under "
                             "concurrency N users = N such pools. EDGE "
                             "mode: applied per worker (process mode) or "
                             "in the parent (thread mode). SERVER mode: "
                             "ignored here — start the sidecar with the "
                             "SAME --decode-threads for the engine-side "
                             "cap. Omit for decord native default")
    parser.add_argument("--preprocess-threads", type=int, default=None,
                        metavar="N",
                        help="cap the terminal's torch/OMP intra-op thread "
                             "count for CPU-side preprocessing AFTER decode "
                             "(torchvision resize / HF processor): exports "
                             "OMP/MKL/OPENBLAS_NUM_THREADS + torch."
                             "set_num_threads in each edge worker. Without "
                             "it the worker sizes these pools by its "
                             "--edge-cpus-per-worker pin (e.g. 4). With a "
                             "pin, values ABOVE the pin's core count are "
                             "clamped to it; values BELOW simulate a "
                             "weaker terminal. No pin: value applies as-is. "
                             "EDGE mode only — server mode warns and "
                             "ignores (the ENGINE-side cap is the sidecar's "
                             "own --preprocess-threads flag; restart it "
                             "with the same value for symmetric "
                             "comparisons). Omit = derive from the pin "
                             "(current behaviour)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="generation token cap per request")
    # ---- dataset-bench mode ------------------------------------------------
    parser.add_argument("--dataset", default=None,
                        choices=["videomme", "mlvu"],
                        help="dataset-bench mode: iterate a QA test set "
                             "instead of the configured prompts — ONE "
                             "barrier-aligned wave of N playlists: the "
                             "barrier aligns only each user's FIRST "
                             "video, then every user runs its playlist "
                             "SERIALLY with no inter-video barrier (user "
                             "u gets question u::N round-robin; a failed "
                             "video is one ok=False entry and the "
                             "playlist continues). Per-VIDEO records "
                             "carry metadata.bench (question/answer/"
                             "model_answer/correct) and turn_index = "
                             "user*1000 + seq. Absent = the historical "
                             "shared-prompt waves")
    parser.add_argument("--dataset-dir", default=None, metavar="PATH",
                        help="dataset root: data/ + videos/ for videomme, "
                             "json/ + video/ for mlvu. Env "
                             "ENCPL_DATASET_DIR wins over this flag when "
                             "set (same precedence as the yaml key)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="cap on questions evaluated (0/absent = full "
                             "set); with --dataset only")
    parser.add_argument("--tasks", default=None, nargs="+", metavar="STEM",
                        help="MLVU task selection: absent = MCQ tasks 1-7; "
                             "'all' = every task (8/9 generation tasks are "
                             "recorded unscored); or a list of json stems "
                             "(e.g. 1_plotQA 8_sub_scene); with --dataset "
                             "mlvu only")
    parser.add_argument("--sample", default=None,
                        choices=["stratified", "head"],
                        help="sampling for --limit: stratified (default; "
                             "per-task/duration quota so a limit covers the "
                             "workload mix) or head (first N rows in file "
                             "order); with --dataset only")
    # ---- runner ----------------------------------------------------------
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="scenarios yaml supplying every key not given "
                             "as a flag")
    parser.add_argument("--db", default="logs/traffic_logs_encpl_concurrent.db",
                        help="output SQLite DB (use a different one per "
                             "mode/N combination for A/B comparison)")
    parser.add_argument("--runs", type=int, default=3,
                        help="number of runs (waves x prompts per run)")
    parser.add_argument("--profile", default="no_emulation",
                        help="network profile name from configs/profiles.yaml; "
                             "APPLIED topology-aware (single host: port-"
                             "filtered shaping on lo:<sidecar port>, both "
                             "directions; dual host: whole-interface egress "
                             "shaping — WARNING: ssh etc. is impaired too, "
                             "run inside tmux) AND recorded in the DB. "
                             "Asymmetric (uplink:) profiles need a remote "
                             "--base-url")
    parser.add_argument("--profile-record-only", action="store_true",
                        help="do NOT apply netem; only record --profile in "
                             "the DB (old behaviour — shape manually with tc "
                             "on the terminal machine's egress interface)")
    parser.add_argument("--netem-iface", default="auto",
                        help="interface for dual-host netem shaping "
                             "(ignored on single host, which always shapes "
                             "lo via port filters)")
    parser.add_argument("--base-url", default=None,
                        help="sidecar base URL override "
                             "(default: client default http://localhost:8012)")
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
    # ---- stream safety timeouts (forwarded verbatim to the client) ----
    # first-event default differs from serial-style callers: a barrier
    # wave of N users against a small running-request cap queues most
    # users behind minutes of prefill, so the tail user legitimately
    # waits before its first SSE event.
    parser.add_argument("--stream-connect-timeout", type=float, default=None,
                        metavar="SEC",
                        help="TCP connect timeout for sidecar HTTP requests "
                             "(default: client class default 20; 0 disables)")
    parser.add_argument("--stream-first-event-timeout", type=float,
                        default=900.0, metavar="SEC",
                        help="budget from chat POST to the first SSE event "
                             "(default 900 — barrier waves queue tail users "
                             "behind minutes of prefill; 0 disables)")
    parser.add_argument("--stream-idle-timeout", type=float, default=None,
                        metavar="SEC",
                        help="max silent gap between SSE events mid-stream "
                             "(default: client class default 20; 0 disables)")
    parser.add_argument("--stream-upload-read-timeout", type=float,
                        default=None, metavar="SEC",
                        help="read timeout for the bundle-upload PUT response "
                             "(default: client class default 1200; 0 disables)")
    parser.add_argument("--no-flush-cache", action="store_true",
                        help="skip the once-per-wave /flush_cache (measures "
                             "steady-state cache-hit behaviour instead of "
                             "cold prefixes)")
    parser.add_argument("--enable-multi-processes", action="store_true",
                        help="run each of the N users as an OS PROCESS "
                             "(own interpreter/GIL/connection; in edge mode "
                             "also own ViT) instead of threads sharing one "
                             "process — truly independent terminals. Works "
                             "in BOTH modes: edge = isolated ViT terminals "
                             "per user; server = independent upload paths "
                             "(thread uploads serialize on the shared GIL, "
                             "measured 1.9->3.2 s staircase for 4 users). "
                             "Spawn overhead lands before 'ready', outside "
                             "the timing window. DEFAULT since the "
                             "edge_processes=true flip — the flag remains "
                             "for explicitness/scripts. (former name: "
                             "--edge-processes)")
    parser.add_argument("--no-multi-processes", action="store_true",
                        help="force THREAD mode (N users share one process/"
                             "GIL) — overrides the edge_processes=true "
                             "default and the yaml key; the historical "
                             "pre-flip behaviour")
    parser.add_argument("--edge-cpus-per-worker", type=int, default=None,
                        metavar="K",
                        help="process mode only (either arm): pin each "
                             "worker to K cpus (round-robin from the "
                             "--edge-cpu-ids pool, default K=1, pool=parent's "
                             "affinity), modelling users on real K-core "
                             "terminals. 0 disables pinning. Thread mode "
                             "warns and skips")
    parser.add_argument("--edge-cpu-ids", default=None, metavar="SPEC",
                        help="cpu pool for --edge-cpus-per-worker, e.g. "
                             "\"0-7,32\" (default: every cpu this runner may "
                             "use). Only meaningful in process mode; "
                             "when the client terminals share a host with "
                             "the sidecar, point this at cpus OUTSIDE the "
                             "sidecar's --server-cpus range so the arms do "
                             "not contend")
    # ---- packet capture --------------------------------------------------
    parser.add_argument("--capture-pcap", action="store_true",
                        help="L3/L4 tcpdump capture of the sidecar traffic "
                             "(needs passwordless sudo tcpdump; the runner "
                             "EXITS if the capture cannot start). Writes a "
                             "turn_index=-2 metadata row into the same --db "
                             "so pcap_analyzer / ran2_metrics can join it")
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
    return parser


def _resolve_capture_interface(single_host: bool, base_url: str) -> str:
    """Interface for the pcap capture, never a bare 'auto'.

    'auto' is a netemu-side CONVENTION (it resolves it internally via
    ip route); tcpdump -i auto is NOT auto-detection and fails with
    "No such device" — the root cause of the phantom 0-byte captures
    (DB rows with interface='auto', response_bytes=0, no file anywhere
    on disk). Resolve it HERE instead: lo for single-host runs, else
    the default-route device; explicit --capture-interface wins by
    never reaching this function.
    """
    if single_host:
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
    raise SystemExit(
        "cannot resolve the capture interface for dual-host "
        f"(base_url={base_url}): 'ip route show default' failed — pass "
        "--capture-interface explicitly")


def _preflight_capture(interface: str, capture_dir: Path) -> None:
    """Fail FAST (SystemExit) when the capture cannot possibly work.

    Checks, in order: tcpdump installed, passwordless sudo, the
    interface exists. Silent degradation here is exactly how the
    phantom-capture incident happened: 50s runs 'succeeded' with
    metadata rows pointing at pcaps that never existed.
    """
    if subprocess.run(["which", "tcpdump"],
                      capture_output=True).returncode != 0:
        raise SystemExit(
            "--capture-pcap: tcpdump is not installed "
            "(apt install tcpdump)")

    probe = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if probe.returncode != 0:
        raise SystemExit(
            "--capture-pcap: passwordless sudo is required "
            "(sudo -n true failed). Add a NOPASSWD sudoers rule for "
            "tcpdump, e.g.: "
            "vivoadmin ALL=(root) NOPASSWD: /usr/bin/tcpdump")

    iface_list = subprocess.run(
        ["ip", "-o", "link", "show"], capture_output=True, text=True)
    if f"{interface}:" not in iface_list.stdout and \
            f"{interface}@ " not in iface_list.stdout:
        # /sys/class/net is authoritative and simplest
        import os as _os
        if interface not in _os.listdir("/sys/class/net"):
            raise SystemExit(
                f"--capture-pcap: interface '{interface}' does not exist "
                f"(available: {', '.join(sorted(_os.listdir('/sys/class/net')))})")

    capture_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _build_arg_parser().parse_args()

    cfg = _load_scenario_config(Path(args.config))

    # CLI > yaml: overlay every flag that was actually given.
    flag_to_key = {
        "mode": "mode",
        "users": "concurrent_users",
        "compressor": "compressor",
        "retention_ratio": "retention_ratio",
        "bundle_quant": "bundle_quant",
        "attn_impl": "attn_impl",
        "model": "model",
        "video_path": "video_path",
        "video_paths": "video_paths",
        "video_base_dir": "video_base_dir",
        "max_frames": "max_frames",
        "frame_ratio": "frame_ratio",
        "decode_threads": "decode_threads",
        "preprocess_threads": "preprocess_threads",
        "max_tokens": "max_tokens",
        "edge_cpus_per_worker": "edge_cpus_per_worker",
        "edge_cpu_ids": "edge_cpu_ids",
        # dataset-bench mode (yaml keys the scenario reads; env
        # ENCPL_DATASET_DIR still wins INSIDE the scenario, so a set
        # env overrides this flag at run time)
        "dataset": "dataset",
        "dataset_dir": "dataset_dir",
        "limit": "limit",
        "tasks": "tasks",
        "sample": "sample",
    }
    for flag, key in flag_to_key.items():
        value = getattr(args, flag)
        if value is not None:
            cfg[key] = value
    # --tasks normalization: nargs="+" always yields a list, but the
    # scenario's task spec accepts 'all'/'mcq' only as a STRING (a
    # list is treated as literal json stems — 'all' would then look
    # for json/all.json). Collapse single special values to str.
    if isinstance(cfg.get("tasks"), list) and len(cfg["tasks"]) == 1 \
            and str(cfg["tasks"][0]).strip().lower() in ("all", "mcq"):
        cfg["tasks"] = str(cfg["tasks"][0]).strip().lower()
    # store_true flags have no None default — overlay only when set
    # (same pattern as --no-flush-cache below).
    if args.vision_only:
        cfg["vision_only"] = True
    if args.enable_multi_processes:
        cfg["edge_processes"] = True
    if args.no_multi_processes:
        cfg["edge_processes"] = False
    if args.no_flush_cache:
        cfg["flush_cache"] = False

    # Sidecar preflight BEFORE anything with side effects: the client
    # (process-global caches), TrafficLogger (CREATES the SQLite file
    # + schema), netem shaping and the tcpdump spawn all come after, so
    # a failed preflight (sidecar down / model mismatch) leaves the box
    # and the --db exactly as they were.
    # _preflight_url mirrors the client's own fallback chain (flag >
    # env SGLANG_ENGINE_BASE_URL > client DEFAULT 8002) without
    # constructing the client first; rstrip matches its normalization.
    _preflight_url = (
        args.base_url
        or os.environ.get("SGLANG_ENGINE_BASE_URL")
        or "http://localhost:8002"
    ).rstrip("/")
    _preflight_sidecar(
        _preflight_url,
        cfg.get("model"),
        wait_sec=args.wait_sidecar,
        check_model=not args.no_model_check,
        log_prefix=_RUNNER_LOG_PREFIX,
    )
    # Environment context for the run log (decode backend, thread caps,
    # patch-4 state) — best-effort, immediately after preflight.
    _print_sidecar_stats(_preflight_url, log_prefix=_RUNNER_LOG_PREFIX)
    # CPU disjointness gate: in process mode with per-worker pinning
    # ON, the sidecar's --server-cpus and the terminal cpu pool are
    # the experiment's independent variables — an overlap is a config
    # contradiction, so exit before ANY side effect (netem, pcap, db).
    # Thread mode shares one process and pins nothing per-user — skip.
    if cfg.get("edge_processes") and int(
            cfg.get("edge_cpus_per_worker", 1) or 0) > 0:
        _check_sidecar_edge_cpu_overlap(
            _preflight_url,
            cfg.get("edge_cpu_ids"),
            log_prefix=_RUNNER_LOG_PREFIX,
            sidecar_hint="start_sidecar_concurrent.sh",
        )

    client = EncoderPlacementClient(
        base_url=args.base_url,
        stream_connect_timeout=args.stream_connect_timeout,
        stream_first_event_timeout=args.stream_first_event_timeout,
        stream_idle_timeout=args.stream_idle_timeout,
        stream_upload_read_timeout=args.stream_upload_read_timeout)
    logger = TrafficLogger(args.db)
    scenario = EncoderPlacementConcurrentScenario(client, logger, cfg)

    # Topology from the EFFECTIVE base URL (flag > env > client default).
    effective_base_url = client.base_url
    single_host = _is_local_base_url(effective_base_url)

    # ---- netem shaping (unless record-only) -----------------------------
    if not args.profile_record_only:
        _apply_netem(effective_base_url, args.profile, args.netem_iface)

    # ---- pcap capture ----------------------------------------------------
    pcap_controller = None
    pcap_start_time = None
    pcap_file = None
    run_id = f"run_{int(time.time())}_{abs(hash(repr(vars(args)))) % 100000:05d}"
    if args.capture_pcap:
        # Resolve 'auto' to a REAL interface BEFORE handing it to
        # tcpdump (see _resolve_capture_interface) and fail fast when
        # the capture cannot work — the old behaviour (spawn, print an
        # optimistic path, discover at stop() that 0 bytes exist) cost
        # a full 50s experiment run with no pcap.
        capture_iface = args.capture_interface or _resolve_capture_interface(
            single_host, effective_base_url)
        _preflight_capture(capture_iface, Path(args.capture_dir))
        sidecar_port = _base_url_port(effective_base_url)
        capture_filter = args.capture_filter or f"tcp port {sidecar_port}"
        try:
            pcap_controller = CaptureController(
                interface=capture_iface,
                capture_dir=args.capture_dir,
            )
            ts = time.strftime("%Y%m%d_%H%M%S")
            arm = (f"{cfg.get('mode', 'server')}"
                   f"_n{cfg.get('concurrent_users', 2)}")
            pcap_start_time = time.time()
            pcap_file = pcap_controller.start(
                filename=f"encpl_conc_{arm}_{args.profile}_{ts}.pcap",
                filter_expr=capture_filter,
            )
            if not pcap_file:
                # start() now also catches spawn-then-instant-death
                # (bad filter, sudo flipped to prompting mid-run, ...);
                # a requested capture that cannot start is a hard error.
                raise SystemExit(
                    "--capture-pcap: tcpdump failed to start on "
                    f"'{capture_iface}' (see the logged stderr tail above; "
                    "check the interface name and BPF filter)")
            print(f"[EncPlC-runner] pcap: {pcap_file} "
                  f"(iface={capture_iface} filter='{capture_filter}')")
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"--capture-pcap: failed to start tcpdump "
                             f"capture ({e})")

    # Edge THREAD mode decodes in THIS process: name the local decode
    # backend next to the sidecar /stats line (engine-side). Process
    # mode is covered per worker — each worker prints its own backend
    # after applying decode_threads (inherited env could differ).
    if cfg.get("mode", "server") == "edge" and not cfg.get("edge_processes"):
        _print_client_decode_backend(log_prefix=_RUNNER_LOG_PREFIX)

    print(f"[EncPlC-runner] scenario=encoder_placement_concurrent "
          f"mode={cfg.get('mode', 'server')} "
          f"users={cfg.get('concurrent_users', 2)} "
          + (f"videos={len(cfg['video_paths'])} "
             if cfg.get("video_paths") else "")
          + (f"frame_ratio={cfg.get('frame_ratio')} "
             if cfg.get("frame_ratio") else "")
          + (f"decode_threads={cfg.get('decode_threads')} "
             if cfg.get("decode_threads") else "")
          + (f"preprocess_threads={cfg.get('preprocess_threads')} "
             if cfg.get("preprocess_threads") else "")
          + (f"multi_processes={cfg.get('concurrent_users', 2)}p "
             if cfg.get("edge_processes") else "")
          + (f"edge_cpus={cfg.get('edge_cpus_per_worker', 1)}c/worker"
             if cfg.get("edge_processes") else "")
          + f"model={cfg.get('model')} "
          f"profile={args.profile}"
          + (" (record-only)" if args.profile_record_only else "")
          + f" runs={args.runs} db={Path(args.db).resolve()}")

    n_ok = 0
    results = []
    try:
        for run_index in range(args.runs):
            t0 = time.time()
            try:
                result = scenario.run(network_profile=args.profile,
                                      run_index=run_index)
            except Exception as exc:  # harness must survive scenario bugs
                # TrafficLogger has no .exception() (only log/log_batch) —
                # print the traceback directly so the crash context is not
                # lost, and keep the run summary honest via the synthetic
                # failure ScenarioResult below.
                import traceback
                traceback.print_exc()
                result = ScenarioResult(
                    scenario_id="encoder_placement_concurrent",
                    session_id=f"crash-{run_index}",
                    network_profile=args.profile,
                    run_index=run_index,
                    success=False,
                    error_message=f"scenario.run crashed: {exc}",
                )
            wall = time.time() - t0
            if result.success:
                n_ok += 1
            results.append(result)
            print(f"[EncPlC-runner] run {run_index + 1}/{args.runs}: "
                  f"turns={result.turn_count} calls={result.api_call_count} "
                  f"ttft={result.ttft_sec and round(result.ttft_sec, 3)} s "
                  f"wall={wall:.1f}s"
                  + (f" error={result.error_message}"
                     if result.error_message else ""))
            _print_per_user_summary(run_index, result)
    finally:
        # netem teardown FIRST: stop impairing the box even if pcap
        # logging below fails; clear is idempotent.
        if not args.profile_record_only:
            _clear_netem(effective_base_url, args.netem_iface)

        if pcap_controller is not None:
            try:
                pcap_stop_time = time.time()
                pcap_path = pcap_controller.stop()
                if pcap_path:
                    capture_duration = None
                    if pcap_start_time is not None:
                        capture_duration = pcap_stop_time - pcap_start_time
                    pcap_size = 0
                    try:
                        pcap_size = pcap_path.stat().st_size
                    except Exception:
                        pass
                    capture_metadata = {
                        "type": "pcap_capture",
                        "pcap_file": str(pcap_path),
                        "capture_dir": args.capture_dir,
                        "capture_filter": (
                            args.capture_filter
                            or f"tcp port {_base_url_port(effective_base_url)}"
                        ),
                        "capture_duration_sec": capture_duration,
                        "capture_stats": pcap_controller.get_capture_stats(
                            pcap_path),
                        # the RESOLVED interface (never 'auto'); the old
                        # rows with interface='auto' + response_bytes=0
                        # are exactly the phantom captures
                        "interface": capture_iface,
                        "run_id": run_id,
                    }
                    capture_record = LogRecord(
                        timestamp=time.time(),
                        scenario_id="encoder_placement_concurrent",
                        session_id=f"pcap_{int(pcap_start_time or time.time())}",
                        turn_index=-2,
                        run_index=-1,
                        provider="tcpdump",
                        model="",
                        request_bytes=0,
                        response_bytes=pcap_size,
                        t_request_start=pcap_start_time or 0.0,
                        latency_sec=capture_duration or 0.0,
                        network_profile=args.profile,
                        http_status=200,
                        metadata=json.dumps(capture_metadata),
                        # success honestly reflects whether the pcap
                        # exists with packets: a 0-size file is a failed
                        # capture even though the run completed
                        success=pcap_size > 0,
                    )
                    logger.log(capture_record)
                    print(f"[EncPlC-runner] pcap stopped: {pcap_path} "
                          f"({pcap_size} bytes, metadata row in "
                          f"{Path(args.db).resolve()})"
                          + ("" if pcap_size else
                             " WARNING: 0 bytes captured — no packets "
                             "matched the filter/interface"))
            except Exception as e:
                print(f"[EncPlC-runner] warning: failed to stop/log tcpdump "
                      f"capture cleanly ({e})")

    _print_final_summary(results)

    print(f"[EncPlC-runner] done: {n_ok}/{args.runs} runs succeeded; "
          f"records in {args.db}")


def _print_per_user_summary(run_index: int, result) -> None:
    """Per-wave x per-user TTFT/latency table for one run.

    Reads result.metadata['waves'] (filled by the scenario):
      [{wave_index, wave_size, ok_per_user: [...], ttft_per_user: [...],
        latency_per_user: [...], wave_empty, ttft_mean/p50/p90/max,
        wave_wall_sec, wave_ok, ...}, ...]

    Cell semantics: ttft/latency pair per user —
      "82.31/88.19"  ok, normal numbers;
      "EMPTY/-"      ok=True but the engine produced ZERO content tokens
                     (greedy first-token EOS / retract; sidecar logged a
                     zero-token-completion warning with finish_reason);
      "FAIL/-"       ok=False (real request failure — error in the DB).

    Playlist mode (bench dataset): ttft/latency/ok/video arrive as a
    PER-VIDEO list per user (seq order) — rendered as one indented line
    per video under the user, upload is the whole-playlist total, and
    the shown conns are the FIRST video's (barrier-aligned start;
    later videos open fresh connections — full detail in the DB rows).

    Layout: one user PER LINE with its upload size/time, FULL 5-tuples
    (src_ip:src_port -> dst_ip:dst_port for the upload PUT and the chat
    SSE legs) and FULL video path on indented follow-up lines
    (heterogeneous waves: which video caused which TTFT is readable at
    a glance); the wave header carries only the aggregate p50/p90/max.
    src_port is the pcap<->user join key (all flows share dst ip/port).
    Fields absent in wave metadata (older scenario / mock clients)
    degrade silently — lines are omitted, never crash the printer.
    """
    waves = (result.metadata or {}).get("waves") or []
    if not waves:
        return
    print(f"[EncPlC-runner] per-user summary (run {run_index}):")
    for w in waves:
        size = w.get("wave_size") or 0
        ttfts = w.get("ttft_per_user") or [None] * size
        lats = w.get("latency_per_user") or [None] * size
        oks = w.get("ok_per_user") or [False] * size
        conns = w.get("conns_per_user") or [None] * size
        vids = w.get("video_paths_per_user") or [None] * size
        up_bytes = w.get("upload_bytes_per_user") or [None] * size
        up_times = w.get("upload_time_per_user") or [None] * size
        p50 = w.get("ttft_p50")
        p90 = w.get("ttft_p90")
        mx = w.get("ttft_max")
        empty = w.get("wave_empty")
        empty_s = f" empty={empty}" if empty else ""
        # wave header: the aggregate stats only; every user's detail
        # moves to its own line below (full video path per user).
        # Playlist mode: ok counts VIDEOS (wave_videos), not users.
        n_videos = w.get("wave_videos")
        denom = n_videos if n_videos is not None else size
        print(f"  wave {w.get('wave_index')} (ok={w.get('wave_ok')}/"
              f"{denom}{empty_s}) || ttft p50={p50 and round(p50, 2)} "
              f"p90={p90 and round(p90, 2)} "
              f"max={mx and round(mx, 2)} s")
        for u in range(size):
            t = ttfts[u] if u < len(ttfts) else None
            l = lats[u] if u < len(lats) else None
            ok_u = oks[u] if u < len(oks) else False
            v = vids[u] if u < len(vids) else None
            if isinstance(t, list):
                # playlist mode: per-video lines under the user
                l_list = l if isinstance(l, list) else [None] * len(t)
                ok_list = (ok_u if isinstance(ok_u, list)
                           else [False] * len(t))
                v_list = v if isinstance(v, list) else [None] * len(t)
                n_ok = sum(1 for o in ok_list if o)
                print(f"    u{u}: {n_ok}/{len(t)} videos ok")
                for s in range(len(t)):
                    tv = t[s] if s < len(t) else None
                    lv = l_list[s] if s < len(l_list) else None
                    ok_v = ok_list[s] if s < len(ok_list) else False
                    if not ok_v:
                        t_s = "FAIL"
                    elif tv is None:
                        t_s = "EMPTY"
                    else:
                        t_s = f"{tv:.2f}"
                    l_s = f"{lv:.2f}" if lv is not None else "-"
                    print(f"      v{s}: ttft={t_s} lat={l_s} s")
                    vv = v_list[s] if s < len(v_list) else None
                    if vv:
                        print(f"        video: {vv}")
            else:
                if not ok_u:
                    t_s = "FAIL"
                elif t is None:
                    t_s = "EMPTY"
                else:
                    t_s = f"{t:.2f}"
                l_s = f"{l:.2f}" if l is not None else "-"
                print(f"    u{u}: ttft={t_s} lat={l_s} s")
                if v:
                    print(f"      video: {v}")
            ub = up_bytes[u] if u < len(up_bytes) else None
            ut = up_times[u] if u < len(up_times) else None
            if ub is not None or ut is not None:
                ub_s = (f"{ub / (1024 * 1024):.2f} MiB ({ub} B)"
                        if ub is not None else "-")
                ut_s = f"{ut:.2f}" if ut is not None else "-"
                print(f"      upload: {ub_s} in {ut_s} s")
            c = conns[u] if u < len(conns) else None
            if c:
                up = c.get("upload") or {}
                ch = c.get("chat") or {}
                if up:
                    print(f"      up:  {up.get('src_ip')}:"
                          f"{up.get('src_port')} -> "
                          f"{up.get('dst_ip')}:{up.get('dst_port')}")
                if ch:
                    print(f"      chat:{ch.get('src_ip')}:"
                          f"{ch.get('src_port')} -> "
                          f"{ch.get('dst_ip')}:{ch.get('dst_port')}")


def _print_final_summary(results: list) -> None:
    """Cross-run TTFT stats over every wave of every successful run."""
    all_p50 = []
    all_p90 = []
    all_max = []
    total_empty = 0
    for r in results:
        for w in (r.metadata or {}).get("waves") or []:
            if w.get("ttft_p50") is not None:
                all_p50.append(w["ttft_p50"])
            if w.get("ttft_p90") is not None:
                all_p90.append(w["ttft_p90"])
            if w.get("ttft_max") is not None:
                all_max.append(w["ttft_max"])
            total_empty += w.get("wave_empty") or 0
    if not all_p50:
        return

    def _pct(vals, p):
        vals = sorted(vals)
        if len(vals) == 1:
            return vals[0]
        # Linear-interpolated percentile (same semantics as
        # analysis/metrics.py _percentile); the old round-to-index
        # version was order-statistics-only (n=2 p50 = minimum).
        k = (len(vals) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(vals) - 1)
        if f == c:
            return vals[f]
        return vals[f] + (vals[c] - vals[f]) * (k - f)

    print(f"[EncPlC-runner] overall TTFT across {len(all_p50)} waves: "
          f"p50={_pct(all_p50, 50):.2f} p90={_pct(all_p90, 50):.2f} "
          f"max={max(all_max):.2f} s"
          + (f" (empty={total_empty})" if total_empty else ""))


if __name__ == "__main__":
    main()
