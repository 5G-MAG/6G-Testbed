#!/usr/bin/env python3
"""One-click WHOLE-image UDP pipeline for One-D-Piece over MMBench.

Runs the whole lossy-link experiment end to end on the Linux testbed box:

    1. start the ODP image UDP server (cloud receiver + decoder) and wait until
       it has bound the port and warmed the decoder;
    2. run the orchestrator scenario ``odp_image_udp_whole`` (1 datagram/image,
       no retransmission) under a chosen network profile;
    3. when the client finishes it sends CTL_SESSION_COMPLETE (x3), the server
       flushes reconstructions + the eval manifest and saves its report;
    4. stop the server, verify the recon folder actually contains images and a
       question ``manifest.jsonl``;
    5. score the reconstructed folder with Qwen3-VL, launched in a SEPARATE
       conda env (``conda run -n <env>``) so vLLM/transformers never touch the
       6gtestbed environment;
    6. print / dump a small combined summary (link stats + VQA accuracy).

Everything is resolved relative to the repo root, so the command is
machine-independent (no /data01 hard-coding)::

    python run_odp_mmbench_pipeline.py \
        --profiles lossy --interface lo --max-images 0 \
        --eval-conda-env qwen3vl --eval-gpu 1

To study source-side (payload) token drop in isolation from real netem link
loss, reuse the "no_emulation" profile -- it already carries zero netem
impairment by definition (configs/profiles.yaml) -- together with the
whole-mode-only drop knobs (see scenarios/odp_image_udp.py for the gating
rationale). Defaults for the drop knobs are "none"/0, so a plain
no_emulation run stays a clean baseline unless you opt in explicitly::

    python run_odp_mmbench_pipeline.py \
        --mode whole --profiles no_emulation \
        --payload-drop-mode tail --payload-drop-pct 30 \
        --interface lo --max-images 0 --eval-conda-env qwen3vl --eval-gpu 1

Notes / design choices
----------------------
* The server is spawned ONLY for --server-host 127.0.0.1 (loopback). Point
  --server-host at a remote cloud box and this script assumes that server is
  managed out of band.
* One server instance per network profile, each with its OWN --fresh recon
  directory, so different loss conditions never overwrite each other.
* ODP encode/decode and the Qwen VLM run in different phases, so they do not
  contend for GPU memory; --odp-device and --eval-gpu are configurable.
* The ODP model package (aicodec/aicodec/models/one_d_piece) is fully
  self-contained: bridge.py, checkpoints/, the vendored architecture code
  (vendor/modeling/) and the config (odp_l256.yaml) all live together, so no
  ODP_REPO/ODP_CONFIG/ODP_CHECKPOINT env overrides are needed for a normal
  run; set them yourself only if you want to point at an external checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

# run_odp_mmbench_pipeline.py lives at the repo root.
REPO_ROOT = Path(__file__).resolve().parent
AITESTBED_DIR = REPO_ROOT / "aitestbed"

DEFAULT_CONFIG = AITESTBED_DIR / "configs" / "scenarios.yaml"  # odp_image_udp_* merged in
DEFAULT_SCENARIO = "odp_image_udp_whole"
DEFAULT_SERVER = AITESTBED_DIR / "server" / "odp_image_udp_server.py"
DEFAULT_EVAL_SCRIPT = "aitestbed/tools/mmbench_eval.py"
DEFAULT_QWEN_MODEL = "aicodec/aicodec/models/Qwen3_VL_30B_A3B_Instruct"
DEFAULT_RESULTS = REPO_ROOT / "results" / "odp_mmbench_pipeline"

SERVER_READY_TOKEN = "ODP image UDP server on"
SERVER_COMPLETE_TOKEN = "session-complete control received"


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)

def wait_for_server_ready(proc: subprocess.Popen, log_files: list[Path],
                          timeout: float) -> None:
    handles = [p.open("a", encoding="utf-8") for p in log_files]
    deadline = time.time() + timeout
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            print(f"[server] {line}", flush=True)
            for h in handles:
                h.write(line + "\n")
                h.flush()
            if SERVER_READY_TOKEN in line:
                log("server is listening and decoder is warm")
                return
            if time.time() > deadline:
                raise TimeoutError("server did not become ready in time")
            if proc.poll() is not None:
                raise RuntimeError("server exited before becoming ready")
    finally:
        for h in handles:
            h.close()
    raise RuntimeError("server stdout closed before ready token")


def drain_server_background(proc: subprocess.Popen, log_files: list[Path]) -> None:
    """Print/tee remaining server output without blocking (best effort)."""
    """非阻塞读取记录子进程服务器的输出日志"""
    import threading

    def _pump():
        handles = [p.open("a", encoding="utf-8") for p in log_files]
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                print(f"[server] {line}", flush=True)
                for h in handles:
                    h.write(line + "\n")
                    h.flush()
        finally:
            for h in handles:
                h.close()

    threading.Thread(target=_pump, daemon=True).start()


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    log("stopping server (SIGINT -> flush + save_report)")
    try:
        if os.name == "posix":
            proc.send_signal(signal.SIGINT)
        else:
            proc.terminate()
        proc.wait(timeout=120)
    except Exception:
        proc.kill()


def verify_recon(recon_dir: Path) -> tuple[int, int]:
    pngs = list(recon_dir.glob("*.png"))
    manifest = recon_dir / "manifest.jsonl"
    n_manifest = 0
    if manifest.exists():
        n_manifest = sum(1 for ln in manifest.read_text(
            encoding="utf-8").splitlines() if ln.strip())
    log(f"recon check: {len(pngs)} png, {n_manifest} manifest rows in "
        f"{recon_dir}")
    if not pngs:
        raise RuntimeError(f"no reconstructed png in {recon_dir}")
    if n_manifest == 0:
        raise RuntimeError(f"no eval manifest rows in {recon_dir}")
    return len(pngs), n_manifest


def build_run_config(base_config: Path, scenario: str, max_images: int | None,
                     tmp_dir: Path, run_timeout: float,
                     payload_drop_mode: str | None = None,
                     payload_drop_pct: float | None = None) -> Path:
    """Materialize a temp config so we never mutate the tracked yaml.

    Injects ``max_images`` into the target scenario (a CLI override is not
    supported by orchestrator.py) and relaxes the per-run timeout for full
    dataset runs. ``payload_drop_mode``/``payload_drop_pct`` (whole mode
    only; see scenarios/odp_image_udp.py) are likewise injected here so the
    CLI can pick a drop ratio without hand-editing scenarios.yaml -- they
    still only take effect when the run's --profiles includes
    "no_emulation". Everything else is inherited unchanged.
    """
    with base_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sc = (cfg.get("scenarios") or {}).get(scenario)
    if sc is None:
        raise ValueError(f"scenario {scenario!r} not in {base_config}")
    if max_images is not None:
        sc["max_images"] = int(max_images)
    if payload_drop_mode is not None:
        if sc.get("mode", "chunked") != "whole":
            raise ValueError(
                "--payload-drop-mode/--payload-drop-pct only apply to the "
                f"whole-mode scenario, but {scenario!r} has "
                f"mode={sc.get('mode', 'chunked')!r}")
        sc["payload_drop_mode"] = payload_drop_mode
    if payload_drop_pct is not None:
        sc["payload_drop_pct"] = float(payload_drop_pct)
    defaults = cfg.setdefault("defaults", {})
    defaults["timeout_sec"] = int(run_timeout)
    out = tmp_dir / f"config_{scenario}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def run_one_profile(args, profile: str) -> dict:
    tag = profile.replace(" ", "_").replace("/", "_")
    profile_dir = args.out_dir / tag
    recon_dir = profile_dir / "recon"
    report_dir = profile_dir / "reports"
    eval_dir = profile_dir / "eval"
    logs_dir = profile_dir / "logs"
    for d in (recon_dir, report_dir, eval_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    server_log = logs_dir / "server.log"
    orch_log = logs_dir / "orchestrator.log"

    summary: dict = {"profile": profile, "recon_dir": str(recon_dir)}

    # ---- phase 1: server -------------------------------------------------
    server_cmd = [
        sys.executable, str(DEFAULT_SERVER),
        "--host", "0.0.0.0",
        "--port", str(args.server_port),
        "--fresh",
        "--network-profile", tag,
        "--dump-recon-dir", str(recon_dir),
        "--report-dir", str(report_dir),
    ]
    server_env = os.environ.copy()
    # Inherits CUDA_VISIBLE_DEVICES from this process, so exporting it before
    # launching the pipeline pins BOTH the server and the orchestrator client.
    if args.odp_device:
        server_env["ODP_DEVICE"] = args.odp_device
    log(f"starting server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(
        server_cmd, cwd=str(AITESTBED_DIR), env=server_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True)
    try:
        wait_for_server_ready(server_proc, [server_log],
                              timeout=args.server_ready_timeout)
        drain_server_background(server_proc, [server_log])

        # ---- phase 2: orchestrator (edge sender) ------------------------
        # max_images has no CLI flag, so materialize a temp config override.
        with tempfile.TemporaryDirectory(prefix="odp_pipeline_") as td:
            if (args.payload_drop_mode is not None
                    or args.payload_drop_pct is not None) and profile != "no_emulation":
                log(f"WARNING: --payload-drop-mode/--payload-drop-pct only "
                    f"take effect under profile 'no_emulation' (see "
                    f"scenarios/odp_image_udp.py); profile {profile!r} will "
                    f"ignore them.")
            run_config = build_run_config(
                Path(args.config), args.scenario, args.max_images,
                Path(td), args.run_timeout,
                payload_drop_mode=args.payload_drop_mode,
                payload_drop_pct=args.payload_drop_pct)
            orch_cmd = [
                sys.executable, "orchestrator.py",
                "--config", str(run_config),
                "--scenario", args.scenario,
                "--profile", profile,
                "--runs", str(args.runs),
                "--interface", args.interface,
                "--run-timeout", str(int(args.run_timeout)),
            ]
            log(f"running orchestrator: {' '.join(orch_cmd)}")
            # The client encoder runs INSIDE the orchestrator process; pin it to
            # the same device as the decoder so a busy GPU0 is never touched.
            orch_env = os.environ.copy()
            if args.odp_device:
                orch_env["ODP_DEVICE"] = args.odp_device
            with orch_log.open("w", encoding="utf-8") as lf:
                rc = subprocess.call(orch_cmd, cwd=str(AITESTBED_DIR),
                                     env=orch_env,
                                     stdout=lf, stderr=subprocess.STDOUT)
        if rc != 0:
            raise RuntimeError(
                f"orchestrator failed rc={rc}; see {orch_log}")
        log("orchestrator finished (client sent SESSION_COMPLETE)")
        # Give the server a moment to finish async recon + report flush.
        time.sleep(args.post_send_settle)
    finally:
        stop_server(server_proc)

    # ---- phase 3: verify --------------------------------------------------
    n_png, n_rows = verify_recon(recon_dir)
    summary["recon_png"] = n_png
    summary["recon_manifest_rows"] = n_rows

    # Attach the server link report (loss / token recovery).
    link_reports = sorted(report_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if link_reports:
        try:
            summary["link_report"] = json.loads(
                link_reports[-1].read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log(f"could not parse link report {link_reports[-1]}: {e}")

    # ---- phase 4: Qwen VLM eval in an isolated conda env -----------------
    if not args.skip_eval:
        eval_model = (Path(args.eval_model).expanduser()
                      if args.eval_model else REPO_ROOT / DEFAULT_QWEN_MODEL)
        eval_cmd = [
            "conda", "run", "--no-capture-output", "-n", args.eval_conda_env,
            "python", str(REPO_ROOT / DEFAULT_EVAL_SCRIPT),
            "--data-dir", str(recon_dir),
            "--output-dir", str(eval_dir),
            "--model", str(eval_model),
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        ]
        if args.eval_limit:
            eval_cmd += ["--limit", str(args.eval_limit)]
        eval_env = os.environ.copy()
        if args.eval_gpu:
            eval_env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu)
        gpu_label = eval_env.get("CUDA_VISIBLE_DEVICES", "<unset>")
        log(f"scoring with Qwen3-VL in conda env '{args.eval_conda_env}' "
            f"(CUDA_VISIBLE_DEVICES={gpu_label})")
        eval_log = logs_dir / "eval.log"
        with eval_log.open("w", encoding="utf-8") as lf:
            rc = subprocess.call(eval_cmd, cwd=str(AITESTBED_DIR),
                                 env=eval_env, stdout=lf, stderr=subprocess.STDOUT)
        if rc != 0:
            raise RuntimeError(f"Qwen eval failed rc={rc}; see {eval_log}")
        sj = eval_dir / "summary.json"
        if sj.exists():
            summary["eval"] = json.loads(sj.read_text(encoding="utf-8"))
            acc = summary["eval"].get("raw_accuracy")
            log(f"profile {tag}: raw_accuracy={acc}")
    else:
        log("--skip-eval: skipping Qwen scoring")

    (profile_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--scenario", default=DEFAULT_SCENARIO)
    p.add_argument("--mode", choices=["whole", "chunked"], default=None,
                   help="convenience alias for --scenario: maps to "
                        "odp_image_udp_{mode}; overrides --scenario if both "
                        "are given")
    p.add_argument("--profiles", nargs="+", default=["lossy"],
                   help="network profiles; one isolated run per profile")
    p.add_argument("--interface", default="lo")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--run-timeout", type=float, default=3600.0,
                   help="per-run timeout sec injected into the temp config "
                        "and --run-timeout (full dataset needs >180s default)")
    p.add_argument("--max-images", type=int, default=None,
                   help="cap images/run (scenario max_images); default = yaml")
    p.add_argument("--payload-drop-mode", choices=["none", "tail", "random"],
                   default=None,
                   help="whole-mode-only synthetic token drop (see "
                        "scenarios/odp_image_udp.py); ONLY takes effect "
                        "when a profile in --profiles is 'no_emulation'. "
                        "Requires --scenario/--mode=whole")
    p.add_argument("--payload-drop-pct", type=float, default=None,
                   help="0..100, percent of tokens dropped under "
                        "--payload-drop-mode; same no_emulation-only gating")
    p.add_argument("--server-port", type=int, default=1237)
    p.add_argument("--odp-device", default="",
                   help="optional explicit ODP device, e.g. cuda:0 (env "
                        "ODP_DEVICE); normally just set CUDA_VISIBLE_DEVICES")
    p.add_argument("--server-ready-timeout", type=float, default=600.0,
                   help="weight warm-up can take minutes on a cold start")
    p.add_argument("--post-send-settle", type=float, default=10.0)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    # eval
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--eval-conda-env", default="qwen3vl")
    p.add_argument("--eval-model", default=None,
                   help="path/Hub id of the Qwen3-VL weights; default is the "
                        "bundled aicodec/aicodec/models/Qwen3_VL_30B_A3B_Instruct")
    p.add_argument("--eval-gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES only for eval; default inherits "
                        "the CUDA_VISIBLE_DEVICES you exported for the pipeline")
    p.add_argument("--eval-limit", type=int, default=None,
                   help="score only N samples (smoke test)")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = p.parse_args()

    if args.mode is not None:
        args.scenario = f"odp_image_udp_{args.mode}"

    args.out_dir = args.out_dir if args.out_dir.is_absolute() \
        else (AITESTBED_DIR / args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"repo root : {REPO_ROOT}")
    log(f"output    : {args.out_dir}")
    log(f"scenario  : {args.scenario}")
    log(f"profiles  : {args.profiles}")
    if args.payload_drop_mode is not None or args.payload_drop_pct is not None:
        log(f"payload drop: mode={args.payload_drop_mode} "
            f"pct={args.payload_drop_pct} (active only under profile "
            f"'no_emulation')")

    all_summaries = []
    for profile in args.profiles:
        log(f"==== profile: {profile} ====")
        all_summaries.append(run_one_profile(args, profile))

    combined = args.out_dir / "combined_summary.json"
    combined.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log("==== done ====")
    for s in all_summaries:
        acc = (s.get("eval") or {}).get("raw_accuracy")
        log(f"{s['profile']:>16}: png={s.get('recon_png')} "
            f"rows={s.get('recon_manifest_rows')} raw_accuracy={acc}")
    log(f"combined summary -> {combined}")


if __name__ == "__main__":
    main()
