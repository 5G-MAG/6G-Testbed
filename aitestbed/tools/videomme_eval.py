#!/usr/bin/env python3
"""
Video-MME dataset + netem helpers shared by the encoder_placement bench
tools (tools/encpl_bench_eval*.py) and scenarios.

Extracted from the former standalone split-inference Video-MME driver;
the evaluation main() was removed together with the split_inference
scenario, the shared dataset / prompt / netem plumbing below stayed.

Dataset layout expected (HF official):
    $DATASET_DIR/data/test-00000-of-00001.parquet
    $DATASET_DIR/videozip or $DATASET_DIR/videos  (unzipped Video-MME videos, mp4)

Netem topology (apply_netem / clear_netem):
    The server base URL decides the netem strategy automatically:
    - localhost/127.0.0.1 (single host): port-filtered shaping on loopback
      (lo:<port from base URL>, only that port's traffic is shaped).
    - remote host (dual host): whole-interface shaping on the edge machine's
      egress interface (--interface, default auto-detect from the default
      route). NOTE: the whole interface is shaped, so other traffic on that
      interface (e.g. your ssh session to the edge machine) is also impaired
      under harsh profiles — use tmux/a out-of-band console for long runs.
      Asymmetric profiles (with an `uplink:` block, e.g. satellite_leo) are
      only supported in the dual-host case; on a single host the caller
      exits with an error instead of silently mis-shaping.
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
AITESTBED_ROOT = REPO_ROOT / "aitestbed"
sys.path.insert(0, str(AITESTBED_ROOT))
# Bootstrap sibling packages (aicodec, netemu) the same way orchestrator.py
# does, so importing clients triggers realtime_webrtc_vlm_client's
# `import aicodec` and netemu is importable for tc shaping.
for _p in (REPO_ROOT / "aicodec", REPO_ROOT / "netemu" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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


def _is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "")


def _base_url_port(base_url: str, default: int = 8001) -> int:
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

PROMPT_TEMPLATE = '''You are an expert video question-answering assistant. Watch the video and select the correct option based STRICTLY on visual evidence.

Question: {}
Options:
{}
{}
{}
{}

Respond with ONLY the single letter (A, B, C, or D) of the best option. No reasoning, no JSON, no markdown, no extra text.'''


def find_video_root(dataset_dir: Path) -> Path | None:
    for cand in ("videozip", "videos", "video", "data/videozip", "data/videos"):
        p = dataset_dir / cand
        if p.is_dir():
            return p
    return None


def find_parquet(dataset_dir: Path) -> Path | None:
    for p in sorted(dataset_dir.rglob("test-*.parquet")):
        return p
    for p in sorted(dataset_dir.rglob("*.parquet")):
        return p
    return None


def apply_netem(base_url: str, profile_name: str, direction: str,
                interface: str = "auto",
                udp_port: int | None = None) -> None:
    """Apply the named profiles.yaml profile, topology-aware.

    Single host (base URL localhost/127.0.0.1):
        port-filtered shaping on loopback lo:<port from base URL>.
        - "both": netemu's apply_profile_to_loopback (dport AND sport u32
          filters — on lo both directions traverse the egress qdisc).
        - "uplink-only": same prio+netem tree but only the dport filter, so
          only client->server (upload) traffic is shaped; the SSE token
          stream stays unimpaired.
        Asymmetric profiles (with `uplink:`) are rejected: the loopback
        single-port filters cannot express per-direction parameters.

        udp_port (optional, default None = legacy TCP-only): also shapes
        that UDP port's traffic in the SAME netem band (both = dport+sport
        pair via netemu's udp_port; uplink-only = a dport-only UDP filter
        beside the TCP one). Used by two-tier (optional retention) edge
        runs whose optional tier rides best-effort UDP datagrams (E2U);
        without it those datagrams bypass netem entirely on single-host
        loopback runs. Ignored in dual-host mode (whole-interface shaping
        already covers UDP) and for no_emulation (cleared anyway).

    Dual host (base URL points at a remote server):
        whole-interface shaping on this (edge) machine's egress interface,
        entirely via native netemu calls. WARNING: the whole interface is
        shaped — other traffic (e.g. ssh) is impaired too; run long evals
        inside tmux or via an out-of-band console.
        - "uplink-only": bidirectional=False (or disable_ingress for
          asymmetric profiles) — shapes the edge egress = the upload only.
        - "both": IFB ingress shaping (downlink = server->edge SSE).

    Raises SystemExit on any failure: silently running on a bare network
    while the report claims a profile would invalidate results.
    """
    import subprocess

    import yaml as _yaml
    from netemu.emulator import NetworkEmulator

    profiles_path = REPO_ROOT / "aitestbed" / "configs" / "profiles.yaml"
    single_host = _is_local_base_url(base_url)
    lo_port = _base_url_port(base_url)

    if single_host:
        emulator = NetworkEmulator(interface="lo",
                                   profiles_path=str(profiles_path))
    else:
        emulator = NetworkEmulator(interface=interface,
                                   profiles_path=str(profiles_path))

    # Raw profile dict (keeps `uplink:` blocks; netemu strips them).
    with open(profiles_path) as f:
        raw_profiles = (_yaml.safe_load(f) or {}).get("profiles", {})
    raw = raw_profiles.get(profile_name)
    is_asymmetric = isinstance(raw, dict) and bool(raw.get("uplink"))

    if profile_name == "no_emulation":
        if single_host:
            emulator.clear_loopback()
        else:
            emulator.clear()
        print(f"[VideoMME] netem: no_emulation (cleared "
              f"{'lo' if single_host else emulator.interface})")
        return

    if profile_name not in emulator.profiles:
        sys.exit(f"unknown profile '{profile_name}' "
                 f"(not in {profiles_path})")

    if single_host:
        if is_asymmetric:
            sys.exit(
                f"profile '{profile_name}' has an `uplink:` block; asymmetric "
                f"profiles are only supported in dual-host mode (remote "
                f"--base-url), where egress and ingress are separate interfaces"
            )
        if direction == "both":
            if not emulator.apply_profile_to_loopback(profile_name, lo_port,
                                                      udp_port=udp_port):
                sys.exit(f"failed to apply netem profile '{profile_name}' to "
                         f"lo:{lo_port} (both directions)")
            print(f"[VideoMME] netem: {profile_name} on lo:{lo_port} "
                  + (f"+udp:{udp_port} " if udp_port else "")
                  + "(single host, both directions)")
            return

        # uplink-only on lo: same tc tree as apply_profile_to_loopback but
        # only the dport filter (commands mirror netemu emulator.py;
        # clear_loopback() is reused verbatim for teardown).
        profile = emulator.profiles[profile_name]
        netem_params = emulator._build_netem_params(
            delay_ms=profile.delay_ms,
            jitter_ms=profile.jitter_ms,
            delay_distribution=profile.delay_distribution,
            delay_correlation_pct=profile.delay_correlation_pct,
            loss_pct=profile.loss_pct,
            loss_correlation_pct=profile.loss_correlation_pct,
            loss_model=profile.loss_model,
            corruption_pct=profile.corruption_pct,
            corruption_correlation_pct=profile.corruption_correlation_pct,
            reorder_pct=profile.reorder_pct,
            reorder_correlation_pct=profile.reorder_correlation_pct,
            duplicate_pct=profile.duplicate_pct,
            duplicate_correlation_pct=profile.duplicate_correlation_pct,
            limit_packets=profile.limit_packets,
        )
        if not netem_params:
            print(f"[VideoMME] netem: profile '{profile_name}' has no "
                  f"impairment fields; lo:{lo_port} left unshaped")
            return

        def _tc(cmd: str) -> None:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=10)
            if r.returncode != 0:
                sys.exit(f"netem command failed: {cmd}\n{r.stderr.strip()}")

        emulator.clear_loopback()  # idempotent defensive clear
        _tc("sudo tc qdisc add dev lo root handle 1: prio bands 3")
        _tc(f"sudo tc qdisc add dev lo parent 1:3 handle 30: netem "
            f"{' '.join(netem_params)}")
        _tc(f"sudo tc filter add dev lo parent 1:0 protocol ip prio 1 u32 "
            f"match ip protocol 6 0xff "
            f"match ip dport {lo_port} 0xffff flowid 1:3")
        if udp_port:
            # Optional-tier UDP wave joins the same band, dport-only
            # (uplink semantics: terminal -> sidecar). IP protocol 17
            # = UDP; same header offsets as TCP.
            _tc(f"sudo tc filter add dev lo parent 1:0 protocol ip prio 1 "
                f"u32 match ip protocol 17 0xff "
                f"match ip dport {int(udp_port)} 0xffff flowid 1:3")
        print(f"[VideoMME] netem: {profile_name} on lo:{lo_port} "
              + (f"+udp:{udp_port} " if udp_port else "")
              + "(single host, uplink-only dport filter): "
              + " ".join(netem_params))
        return

    # ---- Dual host: whole-interface shaping, native netemu only ----
    iface = emulator.interface
    egress_dict = _uplink_merged_profile(raw) if is_asymmetric else None

    if is_asymmetric:
        # egress = base ⊕ uplink, ingress = base (downlink)
        egress_dict = egress_dict or {}
        ingress_dict = {k: v for k, v in raw.items()
                        if k in _PROFILE_IMPAIRMENT_FIELDS}
        if direction == "uplink-only":
            ok = emulator.apply_settings(
                **egress_dict, disable_ingress=True,
                profile_name=f"{profile_name}:uplink")
        else:
            ok = emulator.apply_settings(
                **egress_dict, ingress_settings=ingress_dict,
                profile_name=profile_name)
        if not ok:
            sys.exit(f"failed to apply netem profile '{profile_name}' to "
                     f"{iface} (dual host, {direction})")
        print(f"[VideoMME] netem: {profile_name} on {iface} "
              f"(dual host, {direction}, asymmetric: egress=base+uplink"
              f"{' ingress=base' if direction == 'both' else ''})")
        return

    if direction == "uplink-only":
        if not emulator.apply_profile(profile_name, ingress_profile="none"):
            sys.exit(f"failed to apply netem profile '{profile_name}' to "
                     f"{iface} (dual host, uplink-only)")
        print(f"[VideoMME] netem: {profile_name} on {iface} "
              f"(dual host, uplink-only egress)")
    else:
        if not emulator.apply_profile(profile_name):
            sys.exit(f"failed to apply netem profile '{profile_name}' to "
                     f"{iface} (dual host, both, IFB ingress)")
        print(f"[VideoMME] netem: {profile_name} on {iface} "
              f"(dual host, both directions, IFB ingress)")


def clear_netem(base_url: str) -> None:
    """Remove any netem rules (idempotent, safe on exit).

    Single host: loopback qdisc; dual host: interface + IFB teardown.
    """
    try:
        from netemu.emulator import NetworkEmulator
        if _is_local_base_url(base_url):
            NetworkEmulator(interface="lo").clear_loopback()
        else:
            NetworkEmulator(interface="auto").clear()
    except Exception as e:  # noqa: BLE001 — best-effort cleanup
        print(f"[VideoMME] warning: netem cleanup failed: {e}")


def load_entries(parquet_path: Path, video_root: Path, limit: int,
                 sample: str = "stratified") -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    entries = []
    for _, row in df.iterrows():
        video_id = str(row.get("videoID") or row.get("video_id") or "")
        # Video-MME video file names: YouTubeID + '_' + duration + '.mp4'
        video_id = video_id.replace("YouTube_", "")
        candidates = [
            video_root / f"{video_id}.mp4",
            video_root / video_id / f"{video_id}.mp4",
        ]
        video_path = next((c for c in candidates if c.exists()), None)
        raw_options = row.get("options")
        options = [str(o) for o in raw_options] if raw_options is not None else []
        if len(options) != 4:
            continue
        entry = {
            "video_id": video_id,
            "video_path": str(video_path) if video_path else None,
            "question": str(row.get("question") or ""),
            "options": options,
            "answer": str(row.get("answer") or "").strip().upper(),
            "task_type": str(row.get("task_type") or "unknown"),
            "domain": str(row.get("domain") or ""),
            "duration": str(row.get("duration") or ""),
        }
        entries.append(entry)

    if limit > 0:
        if sample == "stratified":
            # Stratified sampling by duration (short/medium/long): decode time
            # and token count scale with video length, so a head-slice sample
            # (rows clustered by video) misrepresents the workload mix.
            buckets: dict[str, list[dict]] = {}
            for e in entries:
                buckets.setdefault(e["duration"], []).append(e)
            n_buckets = len(buckets)
            per_bucket = limit // n_buckets
            remainder = limit - per_bucket * n_buckets
            # Distribute the remainder to the largest buckets so the sample
            # always hits exactly `limit` (or all available) entries.
            quotas = {dur: per_bucket for dur in buckets}
            for dur in sorted(buckets, key=lambda d: -len(buckets[d])):
                if remainder <= 0:
                    break
                if len(buckets[dur]) > quotas[dur]:
                    quotas[dur] += 1
                    remainder -= 1
            sampled: list[dict] = []
            for dur in sorted(buckets):
                sampled.extend(buckets[dur][:quotas[dur]])
            entries = sampled[:limit]
        else:  # "head": legacy behaviour, first N rows in parquet order
            entries = entries[:limit]
    return entries


def build_prompt(question: str, options: list[str]) -> str:
    opts = [o if ". " in o[:4] else f"{'ABCD'[i]}. {o}" for i, o in enumerate(options)]
    return PROMPT_TEMPLATE.format(question, *opts)


def save(path: Path, results: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
