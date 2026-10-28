#!/usr/bin/env python3
"""
run_qwen36_matrix.py — full-matrix driver for the Qwen3.6
encoder-placement experiment (serial + concurrent, server/edge modes,
vidcom2 retention ladder, two-tier optional retention ladder, network
profiles), with per-cell pcap capture, per-cell SQLite DBs and a final
aggregated report.

Standalone tool: ZERO changes to any existing runner/scenario/client.
Every cell (one configuration x one network profile) is executed as a
subprocess of the EXISTING standalone runners:

  serial cells:   aitestbed/run_encoder_placement_qwen36.py
                  (sidecar :8004 / opt-udp :8005)
  concurrent:     aitestbed/run_encoder_placement_concurrent_qwen36.py
                  (sidecar :8014 / opt-udp :8015)

Per-cell pipeline (identical for every cell):

  1. netem  : NetworkEmulator.apply_profile_to_loopback(profile,
              tcp_port=<sidecar>, udp_port=<opt-udp, tiered only>) —
              the udp_port pair routes the E2U optional-tier datagram
              wave into the same netem band as the TCP traffic (the
              TCP-only filters would leave it unshaped on loopback).
              no_emulation clears lo instead. In DUAL-HOST mode
              (non-local --base-url) whole-interface egress shaping is
              used instead (netemu apply_settings; asymmetric profiles
              apply base+uplink on egress).
  2. pcap   : CaptureController -> results/captures/q36m/<cell_id>.pcap
              filter "tcp port <P> [or udp port <UDP>]".
  3. run    : subprocess of the standalone runner with a SANITIZED env
              (ENCPL_* injected only for the cells that need them).
  4. stop pcap, write turn_index=-2 capture metadata row into the
     cell's DB (same contract as the concurrent runner's own capture).
  5. GET /stats snapshot (opt_udp counters delta for tiered cells).
  6. clear netem, cooldown, next cell.

Sidecar lifecycle is managed sequentially (a single 80GB GPU cannot
hold two 35GB-FP8 engines): serial sidecar for phase 1, concurrent
sidecar for phase 2, started/killed by this driver via the bootstrap
scripts. --sidecar-manage skip assumes they are already up
(dual-host / manual operation).

Artifacts (all unique names per cell):
  logs/q36m/<cell_id>.db                     per-cell traffic DB
  logs/q36m/cell_logs/<cell_id>.log          per-cell runner stdout
  logs/q36m/manifest_<ts>.json               run manifest (resume)
  results/captures/q36m/<cell_id>.pcap       per-cell packet capture
  aitestbed/configs/q36m_overlay/<cell_id>.yaml  (concurrent tiered
             cells only — the concurrent scenario reads tiered knobs
             from the yaml config, NOT from env)
  results/q36m/report_<ts>.md / .json        aggregated final report

Usage:
  python aitestbed/tools/run_qwen36_matrix.py --dry-run
  python aitestbed/tools/run_qwen36_matrix.py --phases serial
  python aitestbed/tools/run_qwen36_matrix.py --phases serial \
      --retentions 0.6 --optional-retention 0.06,0.12,0.18
  python aitestbed/tools/run_qwen36_matrix.py --cell q36m_serial_server__no_emulation,q36m_serial_edge-plain__5g_urban
  python aitestbed/tools/run_qwen36_matrix.py --cell 'q36m_*edge-*-fp8__lossy'
  python aitestbed/tools/run_qwen36_matrix.py --resume logs/q36m/manifest_<ts>.json
  python aitestbed/tools/run_qwen36_matrix.py --report-only logs/q36m/manifest_<ts>.json

Requires (Linux host, target conda env active):
  - tcpdump + passwordless sudo (tc + tcpdump)
  - MLVU dataset root (json/ + video/), default /data01/mlvu/MLVU,
    override with --dataset-dir or env ENCPL_DATASET_DIR
  - sidecar bootstrap scripts at the repo root
"""

import argparse
import fnmatch
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AITESTBED_ROOT = REPO_ROOT / "aitestbed"
TOOLS_DIR = Path(__file__).resolve().parent
for _p in (str(TOOLS_DIR), str(AITESTBED_ROOT),
           str(REPO_ROOT / "netemu" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from capture.controller import CaptureController  # noqa: E402
from netemu.emulator import NetworkEmulator  # noqa: E402

LOG = "[q36m]"

# pcap-derived network stats (single-pass dpkt; O(1) memory so GB-scale
# captures stay reportable). PcapAnalyzer (analysis/pcap_analyzer.py)
# materializes per-packet records -> unusable here. dpkt is pinned in
# requirements-epd.txt (1.9.8). Direction semantics: ul = client->server
# decided by the cell's known server ports (role knowledge), NOT
# PcapAnalyzer's port-order heuristic — see _parse_pcap_flows.
try:
    import dpkt as _dpkt
    HAS_DPKT = True
except ImportError:  # pragma: no cover - Windows dev boxes
    _dpkt = None
    HAS_DPKT = False


# Link-layer handling for _parse_pcap_flows: tcpdump captures on lo
# default to Ethernet framing (linktype 1) — each frame carries a 14B
# (or VLAN-extended) header before the IP packet, which dpkt.ip.IP
# rejects wholesale (UnpackError on every packet -> flows == 0, the
# classic "pkts counted but zero flows" failure). Normalize each frame
# to raw IP before parsing. SLL (113, Linux "any" interface) strips a
# fixed 16B header; 276 is SLL2 (20B). Raw-IP variants need no strip.
_PCAP_LINKTYPE_OFFSETS = {
    0: 4,     # NULL/loopback (BSD): 4-byte family header
    1: 14,    # Ethernet (tcpdump default on lo)
    12: 0,    # raw IP (LINKTYPE_RAW)
    14: 0,    # raw IP (LINKTYPE_EN10MB with cooked IP? — some tun setups)
    101: 0,   # raw IP (old OpenBSD var)
    113: 16,  # Linux SLL ("any" interface)
    276: 20,  # Linux SLL2
}


def _strip_link_layer(buf: bytes, datalink: int) -> bytes | None:
    """Reduce a captured frame to its raw-IP bytes; None = skip frame.

    Ethernet goes through dpkt.ethernet so VLAN/qinq tags resolve to
    the right offset instead of a blind 14-byte cut. eth.data is a
    dpkt Packet OBJECT — bytes() it so the caller's dpkt.ip.IP() gets
    a buffer (Packet input makes ip.IP raise KeyError internally).
    """
    off = _PCAP_LINKTYPE_OFFSETS.get(datalink)
    if off is None:
        # unknown link type: try Ethernet (most common capture shape)
        # and fall back to raw IP — both failures land in the caller's
        # per-packet except-continue, never abort the whole parse.
        try:
            eth = _dpkt.ethernet.Ethernet(buf)
            return (bytes(eth.data) if isinstance(eth.data, _dpkt.ip.IP)
                    else None)
        except (_dpkt.dpkt.NeedData, _dpkt.dpkt.UnpackError):
            return buf
    if off == 0:
        return buf
    if datalink == 1:
        try:
            eth = _dpkt.ethernet.Ethernet(buf)
            return (bytes(eth.data) if isinstance(eth.data, _dpkt.ip.IP)
                    else None)
        except (_dpkt.dpkt.NeedData, _dpkt.dpkt.UnpackError):
            return None
    if len(buf) <= off:
        return None
    return buf[off:]


def _parse_pcap_flows(pcap_path: Path,
                      server_ports: set[int] | None = None) -> dict | None:
    """One-pass TCP/UDP flow aggregation over a cell pcap.

    Returns {"flows": [flow...], "ul_udp_bytes", "dl_udp_bytes",
    "ul_udp_pkts", "dl_udp_pkts", "n_packets", "t_span"} or None when
    the pcap is missing/empty/unparseable (caller renders "-").
    TCP flows carry the fields the two-level per-user join needs:
    src_port/dst_port, ul/dl bytes+pkts, retrans (dup-seq heuristic),
    rtt (SYN->SYN-ACK), first/last epoch ts.

    Byte accounting (v4): full-frame — every TCP frame's ip.len counts
    (pure ACK/SYN/FIN/retransmission alike), Ethernet header excluded;
    numbers reconcile with Wireshark Conversations minus 14B×frames.
    retrans detection stays data-only; ul_data_pkts is the data-frame
    subset used as the retr_rate denominator.

    Direction: ul = client->server. server_ports (capture-side ROLE
    knowledge: the sidecar's HTTP port, plus the opt-UDP port on
    tiered cells) is authoritative — the packet whose DST port is in
    the set is uplink, SRC in the set is downlink. The old
    sport<dport heuristic mislabeled flows whenever the client's
    ephemeral port happened to exceed the server port (>=8004/8014),
    swapping UL/DL bytes and SYN/SYN-ACK sides (bogus RTT). Packets
    matching NEITHER side (other services on the same host) keep the
    port-order fallback so foreign traffic still aggregates sanely.
    """
    if not HAS_DPKT:
        return None
    if not pcap_path or not Path(pcap_path).exists():
        return None
    size = Path(pcap_path).stat().st_size
    if size <= 24:  # header only / empty
        return None
    flows: dict[str, dict] = {}
    ul_udp_b = dl_udp_b = ul_udp_p = dl_udp_p = 0
    n_pkts = 0
    try:
        with open(pcap_path, "rb") as fh:
            try:
                reader = _dpkt.pcap.Reader(fh)
            except ValueError:
                fh.seek(0)
                reader = _dpkt.pcapng.Reader(fh)
            try:
                datalink = int(reader.datalink())
            except Exception:
                datalink = 1
            for ts, buf in reader:
                n_pkts += 1
                pkt = _strip_link_layer(buf, datalink)
                if pkt is None:
                    continue
                try:
                    ip = _dpkt.ip.IP(pkt)
                    if isinstance(ip.data, _dpkt.icmp.ICMP):
                        continue
                except (_dpkt.dpkt.NeedData, _dpkt.dpkt.UnpackError,
                        KeyError):
                    continue
                if not isinstance(ip.data, (_dpkt.tcp.TCP, _dpkt.udp.UDP)):
                    continue
                l4 = ip.data
                fwd = (f"{ip.src}:{l4.sport}->{ip.dst}:{l4.dport}")
                rev = (f"{ip.dst}:{l4.dport}->{ip.src}:{l4.sport}")
                key = fwd if fwd <= rev else rev
                wire = ip.len  # IP-layer bytes (analyzer's accounting)
                # Role-based direction: dst port in the server set ->
                # uplink; src port in the set -> downlink; neither ->
                # port-order fallback (foreign same-host traffic).
                if server_ports and l4.dport in server_ports:
                    ul_dir = True
                elif server_ports and l4.sport in server_ports:
                    ul_dir = False
                else:
                    ul_dir = l4.sport < l4.dport
                if isinstance(l4, _dpkt.udp.UDP):
                    if ul_dir:
                        ul_udp_b += wire
                        ul_udp_p += 1
                    else:
                        dl_udp_b += wire
                        dl_udp_p += 1
                    continue
                f = flows.setdefault(key, {
                    "key": key, "src_port": int(l4.sport),
                    "dst_port": int(l4.dport),
                    "ul_bytes": 0, "dl_bytes": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                    "ul_data_pkts": 0,
                    "retrans": 0, "rtt": None,
                    "syn_ts": None, "synack_ts": None,
                    "first_ts": ts, "last_ts": ts,
                    "_seen_seq": set(),
                })
                f["first_ts"] = min(f["first_ts"], ts)
                f["last_ts"] = max(f["last_ts"], ts)
                if l4.flags & (_dpkt.tcp.TH_SYN):
                    if ul_dir and f["syn_ts"] is None:
                        f["syn_ts"] = ts
                    elif not ul_dir and l4.flags & _dpkt.tcp.TH_ACK \
                            and f["syn_ts"] is not None \
                            and f["synack_ts"] is None:
                        f["synack_ts"] = ts
                        f["rtt"] = round(ts - f["syn_ts"], 4)
                # Full-frame accounting: EVERY TCP frame (pure ACK/SYN/
                # FIN/retransmission alike) counts ip.len into ul/dl
                # bytes — the analyzer's IP-layer口径 (Ethernet header
                # excluded), so numbers reconcile with Wireshark
                # Conversations minus 14B×frames. Retransmissions are
                # counted as normal packets (no exclusion).
                if ul_dir:
                    f["ul_bytes"] += wire
                    f["ul_pkts"] += 1
                else:
                    f["dl_bytes"] += wire
                    f["dl_pkts"] += 1
                # retrans detection stays DATA-only: a pure ACK's seq
                # is constant and would fabricate false retransmissions;
                # ul_data_pkts keeps the retr_rate denominator honest.
                payload = len(l4.data)
                if payload > 0 and ul_dir:
                    f["ul_data_pkts"] = f.get("ul_data_pkts", 0) + 1
                    seq = int(l4.seq)
                    if seq in f["_seen_seq"]:
                        f["retrans"] += 1
                    else:
                        f["_seen_seq"].add(seq)
    except (OSError, ValueError, _dpkt.dpkt.NeedData,
            _dpkt.dpkt.UnpackError, StopIteration):
        return None
    for f in flows.values():
        del f["_seen_seq"]
        f["first_ts"] = round(f["first_ts"], 4)
        f["last_ts"] = round(f["last_ts"], 4)
    return {
        "flows": list(flows.values()),
        "ul_udp_bytes": ul_udp_b, "dl_udp_bytes": dl_udp_b,
        "ul_udp_pkts": ul_udp_p, "dl_udp_pkts": dl_udp_p,
        "n_packets": n_pkts,
    }


def _pcap_cell_stats(pcap_path, cell_id: str, cache_dir: Path | None,
                     enable: bool = True,
                     server_ports: set[int] | None = None) -> dict | None:
    """Cell-level pcap aggregation with an mtime+size cache so
    --report-only re-runs skip the multi-minute re-parse of GB pcaps.
    Returns {"ul_bytes","dl_bytes","tcp_ul_bytes","tcp_dl_bytes",
    "udp_ul_bytes","udp_dl_bytes","retrans","retr_rate","rtt_ms",
    "flows","n_packets","t_span","_flows"} or {"missing": reason}
    when nothing is reportable. server_ports (role knowledge) drives
    UL/DL direction — see _parse_pcap_flows."""
    # bump when the pcap parser's behavior changes (e.g. the link-layer
    # fix that made Ethernet-framed captures parse; v3 = role-based
    # direction + _flows persisted into the cache for the per-user
    # join; v4 = full-frame byte accounting (ACK/SYN/FIN/retrans count
    # into ul/dl bytes, Ethernet header excluded) + Byte units) so
    # stale caches from older parser versions are ignored and
    # recomputed.
    _PCAP_PARSER_VERSION = 4
    if not enable:
        return None
    p = Path(pcap_path) if pcap_path else None
    if not p or not p.exists():
        return {"missing": "pcap file not found"}
    st = p.stat()
    if st.st_size == 0:
        return {"missing": "pcap empty (0 bytes)"}
    if not HAS_DPKT:
        return {"missing": "dpkt not installed (report host)"}
    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cf = cache_dir / f"{cell_id}.json"
        if cf.exists():
            try:
                cache = json.loads(cf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cache = None
        if cache and cache.get("_pcap_mtime") == st.st_mtime \
                and cache.get("_pcap_size") == st.st_size \
                and cache.get("_pcap_parser") == _PCAP_PARSER_VERSION:
            # strip ALL private _pcap_* bookkeeping keys before returning:
            # a leaked _pcap_parser would land in metrics["pcap"] ->
            # report_*.json, and only cache-HIT cells would carry it
            # (fresh-parse cells never do), making the report schema
            # inconsistent across cells. _flows (different prefix) stays:
            # the caller's per-user join needs it.
            for k in ("_pcap_mtime", "_pcap_size", "_pcap_parser"):
                cache.pop(k, None)
            return cache
    parsed = _parse_pcap_flows(p, server_ports)
    if parsed is None:
        return {"missing": "pcap unparseable or truncated"}
    fl = parsed["flows"]
    tcp_ul = sum(f["ul_bytes"] for f in fl)
    tcp_dl = sum(f["dl_bytes"] for f in fl)
    retr = sum(f["retrans"] for f in fl)
    ul_data_pkts = sum(f.get("ul_data_pkts", 0) for f in fl)
    rtts = [f["rtt"] for f in fl if f["rtt"] is not None]
    rtt_ms = _stats_block([r * 1000 for r in rtts]) if rtts else None
    t0 = min((f["first_ts"] for f in fl), default=None)
    t1 = max((f["last_ts"] for f in fl), default=None)
    out = {
        "pcap_bytes": st.st_size,
        "n_packets": parsed["n_packets"],
        "flows": len(fl),
        "tcp_ul_bytes": tcp_ul,
        "tcp_dl_bytes": tcp_dl,
        "udp_ul_bytes": parsed["ul_udp_bytes"],
        "udp_dl_bytes": parsed["dl_udp_bytes"],
        "ul_bytes": tcp_ul + parsed["ul_udp_bytes"],
        "dl_bytes": tcp_dl + parsed["dl_udp_bytes"],
        "retrans": retr,
        "retr_rate": round(retr / ul_data_pkts, 4) if ul_data_pkts else 0.0,
        "rtt_ms": rtt_ms,
        "t_span": (round(t1 - t0, 1)
                   if t0 is not None and t1 is not None else None),
        # raw flows for the per-user join; since v3 they are PERSISTED
        # into the cache too (report-only re-runs used to lose them and
        # silently skip the join; only the cache's bookkeeping keys are
        # private)
        "_flows": fl,
    }
    if cache_dir is not None and t0 is not None:
        cf = cache_dir / f"{cell_id}.json"
        try:
            persist = {k: v for k, v in out.items()
                       if not k.startswith("_pcap_")}
            persist["_pcap_mtime"] = st.st_mtime
            persist["_pcap_size"] = st.st_size
            persist["_pcap_parser"] = _PCAP_PARSER_VERSION
            cf.write_text(json.dumps(persist), encoding="utf-8")
        except OSError:
            pass
    return out


def _manifest_pcap_path(manifest: dict, cell: dict) -> str | None:
    """Locate a cell's pcap: manifest result pcap.path (written when the
    driver stopped the capture), else the CAPTURE_DIR convention path
    (<cell_id>.pcap). Both exist for every completed cell; the report
    host sees them via the same share the runner used."""
    r = (manifest.get("results") or {}).get(cell.get("cell_id")) or {}
    p = ((r.get("pcap") or {}).get("path")
         if isinstance(r.get("pcap"), dict) else r.get("pcap"))
    if p and Path(p).exists():
        return p
    cid = cell.get("cell_id")
    conv = None
    if cid:
        conv = CAPTURE_DIR / f"{cid}.pcap"
        if conv.exists():
            return str(conv)
    return p or (str(conv) if cid else None)


DEFAULT_PROFILES = ["no_emulation", "5g_urban", "wifi_good", "cell_edge"]
DEFAULT_RETENTIONS = [0.8, 0.7, 0.6, 0.4, 0.3]
DEFAULT_QUANTS = ["fp8", "bf16"]
DEFAULT_USERS = [2, 4, 8, 16]
DEFAULT_SERIAL_LIMIT = 50
DEFAULT_CONC_VIDEOS_PER_USER = 3
DEFAULT_DATASET_DIR = "/data01/mlvu/MLVU"

SERIAL_RUNNER = AITESTBED_ROOT / "run_encoder_placement_qwen36.py"
CONCURRENT_RUNNER = AITESTBED_ROOT / "run_encoder_placement_concurrent_qwen36.py"
SERIAL_SIDECAR_SCRIPT = REPO_ROOT / "start_sidecar_qwen36.sh"
CONCURRENT_SIDECAR_SCRIPT = REPO_ROOT / "start_sidecar_concurrent_qwen36.sh"
CONCURRENT_CONFIG_TEMPLATE = (AITESTBED_ROOT / "configs"
                              / "scenarios_encoder_placement_concurrent_qwen36.yaml")

SERIAL_PORT = 8004
SERIAL_UDP_PORT = 8005
CONC_PORT = 8014
CONC_UDP_PORT = 8015

OUT_DIR = REPO_ROOT / "logs" / "q36m"
CELL_LOG_DIR = OUT_DIR / "cell_logs"
CAPTURE_DIR = AITESTBED_ROOT / "results" / "captures" / "q36m"
REPORT_DIR = AITESTBED_ROOT / "results" / "q36m"
OVERLAY_DIR = AITESTBED_ROOT / "configs" / "q36m_overlay"

# tiered timing knobs (serial cells: env; concurrent cells: yaml keys)
TIERED_ENV = {
    "optional": "ENCPL_OPTIONAL_RATIO",
    "tokens_per_packet": "ENCPL_OPT_TOKENS_PER_PACKET",
    "send_gap_ms": "ENCPL_OPT_SEND_GAP_MS",
    "chat_delay_ms": "ENCPL_OPT_CHAT_DELAY_MS",
}
TIERED_DEFAULTS = {"tokens_per_packet": 64, "send_gap_ms": 0.05,
                   "chat_delay_ms": 0.0}

# env vars that must NEVER leak from the driver's environment into a
# cell subprocess (each cell fully specifies its own arm). SGLANG_ is
# stripped wholesale: SGLANG_DECORD_NUM_THREADS etc. would otherwise
# ride the driver's shell into every cell (no cell needs any SGLANG_
# var; --base-url is always passed as a flag, which wins anyway).
SANITIZE_ENV_PREFIXES = ("ENCPL_", "ENC_MODE", "SGLANG_ENGINE",
                         "SGLANG_", "COMPRESSOR", "R_RATIO")


# ---------------------------------------------------------------------------
# matrix definition
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    cell_id: str
    phase: str                     # serial | concurrent
    profile: str
    mode: str                      # server | edge
    arm: str                       # descriptive arm label
    runner: str                    # runner script path
    db: str
    pcap: str
    log: str
    limit: int                     # dataset question count
    users: int = 1
    videos_per_user: int = 0       # concurrent only; 0 = n/a (serial)
    retention: float = 0.0         # 0 = uncompressed edge / server
    optional: float = 0.0          # tiered only
    quant: str = ""                # "" | fp8
    tiered: bool = False
    overlay_config: str = ""       # concurrent tiered cells only
    cmd_env: dict = field(default_factory=dict)
    cmd_args: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id, "phase": self.phase,
            "profile": self.profile, "mode": self.mode, "arm": self.arm,
            "users": self.users,
            "videos_per_user": self.videos_per_user or None,
            "limit": self.limit,
            "retention": self.retention or None,
            "optional": self.optional or None,
            "quant": self.quant or None, "tiered": self.tiered,
            "db": self.db, "pcap": self.pcap, "log": self.log,
            "overlay_config": self.overlay_config or None,
            "runner": self.runner,
        }


def _fmt_ratio(r: float) -> str:
    return ("%g" % r)


def build_matrix(args) -> list[Cell]:
    cells: list[Cell] = []

    def base_id(phase: str, arm: str, profile: str) -> str:
        return f"q36m_{phase}_{arm}__{profile}"

    # ---- serial phase ------------------------------------------------
    if args.phases in ("serial", "both"):
        n = args.serial_limit
        # A: server baseline
        for p in args.profiles:
            cid = base_id("serial", "server", p)
            cells.append(Cell(
                cell_id=cid, phase="serial", profile=p, mode="server",
                arm="server", runner=str(SERIAL_RUNNER), limit=n,
                db=str(OUT_DIR / f"{cid}.db"),
                pcap=str(CAPTURE_DIR / f"{cid}.pcap"),
                log=str(CELL_LOG_DIR / f"{cid}.log"),
                cmd_args=["--mode", "server", "--profile", p,
                          "--dataset", "mlvu", "--dataset-dir",
                          args.dataset_dir, "--limit", str(n),
                          "--runs", "1"]))
        # B: edge uncompressed anchor
        for p in args.profiles:
            cid = base_id("serial", "edge-plain", p)
            cells.append(Cell(
                cell_id=cid, phase="serial", profile=p, mode="edge",
                arm="edge-plain", runner=str(SERIAL_RUNNER), limit=n,
                db=str(OUT_DIR / f"{cid}.db"),
                pcap=str(CAPTURE_DIR / f"{cid}.pcap"),
                log=str(CELL_LOG_DIR / f"{cid}.log"),
                cmd_args=["--mode", "edge", "--profile", p,
                          "--dataset", "mlvu", "--dataset-dir",
                          args.dataset_dir, "--limit", str(n),
                          "--runs", "1"]))
        # C: single-tier retention ladder / D: two-tier ladder
        for r in args.retentions:
            for q in args.quants:
                qtag = q if q in ("fp8",) else ""
                for p in args.profiles:
                    # C: single-tier
                    arm = f"edge-vidcom2-r{_fmt_ratio(r)}" + (f"-{qtag}" if qtag else "")
                    cid = base_id("serial", arm, p)
                    cargs = ["--mode", "edge", "--compressor", "vidcom2",
                             "--retention-ratio", _fmt_ratio(r), "--profile", p,
                             "--dataset", "mlvu", "--dataset-dir",
                             args.dataset_dir, "--limit", str(n), "--runs", "1"]
                    if qtag:
                        cargs += ["--bundle-quant", qtag]
                    cells.append(Cell(
                        cell_id=cid, phase="serial", profile=p, mode="edge",
                        arm=arm, runner=str(SERIAL_RUNNER), limit=n,
                        retention=r, quant=qtag,
                        db=str(OUT_DIR / f"{cid}.db"),
                        pcap=str(CAPTURE_DIR / f"{cid}.pcap"),
                        log=str(CELL_LOG_DIR / f"{cid}.log"),
                        cmd_args=cargs))
                    # D: two-tier ladder — optional-retention values come
                    # from --optional-retention when given (absolute o
                    # ladder), else the legacy single derived o = r*0.30.
                    # The client contract requires 0 < o < r; an invalid
                    # (r, o) pair is warned and SKIPPED (same policy as
                    # the concurrent bench's mismatched-combo skips), not
                    # an abort of the whole matrix.
                    for o in (args.optionals
                              or [round(r * 0.30, 4)]):
                        if not (0.0 < o < r):
                            print(f"{LOG} WARNING: skipping serial tiered "
                                  f"cell r={_fmt_ratio(r)} "
                                  f"optional={_fmt_ratio(o)} "
                                  f"(requires 0 < o < r)")
                            continue
                        arm2 = arm + f"-opt{_fmt_ratio(o)}"
                        cid2 = base_id("serial", arm2, p)
                        cargs2 = list(cargs)
                        cells.append(Cell(
                            cell_id=cid2, phase="serial", profile=p,
                            mode="edge",
                            arm=arm2, runner=str(SERIAL_RUNNER), limit=n,
                            retention=r, optional=o, quant=qtag, tiered=True,
                            db=str(OUT_DIR / f"{cid2}.db"),
                            pcap=str(CAPTURE_DIR / f"{cid2}.pcap"),
                            log=str(CELL_LOG_DIR / f"{cid2}.log"),
                            cmd_args=cargs2,
                            cmd_env={
                                TIERED_ENV["optional"]: str(o),
                                TIERED_ENV["tokens_per_packet"]:
                                    str(TIERED_DEFAULTS["tokens_per_packet"]),
                                TIERED_ENV["send_gap_ms"]:
                                    str(TIERED_DEFAULTS["send_gap_ms"]),
                                TIERED_ENV["chat_delay_ms"]:
                                    str(TIERED_DEFAULTS["chat_delay_ms"]),
                            }))

    # ---- concurrent phase ---------------------------------------------
    if args.phases in ("concurrent", "both"):
        # terminal-side CPU/decode knobs, applied to EVERY concurrent
        # cell (both arms — the runner accepts them in server mode too,
        # mirroring the README's tested commands). --conc-edge-cpu-ids
        # "" opts out (runner default pool).
        edge_ids = args.conc_edge_cpu_ids or ""
        client_tail: list[str] = []
        if edge_ids:
            client_tail += ["--edge-cpu-ids", edge_ids]
        client_tail += ["--edge-cpus-per-worker",
                        str(args.conc_edge_cpus_per_worker)]
        if args.conc_client_decode_threads > 0:
            client_tail += ["--decode-threads",
                            str(args.conc_client_decode_threads)]
        # Every (users, V) pair is a separate cell: the arm tag carries
        # -n{u}-v{v} so different V values never collide on cell_id
        # (cell_id/db/pcap/log/overlay all derive from it) and --resume
        # cannot mistake a V=3 cell for a V=5 one.
        for v in args.conc_videos_per_user:
            for u in args.users:
                n = v * u
                for p in args.profiles:
                    # server baseline
                    arm = f"server-n{u}-v{v}"
                    cid = base_id("conc", arm, p)
                    cells.append(Cell(
                        cell_id=cid, phase="concurrent", profile=p,
                        mode="server",
                        arm=arm, runner=str(CONCURRENT_RUNNER), limit=n, users=u,
                        videos_per_user=v,
                        db=str(OUT_DIR / f"{cid}.db"),
                        pcap=str(CAPTURE_DIR / f"{cid}.pcap"),
                        log=str(CELL_LOG_DIR / f"{cid}.log"),
                        cmd_args=["--mode", "server", "--users", str(u),
                                  "--profile", p, "--profile-record-only",
                                  "--dataset", "mlvu", "--dataset-dir",
                                  args.dataset_dir, "--limit", str(n),
                                  "--runs", "1"] + client_tail))
                    # edge retention arm (single + two-tier, both quants).
                    # Retention ladder: unset --retentions keeps the
                    # historical fixed r=0.6; explicit --retentions is
                    # honored here too (args.conc_retentions mirrors
                    # main()'s split — bare namespaces fall back to 0.6).
                    # NOTE: `or [0.6]` would resurrect an EXPLICIT empty
                    # list (the documented `--retentions ''` "disable
                    # edge cells" mode) — only a MISSING attribute is
                    # the "unset" case.
                    conc_rs = getattr(args, "conc_retentions", None)
                    if conc_rs is None:
                        conc_rs = [0.6]
                    for r in conc_rs:
                        for q in args.quants:
                            qtag = q if q in ("fp8",) else ""
                            # single-tier (legacy naming: no tier tag — the
                            # absence of -opt{g} IS the single-tier marker,
                            # mirroring the serial arm's convention)
                            sarm = (f"edge-r{_fmt_ratio(r)}"
                                    + (f"-{qtag}" if qtag else "")
                                    + f"-n{u}-v{v}")
                            sargs = ["--mode", "edge", "--compressor", "vidcom2",
                                     "--retention-ratio", _fmt_ratio(r),
                                     "--users", str(u), "--profile", p,
                                     "--profile-record-only",
                                     "--dataset", "mlvu", "--dataset-dir",
                                     args.dataset_dir, "--limit", str(n),
                                     "--runs", "1"] + client_tail
                            if qtag:
                                sargs += ["--bundle-quant", qtag]
                            cells.append(Cell(
                                cell_id=base_id("conc", sarm, p),
                                phase="concurrent", profile=p, mode="edge",
                                arm=sarm, runner=str(CONCURRENT_RUNNER),
                                limit=n, users=u, videos_per_user=v,
                                retention=r, quant=qtag,
                                db=str(OUT_DIR / f"{base_id('conc', sarm, p)}.db"),
                                pcap=str(CAPTURE_DIR /
                                         f"{base_id('conc', sarm, p)}.pcap"),
                                log=str(CELL_LOG_DIR /
                                        f"{base_id('conc', sarm, p)}.log"),
                                cmd_args=sargs))
                            # two-tier concurrent: knobs ONLY via yaml overlay.
                            # optional-retention values come from
                            # --optional-retention when given (absolute o ladder),
                            # else the legacy single derived o = r*0.30. The arm
                            # name carries -opt{g} (no 2tier tag — same
                            # convention as the serial arm: opt's presence IS
                            # the two-tier marker) so multiple o values per r
                            # stay collision-free (cell_id/db/pcap/log/overlay
                            # yaml all derive from it). Invalid pairs
                            # (0 < o < r) are warned and skipped, mirroring the
                            # serial ladder.
                            for o in (args.optionals
                                      or [round(r * 0.30, 4)]):
                                if not (0.0 < o < r):
                                    print(f"{LOG} WARNING: skipping concurrent "
                                          f"tiered cell r={_fmt_ratio(r)} "
                                          f"optional={_fmt_ratio(o)} "
                                          f"(requires 0 < o < r)")
                                    continue
                                tarm = (f"edge-r{_fmt_ratio(r)}"
                                        + (f"-{qtag}" if qtag else "")
                                        + f"-opt{_fmt_ratio(o)}" + f"-n{u}-v{v}")
                                tcid = base_id("conc", tarm, p)
                                overlay = str(OVERLAY_DIR / f"{tcid}.yaml")
                                targs = list(sargs)
                                cells.append(Cell(
                                    cell_id=tcid, phase="concurrent",
                                    profile=p, mode="edge", arm=tarm,
                                    runner=str(CONCURRENT_RUNNER), limit=n,
                                    users=u, videos_per_user=v,
                                    retention=r, optional=o, quant=qtag,
                                    tiered=True,
                                    overlay_config=overlay,
                                    db=str(OUT_DIR / f"{tcid}.db"),
                                    pcap=str(CAPTURE_DIR / f"{tcid}.pcap"),
                                    log=str(CELL_LOG_DIR / f"{tcid}.log"),
                                    cmd_args=targs))
    return cells


def finalize_cell_cmd(cell: Cell, args) -> None:
    """Append the per-phase constant flags (db/base-url/config/stream
    timeouts)."""
    port = SERIAL_PORT if cell.phase == "serial" else CONC_PORT
    base = f"http://{args.server_host}:{port}"
    # Stream safety timeouts ride as CLI flags into every cell's
    # runner (which forwards them to the client). The first-event
    # budget is phase-specific: serial arms see no queueing, barrier
    # waves in concurrent cells do (tail users wait minutes).
    stream_flags = [
        "--stream-connect-timeout", str(args.stream_connect_timeout),
        "--stream-first-event-timeout",
        str(args.stream_first_event_timeout
            if cell.phase == "serial"
            else args.conc_stream_first_event_timeout),
        "--stream-idle-timeout", str(args.stream_idle_timeout),
        "--stream-upload-read-timeout",
        str(args.stream_upload_read_timeout),
    ]
    if cell.phase == "serial":
        cell.cmd_args += ["--db", cell.db, "--base-url", base] + stream_flags
    else:
        if cell.overlay_config:
            cell.cmd_args += ["--config", cell.overlay_config]
        cell.cmd_args += ["--db", cell.db, "--base-url", base,
                          "--no-model-check"] + stream_flags


def write_overlay_config(cell: Cell) -> None:
    """Generate the per-cell yaml overlay for concurrent TIERED cells.

    The concurrent scenario reads optional_ratio / opt_* ONLY from the
    config dict (no env, no CLI flag), so tiered knobs must ride a
    modified copy of the default scenarios yaml.
    """
    with open(CONCURRENT_CONFIG_TEMPLATE, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = raw["scenarios"]["encoder_placement_concurrent_qwen36"]
    cfg["optional_ratio"] = cell.optional
    cfg["opt_tokens_per_packet"] = TIERED_DEFAULTS["tokens_per_packet"]
    cfg["opt_send_gap_ms"] = TIERED_DEFAULTS["send_gap_ms"]
    cfg["opt_chat_delay_ms"] = TIERED_DEFAULTS["chat_delay_ms"]
    # The runner anchors a relative video_base_dir to the CONFIG's
    # parent.parent; the overlay sits one level deeper than the
    # default yaml (configs/q36m_overlay/ vs configs/), which would
    # resolve "." to aitestbed/configs instead of aitestbed/. Dataset
    # mode never reads it, but pin it explicitly so the overlay is a
    # faithful copy of the default config's semantics.
    if cfg.get("video_base_dir") == ".":
        cfg["video_base_dir"] = str(AITESTBED_ROOT)
    Path(cell.overlay_config).parent.mkdir(parents=True, exist_ok=True)
    with open(cell.overlay_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# helpers: env, netem, capture, sidecar, stats
# ---------------------------------------------------------------------------

def sanitized_env(extra: dict | None = None) -> dict:
    """Whitelist env for cell subprocesses: PATH/HOME/CONDA_* + native
    library paths + thread caps + dataset + per-cell tiered vars.
    Strips every ENCPL_/ENC_MODE/SGLANG_/COMPRESSOR/R_RATIO var so a
    stale outer env cannot silently flip a cell's arm."""
    keep = {}
    for k, v in os.environ.items():
        if k.startswith(SANITIZE_ENV_PREFIXES):
            continue
        if k in ("PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL",
                 "TERM", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
                 "LD_LIBRARY_PATH", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS") \
                or k.startswith("CONDA_") or k.startswith("CUDA_") \
                or k.startswith("NVIDIA_") or k.startswith("SLURM_"):
            keep[k] = v
    if extra:
        keep.update(extra)
    return keep


def is_local(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "")


def apply_netem(cell: Cell, base_url: str) -> dict:
    """Topology-aware netem, mirroring the concurrent runner's own
    _apply_netem semantics exactly:

    single host: loopback port filters (TCP + optional UDP pair via
      netemu's udp_port; asymmetric profiles are REJECTED — the
      single-port filters cannot express per-direction parameters);
    dual host, symmetric profile: whole-interface egress+ingress via
      netemu apply_profile (IFB ingress), like the runner;
    dual host, asymmetric profile: egress = base + uplink merge
      (upload direction dominant), no ingress shaping;
    no_emulation: clear the loopback qdisc (single) / the egress
      interface (dual) so no previous cell's shaping can leak.
    """
    info: dict = {"mode": "loopback" if is_local(base_url) else "egress",
                  "profile": cell.profile}
    profiles_path = str(AITESTBED_ROOT / "configs" / "profiles.yaml")
    local = is_local(base_url)

    if cell.profile == "no_emulation":
        if local:
            NetworkEmulator(interface="lo").clear_loopback()
        else:
            NetworkEmulator(interface=args_netem_iface(),
                            profiles_path=profiles_path).clear()
        return info

    emu = NetworkEmulator(interface="lo" if local else args_netem_iface(),
                          profiles_path=profiles_path)

    # Raw profile dict keeps `uplink:` blocks (netemu strips them on
    # load); encoding="utf-8" for the em-dash comments in the yaml.
    with open(AITESTBED_ROOT / "configs" / "profiles.yaml",
              encoding="utf-8") as f:
        raw = (yaml.safe_load(f) or {}).get("profiles", {}).get(
            cell.profile, {})
    is_asymmetric = isinstance(raw, dict) and bool(raw.get("uplink"))

    if local:
        if is_asymmetric:
            raise SystemExit(
                f"profile '{cell.profile}' has an `uplink:` block; "
                "asymmetric profiles are only supported in dual-host "
                "mode (remote --server-host)")
        if cell.profile not in emu.profiles:
            raise SystemExit(f"unknown profile '{cell.profile}'")
        port = SERIAL_PORT if cell.phase == "serial" else CONC_PORT
        udp = (SERIAL_UDP_PORT if cell.phase == "serial" else CONC_UDP_PORT) \
            if cell.tiered else None
        ok = emu.apply_profile_to_loopback(cell.profile, port, udp_port=udp)
        info.update({"tcp_port": port, "udp_port": udp})
    elif is_asymmetric:
        # egress = base + uplink (the upload direction), no ingress
        fields = ("delay_ms", "jitter_ms", "delay_distribution",
                  "delay_correlation_pct", "loss_pct",
                  "loss_correlation_pct", "loss_model", "reorder_pct",
                  "reorder_correlation_pct", "duplicate_pct",
                  "duplicate_correlation_pct", "corruption_pct",
                  "corruption_correlation_pct", "rate_mbit",
                  "rate_ceil_mbit", "rate_burst_kbit", "rate_cburst_kbit",
                  "limit_packets")
        merged = {k: raw[k] for k in fields if k in raw}
        for k, v in (raw.get("uplink") or {}).items():
            if k in fields:
                merged[k] = v
        ok = emu.apply_settings(**merged, disable_ingress=True,
                                profile_name=f"{cell.profile}:uplink")
        info["mode"] = "egress-asymmetric"
    else:
        if cell.profile not in emu.profiles:
            raise SystemExit(f"unknown profile '{cell.profile}'")
        ok = emu.apply_profile(cell.profile)
        info["mode"] = "egress+ingress"
    if not ok:
        raise SystemExit(f"netem apply failed for profile {cell.profile}")
    return info


def args_netem_iface() -> str:
    """Dual-host egress interface (default-route device)."""
    try:
        out = subprocess.run(["ip", "-o", "route", "show", "default"],
                             capture_output=True, text=True,
                             timeout=5).stdout.split()
        for i, tok in enumerate(out):
            if tok == "dev" and i + 1 < len(out):
                return out[i + 1]
    except Exception:
        pass
    raise SystemExit("cannot resolve dual-host netem interface "
                     "(ip route failed); pass a routable --server-host")


def clear_netem(base_url: str) -> None:
    try:
        if is_local(base_url):
            NetworkEmulator(interface="lo").clear_loopback()
        else:
            NetworkEmulator(interface=args_netem_iface()).clear()
    except (Exception, SystemExit) as e:  # noqa: BLE001 — best-effort cleanup
        # SystemExit matters: args_netem_iface() raises it when the
        # default-route lookup fails. From the outer main() finally this
        # must NOT propagate, or _save_manifest + generate_report are
        # skipped — a ~70h run with no final report in dual-host mode.
        print(f"{LOG} warning: netem cleanup failed: {e}")


def bpf_filter(cell: Cell, base_url: str) -> str:
    port = SERIAL_PORT if cell.phase == "serial" else CONC_PORT
    udp = (SERIAL_UDP_PORT if cell.phase == "serial" else CONC_UDP_PORT) \
        if cell.tiered else None
    if is_local(base_url):
        filt = f"tcp port {port}"
        if udp:
            filt += f" or udp port {udp}"
        return filt
    return f"tcp port {port}" + (f" or udp port {udp}" if udp else "")


def capture_interface(base_url: str) -> str:
    if is_local(base_url):
        return "lo"
    return args_netem_iface()


def conc_sidecar_args(args) -> list[str]:
    """Concurrent-sidecar engine flags for auto mode (defaults mirror
    the README's tested concurrent-Qwen3.6 command; the sidecar script
    forwards extra args verbatim, and the model stays the script's own
    default — no --model is passed)."""
    a = ["--mem-fraction-static", str(args.conc_mem_frac),
         "--max-running-requests", str(args.conc_max_running_requests),
         "--server-cpus", args.conc_server_cpus]
    if args.conc_mixed_chunk:
        a.append("--enable-mixed-chunk")
    a += ["--mm-io-worker-num", str(args.conc_mm_io_worker_num),
          "--mm-processor-worker-num", str(args.conc_mm_processor_worker_num)]
    if args.conc_decode_threads > 0:
        a += ["--decode-threads", str(args.conc_decode_threads)]
    if args.conc_preprocess_threads > 0:
        a += ["--preprocess-threads", str(args.conc_preprocess_threads)]
    return a


def sidecar_env(args, phase: str) -> dict[str, str]:
    """GPU-split env for a sidecar process group (empty = inherit)."""
    if args.server_gpu is None:
        return {}
    return {"CUDA_VISIBLE_DEVICES": str(args.server_gpu)}


def _planned_sidecar_lines(args) -> list[str]:
    """Dry-run rendering of the exact sidecar launches auto mode would
    perform (bash script + forwarded flags + GPU env). UDP ports come
    from the SAME constants the real Sidecar.start calls use — a
    port+1 derivation would silently drift if a constant changes."""
    model_path = getattr(args, "sidecar_model_path", None)
    mem_frac = getattr(args, "sidecar_mem_frac", None)

    def _line(script: Path, port: int, udp_port: int,
              extra: list[str]) -> str:
        env = sidecar_env(args, "")
        note = (f"  env: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}"
                if env else "")
        flags = list(extra)
        if model_path:
            flags += ["--model", model_path]
        if mem_frac is not None and port == SERIAL_PORT:
            flags += ["--mem-fraction-static", f"{mem_frac:g}"]
        return (f"bash {script} --port {port} --opt-udp-port {udp_port} "
                + " ".join(flags) + note)
    lines = []
    if args.phases in ("serial", "both"):
        lines.append(_line(SERIAL_SIDECAR_SCRIPT, SERIAL_PORT,
                           SERIAL_UDP_PORT, []))
    if args.phases in ("concurrent", "both"):
        lines.append(_line(CONCURRENT_SIDECAR_SCRIPT, CONC_PORT,
                           CONC_UDP_PORT, conc_sidecar_args(args)))
    return lines


# --- sidecar lifecycle ------------------------------------------------------

class Sidecar:
    """Sequentially-managed sidecar (bootstrap script subprocess)."""

    def __init__(self, script: Path, port: int, log: Path):
        self.script = script
        self.port = port
        self.log = log
        self.proc: subprocess.Popen | None = None

    def start(self, extra_args: list[str] | None = None,
              wait_sec: float = 900,
              env_extra: dict[str, str] | None = None) -> None:
        # self.log — the __init__ parameter `log` is NOT in scope here
        # (a bare `log.parent` would NameError on every real launch;
        # only mkdir needs doing before the Popen below).
        self.log.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["bash", str(self.script), "--port", str(self.port)] \
            + (extra_args or [])
        env_note = ""
        if env_extra:
            env_note = "  env: " + " ".join(f"{k}={v}"
                                            for k, v in env_extra.items())
        print(f"{LOG} starting sidecar: {' '.join(cmd)} {env_note}"
              f"(log: {self.log})")
        lf = open(self.log, "ab")
        try:
            self.proc = subprocess.Popen(cmd, stdout=lf, stderr=lf,
                                         start_new_session=True,
                                         env={**os.environ, **env_extra}
                                         if env_extra else None)
        finally:
            lf.close()  # child owns its fds; the parent handle leaks
        url = f"http://localhost:{self.port}/v1/models"
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_sec:
            if self.proc.poll() is not None:
                raise SystemExit(
                    f"sidecar exited during startup (rc="
                    f"{self.proc.returncode}); tail of {self.log}:\n"
                    + _tail(self.log, 40))
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200 and (r.json().get("data") or []):
                    print(f"{LOG} sidecar ready on :{self.port} "
                          f"({time.monotonic() - t0:.0f}s)")
                    return
            except Exception:
                pass
            time.sleep(5)
        raise SystemExit(f"sidecar not ready within {wait_sec}s on "
                         f":{self.port}; tail:\n" + _tail(self.log, 40))

    def stop(self) -> None:
        if self.proc is None:
            return
        # start_new_session=True made the sidecar a process-GROUP
        # leader; sglang's scheduler subprocesses live in the same
        # group. terminate() on the leader alone would orphan them
        # (holding ~35 GB VRAM so the NEXT sidecar cannot start) —
        # signal the whole group instead.
        try:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # already gone
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=30)
        except Exception as e:  # noqa: BLE001 — best-effort teardown
            print(f"{LOG} warning: sidecar kill failed: {e}")
        self.proc = None
        print(f"{LOG} sidecar :{self.port} stopped")

    def stats(self) -> dict:
        try:
            r = requests.get(
                f"http://localhost:{self.port}/stats", timeout=10)
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}


def _tail(path: Path, n: int) -> str:
    """Last n lines without loading the whole file (sidecar logs can
    grow to hundreds of MB over a multi-day run)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - 65536))
            chunk = f.read()
        return "\n".join(chunk.decode(errors="ignore")
                         .splitlines()[-n:])
    except Exception:
        return "(no log)"


def gpu_free_mb(device_id: int | None = None) -> float | None:
    try:
        q = "memory.free"
        cmd = ["nvidia-smi", f"--query-gpu={q}",
               "--format=csv,noheader,nounits"]
        if device_id is not None:
            cmd[-1:-1] = ["-i", str(device_id)]
        out = subprocess.run(cmd,
                             capture_output=True, text=True,
                             timeout=15).stdout
        vals = [float(x) for x in out.split() if x.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# per-cell execution
# ---------------------------------------------------------------------------

def _kill_process_group(proc: subprocess.Popen, sig: int) -> None:
    """Kill the runner's whole process group (runner + spawned edge
    workers). start_new_session=True in the Popen call guarantees the
    pgid == runner pid; fall back to a direct kill if the group is
    already gone (ESRCH)."""
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass


def run_cell(cell: Cell, args, sidecar_stats_before: dict | None = None,
             dry: bool = False) -> dict:
    base_url = f"http://{args.server_host}:" \
               f"{SERIAL_PORT if cell.phase == 'serial' else CONC_PORT}"
    finalize_cell_cmd(cell, args)

    env = sanitized_env({"ENCPL_DATASET_DIR": args.dataset_dir})
    env.update(cell.cmd_env)
    # GPU split: pin every cell subprocess (the runner + its spawned edge
    # workers, which load the terminal-side ViT tower on cuda) to the
    # CLIENT gpu. Deliberately AFTER sanitized_env so an inherited
    # CUDA_VISIBLE_DEVICES value is overridden, not merged.
    if getattr(args, "client_gpu", None) is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.client_gpu)
    cmd = [sys.executable, cell.runner] + cell.cmd_args
    cmd_str = " ".join([sys.executable, cell.runner]
                       + [shlex.quote(a) for a in cell.cmd_args])

    if dry:
        return {"cell_id": cell.cell_id, "dry": True, "cmd": cmd_str,
                "env_extra": dict(cell.cmd_env)}

    if cell.overlay_config:
        write_overlay_config(cell)

    netem_info = apply_netem(cell, base_url)

    iface = capture_interface(base_url)
    filt = bpf_filter(cell, base_url)
    cap = CaptureController(interface=iface, capture_dir=str(CAPTURE_DIR))
    pcap_t0 = time.time()
    buf_kib = (getattr(args, "pcap_buffer_kib", 262144) or None)
    pcap_file = cap.start(filename=Path(cell.pcap).name, filter_expr=filt,
                          buffer_kib=buf_kib)
    if not pcap_file:
        clear_netem(base_url)
        return {"cell_id": cell.cell_id, "ok": False,
                "error": "tcpdump failed to start", "cmd": cmd_str}

    t0 = time.time()
    # log-open / runner-spawn happen AFTER cap.start() — an OSError here
    # must still stop tcpdump and tear netem down, or the orphaned capture
    # keeps recording every LATER cell's traffic into THIS cell's pcap
    # (and exhausts the disk). BaseException: Ctrl-C cleans up too.
    # Wall-clock cap: a wedged runner (SSE stall, upload hang, engine
    # deadlock...) historically blocked subprocess.call FOREVER and
    # froze the whole matrix (2026-10-18: 29.5h freeze). Popen +
    # start_new_session makes the runner its own process group so a
    # timeout can killpg the runner AND its spawned edge workers;
    # rc is then non-zero and the existing cleanup below tears down
    # tcpdump + netem so the next cell starts clean.
    # getattr-guarded: external drivers building their own args namespace
    # (they pin cell semantics via --profile-record-only style flags) must
    # not crash on matrix-only argparse fields; None = no wall-clock cap.
    cell_timeout = (getattr(args, "serial_cell_timeout", None)
                    if cell.phase == "serial"
                    else getattr(args, "conc_cell_timeout", None))
    try:
        logf = open(cell.log, "wb")
        try:
            proc = subprocess.Popen(
                cmd, stdout=logf, stderr=logf, env=env,
                cwd=str(REPO_ROOT), start_new_session=True)
            timed_out = False
            try:
                rc = proc.wait(timeout=cell_timeout or None)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(proc, signal.SIGKILL)
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    rc = -signal.SIGKILL
                print(f"{LOG}   -> CELL TIMEOUT after {cell_timeout}s: "
                      f"killed process group (marked failed, matrix "
                      f"continues)")
        finally:
            logf.close()
    except BaseException:
        cap.stop()
        clear_netem(base_url)
        raise
    wall = time.time() - t0

    pcap_t1 = time.time()
    pcap_path = cap.stop()
    pcap_size = pcap_path.stat().st_size if pcap_path and pcap_path.exists() else 0
    # tcpdump's own exit summary (frames captured / dropped by kernel /
    # dropped by tcpdump). Capture-path loss is the one failure mode
    # that corrupts pcap byte accounting invisibly — persisted into the
    # manifest so the report can flag untrustworthy UL cells.
    td_stats = cap.stats or {}

    stats_after = _sidecar_stats(base_url)
    opt_delta = _opt_udp_delta(sidecar_stats_before, stats_after)

    clear_netem(base_url)

    result = {"cell_id": cell.cell_id, "ok": rc == 0, "exit_code": rc,
              "wall_sec": round(wall, 1), "cmd": cmd_str,
              "netem": netem_info,
              "pcap": {"path": str(pcap_path), "bytes": pcap_size,
                       "filter": filt, "interface": iface,
                       "duration_sec": round(pcap_t1 - pcap_t0, 1),
                       "tcpdump": td_stats or None},
              "db": cell.db, "log": cell.log,
              "opt_udp_delta": opt_delta}
    if td_stats.get("dropped_by_kernel") or td_stats.get(
            "dropped_by_tcpdump"):
        drops = ((td_stats.get("dropped_by_kernel") or 0)
                 + (td_stats.get("dropped_by_tcpdump") or 0))
        print(f"{LOG}   -> WARNING: tcpdump dropped {drops} packets "
              f"(kernel={td_stats.get('dropped_by_kernel')}, "
              f"tcpdump={td_stats.get('dropped_by_tcpdump')}); "
              f"pcap UL bytes for this cell are UNDERCOUNTS")
    if timed_out:
        result["error"] = (f"cell timeout: killed after {cell_timeout}s "
                           f"wall-clock")
    # Only a completed runner leaves a real DB; logging the capture row
    # into a crashed cell's --db would CREATE a pcap-only DB that
    # pollutes the artifact set (and collides with the retry).
    if rc == 0:
        _log_capture_row(cell, pcap_path, pcap_size, pcap_t0, pcap_t1,
                         filt, iface)
    return result


def _sidecar_stats(base_url: str) -> dict:
    try:
        r = requests.get(f"{base_url}/stats", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _opt_udp_delta(before: dict | None, after: dict) -> dict | None:
    b = (before or {}).get("opt_udp") or {}
    a = after.get("opt_udp") or {}
    if not a:
        return None
    keys = ("fragments_rx", "fragments_bad", "parts_reassembled",
            "parts_incomplete_dropped", "parts_overflow_dropped",
            "parts_stale_dropped")
    return {k: a.get(k, 0) - b.get(k, 0) for k in keys}


def _log_capture_row(cell: Cell, pcap_path, pcap_size: int,
                     t0: float, t1: float, filt: str, iface: str) -> None:
    """Write the turn_index=-2 capture metadata row into the cell's DB
    (same join contract as the concurrent runner's own capture)."""
    try:
        sys.path.insert(0, str(AITESTBED_ROOT))
        from analysis.logger import TrafficLogger, LogRecord
        logger = TrafficLogger(cell.db)
        meta = {"type": "pcap_capture", "pcap_file": str(pcap_path),
                "capture_filter": filt,
                "capture_duration_sec": t1 - t0,
                "interface": iface, "cell_id": cell.cell_id}
        rec = LogRecord(
            timestamp=time.time(),
            scenario_id=f"q36m_{cell.phase}",
            session_id=f"pcap_{int(t0)}", turn_index=-2, run_index=-1,
            provider="tcpdump", model="", request_bytes=0,
            response_bytes=pcap_size, t_request_start=t0,
            latency_sec=t1 - t0, network_profile=cell.profile,
            http_status=200, metadata=json.dumps(meta),
            success=pcap_size > 0)
        logger.log(rec)
    except Exception as e:  # noqa: BLE001 — best effort
        print(f"{LOG} warning: capture row not logged: {e}")


# ---------------------------------------------------------------------------
# report generation
# ---------------------------------------------------------------------------

def _pct(vals: list[float], p: float):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def load_cell_metrics(cell: dict) -> dict:
    """Aggregate one cell's DB: bench accuracy + full per-request stats.

    Collects EVERYTHING both scenarios already record per row (the
    client's engine_stats are flattened into metadata's top level):
    timing percentiles (ttft/e2e/latency), stage attribution
    (stage1_edge/stage2_vit/upload/udp_wave/server_ttft + breakdowns),
    upload efficiency (payload vs wire bytes, throughput), token
    counts, task_type/duration-bucket accuracy, failure taxonomy by
    error_type, and stream shape (chunk_count/inter-chunk gaps).
    """
    import sqlite3
    out = {
        "n": 0, "correct": 0, "parsed": 0, "parse_fail": 0,
        "ttft": [], "lat": [], "e2e_ttft": [], "e2e_total": [],
        "up_bytes": [], "udp_wave": [], "retention_actual": [],
        "opt_tokens": [], "failures": 0,
        # stage attribution (edge pipeline; server rows only carry
        # server_ttft/server_total)
        "stage1": [], "stage2": [], "upload_time": [],
        "server_ttft": [], "server_total": [],
        "stage1_bd": {"open": [], "decode_sample": [], "processor": []},
        "stage2_bd": {"h2d": [], "vit_fwd": [], "assemble": []},
        # bytes / throughput
        "video_bytes": [], "wire_bytes": [], "up_throughput": [],
        "bundle_bytes": [],
        # tokens
        "tokens_in": [], "tokens_out": [], "n_tokens": [],
        # frames
        "sampled_nframes": [], "total_frames": [],
        # stream shape
        "chunk_count": [], "inter_chunk_gap": [],
        # accuracy slices
        "by_task": {}, "by_duration": {},
        # failure taxonomy
        "by_error": {},
        # per-user buckets (concurrent cells; keyed by metadata
        # user_index — serial cells have none so this stays empty)
        "by_user": {},
        # per-request join keys for pcap attribution (concurrent):
        # src_ports from metadata conns 5-tuples + the request time
        # window [t_request_start, t_request_start+latency] — both
        # survive sidecar-stats loss (client-side), giving the pcap
        # join its L1 (src_port) and L2 (time-window) levels.
        "join_rows": [],
        # sidecar-stats health: rows where flattened engine_stats keys
        # were absent entirely (stats event lost / never sent)
        "stats_loss": {"rows": 0, "no_conns": 0},
        # per-question rows for the CSV dump
        "questions": [],
        # full per-request rows for the report JSON: every traffic_logs
        # column (scalars as-is) + parsed metadata dict + parsed
        # inter_chunk_times — the DB's complete content, minus the
        # pcap capture row (covered by run.pcap in the manifest)
        "requests": [],
    }
    db = cell.get("db")
    # Cross-host fallback (same rule as _manifest_pcap_path): the
    # manifest records the RUN host's absolute path (/data01/...); a
    # --report-only re-run on a different host (e.g. the Windows report
    # box over the share) must still find the DB via the OUT_DIR
    # convention path (<cell_id>.db). Without this, report-only on a
    # foreign host SILENTLY produced empty metrics (requests=0,
    # by_user={}) with only pcap columns populated — half a report and
    # no warning.
    db_src = "manifest"
    if db and not Path(db).exists():
        conv = OUT_DIR / f"{cell['cell_id']}.db"
        if conv.exists():
            db = str(conv)
            db_src = "convention"
        else:
            db = None
    if not db:
        print(f"{LOG} WARNING: cell {cell['cell_id']}: DB not found "
              f"(manifest path {cell.get('db')!r} and convention "
              f"{OUT_DIR / (cell['cell_id'] + '.db')} both missing) — "
              "DB-derived metrics will be EMPTY (pcap columns, if any, "
              "are unaffected)")
        return out
    if db_src == "convention":
        print(f"{LOG} note: cell {cell['cell_id']}: manifest DB path "
              f"missing on this host; using convention path {db}")
    conn = None
    try:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM traffic_logs "
            "WHERE session_id NOT LIKE 'pcap_%'").fetchall()
    except sqlite3.Error:
        if conn is not None:
            conn.close()
        print(f"{LOG} WARNING: cell {cell['cell_id']}: DB unreadable "
              f"({db}) — DB-derived metrics will be EMPTY")
        return out
    try:
        for r in rows:
            meta_raw = r["metadata"]
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            lat = r["latency_sec"]
            tokens_out = r["tokens_out"]
            ok = r["success"]
            tokens_in = r["tokens_in"]
            chunk_count = r["chunk_count"]
            inter_chunk_raw = r["inter_chunk_times"]
            error_type = r["error_type"]
            turn_index = r["turn_index"]
            t_req = r["t_request_start"]
            t_last = r["t_last_token"]
            # full-fidelity request row for the report JSON
            try:
                gaps_raw = json.loads(inter_chunk_raw) \
                    if inter_chunk_raw else []
            except (json.JSONDecodeError, TypeError):
                gaps_raw = []
            req = {k: r[k] for k in r.keys() if k != "metadata"}
            req["inter_chunk_times"] = gaps_raw
            req["metadata"] = meta
            out["requests"].append(req)
            bench = meta.get("bench") or {}
            if bench and not bench.get("padded"):
                out["n"] += 1
                if bench.get("correct") is True:
                    out["correct"] += 1
                if bench.get("model_answer"):
                    out["parsed"] += 1
                elif bench.get("model_answer") == "":
                    out["parse_fail"] += 1
                # accuracy slices: task_type + duration bucket
                task = bench.get("task_type") or "unknown"
                t = out["by_task"].setdefault(
                    task, {"n": 0, "correct": 0})
                t["n"] += 1
                if bench.get("correct") is True:
                    t["correct"] += 1
                dur = bench.get("duration")
                if dur is not None:
                    try:
                        d = float(dur)
                        bucket = ("<1m" if d < 60 else
                                  "1-2m" if d < 120 else
                                  "2-4m" if d < 240 else
                                  "4-8m" if d < 480 else ">=8m")
                    except (TypeError, ValueError):
                        bucket = "unknown"
                else:
                    bucket = "unknown"
                b = out["by_duration"].setdefault(
                    bucket, {"n": 0, "correct": 0})
                b["n"] += 1
                if bench.get("correct") is True:
                    b["correct"] += 1
            elif not bench and ok:
                out["n"] += 0  # non-bench rows (per-video, etc.) skipped
            if not ok:
                out["failures"] += 1
                et = (error_type or "unknown")[:80]
                out["by_error"][et] = out["by_error"].get(et, 0) + 1
            ttft = meta.get("ttft_sec")
            if ttft is not None and (tokens_out or 0) > 0:
                out["ttft"].append(float(ttft))
                out["lat"].append(float(lat or 0.0))
            if meta.get("e2e_ttft") is not None and (tokens_out or 0) > 0:
                out["e2e_ttft"].append(float(meta["e2e_ttft"]))
                if meta.get("e2e_total") is not None:
                    out["e2e_total"].append(float(meta["e2e_total"]))
            # engine_stats are FLATTENED into the metadata top level by
            # BOTH scenarios (serial: **engine_stats; concurrent:
            # **(r.get("engine_stats") or {})) — there is no nested
            # "encpl_stats" dict anywhere in the schema.
            if meta.get("uploaded_payload_bytes"):
                out["up_bytes"].append(meta["uploaded_payload_bytes"])
            if meta.get("udp_wave_time"):
                out["udp_wave"].append(meta["udp_wave_time"])
            if meta.get("retention_ratio_actual") is not None:
                out["retention_actual"].append(
                    meta["retention_ratio_actual"])
            if meta.get("opt_tokens"):
                out["opt_tokens"].append(meta["opt_tokens"])
            # stage attribution
            for key, dest in (("stage1_edge", "stage1"),
                              ("stage2_vit", "stage2"),
                              ("upload_time", "upload_time"),
                              ("server_ttft", "server_ttft"),
                              ("server_total", "server_total")):
                v = meta.get(key)
                if v is not None:
                    out[dest].append(float(v))
            bd1 = meta.get("stage1_breakdown") or {}
            for k, lst in out["stage1_bd"].items():
                if bd1.get(k) is not None:
                    lst.append(float(bd1[k]))
            bd2 = meta.get("stage2_breakdown") or {}
            for k, lst in out["stage2_bd"].items():
                if bd2.get(k) is not None:
                    lst.append(float(bd2[k]))
            # bytes / throughput / compression
            if meta.get("video_payload_bytes"):
                out["video_bytes"].append(meta["video_payload_bytes"])
            if meta.get("uploaded_wire_bytes"):
                out["wire_bytes"].append(meta["uploaded_wire_bytes"])
            if meta.get("bundle_bytes"):
                out["bundle_bytes"].append(meta["bundle_bytes"])
            if meta.get("uploaded_payload_bytes") and meta.get("upload_time"):
                if meta["upload_time"] > 0:
                    out["up_throughput"].append(
                        meta["uploaded_payload_bytes"]
                        / meta["upload_time"] / 1048576)
            # tokens / frames
            if tokens_in:
                out["tokens_in"].append(int(tokens_in))
            if tokens_out:
                out["tokens_out"].append(int(tokens_out))
            if meta.get("n_tokens"):
                out["n_tokens"].append(meta["n_tokens"])
            if meta.get("sampled_nframes"):
                out["sampled_nframes"].append(meta["sampled_nframes"])
            if meta.get("total_frames"):
                out["total_frames"].append(meta["total_frames"])
            # stream shape
            if chunk_count:
                out["chunk_count"].append(int(chunk_count))
            try:
                gaps = json.loads(inter_chunk_raw) if inter_chunk_raw else []
            except (json.JSONDecodeError, TypeError):
                gaps = []
            if gaps:
                out["inter_chunk_gap"].extend(
                    float(g) for g in gaps if g is not None and g >= 0)
            # per-user bucket (concurrent rows carry user_index in
            # metadata; serial rows do not -> by_user stays empty)
            u_idx = meta.get("user_index")
            ub = None
            if u_idx is not None:
                try:
                    u_key = int(u_idx)
                except (TypeError, ValueError):
                    u_key = None
                if u_key is not None:
                    ub = out["by_user"].setdefault(u_key, _user_bucket(u_key))
                    seq = meta.get("seq")
                    try:
                        seq = int(seq) if seq is not None else None
                    except (TypeError, ValueError):
                        seq = None
                    if bench and not bench.get("padded"):
                        ub["n"] += 1
                        if bench.get("correct") is True:
                            ub["correct"] += 1
                        if bench.get("model_answer"):
                            ub["parsed"] += 1
                    if not ok:
                        ub["failures"] += 1
                    if ttft is not None and (tokens_out or 0) > 0:
                        ub["ttft"].append(float(ttft))
                        ub["lat"].append(float(lat or 0.0))
                        # first (barrier-aligned, queueing-heavy) vs
                        # steady (playlist continuation) split
                        if seq == 0:
                            ub["ttft_first"].append(float(ttft))
                        elif seq is not None:
                            ub["ttft_steady"].append(float(ttft))
                    if meta.get("e2e_ttft") is not None \
                            and (tokens_out or 0) > 0:
                        ub["e2e_ttft"].append(float(meta["e2e_ttft"]))
                        if meta.get("e2e_total") is not None:
                            ub["e2e_total"].append(float(meta["e2e_total"]))
                    if meta.get("uploaded_payload_bytes"):
                        ub["up_bytes"].append(
                            meta["uploaded_payload_bytes"])
                    if meta.get("udp_wave_time"):
                        ub["udp_wave"].append(meta["udp_wave_time"])
                    for key, dest in (
                            ("stage1_edge", "stage1"),
                            ("stage2_vit", "stage2"),
                            ("upload_time", "upload_time"),
                            ("server_ttft", "server_ttft"),
                            ("server_total", "server_total")):
                        v = meta.get(key)
                        if v is not None:
                            ub[dest].append(float(v))
                    if tokens_in:
                        ub["tokens_in"].append(int(tokens_in))
                    if tokens_out:
                        ub["tokens_out"].append(int(tokens_out))
                    if chunk_count:
                        ub["chunk_count"].append(int(chunk_count))
                    if meta.get("wave_wall_sec") is not None:
                        ub["wall"].append(float(meta["wave_wall_sec"]))
                    # pcap join row: src_ports (L1) + request time
                    # window (L2). Both are client-side facts that
                    # survive sidecar-stats loss on lossy profiles.
                    conns = meta.get("conns") or {}
                    src_ports = []
                    for c in (conns.get("upload"), conns.get("chat")):
                        if isinstance(c, dict):
                            sp = c.get("src_port")
                            if sp is not None:
                                try:
                                    src_ports.append(int(sp))
                                except (TypeError, ValueError):
                                    pass
                    if not src_ports:
                        out["stats_loss"]["no_conns"] += 1
                    t_end = None
                    if t_req is not None:
                        t_end = (t_last if t_last is not None
                                 else (t_req + lat if lat is not None
                                       else None))
                    out["join_rows"].append({
                        "user_index": u_key, "seq": seq,
                        "ok": bool(ok), "src_ports": src_ports,
                        "t_start": t_req, "t_end": t_end,
                    })
                    # per-video detail row (video path/sizes/ports/
                    # timing — the per-user video table and JSON
                    # videos array come from this)
                    up_conn = (conns.get("upload")
                               if isinstance(conns.get("upload"), dict)
                               else {})
                    ch_conn = (conns.get("chat")
                               if isinstance(conns.get("chat"), dict)
                               else {})
                    vid_b = meta.get("video_payload_bytes") or 0
                    ub["videos"].append({
                        "seq": seq,
                        "video_path": meta.get("video_path")
                        or bench.get("video_id"),
                        "video_bytes": int(vid_b),
                        "bundle_bytes": int(
                            meta.get("bundle_bytes") or 0),
                        "up_bytes": int(
                            meta.get("uploaded_payload_bytes") or 0),
                        "up_src_port": up_conn.get("src_port"),
                        "chat_src_port": ch_conn.get("src_port"),
                        "ttft": ttft, "latency": lat,
                        "ok": bool(ok),
                        "task_type": bench.get("task_type") or "",
                    })
                    ub["video_bytes_sum"] += int(vid_b)
            # engine-stats-loss health: rows whose metadata carries
            # NONE of the flattened engine_stats markers (a stats event
            # lost to a dropped stream still keeps conns/timing since
            # those come from the client; total loss = abort rows)
            if ok and not any(meta.get(k) is not None for k in (
                    "uploaded_wire_bytes", "stage1_edge", "stage2_vit",
                    "server_ttft", "conns", "uploaded_payload_bytes")):
                out["stats_loss"]["rows"] += 1
            # per-question CSV row (bench rows only; failures included)
            if bench:
                q_conns = meta.get("conns") or {}
                q_up = (q_conns.get("upload")
                        if isinstance(q_conns.get("upload"), dict) else {})
                q_ch = (q_conns.get("chat")
                        if isinstance(q_conns.get("chat"), dict) else {})
                out["questions"].append({
                    "turn": turn_index,
                    "user_index": (u_idx if u_idx is not None else ""),
                    "seq": (meta.get("seq")
                            if meta.get("seq") is not None else ""),
                    "wave_wall_sec": (round(float(meta["wave_wall_sec"]), 3)
                                      if meta.get("wave_wall_sec") is not None
                                      else ""),
                    "video_id": bench.get("video_id"),
                    "task_type": bench.get("task_type") or "",
                    "domain": bench.get("domain") or "",
                    "duration": bench.get("duration"),
                    "answer": bench.get("answer") or "",
                    "model_answer": bench.get("model_answer") or "",
                    "correct": ("" if bench.get("correct") is None
                                else int(bool(bench.get("correct")))),
                    "success": int(bool(ok)),
                    "error_type": (error_type or "")[:80],
                    "ttft_sec": ttft,
                    "latency_sec": lat,
                    "e2e_ttft": meta.get("e2e_ttft"),
                    "e2e_total": meta.get("e2e_total"),
                    "stage1_edge": meta.get("stage1_edge"),
                    "stage2_vit": meta.get("stage2_vit"),
                    "upload_time": meta.get("upload_time"),
                    "udp_wave_time": meta.get("udp_wave_time"),
                    "server_ttft": meta.get("server_ttft"),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "up_payload_bytes": int(
                        meta.get("uploaded_payload_bytes") or 0),
                    "up_wire_bytes": int(
                        meta.get("uploaded_wire_bytes") or 0),
                    "video_bytes": int(
                        meta.get("video_payload_bytes") or 0),
                    "bundle_bytes": int(
                        meta.get("bundle_bytes") or 0),
                    "up_src_port": (q_up.get("src_port")
                                    if q_up.get("src_port") is not None
                                    else ""),
                    "chat_src_port": (q_ch.get("src_port")
                                      if q_ch.get("src_port") is not None
                                      else ""),
                })
    finally:
        conn.close()
    return out


def _user_bucket(u_key: int) -> dict:
    """Per-user accumulator (concurrent cells; user_index in metadata)."""
    return {
        "n": 0, "correct": 0, "parsed": 0, "failures": 0,
        "ttft": [], "ttft_first": [], "ttft_steady": [], "lat": [],
        "e2e_ttft": [], "e2e_total": [],
        "stage1": [], "stage2": [], "upload_time": [],
        "udp_wave": [], "server_ttft": [], "server_total": [],
        "up_bytes": [], "tokens_in": [], "tokens_out": [],
        "chunk_count": [], "wall": [],
        # per-video detail rows (one per playlist entry): seq, video
        # path, video/bundle/up sizes, per-video src ports, ttft/lat,
        # ok flag — feeds the report's per-user video detail table and
        # the JSON by_user[u].videos array. video_bytes_sum = Σ
        # original video payload bytes.
        "videos": [], "video_bytes_sum": 0,
        # pcap-attributed (filled by _join_pcap_user_flows; None until
        # the join runs — distinguishes "join unavailable" from 0)
        "pcap_ul_bytes": None, "pcap_dl_bytes": None,
        "pcap_retrans": None, "pcap_join": None,
    }


def _join_pcap_user_flows(flows: list[dict], join_rows: list[dict]) -> dict:
    """Two-level per-user attribution of pcap TCP flows.

    L1 src_port: a row's conns.upload/chat src_ports match flow
    endpoints directly (client socket facts — survive stats loss).
    L2 time-window: for rows without src_ports (aborted before/at
    SSE), attribute flows whose [first_ts, last_ts] falls inside the
    row's [t_request_start, t_end] window; overlapping candidate rows
    resolve by largest UL-byte share, conflicts are counted.

    Returns per-user {"ul_bytes","dl_bytes","retrans","join"} plus
    {"unattributed_flows", "ambiguous_flows"} diagnostics. Flows
    claimed by no row stay unattributed (cell totals unaffected).
    """
    by_user: dict[int, dict] = {}
    diag = {"unattributed_flows": 0, "ambiguous_flows": 0}
    claimed: dict[str, int] = {}  # flow key -> user_index
    # L1: direct src_port matching
    for r in join_rows:
        u = r.get("user_index")
        if u is None:
            continue
        b = by_user.setdefault(u, {"ul_bytes": 0, "dl_bytes": 0,
                                   "retrans": 0, "src_ports": set()})
        for sp in r.get("src_ports") or []:
            b["src_ports"].add(int(sp))
    for f in flows:
        owner = None
        for u, b in by_user.items():
            if f["src_port"] in b["src_ports"] \
                    or f["dst_port"] in b["src_ports"]:
                owner = u
                break
        if owner is not None:
            claimed[f["key"]] = owner
    # L2: time-window for rows lacking src_ports
    need_l2 = [r for r in join_rows
               if r.get("user_index") is not None and not r.get("src_ports")
               and r.get("t_start") is not None and r.get("t_end") is not None]
    unclaimed = [f for f in flows if f["key"] not in claimed]
    for f in unclaimed:
        cands = [r for r in need_l2
                 if r["t_start"] - 1.0 <= f["first_ts"]
                 and f["last_ts"] <= r["t_end"] + 1.0]
        if not cands:
            continue
        if len(cands) > 1:
            diag["ambiguous_flows"] += 1
            # largest UL share wins (tie -> first row)
            cands.sort(key=lambda r: (r["t_end"] - r["t_start"]),
                       reverse=True)
        claimed[f["key"]] = cands[0]["user_index"]
    # aggregate
    for u, b in by_user.items():
        b["ul_bytes"] = 0
        b["dl_bytes"] = 0
        b["retrans"] = 0
        b["join"] = "src_port" if b["src_ports"] else "time_window"
    for f in flows:
        u = claimed.get(f["key"])
        if u is None:
            if f["ul_bytes"] or f["dl_bytes"]:
                diag["unattributed_flows"] += 1
            continue
        b = by_user[u]
        b["ul_bytes"] += f["ul_bytes"]
        b["dl_bytes"] += f["dl_bytes"]
        b["retrans"] += f["retrans"]
    for b in by_user.values():
        b.pop("src_ports", None)
    return {**{u: v for u, v in by_user.items()}, **diag}


def _stats_block(vals: list) -> dict:
    """Percentile block for a list of floats (empty -> all None)."""
    if not vals:
        return {"n": 0, "mean": None, "p10": None, "p50": None,
                "p90": None, "p99": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 3),
        "p10": round(_pct(vals, 10), 3),
        "p50": round(_pct(vals, 50), 3),
        "p90": round(_pct(vals, 90), 3),
        "p99": round(_pct(vals, 99), 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
    }


def _csv_dump(cell_id: str, questions: list[dict], csv_dir: Path) -> str:
    """Write one cell's per-question detail as CSV; returns the path
    (empty string when there is nothing to dump or the write fails —
    the CSV is an optional artifact and must never abort the report;
    e.g. a report-only re-run on a host without write access to the
    share's csv dir)."""
    if not questions:
        return ""
    try:
        csv_dir.mkdir(parents=True, exist_ok=True)
        path = csv_dir / f"{cell_id}.csv"
        import csv as _csv
        fields = sorted(questions[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for q in questions:
                w.writerow(q)
        return str(path)
    except OSError as exc:
        print(f"{LOG} WARNING: csv dump failed for {cell_id} "
              f"({type(exc).__name__}: {exc}); report proceeds without it")
        return ""


def generate_report(manifest: dict, out_md: Path, out_json: Path,
                    use_pcap: bool = True) -> None:
    cells = manifest["cells"]
    results = manifest.get("results") or {}
    csv_dir = out_md.parent / "csv"
    pcap_cache_dir = out_md.parent / "pcap_cache"
    rows = []
    for c in cells:
        m = load_cell_metrics(c)
        r = results.get(c["cell_id"], {})
        n, cor = m["n"], m["correct"]
        acc = (cor / n) if n else None
        up_mean = (sum(m["up_bytes"]) / len(m["up_bytes"])
                   if m["up_bytes"] else None)
        vid_mean = (sum(m["video_bytes"]) / len(m["video_bytes"])
                    if m["video_bytes"] else None)
        wire_mean = (sum(m["wire_bytes"]) / len(m["wire_bytes"])
                     if m["wire_bytes"] else None)
        # ---- pcap-derived network stats (packet-level ground truth;
        # independent of sidecar/client byte estimates — lossy-profile
        # cells keep full UL/DL/retrans visibility here) ----
        # Role knowledge for UL/DL direction: the cell's sidecar HTTP
        # port (+ opt-UDP port on tiered cells) IS the server side —
        # no port-order guessing (client ephemeral ports >= 8004 used
        # to flip directions and swap SYN/SYN-ACK sides).
        svr_ports = {SERIAL_PORT if c.get("phase") == "serial"
                     else CONC_PORT}
        if c.get("tiered"):
            svr_ports.add(SERIAL_UDP_PORT if c.get("phase") == "serial"
                          else CONC_UDP_PORT)
        pcap_stats = _pcap_cell_stats(
            _manifest_pcap_path(manifest, c), c["cell_id"],
            pcap_cache_dir, enable=use_pcap, server_ports=svr_ports)
        # surface the run's tcpdump drop summary (manifest run.pcap.tcpdump)
        # next to the packet-level parse so the report can flag cells whose
        # pcap undercounts UL bytes (ring overflow)
        if pcap_stats and "missing" not in pcap_stats:
            td = ((manifest.get("results", {}).get(c["cell_id"])
                   or {}).get("pcap") or {}).get("tcpdump")
            if td:
                pcap_stats["tcpdump"] = td
        pcap_join = None
        if pcap_stats and "_flows" in pcap_stats and m["join_rows"]:
            joined = _join_pcap_user_flows(
                pcap_stats.pop("_flows"), m["join_rows"])
            jdiag = {k: joined[k] for k in
                     ("unattributed_flows", "ambiguous_flows")}
            pcap_join = {int(u): v for u, v in joined.items()
                         if isinstance(u, int)}
            pcap_join.update(jdiag)
        elif pcap_stats:
            pcap_stats.pop("_flows", None)
        # merge pcap-attributed bytes into per-user buckets
        if pcap_join:
            for u, v in pcap_join.items():
                if not isinstance(v, dict) or "ul_bytes" not in v:
                    continue
                ub = m["by_user"].get(int(u))
                if ub is not None:
                    ub["pcap_ul_bytes"] = v["ul_bytes"]
                    ub["pcap_dl_bytes"] = v["dl_bytes"]
                    ub["pcap_retrans"] = v["retrans"]
                    ub["pcap_join"] = v.get("join")
        metrics = {
            "n": n, "correct": cor,
            "parsed": m["parsed"], "parse_fail": m["parse_fail"],
            "accuracy": round(acc, 4) if acc is not None else None,
            "ttft": _stats_block(m["ttft"]),
            "e2e_ttft": _stats_block(m["e2e_ttft"]),
            "e2e_total": _stats_block(m["e2e_total"]),
            "latency": _stats_block(m["lat"]),
            "stages": {
                "stage1_edge": _stats_block(m["stage1"]),
                "stage2_vit": _stats_block(m["stage2"]),
                "upload_time": _stats_block(m["upload_time"]),
                "udp_wave_time": _stats_block(m["udp_wave"]),
                "server_ttft": _stats_block(m["server_ttft"]),
                "server_total": _stats_block(m["server_total"]),
                "stage1_breakdown": {
                    k: _stats_block(v) for k, v in m["stage1_bd"].items()},
                "stage2_breakdown": {
                    k: _stats_block(v) for k, v in m["stage2_bd"].items()},
            },
            "upload": {
                "payload_bytes_mean": round(up_mean) if up_mean is not None
                    else None,
                "wire_bytes_mean": round(wire_mean) if wire_mean is not None
                    else None,
                "video_bytes_mean": round(vid_mean) if vid_mean is not None
                    else None,
                "throughput_mb_s": _stats_block(m["up_throughput"]),
                # compression ratio vs the server arm's own full-video
                # upload (per-cell; cross-cell ratio computed in MD)
                "ratio_vs_video": round(up_mean / vid_mean, 4)
                    if (up_mean is not None and vid_mean) else None,
                "bundle_bytes_mean": round(
                    sum(m["bundle_bytes"]) / len(m["bundle_bytes"]))
                    if m["bundle_bytes"] else None,
            },
            "tokens": {
                "in_mean": round(sum(m["tokens_in"]) / len(m["tokens_in"]), 1)
                    if m["tokens_in"] else None,
                "out_mean": round(
                    sum(m["tokens_out"]) / len(m["tokens_out"]), 1)
                    if m["tokens_out"] else None,
                "video_tokens": _stats_block(m["n_tokens"]),
            },
            "frames": {
                "sampled_mean": round(
                    sum(m["sampled_nframes"]) / len(m["sampled_nframes"]), 1)
                    if m["sampled_nframes"] else None,
                "total_mean": round(
                    sum(m["total_frames"]) / len(m["total_frames"]), 1)
                    if m["total_frames"] else None,
            },
            "stream": {
                "chunk_count": _stats_block(
                    [float(x) for x in m["chunk_count"]]),
                "inter_chunk_gap_ms": _stats_block(
                    [g * 1000 for g in m["inter_chunk_gap"]]),
            },
            "retention_actual": round(
                sum(m["retention_actual"]) / len(m["retention_actual"]), 4)
                if m["retention_actual"] else None,
            "opt_tokens": _stats_block(
                [float(x) for x in m["opt_tokens"]]),
            "failures": m["failures"],
            "by_error": dict(sorted(m["by_error"].items(),
                                    key=lambda kv: -kv[1])),
            "by_task": {
                k: {"n": v["n"],
                    "accuracy": round(v["correct"] / v["n"], 4)
                    if v["n"] else None}
                for k, v in sorted(m["by_task"].items())},
            "by_duration": {
                k: {"n": v["n"],
                    "accuracy": round(v["correct"] / v["n"], 4)
                    if v["n"] else None}
                for k, v in sorted(m["by_duration"].items())},
            # per-user blocks (concurrent cells only; {} for serial)
            "by_user": {
                int(u): {
                    "n": v["n"], "correct": v["correct"],
                    "accuracy": round(v["correct"] / v["n"], 4)
                        if v["n"] else None,
                    "failures": v["failures"],
                    "ttft": _stats_block(v["ttft"]),
                    "ttft_first": _stats_block(v["ttft_first"]),
                    "ttft_steady": _stats_block(v["ttft_steady"]),
                    "e2e_ttft": _stats_block(v["e2e_ttft"]),
                    "e2e_total": _stats_block(v["e2e_total"]),
                    "stages": {
                        "stage1_edge": _stats_block(v["stage1"]),
                        "stage2_vit": _stats_block(v["stage2"]),
                        "upload_time": _stats_block(v["upload_time"]),
                        "udp_wave_time": _stats_block(v["udp_wave"]),
                        "server_ttft": _stats_block(v["server_ttft"]),
                        "server_total": _stats_block(v["server_total"]),
                    },
                    "upload_bytes_mean": round(
                        sum(v["up_bytes"]) / len(v["up_bytes"]))
                        if v["up_bytes"] else None,
                    "tokens_in_mean": round(
                        sum(v["tokens_in"]) / len(v["tokens_in"]), 1)
                        if v["tokens_in"] else None,
                    "tokens_out_mean": round(
                        sum(v["tokens_out"]) / len(v["tokens_out"]), 1)
                        if v["tokens_out"] else None,
                    "wave_wall_mean": round(
                        sum(v["wall"]) / len(v["wall"]), 3)
                        if v["wall"] else None,
                    # per-video detail: one row per playlist entry
                    # (path/sizes/ports/timing) + summed original video
                    # bytes + the user's full src-port set
                    "videos": v["videos"],
                    "video_bytes_sum": v["video_bytes_sum"],
                    "src_ports": sorted({
                        p for vid in v["videos"]
                        for p in (vid.get("up_src_port"),
                                  vid.get("chat_src_port"))
                        if p is not None}),
                    # pcap-attributed (None = join unavailable:
                    # pcap missing / no flows matched this user)
                    "pcap_ul_bytes": v["pcap_ul_bytes"],
                    "pcap_dl_bytes": v["pcap_dl_bytes"],
                    "pcap_retrans": v["pcap_retrans"],
                    "pcap_join": v["pcap_join"],
                }
                for u, v in sorted(m["by_user"].items())},
            # pcap-derived network ground truth (packet-level; survives
            # sidecar-stats loss on lossy profiles). "missing" key set
            # when the pcap is absent/unparseable — all pcap columns
            # render "-" then. src/db est columns stay available.
            "pcap": (pcap_stats or None),
            "pcap_user_join": (pcap_join or None),
            "stats_loss": m["stats_loss"],
        }
        csv_path = _csv_dump(c["cell_id"], m["questions"], csv_dir)
        rows.append({
            **c,
            "metrics": metrics,
            # full per-request rows: every traffic_logs column + parsed
            # metadata (conns/bench/engine_stats/video facts) + parsed
            # inter_chunk_times — the DB's complete content in the JSON
            # report (pcap capture row excluded; run.pcap covers it)
            "requests": m["requests"],
            "csv": csv_path or None,
            "run": {k: v for k, v in r.items()
                    if k in ("ok", "exit_code", "wall_sec", "pcap",
                             "opt_udp_delta", "error")},
        })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"manifest": manifest.get("meta", {}),
                   # machine-readable provenance: which fields are DB
                   # estimates vs pcap packet-level ground truth, and
                   # their aggregation scopes (mirrors the MD legend).
                   "legend": {
                       "db_estimate": {
                           "desc": "client/server stats recorded in each "
                                   "cell's SQLite traffic_logs "
                                   "(engine_stats / scenario timings); "
                                   "may be missing on lossy profiles",
                            "fields": ["requests", "csv", "metrics."
                                       "accuracy|ttft|e2e_ttft|latency|"
                                       "stages|tokens|upload|by_user."
                                       "upload_bytes_mean|by_task|by_error|"
                                       "by_duration|dispersion|questions"],
                            "scope": "per-request (mean/percentile) unless "
                                     "noted per-user",
                        },
                        "pcap_ground_truth": {
                            "desc": "packet-level parse of the loopback "
                                    "tcpdump capture; independent of DB "
                                    "(survives request failure)",
                            "fields": ["metrics.pcap.*",
                                       "metrics.by_user.pcap_ul_bytes|"
                                       "pcap_dl_bytes|pcap_retrans|"
                                       "pcap_join"],
                            "scope": "cell total (TCP+UDP, full-frame "
                                     "IP-layer bytes incl. ACK/SYN/FIN/"
                                     "retrans; Ethernet header excluded); "
                                     "by_user pcap "
                                    "fields are per-user totals joined by "
                                    "src_port (time_window fallback "
                                    "marked pcap_join)",
                       },
                   },
                   "cells": rows}, f, indent=2, ensure_ascii=False)

    meta = manifest.get("meta", {})
    model_name = Path(meta.get("sidecar_model") or "").name or "默认(FP8)"
    mem_frac = meta.get("sidecar_mem_frac")
    engine_line = f"- 引擎: {model_name}"
    if mem_frac is not None:
        engine_line += f" serial-mem-frac={mem_frac:g}"
    lines = ["# Qwen3.6 矩阵测试总报告",
             "",
             f"- 运行时间: {manifest.get('meta', {}).get('started', '')}",
             f"- 单元格数: {len(cells)}",
             f"- 侧车: {manifest.get('meta', {}).get('sidecar_mode', '')}",
             f"- 数据集: {manifest.get('meta', {}).get('dataset_dir', '')}",
             engine_line,
             f"- 明细 CSV: {'aitestbed/results/q36m/csv/' if rows else ''}"
             f"（每格一文件，含逐问计时/答案/字节；**全部列为 DB 记录"
             f"（客户端/服务端统计），无 pcap 数据**）",
             "",
             "## 数据来源与口径说明",
             "",
             "本报告每个统计量来自以下两种数据源之一：",
             "",
             "- **DB 估算**：写进各格 SQLite `traffic_logs` 的客户端/"
             "服务端统计（`engine_stats`、scenario 计时）。含全部计时列"
             "（ttft/e2e/lat/stage 分解）、token 数、`up B(db)`、"
             "`ratio`、准确率、CSV 与 per-user 视频明细。丢包场景可能"
             "缺失（请求失败则整行无统计）。",
             "- **pcap 实测**：对 loopback tcpdump 抓包逐包解析"
             "（“网络流量实测”章节；主表的 `UL/DL B(pcap)`、`retr%`、"
             "`rtt`、`drops` 列）。与 DB 完全独立，不受请求成败影响。",
             "",
             "口径（聚合层级）：",
             "",
             "- `up B(db)` / `tok in/out` / 各计时列：**每问（per-request）"
             "均值或分位**。",
             "- `UL/DL B(pcap)`：**整格（cell）TCP+UDP 总字节**（全帧 IP 层"
             "口径——含 ACK/SYN/FIN/重传、TCP/IP 头、握手开销, 不含以太"
             "网头）——与每问均值口径不同，不能直接对除。",
             "- per-user 表的 `up B(db)` 为该用户**每问均值**，"
             "`UL B(pcap)` 为该用户**全部请求合计**（按 src_port 归属，"
             "带 `*` 为时间窗兜底归属）。串行格无 per-user pcap 归属"
             "（单用户串行，只有整格 pcap 总量）。",
             "",
             "两源对照意义：DB 估的是“应用认为发了多少”，pcap 量的是"
             "“线上实际跑了多少字节”（差值 ≈ 重传+协议开销；丢包场景"
             "DB 可能漏记而 pcap 不漏——除非 tcpdump 抓包环溢出, 见"
             " drops 列）。",
             ""]

    def _fmt_p(st: dict, keys=("p50", "p90")) -> str:
        if not st or not st.get("p50") and not st.get("p90"):
            return "-"
        return "/".join(str(st.get(k) if st.get(k) is not None else "-")
                        for k in keys)

    def _fmt_b(v) -> str:
        """Byte counts with thousands separators (report md readability);
        JSON keeps raw integers."""
        return f"{int(v):,}" if v is not None else "-"

    def _fmt_drops(td: dict | None) -> str:
        """tcpdump capture-drop summary; '-' = stats unavailable (old
        manifests / tcpdump died before printing)."""
        if not td:
            return "-"
        kb = td.get("dropped_by_kernel") or 0
        bt = td.get("dropped_by_tcpdump") or 0
        if not kb and not bt:
            return "0"
        return f"krn {kb:,}/tcp {bt:,}"

    def _table(title: str, subset: list[dict]):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| cell | profile | N | v | r | opt | quant | acc | "
                     "ttft p50/p90 | e2e p50/p90 | lat p50/p90 | "
                     "s1/s2/up/srv | up B(db) | ratio | tok in/out | "
                     "UL B(pcap) | DL B | retr% | rtt p95 | drops |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|"
                     "---|---|---|---|---|---|---|---|")
        for row in subset:
            mm, rr = row["metrics"], row.get("run") or {}
            pc = mm.get("pcap") or {}
            acc = mm["accuracy"]
            st = mm["stages"]
            stages = ("/".join(
                str(st[k]["p50"]) for k in ("stage1_edge", "stage2_vit",
                                            "upload_time", "server_ttft")
                if st[k] and st[k].get("p50") is not None)) or "-"
            up = mm["upload"]
            tk = mm["tokens"]
            if "missing" in pc:
                ul = dl = retr = rtt = "-"
            else:
                ul = _fmt_b(pc.get("ul_bytes"))
                dl = _fmt_b(pc.get("dl_bytes"))
                rr_ = pc.get("retr_rate")
                retr = (f"{rr_:.1%}" if rr_ is not None else "-")
                rtt = (pc.get("rtt_ms") or {}).get("p95")
                rtt = rtt if rtt is not None else "-"
            drops = _fmt_drops((mm.get("pcap") or {}).get("tcpdump"))
            lines.append(
                f"| {row['cell_id']} | {row['profile']} | {row['users']} "
                f"| {row.get('videos_per_user') or '-'} "
                f"| {row['retention'] or '-'} | {row['optional'] or '-'} "
                f"| {row['quant'] or '-'} "
                f"| {f'{acc:.1%}' if acc is not None else '-'} "
                f"| {_fmt_p(mm['ttft'])} "
                f"| {_fmt_p(mm['e2e_ttft'])} "
                f"| {_fmt_p(mm['latency'])} "
                f"| {stages} "
                f"| {up['payload_bytes_mean'] if up['payload_bytes_mean'] is not None else '-'} "
                f"| {up['ratio_vs_video'] or '-'} "
                f"| {tk['in_mean'] if tk['in_mean'] is not None else '-'}"
                f"/{tk['out_mean'] if tk['out_mean'] is not None else '-'} "
                f"| {ul} | {dl} | {retr} | {rtt} | {drops} |")
        lines.append("")
        lines.append(
            "- 来源: ttft/e2e/lat/stages/`up B(db)`/`ratio`/`tok` = DB 估算"
            " (engine_stats/客户端统计; 丢包场景可能缺失); `UL/DL B(pcap)`/"
            "`retr%`/`rtt`/`drops` = pcap 逐包实测, 与估算独立。")
        lines.append(
            "- 口径: `up B(db)` 为**每问均值**, `UL/DL B(pcap)` 为**整格"
            " TCP+UDP 总量**（全帧 IP 层字节, 含 ACK/SYN/FIN/重传, 不含"
            " 以太网头）, 两者不可直接对除; 串行格"
            " pcap 仅整格统计（无 per-user 归属），并发格另见"
            " per-user 表。")

    serial = [r for r in rows if r["phase"] == "serial"]
    conc = [r for r in rows if r["phase"] == "concurrent"]
    lines.append(f"## 串行臂 ({len(serial)} cells)")
    lines.append("")
    _table("串行 — 基线 (server / edge-plain)", [r for r in serial
            if r["arm"] in ("server", "edge-plain")])
    _table("串行 — 单层 retention 阶梯",
           [r for r in serial if r["tiered"] is False
            and r["retention"]])
    _table("串行 — 两层 (optional) 阶梯",
           [r for r in serial if r["tiered"]])
    _table("并发臂 (全部)", conc)

    # ---- serial stage breakdown (edge pipeline attribution) ----
    edge_rows = [r for r in rows if r["mode"] == "edge"]
    if edge_rows:
        lines.append("## 边缘管线阶段分解 (串行+并发 edge 格, p50 / 拆分 p50)")
        lines.append("")
        lines.append("| cell | s1 (open+dec+proc) | s2 (h2d+vit+asm) | "
                     "upload | udp_wave | server_ttft | edge_prep |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in edge_rows:
            st = row["metrics"]["stages"]

            def _p50(block: dict) -> str:
                v = block.get("p50") if block else None
                return str(v) if v is not None else "-"

            def _bd(bd: dict, keys) -> str:
                parts = [f"{k[:3]}={bd[k]['p50']}"
                         for k in keys
                         if bd.get(k) and bd[k].get("p50") is not None]
                return "/".join(parts) if parts else "-"
            lines.append(
                f"| {row['cell_id']} "
                f"| {_p50(st['stage1_edge'])}"
                f" ({_bd(st['stage1_breakdown'], ('open', 'decode_sample', 'processor'))}) "
                f"| {_p50(st['stage2_vit'])}"
                f" ({_bd(st['stage2_breakdown'], ('h2d', 'vit_fwd', 'assemble'))}) "
                f"| {_p50(st['upload_time'])} "
                f"| {_p50(st['udp_wave_time'])} "
                f"| {_p50(st['server_ttft'])} "
                f"| {_p50(st['server_total'])} |")
        lines.append("")

    # ---- upload efficiency ----
    up_rows = [r for r in rows
               if r["metrics"]["upload"]["payload_bytes_mean"] is not None]
    if up_rows:
        lines.append("## 上传效率与压缩 (每问均值)")
        lines.append("")
        lines.append("| cell | video B | bundle B | up payload B | "
                     "up wire B | 吞吐 MB/s p50 | 压缩比 vs video |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in up_rows:
            up = row["metrics"]["upload"]
            tp = up["throughput_mb_s"]
            lines.append(
                f"| {row['cell_id']} "
                f"| {_fmt_b(up['video_bytes_mean'])} "
                f"| {_fmt_b(up['bundle_bytes_mean'])} "
                f"| {_fmt_b(up['payload_bytes_mean'])} "
                f"| {_fmt_b(up['wire_bytes_mean'])} "
                f"| {tp.get('p50') if tp.get('p50') is not None else '-'} "
                f"| {up['ratio_vs_video'] or '-'} |")
        lines.append("")
        lines.append(
            "- 本表全部为 DB 估算（客户端 engine_stats / 服务端 on_wire "
            "计算），**无 pcap 数据**: `up wire B` 为服务端估算的线上"
            "字节（含打包开销，不含 TCP/IP 头与重传），线上实测见"
            "“网络流量实测”章节")

    # ---- accuracy slices ----
    task_rows = [r for r in rows if r["metrics"]["by_task"]]
    if task_rows:
        all_tasks = sorted({t for r in task_rows
                            for t in r["metrics"]["by_task"]})
        lines.append("## 按 task_type 准确率")
        lines.append("")
        header = "| cell | " + " | ".join(all_tasks) + " | N |"
        lines.append(header)
        lines.append("|---" * (len(all_tasks) + 2) + "|")
        for row in task_rows:
            bt = row["metrics"]["by_task"]
            cells_str = " | ".join(
                f"{bt[t]['accuracy']:.0%}({bt[t]['n']})"
                if t in bt and bt[t]["accuracy"] is not None else "-"
                for t in all_tasks)
            lines.append(f"| {row['cell_id']} | {cells_str} "
                         f"| {row['metrics']['n']} |")
        lines.append("")

    dur_rows = [r for r in rows if r["metrics"]["by_duration"]]
    if dur_rows:
        order = ["<1m", "1-2m", "2-4m", "4-8m", ">=8m", "unknown"]
        all_durs = [d for d in order if any(
            d in r["metrics"]["by_duration"] for r in dur_rows)]
        lines.append("## 按视频时长桶准确率")
        lines.append("")
        header = "| cell | " + " | ".join(all_durs) + " |"
        lines.append(header)
        lines.append("|---" * (len(all_durs) + 1) + "|")
        for row in dur_rows:
            bd = row["metrics"]["by_duration"]
            cells_str = " | ".join(
                f"{bd[d]['accuracy']:.0%}({bd[d]['n']})"
                if d in bd and bd[d]["accuracy"] is not None else "-"
                for d in all_durs)
            lines.append(f"| {row['cell_id']} | {cells_str} |")
        lines.append("")

    # ---- failure taxonomy ----
    err_rows = [r for r in rows if r["metrics"]["failures"]
                or r["metrics"]["by_error"]]
    if err_rows:
        lines.append("## 失败明细 (error 分类)")
        lines.append("")
        lines.append("| cell | failures | 分类 (类型×次数) | engine_stats 缺失 |")
        lines.append("|---|---|---|---|")
        for row in err_rows:
            mm = row["metrics"]
            errs = ", ".join(f"{k[:60]}×{v}"
                             for k, v in mm["by_error"].items()) or "-"
            sl = mm.get("stats_loss") or {}
            sl_str = "-"
            if sl.get("rows") or sl.get("no_conns"):
                sl_str = (f"{sl.get('rows', 0)} 行无 stats"
                          f" / {sl.get('no_conns', 0)} 行无 conns")
            lines.append(f"| {row['cell_id']} | {mm['failures']} "
                         f"| {errs} | {sl_str} |")
        lines.append("")

    # ---- per-user statistics (concurrent cells; metadata user_index) ----
    # full-column table per the user's request: every user is a row,
    # all key metrics are columns. Fairness summary (cross-user ttft
    # p50 max/min ratio + CV) follows each cell's table.
    conc_user_rows = [r for r in conc if r["metrics"].get("by_user")]
    if conc_user_rows:
        lines.append("## 并发 per-user 统计")
        lines.append("")
        for row in conc_user_rows:
            bu = row["metrics"]["by_user"]
            lines.append(f"### {row['cell_id']}")
            lines.append("")
            lines.append(
                "- 列来源: 计时/stages/`up B(db)`/`tok`/`wall`/`video B` = "
                "DB 估算; `UL B(pcap)`/`retr` = pcap 实测按 src_port "
                "归属到该用户（`*` = 时间窗兜底）。口径: `up B(db)` 为该"
                "用户**每问均值**, `UL B(pcap)` 为该用户**全部请求合计**"
                "（全帧 IP 层字节, 含 ACK/SYN/FIN/重传, 不含以太网头）"
                "，不可直接对除。")
            lines.append("| u | n_ok/n | acc | ttft p50/p90 | "
                         "first ttft | steady ttft | e2e p50/p90 | "
                         "up B(db) | UL B(pcap) | retr | s1/s2/up/srv | "
                         "tok in/out | wall | fail | videos | video B | "
                         "ports |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|"
                         "---|---|---|---|---|---|")
            for u, v in sorted(bu.items()):
                acc = v["accuracy"]
                st = v["stages"]
                stages = ("/".join(
                    str(st[k]["p50"])
                    for k in ("stage1_edge", "stage2_vit",
                              "upload_time", "server_ttft")
                    if st[k] and st[k].get("p50") is not None)) or "-"
                def _fp(block: dict, keys=("p50", "p90")) -> str:
                    if not block or (block.get("p50") is None
                                     and block.get("p90") is None):
                        return "-"
                    return "/".join(
                        str(block.get(k) if block.get(k) is not None
                            else "-") for k in keys)
                ul_p = v.get("pcap_ul_bytes")
                ul_p = (f"{_fmt_b(ul_p)}{'*' if v.get('pcap_join') == 'time_window' else ''}"
                        if ul_p is not None else "-")
                retr_p = v.get("pcap_retrans")
                retr_p = str(retr_p) if retr_p is not None else "-"
                vids = v.get("videos") or []
                vid_names = ", ".join(
                    str((vd.get("video_path") or "").rsplit("/", 1)[-1])
                    for vd in vids) or "-"
                vid_b = _fmt_b(v.get("video_bytes_sum"))
                ports_n = len(v.get("src_ports") or [])
                lines.append(
                    f"| {u} | {v['n'] - v['failures']}/{v['n']} "
                    f"| {f'{acc:.0%}' if acc is not None else '-'} "
                    f"| {_fp(v['ttft'])} "
                    f"| {_fp(v['ttft_first'], ('p50',))} "
                    f"| {_fp(v['ttft_steady'], ('p50',))} "
                    f"| {_fp(v['e2e_ttft'])} "
                    f"| {_fmt_b(v['upload_bytes_mean'])} "
                    f"| {ul_p} "
                    f"| {retr_p} "
                    f"| {stages} "
                    f"| {v['tokens_in_mean'] if v['tokens_in_mean'] is not None else '-'}"
                    f"/{v['tokens_out_mean'] if v['tokens_out_mean'] is not None else '-'} "
                    f"| {v['wave_wall_mean'] if v['wave_wall_mean'] is not None else '-'} "
                    f"| {v['failures']} "
                    f"| {len(vids)} ({vid_names}) "
                    f"| {vid_b} "
                    f"| {ports_n} |")
            # per-user video detail: one row per playlist entry with
            # the FULL video path and per-video src ports (each video
            # opens fresh connections — the ports are the pcap<->user
            # join keys)
            if any((v.get("videos") or []) for v in bu.values()):
                lines.append("")
                lines.append("#### per-user 视频明细")
                lines.append("")
                lines.append(
                    "- 本表全部为 DB 记录（客户端统计）: `up B` 为该问"
                    "上传 payload（估算，不含协议开销）; `up端口`/`chat端口`"
                    " 为该问的连接 src_port —— pcap 侧即以此端口把该问"
                    "流量归属到用户。")
                lines.append("| u | seq | video | video B | bundle B | "
                             "up B | up端口 | chat端口 | ttft s | "
                             "task | ok |")
                lines.append("|---|---|---|---|---|---|---|---|---|---|"
                             "---|")
                for u, v in sorted(bu.items()):
                    for vd in (v.get("videos") or []):
                        lines.append(
                            f"| {u} | {vd.get('seq')} "
                            f"| {vd.get('video_path') or '-'} "
                            f"| {_fmt_b(vd.get('video_bytes'))} "
                            f"| {_fmt_b(vd.get('bundle_bytes'))} "
                            f"| {_fmt_b(vd.get('up_bytes'))} "
                            f"| {vd.get('up_src_port') or '-'} "
                            f"| {vd.get('chat_src_port') or '-'} "
                            f"| {vd.get('ttft') if vd.get('ttft') is not None else '-'} "
                            f"| {vd.get('task_type') or '-'} "
                            f"| {int(bool(vd.get('ok')))} |")
            # fairness summary: cross-user ttft p50 spread
            p50s = [v["ttft"]["p50"] for v in bu.values()
                    if v["ttft"].get("p50") is not None]
            if len(p50s) >= 2:
                mx, mn = max(p50s), min(p50s)
                mean = sum(p50s) / len(p50s)
                cv = (sum((x - mean) ** 2 for x in p50s)
                      / len(p50s)) ** 0.5 / mean if mean else None
                lines.append("")
                lines.append(
                    f"- 跨用户 ttft p50: min={mn} max={mx} "
                    f"max/min={mx / mn:.2f}"
                    + (f" CV={cv:.2f}" if cv is not None else ""))
            # pcap join coverage footnote
            pj = row["metrics"].get("pcap_user_join")
            if pj:
                ua = pj.get("unattributed_flows", 0)
                am = pj.get("ambiguous_flows", 0)
                tw = sum(1 for v in bu.values()
                         if v.get("pcap_join") == "time_window")
                lines.append(
                    f"- pcap 归属: src_port 为主, 时间窗兜底 ({tw} 用户)"
                    f"; 未归属 flow {ua}, 歧义 {am}"
                    " (未归属流量仍计入 cell 总量)")
            elif (row["metrics"].get("pcap") or {}).get("missing"):
                lines.append(
                    f"- pcap 归属: 不可用 ({row['metrics']['pcap']['missing']})")
            lines.append("")

    # ---- pcap ground-truth network stats (packet-level) ----
    pcap_rows = [r for r in rows if r["metrics"].get("pcap")]
    if pcap_rows:
        lines.append("## 网络流量实测 (pcap 逐包)")
        lines.append("")
        lines.append("| cell | UL B | DL B | tcp ul/dl B | udp ul/dl B | "
                     "retrans | retr% | rtt p50/p95 | flows | pkts | "
                     "span s | drops |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in pcap_rows:
            pc = row["metrics"]["pcap"]
            if pc.get("missing"):
                lines.append(
                    f"| {row['cell_id']} | - | - | - | - | - | - | - | - "
                    f"| - | - | ({pc['missing']}) | - |")
                continue
            rtt = pc.get("rtt_ms") or {}
            rtt_s = ("/".join(str(rtt.get(k) if rtt.get(k) is not None
                                    else "-")
                              for k in ("p50", "p95"))
                     if rtt else "-")
            rr_ = pc.get("retr_rate")
            lines.append(
                f"| {row['cell_id']} | {_fmt_b(pc.get('ul_bytes'))} "
                f"| {_fmt_b(pc.get('dl_bytes'))} "
                f"| {_fmt_b(pc.get('tcp_ul_bytes'))}/"
                f"{_fmt_b(pc.get('tcp_dl_bytes'))} "
                f"| {_fmt_b(pc.get('udp_ul_bytes'))}/"
                f"{_fmt_b(pc.get('udp_dl_bytes'))} "
                f"| {pc.get('retrans', '-')} "
                f"| {f'{rr_:.1%}' if rr_ is not None else '-'} "
                f"| {rtt_s} "
                f"| {pc.get('flows', '-')} "
                f"| {pc.get('n_packets', '-')} "
                f"| {pc.get('t_span', '-')} "
                f"| {_fmt_drops(pc.get('tcpdump'))} |")
        lines.append("")
        lines.append(
            "- 全表为 pcap 逐包实测（cell 总量口径, TCP+UDP, 全帧 IP 层"
            "字节——含 ACK/SYN/FIN/重传与协议头, 不含以太网头; 与"
            " Wireshark Conversations 的 Bytes 差 14B×帧数）; 与 DB 估算"
            " 独立 (丢包场景 DB 侧 upload 统计可能缺失, pcap 不受影响); "
            "丢包重传在 retr 列可见。")
        lines.append(
            "- drops 列 = tcpdump 抓包环溢出丢的帧数（krn=内核环/`-B` 不足,"
            " tcp=tcpdump 用户态溢出; `-`=旧 manifest 无此数据）。"
            " drops>0 时该格 UL 数字系统性偏低, 以 DB upload 列为准。")

    # ---- concurrent queueing spread (question-level e2e_ttft) ----
    # question rows pooled; the per-user table above carries the
    # per-user view (user_index/seq from metadata).
    if conc:
        lines.append("## 并发排队离散度 (e2e_ttft 分布)")
        lines.append("")
        lines.append("- 来源: DB 估算（每问 e2e_ttft 的分位，无 pcap 数据）")
        lines.append("| cell | p10 | p50 | p90 | p99 | max | "
                     "(p90-p50)/p50 |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in conc:
            e = row["metrics"]["e2e_ttft"]
            if not e or e.get("p50") is None:
                continue
            spread = ((e["p90"] - e["p50"]) / e["p50"]
                      if (e.get("p90") is not None
                          and e.get("p50")) else None)
            lines.append(
                f"| {row['cell_id']} | {e.get('p10', '-')} "
                f"| {e['p50']} | {e.get('p90', '-')} "
                f"| {e.get('p99', '-')} | {e.get('max', '-')} "
                f"| {f'{spread:.2f}' if spread is not None else '-'} |")
        lines.append("")

    # two-tier vs single-tier comparison
    lines.append("## 两层 vs 单层同 retention 对照 (串行)")
    lines.append("")
    lines.append("| profile | quant | r | opt | 单层 acc | 两层 acc | Δ | "
                 "Δttft p50 | udp_wave | opt parts 丢弃 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in [r for r in serial if r["tiered"]]:
        key = (row["profile"], row["quant"] or "", row["retention"])
        single = next((r for r in serial if not r["tiered"]
                       and (r["profile"], r["quant"] or "",
                            r["retention"]) == key), None)
        if not single:
            continue
        sa = single["metrics"]["accuracy"]
        ta = row["metrics"]["accuracy"]
        delta = (ta - sa) if (sa is not None and ta is not None) else None
        st_t = single["metrics"]["ttft"].get("p50")
        tt_t = row["metrics"]["ttft"].get("p50")
        dttft = (tt_t - st_t) if (st_t is not None
                                  and tt_t is not None) else None
        uw = row["metrics"]["stages"]["udp_wave_time"].get("p50")
        drops = (row.get("run", {}).get("opt_udp_delta") or {}).get(
            "parts_incomplete_dropped")
        lines.append(
            f"| {row['profile']} | {row['quant'] or '-'} "
            f"| {row['retention']} | {row['optional']} "
            f"| {f'{sa:.1%}' if sa is not None else '-'} "
            f"| {f'{ta:.1%}' if ta is not None else '-'} "
            f"| {f'{delta:+.1%}' if delta is not None else '-'} "
            f"| {f'{dttft:+.2f}s' if dttft is not None else '-'} "
            f"| {uw if uw is not None else '-'} "
            f"| {drops if drops is not None else '-'} |")
    lines.append("")

    failed = [r for r in rows if (r.get("run") or {}).get("ok") is False]
    if failed:
        lines.append("## 失败单元格")
        lines.append("")
        for r in failed:
            lines.append(f"- {r['cell_id']}: exit="
                         f"{r['run'].get('exit_code')} "
                         f"{r['run'].get('error') or ''}")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"{LOG} report written: {out_md} (+ {out_json})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _fp8_arch_problems(args) -> list[str]:
    """FP8-e4m3 checkpoint 在 compute capability < 8.9（Ampere 及更早）
    上无法 JIT：Triton fused-MoE 会在引擎 warmup 期抛
    "type fp8e4nv not supported"，调度子进程崩溃 → 整个 sidecar 进程组
    被 SIGKILL（rc=-9，等满 sidecar_wait 才看到）。在这里 fail-fast。
    （与旧标定驱动同语义；其已下线，此处为唯一实现。）"""
    model = getattr(args, "sidecar_model_path", None)
    if model is None or "FP8" not in Path(model).name:
        return []  # 未指定=服务端默认 BF16，或显式非 FP8 模型，无此约束
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader,nounits"]
            + (["-i", str(args.server_gpu)]
               if args.server_gpu is not None else []),
            check=False, capture_output=True, text=True,
            timeout=15).stdout
        caps = [float(x) for x in out.replace(" ", "").split(",")
                if x.strip().replace(".", "").isdigit()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []  # 查不到就不拦（不阻塞无 nvidia-smi 的环境）
    if not caps or min(caps) >= 8.9:
        return []
    return [(f"FP8 checkpoint needs compute capability >= 8.9; this GPU "
             f"has {min(caps):g} (Ampere). Pass "
             f"--sidecar-model Qwen3.6-35B-A3B (BF16) or another "
             f"non-FP8 model")]


def _parse_cpu_spec_ids(spec: str) -> list[int]:
    """Parse a comma cpu-id spec ("0-31", "0-15,32,40-41", "auto-half")
    into a sorted list of ints. "auto-half" resolves against THIS
    process's affinity exactly like the sidecar's _apply_server_cpu_limit
    (FIRST ceil(n/2) cores). Returns [] for empty/"unpinned"/garbage —
    preflight only WARNs on those, never blocks."""
    text = str(spec or "").strip()
    if not text or text.lower() == "unpinned":
        return []
    if text.lower() == "auto-half":
        try:
            own = sorted(os.sched_getaffinity(0))
        except AttributeError:
            return []
        k = max(1, -(-len(own) // 2))
        return own[:k]
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                return []
            if lo < 0 or hi < lo:
                return []
            out.update(range(lo, hi + 1))
        else:
            try:
                n = int(part)
            except ValueError:
                return []
            if n < 0:
                return []
            out.add(n)
    return sorted(out)


def _cpu_spec_overlap_problem(args) -> str | None:
    """Concurrent-phase cpu disjointness: --conc-server-cpus (the
    sidecar pin the driver passes) vs --conc-edge-cpu-ids (the
    terminal pool every concurrent cell inherits). Overlapping sets
    are a config CONTRADICTION — the two budgets are the experiment's
    independent variables — so preflight fails instead of letting a
    70h matrix measure engine/terminal core contention.

    The RUNNER-level /stats check still guards sidecar-manage=skip
    runs (a sidecar the driver did not configure); this check guards
    the driver's OWN default pair (0-31 vs 32-58) and manual flags."""
    if args.phases == "serial":
        return None
    server_ids = _parse_cpu_spec_ids(args.conc_server_cpus)
    edge_ids = _parse_cpu_spec_ids(args.conc_edge_cpu_ids)
    if not server_ids or not edge_ids:
        return None  # unpinned/unparseable — runner-level check warns
    overlap = sorted(set(server_ids) & set(edge_ids))
    if not overlap:
        return None
    def _fmt(ids: list[int]) -> str:
        return (",".join(str(c) for c in ids) if len(ids) <= 16
                else f"{len(ids)} cpus ({ids[0]}..{ids[-1]})")
    return (f"--conc-server-cpus [{_fmt(server_ids)}] overlaps "
            f"--conc-edge-cpu-ids [{_fmt(edge_ids)}] on "
            f"[{_fmt(overlap)}] — the sidecar and the terminal workers "
            f"would compete for the same cores; keep the two pools "
            f"disjoint (e.g. server 0-31 + edge 32-63)")


def preflight(args, cells: list[Cell]) -> None:
    problems = []
    if not SERIAL_RUNNER.exists():
        problems.append(f"missing {SERIAL_RUNNER}")
    if not CONCURRENT_RUNNER.exists():
        problems.append(f"missing {CONCURRENT_RUNNER}")
    # CPU disjointness (concurrent phase only): the sidecar's
    # --conc-server-cpus pin vs the terminals' --conc-edge-cpu-ids
    # pool. Overlap = contaminated wave timings -> hard stop. Runs
    # in dry-run too — the planned sidecar/cell commands are exactly
    # what a real run would execute.
    overlap_problem = _cpu_spec_overlap_problem(args)
    if overlap_problem:
        problems.append(overlap_problem)
    if not args.dry_run:
        if not args.dataset_dir or not Path(args.dataset_dir).exists():
            problems.append(f"dataset dir not found: {args.dataset_dir}")
        if shutil.which("tcpdump") is None:
            problems.append("tcpdump not installed")
        if subprocess.run(["sudo", "-n", "true"],
                          capture_output=True).returncode != 0:
            problems.append("passwordless sudo unavailable (sudo -n true)")
        # Only check GPU headroom when THIS driver owns the sidecar
        # lifecycle: in skip mode the sidecar is already holding ~35GB
        # by design, and free≈45GB would sit right at the threshold,
        # aborting perfectly valid runs (fatal on smaller GPUs).
        # With --server-gpu the check targets THAT card.
        if args.sidecar_manage == "auto":
            gpu = gpu_free_mb(args.server_gpu)
            if gpu is not None and gpu < 40_000:
                gpu_name = (str(args.server_gpu)
                            if args.server_gpu is not None else "max")
                problems.append(
                    f"only {gpu:.0f} MiB GPU free (<40GB) on "
                    f"gpu {gpu_name} — sidecar engine may not fit")
            problems += _fp8_arch_problems(args)
            if args.sidecar_model_path and \
                    not Path(args.sidecar_model_path).exists():
                problems.append(
                    f"sidecar model not found: {args.sidecar_model_path}")
        # GPU-split advisories (print-only; the matrix still runs).
        if args.server_gpu is not None \
                and args.server_gpu == args.client_gpu:
            print(f"{LOG} WARNING: --server-gpu and --client-gpu are the "
                  "same card — no isolation between engine and edge towers")
        if args.server_gpu is not None and args.sidecar_manage != "auto":
            print(f"{LOG} WARNING: --server-gpu set but "
                  "--sidecar-manage skip — the sidecar is pre-started, "
                  "the env injection never happens")
        # Either one-sided GPU split leaves the unpinned side to inherit
        # the ambient CUDA_VISIBLE_DEVICES — it can land on the same
        # card as the pinned side (engine+towers contention).
        one_sided = ((args.client_gpu is not None
                      and args.server_gpu is None)
                     or (args.server_gpu is not None
                         and args.client_gpu is None))
        if one_sided and args.conc_mem_frac >= 0.8 \
                and args.phases != "serial":
            print(f"{LOG} WARNING: --conc-mem-frac {args.conc_mem_frac} "
                  "with a one-sided GPU split (only --server-gpu or only "
                  "--client-gpu) — the unpinned side may share a card "
                  "with the sidecar; consider 0.7 or pin both sides")
    if problems:
        raise SystemExit("preflight failed:\n  - " + "\n  - ".join(problems))
    # name collision check. Resume mode: a cell NOT in resume_done was
    # either never started or FAILED last time — its stale db/pcap
    # would collide with the retry, so delete them (logged) instead of
    # aborting the whole 70h matrix on leftover artifacts.
    # NOTE: gate on args.resume itself, NOT resume_done — a previous run
    # where EVERY cell failed leaves resume_done empty, and a falsy
    # check would fall into the "delete it or --resume" hard error, the
    # exact contradiction resume exists to avoid.
    # Skipped in dry-run: after a real run the artifacts exist BY DESIGN
    # and a later --dry-run must not be blocked on them.
    resuming = args.resume is not None
    if not args.dry_run:
        done = set(args.resume_done or [])
        for c in cells:
            if c.cell_id in done:
                continue
            for p in (c.db, c.pcap):
                if Path(p).exists():
                    if resuming:
                        print(f"{LOG} resume: removing stale artifact "
                              f"from an unfinished/failed cell: {p}")
                        Path(p).unlink()
                    else:
                        problems.append(
                            f"existing artifact for {c.cell_id}: {p} "
                            "(delete it, --resume, or --clean)")
        if problems:
            raise SystemExit("preflight failed:\n  - "
                             + "\n  - ".join(problems))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Qwen3.6 full-matrix driver (serial + concurrent, "
                    "retention + optional-retention ladders, per-cell "
                    "pcap/db, aggregated report).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--phases", default="both",
                    choices=["serial", "concurrent", "both"])
    ap.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    ap.add_argument("--retentions", default=None,
                    help="comma list of retention (r) values for the edge "
                         "ladder. Unset (default) keeps the historical split: "
                         "serial phase runs the full default ladder "
                         f"{DEFAULT_RETENTIONS} while concurrent edge cells "
                         "stay fixed at r=0.6. Explicit values (e.g. "
                         "'0.6,0.7') are honored by BOTH phases; empty "
                         "string '' disables edge cells in both phases")
    ap.add_argument("--optional-retention", default="",
                    metavar="LIST",
                    help="comma list of ABSOLUTE optional-retention (o) "
                         "values for the two-tier ladder, e.g. "
                         "'0.06,0.12,0.18' — every (r, o) pair becomes a "
                         "tiered cell in BOTH phases (invalid pairs with "
                         "0 < o < r are warned and skipped). Empty (default) "
                         "keeps the legacy single derived value o = r*0.30 "
                         "per retention tier")
    ap.add_argument("--quants", default=",".join(DEFAULT_QUANTS),
                    help="fp8 and/or bf16")
    ap.add_argument("--users", default=",".join(str(u) for u in DEFAULT_USERS),
                    help="concurrent user ladder")
    ap.add_argument("--serial-limit", type=int, default=DEFAULT_SERIAL_LIMIT,
                    help=f"videos per serial cell (default {DEFAULT_SERIAL_LIMIT})")
    ap.add_argument("--conc-videos-per-user",
                    default=str(DEFAULT_CONC_VIDEOS_PER_USER),
                    metavar="V[,V...]",
                    help=f"comma list of videos per concurrent user; every "
                         f"(users, V) pair becomes a concurrent cell with "
                         f"limit = V * users (default "
                         f"{DEFAULT_CONC_VIDEOS_PER_USER})")
    ap.add_argument("--dataset-dir",
                    default=os.environ.get("ENCPL_DATASET_DIR",
                                           DEFAULT_DATASET_DIR))
    ap.add_argument("--server-host", default="localhost",
                    help="sidecar host (non-localhost = dual-host mode: "
                         "sidecars assumed pre-started there, whole-"
                         "interface netem on this host's egress)")
    ap.add_argument("--sidecar-manage", default="auto",
                    choices=["auto", "skip"],
                    help="auto = this driver starts/stops the sidecars "
                         "sequentially; skip = already running")
    ap.add_argument("--sidecar-wait", type=float, default=900)
    # ---- stream safety timeouts (forwarded to every cell's runner) ----
    ap.add_argument("--stream-connect-timeout", type=float, default=20,
                    metavar="SEC",
                    help="TCP connect timeout for sidecar HTTP requests "
                         "inside cells (0 disables)")
    ap.add_argument("--stream-first-event-timeout", type=float, default=200,
                    metavar="SEC",
                    help="SERIAL cells: budget from chat POST to the first "
                         "SSE event (0 disables)")
    ap.add_argument("--conc-stream-first-event-timeout", type=float,
                    default=900, metavar="SEC",
                    help="CONCURRENT cells: first-SSE-event budget — must "
                         "cover barrier-wave queueing behind "
                         "max-running-requests prefills (0 disables)")
    ap.add_argument("--stream-idle-timeout", type=float, default=20,
                    metavar="SEC",
                    help="max silent gap between SSE events mid-stream "
                         "(0 disables)")
    ap.add_argument("--stream-upload-read-timeout", type=float, default=1200,
                    metavar="SEC",
                    help="read timeout for the bundle-upload PUT response "
                         "(0 disables)")
    # ---- per-cell wall-clock caps (P2 watchdog) ----
    ap.add_argument("--serial-cell-timeout", type=float, default=36000,
                    metavar="SEC",
                    help="wall-clock cap per serial cell; a cell exceeding "
                         "it is killed (whole process group) and marked "
                         "failed, the matrix continues (36000 = 10h; 0 "
                         "disables — historical hang-forever behaviour)")
    ap.add_argument("--conc-cell-timeout", type=float, default=14400,
                    metavar="SEC",
                    help="wall-clock cap per concurrent cell (14400 = 4h; "
                         "0 disables)")
    # ---- capture fidelity (tcpdump ring) ----
    ap.add_argument("--pcap-buffer-kib", type=int, default=262144,
                    metavar="KIB",
                    help="tcpdump -B AF_PACKET ring size in KiB per cell "
                         "capture. Default 262144 (256MB): no-emulation "
                         "loopback bursts (GSO 64KB super-packets) overflow "
                         "tcpdump's ~2MB default ring and the kernel drops "
                         "30-80%% of uplink frames invisibly (observed: "
                         "pcap held 111MB of a 322MB upload while server "
                         "ACKs proved full delivery). 0 = tcpdump default; "
                         "raise net.core.rmem_max if the kernel clamps it")
    ap.add_argument("--conc-mem-frac", type=float, default=0.85,
                    help="--mem-fraction-static for the concurrent "
                         "sidecar (0.85 assumes the GPU-split layout: "
                         "with --server-gpu/--client-gpu on separate "
                         "cards the engine owns its GPU; on a shared "
                         "card consider 0.7 to leave tower headroom)")
    ap.add_argument("--conc-max-running-requests", type=int, default=2,
                    metavar="R",
                    help="scheduler running-request cap forwarded to the "
                         "concurrent sidecar (auto mode only)")
    ap.add_argument("--conc-server-cpus", default="0-31", metavar="SPEC",
                    help="--server-cpus pin for the concurrent sidecar "
                         "(auto mode only); keep disjoint from "
                         "--conc-edge-cpu-ids")
    ap.add_argument("--conc-mixed-chunk",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="forward --enable-mixed-chunk to the concurrent "
                         "sidecar (auto mode only)")
    ap.add_argument("--conc-mm-io-worker-num", type=int, default=4,
                    metavar="N",
                    help="--mm-io-worker-num for the concurrent sidecar "
                         "(auto mode only)")
    ap.add_argument("--conc-mm-processor-worker-num", type=int, default=4,
                    metavar="N",
                    help="--mm-processor-worker-num for the concurrent "
                         "sidecar (auto mode only)")
    ap.add_argument("--conc-decode-threads", type=int, default=2,
                    metavar="N",
                    help="ENGINE-side --decode-threads forwarded to the "
                         "concurrent sidecar; 0 = omit (auto mode only)")
    ap.add_argument("--conc-preprocess-threads", type=int, default=4,
                    metavar="N",
                    help="ENGINE-side --preprocess-threads forwarded to "
                         "the concurrent sidecar; 0 = omit (auto mode "
                         "only)")
    ap.add_argument("--conc-edge-cpu-ids", default="32-58", metavar="SPEC",
                    help="client-terminal CPU pool appended to every "
                         "concurrent cell's runner command ('' = omit, "
                         "runner default pool)")
    ap.add_argument("--conc-edge-cpus-per-worker", type=int, default=3,
                    metavar="K",
                    help="--edge-cpus-per-worker appended to every "
                         "concurrent cell's runner command")
    ap.add_argument("--conc-client-decode-threads", type=int, default=2,
                    metavar="N",
                    help="client-side --decode-threads appended to every "
                         "concurrent cell's runner command; 0 = omit")
    ap.add_argument("--server-gpu", type=int, default=None, metavar="N",
                    help="GPU id (CUDA_VISIBLE_DEVICES) for the sidecar "
                         "process group in auto mode — the GPU split "
                         "that keeps the engine and the client edge "
                         "towers on separate cards")
    ap.add_argument("--client-gpu", type=int, default=None, metavar="N",
                    help="GPU id (CUDA_VISIBLE_DEVICES) forced onto every "
                         "cell subprocess (runner + spawned edge workers "
                         "that load the terminal ViT tower)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sidecar-model", default=None, metavar="PATH",
                    help="override BOTH sidecars' --model (绝对路径，或相对"
                         " aicodec/aicodec/models/ 的目录名，如 "
                         "Qwen3.6-35B-A3B 走 BF16 权重；默认 = 服务端默认 "
                         "Qwen3.6-35B-A3B（BF16）)")
    ap.add_argument("--sidecar-mem-frac", type=float, default=None,
                    metavar="F",
                    help="override the SERIAL sidecar's --mem-fraction-"
                         "static (默认 = 服务端 0.85；A100-80G BF16-35B "
                         "建议 0.92。并发 sidecar 用 --conc-mem-frac)")
    ap.add_argument("--fail-fast", action="store_true",
                    help="abort the whole matrix on the first cell that "
                         "raises (default: record the failure and run "
                         "the remaining cells; runner exit codes != 0 "
                         "are ALWAYS non-fatal either way)")
    ap.add_argument("--cooldown-sec", type=float, default=15.0,
                    help="settle pause between cells: netem fully torn "
                         "down, sidecar caches quiescent, prior pcap "
                         "flushed to disk before the next cell's netem "
                         "is applied")
    ap.add_argument("--resume", default=None,
                    help="manifest json from a previous run; completed "
                         "cells are skipped and their run results are "
                         "carried into the new manifest/report; stale "
                         "artifacts of FAILED cells are removed")
    ap.add_argument("--report-only", default=None,
                    help="regenerate the report from a manifest and exit")
    ap.add_argument("--no-pcap-report", action="store_true",
                    help="skip pcap parsing in the report (UL/DL/retrans/"
                         "rtt columns render '-'); useful when the report "
                         "host cannot parse GB-scale captures")
    ap.add_argument("--cell", default=None, metavar="LIST",
                    help="run only the cells whose id matches — a comma "
                         "list of EXACT ids and/or fnmatch wildcards "
                         "(e.g. 'id1,id2' or 'q36m_serial_*-fp8__lossy'); "
                         "single id keeps the legacy debug behavior")
    ap.add_argument("--clean", action="store_true",
                    help="remove ALL generated artifacts (manifests, driver/"
                         "sidecar logs, cell DBs, cell logs, pcaps, reports, "
                         "csv/pcap_cache/charts, overlay yamls), ask for "
                         "confirmation, then exit — the matrix is NOT run")
    ap.add_argument("--yes", action="store_true",
                    help="with --clean: skip the interactive confirmation "
                         "(for nohup/cron runs where stdin is not a tty)")
    args = ap.parse_args()

    args.profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    # --retentions unset = historical split (serial full default ladder,
    # concurrent fixed r=0.6); explicit = BOTH phases honor the list.
    if args.retentions is None:
        args.retentions = list(DEFAULT_RETENTIONS)
        args.conc_retentions = [0.6]
    else:
        args.retentions = [float(r) for r in args.retentions.split(",")
                           if r.strip()]
        args.conc_retentions = list(args.retentions)
    args.optionals = [float(o) for o in args.optional_retention.split(",")
                      if o.strip()]
    args.quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    args.users = [int(u) for u in args.users.split(",") if u.strip()]
    try:
        args.conc_videos_per_user = [int(v) for v in
                                     args.conc_videos_per_user.split(",")
                                     if v.strip()]
    except ValueError:
        raise SystemExit("invalid --conc-videos-per-user (expected comma "
                         "list of ints, e.g. '3' or '3,5')")
    if not args.conc_videos_per_user or any(
            v < 1 for v in args.conc_videos_per_user):
        raise SystemExit("--conc-videos-per-user needs >=1 values, e.g. "
                         "'3' or '3,5'")

    dual = not is_local(f"http://{args.server_host}:1")
    if dual:
        args.sidecar_manage = "skip"

    # --sidecar-model 规范化为绝对路径（相对名锚定 models/ 目录）
    args.sidecar_model_path = None
    if args.sidecar_model:
        p = Path(args.sidecar_model)
        if not p.is_absolute():
            p = REPO_ROOT / "aicodec" / "aicodec" / "models" / p
        args.sidecar_model_path = str(p)

    # ---- report-only mode ----------------------------------------------
    # ---- clean-only mode ------------------------------------------------
    if args.clean:
        clean_artifacts(assume_yes=args.yes)
        return

    if args.report_only:
        with open(args.report_only, encoding="utf-8") as f:
            manifest = json.load(f)
        ts = time.strftime("%Y%m%d_%H%M%S")
        generate_report(manifest, REPORT_DIR / f"report_{ts}.md",
                        REPORT_DIR / f"report_{ts}.json",
                        use_pcap=not args.no_pcap_report)
        return

    cells = build_matrix(args)
    if args.cell:
        patterns = [p.strip() for p in args.cell.split(",") if p.strip()]
        cells = [c for c in cells
                 if any(fnmatch.fnmatch(c.cell_id, pat) for pat in patterns)]
        if not cells:
            full = build_matrix(args)
            sample = ", ".join(c.cell_id for c in full[:3])
            raise SystemExit(
                f"no cell matches --cell {args.cell!r} (matrix has "
                f"{len(full)} cells, e.g. {sample} ...); check that "
                f"--profiles/--retentions/--users/--quants and "
                f"--optional-retention cover the requested ids")

    args.resume_done = set()
    resumed_results: dict[str, dict] = {}
    if args.resume:
        with open(args.resume, encoding="utf-8") as f:
            prev = json.load(f)
        prev_results = prev.get("results") or {}
        args.resume_done = {
            r["cell_id"] for r in prev_results.values()
            if isinstance(r, dict) and r.get("ok")}
        # Carry the previous run's result dicts into the new manifest so
        # the final report keeps their pcap bytes / wall_sec /
        # opt_udp_delta (accuracy alone would survive via the per-cell
        # DBs, but the run columns would silently go empty otherwise).
        resumed_results = {
            cid: r for cid, r in prev_results.items()
            if isinstance(r, dict) and r.get("ok")
            and cid in {c.cell_id for c in cells}}
        print(f"{LOG} resume: {len(args.resume_done)} cells already done "
              f"({len(resumed_results)} results carried over)")

    preflight(args, cells)

    for d in (OUT_DIR, CELL_LOG_DIR, CAPTURE_DIR, REPORT_DIR, OVERLAY_DIR):
        d.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    manifest_path = OUT_DIR / f"manifest_{ts}.json"
    # driver heartbeat log: one line per driver lifecycle event (start,
    # sidecar up/down, per-cell completion, teardown). A dead driver is
    # distinguishable from a slow one by tail -f on this file — the
    # 2026-10-20 postmortem relied on inferring "nothing ran" from
    # untouched sidecar logs + stale manifest mtimes.
    driver_log_path = OUT_DIR / f"driver_{ts}.log"

    def _dlog(msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(f"{LOG} {msg}")
        try:
            driver_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(driver_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass  # heartbeat is best-effort; never let it kill the run

    manifest = {
        "meta": {"started": ts, "phases": args.phases,
                      "profiles": args.profiles, "retentions": args.retentions,
                  "optional_retention": args.optionals or None,
                  "quants": args.quants, "users": args.users,
                  "retentions": args.retentions,
                  "conc_retentions": getattr(args, "conc_retentions",
                                             [0.6]),
                  "serial_limit": args.serial_limit,
                  "conc_videos_per_user": args.conc_videos_per_user,
                  "stream_timeouts": {
                      "connect": args.stream_connect_timeout,
                      "first_event_serial":
                          args.stream_first_event_timeout,
                      "first_event_conc":
                          args.conc_stream_first_event_timeout,
                      "idle": args.stream_idle_timeout,
                      "upload_read": args.stream_upload_read_timeout,
                  },
                  "cell_timeouts": {
                      "serial": args.serial_cell_timeout,
                      "conc": args.conc_cell_timeout,
                  },
                  "dataset_dir": args.dataset_dir,
                 "server_host": args.server_host,
                 "server_gpu": args.server_gpu,
                 "client_gpu": args.client_gpu,
                 "sidecar_mode": args.sidecar_manage,
                 "sidecar_model": getattr(args, "sidecar_model_path", None),
                 "sidecar_mem_frac": args.sidecar_mem_frac,
                 "conc": {
                     "mem_frac": args.conc_mem_frac,
                     "max_running_requests": args.conc_max_running_requests,
                     "server_cpus": args.conc_server_cpus,
                     "mixed_chunk": args.conc_mixed_chunk,
                     "mm_io_worker_num": args.conc_mm_io_worker_num,
                     "mm_processor_worker_num":
                         args.conc_mm_processor_worker_num,
                     "decode_threads": args.conc_decode_threads,
                     "preprocess_threads": args.conc_preprocess_threads,
                     "edge_cpu_ids": args.conc_edge_cpu_ids,
                     "edge_cpus_per_worker": args.conc_edge_cpus_per_worker,
                     "client_decode_threads":
                         args.conc_client_decode_threads,
                 },
                  "driver": str(Path(__file__).resolve()),
                  # explicit mode marker: a dry-run manifest must never be
                  # mistakable for a real run's (the 14:30 dry-run pair was
                  # indistinguishable from a silently-dead real run in the
                  # 2026-10-20 postmortem).
                  "dry_run": bool(args.dry_run),
                  "driver_log": str(driver_log_path)},
          "cells": [c.to_dict() for c in cells],
          "results": {},
     }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    total = len(cells)
    serial_cells = [c for c in cells if c.phase == "serial"]
    conc_cells = [c for c in cells if c.phase == "concurrent"]
    print(f"{LOG} matrix: {total} cells "
          f"(serial {len(serial_cells)} + concurrent {len(conc_cells)}); "
          f"manifest: {manifest_path}")

    if args.dry_run:
        if args.sidecar_manage == "auto" and not dual:
            print(f"{LOG} planned sidecar launches (auto mode):")
            for line in _planned_sidecar_lines(args):
                print(f"  {line}")
            print()
        for i, c in enumerate(cells, 1):
            finalize_cell_cmd(c, args)
            env = sanitized_env({"ENCPL_DATASET_DIR": args.dataset_dir})
            env.update(c.cmd_env)
            env_str = " ".join(f"{k}={v}" for k, v in cell_env_sorted(env))
            port = (SERIAL_PORT if c.phase == "serial" else CONC_PORT)
            udp = (SERIAL_UDP_PORT if c.phase == "serial"
                   else CONC_UDP_PORT) if c.tiered else None
            base_url = f"http://{args.server_host}:{port}"
            print(f"\n[{i:3d}/{total}] {c.cell_id}")
            print(f"    db:   {c.db}")
            print(f"    pcap: {c.pcap}")
            print(f"    filter: {bpf_filter(c, base_url)}")
            if c.profile != "no_emulation":
                udp_part = f", udp_port={udp}" if udp else ""
                print(f"    netem: apply_profile_to_loopback("
                      f"{c.profile}, tcp_port={port}{udp_part})")
            else:
                print("    netem: (none — no_emulation clears lo)")
            if env_str:
                print(f"    env:   {env_str}")
            print(f"    cmd:   {sys.executable} {c.runner} "
                  + " ".join(shlex.quote(a) for a in c.cmd_args))
            if c.overlay_config:
                print(f"    overlay: {c.overlay_config} "
                      f"(optional_ratio={c.optional})")
        print(f"\n{LOG} dry-run complete; {total} cells, no execution. "
              f"Manifest: {manifest_path}")
        _dlog(f"dry-run complete: {total} cells planned, no execution")
        return

    # ---- execution -------------------------------------------------------
    # Seeded with the resumed cells' previous results so the final
    # report keeps their run columns (pcap bytes / wall / opt delta).
    results: dict[str, dict] = dict(resumed_results)

    def _run_cells(phase_cells: list[Cell], port: int) -> None:
        for i, c in enumerate(phase_cells, 1):
            if c.cell_id in args.resume_done:
                continue
            print(f"{LOG} [{i}/{len(phase_cells)}] {c.cell_id}")
            _dlog(f"cell start [{i}/{len(phase_cells)}] {c.cell_id}")
            before = _sidecar_stats(f"http://{args.server_host}:{port}")
            try:
                r = run_cell(c, args, sidecar_stats_before=before)
            except (Exception, SystemExit) as e:
                # One bad cell (tc hiccup, netem iface resolution,
                # capture spawn...) must not forfeit the remaining
                # dozens of cells of a ~70h matrix. The runner exit
                # code path inside run_cell is already non-fatal.
                r = {"cell_id": c.cell_id, "ok": False,
                     "error": f"driver-level failure: "
                              f"{type(e).__name__}: {e}"}
                print(f"{LOG}   -> DRIVER-LEVEL FAILURE: {e}")
                if args.fail_fast:
                    raise
            results[c.cell_id] = r
            print(f"{LOG}   -> ok={r.get('ok')} "
                  f"wall={r.get('wall_sec', '?')}s "
                  f"pcap={((r.get('pcap') or {}).get('bytes'))}B")
            _dlog(f"cell done [{i}/{len(phase_cells)}] {c.cell_id}: "
                  f"ok={r.get('ok')} wall={r.get('wall_sec', '?')}s")
            _save_manifest(manifest_path, manifest, results)
            if args.cooldown_sec > 0:
                time.sleep(args.cooldown_sec)

    overall_t0 = time.time()
    _dlog(f"driver start: {total} cells (serial {len(serial_cells)} "
          f"+ concurrent {len(conc_cells)}), dry_run={args.dry_run}, "
          f"manifest={manifest_path}")
    exit_code = 0
    try:
        if serial_cells:
            sidecar = None
            if args.sidecar_manage == "auto":
                sidecar = Sidecar(SERIAL_SIDECAR_SCRIPT, SERIAL_PORT,
                                  OUT_DIR / "sidecar_serial.log")
                serial_extra = ["--opt-udp-port", str(SERIAL_UDP_PORT)]
                if args.sidecar_model_path:
                    serial_extra += ["--model", args.sidecar_model_path]
                if args.sidecar_mem_frac is not None:
                    serial_extra += ["--mem-fraction-static",
                                     f"{args.sidecar_mem_frac:g}"]
                sidecar.start(serial_extra,
                              wait_sec=args.sidecar_wait,
                              env_extra=sidecar_env(args, "serial"))
                _dlog(f"serial sidecar up on :{SERIAL_PORT} "
                      f"(log: {sidecar.log})")
            try:
                _run_cells(serial_cells, SERIAL_PORT)
            finally:
                if sidecar:
                    sidecar.stop()
                    _dlog(f"serial sidecar :{SERIAL_PORT} stopped")
        if conc_cells:
            sidecar = None
            if args.sidecar_manage == "auto":
                sidecar = Sidecar(CONCURRENT_SIDECAR_SCRIPT, CONC_PORT,
                                  OUT_DIR / "sidecar_concurrent.log")
                conc_extra = (["--opt-udp-port", str(CONC_UDP_PORT)]
                              + conc_sidecar_args(args))
                if args.sidecar_model_path:
                    conc_extra += ["--model", args.sidecar_model_path]
                sidecar.start(conc_extra,
                              wait_sec=args.sidecar_wait,
                              env_extra=sidecar_env(args, "concurrent"))
                _dlog(f"concurrent sidecar up on :{CONC_PORT} "
                      f"(log: {sidecar.log})")
            try:
                _run_cells(conc_cells, CONC_PORT)
            finally:
                if sidecar:
                    sidecar.stop()
                    _dlog(f"concurrent sidecar :{CONC_PORT} stopped")
    except BaseException as exc:
        # Driver-level death (sidecar.start failure, unexpected
        # crash...): make the death VISIBLE — full traceback on
        # stderr, "driver_error" in the manifest, a heartbeat line,
        # and a non-zero exit. Without this, a dead driver left a
        # fresh-looking manifest with results={} and nothing else
        # (2026-10-20 15:43 launch: manifest written 15:43:22, zero
        # follow-on activity for 34+ min, no traceback anywhere).
        import traceback
        tb = traceback.format_exc()
        print(f"{LOG} FATAL: driver aborted: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.stderr.write(tb)
        _dlog(f"FATAL driver abort: {type(exc).__name__}: {exc}")
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(OUT_DIR / f"driver_{ts}.crash.log", "w",
                      encoding="utf-8") as f:
                f.write(tb)
        except OSError:
            pass
        manifest["driver_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        exit_code = 1
    finally:
        try:
            clear_netem(f"http://{args.server_host}:{SERIAL_PORT}")
        except Exception as exc:  # noqa: BLE001 — teardown is best-effort
            print(f"{LOG} WARNING: netem clear failed: {exc}")
            _dlog(f"netem clear failed: {exc}")
        manifest["results"] = results
        try:
            _save_manifest(manifest_path, manifest, results)
        except OSError as exc:
            print(f"{LOG} WARNING: final manifest save failed: {exc}")
            _dlog(f"final manifest save failed: {exc}")
        _dlog(f"driver exit: code={exit_code}, "
              f"{len(results)} results recorded")

    if exit_code:
        sys.exit(exit_code)

    wall = time.time() - overall_t0
    n_fail = sum(1 for r in results.values() if not r.get("ok"))
    print(f"{LOG} all cells done in {wall / 3600:.1f}h "
          f"({n_fail} failed); manifest: {manifest_path}")
    generate_report(manifest, REPORT_DIR / f"report_{ts}.md",
                    REPORT_DIR / f"report_{ts}.json",
                    use_pcap=not args.no_pcap_report)


def cell_env_sorted(env: dict) -> list[tuple[str, str]]:
    """Only the cell-EXTRA vars (not the whole sanitized env) for display."""
    interesting = set(TIERED_ENV.values()) | {"ENCPL_DATASET_DIR"}
    return sorted((k, v) for k, v in env.items() if k in interesting)


def _clean_targets() -> list[tuple[str, Path]]:
    """(label, dir) pairs for every artifact category; the dir itself is
    always KEPT (skeleton), only matching files inside are removed."""
    return [
        ("manifests/driver logs/cell DBs (logs/q36m)", OUT_DIR),
        ("cell logs (logs/q36m/cell_logs)", CELL_LOG_DIR),
        ("pcaps (results/captures/q36m)", CAPTURE_DIR),
        ("reports/csv/pcap_cache/charts (results/q36m)", REPORT_DIR),
        ("overlay configs (configs/q36m_overlay)", OVERLAY_DIR),
    ]


def _clean_glob(d: Path) -> list[Path]:
    """Files to delete inside dir d. Timestamped/dated artifacts and all
    per-cell outputs match; unrelated files are left untouched."""
    pats = ["manifest_*.json", "manifest_*.tmp",
            "driver_*.log", "driver_*.crash.log", "sidecar_*.log",
            "*.db", "*.pcap", "report_*.md", "report_*.json", "report_*.tmp"]
    if d == OVERLAY_DIR:
        # q36m_overlay holds only generated per-cell yamls; *.yaml never
        # appears in the other target dirs
        pats = ["*.yaml"]
    files = []
    for p in d.glob("*"):
        if p.is_dir():
            if p.name in ("cell_logs", "csv", "pcap_cache") or \
                    p.name.startswith("charts_"):
                files.extend(q for q in p.glob("*") if q.is_file())
            continue
        if any(fnmatch.fnmatch(p.name, pat) for pat in pats):
            files.append(p)
    return files


def clean_artifacts(assume_yes: bool = False) -> None:
    """--clean: remove ALL q36m generated artifacts (list -> confirm ->
    delete). Directory skeletons are kept so a subsequent run (or the
    self-healing mkdir in the writers) never sees a missing dir.
    assume_yes (--yes) skips the interactive confirm — required under
    nohup/cron where stdin is not a tty."""
    groups = []
    total_n, total_bytes = 0, 0
    for label, d in _clean_targets():
        if not d.is_dir():
            continue
        files = _clean_glob(d)
        if files:
            groups.append((label, d, files))
            total_n += len(files)
            total_bytes += sum(f.stat().st_size for f in files)

    if not groups:
        print(f"{LOG} clean: nothing to clean "
              f"(checked {len(_clean_targets())} artifact dirs)")
        return

    print(f"{LOG} clean: artifacts found:")
    for label, d, files in groups:
        sz = sum(f.stat().st_size for f in files)
        print(f"  {label}: {len(files)} files, {sz / 1e6:.1f} MB")
    print(f"  TOTAL: {total_n} files, {total_bytes / 1e6:.1f} MB")

    if assume_yes:
        print(f"{LOG} clean: --yes given, skipping confirmation")
    elif not sys.stdin.isatty():
        print(f"{LOG} clean: stdin is not a tty — aborting without "
              f"deletion (rerun with --yes, or interactively to confirm)")
        return
    else:
        try:
            ans = input("delete ALL above artifacts? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print(f"{LOG} clean: aborted, nothing deleted")
            return

    ok_n = 0
    for label, d, files in groups:
        for f in files:
            try:
                f.unlink()
                ok_n += 1
            except OSError as exc:
                print(f"{LOG} WARNING: could not delete {f}: "
                      f"{type(exc).__name__}: {exc}")
    print(f"{LOG} clean: deleted {ok_n}/{total_n} files "
          f"({total_bytes / 1e6:.1f} MB) — skeletons kept")


def _save_manifest(path: Path, manifest: dict, results: dict) -> None:
    manifest["results"] = results
    # self-heal: the logs dir may have been wiped mid-run (2026-10-23
    # postmortem: manifest tmp-write crashed with FileNotFoundError after
    # logs/q36m was cleaned externally)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


if __name__ == "__main__":
    main()
