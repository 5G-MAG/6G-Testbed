#!/usr/bin/env python3
"""
SGLang Engine Server (concurrent variant, Qwen3.6) for the 6G AI
Traffic Testbed.

Standalone sibling of server/sglang_engine_server_concurrent.py (the
Qwen3-VL lock-free sidecar) for the Qwen3.6 (qwen3_5 / qwen3_5_moe)
encoder-placement-concurrent experiment — ZERO changes to the
Qwen3-VL files, so both experiments run/debug in parallel on
different ports. Also a concurrency-mixed sibling of the serial
Qwen3.6 sidecar (sglang_engine_server_qwen36.py, port 8004): same
model, same patch, same bundle contract — this variant just drops the
whole-generation engine_lock so N barrier-aligned terminal users hit
the in-process SGLang Engine at once and continuous batching decides
the queueing (that queueing IS the measurement object of the
concurrent scenario).

Default port 8014 (Qwen3-VL sidecars own 8002/8012,
serial Qwen3.6 8004).

Model: aicodec/aicodec/models/Qwen3.6-35B-A3B (model_type
qwen3_5_moe, arch Qwen3_5MoeForConditionalGeneration — hybrid GDN
linear-attention MoE, 35B total / 3B active, BF16 weights ~70GB; the
BF16 default keeps the sidecar loadable on Ampere (cc 8.0) where the
fp8-e4m3 checkpoint cannot run). The FP8 snapshot and the Qwen3.6-27B
dense checkpoint (model_type qwen3_5) work unchanged via --model:
sglang 0.5.20 registers BOTH classes (qwen3_5.py EntryClass), and the
sidecar is model-agnostic.

  server (mode "server"): the terminal uploads the original mp4 via
                            PUT /v1/bundles/{id}?kind=video. The chat
                            request references bundle://video/{id}; the
                            server hands the raw bytes to SGLang through
                            video_data with a video_token placeholder in
                            the prompt, so decode + processor + ViT +
                            prefill all run server-side.
  edge   (mode "edge"):   the terminal runs decode + processor + ViT
                            locally (single-segment latent: Qwen3.6 has
                            NO deepstack, feature is [N, 2048]),
                            serializes the SGLang-native
                            precomputed_embedding dict and uploads it via
                            PUT /v1/bundles/{id}?kind=edge. The chat
                            request references bundle://edge/{id}; the
                            server deserializes it and passes it to
                            engine.async_generate() as input_ids +
                            video_data=[{"format":
                            "precomputed_embedding", ...}], skipping the
                            server-side visual encoder entirely.

Edge-mode sglang patching: same story as the serial Qwen3.6 sidecar —
the PRECOMPUTED_EMBEDDING pass-through for the Qwen3_6 classes lives
in qwen3_vl.py::_get_visual_feature (the method
Qwen3_5MoeForConditionalGeneration INHERITS), so the EXISTING
patch_host_sglang.py (start_sidecar_concurrent.sh step 6) covers
Qwen3.6 too. start_sidecar_concurrent_qwen36.sh therefore REUSES
patch_host_sglang.py verbatim. The parent-process monkey-patch here
(_patch_qwen3_6_precomputed_feature) wraps the same registry-scanned
class names as the serial Qwen3.6 sidecar.

Concurrency architecture — copied VERBATIM from the Qwen3-VL
concurrent sidecar (it is model-agnostic):

  1. NO whole-generation engine_lock. Requests enter the in-process
     Engine as N coroutines on the ENGINE'S OWN asyncio loop
     (engine.loop, uvloop; created by sglang 0.5.20's in-process
     Engine with the tokenizer_manager zmq channels bound to it).
     Driving it from uvicorn's loop raises "Cannot run the event loop
     while another loop is running", so an EngineLoopBridge runs
     engine.loop FOREVER on a dedicated daemon thread and every engine
     interaction is submitted via asyncio.run_coroutine_threadsafe.
     Chat streams: one coroutine per request awaiting
     engine.async_generate(stream=True), relaying chunks through a
     queue.Queue to the uvicorn-side async generator. Structurally
     identical to the standard sglang launch_server: one loop, N
     request coroutines, one scheduler subprocess.

  2. _prepare_video_mode / _prepare_edge_mode run in the loop's
     thread pool via asyncio.to_thread (edge: torch.load; server:
     bytes hand-off) — per-request CPU work that safely overlaps;
     only Engine.generate() is the shared GPU resource.

  3. /flush_cache is drain-and-gated by an active-request counter
     (GenerationGate) instead of the global lock: waits for in-flight
     generations, gates new ones (clear SSE error), submits the flush
     coroutine to the engine loop. The concurrent scenario flushes
     once per wave before the barrier release, so the gate is open in
     practice.

Everything else — endpoints, wire-byte accounting, SSE chunk format,
engine_stats fields — is byte-for-byte the same contract as both the
Qwen3-VL concurrent sidecar and the serial Qwen3.6 sidecar, so
EncoderPlacementQwen36Client works against either Qwen3.6 server
unchanged.

Endpoints:
  GET  /v1/models                — OpenAI-style model list
  PUT  /v1/bundles/{bundle_id}   — binary upload (raw mp4 or
                                    torch.save'd precomputed bundle;
                                    the response carries opt_udp_port
                                    where the tiered optional wave is
                                    received)
  POST /v1/chat/completions      — OpenAI chat completions (SSE
                                    streaming only); messages reference
                                    uploads through bundle:// URLs
  POST /flush_cache              — drain-and-gate radix + mm cache
                                    flush (called by the concurrent
                                    scenario once per wave)
  GET  /stats                    — active generations + flush gate
                                    state + server CPU budget + the
                                    effective video-sampling config +
                                    optional-tier UDP receiver health

Video frame sampling (--frame-ratio / --max-frames) is configured at
sidecar STARTUP via the official mm_process_config ServerArg (see
encpl_video_sampling.py); /stats surfaces the effective values.
"""

import argparse
import asyncio
import io
import json
import logging
import math
import os
import queue
import socket
import struct
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encpl_video_sampling import (
    build_mm_process_config,
    describe_mm_process_config,
    ensure_sglang_patches,
    format_log_line,
    mm_process_config_help,
    parse_body_sampling,
    verify_frame_ratio_patch,
)


# deep_ep (MoE expert-parallel lib) crashes at import time via JIT init
# when CUDA_HOME is unset; the AssertionError escapes sglang's
# `except ImportError` guard and kills `import sglang` entirely. The
# fp8 MoE engine here does not use deep_ep (no expert-parallel
# dispatch on a single-node TP setup); stub it so the import degrades
# to ImportError -> sglang sets use_deepep=False.
sys.modules.setdefault("deep_ep", None)

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SGLangEngineServerConcurrentQwen36")

MSS = 1460


def on_wire(payload_bytes: int) -> tuple[int, int]:
    """TCP overhead model from split_logical.py: payload + 40B/packet + 4."""
    packets = max(1, math.ceil(payload_bytes / MSS))
    return payload_bytes + packets * 40 + 4, packets


class BundleStore:
    """In-memory store for uploaded binaries (raw videos or edge bundles).

    Unchanged from the Qwen3-VL concurrent sidecar: internally locked,
    so N concurrent PUTs (one per terminal user) interleave safely.
    The eviction max must cover one full wave (N bundles) — see
    --max-bundles / SGLANG_SERVER_MAX_BUNDLES below.

    Tiered-mode optional parts live in a separate side table keyed by
    the MUST bundle's id; they are appended by the UDP receiver
    (OptUdpReceiver, E2U datagrams reassembled into parts — the
    receiver is shared with the serial sidecar's design) and
    atomically snapshotted (popped) when the chat request merges —
    parts arriving after the snapshot are counted as late_dropped and
    never retransmitted (best-effort contract).
    """

    # retired ids kept in the side-tables (FIFO beyond this: a purged
    # id can only be re-created by a re-PUT, which un-retires it anyway)
    _RETIRE_KEEP = 1024

    def __init__(self, max_bundles: int = 16):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._max_bundles = max_bundles
        self._parts: dict[str, list[dict]] = {}
        self._part_seq: dict[str, int] = {}
        self._late_dropped: dict[str, int] = {}
        self._retired: set[str] = set()
        # retirement order (FIFO for the bounded purge)
        self._retire_order: deque[str] = deque()
        # per-bundle seen part seqs (dup detection — first arrival
        # wins, duplicates counted into parts_dup_dropped; a duplicate
        # would otherwise DOUBLE-MERGE the same optional tokens)
        self._part_seqs_seen: dict[str, set[int]] = {}
        self._dup_dropped: dict[str, int] = {}

    def _retire(self, bundle_id: str) -> None:
        """Mark an id retired and append it to the FIFO order.

        Caller must hold self._lock. Idempotent per id (pop_parts on an
        already-evicted id must not double-append).
        """
        already = bundle_id in self._retired
        self._retired.add(bundle_id)
        if not already:
            self._retire_order.append(bundle_id)

    def _retire_purge_locked(self) -> None:
        """FIFO-purge the retirement side-tables beyond _RETIRE_KEEP.

        Caller must hold self._lock. Only the last _RETIRE_KEEP retired
        ids stay in _part_seq/_late_dropped/_dup_dropped/_retired — a
        long-running server otherwise grows four leak-style tables
        forever (retirement is terminal per id, so a purged id's stats
        are frozen and its stragglers simply count pending instead of
        late; a re-PUT un-retires and restarts the lifecycle anyway).
        """
        while len(self._retire_order) > self._RETIRE_KEEP:
            old = self._retire_order.popleft()
            self._part_seq.pop(old, None)
            self._late_dropped.pop(old, None)
            self._dup_dropped.pop(old, None)
            self._retired.discard(old)

    def put(self, kind: str, data: bytes, meta: dict | None = None,
            bundle_id: str | None = None) -> str:
        bundle_id = bundle_id or uuid.uuid4().hex
        entry = {
            "kind": kind,
            "size": len(data),
            "data": data,
            "meta": meta or {},
            "created": time.time(),
        }
        with self._lock:
            if len(self._store) >= self._max_bundles:
                oldest = min(self._store, key=lambda k: self._store[k]["created"])
                del self._store[oldest]
                # drop the orphaned parts side-table too and mark the
                # id retired so stragglers count late, not pending
                self._parts.pop(oldest, None)
                self._part_seq.pop(oldest, None)
                self._late_dropped.pop(oldest, None)
                self._part_seqs_seen.pop(oldest, None)
                self._dup_dropped.pop(oldest, None)
                self._retire(oldest)
                self._retire_purge_locked()
            # a re-used id is a NEW bundle: un-retire it AND wipe the
            # stale part-side state (BUG X2: a previous lifecycle's
            # _part_seq made put_part treat the new bundle as already
            # merged — every new part counted late-dropped and the
            # re-sent must wave could never merge)
            if bundle_id in self._retired:
                self._retired.discard(bundle_id)
                try:
                    self._retire_order.remove(bundle_id)
                except ValueError:
                    pass
                self._part_seq.pop(bundle_id, None)
                self._late_dropped.pop(bundle_id, None)
                self._dup_dropped.pop(bundle_id, None)
            # always: parts lifecycle restarts with the new store entry
            self._parts.pop(bundle_id, None)
            self._part_seqs_seen.pop(bundle_id, None)
            self._store[bundle_id] = entry
        return bundle_id

    def pop(self, bundle_id: str) -> dict:
        with self._lock:
            entry = self._store.pop(bundle_id, None)
        if entry is None:
            raise KeyError(bundle_id)
        return entry

    def put_part(self, bundle_id: str, data: bytes, seq: int) -> None:
        """Append one best-effort optional part for a must bundle.

        Lifecycle: the must bundle is PUT into the store, then POPPED by
        the chat request, then merge-snapshotted (pop_parts). A part is
        PENDING while the bundle is in the store OR popped-but-not-yet
        merged (the chat request's prepare/merge is still running —
        those parts belong to this merge window and MUST be accepted).
        After pop_parts (or eviction), the id is RETIRED: any later part
        is LATE — counted and dropped, never retransmitted.

        Duplicated seq for the same id is dropped (first arrival wins)
        and counted into parts_dup_dropped — a duplicate on the wire
        would otherwise DOUBLE-MERGE the same optional tokens and
        corrupt the input_ids/feature row count.
        """
        with self._lock:
            if bundle_id in self._retired or bundle_id in self._part_seq:
                # merged already (snapshot consumed) or evicted —
                # either way the part is unusable.
                self._late_dropped[bundle_id] = (
                    self._late_dropped.get(bundle_id, 0) + 1)
                return
            seen = self._part_seqs_seen.setdefault(bundle_id, set())
            if int(seq) in seen:
                self._dup_dropped[bundle_id] = (
                    self._dup_dropped.get(bundle_id, 0) + 1)
                return
            seen.add(int(seq))
            self._parts.setdefault(bundle_id, []).append(
                {"seq": int(seq), "size": len(data), "data": data,
                 "created": time.time()})

    def pop_parts(self, bundle_id: str) -> tuple[list[dict], int]:
        """Atomic snapshot of the parts arrived so far (merge point).

        Retires the id so any later put_part is counted late-dropped.
        Returns (parts_in_seq_order, late_count BEFORE this call).
        Retirement is bounded (see _retire_purge_locked).
        """
        with self._lock:
            parts = self._parts.pop(bundle_id, [])
            late = self._late_dropped.get(bundle_id, 0)
            self._part_seq[bundle_id] = len(parts)
            self._retire(bundle_id)
            self._part_seqs_seen.pop(bundle_id, None)
            self._retire_purge_locked()
        parts.sort(key=lambda p: p["seq"])
        return parts, late

    def part_stats(self, bundle_id: str) -> dict:
        with self._lock:
            return {
                "parts_pending": len(self._parts.get(bundle_id, [])),
                "parts_merged": self._part_seq.get(bundle_id, 0),
                "parts_late_dropped": self._late_dropped.get(bundle_id, 0),
                "parts_dup_dropped": self._dup_dropped.get(bundle_id, 0),
                "retired": bundle_id in self._retired,
            }

    # stale-pending sweep: TTL and cap for the two tables only
    # pop_parts/put-eviction would otherwise leave behind (BUG K):
    # _parts holds the actual part bytes (~KBs each) and
    # _part_seqs_seen the dedup sets.
    _PARTS_TTL_SEC = 60.0
    _MAX_PENDING_IDS = 512

    def sweep_stale_parts(self, now: float | None = None) -> int:
        """Drop pending parts whose merge will never come.

        Two leak classes (both leave the id un-retired forever, so the
        round-3 retirement purge never touches them):
          1. ABORTED merge — the chat request died between store.pop()
             and _merge_tiered_parts (client disconnect / HTTPException
             in _prepare_edge_mode); parts for that id stay pending.
          2. NEVER-PUT id — parts arriving for an id no store entry
             ever existed for (stale sender / port reuse).

        Policy: an id is swept when its OLDEST pending part is older
        than _PARTS_TTL_SEC (a real merge window is sub-second; 60s is
        ~100x margin), or when more than _MAX_PENDING_IDS un-retired
        ids hold parts (oldest-id-first). Swept ids are retired, so
        stragglers count late-dropped like any other post-merge
        arrival. Caller: OptUdpReceiver._janitor (every 0.5s tick).
        Returns the number of PARTS dropped.
        """
        now = time.time() if now is None else now
        with self._lock:
            dropped = 0
            # 1) TTL sweep (oldest part per id)
            for bid in list(self._parts):
                if bid in self._retired:
                    continue  # pop_parts() will/has consumed them
                parts = self._parts[bid]
                if parts and now - parts[0]["created"] > self._PARTS_TTL_SEC:
                    dropped += len(parts)
                    del self._parts[bid]
                    self._part_seqs_seen.pop(bid, None)
                    self._retire(bid)
                    self._retire_purge_locked()
            # 2) cap sweep (oldest-id-first over un-retired holders)
            pending = [b for b in self._parts if b not in self._retired]
            over = len(pending) - self._MAX_PENDING_IDS
            if over > 0:
                pending.sort(key=lambda b: self._parts[b][0]["created"])
                for bid in pending[:over]:
                    dropped += len(self._parts[bid])
                    del self._parts[bid]
                    self._part_seqs_seen.pop(bid, None)
                    self._retire(bid)
                    self._retire_purge_locked()
            return dropped


# E2U wire format (must match the client's codec exactly):
# 30-byte header "<4s16sIHHH" = magic "E2U1" + bundle_id uuid bytes +
# part_seq u32 + frag_idx u16 + frag_cnt u16 + payload len u16, then
# <= 1370 payload bytes. No CRC — see the client's codec comment.
_E2U_MAGIC = b"E2U1"
_E2U_HEADER = struct.Struct("<4s16sIHHH")
_E2U_HEADER_SIZE = _E2U_HEADER.size
_E2U_MAX_PAYLOAD = 1400 - _E2U_HEADER_SIZE


class OptUdpReceiver:
    """UDP receiver for the tiered optional-tier second wave (E2U).

    N receive daemon threads bound BEFORE uvicorn.run (so the port is
    live when the first must PUT reports it); N>1 uses SO_REUSEPORT on
    one shared (host, port) — the multi-worker PARALLEL receive lives
    HERE (the serial sidecar keeps a single thread on purpose: it
    serves one user at a time). recvfrom runs on its own threads (NOT
    the engine loop or the uvicorn loop); a complete part is fed to
    store.put_part, which is internally locked and safe against the
    concurrent chat merges.

    SO_REUSEPORT affinity: the kernel hashes each flow's 4-tuple onto
    one socket of the group. A user's wave is sent from ONE client
    socket (_send_opt_parts opens a fresh socket per request), so that
    user's ENTIRE wave is served by ONE worker thread — intra-user
    fragment order preserved without cross-thread protocol — while
    different concurrent users land on different workers (true
    concurrent receive under multi-user waves). Each socket has its
    own SO_RCVBUF, so aggregate kernel buffering scales with N
    (blast-mode overflow headroom). tc/netem/pcap filters match on the
    UDP PORT, not the socket count, so the shaping/capture path is
    unchanged. Part OWNERSHIP is decided by the datagram's 16B
    bundle_id (reassembly key (bid, part_seq) -> store bucket), fully
    decoupled from which worker receives it.

    Every worker loop: recvfrom with a 0.5s timeout; every timeout
    tick runs a janitor that evicts incomplete reassembly buffers
    older than 5s (counts them into parts_incomplete_dropped — true
    UDP semantics: a part missing ANY fragment is dropped whole and
    never retransmitted).

    Reassembly key = (bundle_id 16B, part_seq). The shared reassembly
    map is capped (default 256 part keys, FIFO overflow drop,
    counted; raise for high-concurrency waves ~ max concurrent users
    x parts in flight).

    quiesce_drain(bundle_id, ms): bounded per-bundle drain called by
    _merge_tiered_parts BEFORE pop_parts() — polls until no new
    fragment for THIS bundle has arrived for a 3ms quiet window (and
    it owns no reassembly keys) or the ms cap, so datagrams sitting in
    the kernel SO_RCVBUF are reassembled before the merge snapshot
    takes them. Per-bundle matters MOST here: N concurrent users share
    the receiver, so a global quiet window would stall every merge on
    the slowest user's still-arriving wave.

    Locks: _reassemble_lock guards the reassembly map /
    bid_last_seen; _stats_lock guards the counters (bare `+=` under
    the GIL loses updates with N>1 threads, and fragments_rx is the
    kernel-overflow diagnostic — it must stay exact). Order when
    nested: _reassemble_lock -> _stats_lock.
    """

    _SOCK_TIMEOUT = 0.5
    _JANITOR_EVICTION_SEC = 5.0
    _DRAIN_QUIET_SEC = 0.003
    _DRAIN_POLL_SEC = 0.0005

    def __init__(self, store: BundleStore, port: int, host: str = "0.0.0.0",
                 workers: int = 1, reassembly_cap: int = 256):
        self._store = store
        self._port = int(port)
        self._host = host
        # N receive sockets (SO_REUSEPORT) -> true concurrent receive.
        # Default 1 = legacy single-socket behavior. See start().
        self._workers = max(1, int(workers))
        self._reassembly_cap = max(1, int(reassembly_cap))
        self.port = 0          # actual bound port (set on successful bind)
        self._socks: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        # {(id16, part_seq): {"frags": {idx: bytes}, "cnt": int, "t": float}}
        self._reassemble: dict = {}
        self._reassemble_lock = threading.Lock()
        # per-bundle last-fragment time (key = 16B uuid bytes), the
        # per-bundle activity signal quiesce_drain polls; TTL-cleaned
        # by the janitor so idle bundle ids do not accumulate
        self._bid_last_seen: dict[bytes, float] = {}
        # observability counters (read by /stats). Guarded by
        # _stats_lock: with N>1 receive threads the bare `+=` compounds
        # (read-modify-write) can lose updates under the GIL, and
        # fragments_rx is THE diagnostic for kernel-buffer overflow —
        # it must be exact. Lock order: _reassemble_lock ->
        # _stats_lock (janitor nests the two; nothing nests reverse).
        self._stats_lock = threading.Lock()
        self.fragments_rx = 0
        self.fragments_bad = 0
        self.parts_reassembled = 0
        self.parts_incomplete_dropped = 0
        self.parts_overflow_dropped = 0
        # pending parts dropped by the stale sweep (aborted merge /
        # never-PUT id) — janitor-fed
        self.parts_stale_dropped = 0
        self.rcvbuf = 0        # actual granted SO_RCVBUF (per socket)

    def start(self) -> bool:
        """Bind + spawn the receive thread(s). Returns True on success.

        workers == 1: single socket, SO_REUSEADDR only — the legacy
        path, byte-for-byte the former behavior.

        workers > 1: N sockets all set SO_REUSEADDR + SO_REUSEPORT bound
        to the SAME (host, port). The kernel hashes the 4-tuple onto one
        socket of the group: a user's wave is sent from ONE client
        socket (_send_opt_parts opens a fresh socket per request), so
        that user's ENTIRE wave lands on ONE worker — intra-user
        datagram order is preserved without any cross-thread protocol,
        while different users spread across workers (true concurrent
        receive under multi-user waves). Each socket carries its own
        SO_RCVBUF, so aggregate kernel buffering scales with N — the
        other half of the blast-mode overflow risk. Group semantics
        require every member to set the same options; a partial bind
        failure tears the whole group down and falls back to a single
        socket (workers=1), logged loudly. SO_REUSEPORT is Linux-only;
        platforms without it (or where bind fails) get the fallback too.
        """
        def _bind_one(reuse_port: bool):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if reuse_port:
                    if not hasattr(socket, "SO_REUSEPORT"):
                        raise OSError(
                            "SO_REUSEPORT not available on this platform")
                    sock.setsockopt(
                        socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                sock.bind((self._host, self._port))
            except OSError:
                sock.close()
                raise
            sock.settimeout(self._SOCK_TIMEOUT)
            # Raise the kernel receive buffer so a full blast wave
            # (send_gap_ms=0) queues in-kernel instead of overflowing.
            # Best-effort: sysctl rmem_max may clamp it; the ACTUAL
            # granted value is exposed via /stats (opt_udp.rcvbuf) so
            # a clamped run is visible instead of silent. With N>1
            # sockets the AGGREGATE buffering is N * rcvbuf.
            try:
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                granted = int(sock.getsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF))
                self.rcvbuf = max(self.rcvbuf, granted)
            except OSError as exc:
                logger.warning(
                    "OptUdpReceiver: SO_RCVBUF raise failed (%s); "
                    "blast-mode waves may drop datagrams locally", exc)
            return sock

        n = self._workers
        socks: list[socket.socket] = []
        try:
            if n == 1:
                socks.append(_bind_one(reuse_port=False))
            else:
                for _ in range(n):
                    socks.append(_bind_one(reuse_port=True))
        except OSError as exc:
            # close the partial group (or the lone failed socket is
            # already closed by _bind_one) before any fallback
            for sock in socks:
                sock.close()
            socks = []
            if n > 1:
                logger.error(
                    "OptUdpReceiver: cannot bind %s:%d x%d with "
                    "SO_REUSEPORT (%s); falling back to a SINGLE receive "
                    "socket — optional-tier receive stays serial", self._host,
                    self._port, n, exc)
                n = 1
                try:
                    socks.append(_bind_one(reuse_port=False))
                except OSError as exc2:
                    logger.error(
                        "OptUdpReceiver: cannot bind %s:%d (%s); "
                        "optional-tier UDP transport is OFF (tiered clients "
                        "will fail fast on opt_udp_port=0)", self._host,
                        self._port, exc2)
                    return False
            else:
                logger.error(
                    "OptUdpReceiver: cannot bind %s:%d (%s); optional-tier "
                    "UDP transport is OFF (tiered clients will fail fast "
                    "on opt_udp_port=0)", self._host, self._port, exc)
                return False
        self._socks = socks
        self._workers = n
        self.port = socks[0].getsockname()[1]
        self._threads = [
            threading.Thread(
                target=self._run, args=(sock,), daemon=True,
                name=f"opt-udp-receiver-{i}")
            for i, sock in enumerate(socks)]
        for t in self._threads:
            t.start()
        logger.info(
            "OptUdpReceiver listening on udp://%s:%d workers=%d "
            "rcvbuf=%d/socket (aggregate=%d)", self._host, self.port,
            self._workers, self.rcvbuf, self.rcvbuf * self._workers)
        return True

    def stats(self) -> dict:
        with self._stats_lock:
            fragments_rx = self.fragments_rx
            fragments_bad = self.fragments_bad
            parts_reassembled = self.parts_reassembled
            parts_incomplete = self.parts_incomplete_dropped
            parts_overflow = self.parts_overflow_dropped
            parts_stale = self.parts_stale_dropped
        return {
            "port": self.port,
            "workers": self._workers,
            "alive": bool(self._threads) and all(
                t.is_alive() for t in self._threads),
            "alive_workers": sum(
                1 for t in self._threads if t.is_alive()),
            "rcvbuf": self.rcvbuf,
            "fragments_rx": fragments_rx,
            "fragments_bad": fragments_bad,
            "parts_reassembled": parts_reassembled,
            "parts_incomplete_dropped": parts_incomplete,
            "parts_overflow_dropped": parts_overflow,
            "parts_stale_dropped": parts_stale,
        }

    def quiesce_drain(self, bundle_id: str, max_ms: float) -> None:
        """Drain in-flight datagrams for ONE bundle before its snapshot.

        Returns when THIS bundle's last-fragment arrival time has been
        stable for _DRAIN_QUIET_SEC (each new fragment resets the
        window), bounded by max_ms. Per-bundle so concurrent requests
        cannot stall each other: user A's drain must not sit out user
        B's still-arriving wave. Call this on the REQUEST path (before
        pop_parts); the recvfrom loop keeps running on its own thread
        the whole time.

        Quiet is judged SOLELY by the last-seen timestamp: a part whose
        fragments were LOST leaves orphan reassembly keys behind, but
        those keys mean nothing more is arriving — waiting on them
        would pin every lossy merge to the full max_ms cap (BUG M).
        Orphan keys are reclaimed by the janitor's 5s TTL instead.
        """
        if max_ms <= 0:
            return
        # BUG L guard: bundle_id is CLIENT-CONTROLLED (the bundle://
        # path); a non-hex-32 id cannot match any reassembly key
        # (datagram ids are 16B uuids), so draining it is a no-op —
        # never a 500 on the request path.
        try:
            bid = uuid.UUID(hex=bundle_id).bytes
        except (ValueError, AttributeError, TypeError):
            return
        deadline = time.time() + max_ms / 1000.0
        last_seen: float | None = None
        quiet_since: float | None = None
        while time.time() < deadline:
            with self._reassemble_lock:
                seen = self._bid_last_seen.get(bid)
            if seen == last_seen:
                if quiet_since is None:
                    quiet_since = time.time()
                if time.time() - quiet_since >= self._DRAIN_QUIET_SEC:
                    return
            else:
                quiet_since = None
                last_seen = seen
            time.sleep(self._DRAIN_POLL_SEC)

    def _run(self, sock: socket.socket):
        while True:
            try:
                datagram, _addr = sock.recvfrom(65535)
            except socket.timeout:
                self._janitor()
                continue
            except OSError:
                # socket closed / transient error: log, keep serving
                logger.exception("OptUdpReceiver recvfrom failed")
                time.sleep(0.1)
                continue
            try:
                self._handle(datagram)
            except Exception:
                # BUG R: a single malformed datagram must never kill
                # the receiver thread — every LATER wave would land in
                # a dead socket buffer until the janitor evicts it.
                # Log + count + keep the loop alive.
                logger.exception("OptUdpReceiver handle failed "
                                 "(datagram dropped)")
                with self._stats_lock:
                    self.fragments_bad += 1

    def _handle(self, datagram: bytes):
        # header validation (no CRC by design — see client codec)
        if len(datagram) < _E2U_HEADER_SIZE:
            with self._stats_lock:
                self.fragments_bad += 1
            return
        magic, bid, part_seq, frag_idx, frag_cnt, plen = _E2U_HEADER.unpack(
            datagram[:_E2U_HEADER_SIZE])
        if magic != _E2U_MAGIC or plen != len(datagram) - _E2U_HEADER_SIZE:
            with self._stats_lock:
                self.fragments_bad += 1
            return
        if frag_cnt < 1 or frag_idx >= frag_cnt:
            with self._stats_lock:
                self.fragments_bad += 1
            return
        # NOTE: plen == 0 is ACCEPTED (symmetry with the client codec:
        # _e2u_fragments legitimately produces a single zero-payload
        # fragment for an empty part; only the LAST fragment may be
        # short, checked below). Round-7 regression W: rejecting empty
        # payloads here broke that symmetry — a receiver stricter than
        # its own sender drops valid waves.
        if frag_idx < frag_cnt - 1 and plen != _E2U_MAX_PAYLOAD:
            # BUG Q: only the LAST fragment may carry a short payload.
            # A mid-part short fragment can never come from
            # _e2u_fragments (it slices full _E2U_MAX_PAYLOAD chunks);
            # accepting one would silently corrupt the reassembled
            # part (concatenation shorter than the ins_pos/layout the
            # sender packed), so drop it as malformed instead.
            with self._stats_lock:
                self.fragments_bad += 1
            return
        with self._stats_lock:
            self.fragments_rx += 1
        payload = datagram[_E2U_HEADER_SIZE:]

        with self._reassemble_lock:
            self._bid_last_seen[bid] = time.time()
            key = (bid, part_seq)
            buf = self._reassemble.get(key)
            if buf is None:
                if len(self._reassemble) >= self._reassembly_cap:
                    # FIFO overflow: drop the OLDEST incomplete buffer
                    oldest = min(self._reassemble,
                                 key=lambda k: self._reassemble[k]["t"])
                    del self._reassemble[oldest]
                    with self._stats_lock:
                        self.parts_overflow_dropped += 1
                buf = {"frags": {}, "cnt": frag_cnt, "t": time.time()}
                self._reassemble[key] = buf
            elif buf["cnt"] != frag_cnt:
                # BUG Q: a fragment claiming a DIFFERENT total count
                # for the same (bundle, part) than the fragments
                # already buffered cannot come from one
                # _e2u_fragments call. Either the wire corrupted the
                # header or two senders collided on the same id.
                # Believing either count risks joining mismatched
                # payloads: keep the buffered one (drop the stranger)
                # so the part either completes on its original count
                # or times out whole. (Counted as bad below; the
                # fragments_rx bump above is rolled back so ONE
                # datagram lands in exactly ONE bucket.)
                with self._stats_lock:
                    self.fragments_rx -= 1
                    self.fragments_bad += 1
                return
            buf["t"] = time.time()
            buf["frags"][frag_idx] = payload
            if len(buf["frags"]) < buf["cnt"]:
                return
            part = b"".join(buf["frags"][i]
                            for i in range(buf["cnt"]))
            del self._reassemble[key]
        with self._stats_lock:
            self.parts_reassembled += 1
        self._store.put_part(uuid.UUID(bytes=bid).hex, part, seq=part_seq)

    def _janitor(self):
        now = time.time()
        with self._reassemble_lock:
            stale = [k for k, v in self._reassemble.items()
                     if now - v["t"] > self._JANITOR_EVICTION_SEC]
            for k in stale:
                del self._reassemble[k]
                with self._stats_lock:
                    self.parts_incomplete_dropped += 1
            idle = [b for b, t in self._bid_last_seen.items()
                    if now - t > self._JANITOR_EVICTION_SEC]
            for b in idle:
                del self._bid_last_seen[b]
        # BUG K sweep: pending parts whose merge will never come
        # (aborted merge / never-PUT id) — TTL + cap, below.
        # store._lock + _stats_lock, taken OUTSIDE _reassemble_lock:
        # the documented order is _reassemble_lock -> _stats_lock and
        # this is the only place store._lock nests with ours; keeping
        # it outside _reassemble_lock means quiesce_drain (which takes
        # _reassemble_lock) never waits behind a store sweep.
        with self._stats_lock:
            self.parts_stale_dropped += self._store.sweep_stale_parts(now)


# global receiver instance (None until main() starts it; 0-port probes
# read `getattr(_opt_udp_receiver, "port", 0)` for the PUT response)
_opt_udp_receiver: "OptUdpReceiver | None" = None
# merge-snapshot drain budget (ms), set from --opt-merge-drain-ms in
# main(); 0 disables the drain (pure snapshot semantics)
_opt_merge_drain_ms: float = 50.0


class GenerationGate:
    """Drain-and-gate coordination between generations and flush_cache.

    Replaces a coarse global engine_lock (same design as the Qwen3-VL
    concurrent sidecar):

      - enter(): registers an active generation. Returns False (caller
        must abort with an SSE error) when a flush is holding the gate.
      - exit(): deregisters; wakes waiting flushes when the last
        generation drains.
      - flush_close(): waits for active generations to drain, then
        closes the gate so NEW generations cannot start; releases in
        flush_open().

    Fairness note: not strictly FIFO between waiting flushes and new
    enter() attempts — a racing enter() may slip in before the gate
    closes. That is benign here (the flush simply waits one more
    generation); strict fairness would add latency for no measurement
    value.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._active = 0
        self._flushing = False
        self._flush_waiters = 0

    def enter(self) -> bool:
        with self._cond:
            if self._flushing:
                return False
            self._active += 1
            return True

    def exit(self):
        with self._cond:
            self._active -= 1
            if self._active <= 0 and self._flush_waiters > 0:
                self._cond.notify_all()

    @property
    def active(self) -> int:
        with self._cond:
            return self._active

    def flush_close(self):
        with self._cond:
            self._flush_waiters += 1
            while self._active > 0:
                self._cond.wait()
            self._flush_waiters -= 1
            self._flushing = True

    def flush_open(self):
        with self._cond:
            self._flushing = False
            self._cond.notify_all()


def _patch_qwen3_6_precomputed_feature():
    """Monkey-patch PRECOMPUTED_EMBEDDING pass-through (parent process).

    Same mechanism and same registry-scanned class targets as the
    serial Qwen3.6 sidecar's _patch_qwen3_6_precomputed_feature
    (sglang_engine_server_qwen36.py): wrap get_image_feature /
    get_video_feature on the Qwen3_5* model classes so precomputed
    bundles bypass the visual tower. The parent-process patch covers
    THIS process's tokenizer manager; the spawned scheduler children
    need the ON-DISK patch (patch_host_sglang.py, anchored in
    qwen3_vl.py::_get_visual_feature — the method the Qwen3_6 classes
    inherit — applied by start_sidecar_concurrent_qwen36.sh).

    Targets resolved defensively: iterate srt model classes by NAME
    (importing qwen3_5.py directly would break on sglang builds where
    the module is absent; the registry-level scan works on 0.5.20
    which ships qwen3_5.py with both classes registered).
    """
    import torch

    from sglang.srt.managers.schedule_batch import MultimodalInputFormat
    from sglang.srt import models as srt_models

    def _precomputed_only(items):
        return bool(items) and all(
            item.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING
            for item in items
        )

    def _pass_through(items):
        # feature rows are the final ViT output (Qwen3.6: single merged
        # segment [tokens, out_hidden_size]; NO deepstack concat);
        # normalize to [tokens, hidden] exactly like the qwen2_5_vl guard.
        feature = torch.cat([item.feature for item in items], dim=0)
        return feature.reshape(-1, feature.shape[-1])

    def _wrap(cls, name):
        original = getattr(cls, name)

        def patched(self, items):
            if _precomputed_only(items):
                return _pass_through(items)
            return original(self, items)

        setattr(cls, name, patched)

    targets = []
    for attr in dir(srt_models):
        obj = getattr(srt_models, attr)
        if not isinstance(obj, type):
            continue
        if obj.__name__ in (
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen4ExpForConditionalGeneration",
        ):
            targets.append(obj)
    for cls in targets:
        for method in ("get_image_feature", "get_video_feature"):
            if hasattr(cls, method):
                _wrap(cls, method)
    if targets:
        logger.info(
            "qwen3_6 precomputed-embedding patch applied to: %s",
            [c.__name__ for c in targets],
        )
    else:
        logger.warning(
            "qwen3_6 precomputed-embedding patch found no model classes")


def _build_engine(model_path: str, device: str, mem_fraction_static: float,
                  max_running_requests: int | None,
                  chunked_prefill_size: int | None = None,
                  enable_mixed_chunk: bool | None = None,
                  prefill_max_requests: int | None = None,
                  max_prefill_tokens: int | None = None,
                  enable_radix_cache: bool = False,
                  mm_io_worker_num: int | None = None,
                  mm_processor_worker_num: int | None = None,
                  mm_process_config: dict | None = None):
    """Create the in-process SGLang Engine (Qwen3.6, multimodal on).

    Same scheduler-shaping surface as the Qwen3-VL concurrent sidecar
    (all OFFICIAL sglang 0.5.20 ServerArgs fields, consumed via the
    Schedule namespace — no sglang source patching): under concurrency
    the KV-cache budget (mem_fraction_static), the running-request cap
    (max_running_requests), and the prefill batching geometry below
    determine batching behaviour, so a benchmark sweep needs them
    adjustable without editing code. None (CLI default) OMITS the key
    entirely so sglang applies its own default; a typo would raise
    TypeError at startup (fail fast, not silently ignored).

    knob semantics (sglang defaults when omitted):
      - max_running_requests: cap on requests admitted to the decode
        running batch; None = sglang default (computed from KV-cache
        capacity). 0 would be a falsy value with version-dependent
        semantics, so None is the only safe "unlimited" spelling.
      - chunked_prefill_size: max tokens in ONE prefill chunk
        (v0.5.20 default resolves to 8192 on this box). -1 disables
        chunked prefill entirely (whole prompt in one forward); the
        scheduler normalizes <=0 to None by itself, so the value is
        passed through verbatim.
      - enable_mixed_chunk: mix decode reqs into prefill chunks so
        decode does not stall while a long chunked prefill runs.
      - prefill_max_requests: max requests in ONE prefill batch;
        None = unlimited. Bigger values let multiple queued requests
        prefill together (#new-seq > 1) when token budgets allow.
      - max_prefill_tokens: TOTAL token budget of ONE prefill batch
        (v0.5.20 default 16384; the runtime scheduler also caps it at
        >= model context length). Raising it does NOT reduce total GPU
        work (prefill is compute-bound); it only changes the queueing
        texture — keep comparisons within one fixed geometry.
      - mm_io_worker_num: width of the mm I/O thread pool
        (sglang-mm-io) that runs video decode + sample + resize inside
        the tokenizer-manager process. One pool slot per concurrent mm
        item, so N = at most N of the wave's videos decode in PARALLEL
        across requests. Omit/None = model default (qwen_vl.py: 16).
        The SGLANG_IO_WORKERS env var still overrides the default when
        omitted/0.
      - mm_processor_worker_num: width of the multimodal processor
        pool (HF processor calls; qwen_vl default 2). 1 = synchronous
        processing (no pool at all).
    """
    from sglang.srt.entrypoints.engine import Engine

    _patch_qwen3_6_precomputed_feature()

    engine_kwargs = dict(
        model_path=model_path,
        device=device,
        enable_multimodal=True,
        trust_remote_code=True,
        # Deterministic benchmark: greedy decoding is set per-request; keep
        # engine-level flags minimal so both modes share one configuration.
        disable_cuda_graph=False,
        mem_fraction_static=mem_fraction_static,
        log_level="info",
        # Cold-prefill benchmark semantics: the wave's N users share the
        # video tokens (same file) but carry per-user "[u<i>] " text
        # prefixes, so radix ON would still only share the video suffix —
        # harmless but OFF keeps every user a guaranteed cold prefix
        # (same default as the Qwen3-VL concurrent sidecar; the wave's
        # prompt uniquification keeps semantics correct even against a
        # radix-enabled sidecar). OFF unless --enable-radix-cache.
        disable_radix_cache=not enable_radix_cache,
    )
    if max_running_requests is not None:
        engine_kwargs["max_running_requests"] = max_running_requests
    if chunked_prefill_size is not None:
        engine_kwargs["chunked_prefill_size"] = chunked_prefill_size
    if enable_mixed_chunk is not None:
        engine_kwargs["enable_mixed_chunk"] = enable_mixed_chunk
    if prefill_max_requests is not None:
        engine_kwargs["prefill_max_requests"] = prefill_max_requests
    if max_prefill_tokens is not None:
        engine_kwargs["max_prefill_tokens"] = max_prefill_tokens
    if mm_io_worker_num is not None:
        engine_kwargs["mm_io_worker_num"] = mm_io_worker_num
    if mm_processor_worker_num is not None:
        engine_kwargs["mm_processor_worker_num"] = mm_processor_worker_num
    if mm_process_config:
        engine_kwargs["mm_process_config"] = mm_process_config
    engine = Engine(**engine_kwargs)
    return engine


app = FastAPI(title="6G SGLang Engine Server (concurrent, Qwen3.6)")
engine = None
gate = GenerationGate()
# BundleStore capacity is CLI-tunable (--max-bundles, default 64: one
# full wave of N users uploads N bundles before any chat request pops
# them, so the store must hold a wave without evicting still-referenced
# entries). main() rebuilds this global from the parsed args; the
# module-level default keeps import-time use consistent.
store = BundleStore(max_bundles=int(
    os.environ.get("SGLANG_SERVER_MAX_BUNDLES", "64")))
_model_path: str = ""
# --server-cpus spec actually applied by main() ("" = unpinned);
# /stats surfaces it so the harness can verify the server CPU budget
# landed before trusting a comparison run.
_applied_server_cpus: str = ""
# Effective video-sampling knobs (None / 0 = off). Kept for the
# per-request advisory check (parse_body_sampling) and /stats.
_engine_frame_ratio: float | None = None
_engine_max_frames: int | None = None
# Default min_new_tokens for every request (guards zero-token greedy
# completions; see chat_completions). main() rebuilds it from
# --min-new-tokens; requests can override via body min_new_tokens.
_default_min_new_tokens: int = 1

# Parallel video preprocess (6G-Testbed sglang patch 4; ported from the
# Qwen3-VL concurrent sidecar with the same DEFAULT ON since the v2
# fix). The authoritative state is the module global inside
# sglang.srt.multimodal.processors.qwen_vl (read at EVERY preprocess
# call, so flipping it needs no engine restart); this string is only
# what /stats echoes and the endpoint flips. Values: "on"/"off"
# (None = sglang lacks patch 4 — endpoint refuses to flip).
_parallel_video_preprocess: str | None = None


def _qwen_vl_preprocess_module():
    """Import the (patched) qwen_vl processor module lazily.

    Returns None when the ACTIVE sglang lacks the 6G-Testbed patch 4
    (no _PARALLEL_VIDEO_PREPROCESS attribute) — every caller degrades
    to "unavailable" instead of raising (the sidecar must still boot
    and serve; the A/B experiment simply cannot toggle).
    """
    try:
        from sglang.srt.multimodal.processors import qwen_vl as qv
    except Exception:  # pragma: no cover - engine env broken elsewhere
        logger.exception("qwen_vl import failed; patch-4 switch unavailable")
        return None
    if not hasattr(qv, "_PARALLEL_VIDEO_PREPROCESS"):
        return None  # pristine sglang — patch 4 not applied
    return qv


def _sync_parallel_video_preprocess_flag() -> None:
    """One-shot at startup: read the patch-4 switch state into
    _parallel_video_preprocess ("on"/"off"/None=patch missing)."""
    global _parallel_video_preprocess
    qv = _qwen_vl_preprocess_module()
    _parallel_video_preprocess = (
        None if qv is None else ("on" if qv._PARALLEL_VIDEO_PREPROCESS else "off")
    )


def _video_decode_backend() -> str:
    """Which video decode backend the ACTIVE sglang picked ("torchcodec"
    / "decord"), read from the video_decoder module's _BACKEND global.

    Lazy import, degrade to "unknown" instead of raising: /stats must
    stay cheap and safe (an engine env with a different layout must
    still get a stats response, not a 500). Complements decode_threads:
    the decode-threads label is historical — the cap applies to BOTH
    backends, and WHICH one is active changes where decode CPU lands
    (torchcodec parallel-instances vs decord's internal pool).
    """
    try:
        from sglang.srt.utils import video_decoder as vd
    except Exception:  # pragma: no cover - engine env broken elsewhere
        logger.exception("video_decoder import failed; backend unknown")
        return "unknown"
    return str(getattr(vd, "_BACKEND", "unknown"))


# B1 — asyncio bridge over the ENGINE's own loop (verbatim design from
# the Qwen3-VL concurrent sidecar; see its long comment for the three
# attempts that led here).
class EngineLoopBridge:
    """Own the engine's private asyncio loop for cross-thread use.

    Started once from main() AFTER the Engine is constructed:
      bridge = EngineLoopBridge(engine); bridge.start()

    - start(): launches engine.loop.run_forever() on a daemon thread
      and parks the loop reference for run_coroutine_threadsafe.
    - submit(coro): schedule `coro` on the engine loop; returns a
      CONCURRENT.futures.Future (NOT asyncio) usable from any thread
      via .result(timeout) / add_done_callback / cancel.
    - stop(): best-effort loop shutdown (call only at process exit;
      the scheduler subprocess teardown is owned by sglang).

    Everything that touches engine internals (generate consumption,
    flush_cache coroutine) MUST go through submit() — calling the
    engine's sync wrappers from a thread that owns another running
    loop (uvicorn workers) crashes with "Cannot run the event loop
    while another loop is running".
    """

    def __init__(self, engine):
        self._engine = engine
        self._loop = getattr(engine, "loop", None)
        self._thread = None
        if self._loop is None:
            raise RuntimeError(
                "sglang Engine exposes no .loop — layout changed? "
                "The asyncio bridge needs the engine's private event "
                "loop to run it on a dedicated thread.")

    def start(self) -> None:
        if self._thread is not None:
            return
        import threading

        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="sglang-engine-loop",
            daemon=True,
        )
        self._thread.start()

    def submit(self, coro):
        """Schedule `coro` on the engine loop; concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self) -> None:
        if self._loop is not None and self._thread is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)


# Module-level handle set by main() once the engine exists. Request
# handlers reach it through this global (same pattern as `engine` /
# `store` above). None until startup completes; every handler that
# needs it checks first and 503s.
bridge: "EngineLoopBridge | None" = None


def _set_store_capacity(max_bundles: int) -> None:
    """Rebuild the module-level BundleStore with a new capacity.

    Called from main() after argparse (--max-bundles). Safe before any
    request arrives; guards against shrinking below 1.
    """
    global store
    if max_bundles < 1:
        raise ValueError(f"--max-bundles must be >= 1, got {max_bundles}")
    store = BundleStore(max_bundles=max_bundles)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_path or "sglang-engine-qwen36-concurrent",
            "object": "model",
            "owned_by": "sglang-engine",
        }],
    }


@app.put("/v1/bundles/{bundle_id}")
async def put_bundle(bundle_id: str, request: Request):
    kind = request.query_params.get("kind", "video")
    if kind not in ("video", "edge"):
        raise HTTPException(
            status_code=400,
            detail=f"invalid kind {kind!r}: must be 'video' or 'edge'")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty body")
    actual_id = store.put(kind=kind, data=data, bundle_id=bundle_id or None)
    wire_bytes, packets = on_wire(len(data))
    # opt_udp_port tells a tiered client where the optional-tier UDP
    # receiver listens (0 = off; a tiered client fails fast on that).
    opt_udp_port = getattr(_opt_udp_receiver, "port", 0)
    logger.info(
        f"PUT /v1/bundles/{actual_id} kind={kind} size={len(data)} wire={wire_bytes}")
    return {
        "id": actual_id,
        "kind": kind,
        "payload_bytes": len(data),
        "wire_bytes": wire_bytes,
        "packets": packets,
        "opt_udp_port": opt_udp_port,
    }


def _parse_bundle_ref(messages: list) -> tuple[str, str]:
    """Extract (kind, id) from OpenAI-style messages containing a bundle:// ref."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            url = None
            for key in ("video_url", "image_url", "bundle_url"):
                block = part.get(key)
                if isinstance(block, dict) and block.get("url"):
                    url = block["url"]
                    break
            if url and url.startswith("bundle://"):
                rest = url[len("bundle://"):]
                kind, _, bundle_id = rest.partition("/")
                return kind, bundle_id
    raise HTTPException(status_code=400, detail="no bundle:// reference in messages")


def _extract_prompt(messages: list) -> str:
    prompt = ""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    prompt = part.get("text", "")
        elif isinstance(content, str):
            prompt = content
    return prompt


# qwen2-vl conv template video_token (conversation.py: qwen2-vl template):
# the engine's chat parser expands a video_url content part to exactly this
# placeholder; video bytes are supplied separately through video_data.
# Same markup family as Qwen3-VL — the Qwen3.6 tokenizer (Qwen2Tokenizer,
# 248320 vocab) registers the same special-token STRINGS with new ids
# (248053/248057/248054), resolved by the tokenizer at parse time.
VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"


def _prepare_video_mode(entry: dict, prompt: str):
    """server mode: raw mp4 bytes + video_token placeholder for the engine.

    video_data carries the raw bytes (common.py _normalize_video_input
    accepts bytes directly, no base64 round-trip); the text prompt carries
    the <|vision_start|><|video_pad|><|vision_end|> placeholder the qwen
    processor expands against the decoded video.
    """
    text = (
        VIDEO_TOKEN + prompt if "<|vision_start|>" not in prompt else prompt
    )
    return text, {"video_data": [entry["data"]]}, None


def _merge_tiered_parts(bundle: dict, bundle_id: str) -> tuple[dict, dict]:
    """Merge the optional-tier parts that arrived before the snapshot.

    Identical logic to the serial sidecar
    (sglang_engine_server_qwen36.py _merge_tiered_parts) — the merge is
    pure per-request CPU work on already-popped data, so it runs fine
    inside asyncio.to_thread alongside N concurrent requests.

    This IS the merge point: pop_parts() atomically snapshots (and
    pops) whatever best-effort parts (UDP E2U datagrams reassembled by
    OptUdpReceiver) have landed; parts
    still in flight are late — put_part counts them late_dropped and
    their bytes are discarded. No waiting, no retransmit.

    Each part is a torch.save'd dict {ref, seq, slots, ins_pos,
    feature [n, C], mrope_cols [3, n], feature_scale?}. Per token:
      - input_ids column: the video_pad constant (resolved from the
        must bundle's own ids — longest same-id interior run)
      - mrope column: mrope_cols[:, j]
      - feature row: feature[j] (dequantized with the part's scale)
    ins_pos is the number of KEPT COLUMNS in the must bundle strictly
    before the token's original full-sequence column (text + must
    pads). Since every column of the must bundle is kept, kept-count
    == must-layout position, so the merge is: order arrived tokens by
    (ins_pos, slot), then walk the must columns once, emitting each
    token just before must column ins_pos (slot breaks ties in
    retention order). Text/timestamp/boundary columns are replayed
    verbatim with their own mrope entries.

    Returns (merged_bundle, stats{parts_merged, parts_late_dropped,
    opt_tokens_merged}); the bundle is normalized to the plain
    "precomputed_embedding" format on return (must-only when nothing
    arrived — the tiered bundle is already prefillable then).
    """
    import torch

    # Drain window: datagrams already queued in the kernel SO_RCVBUF
    # (or mid-reassembly) belong to THIS merge snapshot — snapshotting
    # while they are still in flight would silently understate the
    # merge (a measurement artifact, not network loss). Per-bundle and
    # bounded by --opt-merge-drain-ms; 0 disables (pure snapshot
    # semantics).
    if _opt_udp_receiver is not None:
        _opt_udp_receiver.quiesce_drain(bundle_id, _opt_merge_drain_ms)
    parts, late = store.pop_parts(bundle_id)
    must_rows = (int(bundle["feature"].shape[0])
                 if hasattr(bundle["feature"], "shape") else 0)
    stats = {
        "parts_merged": len(parts),
        "parts_late_dropped": late,
        # expected optional tokens = retention total - must rows; the
        # tiered client always ships retention_tokens, so a missing key
        # means an unknown/legacy bundle — report None (never a
        # self-canceling 0) so downstream stats stay honest.
        "opt_tokens_expected": (
            int(bundle["retention_tokens"]) - must_rows
            if "retention_tokens" in bundle else None),
    }

    def _normalize(b: dict) -> dict:
        b["format"] = "precomputed_embedding"
        b.pop("must_slots", None)
        b.pop("retention_tokens", None)
        b.pop("video_pad_id", None)
        return b

    # guard: no must ids at all AND no shipped pad id -> the merge can
    # never resolve the pad constant; a must bundle with zero ids is a
    # client contract violation, not a mergeable snapshot
    ids_guard = bundle.get("input_ids")
    ids_guard_1d = (ids_guard.reshape(-1)
                    if hasattr(ids_guard, "reshape") else ids_guard)
    n_guard = (len(ids_guard_1d.tolist())
               if hasattr(ids_guard_1d, "tolist")
               else len(ids_guard_1d or []))
    if n_guard == 0 and bundle.get("video_pad_id") is None:
        raise HTTPException(
            status_code=400,
            detail="tiered merge: empty must input_ids and no "
                   "video_pad_id — cannot resolve pad constant")

    if not parts:
        # must-only: the tiered bundle is ALREADY a complete
        # prefillable dict — nothing to do (accuracy floor path).
        stats["opt_tokens_merged"] = 0
        return _normalize(bundle), stats

    ids = bundle["input_ids"]
    ids_1d = ids.reshape(-1) if hasattr(ids, "reshape") else ids
    ids_list = ids_1d.tolist() if hasattr(ids_1d, "tolist") else list(ids_1d)

    def _pad_id_from_runs(seq_list: list) -> int:
        # video-pad = the id with the longest same-id run in the
        # sequence (per-frame pad runs dominate every text token run).
        best_id, best_len, cur_id, cur_len = None, 0, None, 0
        for v in seq_list:
            if v == cur_id:
                cur_len += 1
            else:
                if cur_id is not None and cur_len > best_len:
                    best_id, best_len = cur_id, cur_len
                cur_id, cur_len = v, 1
        if cur_id is not None and cur_len > best_len:
            best_id, best_len = cur_id, cur_len
        if best_id is None:
            raise HTTPException(
                status_code=400,
                detail="tiered merge: cannot resolve video_pad id")
        return best_id

    # BUG S: the tiered client ships the resolved <|video_pad|> id in
    # the bundle; prefer it — the longest-run heuristic below is a
    # fallback for legacy bundles and can misresolve when a text
    # token's run outlengths the per-frame pad runs (adversarial or
    # single-frame edge cases).
    pad_id = bundle.get("video_pad_id")
    if pad_id is None:
        pad_id = _pad_id_from_runs(ids_list)
    else:
        pad_id = int(pad_id)

    must_feature = bundle["feature"]
    must_mrope = bundle["mrope_positions"]  # [3, 1, n]
    mp = must_mrope.squeeze(1) if must_mrope.ndim == 3 else must_mrope

    # Collect per-token data from the arrived parts (order-free: every
    # token carries its own ins_pos).
    ins_list: list[int] = []
    slot_list: list[int] = []
    feat_parts: list = []
    mrope_parts: list = []
    # BUG U: slots are the identity of an opt token in the retention
    # sequence. The same slot arriving twice (duplicate datagrams +
    # re-PUT re-opening a retired bundle) would double-merge: one
    # insertion per (ins_pos, slot) pair. First arrival wins; the
    # duplicates are counted, not merged.
    seen_slots: set[int] = set()
    slot_dup_dropped = 0
    for p in parts:
        try:
            part = torch.load(io.BytesIO(p["data"]), map_location="cpu",
                              weights_only=False)
        except Exception as exc:  # corrupted part: drop, never retransmit
            logger.warning("edge2 part seq=%s unreadable (%s); dropped",
                           p["seq"], exc)
            continue
        if (not isinstance(part, dict)
                or part.get("format") != "precomputed_embedding_part"):
            logger.warning("edge2 part seq=%s wrong format; dropped",
                           p["seq"])
            continue
        feat = part["feature"]
        scale = part.get("feature_scale")
        if scale is not None:
            # dequant this part's rows with the same per-tensor contract
            feat = scale * feat.to(torch.bfloat16)
        p_ins = [int(v) for v in part["ins_pos"].reshape(-1).tolist()]
        p_slots = [int(v) for v in part["slots"].reshape(-1).tolist()]
        keep_mask = [s not in seen_slots and not seen_slots.add(s)
                     for s in p_slots]
        slot_dup_dropped += len(p_slots) - sum(keep_mask)
        if not any(keep_mask):
            continue  # entire part is a duplicate
        if all(keep_mask):
            ins_list.extend(p_ins)
            slot_list.extend(p_slots)
            feat_parts.append(feat.to(must_feature.dtype))
            mrope_parts.append(part["mrope_cols"].to(mp.dtype))
        else:
            # partial duplicate: boolean-index the surviving rows
            mask_t = torch.tensor(keep_mask, dtype=torch.bool)
            ins_list.extend(v for v, k in zip(p_ins, keep_mask) if k)
            slot_list.extend(v for v, k in zip(p_slots, keep_mask) if k)
            feat_parts.append(feat[mask_t].to(must_feature.dtype))
            mrope_parts.append(
                part["mrope_cols"][:, mask_t].to(mp.dtype))
    if slot_dup_dropped:
        stats["parts_slot_dup_dropped"] = slot_dup_dropped
        logger.warning("tiered merge: %d duplicate-slot opt tokens "
                       "dropped (same retention slot arrived twice)",
                       slot_dup_dropped)

    n_merged = len(ins_list)
    stats["opt_tokens_merged"] = n_merged
    if n_merged == 0:
        return _normalize(bundle), stats

    # Order arrived tokens by (ins_pos, slot): ins_pos is the number of
    # KEPT COLUMNS (text + must pads — the client computed it over the
    # whole-sequence keep mask) strictly before the token's original
    # column; slot breaks ties in retention-order, reproducing the
    # retention-level sequence exactly.
    order = sorted(range(n_merged), key=lambda j: (ins_list[j],
                                                   slot_list[j]))
    ins_sorted = [ins_list[j] for j in order]
    feat_ordered = torch.cat(feat_parts, dim=0)[order]     # [n, C]
    mrope_ordered = torch.cat(mrope_parts, dim=-1)[:, order]  # [3, n]

    # Column-space interleave: walk the must bundle's columns; insert
    # each arrived token just before must column ins_pos (ins_pos =
    # kept columns before it; slot breaks ties in retention order).
    # Trailing text columns are always kept, so every opt token's
    # ins_pos < L; a defensive straggler flush covers the impossible
    # ins_pos == L case anyway.
    L = len(ids_list)
    if max(ins_sorted) >= L:
        raise HTTPException(
            status_code=400,
            detail=f"tiered merge: ins_pos {max(ins_sorted)} exceeds must "
            f"sequence length {L} — part/bundle mismatch (wrong ref?)")

    final_ids: list[int] = []
    feat_list: list = []
    mrope_list: list = []
    mr = 0   # consumed must feature rows (== consumed must PAD columns)
    k = 0    # consumed arrived-opt tokens
    mc = 0   # consumed MUST-BUNDLE columns (ids/mrope share this cursor:
             # every must column — text or pad — has an mrope column)
    # map must column -> feature row: pad columns consume rows in order
    is_pad_list = [v == pad_id for v in ids_list]
    for p in range(L):
        # tokens whose ins_pos == p go BEFORE must column p
        while k < n_merged and ins_sorted[k] == p:
            final_ids.append(pad_id)
            feat_list.append(feat_ordered[k])
            mrope_list.append(mrope_ordered[:, k])
            k += 1
        final_ids.append(ids_list[p])
        mrope_list.append(mp[:, mc])
        if is_pad_list[p]:
            feat_list.append(must_feature[mr])
            mr += 1
        mc += 1
    # trailing tokens (ins_pos == L) would be impossible: the client's
    # keep mask always keeps the trailing text columns, so every opt
    # token's ins_pos < L. Defensive: flush any stragglers.
    while k < n_merged:
        final_ids.append(pad_id)
        feat_list.append(feat_ordered[k])
        mrope_list.append(mrope_ordered[:, k])
        k += 1
    merged_feat = torch.stack(feat_list, dim=0)
    merged_mrope = torch.stack(mrope_list, dim=1)  # [3, n_total]

    ids_t = torch.tensor(final_ids, dtype=ids_1d.dtype
                          if hasattr(ids_1d, "dtype") else None)
    if hasattr(ids, "shape") and getattr(ids, "ndim", 1) == 2:
        # BUG X1: reshape to the MERGED length, not the must-only shape
        # (must bundle's [1, n_must] vs n_must + n_merged elements would
        # raise "shape invalid for input of size ..."); a 2-D client
        # contract means one row — the merged sequence is still one row.
        bundle["input_ids"] = ids_t.reshape(1, len(final_ids))
    else:
        bundle["input_ids"] = ids_t
    bundle["feature"] = merged_feat.contiguous()
    bundle["mrope_positions"] = merged_mrope.unsqueeze(1)  # [3, 1, n]
    bundle["mrope_position_delta"] = torch.tensor(
        [[int(merged_mrope.max()) + 1 - int(merged_mrope.shape[-1])]],
        dtype=mp.dtype)
    logger.info(
        "tiered merge: %d opt tokens merged from %d parts "
        "(%d late parts dropped)", n_merged, len(parts), late)
    return _normalize(bundle), stats


def _maybe_dequant_bundle_feature(bundle: dict) -> float:
    """Dequantize an fp8 edge bundle's feature tensor back to bf16.

    The terminal (clients/encoder_placement_client_qwen36.py
    _quantize_feature) may upload the (VidCom2-pruned) feature as
    float8_e4m3fn plus a per-tensor `feature_scale` float to halve the
    upload. Right after torch.load we reconstruct
    feature = scale * to(bf16)(feature_fp8) and pop the scale key, so
    everything downstream — engine-side hashing, shm transfer,
    embed_multimodal_inputs dtype checks — sees exactly the bf16
    tensors an unquantized run would have produced.

    Returns the dequant wall time (0.0 when the bundle is not
    quantized); raises ValueError when the bundle carries fp8 feature
    bytes this torch cannot cast back (old build without fp8 support).
    """
    import torch

    t_start = time.time()
    scale = bundle.pop("feature_scale", None)
    if scale is None:
        return 0.0
    feature = bundle.get("feature")
    if feature is None or not hasattr(feature, "dtype"):
        raise ValueError(
            "edge bundle has feature_scale but no feature tensor")
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None or str(feature.dtype) not in (
            "torch.float8_e4m3fn", "torch.float8_e4m3"):
        raise ValueError(
            f"edge bundle feature dtype {feature.dtype} with feature_scale "
            f"is not a supported fp8 tensor; rebuild the client or upgrade "
            f"torch")
    try:
        dequant = scale * feature.to(torch.bfloat16)
    except (NotImplementedError, RuntimeError, TypeError):
        # CPU fp8 casts are unsupported on some builds; hop to CUDA
        # (this sidecar always holds the GPU) and cast there.
        if not torch.cuda.is_available():
            raise ValueError(
                "cannot dequantize fp8 edge feature (CPU cast unsupported "
                "and no CUDA device available)") from None
        dequant = (scale * feature.to("cuda").to(torch.bfloat16)).cpu()
    bundle["feature"] = dequant.contiguous()
    return time.time() - t_start


def _prepare_edge_mode(entry: dict, prompt: str,
                       bundle_id: str | None = None):
    """edge mode: torch.save'd precomputed_embedding bundle -> engine kwargs.

    The bundle is a dict produced by the terminal
    (clients/encoder_placement_client_qwen36.py):
      {
        "format": "precomputed_embedding",
        "feature": <visual tower output [tokens, 2048] — single
                    merged segment, no deepstack on Qwen3.6>,
        ... whitelisted HF processor output (input_ids,
            video_grid_thw, second_per_grid_ts, mrope_positions,
            mrope_position_delta)
      }
    Per the official pattern (test_vlm_input_format.py
    test_accepts_precomputed_embeddings), the server passes BOTH the
    bundle's input_ids and the dict itself:
      engine.generate(input_ids=..., video_data=[bundle])
    so the terminal's tokenization is authoritative and the server-side
    processor takes the preprocessed fast path (base_processor.py).

    NOTE: the bundle MUST go through video_data, not image_data. The
    preprocessed fast path derives item modality from the receiving list
    (base_processor.py organize_results / load_mm_data fast path); a video
    bundle routed via image_data is tagged Modality.IMAGE, its offsets are
    then scanned against image_token_id, find no <|image_pad|> spans in a
    video prompt, and the precomputed embedding is silently dropped (LLM
    sees only constant video_pad text embeddings -> hallucination).

    Concurrency note: this runs OUTSIDE any engine lock — torch.load is
    per-request CPU work and the whole point of the concurrent variant
    is letting N of these overlap while the engine scheduler batches
    the GPU side.

    Optionally the terminal may upload the feature tensor as fp8 e4m3
    with a per-tensor `feature_scale` (bundle_quant='fp8' + vidcom2);
    _maybe_dequant_bundle_feature restores bf16 here (same off-lock
    concurrency contract) so the engine path is byte-identical to an
    unquantized run.

    TIERED bundles ("precomputed_embedding_tiered"): the terminal sent a
    must tier reliably and streams optional-tier tokens as best-effort
    parts over UDP (E2U datagrams reassembled by the OptUdpReceiver
    daemon thread). This is the MERGE SNAPSHOT: pop_parts()
    atomically takes whatever arrived so far — a part that has not
    arrived by now is dropped and never retransmitted (best-effort
    contract). _merge_tiered_parts rebuilds the retention-level
    sequence (see its docstring); the merged bundle is normalized back
    to the plain precomputed_embedding format before it reaches the
    engine.
    """
    import torch

    buf = io.BytesIO(entry["data"])
    bundle = torch.load(buf, map_location="cpu", weights_only=False)
    if (not isinstance(bundle, dict)
            or bundle.get("format") not in (
                "precomputed_embedding", "precomputed_embedding_tiered")):
        raise HTTPException(
            status_code=400, detail="edge bundle is not a precomputed_embedding dict")

    try:
        t_dequant = _maybe_dequant_bundle_feature(bundle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if t_dequant:
        logger.info("edge bundle dequant (fp8 -> bf16): %.3f s", t_dequant)

    merge_stats: dict = {}
    if (bundle.get("format") == "precomputed_embedding_tiered"
            and bundle_id is not None):
        t_merge = time.time()
        bundle, merge_stats = _merge_tiered_parts(bundle, bundle_id)
        merge_stats["merge_time"] = time.time() - t_merge
    if merge_stats:
        entry.setdefault("meta", {})["tiered"] = merge_stats

    input_ids = bundle.get("input_ids")
    if input_ids is not None:
        if hasattr(input_ids, "reshape"):
            input_ids = input_ids.reshape(-1).tolist()
        elif input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        else:
            input_ids = list(input_ids)
    if not input_ids:
        raise HTTPException(
            status_code=400, detail="edge bundle has no usable input_ids")
    return None, {"video_data": [bundle]}, input_ids


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")

    body = await request.json()
    if not body.get("stream", False):
        raise HTTPException(status_code=400, detail="only stream=true is supported")
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.0))
    mode = body.get("mode", "server")
    if mode not in ("server", "edge"):
        raise HTTPException(
            status_code=400,
            detail=f"invalid mode {mode!r}: must be 'server' or 'edge'")

    prompt = _extract_prompt(messages)
    kind, bundle_id = _parse_bundle_ref(messages)
    if kind != ("video" if mode == "server" else "edge"):
        raise HTTPException(
            status_code=400,
            detail=f"mode {mode!r} does not match uploaded bundle kind {kind!r}")
    try:
        entry = store.pop(bundle_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"bundle {bundle_id} not found")

    logger.info(
        f"chat/completions: mode={mode} bundle={bundle_id} "
        f"size={entry['size']} max_tokens={max_tokens} "
        f"active_generations={gate.active}")
    # Advisory: the engine samples with its STARTUP-time config; a body
    # carrying different values would silently mislabel the run.
    parse_body_sampling(body, _engine_frame_ratio, _engine_max_frames)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_name = body.get("model", _model_path)
    created = int(time.time())

    def sse_chunk(delta_content: str | None = None, finish: str | None = None,
                  usage: dict | None = None) -> str:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {} if delta_content is None else {"content": delta_content},
                "finish_reason": finish,
            }],
        }
        if usage is not None:
            chunk["usage"] = usage
        return f"data: {json.dumps(chunk)}\n\n"

    async def event_stream():
        stats: dict = {"mode": mode}
        # Register as an active generation BEFORE any engine work. If a
        # flush holds the gate, abort with an SSE error instead of racing
        # the flush coroutine inside the scheduler.
        if not gate.enter():
            err = json.dumps({"error": {
                "message": "flush_cache in progress; retry the request"}})
            yield f"data: {err}\n\n"
            return
        try:
            # t_prepare is per-request work (edge: torch.load; server:
            # bytes hand-off). The HEAVY part of the edge path (torch.load
            # of an uploaded bundle) would block the uvicorn loop for
            # hundreds of ms per request and, under N concurrent requests,
            # serialize them — exactly what the bridge is supposed to
            # remove. Run it in the loop's thread pool via
            # asyncio.to_thread (release the GIL during IO).
            t_prepare_start = time.time()
            if mode == "server":
                text, extra, input_ids_arg = await asyncio.to_thread(
                    _prepare_video_mode, entry, prompt)
            else:
                text, extra, input_ids_arg = await asyncio.to_thread(
                    _prepare_edge_mode, entry, prompt, bundle_id)
            t_prepare = time.time() - t_prepare_start
            stats["t_prepare"] = t_prepare
            # tiered merge bookkeeping (only present for tiered
            # edge bundles): parts/arrived/late/merge_time
            merge_meta = (entry.get("meta") or {}).get("tiered")
            if merge_meta:
                stats.update(merge_meta)

            sampling_params = {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
            }
            # Guard against zero-token completions: greedy (temperature
            # 0) decoding is not batch-invariant — under concurrent
            # prefills a user's first token can argmax to EOS, ending
            # the request with completion_tokens=0, no content chunk,
            # and a client-side ttft=None (shown as FAIL in the runner
            # despite ok=True). min_new_tokens>=1 suppresses EOS for
            # the first token(s) only; single-letter MCQ answers ("A")
            # still terminate at token 2. 0 disables the guard.
            min_new_tokens = int(body.get(
                "min_new_tokens", _default_min_new_tokens))
            if min_new_tokens > 0:
                sampling_params["min_new_tokens"] = min_new_tokens
            gen_kwargs = dict(sampling_params=sampling_params, **extra)
            if input_ids_arg is not None:
                # edge mode: terminal tokenization is authoritative
                # (test_vlm_input_format.py: generate(input_ids=...,
                # video_data=[precomputed dict])); no text prompt.
                gen_kwargs["input_ids"] = input_ids_arg
            else:
                gen_kwargs["prompt"] = text

            t_gen_start = time.time()
            # B1 asyncio bridge: the consumption below runs as ONE
            # coroutine ON THE ENGINE'S OWN LOOP (submitted via
            # bridge.submit / run_coroutine_threadsafe) and relays each
            # engine output dict through a queue.Queue; this
            # uvicorn-side async generator pulls with
            # await asyncio.to_thread(q.get). Two loops, one queue, no
            # locks:
            #   - N requests = N coroutines on the engine loop,
            #     submitted microseconds apart -> all reach the
            #     scheduler (tokenizer_manager zmq -> scheduler
            #     subprocess) before the first prefill finishes.
            #   - The uvicorn loop never blocks on engine work; the
            #     per-chunk thread hop is ~tens of microseconds.
            #   - engine.async_generate(stream=True) is a plain
            #     coroutine that RETURNS the async generator; the await
            #     unpacks it ON THE ENGINE LOOP, which is also where
            #     zmq subscriptions bind.
            out_q: queue.Queue = queue.Queue()

            async def _consume_on_engine_loop():
                # Runs ON the engine loop. Drives the stream to
                # completion and relays every output dict; the queue
                # carries ("out", dict) / ("end", None) / ("err", exc).
                try:
                    generator = await engine.async_generate(
                        **gen_kwargs, stream=True,
                    )
                    async for out in generator:
                        out_q.put(("out", out))
                    out_q.put(("end", None))
                except BaseException as exc:  # relay, don't crash the loop
                    out_q.put(("err", exc))

            consumer = bridge.submit(_consume_on_engine_loop())

            def _cancel_consumer():
                if not consumer.done():
                    consumer.cancel()

            first = True
            prev_text = ""
            n_tok = 0
            usage = {"prompt_tokens": None, "completion_tokens": 0}
            finish_reason: str | None = None
            try:
                while True:
                    # Block on the engine relay queue from a worker
                    # thread so this (uvicorn) coroutine stays reactive
                    # to client disconnects between chunks.
                    kind, payload = await asyncio.to_thread(
                        out_q.get, True, 3600.0)
                    if kind == "end":
                        break
                    if kind == "err":
                        raise RuntimeError(
                            f"engine stream failed: {payload}")
                    out = payload
                    text_piece = out.get("text") or ""
                    if first:
                        stats["server_ttft"] = time.time() - t_gen_start
                        first = False
                    if text_piece.startswith(prev_text):
                        delta = text_piece[len(prev_text):]
                    else:
                        # Defensive: non-prefix text (e.g. if the engine
                        # is ever switched to
                        # incremental_streaming_output) is already a
                        # delta; forward as-is.
                        delta = text_piece
                    prev_text = text_piece
                    if delta:
                        n_tok += 1
                        yield sse_chunk(delta_content=delta)
                    meta = out.get("meta_info") or {}
                    pt = meta.get("prompt_tokens")
                    if pt is not None:
                        usage["prompt_tokens"] = pt
                    # completion_tokens is cumulative across chunks; the
                    # last observed value is the authoritative total.
                    ct = meta.get("completion_tokens")
                    if ct is not None:
                        usage["completion_tokens"] = ct
                    # finish_reason likewise arrives on the final output;
                    # keep the last non-None value for diagnostics
                    # ("stop" vs "abort"/"length" distinguishes greedy-
                    # EOS zero-token finishes from scheduler retracts).
                    fr = meta.get("finish_reason")
                    if fr is not None:
                        finish_reason = fr
            finally:
                _cancel_consumer()
            if first:
                stats["server_ttft"] = None
            if usage["completion_tokens"] == 0:
                # Fallback when the engine never reported a count.
                usage["completion_tokens"] = n_tok
            stats["n_tokens"] = usage["completion_tokens"]
            stats["server_total"] = time.time() - t_gen_start
            stats["finish_reason"] = finish_reason
            if usage["completion_tokens"] == 0:
                # Zero-token completion: the stream produced NO content
                # chunk, so the client records ttft=None (runner shows
                # FAIL despite ok=True). finish_reason tells WHY:
                # "stop" = greedy first-token EOS (batch-invariance
                # noise; mitigate via min_new_tokens), "abort" =
                # scheduler retract/restart, "length" = min>max edge.
                logger.warning(
                    f"zero-token completion: request={request_id} "
                    f"mode={mode} finish_reason={finish_reason!r} "
                    f"prompt_tokens={usage['prompt_tokens']}")

            yield sse_chunk(finish="stop", usage=usage)
            yield f"data: {json.dumps({'id': request_id, 'engine_stats': stats})}\n\n"
        except Exception as exc:
            logger.exception("generation failed")
            err = json.dumps({"error": {"message": str(exc)}})
            yield f"data: {err}\n\n"
        finally:
            gate.exit()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/flush_cache")
async def flush_cache():
    """Flush radix cache + mm preprocess cache between benchmark waves.

    Drain-and-gate semantics (same as the Qwen3-VL concurrent
    sidecar): wait for in-flight generations to drain, close the gate
    so new generations abort with a clear SSE error instead of
    interleaving with the flush, submit the flush COROUTINE to the
    engine loop via the bridge, then reopen.

    Why tokenizer_manager.flush_cache and not engine.flush_cache: the
    sync wrapper drives the engine loop via run_until_complete —
    illegal from inside the running uvicorn loop ("Cannot run the
    event loop while another loop is running"). The underlying
    coroutine is submitted with run_coroutine_threadsafe and awaited
    here as a wrapped future; the gate guarantees no generation
    coroutine is concurrently mid-flight on the engine loop when the
    flush runs.

    The drain (gate.flush_close) runs on a worker thread via
    asyncio.to_thread: gate.exit() fires on THIS (uvicorn) loop, so
    blocking here would deadlock the very generations we are waiting
    for.

    The concurrent scenario calls this once per wave BEFORE releasing
    the client barrier, so in practice the gate is open and the call
    reduces to a plain flush. The gating only matters for accidental
    flushes during a wave — flushing while requests sit in the
    scheduler is undefined behaviour in sglang 0.5.20.
    """
    if engine is None or bridge is None:
        raise HTTPException(status_code=503, detail="engine not initialized")

    t0 = time.time()
    await asyncio.to_thread(gate.flush_close)
    try:
        await asyncio.wrap_future(
            bridge.submit(engine.tokenizer_manager.flush_cache()))
    finally:
        gate.flush_open()
    logger.info("flush_cache: radix + mm caches flushed "
                f"(gate drained in {time.time() - t0:.3f} s)")
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Concurrency + video-sampling observability for the harness."""
    return {
        "active_generations": gate.active,
        "bundles_in_store": len(store._store),
        "server_cpus": _applied_server_cpus or "unpinned",
        "decode_threads": os.environ.get("SGLANG_DECORD_NUM_THREADS")
                          or "decord-native",
        # effective intra-op thread cap for THIS process's CPU-side mm
        # preprocessing (None = native default / flag omitted). env var
        # readback mirrors decode_threads: shows what ACTUALLY applies
        # (a pre-set OMP_NUM_THREADS env would win over the flag).
        "preprocess_threads": os.environ.get("OMP_NUM_THREADS"),
        "video_sampling": {
            "frame_ratio": _engine_frame_ratio,
            "max_frames": _engine_max_frames or "sglang-default",
            "effective": describe_mm_process_config(
                build_mm_process_config(
                    _engine_frame_ratio, _engine_max_frames)),
        },
        # patch-4 switch state: "on"/"off" (None = ACTIVE sglang lacks
        # the parallel-video-preprocess patch — serial upstream only).
        "parallel_video_preprocess": _parallel_video_preprocess,
        # which decode backend the ACTIVE sglang selected at import
        # ("torchcodec" / "decord" / "unknown") — decode_threads above
        # caps BOTH, so naming the backend says where decode CPU lands.
        "video_decode_backend": _video_decode_backend(),
        # optional-tier UDP receiver (tiered mode): port + reassembly
        # health. port 0 = off (tiered clients fail fast on that).
        "opt_udp": (_opt_udp_receiver.stats()
                    if _opt_udp_receiver is not None
                    else {"port": 0}),
    }


@app.post("/video_preprocess_mode")
async def video_preprocess_mode(request: Request):
    """Flip the patch-4 parallel video preprocess switch AT RUNTIME.

    Why an endpoint and not a restart: the engine build takes minutes,
    and the point of the knob is SAME-PROCESS A/B comparison of the
    upstream-serial vs fan-out decode paths under identical model /
    cache / affinity state. The switch lives in the qwen_vl module
    global _PARALLEL_VIDEO_PREPROCESS and is read at EVERY preprocess
    call (tokenizer-manager, i.e. THIS process — no scheduler child
    involvement), so a flip lands on the next request.

    Body: {"mode": "on"|"off"} (also accepts {"parallel": true/false}
    for convenience). Returns the resulting state; 409 when the active
    sglang lacks patch 4 (refuses rather than pretending), 400 on a
    malformed body.
    """
    body = await request.json()
    mode = body.get("mode")
    if mode is None and "parallel" in body:
        mode = "on" if body.get("parallel") else "off"
    if mode not in ("on", "off"):
        raise HTTPException(
            status_code=400,
            detail=f'body must be {"mode": "on"|"off"} '
                   f'(or {"parallel": bool}); got {body!r}')

    qv = _qwen_vl_preprocess_module()
    if qv is None:
        raise HTTPException(
            status_code=409,
            detail="ACTIVE sglang lacks the 6G-Testbed patch 4 "
                   "(parallel video preprocess) — run "
                   "python aitestbed/server/patch_host_sglang.py in the "
                   "sidecar's env and RESTART (the module global is "
                   "absent, so there is nothing to flip)")

    qv._PARALLEL_VIDEO_PREPROCESS = (mode == "on")
    global _parallel_video_preprocess
    _parallel_video_preprocess = mode
    logger.info(
        f"video_preprocess_mode: parallel video preprocess -> {mode} "
        f"(takes effect on the next request; width when on = "
        f"mm_io_worker_num)")
    return {"status": "ok", "parallel_video_preprocess": mode}


def _apply_server_cpu_limit(spec: str):
    """Pin THIS process (and every child it spawns: the in-process
    Engine, its scheduler subprocess, uvicorn workers) to a shared
    CPU set — the server-arm CPU budget.

    Video decode + preprocessing is CPU-heavy and would otherwise grab
    every core of a many-core host, muting the server-vs-edge contrast
    the concurrent scenario measures. Called FIRST in main(), before
    the engine builds, so the scheduler subprocess (forked during
    Engine construction) inherits the affinity and the whole server
    lifetime honors the budget. JIT/warmup also runs inside it (a
    one-off cost, accepted by design).

    spec grammar: comma cpu list with ranges ("0-15", "0-7,32,40-41");
    the special value "auto-half" picks the FIRST HALF of the parent's
    affinity (K = ceil(n/2)) — a sane default that leaves the other
    half for the edge arm when both run on one host. Returns the
    applied spec string ("" when unpinned / unsupported platform).
    """
    if not spec or not str(spec).strip():
        return ""
    text = str(spec).strip().lower()
    try:
        own = os.sched_getaffinity(0)
    except AttributeError:
        logger.warning(
            "--server-cpus given but os.sched_getaffinity is unavailable "
            "on this platform; server CPU budget NOT applied")
        return ""
    if text == "auto-half":
        cpus = sorted(own)
        k = max(1, math.ceil(len(cpus) / 2))
        cpu_list = cpus[:k]
    else:
        cpu_list = set()
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, _, hi_s = part.partition("-")
                try:
                    lo, hi = int(lo_s), int(hi_s)
                except ValueError:
                    raise ValueError(
                        f"bad range {part!r} in --server-cpus {spec!r}")
                if lo < 0 or hi < lo:
                    raise ValueError(
                        f"bad range {part!r} in --server-cpus {spec!r}")
                cpu_list.update(range(lo, hi + 1))
            else:
                try:
                    n = int(part)
                except ValueError:
                    raise ValueError(
                        f"bad cpu id {part!r} in --server-cpus {spec!r}")
                if n < 0:
                    raise ValueError(
                        f"negative cpu id {n} in --server-cpus {spec!r}")
                cpu_list.add(n)
        cpu_list = sorted(cpu_list)
    if not cpu_list:
        return ""
    applied = ",".join(str(c) for c in cpu_list)
    try:
        os.sched_setaffinity(0, set(cpu_list))
    except OSError as exc:
        logger.warning(
            f"cannot pin server to cpus [{applied}] ({exc}); running "
            f"unpinned")
        return ""
    logger.info(f"server CPU budget: pinned to cpus [{applied}] "
                f"(scheduler subprocess inherits this affinity)")
    return applied


def main():
    global engine, _model_path, bridge, _applied_server_cpus
    parser = argparse.ArgumentParser(
        description="6G SGLang Engine Server (concurrent variant, Qwen3.6: "
                    "no engine_lock; continuous batching decides queueing). "
                    "Launch via start_sidecar_concurrent_qwen36.sh (loads "
                    "the sglang host patches + CUDA bootstrap this bare "
                    "launcher skips).")
    parser.add_argument("--model", default=os.environ.get(
        "SGLANG_ENGINE_QWEN36C_MODEL",
        str(REPO_ROOT / "aicodec" / "aicodec" / "models"
            / "Qwen3.6-35B-A3B")))
    parser.add_argument("--host", default=os.environ.get(
        "SGLANG_ENGINE_QWEN36C_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(
        os.environ.get("SGLANG_ENGINE_QWEN36C_PORT", "8014")),
                        help="default 8014 (serial Qwen3.6 sidecar uses "
                             "8004; Qwen3-VL concurrent 8012; run side "
                             "by side on different ports)")
    parser.add_argument("--opt-udp-port", type=int, default=int(
        os.environ.get("SGLANG_ENGINE_QWEN36C_OPT_UDP_PORT", "8015")),
        metavar="N",
        help="UDP port for the tiered optional-tier second wave (E2U "
             "datagrams; reassembled parts feed the same BundleStore "
             "merge path the former kind=edge2 HTTP PUT used). MUST be "
             "bound before uvicorn so tiered clients see a live port "
             "in the must PUT response. 0 = OFF (tiered clients fail "
             "fast with opt_udp_port=0). Default 8015.")
    parser.add_argument("--opt-udp-workers", type=int, default=int(
        os.environ.get("SGLANG_ENGINE_QWEN36C_OPT_UDP_WORKERS", "4")),
        metavar="N",
        help="Number of receive sockets/threads for the optional-tier "
             "UDP port. N>1 uses SO_REUSEPORT on the SAME port: the "
             "kernel hashes each client flow onto one socket, so a "
             "single user's wave stays on one worker (order preserved) "
             "while concurrent users spread across workers — true "
             "concurrent receive, and aggregate kernel SO_RCVBUF "
             "scales with N. netem/pcap filters match the PORT and are "
             "unaffected. Falls back to a single socket when "
             "SO_REUSEPORT is unavailable. Default 4. (Multi-worker "
             "receive lives HERE; the serial sidecar is single-thread "
             "by design.)")
    parser.add_argument("--opt-udp-reassembly-cap", type=int, default=int(
        os.environ.get("SGLANG_ENGINE_QWEN36C_OPT_UDP_REASSEMBLY_CAP",
                       "256")), metavar="N",
        help="Max concurrent part-reassembly keys for the optional-tier "
             "UDP receiver (FIFO overflow drops the oldest incomplete "
             "part, counted in parts_overflow_dropped). Raise for "
             "high-concurrency waves: rule of thumb max concurrent "
             "users x parts in flight. Default 256.")
    parser.add_argument("--opt-merge-drain-ms", type=float, default=float(
        os.environ.get("SGLANG_ENGINE_QWEN36C_OPT_MERGE_DRAIN_MS", "50")),
        metavar="MS",
        help="Merge-snapshot drain budget in ms: before pop_parts(), "
             "the merge waits (bounded) for datagrams already in the "
             "kernel SO_RCVBUF / mid-reassembly to land so the snapshot "
             "does not understate the merge (a measurement artifact, "
             "not real loss). Ends early after a 3ms quiet window. "
             "0 = pure snapshot semantics (no drain). Default 50.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85,
                        help="KV-cache static memory fraction (serial "
                             "Qwen3.6 sidecar default 0.85; concurrent "
                             "batching may need headroom tuning — the 35B "
                             "fp8 weights leave more KV room than bf16, "
                             "lower it on GPUs that must also hold edge "
                             "towers)")
    parser.add_argument("--max-running-requests", type=int, default=None,
                        help="scheduler running-request cap (max requests "
                             "batched on the GPU at once). Omit/None = "
                             "sglang default (computed from KV-cache "
                             "capacity); set e.g. 4 to force queueing "
                             "above 4 in-flight requests")
    parser.add_argument("--chunked-prefill-size", type=int, default=None,
                        metavar="N",
                        help="max tokens in ONE prefill chunk (official "
                             "sglang 0.5.20 ServerArgs field, no source "
                             "patching). Omit/None = sglang default "
                             "(resolves to 8192 on this box). -1 disables "
                             "chunked prefill (whole prompt in one "
                             "forward). Larger values shorten TTFT of the "
                             "first request but stall in-flight decodes "
                             "longer during each prefill chunk (unless "
                             "--enable-mixed-chunk); with this workload's "
                             "~12k-token prompts, raising it above the "
                             "prompt size can let queued requests share "
                             "one prefill batch (#new-seq > 1)")
    parser.add_argument("--enable-mixed-chunk", action="store_true",
                        default=None,
                        help="mix decode requests into prefill chunks so "
                             "in-flight decodes do not stall while a long "
                             "chunked prefill runs (official sglang 0.5.20 "
                             "ServerArgs field). Default off (sglang "
                             "default); note backends are not bit-identical "
                             "across scheduler geometry — keep comparisons "
                             "within one geometry")
    parser.add_argument("--prefill-max-requests", type=int, default=None,
                        metavar="N",
                        help="max requests admitted into ONE prefill batch "
                             "(official sglang 0.5.20 ServerArgs field). "
                             "Omit/None = unlimited (sglang default); the "
                             "token budget (chunked_prefill_size / "
                             "max_prefill_tokens) usually binds first with "
                             "video prompts this large")
    parser.add_argument("--max-prefill-tokens", type=int, default=None,
                        help="TOTAL token budget of ONE prefill batch "
                             "(official sglang 0.5.20 ServerArgs field, "
                             "default 16384, runtime-capped at >= model "
                             "context length). Together with "
                             "--chunked-prefill-size it decides whether "
                             "MULTIPLE queued requests share one prefill "
                             "forward (#new-seq > 1): with ~12k-token "
                             "prompts, two requests co-batch only when "
                             "BOTH budgets are >= ~2x the prompt size. "
                             "Raising it does not reduce total GPU work "
                             "(prefill is compute-bound) — it changes the "
                             "queueing texture; compare runs within one "
                             "fixed geometry only")
    parser.add_argument("--enable-radix-cache", action="store_true",
                        help="re-enable sglang radix cache (OFF by default "
                             "in this benchmark: N concurrent users send "
                             "the same video, so with the cache ON the "
                             "first user's prefill warms the shared video "
                             "tokens and later users hit high cached-token "
                             "ratios — TTFT collapses toward decode-only "
                             "and the cold-prefill concurrency under test "
                             "disappears; the per-user prompt tag keeps "
                             "text prefixes disjoint either way)")
    parser.add_argument("--mm-io-worker-num", type=int, default=None,
                        metavar="N",
                        help="width of the mm I/O thread pool running video "
                             "decode + sample + resize (official sglang "
                             "0.5.20 ServerArgs field, Engine kwargs "
                             "passthrough). N = at most N of the wave's "
                             "videos decode in parallel across requests; "
                             "the rest queue in preprocessing before "
                             "reaching the scheduler — the server-arm "
                             "decode budget under concurrency. Omit/None = "
                             "model default (qwen_vl: 16). SGLANG_IO_WORKERS "
                             "env still applies when this is omitted/0. "
                             "Pair with --server-cpus for a hard affinity "
                             "cap")
    parser.add_argument("--mm-processor-worker-num", type=int, default=None,
                        metavar="N",
                        help="width of the multimodal processor pool (HF "
                             "processor calls: normalize / tensorize; "
                             "official sglang 0.5.20 ServerArgs field). "
                             "1 = synchronous processing (no pool). "
                             "Omit/None = model default (qwen_vl: 2)")
    parser.add_argument("--min-new-tokens", type=int, default=1, metavar="N",
                        help="suppress EOS for the first N generated tokens "
                             "of every request (default 1). Guards against "
                             "zero-token completions: greedy decoding is "
                             "not batch-invariant, so under concurrent "
                             "prefills a user's first token can argmax to "
                             "EOS — the request then finishes ok with NO "
                             "content chunk and the client records "
                             "ttft=None (shown as FAIL despite ok=True). "
                             "0 disables the guard. Requests can override "
                             "per-call via body min_new_tokens.")
    parser.add_argument("--max-bundles", type=int, default=64,
                        help="BundleStore capacity: uploaded bundles held "
                             "before LRU eviction. MUST be >= the wave size "
                             "N (a wave uploads N bundles before the chat "
                             "requests pop them; smaller N would evict live "
                             "bundles -> 404). Default 64.")
    parser.add_argument("--server-cpus", default="auto-half", metavar="SPEC",
                        help="server-arm CPU budget: pin this server and its "
                             "scheduler subprocess to a shared cpu set so "
                             "CPU-bound video decode cannot grab the whole "
                             "host (pairs with the runner's "
                             "--edge-cpus-per-worker; without BOTH caps the "
                             "concurrent server-vs-edge contrast is muted "
                             "on a many-core host). SPEC: comma list with "
                             "ranges ('0-15,32') or 'auto-half' = first "
                             "half of the available cores (default). '' "
                             "disables. Applied BEFORE the engine builds, "
                             "so JIT/warmup pays the budget too (one-off "
                             "cost)")
    parser.add_argument("--frame-ratio", type=float, default=0.0,
                        metavar="R",
                        help=mm_process_config_help(
                            "--frame-ratio R (0.25 = keep 25% of frames)"))
    parser.add_argument("--max-frames", type=int, default=0, metavar="N",
                        help="cap on sampled frames for BOTH sampling modes "
                             "(see --frame-ratio). 0 = sglang default 768 "
                             "(FPS_MAX_FRAMES); an explicit 768 equals the "
                             "default. Native sglang 0.5.20 whitelist key — "
                             "works WITHOUT any sglang patch. ENGINE-wide "
                             "via mm_process_config: restart to change; "
                             "/stats shows the effective values")
    parser.add_argument("--decode-threads", type=int, default=0, metavar="N",
                        help="cap on decord's INTERNAL decode thread pool "
                             "(applied via the 6G-Testbed sglang patch; "
                             "requires re-running the patcher when the "
                             "installed sglang lacks it). decord sizes its "
                             "pool by HOST core count regardless of the "
                             "--server-cpus affinity pin's core BUDGET and "
                             "of OMP/torch/mm-io thread knobs — this is the "
                             "only true per-decode thread cap. Set it to "
                             "match the edge terminal's decode threads for "
                             "symmetric server-vs-edge CPU comparisons. "
                             "0 = decord native default (no cap). Exported "
                             "as SGLANG_DECORD_NUM_THREADS for the engine "
                             "process and inherited by scheduler children.")
    parser.add_argument("--parallel-video-preprocess",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="STARTUP default for the 6G-Testbed patch-4 "
                             "switch: fan concurrent videos' preprocessing "
                             "(decode + resize + pin) out over the mm-io "
                             "pool instead of upstream's serial "
                             "event-loop staircase (the measured cause of "
                             "equal TTFT across heterogeneous videos). "
                             "DEFAULT ON since the v2 fix (single-video "
                             "requests hop too — the executor hop, not "
                             "gather(), creates the cross-request "
                             "overlap). Use --no-parallel-video-preprocess "
                             "for the upstream serial baseline (A/B "
                             "OFF side; bit-identical to unpatched "
                             "sglang). Flippable AT RUNTIME via "
                             "POST /video_preprocess_mode (no engine "
                             "restart); width when ON = "
                             "--mm-io-worker-num; per-video codec threads "
                             "still capped by --decode-threads. Requires "
                             "the sglang patch (applied automatically at "
                             "startup; /stats shows "
                               "parallel_video_preprocess=null when the "
                               "active sglang lacks it).")
    parser.add_argument("--preprocess-threads", type=int, default=0,
                        metavar="N",
                        help="cap the torch/OMP intra-op thread count for "
                             "CPU-side mm preprocessing in THIS process "
                             "(resize / HF processor / edge-mode torch.load+"
                             "dequant): exports OMP_NUM_THREADS / MKL_NUM_"
                             "THREADS / OPENBLAS_NUM_THREADS (setdefault — a "
                             "pre-set env var wins, mirroring --decode-"
                             "threads) BEFORE the engine (and thus torch) is "
                             "imported, plus torch.set_num_threads(N). "
                             "Without it the main process sizes its pools by "
                             "the --server-cpus core BUDGET (e.g. 32 threads "
                             "on a 32-cpu pin), while the edge terminal caps "
                             "its own pools by its per-worker pin — the "
                             "resize-stage thread asymmetry behind 'server "
                             "decodes faster'. Set it to the edge terminals' "
                             "--edge-cpus-per-worker for symmetric server-vs-"
                             "edge CPU comparisons. Scheduler children are "
                             "unaffected (model_runner.py pins them to 1 "
                             "thread regardless). 0 = native default (no "
                             "cap; backwards-compatible).")
    args = parser.parse_args()

    _applied_server_cpus = _apply_server_cpu_limit(args.server_cpus)

    _set_store_capacity(args.max_bundles)

    global _default_min_new_tokens
    _default_min_new_tokens = max(0, args.min_new_tokens)

    global _engine_frame_ratio, _engine_max_frames
    _engine_frame_ratio = args.frame_ratio or None
    _engine_max_frames = args.max_frames or None
    # decord thread cap: must be set BEFORE the first VideoReader is built
    # (env read at construction time by the patched video_decoder.py).
    # A pre-existing SGLANG_DECORD_NUM_THREADS wins (explicit env override).
    if args.decode_threads and args.decode_threads > 0:
        os.environ.setdefault("SGLANG_DECORD_NUM_THREADS",
                              str(args.decode_threads))
    # CPU-side preprocess thread cap (resize/processor/dequant intra-op
    # pools): must be applied BEFORE the engine import pulls torch in —
    # the env vars are read at native-lib init (once per process) and
    # serve as torch's set_num_threads default. setdefault per var: a
    # pre-set env value wins (explicit override, mirroring --decode-
    # threads). torch.set_num_threads is the belt-and-braces in-process
    # cap for the case where torch was already initialized (it never is
    # at this point in this module, but the direct-python start path
    # deserves the same safety as the shell one).
    if args.preprocess_threads and args.preprocess_threads > 0:
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(_var, str(args.preprocess_threads))
        import torch

        torch.set_num_threads(args.preprocess_threads)
        logger.info(
            "preprocess threads: capped to %d "
            "(OMP/MKL/OPENBLAS_NUM_THREADS + torch.set_num_threads; "
            "covers resize/processor/edge-dequant in this process)",
            args.preprocess_threads)
    # Patch-4 startup default: set the env BEFORE the engine imports
    # qwen_vl (the module global initializes from it). A pre-existing
    # env value wins (explicit override, mirroring --decode-threads);
    # the RUNTIME flip (POST /video_preprocess_mode) does not use the
    # env at all — it writes the module global directly.
    if args.parallel_video_preprocess:
        os.environ.setdefault("SGLANG_QWEN_PARALLEL_VIDEO_PREPROCESS", "1")
    try:
        mm_cfg = build_mm_process_config(
            args.frame_ratio, args.max_frames)
    except ValueError as exc:
        parser.error(str(exc))

    # Auto-load the sglang disk patch (same patcher the
    # start_sidecar_qwen36.sh bootstrap runs) BEFORE any sglang import /
    # the canary below — so a direct ``python
    # sglang_engine_server_concurrent_qwen36.py`` start is as protected
    # as the shell-wrapped one (fail = abort, like ``patcher || die``).
    ensure_sglang_patches(reason="sglang_engine_server_concurrent_qwen36 "
                                 "startup")

    # Fail fast (BEFORE the multi-minute engine build) when a frame
    # ratio was requested but the active sglang lacks the patch —
    # otherwise the benchmark would silently sample at 2fps.
    verify_frame_ratio_patch(active=bool(args.frame_ratio))

    _model_path = args.model
    logger.info(f"Starting SGLang Engine (concurrent Qwen3.6, "
                f"model={args.model} "
                f"mm_io_worker_num={args.mm_io_worker_num} "
                f"mm_processor_worker_num={args.mm_processor_worker_num} "
                f"decode_threads={os.environ.get('SGLANG_DECORD_NUM_THREADS') or 'native'} "
                f"preprocess_threads={os.environ.get('OMP_NUM_THREADS') or 'native'}) ...")
    engine = _build_engine(
        args.model, args.device,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        chunked_prefill_size=args.chunked_prefill_size,
        enable_mixed_chunk=args.enable_mixed_chunk,
        prefill_max_requests=args.prefill_max_requests,
        max_prefill_tokens=args.max_prefill_tokens,
        enable_radix_cache=args.enable_radix_cache,
        mm_io_worker_num=args.mm_io_worker_num,
        mm_processor_worker_num=args.mm_processor_worker_num,
        mm_process_config=mm_cfg,
    )

    # B1 bridge: assert the engine surface this server needs BEFORE
    # accepting traffic, then park the engine's private loop on its
    # dedicated thread. Fail fast with a precise message instead of
    # 500ing on the first request.
    if not hasattr(engine, "async_generate"):
        raise RuntimeError(
            "sglang Engine has no async_generate — 0.5.20 expected; "
            "the asyncio bridge drives streams via "
            "engine.async_generate(stream=True). If this sglang "
            "renamed it, adapt _consume_on_engine_loop in "
            "sglang_engine_server_concurrent_qwen36.py.")
    if not hasattr(engine, "tokenizer_manager"):
        raise RuntimeError(
            "sglang Engine has no tokenizer_manager — flush_cache "
            "submits engine.tokenizer_manager.flush_cache() to the "
            "engine loop. Layout changed?")
    bridge = EngineLoopBridge(engine)
    bridge.start()

    # Patch-4 switch: sync the echo state now that the engine build has
    # imported the (patched) qwen_vl — the module global initialized
    # from the env we set above. None => active sglang lacks patch 4
    # (the flip endpoint will refuse with 409; running continues with
    # upstream serial preprocessing).
    _sync_parallel_video_preprocess_flag()
    if args.parallel_video_preprocess and _parallel_video_preprocess is None:
        logger.warning(
            "--parallel-video-preprocess given but the ACTIVE sglang "
            "lacks the 6G-Testbed patch 4 — running with upstream "
            "SERIAL video preprocessing (check patch_host_sglang.py "
            "output above; POST /video_preprocess_mode will 409)")

    logger.info("SGLang Engine ready (concurrent Qwen3.6 variant, no "
                f"engine_lock; max_bundles={args.max_bundles}; "
                f"max_running_requests="
                f"{args.max_running_requests if args.max_running_requests is not None else 'sglang-default'}; "
                f"chunked_prefill_size="
                f"{args.chunked_prefill_size if args.chunked_prefill_size is not None else 'sglang-default'}; "
                f"enable_mixed_chunk="
                f"{args.enable_mixed_chunk if args.enable_mixed_chunk is not None else 'sglang-default'}; "
                f"prefill_max_requests="
                f"{args.prefill_max_requests if args.prefill_max_requests is not None else 'sglang-default'}; "
                f"max_prefill_tokens="
                f"{args.max_prefill_tokens if args.max_prefill_tokens is not None else 'sglang-default'}; "
                f"mm_io_worker_num="
                f"{args.mm_io_worker_num if args.mm_io_worker_num is not None else 'sglang-default'}; "
                f"mm_processor_worker_num="
                f"{args.mm_processor_worker_num if args.mm_processor_worker_num is not None else 'sglang-default'}; "
                f"radix_cache={'ON' if args.enable_radix_cache else 'OFF'}; "
                f"min_new_tokens={_default_min_new_tokens}; "
                f"engine loop parked on dedicated thread; server_cpus="
                f"{_applied_server_cpus or 'unpinned'})")
    logger.info(format_log_line(
        args.frame_ratio or None, args.max_frames or None,
        json.dumps(mm_cfg) if mm_cfg else None))
    # Optional-tier UDP receiver: bind BEFORE uvicorn.run so the port
    # is guaranteed live when the first must PUT response reports it
    # (the client fail-fasts on opt_udp_port=0 in a tiered run).
    global _opt_udp_receiver, _opt_merge_drain_ms
    _opt_merge_drain_ms = float(args.opt_merge_drain_ms)
    if args.opt_udp_port > 0:
        _opt_udp_receiver = OptUdpReceiver(
            store, args.opt_udp_port, host="0.0.0.0",
            workers=args.opt_udp_workers,
            reassembly_cap=args.opt_udp_reassembly_cap)
        if not _opt_udp_receiver.start():
            _opt_udp_receiver = None
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
