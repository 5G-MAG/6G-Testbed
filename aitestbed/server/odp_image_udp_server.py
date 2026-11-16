#!/usr/bin/env python3
"""One-D-Piece image cloud server over a LOSSY UDP transport.

Sibling of ``odp_image_http_server.py``. The HTTP/TCP link cannot express
packet loss: TCP retransmits every lost segment, so a netem ``loss``
profile only ever shows up as added LATENCY and the token stream handed to
the decoder is always bit-perfect. Over UDP a dropped datagram is a
genuinely missing slice of the 12-bit token stream, which is the
degradation the offline study models analytically via
``run_mmbench.sh --keep-ratio/--drop-mode``.

How loss maps onto tokens
-------------------------
Reassembly is delegated to :func:`aicodec.models.one_d_piece.odp_udp_wire.odp_recover_tokens`
(vendored from ``aicodec.odp_codec`` so this UDP path never imports
aiortc), the SAME reassembly logic the WebRTC/RTP link uses, so UDP results
stay comparable with the RTP path. Its contract (gaps -> zeros, recovered
length = ``max(pos + size)``) gives two distinct regimes worth keeping in
mind when reading results:

* **a lost TAIL datagram shortens the frame** -> fewer tokens reach the
  decoder, i.e. the analogue of ``--drop-mode tail``;
* **a lost MIDDLE datagram keeps the token count** but zero-fills that
  slice -> tokens are *corrupted to id 0*, not dropped. This is strictly
  harsher than the offline "keep the first k tokens" condition, so do not
  expect UDP middle-loss points to land exactly on the offline curve.

Both are reported per image (``tokens_expected`` vs ``tokens_recovered``
plus ``chunks_lost``) so the two regimes can be separated in analysis.

Protocol
--------
See ``aicodec/aicodec/models/one_d_piece/odp_udp_wire.py`` for the datagram layout, shared with the
two UDP clients so the three files cannot drift apart.

Because UDP has no connection teardown, an image whose last datagram is
lost would otherwise wait forever. A reaper thread therefore flushes any
image that has been idle for ``--reassembly-timeout`` seconds, handing
whatever survived to the decoder. That is what turns loss into *image
degradation* instead of a hung session.

Run:
    python server/odp_image_udp_server.py --host 0.0.0.0 --port 1237 \\
        --dump-recon --dump-recon-dir results/odp_image_recon/udp_lossy \\
        --report-dir results/odp_image_reports --fresh

The server loads the configured MMBench source manifest, then writes matching
original rows for successfully saved reconstructions to ``manifest.jsonl``.
Transport details remain in ``udp_rx_manifest.jsonl``. The reconstructed dir
can then be consumed directly by
``run_mmbench.sh --data-dir <dir> --tag <cond>``.
"""

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Bootstrap the sibling aicodec package relative to this file's location
# (same convention as odp_image_http_server.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AICODEC_PATH = _REPO_ROOT / "aicodec"
if str(_AICODEC_PATH) not in sys.path:
    sys.path.insert(0, str(_AICODEC_PATH))

# the UDP wire format (incl. token reassembly) lives in odp_udp_wire; it
# has no aiortc dependency, unlike odp_codec.py
import aicodec.models.one_d_piece  # noqa: F401  (registers "one_d_piece")
from aicodec.models import get_bridge
from aicodec.models.one_d_piece import odp_udp_wire as wire
from aicodec.models.one_d_piece.odp_udp_wire import odp_recover_tokens, packed_size

logger = logging.getLogger("odp_image_udp_server")

# The source MMBench questions/answers used to build per-run eval manifests.
# Resolved relative to the repo root by default (self-contained migration);
# env ODP_MMBENCH_MANIFEST overrides for datasets staged elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
MMBENCH_QUESTION_MANIFEST = Path(
    os.environ.get(
        "ODP_MMBENCH_MANIFEST",
        str(_REPO_ROOT / "aicodec" / "aicodec" / "dataset" / "MMBench" / "MMBench_DEV_EN" / "manifest.jsonl"),
    )
).expanduser()


def setup_logger(name: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger(name).setLevel(logging.INFO)


class ODPImageUDPServer:
    """Token sink + reconstructor for UDP-delivered ODP image payloads."""

    def __init__(self, host: str = "0.0.0.0", port: int = 1237,
                 model: str = "one_d_piece", num_tokens: int = 256,
                 dump_recon: bool = True,
                 dump_recon_dir: str = "results/odp_image_recon",
                 report_dir: str = "results/odp_image_reports",
                 network_profile: str = "no_emulation",
                 fresh_recon: bool = False,
                 reassembly_timeout: float = 0.2,
                 recv_buffer_bytes: int = 4 << 20):
        self.host = host
        self.port = port
        self.model = model
        self.num_tokens = num_tokens
        self.dump_recon = dump_recon
        self.dump_recon_dir = Path(dump_recon_dir)
        self.report_dir = Path(report_dir)
        self.network_profile = network_profile
        self.fresh_recon = fresh_recon
        self.reassembly_timeout = reassembly_timeout
        self.recv_buffer_bytes = recv_buffer_bytes

        self._bridge = None
        self._bridge_lock = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._recon_pool = ThreadPoolExecutor(max_workers=4,
                                              thread_name_prefix="recon")
        # Original MMBench rows keyed by dataset index. When configured, the
        # eval manifest is populated only for images that were actually saved.
        self._question_by_index = {}
        self._question_indices_written = set()
        #: index -> partial image state
        self._pending = {}
        self._stop = threading.Event()
        self._report_written = False

        self.t_start = time.time()
        self.stats = {
            "datagrams_received": 0,
            "datagrams_corrupt": 0,
            "bytes_received": 0,
            "images_received": 0,      # images flushed to the decoder
            "images_complete": 0,      # all chunks arrived
            "images_partial": 0,       # flushed with holes / short tail
            "chunks_expected_total": 0,
            "chunks_lost_total": 0,
            "tokens_expected_total": 0,
            "tokens_recovered_total": 0,
            "recon_saved": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------
    # bridge (lazy: only pays the GPU load when recon is requested)
    # ------------------------------------------------------------------
    def bridge(self):
        with self._bridge_lock:
            if self._bridge is None:
                logger.info("[Server] loading %s bridge (dec) for recon...",
                            self.model)
                self._bridge = get_bridge(self.model, "dec")
                logger.info("[Server] bridge ready")
            return self._bridge

    def _bump(self, key: str, n: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] += n

    # ------------------------------------------------------------------
    # reassembly
    # ------------------------------------------------------------------
    def _on_datagram(self, data: bytes) -> None:
        """Accept one datagram; flush the image as soon as it is complete."""
        self._bump("datagrams_received")
        self._bump("bytes_received", len(data))

        kind = wire.parse_control(data)
        if kind is not None:
            if kind == wire.CTL_SESSION_COMPLETE:
                logger.info("[Server] session-complete control received")
                self.flush_all(reason="session_complete")
                self.save_report()
            return

        try:
            pkt = wire.parse_datagram(data)
        except ValueError as e:
            self._bump("datagrams_corrupt")
            logger.debug("[Server] corrupt datagram: %s", e)
            return

        now = time.time()
        index = pkt["index"]
        flush_me = None
        with self._pending_lock:
            entry = self._pending.get(index)
            if entry is None:
                entry = {
                    "name": pkt["name"],
                    "frame_bytes": pkt["frame_bytes"],
                    "nchunks": pkt["nchunks"],
                    "chunks": {},
                    "t_first": now,
                    "t_last": now,
                }
                self._pending[index] = entry
            entry["t_last"] = now
            # keyed by pos so a duplicated datagram cannot double-count
            entry["chunks"][pkt["pos"]] = (pkt["token_header"], pkt["chunk"])
            if len(entry["chunks"]) >= entry["nchunks"]:
                flush_me = self._pending.pop(index)

        if flush_me is not None:
            self._flush_image(index, flush_me, complete=True)

    def _flush_image(self, index: int, entry: dict, complete: bool) -> None:
        """Hand whatever chunks survived to the decoder and record the row."""
        name = entry["name"]
        nchunks = entry["nchunks"]
        got = len(entry["chunks"])
        lost = max(0, nchunks - got)

        # concatenate in ascending byte offset; odp_recover_tokens zero-fills
        # the holes and derives the true length from max(pos + size)
        frame_data = b"".join(
            entry["chunks"][pos][0] + entry["chunks"][pos][1]
            for pos in sorted(entry["chunks"]))

        try:
            picture_id, timestamp, tokens = odp_recover_tokens(frame_data)
        except Exception as e:
            self._bump("errors")
            logger.warning("[Server] recover failed for %s: %s", name, e)
            return

        tokens_expected = entry["frame_bytes"] * 8 // wire.ODP_TOKEN_BITS
        n = len(tokens)

        self._bump("images_received")
        self._bump("images_complete" if complete and lost == 0
                   else "images_partial")
        self._bump("chunks_expected_total", nchunks)
        self._bump("chunks_lost_total", lost)
        self._bump("tokens_expected_total", tokens_expected)
        self._bump("tokens_recovered_total", n)

        if lost:
            logger.info(
                "[Server] image %s: %d/%d chunks, tokens %d/%d (loss %.1f%%)",
                name, got, nchunks, n, tokens_expected,
                100.0 * lost / max(1, nchunks))

        recon_file = None
        if self.dump_recon:
            try:
                recon_file = self._recon_pool.submit(
                    self._save_recon, name, tokens).result()
            except Exception as e:
                self._bump("errors")
                logger.warning("[Server] recon failed for %s: %s", name, e)

        self._append_manifest({
            "index": index,
            "name": name,
            "picture_id": picture_id,
            "timestamp": timestamp,
            "chunks_expected": nchunks,
            "chunks_received": got,
            "chunks_lost": lost,
            "tokens_expected": tokens_expected,
            "tokens_recovered": n,
            "complete": bool(complete and lost == 0),
            "flush_reason": "complete" if complete else "timeout",
            "recon_file": recon_file,
            "network_profile": self.network_profile,
        })

    # ------------------------------------------------------------------
    # recon + manifest + report
    # ------------------------------------------------------------------
    def _save_recon(self, name: str, tokens):
        """Decode tokens to PNG.

        Naming follows the OFFLINE recon convention (``<index>.png``, e.g.
        ``241.png``) rather than the HTTP server's ``img_00241_241.png``:
        the dataset ships ``241.jpg``, so ``Path(name).stem`` already IS the
        MMBench index. That makes this folder structurally identical to the
        existing ``recon/MMBench_DEV_EN/<cond>/`` dirs and hits
        ``mmbench_eval.find_image``'s primary ``<index>.png`` pattern.
        """
        last_err = None
        for attempt in (1, 2):
            try:
                recon = self.bridge().detokenize(tokens)
                self.dump_recon_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(name).stem or "image"
                fname = f"{stem}.png"
                recon.save(self.dump_recon_dir / fname)
                self._bump("recon_saved")
                return fname
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("[Server] recon save failed for %s "
                               "(attempt %d/2): %r", name, attempt, e,
                               exc_info=True)
                time.sleep(1.0)
        self._bump("errors")
        logger.error("[Server] recon save gave up for %s: %r", name, last_err)
        return None

    def _append_manifest(self, entry: dict) -> None:
        """Append one RX row and, when possible, its MMBench question row.

        Deliberately NOT named ``manifest.jsonl``: ``mmbench_eval.py``
        resolves ``<data-dir>/manifest.jsonl`` as the QUESTION manifest
        (index/question/choices/answer). Link metrics stay in
        ``udp_rx_manifest.jsonl``; the question file contains only original
        MMBench rows whose reconstruction PNG was successfully saved.
        """
        try:
            self.dump_recon_dir.mkdir(parents=True, exist_ok=True)
            with self._manifest_lock:
                with open(self.dump_recon_dir / "udp_rx_manifest.jsonl", "a",
                          encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                recon_file = entry.get("recon_file")
                if not recon_file or not self._question_by_index:
                    return

                index = str(entry.get("index", "")).strip()
                question_row = self._question_by_index.get(index)
                if question_row is None:
                    logger.warning(
                        "[Server] no MMBench question for received index %s",
                        index)
                    return
                if index in self._question_indices_written:
                    return
                if not (self.dump_recon_dir / str(recon_file)).is_file():
                    logger.warning(
                        "[Server] skip MMBench row for %s: recon image missing (%s)",
                        index, recon_file)
                    return

                with open(self.dump_recon_dir / "manifest.jsonl", "a",
                          encoding="utf-8") as f:
                    f.write(json.dumps(question_row, ensure_ascii=False) + "\n")
                self._question_indices_written.add(index)
        except Exception as e:
            logger.warning("[Server] manifest append failed: %s", e)

    def install_question_manifest(self, src: str) -> None:
        """Load the dataset MCQ manifest for index-based RX filtering.

        The full source manifest is not copied. ``manifest.jsonl`` is built
        from original rows as their matching reconstruction PNGs are saved.
        """
        try:
            src_path = Path(src)
            if not src_path.exists():
                logger.warning("[Server] question manifest not found: %s", src)
                return
            rows = {}
            with src_path.open("r", encoding="utf-8-sig") as f:
                for line_number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    index = str(row["index"]).strip()
                    rows[index] = row
            self._question_by_index = rows
            logger.info("[Server] loaded %d MMBench question rows from %s",
                        len(rows), src_path)
        except Exception as e:
            logger.warning("[Server] question manifest load failed: %s", e)

    def _rebuild_question_manifest(self) -> None:
        """Rebuild the eval manifest from existing RX rows and saved PNGs."""
        rx_path = self.dump_recon_dir / "udp_rx_manifest.jsonl"
        eligible = set()
        if rx_path.is_file():
            try:
                with rx_path.open("r", encoding="utf-8-sig") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        index = str(entry.get("index", "")).strip()
                        recon_file = entry.get("recon_file")
                        if (index in self._question_by_index and recon_file and
                                (self.dump_recon_dir / str(recon_file)).is_file()):
                            eligible.add(index)
            except Exception as e:
                logger.warning("[Server] cannot read existing RX manifest: %s", e)

        try:
            self.dump_recon_dir.mkdir(parents=True, exist_ok=True)
            with self._manifest_lock:
                dst = self.dump_recon_dir / "manifest.jsonl"
                with dst.open("w", encoding="utf-8") as f:
                    # Preserve the original MMBench row order.
                    for index, row in self._question_by_index.items():
                        if index in eligible:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._question_indices_written = eligible
            logger.info("[Server] MMBench manifest ready: %d rows -> %s",
                        len(eligible), dst)
        except Exception as e:
            logger.warning("[Server] question manifest rebuild failed: %s", e)

    def save_report(self):
        try:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time())
            out = self.report_dir / (
                f"odp_image_udp_{self.network_profile}_{stamp}.json")
            with self._stats_lock:
                stats = dict(self.stats)

            exp_c = stats["chunks_expected_total"]
            exp_t = stats["tokens_expected_total"]
            derived = {
                "datagram_loss_rate": (stats["chunks_lost_total"] / exp_c
                                       if exp_c else 0.0),
                "token_delivery_rate": (stats["tokens_recovered_total"] / exp_t
                                        if exp_t else 0.0),
                "partial_image_rate": (stats["images_partial"]
                                       / stats["images_received"]
                                       if stats["images_received"] else 0.0),
            }
            report = {
                "transport": "udp",
                "model": self.model,
                "num_tokens": self.num_tokens,
                "network_profile": self.network_profile,
                "reassembly_timeout_sec": self.reassembly_timeout,
                "dump_recon_dir": str(self.dump_recon_dir),
                "t_start": self.t_start,
                "t_end": time.time(),
                "duration_sec": time.time() - self.t_start,
                **stats,
                **derived,
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info("[Server] report saved to %s", out)
            self._report_written = True
            return out
        except Exception as e:
            logger.error("[Server] report write failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # timeout reaper: without this, an image whose tail datagram was lost
    # would never complete and the run would appear to hang
    # ------------------------------------------------------------------
    def flush_all(self, reason: str = "timeout") -> int:
        with self._pending_lock:
            stale = list(self._pending.items())
            self._pending.clear()
        for index, entry in stale:
            self._flush_image(index, entry, complete=False)
        if stale:
            logger.info("[Server] flushed %d partial image(s) (%s)",
                        len(stale), reason)
        return len(stale)

    def _reap_once(self) -> None:
        cutoff = time.time() - self.reassembly_timeout
        with self._pending_lock:
            stale_keys = [i for i, e in self._pending.items()
                          if e["t_last"] < cutoff]
            stale = [(i, self._pending.pop(i)) for i in stale_keys]
        for index, entry in stale:
            self._flush_image(index, entry, complete=False)

    def _reaper(self) -> None:
        interval = max(0.02, self.reassembly_timeout / 2.0)
        while not self._stop.wait(interval):
            try:
                self._reap_once()
            except Exception as e:  # keep the reaper alive no matter what
                logger.warning("[Server] reaper error: %s", e)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        if not self.dump_recon:
            return
        if self.fresh_recon:
            # This process owns the recon dir on the run box; clear a
            # previous run's artifacts so the manifest matches this run.
            try:
                for stale in ("udp_rx_manifest.jsonl", "manifest.jsonl"):
                    p = self.dump_recon_dir / stale
                    if p.exists():
                        p.unlink()
                for old in self.dump_recon_dir.glob("*.png"):
                    old.unlink()
                self._question_indices_written.clear()
                logger.info("[Server] --fresh: cleared previous recon "
                            "artifacts in %s", self.dump_recon_dir)
            except Exception as e:  # noqa: BLE001
                logger.warning("[Server] --fresh cleanup failed: %r", e)
        if self._question_by_index:
            self._rebuild_question_manifest()
        # Warm the dec bridge on the MAIN thread before serving: lazy loading
        # inside a worker thread raced with CUDA/flash-attn init and lost the
        # first frames' recon. UDP has no retransmission, so a frame lost to
        # a cold bridge is lost for good -- this warm-up matters more here.
        self.bridge()

    def run(self) -> None:
        self._prepare()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                            self.recv_buffer_bytes)
        except OSError:
            logger.debug("[Server] SO_RCVBUF bump not permitted")
        sock.bind((self.host, self.port))
        sock.settimeout(0.5)

        reaper = threading.Thread(target=self._reaper, name="reaper",
                                  daemon=True)
        reaper.start()

        actual_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        logger.info("[Server] ODP image UDP server on %s:%d "
                    "(rcvbuf=%d B, reassembly_timeout=%.3fs, profile=%s)",
                    self.host, self.port, actual_rcvbuf,
                    self.reassembly_timeout, self.network_profile)
        logger.info("[Server] recon dump: %s (dump_recon=%s)",
                    self.dump_recon_dir, self.dump_recon)

        try:
            while not self._stop.is_set():
                try:
                    data, _addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError as e:
                    logger.warning("[Server] recvfrom error: %s", e)
                    continue
                try:
                    self._on_datagram(data)
                except Exception as e:  # never let one datagram kill the loop
                    self._bump("errors")
                    logger.warning("[Server] datagram handling error: %s", e,
                                   exc_info=True)
        except KeyboardInterrupt:
            logger.info("[Server] interrupted, draining...")
        finally:
            self._stop.set()
            self.flush_all(reason="shutdown")
            self._recon_pool.shutdown(wait=True)
            self.save_report()
            sock.close()
            with self._stats_lock:
                s = dict(self.stats)
            logger.info("[Server] done: %d datagrams, %d images "
                        "(%d complete / %d partial), %d chunks lost, "
                        "%d/%d tokens recovered",
                        s["datagrams_received"], s["images_received"],
                        s["images_complete"], s["images_partial"],
                        s["chunks_lost_total"],
                        s["tokens_recovered_total"],
                        s["tokens_expected_total"])


def main():
    parser = argparse.ArgumentParser(
        description="ODP image UDP receiver (lossy-link counterpart of "
                    "odp_image_http_server.py)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1237)
    parser.add_argument("--model", default="one_d_piece")
    parser.add_argument("--num-tokens", type=int, default=256,
                        help="expected tokens/image; only used for reporting, "
                             "the wire header is authoritative")
    parser.add_argument("--no-dump-recon", action="store_true",
                        help="skip decoding (pure link measurement, no GPU)")
    parser.add_argument("--dump-recon-dir",
                        default="results/odp_image_recon")
    parser.add_argument("--report-dir", default="results/odp_image_reports")
    parser.add_argument("--network-profile", default="no_emulation",
                        help="label for the report filename, e.g. loss1_rtt20")
    parser.add_argument("--fresh", action="store_true",
                        help="clear previous recon artifacts in the dump dir")
    parser.add_argument("--reassembly-timeout", type=float, default=0.2,
                        help="seconds of silence before a partial image is "
                             "flushed with holes (tail-loss recovery)")
    parser.add_argument("--recv-buffer-mb", type=float, default=4.0,
                        help="SO_RCVBUF size; raise it for bursty senders")
    args = parser.parse_args()

    setup_logger("odp_image_udp_server")
    server = ODPImageUDPServer(
        host=args.host, port=args.port, model=args.model,
        num_tokens=args.num_tokens,
        dump_recon=not args.no_dump_recon,
        dump_recon_dir=args.dump_recon_dir,
        report_dir=args.report_dir,
        network_profile=args.network_profile,
        fresh_recon=args.fresh,
        reassembly_timeout=args.reassembly_timeout,
        recv_buffer_bytes=int(args.recv_buffer_mb * (1 << 20)),
    )
    if not args.no_dump_recon:
        server.install_question_manifest(str(MMBENCH_QUESTION_MANIFEST))
    server.run()


if __name__ == "__main__":
    main()
