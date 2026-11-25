"""Edge client for the One-D-Piece image link over UDP, WHOLE-IMAGE mode.

Lossy-link counterpart of ``odp_image_http_client.py``, and the all-or-nothing
sibling of ``odp_image_udp_client_chunked.py``.

Same edge pipeline (``bridge.tokenize`` -> ``pack_tokens_12bit``), but the
entire 12-bit frame rides in ONE datagram (``nchunks=1``).

Why this mode exists: it is the control condition. One lost datagram = one
TOTALLY lost image (no partial recon, no graceful degradation), so comparing
it against the chunked run isolates exactly how much the per-chunk hole
filling in ``odp_recover_tokens`` buys you under the same link loss.

At the default L-256 setting a frame is 384 B, so one datagram stays well
under a 1500 B MTU. Larger token counts will exceed the MTU and get
IP-fragmented -- at which point a single fragment loss drops the whole
datagram anyway; the client warns when that happens.

There is NO retransmission and NO ack: that is the point of the UDP link.

Optional payload-level synthetic token drop (applied BEFORE 12-bit packing,
so it composes with real netem link loss instead of replacing it):
    --payload-drop-mode tail|random|none (default "none")
    --payload-drop-pct  0..100           (default 0.0)
Mirrors the offline ``one_d_piece_folder_recon.py`` token-drop study's
"tail" (keep most-important prefix, shorter sequence) and "random" (zero
out random positions, same length) strategies. Env ``ODP_PAYLOAD_DROP_MODE``
/ ``ODP_PAYLOAD_DROP_PCT`` win over both the CLI flags and the constructor
defaults, same convention as ``ODP_NUM_TOKENS`` in the bridge.

Usage:
    conda activate 6G-Testbed
    python clients/odp_image_udp_client_whole.py \
        --image-dir /data01/Liao/ai_traffic/qwen3_vl/Qwen3-VL/data/images/MMBench_DEV_EN \
        --server 127.0.0.1:1237 --max-images 8 \
        --report results/odp_image_udp_whole_session.json
"""

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
# Bootstrap the sibling aicodec package relative to this file's location
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AICODEC_PATH = _REPO_ROOT / "aicodec"
if str(_AICODEC_PATH) not in sys.path:
    sys.path.insert(0, str(_AICODEC_PATH))

# the UDP wire format (incl. 12-bit token packing) lives in odp_udp_wire;
# it has no aiortc dependency, unlike odp_codec.py
import aicodec.models.one_d_piece  # noqa: F401  (registers "one_d_piece")
from aicodec.models import get_bridge
from aicodec.models.one_d_piece import odp_udp_wire as wire
from aicodec.models.one_d_piece.odp_udp_wire import pack_tokens_12bit

# Reuse odp_udp_wire's discover_images verbatim: dataset ORDER and the
# manifest-derived index are what tie a recon PNG back to its MMBench
# question, so every transport must agree on them exactly.
discover_images = wire.discover_images

#: 1500 B Ethernet MTU - 20 B IP - 8 B UDP
DEFAULT_MTU_PAYLOAD = 1472

logger = logging.getLogger("odp_image_udp_client_whole")


def setup_logger(name: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    logging.getLogger(name).setLevel(logging.INFO)


class ODPImageUDPWholeClient:
    """One image -> exactly ONE datagram (``nchunks=1``)."""

    def __init__(self, server_host: str = "127.0.0.1",
                 server_port: int = 1237,
                 model: str = "one_d_piece", num_tokens: int = 256,
                 inter_image_delay_sec: float = 0.0,
                 network_profile: str = "no_emulation",
                 control_repeat: int = 3,
                 mtu_payload: int = DEFAULT_MTU_PAYLOAD,
                 send_buffer_bytes: int = 4 << 20,
                 payload_drop_mode: str = "none",
                 payload_drop_pct: float = 0.0):
        self.server_host = server_host
        self.server_port = server_port
        self.model = model
        self.num_tokens = num_tokens
        self.network_profile = network_profile
        self.inter_image_delay_sec = inter_image_delay_sec
        self.control_repeat = control_repeat
        self.mtu_payload = mtu_payload
        self.send_buffer_bytes = send_buffer_bytes

        # Payload-level synthetic token drop, applied BEFORE 12-bit packing
        # (see aicodec.models.one_d_piece.odp_udp_wire.apply_payload_token_drop). Env wins over
        # the constructor args, same convention as ODP_NUM_TOKENS.
        self.payload_drop_mode = (
            os.environ.get("ODP_PAYLOAD_DROP_MODE", "").strip() or payload_drop_mode)
        _drop_pct_env = os.environ.get("ODP_PAYLOAD_DROP_PCT", "").strip()
        self.payload_drop_pct = float(_drop_pct_env) if _drop_pct_env else payload_drop_pct

        self._bridge = None
        self._mtu_warned = False
        self.session_metrics = {}

    @property
    def target(self):
        return (self.server_host, self.server_port)

    def bridge(self):
        if self._bridge is None:
            logger.info("[Client] loading %s bridge (enc)...", self.model)
            self._bridge = get_bridge(self.model, "enc")
            logger.info("[Client] bridge ready")
        return self._bridge

    @property
    def provider(self) -> str:
        return "odp-image-udp"
    # ------------------------------------------------------------------
    # send
    # ------------------------------------------------------------------
    def upload_images(self, images, session_id: str = "") -> dict:
        """Name-compatible entry point for the orchestrator scenario.

        Mirrors ``ODPImageHTTPClient.upload_images`` so the scenario layer can
        drive either transport; ``session_id`` is recorded for traceability
        only (UDP is connectionless, there is no session handshake).
        """
        return self.send_images(images, session_id=session_id)

    def send_images(self, images, session_id: str = "") -> dict:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                            self.send_buffer_bytes)
        except OSError:
            logger.debug("[Client] SO_SNDBUF bump not permitted")

        metrics = {
            "transport": "udp",
            "mode": "whole",
            "server": f"{self.server_host}:{self.server_port}",
            "session_id": session_id,
            "network_profile": self.network_profile,
            "model": self.model,
            "num_tokens": self.num_tokens,
            "tokens_per_datagram": self.num_tokens,
            "payload_drop_mode": self.payload_drop_mode,
            "payload_drop_pct": self.payload_drop_pct,
            "images_total": len(images),
            "images_sent": 0,
            "images_ok": 0,
            "images_failed": 0,
            "datagrams_sent": 0,
            "datagram_send_errors": 0,
            "datagrams_over_mtu": 0,
            "payload_bytes_sent": 0,
            "wire_bytes_sent": 0,
            "tokens_sent": 0,
            "tokenize_sec_total": 0.0,
            "t_start": time.time(),
        }
        records = []

        logger.info("[Client] sending %d image(s) to %s:%d "
                    "(whole-image mode, 1 datagram each)",
                    len(images), self.server_host, self.server_port)

        # Warm the encoder BEFORE the first send so the first image is not
        # delayed by a multi-second GPU load (which would skew its latency).
        self.bridge()

        for index, name, path in images:
            rec = self._send_one(sock, index, name, path, metrics)  ## send one image
            records.append(rec)
            if self.inter_image_delay_sec > 0:
                time.sleep(self.inter_image_delay_sec)  ## sleep between image sends

        # Tell the server the session is over so it can flush partials and
        # write its report without waiting for the reassembly timeout.
        # Repeated: this datagram can be lost like any other.
        ctl = wire.build_control(wire.CTL_SESSION_COMPLETE)
        for _ in range(max(1, self.control_repeat)):
            try:
                sock.sendto(ctl, self.target)
                metrics["datagrams_sent"] += 1
                metrics["wire_bytes_sent"] += len(ctl)
            except OSError as e:
                logger.warning("[Client] control send failed: %s", e)
            time.sleep(0.01)

        sock.close()
        metrics["t_end"] = time.time()
        metrics["duration_sec"] = metrics["t_end"] - metrics["t_start"]
        # "per_image" mirrors the HTTP client's report key so the analysis
        # layer can treat both transports uniformly.
        metrics["per_image"] = records
        # TCP had a response body to measure; UDP is fire-and-forget. Kept at
        # 0 so shared report consumers do not special-case the transport.
        metrics["response_bytes_received"] = 0
        lat = sorted(r["send_sec"] for r in records if r.get("ok"))
        if lat:
            metrics["send_latency_sec_stats"] = {
                "min": lat[0],
                "p50": lat[len(lat) // 2],
                "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
                "max": lat[-1],
                "mean": sum(lat) / len(lat),
            }
        self.session_metrics = metrics

        logger.info("[Client] done: %d/%d images, %d datagrams, "
                    "%d tokens, %.2f s",
                    metrics["images_ok"], metrics["images_total"],
                    metrics["datagrams_sent"], metrics["tokens_sent"],
                    metrics["duration_sec"])
        return metrics

    def _send_one(self, sock, index: int, name: str, path: str,
                  metrics: dict) -> dict:
        """Tokenize + fire ONE datagram. Fire-and-forget: no ack, no retry."""
        rec = {"index": index, "name": name, "path": path,
               "num_tokens": 0, "num_tokens_raw": 0, "payload_bytes": 0,
               "datagram_bytes": 0,
               "tokenize_sec": 0.0, "send_sec": 0.0, "ok": False}
        try:
            t_tok = time.time()
            from PIL import Image
            with Image.open(path) as im:
                im = im.convert("RGB")
                tokens = self.bridge().tokenize(im)
            rec["num_tokens_raw"] = len(tokens)
            # Payload-level synthetic token drop, BEFORE 12-bit packing, so
            # it composes with (does not replace) real netem link loss.
            tokens = wire.apply_payload_token_drop(
                tokens, self.payload_drop_mode, self.payload_drop_pct,
                seed=index)
            payload = pack_tokens_12bit(tokens)
            rec["tokenize_sec"] = round(time.time() - t_tok, 6)
            rec["num_tokens"] = len(tokens)
            rec["payload_bytes"] = len(payload)
        except Exception as e:  # tokenize/open failure: record, keep going
            rec["error"] = f"tokenize: {type(e).__name__}: {e}"
            metrics["images_sent"] += 1
            metrics["images_failed"] += 1
            logger.warning("[Client] image %s tokenize failed: %s", name,
                           rec["error"])
            return rec

        t_send = time.time()
        timestamp = int(t_send * 90000) & 0xFFFFFFFF
        dg = wire.build_datagram(
            index=index, name=name, frame_bytes=len(payload),
            nchunks=1, chunk_seq=0,
            picture_id=index, timestamp=timestamp,
            pos=0, chunk=payload)
        rec["datagram_bytes"] = len(dg)

        if len(dg) > self.mtu_payload:
            metrics["datagrams_over_mtu"] += 1
            if not self._mtu_warned:
                self._mtu_warned = True
                # IP will fragment it; losing any one fragment drops the
                # whole datagram, so the measured loss will exceed the
                # link's configured per-packet loss rate.
                logger.warning(
                    "[Client] datagram is %d B > %d B MTU payload: IP will "
                    "fragment, and effective image loss will be HIGHER than "
                    "the configured link loss. Use the chunked client for "
                    "num_tokens this large.", len(dg), self.mtu_payload)

        try:
            sock.sendto(dg, self.target)
            rec["ok"] = True  # left this host; says nothing about delivery
            metrics["wire_bytes_sent"] += len(dg)
            metrics["datagrams_sent"] += 1
        except OSError as e:
            metrics["datagram_send_errors"] += 1
            logger.warning("[Client] sendto failed (%s): %s", name, e)

        rec["send_sec"] = round(time.time() - t_send, 6)
        metrics["images_sent"] += 1
        metrics["payload_bytes_sent"] += rec["payload_bytes"]
        metrics["tokens_sent"] += rec["num_tokens"]
        metrics["tokenize_sec_total"] += rec["tokenize_sec"]
        if rec["ok"]:
            metrics["images_ok"] += 1
        else:
            metrics["images_failed"] += 1
        return rec

    def get_session_report(self) -> dict:
        return self.session_metrics or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ODP image UDP client, WHOLE-IMAGE mode "
                    "(1 datagram per image)")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--manifest", default=None,
                        help="explicit dataset manifest.jsonl; default is "
                             "<image-dir>/manifest.jsonl if present")
    parser.add_argument("--max-images", type=int, default=0,
                        help="0 = full dataset")
    parser.add_argument("--server", default="127.0.0.1:1237",
                        help="host:port of odp_image_udp_server.py")
    parser.add_argument("--model", default="one_d_piece")
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--inter-image-delay", type=float, default=0.0,
                        help="pacing between images")
    parser.add_argument("--network-profile", default="no_emulation",
                        help="label recorded in the report, e.g. lossy")
    parser.add_argument("--control-repeat", type=int, default=3,
                        help="how many session-complete datagrams to send")
    parser.add_argument("--mtu-payload", type=int,
                        default=DEFAULT_MTU_PAYLOAD,
                        help="warn above this datagram size (default 1472 "
                             "= 1500 MTU - 20 IP - 8 UDP)")
    parser.add_argument("--payload-drop-mode", default="none",
                        choices=["none", "tail", "random"],
                        help="synthetic token drop applied BEFORE 12-bit "
                             "packing (composes with real netem loss); "
                             "env ODP_PAYLOAD_DROP_MODE wins")
    parser.add_argument("--payload-drop-pct", type=float, default=0.0,
                        help="0..100, percent of tokens dropped under "
                             "--payload-drop-mode; env ODP_PAYLOAD_DROP_PCT "
                             "wins")
    parser.add_argument("--report", default=None,
                        help="write the session metrics JSON here")
    args = parser.parse_args()

    setup_logger("odp_image_udp_client_whole")

    if ":" not in args.server:
        parser.error("--server must be host:port")
    host, port_s = args.server.rsplit(":", 1)

    images = discover_images(args.image_dir, args.manifest,
                             args.max_images or None)
    if not images:
        logger.error("[Client] no images found under %s", args.image_dir)
        sys.exit(1)

    client = ODPImageUDPWholeClient(
        server_host=host, server_port=int(port_s),
        model=args.model, num_tokens=args.num_tokens,
        inter_image_delay_sec=args.inter_image_delay,
        network_profile=args.network_profile,
        control_repeat=args.control_repeat,
        mtu_payload=args.mtu_payload,
        payload_drop_mode=args.payload_drop_mode,
        payload_drop_pct=args.payload_drop_pct)

    metrics = client.send_images(images)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("[Client] report saved to %s", out)


if __name__ == "__main__":
    main()


