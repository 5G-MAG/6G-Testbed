"""
One-D-Piece Image-over-UDP Scenario for the 6G AI Traffic Testbed.

Lossy-link sibling of ``odp_image_http.py``. Same edge pipeline (tokenize
MMBench-style images with the One-D-Piece L-256 bridge, 12-bit pack), but the
payload is pushed over connectionless UDP with NO retransmission, so network
loss reaches the application instead of being hidden by TCP.

Two transmission modes (``mode`` config key):
  chunked : N datagrams per image (default 64 tokens = 96 B each). A lost
            datagram costs only its own tokens; the server zero-fills the
            hole and still reconstructs -> graceful degradation.
  whole   : exactly 1 datagram per image. Any loss destroys the whole image
            -> all-or-nothing control condition.

Config keys (configs/scenarios.yaml: odp_image_udp_chunked / odp_image_udp_whole):
  server_host / server_port   : image UDP server (default 127.0.0.1:1237)
  mode                        : "chunked" (default) | "whole"
  tokens_per_datagram         : chunked mode only, must be EVEN (default 64)
  model                       : "one_d_piece"
  num_tokens                  : latent tokens per image, 1..256
                                (env ODP_NUM_TOKENS wins inside the bridge)
  image_dir                   : directory of source images (JPG/PNG/...)
  manifest_path               : optional jsonl manifest ordering the images
  max_images                  : cap per run (0 = full dataset)
  inter_image_delay_sec       : pacing between images (default 0 = burst)
  inter_datagram_delay_sec    : chunked mode only, pacing within an image
  payload_drop_mode           : whole mode only, "none" (default) | "tail" |
                                "random" -- synthetic token drop applied
                                BEFORE 12-bit packing. ONLY takes effect when
                                network_profile == "no_emulation" (see
                                configs/profiles.yaml): that profile already
                                carries ZERO netem impairment, so reusing it
                                gives a clean, isolated "payload-drop-only"
                                run that does not compose with real link
                                loss. "tail" truncates the least-important
                                trailing tokens (shorter sequence); "random"
                                zeroes out random positions (same length).
                                Mirrors one_d_piece_folder_recon.py's offline
                                token-drop study. Defaults to "none"/0 so
                                plain no_emulation baseline runs stay clean;
                                set via env ODP_PAYLOAD_DROP_MODE (wins
                                inside the client) when sweeping drop ratios.
  payload_drop_pct            : whole mode only, 0..100, percent of tokens
                                dropped under payload_drop_mode. Same
                                "no_emulation"-only gating as above. Defaults
                                to 0; env ODP_PAYLOAD_DROP_PCT wins.

Network impairment is applied by the orchestrator to the loopback device
(--interface lo), identical to the HTTP and WebRTC ODP runs.

IMPORTANT -- where the loss numbers come from:
UDP is fire-and-forget, so this client-side scenario can only report that a
datagram LEFT the host. Delivery, per-chunk loss, partial-image counts and
token recovery rates are measured by the SERVER
(``server/odp_image_udp_server.py`` -> ``results/odp_image_reports/*.json``
and ``udp_rx_manifest.jsonl``). Read the two together: this scenario gives
the offered load, the server report gives what survived.
"""

import logging
import time
from pathlib import Path

from clients.base import LLMClient
from .base import BaseScenario, ScenarioResult
from analysis.logger import TrafficLogger
from aicodec.models.one_d_piece import odp_udp_wire as wire

logger = logging.getLogger(__name__)

# Repo root (aitestbed/scenarios/odp_image_udp.py -> parents[2]). Config paths
# may be written repo-relative (e.g. aicodec/.../MMBench_DEV_EN) and stay
# machine-independent; absolute paths are honored as-is.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value):
    if not value:
        return value
    p = Path(value).expanduser()
    if p.is_absolute() and p.exists():
        return str(p)
    repo_p = _REPO_ROOT / p
    return str(repo_p if repo_p.exists() else p)


class ODPImageUDPScenario(BaseScenario):
    """Image tokenization + upload over a lossy UDP transport (ODP)."""

    def __init__(
        self,
        client: LLMClient,
        logger: TrafficLogger,
        config: dict,
    ):
        super().__init__(client, logger, config)
        self._odp_client = None
        self._protocol = "udp"

    @property
    def scenario_type(self) -> str:
        return "odp_image_udp"

    def _build_client(self, mode: str, **kw):
        """Import lazily so a missing GPU bridge cannot break scenario
        discovery at orchestrator start-up."""
        if mode == "whole":
            from clients.odp_image_udp_client_whole import (
                ODPImageUDPWholeClient)
            kw.pop("tokens_per_datagram", None)
            kw.pop("inter_datagram_delay", None)
            return ODPImageUDPWholeClient(**kw)
        if mode != "chunked":
            raise ValueError(
                f"{self.scenario_id}: mode must be 'chunked' or 'whole', "
                f"got {mode!r}")
        from clients.odp_image_udp_client_chunked import (
            ODPImageUDPChunkedClient)
        # payload_drop_* is a whole-mode-only knob (see
        # odp_image_udp_client_whole.py); chunked's constructor does not
        # accept it.
        kw.pop("payload_drop_mode", None)
        kw.pop("payload_drop_pct", None)
        return ODPImageUDPChunkedClient(**kw)

    def run(
        self,
        network_profile: str,
        run_index: int = 0,
    ) -> ScenarioResult:
        session_id = self._create_session_id()

        server_host = self.config.get("server_host", "127.0.0.1")
        server_port = int(self.config.get("server_port", 1237))
        mode = str(self.config.get("mode", "chunked")).lower()
        model = self.config.get("model", "one_d_piece")
        num_tokens = self.config.get("num_tokens", 256)
        image_dir = _resolve_path(self.config.get("image_dir"))
        manifest_path = _resolve_path(self.config.get("manifest_path"))
        max_images = self.config.get("max_images", 10)
        inter_delay = float(self.config.get("inter_image_delay_sec", 0.0))
        inter_dg_delay = float(
            self.config.get("inter_datagram_delay_sec", 0.0))
        tokens_per_dg = int(self.config.get("tokens_per_datagram", 64))
        # Payload-level synthetic token drop (whole mode only; see
        # odp_image_udp_client_whole.py / odp_udp_wire.apply_payload_token_drop).
        # Gated to the "no_emulation" network profile ONLY (already zero
        # netem impairment by definition, see configs/profiles.yaml) so a
        # multi-profile sweep over [no_emulation, lossy, 5g_urban, ...] never
        # silently drops tokens on the real netem-loss runs. payload_drop_mode
        # defaults to "none"/0 below, so a plain no_emulation baseline run
        # (e.g. inside that sweep) stays a clean, drop-free control unless the
        # caller explicitly opts in via config or ODP_PAYLOAD_DROP_MODE/PCT.
        PAYLOAD_DROP_PROFILE = "no_emulation"
        if network_profile == PAYLOAD_DROP_PROFILE:
            payload_drop_mode = str(
                self.config.get("payload_drop_mode", "none")).lower()
            payload_drop_pct = float(self.config.get("payload_drop_pct", 0.0))
        else:
            payload_drop_mode = "none"
            payload_drop_pct = 0.0

        if not image_dir:
            raise ValueError(f"{self.scenario_id}: config 'image_dir' missing")

        result = ScenarioResult(
            scenario_id=self.scenario_id,
            session_id=session_id,
            network_profile=network_profile,
            run_index=run_index,
        )

        try:
            self._odp_client = self._build_client(
                mode,
                server_host=server_host,
                server_port=server_port,
                model=model,
                num_tokens=num_tokens,
                network_profile=network_profile,
                inter_image_delay_sec=inter_delay,
                inter_datagram_delay=inter_dg_delay,
                tokens_per_datagram=tokens_per_dg,
                payload_drop_mode=payload_drop_mode,
                payload_drop_pct=payload_drop_pct,
            )

            # discover_images lives in odp_udp_wire (shared with both UDP
            # clients): dataset ORDER and the manifest-derived index are
            # what tie a recon PNG back to its MMBench question, so every
            # run must enumerate identically to stay comparable.
            images = wire.discover_images(
                image_dir, manifest_path, max_images)
            if not images:
                raise ValueError(f"no images found in {image_dir!r}")
            logger.info(f"[ODPImageUDP] run {run_index}: {len(images)} images "
                        f"from {image_dir} (mode={mode})")

            # UDP has no handshake and no /health endpoint, so there is no
            # connection record to make. The real fixed setup cost here is
            # the encoder bridge load, so that is what we log instead.
            t_warm_start = time.time()
            self._odp_client.bridge()
            t_warmed = time.time()
            self._create_log_record(
                session_id=session_id,
                turn_index=-1,
                run_index=run_index,
                network_profile=network_profile,
                request_bytes=0,
                response_bytes=0,
                t_request_start=t_warm_start,
                latency_sec=t_warmed - t_warm_start,
                http_status=200,
                success=True,
                is_streaming=False,
                metadata={"event_type": "udp_encoder_warmup",
                          "protocol": self._protocol,
                          "mode": mode,
                          "model": model,
                          "num_tokens": num_tokens},
            )

            t_session_start = time.time()
            self._odp_client.upload_images(images, session_id)
            metrics = self._odp_client.get_session_report()
            t_session_end = time.time()

            # per-image records. NOTE: success here means "all datagrams for
            # this image left the host" -- on UDP the client cannot know what
            # arrived. Delivery is in the SERVER's report.
            for rec in metrics.get("per_image", []):
                self._create_log_record(
                    session_id=session_id,
                    turn_index=rec["index"],
                    run_index=run_index,
                    network_profile=network_profile,
                    request_bytes=rec["payload_bytes"],
                    response_bytes=0,
                    tokens_in=rec["num_tokens"],
                    t_request_start=t_session_start,
                    latency_sec=rec["send_sec"],
                    http_status=200 if rec["ok"] else 500,
                    success=bool(rec["ok"]),
                    is_streaming=False,
                    metadata={"event_type": "image_datagram_send",
                              "protocol": self._protocol,
                              "mode": mode,
                              "image_index": rec["index"],
                              "image_name": rec["name"],
                              "datagrams_sent": rec.get("datagrams_sent", 1),
                              "tokenize_sec": rec["tokenize_sec"]},
                )

            result.success = (metrics["images_ok"] == metrics["images_sent"]
                              and metrics["images_sent"] > 0)
            result.total_latency_sec = t_session_end - t_session_start
            result.total_request_bytes = metrics["payload_bytes_sent"]
            result.total_response_bytes = metrics["response_bytes_received"]
            result.total_tokens_in = metrics["tokens_sent"]
            result.total_tokens_out = 0
            result.metadata = {
                "protocol": self._protocol,
                "mode": mode,
                "model": model,
                "num_tokens": metrics["num_tokens"],
                "tokens_per_datagram": metrics.get("tokens_per_datagram"),
                "images_sent": metrics["images_sent"],
                "images_ok": metrics["images_ok"],
                "images_failed": metrics["images_failed"],
                "datagrams_sent": metrics["datagrams_sent"],
                "datagram_send_errors": metrics["datagram_send_errors"],
                "datagrams_over_mtu": metrics.get("datagrams_over_mtu", 0),
                "payload_bytes_sent": metrics["payload_bytes_sent"],
                "wire_bytes_sent": metrics["wire_bytes_sent"],
                "tokenize_sec_total": metrics["tokenize_sec_total"],
                "send_latency_sec_stats": metrics.get(
                    "send_latency_sec_stats"),
                "image_dir": str(image_dir),
                "server": f"{server_host}:{server_port}",
                # Reminder for whoever reads these runs: this scenario
                # measures OFFERED load only. Loss / partial-image / token
                # recovery rates live in the server's report, because UDP
                # gives the sender no delivery feedback at all.
                "delivery_metrics_source": (
                    "server/odp_image_udp_server.py report + "
                    "udp_rx_manifest.jsonl"),
            }
            return result
        except Exception as e:
            logger.exception(f"[ODPImageUDP] run failed: {e}")
            result.success = False
            result.metadata = {"error": str(e)}
            return result

    def cleanup(self) -> None:
        self._odp_client = None


