"""Dynamic codec registration for a stock aiortc install (pinned 1.14/1.15).

Usage:  import aicodec; aicodec.register()

Patches six things in the running ``aiortc`` / ``aioice`` packages:
  1. ``aiortc.codecs.CODECS["video"]`` — append token codecs (+rtx pairs)
  2. ``aiortc.codecs.get_encoder`` / ``get_decoder`` / ``depayload`` — token branches
  3. ``RTCRtpReceiver._handle_rtp_packet`` — pass codec name to JitterBuffer
  4. ``aiortc.jitterbuffer.JitterBuffer`` — loss-tolerant framing for token
     codecs (add() gains codecName, _remove_frame skips missing packets)
  5. ``RTCRtpReceiver.receive`` — enlarge the video JitterBuffer (128 -> 512)
  6. ``aioice.ice.get_host_addresses`` — optional interface/address filter for
     ICE host-candidate gathering (env vars AICODEC_ICE_IFACE / AICODEC_ICE_ADDR,
     e.g. AICODEC_ICE_IFACE=lo forces media over the loopback interface so that
     netem shaping on `lo` and tcpdump -i lo both see the media stream).
     Also disables the default Google STUN server when filtering is active,
     so no srflx candidates from other interfaces leak into the SDP.
"""

import logging
import os
from typing import Optional, Union

from aiortc import __version__ as _aiortc_version
from aiortc.codecs import CODECS
from aiortc.jitterbuffer import JitterBuffer
from aiortc.rtcrtpparameters import (
    ParametersDict,
    RTCRtpCodecParameters,
    RTCRtcpFeedback,
)
from aiortc.rtcrtpreceiver import RTCRtpReceiver
from aiortc.rtcrtpsender import RTCRtpSender

logger = logging.getLogger(__name__)

_registered = False

TOKEN_MIME_TYPES = ("video/tokenId", "video/tokenIdTransparent", "video/tokenIdLLM")

_FALLBACK_PTS: dict[str, int] = {}
_dyn_pt = [97]


def _next_dynamic_pt() -> int:
    """Allocate payload types that don't collide with already-registered codecs."""
    used = {
        c.payloadType
        for codecs in CODECS.values()
        for c in codecs
        if c.payloadType is not None
    }
    pt = max(_dyn_pt[0], (max(used) + 2) if used else 96)
    while pt in used:
        pt += 1
    _dyn_pt[0] = pt + 1
    return pt


def _build_token_codec(mimeType: str) -> tuple[RTCRtpCodecParameters, RTCRtpCodecParameters]:
    pt = _next_dynamic_pt()
    codec = RTCRtpCodecParameters(
        mimeType=mimeType,
        clockRate=90000,
        payloadType=pt,
        rtcpFeedback=[
            RTCRtcpFeedback(type="nack"),
            RTCRtcpFeedback(type="nack", parameter="pli"),
            RTCRtcpFeedback(type="goog-remb"),
        ],
    )
    rtx = RTCRtpCodecParameters(
        mimeType="video/rtx",
        clockRate=90000,
        payloadType=pt + 1,
        parameters={"apt": pt},
    )
    return codec, rtx


# ---------------------------------------------------------------------------
# 2. encoder / decoder / depayload dispatch extensions
# ---------------------------------------------------------------------------

_orig_get_encoder = None
_orig_get_decoder = None
_orig_depayload = None


def _patched_depayload(codec, payload: bytes) -> bytes:
    if codec.name in ("tokenId", "tokenIdTransparent", "tokenIdLLM"):
        from aicodec.tokenid import tokenid_depayload

        return tokenid_depayload(payload)
    return _orig_depayload(codec, payload)


def _patched_get_decoder(codec):
    mimeType = codec.mimeType.lower()
    if mimeType == "video/tokenid":
        from aicodec.tokenid import TokenIdDecoder

        return TokenIdDecoder(decMode="detokenize")
    if mimeType == "video/tokenidtransparent":
        from aicodec.tokenid import TokenIdDecoder

        return TokenIdDecoder(decMode="transparent")
    if mimeType == "video/tokenidllm":
        from aicodec.tokenid import AsyncTokenIdDecoder

        return AsyncTokenIdDecoder()
    return _orig_get_decoder(codec)


def _patched_get_encoder(codec):
    mimeType = codec.mimeType.lower()
    if mimeType in ("video/tokenid", "video/tokenidtransparent", "video/tokenidllm"):
        from aicodec.tokenid import TokenIdEncoder

        return TokenIdEncoder()
    return _orig_get_encoder(codec)


# ---------------------------------------------------------------------------
# 4. JitterBuffer: loss-tolerant frame assembly for token codecs
#    (patches both add() — new optional codecName arg — and _remove_frame)
# ---------------------------------------------------------------------------


def _patched_add(self, packet, codecName: Optional[str] = None):
    from aiortc.utils import uint16_add

    pli_flag = False
    if self._origin is None:
        self._origin = packet.sequence_number
        delta = 0
        misorder = 0
    else:
        delta = uint16_add(packet.sequence_number, -self._origin)
        misorder = uint16_add(self._origin, -packet.sequence_number)

    if misorder < delta:
        if misorder >= 100:  # MAX_MISORDER
            self.remove(self.capacity)
            self._origin = packet.sequence_number
            delta = misorder = 0
            if self._is_video:
                pli_flag = True
        else:
            return pli_flag, None

    if delta >= self.capacity:
        # remove just enough frames to fit the received packets
        excess = delta - self.capacity + 1
        if self.smart_remove(excess):
            self._origin = packet.sequence_number
        if self._is_video:
            pli_flag = True

    pos = packet.sequence_number % self._capacity
    self._packets[pos] = packet

    return pli_flag, self._remove_frame(packet.sequence_number, codecName)


def _patched_remove_frame(
    self: JitterBuffer, sequence_number: int, codecName
) -> Optional[object]:
    from aiortc.jitterbuffer import JitterFrame

    frame = None
    frames = 0
    packets = []
    remove = 0
    timestamp = None

    for count in range(self._capacity):
        pos = (self._origin + count) % self._capacity
        packet = self._packets[pos]
        if codecName is not None and "token" in codecName:
            if packet is None:
                continue  # for lossy transmission
            if (timestamp is None) and (packet is not None):  # for lossy tx
                timestamp = packet.timestamp
            elif (packet.timestamp != timestamp) and (
                timestamp is not None
            ):  # for lossy tx
                if frame is None:
                    frame = JitterFrame(
                        data=b"".join([x._data for x in packets]),  # type: ignore
                        timestamp=timestamp,
                    )
                    remove = count

                # check we have prefetched enough
                frames += 1
                if frames >= self._prefetch:
                    self.remove(remove)
                    return frame

                # start a new frame
                packets = []
                timestamp = packet.timestamp
        else:
            if packet is None:
                break  # default
            if timestamp is None:
                timestamp = packet.timestamp
            elif packet.timestamp != timestamp:
                if frame is None:
                    frame = JitterFrame(
                        data=b"".join([x._data for x in packets]),  # type: ignore
                        timestamp=timestamp,
                    )
                    remove = count

                # check we have prefetched enough
                frames += 1
                if frames >= self._prefetch:
                    self.remove(remove)
                    return frame

                # start a new frame
                packets = []
                timestamp = packet.timestamp

        packets.append(packet)

    return None


# ---------------------------------------------------------------------------
# 3. receiver: pass codec name into JitterBuffer.add()
# ---------------------------------------------------------------------------

_orig_handle_rtp_packet = None


async def _patched_handle_rtp_packet(self, packet, arrival_time_ms: int) -> None:
    # mirror of aiortc 1.14/1.15 RTCRtpReceiver._handle_rtp_packet with the
    # single change: JitterBuffer.add(packet, codec.name) — the codec name
    # selects the loss-tolerant branch of _remove_frame.
    try:
        from aiortc.rtcp import (
            RTCP_PSFB_APP,
            RtcpPsfbPacket,
            RtcpByePacket,
            RtcpSrPacket,
            pack_remb_fci,
            unwrap_rtx,
        )
    except ImportError:
        # aiortc >= 1.15: the rtcp module was merged into aiortc.rtp
        from aiortc.rtp import (
            RTCP_PSFB_APP,
            RtcpPsfbPacket,
            RtcpByePacket,
            RtcpSrPacket,
            pack_remb_fci,
            unwrap_rtx,
        )
    from aiortc.rtcrtpreceiver import NackGenerator, TimestampMapper
    from aiortc.rtp import RtpPacket
    from aiortc.stats import (
        RTCInboundRtpStreamStats,
        RTCRemoteOutboundRtpStreamStats,
        RTCStatsReport,
    )
    from aiortc.utils import uint16_add, uint16_gt
    import time
    from aiortc import clock

    self._RTCRtpReceiver__log_debug("< %s", packet)

    # If the receiver is disabled, discard the packet.
    if not self._enabled:
        return

    # feed bitrate estimator
    __remote_bitrate_estimator = self._RTCRtpReceiver__remote_bitrate_estimator
    __rtcp_ssrc = self._RTCRtpReceiver__rtcp_ssrc
    if __remote_bitrate_estimator is not None:
        if packet.extensions.abs_send_time is not None:
            remb = __remote_bitrate_estimator.add(
                abs_send_time=packet.extensions.abs_send_time,
                arrival_time_ms=arrival_time_ms,
                payload_size=len(packet.payload) + packet.padding_size,
                ssrc=packet.ssrc,
            )
            if __rtcp_ssrc is not None and remb is not None:
                # send Receiver Estimated Maximum Bitrate feedback
                rtcp_packet = RtcpPsfbPacket(
                    fmt=RTCP_PSFB_APP,
                    ssrc=__rtcp_ssrc,
                    media_ssrc=0,
                    fci=pack_remb_fci(*remb),
                )
                await self._send_rtcp(rtcp_packet)

    # keep track of sources
    self._RTCRtpReceiver__active_ssrc[packet.ssrc] = clock.current_datetime()

    # check the codec is known
    codecs = self._RTCRtpReceiver__codecs
    codec = codecs.get(packet.payload_type)
    if codec is None:
        self._RTCRtpReceiver__log_debug(
            "x RTP packet with unknown payload type %d", packet.payload_type
        )
        return

    # feed RTCP statistics
    remote_streams = self._RTCRtpReceiver__remote_streams
    if packet.ssrc not in remote_streams:
        from aiortc.rtcrtpreceiver import StreamStatistics

        remote_streams[packet.ssrc] = StreamStatistics(codec.clockRate)
    remote_streams[packet.ssrc].add(packet)

    # unwrap retransmission packet
    from aiortc.codecs import is_rtx

    if is_rtx(codec):
        rtx_ssrc = self._RTCRtpReceiver__rtx_ssrc
        original_ssrc = rtx_ssrc.get(packet.ssrc)
        if original_ssrc is None:
            self._RTCRtpReceiver__log_debug(
                "x RTX packet from unknown SSRC %d", packet.ssrc
            )
            return

        apt = codec.parameters.get("apt")
        if (
            len(packet.payload) < 2
            or not isinstance(apt, int)
            or apt not in codecs
        ):
            return

        packet = unwrap_rtx(packet, payload_type=apt, ssrc=original_ssrc)
        codec = codecs[apt]

    # send NACKs for any missing any packets
    nack_generator = self._RTCRtpReceiver__nack_generator
    if nack_generator is not None and nack_generator.add(packet):
        await self._send_rtcp_nack(packet.ssrc, sorted(nack_generator.missing))

    # parse codec-specific information
    from aiortc.codecs import depayload

    try:
        if packet.payload:
            packet._data = depayload(codec, packet.payload)  # type: ignore
        else:
            packet._data = b""  # type: ignore
    except ValueError as exc:
        self._RTCRtpReceiver__log_debug("x RTP payload parsing failed: %s", exc)
        return

    # try to re-assemble encoded frame
    jitter_buffer = self._RTCRtpReceiver__jitter_buffer
    pli_flag, encoded_frame = jitter_buffer.add(packet, codec.name)
    # check if the PLI should be sent
    if pli_flag:
        await self._send_rtcp_pli(packet.ssrc)

    # if we have a complete encoded frame, decode it
    decoder_queue = self._RTCRtpReceiver__decoder_queue
    decoder_thread = self._RTCRtpReceiver__decoder_thread
    timestamp_mapper = self._RTCRtpReceiver__timestamp_mapper
    if encoded_frame is not None and decoder_thread:
        encoded_frame.timestamp = timestamp_mapper.map(
            encoded_frame.timestamp
        )
        decoder_queue.put((codec, encoded_frame))


# ---------------------------------------------------------------------------
# 5. receiver: enlarge the video JitterBuffer (128 -> 512) on first receive
#    (token frames are large: ~1024 tokens -> 30+ RTP packets per frame; the
#    stock depth of 128 overflows too early under loss)
# ---------------------------------------------------------------------------

_orig_receive = None
VIDEO_JITTER_BUFFER_CAPACITY = 512


async def _patched_receive(self, parameters) -> None:
    if not self._RTCRtpReceiver__started:
        # mirror of the stock __init__ video branch, with capacity enlarged
        if self._RTCRtpReceiver__kind == "video":
            self._RTCRtpReceiver__jitter_buffer = JitterBuffer(
                capacity=VIDEO_JITTER_BUFFER_CAPACITY, is_video=True
            )
    return await _orig_receive(self, parameters)


# ---------------------------------------------------------------------------
# 6. ICE host-candidate gathering: filter to a single network interface.
#    Selected via set_ice_interface("lo" / "eth0" / ...) BEFORE the
#    RTCPeerConnection performs candidate gathering. None disables the
#    filter (stock aioice behaviour: all interfaces, 127.0.0.1 excluded).
# ---------------------------------------------------------------------------

_ice_interface: Optional[str] = None
_orig_get_host_addresses = None
_orig_get_default_ice_servers = None


def set_ice_interface(name: Optional[str]) -> None:
    """Restrict ICE host candidates to one interface (None = no filter)."""
    global _ice_interface
    _ice_interface = name
    if name:
        logger.info("aicodec: ICE host candidates restricted to interface %r", name)


def get_ice_interface() -> Optional[str]:
    return _ice_interface


def _patched_get_host_addresses(use_ipv4: bool, use_ipv6: bool) -> list:
    if _ice_interface is None:
        return _orig_get_host_addresses(use_ipv4, use_ipv6)

    import ifaddr

    addresses = []
    for adapter in ifaddr.get_adapters():
        if adapter.name != _ice_interface:
            continue
        for ip in adapter.ips:
            if isinstance(ip.ip, str) and use_ipv4:
                # NOTE: unlike stock aioice, do NOT exclude 127.0.0.1 — the
                # loopback interface is a legitimate choice when filtering
                addresses.append(ip.ip)
            elif use_ipv6 and not isinstance(ip.ip, str) and ip.ip[2] == 0 and ip.ip[0] != "::1":
                addresses.append(ip.ip[0])

    if not addresses:
        logger.warning(
            "aicodec: no usable address found on interface %r; "
            "falling back to unfiltered candidate gathering",
            _ice_interface,
        )
        return _orig_get_host_addresses(use_ipv4, use_ipv6)
    return addresses


def _patched_get_default_ice_servers(*args):
    # when the interface filter is active, skip the default Google STUN
    # server: srflx candidates would leak addresses of other interfaces
    # into the SDP and invite connectivity checks off the chosen interface
    if _ice_interface is not None:
        return []
    # original is a @classmethod: accessed as Class.method() it arrives with
    # no args here (already unbound by our module-level assignment); accessed
    # through an instance/with explicit cls it arrives as (cls,)
    return _orig_get_default_ice_servers()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def register() -> None:
    """Register token codecs + patches into the running aiortc (idempotent)."""
    global _registered, _orig_get_encoder, _orig_get_decoder, _orig_depayload, _orig_handle_rtp_packet, _orig_receive, _orig_get_host_addresses, _orig_get_default_ice_servers

    if _registered:
        return

    major_minor = ".".join(_aiortc_version.split(".")[:2])
    if tuple(int(x) for x in major_minor.split(".")) not in ((1, 14), (1, 15)):
        logger.warning(
            "aicodec was developed against aiortc 1.14.x/1.15.x (found %s); "
            "receiver/jitterbuffer patches may not apply cleanly",
            _aiortc_version,
        )

    import aiortc.codecs as codecs_mod

    # 1. extend the codec registry
    existing = {c.mimeType for c in CODECS["video"]}
    for mime in TOKEN_MIME_TYPES:
        if mime in existing:
            continue
        codec, rtx = _build_token_codec(mime)
        CODECS["video"] += [codec, rtx]
        logger.debug("registered %s (PT %d) + rtx (PT %d)", mime, codec.payloadType, rtx.payloadType)

    # 2. dispatch functions
    if codecs_mod.get_encoder is not _patched_get_encoder:
        _orig_get_encoder = codecs_mod.get_encoder
        codecs_mod.get_encoder = _patched_get_encoder
    if codecs_mod.get_decoder is not _patched_get_decoder:
        _orig_get_decoder = codecs_mod.get_decoder
        codecs_mod.get_decoder = _patched_get_decoder
    if codecs_mod.depayload is not _patched_depayload:
        _orig_depayload = codecs_mod.depayload
        codecs_mod.depayload = _patched_depayload

    # keep module-level references in rtcrtpsender/rtcrtpreceiver in sync:
    # they did `from .codecs import get_encoder, ...` at import time
    import aiortc.rtcrtpsender as sender_mod
    import aiortc.rtcrtpreceiver as receiver_mod

    sender_mod.get_encoder = _patched_get_encoder
    receiver_mod.get_decoder = _patched_get_decoder
    receiver_mod.depayload = _patched_depayload

    # 3. receiver patch
    if RTCRtpReceiver._handle_rtp_packet is not _patched_handle_rtp_packet:
        _orig_handle_rtp_packet = RTCRtpReceiver._handle_rtp_packet
        RTCRtpReceiver._handle_rtp_packet = _patched_handle_rtp_packet

    # 4. jitterbuffer patch (add() gains optional codecName, _remove_frame tolerates loss)
    if JitterBuffer.add is not _patched_add:
        JitterBuffer.add = _patched_add
    if JitterBuffer._remove_frame is not _patched_remove_frame:
        JitterBuffer._remove_frame = _patched_remove_frame

    # 5. video jitter buffer capacity 128 -> 512 (fork behavior)
    if RTCRtpReceiver.receive is not _patched_receive:
        _orig_receive = RTCRtpReceiver.receive
        RTCRtpReceiver.receive = _patched_receive

    # 6. ICE host-candidate interface filter (active only when
    #    set_ice_interface() was called with a non-None name)
    import aioice.ice as aioice_ice_mod
    import aiortc.rtcicetransport as ice_transport_mod

    if aioice_ice_mod.get_host_addresses is not _patched_get_host_addresses:
        _orig_get_host_addresses = aioice_ice_mod.get_host_addresses
        aioice_ice_mod.get_host_addresses = _patched_get_host_addresses
    if ice_transport_mod.RTCIceGatherer.getDefaultIceServers is not _patched_get_default_ice_servers:
        _orig_get_default_ice_servers = ice_transport_mod.RTCIceGatherer.getDefaultIceServers
        ice_transport_mod.RTCIceGatherer.getDefaultIceServers = _patched_get_default_ice_servers

    _registered = True
    logger.debug("aicodec registered: %s", [c.mimeType for c in CODECS["video"]])


def is_registered() -> bool:
    return _registered
