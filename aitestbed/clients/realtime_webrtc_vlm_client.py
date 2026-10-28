"""
Realtime VLM Client for 6G AI Traffic Testbed.
"""

import asyncio
import time
import json
from struct import unpack_from
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path  

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCRtpSender, MediaStreamError
from aiortc.contrib.signaling import BYE
from aiortc import VideoStreamTrack
import aicodec
from aicodec.codec import TokenIdEncoder

aicodec.register()

import av

logger = logging.getLogger(__name__)

# ======================
# Signaling Helpers
# ======================
def signaling_object_to_dict(obj):
    """Convert aiortc signaling objects to JSON-serializable dict."""
    if isinstance(obj, RTCSessionDescription):
        return {"type": obj.type, "sdp": obj.sdp}
    elif isinstance(obj, RTCIceCandidate):
        return {
            "type": "candidate",
            "candidate": obj.candidate,
            "sdpMid": obj.sdpMid,
            "sdpMLineIndex": obj.sdpMLineIndex
        }
    elif obj is BYE:
        return {"type": "bye"}
    else:
        return obj


def signaling_dict_to_object(obj):
    """Convert received JSON dict back to aiortc signaling objects."""
    if isinstance(obj, dict):
        if obj.get("type") == "bye":
            return BYE
        elif obj.get("type") in ["offer", "answer"]:
            return RTCSessionDescription(type=obj["type"], sdp=obj["sdp"])
        elif obj.get("type") == "candidate" or "candidate" in obj:
            return RTCIceCandidate(
                candidate=obj["candidate"],
                sdpMid=obj.get("sdpMid"),
                sdpMLineIndex=obj.get("sdpMLineIndex")
            )
        elif obj.get("type") == "model_ack":
            return ModelAck(ok=obj.get("ok", False), message=obj.get("message", ""))
        elif obj.get("type") == "model":
            return {"type": "model", "model": obj.get("model", "")}
    return obj


class ModelAck:
    """Server response to the model negotiation message."""
    def __init__(self, ok: bool, message: str = ""):
        self.ok = ok
        self.message = message

    def __repr__(self):
        return f"ModelAck(ok={self.ok}, message={self.message!r})"


# ======================
# Metrics Data Classes
# ======================
@dataclass
class VLMVideoChunk:
    frame_index: int
    timestamp: float
    pts: int
    bytes_sent: int
    width: int
    height: int


@dataclass
class VLMTextChunk:
    content: str
    timestamp: float
    index: int
    bytes_received: int


@dataclass
class VLMTurnMetrics:
    turn_index: int
    video_source: str

    # timing metrics
    t_turn_start: float = 0.0
    t_first_frame_sent: Optional[float] = None
    t_last_frame_sent: Optional[float] = None
    t_first_token: Optional[float] = None
    t_last_token: Optional[float] = None
    t_inference_done: Optional[float] = None

    # data transfer metrics
    video_frames_sent: int = 0
    video_bytes_sent: int = 0
    video_resolution: str = ""
    video_fps: float = 0.0
    text_bytes_received: int = 0

    # chunk metrics
    video_chunks: list = field(default_factory=list)
    text_chunks: list = field(default_factory=list)

    # content
    output_text: str = ""

    # token count
    token_count: int = 0

    # status
    success: bool = True
    error_message: Optional[str] = None

    @property
    def ttft(self) -> Optional[float]:
        if self.t_first_token is None:
            return None
        return self.t_first_token - self.t_turn_start

    @property
    def ttlt(self) -> Optional[float]:
        if self.t_last_token is None:
            return None
        return self.t_last_token - self.t_turn_start

    @property
    def total_latency(self) -> Optional[float]:
        if self.t_inference_done is None:
            return None
        return self.t_inference_done - self.t_turn_start

    @property
    def video_upload_duration(self) -> Optional[float]:
        if self.t_last_frame_sent is None or self.t_first_frame_sent is None:
            return None
        return self.t_last_frame_sent - self.t_first_frame_sent

    @property
    def video_upload_rate_mbps(self) -> Optional[float]:
        duration = self.video_upload_duration
        if duration is None or duration <= 0:
            return None
        return (self.video_bytes_sent * 8) / (duration * 1_000_000)

    @property
    def inter_chunk_times(self) -> List[float]:
        """Time between consecutive chunks (text)."""
        if len(self.text_chunks) < 2:
            return []
        return [
            self.text_chunks[i].timestamp - self.text_chunks[i-1].timestamp
            for i in range(1, len(self.text_chunks))
        ]

    @property
    def chunk_count(self) -> int:
        return len(self.text_chunks)


@dataclass
class VLMSessionMetrics:
    session_id: str
    model: str = "liquid"

    # session timing
    t_session_start: float = 0.0
    t_session_end: Optional[float] = None
    t_signaling_connected: Optional[float] = None
    t_webrtc_connected: Optional[float] = None

    # WebRTC SDP negotiation
    sdp_offer: Optional[str] = None
    sdp_answer: Optional[str] = None
    sdp_offer_bytes: int = 0
    sdp_answer_bytes: int = 0
    sdp_offer_hash: Optional[str] = None
    sdp_answer_hash: Optional[str] = None
    sdp_negotiation_sec: Optional[float] = None

    # aggregated metrics
    total_turns: int = 0
    total_video_frames_sent: int = 0
    total_video_bytes_sent: int = 0
    total_text_bytes_received: int = 0
    total_request_bytes: int = 0
    total_response_bytes: int = 0

    total_token_count: int = 0

    # turn details
    turns: list = field(default_factory=list)

    # ICE negotiation
    ice_candidates_sent: int = 0
    ice_candidates_received: int = 0

    # status
    success: bool = True
    error_message: Optional[str] = None

    @property
    def session_duration(self) -> Optional[float]:
        if self.t_session_end is None:
            return None
        return self.t_session_end - self.t_session_start

    @property
    def connection_setup_time(self) -> Optional[float]:
        if self.t_webrtc_connected is None:
            return None
        return self.t_webrtc_connected - self.t_session_start

    @property
    def avg_ttft(self) -> Optional[float]:
        ttfts = [t.ttft for t in self.turns if t.ttft is not None]
        return sum(ttfts) / len(ttfts) if ttfts else None

    @property
    def avg_turn_latency(self) -> Optional[float]:
        latencies = [t.total_latency for t in self.turns if t.total_latency is not None]
        return sum(latencies) / len(latencies) if latencies else None


# ======================
# TCP Signaling Client 
# ======================
class TcpSignalingClient:
    """
    TCP signaling client.
    """
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._reader = None
        self._writer = None

    async def connect(self):
        self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
        # logger.info(f"[Signaling] Connected to {self._host}:{self._port}")

    async def receive(self):
        if not self._reader:
            raise ConnectionError("Not connected")
        data = await self._reader.readuntil(b'\n')
        parsed = json.loads(data.decode('utf8').strip())
        # print("[Signaling] receive: ", parsed)
        return signaling_dict_to_object(parsed)

    async def send(self, obj):
        if not self._writer:
            raise ConnectionError("Not connected")
        try:
            data = signaling_object_to_dict(obj)
            message = json.dumps(data) + '\n'
            self._writer.write(message.encode('utf8'))
            await asyncio.wait_for(self._writer.drain(), timeout=5.0)
            return len(message.encode('utf8'))
        except asyncio.TimeoutError:
            raise ConnectionError("Signaling send timeout")

    async def close(self):
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except:
                pass


# ======================
# Video Stream Track
# ======================
# wire format per frame: 1024 uint16 tokens + 12-byte token header (+ ~2B
# VP8-like descriptor). Used for byte metrics; raw YUV size is NOT what
# goes on the wire for the token codecs.
TOKENS_PER_FRAME = 1024
TOKEN_HEADER_BYTES = 14


class realtimeVideoStreamTrack(VideoStreamTrack):
    # kind = "video"
    def __init__(self, path, target_fps, pts_offset=0):
        super().__init__()
        self.kind = "video"
        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.fps = self.stream.average_rate
        if self.fps is None:
            self.fps = target_fps
        self.pts_offset = pts_offset
        self.last_pts = pts_offset
        self.resample_ratio = max(int(self.fps // target_fps), 1)
        self.frame_count_origin = -1
        self.frame_count = 0
        self.decoder = self.container.decode(self.stream)
        
        #-----------------------
        # control the frame send rate @FPS
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.next_send_time = 0.0
        #-----------------------

        self._total_bytes = 0
        self._chunks = []
        self._first_frame_time = None
        self._last_frame_time = None
        self._video_send_complete = False

    async def recv(self):
        while True:
            try:
                frame = next(self.decoder)
                self.last_pts = frame.pts
                frame.pts += self.pts_offset
                self.frame_count_origin += 1
                
                if self.frame_count_origin % self.resample_ratio == 0:
                    #-----------------------
                    # control the frame send rate @FPS (with catch-up: if we
                    # fell behind, don't let next_send_time drift into the past)
                    loop = asyncio.get_event_loop()
                    now = loop.time()
                    if self.next_send_time == 0.0:
                        self.next_send_time = now
                    wait_time = self.next_send_time - now
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    else:
                        self.next_send_time = now
                    self.next_send_time += self.frame_interval
                    #-----------------------
                    self.frame_count += 1

                    # actual wire bytes: 1024 uint16 tokens + 12-byte token
                    # header (+ ~2 bytes VP8-like descriptor), not raw YUV
                    frame_bytes = TOKENS_PER_FRAME * 2 + TOKEN_HEADER_BYTES
                    self._total_bytes += frame_bytes

                    if self._first_frame_time is None:
                        self._first_frame_time = time.time()
                    self._last_frame_time = time.time()

                    chunk = VLMVideoChunk(
                        frame_index=self.frame_count,
                        timestamp=time.time(),
                        pts=frame.pts,
                        bytes_sent=int(frame_bytes),
                        width=frame.width,
                        height=frame.height
                    )
                    self._chunks.append(chunk)

                    return frame
            except StopIteration:
                logger.info("Video finished")
                self._video_send_complete = True
                raise MediaStreamError("No more frames")

    @property
    def total_bytes(self):
        return self._total_bytes

    @property
    def chunks(self):
        return self._chunks

    @property
    def first_frame_time(self):
        return self._first_frame_time

    @property
    def last_frame_time(self):
        return self._last_frame_time

    @property
    def video_send_complete(self):
        return self._video_send_complete

# ======================
# Realtime VLM Client
# ======================
class RealtimeWebRTCVLMClient:
    def __init__(
        self,
        signaling_host="127.0.0.1",
        signaling_port=1234,
        target_fps=10,
        model="liquid",
        network_profile='ideal_6g', # for result saving
        ice_iface=None, # restrict ICE host candidates to this interface (e.g. "lo")
    ):
        self.signaling_host = signaling_host
        self.signaling_port = signaling_port
        self.target_fps = target_fps
        self.model = model
        self.ice_iface = ice_iface

        self.pc = None
        self.signaling = None
        self.data_channel = None
        self.video_track = None
        self.video_sender = None

        self.session_metrics = None
        self.current_turn_metrics = None

        self._text_chunk_index = 0
        self._first_token_received = False
        self._inference_complete = False

        self._ice_connected = False
        self._dc_open_event = asyncio.Event()

        self._video_send_complete = False
        self._video_send_complete_time = None
        self._server_analysis_complete = False
        self.post_video_wait_timeout = 5.0

        self.network_profile = network_profile # for result saving
        self._bye_close_event = asyncio.Event()
        self._signaling_task = None
        # apply configured model before preloading the encoder bridge
        aicodec.set_model(self.model)
        TokenIdEncoder.preload_bridge()
        print(f"vlm model preloaded ({self.model})")

    async def _negotiate_model(self):
        """Model negotiation over TCP signaling (before SDP exchange).

        Sends {"type": "model", "model": <name>} and expects a ModelAck.
        The server rejects with ok=False + message if it cannot serve the
        requested model, in which case we abort the session.
        """
        logger.info(f"[Client] Negotiating model with server: {self.model}")
        await self.signaling.send({"type": "model", "model": self.model})
        deadline = time.time() + 30.0
        while time.time() < deadline:
            try:
                obj = await asyncio.wait_for(self.signaling.receive(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(obj, ModelAck):
                if obj.ok:
                    logger.info(f"[Client] Model negotiated: {self.model}")
                    return
                raise ConnectionError(
                    f"server rejected model '{self.model}': {obj.message}"
                )
            # ignore anything received before the ack (e.g. stray SDP)
            logger.debug(f"[Client] Ignoring pre-ack signaling message: {obj}")
        raise ConnectionError(f"model negotiation timeout for '{self.model}'")

    @property
    def provider(self) -> str:
        return "vlm_local"

    async def connect(self):
        t_start = time.time()
        self._ice_connected = False
        self._bye_close_event.clear()
        self.session_metrics = VLMSessionMetrics(
            session_id=f"vlm_{int(t_start)}",
            model=self.model,
            t_session_start=t_start,
        )

        # apply the ICE interface filter (if any) BEFORE candidate gathering;
        # must be set before createOffer() triggers gathering
        if self.ice_iface:
            from aicodec.register import set_ice_interface
            set_ice_interface(self.ice_iface)
            logger.info(f"[Client] ICE host candidates restricted to {self.ice_iface!r}")

        self.pc = RTCPeerConnection()

        # ICE connection state callback
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = self.pc.connectionState
            logger.info(f"[Client] Connection State: {state}")
            if state == "connected":
                self.session_metrics.t_webrtc_connected = time.time()
                self._ice_connected = True
                # print(f"ice state: {self._ice_connected}")
            elif state in ["disconnected", "failed"]:
                if self._bye_close_event.is_set():
                    logger.info("[Client] Connection closed by BYE")
                else:
                    logger.warning("[Client] Connection lost, attempting ICE restart...")
                    try:
                        offer = await self.pc.createOffer(iceRestart=True)
                        await self.pc.setLocalDescription(offer)
                        await self.signaling.send(RTCSessionDescription(type="offer", sdp=offer.sdp))
                    except Exception as e:
                        logger.error(f"[Client] ICE restart failed: {e}")
                        self._bye_close_event.set()

        # Data Channel
        self.data_channel = self.pc.createDataChannel("vlm-results")
        
        @self.data_channel.on("open")
        def on_open():
            logger.info("[DataChannel] Open")
            self._dc_open_event.set()
        
        @self.data_channel.on("message")
        def on_message(message):
            self._on_data_channel_message(message)

        transceiver = self.pc.addTransceiver("video", direction="sendonly")
        capabilities = RTCRtpSender.getCapabilities("video").codecs
        preferences = [c for c in capabilities if c.mimeType == "video/tokenIdLLM"]
        others = [c for c in capabilities if c.mimeType != "video/tokenIdLLM"]
        transceiver.setCodecPreferences(preferences + others)
        self.video_sender = transceiver.sender

        # Signaling
        self.signaling = TcpSignalingClient(self.signaling_host, self.signaling_port)
        t_signaling_start = time.time()
        await self.signaling.connect()
        self.session_metrics.t_signaling_connected = time.time()

        # Model negotiation: tell the server which VLM model to load,
        # then apply it locally before preloading the encoder bridge.
        await self._negotiate_model()

        # SDP Exchange
        await self.pc.setLocalDescription(await self.pc.createOffer())
        origin_sdp = self.pc.localDescription.sdp
        # strip nack
        modify_sdp = self._strip_nack(origin_sdp)
        # Create RTCSessionDescription object for signaling helper
        offer_obj = RTCSessionDescription(type="offer", sdp=modify_sdp)
        offer_bytes = await self.signaling.send(offer_obj)

        self.session_metrics.sdp_offer = modify_sdp
        self.session_metrics.sdp_offer_bytes = offer_bytes
        self.session_metrics.sdp_offer_hash = hashlib.sha256(modify_sdp.encode()).hexdigest()
        self.session_metrics.total_request_bytes += offer_bytes

        # Receive and process signaling messages (including ICE candidates)
        t_answer_recv = None
        signaling_deadline = time.time() + 60.0  # overall budget for answer + ICE
        while True:
            # If answer received and ICE connected, break
            if t_answer_recv and self.pc.connectionState == "connected":
                logger.info("[Client] ICE connected, breaking signaling loop")
                break
            if time.time() > signaling_deadline:
                logger.error("[Client] Signaling negotiation timed out")
                self._bye_close_event.set()
                break
            try:
                obj = await asyncio.wait_for(self.signaling.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except ConnectionError as e:
                logger.warning(f"[Client] Signaling Connection error: {e}")
                self._bye_close_event.set()
                break

            # process signaling
            if isinstance(obj, RTCSessionDescription):
                if obj.type == "answer":
                    await self.pc.setRemoteDescription(obj)

                    if t_answer_recv is None:
                        # first answer: record SDP metrics
                        self.session_metrics.sdp_answer = obj.sdp
                        # Estimate bytes for metrics
                        answer_dict = signaling_object_to_dict(obj)
                        answer_bytes = len(json.dumps(answer_dict).encode())
                        self.session_metrics.sdp_answer_bytes = answer_bytes
                        self.session_metrics.sdp_answer_hash = hashlib.sha256(obj.sdp.encode()).hexdigest()
                        self.session_metrics.total_response_bytes += answer_bytes

                        t_answer_recv = time.time()
                        self.session_metrics.sdp_negotiation_sec = t_answer_recv - t_signaling_start
                    else:
                        logger.info("[Client] Remote answer after ICE restart applied")

            elif isinstance(obj, RTCIceCandidate):
                try:
                    await self.pc.addIceCandidate(obj)
                    self.session_metrics.ice_candidates_received += 1
                except Exception as e:
                    logger.warning(f"[Client] addIceCandidate failed: {e}")

            elif obj is BYE:
                logger.info("[Signaling] Received bye from server")
                self._bye_close_event.set()
                if self.pc:
                    await self.pc.close()
                    logger.info("[Client] PeerConnection closed after receiving BYE")
                break

        try:
            await asyncio.wait_for(self._dc_open_event.wait(), timeout=30.0)
            logger.info("[Client] Data Channel is ready")
        except asyncio.TimeoutError:
            logger.error("[Client] Timeout waiting for Data Channel open")
            raise ConnectionError("Data Channel failed to open")

        # background signaling listener: keeps consuming signaling messages
        # after connect() returns, so late ICE candidates, ICE-restart offers
        # (answered locally) and the server's BYE are all handled promptly
        self._signaling_task = asyncio.create_task(self._signaling_listener())
        logger.info(f"[Client] Connected: session={self.session_metrics.session_id}")
        return self.session_metrics

    async def _signaling_listener(self):
        try:
            while not self._bye_close_event.is_set():
                if self.pc and self.pc.connectionState in ["closed", "failed"]:
                    break
                try:
                    obj = await asyncio.wait_for(self.signaling.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except (ConnectionError, OSError) as e:
                    logger.info(f"[Client] Signaling closed: {e}")
                    break

                if isinstance(obj, RTCSessionDescription):
                    if obj.type == "answer":
                        try:
                            await self.pc.setRemoteDescription(obj)
                            logger.info("[Client] Applied late/renegotiated answer")
                        except Exception as e:
                            logger.warning(f"[Client] setRemoteDescription failed: {e}")
                elif isinstance(obj, RTCIceCandidate):
                    try:
                        await self.pc.addIceCandidate(obj)
                        self.session_metrics.ice_candidates_received += 1
                    except Exception as e:
                        logger.warning(f"[Client] addIceCandidate failed: {e}")
                elif obj is BYE:
                    logger.info("[Signaling] Received bye from server")
                    self._bye_close_event.set()
                    break
        except asyncio.CancelledError:
            pass

    def _strip_nack(self, sdp):
        """Remove rtx payload mappings without corrupting other lines.

        Only drops lines that define/enable rtx:
          - a=rtpmap:<pt> rtx/...
          - a=fmtp:<pt> apt=<...>   (rtx points back at its associated payload)
        Generic lines merely containing the substring "rtx"/"apt=" are kept.
        """
        lines = sdp.splitlines()
        new_lines = []
        for line in lines:
            low = line.lower()
            if low.startswith("a=rtpmap:") and " rtx/" in low + " ":
                continue
            if low.startswith("a=fmtp:") and "apt=" in low:
                continue
            new_lines.append(line)
        return "\n".join(new_lines) + "\n"

    def _on_data_channel_message(self, message):
        try:
            # Normalize: aiortc delivers str for text frames, bytes for binary
            if isinstance(message, bytes):
                if message == b"ping":
                    return
                try:
                    text_data = message.decode('utf-8')
                except UnicodeDecodeError:
                    text_data = message.decode('utf-8', errors='ignore')
                is_binary = True
            elif isinstance(message, str):
                if message in ("ping", "pong"):
                    return
                text_data = message
                is_binary = False
            else:
                return

            # Control messages first: must not pollute token/byte metrics.
            # Check the FULL payload for JSON before stripping the 6-byte header.
            if text_data.startswith('{'):
                try:
                    data = json.loads(text_data)
                    if isinstance(data, dict) and data.get("type") == "VIDEO_ANALYSIS_COMPLETE":
                        logger.info(f"[Client] Received VIDEO_ANALYSIS_COMPLETE from server: {data}")
                        self._server_analysis_complete = True
                        self._inference_complete = True
                        # do not overwrite t_inference_done: the per-question
                        # [DONE] sentinel already recorded the real end of
                        # generation; only fall back if nothing was received
                        if self.current_turn_metrics and self.current_turn_metrics.t_inference_done is None:
                            if self.current_turn_metrics.t_last_token:
                                self.current_turn_metrics.t_inference_done = self.current_turn_metrics.t_last_token
                            else:
                                self.current_turn_metrics.t_inference_done = time.time()
                        return
                except json.JSONDecodeError:
                    pass

            # End-of-stream sentinel: marks inference end, never counts as text
            if text_data == "[DONE]":
                if self.current_turn_metrics:
                    if self.current_turn_metrics.t_inference_done is None and self.current_turn_metrics.t_last_token:
                        self.current_turn_metrics.t_inference_done = self.current_turn_metrics.t_last_token
                print(">", end="", flush=True)
                return

            # Token stream chunk: 4-byte pts header + 2-byte token count + text.
            # Strip the 6-byte header BEFORE text handling so
            # output_text/bytes_received stay clean.
            if is_binary and len(message) >= 7:
                try:
                    token_number = unpack_from("!H", message, 4)
                    if self.current_turn_metrics:
                        self.current_turn_metrics.token_count += token_number[0]
                except Exception:
                    pass
                message = message[6:]
                try:
                    text_data = message.decode('utf-8')
                except UnicodeDecodeError:
                    text_data = message.decode('utf-8', errors='ignore')

            if self.current_turn_metrics:
                if not self._first_token_received:
                    self.current_turn_metrics.t_first_token = time.time()
                    self._first_token_received = True

                chunk = VLMTextChunk(
                    content=text_data,
                    timestamp=time.time(),
                    index=self._text_chunk_index,
                    bytes_received=len(text_data.encode()),
                )
                self.current_turn_metrics.text_chunks.append(chunk)
                self._text_chunk_index += 1

                self.current_turn_metrics.output_text += text_data
                self.current_turn_metrics.text_bytes_received += chunk.bytes_received
                self.current_turn_metrics.t_last_token = chunk.timestamp

        except Exception as e:
            logger.error(f"[DataChannel] Parse error: {e}")
            if self.current_turn_metrics:
                self.current_turn_metrics.success = False
                self.current_turn_metrics.error_message = str(e)

    async def analyze_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ):
        if not self.pc:
            raise RuntimeError("Not connected")

        if not self.video_sender:
            raise RuntimeError("Video sender not initialized. Call connect() first.")

        try:
            video_path_obj = Path(video_path)
            video_name = video_path_obj.parent.name
            bench_config_path = str(video_path_obj.parent / "questions.json")
            video_info = {
                "type": "video_info",
                "path": video_path,
                "name": video_name,
                "bench_config": bench_config_path,
                "network_profile": self.network_profile, # for result saving
            }
            # Send video_info to server via Data Channel
            if self.data_channel and self.data_channel.readyState == 'open':
                video_info_json = json.dumps(video_info)
                self.data_channel.send(video_info_json)
                logger.info(f"[Client] Sent video_info: {video_info}")
            else:
                logger.warning("[Client] Data channel not open, cannot send video_info")
        except Exception as e:
            logger.error(f"[Client] Failed to construct or send video_info: {e}")

        t_turn_start = time.time()
        turn_index = len(self.session_metrics.turns)

        self.current_turn_metrics = VLMTurnMetrics(
            turn_index=turn_index,
            video_source=video_path,
            t_turn_start=t_turn_start,
        )
        self._text_chunk_index = 0
        self._first_token_received = False
        self._inference_complete = False
		
        self._video_send_complete = False
        self._video_send_complete_time = None
        self._server_analysis_complete = False

        # keep pts monotonic across turns on the same sender: continue from
        # the last pts of the previous track so downstream timestamp mappers
        # and question matching don't see time going backwards
        prev_last_pts = 0
        if self.video_track is not None:
            prev_last_pts = self.video_track.last_pts or 0

        self.video_track = realtimeVideoStreamTrack(
            video_path,
            self.target_fps,
            pts_offset=prev_last_pts,
        )

        self.video_sender.replaceTrack(self.video_track)
        logger.info("[Client] Video track replaced")

        while True:
            if self._bye_close_event.is_set():
                logger.info("[Client] Shutdown event detected, stopping video sent")
                break
            if self.pc.connectionState in ["failed", "closed"]:
                logger.warning(f"[Client] WebRTC Connection State: {self.pc.connectionState}, breaking loop")
                self._bye_close_event.set()
                break
            await asyncio.sleep(0.1)
            try:
                if self.video_track:
                    if self.video_track.frame_count >= 1:
                        if self.current_turn_metrics.t_first_frame_sent is None:
                            self.current_turn_metrics.t_first_frame_sent = self.video_track.first_frame_time
                        self.current_turn_metrics.t_last_frame_sent = self.video_track.last_frame_time
            except:
                pass

            if max_frames and self.video_track.frame_count >= max_frames:
                logger.info(f"[Client] Reached max_frames: {max_frames}")
                self._video_send_complete = True
                self._video_send_complete_time = time.time()
                break
            
            if self.video_track.video_send_complete:
                logger.info("[Client] Video track send complete")
                self._video_send_complete = True
                self._video_send_complete_time = time.time()
                break
            if self._inference_complete:
                break
        
        if self.video_track:
            self.current_turn_metrics.video_frames_sent = self.video_track.frame_count
            self.current_turn_metrics.video_bytes_sent = self.video_track.total_bytes
            self.current_turn_metrics.video_resolution = f"{self.video_track.stream.width}x{self.video_track.stream.height}"
            self.current_turn_metrics.video_fps = float(self.video_track.fps)
            self.current_turn_metrics.video_chunks = self.video_track.chunks

        if self._video_send_complete_time:
            logger.info(f"[Client] Video send complete at {self._video_send_complete_time}")
            logger.info(f"[Client] Waiting for server's VIDEO_ANALYSIS_COMPLETE (timeout={self.post_video_wait_timeout}s)...")
            wait_start = time.time()
            while not self._server_analysis_complete:
                if self._bye_close_event.is_set():
                    logger.info("[Client] Shutdown event detected during wait")
                    break
                await asyncio.sleep(0.1)
                if time.time() - wait_start > self.post_video_wait_timeout:
                    logger.warning(f"[Client] Timeout waiting for VIDEO_ANALYSIS_COMPLETE after {self.post_video_wait_timeout}s")
                    if self.current_turn_metrics:
                        self.current_turn_metrics.t_inference_done = time.time()
                    self._inference_complete = True
                    break
                
        if self.current_turn_metrics and self.current_turn_metrics.t_inference_done is None:
            if self.current_turn_metrics.t_last_token:
                self.current_turn_metrics.t_inference_done = self.current_turn_metrics.t_last_token
            else:
                self.current_turn_metrics.t_inference_done = time.time()

        self.session_metrics.turns.append(self.current_turn_metrics)
        self.session_metrics.total_turns += 1
        self.session_metrics.total_video_frames_sent += self.current_turn_metrics.video_frames_sent
        self.session_metrics.total_video_bytes_sent += self.current_turn_metrics.video_bytes_sent
        self.session_metrics.total_text_bytes_received += self.current_turn_metrics.text_bytes_received
        self.session_metrics.total_request_bytes += self.current_turn_metrics.video_bytes_sent
        self.session_metrics.total_response_bytes += self.current_turn_metrics.text_bytes_received
        self.session_metrics.total_token_count += self.current_turn_metrics.token_count

        return self.current_turn_metrics

    async def disconnect(self):
        if self.session_metrics:
            self.session_metrics.t_session_end = time.time()

        # stop the background signaling listener first
        if self._signaling_task and not self._signaling_task.done():
            self._signaling_task.cancel()
            try:
                await asyncio.wait_for(self._signaling_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        if self.signaling:
            try:
                await self.signaling.send(BYE)
            except:
                pass
            await self.signaling.close()

        if self.pc:
            await self.pc.close()

        logger.info("[Client] Disconnected")
        return self.session_metrics

    def get_session_report(self):
        if not self.session_metrics:
            return {}

        report = {
            "session": {
                "session_id": self.session_metrics.session_id,
                "model": self.session_metrics.model,
                "session_duration": self.session_metrics.session_duration,
                "connection_setup_time": self.session_metrics.connection_setup_time,
                "sdp_negotiation_sec": self.session_metrics.sdp_negotiation_sec,
                "total_turns": self.session_metrics.total_turns,
                "ice_candidates_received": self.session_metrics.ice_candidates_received,
                "total_video_bytes": self.session_metrics.total_video_bytes_sent,
                "total_text_bytes": self.session_metrics.total_text_bytes_received,
                "avg_ttft": self.session_metrics.avg_ttft,
                "avg_turn_latency": self.session_metrics.avg_turn_latency,
            },
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "video_source": t.video_source,
                    "ttft": t.ttft,
                    "ttlt": t.ttlt,
                    "total_latency": t.total_latency,
                    "video_frames": t.video_frames_sent,
                    "video_bytes": t.video_bytes_sent,
                    "text_bytes": t.text_bytes_received,
                    "success": t.success,
                }
                for t in self.session_metrics.turns
            ],
        }
        return report
