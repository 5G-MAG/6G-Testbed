import asyncio
import argparse
import logging
import colorlog
import signal
import json
import sys
import os
import time
from pathlib import Path
from typing import Optional, Set, Any, Dict

# Bootstrap the sibling aicodec package relative to this file's location,
# so the server runs without pip-installing it and without PYTHONPATH.
# Works from any CWD (this file lives in <repo>/aitestbed/server/).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AICODEC_PATH = _REPO_ROOT / "aicodec"
if str(_AICODEC_PATH) not in sys.path:
    sys.path.insert(0, str(_AICODEC_PATH))

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamError
from aiortc.contrib.signaling import BYE
import aicodec
from aicodec.codec import AsyncTokenIdDecoder
from aicodec import shared

aicodec.register()
import yaml


# ======================
# Logging Setup
# ======================
def setup_logger(name: str) -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(asctime)s - %(name)s - %(log_color)s%(levelname)-8s%(reset)s - %(message)s",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    # File log
    Path('logs').mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler('logs/server.log', mode='w', encoding='utf-8')
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logger('VlmServer')

# ======================
# Signaling Server
# ======================
class TcpSignalingServer:
    def __init__(self, host: str, port: int, persistent: bool = False):
        self._host = host
        self._port = port
        self._persistent = persistent
        self._server: Optional[asyncio.Server] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = asyncio.Event()
        self._stop_event = asyncio.Event()

    @staticmethod
    def _object_to_dict(obj: Any) -> dict:
        if isinstance(obj, RTCSessionDescription):
            return {"type": obj.type, "sdp": obj.sdp}
        elif isinstance(obj, RTCIceCandidate):
            return {
                "candidate": obj.candidate,
                "sdpMid": obj.sdpMid,
                "sdpMLineIndex": obj.sdpMLineIndex
            }
        elif obj is BYE:
            return {"type": "bye"}
        else:
            return obj

    @staticmethod
    def _dict_to_object(data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        obj_type = data.get("type")
        if obj_type in ["offer", "answer", "pranswer", "rollback"]:
            return RTCSessionDescription(sdp=data["sdp"], type=obj_type)
        elif obj_type == "candidate":
            return RTCIceCandidate(
                candidate = data["candidate"],
                sdpMid = data.get("sdpMid"),
                sdpMLineIndex = data.get("sdpMLineIndex")
            )
        elif obj_type == "bye":
            return BYE
        elif obj_type == "model":
            return {"type": "model", "model": data.get("model", "")}
        else:
            return data

    async def connect(self):
        self._connected.clear()
        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            self._reader = reader
            self._writer = writer
            self._connected.set()
            logger.info("[Signaling Server] Client connected, waiting for messages...")
            try:
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"[Signaling Server] Handler error: {e}")
            finally:
                await self._close_connection()
                if self._persistent:
                    logger.info("[Signaling Server] Client disconnected, waiting for new connection...")
                    self._connected.clear()

        self._server = await asyncio.start_server(
            handle_client, self._host, self._port, reuse_address=True, backlog=100
        )
        logger.info(f"[Signaling Server] Listening on {self._host}:{self._port}...")

        if not self._persistent:
            await self._connected.wait()
            logger.info("[Signaling Server] Client connected, ready to receive.")


    async def _close_connection(self):
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except:
                pass
        self._reader = None
        self._writer = None
        self._connected.clear()

    async def receive(self):
        if not self._reader or not self._connected.is_set():
            raise ConnectionError("Not connected")
        try:
            data = await asyncio.wait_for(self._reader.readuntil(b'\n'), timeout=5.0)
            parsed = json.loads(data.decode('utf8').strip())
            return self._dict_to_object(parsed)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError("Receive timeout")
        except asyncio.IncompleteReadError:
            raise ConnectionError("Client disconnected")
        except json.JSONDecodeError as e:
            logger.error(f"[Signaling Server] JSON decode error: {e}")
            raise

    async def send(self, obj: Any):
        if not self._writer or not self._connected.is_set():
            raise ConnectionError("Not connected")
        try:
            serializable_obj = self._object_to_dict(obj)
            message = json.dumps(serializable_obj) + '\n'
            self._writer.write(message.encode('utf8'))
            await self._writer.drain()
        except Exception as e:
            logger.error(f"[Signaling Server] Send error: {e}")
            raise

    async def close(self):
        logger.info("[Signaling Server] Closing...")
        self._stop_event.set()
        await self._close_connection()
        if self._server:
            self._server.close()
            try:
                await self._server.wait_closed()
            except:
                pass

    async def wait_for_client(self, timeout: Optional[float] = None) -> bool:
        logger.info("[Signaling Server] Waiting for new client connection...")
        try:
            if timeout:
                await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            else:
                await self._connected.wait()
            logger.info("[Signaling Server] New client connected!")
            return True
        except asyncio.TimeoutError:
            logger.info("[Signaling Server] Wait for client timeout")
            return False
        except asyncio.CancelledError:
            logger.info("[Signaling Server] Wait for client cancelled")
            return False


# ======================
# Main Server
# ======================
class VlmServer:
    def __init__(self, host: str, port: int, persistent: bool, preload_bridge: bool, model: str = "liquid"):
        self.host = host
        self.port = port
        self.persistent_mode = persistent
        self.preload_bridge = preload_bridge
        self.default_model = model  # used when a client does not negotiate a model
        self.stop_event = asyncio.Event()
        self.background_tasks: Set[asyncio.Task] = set()
        self.signaling: Optional[TcpSignalingServer] = None
        self.is_shutting_down = False
        self.session_count = 0
        self.scenario_name = "realtime_video_understanding"

    def load_task_config(self):
        config_path = "configs/scenarios.yaml"
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.scenarios_config = yaml.safe_load(f)
        scenario_config = self.scenarios_config["scenarios"].get(self.scenario_name)
        if not scenario_config:
            raise ValueError(f"Unknown scenario: {self.scenario_name}")
        task_config = scenario_config.get("task_config", {})
        task_class = task_config.get("task_class", "real")
        return task_class



    def save_results(self):
        logger.info(f"[Server] Attempting to save benchmark results.")
        if AsyncTokenIdDecoder.bench_config and AsyncTokenIdDecoder.result_output_path:
            try:
                os.makedirs(os.path.dirname(AsyncTokenIdDecoder.result_output_path), exist_ok=True)
                with open(AsyncTokenIdDecoder.result_output_path, 'w', encoding='utf-8') as f:
                    json.dump(AsyncTokenIdDecoder.bench_config, f, indent=4, ensure_ascii=False)
                logger.info(f"[Server] Benchmark results saved to {AsyncTokenIdDecoder.result_output_path}")
            except Exception as e:
                logger.error(f"[Server] Failed to save results: {e}")
        else:
            logger.info("[Server] No benchmark config or output path set, skipping save.")

    async def shutdown(self):
        if self.is_shutting_down:
            return
        self.is_shutting_down = True
        logger.info("\n=== Starting program cleanup ===")
        self.stop_event.set()

        logger.info("Cancelling background tasks...")
        tasks_to_cancel = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks_to_cancel:
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()

        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.info("Task cleanup timeout")
            except Exception as e:
                logger.error(f"Task gather error: {e}")

        self.background_tasks.clear()
        logger.info("Background tasks cleared")

        if self.signaling:
            try:
                await self.signaling.close()
                logger.info("Signaling closed")
            except Exception as e:
                logger.error(f"Signaling close error: {e}")
        if self.preload_bridge:
            try:
                AsyncTokenIdDecoder.cleanup_bridge()
            except Exception as e:
                logger.error(f"Bridge cleanup error: {e}")

        logger.info("=== Program cleanup completed ===")

    async def run(self):
        # model loading is deferred to session negotiation: the client tells
        # us which model it needs ({"type": "model", ...}) before the SDP
        # exchange; --model / default_model is the fallback.
        task_class = self.load_task_config()
        AsyncTokenIdDecoder.task_class = task_class
        AsyncTokenIdDecoder.reset_decoder_state(model_name=self.default_model)

        self.signaling = TcpSignalingServer(self.host, self.port, persistent=self.persistent_mode)
        logger.info(f"[Answer] Initialized Signaling Server on {self.host}:{self.port}")

        if self.persistent_mode:
            logger.info("[Answer] PERSISTENT MODE ENABLED - Will wait for multiple clients")
            await self.signaling.connect()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        try:
            if self.persistent_mode:
                while not self.stop_event.is_set():
                    self.session_count += 1
                    logger.info(f"\n{'=' * 60}")
                    logger.info(f"[Answer] Starting session #{self.session_count}")
                    logger.info(f"{'=' * 60}\n")

                    pc = RTCPeerConnection()
                    session = Session(pc, self.signaling, self)

                    try:
                        success = await session.run()
                        if success:
                            logger.info(f"[Answer] Session #{self.session_count} completed normally")
                        else:
                            logger.warning(f"[Answer] Session #{self.session_count} ended early")
                    except Exception as e:
                        logger.error(f"[Answer] Session #{self.session_count} error: {e}")
                        logger.exception("Session error traceback")
                    finally:
                        try:
                            await session.cleanup()
                        except Exception as e:
                            logger.error(f"[Answer] Session #{self.session_count} cleanup error: {e}")
                    logger.info(f"\n[Answer] Waiting for next client connection...\n")
                    await asyncio.sleep(1.0)
            else:
                pc = RTCPeerConnection()
                session = Session(pc, self.signaling, self)
                try:
                    await session.run()
                finally:
                    await session.cleanup()
        except asyncio.CancelledError:
            logger.info("Main task cancelled by signal")
        finally:
            self.save_results()
            await self.shutdown()


# ======================
# Session Handler
# ======================
class Session:
    def __init__(self, pc: RTCPeerConnection, signaling: TcpSignalingServer, server_ref: VlmServer):
        self.pc = pc
        self.signaling = signaling
        self.server_ref = server_ref
        self.reset_video_state()
        self._session_stop_event = asyncio.Event()
        self._is_clean_up = False
        self._session_tasks: Set[asyncio.Task] = set()
        self._close_session_reason: str = "normal"
        self._pending_offer = None  # legacy client: SDP received before/during model negotiation

    def reset_video_state(self):
        self.video_state = {
            "video_track": None,
            "video_ended": False,
            "t_video_start": None,
            "t_video_end": None,
            "analysis_complete": False
        }

    async def _negotiate_model(self) -> Optional[str]:
        """Model negotiation: wait for {"type": "model"} and ACK/reject it.

        Returns the negotiated model name, or None if the session must be
        aborted (client rejected/unknown model or disconnected).  Once the
        model is known it is applied to the decoder (deferred loading).
        """
        from aicodec import available_models

        model_name = None
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if self.server_ref.stop_event.is_set() or self._session_stop_event.is_set():
                return None
            try:
                obj = await asyncio.wait_for(self.signaling.receive(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except (ConnectionError, Exception) as e:
                logger.error(f"[Model] Signaling error: {e}")
                return None

            if isinstance(obj, dict) and obj.get("type") == "model":
                model_name = obj.get("model", "")
                break
            elif isinstance(obj, RTCSessionDescription):
                # legacy client without negotiation: fall back to default model
                logger.info(f"[Model] Client sent SDP without model negotiation, "
                            f"using default '{self.server_ref.default_model}'")
                model_name = self.server_ref.default_model
                # put the message back by handling it via a pending-offer stash
                self._pending_offer = obj
                break
            elif obj is BYE:
                logger.info("[Model] Peer left before negotiation")
                return None

        if model_name is None:
            logger.error("[Model] Negotiation timeout")
            await self.signaling.send({
                "type": "model_ack", "ok": False,
                "message": "model negotiation timeout",
            })
            return None

        if model_name not in available_models():
            msg = (f"unknown model '{model_name}'; "
                   f"available: {available_models()}")
            logger.error(f"[Model] {msg}")
            await self.signaling.send({
                "type": "model_ack", "ok": False, "message": msg,
            })
            return None

        # apply model + load bridge (deferred load happens here)
        AsyncTokenIdDecoder.reset_decoder_state(model_name=model_name)
        if self.server_ref.preload_bridge:
            AsyncTokenIdDecoder.preload_bridge()
            logger.info(f"[Model] vlm model preloaded ({model_name})")
        await self.signaling.send({"type": "model_ack", "ok": True})
        logger.info(f"[Model] Negotiated model: {model_name}")
        return model_name

    async def on_video_analysis_complete(self):
        if self.video_state["analysis_complete"]:
            logger.info("[Server] Video analysis already complete, skipping")
            return

        self.video_state["video_ended"] = True
        self.video_state["t_video_end"] = time.time()
        self.video_state["analysis_complete"] = True

        logger.info("[Server] === Video Analysis Complete ===")
        logger.info(f"[Server]  Start time: {self.video_state['t_video_start']}")
        logger.info(f"[Server]  End time: {self.video_state['t_video_end']}")
        if self.video_state["t_video_start"]:
            logger.info(
                f"[Server]  Duration: {self.video_state['t_video_end'] - self.video_state['t_video_start']:.2f}s")

        if AsyncTokenIdDecoder.data_channel and AsyncTokenIdDecoder.data_channel.readyState == "open":
            try:
                complete_message = {
                    "type": "VIDEO_ANALYSIS_COMPLETE",
                    "timestamp": self.video_state["t_video_end"],
                    "duration_sec": self.video_state["t_video_end"] - (self.video_state["t_video_start"] or 0)
                }
                AsyncTokenIdDecoder.data_channel.send(json.dumps(complete_message).encode())
                logger.info("[Server] Sent VIDEO_ANALYSIS_COMPLETE to client")
            except Exception as e:
                logger.error(f"[Server] Failed to send complete message: {e}")

    async def run(self) -> bool:
        shared.loop = asyncio.get_running_loop()
        self._is_clean_up = False
        self._session_stop_event.clear()
        self.server_ref.stop_event.clear()
        self.reset_video_state()
        AsyncTokenIdDecoder.reset_decoder_state()

        def track_task(task: asyncio.Task):
            self._session_tasks.add(task)
            task.add_done_callback(self._session_tasks.discard)

        async def heartbeat_loop(pc, channel):
            try:
                while pc.connectionState not in ["closed", "failed"]:
                    if channel.readyState == "open":
                        try:
                            channel.send(b"ping")
                        except Exception as e:
                            logger.error(f"[Heartbeat] Send error: {e}")
                            break
                    elif channel.readyState == "closed":
                        break
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass
            finally:
                logger.info("[Heartbeat] Loop stopped")


        @self.pc.on("datachannel")
        def on_datachannel(channel):
            # bind the channel to this session only; a stale channel from a
            # previous session must not receive this session's inference output
            if AsyncTokenIdDecoder.data_channel is not None and AsyncTokenIdDecoder.data_channel is not channel:
                logger.warning("[Server] Replacing stale data channel from a previous session")
            AsyncTokenIdDecoder.data_channel = channel

            @channel.on("open")
            def on_open():
                logger.info(f"Data_channel status: {channel.readyState}")
                hb_task = asyncio.create_task(heartbeat_loop(self.pc, channel))
                track_task(hb_task)

            @channel.on("message")
            def on_message(message):
                try:
                    msg_data = json.loads(message)
                    if msg_data.get("type") == "video_info":
                        video_path = msg_data.get("path")
                        video_name = msg_data.get("name")
                        bench_config_path = msg_data.get("bench_config")
                        network_profile = msg_data.get("network_profile", "no_emulation")
                        logger.info(f"Received video path: {video_path}")

                        AsyncTokenIdDecoder.current_video_path = video_path
                        AsyncTokenIdDecoder.current_video_name = video_name

                        if bench_config_path:
                            try:
                                with open(bench_config_path, 'r', encoding='utf-8') as f:
                                    bench_config = json.load(f)
                                AsyncTokenIdDecoder.bench_config = bench_config
                                bench_config_dir = os.path.dirname(os.path.abspath(bench_config_path))
                                bench_config_name = os.path.splitext(os.path.basename(bench_config_path))[0]
                                AsyncTokenIdDecoder.result_output_path = os.path.join(
                                    bench_config_dir,
                                    f"{bench_config_name}_results_{network_profile}.json"
                                )
                                logger.info(f"[Server] Loaded benchmark config from {bench_config_path}")
                                logger.info(f"[Server] Results will be saved to  {AsyncTokenIdDecoder.result_output_path}")
                            except Exception as e:
                                logger.error(f"[Server] Failed to load benchmark config: {e}")
                        else:
                            logger.info(f"[Server] No benchmark config provided")
                    else:
                        logger.info(f"[Command]: {message}")
                except json.JSONDecodeError:
                    logger.info(f"[Command]: {message}")

        @self.pc.on("track")
        def on_track(track):
            logger.info(f"Receiving {track.kind}")
            if track.kind == "video":
                self.video_state["video_track"] = track
                self.video_state["t_video_start"] = time.time()
                logger.info(f"[Server] Video track received, start time: {self.video_state['t_video_start']}")

                @track.on("ended")
                def on_video_ended():
                    logger.info("[Server] === Video Track ENDED Event Triggered === ")
                    analysis_complete_task = asyncio.create_task(self.on_video_analysis_complete())
                    track_task(analysis_complete_task)

            async def consume_remote():
                try:
                    while not (self.server_ref.stop_event.is_set() or self._session_stop_event.is_set()) and not self._is_clean_up:
                        try:
                            await asyncio.wait_for(track.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue
                        except MediaStreamError:
                            logger.info("remote track ended")
                            self._session_stop_event.set()
                            break
                        except Exception as e:
                            logger.error(f"Track recv error: {e}")
                            break
                except asyncio.CancelledError:
                    pass
            task = asyncio.create_task(consume_remote())
            track_task(task)

        try:
            if self.server_ref.persistent_mode:
                connected = await self.signaling.wait_for_client(timeout=30.0)
                if not connected:
                    logger.info("[Answer] No client connected, ending session")
                    return False
            else:
                await self.signaling.connect()
            logger.info("[Server] Signaling connected successfully.")
        except Exception as e:
            logger.error(f"[Server] Signaling connect failed: {e}")
            return False

        # ---- model negotiation: client announces its model before the offer ----
        negotiated_model = await self._negotiate_model()
        if negotiated_model is None:
            return False

        # legacy client path: SDP arrived without negotiation, handle stashed offer
        obj = self._pending_offer
        self._pending_offer = None
        while obj is not None or not (self.server_ref.stop_event.is_set() or self._session_stop_event.is_set()) and not self._is_clean_up:
            if obj is None:
                try:
                    obj = await asyncio.wait_for(self.signaling.receive(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                except ConnectionError as e:
                    logger.error(f"[Answer] Client disconnected: {e}")
                    return False
                except Exception as e:
                    logger.error(f"[Answer] Signaling receive error: {e}")
                    return False

            if isinstance(obj, RTCSessionDescription) and obj.type == "offer":
                logger.info("[Answer] Received offer, processing...")
                await self.pc.setRemoteDescription(obj)
                await self.pc.setLocalDescription(await self.pc.createAnswer())
                logger.info("[Answer] Sending answer...")
                await self.signaling.send(self.pc.localDescription)
                break
            elif isinstance(obj, RTCIceCandidate):
                await self.pc.addIceCandidate(obj)
            elif obj is BYE:
                logger.info("Peer left")
                return False
            obj = None  # consumed non-offer message, fetch the next one

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state is {self.pc.connectionState}")
            if self.pc.connectionState in ["closed", "failed"]:
                logger.info("[Server] Connection closed, saving benchmark results...")
                self._close_session_reason = "connection failed"
                logger.info("[Server] Connection closed, benchmark results saved!!!")
                self._session_stop_event.set()

        logger.info("[Server] Entering ICE candidate exchange loop...")
        while not (self.server_ref.stop_event.is_set() or self._session_stop_event.is_set()) and not self._is_clean_up:
            try:
                obj = await asyncio.wait_for(self.signaling.receive(), timeout=2.0)
                logger.debug("[Server] signaling received")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("Signaling receive cancelled")
                break
            except ConnectionError as e:
                logger.error(f"[Server] Peer disconnected: {e}")
                return False
            except Exception as e:
                logger.error(f"Signaling error: {e}")
                break


            if isinstance(obj, RTCSessionDescription):
                if obj.type == "offer":
                    # ICE restart offer: renegotiate and send a fresh answer
                    logger.info("[Server] Received offer (ICE restart), answering...")
                    await self.pc.setRemoteDescription(obj)
                    await self.pc.setLocalDescription(await self.pc.createAnswer())
                    await self.signaling.send(self.pc.localDescription)
                else:
                    await self.pc.setRemoteDescription(obj)
            elif isinstance(obj, RTCIceCandidate):
                try:
                    await self.pc.addIceCandidate(obj)
                except Exception as e:
                    logger.warning(f"[Server] addIceCandidate failed: {e}")
            elif obj is BYE:
                logger.info("Peer left")
                return False

        return True

    async def cleanup(self):
        logger.info(f"\n === Session cleanup ===")
        self._session_stop_event.set()
        self._is_clean_up = True

        # drain pending inference work before saving: the last question's
        # answer is written into bench_config by the inference worker, which
        # may still be generating when the connection drops
        try:
            from aicodec.shared import get_inference_queue
            queue = get_inference_queue()
            deadline = time.time() + 30.0
            while not queue.empty() and time.time() < deadline:
                await asyncio.sleep(0.5)
            # extra grace for an in-flight generation
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.warning(f"[Cleanup] Inference queue drain failed: {e}")

        logger.info("[Server] Connection closed, saving benchmark results...")
        self.server_ref.save_results()

        logger.info(f"[Cleanup] Sending close notification to client (reason: {self._close_session_reason})")
        try:
            await asyncio.wait_for(self.signaling.send(BYE), timeout=2.0)
            logger.info("[Cleanup] Close notification sent to client")
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"[Cleanup] Failed to send close notification: {e}")

        logger.info("Cancelling background tasks...")
        tasks_to_cancel = [t for t in list(self._session_tasks) if t is not asyncio.current_task()]
        for task in tasks_to_cancel:
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()

        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.info("Task cleanup timeout")
            except Exception as e:
                logger.error(f"Task gather error: {e}")


        logger.info("Cleaning up AsyncTokenIdDecoder...")
        try:
            if AsyncTokenIdDecoder.data_channel:
                try:
                    AsyncTokenIdDecoder.data_channel.close()
                except Exception as e:
                    logger.error(f"data_channel close error: {e}")

            AsyncTokenIdDecoder.reset_decoder_state()
        except Exception as e:
            logger.error(f"AsyncTokenIdDecoder cleanup error: {e}")

        logger.info("Closing session resources...")
        if self.pc:
            try:
                await asyncio.wait_for(self.pc.close(), timeout=1.0)
                logger.info("PeerConnection closed")
            except Exception as e:
                logger.error(f"PC close error: {e}")
        self.reset_video_state()

        self._is_clean_up = False
        self._session_stop_event.clear()
        logger.info("=== Session cleanup completed === ")


# ======================
# Entry point
# ======================
async def main():
    parser = argparse.ArgumentParser(description="RTP Data Transport")
    parser.add_argument("--signaling-host", default="127.0.0.1")
    parser.add_argument("--signaling-port", type=int, default=1234)
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction, default=True,
                        help="keep serving multiple clients after a session ends (use --no-persistent for a single session)")
    parser.add_argument("--preload-bridge", action=argparse.BooleanOptionalAction, default=True,
                        help="preload the VLM model bridge (use --no-preload-bridge to load lazily)")
    parser.add_argument("--model", default="liquid",
                        help="default VLM model (used when the client does not negotiate one)")
    parser.add_argument("--ice-interface", default=None,
                        help="restrict ICE host candidates to this network interface (e.g. 'lo'); "
                             "default gathers on all interfaces (stock behaviour)")
    args = parser.parse_args()

    # apply the ICE interface filter before any RTCPeerConnection is created
    if args.ice_interface:
        from aicodec.register import set_ice_interface
        set_ice_interface(args.ice_interface)
        logger.info(f"[Server] ICE host candidates restricted to {args.ice_interface!r}")

    server = VlmServer(
        host = args.signaling_host,
        port = args.signaling_port,
        persistent=args.persistent,
        preload_bridge=args.preload_bridge,
        model=args.model,
    )

    try:
        await server.run()
    except Exception as e:
        logger.error(f"Main Exception: {e}")
        logger.exception("Main exception traceback")
        raise
    finally:
        logger.info("Event loop closed. Exit.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received")
        sys.exit(0)
    except Exception:
        sys.exit(1)
