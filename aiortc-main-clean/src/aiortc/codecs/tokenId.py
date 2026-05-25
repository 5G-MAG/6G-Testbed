import logging
import random
from struct import pack, unpack_from
from typing import Optional, Type, TypeVar, cast

from av import CodecContext, VideoFrame
from av.frame import Frame
from av.packet import Packet

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE, convert_timebase
from .base import Decoder, Encoder

import numpy as np
import time

import asyncio
import concurrent.futures

import aiortc.shared as shared
from ..shared import get_inference_queue, get_liquid_bridge

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BITRATE = 500000
MIN_BITRATE = 250000
MAX_BITRATE = 1500000
PACKET_MAX = 1100 # Max RTP payload size

# Reuse VP8 payload descriptor
DESCRIPTOR_T = TypeVar("DESCRIPTOR_T", bound="VpxPayloadDescriptor")

# ==============================
# 1. reuse VP8 Payload Descriptor
# ==============================
class VpxPayloadDescriptor:
    def __init__(
        self,
        partition_start: int,
        partition_id: int,
        picture_id: Optional[int] = None,
        tl0picidx: Optional[int] = None,
        tid: Optional[tuple[int, int]] = None,
        keyidx: Optional[int] = None,
    ) -> None:
        self.partition_start = partition_start
        self.partition_id = partition_id
        self.picture_id = picture_id
        self.tl0picidx = tl0picidx
        self.tid = tid
        self.keyidx = keyidx

    def __bytes__(self) -> bytes:
        octet = (self.partition_start << 4) | self.partition_id

        ext_octet = 0
        if self.picture_id is not None:
            ext_octet |= 1 << 7
        if self.tl0picidx is not None:
            ext_octet |= 1 << 6
        if self.tid is not None:
            ext_octet |= 1 << 5
        if self.keyidx is not None:
            ext_octet |= 1 << 4

        if ext_octet:
            data = pack("!BB", (1 << 7) | octet, ext_octet)
            if self.picture_id is not None:
                if self.picture_id < 128:
                    data += pack("!B", self.picture_id)
                else:
                    data += pack("!H", (1 << 15) | self.picture_id)
            if self.tl0picidx is not None:
                data += pack("!B", self.tl0picidx)
            if self.tid is not None or self.keyidx is not None:
                t_k = 0
                if self.tid is not None:
                    t_k |= (self.tid[0] << 6) | (self.tid[1] << 5)
                if self.keyidx is not None:
                    t_k |= self.keyidx
                data += pack("!B", t_k)
        else:
            data = pack("!B", octet)

        return data

    def __repr__(self) -> str:
        return (
            f"VpxPayloadDescriptor(S={self.partition_start}, "
            f"PID={self.partition_id}, pic_id={self.picture_id})"
        )

    @classmethod
    def parse(cls: Type[DESCRIPTOR_T], data: bytes) -> tuple[DESCRIPTOR_T, bytes]:
        if len(data) < 1:
            raise ValueError("VPX descriptor is too short")

        octet = data[0]
        extended = octet >> 7
        partition_start = (octet >> 4) & 1
        partition_id = octet & 0xF
        picture_id = None
        tl0picidx = None
        tid = None
        keyidx = None
        pos = 1

        if extended:
            if len(data) < pos + 1:
                raise ValueError("VPX descriptor has truncated extended bits")
            octet = data[pos]
            ext_I = (octet >> 7) & 1
            ext_L = (octet >> 6) & 1
            ext_T = (octet >> 5) & 1
            ext_K = (octet >> 4) & 1
            pos += 1

            if ext_I:
                if len(data) < pos + 1:
                    raise ValueError("VPX descriptor has truncated PictureID")
                if data[pos] & 0x80:
                    if len(data) < pos + 2:
                        raise ValueError("VPX descriptor has truncated long PictureID")
                    picture_id = unpack_from("!H", data, pos)[0] & 0x7FFF
                    pos += 2
                else:
                    picture_id = data[pos]
                    pos += 1

            if ext_L:
                if len(data) < pos + 1:
                    raise ValueError("VPX descriptor has truncated TL0PICIDX")
                tl0picidx = data[pos]
                pos += 1

            if ext_T or ext_K:
                if len(data) < pos + 1:
                    raise ValueError("VPX descriptor has truncated T/K")
                t_k = data[pos]
                if ext_T:
                    tid = ((t_k >> 6) & 3, (t_k >> 5) & 1)
                if ext_K:
                    keyidx = t_k & 0x1F
                pos += 1

        obj = cls(
            partition_start=partition_start,
            partition_id=partition_id,
            picture_id=picture_id,
            tl0picidx=tl0picidx,
            tid=tid,
            keyidx=keyidx,
        )
        return obj, data[pos:]

# ==============================
# 2. tokenizer
# ==============================
class TokenIdEncoder(Encoder):
    bridge = None
    #==============
    def __init__(self) -> None:
        self.picture_id = random.randint(0, (1 << 15) - 1)
        self.__target_bitrate = DEFAULT_BITRATE
        # self.bridge = get_liquid_bridge("enc")

    @classmethod
    def preload_bridge(cls):
        if cls.bridge is not None:
            print(f"directly return {cls.bridge}")
            return True
        else:
            cls.bridge = get_liquid_bridge("enc")
            print(f"vlm model preloaded, {cls.bridge}")
            return True

    @classmethod
    def cleanup_bridge(cls):
        if cls.bridge is None:
            return
        else:
            cls.bridge = None

    @classmethod
    def get_bridge(cls):
        if cls.bridge is None:
            raise RuntimeError("bridge not initialized")
        else:
            return cls.bridge

    def encode(self, frame: VideoFrame, force_keyframe: bool = False) -> tuple[list[bytes], int]:
        # step 1. get pixel values of frame
        # 2. transform to PIL image
        img = frame.to_image()
        # step 2. tokenizer
        # tokens = self.bridge.tokenize(img)
        bridge = TokenIdEncoder.get_bridge()
        tokens = bridge.tokenize(img)
        # step 3. tokens to bytes
        data = pack(f'!{len(tokens)}H', *tokens)
        # step 4. packetize following vp8 descriptor
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        payloads = self._packetize_tokenheader(data, self.picture_id, timestamp)
        # update picture_id
        self.picture_id = (self.picture_id + 1) % (1 << 15)

        return payloads, timestamp

    def pack(self, packet: Packet) -> tuple[list[bytes], int]:
        # directly Packetize（backup）
        payloads = self._packetize_tokenheader(bytes(packet), self.picture_id)
        timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
        self.picture_id = (self.picture_id + 1) % (1 << 15)
        return payloads, timestamp

    @property
    def target_bitrate(self) -> int:
        return self.__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        self.__target_bitrate = max(MIN_BITRATE, min(bitrate, MAX_BITRATE))

    @classmethod
    def _packetize(cls, buffer: bytes, picture_id: int) -> list[bytes]:
        payloads = []
        descr = VpxPayloadDescriptor(
            partition_start=1, partition_id=0, picture_id=picture_id
        )
        pos = 0
        length = len(buffer)
        while pos < length:
            descr_bytes = bytes(descr)
            size = min(length - pos, PACKET_MAX - len(descr_bytes))
            payloads.append(descr_bytes + buffer[pos:pos + size])
            descr.partition_start = 0  # 
            pos += size
        return payloads

    @classmethod
    def _packetize_tokenheader(cls, buffer: bytes, picture_id: int, timestamp: int) -> list[bytes]:
        payloads = []
        descr = VpxPayloadDescriptor(
            partition_start=1, partition_id=0, picture_id=picture_id
        )
        pos = 0
        length = len(buffer)
        token_header_length = 12 # picture_id: 4 byte, timestamp: 4 bytes, pos: 2 bytes, size: 2 bytes
        while pos < length:
            descr_bytes = bytes(descr)
            size = min(length - pos, PACKET_MAX - len(descr_bytes) - token_header_length)
            token_header = pack("!IIHH", picture_id, timestamp, pos, size)
            payloads.append(descr_bytes + token_header + buffer[pos:pos + size])
            descr.partition_start = 0  
            pos += size
        return payloads

# ==============================
# 3.1 de-tokenizer
# ==============================
class TokenIdDecoder(Decoder):
    def __init__(self, decMode) -> None:
        self.dec_mode = decMode
        if self.dec_mode == "detokenize":
            self.bridge = get_liquid_bridge("dec")
        else:
            self.bridge = get_liquid_bridge("llm")
        self.data_cache = None

    def decode(self, encoded_frame: JitterFrame) -> list[VideoFrame]:
        # step 1. bytes to tokens
        # process with lost tokens
        _, _, data = recover_token_from_data(self, encoded_frame.data)
        token_number = len(data) // 2
        tokens = unpack_from(f'!{token_number}H', data)
        if self.dec_mode == "detokenize":
            # step 2. tokenizer to pixel values
            img = self.bridge.detokenize(tokens)
            # step 3. to video frame
            data = np.array(img)
            frame = VideoFrame.from_ndarray(data, format='rgb24')
        else:
            data = tokens.numpy().to_bytes()
            frame = VideoFrame(width=512, height=512, format="rgb24")
            plane_size = frame.planes[0].buffer_size
            print("token codec plane size: ", plane_size, ", data length: ", len(data))
            if len(data) > plane_size:
                logger.warning(f"Data too large ({len(data)} > {plane_size}), truncating")
                data = data[:plane_size]
            padded = data + b'\x00' * (plane_size - len(data))
            frame.planes[0].update(padded)
        # set timestamp
        frame.pts = encoded_frame.timestamp
        frame.time_base = VIDEO_TIME_BASE
        return [frame]  


PROMPT_TEMPLATE = '''You are an expert video question-answering assistant. Analyze the video frame and select the correct option based STRICTLY on visual evidence.
Rules:
1. Verify each option against the image details (objects, actions, text, colors).
2. Output ONLY the single letter (A, B, C or D). No markdown, no extra text.
3. Output the only best option A or B or C or D at the end anyway. Do not output other options except the best option.

Question: {}
Options:
{}
{}
{}
{}
Output Format:
{{
    "reasoning": "Brief visual evidence supporting the choice.",
    "answer": "A"
}}'''

PROMPT_TEMPLATE_PROACTIVE = '''You are an advanced image question-answering AI assistant. You have been provided with images and a question related to the images. Your task is to carefully analyze the images and provide the answer to the question. You need to carefully confirm whether the images content meet the conditions of the question, and then output the correct content. Answer directly.
Question: {}
The answer is:
'''
# ==============================
# 3.2 async detokenize
# ==============================
class AsyncTokenIdDecoder(Decoder):
    # _instance = None
    data_channel = None
    bridge = None
    _dec_tasks = set()
    #==============
    # for streaming bench evaluation
    task_class = "real"
    bench_config = []
    model_name = "liquid"
    question_states = {}
    #==============

    def __init__(self) -> None:
        self.queue = get_inference_queue()
        # continuous inference in backend
        self._worker_started = False
        self.data_cache = None

    @classmethod
    def match_question_by_timestamp(cls, pts_sec):

        if not cls.bench_config:
            logger.info("no bench config")
            return None
                
        for subset in cls.bench_config:
            questions = subset.get("questions", [])
            for q in questions:
                time_stamp = q.get("time_stamp", "00:00:00")
                ground_truth_time_stamp = q.get("ground_truth_time_stamp")
                try:
                    q_parts = time_stamp.split(":")
                    q_time_sec = int(q_parts[0]) * 3600 + int(q_parts[1]) * 60 + int(q_parts[2])
                    if ground_truth_time_stamp is None:
                        if 0 <= (pts_sec - q_time_sec) <= 4:
                            return q
                    else:
                        g_parts = ground_truth_time_stamp.split(":")
                        g_time_sec = int(g_parts[0]) * 3600 + int(g_parts[1]) * 60 + int(g_parts[2])
                        if pts_sec - q_time_sec >= 0 and pts_sec - g_time_sec <= 4:
                            return q
                except:
                    continue
        return None


    @classmethod
    def preload_bridge(cls):
        if cls.bridge is not None:
            print(f"directly return {cls.bridge}")
            return True
        else:
            cls.bridge = get_liquid_bridge("llm")
            print(f"vlm model preloaded, {cls.bridge}")
            return True

    @classmethod
    def cleanup_bridge(cls):
        if cls.bridge is None:
            return
        else:
            cls.bridge = None

    @classmethod
    def get_bridge(cls):
        if cls.bridge is None:
            raise RuntimeError("bridge not initialized")
        else:
            return cls.bridge

    @classmethod
    def reset_decoder_state(cls, model_name=None):
        """Reset AsyncTokenIdDecoder static attributes for a new session"""
        cls.stop_worker()
        cls.bench_config=[]
        cls.question_states={}
        cls.result_output_path = None
        cls.current_video_path = None
        cls.current_video_name = None
        cls.data_channel = None
        # keep model name if needed, or reset based on config
        if model_name:
            cls.model_name = model_name

    @classmethod
    def stop_worker(cls):
        tasks_to_cancel = [t for t in AsyncTokenIdDecoder._dec_tasks]
        for task in tasks_to_cancel:
            if isinstance(task, asyncio.Task) and not task.done():
                task.cancel()

    def decode(self, encoded_frame):
        bridge = AsyncTokenIdDecoder.get_bridge()
        """webrtc thread calling: put data in queue, then return"""
        if not self._worker_started:
            if shared.loop and shared.loop.is_running():
                AsyncTokenIdDecoder._stop_event = asyncio.Event()
                task = shared.loop.create_task(self.__inference_worker())
                AsyncTokenIdDecoder._dec_tasks.add(task)
                task.add_done_callback(AsyncTokenIdDecoder._dec_tasks.discard)
                self._worker_started = True

        # step 1. bytes to tokens
        # process with lost tokens
        picture_id, timestamp, data = recover_token_from_data(self, encoded_frame.data)
        token_number = len(data) // 2
        tokens = unpack_from(f'!{token_number}H', data)
        # step 2. async push to queue
        try:
            self.queue.put_nowait((tokens, encoded_frame.timestamp))
        except asyncio.QueueFull:
            pass
        
        ###### detokenizer
        # img = bridge.detokenize(tokens)
        # # step 3. to video frame
        # data = np.array(img)
        # frame = VideoFrame.from_ndarray(data, format='rgb24')
        # frame.pts = encoded_frame.timestamp
        # frame.time_base = VIDEO_TIME_BASE
        # return [frame]  # return list of Frame
        return []

    async def __inference_worker(self):
        bridge = AsyncTokenIdDecoder.get_bridge()
        loop = asyncio.get_running_loop()
        while True:
            # 1. wait queue, until at least one task in the queue
            item = await self.queue.get()
            # 2. drop old frames, keep the last frame
            dropped_count = 0
            while not self.queue.empty():
                self.queue.task_done() # old task is done
                item = self.queue.get_nowait()
                dropped_count += 1
            if dropped_count > 0:
                pass
            token_tuple, pts = item
            try:
                #=============
                pts_ms = pts
                pts_sec = pts / 1000.0 / 90.0
                question = AsyncTokenIdDecoder.match_question_by_timestamp(pts_sec)
                if AsyncTokenIdDecoder.task_class == "proactive_output":
                    if question is None:
                        print(f">", end='', flush=True)
                    if question is not None:
                        qid = question.get("question_id", str(pts_ms))
                        ground_truth = question.get("ground_truth_output", "")
                        time_stamp = question.get("time_stamp", "")
                        question_text = question.get("question", "")
                        ground_truth_timestamp = question.get("ground_truth_time_stamp", time_stamp)
                        try:
                            parts = time_stamp.split(":")
                            start_time_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                            parts_gt = ground_truth_timestamp.split(":")
                            ground_truth_time_sec = int(parts_gt[0]) * 3600 + int(parts_gt[1]) * 60 + int(parts_gt[2])
                        except:
                            start_time_sec = pts_sec
                            ground_truth_time_sec = pts_sec
                        max_time = ground_truth_time_sec + 4
                        if qid not in AsyncTokenIdDecoder.question_states:
                            AsyncTokenIdDecoder.question_states[qid] = {
                                'phase': 'yes_no',
                                'history': [],
                                'answered': None
                            }

                        state = AsyncTokenIdDecoder.question_states[qid]
                        if state['phase'] == 'done':
                            print("=", end = '', flush=True)
                            continue
                        if state['phase'] == 'yes_no' and pts_sec > max_time:
                            state['phase'] = 'done'
                            print("DONE due MAX TIME", flush=True)
                            continue

                        if state['phase'] == 'yes_no':
                            query = f"{question_text} Is it the right time to output \"{ground_truth}\"? You can only answer yes or no."
                            prompt = PROMPT_TEMPLATE_PROACTIVE.format(query)
                            phase_type = 'yes_no'
                            print(f"YES_NO: {prompt}", flush=True)
                        elif state['phase'] == 'actual':
                            prompt = PROMPT_TEMPLATE_PROACTIVE.format(question_text)
                            phase_type = 'actual'
                            print(f"ACTUAL: {prompt}", flush=True)
                        else:
                            print(f"phase ERROR", flush=True)
                            continue
                        #=============
                        # 3. inference now
                        header = pack('!I', pts)
                        streamer = bridge.generate(token_tuple, user_prompt = prompt)
                        def process_stream_sb():
                            collected_text = ""
                            start_time = time.time()
                            for new_text in streamer:
                                if new_text:
                                    token_number = bridge.estimate_token_number(new_text)
                                    # result_payload = header + new_text.encode('utf-8')
                                    result_payload = header + pack('!H', token_number) + new_text.encode('utf-8')
                                    loop.call_soon_threadsafe(self.send_to_channel, result_payload)
                                    collected_text += new_text
                            end_time = time.time()
                            timecost = end_time - start_time
                            #=====================
                            history_entry = {
                                    'role': 'user',
                                    'content': prompt,
                                    'time': pts_sec,
                                    'cost': timecost
                                }
                            state['history'].append(history_entry)
                            history_entry_ans = {
                                    'role': 'assistant',
                                    'content': collected_text,
                                    'time': pts_sec,
                                    'cost': timecost
                                }
                            state['history'].append(history_entry_ans)
                            if phase_type == 'yes_no':
                                if 'yes' in collected_text.strip().lower():
                                    state['phase'] = 'actual'
                                    print(f"[Server] YES triggered at {pts_sec}s for {qid}, running actual question...", flush=True)
                                    actual_prompt = PROMPT_TEMPLATE_PROACTIVE.format(question_text)
                                    streamer_actual = bridge.generate(token_tuple, user_prompt=actual_prompt)
                                    collected_text_actual = ""
                                    start_time_actual = time.time()
                                    for new_text in streamer_actual:
                                        if new_text:
                                            token_number = bridge.estimate_token_number(new_text)
                                            # result_payload = header + new_text.encode('utf-8')
                                            result_payload = header + pack('!H', token_number) + new_text.encode('utf-8')
                                            loop.call_soon_threadsafe(self.send_to_channel, result_payload)
                                            collected_text_actual += new_text
                                    end_time_actual = time.time()
                                    timecost_actual = end_time_actual - start_time_actual
                                    state['history'].append({'role': 'user', 'content': actual_prompt, 'time': pts_sec, 'cost': timecost_actual})
                                    state['history'].append({'role': 'assistant', 'content': collected_text_actual, 'time': pts_sec, 'cost': timecost_actual})
                                    state['phase'] = 'done'
                                    state['answered'] = pts_sec
                                    question[AsyncTokenIdDecoder.model_name] = {
                                        "answered": state['answered'],
                                        "dialog_history": state['history']
                                    }
                                    print(f"[Server] Question {qid} completed.", flush=True)
                                else:
                                    print(f"[Server] No triggered at {pts_sec}s for {qid}, waiting...", flush=True)
                            elif phase_type == 'actual':
                                state['phase'] = 'done'
                                state['answered'] = pts_sec
                                question[AsyncTokenIdDecoder.model_name] = {
                                    "answered": state['answered'],
                                    "dialog_history": state['history']
                                }
                                print(f"[Server] Question {qid} completed (actual phase).", flush=True)
                        #=====================
                        await loop.run_in_executor(None, process_stream_sb)
                        loop.call_soon_threadsafe(self.send_to_channel, "[DONE]")
                    else:
                        # 3. inference now
                        header = pack('!I', pts)
                        prompt = "What is in the video? Answer directly!"
                        streamer = bridge.generate(token_tuple, user_prompt = prompt)
                        def process_stream():
                            for new_text in streamer:
                                if new_text:
                                    token_number = bridge.estimate_token_number(new_text)
                                    # result_payload = header + new_text.encode('utf-8')
                                    result_payload = header + pack('!H', token_number) + new_text.encode('utf-8')
                                    loop.call_soon_threadsafe(self.send_to_channel, result_payload)
                        await loop.run_in_executor(None, process_stream)
                        loop.call_soon_threadsafe(self.send_to_channel, "[DONE]")
                        #####
                if AsyncTokenIdDecoder.task_class == "real":
                    if question is not None:
                        time_stamp = question.get("time_stamp", "")
                        qid = question.get("question_id", str(time_stamp))
                        if qid not in AsyncTokenIdDecoder.question_states:
                            AsyncTokenIdDecoder.question_states[qid] = {
                                'phase': 'undone'
                            }
                        state = AsyncTokenIdDecoder.question_states[qid]
                        if state['phase'] == 'done':
                            print("=", end='', flush=True)
                            continue
                        ques = question["question"]
                        if "options" in question.keys():
                            options = question["options"]
                            if not options[0].startswith("A."):
                                options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
                            prompt = PROMPT_TEMPLATE.format(ques, *options)
                            prompt += "\n\nWhat is the best option?"
                            # 3. Inference now
                            header = pack('!I', pts)
                            streamer = bridge.generate(token_tuple, user_prompt=prompt)
                            def process_stream_real():
                                collected_text = ""
                                for new_text in streamer:
                                    if new_text:
                                        token_number = bridge.estimate_token_number(new_text)
                                        result_payload = header + pack('!H', token_number) + new_text.encode('utf-8')
                                        loop.call_soon_threadsafe(self.send_to_channel, result_payload)
                                        collected_text += new_text
                                question[AsyncTokenIdDecoder.model_name] = collected_text
                            await loop.run_in_executor(None, process_stream_real)
                            loop.call_soon_threadsafe(self.send_to_channel, "[DONE]")
                            state['phase'] = 'done'
                            print(">", end='', flush=True)
                    else:
                        print("+", end='',flush=True)
                        continue

            except Exception as e:
                pass
            finally:
                self.queue.task_done()
    def send_to_channel(self, text_chunk):
        if AsyncTokenIdDecoder.data_channel and AsyncTokenIdDecoder.data_channel.readyState == "open":
            if text_chunk not in ["ping", "pong"]:
                AsyncTokenIdDecoder.data_channel.send(text_chunk)
        else:
            status = AsyncTokenIdDecoder.data_channel.readyState if AsyncTokenIdDecoder.data_channel else "None"

# ==============================
# 4. Depayload 
# ==============================
def tokenId_depayload(payload: bytes) -> bytes:
    descriptor, data = VpxPayloadDescriptor.parse(payload)
    return data

# ==============================
# recover token sequence from received payload (token_header + tokens), some token chunks may be lost due to RTP packet loss
# ==============================
def recover_token_from_data(self, frame_data, total_size=2048):
    # frame_data is received from rtp, that removes vp8-like header
    # total size is fixed, depending on tokenizer configuration
    if self.data_cache is not None:
        filled_data = self.data_cache
    else:
        filled_data = bytearray(total_size)
    pointer = 0
    data_len = len(frame_data)
    pos_end_last_chunk = -1
    pos_start_this_chunk = 0
    token_header_length = 12
    while pointer + token_header_length <= data_len:
        picture_id, timestamp, pos, size = unpack_from("!IIHH", frame_data[pointer:pointer+token_header_length])
        pos_start_this_chunk = pos
        if (pos_start_this_chunk != pos_end_last_chunk and pos_end_last_chunk != -1) or (pos_start_this_chunk != 0 and pos_end_last_chunk == -1):
            print("*", end='', flush=True)
        pointer += token_header_length
        actual_size = min(size, data_len-pointer)
        chunk = frame_data[pointer:pointer+actual_size]
        end_pos = min(pos+actual_size, total_size)
        pos_end_last_chunk = end_pos
        filled_data[pos:end_pos] = chunk[:end_pos-pos]
        pointer += actual_size

    if pos_end_last_chunk != total_size:
        print("*", end='', flush=True)


    self.data_cache = filled_data
    return picture_id, timestamp, filled_data






