"""Extract ML dataset from pcap files + SQLite labels.

Usage (from the training/ directory):
    python -m dataset \
        --captures-dir ../aitestbed/results/captures \
        --db-path ../aitestbed/logs/traffic_logs.db \
        --output-dir data \
        --max-packets 50

Both --captures-dir and --db-path accept several values and shell-style globs,
so a run can span the live testbed output plus every archived cycle:

    python -m dataset \
        --captures-dir ../aitestbed/results/captures \
                       '../aitestbed/results/backup/captures_*' \
        --db-path ../aitestbed/logs/traffic_logs.db \
                  '../aitestbed/logs/*.db.bak'

Archived databases are cumulative snapshots of each other, so sessions are
deduplicated by session_id (first database wins).
"""

import argparse
import json
import logging
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import dpkt
import numpy as np

logger = logging.getLogger(__name__)

# Scenario ID → traffic pattern category
#
# Categories are based on observable traffic characteristics, not application
# semantics.  This makes the classifier useful for network-level traffic
# identification where the application purpose is unknown.
#
#   request_response  - single request, single response (non-streamed)
#   streaming         - single request, chunked/streamed response (SSE / long-poll)
#   agent_loop        - iterative LLM + tool-call rounds (growing context, mixed UL/DL)
#   realtime_ws       - persistent WebSocket, bidirectional, low-latency text/audio
#   realtime_webrtc   - WebRTC media stream (UDP, high sustained bandwidth)
#   bulk_transfer     - large asymmetric payload (upload or download of images, binaries, video)
#   parallel_burst    - fan-out of many concurrent HTTP requests + LLM synthesis
#   agent_signaling   - agent-to-agent protocol messages (A2A JSON-RPC message/send
#                       and SSE message/stream): small, near-deterministic payloads,
#                       one round trip per interaction, no LLM in the path
#   agent_control     - local control channel between an agent runtime and a local
#                       tool/gateway server (MCP JSON-RPC, OpenClaw gateway): loopback,
#                       unencrypted JSON, request/response with large local responses
#
SCENARIO_CATEGORY = {
    # Request-response (non-streamed chat)
    "chat_basic": "request_response",
    "chat_deepseek": "request_response",
    "chat_azure_openai": "request_response",
    "chat_azure_inference": "request_response",
    # Streaming (chunked response)
    "chat_streaming": "streaming",
    "chat_gemini": "streaming",
    "chat_deepseek_streaming": "streaming",
    "chat_deepseek_coder": "streaming",
    "chat_deepseek_reasoner": "streaming",
    "chat_vllm": "streaming",
    "chat_azure_openai_streaming": "streaming",
    "chat_azure_inference_streaming": "streaming",
    "chat_azure_inference_llama": "streaming",
    # Agent loop (iterative LLM + tool calls via MCP or function calling)
    "shopping_agent": "agent_loop",
    "web_search_agent": "agent_loop",
    "general_agent": "agent_loop",
    "shopping_agent_deepseek": "agent_loop",
    "web_search_agent_deepseek": "agent_loop",
    "shopping_agent_azure_openai": "agent_loop",
    "music_search": "agent_loop",
    "music_playlist": "agent_loop",
    "music_research": "agent_loop",
    "music_search_deepseek": "agent_loop",
    "trading_market_data": "agent_loop",
    "trading_options_scan": "agent_loop",
    "playwright_web_test": "agent_loop",
    "playwright_web_test_text": "agent_loop",
    "computer_control_agent": "agent_loop",
    "exa_research": "agent_loop",
    "maps_route_planning": "agent_loop",
    "weather_route_planning": "agent_loop",
    "weather_risk_assessment": "agent_loop",
    "twilio_emergency_notification": "agent_loop",
    "twilio_communication_chain": "agent_loop",
    "maps_local_discovery": "agent_loop",
    "maps_traffic_analysis": "agent_loop",
    "exa_similarity_search": "agent_loop",
    "smart_home_monitoring": "agent_loop",
    "smart_home_coordination": "agent_loop",
    # OpenClaw is a full agent runtime: its cloud LLM/tool egress is an agent
    # loop, while its gateway channel on lo:18789 is caught by the loopback
    # override below and labelled agent_control.
    "openclaw_personal_assistant": "agent_loop",
    # A2A inter-agent protocol. All three variants are one signalling class:
    # the streaming variant differs only by a handful of SSE chunks inside the
    # same sub-second exchange, which is not separable at the packet level.
    "a2a_single_task_local": "agent_signaling",
    "a2a_streaming_local": "agent_signaling",
    "a2a_multi_agent_local": "agent_signaling",
    "a2a_single_task_remote": "agent_signaling",
    "a2a_streaming_remote": "agent_signaling",
    # Realtime WebSocket
    "realtime_text": "realtime_ws",
    "realtime_interactive": "realtime_ws",
    "realtime_technical": "realtime_ws",
    "realtime_multilingual": "realtime_ws",
    "realtime_audio": "realtime_ws",
    # Realtime WebRTC
    "realtime_audio_webrtc": "realtime_webrtc",
    "realtime_text_webrtc": "realtime_webrtc",
    # Bulk transfer (large asymmetric payload — download or upload)
    "image_generation": "bulk_transfer",
    "image_generation_azure": "bulk_transfer",
    "multimodal_analysis": "bulk_transfer",
    "video_understanding_vllm": "bulk_transfer",
    # Parallel burst (fan-out of concurrent HTTP requests)
    "direct_web_search": "parallel_burst",
    "direct_web_search_google": "parallel_burst",
    "direct_web_search_burst": "parallel_burst",
    "direct_web_search_deepseek": "parallel_burst",
    "parallel_search_benchmark": "parallel_burst",
}

CATEGORY_IDS = {
    "request_response": 0,
    "streaming": 1,
    "agent_loop": 2,
    "realtime_ws": 3,
    "realtime_webrtc": 4,
    "bulk_transfer": 5,
    "parallel_burst": 6,
    "agent_signaling": 7,
    "agent_control": 8,
}

# A flow that stays on loopback is the agent's local control plane, not the
# data plane it is driving, and its signature differs accordingly (no TLS,
# small JSON-RPC requests, large local responses). Re-label those flows so the
# control channel is not conflated with the cloud egress of the same scenario.
LOOPBACK_CATEGORY_OVERRIDE = {
    "agent_loop": "agent_control",
}

# Scenarios whose traffic exists only on loopback, because the agents under
# test are hosted locally. A WAN flow overlapping such a session in time is
# unrelated host traffic that the capture happened to pick up, so it is dropped
# rather than mislabelled as agent traffic.
LOOPBACK_ONLY_SCENARIOS = {
    "a2a_single_task_local",
    "a2a_streaming_local",
    "a2a_multi_agent_local",
}

PROFILE_IDS = {
    # Current profile set (configs/profiles.yaml).
    "no_emulation": 0, "6g_itu_hrllc": 1, "5g_urban": 2, "wifi_good": 3,
    "cell_edge": 4, "satellite_leo": 5, "satellite_geo": 6, "congested": 7,
    "5qi_7": 8, "5qi_80": 9,
    # Legacy names retained so archived cycles keep stable IDs.
    "ideal_6g": 10, "edge_rural": 11, "satellite": 12, "5qi_1": 13, "5qi_3": 14,
}

# Profiles treated as impairment-free when training the generator: either no
# emulation at all, or a near-ideal shaped profile (1 to 2 ms, no loss).
CLEAN_PROFILES = ("no_emulation", "ideal_6g", "6g_itu_hrllc")


@dataclass
class PacketRecord:
    timestamp: float
    size: int
    payload_len: int
    direction: int  # 0=UL, 1=DL
    tcp_flags: int
    tcp_window: int
    protocol: str = "tcp"


@dataclass
class FlowSample:
    flow_id: str
    packets: list[PacketRecord] = field(default_factory=list)
    scenario_id: str = ""
    category: int = -1
    network_profile: str = ""
    profile_id: int = -1
    session_id: str = ""
    loopback: bool = False
    protocol: str = "tcp"
    source_pcap: str = ""
    capture_id: str = ""
    attribution: str = "session_window"
    attribution_score: float = 0.0


# Well-known server ports on the WAN side. Loopback flows are kept regardless
# of port, since a loopback capture only ever contains testbed traffic and the
# local servers bind wherever they are told (MCP 8765, A2A 9001-9003, OpenClaw
# gateway 18789, vLLM 8000, and ephemeral ports for dynamically bound servers).
WAN_SERVER_PORTS = (443, 80, 8000, 8080, 8443)
UDP_SERVICE_PORTS = (443, 3478, 3479, 5349, 5350, 19302)
IGNORED_UDP_PORTS = (53, 67, 68, 123, 5353)


def _ip_to_str(addr: bytes) -> str:
    if len(addr) == 4:
        return ".".join(str(b) for b in addr)
    return ":".join(f"{addr[i]:02x}{addr[i+1]:02x}" for i in range(0, len(addr), 2))


def _is_loopback(ip_str: str) -> bool:
    return ip_str.startswith("127.") or ip_str in ("::1", "0000:0000:0000:0000:0000:0000:0000:0001")


def _parse_window_scale(opts_bytes: bytes) -> int | None:
    """Extract window scale from TCP options."""
    i = 0
    while i < len(opts_bytes):
        kind = opts_bytes[i]
        if kind == 0:
            break
        if kind == 1:
            i += 1
            continue
        if i + 1 >= len(opts_bytes):
            break
        length = opts_bytes[i + 1]
        if length < 2:
            break
        if kind == 3 and length == 3:  # Window Scale
            return opts_bytes[i + 2]
        i += length
    return None


def _load_sessions(db_paths) -> list[dict]:
    """Load experiment sessions from one or more SQLite databases.

    Archived databases are cumulative snapshots that re-contain earlier runs,
    so sessions are deduplicated by session_id: without this, one session is
    matched by several near-identical entries and the same flow is counted
    repeatedly."""
    if isinstance(db_paths, (str, Path)):
        db_paths = [db_paths]

    sessions = []
    seen_sessions: set[str] = set()
    unmapped: Counter = Counter()

    for db_path in db_paths:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT scenario_id, network_profile, session_id,
                       min(t_request_start) as t_start,
                       max(t_request_start + latency_sec) as t_end,
                       min(provider) as provider,
                       min(model) as model,
                       count(*) as n_turns
                FROM traffic_logs
                WHERE session_id NOT LIKE 'pcap_%'
                GROUP BY session_id
                HAVING t_end > t_start
                ORDER BY t_start
            """)
            rows = cur.fetchall()
        except sqlite3.Error as exc:
            logger.warning("Skipping %s: %s", db_path, exc)
            continue

        n_new = 0
        for row in rows:
            if row["session_id"] in seen_sessions:
                continue
            cat_name = SCENARIO_CATEGORY.get(row["scenario_id"])
            if cat_name is None:
                unmapped[row["scenario_id"]] += 1
                continue
            profile = row["network_profile"]
            if profile not in PROFILE_IDS:
                PROFILE_IDS[profile] = len(PROFILE_IDS)
            seen_sessions.add(row["session_id"])
            n_new += 1
            sessions.append({
                "scenario_id": row["scenario_id"],
                "category": CATEGORY_IDS[cat_name],
                "category_name": cat_name,
                "network_profile": profile,
                "profile_id": PROFILE_IDS[profile],
                "session_id": row["session_id"],
                "t_start": row["t_start"],
                "t_end": row["t_end"],
                "provider": row["provider"] or "",
                "model": row["model"] or "",
                "n_turns": row["n_turns"],
            })
        conn.close()
        logger.info("%s: %d sessions (%d new)", db_path, len(rows), n_new)

    if unmapped:
        logger.warning("Skipped unmapped scenario_ids: %s", dict(unmapped))
    logger.info("Loaded %d unique sessions from %d database(s)",
                len(sessions), len(list(db_paths)))
    return sessions


def _load_capture_metadata(db_paths: Iterable[Path]) -> dict[str, dict]:
    """Load the explicit pcap records written by the orchestrator.

    The database stores the capture filename, interface, run_id, scenario and
    profile.  Basenames are used because archives move pcaps into cycle-specific
    directories after the record was written.  Older captures without these
    rows still use the conservative time-window fallback in ``build_dataset``.
    """
    captures: dict[str, dict] = {}
    for db_path in db_paths:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT scenario_id, network_profile, t_request_start,
                       latency_sec, metadata
                FROM traffic_logs
                WHERE session_id LIKE 'pcap_%' AND metadata IS NOT NULL
                ORDER BY t_request_start
            """).fetchall()
        except sqlite3.Error as exc:
            logger.warning("Cannot read capture metadata from %s: %s", db_path, exc)
            continue
        for row in rows:
            try:
                meta = json.loads(row["metadata"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            pcap_file = meta.get("pcap_file")
            if not pcap_file:
                continue
            name = Path(pcap_file).name
            if name in captures:
                continue
            t_start = float(row["t_request_start"] or 0.0)
            captures[name] = {
                "scenario_id": row["scenario_id"] or "",
                "network_profile": row["network_profile"] or "",
                "run_id": meta.get("run_id") or _capture_id_from_name(name),
                "interface": meta.get("interface") or "",
                "t_start": t_start,
                "t_end": t_start + float(row["latency_sec"] or 0.0),
                "source": "database",
            }
        conn.close()
    logger.info("Loaded explicit metadata for %d capture files", len(captures))
    return captures


def _capture_id_from_name(name: str) -> str:
    """Return one group id for the primary/loopback pair of a capture."""
    stem = Path(name).stem
    if stem.startswith("capture_lo_"):
        stem = "capture_" + stem[len("capture_lo_"):]
    return stem


def _extract_flows_from_pcap(pcap_path: str) -> dict[str, list[dict]]:
    """Parse IPv4/IPv6 TCP and UDP into direction-normalized flows.

    TCP server identity comes from SYN/SYN-ACK whenever the handshake is in the
    capture. UDP uses a known service port, then the first datagram destination
    (the initiating peer). Unlike the old extractor, UDP media is retained, so
    the WebRTC category is based on its transport rather than coincident TCP
    signaling.
    """
    raw: dict[tuple, list[dict]] = defaultdict(list)
    syn_targets: dict[tuple, Counter] = defaultdict(Counter)

    with open(pcap_path, "rb") as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            try:
                pcap = dpkt.pcapng.Reader(f)
            except Exception:
                logger.warning("Cannot parse %s", pcap_path)
                return {}

        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue
                transport = ip.data
                if isinstance(transport, dpkt.tcp.TCP):
                    protocol = "tcp"
                    flags = transport.flags
                    payload_len = len(transport.data)
                    win = transport.win
                    opts = transport.opts if (flags & dpkt.tcp.TH_SYN) else b""
                elif isinstance(transport, dpkt.udp.UDP):
                    protocol = "udp"
                    if transport.sport in IGNORED_UDP_PORTS or transport.dport in IGNORED_UDP_PORTS:
                        continue
                    flags, win, opts = 0, 0, b""
                    payload_len = len(transport.data)
                else:
                    continue

                src = (_ip_to_str(ip.src), transport.sport)
                dst = (_ip_to_str(ip.dst), transport.dport)
                endpoints = (src, dst) if src <= dst else (dst, src)
                key = (protocol, endpoints[0], endpoints[1])

                if protocol == "tcp":
                    if (flags & dpkt.tcp.TH_SYN) and not (flags & dpkt.tcp.TH_ACK):
                        syn_targets[key][dst] += 1
                    elif (flags & dpkt.tcp.TH_SYN) and (flags & dpkt.tcp.TH_ACK):
                        syn_targets[key][src] += 1

                raw[key].append({
                    "ts": float(ts),
                    "size": len(ip),
                    "payload_len": payload_len,
                    "src": src,
                    "dst": dst,
                    "flags": flags,
                    "win": win,
                    "opts": opts,
                })
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, TypeError):
                continue

    flows: dict[str, list[dict]] = {}
    for key, pkts in raw.items():
        protocol, endpoint_a, endpoint_b = key
        endpoints = [endpoint_a, endpoint_b]
        server = None
        if protocol == "tcp" and syn_targets.get(key):
            server = syn_targets[key].most_common(1)[0][0]
        else:
            service_ports = WAN_SERVER_PORTS if protocol == "tcp" else UDP_SERVICE_PORTS
            known = [e for e in endpoints if e[1] in service_ports]
            if len(known) == 1:
                server = known[0]
            elif protocol == "udp":
                server = pkts[0]["dst"]
            else:
                server = min(endpoints, key=lambda e: e[1])
        client = endpoints[0] if endpoints[1] == server else endpoints[1]

        loopback = _is_loopback(server[0]) and _is_loopback(client[0])
        if protocol == "tcp" and not loopback and server[1] not in WAN_SERVER_PORTS:
            continue

        scales = {}
        if protocol == "tcp":
            for packet in pkts:
                if packet["opts"]:
                    scale = _parse_window_scale(packet["opts"])
                    if scale is not None:
                        side = "server" if packet["src"] == server else "client"
                        scales[side] = scale
        client_scale = scales.get("client", 0)
        server_scale = scales.get("server", 0)

        flow_key = f"{protocol}:{client[0]}:{client[1]}-{server[0]}:{server[1]}"
        out = []
        for packet in pkts:
            direction = 0 if packet["dst"] == server else 1
            scale = client_scale if packet["src"] == client else server_scale
            out.append({
                "ts": packet["ts"],
                "size": packet["size"],
                "payload_len": packet["payload_len"],
                "direction": direction,
                "flags": packet["flags"],
                "win": packet["win"],
                "eff_win": packet["win"] << scale,
                "loopback": loopback,
                "protocol": protocol,
            })
        flows[flow_key] = out

    logger.debug("Extracted %d transport flows from %s", len(flows), pcap_path)
    return flows


def _match_flows_to_sessions(
    flows: dict[str, list[dict]],
    sessions: list[dict],
    min_packets: int = 5,
    keep_packets: int | None = None,
    source_pcap: str = "",
    capture_id: str = "",
) -> list[FlowSample]:
    """Segment transport flows by exact experiment session windows.

    A persistent HTTP/2/TLS connection can serve many scenario runs. Assigning
    the whole connection to whichever session produced the best overlap leaks
    packets and labels across runs. Here each (capture, transport flow, session)
    is a separate sample containing only packets timestamped inside that
    session. Capture metadata has already constrained ``sessions`` to the
    scenario/profile that produced this pcap.
    """
    from bisect import bisect_left, bisect_right

    sorted_sessions = sorted(sessions, key=lambda s: s["t_start"])
    samples = []

    for flow_key, pkts in flows.items():
        if len(pkts) < min_packets:
            continue

        pkts.sort(key=lambda p: p["ts"])
        timestamps = [p["ts"] for p in pkts]
        for sess in sorted_sessions:
            if sess["t_start"] > timestamps[-1]:
                break
            if sess["t_end"] < timestamps[0]:
                continue
            start = bisect_left(timestamps, sess["t_start"])
            end = bisect_right(timestamps, sess["t_end"])
            segment = pkts[start:end]
            if len(segment) < min_packets:
                continue
            # ACK-only segments carry no application behavior and previously
            # inflated the dataset through idle persistent connections.
            if not any(p["payload_len"] > 0 for p in segment):
                continue

            loopback = bool(segment[0].get("loopback"))
            if not loopback and sess["scenario_id"] in LOOPBACK_ONLY_SCENARIOS:
                continue
            protocol = segment[0].get("protocol", "tcp")
            cat_name = sess["category_name"]
            if loopback:
                cat_name = LOOPBACK_CATEGORY_OVERRIDE.get(cat_name, cat_name)
            # WebRTC media must be represented by UDP. TCP records in those
            # sessions are signaling/control traffic and are not mislabeled as
            # a UDP media class.
            if cat_name == "realtime_webrtc" and protocol != "udp":
                continue

            head = segment[:keep_packets] if keep_packets else segment
            records = [
                PacketRecord(
                    timestamp=p["ts"],
                    size=p["size"],
                    payload_len=p["payload_len"],
                    direction=p["direction"],
                    tcp_flags=p["flags"],
                    tcp_window=p.get("eff_win", p["win"]),
                    protocol=protocol,
                )
                for p in head
            ]
            sample_key = f"{capture_id}|{flow_key}|{sess['session_id']}"
            samples.append(FlowSample(
                flow_id=sample_key,
                packets=records,
                scenario_id=sess["scenario_id"],
                category=CATEGORY_IDS[cat_name],
                network_profile=sess["network_profile"],
                profile_id=sess["profile_id"],
                session_id=sess["session_id"],
                loopback=loopback,
                protocol=protocol,
                source_pcap=source_pcap,
                capture_id=capture_id,
                attribution_score=len(segment) / len(pkts),
            ))

    logger.debug("Matched %d flows to sessions", len(samples))
    return samples


def compute_aggregate_features(packets: list[PacketRecord], k: int) -> np.ndarray:
    """Compute 26 aggregate features from first k packets of a flow."""
    pkts = packets[:k]
    n = len(pkts)
    if n == 0:
        return np.zeros(26, dtype=np.float32)

    sizes = np.array([p.size for p in pkts], dtype=np.float32)
    payloads = np.array([p.payload_len for p in pkts], dtype=np.float32)
    dirs = np.array([p.direction for p in pkts])
    windows = np.array([p.tcp_window for p in pkts], dtype=np.float32)
    times = np.array([p.timestamp for p in pkts])

    ul_mask = dirs == 0
    dl_mask = dirs == 1

    ul_sizes = sizes[ul_mask] if ul_mask.any() else np.array([0.0])
    dl_sizes = sizes[dl_mask] if dl_mask.any() else np.array([0.0])

    # IAT
    if n > 1:
        iat = np.diff(times)
        iat = iat[iat >= 0]  # filter negative (shouldn't happen)
        if len(iat) == 0:
            iat = np.array([0.0])
    else:
        iat = np.array([0.0])

    # Burst count (IAT gaps > 1s)
    burst_count = np.sum(iat > 1.0) if len(iat) > 0 else 0

    # PSH flag fraction
    psh_count = sum(1 for p in pkts if p.tcp_flags & dpkt.tcp.TH_PUSH)

    # --- New discriminative features (14-19) ---

    # 14: time_to_first_dl — time from first packet to first DL packet
    dl_indices = np.where(dl_mask)[0]
    if len(dl_indices) > 0:
        time_to_first_dl = times[dl_indices[0]] - times[0]
    else:
        time_to_first_dl = 0.0

    # 15: dl_count_1s — number of DL packets within 1 s of first packet
    dl_count_1s = int(np.sum(dl_mask & (times - times[0] <= 1.0)))

    # 16: max_consec_same_dir — longest run of consecutive same-direction packets
    max_run = 1
    cur_run = 1
    for idx in range(1, n):
        if dirs[idx] == dirs[idx - 1]:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 1
    max_consec_same_dir = float(max_run)

    # 17: pkt_size_entropy — Shannon entropy over 10 equal-width bins of packet sizes
    max_size = sizes.max()
    if max_size > 0 and len(np.unique(sizes)) > 1:
        counts, _ = np.histogram(sizes, bins=10, range=(0, max_size))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        pkt_size_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        pkt_size_entropy = 0.0

    # 18: payload_ratio — fraction of packets with payload_len > 0
    payload_ratio = float(np.sum(payloads > 0)) / max(n, 1)

    # 19: window_growth_rate — slope of tcp_window vs packet index
    if n > 1:
        window_growth_rate = float(np.polyfit(np.arange(n), windows, 1)[0])
    else:
        window_growth_rate = 0.0

    # --- Directional switching features (20-25) ---
    # These capture the request-response interaction pattern that distinguishes
    # agent_loop (frequent UL↔DL alternation) from streaming (long DL runs).

    # 20: dir_switch_count — number of direction changes (UL→DL or DL→UL)
    dir_switches = int(np.sum(np.diff(dirs) != 0)) if n > 1 else 0

    # 21: dir_switch_rate — switches per packet (normalised)
    dir_switch_rate = dir_switches / max(n - 1, 1)

    # 22: mean_run_length — average length of consecutive same-direction runs
    run_lengths = []
    cur = 1
    for idx in range(1, n):
        if dirs[idx] == dirs[idx - 1]:
            cur += 1
        else:
            run_lengths.append(cur)
            cur = 1
    run_lengths.append(cur)
    mean_run_length = float(np.mean(run_lengths))

    # 23: dl_burst_size_mean — average bytes in each DL run
    dl_run_bytes = []
    cur_bytes = 0.0
    in_dl = False
    for idx in range(n):
        if dirs[idx] == 1:
            cur_bytes += sizes[idx]
            in_dl = True
        else:
            if in_dl and cur_bytes > 0:
                dl_run_bytes.append(cur_bytes)
            cur_bytes = 0.0
            in_dl = False
    if in_dl and cur_bytes > 0:
        dl_run_bytes.append(cur_bytes)
    dl_burst_size_mean = float(np.mean(dl_run_bytes)) if dl_run_bytes else 0.0

    # 24: roundtrip_count — number of UL→DL transitions (request→response cycles)
    roundtrip_count = 0
    for idx in range(1, n):
        if dirs[idx - 1] == 0 and dirs[idx] == 1:
            roundtrip_count += 1

    # 25: iat_cv — coefficient of variation of inter-arrival times
    iat_cv = float(iat.std() / max(iat.mean(), 1e-9)) if len(iat) > 1 else 0.0

    features = np.array([
        ul_sizes.mean(),                                  # 0: mean_pkt_size_ul
        dl_sizes.mean(),                                  # 1: mean_pkt_size_dl
        ul_sizes.std() if len(ul_sizes) > 1 else 0.0,    # 2: std_pkt_size_ul
        dl_sizes.std() if len(dl_sizes) > 1 else 0.0,    # 3: std_pkt_size_dl
        sizes.max(),                                      # 4: max_pkt_size
        iat.mean(),                                       # 5: mean_iat
        iat.std() if len(iat) > 1 else 0.0,              # 6: std_iat
        iat.min() if len(iat) > 0 else 0.0,              # 7: min_iat
        (ul_sizes.sum() / max(dl_sizes.sum(), 1e-9)),     # 8: ul_dl_ratio
        ul_mask.mean(),                                   # 9: ul_pkt_fraction
        float(burst_count),                               # 10: burst_count
        psh_count / max(n, 1),                            # 11: psh_fraction
        sizes.sum(),                                      # 12: total_bytes
        windows.mean(),                                   # 13: mean_window
        time_to_first_dl,                                 # 14: time_to_first_dl
        float(dl_count_1s),                               # 15: dl_count_1s
        max_consec_same_dir,                              # 16: max_consec_same_dir
        pkt_size_entropy,                                 # 17: pkt_size_entropy
        payload_ratio,                                    # 18: payload_ratio
        window_growth_rate,                               # 19: window_growth_rate
        float(dir_switches),                              # 20: dir_switch_count
        dir_switch_rate,                                  # 21: dir_switch_rate
        mean_run_length,                                  # 22: mean_run_length
        dl_burst_size_mean,                               # 23: dl_burst_size_mean
        float(roundtrip_count),                           # 24: roundtrip_count
        iat_cv,                                           # 25: iat_cv
    ], dtype=np.float32)

    return features


FEATURE_NAMES = [
    "mean_pkt_size_ul", "mean_pkt_size_dl", "std_pkt_size_ul", "std_pkt_size_dl",
    "max_pkt_size", "mean_iat", "std_iat", "min_iat", "ul_dl_ratio",
    "ul_pkt_fraction", "burst_count", "psh_fraction", "total_bytes", "mean_window",
    "time_to_first_dl", "dl_count_1s", "max_consec_same_dir",
    "pkt_size_entropy", "payload_ratio", "window_growth_rate",
    "dir_switch_count", "dir_switch_rate", "mean_run_length",
    "dl_burst_size_mean", "roundtrip_count", "iat_cv",
]


def _resolve_paths(patterns, suffixes=None) -> list[Path]:
    """Expand a list of paths/globs into existing files or directories."""
    import glob as _glob

    if isinstance(patterns, (str, Path)):
        patterns = [patterns]
    out: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = _glob.glob(str(pattern)) or ([str(pattern)] if Path(pattern).exists() else [])
        for m in sorted(matches):
            p = Path(m)
            if suffixes and p.is_file() and not any(str(p).endswith(s) for s in suffixes):
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def _flow_time_bounds(flows: dict[str, list[dict]]) -> tuple[float, float]:
    starts = [packets[0]["ts"] for packets in flows.values() if packets]
    ends = [packets[-1]["ts"] for packets in flows.values() if packets]
    return (min(starts), max(ends)) if starts else (0.0, 0.0)


def _sessions_for_capture(
    sessions: list[dict],
    pcap_name: str,
    flows: dict[str, list[dict]],
    capture_metadata: dict[str, dict],
) -> tuple[list[dict], dict]:
    """Constrain session attribution with explicit capture provenance.

    Newer orchestrator runs write one metadata row per pcap. For older files,
    the fallback chooses the dominant scenario/profile among sessions inside
    the pcap's actual packet interval. This is deliberately capture-level: it
    prevents unrelated background HTTPS traffic from borrowing a coincident
    session label.
    """
    start, end = _flow_time_bounds(flows)
    meta = dict(capture_metadata.get(pcap_name) or {})
    overlapping = [
        session for session in sessions
        if session["t_start"] <= end and session["t_end"] >= start
    ]
    scenario = meta.get("scenario_id")
    profile = meta.get("network_profile")
    if scenario and scenario != "test_matrix" and profile and profile != "multiple":
        selected = [
            session for session in overlapping
            if session["scenario_id"] == scenario
            and session["network_profile"] == profile
        ]
        meta["attribution"] = "capture_metadata"
    elif scenario == "test_matrix" or profile == "multiple":
        selected = overlapping
        meta["attribution"] = "matrix_capture_windows"
    else:
        votes = Counter(
            (session["scenario_id"], session["network_profile"])
            for session in overlapping
        )
        if votes:
            (scenario, profile), _ = votes.most_common(1)[0]
            selected = [
                session for session in overlapping
                if (session["scenario_id"], session["network_profile"])
                == (scenario, profile)
            ]
            meta.update({
                "scenario_id": scenario,
                "network_profile": profile,
                "attribution": "dominant_capture_window",
            })
        else:
            selected = []
            meta["attribution"] = "unmatched"
    meta.setdefault("run_id", _capture_id_from_name(pcap_name))
    return selected, meta


def _stratified_group_split(
    targets: np.ndarray,
    group_ids: list[str],
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    trials: int = 2000,
) -> dict[str, list[int]]:
    """Create deterministic, class-balanced splits with disjoint groups.

    Random assignment is searched at the *group* level and scored against the
    requested per-class proportions. Missing a feasible class in a split is
    heavily penalized. A capture (including its primary/loopback pair) can
    therefore never occur in more than one split.
    """
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 3:
        raise ValueError("At least three capture groups are required for train/val/test")
    group_to_index = {group: i for i, group in enumerate(unique_groups)}
    classes = sorted(int(value) for value in np.unique(targets))
    class_to_col = {value: i for i, value in enumerate(classes)}
    counts = np.zeros((len(unique_groups), len(classes)), dtype=np.int64)
    members: list[list[int]] = [[] for _ in unique_groups]
    for sample_index, (target, group) in enumerate(zip(targets, group_ids)):
        group_index = group_to_index[group]
        counts[group_index, class_to_col[int(target)]] += 1
        members[group_index].append(sample_index)

    proportions = np.array(
        [train_fraction, val_fraction, 1.0 - train_fraction - val_fraction],
        dtype=np.float64,
    )
    totals = counts.sum(axis=0)
    desired = proportions[:, None] * totals[None, :]
    class_group_counts = (counts > 0).sum(axis=0)
    rng = np.random.default_rng(seed)
    best_assignment = None
    best_score = float("inf")

    for _ in range(trials):
        assignment = rng.choice(3, size=len(unique_groups), p=proportions)
        # Every split must have at least one group.
        if len(set(int(x) for x in assignment)) < 3:
            continue
        actual = np.stack([counts[assignment == split].sum(axis=0) for split in range(3)])
        relative_error = ((actual - desired) / np.maximum(totals, 1)) ** 2
        score = float(relative_error.sum())
        for col, n_groups in enumerate(class_group_counts):
            if n_groups >= 3:
                score += 1000.0 * float(np.sum(actual[:, col] == 0))
            elif actual[0, col] == 0:
                score += 1000.0
        split_sizes = actual.sum(axis=1)
        score += float(np.sum(
            ((split_sizes / max(split_sizes.sum(), 1)) - proportions) ** 2
        ))
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise ValueError("Could not construct grouped dataset splits")

    split_names = ("train", "val", "test")
    result: dict[str, list[int]] = {}
    for split, name in enumerate(split_names):
        result[name] = sorted(
            index
            for group_index in np.where(best_assignment == split)[0]
            for index in members[group_index]
        )
    return result


def build_dataset(
    captures_dir,
    db_path,
    output_dir: str,
    max_packets: int = 50,
    min_packets: int = 5,
    classify_by: str = "auto",
):
    """Build complete ML dataset from pcap captures and SQLite labels.

    captures_dir and db_path each accept a path, a glob, or a list of either,
    so a run can span the live capture directory plus archived cycles.

    Args:
        classify_by: "category" (scenario type), "profile" (network profile),
                     or "auto" (pick whichever has more classes in the data).
    """
    capture_dirs = _resolve_paths(captures_dir)
    db_files = _resolve_paths(db_path, suffixes=(".db", ".bak"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not db_files:
        raise ValueError(f"No databases matched {db_path}")
    logger.info("Databases: %d", len(db_files))

    sessions = _load_sessions(db_files)
    if not sessions:
        raise ValueError("No sessions found in database")
    capture_metadata = _load_capture_metadata(db_files)

    all_samples: list[FlowSample] = []

    pcap_files: list[Path] = []
    seen_pcaps: set[tuple[str, int]] = set()
    duplicate_pcaps = 0
    for d in capture_dirs:
        found = sorted(d.glob("*.pcap")) + sorted(d.glob("*.pcapng")) if d.is_dir() else [d]
        for p in found:
            try:
                key = (p.name, p.stat().st_size)
            except OSError:
                continue
            if key in seen_pcaps:
                duplicate_pcaps += 1
                continue
            seen_pcaps.add(key)
            pcap_files.append(p)
    logger.info("Found %d pcap files across %d directories", len(pcap_files), len(capture_dirs))
    if duplicate_pcaps:
        logger.info("Skipped %d duplicate archived pcap copies", duplicate_pcaps)

    attribution_counts: Counter = Counter()
    for i, pcap_file in enumerate(pcap_files, 1):
        if i % 50 == 0 or i == len(pcap_files):
            logger.info("Processing pcap %d/%d (%d flows matched so far)",
                        i, len(pcap_files), len(all_samples))
        try:
            flows = _extract_flows_from_pcap(str(pcap_file))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", pcap_file, exc)
            continue
        capture_sessions, meta = _sessions_for_capture(
            sessions, pcap_file.name, flows, capture_metadata
        )
        attribution_counts[meta["attribution"]] += 1
        if not capture_sessions:
            continue
        capture_id = str(meta.get("run_id") or _capture_id_from_name(pcap_file.name))
        samples = _match_flows_to_sessions(
            flows,
            capture_sessions,
            min_packets=min_packets,
            keep_packets=max_packets,
            source_pcap=str(pcap_file),
            capture_id=capture_id,
        )
        for sample in samples:
            sample.attribution = meta["attribution"]
        all_samples.extend(samples)

    if not all_samples:
        raise ValueError("No flows matched to sessions")

    logger.info("Total labeled flows: %d", len(all_samples))

    # Decide classification target
    n_unique_cats = len(set(s.category for s in all_samples))
    n_unique_profs = len(set(s.profile_id for s in all_samples))

    if classify_by == "auto":
        if n_unique_cats >= 3:
            classify_by = "category"
        elif n_unique_profs >= 3:
            classify_by = "profile"
            logger.info("Only %d scenario categories found; using network profile as target (%d classes)",
                        n_unique_cats, n_unique_profs)
        else:
            classify_by = "category"
    logger.info("Classification target: %s", classify_by)

    # Build feature matrices for multiple k values
    k_values = [5, 10, 15, 20, 30, max_packets]
    for k in k_values:
        features = np.stack([
            compute_aggregate_features(s.packets, k) for s in all_samples
        ])
        np.savez_compressed(
            output_path / f"features_k{k}.npz",
            X=features,
        )
        logger.info("Saved features_k%d.npz: shape=%s", k, features.shape)

    # Build sequence data for generator
    categories = np.array([s.category for s in all_samples], dtype=np.int64)
    profiles = np.array([s.profile_id for s in all_samples], dtype=np.int64)
    lengths = np.array([min(len(s.packets), max_packets) for s in all_samples], dtype=np.int64)

    # Set classification targets based on mode
    if classify_by == "profile":
        targets = profiles.copy()
        target_ids = PROFILE_IDS
        target_name = "profile"
    else:
        targets = categories.copy()
        target_ids = CATEGORY_IDS
        target_name = "category"

    # Pad sequences to max_packets x 3 (size, iat, direction)
    sequences = np.zeros((len(all_samples), max_packets, 3), dtype=np.float32)
    for i, sample in enumerate(all_samples):
        pkts = sample.packets[:max_packets]
        for j, p in enumerate(pkts):
            sequences[i, j, 0] = p.size
            if j > 0:
                sequences[i, j, 1] = max(0, p.timestamp - pkts[j - 1].timestamp)
            sequences[i, j, 2] = p.direction

    np.savez_compressed(
        output_path / "sequences.npz",
        sequences=sequences,
        categories=categories,
        profiles=profiles,
        targets=targets,
        lengths=lengths,
    )
    logger.info("Saved sequences.npz: shape=%s", sequences.shape)

    # Save labels and splits
    id_to_target = {v: k for k, v in target_ids.items()}
    target_counts = Counter(int(t) for t in targets)

    labels = {
        "classify_by": classify_by,
        "category_ids": CATEGORY_IDS,
        "profile_ids": PROFILE_IDS,
        "target_ids": target_ids,
        "feature_names": FEATURE_NAMES,
        "n_samples": len(all_samples),
        "flow_ids": [s.flow_id for s in all_samples],
        "scenario_ids": [s.scenario_id for s in all_samples],
        "session_ids": [s.session_id for s in all_samples],
        "capture_ids": [s.capture_id for s in all_samples],
        "source_pcaps": [s.source_pcap for s in all_samples],
        "protocols": [s.protocol for s in all_samples],
        "attribution": [s.attribution for s in all_samples],
        "attribution_scores": [s.attribution_score for s in all_samples],
        "loopback": [int(s.loopback) for s in all_samples],
        "clean_profiles": list(CLEAN_PROFILES),
        "split_strategy": {
            "unit": "capture_id",
            "method": "seeded_stratified_group_search",
            "seed": 42,
            "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        },
        "sources": {
            "capture_dirs": [str(d) for d in capture_dirs],
            "databases": [str(d) for d in db_files],
            "n_pcap_files": len(pcap_files),
            "n_duplicate_pcaps_skipped": duplicate_pcaps,
            "n_sessions": len(sessions),
            "attribution_counts": dict(attribution_counts),
        },
        "class_distribution": {id_to_target.get(k, str(k)): v for k, v in sorted(target_counts.items())},
    }
    logger.info("Class distribution (%s): %s", classify_by, labels["class_distribution"])

    with open(output_path / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    splits = _stratified_group_split(
        targets, [sample.capture_id for sample in all_samples], seed=42
    )
    with open(output_path / "splits.json", "w") as f:
        json.dump(splits, f)

    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    split_groups = {
        name: set(all_samples[index].capture_id for index in indices)
        for name, indices in splits.items()
    }
    assert not (split_groups["train"] & split_groups["val"])
    assert not (split_groups["train"] & split_groups["test"])
    assert not (split_groups["val"] & split_groups["test"])
    logger.info(
        "Grouped splits: train=%d (%d captures), val=%d (%d), test=%d (%d)",
        len(train_idx), len(split_groups["train"]),
        len(val_idx), len(split_groups["val"]),
        len(test_idx), len(split_groups["test"]),
    )
    print(f"\nDataset built: {len(all_samples)} flows")
    print(f"  Classes: {labels['class_distribution']}")
    print(f"  Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"  Output: {output_path}")


def print_stats(data_dir: str):
    """Print dataset statistics."""
    data_path = Path(data_dir)
    with open(data_path / "labels.json") as f:
        labels = json.load(f)

    print(f"Samples: {labels['n_samples']}")
    print(f"\nClass distribution:")
    for name, count in sorted(labels["class_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {count:5d}")

    with open(data_path / "splits.json") as f:
        splits = json.load(f)
    print(f"\nSplits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Show feature stats for k=20
    feat_path = data_path / "features_k20.npz"
    if feat_path.exists():
        data = np.load(feat_path)
        X = data["X"]
        print(f"\nFeatures (k=20): shape={X.shape}")
        for i, name in enumerate(labels["feature_names"]):
            print(f"  {name:20s}  mean={X[:, i].mean():10.2f}  std={X[:, i].std():10.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build ML dataset from pcap + SQLite")
    parser.add_argument("--captures-dir", nargs="+", default=["results/captures"],
                        help="One or more capture directories or globs")
    parser.add_argument("--db-path", nargs="+", default=["logs/traffic_logs.db"],
                        help="One or more SQLite databases or globs")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-packets", type=int, default=50)
    parser.add_argument("--min-packets", type=int, default=5)
    parser.add_argument("--classify-by", choices=["auto", "category", "profile"], default="auto",
                        help="Classification target: scenario category, network profile, or auto-detect")
    parser.add_argument("--stats", action="store_true", help="Print dataset stats")
    parser.add_argument("--data-dir", default="data", help="For --stats")
    args = parser.parse_args()

    if args.stats:
        print_stats(args.data_dir)
    else:
        build_dataset(
            args.captures_dir, args.db_path, args.output_dir,
            args.max_packets, args.min_packets, args.classify_by,
        )
