import numpy as np

from dataset import (
    CATEGORY_IDS,
    _match_flows_to_sessions,
    _stratified_group_split,
)


def _session(session_id, start, end, category="streaming"):
    return {
        "session_id": session_id,
        "scenario_id": "chat_streaming",
        "network_profile": "no_emulation",
        "profile_id": 0,
        "category_name": category,
        "category": CATEGORY_IDS[category],
        "t_start": start,
        "t_end": end,
    }


def _packet(ts, protocol="tcp", payload=20):
    return {
        "ts": ts,
        "size": 100,
        "payload_len": payload,
        "direction": int(ts * 10) % 2,
        "flags": 0,
        "win": 100,
        "eff_win": 100,
        "loopback": False,
        "protocol": protocol,
    }


def test_persistent_flow_is_segmented_per_session_window():
    flows = {
        "tcp:a:1-b:443": [
            *[_packet(1.0 + i * 0.1) for i in range(6)],
            *[_packet(10.0 + i * 0.1) for i in range(6)],
        ]
    }
    samples = _match_flows_to_sessions(
        flows,
        [_session("s1", 1.0, 1.6), _session("s2", 10.0, 10.6)],
        min_packets=5,
        source_pcap="capture.pcap",
        capture_id="run-1",
    )
    assert [sample.session_id for sample in samples] == ["s1", "s2"]
    assert max(packet.timestamp for packet in samples[0].packets) < 2
    assert min(packet.timestamp for packet in samples[1].packets) >= 10
    assert len({sample.flow_id for sample in samples}) == 2


def test_webrtc_category_keeps_udp_media_not_tcp_signaling():
    sessions = [_session("rtc", 1.0, 2.0, "realtime_webrtc")]
    packets = [_packet(1.0 + i * 0.1) for i in range(6)]
    flows = {
        "tcp:a:1-b:443": packets,
        "udp:a:1-b:3478": [{**p, "protocol": "udp"} for p in packets],
    }
    samples = _match_flows_to_sessions(flows, sessions, min_packets=5)
    assert len(samples) == 1
    assert samples[0].protocol == "udp"


def test_group_split_has_no_capture_leakage():
    targets = np.array([cls for cls in (0, 1) for _ in range(60)])
    groups = [f"c{cls}-{group}" for cls in (0, 1) for group in range(12) for _ in range(5)]
    splits = _stratified_group_split(targets, groups, seed=7, trials=500)
    split_groups = {
        name: {groups[index] for index in indices}
        for name, indices in splits.items()
    }
    assert split_groups["train"].isdisjoint(split_groups["val"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["val"].isdisjoint(split_groups["test"])
    for indices in splits.values():
        assert set(targets[indices]) == {0, 1}
