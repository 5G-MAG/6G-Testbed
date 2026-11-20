"""Delivery recovery must not confuse attempts with original byte delivery."""

import math

from sa1_segment_metrics import unique_delivery


def packet(t, seq, length):
    return {"t": t, "seq": seq, "payload": length}


def test_resegmented_retransmission_delivers_original_piece_later():
    source = [packet(1000, 100, 100), packet(4000, 100, 50), packet(5000, 150, 50)]
    destination = [packet(6000, 100, 50), packet(7000, 150, 50)]
    pieces = unique_delivery(source, destination)
    assert len(pieces) == 1
    assert pieces[0]["bytes"] == 100
    assert pieces[0]["t"] == 1000
    assert pieces[0]["finish"] == 7000


def test_reordered_source_intervals_and_sequence_wrap():
    source = [
        packet(1000, 2**32 - 20, 10),
        packet(2000, 0, 10),
        packet(3000, 2**32 - 10, 10),
    ]
    destination = [packet(4000, 2**32 - 20, 30)]
    pieces = unique_delivery(source, destination)
    assert sum(p["bytes"] for p in pieces) == 30
    assert [p["t"] for p in pieces] == [1000, 2000, 3000]
    assert all(p["finish"] == 4000 for p in pieces)


def test_missing_bytes_remain_incomplete_not_zero_delay():
    pieces = unique_delivery([packet(1000, 100, 100)], [packet(2000, 100, 50)])
    assert sum(p["bytes"] for p in pieces) == 100
    assert math.isinf(pieces[0]["finish"])
