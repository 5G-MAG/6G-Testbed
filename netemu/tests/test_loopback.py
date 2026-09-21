"""Loopback shaping: selector to tc filter mapping, rate, and state."""

from unittest.mock import patch

import pytest

from netemu import NetworkEmulator, NetworkProfile
from netemu.exceptions import ProfileNotFoundError


def _emulator(**profile_fields) -> NetworkEmulator:
    emu = NetworkEmulator(interface="eth0", bidirectional=False)
    emu.profiles["p"] = NetworkProfile(name="p", **profile_fields)
    return emu


def _apply(emu, *args, **kwargs):
    cmds = []

    def fake(cmd, ignore_errors=False):
        cmds.append(cmd)
        return True

    with patch.object(emu, "_run_tc_command", side_effect=fake):
        ok = emu.apply_profile_to_loopback(*args, **kwargs)
    return ok, cmds


def _filters(cmds):
    return [c for c in cmds if " filter add " in c]


class TestSelectors:
    def test_legacy_dest_port_selects_tcp_both_ways_v4_and_v6(self):
        emu = _emulator(delay_ms=50)
        ok, cmds = _apply(emu, "p", 8080)

        assert ok
        assert cmds[0] == "sudo tc qdisc del dev lo root"
        assert "prio bands 3" in cmds[1]
        assert "netem delay 50ms" in cmds[2]
        filters = _filters(cmds)
        assert len(filters) == 4
        assert all("protocol 6 0xff" in f for f in filters)
        assert sum("match ip dport 8080 0xffff" in f for f in filters) == 1
        assert sum("match ip sport 8080 0xffff" in f for f in filters) == 1
        assert sum("match ip6 dport 8080 0xffff" in f for f in filters) == 1
        assert sum("protocol ipv6 " in f for f in filters) == 2
        assert all(f.endswith("flowid 1:3") for f in filters)
        assert emu._lo_port == 8080
        assert emu._lo_selectors == (("tcp", 8080),)

    def test_webrtc_selectors_shape_all_udp_plus_signaling_tcp(self):
        emu = _emulator(delay_ms=100, jitter_ms=20, loss_pct=1.0)
        ok, cmds = _apply(emu, "p", selectors=[("udp", None), ("tcp", 1234)])

        assert ok
        filters = _filters(cmds)
        udp = [f for f in filters if "protocol 17 0xff" in f]
        tcp = [f for f in filters if "protocol 6 0xff" in f]
        assert len(udp) == 2 and len(tcp) == 4
        # Protocol-only selectors carry no port match.
        assert not any("port" in f for f in udp)
        assert any("match ip dport 1234 0xffff" in f for f in tcp)
        assert emu._lo_selectors == (("udp", None), ("tcp", 1234))

    def test_no_selectors_shapes_everything(self):
        emu = _emulator(loss_pct=5.0)
        ok, cmds = _apply(emu, "p")

        assert ok
        filters = _filters(cmds)
        assert filters == [
            "sudo tc filter add dev lo parent 1:0 protocol all prio 1 u32 "
            "match u32 0 0 flowid 1:3"
        ]

    def test_dest_port_and_selectors_combine(self):
        emu = _emulator(delay_ms=10)
        ok, cmds = _apply(emu, "p", 9001, selectors=[("udp", 5004)])

        assert ok
        assert emu._lo_selectors == (("tcp", 9001), ("udp", 5004))
        assert sum("match ip dport 5004 0xffff" in f for f in _filters(cmds)) == 1

    def test_unknown_protocol_rejected(self):
        emu = _emulator(delay_ms=10)
        with pytest.raises(ValueError):
            emu.apply_profile_to_loopback("p", selectors=[("sctp", None)])

    def test_unknown_profile_rejected(self):
        emu = _emulator(delay_ms=10)
        with pytest.raises(ProfileNotFoundError):
            emu.apply_profile_to_loopback("missing", 80)


class TestRateAndEmpty:
    def test_rate_limit_uses_netem_rate(self):
        emu = _emulator(delay_ms=120, rate_mbit=5)
        ok, cmds = _apply(emu, "p", selectors=[("udp", None)])

        assert ok
        netem = [c for c in cmds if " netem " in c]
        assert len(netem) == 1
        assert "delay 120ms" in netem[0]
        assert netem[0].endswith("rate 5mbit")

    def test_rate_only_profile_still_installs_netem(self):
        emu = _emulator(rate_mbit=2)
        ok, cmds = _apply(emu, "p")

        assert ok
        assert any("netem rate 2mbit" in c for c in cmds)

    def test_no_impairments_only_clears(self):
        emu = _emulator()
        emu._lo_selectors = (("tcp", 1),)
        ok, cmds = _apply(emu, "p", 80)

        assert ok
        assert cmds == ["sudo tc qdisc del dev lo root"]
        assert emu._lo_selectors == ()
        assert emu._lo_port is None


class TestFailureAndClear:
    def test_filter_failure_returns_false(self):
        emu = _emulator(delay_ms=10)

        def fake(cmd, ignore_errors=False):
            return " filter add " not in cmd

        with patch.object(emu, "_run_tc_command", side_effect=fake):
            assert emu.apply_profile_to_loopback("p", 80) is False
        assert emu._lo_selectors == ()

    def test_clear_loopback_resets_state(self):
        emu = _emulator(delay_ms=10)
        _apply(emu, "p", 80)
        with patch.object(emu, "_run_tc_command", return_value=True) as run:
            assert emu.clear_loopback()
        run.assert_called_once_with("sudo tc qdisc del dev lo root", ignore_errors=True)
        assert emu._lo_port is None
        assert emu._lo_selectors == ()

    def test_fresh_emulator_has_loopback_state(self):
        emu = NetworkEmulator(interface="eth0")
        assert emu._lo_port is None
        assert emu._lo_selectors == ()
