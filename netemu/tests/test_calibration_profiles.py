"""Directional calibration catalogs must retain the measured service model."""

import copy
import shlex

import pytest
import yaml

from netemu import NetworkEmulator, NetworkProfile, load_profile_catalog
from netemu.exceptions import ProfileLoadError
from netemu.tc import netem_parameters, shaping_commands


@pytest.fixture
def catalog():
    return {"schema_version": 1, "cases": {
        "satellite/downlink": {"parameters": {
            "delay_ms": 22, "jitter_ms": 7, "delay_distribution": "normal",
            "rate_mode": "htb_netem", "rate_mbit": 100, "loss_model": "random",
            "loss_pct": 0.5, "limit_packets": 10000,
        }, "targets": {"delay_ms": 999}, "tune": {"delay_ms": {"min": 0}}},
        "satellite/uplink": {"parameters": {
            "delay_ms": 22, "jitter_ms": 8, "delay_distribution": "normal",
            "rate_mode": "htb_netem", "rate_mbit": 15, "loss_model": "random",
            "loss_pct": 0.8, "limit_packets": 5000,
        }},
        **{f"no_emulation/{direction}": {"parameters": {"bypass": True}}
           for direction in ("uplink", "downlink")},
    }}


def write_catalog(tmp_path, data):
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def emulator(tmp_path, catalog, monkeypatch):
    emulator = NetworkEmulator(interface="test0", profiles_path=write_catalog(tmp_path, catalog))
    commands = []
    monkeypatch.setattr(emulator, "_run_tc_command", lambda cmd, **kw: commands.append(cmd) or True)
    return emulator, commands


def netem_command(commands, device):
    return next(cmd for cmd in commands if f"dev {device} " in cmd and " netem " in cmd)


def test_pairs_keep_cli_names_and_ignore_fitting_targets(emulator):
    emu, commands = emulator
    assert set(emu.list_profiles()) == {"satellite", "no_emulation"}
    assert emu.get_profile("satellite").delay_ms == 22
    assert emu.apply_profile("satellite")
    uplink = netem_command(commands, "test0")
    downlink = netem_command(commands, "ifb0")
    assert "rate 15mbit" in uplink and "limit 5000" in uplink and "8ms" in uplink
    assert "rate 100mbit" in downlink and "limit 10000" in downlink and "7ms" in downlink
    assert "loss random 0.8%" in uplink and "loss random 0.5%" in downlink
    assert "999" not in " ".join(commands)


def test_egress_only_and_explicit_direction_select_uplink_or_downlink(emulator):
    emu, commands = emulator
    emu.bidirectional = False
    assert emu.apply_profile("satellite")
    assert "rate 15mbit" in netem_command(commands, "test0")
    assert not any("dev ifb0" in cmd for cmd in commands)
    commands.clear()
    assert emu.apply_profile("satellite/downlink")
    assert "rate 100mbit" in netem_command(commands, "test0")


def test_ingress_override_and_baseline_bypass(emulator):
    emu, commands = emulator
    assert emu.apply_profile("satellite", ingress_profile="no_emulation")
    assert not any(" add dev ifb0 " in cmd for cmd in commands)
    commands.clear()
    assert emu.apply_profile("no_emulation")
    assert not any(" add " in cmd or "modprobe" in cmd for cmd in commands)
    commands.clear()
    assert emu.apply_profile("satellite", ingress_profile="none")
    assert not any(" add dev ifb0 " in cmd for cmd in commands)


def test_loopback_has_distinct_directional_service_models(emulator):
    emu, commands = emulator
    assert emu.apply_profile_to_loopback("satellite", 9001)
    up = next(c for c in commands if "parent 20:11" in c)
    down = next(c for c in commands if "parent 30:11" in c)
    assert "rate 15mbit" in up and "rate 100mbit" in down
    filters = [c for c in commands if "filter add" in c]
    assert len(filters) == 4
    assert any("dst_port 9001 flowid 1:2" in c for c in filters)
    assert any("src_port 9001 flowid 1:3" in c for c in filters)
    assert len({shlex.split(c)[shlex.split(c).index("prio") + 1] for c in filters}) == 4
    assert "priomap " + " ".join(["0"] * 16) in commands[1]


def test_calibration_keeps_webrtc_udp_and_tcp_selectors(emulator, caplog):
    emu, commands = emulator
    assert emu.apply_profile_to_loopback(
        "satellite", selectors=[("udp", None), ("tcp", 1234)])
    filters = [c for c in commands if "filter add" in c]
    assert len(filters) == 6
    assert sum("ip_proto udp flowid 1:3" in c for c in filters) == 2
    assert sum("ip_proto tcp dst_port 1234 flowid 1:2" in c for c in filters) == 2
    assert sum("ip_proto tcp src_port 1234 flowid 1:3" in c for c in filters) == 2
    assert "downlink service model in both directions" in caplog.text
    assert emu._lo_selectors == (("udp", None), ("tcp", 1234))


def test_udp_server_port_precedes_protocol_wide_fallback(emulator):
    emu, commands = emulator
    assert emu.apply_profile_to_loopback(
        "satellite", selectors=[("udp", None), ("udp", 5004)])
    for family in ("ip", "ipv6"):
        filters = [c for c in commands if f"protocol {family} " in c]
        assert "dst_port 5004 flowid 1:2" in filters[0]
        assert "src_port 5004 flowid 1:3" in filters[1]
        assert "ip_proto udp flowid 1:3" in filters[2]


@pytest.mark.parametrize("selectors", [(), (("udp", None),)])
def test_unscoped_calibration_uses_complete_downlink_service(emulator, selectors):
    emu, commands = emulator
    assert emu.apply_profile_to_loopback("satellite", selectors=selectors)
    assert not any("parent 1:2" in c for c in commands)
    assert any("parent 30:11" in c and "rate 100mbit" in c for c in commands)
    assert any("classid 30:1 htb rate 100mbit" in c for c in commands)


@pytest.mark.parametrize("selector", [("sctp", None), ("tcp", 0), ("udp", 65536),
                                      ("tcp", True), ("udp", 1.5)])
def test_bad_loopback_selector_leaves_existing_shaping_untouched(emulator, selector):
    emu, commands = emulator
    with pytest.raises(ValueError):
        emu.apply_profile_to_loopback("satellite", selectors=[selector])
    assert commands == []


def test_calibrated_loopback_failure_rolls_back(emulator, monkeypatch):
    emu, commands = emulator
    def run(cmd, **kwargs):
        commands.append(cmd)
        return "filter add" not in cmd
    monkeypatch.setattr(emu, "_run_tc_command", run)
    assert not emu.apply_profile_to_loopback("satellite", selectors=[("udp", None)])
    assert emu._lo_port is None and emu._lo_selectors == ()
    assert commands[-1] == "sudo tc qdisc del dev lo root"


def test_loopback_baseline_clears_prior_selectors(emulator):
    emu, commands = emulator
    assert emu.apply_profile_to_loopback("satellite", selectors=[("udp", None)])
    commands.clear()
    assert emu.apply_profile_to_loopback("no_emulation", selectors=[("udp", None)])
    assert commands == ["sudo tc qdisc del dev lo root"]
    assert emu._lo_selectors == ()


@pytest.mark.parametrize("mode,native_rate,htb", [
    ("netem", "rate 10mbit", False),
    ("htb", None, True),
    ("htb_netem", "rate 20mbit", True),
])
def test_explicit_rate_modes_and_byte_bursts(mode, native_rate, htb):
    data = {"rate_mode": mode, "rate_mbit": 10}
    if htb:
        data.update(ceil_mbit=20, burst_bytes=1500, cburst_bytes=3000)
    profile = NetworkProfile.from_dict("rate", data)
    commands = shaping_commands(profile, "test0")
    assert any("htb rate 10mbit ceil 20mbit burst 1500 cburst 3000" in c for c in commands) == htb
    leaf = netem_command(commands, "test0")
    if native_rate:
        assert native_rate in leaf
    else:
        assert " rate " not in leaf


def test_combined_native_cap_override_and_slot_fields():
    profile = NetworkProfile.from_dict("rate", {
        "rate_mode": "htb_netem", "rate_mbit": 10, "ceil_mbit": 20,
        "netem_rate_mbit": 12, "slot_distribution": "normal",
        "slot_delay_ms": 2, "slot_jitter_ms": 0.1,
        "slot_max_packets": 5, "slot_max_bytes": 6000,
    })
    params = " ".join(netem_parameters(profile))
    assert "rate 12mbit" in params
    assert "slot distribution normal 2ms 0.1ms packets 5 bytes 6000" in params


@pytest.mark.parametrize("loss,burst", [(1, 100 / 35), (3, 2.5)])
def test_ge_commands_have_requested_stationary_loss_and_burst_length(loss, burst):
    profile = NetworkProfile.from_dict("ge", {
        "loss_model": "gemodel", "loss_pct": loss, "loss_mean_burst_packets": burst,
    })
    tokens = shlex.split(" ".join(netem_parameters(profile)))
    start = tokens.index("gemodel")
    enter, leave = (float(v.rstrip("%")) / 100 for v in tokens[start + 1:start + 3])
    assert 100 * enter / (enter + leave) == pytest.approx(loss)
    assert 1 / leave == pytest.approx(burst)
    assert tokens[start + 3:start + 5] == ["100%", "0%"]


def test_uniform_delay_and_legacy_rate_behavior():
    profile = NetworkProfile.from_dict("legacy", {
        "delay_ms": 2, "jitter_ms": 0.1, "delay_distribution": "uniform",
        "rate_mbit": 10, "rate_burst_kbit": 12,
    })
    commands = shaping_commands(profile, "test0")
    assert any("burst 12kbit" in c for c in commands)
    leaf = netem_command(commands, "test0")
    assert "distribution uniform" not in leaf
    assert " rate " not in leaf


@pytest.mark.parametrize("change,match", [
    (lambda c: c.update(schema_version=2), "schema_version"),
    (lambda c: c["cases"].pop("satellite/uplink"), "both uplink and downlink"),
    (lambda c: c["cases"]["satellite/uplink"].update(parameters=[]), "requires parameters"),
    (lambda c: c["cases"]["satellite/uplink"]["parameters"].update(rate_mod="htb"), "unknown fields"),
    (lambda c: c["cases"]["satellite/uplink"]["parameters"].update(rate_mode="typo"), "rate_mode"),
    (lambda c: c["cases"]["satellite/uplink"]["parameters"].update(delay_ms=float("nan")), "finite"),
    (lambda c: c["cases"]["satellite/uplink"]["parameters"].update(loss_model="gemodel"), "loss_mean_burst"),
])
def test_invalid_catalog_is_rejected_atomically(emulator, tmp_path, catalog, change, match):
    emu, commands = emulator
    original = emu.profiles
    changed = copy.deepcopy(catalog)
    change(changed)
    with pytest.raises(ProfileLoadError, match=match):
        emu.load_profiles(write_catalog(tmp_path, changed))
    assert emu.profiles is original
    assert commands == []


def test_shaping_failure_rolls_back(emulator, monkeypatch):
    emu, commands = emulator
    def run(cmd, **kwargs):
        commands.append(cmd)
        return not ("dev ifb0" in cmd and " netem " in cmd)
    monkeypatch.setattr(emu, "_run_tc_command", run)
    assert not emu.apply_profile("satellite")
    assert emu.current_profile is None
    assert any("del dev ifb0 root" in cmd for cmd in commands)
    assert commands[-1] == "sudo ip link set dev ifb0 down"


def test_legacy_uplink_override_is_applied_without_testbed_logic(tmp_path, monkeypatch):
    path = write_catalog(tmp_path, {"profiles": {
        "legacy": {"delay_ms": 20, "rate_mbit": 100, "uplink": {"rate_mbit": 10}},
    }})
    catalog = load_profile_catalog(path)
    assert catalog["profiles"]["legacy"]["uplink"]["rate_mbit"] == 10
    emu = NetworkEmulator(profiles_path=path, bidirectional=False)
    commands = []
    monkeypatch.setattr(emu, "_run_tc_command", lambda cmd, **kw: commands.append(cmd) or True)
    assert emu.apply_profile("legacy")
    assert any("htb rate 10mbit" in c for c in commands)
