"""The active calibration catalog must work through testbed entry points."""

from pathlib import Path

import pytest
import yaml

from netemu import load_profile_catalog
from orchestrator import TestbedOrchestrator as Orchestrator


APP_DIR = Path(__file__).resolve().parents[1]
CATALOG = APP_DIR / "configs/profiles.yaml"


def test_active_catalog_covers_the_matrix_and_report_readers(tmp_path):
    from analysis.ran2_metrics import _load_profile_loss_pct
    from generate_results_md import _load_profile_descriptions

    raw = yaml.safe_load(CATALOG.read_text())
    assert raw["schema_version"] == 1
    assert len(raw["cases"]) == 22
    normalized = load_profile_catalog(CATALOG)
    assert len(normalized["profiles"]) == 11
    scenarios = yaml.safe_load((APP_DIR / "configs/scenarios.yaml").read_text())
    requested = {p for entry in scenarios["test_matrix"] for p in entry["profiles"]}
    assert requested <= normalized["profiles"].keys()
    assert _load_profile_loss_pct(str(CATALOG))["5qi_80"] == 0.0001
    assert set(_load_profile_descriptions(str(CATALOG))) == set(normalized["profiles"])


def test_orchestrator_applies_calibrated_directions(tmp_path, monkeypatch):
    runner = Orchestrator(config_path=APP_DIR / "configs/scenarios.yaml",
                          profiles_path=CATALOG, db_path=str(tmp_path / "test.db"),
                          network_interface="test0")
    commands = []
    monkeypatch.setattr(runner.emulator, "_run_tc_command",
                        lambda cmd, **kw: commands.append(cmd) or True)
    assert runner._apply_emulator_profile("satellite_geo", None)
    assert any("dev test0" in c and " netem " in c and "rate 3mbit" in c for c in commands)
    assert any("dev ifb0" in c and " netem " in c and "rate 50mbit" in c for c in commands)


def test_failed_profile_does_not_run_a_scenario(tmp_path, monkeypatch):
    runner = Orchestrator(config_path=APP_DIR / "configs/scenarios.yaml",
                          profiles_path=CATALOG, db_path=str(tmp_path / "test.db"),
                          network_interface="test0")
    monkeypatch.setattr(runner, "get_client", lambda provider: None)
    monkeypatch.setattr(runner, "_apply_emulator_profile", lambda *args: False)
    class Scenario:
        def __init__(self, *args):
            pass
        def run(self, **kwargs):
            pytest.fail("Workload executed despite failed shaping")
    runner.scenario_classes["chat"] = Scenario
    with pytest.raises(RuntimeError, match="experiment aborted"):
        runner.run_experiment("chat_basic", "5g_urban", runs=1)
