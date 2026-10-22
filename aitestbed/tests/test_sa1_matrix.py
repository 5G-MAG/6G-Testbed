"""Matrix completeness, stable resumes, and honest attempt accounting."""

import json
from pathlib import Path

import pytest
import yaml

from sa1_matrix import APP, DEFAULT_SCENARIOS, prepare, summarize


def make_plan(tmp_path):
    asset = tmp_path / "photo.jpg"
    asset.write_bytes(b"fixture identity")
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"photo_paths": [str(asset)] * 6}))
    profiles = yaml.safe_load((APP / "configs/scenarios.yaml").read_text())[
        "runner_profile_sets"
    ]["full_suite_profiles"]
    args = (
        tmp_path / "campaign",
        APP / "configs/sa1.yaml",
        assets,
        APP / "configs/profiles.yaml",
        profiles,
        DEFAULT_SCENARIOS,
        10,
        "eth0",
    )
    return prepare(*args), args


def test_exact_400_cells_and_unchanged_resume(tmp_path):
    plan, args = make_plan(tmp_path)
    keys = {(j["scenario"], j["profile"], j["run_index"]) for j in plan["jobs"]}
    assert len(plan["jobs"]) == len(keys) == 400
    assert len({j["id"] for j in plan["jobs"]}) == 400
    assert all(
        {i for s, p, i in keys if s == scenario and p == profile} == set(range(10))
        for scenario in plan["scenarios"]
        for profile in plan["profiles"]
    )
    assert prepare(*args) == plan
    Path(next(iter(plan["input_hashes"]))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prepare(*args)


def test_failed_unresolved_and_blocked_never_count_as_success(tmp_path):
    plan, args = make_plan(tmp_path)
    output = args[0]
    for job, name, data in zip(
        plan["jobs"],
        ["result.json", "result.json", "started.json", "blocked.json"],
        [{"success": True}, {"success": False}, {}, {}],
    ):
        run = output / job["id"]
        run.mkdir()
        (run / name).write_text(json.dumps(data))
    summary = summarize(output, plan)
    assert summary["planned"] == 400
    assert summary["completed"] == 2
    assert summary["successes"] == summary["failures"] == 1
    assert summary["unresolved"] == summary["blocked"] == 1
    assert len(summary["groups"]) == 40
