import json

import yaml

from reachaq.session_validation import ValidationProfile, validate_session
from reachaq.session_validation.engine import _Context, _provenance_rule


def _context(tmp_path, metadata):
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"authoritativeMetadata": "metadata.json"}), encoding="utf-8")
    return _Context(tmp_path, ValidationProfile.QUICK)


def test_provenance_warns_for_simulated_metadata(tmp_path):
    result = _provenance_rule(_context(tmp_path, {
        "simulation": {"enabled": True, "bus": "/tmp/bus", "backends": {}},
    }))

    assert result.rule_id == "session.provenance"
    assert result.status.value == "warning"
    assert result.message == "simulated session: no hardware was recorded"


def test_provenance_passes_without_block(tmp_path):
    result = _provenance_rule(_context(tmp_path, {"sessionId": "s"}))

    assert result.status.value == "pass"
    assert result.message == "recorded on hardware backends"


def test_provenance_rule_runs_in_the_quick_profile(tmp_path):
    _context(tmp_path, {"simulation": {"enabled": True}})

    report = validate_session(
        tmp_path, profile=ValidationProfile.QUICK,
        selected_rules=("session.provenance",),
    )

    result = next(r for r in report.results if r.rule_id == "session.provenance")
    assert result.status.value == "warning"


def _save_session_metadata(app_model, tmp_path):
    project = app_model.project.to_local_value()
    session_dir = tmp_path / "session003"
    (session_dir / "streams").mkdir(parents=True)
    (session_dir / "streams" / "alignment.json").write_text("{}")
    (session_dir / "streams" / "trial_summary.json").write_text("{}")
    output = session_dir / "metadata"
    app_model._save_metadata(project, project.when, str(output), project.session)
    saved = json.loads(output.with_suffix(".json").read_text())
    mirror = yaml.safe_load(output.with_suffix(".yaml").read_text())
    return saved, mirror


def test_metadata_records_simulation_only_when_simulating(
    app_model, tmp_path, monkeypatch,
):
    monkeypatch.delenv("REACHAQ_SIMULATION", raising=False)
    saved, _ = _save_session_metadata(app_model, tmp_path / "real")
    assert "simulation" not in saved

    monkeypatch.setenv("REACHAQ_SIMULATION", "1")
    monkeypatch.setenv("REACHAQ_SIM_BUS", "/tmp/sim-bus")
    saved, mirror = _save_session_metadata(app_model, tmp_path / "sim")
    assert saved["simulation"] == {
        "enabled": True,
        "bus": "/tmp/sim-bus",
        "backends": {
            "camera": "sim", "nidaq": "simulated",
            "laser": "simulated", "can": "emulation",
        },
    }
    assert mirror == saved
