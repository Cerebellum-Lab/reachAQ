"""Live 3D is a setting: on by default, a plain toggle, stored in calibrations/settings.yaml.

Off, 2D pose keeps running; the drift monitor and the DCS are unavailable, and the reason is
"live 3D is off", the way a missing live 3D is already reported (live3dDisabledReason).
"""

import os
from pathlib import Path
from types import SimpleNamespace

import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.core import Offset3DTuple
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.inference.multires_calibration import library
from calibration_metadata_test import _saved_calibration3d  # noqa: E402
from tools.acquisition.view.preferences_content import PreferencesContent  # noqa: E402

LIVE_3D_OFF = "live 3D is off"


def _settings_file(app_model) -> Path:
    return library.settings_path(app_model.calibration_library_root)


def _valid_dcs() -> DiamondTriangleOffsetConfig:
    return DiamondTriangleOffsetConfig(
        used_position=Offset3DTuple(7.0, 4.5, 6.5),
        measured_offset=Offset3DTuple(4.7, -34.0, -8.4),
        diamond_coord=Offset3DTuple(0, 0, 0),
        raw_diamond_coord=Offset3DTuple(3.6, -19.3, 0.7),
        version=DiamondTriangleOffsetConfig.current_config_version,
    )


def test_live_3d_on_by_default(app_model):
    assert not _settings_file(app_model).exists()
    assert app_model.live_3d_enabled is True
    assert app_model._pose_algorithm.stereo_params is not None
    assert app_model._live_3d_disabled_reason is None


def test_a_missing_empty_or_corrupt_settings_file_reads_as_defaults(app_model):
    path = _settings_file(app_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    for text in ("", "{{", "- a\n- list\n", "live3d: [not, a, bool]\n"):
        path.write_text(text, encoding="utf-8")
        assert app_model.live_3d_enabled is True, repr(text)


def test_live_3d_off_keeps_2d_pose(app_model):
    algorithm = app_model._pose_algorithm
    frame_rate = algorithm.frame_rate

    app_model.live_3d_enabled = False

    off = app_model._pose_algorithm
    assert off is not algorithm
    assert off.stereo_params is None and off._live_3d_reason() is not None
    assert off.frame_rate == frame_rate
    assert app_model._live_3d_disabled_reason == LIVE_3D_OFF

    app_model.live_3d_enabled = True
    assert app_model._pose_algorithm.stereo_params is not None
    assert app_model._live_3d_disabled_reason is None


def test_live_3d_off_disables_drift_monitor_and_dcs_with_reason(app_model):
    algo = app_model._behavior.algorithm
    algo.diamond_triangle_config = _valid_dcs()
    app_model.refresh_dcs_state()
    offset, motor = Offset3DTuple(1.0, 2.0, 3.0), Offset3DTuple(0.0, 0.0, 0.0)

    algo.handle_diamond_triangle_offset(offset, motor)
    assert algo.diamond_triangle_config is not None and len(algo._diamond_triangle_prev_drifts) == 1
    assert app_model._hardware._dcs_unavailable_reason is None

    app_model.live_3d_enabled = False

    assert algo.diamond_triangle_config is None
    algo.handle_diamond_triangle_offset(offset, motor)
    assert len(algo._diamond_triangle_prev_drifts) == 1, "the drift monitor does not run"
    assert app_model._hardware._dcs_unavailable_reason == LIVE_3D_OFF
    assert app_model._live_3d_disabled_reason == LIVE_3D_OFF
    # The configuration is held, not discarded, and live 3D off does not make Record wait on DCS.
    assert algo.loaded_diamond_triangle_config is not None
    assert not [b for b in app_model.recording_blockers if "oordinate" in b]

    app_model.live_3d_enabled = True
    assert algo.diamond_triangle_config is not None
    assert app_model._hardware._dcs_unavailable_reason is None


def test_toggle_persists_in_settings_yaml(app_model):
    path = _settings_file(app_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"checks": {"quality": "warn"}}), encoding="utf-8")

    app_model.live_3d_enabled = False

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"checks": {"quality": "warn"}, "live3d": False}
    assert not list(path.parent.glob("settings.yaml.*")), "no temp file is left behind"
    assert app_model.live_3d_enabled is False

    app_model.live_3d_enabled = True
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"checks": {"quality": "warn"}, "live3d": True}


def test_metadata_records_live3d_state(app_model, tmp_path):
    assert _saved_calibration3d(app_model, tmp_path)["live3d"] == {"enabled": True, "disabledReason": None}

    app_model.live_3d_enabled = False
    calibration3d = _saved_calibration3d(app_model, tmp_path)
    assert calibration3d["live3d"] == {"enabled": False, "disabledReason": LIVE_3D_OFF}
    assert calibration3d["live3dDisabledReason"] == LIVE_3D_OFF


def test_preferences_toggle_switches_live_3d_and_shows_the_note_when_off(app_model):
    QApplication.instance() or QApplication([])
    app_model._inference = SimpleNamespace(is_enabled=False, model_location="", pose_parts=[])
    content = PreferencesContent(app_model.preferences, app_model)
    try:
        toggle, note = content._live_3d_toggle, content._live_3d_note
        assert toggle.isChecked() and not note.isVisibleTo(note.parentWidget())
        assert note.text() == "Drift monitor and DCS need live 3D"

        toggle.setChecked(False)
        assert app_model.live_3d_enabled is False and note.isVisibleTo(note.parentWidget())

        toggle.setChecked(True)
        assert app_model.live_3d_enabled is True and not note.isVisibleTo(note.parentWidget())
    finally:
        content.close()
