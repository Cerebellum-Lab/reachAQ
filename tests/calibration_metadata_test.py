"""Each session names the 3D calibration live 3D used, and the app's own analysis finds it again.

hardware.configured.calibration3d sits beside the per-camera entries, which stay as they were.
Its id finds a format-2 calibration offline wherever it then sits, active or backed up; created
and folder say which one it was at a glance. When live 3D was off, the entry still names the
calibration, has no preset, and says why live 3D did not run.

The app's intersession analysis is given the calibration root, not a folder, so each session
is processed with its own calibration, as live 3D was.
"""

import dataclasses
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest
import yaml

import top_fixtures
from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.core import ProjectInfo
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.logging import unregister_fatal_exception_callback
from autotrainer.inference.analysis.intersession_process import resolve_session_calibration
from autotrainer.inference.multires_calibration import library, store
from autotrainer.inference.multires_calibration.fit import BoardSpec
from autotrainer.inference.multires_calibration.geometry import presets_from_base
from multires_calibration_fakes import FIT
from reload_calib_multires_test import (  # noqa: F401 (start_app is a fixture)
    BASE,
    BASE_PARAMS,
    SERIALS,
    _REPORT,
    _active_entry,
    _configure_live_pair,
    _configure_with_serials,
    _legacy_copy_without_offsets,
    _save_live_pair_params,
    _write_multires,
    start_app,
)
from tools.acquisition.model import app_model as app_model_module
from tools.acquisition.model.app_model import AppModel, _camera_configured_metadata
from tools.acquisition.model.inference_model import InferenceModel


# The hardware.configured keys before calibration3d was added beside them.
_HARDWARE_KEYS = {
    "canEnabled", "pelletControllerEnabled", "nidaqEnabled", "laserBackend", "liveInferenceEnabled",
    "pelletFirmwareCompatibility", "cameras",
}
# What schema 3 adds to each camera's entry.
_CAMERA_KEYS = ("serial", "cameraId", "inCalibration", "uncalibratedReason")
# calibration3d's keys at schema 2; schema 3 adds the others beside them.
_SCHEMA_2_KEYS = ("id", "created", "preset", "folder", "override", "live3dDisabledReason")


def _schema_2_view(calibration3d: dict) -> dict:
    return {key: calibration3d[key] for key in _SCHEMA_2_KEYS if key in calibration3d}


def _write_session_metadata(project: ProjectInfo, data: dict) -> None:
    path = Path(project.get_metadata_file() + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _saved_calibration3d(app_model, tmp_path, *, session_scope=True, with_document=False):
    """Save the metadata as a session (or the acquisition) would, and give calibration3d from both files."""
    project = app_model.project.to_local_value()
    output = tmp_path / "session003" / "metadata"
    output.parent.mkdir(parents=True, exist_ok=True)
    app_model._save_metadata(project, project.when, str(output), project.session if session_scope else None)
    saved = json.loads(output.with_suffix(".json").read_text())
    assert saved["metadataSchemaVersion"] == 3
    configured = saved["hardware"]["configured"]
    assert set(configured) == _HARDWARE_KEYS | {"calibration3d"}
    # The per-camera entries are what they were, plus the camera's identity and calibration.
    for camera in app_model._cameras:
        entry = configured["cameras"][camera.name]
        assert {k: v for k, v in entry.items() if k not in _CAMERA_KEYS} == _camera_configured_metadata(camera)
        assert set(_CAMERA_KEYS) <= set(entry)
    from_yaml = yaml.safe_load(output.with_suffix(".yaml").read_text())["hardware"]["configured"]["calibration3d"]
    assert from_yaml == configured["calibration3d"]
    if with_document:
        return configured["calibration3d"], saved
    return configured["calibration3d"]


def test_metadata_names_the_active_calibration(app_model, tmp_path):
    folder = _write_multires(tmp_path / "Autotrainer-calib" / DEFAULT_3D_CALIB_DIR_NAME)
    _configure_live_pair(app_model, **BASE_PARAMS)
    app_model.reload_calib(folder)
    index = app_model.active_calibration_index
    assert index is not None and app_model._pose_algorithm.stereo_params is not None

    for session_scope in (True, False):
        calibration3d = _schema_2_view(_saved_calibration3d(app_model, tmp_path, session_scope=session_scope))
        assert calibration3d == {
            "id": index.id,
            "created": index.created.isoformat(timespec="seconds"),
            "preset": BASE.name,
            "folder": str(folder),
            "override": None,
        }


def test_metadata_names_an_overridden_calibration_with_its_override(app_model, tmp_path):
    override = {"by": "operator", "reason": "baseline warning expected after the remount"}
    folder = tmp_path / "Autotrainer-calib" / DEFAULT_3D_CALIB_DIR_NAME
    folder.parent.mkdir(parents=True)
    store.write_calibration(folder, fit=FIT, presets=presets_from_base(BASE, (4, 2, 1)), board=BoardSpec(),
                            report=dataclasses.replace(_REPORT, override=override), capture={})
    _configure_live_pair(app_model, **BASE_PARAMS)
    app_model.reload_calib(folder)
    assert _saved_calibration3d(app_model, tmp_path)["override"] == override


def test_metadata_names_a_legacy_calibration(app_model, calib_dir, tmp_path):
    # The app_model fixture loaded the legacy fixture calibration when it was built.
    assert app_model._pose_algorithm.stereo_params is not None
    assert _schema_2_view(_saved_calibration3d(app_model, tmp_path)) == {
        "id": None,
        # A legacy folder records no date; its stereo pickle's time stands in, as offline reads it.
        "created": store.calibration_created(calib_dir).isoformat(timespec="seconds"),
        "preset": None,
        "folder": str(calib_dir),
        "override": None,
    }


def test_metadata_says_why_live_3d_was_off(app_model, tmp_path):
    folder = _write_multires(tmp_path / "Autotrainer-calib" / DEFAULT_3D_CALIB_DIR_NAME)
    # The operator moved the ROI: no preset of this calibration has these pixels.
    _configure_live_pair(app_model, **{**BASE_PARAMS, "offsetx": 0, "offsety": 0})
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(folder)
    (_, shown), = [call.args for call in on_error.call_args_list]
    index = app_model.active_calibration_index
    assert app_model._pose_algorithm.stereo_params is None and index is not None

    assert _schema_2_view(_saved_calibration3d(app_model, tmp_path)) == {
        "id": index.id,
        "created": index.created.isoformat(timespec="seconds"),
        "preset": None,
        "folder": str(folder),
        "override": None,
        # What the operator was shown.
        "live3dDisabledReason": shown,
    }


def test_metadata_says_when_there_was_no_calibration(app_model, tmp_path):
    missing = tmp_path / "missing" / DEFAULT_3D_CALIB_DIR_NAME
    app_model.reload_calib(missing)
    calibration3d = _schema_2_view(_saved_calibration3d(app_model, tmp_path))
    reason = calibration3d.pop("live3dDisabledReason")
    assert calibration3d == {"id": None, "created": None, "preset": None, "folder": None, "override": None}
    assert str(missing) in reason


def test_metadata_says_why_a_broken_calibration_was_not_used(
        app_model, tmp_path, system_config, trainer_config_dir):
    folder = _write_multires(tmp_path / "Autotrainer-calib" / DEFAULT_3D_CALIB_DIR_NAME)
    _save_live_pair_params(system_config, trainer_config_dir, **BASE_PARAMS)
    app_model.reload_calib(folder)
    (folder / "presets" / BASE.name / "camera_matrix" / "stereo_params.pickle").unlink()
    # The configuration load's guard turns live 3D off instead of failing the load.
    with mock.patch.object(app_model, "on_error") as on_error:
        assert app_model.load_configuration() is True
    shown, = [call.args[1] for call in on_error.call_args_list if call.args[0] == "3D calibration"]

    calibration3d = _saved_calibration3d(app_model, tmp_path)
    assert calibration3d["preset"] is None and calibration3d["folder"] == str(folder)
    assert calibration3d["live3dDisabledReason"] == shown
    # A missing preset file leaves the index readable, so the calibration is still named.
    index = store.load_index(folder)
    assert calibration3d["id"] == index.id
    assert calibration3d["created"] == index.created.isoformat(timespec="seconds")


def test_the_app_resolves_each_session_s_own_calibration(
        mock_system, user_pref, system_config, monkeypatch, fake_system_msg_handler):
    made = []

    class _Recording(InferenceModel):
        # Records how the app builds its inference model and hands back the test double.
        def __new__(cls, *args, **kwargs):
            made.append(kwargs)
            return mock_system.inference

    monkeypatch.setattr(app_model_module, "InferenceModel", _Recording)
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    app = AppModel(
        user_pref,
        system_machine=mock_system.system_machine,
        analysis=mock_system.sensor_analysis,
        system_message_handler=fake_system_msg_handler,
    )
    try:
        kwargs, = made
        assert kwargs["calib_dir"] is None
        assert kwargs["calib_root"] == Path(user_pref.configuration_location)
    finally:
        unregister_fatal_exception_callback(app._on_fatal_exception)
        top_fixtures.release_multiprocessing_resources(app)


def test_intersession_analysis_is_given_the_root_not_a_folder(tmp_path):
    model = InferenceModel.__new__(InferenceModel)
    model._calib_dir = None
    model._calib_root = tmp_path / "Autotrainer"
    model._event_manager = mock.Mock()
    model._intersession_detection = None
    pool = model._process_pool = mock.Mock()
    pool.apply_async.return_value.get.side_effect = RuntimeError("stop after the call is made")
    detection = mock.Mock()
    detection.configuration.project = ProjectInfo(root=str(tmp_path), device_id="rig", session=3,
                                                  when=datetime(2026, 5, 1))
    detection.configuration.frame_rate = 150

    model._intersession_process(detection)

    (args, kwds), = [(c.kwargs["args"], c.kwargs["kwds"]) for c in pool.apply_async.call_args_list]
    assert args == (detection.configuration.project,)
    assert kwds == {"calib_dir": None, "calib_root": tmp_path / "Autotrainer", "frame_rate": 150}
    detection.configuration.complete.assert_called_once()


def test_metadata_schema_3_fields(app_model, tmp_path):
    """Schema 3: the calibration by id and library path, its cameras with serials, and how
    the recording's cameras compare; each camera's serial and place in the calibration."""
    entry = _active_entry(app_model, tmp_path, extra=[("camera4", "21000004", "too few board views: 4 of 12 needed")])
    root = library.library_root(Path(app_model._preferences.configuration_location))
    override = {"confirmed": True, "by": "ben", "at": "2026-11-20T10:00:00"}
    library.record_decision(root, entry.name, "override", override)
    # left is now another body; right records, left does not.
    _configure_with_serials(app_model, {"left": "21000099", "right": SERIALS["right"]})
    app_model.left_camera.is_recording_enabled = True
    app_model.right_camera.is_recording_enabled = True
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(None)
    app_model.acknowledge_setup_difference()
    index = app_model.active_calibration_index

    calibration3d, saved = _saved_calibration3d(app_model, tmp_path, with_document=True)
    assert saved["metadataSchemaVersion"] == 3
    assert calibration3d == {
        "id": entry.name,
        "created": index.created.isoformat(timespec="seconds"),
        "preset": None,
        "folder": str(entry),
        "override": override,
        "path": f"calibrations/{entry.name}",
        "format": 3,
        "cameras": [
            {"name": "left", "serial": "21000001", "cameraId": 0},
            {"name": "right", "serial": "21000002", "cameraId": 1},
            {"name": "camera4", "serial": "21000004", "cameraId": 3},
        ],
        "matchesRecording": False,
        "mismatch": {
            "added": [],
            "removed": [],
            "swapped": [{"name": "left", "calibratedSerial": "21000001", "recordedSerial": "21000099"}],
        },
        "mismatchAcknowledged": True,
        "live3dDisabledReason": app_model._live_3d_disabled_reason,
        "live3d": {"enabled": True, "disabledReason": app_model._live_3d_disabled_reason},
    }
    cameras = saved["hardware"]["configured"]["cameras"]
    assert {k: cameras["left"][k] for k in _CAMERA_KEYS} == {
        "serial": "21000099", "cameraId": 0, "inCalibration": False,
        "uncalibratedReason": "swapped: calibrated serial 21000001, now 21000099",
    }
    assert {k: cameras["right"][k] for k in _CAMERA_KEYS} == {
        "serial": "21000002", "cameraId": 1, "inCalibration": True, "uncalibratedReason": None,
    }

    # A different setup than the one acknowledged is not acknowledged.
    app_model.right_camera.is_recording_enabled = False
    assert _saved_calibration3d(app_model, tmp_path)["mismatchAcknowledged"] is False


def test_schema_2_session_still_resolves(app_model, user_pref, calib_dir, tmp_path):
    """Sessions recorded before the library: one naming a format-2 id resolves to its imported
    entry, and a legacy one by its time to the old folder, which the import leaves in place."""
    config = Path(user_pref.configuration_location)
    # The rig before the library: a legacy active folder made 2026-01-01, and a format-2
    # backup that stopped being active on 2026-01-15.
    legacy = _legacy_copy_without_offsets(calib_dir, tmp_path, config / DEFAULT_3D_CALIB_DIR_NAME)
    made = datetime(2026, 1, 1).timestamp()
    os.utime(legacy / "camera_matrix" / "stereo_params.pickle", (made, made))
    backup = _write_multires(config / f"{DEFAULT_3D_CALIB_DIR_NAME}_20260115_000000")
    index = store.load_index(backup)
    app_model.reload_calib(None)  # the first load imports both
    root = library.library_root(config)
    assert (root / index.id).is_dir() and app_model.active_calibration_dir.parent == root
    # The legacy entry, named by its id, still loads: its board comes from the name it had.
    assert app_model._pose_algorithm.stereo_params is not None and app_model._pose_algorithm.square_size == 4

    named = ProjectInfo(root=str(tmp_path / "Data"), device_id="rig", when=index.created, session=3)
    _write_session_metadata(named, {
        "metadataSchemaVersion": 2, "scope": "session",
        "boundary": {"startWallTime": index.created.timestamp() + 60},
        "hardware": {"configured": {"calibration3d": {
            "id": index.id, "created": index.created.isoformat(), "preset": BASE.name,
            "folder": str(backup), "override": None}}},
    })
    assert resolve_session_calibration(named, config) == root / index.id
    # Before the library existed (the import's set_active is today), by time: the old folder.
    start = datetime(2026, 2, 1, 10)
    unnamed = ProjectInfo(root=str(tmp_path / "Data2"), device_id="rig", when=start, session=4)
    _write_session_metadata(unnamed, {
        "metadataSchemaVersion": 2, "scope": "session", "boundary": {"startWallTime": start.timestamp()}})
    assert resolve_session_calibration(unnamed, config) == legacy


@pytest.mark.parametrize("damage", [
    "no library", "pointer missing its entry", "corrupt pointer", "truncated index", "truncated preset",
])
def test_missing_active_entry_records_2d_with_reason(start_app, user_pref, tmp_path, damage):
    """A missing, corrupt or half-written active entry never stops the app starting: it records
    2D, with the reason in calibration3d and on every camera."""
    config = Path(user_pref.configuration_location)
    root = library.library_root(config)
    entry_id = None
    if damage != "no library":
        built = _write_multires(tmp_path / "built" / DEFAULT_3D_CALIB_DIR_NAME)
        entry_id = library.add_entry(root, built)
        library.set_active(root, entry_id, by="test")
        entry = root / entry_id
        if damage == "pointer missing its entry":
            shutil.rmtree(entry)
        elif damage == "corrupt pointer":
            (root / library.ACTIVE_FILE).write_text("{{", encoding="utf-8")
        elif damage == "truncated index":
            (entry / store.INDEX_FILE).write_bytes(b"")
        else:
            (entry / "presets" / BASE.name / "camera_matrix" / "stereo_params.pickle").write_bytes(b"")
    app, _, _ = start_app()  # must not raise
    reason = app._live_3d_disabled_reason
    assert app._pose_algorithm.stereo_params is None and reason
    if damage in ("truncated index", "truncated preset"):
        assert entry_id in reason
    else:
        assert reason.startswith("no active 3D calibration")
    output = tmp_path / "out" / "metadata"
    output.parent.mkdir(parents=True)
    project = app.project.to_local_value()
    app._save_metadata(project, project.when, str(output), project.session)
    saved = json.loads(output.with_suffix(".json").read_text())
    calibration3d = saved["hardware"]["configured"]["calibration3d"]
    assert calibration3d["live3dDisabledReason"] == reason
    if damage != "truncated preset":
        # Nothing names the cameras' calibration; with a readable index they stay in it.
        for camera in saved["hardware"]["configured"]["cameras"].values():
            assert camera["inCalibration"] is False and camera["uncalibratedReason"]
