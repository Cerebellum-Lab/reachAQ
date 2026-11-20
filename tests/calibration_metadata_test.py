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
from datetime import datetime
from pathlib import Path
from unittest import mock

import yaml

import top_fixtures
from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.core import ProjectInfo
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.logging import unregister_fatal_exception_callback
from autotrainer.inference.multires_calibration import store
from autotrainer.inference.multires_calibration.fit import BoardSpec
from autotrainer.inference.multires_calibration.geometry import presets_from_base
from multires_calibration_fakes import FIT
from reload_calib_multires_test import (
    BASE,
    BASE_PARAMS,
    _REPORT,
    _configure_left,
    _save_left_camera_params,
    _write_multires,
)
from tools.acquisition.model import app_model as app_model_module
from tools.acquisition.model.app_model import AppModel, _camera_configured_metadata
from tools.acquisition.model.inference_model import InferenceModel


# The hardware.configured keys before calibration3d was added beside them.
_HARDWARE_KEYS = {
    "canEnabled", "pelletControllerEnabled", "nidaqEnabled", "laserBackend", "liveInferenceEnabled",
    "pelletFirmwareCompatibility", "cameras",
}


def _saved_calibration3d(app_model, tmp_path, *, session_scope=True) -> dict:
    """Save the metadata as a session (or the acquisition) would, and give calibration3d from both files."""
    project = app_model.project.to_local_value()
    output = tmp_path / "session003" / "metadata"
    output.parent.mkdir(parents=True, exist_ok=True)
    app_model._save_metadata(project, project.when, str(output), project.session if session_scope else None)
    saved = json.loads(output.with_suffix(".json").read_text())
    configured = saved["hardware"]["configured"]
    assert set(configured) == _HARDWARE_KEYS | {"calibration3d"}
    # The per-camera entries are exactly what they were.
    assert configured["cameras"] == {
        camera.name: _camera_configured_metadata(camera) for camera in app_model._cameras
    }
    from_yaml = yaml.safe_load(output.with_suffix(".yaml").read_text())["hardware"]["configured"]["calibration3d"]
    assert from_yaml == configured["calibration3d"]
    return configured["calibration3d"]


def test_metadata_names_the_active_calibration(app_model, tmp_path):
    folder = _write_multires(tmp_path / "Autotrainer-calib" / DEFAULT_3D_CALIB_DIR_NAME)
    _configure_left(app_model, **BASE_PARAMS)
    app_model.reload_calib(folder)
    index = app_model.active_calibration_index
    assert index is not None and app_model._pose_algorithm.stereo_params is not None

    for session_scope in (True, False):
        calibration3d = _saved_calibration3d(app_model, tmp_path, session_scope=session_scope)
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
    _configure_left(app_model, **BASE_PARAMS)
    app_model.reload_calib(folder)
    assert _saved_calibration3d(app_model, tmp_path)["override"] == override


def test_metadata_names_a_legacy_calibration(app_model, calib_dir, tmp_path):
    # The app_model fixture loaded the legacy fixture calibration when it was built.
    assert app_model._pose_algorithm.stereo_params is not None
    assert _saved_calibration3d(app_model, tmp_path) == {
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
    _configure_left(app_model, **{**BASE_PARAMS, "offsetx": 0, "offsety": 0})
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(folder)
    (_, shown), = [call.args for call in on_error.call_args_list]
    index = app_model.active_calibration_index
    assert app_model._pose_algorithm.stereo_params is None and index is not None

    assert _saved_calibration3d(app_model, tmp_path) == {
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
    calibration3d = _saved_calibration3d(app_model, tmp_path)
    reason = calibration3d.pop("live3dDisabledReason")
    assert calibration3d == {"id": None, "created": None, "preset": None, "folder": None, "override": None}
    assert str(missing) in reason


def test_metadata_says_why_a_broken_calibration_was_not_used(
        app_model, tmp_path, system_config, trainer_config_dir):
    folder = _write_multires(tmp_path / "Autotrainer-calib" / DEFAULT_3D_CALIB_DIR_NAME)
    _save_left_camera_params(system_config, trainer_config_dir, **BASE_PARAMS)
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
