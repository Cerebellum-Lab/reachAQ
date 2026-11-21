"""reload_calib loads the preset of a format-2 calibration that matches the live frames' pixels.

Live frames reach inference pooled to the base capture, so the left camera's
configured base geometry picks the preset. A calibration without that preset turns
live 3D off with the reason shown to the operator; another preset is never
substituted, since its pixels are not the frames' pixels. A legacy calibration
loads as before. A missing camera_offsets.pkl, which a fresh calibration has until
DCS is redone, gives zero offsets instead of a crash.
"""

import dataclasses
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np
import pytest
import yaml

import top_fixtures
from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.core import CameraConfiguration, CameraId
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.logging import unregister_fatal_exception_callback
from autotrainer.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME
from autotrainer.inference.multires_calibration import store
from autotrainer.inference.multires_calibration.fit import BoardSpec, CaptureCamera
from autotrainer.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from autotrainer.inference.multires_calibration.report import CameraVerdict, QualityReport
from autotrainer.inference.multires_calibration import library
from multires_calibration_fakes import BASE, FIT
from tools.acquisition.model.app_model import AppModel, SetupDifference


BASE_PARAMS = dict(hbin=4, vbin=4, width=256, height=256, offsetx=52, offsety=6)
PARTS = ['Pellet', 'RH_flat', 'RH_spread', 'RH_grab', 'LH_flat', 'LH_spread', 'LH_grab',
         'Star', 'Tongue_mid', 'Tongue_tip', 'Nose', 'Triangle', 'Mouth', 'Diamond']
ZERO_OFFSETS = {"x_off": 0.0, "y_off": 0.0, "z_off": 0.0}
_REPORT = QualityReport(created="2026-10-20T12:00:00-06:00", verdict="pass", metrics=[],
                        thresholds={}, notes=[])


def _write_multires(folder: Path, base=BASE, *, binnings=(4, 2, 1), offsets=None) -> Path:
    folder.parent.mkdir(parents=True, exist_ok=True)
    store.write_calibration(folder, fit=FIT, presets=presets_from_base(base, binnings),
                            board=BoardSpec(), report=_REPORT, capture={})
    if offsets is not None:
        with (folder / DEFAULT_CAM_OFFSET_FILE_NAME).open("wb") as fh:
            pickle.dump(offsets, fh)
    return folder


def _legacy_copy_without_offsets(calib_dir: Path, tmp_path: Path, folder: Optional[Path] = None) -> Path:
    folder = folder or tmp_path / "legacy" / DEFAULT_3D_CALIB_DIR_NAME
    (folder / "camera_matrix").mkdir(parents=True)
    for name in ("camera_matrix/stereo_params.pickle", "calibration_userset.yaml"):
        shutil.copyfile(calib_dir / name, folder / name)
    # The camera names come from the source videos' file names alone.
    (folder / "source_videos").mkdir()
    for video in (calib_dir / "source_videos").iterdir():
        (folder / "source_videos" / video.name).touch()
    return folder


def _configure_live_pair(app_model, **params):
    """Configure left and right at the same params: live 3D needs the pair at one geometry."""
    for camera, camera_id in ((app_model.left_camera, CameraId.Left), (app_model.right_camera, CameraId.Right)):
        camera.load_configuration(
            CameraConfiguration(id=camera_id, name=str(camera_id), scheme="random", params=dict(params)))


def _save_live_pair_params(system_config, trainer_config_dir, **params):
    """Set the left and right cameras' params in the configuration file load_configuration reads."""
    for camera in system_config.cameras:
        if camera.id in (CameraId.Left, CameraId.Right):
            camera.params = {**camera.params, **params}
    system_config.save_default(trainer_config_dir)


def _entry_of(app_model, legacy_or_format2_folder: Path) -> Path:
    """The library entry the app imported `legacy_or_format2_folder` as, which it then loads."""
    root = library.library_root(Path(app_model._preferences.configuration_location))
    entry = root / library.active_id(root)
    assert entry.is_dir() and legacy_or_format2_folder.is_dir()
    return entry


def _calibration_errors(on_error):
    return [call.args[1] for call in on_error.call_args_list if call.args[0] == "3D calibration"]


def _assert_live_matches_reference(algorithm, seed=0):
    """Live 3D runs and agrees with the reference on one confident frame per camera."""
    count = len(algorithm._measure_offset_parts)
    data = np.random.default_rng(seed).uniform(20.0, 230.0, size=(2, 1, count, 3))
    data[:, :, :, 2] = 1.0
    detection = [data[0], data[1]]
    live_raw, live_3d = algorithm._compute_3d_locations(detection)
    ref_raw, ref_3d = algorithm._reference_3d_locations(detection)
    assert live_3d and sorted(map(str, live_3d)) == sorted(map(str, ref_3d))
    for live, ref in ((live_raw, ref_raw), (live_3d, ref_3d)):
        for part, value in live.items():
            assert tuple(value) == pytest.approx(tuple(ref[part]), rel=1e-9)


def test_reload_selects_the_base_preset(app_model, tmp_path):
    # A base other than SpinCam's defaults, so the preset can only come from the configuration,
    # and the 1024 preset written first, so taking the first preset would be caught.
    base = PresetGeometry(4, 256, 256, 40, 10)
    offsets = {"x_off": 1.5, "y_off": -2.0, "z_off": 140.0}
    folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME, base, binnings=(1, 2, 4),
                             offsets=offsets)
    _configure_live_pair(app_model, hbin=4, vbin=4, width=256, height=256, offset_x=40, offset_y=10)
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(folder)
    on_error.assert_not_called()

    algorithm = app_model._pose_algorithm
    expected = store.load_preset(folder, base)
    for key in ("cameraMatrix1", "cameraMatrix2", "R1", "R2", "P1", "P2"):
        np.testing.assert_array_equal(algorithm.stereo_params.matrix[key],
                                      expected.stereo_params.matrix[key])
    assert algorithm.stereo_params.matrix["image_shape"] == [(256, 256), (256, 256)]
    np.testing.assert_array_equal(algorithm.frame_transform, expected.frame_transform)
    assert algorithm.cam_names == ["left", "right"]
    assert algorithm.cam_offsets == offsets
    # The angles and the square size belong to legacy calibrations only.
    assert algorithm.calib_metadata is None and algorithm.square_size is None
    assert algorithm._live_3d_reason() is None
    assert app_model.active_calibration_dir == folder
    assert app_model.active_calibration_index == store.load_index(folder)
    algorithm.initialize(PARTS)
    _assert_live_matches_reference(algorithm)


def test_the_default_is_the_library_s_active_entry_imported_on_first_load(app_model, user_pref):
    # A rig from before the library: its 4mm_6r_8c_4x folder is imported and made active.
    folder = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    _configure_live_pair(app_model, **BASE_PARAMS)
    app_model.reload_calib(None)
    entry = _entry_of(app_model, folder)
    assert entry.name == store.load_index(folder).id
    assert app_model.active_calibration_dir == entry
    assert app_model._pose_algorithm.frame_transform is not None
    # The old folder is only read.
    assert store.load_index(folder) == store.load_index(entry)


def test_geometry_mismatch_disables_3d_with_message(app_model, tmp_path):
    folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME)
    # The operator moved the ROI: no preset of this calibration has these pixels.
    _configure_live_pair(app_model, **{**BASE_PARAMS, "offsetx": 0, "offsety": 0})
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(folder)
    on_error.assert_called_once()
    title, message = on_error.call_args.args
    assert title == "3D calibration"
    assert "256x256_bin4" in message and "(0, 0)" in message and "(52, 6)" in message

    algorithm = app_model._pose_algorithm
    # Never substituted: no stereo parameters at all, so live 3D says why it cannot run.
    assert algorithm.stereo_params is None and algorithm.frame_transform is None
    assert algorithm._live_3d_reason() == "no stereo params"
    assert app_model.active_calibration_dir == folder
    assert app_model.active_calibration_index == store.load_index(folder)
    # 2D pose keeps working.
    algorithm.initialize(PARTS)
    frame = np.column_stack([np.linspace(20.0, 230.0, len(PARTS)),
                             np.linspace(30.0, 200.0, len(PARTS)), np.ones(len(PARTS))])
    with pytest.warns(UserWarning, match="no stereo params"):
        response = algorithm.process_frames([frame], [frame], pairs_3d_offsets=())
    assert set(PARTS) <= set(response.locations[0]) and response.locations_3d == {}


def test_unequal_binning_disables_3d_with_message(app_model, tmp_path):
    folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME)
    _configure_live_pair(app_model, **{**BASE_PARAMS, "vbin": 2})
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(folder)
    on_error.assert_called_once()
    title, message = on_error.call_args.args
    assert title == "3D calibration" and "hbin=4 vbin=2" in message
    assert app_model._pose_algorithm._live_3d_reason() == "no stereo params"


def test_a_configuration_load_reloads_with_the_configured_geometry(
        app_model, tmp_path, system_config, trainer_config_dir):
    base = PresetGeometry(4, 256, 256, 40, 10)
    folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME, base)
    # Before any configuration the left camera is at SpinCam's default ROI, for which
    # this calibration has no preset.
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(folder)
    assert app_model._pose_algorithm.stereo_params is None
    _save_live_pair_params(system_config, trainer_config_dir, hbin=4, vbin=4, width=256,
                             height=256, offsetx=40, offsety=10, fps=150)
    with mock.patch.object(app_model, "on_error") as on_error:
        assert app_model.load_configuration() is True
    assert _calibration_errors(on_error) == []

    algorithm = app_model._pose_algorithm
    np.testing.assert_array_equal(algorithm.frame_transform, store.load_preset(folder, base).frame_transform)
    assert algorithm._live_3d_reason() is None
    # The folder last loaded is kept, not the default, and the reload keeps the frame rate.
    assert app_model.active_calibration_dir == folder
    assert algorithm.frame_rate == 150


def test_a_configured_roi_with_no_preset_disables_3d_on_load(
        app_model, tmp_path, system_config, trainer_config_dir):
    folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME)
    app_model.reload_calib(folder)
    assert app_model._pose_algorithm._live_3d_reason() is None, "SpinCam's defaults are the base"
    # The operator's configuration moves the ROI to an offset no preset has.
    _save_live_pair_params(system_config, trainer_config_dir, hbin=4, vbin=4, width=256,
                             height=256, offsetx=60, offsety=6)
    with mock.patch.object(app_model, "on_error") as on_error:
        assert app_model.load_configuration() is True
    message, = _calibration_errors(on_error)
    assert "256x256_bin4 at ROI offset (60, 6)" in message
    assert app_model._pose_algorithm.stereo_params is None
    assert app_model._pose_algorithm._live_3d_reason() == "no stereo params"
    assert app_model.active_calibration_dir == folder


@pytest.mark.parametrize("kind", ["truncated legacy pickle", "missing preset file"])
def test_a_broken_calibration_never_fails_a_configuration_load(
        app_model, calib_dir, tmp_path, system_config, trainer_config_dir, kind):
    if kind == "truncated legacy pickle":
        folder = _legacy_copy_without_offsets(calib_dir, tmp_path)
        broken = folder / "camera_matrix" / "stereo_params.pickle"
    else:
        # calibration.yaml stays valid and the configured geometry has a preset, so
        # neither of the failures reload_calib reports itself applies.
        folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME)
        _save_live_pair_params(system_config, trainer_config_dir, **BASE_PARAMS)
        broken = folder / "presets" / BASE.name / "camera_matrix" / "stereo_params.pickle"
    app_model.reload_calib(folder)
    assert app_model._pose_algorithm._live_3d_reason() is None, "3D ran before the damage"
    if kind == "truncated legacy pickle":
        broken.write_bytes(broken.read_bytes()[:64])
    else:
        broken.unlink()

    with mock.patch.object(app_model, "on_error") as on_error:
        assert app_model.load_configuration() is True
    assert app_model._configuration_load_incomplete is False
    message, = _calibration_errors(on_error)
    assert str(folder) in message
    # Off, not left running on the calibration loaded before the damage.
    algorithm = app_model._pose_algorithm
    assert algorithm.stereo_params is None and algorithm._live_3d_reason() == "no stereo params"
    assert app_model.active_calibration_dir == folder
    # A format-2 index that still reads is kept, so the session metadata names the calibration.
    assert app_model.active_calibration_index == (
        None if kind == "truncated legacy pickle" else store.load_index(folder))
    # An explicit reload, as DCS and the 3D calibration make, still raises to its caller.
    with pytest.raises((OSError, EOFError, pickle.UnpicklingError)):
        app_model.reload_calib(folder)


def test_an_explicit_reload_that_fails_leaves_the_next_configuration_load_retrying_its_folder(
        app_model, tmp_path, system_config, trainer_config_dir):
    # As after an install whose reload fails partway: the folder that was active (the legacy
    # fixture) is not the one reloaded, and the reload cannot read a preset file. The next
    # configuration load must try the installed folder, not the one before it.
    _save_live_pair_params(system_config, trainer_config_dir, **BASE_PARAMS)
    assert app_model.load_configuration() is True
    previous = app_model.active_calibration_dir
    folder = _write_multires(tmp_path / "installed" / DEFAULT_3D_CALIB_DIR_NAME)
    assert previous is not None and previous != folder
    preset_file = folder / "presets" / BASE.name / "camera_matrix" / "stereo_params.pickle"
    kept = preset_file.read_bytes()
    preset_file.unlink()

    with pytest.raises(OSError):
        app_model.reload_calib(folder)
    assert app_model.active_calibration_dir == folder

    preset_file.write_bytes(kept)
    with mock.patch.object(app_model, "on_error") as on_error:
        assert app_model.load_configuration() is True
    assert _calibration_errors(on_error) == []
    assert app_model.active_calibration_index == store.load_index(folder)
    assert app_model._pose_algorithm.stereo_params is not None


@pytest.mark.parametrize("index_text", ["format: [2\n", "format: 2\nid: abc\n"],
                         ids=["not yaml", "missing fields"])
def test_an_unreadable_index_disables_3d_instead_of_loading_legacy(app_model, calib_dir, tmp_path, index_text):
    # The legacy files are there too, so falling through to the legacy loader would succeed.
    folder = _legacy_copy_without_offsets(calib_dir, tmp_path)
    (folder / "calibration.yaml").write_text(index_text)
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(folder)
    on_error.assert_called_once_with("3D calibration", f"calibration.yaml unreadable: {folder}")
    algorithm = app_model._pose_algorithm
    assert algorithm.stereo_params is None and algorithm._live_3d_reason() == "no stereo params"
    assert app_model.active_calibration_dir == folder
    assert app_model.active_calibration_index is None


@pytest.mark.parametrize("kind", ["legacy", "format 2"])
def test_missing_offsets_file_uses_zero_offsets(app_model, calib_dir, tmp_path, caplog, kind):
    if kind == "legacy":
        folder = _legacy_copy_without_offsets(calib_dir, tmp_path)
    else:
        folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME)
        _configure_live_pair(app_model, **BASE_PARAMS)
    assert not (folder / DEFAULT_CAM_OFFSET_FILE_NAME).exists()
    with caplog.at_level(logging.WARNING):
        app_model.reload_calib(folder)

    algorithm = app_model._pose_algorithm
    assert algorithm.cam_offsets == ZERO_OFFSETS
    assert any(record.levelno == logging.WARNING and DEFAULT_CAM_OFFSET_FILE_NAME in record.getMessage()
               for record in caplog.records)
    assert algorithm._live_3d_reason() is None
    assert app_model.active_calibration_dir == folder
    assert (app_model.active_calibration_index is None) == (kind == "legacy")
    algorithm.initialize(PARTS)
    _assert_live_matches_reference(algorithm)


def test_a_legacy_calibration_loads_as_before(app_model, calib_dir):
    # The app_model fixture loaded the legacy fixture calibration when it was built.
    algorithm = app_model._pose_algorithm
    with (calib_dir / DEFAULT_CAM_OFFSET_FILE_NAME).open("rb") as fh:
        assert algorithm.cam_offsets == pickle.load(fh)
    with (calib_dir / "calibration_userset.yaml").open() as fh:
        assert algorithm.calib_metadata == yaml.safe_load(fh)
    assert algorithm.square_size == 4
    assert algorithm.cam_names == ["20241029_agx001_session002_left", "20241029_agx001_session002_right"]
    assert algorithm.stereo_params is not None and algorithm.frame_transform is None
    assert algorithm._live_3d_reason() is None
    assert app_model.active_calibration_dir == calib_dir
    assert app_model.active_calibration_index is None


@pytest.fixture
def start_app(mock_system, user_pref, system_config, monkeypatch, fake_system_msg_handler, caplog):
    """Start an AppModel as the application does: no calib_dir, so the default folder, and
    no camera configuration loaded yet. Returns it with the "3D calibration" errors raised
    while it was built and that build's log records.

    on_error cannot be patched on an instance that does not exist yet, so the errors are
    read from the log line AppModel's own first on_error handler writes for each one.
    """
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    apps = []

    def start():
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            app = AppModel(
                user_pref,
                system_machine=mock_system.system_machine,
                analysis=mock_system.sensor_analysis,
                inference_model=mock_system.inference,
                system_message_handler=fake_system_msg_handler,
            )
        apps.append(app)
        records = list(caplog.records)
        errors = [r.getMessage() for r in records
                  if r.levelno == logging.ERROR and r.getMessage().startswith("3D calibration: ")]
        return app, errors, records

    try:
        yield start
    finally:
        for app in apps:
            unregister_fatal_exception_callback(app._on_fatal_exception)
            top_fixtures.release_multiprocessing_resources(app)


def test_startup_loads_the_base_preset_from_the_default_folder(start_app, user_pref):
    folder = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    app, errors, _ = start_app()
    assert errors == []
    # No camera configuration yet, so the geometry is SpinCam's default capture: the base.
    assert app.left_camera.active_config.params == {}
    algorithm = app._pose_algorithm
    np.testing.assert_array_equal(algorithm.frame_transform, store.load_preset(folder, BASE).frame_transform)
    assert algorithm.cam_offsets == ZERO_OFFSETS
    assert algorithm._live_3d_reason() is None
    assert app.active_calibration_dir == _entry_of(app, folder)
    assert app.active_calibration_index == store.load_index(folder)


def test_a_corrupt_default_calibration_never_stops_the_app_starting(start_app, user_pref, calib_dir, tmp_path):
    folder = _legacy_copy_without_offsets(
        calib_dir, tmp_path, Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    pickle_path = folder / "camera_matrix" / "stereo_params.pickle"
    pickle_path.write_bytes(pickle_path.read_bytes()[:64])

    app, errors, records = start_app()
    # Imported as it was, damage included, and loaded from the library.
    entry = _entry_of(app, folder)
    assert errors == [], "construction told the operator; only the configuration load should"
    assert any(r.exc_info and str(entry) in r.getMessage() for r in records), "no traceback logged"
    assert app._pose_algorithm.stereo_params is None
    assert app._pose_algorithm._live_3d_reason() == "no stereo params"
    # The entry stays active, so the configuration load tries it again and reports it.
    assert app.active_calibration_dir == entry
    with mock.patch.object(app, "on_error") as on_error:
        assert app.load_configuration() is True
    message, = _calibration_errors(on_error)
    assert str(entry) in message


# A base away from SpinCam's default ROI, so the geometry the constructor sees (the
# defaults) has no preset in it, as on a rig whose configured ROI is not the default.
_OFF_DEFAULT_BASE = PresetGeometry(4, 256, 256, 40, 10)


def test_a_roi_with_no_preset_is_reported_once_after_the_configuration_loads(
        start_app, user_pref, system_config, trainer_config_dir):
    _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME, _OFF_DEFAULT_BASE)
    _save_live_pair_params(system_config, trainer_config_dir, hbin=4, vbin=4, width=256,
                             height=256, offsetx=60, offsety=6)
    app, errors, records = start_app()
    assert errors == []
    # Logged, though: the defaults matched no preset.
    assert any(r.levelno == logging.WARNING and "(52, 6)" in r.getMessage() for r in records)
    with mock.patch.object(app, "on_error") as on_error:
        assert app.load_configuration() is True
    message, = _calibration_errors(on_error)
    assert "(60, 6)" in message
    assert app._pose_algorithm._live_3d_reason() == "no stereo params"


def test_a_matching_configuration_is_never_reported(start_app, user_pref, system_config, trainer_config_dir):
    folder = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME,
                             _OFF_DEFAULT_BASE)
    _save_live_pair_params(system_config, trainer_config_dir, hbin=4, vbin=4, width=256,
                             height=256, offsetx=40, offsety=10)
    app, errors, _ = start_app()
    # The constructor's defaults match no preset of this calibration, which is not the
    # operator's problem: the configured ROI does.
    assert errors == []
    with mock.patch.object(app, "on_error") as on_error:
        assert app.load_configuration() is True
    assert _calibration_errors(on_error) == []
    algorithm = app._pose_algorithm
    assert algorithm._live_3d_reason() is None
    np.testing.assert_array_equal(algorithm.frame_transform,
                                  store.load_preset(folder, _OFF_DEFAULT_BASE).frame_transform)


def test_no_calibration_folder_leaves_3d_off(app_model, tmp_path):
    app_model.reload_calib(tmp_path / "missing" / DEFAULT_3D_CALIB_DIR_NAME)
    assert app_model._pose_algorithm._live_3d_reason() == "no stereo params"
    assert app_model.active_calibration_dir is None
    assert app_model.active_calibration_index is None


# Cameras are matched to the active calibration by serial, CameraConfiguration.host of a
# Spinnaker camera: a renamed camera keeps its calibration, a swapped body loses it.
SERIALS = {"left": "21000001", "right": "21000002"}


def _set_serial(camera, serial: str) -> None:
    camera.active_config.scheme = "spinnaker"
    camera.active_config.host = serial


def _active_entry(app_model, tmp_path, *, serials=SERIALS, extra=()) -> Path:
    """A format-3 calibration of left and right with `serials`, plus `extra` (name, serial,
    reason) cameras it left 2D only, added to the app's library and made active."""
    built = tmp_path / "built" / DEFAULT_3D_CALIB_DIR_NAME
    cameras = [CaptureCamera("left", serials["left"], 0), CaptureCamera("right", serials["right"], 1)]
    cameras += [CaptureCamera(name, serial, 3 + i) for i, (name, serial, _) in enumerate(extra)]
    report = dataclasses.replace(_REPORT, cameras={name: CameraVerdict(name, False, reason) for name, _, reason in extra})
    built.parent.mkdir(parents=True)
    store.write_calibration(built, fit=FIT, presets=presets_from_base(BASE, (4, 2, 1)), cameras=cameras,
                            board=BoardSpec(), report=report, capture={})
    root = library.library_root(Path(app_model._preferences.configuration_location))
    calibration_id = library.add_entry(root, built)
    library.set_active(root, calibration_id, by="test")
    return root / calibration_id


def _configure_with_serials(app_model, serials) -> None:
    _configure_live_pair(app_model, **BASE_PARAMS)
    for name, serial in serials.items():
        _set_serial(getattr(app_model, f"{name}_camera"), serial)


def test_the_app_loads_the_active_library_entry_matched_by_serial(app_model, tmp_path):
    entry = _active_entry(app_model, tmp_path, extra=[("camera4", "21000004", "too few board views: 4 of 12 needed")])
    _configure_with_serials(app_model, SERIALS)
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(None)
    assert _calibration_errors(on_error) == []
    assert app_model.active_calibration_dir == entry
    assert app_model.calibrated_cameras == {"left": "left", "right": "right"}
    assert "left" not in app_model.uncalibrated_reasons and "right" not in app_model.uncalibrated_reasons
    algorithm = app_model._pose_algorithm
    assert algorithm._live_3d_reason() is None
    np.testing.assert_array_equal(algorithm.frame_transform, store.load_preset(entry, BASE).frame_transform)
    # camera4 is the calibration's 2D-only camera and is not configured: no difference.
    assert app_model.camera_setup_difference(["left", "right"]) == SetupDifference([], [], [], [])


def test_swapped_serial_is_a_difference(app_model, tmp_path):
    _active_entry(app_model, tmp_path)
    _configure_with_serials(app_model, {"left": "21000099", "right": SERIALS["right"]})
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(None)
    reason = "swapped: calibrated serial 21000001, now 21000099"
    assert app_model.uncalibrated_reasons["left"] == reason
    assert app_model.calibrated_cameras == {"right": "right"}
    difference = app_model.camera_setup_difference(["left", "right"])
    assert difference == SetupDifference(added=[], removed=[], swapped=["left"], uncalibrated_off=[])
    assert difference.differs
    # Live 3D does not triangulate with the old pose for the new body.
    message, = _calibration_errors(on_error)
    assert reason in message
    assert app_model._pose_algorithm.stereo_params is None
    assert app_model._live_3d_disabled_reason == message


def test_renamed_camera_with_same_serial_is_not_a_difference(app_model, tmp_path):
    # The two bodies' configured names are exchanged: each is still its calibrated self.
    _active_entry(app_model, tmp_path)
    _configure_with_serials(app_model, {"left": SERIALS["right"], "right": SERIALS["left"]})
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(None)
    assert app_model.calibrated_cameras == {"left": "right", "right": "left"}
    difference = app_model.camera_setup_difference(["left", "right"])
    assert difference == SetupDifference([], [], [], []) and not difference.differs
    # Today's live 3D needs the calibration's own left and right as left and right.
    assert "left: it is the calibration's right" in app_model._live_3d_disabled_reason


def test_subset_is_only_a_passive_note(app_model, tmp_path):
    _active_entry(app_model, tmp_path)
    _configure_with_serials(app_model, SERIALS)
    app_model.reload_calib(None)
    difference = app_model.camera_setup_difference(["left"])
    assert difference == SetupDifference(added=[], removed=[], swapped=[], uncalibrated_off=["right"])
    assert not difference.differs


def test_an_independent_stim_camera_is_never_added(app_model, tmp_path):
    # stimCam in stimulation mode is never 3D-posed, so recording it against a calibration
    # without it needs no confirmation. It is still 2D only, and says so. The fixture
    # configures left and right only, so right's body stands in for the stimCam.
    _active_entry(app_model, tmp_path)
    _configure_with_serials(app_model, {"left": SERIALS["left"], "right": "21000003"})
    stim = app_model.right_camera
    stim._name = "stimCam"
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(None)
    selected = ["left", "stimCam"]
    assert app_model.camera_setup_difference(selected).added == ["stimCam"]
    with mock.patch.object(AppModel, "_is_independent_stim_camera", side_effect=lambda camera: camera is stim):
        difference = app_model.camera_setup_difference(selected)
    assert difference == SetupDifference(added=[], removed=["right"], swapped=[], uncalibrated_off=[])
    assert app_model.uncalibrated_reasons["stimCam"] == "not in the active calibration"


def test_a_calibrated_camera_no_longer_configured_is_removed(app_model, tmp_path):
    _active_entry(app_model, tmp_path)
    # right's body is replaced, and right is not selected: its calibrated body is gone.
    _configure_with_serials(app_model, {"left": SERIALS["left"], "right": "21000077"})
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(None)
    difference = app_model.camera_setup_difference(["left"])
    assert difference == SetupDifference(added=[], removed=["right"], swapped=[], uncalibrated_off=[])


def test_the_preset_needs_left_and_right_at_one_geometry(app_model, tmp_path):
    folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME)
    _configure_live_pair(app_model, **BASE_PARAMS)
    app_model.right_camera.load_configuration(CameraConfiguration(
        id=CameraId.Right, name="right", scheme="random", params={**BASE_PARAMS, "offsetx": 40}))
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(folder)
    message, = _calibration_errors(on_error)
    assert "same geometry" in message and "(52, 6)" in message and "(40, 6)" in message
    assert app_model._pose_algorithm.stereo_params is None


def test_dcs_offsets_for_a_library_entry_go_beside_it(app_model, tmp_path):
    entry = _active_entry(app_model, tmp_path)
    _configure_with_serials(app_model, SERIALS)
    app_model.reload_calib(None)
    path = app_model.camera_offsets_path()
    root = library.library_root(Path(app_model._preferences.configuration_location))
    assert path == root / ".offsets" / entry.name / DEFAULT_CAM_OFFSET_FILE_NAME
    offsets = {"x_off": 1.0, "y_off": 2.0, "z_off": 3.0}
    path.parent.mkdir(parents=True)
    with path.open("wb") as fh:
        pickle.dump(offsets, fh)
    app_model.reload_calib(entry)
    assert app_model._pose_algorithm.cam_offsets == offsets
    assert not (entry / DEFAULT_CAM_OFFSET_FILE_NAME).exists()
    # Not an entry, so never listed as one.
    assert [e.id for e in library.list_entries(root)] == [entry.name]


def test_a_library_with_entries_and_no_pointer_is_not_imported_into_again(app_model, user_pref, tmp_path):
    # A rig whose active.yaml went missing: an old folder is still there, and importing it
    # again would make its calibration active behind the operator's back.
    entry = _active_entry(app_model, tmp_path)
    old = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    root = entry.parent
    (root / library.ACTIVE_FILE).unlink()
    _configure_with_serials(app_model, SERIALS)

    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(None)

    assert app_model.active_calibration_dir is None
    assert app_model._live_3d_disabled_reason == f"no active calibration in {root}: active.yaml is missing"
    assert app_model._pose_algorithm.stereo_params is None, "2D only"
    assert not (root / library.ACTIVE_FILE).exists()
    assert [e.id for e in library.list_entries(root)] == [entry.name], "the old folder was not imported"
    assert store.load_index(old).id not in [e.id for e in library.list_entries(root)]


def test_a_calibration_that_fails_to_load_puts_no_camera_in_it(app_model, tmp_path):
    from calibration_metadata_test import _saved_calibration3d

    entry = _active_entry(app_model, tmp_path)
    _configure_with_serials(app_model, SERIALS)
    # Entries never change; this stands in for a disk error under one of its presets.
    (entry / "presets" / BASE.name / "camera_matrix" / "stereo_params.pickle").unlink()
    for camera in (app_model.left_camera, app_model.right_camera):
        camera.is_recording_enabled = True

    with pytest.raises(OSError):
        app_model.reload_calib(None)

    reason = app_model.uncalibrated_reasons["left"]
    assert reason.startswith(f"the active calibration {entry.name} could not be read: FileNotFoundError")
    assert app_model.uncalibrated_reasons == {name: reason for name in app_model.uncalibrated_reasons}
    assert app_model.calibrated_cameras == {}
    calibration3d = _saved_calibration3d(app_model, tmp_path)
    assert calibration3d["id"] == entry.name and calibration3d["matchesRecording"] is False
    saved = (tmp_path / "session003" / "metadata.json").read_text()
    cameras = json.loads(saved)["hardware"]["configured"]["cameras"]
    assert {name: (c["inCalibration"], c["uncalibratedReason"]) for name, c in cameras.items()
            if name in ("left", "right")} == {"left": (False, reason), "right": (False, reason)}


def test_an_unreadable_active_entry_says_so_for_every_camera(app_model, tmp_path):
    entry = _active_entry(app_model, tmp_path)
    _configure_with_serials(app_model, SERIALS)
    (entry / store.INDEX_FILE).write_text("format: 3\nid: [half-written\n")

    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(entry)

    reason = app_model.uncalibrated_reasons["left"]
    assert reason.startswith(f"the active calibration {entry.name} could not be read: ")
    assert "not valid YAML" in reason
    assert app_model.calibrated_cameras == {}


def test_dcs_is_refused_without_an_active_calibration(app_model, user_pref):
    root = library.library_root(Path(user_pref.configuration_location))
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(None)
    assert app_model.active_calibration_dir is None
    assert app_model.dcs_refusal() == (
        "Calibrate Coordinate System needs an active 3D calibration, and there is none:"
        " make or activate a 3D calibration first")
    assert not (Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME).exists()
    assert not root.joinpath(".offsets").exists()
