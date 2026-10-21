"""reload_calib loads the preset of a format-2 calibration that matches the live frames' pixels.

Live frames reach inference pooled to the base capture, so the left camera's
configured base geometry picks the preset. A calibration without that preset turns
live 3D off with the reason shown to the operator; another preset is never
substituted, since its pixels are not the frames' pixels. A legacy calibration
loads as before. A missing camera_offsets.pkl, which a fresh calibration has until
DCS is redone, gives zero offsets instead of a crash.
"""

import logging
import pickle
import shutil
from pathlib import Path
from unittest import mock

import cv2
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
from autotrainer.inference.multires_calibration.fit import (
    INTRINSIC_NAMES,
    BoardSpec,
    CameraModel,
    MotorFrameFit,
)
from autotrainer.inference.multires_calibration.geometry import PresetGeometry, presets_from_base
from autotrainer.inference.multires_calibration.report import QualityReport
from tools.acquisition.model.app_model import AppModel


BASE = PresetGeometry(4, 256, 256, 52, 6)
BASE_PARAMS = dict(hbin=4, vbin=4, width=256, height=256, offsetx=52, offsety=6)
PARTS = ['Pellet', 'RH_flat', 'RH_spread', 'RH_grab', 'LH_flat', 'LH_spread', 'LH_grab',
         'Star', 'Tongue_mid', 'Tongue_tip', 'Nose', 'Triangle', 'Mouth', 'Diamond']
ZERO_OFFSETS = {"x_off": 0.0, "y_off": 0.0, "z_off": 0.0}
_REPORT = QualityReport(created="2026-10-20T12:00:00-06:00", verdict="pass", metrics=[],
                        thresholds={}, notes=[])


def _fit() -> MotorFrameFit:
    """A plausible motor-frame fit in 1440x1080 sensor px: two cameras 40 degrees apart,
    150 mm from the motor origin and looking at it."""
    cameras = {}
    for cam, side in (("left", -1.0), ("right", 1.0)):
        a = np.radians(20.0)
        eye = 150.0 * np.array([side * np.sin(a), 0.0, -np.cos(a)])
        z = -eye / np.linalg.norm(eye)
        x = np.cross([0.0, 1.0, 0.0], z)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])  # rows are the camera axes in the motor frame
        cameras[cam] = CameraModel(
            K=np.array([[2300.0, 0.0, 719.5], [0.0, 2300.0, 539.5], [0.0, 0.0, 1.0]]),
            dist=np.zeros(5),
            rvec=cv2.Rodrigues(R)[0].ravel(),
            tvec=-R @ eye,
        )
    return MotorFrameFit(
        cameras=cameras,
        board_rvec=np.zeros(3),
        board_tvec=np.zeros(3),
        motor_scale=1.0,
        image_size=(1440, 1080),
        rms_px={cam: 0.1 for cam in cameras},
        per_view_rms_px={cam: {0: 0.1} for cam in cameras},
        param_std={cam: {name: 0.0 for name in INTRINSIC_NAMES} for cam in cameras},
    )


_FIT = _fit()


@pytest.fixture
def app_model(app_model):
    # reload_calib reads the model location and pose parts off the inference model. The
    # conftest's VoidInference declares neither: AppModel's constructor reloads before it
    # has an inference model, so only a later reload, as these tests make, reaches them.
    # Plain assignment, since reading an undeclared name off it raises (so monkeypatch
    # cannot record a previous value); the inference double is made for each test.
    app_model.inference.model_location = None
    app_model.inference.pose_parts = PARTS
    return app_model


def _write_multires(folder: Path, base=BASE, *, binnings=(4, 2, 1), offsets=None) -> Path:
    folder.parent.mkdir(parents=True, exist_ok=True)
    store.write_calibration(folder, fit=_FIT, presets=presets_from_base(base, binnings),
                            board=BoardSpec(), report=_REPORT, capture={})
    if offsets is not None:
        with (folder / DEFAULT_CAM_OFFSET_FILE_NAME).open("wb") as fh:
            pickle.dump(offsets, fh)
    return folder


def _legacy_copy_without_offsets(calib_dir: Path, tmp_path: Path) -> Path:
    folder = tmp_path / "legacy" / DEFAULT_3D_CALIB_DIR_NAME
    (folder / "camera_matrix").mkdir(parents=True)
    for name in ("camera_matrix/stereo_params.pickle", "calibration_userset.yaml"):
        shutil.copyfile(calib_dir / name, folder / name)
    # The camera names come from the source videos' file names alone.
    (folder / "source_videos").mkdir()
    for video in (calib_dir / "source_videos").iterdir():
        (folder / "source_videos" / video.name).touch()
    return folder


def _configure_left(app_model, **params):
    app_model.left_camera.load_configuration(
        CameraConfiguration(id=CameraId.Left, name="left", scheme="random", params=params))


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
    _configure_left(app_model, hbin=4, vbin=4, width=256, height=256, offset_x=40, offset_y=10)
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


def test_the_default_folder_is_in_the_configuration_location(app_model, user_pref):
    folder = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    _configure_left(app_model, **BASE_PARAMS)
    app_model.reload_calib(None)
    assert app_model.active_calibration_dir == folder
    assert app_model._pose_algorithm.frame_transform is not None


def test_geometry_mismatch_disables_3d_with_message(app_model, tmp_path):
    folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME)
    # The operator moved the ROI: no preset of this calibration has these pixels.
    _configure_left(app_model, **{**BASE_PARAMS, "offsetx": 0, "offsety": 0})
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
    _configure_left(app_model, **{**BASE_PARAMS, "vbin": 2})
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(folder)
    on_error.assert_called_once()
    title, message = on_error.call_args.args
    assert title == "3D calibration" and "hbin=4 vbin=2" in message
    assert app_model._pose_algorithm._live_3d_reason() == "no stereo params"


@pytest.mark.parametrize("kind", ["legacy", "format 2"])
def test_missing_offsets_file_uses_zero_offsets(app_model, calib_dir, tmp_path, caplog, kind):
    if kind == "legacy":
        folder = _legacy_copy_without_offsets(calib_dir, tmp_path)
    else:
        folder = _write_multires(tmp_path / DEFAULT_3D_CALIB_DIR_NAME)
        _configure_left(app_model, **BASE_PARAMS)
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
def started_on_multires(mock_system, user_pref, system_config, monkeypatch, fake_system_msg_handler):
    """An AppModel started with no calib_dir, a format-2 calibration in its default folder,
    and, as at application start, no camera configuration loaded yet."""
    folder = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    app = AppModel(
        user_pref,
        system_machine=mock_system.system_machine,
        analysis=mock_system.sensor_analysis,
        inference_model=mock_system.inference,
        system_message_handler=fake_system_msg_handler,
    )
    try:
        yield app, folder
    finally:
        unregister_fatal_exception_callback(app._on_fatal_exception)
        top_fixtures.release_multiprocessing_resources(app)


def test_startup_loads_the_base_preset_from_the_default_folder(started_on_multires):
    app, folder = started_on_multires
    # No camera configuration yet, so the geometry is SpinCam's default capture: the base.
    assert app.left_camera.active_config.params == {}
    algorithm = app._pose_algorithm
    np.testing.assert_array_equal(algorithm.frame_transform, store.load_preset(folder, BASE).frame_transform)
    assert algorithm.cam_offsets == ZERO_OFFSETS
    assert algorithm._live_3d_reason() is None
    assert app.active_calibration_dir == folder
    assert app.active_calibration_index == store.load_index(folder)


def test_no_calibration_folder_leaves_3d_off(app_model, tmp_path):
    app_model.reload_calib(tmp_path / "missing" / DEFAULT_3D_CALIB_DIR_NAME)
    assert app_model._pose_algorithm._live_3d_reason() == "no stereo params"
    assert app_model.active_calibration_dir is None
    assert app_model.active_calibration_index is None
