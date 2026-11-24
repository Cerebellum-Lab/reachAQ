"""A new 3D calibration makes the DCS configuration stale until DCS is redone.

DCS (diamond_triangle_offset.yaml and the calibration's camera_offsets.pkl) was measured
in the oriented frame of the calibration that was active then. A format-2 calibration has
an id, and DCS records it; when the active calibration's id differs, DCS is stale. Its
consumers then see no configuration at all, Record is blocked with the reason, and the
operator is told once. Redoing DCS against the active calibration makes it valid again.
A legacy calibration has no id, so a DCS without one fits it, as before; a DCS that records
an id was measured in a format-2 calibration's frame, so it is stale under a legacy one.
"""

import importlib
import json
import pickle
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml
from PySide6.QtWidgets import QMessageBox

from reachaq.core import AnimalSubject, Offset3DTuple, ProjectInfo
from reachaq.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from reachaq.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from reachaq.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME
from reachaq.inference.multires_calibration import library, store
from reachaq.inference.multires_calibration.fit import BoardSpec
from reachaq.inference.multires_calibration.store import CalibrationIndex, CalibrationResolutionError
from reload_calib_multires_test import (
    BASE,
    BASE_PARAMS,
    SERIALS,
    _active_entry,
    _configure_with_serials,
    _legacy_copy_without_offsets,
    _save_live_pair_params,
    _write_multires,
)
from reachaq.app.model.app_model import AppModel
from reachaq.app.model.calibration_install import install_calibration
from reachaq.app.model.dcs_state import DcsState, dcs_state
from reachaq.app.view import main_window as main_window_module
from reachaq.app.view.main_window import MainWindow


STALE_BLOCKER = (
    "Coordinate system out of date: it was measured against a different 3D calibration. "
    "Redo Tools → Calibrate Coordinate System"
)
STALE_NOTICE = (
    "Coordinate system out of date: it was measured against a different 3D calibration. "
    "Redo Tools → Calibrate Coordinate System before running sessions."
)
_ID = "0123456789abcdef0123456789abcdef"


def _config(calibration_id=None, **overrides) -> DiamondTriangleOffsetConfig:
    values = dict(
        used_position=Offset3DTuple(7.0, 4.5, 6.5),
        measured_offset=Offset3DTuple(4.7, -34.0, -8.4),
        diamond_coord=Offset3DTuple(0, 0, 0),
        raw_diamond_coord=Offset3DTuple(3.6, -19.3, 0.7),
        version=DiamondTriangleOffsetConfig.current_config_version,
        calibration_id=calibration_id,
    )
    values.update(overrides)
    return DiamondTriangleOffsetConfig(**values)


def _index(calibration_id=_ID) -> CalibrationIndex:
    return CalibrationIndex(id=calibration_id, created=datetime(2026, 10, 20, 12, 0, 0), board=BoardSpec(),
                            presets=(BASE,), verdict="pass", override=None, motor_scale=1.0, capture={})


@pytest.mark.parametrize("cfg, index, expected", [
    # A legacy active calibration has no id: a DCS without one fits it, as before ids. One
    # with an id was measured in a format-2 motor frame, a different frame (after a rollback
    # to a legacy backup, say), so it is stale.
    (_config(), None, DcsState.VALID),
    (_config(calibration_id=_ID), None, DcsState.STALE),
    # A format-2 calibration: only the DCS measured against it.
    (_config(calibration_id=_ID), _index(), DcsState.VALID),
    (_config(calibration_id="f" * 32), _index(), DcsState.STALE),
    (_config(), _index(), DcsState.STALE),
    # No usable DCS at all is missing, never stale, whatever is active.
    (None, None, DcsState.MISSING),
    (None, _index(), DcsState.MISSING),
    (_config(calibration_id=_ID, version=1), _index(), DcsState.MISSING),
    (_config(used_position=Offset3DTuple(float("nan"), 0, 0)), None, DcsState.MISSING),
])
def test_state_table(cfg, index, expected):
    assert dcs_state(cfg, index) is expected


@pytest.fixture
def dcs_path(app_model, tmp_path, monkeypatch) -> Path:
    path = tmp_path / "diamond_triangle_offset.yaml"
    # Never the rig's own file, whichever default the algorithm was built with.
    monkeypatch.setattr(app_model.behavior.algorithm, "_diamond_triangle_offset_config_path", path)
    return path


@pytest.fixture
def stale(app_model, dcs_path, user_pref, system_config, trainer_config_dir):
    """DCS measured with the legacy fixture calibration, then a format-2 calibration goes live.

    Returns (the new calibration's folder, the on_error mock the reload ran under).
    """
    assert app_model.dcs_state is DcsState.MISSING, "nothing is loaded before a configuration load"
    _config().to_file(dcs_path)
    # The left camera at the base geometry, so the new calibration's preset loads without error.
    _save_live_pair_params(system_config, trainer_config_dir, **BASE_PARAMS)
    assert app_model.load_configuration() is True
    assert app_model.dcs_state is DcsState.VALID
    assert app_model.behavior.algorithm.diamond_triangle_config == _config()

    folder = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(folder)
    assert app_model.active_calibration_index is not None
    return folder, on_error


def _titled(on_error, title):
    return [call.args[1] for call in on_error.call_args_list if call.args[0] == title]


def test_stale_blocks_recording_with_reason(app_model, stale):
    assert app_model.dcs_state is DcsState.STALE
    assert STALE_BLOCKER in app_model.recording_blockers


def test_an_id_stamped_dcs_under_a_legacy_calibration_is_stale_and_blocks_recording(
        app_model, dcs_path, system_config):
    # The DCS was measured against a format-2 calibration; then the legacy folder is active
    # again, as after a rollback to its dated backup. The legacy frame is another frame.
    assert app_model.active_calibration_dir is not None and app_model.active_calibration_index is None, \
        "the fixture calibration is legacy"
    _config(calibration_id=_ID).to_file(dcs_path)
    with mock.patch.object(app_model, "on_error") as on_error:
        assert app_model.load_configuration() is True
    assert app_model.active_calibration_index is None
    assert app_model.dcs_state is DcsState.STALE
    assert STALE_BLOCKER in app_model.recording_blockers
    assert app_model.behavior.algorithm.diamond_triangle_config is None
    assert _titled(on_error, "Coordinate system") == [STALE_NOTICE]


def test_an_install_whose_reload_fails_turns_live_3d_off_and_blocks_recording(
        app_model, dcs_path, calib_dir, user_pref, system_config, trainer_config_dir, tmp_path, monkeypatch):
    # christielab10's first install (ruling R56): a legacy calibration is active in the
    # configuration location, and the DCS, measured against it, has no id. The install
    # succeeds, but reloading the installed folder raises. Live 3D must not carry on with the
    # legacy parameters while that folder holds the new calibration: sessions would record no
    # id, and offline would resolve them by time to the new frame.
    location = Path(user_pref.configuration_location)
    legacy = _legacy_copy_without_offsets(calib_dir, tmp_path, location / DEFAULT_3D_CALIB_DIR_NAME)
    app_model.reload_calib(legacy)
    _config().to_file(dcs_path)
    _save_live_pair_params(system_config, trainer_config_dir, **BASE_PARAMS)
    assert app_model.load_configuration() is True
    assert app_model.dcs_state is DcsState.VALID and app_model._pose_algorithm.stereo_params is not None

    new = _write_multires(tmp_path / "session" / f"{DEFAULT_3D_CALIB_DIR_NAME}.new")
    installed = install_calibration(new, location, now=datetime.now())
    # A library entry; the legacy folder from before the library stays where it was.
    assert installed.parent == location / "calibrations" and legacy.is_dir()
    # An I/O fault straight after the copy.
    (installed / "presets" / BASE.name / "camera_matrix" / "stereo_params.pickle").unlink()
    with mock.patch.object(app_model, "on_error"):
        with pytest.raises(OSError):
            app_model.reload_calib(installed)

    index = store.load_index(installed)
    assert app_model._pose_algorithm.stereo_params is None, "live 3D is off"
    assert app_model.active_calibration_index == index
    assert app_model.dcs_state is DcsState.STALE
    assert STALE_BLOCKER in app_model.recording_blockers
    record = app_model._calibration3d_metadata()
    assert record["id"] == index.id and record["preset"] is None
    assert str(installed) in record["live3dDisabledReason"]

    # A session recorded meanwhile names the new calibration, and offline refuses it.
    intersession = importlib.import_module("reachaq.inference.analysis.intersession_process")
    project = ProjectInfo(root=(tmp_path / "Data").as_posix(), device_id="rig", when=datetime.now(), session=1)
    metadata = Path(project.get_metadata_file() + ".json")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps({
        "metadataSchemaVersion": 2,
        "boundary": {"startWallTime": time.time()},
        "hardware": {"configured": {"calibration3d": record}},
    }), encoding="utf-8")

    def not_reached(*args, **kwargs):
        raise AssertionError("offline 3D ran for a session whose live 3D was off")

    monkeypatch.setattr(intersession, "process_raw_data", not_reached)
    with pytest.raises(CalibrationResolutionError, match="live 3D off"):
        intersession.intersession_process(project, calib_root=location)


def test_the_record_button_is_told_when_dcs_goes_stale(app_model, dcs_path, user_pref, system_config,
                                                       trainer_config_dir):
    _config().to_file(dcs_path)
    _save_live_pair_params(system_config, trainer_config_dir, **BASE_PARAMS)
    assert app_model.load_configuration() is True
    published = []
    app_model.property_changed += lambda name, value, _: (
        published.append(value) if name == app_model.Props.RECORDING_BLOCKERS else None)
    folder = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(folder)
    assert published and STALE_BLOCKER in published[-1]


def test_stale_hides_config_from_consumers(app_model, stale):
    folder, on_error = stale
    algo = app_model.behavior.algorithm
    assert algo.diamond_triangle_config is None
    assert algo.loaded_diamond_triangle_config == _config()
    # The consumers were handed None: motor moves in DCS refuse and say why, the pellet
    # monitor has none.
    with pytest.raises(RuntimeError) as refused:
        app_model.hardware.set_dcs_x(1.0)
    assert str(refused.value) == f"DCS move refused: {STALE_BLOCKER}"
    assert app_model.analysis.pellet_misplaced_monitor.dcs_config is None
    # The operator is told once, as it goes stale, and not again while it stays stale.
    assert _titled(on_error, "Coordinate system") == [STALE_NOTICE]
    assert _titled(on_error, "3D calibration") == []
    with mock.patch.object(app_model, "on_error") as again:
        app_model.refresh_dcs_state()
        app_model.reload_calib(folder)
        assert app_model.load_configuration() is True
    again.assert_not_called()
    assert app_model.dcs_state is DcsState.STALE


MISSING_REFUSAL = "DCS config not defined or not fully valid"


def test_a_missing_dcs_refuses_moves_as_before(app_model):
    assert app_model.dcs_state is DcsState.MISSING
    with pytest.raises(RuntimeError, match=MISSING_REFUSAL):
        app_model.hardware.set_dcs_y(1.0)


def test_a_stale_dcs_that_goes_missing_refuses_moves_as_missing(app_model, stale, dcs_path):
    dcs_path.unlink()
    with mock.patch.object(app_model, "on_error"):
        assert app_model.load_configuration() is True
    assert app_model.dcs_state is DcsState.MISSING
    with pytest.raises(RuntimeError, match=MISSING_REFUSAL):
        app_model.hardware.set_dcs_z(1.0)


def test_stale_keeps_the_animals_dcs_position(app_model, stale):
    # Its DCS position means the same place again once DCS is redone: it is kept, and
    # not sent to the motors through a coordinate system that is out of date.
    animal = AnimalSubject(name="m1", pellet_x=1.0, pellet_y=2.0, pellet_z=3.0, is_pellet_dcs=True)
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.selected_animal = animal
    assert (animal.pellet_x, animal.pellet_y, animal.pellet_z, animal.is_pellet_dcs) == (1.0, 2.0, 3.0, True)
    assert _titled(on_error, "Notice") == []


def test_a_configuration_load_holds_a_stale_dcs_before_positioning_the_animal(
        app_model, dcs_path, user_pref, system_config, trainer_config_dir, monkeypatch):
    # The app restarts after a new 3D calibration went live: its first configuration load
    # reads the old DCS file, and then selects and positions the operator's last animal.
    folder = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    _save_live_pair_params(system_config, trainer_config_dir, **BASE_PARAMS)
    app_model.reload_calib(folder)
    assert app_model.dcs_state is DcsState.MISSING, "no DCS loaded yet"
    _config().to_file(dcs_path)
    animal = AnimalSubject(name="m1", pellet_x=1.0, pellet_y=2.0, pellet_z=3.0, is_pellet_dcs=True)
    animal.to_file(Path(user_pref.animal_location) / f"{animal.id}.json")
    user_pref.selected_animal = animal.id
    seen = []
    positions = AppModel._set_animal_base_positions

    def spy(self, subject):
        seen.append(self.behavior.algorithm.diamond_triangle_config)
        return positions(self, subject)

    monkeypatch.setattr(AppModel, "_set_animal_base_positions", spy)
    with mock.patch.object(app_model, "on_error"):
        assert app_model.load_configuration() is True
    assert seen == [None]
    selected = app_model.selected_animal
    assert selected.id == animal.id
    assert (selected.pellet_x, selected.pellet_y, selected.pellet_z, selected.is_pellet_dcs) == (1.0, 2.0, 3.0, True)
    assert app_model.dcs_state is DcsState.STALE


def test_redoing_dcs_clears_stale(app_model, stale, dcs_path):
    index = app_model.active_calibration_index
    _config(calibration_id=index.id).to_file(dcs_path)
    with mock.patch.object(app_model, "on_error") as on_error:
        assert app_model.load_configuration() is True
    assert app_model.dcs_state is DcsState.VALID
    assert STALE_BLOCKER not in app_model.recording_blockers
    assert app_model.behavior.algorithm.diamond_triangle_config == _config(calibration_id=index.id)
    assert _titled(on_error, "Coordinate system") == []


def _run_dcs_calibration(app_model, user_pref, monkeypatch, dcs_path, raw_diamond) -> None:
    """Complete a DCS calibration run through MainWindow's handler, as Calibrate Coordinate System does."""
    shown = []
    # The handler's message box, without a display.
    monkeypatch.setattr(main_window_module, "QMessageBox", SimpleNamespace(
        information=lambda *args, **kwargs: shown.append(args[1:3]),
        StandardButton=QMessageBox.StandardButton,
    ))
    window = SimpleNamespace(
        _app_model=app_model,
        _preferences=user_pref,
        _timer_calibrate_diamond_triangle=mock.Mock(),
        _post_api_event=lambda kind, *args: None,
    )
    # Saved to a new file: the overwrite confirmation and file dialog are not what this tests.
    dcs_path.unlink(missing_ok=True)
    MainWindow._handle_diamond_triangle_calib_run(
        window,
        positions=[Offset3DTuple(7.0, 4.5, 6.5)] * 10,
        offsets=[Offset3DTuple(4.0, -33.0, -8.0)] * 10,
        diamond_locs3d=[Offset3DTuple(0.1, 0.2, 0.3)] * 10,
        raw_diamond_3d=[raw_diamond] * 10,
    )
    assert len(shown) == 1, "the run completed and saved"


def test_the_dcs_calibration_records_the_active_calibration_and_clears_stale(
        app_model, stale, dcs_path, user_pref, monkeypatch):
    folder, _ = stale
    index = app_model.active_calibration_index
    raw_diamond = Offset3DTuple(3.0, -19.0, 0.5)
    _run_dcs_calibration(app_model, user_pref, monkeypatch, dcs_path, raw_diamond)

    written = yaml.safe_load(dcs_path.read_text())
    assert written["calibration_id"] == index.id
    assert app_model.active_calibration_dir == folder
    assert app_model.dcs_state is DcsState.VALID
    assert STALE_BLOCKER not in app_model.recording_blockers
    cfg = app_model.behavior.algorithm.diamond_triangle_config
    assert cfg is not None and cfg.calibration_id == index.id and cfg.raw_diamond_coord == raw_diamond


def test_the_dcs_calibration_writes_into_the_active_calibration_folder(
        app_model, dcs_path, user_pref, system_config, trainer_config_dir, tmp_path, monkeypatch):
    # The active calibration is not the default folder (the app was given another one).
    folder = _write_multires(tmp_path / "elsewhere" / DEFAULT_3D_CALIB_DIR_NAME)
    _save_live_pair_params(system_config, trainer_config_dir, **BASE_PARAMS)
    _config().to_file(dcs_path)
    assert app_model.load_configuration() is True
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(folder)
    assert app_model.dcs_state is DcsState.STALE
    raw_diamond = Offset3DTuple(3.0, -19.0, 0.5)
    _run_dcs_calibration(app_model, user_pref, monkeypatch, dcs_path, raw_diamond)

    with (folder / DEFAULT_CAM_OFFSET_FILE_NAME).open("rb") as fh:
        offsets = pickle.load(fh)
    assert (offsets["x_off"], offsets["y_off"], offsets["z_off"]) == tuple(raw_diamond)
    assert not (Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME).exists()
    assert app_model.active_calibration_dir == folder
    assert yaml.safe_load(dcs_path.read_text())["calibration_id"] == store.load_index(folder).id
    assert app_model.dcs_state is DcsState.VALID


def test_the_dcs_calibration_against_a_legacy_folder_records_no_id(
        app_model, dcs_path, user_pref, calib_dir, tmp_path, monkeypatch):
    # A copy: the run writes camera_offsets.pkl into the active folder, never the repo's fixture.
    folder = _legacy_copy_without_offsets(calib_dir, tmp_path)
    app_model.reload_calib(folder)
    raw_diamond = Offset3DTuple(3.0, -19.0, 0.5)
    _run_dcs_calibration(app_model, user_pref, monkeypatch, dcs_path, raw_diamond)

    assert (folder / DEFAULT_CAM_OFFSET_FILE_NAME).is_file()
    assert "calibration_id" not in yaml.safe_load(dcs_path.read_text())
    assert app_model.active_calibration_dir == folder
    assert app_model.dcs_state is DcsState.VALID


def test_a_dcs_run_without_an_active_calibration_writes_nothing(app_model, dcs_path, user_pref, monkeypatch):
    # No calibration at all: the run used to write its offsets into a stand-in 4mm_6r_8c_4x.
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(None)
    assert app_model.active_calibration_dir is None
    warnings, events = [], []
    monkeypatch.setattr(main_window_module, "QMessageBox", SimpleNamespace(
        warning=lambda parent, title, text, *args: warnings.append((title, text)),
        information=lambda *args, **kwargs: pytest.fail("saved"),
        StandardButton=QMessageBox.StandardButton,
    ))
    dcs_path.unlink(missing_ok=True)
    action = mock.Mock()
    window = SimpleNamespace(_app_model=app_model, _preferences=user_pref, _diamond_triangle_calib_run=None,
                             calib_diamond_triangle_action=action,
                             _timer_calibrate_diamond_triangle=mock.Mock(),
                             _post_api_event=lambda kind, *args: events.append((kind, args)))

    MainWindow.on_calibrate_diamond_triangle(window, True)
    assert window._diamond_triangle_calib_run is None, "the run never started"
    action.setChecked.assert_called_once_with(False)

    MainWindow._handle_diamond_triangle_calib_run(
        window,
        positions=[Offset3DTuple(7.0, 4.5, 6.5)] * 10,
        offsets=[Offset3DTuple(4.0, -33.0, -8.0)] * 10,
        diamond_locs3d=[Offset3DTuple(0.1, 0.2, 0.3)] * 10,
        raw_diamond_3d=[Offset3DTuple(3.0, -19.0, 0.5)] * 10,
    )

    reason = app_model.dcs_refusal()
    assert warnings == [("Calibrate Coordinate System", reason)] * 2
    assert "needs an active 3D calibration" in reason
    assert [kind.name for kind, _ in events] == ["calibrationDcsFailed"]
    assert not dcs_path.exists()
    assert not (Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME).exists()


def _manual_move(app_model, monkeypatch, xyz) -> None:
    """Set the motors as the Hardware Control panel's Set does, with the device accepting each command."""
    hardware = app_model.hardware
    monkeypatch.setattr(hardware, "_send_with_token", lambda *args, **kwargs: uuid.uuid4())
    hardware.set_x(xyz[0])
    hardware.set_y(xyz[1])
    hardware.set_z(xyz[2])


def test_a_manual_move_while_stale_keeps_the_animals_dcs_position(app_model, stale, user_pref, monkeypatch):
    animal = AnimalSubject(name="m1", pellet_x=1.0, pellet_y=2.0, pellet_z=3.0, is_pellet_dcs=True)
    with mock.patch.object(app_model, "on_error"):
        app_model.selected_animal = animal
    saved = Path(user_pref.animal_location) / f"{animal.id}.json"
    before = saved.read_bytes() if saved.exists() else None

    _manual_move(app_model, monkeypatch, (10.0, 20.0, 30.0))

    assert app_model.hardware.last_set_position == Offset3DTuple(10.0, 20.0, 30.0)
    assert (animal.pellet_x, animal.pellet_y, animal.pellet_z, animal.is_pellet_dcs) == (1.0, 2.0, 3.0, True)
    assert (saved.read_bytes() if saved.exists() else None) == before


def test_a_manual_move_while_valid_still_updates_the_animal(
        app_model, dcs_path, system_config, trainer_config_dir, monkeypatch):
    _config().to_file(dcs_path)
    assert app_model.load_configuration() is True
    assert app_model.dcs_state is DcsState.VALID
    animal = AnimalSubject(name="m1", pellet_x=1.0, pellet_y=2.0, pellet_z=3.0, is_pellet_dcs=True)
    app_model.selected_animal = animal

    _manual_move(app_model, monkeypatch, (10.0, 20.0, 30.0))

    # The move that completes the position updates its own axis, converted into DCS.
    expected = _config().motor_to_diamond(Offset3DTuple(10.0, 20.0, 30.0))
    assert animal.is_pellet_dcs
    assert animal.pellet_z == pytest.approx(expected.z)


def test_a_failing_recompute_never_hides_the_calibration_load_error(app_model, monkeypatch):
    def broken_load(calib_dir, no_calibration, *, report):
        raise ValueError("the load failed")

    def broken_refresh():
        raise RuntimeError("the recompute failed")

    monkeypatch.setattr(app_model, "_load_calibration", broken_load)
    monkeypatch.setattr(app_model, "refresh_dcs_state", broken_refresh)
    with pytest.raises(ValueError, match="the load failed"):
        app_model.reload_calib(None)


def test_a_failing_recompute_after_a_good_load_still_raises(app_model, monkeypatch):
    # Nothing to hide then, and a DCS state that did not follow the load must not pass silently.
    def broken_refresh():
        raise RuntimeError("the recompute failed")

    monkeypatch.setattr(app_model, "_load_calibration", lambda calib_dir, no_calibration, *, report: None)
    monkeypatch.setattr(app_model, "refresh_dcs_state", broken_refresh)
    with pytest.raises(RuntimeError, match="the recompute failed"):
        app_model.reload_calib(None)


def test_capture_start_holds_a_dcs_file_gone_stale_before_positioning_the_animal(
        app_model, dcs_path, user_pref, system_config, trainer_config_dir, monkeypatch):
    # The DCS file changed on disk after the configuration load (an older copy restored):
    # Run re-reads it, and must hold it off before it positions the selected animal.
    folder = _write_multires(Path(user_pref.configuration_location) / DEFAULT_3D_CALIB_DIR_NAME)
    _save_live_pair_params(system_config, trainer_config_dir, **BASE_PARAMS)
    app_model.reload_calib(folder)
    _config(calibration_id=app_model.active_calibration_index.id).to_file(dcs_path)
    assert app_model.load_configuration() is True
    assert app_model.dcs_state is DcsState.VALID
    animal = AnimalSubject(name="m1", pellet_x=1.0, pellet_y=2.0, pellet_z=3.0, is_pellet_dcs=True)
    app_model.selected_animal = animal
    _config().to_file(dcs_path)
    seen = []
    positions = AppModel._set_animal_base_positions

    def spy(self, subject):
        seen.append(self.behavior.algorithm.diamond_triangle_config)
        return positions(self, subject)

    monkeypatch.setattr(AppModel, "_set_animal_base_positions", spy)
    try:
        with mock.patch.object(app_model, "on_error") as on_error:
            assert app_model.capture_start() is True
    finally:
        app_model.capture_stop()
    assert seen == [None]
    assert app_model.dcs_state is DcsState.STALE
    assert _titled(on_error, "Coordinate system") == [STALE_NOTICE]
    assert (animal.pellet_x, animal.pellet_y, animal.pellet_z, animal.is_pellet_dcs) == (1.0, 2.0, 3.0, True)


def _reference_copy(app_model, source: Path, tmp_path: Path, *, other_fit: bool = False) -> str:
    """A copy of library entry `source` as "Capture reference images now" makes it, added to the
    library; with `other_fit`, its sensor/fit.yaml differs (a real recalibration)."""
    import cv2
    import numpy as np

    from reachaq.app.model import handle_3d_calibration as h3d

    png = tmp_path / "left.png"
    cv2.imwrite(str(png), np.full((256, 256), 128, np.uint8))
    built = tmp_path / f"copy-{other_fit}"
    new_id = h3d.copy_with_references(source, {"left": png}, built, now=datetime(2026, 11, 20, 9, 30))
    if other_fit:
        fit = built / "sensor" / "fit.yaml"
        fit.write_text(fit.read_text() + "# refitted\n")
    root = library.library_root(Path(app_model._preferences.configuration_location))
    assert library.add_entry(root, built) == new_id
    return new_id


O1 = {"x_off": 3.0, "y_off": -19.0, "z_off": 0.5}
O2 = {"x_off": 4.0, "y_off": -18.0, "z_off": 1.5}


def _parent_and_copy(app_model, tmp_path):
    """A parent entry P, active, with DCS O1 measured on it, and its reference copy C."""
    parent = _active_entry(app_model, tmp_path)
    _configure_with_serials(app_model, SERIALS)
    root = parent.parent
    app_model.write_camera_offsets(parent, O1)
    app_model.reload_calib(None)
    app_model.behavior.algorithm.diamond_triangle_config = _config(parent.name)
    assert app_model.refresh_dcs_state() is DcsState.VALID
    return root, parent.name, _reference_copy(app_model, parent, tmp_path)


def _activate(app_model, root, calibration_id):
    library.set_active(root, calibration_id, by="ben")
    with mock.patch.object(app_model, "on_error") as on_error:
        app_model.reload_calib(None)
    assert app_model.active_calibration_index.id == calibration_id
    return on_error


def test_a_reference_copy_reads_its_parents_offsets_and_keeps_the_dcs(app_model, tmp_path):
    root, parent, copy_id = _parent_and_copy(app_model, tmp_path)

    on_error = _activate(app_model, root, copy_id)

    assert library.fit_lineage(root, copy_id) == (copy_id, parent)
    assert library.offsets_key(root, copy_id) == library.offsets_key(root, parent) == parent
    assert app_model.dcs_state is DcsState.VALID
    assert _titled(on_error, "Coordinate system") == []
    assert app_model._pose_algorithm.cam_offsets == O1
    assert not (root / ".offsets" / copy_id).exists(), "nothing is copied: one folder per fit"


def test_dcs_redone_on_the_copy_is_what_the_parent_loads(app_model, tmp_path):
    root, parent, copy_id = _parent_and_copy(app_model, tmp_path)
    copy_entry = root / copy_id
    _activate(app_model, root, copy_id)
    written = app_model.write_camera_offsets(copy_entry, O2)  # DCS redone on C
    app_model.behavior.algorithm.diamond_triangle_config = _config(copy_id)
    assert written.parent == root / ".offsets" / parent

    _activate(app_model, root, parent)

    assert app_model.dcs_state is DcsState.VALID
    assert app_model._pose_algorithm.cam_offsets == O2


def test_dcs_redone_on_the_parent_is_what_the_copy_loads(app_model, tmp_path):
    root, parent, copy_id = _parent_and_copy(app_model, tmp_path)
    _activate(app_model, root, copy_id)
    assert app_model._pose_algorithm.cam_offsets == O1
    _activate(app_model, root, parent)
    app_model.write_camera_offsets(root / parent, O2)  # DCS redone on P

    _activate(app_model, root, copy_id)

    assert app_model.dcs_state is DcsState.VALID
    assert app_model._pose_algorithm.cam_offsets == O2


def test_offsets_left_under_a_copys_own_id_are_still_read(app_model, tmp_path):
    # Written before offsets were keyed by fit: .offsets/<copy id>/ with the parent's empty.
    import pickle as _pickle

    root, parent, copy_id = _parent_and_copy(app_model, tmp_path)
    for f in (root / ".offsets" / parent).iterdir():
        f.unlink()
    own = root / ".offsets" / copy_id / "camera_offsets-20261120T093000Z.pkl"
    own.parent.mkdir(parents=True)
    own.write_bytes(_pickle.dumps(O2))

    _activate(app_model, root, copy_id)

    assert app_model.camera_offsets_path() == own
    assert app_model._pose_algorithm.cam_offsets == O2


def test_a_recalibration_with_another_fit_still_makes_the_dcs_stale(app_model, tmp_path):
    parent = _active_entry(app_model, tmp_path)
    _configure_with_serials(app_model, SERIALS)
    root = parent.parent
    app_model.write_camera_offsets(parent, {"x_off": 3.0, "y_off": -19.0, "z_off": 0.5})
    app_model.reload_calib(None)
    app_model.behavior.algorithm.diamond_triangle_config = _config(parent.name)
    assert app_model.refresh_dcs_state() is DcsState.VALID

    other = _reference_copy(app_model, parent, tmp_path, other_fit=True)
    library.set_active(root, other, by="ben")
    with mock.patch.object(app_model, "on_error"):
        app_model.reload_calib(None)

    assert library.fit_lineage(root, other) == (other,)
    assert library.offsets_key(root, other) == other != parent.name, "another fit has its own key"
    assert app_model.dcs_state is DcsState.STALE
    assert not (root / ".offsets" / other).exists(), "another fit's offsets are not carried over"
    assert app_model._pose_algorithm.cam_offsets == {"x_off": 0.0, "y_off": 0.0, "z_off": 0.0}


def test_dcs_state_takes_any_id_of_the_same_fit():
    assert dcs_state(_config("parent"), _index("copy"), ("copy", "parent")) is DcsState.VALID
    assert dcs_state(_config("other"), _index("copy"), ("copy", "parent")) is DcsState.STALE
    assert dcs_state(_config(None), _index("copy"), ("copy",)) is DcsState.STALE
