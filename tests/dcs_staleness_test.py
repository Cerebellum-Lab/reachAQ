"""A new 3D calibration makes the DCS configuration stale until DCS is redone.

DCS (diamond_triangle_offset.yaml and the calibration's camera_offsets.pkl) was measured
in the oriented frame of the calibration that was active then. A format-2 calibration has
an id, and DCS records it; when the active calibration's id differs, DCS is stale. Its
consumers then see no configuration at all, Record is blocked with the reason, and the
operator is told once. Redoing DCS against the active calibration makes it valid again.
A legacy calibration has no id, so any valid DCS fits it, as before.
"""

import inspect
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml
from PySide6.QtWidgets import QMessageBox

from autotrainer.core import AnimalSubject, Offset3DTuple
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.inference.multires_calibration.fit import BoardSpec
from autotrainer.inference.multires_calibration.store import CalibrationIndex
from reload_calib_multires_test import BASE, BASE_PARAMS, _save_left_camera_params, _write_multires
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.dcs_state import DcsState, dcs_state
from tools.acquisition.view import main_window as main_window_module
from tools.acquisition.view.main_window import MainWindow


STALE_BLOCKER = "Coordinate system out of date: redo Tools → Calibrate Coordinate System"
STALE_NOTICE = (
    "Out of date after a new 3D calibration: redo Tools → Calibrate Coordinate System "
    "before running sessions."
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
    # A legacy active calibration has no id: any valid DCS fits it, with an id or without.
    (_config(), None, DcsState.VALID),
    (_config(calibration_id=_ID), None, DcsState.VALID),
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
    _save_left_camera_params(system_config, trainer_config_dir, **BASE_PARAMS)
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


def test_the_record_button_is_told_when_dcs_goes_stale(app_model, dcs_path, user_pref, system_config,
                                                       trainer_config_dir):
    _config().to_file(dcs_path)
    _save_left_camera_params(system_config, trainer_config_dir, **BASE_PARAMS)
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
    # The consumers were handed None: motor moves in DCS refuse, the pellet monitor has none.
    with pytest.raises(RuntimeError, match="DCS config"):
        app_model.hardware.set_dcs_x(1.0)
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
    _save_left_camera_params(system_config, trainer_config_dir, **BASE_PARAMS)
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


def test_the_dcs_calibration_records_the_active_calibration_and_clears_stale(
        app_model, stale, dcs_path, user_pref, monkeypatch):
    folder, _ = stale
    index = app_model.active_calibration_index
    shown = []
    # The handler's message box, without a display.
    monkeypatch.setattr(main_window_module, "QMessageBox", SimpleNamespace(
        information=lambda *args, **kwargs: shown.append(args[1:3]),
        StandardButton=QMessageBox.StandardButton,
    ))
    events = []
    window = SimpleNamespace(
        _app_model=app_model,
        _preferences=user_pref,
        _timer_calibrate_diamond_triangle=mock.Mock(),
        _post_api_event=lambda kind, *args: events.append(kind),
    )
    raw_diamond = Offset3DTuple(3.0, -19.0, 0.5)
    # Saved to a new file: the overwrite confirmation and file dialog are not what this tests.
    dcs_path.unlink()
    MainWindow._handle_diamond_triangle_calib_run(
        window,
        positions=[Offset3DTuple(7.0, 4.5, 6.5)] * 10,
        offsets=[Offset3DTuple(4.0, -33.0, -8.0)] * 10,
        diamond_locs3d=[Offset3DTuple(0.1, 0.2, 0.3)] * 10,
        raw_diamond_3d=[raw_diamond] * 10,
    )
    assert len(shown) == 1, "the run completed and saved"

    written = yaml.safe_load(dcs_path.read_text())
    assert written["calibration_id"] == index.id
    assert app_model.active_calibration_dir == folder
    assert app_model.dcs_state is DcsState.VALID
    assert STALE_BLOCKER not in app_model.recording_blockers
    cfg = app_model.behavior.algorithm.diamond_triangle_config
    assert cfg is not None and cfg.calibration_id == index.id and cfg.raw_diamond_coord == raw_diamond


def test_capture_start_recomputes_the_state_after_reloading_dcs():
    # capture_start re-reads the DCS file and then sends the animal's positions; the state
    # must follow that reload before the positions use it. Checked in the source, since
    # capture_start needs cameras and hardware to run.
    source = inspect.getsource(AppModel.capture_start)
    reload_at = source.index("algo.reload_diamond_triangle_config()")
    refresh_at = source.index("self.refresh_dcs_state()", reload_at)
    assert refresh_at < source.index("self._set_animal_base_positions(animal)", reload_at)
