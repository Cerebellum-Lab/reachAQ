import logging
import re
import signal
import subprocess
import sys
import threading
import time
from unittest import mock

import pytest

from pathlib import Path

import verboselogs

from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.core import (
    CameraConfiguration,
    CameraId,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    SystemConfiguration,
)
from autotrainer.video import CaptureCameraAttrs, VideoRecordMode
from autotrainer.inference import GpuRuntimeStatus
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import AppModelStatus, SessionRecordingStatus
from tools.acquisition.model.stimulus_profile_repository import (
    StimulusProfileLibrary,
    StimulusProfileRepository,
)
from tools.acquisition.model.subsystem_status import (
    SubsystemId,
    SubsystemState,
)
from tools.acquisition.model.trial_action import (
    CueIntervalProfile,
    StimulusTriggerProfile,
)
from tools.acquisition.model.user_preferences import UserPreferences


import top_fixtures


def remove_ansi_escape_sequences(s):
    # Regex for common ANSI escape codes
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', s)


@pytest.fixture
def system_config(trainer_config_dir, tmp_path):
    config = SystemConfiguration()
    for cam_member in (CameraId.Left, CameraId.Right, CameraId.Camera3):
        params = dict(width=300, height=200, primary="yes" if cam_member is CameraId.Left else "no")
        cam = CameraConfiguration(name=cam_member.name, params=params, is_enabled=False, is_record_enabled=False)
        cam.scheme = "random"
        cam.id = cam_member
        config.cameras.append(cam)
    config.persistence.output_location = tmp_path.joinpath("Data").as_posix()
    config.save_default(trainer_config_dir)
    return config


@pytest.fixture
def config_file_path(trainer_config_dir):
    return trainer_config_dir.joinpath(SystemConfiguration.make_default_yaml_config_path(trainer_config_dir))


@pytest.fixture
def animals_dir(tmp_path):
    path = tmp_path.joinpath("animals")
    path.mkdir()
    return path


@pytest.fixture
def settings_ini_path(tmp_path):
    return tmp_path.joinpath("settings.ini")


@pytest.fixture
def app_model(
    user_pref,
    calib_dir,
    diamond_config_path,
    system_config,
    monkeypatch,
) -> AppModel:  # noqa
    # for now:
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    assert BehaviorAlgorithm._no_handler_thread is True  # to be safe to start with
    monkeypatch.setenv("AUTOTRAINER_CAN_TRANSPORT", "emulation")
    #
    app = AppModel(user_pref, calib_dir=calib_dir)
    try:
        yield app  # noqa
    finally:
        app.capture_stop(force=True)
        app.on_close()
        top_fixtures.release_multiprocessing_resources(app)


def test_user_preferences(settings_ini_path, user_pref, trainer_config_dir):
    assert Path(user_pref._settings.fileName()) == settings_ini_path
    assert not settings_ini_path.exists()
    user_pref.selected_animal = "foobar"
    user_pref.serial_number = "test-rig"
    user_pref.save()
    assert settings_ini_path.exists()
    user_pref = UserPreferences(settings_file_path=settings_ini_path)
    assert Path(user_pref.configuration_location) == trainer_config_dir
    assert user_pref.selected_animal == ""


def test_legacy_selected_animal_preference_is_cleared_between_sessions(
    settings_ini_path,
):
    settings_ini_path.write_text("[system]\nselected_animal=legacy-animal-id\n")

    user_pref = UserPreferences(settings_file_path=settings_ini_path)

    assert user_pref.selected_animal == ""
    assert not user_pref._settings.contains("system/selected_animal")


def test_hardware_menu_settings_persist_to_system_yaml(
    app_model,
    config_file_path,
    system_config,
):
    assert app_model.load_configuration() is True

    message = app_model.update_hardware_configuration(
        can_enabled=False,
        pellet_controller_enabled=False,
        nidaq_enabled=False,
        rfid_reader_enabled=False,
        rfid_device="/dev/serial/by-id/persistent-rfid",
    )

    saved = SystemConfiguration.load_yaml_file(config_file_path)
    assert saved.hardware.can_enabled is False
    assert saved.hardware.pellet_controller_enabled is False
    assert saved.hardware.nidaq_enabled is False
    assert saved.hardware.rfid_reader_enabled is False
    assert saved.hardware.rfid_device == "/dev/serial/by-id/persistent-rfid"
    assert "saved" in message.lower()


def test_load_config(
    app_model,
    trainer_config_dir,
    animals_dir,
    calib_dir,
    system_config,
    config_file_path,
):
    from tools.acquisition.view.main_content import _visible_reach_cameras

    res = app_model.load_configuration()
    assert res is True
    assert app_model.left_camera.name == "left"
    assert app_model.right_camera.name == "right"
    assert len(app_model.reach_cameras) == 3
    assert app_model.stim_camera is app_model.get_camera_model(CameraId.Camera3)
    assert app_model.stim_camera.name == "stimCam"
    assert not app_model.stim_camera.is_enabled
    assert _visible_reach_cameras(app_model.reach_cameras) == (
        app_model.left_camera,
        app_model.right_camera,
    )
    assert _visible_reach_cameras(
        app_model.reach_cameras,
        include_disabled_optional=True,
    ) == (
        app_model.left_camera,
        app_model.right_camera,
        app_model.stim_camera,
    )
    app_model.stim_camera.is_enabled = True
    assert _visible_reach_cameras(app_model.reach_cameras) == (
        app_model.left_camera,
        app_model.right_camera,
        app_model.stim_camera,
    )
    app_model.save_configuration()
    saved_configuration = SystemConfiguration.load_yaml_file(config_file_path)
    assert saved_configuration.get_camera(CameraId.Camera3).is_enabled
    assert app_model.output_location == system_config.persistence.output_location
    pref = app_model.preferences
    assert Path(pref.animal_location) == animals_dir
    assert Path(pref.configuration_location) == trainer_config_dir


def _fail_the_nidaq_step(app_model, monkeypatch):
    # The step that raised on christielab10 on 2026-09-17: after the cameras
    # are reconfigured, before the load completes.
    def refuse(*_args, **_kwargs):
        raise RuntimeError("NI-DAQ refused to re-apply its timing")

    monkeypatch.setattr(app_model.nidaq_signal_monitor, "load_configuration", refuse)


def test_a_load_that_fails_part_way_is_never_saved(app_model, config_file_path, monkeypatch):
    assert app_model.load_configuration() is True
    on_disk = config_file_path.read_bytes()

    _fail_the_nidaq_step(app_model, monkeypatch)
    with pytest.raises(RuntimeError, match="re-apply"):
        app_model.load_configuration()
    # A change the save would otherwise write, as a half-applied demo reload did.
    app_model.stim_camera.is_enabled = True
    app_model.save_configuration()

    assert config_file_path.read_bytes() == on_disk


def test_saving_resumes_once_a_load_completes(app_model, config_file_path, monkeypatch):
    assert app_model.load_configuration() is True
    _fail_the_nidaq_step(app_model, monkeypatch)
    with pytest.raises(RuntimeError):
        app_model.load_configuration()
    monkeypatch.undo()

    assert app_model.load_configuration() is True
    app_model.stim_camera.is_enabled = True
    app_model.save_configuration()

    assert SystemConfiguration.load_yaml_file(config_file_path).get_camera(CameraId.Camera3).is_enabled


def test_a_run_on_another_file_saves_back_to_that_file(app_model, config_file_path, tmp_path):
    copy = tmp_path / "copy_configuration.yaml"
    copy.write_bytes(config_file_path.read_bytes())
    real = config_file_path.read_bytes()

    assert app_model.load_configuration(copy) is True
    app_model.stim_camera.is_enabled = True
    app_model.save_configuration()

    assert config_file_path.read_bytes() == real
    assert SystemConfiguration.load_yaml_file(copy).get_camera(CameraId.Camera3).is_enabled


def test_load_config_extra_reach_camera_slot(app_model, trainer_config_dir, system_config):
    camera3 = CameraConfiguration(
        id=CameraId.Camera3,
        name=CameraId.Camera3.name,
        is_enabled=True,
        is_record_enabled=True,
        params=dict(width=300, height=200),
        scheme="random",
        record_prebuffer_duration=0,
    )
    system_config.cameras.append(camera3)
    system_config.save_default(trainer_config_dir)

    assert app_model.load_configuration() is True

    loaded_camera3 = app_model.get_camera_model(CameraId.Camera3)
    assert len(app_model.reach_cameras) == 3
    assert loaded_camera3.is_enabled
    assert loaded_camera3.name == "stimCam"
    assert app_model.make_project_info().camera_names == ("stimCam",)


def test_stim_camera_session_modes_are_mutually_exclusive(app_model):
    assert app_model.load_configuration() is True
    camera = app_model.stim_camera
    camera.is_enabled = True
    operator_configuration = camera.save_configuration()

    camera.configure_stim_session_mode(True)
    assert camera.active_config.params["stim_mode"] == "stimulation"
    assert camera.active_config.params["fps"] == 900
    assert camera.is_primary
    assert not camera.is_recording_enabled
    assert camera._stim_detection_configuration is not None
    assert camera.save_configuration().params == operator_configuration.params
    assert camera.save_configuration().is_enabled
    assert camera.save_configuration(runtime_mode=True).params["stim_mode"] == "stimulation"

    camera.configure_stim_session_mode(False)
    assert camera.active_config.params["stim_mode"] == "ordinary"
    assert camera.active_config.params["fps"] == 150
    assert not camera.is_primary
    assert camera.is_recording_enabled
    assert camera._stim_detection_configuration is None
    assert camera.save_configuration().params == operator_configuration.params
    assert camera.save_configuration(runtime_mode=True).params["stim_mode"] == "ordinary"


def test_stimulus_profiles_are_saved_and_reloaded(app_model):
    assert app_model.load_configuration() is True

    tone = app_model.save_tone_profile("trial-cue", 7000, 125)
    laser = app_model.save_laser_profile(
        profile_id="first-reach-pulse",
        amplitude_volts=2.0,
        pulse_duration_ms=5.0,
    )

    assert app_model._tone_profiles[tone.profile_id] == tone
    assert app_model._laser_profiles[laser.profile_id] == laser
    assert app_model._stimulus_profile_repository.load().laser_profiles[-1] == laser
    app_model.delete_stimulus_profile("tone", tone.profile_id)
    assert tone.profile_id not in app_model._tone_profiles


def _save_cue_and_trigger_profiles(config_dir, name):
    repository = StimulusProfileRepository(config_dir / "stimulus_profiles.json")
    library = repository.load()
    repository.save(
        StimulusProfileLibrary(
            revision=library.revision,
            tone_profiles=library.tone_profiles,
            cue_interval_profiles=(CueIntervalProfile(f"cue-{name}", 1),),
            stimulus_trigger_profiles=(
                StimulusTriggerProfile(
                    f"trigger-{name}",
                    1,
                    categories=({
                        "category_id": "first_reach",
                        "trigger": "first_reach",
                        "percentage": 100.0,
                    },),
                ),
            ),
        ),
        expected_revision=library.revision,
    )


def _cue_and_trigger_profile_ids(app_model):
    state = app_model.trial_protocol_state
    return (
        tuple(item["profile_id"] for item in state["cue_interval_profiles"]),
        tuple(item["profile_id"] for item in state["stimulus_trigger_profiles"]),
    )


@pytest.fixture
def startup_cue_and_trigger_profiles(trainer_config_dir):
    # On disk before the application model is built, as they are at startup.
    _save_cue_and_trigger_profiles(trainer_config_dir, "startup")


def test_a_configuration_from_another_directory_brings_its_cue_and_trigger_profiles(
    startup_cue_and_trigger_profiles, app_model, config_file_path, tmp_path,
):
    # A load swapped in the loaded directory's tone, laser and shift profiles
    # but kept the cue-interval and trigger profiles read at startup, and the
    # next profile save wrote those into the loaded directory's store.
    assert app_model.load_configuration() is True
    assert _cue_and_trigger_profile_ids(app_model) == (
        ("cue-startup",), ("trigger-startup",))
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = other_dir / config_file_path.name
    other.write_bytes(config_file_path.read_bytes())
    _save_cue_and_trigger_profiles(other_dir, "other")

    assert app_model.load_configuration(other) is True

    assert _cue_and_trigger_profile_ids(app_model) == (
        ("cue-other",), ("trigger-other",))
    app_model.save_tone_profile("trial-cue", 7000, 125)
    saved = StimulusProfileRepository(other_dir / "stimulus_profiles.json").load()
    assert tuple(item.profile_id for item in saved.cue_interval_profiles) == (
        "cue-other",)
    assert tuple(item.profile_id for item in saved.stimulus_trigger_profiles) == (
        "trigger-other",)


def test_spinnaker_camera_selectors_keep_only_their_configured_binding(
    app_model,
    trainer_config_dir,
    system_config,
):
    # Binding a spinnaker:// camera resolves the camera class, which imports
    # the SDK. See the note in spinnaker_cam_test.py.
    pytest.importorskip("PySpin")
    serials = {
        CameraId.Left: "24095781",
        CameraId.Right: "24095782",
        CameraId.Camera3: "24095783",
    }
    for camera in system_config.cameras:
        if camera.id in (CameraId.Left, CameraId.Right):
            camera.scheme = "spinnaker"
            camera.host = serials[camera.id]
            camera.params = {"width": 256, "height": 256, "fps": 150}
    system_config.cameras.append(
        CameraConfiguration(
            id=CameraId.Camera3,
            name="stimCam",
            scheme="spinnaker",
            host=serials[CameraId.Camera3],
            params={"width": 256, "height": 256, "fps": 150},
        )
    )
    system_config.save_default(trainer_config_dir)
    assert app_model.load_configuration() is True

    discovered_sources = (
        CaptureCameraAttrs("Random Image", "random://0?width=300&height=200"),
        *(
            CaptureCameraAttrs(f"Spinnaker {serial}", f"spinnaker://{serial}")
            for serial in serials.values()
        ),
    )
    for camera, expected_name in (
        (app_model.left_camera, "left"),
        (app_model.right_camera, "right"),
        (app_model.stim_camera, "stimCam"),
    ):
        camera.refresh_camera_list(discovered_sources)
        spinnaker_options = tuple(
            source
            for source in camera.camera_list
            if source.url.startswith("spinnaker://")
        )
        assert camera.camera_source.name == expected_name
        assert spinnaker_options == (camera.camera_source,)
        assert all(not source.name.startswith("Spinnaker ") for source in camera.camera_list)


def test_start_stop(app_model, settings_ini_path):
    assert not settings_ini_path.exists()
    assert app_model.load_configuration() is True
    assert app_model.capture_start() is True
    app_model.capture_stop()
    assert not settings_ini_path.exists()  # still
    app_model.on_close()
    assert settings_ini_path.exists()  # but saved on close
    # ...


def _null_lasers():
    from autotrainer.core import (
        LaserChannelConfiguration,
        LaserChannelId,
        LaserSystemConfiguration,
    )
    return LaserSystemConfiguration.from_channels(
        (
            LaserChannelConfiguration(
                channel_id=LaserChannelId.LASER_1,
                analog_output="Dev1/ao0",
                diode_input="Dev1/ai0",
                shutter_output="Dev1/port0/line2",
            ),
        ),
        backend="null",
        sample_rate_hz=1000.0,
    )


def _blocking_ramp(app_model, monkeypatch):
    """A ramp held inside its controller until released, as one inside DAQmx is."""
    from autotrainer.device import NullLaserController

    inside = threading.Event()
    release = threading.Event()
    controllers = []

    class _Held(NullLaserController):
        closed = False

        def run_calibration_ramp(self, ramp):
            inside.set()
            release.wait(30.0)
            return super().run_calibration_ramp(ramp)

        def close(self):
            self.closed = True
            super().close()

    def open_controller(configuration, **_kwargs):
        controllers.append(_Held(configuration))
        return controllers[-1]

    monkeypatch.setattr(app_model.laser, "open_controller", open_controller)
    return inside, release, controllers


def _ramp(timeout_seconds=None):
    from autotrainer.core import LaserChannelId
    from autotrainer.device import LaserCalibrationRamp
    return LaserCalibrationRamp(
        channel_id=LaserChannelId.LASER_1, start_volts=0.0, stop_volts=5.0,
        steps=3, samples_per_step=10, timeout_seconds=timeout_seconds,
        # 2 samples at the fake rig's 100 kHz: these short steps would keep
        # none of 600 us.
        settle_seconds=20e-6)


def _in_thread(function, *args):
    outcome = []

    def run():
        try:
            outcome.append(function(*args))
        except Exception as error:
            outcome.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def test_closing_mid_ramp_waits_for_the_ramp_to_hand_the_laser_back(app_model, monkeypatch):
    # The ramp's controller is its own, never the laser model's, so the
    # laser close in on_close never reached it, and nothing waited for the
    # ramp: the process could end with the 6713 on its last sample and the
    # shutter line high.
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    inside, release, controllers = _blocking_ramp(app_model, monkeypatch)
    ramp_thread, _ramp_outcome = _in_thread(app_model.run_laser_calibration_ramp, _ramp())
    assert inside.wait(10.0)

    close_thread, close_outcome = _in_thread(app_model.on_close)
    time.sleep(0.5)
    assert close_thread.is_alive(), close_outcome
    controller, = controllers
    assert not controller.closed

    release.set()
    ramp_thread.join(10.0)
    close_thread.join(60.0)

    assert not close_thread.is_alive()
    assert controller.closed
    assert close_outcome == [None]


def test_closing_past_the_ramp_timeout_closes_the_ramp_controller(app_model, monkeypatch):
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_CLOSE_MARGIN_S", 0.5)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    inside, release, controllers = _blocking_ramp(app_model, monkeypatch)
    ramp_thread, _ramp_outcome = _in_thread(
        app_model.run_laser_calibration_ramp, _ramp(timeout_seconds=0.5))
    assert inside.wait(10.0)
    try:
        close_thread, close_outcome = _in_thread(app_model.on_close)
        close_thread.join(60.0)

        # The ramp is still held, and close went on without it once the
        # ramp's own timeout had passed, closing its controller: that drives
        # the command to its minimum and closes the shutters.
        assert not close_thread.is_alive()
        assert close_outcome == [None]
        assert ramp_thread.is_alive()
        controller, = controllers
        assert controller.closed
    finally:
        release.set()
        ramp_thread.join(10.0)


def test_a_forced_ramp_close_that_hangs_is_given_up_on_and_named(
    app_model, monkeypatch, caplog,
):
    # A hang does not make the laser safer: the output stays driven either
    # way (controller ruling, 2026-09-25). The forced close runs with a bound;
    # past it a CRITICAL names the laser and the ramp's highest command, and
    # reachAQ goes on closing.
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_CLOSE_MARGIN_S", 0.5)
    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 0.5)
    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_RAMP_END_WAIT_S", 0.2)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    inside, release, controllers = _blocking_ramp(app_model, monkeypatch)
    ramp_thread, _ramp_outcome = _in_thread(
        app_model.run_laser_calibration_ramp, _ramp(timeout_seconds=0.5))
    assert inside.wait(10.0)
    controller, = controllers
    hang = threading.Event()
    closing = []

    def hung_close():
        closing.append(True)
        hang.wait(30.0)

    monkeypatch.setattr(controller, "close", hung_close)
    try:
        with caplog.at_level("CRITICAL"):
            close_thread, close_outcome = _in_thread(app_model.on_close)
            close_thread.join(20.0)

        assert not close_thread.is_alive()
        assert close_outcome == [None]
        assert closing == [True]
        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        message = critical.getMessage()
        assert "laser 1" in message and "5 V" in message
    finally:
        hang.set()
        release.set()
        ramp_thread.join(10.0)


def _daqmx_fake():
    """The laser device tests' stand-in for NI-DAQmx, loaded from its file.

    auto-trainer-device/tests is a separate import root from this directory.
    """
    import importlib.util

    path = top_fixtures.repo_root_dir.joinpath(
        "auto-trainer-device", "tests", "nidaq_daqmx_fake.py")
    spec = importlib.util.spec_from_file_location("nidaq_daqmx_fake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_forced_ramp_close_that_raises_is_named_and_the_ramp_still_resets(
    app_model, monkeypatch, caplog,
):
    # The abort neither released ao0 nor let the ramp's wait return. close()
    # gave up waiting, its reset was refused at -50103, and it raised. The
    # forced close logged that at ERROR only: no CRITICAL, since it had not
    # hung. And reachAQ went on to exit before the ramp, let go by its
    # driver, could put the command back on its own closed branch.
    from autotrainer.device import NidaqLaserController, nidaq_laser
    from tools.acquisition.model import app_model as app_model_module

    fake = _daqmx_fake()
    daq = fake.FakeDaqmx(block_wait=True, abort_releases=False,
                         abort_unblocks=False, hold_waits=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    monkeypatch.setattr(nidaq_laser, "_CALIBRATION_RELEASE_TIMEOUT_S", 0.1)
    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_CLOSE_MARGIN_S", 0.5)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    controllers = []

    def open_controller(_configuration, **_kwargs):
        controller = NidaqLaserController(fake.rig_lasers())
        close = controller.close

        def close_then_the_driver_lets_go():
            try:
                close()
            finally:
                # After close() returns, as a driver that lets the ramp go late.
                daq.waits_released.set()

        controller.close = close_then_the_driver_lets_go
        controllers.append(controller)
        return controller

    monkeypatch.setattr(app_model.laser, "open_controller", open_controller)
    ramp_thread, ramp_outcome = _in_thread(
        app_model.run_laser_calibration_ramp, _ramp(timeout_seconds=0.5))
    deadline = time.monotonic() + 10.0
    while not any(task.label.endswith("calibration_ao") and task.started
                  for task in daq.tasks):
        assert time.monotonic() < deadline, "the ramp did not start"
        time.sleep(0.01)
    # The ramp's thread also opened the controller, whose own first command
    # write is 0 V: only what it writes from here on is the ramp's cleanup.
    ramp_started = len(daq.writes)
    try:
        with caplog.at_level("WARNING"):
            close_thread, close_outcome = _in_thread(app_model.on_close)
            close_thread.join(30.0)
            # What the ramp had written by the time reachAQ finished closing.
            ramp_writes = daq.writes_to(
                "PXI1Slot4/ao0", task_suffix="manual_ao", thread=ramp_thread,
                since=ramp_started)

        assert not close_thread.is_alive()
        assert close_outcome == [None]
        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        message = critical.getMessage()
        assert "laser 1 failed to close" in message and "5 V" in message
        assert "channel 1 reset" in message
        # The ramp's own closed-branch write, made before closing went on.
        assert ramp_writes == [0.0]
        ramp_thread.join(5.0)
        assert not ramp_thread.is_alive()
        # The CRITICAL comes first, then what the wait for the ramp found; it
        # does not claim the ramp's reset worked.
        messages = [record.getMessage() for record in caplog.records]
        ended = messages.index(
            "The laser calibration ramp ended; see above for any error from "
            "its own reset")
        assert messages.index(message) < ended
        error, = ramp_outcome
        assert "was closed while it ran" in str(error)
    finally:
        daq.waits_released.set()
        ramp_thread.join(10.0)


def test_a_ramp_whose_own_close_hangs_is_named_when_reachaq_closes(
    app_model, monkeypatch, caplog,
):
    # The ramp's own thread had taken its controller to close it, and that
    # close hung. The forced close found no controller, did nothing, and
    # logged nothing: reachAQ exited with the laser as the ramp left it.
    from autotrainer.device import NullLaserController
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_CLOSE_MARGIN_S", 0.5)
    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_RAMP_END_WAIT_S", 0.3)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    closing, hang = threading.Event(), threading.Event()

    class _HangsOnClose(NullLaserController):
        def close(self):
            closing.set()
            hang.wait(30.0)
            super().close()

    monkeypatch.setattr(app_model.laser, "open_controller",
                        lambda configuration, **_kwargs: _HangsOnClose(configuration))
    ramp_thread, _ramp_outcome = _in_thread(
        app_model.run_laser_calibration_ramp, _ramp(timeout_seconds=0.5))
    assert closing.wait(10.0), "the ramp did not reach its own close"
    try:
        with caplog.at_level("ERROR"):
            close_thread, close_outcome = _in_thread(app_model.on_close)
            close_thread.join(20.0)

        assert not close_thread.is_alive()
        assert close_outcome == [None]
        announcement, = [record.getMessage() for record in caplog.records
                         if "after reachAQ began closing" in record.getMessage()]
        assert announcement.endswith("its own thread is closing its laser controller")
        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        message = critical.getMessage()
        assert "laser 1 is being closed by the ramp's own thread" in message
        assert "5 V" in message
    finally:
        hang.set()
        ramp_thread.join(10.0)


class _ReleaseOnCritical(logging.Handler):
    """Lets `event` go the moment a CRITICAL is logged."""

    def __init__(self, event):
        super().__init__(level=logging.CRITICAL)
        self._event = event

    def emit(self, record):
        self._event.set()


def test_a_ramp_that_ends_after_the_critical_is_reported_as_ended(
    app_model, monkeypatch, caplog,
):
    # The CRITICAL came after the wait for the ramp. When the ramp's own
    # thread was closing its controller and finished inside that wait,
    # closing returned in silence, taking the laser as reset; one that
    # finished only after the CRITICAL was reported as not having ended.
    from autotrainer.device import NullLaserController
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_CLOSE_MARGIN_S", 0.5)
    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_RAMP_END_WAIT_S", 3.0)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    closing, hang = threading.Event(), threading.Event()

    class _HangsOnClose(NullLaserController):
        def close(self):
            closing.set()
            hang.wait(30.0)
            super().close()

    monkeypatch.setattr(app_model.laser, "open_controller",
                        lambda configuration, **_kwargs: _HangsOnClose(configuration))
    ramp_thread, _ramp_outcome = _in_thread(
        app_model.run_laser_calibration_ramp, _ramp(timeout_seconds=0.5))
    assert closing.wait(10.0), "the ramp did not reach its own close"
    release = _ReleaseOnCritical(hang)
    logging.getLogger().addHandler(release)
    try:
        with caplog.at_level("WARNING"):
            close_thread, close_outcome = _in_thread(app_model.on_close)
            close_thread.join(20.0)

        assert not close_thread.is_alive()
        assert close_outcome == [None]
        levels = [(record.levelname, record.getMessage()) for record in caplog.records
                  if record.levelname == "CRITICAL" or "calibration ramp" in record.getMessage()]
        critical = next(index for index, (level, _) in enumerate(levels)
                        if level == "CRITICAL")
        assert levels[critical + 1] == (
            "WARNING",
            "The laser calibration ramp ended; see above for any error from "
            "its own reset")
        assert not any("had not ended" in message for _, message in levels)
    finally:
        logging.getLogger().removeHandler(release)
        hang.set()
        ramp_thread.join(10.0)


def test_a_ramp_still_opening_its_controller_is_not_blamed_for_the_output(
    app_model, monkeypatch, caplog,
):
    # Closing while the ramp was still opening its controller: nothing had
    # been opened to close, and none of the ramp's commands written. The
    # ERROR said closing was closing its laser controller, and the CRITICAL
    # that the output may still hold the ramp's last command.
    from autotrainer.device import NullLaserController
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_CLOSE_MARGIN_S", 0.5)
    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_RAMP_END_WAIT_S", 0.2)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    opening, release = threading.Event(), threading.Event()

    def open_controller(configuration, **_kwargs):
        opening.set()
        release.wait(30.0)
        return NullLaserController(configuration)

    monkeypatch.setattr(app_model.laser, "open_controller", open_controller)
    ramp_thread, _ramp_outcome = _in_thread(
        app_model.run_laser_calibration_ramp, _ramp(timeout_seconds=0.5))
    assert opening.wait(10.0), "the ramp did not begin opening its controller"
    try:
        with caplog.at_level("ERROR"):
            close_thread, close_outcome = _in_thread(app_model.on_close)
            close_thread.join(20.0)

        assert not close_thread.is_alive()
        assert close_outcome == [None]
        announcement, = [record.getMessage() for record in caplog.records
                         if "after reachAQ began closing" in record.getMessage()]
        assert "closing its laser controller" not in announcement
        assert "had not opened its laser controller" in announcement
        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        message = critical.getMessage()
        assert "laser 1 had not finished opening" in message
        assert "never wrote a command" in message
        assert "may still hold" not in message and "by hand" not in message
    finally:
        release.set()
        ramp_thread.join(10.0)


def test_a_ramp_that_never_wrote_a_command_is_not_blamed_for_the_output(
    app_model, monkeypatch, caplog,
):
    # The ramp opened its controller as reachAQ began closing, so it refused
    # to start and closed the controller itself, and that close hung. It had
    # written no command, and the CRITICAL said the output may hold its last.
    from autotrainer.device import NullLaserController
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_CLOSE_MARGIN_S", 0.5)
    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_RAMP_END_WAIT_S", 0.2)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(_null_lasers())
    opening, closing, hang = threading.Event(), threading.Event(), threading.Event()
    ramps = []

    class _HangsOnClose(NullLaserController):
        def run_calibration_ramp(self, ramp):
            ramps.append(ramp)
            return super().run_calibration_ramp(ramp)

        def close(self):
            closing.set()
            hang.wait(30.0)
            super().close()

    def open_after_closing_began(configuration, **_kwargs):
        opening.set()
        app_model._closing_event.wait(30.0)
        return _HangsOnClose(configuration)

    monkeypatch.setattr(app_model.laser, "open_controller", open_after_closing_began)
    ramp_thread, ramp_outcome = _in_thread(
        app_model.run_laser_calibration_ramp, _ramp(timeout_seconds=0.5))
    assert opening.wait(10.0), "the ramp did not begin opening its controller"
    try:
        with caplog.at_level("ERROR"):
            close_thread, close_outcome = _in_thread(app_model.on_close)
            assert closing.wait(10.0), "the ramp did not close its controller"
            close_thread.join(20.0)

        assert not close_thread.is_alive()
        assert close_outcome == [None]
        assert ramps == []
        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        message = critical.getMessage()
        assert "laser 1 is being closed by the ramp's own thread" in message
        assert "had not started, so it wrote no command" in message
        assert "may still hold" not in message and "by hand" not in message
    finally:
        hang.set()
        ramp_thread.join(10.0)


def _system_mode_with_the_fake_laser(app_model, monkeypatch, lasers=None, **fake_modes):
    """System Mode running with the NI-DAQ laser on the DAQmx stand-in."""
    from autotrainer.device import nidaq_laser

    fake = _daqmx_fake()
    daq = fake.FakeDaqmx(**fake_modes)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(fake.rig_lasers(**(lasers or {})))
    assert app_model.capture_start() is True
    assert app_model.laser.is_connected
    return daq


#: christielab10's board STIM route for laser 1, whose close releases it.
_ROUTED_LASER = dict(
    trigger_source="/PXI1Slot4/PXI_Trig0", trigger_route_source="/PXI1Slot5/PFI0")


def _wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met"
        time.sleep(0.02)


def _laser_status(app_model):
    return app_model.subsystem_statuses[SubsystemId.LASER.value]


def _refused_without_the_driver(call):
    """What `call` raised, made on a thread of its own and waited for, bounded.

    A call that reaches the hung driver hangs with it; it must not hang the
    test too.
    """
    thread, outcome = _in_thread(call)
    thread.join(5.0)
    assert not thread.is_alive(), "it called into the driver that hung"
    error, = outcome
    assert isinstance(error, RuntimeError), error
    return str(error)


@pytest.mark.parametrize("hang", ["disconnect_terms", "task_close"])
def test_stop_with_a_hung_laser_close_is_bounded_and_holds_off_run_until_it_ends(
    app_model, monkeypatch, caplog, hang,
):
    # The System Mode controller's close had no bound: a driver hung in it
    # hung Stop, and exit, indefinitely, and nothing told the operator.
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 0.5)
    daq = _system_mode_with_the_fake_laser(app_model, monkeypatch, lasers=_ROUTED_LASER)
    # From Stop on: opening the controller closes tasks of its own.
    daq.hang = {hang}
    errors = []
    app_model.on_error += lambda title, message: errors.append((title, message))
    try:
        with caplog.at_level("CRITICAL"):
            stop_thread, stop_outcome = _in_thread(app_model.capture_stop)
            stop_thread.join(10.0)

        assert not stop_thread.is_alive()
        assert stop_outcome == [None]
        assert daq.hung == [hang]
        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        message = critical.getMessage()
        assert "did not close within 0.5 s" in message
        assert "laser 1" in message and "0 V" in message
        assert "make the laser safe by hand" in message

        # Held off while that close is still inside the driver, by name.
        refusal = app_model.laser_controller_close_refusal()
        assert "still closing after a driver hang" in refusal
        # Nor is the laser reconfigured over it: each of these closes the
        # laser model's controller, the one whose close hung.
        assert _refused_without_the_driver(app_model.load_configuration) == (
            f"Loading a configuration is unavailable: {refusal}")
        assert _refused_without_the_driver(
            lambda: app_model.update_daq_port_configuration(
                app_model.nidaq_ports, app_model.laser.configuration)) == (
            f"Changing the DAQ port configuration is unavailable: {refusal}")
        assert daq.hung == [hang]
        # The laser's status says so, not "stopped".
        status = _laser_status(app_model)
        assert status.state is SubsystemState.FAILED
        assert status.error == refusal
        assert app_model.capture_start() is False
        assert errors[-1] == ("Run unavailable", f"System Mode cannot start: {refusal}")
        assert refusal in app_model.laser_calibration_refusal()
        with pytest.raises(RuntimeError, match="still closing after a driver hang"):
            app_model.refresh_hardware_bindings()
        with pytest.raises(RuntimeError, match="still closing after a driver hang"):
            app_model.run_stim_bench_test(None, 1, "direct_ni_software")
        assert daq.hung == [hang]
        # A hardware settings save rewrites every subsystem's intent; the
        # laser's stays the pending close.
        hardware = app_model.loaded_configuration.hardware
        app_model.update_hardware_configuration(
            can_enabled=hardware.can_enabled,
            pellet_controller_enabled=hardware.pellet_controller_enabled,
            nidaq_enabled=hardware.nidaq_enabled,
            rfid_reader_enabled=hardware.rfid_reader_enabled,
            rfid_device=hardware.rfid_device,
        )
        status = _laser_status(app_model)
        assert status.state is SubsystemState.FAILED
        assert status.error == refusal

        daq.hang_released.set()
        _wait_until(lambda: not app_model.laser_controller_close_refusal())
        _wait_until(lambda: _laser_status(app_model).state is SubsystemState.STOPPED)
        assert "late" in _laser_status(app_model).reason
        assert not app_model.laser.is_connected
        assert app_model.capture_start() is True
    finally:
        daq.hang_released.set()
        app_model.capture_stop()


def test_a_close_that_ends_just_past_its_bound_is_still_reported(
    app_model, monkeypatch, caplog,
):
    # Whether a close finished was read from its done flag after the wait
    # had given up on it. One that ended in between read as a clean close:
    # no CRITICAL, although its late finish was logged as a close given up on.
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 0.5)
    daq = _system_mode_with_the_fake_laser(
        app_model, monkeypatch, lasers=_ROUTED_LASER, hang={"disconnect_terms"})
    bounded_close = app_model_module._BoundedClose
    run = bounded_close.run

    def run_then_the_close_ends(self, timeout):
        finished = run(self, timeout)
        daq.hang_released.set()
        assert self.done.wait(5.0)
        return finished

    monkeypatch.setattr(bounded_close, "run", run_then_the_close_ends)
    try:
        with caplog.at_level("WARNING"):
            app_model.capture_stop()
            _wait_until(lambda: any("after it was given up on" in record.getMessage()
                                    for record in caplog.records))

        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        assert "did not close within 0.5 s" in critical.getMessage()
    finally:
        daq.hang_released.set()


def test_a_retry_over_a_hung_laser_close_is_refused_until_the_close_ends(
    app_model, monkeypatch, caplog,
):
    # A Run whose laser failed to start closes the controller it opened, and
    # that close can hang. The subsystem retry opened another over it: that
    # closes the laser model's controller first, calling into the driver
    # that hung. And the close's late finish wrote "stopped" with System Mode
    # still on.
    from autotrainer.device import nidaq_laser
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 0.5)
    fake = _daqmx_fake()
    daq = fake.FakeDaqmx(hang={"disconnect_terms"})
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(fake.rig_lasers(**_ROUTED_LASER))
    receiver = app_model.laser.start_direct_trigger_receiver
    fail = ["once"]

    def fails_once(*args, **kwargs):
        if fail:
            fail.pop()
            raise RuntimeError("the direct trigger receiver did not start")
        return receiver(*args, **kwargs)

    monkeypatch.setattr(app_model.laser, "start_direct_trigger_receiver", fails_once)
    try:
        with caplog.at_level("CRITICAL"):
            assert app_model.capture_start() is True
        assert daq.hung == ["disconnect_terms"]
        refusal = app_model.laser_controller_close_refusal()
        assert "still closing after a driver hang" in refusal
        status = _laser_status(app_model)
        assert status.state is SubsystemState.FAILED
        assert "did not start" in status.error and refusal in status.error

        thread, outcome = _in_thread(app_model.retry_failed_subsystems)
        thread.join(5.0)
        assert not thread.is_alive(), "the retry called into the driver that hung"
        assert daq.hung == ["disconnect_terms"]
        status = _laser_status(app_model)
        assert status.state is SubsystemState.FAILED
        assert status.error == f"laser controller not opened: {refusal}"

        daq.hang_released.set()
        _wait_until(lambda: "late" in _laser_status(app_model).error)
        # Still System Mode: failed, and retried by a refresh, not stopped.
        status = _laser_status(app_model)
        assert status.state is SubsystemState.FAILED
        assert "Refresh Hardware" in status.error
        assert app_model.laser_controller_close_refusal() == ""

        app_model.retry_failed_subsystems()
        assert _laser_status(app_model).state is SubsystemState.READY
        assert app_model.laser.is_connected
    finally:
        daq.hang_released.set()
        app_model.capture_stop()


def test_a_ramp_whose_controller_failed_to_open_shows_the_pending_close_in_idle(
    app_model, monkeypatch,
):
    # The ramp's open fails part-way in Idle, and its close hangs. The watch
    # held laser work off, but LASER never said so, and the late finish then
    # wrote "closed, late, after a driver hang" over a status that had never
    # said the close was pending.
    daq = _failing_open(monkeypatch, app_model)
    try:
        with pytest.raises(RuntimeError, match="refused the laser's shutter line"):
            app_model.run_laser_calibration_ramp(_ramp(timeout_seconds=0.5))

        assert daq.hung == ["disconnect_terms"]
        refusal = app_model.laser_controller_close_refusal()
        assert "still closing after a driver hang" in refusal
        status = _laser_status(app_model)
        assert status.state is SubsystemState.FAILED
        assert status.error == refusal

        daq.hang_released.set()
        _wait_until(lambda: _laser_status(app_model).state is SubsystemState.STOPPED)
        assert "failed to open" in _laser_status(app_model).reason
        assert app_model.laser_controller_close_refusal() == ""
    finally:
        daq.hang_released.set()


@pytest.mark.parametrize("order", ["together", "one_after_the_other"])
def test_a_stop_and_a_close_entering_together_close_the_laser_once(
    app_model, monkeypatch, order,
):
    # The check for a close under way and the listing of this one were two
    # holds of the lock: a Stop and closing that both checked before either
    # listed its close ran two on one controller. One that comes after the
    # other has finished finds nothing to close, and calls nothing either.
    from tools.acquisition.model import app_model as app_model_module

    _system_mode_with_the_fake_laser(app_model, monkeypatch, lasers=_ROUTED_LASER)
    closes = []
    close = app_model.laser.close

    def counted_close():
        closes.append(threading.current_thread().name)
        return close()

    monkeypatch.setattr(app_model.laser, "close", counted_close)
    # Between the look for a close under way and the listing of this one,
    # which read the last commands: two closes that could both be there at
    # once had looked and not listed, and each listed its own. Done in one
    # hold of the lock, the second waits for the lock; the barrier then
    # times out, and the first goes on alone.
    together = threading.Barrier(2, timeout=1.0)
    laser_model_class = type(app_model.laser)
    last_commands = laser_model_class.last_command_volts

    def meet_then_read(model):
        try:
            together.wait()
        except threading.BrokenBarrierError:
            pass
        return last_commands.fget(model)

    if order == "together":
        monkeypatch.setattr(laser_model_class, "last_command_volts", property(meet_then_read))
    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 5.0)
    try:
        stop, stop_outcome = _in_thread(
            app_model._close_laser_within_bound, "System Mode stops")
        if order == "one_after_the_other":
            stop.join(10.0)
        exit_, exit_outcome = _in_thread(
            app_model._close_laser_within_bound, "reachAQ is closing")
        for thread in (stop, exit_):
            thread.join(10.0)
            assert not thread.is_alive()

        assert (stop_outcome, exit_outcome) == ([None], [None])
        assert len(closes) == 1
        assert not app_model.laser.is_connected
    finally:
        app_model.capture_stop()


def test_a_watch_being_listed_is_not_waited_for_as_a_close_under_way(
    app_model, monkeypatch,
):
    # A watch is listed, then given up on at once. A close that looked in
    # between took it for one under way and waited for it, up to its bound,
    # and then went on without closing the laser.
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 3.0)
    _system_mode_with_the_fake_laser(app_model, monkeypatch, lasers=_ROUTED_LASER)
    listed, go_on, ended = threading.Event(), threading.Event(), threading.Event()
    run = app_model_module._BoundedClose.run

    def run_once_listed(closing, timeout):
        if closing.name == "test watch":
            listed.set()
            go_on.wait(5.0)
        return run(closing, timeout)

    monkeypatch.setattr(app_model_module._BoundedClose, "run", run_once_listed)
    watcher, _outcome = _in_thread(lambda: app_model._watch_given_up_laser_close(
        ended.wait, "test watch", ended="The test watch ended", late="test watch ended"))
    assert listed.wait(5.0)
    try:
        started = time.monotonic()
        app_model._close_laser_within_bound("System Mode stops")

        assert time.monotonic() - started < 1.0
        assert not app_model.laser.is_connected
    finally:
        go_on.set()
        ended.set()
        watcher.join(5.0)
        app_model.capture_stop()


def test_laser_reads_failed_while_a_pulse_the_close_gave_up_on_runs(
    app_model, monkeypatch,
):
    # The hung close ended late and its model's disconnect wrote STOPPED
    # "laser controller disconnected"; the late finish then listed the pulse
    # it had given up on as a watch, and, another close being pending, left
    # the status alone: STOPPED, while every laser action was refused.
    from autotrainer.core import LaserChannelId
    from autotrainer.device import LaserPulseTrain, LaserSynchronizedPulseTrain, nidaq_laser
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(nidaq_laser, "_OPERATION_CANCEL_TIMEOUT_S", 0.3)
    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 1.5)
    # A sick driver, whose abort does not end the pulse, and a close that
    # hangs releasing the trigger route.
    daq = _system_mode_with_the_fake_laser(
        app_model, monkeypatch, lasers=_ROUTED_LASER, hang={"disconnect_terms"},
        block_wait=True, hold_waits=True, abort_unblocks=False)
    # Whenever it is asked what the close left running, the refusal: the
    # close was taken off before the watch was listed, and in between laser
    # work read as free with the pulse still in the driver.
    seen = []
    left_running = app_model.laser.work_left_running_after_close

    def asked():
        left = left_running()
        seen.append((bool(left), app_model.laser_controller_close_refusal()))
        return left

    monkeypatch.setattr(app_model.laser, "work_left_running_after_close", asked)
    try:
        app_model.laser.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
            pulse_trains=(LaserPulseTrain(
                channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0,
                duration_ms=1.0),),
            wait=False))
        _wait_for_the_pulse(daq)
        app_model.capture_stop()
        assert "still closing after a driver hang" in _laser_status(app_model).error

        daq.hang_released.set()
        _wait_until(lambda: not app_model.laser_controller_close_refusal().startswith(
            "the laser controller is still closing"))
        time.sleep(0.2)

        status = _laser_status(app_model)
        assert status.state is SubsystemState.FAILED, status
        assert status.error.startswith(
            "a laser operation the close gave up on is still running in the driver")
        assert status.error == app_model.laser_controller_close_refusal()
        assert seen and all(refusal for left, refusal in seen if left), seen
    finally:
        daq.hang_released.set()
        daq.waits_released.set()
        _wait_until(lambda: not app_model.laser_controller_close_refusal())


def test_a_close_within_its_bound_never_reads_stopped(app_model, monkeypatch):
    # The model's disconnect, in the middle of a close still within its
    # bound, wrote STOPPED while laser work was refused as "closing".
    _system_mode_with_the_fake_laser(app_model, monkeypatch, lasers=_ROUTED_LASER)
    written = []
    registry = app_model._acquisition.subsystems
    transition = registry.transition

    def recorded(subsystem_id, state, **fields):
        if getattr(subsystem_id, "value", subsystem_id) == SubsystemId.LASER.value:
            written.append((state, app_model.laser_controller_close_refusal()))
        return transition(subsystem_id, state, **fields)

    monkeypatch.setattr(registry, "transition", recorded)

    app_model.capture_stop()

    assert (SubsystemState.STOPPED, "") in written
    assert not [refusal for state, refusal in written
                if state is SubsystemState.STOPPED and refusal], written


def test_a_close_that_raises_keyboard_interrupt_is_reported(app_model):
    from tools.acquisition.model.app_model import _BoundedClose

    def interrupted():
        raise KeyboardInterrupt

    closing = _BoundedClose(interrupted, "laser controller")

    assert closing.run(2.0) is True
    assert isinstance(closing.error, KeyboardInterrupt)
    assert closing.failure(2.0, True) == "failed to close (KeyboardInterrupt)"


def _trial_laser(monkeypatch):
    """A laser model on the stand-in, synchronized, as a Run gives it."""
    from autotrainer.core import NidaqTimingPlan
    from autotrainer.device import NidaqLaserController, nidaq_laser
    from tools.acquisition.model.laser_model import LaserModel

    fake = _daqmx_fake()
    daq = fake.FakeDaqmx(block_wait=True, hold_waits=True)
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    plan = NidaqTimingPlan(
        requested_mode="auto", resolved_mode="backplane", is_valid=True,
        master_device="PXI1Slot5", slave_devices=("PXI1Slot4",),
        sample_clock_source="/PXI1Slot5/ai/SampleClock", sample_clock_rate_hz=10_000.0,
        start_trigger_source="/PXI1Slot5/ai/StartTrigger",
        hardware_output_devices=("PXI1Slot4",),
        hardware_output_timing_status="declared_not_armed")
    model = LaserModel(NidaqLaserController(
        fake.rig_lasers(**_ROUTED_LASER), timing_plan=plan))
    return daq, model


def test_a_trial_cancel_ends_the_pulse_at_once_and_it_can_be_armed_again(monkeypatch):
    # The trial cancel stopped the armed pulse's task; on the hardware that
    # stop waits behind the pulse's wait for its trigger, the whole timeout
    # (H5a). And not waiting for the cancelled operation's cleanup, the next
    # trial's pulse was refused as the board still held.
    from types import SimpleNamespace

    from tools.acquisition.model.laser_firing import LaserFiring
    from tools.acquisition.model.trial_action import LaserPulseProfile
    from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute

    daq, model = _trial_laser(monkeypatch)
    firing = LaserFiring(1, LaserTriggerRoute.HARDWARE_STIM3, "/PXI1Slot4/PXI_Trig0", 3, 1000)
    profile = LaserPulseProfile("pulse", 1, 2.5, 5)

    def recipe(trial):
        return SimpleNamespace(
            session_id="s", session_generation=1, protocol_id="p", protocol_revision=1,
            logical_trial_id=trial, attempt_id=1, operation_id=f"trial-{trial}")

    try:
        armed = model.prepare_pulse_profile(profile, firing, recipe(1))
        started = time.monotonic()

        AppModel._cancel_protocol_laser(armed)

        assert time.monotonic() - started < 1.0
        assert armed._done.is_set()
        assert armed.state.value == "cancelled"
        again = model.prepare_pulse_profile(profile, firing, recipe(2))
        assert again.state.value == "armed"
    finally:
        daq.waits_released.set()
        model.close()


def test_a_close_whose_look_at_what_it_left_raises_still_lets_laser_work_go(
    app_model, monkeypatch, caplog,
):
    # The finished close was taken off only by the hand-off, and the look
    # at what it left running raising lost the hand-off: listed, it refused
    # laser work until reachAQ restarted.
    _system_mode_with_the_fake_laser(app_model, monkeypatch, lasers=_ROUTED_LASER)

    def broken():
        raise RuntimeError("the controller could not say")

    monkeypatch.setattr(app_model.laser, "work_left_running_after_close", broken)
    with caplog.at_level("ERROR"):
        app_model.capture_stop()

    assert app_model.laser_controller_close_refusal() == ""
    assert any("could not say" in (record.getMessage() + str(record.exc_info))
               for record in caplog.records if record.levelname == "ERROR")
    assert app_model.capture_start() is True
    app_model.capture_stop()


def test_a_close_already_under_way_is_waited_for_not_run_twice(app_model, monkeypatch):
    # Stop and closing can each close the laser: one that came while the
    # other's close was still inside its bound ran a second close on the
    # same controller, at the same time.
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 10.0)
    daq = _system_mode_with_the_fake_laser(
        app_model, monkeypatch, lasers=_ROUTED_LASER, hang={"disconnect_terms"})
    controller = app_model.laser._controller
    close = controller.close
    closes = []

    def counted_close():
        closes.append(threading.current_thread())
        return close()

    monkeypatch.setattr(controller, "close", counted_close)
    try:
        stop_thread, stop_outcome = _in_thread(app_model.capture_stop)
        assert daq.hanging.wait(5.0), "Stop did not reach the driver"
        second, second_outcome = _in_thread(
            app_model._close_laser_within_bound, "reachAQ is closing")
        second.join(0.5)
        assert second.is_alive(), "the second close did not wait for the first"
        assert "is closing" in app_model.laser_controller_close_refusal()

        daq.hang_released.set()
        for thread in (stop_thread, second):
            thread.join(10.0)
            assert not thread.is_alive()
        assert len(closes) == 1
        assert (stop_outcome, second_outcome) == ([None], [None])
        assert app_model.laser_controller_close_refusal() == ""
    finally:
        daq.hang_released.set()
        app_model.capture_stop()


def test_a_pulse_the_close_gave_up_on_holds_laser_work_off_until_it_ends(
    app_model, monkeypatch, caplog,
):
    # close() waits a bounded time for each pulse it cancels, then goes on.
    # One that has not ended can still act: release its clock route, write
    # the PMT line, reset the command, after a new controller has opened on
    # the same lines.
    from autotrainer.core import LaserChannelId
    from autotrainer.device import LaserPulseTrain, LaserSynchronizedPulseTrain, nidaq_laser

    monkeypatch.setattr(nidaq_laser, "_OPERATION_CANCEL_TIMEOUT_S", 0.3)
    # A sick driver, whose abort does not wake the wait: the pulse holds.
    daq = _system_mode_with_the_fake_laser(
        app_model, monkeypatch, block_wait=True, hold_waits=True, abort_unblocks=False)
    try:
        operation = app_model.laser.run_synchronized_pulse_train(
            LaserSynchronizedPulseTrain(
                pulse_trains=(LaserPulseTrain(
                    channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0,
                    duration_ms=1.0),),
                wait=False))
        _wait_for_the_pulse(daq)

        with caplog.at_level("CRITICAL"):
            app_model.capture_stop()

        # The bound still holds where the abort does not end the pulse.
        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        assert "failed to close" in critical.getMessage()
        assert not app_model.laser.is_connected
        refusal = app_model.laser_controller_close_refusal()
        # Not "still closing": the close itself has finished.
        assert refusal.startswith(
            "a laser operation the close gave up on is still running in the driver")
        assert _laser_status(app_model).error == refusal
        assert app_model.capture_start() is False
        assert _refused_without_the_driver(app_model.load_configuration) == (
            f"Loading a configuration is unavailable: {refusal}")

        daq.waits_released.set()
        operation.wait(5.0)
        _wait_until(lambda: not app_model.laser_controller_close_refusal())
        _wait_until(lambda: _laser_status(app_model).state is SubsystemState.STOPPED)
    finally:
        daq.waits_released.set()
        app_model.capture_stop()


def _failing_open(monkeypatch, app_model):
    """A laser controller whose open fails part-way, and whose close then hangs."""
    from autotrainer.device import NidaqLaserController, nidaq_laser

    monkeypatch.setattr(nidaq_laser, "_FAILED_OPEN_CLOSE_TIMEOUT_S", 0.5)
    fake = _daqmx_fake()
    daq = fake.FakeDaqmx(hang={"disconnect_terms"})
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: daq)
    assert app_model.load_configuration() is True
    app_model.laser.set_configuration_offline(fake.rig_lasers(**_ROUTED_LASER))
    connect = NidaqLaserController._connect_trigger_route

    def connect_then_fail(self, channel):
        connect(self, channel)
        raise RuntimeError("DAQmx refused the laser's shutter line")

    monkeypatch.setattr(NidaqLaserController, "_connect_trigger_route", connect_then_fail)
    return daq


@pytest.mark.parametrize("wrapped", [False, True], ids=["raised", "wrapped_once"])
def test_a_run_whose_laser_failed_to_open_waits_for_its_partial_close(
    app_model, monkeypatch, caplog, wrapped,
):
    # The controller closes what it had opened when opening fails, and that
    # close had no bound: a driver hung in it hung the Run start. Bounded
    # now, the laser work behind it is held off until it ends, however the
    # error that says so is wrapped on its way up: read off the top error
    # alone, one "raise ... from" lost it, and the next Run opened a second
    # controller over the close still in the driver.
    daq = _failing_open(monkeypatch, app_model)
    if wrapped:
        load = app_model.laser.load_configuration

        def load_or_wrap(*args, **kwargs):
            try:
                return load(*args, **kwargs)
            except Exception as error:
                raise RuntimeError(f"the laser did not load: {error}") from error

        monkeypatch.setattr(app_model.laser, "load_configuration", load_or_wrap)
    # Every FAILED written for the laser: there were two, the first without
    # the open's own error.
    laser_failures = []
    registry = app_model._acquisition.subsystems
    transition = registry.transition

    def recorded(subsystem_id, state, **fields):
        if (getattr(subsystem_id, "value", subsystem_id) == SubsystemId.LASER.value
                and state is SubsystemState.FAILED):
            laser_failures.append(fields.get("error", ""))
        return transition(subsystem_id, state, **fields)

    monkeypatch.setattr(registry, "transition", recorded)
    caplog.set_level("CRITICAL")
    start_thread, start_outcome = _in_thread(app_model.capture_start)
    try:
        start_thread.join(10.0)
        assert not start_thread.is_alive(), "the Run start hung in the partial close"
        assert start_outcome == [True]
        assert daq.hung == ["disconnect_terms"]
        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        assert "failed to open" in critical.getMessage()
        assert "make the laser safe by hand" in critical.getMessage()
        refusal = app_model.laser_controller_close_refusal()
        assert "still closing after a driver hang" in refusal
        status = _laser_status(app_model)
        assert status.state is SubsystemState.FAILED
        assert "refused the laser's shutter line" in status.error
        assert refusal in status.error
        failure, = laser_failures
        assert failure == status.error

        daq.hang_released.set()
        _wait_until(lambda: not app_model.laser_controller_close_refusal())
    finally:
        daq.hang_released.set()
        start_thread.join(30.0)
        app_model.capture_stop()


def test_closing_reachaq_with_a_hung_laser_close_is_bounded(app_model, monkeypatch, caplog):
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CONTROLLER_CLOSE_S", 0.5)
    daq = _system_mode_with_the_fake_laser(
        app_model, monkeypatch, lasers=_ROUTED_LASER, hang={"disconnect_terms"})
    try:
        with caplog.at_level("CRITICAL"):
            close_thread, close_outcome = _in_thread(app_model.on_close)
            close_thread.join(20.0)

        assert not close_thread.is_alive()
        assert close_outcome == [None]
        # Once: the close that on_close makes itself is not started again
        # over the one still in the driver.
        assert daq.hung == ["disconnect_terms"]
        critical, = [record for record in caplog.records
                     if record.levelname == "CRITICAL"]
        assert "laser 1" in critical.getMessage()
    finally:
        daq.hang_released.set()


def test_a_normal_laser_close_is_unchanged(app_model, monkeypatch, caplog):
    daq = _system_mode_with_the_fake_laser(app_model, monkeypatch, lasers=_ROUTED_LASER)

    with caplog.at_level("ERROR"):
        app_model.capture_stop()

    assert not app_model.laser.is_connected
    assert app_model.laser_controller_close_refusal() == ""
    assert daq.disconnected == [("/PXI1Slot5/PFI0", "/PXI1Slot5/PXI_Trig0")]
    assert not any(record.levelname == "CRITICAL" for record in caplog.records)
    assert app_model.capture_start() is True
    app_model.capture_stop()


def _wait_for_the_pulse(daq, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not any(task.label == "laser_sync_pulse_ao" and task.started
                  for task in daq.tasks):
        assert time.monotonic() < deadline, "the pulse did not start"
        time.sleep(0.01)


def test_stop_mid_pulse_cancels_the_pulse_and_resets_the_laser(
    app_model, monkeypatch, caplog,
):
    # Run Pulse waits for its train, and ran with no operation the laser
    # controller could cancel. System Mode's Stop closed the controller under
    # it: the command reset met the train's task at -50103, and the train ran
    # on to its end with the shutter left to it.
    from autotrainer.core import LaserChannelId
    from autotrainer.device import LaserPulseTrain

    daq = _system_mode_with_the_fake_laser(
        app_model, monkeypatch, block_wait=True, hold_waits=True)
    try:
        pulse_thread, pulse_outcome = _in_thread(
            app_model.laser.run_pulse_train,
            LaserPulseTrain(channel_id=LaserChannelId.LASER_1,
                            amplitude_volts=1.0, duration_ms=1.0))
        _wait_for_the_pulse(daq)
        started = len(daq.writes)

        with caplog.at_level("ERROR"):
            stop_thread, stop_outcome = _in_thread(app_model.capture_stop)
            stop_thread.join(20.0)

        assert not stop_thread.is_alive()
        assert stop_outcome == [None]
        pulse_thread.join(5.0)
        error, = pulse_outcome
        assert "cancelled" in str(error) and "-50103" not in str(error)
        # The close's own reset, not the pulse's: made on the thread Stop
        # closes the controller on.
        assert [write.data for write in daq.writes[started:]
                if write.channels == ("PXI1Slot4/ao0",)
                and write.task.endswith("manual_ao")
                and write.thread is not pulse_thread] == [0.0]
        assert daq.task("laser_1_shutter").writes[-1] is False
        assert not any("-50103" in record.getMessage()
                       or "Failed to close" in record.getMessage()
                       or record.levelname == "CRITICAL"
                       for record in caplog.records)
    finally:
        daq.waits_released.set()
        app_model.capture_stop()


def test_a_manual_run_pulse_stopped_by_system_mode_is_recorded_as_cancelled(
    app_model, monkeypatch,
):
    # Recorded as "failed", operation_failure: the Stop's cancel came back as
    # a plain RuntimeError, like any failure of the train.
    import json

    from autotrainer.core import LaserChannelId
    from autotrainer.device import LaserPulseTrain

    daq = _system_mode_with_the_fake_laser(
        app_model, monkeypatch, block_wait=True, hold_waits=True)
    told = []
    app_model.laser.trace_received += told.append
    try:
        pulse_thread, pulse_outcome = _in_thread(
            lambda: app_model.laser.run_pulse_train(
                LaserPulseTrain(channel_id=LaserChannelId.LASER_1,
                                amplitude_volts=1.0, duration_ms=1.0),
                manual_context={"profile_id": "burst", "profile_revision": 1,
                                "trigger_mode": "internal"}))
        _wait_for_the_pulse(daq)

        stop_thread, stop_outcome = _in_thread(app_model.capture_stop)
        stop_thread.join(20.0)
        assert not stop_thread.is_alive()
        pulse_thread.join(5.0)

        events = [trace for trace in told if trace.source == "manual pulse"]
        assert [trace.event for trace in events] == ["requested", "cancelled"]
        cancelled = events[-1]
        assert cancelled.timing_confidence == "operation_cancelled"
        assert json.loads(cancelled.context_json)["error_class"] == "LaserPulseCancelled"
        # No waveform: a cancelled train is not one that ran.
        assert not [trace for trace in told if trace.source == "internal pulse"]
        from autotrainer.device.laser import LaserPulseCancelled

        error, = pulse_outcome
        assert isinstance(error, LaserPulseCancelled)
        assert "the laser controller was closed while it ran" in str(error)
    finally:
        daq.waits_released.set()
        app_model.capture_stop()


def test_acquisition_owns_configured_signal_stream_lifecycle(app_model, monkeypatch):
    assert app_model.load_configuration() is True
    monitor = app_model.nidaq_signal_monitor
    monitor._configuration = NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration(
                name="cam_frames",
                physical_channel="Dev1/port0/line0",
                kind="digital",
            ),
        ),
        is_enabled=True,
    )
    monitor.set_hardware_enabled(True)
    calls = []
    monkeypatch.setattr(monitor, "start", lambda: calls.append("start") or True)
    monkeypatch.setattr(monitor, "stop", lambda: calls.append("stop"))

    assert app_model.capture_start() is True
    app_model.capture_stop()
    # Back in Idle the stream starts again by itself, in the background.
    rule = vars(app_model).get("_nidaq_stream_autostart")
    if rule is not None:
        assert rule.wait(5.0)

    assert calls[0] == "start"
    assert "stop" in calls[1:]
    assert calls[-1] == "start"
    assert calls.index("stop") < len(calls) - 1


def test_gpu_preflight_failure_does_not_block_cameras_and_hardware(
    app_model,
    system_config,
    trainer_config_dir,
    monkeypatch,
):
    system_config.inference.is_enabled = True
    for camera in system_config.cameras:
        if camera.id in CameraId.reach_camera_ids():
            camera.is_enabled = True
    system_config.save_default(trainer_config_dir)
    assert app_model.load_configuration() is True

    monkeypatch.setattr(
        app_model.inference,
        "check_live_inference_runtime",
        lambda: GpuRuntimeStatus(
            False,
            backend="nvidia-driver",
            error="active kernel driver is nouveau",
        ),
    )

    start_reach = mock.Mock(return_value=True)
    start_can = mock.Mock(return_value=False)
    monkeypatch.setattr(app_model, "_start_reach_camera_domains", start_reach)
    monkeypatch.setattr(app_model, "_start_can_domain", start_can)

    assert app_model.capture_start() is True

    start_reach.assert_called_once()
    start_can.assert_called_once_with(wait_connected=True)
    assert app_model.acquisition_started is True
    inference = app_model.subsystem_statuses[SubsystemId.LIVE_INFERENCE.value]
    assert inference.state is SubsystemState.FAILED
    assert "nouveau" in inference.error


def test_hardware_start_failure_performs_can_safety_shutdown(
    app_model,
    monkeypatch,
):
    assert app_model.load_configuration() is True
    failure = RuntimeError("connection timeout")
    monkeypatch.setattr(app_model.hardware, "connect", mock.Mock(side_effect=failure))
    safety_shutdown = mock.Mock()
    monkeypatch.setattr(app_model.hardware, "safety_shutdown", safety_shutdown)

    assert app_model.capture_start() is True

    safety_shutdown.assert_any_call(
        "CAN/pellet initialization failure: connection timeout",
        wait=True,
    )
    can_status = app_model.subsystem_statuses[SubsystemId.CAN_PELLET.value]
    assert can_status.state is SubsystemState.FAILED
    assert can_status.error == "connection timeout"
    assert app_model.acquisition_started is True
    assert app_model._acquisition.starting is False


def test_periodic_command_producers_stop_without_touching_can():
    events = []
    app_model = object.__new__(AppModel)
    app_model._closing_event = threading.Event()
    app_model._timer_one_minute_repeat = mock.Mock(
        cancel=lambda: events.append("cancel-minute"),
    )
    app_model._timer_daily = mock.Mock(
        cancel=lambda: events.append("cancel-daily"),
    )
    app_model._hardware = mock.Mock()

    app_model._prepare_application_shutdown()

    assert app_model._closing_event.is_set()
    assert events == ["cancel-minute", "cancel-daily"]
    app_model._hardware.safety_shutdown.assert_not_called()
    app_model._hardware.disconnect.assert_not_called()


def test_generic_fatal_callback_is_diagnostic_only(app_model):
    app_model._hardware = mock.Mock()
    app_model.stop_recording = mock.Mock()
    app_model.abort_recording = mock.Mock()
    app_model._recording_session.status = SessionRecordingStatus.RECORDING

    app_model._on_fatal_exception("camera.left", RuntimeError("capture failed"))
    first = app_model.internal_error_diagnostic
    app_model._on_fatal_exception("later", RuntimeError("second failure"))

    app_model._hardware.safety_shutdown.assert_not_called()
    app_model._hardware.disconnect.assert_not_called()
    app_model.stop_recording.assert_not_called()
    app_model.abort_recording.assert_not_called()
    assert app_model.internal_error_diagnostic == first
    assert first["message"] == "capture failed"
    assert first["operationChanged"] is False
    assert app_model._recording_session.status is SessionRecordingStatus.RECORDING
    assert all("internal" not in blocker.lower() for blocker in app_model.recording_blockers)


def test_live_inference_override_is_not_persisted_with_other_configuration_changes(
    app_model,
    system_config,
    trainer_config_dir,
):
    system_config.inference.is_enabled = True
    system_config.save_default(trainer_config_dir)
    assert app_model.load_configuration() is True

    app_model.set_runtime_live_inference_override(False)
    app_model.inference.model_location = "updated-model-location"

    saved = app_model._create_configuration()
    assert app_model.inference.is_enabled is False
    assert saved.inference.is_enabled is True
    assert saved.inference.pose_model_location == "updated-model-location"


def test_cli_help(app_launch_env):
    output = subprocess.check_output(
        [sys.executable, "-m", "tools.acquisition.headless", "-h"], env=app_launch_env).decode()
    assert "usage: " in output
    assert "--random-cameras" in output


def test_load_config_random_camera_override(app_model, trainer_config_dir, system_config):
    for cam in system_config.cameras:
        if cam.id in (CameraId.Left, CameraId.Right):
            cam.scheme = "spinnaker"
            cam.path = cam.name
            cam.params = {"fps": 150}
            cam.is_enabled = True
    system_config.save_default(trainer_config_dir)

    assert app_model.load_configuration(random_cameras=True) is True

    for cam in (app_model.left_camera, app_model.right_camera):
        assert cam.is_enabled
        assert cam.camera_source.url.startswith("random://")
        assert "fps=150" in cam.camera_source.url
    assert not app_model.stim_camera.is_enabled
    assert app_model.stim_camera.camera_source.url.startswith("random://")
    assert app_model.left_camera.is_primary
    assert not app_model.right_camera.is_primary


def test_load_config_random_camera_override_adds_default_reach_cameras(app_model, trainer_config_dir, system_config):
    system_config.cameras = [
        cam for cam in system_config.cameras
        if cam.id not in CameraId.reach_camera_ids()
    ]
    system_config.save_default(trainer_config_dir)

    assert app_model.load_configuration(random_cameras=True) is True

    assert tuple(cam.camera_id for cam in app_model.reach_cameras) == (
        CameraId.Left,
        CameraId.Right,
        CameraId.Camera3,
    )
    assert all(cam.is_enabled for cam in app_model.reach_cameras[:2])
    assert not app_model.stim_camera.is_enabled
    assert all(cam.camera_source.url.startswith("random://") for cam in app_model.reach_cameras)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="hang atm. mostlikely signal related, different on windows")
@pytest.mark.parametrize("record_mode", list(VideoRecordMode))
def test_launch_cli(system_config, config_file_path, user_pref, calib_dir, diamond_config_path, settings_ini_path, record_mode,
                    app_launch_env):
    user_pref.save()  # do not forget ! otherwise default home config dirs/files are used
    for cam in system_config.cameras:
        cam.is_enabled = True
        cam.is_record_enabled = True
        cam.record_mode = record_mode.value
    system_config.save_file(config_file_path, as_yaml=True)
    env = app_launch_env  # its own HOME, so its default data folder is under tmp too
    env['AUTOTRAINER_DIAMOND_TRIANGLE_CONFIG'] = diamond_config_path.as_posix()  # same for this !
    env['AUTOTRAINER_FORCE_CAN_EMULATION_IFACE'] = "1"
    env['AUTOTRAINER_CAN_TRANSPORT'] = "emulation"
    proc = subprocess.Popen([
        sys.executable,
        "-m", "tools.acquisition.headless",
        "-c", config_file_path.as_posix(),
        "--preferences-file", settings_ini_path.as_posix(),
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1)
    #
    app_running = threading.Event()
    out_lines = []
    err_lines = []
    def interrupt_proc():
        app_running.wait(60)
        if proc.poll() is None:
            time.sleep(1)  # give 1s of running time
            logging.info("signal main app with sig interrupt")
            proc.send_signal(signal.SIGINT)
        else:
            logging.warning("app exited unexpectedly")
    interrupt_proc_when_running = threading.Thread(target=interrupt_proc, daemon=True)
    interrupt_proc_when_running.start()
    def communicate(dest, src_fh, dest_fh):
        while proc.poll() is None:
            line = src_fh.readline()  # .decode()
            line = line.strip(b'\n').decode()
            if "App is now running" in line:
                app_running.set()
            print(line, file=dest_fh)
            dest.append(remove_ansi_escape_sequences(line))
        logging.debug("process terminated, reading remaining data on %s", src_fh.name)
        tail = src_fh.read().strip(b'\n').decode()
        print(tail, file=dest_fh)
        dest.extend(tail.split("\n"))

    # use communicate threads, using communicate() might block if process is stuck or smth.
    communicate_out_thread = threading.Thread(target=communicate, daemon=True, args=(out_lines, proc.stdout, sys.stdout))
    communicate_out_thread.start()
    communicate_err_thread = threading.Thread(target=communicate, daemon=True, args=(err_lines, proc.stderr, sys.stderr))
    communicate_err_thread.start()

    interrupt_proc_when_running.join()

    # main app takes quite a bit to finishes/exit once asked to do so, give it at most 15s
    proc.wait(15)
    if proc.poll() is None:
        logging.warning("main app still running, terminating ..")
        proc.terminate()  # send terminate signal
        proc.wait(3)
        if proc.poll() is None:
            # send kill signal
            logging.error("main app still running after terminate, killing ..")
            proc.kill()
            proc.wait()

    # can now wait/join on communicate threads
    communicate_out_thread.join()
    communicate_err_thread.join()

    output = "\n".join(out_lines)

    def assert_is_present(content):
        assert any(content in line for line in out_lines), output

    assert proc.returncode == 0, (out_lines, err_lines)

    assert_is_present(f"Loading diamond-triangle file {diamond_config_path.as_posix()!r}")
    assert_is_present(f"Using setting ini file: {settings_ini_path.as_posix()!r}")
    assert_is_present("Pellet-board hardware or transport support not found. Using emulation interface.")
    assert_is_present(f"Writing to {config_file_path.as_posix()!r}")
    # It opened its startup log under that HOME, not the operator's.
    assert Path(env["HOME"]).joinpath("Documents", "rawdatalocal").is_dir()
    #
    # etc...



# ------------------------------------------------ the final fix round


def test_stop_as_a_pulse_fails_by_itself_logs_no_critical(app_model, monkeypatch, caplog):
    # The pulse's own failure, reached between close() taking the live
    # pulses and cancelling them, was reported as the close's: Stop logged
    # "make the laser safe by hand" for a laser its cleanup had reset.
    from autotrainer.device import (
        LaserChannelId, LaserPulseTrain, LaserSynchronizedPulseTrain)

    _system_mode_with_the_fake_laser(
        app_model, monkeypatch, lasers=_ROUTED_LASER, block_wait=True)
    controller = app_model.laser._controller
    operation = controller.run_synchronized_pulse_train(LaserSynchronizedPulseTrain(
        pulse_trains=(LaserPulseTrain(
            channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0, duration_ms=1.0),),
        trigger_source="/PXI1Slot4/PXI_Trig0", wait=False, timeout_seconds=0.2))
    mark_closed = controller._mark_closed

    def the_pulse_fails_meanwhile():
        taken = mark_closed()
        assert operation.wait_until_finished(5.0)
        return taken

    monkeypatch.setattr(controller, "_mark_closed", the_pulse_fails_meanwhile)
    try:
        with caplog.at_level("ERROR"):
            app_model._close_laser_within_bound("System Mode stops")

        assert operation.state.value == "failed"
        assert [record.getMessage() for record in caplog.records
                if record.levelname == "CRITICAL"] == []
        assert not app_model.laser.is_connected
    finally:
        app_model.capture_stop()


def test_a_bounded_close_names_the_first_line_of_text_of_its_error():
    # The first line as split: a message that starts on a new line, as a
    # DAQmx one can, gave an empty line and then only the error's class.
    from tools.acquisition.model import app_model as app_model_module

    closing = app_model_module._BoundedClose(lambda: None, "test close")
    closing.error = RuntimeError("\nDAQmx refused the reset.\n\nStatus Code: -50103")

    assert closing.failure(1.0, True) == "failed to close (DAQmx refused the reset.)"
