import logging
import os
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
        steps=3, samples_per_step=10, timeout_seconds=timeout_seconds)


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
    # past it a CRITICAL names the laser and the ramp's last command, and
    # reachAQ goes on closing.
    from tools.acquisition.model import app_model as app_model_module

    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_CLOSE_MARGIN_S", 0.5)
    monkeypatch.setattr(app_model_module, "_LASER_CALIBRATION_FORCED_CLOSE_S", 0.5)
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


def test_cli_help():
    output = subprocess.check_output([sys.executable, "-m", "tools.acquisition.headless", "-h"]).decode()
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
def test_launch_cli(system_config, config_file_path, user_pref, calib_dir, diamond_config_path, settings_ini_path, record_mode):
    user_pref.save()  # do not forget ! otherwise default home config dirs/files are used
    for cam in system_config.cameras:
        cam.is_enabled = True
        cam.is_record_enabled = True
        cam.record_mode = record_mode.value
    system_config.save_file(config_file_path, as_yaml=True)
    env = os.environ.copy()
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
    #
    # etc...
