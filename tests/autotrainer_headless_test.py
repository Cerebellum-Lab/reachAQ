import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time

import pytest

from pathlib import Path

import verboselogs

from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.core import SystemConfiguration, CameraConfiguration, CameraId
from autotrainer.video import VideoRecordMode
from autotrainer.inference import GpuRuntimeStatus
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import AppModelStatus
from tools.acquisition.model.user_preferences import UserPreferences


import top_fixtures


def remove_ansi_escape_sequences(s):
    # Regex for common ANSI escape codes
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', s)


@pytest.fixture
def system_config(trainer_config_dir, tmp_path):
    config = SystemConfiguration()
    for cam_member in (CameraId.Left, CameraId.Right, CameraId.Web):
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
    #
    app = AppModel(user_pref, calib_dir=calib_dir)
    try:
        yield app  # noqa
    finally:
        app.capture_stop(force=True)
        app.on_close()


def test_user_preferences(settings_ini_path, user_pref, trainer_config_dir):
    assert Path(user_pref._settings.fileName()) == settings_ini_path
    assert not settings_ini_path.exists()
    user_pref.selected_animal = "foobar"
    user_pref.save()
    assert settings_ini_path.exists()
    user_pref = UserPreferences(settings_file_path=settings_ini_path)
    assert Path(user_pref.configuration_location) == trainer_config_dir
    assert user_pref.selected_animal == "foobar"


def test_load_config(app_model, trainer_config_dir, animals_dir, calib_dir, system_config):
    res = app_model.load_configuration()
    assert res is True
    assert app_model.left_camera.name == "left"
    assert app_model.right_camera.name == "right"
    assert len(app_model.reach_cameras) == 2
    assert app_model.get_camera_model(CameraId.Camera3) is None
    assert app_model.top_camera.name == "web"
    assert app_model.output_location == system_config.persistence.output_location
    pref = app_model.preferences
    assert Path(pref.animal_location) == animals_dir
    assert Path(pref.configuration_location) == trainer_config_dir


def test_load_config_extra_reach_camera_slot(app_model, trainer_config_dir, system_config):
    camera3 = CameraConfiguration(
        id=CameraId.Camera3,
        name=CameraId.Camera3.name,
        is_enabled=True,
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
    assert loaded_camera3.name == "camera3"
    assert app_model.make_project_info().camera_names == ("camera3",)


def test_load_config_without_web_camera_keeps_top_disabled(app_model, trainer_config_dir, system_config):
    system_config.cameras = [
        cam for cam in system_config.cameras
        if cam.id != CameraId.Web
    ]
    system_config.save_default(trainer_config_dir)

    assert app_model.load_configuration() is True

    assert app_model.top_camera.name == "web"
    assert not app_model.top_camera.is_enabled


def test_start_stop(app_model, settings_ini_path):
    assert not settings_ini_path.exists()
    assert app_model.load_configuration() is True
    assert app_model.capture_start() is True
    app_model.capture_stop()
    assert not settings_ini_path.exists()  # still
    app_model.on_close()
    assert settings_ini_path.exists()  # but saved on close
    # ...


def test_gpu_preflight_fails_before_cameras_and_hardware(
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

    def unexpected_hardware_start(*_args, **_kwargs):
        raise AssertionError("hardware must not start after a failed GPU preflight")

    monkeypatch.setattr(app_model.hardware, "connect", unexpected_hardware_start)
    for camera in app_model.cameras:
        monkeypatch.setattr(camera, "on_prepare_capture", unexpected_hardware_start)

    assert app_model.capture_start() is False
    assert app_model.acquisition_started is False


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

    for cam in app_model.reach_cameras:
        assert cam.is_enabled
        assert cam.camera_source.url.startswith("random://")
        assert "fps=150" in cam.camera_source.url
    assert app_model.left_camera.is_primary
    assert not app_model.right_camera.is_primary


def test_load_config_random_camera_override_adds_default_reach_cameras(app_model, trainer_config_dir, system_config):
    system_config.cameras = [
        cam for cam in system_config.cameras
        if cam.id not in CameraId.reach_camera_ids()
    ]
    system_config.save_default(trainer_config_dir)

    assert app_model.load_configuration(random_cameras=True) is True

    assert tuple(cam.camera_id for cam in app_model.reach_cameras) == (CameraId.Left, CameraId.Right)
    assert all(cam.is_enabled for cam in app_model.reach_cameras)
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
    assert_is_present("Alogus hardware or hardware support not found. Using emulation interface.")
    assert_is_present(f"Writing to {config_file_path.as_posix()!r}")
    #
    # etc...
