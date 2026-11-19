"""The rig configuration a `--simulate` run generates in its sim home.

It is built in code, as tests/autotrainer_headless_test.py's system_config
fixture builds one, and saved with SystemConfiguration's own save so it
round-trips through the YAML loader like any rig's file. The NI lines are set
as port roles and the stream channels come from the channel plan builder, so
loading it runs the same plan path a rig's configuration does.
"""

from pathlib import Path
from typing import Union

from autotrainer.core import (
    CameraConfiguration,
    CameraId,
    LaserChannelConfiguration,
    LaserSystemConfiguration,
    NidaqDeviceIdentity,
    NidaqPortConfiguration,
    NidaqSignalStreamConfiguration,
    SystemConfiguration,
)
from autotrainer.core.configuration import NidaqDeliveryCalibration
from autotrainer.device.nidaq_simulated import SIM_NIDAQ_DEVICE

from tools.acquisition.model.nidaq_channel_plan import build_nidaq_acquisition_configuration

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 150
#: Microseconds, as Spinnaker's ExposureTime and the sim camera's URL read it.
CAMERA_EXPOSURE_US = 2000

#: As the reference rig's stream (tests/christielab10_nidaq_blocks.yaml).
NIDAQ_SAMPLE_RATE_HZ = 10000.0
NIDAQ_READ_CHUNK_SIZE = 500
#: The laser sample rate the SimDAQ1 laser shape was checked at.
LASER_SAMPLE_RATE_HZ = 10000.0
#: The rig's is 5 s. The pose and pose-data processes import TensorFlow and
#: torch as they spawn, which alone took 5.1-5.7 s from WSL's /mnt/c.
WATCHDOG_TIMEOUT_S = 20.0


def _line(terminal: str) -> str:
    return f"{SIM_NIDAQ_DEVICE}/{terminal}"


def build_simulation_configuration(home: Union[str, Path]) -> SystemConfiguration:
    """The simulated rig: three sim cameras, SimDAQ1, one NI laser, CAN emulation."""
    home = Path(home).expanduser()
    config = SystemConfiguration()

    for camera_id in (CameraId.Left, CameraId.Right, CameraId.Camera3):
        params = dict(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            exposure=CAMERA_EXPOSURE_US,
            primary="yes" if camera_id is CameraId.Left else "no",
        )
        camera = CameraConfiguration(
            name=camera_id.name, params=params, is_enabled=True, is_record_enabled=True)
        camera.scheme = "sim"
        camera.id = camera_id
        config.cameras.append(camera)

    ports = NidaqPortConfiguration(
        device_name=SIM_NIDAQ_DEVICE,
        tone1=_line("port0/line0"),
        tone2=_line("port0/line1"),
        cam_frames=_line("port0/line2"),
        device_identities=(NidaqDeviceIdentity(
            logical_name=SIM_NIDAQ_DEVICE,
            runtime_name=SIM_NIDAQ_DEVICE,
            product_type="Simulated",
        ),),
    )
    # No triggerSource or boardStimLine: the laser fires on the direct NI
    # software route only. The clock lines stay at their defaults.
    laser = LaserSystemConfiguration.from_channels(
        (LaserChannelConfiguration(
            channel_id=1,
            analog_output=_line("ao0"),
            diode_input=_line("ai0"),
            command_copy_input=_line("ai1"),
            shutter_output=_line("port0/line10"),
        ),),
        backend="nidaq",
        hardware_timed=True,
        sample_rate_hz=LASER_SAMPLE_RATE_HZ,
    )
    # Offset half a sample, spread 0.3 of one; the camera-latch pairing needs
    # a calibration of at least two sessions.
    calibration = NidaqDeliveryCalibration(
        offset_seconds=0.5 / NIDAQ_SAMPLE_RATE_HZ,
        spread_seconds=0.3 / NIDAQ_SAMPLE_RATE_HZ,
        sessions=3,
        measured="simulation",
    )
    config.nidaq_ports = ports
    config.laser = laser
    config.nidaq_stream = build_nidaq_acquisition_configuration(
        NidaqSignalStreamConfiguration(
            sample_rate_hz=NIDAQ_SAMPLE_RATE_HZ,
            read_chunk_size=NIDAQ_READ_CHUNK_SIZE,
            delivery_calibration=calibration,
        ),
        ports,
        laser,
    )
    # The stream is acquired only with the rig's NI-DAQ hardware switched on.
    config.hardware.nidaq_enabled = True
    config.hardware.can_enabled = True
    config.hardware.pellet_controller_enabled = True
    # An empty model location selects MemoryPoseModel.
    config.inference.is_enabled = True
    config.inference.pose_model_location = ""
    config.persistence.output_location = home.joinpath("Data").as_posix()
    config.watchdog.timeout_trigger_delay = WATCHDOG_TIMEOUT_S
    return config
