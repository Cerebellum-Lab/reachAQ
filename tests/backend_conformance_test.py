"""Each backend contract against its real backend (driver faked) and its simulated one.

One script per contract (docs/acquisition/backend-interfaces.md), run on
both backends: the protocol's isinstance, then the calls the app makes, in
its order, checked for the semantics its docstring states. The real
backends run on the same driver fakes their own tests use, so nothing here
touches hardware.
"""

import importlib
import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy
import pytest

from autotrainer.core import (
    LaserChannelId,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)
from autotrainer.core.simulation import SimSignalBusReader
from autotrainer.device import CanInterface, CanTransportConfiguration, LaserPulseTrain
from autotrainer.device import nidaq_laser, nidaq_signal_stream
from autotrainer.device.device_interface import Acknowledge, DigitalOutputs
from autotrainer.device.emulation_interface import EmulationInterface
from autotrainer.device.laser import LaserControllerProtocol, SimulatedLaserController
from autotrainer.device.motor_pellet_backend import MotorPelletBackend
from autotrainer.device.nidaq_input_backend import NidaqInputBackend
from autotrainer.device.nidaq_signal_stream import NidaqSignalStreamController
from autotrainer.device.nidaq_simulated import SimulatedNidaqStream
from autotrainer.video.camera.camera_backend import CameraBackend
from autotrainer.video.camera.camera_base import CameraBase
from autotrainer.video.camera.sim_cam import SimulatedCam

DEVICE_TESTS = Path(__file__).resolve().parents[1] / "auto-trainer-device" / "tests"
WIDTH, HEIGHT = 64, 48


def _device_test_module(name):
    """A driver fake from auto-trainer-device/tests, where it lives beside its own tests."""
    if str(DEVICE_TESTS) not in sys.path:
        sys.path.insert(0, str(DEVICE_TESTS))
    return importlib.import_module(name)


@pytest.fixture
def bus(tmp_path, monkeypatch):
    path = tmp_path / "signals.jsonl"
    monkeypatch.setenv("REACHAQ_SIM_BUS", str(path))
    return SimSignalBusReader(path)


# ------------------------------------------------------------------ camera

class _FakeImage:
    def __init__(self, frame_id, stamp_ns):
        self._frame_id, self._stamp_ns = frame_id, stamp_ns

    def IsIncomplete(self):  # noqa: N802 - PySpin's names
        return False

    def GetFrameID(self):  # noqa: N802
        return self._frame_id

    def GetTimeStamp(self):  # noqa: N802
        return self._stamp_ns

    def GetNDArray(self):  # noqa: N802
        return numpy.full((HEIGHT, WIDTH), self._frame_id % 256, dtype=numpy.uint8)

    def Release(self):  # noqa: N802
        pass


class _FakeSpinCamera:
    """A secondary FLIR camera: each image a frame later on a perf-ns clock."""

    def __init__(self):
        self._frame_id = 100
        self.TriggerMode = types.SimpleNamespace(SetValue=lambda value: None)

    def BeginAcquisition(self):  # noqa: N802
        pass

    def EndAcquisition(self):  # noqa: N802
        pass

    def DeInit(self):  # noqa: N802
        pass

    def GetNextImage(self, _timeout_ms):  # noqa: N802
        self._frame_id += 1
        return _FakeImage(self._frame_id, time.perf_counter_ns())


class _FakeLatchNodes:
    """TimestampLatch / TimestampLatchValue on the camera clock (perf ns here)."""

    def __init__(self):
        self._latched = 0
        self._nodes = {
            "TimestampLatch": types.SimpleNamespace(Execute=self._latch),
            "TimestampLatchValue": types.SimpleNamespace(GetValue=lambda: self._latched),
        }

    def _latch(self):
        self._latched = time.perf_counter_ns()

    def GetNode(self, name):  # noqa: N802
        return self._nodes.get(name)


def _fake_pyspin():
    module = types.ModuleType("PySpin")
    module.SpinnakerException = type("SpinnakerException", (Exception,), {})
    module.SPINNAKER_ERR_TIMEOUT = -1011
    module.TriggerMode_Off = 0
    module.CCommandPtr = module.CIntegerPtr = lambda node: node
    module.IsAvailable = module.IsWritable = module.IsReadable = lambda node: True
    # Anything else (the type annotations SpinCam's methods name) is a placeholder.
    module.__getattr__ = lambda name: type(name, (), {})
    return module


@pytest.fixture
def spin_cam(monkeypatch):
    """SpinCam on a faked PySpin and camera, built as spinnaker_cam_test builds one."""
    try:
        import PySpin  # noqa: F401 - the vendor SDK, where it is installed
        faked = False
    except ImportError:
        monkeypatch.setitem(sys.modules, "PySpin", _fake_pyspin())
        faked = True
    spinnaker_cam = importlib.import_module("autotrainer.video.camera.spinnaker_cam")
    cam = spinnaker_cam.SpinCam.__new__(spinnaker_cam.SpinCam)
    CameraBase.__init__(cam, "right")
    cam._serial_number = "fake"
    cam._camera = _FakeSpinCamera()
    cam._node_map = _FakeLatchNodes()
    cam._start_frames = []
    cam._current_cam_frame_2_perf_offset = cam._current_cam_frame_2_time_offset = float("nan")
    cam._consecutive_late_acquire = 0
    cam._is_primary, cam._is_secondary = False, True
    cam._acquisition_started = False
    cam._skip_duplicate_frame_copy = False
    cam._width, cam._height, cam._fps = WIDTH, HEIGHT, 150
    yield cam
    cam.end_capture()
    if faked:
        # Later tests must find Spinnaker missing again, not this module bound to the fake.
        sys.modules.pop(spinnaker_cam.__name__, None)
        camera_package = sys.modules["autotrainer.video.camera"]
        if getattr(camera_package, "spinnaker_cam", None) is spinnaker_cam:
            delattr(camera_package, "spinnaker_cam")


@pytest.fixture
def sim_cam(monkeypatch):
    monkeypatch.setenv("REACHAQ_SIM_EPOCH", repr(time.perf_counter()))
    cam = SimulatedCam("right")
    for name, value in (("width", WIDTH), ("height", HEIGHT), ("fps", 150), ("exposure", 2000)):
        assert cam.set_property(name, str(value))
    cam.init()
    cam.prepare_capture()
    yield cam
    cam.end_capture()


@pytest.mark.parametrize("backend", ["spin_cam", "sim_cam"])
def test_camera_backend_contract(request, backend):
    cam = request.getfixturevalue(backend)
    assert isinstance(cam, CameraBackend)
    assert cam.set_property("no_such_property", "1") is False
    assert cam.is_trigger_synchronized

    ids = []
    for _ in range(3):
        frame, stamp = cam.capture()
        assert frame.dtype == numpy.uint8 and frame.shape == (cam.height, cam.width)
        assert isinstance(stamp, int)
        assert cam.frame_poll_perf_c <= cam.frame_arrival_perf_c <= time.perf_counter()
        ids.append(cam.frame_id)
    assert ids == sorted(ids) and len(set(ids)) == 3

    latch = cam.latch_clock()
    perf_before, camera_ns, perf_after, _raw_before, _raw_after = latch
    # Both clocks here are perf_counter, so the latched value sits in its bracket.
    assert perf_before <= camera_ns / 1e9 <= perf_after


# ------------------------------------------------------------------ NI-DAQ input

def _stream_configuration():
    return NidaqSignalStreamConfiguration(
        channels=(
            NidaqSignalChannelConfiguration("cam_frames", "Dev1/port0/line2", kind="digital"),
            NidaqSignalChannelConfiguration("tone1", "Dev1/port0/line0", kind="digital"),
        ),
        is_enabled=True,
        sample_rate_hz=10_000.0,
        read_chunk_size=50,
    )


@pytest.fixture
def real_stream(monkeypatch):
    fakes = _device_test_module("nidaq_signal_stream_test")

    class _VerifyingNidaqmx(fakes._FakeNidaqmx):
        """The stream test's fake, with the task modes verify_tasks uses."""
        constants = types.SimpleNamespace(
            **vars(fakes._FakeNidaqmx.constants),
            TaskMode=types.SimpleNamespace(TASK_VERIFY="verify", TASK_COMMIT="commit"))

        def Task(self, name):  # noqa: N802 - nidaqmx's name
            task = super().Task(name)
            task.control = lambda mode: None
            return task

    daq = _VerifyingNidaqmx()
    monkeypatch.setattr(nidaq_signal_stream, "_load_nidaqmx", lambda: daq)
    stream = NidaqSignalStreamController(_stream_configuration())
    yield stream
    stream.close()


@pytest.fixture
def sim_stream(monkeypatch):
    monkeypatch.setenv("REACHAQ_SIM_EPOCH", repr(time.perf_counter()))
    monkeypatch.delenv("REACHAQ_SIM_BUS", raising=False)
    stream = SimulatedNidaqStream(_stream_configuration())
    yield stream
    stream.close()


@pytest.mark.parametrize("backend", ["real_stream", "sim_stream"])
def test_nidaq_input_backend_contract(request, backend):
    stream = request.getfixturevalue(backend)
    assert isinstance(stream, NidaqInputBackend)
    configuration = stream.configuration
    names = [channel.name for channel in configuration.channels]

    verified = stream.verify_tasks(commit=False)
    assert verified and all(isinstance(task_id, str) for task_id in verified)

    stream.start()
    blocks = []
    for _ in range(3):
        before = time.perf_counter()
        block = stream.read_chunk()
        # The stamp is taken before the blocking read, never after it.
        assert before <= block.perf_time <= time.perf_counter()
        blocks.append(block)
    assert [block.sample_index for block in blocks] == [0, 50, 100]
    for block in blocks:
        assert sorted(block.values) == sorted(names)
        assert all(len(block.values[name]) == configuration.read_chunk_size for name in names)
        assert set(numpy.unique(block.values["cam_frames"])) <= {0.0, 1.0}
    assert isinstance(stream.read_telemetry, dict)
    stream.close()
    stream.close()  # may be called more than once


# ------------------------------------------------------------------ laser

@pytest.fixture
def real_laser(monkeypatch):
    fake = _device_test_module("nidaq_daqmx_fake")
    monkeypatch.setattr(nidaq_laser, "_load_nidaqmx", lambda: fake.FakeDaqmx())
    # The diode and command copy are read off the NI input stream, as LaserModel wires it.
    controller = nidaq_laser.NidaqLaserController(fake.rig_lasers(), feedback_reader=lambda terminal: 0.0)
    yield controller
    controller.close()


@pytest.fixture
def sim_laser(bus):
    fake = _device_test_module("nidaq_daqmx_fake")
    controller = SimulatedLaserController(fake.rig_lasers())
    yield controller
    controller.close()


@pytest.mark.parametrize("backend", ["real_laser", "sim_laser"])
def test_laser_controller_contract(request, backend):
    laser = request.getfixturevalue(backend)
    assert isinstance(laser, LaserControllerProtocol)
    channel = LaserChannelId.LASER_1
    assert laser.configuration.get_channel(channel).channel_id == channel

    assert laser.set_command_voltage(channel, 0.5) == pytest.approx(0.5)
    assert laser.read_feedback_sample(channel).command_volts == pytest.approx(0.5)
    laser.set_command_voltage(channel, 0.0)

    # Waited for: it returns once the train has ended, back on its baseline.
    laser.run_pulse_train(LaserPulseTrain(
        channel_id=channel, amplitude_volts=1.0, duration_ms=2.0, pulse_count=3, frequency_hz=100.0))
    baseline = laser.configuration.get_channel(channel).minimum_command_volts
    assert laser.read_feedback_sample(channel).command_volts == pytest.approx(baseline)
    if backend == "sim_laser":
        # Seen on the NI inputs: three windows on each of the command copy and diode.
        events = request.getfixturevalue("bus").poll()
        assert sorted({event.line for event in events}) == ["laser1_command_copy", "laser1_diode"]
        assert len(events) == 6 and all(event.value == 1.0 for event in events)

    laser.close_all_shutters()
    laser.close()


# ------------------------------------------------------------------ motor/pellet

class _FakeJerryCan:
    """The CAN driver's command calls, each answered 0 (sent)."""

    def __init__(self):
        self.sent = []

    def ToneWrite(self, *args):  # noqa: N802 - pyjerrycan's names
        self.sent.append(("tone",) + args)
        return 0

    def GPIOPulse(self, *args):  # noqa: N802
        self.sent.append(("pulse",) + args)
        return 0

    def ReceiveMessage(self):  # noqa: N802
        return None

    def Close(self):  # noqa: N802
        return 0


@pytest.fixture
def can_interface():
    # Never opened on a bus: the driver is faked and the board's address set,
    # as open() leaves them.
    interface = CanInterface(can_transport=CanTransportConfiguration(kind="socketcan"))
    interface._jc = _FakeJerryCan()
    interface.pellet_address = 0
    interface._is_open = True
    yield interface
    interface.close()


@pytest.fixture
def emulator(bus):
    interface = EmulationInterface()
    assert interface.open()
    yield interface
    interface.close()


@pytest.mark.parametrize("backend", ["can_interface", "emulator"])
def test_motor_pellet_backend_contract(request, backend):
    board = request.getfixturevalue(backend)
    assert isinstance(board, MotorPelletBackend)
    assert board.is_open and board.can_read()

    # An acknowledged command allocates a new uuid; CanDevice waits on it.
    before = board.uuid()
    assert board.emit_tone(5000, 100)
    tone_uuid = board.uuid()
    assert tone_uuid != before

    # Refused before anything is sent: the tone lines are not STIM outputs.
    with pytest.raises(ValueError):
        board.pulse_digital_output(DigitalOutputs.STIMULUS_1, 1000)
    with pytest.raises(ValueError):
        board.pulse_digital_output(DigitalOutputs.STIMULUS_3, 50)
    assert board.uuid() == tone_uuid
    assert board.pulse_digital_output(DigitalOutputs.STIMULUS_3, 1000)
    assert board.uuid() != tone_uuid

    messages = board.read(20)
    assert isinstance(messages, list)
    if backend == "emulator":
        acks = [m.uuid for m in messages if isinstance(m, Acknowledge)]
        assert acks == [tone_uuid, board.uuid()]
    else:
        assert [entry[0] for entry in board._jc.sent] == ["tone", "pulse"]

    board.close()
    assert not board.is_open
