import collections
import copy
import contextlib
import logging
import math
import multiprocessing
import queue
import threading
import time
from multiprocessing import synchronize

from pathlib import Path
from functools import partial
from typing import List, Any, Optional, Union, ContextManager, Generator, Callable
from unittest import mock
# from collections.abc import Generator

import pytest
import verboselogs

import autotrainer.core
from autotrainer.behavior.behavior_algorithm import BehaviorAlgoStatus

from autotrainer.core import EventManager, MessageHandler, SystemMessageHandler, ProjectInfo
from autotrainer.core.analysis import ReachAnalysis
from autotrainer.core.multiproc import make_daemon_timer, DaemonTimer
from autotrainer.device import MotorConfigurationFile, CompoundMovements
from autotrainer.inference.analysis import IntersessionResponse

from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.interfaces import RecordingEndingReason
from autotrainer.inference import PoseAlgorithm, PoseResponse, InferenceStatus

from autotrainer.behavior import SystemMachine, PelletDeviceProtocol, BehaviorAlgorithm, \
    InferenceProtocol, SystemState
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.behavior.pellet import PelletState
from tools.acquisition.model.behavior_model import BehaviorModel
from tools.acquisition.model.hardware_model import HardwareModel
from tools.acquisition.model.user_preferences import UserPreferences

logger = logging.getLogger(__name__)


repo_root_dir = Path(__file__).parent  # supposed to be the repo root/top dir
repo_root_tests_subdir = repo_root_dir.joinpath("tests")


fake_perf_now = 0  # used to control time.perf_counter() in BehaviorAlgo/SystemMachine/PelletMachine/Intersession


def simulate_get_perf_now():
    global fake_perf_now
    fake_perf_now += 1e-9  # convenience, so that any call to it will get a different value than the previous
    return fake_perf_now


def get_current_simulate_perf_now():
    return fake_perf_now


def increase_simulate_perf_now(delay: float = 60, refresh_func: Optional[Callable] = None):
    global fake_perf_now
    if delay > 1:
        # eventually try to help mitigate possible issue with background thread(s) doing sleep and
        # checking get_perf_now() vs some previous saved perf_now.
        while delay > 1:
            fake_perf_now += 0.5
            delay -= 0.5
            # but this does not really solve the issue, this is more for emphase it
    fake_perf_now += delay


# for small diff of timers delay:
class AlmostEqualFloat(float):
    def __eq__(self, other):
        return abs(self - other) < 0.01


@pytest.fixture
def mock_get_perf_now(monkeypatch):
    global fake_perf_now
    fake_perf_now = 0
    monkeypatch.setattr(autotrainer.core, "_get_perf_now", simulate_get_perf_now)


@pytest.fixture(autouse=True)
def auto_close_event_manager():
    # allow to close the EventManager and have its worker thread exits gracefully (on each end of test case)
    yield
    EventManager.try_close_default()


@pytest.fixture(autouse=True, scope="session")
def auto_set_misc_log_level():
    # some logger we don't want too verbose in any case
    logging.getLogger('transitions').setLevel(logging.INFO)


@pytest.fixture(autouse=True)
def motor_config(monkeypatch):
    monkeypatch.setattr(MotorConfigurationFile, "DEFAULT_LOCATION",
                        repo_root_tests_subdir.joinpath(MotorConfigurationFile.DEFAULT_LOCATION.name))
    monkeypatch.setattr(CompoundMovements, "DEFAULT_LOCATION",
                        repo_root_tests_subdir.joinpath(CompoundMovements.DEFAULT_LOCATION.name))


@pytest.fixture
def project_info(tmp_path) -> ProjectInfo:
    root = tmp_path.joinpath("root")
    root.mkdir()
    prj = ProjectInfo(root=root.as_posix())
    return prj


@pytest.fixture(autouse=True)
def diamond_config_path(monkeypatch):
    path = repo_root_tests_subdir.joinpath("diamond_triangle_offset.yaml")
    monkeypatch.setattr(DiamondTriangleOffsetConfig, "DEFAULT_CONFIG_PATH", path)
    assert DiamondTriangleOffsetConfig.DEFAULT_CONFIG_PATH is path
    return path


@pytest.fixture
def diamond_triangle_config(diamond_config_path) -> DiamondTriangleOffsetConfig:
    return DiamondTriangleOffsetConfig.load_config(diamond_config_path)



@pytest.fixture
def trainer_config_dir(tmp_path):
    cfg_dir = tmp_path.joinpath("Autotrainer")
    cfg_dir.mkdir()
    return cfg_dir


@pytest.fixture
def animals_dir(tmp_path):
    path = tmp_path.joinpath("animals")
    path.mkdir()
    return path


@pytest.fixture
def settings_ini_path(tmp_path):
    return tmp_path.joinpath("settings.ini")


@pytest.fixture
def user_pref(
    tmp_path,
    trainer_config_dir,
    animals_dir,
    settings_ini_path,
) -> UserPreferences:  # noqa
    pref = UserPreferences(settings_file_path=settings_ini_path)
    pref.configuration_location = trainer_config_dir
    pref.animal_location = animals_dir
    p = tmp_path.joinpath("logs")
    p.mkdir()
    pref.log_location = p
    pref.log_level = int(verboselogs.VERBOSE)
    return pref


def collect_log_queue_to_caplog(log_queue):
    # Drain the queue after the process completes and inject into caplog
    while not log_queue.empty():
        record = log_queue.get()
        logging.getLogger(record.name).handle(record)


@pytest.fixture
def capture_multiprocess_logs(caplog) -> multiprocessing.Queue:  # noqa
    """Listens to a multiprocessing queue and forwards entries to caplog."""
    log_queue = multiprocessing.Queue()
    try:
        yield log_queue  # noqa
    finally:
        collect_log_queue_to_caplog(log_queue)
        log_queue.close()


@pytest.fixture
def make_log_dict_multiproc(capture_multiprocess_logs, monkeypatch):
    def wrapped_make_log_dict():
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'handlers': {
                'queue': {
                    'class': 'autotrainer.core.logging.WithThreadIdQueueHandler',
                    'queue': capture_multiprocess_logs,
                    'level': logging.NOTSET,  # pass everything to the listener
                }
            },
            # root logger is here:
            'root': {
                'handlers': ['queue'],
                # with its own level here:
                'level': logging.NOTSET,  # root_log_level,
                # FORCE NOTSET to relay everything so that file handler can properly get DEBUG as well
            },
            # but eventual level of other loggers have to be defined here:
            'loggers': copy.deepcopy(autotrainer.core.logging._limit_loggers_level),  # noqa
        }
    return wrapped_make_log_dict


##


def property_value_save_transitions(old_value, new_value, *, transitions: List[Any]):
    """Helper to record the transitions of value of a property
    Also ensure/assert that the transitions are consistent.
    """
    if len(transitions) > 0:
        assert transitions[-1] == old_value
        assert new_value != old_value
    transitions.append(new_value)


@pytest.fixture(scope='session', autouse=True)
def _disable_timers():
    def disabled_daemon_timer(delay, func):
        # raise RuntimeError(f"Disabled daemon timer {delay} -> {func}")
        logging.warning("DaemonTimer disabled for delay=%s @ %s", delay, func)
        timer = mock.create_autospec(DaemonTimer)
        # for some reason the finished event isn't present on the mocks, despite the autospec. so set it:
        mock_finished = timer.finished = mock.create_autospec(threading.Event)
        mock_finished.is_set.return_value = True
        return timer
    with mock.patch(f"{DaemonTimer.__module__}.{DaemonTimer.__name__}", new=disabled_daemon_timer):
        yield


@pytest.fixture
def pellet_device() -> PelletDeviceProtocol:
    return mock.create_autospec(PelletDeviceProtocol)


@pytest.fixture
def pose_algo():
    return PoseAlgorithm()


class VoidInference(InferenceProtocol):

    def __init__(self):
        super().__init__()
        self._stop_recorded_event = threading.Event()  # noqa

    @property
    def stop_recorded_event(self) -> synchronize.Event:
        # still required,
        # could maye be added to InferenceProtocol, which is more actually InferenceBase class...
        return self._stop_recorded_event


@pytest.fixture
def inference() -> VoidInference:
    inference = VoidInference()
    inference.status = InferenceStatus.live
    yield inference
    # inference.terminate()


@pytest.fixture
def system_msg_queue():
    # Unused
    q = queue.Queue()
    try:
        yield q
    finally:
        logging.info("system msg qsize after use: %s", q.qsize())
        while True:
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break
        q.join()


@pytest.fixture
def sensor_analysis(mock_get_perf_now) -> ReachAnalysis:
    s = ReachAnalysis()
    try:
        yield s
    finally:
        s.stop()


@pytest.fixture
def system_msg_handler(system_msg_queue, sensor_analysis):
    # Unused
    handler = SystemMessageHandler(system_msg_queue)
    handler.start()
    try:
        yield handler
    finally:
        handler.request_terminate()
        handler.wait_terminated()


class FakeMsgQueue:

    def put_nowait(self, item):
        pass
    def put(self, *args, **kwargs):
        pass
    def get(self, *args, **kwargs):
        raise RuntimeError("not supposed to happen")
    def get_nowait(self):
        raise RuntimeError("not supposed to happen")


class FakeSystemMsgHandler(SystemMessageHandler):

    def __init__(self, input_queue):
        super().__init__(input_queue)

    def start(self):
        pass

    def request_terminate(self):
        pass

    def wait_terminated(self):
        pass


@pytest.fixture
def fake_msg_queue():
    return FakeMsgQueue()


@pytest.fixture
def fake_system_msg_handler(fake_msg_queue, sensor_analysis):
    return FakeSystemMsgHandler(fake_msg_queue)


@pytest.fixture
def machine(project_info, pellet_device, inference, sensor_analysis, monkeypatch, mock_get_perf_now, fake_system_msg_handler) -> SystemMachine:
    # Disable algo handler thread
    assert BehaviorAlgorithm._no_handler_thread is False
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    assert BehaviorAlgorithm._no_handler_thread is True

    del mock_get_perf_now  # not needed here, only used for side effect
    #
    inference.project = project_info
    #
    machine = SystemMachine(
        pellet_device=pellet_device,
        analysis=sensor_analysis,
        inference=inference,
        project_info=project_info,
        msg_handler=fake_system_msg_handler,
    )
    algo = machine.algorithm
    # most tests rely on:
    cfg = algo.pellet_delivery_config
    cfg.is_enabled = True
    cfg.is_pellet_cover_enabled = True
    cfg.pellet_send_wait_delay = 0
    algo.record_prebuffer_duration = 0
    algo.pellet_uncover_delay = 0
    algo.pellet_uncover_y_dcs = -math.inf
    algo.session_minimum_duration = 0  # needed for most current tests
    # might be needed to reset:
    algo.capture_status = CaptureProcessStatus.RUNNING
    algo.status = BehaviorAlgoStatus.RUNNING
    # State-machine unit tests exercise individual transitions without building
    # a complete recording session. Session-bound automation gating is covered
    # separately by behavior_algo_test.
    monkeypatch.setattr(algo, "_pellet_automation_enabled_for_session", lambda: True)
    machine.pellet.state = PelletState.monitoring  # force monitoring for current tests
    return machine


class FifoExitStack(contextlib.ExitStack):
    _exit_callbacks: collections.deque

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_callbacks = collections.deque(reversed(list(self._exit_callbacks)))
        super().__exit__(exc_type, exc_val, exc_tb)


class MockSystemMachine:
    """Allow make test case on a fully prepared 'SystemMachine' instance, with many helper methods included"""

    def _init(self, machine: SystemMachine):
        self._machine: SystemMachine = machine
        self.project = machine.project
        # register state_changed (and system_state_changed for algo at end) transition recorder,
        # so that can be used to ensure/assert that the given states have passed through all the desired values,
        # and in any desired specific order - or not.
        self.machine_state_trans = []
        machine.events.state_changed += partial(
            property_value_save_transitions, transitions=self.machine_state_trans)
        self.pellet_state_trans = []
        self.pellet.events.state_changed += partial(
            property_value_save_transitions, transitions=self.pellet_state_trans)
        self.intersession_state_trans = []
        machine.intersession.events.state_changed += partial(
            property_value_save_transitions, transitions=self.intersession_state_trans)
        self._pose_response_idx = 0

    increment_perf_now = staticmethod(increase_simulate_perf_now)

    get_current_perf_now = staticmethod(get_current_simulate_perf_now)

    @pytest.fixture(autouse=True)
    def machine(self, machine: SystemMachine) -> SystemMachine:  # noqa
        self._init(machine)
        yield machine  # noqa

    @property
    def system_machine(self) -> SystemMachine:
        return self._machine

    @property
    def algo(self):
        return self._machine.algorithm

    @property
    def sensor_analysis(self):
        return self._machine.analysis

    @property
    def inference(self) -> InferenceProtocol:
        return self._machine._inference

    @property
    def pellet(self):
        return self._machine.pellet

    @property
    def msg_handler(self) -> MessageHandler:
        return self._machine._msg_handler

    @property
    def pellet_dev(self) -> Union[mock.MagicMock, PelletDeviceProtocol]:
        return self._machine._pellet_device

    #

    def start_recording_session(self, set_recording_status: bool = False):
        algo = self.algo
        assert not algo.is_in_session
        assert self._machine.state == SystemState.ready
        algo.start_session(reason="manual")
        if set_recording_status:
            algo.set_capture_status(CaptureProcessStatus.RECORDING)
        # assert algo.is_in_session
        assert self._machine.state == SystemState.ready

    def stop_recording_session(self):
        if self.algo.is_in_session:
            self.algo.end_capture_session(reason=RecordingEndingReason.MANUAL_STOP)

    @contextlib.contextmanager
    def patch_timer(self, place, new=None) -> ContextManager[Union[mock.MagicMock, DaemonTimer]]:  # noqa
        kw = {"autospec": DaemonTimer} if new is None else {"new": new}
        with mock.patch(place, **kw) as mock_t:
            # for some reason the finished event isn't present on the mocks, despite the autospec. so set it:
            mock_finished = mock_t.return_value.finished = mock.create_autospec(threading.Event)
            mock_finished.is_set.return_value = False
            yield mock_t  # noqa

    @contextlib.contextmanager
    def mock_intersession_analysis(
        self,
        results: Optional[IntersessionResponse] = None,
        *,
        segmentation_ok: bool = True,
        detection_ok: bool = True,
        concurrent_func: Optional[Callable] = None,
        project: Optional[ProjectInfo] = None,
        stack: Optional["FifoExitStack"] = None,
    ):
        """Allow fake fully 1 intersession/trial analysis (segmentation+detection)"""
        if results is None:
            results = IntersessionResponse()
        results: IntersessionResponse
        if project is None:
            project = self._machine.project.to_local_value()
        if stack is None:
            r_stack = FifoExitStack()
        else:
            r_stack = stack

        @contextlib.contextmanager
        def doit(stack):
            stack.enter_context(self.mock_perform_segmentation())
            logger.info("prepared stack for mock_perform_segmentation")
            yield
            if segmentation_ok:
                stack.enter_context(self.mock_perform_detection())
                logger.info("prepared stack for mock_perform_detection")
            if concurrent_func is not None:
                concurrent_func()
            self.mock_complete_segmentation(segmentation_ok)
            if detection_ok:
                logger.info("sending detection_result_ready")
                self.inference.detection_result_ready(project, results)
            self.mock_complete_detection(detection_ok)
        if r_stack is stack:
            r_stack.enter_context(doit(r_stack))
            yield
        else:
            with r_stack:
                r_stack.enter_context(doit(r_stack))
                yield

    @contextlib.contextmanager
    def mock_perform_segmentation(self):
        """Mock the inference.perform_segmentation() method"""
        with mock.patch.object(self.inference, 'perform_segmentation') as m_seg:
            yield m_seg

    def mock_complete_segmentation(self, success: bool):
        seg_cfg = self._machine.intersession._segmentation_configuration
        logger.info("calling segmentation complete")
        assert seg_cfg is not None
        seg_cfg.complete(success)

    @contextlib.contextmanager
    def mock_perform_detection(self):
        """Mock the inference.perform_detection() method"""
        with mock.patch.object(self.inference, 'perform_detection') as m_det:
            yield m_det

    def mock_complete_detection(self, success: bool):
        det_cfg = self._machine.intersession._detection_configuration
        assert det_cfg is not None
        det_cfg.complete(success)

    def mock_pose_response(
        self,
        *,
        pellet_seen: bool,
        mouse_seen: bool=False,
        triangle_seen: bool=True,
        diamond_seen: bool=True,
        ack_pellet: bool=False,
    ):
        """Send/trigger a PoseResponse via pose_algorithm.pose_changed event"""
        parts_flag = {
            "Pellet": pellet_seen,
            "Tongue": mouse_seen,
            "Nose": mouse_seen,
            "Triangle": triangle_seen,
            "Diamond": diamond_seen,
        }
        parts_flags = (parts_flag, parts_flag, parts_flag)
        self._pose_response_idx += 1
        response = PoseResponse(sequence=self._pose_response_idx, parts_flags=parts_flags, locations=[])
        self.inference.pose_response_ready(response)
        if self.pellet._api_status_token is not None and ack_pellet:
            self.pellet._pellet_device_ack_received(self.pellet._api_status_token)

    def expect_cover_command(self):
        # An explicit cover command should have been set.  Should be in covering state and have an ack from the command.
        assert self.pellet.state == PelletState.covering
        self.mock_pellet_ack()

    def mock_pellet_ack(self, until_none: bool=False, max_limit: int=45):
        """Ack the previous sent pellet command"""
        cur_ack = 0
        while True:
            token = self.pellet._api_status_token
            if token is None:
                break
            cur_ack += 1
            self.increment_perf_now(1e-9)
            self.pellet._pellet_device_ack_received(token)
            if not until_none:
                break
            if cur_ack > max_limit:
                raise RuntimeError("Too many consecutive pellet ack")

    def mock_pellet_missing(self, mouse_seen: bool = False):
        self.mock_pose_response(pellet_seen=False, mouse_seen=mouse_seen)
        # make sure we are beyond the required pellet missing time:
        self.increment_perf_now(self._machine.algorithm.pellet_missing_time + 1e-9)
        self.mock_pose_response(pellet_seen=False, mouse_seen=mouse_seen)

    def make_recording_aged_enough(self):
        algo = self._machine.algorithm
        algo.capture_status = CaptureProcessStatus.RECORDING
        p_now = get_current_simulate_perf_now()
        algo.project.start_record_timestamp = p_now


@pytest.fixture
def mock_system(machine) -> MockSystemMachine:
    """Allow use BaseSystemMachineTest instance helper methods in a simple function test, without having to subclass,
    just use the 'mock_system' fixture"""
    instance = MockSystemMachine()
    # instance.machine_(machine)  # pytest fixture refuse direct call, so:
    instance._init(machine)
    return instance


@pytest.fixture
def hardware_model(fake_system_msg_handler, sensor_analysis) -> HardwareModel:
    return HardwareModel(fake_system_msg_handler)


@pytest.fixture
def behavior_model(sensor_analysis, fake_system_msg_handler, hardware_model, inference) -> BehaviorModel:
    return BehaviorModel(fake_system_msg_handler, sensor_analysis, hardware_model, inference)
