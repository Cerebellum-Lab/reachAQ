import math

import numpy
import pytest

from autotrainer.behavior import BehaviorAlgorithm
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.behavior.behavior_algorithm import BehaviorAlgoStatus
from autotrainer.core.interfaces import CoverServoStatus
from autotrainer.behavior.pellet import PelletState
from autotrainer.core import BehaviorConfiguration, Offset3DTuple, ProjectInfo
from autotrainer.core.capture import CaptureProcessStatus
from top_fixtures import increase_simulate_perf_now


@pytest.fixture
def algo(monkeypatch, mock_get_perf_now, project_info) -> BehaviorAlgorithm:
    del mock_get_perf_now  # used for its side effect
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", True)
    assert BehaviorAlgorithm._no_handler_thread is True
    algo = BehaviorAlgorithm(project_info=project_info)
    algo.active_config.pellet_delivery.pellet_send_wait_delay = 0
    algo.pellet_delivery_enabled = algo.pellet_cover_enabled = True
    algo.status = BehaviorAlgoStatus.RUNNING
    return algo
    # in case need cleanup


def _start_automatic_recording(algo: BehaviorAlgorithm) -> None:
    algo.active_config.session_control.automatic_pellet_cycles_enabled = True
    algo.active_config.pellet_delivery.pellet_send_wait_delay = math.inf
    assert algo.start_session(reason="test") is True
    algo.set_capture_status(CaptureProcessStatus.RECORDING)


def test_project_info(algo):
    prj = algo.project
    new_prj = ProjectInfo()
    algo.project = new_prj
    assert prj != algo.project
    assert algo.project is new_prj


def test_cover_servo_status(algo):
    assert algo.cover_servo_status is CoverServoStatus.OK
    assert not algo.cover_servo_status.is_error
    algo.cover_servo_status = CoverServoStatus.COVER_POSITION_ERROR
    assert algo.cover_servo_status.is_error


def test_diamond_triangle_seen(algo):
    assert math.isinf(algo.triangle_last_seen) and algo.triangle_last_seen < 0
    assert algo.diamond_recently_seen is False
    assert algo.triangle_recently_seen is False
    assert algo.pellet_recently_seen is False


def test_triangle_pellet_offset(algo):
    assert algo.triangle_pellet_offset == Offset3DTuple(math.nan, math.nan, math.nan)
    o3d = algo.triangle_pellet_offset = Offset3DTuple(1, 1, 1)
    assert algo.triangle_pellet_offset == o3d


def test_triangle_pellet_diff_too_far_threshold(algo):
    prev = algo.triangle_pellet_diff_too_far_threshold
    assert prev \
            == algo.triangle_pellet_diff_too_far_threshold \
            == algo.pellet_delivery_config.triangle_pellet_diff_too_far_threshold \
            == algo.active_config.pellet_delivery.triangle_pellet_diff_too_far_threshold
    algo.triangle_pellet_diff_too_far_threshold += 5
    assert prev + 5 \
           == algo.triangle_pellet_diff_too_far_threshold \
           == algo.pellet_delivery_config.triangle_pellet_diff_too_far_threshold \
           == algo.active_config.pellet_delivery.triangle_pellet_diff_too_far_threshold


def test_pellet_delivery_enabled(algo):
    prev = algo.pellet_delivery_enabled
    assert prev == algo.pellet_delivery_enabled == algo.pellet_delivery_config.is_enabled == algo.active_config.pellet_delivery.is_enabled
    algo.pellet_delivery_enabled = not algo.pellet_delivery_enabled
    assert not prev == algo.pellet_delivery_enabled == algo.pellet_delivery_config.is_enabled == algo.active_config.pellet_delivery.is_enabled


def test_pellet_missing_time(algo):
    prev = algo.pellet_missing_time
    assert prev \
           == algo.pellet_missing_time \
           == algo.pellet_delivery_config.max_pellet_missing_seconds \
           == algo.active_config.pellet_delivery.max_pellet_missing_seconds
    algo.pellet_missing_time += 5
    assert prev + 5 \
           == algo.pellet_missing_time \
           == algo.pellet_delivery_config.max_pellet_missing_seconds \
           == algo.active_config.pellet_delivery.max_pellet_missing_seconds


def test_default_diamond_triangle_offset_config_path(algo):
    assert algo.diamond_triangle_offset_config_path == DiamondTriangleOffsetConfig.DEFAULT_CONFIG_PATH


def test_pellet_uncover_y_dcs(algo):
    prev = algo.pellet_uncover_y_dcs
    assert prev == algo.active_config.pellet_uncover.min_y_dcs
    algo.pellet_uncover_y_dcs += 5
    assert prev + 5 == algo.pellet_uncover_y_dcs == algo.active_config.pellet_uncover.min_y_dcs


def test_use_triangle_pellet_distance_too_far(algo):
    prev = algo.use_triangle_pellet_distance_too_far
    assert prev \
           == algo.use_triangle_pellet_distance_too_far \
           == algo.active_config.pellet_delivery.use_triangle_pellet_distance_too_far \
           == algo.pellet_delivery_config.use_triangle_pellet_distance_too_far
    algo.use_triangle_pellet_distance_too_far = not prev
    assert not prev \
           == algo.use_triangle_pellet_distance_too_far \
           == algo.active_config.pellet_delivery.use_triangle_pellet_distance_too_far \
           == algo.pellet_delivery_config.use_triangle_pellet_distance_too_far


def test_triangle_pellet_expected_distance(algo):
    prev = algo.triangle_pellet_expected_distance
    assert prev \
           == algo.triangle_pellet_expected_distance \
           == algo.active_config.pellet_delivery.triangle_pellet_expected_distance \
           == algo.pellet_delivery_config.triangle_pellet_expected_distance
    algo.triangle_pellet_expected_distance += 5
    assert prev + 5 \
               == algo.triangle_pellet_expected_distance \
               == algo.active_config.pellet_delivery.triangle_pellet_expected_distance \
               == algo.pellet_delivery_config.triangle_pellet_expected_distance


def test_capture_status(algo):
    cs = algo.capture_status
    assert cs == CaptureProcessStatus.UNKNOWN
    algo.capture_status = CaptureProcessStatus.RUNNING
    assert algo.capture_status == CaptureProcessStatus.RUNNING


def test_set_put_func_call_mode(algo):

    def record_sync_call_mode(result):
        result.append(getattr(algo._thread_locals, "sync_call_mode"))

    result = []
    prev = getattr(algo._thread_locals, "sync_call_mode", None)
    with algo.set_put_func_call_mode(False):
        algo.put_func_call(record_sync_call_mode, (result,), None)
        assert result[-1] is False
        #
        with algo.set_put_func_call_mode(True):
            algo.put_func_call(record_sync_call_mode, (result,), None)
            assert result[-1] is True
        #
        algo.put_func_call(record_sync_call_mode, (result,), None)
        assert result[-1] is False
    #
    algo.put_func_call(record_sync_call_mode, (result,), None)
    assert prev is result[-1]


@pytest.mark.parametrize("count", [5, 10])
def test_reset_config(algo, count):
    algo.reset_configuration()
    config = BehaviorConfiguration()
    config.pellet_delivery.max_pellet_missing_seconds = count
    algo.load_configuration(config)
    assert algo.pellet_missing_time == count
    algo.pellet_missing_time = 2 * count
    algo.reset_configuration()
    assert algo.pellet_missing_time == count


def test_start_twice_session_fails(algo):
    assert algo.is_in_session is False
    assert algo.start_session(reason="manual") is True
    assert algo.is_in_session is True
    assert algo.start_session(reason="manual") is False
    assert algo.is_in_session is True


def test_end_session_if_not_running_fails(algo):
    assert algo.is_in_session is False
    assert algo.end_capture_session() is False
    assert algo.is_in_session is False


def test_delivery_disabled_defaults(algo):
    algo.pellet_delivery_enabled = False
    assert algo.can_load_pellet() is False
    assert algo.can_send_pellet() is False
    #
    assert algo.can_release_pellet() is False
    assert algo.can_cover_pellet() is False
    #
    algo.pellet_delivery_enabled = True
    _start_automatic_recording(algo)
    #
    assert algo.can_send_pellet() is False
    assert algo.can_load_pellet() is False

    algo.update_triangle_seen(True)
    assert algo.can_load_pellet() is True
    assert algo.can_send_pellet() is False
    #
    assert algo.can_release_pellet() is False
    assert algo.can_cover_pellet() is True
    algo.pellet_cover_enabled = False
    assert algo.can_release_pellet() is True
    assert algo.can_cover_pellet() is False
    #
    algo.active_config.pellet_delivery.pellet_send_wait_delay = 1
    assert algo.can_send_pellet() is False
    increase_simulate_perf_now(algo.active_config.pellet_delivery.pellet_send_wait_delay)
    assert algo.can_send_pellet() is True

    algo.pellet_send_block_reason = "waiting for trial analysis"
    assert algo.can_send_pellet() is False
    algo.pellet_send_block_reason = ""
    assert algo.can_send_pellet() is True


def test_recorded_session_can_enable_pellet_cycles_without_training_mode(algo):
    algo.active_config.session_control.automatic_pellet_cycles_enabled = True
    algo.update_triangle_seen(True)

    assert algo.can_load_pellet() is False
    assert algo.start_session(reason="manual") is True
    assert algo.can_load_pellet() is False

    algo.set_capture_status(CaptureProcessStatus.RECORDING)
    assert algo.can_load_pellet() is True

    algo.pellet_automation_stop_requested = True
    assert algo.can_load_pellet() is False
    assert algo.can_send_pellet() is False


def test_algo_paused(algo):
    _start_automatic_recording(algo)
    algo.algo_paused = True
    assert algo.can_send_pellet() is False
    assert algo.can_release_pellet() is False
    assert algo.can_cover_pellet() is False
    assert algo.can_load_pellet() is False
    algo.algo_paused = False
    assert algo.can_send_pellet() is False
    assert algo.can_cover_pellet() is True
    assert algo.triangle_recently_seen is False
    assert algo.can_load_pellet() is False  # given not triangle recently seen
    algo.update_triangle_seen(True)
    assert algo.triangle_recently_seen is True
    assert algo.can_load_pellet() is True  # given triangle recently seen
    assert algo.pellet_recently_seen is False  # but not pellet


def test_diamond_triangle_drift(algo):
    algo.diamond_triangle_config = DiamondTriangleOffsetConfig(
        used_position=Offset3DTuple(1, 2, 3),
        measured_offset=Offset3DTuple(3, 30, -8),
        version=DiamondTriangleOffsetConfig.current_config_version,
    )
    o1 = Offset3DTuple(0.5, 1, -1)
    p1 = Offset3DTuple(8, 6, 5)
    assert algo.get_diamond_triangle_drifts() is None
    algo.handle_diamond_triangle_offset(o1, p1)
    d1 = algo.get_diamond_triangle_drifts()
    assert d1 is not None
    algo.handle_diamond_triangle_offset(o1, p1)
    d1 = algo.get_diamond_triangle_drifts()
    d2 = algo.get_diamond_triangle_drifts()
    assert d1 == d2
    algo.get_diamond_triangle_drifts(reset=True)
    assert algo.get_diamond_triangle_drifts() is None


def test_cover_pellet_offset(algo):
    o1 = Offset3DTuple(1, 2, 3)
    o2 = Offset3DTuple(3, 4, 5)
    algo.handle_cover_pellet_offset(o1)
    algo.handle_cover_pellet_offset(o2)
    # todo: continue...


def test_release_pellet_offset(algo):
    o1 = Offset3DTuple(1, 2, 3)
    o2 = Offset3DTuple(3, 4, 5)
    algo.handle_release_pellet_offset(o1)
    algo.handle_release_pellet_offset(o2)
    # todo: continue...


@pytest.mark.parametrize("use_dist_too_far", [False, True])
@pytest.mark.parametrize("bad_dist", [0, math.inf, None])  # 0 == too close, inf == far too far
def test_can_load_when_pellet_triangle_too_far(algo, use_dist_too_far, bad_dist):
    _start_automatic_recording(algo)
    algo.pellet_delivery_enabled = True
    algo.pellet_cover_enabled = False
    #
    algo.use_triangle_pellet_distance_too_far = use_dist_too_far
    if bad_dist is None:
        bad_dist = algo.triangle_pellet_expected_distance + algo.triangle_pellet_diff_too_far_threshold
    #
    correct_offset = Offset3DTuple(0, 0, algo.triangle_pellet_expected_distance)
    assert numpy.isclose(correct_offset.distance, algo.triangle_pellet_expected_distance)
    #
    algo.update_triangle_seen(True)
    algo.update_pellet_seen(True)
    #
    algo.triangle_pellet_offset = correct_offset
    assert algo.is_triangle_pellet_distance_too_far() is False
    #
    algo.triangle_pellet_offset = Offset3DTuple(0, 0, bad_dist)
    assert algo.is_triangle_pellet_distance_too_far() is True

    assert algo.can_load_pellet(pellet_state=PelletState.retract) is False
    assert algo.can_load_pellet(pellet_state=PelletState.loading) is False
    assert algo.can_load_pellet(pellet_state=PelletState.home) is False
    assert algo.can_load_pellet(pellet_state=PelletState.sending) is False
    # but:
    assert algo.can_load_pellet(pellet_state=PelletState.monitoring) is use_dist_too_far
    #
    algo.triangle_pellet_offset = correct_offset
    assert algo.is_triangle_pellet_distance_too_far() is False
    assert algo.can_load_pellet(pellet_state=PelletState.monitoring) is False


def test_handle_diamond_triangle_offset_without_config(algo):
    assert algo.diamond_triangle_drift_data_points_size == 0
    algo.handle_diamond_triangle_offset(Offset3DTuple(0, 0, 0), Offset3DTuple(0, 0, 0))
    assert algo.diamond_triangle_drift_data_points_size == 0
