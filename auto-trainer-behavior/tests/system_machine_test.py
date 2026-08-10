import contextlib
import logging
import math
import time
from itertools import chain
from pathlib import Path
from threading import Timer
from unittest import mock

import pytest

from autotrainer.core.pose_elements import SceneElement
from autotrainer.inference import PoseResponse, PoseLocation

from top_fixtures import MockSystemMachine


from autotrainer.core import HeadbarPressureMonitor, get_perf_now, Offset3DTuple
from autotrainer.core import Notification, TriggerNotification, NotificationCenter

from autotrainer.behavior import IntersessionState
from autotrainer.core.interfaces import CaptureAnalysisResult
from autotrainer.behavior import SystemState, SystemMachine
from autotrainer.behavior.pellet import PelletState
from autotrainer.behavior.pellet.pellet_machine import PelletMachine

from autotrainer.inference.analysis import IntersessionResponse


def test_enter_exit_tunnel(mock_system, machine):
    algo = machine.algorithm

    # Observe for video capture being triggered.
    is_capture_triggered = False

    def set_capture_triggered(notification: Notification):
        nonlocal is_capture_triggered
        is_capture_triggered = notification.context

    NotificationCenter.default_center().add_observer(TriggerNotification.CAPTURE_ID, set_capture_triggered)

    # Current code assumes intersession analysis is off by default.
    assert algo.intersession_enabled is False

    # Defaults
    assert machine.state == SystemState.cage
    assert algo.is_in_session is False
    assert not algo.pellet_recently_seen
    algo.update_pellet_seen(True)
    assert algo.pellet_recently_seen
    assert is_capture_triggered is False

    # Manual Record owns session start.
    mock_system.start_session_in_tunnel()

    assert algo.is_in_session is True
    assert is_capture_triggered is True
    assert machine.state == SystemState.tunnel

    mock_system.exit_tunnel()

    assert machine.state == SystemState.cage
    assert algo.is_in_session is False
    assert is_capture_triggered is False
    assert mock_system.machine_state_trans == [
        SystemState.tunnel,
        SystemState.cage,
    ]


def test_manual_session_does_not_require_pellet(mock_system, machine: SystemMachine):
    algo = machine.algorithm
    assert not algo.pellet_recently_seen

    mock_system.start_session_in_tunnel(set_recording_status=True)

    assert algo.is_in_session
    assert machine.state == SystemState.tunnel
    mock_system.exit_tunnel()
    assert not algo.is_in_session
    assert machine.state == SystemState.cage


def test_intersession_enabled(mock_system, machine):
    """
    Placeholder for intersession analysis when ready.  Will not test details of intersession state machine, but that the
    system changes are as expected.
    :return: None
    """
    algo = machine.algorithm
    pellet_m = machine._pellet_machine

    algo.intersession_enabled = True
    algo.pellet_cover_enabled = True

    assert machine.state == SystemState.cage
    assert algo.system_state == machine.state
    assert pellet_m.state == PelletState.monitoring

    mock_system.mock_pose_response(pellet_seen=True)

    # first pellet-seen make the pellet to be covered:
    assert mock_system.pellet_state_trans == [
        PelletState.covering,
        PelletState.retract,
    ]
    mock_system.pellet_state_trans.clear()
    assert pellet_m._api_status_token is not None  # but we wait the cover ack
    mock_system.mock_pellet_ack()
    
    mock_system.start_session_in_tunnel(set_recording_status=True)  # this trigger a start session recording

    assert algo.is_in_session
    assert pellet_m.state == PelletState.retract
    assert machine.state == SystemState.tunnel
    assert algo.system_state == machine.state
    mock_system.mock_pellet_ack(until_none=True)
    assert pellet_m.state == PelletState.retract

    mock_system.mock_pose_response(pellet_seen=False, mouse_seen=True, ack_pellet=True)
    mock_system.mock_pellet_ack(until_none=True)

    assert pellet_m.state == PelletState.monitoring

    with mock_system.mock_perform_segmentation():
        mock_system.exit_tunnel()

    assert algo.intersession_state == IntersessionState.segmentation
    assert machine.state == SystemState.intersession
    assert algo.system_state == machine.state
    assert mock_system.machine_state_trans == [
        SystemState.tunnel,
        SystemState.intersession,
    ]
    assert mock_system.pellet_state_trans == [
        PelletState.sending,
        PelletState.monitoring,
        PelletState.retract,
    ]
    assert pellet_m._api_status_token is not None, \
        "An API status token should be in use for the previous retract"
    assert pellet_m._token_move_retract is pellet_m._api_status_token
    mock_system.mock_pose_response(pellet_seen=True)
    mock_system.mock_pellet_ack()
    assert pellet_m._api_status_token is None
    assert pellet_m._token_move_retract is None


def test_inference_detection_ready(machine):
    algo = machine.algorithm
    result = IntersessionResponse(
        rh_max_vp_list=[Offset3DTuple(50, 0, 0)],
        food_consumed=20,
        pellets_presented=40,
        successful_reaches=4,
    )
    # before:
    assert algo.pellets_consumed == 0
    assert algo.successful_reaches == 0
    assert algo.pellets_presented == 0
    #
    machine._inference.detection_result_ready(machine.project, result)
    # after:
    assert algo.pellets_consumed == 20
    assert algo.successful_reaches == 4
    assert algo.pellets_presented == 0   # NB: this now accounts for pellet-sent
    # now:
    result.food_consumed = 15
    result.successful_reaches = 2
    result.pellets_presented = 30
    machine._inference.detection_result_ready(machine.project, result)
    assert algo.pellets_consumed == 35
    assert algo.successful_reaches == 6
    assert algo.pellets_presented == 0  # NB: this now accounts for pellet-sent


class TestSessionProcessingEnding(MockSystemMachine):

    def test_when_no_intersession(self, machine):
        processing_ended_count = 0
        def processing_ended(prj, status):
            nonlocal processing_ended_count
            processing_ended_count += 1
            assert status == CaptureAnalysisResult.CAPTURE_ONLY
        #
        algo = machine.algorithm
        algo.intersession_enabled = False
        algo.session_ending += processing_ended
        algo.start_session()
        assert processing_ended_count == 0
        algo.end_capture_session()
        assert processing_ended_count == 1

    def test_when_intersession_mouse_not_seen(self, machine):
        processing_ended_count = 0
        def processing_ended(prj, status):
            nonlocal processing_ended_count
            processing_ended_count += 1
            assert status == CaptureAnalysisResult.CAPTURE_ONLY
        #
        algo = machine.algorithm
        algo.intersession_enabled = True
        algo.session_ending += processing_ended
        #
        algo.start_session()
        assert processing_ended_count == 0
        algo.update_mouse_seen(False)
        assert processing_ended_count == 0
        algo.end_capture_session()
        assert processing_ended_count == 1

    @pytest.mark.parametrize("detection_success", [False, True])
    @pytest.mark.parametrize("system_state", [SystemState.cage, SystemState.tunnel])
    def test_when_intersession_mouse_seen(self, machine, detection_success, system_state):
        processing_ended_count = 0
        def processing_ended(prj, status):
            nonlocal processing_ended_count
            processing_ended_count += 1
            assert status == (
                CaptureAnalysisResult.ANALYSIS_SUCCEEDED if detection_success
                else CaptureAnalysisResult.ANALYSIS_FAILED
            )

        #
        algo = machine.algorithm
        algo.intersession_enabled = True
        algo.session_ending += processing_ended
        algo.start_session()
        algo.update_mouse_seen(True)
        assert processing_ended_count == 0
#         with self.mock_analysis(detection_ok=detection_success):
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.mock_perform_segmentation())
            assert processing_ended_count == 0
            stack.enter_context(self.mock_perform_detection())
            assert processing_ended_count == 0
            machine.state = system_state
            algo.end_capture_session()
            assert processing_ended_count == 0
            self.mock_complete_segmentation(True)
            assert processing_ended_count == 0
            self.mock_complete_detection(detection_success)
            assert processing_ended_count == 1
        assert processing_ended_count == 1

    def test_when_intersession_mouse_seen_segmentation_fails(self, machine):
        processing_ended_count = 0
        def processing_ended(prj, status):
            nonlocal processing_ended_count
            processing_ended_count += 1
            assert status == CaptureAnalysisResult.ANALYSIS_FAILED
        #
        algo = machine.algorithm
        algo.intersession_enabled = True
        algo.session_ending += processing_ended
        algo.start_session()
        algo.update_mouse_seen(True)
        assert processing_ended_count == 0
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.mock_perform_segmentation())
            assert processing_ended_count == 0
            algo.end_capture_session()
            assert processing_ended_count == 0
            self.mock_complete_segmentation(False)
            assert processing_ended_count == 1
        assert processing_ended_count == 1


def test_handle_diamond_triangle_offset_full(mock_system, machine):
    self = mock_system
    algo = machine.algorithm
    algo.reload_diamond_triangle_config()  # ensure it's loaded
    mock_system.mock_pose_response(pellet_seen=True)
    mock_system.start_session_in_tunnel(set_recording_status=True)
    mock_system.mock_pellet_ack(until_none=True)
    pellet_m = machine.pellet
    diamond_cfg = machine.algorithm.diamond_triangle_config
    assert machine.state == SystemState.tunnel
    assert pellet_m.state == PelletState.monitoring
    assert pellet_m.can_use_pellet_command()
    #
    last_pos = pellet_m._pellet_device.last_position = diamond_cfg.used_position  # noqa
    inference_pos = diamond_cfg.motor_to_inference(last_pos)
    # diamond_cfg.inference_to_motor()
    rsp_idx = 0
    se = SceneElement
    locs_3d = {
        se.Diamond: Offset3DTuple.get_zero(),
        se.Triangle: diamond_cfg.measured_offset,
    }
    presents = {se.Diamond: True, se.Triangle: True, se.Pellet: True}  # keep pellet seen

    def make_rsp():
        nonlocal rsp_idx
        parts_offset = {
            se.Diamond: {
                se.Triangle: locs_3d[se.Triangle] - locs_3d[se.Diamond],
            }
        }
        locs = []  # 2d inference location unnecessary
        r = PoseResponse(
            sequence=rsp_idx,
            parts_flags=(presents, presents, presents),
            locations=locs,
            locations_3d=locs_3d,
            parts_3d_offsets=parts_offset,
        )
        rsp_idx += 1
        return r
    #
    def pose_changed():
        machine._on_pose_changed(make_rsp())
    #
    assert algo.get_diamond_triangle_drifts() is None
    pose_changed()
    assert algo.get_diamond_triangle_drifts() == (0, 0, 0)
    locs_3d[se.Triangle] += (0.5, 1, -1)
    mock_system.mock_pellet_ack()
    pose_changed()
    assert algo.get_diamond_triangle_drifts() == (0.25, -0.5, 0.5)  # given 2 measures now
    pose_changed()
    assert algo.get_diamond_triangle_drifts(reset=True) == (1 / 3, -2 / 3, 2 / 3)  # given 3 measures now
    assert algo.get_diamond_triangle_drifts() is None
    #
    presents.pop(se.Pellet)
    pose_changed()
    self.increment_perf_now(algo.pellet_missing_time)
    assert algo.get_diamond_triangle_drifts(reset=True) is not None
    pose_changed()
    assert pellet_m.state == PelletState.loading
    assert algo.get_diamond_triangle_drifts() is not None   #
    presents[se.Pellet] = True
    pose_changed()
    # assert algo.get_diamond_triangle_drifts() is None  # given in sending now
    self.mock_pellet_ack()  # ack load
    assert pellet_m.state == PelletState.sending
    pose_changed()
    assert not pellet_m.can_use_pellet_command()  # sending not finished
    self.mock_pellet_ack()  # ack send
    assert pellet_m.state == PelletState.monitoring  # back
    assert pellet_m.can_use_pellet_command()
    pose_changed()
    assert algo.get_diamond_triangle_drifts() == (0.5, -1, 1)  # back
