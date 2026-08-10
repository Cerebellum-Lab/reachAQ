import contextlib
import logging
from pathlib import Path
from threading import Timer
from unittest import mock

import pytest

from autotrainer.core import get_perf_now, Offset3DTuple
from autotrainer.core.pose_elements import SceneElement

from autotrainer.behavior import SystemMachine, SystemState
from autotrainer.core.interfaces import RecordingEndingReason
from autotrainer.inference import PoseResponse

from top_fixtures import MockSystemMachine


Diamond = SceneElement.Diamond
Triangle = SceneElement.Triangle


class TestHomeOnExcessiveDrift(MockSystemMachine):

    @pytest.fixture(autouse=True)
    def set_diamond_triangle_config(self, diamond_triangle_config, machine):
        machine.algorithm.diamond_triangle_config = diamond_triangle_config
        self.pellet_dev.last_position = diamond_triangle_config.used_position

    def _init(self, machine: SystemMachine):
        super()._init(machine)
        algo = self.algo
        cfg = algo.home_on_excessive_drift_distance_config
        cfg.enabled = True
        self._cur_seq = 0

    def make_pose_rsp(self, triangle_pos):
        loc3d = {
            Diamond: Offset3DTuple(0, 0, 0),
            Triangle: triangle_pos,
        }
        off3d = {
            Diamond: {Triangle: loc3d[Diamond] - loc3d[Triangle]},
        }
        rsp = PoseResponse(
            sequence=self._cur_seq,
            locations_3d=loc3d,
            parts_3d_offsets=off3d,
        )
        self._cur_seq += 1
        return rsp

    def _execute_pose_responses(self, rsp, min_samples):
        algo = self.algo
        for idx in range(min_samples - 1):
            assert algo.diamond_triangle_drift_data_points_size == idx
            self.inference.pose_response_ready(rsp)
            assert algo.diamond_triangle_drift_data_points_size == idx + 1
        self.inference.pose_response_ready(rsp)
        expected_count = 0 if algo.home_on_excessive_drift_distance_config.enabled else min_samples
        assert algo.diamond_triangle_drift_data_points_size == expected_count

    def _make_to_monitoring(self):
        self.mock_pose_response(pellet_seen=True)
        self.mock_pellet_ack(until_none=True)
        self.start_session_in_tunnel(set_recording_status=True)
        self.mock_pose_response(pellet_seen=True)
        self.mock_pellet_ack(until_none=True)

    @pytest.mark.parametrize("min_samples", [5, 15])
    def test_without_drift(self, min_samples):
        self._make_to_monitoring()
        cfg = self.algo.home_on_excessive_drift_distance_config
        cfg.min_samples = min_samples
        diam_cfg = self.algo.diamond_triangle_config
        algo = self.algo
        rsp = self.make_pose_rsp(
            # NB: given negative of the measured offset in diam-triangle cfg,
            # given this one is measured as : offset == diamond3d - triangle3d
            triangle_pos=-diam_cfg.measured_offset
        )
        self._execute_pose_responses(rsp, min_samples)
        assert self.pellet_dev.send_home.call_args_list == []
        assert algo.diamond_triangle_drift_data_points_size == 0
        assert self.pellet_dev.send_home.call_args_list == [], "no drift should not execute send_home()"

    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize("min_samples", [5, 15])
    def test_with_drift(self, min_samples, enabled):
        self._make_to_monitoring()
        cfg = self.algo.home_on_excessive_drift_distance_config
        cfg.enabled = enabled
        cfg.min_samples = min_samples
        self.algo.start_session(reason="manual")
        self.mock_pellet_ack(until_none=True)
        diam_cfg = self.algo.diamond_triangle_config
        dist_thresh = cfg.excessive_distance_threshold
        big_drift = Offset3DTuple(dist_thresh, dist_thresh, dist_thresh)
        rsp = self.make_pose_rsp(
            triangle_pos=-diam_cfg.measured_offset + big_drift
        )
        assert self.pellet_dev.send_home.call_args_list == []
        self._execute_pose_responses(rsp, min_samples)
        if cfg.enabled:
            assert self.pellet_dev.send_home.call_args_list == [mock.call()], "pellet.send_home() should have been called"
        else:
            assert self.pellet_dev.send_home.call_args_list == [], "not enabled"

    @pytest.mark.parametrize("min_samples", [5, 15])
    def test_during_session_with_drift(self, min_samples):
        algo = self.algo
        cfg = algo.home_on_excessive_drift_distance_config
        cfg.min_samples = min_samples
        diam_cfg = algo.diamond_triangle_config
        dist_thresh = cfg.excessive_distance_threshold
        big_drift = Offset3DTuple(dist_thresh, dist_thresh, dist_thresh)
        rsp = self.make_pose_rsp(triangle_pos=-diam_cfg.measured_offset + big_drift)
        ended_reasons = []
        def capture_ended(reason):
            ended_reasons.append(reason)
        algo.session_capture_ending += capture_ended
        assert self.pellet_dev.send_home.call_args_list == []
        #
        self.mock_pose_response(pellet_seen=True)
        # algo.update_pellet_seen(True)
        algo.active_config.pellet_delivery.pellet_send_wait_delay = 0
        algo.record_prebuffer_duration = 0
        self.start_session_in_tunnel(set_recording_status=True)
        self.mock_pellet_ack(until_none=True)
        assert algo.is_in_session
        self._execute_pose_responses(rsp, min_samples)
        assert self.pellet_dev.send_home.call_args_list == [mock.call()], "pellet.send_home() should have been called"
        assert algo.is_in_session, "manual recording must not be stopped by drift handling"
        assert ended_reasons == []
