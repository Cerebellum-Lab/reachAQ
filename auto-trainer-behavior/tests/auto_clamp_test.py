
import contextlib
import logging
import math
import time
from itertools import chain
from pathlib import Path
from threading import Timer
from unittest import mock

import pytest


from top_fixtures import MockSystemMachine

from autotrainer.behavior import IntersessionState
from autotrainer.core.interfaces import CaptureAnalysisResult
from autotrainer.behavior import SystemState, SystemMachine
from autotrainer.behavior.pellet import PelletState
from autotrainer.behavior.pellet.pellet_machine import PelletMachine
from autotrainer.inference.analysis import IntersessionResponse


class _AutoClampTestCase(MockSystemMachine):

    def start_session_in_tunnel(self, *, engage_headbar: bool = False):
        self.mock_pose_response(pellet_seen=True)
        super().start_session_in_tunnel(set_recording_status=True)
        if engage_headbar:
            self.sensor_analysis.headbar_pressure_monitor.is_engaged = True
            self.mock_pellet_ack(until_none=True)

    @property
    def headbar_pressure(self):
        return self.sensor_analysis.headbar_pressure_monitor

    @property
    def update_magnet_mock(self) -> mock.MagicMock:
        return self.tunnel_dev.update_head_magnet_intensity  # noqa

    def test_when_intersession_with_exit_tunnel(self, machine, caplog):
        algo = self.algo
        algo.intersession_enabled = True
        self.start_session_in_tunnel(engage_headbar=False)
        assert self.update_magnet_mock.call_args_list == []
        algo.update_mouse_seen(True)  # to have intersession started
        with self.mock_intersession_analysis():
            assert self.update_magnet_mock.call_args_list == []
            assert machine.state == SystemState.tunnel
            assert machine.intersession.state == IntersessionState.idle
            with caplog.at_level(logging.INFO):
                self.headbar_pressure.is_engaged = True
            self.exit_tunnel()
            assert machine.state == SystemState.intersession
            assert machine.intersession.state == IntersessionState.segmentation


class TestDisabled(_AutoClampTestCase):

    @pytest.fixture(autouse=True)
    def _init_autoclamp(self, machine):
        machine.algorithm.head_fixation_enabled = False

    def test_when_not_in_session(self, machine, caplog):
        algo = machine.algorithm
        pressure_monitor = machine.analysis.headbar_pressure_monitor
        assert algo.is_in_session is False
        assert algo.system_state != SystemState.tunnel
        with caplog.at_level(logging.INFO):
            pressure_monitor.is_engaged = True
        assert "auto-clamp: disabled (no action taken)" in caplog.text

    def test_when_in_session(self, machine, caplog):
        algo = machine.algorithm
        pressure_monitor = machine.analysis.headbar_pressure_monitor
        algo.head_fixation_enabled = False
        algo.start_session()
        assert algo.is_in_session
        with caplog.at_level(logging.INFO):
            pressure_monitor.is_engaged = True
        assert "auto-clamp: disabled (no action taken)" in caplog.text
        assert self.update_magnet_mock.call_args_list == []

    def test_when_intersession_with_exit_tunnel(self, machine, caplog):
        super().test_when_intersession_with_exit_tunnel(machine, caplog)  # same
        assert "auto-clamp: disabled (no action taken)" in caplog.text
        assert self.update_magnet_mock.call_args_list == []


class TestEnabled(_AutoClampTestCase):

    @pytest.fixture(autouse=True)
    def _init_autoclamp(self, machine):
        self.algo.head_fixation_enabled = True
        self.algo.head_clamp_config.prerelease_duration = 0  # this disables the prerelease
        self.algo.auto_clamp_release_tone_delay = 0  # this skip an extra timer overhead

    def test_when_not_in_session(self, machine, caplog):
        algo = machine.algorithm
        pressure_monitor = machine.analysis.headbar_pressure_monitor
        assert algo.is_in_session is False
        assert algo.system_state != SystemState.tunnel
        with caplog.at_level(logging.INFO):
            pressure_monitor.is_engaged = True
        assert "auto-clamp: algo not in-session (no action taken)" in caplog.text

    def test_when_intersession_with_exit_tunnel(self, machine, caplog):
        super().test_when_intersession_with_exit_tunnel(machine, caplog)  # same
        assert "auto-clamp setting position to " in caplog.text
        assert self.update_magnet_mock.call_args_list == [
            mock.call(self.algo.auto_clamp_intensity),
            mock.call(0),
        ]

    @pytest.mark.parametrize("baseline_intensity", [10, 80])
    @pytest.mark.parametrize("no_activity_release_delay", [0, 60])
    @pytest.mark.parametrize("release_tone_delay", [0, 1])
    @pytest.mark.parametrize("prerelease_duration, prerelease_intensity", [[0, None], [5, 50], [10, 75]])
    def test_when_in_session(self, machine, baseline_intensity, no_activity_release_delay, release_tone_delay, prerelease_duration, prerelease_intensity, caplog):
        algo = self.algo
        algo.baseline_intensity = baseline_intensity
        #
        cfg = algo.head_clamp_config
        cfg.auto_clamp_no_activity_release_delay = no_activity_release_delay
        cfg.auto_clamp_release_tone_delay = release_tone_delay
        #
        cfg.prerelease_duration = prerelease_duration
        cfg.prerelease_intensity = prerelease_intensity

        with self.patch_timer("autotrainer.behavior.system_machine._consider_disengage_autoclamp_timer") as consider_disengage_timer:
            self.start_session_in_tunnel(engage_headbar=True)

        update_magnet_mock = self.update_magnet_mock
        assert update_magnet_mock.call_args_list == [
            mock.call(algo.auto_clamp_intensity),
        ]
        self.tunnel_dev.reset_mock()

        if no_activity_release_delay > 0:
            assert consider_disengage_timer.call_count == 1
            assert consider_disengage_timer.call_args[0][0] == no_activity_release_delay
            disengage_func = consider_disengage_timer.call_args[0][1]
            with self.patch_timer("autotrainer.behavior.system_machine._auto_clamp_release_timer") as release_clamp_timer:
                with caplog.at_level(logging.INFO):
                    disengage_func()

        else:
            algo.auto_clamp_release_load_count = 3
            with self.patch_timer("autotrainer.behavior.system_machine._auto_clamp_release_timer") as release_clamp_timer:
                for load_attempt in range(algo.auto_clamp_release_load_count):
                    assert release_clamp_timer.call_count == 0
                    with caplog.at_level(logging.INFO):
                        self.pellet.load_pellet(force=True)

        assert "auto-clamp: starting disengage procedure.." in caplog.text

        if release_tone_delay > 0:
            assert release_clamp_timer.call_count == 1
            assert release_clamp_timer.call_args[0][0] == release_tone_delay
            func = release_clamp_timer.call_args[0][1]
            assert func == machine._pre_disengage_auto_clamp
            with self.patch_timer(
                    "autotrainer.behavior.system_machine._auto_clamp_release_timer") as release_clamp_timer:
                func()

        if cfg.prerelease_duration > 0:
            assert update_magnet_mock.call_args_list == [mock.call(cfg.prerelease_intensity)]
            assert release_clamp_timer.call_count == 1
            assert release_clamp_timer.call_args[0][0] == cfg.prerelease_duration
            update_magnet_mock.reset_mock()
            func = release_clamp_timer.call_args[0][1]
            func()
        else:
            assert release_clamp_timer.call_count == 0

        assert update_magnet_mock.call_args_list == [mock.call(algo.baseline_intensity)]

    def test_headbar_reengaged_while_in_progress_do_not_reengage_clamp(self, machine, caplog):
        self.start_session_in_tunnel()
        assert not machine._auto_clamp_in_progress
        self.headbar_pressure.is_engaged = True
        assert machine._auto_clamp_in_progress
        self.headbar_pressure.is_engaged = False
        assert machine._auto_clamp_in_progress
        self.tunnel_dev.reset_mock()
        with caplog.at_level(logging.DEBUG):
            self.headbar_pressure.is_engaged = True
        assert "auto_clamp already in progress" in caplog.text
        assert self.update_magnet_mock.call_count == 0

    def test_exit_tunnel_without_engaged(self, machine):
        self.start_session_in_tunnel()
        assert machine.state == SystemState.tunnel
        assert self.update_magnet_mock.call_args_list == []
        self.exit_tunnel()
        assert machine.state == SystemState.cage
        assert self.update_magnet_mock.call_args_list == []
        assert self.pellet_dev.play_tone.call_args_list == []

    def test_when_algo_paused(self, caplog):
        self.start_session_in_tunnel()
        self.algo.algo_paused = True
        with caplog.at_level(logging.DEBUG):
            self.headbar_pressure.is_engaged = True
        assert "auto_clamp: algo-paused, skipping evaluate" in caplog.text

    def test_reset_to_baseline_when_disabled(self, caplog):
        algo = self.algo
        self.start_session_in_tunnel()
        with caplog.at_level(logging.DEBUG):
            self.headbar_pressure.is_engaged = True
        assert "auto-clamp setting position to" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            algo.head_fixation_enabled = False
        assert "auto-clamp disabled (backing off to baseline intensity)" in caplog.text

    def test_does_disengage_when_exit_tunnel(self, caplog):
        with caplog.at_level(logging.DEBUG):
            self.start_session_in_tunnel(engage_headbar=True)
        assert "auto-clamp setting position to" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.INFO):
            self.exit_tunnel()
        assert f"Disengaging auto-clamp to intensity {self.algo.baseline_intensity}" in caplog.text

    def test_fully_disengage_when_exit_tunnel(self, caplog):
        algo = self.algo
        algo.auto_clamp_release_load_count = 1  # for doing a single load-pellet to trigger disengage
        algo.active_config.head_clamp.prerelease_duration = 3  # > 0 for having multi-steps disengage procedure
        self.start_session_in_tunnel()
        with caplog.at_level(logging.DEBUG):
            self.headbar_pressure.is_engaged = True
        assert "auto-clamp setting position to" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.INFO):
            self.pellet.load_pellet(force=True)
        assert "auto-clamp: starting disengage procedure.." in caplog.text
        caplog.clear()
        self.tunnel_dev.reset_mock()  # ensure cleared
        with caplog.at_level(logging.DEBUG):
            self.exit_tunnel()
        assert self.tunnel_dev.update_head_magnet_intensity.call_args_list == [mock.call(self.algo.baseline_intensity)]
        assert (
            f"Disengaging auto-clamp to intensity {self.algo.baseline_intensity}"
            in caplog.text
        )

    def test_before_reengage_delay(self, caplog):
        algo = self.algo
        pellet = self.pellet
        algo.head_clamp_config.before_reengage_delay = 30
        algo.auto_clamp_release_load_count = 1  # for doing a single load-pellet to trigger disengage
        algo.head_clamp_config.auto_clamp_release_tone_delay = 0
        self.start_session_in_tunnel()
        with caplog.at_level(logging.DEBUG):
            self.headbar_pressure.is_engaged = True
        assert "auto-clamp setting position to" in caplog.text
        self.headbar_pressure.is_engaged = False
        caplog.clear()
        with caplog.at_level(logging.INFO):
            pellet.load_pellet(force=True)
        assert "auto-clamp: starting disengage procedure.." in caplog.text
        assert "Disengaging auto-clamp to intensity" in caplog.text
        algo.update_pellet_seen(True)
        self.mock_pellet_ack(until_none=True)
        with caplog.at_level(logging.DEBUG):
            self.headbar_pressure.is_engaged = True
        self.headbar_pressure.is_engaged = False
        assert f"delaying evaluate auto-clamp in {algo.head_clamp_config.before_reengage_delay:.1f}s" in caplog.text
        half_delay = algo.head_clamp_config.before_reengage_delay // 2
        self.increment_perf_now(half_delay)
        #
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            self.headbar_pressure.is_engaged = True
        self.headbar_pressure.is_engaged = False
        assert f"delaying evaluate auto-clamp in {half_delay:.1f}s" in caplog.text

    def test_not_trigger_when_in_intersession(self, caplog):
        algo = self.algo
        algo.intersession_enabled = True
        algo.head_fixation_enabled = False
        self.start_session_in_tunnel(engage_headbar=False)
        with self.mock_intersession_analysis():
            # algo.update_mouse_seen(True)  # ensure analysis will run
            self.mock_pose_response(pellet_seen=True, mouse_seen=True)
            self.exit_tunnel()
            assert self._machine.state == SystemState.intersession
            algo.head_fixation_enabled = True
            with caplog.at_level(logging.DEBUG):
                self.headbar_pressure.is_engaged = True
            self.headbar_pressure.is_engaged = False  # but back to False
        assert "auto-clamp: intersession not idle (no action taken)" in caplog.text
        assert self.tunnel_dev.update_head_magnet_intensity.call_count == 0
