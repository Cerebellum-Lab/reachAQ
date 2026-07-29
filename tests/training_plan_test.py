import contextlib
import copy
import dataclasses
import logging
import threading
import time
from pathlib import Path
from typing import Optional
from unittest import mock

import pytest

from autotrainer.behavior import SystemMachine, InferenceProtocol, BehaviorAlgorithm, TrainingMode, SystemState, \
    IntersessionState
from autotrainer.behavior.pellet import PelletState
from autotrainer.behavior.pellet_shift import ShiftXYZBufferHandler
from autotrainer.core import Offset3DTuple, EventManager, get_perf_now
from autotrainer.core.configuration.behavior_configuration import ShiftXYZBufferHandlerConfig
from autotrainer.device import CanDevice
from autotrainer.inference import InferenceStatus
from autotrainer.inference.analysis import IntersessionResponse
from autotrainer.core.capture import CaptureProcessStatus
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import AppModelStatus
from tools.acquisition.model.inference_model import InferenceModel
from tools.acquisition.model.training_plan import get_plan_id
from top_fixtures import MockSystemMachine

this_dir = Path(__file__).parent.resolve()


logger = logging.getLogger(__name__)


@pytest.fixture
def inference_model(pose_algo):
    # unused atm
    inference = InferenceModel(pose_algo)
    inference._status = InferenceStatus.live
    yield inference
    inference.terminate()


class BaseTrainingPlan(MockSystemMachine):

    @pytest.fixture(autouse=True)
    def _event_manager(self, monkeypatch):
        m_event_mgr = mock.create_autospec(EventManager)
        monkeypatch.setattr(f"{EventManager.__module__}.{EventManager.__qualname__}", m_event_mgr)
        monkeypatch.setattr(EventManager, "default", m_event_mgr.default)

    @pytest.fixture(autouse=True)
    def training_plans(self, trainer_config_dir):
        # have to copy or link the plan in the config dir given it's "hardcoded" relatively to it for now:
        dst_dir = trainer_config_dir.joinpath("training/protocols")
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in this_dir.joinpath("training/protocols").glob("*.json"):
            dst_dir.joinpath(p.name).write_bytes(p.read_bytes())

    @pytest.fixture(autouse=True)
    def _set_shift_xyz_config_and_others_non_default_config_settings(self, system_config, trainer_config_dir):
        cfg = system_config.behavior.shift_xyz_handler
        cfg.buffer.minimum_reach_fail = 2
        system_config.behavior.emergency_alarm.device_comm_error.use = False
        system_config.save_default(trainer_config_dir)

    @pytest.fixture()
    def app_model(
        self,
        machine,
        user_pref,
        fake_system_msg_handler,
        system_config,
        trainer_config_dir,
        calib_dir,
        training_plans,
        sensor_analysis,
        diamond_triangle_config,
        monkeypatch,
    ):
        machine._msg_handler = fake_system_msg_handler
        user_pref.save()
        msg_handler = machine._msg_handler

        # set and save some specific config we want to ease the below sessions test:
        algo_cfg = system_config.behavior
        algo_cfg.head_clamp.before_reengage_delay = 0
        # alarm_cfg = algo_cfg.emergency_alarm
        # for sub_cfg in (
        #     alarm_cfg.global_animal_presence,
        #     alarm_cfg.system_fault,
        #     alarm_cfg.system_maintenance,
        #     alarm_cfg.device_comm_error,
        # ):
        #     sub_cfg.is_emergency_condition = False  # in case of.
        system_config.save_default(trainer_config_dir)

        # prevent slow exit when issue:
        monkeypatch.setattr(CanDevice, CanDevice._check_tunnel_pellet_status_age.__name__, lambda s: None)

        app_model = self._app_model = AppModel(
            user_pref,
            system_message_handler=msg_handler,
            sensor_analysis=sensor_analysis,
            inference_model=machine._inference,
            calib_dir=calib_dir,
            system_machine=machine,
        )
        app_model.check_diamond_coord_enabled = False
        self._animal = app_model.add_animal("mouse1", select=True)  # select=True: also makes 1st pellet_send

        algo = app_model.behavior.algorithm

        assert app_model.loaded_configuration is None
        # NB: load_config -> reload_training_plan require IDLE
        assert app_model.load_configuration() is True
        assert app_model.loaded_configuration is not None

        assert algo.intersession_enabled is True  # required
        # NB: do not try change some settings after config is loaded,
        # the loaded parameters/settings (from config file) will be reused/reset with training plan enter.

        algo = self.algo
        # TODO:
        expected_plan_id = '57094295-5234-44f3-a925-14ceb772c8e5'
        expected_plan_id = '8fd60c5d-d543-45b9-8287-b5b97602f8e0'
        expected_plan_id = '58cc7a5b-8d75-40ff-9b8c-6d2b3a7ab814'
        plan = app_model.get_training_plan_by_id(expected_plan_id)
        animal = app_model.selected_animal
        animal.training.current_protocol = plan.plan_id
        app_model.training_plan = plan

        app_model.training_mode = TrainingMode.AUTOMATIC
        app_model.capture_start(target_status=AppModelStatus.ANIMAL_IN_TRAINING, wait_connected=False)

        algo.pellet_delivery_enabled = True

        self.mock_pellet_ack(until_none=True)  # for eventual whole start send-pellet/cover pellet sequence(s)

        try:
            yield app_model
        finally:
            app_model.capture_stop()
            app_model.on_close()

    def ack_pending_tokens(self, wait_acked: bool=True):
        tokens = list(self._app_model.hardware._pending_tokens)
        logger.info("acking %s pending tokens", len(tokens))
        for tok in tokens:
            self.msg_handler.ack_received(tok, perf_c=get_perf_now())
        if wait_acked:
            for tok in tokens:
                logger.debug("waiting tock %s", tok)
                self._app_model.hardware.wait_pending_command_acked(tok)


class TestTrainingPlan(BaseTrainingPlan):

    def test_training_plan(self, app_model, user_pref, machine, caplog):
    #     try:
    #         self._test_training_plan(app_model, user_pref, machine, caplog)
    #     finally:
    #         caplog.set_level(logging.CRITICAL)
    #         # NB: for some reason the last finally: above in app_model() fixture isn't called
    #         # when this test case fails for any reason. pytest seems to be stuck in some loop post-analysis code,
    #         # but before teardown, related to/with tmpdir fixture.. maybe the files we are possibly writing in it
    #         # are preventing pytest failure completion code to finish and put it in a kind of infinite loop state.
    #         app_model.capture_stop()
    #         app_model.on_close()
    #
    # def _test_training_plan(self, app_model, user_pref, machine, caplog):
        caplog.set_level(logging.DEBUG)  # REQUIRED to ensure we collect/see all the logs we want to assert on,
        # see below
        algo = self.algo
        plan = app_model.training_plan
        plan_start_phase = plan.current_phase

        assert plan_start_phase.advance_predicate.evaluate(plan_start_phase, plan._system_context) is False
        # NB:
        results = [
            IntersessionResponse(
                pellets_presented=3,
                successful_reaches=2,
                food_consumed=1,
                rh_max_vp_list=[Offset3DTuple(1, 0.5, 0.5)]
            ),
            IntersessionResponse(
                pellets_presented=3,
                successful_reaches=2,
                food_consumed=2,
                rh_max_vp_list=[Offset3DTuple(2, -0.5, -1)]
            ),
        ]

        nb_session = 2
        for session_idx in range(nb_session):
            self.increment_perf_now()
            assert "Received processed shift xyz: " not in caplog.text
            assert plan.current_phase == plan_start_phase
            caplog.clear()
            with caplog.at_level(logging.DEBUG):
                self._make_session(app_model, machine, results[session_idx])
            if session_idx == 0:
                assert plan.current_phase == plan_start_phase
                assert plan_start_phase.advance_predicate.evaluate(plan_start_phase, plan._system_context) is False

        # assert plan_start_phase.advance_predicate.evaluate(plan_start_phase, plan._system_context) is True, "phase should be able advance"
        # assert plan.current_phase != plan_start_phase, "the phase should have advanced"

        assert "Received processed shift xyz: (0, 3.0, -1.2)" in caplog.text, \
            "should be the some avg/mean of the 2 previous sessions, with limits applied"

        assert algo.pellets_consumed == results[-1].food_consumed
        assert algo.successful_reaches == results[-1].successful_reaches
        assert algo.pellets_presented == 1

        prev_phase = plan.current_phase
        #
        result = IntersessionResponse(
            pellets_presented=3,
            successful_reaches=3,
            food_consumed=3,
            rh_max_vp_list=[Offset3DTuple(1, 0.5, 0.5)],
        )
        caplog.clear()
        self._make_session(app_model, machine, result)
        # assert plan.current_phase != prev_phase
        # TODO: check
        #
        prev_phase = plan.current_phase
        result = IntersessionResponse(
            pellets_presented=3,
            successful_reaches=3,
            food_consumed=3,
            rh_max_vp_list=[Offset3DTuple(1, 0.5, 0.5)],
        )
        caplog.clear()
        self._make_session(app_model, machine, result)
        assert plan.current_phase == prev_phase, "3rd phase requires Hands near pellet seen < threshold"
        result.pellets_presented = result.successful_reaches = result.food_consumed = 8
        self._make_session(app_model, machine, result, hands_min_dist=0.001)
        # not working yet:
        # assert plan.current_phase != prev_phase, "3rd phase requires Hands near pellet seen < threshold"
        # TODO: TBC... assert phase action(s)

    def _make_session(self, app_model, machine, analysis_result, *,
                      hands_min_dist: Optional[float]=None):
        algo = app_model.behavior.algorithm
        if hands_min_dist is not None:
            algo.pellet_hands_min_distance = hands_min_dist
        # NB: at least one of the training phase is setting pellet-delivery-enabled to False, reset it here:
        algo.pellet_delivery_enabled = True
        #
        pellet_m = self._machine.pellet
        self.mock_pose_response(pellet_seen=True)
        self.mock_pellet_ack(until_none=True)
        #
        if machine.tunnel_headfix_enabled:
            self.start_session_in_tunnel(set_recording_status=True)
        else:
            assert algo.start_session(reason="manual")
            algo.set_capture_status(CaptureProcessStatus.RECORDING)
        self.mock_pellet_ack(until_none=True)
        #
        assert machine.state in {SystemState.cage, SystemState.tunnel}
        self.mock_pose_response(pellet_seen=True)
        self.mock_pellet_ack(until_none=True)
        assert algo.pellet_recently_seen
        if algo.head_fixation_enabled:
            for _ in range(2):
                self.sensor_analysis.headbar_pressure_monitor.is_engaged = True
                self.mock_pellet_ack(until_none=True)
                self.mock_pose_response(pellet_seen=True)
                self.sensor_analysis.headbar_pressure_monitor.is_engaged = False
        self.mock_pellet_ack(until_none=True)
        assert algo.is_in_session, (
            f"{algo.algo_paused=} {app_model.status=} {machine.state=} {machine.intersession.state=} {algo.head_fixation_enabled=}\n"
            f"{pellet_m.state=} {algo.status=}"
        )
        half = algo.active_config.pellet_delivery.pellet_send_wait_delay / 2
        self.increment_perf_now(half)
        self.mock_pose_response(pellet_seen=True)
        if pellet_m.state == PelletState.home:
            self.mock_pellet_ack(until_none=True)
        self.increment_perf_now(half + 0.01)
        self.mock_pose_response(pellet_seen=True)
        self.mock_pellet_ack(until_none=True)
        if pellet_m.state == PelletState.sending:
            self.mock_pellet_ack(until_none=True)
        self.mock_pose_response(pellet_seen=True)

        if pellet_m.state != PelletState.monitoring:
            x = algo.can_send_pellet()
        assert pellet_m.state == PelletState.monitoring, (
            f"{algo.algo_paused=} {app_model.status=} {machine.state=} {machine.intersession.state=} {algo.head_fixation_enabled=}\n"
            f"{pellet_m.state=} {algo.status=}"
        )
        # assert algo.can_release_pellet()  some training phase sets cover-pellet-enabled to True..
        #   .. making can_release_pellet() False.
        with contextlib.ExitStack() as stack:
            # to be sure:
            algo.update_triangle_seen(True)
            algo.update_mouse_seen(True)
            algo.update_pellet_seen(True)
            stack.enter_context(self.mock_perform_segmentation())
            stack.enter_context(self.mock_perform_detection())
            assert pellet_m.state == PelletState.monitoring  # still
            if machine.state == SystemState.tunnel:
                self.exit_tunnel()
            else:
                algo.end_capture_session()
            assert not algo.is_in_session
            assert algo.system_state == SystemState.intersession
            assert algo.intersession_state == IntersessionState.segmentation
            assert pellet_m.state == PelletState.retract  # Retract !!
            self.mock_complete_segmentation(True)
            assert algo.system_state == SystemState.intersession
            assert algo.intersession_state == IntersessionState.detection
            machine._inference.detection_result_ready(machine.project, analysis_result)
            self.mock_complete_detection(True)
            assert algo.intersession_state == IntersessionState.idle
            assert algo.system_state == SystemState.cage
            assert pellet_m.state == PelletState.retract  # still

        assert not algo.is_in_session

        assert pellet_m.state == PelletState.retract  # still
        # NB: at least one of the training phase is setting pellet-delivery-enabled to False, reset it here:
        algo.pellet_delivery_enabled = True
        # so that pellet-send will be allowed with ack of previous retract:

        self.mock_pellet_ack()  # for retract
        assert pellet_m._api_status_token is None
        # assert pellet_m.covered_state is False  # depends todo

        assert machine.state == SystemState.cage  # still ofc.

        self.increment_perf_now(1)
