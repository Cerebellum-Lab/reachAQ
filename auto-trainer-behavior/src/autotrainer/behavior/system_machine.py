import dataclasses
import math
from functools import partial
from typing import Optional

from transitions import Machine

from autotrainer.api import ApiEventKind

from autotrainer.core import (ProjectInfo, SensorAnalysis, Offset3DTuple,
                              HeadbarPressureMonitor, transitions_allow_functions, SystemMessageHandler, get_perf_now)
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.interfaces import (CaptureAnalysisResult, RecordingEndingReason)
from autotrainer.core.pose_elements import SceneElement, AllHandsParts
from autotrainer.core.multiproc import make_daemon_timer, no_op_timer
from autotrainer.core.video_detection import PresenceDetectionAttrs
from autotrainer.core.analysis.detector import BaseDetector
from autotrainer.core.diamond_triangle_config import DiamondTriangleOffsetConfig
from autotrainer.core.configuration.behavior_configuration import (
    HeadClampReleaseMode,
)

from autotrainer.inference import PoseResponse, InferenceCommandMessageKind
from autotrainer.inference.analysis import IntersessionResponse

from .behavior_algorithm import BehaviorAlgorithm, BehaviorAlgoProps, BehaviorAlgoStatus
from .inference_protocol import InferenceProtocol
from .intersession import IntersessionMachine, IntersessionState
from .pellet import PelletState
from .pellet.pellet_machine import PelletMachine
from .pellet_device_protocol import PelletDeviceProtocol
from .pellet_shift import ShiftXYZHandler
from .state_machine import StateMachine
from .system_machine_state import SystemState
from .tunnel_device_protocol import TunnelDeviceProtocol

logger = get_verbose_logger(__name__)

# NB: this is to ensure we can patch the exact desired one (and only that one) from tests:
_auto_clamp_release_timer = make_daemon_timer
_check_missing_timer = make_daemon_timer
_consider_disengage_autoclamp_timer = make_daemon_timer

#


class SystemMachine(StateMachine):

    # _events_class

    states = list(SystemState)

    def __init__(self,
                 *,
                 msg_handler: SystemMessageHandler,
                 analysis: SensorAnalysis,
                 tunnel_device: TunnelDeviceProtocol,
                 pellet_device: PelletDeviceProtocol,
                 inference: InferenceProtocol,
                 algorithm: Optional[BehaviorAlgorithm] = None,
                 project_info: Optional[ProjectInfo] = None,
                 topcam_presence: Optional[PresenceDetectionAttrs] = None,
                 tunnel_headfix_enabled: bool = True,
                 ):

        initial_state = SystemState.cage
        super().__init__(initial_state=initial_state)

        self.machine = Machine(
            model=[self],
            states=self.states,
            transitions=self.transitions,
            auto_transitions=False,
            initial=initial_state,
            model_override=True,
        )

        if project_info is None:
            project_info = ProjectInfo()
        project_info: ProjectInfo
        self._project_info = project_info
        self._aborted_session_ids = set()
        #
        # todo: should probably make/use a state "Machine" for AutoClamp itself
        self._auto_clamp_in_progress = False
        self._auto_clamp_disengage_in_progress = False
        self._timer_auto_clamp_evaluate = no_op_timer
        self._timer_auto_clamp_disengage = no_op_timer
        self._disengage_auto_clamp_load_count = 0
        self._last_disengage_autoclamp_perf_c = -math.inf

        self._is_handling_diamond_triangle = False

        self._enter_tunnel_pellet_seen = False

        self._session_started_perf_c = -math.inf

        self._pellet_device = pellet_device
        self._tunnel_device = tunnel_device
        self._tunnel_headfix_enabled = tunnel_headfix_enabled
        self._msg_handler = msg_handler

        algo = self._algorithm = BehaviorAlgorithm(
            project_info=project_info,
            topcam_presence=topcam_presence,
        ) if algorithm is None else algorithm
        del algorithm  # using algo

        algo.session_starting += self._on_session_capture_started
        algo.session_capture_ending += self.on_session_capture_ended
        algo.property_changed += self._on_algorithm_property_changed
        algo.relay_transitions(self, wait=False)  # NB: must be done AFTER creation of previous `self.machine` instance

        shift_xyz_handler = self._shift_xyz_handler = ShiftXYZHandler(algo=algo)
        # NB: could use the shift_xyz_handler.property_changed callback handler with LAST_PROCESSED_SHIFT_XYZ name too:
        shift_xyz_handler.set_processed_handler(self._handle_processed_shift_xyz)
        #

        def sync_algo_system_state(_, new_state):
            self._algorithm.system_state = new_state
        self.events.state_changed += sync_algo_system_state
        algo.system_state = self._state  # to be sure

        self._analysis = analysis
        if analysis is not None:
            analysis.headbar_pressure_monitor.property_changed += self._on_headbar_pressure_monitor_property_changed
            analysis.auto_tunnel_sweep_monitor.property_changed += self._on_auto_tunnel_sweep_property_changed
            # analysis.pellet_misplaced_monitor.dcs_config = algo.diamond_triangle_config
            #   handled by property changed.
            # set current configs from monitors:
            algo.active_config.auto_tunnel_sweep = analysis.auto_tunnel_sweep_monitor.config
            # analysis.pellet_misplaced_monitor.config  # not in system config for now

        self._inference = inference
        inference.pose_response_ready += self._on_pose_changed
        inference.detection_result_ready += self._on_detection_result_ready
        inference.segmentation_finished += self._on_inference_segmentation_finished

        pellet_machine = self._pellet_machine = PelletMachine(algo, msg_handler, pellet_device)
        pellet_events = pellet_machine.events
        pellet_events.state_changed += self._on_pellet_state_changed
        pellet_events.pellet_loading += self._on_pellet_loading
        pellet_events.pellet_loaded += self._on_pellet_loaded
        pellet_events.pellet_sent += self._on_pellet_sent
        pellet_events.pellet_load_failed += self._on_pellet_load_failed
        pellet_events.pellet_released += self._on_pellet_released
        self._last_pellet_loaded_perf_c = -math.inf
        self._last_pellet_loading_perf_c = -math.inf
        self._last_pellet_failed_loaded_perf_c = -math.inf

        intersession_machine = self._intersession = IntersessionMachine(
            algorithm=algo,
            inference=inference,
        )
        intersession_machine.events.on_analysis_ended += self._on_intersession_analysis_ended
        intersession_machine.events.state_changed += self._on_intersession_state_changed

    def cancel_timers(self):
        for timer in (
            self._timer_auto_clamp_disengage,
            self._timer_auto_clamp_evaluate,
        ):
            if not timer.finished.is_set():
                logger.debug("cancelling timer %s", timer)
                timer.cancel()
        self._timer_auto_clamp_disengage = no_op_timer

    @property
    def analysis(self) -> SensorAnalysis:
        return self._analysis

    @property
    def algorithm(self) -> BehaviorAlgorithm:
        return self._algorithm

    @property
    def pellet(self) -> PelletMachine:
        return self._pellet_machine

    @property
    def intersession(self) -> IntersessionMachine:
        return self._intersession

    @property
    def tunnel_headfix_enabled(self) -> bool:
        return self._tunnel_headfix_enabled

    @tunnel_headfix_enabled.setter
    def tunnel_headfix_enabled(self, value: bool):
        prev, self._tunnel_headfix_enabled = self._tunnel_headfix_enabled, value
        if prev != value:
            logger.info("tunnel/headfix behavior enabled changed to: %s", value)
        if not value:
            self._timer_auto_clamp_evaluate.cancel()
            self._timer_auto_clamp_disengage.cancel()
            self._autoclamp_set_in_progress(False)

    @property
    def project(self) -> ProjectInfo:
        return self._project_info

    @project.setter
    def project(self, value: ProjectInfo):
        logger.verbose("Received new project-info, relaying to event manager, algo and inference ..")
        self._project_info = value
        self._event_manager.project = value
        self._algorithm.project = value
        self._inference.project = value

    @property
    def shift_xyz_handler(self) -> ShiftXYZHandler:
        return self._shift_xyz_handler

    def can_enter_tunnel(self, *, reason: str = "NA") -> bool:
        if self._tunnel_headfix_enabled:
            return True
        logger.warning("Blocked enter_tunnel(%s): tunnel/headfix hardware is disabled", reason)
        return False

    def can_exit_intersession_to_tunnel(self) -> bool:
        if self._tunnel_headfix_enabled:
            return True
        logger.warning("Blocked exit_intersession_to_tunnel: tunnel/headfix hardware is disabled")
        return False

    def before_enter_tunnel(self, *, reason: str = "NA"):
        pellet_state = self._pellet_machine.state
        self._enter_tunnel_pellet_seen = self._algorithm.pellet_recently_seen
        logger.debug("before_enter_tunnel: reason=%s state=%s pellet_state=%s pellet_recently_seen=%s",
                     reason, self._state, pellet_state, self._enter_tunnel_pellet_seen)
        if self._state == SystemState.cage:
            self._event_manager.post_event_content(ApiEventKind.tunnelEnter)
            # always when enter tunnel, but only if was in cage before.
            self._execute_disengage_auto_clamp_if_in_progress()

    def after_enter_tunnel(self, *, reason: str = "NA"):
        if self._analysis is not None:
            self._evaluate_auto_clamp(caller="after_enter_tunnel")

    def after_exit_tunnel(self, *, reason: str = "NA"):
        logger.verbose("after_exit_tunnel: %s", reason)
        with self._algorithm.set_allow_reentrant(True):
            self._execute_disengage_auto_clamp_if_in_progress()
        self._event_manager.post_event_content(ApiEventKind.tunnelExit)

    def after_enter_intersession(self, project_info: ProjectInfo, *, reason="NA"):
        algo = self._algorithm
        intersession = self._intersession
        inference = self._inference
        logger.verbose("enter_intersession: reason=%s, in-session=%s",
                       reason, algo.is_in_session)

        logger.info("processing session project %s", project_info)
        algo.session_processing_starting()
        with algo.set_allow_reentrant(True):
            intersession.perform_segmentation(project_info)
        inference.send_message(InferenceCommandMessageKind.ProcessOffline, (project_info, True))
        # pellet machine can react to system-state == intersession,
        # which is here the case (def *after*_enter_intersession):
        with algo.set_allow_reentrant(True):
            self._pellet_machine.environment_changed()

    def after_exit_intersession(self):
        if not self._tunnel_headfix_enabled:
            with self._algorithm.set_allow_reentrant(True):
                self.exit_intersession_to_cage()
            return
        # Return to cage; the operator owns the next recording lifecycle.
        self._tunnel_device.open_tunnel_gate()
        self._execute_disengage_auto_clamp_if_in_progress()
        with self._algorithm.set_allow_reentrant(True):
            self.exit_intersession_to_cage()

    def after_exit_intersession_to_cage(self):
        # ensure pellet goes back where necessary:
        self._pellet_machine.environment_changed(caller="exit_intersession_to_cage")

    def after_exit_intersession_to_tunnel(self):
        self.enter_tunnel(reason="exit_intersession_to_tunnel")

    @BehaviorAlgorithm.relay_func(wait=False)
    def _on_session_capture_started(self):
        project = self._project_info
        self._aborted_session_ids.discard(project.short_id)
        dcs_send_pos = self._pellet_device.last_dcs_set_position
        if dcs_send_pos is None:
            logger.warning("current dcs_send_pos None (DCS), diamond-triangle not calibrated?")
        p_now = self._session_started_perf_c = get_perf_now()
        pellet_recent_seen = self._algorithm.is_pellet_recently_seen()
        pellet_m = self._pellet_machine
        logger.verbose("session_capture_started: dcs_send_pos=%s prj.when=%s",
                       dcs_send_pos, project.when)
        # Ensure inference has the current session identity.
        project.send_position = self._pellet_device.last_set_position
        project.dcs_send_position = dcs_send_pos
        logger.info("Associated dcs_send_pos=%s with project", dcs_send_pos)
        if pellet_m.state == PelletState.monitoring and pellet_recent_seen:
            self._set_pellet_delivered_presented(project, 0)
        self._inference.project = project
    @BehaviorAlgorithm.relay_func(wait=False)
    def on_session_capture_ended(self, reason: RecordingEndingReason):
        cur_project = self._project_info.to_local_value()
        logger.verbose("capture_ended: project=%s", cur_project)
        algo = self._algorithm
        if reason == RecordingEndingReason.MANUAL_ABORT:
            self._aborted_session_ids.add(cur_project.short_id)
            self._inference.send_message(InferenceCommandMessageKind.SetOfflineToLive)
            return
        if not algo.active_config.pellet_delivery.retract_enabled:
            # then stays at current position, whatever it is,
            # it will be resumed/continued once intersession finishes (and inference comes back live).
            pass
        elif self._pellet_machine.state == PelletState.monitoring:
            with algo.set_allow_reentrant(True):
                self._pellet_machine.move_retract()

        can_perform_analysis = (
            algo.can_perform_intersession_analysis()
            and self._intersession.can_perform_segmentation(cur_project)
        )
        logger.notice(
            "session ended: prj=%s intersession.state=%s system_machine.state=%s algo.system_state=%s "
            "pellet_machine.state=%s intersession_enabled=%s session_mouse_seen=%s "
            "can_perform_analysis=%s",
            cur_project.short_id,
            self._intersession.state, self.state, algo.system_state,
            self._pellet_machine.state,
            algo.intersession_enabled, algo.session_mouse_seen,
            can_perform_analysis,
        )
        if can_perform_analysis:
            with algo.set_allow_reentrant(True):
                self.enter_intersession(cur_project, reason="capture-ended-and-can-perform-analysis")
        else:
            # at the end of live recording pose-process automatically goes to offline mode,
            # so we ask it to switch back to live:
            self._inference.send_message(InferenceCommandMessageKind.SetOfflineToLive)
            algo.end_session(cur_project, CaptureAnalysisResult.CAPTURE_ONLY)

    @BehaviorAlgorithm.relay_func(wait=False)
    def _on_intersession_analysis_ended(self, project: ProjectInfo, result: CaptureAnalysisResult):
        if project.short_id in self._aborted_session_ids:
            logger.info("Ignoring analysis completion for aborted %s", project.short_id)
            return
        logger.verbose("intersession ended: trial=%s result=%s", project.short_id, result)
        algo = self._algorithm
        with algo.set_allow_reentrant(True):
            self.exit_intersession()

    @BehaviorAlgorithm.relay_func(wait=False)
    def _on_inference_segmentation_finished(self, project: ProjectInfo, success: bool, *, error: str="NA"):
        logger.verbose("got inference segmentation finished: %s ; err=%s prj=%s", success, error, project)
        self._inference.send_message(InferenceCommandMessageKind.SetOfflineToLive)

    @BehaviorAlgorithm.relay_func(wait=False)
    def _on_headbar_pressure_monitor_property_changed(self, name: str, value, _):
        if not self._tunnel_headfix_enabled:
            return
        if name == HeadbarPressureMonitor.IS_ENGAGED_PROPERTY:
            self._event_manager.post_event_content(
                ApiEventKind.headFixationForceDetectorChanged, data=dict(is_enabled=value))
            if value:
                self._evaluate_auto_clamp(caller="headbar_pressure_on")

    @BehaviorAlgorithm.relay_func(wait=False)
    def _evaluate_auto_clamp(self, *, caller: str="NA"):
        if not self._tunnel_headfix_enabled:
            logger.debug("auto-clamp skipped because tunnel/headfix hardware is disabled")
            return
        algo = self._algorithm
        if algo.algo_paused:
            logger.debug("auto_clamp: algo-paused, skipping evaluate")
            return
        if self._auto_clamp_in_progress:
            logger.debug("auto_clamp already in progress")
            return
        is_headbar_pressure_engaged = self._analysis.headbar_pressure_monitor.is_engaged
        if not algo.head_fixation_enabled:
            logger.info("auto-clamp: disabled (no action taken)")
            return
        if self._intersession.state != IntersessionState.idle:
            logger.info("auto-clamp: intersession not idle (no action taken)")
            return
        if not algo.is_in_session:
            logger.info("auto-clamp: algo not in-session (no action taken)")
            return
        if not is_headbar_pressure_engaged:
            logger.info("auto-clamp: detector not engaged (no action taken)")
            return
        self._timer_auto_clamp_evaluate.cancel()
        p_now = get_perf_now()
        cfg = algo.active_config.head_clamp
        disengage_age = p_now - self._last_disengage_autoclamp_perf_c
        remains = cfg.before_reengage_delay - disengage_age
        if remains > 0:
            logger.verbose("delaying evaluate auto-clamp in %.1fs due to recent disengage ; age=%.1fs",
                         remains, disengage_age)
            timer = make_daemon_timer(remains, partial(self._evaluate_auto_clamp, caller="timer"))
            self._timer_auto_clamp_evaluate = timer
            timer.start()
            return
        intensity = cfg.auto_clamp_intensity
        logger.info("auto-clamp setting position to %s ; caller=%s", intensity, caller)
        self._autoclamp_set_in_progress(True)
        self._update_magnet_position(intensity)
        self._event_manager.post_event_content(ApiEventKind.autoClampEngaged, data=dict(intensity=intensity))
        self._disengage_auto_clamp_load_count = 0
        self._timer_auto_clamp_disengage.cancel()  # in case of
        if cfg.release_mode == HeadClampReleaseMode.ACTIVITY:
            t_delay = cfg.auto_clamp_no_activity_release_delay
        else:
            t_delay = cfg.fixed_duration_release_delay
        if t_delay > 0:
            logger.debug("starting new timer for disengage_auto_clamp in %.2f seconds", t_delay)
            new_timer = self._timer_auto_clamp_disengage = _consider_disengage_autoclamp_timer(
                t_delay, self._disengage_auto_clamp,
            )
            new_timer.start()

    def _evaluate_home_on_excessive_drift(self):
        # might be todo: convert to a detector
        algo = self._algorithm
        home_on_drift_cfg = algo.home_on_excessive_drift_distance_config
        nb_points = algo.diamond_triangle_drift_data_points_size
        #
        if not (
            home_on_drift_cfg.enabled
            and nb_points >= home_on_drift_cfg.min_samples
        ):
            return
        # also reset if distance is good,
        # so that we'll have to get min_samples data point before next check
        cur_drift = algo.get_diamond_triangle_drifts(reset=True, show_log=False)
        drift_dist = math.nan if cur_drift is None else cur_drift.distance
        if math.isnan(drift_dist) or drift_dist < home_on_drift_cfg.excessive_distance_threshold:
            return
        logger.notice("Measured motor drift too high (%.1fmm), executing home procedure",
                      drift_dist)
        self._event_manager.post_event_content(
            ApiEventKind.pelletDriftReset, data=dict(drift=dict(x=cur_drift.x, y=cur_drift.y, z=cur_drift.z)))
        self._pellet_machine.move_home()

    # @BehaviorAlgorithm.relay_func(wait=False)
    # not needed, already called by _pose_changed which has already it.
    def _handle_diamond_triangle_offset_changed(self, offset: Optional[Offset3DTuple]):
        if offset is None:
            return
        algo = self._algorithm
        if (
            # self._state == SystemState.tunnel
            # TODO: we could also decide to check in SystemState.cage as well,
            #  as far as we can ensure pellet is at deliver/send position
            self._pellet_machine.state == PelletState.monitoring
            and self._pellet_machine.can_use_pellet_command()
        ):
            last_pos = self._pellet_device.last_position
            if last_pos is not None:
                if not self._is_handling_diamond_triangle:
                    self._is_handling_diamond_triangle = True
                    logger.info("Starting handling diamond-triangle offset ; current offset=%s pos=%s",
                                offset.humanize(), last_pos.humanize())
                    # ensure we get refreshed data:
                    algo.get_diamond_triangle_drifts(reset=True, show_log=False)
                    # don't show log, to not show most likely bad value due to previous motor move
                algo.handle_diamond_triangle_offset(offset, last_pos)
                # if not algo.is_in_session:
                self._evaluate_home_on_excessive_drift()
        else:
            if self._is_handling_diamond_triangle:
                self._is_handling_diamond_triangle = False
                algo.get_diamond_triangle_drifts(show_log=True)

    def _handle_triangle_pellet_offset_changed(self, offset: Optional[Offset3DTuple]):
        if offset is None:  # not sure we should not let it pass to algo
            return
        self._algorithm.triangle_pellet_offset = offset

    def _handle_star_triangle_offset_changed(self, offset: Optional[Offset3DTuple]):
        if offset is None:
            return
        pellet_machine = self._pellet_machine
        if not pellet_machine.can_use_pellet_command():
            # never consider any release or cover check when pellet cannot be used yet.
            return
        algo = self._algorithm
        # only check cover if not in session and pellet covered state is True (== covered) and in monitoring
        check_cover_distance = not algo.is_in_session and (
            (pellet_machine.state == PelletState.monitoring
             and pellet_machine.covered_state is True)
        )
        if check_cover_distance:
            algo.handle_cover_pellet_offset(offset)
            # ofc never consider the check release position when we checked the cover one
            return
        # otherwise, given can_use_pellet_command() is True (check above),
        # we know we have to check release pos distance if state is monitoring and covered state is False ( == released)
        check_release_distance = (
            pellet_machine.state == PelletState.monitoring
            and pellet_machine.covered_state is False
        )
        if check_release_distance:
            algo.handle_release_pellet_offset(offset)

    def _handle_pellet_uncover(self, response: PoseResponse):
        project = self._project_info
        algo = self._algorithm
        active_cfg = self._algorithm.active_config
        if not (algo.is_in_session and active_cfg.pellet_delivery.is_pellet_cover_enabled):
            return
        pellet_m = self._pellet_machine
        if pellet_m.covered_state is False:  # already uncovered/released
            return
        if not math.isfinite(project.t_pellet_delivered):  # wait pellet is delivered
            return
        uncov_cfg = active_cfg.pellet_uncover
        min_y = math.inf
        max_y = -math.inf
        for part in AllHandsParts:
            part_3d = response.locations_3d.get(part, None)
            if part_3d is not None:
                if part_3d.y < min_y:
                    min_y = part_3d.y
                if part_3d.y > max_y:
                    max_y = part_3d.y
        perf_now = get_perf_now()
        valid = max_y < uncov_cfg.min_y_dcs
        ctx = self._algorithm.uncover_context
        prev_valid = ctx.y_dcs_valid
        if not prev_valid and valid:
            logger.verbose("setting pellet-uncover valid ; dist: min=%.1f max=%.1f loc3d=%s",
                           min_y, max_y, response.locations_3d)
            ctx.start_min_y = min_y
        elif not valid and prev_valid:
            logger.verbose("unsetting pellet-uncover valid ; max_dist=%.1f loc3d=%s",
                           max_y, response.locations_3d)
        if valid:
            if not math.isfinite(ctx.start_y_dcs_valid_perf_c):
                ctx.start_y_dcs_valid_perf_c = perf_now
        else:
            # as well, so that compare with minimum duration threshold give expected result
            ctx.start_y_dcs_valid_perf_c = perf_now

        ctx.y_dcs_valid = valid

    @BehaviorAlgorithm.relay_func(wait=False)
    def _on_pose_changed(self, response: PoseResponse):
        p_now = get_perf_now()
        analysis = self._analysis
        algo = self._algorithm
        if algo.is_in_session and not algo.session_mouse_seen and response.mouse_seen:
            logger.success("session first mouse_seen: parts=%s locations=%s", response.parts_flags, response.locations)
        if __debug__:
            t_last = getattr(self, "_last_pose_changed_logged", 0)
            p_now = get_perf_now()
            if p_now - t_last >= 30:
                logger.debug("pose_changed: %s", response.round(2))
                self._last_pose_changed_logged = p_now
        #
        pellet_3d = response.locations_3d.get(SceneElement.Pellet)
        analysis.pellet_misplaced_monitor.update(pellet_3d)
        #
        self._handle_diamond_triangle_offset_changed(
            response.get_parts_3d_offset(SceneElement.Diamond, SceneElement.Triangle))

        self._handle_star_triangle_offset_changed(
            response.get_parts_3d_offset(SceneElement.Star, SceneElement.Triangle))

        self._handle_triangle_pellet_offset_changed(
            response.get_parts_3d_offset(SceneElement.Triangle, SceneElement.Pellet))
        #
        algo.update_parts_seen(response)  # replace many previous update_xxx_seen()
        new_pellet_recently_seen = algo.is_pellet_recently_seen()
        if math.isinf(self._last_pellet_loaded_perf_c) and new_pellet_recently_seen:
            # ensure ok if pellet already loaded on start
            # take the min of response and p_now for loading_perf_c:
            self._last_pellet_loading_perf_c = min(response.perf_c, p_now)
            # so that loaded_perf_c is always >= than loading.
            self._last_pellet_loaded_perf_c = p_now
            logger.info("set first last_pellet_loading=%.4f and last_pellet_loaded=%.4f",
                        self._last_pellet_loading_perf_c, self._last_pellet_loaded_perf_c)
        #
        self._handle_pellet_uncover(response)
        self._pellet_machine.pellet_seen(response.pellet_seen)

    # AUTO-CLAMP / HEAD-BAR

    def _autoclamp_set_in_progress(self, prog: bool):
        self._auto_clamp_in_progress = prog
        self._algorithm.autoclamp_in_progress = prog
        det = self._analysis.autoclamp_evasion_detector
        det.autoclamp_in_progress = prog

    @BehaviorAlgorithm.relay_func(wait=False)
    def _execute_disengage_auto_clamp_if_in_progress(self):
        self._timer_auto_clamp_evaluate.cancel()  # in case of
        self._timer_auto_clamp_disengage.cancel()  # better needed
        if not self._auto_clamp_in_progress:
            return
        baseline_intensity = self._algorithm.baseline_intensity
        logger.info("Disengaging auto-clamp to intensity %s", baseline_intensity)
        self._last_disengage_autoclamp_perf_c = get_perf_now()
        self._update_magnet_position(baseline_intensity)
        self._event_manager.post_event_content(ApiEventKind.autoClampDisengaged, data=dict(intensity=baseline_intensity))
        self._autoclamp_set_in_progress(False)
        self._auto_clamp_disengage_in_progress = False
        with self._algorithm.set_allow_reentrant(True):
            self._pellet_machine.environment_changed()

    @BehaviorAlgorithm.relay_func(wait=False)
    def _pre_disengage_auto_clamp(self):
        clamp_cfg = self._algorithm.head_clamp_config
        self._timer_auto_clamp_evaluate.cancel()  # not sure that we want this one
        self._timer_auto_clamp_disengage.cancel()  # ensure no other disengage timer remain
        pre_duration = clamp_cfg.prerelease_duration
        if pre_duration > 0:
            intensity = clamp_cfg.prerelease_intensity
            logger.verbose("setting head-clamp to pre-release intensity %s", intensity)
            self._update_magnet_position(intensity)
            self._event_manager.post_event_content(ApiEventKind.autoClampPreDisengage, data=dict(intensity=intensity))
            logger.debug("started timer for really disengage auto-clamp in %.1fs", pre_duration)
            timer = self._timer_auto_clamp_disengage = _auto_clamp_release_timer(
                pre_duration, self._execute_disengage_auto_clamp_if_in_progress
            )
            timer.start()
        else:
            self._execute_disengage_auto_clamp_if_in_progress()

    @BehaviorAlgorithm.relay_func(wait=False)
    def _disengage_auto_clamp(self):
        if not self._auto_clamp_in_progress:
            logger.debug("skipping disengage auto-clamp if not in progress")
            return
        if self._auto_clamp_disengage_in_progress:
            logger.debug("skipping new disengage while disengage already in progress")
            return
        self._auto_clamp_disengage_in_progress = True
        logger.info("auto-clamp: starting disengage procedure..")
        self._timer_auto_clamp_evaluate.cancel()  # no sure about this one
        self._timer_auto_clamp_disengage.cancel()
        pellet_dev = self._pellet_device
        algo = self._algorithm
        clamp_cfg = algo.head_clamp_config
        if algo.is_in_session:
            freq = clamp_cfg.auto_clamp_release_tone_freq
            logger.debug("sending tone (freq=%s) to indicate auto-clamp disengaging", freq)
            pellet_dev.play_tone(freq, 0.5)
            self._event_manager.post_event_content(ApiEventKind.autoClampPlayReleaseTone,
                                                   data=dict(frequency=freq, duration=0.5))
        after_tone_delay = algo.auto_clamp_release_tone_delay
        if after_tone_delay > 0:
            logger.debug(
                "changing magnet to baseline intensity in %.2f seconds", after_tone_delay)
            timer = self._timer_auto_clamp_disengage = _auto_clamp_release_timer(
                after_tone_delay,
                self._pre_disengage_auto_clamp,
            )
            timer.start()
        else:
            with BehaviorAlgorithm.set_allow_reentrant(True):
                self._pre_disengage_auto_clamp()

    def _set_pellet_delivered_presented(self, project: ProjectInfo, t_rel_start: float):
        if not math.isfinite(project.t_pellet_delivered):
            logger.debug("set project.t_pellet_delivered=%.3f", t_rel_start)
            project.t_pellet_delivered = t_rel_start
        #
        if (
            math.isfinite(project.t_pellet_delivered)
            and not math.isfinite(project.t_pellet_presented)
            and self._pellet_machine.covered_state is False
        ):
            logger.debug("set project.t_pellet_presented=%.3f", t_rel_start)
            project.t_pellet_presented = t_rel_start
            self._event_manager.post_event_content(
                ApiEventKind.trialPelletPresented, data=dict(trial_id=project.session))

    @BehaviorAlgorithm.relay_func(wait=False)
    def _on_algorithm_property_changed(self, name: str, new_value, old_value):
        # Always back off to the baseline intensity when auto-clamp is disabled.
        pellet_dev = self._pellet_device
        props = BehaviorAlgoProps
        #
        if name == props.HEAD_FIXATION_ENABLED:
            if not new_value:
                logger.debug("auto-clamp disabled (backing off to baseline intensity)")
                if self._tunnel_headfix_enabled:
                    self._disengage_auto_clamp()
                # todo: don't we want : self._execute_disengage_auto_clamp() ?

        elif name == props.AUTO_CORRECT_MOTOR_DRIFT:
            pellet_dev.set_auto_correct_motor_drift(new_value)

        elif name == props.ALGO_PAUSED:
            algo = self._algorithm
            tunnel_dev = self._tunnel_device
            self.cancel_timers()
            # don't leave in-progress:
            if self._auto_clamp_in_progress:
                with algo.set_allow_reentrant(True):
                    self._execute_disengage_auto_clamp_if_in_progress()
            if new_value:
                if algo.status != BehaviorAlgoStatus.IDLE:
                    with algo.set_allow_reentrant(True):
                        action_funcs = [lambda: self._pellet_machine.move_home(force=True)]
                        if self._tunnel_headfix_enabled:
                            action_funcs[:0] = [
                                tunnel_dev.open_tunnel_gate,
                                lambda: self._update_magnet_position(0),
                            ]
                        for action_func in action_funcs:
                            try:
                                action_func()
                            except Exception as err:
                                logger.warning("%s failed: %s, but continuing", action_func, err)
            else:
                if algo.status != BehaviorAlgoStatus.IDLE:
                    if self._tunnel_headfix_enabled:
                        tunnel_dev.open_tunnel_gate()
                        self._update_magnet_position(algo.baseline_intensity)
                # Pellet-machine resumes its current operation from live conditions.

        elif name == props.DIAMOND_TRIANGLE_CONFIG:
            self._analysis.pellet_misplaced_monitor.dcs_config = new_value

    def _on_auto_tunnel_sweep_property_changed(self, name, value, _):
        if not self._tunnel_headfix_enabled:
            return
        if name == BaseDetector.IS_ENGAGED:
            if value:
                self._pellet_device.set_tunnel_fan_on()
            else:
                self._pellet_device.set_tunnel_fan_off()

    def _on_pellet_loading(self):
        algo = self._algorithm
        analysis = self._analysis
        p_now = get_perf_now()

        # ensure this new loading was preceded by a successful pellet_loaded:
        consumed = (
            self._last_pellet_loaded_perf_c >
            self._last_pellet_loading_perf_c >
            self._last_pellet_failed_loaded_perf_c
        )
        logger.verbose("on_pellet_loading: consumed=%s last_loaded=%.4f last_loading=%.4f  last_failed=%.4f",
                    consumed, self._last_pellet_loaded_perf_c, self._last_pellet_loading_perf_c,
                    self._last_pellet_failed_loaded_perf_c)
        self._last_pellet_loading_perf_c = p_now
        if consumed and self._tunnel_headfix_enabled:
            analysis.autoclamp_evasion_detector.increment_pellets_consumed()

        clamp_cfg = algo.active_config.head_clamp
        self._disengage_auto_clamp_load_count += 1
        if self._tunnel_headfix_enabled and clamp_cfg.release_mode == HeadClampReleaseMode.ACTIVITY:
            if self._disengage_auto_clamp_load_count >= clamp_cfg.auto_clamp_release_load_count:
                self._disengage_auto_clamp()

    def _on_pellet_loaded(self):
        p_now = get_perf_now()
        self._last_pellet_loaded_perf_c = p_now
        logger.verbose("received pellet_loaded p_now=%.4f", self._last_pellet_loaded_perf_c)
        self._algorithm.pellet_loaded()
        self._analysis.system_maintenance_alarm.update_failed_pellet_load(consecutive=0)

    def _on_pellet_load_failed(self, *, consecutive: int):
        logger.verbose("received pellet_load_failed consecutive=%s", consecutive)
        self._last_pellet_failed_loaded_perf_c = get_perf_now()
        self._analysis.system_maintenance_alarm.update_failed_pellet_load(consecutive=consecutive)

    def _on_pellet_state_changed(self, old_value, new_value):
        logger.verbose("pellet_state_changed: %s -> %s", old_value, new_value)
        algo = self._algorithm
        if new_value == PelletState.releasing:
            if algo.is_in_session:
                algo.pellet_uncover_context.has_released = True

    def _on_pellet_sent(self, *, perf_c: float):
        if self._algorithm.is_in_session:
            project = self._project_info
            if not math.isfinite(project.t_pellet_delivered):
                t_delivered = perf_c - self._algorithm.recording_start_perf_c
                self._set_pellet_delivered_presented(project, t_delivered)

    def _on_pellet_released(self, *, perf_c: float):
        project = self._project_info
        # for some reason the cover arm looks to sometimes still be finishing its move up to ~2-3 frames after
        # the perf_c we receive, it might be it's related to its inertia.
        perf_c += 0.012  # small correction for best result
        algo = self._algorithm
        logger.debug(
            "_on_pellet_released: perf_c=%.2f in_session=%s sess_started=%.2f project=%s",
            perf_c, algo.is_in_session, self._session_started_perf_c, project)
        if algo.is_in_session:
            self._set_pellet_delivered_presented(project, perf_c - algo.recording_start_perf_c)

    def _update_magnet_position(self, position: float):
        if not self._tunnel_headfix_enabled:
            logger.debug("Skipping head magnet update because tunnel/headfix hardware is disabled")
            return
        self._tunnel_device.update_head_magnet_intensity(position)

    def _on_intersession_state_changed(self, old, new):
        self._algorithm.intersession_state = new

    @BehaviorAlgorithm.relay_func(wait=False)
    def _on_detection_result_ready(self, prj: ProjectInfo, res: IntersessionResponse):
        if prj.short_id in self._aborted_session_ids:
            logger.info("Ignoring detection result for aborted %s", prj.short_id)
            return
        if prj.short_id != self._project_info.short_id:
            logger.warning(
                "Ignoring detection result for non-current session %s (current=%s)",
                prj.short_id,
                self._project_info.short_id,
            )
            return
        logger.success("Intersession analysis result: prj=%s result=%s", prj, res)
        self._shift_xyz_handler.put_intersession_response(prj, res)
        #
        algo = self._algorithm
        algo.set_previous_intersession_analysis_rsp(prj, res)
        #
        if res.food_consumed > 0:
            algo.increase_pellets_consumed(res.food_consumed)
        if res.successful_reaches > 0:
            algo.increase_successful_reaches(res.successful_reaches)
        # NB: now using pellet-sent event to count presented.
        # if res.pellets_presented > 0:
        #     algo.increase_pellets_presented(res.pellets_presented)
        if res.total_reaches > 0:
            algo.increase_pellet_total_reaches(res.total_reaches)
        #

    def _handle_processed_shift_xyz(self, project: ProjectInfo, shift: Offset3DTuple):
        logger.success("Received processed shift xyz: %s ; project=%s", shift.round(1), project)
        self._apply_processed_shift_xyz(shift)

    def _apply_processed_shift_xyz(self, shift: Offset3DTuple):
        algo = self._algorithm
        dev = self._pellet_device
        if dev is None or not algo.intersession_pellet_shift_enabled:
            logger.debug("intersession_pellet_shift_enabled False or pellet_device None")
            return
        # flips are at the moment statics, but handle possible custom flips ; defensive:
        cfg = algo.diamond_triangle_config
        if cfg is None:
            cfg = DiamondTriangleOffsetConfig
        logger.notice("applying pellet send_position shift: %s", shift.round(1))
        # NB: dev.set_x/y/z is in motor coordinate system,
        # but we want the shifts to be in inference system :
        for idx, (val, meth) in enumerate((
            (shift[0], dev.set_x),
            (shift[1], dev.set_y),
            (shift[2], dev.set_z),
        )):
            coord = "XYZ"[idx]
            if val != 0:
                logger.debug("applying shift-%s with: %.1f", coord, val)
                val *= cfg.flips_motor_diamond[idx]
                token = meth(val, absolute=False, sender="processed_shift_xyz")
                if token is None:
                    logger.error("Could not apply shift-%s ; command not successfully sent", coord)
                    # TODO: what todo ?
            else:
                logger.debug("%s == 0 ; skip", coord)

    # region State Machine Requirements
    # Methods required for model_override=True to work.
    def trigger(self):
        """Trigger"""

    def may_trigger(self):
        """Trigger"""

    def enter_tunnel(self, *, reason: str = "NA"):
        """Enter tunnel"""

    def may_enter_tunnel(self):
        """Enter tunnel"""

    def exit_tunnel(self, *, reason: str = "NA"):
        """Exit tunnel"""

    def may_exit_tunnel(self):
        """Exit tunnel"""

    def enter_intersession(self, project_info: ProjectInfo, *, reason: str="NA"):
        """Enter intersession"""

    def may_enter_intersession(self):
        """May Enter intersession"""

    def reenter_intersession(self, project_info: ProjectInfo, *, reason: str="NA"):
        """ReEnter intersession (from previous intersession)"""

    def may_reenter_intersession(self):
        """May ReEnter intersession"""

    def exit_intersession(self):
        """Exit intersession"""

    def exit_intersession_to_tunnel(self):
        """Exit intersession"""

    def exit_intersession_to_cage(self):
        """Exit intersession"""

    def may_exit_intersession(self):
        """Exit intersession"""

    def may_exit_intersession_to_tunnel(self):
        """Exit intersession"""

    def may_exit_intersession_to_cage(self):
        """Exit intersession"""

    def is_cage(self):
        """Is cage"""

    def is_tunnel(self):
        """Is tunnel"""

    def is_intersession(self):
        """Is intersession"""
    # endregion

    transitions = transitions_allow_functions([
        dict(
            trigger=enter_tunnel,
            source=[SystemState.cage, SystemState.tunnel],
            dest=SystemState.tunnel,
            conditions=can_enter_tunnel,
            before=before_enter_tunnel,
            after=after_enter_tunnel,
        ),

        dict(
            trigger=exit_tunnel,
            source=SystemState.tunnel,
            dest=SystemState.cage,
            after=after_exit_tunnel,
        ),

        dict(
            trigger=enter_intersession,
            source=(SystemState.cage, SystemState.tunnel),
            dest=SystemState.intersession,
            after=after_enter_intersession,
        ),

        dict(
            trigger=reenter_intersession,
            source=SystemState.intersession,
            dest=SystemState.intersession,
            after=after_enter_intersession,
        ),

        dict(  # previous behavior
            trigger=exit_intersession,
            source=SystemState.intersession,
            dest=SystemState.intersession,
            after=after_exit_intersession,
        ),

        dict(
            trigger=exit_intersession_to_tunnel,
            source=SystemState.intersession,
            dest=SystemState.tunnel,
            conditions=can_exit_intersession_to_tunnel,
            after=after_exit_intersession_to_tunnel,
        ),
        dict(
            trigger=exit_intersession_to_cage,
            source=SystemState.intersession, dest=SystemState.cage,
            after=after_exit_intersession_to_cage,
        )
    ])
