from functools import partial
from typing import Callable, Optional

from transitions import Machine

from autotrainer.api import ApiEventKind

from autotrainer.core import ProjectInfo, transitions_allow_functions
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.interfaces import CaptureAnalysisResult

from . import IntersessionState
from ..inference_protocol import InferenceProtocol, SegmentationConfiguration, DetectionConfiguration
from ..behavior_algorithm import BehaviorAlgorithm
from ..state_machine import StateMachine, StateMachineEvents


logger = get_verbose_logger(__name__)


class IntersessionMachineEvents(StateMachineEvents):
    on_analysis_started: Callable[[], None]  # unused
    on_analysis_ended: Callable[[ProjectInfo, CaptureAnalysisResult], None]


class IntersessionMachine(StateMachine):

    _events_class = IntersessionMachineEvents

    states = list(IntersessionState)

    def __init__(
        self,
        *,
        algorithm: BehaviorAlgorithm,
        inference: InferenceProtocol,
    ):

        initial_state = IntersessionState.idle

        super().__init__(initial_state=initial_state)

        self._machine = Machine(model=[self], states=IntersessionMachine.states,
                                transitions=IntersessionMachine.transitions, auto_transitions=False,
                                initial=initial_state, model_override=True)

        algorithm.relay_transitions(self, wait=False)  # NB: must be done AFTER creation of previous machine instance

        self._algorithm = algorithm
        self._inference = inference
        self._segmentation_configuration: Optional[SegmentationConfiguration] = None
        self._detection_configuration: Optional[DetectionConfiguration] = None
        self._frame_rate: Optional[int] = None

    @property
    def events(self) -> IntersessionMachineEvents:  # to have correct type hint as well
        return self._events

    def reset_to_idle(self):
        self._detection_configuration = None
        self._segmentation_configuration = None
        self.state = IntersessionState.idle

    @property
    def frame_rate(self):
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, frame_rate):
        self._frame_rate = frame_rate

    def after_enter_segmentation(self, project_info: ProjectInfo):
        logger.success("entering segmentation with %s", project_info)
        segment_config = SegmentationConfiguration(
            project=project_info,
            frame_rate=self._frame_rate,
        )
        segment_config.complete = partial(self._segmentation_complete, segment_config=segment_config)
        self._segmentation_configuration = segment_config
        res = self._inference.perform_segmentation(segment_config)
        if res is None:
            logger.error("perform segmentation didn't started")
            self._segmentation_configuration = None
            with self._algorithm.set_allow_reentrant(True):
                self.end_analysis(project_info, False)
        else:
            self.events.on_analysis_started()
            self.post_event_content(ApiEventKind.intertrialSegmentationBegin, data=dict(trial_id=project_info.session))

    def after_enter_detection(self, segment_config: SegmentationConfiguration):
        detection_config = DetectionConfiguration(
            project=segment_config.project,
            frame_rate=segment_config.frame_rate,
        )
        detection_config.complete = partial(self._detection_complete, detection_config=detection_config)
        res = self._inference.perform_detection(detection_config)
        if res is not None:
            self._detection_configuration = detection_config
            self.post_event_content(ApiEventKind.intertrialDetectionBegin,
                                    data=dict(trial_id=segment_config.project.session))

    def after_end_analysis(self, project: ProjectInfo, success: bool):
        logger.info("end_analysis(success=%s) of %s", success, project)
        seg_cfg = self._segmentation_configuration
        det_cfg = self._detection_configuration
        self._segmentation_configuration = None
        self._detection_configuration = None
        if det_cfg is None and seg_cfg is None:
            logger.warning("Unexpected end_analysis while no segmentation or detection configuration, project=%s",
                           project)
        if seg_cfg is not None and seg_cfg.project != project:
            logger.warning("Unexpected segment config project: %s vs %s", seg_cfg.project, project)
        if det_cfg is not None and det_cfg.project != project:
            logger.warning("Unexpected detection config project: %s vs %s", det_cfg.project, project)
        result = CaptureAnalysisResult.ANALYSIS_SUCCEEDED if success else CaptureAnalysisResult.ANALYSIS_FAILED
        self._algorithm.end_session(project, result)
        self.events.on_analysis_ended(project, result)

    def can_perform_segmentation(self, project_info: ProjectInfo):
        p = project_info is not None
        i = self._inference is not None
        s = self._segmentation_configuration is None
        res = p and i and s
        logger.debug("can_perform_segmentation=%s: prj=%s inference=%s segment=%s", res, p, i, s)
        return res

    def can_perform_detection(self, segment_config: SegmentationConfiguration):
        s = segment_config is not None  # always true
        p = segment_config.project is not None  # always true
        i = self._inference is not None
        d = self._detection_configuration is None
        can_do_detection = p and i and d and s
        logger.debug("can_perform_detection=%s ; prj=%s inference=%s detection_config=%s segment_config=%s",
                    can_do_detection, p, i, d, s)
        return can_do_detection

    @BehaviorAlgorithm.relay_func(wait=False)
    def _segmentation_complete(self, success: bool, *,
                               segment_config: SegmentationConfiguration, error: str="NA"):
        self_seg_cfg = self._segmentation_configuration
        self._segmentation_configuration = None
        logger.verbose("segmentation_complete: success=%s config=%s ; error=%s",
                     success, segment_config, error)
        if self_seg_cfg is None or self_seg_cfg.project != segment_config.project:
            logger.warning("unexcepted internal segment config project: %s vs complete seg: %s",
                           None if self_seg_cfg is None else self_seg_cfg.project,
                           segment_config.project)
        project = segment_config.project
        if success:
            self.post_event_content(ApiEventKind.intertrialSegmentationEnd, data=dict(trial_id=project.session))
            if self.can_perform_detection(segment_config):  # must check, and if cannot must end_analysis
                def algo_action():
                    self.perform_detection(segment_config)
            else:
                logger.warning("cannot perform detection for %s", project)
                def algo_action():
                    self.end_analysis(project, False)
        else:
            logger.error("perform segmentation failed. config=%s", segment_config)
            self.post_event_content(
                ApiEventKind.intertrialSegmentationError, data=dict(error=error, trial_id=project.session))
            def algo_action():
                self.end_analysis(project, False)
        if algo_action is not None:
            with self._algorithm.set_allow_reentrant(True):
                algo_action()

    @BehaviorAlgorithm.relay_func(wait=False)
    def _detection_complete(self, success: bool, *,
                            detection_config: DetectionConfiguration, error: str="NA"):
        trial_id = detection_config.project.session
        if not success:
            logger.error("perform detection failed. det_config=%s", detection_config)
            self.post_event_content(ApiEventKind.intertrialDetectionError, data=dict(error=error, trial_id=trial_id))
        else:
            self.post_event_content(ApiEventKind.intertrialDetectionEnd, data=dict(trial_id=trial_id))
        with self._algorithm.set_allow_reentrant(True):
            self.end_analysis(detection_config.project, success)

    # region State Machine Requirements
    # Methods required for model_override=True to work.
    def trigger(self):
        """Main trigger"""

    def may_trigger(self):
        """Main trigger"""

    def perform_segmentation(self, project_info: ProjectInfo):
        """Perform segmentation"""

    def may_perform_segmentation(self):
        """Perform segmentation"""

    def perform_detection(self, segment_config: SegmentationConfiguration):
        """Perform detection"""

    def may_perform_detection(self):
        """Perform detection"""

    def end_analysis(self, project: ProjectInfo, success: bool):
        """End analysis"""

    def may_end_analysis(self):
        """End analysis"""

    def is_idle(self):
        """Is idle"""

    def is_segmentation(self):
        """Is segmentation"""

    def is_detection(self):
        """Is detection"""
    # endregion

    transitions = transitions_allow_functions([
        dict(
            trigger=perform_segmentation,
            source=IntersessionState.idle,
            dest=IntersessionState.segmentation,
            conditions=can_perform_segmentation,
            after=after_enter_segmentation,
        ),

        dict(
            trigger=perform_detection,
            source=IntersessionState.segmentation,
            dest=IntersessionState.detection,
            conditions=can_perform_detection,
            after=after_enter_detection,
        ),

        dict(
            trigger=end_analysis,
            source=[IntersessionState.segmentation, IntersessionState.detection],
            dest=IntersessionState.idle,
            after=after_end_analysis,
        ),
    ])
