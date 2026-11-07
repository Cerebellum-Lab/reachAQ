import dataclasses
from typing import Optional

from autotrainer.behavior import SystemMachine, InferenceProtocol, BehaviorAlgorithm, SystemState, IntersessionState
from autotrainer.core import (ObservableObject, ProjectInfo, BehaviorConfiguration,
                              SystemMessageHandler, ApiEventKind)
from autotrainer.core.analysis import ReachAnalysis
from autotrainer.core.event import post_api_event_content
from autotrainer.core.logging import get_verbose_logger
from tools.acquisition.model.hardware_model import HardwareModel

from autotrainer.core.project import ProjectDependentProtocol

logger = get_verbose_logger(__name__)


class BehaviorModel(ObservableObject, ProjectDependentProtocol):
    """
    Encapsulation of the Behavior Module (autotrainer-behavior) for the application layer.  This model class manages
    aspects of the behavior system that are specific to the application.  General behavior functionality should be
    located in the module.

    Retired AutoTrainer alarm, emergency, and tunnel behavior is not part of
    this model.
    """

    def __init__(
        self,
        msg_handler: SystemMessageHandler,
        analysis: ReachAnalysis,
        hardware_model: HardwareModel,
        inference: InferenceProtocol,
        *,
        system_machine: Optional[SystemMachine] = None,
    ):
        super().__init__()

        self._project: Optional[ProjectInfo] = None

        self._analysis = analysis
        if system_machine is None:
            system_machine = SystemMachine(
                msg_handler=msg_handler,
                analysis=analysis,
                pellet_device=hardware_model,
                inference=inference,
            )
        self._system_machine: SystemMachine = system_machine
        self._hardware_model = hardware_model
        #
        # system_machine.pellet.events.state_changed += lambda old_val, new_val: self._on_property_changed(
        #     f"pellet.{StateMachine.Properties.STATE_PROPERTY}", new_val, old_val)
        # actually unused event (pellet.state)

    @property
    def analysis(self) -> ReachAnalysis:
        return self._analysis

    @property
    def project(self) -> ProjectInfo:
        return self._project

    @project.setter
    def project(self, value: ProjectInfo) -> None:
        self._project = value
        self._system_machine.project = value
        # self._machine.project = value  # instead of having to do it in on_prepare_capture()

    @property
    def system_machine(self) -> SystemMachine:
        return self._system_machine

    @property
    def algorithm(self) -> BehaviorAlgorithm:
        return self._system_machine.algorithm

    def load_configuration(self, config: BehaviorConfiguration):
        system_m = self._system_machine
        system_m.shift_xyz_handler.set_config(config.shift_xyz_handler)
        system_m.algorithm.load_configuration(config)

    def save_configuration(self) -> BehaviorConfiguration:
        algo = self._system_machine.algorithm

        assigned = {}
        created = False
        class ConfigWrap(BehaviorConfiguration):
            def __setattr__(self, key, value):
                if created:
                    assigned[key] = value

        config = ConfigWrap()
        created = True

        config = dataclasses.replace(algo.active_config, **assigned)
        orig_fields = {f.name for f in dataclasses.fields(config)}
        missed = orig_fields - set(assigned)
        if len(missed) > 0:
            logger.debug("Fields %s not assigned / missed during save_config", missed)

        return config

    def on_prepare_capture(self):
        self._system_machine.project = self._project
        self._system_machine.state = SystemState.ready  # forced,
        self._system_machine.intersession.state = IntersessionState.idle
        # if acquisition is/was stopped during an intersession analysis,
        # then it's left on intersession+(segmentation | detection) state..
        # which further prevent everything after.
        # todo: try have intersession stop "normally" too

    def get_led_color(self):
        """Use a steady green pellet-board indicator while acquisition runs."""
        return (0, 100, 0)
