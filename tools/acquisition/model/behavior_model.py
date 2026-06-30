import dataclasses
from datetime import datetime
from typing import Optional, Callable

from autotrainer.behavior import SystemMachine, InferenceProtocol, BehaviorAlgorithm, SystemState, IntersessionState
from autotrainer.core import (ObservableObject, ProjectInfo, SensorAnalysis, BehaviorConfiguration,
                              SystemMessageHandler, ApiEventKind)
from autotrainer.core.event import post_api_event_content
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.video_detection import PresenceDetectionAttrs
from tools.acquisition.model.hardware_model import HardwareModel

from autotrainer.core.project import ProjectDependentProtocol

logger = get_verbose_logger(__name__)


class BehaviorModel(ObservableObject, ProjectDependentProtocol):
    """
    Encapsulation of the Behavior Module (autotrainer-behavior) for the application layer.  This model class manages
    aspects of the behavior system that are specific to the application.  General behavior functionality should be
    located in the module.

    Emergency stopped/resumed events are retained for compatibility with upstream code, but reachAQ disables alarm-driven
    emergency behavior and rejects direct emergency-stop/resume requests explicitly.
    """

    # events type hint
    emergency_stopped: Callable[[str], None]
    emergency_resumed: Callable[[str], None]

    def __init__(
        self,
        msg_handler: SystemMessageHandler,
        analysis: SensorAnalysis,
        hardware_model: HardwareModel,
        inference: InferenceProtocol,
        *,
        topcam_presence: Optional[PresenceDetectionAttrs] = None,
        system_machine: Optional[SystemMachine] = None,
    ):
        super().__init__(("emergency_stopped", "emergency_resumed"))

        self._project: Optional[ProjectInfo] = None

        self._analysis = analysis
        if system_machine is None:
            system_machine = SystemMachine(
                msg_handler=msg_handler,
                analysis=analysis,
                tunnel_device=hardware_model,
                pellet_device=hardware_model,
                inference=inference,
                topcam_presence=topcam_presence,
                tunnel_headfix_enabled=hardware_model.tunnel_headfix_enabled,
            )
        self._system_machine: SystemMachine = system_machine
        self._hardware_model = hardware_model
        #
        self._source_emergency: Optional[str] = None
        #
        # system_machine.pellet.events.state_changed += lambda old_val, new_val: self._on_property_changed(
        #     f"pellet.{StateMachine.Properties.STATE_PROPERTY}", new_val, old_val)
        # actually unused event (pellet.state)

        hardware_model.property_changed += self._hardware_model_property_changed

    @BehaviorAlgorithm.relay_func(wait=False)
    def _hardware_model_property_changed(self, name, value, _):
        if name == HardwareModel.TUNNEL_HEADFIX_ENABLED:
            self._system_machine.tunnel_headfix_enabled = value
            if not value:
                self._disable_tunnel_headfix_behavior()

    def _disable_tunnel_headfix_behavior(self):
        analysis = self._analysis
        algo = self._system_machine.algorithm
        logger.info("Disabling tunnel/headfix-dependent behavior because the hardware is disabled")
        algo.head_fixation_enabled = False
        algo.active_config.head_clamp.enabled = False
        algo.auto_close_gate_on_intersession_config.enabled = False
        analysis.auto_tunnel_sweep_monitor.config.enabled = False
        analysis.auto_tunnel_sweep_monitor.stop()

    @property
    def analysis(self) -> SensorAnalysis:
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
        analysis = self._analysis
        analysis.headbar_pressure_monitor.config = config.headbar_pressure
        analysis.load_cell_monitor.load_configuration(config.load_cell)
        analysis.load_cell_tare_monitor.config = config.auto_tare
        analysis.audio_thrashing_monitor.config = config.audio
        analysis.auto_tunnel_sweep_monitor.config = config.auto_tunnel_sweep
        analysis.autoclamp_evasion_detector.config = config.autoclamp_evasion_detector
        system_m.tunnel_headfix_enabled = self._hardware_model.tunnel_headfix_enabled
        if not self._hardware_model.tunnel_headfix_enabled:
            self._disable_tunnel_headfix_behavior()

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

        analysis = self._analysis

        # NB: monitors/detectors configuration:
        config.load_cell = analysis.load_cell_monitor.save_configuration()
        config.auto_tare = analysis.load_cell_tare_monitor.save_configuration()
        config.headbar_pressure = analysis.headbar_pressure_monitor.config
        config.audio = analysis.audio_thrashing_monitor.config

        top_cam = algo.top_camera_presence_detection
        config.topcam_presence_detection = None if top_cam is None else top_cam.to_config()
        config.autoclamp_evasion_detector = analysis.autoclamp_evasion_detector.config

        config = dataclasses.replace(algo.active_config, **assigned)
        orig_fields = {f.name for f in dataclasses.fields(config)}
        missed = orig_fields - set(assigned)
        if len(missed) > 0:
            logger.debug("Fields %s not assigned / missed during save_config", missed)

        return config

    def on_prepare_capture(self):
        self._system_machine.project = self._project
        self._system_machine.state = SystemState.cage  # forced,
        self._system_machine.intersession.state = IntersessionState.idle
        # if acquisition is/was stopped during an intersession analysis,
        # then it's left on intersession+(segmentation | detection) state..
        # which further prevent everything after.
        # todo: try have intersession stop "normally" too

    def use_current_head_magnet_position_as_baseline(self):
        head_magnet_intensity = self._hardware_model.head_magnet_intensity
        if head_magnet_intensity is not None:
            algo = self._system_machine.algorithm
            algo.baseline_intensity = head_magnet_intensity
            # NB: behavior_algo.baseline_intensity is currently not connected to config value,
            # but we want save it here:
            algo.active_config.head_clamp.baseline_intensity = head_magnet_intensity
            post_api_event_content(ApiEventKind.headfixBaselineChanged,
                                   data=dict(baseline=head_magnet_intensity))

    @property
    def source_emergency(self) -> Optional[str]:
        return self._source_emergency

    @BehaviorAlgorithm.relay_func()
    def emergency_stop(self, source: str):
        raise RuntimeError(
            f"Emergency stop is disabled in reachAQ; request source={source!r}. "
            "Use stop acquisition for controlled shutdown."
        )

    @BehaviorAlgorithm.relay_func()
    def emergency_resume(self, source: str):
        raise RuntimeError(
            f"Emergency resume is disabled in reachAQ; request source={source!r}. "
            "No emergency pause state is maintained."
        )

    def get_led_color(self, *, now: Optional[datetime]=None):
        if now is None:
            now = datetime.now()
        algo = self._system_machine.algorithm
        cfg_led = algo.active_config.led_alarm
        color = (0, 100, 0)
        cur_time = now.time()
        start, stop = cfg_led.start_ignore_hour, cfg_led.stop_ignore_hour
        in_ignore_window = (
            (start <= cur_time <= stop)
            if start < stop
            else (cur_time >= start or cur_time <= stop)
        )
        if in_ignore_window:
            color = (0, 0, 0)
        return color
