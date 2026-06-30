import functools
import logging
import threading
from queue import Queue
from typing import Callable, List, Tuple, Optional

from ..analysis.sensor_analysis import SensorAnalysis

from .message_handler import MessageHandler
from .system_status_message import SystemStatusMessageKind

logger = logging.getLogger(__name__)


class SystemMessageHandler(MessageHandler):

    def __init__(self, input_queue: Queue, *, sensor_analysis: Optional[SensorAnalysis] = None):
        super().__init__(input_queue, name="system-message-handler")

        self._measurement_callback = None
        self._audio_callback = None

        self._analysis = SensorAnalysis() if sensor_analysis is None else sensor_analysis

    @property
    def measurement_callback(self):
        return self._measurement_callback

    @measurement_callback.setter
    def measurement_callback(self, measurement_callback: Callable[[Tuple], None]) -> None:
        self._measurement_callback = measurement_callback

    @property
    def audio_callback(self):
        return self._audio_callback

    @audio_callback.setter
    def audio_callback(self, audio_callback: Callable[[List[float]], None]) -> None:
        prev = self._audio_callback
        if prev is not None:
            logger.info("Replacing audio callback %s with %s", prev, audio_callback)
        self._audio_callback = audio_callback

    @property
    def analysis(self):
        return self._analysis

    def message_received(self, msg, data):
        # TODO: These are treated as if the property has changed.  If the number of event listeners increases or their
        #  behaviors are complex and do not check for change themselves, this could become a bottleneck.  This could be
        #  updated to store previous values and only notify listeners on change, like a typical ObservableObject
        #  implementation.  Keeping things simple for the time being.
        if msg == SystemStatusMessageKind.MEASUREMENT or msg == SystemStatusMessageKind.MEASUREMENTS:
            measures = self._analysis.measurements_received(data)
            if self._measurement_callback is not None and len(measures) > 0:
                self._measurement_callback(measures)

        elif msg == SystemStatusMessageKind.AUDIO_SPECTRUM:
            self._analysis.audio_spectrum_received(data)
            if self._audio_callback is not None:
                self._audio_callback(data.magnitudes)

        elif msg == SystemStatusMessageKind.PELLET_X:
            self.property_changed(MessageHandler.DEVICE_X_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.PELLET_Y:
            self.property_changed(MessageHandler.DEVICE_Y_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.PELLET_Z:
            self.property_changed(MessageHandler.DEVICE_Z_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.PELLET_MOTOR_X:
            self.property_changed(MessageHandler.DEVICE_X_PROPERTY, data.position, None)
            self.property_changed(MessageHandler.STEPPER_X_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.PELLET_MOTOR_Y:
            self.property_changed(MessageHandler.DEVICE_Y_PROPERTY, data.position, None)
            self.property_changed(MessageHandler.STEPPER_Y_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.PELLET_MOTOR_Z:
            self.property_changed(MessageHandler.DEVICE_Z_PROPERTY, data.position, None)
            self.property_changed(MessageHandler.STEPPER_Z_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.PELLET_LOAD:
            self.property_changed(MessageHandler.LOAD_ARM_ANGLE_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.PELLET_COVER:
            self.property_changed(MessageHandler.COVER_ARM_ANGLE_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.HEAD_MAGNET:
            self.property_changed(MessageHandler.HEAD_MAGNET_INTENSITY_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.TUNNEL_GATE_SERVO:
            self.property_changed(MessageHandler.HEAD_GATE_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.TUNNEL_GATE_OPEN_STATUS:
            self.property_changed(MessageHandler.TUNNEL_GATE_OPEN_STATUS, data, None)

        elif msg == SystemStatusMessageKind.TUNNEL_FAN:
            self.property_changed(MessageHandler.TUNNEL_FAN_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.FRONT_DOOR:
            # directly update related monitor, hopefully is very fast
            if self._analysis.alarms:
                self._analysis.external_doors_alarm.update_door_state(msg, data)
            self.property_changed(MessageHandler.FRONT_DOOR_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.DRAWER_DOOR:
            # directly update related monitor, hopefully is very fast
            if self._analysis.alarms:
                self._analysis.external_doors_alarm.update_door_state(msg, data)
            self.property_changed(MessageHandler.DRAWER_DOOR_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.SPARE_DOOR:
            self.property_changed(MessageHandler.SPARE_DOOR_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.EXT_BUTTON:
            self.property_changed(MessageHandler.EXT_BUTTON_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.STIMULUS_INPUTS:
            self.property_changed(MessageHandler.STIMULI_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.MOTOR_CONFIGURATION:
            self._on_property_changed(MessageHandler.CONFIG_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.COLOR_LED:
            self._on_property_changed(MessageHandler.COLOR_LED, data, None)

        else:
            logger.warning("unhandled msg %s", msg)
