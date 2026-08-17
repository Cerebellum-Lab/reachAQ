import logging
from queue import Queue

from .message_handler import MessageHandler
from .system_status_message import SystemStatusMessageKind

logger = logging.getLogger(__name__)


class SystemMessageHandler(MessageHandler):

    def __init__(self, input_queue: Queue):
        super().__init__(input_queue, name="system-message-handler")

    def message_received(self, msg, data):
        # TODO: These are treated as if the property has changed.  If the number of event listeners increases or their
        #  behaviors are complex and do not check for change themselves, this could become a bottleneck.  This could be
        #  updated to store previous values and only notify listeners on change, like a typical ObservableObject
        #  implementation.  Keeping things simple for the time being.
        if msg == SystemStatusMessageKind.PELLET_X:
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

        elif msg == SystemStatusMessageKind.STIMULUS_INPUTS:
            self.property_changed(MessageHandler.STIMULI_PROPERTY, data, None)

        elif msg in {
            SystemStatusMessageKind.TONE_STATUS,
            SystemStatusMessageKind.BOARD_TIME_SYNC,
            SystemStatusMessageKind.BOARD_CAPABILITIES,
            SystemStatusMessageKind.DIGITAL_PULSE_STATUS,
        }:
            # The decoded-message event carries this timestamp-sensitive state to
            # acquisition consumers.  It has no legacy UI property of its own.
            pass

        elif msg == SystemStatusMessageKind.MOTOR_CONFIGURATION:
            self._on_property_changed(MessageHandler.CONFIG_PROPERTY, data, None)

        elif msg == SystemStatusMessageKind.COLOR_LED:
            self._on_property_changed(MessageHandler.COLOR_LED, data, None)

        else:
            logger.warning("unhandled msg %s", msg)
