import contextlib
import logging
import math
import time
import uuid
from queue import Queue, Empty
from threading import Thread
from typing import Callable, Union, Optional, Any, Set


from autotrainer.api import ApiEventKind

from autotrainer.core import (
    MotorConfigurations,
    SystemCommandKind,
    SystemStatusMessageKind,
    get_perf_now,
)
from autotrainer.core.event import post_api_event_content

import autotrainer.device
from .can_device import HAVE_CAN_DEVICE
from .can_failure import CanFailure, CanFailureKind
from .device import Device
from .device_api import DeviceApi
from .device_interface import DeviceInterface, ServoConfig, StepperConfig
from .device_connection_protocol import DeviceConnectionProtocol
from .motor_steps import CompoundMovementDataSet, MotorSteps
from autotrainer.core.logging import get_verbose_logger

logger = get_verbose_logger(__name__)

_REQUEST_CONNECT = -1002
_REQUEST_DISCONNECT = -1003


class DeviceConnection(DeviceConnectionProtocol):
    """
    Convenience class to connect a hardware device and a responder for a client to control and receive information from
    the device.

    The Device object is responsible for interpreting data from the device to message the client, and interpreting
    messages from the client to send data to the device.

    The optional message queue and callback are interfaces for the client script or application to exchange data with
    the device.

    It is this class's responsibility to enable one- or two-way communication with the device, depending on the
    arguments provided, in a non-blocking fashion.
    """

    def __init__(self,
                 device: Device,
                 message_queue: Queue,
                 message_callback: Callable[[int, object], None] = None,
                 name="device-connection",
                 failure_callback: Optional[Callable[[CanFailure], None]] = None):

        super().__init__()

        # The message queue and the callback are ways to get data from the device back to the client script or
        # application that created this object.  Commands and data to the device are sent through send message.

        # without explicit import, this allows to have completion working on these instances attributes access:
        self._device: Union[Device, "autotrainer.device.can_device.CanDevice"] = device
        self._interface: Union[DeviceInterface, "autotrainer.device.can_interface.CanInterface"] = device.device_interface
        self._message_callback = message_callback
        self._message_queue = message_queue
        self._cmd_queue: Queue = Queue()

        self._api = DeviceApi(message_callback=message_callback, message_queue=message_queue)
        self._device.api = self._api

        self._name = name
        self._failure_callback = failure_callback

        self._read_limit: int = 50 if HAVE_CAN_DEVICE else 2000
        self._collect_ms: int = 5  # so freq == 200 Hz

        # The means of providing non-blocking access to the device.
        self._current_thread: Optional[Thread] = None
        self._current_thread_watchdog_perf_c = math.nan
        # NB: this is simply the dedicated CAN bus reader thread

    @property
    def watchdog_reader_perf_c(self) -> float:
        thread = self._current_thread
        return math.nan if (thread is None or not thread.is_alive()) else self._current_thread_watchdog_perf_c

    @property
    def watchdog_writer_perf_c(self) -> float:
        dev = self._device
        return math.nan if dev is None else dev.writer_watchdog_perf_c

    @property
    def device(self) -> Device:
        return self._device

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def read_limit(self) -> int:
        return self._read_limit

    @read_limit.setter
    def read_limit(self, value: int):
        self._read_limit = value

    def join(self):
        # TODO this is for legacy compatibility when the connection was exposed directly as a thread for clients that
        #  wanted to know when the connection was guaranteed to be terminated.  This should be accommodated a different
        #  way - see TODOs in request_(dis)connect.
        thread = self._current_thread
        if thread is not None:
            logger.debug("joining %s", thread)
            thread.join(3)
            if thread.is_alive():
                logger.warning("thread %s still alive, but continuing", thread)
            self._current_thread = None

    def request_connect(self):
        """
        Attempts to establish a connection to the device.  This is framed as a request for two reasons.  Even in the
        case of a successful connection, it will not have been fully established at the time this call returns.  The
        second reason is that the connection attempt may fail.
        """
        # TODO provide a mechanism for the caller to be notified when a connection attempt succeeds or fails.  Could be
        #  an optional callback provided in this call, a dedicated callback, an observable property, etc.
        self._start()

        if self._cmd_queue is not None:
            self._cmd_queue.put((_REQUEST_CONNECT, None, None))

    def wait_connected(self, *, timeout: float = 3):
        """Wait for a requested connection without consuming command timeout."""
        started = time.perf_counter()
        deadline = started + timeout
        while not self._device.connected:
            thread = self._current_thread
            if thread is not None and not thread.is_alive():
                raise RuntimeError(
                    f"device connection failed after {time.perf_counter() - started:.3f}s: "
                    "connection worker stopped before device.connected"
                )
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"device connection timeout after {timeout:.3f}s waiting for device.connected"
                )
            time.sleep(0.01)

    def request_disconnect(self):
        """
        Sends a disconnect request to the device connection queue.  It is framed as a request because the device may not
        be disconnected and relevant objects yet disposed when this call returns.  However, anything running and
        allocated with be terminated and disposed.
        """
        # TODO provide a mechanism for the caller to be notified when disconnection is complete.
        cmd_queue = self._cmd_queue
        if cmd_queue is not None:
            logger.debug("requesting disconnect")
            cmd_queue.put((_REQUEST_DISCONNECT, None, None))
        dev = self._device
        if dev is not None:
            logger.verbose("disconnecting from %s", dev)
            dev.disconnect()

    @contextlib.contextmanager
    def await_acknowledge(self, tokens: Set, *, timeout: float=1, raise_on_timeout=True):
        orig_cb = self._api.message_callback
        tokens_acked = []
        def cb(kind, context):
            if kind == SystemStatusMessageKind.ACKNOWLEDGE:
                tok, *r_args = context
                if tok in tokens:
                    tokens_acked.append(tok)
                    tokens.remove(tok)
            elif orig_cb is not None:
                orig_cb(kind, context)
        self._api.message_callback = cb
        try:
            yield
            logger.verbose("Now waiting tokens %s", tokens)
            perf_timeout = time.perf_counter() + timeout
            while len(tokens) > 0:
                if time.perf_counter() > perf_timeout:
                    if raise_on_timeout:
                        raise RuntimeError(f"command timeout waiting for acknowledge tokens: {tokens}")
                    logger.warning("command timeout waiting for acknowledge tokens, but continuing: %s", tokens)
                    break
                time.sleep(0.001)
            if len(tokens) == 0:
                logger.info("successfully obtained %s acknowledge", len(tokens_acked))
        finally:
            self._api.message_callback = orig_cb

    def send_message(self, kind: int, data: Optional[Any] = None, context: Optional[Any] = None):
        """Send a command/message to the device (writer-thread)"""
        post_api_event_content(ApiEventKind.deviceCommandSend, data=dict(context=context))
        self._device.notify_message(kind, data, context)

    def use_compound_movements(self, data: CompoundMovementDataSet):
        for kind, steps in (
            (SystemCommandKind.SET_LOAD_PELLET_PROCEDURE, data.load_pellet),
            (SystemCommandKind.SET_SEND_PELLET_PROCEDURE, data.send_pellet),
            (SystemCommandKind.SET_COVER_PELLET_PROCEDURE, data.cover_pellet),
            (SystemCommandKind.SET_RELEASE_PELLET_PROCEDURE, data.release_pellet),
            (SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE, data.move_retract),
        ):
            if steps.is_empty:
                logger.notice("config compound %s empty, not using. builtin default config will remain.", kind)
            else:
                self.send_message(kind, steps)

    def use_motor_configurations(self, data: MotorConfigurations):
        logger.notice("Setting motor configurations")
        tokens = set()
        def make_token():
            tok = str(uuid.uuid4())
            tokens.add(tok)
            return tok

        def send(cfg):
            self.send_message(
                SystemCommandKind.WRITE_MOTOR_CONFIGURATION, cfg,
                context=make_token(),
            )

        for conf in (
            data.x_config,
            data.y_config,
            data.z_config,
            data.load_config,
            data.cover_config,
        ):
            if not self._motor_configuration_is_required(conf[0]):
                continue
            with self.await_acknowledge(tokens, timeout=3):
                send(conf)

    def set_load_procedure(self, load_steps: MotorSteps):
        self.send_message(SystemCommandKind.SET_LOAD_PELLET_PROCEDURE, load_steps)

    def set_send_procedure(self, send_steps: MotorSteps):
        self.send_message(SystemCommandKind.SET_SEND_PELLET_PROCEDURE, send_steps)

    def set_cover_procedure(self, cover_steps: MotorSteps):
        self.send_message(SystemCommandKind.SET_COVER_PELLET_PROCEDURE, cover_steps)

    def set_release_procedure(self, release_steps: MotorSteps):
        self.send_message(SystemCommandKind.SET_RELEASE_PELLET_PROCEDURE, release_steps)

    def set_retract_procedure(self, steps: MotorSteps):
        self.send_message(SystemCommandKind.SET_MOVE_RETRACT_PROCEDURE, steps)

    def set_motor_configuration(self, config: Union[ServoConfig, StepperConfig]):
        self.send_message(SystemCommandKind.WRITE_MOTOR_CONFIGURATION, config)

    def _start(self):
        if self._current_thread is None or not self._current_thread.is_alive():
            logger.verbose("Starting new reader thread for %s", self._name)
            self._current_thread_watchdog_perf_c = time.perf_counter()  # get_perf_now()
            thread = Thread(target=self._run, name=self._name)
            thread.start()
            self._current_thread = thread

    def _run(self) -> None:
        logger.debug(f"<{self._name}> thread started")

        try:
            while True:
                if not self._run_unconnected():
                    break

                if not self._run_connected():
                    break
        except BaseException as err:
            logger.exception("<%s> CAN transport worker failed: %s", self._name, err)
            callback = self._failure_callback
            if callback is not None:
                try:
                    callback(CanFailure(
                        CanFailureKind.TRANSPORT,
                        str(err) or err.__class__.__name__,
                    ))
                except Exception:
                    logger.exception("Failed to report CAN transport failure")
        finally:
            if self._interface.is_open:
                try:
                    self._device.disconnect()
                finally:
                    self._interface.close()

        logger.debug(f"<{self._name}> thread terminated")

    def _run_unconnected(self) -> bool:
        logger.info("running unconnected")
        while True:
            self._current_thread_watchdog_perf_c = time.perf_counter()
            try:
                cmd, data, context = self._cmd_queue.get(timeout=0.25)
                self._cmd_queue.task_done()
            except Empty:
                continue

            if cmd == _REQUEST_DISCONNECT:
                logger.debug(f"<{self._name}> message: _REQUEST_DISCONNECT")
                return False
            elif cmd == _REQUEST_CONNECT:
                logger.debug(f"<{self._name}> message: _REQUEST_CONNECT")
                break
            else:
                logger.warning("<%s> message: command %s ignored", self._name, cmd)

        if not self._interface.is_open:
            try:
                success = self._interface.open()
                if success:
                    logger.debug(f"<{self._name}> interface open")
                    self._device.connect()
                    logger.debug(f"<{self._name}> device connected")
                else:
                    logger.warning(f"<{self._name}> failed to open device")
                    return False
            except Exception as err:
                logger.exception("<%s>: %s", self._name, err)
                return False
        else:
            logger.warning(f"<{self._name}>CONNECT cmd while device already open")

        return True

    def _run_connected(self) -> bool:
        logger.info("running connected")
        t_next_cmd_queue_read = time.perf_counter()
        idle_poll_period = max(0.001, self._collect_ms / 1000.0)
        while True:
            self._current_thread_watchdog_perf_c = time.perf_counter()

            # Data from the device for the device listener to process.
            read_started = time.perf_counter()
            messages = []
            if self._interface.can_read():
                messages = self._interface.read(self._read_limit, collect_ms=self._collect_ms)
                if len(messages) > 0:
                    self._device.notify_data(messages)

            # Some CAN backends return immediately when no frame is available even
            # when ``collect_ms`` is requested.  Without an explicit idle wait this
            # loop consumes an entire CPU core and can starve the Qt thread of the
            # GIL.  Preserve the requested collection cadence while still draining
            # bursts without delay.
            if not messages:
                read_elapsed = time.perf_counter() - read_started
                idle_wait = idle_poll_period - read_elapsed
                if idle_wait > 0:
                    time.sleep(idle_wait)

            perf_now = get_perf_now()
            if perf_now > t_next_cmd_queue_read:
                # Messages from the client of this class to control the device listener (or this class, such as TERMINATE).
                try:
                    cmd, data, context = self._cmd_queue.get_nowait()
                except Empty:
                    # no need check too often for request disconnect only
                    t_next_cmd_queue_read = perf_now + 0.25
                else:
                    if cmd == _REQUEST_DISCONNECT:
                        self._cmd_queue.task_done()
                        logger.debug(f"<{self._name}> message: _REQUEST_DISCONNECT")
                        break
                    else:
                        assert False,  f"should not be needed anymore but got unknown {cmd}"
                        # we should simply make the request disconnect be handled differently,
                        # and have the senders of these cmd/data/context directly put to the device
                        self._device.notify_message(cmd, data, context)
                        self._cmd_queue.task_done()

        if self._interface.is_open:
            self._device.disconnect()
            self._interface.close()

            logger.debug(f"<{self._name}> interface closed")
        else:
            logger.warning(f"<{self._name} DISCONNECT cmd while device already disconnected")

        return False
