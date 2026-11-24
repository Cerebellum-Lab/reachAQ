"""The NI laser controller in a process of its own, which drives the laser safe when its parent goes.

The application holds a LaserProcessController; the NI tasks are opened, driven
and closed by NidaqLaserController (or the simulated controller) in a child
process, which serve() runs. The child drives every output to its safe level,
by the controller's own close(), and exits, when any of these happens:

- its pipe to the application reaches EOF: the application has died;
- on Linux, its parent dies (PR_SET_PDEATHSIG), or the application's pid is
  gone. Under forkserver the parent is the forkserver, which exits as the
  application does;
- the application's heartbeat, sent every HEARTBEAT_PERIOD_S, has not come for
  HEARTBEAT_TIMEOUT_S: the application is alive but hung;
- an exception goes uncaught in any of the child's threads;
- it is sent SIGTERM, as multiprocessing does to a daemonic child at exit.

Safe is what close() leaves (NidaqLaserController.close): every shutter
closed first, the calibration ramp and every pulse aborted, then each
channel's shutter closed, its auxiliary output off and its command at
minimumCommandVolts.

The application side watches the child too. When the child dies without
saying it went safe, or stops heartbeating, the application kills it, starts
a one-shot reset child that opens the controller (which drives every output
safe) and closes it, and reports the fault (on_fault). Nothing restarts laser
service after that until a controller is opened again (Ben, 2026-11-18).

What this cannot cover: a laser process that dies hard leaves its outputs to
the driver until the reset child reopens them, and whether that reopen resets
them is what the laser kill test measures. Both processes dying together is
left to the hardware interlock.
"""

from __future__ import annotations

import collections
import ctypes
import importlib
import importlib.util
import itertools
import logging
import os
import pickle
import queue
import signal
import sys
import threading
import time
from typing import Callable, Dict, Optional, Tuple

from reachaq.core import LaserSystemConfiguration
from reachaq.core.logging import log_hardware_initialization
from reachaq.core.multiproc import get_nidaq_mp_ctx

from .laser import (
    LaserCalibrationRamp,
    LaserFeedbackSample,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    normalize_laser_channel_id,
)
from .nidaq_laser import LaserControllerStillClosing, LaserOperationState

logger = logging.getLogger(__name__)

#: How often each side heartbeats the other, and how long either waits for
#: one before it acts (Ben, 2026-11-18: 100 ms / 1 s).
HEARTBEAT_PERIOD_S = 0.1
HEARTBEAT_TIMEOUT_S = 1.0
#: How long the child gives the controller's close() to make the laser safe
#: before it exits regardless. close() bounds its own waits: 5 s for each
#: pulse it cancels, and the ramp's release.
SAFE_CLOSE_BOUND_S = 20.0
#: How long the application waits for the child to open the controller. The
#: NI-DAQmx runtime alone takes seconds to load.
OPEN_TIMEOUT_S = 60.0
#: How long a child that has closed cleanly waits for what its close stopped
#: waiting for (work_left_running) before it exits.
WORK_LEFT_BOUND_S = 30.0
#: How long a reset child has to open and close the controller.
RESET_TIMEOUT_S = 60.0

EXIT_CLOSED = 0
EXIT_PARENT_DIED = 70
EXIT_HEARTBEAT_LOST = 71
EXIT_INTERNAL_ERROR = 72
EXIT_OPEN_FAILED = 73
EXIT_TERMINATED = 74

EXIT_REASONS = {
    EXIT_CLOSED: "closed",
    EXIT_PARENT_DIED: "the application died",
    EXIT_HEARTBEAT_LOST: "the application's heartbeat was lost",
    EXIT_INTERNAL_ERROR: "an internal error in the laser process",
    EXIT_OPEN_FAILED: "the laser controller failed to open",
    EXIT_TERMINATED: "the laser process was terminated",
}

#: The exits of a child that tripped by itself, and drove the laser safe
#: first unless it said otherwise as it exited.
_SELF_TRIPPED = frozenset({
    EXIT_PARENT_DIED, EXIT_HEARTBEAT_LOST, EXIT_INTERNAL_ERROR, EXIT_TERMINATED})

#: Set to "1" to open the NI laser controller in this process, as before the
#: laser had a process of its own. For tests that replace NI-DAQmx in this
#: process, which a child would not see; the application never sets it.
ENV_IN_PROCESS = "REACHAQ_LASER_IN_PROCESS"


def isolation_enabled() -> bool:
    return os.environ.get(ENV_IN_PROCESS) != "1"


class LaserProcessFailed(RuntimeError):
    """The laser process is gone or unresponsive; laser service is off until it is reopened."""


class LaserRemoteError(RuntimeError):
    """An error raised in the laser process whose own class could not be carried here.

    Its text is "ClassName: message", and a DAQmx status code, when the
    original had one, is kept on error_code.
    """


def _portable_error(error: BaseException) -> BaseException:
    """`error`, or a LaserRemoteError of its text where it does not survive pickling.

    nidaqmx's DaqError takes more than its message, so pickle cannot rebuild it.
    """
    try:
        clone = pickle.loads(pickle.dumps(error))
        if type(clone) is type(error) and str(clone) == str(error):
            return error
    except Exception:
        pass
    remote = LaserRemoteError(f"{type(error).__name__}: {error}")
    code = getattr(error, "error_code", None)
    if code is not None:
        remote.error_code = code
    return remote


def _remote_feedback_reader(physical_channel: str) -> float:
    """Stands in, in the child, for the application's reader of the NI stream.

    Given one, the controller opens no input tasks of its own: the stream holds
    those inputs. The application reads them itself (LaserProcessController).
    """
    raise RuntimeError(
        f"laser feedback for {physical_channel} is read from the NI-DAQ stream in "
        "the application process")


def _load_factory(spec: str) -> Callable:
    """`module:attribute`, or `path/to/file.py:attribute`."""
    target, _, attribute = spec.rpartition(":")
    if target.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location(
            "_laser_process_factory", target)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, attribute)


def _exit_description(code: Optional[int]) -> str:
    if code is None:
        return "still running"
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return EXIT_REASONS.get(code, f"exit status {code}")


def _log_config():
    try:
        from reachaq.core.logging import make_log_dict_config
        return make_log_dict_config()
    except Exception:
        return None


# --- the child ---------------------------------------------------------------

#: Requests run on the pipe's reader thread itself: the start of an armed
#: pulse is the click's or the trial's latency path.
_INLINE = frozenset({"op_trigger"})
#: Requests that can block for as long as a pulse, a ramp, a cancel or a close
#: takes; each runs on a thread of its own, so nothing queues behind it.
_BLOCKING = frozenset({
    "run_pulse_train", "run_synchronized_pulse_train", "run_calibration_ramp",
    "op_cancel", "close",
})
#: The rest run in order on one command thread.
_COMMANDS = frozenset({
    "set_command_voltage", "set_shutter_open", "set_auxiliary_output",
    "read_diode_voltage", "read_command_copy_voltage", "read_feedback_sample",
    "command_volts", "close_all_shutters", "timing_status",
})


class _LaserServer:
    def __init__(self, conn, heartbeat_conn, spec: dict):
        self._conn = conn
        #: Heartbeats both ways, on a pipe of their own: sharing the request
        #: pipe, a heartbeat's send held its lock while it waited for the GIL,
        #: and a trigger queued behind it (laser_process_latency_test).
        self._heartbeat_conn = heartbeat_conn
        self._spec = spec
        self._send_lock = threading.Lock()
        self._controller = None
        self._operations: Dict[str, object] = {}
        self._operations_lock = threading.Lock()
        self._commands: "queue.Queue" = queue.Queue()
        self._events: "queue.Queue" = queue.Queue()
        self._last_heartbeat = time.monotonic()
        #: (exit code, reason), set once by whatever ends the process first.
        self._trip = None
        self._trip_lock = threading.Lock()
        self._tripped = threading.Event()
        #: Set once a requested close has run: the laser is already safe.
        self._closed = threading.Event()
        self._parent_pid = os.getppid()
        self._application_pid = spec.get("application_pid")

    # Sending ------------------------------------------------------------------

    def _send(self, message) -> None:
        try:
            with self._send_lock:
                self._conn.send(message)
        except (EOFError, OSError):
            self.trip(EXIT_PARENT_DIED, "the pipe to the application broke")

    def _send_events(self) -> None:
        while True:
            self._send(self._events.get())

    # Ending -------------------------------------------------------------------

    def trip(self, code: int, reason: str) -> None:
        """End the process: the laser driven safe first, unless already closed."""
        with self._trip_lock:
            if self._trip is not None:
                return
            self._trip = (code, reason)
        self._tripped.set()

    def _go_safe_and_exit(self) -> None:
        code, reason = self._trip
        safe = True
        if not self._closed.is_set() and self._controller is not None:
            logger.critical(
                "Laser process: %s; driving the laser safe (shutters closed, "
                "pulses aborted, commands to their minimum)", reason)
            done = threading.Event()
            errors = []

            def close():
                try:
                    self._controller.close()
                except Exception as error:
                    errors.append(error)
                    logger.exception("Laser process: the safe close failed")
                finally:
                    done.set()

            threading.Thread(target=close, name="LaserSafeClose", daemon=True).start()
            if not done.wait(SAFE_CLOSE_BOUND_S):
                safe = False
                logger.critical(
                    "Laser process: the safe close had not ended %.0f s later; "
                    "exiting with the laser in an unknown state", SAFE_CLOSE_BOUND_S)
            elif errors:
                safe = False
                logger.critical(
                    "Laser process: the safe close failed (%s); the laser may "
                    "still be driven", errors[0])
            else:
                logger.critical("Laser process: the laser is safe; exiting")
        self._exit(code, reason, safe)

    def _exit(self, code: int, reason: str, safe: bool) -> None:
        try:
            with self._send_lock:
                self._conn.send(("exiting", code, reason, safe))
        except Exception:
            pass
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass
        # A queue handler's feeder thread sends what it holds.
        time.sleep(0.05)
        os._exit(code)

    # Watching the parent ------------------------------------------------------

    def _arm_signals(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(
            signal.SIGTERM,
            lambda *_: self.trip(EXIT_TERMINATED, "the laser process was sent SIGTERM"))
        if not sys.platform.startswith("linux"):
            return
        signal.signal(
            signal.SIGUSR1,
            lambda *_: self.trip(EXIT_PARENT_DIED, "the parent process died (PDEATHSIG)"))
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            pr_set_pdeathsig = 1
            if libc.prctl(pr_set_pdeathsig, int(signal.SIGUSR1), 0, 0, 0) != 0:
                logger.warning("Laser process: prctl(PR_SET_PDEATHSIG) failed, errno %s",
                               ctypes.get_errno())
        except Exception:
            logger.warning("Laser process: PR_SET_PDEATHSIG is unavailable", exc_info=True)
        # A parent gone before the signal was armed sends none.
        if os.getppid() != self._parent_pid:
            self.trip(EXIT_PARENT_DIED, "the parent process died as the laser process started")

    def _parent_gone(self) -> Optional[str]:
        if os.getppid() != self._parent_pid:
            return "the parent process died"
        if self._application_pid and sys.platform.startswith("linux"):
            # Linux only: on Windows os.kill(pid, 0) terminates the process.
            try:
                os.kill(self._application_pid, 0)
            except ProcessLookupError:
                return "the application process is gone"
            except PermissionError:
                pass
        return None

    def _watch(self) -> None:
        """The main thread, once open: heartbeats, the deadline, the parent, the end."""
        next_heartbeat = 0.0
        while not self._tripped.wait(0.02):
            now = time.monotonic()
            if now >= next_heartbeat:
                try:
                    self._heartbeat_conn.send(("hb",))
                except (EOFError, OSError):
                    self.trip(EXIT_PARENT_DIED, "the heartbeat pipe to the application broke")
                next_heartbeat = now + HEARTBEAT_PERIOD_S
            gone = self._parent_gone()
            if gone:
                self.trip(EXIT_PARENT_DIED, gone)
            elif now - self._last_heartbeat > HEARTBEAT_TIMEOUT_S:
                self.trip(EXIT_HEARTBEAT_LOST,
                          f"no heartbeat from the application for "
                          f"{now - self._last_heartbeat:.2f} s")
        self._go_safe_and_exit()

    # Requests -----------------------------------------------------------------

    def _read(self) -> None:
        try:
            while True:
                try:
                    message = self._conn.recv()
                except (EOFError, OSError):
                    self.trip(EXIT_PARENT_DIED, "the pipe to the application closed")
                    return
                kind = message[0]
                if kind == "call":
                    self._dispatch(*message[1:])
                else:
                    raise RuntimeError(f"unknown message {kind!r}")
        except Exception as error:
            logger.exception("Laser process: the request reader failed")
            self.trip(EXIT_INTERNAL_ERROR, f"the request reader failed: {error}")

    def _read_heartbeats(self) -> None:
        while True:
            try:
                self._heartbeat_conn.recv()
            except (EOFError, OSError):
                self.trip(EXIT_PARENT_DIED, "the heartbeat pipe to the application closed")
                return
            self._last_heartbeat = time.monotonic()

    def _dispatch(self, request_id, method, args, kwargs) -> None:
        if method in _INLINE:
            self._handle(request_id, method, args, kwargs)
        elif method in _BLOCKING:
            threading.Thread(
                target=self._handle, args=(request_id, method, args, kwargs),
                name=f"Laser-{method}", daemon=True).start()
        elif method in _COMMANDS:
            self._commands.put((request_id, method, args, kwargs))
        else:
            self._send(("reply", request_id, False,
                        RuntimeError(f"the laser process has no request {method!r}")))

    def _run_commands(self) -> None:
        while True:
            self._handle(*self._commands.get())

    def _handle(self, request_id, method, args, kwargs) -> None:
        try:
            handler = getattr(self, f"_do_{method}", None)
            if handler is not None:
                result = handler(*args, **kwargs)
            else:
                result = getattr(self._controller, method)(*args, **kwargs)
        except Exception as error:
            self._send(("reply", request_id, False, _portable_error(error)))
        else:
            self._send(("reply", request_id, True, result))
        if method == "close":
            self._after_close()

    def _do_timing_status(self):
        return dict(getattr(self._controller, "timing_status", None) or {})

    def _do_command_volts(self, channel_id):
        # Not read_feedback_sample(), which reads inputs the application reads
        # itself from the NI stream.
        return self._controller._command_volts[normalize_laser_channel_id(channel_id)]

    def _do_run_synchronized_pulse_train(self, pulse_train):
        operation = self._controller.run_synchronized_pulse_train(pulse_train)
        if operation is None:
            return None
        self._track(operation)
        return self._snapshot(operation)

    def _do_op_trigger(self, operation_id):
        with self._operations_lock:
            operation = self._operations.get(operation_id)
        if operation is None:
            raise RuntimeError("Laser operation must be armed, found it ended")
        return operation.trigger()

    def _do_op_cancel(self, operation_id):
        with self._operations_lock:
            operation = self._operations.get(operation_id)
        return False if operation is None else operation.cancel()

    def _do_close(self):
        try:
            self._controller.close()
        finally:
            self._closed.set()
        return tuple(self._work_left())

    def _work_left(self):
        work_left_running = getattr(self._controller, "work_left_running", None)
        return () if work_left_running is None else work_left_running()

    def _after_close(self) -> None:
        """A requested close has run, whatever it raised: exit once its work has ended."""
        self._closed.set()
        wait = getattr(self._controller, "wait_for_work_left_running", None)
        if wait is not None and self._work_left():
            wait(WORK_LEFT_BOUND_S)
        self.trip(EXIT_CLOSED, "closed")

    # Operations ---------------------------------------------------------------

    def _track(self, operation) -> None:
        with self._operations_lock:
            self._operations[operation.operation_id] = operation
        transition = operation._transition_locked

        def transition_and_tell(state, detail=""):
            # Under the operation's lock: only queued here, sent by
            # _send_events, so a full pipe never holds a cancel at the lock.
            transition(state, detail)
            self._events.put(("op", self._snapshot(operation)))

        operation._transition_locked = transition_and_tell
        operation.add_terminal_callback(self._operation_ended)

    def _operation_ended(self, operation) -> None:
        with self._operations_lock:
            self._operations.pop(operation.operation_id, None)
        self._events.put(("op", self._snapshot(operation, done=True)))

    @staticmethod
    def _snapshot(operation, done: bool = False) -> dict:
        record = operation.to_record()
        error = operation.error
        return {
            "operation_id": record["operation_id"],
            "state": record["state"],
            "resources": tuple(record["resources"]),
            "context": record["context"],
            "observations": tuple(
                (item["state"], item["perf_time"], item["detail"])
                for item in record["observations"]),
            "timing_status": record["timing_status"],
            "error": None if error is None else _portable_error(error),
            "done": done or operation._done.is_set(),
        }

    # Running ------------------------------------------------------------------

    def _open(self):
        spec = self._spec
        configuration = spec["configuration"]
        factory = spec.get("factory")
        if factory:
            return _load_factory(factory)(configuration, **(spec.get("factory_kwargs") or {}))
        from .laser import make_laser_controller
        return make_laser_controller(
            configuration,
            feedback_reader=(_remote_feedback_reader if spec.get("feedback_reader") else None),
            timing_plan=spec.get("timing_plan"),
            analog_terminal_config=spec.get("analog_terminal_config"),
            isolate=False,
        )

    def run(self) -> None:
        self._arm_signals()
        threading.excepthook = self._thread_died
        threading.Thread(target=self._send_events, name="LaserEvents", daemon=True).start()
        # The pipe is read from the start, so the application's EOF trips the
        # process even while the controller opens, which takes seconds; the
        # trip is acted on once it has opened (_watch).
        threading.Thread(target=self._read, name="LaserRequests", daemon=True).start()
        threading.Thread(target=self._read_heartbeats, name="LaserHeartbeats",
                         daemon=True).start()
        try:
            self._controller = self._open()
        except Exception as error:
            # LaserControllerStillClosing: the open's own close goes on, and
            # the process stays until it ends.
            still_closing = getattr(error, "still_closing", None)
            cause = getattr(error, "error", error)
            self._send(("open_failed", _portable_error(cause), still_closing is not None))
            if still_closing is not None:
                still_closing.wait()
            self._exit(EXIT_OPEN_FAILED, "the laser controller failed to open", True)
        if self._spec.get("mode") == "reset":
            try:
                self._controller.close()
            except Exception as error:
                logger.critical("Laser reset process: the close failed: %s", error)
                self._exit(EXIT_INTERNAL_ERROR, f"the reset's close failed: {error}", False)
            self._exit(EXIT_CLOSED, "reset", True)
        self._send(("ready", {"timing_status": self._do_timing_status()}))
        threading.Thread(target=self._run_commands, name="LaserCommands", daemon=True).start()
        self._watch()

    def _thread_died(self, args) -> None:
        name = getattr(args.thread, "name", "?")
        logger.critical("Laser process: thread %s died", name,
                        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        self.trip(EXIT_INTERNAL_ERROR,
                  f"{args.exc_type.__name__} in thread {name}: {args.exc_value}")


def serve(conn, heartbeat_conn, spec: dict) -> None:
    """The laser process's entry point."""
    # The application's REACHAQ_* settings, simulation among them: a
    # forkserver child inherits the forkserver's environment, not the
    # application's.
    os.environ.update(spec.get("environment") or {})
    log_config = spec.get("log_config")
    if log_config is not None:
        import logging.config
        logging.config.dictConfig(log_config)
    server = _LaserServer(conn, heartbeat_conn, spec)
    try:
        server.run()
    except BaseException as error:
        logger.critical("Laser process: the server failed", exc_info=True)
        server.trip(EXIT_INTERNAL_ERROR, f"the server failed: {error}")
        server._go_safe_and_exit()


# --- the application side -----------------------------------------------------

class LaserProcessOperation:
    """The application's view of an operation the laser process runs.

    Its state, observations, error and timing status follow the process's
    reports; cancel() and trigger() are requests to it.
    """

    TERMINAL = {
        LaserOperationState.COMPLETED,
        LaserOperationState.FAILED,
        LaserOperationState.CANCELLED,
    }

    def __init__(self, controller: "LaserProcessController", snapshot: dict):
        self._controller = controller
        self.operation_id = snapshot["operation_id"]
        self.resources = tuple(snapshot["resources"])
        self.context = dict(snapshot["context"])
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._armed = threading.Event()
        self._callbacks = []
        self._state = LaserOperationState.PREPARED
        self._observations = ()
        self._error = None
        self._timing_status = {}
        self._update(snapshot)

    def _update(self, snapshot: dict) -> None:
        callbacks = ()
        with self._lock:
            if self._done.is_set():
                return
            # Reports are made on several threads; the longer history is the
            # later one.
            if snapshot["done"] or len(snapshot["observations"]) >= len(self._observations):
                self._state = LaserOperationState(snapshot["state"])
                self._observations = tuple(snapshot["observations"])
                if snapshot["timing_status"]:
                    self._timing_status = dict(snapshot["timing_status"])
            if snapshot["error"] is not None:
                self._error = snapshot["error"]
            if self._state is not LaserOperationState.PREPARED:
                self._armed.set()
            if snapshot["done"]:
                self._done.set()
                self._armed.set()
                callbacks = tuple(self._callbacks)
                self._callbacks.clear()
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                logger.exception("Laser terminal callback failed")

    def _lost(self, error: BaseException) -> None:
        """The laser process is gone: this operation ended with it."""
        with self._lock:
            observations = self._observations + (
                (LaserOperationState.FAILED.value, time.perf_counter(), str(error)),)
        self._update({
            "state": LaserOperationState.FAILED.value, "observations": observations,
            "timing_status": {}, "error": error, "done": True,
        })

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def observations(self):
        with self._lock:
            return self._observations

    def wait_until_finished(self, timeout=None) -> bool:
        return self._done.wait(timeout)

    def wait(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError(f"Laser operation {self.operation_id} did not finish")
        if self.error is not None:
            raise self.error
        return self.state

    def wait_until_armed(self, timeout=None):
        if not self._armed.wait(timeout):
            raise TimeoutError(f"Laser operation {self.operation_id} was not armed")
        if self.error is not None:
            raise self.error
        if self.state in {LaserOperationState.CANCELLED, LaserOperationState.FAILED}:
            raise RuntimeError(
                f"Laser operation reached {self.state.value} before it was armed")
        return self.state

    def cancel(self) -> bool:
        if self._done.is_set():
            return False
        return bool(self._controller._call("op_cancel", self.operation_id))

    def trigger(self) -> float:
        """Start the armed output; when the start was entered, in the laser process."""
        return self._controller._call("op_trigger", self.operation_id)

    def to_record(self) -> dict:
        with self._lock:
            return {
                "operation_id": self.operation_id,
                "state": self._state.value,
                "resources": list(self.resources),
                "context": dict(self.context),
                "observations": [
                    {"state": state, "perf_time": perf, "detail": detail}
                    for state, perf, detail in self._observations
                ],
                "error": (None if self._error is None
                          else f"{type(self._error).__name__}: {self._error}"),
                "timing_status": dict(self._timing_status),
            }

    def add_terminal_callback(self, callback) -> None:
        with self._lock:
            if not self._done.is_set():
                self._callbacks.append(callback)
                return
        try:
            callback(self)
        except Exception:
            logger.exception("Laser terminal callback failed")


class LaserProcessController:
    """LaserControllerProtocol, with the NI tasks in a laser process (module docstring)."""

    #: How many ended operations are kept, for a report that comes after its end.
    _FINISHED_KEPT = 256

    def __init__(
        self,
        configuration: LaserSystemConfiguration,
        *,
        feedback_reader: Optional[Callable[[str], float]] = None,
        timing_plan=None,
        analog_terminal_config: Optional[str] = None,
        factory: Optional[str] = None,
        factory_kwargs: Optional[dict] = None,
        on_fault: Optional[Callable[[str], None]] = None,
        mp_ctx=None,
        open_timeout: float = OPEN_TIMEOUT_S,
    ):
        self._configuration = configuration
        self._feedback_reader = feedback_reader
        #: Told, once, why the laser process failed; set before the process
        #: starts or with set_fault_callback.
        self._on_fault = on_fault
        self._mp_ctx = get_nidaq_mp_ctx() if mp_ctx is None else mp_ctx
        self._spec = {
            "configuration": configuration,
            "feedback_reader": feedback_reader is not None,
            "timing_plan": timing_plan,
            "analog_terminal_config": analog_terminal_config,
            "factory": factory,
            "factory_kwargs": dict(factory_kwargs or {}),
            "application_pid": os.getpid(),
            "environment": {
                key: value for key, value in os.environ.items()
                if key.startswith("REACHAQ_") and key != ENV_IN_PROCESS
            },
            "log_config": _log_config(),
        }
        self._send_lock = threading.Lock()
        self._pending: Dict[int, list] = {}
        self._pending_lock = threading.Lock()
        self._request_ids = itertools.count(1)
        self._operations: Dict[str, LaserProcessOperation] = {}
        self._finished: "collections.OrderedDict[str, LaserProcessOperation]" = (
            collections.OrderedDict())
        self._operations_lock = threading.Lock()
        self._events: "queue.Queue" = queue.Queue()
        #: Set as close() begins, as NidaqLaserController marks itself
        #: (LaserModel reads it, _controller_closed).
        self._closed = False
        self._close_requested = threading.Event()
        self._ended = threading.Event()
        #: What the child said as it exited: (code, reason, safe).
        self._exit_report = None
        self._exit_reported = threading.Event()
        self._fault: Optional[str] = None
        self._fault_lock = threading.Lock()
        self._timing_status = {"status": "independent",
                               "reason": "No finite laser waveform has been executed"}
        self._work_left: Tuple[str, ...] = ()
        self._last_child_heartbeat = time.monotonic()
        self._ready = threading.Event()
        self._opened = False
        self._open_error = None
        self._open_still_closing = False
        self._process, self._conn, self._heartbeat_conn = self._start_child(self._spec)
        #: Set once the child has been joined, by one thread alone: two
        #: threads joining a forkserver child at once race for its exit
        #: status, and the loser reads EOF, which multiprocessing reports as
        #: exit status 255.
        self._exited = threading.Event()
        threading.Thread(target=self._join_child, name="LaserProcessExit", daemon=True).start()
        for target, name in ((self._read, "LaserProcessReader"),
                             (self._dispatch_events, "LaserProcessEvents"),
                             (self._heartbeat, "LaserProcessHeartbeat")):
            threading.Thread(target=target, name=name, daemon=True).start()
        self._wait_for_open(open_timeout)

    def _join_child(self) -> None:
        self._process.join()
        self._exited.set()

    def _exit_code(self) -> Optional[int]:
        return self._process.exitcode if self._exited.is_set() else None

    def set_fault_callback(self, on_fault: Optional[Callable[[str], None]]) -> None:
        self._on_fault = on_fault

    def _start_child(self, spec: dict):
        parent_conn, child_conn = self._mp_ctx.Pipe(duplex=True)
        parent_heartbeat, child_heartbeat = self._mp_ctx.Pipe(duplex=True)
        # Daemonic: multiprocessing sends it SIGTERM as this process exits,
        # which it takes as a trip.
        process = self._mp_ctx.Process(
            target=serve, args=(child_conn, child_heartbeat, spec),
            name="LaserProcess", daemon=True)
        process.start()
        # Only the child holds its ends, so the child's death is these ends' EOF.
        child_conn.close()
        child_heartbeat.close()
        return process, parent_conn, parent_heartbeat

    def _wait_for_open(self, timeout: float) -> None:
        if not self._ready.wait(timeout):
            self._kill_child()
            raise LaserProcessFailed(
                f"the laser process did not open the laser controller within {timeout:.0f} s")
        if self._open_error is not None:
            error = self._open_error
            if self._open_still_closing:
                # The child closes what it had opened, and exits when done.
                still_closing = threading.Event()

                def watch():
                    self._exited.wait()
                    still_closing.set()

                threading.Thread(target=watch, name="LaserProcessFailedOpen",
                                 daemon=True).start()
                raise LaserControllerStillClosing(error, still_closing) from error
            self._exited.wait(5.0)
            raise error
        if not self._opened:
            self._exited.wait(5.0)
            raise LaserProcessFailed(
                "the laser process exited as it opened: "
                + _exit_description(self._exit_code()))
        log_hardware_initialization(
            logger, "READY | laser process | pid=%s application_pid=%s", self._process.pid,
            os.getpid())

    # The pipe -------------------------------------------------------------------

    def _read(self) -> None:
        while True:
            try:
                message = self._conn.recv()
            except (EOFError, OSError):
                self._child_gone()
                return
            kind = message[0]
            if kind == "reply":
                _, request_id, ok, value = message
                with self._pending_lock:
                    slot = self._pending.pop(request_id, None)
                if slot is not None:
                    slot[1], slot[2] = ok, value
                    slot[0].set()
            elif kind == "op":
                self._events.put(message[1])
            elif kind == "exiting":
                self._exit_report = tuple(message[1:])
                self._exit_reported.set()
            elif kind == "ready":
                self._timing_status = dict(message[1]["timing_status"])
                self._last_child_heartbeat = time.monotonic()
                self._opened = True
                self._ready.set()
            elif kind == "open_failed":
                self._open_error, self._open_still_closing = message[1], message[2]
                self._ready.set()

    def _dispatch_events(self) -> None:
        """Operation reports and their terminal callbacks, off the reader thread.

        A callback that makes a request would otherwise wait for the reply its
        own thread has to read.
        """
        while True:
            snapshot = self._events.get()
            if snapshot is None:
                return
            self._operation(snapshot)
            if snapshot["timing_status"]:
                self._timing_status = dict(snapshot["timing_status"])

    def _operation(self, snapshot: dict) -> LaserProcessOperation:
        """The operation a report or a reply is of, made at the first; kept while live."""
        operation_id = snapshot["operation_id"]
        with self._operations_lock:
            operation = (self._operations.get(operation_id)
                         or self._finished.get(operation_id))
            made = operation is None
            if made:
                operation = LaserProcessOperation(self, snapshot)
                self._operations[operation_id] = operation
        if not made:
            operation._update(snapshot)
        if operation._done.is_set():
            with self._operations_lock:
                if self._operations.pop(operation_id, None) is not None:
                    self._finished[operation_id] = operation
                    while len(self._finished) > self._FINISHED_KEPT:
                        self._finished.popitem(last=False)
        return operation

    def _heartbeat(self) -> None:
        """Sends this process's heartbeat, and reads the child's, on their own pipe."""
        while not self._ended.wait(HEARTBEAT_PERIOD_S):
            try:
                self._heartbeat_conn.send(("hb",))
                while self._heartbeat_conn.poll():
                    self._heartbeat_conn.recv()
                    self._last_child_heartbeat = time.monotonic()
            except (OSError, EOFError, ValueError):
                return
            silent = time.monotonic() - self._last_child_heartbeat
            if (self._opened and not self._close_requested.is_set()
                    and silent > HEARTBEAT_TIMEOUT_S):
                self._fault_detected(
                    f"the laser process stopped answering for {silent:.2f} s")
                return

    def _call(self, method: str, *args, **kwargs):
        if self._fault is not None:
            raise LaserProcessFailed(f"Laser process failed: {self._fault}")
        if self._ended.is_set():
            raise LaserProcessFailed("the laser process has closed")
        slot = [threading.Event(), None, None]
        request_id = next(self._request_ids)
        with self._pending_lock:
            self._pending[request_id] = slot
        try:
            with self._send_lock:
                self._conn.send(("call", request_id, method, args, kwargs))
        except (OSError, EOFError, ValueError) as error:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise LaserProcessFailed(f"the laser process is gone ({error})") from error
        slot[0].wait()
        if not slot[1]:
            raise slot[2]
        return slot[2]

    # The child's end ----------------------------------------------------------------

    def _child_gone(self) -> None:
        self._exited.wait(2.0)
        code = self._exit_code()
        if not self._opened or (self._close_requested.is_set() and code == EXIT_CLOSED):
            self._end(LaserProcessFailed("the laser process has closed"))
            return
        self._fault_detected(
            f"the laser process exited: {_exit_description(code)}", child_exited=True)

    def _fault_detected(self, reason: str, *, child_exited: bool = False) -> None:
        with self._fault_lock:
            if self._fault is not None:
                return
            self._fault = reason
        logger.critical("Laser process fault: %s", reason)
        if not child_exited:
            self._kill_child()
        # Every caller and operation is told at once; the reset follows.
        self._end(LaserProcessFailed(f"Laser process failed: {reason}"))
        self._exited.wait(2.0)
        code = self._exit_code()
        self._exit_reported.wait(0.5)
        report = self._exit_report
        went_safe = code in _SELF_TRIPPED and bool(report and report[2])
        if went_safe:
            self._fault = f"{reason}; it drove the laser safe first ({report[1]})"
        else:
            self._fault = f"{reason}; {self._reset_outputs()}"
        logger.critical("Laser process fault: %s", self._fault)
        on_fault = self._on_fault
        if on_fault is not None:
            try:
                on_fault(self._fault)
            except Exception:
                logger.exception("Laser fault callback failed")

    def _reset_outputs(self) -> str:
        """Open and close the controller in a one-shot child: every output to its safe level."""
        logger.critical("Laser process fault: resetting the laser outputs from a new process")
        try:
            process, conn, heartbeat_conn = self._start_child(dict(self._spec, mode="reset"))
        except Exception as error:
            logger.exception("Laser reset process did not start")
            return f"the laser outputs could not be reset ({error}); make the laser safe by hand"
        problem = None
        safe = False
        deadline = time.monotonic() + RESET_TIMEOUT_S
        try:
            while time.monotonic() < deadline:
                if not conn.poll(0.1):
                    if not process.is_alive():
                        break
                    continue
                try:
                    message = conn.recv()
                except (EOFError, OSError):
                    break
                if message[0] == "open_failed":
                    problem = f"the reset could not open the laser controller: {message[1]}"
                elif message[0] == "exiting":
                    safe = bool(message[3]) and message[1] == EXIT_CLOSED
                    if not safe and problem is None:
                        problem = f"the reset failed: {message[2]}"
                    break
        finally:
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join(5.0)
                problem = problem or "the reset did not finish"
            conn.close()
            heartbeat_conn.close()
        if safe:
            return "the laser outputs were reset to safe"
        problem = problem or f"the reset exited: {_exit_description(process.exitcode)}"
        return f"{problem}; make the laser safe by hand"

    def _kill_child(self) -> None:
        if not self._exited.is_set():
            self._process.kill()
            self._exited.wait(5.0)

    def _end(self, error: BaseException) -> None:
        if self._ended.is_set():
            return
        self._ended.set()
        self._ready.set()
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot[1], slot[2] = False, error
            slot[0].set()
        with self._operations_lock:
            operations = list(self._operations.values())
        for operation in operations:
            operation._lost(error)
        self._events.put(None)

    # LaserControllerProtocol ----------------------------------------------------------

    @property
    def configuration(self) -> LaserSystemConfiguration:
        return self._configuration

    @property
    def timing_status(self) -> dict:
        return dict(self._timing_status)

    @property
    def fault(self) -> Optional[str]:
        """Why the laser process failed, or None while it serves."""
        return self._fault

    @property
    def process_pid(self) -> Optional[int]:
        return self._process.pid

    def set_command_voltage(self, channel_id, volts: float) -> float:
        return self._call("set_command_voltage", channel_id, volts)

    def set_shutter_open(self, channel_id, is_open: bool) -> None:
        self._call("set_shutter_open", channel_id, is_open)

    def set_auxiliary_output(self, channel_id, enabled: bool) -> None:
        self._call("set_auxiliary_output", channel_id, enabled)

    def read_diode_voltage(self, channel_id) -> float:
        if self._feedback_reader is None:
            return self._call("read_diode_voltage", channel_id)
        channel = self._configuration.get_channel(channel_id)
        return float(self._feedback_reader(channel.diode_input))

    def read_command_copy_voltage(self, channel_id) -> float:
        if self._feedback_reader is None:
            return self._call("read_command_copy_voltage", channel_id)
        channel = self._configuration.get_channel(channel_id)
        if channel.command_copy_input is None:
            raise RuntimeError(
                f"laser channel {channel.channel_id.value} has no AI command-copy input "
                "configured")
        return float(self._feedback_reader(channel.command_copy_input))

    def read_feedback_sample(self, channel_id) -> LaserFeedbackSample:
        if self._feedback_reader is None:
            return self._call("read_feedback_sample", channel_id)
        channel = self._configuration.get_channel(channel_id)
        return LaserFeedbackSample(
            channel_id=channel.channel_id,
            command_volts=self._call("command_volts", channel.channel_id),
            diode_volts=self.read_diode_voltage(channel.channel_id),
            command_copy_volts=(
                None if channel.command_copy_input is None
                else float(self._feedback_reader(channel.command_copy_input))),
        )

    def run_pulse_train(self, pulse_train: LaserPulseTrain) -> None:
        self._call("run_pulse_train", pulse_train)

    def run_synchronized_pulse_train(self, pulse_train: LaserSynchronizedPulseTrain):
        snapshot = self._call("run_synchronized_pulse_train", pulse_train)
        if snapshot is None:
            return None
        if snapshot["timing_status"]:
            self._timing_status = dict(snapshot["timing_status"])
        return self._operation(snapshot)

    def run_calibration_ramp(self, ramp: LaserCalibrationRamp):
        return self._call("run_calibration_ramp", ramp)

    def close_all_shutters(self) -> None:
        self._call("close_all_shutters")

    def close(self) -> None:
        """Close the controller in the laser process, which then exits.

        Raises what the controller's close raised. What it stopped waiting for
        keeps the process alive until it ends (work_left_running).
        """
        self._closed = True
        if self._ended.is_set():
            return
        self._close_requested.set()
        try:
            self._work_left = tuple(self._call("close") or ())
        except LaserProcessFailed:
            return
        finally:
            if not self._work_left:
                self._exited.wait(5.0)

    def work_left_running(self) -> Tuple[str, ...]:
        if self._work_left and not self._exited.is_set():
            return self._work_left
        return ()

    def wait_for_work_left_running(self, timeout: Optional[float] = None) -> bool:
        if self._work_left:
            self._exited.wait(timeout)
        return not self.work_left_running()
