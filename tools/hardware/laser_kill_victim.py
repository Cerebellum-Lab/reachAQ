"""The process the laser kill test kills.

It opens the laser output the way reachAQ does (a LaserModel over a
NidaqLaserController), puts the output in one of the states the test asks about
(a DC level, or a running pulse train), prints READY, and waits for a SIGUSR1
cue. On the cue it dies the way --mode says, or waits to be killed by the
orchestrator. What the output does then is read by the orchestrator through a
reader of its own; nothing here reads the output back.

Nothing in `run` or `main` closes the controller on the way out: the exception
mode exists to leave it open, as an uncaught error in the application would.
Only the `clean` mode, and a victim that was never cued, close it.

This drives the real command output. Run it only with the laser unable to
emit (tools/hardware/qualify_laser_kill.py states the preconditions, and runs
this script itself); --laser-disconnected says that is so, and is required.

If the process that started it goes away, this one closes the output and exits
rather than wait for a cue that can no longer come (see orphaned).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    ".",
    "auto-trainer-core/src",
    "auto-trainer-device/src",
    "auto-trainer-model/src",
    "auto-trainer-pyside/src",
    "auto-trainer-behavior/src",
    "auto-trainer-inference/src",
    "auto-trainer-video/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _path))

from autotrainer.core.configuration import SystemConfiguration
from autotrainer.device import (
    LaserChannelId,
    LaserPulseTrain,
    LaserSynchronizedPulseTrain,
    LaserSystemConfiguration,
    NidaqLaserController,
)
from autotrainer.device.laser import LaserPulseRefused
from tools.acquisition.model.laser_kill_matrix import MODES, TIMINGS, WAVEFORMS
from tools.acquisition.model.laser_model import LaserModel

_PULSE_HZ = 5.0
_PULSE_MS = 180.0  # 90 % duty at 5 Hz
_CANCEL_WAIT_S = 5.0
_CUE_POLL_S = 0.01

#: Set by the SIGUSR1 handler; wait_for_cue polls it.
_cued = threading.Event()

#: The pid of the process that started this one. When os.getppid() stops being
#: it, that process is gone (this one was reparented) and no cue or kill can
#: come. The orchestrator passes its own pid (--orchestrator-pid), so a parent
#: that died while this process was still importing is noticed too; a victim run
#: by hand takes its parent at the start of main. adopt_parent sets it.
_PARENT_PID = os.getppid()

_REFUSAL = (
    "refusing to drive the laser output: pass --laser-disconnected once the key "
    "is off, the command BNC and shutter cable are unplugged and the output is "
    "on the dummy load"
)


def adopt_parent(orchestrator_pid: Optional[int]) -> None:
    """Take `orchestrator_pid` as the parent to watch; None takes the current one."""
    global _PARENT_PID
    _PARENT_PID = os.getppid() if orchestrator_pid is None else orchestrator_pid


def orphaned() -> bool:
    return os.getppid() != _PARENT_PID


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hold the laser output in one state and die on a SIGUSR1 cue.")
    parser.add_argument(
        "--config", default=os.path.expanduser("~/Autotrainer/system_configuration.yaml"))
    parser.add_argument("--laser", type=int, default=1)
    parser.add_argument("--timing", choices=TIMINGS, required=True)
    parser.add_argument("--waveform", choices=WAVEFORMS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--level", type=float, default=2.0, help="command volts")
    parser.add_argument("--pulse-seconds", type=float, default=30.0)
    parser.add_argument("--ready-settle", type=float, default=0.5)
    parser.add_argument("--cue-timeout", type=float, default=60.0)
    parser.add_argument(
        "--orchestrator-pid", type=int, default=None,
        help="the pid of the process that started this one; this one closes the "
             "output and exits if that process goes away (default: its parent at start)")
    parser.add_argument(
        "--laser-disconnected", action="store_true",
        help="the laser is unable to emit (key off, command BNC and shutter cable "
             "unplugged, output on the dummy load); required")
    args = parser.parse_args(argv)
    if not args.laser_disconnected:
        parser.error(_REFUSAL)
    return args


def laser_configuration_for(
    configuration: LaserSystemConfiguration, timing: str,
) -> LaserSystemConfiguration:
    """The laser configuration a row runs under: software or hardware timed."""
    if timing == "software":
        return dataclasses.replace(configuration, hardware_timed=False, sample_rate_hz=None)
    if timing != "hardware":
        raise ValueError(f"unknown timing {timing!r}")
    if not configuration.hardware_timed or configuration.sample_rate_hz is None:
        raise ValueError(
            "hardware-timed rows need hardwareTimed: true and sampleRateHz in the configuration")
    return configuration


def pulse_train_for(channel_id, level_volts: float, seconds: float) -> LaserPulseTrain:
    """`seconds` of 5 Hz at 90 % duty, armed and returned without waiting."""
    return LaserPulseTrain(
        channel_id=channel_id,
        amplitude_volts=level_volts,
        duration_ms=_PULSE_MS,
        pulse_count=max(1, int(seconds * _PULSE_HZ)),
        frequency_hz=_PULSE_HZ,
        open_shutter=True,
        close_shutter=True,
        wait=False,
    )


def open_controller(
    configuration: SystemConfiguration, laser_configuration: LaserSystemConfiguration,
) -> Tuple[LaserModel, NidaqLaserController]:
    """The controller and the model that holds it, as the application builds them.

    `feedback_reader` is a stub, so the controller opens no 6221 input task:
    the orchestrator's reader owns those inputs.
    """
    controller = NidaqLaserController(
        laser_configuration,
        feedback_reader=lambda _channel: 0.0,
        timing_plan=None,
        analog_terminal_config=configuration.nidaq_stream.analog_terminal_config or None,
    )
    model = LaserModel()
    model.set_controller(controller)
    return model, controller


def hold(
    model: LaserModel,
    controller: NidaqLaserController,
    channel_id,
    waveform: str,
    level_volts: float,
    pulse_seconds: float,
) -> Optional[object]:
    """Put the output in the state the row asks about; the pulse's operation, if any."""
    if waveform == "dc":
        model.set_command_voltage(channel_id, level_volts)
        return None
    train = pulse_train_for(channel_id, level_volts, pulse_seconds)
    # NidaqLaserController.run_pulse_train returns None, and wraps its train in
    # a synchronized one whose per-laser trains must all wait
    # (LaserSynchronizedPulseTrain refuses one with wait=False). Only
    # run_synchronized_pulse_train hands back the operation, which the clean mode
    # cancels. So the train's own wait=False becomes the wrapper's, as
    # LaserModel.arm_manual_pulse builds it, and the train inside waits.
    return controller.run_synchronized_pulse_train(
        LaserSynchronizedPulseTrain(
            pulse_trains=(dataclasses.replace(train, wait=True),),
            trigger_source=train.trigger_source,
            trigger_edge=train.trigger_edge,
            enable_pmt_shutter=train.enable_pmt_shutter,
            wait=train.wait,
            timeout_seconds=train.timeout_seconds,
        )
    )


def ready_line(pid: int, args, applied_volts: float) -> str:
    return (f"READY pid={pid} timing={args.timing} waveform={args.waveform} "
            f"mode={args.mode} level={applied_volts:.4f}")


def _shut_down(model, operation) -> None:
    """What the code does on purpose: end the train, if any, then close the model."""
    if operation is not None:
        operation.cancel()
        operation.wait_until_finished(_CANCEL_WAIT_S)
    model.close()


def _ignore_sigterm() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def act(
    mode: str,
    *,
    model,
    operation,
    abort: Callable[[], None] = os.abort,
    pause: Callable[[float], None] = time.sleep,
    ignore_sigterm: Callable[[], None] = _ignore_sigterm,
    is_orphaned: Callable[[], bool] = orphaned,
) -> int:
    """Die, or wait to be killed, as `mode` says. Returns an exit status if it lives.

    Only `clean` touches the model or the operation: the other modes leave the
    output exactly as the process held it, with one exception: a `hang` whose
    parent has gone away closes the output and exits, because nothing is left to
    end it and it would hold the output for good.
    """
    if mode == "clean":
        _shut_down(model, operation)
        return 0
    if mode in ("sigterm", "sigkill"):
        # The orchestrator's signal ends this sleep. Getting past it means no
        # signal came.
        pause(3600)
        return 3
    if mode == "exception":
        raise RuntimeError("laser kill test: exception mode")
    if mode == "abort":
        abort()
        return 3
    if mode == "hang":
        ignore_sigterm()
        while True:
            pause(1)
            if is_orphaned():
                _shut_down(model, operation)
                return 3
    raise ValueError(f"unknown mode {mode!r}")


def _on_cue(_signum, _frame) -> None:
    _cued.set()


def wait_for_cue(timeout: float) -> bool:
    """True once SIGUSR1 has arrived; False if `timeout` seconds pass first, or at
    once if the parent process has gone away."""
    deadline = time.monotonic() + timeout
    while not _cued.is_set():
        # Gone: nothing can cue this process now, so do not wait out the timeout.
        if orphaned() or time.monotonic() >= deadline:
            return False
        time.sleep(_CUE_POLL_S)
    return True


def _announce(line: str) -> None:
    # Flushed: the orchestrator reads this through a pipe, where print is
    # block-buffered, and READY must reach it while the process lives. A pipe
    # whose reader has gone is not an error to die of: the orphan check closes
    # the output.
    try:
        print(line, flush=True)
    except OSError:
        pass


def run(
    args,
    *,
    load=SystemConfiguration.load_yaml_file,
    open_controller=open_controller,
    announce=_announce,
    wait_for_cue=wait_for_cue,
) -> int:
    # The orchestrator may have gone while this process was starting up: then
    # there is nobody to cue it or kill it, and nothing has been opened yet.
    if orphaned():
        announce("ORPHANED")
        return 3
    # Only .laser and .nidaq_stream.analog_terminal_config are read. Never saved:
    # save_backup=False, and no AppModel is built (it saves on exit).
    configuration = load(args.config, save_backup=False)
    laser_configuration = laser_configuration_for(configuration.laser, args.timing)
    channel_id = LaserChannelId(args.laser)
    model, controller = open_controller(configuration, laser_configuration)
    try:
        operation = hold(
            model, controller, channel_id, args.waveform, args.level, args.pulse_seconds)
    except LaserPulseRefused as error:
        announce(f"REFUSED {error}")
        model.close()
        return 2
    # Pulse amplitudes are refused out of range, not clamped, so for a hold that
    # got this far the clamp is the level the output was given.
    applied = laser_configuration.get_channel(channel_id).clamp_command_voltage(args.level)
    time.sleep(args.ready_settle)
    announce(ready_line(os.getpid(), args, applied))
    if not wait_for_cue(args.cue_timeout):
        abandoned = orphaned()
        _shut_down(model, operation)
        announce("ORPHANED" if abandoned else "NOT CUED")
        return 3
    return act(args.mode, model=model, operation=operation)


def main() -> int:
    # No core file from an abort: the row measures the output, not a dump.
    import resource  # POSIX only
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # A handler for the cue only. The application has none for SIGTERM or
    # SIGINT either; `hang` installs its own ignore when cued.
    signal.signal(signal.SIGUSR1, _on_cue)
    args = parse_args()
    adopt_parent(args.orchestrator_pid)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
