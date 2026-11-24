# The laser process and what it does when something dies

The NI laser no longer runs inside reachAQ's own process. When System Mode
opens the laser, or the calibration ramp opens it in Idle, reachAQ starts a
**laser process** that owns every laser NI-DAQ task: the command outputs (AO),
the shutter and auxiliary lines, and every pulse train and ramp. reachAQ sends
it commands over a pipe. Code: `reachaq.device.laser_process`.

The point is that the laser no longer depends on reachAQ living long enough to
turn it off. The laser process watches reachAQ, and turns the laser off by
itself when reachAQ goes.

## When the laser process drives the laser safe

| Event | How it is noticed | Measured (simulation and fake NI) |
| --- | --- | --- |
| reachAQ crashes or is killed (`kill -9`) | the pipe to reachAQ closes | 0.16 s, simulated rig, headless app killed |
| reachAQ hangs (alive, but frozen) | its heartbeat (every 100 ms) stops for 1 s | 0.9–1.1 s after the hang |
| the laser process hits an internal error | any uncaught exception in it | at once |
| the laser process is sent SIGTERM | signal (multiprocessing sends this as reachAQ exits) | at once |

On Linux it also arms `PR_SET_PDEATHSIG` and checks that reachAQ's process
still exists. Do not count on the parent-death signal: the laser process's parent
is multiprocessing's forkserver, which was seen to outlive a killed reachAQ.
The pipe and the pid check are what act.

**Safe** is what the controller's own close leaves (`NidaqLaserController.close`),
in this order:

1. every shutter written closed;
2. a running calibration ramp aborted;
3. each pulse train cancelled: its shutters closed, then its output aborted;
4. per laser: shutter closed, auxiliary line off, command written to
   `minimumCommandVolts` (0.0 V on christielab10);
5. tasks closed and backplane routes released.

The laser process then exits. The close is given 20 s; if the driver has not
finished by then it exits anyway and logs CRITICAL that the laser state is
unknown.

The shutter is written False. That assumes the shutter opens on a high line,
which has not been verified on the rig yet (DO polarity is still open in todo).

## When the laser process itself dies or freezes

reachAQ watches the laser process the same way: its pipe and its own heartbeat.
If the laser process dies (`kill -9`, a segfault, out of memory) or stops
answering for 1 s:

1. reachAQ kills it if it is still alive;
2. reachAQ starts a one-shot **reset process**, which opens the laser
   controller (opening drives every output safe, as it always has) and closes it;
3. the laser subsystem latches **FAILED**, and a recording session in progress
   is told the laser failed;
4. every laser command is refused until the laser is opened again, by the next
   Run or a Refresh Hardware.

Nothing restarts laser service by itself. A session does not carry on with a
laser whose state at the fault is unknown.

If the laser process went safe by itself first (reachAQ hung and then came
back, or an internal error), there is no reset. The FAILED reason says which
happened.

## What the FAILED reason says

The Hardware Status panel and the log show `laser process failed: <reason>`:

| Reason contains | Meaning | What to do |
| --- | --- | --- |
| `it drove the laser safe first` | the laser process made it safe itself | Run or Refresh Hardware to reopen |
| `the laser outputs were reset to safe` | the laser process died; the reset process made it safe | Run or Refresh Hardware to reopen; report the log |
| `make the laser safe by hand` | neither the laser process nor the reset could confirm it safe | turn the laser off at its key or driver **before anything else**, then investigate |

## What this does not cover

- **A hard death of the laser process.** Its outputs stay as the driver
  leaves them until the reset process reopens them. Whether a reopen actually
  resets a command left by a killed process is what the laser kill test
  (`src/reachaq/hardware/qualify_laser_kill.py`) measures on the rig; it has not
  been run yet.
- **reachAQ and the laser process dying together** (power loss, a kernel
  panic, both killed at once). Only a hardware interlock covers that. It is
  roadmap Phase 2, sized by the kill test.
- **A driver that hangs inside a call.** The process can go on heartbeating
  while one of its threads is stuck in DAQmx, so the hang is noticed only
  through that call's own timeout.
- A laser process killed while writing to the shared log queue can leave that
  queue's write lock held, so later log records from other processes stall.
  This is a multiprocessing limitation shared by every worker process.

## Command latency

The process boundary adds a pipe round trip. Measured in WSL against the fake
NI-DAQmx, so the driver's own time (about 0.6 ms to the edge for a start on
christielab10) is the same on both sides and is not included:

| Measure | In-process (before) p50 / p95 | Laser process p50 / p95 |
| --- | --- | --- |
| Armed pulse: `trigger()` call to DAQmx start entry, idle | 2 / 3 µs | 102 / 127 µs |
| same, with a thread spinning on reachAQ's GIL | 3 / 5 µs | 192 / 280 µs (max 1.1 ms) |
| `trigger()` call to return, idle | 39 / 64 µs | 326 / 397 µs |
| same, GIL-bound | 67 / 99 µs | 15.6 / 26.5 ms |
| `set_command_voltage`, idle | 59 / 114 µs | 409 / 745 µs |
| same, GIL-bound | 0.17 / 10.4 ms | 15.6 / 25.2 ms |

The start itself, which is what times a pulse, stays sub-millisecond even
under GIL load. Heartbeats have a pipe of their own so a request never waits
behind one. Waiting for the reply is what costs under load: with another
thread holding reachAQ's GIL, each thread hand-off waits for Python's 5 ms
switch interval. The Qt thread can be held for that long after a Run Pulse
click; the pulse has already started by then.

Regenerate the table with `pytest --run-bench tests/device/laser_process_latency_test.py`.

## `hardwareTimed`

With `hardwareTimed: false`, the NI controller refuses every pulse train and the
calibration ramp. That leaves Run Pulse, Test stim and protocol stim unusable;
only a held DC command works. The example configurations now ship
`hardwareTimed: true` with `sampleRateHz: 100000` (christielab10's values). An
NI laser loaded with `false` comes up READY with a reason that says pulses and
the ramp are refused, and the log warns. The field's default in code is still
false (Ben, 2026-11-18). Configurations saved by reachAQ always write the field
explicitly.

## Tests

- `tests/device/laser_process_test.py`: three processes (test,
  stand-in application, laser process over the fake NI-DAQmx, which logs every
  write to a file). Covers the application SIGKILLed, the application stopped
  (heartbeat loss), an exception in the laser process, the laser process
  SIGKILLed (reset from a new process), the laser process stopped (killed and
  reset), a clean close, and the simulated laser (0 V published on the
  simulation bus for the cut-short pulse).
- `tests/laser_process_model_test.py`: LaserModel's Run Pulse arm, fire and
  finish, a waited train, and a fault reaching `LaserModel.CONTROLLER_FAULT`.
- The rest of the suite opens the NI controller in-process
  (`REACHAQ_LASER_IN_PROCESS=1`, set in the root `conftest.py`), because those
  tests replace NI-DAQmx inside the test process, where a laser process would not
  see the replacement. The application never sets it.
