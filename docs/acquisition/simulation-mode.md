# Simulation mode

`--simulate` runs the whole acquisition app with no hardware attached: three
cameras, the NI-DAQ input stream, the CAN pellet board and the NI laser are
replaced by simulated backends ([backend interfaces](backend-interfaces.md)).
A session records, runs a reach-training protocol with tones, takes laser
commands and live pose, finalises, and passes `reachaq-validate-session` on its
merits. It is for development, demos, CI and onboarding. It is not data:
every simulated session says so in its metadata and in validation.

## Starting it

```bash
python -m reachaq.app --simulate                          # GUI, starts idle as usual
python -m reachaq.app --simulate --start-mode running     # GUI, System Mode running at once
python -m reachaq.app.headless --simulate           # headless, starts running
```

The 3D calibration is copied from files stored in Git LFS. In a checkout without
them, run `git lfs checkout` first; the simulation refuses to start from pointer files.

| Flag | Effect |
|---|---|
| `--simulate` | Use the simulated backends, with configuration, preferences and data in the sim home. |
| `--sim-home PATH` | The sim home. Default `~/.reachaq-sim`. |
| `--sim-reset` | Regenerate the home's configuration, preferences, motor files and calibration. Without it they are generated on first run only, so edits made in the app persist. |

`-c` and `--preferences-file` still work; given, they replace the home's files.
A simulation never reads or writes `~/Autotrainer`, the operator's preferences
or the rig's instance lock. It takes its own lock, `<home>/app.lock`, so it can
run beside the rig's instance, and a second simulation on the same home exits
with status 3.

The sim home:

```
system_configuration.yaml   generated on first run; protocols, stimulus profiles and the wiring record sit beside it
preferences.ini             every location in it points into the home
motor_config.yaml           copied from the repository's tests/
move_config.yaml            the same
4mm_6r_8c_4x/               the 3D calibration, copied from the inference package's test fixture
Data/                       session data and logs (persistence.outputLocation)
animals/                    the animal registry
runs/<utc>/signals.jsonl    this run's signal bus
app.lock                    the simulation's instance lock
```

`signals.jsonl` is the record of every simulated edge: one JSON line per tone,
STIM pulse and laser window, in host `perf_counter` seconds. It is kept for
debugging beside the run, not in the session.

Everything is set up in `main()` before `AppModel` exists, because the NI-DAQ
worker runs in a forkserver child that keeps the environment it started with.
The variables are `REACHAQ_SIMULATION=1`, `REACHAQ_SIM_BUS`, `REACHAQ_SIM_EPOCH`,
`REACHAQ_SIM_CAMERA_FPS`, `REACHAQ_SIM_CAMERA_EXPOSURE_S`,
`REACHAQ_SIM_LINE_ALIASES`, `REACHAQ_DIAMOND_TRIANGLE_CONFIG` and
`REACHAQ_CAN_TRANSPORT=emulation`. Use the
flag rather than setting them by hand: by hand, the configuration, preferences,
data and lock are not moved into a sim home.

The local API listens on its own OS-picked port, so a simulation runs beside
the rig's instance without taking its API, and its discovery record says
`"simulated": true`. Connect with `ApiClient(simulated=True)`
([local-api.md](local-api.md#finding-an-instance)).

## What each simulated backend produces

All of them share one time grid: simulated exposure k starts at
`epoch + k / fps` on `perf_counter`, which every process on the host reads
alike, with the epoch and the primary camera's fps passed in the environment.
Every simulated camera therefore has to run at the primary camera's integer fps:
the NI `cam_frames` grid takes the primary's fps from `REACHAQ_SIM_CAMERA_FPS`,
while each camera reads its own fps from its URL.

**Cameras** (`SimulatedCam`, `sim://`). Left (primary), Right and Camera3
(stimCam), 640x480 at 150 fps, 2 ms exposure. Each frame is a gradient with a
bright disc that circles once every two seconds. The stamp is the end of the
exposure in perf ns, arrival is 1 ms later, and every camera gives the same
exposure the same frame id. A late capture thread catches up from a 10-frame
buffer, as Spinnaker's default stream buffer does; later than that, frames are
lost and the id gap is left. `latch_clock()` is a perf read, so the session's
camera-latch pairing runs.

**NI-DAQ** (`SimulatedNidaqStream`, device `SimDAQ1`). 10 kHz in 500-sample
chunks, stamped before each read as the real stream is. `cam_frames` is a
square wave that changes level at each exposure start, as the rig camera's
output does. `tone1` and `tone2` are high while the emulated board plays a 5000
or 6000 Hz tone. `laser1_command_copy` and `laser1_diode` carry the laser's
commanded volts inside its pulse windows, 0 V outside, plus 0.5 mV of noise.
Discovery lists `SimDAQ1` with product type `Simulated`. The configuration has
a delivery calibration with `measured: "simulation"` (half a sample offset,
0.3 sample spread, 3 sessions), so `alignment.frames` can reach
`hardware_edge`.

**CAN pellet board** (`EmulationInterface`, firmware `pellet_emulator` 0.1.0).
Motors, servos and acknowledgements as the emulator has always done. In a
simulation it also publishes each tone on `tone1`/`tone2` from 1 ms after the
command for its duration, each STIM2/STIM3 pulse on `board_stim2`/`board_stim3`,
reports its stimulus inputs idle, and stamps every message with its receipt
time.

**Laser** (`SimulatedLaserController`). The configuration's laser is
`backend: nidaq` on `SimDAQ1` (AO `ao0`, diode `ai0`, command copy `ai1`), so
the NI laser's timing-plan path runs. Each pulse train's on-windows are
published, 1 ms ahead of its start, on the command copy and diode lines. Its
timing status is `simulated`.

**Live pose.** The model location is empty, which selects `MemoryPoseModel`.
In a simulation that model runs on the CPU without the GPU runtime check. With
a model configured, the check runs as on a rig.

**Watchdog.** The generated configuration allows 20 s, not the rig's 5 s: the
pose processes import TensorFlow and torch as they spawn, which took 5.1-5.7 s
from WSL's `/mnt/c`.

## Validation and provenance

The session metadata carries
`simulation: {"enabled": true, "bus": ..., "backends": {"camera": "sim",
"nidaq": "simulated", "laser": "simulated", "can": "emulation"}}`, and the
`session.provenance` rule warns "simulated session: no hardware was recorded".
Every other rule is judged on its merits. A simulated session's expected result
with `--full` is that warning and no other warning or failure.

Because of that warning, `reachaq-validate-session --strict-warnings` exits 1
on every simulated session. That is intended: a strict check must never pass
simulated data.

## Known limits

- **Board-STIM laser routes are not emulated.** The laser has no trigger source
  or board STIM line, so only the direct NI software route exists. That route
  needs the First Reach trigger from the stim camera in stimulation mode at
  900 Hz, which is not simulated, so a protocol row cannot fire the laser. Use
  the Laser tab's Run Pulse.
- **The calibration is a test fixture.** 3D coordinates are plausible, not
  meaningful. The diamond-triangle file is not seeded, so recording logs
  "Animal Send Pos reset to 0 due to not fully valid diamond-triangle config".
- **The emulator's stimulus inputs** read idle in a simulation. Outside one the
  emulator still reports its old fixed pattern (stimulus 1 and 3 high), which
  the host decodes as a constant Tone 1 GPIO observation. Tones in a simulation
  are matched to their PLAY_TONE command rows, never to the periodic inputs.
- **No realistic reaches.** `MemoryPoseModel` returns a constant output, so
  trials run and end but are not scored, and automatic sends follow whatever
  that output implies. Send a trial with the Send button.
- **Refresh Hardware's camera scan** looks for Spinnaker cameras and logs a
  failure where PySpin is not installed. The simulated cameras are unaffected.
- **Host clocks.** The simulated NI board and cameras run on `perf_counter`,
  so their RAW stamps (the NI block's `raw_time`, the latch's RAW bracket) are
  `perf_counter` reads too, not CLOCK_MONOTONIC_RAW. A real board's or
  camera's oscillator is smooth against RAW. `perf_counter` is not where NTP
  slews it: in WSL it ran 0.34% fast against RAW, easing to 0.10% over 10 s.
  Before 2026-11-19 the sim stamped real RAW, so that slew went into the
  NI-to-RAW and latch fits. They passed or failed by chance, and about 2% of
  e2e runs warned `ambiguous_hardware_edge` in `alignment.frames`. A
  simulated session's RAW column is therefore not a host RAW reading.

## Tests

- `tests/simulation_end_to_end_test.py` records a 20 s session in a temporary
  sim home in a fresh interpreter (the forkserver rule above): it connects,
  creates an animal and a five-row protocol with a Tone 1 before each send,
  sends one trial, runs a laser train, stops, and validates with `--full`. It
  takes about 70 s, so it is marked `functional` and runs only with
  `pytest --functional tests/simulation_end_to_end_test.py`.
- `tests/backend_conformance_test.py` runs each contract's script on the real
  and the simulated backend, in the default suite.
