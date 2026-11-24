# Backend interfaces

reachAQ talks to four hardware subsystems through four contracts. Each is a
typing `Protocol` that the existing classes already satisfy, so nothing was
renamed or moved to introduce it. Backends are chosen where they always were;
there is no backend registry and no configuration key for it.

| Subsystem | Contract | Backends | Selection point |
|---|---|---|---|
| Camera | `reachaq.video.camera.camera_backend.CameraBackend` | `SpinCam` (`spinnaker://`), `RandomCam` (`random://`), `PlaybackCam` (`playback://`), `SimulatedCam` (`sim://`) | the camera URL's scheme, in `VideoManager.create_camera` (`video_manager.py`); `--random-cameras` rewrites the scheme for one run |
| NI-DAQ input stream | `reachaq.device.nidaq_input_backend.NidaqInputBackend` | `NidaqSignalStreamController`, `SimulatedNidaqStream` | `make_nidaq_input_backend`, called by the stream worker (`nidaq_signal_monitor_model.py`) and both preflight workers (`nidaq_preflight.py`); device discovery is `discover_nidaq_devices` (`nidaq_discovery.py`) |
| Laser | `reachaq.device.laser.LaserControllerProtocol` | `NidaqLaserController`, `NullLaserController`, `SimulatedLaserController` | `make_laser_controller`, called by `LaserModel`'s two construction points (`configure_nidaq`, `open_controller`) |
| Motor/pellet board over CAN | `reachaq.device.motor_pellet_backend.MotorPelletBackend` (a subset of `DeviceInterface`) | `CanInterface` (pyjerrycan, SocketCAN, PCAN transports), `EmulationInterface` | `CanDevice._make_device_interface`, from `AUTOTRAINER_CAN_TRANSPORT` |

Every factory returns today's class unless `REACHAQ_SIMULATION=1` is set, so a
rig run constructs exactly what it did before. The simulated backends are
described in [simulation mode](simulation-mode.md).

## What each contract promises

The protocol's docstring is the full statement; this is the part a new backend
most often gets wrong.

**Camera** (`CameraBackend`). Query values from the URL reach `set_property`
before `init()`; `set_property` returns False for a name it does not know. The
capture process calls `prepare_capture` once, `capture()` repeatedly on its own
thread, and `end_capture` once. `capture()` returns a uint8 `(height, width)`
frame and a stamp in ns on the camera's clock. `frame_id` names the exposure:
when `is_trigger_synchronized` is True every camera gives the same exposure the
same id, and a gap in ids is a dropped frame. `frame_poll_perf_c` and
`frame_arrival_perf_c` bracket the host poll that returned the frame.
`latch_clock()` returns `(perf before, camera ns, perf after, RAW before, RAW
after)`, or None for a camera without a latchable clock, and never raises.

**NI-DAQ input** (`NidaqInputBackend`). The worker constructs it, calls `start`
once, `read_chunk` in a loop, then `close`; the preflight only calls
`verify_tasks` and `close`. Sample i is at `epoch + i / sample_rate_hz` on the
board's clock, and blocks are contiguous (`sample_index` counts from start).
The block stamp rule: `perf_time`, `wall_time` and `raw_time` are taken before
the blocking read, so a block's stamp is about the end of the previous block.
The session's NI-to-host fit depends on that order. `values` are scaled (volts,
or 0/1 for digital lines). `close` may be called more than once.

**Laser** (`LaserControllerProtocol`). A waited-for `run_pulse_train` returns
once the train has ended, with the command back on the channel's minimum. A
train refused before anything was driven raises `LaserPulseRefused`; one
cancelled while it ran raises `LaserPulseCancelled`. `close` closes the
shutters itself: `LaserModel.close` calls nothing else first.

**Motor/pellet** (`MotorPelletBackend`). `open`, `close`, `can_read` and `read`
run on the reader thread; commands run on CanDevice's command thread and return
True once handed to the bus, without waiting for the board. A command the board
acknowledges allocates a new uuid: CanDevice reads `uuid()` before and after
the call and waits for `Acknowledge(uuid=uuid())`. `read` returns the decoded
`device_interface` messages received since the last call. `CanInterface` stamps
each with its host receipt time (`timestamp_ns`, and `index` in perf ns), which
the session times acknowledgements by; `EmulationInterface` does so in a
simulation run only. `pulse_digital_output`
raises ValueError for a line other than board STIM2/STIM3, or a duration outside
100 us to 5 s, before anything is sent.

## Adding a backend

1. Implement the protocol. Check it with `isinstance(obj, Protocol)`: all four
   are `runtime_checkable`.
2. Add one branch at the selection point above. Keep today's class the default,
   so nothing changes on a rig that does not ask for the new one. A camera
   backend also needs a `CameraKind` member and its scheme in `create_camera`.
3. Add it to `tests/backend_conformance_test.py`. Each contract has one
   behavioural script there, parametrized over its backends; the real backends
   run on the driver fakes their own tests use (`nidaq_daqmx_fake`, the
   NI-DAQ stream test's `_FakeNidaqmx`, a faked PySpin camera, a faked
   pyjerrycan driver). A new backend gets a fixture and a parameter.
4. If the backend needs configuration, that is a schema change and a
   `SystemConfiguration` version bump, which this mechanism deliberately avoids.
   A config-selected backend registry is planned for Phase 2.

Run the conformance tests:

```bash
pytest tests/backend_conformance_test.py
```
