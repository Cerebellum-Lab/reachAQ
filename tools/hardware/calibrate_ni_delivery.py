"""Measure this rig's NI delivery calibration with an AO-to-AI loopback.

The camera-latch pairing's residual sits at the NI-to-host fit's delivery floor
plus the camera's own stamp offset, minus the exposure, and the residual gate
(tools.latency.clocks.latch_residual_band) is centred there. Nothing inside a
session measures that floor, so this does: it records isolated sessions while
the host steps an unused analog output, jumpered to an unused analog input, at
random times it stamps on both host clocks. Each step lands in nidaq.h5, and
the session's own NI-to-host fit says how late it maps it (L). A session whose
fit is a whole NI read late, or whose latch pairing is a frame off, is left
out and named. The rest give residual + exposure; their mean and standard
deviation are nidaqStream.deliveryCalibration.

    python -m tools.hardware.calibrate_ni_delivery --ao PXI1Slot5/ao0 --ai PXI1Slot5/ai6
    python -m tools.hardware.calibrate_ni_delivery --ao PXI1Slot5/ao0 --ai PXI1Slot5/ai6 --write

Before a run: a BNC cable from the AO terminal to the AI terminal (on a
BNC-2090A, the AI pair's switch at SE), both channels unused in the live
configuration, and nothing else connected to them. reachAQ and SpinView must be
closed. The live configuration is only read: every session runs from a scratch
copy (CAN, pellet controller and laser off, the loopback input added) with
scratch preferences and data under --out, and the live file is fingerprinted
before and after each session. Only --write changes it, once every session has
finished, after copying it to a timestamped backup beside it. The AO output is
left at 0 V.

Each session takes its length plus about 40 s; the default plan, 12 sessions of
120 s, is about half an hour. A report (report.json, and a table on stdout)
records every session, used or not. Exit status 0 only when a calibration was
measured (and, with --write, written).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence

import numpy

HIGH = 1.0                      # volts on the loopback output
STEP_SECONDS = (0.04, 0.16)     # the random interval between output steps
# Steps of one level are at least two minimum intervals (80 ms) apart, so 30 ms
# matches each detected edge to its own command even when the fit is a whole NI
# read (5 ms at 200 Hz display refresh) late.
MATCH_SECONDS = 0.030
SESSION_OVERHEAD_SECONDS = 600  # startup, READY and validation, past the recording
DEFAULT_REFRESH_HZ = (60, 120, 144, 165)
DEFAULT_FRAME_RATES = (50, 100, 150)
LOOPBACK_CHANNEL = "loopback"
# DAQmx gives the sample-clock-to-convert delay in ticks of the AI convert timebase.
_CONVERT_TIMEBASE_HZ = {"TWENTY_MHZ_TIMEBASE": 20e6, "ONE_HUNDRED_KHZ_TIMEBASE": 100e3,
                        "EIGHTY_MHZ_TIMEBASE": 80e6}


def is_primary(camera) -> bool:
    """The camera's primary flag, read as reachAQ reads it (VideoCaptureModel)."""
    raw = (camera.params or {}).get("primary") or "false"
    return raw.lower() in {"yes", "true", "1", "on"} if isinstance(raw, str) else raw is True


def primary_camera(configuration):
    return next((c for c in configuration.cameras if c.is_enabled and is_primary(c)), None)


@dataclasses.dataclass(frozen=True)
class PlannedSession:
    name: str
    refresh_hz: float
    fps: int
    exposure_us: int
    seconds: float


def plan_sessions(refresh_rates: Sequence[float], frame_rates: Sequence[int],
                  exposures_us: Sequence[int], *, passes: int, seconds: float,
                  seed: int) -> List[PlannedSession]:
    """Every refresh x frame rate x exposure, `passes` times, each pass in its own random order."""
    rng = random.Random(seed)
    sessions = []
    for number in range(1, passes + 1):
        cells = [(r, f, e) for r in refresh_rates for f in frame_rates for e in exposures_us]
        rng.shuffle(cells)
        sessions.extend(PlannedSession(f"r{r:g}_f{f}_e{e}_p{number}", float(r), int(f), int(e),
                                       float(seconds)) for r, f, e in cells)
    return sessions


# ---------------------------------------------------------------- analysis


def loopback_edges(values, sample_index, rate: float, *, high: float = HIGH):
    """(NI seconds, rising) of each clean step on the loopback channel.

    An edge is the first sample at the new level. Clean: no gap in sample_index
    beside it, and the levels either side lasting at least the minimum step
    interval less a scan each side (the first level may be shorter).
    """
    values = numpy.asarray(values, dtype=numpy.float64)
    index = numpy.asarray(sample_index, dtype=numpy.int64)
    above = values > 0.5 * high
    change = numpy.flatnonzero(above[1:] != above[:-1]) + 1
    if not change.size:
        return numpy.empty(0), numpy.empty(0, dtype=bool)
    runs = numpy.diff(numpy.concatenate([[0], change, [values.size]]))
    shortest = (STEP_SECONDS[0] - 2.0 / rate) * rate
    clean = index[change] - index[change - 1] == 1
    clean &= (runs[:-1] >= shortest) | (numpy.arange(change.size) == 0)
    clean &= runs[1:] >= shortest
    return index[change][clean] / float(rate), above[change][clean]


def match_commands(edge_host, rising, commands):
    """For each edge mapped to host time, the row of its command (level, before, after), or -1.

    The nearest command of the same level within MATCH_SECONDS of the write's midpoint."""
    commands = numpy.asarray(commands, dtype=numpy.float64).reshape(-1, 3)
    level, middle = commands[:, 0], commands[:, 1:3].mean(axis=1)
    matched = numpy.full(len(edge_host), -1)
    for position, (when, up) in enumerate(zip(edge_host, rising)):
        candidates = numpy.flatnonzero(level == (HIGH if up else 0.0))
        if not candidates.size or not math.isfinite(when):
            continue
        nearest = candidates[numpy.argmin(numpy.abs(middle[candidates] - when))]
        if abs(middle[nearest] - when) < MATCH_SECONDS:
            matched[position] = nearest
    return matched


def delivery_floor(differences, convert_offset: float, rate: float) -> float:
    """How late the fit maps the sample clock, from edge minus command differences.

    The first scan to see a step is up to one sample period after it, uniformly,
    and the loopback input is converted convert_offset after the sample clock,
    so the fit's lateness is the mean difference plus that offset, less half a period."""
    differences = numpy.asarray(differences, dtype=numpy.float64)
    differences = differences[numpy.isfinite(differences)]
    if not differences.size:
        return math.nan
    return float(differences.mean()) + convert_offset - 0.5 / rate


def session_verdict(*, floor: float, read_period: float, residual: Optional[float],
                    exposure: Optional[float], frame_period: Optional[float]) -> str:
    """Why a session cannot calibrate, or "" when it can.

    A fit paired a whole NI read off is late by about that read, so a floor of
    half a read or more is no floor; and residual + exposure - floor is the
    camera's own stamp offset, tens of microseconds, which a latch map or
    pairing gone wrong moves by milliseconds, so it must stay under a quarter
    period."""
    if not math.isfinite(floor):
        return "no loopback edge was matched"
    if not floor < read_period / 2:
        return (f"the NI-to-host fit is {floor * 1e3:.3f} ms late, half an NI read "
                f"({read_period * 1e3:.3f} ms) or more")
    if residual is None or exposure is None or frame_period is None:
        return "no latch residual, exposure or frame period in alignment.json"
    offset = residual + exposure - floor
    if not abs(offset) < frame_period / 4:
        return (f"the camera's stamp offset reads {offset * 1e3:.3f} ms, a quarter period "
                f"({frame_period / 4 * 1e3:.3f} ms) or more: the latch pairing is not where "
                "the NI edges are")
    return ""


def analyze_session(run_dir: Path, convert_offset: float, camera_serial: str = "") -> dict:
    """One recorded session's loopback result, from its nidaq.h5, alignment.json and commands."""
    import h5py
    from tools.latency import clocks

    result = json.loads((run_dir / "result.json").read_text())
    streams = Path(result["session"]) / "streams"
    alignment = json.loads((streams / "alignment.json").read_text())["cameraNidaqAlignment"]
    commands = json.loads((run_dir / "commands.json").read_text())
    with h5py.File(streams / "nidaq.h5", "r") as store:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in store.attrs["channel_names"]]
        rate = float(store.attrs["sample_rate_hz"])
        values = store["values"][names.index(LOOPBACK_CHANNEL), :]
        index = store["sample_index"][()]
    latch = alignment.get("latch") or {}
    clock = latch.get("clock") or "perf"
    fit = clocks.fit_from_record(alignment.get("niToHost"))
    out = {"name": run_dir.name, "session": result["session"], "refreshHz": result["refresh_hz"],
           "readChunkSamples": result["read_chunk_samples"], "fps": result["fps"],
           "exposureSeconds": alignment.get("exposureSeconds"), "clock": clock,
           "framePeriodSeconds": alignment.get("framePeriodSeconds"),
           "residualSeconds": latch.get("medianResidualSeconds"),
           "latchStatus": latch.get("status"), "commanded": len(commands["rows"]),
           "cameraSerial": camera_serial}
    ni_seconds, rising = loopback_edges(values, index, rate)
    out["edges"] = int(ni_seconds.size)
    floor, spread = math.nan, math.nan
    if fit is not None and fit.valid and ni_seconds.size:
        rows = numpy.asarray(commands["rows"], dtype=numpy.float64).reshape(-1, 5)
        if clock == "raw" and fit.has_raw:
            host, stamps = fit.to_raw(ni_seconds), rows[:, [0, 3, 4]]
        else:
            host, stamps = fit.map(ni_seconds), rows[:, [0, 1, 2]]
        matched = match_commands(host, rising, stamps)
        used = matched >= 0
        differences = host[used] - stamps[matched[used], 1:3].mean(axis=1)
        out["matched"] = int(used.sum())
        if used.sum() >= 2:
            floor = delivery_floor(differences, convert_offset, rate)
            spread = float(numpy.std(differences, ddof=1))
    out["deliveryFloorSeconds"] = floor
    out["differenceSpreadSeconds"] = spread
    read_period = float(result["read_chunk_samples"]) / rate
    out["reason"] = session_verdict(floor=floor, read_period=read_period,
                                    residual=out["residualSeconds"],
                                    exposure=out["exposureSeconds"],
                                    frame_period=out["framePeriodSeconds"])
    out["offsetSeconds"] = (None if out["reason"] else
                            float(out["residualSeconds"]) + float(out["exposureSeconds"]))
    return out


def aggregate(sessions: Iterable[dict], *, measured: str):
    """The calibration from the sessions that can calibrate, or (None, why)."""
    from autotrainer.core.configuration import NidaqDeliveryCalibration

    used = [s for s in sessions if not s.get("reason")]
    if len(used) < 2:
        return None, f"{len(used)} usable session(s); a spread needs two"
    offsets = [s["offsetSeconds"] for s in used]
    floors = [s["deliveryFloorSeconds"] for s in used]
    calibration = NidaqDeliveryCalibration(
        offset_seconds=float(statistics.mean(offsets)),
        spread_seconds=float(statistics.stdev(offsets)),
        sessions=len(used),
        measured=measured,
        read_chunk_samples=tuple(sorted({int(s["readChunkSamples"]) for s in used})),
        frame_rates_hz=tuple(sorted({float(s["fps"]) for s in used})),
        exposures_seconds=tuple(sorted({float(s["exposureSeconds"]) for s in used})),
        camera_serials=tuple(sorted({str(s["cameraSerial"]) for s in used if s["cameraSerial"]})),
        delivery_floor_min_seconds=float(min(floors)),
        delivery_floor_max_seconds=float(max(floors)),
    )
    return calibration, ""


def write_calibration(config_path: Path, calibration) -> Path:
    """Put calibration into config_path's nidaqStream, after copying the file to a backup."""
    from autotrainer.core import SystemConfiguration

    config_path = Path(config_path)
    live = SystemConfiguration.load_yaml_file(config_path)
    backup = config_path.with_name(
        f"{config_path.stem}_before_delivery_calibration_{datetime.now():%Y%m%d_%H%M%S}.yaml")
    shutil.copy2(config_path, backup)
    live.nidaq_stream = dataclasses.replace(live.nidaq_stream, delivery_calibration=calibration)
    live.save_file(config_path.with_suffix(""), as_yaml=True)
    reread = SystemConfiguration.load_yaml_file(config_path)
    if reread.nidaq_stream.delivery_calibration != calibration:
        raise RuntimeError(f"{config_path} did not read back the calibration; backup {backup}")
    return backup


# ---------------------------------------------------------------- the rig


def convert_offset(ai: str, scan: Sequence[str], terminal: str, rate: float) -> dict:
    """When, after the sample clock, the stream's scan converts the loopback input.

    The same channels, order, terminal configuration and rate as the stream, committed
    (never started) so DAQmx picks the convert rate it will use."""
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration

    with nidaqmx.Task() as task:
        for name in scan:
            task.ai_channels.add_ai_voltage_chan(
                name, terminal_config=getattr(TerminalConfiguration, terminal.upper()))
        task.timing.cfg_samp_clk_timing(rate, sample_mode=AcquisitionType.CONTINUOUS,
                                        samps_per_chan=int(rate * 10))
        task.control(nidaqmx.constants.TaskMode.TASK_COMMIT)
        conv = float(task.timing.ai_conv_rate)
        delay = float(task.timing.delay_from_samp_clk_delay)
        units = str(task.timing.delay_from_samp_clk_delay_units)
        timebase_source = str(task.timing.ai_conv_timebase_src).rsplit(".", 1)[-1]
    if "TICKS" in units.upper():
        if timebase_source not in _CONVERT_TIMEBASE_HZ:
            raise RuntimeError(f"unknown AI convert timebase {timebase_source}; the delay "
                               f"from the sample clock ({delay} ticks) cannot be converted")
        delay_seconds = delay / _CONVERT_TIMEBASE_HZ[timebase_source]
    else:
        delay_seconds = delay
    position = list(scan).index(ai)
    return {"convertRateHz": conv, "delayFromSampleClockSeconds": delay_seconds,
            "scanPosition": position, "offsetSeconds": delay_seconds + position / conv}


def jumper_check(ao: str, ai: str, terminal: str) -> dict:
    """Step the output through 0, HIGH/2 and HIGH and read the input: does it follow?"""
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration
    from nidaqmx.stream_writers import AnalogSingleChannelWriter

    levels = (0.0, HIGH / 2, HIGH, 0.0)
    readings = []
    with nidaqmx.Task() as out, nidaqmx.Task() as inp:
        out.ao_channels.add_ao_voltage_chan(ao, min_val=-2 * HIGH, max_val=2 * HIGH)
        inp.ai_channels.add_ai_voltage_chan(
            ai, terminal_config=getattr(TerminalConfiguration, terminal.upper()),
            min_val=-2 * HIGH, max_val=2 * HIGH)
        writer = AnalogSingleChannelWriter(out.out_stream, auto_start=False)
        out.start()
        inp.start()
        for level in levels:
            writer.write_one_sample(level)
            time.sleep(0.05)
            readings.append(float(numpy.mean([inp.read() for _ in range(50)])))
        writer.write_one_sample(0.0)
    # Within a twentieth of the step: a floating input sits at a rail or wanders.
    follows = all(abs(r - level) < HIGH / 20 for r, level in zip(readings, levels))
    return {"levels": list(levels), "readings": readings, "follows": follows}


def _edge_writer(stop, count, out_path, seed, ao, ref_source, ref_rate):
    """Spawned child: step `ao` until stop, stamping perf and RAW around each write.

    A child of its own, never a thread of the recording process: closing an AO
    task can start NI's helper processes, and forked from the recording
    process they would inherit the session's NI spool and hold its HDF5 lock,
    so the session could not be finalized. A spawned child passes them only
    its own descriptors. Rows (level, perf before, perf after, RAW before, RAW
    after) go to out_path when it ends."""
    rows, errors = [], []
    try:
        import nidaqmx
        from nidaqmx.stream_writers import AnalogSingleChannelWriter
        from autotrainer.core.latency.raw_clock import monotonic_raw_time
        from autotrainer.device.nidaq_reference_clock import apply_reference_clock

        rng = random.Random(seed)
        with nidaqmx.Task("reachaq_delivery_loopback_ao") as task:
            task.ao_channels.add_ao_voltage_chan(ao, min_val=-2 * HIGH, max_val=2 * HIGH)
            # The stream locks the board to the chassis clock; a task asking for
            # another reference clock on the same board is refused at start.
            apply_reference_clock(nidaqmx, task, SimpleNamespace(
                reference_clock_source=ref_source, reference_clock_rate_hz=ref_rate))
            writer = AnalogSingleChannelWriter(task.out_stream, auto_start=False)
            task.start()
            writer.write_one_sample(0.0)
            level = 0.0
            while True:
                stopping = stop.wait(rng.uniform(*STEP_SECONDS))
                if stopping and level == 0.0:
                    break
                level = 0.0 if stopping else HIGH - level
                p0, r0 = time.perf_counter(), monotonic_raw_time()
                writer.write_one_sample(level)
                r1, p1 = monotonic_raw_time(), time.perf_counter()
                rows.append((level, p0, p1, r0, r1))
                count.value = len(rows)
    except BaseException as error:  # noqa: BLE001 - reported to the recording process
        errors.append(repr(error))
    Path(out_path).write_text(json.dumps({"rows": rows, "errors": errors}))
    sys.exit(1 if errors else 0)


def record_session(cfg: Path, prefs: Path, seconds: float, refresh_hz: float, fps: int,
                   out_dir: Path, ao: str) -> int:
    """One isolated recording with the loopback output stepping; writes result.json and commands.json."""
    import faulthandler
    import logging
    import multiprocessing

    faulthandler.enable()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    multiprocessing.set_start_method("spawn")
    out_dir.mkdir(parents=True, exist_ok=True)

    from tools.acquisition.instance_lock import acquire_instance_lock
    lock = acquire_instance_lock()  # noqa: F841 - held until exit
    from autotrainer.core.logging import setup_logging, stop_multiproc_logging
    logger = setup_logging(logger_level=logging.INFO, multiprocess_enabled=True)
    from tools.acquisition.model.app_model import AppModel
    from tools.acquisition.model.app_model_status import AppModelStatus, SessionRecordingStatus
    from tools.acquisition.model.user_preferences import UserPreferences

    def say(message):
        print(f"CALIBRATE {time.strftime('%H:%M:%S')} {message}", flush=True)

    result = {"refresh_hz": refresh_hz, "fps": fps, "seconds": seconds, "ok": False}
    stop, count = multiprocessing.Event(), multiprocessing.Value("i", 0)
    rows_path = out_dir / "commands.json"
    rows_path.unlink(missing_ok=True)
    writer = None

    def collect():
        stop.set()
        if writer is not None:
            writer.join(10.0)
        if not rows_path.exists():
            raise RuntimeError(f"the output stepper left no commands "
                               f"(exit {None if writer is None else writer.exitcode})")
        written = json.loads(rows_path.read_text())
        if written["errors"]:
            raise RuntimeError(f"the output stepper failed: {written['errors']}")
        return len(written["rows"])

    app = AppModel(UserPreferences(settings_file_path=prefs))
    try:
        app.load_configuration(cfg)
        if app.nidaq_plan_error:
            raise RuntimeError(f"NI-DAQ plan refused: {app.nidaq_plan_error}")
        monitor = app._nidaq_signal_monitor
        monitor.set_display_refresh_rate(refresh_hz)
        result["read_chunk_samples"] = int(monitor.effective_read_chunk_size)
        if not app.capture_start(target_status=AppModelStatus.RUNNING):
            raise RuntimeError("capture_start failed")
        app.on_activated()
        time.sleep(10)
        app.add_animal("DELIVERY-CALIBRATION", select=True)
        if not app.start_recording():
            raise RuntimeError(f"start_recording refused: {list(app.recording_blockers)}")
        say(f"recording {seconds:g} s at {fps} fps, NI read {result['read_chunk_samples']}")
        time.sleep(2.0)
        # The plan the running stream applied, its reference clock included: set when the
        # stream starts, so read now, not after loading the configuration.
        plan = monitor.timing_plan
        if plan is None:
            raise RuntimeError("the NI stream is recording without a timing plan")
        writer = multiprocessing.Process(
            target=_edge_writer, name="delivery-loopback-ao", daemon=True,
            args=(stop, count, str(rows_path), int(time.time()), ao,
                  getattr(plan, "reference_clock_source", None),
                  getattr(plan, "reference_clock_rate_hz", None)))
        writer.start()
        end = time.time() + seconds
        while time.time() < end:
            time.sleep(min(10.0, max(0.1, end - time.time())))
            if writer.exitcode is not None:
                collect()
                raise RuntimeError(f"the output stepper stopped early (exit {writer.exitcode})")
        result["commands"] = collect()
        time.sleep(2.0)
        started = time.time()
        app.stop_recording()
        while app.session_recording_status != SessionRecordingStatus.READY:
            if time.time() - started > 300:
                raise RuntimeError("not READY 300 s after Stop")
            time.sleep(0.05)
        result["session"] = str(app.last_stopped_session_path)
        while app.session_validation_state["status"] in ("pending", "running"):
            if time.time() - started > 600:
                raise RuntimeError("validation still running 600 s after Stop")
            time.sleep(0.1)
        result["validation"] = app.session_validation_state.get("status")
        result["ok"] = result["session"] not in ("", "None")
        say(f"READY; {result['session']}; validation {result['validation']}; "
            f"{result['commands']} steps")
    except BaseException as error:  # noqa: BLE001 - reported in result.json
        logger.exception("delivery calibration session failed")
        result["error"] = repr(error)
        say(f"failed: {error!r}")
    finally:
        stop.set()
        if writer is not None:
            writer.join(10.0)
        app.on_close()
        from autotrainer.behavior import BehaviorAlgorithm
        from autotrainer.core.event.event_manager import EventManager
        BehaviorAlgorithm.close_algorithm_handler()
        EventManager.try_close_default()
        stop_multiproc_logging()
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return 0 if result["ok"] else 1


def scratch_configuration(live, root: Path, ai: str, fps: int, exposure_us: int):
    """The live configuration with the loopback input, at one frame rate and exposure,
    recording under root with CAN, the pellet controller and the laser off."""
    from autotrainer.core.configuration import NidaqSignalChannelConfiguration

    conf = live.__class__.load_yaml(_yaml_stream(live.dump_yaml()))
    primary = primary_camera(conf)
    if primary is None:
        raise RuntimeError("the live configuration has no enabled primary camera")
    # The cameras that run with the primary (its triggered secondaries too); a stimulation
    # camera at its own rate is left alone.
    rate = (primary.params or {}).get("fps")
    for camera in conf.cameras:
        if camera.is_enabled and (camera.params or {}).get("fps") == rate:
            camera.params["fps"] = fps
            camera.params["exposure"] = exposure_us
    conf.hardware.can_enabled = False
    conf.hardware.pellet_controller_enabled = False
    conf.hardware.nidaq_enabled = True
    conf.laser = dataclasses.replace(conf.laser, backend="disabled")
    conf.persistence = dataclasses.replace(conf.persistence, output_location=str(root / "data"))
    loopback = NidaqSignalChannelConfiguration(name=LOOPBACK_CHANNEL, physical_channel=ai,
                                               kind="analog", unit="V")
    conf.nidaq_stream = dataclasses.replace(
        conf.nidaq_stream, channels=tuple(conf.nidaq_stream.channels) + (loopback,),
        delivery_calibration=None)
    return conf


def _yaml_stream(text: str):
    import io
    return io.StringIO(text)


def _fingerprint(config_path: Path) -> str:
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()


def _run_session(planned: PlannedSession, cfg: Path, prefs: Path, run_dir: Path, ao: str) -> bool:
    """Record one session in its own process; False if the run was interrupted.

    An interrupt (SIGINT, or SIGTERM, which main turns into one) is handed on to the
    session as SIGINT, which closes the app the normal way, and waited for: a session
    left running after its run would hold the cameras and NI with nobody watching."""
    child = subprocess.Popen([sys.executable, "-m", "tools.hardware.calibrate_ni_delivery",
                              "record", str(cfg), str(prefs), str(planned.seconds),
                              str(planned.refresh_hz), str(planned.fps), str(run_dir), ao])
    try:
        child.wait(timeout=planned.seconds + SESSION_OVERHEAD_SECONDS)
        return True
    except subprocess.TimeoutExpired:
        print(f"{planned.name} timed out; stopping it")
        _stop(child)
        return True
    except KeyboardInterrupt:
        _stop(child)
        return False


def _stop(child: subprocess.Popen) -> None:
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=120)
    except subprocess.TimeoutExpired:
        print(f"session pid {child.pid} did not stop on SIGINT; killing it")
        child.kill()
        child.wait()


def _refuse_if_running() -> None:
    from tools.acquisition.instance_lock import acquire_instance_lock
    lock = acquire_instance_lock()  # raises if reachAQ is running
    if lock is not None:
        lock.close()
    spinview = subprocess.run(["pgrep", "-x", "SpinView_QT"], capture_output=True)
    if spinview.returncode == 0:
        raise RuntimeError("SpinView is running and holds the cameras; close it")


def _print_table(sessions: Sequence[dict]) -> None:
    def us(value):
        return "     -" if value is None or not math.isfinite(value) else f"{value * 1e6:6.1f}"
    print(f"{'session':22} {'read':>4} {'fps':>4} {'edges':>5} {'L':>6} {'sd':>6} "
          f"{'resid':>6} {'D':>6}  verdict")
    for s in sessions:
        print(f"{s['name']:22} {s.get('readChunkSamples') or 0:>4} {s.get('fps') or 0:>4} "
              f"{s.get('matched', 0):>5} {us(s.get('deliveryFloorSeconds'))} "
              f"{us(s.get('differenceSpreadSeconds'))} {us(s.get('residualSeconds'))} "
              f"{us(s.get('offsetSeconds'))}  {s.get('reason') or 'used'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    record = sub.add_parser("record", help=argparse.SUPPRESS)
    for name in ("cfg", "prefs", "seconds", "refresh", "fps", "out", "ao"):
        record.add_argument(name)
    parser.add_argument("--ao", help="the loopback output, e.g. PXI1Slot5/ao0")
    parser.add_argument("--ai", help="the loopback input, e.g. PXI1Slot5/ai6")
    parser.add_argument("--config", type=Path,
                        default=Path.home() / "Autotrainer" / "system_configuration.yaml")
    parser.add_argument("--out", type=Path, default=None,
                        help="where sessions, data and the report go (default /tmp/reachaq_delivery_<time>)")
    parser.add_argument("--refresh", type=float, nargs="+", default=list(DEFAULT_REFRESH_HZ),
                        help="display refresh rates, which set the NI read size")
    parser.add_argument("--fps", type=int, nargs="+", default=list(DEFAULT_FRAME_RATES))
    parser.add_argument("--exposure-us", type=int, nargs="+", default=None,
                        help="camera exposures (default: the live primary camera's)")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--write", action="store_true",
                        help="put the calibration into --config (after a backup)")
    args = parser.parse_args(argv)
    if args.command == "record":
        return record_session(Path(args.cfg), Path(args.prefs), float(args.seconds),
                              float(args.refresh), int(args.fps), Path(args.out), args.ao)
    if not (args.ao and args.ai):
        parser.error("--ao and --ai name the jumpered loopback output and input")
    # A SIGTERM stops the run the way Ctrl-C does, so the session in hand is shut down too.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    from autotrainer.core import SystemConfiguration

    live = SystemConfiguration.load_yaml_file(args.config)
    stream = live.nidaq_stream
    used = {c.physical_channel for c in stream.channels}
    if args.ai in used:
        raise SystemExit(f"{args.ai} is already a channel of the live NI stream")
    if any(str(c.analog_output) == args.ao for c in getattr(live.laser, "channels", ())):
        raise SystemExit(f"{args.ao} is a laser output in the live configuration")
    primary = primary_camera(live)
    if primary is None:
        raise SystemExit("the live configuration has no enabled primary camera")
    exposures = args.exposure_us or [int((primary.params or {}).get("exposure"))]
    serial = str(primary.host)
    root = args.out or Path(f"/tmp/reachaq_delivery_{datetime.now():%Y%m%d_%H%M%S}")
    root.mkdir(parents=True, exist_ok=True)
    seed = int(time.time()) if args.seed is None else args.seed
    sessions = plan_sessions(args.refresh, args.fps, exposures, passes=args.passes,
                             seconds=args.seconds, seed=seed)
    report = {"config": str(args.config), "ao": args.ao, "ai": args.ai, "seed": seed,
              "started": datetime.now().astimezone().isoformat(timespec="seconds"),
              "sessions": []}
    print(f"{len(sessions)} sessions of {args.seconds:g} s; output in {root}", flush=True)

    _refuse_if_running()
    report["jumper"] = jumper_check(args.ao, args.ai, stream.analog_terminal_config)
    print(f"jumper: {args.ai} read {report['jumper']['readings']} for "
          f"{report['jumper']['levels']} V on {args.ao}", flush=True)
    if not report["jumper"]["follows"]:
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(f"{args.ai} does not follow {args.ao}: check the jumper and the input's switch")
        return 2
    scan = [c.physical_channel for c in stream.analog_channels] + [args.ai]
    report["convert"] = convert_offset(args.ai, scan, stream.analog_terminal_config,
                                       stream.sample_rate_hz)

    prefs = root / "prefs.ini"
    settings = root / "Autotrainer"
    prefs.write_text(f"[system]\nconfiguration_location={settings}\n"
                     f"animal_location={settings / 'animals'}\n\n"
                     f"[softmouse]\nmanifest_path={root / 'no_softmouse_manifest.csv'}\n"
                     f"nightly_refresh=false\n")
    before = _fingerprint(args.config)
    for planned in sessions:
        cfg = root / "cfgs" / f"f{planned.fps}_e{planned.exposure_us}.yaml"
        if not cfg.exists():
            scratch_configuration(live, root, args.ai, planned.fps,
                                  planned.exposure_us).save_file(cfg.with_suffix(""), as_yaml=True)
        run_dir = root / "runs" / planned.name
        print(f"=== {planned.name} {datetime.now():%H:%M:%S}", flush=True)
        if not _run_session(planned, cfg, prefs, run_dir, args.ao):
            report["aborted"] = f"stopped during {planned.name}"
            (root / "report.json").write_text(json.dumps(report, indent=2, default=str))
            print(f"stopped; {planned.name} was shut down and nothing was written")
            return 130
        if _fingerprint(args.config) != before:
            report["aborted"] = f"{args.config} changed during {planned.name}"
            (root / "report.json").write_text(json.dumps(report, indent=2, default=str))
            print(f"ABORT: {report['aborted']}")
            return 1
        try:
            outcome = json.loads((run_dir / "result.json").read_text())
            entry = ({"name": planned.name, "reason": f"recording failed: {outcome.get('error')}"}
                     if not outcome.get("ok") else
                     analyze_session(run_dir, report["convert"]["offsetSeconds"], serial))
        except Exception as error:  # noqa: BLE001 - one session's failure is data
            entry = {"name": planned.name, "reason": f"analysis failed: {error!r}"}
        report["sessions"].append(entry)
        (root / "report.json").write_text(json.dumps(report, indent=2, default=str))

    _print_table(report["sessions"])
    calibration, why = aggregate(report["sessions"],
                                 measured=datetime.now().astimezone().isoformat(timespec="seconds"))
    if calibration is None:
        report["calibration"] = None
        (root / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"no calibration: {why}")
        return 1
    report["calibration"] = dataclasses.asdict(calibration)
    print(f"\ndeliveryCalibration: offset {calibration.offset_seconds * 1e3:.4f} ms, spread "
          f"{calibration.spread_seconds * 1e3:.4f} ms over {calibration.sessions} sessions; "
          f"delivery floor {calibration.delivery_floor_min_seconds * 1e3:.3f}-"
          f"{calibration.delivery_floor_max_seconds * 1e3:.3f} ms")
    if args.write:
        backup = write_calibration(args.config, calibration)
        report["written"] = {"config": str(args.config), "backup": str(backup)}
        print(f"written to {args.config} (backup {backup})")
    else:
        print("not written; run again with --write, or add it to nidaqStream by hand")
    (root / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
