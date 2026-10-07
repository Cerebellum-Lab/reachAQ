"""Build streams/latency.h5 from a session's raw latency streams.

Runs inside session finalization and from the rebuild CLI. It never raises:
each raw stream is read on its own, each clock fit and each loop is analysed on
its own, and every failure is recorded as a reason, so one unreadable file or
one bad fit costs only what depends on it and a session always publishes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy

from autotrainer.core.latency.schema import CLOCK_PAIRS, LATENCY_SCHEMA_VERSION

from . import analysis, clocks

logger = logging.getLogger(__name__)

LATENCY_OUTPUT_NAME = "latency.h5"
OPEN_RETRY_SECONDS = 5.0
COMMAND_MIN_SWING = 0.1   # volts
DIODE_MIN_SWING = 0.05    # volts
TRIGGER_MIN_SWING = 0.5
# The NI channels whose edges the loops use, by name suffix, with the swing below
# which a channel is treated as not carrying a signal.
EDGE_CHANNELS = (("_command_copy", COMMAND_MIN_SWING), ("_diode", DIODE_MIN_SWING),
                 ("_trigger", TRIGGER_MIN_SWING))
# One group per mapping in latency.h5's /clock, keyed by its name in the status dict.
CLOCK_GROUPS = (("wallToHost", "wall_to_host"), ("niToHost", "ni_to_host"),
                ("cameraToNi", "camera_to_ni"))
# An HDF5 attribute is limited to 64 KB; a wall clock that stepped constantly
# must not cost the whole file, so only the first segments are listed.
MAX_LISTED_WALL_FITS = 16
ENVELOPE_BIAS_FLOOR = ("unmeasured; mapped NI times are late by about the minimum driver "
                       "delivery delay, likely under 1 ms")
SECONDARY_TRIGGER_DELAY = ("unmeasured; a secondary camera's exposure starts a trigger delay "
                           "(microseconds) after the primary's")
HOST_ARRIVAL_PAIRING = ("camera-to-NI pairing by host arrival assumes exposure→arrival "
                        "under one frame period")
HOST_ARRIVAL_PAIRING_REASON = f"{HOST_ARRIVAL_PAIRING} (no camera clock latch)"


def _text(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, numpy.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, numpy.bool_)):
        return bool(value)
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _app_version() -> str:
    # Imported here because importing it can ask git for a version, and the
    # finalizer must not pay for that, or fail on it, until it has something to write.
    try:
        from tools.autotrainer_version import __version__
        return str(__version__)
    except Exception:
        return ""


def _read_h5(path: Path) -> dict:
    """Every dataset and attribute of a raw stream, retrying while a writer still holds it."""
    import h5py
    deadline = time.monotonic() + OPEN_RETRY_SECONDS
    while True:
        try:
            with h5py.File(path, "r") as store:
                data = {name: store[name][()] for name in store
                        if isinstance(store[name], h5py.Dataset)}
                data["attrs"] = dict(store.attrs)
                return data
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def _read_nidaq(path: Path) -> dict:
    """What the loops need from nidaq.h5, reduced one array at a time.

    An hour at 10 kHz is tens of millions of samples per channel, and this runs
    inside the live app's finalization, so nothing sample-sized outlives its
    use: the block end times, the camera line's transitions and each laser
    channel's edges are kept, and the samples they came from are dropped.
    """
    import h5py
    with h5py.File(path, "r") as store:
        names = [_text(name) for name in store.attrs.get("channel_names", [])]
        rate = float(store.attrs.get("sample_rate_hz", math.nan))
        result = {
            "names": names, "rate": rate,
            "block_ends": (numpy.empty(0, dtype=numpy.int64), numpy.empty(0)),
            "transitions": None, "edges": {}, "problems": {},
        }
        if not math.isfinite(rate):
            return result
        stored = store["sample_index"]
        count = int(stored.shape[0])
        first = int(stored[0]) if count else 0
        # The index is almost always one unbroken run, first + position, and then
        # there is no need to hold or read the whole array. A gap (lost samples,
        # a restarted task) makes the last value differ from first + count - 1,
        # and the stored array is used.
        consecutive = count > 0 and int(stored[count - 1]) - first == count - 1
        index = (numpy.arange(first, first + count, dtype=numpy.int64) if consecutive
                 else stored[()])
        seen = store["block_observation_perf_time"][()]
        result["block_ends"] = clocks.ni_block_ends(index, seen)
        del seen
        if consecutive:
            index = None

        def sample_of(positions):
            return positions + first if index is None else index[positions]

        swings = dict(EDGE_CHANNELS)
        values = store["values"]
        for row, name in enumerate(names):
            suffix = next((suffix for suffix in swings if name.endswith(suffix)), None)
            if name != "cam_frames" and suffix is None:
                continue
            channel = None  # one channel's samples at a time
            try:
                channel = values[row, :]
                if name == "cam_frames":
                    result["transitions"] = sample_of(clocks.transitions(channel))
                else:
                    found = clocks.rising_edges(channel, min_swing=swings[suffix])
                    if found.usable:
                        result["edges"][name] = sample_of(found.positions) / rate
                    else:
                        result["problems"][name] = found.reason
            except Exception as error:
                result["problems"][name] = f"unreadable: {_describe(error)}"
            finally:
                channel = None
        return result


def _frame_period(frames) -> float:
    try:
        if frames is None or len(frames) < 2:
            return math.nan
        ids = numpy.asarray(frames["frame_id"], dtype=numpy.int64)
        ts = numpy.asarray(frames["camera_ts_ns"], dtype=numpy.float64) / 1e9
        consecutive = numpy.diff(ids) == 1
        if not consecutive.any():
            return math.nan
        return float(numpy.median(numpy.diff(ts)[consecutive]))
    except Exception:
        return math.nan


def _guarded(name: str, fit) -> clocks.ClockFit:
    """A fit that raised is an invalid fit with the exception as its reason."""
    try:
        return fit()
    except Exception as error:
        logger.exception("Latency %s fit failed", name)
        return clocks.invalid_fit(_describe(error))


def _wall_record(wall_map: clocks.WallMap, reason: str = "") -> dict:
    fits = wall_map.fits
    record = {
        "valid": any(fit.valid for fit in fits),
        "segments": len(fits),
        "points": sum(fit.points for fit in fits),
        "fits": [fit.as_record() for fit in fits[:MAX_LISTED_WALL_FITS]],
    }
    if not fits:
        record["reason"] = reason or "no clock pairs"
    return record


def _latch_direct_lag_p50(frames, latches) -> float:
    """Median of arrival minus camera timestamp mapped to host by the latches; NaN if none."""
    try:
        fit = clocks.fit_latches(latches)
        camera = numpy.asarray(frames["camera_ts_ns"], dtype=numpy.float64) / 1e9
        near = clocks.latch_window(fit, camera) if fit.valid else numpy.zeros(camera.shape, bool)
        lag = numpy.asarray(frames["arrival_perf"], dtype=numpy.float64)[near] - fit.map(camera[near])
        lag = lag[numpy.isfinite(lag)]
        return float(numpy.median(lag)) if lag.size else math.nan
    except Exception:
        logger.exception("Latency latch-direct lag failed")
        return math.nan


def _host_arrival_reason(latches) -> str:
    """Why the fallback pairing ran: no latch at all, or how many and why none served."""
    rows = 0 if latches is None else len(latches)
    if not rows:
        return HOST_ARRIVAL_PAIRING_REASON
    return (f"{HOST_ARRIVAL_PAIRING} ({rows} camera clock latch{'es' if rows != 1 else ''}, "
            f"none usable: bracket over {clocks.LATCH_MAX_BRACKET_SECONDS * 1e3:g} ms or not finite)")


def _ni_derived_lag_p50(frames, latches, camera_to_ni, ni_to_host) -> float:
    """Median of arrival minus the NI-derived exposure, over the frames the latches cover."""
    try:
        if not (camera_to_ni.valid and ni_to_host.valid):
            return math.nan
        fit = clocks.fit_latches(latches)
        camera = numpy.asarray(frames["camera_ts_ns"], dtype=numpy.float64) / 1e9
        near = clocks.latch_window(fit, camera) if fit.valid else numpy.zeros(camera.shape, bool)
        exposure = ni_to_host.map(camera_to_ni.map(camera[near]))
        lag = numpy.asarray(frames["arrival_perf"], dtype=numpy.float64)[near] - exposure
        lag = lag[numpy.isfinite(lag)]
        return float(numpy.median(lag)) if lag.size else math.nan
    except Exception:
        logger.exception("Latency NI-derived lag failed")
        return math.nan


def _flagged_primary(stream: dict) -> bool:
    try:
        return bool(stream["attrs"].get("is_primary", False))
    except Exception:
        return False


class _Context:
    """Everything the loops share: the raw streams, NI data and the clock fits."""

    def __init__(self, raw: dict, faults: dict, streams: Path, primary_camera: str,
                 status: dict):
        self.status = status
        self.faults = dict(faults)  # stream name -> why it could not be read
        self.events = raw.get("events") or {}
        self.pose = raw.get("pose")
        self.cameras = {key[len("camera_"):]: value for key, value in raw.items()
                        if key.startswith("camera_")}
        self.records = {key[len("record_"):]: value for key, value in raw.items()
                        if key.startswith("record_")}
        self.primary_name = self._choose_primary(primary_camera)
        self.primary_frames = self.cameras.get(self.primary_name, {}).get("frames")
        self.frame_period = _frame_period(self.primary_frames)
        self.nidaq: Optional[dict] = None
        nidaq_path = streams / "nidaq.h5"
        if nidaq_path.is_file():
            try:
                self.nidaq = _read_nidaq(nidaq_path)
            except Exception as error:
                logger.exception("nidaq.h5 could not be read for the latency record")
                self.faults["nidaq"] = _describe(error)
                status["reasons"].append(f"nidaq.h5 unreadable: {_describe(error)}")
        self.ni_to_host = clocks.invalid_fit("no NI stream")
        self.camera_to_ni = clocks.invalid_fit("not fitted")
        self.pairing: Optional[clocks.Pairing] = None
        self._fit_clocks()
        self._fit_wall(raw)

    def _choose_primary(self, requested: str) -> str:
        names = sorted(self.cameras)
        if not names or requested in self.cameras:
            return requested if requested in self.cameras else ""
        # The stream itself says which camera drove the exposure line; the
        # alphabetical first is only the last resort.
        flagged = [name for name in names if _flagged_primary(self.cameras[name])]
        chosen = flagged[0] if flagged else names[0]
        if not requested:
            why = "no primary camera given"
        elif f"camera_{requested}" in self.faults:
            why = f"primary camera {requested!r} stream is unreadable"
        else:
            why = f"primary camera {requested!r} has no latency stream"
        self.status["reasons"].append(
            f"{why}; using {chosen!r}" + (" (its stream marks it primary)" if flagged else ""))
        return chosen

    def _fit_wall(self, raw: dict) -> None:
        pairs = [stream[CLOCK_PAIRS] for stream in raw.values()
                 if isinstance(stream, dict) and CLOCK_PAIRS in stream]
        self.wall_map = clocks.WallMap(())
        reason = ""
        if pairs:
            try:
                joined = numpy.concatenate(pairs)
                self.wall_map = clocks.fit_wall_to_host(joined["wall"], joined["perf"])
            except Exception as error:
                logger.exception("Latency wall-clock fit failed")
                reason = _describe(error)
        self.status["clocks"]["wallToHost"] = _wall_record(self.wall_map, reason)

    def _fit_clocks(self) -> None:
        record = self.status["clocks"]
        ni = self.nidaq
        if ni is None or not math.isfinite(ni["rate"]):
            record["niToHost"] = self.ni_to_host.as_record()
            record["cameraToNi"] = {"valid": False, "reason": "no NI stream"}
            return
        ends, seen = ni["block_ends"]
        self.ni_to_host = _guarded(
            "NI-to-host", lambda: clocks.fit_ni_to_host(ends, seen, ni["rate"]))
        record["niToHost"] = {
            **self.ni_to_host.as_record(),
            "note": "mapped NI times are late by the unmeasured minimum delivery delay",
        }
        transition_index = ni["transitions"]
        frames = self.primary_frames
        if (transition_index is None or frames is None or len(frames) == 0
                or not self.ni_to_host.valid):
            record["cameraToNi"] = {
                "valid": False,
                "reason": ni["problems"].get("cam_frames")
                or "needs a cam_frames channel, primary frames and a valid NI mapping",
            }
            return
        # Older sessions, and cameras with no clock to latch, have no latches.
        latches = self.cameras.get(self.primary_name, {}).get("clock_latches")
        try:
            transition_perf = self.ni_to_host.map(transition_index / ni["rate"])
            if clocks.usable_latches(latches)[0].size:
                self.pairing = clocks.choose_pairing_by_camera_clock(
                    frames["frame_id"], frames["camera_ts_ns"], latches, transition_perf,
                    self.frame_period)
            else:
                self.pairing = clocks.choose_pairing(frames["frame_id"], frames["arrival_perf"],
                                                     transition_perf, self.frame_period)
        except Exception as error:
            logger.exception("Latency camera pairing failed")
            self.camera_to_ni = clocks.invalid_fit(_describe(error))
            record["cameraToNi"] = self.camera_to_ni.as_record()
            return
        pairing = self.pairing
        self.camera_to_ni = _guarded("camera-to-NI", lambda: clocks.fit_camera_to_ni(
            frames["frame_id"], frames["camera_ts_ns"], transition_index, ni["rate"],
            pairing, self.frame_period))
        record["cameraToNi"] = {**self.camera_to_ni.as_record(),
                                "pairing": dataclasses.asdict(self.pairing)}
        if pairing.method == clocks.HOST_ARRIVAL:
            # Only arrival placed each exposure, so a lag over one period would
            # read whole periods short; the next candidate is kept beside it.
            record["cameraToNi"]["assumes_lag_below_period"] = True
            record["cameraToNi"]["alternative_median_lag"] = pairing.median_lag + self.frame_period
            self.status["reasons"].append(_host_arrival_reason(latches))
        else:
            # Exposure to arrival from the latches alone, beside the NI path's
            # figure over the same frames. Not two measurements: their
            # difference is the pairing residual again (ni - latch is about
            # -median_residual), seen through the fitted camera-to-NI line, so
            # this is a consistency check on fit_camera_to_ni. The residual
            # gate in choose_pairing_by_camera_clock is the safeguard against
            # whole-period mispairing and must not be relaxed on the strength
            # of this check. Both figures stay in the record as diagnostics.
            latch_lag = _latch_direct_lag_p50(frames, latches)
            ni_lag = _ni_derived_lag_p50(frames, latches, self.camera_to_ni, self.ni_to_host)
            record["cameraToNi"]["latchDirectLagP50"] = latch_lag
            record["cameraToNi"]["niDerivedLagP50"] = ni_lag
            tolerance = clocks.latch_tolerance(self.frame_period)
            if not pairing.ambiguous and abs(ni_lag - latch_lag) > tolerance:
                self.pairing = dataclasses.replace(pairing, ambiguous=True, reason=(
                    f"NI-derived exposure-to-arrival p50 {ni_lag * 1e3:.3f} ms and latch-direct "
                    f"p50 {latch_lag * 1e3:.3f} ms differ by more than {tolerance * 1e3:.3f} ms"))
                record["cameraToNi"]["pairing"] = dataclasses.asdict(self.pairing)
        if self.pairing.ambiguous:
            record["cameraToNi"]["valid"] = False
            record["cameraToNi"]["reason"] = self.pairing.reason

    def exposure_of(self):
        usable = (self.camera_to_ni.valid and self.ni_to_host.valid
                  and self.pairing is not None and not self.pairing.ambiguous)
        if not usable:
            return None
        frames = self.primary_frames

        def exposure(frame_ids):
            camera_ts = analysis.take_by_key(frames["frame_id"],
                                             frames["camera_ts_ns"].astype(numpy.float64), frame_ids)
            return self.ni_to_host.map(self.camera_to_ni.map(camera_ts / 1e9))

        return exposure

    def edges(self, suffix: str):
        """Rising edges (NI seconds) of the channels ending in ``suffix``, and why others have none."""
        ni = self.nidaq
        if ni is None:
            return {}, {}
        return ({name: found for name, found in ni["edges"].items() if name.endswith(suffix)},
                {name: why for name, why in ni["problems"].items() if name.endswith(suffix)})


def _sorted_union(edges: dict) -> numpy.ndarray:
    if not edges:
        return numpy.empty(0)
    return numpy.sort(numpy.concatenate(list(edges.values())))


def _blame(result: dict, ctx: _Context, *keys: str) -> dict:
    """An absent loop whose input could not be read says so, not 'no rows'."""
    if result.get("status") != "absent":
        return result
    broken = [f"{key}.h5 unreadable: {ctx.faults[key]}" for key in keys if key in ctx.faults]
    return {**result, "reason": "; ".join(broken)} if broken else result


def _pose(ctx: _Context) -> dict:
    if ctx.pose is None:
        return _blame({"status": "absent", "reason": "no pose stream"}, ctx, "pose")
    cameras = json.loads(_text(ctx.pose["attrs"].get("cameras", "[]")))
    return analysis.pose_loop(
        batches=ctx.pose.get("batches"), forwards=ctx.pose.get("forwards"),
        live_poses=ctx.events.get("live_poses"), gate=ctx.events.get("gate_observations"),
        primary_frames=ctx.primary_frames,
        camera_frames=[ctx.cameras.get(name, {}).get("frames") for name in cameras],
        exposure_of=ctx.exposure_of(), frame_period=ctx.frame_period,
    )


def _recording(ctx: _Context) -> dict:
    result = analysis.recording_loop({
        name: (stream.get("record_batches"), ctx.records.get(name, {}).get("record_writes"))
        for name, stream in ctx.cameras.items()
    })
    return _blame(result, ctx, *sorted(
        key for key in ctx.faults if key.startswith(("camera_", "record_"))))


def _stim(ctx: _Context) -> dict:
    command, _ = ctx.edges("_command_copy")
    trigger, _ = ctx.edges("_trigger")
    return _blame(analysis.stim_loop(
        dispatch=ctx.events.get("stim_dispatch"), stim3_pulses=ctx.events.get("stim3_pulses"),
        can_events=ctx.events.get("can_events"),
        command_edges=_sorted_union(command), trigger_edges=_sorted_union(trigger),
        ni_to_host=ctx.ni_to_host if ctx.ni_to_host.valid else None,
    ), ctx, "events")


def _can(ctx: _Context) -> dict:
    return _blame(analysis.can_loop(can_events=ctx.events.get("can_events"),
                                    wall_map=ctx.wall_map), ctx, "events")


def _hardware(ctx: _Context) -> dict:
    command, command_problems = ctx.edges("_command_copy")
    diode, diode_problems = ctx.edges("_diode")
    trigger, trigger_problems = ctx.edges("_trigger")
    return _blame(analysis.hardware_loop(
        command_edges=command, diode_edges=diode, trigger_edges=trigger,
        unusable={**command_problems, **diode_problems, **trigger_problems},
    ), ctx, "nidaq")


LOOPS = (("pose", _pose), ("recording", _recording), ("stim", _stim),
         ("can", _can), ("hardware", _hardware))


def _writer_problem(path: str, attrs: dict) -> str:
    """Why a raw stream's own writer says it lost rows, or '' when it reports none."""
    stats_text = attrs.get("writer_stats")
    if stats_text is None:
        return ""
    try:
        stats = json.loads(_text(stats_text))
        dropped = stats.get("rowsDropped") or 0
        dropped = (sum(int(count) for count in dropped.values()) if isinstance(dropped, dict)
                   else int(dropped))
        rejected = int(stats.get("rowsRejected") or 0)
        failed = bool(stats.get("failed"))
        first_error = stats.get("firstError")
    except Exception as error:
        return f"{path}: writer_stats could not be read ({_describe(error)})"
    if not (dropped or rejected or failed):
        return ""
    detail = f"rowsDropped={dropped}, rowsRejected={rejected}, failed={failed}"
    return f"{path}: writer lost rows: {detail}" + (f" ({first_error})" if first_error else "")


def _overall(results: dict, reasons: list, unreadable: list) -> str:
    states = {result.get("status") for result in results.values()}
    produced = any(
        result.get("status") in ("complete", "partial") and len(result.get("stages", ())) > 0
        for result in results.values())
    if not produced:
        # Nothing was measured. That is only "absent" when nothing went wrong:
        # a loop that failed, or a stream that could not be read, is a failure.
        return "failed" if ("failed" in states or unreadable) else "absent"
    complete = states <= {"complete", "absent"} and not reasons
    return "complete" if complete else "partial"


def _flat_attrs(prefix: str, record: dict):
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _flat_attrs(f"{name}_", value)
        elif isinstance(value, (list, tuple)):
            yield name, json.dumps(value, sort_keys=True)
        elif value is not None:  # an HDF5 attribute cannot hold None; absent means unknown
            yield name, value


def _write_output(path: Path, results: dict, status: dict, *, session_id: str,
                  app_version: str) -> None:
    import h5py
    staged = path.with_name(path.name + ".partial")
    try:
        with h5py.File(staged, "w") as store:
            store.attrs["schema_version"] = LATENCY_SCHEMA_VERSION
            store.attrs["session_id"] = session_id
            store.attrs["app_version"] = app_version
            store.attrs["status"] = json.dumps(_json_safe(status), sort_keys=True)
            clock = store.create_group("clock")
            for key, name in CLOCK_GROUPS:
                group = clock.create_group(name)
                for attr, value in _flat_attrs("", _json_safe(status["clocks"].get(key, {}))):
                    group.attrs[attr] = value
                if key == "niToHost":
                    group.attrs["envelope_bias_floor"] = ENVELOPE_BIAS_FLOOR
                    group.attrs["secondary_trigger_delay"] = SECONDARY_TRIGGER_DELAY
            for loop, result in results.items():
                group = store.create_group(loop)
                group.attrs["status"] = result.get("status", "failed")
                group.attrs["reason"] = result.get("reason", "")
                group.attrs["summary"] = json.dumps(_json_safe(result.get("summary", {})),
                                                    sort_keys=True)
                for key in ("rows", "stages"):
                    data = result.get(key)
                    if data is not None:
                        options = {"compression": "lzf"} if len(data) else {}
                        group.create_dataset(key, data=data, **options)
        staged.replace(path)
    except BaseException:
        # A half-written file left in streams/ would look like output.
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_streams(session_dir: Path, raw_dir: Path, status: dict) -> tuple:
    """Every raw stream that opens, and a reason for each that does not."""
    raw, faults = {}, {}
    for path in sorted(raw_dir.glob("*.h5")):
        try:
            raw[path.stem] = _read_h5(path)
        except Exception as error:
            relative = path.relative_to(session_dir).as_posix()
            logger.warning("Latency stream %s could not be read: %s", relative, _describe(error))
            faults[path.stem] = _describe(error)
            status["unreadable"].append({"path": relative, "reason": _describe(error)})
            status["reasons"].append(f"{relative} unreadable: {_describe(error)}")
    return raw, faults


def _finalize(session_dir, primary_camera: str, output_name: str, session_id: str,
              status: dict) -> None:
    session_dir = Path(session_dir)
    streams = session_dir / "streams"
    raw_dir = streams / "latency"
    if not raw_dir.is_dir() or not any(raw_dir.glob("*.h5")):
        status["status"] = "absent"
        status["reasons"].append("no raw latency streams in this session")
        return
    raw, faults = _read_streams(session_dir, raw_dir, status)
    context = _Context(raw, faults, streams, primary_camera, status)
    for stem, stream in raw.items():
        problem = _writer_problem(f"streams/latency/{stem}.h5", stream.get("attrs", {}))
        if problem:
            status["reasons"].append(problem)
    results = {}
    for loop, build in LOOPS:
        try:
            results[loop] = build(context)
        except Exception as error:
            logger.exception("Latency loop %s failed", loop)
            results[loop] = {"status": "failed", "reason": _describe(error)}
        status["loops"][loop] = {key: results[loop].get(key)
                                 for key in ("status", "reason", "summary")}
    status["status"] = _overall(results, status["reasons"], status["unreadable"])
    status["output"] = output_name
    try:
        _write_output(streams / output_name, results, status, session_id=session_id,
                      app_version=_app_version())
    except Exception as error:
        logger.exception("%s was not written", output_name)
        status["output"] = None
        status["status"] = "failed"
        status["reasons"].append(f"{output_name} not written: {_describe(error)}")


def finalize_session_latency(session_dir, *, primary_camera: str,
                             output_name: str = LATENCY_OUTPUT_NAME,
                             session_id: str = "") -> dict:
    """Analyse one session's latency streams; never raises."""
    status = {"status": "failed", "output": None, "loops": {}, "clocks": {}, "reasons": [],
              "unreadable": []}
    try:
        _finalize(session_dir, primary_camera, output_name, session_id, status)
    except Exception as error:
        logger.exception("Latency finalization failed")
        status["status"] = "failed"
        status["reasons"].append(_describe(error))
    try:
        return _json_safe(status)
    except Exception as error:
        return {"status": "failed", "output": None, "loops": {}, "clocks": {},
                "reasons": [f"status could not be serialised: {_describe(error)}"],
                "unreadable": []}
