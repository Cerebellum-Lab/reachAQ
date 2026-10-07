"""Build streams/latency.h5 from a session's raw latency streams.

Runs inside session finalization and from the rebuild CLI. It never raises:
each loop is analysed on its own and its status recorded, so a failure in one
leaves the others intact, and a session always publishes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy

from autotrainer.core.latency.schema import CLOCK_PAIRS, LATENCY_SCHEMA_VERSION

from . import analysis, clocks

logger = logging.getLogger(__name__)

LATENCY_OUTPUT_NAME = "latency.h5"
OPEN_RETRY_SECONDS = 5.0
COMMAND_MIN_SWING = 0.1   # volts
DIODE_MIN_SWING = 0.05    # volts
TRIGGER_MIN_SWING = 0.5


def _text(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, numpy.bool_)):
        return bool(value)
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


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
    import h5py
    with h5py.File(path, "r") as store:
        names = [_text(name) for name in store.attrs.get("channel_names", [])]
        rate = float(store.attrs.get("sample_rate_hz", math.nan))
        index = store["sample_index"][()]
        seen = store["block_observation_perf_time"][()]
        channels = {}
        for row, name in enumerate(names):
            if name == "cam_frames" or name.endswith(("_command_copy", "_diode", "_trigger")):
                channels[name] = store["values"][row, :]
    return {"names": names, "rate": rate, "index": index, "seen": seen, "channels": channels}


def _frame_period(frames) -> float:
    if frames is None or len(frames) < 2:
        return math.nan
    ids = numpy.asarray(frames["frame_id"], dtype=numpy.int64)
    ts = numpy.asarray(frames["camera_ts_ns"], dtype=numpy.float64) / 1e9
    consecutive = numpy.diff(ids) == 1
    if not consecutive.any():
        return math.nan
    return float(numpy.median(numpy.diff(ts)[consecutive]))


class _Context:
    """Everything the loops share: the raw streams, NI data and the clock fits."""

    def __init__(self, raw: dict, streams: Path, primary_camera: str, status: dict):
        self.status = status
        self.events = raw.get("events") or {}
        self.pose = raw.get("pose")
        self.cameras = {key[len("camera_"):]: value for key, value in raw.items()
                        if key.startswith("camera_")}
        self.records = {key[len("record_"):]: value for key, value in raw.items()
                        if key.startswith("record_")}
        self.primary_name = (primary_camera if primary_camera in self.cameras
                             else next(iter(sorted(self.cameras)), ""))
        if self.cameras and self.primary_name != primary_camera:
            status["reasons"].append(
                f"primary camera {primary_camera!r} has no latency stream; using {self.primary_name!r}")
        self.primary_frames = self.cameras.get(self.primary_name, {}).get("frames")
        self.frame_period = _frame_period(self.primary_frames)
        self.nidaq: Optional[dict] = None
        nidaq_path = streams / "nidaq.h5"
        if nidaq_path.is_file():
            try:
                self.nidaq = _read_nidaq(nidaq_path)
            except Exception as error:
                status["reasons"].append(f"nidaq.h5 unreadable: {type(error).__name__}: {error}")
        self.ni_to_host = clocks.invalid_fit("no NI stream")
        self.camera_to_ni = clocks.invalid_fit("not fitted")
        self.pairing: Optional[clocks.Pairing] = None
        self._edges: Dict[Tuple[str, float], tuple] = {}
        self._fit_clocks()
        pairs = [stream[CLOCK_PAIRS] for stream in raw.values()
                 if isinstance(stream, dict) and CLOCK_PAIRS in stream]
        if pairs:
            joined = numpy.concatenate(pairs)
            self.wall_map = clocks.fit_wall_to_host(joined["wall"], joined["perf"])
        else:
            self.wall_map = clocks.WallMap(())

    def _fit_clocks(self) -> None:
        record = self.status["clocks"]
        ni = self.nidaq
        if ni is None or not math.isfinite(ni["rate"]):
            record["niToHost"] = self.ni_to_host.as_record()
            record["cameraToNi"] = {"valid": False, "reason": "no NI stream"}
            return
        ends, seen = clocks.ni_block_ends(ni["index"], ni["seen"])
        self.ni_to_host = clocks.fit_ni_to_host(ends, seen, ni["rate"])
        record["niToHost"] = {
            **self.ni_to_host.as_record(),
            "note": "mapped NI times are late by the unmeasured minimum delivery delay",
        }
        cam = ni["channels"].get("cam_frames")
        frames = self.primary_frames
        if cam is None or frames is None or len(frames) == 0 or not self.ni_to_host.valid:
            record["cameraToNi"] = {
                "valid": False,
                "reason": "needs a cam_frames channel, primary frames and a valid NI mapping",
            }
            return
        transition_index = ni["index"][clocks.transitions(cam)]
        transition_perf = self.ni_to_host.map(transition_index / ni["rate"])
        self.pairing = clocks.choose_pairing(frames["frame_id"], frames["arrival_perf"],
                                             transition_perf, self.frame_period)
        self.camera_to_ni = clocks.fit_camera_to_ni(
            frames["frame_id"], frames["camera_ts_ns"], transition_index, ni["rate"],
            self.pairing, self.frame_period)
        record["cameraToNi"] = {**self.camera_to_ni.as_record(),
                                "pairing": dataclasses.asdict(self.pairing)}
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

    def edges(self, suffix: str, min_swing: float):
        """Rising edges, in NI seconds, of every channel ending in ``suffix``."""
        key = (suffix, min_swing)
        if key not in self._edges:
            found, problems = {}, {}
            ni = self.nidaq
            if ni is not None and math.isfinite(ni["rate"]):
                for name, values in ni["channels"].items():
                    if not name.endswith(suffix):
                        continue
                    result = clocks.rising_edges(values, min_swing=min_swing)
                    if result.usable:
                        found[name] = ni["index"][result.positions] / ni["rate"]
                    else:
                        problems[name] = result.reason
            self._edges[key] = (found, problems)
        return self._edges[key]


def _sorted_union(edges: dict) -> numpy.ndarray:
    if not edges:
        return numpy.empty(0)
    return numpy.sort(numpy.concatenate(list(edges.values())))


def _pose(ctx: _Context) -> dict:
    if ctx.pose is None:
        return {"status": "absent", "reason": "no pose stream"}
    cameras = json.loads(_text(ctx.pose["attrs"].get("cameras", "[]")))
    return analysis.pose_loop(
        batches=ctx.pose.get("batches"), forwards=ctx.pose.get("forwards"),
        live_poses=ctx.events.get("live_poses"), gate=ctx.events.get("gate_observations"),
        primary_frames=ctx.primary_frames,
        camera_frames=[ctx.cameras.get(name, {}).get("frames") for name in cameras],
        exposure_of=ctx.exposure_of(), frame_period=ctx.frame_period,
    )


def _recording(ctx: _Context) -> dict:
    return analysis.recording_loop({
        name: (stream.get("record_batches"), ctx.records.get(name, {}).get("record_writes"))
        for name, stream in ctx.cameras.items()
    })


def _stim(ctx: _Context) -> dict:
    command, _ = ctx.edges("_command_copy", COMMAND_MIN_SWING)
    trigger, _ = ctx.edges("_trigger", TRIGGER_MIN_SWING)
    return analysis.stim_loop(
        dispatch=ctx.events.get("stim_dispatch"), stim3_pulses=ctx.events.get("stim3_pulses"),
        can_events=ctx.events.get("can_events"),
        command_edges=_sorted_union(command), trigger_edges=_sorted_union(trigger),
        ni_to_host=ctx.ni_to_host if ctx.ni_to_host.valid else None,
    )


def _can(ctx: _Context) -> dict:
    return analysis.can_loop(can_events=ctx.events.get("can_events"), wall_map=ctx.wall_map)


def _hardware(ctx: _Context) -> dict:
    command, command_problems = ctx.edges("_command_copy", COMMAND_MIN_SWING)
    diode, diode_problems = ctx.edges("_diode", DIODE_MIN_SWING)
    trigger, trigger_problems = ctx.edges("_trigger", TRIGGER_MIN_SWING)
    return analysis.hardware_loop(
        command_edges=command, diode_edges=diode, trigger_edges=trigger,
        unusable={**command_problems, **diode_problems, **trigger_problems},
    )


LOOPS = (("pose", _pose), ("recording", _recording), ("stim", _stim),
         ("can", _can), ("hardware", _hardware))


def _write_output(path: Path, results: dict, status: dict) -> None:
    import h5py
    staged = path.with_name(path.name + ".partial")
    with h5py.File(staged, "w") as store:
        store.attrs["schema_version"] = LATENCY_SCHEMA_VERSION
        store.attrs["status"] = json.dumps(_json_safe(status), sort_keys=True)
        for loop, result in results.items():
            group = store.create_group(loop)
            group.attrs["status"] = result.get("status", "failed")
            group.attrs["reason"] = result.get("reason", "")
            group.attrs["summary"] = json.dumps(_json_safe(result.get("summary", {})), sort_keys=True)
            for key in ("rows", "stages"):
                data = result.get(key)
                if data is not None:
                    options = {"compression": "lzf"} if len(data) else {}
                    group.create_dataset(key, data=data, **options)
    staged.replace(path)


def finalize_session_latency(session_dir, *, primary_camera: str,
                             output_name: str = LATENCY_OUTPUT_NAME) -> dict:
    """Analyse one session's latency streams; never raises."""
    session_dir = Path(session_dir)
    streams = session_dir / "streams"
    raw_dir = streams / "latency"
    status = {"status": "failed", "output": None, "loops": {}, "clocks": {}, "reasons": []}
    if not raw_dir.is_dir() or not any(raw_dir.glob("*.h5")):
        status["status"] = "absent"
        status["reasons"].append("no raw latency streams in this session")
        return _json_safe(status)
    try:
        raw = {path.stem: _read_h5(path) for path in sorted(raw_dir.glob("*.h5"))}
        context = _Context(raw, streams, primary_camera, status)
    except Exception as error:
        logger.exception("Latency streams could not be read")
        status["reasons"].append(f"raw streams unreadable: {type(error).__name__}: {error}")
        return _json_safe(status)
    results = {}
    for loop, build in LOOPS:
        try:
            results[loop] = build(context)
        except Exception as error:
            logger.exception("Latency loop %s failed", loop)
            results[loop] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
        status["loops"][loop] = {key: results[loop].get(key)
                                 for key in ("status", "reason", "summary")}
    states = {entry["status"] for entry in status["loops"].values()}
    status["status"] = "complete" if states <= {"complete", "absent"} else "partial"
    try:
        _write_output(streams / output_name, results, status)
        status["output"] = output_name
    except Exception as error:
        logger.exception("%s was not written", output_name)
        status["status"] = "failed"
        status["reasons"].append(f"{output_name} not written: {type(error).__name__}: {error}")
    return _json_safe(status)
