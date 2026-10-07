"""Joins and stage summaries for the latency record.

Pure functions over the arrays the raw streams hold, so every stage can be
checked against synthetic data with known delays. Each stage carries the
confidence of its weaker end (spec §7):

    hardware  both ends are NI edges
    mixed     one NI edge and one host stamp, through the NI-to-host fit
    software  host stamps only; a "sensor" stage then starts at host arrival
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy

HARDWARE = "hardware"
MIXED = "mixed"
SOFTWARE = "software"
NAN = math.nan

STAGE_DTYPE = numpy.dtype([
    ("stage", "S48"),
    ("confidence", "S12"),
    ("n", "<i8"),
    ("p50", "<f8"),
    ("p95", "<f8"),
    ("p99", "<f8"),
    ("max", "<f8"),
    ("deadline", "<f8"),
    ("miss_rate", "<f8"),
])

POSE_ROW_DTYPE = numpy.dtype([
    ("pose_seq", "<i8"),
    ("frame_id", "<i8"),
    ("pairing_skew", "<i8"),
    ("exposure", "<f8"),
    ("arrival", "<f8"),
    ("put", "<f8"),
    ("dequeue", "<f8"),
    ("predict_start", "<f8"),
    ("predict_done", "<f8"),
    ("monitor_recv", "<f8"),
    ("live_put", "<f8"),
    ("live_recv", "<f8"),
    ("triangulated", "<f8"),
    ("gui_recv", "<f8"),
])


def summarize(stage: str, values, confidence: str, *, deadline: float = NAN) -> tuple:
    values = numpy.asarray(values, dtype=numpy.float64)
    values = values[numpy.isfinite(values)]
    if values.size == 0:
        return (stage.encode(), confidence.encode(), 0, NAN, NAN, NAN, NAN, float(deadline), NAN)
    p50, p95, p99 = numpy.percentile(values, [50, 95, 99])
    miss = float(numpy.mean(values > deadline)) if math.isfinite(deadline) else NAN
    return (stage.encode(), confidence.encode(), int(values.size), float(p50), float(p95),
            float(p99), float(values.max()), float(deadline), miss)


def stage_table(rows: Iterable[tuple]) -> numpy.ndarray:
    return numpy.array(list(rows), dtype=STAGE_DTYPE)


def take_by_key(source_keys, source_values, keys, fill: float = NAN) -> numpy.ndarray:
    """source_values at each key's position in source_keys; ``fill`` where absent."""
    keys = numpy.asarray(keys)
    out = numpy.full(keys.shape, fill, dtype=numpy.float64)
    source_keys = numpy.asarray(source_keys)
    if source_keys.size == 0 or keys.size == 0:
        return out
    order = numpy.argsort(source_keys, kind="stable")
    sorted_keys = source_keys[order]
    position = numpy.clip(numpy.searchsorted(sorted_keys, keys), 0, sorted_keys.size - 1)
    found = sorted_keys[position] == keys
    values = numpy.asarray(source_values, dtype=numpy.float64)[order]
    out[found] = values[position[found]]
    return out


def first_after(times, events, *, within: float) -> numpy.ndarray:
    """For each time, the first event at or after it within ``within`` seconds."""
    times = numpy.asarray(times, dtype=numpy.float64)
    out = numpy.full(times.shape, NAN)
    events = numpy.sort(numpy.asarray(events, dtype=numpy.float64))
    events = events[numpy.isfinite(events)]
    if events.size == 0 or times.size == 0:
        return out
    position = numpy.searchsorted(events, times, side="left")
    candidate = events[numpy.clip(position, 0, events.size - 1)]
    inside = (position < events.size) & numpy.isfinite(times) & ((candidate - times) <= within)
    out[inside] = candidate[inside]
    return out


def last_before(times, events, *, within: float) -> numpy.ndarray:
    """For each time, the last event at or before it within ``within`` seconds."""
    times = numpy.asarray(times, dtype=numpy.float64)
    out = numpy.full(times.shape, NAN)
    events = numpy.sort(numpy.asarray(events, dtype=numpy.float64))
    events = events[numpy.isfinite(events)]
    if events.size == 0 or times.size == 0:
        return out
    position = numpy.searchsorted(events, times, side="right") - 1
    candidate = events[numpy.clip(position, 0, events.size - 1)]
    inside = (position >= 0) & numpy.isfinite(times) & ((times - candidate) <= within)
    out[inside] = candidate[inside]
    return out


def _empty(table) -> bool:
    return table is None or len(table) == 0


def _row_min(values) -> numpy.ndarray:
    values = numpy.asarray(values, dtype=numpy.float64)
    lowest = numpy.where(numpy.isfinite(values), values, numpy.inf).min(axis=1)
    return numpy.where(numpy.isfinite(lowest), lowest, NAN)


def _result(stages, reasons, summary, **extra) -> dict:
    return {
        "status": "partial" if reasons else "complete",
        "reason": "; ".join(reasons),
        "stages": stage_table(stages),
        "summary": summary,
        **extra,
    }


def _coverage(posed_ids, primary_frames) -> float:
    posed = numpy.unique(posed_ids[posed_ids >= 0])
    if posed.size == 0 or _empty(primary_frames):
        return NAN
    acquired = numpy.asarray(primary_frames["frame_id"])
    acquired = numpy.unique(acquired[(acquired >= posed[0]) & (acquired <= posed[-1])])
    return float(posed.size / max(1, acquired.size))


def pose_loop(*, batches, forwards, live_poses, gate, primary_frames,
              camera_frames: Sequence, exposure_of: Optional[Callable],
              frame_period: float) -> dict:
    if _empty(batches):
        return {"status": "absent", "reason": "no live pose batches were recorded"}
    reasons = []
    ids = numpy.asarray(batches["frame_ids"], dtype=numpy.int64)
    if ids.ndim == 1:
        ids = ids[:, None]
    count, cameras = ids.shape
    arrival = numpy.full((count, cameras), NAN)
    for index in range(cameras):
        frames = camera_frames[index] if index < len(camera_frames) else None
        if _empty(frames):
            reasons.append(f"camera {index} has no frame stream")
            continue
        arrival[:, index] = take_by_key(frames["frame_id"], frames["arrival_perf"], ids[:, index])
    if exposure_of is not None:
        # The secondary is triggered by the primary's exposure, so one exposure
        # time serves both frames of a pair.
        exposure = exposure_of(ids[:, 0])
        confidence = MIXED
    else:
        exposure = _row_min(arrival)
        confidence = SOFTWARE
        reasons.append("no valid camera-to-NI mapping: 'sensor' stages start at host arrival")
    put = numpy.asarray(batches["put_perf"], dtype=numpy.float64).reshape(count, cameras)
    sequence = numpy.asarray(batches["pose_seq"])
    if _empty(forwards):
        live_sequence = numpy.full(count, -1.0)
        live_put = numpy.full(count, NAN)
        reasons.append("no batch was forwarded to live 3D")
    else:
        live_sequence = take_by_key(forwards["pose_seq"], forwards["live_sequence"], sequence,
                                    fill=-1.0)
        live_put = take_by_key(forwards["pose_seq"], forwards["live_put_perf"], sequence)
    if _empty(live_poses):
        live_recv = triangulated = gui_recv = numpy.full(count, NAN)
        if not _empty(forwards):
            reasons.append("no live pose reached the GUI")
    else:
        live_recv = take_by_key(live_poses["live_sequence"], live_poses["live_recv_perf"], live_sequence)
        triangulated = take_by_key(live_poses["live_sequence"], live_poses["triangulated_perf"], live_sequence)
        gui_recv = take_by_key(live_poses["live_sequence"], live_poses["gui_recv_perf"], live_sequence)
    paired = (ids >= 0) & (ids[:, :1] >= 0)
    skew = numpy.abs(numpy.where(paired, ids - ids[:, :1], 0)).max(axis=1)

    rows = numpy.zeros(count, dtype=POSE_ROW_DTYPE)
    rows["pose_seq"] = sequence
    rows["frame_id"] = ids[:, 0]
    rows["pairing_skew"] = skew
    rows["exposure"] = exposure
    rows["arrival"] = arrival[:, 0]
    rows["put"] = _row_min(put)
    rows["dequeue"] = batches["dequeue_perf"]
    rows["predict_start"] = batches["predict_start_perf"]
    rows["predict_done"] = batches["predict_done_perf"]
    rows["monitor_recv"] = batches["monitor_recv_perf"]
    rows["live_put"] = live_put
    rows["live_recv"] = live_recv
    rows["triangulated"] = triangulated
    rows["gui_recv"] = gui_recv

    done = rows["predict_done"]
    stages = []
    # Exposure-to-arrival is only meaningful when we have a camera-to-NI mapping; when confidence
    # is software (fallback to host arrival), it would measure inter-camera arrival skew, not exposure time.
    if confidence != SOFTWARE:
        stages.append(summarize("exposure_to_arrival", rows["arrival"] - exposure, confidence))
    stages += [
        summarize("arrival_to_queue", put[:, 0] - rows["arrival"], SOFTWARE),
        summarize("queue_wait", rows["dequeue"] - rows["put"], SOFTWARE),
        summarize("predict", done - rows["predict_start"], SOFTWARE),
        summarize("predict_to_monitor", rows["monitor_recv"] - done, SOFTWARE),
        summarize("monitor_to_live", live_put - rows["monitor_recv"], SOFTWARE),
        summarize("live_queue", live_recv - live_put, SOFTWARE),
        summarize("triangulate", triangulated - live_recv, SOFTWARE),
        summarize("to_gui", gui_recv - triangulated, SOFTWARE),
        summarize("sensor_to_pose", done - exposure, confidence, deadline=frame_period),
        summarize("sensor_to_3d", triangulated - exposure, confidence),
        summarize("sensor_to_gui", gui_recv - exposure, confidence),
    ]
    if not _empty(gate):
        forwarded = live_sequence >= 0
        base = take_by_key(live_sequence[forwarded], exposure[forwarded], gate["live_sequence"])
        stages.append(summarize("gate_pose_age", gate["observe_perf"] - base, confidence))
    summary = {
        "batches": int(count),
        "skippedBatches": int(numpy.sum(batches["skipped_before"])),
        "pairingSkewBatches": int(numpy.count_nonzero(skew)),
        "confidence": confidence,
        "coverage": _coverage(ids[:, 0], primary_frames),
    }
    return _result(stages, reasons, summary, rows=rows)


def recording_loop(per_camera: Mapping[str, Tuple[Optional[numpy.ndarray], Optional[numpy.ndarray]]]) -> dict:
    stages, reasons, summary = [], [], {}
    for name, (batches, writes) in sorted(per_camera.items()):
        if _empty(batches):
            continue
        keys = batches["first_frame_id"]
        if _empty(writes):
            reasons.append(f"{name}: no recorder rows")
            dequeue = write_entry = write_return = numpy.full(len(batches), NAN)
        else:
            dequeue = take_by_key(writes["first_frame_id"], writes["dequeue_perf"], keys)
            write_entry = take_by_key(writes["first_frame_id"], writes["write_entry_perf"], keys)
            write_return = take_by_key(writes["first_frame_id"], writes["write_return_perf"], keys)
        # Lost rows are retried failed puts; do not join them to the retried batch's dequeue/write.
        lost_mask = batches["lost"]
        dequeue = numpy.where(lost_mask, NAN, dequeue)
        write_entry = numpy.where(lost_mask, NAN, write_entry)
        write_return = numpy.where(lost_mask, NAN, write_return)
        stages += [
            summarize(f"{name}.put_block", batches["put_return_perf"] - batches["put_entry_perf"], SOFTWARE),
            summarize(f"{name}.recorder_queue", dequeue - batches["put_return_perf"], SOFTWARE),
            summarize(f"{name}.write", write_return - write_entry, SOFTWARE),
        ]
        # Unrecovered batches: lost rows whose first frame id does not appear in a later lost=False row.
        failed_puts = int(numpy.count_nonzero(lost_mask))
        lost_ids = set(batches[lost_mask]["first_frame_id"].tolist())
        recovered_ids = set(batches[~lost_mask]["first_frame_id"].tolist())
        unrecovered = len(lost_ids - recovered_ids)
        summary[name] = {
            "batches": int(len(batches)),
            "failedPuts": failed_puts,
            "unrecoveredBatches": unrecovered,
            "maxQueueDepth": int(numpy.max(batches["queue_depth"])),
        }
    if not summary:
        return {"status": "absent", "reason": "no recording batches"}
    return _result(stages, reasons, summary)


def stim_loop(*, dispatch, stim3_pulses, can_events, command_edges, trigger_edges,
              ni_to_host) -> dict:
    """Stim triggers on both routes. Edges arrive in NI seconds."""
    if _empty(dispatch):
        return {"status": "absent", "reason": "no stim triggers"}
    reasons = []
    if ni_to_host is None:
        reasons.append("no valid NI-to-host mapping: output edges are not placed")
        mapped_command = mapped_trigger = numpy.empty(0)
    else:
        mapped_command = ni_to_host.map(command_edges)
        mapped_trigger = ni_to_host.map(trigger_edges)
    route = dispatch["route"]
    stages = []

    direct = dispatch[route == b"direct_ni_software"]
    if len(direct):
        output = first_after(direct["start_entry_perf"], mapped_command, within=0.05)
        if ni_to_host is not None and not numpy.isfinite(output).any():
            reasons.append("no laser command edge followed any direct trigger")
        stages += [
            summarize("direct.detect", direct["decision_perf"] - direct["frame_perf"], SOFTWARE),
            summarize("direct.evidence", direct["evidence_done_perf"] - direct["decision_perf"], SOFTWARE),
            summarize("direct.clip", direct["clip_done_perf"] - direct["evidence_done_perf"], SOFTWARE),
            summarize("direct.ipc", direct["gui_recv_perf"] - direct["send_perf"], SOFTWARE),
            summarize("direct.validate", direct["validated_perf"] - direct["gui_recv_perf"], SOFTWARE),
            summarize("direct.daq_start", direct["start_return_perf"] - direct["start_entry_perf"], SOFTWARE),
            summarize("direct.start_to_output_edge", output - direct["start_entry_perf"], MIXED),
            summarize("direct.frame_to_output_edge", output - direct["frame_perf"], MIXED),
        ]

    stim3 = dispatch[route == b"hardware_stim3"]
    if len(stim3):
        operation = stim3["operation_id"]
        if _empty(stim3_pulses):
            reasons.append("STIM3 triggers without pulse rows")
            pulse_call = pulse_return = numpy.full(len(stim3), NAN)
            tokens = numpy.zeros(len(stim3), dtype="S36")
        else:
            pulse_call = take_by_key(stim3_pulses["operation_id"], stim3_pulses["pulse_call_perf"], operation)
            pulse_return = take_by_key(stim3_pulses["operation_id"], stim3_pulses["pulse_return_perf"], operation)
            token_of = dict(zip(stim3_pulses["operation_id"].tolist(), stim3_pulses["token"].tolist()))
            tokens = numpy.array([token_of.get(op, b"") for op in operation.tolist()], dtype="S36")
        if _empty(can_events):
            can_send = numpy.full(len(stim3), NAN)
        else:
            sends = can_events[can_events["stage"] == b"send"]
            can_send = take_by_key(sends["token"], sends["perf"], tokens)
        board_edge = first_after(can_send, mapped_trigger, within=0.05)
        output = first_after(board_edge, mapped_command, within=0.01)
        stages += [
            summarize("stim3.detect", stim3["decision_perf"] - stim3["frame_perf"], SOFTWARE),
            summarize("stim3.message", stim3["gui_recv_perf"] - stim3["send_perf"], SOFTWARE),
            summarize("stim3.to_pulse_call", pulse_call - stim3["gui_recv_perf"], SOFTWARE),
            summarize("stim3.pulse_call", pulse_return - pulse_call, SOFTWARE),
            summarize("stim3.to_can_send", can_send - pulse_return, SOFTWARE),
            summarize("stim3.can_send_to_board_edge", board_edge - can_send, MIXED),
            summarize("stim3.frame_to_output_edge", output - stim3["frame_perf"], MIXED),
        ]
    summary = {"directTriggers": int(len(direct)), "stim3Triggers": int(len(stim3))}
    return _result(stages, reasons, summary)


def can_loop(*, can_events, wall_map) -> dict:
    if _empty(can_events):
        return {"status": "absent", "reason": "no CAN commands"}
    stage = can_events["stage"]

    def rows(name):
        return can_events[stage == name]

    token_rows = rows(b"token")
    if len(token_rows) == 0:
        return {"status": "partial", "reason": "CAN events without token rows",
                "stages": stage_table([]), "summary": {}}
    tokens = token_rows["token"]
    token_perf = token_rows["perf"]
    enqueue = take_by_key(rows(b"enqueue")["token"], rows(b"enqueue")["perf"], tokens)
    dequeue = take_by_key(rows(b"dequeue")["token"], rows(b"dequeue")["perf"], tokens)
    sends = rows(b"send")
    send = take_by_key(sends["token"], sends["perf"], tokens)
    send_end = take_by_key(sends["token"], sends["perf_end"], tokens)
    uuid = take_by_key(sends["token"], sends["can_uuid"], tokens, fill=-1.0)
    acks = rows(b"ack")
    ack_perf = numpy.full(len(tokens), NAN)
    ack_wall = numpy.full(len(tokens), NAN)
    for index in range(len(tokens)):
        if uuid[index] < 0 or not math.isfinite(send[index]):
            continue
        # The 8-bit uuid wraps, so the ack is the first one with this uuid
        # after this send, and within two seconds of it.
        match = numpy.flatnonzero(
            (acks["can_uuid"] == uuid[index])
            & (acks["perf"] >= send[index])
            & (acks["perf"] <= send[index] + 2.0)
        )
        if match.size:
            first = match[numpy.argmin(acks["perf"][match])]
            ack_perf[index] = acks["perf"][first]
            ack_wall[index] = acks["kernel_wall"][first]
    trial_perf = numpy.full(len(tokens), NAN)
    trial = rows(b"trial_send")
    is_send = token_rows["kind"] == b"SEND_PELLET"
    if len(trial) and is_send.any():
        trial_perf[is_send] = last_before(token_perf[is_send], trial["perf"], within=1.0)
    kernel_perf = wall_map.map(ack_wall) if wall_map is not None else numpy.full(len(tokens), NAN)
    stages = [
        summarize("trial_to_token", token_perf - trial_perf, SOFTWARE),
        summarize("token_to_enqueue", enqueue - token_perf, SOFTWARE),
        summarize("command_queue", dequeue - enqueue, SOFTWARE),
        summarize("send_call", send_end - send, SOFTWARE),
        summarize("round_trip", ack_perf - send, SOFTWARE),
        summarize("ack_kernel_to_decode", ack_perf - kernel_perf, SOFTWARE),
    ]
    summary = {"commands": int(len(tokens)), "acked": int(numpy.isfinite(ack_perf).sum())}
    return _result(stages, [], summary)


def hardware_loop(*, command_edges: Mapping[str, numpy.ndarray],
                  diode_edges: Mapping[str, numpy.ndarray],
                  trigger_edges: Mapping[str, numpy.ndarray],
                  unusable: Mapping[str, str]) -> dict:
    """NI edge to NI edge, in NI seconds: exact to one sample, no host clock involved."""
    problems = [f"{name}: {reason}" for name, reason in sorted(unusable.items())]
    if not command_edges:
        return {"status": "absent", "reason": "; ".join(problems) or "no laser output edges",
                "stages": stage_table([]), "summary": {"unusable": dict(unusable)}}
    stages, summary = [], {"unusable": dict(unusable)}
    for name, command in sorted(command_edges.items()):
        laser = name[: -len("_command_copy")]
        diode = diode_edges.get(f"{laser}_diode")
        if diode is not None:
            stages.append(summarize(f"{laser}.command_to_diode",
                                    first_after(command, diode, within=0.02) - command, HARDWARE))
        trigger = trigger_edges.get(f"{laser}_trigger")
        if trigger is not None:
            stages.append(summarize(f"{laser}.trigger_to_command",
                                    first_after(trigger, command, within=0.01) - trigger, HARDWARE))
        summary[laser] = {"commandEdges": int(len(command))}
    return _result(stages, problems, summary)
