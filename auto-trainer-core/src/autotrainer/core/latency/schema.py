"""Row layouts for the per-process latency streams.

Every stamp is a ``time.perf_counter()`` value in seconds, stored as float64:
the same clock and unit the frame and stim stamps already carry, so a stamp
taken in one process compares directly with one taken in another. NaN means
the stage was not stamped; -1 marks a missing integer identifier.
"""

from __future__ import annotations

import numpy

LATENCY_SCHEMA_VERSION = 1

CLOCK_PAIRS = "clock_pairs"
"""Dataset every stream carries: (wall, perf) pairs, so the finalizer can map
wall-clock stamps (CAN kernel receive times) onto perf_counter."""

CLOCK_PAIR_DTYPE = numpy.dtype([("wall", "<f8"), ("perf", "<f8")])

# camera_<name>.h5 - the capture process, one row per frame received.
CAMERA_FRAME_DTYPE = numpy.dtype([
    ("frame_id", "<i8"),
    ("camera_ts_ns", "<i8"),
    ("poll_perf", "<f8"),
    ("arrival_perf", "<f8"),
    ("capture_return_perf", "<f8"),
    ("pose_put_result", "<i1"),
])
POSE_PUT_NOT_ATTEMPTED = -1
POSE_PUT_OK = 0
POSE_PUT_OVERFLOW = 1

# camera_<name>.h5 - the capture process, one row per batch put to the recorder.
RECORD_BATCH_DTYPE = numpy.dtype([
    ("first_frame_id", "<i8"),
    ("last_frame_id", "<i8"),
    ("frame_count", "<i4"),
    ("put_entry_perf", "<f8"),
    ("put_return_perf", "<f8"),
    ("queue_depth", "<i4"),
    ("lost", "?"),
])

# camera_<name>.h5 - the capture process, a few rows when the stream opens and
# a few when it closes: the camera's own clock latched between two host reads,
# so the finalizer can place each frame's camera timestamp on perf_counter.
CLOCK_LATCH_DTYPE = numpy.dtype([
    ("perf_before", "<f8"),
    ("camera_ns", "<i8"),
    ("perf_after", "<f8"),
])

# record_<name>.h5 - the recorder thread, one row per batch written.
RECORD_WRITE_DTYPE = numpy.dtype([
    ("first_frame_id", "<i8"),
    ("last_frame_id", "<i8"),
    ("frame_count", "<i4"),
    ("dequeue_perf", "<f8"),
    ("write_entry_perf", "<f8"),
    ("write_return_perf", "<f8"),
])


def pose_batch_dtype(camera_count: int) -> numpy.dtype:
    """pose.h5 ``batches`` - the inference monitor, one row per live batch."""
    return numpy.dtype([
        ("pose_seq", "<i8"),
        ("frame_ids", "<i8", (camera_count,)),
        ("put_perf", "<f8", (camera_count,)),
        ("dequeue_perf", "<f8"),
        ("skipped_before", "<i4"),
        ("predict_start_perf", "<f8"),
        ("predict_done_perf", "<f8"),
        ("data_put_perf", "<f8"),
        ("monitor_recv_perf", "<f8"),
    ])


# pose.h5 ``forwards`` - one row per batch handed to live 3D.
POSE_FORWARD_DTYPE = numpy.dtype([
    ("pose_seq", "<i8"),
    ("live_sequence", "<i8"),
    ("live_put_perf", "<f8"),
])

# events.h5 - the GUI process, kept in memory and written at stop.
LIVE_POSE_DTYPE = numpy.dtype([
    ("live_sequence", "<i8"),
    ("live_recv_perf", "<f8"),
    ("triangulated_perf", "<f8"),
    ("live_put_perf", "<f8"),
    ("gui_recv_perf", "<f8"),
])
GATE_OBSERVATION_DTYPE = numpy.dtype([
    ("observe_perf", "<f8"),
    ("live_sequence", "<i8"),
    ("triangulated_perf", "<f8"),
    ("reaching", "<i1"),  # 1, 0, or -1 when unknown
])
STIM_DISPATCH_DTYPE = numpy.dtype([
    ("operation_id", "S64"),
    ("route", "S24"),
    ("stim_frame_id", "<i8"),
    ("frame_perf", "<f8"),
    ("decision_perf", "<f8"),
    ("evidence_done_perf", "<f8"),
    ("clip_done_perf", "<f8"),
    ("send_perf", "<f8"),
    ("gui_recv_perf", "<f8"),
    ("validated_perf", "<f8"),
    ("start_entry_perf", "<f8"),
    ("start_return_perf", "<f8"),
    ("accepted", "<i1"),  # 1, 0, or -1 when the route does not report it
])
STIM3_PULSE_DTYPE = numpy.dtype([
    ("operation_id", "S64"),
    ("token", "S36"),
    ("pulse_call_perf", "<f8"),
    ("pulse_return_perf", "<f8"),
])
CAN_EVENT_DTYPE = numpy.dtype([
    ("stage", "S16"),
    ("token", "S36"),
    ("kind", "S40"),
    ("can_uuid", "<i2"),
    ("perf", "<f8"),
    ("perf_end", "<f8"),
    ("kernel_wall", "<f8"),
])
CAN_STAGES = ("trial_send", "token", "enqueue", "dequeue", "send", "ack")

EVENT_TABLES = {
    "live_poses": LIVE_POSE_DTYPE,
    "gate_observations": GATE_OBSERVATION_DTYPE,
    "stim_dispatch": STIM_DISPATCH_DTYPE,
    "stim3_pulses": STIM3_PULSE_DTYPE,
    "can_events": CAN_EVENT_DTYPE,
}
