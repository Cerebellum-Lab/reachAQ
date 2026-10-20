import functools
import math
import threading
import time
import typing
from pathlib import Path

from autotrainer.core.observable_object import ObservableObject


def _telemetry_locked(method):
    @functools.wraps(method)
    def serialized(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return serialized


class SessionTelemetry(ObservableObject):
    """Live acquisition and inference counters for one recording session.

    Four numbers an operator cannot currently see while recording, and which
    only matter during the recording:

      elapsed         how long this session has been going.
      dropped frames  gaps in the camera's own frame-ID sequence - frames the
                      sensor produced that never reached the host. Distinct
                      from the pose path skipping frames, which is by design
                      and is not a fault. A drop means the recording itself
                      has a hole in it, which is worth knowing before the
                      animal has finished rather than during analysis.
      inferenced      what share of captured frames got a pose. Below 100% is
                      expected - the live path drains to the newest frame and
                      discards what it passed - but how far below sets how
                      often a closed-loop decision can be revised.
      inference time  two separate figures, because they answer different
                      questions and only one of them is the target.

                      The CALL is what the model costs: preprocess, forward,
                      decode. It is what changes when the model changes.

                      SENSOR TO RESULT runs from when the host received the
                      frame to the pose, so it also carries the queue wait
                      between the two - which on this rig has been the larger
                      term. It does not include exposure, readout or transfer:
                      the frame's time is fitted to host arrival. The latency
                      record (streams/latency.h5) measures those against the NI
                      exposure edge after the session. The metadata keys keep
                      their sensorToResult names for compatibility.

                      Both carry a max as well as a mean: a model averaging
                      4 ms with a 30 ms tail misses deadlines the mean hides.

    Counters are per camera because that is how they arrive. Frames acquired
    is taken as the maximum across cameras rather than the sum: the cameras are
    synchronised and a pose consumes one frame from each, so the sum would
    double the denominator and halve the percentage.

    Deliberately free of Qt. This is model state that a view observes, and
    keeping it plain means it can be exercised without a UI.
    """

    ELAPSED_PROP = "elapsed_seconds"
    DROPPED_PROP = "dropped_frames"
    ACQUIRED_PROP = "frames_acquired"
    INFERENCED_PROP = "frames_inferenced"
    INFERENCE_TIME_PROP = "inference_time"
    ACTIVE_PROP = "is_active"
    STIM_LATENCY_PROP = "stim_latency"
    LATENCY_STATUS_PROP = "latency_status"

    def __init__(self):
        super().__init__()
        # Frame counts, pose reports, begin/end and the panel's readings each
        # arrive on their own thread. Unserialized, a reset could be undone by
        # a report in flight, and a camera's first report could break a
        # reading in progress. Change notifications go out after the lock is
        # released, so an observer is free to read back.
        self._lock = threading.RLock()
        self._active = False
        self._started_perf: typing.Optional[float] = None
        self._ended_perf: typing.Optional[float] = None
        # Latest absolute totals, tracked whether or not a session is running.
        # They have to be known before begin() so it has something to subtract.
        self._acquired_by_camera: typing.Dict[int, int] = {}
        self._dropped_by_camera: typing.Dict[int, int] = {}
        self._inferenced = 0
        self._baseline_acquired: typing.Dict[int, int] = {}
        self._baseline_dropped: typing.Dict[int, int] = {}
        self._baseline_inferenced = 0
        self._frozen: typing.Optional[typing.Dict[str, typing.Any]] = None
        self._inference_total_ms = 0.0
        self._inference_count = 0
        self._inference_max_ms = 0.0
        self._e2e_total_ms = 0.0
        self._e2e_count = 0
        self._e2e_max_ms = 0.0
        self._stim_p99_ms = float("nan")
        self._latency_status = ""

    # -- lifecycle --------------------------------------------------------

    def begin(self, started_perf: typing.Optional[float] = None) -> None:
        """Reset for a new session. Every counter is per session, not lifetime.

        The counts arrive as totals accumulated by the capture and pose
        processes since they started, which is when the system went to Running
        - routinely tens of seconds before the operator pressed Record. So the
        session starts by remembering where those totals stood, and every
        figure below is reported relative to that mark. Clearing the
        dictionaries instead would let the next absolute report land whole: a
        45 s session measured that way claimed 7820 frames and 174 fps from a
        pair of 150 fps cameras.
        """
        with self._lock:
            self._active = True
            self._started_perf = time.perf_counter() if started_perf is None else started_perf
            self._ended_perf = None
            self._frozen = None
            self._baseline_acquired = dict(self._acquired_by_camera)
            self._baseline_dropped = dict(self._dropped_by_camera)
            self._baseline_inferenced = self._inferenced
            self._inference_total_ms = 0.0
            self._inference_count = 0
            self._inference_max_ms = 0.0
            self._e2e_total_ms = 0.0
            self._e2e_count = 0
            self._e2e_max_ms = 0.0
            self._stim_p99_ms = float("nan")
            self._latency_status = ""
        self.property_changed(self.ACTIVE_PROP, True, False)

    def end(self, ended_perf: typing.Optional[float] = None) -> None:
        """Freeze the counters. The summary stays readable for the metadata write."""
        with self._lock:
            if not self._active:
                return
            self._ended_perf = time.perf_counter() if ended_perf is None else ended_perf
            # Snapshot before clearing the flag: acquisition continues while the
            # system stays in Running, so the underlying totals keep climbing and
            # a live subtraction would make a finished session grow.
            self._frozen = {
                "acquired": self.frames_acquired,
                "dropped": self.dropped_frames,
                "dropped_by_camera": self.dropped_by_camera,
                "inferenced": self.frames_inferenced,
                # The metadata is written after end(), while triggers still reach
                # the live figure; without this the file would carry whichever
                # p99 happened to be current when it was written.
                "stim_p99_ms": self._stim_p99_ms,
            }
            self._active = False
        self.property_changed(self.ACTIVE_PROP, False, True)

    # -- producers --------------------------------------------------------

    def record_capture(self, camera_index: int, acquired: int, dropped: int) -> None:
        """Absolute per-camera totals from the capture process.

        Totals rather than deltas on purpose: the capture process emits these
        periodically, and a dropped or reordered message would silently corrupt
        a running sum while an absolute value simply corrects itself.

        Recorded even between sessions, because begin() subtracts whatever the
        totals had reached by then and it can only do that if it has seen them.
        """
        with self._lock:
            previous_acquired = self.frames_acquired
            previous_dropped = self.dropped_frames
            self._acquired_by_camera[camera_index] = int(acquired)
            self._dropped_by_camera[camera_index] = int(dropped)
            if not self._active:
                return
            frames_acquired = self.frames_acquired
            dropped_frames = self.dropped_frames
        if frames_acquired != previous_acquired:
            self.property_changed(self.ACQUIRED_PROP, frames_acquired,
                                  previous_acquired)
        if dropped_frames != previous_dropped:
            self.property_changed(self.DROPPED_PROP, dropped_frames,
                                  previous_dropped)

    def record_inference(self, count: int, mean_ms: float, max_ms: float,
                         *, sensor_to_result_mean_ms: float = float("nan"),
                         sensor_to_result_max_ms: float = float("nan")) -> None:
        """Absolute pose count, with the mean and max of the calls since the last report.

        The mean arrives per window rather than per call because the pose
        process reports on a window; it is folded into a session mean weighted
        by how many calls it covered, so a slow burst is not averaged away by a
        later quiet one.
        """
        with self._lock:
            previous = self._inferenced
            new_calls = max(0, int(count) - previous)
            self._inferenced = int(count)
            if not self._active:
                return
            if new_calls and math.isfinite(mean_ms):
                self._inference_total_ms += mean_ms * new_calls
                self._inference_count += new_calls
            if math.isfinite(max_ms):
                self._inference_max_ms = max(self._inference_max_ms, float(max_ms))
            if new_calls and math.isfinite(sensor_to_result_mean_ms):
                self._e2e_total_ms += sensor_to_result_mean_ms * new_calls
                self._e2e_count += new_calls
            if math.isfinite(sensor_to_result_max_ms):
                self._e2e_max_ms = max(self._e2e_max_ms,
                                       float(sensor_to_result_max_ms))
            if self._inferenced == previous:
                return
            frames_inferenced = self.frames_inferenced
            inference_mean_ms = self.inference_mean_ms
        self.property_changed(self.INFERENCED_PROP, frames_inferenced, None)
        self.property_changed(self.INFERENCE_TIME_PROP, inference_mean_ms, None)

    def record_stim_p99(self, p99_ms: float) -> None:
        """The direct stim route's rolling p99, from StimLatencyBudget."""
        current = float(p99_ms)
        with self._lock:
            previous, self._stim_p99_ms = self._stim_p99_ms, current
        self.property_changed(self.STIM_LATENCY_PROP, current, previous)

    def set_latency_status(self, text: str) -> None:
        """One line on the session's latency record, set once it is finalized."""
        current = str(text)
        with self._lock:
            previous, self._latency_status = self._latency_status, current
        self.property_changed(self.LATENCY_STATUS_PROP, current, previous)

    # -- state ------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    @_telemetry_locked
    def elapsed_seconds(self) -> float:
        if self._started_perf is None:
            return 0.0
        end = self._ended_perf if self._ended_perf is not None else time.perf_counter()
        return max(0.0, end - self._started_perf)

    @staticmethod
    def _since_baseline(latest: int, baseline: int) -> int:
        """One camera's contribution to this session.

        A capture process that restarted reports a total below the mark taken
        at begin(). Counting from zero there rather than returning a negative
        is the conservative direction: over-reporting a frame that predates the
        restart is better than hiding one that did not.
        """
        return latest if latest < baseline else latest - baseline

    @property
    @_telemetry_locked
    def frames_acquired(self) -> int:
        if self._frozen is not None:
            return self._frozen["acquired"]
        if self._started_perf is None:
            return 0
        return max(
            (self._since_baseline(value, self._baseline_acquired.get(index, 0))
             for index, value in self._acquired_by_camera.items()),
            default=0,
        )

    @property
    @_telemetry_locked
    def dropped_frames(self) -> int:
        if self._frozen is not None:
            return self._frozen["dropped"]
        if self._started_perf is None:
            return 0
        return sum(
            self._since_baseline(value, self._baseline_dropped.get(index, 0))
            for index, value in self._dropped_by_camera.items()
        )

    @property
    def has_dropped_frames(self) -> bool:
        return self.dropped_frames > 0

    @property
    @_telemetry_locked
    def frames_inferenced(self) -> int:
        """Poses computed this session, never more than frames acquired.

        The two counts come from separate periodic reports - the capture
        process every 0.5 s, the pose process every 0.5 s - so at any instant
        one can be up to half a second fresher than the other. At 150 fps that
        is 75 frames of skew, which showed up as a session reporting 101.4% of
        frames inferenced.

        A pose is computed from an acquired frame, so more poses than frames is
        not a thing that can happen; the excess is always the newer counter
        running ahead of the older one. Capping keeps the count and the
        percentage consistent with each other rather than leaving an operator
        to reconcile 4575 of 4512.
        """
        if self._frozen is not None:
            return self._frozen["inferenced"]
        if self._started_perf is None:
            return 0
        inferenced = self._since_baseline(self._inferenced,
                                          self._baseline_inferenced)
        acquired = self.frames_acquired
        return min(inferenced, acquired) if acquired > 0 else inferenced

    @property
    @_telemetry_locked
    def inferenced_percent(self) -> float:
        acquired = self.frames_acquired
        if acquired <= 0:
            return 0.0
        return 100.0 * self.frames_inferenced / acquired

    @property
    @_telemetry_locked
    def inference_mean_ms(self) -> float:
        if self._inference_count <= 0:
            return float("nan")
        return self._inference_total_ms / self._inference_count

    @property
    @_telemetry_locked
    def inference_max_ms(self) -> float:
        return self._inference_max_ms if self._inference_count else float("nan")

    @property
    @_telemetry_locked
    def sensor_to_result_mean_ms(self) -> float:
        if self._e2e_count <= 0:
            return float("nan")
        return self._e2e_total_ms / self._e2e_count

    @property
    @_telemetry_locked
    def sensor_to_result_max_ms(self) -> float:
        return self._e2e_max_ms if self._e2e_count else float("nan")

    @property
    @_telemetry_locked
    def stim_p99_ms(self) -> float:
        if self._frozen is not None:
            return self._frozen["stim_p99_ms"]
        return self._stim_p99_ms

    @property
    def latency_status(self) -> str:
        return self._latency_status

    @property
    @_telemetry_locked
    def dropped_by_camera(self) -> typing.Dict[int, int]:
        if self._frozen is not None:
            return dict(self._frozen["dropped_by_camera"])
        if self._started_perf is None:
            return {}
        return {
            index: self._since_baseline(value,
                                        self._baseline_dropped.get(index, 0))
            for index, value in self._dropped_by_camera.items()
        }

    @_telemetry_locked
    def summary(self) -> typing.Dict[str, typing.Any]:
        """What goes into the session metadata.

        Plain JSON-safe types, and non-finite values become None rather than
        NaN, which is not valid JSON and which the metadata writer strips
        anyway.
        """

        def finite(value):
            return float(value) if math.isfinite(value) else None

        # camelCase to match the session metadata this lands in, rather than
        # introducing a translation layer for one dictionary.
        return {
            "elapsedSeconds": round(self.elapsed_seconds, 3),
            "framesAcquired": self.frames_acquired,
            "framesDropped": self.dropped_frames,
            "framesDroppedByCamera": {str(k): v
                                      for k, v in self.dropped_by_camera.items()},
            "framesInferenced": self.frames_inferenced,
            "inferencedPercent": round(self.inferenced_percent, 2),
            "inferenceCallMeanMs": finite(self.inference_mean_ms),
            "inferenceCallMaxMs": finite(self.inference_max_ms),
            "sensorToResultMeanMs": finite(self.sensor_to_result_mean_ms),
            "sensorToResultMaxMs": finite(self.sensor_to_result_max_ms),
            "stimP99Ms": finite(self.stim_p99_ms),
        }


def describe_latency_status(status) -> str:
    """'complete (mixed)', 'partial (software)', 'failed' - the panel's latency line."""
    if not status:
        return ""
    pose = ((status.get("loops") or {}).get("pose") or {}).get("summary") or {}
    confidence = pose.get("confidence")
    text = str(status.get("status", "failed"))
    return f"{text} ({confidence})" if confidence else text


def exact_session_capture(alignment: dict,
                          posed_frame_paths: typing.Mapping[str, Path]
                          ) -> typing.Optional[typing.Dict[str, typing.Any]]:
    """Exact counts for a finished session, from its own files.

    The counters above come from totals reported every 0.5 s, so a session's
    frames acquired and inferenced are each good to about 75 frames at 150 fps
    either way, and a stale frame count reads as exactly 100% inferenced. The
    files are exact. The merged timeline (frame_timing.csv, counted in
    camera_alignment.json) has one row per recorded frame, and each camera's
    processed-frames file lists the rows that got a live pose: the pose queue
    indexes a frame by its camera frame id less the recording's first, which is
    its row. That includes poses of recorded frames computed in the moment
    after Stop, which the counters exclude.

    Percentages are of the frames the camera delivered. A blank the capture
    inserted for a dropped frame goes to the writers only, never to the pose
    queue, so it can never be posed. None when the timeline was not merged;
    per camera, None where the file is missing.
    """
    frames_recorded = int(alignment.get("primaryFrameCount") or 0)
    if frames_recorded <= 0:
        return None
    cameras = {}
    for name, camera in alignment.get("cameras", {}).items():
        present = camera.get("presentFrames")
        delivered = frames_recorded if present is None else int(present)
        path = posed_frame_paths.get(name)
        posed = (None if path is None or not path.is_file()
                 else _count_posed_frames(path, frames_recorded))
        cameras[name] = {
            "framesDropped": (None if present is None
                              else frames_recorded - delivered),
            "framesPosed": posed,
            "posedPercent": (None if posed is None or delivered <= 0
                             else round(100.0 * posed / delivered, 2)),
        }
    return {"framesRecorded": frames_recorded, "cameras": cameras}


def _count_posed_frames(path: Path, frames_recorded: int) -> int:
    with path.open(encoding="utf-8") as stream:
        slots = {int(line) for line in stream if line.strip()}
    return sum(1 for slot in slots if 0 <= slot < frames_recorded)
