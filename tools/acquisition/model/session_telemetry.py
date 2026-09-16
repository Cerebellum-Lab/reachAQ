import math
import time
import typing

from autotrainer.core.observable_object import ObservableObject


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

                      SENSOR TO RESULT is exposure to pose, so it also carries
                      the queue wait between the two - which on this rig has
                      been the larger term. This is the number the sub-5 ms
                      goal is about, and a model can look fast on the first
                      figure while missing the deadline on the second.

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

    def __init__(self):
        super().__init__()
        self._active = False
        self._started_perf: typing.Optional[float] = None
        self._ended_perf: typing.Optional[float] = None
        self._acquired_by_camera: typing.Dict[int, int] = {}
        self._dropped_by_camera: typing.Dict[int, int] = {}
        self._inferenced = 0
        self._inference_total_ms = 0.0
        self._inference_count = 0
        self._inference_max_ms = 0.0
        self._e2e_total_ms = 0.0
        self._e2e_count = 0
        self._e2e_max_ms = 0.0

    # -- lifecycle --------------------------------------------------------

    def begin(self, started_perf: typing.Optional[float] = None) -> None:
        """Reset for a new session. Every counter is per session, not lifetime."""
        self._active = True
        self._started_perf = time.perf_counter() if started_perf is None else started_perf
        self._ended_perf = None
        self._acquired_by_camera = {}
        self._dropped_by_camera = {}
        self._inferenced = 0
        self._inference_total_ms = 0.0
        self._inference_count = 0
        self._inference_max_ms = 0.0
        self._e2e_total_ms = 0.0
        self._e2e_count = 0
        self._e2e_max_ms = 0.0
        self.property_changed(self.ACTIVE_PROP, True, False)

    def end(self, ended_perf: typing.Optional[float] = None) -> None:
        """Freeze the counters. The summary stays readable for the metadata write."""
        if not self._active:
            return
        self._ended_perf = time.perf_counter() if ended_perf is None else ended_perf
        self._active = False
        self.property_changed(self.ACTIVE_PROP, False, True)

    # -- producers --------------------------------------------------------

    def record_capture(self, camera_index: int, acquired: int, dropped: int) -> None:
        """Absolute per-camera totals from the capture process.

        Totals rather than deltas on purpose: the capture process emits these
        periodically, and a dropped or reordered message would silently corrupt
        a running sum while an absolute value simply corrects itself.
        """
        if not self._active:
            return
        previous_acquired = self.frames_acquired
        previous_dropped = self.dropped_frames
        self._acquired_by_camera[camera_index] = int(acquired)
        self._dropped_by_camera[camera_index] = int(dropped)
        if self.frames_acquired != previous_acquired:
            self.property_changed(self.ACQUIRED_PROP, self.frames_acquired,
                                  previous_acquired)
        if self.dropped_frames != previous_dropped:
            self.property_changed(self.DROPPED_PROP, self.dropped_frames,
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
        if not self._active:
            return
        previous = self._inferenced
        new_calls = max(0, int(count) - previous)
        self._inferenced = int(count)
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
        if self._inferenced != previous:
            self.property_changed(self.INFERENCED_PROP, self._inferenced, previous)
            self.property_changed(self.INFERENCE_TIME_PROP,
                                  self.inference_mean_ms, None)

    # -- state ------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def elapsed_seconds(self) -> float:
        if self._started_perf is None:
            return 0.0
        end = self._ended_perf if self._ended_perf is not None else time.perf_counter()
        return max(0.0, end - self._started_perf)

    @property
    def frames_acquired(self) -> int:
        return max(self._acquired_by_camera.values(), default=0)

    @property
    def dropped_frames(self) -> int:
        return sum(self._dropped_by_camera.values())

    @property
    def has_dropped_frames(self) -> bool:
        return self.dropped_frames > 0

    @property
    def frames_inferenced(self) -> int:
        return self._inferenced

    @property
    def inferenced_percent(self) -> float:
        acquired = self.frames_acquired
        if acquired <= 0:
            return 0.0
        return 100.0 * self._inferenced / acquired

    @property
    def inference_mean_ms(self) -> float:
        if self._inference_count <= 0:
            return float("nan")
        return self._inference_total_ms / self._inference_count

    @property
    def inference_max_ms(self) -> float:
        return self._inference_max_ms if self._inference_count else float("nan")

    @property
    def sensor_to_result_mean_ms(self) -> float:
        if self._e2e_count <= 0:
            return float("nan")
        return self._e2e_total_ms / self._e2e_count

    @property
    def sensor_to_result_max_ms(self) -> float:
        return self._e2e_max_ms if self._e2e_count else float("nan")

    @property
    def dropped_by_camera(self) -> typing.Dict[int, int]:
        return dict(self._dropped_by_camera)

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
        }
