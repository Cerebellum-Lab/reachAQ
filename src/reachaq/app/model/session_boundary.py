from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class SessionBoundary:
    """Canonical immutable recording boundary shared by every session writer."""

    session_id: str
    primary_camera: str
    primary_frame_id: int
    start_perf_time: float
    start_wall_time: float
    camera_when: float
    end_perf_time: Optional[float] = None
    end_wall_time: Optional[float] = None
    nidaq_sample_index: Optional[int] = None

    def with_end(self, end_perf_time: float) -> "SessionBoundary":
        end_perf_time = max(self.start_perf_time, float(end_perf_time))
        return dataclasses.replace(
            self,
            end_perf_time=end_perf_time,
            end_wall_time=self.start_wall_time
            + (end_perf_time - self.start_perf_time),
        )

    def with_nidaq_sample_index(self, sample_index: int) -> "SessionBoundary":
        return dataclasses.replace(self, nidaq_sample_index=int(sample_index))

    def to_metadata(self) -> dict:
        duration = (
            None
            if self.end_perf_time is None
            else max(0.0, self.end_perf_time - self.start_perf_time)
        )
        return {
            "sessionId": self.session_id,
            "source": "primary_camera_recorded_frames",
            "primaryCamera": self.primary_camera,
            "primaryFrameId": self.primary_frame_id,
            "startPerfTime": self.start_perf_time,
            "startWallTime": self.start_wall_time,
            "cameraWhen": self.camera_when,
            "endPerfTime": self.end_perf_time,
            "endWallTime": self.end_wall_time,
            "durationSeconds": duration,
            "nidaqSampleIndex": self.nidaq_sample_index,
        }
