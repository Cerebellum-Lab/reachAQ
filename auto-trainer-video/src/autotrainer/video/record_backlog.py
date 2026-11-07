"""Notice an encoder that is falling behind capture, before capture stops."""
from typing import Optional


class RecordBacklog:
    """Says once when the recorder queue passes a fraction of its capacity.

    The capture thread hands frames to the recorder thread in batches through
    a bounded queue. A queue that keeps filling means the encoder is slower
    than the camera; once it is full the put times out, and after a few of
    those capture stops. The queue also holds every queued frame in memory
    (2 GB per camera at 512x512, 8 GB at 1024x1024). Warning at a quarter
    full names the cause well before either happens.
    """

    def __init__(self, maxsize: int, batch_frames: int, frame_bytes: int,
                 *, warn_fraction: float = 0.25, rearm_fraction: float = 1 / 16):
        self._maxsize = int(maxsize)
        self._batch_frames = int(batch_frames)
        self._frame_bytes = int(frame_bytes)
        self._warn_at = max(1, int(self._maxsize * warn_fraction)) if self._maxsize > 0 else None
        self._rearm_below = max(1, int(self._maxsize * rearm_fraction))
        self._armed = True
        self.peak = 0

    def observe(self, depth: int) -> Optional[str]:
        """Note the queue depth in batches; return a warning the first time it crosses."""
        depth = int(depth)
        self.peak = max(self.peak, depth)
        if self._warn_at is None:
            return None
        if self._armed and depth >= self._warn_at:
            self._armed = False
            frames = depth * self._batch_frames
            return (f"recorder backlog: {depth}/{self._maxsize} batches queued "
                    f"({frames} frames, {frames * self._frame_bytes / 1e6:.0f} MB held); "
                    "the encoder is not keeping up with capture")
        if not self._armed and depth < self._rearm_below:
            self._armed = True
        return None

    def take_peak(self) -> int:
        """The deepest backlog since the last call; the count restarts from zero."""
        peak, self.peak = self.peak, 0
        return peak
