"""A synthetic rig for the frame-alignment tests: the NI spool, the primary
camera's frame ledger and its latency stream, from one model of the clocks
(drift/drift_align.py and ntp/pll_sim.py in the 2026-10-07 drift report).

* NI time t = sample index / RATE. Its true host time is
      H(t) = HOST0 + (1 + DRIFT) * t + slew(t),
  DRIFT the NI clock against perf_counter (-10 ppm, as christielab10 runs),
  slew(t) the timesyncd corrections: a poll at T with offset theta adds
  theta * (1 - exp(-(t - T) / TAU)) and a permanent rate step theta * PLL_GAIN.
  Polls are superposed, each acting as if it were alone. The kernel instead
  replaces what is left of the previous offset at every poll, so the model is
  exact only for polls far apart against TAU: the Task 6 review measured the
  difference at 1.8 % with the real 2048 s spacing and 31 % at 90 s.
* The epoch timeline the recorder stores (nidaq.h5 perf_time) is
      HOST0 - EPOCH_EARLY + t:
  stamped EPOCH_EARLY before sample 0, knowing no drift and no slew. Against
  the host it reads
      epoch - H = -EPOCH_EARLY - DRIFT * t - slew(t)
  so with DRIFT = -10 ppm it is 1.65 ms early at 5 s and 5.3 ms late at 700 s.
* NI reads of READ samples run back to back from index 0. Every sample of a
  read carries the stamp taken before the read's blocking call, which is when
  the read before it returned: H(last sample of that read) plus a 0.2-3.2 ms
  delivery delay. One run of equal stamps per read; the recorder's 10 ms
  copies, which can fold two reads into one run, are not modelled (the
  pairing rule is the same either way).
* CLOCK_MONOTONIC_RAW is the same oscillator without the corrections,
  HOST0 + RAW_OFFSET + (1 + DRIFT) * t, read beside each stamp; NaN unless
  asked for, as on a host without the clock.
* cam_frames toggles at every exposure start t_n = EDGE0 + n * PERIOD, at
  sample round(t_n * RATE). Frame n has id FIRST_ID + n, a camera stamp at its
  exposure end on a camera clock = NI seconds + CAMERA_OFFSET, and
  frame_perf_c = H(t_n) + lag. The lag is against the true edge H(t_n). The
  drift report's L (0.8-1.8 ms measured) is against the edge as the NI-to-host
  fit maps it, which is 0.2-0.3 ms late on the perf stamps of a spool a few
  tens of seconds long (a self-test pins that, with and without two small
  polls), so the same frames have a lag here of about L plus 0.2-0.3 ms. That
  is not a property of the perf-only envelope in general: across a long
  session of NTP polls it is not late-never-early (the final review's 4 h proof
  measured up to 46 us early, 260 us late at most), and only the RAW bridge is.
* Clock latches: three 1 ms after a frame's exposure start, each a +-0.1 ms
  perf bracket around H(t) holding the camera clock at t. The RAW pair of the
  latch row is NaN: no consumer reads it yet, and this rig does not model a
  camera-side RAW latch.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import h5py
import numpy as np

from autotrainer.core.latency.schema import (
    CAMERA_FRAME_DTYPE,
    CLOCK_LATCH_DTYPE,
    CLOCK_PAIR_DTYPE,
)
from tools.acquisition.model.session_boundary import SessionBoundary
from tools.acquisition.model.session_data_recorder import _NidaqSpoolSnapshot

RATE = 10_000.0
READ = 167
FPS = 150.0
PERIOD = 1.0 / FPS
EDGE0 = 0.0123
HOST0 = 1000.0
EPOCH_EARLY = 0.0017
DRIFT = -10e-6
TAU = 512.0
PLL_GAIN = 1024 / 2 ** 22
CAMERA_OFFSET = 7.0
RAW_OFFSET = 3.0
EXPOSURE = 175e-6
FIRST_ID = 1000
ARRIVAL = 0.0023


class SyntheticRig:
    def __init__(self, *, polls=(), seed=5):
        self.polls = tuple(polls)
        self.rng = np.random.default_rng(seed)

    def slew(self, t):
        """The timesyncd corrections at NI time t, polls superposed (see the module docstring)."""
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros_like(t)
        for when, theta in self.polls:
            after = np.clip(t - when, 0.0, None)
            out = out + theta * (1 - np.exp(-after / TAU)) + theta * PLL_GAIN * after
        return out

    def host(self, t):
        t = np.asarray(t, dtype=np.float64)
        return HOST0 + (1 + DRIFT) * t + self.slew(t)

    @staticmethod
    def raw(t):
        return HOST0 + RAW_OFFSET + (1 + DRIFT) * np.asarray(t, dtype=np.float64)

    @staticmethod
    def epoch_perf(t):
        return HOST0 - EPOCH_EARLY + np.asarray(t, dtype=np.float64)

    @staticmethod
    def edge_time(n):
        return EDGE0 + np.asarray(n, dtype=np.float64) * PERIOD

    @staticmethod
    def edge_sample(n):
        return int(round(float(SyntheticRig.edge_time(n)) * RATE))

    @staticmethod
    def frame_id(n):
        return FIRST_ID + n

    @staticmethod
    def first_edge_at_or_after(t0):
        return int(math.ceil((t0 - EDGE0) / PERIOD))

    @staticmethod
    def camera_ns(t):
        return int(round((float(t) + CAMERA_OFFSET) * 1e9))

    def frame_perf_c(self, n, lag):
        return float(self.host(self.edge_time(n))) + lag

    def write_spool(self, path, *, t_from, t_to, tone_pulses=(), raw=False, stamps=True):
        """The recorder's spool for NI seconds [t_from, t_to): the layout
        _open_nidaq_spool_locked creates, and the snapshot _write_session takes."""
        first = int(round(t_from * RATE))
        last = int(round(t_to * RATE))
        index = np.arange(first, last, dtype=np.int64)
        t = index / RATE
        read_number = index // READ
        reads = np.arange(read_number[0], read_number[-1] + 1)
        previous_last = reads * READ - 1
        delay = 0.0002 + self.rng.uniform(0, 0.003, reads.size)
        stamp = self.host(previous_last / RATE) + delay
        raw_stamp = self.raw(previous_last / RATE) + delay if raw else np.full(reads.size, np.nan)
        if not stamps:
            stamp = np.full(reads.size, np.nan)
        per_sample = read_number - reads[0]
        block_perf = stamp[per_sample]
        block_raw = raw_stamp[per_sample]
        edges = np.array([self.edge_sample(n) for n in range(int((t_to + 1.0 - EDGE0) / PERIOD) + 1)])
        cam = (np.searchsorted(edges, index, side="right") % 2).astype(np.float32)
        channels, rows = ["cam_frames"], [cam]
        if tone_pulses:
            tone = np.zeros(index.size, dtype=np.float32)
            for start, duration in tone_pulses:
                low = int(round(start * RATE)) - first
                high = int(round((start + duration) * RATE)) - first
                tone[max(low, 0):max(high, 0)] = 1.0
            channels.append("tone1")
            rows.append(tone)
        perf = self.epoch_perf(t)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as store:
            store.create_dataset("sample_index", data=index)
            store.create_dataset("perf_time", data=perf)
            store.create_dataset("wall_time", data=perf + 1e9)
            store.create_dataset("epoch", data=np.ones(index.size, dtype=np.uint64))
            store.create_dataset("block_observation_perf_time", data=block_perf)
            store.create_dataset("block_observation_wall_time", data=block_perf + 1e9)
            store.create_dataset("block_observation_raw_time", data=block_raw)
            store.create_dataset("values", data=np.vstack(rows))
            store.attrs["channel_names"] = tuple(channels)
            store.attrs["sample_rate_hz"] = RATE
        return _NidaqSpoolSnapshot(
            path=path, channel_names=tuple(channels), sample_rate_hz=RATE,
            sample_count=int(index.size), first_sample_index=first,
            last_sample_index=last - 1, first_perf_time=float(perf[0]),
            last_perf_time=float(perf[-1]), gap_count=0, overrun_samples=0,
            first_error=None, last_error=None, error_count=0, worker_failed=False,
        )

    def start(self, project, n, lag):
        """The recording boundary a session starting at frame n commits."""
        start_perf = self.frame_perf_c(n, lag)
        start_wall = 1.7e9 + start_perf
        boundary = SessionBoundary(
            session_id=project.short_id, primary_camera="left",
            primary_frame_id=self.frame_id(n), start_perf_time=start_perf,
            start_wall_time=start_wall,
            camera_when=float(self.camera_ns(float(self.edge_time(n)) + EXPOSURE)),
        )
        return start_perf, start_wall, boundary

    def write_frame_ledger(self, project, n_first, count, lag):
        """frame_timing.csv as the writer leaves it: utc_when is the wall anchor plus
        each frame's frame_perf_c offset."""
        start_perf = self.frame_perf_c(n_first, lag)
        start_wall = 1.7e9 + start_perf
        path = Path(project.get_frame_timing_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_id", "frame_when", "frame_present_primary",
                             "frame_present_secondary", "utc_when"])
            for n in range(n_first, n_first + count):
                writer.writerow([
                    self.frame_id(n), self.camera_ns(float(self.edge_time(n)) + EXPOSURE), 1, 1,
                    f"{start_wall + self.frame_perf_c(n, lag) - start_perf:.6f}",
                ])

    def write_latency_stream(self, project, name, n_first, count, lag, *, latches=True):
        """streams/latency/camera_<name>.h5 as the capture process writes it."""
        session = Path(project.get_session_path().location)
        path = session / "streams" / "latency" / f"camera_{name}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        n = np.arange(n_first, n_first + count)
        edge = self.edge_time(n)
        frames = np.zeros(count, dtype=CAMERA_FRAME_DTYPE)
        frames["frame_id"] = self.frame_id(n)
        frames["camera_ts_ns"] = np.round((edge + EXPOSURE + CAMERA_OFFSET) * 1e9).astype(np.int64)
        frames["arrival_perf"] = self.host(edge) + ARRIVAL
        frames["poll_perf"] = frames["arrival_perf"] - 0.0005
        frames["capture_return_perf"] = frames["arrival_perf"] + 0.0001
        rows = []
        if latches:
            for when in (float(edge[0]) + 0.001, float(edge[-1]) + 0.001):
                host = float(self.host(when))
                rows.extend([(host - 0.0001, self.camera_ns(when), host + 0.0001, np.nan, np.nan)] * 3)
        latch_rows = np.zeros(len(rows), dtype=CLOCK_LATCH_DTYPE)
        for position, row in enumerate(rows):
            latch_rows[position] = row[:len(CLOCK_LATCH_DTYPE.names)]
        pairs = np.array([(1.7e9 + value, value) for value in
                          (float(frames["arrival_perf"][0]), float(frames["arrival_perf"][-1]))],
                         dtype=CLOCK_PAIR_DTYPE)
        with h5py.File(path, "w") as store:
            store.attrs["schema_version"] = 1
            store.attrs["camera"] = name
            store.attrs["camera_index"] = 0
            store.attrs["is_primary"] = True
            store.create_dataset("frames", data=frames)
            store.create_dataset("clock_latches", data=latch_rows)
            store.create_dataset("clock_pairs", data=pairs)
        return path
