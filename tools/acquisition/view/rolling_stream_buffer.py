from __future__ import annotations

from typing import Tuple

import numpy as np


class RollingStreamBuffer:
    """Fixed-capacity NumPy ring buffer used by live graph curves.

    Appending never shifts the existing window.  A contiguous ordered copy is
    made only when a graph is repainted, which is substantially cheaper than
    calling ``np.roll`` for every incoming sample block.
    """

    def __init__(self, capacity: int):
        self._capacity = max(1, int(capacity))
        self._x = np.empty(self._capacity, dtype=np.float64)
        self._y = np.empty(self._capacity, dtype=np.float64)
        self._next = 0
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def latest_x(self):
        if self._size == 0:
            return None
        return float(self._x[(self._next - 1) % self._capacity])

    def clear(self) -> None:
        self._next = 0
        self._size = 0

    def append(self, x_values, y_values) -> None:
        x = np.asarray(x_values, dtype=np.float64).reshape(-1)
        y = np.asarray(y_values, dtype=np.float64).reshape(-1)
        count = min(x.size, y.size)
        if count <= 0:
            return
        if count >= self._capacity:
            self._x[:] = x[count - self._capacity : count]
            self._y[:] = y[count - self._capacity : count]
            self._next = 0
            self._size = self._capacity
            return

        first = min(count, self._capacity - self._next)
        self._x[self._next : self._next + first] = x[:first]
        self._y[self._next : self._next + first] = y[:first]
        remaining = count - first
        if remaining:
            self._x[:remaining] = x[first:count]
            self._y[:remaining] = y[first:count]
        self._next = (self._next + count) % self._capacity
        self._size = min(self._capacity, self._size + count)

    def resize(self, capacity: int) -> None:
        """Resize the ring while retaining the newest samples."""
        capacity = max(1, int(capacity))
        if capacity == self._capacity:
            return
        x_values, y_values = self.ordered()
        self._capacity = capacity
        self._x = np.empty(capacity, dtype=np.float64)
        self._y = np.empty(capacity, dtype=np.float64)
        self._next = 0
        self._size = 0
        self.append(x_values[-capacity:], y_values[-capacity:])

    def ordered(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        start = (self._next - self._size) % self._capacity
        end = start + self._size
        if end <= self._capacity:
            return self._x[start:end].copy(), self._y[start:end].copy()
        split = self._capacity - start
        return (
            np.concatenate((self._x[start:], self._x[: self._size - split])),
            np.concatenate((self._y[start:], self._y[: self._size - split])),
        )

    def ordered_for_plot(self, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a bounded peak envelope suitable for a live plot.

        Each output bucket retains both its minimum and maximum in time order.
        This keeps narrow TTL pulses visible while avoiding the cost of sending
        an entire high-rate acquisition window through Qt on every repaint.
        """
        x_values, y_values = self.ordered()
        max_points = max(2, int(max_points))
        if x_values.size <= max_points:
            return x_values, y_values

        bucket_count = max(1, max_points // 2)
        bucket_size = int(np.ceil(x_values.size / bucket_count))
        usable_size = (x_values.size // bucket_size) * bucket_size
        if usable_size < bucket_size:
            return x_values[-max_points:], y_values[-max_points:]

        # Discard at most one partial bucket from the oldest edge so the newest
        # sample is always represented in the rolling display.
        x_buckets = x_values[-usable_size:].reshape(-1, bucket_size)
        y_buckets = y_values[-usable_size:].reshape(-1, bucket_size)
        min_indices = np.argmin(y_buckets, axis=1)
        max_indices = np.argmax(y_buckets, axis=1)
        first_indices = np.minimum(min_indices, max_indices)
        second_indices = np.maximum(min_indices, max_indices)
        rows = np.arange(y_buckets.shape[0])

        plot_x = np.empty(y_buckets.shape[0] * 2, dtype=np.float64)
        plot_y = np.empty(y_buckets.shape[0] * 2, dtype=np.float64)
        plot_x[0::2] = x_buckets[rows, first_indices]
        plot_x[1::2] = x_buckets[rows, second_indices]
        plot_y[0::2] = y_buckets[rows, first_indices]
        plot_y[1::2] = y_buckets[rows, second_indices]
        return plot_x, plot_y
