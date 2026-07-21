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
