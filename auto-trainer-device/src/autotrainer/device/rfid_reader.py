from __future__ import annotations

import dataclasses
import enum
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from autotrainer.core.animal.external_metadata import normalize_rfid


STX = 0x02
ETX = 0x03
PAYLOAD_LENGTH = 26
FRAME_LENGTH = 30
DEFAULT_RFID_DEVICE = Path(
    "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG01OZ68-if00-port0"
)
_PAYLOAD_RE = re.compile(rb"^[0-9A-Fa-f]{26}$")


class RfidReaderState(str, enum.Enum):
    STOPPED = "stopped"
    CONNECTING = "connecting"
    READY = "ready"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass(frozen=True)
class RfidTagRead:
    rfid: str
    raw_frame: bytes
    monotonic_time: float


@dataclass(frozen=True)
class RfidReaderStatus:
    state: RfidReaderState
    device: str
    reason: str = ""


@dataclass
class RfidParserStats:
    valid_frames: int = 0
    invalid_checksum_frames: int = 0
    invalid_payload_frames: int = 0
    discarded_bytes: int = 0
    buffer_overflows: int = 0


def payload_checksum(payload: bytes) -> int:
    checksum = 0
    for byte in payload:
        checksum ^= byte
    return checksum


def make_rfid_frame(payload: str) -> bytes:
    """Build a frame for tests and diagnostic fixtures."""
    normalized = normalize_rfid(payload)
    payload_bytes = normalized.encode("ascii")
    checksum = payload_checksum(payload_bytes)
    return bytes((STX,)) + payload_bytes + bytes((checksum, checksum ^ 0xFF, ETX))


class RfidFrameParser:
    """Incrementally parse the observed 30-byte RFID serial protocol."""

    def __init__(self, *, maximum_buffer_bytes: int = 4096):
        if maximum_buffer_bytes < FRAME_LENGTH:
            raise ValueError("maximum_buffer_bytes must hold at least one frame")
        self._maximum_buffer_bytes = int(maximum_buffer_bytes)
        self._buffer = bytearray()
        self.stats = RfidParserStats()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    def feed(
        self,
        data: bytes,
        *,
        monotonic_time: Optional[float] = None,
    ) -> List[RfidTagRead]:
        if not data:
            return []
        self._buffer.extend(data)
        if len(self._buffer) > self._maximum_buffer_bytes:
            self.stats.buffer_overflows += 1
            last_stx = self._buffer.rfind(STX)
            if last_stx < 0:
                self.stats.discarded_bytes += len(self._buffer)
                self._buffer.clear()
            else:
                self.stats.discarded_bytes += last_stx
                del self._buffer[:last_stx]
                if len(self._buffer) > self._maximum_buffer_bytes:
                    excess = len(self._buffer) - self._maximum_buffer_bytes
                    self.stats.discarded_bytes += excess
                    del self._buffer[:excess]

        emitted: List[RfidTagRead] = []
        event_time = time.monotonic() if monotonic_time is None else monotonic_time
        while True:
            try:
                start = self._buffer.index(STX)
            except ValueError:
                self.stats.discarded_bytes += len(self._buffer)
                self._buffer.clear()
                break
            if start:
                self.stats.discarded_bytes += start
                del self._buffer[:start]
            if len(self._buffer) < FRAME_LENGTH:
                break

            candidate = bytes(self._buffer[:FRAME_LENGTH])
            if candidate[-1] != ETX:
                self.stats.invalid_payload_frames += 1
                self.stats.discarded_bytes += 1
                del self._buffer[0]
                continue

            payload = candidate[1:1 + PAYLOAD_LENGTH]
            if _PAYLOAD_RE.fullmatch(payload) is None:
                self.stats.invalid_payload_frames += 1
                self.stats.discarded_bytes += FRAME_LENGTH
                del self._buffer[:FRAME_LENGTH]
                continue

            checksum = payload_checksum(payload)
            if candidate[27] != checksum or candidate[28] != (checksum ^ 0xFF):
                self.stats.invalid_checksum_frames += 1
                self.stats.discarded_bytes += FRAME_LENGTH
                del self._buffer[:FRAME_LENGTH]
                continue

            normalized = normalize_rfid(payload.decode("ascii"))
            emitted.append(RfidTagRead(normalized, candidate, float(event_time)))
            self.stats.valid_frames += 1
            del self._buffer[:FRAME_LENGTH]
        return emitted


class RfidDuplicateSuppressor:
    def __init__(self, window_seconds: float = 1.0):
        if window_seconds < 0:
            raise ValueError("window_seconds cannot be negative")
        self.window_seconds = float(window_seconds)
        self._last_rfid: Optional[str] = None
        self._last_time = float("-inf")

    def accept(self, event: RfidTagRead) -> bool:
        duplicate = (
            event.rfid == self._last_rfid
            and event.monotonic_time - self._last_time < self.window_seconds
        )
        if not duplicate:
            self._last_rfid = event.rfid
            self._last_time = event.monotonic_time
        return not duplicate


class RfidReaderService:
    """Reconnectable pyserial reader with no Qt or AppModel dependency."""

    def __init__(
        self,
        *,
        device: str = str(DEFAULT_RFID_DEVICE),
        baud: int = 9600,
        on_tag: Callable[[RfidTagRead], None],
        on_status: Optional[Callable[[RfidReaderStatus], None]] = None,
        duplicate_window_seconds: float = 1.0,
        reconnect_initial_seconds: float = 0.5,
        reconnect_max_seconds: float = 10.0,
        maximum_reconnect_attempts: Optional[int] = None,
        serial_factory=None,
    ):
        self.device = str(device)
        self.baud = int(baud)
        self._on_tag = on_tag
        self._on_status = on_status or (lambda _status: None)
        self._parser = RfidFrameParser()
        self._suppressor = RfidDuplicateSuppressor(duplicate_window_seconds)
        self._reconnect_initial_seconds = float(reconnect_initial_seconds)
        self._reconnect_max_seconds = float(reconnect_max_seconds)
        self._maximum_reconnect_attempts = maximum_reconnect_attempts
        if maximum_reconnect_attempts is not None and maximum_reconnect_attempts < 1:
            raise ValueError("maximum_reconnect_attempts must be at least 1")
        self._serial_factory = serial_factory
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._serial = None
        self._lock = threading.RLock()

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="rfid-reader",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        with self._lock:
            serial_port = self._serial
        if serial_port is not None:
            try:
                serial_port.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        self._thread = None
        self._publish_status(RfidReaderState.STOPPED)

    close = stop

    def _make_serial(self):
        if self._serial_factory is not None:
            return self._serial_factory(self.device, self.baud)
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required for the RFID reader") from exc
        return serial.Serial(
            self.device,
            baudrate=self.baud,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.1,
        )

    def _publish_status(self, state: RfidReaderState, reason: str = "") -> None:
        self._on_status(RfidReaderStatus(state, self.device, reason))

    def _run(self) -> None:
        backoff = self._reconnect_initial_seconds
        first_attempt = True
        consecutive_failures = 0
        while not self._stop_event.is_set():
            self._publish_status(
                RfidReaderState.CONNECTING if first_attempt else RfidReaderState.RECONNECTING
            )
            first_attempt = False
            try:
                serial_port = self._make_serial()
                with self._lock:
                    self._serial = serial_port
                self._parser.reset()
                backoff = self._reconnect_initial_seconds
                self._publish_status(RfidReaderState.READY)
                while not self._stop_event.is_set():
                    chunk = serial_port.read(512)
                    if not chunk:
                        continue
                    for event in self._parser.feed(chunk):
                        if self._suppressor.accept(event):
                            self._on_tag(event)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                consecutive_failures += 1
                fatal_dependency_error = isinstance(exc, RuntimeError) and (
                    "pyserial is required" in str(exc)
                )
                attempts_exhausted = (
                    self._maximum_reconnect_attempts is not None
                    and consecutive_failures >= self._maximum_reconnect_attempts
                )
                if fatal_dependency_error or attempts_exhausted:
                    self._publish_status(RfidReaderState.FAILED, str(exc))
                    return
                self._publish_status(RfidReaderState.RECONNECTING, str(exc))
                self._stop_event.wait(backoff)
                backoff = min(self._reconnect_max_seconds, max(backoff * 2, 0.1))
            finally:
                with self._lock:
                    serial_port, self._serial = self._serial, None
                if serial_port is not None:
                    try:
                        serial_port.close()
                    except Exception:
                        pass
        self._publish_status(RfidReaderState.STOPPED)
