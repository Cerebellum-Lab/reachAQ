import pytest
import threading
import time

from autotrainer.device.rfid_reader import (
    RfidDuplicateSuppressor,
    RfidFrameParser,
    RfidTagRead,
    RfidReaderService,
    RfidReaderState,
    payload_checksum,
    make_rfid_frame,
)


TAG_A = "D4D47231005A30010000000000"
TAG_B = "B4D47231005A30010000000000"


@pytest.mark.parametrize("split_at", range(31))
def test_parser_accepts_captured_protocol_at_every_fragment_boundary(split_at):
    frame = make_rfid_frame(TAG_A)
    parser = RfidFrameParser()

    events = parser.feed(frame[:split_at], monotonic_time=1.0)
    events += parser.feed(frame[split_at:], monotonic_time=2.0)

    assert [event.rfid for event in events] == [TAG_A]
    assert events[0].raw_frame == frame
    assert parser.stats.valid_frames == 1


def test_parser_handles_noise_and_multiple_alternating_tags():
    parser = RfidFrameParser()
    events = parser.feed(b"noise" + make_rfid_frame(TAG_A) + make_rfid_frame(TAG_B))

    assert [event.rfid for event in events] == [TAG_A, TAG_B]
    assert parser.stats.discarded_bytes == 5


def test_parser_rejects_checksum_and_resynchronizes():
    broken = bytearray(make_rfid_frame(TAG_A))
    broken[27] ^= 1
    parser = RfidFrameParser()

    events = parser.feed(bytes(broken) + make_rfid_frame(TAG_B))

    assert [event.rfid for event in events] == [TAG_B]
    assert parser.stats.invalid_checksum_frames == 1


def test_parser_recovers_after_bounded_buffer_overflow():
    parser = RfidFrameParser(maximum_buffer_bytes=60)
    parser.feed(b"x" * 100)

    events = parser.feed(make_rfid_frame(TAG_A))

    assert [event.rfid for event in events] == [TAG_A]
    assert parser.stats.buffer_overflows == 1


def test_parser_normalizes_lowercase_payload_after_checksum_validation():
    payload = TAG_A.lower().encode("ascii")
    checksum = payload_checksum(payload)
    frame = b"\x02" + payload + bytes((checksum, checksum ^ 0xFF, 0x03))

    assert RfidFrameParser().feed(frame)[0].rfid == TAG_A


def test_duplicate_suppression_is_per_tag_and_monotonic():
    suppressor = RfidDuplicateSuppressor(window_seconds=1.0)
    event = lambda tag, timestamp: RfidTagRead(tag, b"", timestamp)

    assert suppressor.accept(event(TAG_A, 1.0))
    assert not suppressor.accept(event(TAG_A, 1.5))
    assert suppressor.accept(event(TAG_B, 1.6))
    assert suppressor.accept(event(TAG_A, 1.7))
    assert suppressor.accept(event(TAG_A, 2.7))


@pytest.mark.parametrize("value", ["", "abc", "G" * 26, "A" * 25, "A" * 27])
def test_frame_builder_rejects_noncanonical_values(value):
    with pytest.raises(ValueError):
        make_rfid_frame(value)


class _FakeSerial:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False

    def read(self, _size):
        if self.closed:
            return b""
        try:
            chunk = next(self.chunks)
        except StopIteration:
            time.sleep(0.005)
            return b""
        if isinstance(chunk, Exception):
            raise chunk
        return chunk

    def close(self):
        self.closed = True


def test_reader_reconnects_after_disconnect_and_shuts_down_cleanly():
    first = _FakeSerial([OSError("unplugged")])
    second = _FakeSerial([make_rfid_frame(TAG_A)])
    ports = iter((first, second))
    received = []
    statuses = []
    arrived = threading.Event()

    service = RfidReaderService(
        on_tag=lambda event: (received.append(event), arrived.set()),
        on_status=statuses.append,
        serial_factory=lambda _device, _baud: next(ports),
        reconnect_initial_seconds=0.001,
        reconnect_max_seconds=0.001,
    )
    service.start()
    assert arrived.wait(1)
    service.stop()

    assert [event.rfid for event in received] == [TAG_A]
    assert any(status.state is RfidReaderState.RECONNECTING for status in statuses)
    assert statuses[-1].state is RfidReaderState.STOPPED
    assert first.closed and second.closed
    assert not service.running


def test_reader_enters_failed_state_after_bounded_connection_attempts():
    statuses = []
    finished = threading.Event()

    def status_changed(status):
        statuses.append(status)
        if status.state is RfidReaderState.FAILED:
            finished.set()

    service = RfidReaderService(
        on_tag=lambda _event: None,
        on_status=status_changed,
        serial_factory=lambda _device, _baud: (_ for _ in ()).throw(
            OSError("missing reader")
        ),
        reconnect_initial_seconds=0,
        reconnect_max_seconds=0,
        maximum_reconnect_attempts=2,
    )
    service.start()
    assert finished.wait(1)

    assert statuses[-1].state is RfidReaderState.FAILED
    service.stop()
