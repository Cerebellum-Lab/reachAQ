import numpy as np
import pytest

from autotrainer.core import ObservableObject, SystemStatusMessageKind
from autotrainer.device import PressureReading, Target
from tools.acquisition.model.pressure_stream_model import (
    PressureStreamModel,
    counts_to_volts,
)


class _EventSource(ObservableObject):
    def __init__(self, *event_names):
        super().__init__(event_names)


def _reading(instance: int, pressure: int, *, event_perf_time=None) -> PressureReading:
    reading = PressureReading(
        target=Target.PELLET_DEVICE,
        instance=instance,
        pressure=pressure,
    )
    reading.event_perf_time = event_perf_time
    return reading


def _send(handler, reading, perf_time: float) -> None:
    handler.decoded_message_received(
        SystemStatusMessageKind.PRESSURE_READING,
        reading,
        perf_time,
        perf_time + 1_000.0,
    )


@pytest.fixture
def handler():
    return _EventSource("decoded_message_received")


@pytest.fixture
def model(handler):
    model = PressureStreamModel(handler)
    yield model
    model.close()


def test_readings_are_routed_to_the_buffer_for_their_instance(handler, model):
    _send(handler, _reading(0, 4095), 10.0)
    _send(handler, _reading(1, 458), 10.1)

    frame = model.snapshot()

    assert frame is not None
    _, j11_counts = frame.series[0]
    _, j21_counts = frame.series[1]
    assert j11_counts.tolist() == [4095.0]
    assert j21_counts.tolist() == [458.0]


def test_pressure_is_buffered_as_the_raw_adc_count_not_volts(handler, model):
    _send(handler, _reading(0, 4095), 10.0)

    _, counts = model.snapshot().series[0]

    assert counts.tolist() == [4095.0]


def test_other_message_kinds_are_ignored(handler, model):
    handler.decoded_message_received(
        SystemStatusMessageKind.STIMULUS_INPUTS,
        {"tone1": True},
        10.0,
        1_010.0,
    )

    assert model.snapshot() is None


def test_snapshot_is_none_before_any_reading_arrives(model):
    assert model.snapshot() is None


def test_newest_sample_sits_at_x_zero_and_history_runs_negative(handler, model):
    _send(handler, _reading(0, 100), 10.0)
    _send(handler, _reading(0, 200), 11.0)
    _send(handler, _reading(0, 300), 12.0)

    seconds, _ = model.snapshot().series[0]

    assert seconds.tolist() == [-2.0, -1.0, 0.0]


def test_both_graphs_share_one_time_origin(handler, model):
    _send(handler, _reading(0, 100), 10.0)
    _send(handler, _reading(1, 200), 12.0)

    frame = model.snapshot()

    assert frame.latest_perf_time == 12.0
    assert frame.series[0][0].tolist() == [-2.0]
    assert frame.series[1][0].tolist() == [0.0]


def test_board_event_time_is_preferred_over_host_receive_time(handler, model):
    _send(handler, _reading(0, 100, event_perf_time=50.0), 10.0)

    assert model.snapshot().latest_perf_time == 50.0


def test_host_receive_time_is_used_when_the_board_event_time_is_not_finite(
    handler, model,
):
    _send(handler, _reading(0, 100, event_perf_time=float("nan")), 10.0)

    assert model.snapshot().latest_perf_time == 10.0


def test_samples_older_than_the_window_are_dropped(handler, model):
    _send(handler, _reading(0, 100), 0.0)
    _send(handler, _reading(0, 200), 5.0)
    _send(handler, _reading(0, 300), 10.5)

    seconds, counts = model.snapshot().series[0]

    assert counts.tolist() == [200.0, 300.0]
    assert seconds.tolist() == [-5.5, 0.0]


def test_clear_discards_buffered_history(handler, model):
    _send(handler, _reading(0, 100), 10.0)

    model.clear()

    assert model.snapshot() is None


def test_closing_stops_further_buffering(handler):
    model = PressureStreamModel(handler)
    model.close()

    _send(handler, _reading(0, 100), 10.0)

    assert model.snapshot() is None


def test_readings_from_an_unknown_instance_are_ignored(handler, model):
    _send(handler, _reading(7, 100), 10.0)

    assert model.snapshot() is None


@pytest.mark.parametrize("counts, volts", [(0, 0.0), (4095, 3.3), (2048, 1.6504029)])
def test_counts_convert_to_volts_across_the_adc_range(counts, volts):
    assert counts_to_volts(np.asarray([counts])) == pytest.approx([volts], abs=1e-5)
