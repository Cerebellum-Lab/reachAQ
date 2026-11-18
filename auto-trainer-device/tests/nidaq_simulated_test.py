import time

import numpy
import pytest

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)
from autotrainer.core.simulation import publish_signal
from autotrainer.device import nidaq_input_backend
from autotrainer.device.nidaq_input_backend import (
    NidaqInputBackend,
    make_nidaq_input_backend,
)
from autotrainer.device.nidaq_simulated import SIM_NIDAQ_DEVICE, SimulatedNidaqStream

RATE = 10_000.0
CHUNK = 500


@pytest.fixture(autouse=True)
def _clean_sim_env(monkeypatch):
    for name in (
        "REACHAQ_SIMULATION", "REACHAQ_SIM_BUS", "REACHAQ_SIM_EPOCH",
        "REACHAQ_SIM_CAMERA_FPS", "REACHAQ_SIM_CAMERA_EXPOSURE_S",
        "REACHAQ_SIM_LINE_ALIASES",
    ):
        monkeypatch.delenv(name, raising=False)


def _configuration(*channels):
    return NidaqSignalStreamConfiguration(
        tuple(
            NidaqSignalChannelConfiguration(name, f"{SIM_NIDAQ_DEVICE}/{line}", kind)
            for name, line, kind in channels
        ),
        is_enabled=True,
        sample_rate_hz=RATE,
        read_chunk_size=CHUNK,
    )


def _read(stream, chunks):
    blocks = [stream.read_chunk() for _ in range(chunks)]
    stream.close()
    return blocks


def _joined(blocks, name):
    return numpy.concatenate([numpy.asarray(block.values[name]) for block in blocks])


@pytest.mark.parametrize("fps, exposure", [(None, None), ("100", "0.003")])
def test_cam_frames_rises_at_each_grid_exposure(monkeypatch, fps, exposure):
    grid_epoch = time.perf_counter() - 0.0013
    monkeypatch.setenv("REACHAQ_SIM_EPOCH", repr(grid_epoch))
    if fps is not None:
        monkeypatch.setenv("REACHAQ_SIM_CAMERA_FPS", fps)
        monkeypatch.setenv("REACHAQ_SIM_CAMERA_EXPOSURE_S", exposure)
    period = 1.0 / float(fps or 150)
    exposure_s = float(exposure or 0.002)

    blocks = _read(SimulatedNidaqStream(
        _configuration(("cam_frames", "port0/line0", "digital"))), 3)
    samples = _joined(blocks, "cam_frames")
    epoch_perf = blocks[0].epoch_perf_time

    rises = numpy.flatnonzero((samples[1:] == 1.0) & (samples[:-1] == 0.0)) + 1
    # 150 ms of stream holds 22 exposures at 150 fps, 15 at 100 fps.
    assert len(rises) >= int(0.15 / period) - 1
    for index in rises:
        t = epoch_perf + index / RATE
        k = round((t - grid_epoch) / period)
        assert abs(t - (grid_epoch + k * period)) <= 1.0 / RATE
    falls = numpy.flatnonzero((samples[1:] == 0.0) & (samples[:-1] == 1.0)) + 1
    widths = [fall - rise for rise, fall in zip(rises, falls[falls > rises[0]])]
    assert all(abs(width - exposure_s * RATE) <= 1 for width in widths)


def test_bus_window_drives_named_digital_line(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHAQ_SIM_BUS", str(tmp_path / "signals.jsonl"))
    stream = SimulatedNidaqStream(_configuration(
        ("tone1", "port0/line1", "digital"),
        ("tone2", "port0/line2", "digital"),
    ))
    stream.start()
    start = time.perf_counter() + 0.02
    publish_signal("tone1", start, start + 0.050)

    blocks = _read(stream, 3)
    tone1 = _joined(blocks, "tone1")
    assert set(numpy.unique(tone1)) == {0.0, 1.0}
    assert abs(int(tone1.sum()) - 0.050 * RATE) <= 1
    first_high = int(numpy.flatnonzero(tone1)[0])
    assert abs(blocks[0].epoch_perf_time + first_high / RATE - start) <= 1.0 / RATE
    assert not _joined(blocks, "tone2").any()


def test_aliased_analog_line_takes_the_bus_value(monkeypatch, tmp_path):
    monkeypatch.setenv("REACHAQ_SIM_BUS", str(tmp_path / "signals.jsonl"))
    monkeypatch.setenv("REACHAQ_SIM_LINE_ALIASES", '{"laser1_trigger": "board_stim2"}')
    stream = SimulatedNidaqStream(_configuration(
        ("laser1_trigger", "ai0", "analog"),
        ("laser1_diode", "ai1", "analog"),
    ))
    stream.start()
    start = time.perf_counter() + 0.01
    publish_signal("board_stim2", start, start + 0.020, 3.3)
    publish_signal("laser1_trigger", start, start + 0.020, 5.0)  # not its line now

    blocks = _read(stream, 2)
    trigger = _joined(blocks, "laser1_trigger")
    high = trigger > 1.0
    assert abs(int(high.sum()) - 0.020 * RATE) <= 1
    assert numpy.allclose(trigger[high], 3.3, atol=0.005)
    assert numpy.abs(trigger[~high]).max() < 0.005
    assert 0.0002 < trigger[~high].std() < 0.001  # 0.5 mV of noise, not silence
    assert numpy.abs(_joined(blocks, "laser1_diode")).max() < 0.005


def test_read_stamps_before_samples_exist():
    stream = SimulatedNidaqStream(_configuration(("cam_frames", "port0/line0", "digital")))
    for _ in range(3):
        block = stream.read_chunk()
        returned = time.perf_counter()
        last_sample_exists = (
            block.epoch_perf_time + (block.sample_index + block.sample_count) / RATE)
        assert block.perf_time < last_sample_exists <= returned
    stream.close()


def test_sample_index_contiguous():
    blocks = _read(SimulatedNidaqStream(
        _configuration(("cam_frames", "port0/line0", "digital"))), 4)
    assert [block.sample_index for block in blocks] == [0, CHUNK, 2 * CHUNK, 3 * CHUNK]
    assert all(block.sample_count == CHUNK for block in blocks)
    assert {block.epoch_perf_time for block in blocks} == {blocks[0].epoch_perf_time}


def test_restart_after_close_starts_at_sample_zero():
    stream = SimulatedNidaqStream(_configuration(("cam_frames", "port0/line0", "digital")))
    first = _read(stream, 2)
    stream.start()
    block = stream.read_chunk()
    stream.close()
    assert block.sample_index == 0
    assert block.epoch_perf_time > first[-1].epoch_perf_time


def test_verify_tasks_and_telemetry_match_the_real_shape():
    stream = SimulatedNidaqStream(_configuration(("cam_frames", "port0/line0", "digital")))
    assert stream.verify_tasks(commit=True) == (f"simulated:{SIM_NIDAQ_DEVICE}",)
    stream.read_chunk()
    telemetry = stream.read_telemetry
    assert set(telemetry) == {"blocks", "read_seconds", "read_failures", "short_reads"}
    assert telemetry["blocks"] == 1
    stream.close()


def test_factory_returns_hardware_class_when_not_simulating(monkeypatch):
    constructed = []

    class Controller:
        def __init__(self, configuration, *, timing_plan):
            constructed.append((configuration, timing_plan))

    monkeypatch.setattr(nidaq_input_backend, "NidaqSignalStreamController", Controller)
    backend = make_nidaq_input_backend("configuration", timing_plan="plan")
    assert isinstance(backend, Controller)
    assert constructed == [("configuration", "plan")]


def test_factory_returns_simulated_stream_when_simulating(monkeypatch):
    monkeypatch.setenv("REACHAQ_SIMULATION", "1")

    class Controller:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("the hardware class must not be constructed")

    backend = make_nidaq_input_backend(
        _configuration(("cam_frames", "port0/line0", "digital")),
        hardware_class=Controller,
    )
    assert isinstance(backend, SimulatedNidaqStream)
    assert isinstance(backend, NidaqInputBackend)
