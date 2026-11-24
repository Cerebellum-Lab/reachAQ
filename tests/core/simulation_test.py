import json
import logging
import threading

from reachaq.core.simulation import (
    SimSignalBusReader,
    SimSignalEvent,
    is_simulation,
    publish_signal,
    sim_camera_exposure_s,
    sim_camera_fps,
    sim_epoch,
)


def test_publish_without_bus_is_noop(monkeypatch, tmp_path):
    bus = tmp_path / "bus.jsonl"
    monkeypatch.setenv("REACHAQ_SIM_BUS", str(bus))
    publish_signal("tone1", 1.0, 2.0)
    assert bus.is_file()
    bus.unlink()
    # The same path, once the variable is gone, is never written again.
    monkeypatch.delenv("REACHAQ_SIM_BUS")
    publish_signal("tone1", 1.0, 2.0)
    assert not bus.exists()
    assert list(tmp_path.iterdir()) == []


def test_reader_missing_file_is_empty(tmp_path):
    assert SimSignalBusReader(tmp_path / "bus.jsonl").poll() == ()


def test_reader_returns_only_complete_lines(monkeypatch, tmp_path):
    bus = tmp_path / "bus.jsonl"
    reader = SimSignalBusReader(bus)
    bus.write_bytes(b'{"line": "tone1", "start": 1.0, "end"')
    assert reader.poll() == ()
    with open(bus, "ab") as f:
        f.write(b': 2.0, "value": 3.0}\n')
    assert reader.poll() == (SimSignalEvent("tone1", 1.0, 2.0, 3.0),)
    assert reader.poll() == ()


def test_reader_skips_malformed_lines(tmp_path, caplog):
    bus = tmp_path / "bus.jsonl"
    reader = SimSignalBusReader(bus)
    good = '{"line": "a", "start": 1.0, "end": 2.0}\n'
    bus.write_text(
        good
        + "not json\n"
        + '{"line": "x", "start": 1.0}\n'  # no end
        + '{"line": "x", "start": "soon", "end": 2.0}\n'
        + '["a", 1.0, 2.0]\n'
        + good.replace('"a"', '"b"'))
    with caplog.at_level(logging.WARNING, logger="reachaq.core.simulation"):
        assert reader.poll() == (SimSignalEvent("a", 1.0, 2.0), SimSignalEvent("b", 1.0, 2.0))
    assert len([r for r in caplog.records if "malformed" in r.getMessage()]) == 1


def test_reader_restarts_after_truncation(tmp_path):
    bus = tmp_path / "bus.jsonl"
    reader = SimSignalBusReader(bus)
    bus.write_text('{"line": "a", "start": 1.0, "end": 2.0}\n' * 3)
    assert len(reader.poll()) == 3
    bus.write_text('{"line": "b", "start": 3.0, "end": 4.0}\n')  # recreated, shorter
    assert reader.poll() == (SimSignalEvent("b", 3.0, 4.0),)


def test_concurrent_publishers_keep_lines_whole(monkeypatch, tmp_path):
    bus = tmp_path / "bus.jsonl"
    monkeypatch.setenv("REACHAQ_SIM_BUS", str(bus))

    def publish(line):
        for index in range(200):
            publish_signal(line, float(index), index + 0.5)

    threads = [threading.Thread(target=publish, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    raw = bus.read_text().splitlines()
    assert len(raw) == 400
    records = [json.loads(line) for line in raw]  # every line parses whole
    assert sorted(r["line"] for r in records) == ["a"] * 200 + ["b"] * 200
    assert len(SimSignalBusReader(bus).poll()) == 400


def test_publish_round_trip(monkeypatch, tmp_path):
    bus = tmp_path / "bus.jsonl"
    monkeypatch.setenv("REACHAQ_SIM_BUS", str(bus))
    reader = SimSignalBusReader(bus)
    publish_signal("a", 1.0, 2.0)
    publish_signal("b", 3.0, 4.0, 0.5)
    assert reader.poll() == (
        SimSignalEvent("a", 1.0, 2.0),
        SimSignalEvent("b", 3.0, 4.0, 0.5),
    )


def test_switch_reads_environment(monkeypatch):
    monkeypatch.delenv("REACHAQ_SIMULATION", raising=False)
    monkeypatch.delenv("REACHAQ_SIM_EPOCH", raising=False)
    assert not is_simulation()
    assert sim_epoch() == 0.0
    monkeypatch.setenv("REACHAQ_SIMULATION", "1")
    monkeypatch.setenv("REACHAQ_SIM_EPOCH", "12.5")
    assert is_simulation()
    assert sim_epoch() == 12.5
    monkeypatch.setenv("REACHAQ_SIMULATION", "0")
    assert not is_simulation()


def test_camera_grid_defaults_and_environment(monkeypatch):
    monkeypatch.delenv("REACHAQ_SIM_CAMERA_FPS", raising=False)
    monkeypatch.delenv("REACHAQ_SIM_CAMERA_EXPOSURE_S", raising=False)
    assert (sim_camera_fps(), sim_camera_exposure_s()) == (150.0, 0.002)
    monkeypatch.setenv("REACHAQ_SIM_CAMERA_FPS", "100")
    monkeypatch.setenv("REACHAQ_SIM_CAMERA_EXPOSURE_S", "0.004")
    assert (sim_camera_fps(), sim_camera_exposure_s()) == (100.0, 0.004)
