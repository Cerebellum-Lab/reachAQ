from autotrainer.core.simulation import (
    SimSignalBusReader,
    SimSignalEvent,
    is_simulation,
    publish_signal,
    sim_epoch,
)


def test_publish_without_bus_is_noop(monkeypatch, tmp_path):
    monkeypatch.delenv("REACHAQ_SIM_BUS", raising=False)
    publish_signal("tone1", 1.0, 2.0)
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
