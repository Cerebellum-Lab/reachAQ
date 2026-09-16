from pathlib import Path

from autotrainer.core.configuration.demo_sources import DemoSources

from tools.acquisition.model.app_model import AppModel


class _FakeModel:
    """Exercises the tagging helpers without constructing a whole AppModel."""

    _demo_metadata_block = AppModel._demo_metadata_block
    _write_demo_marker = AppModel._write_demo_marker

    def __init__(self, sources=None):
        self._demo_sources = sources

    @property
    def demo_sources(self):
        return self._demo_sources

    @property
    def is_demo_mode(self):
        return self._demo_sources is not None


def test_no_demo_block_when_not_in_demo_mode():
    assert _FakeModel()._demo_metadata_block() is None


def test_demo_block_records_sources_and_fps(tmp_path):
    sources = DemoSources(
        fps=150.0, loop=True,
        cameras={"left": tmp_path / "l.mp4", "right": tmp_path / "r.mp4"},
    )

    block = _FakeModel(sources)._demo_metadata_block()

    assert block["active"] is True
    assert block["fps"] == 150.0
    assert block["sources"]["left"] == (tmp_path / "l.mp4").as_posix()
    assert block["sources"]["right"] == (tmp_path / "r.mp4").as_posix()


def test_marker_file_is_written_in_demo_mode(tmp_path):
    sources = DemoSources(fps=150.0, loop=True, cameras={"left": tmp_path / "l.mp4"})

    _FakeModel(sources)._write_demo_marker(tmp_path)

    marker = tmp_path / "DEMO"
    assert marker.is_file()
    text = marker.read_text()
    assert "not experimental data" in text
    assert "l.mp4" in text


def test_no_marker_file_when_not_in_demo_mode(tmp_path):
    _FakeModel()._write_demo_marker(tmp_path)

    assert not (tmp_path / "DEMO").exists()
