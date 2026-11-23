"""The calibration library: immutable entries, one active pointer, an append-only decision log.

Sessions name a calibration by id, so an entry must not change once it is in place: adding
copies a finished folder in through <id>.partial, the pointer is replaced whole, and what
happens to an entry afterwards (activated, installed over a failed report) is a line in
decisions.jsonl. Old folders (format 2, legacy, dated backups) are copied in and never touched.
"""

import hashlib
import importlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from autotrainer.core import ProjectInfo
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.inference.analysis.intersession_process import intersession_process, resolve_session_calibration
from autotrainer.inference.multires_calibration import library, store
from autotrainer.inference.multires_calibration.geometry import PresetGeometry, presets_for_cameras, presets_from_base
from autotrainer.inference.multires_calibration.library import (
    ACTIVE_FILE,
    DECISIONS_FILE,
    active_id,
    add_entry,
    entry_path,
    import_existing,
    library_root,
    list_entries,
    record_decision,
    set_active,
    settings_path,
)
from autotrainer.inference.multires_calibration.report import QualityReport
from autotrainer.inference.multires_calibration.store import CalibrationGeometryError, CalibrationResolutionError
from multires_synthetic import synthetic_rig

intersession_module = importlib.import_module("autotrainer.inference.analysis.intersession_process")

NAME = DEFAULT_3D_CALIB_DIR_NAME
BASE = PresetGeometry(4, 256, 256, 52, 6)
RIG = synthetic_rig(noise_px=0.0)
RIG_AB = synthetic_rig(noise_px=0.0, cameras=("cam_a", "cam_b"))
FORMAT2_FIXTURE = Path(__file__).parent / "multires_format2_calibration"
FORMAT2_ID = "2692b0b23fa2499f8189447f9631aa4a"
_REPORT = QualityReport(created="2026-10-20T12:00:00-06:00", verdict="pass", metrics=[], thresholds={}, notes=[])


class _Stop(Exception):
    """Raised by a stand-in for process_raw_data once it has seen its arguments."""


def _built(folder: Path, rig=RIG) -> str:
    """A finished format-3 calibration in `folder`, as the capture tool leaves it; its id."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    live = {cam: BASE for cam in rig.truth.cameras}
    return store.write_calibration(
        folder, fit=rig.truth, preset_sets=presets_for_cameras(live), board=rig.board, report=_REPORT, capture={},
    ).id


def _hashes(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _decisions(root: Path) -> list:
    return [json.loads(line) for line in (root / DECISIONS_FILE).read_text(encoding="utf-8").splitlines()]


def _legacy(folder: Path, made: datetime) -> Path:
    pickle_path = folder / "camera_matrix" / "stereo_params.pickle"
    pickle_path.parent.mkdir(parents=True)
    pickle_path.write_bytes(b"legacy pickle")
    (folder / "calibration_userset.yaml").write_text("camera_names: [left, right]\n")
    os.utime(pickle_path, (made.timestamp(), made.timestamp()))
    return folder


def _config(tmp_path: Path) -> Path:
    """A configuration directory as a rig had it before the library: a format-2 active
    folder, a format-3 backup and a legacy backup."""
    config = tmp_path / "Autotrainer"
    config.mkdir()
    shutil.copytree(FORMAT2_FIXTURE, config / NAME)
    _built(config / f"{NAME}_20260701_120000")
    _legacy(config / f"{NAME}_20260303_090815", datetime(2026, 1, 10))
    return config


# --- add, activate ---------------------------------------------------------------------

def test_add_then_activate(tmp_path):
    root = library_root(tmp_path / "config")
    built = tmp_path / "session" / f"{NAME}.new"
    calibration_id = _built(built)
    before = _hashes(built)

    assert add_entry(root, built) == calibration_id
    assert entry_path(root, calibration_id) == root / calibration_id
    assert _hashes(root / calibration_id) == before and _hashes(built) == before
    assert not list(root.glob("*.partial"))
    assert active_id(root) is None

    set_active(root, calibration_id, by="ben")
    assert active_id(root) == calibration_id
    assert yaml.safe_load((root / ACTIVE_FILE).read_text())["by"] == "ben"
    assert [(d["event"], d["id"], d["by"]) for d in _decisions(root)] == [("set_active", calibration_id, "ben")]
    assert [(e.id, e.active, e.verdict, e.cameras, e.uncalibrated) for e in list_entries(root)] == [
        (calibration_id, True, "pass", ["left", "right"], {})]


def test_add_refuses_an_existing_id_with_other_content(tmp_path):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    before = _hashes(root / calibration_id)
    (built / store.INDEX_FILE).write_text((built / store.INDEX_FILE).read_text() + "# edited\n")
    with pytest.raises(FileExistsError, match=calibration_id):
        add_entry(root, built)
    assert _hashes(root / calibration_id) == before
    assert not list(root.glob("*.partial"))


def test_a_retry_of_an_identical_entry_is_kept_and_can_be_activated(tmp_path):
    # A crash between add_entry and set_active left the entry; the retry adds the same build.
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    before = _hashes(root / calibration_id)

    assert add_entry(root, built) == calibration_id
    set_active(root, calibration_id, by="ben")

    assert _hashes(root / calibration_id) == before
    assert active_id(root) == calibration_id
    assert not list(root.glob("*.partial"))


def test_add_flushes_the_entry_before_it_is_renamed_into_place(tmp_path, monkeypatch):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    _built(built)
    events = []
    real_fsync, real_replace = library.os.fsync, library.os.replace
    monkeypatch.setattr(library.os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(library.os, "replace", lambda a, b: (events.append(("replace", Path(b).name)),
                                                             real_replace(a, b))[1])
    calibration_id = add_entry(root, built)
    n_files = sum(1 for p in built.rglob("*") if p.is_file())
    rename = events.index(("replace", calibration_id))
    assert events[:rename].count("fsync") >= n_files

    events.clear()
    set_active(root, calibration_id, by="ben")
    assert events.index("fsync") < events.index(("replace", ACTIVE_FILE)), "active.yaml's temp file is flushed first"


def test_an_entry_with_an_empty_index_is_not_an_entry(tmp_path):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    (root / calibration_id / store.INDEX_FILE).write_bytes(b"")
    with pytest.raises(CalibrationResolutionError, match="empty"):
        entry_path(root, calibration_id)


def test_add_refuses_a_folder_that_is_not_a_calibration(tmp_path):
    (tmp_path / "junk").mkdir()
    with pytest.raises(ValueError, match="not a multi-resolution calibration"):
        add_entry(library_root(tmp_path), tmp_path / "junk")


def test_a_failed_add_leaves_no_partial_and_no_entry(tmp_path, monkeypatch):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    _built(built)

    def refuse(src, dst):
        raise PermissionError("no")

    monkeypatch.setattr(library.os, "replace", refuse)
    with pytest.raises(PermissionError):
        add_entry(root, built)
    assert list(root.iterdir()) == []


# --- the pointer -----------------------------------------------------------------------

@pytest.mark.parametrize("content", [None, "{{", "- a\n- list\n", "id: ''\n", "id: not-an-entry\n"],
                         ids=["missing", "corrupt", "not_a_mapping", "empty_id", "dangling"])
def test_active_id_is_none_for_missing_corrupt_or_dangling_pointer(tmp_path, content, caplog):
    root = library_root(tmp_path)
    root.mkdir(parents=True)
    if content is not None:
        (root / ACTIVE_FILE).write_text(content)
    with caplog.at_level(logging.WARNING):
        assert active_id(root) is None
    assert any("no active calibration" in r.getMessage() for r in caplog.records)
    assert list_entries(root) == []


@pytest.mark.parametrize("content", [b"id: %s\nat: 2026-13-45\n", b"\x00\xff\xfe\x81garbage\xc3", b"- id: %s\n"],
                         ids=["bad_timestamp", "binary_garbage", "list_at_top_level"])
def test_active_id_never_raises(tmp_path, content, caplog):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    (root / ACTIVE_FILE).write_bytes(content.replace(b"%s", calibration_id.encode()))
    with caplog.at_level(logging.WARNING):
        assert active_id(root) is None
    assert any("no active calibration" in r.getMessage() for r in caplog.records)
    assert [e.active for e in list_entries(root)] == [False]


def test_a_failed_decision_puts_the_previous_pointer_back(tmp_path, monkeypatch):
    root = library_root(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    first_id, second_id = _built(first), _built(second)
    add_entry(root, first)
    add_entry(root, second)
    set_active(root, first_id, by="ben")
    old_pointer = (root / ACTIVE_FILE).read_bytes()
    old_decisions = (root / DECISIONS_FILE).read_bytes()

    def refuse(root, record):
        raise OSError("decisions.jsonl: read-only file system")

    monkeypatch.setattr(library, "_append_decision", refuse)
    with pytest.raises(OSError, match="read-only file system"):
        set_active(root, second_id, by="ben")
    monkeypatch.undo()

    assert (root / ACTIVE_FILE).read_bytes() == old_pointer
    assert active_id(root) == first_id
    assert (root / DECISIONS_FILE).read_bytes() == old_decisions
    assert not list(root.glob(f"{ACTIVE_FILE}.*.tmp"))


def test_a_failed_first_decision_leaves_no_pointer(tmp_path, monkeypatch):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    monkeypatch.setattr(library, "_append_decision", lambda root, record: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        set_active(root, calibration_id, by="ben")
    assert not (root / ACTIVE_FILE).exists()


def test_set_active_is_atomic(tmp_path, monkeypatch):
    root = library_root(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    first_id, second_id = _built(first), _built(second)
    add_entry(root, first)
    add_entry(root, second)
    set_active(root, first_id, by="ben")
    old_text = (root / ACTIVE_FILE).read_text()
    old_decisions = (root / DECISIONS_FILE).read_text()

    def refuse(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(library.os, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        set_active(root, second_id, by="ben")
    monkeypatch.undo()

    assert (root / ACTIVE_FILE).read_text() == old_text
    assert active_id(root) == first_id
    assert (root / DECISIONS_FILE).read_text() == old_decisions, "no decision for a pointer that was not moved"
    assert not (root / f"{ACTIVE_FILE}.tmp").exists()


def test_set_active_refuses_an_unknown_entry(tmp_path):
    root = library_root(tmp_path)
    root.mkdir()
    with pytest.raises(CalibrationResolutionError):
        set_active(root, "0" * 32, by="ben")
    assert not (root / ACTIVE_FILE).exists()


def test_decisions_append_only(tmp_path):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    set_active(root, calibration_id, by="ben")
    first_line = (root / DECISIONS_FILE).read_bytes()
    record_decision(root, calibration_id, "override", {"confirmed": True, "by": "ben"})
    record_decision(root, calibration_id, "note", {"text": "remounted"})

    assert (root / DECISIONS_FILE).read_bytes().startswith(first_line)
    records = _decisions(root)
    assert [r["event"] for r in records] == ["set_active", "override", "note"]
    assert records[1]["detail"] == {"confirmed": True, "by": "ben"}
    assert records[1]["id"] == calibration_id and "at" in records[1]
    with pytest.raises(CalibrationResolutionError):
        record_decision(root, "f" * 32, "note", {})
    assert len(_decisions(root)) == 3


def test_a_decision_after_a_torn_last_line_starts_a_new_line(tmp_path):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    (root / DECISIONS_FILE).write_text('{"event": "set_act')
    record_decision(root, calibration_id, "note", {})
    lines = (root / DECISIONS_FILE).read_text().splitlines()
    assert lines[0] == '{"event": "set_act' and json.loads(lines[1])["event"] == "note"


def test_override_is_a_decision_not_an_edit(tmp_path):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    before = _hashes(root / calibration_id)

    set_active(root, calibration_id, by="ben")
    record_decision(root, calibration_id, "override", {"confirmed": True, "by": "ben", "at_confirm": "2026-11-20"})

    assert _hashes(root / calibration_id) == before
    assert store.load_index(root / calibration_id).override is None
    assert _decisions(root)[-1]["event"] == "override"


# --- listing ---------------------------------------------------------------------------

def test_list_skips_derived_dot_names_partials_and_files(tmp_path):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    (root / ".derived" / calibration_id / "abc").mkdir(parents=True)
    (root / ".hidden").mkdir()
    shutil.copytree(built, root / "deadbeef.partial")
    (root / "settings.yaml").write_text("{}\n")
    assert settings_path(root) == root / "settings.yaml"
    assert [e.id for e in list_entries(root)] == [calibration_id]


def test_list_names_a_two_d_only_camera_with_its_reason(tmp_path):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    index_path = built / store.INDEX_FILE
    data = yaml.safe_load(index_path.read_text())
    data["cameras"].append({"name": "stimCam", "serial": "1", "camera_id": 2, "calibrated": False,
                            "reason": "too few board views: 4 of 12 needed"})
    index_path.write_text(yaml.safe_dump(data, sort_keys=False))
    add_entry(root, built)
    entry, = list_entries(root)
    assert entry.id == calibration_id and entry.cameras == ["left", "right", "stimCam"]
    assert entry.uncalibrated == {"stimCam": "too few board views: 4 of 12 needed"}


def test_an_unreadable_entry_is_left_out_with_a_warning(tmp_path, caplog):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    (root / "baddie").mkdir()
    (root / "baddie" / "calibration.yaml").write_text("{{")
    with caplog.at_level(logging.WARNING):
        assert [e.id for e in list_entries(root)] == [calibration_id]
    assert any("baddie" in r.getMessage() for r in caplog.records)


# --- DCS offsets beside an entry -------------------------------------------------------

def test_each_dcs_run_keeps_its_own_offsets_file_and_the_newest_is_current(tmp_path):
    from datetime import timezone
    from autotrainer.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME

    root = library_root(tmp_path)
    assert library.OFFSETS_FILE_NAME == DEFAULT_CAM_OFFSET_FILE_NAME
    assert library.current_offsets_path(root, "abc") is None
    first = library.write_offsets(root, "abc", b"first", now=datetime(2026, 11, 20, 9, 0, 0, tzinfo=timezone.utc))
    second = library.write_offsets(root, "abc", b"second", now=datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc))

    assert first.name == "camera_offsets-20261120T090000Z.pkl" and first.read_bytes() == b"first"
    assert second.name == "camera_offsets-20261120T100000Z.pkl"
    assert library.current_offsets_path(root, "abc") == second
    assert first.parent == second.parent == root / ".offsets" / "abc"
    # Two runs in one second, or a clock set back, still never overwrite and stay in name order.
    third = library.write_offsets(root, "abc", b"third", now=datetime(2026, 11, 20, 10, 0, 0, tzinfo=timezone.utc))
    earlier = library.write_offsets(root, "abc", b"back", now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert third.name == "camera_offsets-20261120T100001Z.pkl" and second.read_bytes() == b"second"
    assert earlier.name == "camera_offsets-20261120T100002Z.pkl"
    assert library.current_offsets_path(root, "abc") == earlier
    assert len(list((root / ".offsets" / "abc").iterdir())) == 4


def test_a_plain_offsets_file_alone_in_the_store_is_current(tmp_path):
    root = library_root(tmp_path)
    plain = root / ".offsets" / "abc" / "camera_offsets.pkl"
    plain.parent.mkdir(parents=True)
    plain.write_bytes(b"from before the files were versioned")
    assert library.current_offsets_path(root, "abc") == plain
    newer = library.write_offsets(root, "abc", b"new")
    assert library.current_offsets_path(root, "abc") == newer and plain.is_file()


# --- importing what predates the library ----------------------------------------------

def test_import_keeps_format2_ids_and_marks_active(tmp_path):
    config = _config(tmp_path)
    backup_id = store.load_index(config / f"{NAME}_20260701_120000").id

    added = import_existing(config)

    root = library_root(config)
    assert FORMAT2_ID in added and backup_id in added and len(added) == 3
    assert active_id(root) == FORMAT2_ID
    entries = {e.id: e for e in list_entries(root)}
    assert set(entries) == set(added)
    assert [e.id for e in entries.values() if e.active] == [FORMAT2_ID]
    assert store.load_index(root / FORMAT2_ID).format == 2
    legacy_id, = set(added) - {FORMAT2_ID, backup_id}
    legacy = entries[legacy_id]
    assert legacy.created == datetime(2026, 1, 10) and legacy.verdict == "unknown"
    marker = yaml.safe_load((legacy.path / "calibration.yaml").read_text())
    assert marker["format"] == "legacy" and marker["id"] == legacy_id and marker["source"] == f"{NAME}_20260303_090815"
    assert (legacy.path / "camera_matrix" / "stereo_params.pickle").read_bytes() == b"legacy pickle"
    assert [r["event"] for r in _decisions(root)].count("import") == 3


def test_import_is_idempotent(tmp_path):
    config = _config(tmp_path)
    first = import_existing(config)
    root = library_root(config)
    state = _hashes(root)
    set_active(root, [i for i in first if i != FORMAT2_ID][0], by="ben")
    after_set = _hashes(root)

    assert import_existing(config) == []
    # Nothing changed, not even the pointer ben moved.
    assert _hashes(root) == after_set != state
    assert len(list_entries(root)) == 3


def test_import_leaves_old_folders_unchanged(tmp_path):
    config = _config(tmp_path)
    folders = [p for p in config.iterdir() if p.is_dir()]
    before = {p.name: _hashes(p) for p in folders}
    mtimes = {p.name: (p / "camera_matrix" / "stereo_params.pickle").stat().st_mtime
              for p in folders if (p / "camera_matrix" / "stereo_params.pickle").exists()}

    import_existing(config)

    assert {p.name: _hashes(p) for p in folders} == before
    assert sorted(p.name for p in config.iterdir()) == sorted([p.name for p in folders] + ["calibrations"])
    for name, mtime in mtimes.items():
        assert (config / name / "camera_matrix" / "stereo_params.pickle").stat().st_mtime == mtime


def test_import_skips_a_corrupt_folder_and_keeps_going(tmp_path, caplog):
    config = _config(tmp_path)
    bad = config / f"{NAME}_20260801_000000"
    bad.mkdir()
    (bad / "calibration.yaml").write_text("{{")
    with caplog.at_level(logging.WARNING):
        added = import_existing(config)
    assert len(added) == 3
    assert any(bad.name in r.getMessage() for r in caplog.records)


def test_import_with_nothing_to_import_makes_an_empty_library(tmp_path):
    assert import_existing(tmp_path) == []
    assert active_id(library_root(tmp_path)) is None


def test_a_legacy_folder_renamed_to_a_backup_is_not_imported_twice(tmp_path):
    config = tmp_path / "Autotrainer"
    config.mkdir()
    _legacy(config / NAME, datetime(2026, 7, 7))
    legacy_id, = import_existing(config)
    root = library_root(config)
    # The legacy install's next calibration moves the active folder to a dated backup.
    (config / NAME).rename(config / f"{NAME}_20260801_120000")

    assert import_existing(config) == []
    assert [e.id for e in list_entries(root)] == [legacy_id]
    assert [(r["event"], r["id"]) for r in _decisions(root) if r["event"] == "import"] == [("import", legacy_id)]
    assert active_id(root) == legacy_id


def test_a_roll_forward_import_never_sets_the_pointer(tmp_path):
    config = _config(tmp_path)

    added = import_existing(config, activate=False)

    root = library_root(config)
    assert len(added) == 3 and not (root / library.ACTIVE_FILE).exists()
    assert [r["event"] for r in _decisions(root)] == ["import"] * 3
    assert library.active_folder_id(config) == FORMAT2_ID


def test_the_active_folder_id_of_a_legacy_folder_is_its_content_id(tmp_path):
    config = tmp_path / "Autotrainer"
    config.mkdir()
    assert library.active_folder_id(config) is None
    _legacy(config / NAME, datetime(2026, 7, 7))
    assert library.active_folder_id(config) == library.legacy_id(config / NAME / "camera_matrix" / "stereo_params.pickle")


def test_a_legacy_id_comes_from_the_content_not_the_name(tmp_path):
    a = _legacy(tmp_path / "a", datetime(2026, 7, 7))
    b = _legacy(tmp_path / "b", datetime(2026, 7, 7))
    c = _legacy(tmp_path / "c", datetime(2026, 7, 8))
    pickle = Path("camera_matrix", "stereo_params.pickle")
    assert library.legacy_id(a / pickle) == library.legacy_id(b / pickle) != library.legacy_id(c / pickle)
    (b / pickle).write_bytes(b"another pickle")
    os.utime(b / pickle, (datetime(2026, 7, 7).timestamp(),) * 2)
    assert library.legacy_id(a / pickle) != library.legacy_id(b / pickle)


def test_a_failed_import_decision_is_retried_on_the_next_import(tmp_path, monkeypatch):
    config = _config(tmp_path)
    root = library_root(config)
    real_record = library.record_decision
    failed = []

    def flaky(root, calibration_id, kind, detail):
        if kind == "import" and not failed:
            failed.append(calibration_id)
            raise OSError("disk full")
        return real_record(root, calibration_id, kind, detail)

    monkeypatch.setattr(library, "record_decision", flaky)
    added = import_existing(config)
    monkeypatch.undo()
    assert len(added) == 3 and active_id(root) == FORMAT2_ID, "a failed decision does not stop the import"
    assert failed[0] not in [r["id"] for r in _decisions(root) if r["event"] == "import"]

    assert import_existing(config) == []
    imports = [r["id"] for r in _decisions(root) if r["event"] == "import"]
    assert sorted(imports) == sorted(added), "each entry has exactly one import decision"


def test_a_discard_on_a_first_install_does_not_warn_about_the_pointer(tmp_path, caplog):
    root = library_root(tmp_path)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    with caplog.at_level(logging.WARNING):
        library.discard_entry(root, calibration_id)
    assert not any("no active calibration" in r.getMessage() for r in caplog.records)
    assert not (root / calibration_id).exists()


def test_an_imported_legacy_entry_is_dated_by_its_marker(tmp_path):
    config = tmp_path / "Autotrainer"
    config.mkdir()
    _legacy(config / NAME, datetime(2026, 7, 7, 8, 30))
    legacy_id, = import_existing(config)
    folder = library_root(config) / legacy_id
    # A copy that lost the file times (a restore from a backup, say) still has the marker.
    pickle_path = folder / "camera_matrix" / "stereo_params.pickle"
    os.utime(pickle_path, (datetime(2026, 11, 1).timestamp(),) * 2)

    assert store.calibration_created(folder) == datetime(2026, 7, 7, 8, 30)
    # The made-by check reads the marker: a session after 07-07 resolves; one before refuses.
    assert store._check_made_by(folder, datetime(2026, 8, 1)) is None
    with pytest.raises(CalibrationResolutionError, match="was made later"):
        store._check_made_by(folder, datetime(2026, 7, 1))


def test_a_corrupt_library_entry_does_not_shadow_an_old_folder_with_its_id(tmp_path, caplog):
    config = tmp_path / "Autotrainer"
    config.mkdir()
    old_id = _built(config / NAME)
    root = library_root(config)
    add_entry(root, config / NAME)
    (root / old_id / store.INDEX_FILE).write_text("format: 3\nid: [half-written\n")

    with caplog.at_level(logging.WARNING):
        assert store.find_calibration(config, old_id) == config / NAME
    assert any("cannot be read" in r.getMessage() for r in caplog.records)

    shutil.rmtree(config / NAME)
    with pytest.raises(CalibrationResolutionError, match="unreadable"):
        store.find_calibration(config, old_id)


# --- sessions --------------------------------------------------------------------------

def _project(tmp_path: Path, when: datetime = datetime(2026, 11, 20)) -> ProjectInfo:
    return ProjectInfo(root=(tmp_path / "Data").as_posix(), device_id="rig", when=when, session=3)


def _write_metadata(project: ProjectInfo, start: datetime, **calibration3d) -> None:
    data = {"metadataSchemaVersion": 2, "scope": "session", "boundary": {"startWallTime": start.timestamp()}}
    if calibration3d:
        data["hardware"] = {"configured": {"calibration3d": calibration3d}}
    path = Path(project.get_metadata_file() + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_session_resolves_by_id_from_library_then_falls_back(tmp_path):
    config = tmp_path / "Autotrainer"
    config.mkdir()
    root = library_root(config)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    # A calibration only the old folders hold, found the old way.
    old_id = _built(config / f"{NAME}_20260701_120000")
    _built(config / NAME)
    project = _project(tmp_path)

    _write_metadata(project, datetime(2026, 11, 20, 9), id=calibration_id, preset=BASE.name)
    assert resolve_session_calibration(project, config) == root / calibration_id

    _write_metadata(project, datetime(2026, 11, 20, 9), id=old_id, preset=BASE.name)
    assert resolve_session_calibration(project, config) == config / f"{NAME}_20260701_120000"

    _write_metadata(project, datetime(2026, 11, 20, 9), id="0" * 32)
    with pytest.raises(CalibrationResolutionError, match="pass --calib-dir"):
        resolve_session_calibration(project, config)


def test_a_session_without_an_id_uses_what_the_library_had_active_then(tmp_path):
    config = tmp_path / "Autotrainer"
    config.mkdir()
    root = library_root(config)
    built = tmp_path / "built"
    calibration_id = _built(built)
    add_entry(root, built)
    set_active(root, calibration_id, by="ben")
    project = _project(tmp_path)

    _write_metadata(project, datetime.now())
    assert resolve_session_calibration(project, config) == root / calibration_id
    # Before the first set_active the library says nothing, so the old folders decide.
    _built(config / NAME)
    _write_metadata(project, datetime(2020, 1, 1))
    with pytest.raises(CalibrationResolutionError, match="pass --calib-dir"):
        resolve_session_calibration(project, config)


def test_an_imported_legacy_entry_is_read_as_legacy_offline(tmp_path, monkeypatch):
    config = tmp_path / "Autotrainer"
    config.mkdir()
    _legacy(config / NAME, datetime(2026, 7, 7))
    legacy_id, = import_existing(config)
    folder = library_root(config) / legacy_id
    seen = {}

    def fake(project, session, vid_tag, dlc_seg, calib_src_dir, center_method, *, frame_rate, preset=None):
        seen.update(calib_src_dir=calib_src_dir, preset=preset)
        raise _Stop

    monkeypatch.setattr(intersession_module, "process_raw_data", fake)
    project = _project(tmp_path)
    _write_metadata(project, datetime(2026, 11, 20, 9))
    with pytest.raises(_Stop):
        intersession_process(project, calib_dir=folder)
    assert seen == {"calib_src_dir": folder.as_posix(), "preset": None}


# --- a format-3 entry and the session's preset -----------------------------------------

def _preset_seen(tmp_path, monkeypatch, folder: Path, **calibration3d):
    seen = {}

    def fake(project, session, vid_tag, dlc_seg, calib_src_dir, center_method, *, frame_rate, preset=None):
        seen["preset"] = preset
        raise _Stop

    monkeypatch.setattr(intersession_module, "process_raw_data", fake)
    project = _project(tmp_path)
    _write_metadata(project, datetime(2026, 11, 20, 9), **calibration3d)
    with pytest.raises(_Stop):
        intersession_process(project, calib_dir=folder)
    return seen["preset"]


def test_a_format_3_session_with_no_preset_is_read_at_the_live_set(tmp_path, monkeypatch):
    folder = tmp_path / "entry"
    _built(folder)
    preset = _preset_seen(tmp_path, monkeypatch, folder)
    assert preset.geometry == BASE


def test_a_format_3_session_with_no_preset_and_no_live_set_is_refused(tmp_path, monkeypatch):
    # The live set is what the cameras recorded at; the largest binning would be a guess.
    folder = tmp_path / "entry"
    store.write_calibration(folder, fit=RIG.truth, presets=presets_from_base(BASE), board=RIG.board,
                            report=_REPORT, capture={})
    index = store.load_index(folder)
    assert index.format == 3 and index.presets and "live" not in [s["name"] for s in index.preset_sets]
    monkeypatch.setattr(intersession_module, "process_raw_data", lambda *a, **k: pytest.fail("processed"))
    project = _project(tmp_path)
    _write_metadata(project, datetime(2026, 11, 20, 9), id=index.id)
    with pytest.raises(CalibrationGeometryError,
                       match=r"names no preset, and the calibration in .* has no live preset set"):
        intersession_process(project, calib_dir=folder)


def test_a_format_3_session_names_its_preset(tmp_path, monkeypatch):
    folder = tmp_path / "entry"
    _built(folder)
    preset = _preset_seen(tmp_path, monkeypatch, folder, preset=presets_from_base(BASE)[1].name)
    assert preset.geometry == presets_from_base(BASE)[1]


@pytest.mark.parametrize("calibration3d", [{}, {"preset": BASE.name}], ids=["no_preset_named", "preset_named"])
def test_a_format_3_entry_with_no_left_right_pair_says_so_instead_of_crashing(tmp_path, monkeypatch, calibration3d):
    folder = tmp_path / "entry"
    _built(folder, RIG_AB)
    assert store.load_index(folder).presets == ()
    monkeypatch.setattr(intersession_module, "process_raw_data", lambda *a, **k: pytest.fail("processed"))
    project = _project(tmp_path)
    _write_metadata(project, datetime(2026, 11, 20, 9), **calibration3d)
    with pytest.raises(CalibrationGeometryError, match="left-right preset|no stereo preset|no preset"):
        intersession_process(project, calib_dir=folder)
