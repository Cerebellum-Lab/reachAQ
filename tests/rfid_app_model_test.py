import logging
import threading

from autotrainer.core import AnimalSubject, ExternalAnimalRecord, ExternalIdentity
from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.model.rfid_resolution import RfidResolutionKind
from tools.acquisition.model.subsystem_status import SubsystemId, SubsystemState
from autotrainer.device.rfid_reader import RfidReaderState, RfidReaderStatus


TAG = "360002353933099"


def record(subject="PT-1", name="Mouse A"):
    return ExternalAnimalRecord(
        identity=ExternalIdentity(subject),
        physical_rfid=TAG,
        new_animal_name_candidate=name,
        state="Stock",
        source_payload={"Physical Tag": subject, "Plate ID": TAG},
        source_hash="record-hash",
    )


def test_first_scan_waits_for_setup_then_links_existing_animal(app_model):
    manual = app_model.add_animal("Mouse A")

    result = app_model.resolve_external_record(record(), scanned_rfid=TAG)

    assert result.kind is RfidResolutionKind.SETUP_REQUIRED
    assert result.animal is None
    assert manual.external_identity is None
    assert app_model.selected_animal is None

    completed = app_model.complete_rfid_animal_setup(
        result.record,
        scanned_rfid=TAG,
        name="Mouse A",
        notes="persistent note",
        existing_animal_id=manual.id,
    )
    assert completed.kind is RfidResolutionKind.LINKED_AND_SELECTED
    assert completed.animal.id == manual.id
    assert completed.animal.notes == "persistent note"
    assert app_model.selected_animal.id == manual.id

    updated = record(name="New SoftMouse Display")
    second = app_model.resolve_external_record(updated, scanned_rfid=TAG)
    assert second.kind is RfidResolutionKind.SELECTED
    assert second.animal.id == manual.id
    assert second.animal.name == "Mouse A"


def test_scan_creation_selection_and_link_are_logged(app_model, caplog):
    caplog.set_level(logging.INFO, logger="tools.acquisition.model.app_model")

    pending = app_model.handle_rfid_scan(record(), scanned_rfid=TAG)
    result = app_model.complete_rfid_animal_setup(
        pending.record,
        scanned_rfid=TAG,
        name="Mouse A",
        notes="",
    )

    assert result.kind is RfidResolutionKind.CREATED_AND_SELECTED
    messages = [entry.getMessage() for entry in caplog.records]
    assert any(
        f"RFID scan resolution started: rfid={TAG}" in message
        for message in messages
    )
    assert any(
        "SoftMouse link saved:" in message and f"rfid={TAG}" in message
        for message in messages
    )
    assert any(
        "Animal selection changed:" in message
        and f"animal_id={result.animal.id}" in message
        for message in messages
    )
    assert any(
        "RFID scan resolution complete:" in message
        and "result=created_and_selected" in message
        for message in messages
    )


def test_scan_never_mutates_or_queues_selection_while_busy(app_model):
    app_model._set_session_recording_status(SessionRecordingStatus.ARMING)
    before = tuple(app_model.animals)

    result = app_model.resolve_external_record(record(), scanned_rfid=TAG)

    assert result.kind is RfidResolutionKind.BUSY
    assert tuple(app_model.animals) == before
    assert app_model.selected_animal is None


def test_duplicate_external_links_are_an_integrity_conflict(app_model):
    identity = ExternalIdentity("PT-1")
    first = AnimalSubject(name="one", external_identity=identity)
    second = AnimalSubject(name="two", external_identity=identity)

    app_model.animals = [first, second]

    assert identity.key in app_model.external_link_conflicts
    assert app_model.get_animal_by_external_identity(identity) is None


def test_unknown_scan_reports_stale_cache_without_creating_animal(app_model):
    before = tuple(app_model.animals)

    result = app_model.resolve_external_record(
        None, scanned_rfid=TAG, cache_stale=True
    )

    assert result.kind is RfidResolutionKind.UNKNOWN_RFID
    assert "cache is stale" in result.message
    assert tuple(app_model.animals) == before


def test_reader_health_is_visible_but_never_a_recording_blocker(app_model):
    app_model.set_rfid_reader_status(
        RfidReaderStatus(RfidReaderState.FAILED, "/dev/rfid", "unplugged")
    )

    status = app_model.subsystem_statuses[SubsystemId.RFID_READER.value]
    assert status.state is SubsystemState.FAILED
    assert not status.required_for_recording
    assert all("rfid" not in blocker for blocker in app_model.recording_blockers)


def test_launch_catchup_refreshes_only_when_manifest_is_new(app_model):
    class Sync:
        manifest_path = "manifest.json"

        def __init__(self, due):
            self.due = due
            self.refresh_calls = 0

        def refresh_due(self):
            return self.due

        def refresh_now(self):
            self.refresh_calls += 1
            raise RuntimeError("stop after proving the refresh was requested")

    not_due = Sync(False)
    app_model._animal_metadata_sync = not_due
    app_model._run_animal_metadata_catchup()
    app_model._animal_metadata_refresh_thread.join(2)
    assert not_due.refresh_calls == 0
    assert not app_model.animal_metadata_refresh_busy

    due = Sync(True)
    app_model._animal_metadata_sync = due
    app_model._run_animal_metadata_catchup()
    app_model._animal_metadata_refresh_thread.join(2)
    assert due.refresh_calls == 1
    assert not app_model.animal_metadata_refresh_busy


def test_refresh_runs_off_caller_thread_and_coalesces_duplicates(app_model):
    started = threading.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()

    class Sync:
        manifest_path = "manifest.json"

        def __init__(self):
            self.thread_ids = []

        def refresh_now(self):
            self.thread_ids.append(threading.get_ident())
            started.set()
            assert release.wait(2)
            raise RuntimeError("test completion")

    sync = Sync()
    app_model._animal_metadata_sync = sync

    assert app_model.request_animal_metadata_refresh("first")
    assert started.wait(2)
    assert not app_model.request_animal_metadata_refresh("duplicate")
    assert app_model.animal_metadata_refresh_busy
    release.set()
    first_thread = app_model._animal_metadata_refresh_thread
    first_thread.join(2)
    # A coalesced request is run exactly once after the first exits.
    second_thread = app_model._animal_metadata_refresh_thread
    second_thread.join(2)

    assert len(sync.thread_ids) == 2
    assert all(thread_id != caller_thread for thread_id in sync.thread_ids)
    assert not app_model.animal_metadata_refresh_busy


def test_refresh_is_deferred_until_session_is_ready(app_model):
    calls = []

    class Sync:
        manifest_path = "manifest.json"

        def refresh_now(self):
            calls.append("refresh")
            raise RuntimeError("test completion")

    app_model._animal_metadata_sync = Sync()
    app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)

    assert not app_model.request_animal_metadata_refresh("during session")
    assert calls == []
    assert not app_model.animal_metadata_refresh_busy

    app_model._set_session_recording_status(SessionRecordingStatus.READY)
    app_model._animal_metadata_refresh_thread.join(2)
    assert calls == ["refresh"]
    assert not app_model.animal_metadata_refresh_busy
