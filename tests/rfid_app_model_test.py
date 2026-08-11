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


def test_scan_does_not_name_link_or_rename_existing_animal(app_model):
    manual = app_model.add_animal("Mouse A")

    result = app_model.resolve_external_record(record(), scanned_rfid=TAG)

    assert result.kind is RfidResolutionKind.CREATED_AND_SELECTED
    assert result.animal.id != manual.id
    assert result.animal.name == "Mouse A"
    assert manual.external_identity is None

    updated = record(name="New SoftMouse Display")
    second = app_model.resolve_external_record(updated, scanned_rfid=TAG)
    assert second.kind is RfidResolutionKind.SELECTED
    assert second.animal.id == result.animal.id
    assert second.animal.name == "Mouse A"


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
        def __init__(self, due):
            self.due = due

        def refresh_due(self):
            return self.due

    calls = []
    app_model.refresh_animal_metadata = lambda: calls.append("refresh")
    app_model._animal_metadata_sync = Sync(False)
    app_model._run_animal_metadata_catchup()
    assert calls == []

    app_model._animal_metadata_sync = Sync(True)
    app_model._run_animal_metadata_catchup()
    assert calls == ["refresh"]
