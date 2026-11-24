"""The animal identifier: an ISO 11784 RFID or any other unique code.

The column setup calls RFID may hold a 15-digit RFID, a barcode or any other
code. Values are trimmed and matched case-sensitively; a value shared by more
than one animal identifies none of them.
"""

from pathlib import Path

import pytest

from reachaq.core.animal.external_metadata import (
    ExternalAnimalRecord,
    ExternalIdentity,
    is_iso11784_rfid,
    normalize_rfid,
)
from reachaq.app.model.animal_registry import AnimalRegistry
from reachaq.app.model.animal_sources.mapping import (
    AnimalTableMapper,
    ColumnMapping,
    SourceTable,
)
from reachaq.app.model.animal_sources.source_setup import (
    OPERATOR_COLUMNS,
    build_mapping,
    suggest_mapping,
)
from reachaq.app.model.animal_sources.spreadsheet import SpreadsheetSource
from reachaq.app.model.rfid_resolution import RfidResolutionKind


RFID = "999000000000001"
FAKE_MICE = Path(__file__).parent / "fixtures" / "fake-mice.csv"


def _batch(rows, *, require_rfid=False, rfid_column="Alt. ID"):
    mapper = AnimalTableMapper(
        ColumnMapping(
            external_identity_column="Physical Tag",
            new_animal_name_column="Physical Tag",
            rfid_column=rfid_column,
            require_rfid=require_rfid,
            state_column="State",
        ),
        provider="softmouse",
    )
    return mapper.preview_table(
        SourceTable(
            headers=("Physical Tag", "Alt. ID", "State"),
            rows=rows,
            content_sha256=str(len(rows)) * 64,
            source_name="animals.csv",
        )
    ).batch


def _ids(batch):
    return {record.identity.subject_id: record.physical_rfid for record in batch.records}


# --- normalisation --------------------------------------------------------


def test_any_non_empty_code_is_an_identifier_and_is_trimmed():
    assert normalize_rfid(RFID) == RFID
    assert normalize_rfid("  BC-0003-A\t") == "BC-0003-A"
    assert normalize_rfid("201B") == "201B"
    # A 14-digit value is not an ISO RFID, but is still a valid code.
    assert normalize_rfid("99900000000000") == "99900000000000"


def test_identifiers_are_case_sensitive():
    assert normalize_rfid("ab12") == "ab12"
    assert normalize_rfid("ab12") != normalize_rfid("AB12")


@pytest.mark.parametrize("value", [None, "", "   ", "nan", "NULL"])
def test_empty_identifiers_are_refused(value):
    with pytest.raises(ValueError):
        normalize_rfid(value)


def test_control_characters_inside_an_identifier_are_refused():
    with pytest.raises(ValueError, match="control"):
        normalize_rfid("BC\t0003")


def test_iso11784_form_is_still_recognised():
    assert is_iso11784_rfid(RFID)
    assert not is_iso11784_rfid("201B")
    assert not is_iso11784_rfid("99900000000000")


# --- import: accepted values and the duplicate rule ----------------------


def test_alt_id_barcodes_import_as_identifiers():
    batch = _batch([("E1", "BC-1", "Stock"), ("E2", RFID, "Stock"), ("E3", "", "Stock")])

    assert _ids(batch) == {"E1": "BC-1", "E2": RFID, "E3": None}
    assert batch.duplicate_rfids == ()


def test_a_shared_value_is_ignored_for_every_animal_that_has_it():
    batch = _batch(
        [
            ("E978", "201B", "Stock"),
            ("E979", "BC-9", "Stock"),
            ("E980", " 201B ", "Stock"),
        ]
    )

    assert _ids(batch) == {"E978": None, "E979": "BC-9", "E980": None}
    assert batch.duplicate_rfids == (("201B", ("E978", "E980")),)
    assert any("201B" in w and "E978" in w and "E980" in w for w in batch.warnings)


def test_values_differing_only_in_case_are_not_duplicates():
    batch = _batch([("E1", "ab12", "Stock"), ("E2", "AB12", "Stock")])

    assert _ids(batch) == {"E1": "ab12", "E2": "AB12"}


def test_duplicates_still_import_when_animals_without_an_id_are_skipped():
    batch = _batch(
        [("E1", "201B", "Stock"), ("E2", "201B", "Stock"), ("E3", "", "Stock")],
        require_rfid=True,
    )

    assert _ids(batch) == {"E1": None, "E2": None}
    assert batch.ignored_missing_rfid_rows == 1


def test_ended_animals_do_not_count_towards_a_duplicate():
    batch = _batch([("E1", "201B", "Stock"), ("E2", "201B", "Ended")])

    assert _ids(batch) == {"E1": "201B"}
    assert batch.duplicate_rfids == ()


# --- registry lookup ------------------------------------------------------


def test_registry_resolves_a_barcode_exactly(tmp_path):
    registry = AnimalRegistry(tmp_path / "cache.sqlite3")
    registry.replace(_batch([("E1", "BC-1", "Stock"), ("E2", "201B", "Stock"),
                             ("E3", "201B", "Stock")]))

    assert registry.resolve_rfid(" BC-1 ").identity.subject_id == "E1"
    assert registry.resolve_rfid("bc-1") is None
    assert registry.resolve_rfid("201B") is None


# --- setup: the name column -----------------------------------------------


def test_setup_asks_for_the_name_column_as_well():
    assert [name for name, _ in OPERATOR_COLUMNS] == [
        "external_identity_column",
        "new_animal_name_column",
        "rfid_column",
    ]
    mapping = build_mapping(
        SpreadsheetSource, {}, "Physical Tag", "RFID", name_column="Name",
        require_rfid=False,
    )
    assert mapping.external_identity_column == "Physical Tag"
    assert mapping.new_animal_name_column == "Name"


def test_name_column_defaults_to_the_physical_tag():
    mapping = build_mapping(SpreadsheetSource, {}, "Physical Tag", None, require_rfid=False)
    assert mapping.new_animal_name_column == "Physical Tag"


# --- the fake-mice fixture, end to end ------------------------------------


def _fake_mice_batch(tmp_path):
    source = SpreadsheetSource({"path": str(FAKE_MICE)}, credentials=None)
    table = source.fetch(tmp_path)
    suggestion = suggest_mapping(SpreadsheetSource, table.headers)
    mapping = build_mapping(
        SpreadsheetSource,
        suggestion,
        suggestion["external_identity_column"],
        suggestion["rfid_column"],
        name_column=suggestion["new_animal_name_column"],
        require_rfid=False,
    )
    assert (mapping.external_identity_column, mapping.new_animal_name_column,
            mapping.rfid_column) == ("Physical Tag", "Name", "RFID")
    return AnimalTableMapper(mapping, provider="spreadsheet").preview_table(table).batch


def test_fake_mice_fixture_imports_under_every_rule(tmp_path):
    batch = _fake_mice_batch(tmp_path)

    assert _ids(batch) == {
        "FAKE-001": "999000000000001",
        "FAKE-002": "999000000000002",
        "FAKE-003": "BC-0003-A",
        "FAKE-004": "bc-0004-b",
        "FAKE-005": None,
        "FAKE-006": None,
        "FAKE-007": None,
    }
    assert batch.duplicate_rfids == (("DUP-0006", ("FAKE-006", "FAKE-007")),)
    names = {r.identity.subject_id: r.new_animal_name_candidate for r in batch.records}
    assert names["FAKE-003"] == "Fake Barcode Mouse 3"


# --- app: typed or barcode-scanned identifiers -----------------------------


@pytest.fixture
def registry_app(app_model, tmp_path):
    registry = AnimalRegistry(tmp_path / "cache.sqlite3")
    registry.replace(_fake_mice_batch(tmp_path))
    app_model._animal_registry = registry
    return app_model


def test_a_typed_identifier_resolves_like_a_reader_scan(registry_app):
    pending = registry_app.resolve_entered_rfid("  BC-0003-A ")

    assert pending.kind is RfidResolutionKind.SETUP_REQUIRED
    assert pending.rfid == "BC-0003-A"
    assert pending.record.identity.subject_id == "FAKE-003"
    assert registry_app.rfid_scan_result is pending

    created = registry_app.complete_rfid_animal_setup(
        pending.record, scanned_rfid=pending.rfid, name="Fake Barcode Mouse 3", notes=""
    )
    assert created.kind is RfidResolutionKind.CREATED_AND_SELECTED

    again = registry_app.resolve_entered_rfid("BC-0003-A")
    assert again.kind is RfidResolutionKind.SELECTED
    assert again.animal.id == created.animal.id
    assert registry_app.selected_animal.id == created.animal.id


def test_a_typed_identifier_is_case_sensitive(registry_app):
    result = registry_app.resolve_entered_rfid("BC-0004-B")

    assert result.kind is RfidResolutionKind.UNKNOWN_RFID


def test_a_duplicate_identifier_selects_nobody(registry_app):
    result = registry_app.resolve_entered_rfid("DUP-0006")

    assert result.kind is RfidResolutionKind.UNKNOWN_RFID
    assert registry_app.selected_animal is None


def test_typed_entry_without_a_registry_reports_unknown(app_model):
    app_model._animal_registry = None

    result = app_model.resolve_entered_rfid("BC-1")

    assert result.kind is RfidResolutionKind.UNKNOWN_RFID
    assert "not configured" in result.message


def test_typed_entry_refuses_an_empty_value(registry_app):
    with pytest.raises(ValueError):
        registry_app.resolve_entered_rfid("   ")


def test_reader_and_typed_entry_reach_the_same_record(registry_app):
    record = registry_app._animal_registry.resolve_rfid("999000000000001")
    typed = registry_app.resolve_entered_rfid("999000000000001")

    assert isinstance(record, ExternalAnimalRecord)
    assert typed.record == record
    assert typed.record.identity == ExternalIdentity(
        "FAKE-001", provider="spreadsheet", subject_id_kind=typed.record.identity.subject_id_kind
    )


def test_the_toolbar_field_sends_its_text_and_clears(qapp):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QLineEdit

    from reachaq.app.view.main_window import MainWindow

    entered, messages = [], []

    def resolve(text):
        entered.append(text)
        if not text.strip("x"):
            raise ValueError("RFID cannot contain control characters")

    window = SimpleNamespace(
        _rfid_entry=QLineEdit(),
        _app_model=SimpleNamespace(resolve_entered_rfid=resolve),
        statusBar=lambda: SimpleNamespace(showMessage=lambda m, _t: messages.append(m)),
    )

    window._rfid_entry.setText("BC-0003-A")
    MainWindow._rfid_entered(window)
    assert entered == ["BC-0003-A"]
    assert window._rfid_entry.text() == ""

    window._rfid_entry.setText("   ")
    MainWindow._rfid_entered(window)
    assert entered == ["BC-0003-A"]

    window._rfid_entry.setText("xx")
    MainWindow._rfid_entered(window)
    assert messages and messages[-1].startswith("RFID not accepted")


@pytest.fixture
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
