"""Recorded event files must keep decoding, and new records must match them.

The fixture holds real rows copied from christielab10's hourly event files
(2026-07-22 to 2026-11-13): one row per recorded event id and one per enum repr
found inside a payload.
"""

import csv
import datetime as dt
import io
import re
from pathlib import Path
from uuid import UUID

import pytest

from reachaq.core import EventInfo, ProjectInfo
from reachaq.core import api
from reachaq.core.api import ApiEvent, ApiEventKind
from reachaq.core.event.file_event_plugin import FileEventPlugin
from reachaq.core.interfaces import CaptureAnalysisResult, RecordingEndingReason

FIXTURE = Path(__file__).parent / "fixtures" / "recorded_events" / "christielab10_events.csv"

# Recorded but no longer published by reachAQ, so there is nothing to rebuild.
RETIRED_REPR_CLASSES = {"ApiAlarmKind"}
REPR = re.compile(r"<(Api[A-Za-z]+)\.([A-Za-z0-9_]+): (-?\d+)>")


def _rows():
    with FIXTURE.open(newline="") as fh:
        return list(csv.DictReader(fh, escapechar="\\"))


ROWS = _rows()


def test_fixture_covers_every_recorded_kind():
    assert len({row["EventId"] for row in ROWS}) == 35


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['EventId']}-{r['Time']}")
def test_recorded_id_and_name_decode(row):
    kind = ApiEventKind(int(row["EventId"]))
    assert str(kind) == row["EventName"]
    assert f"{kind}" == row["EventId"]  # how log lines format it


def test_every_recorded_enum_repr_is_reproduced():
    seen = set()
    for row in ROWS:
        for cls_name, member, value in REPR.findall(row["Data"]):
            if cls_name in RETIRED_REPR_CLASSES:
                continue
            cls = getattr(api, cls_name)
            assert int(cls[member]) == int(value)
            assert repr(cls[member]) == f"<{cls_name}.{member}: {value}>"
            seen.add(cls_name)
    assert seen == {"ApiApplicationMode", "ApiTrainingMode", "ApiDetectorKind"}


def _payload(data: str):
    namespace = {name: getattr(api, name) for name in api.__all__}
    namespace.update(UUID=UUID, RecordingEndingReason=RecordingEndingReason,
                     CaptureAnalysisResult=CaptureAnalysisResult, nan=float("nan"))
    # Recorded reprs are "<Class.member: value>"; rebuild them as members.
    source = re.sub(r"<([A-Za-z]+)\.([A-Za-z0-9_]+): [^>]+>", r"\1.\2", data)
    return eval(source, namespace)  # noqa: S307 - fixture written by reachAQ


REBUILDABLE = [row for row in ROWS
               if not {cls_name for cls_name, _, _ in REPR.findall(row["Data"])} & RETIRED_REPR_CLASSES]


@pytest.mark.parametrize("row", REBUILDABLE, ids=lambda r: f"{r['EventId']}-{r['Time']}")
def test_file_plugin_writes_recorded_rows_byte_for_byte(row, tmp_path):
    when = dt.datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S.%f")
    info = EventInfo(ApiEventKind(int(row["EventId"])), when=when, index=int(row["Index"]),
                     context=_payload(row["Data"]))
    plugin = FileEventPlugin()
    plugin.set_project(ProjectInfo(root=str(tmp_path), device_id="christielab10", ensure_exists=True))
    plugin.process_event(info, int(row["Repeat"]))
    plugin.close()

    written = list(tmp_path.rglob("*_events.csv"))
    assert len(written) == 1
    with written[0].open(newline="") as fh:
        lines = fh.read().splitlines()
    expected = io.StringIO()
    writer = csv.DictWriter(expected, fieldnames=list(row), quotechar='"', escapechar="\\",
                            quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow({**row, "Index": int(row["Index"]), "EventId": int(row["EventId"]),
                     "Repeat": int(row["Repeat"])})
    assert lines[1] == expected.getvalue().rstrip("\r\n")
    assert lines[1].split(",", 4)[2:4] == [row["EventId"], f'"{row["EventName"]}"']


def test_repeats_coalesce_on_equal_kind_and_payload():
    # Recorded behaviour: same kind with a different payload is a new event
    # (1142 adjacent pairs written with Repeat 0); same kind and payload repeats.
    now = dt.datetime.now()
    a = ApiEvent(ApiEventKind.systemStatus, now, 1, {"acquisition_state": "idle"})
    assert a.is_same(ApiEvent(ApiEventKind.systemStatus, now, 2, {"acquisition_state": "idle"}))
    assert not a.is_same(ApiEvent(ApiEventKind.systemStatus, now, 2, {"acquisition_state": "running"}))
    assert not a.is_same(ApiEvent(ApiEventKind.projectChanged, now, 2, {"acquisition_state": "idle"}))
    assert ApiEvent(1).is_same(ApiEvent(1, now, 3))


def test_session_stream_uses_bare_member_name():
    # streams/events.csv records event_name "sessionStarted" with event_id 1331.
    assert ApiEventKind.sessionStarted.name == "sessionStarted"
    assert int(ApiEventKind.sessionStarted) == 1331


def test_ids_are_unique():
    for cls in (api.ApiEventKind, api.ApiApplicationMode, api.ApiTrainingMode, api.ApiDetectorKind,
                api.ApiPelletShiftSource, api.ApiCommand):
        # An IntEnum silently turns a duplicate value into an alias.
        assert len(cls.__members__) == len(list(cls)), cls
