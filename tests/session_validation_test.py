import hashlib
import json
from pathlib import Path

import yaml

from tools.session_validation import ValidationProfile, validate_session
from tools.session_validation.cli import main


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _session(tmp_path):
    root = tmp_path / "session001"
    generation = "session001-g1"
    metadata = {
        "metadataSchemaVersion": 2,
        "metadataGenerationId": generation,
        "sessionId": "session001",
    }
    metadata_json = root / "session001_metadata.json"
    _write(metadata_json, json.dumps(metadata))
    _write(root / "session001_metadata.yaml", yaml.safe_dump(metadata))
    stream_manifest = {
        "schemaVersion": 2,
        "metadataGenerationId": generation,
        "sessionId": "session001",
        "sessionComplete": True,
        "enabledSources": [],
    }
    _write(root / "streams/stream_manifest.json", json.dumps(stream_manifest))
    _write(root / "streams/alignment.json", json.dumps({
        "schemaVersion": 2, "metadataGenerationId": generation,
    }))
    files = []
    for path in (
        metadata_json,
        root / "session001_metadata.yaml",
        root / "streams/stream_manifest.json",
    ):
        data = path.read_bytes()
        files.append({
            "path": path.relative_to(root).as_posix(),
            "sizeBytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    _write(root / "manifest.json", json.dumps({
        "schemaVersion": 1,
        "metadataGenerationId": generation,
        "authoritativeMetadata": metadata_json.name,
        "files": files,
    }))
    return root


def test_quick_validator_is_read_only(tmp_path):
    root = _session(tmp_path)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }
    report = validate_session(root, profile=ValidationProfile.QUICK)
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*") if path.is_file()
    }
    assert report.exit_code == 0
    assert before == after


def test_full_validator_detects_manifest_hash_failure(tmp_path):
    root = _session(tmp_path)
    (root / "session001_metadata.yaml").write_text("changed", encoding="utf-8")
    report = validate_session(root, profile=ValidationProfile.FULL)
    assert report.exit_code in {1, 2}
    manifest = next(item for item in report.results if item.rule_id == "session.manifest")
    assert manifest.status.value == "fail"


def test_cli_writes_only_explicit_output(tmp_path):
    root = _session(tmp_path)
    output = tmp_path / "report.json"
    assert main([str(root), "--quick", "--json", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["profile"] == "quick"


def test_fast_validator_checks_board_clock_and_sequence_evidence(tmp_path):
    root = _session(tmp_path)
    columns = (
        "perf_time", "host_receive_perf_time", "board_boot_id",
        "board_sequence", "board_time_us", "board_timestamp_kind",
        "board_aligned_perf_time", "board_clock_model_id",
        "board_clock_uncertainty_seconds", "estimated_transport_delay_seconds",
        "event_perf_time", "event_timestamp_method", "event_timing_confidence",
    )
    rows = (
        (10.0, 10.01, 7, 1, 1000, "physical_start", 10.0, "m1", 0.001,
         0.01, 10.0, "board_clock_affine", "board_timestamp"),
        (10.1, 10.11, 7, 2, 101000, "completed", 10.1, "m1", 0.001,
         0.01, 10.1, "board_clock_affine", "board_timestamp"),
    )
    path = root / "streams/device.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        import csv
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)

    report = validate_session(root, profile=ValidationProfile.FAST)

    result = next(item for item in report.results if item.rule_id == "events.board_time")
    assert result.status.value == "pass"
    assert result.observed["timestamped_rows"] == 2
