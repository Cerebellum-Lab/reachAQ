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


def test_validator_checks_merged_frame_ids_and_event_association(tmp_path):
    root = _session(tmp_path)
    stream_path = root / "streams/stream_manifest.json"
    stream = json.loads(stream_path.read_text(encoding="utf-8"))
    stream["enabledSources"] = [{
        "id": "camera.left",
        "kind": "camera",
        "path": "left.mp4",
        "sampleCount": 2,
        "persistenceStatus": "written",
        "diagnostics": {
            "writer_frame_count": 2,
            "timestamp_row_count": 2,
            "decoded_frame_count": 2,
        },
    }]
    _write(stream_path, json.dumps(stream))
    _write(root / "left.mp4", "fixture")
    _write(
        root / "session001_frame_timing.csv",
        "frame_id,utc_when,frame_present_left\n10,1000.0,1\n11,1000.01,1\n",
    )
    event_columns = (
        "perf_time,offset_seconds,wall_time,event_name,event_id,event_index,"
        "context_json,timestamp_method,frame_id,recorded_frame_index,"
        "frame_relation,frame_start_perf_time,frame_start_offset_seconds,"
        "frame_start_wall_time,event_to_frame_start_seconds,alignment_method,"
        "alignment_confidence\n"
    )
    _write(
        root / "streams/events.csv",
        event_columns
        + "20.0,1.0,1000.005,event,id,1,{},host,11,1,following,20.005,"
        "1.005,1000.01,0.005,writer_timeline,writer_timestamp\n",
    )

    report = validate_session(
        root,
        profile=ValidationProfile.FAST,
        selected_rules=("camera.ledger", "events.frames"),
    )

    assert next(
        item for item in report.results if item.rule_id == "camera.ledger"
    ).status.value == "pass"
    assert next(
        item for item in report.results if item.rule_id == "events.frames"
    ).status.value == "pass"


def test_validator_reconciles_trial_and_protocol_operation(tmp_path):
    root = _session(tmp_path)
    recipe = {
        "operation_id": "prepared-op",
        "protocol_id": "p",
        "logical_trial_id": 1,
        "attempt_id": 1,
        "requested_row": {"trial_id": 1},
        "resolved_dcs_target": [1, 2, 3],
        "resolved_motor_target": [4, 5, 6],
        "position_evidence": {"mode": "base"},
        "stimulus_selected": False,
        "stimulus_seed": 1,
        "stimulus_draw": 0.5,
    }
    record = {
        "session_id": "session001",
        "operation_id": "can-send-context",
        "trial_id": 1,
        "attempt_id": 1,
        "attempt_label": "1.1",
        "send_perf_time": 10.0,
        "send_ack_perf_time": 10.1,
        "finalized_perf_time": 11.0,
        "outcome": "success",
        "reach_count": 1,
        "success_count": 1,
        "consumption_count": 1,
        "protocol_context": {
            "protocol_id": "p",
            "compiled_recipe": recipe,
        },
        "protocol_operation": {
            "recipe": recipe,
            "state": "completed",
            "observations": [
                {"state": "preparing", "perf_time": 9.0},
                {"state": "completed", "perf_time": 11.0},
            ],
        },
    }
    _write(root / "streams/trials.jsonl", json.dumps(record) + "\n")
    _write(root / "streams/trial_summary.json", json.dumps({
        "physical_attempts": 1,
        "hardware_errors": 0,
        "incomplete_attempts": 0,
        "pending_analysis_attempts": 0,
        "pellets_presented": 1,
        "reaches": 1,
        "successful_reaches": 1,
        "pellets_consumed": 1,
    }))

    report = validate_session(
        root,
        profile=ValidationProfile.FAST,
        selected_rules=("trials.lifecycle", "trials.protocol"),
    )

    assert next(
        item for item in report.results if item.rule_id == "trials.lifecycle"
    ).status.value == "pass"
    assert next(
        item for item in report.results if item.rule_id == "trials.protocol"
    ).status.value == "pass"


def test_event_sources_may_be_empty_without_failing_continuous_coverage(tmp_path):
    root = _session(tmp_path)
    stream_path = root / "streams/stream_manifest.json"
    stream = json.loads(stream_path.read_text(encoding="utf-8"))
    stream["enabledSources"] = [{
        "id": "laser_outputs",
        "kind": "laser_commands_and_states",
        "path": "streams/laser.csv",
        "sampleCount": 0,
        "persistenceStatus": "written",
    }]
    _write(stream_path, json.dumps(stream))
    _write(root / "streams/laser.csv", "perf_time,event\n")

    report = validate_session(
        root,
        profile=ValidationProfile.FAST,
        selected_rules=("sources.contract", "events.frames"),
    )

    assert next(
        item for item in report.results if item.rule_id == "sources.contract"
    ).status.value == "pass"
    assert next(
        item for item in report.results if item.rule_id == "events.frames"
    ).status.value == "not_applicable"


def test_tone_confirmation_rejects_unmatched_in_session_evidence(tmp_path):
    root = _session(tmp_path)
    alignment_path = root / "streams/alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    alignment.update({
        "canonicalBoundary": {"startPerfTime": 10.0, "endPerfTime": 20.0},
        "toneConfirmation": {
            "matched": [],
            "unmatchedEvents": [{"eventPerfTime": 12.0}],
            "unmatchedEdges": [{"perfTime": 13.0}],
            "artifacts": [],
        },
    })
    _write(alignment_path, json.dumps(alignment))

    report = validate_session(
        root,
        profile=ValidationProfile.FAST,
        selected_rules=("events.tones",),
    )

    result = next(item for item in report.results if item.rule_id == "events.tones")
    assert result.status.value == "fail"
    assert result.observed["unmatched_events"] == 1
    assert result.observed["unmatched_edges"] == 1
