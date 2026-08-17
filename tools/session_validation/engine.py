"""Deterministic rule engine over persisted session artifacts only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .model import (
    ValidationProfile,
    ValidationReport,
    ValidationResult,
    ValidationStatus,
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _Context:
    def __init__(self, root, profile):
        self.root = Path(root).resolve()
        self.profile = ValidationProfile(profile)
        self.cache = {}

    def path(self, relative):
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"Artifact path escapes session: {relative}")
        return candidate

    def json(self, relative):
        path = self.path(relative)
        if path not in self.cache:
            with path.open("r", encoding="utf-8") as stream:
                self.cache[path] = json.load(stream)
        return self.cache[path]


def _result(rule_id, status, message, **kwargs):
    return ValidationResult(rule_id, ValidationStatus(status), message, **kwargs)


def validate_session(
    session_path,
    *,
    profile=ValidationProfile.FAST,
    selected_rules: Iterable[str] = (),
    skipped_rules: Iterable[str] = (),
) -> ValidationReport:
    started_perf = time.perf_counter()
    started_utc = _utc_now()
    context = _Context(session_path, profile)
    selected, skipped = set(selected_rules), set(skipped_rules)
    rules = (
        _layout_rule,
        _schema_rule,
        _generation_rule,
        _manifest_rule,
        _metadata_mirror_rule,
        _source_contract_rule,
        _camera_rule,
        _camera_ledger_rule,
        _nidaq_rule,
        _nidaq_timing_graph_rule,
        _event_rule,
        _event_frame_rule,
        _tone_confirmation_rule,
        _laser_confirmation_rule,
        _stim_evidence_rule,
        _board_time_rule,
        _trial_rule,
        _protocol_action_rule,
    )
    results = []
    for rule in rules:
        rule_id = rule.RULE_ID
        if (selected and rule_id not in selected) or rule_id in skipped:
            results.append(_result(
                rule_id,
                ValidationStatus.NOT_APPLICABLE,
                "Rule skipped by command selection",
            ))
            continue
        minimum = getattr(rule, "MINIMUM", ValidationProfile.QUICK)
        order = {
            ValidationProfile.QUICK: 0,
            ValidationProfile.FAST: 1,
            ValidationProfile.FULL: 2,
        }
        if order[context.profile] < order[minimum]:
            results.append(_result(
                rule_id,
                ValidationStatus.NOT_APPLICABLE,
                f"Rule requires {minimum.value} validation",
            ))
            continue
        try:
            produced = rule(context)
            results.extend(
                produced if isinstance(produced, (tuple, list)) else (produced,)
            )
        except Exception as error:
            results.append(_result(
                rule_id,
                ValidationStatus.TOOL_ERROR,
                f"{type(error).__name__}: {error}",
                remediation="Inspect this artifact manually and rerun the targeted rule.",
            ))
    ended_utc = _utc_now()
    return ValidationReport(
        session_path=context.root.as_posix(),
        profile=context.profile,
        started_utc=started_utc,
        ended_utc=ended_utc,
        duration_seconds=time.perf_counter() - started_perf,
        results=tuple(results),
    )


def _layout_rule(context):
    if not context.root.is_dir():
        return _result("session.layout", "fail", "Session directory does not exist")
    manifest = context.path("manifest.json")
    if not manifest.is_file():
        return _result(
            "session.layout", "fail", "Published manifest.json is missing",
            paths=(manifest.as_posix(),),
        )
    data = context.json("manifest.json")
    metadata = context.path(data.get("authoritativeMetadata", ""))
    if not metadata.is_file():
        return _result(
            "session.layout", "fail", "Authoritative metadata is missing",
            paths=(metadata.as_posix(),),
        )
    return _result("session.layout", "pass", "Published session layout is present")
_layout_rule.RULE_ID = "session.layout"


def _schema_rule(context):
    manifest = context.json("manifest.json")
    metadata = context.json(manifest["authoritativeMetadata"])
    observed = {
        "manifest": manifest.get("schemaVersion"),
        "metadata": metadata.get("metadataSchemaVersion"),
    }
    if observed != {"manifest": 1, "metadata": 2}:
        return _result(
            "session.schema", "fail", "Unsupported pre-release session schema",
            expected={"manifest": 1, "metadata": 2}, observed=observed,
            remediation="Record a new session with the current first-release schema.",
        )
    return _result("session.schema", "pass", "Current session schema is supported")
_schema_rule.RULE_ID = "session.schema"


def _generation_rule(context):
    manifest = context.json("manifest.json")
    metadata = context.json(manifest["authoritativeMetadata"])
    generation = manifest.get("metadataGenerationId")
    values = {"manifest": generation, "metadata": metadata.get("metadataGenerationId")}
    for relative in ("streams/stream_manifest.json", "streams/alignment.json"):
        if context.path(relative).is_file():
            values[relative] = context.json(relative).get("metadataGenerationId")
    trial_summary = context.path("streams/trial_summary.json")
    if trial_summary.is_file():
        values["trial_summary"] = context.json(
            "streams/trial_summary.json"
        ).get("metadata_generation_id")
    mismatched = {key: value for key, value in values.items() if value != generation}
    if not generation or mismatched:
        return _result(
            "session.generation", "fail", "Metadata generation identities differ",
            expected=generation, observed=values,
        )
    return _result("session.generation", "pass", "Generation identity is consistent")
_generation_rule.RULE_ID = "session.generation"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_rule(context):
    manifests = (
        ("manifest.json", context.json("manifest.json")),
        (
            "streams/stream_manifest.json",
            context.json("streams/stream_manifest.json"),
        ),
    )
    seen = set()
    errors = []
    for manifest_name, manifest in manifests:
        local_seen = set()
        for item in manifest.get("files", ()):
            relative = item.get("path", "")
            try:
                path = context.path(relative)
            except ValueError as error:
                errors.append(str(error))
                continue
            if relative in local_seen:
                errors.append(f"{manifest_name}: duplicate path {relative}")
            local_seen.add(relative)
            seen.add(relative)
            if not path.is_file():
                errors.append(f"missing {relative}")
                continue
            if path.stat().st_size != item.get("sizeBytes"):
                errors.append(f"size mismatch {relative}")
            if (
                context.profile is ValidationProfile.FULL
                and _sha256(path) != item.get("sha256")
            ):
                errors.append(f"hash mismatch {relative}")
    return _result(
        "session.manifest",
        "fail" if errors else "pass",
        "; ".join(errors) if errors else "Manifest paths and declared values are valid",
        observed=errors or len(seen),
    )
_manifest_rule.RULE_ID = "session.manifest"


def _metadata_mirror_rule(context):
    manifest = context.json("manifest.json")
    json_path = context.path(manifest["authoritativeMetadata"])
    yaml_path = json_path.with_suffix(".yaml")
    if not yaml_path.is_file():
        return _result("metadata.mirror", "fail", "YAML metadata mirror is missing")
    import yaml
    with yaml_path.open("r", encoding="utf-8") as stream:
        yaml_record = yaml.safe_load(stream)
    json_record = context.json(json_path.relative_to(context.root).as_posix())
    status = "pass" if yaml_record == json_record else "fail"
    return _result("metadata.mirror", status, f"JSON/YAML metadata semantic match: {status}")
_metadata_mirror_rule.RULE_ID = "metadata.mirror"
_metadata_mirror_rule.MINIMUM = ValidationProfile.FAST


def _source_contract_rule(context):
    path = context.path("streams/stream_manifest.json")
    if not path.is_file():
        return _result("sources.contract", "fail", "Stream manifest is missing")
    stream = context.json("streams/stream_manifest.json")
    errors, warnings = [], []
    for source in stream.get("enabledSources", ()):
        source_id = source.get("id", "unknown")
        source_kind = str(source.get("kind", ""))
        relative = source.get("path")
        if relative and not context.path(relative).is_file():
            errors.append(f"{source_id}: artifact missing")
        if source.get("persistenceStatus") != "written":
            errors.append(f"{source_id}: not written")
        if source.get("failure"):
            errors.append(f"{source_id}: {source['failure']}")
        requires_continuous_coverage = (
            source_kind in {"camera", "pose", "logs", "stim_camera_detection_evidence"}
            or source_kind.startswith("nidaq")
        )
        if requires_continuous_coverage and source.get("sampleCount", 0) <= 0:
            errors.append(f"{source_id}: empty")
        warnings.extend(f"{source_id}: {item}" for item in source.get("warnings", ()))
    if not stream.get("sessionComplete", False):
        errors.append("stream manifest marks session incomplete")
    status = "fail" if errors else ("warning" if warnings else "pass")
    return _result(
        "sources.contract", status,
        "; ".join(errors or warnings) if (errors or warnings) else "Enabled sources are complete",
    )
_source_contract_rule.RULE_ID = "sources.contract"


def _camera_rule(context):
    stream = context.json("streams/stream_manifest.json")
    errors = []
    for source in stream.get("enabledSources", ()):
        if source.get("kind") != "camera":
            continue
        diagnostics = source.get("diagnostics", {})
        counts = {
            int(value) for value in (
                source.get("sampleCount", 0),
                diagnostics.get("writer_frame_count", 0),
                diagnostics.get("timestamp_row_count", 0),
                diagnostics.get("decoded_frame_count", 0),
            ) if value is not None
        }
        if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
            errors.append(f"{source.get('id')}: stored frame counts differ {sorted(counts)}")
        if context.profile is ValidationProfile.FULL:
            from tools.acquisition.model.camera_recording_validation import _ffprobe_frame_count
            decoded = _ffprobe_frame_count(context.path(source["path"]), 120.0)
            if decoded != source.get("sampleCount"):
                errors.append(
                    f"{source.get('id')}: decoded {decoded}, stored {source.get('sampleCount')}"
                )
    return _result(
        "camera.frames", "fail" if errors else "pass",
        "; ".join(errors) if errors else "Camera frame counts are consistent",
    )
_camera_rule.RULE_ID = "camera.frames"
_camera_rule.MINIMUM = ValidationProfile.FAST


def _frame_ledger_path(context):
    candidates = sorted(context.root.glob("*_frame_timing.csv"))
    if len(candidates) != 1:
        return None, candidates
    return candidates[0], candidates


def _camera_ledger_rule(context):
    stream_manifest = context.json("streams/stream_manifest.json")
    cameras = {
        item["id"].split(".", 1)[-1]: item
        for item in stream_manifest.get("enabledSources", ())
        if item.get("kind") == "camera"
    }
    if not cameras:
        return _result("camera.ledger", "not_applicable", "No camera source was enabled")
    path, candidates = _frame_ledger_path(context)
    if path is None:
        return _result(
            "camera.ledger", "fail",
            f"Expected one merged frame ledger, found {len(candidates)}",
            paths=tuple(item.as_posix() for item in candidates),
        )
    errors, warnings = [], []
    row_count = 0
    previous_id = None
    previous_wall = -math.inf
    present_counts = {name: 0 for name in cameras}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frame_id", "utc_when"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            errors.append("missing columns: " + ", ".join(sorted(missing)))
        for line_number, row in enumerate(reader, 2):
            row_count += 1
            try:
                frame_id = int(row["frame_id"])
                wall = float(row["utc_when"])
                if previous_id is not None:
                    if frame_id <= previous_id:
                        errors.append(f"row {line_number}: non-increasing frame ID")
                    elif frame_id != previous_id + 1:
                        warnings.append(
                            f"row {line_number}: {frame_id - previous_id - 1} "
                            "frame ID(s) explicitly missing"
                        )
                if not math.isfinite(wall) or wall <= previous_wall:
                    errors.append(f"row {line_number}: non-increasing wall time")
                previous_id, previous_wall = frame_id, wall
                for name in cameras:
                    value = row.get(f"frame_present_{name}")
                    if value is None:
                        errors.append(f"row {line_number}: camera {name} has no presence column")
                    elif int(value):
                        present_counts[name] += 1
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"row {line_number}: {error}")
    for name, source in cameras.items():
        expected = int(source.get("sampleCount", -1))
        if present_counts[name] != expected:
            errors.append(
                f"camera {name}: ledger={present_counts[name]} expected={expected}"
            )
    if cameras and row_count != max(present_counts.values(), default=0):
        warnings.append(
            f"merged rows={row_count}; largest camera coverage={max(present_counts.values(), default=0)}"
        )
    status = "fail" if errors else ("warning" if warnings else "pass")
    return _result(
        "camera.ledger", status,
        "; ".join((errors or warnings)[:20])
        if errors or warnings else "Merged frame IDs and camera coverage are consistent",
        observed={"rows": row_count, "camera_counts": present_counts},
        paths=(path.as_posix(),),
    )
_camera_ledger_rule.RULE_ID = "camera.ledger"
_camera_ledger_rule.MINIMUM = ValidationProfile.FAST


def _nidaq_rule(context):
    stream = context.json("streams/stream_manifest.json")
    sources = [item for item in stream.get("enabledSources", ()) if str(item.get("kind", "")).startswith("nidaq")]
    if not sources:
        return _result("nidaq.continuity", "not_applicable", "NI-DAQ was not enabled")
    import h5py
    import numpy
    path = context.path(sources[0]["path"])
    errors, warnings = [], []
    with h5py.File(path, "r") as store:
        required = {
            "sample_index", "perf_time", "offset_seconds", "wall_time",
            "values", "epoch",
        }
        missing = required - set(store)
        if missing:
            errors.append("missing datasets: " + ", ".join(sorted(missing)))
        else:
            count = int(store["sample_index"].shape[0])
            timelines = ("perf_time", "offset_seconds", "wall_time", "epoch")
            for name in timelines:
                if store[name].shape != (count,):
                    errors.append(f"{name} length differs from sample_index")
            if len(store["values"].shape) != 2 or store["values"].shape[1] != count:
                errors.append("values/sample timeline lengths differ")
            channel_names = tuple(
                item.decode() if isinstance(item, bytes) else str(item)
                for item in store.attrs.get("channel_names", ())
            )
            if store["values"].shape[0] != len(channel_names):
                errors.append("values channel axis differs from channel_names")
            if len(channel_names) != len(set(channel_names)):
                errors.append("channel_names contains duplicates")
            expected_channels = {
                item["id"].split(".", 1)[-1]
                for item in sources
            }
            if set(channel_names) != expected_channels:
                errors.append(
                    f"HDF5 channels={sorted(channel_names)}; manifest={sorted(expected_channels)}"
                )
            if count <= 0:
                errors.append("NI stream contains no samples")
            sample_rate = float(store.attrs.get("sample_rate_hz", 0))
            if not math.isfinite(sample_rate) or sample_rate <= 0:
                errors.append("sample_rate_hz is invalid")
            if count and sample_rate > 0:
                first_index = int(store["sample_index"][0])
                last_index = int(store["sample_index"][-1])
                if last_index - first_index != count - 1:
                    errors.append("sample-index endpoints do not match sample count")
                first_offset = float(store["offset_seconds"][0])
                last_offset = float(store["offset_seconds"][-1])
                expected_duration = (count - 1) / sample_rate
                observed_duration = last_offset - first_offset
                if abs(observed_duration - expected_duration) > 1.5 / sample_rate:
                    errors.append(
                        "NI timeline duration differs from sample-index duration"
                    )
                start = float(store.attrs.get("recording_start_perf", math.nan))
                end = float(store.attrs.get("recording_end_perf", math.nan))
                first_perf = float(store["perf_time"][0])
                last_perf = float(store["perf_time"][-1])
                if not all(map(math.isfinite, (start, end, first_perf, last_perf))):
                    errors.append("canonical NI boundary attributes are not finite")
                elif first_perf < start - 1.5 / sample_rate or last_perf > end + 1.5 / sample_rate:
                    errors.append("NI samples extend outside the saved boundary")
                else:
                    missing_start = max(0.0, first_perf - start)
                    missing_end = max(0.0, end - last_perf)
                    if max(missing_start, missing_end) > 2.0 / sample_rate:
                        warnings.append(
                            f"boundary coverage start={missing_start:.6g}s end={missing_end:.6g}s"
                        )
            if context.profile is ValidationProfile.FULL and count:
                indices = store["sample_index"][:]
                perf = store["perf_time"][:]
                offsets = store["offset_seconds"][:]
                if not numpy.all(numpy.diff(indices) == 1):
                    errors.append("sample_index is not strictly contiguous")
                if not numpy.all(numpy.diff(perf) > 0):
                    errors.append("perf_time is not strictly increasing")
                if not numpy.all(numpy.diff(offsets) > 0):
                    errors.append("offset_seconds is not strictly increasing")
            if bool(store.attrs.get("collection_worker_failed", False)):
                errors.append(
                    "NI collection worker failed: "
                    + str(store.attrs.get("collection_first_error", "unknown error"))
                )
            for key in ("collection_error_count", "gap_count", "overrun_samples"):
                if int(store.attrs.get(key, 0)):
                    warnings.append(f"{key}={int(store.attrs[key])}")
    status = "fail" if errors else ("warning" if warnings else "pass")
    return _result(
        "nidaq.continuity", status,
        "; ".join(errors or warnings) if (errors or warnings) else "NI sample timeline is continuous",
    )
_nidaq_rule.RULE_ID = "nidaq.continuity"
_nidaq_rule.MINIMUM = ValidationProfile.FAST


def _nidaq_timing_graph_rule(context):
    sources = [
        item for item in context.json("streams/stream_manifest.json").get("enabledSources", ())
        if str(item.get("kind", "")).startswith("nidaq")
    ]
    if not sources:
        return _result("nidaq.timing_graph", "not_applicable", "NI-DAQ was not enabled")
    import h5py
    path = context.path(sources[0]["path"])
    errors, warnings = [], []
    with h5py.File(path, "r") as store:
        raw = store.attrs.get("timing_plan_json", "")
        if isinstance(raw, bytes):
            raw = raw.decode()
        if not raw:
            errors.append("timing_plan_json is missing")
            plan = {}
        else:
            plan = json.loads(raw)
        if plan and not plan.get("is_valid", False):
            errors.append("persisted NI timing plan is invalid")
        devices = plan.get("resolved_devices", ())
        if len(devices) > 2:
            errors.append("first-release timing graph contains more than two devices")
        if len(devices) > 1 and plan.get("synchronization_quality") in {
            "independent", "software_only", "unresolved", None,
        }:
            errors.append("multiple NI devices lack a deterministic synchronization claim")
        graph = plan.get("task_graph")
        if not graph:
            errors.append("exact NI task graph is missing")
        else:
            task_ids = [item.get("task_id") for item in graph.get("tasks", ())]
            if not task_ids or len(task_ids) != len(set(task_ids)):
                errors.append("NI task graph identifiers are missing or duplicated")
            if set(graph.get("create_order", ())) != set(task_ids):
                errors.append("NI task graph create order does not cover every task")
        status = plan.get("multidevice_probe_status", "not_requested")
        strategy = (graph or {}).get("strategy", "per_device")
        if strategy == "forced_multidevice" and status != "verified":
            errors.append("forced multidevice strategy was not verified")
        elif strategy == "auto_multidevice" and status not in {
            "verified", "fallback_per_device", "not_applicable_single_device",
        }:
            warnings.append(f"multidevice probe status is {status}")
    outcome = "fail" if errors else ("warning" if warnings else "pass")
    return _result(
        "nidaq.timing_graph", outcome,
        "; ".join(errors or warnings) if errors or warnings
        else "Persisted NI task graph and synchronization claim are valid",
        observed={
            "devices": len(plan.get("resolved_devices", ())),
            "quality": plan.get("synchronization_quality"),
            "multidevice_probe_status": plan.get("multidevice_probe_status"),
        },
        paths=(path.as_posix(),),
    )
_nidaq_timing_graph_rule.RULE_ID = "nidaq.timing_graph"
_nidaq_timing_graph_rule.MINIMUM = ValidationProfile.FAST


def _event_rule(context):
    path = context.path("streams/device.csv")
    if not path.is_file():
        return _result("events.alignment", "not_applicable", "Device stream is absent")
    errors = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        previous = -math.inf
        for index, row in enumerate(rows, 2):
            value = row.get("perf_time") or row.get("host_perf_time") or row.get("timestamp")
            if value in (None, ""):
                continue
            try:
                current = float(value)
            except ValueError:
                errors.append(f"row {index}: invalid performance timestamp")
                continue
            if not math.isfinite(current) or current < previous:
                errors.append(f"row {index}: nonmonotonic performance timestamp")
            previous = current
    return _result(
        "events.alignment", "fail" if errors else "pass",
        "; ".join(errors[:10]) if errors else "Device event ordering is valid",
    )
_event_rule.RULE_ID = "events.alignment"
_event_rule.MINIMUM = ValidationProfile.FAST


def _event_frame_rule(context):
    cameras = tuple(
        source for source in context.json("streams/stream_manifest.json").get(
            "enabledSources", ()
        ) if source.get("kind") == "camera"
    )
    if not cameras:
        return _result(
            "events.frames", "not_applicable",
            "No recorded camera source exists for frame association",
        )
    event_paths = []
    for path in (
            context.path("streams/device.csv"),
            context.path("streams/events.csv"),
            context.path("streams/laser.csv"),
    ):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as stream:
            if next(csv.DictReader(stream), None) is not None:
                event_paths.append(path)
    event_paths = tuple(event_paths)
    if not event_paths:
        return _result("events.frames", "not_applicable", "No event stream is present")
    frame_path, _ = _frame_ledger_path(context)
    if frame_path is None:
        return _result("events.frames", "fail", "Recorded frame ledger is unavailable")
    frame_ids = set()
    with frame_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame_ids.add(int(row["frame_id"]))
    paths = event_paths
    errors, warnings, associated = [], [], 0
    required = {
        "perf_time", "offset_seconds", "wall_time", "frame_id",
        "recorded_frame_index", "frame_relation", "frame_start_perf_time",
        "event_to_frame_start_seconds", "alignment_method",
        "alignment_confidence",
    }
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required - set(reader.fieldnames or ())
            if missing:
                errors.append(f"{path.name}: missing {', '.join(sorted(missing))}")
                continue
            previous = -math.inf
            for line_number, row in enumerate(reader, 2):
                try:
                    perf = float(row["perf_time"])
                    if not math.isfinite(perf) or perf < previous:
                        errors.append(f"{path.name}:{line_number}: nonmonotonic event")
                    previous = perf
                    frame_value = row.get("frame_id")
                    if frame_value in (None, ""):
                        warnings.append(f"{path.name}:{line_number}: no frame association")
                        continue
                    frame_id = int(frame_value)
                    if frame_id not in frame_ids:
                        errors.append(f"{path.name}:{line_number}: unknown frame {frame_id}")
                    delta = float(row["event_to_frame_start_seconds"])
                    if not math.isfinite(delta):
                        errors.append(f"{path.name}:{line_number}: nonfinite frame delta")
                    if not row["alignment_method"] or not row["alignment_confidence"]:
                        errors.append(f"{path.name}:{line_number}: missing alignment evidence")
                    associated += 1
                except (TypeError, ValueError) as error:
                    errors.append(f"{path.name}:{line_number}: {error}")
    status = "fail" if errors else ("warning" if warnings else "pass")
    return _result(
        "events.frames", status,
        "; ".join((errors or warnings)[:20])
        if errors or warnings else "All persisted events map to recorded frames",
        observed={"associated": associated, "warnings": len(warnings)},
        paths=tuple(path.as_posix() for path in paths),
    )
_event_frame_rule.RULE_ID = "events.frames"
_event_frame_rule.MINIMUM = ValidationProfile.FAST


def _tone_confirmation_rule(context):
    path = context.path("streams/alignment.json")
    if not path.is_file():
        return _result("events.tones", "fail", "Alignment metadata is missing")
    tone = context.json("streams/alignment.json").get("toneConfirmation")
    if not tone:
        return _result("events.tones", "not_applicable", "No tone confirmation data")
    errors, warnings = [], []
    confirmations = tone.get("matched", ())
    for index, item in enumerate(confirmations, 1):
        if not item.get("valid", True):
            continue
        for field in ("physicalEventPerfTime", "recordingOffsetSeconds"):
            if item.get(field) is None:
                errors.append(f"confirmation {index}: missing {field}")
        frame = item.get("frameAssociation") or {}
        if frame.get("frameId") is None:
            errors.append(f"confirmation {index}: missing recorded frame ID")
        if not frame.get("method") or not frame.get("confidence"):
            errors.append(f"confirmation {index}: missing alignment method/confidence")
    unmatched_events = tone.get("unmatchedEvents", ())
    if unmatched_events:
        errors.append(
            f"{len(unmatched_events)} decoded tone event(s) lack NI confirmation"
        )
    unmatched = tone.get("unmatchedEdges", ())
    boundary = context.json("streams/alignment.json").get("canonicalBoundary", {})
    start = boundary.get("startPerfTime")
    end = boundary.get("endPerfTime")
    for index, edge in enumerate(unmatched, 1):
        perf = edge.get("perfTime")
        if perf is not None and start is not None and end is not None:
            if not float(start) <= float(perf) <= float(end):
                errors.append(f"unmatched edge {index} lies outside saved boundary")
    in_boundary_unmatched = sum(
        1
        for edge in unmatched
        if (
            edge.get("perfTime") is not None
            and start is not None
            and end is not None
            and float(start) <= float(edge["perfTime"]) <= float(end)
        )
    )
    if in_boundary_unmatched:
        errors.append(
            f"{in_boundary_unmatched} valid in-session NI tone edge(s) lack a decoded event"
        )
    artifact_count = int(tone.get("artifactCount", len(tone.get("artifacts", ()))))
    if artifact_count:
        warnings.append(f"{artifact_count} short-pulse artifact(s) retained")
    status = "fail" if errors else ("warning" if warnings else "pass")
    return _result(
        "events.tones", status,
        "; ".join(errors or warnings) if errors or warnings
        else "Tone commands, NI confirmations, and recorded frames agree",
        observed={
            "confirmations": len(confirmations),
            "unmatched_events": len(unmatched_events),
            "unmatched_edges": len(unmatched),
            "artifacts": artifact_count,
        },
        paths=(path.as_posix(),),
    )
_tone_confirmation_rule.RULE_ID = "events.tones"
_tone_confirmation_rule.MINIMUM = ValidationProfile.FAST


def _laser_confirmation_rule(context):
    path = context.path("streams/laser.csv")
    if not path.is_file():
        return _result("events.laser", "not_applicable", "Laser stream is absent")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        return _result("events.laser", "not_applicable", "No laser events were recorded")
    errors, warnings = [], []
    for line_number, row in enumerate(rows, 2):
        event = row.get("event", "")
        if not event:
            errors.append(f"row {line_number}: event is missing")
        if row.get("frame_id") in (None, ""):
            warnings.append(f"row {line_number}: no recorded-frame association")
    return _result(
        "events.laser", "fail" if errors else ("warning" if warnings else "pass"),
        "; ".join((errors or warnings)[:20]) if errors or warnings
        else "Laser events have recorded-frame evidence",
        observed=len(rows), paths=(path.as_posix(),),
    )
_laser_confirmation_rule.RULE_ID = "events.laser"
_laser_confirmation_rule.MINIMUM = ValidationProfile.FAST


def _stim_evidence_rule(context):
    sources = context.json("streams/stream_manifest.json").get("enabledSources", ())
    source = next((item for item in sources if item.get("id") == "stim_camera_evidence"), None)
    if source is None:
        return _result("stim.evidence", "not_applicable", "Stim-camera evidence mode was not enabled")
    import h5py
    path = context.path(source["path"])
    errors, warnings = [], []
    with h5py.File(path, "r") as store:
        if "evidence" not in store:
            errors.append("evidence dataset is missing")
            count = 0
        else:
            evidence = store["evidence"]
            count = int(evidence.shape[0])
            names = set(evidence.dtype.names or ())
            required = {
                "frame_id", "camera_timestamp_ns", "frame_perf_time",
                "decision_perf_time", "session_generation", "logical_trial_id",
                "attempt_id", "armed", "triggered", "gap",
            }
            missing = required - names
            if missing:
                errors.append("missing evidence fields: " + ", ".join(sorted(missing)))
            if context.profile is ValidationProfile.FULL and count and "frame_id" in names:
                import numpy
                ids = evidence["frame_id"][:]
                if not numpy.all(numpy.diff(ids) == 1):
                    errors.append("stim evidence frame IDs are not contiguous")
            if count and not missing:
                import numpy
                gaps = int(numpy.count_nonzero(evidence["gap"][:]))
                if gaps:
                    errors.append(f"stim evidence reports {gaps} acquired-frame gap(s)")
                triggered = numpy.flatnonzero(evidence["triggered"][:])
                ownership = set()
                for position in triggered:
                    key = (
                        int(evidence["session_generation"][position]),
                        int(evidence["logical_trial_id"][position]),
                        int(evidence["attempt_id"][position]),
                    )
                    if key in ownership:
                        errors.append(f"attempt {key} emitted more than one trigger")
                    ownership.add(key)
                    if not bool(evidence["armed"][position]):
                        errors.append(f"attempt {key} triggered while unarmed")
                clip_group = store.get("clips")
                clip_count = 0 if clip_group is None else len(clip_group)
                if clip_count != len(triggered):
                    errors.append(
                        f"triggered rows={len(triggered)}; bounded clips={clip_count}"
                    )
                if clip_group is not None:
                    for clip_id, clip in clip_group.items():
                        if not bool(clip.attrs.get("complete", False)):
                            warnings.append(f"clip {clip_id} is truncated")
        dropped = int(store.attrs.get("dropped_batches", 0))
        clip_drops = int(store.attrs.get("dropped_clips", 0))
        if dropped:
            errors.append(f"dropped evidence batches={dropped}")
        if clip_drops:
            warnings.append(f"dropped clips={clip_drops}")
    if count != int(source.get("sampleCount", count)):
        errors.append(f"evidence count={count}; manifest={source.get('sampleCount')}")
    status = "fail" if errors else ("warning" if warnings else "pass")
    return _result(
        "stim.evidence", status,
        "; ".join(errors or warnings) if errors or warnings
        else "Stim-camera evidence is continuous and complete",
        observed=count, paths=(path.as_posix(),),
    )
_stim_evidence_rule.RULE_ID = "stim.evidence"
_stim_evidence_rule.MINIMUM = ValidationProfile.FAST


def _optional_float(row, name):
    value = row.get(name)
    if value in (None, ""):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return value


def _optional_int(row, name):
    value = row.get(name)
    if value in (None, ""):
        return None
    return int(value)


def _board_time_rule(context):
    path = context.path("streams/device.csv")
    if not path.is_file():
        return _result("events.board_time", "not_applicable", "Device stream is absent")
    errors, warnings = [], []
    timestamped = 0
    previous = {}
    model_ids = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "host_receive_perf_time", "board_boot_id", "board_sequence",
            "board_time_us", "board_timestamp_kind", "board_aligned_perf_time",
            "board_clock_model_id", "board_clock_uncertainty_seconds",
            "estimated_transport_delay_seconds", "event_perf_time",
            "event_timestamp_method", "event_timing_confidence",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            return _result(
                "events.board_time", "fail",
                "Current device timing schema is incomplete: " + ", ".join(sorted(missing)),
                paths=(path.as_posix(),),
            )
        for line_number, row in enumerate(reader, 2):
            try:
                boot = _optional_int(row, "board_boot_id")
                sequence = _optional_int(row, "board_sequence")
                board_us = _optional_int(row, "board_time_us")
                if boot is None and sequence is None and board_us is None:
                    continue
                timestamped += 1
                if None in (boot, sequence, board_us):
                    errors.append(f"row {line_number}: partial board timing envelope")
                    continue
                prior = previous.get(boot)
                if prior is not None:
                    prior_sequence, prior_us = prior
                    delta = (sequence - prior_sequence) & 0xFFFFFFFF
                    if delta == 0:
                        errors.append(f"row {line_number}: duplicate board sequence {sequence}")
                    elif delta >= 0x80000000:
                        errors.append(f"row {line_number}: board sequence moved backward")
                    elif delta > 1:
                        warnings.append(
                            f"row {line_number}: {delta - 1} board message(s) missing"
                        )
                    if board_us < prior_us:
                        errors.append(f"row {line_number}: board clock moved backward")
                previous[boot] = (sequence, board_us)

                aligned = _optional_float(row, "board_aligned_perf_time")
                model_id = row.get("board_clock_model_id") or None
                uncertainty = _optional_float(row, "board_clock_uncertainty_seconds")
                transport = _optional_float(row, "estimated_transport_delay_seconds")
                event_perf = _optional_float(row, "event_perf_time")
                method = row.get("event_timestamp_method")
                confidence = row.get("event_timing_confidence")
                if aligned is None:
                    if method == "board_clock_affine" or confidence == "board_timestamp":
                        errors.append(f"row {line_number}: board confidence lacks aligned time")
                    continue
                if not model_id or uncertainty is None or uncertainty < 0:
                    errors.append(f"row {line_number}: aligned board time lacks a valid clock model")
                else:
                    model_ids.add(model_id)
                if method == "board_clock_affine":
                    if event_perf is None or abs(event_perf - aligned) > 1e-9:
                        errors.append(f"row {line_number}: selected event time differs from board alignment")
                    if confidence != "board_timestamp":
                        errors.append(f"row {line_number}: board event has incorrect confidence")
                if transport is not None and uncertainty is not None:
                    if transport < -uncertainty:
                        errors.append(f"row {line_number}: transport delay precedes uncertainty bound")
                    elif transport < 0:
                        warnings.append(f"row {line_number}: transport delay is slightly negative")
                    elif transport > 1.0:
                        warnings.append(f"row {line_number}: transport delay exceeds 1 s")
            except (TypeError, ValueError) as error:
                errors.append(f"row {line_number}: {error}")
    if not timestamped:
        return _result(
            "events.board_time", "not_applicable",
            "No board timing envelopes were present; host receive timing remains in use",
        )
    status = "fail" if errors else ("warning" if warnings else "pass")
    details = errors or warnings
    return _result(
        "events.board_time", status,
        "; ".join(details[:20]) if details else "Board clocks, sequences, and aligned events are valid",
        observed={
            "timestamped_rows": timestamped,
            "boot_epochs": len(previous),
            "clock_models": sorted(model_ids),
            "warnings": len(warnings),
            "errors": len(errors),
        },
        paths=(path.as_posix(),),
    )
_board_time_rule.RULE_ID = "events.board_time"
_board_time_rule.MINIMUM = ValidationProfile.FAST


def _trial_rule(context):
    path = context.path("streams/trials.jsonl")
    if not path.is_file():
        return _result("trials.lifecycle", "not_applicable", "No pellet attempts were recorded")
    errors, identities = [], set()
    records = []
    terminal_outcomes = {
        "success", "failure", "pellet_missing", "no_reach", "unscored",
        "hardware_error", "incomplete", "aborted",
    }
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            identity = (
                record.get("trial_id"), record.get("attempt_id"), record.get("operation_id")
            )
            if identity in identities:
                errors.append(f"row {index}: duplicate attempt identity")
            identities.add(identity)
            if not record.get("operation_id"):
                errors.append(f"row {index}: operation identity is empty")
            expected_label = (
                f"unindexed.{record.get('attempt_id')}"
                if record.get("trial_id") is None
                else f"{record.get('trial_id')}.{record.get('attempt_id')}"
            )
            if record.get("attempt_label") != expected_label:
                errors.append(f"row {index}: attempt label is inconsistent")
            send = _record_float(record, "send_perf_time", errors, index)
            ack = _record_float(record, "send_ack_perf_time", errors, index, optional=True)
            finalized = _record_float(
                record, "finalized_perf_time", errors, index, optional=True,
            )
            if ack is not None and send is not None and ack < send:
                errors.append(f"row {index}: acknowledgement precedes SEND")
            if finalized is not None and ack is not None and finalized < ack:
                errors.append(f"row {index}: finalization precedes acknowledgement")
            if record.get("outcome") not in terminal_outcomes:
                errors.append(f"row {index}: nonterminal outcome {record.get('outcome')!r}")
            if finalized is None:
                errors.append(f"row {index}: finalized time is missing")
            if int(record.get("reach_count", 0)) < 0:
                errors.append(f"row {index}: negative reach count")
            if int(record.get("success_count", 0)) > int(record.get("reach_count", 0)):
                errors.append(f"row {index}: successes exceed reaches")
    summary_path = context.path("streams/trial_summary.json")
    if summary_path.is_file():
        summary = context.json("streams/trial_summary.json")
        if int(summary.get("physical_attempts", -1)) != len(records):
            errors.append("trial summary physical_attempts differs from trials.jsonl")
        if int(summary.get("incomplete_attempts", 0)):
            errors.append("finalized session contains incomplete attempts")
        recomputed = {
            "physical_attempts": len(records),
            "hardware_errors": sum(
                item.get("outcome") == "hardware_error" for item in records
            ),
            "pending_analysis_attempts": sum(
                item.get("outcome") == "pending_analysis" for item in records
            ),
            "pellets_presented": sum(
                item.get("send_ack_perf_time") is not None for item in records
            ),
            "reaches": sum(int(item.get("reach_count", 0)) for item in records),
            "successful_reaches": sum(
                int(item.get("success_count", 0)) for item in records
            ),
            "pellets_consumed": sum(
                int(item.get("consumption_count", 0)) for item in records
            ),
        }
        for key, value in recomputed.items():
            if int(summary.get(key, -1)) != value:
                errors.append(f"trial summary {key}={summary.get(key)}; recomputed={value}")
    return _result(
        "trials.lifecycle", "fail" if errors else "pass",
        "; ".join(errors) if errors else "Pellet-attempt identities and summary are complete",
    )
_trial_rule.RULE_ID = "trials.lifecycle"
_trial_rule.MINIMUM = ValidationProfile.FAST


def _record_float(record, field, errors, row_index, *, optional=False):
    value = record.get(field)
    if value is None and optional:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        errors.append(f"row {row_index}: invalid {field}")
        return None
    if not math.isfinite(value):
        errors.append(f"row {row_index}: nonfinite {field}")
        return None
    return value


def _protocol_action_rule(context):
    path = context.path("streams/trials.jsonl")
    if not path.is_file():
        return _result("trials.protocol", "not_applicable", "No pellet attempts were recorded")
    errors, checked = [], 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            protocol = record.get("protocol_context") or {}
            selected_protocol = protocol.get("protocol_id")
            operation = record.get("protocol_operation")
            if not selected_protocol:
                continue
            checked += 1
            if not operation:
                errors.append(f"row {line_number}: selected protocol lacks action evidence")
                continue
            recipe = operation.get("recipe") or {}
            requested = recipe.get("requested_row") or {}
            if str(recipe.get("protocol_id")) != str(selected_protocol):
                errors.append(f"row {line_number}: protocol identity differs")
            if recipe.get("logical_trial_id") != record.get("trial_id"):
                errors.append(f"row {line_number}: recipe trial identity differs")
            if recipe.get("attempt_id") != record.get("attempt_id"):
                errors.append(f"row {line_number}: recipe attempt identity differs")
            if requested.get("trial_id") != record.get("trial_id"):
                errors.append(f"row {line_number}: frozen row identity differs")
            snapshotted = protocol.get("compiled_recipe") or {}
            if snapshotted and snapshotted != recipe:
                errors.append(f"row {line_number}: compiled recipe snapshot differs")
            if operation.get("state") not in {"completed", "failed", "cancelled"}:
                errors.append(f"row {line_number}: protocol operation is not terminal")
            observations = operation.get("observations") or ()
            times = [item.get("perf_time") for item in observations]
            try:
                if any(not math.isfinite(float(item)) for item in times):
                    raise ValueError
                if any(float(b) < float(a) for a, b in zip(times, times[1:])):
                    errors.append(f"row {line_number}: action observations moved backward")
            except (TypeError, ValueError):
                errors.append(f"row {line_number}: invalid action observation time")
            for field in (
                "resolved_dcs_target", "resolved_motor_target", "position_evidence",
                "stimulus_selected", "stimulus_seed", "stimulus_draw",
            ):
                if field not in recipe:
                    errors.append(f"row {line_number}: recipe lacks {field}")
    if not checked:
        return _result("trials.protocol", "not_applicable", "Session used No protocol mode")
    return _result(
        "trials.protocol", "fail" if errors else "pass",
        "; ".join(errors[:20]) if errors
        else "Frozen protocol recipes and action lifecycles are complete",
        observed=checked, paths=(path.as_posix(),),
    )
_protocol_action_rule.RULE_ID = "trials.protocol"
_protocol_action_rule.MINIMUM = ValidationProfile.FAST
