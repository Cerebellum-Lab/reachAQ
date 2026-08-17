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
        _nidaq_rule,
        _event_rule,
        _board_time_rule,
        _trial_rule,
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
    manifest = context.json("manifest.json")
    seen = set()
    errors = []
    for item in manifest.get("files", ()):
        relative = item.get("path", "")
        try:
            path = context.path(relative)
        except ValueError as error:
            errors.append(str(error))
            continue
        if relative in seen:
            errors.append(f"duplicate path {relative}")
        seen.add(relative)
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        if path.stat().st_size != item.get("sizeBytes"):
            errors.append(f"size mismatch {relative}")
        if context.profile is ValidationProfile.FULL and _sha256(path) != item.get("sha256"):
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
        relative = source.get("path")
        if relative and not context.path(relative).is_file():
            errors.append(f"{source_id}: artifact missing")
        if source.get("persistenceStatus") != "written":
            errors.append(f"{source_id}: not written")
        if source.get("failure"):
            errors.append(f"{source_id}: {source['failure']}")
        if source.get("sampleCount", 0) <= 0:
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
        required = {"sample_index", "perf_time", "offset_seconds", "values"}
        missing = required - set(store)
        if missing:
            errors.append("missing datasets: " + ", ".join(sorted(missing)))
        else:
            count = int(store["sample_index"].shape[0])
            if store["values"].shape[1] != count:
                errors.append("values/sample timeline lengths differ")
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
    summary_path = context.path("streams/trial_summary.json")
    if summary_path.is_file():
        summary = context.json("streams/trial_summary.json")
        if int(summary.get("physical_attempts", -1)) != len(records):
            errors.append("trial summary physical_attempts differs from trials.jsonl")
        if int(summary.get("incomplete_attempts", 0)):
            errors.append("finalized session contains incomplete attempts")
    return _result(
        "trials.lifecycle", "fail" if errors else "pass",
        "; ".join(errors) if errors else "Pellet-attempt identities and summary are complete",
    )
_trial_rule.RULE_ID = "trials.lifecycle"
_trial_rule.MINIMUM = ValidationProfile.FAST
