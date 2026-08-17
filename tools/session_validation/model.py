from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Mapping, Tuple


VALIDATOR_VERSION = "1.0"


class ValidationProfile(str, enum.Enum):
    QUICK = "quick"
    FAST = "fast"
    FULL = "full"


class ValidationStatus(str, enum.Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    TOOL_ERROR = "tool_error"


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    rule_id: str
    status: ValidationStatus
    message: str
    expected: object = None
    observed: object = None
    paths: Tuple[str, ...] = ()
    remediation: str = ""
    rule_version: int = 1

    def to_record(self):
        result = dataclasses.asdict(self)
        result["status"] = self.status.value
        return result


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    session_path: str
    profile: ValidationProfile
    started_utc: str
    ended_utc: str
    duration_seconds: float
    results: Tuple[ValidationResult, ...]
    validator_version: str = VALIDATOR_VERSION
    schema_version: int = 1

    @property
    def counts(self):
        return {
            status.value: sum(item.status is status for item in self.results)
            for status in ValidationStatus
        }

    @property
    def exit_code(self):
        if any(item.status is ValidationStatus.TOOL_ERROR for item in self.results):
            return 2
        if any(item.status is ValidationStatus.FAIL for item in self.results):
            return 1
        return 0

    def to_record(self):
        return {
            "schema_version": self.schema_version,
            "validator_version": self.validator_version,
            "session_path": self.session_path,
            "profile": self.profile.value,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "duration_seconds": self.duration_seconds,
            "counts": self.counts,
            "exit_code": self.exit_code,
            "results": [item.to_record() for item in self.results],
        }
