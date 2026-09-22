"""What the NI-DAQ configuration claims, and whether anyone has checked.

A channel map is a set of assertions about cables: `laser1_diode` on
`PXI1Slot5/ai8` asserts a photodiode is wired to that pin. Nothing in the
application ever tested one, and on this rig several such assertions were
wrong for weeks while the stream recorded them happily - an input that is not
connected to what it is named for still answered when a laser was driven, so
it produced a plausible trace rather than an empty one. Fiction is harder to
notice than silence.

This module holds the two halves of the remedy that are not hardware: the
list of points the configuration asserts, and a record of which ones have
been confirmed and when. `tools/hardware/verify_nidaq_wiring.py` produces the
record by driving each output and observing what moves; the application reads
it back and says what is unverified rather than implying everything is fine.

Each check is fingerprinted against the assignment it verified, so moving a
channel to a different pin invalidates its own evidence instead of silently
carrying it over.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

#: A point that can be confirmed by observing it while something is driven.
OBSERVABLE = "observable"
#: A point that drives rather than observes; confirmed indirectly, by what it
#: makes move somewhere else.
DRIVER = "driver"
#: A point nothing on this side can see - an output with no readback path.
OPAQUE = "opaque"

#: Responded to the thing it is assigned to.
CONFIRMED = "confirmed"
#: Its own driver was exercised and it did not respond. This is the status
#: that means "not connected", and it is deliberately distinct from UNTESTED:
#: silence under a driver is evidence, silence with no driver is not.
SILENT = "silent"
#: Responded to something it is not assigned to. The run cannot say why - a
#: mislabelled cable, a split, or coupling between inputs all look the same
#: from here - so it reports the contradiction and leaves the cause open.
UNEXPECTED = "unexpected"
#: Nothing in the run could drive it, so the run says nothing about it.
UNTESTED = "untested"

#: Statuses that mean the point carries what the configuration says it does.
GOOD_STATUSES = frozenset({CONFIRMED})


def _fingerprint(name: str, physical_channel: str) -> str:
    """Identity of one assertion, so rewiring invalidates its own evidence."""
    digest = hashlib.sha256(f"{name}|{physical_channel}".encode("utf-8"))
    return digest.hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class WiringPoint:
    """One assertion the configuration makes about a cable."""

    name: str
    physical_channel: str
    role: str
    kind: str = OBSERVABLE
    #: Why this point cannot be checked, when kind is OPAQUE.
    opaque_reason: str = ""

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.name, self.physical_channel)


@dataclasses.dataclass(frozen=True)
class WiringCheck:
    """What was found when a point was driven and watched."""

    name: str
    physical_channel: str
    status: str
    method: str
    detail: str
    checked_on: str
    fingerprint: str

    @classmethod
    def for_point(cls, point: WiringPoint, status: str, method: str,
                  detail: str, checked_on: Optional[str] = None):
        return cls(
            name=point.name,
            physical_channel=point.physical_channel,
            status=status,
            method=method,
            detail=detail,
            checked_on=checked_on or datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            fingerprint=point.fingerprint,
        )

    def to_record(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    @classmethod
    def from_record(cls, record):
        return cls(
            name=str(record.get("name", "")),
            physical_channel=str(record.get("physical_channel", "")),
            status=str(record.get("status", UNTESTED)),
            method=str(record.get("method", "")),
            detail=str(record.get("detail", "")),
            checked_on=str(record.get("checked_on", "")),
            fingerprint=str(record.get("fingerprint", "")),
        )


@dataclasses.dataclass(frozen=True)
class WiringVerification:
    """Every check a verification run produced."""

    checks: Tuple[WiringCheck, ...] = tuple()
    generated_on: str = ""

    def check_for(self, point: WiringPoint) -> Optional[WiringCheck]:
        """The check that verified this exact assignment, if any.

        Matching on the fingerprint rather than the name is the whole point:
        a channel moved to a different pin has no evidence about its new pin,
        however recently its old one was confirmed.
        """
        for check in self.checks:
            if check.fingerprint == point.fingerprint:
                return check
        return None

    def to_record(self) -> Dict[str, object]:
        return {
            "schema": 1,
            "generated_on": self.generated_on,
            "checks": [check.to_record() for check in self.checks],
        }

    @classmethod
    def from_record(cls, record):
        checks = record.get("checks", ()) if isinstance(record, dict) else ()
        return cls(
            checks=tuple(WiringCheck.from_record(item) for item in checks),
            generated_on=str(record.get("generated_on", ""))
            if isinstance(record, dict) else "",
        )

    @classmethod
    def load(cls, path: Path):
        """The record at this path, or an empty one when there is none."""
        try:
            return cls.from_record(json.loads(Path(path).read_text()))
        except FileNotFoundError:
            return cls()
        except Exception:
            # A corrupt record must not stop the application; it means the
            # same thing an absent one does, which is that nothing is known.
            return cls()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_record(), indent=2) + "\n")


def wiring_points(configuration) -> Tuple[WiringPoint, ...]:
    """Every cable the configuration asserts, from one place.

    Both the verification tool and the application read this, so neither can
    drift from the other's idea of what is being claimed.
    """
    points = []

    stream = getattr(configuration, "nidaq_stream", None)
    for channel in getattr(stream, "channels", ()) or ():
        points.append(WiringPoint(
            name=channel.name,
            physical_channel=channel.physical_channel,
            role=f"{channel.kind} stream input",
        ))

    laser = getattr(configuration, "laser", None)
    for channel in getattr(laser, "channels", ()) or ():
        number = channel.channel_id.value
        points.append(WiringPoint(
            name=f"laser{number}_command",
            physical_channel=channel.analog_output,
            role="laser command output",
            kind=DRIVER,
        ))
        if channel.shutter_output:
            points.append(WiringPoint(
                name=f"laser{number}_shutter",
                physical_channel=channel.shutter_output,
                role="laser shutter output",
                kind=OPAQUE,
                opaque_reason="digital output with no readback path",
            ))
        if channel.trigger_route_source:
            points.append(WiringPoint(
                name=f"laser{number}_trigger_in",
                physical_channel=channel.trigger_route_source,
                role="stimulus trigger input",
            ))

    seen = set()
    unique = []
    for point in points:
        if point.fingerprint in seen:
            continue
        seen.add(point.fingerprint)
        unique.append(point)
    return tuple(unique)


@dataclasses.dataclass(frozen=True)
class WiringSummary:
    """What the application says about the map it is about to rely on."""

    confirmed: Tuple[str, ...] = tuple()
    unverified: Tuple[str, ...] = tuple()
    failed: Tuple[str, ...] = tuple()
    opaque: Tuple[str, ...] = tuple()

    @property
    def is_clean(self) -> bool:
        return not self.unverified and not self.failed

    def detail(self) -> str:
        total = (len(self.confirmed) + len(self.unverified)
                 + len(self.failed) + len(self.opaque))
        return f"NI-DAQ wiring {len(self.confirmed)}/{total} confirmed"

    def warning(self) -> str:
        """One sentence naming what is not known, or empty when all is well.

        Names the channels rather than counting them: "3 unverified" sends
        someone to a file, a name sends them to a cable.
        """
        parts = []
        if self.failed:
            parts.append(
                "did not respond when last checked: " + ", ".join(self.failed))
        if self.unverified:
            parts.append("never checked: " + ", ".join(self.unverified))
        if not parts:
            return ""
        return (
            "NI-DAQ wiring is unverified (" + "; ".join(parts)
            + "); run tools/hardware/verify_nidaq_wiring.py"
        )

    def contradictions(self) -> Tuple[str, ...]:
        """Channels that answered to the wrong thing, which is the worst case.

        A silent channel records nothing and is obvious in the data. One that
        answers to another channel's driver records something plausible, which
        is not.
        """
        return self.failed


def summarize(configuration, verification: WiringVerification) -> WiringSummary:
    """Compare what the configuration claims against what has been checked."""
    confirmed, unverified, failed, opaque = [], [], [], []
    for point in wiring_points(configuration):
        if point.kind == OPAQUE:
            opaque.append(point.name)
            continue
        check = verification.check_for(point)
        if check is None or check.status == UNTESTED:
            unverified.append(point.name)
        elif check.status in GOOD_STATUSES:
            confirmed.append(point.name)
        else:
            failed.append(point.name)
    return WiringSummary(
        confirmed=tuple(confirmed),
        unverified=tuple(unverified),
        failed=tuple(failed),
        opaque=tuple(opaque),
    )
