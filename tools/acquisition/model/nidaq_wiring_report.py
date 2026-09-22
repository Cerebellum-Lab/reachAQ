"""One wiring report, rendered as text on a terminal and as rows in a window.

The verification already knew what it found. What it could not say is where
to go and what to touch: "laser2_diode is silent on PXI1Slot5/ai4" sends
somebody to a configuration file, and the next question is always which
connector that is. With a breakout named for the device, the same line ends
"BNC-2090A "AI 4" (BNC)", which is a thing on a bench.

Built here rather than in either caller so the headless run and the window
cannot disagree about what was found - they differ in how they draw it and
in nothing else.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from tools.acquisition.model.nidaq_breakout import (
    breakout_for_device,
    describe_terminal,
)
from tools.acquisition.model.nidaq_wiring_verification import (
    CONFIRMED,
    OPAQUE,
    SILENT,
    UNEXPECTED,
    UNTESTED,
    WiringVerification,
    summarize,
    wiring_points,
)

#: Worst first. A channel answering to another channel's driver records
#: something plausible, which is worse than one recording nothing.
_ORDER = {UNEXPECTED: 0, SILENT: 1, UNTESTED: 2, OPAQUE: 3, CONFIRMED: 4}

_ADVICE = {
    UNEXPECTED: "answered to another channel's driver; the cable is on the "
                "wrong connector or two are swapped",
    SILENT: "did not respond when driven; check the cable at",
    UNTESTED: "never checked; nothing is known about",
    OPAQUE: "cannot be checked from here",
}


def _device_of(physical_channel: str) -> str:
    return str(physical_channel or "").strip("/").split("/", 1)[0]


def _terminal_of(physical_channel: str) -> str:
    text = str(physical_channel or "").strip("/")
    return text.split("/", 1)[-1] if "/" in text else ""


@dataclasses.dataclass(frozen=True)
class ReportLine:
    """One asserted cable, what is known about it, and where to find it."""

    name: str
    physical_channel: str
    role: str
    status: str
    #: The breakout connector, when a block is named for this device.
    label: str = ""
    detail: str = ""
    checked_on: str = ""

    @property
    def is_good(self) -> bool:
        return self.status == CONFIRMED

    @property
    def needs_attention(self) -> bool:
        return self.status in (SILENT, UNEXPECTED)

    @property
    def where(self) -> str:
        """The connector if there is one, otherwise the terminal name."""
        return self.label or self.physical_channel

    def advice(self) -> str:
        """What to do about this line, naming the thing to touch."""
        if self.status == CONFIRMED:
            return ""
        text = _ADVICE.get(self.status, "")
        if self.status in (SILENT, UNTESTED):
            return f"{text} {self.where}"
        if self.status == OPAQUE:
            return f"{text}: {self.detail}" if self.detail else text
        return text


@dataclasses.dataclass(frozen=True)
class WiringReport:
    """Everything one run has to say, in the order it should be read."""

    lines: Tuple[ReportLine, ...] = tuple()
    generated_on: str = ""
    checked_on: str = ""
    #: Configuration problems found without touching the hardware.
    issues: Tuple[str, ...] = tuple()
    notes: Tuple[str, ...] = tuple()

    @property
    def counts(self) -> Dict[str, int]:
        found: Dict[str, int] = {}
        for line in self.lines:
            found[line.status] = found.get(line.status, 0) + 1
        return found

    @property
    def confirmed(self) -> int:
        return self.counts.get(CONFIRMED, 0)

    @property
    def checkable(self) -> int:
        """Points that could be checked at all - OPAQUE ones cannot."""
        return sum(1 for line in self.lines if line.status != OPAQUE)

    @property
    def is_clean(self) -> bool:
        return not any(line.status in (SILENT, UNEXPECTED, UNTESTED)
                       for line in self.lines) and not self.issues

    def headline(self) -> str:
        text = f"{self.confirmed}/{self.checkable} confirmed"
        attention = sum(1 for line in self.lines if line.needs_attention)
        if attention:
            text += f", {attention} needing attention"
        unknown = self.counts.get(UNTESTED, 0)
        if unknown:
            text += f", {unknown} never checked"
        return text


def build_report(configuration, verification: Optional[WiringVerification] = None,
                 *, issues=(), notes=()) -> WiringReport:
    """Cross what the configuration claims with what was found, and locate it.

    A missing verification is not an error: it means nothing has been checked
    yet, which the report says on every line rather than by being empty.
    """
    verification = verification or WiringVerification()
    ports = getattr(configuration, "nidaq_ports", None)
    models: Dict[str, object] = {}

    def label_for(physical_channel: str) -> str:
        device = _device_of(physical_channel)
        if device not in models:
            models[device] = breakout_for_device(ports, device)
        return describe_terminal(models[device], _terminal_of(physical_channel))

    lines = []
    for point in wiring_points(configuration):
        check = verification.check_for(point)
        if point.kind == OPAQUE:
            status, detail, checked_on = OPAQUE, point.opaque_reason, ""
        elif check is None:
            status, detail, checked_on = UNTESTED, "", ""
        else:
            status, detail, checked_on = check.status, check.detail, check.checked_on
        lines.append(ReportLine(
            name=point.name,
            physical_channel=point.physical_channel,
            role=point.role,
            status=status,
            label=label_for(point.physical_channel),
            detail=detail,
            checked_on=checked_on,
        ))

    lines.sort(key=lambda line: (_ORDER.get(line.status, 9), line.name))
    return WiringReport(
        lines=tuple(lines),
        generated_on=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        checked_on=verification.generated_on,
        issues=tuple(issues),
        notes=tuple(notes),
    )


def format_report(report: WiringReport) -> str:
    """The report as text, for a terminal and for a file beside a session."""
    out = ["NI-DAQ wiring report", "=" * 20, "",
           f"generated {report.generated_on}"]
    out.append(f"last hardware check {report.checked_on}"
               if report.checked_on else
               "no hardware check has been run")
    out.append(f"result: {report.headline()}")
    out.append("")

    if report.issues:
        out.append("Configuration problems, found without touching anything:")
        out.extend(f"  ! {issue}" for issue in report.issues)
        out.append("")

    name_width = max([len(line.name) for line in report.lines] + [4])
    channel_width = max([len(line.physical_channel) for line in report.lines] + [7])
    for line in report.lines:
        out.append(f"  {line.status.upper():<10} {line.name:<{name_width}}  "
                   f"{line.physical_channel:<{channel_width}}  "
                   f"{line.label or '-'}")
        # Only where there is something to do. An untested line's advice
        # restates the label already printed beside it, and a report whose
        # every row carries the same sentence is one nobody reads.
        advice = line.advice() if line.needs_attention else ""
        if advice:
            out.append(" " * 13 + advice)
        if line.detail and line.status not in (OPAQUE,):
            out.append(" " * 13 + line.detail)

    if report.notes:
        out.append("")
        out.extend(f"note: {note}" for note in report.notes)

    if not report.is_clean:
        out.append("")
        out.append("Nothing above is a reading of the signal itself. A "
                   "confirmed line means the")
        out.append("rig answered where the configuration said it would, not "
                   "that the value is right.")
    return "\n".join(out)


def summarize_for_status(configuration, verification: WiringVerification):
    """The one-line form the hardware panel already shows."""
    return summarize(configuration, verification)
