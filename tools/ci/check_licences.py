"""Fail CI when an installed dependency carries an AGPL or GPL licence.

Reads the licence metadata of every distribution installed in the current
environment (standard library only, so it runs before or after the project is
installed) and compares the AGPL/GPL ones against an allowlist of known,
scheduled-for-removal exceptions. The allowlist may only shrink: an entry that
no longer matches an installed AGPL/GPL distribution is stale and also fails
the check. LGPL is allowed. Distributions whose licence cannot be identified
are reported but do not fail the check.

Usage, from the repository root:
    python -m tools.ci.check_licences --allowlist .github/licence-allowlist.txt
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

_FLAGS_NEEDING_ALLOWLIST = ("AGPL", "GPL")
_FLAGS_UNIDENTIFIED = ("unknown", "none")
_MAX_CELL_CHARS = 80
_MAX_DECLARATION_CHARS = 500  # a longer License field is a bundled-notice blob

# Checked in this order. A letter directly before the name is excluded so that
# "LGPL" does not read as "GPL" and "GPLv3" still matches.
_AGPL = re.compile(r"(?<![a-z])agpl|affero")
_LGPL = re.compile(r"(?<![a-z])lgpl|lesser general public|library general public")
_GPL = re.compile(r"(?<![a-z])gpl|general public licen[cs]e")
_PERMISSIVE = re.compile(
    r"bsd|(?<![a-z])(?:mit|apache|isc|unlicense|psf|mpl)(?![a-z])"
    r"|python software foundation|mozilla public"
)


def classify_licence(text: str | None) -> str:
    """Return "AGPL", "LGPL", "GPL", "permissive", "none" or "unknown"."""
    if text is None or not text.strip():
        return "none"
    lowered = text.lower()
    if _AGPL.search(lowered):
        return "AGPL"
    if _LGPL.search(lowered):
        return "LGPL"
    if _GPL.search(lowered):
        return "GPL"
    if _PERMISSIVE.search(lowered):
        return "permissive"
    return "unknown"


def _normalize(name: str) -> str:
    """PEP 503 name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class DistLicence:
    name: str  # PEP 503-normalized
    version: str
    licence_text: str
    flag: str


def _join(parts) -> str:
    # Old setuptools wrote the literal "UNKNOWN" when no licence was declared.
    return " | ".join(p.strip() for p in parts if p and p.strip() and p.strip().upper() != "UNKNOWN")


def _is_notice_blob(text: str) -> bool:
    """True for a License field that holds licence texts rather than a declaration."""
    return "\n" in text or len(text) > _MAX_DECLARATION_CHARS


def _licence_text(dist: metadata.Distribution) -> str:
    """The licence declared by a distribution, from its metadata.

    A License-Expression is authoritative (PEP 639) and is used alone. Otherwise
    the free-text License field is joined with the "License ::" classifiers, so
    a short "GPL-3.0-or-later" is caught even next to an MIT classifier. The one
    exception is a License field that is a bundled-notice blob (multi-line or
    long): numpy, scipy and pandas put the texts of their bundled third-party
    components (GPL, Affero, ...) there while declaring BSD in the classifiers,
    and matching the blob would flag them as AGPL/GPL. The classifiers override
    a blob only when they identify a licence; otherwise the blob is still used.
    """
    meta = dist.metadata
    expression = _join([meta.get("License-Expression")])
    if expression:
        return expression
    free_text = _join([meta.get("License")])
    classifiers = [c for c in meta.get_all("Classifier") or () if c.startswith("License ::")]
    if _is_notice_blob(free_text):
        structured = _join(classifiers)
        if classify_licence(structured) not in _FLAGS_UNIDENTIFIED:
            return structured
    return _join([free_text, *classifiers])


def installed_licences() -> list[DistLicence]:
    found: dict[str, DistLicence] = {}
    for dist in metadata.distributions():
        raw_name = dist.metadata.get("Name")
        if not raw_name:
            continue
        name = _normalize(raw_name)
        if name in found:  # same distribution visible twice on sys.path
            continue
        text = _licence_text(dist)
        found[name] = DistLicence(
            name=name,
            version=dist.version or "",
            licence_text=text,
            flag=classify_licence(text),
        )
    return sorted(found.values(), key=lambda d: d.name)


def load_allowlist(text: str) -> dict[str, str]:
    """Parse "name  # reason" lines into {normalized name: reason}."""
    allowlist: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition("#")
        name = name.strip()
        if not name or len(name.split()) != 1:
            raise ValueError(f"allowlist line {number}: expected 'name  # reason', got {line!r}")
        allowlist[_normalize(name)] = reason.strip()
    return allowlist


@dataclass
class Evaluation:
    violations: list[DistLicence]
    stale: list[str]
    unknown: list[DistLicence]

    @property
    def ok(self) -> bool:
        return not self.violations and not self.stale


def evaluate(dists: list[DistLicence], allowlist: dict[str, str]) -> Evaluation:
    needing = [d for d in dists if d.flag in _FLAGS_NEEDING_ALLOWLIST]
    needing_names = {d.name for d in needing}
    return Evaluation(
        violations=[d for d in needing if d.name not in allowlist],
        stale=sorted(name for name in allowlist if name not in needing_names),
        unknown=[d for d in dists if d.flag in _FLAGS_UNIDENTIFIED],
    )


def _cell(text: str) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    if len(text) > _MAX_CELL_CHARS:
        text = text[: _MAX_CELL_CHARS - 3] + "..."
    return text


def _table(dists: list[DistLicence]) -> list[str]:
    lines = ["| Package | Version | Class | Licence |", "| --- | --- | --- | --- |"]
    lines.extend(
        f"| {d.name} | {_cell(d.version)} | {d.flag} | {_cell(d.licence_text) or '-'} |" for d in dists
    )
    return lines


def render_markdown(ev: Evaluation, dists: list[DistLicence]) -> str:
    allowed = [d for d in dists if d.flag in _FLAGS_NEEDING_ALLOWLIST and d not in ev.violations]
    lines = [
        f"## Licence check: {'PASS' if ev.ok else 'FAIL'}",
        "",
        f"checked: {len(dists)} | violations: {len(ev.violations)} | "
        f"stale allowlist entries: {len(ev.stale)} | unknown: {len(ev.unknown)}",
    ]
    if ev.violations:
        lines += ["", "### AGPL/GPL packages not on the allowlist", ""] + _table(ev.violations)
    if ev.stale:
        lines += [
            "",
            "### Stale allowlist entries (no installed AGPL/GPL package matches; remove them)",
            "",
        ] + [f"- {name}" for name in ev.stale]
    if allowed:
        lines += ["", "### Allowlisted AGPL/GPL packages still installed", ""] + _table(allowed)
    if ev.unknown:
        lines += ["", "### Licence not identified (reported, not failed)", ""] + _table(ev.unknown)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check installed dependency licences")
    parser.add_argument("--allowlist", type=Path, required=True)
    args = parser.parse_args(argv)

    allowlist = load_allowlist(args.allowlist.read_text(encoding="utf-8"))
    dists = installed_licences()
    ev = evaluate(dists, allowlist)
    markdown = render_markdown(ev, dists)

    sys.stdout.write(markdown)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    return 0 if ev.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
