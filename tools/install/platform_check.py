"""Check this host against tools/install/supported-platforms.json.

Run by reachaq-linux-install.sh with the host's python3 before anything is
installed, so it needs only the standard library and must not import reachAQ.

    python3 platform_check.py [--spec FILE] [--os-release FILE] [--machine NAME]

Prints one line per check, "ok: ..." or "unsupported: ...", and exits 0 when
every check passed, 1 when any found an unsupported platform and 2 when the
list itself cannot be read.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

DEFAULT_SPEC = Path(__file__).resolve().with_name("supported-platforms.json")
NVIDIA_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,driver_version,compute_cap",
    "--format=csv,noheader",
]

Finding = Tuple[bool, str]


def load_spec(path: Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        spec = json.load(stream)
    if spec.get("schema") != 1:
        raise ValueError(f"{path}: unknown schema {spec.get('schema')!r}")
    return spec


def version_tuple(text: str) -> Tuple[int, ...]:
    """'570.26' -> (570, 26). Raises ValueError for anything not dotted numbers."""
    return tuple(int(part) for part in text.strip().split("."))


def parse_os_release(text: str) -> dict:
    values = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def check_os(spec: dict, os_release: dict) -> Finding:
    found_id = os_release.get("ID", "").lower()
    found_version = os_release.get("VERSION_ID", "")
    found = os_release.get("PRETTY_NAME") or f"{found_id} {found_version}".strip() or "unknown"
    supported = spec["operating_systems"]
    for entry in supported:
        if entry["id"] == found_id and entry["version_id"] == found_version:
            return True, f"operating system {found}"
    names = ", ".join(entry["name"] for entry in supported)
    return False, f"operating system {found}; supported: {names}"


def check_architecture(spec: dict, machine: str) -> Finding:
    supported = spec["architectures"]
    if machine in supported:
        return True, f"architecture {machine}"
    return False, f"architecture {machine}; supported: {', '.join(supported)}"


def check_nvidia(spec: dict, query_output: Optional[str]) -> List[Finding]:
    """query_output is the nvidia-smi CSV, or None when nvidia-smi is unusable."""
    nvidia = spec["nvidia"]
    if query_output is None:
        return [(False, "no NVIDIA driver: nvidia-smi is missing or failed. Install the "
                        f"NVIDIA driver {nvidia['min_driver']} or newer "
                        "(docs/linux-install/nvidia-inference.md), then rerun")]
    rows = [line for line in query_output.splitlines() if line.strip()]
    if not rows:
        return [(False, "no NVIDIA GPU reported by nvidia-smi")]
    minimum_driver = version_tuple(nvidia["min_driver"])
    lowest = version_tuple(nvidia["min_compute_capability"])
    highest = version_tuple(nvidia["max_compute_capability"])
    findings = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3:
            findings.append((False, f"cannot read nvidia-smi output: {row!r}"))
            continue
        name, driver, capability = fields
        try:
            driver_ok = version_tuple(driver) >= minimum_driver
            capability_ok = lowest <= version_tuple(capability) <= highest
        except ValueError:
            findings.append((False, f"{name}: cannot read driver {driver!r} "
                                    f"or compute capability {capability!r}"))
            continue
        detail = f"{name}, driver {driver}, compute capability {capability}"
        problems = []
        if not driver_ok:
            problems.append(f"driver must be {nvidia['min_driver']} or newer")
        if not capability_ok:
            problems.append("compute capability must be "
                            f"{nvidia['min_compute_capability']} to "
                            f"{nvidia['max_compute_capability']}")
        if problems:
            findings.append((False, f"GPU {detail}; " + "; ".join(problems)))
        else:
            findings.append((True, f"GPU {detail}"))
    return findings


def query_nvidia(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Optional[str]:
    if shutil.which(NVIDIA_QUERY[0]) is None:
        return None
    try:
        completed = run(NVIDIA_QUERY, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def check_host(spec: dict, os_release: dict, machine: str,
               nvidia_output: Optional[str]) -> List[Finding]:
    return [
        check_os(spec, os_release),
        check_architecture(spec, machine),
        *check_nvidia(spec, nvidia_output),
    ]


def print_findings(findings: Iterable[Finding]) -> bool:
    all_ok = True
    for ok, text in findings:
        print(("ok: " if ok else "unsupported: ") + text)
        all_ok = all_ok and ok
    return all_ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--os-release", type=Path, default=Path("/etc/os-release"))
    parser.add_argument("--machine", default=platform.machine())
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.spec)
    except (OSError, ValueError, KeyError) as error:
        print(f"cannot read the supported platform list: {error}", file=sys.stderr)
        return 2
    try:
        os_release = parse_os_release(args.os_release.read_text(encoding="utf-8"))
    except OSError:
        os_release = {}
    findings = check_host(spec, os_release, args.machine, query_nvidia())
    return 0 if print_findings(findings) else 1


if __name__ == "__main__":
    sys.exit(main())
