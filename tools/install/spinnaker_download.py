"""Find and unpack the customer's own Spinnaker download for the installer.

reachAQ does not ship FLIR software. The operator downloads the Spinnaker SDK
for Linux from Teledyne FLIR into a folder (~/Downloads by default) and
reachaq-linux-install.sh takes the SDK packages and the Python binding from
there. Teledyne ships both in one Spinnaker-<version>-Linux.zip; the zip, its
unpacked folder, or the two files on their own are all accepted:

    spinnaker-<version>-amd64-pkg-<ubuntu>.tar.gz              the SDK .debs
    spinnaker_python-<version>-cp310-cp310-linux_x86_64.tar.gz  the binding
    (or the .whl itself)

Standard library only: it runs with the host's python3, before and outside the
reachAQ environment.

    python3 spinnaker_download.py supported <version>
    python3 spinnaker_download.py extract-sdk --dir D --dest T --ubuntu 22.04 --arch amd64
    python3 spinnaker_download.py extract-python --dir D --dest T --sdk-version V \\
        --python cp310 --machine x86_64

extract-* print the unpacked path on stdout. Exit 0 on success, 1 when nothing
usable is found (with what to download on stderr), 2 on a bad argument.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, List, Optional, Sequence, Tuple

from platform_check import DEFAULT_SPEC, load_spec

DOWNLOAD_PAGE = "https://www.teledynevisionsolutions.com/products/spinnaker-sdk/"
VERSION = r"(\d+(?:\.\d+){3})"


@dataclass(frozen=True)
class Candidate:
    version: Tuple[int, ...]
    path: Path            # the file, or the zip that holds it
    member: Optional[str]  # its name inside that zip

    @property
    def text(self) -> str:
        return ".".join(map(str, self.version))

    @property
    def where(self) -> str:
        return f"{self.path}:{self.member}" if self.member else str(self.path)


def parse_version(text: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in text.strip().split("."))


def series(version: Tuple[int, ...]) -> Tuple[int, ...]:
    return version[:2]


def supported_series(spec: dict) -> List[Tuple[int, ...]]:
    return [parse_version(text) for text in spec["spinnaker"]["versions"]]


def is_supported(spec: dict, version: Tuple[int, ...]) -> bool:
    return series(version) in supported_series(spec)


def _search_roots(directory: Path) -> List[Path]:
    """The folder, and the folders directly inside it (an unpacked zip)."""
    if not directory.is_dir():
        return []
    return [directory, *sorted(path for path in directory.iterdir() if path.is_dir())]


def find_candidates(directory: Path, pattern: re.Pattern) -> Iterator[Candidate]:
    for root in _search_roots(directory):
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if match:
                yield Candidate(parse_version(match.group(1)), path, None)
            elif path.suffix == ".zip":
                try:
                    with zipfile.ZipFile(path) as archive:
                        names = archive.namelist()
                except (OSError, zipfile.BadZipFile):
                    continue
                for name in names:
                    match = pattern.match(PurePosixPath(name).name)
                    if match:
                        yield Candidate(parse_version(match.group(1)), path, name)


def sdk_pattern(arch: str, ubuntu: str) -> re.Pattern:
    return re.compile(rf"^spinnaker-{VERSION}-{re.escape(arch)}-pkg-{re.escape(ubuntu)}\.tar\.gz$")


def python_pattern(python_tag: str, machine: str) -> re.Pattern:
    tag = re.escape(python_tag)
    return re.compile(
        rf"^spinnaker_python-{VERSION}-{tag}-{tag}-linux_{re.escape(machine)}\.(?:whl|tar\.gz)$")


def choose_sdk(spec: dict, candidates: Sequence[Candidate]) -> Optional[Candidate]:
    usable = [candidate for candidate in candidates if is_supported(spec, candidate.version)]
    return max(usable, key=lambda candidate: candidate.version, default=None)


def choose_python(candidates: Sequence[Candidate],
                  sdk_version: Tuple[int, ...]) -> Optional[Candidate]:
    """The binding built for this SDK release, else the newest of its series."""
    exact = [candidate for candidate in candidates if candidate.version == sdk_version]
    if exact:
        return exact[0]
    same_series = [candidate for candidate in candidates
                   if series(candidate.version) == series(sdk_version)]
    return max(same_series, key=lambda candidate: candidate.version, default=None)


def _materialise(candidate: Candidate, dest: Path) -> Path:
    """The candidate as a plain file under dest (unzipped when it is in a zip)."""
    if candidate.member is None:
        return candidate.path
    with zipfile.ZipFile(candidate.path) as archive:
        return Path(archive.extract(candidate.member, dest))


def _safe_extract(tar_path: Path, dest: Path, wanted=lambda member: True) -> None:
    with tarfile.open(tar_path, "r:gz") as archive:
        members = []
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"{tar_path}: unsafe path {member.name!r}")
            if not (member.isfile() or member.isdir()):
                continue
            if wanted(member):
                members.append(member)
        if hasattr(tarfile, "data_filter"):
            archive.extractall(dest, members=members, filter="data")
        else:
            archive.extractall(dest, members=members)


def extract_sdk(candidate: Candidate, dest: Path) -> Path:
    archive = _materialise(candidate, dest)
    _safe_extract(archive, dest, lambda member: member.isdir() or member.name.endswith(".deb"))
    for deb in sorted(dest.rglob("libspinnaker_*.deb")):
        return deb.parent
    raise ValueError(f"{candidate.where} holds no libspinnaker_*.deb")


def extract_python(candidate: Candidate, dest: Path) -> Path:
    found = _materialise(candidate, dest)
    if found.name.endswith(".whl"):
        return found
    _safe_extract(found, dest, lambda member: member.name.endswith(".whl"))
    wheels = sorted(dest.rglob("spinnaker_python-*.whl"))
    if not wheels:
        raise ValueError(f"{candidate.where} holds no spinnaker_python wheel")
    return wheels[0]


def _missing(spec: dict, what: str, directory: Path, found: Sequence[Candidate]) -> str:
    versions = " or ".join(spec["spinnaker"]["versions"])
    lines = [
        f"No usable {what} in {directory}.",
        f"Download Spinnaker {versions} for Linux from {DOWNLOAD_PAGE}",
        f"(a free Teledyne account is needed) and save the Spinnaker-<version>-Linux.zip in {directory},",
        "or point REACHAQ_SPINNAKER_DOWNLOADS at the folder that holds it, then rerun the installer.",
    ]
    if found:
        lines.append("Found, but not usable here: " + ", ".join(
            f"{candidate.text} ({candidate.where})" for candidate in found))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    commands = parser.add_subparsers(dest="command", required=True)
    supported = commands.add_parser("supported")
    supported.add_argument("version")
    sdk = commands.add_parser("extract-sdk")
    sdk.add_argument("--dir", type=Path, required=True)
    sdk.add_argument("--dest", type=Path, required=True)
    sdk.add_argument("--ubuntu", required=True)
    sdk.add_argument("--arch", required=True)
    binding = commands.add_parser("extract-python")
    binding.add_argument("--dir", type=Path, required=True)
    binding.add_argument("--dest", type=Path, required=True)
    binding.add_argument("--sdk-version", required=True)
    binding.add_argument("--python", required=True)
    binding.add_argument("--machine", required=True)
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)

    version_text = {"supported": getattr(args, "version", None),
                    "extract-python": getattr(args, "sdk_version", None)}.get(args.command)
    if version_text is not None:
        try:
            version = sdk_version = parse_version(version_text)
        except ValueError:
            print(f"not a Spinnaker version: {version_text!r}", file=sys.stderr)
            return 2

    if args.command == "supported":
        if is_supported(spec, version):
            print(f"Spinnaker {args.version} is supported")
            return 0
        print(f"Spinnaker {args.version} is not supported; supported: "
              + ", ".join(spec["spinnaker"]["versions"]), file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "extract-sdk":
            found = list(find_candidates(args.dir, sdk_pattern(args.arch, args.ubuntu)))
            chosen = choose_sdk(spec, found)
            if chosen is None:
                print(_missing(spec, f"Spinnaker SDK for Ubuntu {args.ubuntu} {args.arch}",
                               args.dir, found), file=sys.stderr)
                return 1
            print(f"Using the Spinnaker {chosen.text} SDK from {chosen.where}", file=sys.stderr)
            print(extract_sdk(chosen, args.dest))
            return 0
        found = list(find_candidates(args.dir, python_pattern(args.python, args.machine)))
        chosen = choose_python(found, sdk_version)
        if chosen is None:
            print(_missing(spec, f"Spinnaker Python binding for SDK {args.sdk_version} "
                                 f"({args.python}, {args.machine})", args.dir, found),
                  file=sys.stderr)
            return 1
        print(f"Using the Spinnaker {chosen.text} Python binding from {chosen.where}",
              file=sys.stderr)
        print(extract_python(chosen, args.dest))
        return 0
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"Cannot unpack the Spinnaker download: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
