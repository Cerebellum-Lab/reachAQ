"""Move a v3 installation's settings to the v4 locations: reachaq-migrate-settings.

* ~/Autotrainer is renamed to ~/reachAQ. A rename keeps everything in it as it
  is, so both must be on one filesystem; the command refuses otherwise.
* Every live yaml file in it that names the old directory, as an absolute path
  or as ~/Autotrainer, gets the new directory instead. Each one is kept first as
  FILE.bak-reachaq-migration. The calibration library (calibrations/) is never
  rewritten: its entries are immutable once in place. Backups such as
  FILE.yaml.bak are not live files, so they are left as they are.
* The QSettings files in ~/.config/Colorado are copied to ~/.config/reachAQ
  ("Auto Trainer.conf" becomes "reachAQ.conf"), with the same rewrite. The
  originals stay, so v3 still finds its preferences after a rollback.
* The launcher's own file, ~/.config/reachaq/launcher.conf, gets the same rewrite
  in place (kept as launcher.conf.bak-reachaq-migration). The desktop installer
  rewrites it again later; this keeps the launcher working if an install stops
  between the two.

What was done is recorded in ~/reachAQ/.settings-migration.json, which
--reverse reads to put everything back byte for byte. --dry-run prints the plan
and changes nothing.

Recorded sessions are never touched. No reader takes a settings path from a
recording: calibrations are found by id and time under the current settings
directory, and the model and demo paths a session records are read from the
live configuration, which this rewrites.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROG = "reachaq-migrate-settings"
OLD_DIR_NAME = "Autotrainer"
NEW_DIR_NAME = "reachAQ"
OLD_QSETTINGS_DIR = "Colorado"
NEW_QSETTINGS_DIR = "reachAQ"
#: QSettings files whose name changes; any other *.conf keeps its name.
QSETTINGS_RENAMES = {"Auto Trainer.conf": "reachAQ.conf"}
BACKUP_SUFFIX = ".bak-reachaq-migration"
RECORD_NAME = ".settings-migration.json"
#: Never rewritten (multires_calibration.library.LIBRARY_DIR_NAME).
IMMUTABLE_DIRS = ("calibrations",)
LIVE_SUFFIXES = (".yaml", ".yml")
#: Files outside the settings directory that name it, relative to the config home
#: (install-reachaq-desktop.sh writes the launcher's).
OUTSIDE_FILES = ("reachaq/launcher.conf",)
#: The exit status of a reachAQ start refused by unmigrated_settings_message.
UNMIGRATED_SETTINGS_EXIT = 4


class MigrationError(Exception):
    pass


@dataclass
class Plan:
    old: Path
    new: Path
    #: Live yaml files that name the old directory, relative to it.
    rewrites: List[Path] = field(default_factory=list)
    #: QSettings files to copy, (source, target).
    copies: List[Tuple[Path, Path]] = field(default_factory=list)
    #: Files outside the settings directory that name it, rewritten in place (absolute).
    outside: List[Path] = field(default_factory=list)


def config_home(home: Path, environ=os.environ) -> Path:
    """Where Qt keeps QSettings files on Linux: XDG_CONFIG_HOME, else ~/.config."""
    return Path(environ.get("XDG_CONFIG_HOME") or home / ".config")


_TILDE_OLD = b"~/" + OLD_DIR_NAME.encode()
_TILDE_NEW = b"~/" + NEW_DIR_NAME.encode()


def _prefix_pattern(old: Path) -> "re.Pattern[bytes]":
    # The old directory itself, not a longer name that starts with it (~/Autotrainer2).
    forms = [os.fsencode(str(old)), _TILDE_OLD]
    return re.compile(b"(?:" + b"|".join(re.escape(f) for f in forms) + rb")(?![\w.-])")


def _rewritten(data: bytes, old: Path, new: Path) -> bytes:
    """`data` with the old directory replaced, keeping the form it was written in."""
    return _prefix_pattern(old).sub(
        lambda m: _TILDE_NEW if m.group(0) == _TILDE_OLD else os.fsencode(str(new)), data)


def _live_files(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.parts[0] in IMMUTABLE_DIRS or path.is_symlink() or not path.is_file():
            continue
        if path.suffix in LIVE_SUFFIXES:
            yield rel


def make_plan(home: Path, cfg_home: Path) -> Optional[Plan]:
    """What a migration of `home` would do; None when there is nothing to migrate."""
    old, new = home / OLD_DIR_NAME, home / NEW_DIR_NAME
    if not old.is_dir():
        return None
    if new.exists():
        raise MigrationError(f"both {old} and {new} exist; move one of them aside first")
    plan = Plan(old, new)
    pattern = _prefix_pattern(old)
    for rel in _live_files(old):
        if pattern.search((old / rel).read_bytes()):
            backup = old / (str(rel) + BACKUP_SUFFIX)
            if backup.exists():
                raise MigrationError(f"{backup} already exists")
            plan.rewrites.append(rel)
    source_dir, target_dir = cfg_home / OLD_QSETTINGS_DIR, cfg_home / NEW_QSETTINGS_DIR
    if source_dir.is_dir():
        for source in sorted(source_dir.glob("*.conf")):
            target = target_dir / QSETTINGS_RENAMES.get(source.name, source.name)
            if target.exists():
                raise MigrationError(f"{target} already exists")
            plan.copies.append((source, target))
    for name in OUTSIDE_FILES:
        path = cfg_home / name
        if path.is_file() and not path.is_symlink() and pattern.search(path.read_bytes()):
            backup = path.with_name(path.name + BACKUP_SUFFIX)
            if backup.exists():
                raise MigrationError(f"{backup} already exists")
            plan.outside.append(path)
    return plan


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def migrate(plan: Plan) -> None:
    try:
        os.rename(plan.old, plan.new)
    except OSError as error:
        raise MigrationError(f"cannot rename {plan.old} to {plan.new}: {error}") from error
    record = {"version": 1, "old": str(plan.old), "new": str(plan.new), "rewritten": [], "copied": [],
              "outside": []}
    record_path = plan.new / RECORD_NAME

    def save():
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    save()  # before anything else changes, so --reverse can undo a run that stops part way
    for rel in plan.rewrites:
        path = plan.new / rel
        data = path.read_bytes()
        (plan.new / (str(rel) + BACKUP_SUFFIX)).write_bytes(data)
        record["rewritten"].append(str(rel))
        save()
        staging = path.with_name(path.name + ".reachaq-migration-tmp")
        staging.write_bytes(_rewritten(data, plan.old, plan.new))
        os.replace(staging, path)
    for source, target in plan.copies:
        data = _rewritten(source.read_bytes(), plan.old, plan.new)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        record["copied"].append({"source": str(source), "target": str(target), "sha256": _sha256(data)})
        save()
    for path in plan.outside:
        data = path.read_bytes()
        path.with_name(path.name + BACKUP_SUFFIX).write_bytes(data)
        record["outside"].append(str(path))
        save()
        staging = path.with_name(path.name + ".reachaq-migration-tmp")
        staging.write_bytes(_rewritten(data, plan.old, plan.new))
        os.replace(staging, path)


def reverse(home: Path) -> List[str]:
    """Undo a migration of `home` from its record. Returns warnings about anything left in place."""
    old, new = home / OLD_DIR_NAME, home / NEW_DIR_NAME
    record_path = new / RECORD_NAME
    if not record_path.is_file():
        raise MigrationError(f"{record_path} does not exist; {new} was not made by {PROG}")
    if old.exists():
        raise MigrationError(f"{old} exists; move it aside before reversing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("version") != 1 or Path(record.get("old", "")) != old:
        raise MigrationError(f"{record_path} records a migration of {record.get('old')}, not {old}")
    warnings = []
    for entry in record["copied"]:
        target = Path(entry["target"])
        if not target.exists():
            continue
        if _sha256(target.read_bytes()) == entry["sha256"]:
            target.unlink()
        else:
            warnings.append(f"kept {target}: it changed after the migration ({entry['source']} is the v3 copy)")
    for target_dir in sorted({Path(e["target"]).parent for e in record["copied"]}):
        try:
            target_dir.rmdir()
        except OSError:
            pass  # not empty: something else lives there
    for rel in record["rewritten"]:
        backup = new / (rel + BACKUP_SUFFIX)
        if backup.exists():
            os.replace(backup, new / rel)
    for staging in new.rglob("*.reachaq-migration-tmp"):
        staging.unlink()
    for name in record.get("outside", []):
        path = Path(name)
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if backup.exists():
            os.replace(backup, path)
        staging = path.with_name(path.name + ".reachaq-migration-tmp")
        if staging.exists():
            staging.unlink()
    record_path.unlink()
    os.rename(new, old)
    return warnings


def unmigrated_settings_message(home: Optional[Path] = None, environ=os.environ) -> Optional[str]:
    """Why reachAQ must not start, or None. It must not when only the v3 settings directory exists.

    Starting then would make a fresh, empty ~/reachAQ beside the operator's real settings.
    REACHAQ_CONFIG_DIR names the settings directory outright, so it is never refused.
    """
    from reachaq.core.config_directory import CONFIG_DIR_ENV

    if environ.get(CONFIG_DIR_ENV):
        return None
    home = Path.home() if home is None else home
    old, new = home / OLD_DIR_NAME, home / NEW_DIR_NAME
    if old.is_dir() and not new.exists():
        return (f"reachAQ keeps its settings in {new}, but only {old} exists.\n"
                f"Run `{PROG}` once to move them (`{PROG} --dry-run` shows what it does).")
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print what would be done and change nothing")
    parser.add_argument("--reverse", action="store_true", help="undo an earlier migration")
    # For tests: a scratch home, whose QSettings files are in <home>/.config.
    parser.add_argument("--home", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.home is None:
        home, cfg_home = Path.home(), config_home(Path.home())
    else:
        home, cfg_home = args.home, args.home / ".config"
    try:
        if args.reverse:
            if args.dry_run:
                print(f"would move {home / NEW_DIR_NAME} back to {home / OLD_DIR_NAME}, restore the rewritten"
                      f" files (launcher.conf too) from their {BACKUP_SUFFIX} copies and remove the copied"
                      f" QSettings files")
                return 0
            for warning in reverse(home):
                print(f"warning: {warning}", file=sys.stderr)
            print(f"moved {home / NEW_DIR_NAME} back to {home / OLD_DIR_NAME}")
            return 0
        plan = make_plan(home, cfg_home)
        if plan is None:
            print(f"nothing to migrate: {home / OLD_DIR_NAME} does not exist")
            return 0
        verb = "would" if args.dry_run else "will"
        print(f"{verb} rename {plan.old} to {plan.new}")
        for rel in plan.rewrites:
            print(f"{verb} rewrite {plan.new / rel} (kept as {rel}{BACKUP_SUFFIX})")
        for source, target in plan.copies:
            print(f"{verb} copy {source} to {target}")
        for path in plan.outside:
            print(f"{verb} rewrite {path} (kept as {path.name}{BACKUP_SUFFIX})")
        if not args.dry_run:
            migrate(plan)
            print("done; undo with --reverse")
        return 0
    except MigrationError as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
