"""Set up and run this computer's animal metadata source from a terminal.

    reachaq-animal-metadata setup [--kind softmouse]
    reachaq-animal-metadata status
    reachaq-animal-metadata refresh     # fetch into this computer's cache
    reachaq-animal-metadata publish     # fetch and write the shared publication

``publish`` is what the systemd timer runs on the one computer that
publishes for the others.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .base import available_source_kinds
from .config import build_source, load_source_configuration, publish_target
from .credentials import CredentialStore
from .publication import AnimalTablePublisher, PublicationSource, ensure_publication_directory
from .source_setup import (
    build_mapping,
    describe_other_columns,
    list_choices,
    read_columns,
    save_setup,
    suggest_mapping,
)


logger = logging.getLogger(__name__)


def _preferences(path: Optional[Path]):
    from tools.acquisition.model.user_preferences import UserPreferences

    return UserPreferences(settings_file_path=path) if path else UserPreferences()


def _choose(prompt: Callable[[str], str], label: str, options, default: str = "") -> str:
    options = list(options)
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {option}")
    suffix = f" [{default}]" if default else ""
    while True:
        answer = prompt(f"{label}{suffix}: ").strip() or default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer in options or (answer and not options):
            return answer
        print("Enter a number from the list.")


def run_setup(
    preferences,
    credentials: CredentialStore,
    *,
    kind_name: Optional[str] = None,
    prompt: Callable[[str], str] = input,
    secret_prompt: Callable[[str], str] = getpass.getpass,
    kinds=None,
) -> str:
    kinds = available_source_kinds() if kinds is None else kinds
    existing = load_source_configuration(preferences)
    if kind_name is None:
        print("Source kinds:")
        kind_name = _choose(
            prompt,
            "Kind",
            list(kinds),
            "" if existing is None else existing.kind,
        )
    if kind_name not in kinds:
        raise ValueError(f"Unknown source kind {kind_name!r}; installed: {', '.join(kinds)}")
    kind = kinds[kind_name]
    previous = {} if existing is None or existing.kind != kind_name else dict(existing.settings)
    values = {}
    for item in kind.setup_fields():
        if item.help:
            print(f"  ({item.help})")
        if item.kind == "secret":
            values[item.name] = secret_prompt(
                f"{item.label}{'' if item.required else ' (optional)'}"
                f"{' [keep stored]' if previous else ''}: "
            )
            continue
        default = previous.get(item.name, item.default)
        if item.kind == "choice":
            options = (
                list_choices(kind, values, item.name, credentials)
                if item.dynamic_choices
                else list(item.choices)
            )
            values[item.name] = _choose(prompt, item.label, options, default)
            continue
        suffix = f" [{default}]" if default else ("" if item.required else " (optional)")
        values[item.name] = prompt(f"{item.label}{suffix}: ").strip() or default

    print("Reading the source's columns…")
    table = read_columns(kind, values, credentials)
    print(f"{len(table.rows)} rows; columns: {', '.join(h for h in table.headers if h)}")
    headers = [header for header in table.headers if header]
    suggestion = suggest_mapping(kind, headers)
    chosen = dict(suggestion)
    if existing is not None and existing.kind == kind_name and existing.mapping is not None:
        chosen.update(
            external_identity_column=existing.mapping.external_identity_column,
            new_animal_name_column=existing.mapping.new_animal_name_column,
            rfid_column=existing.mapping.rfid_column,
        )
    print("Every column is kept in each animal's metadata. Choose the ones that identify it.")
    tag_default = chosen.get("external_identity_column") or ""
    physical_tag = _choose(prompt, "Physical tag column", headers, tag_default)
    name_default = chosen.get("new_animal_name_column") or physical_tag
    name_column = _choose(prompt, "Name column (a new animal's name)", headers, name_default)
    rfid_default = chosen.get("rfid_column") or "-"
    rfid_answer = _choose(prompt, "RFID column (- for none)", headers + ["-"], rfid_default)
    rfid_column = None if rfid_answer == "-" else rfid_answer
    base = kind.default_mapping()
    require_default = (
        "y" if (existing.mapping.require_rfid if existing and existing.mapping else
                base is not None and base.require_rfid) else "n"
    )
    require_rfid = rfid_column is not None and (
        prompt(f"Skip animals without a valid RFID? (y/n) [{require_default}]: ").strip().lower()
        or require_default
    ).startswith("y")
    mapping = build_mapping(
        kind, suggestion, physical_tag, rfid_column,
        name_column=name_column, require_rfid=require_rfid,
    )
    print(f"Also read by name: {describe_other_columns(mapping)}")
    publish_directory = publish_mount = ""
    if kind_name != PublicationSource.kind:
        publish_default = "" if existing is None else existing.publish_directory
        publish_directory = prompt(
            "Also publish a shared copy to (directory, optional)"
            f"{f' [{publish_default}]' if publish_default else ''}: "
        ).strip() or publish_default
        if publish_directory:
            mount_default = "" if existing is None else existing.publish_mount
            publish_mount = prompt(
                "Only publish while this path is mounted (optional)"
                f"{f' [{mount_default}]' if mount_default else ''}: "
            ).strip() or mount_default
    saved = save_setup(
        preferences,
        credentials,
        kind,
        values,
        mapping,
        publish_directory=publish_directory,
        publish_mount=publish_mount,
    )
    preferences.save()
    for location in saved.secret_locations:
        print(f"Secret stored in {location}.")
    print(f"Saved: {build_source(saved.configuration, credentials=credentials).source.describe()}")
    return kind_name


def _configured(preferences, credentials):
    configuration = load_source_configuration(preferences)
    if configuration is None:
        raise RuntimeError(
            "No animal metadata source is set up; run `reachaq-animal-metadata setup`"
        )
    return configuration, build_source(configuration, credentials=credentials)


def run_publish(
    preferences,
    credentials: CredentialStore,
    *,
    destination: Optional[Path] = None,
    mount: Optional[Path] = None,
):
    configuration, configured = _configured(preferences, credentials)
    if configuration.kind == PublicationSource.kind:
        raise RuntimeError(
            "This computer reads a shared publication, so it has nothing to publish; "
            "set up the lab's source here with `reachaq-animal-metadata setup`"
        )
    target, target_mount = publish_target(configuration)
    destination = destination or target
    mount = mount or target_mount
    if destination is None:
        raise RuntimeError(
            "No publication directory is set up; set one in setup or pass --destination"
        )
    destination = ensure_publication_directory(destination, mount_root=mount)
    with tempfile.TemporaryDirectory(prefix="animal-publish-fetch-") as temporary:
        table = configured.source.fetch(Path(temporary))
        mapper = configured.mapper_for(table.mapping, table.provider)
        return AnimalTablePublisher(destination).publish(table, mapper)


def main(argv=None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    parser = argparse.ArgumentParser(
        prog="reachaq-animal-metadata",
        description="Set up and run this computer's animal metadata source",
    )
    parser.add_argument("--preferences", type=Path, help="user preferences file (default: the app's)")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="choose and set up the source, interactively")
    setup.add_argument("--kind", help="source kind (default: ask)")
    commands.add_parser("status", help="show the source and this computer's cache")
    commands.add_parser("refresh", help="fetch the source into this computer's cache")
    publish = commands.add_parser("publish", help="fetch the source and write the shared publication")
    publish.add_argument("--destination", type=Path, help="publication directory (default: from setup)")
    publish.add_argument("--require-mount", type=Path, help="refuse unless this path is a mount point")
    args = parser.parse_args(argv)

    try:
        preferences = _preferences(args.preferences)
        credentials = CredentialStore()
        if args.command == "setup":
            run_setup(preferences, credentials, kind_name=args.kind)
            return 0
        if args.command == "publish":
            result = run_publish(
                preferences,
                credentials,
                destination=args.destination,
                mount=args.require_mount,
            )
            print(
                f"Published {result.total_source_rows} rows ({result.tagged_rows} imported), "
                f"SHA-256 {result.sha256}"
            )
            return 0
        configuration, configured = _configured(preferences, credentials)
        from tools.acquisition.model.animal_metadata_sync import AnimalMetadataSyncService
        from tools.acquisition.model.animal_registry import AnimalRegistry

        cache = Path(preferences.configuration_location).expanduser() / "softmouse"
        registry = AnimalRegistry(cache / "animal-registry.sqlite3")
        directory, mount = publish_target(configuration)
        service = AnimalMetadataSyncService(
            configured=configured,
            local_staging_directory=cache / "source",
            registry=registry,
            can_refresh=lambda: True,
            publish_directory=directory,
            publish_mount=mount,
        )
        if args.command == "refresh":
            result = service.refresh_now()
            batch = result.preview.batch
            print(
                f"Imported {batch.accepted_rows} of {batch.total_source_rows} rows "
                f"({sum(r.physical_rfid is not None for r in batch.records)} with RFID) "
                f"from {service.describe()}"
            )
            for warning in batch.warnings:
                print(f"Warning: {warning}")
            return 0
        status = service.cache_status()
        print(f"Source: {service.describe()}")
        print(f"Secrets: {'OS keyring' if credentials.uses_keyring else credentials.fallback_path}")
        if status.last_import_utc is None:
            print("Cache: nothing imported yet")
        else:
            print(
                f"Cache: {status.tagged_rows} animals from {status.total_source_rows} rows, "
                f"imported {status.last_import_utc}"
            )
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyboardInterrupt) as exc:
        logger.exception("Animal metadata %s stopped", args.command)
        print(f"Animal metadata {args.command} stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
