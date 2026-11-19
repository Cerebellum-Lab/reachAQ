# Animal metadata sources

reachAQ reads each lab's mouse records (ID, RFID, sex, date of birth, strain,
genotype, cage, protocol, state) from wherever the lab keeps them, into a local
cache on each computer. A first RFID scan or a manual link copies a record into
the animal's JSON, and every session's metadata snapshot carries it.

Each computer's user sets up **one** source:

| Kind | Reads | Needs |
|---|---|---|
| SoftMouse | the active animals of one colony, with the user's own login | a SoftMouse account |
| Spreadsheet | a `.csv` or `.xlsx` file, local or on a share | read access to the file |
| Database | one `SELECT` against a SQLite file or any SQLAlchemy URL | a server URL also needs `pip install sqlalchemy <driver>` |
| Shared publication | a snapshot another reachAQ computer publishes | read access to the share |
| A lab's own kind | whatever its plug-in reads | the lab's package installed (see below) |

Nothing about a lab is built in: no colony, path, column name or account.

## Setting a source up

Set it up as the operator who runs reachAQ on that computer, not with `sudo`.

**In the app:** **Preferences → Animal metadata → Set up…**

1. Choose the **Source** kind and fill its settings. For SoftMouse, enter the
   username and password, then click **List** to log in and choose the colony.
2. Click **Read columns**. reachAQ fetches the source once and lists its
   columns.
3. Choose the two columns that identify an animal: **Physical tag** (its
   permanent ID, also the name a new animal gets) and **RFID** (**(none)** if
   the source has no RFID column). For SoftMouse they default to Physical Tag
   and Plate ID. Every column of the source is kept in each animal's metadata;
   state, sex, date of birth, strain, genotype, cage and protocol are also read
   by name when a column matches, and the dialog lists which ones it found.
4. Decide **Skip animals without a valid RFID** (see "RFID" below).
5. Optionally, **Publish to** a shared directory (see "Sharing one source").
6. **Save**. The app reconnects to the new source; click **Refresh now** to
   import immediately.

**From a terminal** (the same steps, for a computer with no display):

```bash
cd ~/Documents/reachAQ
conda activate reachaq
reachaq-animal-metadata setup            # or: setup --kind softmouse
reachaq-animal-metadata status
reachaq-animal-metadata refresh
```

`python -m tools.acquisition.model.animal_sources.cli` is the same command when
the script is not on `PATH`.

The setup is per user: settings and the column mapping are saved as the
`animal_metadata/source` entry of that user's preferences
(`~/.config/Colorado/Auto Trainer.conf`). Another user on the same computer
sets up their own.

### Kind settings

- **SoftMouse:** username, password, colony. The export is the colony's active
  animals (Mating, Stock, Weanling, Ordered); the download stops on a
  CAPTCHA or two-factor prompt rather than guess.
- **Spreadsheet:** the file, and for `.xlsx` the worksheet (empty means the
  first). Keep RFIDs as text cells: a numeric RFID cell is refused, because a
  spreadsheet can silently round a 15-digit number.
- **Database:** a SQLite path (or `sqlite:///path`), or an SQLAlchemy URL
  **without** the password (`postgresql+psycopg://host/colony`); an optional
  username and password; and the query, one `SELECT` (or `WITH … SELECT`)
  returning a row per animal. SQLite is opened read-only. The query's column
  names are what you map.
- **Shared publication:** the publication's `.manifest.json`. The manifest
  carries the publisher's column mapping, so there is nothing to map.

## Passwords and tokens

A secret is never written to the preferences file, the shared settings
directory or the repository.

- It goes to the **OS keyring** (Secret Service on Ubuntu) when one works,
  under service `reachAQ-animal-metadata`.
- On a computer without a usable keyring (for example a headless publisher),
  it goes to `~/.config/reachAQ/animal-metadata-credentials.json`, created
  mode 600 in a mode 700 directory. reachAQ refuses to read that file if other
  users can read it. `reachaq-animal-metadata status` says which one is used.
- A SoftMouse password stored by the earlier publisher (service
  `reachAQ-softmouse-publisher`) is still found, so an existing publisher need
  not re-enter it.

To change a password, run setup again and type the new one; leave it empty to
keep the stored one.

## When it refreshes

- **Refresh now** in Preferences, or `reachaq-animal-metadata refresh`.
- **Refresh this computer's local cache daily**: at app start and at each
  midnight. For a spreadsheet or a shared publication, reachAQ first compares
  the file's hash with the last import and skips an unchanged one. SoftMouse
  and databases are always fetched.
- **Never during a session.** A refresh asked for while a session is active is
  held and runs when the session is Ready again. A running refresh blocks
  starting a session until it finishes, and one that would finish after a
  session starts is cancelled before it changes the cache.

The fetch runs in the background; the app stays responsive.

## The local cache, and working offline

Each computer keeps a SQLite cache of the source's current records
(`<settings directory>/softmouse/animal-registry.sqlite3`; the folder keeps
its first name). A refresh replaces it in one transaction, or not at all:

- The source must be reachable and return a complete snapshot. A snapshot
  that is empty, or more than 35% smaller than the last one, is refused as a
  likely filtered export.
- Duplicate IDs or RFIDs, a missing required column, or an ambiguous column
  name fail the refresh.
- On any failure the previous cache stays in use, and RFID scans keep
  resolving against it.

So a computer that loses the network, or a SoftMouse outage, keeps working
from its last good import. Animals already linked keep their metadata in their
own JSON regardless of the cache.

Switching to another source retires the old source's records from the cache.
Animals already linked keep their link and metadata.

## RFID

With **Skip animals without a valid RFID** on (the SoftMouse default), only
rows holding an exact 15-digit ISO 11784 RFID are imported; the others are
counted as "without RFID".

With it off (the default for other kinds), every active row is imported.
Animals with an RFID resolve on a scan as before. Animals without one appear
in **Animal RFID links and duplicates → Link RFID** and are linked to a local
animal by hand; a scan can never select them. A value in the RFID column that
is not a 15-digit RFID is treated as no RFID.

## Status and errors

**Preferences → Animal metadata** shows the source, the last refresh's
counts (rows, imported, with RFID, skipped, ended), the mapped and unmapped
columns, and the latest status or error, for example:

- `Refresh failed: SoftMouse rejected the stored username or password`
- `Refresh failed: Missing required source column 'Mouse ID'`
- `Source unavailable: Animal metadata source kind 'labdb' is not installed`
- `Refresh deferred until the session is Ready`

The animal registry entry in **Hardware Status** goes to Failed with the same message. A
metadata failure never blocks recording. The app log has the full record of
each refresh (source, counts, hashes) without passwords or request bodies.

## Sharing one source between computers

When only one computer should log in to the source (one SoftMouse account for
the lab, say), that computer publishes a snapshot and the others read it:

1. On the publishing computer, set up the real source and fill **Publish to**
   (and **Only if mounted** when the directory is on a network mount). Every
   refresh there then also writes the publication, before its own cache
   changes.
2. On every other computer, set up **Shared publication** with the manifest
   it writes:
   `SoftMouse-AnimalList-current.manifest.json` for SoftMouse, and
   `<kind>-AnimalList-current.manifest.json` for other kinds.

The publication is a CSV, its `.manifest.json` (written last, the completion
marker, with the hash, counts and column mapping), monthly archives under
`archive/YYYY/MM/`, and a `.publish.lock` that keeps two publishers from
writing at once. Readers verify the size and SHA-256 before importing.

To publish nightly without the app open, install the user timer on the
publishing computer:

```bash
cd ~/Documents/reachAQ
mkdir -p ~/.config/systemd/user
cp tools/softmouse_sync/systemd/reachaq-softmouse-publisher.service \
  tools/softmouse_sync/systemd/reachaq-softmouse-publisher.timer \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now reachaq-softmouse-publisher.timer
```

It runs `reachaq-animal-metadata publish` at midnight America/Denver (edit
`OnCalendar` in the installed timer for another zone). When the publication
is on a network mount, add `RequiresMountsFor=` in a drop-in
(`systemctl --user edit reachaq-softmouse-publisher.service`). Check it with
`journalctl --user -u reachaq-softmouse-publisher.service -n 100 --no-pager`.

## Adding a lab's own source

A source kind is a Python class. Subclass
`tools.acquisition.model.animal_sources.AnimalMetadataSource`, set `kind`,
`label` and `provider`, list its settings, and return the table from
`fetch()`:

```python
from tools.acquisition.model.animal_sources import (
    AnimalMetadataSource, SetupField, SourceTable,
)
from tools.acquisition.model.animal_sources.mapping import table_sha256


class LabDatabaseSource(AnimalMetadataSource):
    kind = "labdb"            # stored in the setup; must equal the entry-point name
    label = "Lab database"    # shown in the setup dialog
    provider = "labdb"        # the external-identity provider of its records

    @classmethod
    def setup_fields(cls):
        return (
            SetupField("server", "Server"),
            SetupField("username", "Username"),
            SetupField("token", "API token", kind="secret"),
        )

    def fetch(self, work_directory):
        rows = my_client(self.settings["server"], self.secret("token")).animals()
        headers = ("id", "rfid", "sex", "status")
        values = tuple(tuple(row[h] for h in headers) for row in rows)
        return SourceTable(
            headers=headers,
            rows=values,
            content_sha256=table_sha256(headers, values),
            source_name=self.describe(),
            exact_numbers=True,
        )
```

Field kinds are `text`, `path`, `secret`, `choice` and `multiline`. A `choice`
with `dynamic_choices=True` gets its options from `choices(field_name)`, which
may contact the source (SoftMouse lists colonies this way). A `secret` is
stored by the credential store and read with `self.secret(name)`.
`default_mapping()` may return a `ColumnMapping` to pre-fill the columns, and
`current_version()` may return a cheap content hash so a daily refresh of an
unchanged source is skipped.

Register it in the lab package's `pyproject.toml` and install that package
into the `reachaq` environment:

```toml
[project.entry-points."reachaq.animal_metadata_sources"]
labdb = "lab_package.sources:LabDatabaseSource"
```

The kind then appears in the setup dialog and the CLI. A plug-in that fails to
import, is not an `AnimalMetadataSource`, names a different `kind`, or reuses
a built-in name is logged and left out; it cannot stop the app.

## Computers set up before sources existed

A computer whose preferences have only the old `[softmouse] manifest_path`
reads that publication as before, with the SoftMouse columns and the old
new-animal name choice. Nothing needs changing there.

The computer that ran the nightly SoftMouse publisher does need setup once:
the colony is no longer built in. Run `reachaq-animal-metadata setup --kind
softmouse` (or `python -m tools.softmouse_sync.cli --setup`), choose the
colony, and set **Publish to** the directory that holds the old manifest and
**Only if mounted** to its mount. The old command,
`python -m tools.softmouse_sync.cli`, still publishes, so an installed timer
from before keeps working once that setup is saved.
