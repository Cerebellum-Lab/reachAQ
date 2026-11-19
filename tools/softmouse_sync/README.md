# SoftMouse and RFID metadata

SoftMouse is one of reachAQ's animal metadata sources. Setup, credentials,
refresh, offline behaviour, the shared publication and the nightly timer are
the same for every source and are in
[Animal metadata sources](../../docs/acquisition/animal-metadata-sources.md).
This page covers what is particular to SoftMouse and the USB RFID reader.

For a short operator checklist, see
[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).

## What the SoftMouse source does

- Logs in with the username and password of the user who set it up. Each user
  uses their own account; nothing is shared through the repository.
- Opens the colony chosen at setup. Setup logs in and lists the colonies the
  account can open; no colony is built in.
- Downloads the colony's active animals (Mating, Stock, Weanling, Ordered),
  page by page, and checks that the row count is stable across pages, every row
  belongs to the colony's owner and is active, and no animal ID repeats.
- Stops, rather than guessing, when SoftMouse asks for a CAPTCHA, two-factor or
  verification code, or rejects the login.

The export is mapped with SoftMouse's columns: `Physical Tag` is the permanent
external ID and the default new-animal name, `Plate ID` the RFID, and Sex, Date
of Birth, Strain, Genotype, Cage Tag, Protocol and State the rest. Setup lets
you choose another new-animal name column. By default only animals with a valid
RFID are imported.

The HTTPS client is `tools/softmouse_sync/https_source.py`; the source kind is
`tools/acquisition/model/animal_sources/softmouse.py`.

## Installation requirements

Run the repository's portable installer on every acquisition computer:

```bash
cd ~/Documents/reachAQ
./tools/install/reachaq-linux-install.sh
```

The normal editable package install supplies `requests`, `openpyxl`, `keyring`,
and `pyserial`. The installer also supplies the Ubuntu Secret Service/D-Bus
packages required by keyring, the complete portable Qt libraries, and
`dialout` membership for USB serial access. It verifies the Python imports,
reports whether secrets go to the OS keyring or the user-only fallback file,
checks attached `/dev/serial/by-id/` devices for read/write permission,
validates the systemd units, and exercises the metadata/RFID test suite.

If the installer adds the operator to `dialout`, log out and back in once and
rerun it. Group changes do not affect applications launched from the old login
session.

## Older setups

`python -m tools.softmouse_sync.cli --setup` now runs the SoftMouse source
setup, and `python -m tools.softmouse_sync.cli` publishes from the configured
source, so the installed `reachaq-softmouse-publisher` service keeps working
once the publishing computer has been set up. A password stored by the old
publisher (keyring service `reachAQ-softmouse-publisher`) is still found.

## RFID setup and status

In **File → Hardware**:

1. Enable **USB RFID reader** only on computers with the reader attached.
2. Use **RFID Serial Device…** to keep the stable `/dev/serial/by-id/...` path
   rather than `/dev/ttyUSB0`.
3. Each selection takes effect immediately and is persisted in the system YAML.

The **Hardware Status → RFID Reader** category reports whether the reader is
disabled, connecting, ready, stopped, or failed. Expanding it shows the serial
device, runtime reason/error, and most recent scan result. RFID failure does not
block manual recording.

Only an exact 15-digit ISO 11784 value is accepted as an RFID. The USB reader
sends a validated 26-hex transport payload; reachAQ reverses the reader's
least-significant-nibble-first animal and country fields to produce the same
15-digit value before lookup. The transport payload is never used as the
database identifier. A repeated read of the same physical tag is suppressed
briefly by the reader service.

On the first scan of an RFID that has no permanent JSON link, reachAQ offers
**Create new animal JSON**, every existing unlinked animal JSON, an editable
subject name, and persistent animal notes. Cancel creates nothing. Save writes
the UUID-named JSON, stores the RFID/source link, and selects the subject.
Subsequent scans load that JSON link and select the subject without showing the
dialog. The toolbar's RFID label shows the raw 15-digit RFID while the Subject
control shows the linked local name.

## Data ownership and safety

- The source (SoftMouse, a spreadsheet, a database) is the lab's record.
- Each computer's SQLite registry is a rebuildable cache of the current source
  records and RFID assignments.
- Each `AnimalSubject` JSON file is authoritative for the permanent reachAQ
  UUID-to-source link and keeps the linked metadata.
- The registry does not own training state or permanent local links.
- Ended animals are excluded from the cache, and so are animals without a
  valid RFID when the source requires one.
- Unknown RFID scans remain unlinked and can be resolved manually.
- Duplicate condensation keeps the chosen local JSON's name, notes, training,
  reach position, and limits; transfers the selected source/RFID link;
  archives the duplicate JSON; and then opens the survivor for editing.
- Metadata import and reconciliation are unavailable during recording.

## Troubleshooting

- **`No module named autotrainer.core`:** from the intended repository run
  `conda run -n reachaq python -m pip install -e '.[test]'`, then rerun the
  portable installer. Its import check must report the current checkout path.
- **SoftMouse rejected the stored username or password:** run setup again and
  type the current password.
- **SoftMouse authentication requires an interactive challenge:** log in once
  in a browser and clear the challenge; unattended download cannot answer it.
- **Expected exactly one SoftMouse colony named …:** the colony was renamed or
  the account lost access; run setup and choose the colony again.
- **No SoftMouse password is stored:** run setup on that computer as the
  operator, not with `sudo`.
- **RFID reader failed:** confirm it is attached at the configured
  `/dev/serial/by-id/...` path, confirm `id -nG` includes `dialout`, then apply
  reader settings or refresh hardware. Log out and back in after any group
  change.
