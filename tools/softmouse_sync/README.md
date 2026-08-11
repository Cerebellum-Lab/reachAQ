# SoftMouse and RFID metadata

This feature synchronizes current Christie colony metadata from SoftMouse,
publishes one validated snapshot to Isilon, and builds a rebuildable SQLite
cache on each reachAQ computer. A USB RFID scan can then select, create, or
manually link a local reachAQ animal without renaming an existing animal.

For a short operator checklist, see
[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).

## Machine roles

| Machine | SoftMouse login | Manual publication | Midnight timer | Local cache |
|---|---:|---:|---:|---:|
| Designated publisher | Required | Yes | Enabled | Yes |
| Other acquisition computer | Required to publish manually | Yes | Disabled | Yes |

Exactly one machine should enable the systemd timer. Every acquisition
computer may run the same manual publication because the shared lock allows
only one writer at a time. Computers that only consume an existing publication
do not need a SoftMouse login.

## Installation requirements

Run the repository's portable installer on every acquisition computer:

```bash
cd /home/christielab10/Documents/reachAQ
./tools/install/reachaq-linux-install.sh
```

The normal editable package install supplies `requests`, `openpyxl`, `keyring`,
and `pyserial`. The installer also supplies the Ubuntu Secret Service/D-Bus
packages required by keyring, the complete portable Qt libraries, and
`dialout` membership for USB serial access. It verifies the Python imports,
selects a usable OS keyring backend, checks attached `/dev/serial/by-id/`
devices for read/write permission, validates the systemd units, and exercises
the SoftMouse/RFID test suite.

If the installer adds the operator to `dialout`, log out and back in once and
rerun it. Group changes do not affect applications launched from the old login
session. Vendor camera, NI-DAQ, CAN, and NVIDIA kernel drivers remain separate
rig-specific installs described in the main Linux installation guide.

## Fixed application defaults

These values are built into the application and are not operator configuration:

- SoftMouse colony: `Jason Christie`
- Included states: Mating, Stock, Weanling, and Ordered
- RFID source field: `Plate ID` (SoftMouse response key `plateIdPattern`)
- Permanent external identity: `Physical Tag`
- Default name for a newly created local animal: `Physical Tag`
- Publication format: normalized CSV plus a JSON completion manifest
- Shared directory: `/mnt/isilon/Data/ReachingData/SoftMouse`
- Shared manifest:
  `/mnt/isilon/Data/ReachingData/SoftMouse/SoftMouse-AnimalList-current.manifest.json`

The new-animal name field remains selectable in **Preferences → Animal
metadata**. This affects only newly created animals; synchronization never
renames an existing local animal.

## Credentials

From the repository in the reachAQ environment, run once on each machine that
may publish:

```bash
cd /home/christielab10/Documents/reachAQ
conda activate reachaq
python -m tools.softmouse_sync.cli --setup
```

Enter the SoftMouse username and password at the prompts. The password prompt
does not display characters. Both values are stored by the operating system
keyring under service `reachAQ-softmouse-publisher`; they are not passed on the
command line or written to the repository. Repeat `--setup` to replace either
credential.

Run credential setup as the normal desktop operator, not with `sudo`. The Linux
Secret Service must be available in that user's logged-in session. The nightly
user service must also run as this same account so it can read the credential.

No JSON configuration is read. A legacy
`~/.config/reachaq/softmouse-publisher.json` is ignored and may be deleted after
confirming the keyring setup works.

## Manual synchronization

The command-line operation is:

```bash
python -m tools.softmouse_sync.cli
```

The equivalent GUI action is **Preferences → Animal metadata → Sync from
SoftMouse**. It performs two ordered operations:

1. Download, validate, and atomically replace the shared Isilon publication.
2. Refresh that computer's local SQLite cache from the completed publication.

The operation runs without blocking the Qt interface. The button is disabled
during recording and while another publication launched by that application is
running. A second computer that encounters the shared publication lock stops
safely instead of writing concurrently.

**Refresh local cache** is different: it only imports the latest already
published Isilon snapshot and does not contact SoftMouse or change shared
files. **Refresh this computer's local cache daily** has the same local-only
meaning; it is not the systemd publication schedule.

## Nightly publisher setup

Run the following only on the designated publisher computer:

```bash
cd /home/christielab10/Documents/reachAQ
mkdir -p ~/.config/systemd/user
cp tools/softmouse_sync/systemd/reachaq-softmouse-publisher.service \
  tools/softmouse_sync/systemd/reachaq-softmouse-publisher.timer \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now reachaq-softmouse-publisher.timer
systemctl --user list-timers reachaq-softmouse-publisher.timer
```

The timer runs at midnight in `America/Denver`, allows up to two minutes of
randomized delay, and is persistent. If the user timer was unavailable at
midnight, systemd runs the missed job after it next starts. The tracked service
uses `~/Documents/reachAQ`, finds the `reachaq` environment in the standard
Anaconda, Miniconda, or Mambaforge locations, and requires `/mnt/isilon` to be
mounted. Edit the installed unit's `WorkingDirectory` only when the checkout
uses a nonstandard path.

Verify or run the installed job with:

```bash
systemctl --user status reachaq-softmouse-publisher.timer
systemctl --user start reachaq-softmouse-publisher.service
journalctl --user -u reachaq-softmouse-publisher.service -n 100 --no-pager
```

Disable nightly publication without affecting manual synchronization:

```bash
systemctl --user disable --now reachaq-softmouse-publisher.timer
```

## RFID setup and status

In **Preferences → Animal metadata**:

1. Enable **USB RFID reader** only on computers with the reader attached.
2. Keep the stable `/dev/serial/by-id/...` path rather than `/dev/ttyUSB0`.
3. Choose the SoftMouse field used to name a newly created local animal.
4. Click **Apply reader settings**.

The **Hardware Status → RFID Reader** category reports whether the reader is
disabled, connecting, ready, stopped, or failed. Expanding it shows the serial
device, runtime reason/error, and most recent scan result. RFID failure does not
block manual recording.

Only an exact 15-digit ISO 11784 `Plate ID` is accepted as an RFID. Empty values
and other identifiers are treated as missing RFID and ignored. Numeric
spreadsheet cells are rejected to prevent identifier corruption. The USB reader
sends a validated 26-hex transport payload; reachAQ converts its first 64 bits
from the reader's animal-ID-on-the-right ordering to the same 15-digit Plate ID
before lookup. The transport payload is never used as the database identifier.
A repeated read of the same physical tag is suppressed briefly by the reader
service.

## Data ownership and safety

- The Isilon CSV and manifest are the shared, complete publication.
- Each computer's SQLite registry is a rebuildable cache of current SoftMouse
  records and RFID assignments.
- Each `AnimalSubject` JSON file is authoritative for the permanent reachAQ
  UUID-to-SoftMouse SID link.
- The registry does not own training state or permanent local links.
- Ended animals and rows without valid RFID are excluded from the cache.
- Unknown RFID scans remain unlinked and can be resolved manually.
- Conflicting local animals are condensed only through the manual reconciliation
  workflow with explicit overwrite choices.
- Metadata import and reconciliation are unavailable during recording.

Publication validates the colony, active states, pagination consistency, row
counts, duplicate identities, RFID uniqueness, and output hash before replacing
the stable files. It refuses to replace a prior good publication when Isilon is
not mounted, no valid RFID-tagged animals exist, validation fails, or the export
looks incomplete.

## Shared files

Successful publication creates:

```text
SoftMouse-AnimalList-current.csv
SoftMouse-AnimalList-current.manifest.json
archive/YYYY/MM/SoftMouse-AnimalList-<timestamp>-<hash>.csv
.publish.lock
```

Consumers use the manifest as the completion marker and verify the CSV size and
SHA-256 before replacing their local cache. They never import an in-progress
publication.

## Troubleshooting

- **No usable keyring backend:** rerun the portable installer, log into the
  desktop as the acquisition operator, and run credential setup without
  `sudo`.
- **No credentials are stored:** rerun `python -m tools.softmouse_sync.cli
  --setup` on that computer.
- **Isilon is not mounted:** restore `/mnt/isilon`; the publisher deliberately
  will not create a look-alike local directory.
- **No valid RFID-tagged active animals:** add permanent 15-digit ISO 11784 RFID
  values to the active animals' SoftMouse `Plate ID` fields. The current publication is
  preserved.
- **Another publication is running:** wait for the other computer to finish and
  retry. Do not remove `.publish.lock`.
- **RFID reader failed:** confirm it is attached at the configured
  `/dev/serial/by-id/...` path, confirm `id -nG` includes `dialout`, then apply
  reader settings or refresh hardware. Log out and back in after any group
  change.
- **Shared publication updated but local refresh failed:** use **Refresh local
  cache** after resolving the reported manifest, mount, or recording-state
  error.
