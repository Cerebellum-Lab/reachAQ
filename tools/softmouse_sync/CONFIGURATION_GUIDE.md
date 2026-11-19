# Animal metadata and RFID operator checklist

Nothing about the lab is built in. Each user sets up the source this computer
reads: SoftMouse with their own login, a spreadsheet, a database, or the
shared publication another computer writes. The full guide is
[Animal metadata sources](../../docs/acquisition/animal-metadata-sources.md).

## Install once on every computer

```bash
cd ~/Documents/reachAQ
./tools/install/reachaq-linux-install.sh
```

It installs the SoftMouse HTTPS, spreadsheet and keyring dependencies plus the
RFID serial package and `dialout` permission. If it says the account was added
to `dialout`, log out and back in, then rerun it. The final report should show
`PASS` for **Verify SoftMouse runtime**, **Verify RFID runtime** and **Verify
SoftMouse systemd units**.

## Set up the source, once per user

In reachAQ: **Preferences → Animal metadata → Set up…**. Choose the source,
fill its settings (for SoftMouse: username, password, then **List** and pick
the colony), click **Read columns**, check the column mapping, and **Save**.
Then click **Refresh now**.

Or from a terminal, as the operator and not with `sudo`:

```bash
cd ~/Documents/reachAQ
conda activate reachaq
reachaq-animal-metadata setup
reachaq-animal-metadata refresh
```

Passwords go to the OS keyring, or to a user-only file when the computer has
none. `reachaq-animal-metadata status` says which.

## One computer publishing for the others (optional)

On that computer, fill **Publish to** (and **Only if mounted** for a network
share) in setup, then install the nightly timer as in the guide's "Sharing one
source between computers". On the others, set up **Shared publication** with
the manifest it writes.

## RFID reader

Under **File → Hardware**, enable **USB RFID reader** and use **RFID Serial
Device…** to confirm the stable `/dev/serial/by-id/...` path. Every selection
takes effect immediately and is written to the `hardware:` block of
`system_configuration.yaml` in the settings directory. Expand **RFID Reader**
under **Hardware Status** to view connection and last-scan details.
