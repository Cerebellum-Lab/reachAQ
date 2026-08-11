# SoftMouse/RFID operator checklist

The website paths, colony, export rules, and Isilon destination are built in.
Only the SoftMouse username and password are entered by the operator.

## Install once on every computer

From the repository, run the normal reachAQ installer:

```bash
cd /home/christielab10/Documents/reachAQ
./tools/install/reachaq-linux-install.sh
```

It installs the SoftMouse HTTPS, spreadsheet, and operating-system keyring
dependencies plus the RFID serial package and `dialout` permission. If it says
the account was added to `dialout`, log out and back in, then rerun the same
command. The final report should show `PASS` for **Verify SoftMouse runtime**,
**Verify RFID runtime**, and **Verify SoftMouse systemd units**.

## Every computer allowed to publish

```bash
cd /home/christielab10/Documents/reachAQ
conda activate reachaq
python -m tools.softmouse_sync.cli --setup
```

Type the SoftMouse username and password at the prompts. Nothing appears while
typing the password; that is normal. The operating system keyring stores both
values. Repeat this command to replace either value.

Confirm Isilon is mounted and test publication:

```bash
findmnt /mnt/isilon
python -m tools.softmouse_sync.cli
```

The credential setup requires a logged-in graphical user session so the Linux
Secret Service can unlock the operator's keyring. Do not run it with `sudo`.

Or open **Preferences → Animal metadata** in reachAQ and click **Sync from
SoftMouse**. The button is available on every acquisition computer; run the
one-time login setup on each computer that should use it.

The metadata actions remain in that panel:

- **Sync from SoftMouse** updates Isilon and then this computer's cache.
- **Refresh local cache** only imports the latest existing Isilon publication.

RFID hardware settings are grouped with the other rig-level switches under
**Hardware Status → Hardware Configuration**. Enable **USB RFID reader**,
confirm the stable `/dev/serial/by-id/...` path, and click **Apply and save**.
The selection is written to the `hardware:` block in
`~/Autotrainer/system_configuration.yaml`. Expand **RFID Reader** in the same
panel to view connection and last-scan details.

There is no JSON configuration file to create or edit. An old
`~/.config/reachaq/softmouse-publisher.json` file is ignored and may be removed.

## Designated nightly publisher only

Do this on exactly one computer:

```bash
mkdir -p ~/.config/systemd/user
cp tools/softmouse_sync/systemd/reachaq-softmouse-publisher.{service,timer} \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now reachaq-softmouse-publisher.timer
systemctl --user list-timers reachaq-softmouse-publisher.timer
```

Check a run with:

```bash
systemctl --user start reachaq-softmouse-publisher.service
journalctl --user -u reachaq-softmouse-publisher.service -n 100 --no-pager
```

The service runs as the same user who stored the credentials. It automatically
finds the `reachaq` environment in the standard Anaconda, Miniconda, or
Mambaforge locations. If the checkout is not at
`~/Documents/reachAQ`, edit `WorkingDirectory` in the copied file under
`~/.config/systemd/user/`, run `systemctl --user daemon-reload`, and verify it
again before enabling the timer.

Do not install or enable that timer on the other computers. Their manual button
continues to work after local credential setup.

## Expected safety failures

Publication stops without replacing a good snapshot if Isilon is unmounted,
another publisher holds the lock, SoftMouse returns an incomplete or wrong-
colony result, or there are no active animals with valid 15-digit ISO 11784 RFID
values in `Plate ID`.

See [README.md](README.md) for data ownership, RFID acceptance rules, shared
files, UI behavior, and detailed troubleshooting.
