# SoftMouse/RFID operator checklist

The website paths, colony, export rules, and Isilon destination are built in.
Only the SoftMouse username and password are entered by the operator.

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

Or open **Preferences → Animal metadata** in reachAQ and click **Sync from
SoftMouse**. The button is available on every acquisition computer; run the
one-time login setup on each computer that should use it.

In the same panel:

- **Sync from SoftMouse** updates Isilon and then this computer's cache.
- **Refresh local cache** only imports the latest existing Isilon publication.
- Enable **USB RFID reader**, confirm the stable `/dev/serial/by-id/...` path,
  and click **Apply reader settings** on computers with an attached reader.
- View connection and last-scan details under **Hardware Status → RFID Reader**.

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

Do not install or enable that timer on the other computers. Their manual button
continues to work after local credential setup.

## Expected safety failures

Publication stops without replacing a good snapshot if Isilon is unmounted,
another publisher holds the lock, SoftMouse returns an incomplete or wrong-
colony result, or there are no active animals with valid 15-digit ISO 11784 RFID
values in `Plate ID`.

See [README.md](README.md) for data ownership, RFID acceptance rules, shared
files, UI behavior, and detailed troubleshooting.
