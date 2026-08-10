# SoftMouse export publisher

This is the designated-host job for the spreadsheet implementation. It logs in
with a normal HTTPS session, requests one complete Animals export, validates the
workbook, and publishes the workbook and completion manifest atomically. It does
not use a browser engine and acquisition computers do not receive credentials.

The exact export-start request is not a public SoftMouse API and is therefore not
hard-coded. Populate a private copy of `config.example.json` from an authorized,
locally redacted browser Network capture. Do not save a HAR, cookies, CSRF values,
passwords, or private configuration in this repository. A SoftMouse API adapter
can later replace `SoftMouseHttpsSource` without changing publication or imports.

Store the password in the publisher account's OS keyring:

```bash
python -m keyring set reachAQ-softmouse-publisher YOUR_USERNAME
```

Then run a manual publication:

```bash
python -m tools.softmouse_sync.cli \
  --config ~/.config/reachaq/softmouse-publisher.json \
  --username YOUR_USERNAME
```

The unit templates in `systemd/` provide the midnight schedule after their paths
and username are configured. The timer is persistent, so a powered-off publisher
runs once after it next starts.
