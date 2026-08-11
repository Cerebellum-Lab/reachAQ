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

Create a private configuration and fill in the HTTP/export values from the
authorized redacted Network capture:

```bash
mkdir -p ~/.config/reachaq
cp tools/softmouse_sync/config.example.json \
  ~/.config/reachaq/softmouse-publisher.json
chmod 600 ~/.config/reachaq/softmouse-publisher.json
```

Configure the login once. This prompts once for the username and once for the
password. The username is saved in the private configuration; the password is
saved only in the publisher account's OS keyring. The command verifies that the
keyring can return it without displaying it:

```bash
python -m tools.softmouse_sync.cli \
  --config ~/.config/reachaq/softmouse-publisher.json \
  --configure-credentials
```

All later manual publications need only the config path:

```bash
python -m tools.softmouse_sync.cli \
  --config ~/.config/reachaq/softmouse-publisher.json
```

The unit templates in `systemd/` provide the midnight schedule after their paths
are configured. The timer is persistent, so a powered-off publisher runs once
after it next starts.
