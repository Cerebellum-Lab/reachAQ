# SoftMouse/RFID configuration quick guide

This setup has two parts:

1. One publisher computer downloads a complete SoftMouse export to Isilon.
2. Each reachAQ computer reads that publication and uses its local RFID reader.

## Publisher computer

Create the private configuration:

```bash
mkdir -p ~/.config/reachaq
cp tools/softmouse_sync/config.example.json \
  ~/.config/reachaq/softmouse-publisher.json
chmod 600 ~/.config/reachaq/softmouse-publisher.json
nano ~/.config/reachaq/softmouse-publisher.json
```

Set these top-level values:

- `publicationDirectory`: shared Isilon folder, normally
  `/mnt/isilon/Data/BR/SoftMouse`.
- `credentials.username`: leave blank; the credential setup command fills it.
- `credentials.keyringService`: leave as `reachAQ-softmouse-publisher`.
- `lockPath`: optional. Omit it to use `.publish.lock` in the publication folder.

Set the `https` values from the authorized SoftMouse Network capture:

- `base_url`: SoftMouse HTTPS website root.
- `login_path`: login-page path, usually `login.do`.
- `export_start_path`: request that starts the complete Animals export.
- `download_path_template`: download request containing `{task_id}`.
- `csrf_field`: hidden login token field name.
- `login_username_field` and `login_password_field`: login form field names.
- `export_start_method`: `POST` or `GET`, matching the captured request.
- `export_parameters`: required export form/query values; otherwise `{}`.
- `task_id_json_path`: JSON keys leading to the export task ID.
- `authentication_check_path`: page that is available only after login.
- `authenticated_page_marker`: stable text expected on that page.
- `expected_colony_marker`: text that identifies the correct colony/account.
- `export_suffix`: `.xlsx` or `.csv`, matching the downloaded file.

Usually leave the `mapping` values as:

```json
"mapping": {
  "sheet_name": "Animal List",
  "external_identity_column": "Physical Tag",
  "rfid_column": "Alt. ID",
  "new_animal_name_column": "Physical Tag",
  "state_column": "State"
}
```

`Physical Tag` is the permanent SoftMouse identity. `Alt. ID` contains the
RFID. Change `new_animal_name_column` only if new reachAQ animals should use a
different SoftMouse column as their initial display name.

Set the safety limits under `guardrails`:

- `minimum_source_rows`: smallest believable complete Animals export. Set this
  from the normal colony size rather than leaving it at `1`.
- `maximum_fractional_row_drop`: permitted drop from the previous export.
  `0.35` permits a 35% decrease.

The publisher must always request a complete, unfiltered Animals export.

Store the username and password once:

```bash
python -m tools.softmouse_sync.cli \
  --config ~/.config/reachaq/softmouse-publisher.json \
  --configure-credentials
```

Test a publication:

```bash
python -m tools.softmouse_sync.cli \
  --config ~/.config/reachaq/softmouse-publisher.json
```

## Each reachAQ acquisition computer

Open reachAQ Preferences and find the SoftMouse/RFID section.

- **Publication manifest:** select
  `/mnt/isilon/Data/BR/SoftMouse/SoftMouse-AnimalList-current.manifest.json`.
  Select the manifest JSON, not the XLSX file.
- **Permanent external ID:** leave as `Physical Tag`.
- **RFID field:** leave as `Alt. ID`.
- **New-animal name field:** select the SoftMouse column used only when a scan
  creates a new local animal.
- **Enable USB RFID reader:** enable this on computers with the reader attached.
- **RFID serial device:** use the stable `/dev/serial/by-id/...` FTDI path.
- **Refresh local cache at midnight:** enable for automatic local updates.

Click **Apply reader settings**, then click **Refresh now**. Confirm that the
displayed totals and ignored-row counts are reasonable.

Only current animals with valid RFID values enter the local cache. Ended animals
and rows without RFID are ignored. Metadata refresh is disabled during a
recording session, and an RFID/cache failure does not block manual recording.

## Nightly publisher timer

In `reachaq-softmouse-publisher.service`, confirm:

- `ExecStart` uses the full reachAQ Conda Python path.
- `WorkingDirectory` is `/home/christielab10/Documents/reachAQ`.
- `--config` points to the private publisher JSON.

In `reachaq-softmouse-publisher.timer`:

- `OnCalendar` controls the publication time.
- `Persistent=true` runs a missed job after the user service starts again.
- `RandomizedDelaySec` permits a small randomized start delay.

Schedule the publisher before the acquisition computers' midnight refresh so
they see the newly published manifest.
