# Operator alerts, first-run setup and the diagnostic bundle

## Alerts

Failures that need the operator's attention appear in a banner under the
toolbar. The banner shows the most severe alert that has not been dismissed:
what failed, the detail, and a "What to do" line. **Details…** opens the alert
log; **Dismiss** hides the alert. The status bar still shows every error line,
as before.

| Severity | Shown as | Examples |
|---|---|---|
| Critical | Banner, plus a dialog | The recording stopped early (primary camera failed, the data drive filled, writing did not finish); a camera or pose writer did not close; video, timestamps or session data incomplete; the configuration could not be loaded |
| Critical | Banner | Data drive below 1 minute (the "Recording stopped early" dialog follows) |
| Error | Banner, plus a dialog | Pellet stage stopped (an ESTOP: a move not acknowledged, the board still moving, a failed calibration capture, the connection closed during a move); it clears when Hardware Control ▸ Home succeeds |
| Error | Banner | A subsystem failed: a camera, reach camera synchronization, live pose, the pellet board (CAN), the NI-DAQ input stream, the laser, the RFID reader; a refused NI-DAQ plan; a pellet motor driver fault; recording failed to start (nothing was recorded); an internal software error |
| Warning | Banner | Data drive below 30 or 10 minutes; projected capacity below the session length; pellet axes need homing; pellet stage stopped by the operator (STOP); camera synchronization incomplete |

A failure that happens while recording says so: the NI-DAQ stream, the laser,
live pose, a secondary camera or the pellet board can fail while the recording
carries on, and the alert asks for the session to be reviewed before analysis.

Alerts for a subsystem clear themselves when the subsystem recovers, for
example after **Refresh Hardware**. A dismissed subsystem alert stays hidden
until its detail changes; a recording starting or stopping does not bring it
back. Other alerts stay until dismissed. A dialog opens at
most once a minute for the same alert, and only one dialog is open at a time.

**Help ▸ Alerts…** lists every alert since the app started, including the
ones that cleared or were dismissed.

## First-run setup

A dialog opens before the main window when:

- there is no configuration file yet (a new install), or
- the configuration's data folder does not exist or cannot be written, for
  example a drive that is not mounted.

It shows the configuration, data and log locations, checks the data folder
(writable, free space), and lists the hardware it can find without opening
any device: FLIR cameras, the NI chassis, the CAN adapter and the GPU. Missing
hardware does not stop setup. **Create configuration and start** writes a
default configuration that records into the chosen folder. For an existing
configuration, **Use this folder and start** changes only its `outputLocation`
line and keeps the previous file beside it as `*.before-first-run` (numbered,
never overwritten). The configured folder itself is never created here: if its
drive is not mounted, a folder made at the mount point would put recordings on
the system disk. Mount the drive and choose **Check again**, choose another
folder, or **Start without changing** (recording stays unavailable until the
folder can be written). **Quit** writes nothing.

The step is skipped with `--simulate`, on an offscreen display, and when
`REACHAQ_SKIP_FIRST_RUN` is set.

## Diagnostic bundle

**Help ▸ Export Diagnostic Bundle…** (also in the alert log and in every
critical dialog) writes one zip for support:

- `versions.json`: reachAQ version and git commit, Python, key packages, OS,
  kernel;
- `hardware_inventory.json`: CAN adapters, GPUs (`nvidia-smi`), NI PCI and FLIR
  USB devices, disks and free space. Nothing here opens a device;
- `subsystem_statuses.json`, `operator_alerts.json`, `app_state.json`: when
  exported from the running app;
- `configuration/` and `preferences/`: the active configuration and the
  preferences file;
- `logs/`: app logs from the last 7 days, newest first, up to 50 MB;
- `sessions/`: the small text files (JSON, YAML, CSV, TXT up to 2 MB each,
  20 MB in all) of the 5 most recent sessions. No video and no HDF5;
- `manifest.json`: what was included, what was skipped and why.

When the app will not start, build it from a terminal:

```
reachaq-diagnostics -o ~/reachaq-diagnostics.zip
```

`-c` names the configuration, `--preferences-file` the preferences, and
`--data-dir` (repeatable) the folders to search for logs and sessions.

From the app, the bundle is built by this same command in a separate process,
so redacting the logs does not slow the running app. While a recording runs it
skips `nvidia-smi`, which briefly stalls live pose inference.

### Secrets

- The OS keyring is never read.
- Credential files (the animal-metadata credential file, keys, `.netrc`) are
  never added, and only files the bundle names are read.
- In every file, values of fields named like password, token, secret, API key,
  credential, authorization or cookie are replaced with `<redacted>`, as are
  credentials in URLs and bearer tokens.
- Values stored in the local credential file are removed wherever they appear,
  and the finished bundle is searched for them again. If one is found, the
  export fails and no file is written.
- A secret held only in the OS keyring is protected by the field redaction
  alone, since its value is never read. The manifest says so.
