# Pellet firmware compatibility and rollout

reachAQ accepts only pellet firmware versions listed in
`config/pellet-firmware-compatibility.yaml`. At startup it asks the board for
its version and capabilities before homing or configuring it. An unlisted
version, or a listed one missing a required capability, blocks pellet
movement, pellet-dependent protocols, and Record when the pellet board is
required; camera-only operation is unaffected when CAN/pellet is disabled.

## Accepted versions

| Firmware | Requires | Status |
|---|---|---|
| `2.3.0` | `finite_stim3_pulse` (the first release to report its capabilities) | Rig-verified; recorded-session validation pending |
| `2.2.0` | nothing (it reports no capabilities) | Rollout pending |
| `2.1.0` | `timing_trailer`, `time_sync`, `finite_stim3_pulse` | Rollout pending |
| `2.0.0` | nothing | Legacy fallback: host-receive timing only, no finite STIM3 pulse |
| Emulator `0.1.0` | nothing | Automated tests only |

Each entry's `limitations` in the YAML file say what that version cannot do.

Raw kernel/host receive time, board time, aligned board time, clock model,
uncertainty, transport estimate, and selected event time remain distinct in
`streams/device.csv`. Alignment preference is an unambiguous NI physical edge,
then a valid in-range board clock model, then kernel/host receive time. No
derived value overwrites raw evidence.

## Updating a rig to new firmware

Update reachAQ first, then the board. reachAQ reads the list from the checkout
it runs from, so a board flashed to a version that checkout does not list is
refused at the next startup.

1. Update reachAQ: `reachaq-sync`.
2. Flash the board with the release bundle's `reachaq-update`, following the
   firmware repository's `docs/pellet-firmware-release-and-deployment.md`. It
   checks that this rig's reachAQ accepts the version before asking for
   `FLASH`, and refuses otherwise. `--allow-unqualified` overrides that, for
   bringing up a release that is not yet listed.
3. Start reachAQ (`reachaq` or the desktop icon) and confirm the pellet
   controller connects. A version request or the CAN validator is not a
   substitute: neither applies this list.

Bundles up to and including v2.3.0 do not contain the check. With one of those,
confirm the version is listed before flashing; the first line printed is the
checkout that was read, and should be the rig's operator checkout, normally
`~/Documents/reachAQ`:

```bash
conda run -n reachaq python -c "import tools.acquisition.model.firmware_compatibility as m; r = m.FirmwareCompatibilityPolicy.load().evaluate('2.3.0'); print(m.__file__); print(r.version, 'listed' if r.supported else 'NOT LISTED', 'requires', list(r.required_capabilities))"
```

### If startup refuses the board

The hardware refresh reports `controller connection failed`, CAN safety
shutdown starts, and the log names the reason:

- `Firmware version has not been explicitly qualified (detected=X.Y.Z)`: the
  checkout reachAQ runs from does not list that version. Run `reachaq-sync`,
  restart reachAQ, and check again; if the version is still unlisted, reflash a
  listed one.
- `Required capability mismatch`: the version is listed but the board did not
  report a capability the entry requires.

## Qualifying a new firmware version

A version moves to qualified only after rig evidence and a validated recorded
session. Collect the rig evidence with the pellet board connected, lasers off or
shutters closed, and reachAQ closed:

```bash
conda run --no-capture-output -n reachaq python tools/hardware/qualify_pellet_firmware.py --expect-version 2.3.0
```

It checks version and capabilities against the list, finds where board STIM2
and STIM3 land on the NI-DAQ, confirms the tone mapping, times finite pulses and
their return to low, confirms the board refuses pulses on tone lines, counts
50 short pulses at both ends under normal CAN traffic, checks that no event
claims board time without the capability for it, and reconnects and reboots
the board. It writes a JSON record under
`~/Autotrainer/pellet_firmware_qualification/` and exits non-zero if any check
fails. It drives STIM0-STIM3 and plays tones, but moves no motor and opens no
shutter.

Then record a short session with a tone and a STIM pulse, check it with
`reachaq-validate-session --full <session>`, and update the entry's
`qualification_status` and `qualification_date` in a reviewed commit.
