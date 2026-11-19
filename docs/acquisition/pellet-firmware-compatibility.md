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
| `2.3.1` | `finite_stim3_pulse`; `stepper_driver_fault` (bit 6) is optional, and only a pellet v1.3 board reports it | Rollout pending; 19 of 19 qualification checks on a pellet v1.2 board; no v1.3 board has run it; recorded-session validation pending; required on a pellet v1.3 board |
| `2.3.0` | `finite_stim3_pulse` (the first release to report its capabilities) | Rig-verified; recorded-session validation pending |
| `2.2.0` | nothing (it reports no capabilities) | Rollout pending |
| `2.1.0` | nothing (it implements no capability report or time sync) | Rollout pending |
| `2.0.0` | nothing | Legacy fallback: host-receive timing only, no finite STIM3 pulse |
| Emulator `0.1.0` | nothing | Automated tests only |

Each entry's `limitations` in the YAML file say what that version cannot do.

A pellet v1.3 board needs firmware 2.3.1 or later. On 2.3.0 or older it
accepts moves and does not move them, and reachAQ cannot tell: before 2.3.1
the board reports nothing that tells a v1.3 board from a v1.2 board.

Raw kernel/host receive time, board time, aligned board time, clock model,
uncertainty, transport estimate, and selected event time remain distinct in
`streams/device.csv`. Alignment preference is an unambiguous NI physical edge,
then a valid in-range board clock model, then kernel/host receive time. No
derived value overwrites raw evidence.

Pellet firmware 2.4.0 is the first to advertise `time_sync`; no earlier release
reports the capability. A board that advertises it makes `open()` run four
time-sync exchanges a few milliseconds apart. Earlier reachAQ fitted a clock
slope to those closely spaced exchanges and raised `ValueError` when it fell
outside 0.999 to 1.001, so it failed to open the board. The board-clock fit now
fits an offset only until the exchanges span 10 s of board time, and falls back
to an offset when a slope is implausible. `observe` still raises `ValueError`
for non-finite or out-of-order timestamps; `_translate_time_sync` catches it,
logs a warning and drops that exchange, so a bad exchange does not abort the
read. The `2.4.0` entry is therefore not in the table above. It is added
at that firmware's release, and only in a reachAQ that has this fix; a checkout
has it when `auto-trainer-device/src/autotrainer/device/board_clock.py` defines
`MIN_SLOPE_SPAN_SECONDS`.

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
- `... was refused by the board: <reason>`: the board answered the startup
  home, or a motor-configuration write, with an error. The message ends with
  the error's name and number and a closing parenthesis, for example
  `: EIO -5)`. reachAQ does not retry a refused startup on its own; the next
  home comes from Refresh Hardware or from starting System Mode. If the same
  refusal comes while reachAQ is reconnecting by itself after a CAN fault, the
  log reads `CAN recovery stopped: <reason>` instead, and recovery makes no
  further attempt and closes the connection; Refresh Hardware connects and
  homes again, and if that is refused too, act on its code as below. What to
  do for a refused startup depends on the code:
  - `EPERM -1`: E-stop asserted by another CAN client (reachAQ never sends
    one). Stop that client, or reboot the pellet board, which releases it;
    then Refresh Hardware.
  - `EIO -5`: a driver fault stopped the home, or the drivers failed to
    re-prove. Refresh Hardware once, which homes again
    (`HardwareModel._connect_locked` in
    `tools/acquisition/model/hardware_model.py`). If that home is refused
    too, reboot the pellet board and check its 12 V supply.
  - `EBUSY -16` on the home: an axis is busy or stuck. Reboot the pellet
    board.
  - Any other code, or a refused motor-configuration write: the message names
    the error and, apart from `ETIMEDOUT -116` (it says to home the pellet
    axes), gives no action. This page has none either; report the code.

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
