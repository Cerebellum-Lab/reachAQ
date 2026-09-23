# Pellet firmware compatibility and rollout

Every reachAQ release ships
`config/pellet-firmware-compatibility.yaml`, an exact-version allowlist. Startup
requests the running board version and capabilities before home/configuration.
An unknown version or missing required capability blocks pellet movement,
pellet-dependent protocols, and Record when the pellet board is required;
camera-only operation remains independent when CAN/pellet is disabled.

Current policy:

- Pellet firmware `2.3.0`: requires `finite_stim3_pulse`, which this release
  is the first to report, through `CAPABILITIES_REQUEST` (`0x23`).
- Pellet firmware `2.2.0`: requires nothing; it answers no capability request,
  so every capability is optional rather than impossible to satisfy.
- Pellet firmware `2.1.0`: requires `timing_trailer`, `time_sync`, and
  `finite_stim3_pulse`; rollout/physical qualification is still pending.
- Pellet firmware `2.0.0`: accepted legacy fallback with host-receive timing;
  it cannot supply board-time confidence or finite First Reach STIM3.
- Emulator `0.1.0`: automated testing only, never physical qualification.

Raw kernel/host receive time, board time, aligned board time, clock model,
uncertainty, transport estimate, and selected event time remain distinct in
`streams/device.csv`. Alignment preference is an unambiguous NI physical edge,
then a valid in-range board clock model, then kernel/host receive time. No
derived value overwrites raw evidence.

The coordinated firmware work lives in the sibling `reachAQ-hardware` repository
and is released as `v2.1.0`. Follow its
`docs/pellet-firmware-operator-quick-start.md` and
`docs/releases/v2.1.0.md`. The rollout is not complete until the board is flashed
and retained real-rig version/capability, reconnect, Tone 1/Tone 2, STIM3,
NI/board alignment, CAN-load, and validator evidence passes. Only then update
the allowlist qualification status/date in a reviewed reachAQ commit.

## Roll out the host before the board

The allowlist is read from whichever checkout the rig's `reachaq` command
imports, not from the branch the allowlist entry was committed to. A board
flashed to a version that checkout does not list is refused at the next
startup, the pellet controller does not connect, and CAN safety shutdown
starts. That refusal is correct and must stay; the mistake is flashing first.

So on every rig, in this order:

1. Get a reachAQ build that lists the new version onto the rig's installed
   checkout, normally with `reachaq-sync`.
2. Confirm that the checkout `reachaq` actually runs lists it. This names the
   file it read, which is the point: it must be the operator's checkout, not a
   development copy elsewhere on the machine.

   ```bash
   conda run -n reachaq python -c "import tools.acquisition.model.firmware_compatibility as m; r = m.FirmwareCompatibilityPolicy.load().evaluate('2.3.0'); print(m.__file__); print(r.version, 'listed' if r.supported else 'NOT LISTED', 'requires', list(r.required_capabilities))"
   ```

3. Only then flash the board.
4. Start reachAQ with the operator's own command (`reachaq`, or the desktop
   icon) and confirm the pellet controller connects. A version request or a
   validator script is not enough: neither applies this policy, so both pass
   against a board the application will refuse.

What this prevents: on 2026-09-23 christielab10's board was flashed to 2.3.0
and the release was qualified on `feature-dev` and `devel`, and connected
cleanly when tested from a development checkout. The operator's `reachaq`
imported `~/Documents/reachAQ`, which was still on the closed
`reach-training-protocol-features` branch with an allowlist ending at 2.1.0,
and startup failed with `Firmware version has not been explicitly qualified
(detected=2.3.0)`. That checkout did not list 2.2.0 either, so the gap
predated the flash; 2.3.0 is what exposed it.
