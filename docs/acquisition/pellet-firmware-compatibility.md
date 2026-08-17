# Pellet firmware compatibility and rollout

Every reachAQ release ships
`config/pellet-firmware-compatibility.yaml`, an exact-version allowlist. Startup
requests the running board version and capabilities before home/configuration.
An unknown version or missing required capability blocks pellet movement,
pellet-dependent protocols, and Record when the pellet board is required;
camera-only operation remains independent when CAN/pellet is disabled.

Current policy:

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
