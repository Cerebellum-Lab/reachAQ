# Pellet firmware: the GPIO pulse command

> **Resolved in pellet firmware v2.2.0** (`reachAQ-hardware` tag `v2.2.0`,
> commit `01ee28b`), flashed and verified on `christielab10` on 2026-09-17.
> This document is kept as the specification that release was built against and
> as the record of what was wrong before it. Everything below describes
> v2.1.0 and earlier.

The host drives the `hardware_stim3` laser trigger route by asking the pellet
board for a finite, firmware-timed digital pulse. **No pellet firmware through
v2.1.0 implements that command.** The board drops the frame, so the route does
not trigger anything on real hardware.

It does not fail silently end to end: the board sends no acknowledgement for a
command it has no handler for, so the caller's `wait_pending_command_acked`
raises a timeout roughly three seconds later. `pulse_digital_output` itself
does return success, because it only reports that the frame was queued locally,
but the layer above catches the missing acknowledgement.

This document is the implementation specification for adding it, for **STIM2 and
STIM3**, and the evidence that it was absent.

## Status

| | |
| --- | --- |
| Affected firmware | all releases through v2.1.0 |
| Affected host | `CanInterface.pulse_digital_output`, the `hardware_stim3` laser route |
| Symptom | a trial or bench test arms the NI analog output and waits for a trigger edge that never arrives |
| Host-visible | `pulse_digital_output` returns success for a frame the board ignores, but the board sends no acknowledgement either, so the caller's `wait_pending_command_acked` raises a timeout about three seconds later |
| Confirmed | 2026-09-17 on `christielab10` against a live board, firmware 2.1.0 |

## Evidence that it is absent

**Source.** The firmware command enum in
`software/libjerrycan/inc/jerrycan_types.h` ends at `SERVO_DETACH = 0x1E`, plus
`JERRYCAN_RSP_ACK = 0x30` and the `MIN`/`MAX` bounds. The host sends
`GPIO_PULSE = 0x21`, `GPIO_PULSE_STATUS = 0x22`, `CAPABILITIES_REQUEST = 0x23`,
`CAPABILITIES_RESPONSE = 0x24` and `TIME_SYNC_REQUEST/RESPONSE = 0x1F/0x20`.
None of those five exist on the board.

`firmware/lib/jerrycan/modules/gpio.c` implements exactly two things: a periodic
`GPIO_READ` transmit and a `GPIO_WRITE` on/off handler. There is no pulse
command, no timer, and no return-low guarantee.

`git log --all -S "GPIO_PULSE"` across the whole hardware repository returns
nothing, on any branch, through tag v2.1.0. It has never been implemented.

**Hardware.** A `candump` capture on `can0` while the host sent one known-good
command and one command under test. The CAN arbitration id is
`command_type << 5`.

Control, `GPIO_WRITE` (`0x160`), which the firmware does implement:

```
(001.459813) can0 160 [05] 00 07 00 01 02     <- STIM3 high
(002.475411) can0 160 [05] 00 07 00 00 03     <- STIM3 low
```

The board answered. `GPIO_READ` (`0x140`) reported payload
`00 80 00 00 00 00` — bit 7, STIM3 — for ten consecutive 100 ms reports from
1.553 s to 2.454 s, bracketing the write exactly. The board is alive and
responsive.

Test, `GPIO_PULSE` (`0x420`):

```
(003.491264) can0 420 [08] 00 07 00 20 A1 07 00 04
```

Correctly encoded by the host: instance `0`, `gpio_idx` `7` (STIM3), duration
`0x0007A120` = 500000 us, then the uuid trailer. Every `GPIO_READ` after that
moment reports `00 00 00 00 00 00`. **STIM3 never rose**, through the whole
500 ms the pulse should have lasted and beyond.

## Pin mapping, and a naming trap

The board device tree names its four stimulus outputs `STIM0`..`STIM3`:

| Board name | MCU pin | `gpio_idx` on the wire | Host `DigitalOutputs` | Free for stim? |
| --- | --- | --- | --- | --- |
| STIM0 | `gpiob 11` | 4 | `STIMULUS_1` | **no** — Tone 1 TTL confirmation |
| STIM1 | `gpiob 12` | 5 | `STIMULUS_2` | **no** — Tone 2 TTL confirmation |
| STIM2 | `gpiob 13` | 6 | `STIMULUS_3` | yes |
| STIM3 | `gpiob 14` | 7 | `STIMULUS_4` | yes |

Two hazards here.

- **The host names are off by one from the board names.** Host `STIMULUS_4` is
  board `STIM3`. Read every identifier carefully; the existing host comment
  "Physical STIM3 = the fourth logical stimulus output = GPIO 0:7" is correct.
- **STIM0 and STIM1 are not available.** `tone_generator.c` drives `gpiob 11`
  and `gpiob 12` directly as the Tone 1 (5 kHz) and Tone 2 (6 kHz) TTL
  confirmations the NI-DAQ records, declared in the DTS as
  `tone-output-gpios`. A stim pulse on those pins would contend with tone
  confirmation. **Implement the pulse for STIM2 and STIM3 only**, and reject
  `gpio_idx` 4 and 5 with an explicit error rather than pulsing them.

`gpio_idx` is an index into the driver's *readable* pin names, which are offset
by the inputs — see `ll_generic_gpio_lookup_readable_pin_name`. The capture
confirms index 7 reaches STIM3, so the existing offset is right; do not
recompute it.

## Wire protocol to implement

Both payloads are little-endian and packed. The uuid byte is appended after the
payload by the transport, as for every other command.

### `GPIO_PULSE = 0x21`, host to board, 7 bytes

| Field | Type | Notes |
| --- | --- | --- |
| `instance` | `uint8` | generic-gpio instance, `0` on the pellet board |
| `gpio_idx` | `uint16` | readable-pin index: `6` = STIM2, `7` = STIM3 |
| `duration_us` | `uint32` | host already validates 100 us .. 5 s |

Firmware struct to add beside `jerrycan_cmd_gpio_write_t`:

```c
typedef struct __attribute__((packed)) {
    uint8_t instance;
    uint16_t gpio_idx;
    uint32_t duration_us;
} jerrycan_cmd_gpio_pulse_t;

SIZE_CHECK(jerrycan_cmd_gpio_pulse_t, 7);
```

### `GPIO_PULSE_STATUS = 0x22`, board to host, 12 bytes

| Field | Type | Notes |
| --- | --- | --- |
| `instance` | `uint8` | echoed |
| `gpio_idx` | `uint16` | echoed |
| `duration_us` | `uint32` | echoed, the duration actually applied |
| `phase` | `uint8` | see below |
| `error` | `int32` | zero on success, negative errno otherwise |

```c
typedef struct __attribute__((packed)) {
    uint8_t instance;
    uint16_t gpio_idx;
    uint32_t duration_us;
    uint8_t phase;
    int32_t error;
} jerrycan_cmd_gpio_pulse_status_t;

SIZE_CHECK(jerrycan_cmd_gpio_pulse_status_t, 12);
```

`phase` is not yet pinned down by any host consumer, so define it here and we
will match it on the host: `0` = rejected, `1` = asserted (line went high),
`2` = completed (line returned low). Emit at least the completion status; the
assertion status is useful for latency work and cheap to add.

## Firmware changes

1. **`software/libjerrycan/inc/jerrycan_types.h`**
   - add `JERRYCAN_CMD_GPIO_PULSE = 0x21` and
     `JERRYCAN_CMD_GPIO_PULSE_STATUS = 0x22` to `jerrycan_cmd_type_t`
   - add both structs above with their `SIZE_CHECK`s
   - add both to the payload union that already carries `gpio_write` / `gpio_read`
2. **`firmware/lib/jerrycan/jerrycan.c`** — add both entries to
   `jerrycan_size_map`. Without this the payload size resolves to zero and the
   frame is discarded before dispatch, which is the present failure.
3. **`firmware/lib/jerrycan/modules/gpio.c`** — add the handler:
   - register a second `jerrycan_rx_callback_t` filtered on
     `JERRYCAN_CMD_GPIO_PULSE`
   - resolve the instance the same way `jerrycan_generic_gpio_write_handler`
     does, then the pin via `ll_generic_gpio_lookup_readable_pin_name`
   - **reject `gpio_idx` 4 and 5** (STIM0/STIM1, owned by the tone generator)
     with `-EPERM`, and reject a duration outside 100 us .. 5 s with `-EINVAL`
   - reject a pulse on a line whose pulse is already in flight with `-EBUSY`
     rather than extending or restarting it
   - assert the line, schedule the return-low, and emit `GPIO_PULSE_STATUS`
   - the handler's return code flows into the existing
     `jerrycan_send_ack(uuid, error)`, so a rejection still acknowledges

**The return-low must be guaranteed.** If the pulse is armed, the line comes
back low even if the board is busy, another command arrives, or the handler
errors after assertion. A missed return-low leaves a laser gate asserted.

## The timing decision, which needs a real choice

The host permits 100 us to 5 s. A Zephyr `k_timer` is tick-granular, and this
application sets no explicit tick rate, so it inherits the default. At a 10 kHz
tick that is 100 us of granularity — the same size as the shortest permitted
pulse — and at 100 Hz it is hopeless. A software timer therefore cannot honour
the short end of the range with the accuracy an optogenetics pulse needs.

Options, in the order I would consider them:

1. **Drive it from a hardware timer** (`CONFIG_COUNTER` with an STM32 timer, or
   a one-shot PWM channel). Gives sub-microsecond edges independent of the
   scheduler, which is what the stim path actually wants. Most work, right
   answer.
2. **Raise `CONFIG_SYS_CLOCK_TICKS_PER_SEC`** and use `k_timer` with
   `K_USEC()`. Cheap, but it taxes the whole system's interrupt load and still
   leaves jitter under load, so validate the pellet motion paths afterwards.
3. **Busy-wait in the handler with `k_busy_wait()`** for short pulses only.
   Accurate, but it blocks the jerrycan RX thread for the pulse duration, which
   is unacceptable at anything approaching 5 s.

Whoever takes this should pick 1 unless there is a reason not to, and say in the
release notes what the measured edge accuracy is.

## Capability reporting

The host gates on a capability bitmask it never receives, because
`CAPABILITIES_REQUEST = 0x23` is also unimplemented. Two parts:

- Implementing `0x23` and replying `CAPABILITIES_RESPONSE = 0x24` with
  `wire_schema_version:uint8, capabilities:uint32, boot_id:uint32` (9 bytes)
  would let the host learn what a board can do. Worth doing, but it is a
  separate piece of work from the pulse.
- The relevant bit is `1 << 2`, named `finite_stim3_pulse` in
  `tools/acquisition/model/firmware_compatibility.py`. The name is legacy: set
  it to mean "the finite GPIO pulse command is implemented", covering STIM2 and
  STIM3.

The host tolerates a board that reports nothing — it refuses only a board that
reports capabilities and omits this bit — so the pulse can ship before
capability reporting does.

## Host changes, once firmware ships

1. `auto-trainer-device/src/autotrainer/device/can_interface.py` —
   `pulse_digital_output` currently raises unless the target is `STIMULUS_4`.
   Widen it to `STIMULUS_3` and `STIMULUS_4` (board STIM2 and STIM3), still
   rejecting `STIMULUS_1`/`STIMULUS_2` with the tone-contention reason.
2. `auto-trainer-device/src/autotrainer/device/emulation_interface.py` — the
   same widening, so the emulator keeps matching the board.
3. `tools/acquisition/model/firmware_compatibility.py` and
   `config/pellet-firmware-compatibility.yaml` — add the new firmware version
   and list the capability.
4. `tools/acquisition/model/trial_action.py` — add line selection to
   `LaserPulseProfile` so a profile can name STIM2 or STIM3, defaulting to
   STIM3 for every existing profile.
5. `tools/acquisition/view/laser_control_content.py` — expose the line on the
   bench stim test.
6. **Make the silent failure loud.** `pulse_digital_output` returns success for
   a frame the board may drop. Once `GPIO_PULSE_STATUS` exists, wait for it and
   fail when it does not arrive, so a board without the command reports an
   error instead of a successful-looking no-op.

## How to verify

Re-run the capture that produced the evidence above. It needs no laser and no
scope, and it is decisive.

```bash
candump -tz can0 > /tmp/can_capture.log &
# send a 500 ms pulse on STIM3 (gpio_idx 7) and on STIM2 (gpio_idx 6)
```

Then check that `GPIO_READ` (`0x140`) reports the corresponding bit set for
about 500 ms after each `GPIO_PULSE` (`0x420`) frame, and returns low on its
own. Bit 7 is STIM3, bit 6 is STIM2. Today the bit never rises.

Follow that with a scope on the BNC for the short end of the range: a 100 us
and a 1 ms pulse, checking width and jitter against whichever timing option was
chosen. The CAN capture cannot resolve those.

The physical BNC-to-STIM mapping is still unconfirmed. The board carries six
BNCs (J11, J15-J18, J21) and the schematic has not been traced, so confirm
which connector carries STIM2 and STIM3 before wiring anything to them.
