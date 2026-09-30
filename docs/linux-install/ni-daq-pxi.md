# NI-DAQmx and PXI/MXI setup

Use this guide only when the reachAQ host controls National Instruments DAQ or
PXI/PXIe hardware. The portable installer deliberately excludes NI repositories,
kernel modules, and device packages because package selection depends on the OS,
kernel, chassis, and cards.

## Package categories

| Hardware path | Required package categories |
|---|---|
| Direct USB/PCI DAQ | NI Linux Device Drivers and NI-DAQmx |
| PXI/PXIe over MXI | NI-DAQmx, PXI Platform Services, QPXI, hardware configuration utility |
| VISA instruments | Optional NI-VISA packages in addition to the applicable row above |

Raw `lspci`/`lsusb` visibility is not enough. A device must appear through
NI-DAQmx before reachAQ can use its named channels.

## 1. Install the vendor repository and packages

For the current workstation's NI 2026 Q2 Ubuntu 22.04 repository bundle:

```bash
cd "$HOME/Downloads/NILinux2026Q2DeviceDrivers"
sudo apt install ./ni-ubuntu2204-drivers-2026Q2.deb
sudo apt update
apt-cache search ni-daqmx
sudo apt install -y \
  ni-daqmx \
  ni-pxiplatformservices \
  ni-qpxi \
  ni-hwcfg-utility
sudo dkms autoinstall
sudo reboot
```

Optional VISA support, only for workflows that actually use VISA:

```bash
sudo apt install -y ni-visa ni-visa-passport-pxi
```

Select a repository release supported by the target Ubuntu/kernel combination;
do not blindly reuse the workstation-specific bundle on another release.

## 2. Verify drivers, services, and DAQ discovery

Check each layer in order:

```bash
systemctl --no-pager --plain list-units 'ni*' 'nidaq*' 'nipal*' 'nim*'
dkms status | grep -Ei 'ni|nidaq|nipal|nistc|nic'
lsmod | grep -Ei '^(ni|nidaq|nipal|nimbus|nistc|nic|nix|nim)'

lspci -nnk | grep -A4 -Ei '1093|national|daq|pxi|plx|10b5'
lsusb | grep -Ei 'national|instruments' || true
lsni -v
nipxiconfig --list-system --verbose
nilsdev
```

Then verify through the same Python environment used by reachAQ:

```bash
conda run -n reachaq python - <<'PY'
import nidaqmx

system = nidaqmx.system.System.local()
devices = tuple(system.devices)
print("device_count", len(devices))
for device in devices:
    print(device.name, device.product_type, device.product_num, device.serial_num)
PY
```

## PXIe-1073 / PXI cards over MXI

This path is PCI/PXI, not USB. Cold-start it in this order:

1. Power down the computer and PXIe-1073.
2. Seat the configured NI cards and MXI cards/cable.
3. Power on the PXIe chassis first and wait for its link LEDs.
4. Boot the computer with the chassis already powered.
5. Verify PCIe enumeration before testing NI-DAQmx.

```bash
lspci -tv
lspci -nnk | grep -A4 -Ei '1093|national|6713|10b5|plx'
journalctl -k -b --no-pager \
  | grep -Ei 'pxi|mxi|8361|1073|6713|1093|plx|bus number' \
  | tail -200
lsni -v
nipxiconfig --list-system --verbose
nilsdev
```

Expected progression:

1. `lspci` shows the MXI/PLX bridge chain and NI vendor ID `1093` endpoint.
2. `lsni -v` shows the MXI link and chassis.
3. `nipxiconfig` reports PXI/PXIe system information.
4. `nilsdev` lists the configured DAQmx aliases, such as `PXI1Slot4` and
   `PXI1Slot5` on the documented rig.

If the kernel reports `No bus number available for hot-added bridge`, cold boot
again and inspect BIOS options for PCIe pre-boot enumeration, hotplug, Above 4G
Decoding, and ASPM. A different host PCIe slot can also change downstream bus
allocation.

If PCI enumeration succeeds but NI discovery does not:

```bash
sudo dkms autoinstall
sudo systemctl restart \
  nipal nidevldu nidrum nimxssvr ni-pxipf-nipxirm-bind nipxicmsd
sudo reboot
```

## Configure reachAQ channel roles

Open **Edit → Edit DAQ Ports** while the application is idle. Discovery runs in
a background worker and the dialog lists channels reported for the selected
device. It rejects duplicate assignments and disables role types unsupported by
the device.

The digital roles - tone1, tone2, tone3R, tone3L, camFrames and barcode - must
be **port0 lines on the input card**. The stream samples them all in one
clocked digital-input task, and an M Series board such as the 6221 clocks
port0 only: port1 and port2 are its PFI pins, and the 6713's lines cannot be
clocked at all. The dialog offers only port0 lines of a board that clocks
digital input; a stored value that is not one stays on its field, the status
line says why, and OK stays disabled until it is changed. Loading a file
checks names only, since no board is asked then: a digital role that is not
one port0 line (a port1 or port2 PFI pin, a whole port, an analog input)
still loads, but its NI-DAQ plan is refused, naming the field; see below. A
port0 line on a board that cannot clock digital input, such as the 6713,
passes the load. Edit DAQ Ports marks it, and the stream refuses to start on
it, naming the channel and the board: in Idle, when the stream starts by
itself or after a Hardware refresh, and at Run. NI-DAQ then reads failed, the
same in Idle as at Run, every time the start is refused, and no task is
created; DAQmx used to answer only with error -200452. The driver's answer
is the test, not the board model (the 6713 reports no digital-input rate,
the 6221 1 MHz).
christielab10 uses `PXI1Slot5/port0/line0`-`line3` for tone1, tone2,
camFrames and barcode. Clearing a role (choosing the empty entry) stops its
channel being acquired and recorded from the next save on; the log notes the
dropped channel. The same holds for a laser tab cleared whole while the laser
backend is enabled: its diode, command copy and trigger readback channels
are dropped from the plan and logged, not kept as hidden custom inputs.

A configuration file with a line the stream cannot acquire - a duplicate pin,
a PFI pin on a digital role, a PFI trigger readback - still loads. Everything
else is applied, the NI-DAQ stream is held with no channels, and Hardware
Status shows NI-DAQ blocked with the reason, through Run and Stop; Record is
refused on it. The status bar says *NI-DAQ inputs are not acquired; fix it in
Edit → Edit DAQ Ports. Refused: ...*, the remedy first because the status bar
cuts a long message. Open the dialog: the bad value stays on its field with
the reason, and OK stays disabled until it is fixed. Saving a valid assignment
starts the stream. Until then the file keeps the values it had, including when
reachAQ saves on exit. Headless reachAQ has no dialog to fix it in, so it exits
with code 1 instead, naming the line.

The current NI PXI-6713 appears as:

```text
Device: PXI1Slot4
AO: PXI1Slot4/ao0 through PXI1Slot4/ao7
AI: none
DIO: PXI1Slot4/port0/line0 through PXI1Slot4/port0/line7
Counters: PXI1Slot4/ctr0, PXI1Slot4/ctr1, PXI1Slot4/freqout
```

The 6713 supplies analog output, digital I/O, and counters, but no analog input.
The documented rig also has a PXI-6221 at `PXI1Slot5`; it supplies the sampled
analog feedback inputs, hardware-clocked digital inputs, and counters used by
the acquisition timeline. Laser `diodeInput` and `commandCopyInput` therefore
belong on the 6221 (or another discovered analog-input device), and so does a
laser's board-trigger readback, `triggerMonitorInput`: the board STIM line
wired into an analog input (`aiN`) or a port0 line (`port0/lineN`) - the 6221
clocks digital input on port0 only, and port1/port2 are its PFI pins - never
a PFI terminal, set as **trigger readback input** on the laser's tab in
**Edit DAQ Ports** and recorded as `laserN_trigger`. Laser analog commands can
remain on the 6713. Digital and analog inputs may share the 6221 task; a
future hardware-timed 6713 output task must join the validated multi-device
timing topology described below.

The stream and the laser controller read every analog input with the
stream's `analogTerminalConfig` (`rse` by default). Each input also needs a DC
reference to AI GND, which the terminal setting cannot give it. On a
BNC-2090A that means the channel's AI x / AI x+8 switch on SE, with the
RSE/NRSE switch on RSE, or a differential input with a bias resistor from AI-
to AI GND. An input left floating reads a level that depends on the channels
scanned around it: its offset, and a laser calibration's intercept, cannot be
trusted, though a laser calibration's gain still can. christielab10's laser 1
diode input, `PXI1Slot5/ai8`, is a known case: it floats, and at idle reads
-0.16 to -0.2 V, depending on the channels scanned with it, in the
calibration ramp and in recordings alike.

The laser's clocked digital outputs - the PMT shutter (`pmtShutterOutput`)
and each laser's `triggerOutput` and `timingTriggerOutput` - run as clocked
digital output tasks beside a pulse's analog waveform. Each must be a line on
a board that can run clocked digital output: on christielab10 a port0 line on
the 6221, never one of the 6713's lines. Every such line runs on the analog
output's own sample clock, with no start trigger: that clock only runs once
the output has triggered, and the lines start before it. None can wait for a
start trigger instead, because an M Series board's clocked digital output
takes none: on christielab10 both boards report an empty `do_trig_usage`,
and a DO task with a start trigger fails DAQmx's verify with -200452, while
one clocked from a PXI_Trig line with no trigger verifies (measured
2026-09-25, verify only). A line on the output's own board names that
board's clock. A line on another board cannot name it across the boards, so
the output's clock is driven onto `pulseClockLine` (default `PXI_Trig3`) for
that pulse and read on the line's board, whether or not the pulse is
synchronized to the input stream; `backplaneClockLine` carries the stream's
clock to the 6713 and the calibration ramp's clock, nothing else. A pulse
that would put a second signal on `pulseClockLine` is refused before
anything fires, naming the line. christielab10 configures none of these
lines, and this path has not yet been run on its hardware.

`backplaneClockLine` (default `PXI_Trig1`) and `pulseClockLine` (default
`PXI_Trig3`) are each a bare PXI_Trig line, such as `PXI_Trig3`, with no
board: the controller names it on whichever board drives or reads it. A
board name, an empty value, anything but a PXI_Trig line, and a line the
backplane does not have (PXI has `PXI_Trig0` to `PXI_Trig7`) are refused, and
the case is taken as DAQmx spells it. The two must be free lines: different
from each other, from every laser's `triggerSource` line (which is also its
trigger route's destination), from every `triggerListenerInputs` line, and
from a PXI_Trig `triggerRouteSource`, which would carry the clock into the
trigger so that the laser arms on its first edge. A trigger on a clock's line
is a second driver on it, which DAQmx does not notice across these boards,
and the clock or the trigger is corrupted. A trigger input drives nothing,
so it cannot corrupt a clock; one on a clock's line is refused as a
defensive check all the same, since it would read the clock rather than a
trigger. Lines compare by name alone,
whatever the board and case (`/PXI1Slot4/PXI_Trig1` and `pxi_trig1` are one
line); a PFI never clashes. Loading a configuration that clashes is refused,
naming the field and the line, and Edit DAQ Ports refuses a trigger input on
either before it closes. christielab10 keeps the shared clock on PXI_Trig1,
its triggers and trigger inputs on PXI_Trig0 and PXI_Trig2, and leaves
PXI_Trig3 to `pulseClockLine`.

`pulseClockLine` came without a configuration version change, as
`backplaneClockLine` did: a file without it loads with the default, and a
file this build saves carries it. An older build refuses such a file, since
the configuration's version is the same and the key is unknown to it; to go
back to one, delete the `pulseClockLine:` line from the laser block of
`system_configuration.yaml` first.

Validate configured laser tasks only after confirming the real wiring, and
with reachAQ closed: its NI-DAQ input stream runs whenever it is open, and
holds the input lines this opens tasks on.

```bash
cd "$HOME/Documents/reachAQ"
conda run -n reachaq python tools/hardware/validate_laser_hardware.py \
  --config "$HOME/Autotrainer/system_configuration.yaml" \
  --action connect
```

Main Analysis camera/barcode/tone selections and the diode, command-copy and
board-trigger selections owned by each Laser Control tab are saved immediately
under `nidaqStream.displayChannels`, and can be changed while the stream runs.
Plot visibility does not change acquisition: every mapped camera-frame,
barcode, tone, laser-feedback, laser trigger-readback, and custom input is
included in the recording task and saved to `streams/nidaq.h5`.

The input stream has no Start button. With NI-DAQ enabled and at least one
input mapped, reachAQ starts it by itself while idle - when the configuration
loads, after a hardware refresh, after **Edit DAQ Ports** saves, and when
acquisition stops. Starting System Mode restarts it, so recorded timing starts
from a fresh task-start anchor rather than carrying the Idle stream's clock
drift into a session. A start that fails is shown in Hardware Status and the
log and is not retried by itself; a hardware refresh tries again. Stream task
creation happens in an isolated child process because a broken or incompatible
NI-DAQmx native runtime can terminate the Python interpreter. If the worker
exits with `SIGSEGV` or does not become ready within 10 seconds, reachAQ
remains open and displays the failure. Treat that message as a driver/device
problem: re-run the discovery checks above and verify that each selected
channel supports the requested input task.

For camera/barcode TTL streams, use a hardware-clocked input rate of at least
5 kHz; reachAQ defaults to 10 kHz and derives the runtime read size from the
refresh rate of the screen containing the application. Digital-only tasks use
`ctr0` on the input device to generate the sample clock. Confirm that the
counter is not reserved by another task. On Linux/PXI-6221 systems, reachAQ uses
interrupt transfers and requests data whenever onboard memory is non-empty;
this avoids the roughly 100 ms burst delivery seen with the default half-full
FIFO condition while preserving correct digital values. The UI redraws a peak-preserving,
display-bounded view at that screen's refresh rate; this visualization is never
persisted and does not reduce the hardware capture rate. Rolling-buffer and
per-pixel peak-envelope work runs in a dedicated plot-data process that
publishes double-buffered shared memory, while the GUI process performs only the
final bounded Qt curve draw. The digital graph uses fixed limits of `[-10, 0]`
seconds and `[-0.2, 1.2]`, supports horizontal-only zoom, and provides **Live**
to move the right edge to zero while preserving the current zoom width.

## Check the wiring

A channel role in the configuration is a claim about which cable is on which
pin. reachAQ records whatever arrives on the pin, so a wrong cable produces
plausible data rather than an error. Check the claims after any rewiring.

### DAQ Monitor

**Tools → DAQ Monitor**, while acquisition is idle, opens a separate window
over the same hardware. It has one tab per card. Each tab streams every analog
input and every clockable digital line (at 2 kHz by default; **Rate** changes
it) and polls the remaining PFI and static lines for their level. Every line
is labelled with both its terminal name and the connector printed on the
breakout block. **Start monitoring** starts the stream; **Rescan hardware**
repeats discovery. The application's own input stream is paused while this
window is open, because both would open tasks on the same lines, and starts
again by itself when the window closes.

The bar along the bottom drives one thing at a time, so you can see which line
moves:

- **Tone** / **Play 0.5 s** plays a tone. Only 5 kHz and 6 kHz raise a
  confirmation line (tone1 and tone2); any other frequency sounds without
  moving a line.
- **Pulse board STIM2** and **Pulse board STIM3** pulse the pellet board's
  stimulus outputs. STIM0 and STIM1 are the tone confirmation lines.
- **Laser**, a level and **Hold 0.5 s** holds a laser command at that voltage.
  The shutter stays closed unless **open shutter** is ticked.

### Wiring test

The monitor's **Wiring test** tab runs the check automatically. **Run wiring
test** stops the monitor stream, then holds each board output high and each
laser command at a DC level in turn, and records which line followed. The
shutters stay closed unless **open shutters during the test** is ticked; with
them closed, the photodiode inputs cannot be confirmed. Each configured channel
gets one status:

| Status | Meaning |
|---|---|
| `CONFIRMED` | Responded to its own driver and nothing else |
| `SILENT` | Did not respond when driven; check the cable at the named connector |
| `UNEXPECTED` | Responded to another channel's driver: the cable is on the wrong connector, or two are swapped |
| `UNTESTED` | Nothing in the run could drive it (camera frames and barcode, for example). A laser's trigger readback (`laserN_trigger`, or `laserN_trigger_readback` as christielab10 acquires it) follows the board STIM, not the laser command, so the laser drive does not check it. An analog one reads *trigger readback: not checked against the command (it follows the board STIM)*. A digital one is checked by the board output sweep, confirmed if it went high and silent if not; it reads *not checked* only when CAN was down |
| `OPAQUE` | Cannot be checked from the DAQ, such as a shutter output with no readback |

The test reports what it saw, not why. A mislabelled cable, a split, and
coupling between inputs look the same from the DAQ. **Save report...** writes
the result as text. The result is kept in
`~/Autotrainer/nidaq_wiring_verification.json`, and every hardware refresh
compares the configuration with it. Hardware Status then shows
`NI-DAQ wiring N/M confirmed`. When any channel is failed or unchecked, a
warning names each one, for example:

```text
NI-DAQ wiring is unverified (did not respond when last checked: laser2_diode; never checked: cam_frames, barcode); run tools/hardware/verify_nidaq_wiring.py
```

The same test runs from a terminal, with reachAQ closed:

```bash
conda run --no-capture-output -n reachaq python tools/hardware/verify_nidaq_wiring.py
```

| Option | Effect |
|---|---|
| `--dry-run` | List the channels the configuration claims and stop, touching nothing |
| `--open-shutters` | Open each shutter while its laser is driven, so the photodiodes are checked. This emits light |
| `--observe SECONDS` | Afterwards, watch the lines nothing here can drive (`cam_frames`, `barcode`) for that long and confirm any that change. Run the cameras or trigger a barcode during the window |
| `--command-volts V` | Laser command level (default 1.0) |
| `--report PATH` | Also write the text report to `PATH` |
| `--config PATH`, `--record PATH` | Configuration read and result file written; default to the files above |

## Device identity and timing configuration

The DAQ Ports dialog stores the discovered product and serial identity for each
selected device. The configured logical binding can therefore follow the same
physical card if NI-DAQmx later changes its runtime alias. Missing, substituted,
ambiguous, and serial-mismatched devices fail validation instead of silently
binding to another card.

Discovery also records AI/AO/DI/DO channels, counters, terminals, maximum rates,
timed analog-output support, digital-trigger support, bus type, and PXI chassis.
The timing planner validates every active channel, requested rate, manual timing
master, and explicit route before task startup.

Timing policy belongs under `nidaqPorts.timing`:

```yaml
timing: !NidaqTimingConfiguration
  syncMode: auto
  timingMaster: null
  requireHardwareSynchronization: true
  taskStrategy: per_device
  referenceClockSource: null
  startTriggerSource: null
  sampleClockSource: null
```

`auto` uses one device when possible, otherwise a discovered common PXI
backplane, otherwise requires explicit external start/sample routes. `backplane`
requires a common PXI chassis. `external` requires configured trigger and sample
clock terminals. `independent` is diagnostic-only when hardware synchronization
is required and blocks aligned recording across multiple devices.

The optional master is a device/task role, not a port flag. Automatic selection
prefers the device that owns the canonical sampled inputs. On compatible PXI
hardware, reachAQ uses `PXI_CLK10`, a shared start trigger, and a routed master
sample clock; slave tasks arm before the master. Finite hardware-timed laser AO
uses the same verified reference/clock graph plus a trial-local future trigger.
Do not copy the current rig's slot names or routes to another rig without
discovery and validation.

First-release graphs contain at most two active devices across PXI, PCIe, or a
properly wired mixed topology. Use `per_device` normally. `auto_multidevice`
tries the exact disposable channel-expansion task and uses it only after DAQmx
Verify/Commit; unsupported expansion falls back to the independently verified
per-device graph. `forced_multidevice` turns that rejection into a startup
failure. Three-card configurations are rejected rather than labeled supported.

Requested and resolved timing, device identities, task ordering, routes, and
synchronization quality are saved in each trial's
`streams/alignment.json`. See
[Session recording, synchronization, and hardware isolation](../acquisition/session-recording-and-synchronization.md)
for the complete persistence schema and physical acceptance procedure.

## References

- [NI Ubuntu installation](https://www.ni.com/docs/en-US/bundle/ni-platform-on-linux-desktop/page/installing-ni-products-ubuntu.html)
- [NI supported Linux drivers](https://www.ni.com/docs/en-US/bundle/ni-platform-on-linux-desktop/page/supported-drivers-for-linux-distributions.html)
- [NI-DAQmx Python API](https://nidaqmx-python.readthedocs.io/en/latest/)
- [NI PXI Platform Services Linux readme](https://www.ni.com/pdf/manuals/378682a.html)
