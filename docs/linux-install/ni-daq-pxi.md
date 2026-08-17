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
belong on the 6221 (or another discovered analog-input device), while laser
analog commands can remain on the 6713. Digital and analog inputs may share the
6221 task; a future hardware-timed 6713 output task must join the validated
multi-device timing topology described below.

Validate configured laser tasks only after confirming the real wiring:

```bash
cd "$HOME/Documents/reachAQ"
conda run -n reachaq python tools/hardware/validate_laser_hardware.py \
  --config "$HOME/Autotrainer/system_configuration.yaml" \
  --action connect
```

Main Analysis camera/barcode/tone selections and the diode/command-copy
selections owned by each Laser Control tab are saved immediately under
`nidaqStream.displayChannels`. Plot visibility does not change acquisition:
every mapped camera-frame, barcode, tone, laser-feedback, and custom input is
included in the recording task and saved to `streams/nidaq.h5`. Stream task
creation happens in an isolated child
process because a broken or incompatible NI-DAQmx native runtime can terminate
the Python interpreter. If the worker exits with `SIGSEGV` or does not become
ready within 10 seconds, reachAQ remains open and displays the failure. Treat
that message as a driver/device problem: re-run the discovery checks above and
verify that each selected channel supports the requested input task.

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
