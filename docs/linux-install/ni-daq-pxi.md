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

## PXIe-1073 / NI 6713 over MXI

This path is PCI/PXI, not USB. Cold-start it in this order:

1. Power down the computer and PXIe-1073.
2. Seat the NI 6713 and MXI cards/cable.
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
4. `nilsdev` lists a DAQmx alias such as `PXI1Slot4`.

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
Laser `diodeInput` or `commandCopyInput` feedback therefore requires a separate
supported analog-input device or a disabled/adjusted feedback path.

Validate configured laser tasks only after confirming the real wiring:

```bash
cd "$HOME/Documents/reachAQ"
conda run -n reachaq python tools/hardware/validate_laser_hardware.py \
  --config "$HOME/Autotrainer/system_configuration.yaml" \
  --action connect
```

## References

- [NI Ubuntu installation](https://www.ni.com/docs/en-US/bundle/ni-platform-on-linux-desktop/page/installing-ni-products-ubuntu.html)
- [NI supported Linux drivers](https://www.ni.com/docs/en-US/bundle/ni-platform-on-linux-desktop/page/supported-drivers-for-linux-distributions.html)
- [NI-DAQmx Python API](https://nidaqmx-python.readthedocs.io/en/latest/)
- [NI PXI Platform Services Linux readme](https://www.ni.com/pdf/manuals/378682a.html)
