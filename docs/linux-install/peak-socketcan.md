# PEAK CAN / SocketCAN setup

Use this guide on rigs with the pellet-board CAN connection. reachAQ uses
`python-can`; Linux SocketCAN is the preferred path for PEAK PCIe/USB adapters.

The portable Linux installer installs `can-utils` and `iproute2`, but it does
not install, configure, or enable the rig-specific CAN service described here.
Complete this guide separately on every physical rig.

The reachAQ CAN reader polls at a bounded 5 ms cadence when no frame is
available. Some CAN backends return an empty read immediately instead of
honoring the requested collection interval; retrying without this idle wait can
consume a full CPU core and starve the Qt event loop. Available frames and CAN
writes are not delayed by the idle throttle. A frame arriving just after an
empty poll can wait at most 5 ms before being read.

## Configuration ownership

The Linux service and the application have separate responsibilities and
separate environment files:

| Layer | Configuration | Responsibility |
|---|---|---|
| Kernel/driver | `peak_pciefd` and SocketCAN | Expose PEAK channels as network devices such as `can0` |
| Boot service | `/etc/default/reachaq-can` | Select one application channel and configure its bitrate, CAN FD mode, queue length, and `UP` state |
| Application | `AUTOTRAINER_CAN_*` environment variables | Select the already-configured channel and open a SocketCAN socket |
| Safety reset | `/usr/local/sbin/reachaq-reset-can` | Restart only `reachaq-can.service`, bringing its configured channel down and back up |

`REACHAQ_CAN_INTERFACE` and `AUTOTRAINER_CAN_CHANNEL` must name the same
interface. Values in `/etc/default/reachaq-can` are visible to systemd only;
sourcing `reachaq_hardware.env.example` affects the application shell only.

For SocketCAN, the application does not program Linux bit timing. The service
must bring the network device up with the correct arbitration bitrate, data
bitrate, and FD mode before reachAQ opens it.

## 1. Verify the kernel driver and adapter

```bash
kernel_config="/boot/config-$(uname -r)"
grep 'CONFIG_CAN_PEAK' "$kernel_config"

sudo modprobe peak_pciefd
lsmod | grep '^peak'
modinfo peak_pciefd | sed -n '1,12p'

ip -details link show type can
lspci -nnk | grep -A3 -Ei 'peak|pcan|can' || true
lsusb | grep -Ei 'peak|pcan' || true
```

The current PEAK PCIe card uses the in-kernel `peak_pciefd` driver and exposes
`can0` and `can1`. A PEAK USB adapter normally uses `peak_usb` instead:

```bash
sudo modprobe peak_usb
```

The shell substitution in `/boot/config-$(uname -r)` is intentional. Do not
quote `uname -r` as literal text, and do not append the word `check` to the
`lsmod | grep` command. For the current PCIe FD card, the kernel configuration
should include `CONFIG_CAN_PEAK_PCIEFD=m` or `=y`, and the loaded-module line
should begin with `peak_pciefd`.

Use PEAK's proprietary driver package only when the in-kernel driver does not
support the adapter, or when the application intentionally requires PEAK's
character-device APIs such as PCAN-Basic or `libpcan`. Do not install both
driver models for the same application path without a specific reason.

## 2. Verify wiring and termination

Confirm the PEAK connector pinout from the exact adapter manual and confirm the
custom-board connector from its schematic. At minimum the bus requires
CAN-H-to-CAN-H, CAN-L-to-CAN-L, an appropriate reference/ground, and power for
the custom board and its CAN transceiver.

Measure resistance only while the computer, adapter, board, and all other CAN
nodes are unpowered:

| Resistance across CAN-H and CAN-L | Interpretation |
|---|---|
| Approximately 60 ohms | Normal two-end termination: two 120-ohm terminators in parallel |
| Approximately 120 ohms | Only one terminator is present |
| Substantially below 60 ohms | Too many terminators, another parallel load, or a wiring fault |
| Open circuit or very high resistance | No effective termination or an open cable |

Do not interpret resistance measurements made on a powered bus; active
transceivers invalidate the reading. CAN-H/CAN-L voltage is useful for finding
gross wiring faults, but idle voltage alone does not prove matching bit timing
or valid frames.

## 3. Bring up a confirmed bus manually

Confirm the bench bitrate and termination before sending traffic. The
custom-board JerryCAN firmware uses CAN FD with bit-rate switching: 1 Mbit/s
for arbitration and 5 Mbit/s for the data phase:

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can \
  bitrate 1000000 \
  dbitrate 5000000 \
  fd on \
  restart-ms 100
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
ip -details -statistics link show can0
```

The verification output must include `UP`, `mtu 72`, `<FD>`, and
`dbitrate 5000000`. An interface showing `mtu 16` is in Classical CAN mode and
cannot carry JerryCAN configuration, movement, status, audio, or bootloader
frames.

`state ERROR-ACTIVE` means the controller is in its normal CAN error state; it
does not prove that a board is connected. Board communication is confirmed
only when valid RX frames or a successful validator response are observed.
Rising receive error counters, `ERROR-PASSIVE`, or `BUS-OFF` point to physical
wiring, termination, power, or bit-timing problems.

## 4. Install the boot service and reset helper

The repository includes a reusable interface setup script and systemd unit.
Installing and enabling this unit makes Linux configure and bring up `can0`
automatically during every boot, before reachAQ is launched:

```bash
cd "$HOME/Documents/reachAQ"
sudo install -m 0755 \
  tools/hardware/reachaq-bring-up-can.sh \
  /usr/local/sbin/reachaq-bring-up-can
sudo install -m 0755 \
  tools/hardware/reachaq-bring-down-can.sh \
  /usr/local/sbin/reachaq-bring-down-can
sudo install -m 0755 \
  tools/hardware/reachaq-reset-can.sh \
  /usr/local/sbin/reachaq-reset-can
sudo install -m 0644 \
  tools/hardware/reachaq-can.default \
  /etc/default/reachaq-can
sudo install -m 0644 \
  tools/hardware/reachaq-can.service \
  /etc/systemd/system/reachaq-can.service
sudo systemctl daemon-reload
sudo systemctl enable --now reachaq-can.service
```

`enable` registers the service for future boots. `--now` also starts it
immediately, so a reboot is not required during installation. The service only
configures the one SocketCAN interface named by `REACHAQ_CAN_INTERFACE`; it
does not configure `can1` unless `can1` is explicitly selected, and it does not
send pellet or motor commands.

The application safety shutdown invokes only the root-owned reset helper. Grant
that exact permission through a dedicated operator group:

```bash
sudo groupadd -f reachaq
sudo usermod -aG reachaq "$USER"
sudo install -m 0440 \
  tools/hardware/reachaq-can-reset.sudoers \
  /etc/sudoers.d/reachaq-can-reset
sudo visudo -cf /etc/sudoers.d/reachaq-can-reset
```

Log out of the desktop session and back in after adding the group. Opening a
new terminal inside the existing session is not sufficient. Verify the active
login and the narrow permission:

```bash
getent group reachaq
id -nG | tr ' ' '\n' | grep -x reachaq
sudo -n -l /usr/local/sbin/reachaq-reset-can
```

If `getent` lists the user but `id -nG` does not, the current login still has
the old group list. The application uses non-interactive `sudo -n`; it cannot
display a password prompt and its safety reset will fail until the new group
membership is active. The sudoers rule does not grant general `systemctl`,
`ip`, or root-shell access.

Verify that every privileged artifact is root-owned and has the expected mode:

```bash
stat -c '%U:%G %a %n' \
  /usr/local/sbin/reachaq-bring-up-can \
  /usr/local/sbin/reachaq-bring-down-can \
  /usr/local/sbin/reachaq-reset-can \
  /etc/default/reachaq-can \
  /etc/systemd/system/reachaq-can.service \
  /etc/sudoers.d/reachaq-can-reset
```

Expected modes are `755` for the three helpers, `644` for the defaults and
unit, and `440` for the sudoers rule. All should be owned by `root:root`.

Edit `/etc/default/reachaq-can` when a rig uses a different interface,
bitrate, driver, or CAN-FD setting. Do not configure the same interfaces through
multiple boot mechanisms.

The tracked defaults are:

| Variable | Current value | Meaning |
|---|---:|---|
| `REACHAQ_CAN_INTERFACE` | `can0` | The only channel managed by this unit |
| `REACHAQ_CAN_BITRATE` | `1000000` | Arbitration bitrate |
| `REACHAQ_CAN_FD` | `true` | Enable CAN FD |
| `REACHAQ_CAN_DBITRATE` | `5000000` | CAN FD data-phase bitrate |
| `REACHAQ_CAN_RESTART_MS` | `100` | Automatic recovery delay after bus-off |
| `REACHAQ_CAN_TXQUEUELEN` | `1000` | Linux transmit queue length |
| `REACHAQ_CAN_WAIT_SECONDS` | `30` | Maximum boot wait for the interface to appear |
| `REACHAQ_CAN_DRIVER` | `peak_pciefd` | Kernel module requested before configuration |

Verify the service:

```bash
systemctl is-enabled reachaq-can.service
systemctl is-active reachaq-can.service
systemctl --no-pager status reachaq-can.service
ip -details link show can0
journalctl -u reachaq-can.service -b --no-pager
```

The first two commands must report `enabled` and `active`. The `can0` output
must report `UP`, `mtu 72`, `<FD>`, arbitration `bitrate 1000000`, and
`dbitrate 5000000`. `systemctl status` normally displays `active (exited)`:
this is the healthy state for a `Type=oneshot` service with
`RemainAfterExit=yes`.

With reachAQ closed and the mechanism in a safe state, commission the same
non-interactive reset path that the application uses:

```bash
sudo -n /usr/local/sbin/reachaq-reset-can
systemctl is-active reachaq-can.service
ip -details link show can0
```

The reset helper restarts exactly `reachaq-can.service`. Its `ExecStop` brings
the configured channel down; `ExecStart` reapplies its settings and brings it
up. This flushes the host SocketCAN interface and kernel transmit queue. It
does not power-cycle the custom board, cancel motion already accepted by the
board, or replace a physical emergency stop.

Confirm persistence across a real boot when initially commissioning the rig:

```bash
sudo reboot
```

After logging back in:

```bash
systemctl is-enabled reachaq-can.service
systemctl is-active reachaq-can.service
ip -details link show can0
journalctl -u reachaq-can.service -b --no-pager
```

If `can0` is down after reboot, inspect the current-boot journal before starting
reachAQ. Do not add a second NetworkManager, systemd-networkd, or shell-based
CAN startup mechanism; repair the tracked service instead.

### Updating an existing installation

After pulling repository changes to any helper, unit, or sudoers file, reinstall
the root-owned copies. Do not assume the files under `/usr/local` or `/etc`
update with the Git checkout:

```bash
cd "$HOME/Documents/reachAQ"
sudo install -m 0755 tools/hardware/reachaq-bring-up-can.sh \
  /usr/local/sbin/reachaq-bring-up-can
sudo install -m 0755 tools/hardware/reachaq-bring-down-can.sh \
  /usr/local/sbin/reachaq-bring-down-can
sudo install -m 0755 tools/hardware/reachaq-reset-can.sh \
  /usr/local/sbin/reachaq-reset-can
sudo install -m 0644 tools/hardware/reachaq-can.service \
  /etc/systemd/system/reachaq-can.service
sudo install -m 0440 tools/hardware/reachaq-can-reset.sudoers \
  /etc/sudoers.d/reachaq-can-reset
sudo visudo -cf /etc/sudoers.d/reachaq-can-reset
sudo systemctl daemon-reload
sudo systemctl restart reachaq-can.service
```

Preserve a rig-specific `/etc/default/reachaq-can`. Compare it with the tracked
template before replacing it:

```bash
diff -u /etc/default/reachaq-can \
  tools/hardware/reachaq-can.default || true
```

## 5. Configure the reachAQ application

Load the tracked environment defaults before launching:

```bash
cd "$HOME/Documents/reachAQ"
set -a
source tools/hardware/reachaq_hardware.env.example
set +a
```

These values select `socketcan`, channel `can0`, and CAN FD for the application.
They do not bring the interface up. Source them in the same shell that launches
reachAQ, or put equivalent variables in the launcher/service environment.
Confirm the effective values:

```bash
env | grep '^AUTOTRAINER_CAN_' | sort
```

The application channel must match `/etc/default/reachaq-can`:

```bash
. /etc/default/reachaq-can
test "$AUTOTRAINER_CAN_CHANNEL" = "$REACHAQ_CAN_INTERFACE"
```

Do not source `/etc/default/reachaq-can` as the application configuration; its
`REACHAQ_*` names are intentionally service-specific.

## 6. Validate receive traffic before motion

Monitor and validate without commanding motion:

```bash
timeout 5 candump -L can0
conda run -n reachaq python tools/hardware/validate_can_hardware.py \
  --transport socketcan \
  --channel can0 \
  --action discover
```

Use the validator's motion actions only after reviewing `--help`, confirming the
target device and safe mechanism state, and supplying its explicit motion
permission flag.

Capture interface counters before and after a listening interval:

```bash
ip -details -statistics link show can0
timeout 5 candump -L can0
ip -details -statistics link show can0
```

Increasing RX packets with valid JerryCAN IDs confirms physical board traffic.
An empty `candump` does not by itself prove failure if the connected board is
silent; use the validator's discovery action for an application-level check.

## Service operations

Run these operations only while reachAQ is closed:

```bash
sudo systemctl start reachaq-can.service
sudo systemctl restart reachaq-can.service
sudo systemctl stop reachaq-can.service
```

- `start` applies the configured timing and brings the selected channel up.
- `restart` brings that channel down and up, flushing the host interface.
- `stop` runs `reachaq-bring-down-can` and leaves the selected channel down.
- Starting or restarting this service does not cycle the unused `can1`.

## PEAK PCAN-View distinction

`pcanview` expects PEAK's `/dev/pcan*` character-device driver path. It does not
monitor Linux SocketCAN interfaces such as `can0`. On a SocketCAN rig use:

```bash
candump can0
cansniffer can0
ip -details link show can0
```

Install/use `pcanview-ncurses` only when the rig intentionally uses PEAK's
out-of-tree PCAN driver and `/dev/pcan*` exists.

## Troubleshooting

| Symptom | Check |
|---|---|
| No `can0`/`can1` | Adapter enumeration and `peak_pciefd`/`peak_usb` module |
| Interface `DOWN` | Boot service status or manual `ip link` setup |
| Interface `STOPPED` | Bitrate, termination, wiring, and bus-off logs |
| `active (exited)` service | Expected healthy oneshot-service state |
| Safety reset says `sudo: a password is required` | Log out/in so the `reachaq` group is active; verify `sudo -n -l` |
| Application and service use different channels | Match `AUTOTRAINER_CAN_CHANNEL` to `REACHAQ_CAN_INTERFACE` |
| `REQUEST_VERSION` times out while RX traffic continues | Host startup/ACK handling issue, not proof of a dead board; retain the complete application log |
| Validator opens bus but misses board | Correct interface, board power, bitrate, and target firmware |
| Another backend is selected | Source `tools/hardware/reachaq_hardware.env.example` |

References:

- [PEAK Linux drivers](https://www.peak-system.com/fileadmin/media/linux/index.php)
- [PEAK SocketCAN implementation](https://www.peak-system.com/fileadmin/media/linux/can-implementation.php)
- [Linux SocketCAN documentation](https://www.kernel.org/doc/Documentation/networking/can.txt)
- [PCAN-PCI Express FD manual](https://www.peak-system.com/produktcd/Pdf/English/PCAN-PCI-Express-FD_UserMan_eng.pdf)
