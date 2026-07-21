# PEAK CAN / SocketCAN setup

Use this guide on rigs with the pellet-board CAN connection. reachAQ uses
`python-can`; Linux SocketCAN is the preferred path for PEAK PCIe/USB adapters.

## 1. Discover the adapter and interfaces

```bash
sudo modprobe peak_pciefd || sudo modprobe peak_usb
ip -details link show type can
lspci -nnk | grep -A3 -Ei 'peak|pcan|can' || true
lsusb | grep -Ei 'peak|pcan' || true
```

The current PEAK PCIe card uses the in-kernel `peak_pciefd` driver and exposes
`can0` and `can1`.

## 2. Bring up a confirmed bus

Confirm the bench bitrate and termination before sending traffic. The current
pellet bus uses 1 Mbit/s:

```bash
sudo ip link set can0 down || true
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
ip -details link show can0
```

## 3. Install the tracked boot service

The repository includes the reusable interface setup script and systemd unit:

```bash
cd "$HOME/Documents/reachAQ"
sudo install -m 0755 \
  tools/hardware/reachaq-bring-up-can.sh \
  /usr/local/sbin/reachaq-bring-up-can
sudo install -m 0644 \
  tools/hardware/reachaq-can.default \
  /etc/default/reachaq-can
sudo install -m 0644 \
  tools/hardware/reachaq-can.service \
  /etc/systemd/system/reachaq-can.service
sudo systemctl daemon-reload
sudo systemctl enable --now reachaq-can.service
```

Edit `/etc/default/reachaq-can` when a rig uses a different interface list,
bitrate, driver, or CAN-FD setting. Do not configure the same interfaces through
multiple boot mechanisms.

Verify the service:

```bash
systemctl --no-pager status reachaq-can.service
ip -details -brief link show type can
journalctl -u reachaq-can.service -b --no-pager
```

## 4. Configure and validate reachAQ

Load the tracked environment defaults before launching:

```bash
cd "$HOME/Documents/reachAQ"
set -a
source tools/hardware/reachaq_hardware.env.example
set +a
```

Monitor and validate without commanding motion:

```bash
candump can0
conda run -n reachaq python tools/hardware/validate_can_hardware.py \
  --transport socketcan \
  --channel can0 \
  --action discover
```

Use the validator's motion actions only after reviewing `--help`, confirming the
target device and safe mechanism state, and supplying its explicit motion
permission flag.

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
| Validator opens bus but misses board | Correct interface, board power, bitrate, and target firmware |
| Another backend is selected | Source `tools/hardware/reachaq_hardware.env.example` |

References: [PEAK Linux drivers](https://www.peak-system.com/fileadmin/media/linux/index.php),
[SocketCAN implementation](https://www.peak-system.com/fileadmin/media/linux/can-implementation.php).
