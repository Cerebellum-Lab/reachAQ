#!/usr/bin/env bash
set -euo pipefail

interface="${REACHAQ_CAN_INTERFACE:-can0}"
bitrate="${REACHAQ_CAN_BITRATE:-${AUTOTRAINER_CAN_BITRATE:-1000000}}"
restart_ms="${REACHAQ_CAN_RESTART_MS:-100}"
txqueuelen="${REACHAQ_CAN_TXQUEUELEN:-1000}"
wait_seconds="${REACHAQ_CAN_WAIT_SECONDS:-30}"
driver="${REACHAQ_CAN_DRIVER:-peak_pciefd}"
can_fd="${REACHAQ_CAN_FD:-${AUTOTRAINER_CAN_FD:-false}}"
dbitrate="${REACHAQ_CAN_DBITRATE:-$bitrate}"

if [[ "$(id -u)" != "0" ]]; then
  echo "reachaq-bring-up-can must run as root because ip link CAN setup requires CAP_NET_ADMIN." >&2
  exit 1
fi

if [[ -n "$driver" ]]; then
  modprobe "$driver" || true
fi

wait_for_interface() {
  local dev="$1"
  local deadline=$((SECONDS + wait_seconds))
  until ip link show "$dev" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for CAN interface $dev" >&2
      return 1
    fi
    sleep 1
  done
}

wait_for_interface "$interface"

ip link set "$interface" down >/dev/null 2>&1 || true
if [[ "$can_fd" == "true" || "$can_fd" == "1" || "$can_fd" == "yes" ]]; then
  ip link set "$interface" type can bitrate "$bitrate" dbitrate "$dbitrate" fd on restart-ms "$restart_ms"
else
  ip link set "$interface" type can bitrate "$bitrate" restart-ms "$restart_ms"
fi
ip link set "$interface" txqueuelen "$txqueuelen"
ip link set "$interface" up
ip -details -brief link show "$interface"
