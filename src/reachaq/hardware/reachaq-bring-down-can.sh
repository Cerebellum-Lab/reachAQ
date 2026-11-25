#!/usr/bin/env bash
set -euo pipefail

interface="${REACHAQ_CAN_INTERFACE:-can0}"

if [[ "$(id -u)" != "0" ]]; then
  echo "reachaq-bring-down-can must run as root because ip link requires CAP_NET_ADMIN." >&2
  exit 1
fi

if ip link show "$interface" >/dev/null 2>&1; then
  ip link set "$interface" down
fi
