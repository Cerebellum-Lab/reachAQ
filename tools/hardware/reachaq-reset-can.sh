#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" != "0" ]]; then
  echo "reachaq-reset-can must run as root." >&2
  exit 1
fi

configured_channel="can0"
if [[ -r /etc/default/reachaq-can ]]; then
  # This file is installed root-owned with mode 0644 alongside the service.
  # shellcheck disable=SC1091
  source /etc/default/reachaq-can
  configured_channel="${REACHAQ_CAN_INTERFACE:-can0}"
fi
channel="${1:-$configured_channel}"
if [[ ! "$channel" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid CAN channel: $channel" >&2
  exit 2
fi
if [[ "$channel" != "$configured_channel" ]]; then
  echo "CAN reset refused: requested $channel but reachaq-can.service owns $configured_channel." >&2
  exit 2
fi

# The ownership lock is also held by reachAQ while its device socket is open.
# Refuse to reset a channel that another application acquired after the failed
# owner closed; resetting it would disrupt a healthy process.
lock_root="/run/lock/reachaq"
ownership_lock="${lock_root}/reachaq-can-${channel}.lock"
reset_lock="${lock_root}/reachaq-can-reset-${channel}.lock"
reset_stamp="/run/reachaq-can-reset-${channel}.stamp"
debounce_seconds=15

if [[ ! -d "$lock_root" ]]; then
  echo "CAN reset refused: lock directory is missing: $lock_root" >&2
  exit 5
fi
touch "$ownership_lock"
chown root:reachaq "$ownership_lock"
chmod 0660 "$ownership_lock"
exec 9<>"$ownership_lock"
if ! /usr/bin/flock --exclusive --nonblock 9; then
  echo "CAN reset refused: channel $channel is owned by another process." >&2
  exit 3
fi

exec 8>"$reset_lock"
if ! /usr/bin/flock --exclusive --timeout 10 8; then
  echo "CAN reset suppressed: another reset is already in progress for $channel." >&2
  exit 4
fi

now="$(date +%s)"
previous=0
if [[ -r "$reset_stamp" ]]; then
  read -r previous < "$reset_stamp" || previous=0
fi
if (( now - previous < debounce_seconds )); then
  echo "CAN reset suppressed: $channel was reset less than ${debounce_seconds}s ago." >&2
  exit 0
fi

# Stamp before restart so a failing service cannot create a restart storm.
printf '%s\n' "$now" > "$reset_stamp"
exec /usr/bin/systemctl restart reachaq-can.service
