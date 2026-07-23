#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" != "0" ]]; then
  echo "reachaq-reset-can must run as root." >&2
  exit 1
fi

exec /usr/bin/systemctl restart reachaq-can.service
