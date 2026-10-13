#!/usr/bin/env bash
# Remove the Blue Onyx Prism systemd service installed by scripts/install_systemd.sh.
# Usage: sudo scripts/uninstall_systemd.sh [--purge]
#
# Stops and disables the unit and removes its unit file. The program dir (e.g.
# /opt/blue-onyx-prism) is left alone: delete it yourself. The data root
# /var/lib/blue-onyx-prism (config, models, cache, downloads, benchmark.json) and the service
# user `blueonyx` are kept unless --purge is given.
set -euo pipefail

DATA_DIR=/var/lib/blue-onyx-prism
UNIT=/etc/systemd/system/blue-onyx-prism.service
PURGE=0
case "${1:-}" in
  "") ;;
  --purge) PURGE=1 ;;
  *) echo "Usage: sudo $0 [--purge]" >&2; exit 2 ;;
esac

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root (sudo)." >&2
  exit 1
fi

if [ -f "$UNIT" ]; then
  systemctl disable --now blue-onyx-prism
  rm -f "$UNIT"
  systemctl daemon-reload
  systemctl reset-failed blue-onyx-prism 2>/dev/null || true
  echo "Removed the blue-onyx-prism service."
else
  echo "No unit file at $UNIT; nothing to stop."
fi

if [ "$PURGE" -eq 1 ]; then
  rm -rf "$DATA_DIR"
  if id -u blueonyx >/dev/null 2>&1; then
    userdel blueonyx
  fi
  echo "Deleted $DATA_DIR and the user blueonyx."
else
  echo "Kept $DATA_DIR (config, models, cache) and the user blueonyx; run with --purge to delete them."
fi
