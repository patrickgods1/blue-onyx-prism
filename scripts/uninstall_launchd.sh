#!/usr/bin/env bash
# Remove the Blue Onyx Prism launchd daemon installed by scripts/install_launchd.sh.
# Usage: sudo scripts/uninstall_launchd.sh [--purge]
#
# Unloads the daemon and removes its plist and newsyslog entry. The program dir (e.g.
# /usr/local/blue-onyx-prism) is left alone: delete it yourself. The data root
# /usr/local/var/blue-onyx-prism, the logs in /var/log/blue-onyx-prism and the role account
# _blueonyx are kept unless --purge is given. Never prompts, so it runs unattended (CI, scripts).
# Also works from the .pkg install (/usr/local/blue-onyx-prism/scripts); it needs no other file.
set -euo pipefail

DATA_DIR=/usr/local/var/blue-onyx-prism
LOG_DIR=/var/log/blue-onyx-prism
SVC=_blueonyx
PLIST_DST="/Library/LaunchDaemons/com.blueonyx.prism.plist"
NEWSYSLOG_CONF=/etc/newsyslog.d/blue-onyx-prism.conf
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

if launchctl print system/com.blueonyx.prism >/dev/null 2>&1; then
  launchctl bootout system/com.blueonyx.prism
fi
rm -f "$PLIST_DST" "$NEWSYSLOG_CONF"
echo "Removed the com.blueonyx.prism daemon."

# Installed from the .pkg: drop the installer receipt so a later install is not seen as an upgrade.
if pkgutil --pkg-info com.blueonyx.prism >/dev/null 2>&1; then
  pkgutil --forget com.blueonyx.prism >/dev/null
  echo "Forgot the installer receipt com.blueonyx.prism."
fi

if [ "$PURGE" -eq 1 ]; then
  rm -rf "$DATA_DIR" "$LOG_DIR"
  if dscl . -read "/Users/$SVC" >/dev/null 2>&1; then
    dscl . -delete "/Users/$SVC"
  fi
  if dscl . -read "/Groups/$SVC" >/dev/null 2>&1; then
    dscl . -delete "/Groups/$SVC"
  fi
  echo "Deleted $DATA_DIR, $LOG_DIR and the account $SVC."
else
  echo "Kept $DATA_DIR, $LOG_DIR and the account $SVC; run with --purge to delete them."
fi
