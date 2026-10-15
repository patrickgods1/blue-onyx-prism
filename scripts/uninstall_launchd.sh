#!/usr/bin/env bash
# Remove the Blue Onyx Prism launchd daemon installed by scripts/install_launchd.sh or the .pkg.
# Usage: sudo scripts/uninstall_launchd.sh [--purge] [--keep-program]
#
# Unloads the daemon and removes its plist, newsyslog entry, installer receipt, the program dir
# (the one the daemon ran from, e.g. /usr/local/blue-onyx-prism, including runtimes bundled there)
# the menu bar app ("Blue Onyx Prism" and its LaunchAgent) and the "Uninstall Blue Onyx Prism" app. The data root /usr/local/var/blue-onyx-prism (config,
# models, downloads), the logs in /var/log/blue-onyx-prism and the role account _blueonyx are kept
# for a reinstall unless --purge is given. --keep-program leaves the program dir in place. Never
# prompts, so it runs unattended (CI, scripts, the uninstaller app); it needs no other file.
# Exit status 4: everything was removed except the account, which macOS refused to delete
# (eDSPermissionError: the calling app, e.g. the uninstaller app, may not change user accounts;
# Terminal may). The uninstaller app then offers to finish in Terminal.
set -euo pipefail

DATA_DIR=/usr/local/var/blue-onyx-prism
LOG_DIR=/var/log/blue-onyx-prism
SVC=_blueonyx
LABEL=com.blueonyx.prism
PLIST_DST="/Library/LaunchDaemons/$LABEL.plist"
NEWSYSLOG_CONF=/etc/newsyslog.d/blue-onyx-prism.conf
UNINSTALLER_APP="/Applications/Uninstall Blue Onyx Prism.app"
TRAY_LABEL=com.blueonyx.prism.tray
TRAY_PLIST="/Library/LaunchAgents/$TRAY_LABEL.plist"
TRAY_APP="/Applications/Blue Onyx Prism.app"
PURGE=0
KEEP_PROGRAM=0
for arg in "$@"; do
  case "$arg" in
    --purge) PURGE=1 ;;
    --keep-program) KEEP_PROGRAM=1 ;;
    *) echo "Usage: sudo $0 [--purge] [--keep-program]" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root (sudo)." >&2
  exit 1
fi

# The program dir: where the installed daemon runs the binary from, else the dir holding this script.
program_dir() {
  local bin=""
  if [ -f "$PLIST_DST" ]; then
    bin="$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:0' "$PLIST_DST" 2>/dev/null || true)"
  fi
  if [ -n "$bin" ]; then
    dirname "$bin"
  else
    (cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  fi
}
PROGRAM_DIR="$(program_dir)"

if launchctl print "system/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "system/$LABEL" || true
  # bootout returns before the daemon has exited (the plist allows ExitTimeOut 30 s); launchd
  # keeps the job listed until then.
  for _ in $(seq 1 40); do
    launchctl print "system/$LABEL" >/dev/null 2>&1 || break
    sleep 1
  done
  if launchctl print "system/$LABEL" >/dev/null 2>&1; then
    echo "$LABEL is still loaded 40 s after bootout; check 'launchctl print system/$LABEL'." >&2
    exit 1
  fi
fi
rm -f "$PLIST_DST" "$NEWSYSLOG_CONF"
echo "Removed the $LABEL daemon."

# Installed from the .pkg: drop the installer receipt so a later install is not seen as an upgrade.
if pkgutil --pkg-info "$LABEL" >/dev/null 2>&1; then
  pkgutil --forget "$LABEL" >/dev/null
  echo "Forgot the installer receipt $LABEL."
fi

# The menu bar app: stop it for the console user, then remove its LaunchAgent and bundle.
console_uid="$(stat -f %u /dev/console 2>/dev/null || true)"
if [ -n "$console_uid" ] && [ "$console_uid" != 0 ]; then
  launchctl bootout "gui/$console_uid/$TRAY_LABEL" >/dev/null 2>&1 || true
fi
pkill -x blue-onyx-prism-tray >/dev/null 2>&1 || true
# Its single-instance lock in each user's temp folder (/private/var/folders/<x>/<y>/T/).
find /private/var/folders -maxdepth 4 -path '*/T/blue-onyx-prism-tray.lock' -delete 2>/dev/null || true
if [ -e "$TRAY_PLIST" ] || [ -d "$TRAY_APP" ]; then
  rm -rf "$TRAY_PLIST" "$TRAY_APP"
  echo "Removed the menu bar app ($TRAY_APP) and its LaunchAgent."
fi

if [ -d "$UNINSTALLER_APP" ]; then
  rm -rf "$UNINSTALLER_APP"
  echo "Removed $UNINSTALLER_APP."
fi

if [ "$KEEP_PROGRAM" -eq 1 ]; then
  echo "Kept the program files in $PROGRAM_DIR (--keep-program)."
else
  # Only a dir that is an install of this program (the binary and the scripts every pkg and
  # archive install has), never a top-level or home dir.
  parent="$(dirname "$PROGRAM_DIR")"
  if [ "$parent" != / ] && [ "$parent" != /Users ] \
    && [ -f "$PROGRAM_DIR/blue-onyx-prism" ] && [ -f "$PROGRAM_DIR/scripts/launchd_common.sh" ]; then
    rm -rf "$PROGRAM_DIR"
    echo "Deleted the program files in $PROGRAM_DIR."
  elif [ -e "$PROGRAM_DIR" ]; then
    echo "Left $PROGRAM_DIR in place: it does not look like a Blue Onyx Prism program dir; delete it yourself if it is one." >&2
  fi
fi

if [ "$PURGE" -eq 1 ]; then
  rm -rf "$DATA_DIR" "$LOG_DIR"
  # The parent the install created on a Mac without Homebrew; never one that holds anything.
  rmdir "$(dirname "$DATA_DIR")" 2>/dev/null || true
  echo "Deleted $DATA_DIR and $LOG_DIR."
  # Best effort: macOS lets only some apps change user accounts, whatever the user id.
  account_left=0
  if dscl . -read "/Users/$SVC" >/dev/null 2>&1 && ! dscl . -delete "/Users/$SVC"; then
    account_left=1
  fi
  if dscl . -read "/Groups/$SVC" >/dev/null 2>&1 && ! dscl . -delete "/Groups/$SVC"; then
    account_left=1
  fi
  if [ "$account_left" -eq 1 ]; then
    echo "macOS did not allow deleting the hidden account $SVC (harmless; a reinstall reuses it). To remove it, run in Terminal:" >&2
    echo "  sudo dscl . -delete /Users/$SVC; sudo dscl . -delete /Groups/$SVC" >&2
    exit 4
  fi
  echo "Deleted the account $SVC."
else
  echo "Kept $DATA_DIR (config, models, downloads), $LOG_DIR and the account $SVC for a reinstall; run with --purge to delete them."
fi
