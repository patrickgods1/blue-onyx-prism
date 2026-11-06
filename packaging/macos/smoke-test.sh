#!/usr/bin/env bash
# Install the pkg, check the daemon answers the uninstaller app and the tray app are in place, uninstall keeping
# the data (program and app gone, data kept, a seeded dataset file included), reinstall over it with
# the account's id changed (the data follows and survives), then purge. For a macOS CI runner
# (passwordless sudo). Usage: packaging/macos/smoke-test.sh <pkg>
set -euo pipefail

PKG="${1:?Usage: $0 <pkg>}"
DATA=/usr/local/var/blue-onyx-prism
CONFIG="$DATA/blue_onyx_prism_config.json"
# Stands for what no download brings back (dataset frames, reviewed labels, training runs). In a
# folder of its own: nothing the service prunes walks it.
SEED="$DATA/dataset/smoke-test/frame.labels.json"
PROG=/usr/local/blue-onyx-prism
APP="/Applications/Uninstall Blue Onyx Prism.app"
fail() {
  echo "SMOKE TEST FAILED: $*" >&2
  # Why the daemon stopped: a restart count alone says nothing.
  sudo launchctl print system/com.blueonyx.prism 2>&1 \
    | grep -E '^\s*(state|runs|pid|last exit|immediate reason|job state)' >&2 || true
  id _blueonyx >&2 || true
  sudo ls -ldn "$DATA" "$CONFIG" >&2 || true
  sudo sh -c 'cd /var/log/blue-onyx-prism && ls -la && for f in $(ls -t | head -3); do echo "== $f"; tail -n 40 "$f"; done' >&2 || true
  exit 1
}

# The daemon is loaded, owns its config and comes up ready, started once.
check_daemon() {
  sudo launchctl print system/com.blueonyx.prism >/dev/null || fail "daemon not loaded"
  sudo test -f "$CONFIG" || fail "config $CONFIG missing"
  local owner
  owner="$(sudo stat -f %Su "$CONFIG")"
  [ "$owner" = "_blueonyx" ] || fail "config owned by $owner, expected _blueonyx"

  local code=000
  for _ in $(seq 1 60); do
    code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:32168/v1/status || true)"
    case "$code" in 200 | 503) break ;; esac
    sleep 2
  done
  case "$code" in 200 | 503) echo "/v1/status answered $code" ;; *) fail "/v1/status answered $code within 120 s" ;; esac

  # Ready means the downloaded runtime loads and the model serves; launchd must not have had to
  # restart the daemon (KeepAlive would hide a crash loop behind an answering port).
  for _ in $(seq 1 180); do
    code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:32168/v1/status || true)"
    [ "$code" = 200 ] && break
    sleep 2
  done
  [ "$code" = 200 ] || fail "/v1/status not ready (200) within 6 min, last answer $code"
  local runs
  runs="$(sudo launchctl print system/com.blueonyx.prism | awk '/^\truns = / { print $3 }')"
  [ "$runs" = 1 ] || fail "the daemon was started $runs times (expected 1)"
  echo "daemon ready, started once"
}

sudo installer -pkg "$PKG" -target / -verbose
check_daemon

[ -d "$APP" ] || fail "$APP missing"
codesign --verify --strict "$APP" || fail "$APP has an invalid signature"
[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP/Contents/Info.plist")" = com.blueonyx.prism.uninstaller ] \
  || fail "$APP has the wrong bundle id"

TRAY_APP="/Applications/Blue Onyx Prism.app"
TRAY_PLIST=/Library/LaunchAgents/com.blueonyx.prism.tray.plist
[ -d "$TRAY_APP" ] || fail "$TRAY_APP missing"
[ -x "$TRAY_APP/Contents/MacOS/blue-onyx-prism-tray" ] || fail "tray executable missing in $TRAY_APP"
codesign --verify --strict "$TRAY_APP" || fail "$TRAY_APP has an invalid signature"
[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$TRAY_APP/Contents/Info.plist")" = com.blueonyx.prism.tray ] \
  || fail "$TRAY_APP has the wrong bundle id"
[ "$(/usr/libexec/PlistBuddy -c 'Print :LSUIElement' "$TRAY_APP/Contents/Info.plist")" = true ] \
  || fail "$TRAY_APP is not LSUIElement"
[ -f "$TRAY_PLIST" ] || fail "$TRAY_PLIST missing"
plutil -lint "$TRAY_PLIST" >/dev/null || fail "$TRAY_PLIST is not a valid plist"

sudo install -d -o _blueonyx -g _blueonyx "$DATA/dataset" "$(dirname "$SEED")"
sudo sh -c "echo 'reviewed labels' > '$SEED'"
sudo chown _blueonyx:_blueonyx "$SEED"

# What the app runs for "Keep My Data".
sudo "$PROG/scripts/uninstall_launchd.sh"
[ ! -e "$PROG" ] || fail "$PROG left behind"
[ ! -e "$APP" ] || fail "$APP left behind"
[ ! -e "$TRAY_APP" ] || fail "$TRAY_APP left behind"
[ ! -e "$TRAY_PLIST" ] || fail "$TRAY_PLIST left behind"
sudo test -f "$CONFIG" || fail "config deleted without --purge"
sudo test -f "$SEED" || fail "$SEED deleted without --purge"
dscl . -read /Users/_blueonyx >/dev/null 2>&1 || fail "account _blueonyx deleted without --purge"

# A reinstall picks the data up, at once.
sudo installer -pkg "$PKG" -target / -verbose
check_daemon
sudo test -f "$SEED" || fail "$SEED lost over the reinstall"
sudo "$PROG/scripts/uninstall_launchd.sh"
sudo test -f "$SEED" || fail "$SEED deleted by the second uninstall"

# Then with the account's id changed meanwhile (deleted by hand and made again): the daemon must
# own and read what the old id wrote. The kernel refuses a bind while a socket of the port belongs
# to another uid, and the old daemon's closed connections stay in TIME_WAIT for 30 s, so wait them
# out: launchd would otherwise restart the daemon until they are gone.
sleep 35
new_uid=""
for i in $(seq 499 -1 300); do
  if ! dscl . -list /Users UniqueID | awk -v id="$i" '$2 == id { f = 1 } END { exit !f }'; then new_uid="$i"; break; fi
done
[ -n "$new_uid" ] || fail "no free id below 500"
sudo dscl . -create /Users/_blueonyx UniqueID "$new_uid"
sudo dscacheutil -flushcache
sudo installer -pkg "$PKG" -target / -verbose
check_daemon
[ "$(sudo stat -f %u "$SEED")" = "$new_uid" ] || fail "$SEED was not given to the new id $new_uid"
[ "$(sudo cat "$SEED")" = "reviewed labels" ] || fail "$SEED changed over the reinstall"

# What the app runs for "Delete Everything".
sudo "$PROG/scripts/uninstall_launchd.sh" --purge
[ ! -e "$PROG" ] || fail "$PROG left behind"
[ ! -e "$DATA" ] || fail "data root left behind"

if sudo launchctl print system/com.blueonyx.prism >/dev/null 2>&1; then fail "daemon still loaded"; fi
[ ! -e /Library/LaunchDaemons/com.blueonyx.prism.plist ] || fail "plist still present"
if dscl . -read /Users/_blueonyx >/dev/null 2>&1; then fail "account _blueonyx still present"; fi
if pkgutil --pkg-info com.blueonyx.prism >/dev/null 2>&1; then fail "receipt still present"; fi
left="$(sudo find /Applications /Library /usr/local /private/etc /private/var/db/receipts /private/var/log \
  \( -iname '*blue-onyx-prism*' -o -iname '*blueonyx*' -o -iname 'Blue Onyx Prism*' -o -iname 'Uninstall Blue Onyx Prism*' \) \
  -not -path '/private/var/log/install.log*' 2>/dev/null || true)"
[ -z "$left" ] || fail "left behind after the purge: $left"
echo "Smoke test passed."
