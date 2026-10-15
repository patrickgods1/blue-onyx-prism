#!/usr/bin/env bash
# Install the pkg, check the daemon answers the uninstaller app and the tray app are in place, uninstall keeping
# the data (program and app gone, data kept), then purge. For a macOS CI runner (passwordless
# sudo). Usage: packaging/macos/smoke-test.sh <pkg>
set -euo pipefail

PKG="${1:?Usage: $0 <pkg>}"
CONFIG=/usr/local/var/blue-onyx-prism/blue_onyx_prism_config.json
PROG=/usr/local/blue-onyx-prism
APP="/Applications/Uninstall Blue Onyx Prism.app"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fail() { echo "SMOKE TEST FAILED: $*" >&2; exit 1; }

sudo installer -pkg "$PKG" -target / -verbose

sudo launchctl print system/com.blueonyx.prism >/dev/null || fail "daemon not loaded"
sudo test -f "$CONFIG" || fail "config $CONFIG missing"
owner="$(sudo stat -f %Su "$CONFIG")"
[ "$owner" = "_blueonyx" ] || fail "config owned by $owner, expected _blueonyx"

code=000
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
runs="$(sudo launchctl print system/com.blueonyx.prism | awk '/^\truns = / { print $3 }')"
[ "$runs" = 1 ] || fail "the daemon was started $runs times (expected 1)"
echo "daemon ready, started once"

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

# What the app runs for "Keep My Data".
sudo "$PROG/scripts/uninstall_launchd.sh"
[ ! -e "$PROG" ] || fail "$PROG left behind"
[ ! -e "$APP" ] || fail "$APP left behind"
[ ! -e "$TRAY_APP" ] || fail "$TRAY_APP left behind"
[ ! -e "$TRAY_PLIST" ] || fail "$TRAY_PLIST left behind"
sudo test -f "$CONFIG" || fail "config deleted without --purge"
dscl . -read /Users/_blueonyx >/dev/null 2>&1 || fail "account _blueonyx deleted without --purge"

# "Delete Everything" (the program dir is gone by now: use the repo's copy of the script).
sudo "$REPO/scripts/uninstall_launchd.sh" --purge --keep-program
[ ! -e /usr/local/var/blue-onyx-prism ] || fail "data root left behind"

if sudo launchctl print system/com.blueonyx.prism >/dev/null 2>&1; then fail "daemon still loaded"; fi
[ ! -e /Library/LaunchDaemons/com.blueonyx.prism.plist ] || fail "plist still present"
if dscl . -read /Users/_blueonyx >/dev/null 2>&1; then fail "account _blueonyx still present"; fi
if pkgutil --pkg-info com.blueonyx.prism >/dev/null 2>&1; then fail "receipt still present"; fi
left="$(sudo find /Applications /Library /usr/local /private/etc /private/var/db/receipts /private/var/log \
  \( -iname '*blue-onyx-prism*' -o -iname '*blueonyx*' -o -iname 'Blue Onyx Prism*' -o -iname 'Uninstall Blue Onyx Prism*' \) \
  -not -path '/private/var/log/install.log*' 2>/dev/null || true)"
[ -z "$left" ] || fail "left behind after the purge: $left"
echo "Smoke test passed."
