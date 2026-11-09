#!/usr/bin/env bash
# Upgrade test (CI only: it changes the machine, needs passwordless sudo): install the published
# OLD package, seed what a user has (a config edit, `allow_remote_admin: true`, a reviewed label
# file and its frame, a training runs.json), upgrade to the NEW package and check the service is
# up on the new binary (the retired `allow_remote_admin` is no reason to refuse the config), the
# API token was rotated once for it, the web UI answers over HTTPS in setup mode, it answers a
# detection, and every seeded file and the config edit are still there, unchanged. The label
# file is seeded root-owned, as `sudo cp` leaves it: the upgrade must hand it to the service user
# (RR-U6), its content unchanged. Ends with a purge.
# Linux (.deb, systemd) and macOS (.pkg, launchd). Called by packaging/upgrade-test.sh.
# Usage: packaging/upgrade-test-unix.sh <old installer> <new installer> <new blue-onyx-prism binary>
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: $0 <old installer> <new installer> <new blue-onyx-prism binary>" >&2
  exit 2
fi
OLD="$1"
NEW="$2"
NEW_BIN="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRY="$ROOT/scripts/ci/retry.sh"
URL=http://127.0.0.1:32168
IMG="$ROOT/tests/data/dog_bike_car.jpg"
RUNS_FIXTURE="$ROOT/tests/data/runs-v0-with-history.json"
TIMEOUT_EDIT=23 # request_timeout_secs: not the default (15), so a reset to defaults shows

# logs: defined per OS below.
fail() { echo "UPGRADE TEST FAILED: $*" >&2; logs >&2; exit 1; }
sha() { if command -v sha256sum >/dev/null; then sudo sha256sum "$1" | cut -d' ' -f1; else sudo shasum -a 256 "$1" | cut -d' ' -f1; fi; }

case "$(uname -s)" in
  Linux)
    DATA=/var/lib/blue-onyx-prism
    PROG=/opt/blue-onyx-prism
    SVC_USER=blueonyx
    owner_of() { sudo stat -c %U "$1"; }
    logs() { sudo journalctl -u blue-onyx-prism --no-pager -n 80 || true; }
    install_pkg() { # install_pkg FILE [--reinstall]
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ${2:-} "$1"
    }
    svc_stop() { sudo systemctl stop blue-onyx-prism; }
    svc_start() { sudo systemctl start blue-onyx-prism; }
    svc_pid() { systemctl show -p MainPID --value blue-onyx-prism; }
    svc_restarts() { systemctl show -p NRestarts --value blue-onyx-prism; }
    purge() { sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y blue-onyx-prism; }
    pkg_version() { dpkg-query -W -f='${Version}' blue-onyx-prism 2>/dev/null || echo none; }
    ;;
  Darwin)
    DATA=/usr/local/var/blue-onyx-prism
    PROG=/usr/local/blue-onyx-prism
    SVC_USER=_blueonyx
    PLIST=/Library/LaunchDaemons/com.blueonyx.prism.plist
    owner_of() { sudo stat -f %Su "$1"; }
    logs() {
      sudo launchctl print system/com.blueonyx.prism 2>&1 | grep -E '^\s*(state|runs|pid|last exit)' || true
      sudo sh -c 'cd /var/log/blue-onyx-prism && for f in $(ls -t | head -2); do echo "== $f"; tail -n 60 "$f"; done' || true
    }
    install_pkg() { sudo installer -pkg "$1" -target /; }
    svc_stop() {
      sudo launchctl bootout system/com.blueonyx.prism 2>/dev/null || true
      for _ in $(seq 1 40); do sudo launchctl print system/com.blueonyx.prism >/dev/null 2>&1 || return 0; sleep 1; done
      fail "the daemon did not stop within 40 s"
    }
    svc_start() { sudo launchctl bootstrap system "$PLIST"; }
    svc_pid() { sudo launchctl print system/com.blueonyx.prism 2>/dev/null | awk '/^\tpid = / { print $3 }'; }
    # launchd counts starts, not restarts: one start = no restart.
    svc_restarts() { echo $(($(sudo launchctl print system/com.blueonyx.prism | awk '/^\truns = / { print $3 }') - 1)); }
    purge() { sudo "$PROG/scripts/uninstall_launchd.sh" --purge; }
    pkg_version() { pkgutil --pkg-info com.blueonyx.prism 2>/dev/null | awk '/^version:/ { print $2 }' || echo none; }
    ;;
  *) echo "unsupported OS $(uname -s)" >&2; exit 2 ;;
esac
CONFIG="$DATA/blue_onyx_prism_config.json"
LABELS="$DATA/dataset/upgrade-test/frame.labels.json"
FRAME="$DATA/dataset/upgrade-test/frame.jpg"
RUNS="$DATA/training/runs.json"

# Ready (200): the runtime loads and the default model serves.
wait_ready() {
  local code=""
  for _ in $(seq 1 240); do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$URL/v1/status" || true)"
    [ "$code" = 200 ] && { echo "/v1/status 200 ($1)"; return 0; }
    sleep 2
  done
  fail "/v1/status not ready (200) within 8 min $1, last answer '$code'"
}

timeout_value() { sudo grep -Eo '"request_timeout_secs": *[0-9]+' "$CONFIG" | grep -Eo '[0-9]+$' || true; }

echo "== install the published package: $OLD"
if [ "$(uname -s)" = Linux ]; then "$RETRY" 4 20 sudo apt-get update -qq; fi
install_pkg "$OLD"
echo "installed version $(pkg_version)"
wait_ready "on the old version"

echo "== seed a user's data (the service stopped)"
svc_stop
[ -n "$(timeout_value)" ] || fail "$CONFIG has no request_timeout_secs to edit (the old service writes every key on its first start)"
sudo sed -i.bak -E "s/\"request_timeout_secs\": *[0-9]+/\"request_timeout_secs\": $TIMEOUT_EDIT/" "$CONFIG"
# Remote admin on, as many 0.2 installs have it: the new version retires the key (read, warned,
# never written) and must still start.
if sudo grep -Eq '"allow_remote_admin": *(true|false)' "$CONFIG"; then
  sudo sed -i.bak -E 's/"allow_remote_admin": *(true|false)/"allow_remote_admin": true/' "$CONFIG"
else
  sudo sed -i.bak -E '1 s/^\{/{ "allow_remote_admin": true,/' "$CONFIG"
fi
sudo rm -f "$CONFIG.bak"
sudo grep -Eq '"allow_remote_admin": *true' "$CONFIG" || fail "could not set allow_remote_admin in $CONFIG"
sudo install -d -o "$SVC_USER" -g "$SVC_USER" "$DATA/dataset" "$DATA/dataset/upgrade-test" "$DATA/training"
sudo install -o "$SVC_USER" -g "$SVC_USER" -m 0640 "$IMG" "$FRAME"
sudo install -o "$SVC_USER" -g "$SVC_USER" -m 0640 "$RUNS_FIXTURE" "$RUNS"
# Root-owned, as `sudo cp` leaves it: deep inside, the top folder is the service user's.
printf '{"version":1,"status":"reviewed","boxes":[{"label":"vehicle","bbox":[10,20,110,220],"state":"accepted"}],"note":"upgrade test"}\n' \
  | sudo tee "$LABELS" >/dev/null
svc_start
wait_ready "on the old version, with the seeded data"
[ "$(timeout_value)" = "$TIMEOUT_EDIT" ] || fail "the old version dropped the config edit before the upgrade (request_timeout_secs is '$(timeout_value)')"
for f in "$LABELS" "$FRAME" "$RUNS"; do sudo test -f "$f" || fail "the old version removed the seeded $f before the upgrade"; done
before_labels="$(sha "$LABELS")"
before_frame="$(sha "$FRAME")"
before_runs="$(sha "$RUNS")"
old_pid="$(svc_pid)"
[ -n "$old_pid" ] && [ "$old_pid" != 0 ] || fail "no service process before the upgrade"

echo "== upgrade to $NEW"
reinstall=""
if [ "$(uname -s)" = Linux ] && [ "$(dpkg-deb -f "$NEW" Version)" = "$(pkg_version)" ]; then
  # Same version (main between releases): apt would skip the file without --reinstall. dpkg runs
  # the same maintainer scripts as on an upgrade (prerm upgrade, postinst configure <old>).
  echo "the new package has the installed version $(pkg_version): reinstalling"
  reinstall=--reinstall
fi
install_pkg "$NEW" $reinstall
echo "installed version $(pkg_version)"
[ "$(sha "$PROG/blue-onyx-prism")" = "$(sha "$NEW_BIN")" ] || fail "$PROG/blue-onyx-prism is not the new binary $NEW_BIN"
wait_ready "after the upgrade"
new_pid="$(svc_pid)"
[ -n "$new_pid" ] && [ "$new_pid" != 0 ] || fail "no service process after the upgrade"
[ "$new_pid" != "$old_pid" ] || fail "the service process is still $old_pid: the upgrade did not restart it on the new binary"
[ "$(svc_restarts)" = 0 ] || fail "the service restarted $(svc_restarts) times after the upgrade"
echo "service up on the new binary (pid $old_pid -> $new_pid), no restarts"

echo "== the retired allow_remote_admin, the API token and the web UI"
# The token the old version may have sent over plain HTTP was rotated once (marker), into the
# protected auth folder; no account exists, so the service is in setup mode (setup code made).
sudo test -f "$DATA/auth/.token-rotated-v1" || fail "no $DATA/auth/.token-rotated-v1: the API token was not rotated for allow_remote_admin"
sudo test -f "$DATA/auth/admin-token" || fail "no API token in $DATA/auth"
sudo test -f "$DATA/auth/setup-code" || fail "no setup code in $DATA/auth: the service is not in setup mode"
"$ROOT/scripts/ci/check_https.sh" 32168 32169 || fail "the web UI over HTTPS after the upgrade"

echo "== a detection"
out="$(curl -s --max-time 60 -F image=@"$IMG" -F min_confidence=0.2 "$URL/v1/vision/detection" || true)"
echo "detection: $out"
echo "$out" | grep -q '"success":true' || fail "the detection did not succeed"
echo "$out" | grep -q '"label":"vehicle"' || fail "the detection found no vehicle on the test image"

echo "== the user's data"
[ "$(timeout_value)" = "$TIMEOUT_EDIT" ] || fail "the config edit is gone: request_timeout_secs is '$(timeout_value)', was $TIMEOUT_EDIT"
for f in "$LABELS" "$FRAME" "$RUNS"; do sudo test -f "$f" || fail "$f is gone after the upgrade"; done
[ "$(sha "$LABELS")" = "$before_labels" ] || fail "$LABELS changed over the upgrade"
[ "$(sha "$FRAME")" = "$before_frame" ] || fail "$FRAME changed over the upgrade"
[ "$(sha "$RUNS")" = "$before_runs" ] || fail "$RUNS changed over the upgrade"
if sudo ls "$DATA/training" | grep -q 'corrupt'; then fail "the new version set runs.json aside as damaged: $(sudo ls "$DATA/training")"; fi
[ "$(owner_of "$LABELS")" = "$SVC_USER" ] || fail "$LABELS (seeded root-owned) belongs to $(owner_of "$LABELS"), not $SVC_USER, after the upgrade"
echo "config edit kept; label file, frame and runs.json unchanged; the label file handed to $SVC_USER"

echo "== purge"
purge
echo "UPGRADE TEST PASSED ($(basename "$OLD") -> $(basename "$NEW"))"
