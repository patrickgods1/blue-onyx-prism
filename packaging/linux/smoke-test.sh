#!/usr/bin/env bash
# CI smoke test (ubuntu runner with sudo + systemd + docker): install the .deb, check user / unit /
# HTTP / HTTPS / setup code / config; `apt remove` keeps only the data and the user, a seeded dataset file included; a
# reinstall over them, with the user's uid changed, enables and starts the service and hands it the
# data; `apt purge` leaves nothing behind. Then the .rpm: contents and scriptlets, and in a Fedora
# container `dnf remove`, which keeps the same things (RPM has no purge), a reinstall the same way,
# and the commands the erase prints, which must leave nothing behind.
# Usage: packaging/linux/smoke-test.sh <file.deb> <file.rpm>
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <file.deb> <file.rpm>" >&2
  exit 2
fi
DEB="$(realpath "$1")"
RPM="$(realpath "$2")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RETRY="$REPO/scripts/ci/retry.sh"
UNIT=blue-onyx-prism
DATA=/var/lib/blue-onyx-prism
CONFIG="$DATA/blue_onyx_prism_config.json"
# Stands for what no download brings back (dataset frames, reviewed labels, training runs). In a
# folder of its own: nothing the service prunes walks it.
SEED="$DATA/dataset/smoke-test/frame.labels.json"
# A uid no earlier blueonyx had: well below the 999 where system uids are handed out downwards.
NEW_UID=900

fail() { echo "SMOKE TEST FAILED: $*" >&2; sudo journalctl -u "$UNIT" --no-pager -n 50 || true; exit 1; }

echo "== deb install"
# Only the network is retried. The install itself runs once: a second try would finish a failed
# postinst and hide it, and that is what this test is for.
"$RETRY" 4 20 sudo apt-get update -qq
"$RETRY" 4 20 sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --download-only "$DEB"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$DEB"

id -u blueonyx >/dev/null 2>&1 || fail "user blueonyx missing"
[ -x /opt/blue-onyx-prism/blue-onyx-prism ] || fail "binary missing"
[ -x /opt/blue-onyx-prism/blue-onyx-prism-benchmark ] || fail "benchmark binary missing"
[ -f /opt/blue-onyx-prism/scripts/systemd_common.sh ] || fail "systemd_common.sh missing"
[ -f /opt/blue-onyx-prism/scripts/train_model.py ] || fail "train_model.py missing"
[ -f /usr/lib/systemd/system/$UNIT.service ] || fail "unit missing"

# The unit is enabled and comes up ready without a restart.
check_service() {
  systemctl is-enabled "$UNIT" >/dev/null || fail "unit not enabled"

  local state=""
  for _ in $(seq 1 60); do
    state="$(systemctl is-active "$UNIT" || true)"
    [ "$state" = active ] && break
    [ "$state" = activating ] || [ "$state" = active ] || break
    sleep 1
  done
  [ "$state" = active ] || fail "unit state is '$state', expected active"

  local code=""
  for _ in $(seq 1 120); do
    code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:32168/v1/status || true)"
    case "$code" in 200 | 503) break ;; esac
    sleep 1
  done
  case "$code" in 200 | 503) echo "status endpoint answered $code" ;; *) fail "/v1/status answered '$code'" ;; esac

  # Answering is not enough: the service must load the runtime it downloads and serve. A runtime
  # that cannot load (e.g. one built for a newer glibc) makes it exit and restart at every start,
  # while HTTP still answers between restarts.
  echo "== waiting until the service is ready (runtime + model downloaded, model compiled)"
  for _ in $(seq 1 180); do
    code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:32168/v1/status || true)"
    [ "$code" = 200 ] && break
    sleep 2
  done
  [ "$code" = 200 ] || fail "/v1/status not ready (200) within 6 min, last answer '$code'"
  local restarts
  restarts="$(systemctl show -p NRestarts --value "$UNIT")"
  [ "$restarts" = 0 ] || fail "the service restarted $restarts times"
  echo "service ready, no restarts"

  # The web UI: HTTPS on 32169 behind a sign-in, the plain port sends browsers there.
  "$REPO/scripts/ci/check_https.sh" 32168 32169 || fail "the HTTPS port"
  # No account yet: the service made the one-time setup code in its private auth folder, and
  # `setup-code` (as the service's account, with its config) prints it.
  [ "$(sudo stat -c '%a %U' "$DATA/auth")" = "700 blueonyx" ] || fail "$DATA/auth is not 0700 blueonyx"
  [ "$(sudo stat -c '%a %U' "$DATA/auth/setup-code")" = "600 blueonyx" ] || fail "no 0600 setup-code file in $DATA/auth"
  local out
  out="$(cd / && sudo -u blueonyx /opt/blue-onyx-prism/blue-onyx-prism --config "$CONFIG" setup-code)" || fail "setup-code failed"
  echo "$out" | grep -Eq '^Setup code: [A-Z2-7-]{20,}' || fail "setup-code printed no code"
  echo "setup-code prints a code"
}
check_service

sudo test -f "$CONFIG" || fail "config $CONFIG missing"
[ "$(sudo stat -c %U:%G "$CONFIG")" = blueonyx:blueonyx ] || fail "config not owned by blueonyx"
sudo grep -q '"download_dir"' "$CONFIG" || fail "config lacks download_dir"
# The service rewrites the config on first start with every key ("openvino_dir": null): only a
# path value names a runtime.
if sudo grep -Eq '"(openvino_dir|onnxruntime_dir)": *"' "$CONFIG"; then fail "config names bundled runtimes"; fi

# Files owned by no user/group before the install: after the purge there must be no new ones
# (files the deleted blueonyx user still owned somewhere).
orphans() { sudo find / -xdev \( -nouser -o -nogroup \) -not -path '/proc/*' 2>/dev/null | sort; }
orphans_before="$(orphans)"
# What the CLI creates when run from the program dir (config, downloads): not package files.
add_cli_files() {
  sudo mkdir -p /opt/blue-onyx-prism/openvino
  sudo touch /opt/blue-onyx-prism/openvino/placeholder.so /opt/blue-onyx-prism/blue_onyx_prism_config.json
}
add_cli_files
sudo install -d -o blueonyx -g blueonyx "$DATA/dataset" "$(dirname "$SEED")"
sudo sh -c "echo 'reviewed labels' > '$SEED'"
sudo chown blueonyx:blueonyx "$SEED"

echo "== deb remove (keeps the data and the user)"
sudo DEBIAN_FRONTEND=noninteractive apt-get remove -y blue-onyx-prism
if systemctl is-active --quiet "$UNIT"; then fail "unit still active after remove"; fi
[ ! -e /usr/lib/systemd/system/$UNIT.service ] || fail "unit file left behind by remove"
[ ! -e /opt/blue-onyx-prism ] || fail "/opt/blue-onyx-prism left behind by remove"
sudo test -f "$CONFIG" || fail "remove deleted the config (only purge may)"
sudo test -f "$SEED" || fail "remove deleted $SEED (only purge may)"
id -u blueonyx >/dev/null 2>&1 || fail "remove deleted the user (only purge may)"

echo "== deb reinstall over the kept data, with the user's uid changed"
# The user deleted by hand and made again: the service must own and read what the old uid wrote.
sudo userdel blueonyx
sudo useradd --system --uid "$NEW_UID" --user-group --no-create-home --home-dir "$DATA" --shell /usr/sbin/nologin blueonyx
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$DEB"
# dpkg calls this like an upgrade, but the removal disabled the unit.
check_service
[ "$(sudo stat -c %u "$SEED")" = "$NEW_UID" ] || fail "$SEED was not given to the new uid"
[ "$(sudo cat "$SEED")" = "reviewed labels" ] || fail "$SEED changed over the reinstall"
add_cli_files

echo "== deb purge (leaves nothing behind)"
sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y blue-onyx-prism
[ ! -e "$DATA" ] || fail "$DATA left behind"
if id -u blueonyx >/dev/null 2>&1; then fail "user blueonyx left behind"; fi
if getent group blueonyx >/dev/null 2>&1; then fail "group blueonyx left behind"; fi
if dpkg -s blue-onyx-prism >/dev/null 2>&1; then fail "dpkg still knows blue-onyx-prism after purge"; fi
[ ! -e /usr/lib/sysusers.d/blue-onyx-prism.conf ] || fail "sysusers file left behind"
[ ! -e "/etc/systemd/system/multi-user.target.wants/$UNIT.service" ] || fail "enable symlink left behind"
left="$(sudo find / -xdev \( -iname '*blue-onyx-prism*' -o -iname '*blueonyx*' \) \
  -not -path '/proc/*' -not -path '/home/runner/*' -not -path "${DEB%/*}/*" -not -path "${RPM%/*}/*" 2>/dev/null || true)"
[ -z "$left" ] || fail "left behind after purge: $left"
new_orphans="$(comm -13 <(echo "$orphans_before") <(orphans))"
[ -z "$new_orphans" ] || fail "files owned by the deleted user left behind: $new_orphans"

echo "== rpm sanity"
command -v rpm >/dev/null || "$RETRY" 4 20 sudo apt-get install -y rpm
files="$(rpm -qpl "$RPM")"; echo "$files"
echo "$files" | grep -q "/opt/blue-onyx-prism/blue-onyx-prism$" || fail "rpm lacks the binary"
echo "$files" | grep -q '/usr/lib/systemd/system/blue-onyx-prism.service' || fail "rpm lacks the unit"
echo "$files" | grep -q '/usr/lib/sysusers.d/blue-onyx-prism.conf' || fail "rpm lacks the sysusers file"
scripts="$(rpm -qp --scripts "$RPM")"
echo "$scripts" | grep -q 'pkg_scriptlet.sh post-install' || fail "rpm lacks the %post scriptlet"
echo "$scripts" | grep -q 'pkg_scriptlet.sh pre-remove' || fail "rpm lacks the %preun scriptlet"
rpm -qp --requires "$RPM" | grep -q systemd || fail "rpm does not require systemd"

echo "== rpm install + dnf remove + reinstall in Fedora"
# No systemd in the container: the scriptlets skip systemctl, the files and the user are real.
FEDORA=fedora:42@sha256:99e203b80b1c3d8f7e161ec10a68fd02b081ef83a3963553e513c82846b97814
docker run --rm -v "${RPM%/*}:/pkg:ro" "$FEDORA" bash -euo pipefail -c '
  rpm=/pkg/'"${RPM##*/}"'
  # Named here so the first dnf remove keeps it (it could go as an unused dependency otherwise).
  dnf install -y -q shadow-utils
  dnf install -y -q "$rpm"
  id -u blueonyx >/dev/null || { echo "user blueonyx missing after install"; exit 1; }
  test -f /var/lib/blue-onyx-prism/blue_onyx_prism_config.json || { echo "config not seeded"; exit 1; }
  # What a running service and a CLI run from /opt would add.
  data=/var/lib/blue-onyx-prism
  seed='"$SEED"'
  install -d -o blueonyx -g blueonyx "$data/models" "$data/dataset" "$(dirname "$seed")"
  install -o blueonyx -g blueonyx /dev/null "$data/models/placeholder.onnx"
  echo "reviewed labels" > "$seed" && chown blueonyx:blueonyx "$seed"
  mkdir -p /opt/blue-onyx-prism/openvino && touch /opt/blue-onyx-prism/openvino/placeholder.so
  dnf remove -y blue-onyx-prism
  ! rpm -q blue-onyx-prism >/dev/null || { echo "rpm still installed"; exit 1; }
  [ ! -e /opt/blue-onyx-prism ] || { echo "/opt/blue-onyx-prism left behind by dnf remove"; exit 1; }
  [ ! -e /usr/lib/sysusers.d/blue-onyx-prism.conf ] || { echo "sysusers file left behind"; exit 1; }
  [ ! -e /usr/lib/systemd/system/blue-onyx-prism.service ] || { echo "unit left behind"; exit 1; }
  test -f "$data/blue_onyx_prism_config.json" || { echo "dnf remove deleted the config"; exit 1; }
  test -f "$seed" || { echo "dnf remove deleted $seed"; exit 1; }
  test -f "$data/models/placeholder.onnx" || { echo "dnf remove deleted the models"; exit 1; }
  id -u blueonyx >/dev/null || { echo "dnf remove deleted the user"; exit 1; }
  left="$(find / -xdev \( -iname "*blue-onyx-prism*" -o -iname "*blueonyx*" \) -not -path "/pkg/*" -not -path "/proc/*" -not -path "$data" -not -path "$data/*" 2>/dev/null || true)"
  [ -z "$left" ] || { echo "left behind by dnf remove, outside the data: $left"; exit 1; }

  # A reinstall hands the data to the service user, also one with another uid (deleted by hand
  # and made again).
  userdel blueonyx
  useradd --system --uid '"$NEW_UID"' --user-group --no-create-home --home-dir "$data" --shell /usr/sbin/nologin blueonyx
  dnf install -y -q "$rpm"
  [ "$(stat -c %u "$seed")" = '"$NEW_UID"' ] || { echo "$seed was not given to the new uid"; exit 1; }
  [ "$(cat "$seed")" = "reviewed labels" ] || { echo "$seed changed over the reinstall"; exit 1; }

  # rpm prints what a scriptlet prints: the erase says how to delete the data, and doing that
  # leaves nothing behind.
  out="$(rpm -e blue-onyx-prism)"
  echo "$out"
  echo "$out" | grep -qF "sudo rm -rf $data && sudo userdel blueonyx" || { echo "the erase did not print how to delete the data"; exit 1; }
  rm -rf "$data" && userdel blueonyx
  ! id -u blueonyx >/dev/null 2>&1 || { echo "user blueonyx left behind"; exit 1; }
  ! getent group blueonyx >/dev/null || { echo "group blueonyx left behind"; exit 1; }
  left="$(find / -xdev \( -iname "*blue-onyx-prism*" -o -iname "*blueonyx*" \) -not -path "/pkg/*" -not -path "/proc/*" 2>/dev/null || true)"
  [ -z "$left" ] || { echo "left behind after dnf remove: $left"; exit 1; }
  # /pkg is the host folder mounted read-only (owned by the runner user, unknown in here).
  orphans="$(find / -xdev \( -nouser -o -nogroup \) -not -path "/proc/*" -not -path /pkg -not -path "/pkg/*" 2>/dev/null || true)"
  [ -z "$orphans" ] || { echo "files owned by the deleted user left behind: $orphans"; exit 1; }
  echo "the printed commands left nothing behind"
' || fail "rpm install / remove in Fedora"

echo "SMOKE TEST PASSED"
