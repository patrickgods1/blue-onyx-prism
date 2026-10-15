#!/usr/bin/env bash
# CI smoke test (ubuntu runner with sudo + systemd + docker): install the .deb, check user / unit /
# HTTP / config; `apt remove` keeps only the data and the user; `apt purge` leaves nothing behind.
# Then the .rpm: contents and scriptlets, and an install + `dnf remove` (which must leave nothing
# behind: RPM has no purge) in a Fedora container.
# Usage: packaging/linux/smoke-test.sh <file.deb> <file.rpm>
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <file.deb> <file.rpm>" >&2
  exit 2
fi
DEB="$(realpath "$1")"
RPM="$(realpath "$2")"
UNIT=blue-onyx-prism
DATA=/var/lib/blue-onyx-prism
CONFIG="$DATA/blue_onyx_prism_config.json"

fail() { echo "SMOKE TEST FAILED: $*" >&2; sudo journalctl -u "$UNIT" --no-pager -n 50 || true; exit 1; }

echo "== deb install"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$DEB"

id -u blueonyx >/dev/null 2>&1 || fail "user blueonyx missing"
[ -x /opt/blue-onyx-prism/blue-onyx-prism ] || fail "binary missing"
[ -x /opt/blue-onyx-prism/blue-onyx-prism-benchmark ] || fail "benchmark binary missing"
[ -f /opt/blue-onyx-prism/scripts/systemd_common.sh ] || fail "systemd_common.sh missing"
[ -f /usr/lib/systemd/system/$UNIT.service ] || fail "unit missing"
systemctl is-enabled "$UNIT" >/dev/null || fail "unit not enabled"

state=""
for _ in $(seq 1 60); do
  state="$(systemctl is-active "$UNIT" || true)"
  [ "$state" = active ] && break
  [ "$state" = activating ] || [ "$state" = active ] || break
  sleep 1
done
[ "$state" = active ] || fail "unit state is '$state', expected active"

code=""
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
restarts="$(systemctl show -p NRestarts --value "$UNIT")"
[ "$restarts" = 0 ] || fail "the service restarted $restarts times"
echo "service ready, no restarts"

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
sudo mkdir -p /opt/blue-onyx-prism/openvino
sudo touch /opt/blue-onyx-prism/openvino/placeholder.so /opt/blue-onyx-prism/blue_onyx_prism_config.json

echo "== deb remove (keeps the data and the user)"
sudo DEBIAN_FRONTEND=noninteractive apt-get remove -y blue-onyx-prism
if systemctl is-active --quiet "$UNIT"; then fail "unit still active after remove"; fi
[ ! -e /usr/lib/systemd/system/$UNIT.service ] || fail "unit file left behind by remove"
[ ! -e /opt/blue-onyx-prism ] || fail "/opt/blue-onyx-prism left behind by remove"
sudo test -f "$CONFIG" || fail "remove deleted the config (only purge may)"
id -u blueonyx >/dev/null 2>&1 || fail "remove deleted the user (only purge may)"

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
command -v rpm >/dev/null || sudo apt-get install -y rpm
files="$(rpm -qpl "$RPM")"; echo "$files"
echo "$files" | grep -q "/opt/blue-onyx-prism/blue-onyx-prism$" || fail "rpm lacks the binary"
echo "$files" | grep -q '/usr/lib/systemd/system/blue-onyx-prism.service' || fail "rpm lacks the unit"
echo "$files" | grep -q '/usr/lib/sysusers.d/blue-onyx-prism.conf' || fail "rpm lacks the sysusers file"
scripts="$(rpm -qp --scripts "$RPM")"
echo "$scripts" | grep -q 'pkg_scriptlet.sh post-install' || fail "rpm lacks the %post scriptlet"
echo "$scripts" | grep -q 'pkg_scriptlet.sh pre-remove' || fail "rpm lacks the %preun scriptlet"
rpm -qp --requires "$RPM" | grep -q systemd || fail "rpm does not require systemd"

echo "== rpm install + dnf remove in Fedora (leaves nothing behind)"
# No systemd in the container: the scriptlets skip systemctl, the files and the user are real.
FEDORA=fedora:42@sha256:99e203b80b1c3d8f7e161ec10a68fd02b081ef83a3963553e513c82846b97814
docker run --rm -v "${RPM%/*}:/pkg:ro" "$FEDORA" bash -euo pipefail -c '
  rpm=/pkg/'"${RPM##*/}"'
  dnf install -y -q "$rpm"
  id -u blueonyx >/dev/null || { echo "user blueonyx missing after install"; exit 1; }
  test -f /var/lib/blue-onyx-prism/blue_onyx_prism_config.json || { echo "config not seeded"; exit 1; }
  # What a running service and a CLI run from /opt would add.
  install -d -o blueonyx -g blueonyx /var/lib/blue-onyx-prism/models
  install -o blueonyx -g blueonyx /dev/null /var/lib/blue-onyx-prism/models/placeholder.onnx
  mkdir -p /opt/blue-onyx-prism/openvino && touch /opt/blue-onyx-prism/openvino/placeholder.so
  dnf remove -y -q blue-onyx-prism
  ! rpm -q blue-onyx-prism >/dev/null || { echo "rpm still installed"; exit 1; }
  ! id -u blueonyx >/dev/null 2>&1 || { echo "user blueonyx left behind"; exit 1; }
  ! getent group blueonyx >/dev/null || { echo "group blueonyx left behind"; exit 1; }
  left="$(find / -xdev \( -iname "*blue-onyx-prism*" -o -iname "*blueonyx*" \) -not -path "/pkg/*" -not -path "/proc/*" 2>/dev/null || true)"
  [ -z "$left" ] || { echo "left behind after dnf remove: $left"; exit 1; }
  # /pkg is the host folder mounted read-only (owned by the runner user, unknown in here).
  orphans="$(find / -xdev \( -nouser -o -nogroup \) -not -path "/proc/*" -not -path /pkg -not -path "/pkg/*" 2>/dev/null || true)"
  [ -z "$orphans" ] || { echo "files owned by the deleted user left behind: $orphans"; exit 1; }
  echo "dnf remove left nothing behind"
' || fail "rpm install / remove in Fedora"

echo "SMOKE TEST PASSED"
