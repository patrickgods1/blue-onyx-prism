#!/usr/bin/env bash
# CI smoke test (ubuntu runner with sudo + systemd): install the .deb, check user / unit / HTTP /
# config, purge, check everything is gone; then sanity-check the .rpm contents and scriptlets.
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

sudo test -f "$CONFIG" || fail "config $CONFIG missing"
[ "$(sudo stat -c %U:%G "$CONFIG")" = blueonyx:blueonyx ] || fail "config not owned by blueonyx"
sudo grep -q '"download_dir"' "$CONFIG" || fail "config lacks download_dir"
if sudo grep -q 'openvino_dir\|onnxruntime_dir' "$CONFIG"; then fail "config names bundled runtimes"; fi

echo "== deb purge"
sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y blue-onyx-prism
if systemctl is-active --quiet "$UNIT"; then fail "unit still active after purge"; fi
[ ! -e /usr/lib/systemd/system/$UNIT.service ] || fail "unit file left behind"
[ ! -e /opt/blue-onyx-prism ] || fail "/opt/blue-onyx-prism left behind"
[ ! -e "$DATA" ] || fail "$DATA left behind"
if id -u blueonyx >/dev/null 2>&1; then fail "user blueonyx left behind"; fi

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

echo "SMOKE TEST PASSED"
