#!/usr/bin/env bash
# Runs the .deb and .rpm maintainer scripts against a throwaway root: no root needed, nothing on
# the machine changes (any OS with bash). The scripts' absolute paths are rewritten into the
# temp folder and every system command is a stub that records its arguments, so this checks the
# decisions (what stays, what goes, install or upgrade); smoke-test.sh checks the real packages.
# Usage: packaging/linux/test-scriptlets.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ROOT="$WORK/root"
BIN="$WORK/bin"
CALLS="$WORK/calls"
OUT="$WORK/out"
OPT="$ROOT/opt/blue-onyx-prism"
DATA="$ROOT/var/lib/blue-onyx-prism"
failed=0

fail() { echo "FAILED [$scenario]: $*" >&2; failed=1; }
called() { grep -q -- "$1" "$CALLS"; }
expect_call() { called "$1" || fail "expected a call to '$1'; calls: $(tr '\n' ';' < "$CALLS")"; }
expect_no_call() { if called "$1"; then fail "unexpected call to '$1'"; fi; }
expect_file() { [ -e "$1" ] || fail "$1 is missing"; }
expect_no_file() { [ ! -e "$1" ] || fail "$1 should be gone"; }
expect_out() { grep -q -- "$1" "$OUT" || fail "output lacks '$1': $(cat "$OUT")"; }

# Copy $1 to $2 with its absolute paths moved into $ROOT.
rewrite() {
  sed -e "s#/opt/blue-onyx-prism#$OPT#g" -e "s#/var/lib/blue-onyx-prism#$DATA#g" \
    -e "s#/usr/lib/sysusers.d#$ROOT/usr/lib/sysusers.d#g" -e "s#/run/systemd/system#$ROOT/run/systemd/system#g" \
    -e "s#/etc/systemd/system#$ROOT/etc/systemd/system#g" "$1" > "$2"
  chmod +x "$2"
}

stub() { # stub NAME BODY: a command that records its arguments, then runs BODY
  printf '#!/bin/sh\necho "%s $*" >> "%s"\n%s\n' "$1" "$CALLS" "$2" > "$BIN/$1"
  chmod +x "$BIN/$1"
}

mkdir -p "$BIN"
for c in systemctl useradd usermod groupdel systemd-sysusers chown; do stub "$c" 'exit 0'; done
stub userdel "rm -f '$WORK/user'"
# The user: its uid and gid are in $WORK/user, the data root's owner in $WORK/owner.
stub id "[ -f '$WORK/user' ] || exit 1; cat '$WORK/user'"
stub getent "[ -f '$WORK/user' ]"
stub stat "cat '$WORK/owner'"
# install -d -o u -g g -m 0750 DIR...: create the folders without the (root only) owner.
stub install 'while [ $# -gt 0 ]; do case "$1" in -d) ;; -o | -g | -m) shift ;; *) mkdir -p "$1" ;; esac; shift; done'

# A machine with the package installed, its data seeded and its user present (same owner).
setup() {
  scenario="$1"
  rm -rf "$ROOT"
  : > "$CALLS"
  mkdir -p "$OPT/scripts" "$OPT/openvino" "$DATA/dataset" "$ROOT/usr/lib/sysusers.d" "$ROOT/run/systemd/system"
  : > "$OPT/openvino/created-by-the-cli.so"
  : > "$DATA/dataset/frame.labels.json"
  echo '{}' > "$DATA/blue_onyx_prism_config.json"
  echo 990 > "$WORK/user"
  echo 990:990 > "$WORK/owner"
  rewrite "$HERE/pkg-scriptlet.sh" "$OPT/scripts/pkg_scriptlet.sh"
  rewrite "$REPO/scripts/systemd_common.sh" "$OPT/scripts/systemd_common.sh"
}

# run SCRIPT ARGS...: a maintainer script as sh runs it, its output in $OUT.
run() {
  local script="$1"
  shift
  rewrite "$script" "$WORK/script"
  PATH="$BIN:$PATH" sh "$WORK/script" "$@" > "$OUT" 2>&1 || fail "$(basename "$script") $* exited with $?: $(cat "$OUT")"
}

setup "rpm erase keeps the data and the user"
run "$HERE/rpm/post_uninstall.sh" 0
expect_no_file "$OPT"
expect_file "$DATA/dataset/frame.labels.json"
expect_file "$DATA/blue_onyx_prism_config.json"
expect_no_call userdel
expect_no_call groupdel
expect_call 'systemctl daemon-reload'
expect_out "$DATA"
expect_out "rm -rf $DATA"
expect_out 'userdel blueonyx'

setup "rpm upgrade touches nothing"
run "$HERE/rpm/post_uninstall.sh" 1
expect_file "$OPT/openvino/created-by-the-cli.so"
expect_file "$DATA/dataset/frame.labels.json"
expect_no_call userdel

setup "deb remove keeps the data and the user"
run "$HERE/debian/postrm" remove
expect_no_file "$OPT"
expect_file "$DATA/dataset/frame.labels.json"
expect_no_call userdel
expect_out "$DATA"
expect_out 'apt purge blue-onyx-prism'

setup "deb purge deletes the data and the user"
run "$HERE/debian/postrm" remove
run "$HERE/debian/postrm" purge
expect_no_file "$DATA"
expect_call 'userdel blueonyx'

setup "deb upgrade touches nothing"
run "$HERE/debian/postrm" upgrade 0.2.0
expect_file "$OPT/openvino/created-by-the-cli.so"
expect_file "$DATA/dataset/frame.labels.json"
expect_no_call userdel

setup "deb first install enables and starts"
run "$HERE/debian/preinst" install
run "$HERE/debian/postinst" configure ""
expect_call 'systemctl enable blue-onyx-prism'
expect_call 'systemctl start blue-onyx-prism'

setup "deb upgrade restarts a running service and leaves a disabled one alone"
run "$HERE/debian/preinst" upgrade 0.2.0 0.3.0
run "$HERE/debian/postinst" configure 0.2.0
expect_call 'systemctl try-restart blue-onyx-prism'
expect_no_call 'systemctl enable'

# `apt remove` leaves the package in config-files state, so dpkg passes the old version to the
# postinst as on an upgrade, though the removal disabled the unit. Only the preinst can tell.
setup "deb reinstall after a remove enables and starts"
run "$HERE/debian/postrm" remove
mkdir -p "$OPT/scripts"
rewrite "$HERE/pkg-scriptlet.sh" "$OPT/scripts/pkg_scriptlet.sh"
rewrite "$REPO/scripts/systemd_common.sh" "$OPT/scripts/systemd_common.sh"
run "$HERE/debian/preinst" install 0.2.0 0.3.0
run "$HERE/debian/postinst" configure 0.2.0
expect_call 'systemctl enable blue-onyx-prism'
expect_call 'systemctl start blue-onyx-prism'
expect_no_call 'systemctl try-restart'
expect_no_file "$OPT/.reinstall"

setup "install over data of the same user changes no owner"
run "$HERE/rpm/post_install.sh" 1
expect_no_call 'chown -hR'

# Before `install -d`, which corrects the top folders only and would hide the mismatch from this
# check and from systemd's own StateDirectory= one.
setup "install over data of an older uid takes the whole tree"
echo 777:777 > "$WORK/owner"
run "$HERE/rpm/post_install.sh" 1
expect_call "chown -hR blueonyx:blueonyx $DATA/."
if [ "$(grep -n 'chown -hR' "$CALLS" | head -1 | cut -d: -f1)" -gt "$(grep -n '^install -d' "$CALLS" | head -1 | cut -d: -f1)" ]; then
  fail "the owner check ran after install -d"
fi

setup "a chown that fails warns and the install goes on"
echo 777:777 > "$WORK/owner"
stub chown 'case "$1" in -hR) exit 1 ;; esac'
run "$HERE/rpm/post_install.sh" 1
expect_out 'Could not'
expect_call 'systemctl enable blue-onyx-prism'
stub chown 'exit 0'

[ "$failed" -eq 0 ] || exit 1
echo "scriptlet checks passed"
