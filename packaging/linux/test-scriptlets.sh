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
# Owners: everything belongs to the user (uid = gid = $WORK/user) unless $WORK/owners names it in
# a line "<path> <uid>:<gid>". The stubbed find walks the real tree and prints what
# `find DIR/. ( ! -user U -o ! -group U ) -print -quit` would: the first entry of another owner.
# The stubbed stat answers `stat -Lc %u:%g DIR` from the same table (what an older scriptlet
# looked at), so a check of the top folder alone gets a fair run and misses a deeper file.
REAL_FIND="$(command -v find)"
cat > "$BIN/owner_of" <<'EOF'
#!/bin/sh
# owner_of WORK PATH: "uid:gid" of PATH in WORK/owners (PATH with its /./ dropped), else the user's.
p="$(printf '%s' "$2" | sed -e 's#/\./#/#g' -e 's#/\.$##')"
o="$(awk -v p="$p" '$1 == p { print $2 }' "$1/owners")"
[ -n "$o" ] || o="$(cat "$1/user"):$(cat "$1/user")"
echo "$o"
EOF
chmod +x "$BIN/owner_of"
cat > "$BIN/find" <<EOF
#!/bin/sh
echo "find \$*" >> "$CALLS"
want="\$(cat "$WORK/user"):\$(cat "$WORK/user")"
"$REAL_FIND" "\$1" | while IFS= read -r p; do
  [ "\$("$BIN/owner_of" "$WORK" "\$p")" = "\$want" ] || { echo "\$p"; break; }
done
EOF
chmod +x "$BIN/find"
cat > "$BIN/stat" <<EOF
#!/bin/sh
echo "stat \$*" >> "$CALLS"
for last in "\$@"; do :; done
"$BIN/owner_of" "$WORK" "\$last"
EOF
chmod +x "$BIN/stat"
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
  : > "$WORK/owners"
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
echo "$DATA 777:777" > "$WORK/owners"
run "$HERE/rpm/post_install.sh" 1
expect_call "chown -hR blueonyx:blueonyx $DATA/."
if [ "$(grep -n 'chown -hR' "$CALLS" | head -1 | cut -d: -f1)" -gt "$(grep -n '^install -d' "$CALLS" | head -1 | cut -d: -f1)" ]; then
  fail "the owner check ran after install -d"
fi

# RR-U6: the top folder is the user's, so `install -d` and systemd's StateDirectory= see nothing
# wrong, but a file inside is not, and the service cannot read it.
setup "install over data with one file of an older uid deep inside takes the whole tree"
mkdir -p "$DATA/training/run-1"
: > "$DATA/training/run-1/history.jsonl"
echo "$DATA/training/run-1/history.jsonl 777:777" > "$WORK/owners"
run "$HERE/rpm/post_install.sh" 1
expect_call "chown -hR blueonyx:blueonyx $DATA/."
expect_out 'history.jsonl'

setup "install over data with only another group deep inside takes the whole tree"
echo "$DATA/dataset/frame.labels.json 990:777" > "$WORK/owners"
run "$HERE/debian/postinst" configure ""
expect_call "chown -hR blueonyx:blueonyx $DATA/."

setup "a chown that fails warns and the install goes on"
echo "$DATA 777:777" > "$WORK/owners"
stub chown 'case "$1" in -hR) exit 1 ;; esac'
run "$HERE/rpm/post_install.sh" 1
expect_out 'Could not'
expect_call 'systemctl enable blue-onyx-prism'
stub chown 'exit 0'

[ "$failed" -eq 0 ] || exit 1
echo "scriptlet checks passed"
