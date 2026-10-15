#!/usr/bin/env bash
# Shared helpers for the Blue Onyx Prism macOS launchd install: sourced by scripts/install_launchd.sh
# (zip flow) and by the .pkg postinstall (packaging/macos). Source it, do not run it. It sets no
# shell options; the caller is expected to run with `set -euo pipefail`. Functions print errors to
# stderr and `exit 1` (they are meant for install scripts).

DATA_DIR="${DATA_DIR:-/usr/local/var/blue-onyx-prism}"
LOG_DIR="${LOG_DIR:-/var/log/blue-onyx-prism}"
CONFIG="${CONFIG:-$DATA_DIR/blue_onyx_prism_config.json}"
SVC="${SVC:-_blueonyx}"
PLIST_DST="${PLIST_DST:-/Library/LaunchDaemons/com.blueonyx.prism.plist}"
NEWSYSLOG_CONF="${NEWSYSLOG_CONF:-/etc/newsyslog.d/blue-onyx-prism.conf}"
TRAY_LABEL="${TRAY_LABEL:-com.blueonyx.prism.tray}"
TRAY_PLIST="${TRAY_PLIST:-/Library/LaunchAgents/$TRAY_LABEL.plist}"

# Role account _blueonyx (hidden, no shell, no password) with a group of the same name; one free
# id below 500 serves as both UID and GID.
# (awk reads its whole input: `grep -q` would exit early and fail the pipe under pipefail.)
bop_uid_taken() { dscl . -list /Users UniqueID | awk -v id="$1" '$2 == id { f = 1 } END { exit !f }'; }
bop_gid_taken() { dscl . -list /Groups PrimaryGroupID | awk -v id="$1" '$2 == id { f = 1 } END { exit !f }'; }

bop_create_account() {
  local gid uid i
  if dscl . -read "/Groups/$SVC" PrimaryGroupID >/dev/null 2>&1; then
    gid="$(dscl . -read "/Groups/$SVC" PrimaryGroupID | awk '{print $2}')"
  else
    gid=""
  fi
  if ! dscl . -read "/Users/$SVC" >/dev/null 2>&1; then
    uid=""
    if [ -n "$gid" ] && ! bop_uid_taken "$gid"; then
      uid="$gid"
    else
      for i in $(seq 499 -1 300); do
        if ! bop_uid_taken "$i" && ! bop_gid_taken "$i"; then uid="$i"; break; fi
      done
    fi
    if [ -z "$uid" ]; then
      echo "No free UID/GID below 500 for $SVC." >&2
      exit 1
    fi
    if [ -z "$gid" ]; then
      gid="$uid"
      dscl . -create "/Groups/$SVC"
      dscl . -create "/Groups/$SVC" PrimaryGroupID "$gid"
      dscl . -create "/Groups/$SVC" RealName "Blue Onyx Prism"
      dscl . -create "/Groups/$SVC" Password '*'
    fi
    dscl . -create "/Users/$SVC"
    dscl . -create "/Users/$SVC" UniqueID "$uid"
    dscl . -create "/Users/$SVC" PrimaryGroupID "$gid"
    dscl . -create "/Users/$SVC" UserShell /usr/bin/false
    dscl . -create "/Users/$SVC" NFSHomeDirectory "$DATA_DIR"
    dscl . -create "/Users/$SVC" RealName "Blue Onyx Prism"
    dscl . -create "/Users/$SVC" Password '*'
    dscl . -create "/Users/$SVC" IsHidden 1
    echo "Created role account $SVC (uid $uid, gid $gid)."
  elif [ -z "$gid" ]; then
    echo "User $SVC exists but group $SVC does not; create the group or remove the user." >&2
    exit 1
  fi
}

# Data root and log dir: the only places the daemon writes.
bop_create_dirs() {
  install -d -o "$SVC" -g "$SVC" -m 0750 "$DATA_DIR" "$DATA_DIR/models" "$DATA_DIR/cache" "$LOG_DIR"
  touch "$LOG_DIR/stdout.log"
  chown "$SVC:$SVC" "$LOG_DIR/stdout.log"
  chmod 0640 "$LOG_DIR/stdout.log"
}

# stdout/stderr rotation: keep 3 old copies, rotate at 1 MB, no signal (launchd holds the file).
bop_install_newsyslog() {
  mkdir -p "$(dirname "$NEWSYSLOG_CONF")"
  cat > "$NEWSYSLOG_CONF" <<CONF
# logfilename                      [owner:group]       mode count size(KB) when  flags
$LOG_DIR/stdout.log    $SVC:$SVC   640  3     1024     *     N
CONF
  chmod 644 "$NEWSYSLOG_CONF"
}

# Settings that point the daemon at the data root and the log dir. Prints the JSON members (each
# line ending in a comma). An optional extra block (already formatted, comma-terminated) can be
# passed as $1, e.g. the runtime dirs of the zip flow.
bop_config_keys() {
  printf '  "download_dir": "%s",\n  "cache_dir": "%s/cache",\n  "log_path": "%s",' "$DATA_DIR" "$DATA_DIR" "$LOG_DIR"
  if [ -n "${1:-}" ]; then printf '\n%s' "$1"; fi
  printf '\n'
}

# Write a fresh config with the keys above; never touches an existing file.
bop_seed_config() {
  if [ ! -f "$CONFIG" ]; then
    printf '{\n%s\n  "models_dir": "models"\n}\n' "$(bop_config_keys "${1:-}")" > "$CONFIG"
  fi
  chown "$SVC:$SVC" "$CONFIG"
  chmod 0640 "$CONFIG"
}

# Install the plist ($1) at $PLIST_DST (unless it already is there), root:wheel 644.
bop_install_plist() {
  local src="$1"
  if [ "$src" != "$PLIST_DST" ]; then
    cp "$src" "$PLIST_DST"
  fi
  chown root:wheel "$PLIST_DST"
  chmod 644 "$PLIST_DST"
}

# (Re)load the daemon from $PLIST_DST.
bop_bootstrap() {
  launchctl bootout system "$PLIST_DST" 2>/dev/null || true
  # An upgrade boots out the running daemon: wait until it has exited (ExitTimeOut 30 s), or the
  # bootstrap below fails with "Input/output error".
  local i
  for i in $(seq 1 40); do
    launchctl print system/com.blueonyx.prism >/dev/null 2>&1 || break
    sleep 1
  done
  launchctl enable system/com.blueonyx.prism 2>/dev/null || true
  if ! launchctl bootstrap system "$PLIST_DST"; then
    echo "launchctl bootstrap failed for $PLIST_DST; see 'launchctl print system/com.blueonyx.prism' and /var/log/system.log." >&2
    exit 1
  fi
}

# uid of the user logged in at the console; empty when nobody is (login window, CI, ssh only).
bop_console_uid() {
  local uid
  uid="$(stat -f %u /dev/console 2>/dev/null || true)"
  if [ -n "$uid" ] && [ "$uid" != 0 ]; then printf '%s' "$uid"; fi
}

# Stop the menu bar app of the console user and any other running copy. Never fails.
bop_tray_stop() {
  local uid
  uid="$(bop_console_uid)"
  if [ -n "$uid" ]; then
    launchctl bootout "gui/$uid/$TRAY_LABEL" >/dev/null 2>&1 || true
  fi
  pkill -x blue-onyx-prism-tray >/dev/null 2>&1 || true
}

# (Re)start the menu bar app for the console user right away; other users get it at their next
# login from the LaunchAgent. Never fails: a machine without a GUI session simply skips it.
bop_tray_start() {
  local uid
  uid="$(bop_console_uid)"
  [ -n "$uid" ] || return 0
  [ -f "$TRAY_PLIST" ] || return 0
  launchctl print "gui/$uid" >/dev/null 2>&1 || return 0
  launchctl bootout "gui/$uid/$TRAY_LABEL" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$uid" "$TRAY_PLIST" >/dev/null 2>&1 || true
  return 0
}
