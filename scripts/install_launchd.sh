#!/usr/bin/env bash
# Install Blue Onyx Prism as a launchd daemon on macOS.
# Usage: sudo scripts/install_launchd.sh [/usr/local/blue-onyx-prism]
#
# Layout: the program dir (binary, openvino/, onnxruntime/, scripts) stays owned by root; the
# daemon runs as the role account _blueonyx, which owns only the data root
# /usr/local/var/blue-onyx-prism (config, models, cache, downloads, benchmark.json, tools/) and
# the log dir /var/log/blue-onyx-prism.
set -euo pipefail

INSTALL_DIR="${1:-/usr/local/blue-onyx-prism}"
DATA_DIR=/usr/local/var/blue-onyx-prism
LOG_DIR=/var/log/blue-onyx-prism
CONFIG="$DATA_DIR/blue_onyx_prism_config.json"
SVC=_blueonyx
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SRC="$HERE/deploy/com.blueonyx.prism.plist"
PLIST_DST="/Library/LaunchDaemons/com.blueonyx.prism.plist"
NEWSYSLOG_CONF=/etc/newsyslog.d/blue-onyx-prism.conf

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root (sudo)." >&2
  exit 1
fi
if [ ! -x "$INSTALL_DIR/blue-onyx-prism" ]; then
  echo "Copy the release bundle (binary, openvino/, models/) to $INSTALL_DIR first." >&2
  exit 1
fi
case "$INSTALL_DIR" in
  *'#'* | *'"'* | *'\'* | *' '*) echo "The install path must not contain '#', '\"', '\\' or spaces." >&2; exit 1 ;;
esac

# Role account _blueonyx (hidden, no shell, no password) with a group of the same name; one free
# id below 500 serves as both UID and GID.
# (awk reads its whole input: `grep -q` would exit early and fail the pipe under pipefail.)
uid_taken() { dscl . -list /Users UniqueID | awk -v id="$1" '$2 == id { f = 1 } END { exit !f }'; }
gid_taken() { dscl . -list /Groups PrimaryGroupID | awk -v id="$1" '$2 == id { f = 1 } END { exit !f }'; }
if dscl . -read "/Groups/$SVC" PrimaryGroupID >/dev/null 2>&1; then
  gid="$(dscl . -read "/Groups/$SVC" PrimaryGroupID | awk '{print $2}')"
else
  gid=""
fi
if ! dscl . -read "/Users/$SVC" >/dev/null 2>&1; then
  uid=""
  if [ -n "$gid" ] && ! uid_taken "$gid"; then
    uid="$gid"
  else
    for i in $(seq 499 -1 300); do
      if ! uid_taken "$i" && ! gid_taken "$i"; then uid="$i"; break; fi
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

# Program files: root-owned, not writable by the daemon.
chown -R root:wheel "$INSTALL_DIR"
chmod -R go-w "$INSTALL_DIR"

# Data root and log dir: the only places the daemon writes.
install -d -o "$SVC" -g "$SVC" -m 0750 "$DATA_DIR" "$DATA_DIR/models" "$DATA_DIR/cache" "$LOG_DIR"
if [ -d "$INSTALL_DIR/models" ]; then
  # Bundled / previously installed models; files already in the data root win.
  cp -Rn "$INSTALL_DIR/models/." "$DATA_DIR/models/"
  chown -R "$SVC:$SVC" "$DATA_DIR/models"
fi
touch "$LOG_DIR/stdout.log"
chown "$SVC:$SVC" "$LOG_DIR/stdout.log"
chmod 0640 "$LOG_DIR/stdout.log"

# stdout/stderr rotation: keep 3 old copies, rotate at 1 MB, no signal (launchd holds the file).
cat > "$NEWSYSLOG_CONF" <<CONF
# logfilename                      [owner:group]       mode count size(KB) when  flags
$LOG_DIR/stdout.log    $SVC:$SVC   640  3     1024     *     N
CONF
chmod 644 "$NEWSYSLOG_CONF"

# Settings that point the daemon at the data root, the log dir and the root-owned runtimes (a
# runtime named by openvino_dir / onnxruntime_dir is user-managed: never downloaded or replaced).
keys="  \"download_dir\": \"$DATA_DIR\",
  \"cache_dir\": \"$DATA_DIR/cache\",
  \"log_path\": \"$LOG_DIR\","
if [ -d "$INSTALL_DIR/openvino" ]; then
  keys="$keys
  \"openvino_dir\": \"$INSTALL_DIR/openvino\","
fi
ort=""
if [ -f "$INSTALL_DIR/onnxruntime/active.txt" ]; then
  flavor="$(tr -d '[:space:]' < "$INSTALL_DIR/onnxruntime/active.txt")"
  [ -n "$flavor" ] && [ -d "$INSTALL_DIR/onnxruntime/$flavor" ] && ort="$INSTALL_DIR/onnxruntime/$flavor"
elif ls "$INSTALL_DIR"/onnxruntime/libonnxruntime*.dylib >/dev/null 2>&1; then
  ort="$INSTALL_DIR/onnxruntime"
fi
if [ -n "$ort" ]; then
  keys="$keys
  \"onnxruntime_dir\": \"$ort\","
fi

if [ ! -f "$CONFIG" ]; then
  src=""
  [ -f "$INSTALL_DIR/blue_onyx_prism_config.json" ] && src="$INSTALL_DIR/blue_onyx_prism_config.json"
  if [ -n "$src" ] && grep -q '"download_dir"' "$src"; then
    cp "$src" "$CONFIG"
    echo "Copied $src to $CONFIG; it already sets download_dir: check that it points at $DATA_DIR."
  elif [ -n "$src" ] && [ "$(head -n 1 "$src")" = "{" ]; then
    # Pretty JSON as the service writes it: insert the keys after the opening brace.
    KEYS="$keys" awk 'NR == 1 { print; print ENVIRON["KEYS"]; next } { print }' "$src" > "$CONFIG"
    echo "Moved the config to $CONFIG (download_dir, cache_dir, log_path and runtime dirs added)."
  elif [ -n "$src" ]; then
    cp "$src" "$CONFIG"
    echo "Copied $src to $CONFIG. Add these keys to it by hand:" >&2
    echo "$keys" >&2
  else
    printf '{\n%s\n  "models_dir": "models"\n}\n' "$keys" > "$CONFIG"
  fi
fi
chown "$SVC:$SVC" "$CONFIG"
chmod 0640 "$CONFIG"

# Pre-rename daemon (Blue Onyx OpenVINO): unload and remove it so the two don't fight over the port.
OLD_PLIST="/Library/LaunchDaemons/com.blueonyx.openvino.plist"
if [ -f "$OLD_PLIST" ]; then
  launchctl bootout system "$OLD_PLIST" 2>/dev/null || true
  rm -f "$OLD_PLIST"
fi

sed "s#/usr/local/blue-onyx-prism#$INSTALL_DIR#g" "$PLIST_SRC" > "$PLIST_DST"
chown root:wheel "$PLIST_DST"
chmod 644 "$PLIST_DST"
launchctl bootout system "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap system "$PLIST_DST"
launchctl print system/com.blueonyx.prism | head -20 || true
echo "Config: $CONFIG (data: $DATA_DIR)"
echo "Logs:   $LOG_DIR/blue_onyx_prism.log.<date> (stdout/stderr: $LOG_DIR/stdout.log)"
