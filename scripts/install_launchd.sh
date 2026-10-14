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
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SRC="$HERE/deploy/com.blueonyx.prism.plist"
# shellcheck source=scripts/launchd_common.sh
. "$HERE/scripts/launchd_common.sh"

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

bop_create_account

# Program files: root-owned, not writable by the daemon.
chown -R root:wheel "$INSTALL_DIR"
chmod -R go-w "$INSTALL_DIR"

bop_create_dirs
if [ -d "$INSTALL_DIR/models" ]; then
  # Bundled / previously installed models; files already in the data root win.
  cp -Rn "$INSTALL_DIR/models/." "$DATA_DIR/models/"
  chown -R "$SVC:$SVC" "$DATA_DIR/models"
fi

bop_install_newsyslog

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
bop_install_plist "$PLIST_DST"
bop_bootstrap
launchctl print system/com.blueonyx.prism | head -20 || true
echo "Config: $CONFIG (data: $DATA_DIR)"
echo "Logs:   $LOG_DIR/blue_onyx_prism.log.<date> (stdout/stderr: $LOG_DIR/stdout.log)"
