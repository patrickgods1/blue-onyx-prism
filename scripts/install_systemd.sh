#!/usr/bin/env bash
# Install Blue Onyx Prism as a systemd service on Linux.
# Usage: sudo scripts/install_systemd.sh [/opt/blue-onyx-prism]
#
# Layout: the program dir (binary, openvino/, onnxruntime/, scripts) stays owned by root and
# read-only to the service; the service user `blueonyx` owns only the data root
# /var/lib/blue-onyx-prism (config, models, cache, downloads, benchmark.json, tools/).
set -euo pipefail

INSTALL_DIR="${1:-/opt/blue-onyx-prism}"
DATA_DIR=/var/lib/blue-onyx-prism
CONFIG="$DATA_DIR/blue_onyx_prism_config.json"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$HERE/deploy/blue-onyx-prism.service"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root (sudo)." >&2
  exit 1
fi
if [ ! -x "$INSTALL_DIR/blue-onyx-prism" ]; then
  echo "Copy the release bundle (binary, openvino/, models/) to $INSTALL_DIR first." >&2
  exit 1
fi
case "$INSTALL_DIR$DATA_DIR" in
  *'#'* | *'"'* | *'\'*) echo "Paths must not contain '#', '\"' or '\\'." >&2; exit 1 ;;
esac

if ! id -u blueonyx >/dev/null 2>&1; then
  useradd --system --home-dir "$DATA_DIR" --no-create-home --shell /usr/sbin/nologin blueonyx
else
  usermod --home "$DATA_DIR" blueonyx
fi
for g in render video; do
  if getent group "$g" >/dev/null; then usermod -aG "$g" blueonyx; fi
done

# Program files: root-owned, not writable by the service (an earlier install chowned them).
chown -R root:root "$INSTALL_DIR"
chmod -R go-w "$INSTALL_DIR"

# Data root: the only place the service writes.
install -d -o blueonyx -g blueonyx -m 0750 "$DATA_DIR" "$DATA_DIR/models" "$DATA_DIR/cache"
if [ -d "$INSTALL_DIR/models" ]; then
  # Bundled / previously installed models; files already in the data root win.
  cp -Rn "$INSTALL_DIR/models/." "$DATA_DIR/models/"
  chown -R blueonyx:blueonyx "$DATA_DIR/models"
fi

# Settings that point the service at the data root and at the root-owned runtimes (a runtime
# named by openvino_dir / onnxruntime_dir is user-managed: never downloaded or replaced).
keys="  \"download_dir\": \"$DATA_DIR\",
  \"cache_dir\": \"$DATA_DIR/cache\","
if [ -d "$INSTALL_DIR/openvino" ]; then
  keys="$keys
  \"openvino_dir\": \"$INSTALL_DIR/openvino\","
fi
ort=""
if [ -f "$INSTALL_DIR/onnxruntime/active.txt" ]; then
  flavor="$(tr -d '[:space:]' < "$INSTALL_DIR/onnxruntime/active.txt")"
  [ -n "$flavor" ] && [ -d "$INSTALL_DIR/onnxruntime/$flavor" ] && ort="$INSTALL_DIR/onnxruntime/$flavor"
elif ls "$INSTALL_DIR"/onnxruntime/libonnxruntime.so* >/dev/null 2>&1; then
  ort="$INSTALL_DIR/onnxruntime"
fi
if [ -n "$ort" ]; then
  keys="$keys
  \"onnxruntime_dir\": \"$ort\","
fi

if [ ! -f "$CONFIG" ]; then
  if [ -f "$INSTALL_DIR/blue_onyx_prism_config.json" ]; then
    src="$INSTALL_DIR/blue_onyx_prism_config.json"
  else
    src=""
  fi
  if [ -n "$src" ] && grep -q '"download_dir"' "$src"; then
    cp "$src" "$CONFIG"
    echo "Copied $src to $CONFIG; it already sets download_dir: check that it points at $DATA_DIR."
  elif [ -n "$src" ] && [ "$(head -n 1 "$src")" = "{" ]; then
    # Pretty JSON as the service writes it: insert the keys after the opening brace.
    KEYS="$keys" awk 'NR == 1 { print; print ENVIRON["KEYS"]; next } { print }' "$src" > "$CONFIG"
    echo "Moved the config to $CONFIG (download_dir, cache_dir and runtime dirs added)."
  elif [ -n "$src" ]; then
    cp "$src" "$CONFIG"
    echo "Copied $src to $CONFIG. Add these keys to it by hand:" >&2
    echo "$keys" >&2
  else
    printf '{\n%s\n  "models_dir": "models"\n}\n' "$keys" > "$CONFIG"
  fi
fi
chown blueonyx:blueonyx "$CONFIG"
chmod 0640 "$CONFIG"

# Pre-rename unit (Blue Onyx OpenVINO): stop and remove it so the two don't fight over the port.
if [ -f /etc/systemd/system/blue-onyx-openvino.service ]; then
  systemctl disable --now blue-onyx-openvino 2>/dev/null || true
  rm -f /etc/systemd/system/blue-onyx-openvino.service
fi

sed "s#/opt/blue-onyx-prism#$INSTALL_DIR#g" "$UNIT_SRC" > /etc/systemd/system/blue-onyx-prism.service
systemctl daemon-reload
systemctl enable blue-onyx-prism
systemctl restart blue-onyx-prism
systemctl --no-pager status blue-onyx-prism | head -20 || true
echo "Config: $CONFIG (data: $DATA_DIR)"
echo "Logs:   journalctl -u blue-onyx-prism -f"
