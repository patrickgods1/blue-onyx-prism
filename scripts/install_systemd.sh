#!/usr/bin/env bash
# Install Blue Onyx Prism as a systemd service on Linux.
# Usage: sudo scripts/install_systemd.sh [/opt/blue-onyx-prism]
#
# Layout: the program dir (binary, openvino/, onnxruntime/, scripts) stays owned by root and
# read-only to the service; the service user `blueonyx` owns only the data root
# /var/lib/blue-onyx-prism (config, models, cache, downloads, benchmark.json, tools/).
# Shared steps live in scripts/systemd_common.sh (also used by the .deb / .rpm packages).
set -euo pipefail

INSTALL_DIR="${1:-/opt/blue-onyx-prism}"
DATA_DIR=/var/lib/blue-onyx-prism
CONFIG="$DATA_DIR/blue_onyx_prism_config.json"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="$HERE/deploy/blue-onyx-prism.service"
# shellcheck source=systemd_common.sh
. "$HERE/scripts/systemd_common.sh"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root (sudo)." >&2
  exit 1
fi
if [ ! -x "$INSTALL_DIR/blue-onyx-prism" ]; then
  echo "Copy the release bundle (binary, openvino/, models/) to $INSTALL_DIR first." >&2
  exit 1
fi
bop_check_paths "$INSTALL_DIR$DATA_DIR" || exit 1

bop_ensure_user "$DATA_DIR"

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

# Settings that point the service at the data root and at the root-owned runtimes.
bop_seed_config "$INSTALL_DIR" "$DATA_DIR" "$CONFIG" 1

# Pre-rename unit (Blue Onyx OpenVINO): stop and remove it so the two don't fight over the port.
bop_remove_legacy_unit

bop_install_unit "$UNIT_SRC" /etc/systemd/system/blue-onyx-prism.service "$INSTALL_DIR"
systemctl daemon-reload
systemctl enable blue-onyx-prism
systemctl restart blue-onyx-prism
systemctl --no-pager status blue-onyx-prism | head -20 || true
echo "Config: $CONFIG (data: $DATA_DIR)"
echo "Logs:   journalctl -u blue-onyx-prism -f"
