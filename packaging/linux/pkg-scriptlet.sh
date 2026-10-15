#!/bin/sh
# Package scriptlet logic shared by the .deb and .rpm maintainer scripts. Installed as
# /opt/blue-onyx-prism/scripts/pkg_scriptlet.sh; called by packaging/linux/{debian,rpm}/*.
# Usage: pkg_scriptlet.sh post-install | post-upgrade | pre-remove
# (purge / daemon-reload after removal run from the wrappers: the program dir is gone by then.)
set -eu

INSTALL_DIR=/opt/blue-onyx-prism
DATA_DIR=/var/lib/blue-onyx-prism
CONFIG="$DATA_DIR/blue_onyx_prism_config.json"
UNIT=blue-onyx-prism

# shellcheck source=../../scripts/systemd_common.sh
. "$INSTALL_DIR/scripts/systemd_common.sh"

# systemd running as PID 1 (not a chroot / container build); otherwise skip systemctl.
have_systemd() { [ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1; }

case "${1:-}" in
  post-install | post-upgrade)
    if command -v systemd-sysusers >/dev/null 2>&1 \
      && [ -f /usr/lib/sysusers.d/blue-onyx-prism.conf ]; then
      systemd-sysusers /usr/lib/sysusers.d/blue-onyx-prism.conf || true
    fi
    # Fallback (and idempotent top-up): useradd when sysusers is unavailable or failed.
    bop_ensure_user "$DATA_DIR"

    install -d -o "$BOP_USER" -g "$BOP_USER" -m 0750 "$DATA_DIR" "$DATA_DIR/models" "$DATA_DIR/cache"
    bop_seed_config "$INSTALL_DIR" "$DATA_DIR" "$CONFIG" 0
    bop_remove_legacy_unit

    if have_systemd; then
      systemctl daemon-reload || true
      if [ "$1" = post-install ]; then
        systemctl enable "$UNIT" || true
        systemctl start "$UNIT" || echo "Could not start $UNIT: see journalctl -u $UNIT" >&2
      else
        systemctl try-restart "$UNIT" || true
      fi
    fi
    echo "Config: $CONFIG (data: $DATA_DIR)"
    echo "Logs:   journalctl -u $UNIT -f"
    ;;
  pre-remove)
    if have_systemd; then
      systemctl stop "$UNIT" || true
      systemctl disable "$UNIT" || true
    fi
    ;;
  *)
    echo "Usage: $0 post-install | post-upgrade | pre-remove" >&2
    exit 2
    ;;
esac
