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

# An uninstall keeps the data root, so an install can meet one written by an earlier blueonyx
# with another uid (the user was deleted by hand, or the folder came from a backup). The check
# must run before `install -d` below: that corrects the top folders only, which would also hide
# the mismatch from systemd's StateDirectory=, which looks at the top folder. -h and the `/.`:
# links are never followed (uv's Python environment has some), a data root that is itself a link
# to a larger disk is. A failure warns: the data is intact and the service may still run.
own_data_root() {
  [ -d "$DATA_DIR" ] || return 0
  want="$(id -u "$BOP_USER"):$(id -g "$BOP_USER")"
  have="$(stat -Lc %u:%g "$DATA_DIR")"
  [ "$have" != "$want" ] || return 0
  echo "$DATA_DIR belongs to $have, not to $BOP_USER ($want): changing its owner, which can take a while." >&2
  chown -hR "$BOP_USER:$BOP_USER" "$DATA_DIR/." \
    || echo "Could not change the owner of everything in $DATA_DIR, so the service may fail to read it (is it a network share?). Fix it with: sudo chown -hR $BOP_USER:$BOP_USER $DATA_DIR/." >&2
}

case "${1:-}" in
  post-install | post-upgrade)
    if command -v systemd-sysusers >/dev/null 2>&1 \
      && [ -f /usr/lib/sysusers.d/blue-onyx-prism.conf ]; then
      systemd-sysusers /usr/lib/sysusers.d/blue-onyx-prism.conf || true
    fi
    # Fallback (and idempotent top-up): useradd when sysusers is unavailable or failed.
    bop_ensure_user "$DATA_DIR"
    own_data_root

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
