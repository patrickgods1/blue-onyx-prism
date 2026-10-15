#!/bin/sh
# %postun: $1 = 0 on erase, 1+ on upgrade (then nothing happens: the new version keeps the data).
# RPM has no purge, so an erase leaves nothing behind: what the program created in
# /opt/blue-onyx-prism, the data root /var/lib/blue-onyx-prism (config, models, cache, downloads)
# and the service user, as uninstall_systemd.sh --purge does.
if [ "${1:-0}" -eq 0 ]; then
  if [ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload || true
    systemctl reset-failed blue-onyx-prism 2>/dev/null || true
  fi
  rm -rf /opt/blue-onyx-prism /var/lib/blue-onyx-prism
  if id -u blueonyx >/dev/null 2>&1; then
    userdel blueonyx || true
  fi
  if getent group blueonyx >/dev/null 2>&1; then
    groupdel blueonyx 2>/dev/null || true
  fi
fi
exit 0
