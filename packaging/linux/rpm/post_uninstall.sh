#!/bin/sh
# %postun: $1 = 0 on erase. RPM has no purge: the data root (/var/lib/blue-onyx-prism) and the
# user stay, as uninstall_systemd.sh does without --purge.
if [ "${1:-0}" -eq 0 ]; then
  if [ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload || true
    systemctl reset-failed blue-onyx-prism 2>/dev/null || true
  fi
fi
exit 0
