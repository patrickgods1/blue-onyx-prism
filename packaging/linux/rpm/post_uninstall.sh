#!/bin/sh
# %postun: $1 = 0 on erase, 1+ on upgrade (then nothing happens: the new version keeps the data).
# An erase removes the program, with what the program created in /opt/blue-onyx-prism. It keeps
# the data root /var/lib/blue-onyx-prism (config, models, dataset, reviewed labels, training
# runs) and the service user: that folder can hold work no download brings back, and a reinstall
# picks it up. RPM has no purge, so the erase prints how to delete them.
if [ "${1:-0}" -eq 0 ]; then
  if [ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload || true
    systemctl reset-failed blue-onyx-prism 2>/dev/null || true
  fi
  rm -rf /opt/blue-onyx-prism
  if [ -d /var/lib/blue-onyx-prism ]; then
    echo "Kept /var/lib/blue-onyx-prism (config, models, dataset, reviewed labels, training runs) and the user blueonyx for a reinstall. To delete them too:"
    echo "  sudo rm -rf /var/lib/blue-onyx-prism && sudo userdel blueonyx"
  fi
fi
exit 0
