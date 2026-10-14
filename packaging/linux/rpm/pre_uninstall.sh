#!/bin/sh
# %preun: $1 = 0 on erase, 1+ on upgrade (the new package's %post try-restarts).
if [ "${1:-0}" -eq 0 ] && [ -x /opt/blue-onyx-prism/scripts/pkg_scriptlet.sh ]; then
  /opt/blue-onyx-prism/scripts/pkg_scriptlet.sh pre-remove
fi
exit 0
