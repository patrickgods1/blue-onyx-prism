#!/bin/sh
# %post: $1 = number of installed versions after this transaction (1 = install, 2+ = upgrade).
if [ "${1:-1}" -ge 2 ]; then
  /opt/blue-onyx-prism/scripts/pkg_scriptlet.sh post-upgrade
else
  /opt/blue-onyx-prism/scripts/pkg_scriptlet.sh post-install
fi
exit 0
