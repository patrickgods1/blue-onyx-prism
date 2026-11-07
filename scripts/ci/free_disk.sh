#!/usr/bin/env bash
# The hosted Ubuntu runner reached 793 MB free during this job, which failed tests for no reason
# of their own. Deletes preinstalled toolchains this repository never uses, and logs what each
# gave back.
# Do not add /opt/hostedtoolcache as a whole: actions/setup-python installs there.
set -euo pipefail

free_gb() { df -BG --output=avail / | tail -n 1 | tr -d ' G'; }

df -h /
for dir in /usr/local/lib/android /usr/share/dotnet /opt/ghc /usr/local/.ghcup /opt/hostedtoolcache/CodeQL; do
  [ -d "$dir" ] || { echo "$dir: not on this image"; continue; }
  before=$(free_gb)
  sudo rm -rf "$dir"
  echo "$dir: removed, $((  $(free_gb) - before )) GB freed"
done
df -h /
