#!/usr/bin/env bash
# Install, check and uninstall this runner's installer(s) in <dir> (CI only: it changes the
# machine and needs admin / passwordless sudo). Usage: packaging/smoke-test-installers.sh <dir>
set -euo pipefail

DIR="${1:?Usage: $0 <dir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
one() { # exactly one file matching the glob
  # shellcheck disable=SC2086 # the glob must expand
  set -- $1
  [ $# -eq 1 ] && [ -f "$1" ] || { echo "expected one installer, found: $*" >&2; exit 1; }
  echo "$1"
}
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    pwsh -NoProfile -File "$ROOT/packaging/windows/smoke-test.ps1" -Msi "$(one "$DIR/*.msi")" ;;
  Linux)
    "$ROOT/packaging/linux/smoke-test.sh" "$(one "$DIR/*.deb")" "$(one "$DIR/*.rpm")" ;;
  Darwin)
    "$ROOT/packaging/macos/smoke-test.sh" "$(one "$DIR/*.pkg")" ;;
  *) echo "unsupported OS $(uname -s)" >&2; exit 2 ;;
esac
