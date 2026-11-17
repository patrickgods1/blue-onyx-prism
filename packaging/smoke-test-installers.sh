#!/usr/bin/env bash
# Install, check and uninstall this runner's installer(s) in <dir> (CI only: it changes the
# machine and needs admin / passwordless sudo). Usage: packaging/smoke-test-installers.sh <dir>
#
# Windows: the smoke test bounds each program it starts (msiexec, icacls, the CLI); as a last line
# it runs under `timeout` too (SMOKE_TEST_TIMEOUT, default 20m, under the workflow step's 25 min),
# so a hang ends here with packaging/windows/diagnose.ps1's report, in the step log and in
# <dir>/smoke-diagnostics.log (uploaded with the msiexec logs), instead of in the runner's cancel.
set -euo pipefail

DIR="${1:?Usage: $0 <dir>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
one() { # exactly one file matching the glob
  # shellcheck disable=SC2086 # the glob must expand
  set -- $1
  [ $# -eq 1 ] && [ -f "$1" ] || { echo "expected one installer, found: $*" >&2; exit 1; }
  echo "$1"
}

windows() {
  local msi limit rc=0
  msi="$(one "$DIR/*.msi")"
  limit="${SMOKE_TEST_TIMEOUT:-20m}"
  # -NonInteractive: a prompt fails instead of waiting; stdin from /dev/null: nothing reads the
  # runner's.
  local cmd=(pwsh -NoProfile -NonInteractive -File "$ROOT/packaging/windows/smoke-test.ps1" -Msi "$msi")
  # coreutils' timeout (Git Bash has it), not Windows' timeout.exe, which only waits.
  if ! timeout --version >/dev/null 2>&1; then
    "${cmd[@]}" </dev/null
    return
  fi
  timeout --kill-after=30s "$limit" "${cmd[@]}" </dev/null || rc=$?
  # 124: the limit passed (timeout ended it); 137: it had to be killed after --kill-after.
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "SMOKE TEST FAILED: still running after $limit (stopped; anything it started, msiexec say, may still run). The machine's state follows." >&2
    timeout 180s pwsh -NoProfile -NonInteractive -File "$ROOT/packaging/windows/diagnose.ps1" -LogDir "$DIR" \
      </dev/null 2>&1 | tee "$DIR/smoke-diagnostics.log" || true
  fi
  return "$rc"
}

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    windows ;;
  Linux)
    "$ROOT/packaging/linux/smoke-test.sh" "$(one "$DIR/*.deb")" "$(one "$DIR/*.rpm")" ;;
  Darwin)
    "$ROOT/packaging/macos/smoke-test.sh" "$(one "$DIR/*.pkg")" ;;
  *) echo "unsupported OS $(uname -s)" >&2; exit 2 ;;
esac
