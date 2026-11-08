#!/usr/bin/env bash
# Fail when a Linux binary needs a newer glibc than the one the README promises
# (packaging/linux/glibc-floor). The release builds on ubuntu-22.04 (glibc 2.35), so a new
# dependency or a toolchain change can pull in a GLIBC_2.35 symbol, and the rpm would then stop
# starting on RHEL 9 (2.34) with "version `GLIBC_2.35' not found".
# Usage: packaging/linux/check-glibc.sh [--floor-file F] BINARY...
# Needs `objdump` (binutils; LLVM's works too).
set -euo pipefail

FLOOR_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/glibc-floor"
if [ "${1:-}" = --floor-file ]; then
  FLOOR_FILE="${2:?--floor-file needs a path}"
  shift 2
fi
if [ $# -eq 0 ]; then
  echo "Usage: $0 [--floor-file F] BINARY..." >&2
  exit 2
fi
FLOOR="$(grep -Ev '^[[:space:]]*(#|$)' "$FLOOR_FILE" | head -n 1 | tr -d '[:space:]')"
case "$FLOOR" in
  [0-9]*.[0-9]*) ;;
  *) echo "$FLOOR_FILE holds no glibc version (got '$FLOOR')." >&2; exit 2 ;;
esac

# newer A B: true when version A > version B (numeric, dot by dot).
newer() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    na = split(a, x, "."); nb = split(b, y, "."); n = na > nb ? na : nb
    for (i = 1; i <= n; i++) { if (x[i] + 0 > y[i] + 0) exit 0; if (x[i] + 0 < y[i] + 0) exit 1 }
    exit 1
  }'
}

bad=0
for bin in "$@"; do
  [ -f "$bin" ] || { echo "$bin: no such file" >&2; bad=1; continue; }
  # Dynamic symbol table: the versions the loader checks. GLIBC_PRIVATE is not a version.
  versions="$(objdump -T "$bin" | grep -Eo 'GLIBC_[0-9]+(\.[0-9]+)+' | sed 's/^GLIBC_//' | sort -u)" \
    || { echo "$bin: objdump -T found no GLIBC_ symbol (not a dynamically linked glibc binary?)" >&2; bad=1; continue; }
  max=""
  for v in $versions; do
    if [ -z "$max" ] || newer "$v" "$max"; then max="$v"; fi
  done
  if newer "$max" "$FLOOR"; then
    # The symbols behind it: name@version for every version above the floor.
    syms="$(objdump -T "$bin" | awk -v floor="$FLOOR" '
      function gt(a, b,   x, y, na, nb, n, i) {
        na = split(a, x, "."); nb = split(b, y, "."); n = na > nb ? na : nb
        for (i = 1; i <= n; i++) { if (x[i] + 0 > y[i] + 0) return 1; if (x[i] + 0 < y[i] + 0) return 0 }
        return 0
      }
      match($0, /GLIBC_[0-9]+(\.[0-9]+)+/) {
        v = substr($0, RSTART + 6, RLENGTH - 6)
        if (gt(v, floor)) print "  " $NF " (GLIBC_" v ")"
      }' | sort -u | head -n 20)"
    echo "::error::$bin needs glibc $max, newer than the $FLOOR the README promises ($FLOOR_FILE). Symbols:" >&2
    echo "$syms" >&2
    echo "Either keep the build on an older glibc (the runner image, or a dependency that pulled the symbol in), or raise the claim: $FLOOR_FILE together with the README's installer table." >&2
    bad=1
  else
    echo "$bin: needs glibc $max (floor $FLOOR): OK"
  fi
done
exit "$bad"
