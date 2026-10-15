#!/usr/bin/env bash
# Run a command up to <attempts> times, <delay> seconds apart (CI download steps: the CLI fails
# fast on a network error by design, only the service retries, so one dropped connection to a
# download host would otherwise fail the whole job).
# Usage: scripts/ci/retry.sh <attempts> <delay-seconds> <command> [args...]
set -uo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <attempts> <delay-seconds> <command> [args...]" >&2
  exit 2
fi
attempts="$1"
delay="$2"
shift 2
rc=1
for i in $(seq 1 "$attempts"); do
  "$@" && exit 0
  rc=$?
  [ "$i" -lt "$attempts" ] || break
  echo "::warning::attempt $i of $attempts failed (exit $rc): $*; retrying in $delay s" >&2
  sleep "$delay"
done
echo "::error::failed $attempts times (last exit $rc): $*" >&2
exit "$rc"
