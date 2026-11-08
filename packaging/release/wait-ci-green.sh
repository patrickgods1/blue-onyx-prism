#!/usr/bin/env bash
# release.yml, job ci-green: refuse to build a release unless CI passed on the commit. Asks GitHub
# for the commit's check runs and workflow runs and lets ci_green.py decide; while CI still runs
# (a tag pushed right after the merge), asks again every POLL_SECONDS.
# Env: SHA (the commit), GITHUB_REPOSITORY (owner/repo), GH_TOKEN (checks: read, actions: read).
# Optional: WAIT_RUNNING_MIN (80: a cold CI run takes about 40 min), WAIT_MISSING_MIN (10: how
# long a CI run that has not appeared yet is waited for), POLL_SECONDS (60).
set -euo pipefail

: "${SHA:?SHA is the commit to check}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is owner/repo}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_RUNNING_MIN="${WAIT_RUNNING_MIN:-80}"
WAIT_MISSING_MIN="${WAIT_MISSING_MIN:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
start=$SECONDS

while :; do
  # The API itself failing (rate limit, outage) is no answer about CI: say so and stop.
  if ! gh api --paginate "repos/$GITHUB_REPOSITORY/commits/$SHA/check-runs?per_page=100" > "$WORK/check-runs.json" \
    || ! gh api --paginate "repos/$GITHUB_REPOSITORY/actions/runs?head_sha=$SHA&per_page=100" > "$WORK/workflow-runs.json"; then
    echo "::error::Could not read the check runs of $SHA from the GitHub API (see above); the release is not built without them. Re-run this workflow." >&2
    exit 1
  fi
  rc=0
  python3 -I "$HERE/ci_green.py" --sha "$SHA" --check-runs "$WORK/check-runs.json" \
    --workflow-runs "$WORK/workflow-runs.json" --self .github/workflows/release.yml \
    --require .github/workflows/ci.yml || rc=$?
  waited=$(((SECONDS - start) / 60))
  case "$rc" in
    0) exit 0 ;;
    3) limit="$WAIT_RUNNING_MIN" ;;
    4) limit="$WAIT_MISSING_MIN" ;;
    *) echo "::error::CI is not green on $SHA (the checks are listed above): no release is built." >&2; exit 1 ;;
  esac
  if [ "$waited" -ge "$limit" ]; then
    echo "::error::Waited $waited min for CI on $SHA (limit $limit min): no release is built. Re-run this workflow once CI has finished." >&2
    exit 1
  fi
  echo "Waiting for CI ($waited of $limit min) ..."
  sleep "$POLL_SECONDS"
done
