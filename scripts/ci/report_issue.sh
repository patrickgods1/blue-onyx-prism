#!/usr/bin/env bash
# Open, update or close the one issue a scheduled check reports to (.github/workflows/check-urls.yml).
#
#   report_issue.sh comment <title> <body file>   add the body as a comment (open the issue if none)
#   report_issue.sh replace <title> <body file>   make the body the issue's text (open it if none);
#                                                 a weekly report that changes keeps one current issue
#   report_issue.sh close   <title> <body file>   close the open issue with the body as a comment
#
# Needs GH_TOKEN and GITHUB_REPOSITORY (set by Actions) and a job with `issues: write`.
set -euo pipefail
mode=$1 title=$2 body=$3
num=$(gh issue list --repo "$GITHUB_REPOSITORY" --state open --search "\"$title\" in:title" \
  --json number,title --jq "map(select(.title == \"$title\")) | .[0].number // empty")
case "$mode" in
  comment)
    if [ -n "$num" ]; then gh issue comment "$num" --repo "$GITHUB_REPOSITORY" --body-file "$body"
    else gh issue create --repo "$GITHUB_REPOSITORY" --title "$title" --body-file "$body"; fi ;;
  replace)
    if [ -n "$num" ]; then gh issue edit "$num" --repo "$GITHUB_REPOSITORY" --body-file "$body"
    else gh issue create --repo "$GITHUB_REPOSITORY" --title "$title" --body-file "$body"; fi ;;
  close)
    if [ -n "$num" ]; then gh issue close "$num" --repo "$GITHUB_REPOSITORY" --comment "$(cat "$body")"; fi ;;
  *) echo "usage: report_issue.sh comment|replace|close <title> <body file>" >&2; exit 2 ;;
esac
