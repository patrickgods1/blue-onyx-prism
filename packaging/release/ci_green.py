#!/usr/bin/env python3
"""Decide whether a commit's CI is green enough to build a release from it (release.yml, job ci-green).

Usage:
  ci_green.py --sha SHA --check-runs FILE --workflow-runs FILE [--self PATH] [--require PATH]...
  ci_green.py --self-test     # the fixtures in testdata/ next to this script

FILE holds what `gh api --paginate` printed: the pages of
`GET /repos/{repo}/commits/{sha}/check-runs` (or of `GET /repos/{repo}/actions/runs?head_sha=`),
or bare arrays / objects of check runs (workflow runs), one JSON value after another.

Rules:
- `--self` (the release workflow) is left out: its own runs on the same commit, this one and any
  earlier attempt, are not CI.
- Every `--require`d workflow (path as in `.github/workflows/ci.yml`) has a run on the commit, and
  its latest run succeeded.
- No other check run on the commit failed: every one is completed with success, neutral or
  skipped. Of check runs with one name in one workflow, the latest counts (a re-run replaces).

Exit codes: 0 green, 1 red (names the failing checks), 3 still running (wait and ask again),
4 a required workflow has not run on the commit (yet, or ever). 2 bad usage or input.
"""
import argparse
import json
import sys
from pathlib import Path

GREEN = 0
RED = 1
RUNNING = 3
MISSING = 4
OK_CONCLUSIONS = {"success", "neutral", "skipped"}


def values(text, key):
    """Every item of `key` lists (or bare items) in a stream of JSON values."""
    dec = json.JSONDecoder()
    i, out = 0, []
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            return out
        v, i = dec.raw_decode(text, i)
        stack = [v]
        while stack:
            x = stack.pop(0)
            if isinstance(x, list):
                stack[:0] = x
            elif isinstance(x, dict) and key in x and isinstance(x[key], list):
                stack[:0] = x[key]
            elif isinstance(x, dict):
                out.append(x)
            else:
                raise ValueError(f"unexpected JSON value {x!r} in the {key} input")


def workflow_path(run):
    # `path` is ".github/workflows/ci.yml", for some events with "@<ref>" after it.
    return (run.get("path") or "").split("@", 1)[0]


def decide(sha, check_runs, workflow_runs, self_path, required):
    """(exit code, message lines)."""
    by_suite = {}
    latest_run = {}
    for r in workflow_runs:
        path = workflow_path(r)
        by_suite[r.get("check_suite_id")] = path
        key = (r.get("created_at") or "", r.get("id") or 0)
        if path not in latest_run or key > latest_run[path][0]:
            latest_run[path] = (key, r)

    latest_check = {}
    for c in check_runs:
        suite = (c.get("check_suite") or {}).get("id")
        wf = by_suite.get(suite)
        if self_path and wf == self_path:
            continue
        source = wf or (c.get("app") or {}).get("slug") or "another app"
        key = (c.get("started_at") or "", c.get("completed_at") or "", c.get("id") or 0)
        name = (source, c.get("name") or "?")
        if name not in latest_check or key > latest_check[name][0]:
            latest_check[name] = (key, c)

    failing, running, missing, lines = [], [], [], []
    for path in required:
        if path not in latest_run:
            missing.append(path)
            continue
        r = latest_run[path][1]
        if r.get("status") != "completed":
            running.append(f"{path} (run {r.get('html_url') or r.get('id')}: {r.get('status')})")
        elif r.get("conclusion") not in OK_CONCLUSIONS:
            failing.append(f"{path}: {r.get('conclusion')} ({r.get('html_url') or r.get('id')})")
    for (source, name), (_, c) in sorted(latest_check.items()):
        where = f"{name} [{source}]"
        if c.get("status") != "completed":
            running.append(f"{where}: {c.get('status')}")
        elif c.get("conclusion") not in OK_CONCLUSIONS:
            failing.append(f"{where}: {c.get('conclusion')} ({c.get('html_url') or 'no link'})")

    if failing:
        lines.append(f"CI is not green on {sha}; these checks did not pass:")
        lines += [f"  - {f}" for f in failing]
        lines.append(
            "A release is built only from a commit whose CI passed. Fix the failure on main (or "
            "re-run the failed jobs if they failed for a reason outside the code), then run the "
            "release again for this tag once they are green."
        )
        return RED, lines
    if missing:
        lines.append(f"No run of {', '.join(missing)} on {sha}.")
        lines.append(
            "Tag a commit of main that CI has run on (every push to main runs it): the tag's "
            "commit may be on another branch, or its CI run has not started yet."
        )
        return MISSING, lines
    if running:
        lines.append(f"CI is still running on {sha}:")
        lines += [f"  - {r}" for r in running]
        return RUNNING, lines
    lines.append(f"CI is green on {sha}:")
    lines += [f"  - {name} [{source}]: {c.get('conclusion')}" for (source, name), (_, c) in sorted(latest_check.items())]
    return GREEN, lines


def self_test():
    here = Path(__file__).resolve().parent / "testdata"
    files = sorted(here.glob("*.json"))
    if not files:
        print(f"no fixtures in {here}", file=sys.stderr)
        return 1
    bad = 0
    # What `gh api --paginate` prints: page objects back to back (pretty or compact), or arrays.
    pages = '{"total_count": 2, "check_runs": [{"name": "a"}]}\n{"total_count": 2,\n "check_runs": [{"name": "b"}]}'
    for text, want in [(pages, ["a", "b"]), ('[{"name": "a"}]\n[]\n[{"name": "b"}]', ["a", "b"]),
                       ('{"name": "a"}\n{"name": "b"}\n', ["a", "b"]), ("", [])]:
        got = [x["name"] for x in values(text, "check_runs")]
        if got != want:
            bad += 1
            print(f"FAIL values({text!r}) = {got}, expected {want}", file=sys.stderr)
    for f in files:
        t = json.loads(f.read_text(encoding="utf-8"))
        code, lines = decide(
            t["sha"], t["check_runs"], t["workflow_runs"],
            t.get("self", ".github/workflows/release.yml"),
            t.get("require", [".github/workflows/ci.yml"]),
        )
        text = "\n".join(lines)
        problems = []
        if code != t["expect"]:
            problems.append(f"exit {code}, expected {t['expect']}")
        for s in t.get("mentions", []):
            if s not in text:
                problems.append(f"message lacks {s!r}")
        for s in t.get("never_mentions", []):
            if s in text:
                problems.append(f"message names {s!r}")
        if problems:
            bad += 1
            print(f"FAIL {f.name}: {'; '.join(problems)}\n{text}", file=sys.stderr)
        else:
            print(f"ok   {f.name} (exit {code})")
    print(f"{len(files) - bad} of {len(files)} fixtures passed")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--sha")
    p.add_argument("--check-runs", type=Path)
    p.add_argument("--workflow-runs", type=Path)
    p.add_argument("--self", dest="self_path", default=".github/workflows/release.yml")
    p.add_argument("--require", action="append", default=[])
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not (a.sha and a.check_runs and a.workflow_runs and a.require):
        p.error("--sha, --check-runs, --workflow-runs and at least one --require are needed")
    try:
        checks = values(a.check_runs.read_text(encoding="utf-8"), "check_runs")
        runs = values(a.workflow_runs.read_text(encoding="utf-8"), "workflow_runs")
    except (OSError, ValueError) as e:
        print(f"cannot read the GitHub answers: {e}", file=sys.stderr)
        return 2
    code, lines = decide(a.sha, checks, runs, a.self_path, a.require)
    print("\n".join(lines), file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
