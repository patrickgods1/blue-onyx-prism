"""End-to-end check of the Python children's parent watch (audit 2026-11, RA-012, phase 11).

The service starts every Python script (export, training, auto-label, import) as the leader of its
own process group, with a pipe as stdin it never writes to and BOP_PARENT_PIPE=1. Each script's
`watch_parent()` reads that pipe: end of file means the service is gone, and the script kills its
whole group. This test reproduces that with real processes, per script:

  test (this process)
    -> parent: a stand-in for the service, in its own session
         -> child: imports the script, calls watch_parent(), starts a helper (`sleep`) in its group
              -> helper

The parent is then killed with SIGKILL, and the child and its helper must be gone within a few
seconds. A second case first stops the child's group with SIGSTOP (a paused training run): its
watch thread cannot run, and the kernel's orphaned-group rule (SIGHUP + SIGCONT to a stopped group
whose last outside parent dies) must end it instead; Linux's PR_SET_PDEATHSIG, which the service
also sets, is left out here so the test shows the portable path.

Windows: the service's job object ends the tree there and watch_parent() returns at once, so this
test reports a skip. Stdlib only; run with `-I`. Usage: python -I scripts/ci/test_parent_watch.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ["export_model.py", "train_model.py", "autolabel.py", "import_footage.py"]
GONE_WITHIN = 10.0  # seconds; the pipe watch reacts at once, the fallback poll every 2 s

# The child: imports the script without running its main, starts the watch, starts a helper in its
# own group (like a data loader worker), reports both pids on stdout and waits.
CHILD = r"""
import importlib.util, json, os, subprocess, sys, time
spec = importlib.util.spec_from_file_location("under_test", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.watch_parent()
helper = subprocess.Popen(["sleep", "300"], stdin=subprocess.DEVNULL)
print(json.dumps({"child": os.getpid(), "helper": helper.pid}), flush=True)
time.sleep(300)
"""

# The parent (the service's stand-in): starts the child as its own group leader with a pipe stdin
# it never writes, passes the child's report on, then waits to be killed.
PARENT = r"""
import os, subprocess, sys, time
env = dict(os.environ, BOP_PARENT_PIPE="1")
child = subprocess.Popen([sys.executable, "-I", "-c", sys.argv[1], sys.argv[2]],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                         preexec_fn=lambda: os.setpgid(0, 0))
sys.stdout.write(child.stdout.readline().decode())
sys.stdout.flush()
time.sleep(300)
"""


def alive(pid: int) -> bool:
    """True while `pid` runs (a zombie waiting for its new parent to reap it counts as gone)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def case(script: str, paused: bool) -> str | None:
    """Runs one case; returns None when it passed, else what went wrong."""
    parent = subprocess.Popen(
        [sys.executable, "-I", "-c", PARENT, CHILD, str(ROOT / "scripts" / script)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, start_new_session=True)
    pids: dict[str, int] = {}
    try:
        line = parent.stdout.readline().decode().strip()
        if not line:
            return "the child reported nothing (import failed?)"
        pids = json.loads(line)
        if paused:
            os.killpg(pids["child"], signal.SIGSTOP)
            time.sleep(0.3)
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait()
        deadline = time.monotonic() + GONE_WITHIN
        while time.monotonic() < deadline:
            if not any(alive(p) for p in pids.values()):
                return None
            time.sleep(0.1)
        left = {k: v for k, v in pids.items() if alive(v)}
        return f"still running {GONE_WITHIN:.0f} s after the parent's kill -9: {left}"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        for p in pids.values():
            try:
                os.kill(p, signal.SIGCONT)
                os.kill(p, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def main() -> int:
    if os.name == "nt":
        print("parent watch: skipped on Windows (the job object ends the tree there)")
        return 0
    failed = []
    for script in SCRIPTS:
        for paused in (False, True):
            what = f"{script} ({'paused' if paused else 'running'})"
            problem = case(script, paused)
            print(f"{what}: {'ok' if problem is None else 'FAILED: ' + problem}")
            if problem is not None:
                failed.append(what)
    if failed:
        print(f"parent watch: {len(failed)} case(s) failed: {', '.join(failed)}")
        return 1
    print("parent watch ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
