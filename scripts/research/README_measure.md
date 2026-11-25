# Resource measurement harness

`measure_resources.sh` (macOS, Linux; bash 3.2 compatible) and `measure_resources.ps1` (Windows 11,
PowerShell 5.1) measure what the service costs while idle, under load, across a generation restart, and what it
leaves behind after `kill -9` and a graceful stop. They are reports: they always exit 0. No root needed.

## Run

```sh
cargo build --release                       # also builds test-blue-onyx-prism
target/release/blue-onyx-prism download-models --name IPcam-general
scripts/research/measure_resources.sh                         # full run, about 30 min
scripts/research/measure_resources.sh --idle 20 --load-secs 20 --post-idle 10 --restart-idle 15 --final-run 10   # smoke
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\research\measure_resources.ps1
powershell -ExecutionPolicy Bypass -File scripts\research\measure_resources.ps1 -Idle 20 -LoadSecs 20 -PostIdle 10 -RestartIdle 15 -FinalRun 10
```

Options (sh / ps1): `--bin-dir/-BinDir` (default `target/release`), `--scratch/-Scratch` (default a new temp
dir), `--model/-Model` (default `IPcam-general`, an `.onnx` already in `<bin dir>/models`), `--idle` 600,
`--load-rate` 10, `--load-secs` 300, `--post-idle` 300, `--restart-idle` 300, `--interval` 5, `--final-run` 60.

## What it does

1. Writes a scratch config: a free port pair on 127.0.0.1, `download_dir` = an empty `<scratch>/data`,
   `auto_download: false`, `openvino_dir` / `onnxruntime_dir` pointing at the binary dir's existing runtimes and
   the model by absolute path, so nothing big is copied and the data root starts empty.
2. Snapshots the data root (path + size), starts the service, waits for `GET /v1/status` 200.
3. Samples every `--interval` s into `<scratch>/samples.csv`: RSS (Windows: working set and private bytes),
   CPU, threads, open fds (Windows: handles), voluntary context switches (Linux).
4. Phases: idle, `test-blue-onyx-prism --rate R --duration S` on the plain port, idle, generation restart
   (`POST /config/restart` over HTTPS with the token from `admin-token --config`, `curl -k` to 127.0.0.1 only
   because the certificate is self-signed), idle.
5. `kill -9` / `Stop-Process -Force`; lists surviving descendants; diffs the data root against the snapshot.
   Starts again for `--final-run` s, stops gracefully (SIGTERM), diffs again.

## Reading the summary

- Idle CPU is the delta of cumulative CPU time over the phase (the `ps %cpu` column is a lifetime average and
  is only kept in the CSV). Windows: percent of all cores.
- Wakeups: Linux reports voluntary context switches per second (sum over all threads of `/proc/<pid>/task`).
  macOS has no per-process idle wakeup counter without root (`top -stats idlew` and `powermetrics` need it), so the
  line says n/a. Windows needs ETW or Resource Monitor.
- "New/changed files" counts lines of the snapshot that differ, so a file that grew counts. Expected after a first
  run: `auth/` (token, certificate, setup code, lock). The listing at the end shows up to 40.
- The first run also creates `auth/` once, so "files vs empty start" is never 0.

## Limits

- Windows graceful stop: the script launches the service hidden, which cannot be sent Ctrl-C without P/Invoke, so
  it tries `CloseMainWindow` and reports "needed Stop-Process -Force" when that fails. For the installed service
  use `blue-onyx-prism service stop` (admin) and compare the data root by hand. The Windows script has not been
  run (no Windows host available when written); it is PowerShell 5.1 syntax only.
- The restart exercised is a generation restart in the same process (the process id does not change).
- macOS (M1, end of the 2026-11 audit, full run): idle CPU 0.04 % in every phase, RSS about 265 MB, 426 MB right
  after a generation restart falling to 346 MB within 5 min, threads stable, nothing left after kill -9 or SIGTERM.
  RSS on macOS counts lazily freed allocator pages; see docs/PRODUCTION_REVIEW.md, "phase 11 notes".
- The harness starts no job child; `scripts/ci/test_parent_watch.py` checks that the Python children end when their
  parent is killed.
