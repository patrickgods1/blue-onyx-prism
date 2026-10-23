# Prompt: production-readiness review and hardening of Blue Onyx Prism

Paste everything below the line into a fresh Claude Code session (or another coding agent) at
the repository root. It is written for an agent with file, shell and git access. It assumes the
agent reads `CLAUDE.md` first.

---

You are the lead reviewer and hardening engineer for **Blue Onyx Prism**, a Rust (edition 2024,
toolchain pinned in `rust-toolchain.toml`) HTTP object-detection service that Blue Iris calls
through a CodeProject.AI-compatible API. It runs native OpenVINO (`openvino` crate, runtime
linking) and ONNX Runtime (`ort`, load-dynamic) behind a `Backend` enum, loads N models each on
its own worker thread, downloads pinned runtimes and models on demand, and ships a web UI, a
Prometheus endpoint, a Windows service, systemd/launchd units and a Docker image.

Read `CLAUDE.md` and `README.md` before touching anything. They are authoritative on
architecture, device selection rules, conventions and the test layout. `docs/REVIEW_PLAN.md`
records decisions from an earlier design review; do not re-litigate those.

## Goal

Review **every file in the repository**, not a sample, and bring the project to a state a
careful team would be comfortable running unattended on a home NVR box for years: no panics on
untrusted input, no way for a LAN client to read or write outside the intended directories, no
silent resource growth, predictable behavior under load and on failure, reproducible and
verifiable builds, and documentation that matches the code. Apply fixes, do not only list them.

"Every file" means every tracked file: about 90 Rust sources under `src/` (roughly 60,000
lines), `tests/`, `templates/*.html`, `assets/*.{js,css,yaml,json}`, `scripts/**` (Python,
bash, PowerShell, lock files), `.github/**`, `deploy/**`, `Dockerfile`, `Cargo.toml`,
`deny.toml`, `rust-toolchain.toml`, `.gitignore`, `LICENSE`, `README.md`, `CLAUDE.md` and
`docs/**`. A file you did not open is a file you did not review; the coverage ledger in phase 0
makes that visible.

## Ground rules

1. **Work on a branch per phase**, PR-sized commits, each green on:
   ```
   cargo fmt --all -- --check
   cargo clippy --locked --all-targets -- -D warnings
   cargo clippy --locked --all-targets --no-default-features -- -D warnings
   cargo test --locked
   cargo build --locked --release --bins
   cargo deny check advisories bans licenses sources
   ```
   End every commit message with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
2. **Behavior-preserving by default.** The Blue Iris response shape (`success, message, error,
   predictions[{x_min,y_min,x_max,y_max,confidence,label}], count, command, moduleId,
   executionProvider, canUseGPU, inferenceMs, processMs, analysisRoundTripMs`) must stay
   byte-compatible. Config file keys stay backward compatible; new keys get defaults that match
   today's behavior. If a hardening change must alter behavior (for example a new default limit),
   say so in the commit message and in `README.md`.
3. **Keep the conventions**: `anyhow` for app errors, `tracing` for logs, no `build.rs`, no
   `unsafe` outside `src/backend/`, all `ort` code in `src/backend/ort.rs`, everything portable
   unless it must be `#[cfg(windows)]`. Never commit model weights.
4. **Do not change pinned resource hashes or URLs** in `src/resources/catalog.rs` unless the
   finding is specifically about them, and then explain why.
5. **Verify before you claim.** Every finding you report must cite a file and line and state the
   concrete input or state that triggers it. Run the tests you add. If something cannot be verified
   on this machine (CUDA, DirectML, a Windows service), say so explicitly instead of guessing.
6. **Ask nothing mid-task.** Make routine calls yourself, state assumptions in the final report, and
   reserve open questions for the report's last section.
7. Delegate well-specified mechanical work to subagents that own disjoint files (see
   `CLAUDE.md`, "Working style for agents"); keep concurrency, backend and unsafe code with the
   most capable model. Each subagent finishes with build and tests green for its files.

## Phase 0: file-by-file review and risk map (read-only, report before changing anything)

This is the broad pass. Walk the tree in a fixed order (`git ls-files | sort`), open each file
in full, and apply the rubric below. Delegate by directory to parallel read-only subagents if
you like (`src/backend`, `src/model`, `src/resources`, `src/server`, `src/benchmark`, the rest
of `src`, `tests`, and all non-Rust files are natural splits), but you own the merged ledger
and must spot-check each subagent's entries against the code.

### Coverage ledger

`docs/PRODUCTION_REVIEW.md` starts with a table with one row per tracked file: path, lines,
category (core, backend, model, resources, server, benchmark, cli, test, template, asset,
script, workflow, deploy, docs, build), reviewer (you or which subagent), and a verdict:
*clean*, *findings* (with their IDs), or *not reviewed* with the reason. The goal is zero
*not reviewed* rows. A file over 1500 lines gets one row per top-level section or `impl` block
so a long file cannot hide an unread region.

### Rubric applied to every Rust file

For each file, answer in writing, briefly, even when the answer is "nothing found":

1. **Inputs.** What untrusted data enters this file (HTTP body, form field, query string, path
   string, config value, model file, class YAML, archive entry, downloaded JSON, child-process
   output, environment variable, CLI argument, filesystem listing)? Is each validated at the
   boundary or deeper, and is the validation complete?
2. **Failure paths.** Every `unwrap`, `expect`, `panic!`, `unreachable!`, `assert!`, `[i]`
   index, slice range, `as` cast, unchecked arithmetic, `unsafe` block, `.lock()` on a mutex
   that can be poisoned, and `?` that drops context. For each: can an outside input reach it,
   and what happens when it fires?
3. **Resources.** What this file allocates, opens, spawns or holds across awaits or across
   requests: buffers, files, sockets, threads, channels, OpenVINO or ORT handles, temp dirs,
   locks. Is each bounded and released on every exit path, including error, timeout, cancel and
   shutdown?
4. **Concurrency.** Shared state, lock order, atomics and their orderings, channel capacities,
   `ArcSwap` and `RwLock` use, blocking calls inside async handlers, anything that can deadlock
   or starve the detection route.
5. **Platform.** `#[cfg]` branches, path and separator handling, case sensitivity, line endings,
   file permissions, Windows verbatim paths, signals versus service control; what is untested on
   each of the three target platforms.
6. **Correctness.** Does the code do what its doc comment and `CLAUDE.md` say? Off-by-one in
   letterbox or NMS math, wrong class-index base, confidence or coordinate normalization, time
   unit mixups, stale comments, dead branches, duplicated logic that has drifted.
7. **API and config stability.** Any serialized shape or config key, and whether a change here
   would break Blue Iris, an old config file, `benchmark.json`, `.installed.json` or the UI.
8. **Tests.** Which behaviors in this file have a test, which do not, and which of the untested
   ones matter.
9. **Readability.** Function length, nesting, naming, module docs, and whether a new maintainer
   could find and change the behavior safely.

### Rubric for non-Rust files

- **`templates/*.html`, `assets/live.js`, `assets/style.css`**: every interpolation is escaped
  by Askama or escaped by hand where `|safe` is used; no inline event handlers or inline scripts
  that a CSP would block; every form that changes state posts to an admin route; `live.js` handles
  fetch failures, non-JSON responses and a restarting server without unbounded retries or memory
  growth; no third-party resources are loaded from the network; the pages work with JavaScript
  disabled for the critical actions (config save, restart) or state that they do not.
- **`assets/*.yaml`, `assets/bench/*.json`, `scripts/model-export/locks/*.txt`**: shape
  matches what the Rust parser expects; a malformed or truncated file produces an error, not a
  panic; hashes and URLs are pinned and the pinning is verified by a test or CI job.
- **`scripts/*.py`** (`export_model.py`, `build_bench_sets.py`, `convert_onnx_to_ir.py`):
  pinned dependencies, no `shell=True`, no path built from untrusted names without
  normalization, deterministic outputs, `--help` accurate, failures exit non-zero with a clear
  message, no network access other than what is documented, and `export_model.py` matches what
  `src/resources/export.rs` embeds and invokes (argument names, exit codes, output file names,
  the `COREML_CACHE_KEY` stamp).
- **`scripts/*.sh`, `scripts/ci/smoke.sh`, `scripts/*.ps1`**: `set -euo pipefail` or the
  PowerShell equivalent (`$ErrorActionPreference = 'Stop'`), quoted variables, no `curl | sh`,
  idempotent install and uninstall, explicit privilege requirements, service user creation and
  file ownership correct, temp files cleaned on failure, and the smoke test actually fails when
  detection returns `success:false` or the wrong provider.
- **`.github/workflows/*.yml`, `.github/dependabot.yml`**: actions pinned by SHA (done), least
  `permissions:` per job, no `pull_request_target`, no secrets in logs, caches keyed safely,
  `--locked` everywhere, release workflow cannot be triggered to overwrite an existing release,
  `continue-on-error` steps are listed as known gaps, and the weekly URL check reports to
  somewhere a human will see.
- **`deploy/*`, `Dockerfile`**: least privilege (see phase 4), volumes and paths consistent with
  the config defaults, image reproducible from a clean checkout, base image tag equals the
  catalog's OpenVINO version, healthcheck present, no build-time network access that is not
  pinned and hashed.
- **`Cargo.toml`, `Cargo.lock`, `deny.toml`, `rust-toolchain.toml`**: every dependency is used
  (`cargo machete` or `cargo udeps` if available), features are minimal, versions that must
  stay pinned are commented with why, release profile is intentional (`strip`, `lto`,
  `panic` strategy decided and documented), advisory ignores dated and justified.
- **`README.md`, `CLAUDE.md`, `docs/*.md`, `LICENSE`**: every command shown runs as written,
  every config key and CLI flag described exists with that name and default, attribution for
  blue-onyx and model licenses is correct, and nothing describes behavior the code no longer has.

### Risk map

After the ledger, `docs/PRODUCTION_REVIEW.md` continues with:

- A one-paragraph threat model: who can reach the port (Blue Iris on another LAN host, browsers
  on the same host, anything on the LAN when `bind_address` is `0.0.0.0`), what they can send
  (multipart images up to `BODY_LIMIT`, form posts to admin routes, path strings to the browse and
  add-model endpoints), and what the process can do (download and extract archives, run a
  user-triggered model export with `uv` and Python, write config, cache and image folders).
- Every network listener, every filesystem write path, every place downloaded content is parsed
  or extracted, every place external programs are executed, every `unsafe` block (currently in
  `backend/{device,libs,ort,detect}.rs`) with a one-line justification each.
- A table of non-test `unwrap`/`expect`/`panic!`/index operations in library and binary code
  (there are a few dozen; `src/server/test_util.rs` is test support and may be excluded), each
  classified as *provably infallible*, *should be an error*, or *should be a controlled abort*.
- Hot-path allocations and copies per request (multipart read, JPEG decode, resize, letterbox,
  tensor fill, postprocess, JSON encode, optional image save), with what is bounded and what is
  not.
- Findings ranked by severity (crash or data loss, security boundary, resource exhaustion,
  correctness, operability, maintainability), each with an ID, file:line, trigger, proposed fix
  and the phase that will fix it. Findings from the rubric that fit none of phases 1 to 5 get
  their own section and are still fixed, in the phase whose files they touch.

Stop and commit this document before phase 1 so the review stands on its own. Phases 1 to 5
below are the minimum; the ledger's findings extend them. At the end of each phase, update the
ledger's verdict column for every file that phase touched.

## Phase 1: correctness and crash safety

- Eliminate every panic reachable from a request, a config file, a model file, a class YAML, a
  downloaded archive, `benchmark.json`, `.installed.json` or a CLI argument. Prefer returning an
  error through the existing `anyhow`/`api.rs` paths. Where a panic is genuinely the right
  response (invariant violation in `worker.rs` or `registry.rs`), keep it but make the message
  name the invariant.
- Add a crate-level lint policy in `src/lib.rs` and the binaries, enforced by CI, at minimum:
  `#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing,
  clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::large_futures,
  clippy::unused_async, clippy::dbg_macro, missing_debug_implementations)]`
  plus `unsafe_op_in_unsafe_fn`, `unreachable_pub` where it is not noisy, and
  `clippy::undocumented_unsafe_blocks` with `// SAFETY:` comments on each existing block.
  Allow lints per item with a reason rather than crate-wide. Keep `-D warnings` the gate.
- Audit integer arithmetic on sizes and image geometry (`src/image.rs`, `src/model/preprocess.rs`,
  `src/model/nms.rs`, each family's postprocess, `src/backend/onnx_rewrite.rs`,
  `src/resources/extract.rs`): use checked or saturating arithmetic where an input controls the
  operand, and reject before allocating.
- Decode limits: before decoding a JPEG or PNG, read the header and reject images whose width x
  height x channels exceeds a configured ceiling (default on the order of 50 megapixels) and
  whose dimensions are zero. Apply the same ceiling to the image-saving path and to benchmark
  dataset loading.
- Model output validation: every family's postprocess must check the output tensor's rank, shape
  and element count against what it indexes (`[1,300,6]`, `[1,N,5+C]`, `[1,4+C,8400]`,
  `logits/pred_boxes`, `labels/boxes/scores`) and return an error naming the mismatch instead of
  indexing out of bounds. Cover each with a synthetic-tensor test in `tests/postprocess.rs`.
- Config validation: every numeric field has a documented range enforced at load and at save
  (thresholds 0..=1, ports 1..=65535, timeouts and thread counts bounded, paths non-empty); an
  invalid field names the key and the accepted range. A config that fails validation must not be
  overwritten by the CLI write-back.
- Float handling: NaN or infinite confidences and coordinates are dropped before NMS and before
  serialization; coordinates are clamped to the image.

## Phase 2: security boundaries

- **Admin surface.** Confirm that every state-changing route (config save, restart, log level,
  resource download, remove, add-to-config, model add, export, export cancel, benchmark run and
  apply, test-page upload if it writes) is behind the loopback-or-`allow_remote_admin` middleware,
  and that detection and `custom/list` are not. Add a router test that enumerates the routes and
  asserts the classification so a new route cannot be added unguarded by accident. Consider a
  per-request origin check on admin form posts (same-origin via `Origin`/`Sec-Fetch-Site`) to
  block CSRF from a browser on the same host; document the decision either way.
- **Path handling.** `src/server/browse.rs` and the add-model-by-path flow must only ever
  canonicalize and then check `starts_with` one of the configured roots; symlinks that escape a
  root are refused; Windows verbatim and UNC prefixes, drive-relative paths, NUL bytes and
  overlong names are rejected with a test each. The same check guards `cache_dir`, `download_dir`,
  the saved-image directory and the log directory when set from the UI.
- **Archive extraction** (`src/resources/extract.rs`): keep the whitelist, and also enforce a
  per-file and total decompressed size ceiling (zip bombs), refuse absolute paths, `..`, symlinks
  and hard links (already partly there), refuse entries with setuid/setgid bits, and extract into
  a staging directory that is renamed atomically only after every entry passed. Add tests with
  crafted archives for each refusal.
- **Downloads** (`src/resources/manager.rs`): HTTPS only (done), SHA-256 verified before install
  (done), plus: a maximum redirect count, a bound on `Content-Length` versus the catalog size,
  verification that a resumed Range download still hashes correctly, and a test that a
  tampered body is rejected and leaves no partial file in the install location. The update check
  in `src/update.rs` must tolerate a malformed or hostile GitHub response without panicking and
  must never auto-apply anything.
- **Model export** (`src/resources/export.rs`): it is the only code that runs downloaded
  programs. Confirm hashes are checked for `uv`, the lock files and the source archives before
  execution; the environment passed to the child is minimal and explicit (no inherited
  `PYTHONPATH`, `PIP_*`, proxies only if configured); the working directory is a fresh staging
  dir; the child process tree is killed on cancel and on service stop (process group on Unix, job
  object on Windows); stdout and stderr are bounded when captured. Write this down in the module
  doc.
- **Dynamic library loading** (`src/backend/libs.rs`, `setup_*`): libraries are loaded only from
  the configured runtime directories or the OS path, never from the current working directory or a
  user-supplied request path; on Windows use the safe search flags. Document the trust boundary:
  anyone who can write the runtime directory owns the process.
- **Logging hygiene.** Nothing logs full request bodies, image bytes, or absolute paths of
  user home directories at `info` or above; errors returned to clients do not leak internal
  paths or stack traces. Prometheus and `/v1/devices` expose no secrets.
- **HTTP hygiene.** Set `Content-Type` with charset on HTML, `X-Content-Type-Options: nosniff`,
  a restrictive `Content-Security-Policy` for the UI (no inline event handlers; move inline
  scripts to served assets or add nonces), `Referrer-Policy`, and `Cache-Control: no-store` on
  JSON state endpoints. Reject unknown methods explicitly. Make sure the fallback handler does not
  reflect the request path unescaped.
- **Dependencies.** Keep `deny.toml` and justify each advisory ignore in a comment with a review
  date; add `cargo audit` or keep the cargo-deny job as the gate; pin GitHub Actions by SHA (done)
  and add `permissions:` least privilege to every workflow job; enable Dependabot for Cargo and
  Actions (check `.github/dependabot.yml`); add a `cargo vet` or `cargo crev` note in the release
  checklist only if the team wants it, otherwise skip.

## Phase 3: resource limits and behavior under load

The design target is at least 30 requests/s sustained from Blue Iris on an i5-8500 with headroom.

- **Backpressure.** Each model has a bounded crossbeam queue sized from the request timeout. Add
  an explicit server-wide concurrency cap on in-flight detection requests (a `tower` concurrency
  limit or a `Semaphore` in `AppState`) so a flood is rejected fast with a 503-style
  `success:false` message rather than queueing in tokio; expose the rejection count as a metric.
- **Timeouts everywhere.** Header read timeout and idle keep-alive timeout on the listener
  (slowloris), the existing per-request `request_timeout_secs` for inference, an upper bound on
  multipart read time, HTTP client timeouts on every outbound request (download, update check),
  and a bounded wait for model compile in `provision.rs`. None of these may be unbounded.
- **Memory.** `BODY_LIMIT` is 32 MB per request; combined with the concurrency cap this gives a
  computable worst case. Document it. Make sure the multipart reader streams to a `Bytes` buffer
  without a second copy, and that the decoded RGB buffer and the f32 tensor are reused per worker
  (allocation per request only when the input size changes). Check `src/logbuf.rs` (the in-memory
  log ring) and the per-model metrics for unbounded growth; the config change log and the
  benchmark history must be capped.
- **File descriptors and threads.** Confirm worker threads, the download thread and the export
  task are joined or detached deliberately on shutdown; a reload of one model must not leak the
  old `CompiledModel`, ORT session or thread. Add a test that reloads a model N times in a loop
  under `tests/live_config.rs` and asserts thread count and RSS do not climb (RSS check can be
  `#[ignore]` and run in the integration job).
- **Disk.** Saved images, logs (`tracing_appender::rolling::daily`), cache and download staging
  need retention: cap saved-image folder size or age, delete rolled logs older than N days, clean
  stale `.part` files and staging directories at startup, and never fill the disk silently; log a
  warning when free space under the data root drops below a threshold.
- **Shutdown.** `SIGTERM`/`SIGINT`, Windows service stop and the restart-from-UI path all drain
  in-flight requests with a deadline, flush logs, and exit non-zero on failure. Test the graceful
  path with the existing HTTP test harness.
- **Startup.** The HTTP listener must serve immediately while models compile (existing
  behavior); add a `GET /v1/status` or confirm the existing status route returns a machine-readable
  readiness that Blue Iris or a container orchestrator can poll, and wire it as the Docker
  `HEALTHCHECK` and the systemd `Type=notify` or a watchdog if cheap to add.

## Phase 4: operability

- **Structured logging.** Offer a JSON log format (`log_format: text|json`) for the file appender;
  keep text for the console. Every request log line carries model name, device, request id and
  the three timing fields. Log level is runtime-switchable (done) and the switch itself is logged.
- **Metrics.** Prometheus counters are process-lifetime and latencies are histograms (per the
  review decisions). Add: queue depth gauge per model, rejection counters by reason
  (initializing, queue full, timeout, body too large, decode error), download bytes and failures,
  process RSS and open file descriptors where cheap, build info gauge with version and git SHA.
- **Config.** Document every key in one table in `README.md` with type, default, range, whether
  it hot-applies or needs a restart, and whether it is exposed in the UI. Generate the table from
  the `Config` struct doc comments if feasible, or add a test that fails when a key is missing
  from the table.
- **Deployment hardening.** `deploy/blue-onyx-prism.service` gets `NoNewPrivileges=yes`,
  `ProtectSystem=strict`, `ProtectHome=yes`, `PrivateTmp=yes`, `ReadWritePaths=` for the data
  root, `DevicePolicy=closed` plus `DeviceAllow=/dev/dri` for the iGPU,
  `RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX`, `SystemCallFilter=@system-service`,
  `MemoryDenyWriteExecute=no` (JIT in OpenVINO) with a comment explaining each line and which
  ones break GPU access if tightened. The launchd plist and the Windows service get the
  equivalent least-privilege settings that apply. The Dockerfile already runs as non-root; add
  `HEALTHCHECK`, a read-only root filesystem with explicit writable volumes, and build with
  `--locked` (done); verify the image runs offline.
- **Release integrity.** Keep the `.sha256` per archive and add a provenance attestation
  (`actions/attest-build-provenance`) or a signed checksum manifest; document how a user verifies
  a download in `docs/RELEASE_CHECKLIST.md`. Embed the git SHA and build date in `--version` and
  the UI footer.
- **Error messages.** Every user-facing failure (config, download, model load, device fallback)
  states what happened, the most likely cause and the next action, in one sentence each.
  Review the strings in `startup.rs`, `provision.rs`, `registry.rs` and the config page.

## Phase 5: maintainability (behavior-neutral, last)

- The largest files (`benchmark/mod.rs`, `resources/export.rs`, `resources/catalog.rs`,
  `benchmark/cli.rs`, `backend/onnx_rewrite.rs`, `server/benchmark.rs`, `benchmark/report.rs`,
  each over 1900 lines) are split along existing seams with moves only; `git diff --stat`
  shows moves, and tests change only `use` paths.
- Dead code, duplicate helpers and copy-pasted parsing are consolidated. Public items that are
  only used inside the crate become `pub(crate)`.
- Every module has a `//!` doc that states its responsibility and its invariants; every
  `unsafe` block has a `// SAFETY:` comment; `cargo doc --no-deps` builds without warnings.
- Test gaps found in phases 1 to 4 are filled: a property-style test for NMS, fuzz-style tests
  for the multipart parser and for `extract.rs` using crafted inputs, a test per config
  validation rule, a router classification test, and a concurrency test that sends more requests
  than the cap and asserts fast rejection. Keep fixture-dependent tests `#[ignore]` as CI expects.
- Non-Rust findings from the phase 0 rubric are applied here if no earlier phase took them:
  template escaping and CSP cleanup, `live.js` retry and error handling, script strict modes and
  quoting, workflow permissions, dependency pruning, and doc corrections.
- `README.md`, `CLAUDE.md` and `docs/RELEASE_CHECKLIST.md` match the code after the changes.
  Remove or mark historical any doc section that no longer applies.

## Deliverables

1. `docs/PRODUCTION_REVIEW.md`: the coverage ledger with no *not reviewed* rows, the per-file
   rubric answers, the risk map, and at the end a per-finding status (fixed in commit X,
   deferred because Y, not an issue because Z).
2. One branch and PR per phase, each with a description listing the behavior changes, new config
   keys and their defaults, and the verification commands you ran with their results.
3. A final summary in the PR for phase 5 that states, honestly, what was verified on this
   machine, what was verified only by tests, and what still needs a run on Windows, on an NVIDIA
   box, or under real Blue Iris load. Unverified claims are worse than open items.

Start with phase 0 now.
