# Review follow-up plan

Outcome of the 2026-10-11 design review. Five phases, each one PR, each green on `cargo test`,
`cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` (plus the `--no-default-features`
clippy/build on Linux). Phases 1 and 4 touch disjoint files and can run in parallel; phase 2 is
the concurrency-sensitive one; phase 5 is behavior-neutral and goes last so it does not conflict.

Design target: Blue Iris with at least 6 cameras x 5 requests/s each = 30 requests/s sustained on
an i5-8500 + UHD 630, with headroom. Users do not administer the service from another machine, but
Blue Iris may run on another machine. Two browser tabs are the only concurrent config writers.

Decisions taken in the review (do not re-litigate in the PRs):
- One request timeout (`request_timeout_secs`), not a second hard-coded one.
- A request for a model that is still compiling (lazy or not) is answered immediately, never queued.
- The HTTP listener survives generation restarts.
- Config changes hot-apply wherever the worker can pick them up per request; a restart is only for
  runtime-level settings. The 3-way form merge and the 409 conflict path are removed (last write
  wins, the change log stays).
- ORT is verified in CI with the providers hosted runners have (`cpu`, `coreml`, `directml` on
  WARP if it works); CUDA/TensorRT are a release checklist step with a benchmark report.
- Prometheus stays; counters become process-lifetime and latencies become histograms.

---

## Phase 1 — latency and restart fixes

Four independent changes; each is a small commit inside the PR.

### 1.1 One timeout, immediate answers while initializing

Files: `src/server.rs` (`dispatch`, `RESPONSE_TIMEOUT`), `src/worker.rs` (`run`, lazy path),
`src/config.rs` (validation message), `tests/enabled_models_http.rs`, `tests/multi_model_http.rs`.

- Delete `RESPONSE_TIMEOUT`. `dispatch` waits `handle.request_timeout()` (a new field on
  `WorkerHandle` set from `WorkerConfig::request_timeout`, updated by phase 3) plus a fixed
  500 ms grace so the worker's own "waited longer than the timeout" drop wins the race and the
  client sees the worker's message, not the server's.
- `dispatch`: the `ModelState::Initializing if handle.accepts_while_initializing()` arm no
  longer enqueues the request. It sends a *trigger* instead: a new `WorkerHandle::trigger_load()`
  that pushes a `WorkItem::Load` variant (see below) if `load_started` is false (compare-and-swap),
  and returns the normal "Model 'x' is initializing (compiling on <planned label>)" error at once.
  `state.detail()` is set by the worker to `"compiling on <candidate device>"` while it tries a
  candidate, so the message names the device.
- `WorkItem` becomes an enum: `Detect(VisionDetectionRequest, oneshot::Sender<..>, Instant)` and
  `Load`. The worker loop handles `Load` by calling `try_load()` when `engine.is_none()` and the
  state is not `Failed`; a `Detect` that arrives while `engine.is_none()` for a lazy model is
  answered with the initializing error (it can only happen in the window between the CAS and the
  state update; it must not block).
- Metrics: a request answered "initializing" counts in `requests` but not in `dropped`; add
  `ModelMetrics::rejected_initializing: AtomicU64` and export it (`model_rejected_total{reason}`
  in phase 4; for now a plain counter).
- Tests: `tests/multi_model_http.rs` lazy test asserts the first request returns immediately with
  `success:false` and the message containing "initializing", and that a later request succeeds.
  A unit test in `worker.rs` for the CAS (`trigger_load` is idempotent).

### 1.2 Multi-thread tokio runtime + blocking offload

Files: `src/runner.rs` (`build_runtime`), `src/server.rs` (`devices_snapshot`, `config_page`,
`stats_page`, `welcome`, `test_submit`), `src/server/live.rs` (`config_json`),
`src/bin/blue_onyx_prism_service.rs` (uses `build_runtime`).

- `build_runtime`: `new_multi_thread().worker_threads(2).enable_all()`. Two is enough: handlers
  are short; the point is that one slow handler cannot stall the detection route. Everything in
  `AppState` is already `Send + Sync` (`Arc`, `RwLock`, atomics); `cargo build` confirms.
- Wrap in `tokio::task::spawn_blocking`:
  - `devices_snapshot(&state)` (it walks `download_dir`, probes ORT libraries and reads the
    config file). Callers: `devices_json`, `welcome`, `config_page`, `stats_page`. Return the
    snapshot through `spawn_blocking(move || devices_snapshot(&state)).await` with
    `state: Arc<AppState>` cloned in.
  - `ConfigStore::check_disk` when its cheap pre-check says the file changed (the read + hash).
    Simplest: `live::config_json` calls `spawn_blocking` around `payload(&state)` as a whole.
  - `bench_summary` cache refresh in `live.rs` (reads `benchmark.json`).
  - `logs_json` / `logs_page` only format an in-memory ring: no change.
- Do not offload `dispatch`: it is a channel send.
- Test: none new beyond the existing HTTP tests passing on the multi-thread runtime
  (`AppState` tests in `server.rs` build their own runtimes with `#[tokio::test]`; leave them).

### 1.3 The listener survives restarts

Files: `src/server.rs` (`serve`, `router`, `AppState`), `src/runner.rs`, `src/bin/*.rs` callers of
`server::serve`, `tests/*_http.rs` that call `serve`.

- `AppState` is what changes per generation. Introduce `pub struct Shared { current: RwLock<Arc<AppState>> }`
  (std `RwLock`; reads are one `Arc` clone per request). The router's state type becomes
  `Arc<Shared>`; every handler does `let state = shared.current();` first. Mechanical change:
  a `State(shared): State<Arc<Shared>>` extractor plus one line; keep handler bodies untouched.
  (An `axum` middleware that inserts the current `Arc<AppState>` as a request extension and an
  extractor `Current(state)` is the alternative if the one-line-per-handler edit turns out noisier
  than expected. Either is fine; pick one and use it everywhere.)
- `serve` is split: `bind(port) -> Listeners` (v4 any + v6 loopback, best effort, called once
  in `run_server_with` before the loop) and `serve_on(listeners, shared, shutdown)` which runs
  for the whole process. Generation restarts no longer stop the server; they swap
  `shared.current`.
- `run_server_with`: the loop keeps its shape (prepare provision, start registry, build
  `AppState`, swap it in). The "restart" signal (`generation` token) is awaited with
  `rt.block_on(generation.cancelled())` instead of `block_on(serve(..))`. When the registry for
  the new config fails to start, the previous `AppState` stays current (today the previous
  config is restored by re-running the loop; keep that, but the server never went down).
- In-flight requests on the old generation: they hold an `Arc<AppState>` -> `Arc<ModelRegistry>`,
  so the old workers stay alive until the last request drops the state; `reg.shutdown()` via
  `Arc::try_unwrap` already handles the "still referenced" case (workers stop through their
  token). Keep a 5 s drain: cancel `workers` only after `generation` has been cancelled and
  either the old `Arc<AppState>` strong count is 1 or 5 s passed.
- Port change in the config: binding is per process now. `port` moves to the "process restart"
  list in `config_merge::pending` (same bucket as `log_path`) and the Config page says so.
- Remove the `AddrInUse` retry loop; a refused bind is a startup error again.
- Tests: `tests/enabled_models_http.rs` restart test gains an assertion that a request issued
  *during* the restart (spawned right after `POST /config/restart`) gets an HTTP answer, not a
  connection error. Unit test for `Shared::swap` + a handler reading the new state.

### 1.4 Admin routes answered from loopback only

Files: `src/server.rs` (router, new middleware), `src/config.rs` (two fields), `src/cli.rs`
(flags), `README.md`, `templates/config.html` (two inputs).

- Config: `bind_address: IpAddr` (default `0.0.0.0`; the v6 loopback listener is still added
  when the address is unspecified) and `allow_remote_admin: bool` (default `false`).
- `serve_on` uses `axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())`.
- Middleware `admin_only` applied with `route_layer` to every route except
  `/v1/vision/*`, `/v1/status/updateavailable`, `/static/*`, `/favicon.ico`, `/prometheus`,
  `/stats.json`, `/v1/devices`, `/v1/config` (read-only JSON stays reachable). It reads
  `ConnectInfo<SocketAddr>`; a non-loopback peer gets 403 with a short HTML/JSON body saying
  which setting enables remote administration, unless `allow_remote_admin`.
  `/` (welcome) stays open; it links to the admin pages, which return the 403 page.
- Tests in `server.rs`: router tests with `ConnectInfo` set to `127.0.0.1` and `10.0.0.5` for
  `/config` (GET and POST), `/v1/fs/browse`, `/v1/resources/export`, `/config/restart`, and
  one detection route from the remote address (must pass through).

---

## Phase 2 — pipelined worker

Goal: on the UHD 630, inference (~40-80 ms per frame) never waits for JPEG decode + letterbox
(~10-20 ms for 1080p on one core); and on a fast dGPU the CPU side runs on several cores. Target
is measured, not assumed: before/after numbers from `benchmark` and
`test-blue-onyx-prism --repeat 200 --parallel 6` on the i5-8500 box go into the PR description.

Files: `src/worker.rs` (most of it), `src/backend/mod.rs`, `src/backend/device.rs`,
`src/backend/ort.rs`, `src/model/mod.rs` (`NamedOutput` buffers), `src/registry.rs` (pool
handle), `src/config.rs` (`preprocess_threads`), `tests/postprocess.rs` (unchanged API).

### 2.1 Decode + preprocess pool

- One process-wide pool (not per model): `std::thread` x `preprocess_threads` (config, default
  `num_cpus::get().clamp(2, 6)`), fed by a bounded `crossbeam_channel` of `PrepJob`. Owned by the
  registry (`Arc<PrepPool>`), shared by every worker; dropped with the registry generation.
- `PrepJob { image: Bytes, image_name, min_confidence, geometry: PrepSpec, reply: Sender<Prepared> }`.
  `PrepSpec` is what `Preprocessor::for_family` needs: input size, resize mode, normalization.
  The pool thread keeps a small `HashMap<PrepSpec, Preprocessor>` so buffers are still reused
  (one preprocessor per distinct model geometry per pool thread).
- `Prepared { chw: Arc<[f32]> or a pooled Vec<f32>, ctx: PreprocessCtx, img: Option<RgbImage>,
  decode_ms, preprocess_ms }`. The decoded `RgbImage` is kept only when `save_image_path` is
  set (it is needed for the annotated image); otherwise it is dropped on the pool thread.
- Tensor buffers: a `TensorPool` (`Mutex<Vec<Vec<f32>>>` keyed by length) so steady state does
  not allocate 4.9 MB per request. `Prepared` returns its buffer on drop.

### 2.2 Two in-flight inferences per model

- OpenVINO: `OvBackend` holds two `InferRequest`s (+ their input tensors) created from the same
  `CompiledModel` (cheap; the compiled model is shared). `Backend::infer` is replaced by
  `Backend::submit(slot, chw, extra) -> Result<()>` (`infer_async`) and
  `Backend::collect(slot) -> Result<Vec<NamedOutput>>` (`wait` then read outputs). The crate
  exposes `InferRequest::infer_async` and `wait` (openvino 0.11, `request.rs`).
  Note: `apply_core_properties` sets `PERFORMANCE_HINT=LATENCY` today; with two requests the
  GPU plugin may serialize them, which is still fine (the win is overlapping with the CPU side).
  Measure `THROUGHPUT` too and keep whichever is faster at 2 in flight; do not go beyond 2 on the
  iGPU (latency grows with no throughput gain).
- ONNX Runtime: `Session::run` takes `&mut self` in ort 2.0.0-rc.13, so a second in-flight run
  on the same session needs `run_async` (its own thread-safety comment is discouraging) or a
  second session (doubles model memory). Decision: ORT stays at one in-flight inference per
  model in this phase; it still benefits from 2.1 (decode/preprocess overlap). `submit`/`collect`
  on `OrtBackend` run synchronously inside `submit` and `collect` just returns the stored
  result. Revisit with a second session as an opt-in (`ort_sessions: 2`) if a dGPU user asks.
- Worker loop becomes a small state machine over `slots: [Option<InFlight>; 2]`:
  1. Receive a `WorkItem::Detect`; check timeouts as today; send a `PrepJob` to the pool with a
     per-request `oneshot` (sync `crossbeam` oneshot-equivalent: a bounded(1) channel).
  2. Keep a FIFO `pending: VecDeque<(reply, enqueued, prep_rx)>` of requests waiting for prep.
  3. Loop: when a slot is free and the head of `pending` has its `Prepared`, `submit` it into the
     slot; when the oldest slot is busy, `collect` it (blocking wait on the GPU), postprocess on
     the worker thread (cheap), reply. Order is preserved per model (FIFO), which keeps
     `analysisRoundTripMs` monotone for Blue Iris.
  4. `recv_timeout(POLL)` on the work channel is replaced by `crossbeam_channel::select!` over
     the work channel, the prep replies, and a tick, so the worker never spins and never sleeps
     while work is ready.
- Soft capacity: `auto_queue_size` keeps its formula; `warmup_ms` is now the warm-up inference
  only (decode/preprocess are off the thread). Hard cap stays `MAX_AUTO_QUEUE`.
- Postprocess stays on the worker thread (NMS is sub-millisecond at these sizes).

### 2.3 Output buffers

- `read_output` (`device.rs:654`) and its ORT twin copy into reusable `Vec`s owned by the slot:
  `NamedOutput.data` becomes an `OutputBuf` that is taken from and returned to the slot. The
  families' `postprocess(&[NamedOutput], ..)` signature is unchanged. `NamedOutput::as_i64`
  gets a `Cow`-free path: return `&[i64]` when stored as i64, else convert into a scratch
  buffer owned by the family call (rtdetr is the only user).

### 2.4 Verification

- `tests/postprocess.rs` unchanged (API stable).
- New `tests/pipeline.rs` (skips without `models/` + `openvino/`): 200 requests with 6 parallel
  clients through `/v1/vision/detection`; asserts all succeed, no drops, and that per-request
  responses keep per-model FIFO order (`image_name` sequence numbers).
- Benchmark page/CLI: nothing changes (it runs the serial pipeline deliberately, as the doc in
  `benchmark/mod.rs` says); add a one-line note that service throughput is higher than the
  benchmark's per-image latency implies.
- Numbers to record in the PR: requests/s and p95 round trip at 6 clients, for `openvino:gpu`
  and `openvino:cpu`, before and after, on the i5-8500 box.

---

## Phase 3 — hot apply and per-model reload

Files: `src/config_merge.rs` (shrinks), `src/config_store.rs`, `src/server.rs` (config
handlers), `src/worker.rs`, `src/registry.rs`, `src/runner.rs`, `templates/config.html`,
`assets/live.js`, `tests/enabled_models_http.rs`.

### 3.1 Live settings

- `pub struct LiveSettings { confidence_threshold, nms_iou, object_filter: Vec<String>,
  save_image_path: Option<PathBuf>, save_ref_image, request_timeout: Duration }` per model
  (after the per-model override is resolved), held as `Arc<ArcSwap<LiveSettings>>` on the
  `WorkerHandle` (add the `arc-swap` crate; a std `RwLock` read per request is acceptable if
  the dependency is unwanted). The worker loads it once per request.
- `ModelRegistry::apply_live(&Config)` recomputes each enabled model's `LiveSettings` and swaps
  them; it also updates `default_idx` (behind an `AtomicUsize`/`RwLock`) so `default_model`
  applies at once.
- `save_config` calls `state.registry.apply_live(&new_config)` after a successful write; the
  change log entry says which fields applied live.
- `config_merge::live_global` / the per-model classification grows accordingly; `pending()`
  returns only: runtime-level globals (`port` [process], `bind_address` [process], `log_path`
  [process], `openvino_dir`, `onnxruntime_dir`, `cache_dir`, `intra_threads`, `download_dir`,
  `models_dir`, `force_cpu`, `gpu_index`, `device`, `auto_download`, `allow_large_downloads`,
  `worker_queue_size`, `preprocess_threads`) and per-model structural changes (3.2).

### 3.2 Per-model reload

- `ModelRegistry` holds `workers: RwLock<Vec<WorkerSlot>>` where a slot can be replaced.
  `registry.reload_model(name, &Config)` stops the old worker (drop sender, cancel its own
  child token, join on a helper thread so the handler does not block) and spawns a new one
  from the same `plan` as startup (phase 5.2 extracts that plan; until then, factor the
  per-model block of `start_with_benchmark` into `fn spawn_model(..) -> WorkerHandle`).
  `enable`/`disable` are `spawn_model` / stop. The LoadGate chain is only for startup order;
  reloads do not wait on it.
- `POST /config/models` and the device dropdown call `reload_model` instead of asking for a
  restart; the response names the model and "reloading on <device>". The header's pending
  counter only counts 3.1's runtime-level fields now.
- A device change that needs a download keeps today's path (queue the download; the
  provisioning watcher triggers a reload of that model instead of a whole generation: change
  `spawn_watcher` to call `registry.reload_model` for the blocked models).
- Adding a model by path / "Add to config" / export-done: `spawn_model` immediately.

### 3.3 Last write wins

- Delete the 3-way merge (`config_merge::merge_form`, `Conflict`, the 409 response and the
  conflict card in `config.html`, the `base` hidden fields in the forms and the matching code in
  `live.js`). Keep `describe_changes`, `pending`, the field flattening used by the change log.
  Expected: ~700 lines removed, `config_merge.rs` ends around 400 lines with tests.
- The form posts the full config section it edits; the server applies it as is. The Config page
  still refreshes untouched fields from `/v1/config` polling (that part of `live.js` stays).

### 3.4 Tests

- `config_merge` tests for the new `pending()` classification.
- HTTP test: change threshold through `POST /config`, then a detection returns with the new
  threshold without restart (use the IPcam model at 0.9 vs 0.1 on the test image).
- HTTP test: disable a model, `/v1/vision/custom/list` drops it within 1 s; enable again, it
  loads.

---

## Phase 4 — observability and CI

Files: `src/metrics.rs`, `src/runner.rs`, `src/registry.rs`, `.github/workflows/ci.yml`,
`.github/workflows/release.yml`, `docs/RELEASE_CHECKLIST.md` (new), `README.md`.

### 4.1 Process-lifetime metrics with histograms

- `Metrics` is created once in `run_server_with` (outside the loop) and passed to every
  generation; `Metrics::model(name)` returns the existing `Arc<ModelMetrics>` or creates one
  (keyed by normalized name) so counters survive restarts. `requested_device` / provider become
  `RwLock`ed fields already present.
- `Stat` becomes `Histogram { buckets: [AtomicU64; 13], sum: AtomicU64, count: AtomicU64 }`
  with bounds `[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, +Inf]` ms; `record(ms)`
  is one `fetch_add` on the first bucket `>= ms` plus sum/count. Keep `min`/`max` for the
  Stats page. Prometheus text: `_bucket{le=..}`, `_sum`, `_count` with `# TYPE .. histogram`,
  for `inference_ms`, `process_ms`, `round_trip_ms` (names become `_seconds` per Prometheus
  convention? No: keep `_ms` to stay compatible with existing blue-onyx dashboards; note it in
  the README).
- Counters: `requests_total`, `dropped_total{reason="queue_full"|"timeout"|"initializing"|"failed"}`,
  `restarts_total`. Gauges unchanged (`queue_length`, `queue_capacity`, `model_ready`).
- Tests: the existing `metrics.rs` tests updated; a test that `record` lands in the right
  bucket at the boundaries.

### 4.2 CI

- Smoke job additions after the OpenVINO smoke test, same server config file:
  - All OSes: `cargo run --release -- setup-onnxruntime --flavor cpu`, start with
    `--device ort:cpu`, run the same curl + client checks; assert the `executionProvider` in the
    detection JSON contains "ONNX Runtime".
  - macOS: a second run with `--device ort:coreml` (the runner is Apple Silicon). The IPcam
    YOLOv5 model is the one the CoreML rewrite path exists for; assert `executionProvider`
    contains "CoreML".
  - Windows: `setup-onnxruntime --flavor directml`, `--device ort:directml`, allowed to fail
    with `continue-on-error: true` for the first week; if it passes on WARP consistently, make
    it required.
- A `self-test` subcommand is *not* added: `benchmark --all-devices` already is the on-machine
  verification.
- `docs/RELEASE_CHECKLIST.md`: tag steps from `release.yml` plus "run
  `blue-onyx-prism benchmark --all-devices --report cuda.html` on the NVIDIA box (GTX 1070 Ti),
  confirm every device's agreement with the CPU reference is `ok`, attach the report to the
  release".

---

## Phase 5 — structural splits (behavior-neutral)

Files: `src/server.rs` -> `src/server/{detect,config_page,resources,pages}.rs`; `src/registry.rs`.
No functional change; `git diff --stat` should show moves, and every existing test passes without
edits other than `use` paths.

### 5.1 `server.rs`

- `server/detect.rs`: `read_multipart`, `parse_detection_body`, `detection_default`,
  `detection_custom`, `run_detection`, `dispatch`, `custom_list`, `BODY_LIMIT`.
- `server/config_page.rs`: `ConfigView`, `ConfigTemplate`, `ModelSelectRow`, `BenchHint`,
  `DeviceChoice`, `device_choices`, `model_choices`, `model_device_choices`, `config_template*`,
  `config_page`, `config_submit`, `config_models`, `config_restart`, `config_loglevel`,
  `save_config`, `apply_log_level`, `schedule_restart`, `restart_response*`.
- `server/resources.rs`: `resources_json`, `resources_download`, `resources_remove`,
  `resources_add_to_config`, `models_add`, `resources_export`, `resources_export_cancel`,
  `add_export_to_config`, `action_fields`, `action_response`, `no_manager`, `family_*`.
- `server/pages.rs`: `welcome`, `stats_page`, `stats_json`, `prometheus`, `test_page`,
  `test_submit`, `logs_page`, `logs_json`, static assets, `fallback`, the small view structs.
- `server.rs` keeps `AppState`, `Shared`, `router`, `bind`/`serve_on`, `render`, `wants_html`,
  `DevicesSnapshot`/`devices_snapshot`, `state_label`/`state_view`, the admin middleware.
- Tests move with their handlers (the `mod tests` in `server.rs` is ~1800 lines; split it the
  same way).

### 5.2 `registry.rs`

- `pub struct ModelPlan { name, path, device: DeviceSpec, load: LoadRequest, classes, live:
  LiveSettings, lazy, wait: Option<Wait>, outcome: PlanOutcome }` with
  `enum PlanOutcome { Spawn, Failed(String) }`.
- `pub fn plan_models(config, runtimes: &Runtimes, provision, best_cpu) -> Result<Vec<ModelPlan>>`:
  pure (no threads, no OpenVINO calls beyond `runtimes.plan`/`selection`), covers name
  validation, `default_model` resolution, `manual_failure`, interim-device decision, pending
  export, missing file, class names. Unit tests cover each outcome with a `Runtimes` built from
  `select::RuntimeProbe` fixtures (no real runtime; see `tests/select.rs` for how probes are
  faked).
- `spawn_model(plan, runtimes, metrics, shutdown, after) -> WorkerHandle` is what phase 3.2
  reuses for reloads; `start_with_benchmark` becomes `plan_models` + a loop of `spawn_model`.

---

## Order and parallelism

1. Phase 1 (porter agent; 1.3 may need an engineer for the `Shared` swap) and phase 4
   (porter) in parallel. Disjoint files except `runner.rs` (1.3 changes the loop, 4.1 moves
   `Metrics::new` out of it): phase 4 rebases on phase 1.
2. Phase 2 (engineer agent) after phase 1 is merged (it relies on 1.1's `WorkItem` enum and the
   single timeout). Needs the before numbers from the i5-8500 box first.
3. Phase 3 after phase 2 (worker loop is stable by then).
4. Phase 5 last.

Each PR ends with the attribution line from CLAUDE.md.
