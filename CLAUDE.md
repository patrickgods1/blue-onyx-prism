# Blue Onyx Prism

Cross-platform object-detection HTTP service for Blue Iris (CodeProject.AI-compatible API),
written in Rust on **native OpenVINO** via the `openvino` crate (runtime-linking), with
**ONNX Runtime** (`ort` crate, load-dynamic) as a second runtime for NVIDIA (CUDA/TensorRT),
DirectML and CoreML. Modeled on [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT), with
multi-model support, `auto` device selection by detected hardware, and a dataset collector, review,
auto-labelling and training pipeline for custom models.

Primary target: Windows 11, Intel i5-8500 + UHD 630 iGPU. Must also build/run on Linux x86_64 and macOS arm64.

Where detail lives: `README.md` (users: install, every config key, HTTPS / accounts / network rule, devices, API,
benchmark), `docs/TRAINING.md` (import, auto-label, overlaps, dataset check, Refine tab, Train area, training, CLI),
`docs/DATASET.md` (collector layout, sidecar and label file formats, Review queue, export), `docs/NIGHT_IR.md`,
`docs/BENCH_IMAGES.md`, `docs/RELEASE_CHECKLIST.md`, `docs/PRODUCTION_REVIEW.md` (audit findings, measurements).
`docs/PLAN.md` and `docs/REVIEW_PLAN.md` are history and may not match the code. Every source file starts with a
`//!` doc stating its responsibility (constants and their rationale are documented there); `src/lib.rs` has the module map.

## Build & run

```powershell
cargo build --release
cargo run -- setup-openvino                 # OpenVINO runtime libs into ./openvino (next to exe)
cargo run -- setup-onnxruntime [--flavor auto|cpu|cuda|directml] [--dir <d>]   # ORT libs into ./onnxruntime
cargo run -- list-devices                    # runnable device options and the `auto` pick
cargo run -- download-models --name IPcam-general
cargo run -- add-model --path <file|dir> [--name n] [--family rfdetr] [--classes c.yaml]   # custom model
cargo run -- fetch [--for-config] [--resource <id>] [--all-for-platform] [--allow-large]   # pre-download what the config needs
cargo run -- fetch --resource model:yolo26s --allow-large   # on-demand model export (pinned uv + Python env in <data root>/tools)
cargo run -- list-resources [--check-urls]   # installed / needed / available; --check-urls checks every pinned URL (weekly CI)
cargo run -- --model models/IPcam-general.onnx --family yolo5 --force-cpu
cargo run -- --model models/IPcam-general.onnx --family yolo5 --device ort:cpu
cargo run -- benchmark --all-devices [--apply] [--report r.html] [--dataset coco-cctv|bmd45-cctv|exdark-night|dir:<p>] [--max-images 20]
cargo run -- benchmark --list-datasets       # built-in image sets (assets/bench/*.json, resource bench:<id>)
cargo run -- train refine [--dry-run] [--with-model <name>]   # every `train` subcommand: docs/TRAINING.md, "Command line"
cargo run -- train autolabel --sam3 [--dry-run]   # --dry-run previews frames and a rough time, runs nothing
cargo run -- train retry-export <run>        # a failed run's kept checkpoint: export, install, benchmark again
cargo run -- admin-token [--rotate] [--config <file>]   # API token for scripts / Prometheus (stdout holds nothing else)
cargo run -- user add|remove|reset-password|unlock <name> | user list   # web UI accounts; passwords never in argv
cargo run -- setup-code | tls-fingerprint | tls-regenerate   # first-account code; cert SHA-256 + URLs; new self-signed cert
cargo run -- hf-login | hf-logout            # Hugging Face token for gated downloads (no echo or stdin, never argv)
cargo run -- service status|start|stop|restart   # control the installed service (start/stop/restart need admin)
cargo run -- diagnostics                     # bug-report zip (never tokens, frames or labels)
cargo run --bin blue-onyx-prism-tray -- [--port n] [--cli <path>]   # tray / menu bar app (Windows, macOS)
cargo run --bin test-blue-onyx-prism -- --rate <r/s> --duration <s>   # load test
cargo run -- --version                       # BLUE_ONYX_PRISM_GIT_SHA / _BUILD_DATE set at build time by release.yml, `unknown` locally
cargo build --no-default-features            # OpenVINO-only build (drops the `onnxruntime` feature / `ort`)
cargo test                                   # unit tests need no OpenVINO
cargo clippy --all-targets -- -D warnings
cargo fmt --check
python3 -I scripts/ci/check_deps.py consistency|upstream|audit|conflicts   # Python pins (docs/RELEASE_CHECKLIST.md)
python scripts/research/eval_autolabel.py --self-test   # auto-label scoring (docs/TRAINING.md, "Measuring auto-label")
```

Commands that act on a service (`admin-token`, `user`, `setup-code`, `tls-*`, `hf-login`) take `--config <file>` and
must run as the service's account.

**Runtimes and libraries.**
- OpenVINO is found through `openvino-finder`: `OPENVINO_INSTALL_DIR` (Windows; on Unix `backend/libs.rs` loads
  `openvino_c` by path) pointing at `<download root>/openvino` when present, else the OS library path.
- ONNX Runtime: `onnxruntime_dir` in the config, else `ORT_DYLIB_PATH`, else `onnxruntime/<flavor>/` named by
  `onnxruntime/active.txt`, else the old flat `onnxruntime/`, else any flavor dir. A process loads one ORT flavor only.
  `onnxruntime/cuda-libs` (opt-in NVIDIA wheels, ~1.9 GB) is preloaded by path before the CUDA provider loads.
- Pins are per flavor in `src/resources/catalog/` (`ONNXRUNTIME_VERSION` for cpu/directml/coreml: the last release with
  a DirectML package; `ONNXRUNTIME_CUDA_VERSION` for cuda; `DIRECTML_VERSION`; `CUDNN_VERSION`, which must stay < 9.11
  while the CUDA 12 lane covers Pascal, a test checks it). Which compute capabilities each ORT CUDA build has kernels for:
  `catalog::ORT_CUDA_BUILDS` (by OS and version).
- The `ort` crate is built without an `api-NN` feature, so any ORT 1.17+ library loads; never add one above the
  oldest pinned flavor.
- Updates: an outdated runtime is updated in place, or, while a process has it loaded, staged as
  `<parent>/.<name>.pending` and swapped in by the next process before it loads (`manager::in_use`: every loader holds a
  shared lock on `.<name>.in-use`; covers OpenVINO, ORT flavors and `cuda-libs`). A pin bump reaches existing installs
  (OpenVINO and cuda-libs record their version; the old one stays usable; a large one without `allow_large_downloads` is
  only offered); a catalog model whose manifest names another version is re-downloaded; a model without a manifest stays.

**On-demand resources.** Pins, URLs, SHA-256 and sizes of everything downloadable live in `src/resources/catalog/`;
change them only in reviewed commits. The service downloads what config + hardware need at startup (`auto_download`,
default true; `allow_large_downloads`, default false, gates > 500 MB; `download_dir`, default the exe dir).

**Model export** (`src/resources/export/`, the only code that runs downloaded programs; user-triggered only, never by
`auto_download`): `tool:uv` + hashed `scripts/model-export/locks/*.txt` (regenerate with `gen_locks.sh` /
`gen_train_locks.sh`) + the embedded `scripts/export_model.py`, all under `<data root>/tools/`.
- Recipes (`catalog::ExportRecipe`): Ultralytics YOLOv8/9/10/11/12/26 (AGPL-3.0), RF-DETR L, DEIMv2 (code pinned to
  the last Apache-2.0 commit; marked non-commercial, the upstream licence now covers the weights), MegaDetector v6, the
  ExDark RF-DETRs. Static-shape ONNX checked against PyTorch with ONNX Runtime, stamped with `COREML_CACHE_KEY` (ORT's
  CoreML cache is otherwise keyed by path).
- A manifest of another exporter version = "needs export" (not loaded), except one listed in
  `catalog::EARLIER_EXPORTERS`. `catalog::F32_MODELS` overflow f16: OpenVINO compiles them in f32 on CPU and GPU.
- By hand: `uv venv --python 3.11 .venv && uv pip install -r scripts/requirements-export.txt` (the service's `==` pins
  plus `openvino`), then `.venv/bin/python scripts/export_model.py --name yolo26s` (Windows: `.venv/Scripts/python.exe`).

**Device spec** (`device` in config / model entry, `--device`): `auto` (default), `openvino:gpu[.N]`, `openvino:cpu`,
`openvino:npu`, `ort:cuda[:N]`, `ort:tensorrt[:N]`, `ort:directml[:N]`, `ort:coreml`, `ort:cpu`; legacy `GPU`/`GPU.N`/`CPU`
mean `openvino:*`. The `auto` ranking (discrete GPUs by VRAM, then integrated, CoreML, CPU last; CUDA skipped for a GPU the
build has no kernels for) is in README "Hardware and device selection". A model with results in `benchmark.json`
orders its `auto` devices by measured speed. NPU is never auto. ORT needs `.onnx` files. `OrtProbe.version` is the
loaded library's version (the resolver's hypothetical probe carries the version the plan installs).
- CPU inference uses all physical cores but one (`backend/cpu_threads.rs`); ORT thread pools do not spin; benchmarks
  and training helpers run below normal priority (`backend/priority.rs`).
- Windows CPUs with AMX but no AMX-FP16 get `ONEDNN_MAX_CPU_ISA=AVX512_CORE_FP16` unless set (OpenVINO's AMX kernels
  raise #UD there; `backend/cpu_isa.rs`). RT-DETR runs on CoreML only with every input dim pinned (a dynamic batch dim
  aborts MPSGraph).

## Architecture

- **Layout.** Large modules are directories split by concern (`benchmark/`, `server/`, `resources/`, `training/`,
  `collect/labels/`, `backend/onnx_rewrite/`, `worker/`, ...).

- **HTTP** (`src/server.rs` + `src/server/`): axum on a 2-thread tokio runtime (`runner.rs`); fs-heavy handlers run in
  `server::blocking`. The listener is bound once per process (`bind` + `serve_on`) and serves `Shared`, the swappable
  current `AppState`, so it survives generation restarts.
  - Blue Iris calls `POST /v1/vision/detection` (default model), `POST /v1/vision/custom/{model}`,
    `POST /v1/vision/custom/list`. Responses use `api.rs` structs (camelCase; `Prediction` snake_case), byte-compatible
    with CodeProject.AI (see Conventions).
  - Overload: a process-wide in-flight cap answers 503 (`metrics.rs`, `requests_rejected_total`; overload 503s drain the
    upload, `OVERLOAD_DRAINS`); uploads still arriving have a 10 s deadline, a per-address read limit
    (`BODY_READS_PER_PEER`, never a cap on one Blue Iris's total requests) and a shared byte budget
    (`server/body_budget.rs`). `GET /v1/status` is readiness (200/503; Docker HEALTHCHECK, systemd `Type=notify` +
    watchdog via `sdnotify.rs`).

- **Network, TLS and sign-in** (security boundary; user view in README "HTTPS", "Local network only", "Accounts",
  "Signing in").
  - Only the local network is answered (`server/network.rs`: loopback, RFC 1918, link-local, ULA, plus config-file-only
    `allowed_networks`): closed at accept by `guard::admit`, else 403 from the outermost middleware
    `guard::network_rule`. Forwarding headers (the one list in `guard::proxy_header`) can only refuse.
  - Two listeners (`server/listen.rs`), one router, marked `Transport::{Plain,Tls}` (none = plain): plain HTTP on `port`
    (startup error, `/v1/status`, `READY=1` and the Windows service's Running follow it) and HTTPS on `https_port`
    (retried every 30 s while its port or certificate fails; bound port in `Shared::https_port`). At accept:
    `guard::admit`, then connection caps, then the TLS handshake (10 s) in the connection task.
  - Plain serves `open_routes` only (`server::plain_gate`: browser GET/HEAD 308 to HTTPS, foreign host 403, HTTPS down
    503, else 403 JSON; `Authorization` there is refused uncounted; `guard::legacy_cookie` clears `bop_admin`; no HSTS).
    Every redirect target goes through `safe_next` (a refused non-GET's `next` is its same-host `Referer`,
    `guard::referer_on_host`). Icons and the manifest stay open routes: browsers fetch them without cookies.
  - Everything else is HTTPS and signed in, every address alike (`admin_only` -> `auth::authenticate`): a
    `__Host-bop_session` cookie or `Authorization: Bearer <API token>` (`admin_token.rs`, read per check, own
    `admin_failures` bucket). `X-Prism-Poll: 1` (or the `__Host-bop_poll` cookie) marks a poll that leaves the idle
    clock alone. `users.json` missing -> browsers 303 `/setup`; broken -> 503, never setup. Unauthenticated answers are
    `no-store`; HTML pages get the user name at `<!--bop:user-->` and the zone at `<html data-tz="<!--bop:tz-->">`
    (`auth::personalize`; the outermost `auth::fill_zone` fills the zone on pages without a session).
  - Routes answer only under an IP address, `localhost` or an `allowed_hosts` name (`guard::foreign_host`, DNS
    rebinding; `GET /v1/config` is open and applies the Host check and sign-in itself). CSRF on TLS: `Origin`/`Referer`
    must be `https://<Host>`, `Sec-Fetch-Site: same-site` refused. `direct_local` (loopback, no forwarding header) is
    only an extra condition, for `/v1/fs/pick` and token rotation.
  - `/login`, `/logout`, `/setup` (`login_routes`): per-address `FailureLimit::reserve` before hashing, lockout decided
    with in-flight attempts counted as failures (`accounts::process::Attempts`), a `Logins` permit, the result written
    after the answer (`auth::record_later`), a new session id each time. The Config page's Accounts card
    (`server/accounts.rs`) re-checks the current password through the same path (`server::accounts::reauth`; wrong =
    403, never 401; the API token is refused).
  - Accounts (`src/accounts/`): `auth/users.json` fails closed (only a missing file is setup mode), Argon2id behind a
    hashing semaphore, every `auth/` read-modify-write under `auth/.lock` and never while hashing. Sessions and device
    cookies (`accounts/sessions.rs`, `devices.rs`, `cookies.rs`) store only SHA-256 of their ids. One `accounts::Auth`
    and one `Logins` per process (`runner`), flushed at shutdown.
  - Every auth file (accounts, sessions, devices, setup code, API token) is under `Auth::data_root()`, the start's data
    root, never `running()` (a generation restart that changes `download_dir` only warns).
  - TLS (`tls.rs`): self-signed ECDSA P-256 in `<data root>/auth/tls/` (`private_dir.rs`: 0700/0600, protected DACL),
    stable, renewed under 30 days left, reloaded on file change; own `tls_cert_path`/`tls_key_path` are config-file only
    and never written. `tls::pinned_client` (reqwest pinned to the cert's SHA-256) serves the CLI's `diagnostics`, the
    load test and `tests/common`'s `admin_client`. No page shows the fingerprint (it would prove nothing).
  - `allow_remote_admin` is retired (read, `skip_serializing`, warned; true at start = one token rotation, marker
    `auth/.token-rotated-v1`).
  - Unit tests: `test_util::from_peer` adds `Transport::Tls` and the seeded `TEST_SESSION` (`seed_tester`, temp data
    roots only) unless `anonymous` / `plain` / `bearer` / `as_peer(PUBLIC)`.

- **Registry and workers** (`src/registry.rs`, `src/worker/`, `startup.rs`, `supervisor.rs`).
  - The registry loads every enabled model through shared `Runtimes` (one OpenVINO `Core`, one ORT library), in config
    order, one compile at a time; each model gets its own worker thread and bounded crossbeam channel. `startup.rs`
    tracks Initializing/Ready/Failed so HTTP serves while models compile.
  - A reload or restart loads the replacement beside the old worker, which serves until it is Ready (`Slots::pending`,
    `finish_replacement`); a memory failure retires the old worker and tries again (`LoadFailure` in
    `backend/plan.rs`, `make_room`). A Failed or stuck model is retired first. A model on a fallback after a memory or
    device-lost failure retries the better candidate (`retry_better`). `worker/reap.rs` frees abandoned backends.
  - Health: inference that never returns -> Failed("inference stuck ...") (`worker::watch::collect_watched`); repeated
    errors or all-zero/NaN outputs (a reset GPU) -> Failed("inference broken ...") (`note_health` in `worker/run.rs`);
    the generation's `supervisor` reloads models that failed while serving, with a doubling backoff. A warm-up past
    `worker::run::WARMUP_TIMEOUT` abandons that device (backend leaked, next candidate tried).
  - Logs: per-request debug line (`model/device/request_id/inference_ms/process_ms/round_trip_ms`) and an
    `X-Request-Id` header; `log_format: text|json` (file appender only); `cli::log_files` caps size and age, today's
    file never deleted. Error messages state what happened, the likely cause and the next action.

- **Backend** (`src/backend/`, the only place `unsafe` is allowed): runtimes behind a `Backend` enum
  (`OpenVino | Ort`). OpenVINO wrapper (properties CACHE_DIR / LATENCY / f16 on GPU; cache dir
  `<cache>/openvino-<device>-<prec>`, the CPU cache hash ignores precision); `ort.rs` holds all `ort` code; `spec.rs`
  parses device specs; `detect.rs` finds GPUs (DXGI / sysfs / Apple); `select.rs` builds the ranked, pure `DeviceOption`
  list and the `auto` pick; `plan.rs` turns it into per-model load candidates (compile + warm-up, falling down the list,
  CPU last); `vram.rs` reads free GPU memory live (NVML / DXGI / amdgpu sysfs, never a CUDA context); `memory.rs` the
  commit charge.
  - `onnx_rewrite/`: `ort:coreml` loads a cached copy without HardSigmoid/HardSwish/opset<13 Split (protobuf
    wire-level, `cache/coreml/`, `BOP_COREML_REWRITE=0` off).

- **Model families** (`src/model/`): `yolo26` (YOLO26/YOLOv10 end-to-end `[1,300,6]`), `yolo5` (`[1,N,5+C]` + NMS), `yolo8`
  (YOLOv8/9/11/12 `[1,4+C,A]` + NMS), `detr` (D-FINE, RT-DETRv2 trained here: `pixel_values` -> `logits/pred_boxes`), `rfdetr`
  (same ports, ImageNet-normalized), `rtdetr` (RT-DETRv2, DEIMv2: `images` + i64 `orig_target_sizes`; `labels/boxes/scores`),
  `auto` (picked from the output shapes).
  - Preprocess: letterbox (YOLO) or stretch (DETRs) to the model's input size, RGB f32 0..1, CHW. Static input dims win;
    dynamic ones use `model::dynamic_input_size` (catalog hint, HF `preprocessor_config.json`, RF-DETR 560, else 640).
  - `model/name.rs` (`normalize_name`) and `model/coco.rs` (`coco_class`, `same_class`, `map_label`) are std-only leaves;
    parity tests cover `scripts/autolabel.py`, `train_model.py` (one extra alias on purpose), `export_model.py` and
    `assets/coco_classes.yaml`.

- **Night / IR and verify** (docs/NIGHT_IR.md).
  - `image_stats.rs` gives each request a `Scene` (day|night|ir, computed in `prep.rs` after decode); `model/filters.rs`
    (night / per-label thresholds, aspect, min area, overlap suppression; all default off, applied in
    `worker::run::finish` before `to_predictions`); `static_memory.rs` (per-camera static-box suppression);
    `model/enhance.rs` (`preprocess` / `night_preprocess`; the kept frame is always the original).
  - `night_model` routes night/IR frames, decoded, to another model's worker (`WorkItem::DetectDecoded`, weak
    `worker::Peers`; never re-forwarded; falls back to itself); `night_threshold` keeps the model and replaces its
    threshold (`LiveSettings::threshold_for_scene`).
  - `verify` (`src/verify.rs` pure decision, `worker/verify.rs` per-worker lane): night person candidates re-checked by
    another model, fail-open, never blocks the inference thread; reads `VisionDetectionResponse.dropped`, never message
    text. `benchmark --variants` (`benchmark/variant.rs`) runs stats -> enhance -> filters like the service.

- **Time zone** (`src/clock.rs`, std + chrono leaf): config `timezone` (default `local`; the retired `log_timezone` is
  read, `"utc"` = unset) for log stamps, file / folder names (run ids, dataset day folders and frame stems, exports,
  reports, log files) and every time shown (pages via `PrismTime` in `assets/time.js`, CLI, reports). Stored record
  timestamps stay UTC, and so does the Blue Iris `timestampUTC`. Pure `Zone` methods (`display`, `day`, `frame_stamp`,
  ...) plus wrappers over the process-wide zone (`clock::set` / `change` / `zone`); tests use the methods and never
  assert text from the global.

- **Config** (`src/config.rs`, `config_store.rs`, `config_merge.rs`): JSON next to the exe; CLI overrides only non-default
  values and writes back. Every UI writer goes through `save_config` (last write wins).
  - `ConfigStore` publishes an `Arc<Published>` through `ArcSwap` (`config_read()` = one atomic load, never waits for a
    save's fsync; writers serialize on `writer`, taken after `apply_lock`, never before); `config_store::spawn_disk_watch`
    applies on-disk edits within 2 s.
  - A save applies live: per-request settings swap into workers (`LiveSettings`), structural model changes reload only
    that model (`ModelRegistry::reload_model`), runtime-level fields wait for a restart (`config_merge::pending`).
  - One-off flags (`Config::one_off_flags`) are kept out of every write (`cli::without_one_off_flags`,
    `ConfigStore::update`). `download_dir`, `onnxruntime_dir`, `openvino_dir`, `allowed_networks` and the TLS paths are
    config-file / CLI only, never web-editable (a loaded library runs with the service's rights).
  - Formats: `Config.format` (`CONFIG_FORMAT`; a later one is refused at load, by the disk check and by
    `ConfigStore::update`), `task::RECORD_FORMAT` per runs/tasks record; `collect.*` warns on unknown keys.
  - Confidence: `ModelConfig.device_thresholds` win over `confidence_threshold` for the device the model actually loaded
    on (`LiveSettings::threshold_for`, so `auto` and fallbacks follow). A config with no models is seeded with
    IPcam-general.
  - Removing models: Resources Remove also drops the model's disabled config entries (an enabled one blocks it); the
    Models card's Remove (`POST /config/models/remove`) drops a disabled entry, then deletes its files only inside
    `models_dir`, never a catalog model or one other entries use.

- **Resources** (`src/resources/`): `catalog` (pins), `resolve` (pure needs from config + hardware + installed),
  `manager` (one download thread, Range resume, SHA-256, staging + atomic rename, `.installed.json`, `.downloads.lock`,
  backoff, `SERVICE_MAX_ATTEMPTS`; a runtime missing a listed file is not installed, `manager::install_damage`),
  `extract` (whitelist-only, nothing executed), `provision` (models wait with progress; a new runtime starts a new
  registry generation), `status` (`/v1/resources`), `commands` (`fetch`, `list-resources`), `export` (fake-able
  `CommandRunner` / `Fetcher`, one at a time, cancellable).
  - Gated downloads (SAM 3; `catalog::GATED`): `resources/hf_token.rs` finds the token (`HF_TOKEN`,
    `<data root>/tools/hf/token`, the `hf` CLI's file). The value is never logged, returned by a route or passed to
    Python (`inherited()` drops `HF_*`); the bearer goes only to huggingface.co; a 401/403 there is a typed refusal
    (`Refusal::NeedsLogin|NeedsAccess`); `without_url()` keeps signed CDN URLs out of logs and state.

- **Collector** (`src/collect/`, `collect` config, default off; docs/DATASET.md): after the reply the worker `try_send`s the
  upload + a v1 sidecar to one collector thread (buckets, per-camera rate limits, dHash dedup that most buckets skip,
  disk guard that re-arms; atomic writes under `<data root>/dataset/<scene>/<camera>/<date>/`;
  `retention::prune_dataset`). `collect/health.rs` feeds `GET /v1/dataset`. Collect tab `/train/collect`.

- **Data safety** (applies to every writer of user data).
  - One label write path: `labels::update` under `<dataset root>/.labels.lock` (`fs4`, CLI and service alike; a held
    lock is never given up on, only one that cannot work falls back to the process lock). A failed label write keeps the
    job's `work` and is named in its summary.
  - `task::load_or_move_aside` moves a damaged `runs.json` / `tasks.json` to `.corrupt-<time>` and leaves an unreadable
    one alone, unwritten. Runs are never dropped. JSONL reads are lossy per line and appends repair a torn tail
    (`layout::read_lines`); `layout::compact_review` and appends to `review-queue.jsonl` share `.review-queue.lock`.
  - Exports are built hidden, then renamed. A crashed auto-label's `proposals.jsonl` is written through the job's own
    path first (`autolabel::job::recover_crashed`) and kept on failure.
  - An unavailable review folder is `folder_missing`, never an empty dataset. Uninstall keeps data unless asked.

- **Review labels** (`src/collect/labels/`, `/train/review`, `assets/review.js`; formats and rules in docs/DATASET.md
  "Review labels", tiers and labelers in docs/TRAINING.md "Auto-label").
  - Per-frame `<stem>.labels.json` (v1), written only through `labels::update`. A save checks `revision` (409). A save whose
    body lacks `refine`, `overlap_suggestions` or `review_pick` keeps the stored value (`server::review::SentLists`); a
    present key, `[]` included, is the client's.
  - Frames with a human decision or boxes awaiting review are never pruned (`labels::is_protected`). Reopened frames
    (`disagreement` + `reviewed_at`) stay out of exports until saved or restored.
  - Ignore regions (`LabelBox::is_ignore`: `difficult` / unsure and `too_small` rejects) are out of YOLO and listed in
    `ignore.json`. `collect.size_floor` (`labels::under_floor`, JS twin `underFloor`) applies at export, in the dataset
    check's judge and at a run's input only; accepted/added labels are never rewritten for it, so a floor change is a
    re-export and held-out scores across it do not compare. Its only writer is `bulk::bulk_resolve_small` (undo
    `bulk_restore_small`).
  - Dismissals (`bulk::is_dismissal`: `auto_dismissed`, `spot_dismissed`, `size_floor`) and duplicates are no negatives:
    out of `hard_negatives.json`, `rejected_boxes` and the repeats history. `reject_reason` is an enum (unknown =
    `RejectReason::Other`, round-trips); `reason` stays provenance.
  - Unwanted classes = labels mapping to no `collect.classes` entry plus `collect.hidden_classes`: out of prefill and
    auto-label runs, not drawn; making a class unwanted rejects pending model/auto boxes on unreviewed frames only
    (`bulk::bulk_reject_unwanted`); reviewed frames change only on the user's confirmation.
  - Duplicates (`stacks.rs`, twin of the page's `dupGroups`): human boxes are never dropped; nested or partial overlaps
    go to a human (`auto=overlaps`), never a bulk action; `overlaps_checked` marks a reviewer's look
    (`ReviewRules.overlapsCheckedOnSave`).
  - Queue (`queue.rs`): `queue::build` is the reference walk for bulk actions, previews, the CLI, coverage and spots
    (uncertainty order). `GET /v1/review/next` alone defaults to Recommended (`recommend.rs`: reopened, event rank,
    balance bands, uncertainty; `order` is read by `server::review::next_order` only, never `Filters::from_query`);
    `auto=` and `balance=` keep their own orders. Balance weighs each camera's kept share of a class's proposals
    (`queue::proposed_labels`, `BalanceTally.precision`).
  - Random picks: in the default view only (`recommend::random_slots`) one slot in `RANDOM_EVERY` is a frame of a fixed
    hash pool (`recommend::in_pool`; nothing a model or reviewer does moves a frame in or out). The page saves
    `LabelFile.review_pick: random` on frames served as one (`ReviewRules.servedPicks`); the server drops the mark on
    a non-pool frame; an unknown value round-trips (`ReviewPick::Other`). Training scores them apart (`heldout:random`).
  - Index (`labels/index/`): only `GET /v1/review/next`, for folders of `MIN_INDEXED_FRAMES` or more, answers from it;
    it must equal `queue::build` (`index/tests.rs` `equivalence` over the filter grid; review route tests run both ways,
    `both_ways!`). `index::written` (pushed only under the label write lock) makes the next query re-read a file;
    `INDEX_CEILING_BYTES`, then walking. Walks use `layout::Walker` and `crate::parallel::map_all`; label writes stay on
    one thread. Its row sizes are asserted in tests (keep them compact).
  - Repeats (`repeats.rs`), rejected spots (`spots.rs`, `triage::apply_spots`) and parked spots (`<review_dir>/spots/`,
    own mutex, never under the label write lock): docs/TRAINING.md "Spot memory of fixed cameras".
  - Server routes: `server/review.rs` (page, `next`, labels) and `server/review/` (bulk, spots, folder, export);
    handlers return `Answer` (`Refused` via `?`). The review folder is `collect.review_dir` (default `collect.dir`);
    auto-label runs follow it.
  - Page-only decisions with no Rust twin (`refineCurrent`, `suggestionKey`, `refineForSave`, `adjustingBox`, ...) are
    tested in `tests/it/js_review_refine.rs`; `q` looks in zoomed quarters (`ReviewTouch.quarterView`).

- **Overlap suggestions** (`collect/labels/overlaps.rs`, job `training/overlaps/`, `train overlaps`; docs/TRAINING.md
  "Overlap suggestions"): assist mode only, because a blind test of automatic drops removed a real object 13 of 99
  times. **Only ever writes `LabelFile.overlap_suggestions`** (`overlaps::apply` under `labels::update`; no box state,
  status, `reviewed_at` or `overlaps_checked`). The page's `y` (`acceptOverlap`, twin `overlaps::accept`) rejects as
  `Duplicate` and never leaves the object without a box. Ported from `scripts/research/eval_overlaps.py`, which was never
  committed: `tests/data/overlaps_parity.json` is frozen and cannot be regenerated, so a difference is a change in the
  code, never a reason to edit the fixture.

- **Dataset check / Refine** (`src/training/refine/`, `train refine`, Refine tab; docs/TRAINING.md "Dataset check
  (cross-check)", "Quick check with a model", "Closer look at loose labels", "Refine tab"; field format docs/DATASET.md).
  - Fold models that never saw a camera day judge its reviewed frames (`refine/judge.rs`, pure, constants documented in
    TRAINING.md); a quick check (`--with-model`, `refine/quick.rs`) judges with a trained model the frames it never
    trained or validated on, refused rather than guessed when that cannot be proven (`quick::seen_of`,
    `quick::check_weights`).
  - **Writes only `LabelFile.refine`** (`collect/labels/refine.rs`, pure) through `refine::apply` under
    `labels::update`; answered suggestions and other kinds are kept, an unchanged result leaves the file untouched. At most
    `refine::MAX_SUGGESTIONS`, strongest first: bands relabel > add > verify > tighten (what costs a model most per
    error), then score.
  - Fold safety: `check_unseen` hard-fails when a fold frame is in the fold model's train/valid lists (by name and by
    `image_sources.json`). Fold training registers and benchmarks nothing (`NoHooks`); the check holds the training lock
    from after the selection to the last fold's write. `tighten` comes only from the closer look (`refine/tighten.rs`).
  - `refine::live` decides staleness at read time; `accept` / `dismiss` are pure; JS twins `liveRefine` / `acceptRefine`
    / `dismissRefine` in `assets/review-rules.js`, and `refine::live_json` key for key, are parity-tested in
    `tests/it/js_rules.rs`.
  - Reports: `<review folder>/refine/report.json` (full) and `refine/report-quick.json` (quick); neither replaces the
    other; a newer format is "newer", never "damaged". Server: `server/refine.rs`, `refine_spots.rs`,
    `refine_floor.rs`, `refine_coverage.rs`; pure page sentences in `globalThis.RefineFloor` (`assets/train-refine.js`,
    node-tested).

- **Train area** (`server/train_area.rs`, `templates/train/`, `assets/train-area.js`; docs/TRAINING.md "Train area"):
  nine tabs `/train/{collect,import,autolabel,review,refine,dataset,train,models,jobs}`; old `/dataset`, `/review`,
  `/training` redirect (307). `GET /v1/train/summary` + `train_area::next_step` fill the strip (`queue::tally` on its own
  thread, a stale answer served while it walks; the check steps' counts come from `server::refine::open_cached`, never
  waited for). Jobs tab: `server/train_jobs.rs`, `training/eta.rs` (`JobClock`, memory only).

- **Training** (`src/training/`, `scripts/train_model.py`; docs/TRAINING.md): user-triggered only; `capability` gates by
  GPU (CUDA lanes by compute capability + driver, MPS on Apple); a separate `tools/train-env` from
  `scripts/model-export/locks/train-*.txt`; families YOLO26/11/v8, RF-DETR, RT-DETRv2, D-FINE.
  - Dataset assembly (`training/dataset.rs`, `dataset/`): held-out chosen from all review frames before the scene filter
    (`heldout_fingerprint`; random picks never in it), validation as whole camera days (`dataset/valid.rs`), ignore
    mode (`dataset::IgnoreMode`), static cap (`collect/labels/statics.rs`, pure: persons and animals never form spots),
    parked cap, opt-in balancing, size-floor restore at the run's input. `training/preflight.rs` is the one report
    behind the page, `balance-preview` and `train prepare --dry-run`.
  - One convergence rule for every trainer: `Convergence` in `train_model.py`, Rust twin `progress::convergence_run`,
    parity-tested. `RunSettings` records what changes results (ignore mode, negative share, static cap) and
    `compare.rs` warns on mixed values.
  - Protection: a failed run keeps `work` (`training/retry.rs`, `train retry-export`; training itself does not resume);
    an OOM restarts only before 3 finished epochs; an early GPU OOM retries at half the batch, a host OOM halves the
    loader workers, never the batch (`training::workers_for`); `check_disk` refuses up front.
  - Model switch (`training/switch.rs`): Revert swaps `previous_default` (one level); a verifier is replaced only when
    named (`replace_verifier`); `Hooks::baseline` benchmarks into the run folder, never `benchmark.json`.
  - Jobs: import, auto-label, overlaps and refine are `task::Task`s on the one-at-a-time queue, recorded in
    `<training dir>/tasks.json` beside `runs.json`. A running job pauses / resumes (`PausableRunner`); paused time does
    not count toward timeouts. Epoch history is `<run dir>/history.jsonl`; readers that draw no charts use
    `TrainingService::records` (runs without history).

- **Import and auto-label** (`training/import/`, `training/autolabel/`; docs/TRAINING.md "Footage import",
  "Auto-label"): import selects frames by scene change + dHash + SigLIP k-center, licence confirmed, free-space checked.
  Auto-label proposes pending `auto` boxes (MM-Grounding-DINO + RF-DETR L, VLM / SigLIP checks); a VLM no ranks, never
  hides.
  - Reruns (`triage::apply_labelers`) decide under the write lock by the frame's status at write time: reviewed frames
    only get votes and gated missing objects (a check that never ran is no evidence), confirmed-empty and skipped frames
    only votes and the marker. `--with-model --missing-only` (`triage::apply_missing`) skips frames the model trained
    on. Undo: `bulk::bulk_restore_reopened`.
  - SAM 3 (opt-in, gated, `catalog/autolabel.rs`) takes gdino's seat; nothing here claims it is better (its full
    evaluation has not run). **SAM 3 boxes never triage `accept`** (`ACCEPT_SCORE` was tuned for gdino + RF-DETR and a
    bulk accept reaches exports), are never dismissed by tier or spot. Access UI: `server/sam3_access.rs`.
  - Tiers (`triage::tier`, constants in `triage.rs`): dismissed boxes are `state: rejected` with a dismissal reason;
    `labels::needs_decision` refuses a save that would turn a flagged box into a label.

- **Storage and memory** (`src/storage/`): the sweeper deletes only rebuildable files (caches unused 60 days by a
  `.last-used` marker, no marker = the clock starts; stale compiled models; staging and job `work` whose `.busy` marker,
  `storage::busy`, is unlocked). Never labels, datasets, exports, models, runs, a failed run's `work` (the retry
  checkpoint) or logs. GPU memory: jobs warn, never refuse, on short free memory (except a labeler that cannot fit the
  card at all, `autolabel::memory::too_big`); loaded models register as holders (`vram::hold`). `MALLOC_ARENA_MAX=2` in
  the systemd unit and the Docker image.

- **Benchmark** (`src/benchmark/`, `server/benchmark/`): grades models x devices into `benchmark.json` next to the config
  (preds in `benchmark-preds.json`).
  - `history/`: `benchmark-history.json` (recent runs per model + one pinned baseline; failed runs never recorded); pure
    `comparable` / `compare` / `trust` with the noise rules as named constants in `history/compare.rs`.
  - `lineup/`: models side by side on exactly the same images, rescored on shared classes; `comparable_models` is the
    model-vs-model rule; a set with training frames is left out of "best" (`trained_on.rs`).
  - Profiles: `POST /v1/benchmark/apply-profiles` / `--apply-threshold` save `device_thresholds`. Re-ranking saved
    results: `benchmark/rescore.rs`.
  - A benchmark runs each model's own device by default; a sweep over two or more GPU devices needs `--confirm-gpus` / `confirm_gpus=1` (a sweep has hung a machine; a
    queue job refuses it); `benchmark-breadcrumb.json` names the last device after a hang.

- **Web UI assets** (`assets/`, `templates/`, askama): debug builds read `assets/` from the source tree per request
  (`embedded!` in `server/pages.rs`; a debug binary moved off its tree answers 500 on `/static/*`); release embeds them.
  - `assets/live.js`: `PrismPoll` (chained, never two in flight, nothing while hidden; `tests/it/js_poll.rs`; polls send
    `X-Prism-Poll: 1`, reloads they make go through `PrismPoll.navigate`), the in-page sign-in dialog on a 401 that
    replays waiting requests once (native POST forms check the session first unless `data-no-auth-check`),
    `PrismDraft` (sessionStorage drafts restored only at the same revision), `PrismViewport`, `PrismPointer`,
    `PrismTip` (information only in a `title` needs a tap-to-pin `button.term`, touch has no hover). Incremental polls: `server/training/poll.rs`
    (`TrainMetrics.poll`, node parity test), `server/benchmark/poll.rs`.
  - Pure modules are node-tested from `tests/it/`: `review-rules.js` (`js_rules.rs`), `review-touch.js`
    (`js_review_touch.rs`), charts, viewport, poll.
  - Phones (`assets/style.css`, *Phones and tablets*): every `vh` height is followed by `dvh`/`svh` and every `:hover`
    rule sits in `@media (hover: hover)` (pages.rs tests check both; iOS keeps `:hover` after a tap);
    `viewport-fit=cover`, never `maximum-scale` / `user-scalable`; side paddings `max(<pad>, env(safe-area-inset-*))`.
    Wide tables become cards under 700 px with `table.rcards` + `data-label`. Selects clip their text on touch widths
    (WebKit otherwise widens the page). The
    Browse dialog is `templates/file_browser.html`, not `browse_dialog.html` (askama would include
    `train/browse_dialog.html` into itself). Page tests assert `<td>name</td>`, `<td class="state">` and the Stats
    table's class string, so add classes there with care. Icons: `scripts/gen_icons.py` (`--check` in a test and CI).
  - Browse: `/v1/fs/pick` (`server/pick.rs`, native dialog, loopback only, 409 without a desktop) falls back to
    `/v1/fs/browse` (`server/browse.rs`, roots only: data root, models, downloads, home, mounted drives, never a
    symlinked mount); a natively picked path passes `browse::allowed_path` for 1 h.

- **Packaging and services**: Windows service `src/bin/blue_onyx_prism_service.rs` (reports Running on `Event::Bound`
  only, gives up after `runner::BIND_GIVE_UP`); systemd / launchd / Docker in `deploy/`; install scripts in `scripts/`
  (re-own the data root, `bop_own_data_root`); native installers in `packaging/` (WiX MSI, cargo-deb /
  cargo-generate-rpm metadata in Cargo.toml, pkgbuild pkg; built and smoke-tested by `packaging/build-installers.sh` /
  `smoke-test-installers.sh` in `release.yml` and `packaging.yml`; unsigned).
  - The MSI sets a protected DACL with `icacls` (not `util:PermissionEx`, which merges), keeps only
    `SeChangeNotifyPrivilege`, scopes the firewall rule to the local subnet, sets `sc failureflag 1` (failure actions
    apply to an error exit) and runs the service as LocalService with
    `--config C:\ProgramData\BlueOnyxPrism\blue_onyx_prism_config_service.json`; a service config outside the exe dir
    seeds `download_dir` / `cache_dir` / `log_path` beside it. `port_holder` names what holds a port.
  - deb/rpm and pkg share `scripts/systemd_common.sh` / `scripts/launchd_common.sh` with the archive install scripts;
    the Docker image starts through `deploy/docker-entrypoint.sh` (checks the volumes are writable). Uninstall keeps
    data unless asked (README "Uninstalling keeps your data").
  - `src/service_ctl/` (SCM / `launchctl` / `systemctl`, pure parsers tested everywhere) backs `service ...`;
    `src/tray/` + `src/bin/blue_onyx_prism_tray.rs` (Windows and macOS only; pure parts tested on every OS).
  - Release: `panic::install()` first in every binary, then the CLI's body runs on an 8 MiB thread named `main`
    (`main_thread::run`; the service's server thread gets the same stack): a debug build's clap parse alone
    overflows Windows' 1 MiB main stack; `blue-onyx-prism diagnostics` / `GET /v1/diagnostics`
    (`src/diagnostics.rs`) scrub known tokens and never include token files, frames or labels; app-local VC++ DLLs
    (`packaging/windows/stage-vcruntime.ps1`); `ci-green` gate (`packaging/release/ci_green.py`); glibc floor in
    `packaging/linux/glibc-floor`; `THIRD_PARTY_LICENSES.md` from `python3 -I scripts/gen_third_party.py` (CI `--check`).
    Steps in docs/RELEASE_CHECKLIST.md.

## Conventions

- Edition 2024, `anyhow` for app errors, `tracing` for logs. No `build.rs`. No `unsafe` outside `backend/`
  (`#![deny(unsafe_code)]` elsewhere).
- Everything portable unless it must be `#[cfg(windows)]` (service, event log, DXGI).
- Copy ideas and API shapes from the blue-onyx reference freely (MIT, attributed in LICENSE), including its ONNX
  Runtime usage. Keep all `ort` code in `src/backend/ort.rs`.
- Model weights, `models/`, `cache/`, `openvino/` and `*_config*.json` are git-ignored. Never commit weights (YOLO26 is
  AGPL-3.0).
- Keep the Blue Iris response shape byte-compatible with CodeProject.AI: `success, message, error,
  predictions[{x_min,y_min,x_max,y_max,confidence,label}], count, command, moduleId, executionProvider, canUseGPU,
  inferenceMs, processMs, analysisRoundTripMs`.
- A new config key needs a `| `key` |` row in README.md's config tables: tests in `src/config.rs`,
  `src/training/config.rs` and `src/collect/config.rs` fail on a missing one.
- Twins must stay equal: rules that decide what reaches exports live in `assets/review-rules.js` (pure, no DOM,
  `globalThis.ReviewRules`) with Rust twins, checked through `node` by `tests/it/js_rules.rs`; Python twins
  (`Convergence`, COCO tables) have parity tests.
- Frozen fixtures: `tests/data/overlaps_parity.json` and `tests/data/0.2.0/`. A difference is a change in the code,
  never a reason to edit them.
- CI sets `BOP_REQUIRE_PYTHON=1` and `BOP_REQUIRE_NODE=1`: a test that would skip for a missing Python, numpy or node
  fails there (local runs skip). Tests run scripts through `progress::test_python`. CI installs Python 3.11.9 and
  `scripts/ci/requirements-test.txt` (base install, because tests use `-I`) and runs `scripts/ci/python_self_tests.sh`.
- Tests: integration tests are one binary, `tests/it/` (`cargo test --test it`; `tests/pipeline.rs` stays apart, it
  installs a global tracing subscriber). Pure post-processing tests use synthetic tensors (`tests/it/postprocess.rs`).
  HTTP and backend tests (`tests/it/*_http.rs`, `backend_*.rs`, ...) are `#[ignore]`d and need the fixtures in
  `tests/common/mod.rs` (`models/IPcam-general.*`, `openvino/`, `onnxruntime/`; run with
  `cargo test -- --ignored --test-threads=1`). `common::temp_dir` returns a guard that removes the folder (keep with
  `BOP_KEEP_TEST_DIRS=1`). `review_walk` (`BOP_WALK_FRAMES=10000,50000`, `--ignored`) measures the review queue.
- When proving a test can fail, mutate a private copy with its own target dir, never a checkout other agents build from.
- Disk trap: `cargo clean` wipes data roots under `target/` (here `target/release` holds the OpenVINO / ORT libs, the
  training env, weights such as SAM 3's 3.4 GB, the config, and any dataset or training folder placed there). Do not
  run it without moving them first.

## Working style for agents

- The main agent orchestrates; implementation is delegated to the subagents in `.claude/agents/`: `porter`
  (well-specified porting and boilerplate with a reference to follow) and `engineer` (backend, inference math,
  worker/registry concurrency, benchmarking); each file's frontmatter sets its model. Parallel agents own disjoint
  files and build against the shared interfaces in `src/lib.rs`, `src/api.rs`, `src/config.rs`, `src/model/mod.rs`,
  `src/backend/mod.rs`, which they edit only when the task allows.
- Every subagent finishes with `cargo build`, `cargo test` and `cargo clippy --all-targets -- -D warnings` passing for
  the whole crate (cargo's target lock serializes concurrent builds; that is fine).
- Commit at the end of each phase. A commit message ends with `Co-Authored-By: Claude <model> <noreply@anthropic.com>`
  naming the model that actually made the commit (e.g. `Claude Opus 5.5`; see the agent files for which model a
  subagent runs), never a model that did not write it.
