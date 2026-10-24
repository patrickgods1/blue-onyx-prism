# Blue Onyx Prism

Cross-platform object-detection HTTP service for Blue Iris (CodeProject.AI-compatible API),
written in Rust on **native OpenVINO** via the `openvino` crate (runtime-linking), with
**ONNX Runtime** (`ort` crate, load-dynamic) as a second runtime for NVIDIA (CUDA/TensorRT),
DirectML and CoreML. Modeled on [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT), with
multi-model support and `auto` device selection by detected hardware (see `docs/PLAN.md`, phase 6).

Primary target: Windows 11, Intel i5-8500 + UHD 630 iGPU. Must also build/run on Linux x86_64 and macOS arm64.

## Build & run

```powershell
cargo build --release
cargo run -- setup-openvino                 # downloads OpenVINO runtime libs into ./openvino (next to exe)
cargo run -- setup-onnxruntime [--flavor auto|cpu|cuda|directml] [--dir <d>]   # ORT libs into ./onnxruntime
cargo run -- list-devices                    # runnable device options and the `auto` pick
cargo run -- download-models --name IPcam-general
cargo run -- add-model --path <file|dir> [--name n] [--family rfdetr] [--classes c.yaml]   # custom model (config page: Add a model by path)
cargo run -- fetch [--for-config] [--resource <id>] [--all-for-platform] [--allow-large]   # pre-download what the config needs
cargo run -- list-resources [--check-urls]   # installed / needed / available; --check-urls checks every pinned URL (weekly CI)
cargo run -- fetch --resource model:yolo26s --allow-large   # on-demand model export (pinned uv + Python env in <data root>/tools)
cargo run -- --model models/IPcam-general.onnx --family yolo5 --force-cpu
cargo run -- --model models/IPcam-general.onnx --family yolo5 --device ort:cpu
cargo run -- benchmark --all-devices [--apply] [--report r.html] [--dataset coco-cctv|bmd45-cctv|exdark-night|dir:<p>] [--max-images 20]   # grade models x devices; benchmark.json next to the config
cargo run -- benchmark --list-datasets       # built-in image sets (assets/bench/*.json, resource bench:<id>)
cargo run -- service status|start|stop|restart   # control the installed service (start/stop/restart need admin)
cargo run --bin blue-onyx-prism-tray -- [--port n] [--cli <path>]   # tray / menu bar app (Windows, macOS)
cargo run -- --version                      # <version> (<git sha>, <build date>): BLUE_ONYX_PRISM_GIT_SHA / _BUILD_DATE at build time (release.yml), `unknown` locally
cargo build --no-default-features            # OpenVINO-only build (drops the `onnxruntime` feature / `ort`)
cargo test                                   # unit tests need no OpenVINO
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

On-demand model export (`src/resources/export/`, the only code that runs downloaded programs; user-triggered
only, never by `auto_download`): `tool:uv` + `scripts/model-export/locks/*.txt` (hashed; regenerate with
`scripts/model-export/gen_locks.sh`) + the embedded `scripts/export_model.py`; everything under `<data root>/tools/`.
Recipes (`catalog::ExportRecipe`): Ultralytics YOLOv8/9/10/11/12/26 (`.pt`, AGPL-3.0), RF-DETR L (transformers,
safetensors, 704 px), DEIMv2 (official code `tool:deimv2-src` pinned to the last Apache-2.0 commit + safetensors; marked non-commercial / personal use, the upstream license now covers the weights);
all offline, static-shape ONNX checked against PyTorch with ONNX Runtime, stamped with `COREML_CACHE_KEY` (ORT's
CoreML cache is otherwise keyed by path). A manifest of another version = "needs export". `catalog::F32_MODELS`
(RF-DETR L, DEIMv2 L/X) overflow f16: OpenVINO compiles them with f32 on CPU and GPU.

Python export env (models -> ONNX + OpenVINO IR, by hand): `.venv` (Python 3.11, `ultralytics`, `openvino`,
`transformers`), created with `uv venv --python 3.11 .venv && uv pip install -r scripts/requirements-export.txt`;
run `.venv/bin/python scripts/export_model.py --name yolo26s` (Windows: `.venv/Scripts/python.exe`).

OpenVINO libs are discovered through `openvino-finder`: `OPENVINO_INSTALL_DIR` (Windows; on Unix
`backend/libs.rs` loads `openvino_c` by path) pointing at `<download root>/openvino` when present, or the OS library path.

On-demand resources: pins, URLs, SHA-256 and sizes of everything downloadable (OpenVINO, ONNX Runtime
flavors, DirectML, `nvidia-cuda-libs`, models) live in `src/resources/catalog/`; change them only in
reviewed commits. The service downloads what the config + hardware need at startup (`auto_download`, default
true; `allow_large_downloads`, default false, gates > 500 MB; `download_dir`, default the exe dir).

ONNX Runtime libs: `onnxruntime_dir` in the config, else `ORT_DYLIB_PATH`, else `onnxruntime/<flavor>/` named by
`onnxruntime/active.txt`, else the old flat `onnxruntime/`, else any flavor dir. Versions are per flavor in the
catalog: `ONNXRUNTIME_VERSION` 1.24.4 for cpu/directml/coreml (the last release with a DirectML package),
`ONNXRUNTIME_CUDA_VERSION` 1.31.0 for cuda (`gpu_cuda12` archives; its Windows build has Pascal `sm_61` kernels
again), `DIRECTML_VERSION`, `CUDNN_VERSION` 9.10.2.21 (the last cuDNN with Pascal kernels; must stay < 9.11 while the
CUDA 12 lane covers Pascal, a test checks it). The `ort` crate is built without an `api-NN` feature, so any ORT 1.17+
library loads; never add one above the oldest pinned flavor. A process loads one ORT flavor only; an outdated flavor
is updated in place (used after a restart when loaded). `onnxruntime/cuda-libs` (opt-in NVIDIA wheels, ~1.9 GB) is
preloaded by path before the CUDA provider loads.

Device spec (`device` in config/model entry, `--device`): `auto` (default), `openvino:gpu[.N]`,
`openvino:cpu`, `openvino:npu`, `ort:cuda[:N]`, `ort:tensorrt[:N]`, `ort:directml[:N]`, `ort:coreml`,
`ort:cpu`; legacy `GPU`/`GPU.N`/`CPU` mean `openvino:*`. `auto` ranks: discrete GPUs first (larger VRAM
first), each with its runtimes in order (NVIDIA: `ort:cuda`, `ort:tensorrt` when its libs load, `ort:directml` on
Windows; Intel Arc: `openvino:gpu.N`, then `ort:directml`; AMD/other: `ort:directml` on Windows), then integrated
GPUs (Intel via OpenVINO, else DirectML), macOS arm64 (CoreML), CPU (`openvino:cpu`, else `ort:cpu`). An NVIDIA
GPU whose compute capability has no kernels in the ORT CUDA build that will load (`catalog::ORT_CUDA_BUILDS`, keyed by
OS and version: 1.31.0 Windows = 6.1/7.5/8.6/8.9/12.0, Linux = 6.0..12.0 without 10.0) skips CUDA; TensorRT needs 7.5+;
a driver below CUDA 12 skips both. `OrtProbe.version` is the loaded library's version (the resolver's hypothetical
probe carries the version the plan installs). CPU inference uses all physical cores but one by default
(`backend/cpu_threads.rs`), ORT thread pools do not spin, benchmarks run at below-normal priority (`backend/priority.rs`).
A model with results in `benchmark.json` orders its `auto` devices by measured speed (fastest eligible first,
then the static order, CPU last). NPU is never auto. RT-DETR runs on CoreML only with every input dim pinned (a dynamic batch dim aborts MPSGraph in ORT 1.24.4). ORT needs `.onnx` files.

## Architecture (see the plan in the repo history / README for detail)

- Large modules are directories split by concern (`src/benchmark/{run,scoring,sweep}.rs`, `benchmark/cli/`,
  `benchmark/report/`, `server/benchmark/`, `resources/{export,manager,catalog,extract}/`, `registry/`,
  `backend/onnx_rewrite/`, `backend/memory.rs`); each file starts with a `//!` doc stating its responsibility.
- `src/server.rs` axum HTTP (2-thread tokio runtime; fs-heavy handlers run in `server::blocking`). The
  listener is bound once per process (`bind` + `serve_on`) and serves `Shared`, the swappable current
  `AppState`; admin routes answer loopback peers only unless `allow_remote_admin`. Blue Iris calls
  `POST /v1/vision/detection` (default model) and `POST /v1/vision/custom/{model}` (named model),
  `POST /v1/vision/custom/list`. Responses use `api.rs` structs (camelCase; `Prediction` snake_case).
- `src/registry.rs` loads N models from config sequentially on one `Core`; each model gets its own
  worker thread (`worker.rs`) and bounded crossbeam channel. `startup.rs` tracks Initializing/Ready/Failed
  so HTTP serves immediately while models compile.
- `src/backend/` runtimes behind a `Backend` enum (`OpenVino | Ort`): OpenVINO wrapper (`Core`,
  `CompiledModel`, `InferRequest`, properties CACHE_DIR/LATENCY/f16 on GPU, tensor IO); `spec.rs` parses
  device specs; `detect.rs` finds GPUs (DXGI / sysfs / Apple); `select.rs` builds the ranked, pure
  `DeviceOption` list and the `auto` pick; `plan.rs` turns it into per-model load candidates (compile +
  warm-up, falling down the list, CPU last); `ort.rs` is the ONNX Runtime backend (all `ort` code lives here).
  `onnx_rewrite.rs`: `ort:coreml` loads a cached copy without HardSigmoid/HardSwish/opset<13 Split (protobuf wire-level, `cache/coreml/<stem>-<sha16>-v<N>.onnx`, `BOP_COREML_REWRITE=0` off).
  `GET /v1/status` (readiness, 200/503; Docker HEALTHCHECK, systemd `Type=notify` + watchdog via `sdnotify.rs`); a process-wide in-flight cap on detections answers 503 (`metrics.rs`, `requests_rejected_total`); `retention.rs` prunes saved images and warns on low disk.
  Also `GET /v1/devices` and the config-page device dropdown; `GET /v1/fs/browse` (`server/browse.rs`, folders + model/YAML files only) backs the Browse... buttons of "Add a model by path".
- `src/model/` model families: `yolo26` (YOLO26/YOLOv10 end-to-end `[1,300,6]`), `yolo5` (`[1,N,5+C]` + NMS),
  `yolo8` (YOLOv8/9/11/12 `[1,4+C,8400]` + NMS), `detr` (D-FINE: `pixel_values` -> `logits/pred_boxes`), `rfdetr` (same, ImageNet-normalized, COCO-91 head), `rtdetr` (RT-DETRv2, DEIMv2: `images` + i64 `orig_target_sizes`; `labels/boxes/scores`; input size from the model's static dims).
  Preprocess = letterbox (YOLO) or stretch (RT-DETR) to 640x640 RGB f32 0..1, CHW.
- Night / IR: `image_stats.rs` (per-request `Scene` day|night|ir from a 64x64 sample: IR = chroma or tint spread < 3,
  dark = BT.601 luma < 50, computed in `prep.rs` after decode), `model/filters.rs` (night/per-label thresholds, person
  aspect, min area, overlap suppression; all default off, applied in `worker::run::finish` before `to_predictions`),
  `static_memory.rs` (per-camera static-box suppression, key from the upload file name). Bench sets `exdark-ir`
  (grayscale transform of exdark-night, `images_from`) and `nir-night` (LILA camera traps, night IR, no persons: person
  scores only as FP; wildlife GT a model cannot emit becomes an animal ignore region) are opt-in (`DEFAULT_SET_IDS`).
  `model/enhance.rs` (`preprocess`/`night_preprocess`: none|grayscale|gamma|clahe, in `prep.rs`; the kept frame is
  always the original). `night_model` routes night/IR frames, decoded, to another model's worker
  (`WorkItem::DetectDecoded`, `worker::Peers` of weak handles published by the registry; never re-forwarded; falls back
  to the model itself). `benchmark --variants` (`benchmark/variant.rs`) runs stats -> enhance -> filters like the
  service. Results and defaults: `docs/NIGHT_IR.md`.
  `verify` (`src/verify.rs` pure decision, `worker/verify.rs` per-worker continuation lane): night person candidates
  re-checked by another model (crop or frame), fail-open, never blocks the inference thread; MegaDetector v6 and the
  ExDark RF-DETRs are export recipes (ExDark ones in `F32_MODELS`, non-commercial); benchmark `animal` class group.
- `src/collect/` dataset collector (`collect` config, default off, live): after the reply, the worker `try_send`s the
  original upload + a v1 sidecar record (raw/suppressed boxes, scene, camera, verify outcomes) to one collector thread
  (buckets positive/empty/suppressed/verify_*/near_threshold/disagreement, per-camera rate limits, dHash dedup, atomic
  writes under `<data root>/dataset/<scene>/<camera>/<date>/`, `retention::prune_dataset`, disk guard); admin `/dataset`
  page (frames by detected class; Review queue and export filter by class) and `review-queue.jsonl`; format in `docs/DATASET.md`. OpenVINO CACHE_DIR is `<cache>/openvino-<device>-<prec>`
  (the CPU cache hash ignores precision); `config_store::spawn_disk_watch` applies on-disk config edits within 2 s.
- Review (`src/collect/labels/`, `/review`, `assets/review.js`): per-frame `<stem>.labels.json` (v1; states pending/
  accepted/rejected/added; `merge_proposals` for auto-labels), uncertainty-sorted queue, exports YOLO / COCO /
  X-AnyLabeling / benchmark `dir:` with a camera+day split; labelled frames are never pruned.
- Training (`src/training/`, `/training`, `blue-onyx-prism train ...`, `scripts/train_model.py`): user-triggered only;
  `capability` gates by GPU (CUDA lanes cu126/cu130 by compute capability + driver; MPS on Apple, D-FINE CUDA-only),
  separate `tools/train-env` from `scripts/model-export/locks/train-*.txt`; families YOLO26/11/v8, RF-DETR n..l,
  RT-DETRv2, D-FINE (base pins in `catalog/train.rs`); head rows initialised from the base's COCO rows; static ONNX
  export + ORT check, install, benchmark on the held-out split; docs in `docs/TRAINING.md`.
  `train import` (`training/import/`: video / image folder / URL via yt-dlp / pinned `import-data:meva-sample` into the
  dataset folder, scene change + dHash + SigLIP k-center, licence confirmed) and `train autolabel` (`training/autolabel/`:
  MM-Grounding-DINO + RF-DETR L, Qwen3-VL / SigLIP 2 checks, pending `auto` boxes; `--with-model` self-training re-label)
  are `task::Task`s on the same one-at-a-time queue, recorded in `<training dir>/tasks.json` beside `runs.json`; pins in
  `catalog/autolabel.rs`. The public night subset (`train-data:lila-night`, 926 MB) and other > 500 MB downloads need
  `--allow-large` / the page's box / `allow_large_downloads`.
- `src/benchmark/history/`: `benchmark-history.json` beside `benchmark.json` (last 20 runs per model + one pinned
  baseline, seeded from `benchmark.json`; every finished model of a service or CLI run is appended, failed ones never;
  a write failure only warns). `ModelResult.run_id` = the run's start time. Pure `comparable`/`compare` (speed counts at
  >= max(5%, 0.5 ms), accuracy past 2 standard errors, a threshold inside the old plateau is rounding), `trust`, `pareto`;
  `GET /v1/benchmark` carries `latest_run_id`, `noise`, per model `this_run`/`trust`/`changes`, `points`. Admin routes
  `GET /v1/benchmark/history[/compare]`, `POST /v1/benchmark/history/{pin,delete}`; `POST /v1/benchmark/settings`
  re-ranks the saved results (`benchmark/rescore.rs`, exact advice from `benchmark-preds.json`).
- `src/resources/` on-demand resources: `catalog` (pins), `resolve` (pure needs from config + hardware +
  installed), `manager` (one download thread, Range resume, SHA-256, staging + atomic rename, `.installed.json`,
  `.downloads.lock`, backoff), `extract` (whitelist-only, nothing executed), `provision` (startup: models wait
  with progress; a new runtime starts a new registry generation), `status` (`/v1/resources` + UI actions),
  `commands` (`fetch`, `list-resources`), `export` (model export: stages uv -> Python -> packages -> weights ->
  export -> install, fake-able `CommandRunner`/`Fetcher`, one at a time, cancellable).
- `src/config.rs` JSON config next to the exe; CLI overrides only non-default values and writes back.
  The server holds it in `config_store.rs` (revision + change log, detects edits on disk); every UI writer goes
  through `save_config` (last write wins). Resources Remove of a model also drops its disabled config entries (an enabled
  one blocks it); the Models card's Remove (`POST /config/models/remove`, `name`) drops a disabled entry, never files. A save applies live: per-request settings (thresholds, filters, timeout,
  image saving, default model) swap into the workers (`LiveSettings`), structural model changes (enable/disable,
  device, family, classes) reload only that model (`ModelRegistry::reload_model`), and only runtime-level fields
  (port, bind address, runtime dirs, cache, threads, profile/concurrency) stay pending until a restart
  (`config_merge::pending`).
- Native installers in `packaging/` (WiX v7 MSI, cargo-deb/cargo-generate-rpm with metadata in Cargo.toml, pkgbuild pkg),
  built and smoke-tested (install, `/v1/status`, uninstall) by `packaging/build-installers.sh` / `smoke-test-installers.sh`
  in `release.yml` and `packaging.yml`; unsigned; runtimes are downloaded on first start. The MSI runs the service as
  LocalService with `--config C:\ProgramData\BlueOnyxPrism\blue_onyx_prism_config_service.json` (a service config created
  outside the exe dir seeds `download_dir`/`cache_dir`/`log_path` beside it). The deb/rpm and pkg share
  `scripts/systemd_common.sh` / `scripts/launchd_common.sh` with the archive install scripts. The pkg also installs
  `/Applications/Uninstall Blue Onyx Prism.app` (`packaging/macos/uninstaller.applescript`, osacompile, not relocatable),
  which only runs `uninstall_launchd.sh` as root; that script removes the program dir too (`--keep-program` to keep it).
- `src/service_ctl/` queries and starts/stops/restarts the installed service (Windows SCM, `launchctl`, `systemctl`;
  pure output parsers tested everywhere) for `blue-onyx-prism service ...`. `src/tray/` + `src/bin/blue_onyx_prism_tray.rs`
  is the tray / menu bar app (tray-icon + tao, Windows and macOS only; pure model/probe/icon/config/elevate parts are
  tested on every OS): polls `service_ctl::query` and `/v1/status`, and runs `<cli> service <verb>` elevated (UAC via
  PowerShell `-Verb RunAs`, macOS `osascript ... with administrator privileges`). Installed by the MSI (HKLM Run key,
  Start Menu) and the pkg (`/Applications/Blue Onyx Prism.app`, LaunchAgent `com.blueonyx.prism.tray`).
- Windows service in `src/bin/blue_onyx_prism_service.rs` (cfg windows). systemd/launchd/Docker in `deploy/`.
  Install/uninstall scripts in `scripts/` (`uninstall_systemd.sh`, `uninstall_launchd.sh`, `*_service.ps1`); the Docker
  image starts through `deploy/docker-entrypoint.sh` (checks the volumes are writable). `THIRD_PARTY_LICENSES.md` ships in
  every release archive; the release workflow attests the archives (`actions/attest-build-provenance`).
- Operability: `log_format: text|json` (file appender only), per-request debug log with `model/device/request_id/
  inference_ms/process_ms/round_trip_ms` and an `X-Request-Id` response header; a model whose inference never returns is
  marked Failed("inference stuck ...") after a second request timeout (`worker::collect_watched`); 5 inference errors in a
  row or 25 all-zero/NaN outputs in a row (a reset GPU) mark it Failed("inference broken ...") (`worker::run::note_health`),
  and a model that failed while serving (stuck, broken, or a dead ORT session) is reloaded by the generation's `supervisor`
  thread with a doubling backoff
  (10 s .. 1 h, reset after 10 min of service); a warm-up that does not finish in 3 min (`worker::WARMUP_TIMEOUT`) abandons
  that device (backend leaked, next candidate tried); a compile over 5 min warns; the "queue full" warning is throttled;
  downloads give up after 12 attempts in the service (`manager::SERVICE_MAX_ATTEMPTS`); a config with no models is seeded with
  IPcam-general. Error messages state what happened, the likely cause and the next action.

## Conventions

- Edition 2024, `anyhow` for app errors, `tracing` for logs. No `build.rs`. No `unsafe` outside `backend/`.
- Everything portable unless it must be `#[cfg(windows)]` (service, event log, DXGI).
- Copy ideas and API shapes from the blue-onyx reference freely (MIT, attributed in LICENSE),
  including its ONNX Runtime usage. Keep all `ort` code in `src/backend/ort.rs`.
- Model weights, `models/`, `cache/`, `openvino/` and `*_config*.json` are git-ignored. Never commit weights
  (YOLO26 is AGPL-3.0).
- Keep the Blue Iris response shape byte-compatible with CodeProject.AI: `success, message, error,
  predictions[{x_min,y_min,x_max,y_max,confidence,label}], count, command, moduleId, executionProvider,
  canUseGPU, inferenceMs, processMs, analysisRoundTripMs`.
- Tests: pure post-processing tests in `tests/postprocess.rs` with synthetic tensors; the HTTP and backend
  integration tests (`tests/*_http.rs`, `backend_*.rs`, ...) are `#[ignore]`d and need the fixtures described in
  `tests/common/mod.rs` (`models/IPcam-general.*`, `openvino/`, `onnxruntime/`; run with
  `cargo test -- --ignored --test-threads=1`). `common::temp_dir` returns a guard that removes the folder
  (keep with `BOP_KEEP_TEST_DIRS=1`).

## Working style for agents

- The main agent orchestrates; implementation work is delegated to subagents (Sonnet for well-specified
  porting/boilerplate, Opus for inference/backend/concurrency code). Parallel agents own disjoint files
  and build against the shared interfaces in `src/lib.rs`, `src/api.rs`, `src/config.rs`, `src/model/mod.rs`.
- Every subagent must finish with `cargo build` and `cargo test` passing for its files (cargo's target
  lock serializes concurrent builds; that is fine).
- Commit at the end of each phase with a message ending in `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
