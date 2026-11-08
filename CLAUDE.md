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
cargo run -- train overlaps [--frames "scene=ir,limit=200,again=1"] [--dry-run] [--allow-large]   # suggestions for overlapping boxes; changes no box
python scripts/research/eval_autolabel.py --self-test   # scores auto-label proposals against known boxes: truth / yesno / score (docs/TRAINING.md, "Measuring auto-label")
cargo run -- train retry-export <run> [--no-register] [--no-benchmark]   # a failed run's kept checkpoint: export, install, benchmark again
cargo run -- admin-token [--rotate] [--config <file>]   # prints the token remote admin needs (`<data root>/admin-token`; stdout holds nothing else)
cargo run -- hf-login                        # saves a Hugging Face token for gated downloads (no-echo prompt or stdin, never argv); `hf-logout` deletes it
cargo run -- train autolabel --sam3 [--sam3-threshold 0.3] [--frames "status=reviewed,auto=overlaps"] [--dry-run]   # SAM 3 in gdino's seat; --dry-run previews frames and a rough time, runs nothing
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

Python export env (models -> ONNX + OpenVINO IR, by hand): `.venv` (Python 3.11; `scripts/requirements-export.txt` = `-r requirements-model-export.txt`, the exact `==` pins the service uses, plus `openvino==`; the hashed per-platform locks in `scripts/model-export/locks/` are the stricter option), created with `uv venv --python 3.11 .venv && uv pip install -r scripts/requirements-export.txt`;
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
(`backend/cpu_threads.rs`), ORT thread pools do not spin, benchmarks run at below-normal priority (`backend/priority.rs`);
on Windows a CPU with AMX but no AMX-FP16 (Sapphire/Emerald Rapids) gets `ONEDNN_MAX_CPU_ISA=AVX512_CORE_FP16` unless set
(OpenVINO's AMX kernels raise #UD there; `backend/cpu_isa.rs`).
A model with results in `benchmark.json` orders its `auto` devices by measured speed (fastest eligible first,
then the static order, CPU last). NPU is never auto. RT-DETR runs on CoreML only with every input dim pinned (a dynamic batch dim aborts MPSGraph in ORT 1.24.4). ORT needs `.onnx` files.

## Architecture (see the plan in the repo history / README for detail)

- Large modules are directories split by concern (`src/benchmark/{run,scoring,sweep}.rs`, `benchmark/cli/`,
  `benchmark/report/`, `benchmark/lineup/`, `server/benchmark/`, `resources/{export,manager,catalog,extract}/`, `registry/`,
  `backend/onnx_rewrite/`, `backend/memory.rs`); each file starts with a `//!` doc stating its responsibility.
- `src/server.rs` axum HTTP (2-thread tokio runtime; fs-heavy handlers run in `server::blocking`). The
  listener is bound once per process (`bind` + `serve_on`) and serves `Shared`, the swappable current
  `AppState`; admin routes answer this machine (`direct_local`: a loopback peer with no `X-Forwarded-For` / `Forwarded` / `X-Real-IP`, the one list in `guard::proxy_header`), and with `allow_remote_admin` remote peers that show the admin token (`Authorization: Bearer` or the `bop_admin` cookie from `/login`; `server/auth.rs`, file in `admin_token.rs`, read per check so a rotation applies at once, missing or bad file = nobody remote; 5 wrong a minute per address = 429), and only under an IP address, `localhost` or an `allowed_hosts` name (`guard::foreign_host`, DNS rebinding; `GET /v1/config` is open and applies both itself: minimal answer otherwise). `/login` and `/logout` are a third class (`login_routes`): no token, but the same peer, `Host` and CSRF checks. A local proxy that adds no forwarding header is indistinguishable from this machine. One-off flags (`Config::one_off_flags`) are kept out of every write: `cli::without_one_off_flags` at startup, `ConfigStore::update` while running. `download_dir`, `onnxruntime_dir` and `openvino_dir` choose libraries the service loads, so they are config-file or CLI only, never web-editable. Blue Iris calls
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
  `GET /v1/status` (readiness, 200/503; Docker HEALTHCHECK, systemd `Type=notify` + watchdog via `sdnotify.rs`); a process-wide in-flight cap on detections answers 503 (`metrics.rs`, `requests_rejected_total`), and uploads still arriving have a 10 s deadline and a per-address limit (`BODY_READS_PER_PEER`, never a cap on one Blue Iris's total requests); `retention.rs` prunes saved images and warns on low disk.
  Also `GET /v1/devices` and the config-page device dropdown; Browse... buttons (`assets/browse.js`, every page) first try `GET|POST /v1/fs/pick` (`server/pick.rs`: native dialog on the server's screen via osascript / PowerShell WinForms / zenity|kdialog; loopback only, one at a time, killed after 10 min; no desktop, e.g. the Windows service in session 0 -> 409) and fall back to `GET /v1/fs/browse` (`server/browse.rs`, in-page, roots only: data root, models, downloads, home, and mounted drives: Windows letters, mount points under `/Volumes`, `/media`, `/run/media`, `/mnt`, never a symlink such as `/Volumes/Macintosh HD`). A natively picked path (and what is under it) passes `browse::allowed_path` for 1 h even outside the roots.
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
  to the model itself); `night_threshold` instead keeps the model and replaces its threshold on night/IR frames (can lower it;
  `LiveSettings::threshold_for_scene`, Models card `@self`). `benchmark --variants` (`benchmark/variant.rs`) runs stats -> enhance -> filters like the
  service. Results and defaults: `docs/NIGHT_IR.md`.
  `verify` (`src/verify.rs` pure decision, `worker/verify.rs` per-worker continuation lane): night person candidates
  re-checked by another model (crop or frame), fail-open, never blocks the inference thread; MegaDetector v6 and the
  ExDark RF-DETRs are export recipes (ExDark ones in `F32_MODELS`, non-commercial); benchmark `animal` class group.
- `src/collect/` dataset collector (`collect` config, default off, live): after the reply, the worker `try_send`s the
  original upload + a v1 sidecar record (raw/suppressed boxes, scene, camera, verify outcomes) to one collector thread
  (buckets positive/empty/suppressed/verify_*/near_threshold/disagreement, per-camera rate limits, dHash dedup (`Bucket::skips_dedup`: all but positive and empty skip it, at most `DEDUP_SKIPS_PER_CAMERA` a minute), `collect/health.rs` (`collection` and `usage` on `GET /v1/dataset`), a disk stop re-arms itself, atomic
  writes under `<data root>/dataset/<scene>/<camera>/<date>/`, `retention::prune_dataset`, disk guard); admin `/dataset`
  page (frames by detected class; Review queue and export filter by class) and `review-queue.jsonl`; format in `docs/DATASET.md`. OpenVINO CACHE_DIR is `<cache>/openvino-<device>-<prec>`
  (the CPU cache hash ignores precision); `config_store::spawn_disk_watch` applies on-disk config edits within 2 s.
- Training protection: a failed run keeps `work` (`training/retry.rs`, `train_model.py --export-only`, YOLO, RF-DETR, RT-DETRv2 / D-FINE via `hf-best/`; training itself does not resume), an OOM restarts only before 3 finished epochs. `training/preflight.rs` is the one report behind the page's panel, `balance-preview` and `train prepare --dry-run`: an empty held-out set is refused, `MIN_CLASS_BOXES` etc. warn; `check_disk` refuses up front. Exports write `frames.json` (`empty_confirmed`, `bulk_accepted`); negatives rank by it and `hard_negatives.json`; frames are deduplicated across exports by flat name, held-out wins.
- Model switch (`training/switch.rs`, `server/training_switch.rs`, `GET /v1/training/switch-preview`, `POST /training/revert`): the Use card shows a settings diff (a lost class loudly), copies per-model behaviour by default (not classes, device, `device_thresholds`), suggests a held-out threshold; `previous_default` holds what the switch changed, so Revert swaps it and a second Revert redoes it (one level). A verifier is replaced only with `replace_verifier` naming it. `Hooks::baseline` benchmarks the current default on the same frames into `<run>/baseline/` (`RunRecord.baseline`), never into `benchmark.json`. `benchmark/trained_on.rs`: `ModelResult.trained_on_share`, backfilled when served, warns where a score includes training frames. The export's benchmark set defaults to held-out (`BenchSplit::Val`).
- Data safety: a failed label write keeps the job's `work` and is in its summary; `task::load_or_move_aside` moves damaged `runs.json` / `tasks.json` to `.corrupt-<time>` and leaves an unreadable one alone, unwritten; JSONL reads are lossy per line and appends repair a torn tail (`layout::read_lines`); exports are built hidden, then renamed; `<dataset root>/.labels.lock` (`fs4`, waits 30 s; a held lock is never given up on, only one that cannot work falls back to the process lock) covers CLI and service label writes; an unavailable review folder is `folder_missing`, never an empty dataset. Uninstall keeps data unless `REMOVEDATA=1` / `--purge` / `apt purge`; `KEEPDATA=1` wins.
- Review (`src/collect/labels/`, `/train/review`, `assets/review.js`, `assets/review-rules.js`; export, classes and coverage are on the Dataset tab, `assets/train-dataset.js`): per-frame `<stem>.labels.json` (v1; states pending/
  accepted/rejected/added; `merge_proposals` + `triage::apply` for auto-labels: evidence, class suggestions
  `auto_label`, `auto_run` marker; a confirmed hint is accepted on save; `revision` check (409) on save; bulk
  accept of `auto=accept` frames `POST /v1/review/accept-auto`; bulk reject of pending auto boxes below a score `POST /v1/review/reject-low`, `reason: low_score`; all in `collect/labels/bulk.rs`); box `difficult` (page `d`, "unsure") and `too_small` rejects are ignore regions (`LabelBox::is_ignore`: out of YOLO, in `ignore.json`, the frame leaves train and valid and stays held-out with the region `ignore: true`; COCO attribute,
  X-AnyLabeling `difficult`) and `reject_reason` (enum, rejected boxes only, Shift+1-9, into `hard_negatives.json`, except `too_small` and duplicates;
  `reason` stays provenance; an unknown one loads as `RejectReason::Other` and round-trips); repeats of fixed cameras
  (`collect/labels/repeats.rs`): `GET /v1/review/history?camera=` (the reviewer's own decisions per camera, cached per
  folder+camera, dropped by every label write, 60 s TTL) drives the page's "rejected here N times" hint and "Reject N
  repeats"; `POST /v1/review/reject-spot|accept-spot` (`bulk::bulk_spot`: pending model/auto boxes at IoU >= 0.6 / 0.8,
  same object class (vehicles one class) / same label, unreviewed frames of that camera and size, `skip_frame`, file
  marker `repeat_spot`) and `POST /v1/review/accept-repeats` (repeat-only frames, nothing pending, sorted last,
  `auto=repeats`); queue sorted by likely boxes, then fewer flagged, then uncertainty (`auto=` filter, wraps), exports YOLO / COCO /
  X-AnyLabeling / benchmark `dir:` with a camera+day split; frames with a human decision or boxes awaiting review are never pruned (`labels::is_protected`), a label file with nothing left to look at goes with its frame. The page reviews `collect.review_dir`
  (default `collect.dir`; *Folder under review*, `POST /v1/review/folder`), e.g. an import saved with `train import --out-dir`;
  auto-label runs follow it. Unwanted classes = labels mapping to no `collect.classes` entry (aliases such as `pottedplant` map) ∪
  `collect.hidden_classes` (Review "don't want (hide + reject)"; class ids and exports unchanged): left out of `prefill` and of auto-label
  runs (`autolabel::without_hidden`; the job also rejects pending unwanted boxes in files it writes), not drawn on the page (a page save
  rejects their pending proposals). A `POST /v1/review/classes` that makes classes newly unwanted rejects their pending model/auto boxes
  on every unreviewed label file of the review folder (`bulk::bulk_reject_unwanted`, `reason: other_class`) and answers `auto_rejected`
  + `reviewed_with_unwanted`; reviewed frames change only when the user confirms (`POST /v1/review/reject-unwanted-reviewed`, frames
  stay reviewed). *Reject boxes of unwanted classes* (`POST /v1/review/reject-other-class`, unreviewed, queue filters) does the same
  by hand; the auto-label form asks to hide classes left unticked. Duplicate boxes (`collect/labels/stacks.rs`, twin of the page's
  `dupGroups`): stacks = live boxes of one object at IoU >= 0.7 (keeper: drawn > edited > accepted > pending > hint, then score; human
  boxes never dropped), rejected on reviewed and unreviewed frames by `POST /v1/review/reject-stacked` (`bulk_reject_stacked`, walks the
  label files via `queue::labelled`, `source` does not narrow; `reject_reason: duplicate`, frames keep their status, `overlaps_left`,
  `overlaps_left_reviewed`), and at save / Shift+A / `accept_frame` for pending boxes; nested or partial overlaps (IoU >= 0.5, or >= 80%
  inside, `inside_share`) go to a human, never a bulk action: `auto=overlaps`, `u`, and `overlaps_checked` on the file once a reviewer
  saved with them kept (the bulk skips such a frame; a merged-in box clears it). A duplicate is no negative: out of `hard_negatives.json`
  (`LabelFile::hard_negatives`), `rejected_boxes` and the repeats history. Overlap suggestions (`collect/labels/overlaps.rs`, pure, ported
  from `scripts/research/eval_overlaps.py`, which was never committed; `tests/data/overlaps_parity.json` holds its answers and is frozen: it cannot be regenerated, so a difference is a change in the code, never a reason to edit the fixture; job `training/overlaps/`, `train overlaps`, Review
  *Suggest overlaps...*, `POST /v1/review/suggest-overlaps`): assist mode, because a blind test failed the automatic rule (13 of 99 drops removed a real
  object). `TaskKind::Overlaps`: select pairs -> env -> zoom (`scripts/autolabel.py --zoom-pairs`: RF-DETR L on the pair's padded union, one flushed JSONL
  line per crop, detections only; ~2.6 GB free commit) -> write. **Only ever writes `LabelFile.overlap_suggestions`** (`overlaps::apply` under `labels::update`,
  re-judged there; no box state, status, `reviewed_at` or `overlaps_checked`; reviewed and unreviewed frames, not skipped or checked). `suggest`: zoom match
  IoU 0.6 + score 0.5 (constants in one place) and `witnesses` (independent detectors at IoU >= 0.7); no zoom answer = witnesses, `low_confidence`; `other_score`
  >= 0.3 = `weak`, the page warns; `unclear` records a looked-at pair so a rerun resumes. Keyed by both bboxes, stale ones ignored (`overlaps::live`, also
  `overlap_live` of `GET /v1/review/labels`), at most 64 per frame. Guards on the dropped box: human, `orig_bbox`, a `model_label` of another class (a
  mapped spelling such as `motorbike` on `motorcycle` does not guard). `auto=overlaps` sorts by `overlaps::strength`; choosing it on the page sets
  status any + frames all. Key `y` (`acceptOverlap`, twin `overlaps::accept`; `liveSuggestions` twin of `live`): rejects as `Duplicate`, promotes a hint keeper so the
  save cannot leave the object with no box; a two-object accept keeps both. Class selection: a run may pick a subset of the wanted classes
  (`AutoRun.classes`; skipped only when the same labelers already covered every picked class), `GET /v1/review/class-catalog`
  (`training/autolabel/catalog.rs`, `coco_class` parity-tested against `scripts/autolabel.py`, default picks = classes the enabled models'
  `object_filter`s keep; `assets/class-picker.js`, also the Config page's per-model Classes column).
- Train area (`server/train_area.rs`, `templates/train/`, `assets/train-area.js`): one nav entry, tabs `/train/{collect,import,autolabel,review,dataset,train,models,jobs}`, old `/dataset`, `/review`, `/training` redirect (307); `GET /v1/train/summary` and `train_area::next_step` fill the strip (`queue::tally` on its own thread, a stale answer is served while a walk runs). Import chains an auto-label with the last run's settings (browser storage). Jobs tab (`server/train_jobs.rs`, `templates/train/running.html`, `GET /v1/train/jobs`, polled every 3 s on that tab, memory only; every tab's label shows the count): every running and queued job with step, percent, time left and a link to its row on its own tab (`#job-<id>`); `training/eta.rs` `JobClock` (service memory, pauses left out, measured from a step's first gain), a training ETA is a range (`TrainProgress::time_left`: early stop to epoch cap), any other estimate covers its step only.
- Training (`src/training/`, tabs Train and Models, `blue-onyx-prism train ...`, `scripts/train_model.py`): user-triggered only;
  `capability` gates by GPU (CUDA lanes cu126/cu130 by compute capability + driver; MPS on Apple, D-FINE CUDA-only),
  separate `tools/train-env` from `scripts/model-export/locks/train-*.txt`; families YOLO26/11/v8, RF-DETR n..l,
  RT-DETRv2, D-FINE (base pins in `catalog/train.rs`); head rows initialised from the base's COCO rows; static ONNX
  export + ORT check, install, benchmark on the held-out split; `epoch`/`eval` events carry every scalar `metrics`
  (raw keys -> ids by `progress::canonical`, F-beta sweeps, `mem_gb`, `img_s`), kept as `progress.history` + `best`,
  table columns in `training.metric_columns` (`POST /v1/training/metric-columns`); an early GPU OOM retries at half the
  batch (MPS watermark 1.0, RF-DETR memory at its multi-scale side); data loader workers are sized at the train stage
  by free commit on Windows (`training::workers_for`, ~1.4 GB per spawned worker x loaders; `training.workers` overrides)
  and a host OOM (WinError 1455, `DefaultCPUAllocator`, which also wins over a WDDM "CUDA out of memory") halves the
  workers, never the batch; training-env helpers (train, import, auto-label: `train_python_cmd`) run below normal
  priority (`Cmd::low_priority`, Windows creation flag / Unix nice >= 10, `backend::priority`), and between training and
  export `train_model.py` `prepare_export` frees the training state (DataLoader workers via gc, trainer, one RF-DETR model at
  a time), caps CPU threads (`BOP_EXPORT_THREADS`) and waits up to 5 min for 3 GB RAM / 4 GB commit; docs in `docs/TRAINING.md`.
  Starting points per family in `recipe::recommended` (epochs cap, patience, min epochs, min_delta; `training.epochs`/
  `.patience` None = recommended, a stored legacy 50 = None); learning rate `training.lr` / `.lr_schedule` (cosine|linear|step, per family `recipe::lr_schedules`) / `.lr_min_factor` (form or CLI value, else config where the family takes it, `TrainingConfig::apply_lr`, else the family default; `--lr-scheduler` / `--lr-min-factor` passed only when set), `batch` events carry the main group's `lr` and `progress` has `lr`, `stop_epoch`, `batch_size`, `imgsz`, `settings` (`RunSettings`); one convergence rule for every trainer (`Convergence` in
  `train_model.py`, Rust twin `progress::convergence_run`, parity-tested; native early stopping off; best raw epoch of the selected metric
  exported (`training.select_metric`, `recipe::SELECT_METRICS`, `--select-metric`, default `map`: drives both the rule and the checkpoint; a failed source falls back to mAP for the run, `select_fallback`); gains judged on a `smooth`-epoch mean (3); a recommended patience doubles under 50 validation images
  (`TrainRecipe::convergence_patience`); a stop point with the training loss still falling (>2%, `loss_falling`) becomes a
  `converge_point` extension, at most 2, shown on the page with Stop now = Finish now) and Finish now (`POST /training/finish` -> `<run dir>/work/stop-request`, `stop` event). `SceneFilter` is a
  day/night/ir set (`night` alone = night only); the held-out set is chosen from all review frames before the scene
  filter (tags `heldout:<scene>|all|in-scope`, `heldout_fingerprint`), so runs queued with several scenes share a
  `compare_group` and compare per scene (`training/compare.rs`, Comparison card, `GET /v1/training/compare`; columns also
  person/vehicle recall + false alerts on empty frames at the model's threshold, from `BenchSummary` `person_recall`,
  `vehicle_recall`, `empty_frames`/`empty_frame_alerts`, `threshold`). RF-DETR: cosine LR schedule (min_factor 0.1) and lr
  5e-5 / lr_encoder 7.5e-5 under 1000 unique train frames, not repeats (`TrainRecipe::effective_lr`, `Prepared.train_frames`); anneal-then-stop at a stop point
  (`--anneal`, max(5, ceil(patience/2)) epochs, Rust twin `progress::convergence_run`). Validation split (`dataset/valid.rs`) = 10% but at
  least 50 frames, at most 25% of the pool as a target (`Options.min_valid`), taken as whole camera days never shared with training:
  seeded rank, class-adding days first, at most half of a class's boxes, up to 1.5x the target, then up to the 25% cap. It falls back to
  the frame split with a warning under 3 days, under half the target, no group with a box, or no camera/day tags (`Stats.valid_groups`).
  "Repeat with N seeds" (form `seeds` 1-3, `train run
  --seeds`): `TrainRequest.train_seed` is the trainer's `--seed` only (dataset assembly seed unchanged, so same frames),
  names get `-s<i>`, one compare group, and the Comparison card adds "mean of N seeds" rows (mean + range, never best). Class
  filter: `dataset::inventory` (`train classes`, `GET /v1/training/dataset-classes`, the page's picker) and
  `Excluded` (background | drop-frame for objects of unpicked classes). Balancing `dataset::balance` (step 7 of
  `dataset::plan`, opt-in: class / scene / camera / camera-day repeat factors capped at max repeat, optional cap of
  dominant groups, negative share kept, held-out/valid untouched; `training::preview_plan`,
  `POST /v1/training/balance-preview`, `train prepare --dry-run`).
  `train import` (`training/import/`: video / image folder / URL via yt-dlp / pinned `import-data:meva-sample` into the
  dataset folder, scene change + dHash + SigLIP k-center, licence confirmed; the 20k candidate budget split per camera then
  per video before extraction (`select::budget`), videos extracted in parallel (`import/parallel.rs`, `--workers`, results
  in video order), analysis on all cores but one, `--hwaccel auto|off` (D3D11VA / CUDA / VAAPI, software on macOS where
  VideoToolbox measured slower); two-pass extraction (640 px candidates, then only the kept frames at full resolution) with a free-space
  preflight and running guard (`import/disk.rs`, ENOSPC -> one actionable error, work folder removed); `GET /v1/training/import/preview` backs the page's preview line) and `train autolabel` (`training/autolabel/`:
  MM-Grounding-DINO + RF-DETR L, Qwen3-VL (`vlm_question`, per-class yes/no, `expected: "yes"`; older files hold the seven-way answer; a no never hides a box, on public sets 40% of them were real) / SigLIP 2 checks, pending `auto` boxes; `--with-model` self-training re-label;
  `--with-model <id> --missing-only` (`triage::apply_missing`, `autolabel::trained_frames`) proposes unlabelled real objects on reviewed
  frames, bypassing `triage::apply` and `merge_proposals`: frames the model trained on are skipped (`<model>.provenance.json` ->
  `<training dir>/<run_id>/data/train.txt`), a same-object training or pending box at IoU >= 0.3 or holding 50% of the detection covers
  it, judged rejects block at IoU >= 0.5, unjudged ones (`low_score`, hints rejected by the save default) are overwritten in place; only
  frames that gain a box are written (`unreviewed` + `disagreement`, so out of exports until saved); no `auto_run` marker, so models run in turn;
  run options: batched VLM check, `--vlm-min-score`, prefetch; a Cancel keeps the finished frames, a rerun resumes)
  are `task::Task`s on the same one-at-a-time queue (an import can queue an auto-label of its frames: `Task::after`, skipped unless the import is Done with new frames; post-steps reject-low / accept / review folder in `training/finish.rs`), recorded in `<training dir>/tasks.json` beside `runs.json`; pins in
  `catalog/autolabel.rs`. A running job pauses / resumes (`PausableRunner`: SIGSTOP/SIGCONT of the helper's process group,
  Windows: the job object's threads suspended; paused time does not count toward the timeout). Split Blue Iris exports
  (`Cam.<time>_N.mp4`) get per-part start times (`source::segment_offsets`). Several NVIDIA GPUs: `--gpu cuda:N` / the
  page's GPU select (`Capability::cuda_options`). The public night subset (`train-data:lila-night`, 926 MB) and other > 500 MB downloads need
  `--allow-large` / the page's box / `allow_large_downloads`.
- SAM 3 (`facebook/sam3`, SAM License, gated, 3.4 GB, pin `autolabel:sam3` in `catalog/autolabel.rs`) is an opt-in labeler: it takes
  gdino's seat in fusion (`--sam3`, form `gdino=sam3`; refused with `with_model`); the default stays gdino and
  nothing here claims it is better, its full evaluation has not run. `--sam3-threshold` (needs `--sam3`) / form `sam3_threshold` (Training
  and Review cards, shown and posted only while SAM 3 is chosen, `al.sam3_threshold` through an import, ignored for other labelers) is SAM 3's
  own score before fusion, 0.01..=1 by `validate`, default 0.3, in the run summary; the marker does not hold it, so a rerun at a new value
  needs `again=1` (the preview warns). `scripts/autolabel.py` `Sam3` (masks off, phrases cached, f32,
  loaded first), `participants_of` generalises the fusion seats, `labeler()` names it so gdino markers do not skip its frames, and
  `--self-check` imports `Sam3Model` before the download. **SAM 3 boxes never triage `accept`** (`ACCEPT_SCORE` 0.6 was tuned for
  gdino + RF-DETR and a bulk accept reaches exports). Cost: ~2 s + 0.2 s x phrases per frame on a GTX 1070 Ti (about 5 s with 22
  phrases; gdino 0.55 s), unmeasured on Apple GPUs. Gated download: `resources/hf_token.rs` looks the token up as `HF_TOKEN`, then
  `<data root>/tools/hf/token` (`hf-login`; 0600 on Unix, chowned to the data root's owner; on Windows a protected DACL on `tools\hf`), then the `hf` CLI's file
  (`HF_TOKEN_PATH`, `HF_HOME/token`, `~/.cache/huggingface/token`), with a source label; the value is never logged, returned by a
  route or passed to Python (`inherited()` drops `HF_*`). `catalog::GATED` (by resource id: repo page, license) marks gated
  resources (exempt from the Apache test; the weekly URL check counts 401 + `GatedRepo` as OK). The manager sends the bearer only to
  huggingface.co (re-read each attempt; reqwest drops it on cross-host redirects); a 401/403 from there, or with `X-Error-Code:
  GatedRepo`, is a typed non-retryable refusal (`Refusal::NeedsLogin|NeedsAccess`; a CDN 403 stays an ordinary failure), and
  `without_url()` keeps signed CDN URLs out of logs and `State::Failed`. `src/server/sam3_access.rs` (admin): `GET /v1/autolabel/sam3`
  (state not_requested|needs_login|token_unreadable|needs_access|checking|downloading|installed|failed, token source, login command),
  `POST .../download` (the "I have access" click: files already at their pinned sizes are adopted by hash with no token or
  large-download gate, else one `config.json` probe answers needs_login / needs_access at once, then free space, then the queue),
  `POST .../remove` (deletes the whole weights folder; refused while a run uses it: `Task::uses` / `TaskRecord::uses` narrow it to
  runs using SAM 3). The login command names the sibling CLI exe by full path with `--config` before `hf-login`, `sudo -u <account>`
  on Unix. `plan_autolabel` fails fast with it when SAM 3 is missing, there is no token and the parts are not on disk.
- Tiers (`triage::tier`, only on the Merge path of a labelers run, constants in `triage.rs`, docs "Likely, unlikely and hidden boxes"): dismissed = pending auto box
  at fused < 0.15, not `truck`, never on a SAM 3 run: `state: rejected`, `reason: auto_dismissed` (`bulk::DISMISSED_REASON`; "hidden" in code means unwanted classes), out of
  `hard_negatives` and the repeats history, `undismiss_found` lets a stronger find through, frames with one are refused by bulk accept and `repeats_only`; 1 in 50 stays visible as
  `auto.audit`; likely = fused >= 0.25 with a yes-contract VLM yes (`auto.likely`, triage stays `review`). A VLM no ranks, never hides. Page keys `c` (Accept likely), `i` (show hidden), `Shift+R` (`REST_REASON`), `auto=dismissed`; `labels::needs_decision` refuses a save holding a flagged box that would become a label.
- Spots (`collect/labels/spots.rs`, `triage::apply_spots`, Rust only, after tiering): rejected spots come from `repeats::scan`, snapshotted once per camera at job start (`Reject the rest` counts), and dismiss new auto boxes as `spot_dismissed` (same readers as `auto_dismissed`, one list: `bulk::is_dismissal`), never at fused >= 0.5, a likely box, an audit sample, or on a SAM 3 run, within 30 days of the frame. Parked spots (`t`, `<review_dir>/spots/<camera>.json`, own mutex, never under the label write lock, `moved_at`, `GET /v1/review/spots`, `POST .../spots/{parked,retire,moved}`) auto-accept the exact label at IoU >= 0.8 unless the VLM said no (`reason: parked_spot`); exports list them in `ignore.json` `parked`, `PARKED_CAP` 20 per spot in `dataset::plan`. `Source::Model` boxes are not covered.
- Labelers reruns (`training/autolabel/triage.rs`): `triage::apply_labelers(f, boxes, name, frame)` decides under the write lock by the
  frame's status at write time: unreviewed = `apply` + votes; reviewed, or sent back by a re-label (unreviewed with `disagreement` and
  `reviewed_at`) = votes, the confidence gate (`gate_drop`), then `apply_missing`; `empty_confirmed` on any status, and skipped = votes
  and the marker only (a save refuses a box on a confirmed-empty frame; a skipped frame must not come back and reach exports by a bulk
  accept). The gate keeps a proposal only if the labelers agree or a source score is >= `MISSING_THRESHOLD` 0.5 (the fused score halves
  single-labeler boxes), no VLM / SigLIP objection, it is on no decided box of another label nor a reject, and a box one labeler found
  has a VLM yes (`vlm.ok`; a check that never ran is no evidence, so `--vlm off`, `vlm_skipped` and a class the VLM is never asked
  about, `Category::Object`, count as `filtered_unchecked`: with the checks off a real run reopened 22 of 41 reviewed frames, mostly on
  false boxes, with them on 0 of 16); survivors get a `why` and triage `review`, and a frame that gains one goes back to
  unreviewed with a `disagreement` note naming `sam3` and loses `bulk_accepted`; `reopened` and the filtered counts are in the summary
  with `REOPENED_WARNING`. Votes: the new labeler's score joins a same-object box's `auto.labelers` at IoU >= `STACK_IOU` 0.7 (best
  match, `or_insert`, `labels::MAX_LABELERS`) so it can witness overlap suggestions; nothing else on the box changes, a vote-only
  write is `queue: false`. A `--with-model` run votes too (`apply_relabel`: `apply`, then `vote`, the model's name as the key, `votes` in
  its summary), with no change to status, `disagreement` or `annotated`; `--missing-only` never votes. `auto=both` (`AutoFilter::Both`, page *found
  by 2+ labelers*): a pending box with 2+ `labelers` of a wanted class (`Filters::wanted`, as `auto=overlaps`), most such boxes first (item field
  `agreed_boxes`); a decided box never counts. `POST /v1/review/restore-reopened` (`bulk::bulk_restore_reopened`; filters, `dry_run`, `limit` 500) puts
  unreviewed frames with `disagreement`, a `reviewed_at` and nothing pending back to reviewed (the undo; reopened frames stay out of
  exports until saved or restored, exports already made are unaffected). `overlaps_checked` comes back with it: `apply_missing` moves
  a true one to `checked_before_reopen`, which a restore or `accept-repeats` gives back, holds `reject-stacked` back while set, and a spot accept of a new box or a save ends. A failed labelers run with a readable `proposals.jsonl`
  writes its finished frames like a Cancel, then still fails; a rerun resumes through the marker. Choosing frames: the Training
  card's *Which frames* presets (unreviewed, reviewed, each with `auto=overlaps`, all, custom; they cooperate with `#al-preset`),
  `autolabel/preview.rs` (`POST /training/autolabel/preview`, `train autolabel --dry-run`; through `server::blocking`, no GPU /
  download / login gate; shares `AutolabelJob::preview`'s walk, which does not stop at `limit`: matched, taken, skipped by marker, split
  by status, a rough time from 2 s + 0.2 s x phrases for SAM 3 or 0.55 s for gdino before the VLM, warnings that reviewed frames may
  reopen, that with the VLM off lone finds cannot, or that `limit` makes the run partial); `select` sets `wanted` like the Review page. *Suggest overlaps...* has *Ask again*
  (`again`): answered pairs are otherwise kept, so a rerun's new votes reach only unchecked frames and unanswered pairs.
- Memory and storage (phase 7): `src/storage/` sweeps only rebuildable files, 30 s after start then daily (`retention` sweeper): cache folders (`openvino-*`, `coreml`, `tensorrt`) unused 60 days by a `.last-used` marker (`storage::touch`, from `backend/device.rs` and `ort.rs` `cache_subdir`; no marker = the clock starts, nothing deleted on a guess), coreml rewrites and compiled models of gone or changed models, train staging and job `work` whose `<folder>.busy` marker (`storage::busy`, fs4, held by every running job) is no longer locked; a crashed auto-label's `proposals.jsonl` is written through the job's own path first (`autolabel::job::recover_crashed`), kept on failure; `layout::compact_review` drops duplicate and gone-frame lines of `review-queue.jsonl` under `.review-queue.lock` (cross-process, appends take it too). Never labels, datasets, exports, models, runs, a failed run's `work` (the retry checkpoint) or logs. Storage card on Config > Resources (`GET /v1/storage`, `POST /v1/storage/clear`, `POST /v1/storage/exports/remove`: refused while a busy run or `training.datasets` uses the export). `training::retry::disk_use` counts a hard-linked file once on every OS. Training history is `<run dir>/history.jsonl` (`training/service/history.rs`; `runs.json` keeps `best` and the latest values, an old one migrates at start), progress saved at most every 5 min, `tasks.json` keeps 200 finished tasks, runs are never dropped. Uploads in flight share a 1 GiB byte budget (`server/body_budget.rs`, charged per chunk, `upload_budget` reject reason); `image_name` / `camera_key` capped at 255 bytes; the collector forgets cameras idle a day (at most 1024). Logs: service default Info; `cli::log_files` keeps 14 days, 500 MB total, 200 MB a day (then one WARN and silence until tomorrow), today's file never deleted. `PREDS_CACHE` / `SAVED_CACHE` drop after 10 idle minutes (`search::IdleCache`); benchmark preds stream. `MALLOC_ARENA_MAX=2` in the systemd unit and the Docker image. GPU memory: `backend::vram` reads free memory live (NVML by PCI bus id, DXGI `QueryVideoMemoryInfo` by LUID, amdgpu sysfs; never a CUDA context; integrated GPUs are `shared`), shown in `GET /v1/devices` `gpuMemory`, metrics `gpu_memory_{total,free}_bytes`, and the model load line; loaded models register as holders (`vram::hold`). Jobs warn, never refuse, on short free memory (training preview and start, auto-label preview and start, import embeddings), naming the models holding it; a labeler that alone cannot fit the card's total is refused (`autolabel::memory::too_big`, page options carry `data-need-mb`). ORT CUDA: `kSameAsRequested`, `HEURISTIC`, optional `ort_gpu_mem_limit_mb`; the second ORT session goes to the default model only (`ort_sessions` overrides). SigLIP runs on CPU in auto-label when other models use the GPU (not loaded under 2 animal classes); import embeddings load the vision tower only, f16 on CUDA 7.0+.
- Release readiness (phase 10): `panic::install()` first in every binary (logs through tracing); the Windows service reports Running on `Event::Bound` only, gives up after `runner::BIND_GIVE_UP` (60 s) with exit code 3/4/1, `sc failureflag 1` makes its failure actions apply, the MSI does not wait for the start; `port_holder` names what holds a port. `blue-onyx-prism diagnostics` / `GET /v1/diagnostics` (`src/diagnostics.rs`): never token files, every part scrubbed of known tokens, never frames or labels. Formats: `Config.format` (`CONFIG_FORMAT`; later = refused at load, by the disk check and by `ConfigStore::update`), `task::RECORD_FORMAT` per runs/tasks record; `collect.*` warns on unknown keys instead of refusing; frozen fixtures in `tests/data/0.2.0/` (like overlaps_parity.json: a difference is a code change). `check_updates` (default true) turns the GitHub update check off. Packaging: app-local VC++ DLLs (`packaging/windows/stage-vcruntime.ps1`), release.yml `ci-green` gate (`packaging/release/ci_green.py --self-test`), upgrade test from the latest release in packaging.yml, `packaging/linux/glibc-floor` (2.34). `THIRD_PARTY_LICENSES.md` crate table: `python3 -I scripts/gen_third_party.py` (CI `--check`). Load test: `test-blue-onyx-prism --rate <r/s> --duration <s>`.
- Structure (phase 9): `model/name.rs` (`normalize_name`) and `model/coco.rs` (COCO tables, `coco_class`, `same_class`, `map_label`) are std-only leaves, re-exported at their old paths; parity tests cover `scripts/autolabel.py`, `train_model.py` (one extra alias on purpose), `export_model.py` and `assets/coco_classes.yaml`. `server/review.rs` keeps the page, `next` and labels; bulk routes, spots and folder/export routes live in `server/review/`; handlers return `Answer` (`Refused` via `?`), messages built by `push_failed` / `push_past_limit` with each route's own words. `GET /v1/review/labels` also answers `overlap_live_boxes` and the page keeps to those boxes (`R.markServed`, `_n` tags never saved). The verify lane reads `VisionDetectionResponse.dropped` (`#[serde(skip)]`), never the message text. CLI benchmark sweeps write `benchmark-breadcrumb.json` and need `--confirm-gpus` over two or more GPU devices (`gpu_gate`). Install scripts re-own the data root when anything in it has another owner (`bop_own_data_root`, `bop_own_kept_dirs`).
- Polling (phase 8): pages poll through `PrismPoll` in `assets/live.js` (chained, never two in flight, nothing while hidden, at once on visible, backoff; tests in `tests/it/js_poll.rs`). `GET /v1/training?poll=1&have=<id>:<last epoch>:<prefix key>:<full key>,...` (`server/training/poll.rs`, merged by `TrainMetrics.poll` in `assets/train-metrics.js`, node parity test) sends only missing history rows; `training::records` = runs without history for every reader that does not draw charts. `/v1/train/jobs?task=<id>|tasks=busy` is how Review follows a task. `GET /v1/benchmark?only=run&models=<tag>` (`server/benchmark/poll.rs`) is the 1 s run poll; the full answer has an ETag and keeps its bytes. Config: `ConfigStore` publishes an `Arc<Published>` through `ArcSwap` after each committed change (`config_read()` = one atomic load, never waits for a save's fsync; writers serialize on `writer`, taken after `apply_lock`, never before); `/v1/config` bodies and ETags are cached while `live::Inputs` are the same.
- Review index (`src/collect/labels/index/`): `GET /v1/review/next` alone answers from it, for a review folder of `MIN_INDEXED_FRAMES` (2,000) frames or more; smaller ones, bulk actions, previews and the CLI walk the files (`queue::build`, which the index must equal: `index/tests.rs` `equivalence` compares the whole answer over 73 filters after builds, real writers, a class change, scans and full passes; review route tests run both ways, `both_ways!`, and the indexed ones assert `index::tests::served`). One folder and one wanted-class list (a class change rebuilds); compact rows, no box geometry. `index::written` (beside `repeats::written`, only a push under the label write lock) makes the next query re-read the file; `append_review` / `rewrite_index` mark the queue / a day; a stat pass every 60 s, a full pass every 30 min, idle drop after 10 min, `INDEX_CEILING_BYTES` 128 MB then walking. While building, answers carry `indexing {done,total}` (only while done < total) and the page polls every 2 s (`R.nextLanding`). Walks: `layout::Walker` canonicalizes each day folder once (same answers as `layout::resolve`), `crate::parallel::map_all` with `walk_threads()` (cores - 1, at most 4); label writes stay on one thread.
- Debug builds read `assets/` from the source tree per request (`no-cache`, no rebuild for a JS or CSS edit; `embedded!` in `server/pages.rs`), so a debug binary moved off its tree answers 500 on `/static/*`; release embeds them. `review_walk` in `tests/it/` measures the review queue (`BOP_WALK_FRAMES=10000,50000`, `--ignored`); results in `docs/PRODUCTION_REVIEW.md`.
- Disk trap: `cargo clean` wipes data roots under `target/` (here `target/release` holds the OpenVINO / ORT libs, the training env,
  weights such as SAM 3's 3.4 GB, the config; a dataset or training folder placed there too). Do not run it without moving them first.
- `src/benchmark/history/`: `benchmark-history.json` beside `benchmark.json` (last 20 runs per model + one pinned
  baseline, seeded from `benchmark.json`; every finished model of a service or CLI run is appended, failed ones never;
  a write failure only warns). `ModelResult.run_id` = the run's start time. Pure `comparable`/`compare` (speed counts at
  >= max(5%, 0.5 ms), accuracy past 2 standard errors, a threshold inside the old plateau is rounding), `trust`, `pareto`;
  `GET /v1/benchmark` carries `latest_run_id`, `noise`, per model `this_run`/`trust`/`changes`, `points`. Admin routes
  `GET /v1/benchmark/history[/compare]`, `POST /v1/benchmark/history/{pin,delete}`; `POST /v1/benchmark/settings`
  re-ranks the saved results (`benchmark/rescore.rs`, exact advice from `benchmark-preds.json`).
  Confidence profiles: `ModelConfig.device_thresholds` (canonical spec -> threshold) win over `confidence_threshold`
  for the device the model actually loaded on (`LiveSettings::threshold_for(eng.device)`, so `auto`/fallback follow);
  `POST /v1/benchmark/apply-profiles` / `--apply-threshold` save `ThresholdAdvice.by_device`; `GET /v1/benchmark`
  `profiles` (`server/benchmark/profiles.rs`). The page's tabs are Compare (`assets/bench-lineup.js`), Run, Tune (`assets/bench-profiles.js`), History, Details;
  the Config page's device dropdown swaps the threshold field to that device's profile / benchmark best.
- `src/benchmark/lineup/` (`sets`, `score`, `comparable`, `verdict`): `lineup` in `GET /v1/benchmark` puts models side by side on exactly the same images, rescored from `benchmark-preds.json` on the classes they share; `comparable_models` is the model-vs-model rule (hardware and runtime flag only speed), a set with training frames leaves that row out of "best", a sentence needs `MIN_OBJECTS_TO_CALL`; `?set=<id>` opens a set.
- `src/resources/` on-demand resources: `catalog` (pins), `resolve` (pure needs from config + hardware +
  installed), `manager` (one download thread, Range resume, SHA-256, staging + atomic rename, `.installed.json`,
  `.downloads.lock`, backoff), `extract` (whitelist-only, nothing executed), `provision` (startup: models wait
  with progress; a new runtime starts a new registry generation), `status` (`/v1/resources` + UI actions),
  `commands` (`fetch`, `list-resources`), `export` (model export: stages uv -> Python -> packages -> weights ->
  export -> install, fake-able `CommandRunner`/`Fetcher`, one at a time, cancellable).
- `src/config.rs` JSON config next to the exe; CLI overrides only non-default values and writes back.
  The server holds it in `config_store.rs` (revision + change log, detects edits on disk); every UI writer goes
  through `save_config` (last write wins). Resources Remove of a model also drops its disabled config entries (an enabled
  one blocks it); the Models card's Remove (`POST /config/models/remove`, `name`) drops a disabled entry, then deletes its files (model, `.bin`, `<stem>.yaml`, `<stem>.provenance.json`, classes) when inside `models_dir`, not a catalog model and unused by other entries. A save applies live: per-request settings (thresholds, filters, timeout,
  image saving, default model) swap into the workers (`LiveSettings`), structural model changes (enable/disable,
  device, family, classes) reload only that model (`ModelRegistry::reload_model`), and only runtime-level fields
  (port, bind address, runtime dirs, cache, threads, profile/concurrency) stay pending until a restart
  (`config_merge::pending`).
- Native installers in `packaging/` (WiX v7 MSI, cargo-deb/cargo-generate-rpm with metadata in Cargo.toml, pkgbuild pkg),
  built and smoke-tested (install, `/v1/status`, uninstall) by `packaging/build-installers.sh` / `smoke-test-installers.sh`
  The MSI sets a protected DACL on the data folder (`icacls`, not `util:PermissionEx`, which merges), keeps only `SeChangeNotifyPrivilege`, and scopes the firewall rule to the local subnet on Domain and Private profiles.
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

- Reloads: a healthy reload or restart loads the replacement beside the old worker, which serves until it is Ready (`Registry` `pending`, `finish_replacement`; `beside` leaves the stats alone until the swap); a memory failure retires the old worker and tries again (`LoadFailure` in `backend/plan.rs`, `make_room`). A Failed or stuck model is retired first. A model on a fallback after a memory or device-lost failure retries the better candidate (5 min .. 1 h, `retry_better`), others never. `worker/reap.rs`: at most 4 reaper threads free abandoned backends. Overload 503s drain the upload (`OVERLOAD_DRAINS` 8). A runtime that lost a listed file is not installed (`manager::install_damage`). A benchmark runs each model's own device by default, several GPUs need `confirm_gpus=1`, a queue job refuses it (409), and `benchmark-breadcrumb.json` names the last device after a hang.
- Phones: `assets/style.css` section *Phones and tablets*: `table.rcards` (`data-label`; `rc-title`, `rc-block`, `rc-action`; `rgrid` = 3 columns) stacks under 700 px, `table.model-card` under 900 px. Review on one column: `.rv-actions-main` sticky bar, Details toggle `#rv-info-toggle`, handles scale with box size (`handleCss`, `handlePoints`; desktop unchanged). Page tests assert `<td>name</td>`, `<td class="state">` and the Stats table's class string, so add classes there with care.

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
- CI sets `BOP_REQUIRE_PYTHON=1` and `BOP_REQUIRE_NODE=1`: a test that would skip for a missing Python, numpy or node fails there, while local runs still skip. Tests run scripts through `progress::test_python`. CI installs Python 3.11.9 and `scripts/ci/requirements-test.txt` (base install, because tests use `-I`) and runs `scripts/ci/python_self_tests.sh`. Review rules that decide what reaches exports are in `assets/review-rules.js` (pure, no DOM; `globalThis.ReviewRules`), checked against the Rust twins through `node` by `tests/js_rules.rs`. When proving a test can fail, mutate a private copy with its own target dir, never a checkout other agents build from.
- Tests: integration tests are one binary, `tests/it/` (`cargo test --test it`; `tests/pipeline.rs` stays apart, it installs a global tracing subscriber): pure post-processing tests in `tests/it/postprocess.rs` with synthetic tensors; the HTTP and backend
  tests (`tests/it/*_http.rs`, `backend_*.rs`, ...) are `#[ignore]`d and need the fixtures described in
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
