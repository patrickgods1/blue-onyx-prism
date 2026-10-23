# Training custom models

Blue Onyx Prism can fine-tune a detection model on frames from your own cameras: frames the
[dataset collector](DATASET.md) saved, reviewed on the **Review** page and exported in YOLO format.
The result is a static ONNX model next to the others, added to the config and benchmarked on frames
from cameras and days it did not train on. Use it as the default model, as the `night_model` of
another model (IR and dark frames) or as a `verify` model.

Training runs only when you start it (Training page or `blue-onyx-prism train run`), never from
`auto_download`.

## Can this machine train?

`blue-onyx-prism train capability` and the top of the Training page answer this. Training needs one
of:

| Hardware | PyTorch build | Notes |
|---|---|---|
| NVIDIA GPU, compute capability 6.1-9.0 (GTX 10xx Pascal to Hopper), Windows or Linux x86_64 | CUDA 12.6 (`cu126`) | Any driver that supports CUDA 12. The 12.6 wheels carry kernels for 5.0, 6.0, (6.1 on Windows), 7.0, 7.5, 8.0, 8.6, 9.0; a GPU uses the kernels of its major version with the highest minor not above its own (8.9 Ada runs the 8.6 kernels). |
| NVIDIA GPU, compute capability 10.0 / 12.0 (Blackwell, RTX 50xx) | CUDA 13.0 (`cu130`) | Needs a CUDA 13 driver; the 12.6 build has no Blackwell kernels. |
| Apple silicon, macOS 14+ | PyPI (MPS) | The families that passed a 1-epoch MPS smoke run (below). |

Not supported: Intel and AMD GPUs (no pinned PyTorch build), Intel Macs, Linux arm64, CPU-only
machines. With several NVIDIA GPUs the qualifying one with the most VRAM is used. When training is
not available the page and the CLI say why for every GPU and point at the by-hand path: run
`scripts/train_model.py` in an environment from `scripts/requirements-model-train.txt` on another
machine and copy the `.onnx` and `.yaml` into the models folder (Config page, *Add a model by path*).

Sizes whose estimated memory does not fit the GPU even at batch 1 are shown but cannot be ticked.

## Families

| Family | Sizes | Trainer | Base weights | Licence of the result | Service family |
|---|---|---|---|---|---|
| YOLO26 | n, s, m, l, x | Ultralytics 8.4.163 | the model export's `model:yolo26<size>` `.pt` pins | AGPL-3.0 | `yolo26` (end-to-end `[1, 300, 6]`) |
| YOLO11 | n, s, m, l, x | Ultralytics | `model:yolo11<size>` | AGPL-3.0 | `yolo8` (`[1, 4 + C, A]`, NMS in the service) |
| YOLOv8 | n, s, m, l, x | Ultralytics | `model:yolov8<size>` | AGPL-3.0 | `yolo8` |
| RF-DETR | nano, small, medium, base, large | `rfdetr` 1.11.2 (its training stack) | Roboflow's COCO checkpoints (`train-base:rfdetr-<size>`, the package's own pins, SHA-256 checked) | Apache-2.0 | `rfdetr` |
| RT-DETRv2 | s (R18), m (R34), l (R50), x (R101) | transformers 5.17 `RTDetrV2ForObjectDetection` + `Trainer` | `PekingU/rtdetr_v2_r*vd` at pinned commits | Apache-2.0 | `detr` |
| D-FINE | s, m, l, x | transformers `DFineForObjectDetection` + `Trainer` | `ustc-community/dfine-*-coco` (COCO-only checkpoints) | Apache-2.0 | `detr` |

- **Not offered:** DEIMv2 (its licence now claims the weights), RF-DETR XL / 2XL (Roboflow PML,
  non-commercial), the D-FINE `obj2coco` / `obj365` checkpoints (Objects365 pre-training; its dataset
  terms are research-oriented, so the COCO-only checkpoints are the base instead).
- **AGPL:** a model trained from Ultralytics weights with Ultralytics' trainer is AGPL-3.0. Keep it for
  your own use, or publish it and the code that serves it under AGPL-3.0. The page shows this next to
  the family.
- **Exports:** every family exports a static ONNX through `scripts/export_model.py` (Ultralytics'
  exporter with `nms=False`; transformers + `torch.onnx` for the DETRs), stamped with
  `COREML_CACHE_KEY`, and the run fails unless ONNX Runtime and PyTorch agree (top-20 scores within
  1e-2, boxes within 0.5 px). RF-DETR is exported through transformers' `RfDetrForObjectDetection`
  and also compared with the `rfdetr` package's own PyTorch model; its extra unnamed logit is cut off
  so the service never reports a `class_<n>` label. RT-DETRv2 and D-FINE heads are sigmoid per class
  with RGB 0..1 input (no normalization), which is the service's `detr` family; a 91-class list is
  refused (it would read as a COCO category-id head).

### Class head initialization

The new model's class head has one row per class of the run. Rows of classes the base model knows
(COCO names, case-insensitive, with aliases such as `motorbike` -> `motorcycle`) start from the base
model's row of that class; new classes (`raccoon`, ...) start fresh. Without this, a 3-class
fine-tune starts its whole class head from scratch and needs many epochs to reach what the COCO
model already detects:

- **Ultralytics**: the class branch's hidden width depends on the class count
  (`max(ch[0], min(nc, 100))`), so Ultralytics' own name remap can only copy the biases. The trainer
  builds the COCO-sized model instead, loads every tensor (708/708 for YOLO26n) and replaces only
  the last class conv of each scale (one-to-many and one-to-one heads) by one with a row per class.
- **RF-DETR**: its loader keeps the first `classes + 1` rows of the 91-wide head (COCO category ids),
  which would give class 0 the head of the unused category 0, class 1 person's, ...; the trainer
  first reorders the base checkpoint's `class_embed` / `enc_out_class_embed` rows by category id.
- **transformers**: after `from_pretrained(..., ignore_mismatched_sizes=True)` the rows of
  `class_embed`, `enc_score_head` and `denoising_class_embed` (plus its padding row) are copied from
  the base safetensors by the base config's `id2label`.

Effect on the 1-epoch smoke runs (held-out AP50 on 12 frames, OpenVINO CPU): yolo26n 0.000 ->
0.572, rfdetr-nano 0.000 -> 0.167, rt-detrv2-s 0.054 -> 0.719. (A 30-epoch yolo26n run without the
head init still had AP50 0.000 on the held-out frames: its fresh head scored everything below 0.01.)

### Apple MPS smoke runs (M1, macOS, PyTorch 2.14.0, 1 epoch)

| Run | Result | Wall time (38 train / 10 val images) |
|---|---|---|
| yolo26n, 640, batch 16 | trains, exports, ONNX Runtime check passes (top-20 diff 3.6e-8); held-out AP50 0.572 | ~13 s training; 58 s for the whole CLI run (environment reused, export, install, register, benchmark) |
| rfdetr-nano, 384, batch 4 | trains, exports through transformers (diff vs rfdetr's own model 2.8e-5 after the logit cut); AP50 0.167 | ~45 s (rfdetr trains multi-scale, up to 544 px) |
| rfdetr-base, 560, batch 4 | trains (multi-scale up to 840 px, about 1 min per batch of 4), exports with its patch-14 position embeddings interpolated (diff 3.7e-6); AP50 0.695 | ~31 min: base and large are impractical on an M1 |
| rt-detrv2-s, 640, batch 8 | trains, exports (diff 2.5e-6); AP50 0.719 | ~30 s |
| dfine-s, 640, batch 4 | **fails** in the backward pass on MPS: `RuntimeError: mat2 must be a matrix`, also with `PYTORCH_ENABLE_MPS_FALLBACK=1`. The same run passes on the CPU (`--device cpu`), so it is an MPS kernel issue | - |

So on Apple silicon YOLO, RF-DETR and RT-DETRv2 are offered and D-FINE is CUDA-only.

Through the service (Training page form, `POST /training/queue`): a 1-epoch yolo26n run took 14 s
end to end with the environment and weights in place, was added to the config (applied live) and
benchmarked by the service's benchmark runner (held-out AP50 0.572, 92 ms p50 on OpenVINO CPU);
**night model of** IPcam-general then set `night_model`, loaded the model, and
`/v1/vision/custom/<name>` answered with the trained classes (a dog at 0.91, cars at 0.92 / 0.90).

## The training environment

`<data root>/tools/train-env`, a second Python 3.11 environment beside the model export's
`tools/export-env`, built with the same trust model (see `src/resources/export/mod.rs`):

- `tool:uv` (SHA-256 pinned), Python 3.11 installed by that uv, and the packages of one lock
  (`scripts/model-export/locks/train-*.txt`), installed with `--require-hashes --no-deps --only-binary
  :all:`; PyTorch from the PyTorch CUDA index for the CUDA locks (`--torch-backend cu126|cu130`).
- Locks: `train-macos-arm64.txt`, `train-cu126-windows-x86_64.txt`, `train-cu126-linux-x86_64.txt`,
  `train-cu130-windows-x86_64.txt`, `train-cu130-linux-x86_64.txt`, generated from
  `scripts/requirements-model-train.txt` with `scripts/model-export/gen_train_locks.sh <pinned uv>`.
  All five are resolved with uv's cross-platform resolution (`--python-platform`); the CUDA ones were
  generated on macOS and have not been installed on a Windows or Linux machine yet.
- Contents: torch 2.14.0 / torchvision 0.29.0, ultralytics 8.4.163, transformers 5.17.0, accelerate
  1.15.0, rfdetr 1.11.2 with its training extra except `roboflow` (only for uploading to Roboflow; it
  pins a second OpenCV build that would overwrite Ultralytics' `cv2`), onnx, onnxruntime, onnxslim;
  for the footage import and auto-labelling (below) PyAV 18.1.0 (`av`) and yt-dlp 2026.8.19.
  The shared pins match the export environment, so a trained model is exported like a downloaded one.
- Size: about 450 MB download / 1.5 GB on disk on macOS; about 3 GB download / 6.5 GB on disk with
  CUDA. Over 500 MB, so the first run needs `allow_large_downloads`, the page's confirmation box or
  `--allow-large`.
- Shared with the export: uv, `tools/python`, the uv cache and the Ultralytics `.pt` files
  (`tools/export-weights/<name>`). The environment is built under the export lock (an export and a
  training setup never run uv at the same time); a run holds `tools/.train.lock`, so one training
  runs at a time across processes while exports go on.
- Other base weights: `tools/train-weights/<family>-<size>`. Settings and caches of the libraries:
  `tools/train-config` (`YOLO_CONFIG_DIR`, `HF_HOME`, `TORCH_HOME`, `MPLCONFIGDIR`, `RF_HOME`).
- Offline: `YOLO_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; the child starts from an
  empty environment (the export's allow-list). The scripts are embedded in the binary and written to
  `tools/train-work` before each run.
- **Remove training environment** (page) or `train remove-env` deletes `train-env`,
  `train-weights`, `train-config` and `train-work`; uv and Python stay for the export. The Config
  page's *Remove export toolchain* deletes uv and `tools/python` too; the training environment is then
  downloaded and rebuilt on the next run.
- **Extending it**: add the packages to `requirements-model-train.txt`, regenerate the locks and
  extend the scripts' `--self-check`. The environment's marker records the lock's name and SHA-256,
  so every existing environment is rebuilt once with the new packages (the import / auto-label
  packages were added this way: about 40 s on the M1 from the uv cache).
- The same queue runs the imports and auto-label runs (one GPU job at a time, `tools/.train.lock`
  across processes); their records are in `<training dir>/tasks.json`.

## Dataset

### Input: review exports

The Training page lists the folders in `training.datasets` and the Review page's exports under
`<training dir>/exports/`. Two layouts are read:

1. **The Review page's YOLO export** (`src/collect/labels/export.rs`):

   ```
   <export>/
     classes.txt                                class id = line number
     data.yaml
     images/train/<scene>.<camera>.<date>.<stem>.jpg
     images/val/...                             held out by camera + day by the export
     labels/train/<same stem>.txt               YOLO `class cx cy w h`, normalized; empty = no objects
     labels/val/...
     benchmark/manifest.json                    tags: scene, camera:<c>, date:<d>, hard-negative
   ```

   Scene, camera and date come from the flat file name (or the manifest tags). The export's `val`
   split (whole camera + day groups) becomes the run's **held-out** set: never trained on, used for
   the benchmark. `test/`, `valid/` and `heldout/` folders count as held out too.
2. **A flat YOLO folder**: `images/` (or the folder itself), `labels/<stem>.txt` or `<stem>.txt` beside
   the image, `classes.txt` or `data.yaml` (`names`), optionally `heldout.txt` / `test.txt` listing
   the held-out images, and a `manifest.json` in the benchmark format whose tags (`day` | `night` |
   `ir` | `grayscale`, `camera:<c>`, `date:<YYYY-MM-DD>`, `hard-negative`) give scene, camera and day.

A missing label file means "no objects" (the Ultralytics convention).

### Assembly (`src/training/dataset.rs`)

1. **Classes**: the run's list (`--classes`, the page's field) in head order, else every class of the
   exports in first-seen order. Labels are matched by name across exports (each export's ids are
   remapped), `--rename from=to` renames, an empty `to` drops; boxes of other labels are dropped and
   counted.
2. **Scene filter**: `all`, `day` or `night+ir` (a night model). Frames without a scene tag are left
   out by `day` / `night+ir`, with a warning.
3. **Held out**: the export's split when it has one; else whole `camera/day` groups, in a seeded
   hash order, until `training.heldout_share` (0.2) of the frames, never all groups. With a single
   camera and day the frames are split one by one, with a warning that the benchmark is optimistic.
4. **Hard negatives**: about `training.negative_share` (15%) of the training frames have no objects:
   extra empty frames are dropped (reviewer-confirmed ones are kept first); too few give a warning.
5. **Validation**: 10% of the training frames (at least one, and at least one frame with objects
   stays in training), for the trainer's checkpoint choice.
6. **Public subset**: optional, train only, never held out; your own frames then repeat
   `training.oversample` (3) times.

Output in `<training dir>/<run id>/data/`: `images/`, `labels/` (hard links, else copies),
`classes.txt`, `train.txt` / `valid.txt` (repeats as repeated lines), `data.yaml` (Ultralytics),
`train.coco.json` / `valid.coco.json` (category ids 0..C-1, `file_name` relative to the folder; a
repeated frame is a separate image entry), `heldout/` (a benchmark `dir:` set: `images/`, YOLO
`labels/`, `classes.txt`, `manifest.json` with tags and `scored_labels`) and `dataset.json` (sources,
licences, counts, warnings). `train prepare --out <dir>` writes the same without training.

### Public night subset (`train-data:lila-night`)

`assets/train/lila-night.json` defines it: a benchmark-format manifest (per image `file`, `url`,
`sha256`, `size`, `width`, `height`, `tags`, `objects` with `label` and pixel `bbox`), downloaded and
verified by the download manager into `<data root>/train-data/lila-night/`. Source: LILA BC camera
traps (CDLA-Permissive-1.0) night IR frames with raccoon, opossum, skunk, dog, cat and vehicle boxes
plus empty frames, disjoint from the `nir-night` benchmark set. **This build ships the format only:
the manifest lists no image yet**, so the option is disabled; build it with
`scripts/build_bench_sets.py`-style code (LILA metadata, the night / grayscale test, label map, a
hand screen of the empty frames) in a reviewed commit. Images with `ignore` regions are left out (an
unlabelled animal would be trained as background).

ExDark is never used for training (its terms are non-commercial).

### Augmentation

`--night-aug` (page: *Night augmentation*) sets darkness 0.5, noise 0.3, grayscale 0.5 (per-image
probabilities; each also settable): grayscale = BT.601 luma in all channels (IR-like); darkness =
exposure x0.05-0.4 in linear light (gamma 2.2), MAET-style; noise = shot noise proportional to the
signal plus read noise, sigma 0.01-0.05. Applied on the fly to the training batches for Ultralytics
(on top of its own mosaic / HSV / flip) and transformers (plus a horizontal flip); for RF-DETR, whose
training loop is the `rfdetr` package's, one augmented copy of each training image the probabilities
pick is added before training.

## A run

Stages (page and `train run` show them with percent and ETA):

1. **setup**: uv, Python, the lock's packages, on Windows the Visual C++ runtime check (the export's
   code).
2. **self-check**: `train_model.py --self-check` imports the trainer and reports what PyTorch sees; a
   CUDA run also launches a test kernel, so a driver or kernel problem fails here in seconds.
3. **weights**: the base weights (and the public subset), SHA-256 checked.
4. **dataset**: assembled as above.
5. **train**: `train_model.py` trains, then exports and checks the ONNX. Progress: every event is one
   stdout line `@train {json}` (`start`, `batch`, `epoch` with loss and validation mAP50 where the
   trainer reports it, `eval`, `phase`, `done`); everything else goes to `<run>/train.log` (capped
   at 20 MB). Out of GPU memory, missing kernels and unusable images get their own messages (lower the
   batch or input size, ...); other failures go through the export's classifier.
6. **install**: `models/<name>.onnx`, `<name>.yaml` (the class names) and `<name>.provenance.json`
   (licence and note, family, size, recipe, base weights with URL and SHA-256, datasets with their
   licences and counts, `exdark_used: false`, device, training time, the ONNX's SHA-256, the Blue Onyx
   Prism version). The checkpoints in `<run>/work` are deleted.
7. **register**: the same code as `add-model` adds the model to the config, disabled (enable it, or
   use one of the actions below, which also enable it).
8. **benchmark**: the regular benchmark on the held-out `dir:` set (and `nir-night`, `exdark-ir`,
   `exdark-night` when ticked), so the results are in `benchmark.json` and its history like any other
   run. A failed benchmark is reported but keeps the model.

Default name: `night-<model>-<YYYYMMDD-HHMM>` for a `night+ir` run, else `custom-...`. Several models
can be queued at once (one per ticked size, or `--model` repeated); they run one at a time. Cancel
kills the process tree. Run records are kept in `<training dir>/runs.json`; a run interrupted by a
stopped service is shown as failed. On Unix the trainer exits when the service process is gone; on
Windows the job object does it.

**Use as** (Trained models card): *Default model* sets `default_model`; *night model of* sets the
chosen model's `night_model` (IR and dark frames go to the trained model); *verifier of* sets its
`verify` model. All go through the config page's save path and apply at once.

## Estimates

The page and `train estimate` show GPU memory and time per epoch from a small table
(`src/training/estimate.rs`): `vram = base + per_image x batch x (imgsz / default)^2` MB,
`epoch = images / (images per second x GPU factor) x (imgsz / default)^2`, images per second for a GTX
1080-class card (or an M1), the GPU factor by generation (Volta 2.0, Turing 1.6, A100 4.0, Ampere
consumer 2.2, Ada 3.2, Hopper 6.0, Blackwell 4.5). The M1 numbers of yolo26n, rfdetr-nano and
rt-detrv2-s come from the smoke runs above; everything else is a round number from public reports,
good to a factor of about 1.5. The suggested batch is the largest power of two whose estimate fits 90%
of the memory (capped at 16 when not given). The inference hint (light / medium / heavy) is the size
class on the service's devices: nano / small run on a CPU or a UHD 630; medium and up want a discrete
GPU; the benchmark at the end of the run measures it.

## Footage import

`blue-onyx-prism train import` and the Training page's **Import footage** card put frames from other
sources into the dataset folder (`collect.dir`), in the collector's layout, so they show on the
Dataset page (*Imported footage*) and on the Review page (they are queued for review) like collected
frames.

### Sources

| Source | How | Licence |
|---|---|---|
| A video (`.mp4`, `.mkv`, `.avi`, `.mov`), an image, or a folder of them | `--path`; the page's *Browse...* lists folders and videos (`GET /v1/fs/browse?kind=video`). Folders are walked 6 levels deep, up to 20 000 files. | You confirm it (`--confirm-license`; default text "own footage") |
| Blue Iris clip and alert folders | the same `--path`: `Cam.20240315_083012.mp4` clips and `Cam.20240315_083012.123456.jpg` alert snapshots. The camera folder and the recording time come from the file name. `.bvr` clips cannot be read (Blue Iris's own format): they are counted and the report says to export them to MP4 in Blue Iris. | You confirm it |
| A video URL | `--url`, downloaded with yt-dlp (pinned in the environment): no playlists, no live streams, a video-only format up to 1080 p (so no ffmpeg binary is needed to merge streams), at most 2 GB. What the site reports (title, uploader, page URL, licence) goes into the provenance next to your confirmation; a different site licence is a warning. | `--license` (required) and `--confirm-license` |
| Pinned open footage | `--dataset meva-sample`: six five-minute MEVA clips (Kitware / IARPA, mevadata.org, **CC BY 4.0**; fixed outdoor cameras: a night IR camera, two thermal cameras, a lit night camera, two day cameras; 307 MB), SHA-256 pinned, downloaded by the download manager into `<data root>/train-data/meva-sample`. Frames carry the attribution. The full MEVA corpus (328 h, 516 GB, `s3://mevadata-public-01/drops-123-r13/`) is not pinned; more clips can be added to the catalog the same way. | Pinned (CC BY 4.0) |

**Refused**: URLs of sites that list unsecured or private cameras without their owners' consent
(Insecam and the like: `insecam.*`, `opentopia.com`, `shodan.io`, ... `training::import::source::DENYLIST`,
subdomains included) whatever licence is claimed, and anything but `http://` / `https://` (a camera's
`rtsp://` stream: record a clip and import the file). Scraping of cameras is not supported.

### Frame selection

1. **Extraction** (`scripts/import_footage.py extract`, PyAV): FFmpeg's `select` filter keeps the
   first frame, a frame whose scene-change score is above `--scene` (default 0.01) at most
   `--max-fps` times a second (1), and one frame every `--every` seconds anyway (10), at most
   `--max-candidates` per video (1500):
   `if(isnan(prev_selected_t),1,gt(scene,0.01)*gte(t-prev_selected_t,1)+gte(t-prev_selected_t,10))`.
   FFmpeg's score is the mean frame difference: a person crossing a fixed 1080 p camera scores about
   0.002-0.015 (measured on the MEVA day clip: p99 0.012, maximum 0.014), a cut in edited footage
   0.3 or more, so 0.3 would only ever keep the periodic frames of surveillance video.
2. **Analysis** (Rust): every candidate is decoded once for its dHash (the collector's) and its scene
   (`day | night | ir`, `image_stats`).
3. **Dedup**: a frame is dropped when its dHash is within `--dedup` bits (6) of a kept frame of the
   same camera at most `--dedup-window` seconds apart (10; 0 = any time). The window matters: a
   fixed camera's frames are near duplicates by dHash whatever moves in them (a person flips a few of
   the 64 bits), so without it a long clip keeps a single frame (measured: 3 clips, 90 candidates, 3
   frames).
4. **Selection**: SigLIP 2 (`google/siglip2-base-patch16-224`, Apache-2.0, pinned) image embeddings
   of the unique frames (`import_footage.py embed`, L2-normalized) and k-center greedy on the cosine
   distance (Gonzalez; exact; starting at the frame closest to the mean), per scene with equal quotas
   (water-filled when a scene has fewer frames) unless `--no-balance`. `--select dhash` runs k-center
   on the dHash Hamming distance instead (no embeddings; an image-folder import then needs no Python,
   no GPU and no download).
5. **Writing**: each frame becomes `<scene>/<camera>/<date>/<UTC ms>_imp-<id>-<n>.jpg` (the JPEG copied
   as is) with a version-1 sidecar without detections, `source: "import:<id>"` and its
   `provenance` (docs/DATASET.md, "Imported frames"), an `index.jsonl` line, a `review-queue.jsonl`
   line and the import's line in `<dataset>/imports.jsonl`. The date is the recording time (Blue Iris
   / MEVA file names, else the file's modification time; a URL: the import time). A frame imported
   before (same camera, millisecond and dHash) is skipped. The extracted candidates
   (`<training dir>/imports/<id>`) are deleted (`--keep-work` keeps them).

Imported frames count against `collect.max_gb` and age out after `collect.max_age_days` like collected
ones until they are labelled (a frame with a label file is never pruned; auto-labelling writes one).

**ffmpeg**: not a separate tool. The PyAV wheels of the environment's lock bundle an FFmpeg build that
reports LGPL-3.0-or-later (`av._core.library_meta`; it includes the GPL x264/x265 *encoders*, which the
import never uses), the filter graph runs in-process, and the wheels are pinned by the lock's SHA-256
like every other package. A `tool:ffmpeg` resource would have needed per-OS static builds (BtbN's
Windows / Linux builds, a third party for macOS), their hashes and an extractor for each; PyAV was
already in the lock (a dependency of the trainers) and decodes every format the import reads.

### M1 run (debug build, MPS)

| Import | Time | Result |
|---|---|---|
| one MEVA night IR clip (G639, 5 min, 1072x1920) | 3 min 40 s, of it the environment rebuild for the new lock (95 packages, 37 s from the uv cache) and the SigLIP 2 download (1.5 GB) | before the dedup window: 30 candidates, 1 unique (a static night scene) |
| three MEVA clips (G419 day, G423 lit night, G475 thermal; 15 min) | 71 s | 331 candidates, 68 unique, 24 frames: 12 day, 12 IR (SigLIP 2 k-center) |
| 40 LILA night camera-trap images (CDLA-Permissive-1.0), `--camera lila-cct` | 24 s | 40 candidates, 27 unique, 10 IR frames |
| a public-domain parking-lot camera video on Wikimedia Commons, `--url https://commons.wikimedia.org/wiki/File:...webm` (yt-dlp's Wikimedia extractor, 68 MB WebM) | 58 s | 133 candidates, 58 unique, 12 frames (10 day, 2 IR); the site's licence ("Public domain"), title and uploader in the provenance |

Decoding ran at about 1400 frames/s (PyAV with threads, the 5-minute 1072x1920 H.264 clip in 6.2 s).

## Auto-label

`blue-onyx-prism train autolabel` and the **Auto-label** cards (Training page; Review page: the frames
its filters select) propose boxes for dataset frames. They land in the frames' label files as
**pending `source: auto` boxes** through `labels::merge_proposals` (the service writes every label file;
the script only returns proposals as JSONL), so the reviewer decides. Nothing is accepted, rejected or
deleted automatically; confirmed empty frames (hard negatives) come only from the review.

**Frames**: `--frames` takes the Review queue's filters as `key=value` pairs: `bucket`, `scene`, `camera`,
`status` (default `unreviewed`), `source` (default `all`), `import=<id>|any`, `limit` (default 200). They
are taken in the queue's order (most uncertain first).

### Labelers

| Step | Model | Licence | Pin | Download |
|---|---|---|---|---|
| 1. Open vocabulary | MM-Grounding-DINO tiny (Swin-T, O365 + GoldG + V3Det) by default; base (Swin-B) or large (Swin-L, all data) with `--gdino` | Apache-2.0 | `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det` at `ef2053aa`, `..._base_o365v1_goldg_v3det` at `6f52a617`, `..._large_all` at `0445d0a0` | 0.7 / 0.9 / 1.4 GB |
| 2. Closed set (COCO classes) | RF-DETR L | Apache-2.0 | `train-base:rfdetr-large` (the training base pin) | 0.14 GB |
| 4. Crop check | Qwen3-VL 2B Instruct by default; 4B / 8B with `--vlm`, `off` | Apache-2.0 | `Qwen/Qwen3-VL-2B-Instruct` at `89644892`, `-4B-` at `ebb281ec`, `-8B-` at `0c351dd0` | 4.3 / 8.9 / 17.5 GB |
| 4. Species ranking | SigLIP 2 base | Apache-2.0 | `google/siglip2-base-patch16-224` at `75de2d55` | 1.5 GB |

Every file is pinned by commit and SHA-256 (`src/resources/catalog/autolabel.rs`; the LFS SHA-256 of the
safetensors, the small files downloaded and hashed) and downloaded by the download manager into
`tools/train-weights/<key>`; downloads over 500 MB need `--allow-large` (page: the download box) or
`allow_large_downloads`.

1. **MM-Grounding-DINO**: one prompt phrase per class plus its synonyms (built in for the default
   classes, e.g. `opossum` -> "opossum", "Virginia opossum"; `training.synonyms` replaces a class's
   phrases), joined as `a. b. c.` and split into batches of at most 200 text tokens (the encoder takes
   256); a phrase it reports partly or run together is mapped back to its class by the most shared
   words. Boxes at or above 0.2, class-aware NMS at IoU 0.6. transformers 5.17 declares the per-layer
   decoder box heads of `MMGroundingDinoForObjectDetection` tied to layer 0's, but the checkpoints
   have one head per layer: loaded as is, every box came out a few pixels wide and the full frame
   high. The script loads each layer's head from the checkpoint again (`untie_decoder_heads`); after
   that the boxes are right (checked against the same image on CPU and MPS).
2. **RF-DETR L** for the classes that are COCO classes (with aliases: `motorbike` -> `motorcycle`), at
   or above 0.3.
3. **Fusion**: weighted boxes fusion (Solovyev et al., `conf_type="avg"`, IoU 0.55, weights 1:1), per
   class, written in the script and covered by its `--self-test` (also run by `cargo test` when a
   Python 3 is on PATH, and by every auto-label run). A class only MM-Grounding-DINO knows
   (`raccoon`) is fused over it alone; a box only one of two labelers found keeps half its score.
   Each box records each labeler's score (`labelers`) and whether both found it (`agree`).
4. **Checks**: Qwen3-VL answers "What is in this image: a person, a vehicle, an animal, a statue, a
   rock, a plant, or nothing? Answer with one word." for the crop (20% padding, at most 448 px) of
   every person box and every single-labeler box of the person / vehicle / animal classes; the
   answer is compared with the class's category (`vlm`). SigLIP 2 ranks the animal classes on the crop
   (prompt ensembles; at night "an infrared night-vision photo of a ...") and stores the probabilities
   (`species_scores`).
5. **Triage** (the service, `training::autolabel::triage`): `accept` only when both labelers agree, the
   fused score is at least 0.6, the VLM did not object and SigLIP ranks the box's own species first;
   a person or an animal on a night or IR frame is always `review`. Every box gets `triage` and the
   reasons (`why`). A proposal that matches an existing box of the same label (IoU >= 0.5) is dropped
   by the merge rule, but its evidence (VLM answer, species) is put on that box: a model box the VLM
   calls a rock shows it on the Review page.

On Apple silicon everything runs on MPS (float16 for Qwen3-VL); `PYTORCH_ENABLE_MPS_FALLBACK=1` covers
missing ops.

### Self-training

`train autolabel --with-model <configured model>` (the page: *Or re-label with a trained model*)
re-labels the frames with a trained model through the service's own pipeline (the benchmark's decode,
preprocess, infer, postprocess; the device the model is configured for; no Python, no GPU needed),
default threshold 0.3, by default over every frame (`status=any`). For a frame that was reviewed, its
detections are compared with the training labels (accepted and added boxes): a detection matching
none at IoU 0.5 is *new*, one whose best match has another label a *class change*, a training box no
detection matches *missed*. Any of them puts the frame back to `unreviewed` with
`disagreement: "re-label by <model>: 1 new box, 1 missed"`, and the Review queue lists such frames
first. The detections are merged as pending auto boxes (always `review`). Train, re-label, review the
disagreements, train again: 2-3 rounds, IR frames and rare species first.

### M1 run (MPS)

| Run | Time | Result |
|---|---|---|
| 16 imported MEVA frames, all four labelers, first run | 10 min 17 s, of it the downloads (MM-Grounding-DINO tiny 0.7 GB, RF-DETR L 0.14 GB, Qwen3-VL 2B 4.3 GB) | 34 boxes, all `review`: small, far people and vehicles at night (RF-DETR found people at 0.31-0.37, MM-Grounding-DINO alone vehicles at 0.2-0.4; Qwen3-VL answered "nothing" on the tiny crops) |
| 10 imported LILA night frames, all four labelers, weights in place | 54 s (about 20 s environment check and model loading) | 17 boxes: a dog both labelers found (fused 0.75, SigLIP dog 0.99) and a cat both found (0.72) were `review` only for being animals at night; a single-labeler "dog" SigLIP ranks as an opossum (0.86); a "truck" Qwen3-VL calls a rock |
| re-label of the same 10 frames with a trained yolo26n (person, car, dog) | 6.7 s (ONNX Runtime CoreML, debug build) | the reviewed frame whose dog the model missed went back to review first: "re-label by ...: 1 missed" |

Per frame on the M1 (from the script's `ms` per frame): MM-Grounding-DINO tiny 1.3-2.8 s, RF-DETR L
0.17-0.8 s, Qwen3-VL 2B 1-3 s per frame with checked crops, SigLIP 2 about 1.8 s for the first frame
with animals (the text prompts are computed once).

## Command line

```sh
blue-onyx-prism train capability [--json]
blue-onyx-prism train estimate --model rfdetr-small --images 2000 --epochs 50 [--batch 8] [--imgsz 512]
blue-onyx-prism train prepare --dataset <export> [--dataset <export>] [--scene night+ir] \
    [--classes person,car,dog,cat,raccoon] [--rename motorbike=motorcycle] [--public] --out <dir>
blue-onyx-prism train run --model yolo26n [--model rfdetr-nano ...] --dataset <export> [--scene all|day|night+ir] \
    [--epochs 50] [--batch 16] [--imgsz 640] [--lr 0.001] [--night-aug] [--aug-dark p --aug-noise p --aug-gray p] \
    [--name n] [--benchmark-set nir-night] [--no-register] [--no-benchmark] [--keep-work] [--allow-large] [--verbose]
blue-onyx-prism train import --path <video, image or folder> [--camera c] [--frames 200] [--select siglip|dhash] \
    [--scene 0.01] [--max-fps 1] [--every 10] [--dedup 6] [--dedup-window 10] [--no-balance] \
    [--license "own footage"] --confirm-license [--allow-large] [--keep-work] [--verbose]
blue-onyx-prism train import --url <video URL> --license "CC BY 4.0" --confirm-license [...]
blue-onyx-prism train import --dataset meva-sample [...]
blue-onyx-prism train import --list-datasets
blue-onyx-prism train autolabel [--frames "scene=ir,status=unreviewed,import=any,limit=200"] \
    [--gdino tiny|base|large|off] [--no-rfdetr] [--vlm 2b|4b|8b|off] [--no-siglip] [--classes a,b] \
    [--allow-large] [--verbose]
blue-onyx-prism train autolabel --with-model <trained model> [--threshold 0.3] [--frames ...]
blue-onyx-prism train list
blue-onyx-prism train remove-env
```

Model labels: `yolo26n`..`yolo26x`, `yolo11n`.., `yolov8n`.., `rfdetr-nano|small|medium|base|large`,
`rt-detrv2-s|m|l|x`, `dfine-s|m|l|x`.

## Still to verify on an NVIDIA machine

- Import and auto-label on CUDA: MM-Grounding-DINO base / large and Qwen3-VL 4B / 8B memory and speed
  on 8-12 GB cards (8B in bfloat16 needs about 17 GB: on a 12 GB card it does not fit without
  quantization, which is not pinned; use 2B or 4B there); a URL import through yt-dlp from Windows.

- The CUDA locks install (`cu126` on Windows and Linux, `cu130` with a Blackwell card) and the
  self-check's test kernel passes on a Pascal card (GTX 10xx, `sm_61`).
- 1-epoch runs of every family (D-FINE only trains there), and the estimate table against the real
  memory use (`nvidia-smi` during a run); adjust the constants.
- RF-DETR base (patch 14, 560 px) through the transformers export (the other sizes use patch 16).
- Ultralytics AMP on CUDA (the AMP self-test is replaced, see `train_model.py`; GTX 16xx cards run
  without AMP as Ultralytics recommends).
