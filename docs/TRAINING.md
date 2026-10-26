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
machines. When training is
not available the page and the CLI say why for every GPU and point at the by-hand path: run
`scripts/train_model.py` in an environment from `scripts/requirements-model-train.txt` on another
machine and copy the `.onnx` and `.yaml` into the models folder (Config page, *Add a model by path*).

Sizes whose estimated memory does not fit the GPU even at batch 1 are shown but cannot be ticked.

**Several NVIDIA GPUs.** Every qualifying GPU is an option (`cuda_options` in `GET /v1/training`
and `train capability --json`, the one with the most VRAM first); that first one is the automatic
pick. With more than one, the Training page shows a **GPU** select in the Auto-label card (labelers
mode) and in the Train form (*Options*), and the CLI takes `--gpu cuda:N` (`train run`,
`train autolabel`; `train capability` lists them). The choice is checked when the run is queued: an
unknown ordinal, a GPU PyTorch cannot run on (with its reason) or a CUDA ordinal on a Mac is an
error naming the usable GPUs; on Apple silicon and with one GPU there is nothing to choose (`mps` /
empty only). The run records the device it resolved to (`gpu` in `runs.json` / the task's request;
the auto-label title says `auto-label on cuda:N`), trains with `--device cuda:N`, and the self-check
runs its test kernel on that GPU. A GPU on the other PyTorch build (a Blackwell `cu130` card next to
a `cu126` one) sets up that build's environment, which replaces the other one in `tools/train-env`
(the large-download confirmation names it). The batch is suggested for the chosen GPU's memory; the
page's size grid shows the automatic pick's estimates. `N` is the CUDA driver's ordinal, which is
PyTorch's because the child process inherits the service's `CUDA_VISIBLE_DEVICES` and
`CUDA_DEVICE_ORDER` (the only GPU variables the export runner passes through).

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
5. **Validation**: 10% of your own training frames (of all training frames when there are none; at
   least one), for the trainer's checkpoint choice. A frame holding the last training box of a
   class stays in training. Seeded choices rank a frame by its path relative to its export folder,
   so the same exports give the same split wherever they are.
   A class with no training box gets a warning that says why: its boxes are only in the held-out
   split (the export's `val`: hold out other cameras or days when exporting, or review frames with
   it from the training cameras and days), or no frame has one.
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
verified by the download manager into `<data root>/train-data/lila-night/` (one SHA-256 pinned part
per image; `list-resources --check-urls` checks every URL). Built by
`scripts/build_bench_sets.py --set lila-night-train` (seed 20261010) in a reviewed commit.

- **Content**: 2714 night infrared (monochrome) frames of fixed outdoor camera traps at 462 camera
  locations, **926 MB** (925,830,552 bytes). 2421 boxes: deer 577, fox 544, opossum 527, skunk 525,
  raccoon 114, bird 107, cat 12, squirrel 9, dog 5, car 1; 407 empty frames (15%, tagged `empty`,
  read as hard negatives). Labels are `collect.classes` names (species mapped: brocket deer and
  muntjac are deer, gray and culpeo foxes fox, wildcats cat, vehicles car); frames with any other
  animal (coyote, bobcat, rabbit, rodent, ...) are left out, so every animal in a frame is boxed.
  No person boxes: a `person` class needs your own frames.
- **Sources** (LILA BC, CDLA-Permissive-1.0, attribution in the manifest): Caltech Camera Traps (764)
  and WCS Camera Traps (1059) by capture time 20:00-05:00, Channel Islands Camera Traps (890) and
  NACTI (1) by a mean luma below 110 when there is no time; every frame passes the grayscale (IR)
  test (chroma spread < 3.0/255). Not reviewed by hand: LILA's empty labels are per sequence, so a
  few empty frames may hold a missed small animal.
- **Leakage rule**: disjoint from the `nir-night` benchmark set by image and by camera location (none
  of its cameras is used; ENA24, which has no camera ids, is left out entirely), so `nir-night`
  stays a fair test of a model trained with the subset (a test checks the image side). In a run the
  public frames are only trained on: never held out, never oversampled, in the validation split
  only when none of your own frames trains; your own frames repeat `training.oversample` times beside them.
- **Large-download gate**: over the 500 MB limit, so a run that mixes it in (`--public`, the page's
  *Mix in the public night subset*) and has not downloaded it yet is refused unless
  `allow_large_downloads` is on, `--allow-large` is given (`train run`, `train prepare`, which also
  downloads it) or the page's *Download what is missing* box is ticked; the job checks again before
  it downloads. Once installed, nothing is asked.

Images with `ignore` or `crowd` regions would be left out by the reader (an unlabelled animal would be
trained as background); the subset has none.

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
6. **install**: `<name>.onnx`, `<name>.yaml` (the class names) and `<name>.provenance.json` into the
   models folder, or the folder given in *Install the model in* / `--install-dir` (relative = under the
   data root; the page accepts a browsable folder or one picked in the native dialog). The config entry,
   the benchmark and *Use as* use the file where it was installed (an absolute path outside the data
   root). The `.provenance.json` holds the licence and note, family, size, recipe, base weights with
   URL and SHA-256, datasets with their licences and counts, `exdark_used: false`, device, training
   time, the ONNX's SHA-256 and the Blue Onyx Prism version. The checkpoints in `<run>/work` are deleted.
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

**Pause and resume.** A running training run, import or auto-label job has a **Pause** button on the
Jobs card (Resume while paused, Cancel in both). Pause suspends the job's helper processes (Python,
ffmpeg and what they started) where they are: Unix sends `SIGSTOP` to the process group, Windows
suspends every thread of the job object's processes. Resume continues exactly where it stopped. While
paused, the models stay loaded and the GPU memory stays in use (pause frees CPU and GPU time, not
memory). The paused job keeps its place: nothing else on the queue starts until you resume or cancel it,
and paused time does not count toward a stage's time limit. A helper that would start while the job is
paused waits. Work the service does itself between helpers (analysing imported frames, the in-process
benchmark of a trained model, a download) finishes first; the pause takes hold at the next helper.
Cancel works while paused. A service restart while paused shows the job as interrupted, like a running
one.

**Training folder** (`training.dir`, default `<data root>/training`; This machine card, with a
Folder... button): run records (`runs.json`, `tasks.json`), prepared datasets, held-out sets, logs,
import work and the Review page's exports. A new folder must exist and pass the same checks as other
folder settings (no system folder, browsable or natively picked). It applies at once, without a
restart: the queue switches to the new folder's records when no run, import or auto-label job is
queued or running, and the page refuses the change (409) while one is. Runs, imports and exports in the
old folder stay there and are not moved (set the old folder again to see them).

The dataset folders a run uses are checked too: a ticked folder is used as is only when the page
offered it (`training.datasets` or an export under `<training dir>/exports`); any other value, the typed
*Another folder* and its *Save folder* must be a browsable folder or one picked in the native dialog.

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

`--out-dir <dir>` (the page's *Save frames to*, with a native folder picker) writes them to another
folder instead, in the same layout with its own `imports.jsonl` and review queue. The Dataset page
shows only `collect.dir`; to review the frames, choose the folder on the Review page (*Folder under
review*, `collect.review_dir`). The page accepts a browsable folder or one picked in the native
dialog, like the source path.

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

### Preview and the candidate budget

An import reads at most 20 000 candidate frames (`MAX_CANDIDATES_TOTAL`). Before anything is
extracted, `select::budget` shares them fairly: the cameras of the videos (from the file names, or
`--camera`) and the images (one more group) get equal shares, water-filled (a group that cannot use
its share leaves the rest to the others); a camera's share is split equally by its videos, each at
most `--max-candidates` (1500) and at least 1. Each video's share is its `--max-frames`. Images over
their share are an even spread (stride-sampled, not cut from the end); with more videos than the
budget has frames, a camera reads one frame of an even spread of its videos and the report warns.
So 200 clips of four cameras plus 5 000 alert snapshots read 80 frames per clip and 4 000 snapshots,
where before the first clips by name took everything and the snapshots none. The final cut to
20 000 stays as a guard only.

`GET /v1/training/import/preview?path=<p>[&camera=<c>][&max_candidates=<n>]` (admin, the same path
check as the import, so natively picked folders work) answers what an import would read: the video
and image counts, videos per camera, the total size, the `.bvr` and other files skipped, whether
the listing was cut at 20 000 files, the largest and smallest per-video share, skipped videos, images
used, the warnings and one `summary` line. The Training page shows that line under the path field
(debounced, also after *Browse...* fills it), e.g. "12 videos from 4 cameras, 240 images (3.1 GB);
2 .bvr skipped; up to 1,500 frames per video". *Candidates per video* in the page's Sampling options
is `--max-candidates`.

### Parallel extraction and hardware decoding

Videos are extracted `--workers` at a time (default 0 = half the physical cores, 1 to 4; at most 16;
never more than the videos), one `import_footage.py extract` process each, on scoped threads. The
workers send their output lines and results to the job thread, which alone reports progress
("videos 3/12 done; reading a.mp4, b.mp4"); the results are put back in video order before the dedup
(which keeps the first of near duplicates), so the outcome does not depend on which video finished
first. A video that fails is a warning (`<file>: not read (...)`); a cancel stops every worker (the
running helpers are killed, no further video starts). The helpers run at normal priority:
`backend/priority.rs` only lowers the service's own process, and the command runner has no
per-child priority yet.

`--hwaccel auto` (default) decodes on the platform's hardware decoder when PyAV's FFmpeg has it
(`av.codec.hwaccel.hwdevices_available()`): D3D11VA on Windows, CUDA then VAAPI on Linux (macOS
decodes in software, see the measurement below), opened with `HWAccel(device_type=..., allow_software_fallback=True)` (PyAV 18.1.0,
pinned in the `train-*` locks); a stream the hardware decoder cannot take is decoded in software by
PyAV itself. A decoder that cannot open the video makes it try the next one of the platform's list,
then software; a hardware decode error decodes the video again in software from the start (no seek;
frames already passed are skipped by time, or by count when frames carry none) and sends a `decoder`
event. `done` reports `capped` only when the video had more to give than its share: a further frame
passed the filter within 12 s of video after the quota, or decoding stopped there with frames left. The filter graph's buffer takes the first decoded
frame's size and pixel format (a hardware context reports the device's format, the frames arrive
downloaded, e.g. `nv12`); a frame that differs is converted to it. The `start` event names the
decoder (`hw:d3d11va`, `hw:cuda`, ... or `sw`), shown per video in the job's log with its candidate count; the
self-check lists the hardware decoders this FFmpeg has. `--hwaccel off` decodes in software.

Measured on an M1 (PyAV 18.1.0 wheel, whose FFmpeg has `videotoolbox` only; synthetic 40 s 1080p
15 fps clips at 4 Mbit/s with sensor-like noise; the helper's whole run): VideoToolbox decodes the
same frames and keeps the same candidates, but one clip takes longer and uses about a third of the
CPU: H.264 4.0 s wall / 1.8 s CPU against 1.1 s / 4.6 s in software, HEVC 3.8 s / 1.7 s against
1.7 s / 4.7 s; four H.264 clips at once (4 workers) 4.0 s / 2.9 s against 2.6 s / 20 s. VideoToolbox is
therefore not used: on Apple silicon `auto` decodes in software, the faster import. D3D11VA, CUDA and VAAPI are not measured yet.

The analysis (decode, dHash, scene of every candidate) runs on all physical cores but one
(work-stealing by index, results by index, so the order and the result are the same as on one
thread). Measured on an M1 (4 performance + 4 efficiency cores, release build, 2 000 synthetic
1280x720 JPEGs, `--select dhash`, 200 frames, stage times from `GET /v1/training`): analysis and
selection 2.95-3.34 s on one thread, 0.65-0.66 s on seven; the whole job 5.0-5.4 s, then 2.7 s, of
which writing the 200 frames (an fsync per file) is 2.0 s either way (3 runs each).

### Disk space

An import first writes candidate frames into its work folder (`<training dir>/imports/<id>`). These are JPEGs at
most 640 px on the long side, which is all the dedup, scene stats and SigLIP selection need. Each one records its
time and frame index in the video. Only the kept frames are then extracted again from the video at full resolution
and written to the dataset folder, so the stored frames are the same as before (byte-identical in a PyAV check)
while the work folder is about 5-6x smaller for 1080p footage (much more for 4K). Image folders are read in place.

Before an import is queued, after the downloads, and again before the full-resolution pass, it estimates the space
it needs (`import/disk.rs`): about 190 kB per 1080p candidate, plus about 0.5 byte per pixel for each kept frame in
both the work folder and the dataset (the two add up when they share a volume). It refuses to start unless each
volume keeps 5 % of its size free (at least 1 GiB, at most 5 GiB), and the message says what to lower (Candidates per
video, the sampling rate, Frames to keep) or where to move `training.dir` / Save frames to. While it runs it stops
the helpers once free space falls below half that margin; a full disk (ENOSPC from Rust or the helper) ends the
import with the same advice. The work folder is removed in both cases (unless keep work), and frames already
written to the dataset stay complete and are queued for review. If a source video is gone or one of its kept
frames cannot be found again, that frame is skipped with a warning; a downscaled candidate is never written to the
dataset.

### Frame selection

1. **Extraction** (`scripts/import_footage.py extract`, PyAV): FFmpeg's `select` filter keeps the
   first frame, a frame whose scene-change score is above `--scene` (default 0.01) at most
   `--max-fps` times a second (1), and one frame every `--every` seconds anyway (10), at most the
   video's share of the budget (above; at most `--max-candidates`, 1500):
   `if(isnan(prev_selected_t),1,gt(scene,0.01)*gte(t-prev_selected_t,1)+gte(t-prev_selected_t,10))`.
   FFmpeg's score is the mean frame difference: a person crossing a fixed 1080 p camera scores about
   0.002-0.015 (measured on the MEVA day clip: p99 0.012, maximum 0.014), a cut in edited footage
   0.3 or more, so 0.3 would only ever keep the periodic frames of surveillance video.
2. **Analysis** (Rust, all cores but one): every candidate is decoded once for its dHash (the
   collector's) and its scene (`day | night | ir`, `image_stats`).
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
   / MEVA file names, else the file's modification time; a URL: the import time). The parts of a
   Blue Iris recording exported in segments (`Back.20261010_160000_1.mp4`, `_2`, ...: every part
   named after the recording's start) follow each other in part order, each after the lengths of
   the lower parts present, so they neither claim the same minute nor dedup each other away. A frame imported
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

### Auto-label after import

The Import footage card's **Auto-label the imported frames when the import finishes** box queues an
[auto-label](#auto-label) run right behind the import, with the Auto-label card's labelers (or model),
classes and run options (the page posts them as `al.<field>`); the card's *Which frames* fields are
ignored: the run takes this import's frames (`import=<id>, status=any, source=all`, limit = Frames to
keep) in the folder the import writes. Both are planned before anything is queued, so a bad auto-label
setting (no labeler, no model chosen, a class outside the list, a GPU it cannot use) refuses the import
too ("Not imported: auto-label: ..."). Follow-up options, after the run, in this order and scoped to the
import's frames:

- *Then reject auto boxes scoring below* (0.3): the pending auto boxes under that score on the unreviewed
  frames become `rejected` with `reason: low_score` (hard negatives; a reviewer can still change them);
- *Then accept frames whose boxes are all auto: accept* (off by default): those frames are saved as
  reviewed, like the Review page's bulk accept; after the reject, a frame left with only `accept` boxes
  qualifies;
- *Show this folder on the Review page when done* (needs *Save frames to*; also without auto-label): sets
  `collect.review_dir` to the import's folder.

A failing step is a `warning:` line in the job's summary, not a failed job. The auto-label waits for the
import (`Task::after`; the queue shows "queued (after import <id>)") and runs only when it ended Done
with new frames: when the import failed or was cancelled it ends Cancelled "not run: the import it
follows failed / was cancelled", when it wrote no new frame (all imported before) "not run: the import
wrote no new frames"; cancelling the queued or running import cancels it too. The follow-up steps run
after the last job of the chain, and only when it ended Done.

**Recent imports** (under the card) lists the imports of the folder auto-label works on (the Review
page's, `collect.review_dir` or the dataset folder); a row's **Auto-label** button fills the Auto-label
card's filter with `import=<id>`, status `any` and the import's frame count.

On the command line: `train import ... --autolabel` (the labelers with their defaults) or
`--autolabel-with-model <name|local:<file>|resource:<id>>` (implies it), `--reject-below <0.01..1>` and
`--accept-auto` (both need one of those), `--review-folder` (needs `--out-dir`; a model entry the run
adds and the folder switch are saved to the config file). It runs the import, then the auto-label, and
exits non-zero when either fails (a run skipped because nothing new was imported is not a failure).

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

**Frames**: `--frames` takes the Review queue's filters as `key=value` pairs: `bucket`, `scene`, `camera`, `label`,
`status` (default `unreviewed`), `source` (default `all`), `auto` (`review`, `accept`, `nothing`, `none`;
`label=person|car` = frames with any of these classes; the queue's `label=` takes a comma list;
docs/DATASET.md "Queue"), `import=<id>|any`, `again=1`, `limit` (default 200). They are taken in the
queue's order (most uncertain first). The Review page's button sends its own filters.

**Marker**: every frame the labelers return (boxes or not) gets `auto_run: {at, labeler, classes}` in its label
file (a frame without one gets its pre-fill plus the marker). Only frames whose boxes changed are queued
for review. A later run with the same labelers (`labeler`, e.g. `mm-gdino-tiny + rfdetr-large +
qwen3-vl-2b + siglip2-base`) skips those frames (`skipped_done` in the report) unless the filter has
`again=1` (Review page: **Label again**); when every matching frame was skipped the run fails saying so.
A run with a model writes the marker too, with `labeler: "model:<name>@<device>"` (the device it asked
for), so the same model on the same device skips the frames it looked at unless `again=1`, and `auto=`
filters see them. Writes take the label write lock,
so a Review save with a stale `revision` gets a 409 instead of overwriting merged boxes.

**Run options** (Training page's Auto-label card; the Review page's button uses the defaults):

| Option | Default | CLI | Script flag |
|---|---|---|---|
| Batch the VLM check: up to 8 of a frame's boxes per Qwen3-VL call | on | `--no-vlm-batch` | `--vlm-batch 8` (off: `1`) |
| Skip the VLM check below a fused score (0 < x <= 1) | off (every eligible box) | `--vlm-min-score 0.2` | `--vlm-min-score X` |
| Load the next frame on a background thread while the GPU works | on | `--no-prefetch` | `--prefetch` |
| Keep finished frames if cancelled (both modes) | on | `--discard-on-cancel` | - |
| GPU of the labelers (shown with more than one usable NVIDIA GPU) | automatic (most VRAM) | `--gpu cuda:N` | `--device cuda:N` |

The VLM flags are left out when the VLM is off. Measured per frame on an M1 (MPS, median, one box per
call): Qwen3-VL check 7.6 s (about 1 s per box, ~8 boxes), MM-Grounding-DINO 1.9 s, RF-DETR 0.2 s,
SigLIP 0.07 s; 89% of the checked boxes had a fused score below 0.2, which is what the minimum score
skips (such a box is triaged without a VLM answer).

**Cancel** while the frames are labelled, with *Keep finished frames* on: the frames finished so far
are not thrown away. The labelers' script writes `proposals.jsonl` line by line (flushed per frame);
the run triages and writes those frames exactly like a finished run's (pending boxes, the `auto_run`
marker, the review queue; a line cut off by the kill is skipped), removes its work folder and ends with
`stopped by Cancel after 412 of 2,098 frames; those 412 were saved for review (...)`; its report has
`cancelled: true`, `frames_done`, `frames_total`. A run with a model does the same with the frames the
model finished. Running the same auto-label again **resumes**: the marker makes it skip the saved frames
(`skipped_done`), and the frame limit counts only the frames it labels. A cancel before any frame
finished, or in another stage (selecting frames, environment, downloads, installing a model), discards
the run as before; so does any cancel with the option off.

### Choosing classes

A run looks for `collect.classes` unless it names some of them (`--classes a,b`; the pages' class
picker): each must be in the list (matched case-insensitively, the list's spelling kept), anything else
is refused before the run starts, because the reviewer could not accept its boxes. Proposals of other
classes are dropped and counted (`dropped_other_class` in the report). A detection label is first
mapped onto the list's spelling (`triage::map_label`: the same name, else the same name ignoring case,
else the same COCO class through the aliases): a VOC-spelled model's `motorbike`, `tvmonitor`,
`pottedplant`, `aeroplane`, `sofa`, `diningtable` count as the list's `motorcycle`, `tv`, `potted plant`,
`airplane`, `couch`, `dining table`. The box gets the list's class and keeps the model's spelling in
`model_label` (the Review page shows "was motorbike"); the report lists the renames (`renamed`) and
counts boxes per mapped class (`by_label`). The marker records the run's classes (`auto_run.classes`; files written before
runs could pick classes have none, which counts as the whole list), and the skip rule above becomes:
a frame is skipped only when the same labelers looked at it for **every** class of this run. A run
for `raccoon` after one for `person, car` labels the frames again; a run for `car` after the full list
skips them. `auto=nothing` still means "auto-labelled, no auto evidence"; the Review page reads
`auto_run.classes` to say which classes were found absent, which says nothing about the others.
Self-training with `--classes` compares only detections and reviewed boxes of those classes (a
detection of one may still match a reviewed box of another class: a class change).

RF-DETR L proposes only COCO classes (`coco_class`, mirrored in Rust by
`training::autolabel::catalog::coco_class` with a parity test against `scripts/autolabel.py`); every
other class (`raccoon`, `opossum`, `package`) comes from MM-Grounding-DINO alone. Such boxes can never
have both labelers agree, so the triage always says `review` for them: pick them for a run when there
is time to review. `GET /v1/review/class-catalog` (admin) lists every class worth offering: the review
list, the default phrases, vehicle and animal names, the 80 COCO classes and every configured model's
classes, each with its category, prompt phrases, `coco` flag and the models that emit it, plus
`relevant` (the review classes the enabled models report after their `object_filter`).

`collect.hidden_classes` (subset of `collect.classes`; `POST /v1/review/classes` with `hidden`) hides
rare classes from the Review page's pickers only: they stay in the exports and keep their YOLO class
ids. Removing a class from the list drops it from the hidden list too.

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
   calls a rock shows it on the Review page, and a suppressed / below-threshold / verifier-found box
   triaged `accept` is accepted on save instead of rejected. A proposal on a *pending* box of another
   label (and no box of its own label) becomes a **class suggestion** on that box (`auto_label`,
   `auto_score`, triage `review`; `class_suggestions` in the report) instead of being lost.
6. **Bulk accept**: frames whose triage alone settles them (`auto=accept`: boxes `accept`, nothing else
   pending but unconfirmed model hints) can be saved as reviewed in one go from the Review page
   (`POST /v1/review/accept-auto`, `bulk_accepted: true` in the file); everything else waits for a
   reviewer.

On Apple silicon everything runs on MPS (float16 for Qwen3-VL); `PYTORCH_ENABLE_MPS_FALLBACK=1` covers
missing ops.

### Label with any model

`train autolabel --with-model <id>` (the Training page's *Label with a model*; `POST /training/autolabel`
`with_model`) labels the frames with a detection model through the service's own pipeline (the
benchmark's decode, preprocess, infer, postprocess; no Python, no training GPU needed), default
threshold 0.3, by default over the **unreviewed** frames (first-pass labelling). Its boxes are pending
`source: auto` boxes like the labelers'.

**Sources** (`GET /v1/autolabel/models`, admin, `training::autolabel::models`):

| `source` | What | `id` |
|---|---|---|
| `config` | every entry of `models`, enabled or not | its name |
| `local` | a model file in `models_dir` no entry uses (the Config page's *Local models*), family `auto` | `local:<file>` |
| `catalog` | a model resource of this platform no entry uses: downloaded, or exported on this machine (`needs_export`) | `resource:<catalog id>`, e.g. `resource:model:rfdetr-large` |

```json
{"success": true,
 "models": [{"id": "rfdetr-large", "source": "config", "name": "rfdetr-large", "installed": true,
             "in_config": true, "enabled": false, "family": "rfdetr", "classes": 80, "coco": true,
             "size_bytes": 135795376, "needs_export": false, "large": false, "license": "Apache-2.0",
             "note": "", "device": "auto"}],
 "devices": [{"value": "auto", "label": "auto (openvino:gpu)"}, {"value": "openvino:cpu", "label": "..."}],
 "default_threshold": 0.3}
```

Sorted config (enabled first), local, then catalog by size (bigger is usually more accurate).
`classes` is null when the class names are not known before a download (the YOLOv5 models with their
own class file); `coco` says every class is a COCO class (aliases applied). `size_bytes` is the file
size, or for a catalog model the download (an export: its weights). `large` means installing it needs
the large-download permission: over 500 MB, or an export whose toolchain is not set up yet. `devices`
are `auto` (with what it picks now) and every runnable device option. The CLI also takes a catalog
model's plain name (`--with-model rfdetr-base`).

**Install on demand**: a catalog model that is not on disk is installed by the run itself, in its stage
2, *installing the model*: downloaded by the download manager (SHA-256, resume), or exported with the
export toolchain (`resources::export::run_export`, the export lock: one export at a time with the Config
page's), with progress on the task and cancel. It is gated like the Config page: a large download needs
`allow_large` (page: the download box; CLI `--allow-large`) or `allow_large_downloads`, an export this
platform cannot run is refused, and a non-commercial / personal-use licence (DEIMv2, the ExDark
RF-DETRs) needs `confirm_license` (CLI `--confirm-license`), refused before the run starts. Stage 3
then adds a local or catalog model to the config **disabled** (the Config page's Add to config rules,
through `save_config` in the service, the config file in the CLI), so it is never loaded for serving and
the next run finds it as a config model. Nothing is downloaded outside a run the user started. A local
model has family `auto` (detected from its outputs when it loads); when that fails the run says to set
the family on the Config page (Models card).

**Device per run**: `device` (CLI `--device`, any device spec, `auto` included) overrides the model's
configured device for this run only (not saved); empty = the model's device. The report records the
device asked for (`requested_device`) and the one that ran (`device`), and the marker uses the asked one
(`model:<name>@<device>`). On a CPU, large models are slow: RF-DETR L takes about 1 s per frame, so
pick a GPU device or a smaller model for big batches.

**Pre-accept**: `model_accept_score` (CLI `--accept-score-model`, 0.01..1, at least the threshold;
default off) triages boxes at or above it `accept` (why: "model <name> at 0.87 >= 0.80"), the others
`review` ("score 0.55 below 0.80"); a person or an animal on a night or IR frame stays `review` (the
labelers' rule). Without it every box is `review`. Pre-accepted frames are what the bulk accept
(`auto=accept`, `POST /v1/review/accept-auto`) takes.

### Self-training

Self-training is a run with a trained model over the reviewed frames too: an explicit `status=any`
(the Training page's preset). For a frame that was reviewed, its detections (labels mapped onto the
list) are compared with the training labels (accepted and added boxes): a detection matching
none at IoU 0.5 is *new*, one whose best match has another label a *class change*, a training box no
detection matches *missed*. Any of them puts the frame back to `unreviewed` with
`disagreement: "re-label by <model>: 1 new box, 1 missed"`, and the Review queue lists such frames
first. The detections are merged as pending auto boxes (`review`, or `accept` at a pre-accept score).
Train, re-label, review the disagreements, train again: 2-3 rounds, IR frames and rare species first.

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
    [--classes person,car,dog,cat,raccoon] [--rename motorbike=motorcycle] [--public [--allow-large]] --out <dir>
blue-onyx-prism train run --model yolo26n [--model rfdetr-nano ...] --dataset <export> [--scene all|day|night+ir] \
    [--epochs 50] [--batch 16] [--imgsz 640] [--lr 0.001] [--night-aug] [--aug-dark p --aug-noise p --aug-gray p] \
    [--name n] [--install-dir <dir>] [--gpu cuda:N] [--benchmark-set nir-night] [--no-register] [--no-benchmark] [--keep-work] [--allow-large] [--verbose]
blue-onyx-prism train import --path <video, image or folder> [--camera c] [--frames 200] [--select siglip|dhash] \
    [--scene 0.01] [--max-fps 1] [--every 10] [--dedup 6] [--dedup-window 10] [--no-balance] \
    [--max-candidates 1500] [--workers 0] [--hwaccel auto|off] \
    [--license "own footage"] --confirm-license [--allow-large] [--keep-work] [--verbose] \
    [--out-dir <dir> [--review-folder]] [--autolabel | --autolabel-with-model <model>] \
    [--reject-below 0.3] [--accept-auto]
blue-onyx-prism train import --url <video URL> --license "CC BY 4.0" --confirm-license [...]
blue-onyx-prism train import --dataset meva-sample [...]
blue-onyx-prism train import --list-datasets
blue-onyx-prism train autolabel [--frames "scene=ir,status=unreviewed,import=any,limit=200"] \
    [--gdino tiny|base|large|off] [--no-rfdetr] [--vlm 2b|4b|8b|off] [--no-siglip] [--classes a,b] \
    [--no-vlm-batch] [--vlm-min-score 0.2] [--no-prefetch] [--discard-on-cancel] [--gpu cuda:N] [--allow-large] [--verbose]
blue-onyx-prism train autolabel --with-model <name|local:<file>|resource:<id>> [--threshold 0.3] \
    [--device <spec>] [--accept-score-model 0.8] [--confirm-license] [--allow-large] [--frames "status=any,..."]
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
- The GPU picker on a multi-GPU PC: `cuda:N` of the service's probe is PyTorch's `cuda:N` (same
  `CUDA_DEVICE_ORDER`), a second-GPU run lands on that GPU (`nvidia-smi`), and switching between a
  `cu126` and a `cu130` GPU rebuilds the environment once each way.
- 1-epoch runs of every family (D-FINE only trains there), and the estimate table against the real
  memory use (`nvidia-smi` during a run); adjust the constants.
- RF-DETR base (patch 14, 560 px) through the transformers export (the other sizes use patch 16).
- Ultralytics AMP on CUDA (the AMP self-test is replaced, see `train_model.py`; GTX 16xx cards run
  without AMP as Ultralytics recommends).
