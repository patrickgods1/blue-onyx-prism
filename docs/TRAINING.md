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

1. **Classes**: the run's list (`--classes`, repeated or comma-separated; the page's class picker) in
   head order, else every class of the exports in first-seen order. Labels are matched by name across
   exports (each export's ids are remapped), `--rename from=to` renames, an empty `to` drops. What
   happens to a box of a label that is not in the list is the **excluded mode** (`--excluded`, the
   page's radio under the picker): `background` (default) keeps the frame and the object is
   background (a frame left with no box is a negative, counted toward the negative share);
   `drop-frame` leaves out every training / validation frame that holds such an object, so a
   look-alike of a trained class (`dog` while training `cat`) is never taught as background. Dropped
   frames are counted (`excluded_frames` in `dataset.json`) and a run that drops more than 30% of
   its in-scope frames gets a warning. Held-out frames are never dropped, so a comparison stays fair.
   See [Class inventory](#class-inventory).
2. **Held out**: chosen from **all** review frames, *before* the scene filter: the export's split when
   it has one; else whole `camera/day` groups, in a seeded hash order, until
   `training.heldout_share` (0.2) of the frames, never all groups. With a single camera and day the
   frames are split one by one, with a warning that the benchmark is optimistic. The held-out set
   keeps every scene, so a day model is also measured on night frames, and the same exports give the
   same held-out frames whatever the scene filter. `dataset.json` holds `heldout_fingerprint` (SHA-256
   of the sorted held-out frame keys), copied into the run's benchmark summary; the held-out
   `manifest.json` tags each frame with its scene plus `heldout:<scene>`, `heldout:all` and, for a
   frame of the run's scenes, `heldout:in-scope`, which the benchmark summary reads for per-scene AP.
3. **Scene filter** (training and validation frames only): `all`, or any combination of `day`,
   `night` and `ir` joined by `+` (`day`, `night`, `ir`, `day+night`, `day+ir`, `night+ir`; the
   page's checkboxes). Frames without a scene tag are left out by every filter but `all`, with a
   warning. **Behaviour change**: `night` alone now means night frames only and `ir` IR frames only;
   use `night+ir` for both (what `night` used to mean). Saved requests and configs with the old
   spellings still load.
4. **Hard negatives**: about `training.negative_share` (15%) of the training frames have no objects:
   extra empty frames are dropped (reviewer-confirmed ones are kept first); too few give a warning.
5. **Validation**: 10% of your own training frames (of all training frames when there are none),
   but at least 50 frames when the pool has them (`Options.min_valid`; never more than a quarter of the
   pool, so 100 frames give 25, 241 give 50, 1000 give 100; always at least one), for the trainer's checkpoint choice. A frame holding the last training box of a
   class stays in training. Seeded choices rank a frame by its path relative to its export folder,
   so the same exports give the same split wherever they are.
   A class with no training box gets a warning that says why: its boxes are only in the held-out
   split (the export's `val`: hold out other cameras or days when exporting, or review frames with
   it from the training cameras and days), or no frame has one.
6. **Public subset**: optional, train only, never held out; your own frames then repeat
   `training.oversample` (3) times.
7. **Balance** (optional, off by default; `--balance`, the page's Balance fieldset, `training.balance`
   for the default), on the training frames only. It repeats existing frames of rare groups (oversampling)
   and can drop frames of dominant ones. The axes, which can be combined:
   - `class`: a frame belongs to every class that has a box in it.
   - `scene`: `day`, `night`, `ir` (`unknown` without a tag).
   - `camera`: the frame's camera (a public frame: its source).
   - `camera-day`: `camera/day`, compared only with the other days of the *same* camera.

   A group `g` with `n_g` training frames gets the weight `w(g) = (n_max / n_g)^strength`, `n_max` being
   the largest group of the axis (the largest day of the same camera for `camera-day`). `strength` is
   0.5 by default, the square root, which evens out gently as in repeat-factor sampling; 1 equalises
   fully. A frame's **class** factor is the largest weight over its classes (Gupta et al., "LVIS: A
   Dataset for Large Vocabulary Instance Segmentation", CVPR 2019,
   <https://arxiv.org/abs/1908.03195>); empty frames get 1. Boxes of a common class rise too when
   they share a frame with a rare one (a person beside a raccoon), so balancing narrows the gap rather
   than equalising box counts. The factors of the ticked axes multiply and
   are capped at `--balance-max-repeat` (4 by default, at most 10). The repeats are `round(base x
   factor)` (`base` is the `training.oversample` repeat of the public mix, else 1) with a **seeded
   systematic rounding**: each frame rounds up with the probability of its fraction, so the total is
   within +-1 of the exact sum and the same inputs give the same plan.

   The optional **cap** (`--balance-cap r`, the page's "Cap dominant groups") first drops frames of
   groups above `r x` the median group size, by seeded rank, down to that size: the frame-level axes
   first, then `class`, where a frame leaves only when every class in it is above the limit. A
   reviewer-confirmed hard negative, a frame holding the last training box of a class, and frames of a
   group at or below the median are never dropped. Dropped frames are counted (`balance_dropped`).
   Repeating raises only positive frames; the **negative share** (step 4) is then restored: the empty
   frames repeat as much as needed (at most `max_repeat` times) to keep their share of the images per
   epoch.

   **Never touched**: the held-out frames, the validation frames, the scene filter, the class list.
   The `balance` report in `dataset.json` has the counts before and after (boxes and frames per class,
   frames per scene, camera and camera/day, images per epoch, the largest repeat, dropped frames), and
   the same report is shown **before** a run: the Training page's preview panel, or
   `train prepare --dry-run` (also without balancing: the split and the boxes per class). Warnings: a
   group with a single frame repeated `max_repeat` times (only repeats, no new information), and
   images per epoch more than doubled.

   RF-DETR's night augmentation adds one augmented copy per *unique* training image, so repeats do not
   multiply the augmented copies.

**When to balance.** `class` when animals or other rare classes sit beside many people and cars;
`scene` for a mixed model with few night frames; `camera` or `camera-day` when one camera or one busy
day dominates the set. Repeats add no new information: they only make the trainer see the same frames
more often, so collect more frames of rare groups when you can, and keep `max_repeat` low for groups of
a few frames (they overfit first). Images per epoch grow, so epochs take longer; patience counts epochs.
Try it with `--dry-run` first, and compare against an unbalanced run on the same held-out frames.

Output in `<training dir>/<run id>/data/`: `images/`, `labels/` (hard links, else copies),
`classes.txt`, `train.txt` / `valid.txt` (repeats as repeated lines), `data.yaml` (Ultralytics),
`train.coco.json` / `valid.coco.json` (category ids 0..C-1, `file_name` relative to the folder; a
repeated frame is a separate image entry), `heldout/` (a benchmark `dir:` set: `images/`, YOLO
`labels/`, `classes.txt`, `manifest.json` with tags and `scored_labels`) and `dataset.json` (sources,
licences, counts, warnings). `train prepare --out <dir>` writes the same without training.

### Class inventory

`blue-onyx-prism train classes --dataset <export>... [--scene <s>] [--rename from=to] [--json]` lists
the classes of the datasets after the renames: boxes, frames holding one, and boxes per scene (day,
night, ir, unknown scene). With `--scene` an **in scope** column adds the boxes of those scenes
(unknown-scene boxes count only for `all`). The Training page's class picker shows the same counts
(`GET /v1/training/dataset-classes`) and warns about a class with fewer than 50 boxes (a fine-tuning
floor, far below Ultralytics' best-case 10,000 instances) and about a class with no box in the
scenes the run trains on (`person` with no night boxes in a night run).

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

## Starting points and convergence

With nothing typed in, a run starts from the family's recommendation (`recipe::recommended`, shown
beside each family on the page). With early stopping on, **epochs is a cap**: the run normally ends
sooner.

| Family | Epochs (cap) | Patience | Min epochs | Min delta | Smoothing | Learning rate |
|---|---|---|---|---|---|---|
| YOLO26 / YOLO11 / YOLOv8 | 100 | 15 | 10 | 0.001 | 3 epochs | the trainer's |
| RF-DETR | 100 | 10 | 10 | 0.001 | 3 epochs | 1e-4 (rfdetr's default); 5e-5 below 1000 training images |
| RT-DETRv2 / D-FINE | 80 | 10 | 10 | 0.001 | 3 epochs | 1e-4 |

With fewer than 50 validation images a **recommended** patience doubles (10 -> 20, at most the
epochs; `recipe::SMALL_VALID_IMAGES`): the mAP of so few images jumps by several points between
epochs. A typed patience (form, `--patience`, `training.patience`) is never changed. The first lines
of the train stage in `train.log` say which applied, e.g. `# convergence: 16 validation images
(< 50): patience 20 instead of 10; mAP averaged over the last 3 epochs`.

**RF-DETR learning rate and schedule.** rfdetr 1.11 defaults to a *step* schedule that drops the
learning rate x0.1 only after epoch 100 (`lr_drop`); the convergence rule usually stops before, so the
rate never decayed (run `20261031-025744-2a83` stopped at epoch 93). The trainer therefore passes
`lr_scheduler="cosine"` with `lr_scheduler_kwargs={"min_factor": 0.1}` (`--lr-scheduler cosine|step`,
`--lr-min-factor`): rfdetr's managed cosine decays from the learning rate to 0.1x of it over the
planned epochs (its total steps are Lightning's `estimated_stepping_batches`), after its warm-up.
With no learning rate typed and fewer than 1000 training images (`recipe::SMALL_TRAIN_IMAGES`) an
RF-DETR run uses lr 5e-5 and encoder (backbone) lr 7.5e-5 instead of rfdetr's 1e-4 / 1.5e-4; a typed
lr is used as is with an encoder lr of 1.5x it (`TrainRecipe::effective_lr`, `--lr` / `--lr-encoder`).
`train.log` says which applied, e.g. `# lr: 0.00005, encoder 0.000075 (312 training images < 1000:
lower than RF-DETR's default 0.0001 for small datasets); schedule cosine (default), final factor 0.1
(default)`. An rfdetr that does not take these options trains without them (logged).

### Learning rate settings

The Options step has a **Learning rate** field (empty = the family's default, the placeholder follows
the ticked family: `data-default-lr` on each size checkbox); *Advanced* has **LR schedule** (default,
cosine, linear, step) and **Final LR factor** (the last epoch's rate as a share of the starting one,
0-1). Blank fields fall back to `training.lr`, `training.lr_schedule` and `training.lr_min_factor`
(prefilled on the page; a stored schedule or factor a family does not take is skipped for that
family: `TrainingConfig::apply_lr`), else to the family's default. `train run` takes `--lr`,
`--lr-schedule` and `--lr-min-factor` the same way. The job passes `--lr-scheduler` /
`--lr-min-factor` to the script only when one is set, so each trainer otherwise keeps its own.

| Family (trainer) | Learning rate | Schedules (default first) | Final factor (default) |
|---|---|---|---|
| YOLO26 / YOLO11 / YOLOv8 (Ultralytics) | auto: the optimizer picks (AdamW `0.002 * 5 / (4 + classes)` up to 10,000 iterations, else MuSGD 0.01); a typed rate uses AdamW | linear, cosine (`cos_lr`) | 0.01 (`lrf`) |
| RF-DETR (rfdetr) | 1e-4, 5e-5 below 1000 training images; encoder 1.5x | cosine, step (rfdetr's own x0.1 after epoch 100) | 0.1 (the cosine's end and the anneal's) |
| RT-DETRv2 / D-FINE (transformers) | 1e-4 | cosine, linear | 0 (`cosine`; a factor above 0 is `cosine_with_min_lr` with `min_lr_rate`, transformers 4.38+, pinned 5.17.0) |

A schedule a family does not take is refused when the run is queued, with the allowed ones in the
message (`recipe::lr_schedule_error`); so is a factor above 0 with transformers' linear schedule (it
always ends at 0). With RF-DETR's step schedule the factor only ends the anneal before a convergence
stop. `train.log` names everything: `# lr: ...; schedule linear (default), final factor 0.01
(default)`.

**Live learning rate.** Every `batch` event carries `lr`, the main parameter group's current rate
(None when unreadable): Ultralytics' non-bias group with the lowest base rate (in 8.4 `build_optimizer`
orders weight / bn / bias groups and MuSGD adds a x3 copy for the class heads; the bias group falls
from `warmup_bias_lr` 0.1 during warm-up, so it is never the one shown), rfdetr's first group (the
head, at `lr`; the backbone and decoder groups are scaled), transformers' `lr_scheduler.get_last_lr()`.
Ultralytics' epoch event also has that rate as an explicit `lr`, which wins over `lr/pg0` (a raw key
that is a canonical id itself wins over the trainer's aliases of it, `progress::TrainProgress::merge`).

**Progress fields** (`runs[].progress` of `GET /v1/training`, all absent / null on older records):
`lr` (the latest batch event's rate, else the latest epoch's `lr`), `stop_epoch` (the epoch the
convergence rule stops after when nothing improves from now on; null while early stopping is off and
after a stop), `batch_size` and `imgsz` (the start event's, the job's halved batch before it after an
out-of-memory retry), and `settings`: `lr` / `lr_encoder` (null = the trainer's own), `lr_schedule`
and `lr_min_factor` in effect, `epochs`, `patience` in effect and `patience_doubled` (a small
validation split), the requested `batch`, `workers`, `seed` (a seed repeat's), `scene` (`all`,
`night+ir`, ...) and `gpu` (its name). The stage text ends with `lr 2.3e-4`.

**On the page** the metrics view shows a Learning rate tile (the current rate with its change from
the previous epoch) and an LR curve once the history has `lr`, a Convergence tile ("since best 4 / 10,
stops at epoch E", the anneal or extension when active; hidden with patience 0), the finish time
under Time left, a run settings line (GPU, batch and input size, learning rate and schedule, cap and
patience, seed, scene, workers) and an overfitting hint when the validation loss rises while the
training loss falls.

Sources: the Ultralytics fine-tuning guide (patience 10-20 when fine-tuning) and the `cfg` reference
(<https://docs.ultralytics.com/guides/finetuning-guide>, <https://docs.ultralytics.com/usage/cfg>);
the RF-DETR training docs (100 epochs, lr 1e-4, effective batch 16, early stopping patience 10 and
min delta 0.001, <https://rfdetr.roboflow.com/learn/train/>). RT-DETRv2 and D-FINE converge in fewer
epochs from COCO weights. Fine-tuning trains **all layers** (no frozen backbone):
<https://docs.ultralytics.com/yolov5/tutorials/transfer-learning-with-frozen-layers>.

**How much data.** Ultralytics' best-case targets are at least 1500 images and 10,000 instances per
class, and varied images (different lighting, weather, cameras) matter more than their count
(<https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results>); Frigate+ asks for
at least 100 verified images per camera (<https://docs.frigate.video/plus>). Fewer still trains, but
expect a noisy held-out AP.

**The convergence rule** is one rule for every trainer (`Convergence` in `scripts/train_model.py`;
its display twin is `progress::convergence_run`, parity-tested), and the trainers' own early stopping
is switched off so only this one decides:

- The value is the validation **mAP50-95**, else mAP50 when the trainer reports no mAP50-95,
  **averaged over the last `smooth` epochs** (3; fewer at the start; `--smooth 1` = each epoch's
  value). A single noisy peak no longer starts the countdown: run `20261030-010233-b69d` (RF-DETR
  nano, 16 validation images) stopped at epoch 15 on its epoch-5 peak with the raw value, while its
  3-epoch mean still gained at epochs 7, 9 and 15.
- An improvement is a mean above the best mean so far by more than `min_delta` (0.001).
- A **stop point** is reached once `epoch >= min_epochs` (10) and `patience` epochs passed since
  the last improvement. `patience` 0 turns it off (the run trains all its epochs); it may not exceed
  the epochs, and a patience that is not below the epochs never fires (a warning says so).
- At a stop point the run **goes on while the training loss still falls**: over the last `patience`
  epochs, the mean training loss of the later half (n / 2 epochs; an odd middle one counts in
  neither) is more than 2% below the earlier half's (`loss_falling`; fewer than 4 losses = not
  falling). Then the countdown restarts there (the next stop point is `patience` epochs later, or
  sooner past a gain), at most 2 times (`MAX_EXTENSIONS`) and never at the last epoch; otherwise it
  stops. The script prints `@train {"event":"converge_point","epoch":E,"best_epoch":B,"best":v,
  "loss_falling":true,"loss_from":a,"loss_to":b,"extension":n,"next_check":E+patience}` and the
  Training page shows "Early stop reached at epoch 15 (best epoch 5, mAP 0.835); training loss is
  still falling (3.569 -> 3.315). Continuing to epoch 25." with **Stop now and keep epoch 5** (the
  Finish now action) and **Continue** (hides the notice; the run goes on by itself, so an unattended
  run never waits). The incident run with its losses passes its epoch-15 stop point this way.
- **Anneal, then stop** (RF-DETR, `--anneal N`, 0 = off for the other trainers, whose schedules
  already decay): the first stop point that would stop (no extension left or the loss no longer
  falling, before the last epoch) instead trains `N` more epochs, at most up to the epochs, with a
  lowered learning rate, then stops ("converged"); stop points during the anneal are ignored and
  Finish now still stops at once. `N = max(5, ceil(patience / 2))` with early stopping on
  (`TrainRecipe::anneal_epochs`; 5 for the recommended patience 10, 10 for a doubled 20), in the
  `# convergence:` line of `train.log`. The script bends rfdetr's per-step `LambdaLR` from its
  current factor f0 down to `--lr-min-factor` (0.1) along a half cosine over those epochs' optimizer
  steps (never above the schedule; `bend_lambda_lr`, tested with a stand-in scheduler); another
  scheduler is logged and only anneals-then-stops. It prints `@train {"event":"anneal","epoch":E,
  "until":U,"best_epoch":B,"best":v,"loss_falling":f,"extensions":n}` and the page shows "Early stop
  reached at epoch E (best epoch B, mAP v). Lowering the learning rate until epoch U, then
  stopping." (with Stop now / Continue) and "lowering the learning rate until epoch U, then
  stopping" in the run's progress line. The best checkpoint is still the one exported, usually one
  of the annealed epochs.
- The **best checkpoint** (highest raw mAP, not the smoothed one) is exported, not the last: Ultralytics `best.pt`, RF-DETR
  `checkpoint_best_total.pth`, and for RT-DETRv2 / D-FINE (transformers) the best model state, which is
  restored in memory before the export (the log names the exported epoch).

**Finish now.** A run in its train stage can be stopped after the current epoch, keeping the best
checkpoint (it then exports, installs and benchmarks as usual): the **Finish now** button beside Cancel
on the Training page (loopback / admin, like Cancel; a paused run is resumed first), or create the file
`<run dir>/work/stop-request` (the run dir is `<training dir>/<run id>`, see `train list`). The trainer
checks the file after each epoch; a stale one is removed before training starts. Ctrl-C in the CLI still
cancels.

**The `stop` event.** When the rule (or the stop file) ends training the script prints
`@train {"event":"stop","reason":"converged"|"finish","epoch":N,"best_epoch":B,"best":v,"loss_falling":f,"extensions":n,"anneal_until":U|null}`
(the log line says why: "training loss no longer falling", that the extensions or epochs ran out, or
that the learning rate was lowered for N epochs after the stop point). The page shows
"best epoch B, M epochs without improvement (stops at patience P)" while training and then
"converged at epoch N (best B)" or "finished on request at epoch N (best B)"; `train run` prints the same
in its progress lines.

**Config.** `training.epochs` and `training.patience` are unset by default: unset means the family's
recommendation, a number overrides it for every family. A config written by an older version that has
`epochs: 50` (the old default, which every save wrote out) reads as unset. The CLI's `--epochs` and
`--patience` and the page's blank fields follow the same order: the typed value, else the config, else
the recommendation.

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
   trainer reports it, `eval`, `phase`, `done`; `epoch` / `eval` also carry every scalar metric, see
   [Metrics per epoch](#metrics-per-epoch)); everything else goes to `<run>/train.log` (capped
   at 20 MB). Out of GPU memory, missing kernels and unusable images get their own messages (lower the
   batch or input size, ...); other failures go through the export's classifier. **Out of GPU memory
   before 3 finished epochs** restarts the training at half the batch (down to 1, at most 3 restarts):
   the log, the stage text and `describe()` say "out of GPU memory at batch 16, retrying with batch 8",
   and the run record and the provenance keep the batch that trained. RF-DETR keeps its effective batch
   16 through gradient accumulation (`grad_accum_steps = 16 / batch`), so a smaller batch only costs
   speed. On macOS the trainer runs with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` and
   `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8` (unless the service's environment sets them): MPS may
   otherwise allocate 1.7x its recommended working set, more than the RAM of a 16 GB Mac, and an
   oversized batch then swaps for half an hour before it fails (run `20261027-071956-7f1c`: RF-DETR
   nano 384, batch 16, 35 min for 148 images, then out of memory). With the cap, batch 16 fails in
   about a minute and the retry takes over.

   **Data loader workers** (`--workers`): none on macOS; else all cores but one (1 to 8), and on
   Windows also at most what the free system commit (RAM + paging file) holds when training starts:
   `(free commit - 6 GB) / (1.4 GB x loaders)`, down to 0 (data loaded in the training process:
   slower, never short of memory). On Windows every worker is a spawned Python that imports PyTorch
   (measured ~0.9 GB commit for the import, ~1.4 GB per worker while an RF-DETR nano run trained; its
   trainer process itself committed 5.5 GB, hence the 6 GB reserve), and the validation loader's workers stay alive beside the training
   loader's: `loaders` is 2 for RF-DETR and transformers (both loaders persistent) and 3 for
   Ultralytics (its validation loader gets twice the workers). The decision is the first line of the
   train stage in `train.log`: `memory: commit 21.3 GB limit, 5.1 GB free -> 0 data loader workers
   (cores allow 5)`. `training.workers` in the config forces a count (0 = none; absent = automatic).
   **Out of system memory** (`[WinError 1455]` "the paging file is too small", `DefaultCPUAllocator:
   not enough memory`, `can't start new thread`, `MemoryError`, a worker killed by the OOM killer)
   restarts the training with half the workers and the same batch (at most 3 restarts, at any epoch:
   the worker count does not change the result); the log says "out of system memory with 5 data
   loader workers, retrying with 2". Such a line also means the GPU was not the problem: on Windows
   (WDDM) a CUDA allocation fails too when the commit is exhausted ("CUDA out of memory ... 6.17 GiB
   is free"), so a failure with both is not retried at a smaller batch (run `20261029-235307-ae21`:
   RF-DETR nano on a GTX 1070 Ti with a fixed 5.6 GB paging file, 5 + 5 workers, halved to batch 1
   and still failed when the validation workers spawned). On a Windows machine with a fixed paging
   file, let Windows manage its size (System Properties > Advanced > Performance > Virtual memory).
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
   run (they show on the Benchmark page). It runs on the devices ticked under **Benchmark on
   devices** (`training.benchmark_devices`, saved from the Train form; `--benchmark-device`,
   repeatable), one after the other, each released before the next loads; none ticked = only the
   device the model would load on (its own device, else what `auto` picks). It never sweeps every
   device of the machine (a GTX 1070 Ti + UHD 630 box hard-hung within seconds of that sweep
   starting, twice). A failed benchmark is reported but keeps the model.

Default name: `<scene>-<model>-<YYYYMMDD-HHMM>`: `custom-` for `all`, `day-`, `night-` for `night+ir`,
`night-only-`, `ir-`, `day-night-`, `day-ir-`. Several models can be queued at once (one per ticked
size, or `--model` repeated), and several scenes queue one run per scene per model
([Comparing](#comparing-day-nightir-and-mixed-models)); they run one at a time. Cancel
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

### System load during training

Training, import and auto-label keep every core (and the GPU) busy for hours, so their helpers run
**below normal priority**: `Cmd::low_priority` on the training environment's commands
(`train_python_cmd`), applied by the command runner at spawn. Windows: `BELOW_NORMAL_PRIORITY_CLASS`
added to the creation flags (beside `CREATE_NO_WINDOW`; the job object is unchanged), inherited by
what the helper starts (DataLoader workers, ffmpeg). Unix: nice raised to at least 10 between fork
and exec (`backend::priority::lower_child`; a higher nice, e.g. `Nice=15` in the unit, is kept),
inherited by its threads and children. The model export's own commands (uv, the export script) keep
the normal priority.

The export after training runs in the same process as the training, which is where a 16 GB Windows
machine with a fixed paging file froze (hard reset right after an RF-DETR run finished: the trainer
committed 5.5 GB, four persistent DataLoader workers 1.4 GB each, 1.2 GB commit free). Between
training and export `train_model.py` now (`prepare_export`):

- **releases the training state** (`release_training_memory`): the trainer drops its trained model
  / trainer objects (Ultralytics' YOLO with its trainer and the base model; rfdetr's model with its
  Lightning trainer; transformers' `Trainer`, accelerate's references and the best epoch's CPU copy),
  stops every live multi-process DataLoader iterator found by `gc` (`_shutdown_workers()`: persistent
  workers outlive `fit`), collects and empties the CUDA / MPS cache. With psutil (in the training
  environment) the log line gives the process's RSS, commit and child processes before -> after.
  RF-DETR holds one model at a time: the export's transformers model is dropped before the ONNX
  Runtime check (`release_model`, an option of `export_rfdetr` that the plain model export leaves
  off), and rfdetr's own comparison model before the comparison session;
- **caps the CPU threads** of the trace and the ONNX Runtime checks to all physical cores but one
  (`BOP_EXPORT_THREADS`, set by the service; `torch.set_num_threads`, and the checks' session
  `intra_op_num_threads`; the plain model export leaves ONNX Runtime's default);
- **waits for memory**: with less than 3 GB available physical memory or (Windows) 4 GB available
  commit it logs `warning: low memory (...): waiting up to 5 min for memory to free` and an export
  `phase` event with that detail (in `train.log`), polls every 10 s, and after 5 minutes (or a
  Finish now / stop file that appears during the wait) exports anyway: the check never fails a run.

Training itself keeps PyTorch's thread count (the GPU does the work).

### Metrics per epoch

Each `epoch` and `eval` event has `metrics`: every scalar the trainer reports for the epoch under its
own key, plus the script's own:

| Trainer | Keys (examples) |
|---|---|
| RF-DETR (rfdetr 1.11 Lightning `callback_metrics`) | `val/mAP_50_95`, `val/mAP_50`, `val/mAP_75`, `val/mAR` (the EMA model by default), `val/ema_*`, `val/F1`, `val/precision`, `val/recall` (at the best macro-F1 confidence), `train/loss`, `train/lr`, `train/loss_ce`, ... |
| Ultralytics (`trainer.metrics`, `tloss`, `lr`) | `metrics/mAP50-95(B)`, `metrics/mAP50(B)`, `metrics/precision(B)`, `metrics/recall(B)`, `val/box_loss`, `train/cls_loss`, `lr/pg0`, ...; `val_loss` = the sum of the `val/*_loss` |
| transformers (RT-DETRv2, D-FINE) | `eval_loss`, `learning_rate`, and a detection evaluation after each `evaluate` (an extra pass over the validation split): mAP 50-95 / 50 / 75 and mAR@100 with torchmetrics (COCO rules), P / R / F-beta from a greedy IoU-0.5 matching swept over 101 confidences |
| all (the script) | `f1`, `f1_thr`, `f2`, `f2_thr`, `f2_precision`, `f2_recall`, `f0_5`, `f0_5_thr`, ...: macro F-beta (`(1+b^2)PR/(b^2P+R)` per class, averaged over the classes with ground truth) at the best confidence of the sweep (rfdetr's own 101-threshold rows, Ultralytics' `p_curve` / `r_curve`); `mem_gb` (peak GPU memory: CUDA `max_memory_reserved`, MPS the driver's allocation sampled after every batch); `img_s` (training images per second, validation excluded) |

F2 weighs recall four times precision ("don't miss a person"), F0.5 the other way; the `*_thr` columns
are a starting confidence threshold for the trained model. NaN, infinite and non-numeric values are
dropped; at most 64 metrics per epoch (the shortest keys first).

The service maps the keys to stable ids (`src/training/progress.rs` `canonical`): `map`, `map50`,
`map75`, `mar`, `precision`, `recall`, `f1`, `f1_thr`, `f2`, `f2_thr`, `f2_precision`, `f2_recall`,
`f0_5`, `f0_5_thr`, `loss`, `val_loss`, `lr`, `mem_gb`, `img_s`, and `ema_map` ... for rfdetr's
`val/ema_*`; unknown keys keep their raw name (`val/AP/raccoon`, `train/loss_ce`), so nothing is lost
when a library renames one. Each run's `progress` in `runs.json` and `GET /v1/training` has
`history` (one point per epoch: `epoch`, `elapsed_s`, `metrics`; an `eval` event fills its epoch's
point; at most 1000 points) and `best` (`{metric, epoch, value}` on `map`, else `map50`, else the
lowest `val_loss`), which the stage text shows as "best mAP50-95 0.711 @ epoch 3". `GET /v1/training`
also has `metrics` (label, short label, `higher_better`, `format` of every id) and `metric_columns`.
Records written before this keep loading (no `history`: the page shows no metrics block).

**The metrics view** (Training page, every run with a history, running or finished): KPI tiles
(epoch, ETA, best primary metric with its epoch, latest loss and validation loss with the change
from the previous epoch), curves of the losses and of the quality metrics with the best epoch marked,
and a per-epoch table, newest first. **Columns** picks the table's columns from every metric in the
run's history (known ids by label, raw keys by name) and their order; the choice is saved as
`training.metric_columns` (`POST /v1/training/metric-columns` with `{"columns": [...]}`: at most 20
distinct names of 1-64 characters, known ids or raw names; answers the saved list, 400 otherwise) and
applies at once. Empty (the default) = `map, map50, f1, f2, precision, recall, loss`. The raw trainer
log is under a closed *Log* disclosure.

## Comparing day, night/IR and mixed models

Whether a night model or a mixed model serves a camera better is something to measure on your own
frames: the research behind these defaults found no source that compared one mixed model against
separate day and night models on CCTV near-infrared footage, which is why the comparison exists.

**Queue it.** On the page tick several scene checkboxes (day, night, ir; each combination is a run), or
repeat the flag:

```sh
blue-onyx-prism train run --model yolo26s --dataset <export> \
    --scene day --scene night+ir --scene all
```

One run per scene per model is queued (several `--model` x several `--scene` = the product; identical
scenes are queued once; the page lists the run names first) and they run one at a time in that order.
They share a `compare_group` (`cmp-<YYYYMMDD-HHMMSS>`), the class list and the excluded mode.

**Same held-out frames.** The held-out set is chosen before the scene filter, so every run of the
group is benchmarked on the same frames, all scenes included. The fingerprint in each run's benchmark
summary proves it. If the fingerprints differ (an export was edited between queuing and running) the
comparison shows a warning instead of comparing.

**The table.** The Comparison card of the Training page (one per group) and, at the end of a
`train run` with several scenes, the CLI table have one row per run: model, scene filter, best epoch
and the epoch it stopped at, held-out AP50-95 for `day`, `night`, `ir` and `all` frames
(from the benchmark's per-tag results), the AP over the scenes the run trained on (**in-scope**, the
run's headline), the p50 time, and what alerting sees at the model's confidence threshold over every
held-out frame: **Person R** (person recall), **Vehicle R** (vehicle recall, car / truck / bus merged,
so a car found as a truck counts) and **False alerts** (`3/130`: frames without any object that got a
detection at or above the threshold, of all such frames; a detection inside an ignore region does not
count). The best value of each column is marked (`*` in the CLI): highest AP and recall, lowest time
and false-alert rate. Runs of older versions show a dash in the new columns. Runs that differ in
classes, excluded mode or threshold get a warning.

The same figures close each run's benchmark line on the page (`person recall 0.39 · vehicle recall
0.67 · false alerts 4/130 empty frames @ 0.50`) and are stored in its benchmark summary (`threshold`,
`person_recall`, `vehicle_recall`, `empty_frames`, `empty_frame_alerts`), from the benchmark's overall
score (`benchmark.json` `accuracy.overall` has the same four fields; IoU 0.5, same matching as its
precision / recall). Like the overall AP, they cover every dataset of the run, built-in sets added to
the held-out set included.

**The routed row.** When the group has a day run and a night or IR run, a **day + night routed** row
puts the day run's day AP beside the night run's night / IR AP, which is what a service with
`night_model` routing would score. The scenes are scored separately, so no combined AP is invented.

**Deploying.** Either a day model as the `default_model` plus the night model as its `night_model`
(Trained models card: *Default model*, *night model of*), or the mixed model alone, whichever the table
favours on the scenes your cameras see most. A scene with few held-out frames has a noisy AP; read the
frame counts (`images`) beside it.

## Estimates

The page and `train estimate` show GPU memory and time per epoch from a small table
(`src/training/estimate.rs`): `vram = base + per_image x batch x (side / default)^2` MB, where `side`
is the input size, except for RF-DETR trained multi-scale: rfdetr (`multi_scale="per-batch"`,
`expanded_scales=True`) resizes every sample to its largest scale `(imgsz / (patch x windows) + 5) x
patch x windows` (nano 384 -> 544, small 512 -> 672, medium 576 -> 736, base 560 -> 840, large 704
-> 864), so its memory follows that side (about twice the input size's). RF-DETR nano's per-image cost
(650 MB at 384 px) is calibrated on the run that ran out of memory on a 16 GB M1 (batch 16 needed over
20.4 GB; the estimate now says 22.4 GB) and on the measurements below; a 16 GB Mac now gets batch 4
suggested instead of 16. The other RF-DETR sizes keep their constants, now scaled by their multi-scale
side, so their estimates rose 1.5-2.25x (conservative until measured).

**RF-DETR multi-scale** (`multi_scale` in the recipe: page form field `multi_scale`, `train run
--multi-scale auto|on|off`, script `--multi-scale on|off`): `auto` (default) keeps rfdetr's
multi-scale training when the estimate fits batch 4 or more with it, else trains at the input size
only (`multi_scale=False`), which halves the memory; `GET /v1/training` shows the choice per size
(`multi_scale` of each RF-DETR size). Multi-scale is an augmentation: off may cost some accuracy.

**Measured batches.** Once a run reported `mem_gb`, the suggested batch of that family, size and
default input size (and RF-DETR multi-scale setting) on the same GPU comes from the measured peak
(`measured: true` in `GET /v1/training`): linear in the batch above the table's fixed part (at most
half the peak), 90% of the VRAM on CUDA, the whole MPS share on a Mac (the driver's allocation
includes the allocator's cache, the share is already 60% of the RAM against PyTorch's own 74%).

Time: `epoch = images / (images per second x GPU factor) x (imgsz / default)^2`, images per second
for a GTX
1080-class card (or an M1), the GPU factor by generation (Volta 2.0, Turing 1.6, A100 4.0, Ampere
consumer 2.2, Ada 3.2, Hopper 6.0, Blackwell 4.5). The M1 numbers of yolo26n, rfdetr-nano and
rt-detrv2-s come from the smoke runs above; everything else is a round number from public reports,
good to a factor of about 1.5. The suggested batch is the largest power of two whose estimate fits 90%
of the memory (capped at 16 when not given). The inference hint (light / medium / heavy) is the size
class on the service's devices: nano / small run on a CPU or a UHD 630; medium and up want a discrete
GPU; the benchmark at the end of the run measures it.

### Speed and memory on an M1 (measured)

Apple M1, 16 GB, macOS, PyTorch 2.14.0 (MPS), rfdetr 1.11.2, Ultralytics 8.4.163; the dataset of run
`20261027-071956-7f1c` (148 training, 17 validation images, 17 classes). `train_model.py` run directly
for 2-3 epochs (`img_s` and `mem_gb` from its epoch events; epoch time includes the validation). One
run per row; repeats of the baseline varied by about 4% in `img_s`.

| Change (RF-DETR nano, 384, batch 4) | Epoch time | `img_s` (epoch 1 / 2) | Peak `mem_gb` (epoch 1 / 2) | Adopted |
|---|---|---|---|---|
| Baseline: workers 0, multi-scale on, fp16 AMP | 86-90 s | 1.90-1.95 / 1.84-1.86 | 6.5 / 8.0-9.0 | - |
| 2 data loader workers (`spawn`, persistent) | 85-91 s | 1.96 / 1.94 | 6.5 / 9.0 | no: within noise (GPU-bound) |
| 4 workers | 85-93 s | 1.95 / 1.94 | 6.5 / 8.0 | no |
| Multi-scale off (384 px only) | 84 s | 1.98 / 1.98 | 3.9 / 3.8 | as `auto` when batch 4 does not fit |
| AMP off (fp32) | 88 s | 1.92 / 1.88 | 7.5 / 9.9 | no: no faster, more memory |
| AMP bf16 | - | - | - | n/a: rfdetr keeps fp16 on MPS (warns) |
| No `gc.collect()` + `empty_cache()` after validation | 87-89 s | 1.94 / 1.86 | 6.5 / 9.2 | kept the release (0.2-1.2 GB lower in epoch 2, same speed) |
| Batch 16 (the failed run's), MPS cap 1.0 | - | - | out of memory after 52 s | the cap: fails fast instead of swapping 35 min |
| Batch 8, MPS cap 1.0 | 86 s (through the CLI) | 1.97 | 11.4 (one run ran out of memory after 87 s) | batch 4 suggested |

| Change (YOLO26n, 640, batch 16) | Epoch time (epochs 2-3) | `img_s` | Peak `mem_gb` | Adopted |
|---|---|---|---|---|
| Baseline: no image cache | 15.8 s | 11.0 | 5.4 | - |
| `cache="ram"` (0.17 GB for 148 images) | 14.7-14.9 s | 11.5 | 5.4 | yes: `--cache auto` when the decoded set fits a quarter of the free RAM |
| 2 workers | 15.7 s | 11.0 | 5.4 | n/a: Ultralytics uses no workers on MPS |

Through the CLI (`train run --model rfdetr-nano --batch 16 --epochs 1`, debug build): out of memory
at batch 16 after about a minute, retried at batch 8, trained (11.4 GB peak, 1.97 img/s, mAP50-95
0.71) and exported: 2 min 48 s in total, against 35 min of swapping and a failure before.
`train run --model rt-detrv2-s --epochs 2`: the detection evaluation added mAP 0.58 / 0.68, F2 0.62
/ 0.82, 2.3-2.5 img/s, 7.1 GB.

Not measurable here (no NVIDIA GPU), implemented behind the capability checks and unmeasured: TF32
matmuls and cuDNN convolutions on compute capability 8.0+, `cudnn.benchmark` for Ultralytics (fixed
input size), transformers `bf16` instead of `fp16` on 8.0+, `pin_memory` and persistent workers on
CUDA, and Linux / Windows data loader workers raised from half the cores (at most 4) to all cores but
one (at most 8).

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
running helpers are killed, no further video starts). The helpers run below normal priority, like
every training-environment helper ([System load](#system-load-during-training)).

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

`collect.hidden_classes` (subset of `collect.classes`; `POST /v1/review/classes` with `hidden`; the
Review page's *don't want*) are classes you do not review: hidden from the Review page, left out of the
pre-fill and of auto-label runs (a run of hidden classes only is refused), and their pending boxes are
rejected (`reason: other_class`) when a class becomes hidden; they stay in the list and keep their YOLO
class ids (docs/DATASET.md, "Unwanted classes"). Removing a class from the list drops it from the hidden
list too.

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
blue-onyx-prism train estimate --model rfdetr-small --images 2000 --epochs 100 [--batch 8] [--imgsz 512]
blue-onyx-prism train classes --dataset <export> [--dataset <export>] [--scene night+ir] [--rename a=b] [--json]
blue-onyx-prism train prepare --dataset <export> [--dataset <export>] [--scene night+ir] \
    [--classes person,car,dog,cat,raccoon] [--excluded background|drop-frame] \
    [--rename motorbike=motorcycle] [--public [--allow-large]] \
    [--balance class,scene,camera,camera-day|off] [--balance-strength 0.5] [--balance-max-repeat 4] [--balance-cap 3] \
    (--out <dir> | --dry-run)
blue-onyx-prism train run --model yolo26n [--model rfdetr-nano ...] --dataset <export> \
    [--scene all|day|night|ir|night+ir|day+ir ... (repeat for a comparison)] \
    [--classes a,b] [--classes c] [--excluded background|drop-frame] \
    [--balance class,scene,camera,camera-day|off] [--balance-strength 0.5] [--balance-max-repeat 4] [--balance-cap 3] \
    [--epochs 100] [--patience 15 (0 = off)] [--batch 16] [--imgsz 640] [--lr 0.001] [--lr-schedule cosine|linear|step] [--lr-min-factor 0.05] [--night-aug] [--aug-dark p --aug-noise p --aug-gray p] \
    [--name n] [--install-dir <dir>] [--gpu cuda:N] [--benchmark-set nir-night] [--benchmark-device ort:cuda:0] [--no-register] [--no-benchmark] [--keep-work] [--allow-large] [--verbose]
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

`--epochs` and `--patience` default to the family's recommendation ([Starting points and
convergence](#starting-points-and-convergence)). A running training is finished early, keeping its best
checkpoint, by creating `<run dir>/work/stop-request` (or Finish now on the page). `--scene` is
repeatable: several scenes (or models x scenes) run one after another on the same held-out frames and the
comparison table is printed at the end. `night` alone is night frames only; use `night+ir` for both.
`--balance` takes axes comma-separated and/or repeated (no flag: `training.balance`, else off); the
strength, max repeat and cap flags change that default's values. See [Assembly](#assembly-srctrainingdatasetrs)
step 7. `prepare --dry-run` writes nothing and prints images per epoch before -> after, boxes and frames per
class, frames per scene, the top cameras (and camera-days), dropped frames and warnings; `prepare` and
`run` print the balance summary.

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
