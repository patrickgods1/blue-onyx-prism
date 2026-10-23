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
  pins a second OpenCV build that would overwrite Ultralytics' `cv2`), onnx, onnxruntime, onnxslim.
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
- **Extending it** (import, auto-label): add the packages to `requirements-model-train.txt`,
  regenerate the locks and add a `--self-check` module list in `scripts/train_model.py`. The
  environment's marker records the lock's name and SHA-256, so every existing environment is rebuilt
  once with the new packages.

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

## Command line

```sh
blue-onyx-prism train capability [--json]
blue-onyx-prism train estimate --model rfdetr-small --images 2000 --epochs 50 [--batch 8] [--imgsz 512]
blue-onyx-prism train prepare --dataset <export> [--dataset <export>] [--scene night+ir] \
    [--classes person,car,dog,cat,raccoon] [--rename motorbike=motorcycle] [--public] --out <dir>
blue-onyx-prism train run --model yolo26n [--model rfdetr-nano ...] --dataset <export> [--scene all|day|night+ir] \
    [--epochs 50] [--batch 16] [--imgsz 640] [--lr 0.001] [--night-aug] [--aug-dark p --aug-noise p --aug-gray p] \
    [--name n] [--benchmark-set nir-night] [--no-register] [--no-benchmark] [--keep-work] [--allow-large] [--verbose]
blue-onyx-prism train list
blue-onyx-prism train remove-env
```

Model labels: `yolo26n`..`yolo26x`, `yolo11n`.., `yolov8n`.., `rfdetr-nano|small|medium|base|large`,
`rt-detrv2-s|m|l|x`, `dfine-s|m|l|x`.

## Still to verify on an NVIDIA machine

- The CUDA locks install (`cu126` on Windows and Linux, `cu130` with a Blackwell card) and the
  self-check's test kernel passes on a Pascal card (GTX 10xx, `sm_61`).
- 1-epoch runs of every family (D-FINE only trains there), and the estimate table against the real
  memory use (`nvidia-smi` during a run); adjust the constants.
- RF-DETR base (patch 14, 560 px) through the transformers export (the other sizes use patch 16).
- Ultralytics AMP on CUDA (the AMP self-test is replaced, see `train_model.py`; GTX 16xx cards run
  without AMP as Ultralytics recommends).
