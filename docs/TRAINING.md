# Training custom models

Blue Onyx Prism can fine-tune a detection model on frames from your own cameras: frames the
[dataset collector](DATASET.md) saved, reviewed on the **Review** page and exported in YOLO format.
The result is a static ONNX model next to the others, added to the config and benchmarked on frames
it did not train on: whole camera days when there are several (other days of the same camera may
train), a share of the frames when there is only one. Use it as the default model, as the `night_model` of
another model (IR and dark frames) or as a `verify` model.

Training runs only when you start it (the Train area's Train tab or `blue-onyx-prism train run`), never from
`auto_download`.

## Can this machine train?

`blue-onyx-prism train capability` and the top of the Train tab answer this. Training needs one
of:

| Hardware | PyTorch build | Notes |
|---|---|---|
| NVIDIA GPU, compute capability 6.1-9.0 (GTX 10xx Pascal to Hopper), Windows or Linux x86_64 | CUDA 12.6 (`cu126`) | Any driver that supports CUDA 12. The 12.6 wheels carry kernels for 5.0, 6.0, (6.1 on Windows), 7.0, 7.5, 8.0, 8.6, 9.0; a GPU uses the kernels of its major version with the highest minor not above its own (8.9 Ada runs the 8.6 kernels). |
| NVIDIA GPU, compute capability 10.0 / 12.0 (Blackwell, RTX 50xx) | CUDA 13.0 (`cu130`) | Needs a CUDA 13 driver; the 12.6 build has no Blackwell kernels. |
| Apple silicon, macOS 14+ | PyPI (MPS) | The families that passed a 1-epoch MPS smoke run (below). |

Not supported: Intel and AMD GPUs (no pinned PyTorch build), Intel Macs, Linux arm64, CPU-only
machines. When training is
not available the page and the CLI say why for every GPU and give the by-hand path. The scripts are
in the `scripts` folder of the release archives and installers. The service does not read those copies:
it runs the ones embedded in the binary. On a machine with a GPU, or slowly on this one's CPU:

```sh
blue-onyx-prism train prepare --out prepared   # the same data options as `train run`
uv venv --python 3.11 .venv
uv pip install -r scripts/requirements-model-train.txt
# yolo26n.pt from https://github.com/ultralytics/assets/releases (AGPL-3.0, so is the result)
.venv/bin/python scripts/train_model.py --trainer ultralytics --family yolo26 --size n --name my-model \
  --weights yolo26n.pt --data prepared --out-dir models --work train-work \
  --export-script scripts/export_model.py --allow-cpu    # Windows: .venv\Scripts\python.exe
```

Then add `models/my-model.onnx` with its `.yaml` on the Config page (*Add a model by path*).
`--allow-cpu` lets the run fall back to the CPU and changes nothing when PyTorch sees a GPU. It ran end to
end, on a small synthetic set: 18 images, 1 epoch at 640 px, ONNX export and its check, 9 s on an Apple
M-series CPU with `--device cpu`. A real dataset on a CPU takes far longer than that suggests, and the
pinned environment is a large download. Docker images do not hold these scripts.

Sizes whose estimated memory does not fit the GPU even at batch 1 are shown but cannot be ticked.

**Several NVIDIA GPUs.** Every qualifying GPU is an option (`cuda_options` in `GET /v1/training`
and `train capability --json`, the one with the most VRAM first); that first one is the automatic
pick. With more than one, the Auto-label tab shows a **GPU** select under Advanced (labelers
mode) and the Train tab shows one under Advanced, and the CLI takes `--gpu cuda:N` (`train run`,
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
| YOLO26 | n, s, m, l, x | Ultralytics 8.4.184 | the model export's `model:yolo26<size>` `.pt` pins | AGPL-3.0 | `yolo26` (end-to-end `[1, 300, 6]`) |
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

Through the service (Train tab form, `POST /training/queue`): a 1-epoch yolo26n run took 14 s
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
- Contents: torch 2.14.1 / torchvision 0.29.1, ultralytics 8.4.184, transformers 5.20.2, accelerate
  1.15.0, rfdetr 1.11.2 with its training extra except `roboflow` (only for uploading to Roboflow; it
  pins a second OpenCV build that would overwrite Ultralytics' `cv2`), onnx, onnxruntime, onnxslim;
  for the footage import and auto-labelling (below) PyAV 18.1.0 (`av`; 19 needs Python 3.12) and yt-dlp 2026.10.21.
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

The Train tab lists the folders in `training.datasets` and the Dataset tab's exports under
`<training dir>/exports/`. Two layouts are read:

1. **The Dataset tab's YOLO export** (`src/collect/labels/export.rs`):

   ```
   <export>/
     classes.txt                                class id = line number
     data.yaml
     images/train/<scene>.<camera>.<date>.<stem>.jpg
     images/val/...                             held out by the export, whole camera days
     labels/train/<same stem>.txt               YOLO `class cx cy w h`, normalized; empty = no objects
     labels/val/...
     benchmark/manifest.json                    tags: scene, camera:<c>, date:<d>, hard-negative
     ignore.json                                ignore regions per image, frames with only parked-spot boxes
     frames.json                                frames confirmed empty, frames accepted in bulk
     hard_negatives.json                        rejected boxes
   ```

   Scene, camera and date come from the flat file name (or the manifest tags). The export's `val`
   split (whole camera + day groups) becomes the run's **held-out** set: never trained on, used for
   the benchmark. `test/`, `valid/` and `heldout/` folders count as held out too. An export that
   holds nothing out (a single camera day goes wholly to one side) is held out by the run instead,
   step 2.
2. **A flat YOLO folder**: `images/` (or the folder itself), `labels/<stem>.txt` or `<stem>.txt` beside
   the image, `classes.txt` or `data.yaml` (`names`), optionally `heldout.txt` / `test.txt` listing
   the held-out images, and a `manifest.json` in the benchmark format whose tags (`day` | `night` |
   `ir` | `grayscale`, `camera:<c>`, `date:<YYYY-MM-DD>`, `hard-negative`) give scene, camera and day.

A missing label file means "no objects" (the Ultralytics convention). `ignore.json` beside the labels
lists [ignore regions](#ignore-regions); no file means none.

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
2. **Same frame in several exports**: a frame ticked twice would both train and be measured on. By
   the flat image name (`<scene>.<camera>.<date>.<stem>.jpg`, what survives into every export; other
   names never match), one copy stays: the held-out one, else the first export's. Counted
   (`duplicates_dropped`) and warned.

   **Held out**: chosen from **all** review frames, *before* the scene filter: the export's split when
   it leaves frames on both sides; else whole `camera/day` groups, in a seeded hash order, until
   `training.heldout_share` (0.2) of the frames, never all groups. With a single camera and day the
   frames are split one by one, with a warning that frames of one day are similar and the benchmark
   reads better than the model is. The explicit options stay: `holdout_cameras` / `holdout_days` on
   the export, `training.heldout_share` 0 for no held-out set (a warning; every other empty held-out
   set is refused, [Pre-flight](#pre-flight-report)). A whole camera day held out means that day's
   frames are unseen, not that the camera is: other days of it train unless the export names it in
   `holdout_cameras`. The held-out set
   keeps every scene, so a day model is also measured on night frames, and the same exports give the
   same held-out frames whatever the scene filter. `dataset.json` holds `heldout_fingerprint` (SHA-256
   of the sorted held-out frame keys, a frame's ignore regions part of its key, so runs scored with other
   regions do not compare; a set without regions keeps the value older versions wrote), copied into the
   run's benchmark summary; the held-out
   `manifest.json` tags each frame with its scene plus `heldout:<scene>`, `heldout:all` and, for a
   frame of the run's scenes, `heldout:in-scope`, which the benchmark summary reads for per-scene AP.
3. **Scene filter** (training and validation frames only): `all`, or any combination of `day`,
   `night` and `ir` joined by `+` (`day`, `night`, `ir`, `day+night`, `day+ir`, `night+ir`; the
   page's checkboxes). Frames without a scene tag are left out by every filter but `all`, with a
   warning. **Behaviour change**: `night` alone now means night frames only and `ir` IR frames only;
   use `night+ir` for both (what `night` used to mean). Saved requests and configs with the old
   spellings still load. In-scope frames with [ignore regions](#ignore-regions) leave the pool next.
   **Parked-spot cap**: frames whose only boxes a labelers run accepted at a parked spot
   ([Spot memory](#spot-memory-of-fixed-cameras)) are nearly the same crop again and again, so at most
   20 of them per spot stay for training and validation together, by seeded rank
   (`dataset::PARKED_CAP`). A frame with several spots stays while one of them has room. The rest
   count as `parked_capped` in `dataset.json`, with a warning. Held-out frames are not capped. The
   frame where you pressed *Real, parked* holds a reviewer's box, so it is not capped.
   **Static cap** (after the parked-spot cap, on the positive frames it left): fixed cameras see the same
   parked cars at the same pixels in most frames (on one set of 851 reviewed frames, 2,138 of 2,357 car
   boxes stood at [static spots](#static-spots)), whoever labelled them, so the plan finds the spots itself,
   by geometry over the training and validation pool's own frames after the scene filter (exports of any
   version, no `ignore.json` needed; the frames the parked-spot cap took out are in that pool too: they
   are looks at the place like any other, they only stay out of training): per camera, the places where
   boxes of one object (car, truck, bus and van are one) sit at IoU 0.6 or more in normalised coordinates
   on 10 or more frames of 2 or more days on which the object stayed (the dwell rule, see
   [Static spots](#static-spots)). Only vehicles and other known inanimate COCO classes form spots:
   **people and animals never do**, so a person or an animal frame is never capped, and neither is a class
   renamed into a name of its own (a merged `vehicle`: the pre-flight says so once when the cap is on and no
   class of the run can form a spot). A frame whose every box of the run's classes (after the class
   mapping) lies at a spot shows nothing but parked objects; of those, at most `training.static_cap`
   (default 20, `0` = off, `--static-cap N` on `train prepare` / `train run`) per spot **and class of the
   box there** stay for training and validation together: a pickup labelled `truck` on 2 days at the place
   of a car labelled on 60 keeps its frames (one quota for the place lost both days about half the time).
   They are chosen for variety: round-robin over the quota's (scene, day) groups, the scenes alternating,
   so the first frame of every group goes before the second of any (seeded order of scenes, days and
   frames). A frame counting toward several quotas stays while one of them has room and then counts toward
   all of them, so a quota can end over the cap (its frames were kept for another); each keeps at least
   the cap or all its frames, whichever is fewer. Never capped: a frame with any other box of the run's
   classes (a person, a car passing; a box of a class the run does not train does not count, so a parked
   car beside an untrained label is capped), an empty frame or one whose boxes are all of classes the run
   does not train (a negative), held-out frames (decided before, so `heldout_fingerprint` is the same
   with the cap on and off), public frames, frames of an `unknown` camera. The negatives (step 4) and the
   validation split (step 5) follow on what is left, balancing (step 7) after. `dataset.json` counts
   `static_capped` and `static_spots`, lists `static_caps` (one entry per spot and class: `id` the spot,
   `camera`, `label` the class of the quota, `frames`: frames of the pool showing nothing but static
   objects with a box of that class at that spot, `kept`), largest first, and `static_capped_boxes` (boxes
   per class on the frames left out); the pre-flight and `train prepare` say *N frame(s) showing nothing
   but objects parked at M static spot(s) were left out (at most 20 kept per spot and class, across scenes
   and days); the largest: ...*, and a class with no or fewer than 50 training boxes says how many of its
   boxes the cap left out and how to turn it off. The plan's spots come from its own pool (the train and
   validation frames of the ticked exports after the scene filter), the Refine tab's Static objects card
   from every reviewed frame of the folder under review, so the two can differ (in spots and ids).
4. **Hard negatives**: about `training.negative_share` (15%) of the training frames have no objects:
   extra empty frames are dropped, those a reviewer confirmed empty or rejected a box on kept first
   (`frames.json`, `hard_negatives.json`, so the Benchmark format need not be ticked; a labelers
   run's dismissals and rejects for the class list do not count); too few give a warning.
5. **Validation**: 10% of your own training frames (of all training frames when there are none),
   but at least 50 frames when the pool has them (`Options.min_valid`; never more than a quarter of the
   pool, so 100 frames give 25, 241 give 50, 1000 give 100; always at least one), for the trainer's checkpoint choice.
   **Whole camera days** (`dataset/valid.rs`, always on): frames of one camera on one day are
   near-duplicates, so validation takes whole `camera/day` groups, never one that also trains.
   Groups go by seeded rank, those adding a class validation lacks first, until the size is reached:
   the first pass accepts a group while the total stays within 1.5x that size, the second, when
   short, within `max(size, a quarter of the frames)`, so a day of 74 frames can stand in for 50 and a
   small pool can give up to 37.5% of its frames. A group is skipped when validation would then hold
   more than half of a class's boxes in the pool, which also keeps a class's last training box in
   training (`Stats.valid_groups` in `dataset.json` and the `dataset:` line of `train.log` list the
   groups). Frames with neither a camera nor a day tag are no group and stay in training.
   **Fallback**: under 3 camera days left after the held-out split, groups reaching under half of the
   size, no group with a box, or no camera or day tag at all give a warning that says so and the
   frame-by-frame split (seeded rank, a frame holding a class's last training box stays in training);
   validation then shares camera days with training and reads optimistic. Pools of public frames
   only use the frame split without a warning. Seeded choices rank by the path relative to the export
   folder or by the `camera/day` name, so the same exports give the same split wherever they are, the
   training seed (`--seeds`) never changes it, and the held-out set and `heldout_fingerprint` do not
   depend on it.
   **Validation mAP now measures unseen camera days**: it reads lower than in runs of older versions
   and does not compare with them. Compare runs on the held-out benchmark.
   A class with no training box gets a warning that says why: its boxes are only in the held-out
   split (the export's `val`: hold out other cameras or days when exporting, or review frames with
   it from the camera days that train), or no frame has one.
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
   the same report is shown **before** a run: the Train tab's preview panel, or
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

### Pre-flight report

Above **Queue training** the page shows what the ticked data would give, for each run (the answer of
`POST /v1/training/balance-preview`; `train prepare --dry-run` prints the same, `src/training/preflight.rs`):
boxes per class (train, validation, held out, and by scene), training frames per scene, the splits,
frames confirmed empty or with a rejected box, frames left out (ignore regions, parked-spot cap, static
cap with its largest spots, frames repeated across exports, the scene filter), the share accepted in bulk, frames of the folder under review
not reviewed yet, and disk needed against free per volume.

| Check | Limit (`preflight.rs`) | Result |
|---|---|---|
| No training box of any picked class | none | refused |
| No held-out frame, `training.heldout_share` above 0 | none | refused: the run could not be measured |
| Disk | need + the free space a volume keeps (5%, 1 to 5 GiB, as an import) | refused, with the numbers |
| Few training boxes of a class | under 50 (`MIN_CLASS_BOXES`) | warning |
| Few held-out frames | under 10 (`MIN_HELDOUT_FRAMES`) | warning |
| Few training frames of a scene | under 50 (`MIN_SCENE_FRAMES`) | warning, e.g. "Only 12 night frames: review more, or train day only." |

The checks run when a run is queued (page, `train run`) and again when it starts. Warnings stay with the
run: on its row, in `dataset.json` and in `train.log`. The disk need is the frames that cannot be hard-linked
(another volume, or exFAT, probed with one link), labels, four times the base weights for the work folder
(measured: YOLO26n 16 MB, RT-DETRv2 S 77 MB, RF-DETR nano 1.2 GB), RF-DETR's augmented copies, the model, and
what the environment and weights still have to download.

Output in `<training dir>/<run id>/data/`: `images/`, `labels/` (hard links, else copies),
`classes.txt`, `train.txt` / `valid.txt` (repeats as repeated lines), `data.yaml` (Ultralytics),
`train.coco.json` / `valid.coco.json` (category ids 0..C-1, `file_name` relative to the folder; a
repeated frame is a separate image entry), `heldout/` (a benchmark `dir:` set: `images/`, YOLO
`labels/`, `classes.txt`, `manifest.json` with tags, `scored_labels` and every frame's `objects`) and
`dataset.json` (sources, licences, counts, warnings). `train prepare --out <dir>` writes the same without
training.

### Ignore regions

> **Changed default (this version): `training.ignore` is `background`.** Earlier versions left out of
> training and validation every frame that held an unsure or too-small object. Now such a frame trains
> with the object unlabelled when it also holds a label of the run's classes; only a frame of nothing but
> such objects is left out. Runs on the same data therefore differ from earlier ones: more training
> frames, other validation days (a validated day takes its frames with such objects out of both sides),
> and another number of negatives (their target follows the positive frames). To get the old behaviour
> back, set `training.ignore` to `drop-frame` (the Refine tab's *Too small* card, or `--ignore drop-frame`
> on `train prepare|run`). Each run records its mode (`progress.settings.ignore`), and the Comparison card
> warns when the runs it compares differ in it.

An unsure box (Review tab `d`), a box rejected as too small (the *Too small* button) or, with
`collect.size_floor` set, a training box under the size floor (`docs/DATASET.md`, Size floor) means "a
real object, do not train on it" (`docs/DATASET.md`, Ignore regions). None of the trainers here can leave
a box out of its loss: Ultralytics and transformers show no ignore support and RF-DETR's crowd handling
makes such a box background. So either the object trains as background or its frame goes. Measured on 851
reviewed frames with fold models that never saw the camera day (384 px input), objects under about one
patch are found 6% of the time (under 8 px; 37% at 8-12 px), so as background they teach little that is
wrong, while leaving the frame out cost every other label on it (with the floor at 10 px / 384 on that
data, `drop-frame` trained on 527 frames with 136 person boxes, `background` on 568 with all 164).
`training.ignore` (`--ignore` on `train prepare|run`) decides:

- **`background` (default):** a train frame with a region and at least one box of the run's classes trains
  with the region simply unlabelled (`ignore_kept`, the regions per class in `ignore_boxes_background`). A
  frame with nothing but regions is left out (`ignore_only`): an image whose only object is unlabelled
  would teach "nothing here" exactly where something is, so it is neither a positive nor a negative,
  confirmed empty or not. Validation never holds a frame with a region (the trainer's validation cannot
  ignore one): when its camera day is picked for validation it leaves both sides (`ignore_valid_day`; the
  day is validation's), and that day's loss counts toward the rule that validation takes at most half of a
  class's boxes. Positives that leave this way do not count for the negative target: the training negatives
  are trimmed to the positives that train or validate (`negatives_trimmed`, also in `negatives_dropped`;
  validation's negatives stay). The parked and static caps run before validation is chosen and are not run
  again, so a capped spot may keep one frame fewer than its cap. When every frame of your own has a region
  and the public subset is mixed in, validation is the public frames only, with a warning. Without a
  validation split the trainer validates on the training frames without regions.
- **`drop-frame`:** every train / valid frame with a region is left out, as earlier versions did; one
  unsure or tiny object takes its whole frame out of training. No frame with a region is in the pool, so
  nothing leaves with a validation day and no negative is trimmed.

`ignore_frames` in `dataset.json` counts the frames left out for regions (in `background` mode
`ignore_only + ignore_valid_day`), with a warning; `dataset.json` records the mode in `options.ignore`, and
the pre-flight report says it (`ignore_mode`, `ignore_kept`, `ignore_only`, `ignore_valid_day`). The
scene filter comes first, so only in-scope frames count. The held-out set and its fingerprint do not depend
on the mode.
- **Held out:** the frames stay. Their `manifest.json` entry has `objects` for every frame: the boxes read
  back from the frame's YOLO lines, so they score exactly as `labels/` did, and the regions with
  `ignore: true`, renamed like the boxes. A region of a label the run has no class for is left out, as
  predictions of that label are not scored. A detection of the region's label that overlaps it at the IoU
  being scored counts neither way (a detection of another label is still a false positive), the region is
  never a miss, and a frame of regions only is scored as empty. A frame with no objects stays an empty list,
  so it stays scored as empty. So the benchmark's "empty frames" (false alerts on frames without objects)
  include frames whose only objects are ignore regions: an alert on such a frame away from the region
  counts as a false alert, one on it does not count. That was so for unsure boxes before; a size floor adds
  frames of only tiny objects.

An export from an older version has no `ignore.json`, so its unsure objects stay unlabelled in training:
export again.

**The size floor at the run's input.** An export applies `collect.size_floor` at the floor's own `input`
(`ignore.json` holds the floor, and marks each region it made `size_floor`). A run trains at its recipe's
`imgsz` (`dataset::Options::imgsz`), and the plan compares the two:

- **Larger input** (a floor at 384, a run at 640): every size floor region of a train / valid frame is judged
  again at the run's input; one that is not under the floor's `px` there is given back as an ordinary box of
  its label, mapped through the run's classes and renames like any box (`floor_restored` per label in
  `dataset.json`, the pre-flight's "under the export's size floor but not at this run's input" line, and a
  warning). A frame whose regions all come back is a plain frame again (it may validate). Held-out frames
  keep their regions exactly as exported, so the held-out set and its fingerprint do not change, and a
  detection on such a region still counts neither way.
- **Smaller input** (a floor at 640, a run at 384): a box cannot be turned into a region by the plan, so it
  warns: "this run trains at 384 px, but the size floor of `<export>` was applied at 640: objects of 6.0 to
  10 px at this run's input are still labels (N on its training frames), though it cannot learn them;
  export again with the floor's input set to 384".
- **Unknown** (`train prepare` names no model, so `train prepare` and its `--dry-run` plan with no input
  size): nothing is given back, and a warning names the export's floor and the regions it made.

The plan also compares the exports' floors: exports made with different floors are named together ("the
exports were made with different size floors: ..."; a frame in two exports keeps the first one's copy), and
with `collect.size_floor` set, an export made without a floor ("before it was set, or by an older version")
or with another one is named with "export again". A floor that is off is not compared (`dataset::Options::
size_floor` None).

A size floor changes what an export marks as ignore regions, held-out frames included, and so the
held-out set's fingerprint: benchmark scores of runs trained before and after a floor change do not
compare (the Comparison card says so: their held-out fingerprints differ).

### Class inventory

`blue-onyx-prism train classes --dataset <export>... [--scene <s>] [--rename from=to] [--json]` lists
the classes of the datasets after the renames: boxes, frames holding one, and boxes per scene (day,
night, ir, unknown scene). With `--scene` an **in scope** column adds the boxes of those scenes
(unknown-scene boxes count only for `all`). The Train tab's class picker shows the same counts
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
| RF-DETR | 100 | 10 | 10 | 0.001 | 3 epochs | 1e-4 (rfdetr's default); 5e-5 below 1000 unique training frames |
| RT-DETRv2 / D-FINE | 80 | 10 | 10 | 0.001 | 3 epochs | 1e-4 |

With fewer than 50 validation images a **recommended** patience doubles (10 -> 20, at most the
epochs; `recipe::SMALL_VALID_IMAGES`): the metrics of so few images jump by several points between
epochs. A typed patience (form, `--patience`, `training.patience`) is never changed. The first lines
of the train stage in `train.log` say which applied, e.g. `# convergence: 16 validation images
(< 50): patience 20 instead of 10; mAP averaged over the last 3 epochs`. Whole camera days can
land under 50 in a pool that has more frames; the rule counts the frames validation really got.

**RF-DETR learning rate and schedule.** rfdetr 1.11 defaults to a *step* schedule that drops the
learning rate x0.1 only after epoch 100 (`lr_drop`); the convergence rule usually stops before, so the
rate never decayed (run `20261031-025744-2a83` stopped at epoch 93). The trainer therefore passes
`lr_scheduler="cosine"` with `lr_scheduler_kwargs={"min_factor": 0.1}` (`--lr-scheduler cosine|step`,
`--lr-min-factor`): rfdetr's managed cosine decays from the learning rate to 0.1x of it over the
planned epochs (its total steps are Lightning's `estimated_stepping_batches`), after its warm-up.
With no learning rate typed and fewer than 1000 unique training frames (`recipe::SMALL_TRAIN_IMAGES`;
balancing and public-mix repeats count once, so 423 frames repeated to 1,019 lines are 423) an
RF-DETR run uses lr 5e-5 and encoder (backbone) lr 7.5e-5 instead of rfdetr's 1e-4 / 1.5e-4; a typed
lr is used as is with an encoder lr of 1.5x it (`TrainRecipe::effective_lr`, `--lr` / `--lr-encoder`).
`train.log` says which applied, e.g. `# lr: 0.00005, encoder 0.000075 (312 unique training frames < 1000:
lower than RF-DETR's default 0.0001 for small datasets); schedule cosine (default), final factor 0.1
(default)`. An rfdetr that does not take these options trains without them (logged).

### Learning rate settings

The Train tab's Advanced section has a **Learning rate** field (empty = the family's default, the placeholder follows
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
| RF-DETR (rfdetr) | 1e-4, 5e-5 below 1000 unique training frames; encoder 1.5x | cosine, step (rfdetr's own x0.1 after epoch 100) | 0.1 (the cosine's end and the anneal's) |
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
and `lr_min_factor` in effect, `select_metric` (absent on older records: `map`), `epochs`, `patience`
in effect and `patience_doubled` (a small validation split), the requested `batch`, `workers`, `seed`
(a seed repeat's), `scene` (`all`, `night+ir`, ...) and `gpu` (its name). `select_fallback`
(`{metric, epoch, reason}`, absent normally) says the selected metric's source failed and the run went
on by mAP. The stage text ends with `lr 2.3e-4`.

**On the page** the metrics view shows a Learning rate tile (the current rate with its change from
the previous epoch) and an LR curve once the history has `lr`, a Convergence tile ("since best 4 / 10,
stops at epoch E", the anneal or extension when active; hidden with patience 0), the finish time
under Time left, a run settings line (GPU, batch and input size, learning rate and schedule, cap and
patience, "selected by F2" when the metric is not mAP, seed, scene, workers) and an overfitting hint
when the validation loss rises while the training loss falls.

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

- The value is the **selected metric** (see *Select best by* below; by default the validation
  **mAP50-95**, else mAP50 when the trainer reports no mAP50-95), **averaged over the last `smooth`
  epochs** (3; fewer at the start; `--smooth 1` = each epoch's value). A single noisy peak no longer starts the countdown: run `20261030-010233-b69d` (RF-DETR
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
  Train tab shows "Early stop reached at epoch 15 (best epoch 5, mAP 0.835; the selected metric's
  name when it is not mAP); training loss is still falling (3.569 -> 3.315). Continuing to epoch 25." with **Stop now and keep epoch 5** (the
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
- The **best checkpoint** (highest raw value of the selected metric, not the smoothed one; the
  earliest epoch on a tie) is exported, not the last: Ultralytics `best.pt`, RF-DETR
  `checkpoint_best_total.pth`, and for RT-DETRv2 / D-FINE (transformers) the best model state, which is
  restored in memory before the export (the log names the exported epoch).

### Select best by

The Train tab's Advanced **Select best by** dropdown (`training.select_metric`, `train run --select-metric`,
the script's `--select-metric`; none of them set = `map`) picks the metric a run optimizes. It decides
both which epoch's weights are exported and the convergence rule's value, so a run never stops on mAP
while selecting on something else. The one list with labels and hints is `recipe::SELECT_METRICS`.

| Id | Metric | What it does |
|---|---|---|
| `map` (default) | mAP 50-95 | Average precision over IoU 0.5-0.95, per class: rewards finding objects and tight boxes |
| `map50` | mAP 50 | Average precision at IoU 0.5: finding objects counts, exact box fit does not |
| `f1` | F1 | Weighs catching objects and false alarms equally |
| `f2` | F2 | Favours catching objects over false alarms |
| `f0_5` | F0.5 | Favours fewer false alarms over catching objects |
| `iou` | Mean IoU | Box tightness of the matched boxes only: **it ignores missed objects, so it can prefer a model that misses more** |

`f1`, `f2` and `f0_5` are macro averages over the classes at the one confidence threshold that
maximizes that average, so they are not person or vehicle specific: a rare class counts as much as
`person`. `iou` is pooled over the matched boxes of all classes, at no threshold. All are on a 0..1
scale, so `min_delta` stays 0.001.

**A small validation split makes any choice but mAP noisier**: F-beta and IoU take the best threshold
of few objects, and a single box moves them. Whatever a run selects by, **the held-out benchmark stays
the number to trust**; two runs that differ only in this setting are compared there (the Comparison
card shows "best 12 by F2" and warns when runs of a group were selected by different metrics).

How each trainer follows it (`Selection` and `pick_metric` in `train_model.py`; with `map` every trainer
keeps its own path, byte for byte):

- **Ultralytics**: `NightTrainer.validate` restores `best_fitness` after Ultralytics' `validate` and
  applies the same comparison to the selected value, computed there from `validator.metrics.box` and
  a peek at the IoU data (`on_fit_epoch_end`, which reports them, runs later). `best.pt` follows it,
  and `conv.step` gets the same number. `save_model` also writes on a tie, so a tie returns a fitness
  a hair below the best: the earlier epoch stays, as in the rule. Ultralytics' own stopper never fires
  (its patience is above the epochs), and a NaN mAP fitness stays Ultralytics' NaN recovery's.
- **RF-DETR**: `BestModelCallback` writes `checkpoint_best_total.pth` and knows only mAP / mAR. For its
  `on_validation_end` the script hands it a view of the trainer whose `callback_metrics` has the
  selected value (same dtype and device) under `monitor` and `_monitor_ema`. Writing into
  `trainer.callback_metrics` itself, as rfdetr's `smooth_alpha` does, has no effect: Lightning's
  property refreshes that dict from the logged results on every read until the validation loop resets
  them, and `val/ema_mAP_50_95` is logged. The real metrics are never changed, so nothing reaches
  `metrics.csv`, the epoch event or the `map` columns. With EMA (rfdetr's default) validation scores
  only the EMA model, the sweep and the IoU included, and the EMA track is the only one. rfdetr's
  high-water mark starts at 0.0 with a strict `>`, and it would backfill the final weights if nothing
  beat it; a non-default run starts it at -1, so a metric of exactly 0.0 at every epoch keeps epoch 1,
  as the rule and the page say. (A default run keeps rfdetr's behaviour: all-zero mAP exports the
  final EMA weights.)
- **RT-DETRv2 / D-FINE**: the rule's `improved` already keeps the weights, so only the value changes.

An F or the IoU missing at an epoch means nothing matched and counts as 0.0 (`pick_metric`, Rust twin
`progress::rule_values`). If the source itself failed (the F-beta sweep raised, a changed library
internal took the IoU hook, or rfdetr's callback cannot be wrapped) the script logs it once, prints
`@train {"event":"select_fallback","metric":"f2","epoch":E,"reason":"..."}` and selects by mAP for the
whole run; it never fails a run for this. Values on another scale cannot be compared, so the rule and
the trainer's best start again at epoch E, and the page's best epoch and settings line follow.
`progress.best`, the Best tile, the "best epoch" text and the Stop now / Continue notices follow the
selected metric. Records from before the setting read as `map`.

**Finish now.** A run in its train stage can be stopped after the current epoch, keeping the best
checkpoint (it then exports, installs and benchmarks as usual): the **Finish now** button beside Cancel
on the Train tab (loopback / admin, like Cancel; a paused run is resumed first), or create the file
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

### Retry export

One process trains and exports, so a failure late in a run used to take the checkpoints with it. Now
`<run>/work` goes only once the model is installed, or when you cancel. A failed run keeps it, and when it
holds a best checkpoint the run's row (or `train retry-export <run>`) offers **Retry export**: a new process
with `train_model.py --export-only <checkpoint>`, on the CPU, then install, config and benchmark as usual,
under the run's own name. The checkpoints go when that succeeds, or with **Remove and delete files**; the
row states their size. Cancelling a retry returns the run to its failure.

| Family | Checkpoint | From a run that died while training |
|---|---|---|
| YOLO26, YOLO11, YOLOv8 | `run/weights/best.pt`, else `last.pt` | yes, the best epoch so far |
| RF-DETR | `rfdetr-out/checkpoint_best_total.pth`, else the running `checkpoint_best_ema.pth` / `_regular.pth` | yes, the best epoch so far |
| RT-DETRv2, D-FINE | `hf-best/` (saved at every new best epoch; `save_strategy="no"` keeps nothing else) | yes, the best epoch so far; not for runs of older versions |

Measured on an M1 with a 16-frame set, 2 epochs: YOLO26n, RF-DETR nano and RT-DETRv2 S each exported after
a failed export and after `kill -9` during training (RF-DETR from `checkpoint_best_ema.pth`), in 3 to 7 s. D-FINE
has not been run: it trains on CUDA only. Training itself does not resume: the convergence rule, the
selection and the learning-rate bends live in the trainer's process.

A run out of memory restarts only while that costs little: not once training ended, and not after 3
finished epochs when a checkpoint exists. It then fails with the checkpoint kept, and the row offers the
best epoch so far.

Stages (page and `train run` show them with percent and ETA):

1. **setup**: uv, Python, the lock's packages, on Windows the Visual C++ runtime check (the export's
   code).
2. **self-check**: `train_model.py --self-check` imports the trainer and reports what PyTorch sees; a
   CUDA run also launches a test kernel, so a driver or kernel problem fails here in seconds.
3. **weights**: the base weights (and the public subset), SHA-256 checked.
4. **dataset**: assembled as above.
5. **train**: `train_model.py` trains, then exports and checks the ONNX; a failure at any later point
   keeps `<run>/work`, see [Retry export](#retry-export). Progress: every event is one
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
   restarts the training with half the workers and the same batch (at most 3 restarts; after 3
   finished epochs with a checkpoint on disk, or in the export, it fails instead and keeps the
   checkpoint, [Retry export](#retry-export)); the log says "out of system memory with 5 data
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
   starting, twice; the cause was never found). The Benchmark page keeps away from it too: it
   starts with the devices the models run on, asks before a run of several GPUs, refuses while a
   queue job runs, and leaves `<data root>/benchmark-breadcrumb.json` naming the device it was
   about to start, so a hang leaves the last device on record. The `benchmark` CLI does the same:
   a sweep (`--all-devices`, implied by `--apply`, `--apply-threshold`, `--threshold-search` and
   `--report`) over two or more GPUs is refused before any model loads unless `--confirm-gpus` is
   given (or `benchmark.devices` lists fewer), and it leaves the same breadcrumb, reported by the
   next run or service start. A failed benchmark is reported but keeps the model.

Default name: `<scene>-<model>-<YYYYMMDD-HHMM>`: `custom-` for `all`, `day-`, `night-` for `night+ir`,
`night-only-`, `ir-`, `day-night-`, `day-ir-`. Several models can be queued at once (one per ticked
size, or `--model` repeated), and several scenes queue one run per scene per model
([Comparing](#comparing-day-nightir-and-mixed-models)); they run one at a time. Cancel
kills the process tree. Run records are kept in `<training dir>/runs.json` (state changes are saved at once, other progress at most every 5 minutes; `tasks.json` keeps the newest 200 finished tasks, runs are never dropped; a damaged one moves to `runs.json.corrupt-<time>` and is logged, and a file that cannot be read is left alone and not saved over; `tasks.json` likewise; the working folder goes only after the model is registered); a run interrupted by a
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

**Training folder** (`training.dir`, default `<data root>/training`; This machine card of the Train tab, under *Folders and training environment*, with a
Folder... button): run records (`runs.json`, `tasks.json`), prepared datasets, held-out sets, logs,
import work and the Dataset tab's exports. A new folder must exist and pass the same checks as other
folder settings (no system folder, browsable or natively picked). It applies at once, without a
restart: the queue switches to the new folder's records when no run, import or auto-label job is
queued or running, and the page refuses the change (409) while one is. Runs, imports and exports in the
old folder stay there and are not moved (set the old folder again to see them).

The dataset folders a run uses are checked too: a ticked folder is used as is only when the page
offered it (`training.datasets` or an export under `<training dir>/exports`); any other value, the typed
*Another folder* and its *Save folder* must be a browsable folder or one picked in the native dialog.

**Use as** (Trained models card): *Use as default* first shows what changes (`GET
/v1/training/switch-preview`, `training/switch.rs`): thresholds, classes (a class reported today and
not after the switch is called out, since Blue Iris may filter on labels), night model, filters,
verifier, static suppression, device. *Copy settings from <current>* (default on) copies the per-model
behaviour settings onto the new model, narrowed to its classes; classes, device and device profiles
stay. The held-out benchmark's threshold is offered in words (`suggest`, `ThresholdAdvice`, saved as the
device's confidence profile) and is part of the switch. The switch records `previous_default`, and
**Revert** (this card and Home, `POST /training/revert`) restores `default_model` and the fields the
switch changed; it writes the opposite record, so a second Revert redoes the switch. Revert is refused
with the config unchanged when the old model is gone (the page offers to forget the record), disabled, or
the default was changed by hand since. *night model of* sets the chosen model's `night_model` (IR and
dark frames go to the trained model); *verifier of* sets its `verify` model and refuses to replace a
different verifier unless the confirm names it (`replace_verifier`). All go through the config page's
save path and apply at once. A camera that names a custom model in Blue Iris keeps using that model.

After a run's benchmark the current default runs on the same sets and devices (`Hooks::baseline`,
Train tab only; `train run` and `train retry-export` print one line saying so), so the Models tab's Use card can say "On N held-out frames: new finds ...; current finds ...".
Its results stay in `<run>/baseline/`, removed afterwards: they never touch `benchmark.json`, the
history or the `auto` order. It is skipped, with the reason in the row, when the default is the new
model, is not loaded, or reports none of its classes.

`trained_on_share` on a benchmark result (`benchmark/trained_on.rs`) is the share of its frames the
model trained on (frame identity against the run's `train.txt`; None without a training record). Older
results get it when served. Profiles, Compare and the run row show a chip when it is above 0.

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
| all (the script) | `f1`, `f1_thr`, `f2`, `f2_thr`, `f2_precision`, `f2_recall`, `f0_5`, `f0_5_thr`, ...: macro F-beta (`(1+b^2)PR/(b^2P+R)` per class, averaged over the classes with ground truth) at the best confidence of the sweep (rfdetr's own 101-threshold rows, Ultralytics' `p_curve` / `r_curve`); `iou` (mean IoU of the matched boxes, below); `mem_gb` (peak GPU memory: CUDA `max_memory_reserved`, MPS the driver's allocation sampled after every batch); `img_s` (training images per second, validation excluded) |

F2 weighs recall four times precision ("don't miss a person"), F0.5 the other way; the `*_thr` columns
are a starting confidence threshold for the trained model. NaN, infinite and non-numeric values are
dropped; at most 64 metrics per epoch (the shortest keys first).

`iou` is the same measurement for every family, taken at each validation on the validator's own
predictions (all scores, so no confidence threshold): COCO-style greedy matching, highest score first,
per class, each ground truth matched once, IoU >= 0.5, then the mean IoU of the matched pairs (pooled
over classes). It measures how tight the boxes are, so it starts at 0.5 and missed objects do not lower
it: read it alongside recall. YOLO `box_loss` and RF-DETR `loss_giou` are loss terms on other scales and
not comparable with it. By default it is display only; `--select-metric iou` makes it the selected metric
(*Select best by*).

The service maps the keys to stable ids (`src/training/progress.rs` `canonical`): `map`, `map50`,
`map75`, `mar`, `iou`, `precision`, `recall`, `f1`, `f1_thr`, `f2`, `f2_thr`, `f2_precision`, `f2_recall`,
`f0_5`, `f0_5_thr`, `loss`, `val_loss`, `lr`, `mem_gb`, `img_s`, and `ema_map` ... for rfdetr's
`val/ema_*`; unknown keys keep their raw name (`val/AP/raccoon`, `train/loss_ce`), so nothing is lost
when a library renames one. Each run's `progress` in `GET /v1/training` has
`history` (kept in `<run dir>/history.jsonl`, one line per new or changed epoch, so `runs.json` holds only `best` and the latest values; an older `runs.json` that still holds history is moved into the files at start; one point per epoch: `epoch`, `elapsed_s`, `metrics`; an `eval` event fills its epoch's
point; at most 1000 points) and `best` (`{metric, epoch, value}` on the selected metric, by default
`map`, else `map50`, else the lowest `val_loss`), which the stage text shows as "best mAP50-95 0.711 @ epoch 3". `GET /v1/training`
also has `metrics` (label, short label, `higher_better`, `format` of every id) and `metric_columns`.
Records written before this keep loading (no `history`: the page shows no metrics block).

**The metrics view** (Jobs card and Models tab, every run with a history, running or finished): KPI tiles
(epoch, ETA, best primary metric with its epoch, latest loss, validation loss and mean IoU with the
change from the previous epoch), curves of the losses and of the quality metrics with the best epoch marked,
and a per-epoch table, newest first. **Columns** picks the table's columns from every metric in the
run's history (known ids by label, raw keys by name) and their order; the choice is saved as
`training.metric_columns` (`POST /v1/training/metric-columns` with `{"columns": [...]}`: at most 20
distinct names of 1-64 characters, known ids or raw names; answers the saved list, 400 otherwise) and
applies at once. Empty (the default) = `map, map50, iou, f1, f2, precision, recall, loss`. The raw trainer
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

**The table.** The Comparison card of the Models tab (one per group) and, at the end of a
`train run` with several scenes, the CLI table have one row per run: model, scene filter, best epoch
(with "by F2" when not selected by mAP; a warning when runs of a group differ) and the epoch it
stopped at, held-out AP50-95 for `day`, `night`, `ir` and `all` frames
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

`blue-onyx-prism train import` and the **Import** tab put frames from other
sources into the dataset folder (`collect.dir`), in the collector's layout, so they show on the
Collect tab (*Imported footage*) and on the Review tab (they are queued for review) like collected
frames.

`--out-dir <dir>` (the page's *Save frames to*, with a native folder picker) writes them to another
folder instead, in the same layout with its own `imports.jsonl` and review queue. The Collect tab
shows only `collect.dir`; to review the frames, choose the folder on the Review tab (*Folder under
review*, `collect.review_dir`). The page accepts a browsable folder or one picked in the native
dialog, like the source path.

### Sources

| Source | How | Licence |
|---|---|---|
| A video (`.mp4`, `.mkv`, `.avi`, `.mov`), an image, or a folder of them | `--path`; the page's *Browse...* lists folders and videos (`GET /v1/fs/browse?kind=video`). Folders are walked 6 levels deep, up to 20 000 files. | You confirm it (`--confirm-license`; default text "own footage") |
| Blue Iris clip and alert folders | the same `--path`: `Cam.20240315_083012.mp4` clips and `Cam.20240315_083012.123456.jpg` alert snapshots. The camera folder and the recording time come from the file name. `.bvr` clips cannot be read (Blue Iris's own format): they are counted and the report says to export them to MP4 in Blue Iris. | You confirm it |
| A video URL | `--url`, downloaded with yt-dlp (pinned in the environment): loopback, private and other internal addresses are refused, by name and after resolving it (a redirect to one is not checked, because yt-dlp follows redirects itself); no playlists, no live streams, a video-only format up to 1080 p (so no ffmpeg binary is needed to merge streams), at most 2 GB. What the site reports (title, uploader, page URL, licence) goes into the provenance next to your confirmation; a different site licence is a warning. | `--license` (required) and `--confirm-license` |
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
used, the warnings and one `summary` line. The Import tab shows that line under the path field
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
ones until they are reviewed (a reviewed frame or one with a human decision is never pruned; a label
file from auto-labelling alone does not protect a frame, see Disk in `docs/DATASET.md`).

**ffmpeg**: not a separate tool. The PyAV wheels of the environment's lock bundle an FFmpeg build that
reports LGPL-3.0-or-later (`av._core.library_meta`; it includes the GPL x264/x265 *encoders*, which the
import never uses), the filter graph runs in-process, and the wheels are pinned by the lock's SHA-256
like every other package. A `tool:ffmpeg` resource would have needed per-OS static builds (BtbN's
Windows / Linux builds, a third party for macOS), their hashes and an extractor for each; PyAV was
already in the lock (a dependency of the trainers) and decodes every format the import reads.

### Auto-label after import

The Import tab's **Auto-label the imported frames when the import finishes** box queues an
[auto-label](#auto-label) run right behind the import, with the settings of your last Auto-label tab run, or the defaults, and the classes picked on the Import tab (labelers or model),
classes and run options (the page posts them as `al.<field>`); the card's *Which frames* fields are
ignored: the run takes this import's frames (`import=<id>, status=any, source=all`, limit = Frames to
keep) in the folder the import writes. Both are planned before anything is queued, so a bad auto-label
setting (no labeler, no model chosen, a class outside the list, a GPU it cannot use) refuses the import
too ("Not imported: auto-label: ..."). Follow-up options, after the run, in this order and scoped to the
import's frames:

- *Then reject auto boxes scoring below* (0.3): the pending auto boxes under that score on the unreviewed
  frames become `rejected` with `reason: low_score` (hard negatives; a reviewer can still change them);
- *Then accept frames whose boxes are all auto: accept* (off by default): those frames are saved as
  reviewed, like the Review tab's bulk accept; after the reject, a frame left with only `accept` boxes
  qualifies;
- *Show this folder on the Review tab when done* (needs *Save frames to*; also without auto-label): sets
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

`blue-onyx-prism train autolabel` and the **Auto-label** tab (from the Review tab, *Auto-label these frames* passes the frames
its filters select) propose boxes for dataset frames. They land in the frames' label files as
**pending `source: auto` boxes** through `labels::merge_proposals` (the service writes every label file;
the script only returns proposals as JSONL), so the reviewer decides. Nothing is accepted, rejected or
deleted automatically; confirmed empty frames (hard negatives) come only from the review.

**Frames**: `--frames` takes the Review queue's filters as `key=value` pairs: `bucket`, `scene`, `camera`, `label`,
`status` (default `unreviewed`), `source` (default `all`), `auto` (`review`, `accept`, `nothing`, `none`;
`label=person|car` = frames with any of these classes; the queue's `label=` takes a comma list;
docs/DATASET.md "Queue"), `import=<id>|any`, `again=1`, `limit` (default 200). They are taken in the
queue's order (most uncertain first). The Review tab's button sends its own filters.

**Marker**: every frame the labelers return (boxes or not) gets `auto_run: {at, labeler, classes}` in its label
file (a frame without one gets its pre-fill plus the marker). Only frames whose boxes changed are queued
for review. A later run with the same labelers (`labeler`, e.g. `mm-gdino-tiny + rfdetr-large +
qwen3-vl-2b + siglip2-base`) skips those frames (`skipped_done` in the report) unless the filter has
`again=1` (Review tab: **Label again**); when every matching frame was skipped the run fails saying so.
A run with a model writes the marker too, with `labeler: "model:<name>@<device>"` (the device it asked
for), so the same model on the same device skips the frames it looked at unless `again=1`, and `auto=`
filters see them. Writes take the label write lock,
so a Review save with a stale `revision` gets a 409 instead of overwriting merged boxes.

**Run options** (Auto-label tab, Advanced):

| Option | Default | CLI | Script flag |
|---|---|---|---|
| Batch the VLM check: up to 8 of a frame's boxes per Qwen3-VL call | on | `--no-vlm-batch` | `--vlm-batch 8` (off: `1`) |
| Skip the VLM check below a fused score (0 < x <= 1); 2% of those boxes are asked anyway, to audit the cutoff | on, 0.15 | `--vlm-min-score X`, `--vlm-all-boxes` | `--vlm-min-score X --vlm-sample 0.02` |
| Load the next frame on a background thread while the GPU works | on | `--no-prefetch` | `--prefetch` |
| Keep finished frames if cancelled (both modes) | on | `--discard-on-cancel` | - |
| GPU of the labelers (shown with more than one usable NVIDIA GPU) | automatic (most VRAM) | `--gpu cuda:N` | `--device cuda:N` |

The VLM flags are left out when the VLM is off. Measured per frame on an M1 (MPS, median, one box per
call): Qwen3-VL check 7.6 s (about 1 s per box, ~8 boxes), MM-Grounding-DINO 1.9 s, RF-DETR 0.2 s,
SigLIP 0.07 s; 89% of the checked boxes had a fused score below 0.2, which is what the minimum score
skips (such a box is triaged without a VLM answer).

### Measuring auto-label

`scripts/research/eval_autolabel.py` scores the labelers' proposals against known boxes. Use it before
changing a threshold, a prompt or the question the VLM is asked; it changes no label file.

Measured on MM-Grounding-DINO tiny + RF-DETR L + Qwen3-VL 2B, the default 14 classes. A proposal is real
when it has the label of a known box at IoU >= 0.5. *Public* is 588 frames of `coco-cctv`, `bmd45-cctv`,
`exdark-night` and `nir-night`. *One camera* is 188 reviewed frames of a fixed overhead camera, nearly all
people, so a figure there says nothing about vehicles or animals.

| Proposals | Public: per frame, real | One camera: per frame, real |
|---|---|---|
| All | 7.24, 39% | 7.20, 13% |
| Score >= 0.25, VLM yes | 2.74, 77% | 0.92, 96% |
| The rest at score >= 0.15 | 2.59, 24% | 2.35, 1.8% |
| Score < 0.15 | 1.91, 5.3% | 3.93, 0% |

- **A VLM no ranks a box; it does not disprove one.** On public sets 40% of the boxes at score >= 0.25 it
  said no to were real. Small, distant and dark objects are where it fails.
- **One camera does not generalize.** Score >= 0.25 with a VLM yes hid 3% of the real objects on the fixed
  camera, 25% on the public sets and 45% of the people on `exdark-night`.
- **Score < 0.15** holds 3.6% of the real objects the labelers found on public sets, per class: `truck` 10.7%,
  `cat` 5.1%, `bicycle` 5.0%, `motorcycle` 4.9%, `person` 3.1%, `car` 2.9%, `bird` 2.4%, `dog` 0%.
- **The question matters.** The earlier "What is in this image: a person, a vehicle, ..., a plant, or
  nothing?" called 75 of 171 real people `plant` (the same at batch size 1). "Is there a real person in this
  image? Answer yes or no." said yes to 169 of 173 real objects and 50 of 1,352 false boxes on the same crops.

### Likely, unlikely and hidden boxes

A labelers run uses those figures to sort the new boxes of frames nobody has reviewed. The constants are
in `training/autolabel/triage.rs`; change them only after measuring again.

| Tier | Rule | What the page does |
|---|---|---|
| Hidden | Pending auto box, score < 0.15, not a `truck`, not found by SAM 3 | Rejected with `reason: "auto_dismissed"`: not drawn or listed until you ask |
| Likely | Pending auto box, score >= 0.25, the VLM said yes to the yes/no question | Marked with a star, first in the list, and its frame first in the queue. `c` or *Accept likely (N)* accepts them all |
| Unlikely | Everything else | Drawn quieter, listed after the likely boxes. `Shift+R` clears them |

- **Only the score hides a box.** A VLM no never does, because 40% of those boxes were real on public sets.
  With the VLM off nothing is likely, and the score still hides boxes.
- **`truck` is never hidden.** 10.7% of the real trucks scored below 0.15.
- **SAM 3 never hides a box.** A box it found stays, and so does anything else on a SAM 3 run, because
  these thresholds were measured with MM-Grounding-DINO. A run with a model and `--missing-only` hide nothing.
- **Classes only one labeler can report are safe.** The fusion does not halve them, so their score is at
  least that labeler's own threshold: 0.2 for MM-Grounding-DINO. A test holds a raccoon at 0.2 to this.
- **Hidden boxes are recoverable.** They stay in the label file. *Show N hidden* or `i` shows a frame's
  boxes dimmed, `auto=dismissed` finds the frames, and accepting one makes it a label. They are no hard
  negatives and teach no spot memory, since nobody judged them. Before you save, *Anything missing?* says
  how many are hidden. No bulk save takes a frame with hidden boxes: a saved frame claims it is complete,
  so a real object nobody looked at would train as background.
- **The filter is checked.** About 2% of the boxes that would be hidden stay visible, marked *audit sample:
  normally hidden*. The same box gets the same answer on a rerun. Your verdicts show on the Review tab
  as *Hidden-box check: 1 of 40 was real*, with a percentage from 20 boxes on.
- **A rerun can bring a box back.** A later proposal of the same label on a hidden box takes its place and is
  sorted again: the same score hides it again, a higher one or SAM 3's shows it. A run with a model shows it too.
- **Only new boxes are sorted.** Boxes of older runs stay as they are, an old seven-way VLM answer is
  never likely, and reviewed, sent-back, skipped and confirmed-empty frames follow the rules under
  [Rerunning labeling](#rerunning-labeling-on-part-of-the-dataset) with no tiers.

The run's summary and report give the counts: `likely`, `dismissed` (left out of `added`) and `audit`.

### Spot memory of fixed cameras

On one fixed camera, 87% of the false boxes sat in 33 fixed spots. In a time-split test on that camera,
spots learned from the first 53 frames hid 657 of the 1,008 later false boxes and 0 of the 111 real ones.
Real parked objects there were also being rejected frame after frame, which teaches the model that a
real car is background. One camera is all the evidence so far, so treat these numbers as a first
result. The run does two things with what you decided, in Rust, after the tiers above, on boxes of
frames nobody has reviewed. Serving does not change: alerts are as before.

**Rejected spots.** A place is a rejected spot when you rejected a box there on at least 2 frames: same
camera and frame size, IoU >= 0.6, the same object class (vehicle classes count as one). Spot edits, bulk
rejects, and duplicates and too-small rejects do not count, as in the repeats history. A new box at such
a spot is hidden with `reason: "spot_dismissed"`, which every reader treats like `auto_dismissed`, and
the page says *hidden: you rejected this spot N times*.

- Only rejections of the 30 days up to the frame's date count (`spots::REJECTED_WINDOW_DAYS`). One dated
  after the frame does not.
- A place where a box of that object was ever accepted is no spot, however often you rejected it there.
- A box with a fused score of 0.5 or more stays visible, and so does a likely box (0.25 or more with a
  VLM yes): a real object at a bad spot on an unlucky frame must not disappear.
- `truck` may be hidden here: that exception is for the score rule, and a spot rests on your own look.
- A SAM 3 run and a run with a model use no spots, and a box SAM 3 found is never hidden.
- The same 1 in 50 stays visible, marked `audit`. Its verdict counts in the *Hidden-box check* with the
  score filter's samples.
- A box below 0.15 is hidden by the score first, so it reads as `auto_dismissed`.

**Parked spots.** *Real, parked* (`t` on the decision card) accepts the box and adds a spot to
`<review folder>/spots/<camera>.json` ([docs/DATASET.md](DATASET.md#spots)). This store exists because the
repeats history is capped and forgets. Later a pending auto box of the exact same label at IoU >= 0.8 is
accepted with `reason: "parked_spot"` and keeps its evidence; its frame stays unreviewed. If the VLM said
no to the yes/no question, the box stays flagged instead, so a car that left is not labelled. A box the
VLM was not asked about is accepted. *Accept repeat-only frames* follows the same rule: it does not save
a frame with an accepted box the VLM said no to. A SAM 3 run accepts nothing here, because a bulk accept
reaches the exports. An existing box from an older run is taken too, but an audit sample and a box with
another class suggested are not. A model box from the pre-fill stays pending, even when the labelers
confirm it.

**The camera moved.** *Camera moved: forget spots* in Panel options retires every parked spot and sets
`moved_at`, so that rejections and accepts from frames before that moment stop counting. On the day of the
move the frame's time decides; a frame name with no time counts as before the move. A run reads the spots
once, at its start, so a run already going keeps what it read. It also learns nothing from its own writes.

The run's summary and report give `spot_dismissed` (left out of `added`) and `parked_accepted`.

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
`model_label` (the Review tab shows "was motorbike"); the report lists the renames (`renamed`) and
counts boxes per mapped class (`by_label`). The marker records the run's classes (`auto_run.classes`; files written before
runs could pick classes have none, which counts as the whole list), and the skip rule above becomes:
a frame is skipped only when the same labelers looked at it for **every** class of this run. A run
for `raccoon` after one for `person, car` labels the frames again; a run for `car` after the full list
skips them. `auto=nothing` still means "auto-labelled, no auto evidence"; the Review tab reads
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
Review tab's *don't want*) are classes you do not review: hidden from the Review tab, left out of the
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
4. **Checks**: Qwen3-VL answers "Is there a real <label> in this image? Answer yes or no." for the
   crop (20% padding, at most 448 px) of every box at or above `--vlm-min-score` (default 0.15), plus
   2% of the boxes below it for the audit (`vlm`, with `expected: "yes"`). A reply that is neither yes
   nor no leaves the box unchecked, never a no. Older label files hold the earlier seven-way answer
   (`plant`, expected `person`), which still loads and shows. SigLIP 2 ranks the animal classes on the crop
   (prompt ensembles; at night "an infrared night-vision photo of a ...") and stores the probabilities
   (`species_scores`).
5. **Triage** (the service, `training::autolabel::triage`): `accept` only when both labelers agree, the
   fused score is at least 0.6, the VLM did not object and SigLIP ranks the box's own species first;
   a person or an animal on a night or IR frame is always `review`. Every box gets `triage` and the
   reasons (`why`). A proposal that matches an existing box of the same label (IoU >= 0.5) is dropped
   by the merge rule, but its evidence (VLM answer, species) is put on that box: a model box the VLM
   calls a rock shows it on the Review tab, and a suppressed / below-threshold / verifier-found box
   triaged `accept` is accepted on save instead of rejected. A proposal on a *pending* box of another
   label (and no box of its own label) becomes a **class suggestion** on that box (`auto_label`,
   `auto_score`, triage `review`; `class_suggestions` in the report) instead of being lost. New boxes on
   unreviewed frames are then sorted into [likely, unlikely and hidden](#likely-unlikely-and-hidden-boxes).
6. **Bulk accept**: frames whose triage alone settles them (`auto=accept`: boxes `accept`, nothing else
   pending but unconfirmed model hints, no hidden box) can be saved as reviewed in one go from the Review tab
   (`POST /v1/review/accept-auto`, `bulk_accepted: true` in the file); everything else waits for a
   reviewer.

On Apple silicon everything runs on MPS (float16 for Qwen3-VL); `PYTORCH_ENABLE_MPS_FALLBACK=1` covers
missing ops.

### SAM 3 as the open-vocabulary labeler

SAM 3 (`facebook/sam3`) can take MM-Grounding-DINO's seat (step 1): `train autolabel --sam3`, or *SAM 3*
in the open-vocabulary select under Advanced on the Auto-label tab. It is **opt-in**: the default stays MM-Grounding-DINO,
and the full comparison of the two has not been run, so nothing here says SAM 3 is better. Steps 2 to 5
are unchanged, with one exception: **a SAM 3 box is never triaged `accept`** (the 0.6 rule was tuned for
MM-Grounding-DINO + RF-DETR), so *Accept auto-labeled frames* never takes a frame on SAM 3's word. It
cannot be combined with *Label with a model*.

**Threshold.** `--sam3-threshold` (needs `--sam3`), or the *SAM 3 threshold* field on the Training and
Review cards, shown while SAM 3 is the chosen labeler (form field `sam3_threshold`; an import's chained
auto-label takes it too). It is SAM 3's own score before fusion, from 0.01 to 1, default 0.3: a higher
value means fewer boxes and missed faint objects. A box only one labeler found scores half that after
fusion. The value appears in the run's summary. The frames' marker names the labelers, not the threshold,
so a rerun at a new value skips the frames already looked at unless the filter has `again=1` (*Label
again*); Preview says so. A rerun adds boxes and removes none, so a higher value does not clear the boxes
of an earlier, lower one.

**Cost.** About 5 s per frame on a GTX 1070 Ti with 22 prompt phrases (about 2 s plus 0.2 s per phrase),
against 0.55 s for MM-Grounding-DINO tiny. It was not measured on Apple GPUs. *Preview* (below) gives a
rough time before you start.

**Licence.** The SAM License allows commercial use and modification; redistribution must carry the licence,
and it bars military and ITAR uses. The app does not ship the weights: it downloads them (3.4 GB, pinned
by commit and SHA-256) after you accept the licence on Hugging Face.

**Getting the weights.** The repository is gated, so the download needs a Hugging Face account that has
been approved:

1. Open <https://huggingface.co/facebook/sam3> (the Auto-label tab links it, under Advanced) and request access. Approval
   can take a while.
2. On the Auto-label tab, under Advanced, click **I have access, download**. This is also your consent to the 3.4 GB. If
   the files are already in the weights folder (a copy you made by hand), they are checked by hash and
   used with no login and no network.
3. If the page asks for a login, run the command it shows, from the account the service runs as (below).
   It asks for a Hugging Face token without echoing it and checks it; a **read-only** token is enough
   (make one at <https://huggingface.co/settings/tokens>). A fine-grained token also needs read access to
   public gated repos. Click **Try again**.

What a refusal means:

| The page says | Cause | What to do |
|---|---|---|
| needs login | No token found, or Hugging Face refused it (expired, revoked, mistyped) | Run the shown command with a new token |
| token unreadable | A token file exists but this process cannot read it (often the service account) | Run the command as the service's account |
| needs access | The token is valid but the account is not approved for `facebook/sam3` yet, or a fine-grained token lacks the gated-repos permission | Request access, wait for approval, then Try again |

A refusal is final for that attempt: the app does not retry a 401 or 403 from Hugging Face. The token is
never shown, logged or passed to Python. It is looked up as `HF_TOKEN`, then the token saved by
`hf-login` (`<data root>/tools/hf/token`), then the `hf` CLI's own file; the page says which one is in
use and warns when `HF_TOKEN` hides a saved login. `blue-onyx-prism hf-logout` deletes the saved token.

**Service accounts.** `hf auth login` alone does not reach a service: the `hf` tool exists only after the
training environment is built, and a service account cannot read your profile's login file. Run the
app's own `hf-login` as the account that runs the service, with the service's config: (on Windows it gives
`tools\hf` a protected ACL: Administrators, SYSTEM, you, and the service account read-only; a failure is a warning):

- Windows, MSI install (service runs as LocalService): an elevated PowerShell (Run as administrator) and
  the command the page shows, which names the installed `blue-onyx-prism.exe` by full path and passes
  `--config C:\ProgramData\BlueOnyxPrism\blue_onyx_prism_config_service.json` (the page prints the exact
  path) before `hf-login`.
- Linux and macOS: `sudo -u <service account> /path/to/blue-onyx-prism --config <config> hf-login`.
- Docker: `docker exec -it <container> blue-onyx-prism hf-login`, or start the container with
  `HF_TOKEN` set.

**Remove** (the panel's button) deletes the whole SAM 3 weights folder and is refused while a queued or
running auto-label uses it.

### Rerunning labeling on part of the dataset

The Auto-label tab's **Which frames** select (under Advanced) picks the frames: *Unreviewed*, *Reviewed*, *Reviewed with
overlaps or duplicates*, *Unreviewed with overlaps or duplicates*, *All*, or *Custom* (the existing
fields). On the command line the same is `--frames "status=reviewed,auto=overlaps"` (`auto=overlaps`
selects frames with overlapping boxes you have not yet checked; frames whose overlaps a reviewer already
checked stay out). A frame the same labelers already looked at is skipped unless the filter has `again=1`
(Review tab: *Label again*).

**Found by two.** `auto=both` (Review tab's Auto filter and the card's Custom *Auto-labelled* select:
*found by 2+ labelers*) selects frames with a pending box that two or more labelers found, most such boxes
first, so the likely real and still unsettled boxes come up early. Only pending boxes count, so a reviewed
frame has none, and boxes of unwanted classes do not count on the page or in a run.

**Preview** (the button on the card; `train autolabel ... --dry-run` on the command line; nothing is
run or written, and no GPU, download or login is needed) shows how many frames match, how many a run
takes after `limit`, how many are skipped because the labelers already ran, the split by status, and a
rough time (SAM 3: 2 s + 0.2 s per phrase per frame; MM-Grounding-DINO 0.55 s; both before the VLM
check, on a GTX 1070 Ti). It warns when reviewed frames are in scope, adds with the VLM off that a box
only one labeler found cannot reopen them, and warns when `limit` is below the match count (the preset
never changes `limit` for you).

**What happens to reviewed frames.** A rerun never changes a box you decided: accepted and rejected
boxes stay as they are. On a reviewed frame the run adds a proposal only when it is a confident new
object: the labelers agree, or one of them scored it 0.5 or more and Qwen3-VL said yes to it; no check
(Qwen3-VL, SigLIP 2) objected; and it does not sit on a box you decided as another class or on one you
rejected. A check that did not run counts for nothing: a box only one labeler found cannot reopen a frame
at any score when the VLM is off, when *Skip boxes scoring below* skipped it, or when its reply was unclear.
The report counts these as
`filtered_unchecked`, which also takes in such boxes that only repeat one already on the frame. In a real
run with both checks off, 22 of 41 reviewed frames reopened, mostly on false boxes; with them on, 0 of 16.
A frame that gains a box goes back to **unreviewed** with a note ("re-label by sam3: ...") and **leaves
the exports until you save it again**; exports you already made are unaffected. Frames you skipped and
frames you confirmed empty, reviewed or not, never gain a box; the run only records its scores and its
marker. These rules also apply to a frame an earlier re-label sent back.

**Undo.** *Restore reopened frames...* on the Review tab (`POST /v1/review/restore-reopened`, with a
dry-run count first) puts back to reviewed every frame the run reopened that has no pending box left,
for example after you rejected the proposals (*Reject auto boxes below...*). Frames with a pending box are left alone.
A frame you had checked for overlaps gets that check back, so *Reject stacked duplicates* still leaves your
kept boxes alone, unless *Accept this spot* took a new box on it: that box was not part of your check.

**Votes and overlap suggestions.** When a rerun's labeler finds a box that is already there, its score is
added to that box's list of labelers (an extra witness for [Overlap suggestions](#overlap-suggestions));
nothing else on the box changes. New votes only count in the suggestions of frames that are not yet
checked, and pairs already answered stay answered, so tick **Ask again** on *Suggest overlaps...*
(`again=1` on the command line) after a rerun. Suggestions stay for you to confirm; how useful SAM 3 is
as a witness has not been measured.

**If a run fails.** A labelers run that fails after some frames finished (a crash, a reset GPU) writes
those frames like a Cancel does, then still reports the failure; run it again and it resumes.

### Label with any model

`train autolabel --with-model <id>` (the Auto-label tab's *Label with a model*, under Advanced; `POST /training/autolabel`
`with_model`) labels the frames with a detection model through the service's own pipeline (the
benchmark's decode, preprocess, infer, postprocess; no Python, no training GPU needed), default
threshold 0.3, by default over the **unreviewed** frames (first-pass labelling). Its boxes are pending
`source: auto` boxes like the labelers'. When it finds an existing box again, the merge drops its proposal
as for the labelers, and the model's name and score join that box's labelers: a vote for overlap
suggestions and `auto=both`, counted in the run's summary. The vote takes the best box of the same object
at IoU 0.7 or more and changes nothing else on it: no state, frame status, disagreement note or review
queue. A pending box with no evidence gets the model's evidence as before, not a vote.

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
(the Auto-label tab's preset). For a frame that was reviewed, its detections (labels mapped onto the
list) are compared with the training labels (accepted and added boxes): a detection matching
none at IoU 0.5 is *new*, one whose best match has another label a *class change*, a training box no
detection matches *missed*. Any of them puts the frame back to `unreviewed` with
`disagreement: "re-label by <model>: 1 new box, 1 missed"`, and the Review queue lists such frames
first. The detections are merged as pending auto boxes (`review`, or `accept` at a pre-accept score).
Train, re-label, review the disagreements, train again: 2-3 rounds, IR frames and rare species first.

### Find missing labels

`train autolabel --with-model <id> --missing-only` (the *Find missing labels* box on the Training and
Review tabs; `POST /training/autolabel` `missing_only`) looks for real objects with no box on frames
you already reviewed, and nothing else. An unlabelled object teaches a model that the object is
background and counts as a false alarm in validation. Existing boxes get no evidence and no vote. Run it
once per model, one task each, then review the proposals. The self-training loop above does not fit this job: it also reopens frames for missed
boxes and class changes (with `truck=car`, every truck), asks a model about frames it trained on, and
its merge drops any proposal on a rejected box, so a real car under an automatic reject stays hidden.

**Frames:** every frame of the folder under review (`status` and `source` of the filter are set to `any`
and `all`, other filters apply) that is `reviewed`, or `unreviewed` with a `disagreement` note, so a
second model can add to the first one's proposals. Frames confirmed empty are left out. So are the
frames the model trained on: `<model>.provenance.json` beside the model file gives the run,
`<training dir>/<run_id>/data/train.txt` the frames; validation and held-out frames stay in. A model
without that record (a catalog or plain local model) gets a warning and no frames are left out. When
the record is there but `train.txt` is gone the run refuses to start, since it can no longer tell.

**Defaults:** the threshold 0.3 becomes 0.5 and the limit 200 becomes 20000; any other value is used as
given, so `--threshold 0.3` cannot be asked for (0.31 can). `--accept-score-model` is refused: you check
every box.

**Rules** (`triage::apply_missing`), per detection, highest score first, against the boxes of the same
object (the same class, or two of car, truck, bus, van, so a car on a labelled truck is no new object):

| Existing box | Result |
|---|---|
| Accepted, added or pending, at IoU >= 0.3, or with at least 50% of the detection inside it (this run's boxes count) | Dropped. |
| Rejected at IoU >= 0.5 by a reviewer: by hand, as a repeat spot, as a duplicate, or with a reject reason | Dropped, whatever else matches. |
| Rejected at IoU >= 0.5 with no judgement: `reason: low_score`, or a model hint (suppressed, below threshold, verifier find) the save default rejected | The detection takes that box's place (other such boxes there are removed), so `hard_negatives.json` does not call the object a false positive. |
| None of these | Added. |

Boxes are pending `source: auto`, triaged `review`; existing boxes get no evidence and no class
suggestion. A frame that gains a box goes back to `unreviewed` with a `disagreement` note
(`re-label by <models>: N new boxes`, first in the queue), loses `overlaps_checked` until a restore
gives it back (`checked_before_reopen`) and is queued for review. **It leaves exports until you save it again**, because exports take reviewed frames only; the
summary says so. Every other frame is not written at all.

No `auto_run` marker is written: it is one slot, and several models run in turn. A rerun detects again,
and the pending boxes keep it from adding anything twice. The report adds `missing_only`, `candidates`
(`added` + `overwritten`), `overwritten`, `dropped_judged`, `skipped_trained` to `reopened`.

*Accept this spot on all frames* takes the proposals like any pending box (parked cars), `d` marks an
unclear one. For a labelers run, see [Spot memory](#spot-memory-of-fixed-cameras).

### Overlap suggestions

`train overlaps` (the Review tab's *Suggest overlaps...*; `POST /v1/review/suggest-overlaps`) takes a
closer look at the nested and partly overlapping boxes that `Reject stacked duplicates` leaves to a
reviewer, and writes a suggestion for each pair. **It changes no box, frame status, `reviewed_at` or
`overlaps_checked`**: a box changes when you accept a suggestion on the Review tab. An automatic rule
failed a blind test (13 of 99 drops removed a real object), while it named the right box on 86 of 99, so
it saves looking, not deciding. Run *Reject stacked duplicates* first: a stack is no open pair.

**Frames:** every label file of the folder under review with an open pair, reviewed frames too (382 of
the 1,169 frames measured were reviewed); `bucket`, `scene`, `camera`, `label`, `import` and `limit`
narrow it, `status` and `source` do not. Skipped frames and frames a reviewer checked
(`overlaps_checked`) are passed over.

**The check.** RF-DETR L runs on a crop of the pair's union (20% padding, so a small pair fills the
model's input) at score 0.1, and its detections are matched one-to-one to the two boxes of the same
object (car, truck, bus and van are one object) at IoU >= 0.6 and score >= 0.5, the point the research
picked (`labels/overlaps.rs` holds these once). The other signal is the *witnesses* of a box: the
independent detectors that found it (the serving model and each auto-labeler, counted once, in any state,
at IoU >= 0.7).

| The zoom sees | Witnesses | Suggestion |
|---|---|---|
| Both boxes | Any | Two objects |
| Only X | X has more, or a tie | Drop the other |
| Only X | The other has more | None |
| Neither, or it cannot look | One box has more | Drop the other, marked lower confidence |
| Neither, or it cannot look | A tie | None |

It cannot look at a class outside RF-DETR's COCO label space (raccoon, opossum, ...) or an unreadable frame.

**Guards.** A drop never takes a drawn box, a box you moved (`orig_bbox`) or one you relabelled; that pair
gets no suggestion. `model_label` alone is not the mark of a relabel: an auto-label run and the pre-fill
set it too when they map a spelling (`motorbike` on a `motorcycle` box), so a label that names the box's
own class does not guard; another class (`truck` on a `car` box) or none of the list does.

**What was measured** (the 99 + 45 judged pairs at this point): the right box on 86 of 99 drops; 13 removed
a real object. By the best other detection on the dropped box: none 2 of 31 failed, under 0.2 3 of 35,
under 0.3 2 of 17, 0.3 or more 6 of 17, which the page warns about. Two-object suggestions were right on
36 of 45. Drops from the witnesses alone were not judged; on 197 settled pairs they went 187 right, 10
wrong, a set that cannot show a real second object removed.

**A run** is a task of the training queue, like an auto-label run: one job at a time, the training lock,
low priority, pause, and Cancel keeps the finished crops (a rerun skips every pair with an answer, `unclear`
ones too, so it resumes; `again=1` in the filter looks afresh). The stages are *selecting pairs*, *setting
up the environment* (the training environment, RF-DETR L's weights through the pinned download if they are
missing), *zoom check* (`scripts/autolabel.py --zoom-pairs`) and *writing suggestions*. A failed script
keeps what finished too. It needs an NVIDIA GPU or Apple silicon, about 500 MB of VRAM and on Windows about
2.6 GB of free commit, checked before the model loads. Measured on a GTX 1070 Ti: 79 ms per crop, about 5
minutes for 3,578 crops.

**Live reviews.** Each frame is judged again under the label write lock with the file as it stands: a pair
whose box moved or went is passed over, a frame you skipped or checked meanwhile is left alone, and only a
frame that gained a suggestion is written. A frame open on the page when the job writes it answers its
save with the usual "changed since you opened it" conflict.

**On the Review tab**, *Auto: overlapping boxes* (which sets Status to any and Frames to all collected)
lists the frames strongest suggestion first. A frame opens on the box it would drop, marked `drop?` on the
frame and in the list, with its partner marked `keep`, and the decision bar says why. `y` accepts: the box
is rejected as a duplicate (out of `hard_negatives.json` and the repeats history), and a keeper that is a
hint a save would reject is accepted, so the object keeps a box. On a two-object suggestion `y` keeps both,
and the save records `overlaps_checked`. To turn a suggestion down, do anything else: decide the boxes
yourself, or save with both kept. The twin rules are `liveSuggestions` / `acceptOverlap` in
`assets/review-rules.js` and `live` / `accept` in `collect/labels/overlaps.rs`.

### Dataset check (cross-check)

`train refine` looks for what the labels of your reviewed frames got wrong or left out: objects no box
labels, labels no model finds, and a box of another class than the one an independent model sees. It
exists for the frames every trained model has already seen, which `--with-model --missing-only` cannot
check (it skips them); for the frames reviewed since your last training run, the
[quick check with a model](#quick-check-with-a-model) asks that model instead and trains nothing. It
starts from the Refine tab, the command line or the job queue. **It changes no box, frame
status, config entry, model file, `runs.json` or `benchmark.json`**: it writes suggestions to
`LabelFile.refine` and a report, and a box changes only when a reviewer accepts a suggestion.

**How it works.** The camera days of the reviewed frames (a camera on one date; frames of one day are near
copies) are split into K folds, whole days at a time (`refine/folds.rs`: largest day first, each into the
fold with the fewest frames so far, a fixed seed, so a rerun over the same frames uses the same folds).
For each fold a short model trains on the other folds' frames and is then asked only about the days it
never saw. Before it runs, the job reads the fold model's `train.txt` and `valid.txt` and stops with an
error if a frame of its own fold is in either, by name and by the source the dataset recorded for each
image (`image_sources.json` beside the lists: the dataset cuts long names and numbers a name taken twice,
so a name alone could miss one). The fold model is the training run as it is (`train run`'s
stages), installed in the job's work folder, registered nowhere and benchmarked never. Its export applies no
size floor, whatever `collect.size_floor` says: the fold models try to learn every label, so the recall by
size measures what a model of the recipe finds (a model trained under the floor would never find what the
floor left out, and the table would only repeat the floor). The judge still treats a box under the floor
as an ignore region, and the per-fold class counts behind `verify` count what the fold model trained on,
boxes under the floor included.

**The judge** (`refine/judge.rs`, pure) compares the detections of a fold model at score 0.15 or more
with the frame's training boxes, ignore regions left out. It matches same-object pairs one to one first (best IoU first; car, truck,
bus and van are one object), so a loosely drawn label is not reported as a missing box plus a label to
verify. Then:

| Result | When |
|---|---|
| Loose label (counted in the report) | A matched pair below IoU 0.75 on a box no reviewer drew, moved or relabelled. With a detection at score 0.5 or more, of a class RF-DETR L can see, on a label not too small (the `add` floor), it is a candidate for the [closer look](#closer-look-at-loose-labels-tighten), which may propose a `tighten`. |
| `add`: a box for an object no box labels | An unmatched detection at score 0.5 or more (0.3 or more at a [static spot](#static-spots) that holds without the frame's camera-day), 10 px or more on its short side at model input, that no live box of its object and no ignore box covers, and no judged reject of its object lies on at IoU 0.5 or more. |
| `relabel`: another class | A training box no reviewer drew or relabelled, with a detection of another object at score 0.8 or more at IoU 0.7 or more, and no detection of its own object at score 0.3 or more at IoU 0.3 or more. The detection makes no `add`, the box no `verify`. |
| `verify`: a label no model found | An unmatched training box no reviewer drew or moved, with no detection of its object at score 0.15 or more at IoU 0.3 or more, of a class with 50 boxes or more in the frames its fold model trained on, 10 px or more on its short side. |

The constants (`judge.rs`):

| Name | Value | Meaning |
|---|---|---|
| `MATCH_MIN_SCORE` | 0.15 | Detections below it are not used at all (no pair, no "found it"); the fold models run with it as their threshold. At 0.05 a fold model's faint guesses overlapped nearly every label and silenced `verify`: 2 of 2,755 training boxes on the first real run, although the fold models missed 304 at score 0.3. Labels whose best detection scored under 0.15 were about half label problems on a by-hand check, those at 0.15-0.3 almost all real, dim objects. |
| `LOOSE_IOU_MIN` | 0.1 | IoU from which a detection and a training box of one object are a pair. |
| `GOOD_IOU` | 0.75 | IoU from which a pair is a good label; below it the label is loose. |
| `ADD_SCORE` | 0.5 | Score from which an unmatched detection is a missing box. |
| `ADD_SCORE_AT_SPOT` | 0.3 | The same at a static spot (never for people or animals, which form no spot): measured on the kept 8-epoch fold detections of 851 frames, an uncovered detection at a place labelled on 10 or more other frames of its camera is real about 90% of the time down to 0.3, while elsewhere detections at 0.3-0.5 are almost all false. |
| `FLOOR_PX` | 10 | Short side `min(w / W, h / H) * imgsz` under which no `add` or `verify` is made. With `collect.size_floor` set, that rule applies as well (the export's, at its own input size; nothing under either is suggested, so a permissive floor never lets through what the fold models cannot see), and a training box under the floor is an ignore region for the check: no `verify`, `relabel` or loose count, out of the recall but for `recall_by_size` (where it still counts, so a floor can be chosen from it), while it still covers detections of its object (no `add` on it). |
| `VERIFY_IOU` | 0.3 | IoU from which a detection of the object has found a training box. |
| `MIN_CLASS_BOXES` | 50 | Boxes of a class in what the fold model trained on (the selection without the fold's frames) from which a `verify` is made for it: a camera day that holds most of a class leaves its own fold model few. |
| `RELABEL_IOU` | 0.7 | IoU from which another object's detection lies on a training box. |
| `RELABEL_SCORE` | 0.8 | Score from which that detection is evidence for a `relabel`. |
| `RELABEL_BLOCK_SCORE` | 0.3 | Score from which a detection of the labelled object blocks a `relabel`. |
| `REJECT_BLOCK_IOU` | 0.5 | IoU from which a judged reject of the object blocks an `add` (`labels::MATCH_IOU`). |
| `FOUND_SCORE`, `FOUND_IOU` | 0.3, 0.5 | When a detection finds a training box in the report's recall. |
| `TIGHTEN_FOLD_SCORE` | 0.5 | Score from which a fold model's detection of a loose label makes it a candidate for the closer look: weaker finds are too often a short-trained model's guess to spend a crop on (the `add` cut-off). |
| `TIGHTEN_ZOOM_SCORE` | 0.5 | Score from which a detection of the closer look counts as a box: the overlap check's measured operating point (`overlaps::ZOOM_SCORE`), same model, same crops. |
| `TIGHTEN_AGREE_IOU` | 0.6 | IoU from which the closer look's box agrees with the fold model's: the overlap check's match IoU (`overlaps::ZOOM_MATCH_IOU`). Two independent looks that agree outvote the label; one alone does not. |
| `TIGHTEN_SPAN_INSIDE` | 0.8 | Share of a second box of the closer look inside the label from which the label spans that object too, and is not moved onto `Z`: the overlap check's nested share (`stacks::NESTED_SHARE`). |
| `TIGHTEN_EDGE_PX` | 1 | Pixels within which an edge of `Z` lies on an edge of its crop (inside the image): the object was cut there, no proposal. The script keeps boxes to 0.1 px on whole-pixel crops. |
| `MAX_TIGHTEN_CANDIDATES` | 2,000 | Most loose labels one check looks at closer, the most confident fold detections first: under 3 minutes at the measured 79 ms per crop, about half an hour on a GPU ten times slower. |

#### Static spots

`collect/labels/statics.rs` (pure) finds the places of a fixed camera where the same object is labelled
frame after frame (a parked car): the training plan caps the frames that show nothing else
([static cap](#assembly-srctrainingdatasetrs)), and the dataset check accepts a weaker detection there as
an `add`. A spot is a cluster of training boxes (accepted or added, no ignore region) of one object of one
camera, in coordinates normalised to the frame, on at least `MIN_FRAMES` 10 distinct frames of at least
`MIN_DAYS` 2 days on which the object stayed. **Dwell rule** (`MIN_DAY_SHARE` 0.3): a day counts toward
`MIN_DAYS` only when the spot is labelled on at least 30% of that camera's frames of that day that have any
training box (of any class: what the reviewer labelled that day; empty frames are left out, their number
follows the collector's rate limits rather than the place). Visitors' cars stopping at a gate or a
driveway mouth now and then, among the day's other traffic, are on a small share of its frames; a parked
car is on most of them. Clustering per camera and object: the boxes are put in a canonical order (box,
frame, day, scene, label) so the result depends on the set of boxes only; the box with the most support
(distinct frames with a box at IoU >= `SPOT_IOU` with it) seeds a cluster, its members are, per frame, the
unassigned box closest to the representative at IoU >= `SPOT_IOU`, and the representative becomes the
coordinate-wise (lower) median of the members, both taken again until they settle (at most 20 rounds). A
cluster on enough frames and days is a candidate; one that is not frees its boxes but none of them seeds
again. Every member lies within `SPOT_IOU` of one representative box, so two neighbouring parking places
do not chain into one as single linkage would. Then every box (sampled or not, below) is counted once, in
one pass, at the candidate of its object with the best IoU at `SPOT_IOU` or more (ties to the first id):
a spot's `frames`, `days`, `stay_days`, `by_day`, `scenes` and label are those boxes' (one per frame, the
closest), so they are exactly what `statics::spot_at` finds later; a candidate that no longer meets the
rule on them (a neighbour took its boxes) is dropped and the rest counted again. A box lies at a spot
(`statics::spot_at`) at IoU >= `SPOT_IOU` with its representative, same object, best IoU first. Ids are
`<camera>/<object>/<x0>-<y0>-<x1>-<y1>` with the representative in steps of 2% of the frame (`vehicle` for
every vehicle class): the same labels give the same ids, a rerun after more labels keeps an id unless the
median crosses a step, two spots that round alike get `~2`.

**Bounds** (`MAX_OBS_PER_CAMERA` 4,000): a camera with more boxes of classes that may be static is
clustered on the boxes of a fixed sample of its frames (lowest FNV-1a hash of the frame's path first, whole
frames; the dwell rule uses the sample's own frames per day), then counted on every frame as above. A spot
needs 10 frames on 2 days, so a place labelled on 1 in 400 of a camera's boxes still shows up. No neighbour
list is stored: support is counted on demand by a sweep over the boxes sorted by `x0` within the window an
IoU of `SPOT_IOU` allows (`x0` within `[x0 - (1-t)/t w, x0 + (1-t) w]` for a box of width `w`), kept as an
upper bound and recounted only when it reaches the top of a max-heap (supports only fall as boxes are
assigned, so the seed is the one the eager rule picks; a test compares both on the same boxes).
Clustering a camera costs O(S x W) IoUs for the S <= 4,000 boxes clustered (W the boxes in one window,
at most S: at most 16 million for one car parked in every frame) and O(S) memory; counting costs O(N x K)
for N boxes and K spots of their object, O(N) memory. Storing every neighbour pair had cost one jittered
car in 20,000 frames 21.6 s and 4.1 GB in a release build; now a test builds the tables of that car in
20,000 and in 40,000 frames (plus scattered boxes) in under 2 s together in a debug build, and bounds the
IoUs computed (`clustering_is_bounded_by_the_sample_and_counting_is_linear`). The 851 frames below take
about 60 ms per table in a debug build.

**Only things that park form spots** (`statics::may_be_static`, one place in the code): a vehicle class
(car, truck, bus, van, bicycle, motorcycle, train, boat, airplane) or another COCO class that is neither a
person nor an animal (a bench, a potted plant; aliases such as `pottedplant` apply). People and animals
(`statics::is_animate`: `person` and the auto-labeler's 33 animal names, `autolabel::Category`, a superset of
the benchmark's COCO `animal` group, with raccoon, opossum, skunk, fox, deer, squirrel and others) are not
clustered at all: a person or an animal is never parked, their frames are the scarce data a dataset needs
most, and the 0.3 `add` cut-off was measured on vehicles. So they make no spot, never make a frame
static-only (a parked car with a person beside it is not static-only), are never capped and never get the
lower cut-off. A class this build cannot place, a custom name outside those lists (`groundhog`), forms no
spot either: the safe side, since a missed cap or a missed low-score `add` costs little while a capped-away
animal frame costs a scarce class; give such a class a COCO name, or accept that its repeats are not
capped.

`SPOT_IOU` 0.6 was measured on the 851 reviewed frames of one installation (`static_spots_measure_on_a_copy`,
2,756 training boxes of wanted classes), before people were excluded (the person spots are listed for the
record; since then they are none, and nothing else changed: 19 spots at 0.6, the same car and truck counts,
420 static-only frames instead of 453):

| `SPOT_IOU` | Spots (back / front / garage / patio) | Car boxes at a spot | Truck | Person | Static-only frames (of 731 with boxes) | Largest IoU between two representatives | Frames with two boxes at one spot |
|---|---|---|---|---|---|---|---|
| 0.5 | 26 (2 / 7 / 3 / 14) | 2,274 of 2,352 | 126 of 129 | 64 of 243 | 513 | 0.36 | 19 |
| **0.6** | **23 (2 / 7 / 3 / 11)** | **2,206** | **126** | **43** | **453** | **0.49** | **18** |
| 0.7 | 24 (1 / 8 / 2 / 13) | 2,162 | 126 | 10 | 418 | 0.63 | 18 |
| 0.8 | 28 (0 / 9 / 3 / 16) | 2,058 | 126 | 0 | 382 | 0.81 | 16 |

Neighbours do not merge at any of them (the frames with two boxes of one object at one spot, 16 to 19, are
the same few double labels at every IoU). From 0.7 on, one place splits into two or more spots: two
representatives overlap at IoU 0.62 to 0.81 (the patio camera's two largest places, 373 and 381 frames at
0.6, become spots of 308, 338 and smaller ones at 0.8), which would give a place twice the cap. 0.6 keeps
each place one spot (the closest two representatives, 0.49, are two cars of a row seen at an angle), is the
IoU the dataset check's 90% measurement used, and puts about 94% of the car boxes in spots on all 851 frames
(the earlier estimate on one run's training split was 1,243 of 1,472, 84%: more frames make more spots).

`MIN_DAY_SHARE` 0.3 was measured on the same frames (a later copy, 2,761 boxes: five accepted `add`s
more), at `SPOT_IOU` 0.6, every count over all 851 frames (`static_spots_measure_on_a_copy`):

| Dwell share | Spots (front / garage / patio) | Car boxes at a spot (of 2,357) | Truck (of 129) | Static-only frames (of 731) | Spots it removes |
|---|---|---|---|---|---|
| none (0) | 19 (7 / 2 / 10) | 2,211 | 126 | 420 | - |
| 0.2 | 15 (6 / 2 / 7) | 2,138 | 126 | 412 | four: front 41-1-48-9 (19 frames on 4 days, median day share 0.22), patio 21-5-28-10 (21 on 6, 0.04), 40-0-41-1 (19 on 6, 0.03), 42-0-43-1 (23 on 7, 0.12) |
| **0.3** | **15 (6 / 2 / 7)** | **2,138** | **126** | **412** | **the same four** |
| 0.5 | 13 (6 / 2 / 5) | 2,035 | 126 | 371 | those and patio 39-4-44-8 (51 on 2 days, 0.38) and 39-0-41-2 (51 on 8, 0.10) |

The four places 0.2 and 0.3 remove are the smallest of the 19 (19 to 23 frames spread over 4 to 7 days,
on a few percent of each day's frames): cars at the frame's edge now and then, not parked. Every place of
hundreds of frames stays at any share (patio 373 and 381 frames at a day share of 0.95 or more, front 158 at
1.0, the garage truck 123 at 0.94 or more, patio 289 and 286 at 0.47 or more). 0.5 also removes two places
of 51 frames and leaves mid-sized ones on 2 or 3 qualifying days (patio 0-9-2-14, 137 frames: 2), at the
edge of `MIN_DAYS`; 0.3 keeps them with the same result as 0.2, so it is the value.

The dataset check builds one spot table per check from its selection (`Selected::spots`, the label
snapshot every fold uses). Its evidence for a frame leaves out the frame's camera-day
(`Spot::holds_without`: the spot's frames without that day must still reach 10, and the days the object
stayed without that day 2), since one day's frames are near copies of each other; it does not leave out
the whole fold, because the evidence is the reviewer's labels, which the fold model's blindness does not
concern, and other days of the fold are other looks at the place. An `add` at a spot carries `spot` (the
id: the check's evidence, shown; the Static objects card places a suggestion by its box, not by this id,
see the Refine tab), its own class from 0.5 up (a clear truck at a car's place is another vehicle parked
there) and the spot's most common label through the class list below (a truck found at 0.4 where a car is
labelled becomes a car), and the reason *a car is labelled at this place on N other frames of this camera;
a model that never saw this camera-day found one here, score 0.xx* (*found a truck here* when its class
is not the spot's). Its strength is that of any `add` of its score (`refine::strength` is unchanged: the
same score ranks the same, at a spot or not). The check's counts (`found`, `adds_at_spots`) are what the
label files keep: `refine::apply` drops a weaker copy of one finding (two detections of one parked vehicle
both rewritten to the spot's label), and so does the count.

Replayed on the same 851 frames from the kept detections of a 4 folds x 8 epochs check
(`refine_replay_on_a_copy`, people and animals excluded): 19 spots, all of vehicles; `add` 172 (151 at
static spots, 21 elsewhere; 123 before the spot rule), `verify` 20, `relabel` 0; a parked pickup labelled
on 286 frames of its camera gets an `add` on 64 more frames (41 before). Recall at score 0.3: 2,123 of
2,266 boxes at static spots (93.7%, 95% 92.6-94.6) against 330 of 490 elsewhere (67.3%, 63.1-71.4). The
answered suggestions (5 dismissed adds, 1 kept verify) stayed. Again with the dwell rule, the kept counts
and the confident class, on the later copy (six `add`s of the first replay accepted in between): 15 spots;
`add` 165 (147 at static spots, 18 elsewhere), `verify` 19, `relabel` 0; recall at score 0.3: 2,093 of
2,198 boxes at static spots (95.2%, 95% 94.2-96.0) against 365 of 563 elsewhere (64.8%, 60.8-68.7); the
same 6 answered suggestions stayed.

Detection scores are clamped to 0..1 and boxes to the frame first, so a suggestion never holds what a save
would refuse. A box "covers" a detection (`labels::covers`) at IoU 0.3 or more, or with at least half of the detection
inside it, never the other way round. Only suggestions a reviewer would see (`refine::live`) are kept.
Hidden classes are not judged. Only `reviewed` frames are judged, at write time, under the label write
lock and on the file as it is then: a frame saved back to unreviewed or skipped meanwhile gets nothing,
and a confirmed-empty frame can get an `add`.

**What it writes.** `refine::apply` changes `LabelFile.refine` and nothing else: the unanswered suggestions
of the kinds a fold makes (`add`, `verify`, `relabel`) are replaced by what it found, and the closer look
replaces the unanswered `tighten` of the labels it answered (see below); answered ones and
other kinds stay, except an answered `verify`, `relabel` or `tighten` whose box is gone (moved, relabelled, deleted):
it could never be live again and is dropped (an answered `add` stays: it keeps the question from coming
back). A found suggestion is dropped when it is the same finding as an answered one (the same `target` and,
when both have one, `target_label`; for an `add`, a box of the same object at IoU 0.5 or more), a weaker
copy of another found one, or outside what a save accepts.
A frame holds at most 64, strongest first (`add`, then `relabel`, then `tighten`, then `verify`, then by
score). Over 64, a write gives up its own found suggestions, the weakest first: it never pushes out an
answered one or one of a kind another write owns (the closer look's `tighten` never displaces a fold's
`verify`, nor a fold's `add` an open `tighten`). A frame whose suggestions come out the same is not written, so its revision stays and an open Review
page can still save. The file format is in [DATASET.md](DATASET.md#label-file-version-1).

**What a reviewer's answer does** (`refine::accept`, `refine::dismiss`; stale suggestions change nothing):

| Suggestion | Accept | Dismiss |
|---|---|---|
| `add` | A new accepted box (`source: auto`, `reason: refine`, the suggestion's class and score) takes the place of the closest unjudged reject of its object at IoU 0.5 or more and removes the other such rejects, so `hard_negatives.json` does not call the real object a false positive (an unjudged reject is `low_score`, auto-dismissed, or a model hint the save default rejected; a reject with a reason, a spot edit or a duplicate is judged and blocks the `add`). The frame is no longer confirmed empty, and `overlaps_checked` ends if the new box is in a stack or an open pair. The suggestion goes. | Answered `dismissed`. |
| `relabel` | The box takes the class, the old label stays in `model_label`; `overlaps_checked` ends if the box is now in a stack or an open pair. The suggestion goes. | Answered `dismissed`. |
| `tighten` | `orig_bbox` keeps the old box, `bbox` becomes the proposal; `overlaps_checked` ends if the box is now in a stack or an open pair. The suggestion goes. | Answered `dismissed`. |
| `verify` | Answered `kept`; the label stays. | Answered `dismissed`. |

A dismissal changes no box. Answers are remembered: a rerun does not ask the same question again. A
suggestion names its box by `target` and `target_label`, so two boxes of different objects drawn at the
same place each get their own.
The reviewer's own edits end a suggestion without an answer (`refine::live`, computed when the file is
read): a `relabel`, `tighten` or `verify` is stale once its box is rejected (also as too small), marked
unsure (`d`), deleted, moved (`tighten`, `verify`) or relabelled (`relabel`). An unsure
or too-small box is an ignore region, which the judge never judges either; taking the unsure mark back
makes an unanswered suggestion about it live again. An `add` is stale once a box of its object or an ignore
box covers it, or a judged reject of its object lies on it.

**Finding them on the Review tab.** `GET /v1/review/next?auto=refine` lists the frames with a live
suggestion, of any review status (they are mostly reviewed frames, so add `status=any&source=all`),
strongest first by the frame's strongest suggestion (item fields `refine_strength` and `refine_open`, the
number of live suggestions); `refine_kind=add|tighten|relabel|verify` (only with `auto=refine`) narrows the
filter, the order and the count to one kind. A frame without a label file is never listed, a skipped one
neither. Classes are spelled and filtered through `collect.classes` and `collect.hidden_classes`, as the page
shows them. `GET /v1/review/labels` answers `refine_live` beside `overlap_live`: the live suggestions, each
`{index, kind, target, target_label, bbox, label, score, why, spot, run, target_box, replaces}` (`index`
into `labels.refine`, `target_box` and `replaces` into `labels.boxes`). The page answers them in
`assets/review-rules.js` (`liveRefine`, `acceptRefine`, `dismissRefine`, twins of `refine::live`, `accept`
and `dismiss`, compared in `tests/it/js_rules.rs`) and saves the file as usual: an answer is a change to
the file a save checks (`refine::check`) and keeps; a check that wrote the frame meanwhile makes the save a
409 like any other write. A big folder answers `auto=refine` from the review index like the other filters.

**The report** is `<folder under review>/refine/report.json` (written atomically, replaced by each check;
the dataset walkers and retention do not look in `refine/`). Fields: `version`, `run`, `created`,
`cancelled`, `folds`, `folds_done`, `recipe` (`model`, `imgsz`, `epochs`, `batch`, `from`), `camera_days`,
`frames_selected`, `frames_checked`, `left_out` (`pending`, `off_class_list`, `lost`, `unreadable`,
`unreadable_folders`, `filtered`), `written` and `found` (`add`, `verify`, `relabel`, `tighten`; `tighten`
absent in earlier reports, read as 0), `frames_written` (a frame the folds and the closer look both wrote
counts once), `loose`, `tighten` (the check took the closer look; false in earlier reports),
`tighten_candidates` (loose labels sent to it), `tighten_left_out` (candidates past its limit of 2,000,
not sent; absent in earlier reports), `zoomed` (of those sent, the crops it answered), `found.tighten`
(this check's proposals that were kept, not earlier checks' open ones left in place), `found_score`,
`found_iou`, `recall`, `recall_by_group` (camera x scene x class), `recall_by_size`
(short side at model input: under 8, 8-12, 12-24, 24-48, 48+ px), `static_spots` (spots of the selection),
`recall_at_spots` and `recall_elsewhere` (training boxes at a static spot that holds without their
camera-day, and the others), `adds_at_spots` (of `found.add`, those at a spot; the last four absent in
reports of earlier checks, read as 0), `folds_every_label` (true: the fold exports applied no size floor, so
the fold models trained on every label; false in reports of earlier checks, whose fold exports applied the
floor in force then), `size_floor` (the `collect.size_floor` the judge used; absent when off or in earlier
reports), `ignore_mode` (`training.ignore` the fold models trained with; absent in earlier reports and in
quick checks), `mode` (`folds`, or `model` for a [quick check](#quick-check-with-a-model); absent in earlier
reports, read as `folds`), `model` (a quick check's: `name`, `run`, `recipe`, `imgsz`, `trained_on`,
`validated_on`, `same_day`, `classes_missing`, `by_source`, `device`; absent for fold models),
`predict_s_per_frame` (seconds the detections took per frame, model loading included: a later quick
check's time estimate; absent in earlier reports) and `warnings`. Each recall has `boxes`,
`found`, `recall` and a Wilson 95% interval (`low`, `high`): how often a fold model found the labelled
objects of days it never saw, at score 0.3 and IoU 0.5 (in a quick check: how often its model found the
labelled objects of the frames it never saw). `REPORT_VERSION` stays 1: the new fields are optional, so a
build from before reads a quick check's report as a full one (it shows the counts, not the mode).

**Frames and recipe.** Every reviewed frame of the folder under review that can be exported (reviewed
frames with pending boxes or labels off the class list are left out and counted); `--frames` narrows it by
`scene=` and `camera=`, and then narrows the fold models' training too. At least 40 such frames on at least
2 camera days are needed; K is lowered to the number of camera days when there are fewer (a warning says
so). The preview and the report warn when a fold's model trains on under 90% of the frames the fold holds
out (one camera day dominates the check: its fold model learns from less than it is asked about; two even
folds that differ by a frame are no warning) and when a fold model
trains on fewer than 50 boxes of a class the selection has 50 of (that fold's labels of it get no
`verify`). A folder under review that is not available (an unplugged drive) refuses the check and the
preview (`folder_missing`) with the Review page's message, never as an empty dataset, and the report is
never written into a folder that is gone (only `refine/` is created). The fold models use the newest finished training run's model, input size, batch and multi-scale choice,
else RF-DETR nano at the GPU's suggested batch; `--epochs` epochs, early stopping off (every epoch runs, the
best raw epoch is exported), the default learning rate, no augmentation.

**A run** is a task of the training queue (`TaskKind::Refine`, one job at a time), with the stages
*selecting the reviewed frames*, one *checking fold N of K* each (export, train, check the fold is unseen,
run the model, write), with the closer look *looking closer at N loose labels*, and *writing the report*. It needs an NVIDIA GPU or Apple silicon and the training
environment (a download over 500 MB needs `--allow-large` or `allow_large_downloads`). It takes the training
lock after the selection and holds it to the last fold's writes, as the auto-label and overlap jobs hold it;
each fold's training uses that hold, so no other process (the service, a `train` command) trains or labels
between a fold's training and its prediction or between two folds, and a lock already held elsewhere
refuses the check before anything trains. The selection is one snapshot for every fold: a frame whose image
or label file went since is left out of the fold's training and check, and a frame the fold model cannot
read (a damaged JPEG) gets no result; both are counted in a warning and the fold goes on. A frame whose
suggestions cannot be written is a warning (the first 20 named, the rest counted in one line); a fold none
of whose frames could be written counts as failed. Each fold's folder
(`<training dir>/refine/<id>/fold-N`) is deleted when the fold is written, the work folder at the end however
the run ends; a crashed run's is removed by the storage sweep (`WORK_DIRS`). Pause suspends the training
processes; the export, prediction and writes of the running fold finish first.

- **Cancel** keeps the suggestions of the finished folds (a fold is written whole or not at all) and ends
  with a report marked `cancelled` and a summary saying how many folds ran. A cancel before any fold
  finished is a plain cancel.
- **A failed fold** (training fails, the fold's frames are found in its training lists, none of its frames
  could be written, or the folder under review went) stops the job and fails it. The folds before keep their suggestions and the report, with a warning that the check stopped.
  A fold model is throwaway: there is nothing to retry, run the check again.
- **A rerun** checks every fold again. The folds are the same for the same frames; unanswered suggestions are
  replaced, answered ones stay (but for those whose box is gone).

**What a rerun replaces.** On every frame it checks, a check replaces the unanswered `add`, `verify` and
`relabel` suggestions with what its own fold models find: a suggestion they do not find again is gone, even
when an earlier, stronger check made it. Answered suggestions (kept or dismissed) stay, an unanswered
`tighten` stays unless the check's closer look answered its label (then it is replaced by what that look
supports, or goes when it supports nothing; a check without the look, `--no-tighten`, never touches one), frames outside the check (another scene or camera of `--frames`, frames left out) are not touched, and no
box changes. So a short check after a good one costs findings: on one dataset a check of 4 folds x 8 epochs
left 139 open suggestions, and two checks of 2 folds x 1 epoch, whose fold models found almost nothing, cut
them to 88. The preview counts what a start would replace (`open_suggestions`, `open_frames`: unanswered,
of those three kinds and live as the Review tab shows them, on the selected frames; with the closer look
also `open_tightens`, the live open `tighten` there, a bound since the look replaces only those of the
labels it answers, and `open_frames` then counts the frames holding either) and warns *this check
replaces the N open suggestions on M frames ...* (with the look: *... and up to K open proposed boxes for
loose labels ...*); it also warns when the fold models are short, under half
the family's recommended minimum epochs (`recipe::recommended(..).min_epochs`, 10 for every family, so 1 to
4 epochs; `refine::weak_epochs`), since fold models of 1 epoch found no missing box where 8 epochs found
123. On the Refine tab a start over open suggestions (with the look, open `tighten` count too) is refused (409
`{success: false, needs_confirm: true, open_suggestions, open_tightens, open_frames, folder_missing: false,
message}`) and the card asks inline: *Start and replace N open suggestions* (*up to N* when open `tighten`
are among them) posts again with `replace_open: true`, *Keep them* starts nothing. Answer them
on the Review tab first, or confirm to recompute them. `train refine` is an explicit command and asks
nothing, but prints the same warnings before it starts.

```sh
blue-onyx-prism train refine [--dry-run] [--folds 4] [--epochs 8] [--frames "scene=ir,camera=patio"] \
    [--gpu cuda:N] [--allow-large] [--tighten | --no-tighten] [--verbose]
```

`--folds` is 2 to 8 and `--epochs` 1 to 50 (defaults 4 and 8). `--frames` takes `scene=<day|night|ir>` and
`camera=<name>` pairs, comma-separated, a key may repeat. `--dry-run` runs the preview: frames, what the
export leaves out and why, camera days, each fold's size, the model, whether loose labels get the closer
look, and a rough time (shown only when a GPU can train). It needs no GPU or download, writes nothing, and
prints `refused:` with the reason when the check would not run. `--tighten` / `--no-tighten`: the closer
look at loose labels (below); by default it runs when RF-DETR L's weights are downloaded.

#### Closer look at loose labels (`tighten`)

A loose label (an auto-label box half off its object, say) turns a correct detection into a miss plus a
false alarm when a model is scored, and teaches sloppy boxes. The fold models see them (148 loose pairs
on the 851 reviewed frames measured), but a fold model's box alone is not precise enough to overrule a
label: fold models train a few epochs at a small input size (384 px). So a second, independent and sharper
look decides, the overlap check's zoom stage (`refine/tighten.rs`):

1. **Candidates** (`judge.rs`, pure): each loose pair the judge counts (not a reviewer's box: not drawn,
   no `orig_bbox`, no `model_label`; not an ignore region, not under the size floor or 10 px at the fold
   models' input), whose fold detection `D` scores `TIGHTEN_FOLD_SCORE` 0.5 or more, on a label of a class
   RF-DETR L can see (`coco_class`: a `van` cannot be looked at). Each fold appends its candidates, with the
   fold model's detections of the frame, to `<work>/tighten-candidates.jsonl` as it finishes.
2. **The look**, once after the last fold, under the training lock the check holds:
   `scripts/autolabel.py --zoom-pairs` (unchanged; the overlap check's runner, `overlaps::zoom::prepare` and
   `run`) with RF-DETR L on one crop per candidate, the union of the label `L` and `D`, padded by 20%
   (`--zoom-pad`), at score 0.1, one flushed JSONL line per crop (with the crop it cut and the image's size,
   which rule (g) uses). At most `MAX_TIGHTEN_CANDIDATES` 2,000 candidates are sent, those with the most
   confident `D` (ties by frame and place, so the same on every run); the rest are left out with a warning
   in the report and the summary (`tighten_left_out`), and their open `tighten` stays. It is the job's step
   *looking closer at N loose labels* (N the labels sent; its progress line counts *i of N crops*), and
   needs what the overlap check needs: the training environment, RF-DETR L's weights
   (0.14 GB, `train-base:rfdetr-large`), about 500 MB of VRAM and on Windows about 2.6 GB of free commit,
   checked before the model loads.
3. **The decision** (`judge::tighten_verdict`, pure): the closer look's boxes of the object are its
   detections of the same object at `TIGHTEN_ZOOM_SCORE` 0.5 or more, clamped to the frame; `Z` is the one
   with the best IoU to `D` (ties: the higher score, then the earlier detection). A `tighten` to `Z`
   (`target` = `L` with its label, `score` = `Z`'s, the reason naming both looks) only when, in this order:
   (a) `Z` agrees with the fold model: IoU(`Z`, `D`) >= `TIGHTEN_AGREE_IOU` 0.6 (else *no_agree*, also
   with no `Z`); (b) the label is loose against `Z` too: IoU(`L`, `Z`) < 0.75 (else the label was fine and
   the fold model imprecise: *label_fine*); (e) no box of the closer look fits the label at IoU 0.75 or
   more (else it found the label's object as labelled and `D`, `Z` are another object, say a parked car
   behind the labelled one that the fold model found instead: *label_found*); (f) the label spans no
   second object the closer look tells apart: no other of its boxes, not `Z` again (IoU with `Z` under
   0.6 and not nested in or around it), no larger than `L` and lying `TIGHTEN_SPAN_INSIDE` 80% or more
   inside `L` (the overlap check's nested share), that no other box of the object on the frame covers
   (else moving the label onto `Z` would leave that object unlabelled: *spans_two*); (c) `Z` is a proper
   box not under the floor (the `add` floor; *small*); (g) `Z` was not cut by its crop: no edge of `Z`
   within `TIGHTEN_EDGE_PX` 1 px of an edge of the crop that is not the image's edge (the object goes on
   past the padded union: *crop_edge*); (d) `Z` is no other object's box: no other training box or ignore
   region of the object overlaps `Z` more than `L` does (the two looks found the neighbour: *neighbour*).
   The replay (`refine_replay_on_a_copy` with `BOP_REFINE_ZOOM=1`) prints each answered candidate without
   a proposal with its word, and the totals.
4. **The write** judges each frame afresh under the label write lock: the frame still reviewed, the
   candidates found again from the kept fold detections on the file as it is now, each matched to its
   crop's answer by its exact boxes. Only the open `tighten` of a label whose candidate the look answered
   is replaced (by what the answer supports, or nothing); an open `tighten` on any other label of the frame
   (not a candidate this time, its crop unanswered, left out by the limit) is kept as it is; so a check
   without the look, a failed or cancelled look, or an unreadable frame never deletes an earlier one.
   `found.tighten` counts this look's proposals that were kept, never the earlier ones left in place.
   An answered `tighten` stays and is not asked again: `refine::same_finding` keys a `tighten` on its target
   box (exact `bbox` and label), whatever box it proposes. Accepting one moves the box and keeps the old one
   in `orig_bbox`, so the label is a reviewer's from then on and is never a candidate again.

**Default.** On when RF-DETR L's weights are downloaded (the overlap check downloads them, or *Download
RF-DETR L if missing* on the Review tab); else the check runs without it and the preview says so in one
warning. Asked for (`--tighten`, the Refine tab's box ticked) the weights come with the check, through the
same large-download gate as the overlap check's (at 0.14 GB they are under its 500 MB limit). `--no-tighten`
counts loose labels only.

**Cancel, pause, failure.** The look pauses with the queue (its script is suspended like the overlap
check's). A Cancel during it keeps the crops that finished: their boxes are written, the report is
`cancelled` and the summary says *stopped by Cancel during the closer look ... after N of M crops*. A Cancel
during the folds skips it (Cancel means stop; the report says the look did not run). After a failed fold it
still runs on what the finished folds found, and the task's error says what it did (*... and the closer
look at their loose labels then answered N of M and saved K better-fitting boxes*, or that it failed). A failure of the look (the environment, the download, the
memory check, the script) keeps every fold's suggestions: the check ends as usual with the failure in its
summary and in the report's warnings.

**Time.** About 79 ms per crop on a GTX 1070 Ti (the overlap check's measurement) plus the environment
check, the script's self-check and loading RF-DETR L; the preview adds `90 s + 0.079 s` per selected frame,
at most 2,000 (one crop per frame on average; 851 frames: about 3 minutes), and, since the count of loose
labels is known only after the folds, words it as a bound, not a total: "the closer look adds a minute or
two to set up, then up to about a second per loose label it looks at (at most 2,000)". `time_basis` is the
folds' as before. Replayed on the 851 frames (kept fold detections,
`refine_replay_on_a_copy`): 148 loose pairs, 82 candidates on 74 frames (77 cars, 4 persons, 1 truck).

#### Quick check with a model

The full check trains throwaway fold models so that every reviewed frame is judged by a model that never
saw its camera day: about 70 minutes of GPU for 851 frames. After you review a few hundred new frames, the
custom model you already trained has never seen those either. `train refine --with-model <name>` (the
Refine tab's *Quick check with a model you trained*) judges them with that model instead: **no training**,
seconds to minutes, and no GPU unless the closer look runs (the model runs on its own configured device,
an integrated GPU or the CPU included). It is the same check otherwise (`refine/quick.rs`): the same
selection, judge, writer, report, queue, lock and promise of changing nothing but `LabelFile.refine` and the
report.

**Which frames.** The reviewed frames of the selection (the same `--frames` scene and camera filter) minus
every frame the model trained or validated on. The model's `<stem>.provenance.json` (beside its model file)
names its training run; the run folder is that run's `run_dir` in `runs.json` of the training folder, else
`<training dir>/<run id>`. Its `data/train.txt` lists what it trained on and `data/valid.txt` what it
validated on: a validation frame chose the checkpoint, so it is not unseen. Its held-out frames
(`data/heldout/`) and frames reviewed since are unseen and are checked. A frame counts as seen when a listed
image's name is the frame's flat name as the dataset spells it (with each `s<N>-` source prefix the lists
use; the dataset's 120-byte cut included, so a name the cut made ambiguous leaves out every frame it may
be), and, for runs that record their image sources (`image_sources.json`, newer runs), when the source's
file name is the frame's flat name (exact whatever the dataset named the image). The strictest reading
available wins: a frame matched either way is skipped.

**Refused, never guessed.** A model without a training record (a stock, catalog or downloaded model:
what it trained on is unknown), a record that does not read or names no run, a run folder that is gone,
lists that do not read, or a source record that does not name a listed image: the check refuses with what
is missing, because a model judging frames it trained on finds every label "right". So is a selection
with no frame it never saw (`quick::MIN_UNSEEN_FRAMES` is 1: unlike the fold check's 40, which its models
need to learn from, a trained model can judge any frame it never saw; the preview says how few that is).

**What differs from a fold.** The model runs through the same predictor (`FoldPredictor` /
`LocalPredictor`, the benchmark's per-image pipeline) with **its own configured entry**: its family, class
file and device, but no confidence threshold of its own (per device or not) and no object filter
(`quick::predict_entry`), so it answers from 0.15 like a fold model (its own 0.5 would turn every faint
find into a `verify`). The judge's input size is the model's (its recipe's `imgsz`). The `verify` guard
(`MIN_CLASS_BOXES`, 50) counts what that model's run trained on (`dataset.json` `stats.train_boxes`,
mapped through the class list); without that file no label gets a `verify`. A wanted class the model does
not have (`collect.classes` it lacks, `classes_missing`; e.g. a model trained on 7 of 14 classes) gets no
`verify` (its silence on such a label is no miss) and no `relabel` (its detections of another class cannot
say the label is wrong), is not in the recall, and such a label covers a detection of any class (a model
names an object it cannot name by another class: no `add` on top of the label). Its detections of classes
off the list are not used, as a fold model's. Every `why` names the model (*custom-rf-detr-nano-3 (never
trained on this frame) found a person here, score 0.82*), never "a model that never saw this camera-day".
The static spots come from every reviewed frame of the selection, the ones it trained on included: the
spots are the reviewer's labels, which the model's training does not concern. The size floor applies as
configured. **Only the frames it checks are written**: the frames it trained or validated on are never
passed to the writer, so their suggestions, open or answered, stay exactly as they are; the replace count
and question (`open_to_replace`, the preview, the 409 of `POST /v1/refine/check`) cover only the frames it
checks. The closer look runs on its loose candidates after the pass, when on (it needs a GPU: asked for
without one, the check is refused; on by default without one, it is off with a warning).

**What it cannot tell you.** A frame it never saw may be of a camera day it trained or validated on (static
cameras: near copies of what it learned), so it finds those labels more easily than a fold model blind to
the whole day; the preview and the report warn with the count (`same_day`). It is one model at its own
input size, not short models of the recipe, so its recall measures that model on new frames, and its
`verify` and `relabel` come from a model that learned from your labels, mistakes included. Use the full
check for the frames every model has seen; the quick check is for what was reviewed since.

**The run.** Steps: *selecting the reviewed frames* (the model's frames read again: a run folder gone since
the start fails it), *checking with <model>* (the pass, then the judge and the writes), the closer look
when on, *writing the report*. It takes the training lock after the selection, as a fold check. A Cancel
during the pass writes nothing (a plain cancel); a failed pass fails the check with nothing written; a
Cancel during the closer look keeps what the pass wrote. Pause holds the closer look's helper only. The
title is *quick dataset check with <model> (N frames it never saw, <filter>)*.

**The report** replaces `report.json` like any check (the tab shows the last check), with `mode: model`,
`model` and `folds` / `folds_done` 0; its counts and recall tables cover only the frames it checked, and the
Findings and Coverage cards name the model and say *N frames <model> never saw*.

**Preview** (`--dry-run`, `GET /v1/refine/preview?model=`): the frames it would check and the camera days,
the frames skipped as trained and validated on, the model, its run, recipe, input size and configured
device, whether frames were matched by source or by name only, the replace count of those frames, the
same-day, missing-class, few-boxes and closer-look warnings, and a time: what the last quick check with
the same model measured per frame (`time_basis: last_check`), else a documented guess (`time_basis:
guess`): 60 s (loading and compiling the model) + 0.25 s per frame (decode and an RF-DETR nano at 384 on
an integrated GPU, about 0.11 s, plus the judge and the write; a discrete GPU is several times faster),
plus the closer look's own bound. Not measured.

On the copy of the real dataset (November 2026, read-only preview): `custom-rf-detr-nano-3` (run
`20261111-020504-cad6`, no image source record, so by name) skips 564 frames it trained on and 83 it
validated on and would check 204 frames on 19 camera days (its 169 held-out frames and 35 others), 34 of
them of camera days it trained on; it lacks 7 of the 14 classes (raccoon, opossum, skunk, fox, deer,
squirrel, bird) and trained on fewer than 50 bicycles, cats, dogs and motorcycles, so only persons, cars and
trucks can get a `verify`.

```sh
blue-onyx-prism train refine --with-model <name> [--dry-run] [--frames "scene=ir,camera=patio"] \
    [--gpu cuda:N] [--allow-large] [--tighten | --no-tighten] [--verbose]
```

`--with-model` conflicts with `--folds` and `--epochs` (it trains nothing).

### Refine tab

The Train area's Refine tab (`/train/refine`, between Review and Dataset) starts the dataset check from the
web UI and shows its results; the check itself is the one above and changes no box.

- **Check the dataset** card: *How to check*: *Full check (fold models, about an hour)* (the default) or
  *Quick check with a model you trained (only frames it never saw)* ([quick check](#quick-check-with-a-model);
  `check_mode`). The quick check's *Model* select lists the configured models with a training record whose
  run folder and lists are there (`GET /v1/refine` `capability.models`: `{name, run, recipe, imgsz,
  enabled}`, newest first; `capability.unusable` names the ones with a record it cannot use, with why, and
  the note under the select says so); how many reviewed frames each never saw is counted on *Preview* (a walk
  of the label files). In quick mode folds and epochs are hidden and not sent (`model` is: `?model=` on the
  preview, `"model"` in the start; with folds or epochs it is a 400), *Start* needs a model but no GPU
  (`capability.can_check` is the full check's), and the preview names the model, its run and the frames it
  skips. Then folds (2 to 8), epochs (1 to 50), a scene and a camera filter, the GPU when
  there are several, *Propose better boxes for loose labels* (the [closer look](#closer-look-at-loose-labels-tighten);
  ticked when RF-DETR L's weights are downloaded, `GET /v1/refine` `capability.tighten_default`; the page
  sends it as `tighten` with the preview, `tighten=1|0`, and the start, `"tighten": true|false`) and *Allow
  downloads over 500 MB*. *Preview* is `train refine --dry-run`
  (`GET /v1/refine/preview`: frames, what the export leaves out, camera days, each fold's size, the model,
  a rough time, warnings, or why a start would be refused). *Start* (`POST /v1/refine/check`) plans the check,
  gives it the running service's runtimes for the fold models' predictions (no second OpenVINO / ONNX Runtime
  instance) and queues it on the one-at-a-time queue. It refuses, in the check's own words, when too few
  reviewed frames can be checked, the folder under review is unplugged (503, `folder_missing`), a large
  download is not allowed, there is no GPU to train on, or a check is already queued or running (409). A
  start over open suggestions asks first (409 `needs_confirm`, the inline *Start and replace N open
  suggestions* button; see "What a rerun replaces" above). While
  it runs the card shows the check's progress row (step, percent, time left, Pause / Resume / Cancel; the
  Jobs tab links to it as `#job-<id>`).
- **Findings**: the open suggestions per kind (*Missing boxes* = `add`, *Labels no model found* = `verify`,
  *Other class suggested* = `relabel`, *Loose boxes* = `tighten`; `GET /v1/refine` `open`), counted over the label files of the folder under review
  (`refine::live`: answered and stale ones are not counted; kept 60 s, dropped when a check starts or ends),
  each with a link to the Review tab (`/train/review?auto=refine&refine_kind=<kind>`). Also the last report's
  date, folds done of folds (a quick check: *quick check with <model> (training run <id>). N frames it never
  saw checked; M frames it trained on and V it validated on skipped*, `RefineFloor.lastLine`; the Coverage
  card's recall then reads *over the N frames <model> never saw*), `cancelled`, its warnings and *Loose labels counted: N, M with a proposed box*
  (`loose`, `found.tighten`; *N (no closer look in this check)* without one). On the Review tab a `tighten` is
  *the tighter box* when the proposal lies (95% or more of it) inside the label, else *the better-fitting box*
  (larger or shifted; `ReviewRules.tightenWord`); the answers stay accept / *Keep my box*. Without a report the card
  says no check has run; a damaged or newer-format `report.json` is shown as a problem, never as an error page.
- **Static objects** (between Findings and Coverage; `server/refine_spots.rs`): the [static spots](#static-spots)
  of the reviewed frames of the folder under review (`statics::overview`; kept 60 s, dropped when a check starts
  or ends and after a bulk accept), the busiest 200 first (most missing boxes, then most frames). Per spot: the
  camera, what (the label), the place (the newest frame labelled there, `/dataset/image`, cropped around the box
  with the box drawn over it by the page), frames labelled there, days, and *Missing here* = the dataset check's
  live `add` suggestions that lie at the spot now (`statics::Table::spot_of_add`: the suggestion's box and
  class against the card's spots, as `statics::spot_at` places a label). The `spot` id a check stored on a
  suggestion is that check's evidence only: the check made its spots from its own selection (maybe scene
  filtered) and the labels as they were then, so its ids can differ from the card's (the 2% rounding, a `~2`
  suffix); an `add` with another id at this place counts here, one with this id that lies elsewhere does
  not. Actions: *Review* (`/train/review?auto=refine&refine_kind=add&camera=<camera>`)
  and, with missing boxes, *Accept the N at this spot...*.
  - **Accept at a spot** (`POST /v1/refine/accept-spot` `{spot, min_score?, dry_run?, limit?}`, 4 KiB, unknown
    fields refused): every live `add` suggestion lying at that spot (by geometry, as *Missing here*) scoring
    `min_score` (default 0.5, 0..=1) or more on the reviewed frames, each through `labels::update` +
    `refine::accept` (never around it: it replaces unjudged rejects, refuses a stale suggestion, keeps the frame
    reviewed; the new box is `source: auto`, accepted, `reason: "refine"`; nothing else in the file changes). A
    frame a reviewer confirmed empty gets nothing in bulk (`accept` would end its "empty", and about 1 in 10 such
    suggestions is wrong): it is counted in `left_empty` and left for one-by-one review, where a reviewer looks
    at it; the card's question says *N frames confirmed empty are left for one-by-one review*. At most `limit` frames (default and
    maximum 500) are written per call; a rerun takes the rest. `dry_run` runs the same code on the files as read
    and writes nothing, so on an unchanged folder its `accepted` and `frames` are what the apply does, and a
    second apply accepts 0. The card asks first, inline: the dry run's numbers, "about 1 in 10 such suggestions is
    wrong", then *Add them* / *Cancel*; after the write it lists up to 10 frames spread over the days under *Check
    these*, each opening that frame in Review (`/train/review?status=any&source=all&camera=<camera>&frame=<rel>`).
    Answer: `{success, dry_run, spot, min_score, accepted, frames, eligible, below_score, left_empty, stale,
    failed, failed_count, sample, message}`: `below_score` = live adds at the spot under `min_score` (or no longer
    fitting once the others were added), left for one-by-one review; `left_empty` = frames confirmed empty
    with a live add at the spot (the dry run counts them the same); `eligible` = frames with something to
    add, also past the limit; `stale` = frames that changed between the look and the write (passed over);
    `failed` = the first 20 `<frame>: <why>`. A spot id the reviewed frames do not make is 404 (a label change
    can move or drop a spot: reload the card), an unplugged folder 503 `folder_missing`, a bad field 400.
  - **Training cap**: the current `training.static_cap` (0 = off, 0..=100000) in a number field with *Save*
    (`POST /v1/refine/static-cap` `{cap}`, through the config's one writer like the other settings routes).
    Applies to the next training run; the pre-flight on the Train tab shows what it leaves out (its "Left out of
    training" list carries the static-cap line, once).
- **Too small** (between Static objects and Coverage; `server/refine_floor.rs`): the size floor
  (`collect.size_floor` `{px, input}`: a box whose short side at the model input is under `px` pixels is too
  small; default off) and the ignore mode (`training.ignore`). Objects below about one patch of the model's
  input cannot be learned, so with a floor they are left out of the labels at export (accepted and added
  labels are never rewritten) and a detection on one counts neither for nor against a model on held-out
  frames. The one rule here that writes only hides *pending proposals* under the floor on unreviewed
  frames, and those can be restored. The card shows, top to bottom:
  - the last check's **recall by size** (the report's `recall_by_size` and its input size; it lives here, not on
    Coverage, which points to it), with how it was measured: "Measured with fold models trained on every
    label; the check treated labels under N px at <input> as ignore regions (counted in this table only)." (or
    "no size floor was set."); a report of an earlier check says its fold models may have trained under the
    floor then saved;
  - the **floor form**: `px` (1..=64) and `input` (64..=4096), prefilled with the saved floor or, when off, the
    suggestion (10 px at the last check's input size, else the newest finished training run's, else 384).
    *Preview* asks `GET /v1/refine/floor?px=&input=` for the form's values and shows, per class, the boxes,
    those under the floor and what is left, with a sentence where a floor leaves a class nothing ("every
    motorcycle label would be left out: the model cannot learn the class from these frames") or under 50
    labels; a class the floor does not touch is never flagged. *Save* and *Turn off* (shown when one is
    saved) post `POST /v1/refine/size-floor`. Next to them: "Applies to the next export. Held-out scores from
    before and after a change do not compare, because the held-out labels change.";
  - with a floor saved and pending proposals under it, *Hide N pending proposals under the floor...*: the rule
    `POST /v1/refine/resolve-small`, asked inline like the static spots (a dry run, "Hide N pending
    auto-label proposals on M unreviewed frames as under the floor. They are set aside unseen like the
    auto-labelers' hidden boxes: no label, no negative, not counted as your decisions. i on the Review tab
    shows them, and they can be restored here if the floor goes down or off.", then the write);
  - with hidden proposals the saved floor (or none, when off) no longer covers, *Restore N proposals the floor
    no longer covers...*: the same route with `restore: true`, asked the same way ("Restore N hidden
    proposals on M unreviewed frames that the size floor no longer covers: they become pending again, for you
    to decide on the Review tab.");
  - the **ignore mode**: *Keep the frame, train the object as background* (default) or *Leave the frame out of
    training*, with *Save* (`POST /v1/refine/ignore-mode`); see "Ignore regions". The Train tab's pre-flight
    shows its effect: frames trained with an unlabelled object (`ignore_kept`, per class), frames left out
    for nothing but ignore regions (`ignore_only`) or because their camera day validates
    (`ignore_valid_day`), or, in `drop-frame`, every frame with a region (`ignore_frames`), and the size floor
    regions given back at the run's input (`floor_restored`).
  - Routes (admin, HTTPS, same-site; POSTs take 4 KiB JSON, unknown fields refused):
    - `GET /v1/refine/floor[?px=&input=]` answers `{success, folder_missing, message?, saved: {px,input}|null,
      shown: {px,input}|null, suggested: {px, input, from: "last_check"|"last_run"|"default"}, bounds:
      {px_min, px_max, input_min, input_max}, counts: {frames, frames_with, frames_only, per_class: [{class,
      boxes, under, left}], warnings}|null, pending: {boxes, frames}, restorable: {boxes, frames},
      ignore_mode}`. `shown` is the query's floor, else the saved one (no query); both `px` and `input` or
      neither, checked like a saved floor (400). `counts` (`export::floor_counts` over the reviewed frames) and
      `pending` (the rule's dry run for the shown floor) are null / zero with no floor shown; `restorable` is the
      restore's dry run with the **saved** floor (every size floor dismissal when the floor is off). Off the
      async workers; kept 60 s per folder, class list, shown and saved floor (at most 8), dropped by a floor
      save and by the rule's apply.
    - `POST /v1/refine/size-floor` `{px, input}` or `{off: true}` answers `{success, message, size_floor}`
      (null when off); 400 out of bounds or for both / neither.
    - `POST /v1/refine/ignore-mode` `{mode: "background" | "drop-frame"}` answers `{success, message, mode}`.
    - `POST /v1/refine/resolve-small` `{dry_run?, limit?, restore?}` (`limit` 1..=500, default 500) runs
      `bulk::bulk_resolve_small` with the **saved** floor (400 when none is saved; 503 `folder_missing`) over
      the unreviewed frames of the folder under review (the queue and the index files): pending model and
      auto-labeler boxes under the floor are dismissed (`state: rejected`, `reason: size_floor`, no
      `reject_reason`), exactly like the labelers' `auto_dismissed` boxes: no ignore region, no hard negative,
      not in the repeats history, no reviewer's decision, hidden on the Review tab until `i` (`auto=dismissed`
      finds them), and a frame holding one is never bulk-accepted (`auto=accept`) or repeat-only. Human,
      moved, unsure and audit boxes, reviewed and skipped frames are never touched and a frame keeps its
      status. With `restore: true` (no saved floor needed) it runs `bulk::bulk_restore_small`: the size floor
      dismissals on unreviewed frames that the saved floor no longer covers (all of them when it is off) go
      back to pending, their reason gone; a reviewed frame keeps its dismissals (it holds no pending box).
      Answer `{success, dry_run, restore, boxes, frames, eligible, failed, failed_count, message}`: a dry
      run's `boxes` and `frames` cover every eligible frame, so on an unchanged folder they are what an apply
      with a large enough limit does, and a second apply finds 0.
  - A later labelers run that finds a hidden proposal again brings it back only when the floor in force does
    not cover what it found (`triage::undismiss_found` with the frame's size; a find still under the floor
    stays hidden, covered by the dismissal); a `--missing-only` run on a reviewed frame overwrites it in place
    like any unjudged reject. Retention treats a hidden proposal like the other dismissals: an unreviewed
    frame with nothing but dismissals and no reviewer's decision can be pruned.
  - On the **Review tab** a box under the saved floor shows a *too small* chip in the box list and on its tag
    and, when selected, "Under the size floor (N px at <input>): left out of the labels at export." It is
    computed live in the page (`ReviewRules.floorNote` on `underFloor`, from the frame's size, so a resize or
    a box being drawn follows), from `size_floor` in the labels answer (`GET|POST /v1/review/labels`, null =
    off: no chip), read with every frame. A rejected or unsure box (already an ignore region) shows none. It
    changes no box and saves nothing, and it speaks of the saved floor: an export made before it was saved
    (or with another) still holds the box as a label until exported again (the training pre-flight names
    such exports). A hidden proposal reads "hidden: under the size floor".
  - Not updated by a floor change until the next check: the dataset check's suggestions (`refine::live` knows
    no frame size, so one made before a floor was set stays until a check replaces it). The Static objects
    card counts boxes under the floor in its spots, while the training static cap (on the export's YOLO
    boxes) does not, so the two can differ; and an export's "parked only" frames are judged on the boxes left
    after the floor (a parked car with a tiny accepted person beside it is parked only there).
- **Coverage**: the report's recall per camera, scene and class, each with its 95% interval (the per-size table is
  on the Too small card).

`GET /v1/refine` carries all of it for the page (`review_folder`, `folder_missing`, `report`, `report_problem`,
`open`, `job`, `last`, `capability`, `limits`). `GET /v1/refine/spots` answers the Static objects card
`{success, folder_missing, message?, spots: [{id, camera, object, label, bbox, frames, days, stay_days, scenes,
by_day, labelled, open_adds, crop, sample, sample_size}], truncated, totals: {spots, labelled_boxes, open_adds},
static_cap, static_cap_default, static_cap_max}` (`bbox` and `crop` normalised; `sample` is the newest labelled
frame's rel path, `sample_size` its pixels). The tab has no card for tighter boxes yet.

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
    [--static-cap 20 (0 = off)] (--out <dir> | --dry-run)
blue-onyx-prism train run --model yolo26n [--model rfdetr-nano ...] --dataset <export> \
    [--scene all|day|night|ir|night+ir|day+ir ... (repeat for a comparison)] \
    [--classes a,b] [--classes c] [--excluded background|drop-frame] \
    [--balance class,scene,camera,camera-day|off] [--balance-strength 0.5] [--balance-max-repeat 4] [--balance-cap 3] \
    [--static-cap 20 (0 = off)] [--epochs 100] [--patience 15 (0 = off)] [--batch 16] [--imgsz 640] [--lr 0.001] [--lr-schedule cosine|linear|step] [--lr-min-factor 0.05] [--select-metric map|map50|f1|f2|f0_5|iou] [--night-aug] [--aug-dark p --aug-noise p --aug-gray p] \
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
blue-onyx-prism train autolabel --sam3 [--dry-run] [...]   # SAM 3 in MM-Grounding-DINO's seat; --dry-run only previews
blue-onyx-prism hf-login | hf-logout      # Hugging Face token for the SAM 3 download
blue-onyx-prism train overlaps [--frames "scene=ir,camera=patio,limit=200,again=1"] [--gpu cuda:0] \
    [--allow-large] [--dry-run] [--verbose]    # suggestions for overlapping boxes; changes no box
blue-onyx-prism train refine [--dry-run] [--folds 4] [--epochs 8] [--frames "scene=ir,camera=patio"] \
    [--gpu cuda:N] [--allow-large] [--verbose]    # dataset check by fold models; changes no box
blue-onyx-prism train refine --with-model <name> [--dry-run] [--frames ...] [--no-tighten]   # quick check:
    # a model trained here judges the reviewed frames it never trained or validated on; no training
blue-onyx-prism train autolabel --with-model <name|local:<file>|resource:<id>> [--threshold 0.3] \
    [--device <spec>] [--accept-score-model 0.8] [--missing-only] [--confirm-license] [--allow-large] \
    [--frames "status=any,..."]
blue-onyx-prism train list
blue-onyx-prism train retry-export <run> [--no-register] [--no-benchmark]    # a failed run's best checkpoint -> the model
blue-onyx-prism train remove-env
```

`--epochs` and `--patience` default to the family's recommendation ([Starting points and
convergence](#starting-points-and-convergence)). A running training is finished early, keeping its best
checkpoint, by creating `<run dir>/work/stop-request` (or Finish now on the page). `--scene` is
repeatable: several scenes (or models x scenes) run one after another on the same held-out frames and the
comparison table is printed at the end. `night` alone is night frames only; use `night+ir` for both.
`--balance` takes axes comma-separated and/or repeated (no flag: `training.balance`, else off); the
strength, max repeat and cap flags change that default's values. See [Assembly](#assembly-srctrainingdatasetrs)
step 7. `--static-cap N` (`prepare`, `run`) overrides `training.static_cap` for this command (0 = off; see
the static cap in step 3). `prepare --dry-run` writes nothing and prints images per epoch before -> after, boxes and frames per
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
