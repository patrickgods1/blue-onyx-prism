# Dataset collector

The collector saves frames that Blue Iris sends, together with what the model saw and did. Later stages
use them: the review page, footage import, auto-labelling and training (plan, Phase 5). It is off by
default. Turn it on with `collect.enabled` or on the **Collect** tab (`/train/collect`). Every setting
applies at once; the README config table lists them.

**Privacy:** the frames contain whatever your cameras see. They stay on this machine, in the dataset
folder. Nothing is uploaded anywhere.

## How a frame is saved

1. **Record.** The worker finishes a request as usual. If collection is on, it fills in a small record:
   - the model, device, request id and camera key;
   - the scene and image statistics;
   - the detections before and after the rules;
   - what the verifier did.
2. **Hand-off.** After the reply has gone out, the worker passes the record and the upload's own
   encoded bytes (a reference, no copy, no re-encode) to a bounded queue of 16 frames.
   - A full queue drops the frame (`dataset_dropped_total{reason="queue_full"}`).
   - Neither the inference thread nor the reply ever waits for the collector.
3. **Collector thread.** One thread does the rest:
   - decides the buckets;
   - applies sampling and per-camera rate limits;
   - drops near duplicates;
   - writes the files;
   - keeps the index;
   - applies the size and age caps.
4. **Verified requests.** A request that goes through the verifier cascade is handed over by the verify
   lane, after its reply, with the verifier outcomes added.
5. **Not collected:** verifier checks (crops sent to the verifier model) and failed requests.

## Layout

```
<dir>/                                   collect.dir, default <data root>/dataset
  review-queue.jsonl                     frames sent to review (the review page consumes it)
  spots/<camera>.json                    parked spots of the camera (below, Review labels: Spots)
  refine/report.json                     report of the last dataset check, full or quick (`mode`; TRAINING.md, Dataset check)
  <scene>/                               day | night | ir (src/image_stats.rs)
    <camera>/                            camera folder (see below)
      <YYYY-MM-DD>/                      UTC date
        index.jsonl                      one line per frame of this folder
        <ts>_<request_id>.jpg            the frame
        <ts>_<request_id>.json           its sidecar
```

- **`<ts>`** is the UTC time in milliseconds: `20261022T123456789Z`. `<request_id>` is the request's id
  from the logs and the `X-Request-Id` header, limited to `[A-Za-z0-9_-]` and at most 32 characters.
- **`<camera>`** is the camera part of `static_memory::camera_key`: the upload's file name up to the
  first `.`, without trailing digits, `_` and `-`, lowercase. So `Driveway.20240101_120000.jpg` becomes
  `driveway`.
  - A generic name (`image.jpg`, `blob`, ...) becomes `frame-<w>x<h>`.
  - Only `[a-z0-9_-]` is kept, at most 64 characters.
  - Rate limits and dedup work per camera folder, across models.
- **The `.jpg`** is the upload byte for byte when it is a JPEG (`image.original: true`). Any other
  format is re-encoded as a quality-92 JPEG (`original: false`).
- **Writes** go to a temporary file in the same folder and are then renamed, in a fixed order: image
  first, then sidecar, then the index line.
  - A sidecar always has its image.
  - A frame counts as complete once its sidecar exists.
  - Temporary files of an interrupted write (`.<name>.<uuid>.tmp`) are deleted by retention after an
    hour.

## Buckets

A frame can match several buckets. It is stored once, and its sidecar lists the buckets it was kept
for. Every bucket has `enabled`, a sampling `rate` (0 to 1, deterministic per request id) and
`per_camera_min_interval_s`.

| Bucket | When | Default interval |
|---|---|---|
| `positive` | The reply has predictions. | 30 s |
| `empty` | `raw` is empty: nothing at or above the threshold. Boxes only below it still count as empty. | 600 s |
| `suppressed` | A rule removed a box: `night_threshold`, `label_threshold`, `aspect`, `min_area`, `overlap` or `static`. These are false-positive candidates, or misses when the rule was wrong. | 0 |
| `verify_rejected` | The verifier rejected a candidate. | 0 |
| `verify_kept` | The verifier confirmed a candidate. | 0 |
| `near_threshold` | For some label, the top score over `raw` (suppressed boxes included) and `below_threshold` is within `margin` (default 0.1) of the request's threshold: `abs(top - threshold) <= margin`. It is per label, so an uncertain dog next to a sure person still counts. | 0 |
| `disagreement` | The verifier rejected a candidate, or in mode `frame` reported a box at or above its `min_confidence` that matches no `raw` box (same label, IoU at or above its `iou`): a miss of the model. | 0 |

**What "disagreement" can and cannot measure:**

- With the night model switch (`night_model`), a night or IR frame runs on the night model only. The
  primary never sees it, so there is nothing to compare without a second inference, and none is spent.
  A forwarded frame (`routed_from` is set) can still become a `disagreement` through the night model's
  own verifier.
- Comparing the primary and the night model on the same frame is left to the offline stages.

**How a frame is kept or dropped:**

1. The frame's matching enabled buckets are collected in `matched`.
2. A bucket passes when its sampling and its per-camera interval (since the camera's last saved frame
   in that bucket) allow it. The passing buckets become `buckets`. If none passes, the frame is
   dropped (`reason="rate"`).
3. The frame is dropped as a near duplicate (`reason="dup"`) when its dHash is within `dedup_hamming`
   bits (default 6) of a frame of the same camera saved in the last 10 minutes (at most 32 frames are
   remembered).
   - dHash is 64 bits: the luma is box-averaged to 9 x 8 cells, and a bit is set when a cell is brighter
     than its right neighbour.
   - This check is only for frames kept for `positive` or `empty`. A frame kept for `near_threshold`,
     `suppressed`, `verify_rejected`, `verify_kept` or `disagreement` skips it, because a distant person
     changes only a few of the 64 bits and those are the frames review needs most. If such a frame also
     passed `positive` or `empty`, it keeps those buckets too.
   - **The limit on skipping:** at most 10 near duplicates per camera per minute are saved this way
     (`store::DEDUP_SKIPS_PER_CAMERA`, `DEDUP_SKIP_WINDOW`). Past that they are dropped as duplicates.
     These five buckets have an interval of 0 by default, so in a storm of look-alike frames this limit
     is what holds: about 14,400 frames a day per camera at most, and `collect.max_gb` still applies.
     Frames that do not look like a recent one are not limited by it, as before. To limit those too,
     set the bucket's interval.
4. The frame is saved, and the passing buckets' interval clocks restart.

**`below_threshold`:** with `near_threshold` on, postprocessing runs `margin` below the request's
threshold (never below the 0.01 floor). Boxes under the threshold are only recorded, never reported.
Greedy NMS and its caps keep the highest scores, so the reported boxes are exactly those of a run at
the threshold (a unit test checks this). At most the 20 highest are kept.

## Sidecar JSON (version 1)

```json
{
  "version": 1,
  "id": "20261022T123456789Z_ab12cd34",
  "request_id": "ab12cd34",
  "timestamp": "2026-10-22T12:34:56.789Z",
  "model": "IPcam-general",
  "device": "openvino:cpu",
  "routed_from": "primary-model",
  "image_name": "Driveway.20261022_123456.jpg",
  "camera": "driveway",
  "camera_key": "IPcam-general|cam:driveway",
  "scene": "ir",
  "stats": {"mean_luma": 61.2, "chroma_spread": 0.4, "tint_spread": 0.3, "is_ir": true, "is_dark": false},
  "width": 1920, "height": 1080,
  "image": {"file": "20261022T123456789Z_ab12cd34.jpg", "bytes": 412345, "original": true},
  "threshold": 0.5,
  "near_margin": 0.1,
  "raw": [{"label": "person", "score": 0.55, "box": [812.0, 400.5, 870.2, 560.0]}],
  "below_threshold": [{"label": "dog", "score": 0.43, "box": [100.0, 700.0, 180.0, 760.0]}],
  "suppressed": [{"label": "person", "score": 0.55, "box": [812.0, 400.5, 870.2, 560.0], "reason": "night_threshold"}],
  "predictions": [],
  "verify": {
    "verifier": "yolo26s", "mode": "frame", "failure": null,
    "candidates": [{"label": "person", "score": 0.7, "box": [10, 10, 50, 90], "outcome": "rejected"}],
    "found": [{"label": "car", "score": 0.61, "box": [600, 500, 900, 700]}],
    "min_confidence": 0.3, "iou": 0.3
  },
  "buckets": ["suppressed", "near_threshold"],
  "matched": ["suppressed", "near_threshold"],
  "dhash": "f0e1d2c3b4a59687"
}
```

**Optional fields:**

- `routed_from` is absent when the frame was not forwarded by another model.
- `near_margin` and `below_threshold` appear only with `near_threshold` on.
- `verify` appears only when the request went through the verifier. Its `failure` is set when
  candidates stayed unverified (`timeout`, `busy`, `not ready`, ...).
- `found` appears in frame mode only: crop-mode verifier boxes are in crop coordinates and are not
  kept.
- `source` and `provenance` appear on imported frames only (below).

**Detections:**

- Every box is `[x1, y1, x2, y2]` in pixels of the original frame, clamped to the frame and rounded to
  0.1 px. Scores are rounded to 4 decimals.
- `raw` holds the model's detections at or above the threshold, after NMS and before any rule or the
  object filter.
- `predictions` is exactly the reply: object filter applied, verifier rejections removed.
- `suppressed` lists every box a rule removed, plus the verifier's rejections (`reason: "verify"`).
- Labels are the model's class names (`class_<n>` for an id beyond the class list).

**Versioning:** `version` is bumped when a field changes meaning or is removed. Adding a field keeps
the version. Readers ignore unknown fields and skip records with a newer `version` than they know.

### Imported frames

Frames of a footage import (`train import`, docs/TRAINING.md "Footage import") use the same layout,
file names `<UTC ms>_imp-<import id>-<n>.jpg`, and a version-1 sidecar with no detections (`raw`,
`predictions`, `suppressed` empty, `buckets` empty, `model: "import"`, `threshold: 0`) plus:

```json
{
  "source": "import:20261023-055526-88d5",
  "provenance": {
    "import_id": "20261023-055526-88d5",
    "kind": "video",
    "path": "/footage/2018-03-07.17-00-00.17-05-00.school.G419.r13.avi",
    "license": "CC BY 4.0 (MEVA, mevadata.org)",
    "license_confirmed": true,
    "confirmed_at": "2026-10-23T05:41:02.115Z",
    "time_offset_s": 36.0
  }
}
```

- `kind`: `video`, `image`, `url` (with `url` and `site`: what yt-dlp reported, such as the title,
  uploader, page URL and the site's licence) or `dataset` (with `dataset`, the pinned resource, and
  `attribution`; nothing to confirm).
- `time_offset_s`: seconds from the start of the video (absent for an image). `timestamp` is the
  recording time when the file name has one (Blue Iris, MEVA), else the file's modification time.
  A Blue Iris name holds local time and is stored as if it were UTC, while collected frames use real
  UTC, so frames of the same hour from the two sources can land in different day folders.
- The camera folder is `--camera`, else the Blue Iris camera of the file name, `meva-<site>-<camera>`
  for MEVA, a URL's host, or `import`.
- The import appends its frames to `review-queue.jsonl` and one line to `<dir>/imports.jsonl` (id,
  time, source, kind, licence, attribution, frames, cameras, frames per scene), which the Collect tab
  lists. Imported frames have no bucket, so the scene x bucket table does not count them; the frame
  total and the camera table do.

### `index.jsonl`

One JSON object per line, appended when a frame is saved:

```json
{"id": "...", "timestamp": "...", "scene": "ir", "camera": "driveway", "model": "IPcam-general", "buckets": ["suppressed"], "labels": [], "bytes": 415000}
```

- `bytes` is the image plus the sidecar.
- When frames are deleted, retention rewrites the day's index atomically.
- At startup, the service rebuilds the Collect tab's summary (counts by scene x bucket, by detected
  class and per camera, bytes, the newest 8 frames per bucket) from the index files.

### Frames by detected class

The Collect tab's **Frames by detected class** table counts frames per class and scene (`by_label`
in `GET /v1/dataset`).

- The classes are the `labels` of the index line: the reply's predictions at collection time, not
  reviewed labels. Reviewed labels are counted on the Dataset tab and in the export report.
- A frame counts once per class, however many boxes of that class it has.
- A frame without a prediction counts under `(none)`.
- Rows are sorted by total frames, most first. `(none)` is last.

### `review-queue.jsonl`

The Collect tab's **Send to review** appends one line per ticked frame that is not queued yet:

```json
{"version": 1, "frame": "ir/driveway/2026-10-22/<stem>.jpg", "sidecar": "ir/driveway/2026-10-22/<stem>.json", "id": "<stem>", "buckets": ["suppressed"], "scene": "ir", "camera": "driveway", "queued_at": "2026-10-22T13:00:00.000Z"}
```

Paths are relative to the dataset folder, with `/` separators. The page and the HTTP routes accept
only `<day|night|ir>/<camera>/<YYYY-MM-DD>/<stem>.jpg` with those character sets. A path that does not
resolve inside the folder to a regular file (symlinks refused) is a 404.

## Disk

- **Caps:** `collect.max_gb` (default 5) and `collect.max_age_days` (default 90, 0 = keep) are applied
  by `retention::prune_dataset`, oldest first.
  - It runs on the collector thread, the only writer: every 10 minutes (asked by the retention sweeper,
    also while collection is off, as long as the folder exists) and right after a write that crossed
    the size cap.
  - It deletes only frames in the layout above: the image, the sidecar and the frame's `.labels.json`.
    It never deletes `review-queue.jsonl`, `spots/` or anything else. A frame in the review queue is
    pruned like any other.
- **What is protected:** a reviewed or skipped frame, a frame that was reviewed before a re-label sent it
  back, and a frame with a human decision on it are never deleted and do not count toward `max_gb`. A
  human decision is an accepted or rejected box, a drawn, moved or relabelled box, an unsure mark, or
  a confirmed empty frame.
  - A frame with boxes waiting for review is kept too, because an auto-label run takes hours. Such
    frames can take the folder past `max_gb`; the disk guard still protects the volume, and the
    Collect tab's disk line shows them.
  - A label file with nothing left to look at does not protect its frame. That covers an auto-label
    run that found nothing, boxes a labeler dismissed or accepted at a parked spot, and bulk rejects
    by score, by class or as a stacked duplicate.
  - A label file that cannot be read protects its frame, in case it holds a person's work.
  - The size cap applies to the frames that can be deleted. The Collect tab's *Disk used* line shows
    the total, the part that counts toward the cap and the protected reviewed data. The protected
    figure can lag a review by up to 10 minutes.
- **Low disk:** if free space under the folder drops below 1 GB (`retention::MIN_FREE_MB`, the same
  threshold as the low-space warning), or a write fails with a full disk or exhausted quota:
  - collection stops with a warning;
  - `dataset_collect_disabled` becomes 1 and frames count as `dataset_dropped_total{reason="disk"}`;
  - the Dataset and Home pages show the reason.
  - It starts again by itself once 1.25 GB is free (the stop point plus `store::REARM_MARGIN_MB`,
    256 MB), checked every 30 seconds whether or not frames arrive. **Resume** on the page or a change
    to the `collect` settings restart it sooner. If free space cannot be read, only those two do.
- **Other write errors** (permissions, a read-only folder) drop the frame (`reason="error"`) with a
  throttled warning, and show as the last error on the Collect tab.

## Is it working?

The Collect tab's Status card and `GET /v1/dataset` (`collection`) answer this, so an unwritable folder
does not look like a quiet camera. Counts start from zero when the service starts.

- **Requests seen:** detection requests received, collection on or off. Test page uploads do not count.
- **Frames saved:** frames written.
- **Frames not saved:** drops by reason in plain words (`dropped`: `reason`, `plain`, `count`), plus
  images that matched no bucket that is on (`no_bucket`).
- **Last frame:** the time of the last saved frame, or the newest in the folder after a restart.
- **Last error:** the last failure with its time. It stays after later frames save, so check the time.
- **A sentence** (`note`) when collection is on and nothing was saved: "Blue Iris has not sent any images
  since this service started" with no requests at all; otherwise how many images arrived and the likely
  reason.

Home shows one line: collecting or not, the frames so far, a link to this page, and the stop reason
while collection is stopped.

**Test page uploads** are never collected, so a test image `dog.jpg` does not become a camera `dog`. The
page's script sends the header `x-prism-no-collect` to the API routes, and the form path `POST /test`
marks the request itself.

## Metrics

| Metric | Meaning |
|---|---|
| `blue_onyx_prism_dataset_requests_total` | Detection requests received, Test page uploads excluded. |
| `blue_onyx_prism_dataset_offered_total` | Answered requests handed to the collector while it was on. |
| `blue_onyx_prism_dataset_saved_total` | Frames saved (once each). |
| `blue_onyx_prism_dataset_no_bucket_total` | Frames that matched no enabled bucket. |
| `blue_onyx_prism_dataset_frames_total{bucket}` | Frames saved per bucket (a frame counts in each bucket it was kept for). |
| `blue_onyx_prism_dataset_dropped_total{reason}` | `queue_full`, `rate`, `dup`, `disk`, `error`. |
| `blue_onyx_prism_dataset_bytes` | Image and sidecar bytes in the folder (from the index). |
| `blue_onyx_prism_dataset_collect_disabled` | 1 while collection is stopped for disk space. |

## How the later stages use it

- **Review tab** (`/train/review`, see [Review labels](#review-labels)):
  - reads `review-queue.jsonl` (or every indexed frame) and sorts it by uncertainty;
  - shows the `predictions`, `suppressed`, `below_threshold` and `verify.found` boxes from the sidecar;
  - writes `<stem>.labels.json` beside the frame; exports YOLO / COCO / X-AnyLabeling / a benchmark set.
  - Rejected boxes are kept as hard-negative evidence. Frames confirmed empty become negative images.
- **Auto-label** (offline, training env):
  - re-detects `empty` frames with the large labellers; a confident box is a candidate miss;
  - uses `below_threshold` and `verify.found` as cheap hints;
  - adds its boxes (`source: auto`) to the label files through the merge rule below.
- **Dataset assembly** (the export does the first part):
  - splits by whole camera days (camera and date folder), so frames of one camera on one day never sit
    on both sides; other days of a held-out camera can train unless the export names that camera. An
    export of a single camera day has no `val` split, so a run holds out a share of its frames, with a
    warning that frames of one day are similar;
  - uses a frame once when several ticked exports hold it (by the flat image name; the held-out copy
    wins);
  - balances by scene, class, camera or camera day only when asked (off by default,
    [Assembly](TRAINING.md#assembly-srctrainingdatasetrs) step 7).

  `dhash` is not used here: the collector and the import drop near duplicates when frames are saved,
  each within its own window ([Imported frames](#imported-frames)), and nothing compares frames across
  days when a dataset is assembled.
- **Import:** imported footage uses the same layout and sidecar, with `source` and `provenance`
  added ([Imported frames](#imported-frames)).

## Review labels

The **Review** tab (`/train/review`, admin only like the rest of the Train area) shows one frame at a time with its boxes.
Each decision is saved in `<stem>.labels.json` beside the sidecar. The sidecar is never changed.

The page works on the dataset folder, or on another folder in the same layout (an import saved with
*Save frames to* / `train import --out-dir`): *Folder under review* (a native folder picker, `POST
/v1/review/folder` with `{dir}`, empty = back to the dataset folder) saves it as `collect.review_dir`.
The queue, the label saves, the exports, the X-AnyLabeling import and auto-label runs then use that
folder; the collector and the Collect tab keep `collect.dir`.

### Label file (version 1)

```json
{
  "version": 1,
  "status": "reviewed",
  "empty_confirmed": false,
  "boxes": [
    {"label": "person", "bbox": [812.0, 400.5, 870.2, 560.0], "source": "model", "score": 0.55,
     "state": "accepted", "origin": "suppressed", "reason": "night_threshold"},
    {"label": "raccoon", "bbox": [100.0, 700.0, 180.0, 760.0], "source": "model", "score": 0.43,
     "state": "accepted", "origin": "below_threshold", "model_label": "cat"},
    {"label": "dog", "bbox": [10.0, 10.0, 50.0, 90.0], "source": "model", "score": 0.71,
     "state": "rejected", "origin": "prediction"},
    {"label": "deer", "bbox": [600.0, 500.0, 900.0, 700.0], "source": "human", "state": "added"}
  ],
  "reviewed_at": "2026-10-22T13:05:00.000Z",
  "classes_version": "3f2a9c0d1e4b5a68"
}
```

- **`status`:** `unreviewed`, `reviewed` (every box decided, exported) or `skipped` (never exported).
- **`bbox`:** `[x0, y0, x1, y1]` in pixels of the original frame. Boxes are ordered and clamped to the
  frame on save, and both sides must be at least 1 px.
- **`source`:** `model` (from the sidecar), `auto` (the auto-labeler) or `human` (drawn on the page).
- **`state`:**
  - `accepted`: a model or auto box that is a real object;
  - `added`: a box the reviewer drew (a miss);
  - `rejected`: a wrong box. It is kept as hard-negative evidence and never becomes a training label;
    a pending auto box rejected in bulk for its score has `reason: "low_score"`, a box rejected for an
    unwanted class (outside the list, or in `collect.hidden_classes`) `reason: "other_class"`, a box a
    labelers run set aside for its score `reason: "auto_dismissed"`, a box it set aside at a spot you
    rejected `reason: "spot_dismissed"`, a pending proposal the size floor rule set aside
    `reason: "size_floor"` (below);
  - `pending`: not decided yet. Only `unreviewed` and `skipped` files have pending boxes.
- **`origin`** (model boxes): `prediction` (the reply), `suppressed` (with the rule in `reason`),
  `below_threshold`, or `verify_found` (a box the frame-mode verifier found that matches nothing else).
- **`model_label`:** the label the model gave, when the reviewer changed it or it was mapped onto the
  class list's spelling (pre-fill, auto-label runs, *Reject boxes of unwanted classes*).
- **`orig_bbox`:** the proposed box before the reviewer moved or resized it (set by the page; checked
  and clamped like `bbox`, dropped when equal to it).
- **`difficult`** (written only when true): the reviewer marked the box *unsure* (Review tab `d`), a
  real object they cannot name for sure (PASCAL VOC `difficult`). It is an [ignore region](#ignore-regions):
  left out of the YOLO labels and listed in `ignore.json`, `ignore` in the benchmark manifest. It carries
  `"attributes": {"difficult": true}` in COCO and `difficult` in X-AnyLabeling (both ways). Cleared on a
  rejected box.
- **`reject_reason`** (optional, rejected boxes only): why the reviewer rejected it (Review tab,
  Shift+1-9 after `r`): `false_positive`, `duplicate`, `insect_web`, `rain_snow`, `ir_glare`,
  `reflection`, `static_object`, `other_object`, `too_small`. Absent = a plain false positive; a
  value this build does not know (a newer build's) loads and is written back unchanged (the API refuses
  one it is sent). Separate from `reason`, which is provenance. Copied to `hard_negatives.json`, which
  leaves out `duplicate` and `too_small` boxes (they sit on real objects). `too_small` (the *Too small*
  button of the decision card, or its reason chip) also makes the box an
  [ignore region](#ignore-regions), and like `duplicate` it says nothing about the spot, so the repeats
  history and "rejected here N times" skip it. Frames reviewed earlier follow this without a rewrite.
- **`reason: "auto_dismissed"`** (a *hidden* box on the Review tab): a labelers run found the box at a
  fused score below 0.15 on a frame nobody had reviewed. Those boxes were 5% real on public sets and 0% on
  a fixed camera, so they do not take a reviewer's time ([docs/TRAINING.md](TRAINING.md#likely-unlikely-and-hidden-boxes)).
  The box is `rejected` so that it is no label and no live box, but nobody judged it: it is left out of
  `hard_negatives.json` and the repeats history, and a later run may replace it. Nothing is lost. Press
  `i` or *Show N hidden* on the page, or filter by `auto=dismissed`; accepting the box clears the reason, and
  so does rejecting it by hand, which makes it an ordinary negative. A frame with such a box is never
  saved in bulk (`auto=accept`, `auto=repeats`): a saved frame claims it is complete. A reviewer who saves it
  has been told how many are hidden, and a real object among them then counts as background, like any
  object left unlabelled.
- **`reason: "spot_dismissed"`** (a *hidden* box too): a labelers run found the box at a spot you rejected
  on at least 2 frames ([docs/TRAINING.md](TRAINING.md#spot-memory-of-fixed-cameras)). Every reader treats
  it exactly like `auto_dismissed`: no hard negative, not in the repeats history, `auto=dismissed` finds
  it, a later run may replace it, no bulk save takes its frame. `spot_rejections` holds the frames rejected
  there when the run looked, for the page's *hidden: you rejected this spot N times*.
- **`reason: "size_floor"`** (a *hidden* box too, *hidden: under the size floor*): the Refine tab's *Too
  small* rule set aside a pending model or auto proposal under the saved [size floor](#size-floor) on an
  unreviewed frame. Every reader treats it exactly like `auto_dismissed`: no ignore region (nobody looked,
  so it is not known to be a real object), no hard negative, not in the repeats history, no reviewer's
  decision, `auto=dismissed` finds it, no bulk save takes its frame; accepting or rejecting it by hand
  clears the reason (a reject with *Too small* then makes it an ordinary ignore region). The rule's
  *Restore* puts those the saved floor no longer covers (all of them when it is off) back to pending on
  unreviewed frames. A later labelers run brings one back only when what it finds is not under the floor in
  force. Like the other dismissals it keeps no frame from retention: an unreviewed frame with nothing but
  dismissals and no reviewer's decision ages out.
- **`reason: "parked_spot"`** with **`parked`** (`{since, bbox}`, the spot's date and own box): a labelers
  run accepted the box at a parked spot. It stays out of the repeats history, because a rule made the
  decision. The frame stays `unreviewed`, with `repeat_spot` set. Rejecting the box clears both fields,
  and then it is a look like any other.
- **`likely`** (written only when true): a labelers run's box with a fused score of 0.25 or more and a
  yes to the VLM's yes/no question, 77% real on public sets and 96% on a fixed camera. It is listed first,
  and the frame comes first in the queue. `triage` stays `review`. Boxes of older runs have no `likely`,
  and a seven-way VLM answer never gives one.
- **`audit`** (written only when true): one of the boxes that would have been dismissed, about 2%, left
  pending with the reason *audit sample: normally hidden*, so that the filter can be checked. A hash of
  the frame, the label and the whole-pixel box picks them, so a rerun picks the same boxes.
- **`empty_confirmed`:** the frame holds no object of any class. It is a hard-negative image, and it
  cannot have accepted or added boxes.
- **`classes_version`:** the first 16 hex digits of the SHA-256 of the class list (`collect.classes`,
  names joined by `\n`). The page flags a frame that was reviewed with another list.
- **`disagreement`:** set by a re-label with a trained model (`train autolabel --with-model`) when its
  detections disagree with the reviewed labels (`"re-label by night-yolo26n: 1 new box, 1 missed"`), or
  by a `--missing-only` run that found an object with no box (`"re-label by a, b: 2 new boxes"`); the
  file goes back to `unreviewed` and the Review queue lists it first. Cleared when the frame is saved
  as reviewed or skipped. At most 500 characters.
- **`auto_run`:** `{"at": "<RFC 3339>", "labeler": "mm-gdino-tiny + rfdetr-large + ..."}`, the last
  labelers-mode auto-label run that looked at the frame, boxes found or not (a frame with nothing found
  gets a label file with just this). A later run of the same labelers skips the frame unless its filter
  has `again=1`. The page round-trips it.
- **`bulk_accepted`:** `true` when the file was saved by a bulk accept (`POST /v1/review/accept-auto`,
  `accept-repeats`) rather than by a reviewer looking at the frame. Any other save clears it.
- **`repeat_spot`** (written only when true): a spot edit ("Reject / Accept this spot on all frames",
  `POST /v1/review/reject-spot` / `accept-spot`) changed boxes of this unreviewed frame. The boxes it
  rejected or accepted carry `reason: repeat_spot` when they had no provenance. With no pending box left
  the frame is *repeat-only*: last in the queue, `auto=repeats`, saved in bulk by `POST
  /v1/review/accept-repeats`. Cleared once the frame is reviewed or skipped.
- **`overlaps_checked`** (written only when true): the reviewer saved the frame as reviewed with stacked
  or overlapping boxes still on it, so they looked and kept them (two real nested cars). `auto=overlaps`
  passes the frame over and `POST /v1/review/reject-stacked` leaves it alone. Kept by saves; cleared when
  a merge adds a box. The Review page decides it on every reviewed save: true when a stack or an
  overlapping pair is left, whatever made it on that visit (a box drawn, moved or relabelled by hand, or
  an accepted `refine` suggestion, whose accept alone ends the flag on the file). That is sound because
  the save is the look: the page marks every pair it holds (the `dup` / `overlap` chips, "Overlap: a+b" in
  the progress line, "Overlaps box N" with `u` on the selected box), and the save's check pass steps
  through every decided box not looked at on that visit, so the partner of a new pair is shown before the
  save unless the reviewer looked at it already or saves again past the check.
- **`checked_before_reopen`** (written only when true): a rerun's new box took `overlaps_checked` off this
  frame. `POST /v1/review/restore-reopened` gives it back; a restore, a save or an accept-spot of a new box
  clears this field.
- **`overlap_suggestions`** (written only when not empty, at most 64): what the overlap check
  (`train overlaps`, [Overlap suggestions](TRAINING.md#overlap-suggestions)) suggests for overlapping
  pairs. Suggestions only: they change no box, status or `reviewed_at`.

  ```json
  {"a": [812.0, 314.6, 2589.1, 1507.5], "b": [857.0, 314.0, 1828.0, 996.0], "verdict": "drop_b",
   "why": "A closer look found only the kept box (0.61): 2 detectors (gdino, rfdetr) found it, 1 detector (serving) found the dropped one",
   "kept_score": 0.61, "other_score": 0.19}
  ```

  `a`, `b`: the two boxes by `bbox`, the lower first. That is the key: a suggestion whose boxes moved or
  went, or whose pair is no longer open, is stale and ignored, and so is any suggestion on a skipped or
  `overlaps_checked` frame. `verdict`: `drop_a`, `drop_b`, `two_objects`, or `unclear` (looked at, no
  suggestion: the page shows nothing, a rerun passes it over; an unknown value loads and is ignored).
  `kept_score`: the zoom's score on the kept box; `other_score`: the best other detection on the box to
  drop, 0 if none (`two_objects`: the zoom's scores on `a` and `b`). `low_confidence`: the zoom gave no
  answer, so only the witnesses speak. `weak`: `other_score` >= 0.3, where 6 of 17 judged drops removed a
  real object; the page warns. Kept by saves, merges, X-AnyLabeling imports and every bulk edit; a save
  whose body has no `overlap_suggestions` key keeps the stored list (a key that is there, `[]` too, is
  the client's list).
- **`refine`** (written only when not empty, at most 64): what the dataset check (`train refine`,
  [Dataset check](TRAINING.md#dataset-check-cross-check)) suggests. Suggestions only: the check changes no
  box, status or `reviewed_at`, and nothing but this list.

  ```json
  {"kind": "add", "bbox": [412.0, 220.5, 498.0, 371.0], "label": "person", "score": 0.82,
   "why": "a model that never saw this camera-day found a person here, score 0.82", "run": "<check id>"}
  ```

  `kind`: `add` (a box for an object no box labels: `bbox`, `label`), `relabel` (another class for a
  training box: `target` and `label`), `verify` (a training box no model found: `target`) or `tighten` (a
  better-fitting `bbox` for a loose label `target`, written by the check's closer look: RF-DETR L on a crop
  of the label and the fold model's box, docs/TRAINING.md "Closer look at loose labels"; `score` is RF-DETR
  L's, `label` absent; the box may be tighter than the label, larger or shifted).
  `target` names an existing box by its exact `bbox`: that is the key. `target_label` (written by the check
  for `verify`, `relabel` and `tighten`; same bounds as `label`): the label of that box when the check ran, compared
  trimmed and case-insensitive, so of two boxes of different objects with the very same `bbox` the suggestion
  names the right one; without it (files of the first checks) the `bbox` alone is the key. `score`: the fold
  model's score, 0 to 1 (a quick check's, `train refine --with-model`: its model's). `why`: the evidence in
  plain words, at most 300 characters (a quick check's names its model: *mine (never trained on this frame)
  found a person here, score 0.82*). `spot`: an opaque group
  name, at most 200 bytes; the check writes it on an `add` at a static spot, the spot's id
  `<camera>/<object>/<x0>-<y0>-<x1>-<y1>` (`collect::labels::statics`, docs/TRAINING.md "Static spots"):
  the check's evidence, shown with the suggestion. It names a spot of that check's own table; the Refine
  tab's Static objects card and its bulk accept place an `add` by its box among the folder's spots now,
  never by this id. `run`: the id of the check that wrote it, at most 64 bytes. A suggestion whose box
  moved, went or was relabelled away from `target_label`, or that no longer holds, is stale and ignored: an
  `add` that a box of its object now covers; a `tighten` on a box a reviewer drew or moved; a `relabel` on a
  box a reviewer drew or relabelled, or to a class that names the same object; a `verify` on a box a reviewer
  drew or moved; a `tighten`, `relabel` or `verify` whose box is an ignore region (marked unsure,
  `difficult`, or rejected as `too_small`: the check never judges one). (A `relabel` on a box a reviewer
  only moved, and a `tighten` on a box a reviewer only relabelled, stay live; taking an unsure mark back
  makes an unanswered suggestion about that box live again.) Staleness is computed when the file is read, never stored, and never an error. So
  is any suggestion on a skipped frame. `answer`: `kept` (a `verify` the reviewer confirmed: the label stays)
  or `dismissed` (the reviewer said no); an answered suggestion is never live again and stays in the file, so
  a rerun does not ask again, until a check finds its `target` box gone (moved, relabelled, deleted): then
  it is dropped (an answered `add` has no target and stays). Accepting an `add` writes a box (`source:
  auto`, `state: accepted`, `reason: refine`) in the place of the unjudged rejects it covers; accepting a
  `tighten` keeps the old box in `orig_bbox`; accepting a `relabel` keeps the old label in `model_label`;
  accepting any of these three removes the suggestion and ends `overlaps_checked` when the box is now in a
  stack or an open pair; accepting a `verify` answers it `kept`.

  `GET /v1/review/labels` serves the live ones as `refine_live`: `[{"index", "kind", "target",
  "target_label", "bbox", "label", "score", "why", "spot", "run", "target_box", "replaces"}]`, absent fields
  `null`; `index` is the place in `refine`, `target_box` (the box `target` names, `null` for an `add`) and
  `replaces` (the unjudged rejects an `add` takes the place of, closest first) are places in `boxes`. The
  queue finds the frames with `auto=refine` ([Queue](#queue)).

  A `kind` or an `answer` this build does not know (a newer build's) loads as `unknown`, is never live
  and is written back as `unknown`. A check rewrites only the unanswered suggestions of the kinds it makes
  (`add`, `verify`, `relabel` on every frame it checks; `tighten` only of the labels its closer look
  answered); answered ones and other kinds stay (over the 64 a frame holds, a check gives up its own new
  suggestions, never those), and when
  nothing changed the file is not written. It never writes a suggestion a save would refuse (scores are clamped to 0..1 and boxes to the
  frame first). Kept by saves: a save (`POST /v1/review/labels`) whose body has no `refine` key keeps
  the stored list with its answers (a script or client that rebuilds the file from the boxes); a key that
  is there, `[]` too, is the client's list (the page always sends it). A build from before the dataset check ignores the field
  and drops it when it rewrites the file; the list is rebuildable by running the check again (answers are
  lost with it).
- **Auto-labeler fields** (`source: auto` boxes, and existing boxes an auto proposal confirmed), on the
  box itself, all optional:

  ```json
  {"label": "dog", "bbox": [1036.8, 489.6, 1372.0, 1189.1], "source": "auto", "score": 0.7541,
   "state": "pending", "reason": "auto: gdino + rfdetr", "agree": true, "triage": "review",
   "why": ["an animal at night: always reviewed"], "labelers": {"gdino": 0.6372, "rfdetr": 0.871},
   "vlm": {"model": "qwen3-vl-2b", "answer": "animal", "expected": "animal", "ok": true},
   "species_scores": {"dog": 0.9932, "cat": 0.0034, "fox": 0.0021}}
  ```

  `agree`: both labelers found it; `triage`: `accept` or `review` with the reasons in `why`;
  `labelers`: each labeler's (or the re-labelling model's) score; `vlm`: the crop check; `species_scores`:
  SigLIP's probabilities of the animal classes; `auto_label` / `auto_score`: a **class suggestion**, the
  class (and fused score) the auto-labelers give a pending box that is labelled otherwise (triage is then
  `review`, `why` says `"the auto-labelers say raccoon (0.90); the box says dog"`). The page shows them in
  the box list. They change what saving does in one case only: see the save rule.
- **Writes** go through a temporary file and a rename. A newer `version` is refused. Unknown fields are
  ignored.

### Pre-fill and the save rule

A frame without a label file starts with every sidecar box of a wanted class as `pending`, in this
order: `predictions`, `suppressed`, `below_threshold`, `verify.found`. Labels are mapped onto the list's
spelling (exact, ignoring case, then the COCO / VOC aliases: a VOC-spelled model's `pottedplant` is
`potted plant` when that is listed, with `model_label: "pottedplant"`); a box of no listed class, or of a
hidden class (see "Unwanted classes" below), is left out (the sidecar keeps it). A box is left out if an
earlier box has the same (mapped) label and IoU >= 0.95 (for `verify.found`, IoU >= 0.5). Label files
written before this rule can still hold model boxes of other classes: *Reject boxes of unwanted
classes* clears them (below).

When a frame is saved as `reviewed`, each `pending` box becomes:

| Box | Becomes |
|---|---|
| A prediction | `accepted` |
| An `auto` box | `accepted` |
| A `human` box | `added` |
| A suppressed, below-threshold or verifier-found box the auto-labelers confirmed (`triage: accept`, no `auto_label`) | `accepted` |
| Any other suppressed, below-threshold or verifier-found box | `rejected` |

In short, the reply's boxes are kept unless you reject them. A box the reply did not report must be
restored explicitly, or confirmed by the auto-labelers. Restoring a wrongly suppressed person is
false-negative evidence for that rule. The page applies the same rule before you save.

Every accepted or added label must be in `collect.classes`. A box with any other label must be
relabeled or rejected before the frame can be saved as reviewed.

### Merge rule (auto-labeler, re-labelling)

Later proposals (`source: auto` boxes, or a newer model's detections) are merged by
`labels::merge_proposals`, never by overwriting the file. Each proposal is compared with the existing box
of highest IoU at or above 0.5 (any label):

| Match | Result |
|---|---|
| No box | The proposal is added as `pending`. |
| A box with the same label | The proposal is dropped. The existing decision wins: a rejected box stays rejected. |
| A box with another label that is decided (`accepted`, `added` or `rejected`) | The proposal is added as `pending`, as a disagreement. |
| A box with another label that is still `pending` | The proposal is dropped. |

The auto-labeler (`training::autolabel::triage::apply`) runs two steps before this rule, per proposal,
against the boxes the file had before: the best box of the same label without auto evidence gets the
proposal's evidence (then dropped by the rule above); when no box of the same label matches and the best
box is a `pending` one of another label without auto evidence, that box gets the proposal's evidence as
a class suggestion (`auto_label`, `auto_score`, triage `review`) and the proposal is dropped.

A `reviewed` or `skipped` file that gains a box goes back to `unreviewed` and keeps its other decisions.
This is how the self-training loop sends disagreements back to review.

A `--missing-only` run (`triage::apply_missing`) uses its own rule instead: a looser match against
training and pending boxes, and proposals may replace rejected boxes no reviewer judged. See
[Find missing labels](TRAINING.md#find-missing-labels).

### Queue

`GET /v1/review/next` lists the frames of `review-queue.jsonl`, or with `source=all` also every indexed
frame. It accepts these filters: `bucket`, `scene`, `camera`, `label`, `status` (default
`unreviewed`; also `reviewed`, `skipped`, `any`) and `auto`:

| `auto=` | Frames |
|---|---|
| `review` | a pending box triaged `review` (auto boxes, annotated model boxes, class suggestions) |
| `accept` | the auto-labelers' triage alone settles the frame (below) |
| `nothing` | `auto_run` set and no box carries auto evidence (the labelers found nothing) |
| `none` | never auto-labelled: no `auto_run` and no auto evidence (frames without a label file too) |
| `repeats` | spot edits or parked spots settled every box (`repeat_spot`, nothing pending, no accepted box the VLM said no to) |
| `overlaps` | stacked or overlapping boxes no reviewer has looked at (no `overlaps_checked`), any review status |
| `both` | a pending box two or more labelers found (`labelers`), of a wanted class; most such boxes first (item field `agreed_boxes`) |
| `dismissed` | a dismissed box (`reason: "auto_dismissed"` or `"spot_dismissed"` from a labelers run, `"size_floor"` from the size floor rule), pending boxes or not. The page shows those boxes on these frames |
| `refine` | a live suggestion of the dataset check (`refine`, below; `refine::live` with the class list), any review status; never a skipped frame or one without a label file. Strongest first (item fields `refine_strength` and `refine_open`). `refine_kind=add\|tighten\|relabel\|verify` (only with `auto=refine`, else 400) counts that kind alone |

`auto=overlaps` obeys `status` and `source` like the other filters, and most frames with overlaps need
`status=any&source=all`: a reviewed frame is often not in `review-queue.jsonl`, and reviewed frames need
this cleaning too. Choosing *overlapping boxes* on the page sets both.

A frame is **accept-ready** (`labels::auto_accept_eligible`) when it is `unreviewed`, not confirmed
empty, has no `disagreement`, has at least one pending box triaged `accept` without a class suggestion,
and every other pending box is a suppressed, below-threshold or verifier-found model box without auto
evidence (rejected on save). A pending prediction the labelers did not confirm keeps the frame out, and
so does a dismissed box: nobody has seen it, and saving would call the frame complete.

**`label`** keeps the frames that have a box of that class. With a label file, these are its boxes
that are not rejected (accepted, added and pending ones). Without one, they are the sidecar's
`predictions`. The value must be a valid class name (1 to 64 characters, no control characters). It
need not be in the class list, so a model label such as `class_7` works too. The answer's `labels`
lists the classes of the frames the other filters keep, for the page's **Class** list (which also
holds `collect.classes`). The auto-label form passes it on like the other filters. Frames are sorted most uncertain first:

1. a re-label disagreement (the label file's `disagreement`, item field `relabel`);
2. more pending likely boxes (item field `likely_boxes`), so a frame with one comes before a frame with none;
3. fewer pending boxes triaged `review` (item field `auto_review`): many flagged boxes with no likely one
   are mostly false, and such a frame used to open the queue;
4. the near-threshold margin: the smallest `|top score - threshold|` over the labels of `raw` and
   `below_threshold`, in hundredths;
5. verifier disagreement (the `disagreement` bucket, or a rejected verifier candidate);
6. more suppressed boxes;
7. newest.

Under `auto=overlaps` the frames with a live suggestion come first, strongest first (item field
`overlap_strength`; `collect/labels/overlaps.rs` `strength`): drops with no other detection on the box,
then with one under 0.2, then under 0.3, then two objects, then the weak band, then the witnesses alone,
each by the zoom's score. Frames with no live suggestion follow in the order above.

Under `auto=refine` every listed frame has a live suggestion; they come strongest first by the frame's
strongest one (item field `refine_strength`, `refine::strength`: `add` 4, `relabel` 3, `tighten` 2,
`verify` 1, times 1000, plus the score in thousandths), then in the order above. `refine_open` is how many of
the frame's suggestions are live (of `refine_kind` when given).

`after=<frame>` returns the frame after the one shown, which is how **Next** skips a frame without saving.
After the last frame it returns the first one again with `"wrapped": true`. The answer also carries the
counts: reviewed, remaining, skipped, confirmed empty, rejected boxes, accepted boxes per class, and the
frames to review (`auto_review`) and accept-ready (`auto_accept`) by the `auto` rules. `audit_accepted`
and `audit_rejected` count the audit boxes of reviewed frames a reviewer found real or not; a bulk reject
by score or class, a duplicate and a too-small reject are no verdict and count as neither, while *Reject the
rest* counts as not real. The page shows them as *Hidden-box check: 1 of 40 was real*, with a percentage
from 20 boxes on. The counts follow every filter except `status` and `auto`.

### Saving, revisions and bulk accept

`GET /v1/review/labels` answers the file's `revision`: the first 16 hex digits of the SHA-256 of its
bytes, `""` when it is not saved yet. `POST /v1/review/labels?frame=...&revision=<rev>` saves only if the
file is still at that revision (checked and written under one lock, which auto-label runs take too);
otherwise it answers 409 `{"success": false, "conflict": true, "message": ...}` and writes nothing.
Without `revision` a save always overwrites (X-AnyLabeling and scripts).

`POST /v1/review/accept-auto` (JSON `{bucket?, scene?, camera?, label?, source?, dry_run, limit?}`, empty =
any, `limit` 1..20000, default 500) takes the queue with those filters, `status=unreviewed` and
`auto=accept`. `dry_run` answers `{success, eligible, accepted: 0, frames}` (the first 20). Otherwise
each frame, re-checked under the lock, is saved as `reviewed` by the save rule with `bulk_accepted:
true`: `{success, eligible, accepted, failed: ["<frame>: <why>", ...], frames, message}` (at most 20
listed each).

`POST /v1/review/reject-low` (JSON `{bucket?, scene?, camera?, label?, source?, min_score, dry_run,
limit?}`, `min_score` required, 0.01..1; the Review tab's Bulk tools, *Reject auto boxes below*) takes the
unreviewed frames of those filters and rejects their pending `source: auto` boxes scoring below
`min_score` (`state: rejected`, `reason: "low_score"`); the frame's status, its other boxes and reviewed
or skipped frames are left alone. `dry_run` answers `{success, eligible, boxes, rejected: 0, frames}`
(frames with such boxes, the boxes on them); otherwise each frame is re-checked under the lock:
`{success, eligible, boxes, rejected, failed, frames, message}` (`boxes` rejected, `rejected` frames
changed). The auto-label chain's reject step (docs/TRAINING.md, "Auto-label after import") is the
same code (`collect/labels/bulk.rs`), scoped to the import's frames.

### Unwanted classes

A label is **unwanted** when it maps to no class of `collect.classes` (never listed, or removed) or to a
class of `collect.hidden_classes` (the Classes card's *don't want (hide + reject)*; each a listed class,
case-insensitive). Hiding keeps the class in the list, so exports and YOLO class ids do not change; it
means "I do not review this class": the Review tab leaves it out of its pickers and shortcuts and does
not draw or list its boxes (nor rejected boxes of unlisted classes; they stay in the file, and a save
from the page rejects pending model / auto boxes of a hidden class instead of resolving them), the
pre-fill leaves its boxes out, auto-label runs leave it out of their classes (a run of hidden classes
only is refused) and reject pending boxes of unwanted classes in every frame they write.

When `POST /v1/review/classes` makes classes newly unwanted (removed or newly hidden; the answer's
`newly_unwanted`), the server right away rejects, on every unreviewed label file of the folder under
review, the pending `model` and `auto` boxes of unwanted classes (`reason: "other_class"`; an alias of a
wanted class is renamed instead) and answers `auto_rejected: {frames, boxes, relabelled, failed}` and
`reviewed_with_unwanted: {frames, boxes, classes}`: the reviewed frames whose accepted or added boxes are
of unwanted classes (both zero when nothing became unwanted). Human pending boxes are left to the
reviewer. A frame whose only pending boxes were unwanted leaves `auto=review` (and becomes accept-ready
when an `accept` box remains); it stays unreviewed. Reviewed frames are changed only on request: the page
asks "N reviewed frames have M accepted boxes of X, Y. Reject those too?" after a class save, after the
reject button below, and when the auto-label form leaves listed classes unticked (it first offers to hide
them: *Hide and auto-reject X, Y in Review? (Cancel = only skip them in this run)*).

`POST /v1/review/reject-other-class` (JSON `{bucket?, scene?, camera?, label?, source?, dry_run,
limit?}`; the Review tab's Bulk tools, *Reject boxes of unwanted classes*) takes the unreviewed frames of those
filters and, on each pending `source: model` or `source: auto` box whose label is not a wanted class in
the list's spelling: when the label is a wanted class under another spelling (ignoring case, or a COCO /
VOC alias such as `pottedplant` for `potted plant`), renames it to the list's spelling (still `pending`;
the old label in `model_label` unless only the case differs); otherwise rejects it (`state: rejected`,
`reason: "other_class"`). Human boxes, decided boxes, the frame's status and reviewed or skipped frames
are left alone. It clears the boxes of other classes a save would otherwise accept, which also makes a
frame accept-ready when they were all that kept it out. `dry_run` answers `{success, eligible, boxes,
relabelled, rejected: 0, frames, reviewed_with_unwanted}`; otherwise `{success, eligible, boxes,
relabelled, rejected, failed, frames, message, reviewed_with_unwanted}` (`boxes` rejected or renamed,
`relabelled` the renamed ones among them, `rejected` frames changed).

`POST /v1/review/reject-stacked` (JSON `{bucket?, scene?, camera?, label?, source?, dry_run, limit?}`;
the Review tab's Bulk tools, *Reject stacked duplicates*) walks every label file those filters select, reviewed
frames too (`source` is checked but does not narrow it). A stack is boxes of one object (the same label,
or two of car / truck / bus / van), neither rejected nor unwanted, at IoU >= 0.7; a chain is one stack.
The keeper is the drawn box, else a reviewer-edited one, else an accepted one, else a pending one (a hint
saving would reject last), then the highest score. The others get `state: rejected`, `reject_reason:
"duplicate"`, `difficult` off and `reason: "duplicate"` when they had none. `human` boxes are never
rejected; skipped frames and frames with `overlaps_checked` are left alone; the frame's status and
`reviewed_at` do not change. Nested or partly overlapping boxes (IoU >= 0.5, or the smaller >= 80% inside
the larger) are never taken: `auto=overlaps` lists them, and any stack still left, for a reviewer. Both
`dry_run` and a run answer `{success, eligible, boxes, rejected, failed, frames, message, overlaps_left,
overlaps_left_reviewed}` (`boxes` rejected, `rejected` frames changed, `overlaps_left` the frames
`auto=overlaps` would still list once the duplicates are gone, `overlaps_left_reviewed` the reviewed ones
among them). A save as reviewed, Shift+A and the bulk accept reject pending boxes stacked
on a better one the same way. A duplicate sits on a real object: it is left out of `hard_negatives.json`,
the rejected-box count and the per-camera repeat history.

`POST /v1/review/suggest-overlaps` (JSON `{bucket?, scene?, camera?, label?, gpu?, allow_large?, again?,
dry_run?}`; the Review tab's *Suggest overlaps...*) queues the overlap check on the label files of the
folder under review those filters select, reviewed frames too. `dry_run` needs no GPU and answers
`{success, frames, pairs, no_zoom, crops, answered, passed_over, message}`; otherwise `{success, message,
id}`, the task's id in `GET /v1/training`. `GET /v1/review/labels` carries `overlap_live`, the indices of
the frame's `overlap_suggestions` a reviewer can act on.

`POST /v1/review/reject-unwanted-reviewed` (JSON `{dry_run, limit?}`, `limit` 1..20000, default 20000)
rejects the accepted and added boxes of unwanted classes on every reviewed label file of the folder under
review (`reason: "other_class"`; a rejected `human` box is dropped at the frame's next save). The frames
stay `reviewed`, and their remaining training boxes are all of listed classes, so they still save and
export. `dry_run` answers `{success, eligible, boxes, classes, rejected: 0}`; otherwise `{success,
eligible, boxes, rejected, failed, frames, message}`. Both routes are admin routes like reject-low.

<a id="spots"></a>
### Spots

`<review folder>/spots/<camera>.json` holds a camera's parked spots: one file per camera, entries per frame
size. The folder sits outside the scene folders, so `layout::day_dirs`, the label walks and retention never
go there. Only the service writes it, atomically under its own lock. A labelers run reads it once, at its
start. A file that does not parse stops the routes that use it, and a run then gives that camera no spots
this time, with a warning.

```json
{
  "version": 1,
  "moved_at": "2026-11-05T10:20:30.123Z",
  "parked": {
    "1920x1080": [
      {"label": "car", "bbox": [100.0, 200.0, 400.0, 420.0], "from": "2026-10-01"},
      {"label": "truck", "bbox": [900.0, 200.0, 1300.0, 500.0], "from": "2026-10-03", "retired": "2026-10-20"}
    ]
  }
}
```

- **`from`:** the date of the frame where you pressed *Real, parked*.
- **`retired`** (optional): the date the spot stopped counting. The entry stays as a record.
- **`moved_at`** (optional): when you pressed *Camera moved: forget spots*. Decisions on frames from before
  it no longer count.

Rejected spots are not stored: they are derived from your decisions in the label files. What a run does
with both kinds is in [docs/TRAINING.md](TRAINING.md#spot-memory-of-fixed-cameras).

| Route (admin) | What it does |
|---|---|
| `GET /v1/review/spots?camera=` | parked spots, derived rejected spots (`rejections`, `last`), `moved_at`, and `live`, the number *Camera moved* would forget |
| `POST /v1/review/spots/parked` | `{frame, label, bbox}`: add a spot, from the frame's camera, size and date. The page accepts the box itself |
| `POST /v1/review/spots/retire` | `{camera, width, height, label, bbox}`: retire one spot, 404 if there is none |
| `POST /v1/review/spots/moved` | `{camera}`: retire every parked spot and set `moved_at` |

### Keyboard

| Key | Action |
|---|---|
| `a` | Accept every pending box. |
| `r` | Reject or restore the selected box. |
| `t` | Real, parked: accept the selected box and remember its spot. Not on drawn boxes. |
| `1`-`9`, `0` | Pick class 1-10: relabel the selected box, else set the class of new boxes. |
| `u` | Reject the selected box as a duplicate of the box it overlaps (Shift: every stack on the frame). |
| `y` | Accept the overlap suggestion on screen: reject the box marked `drop?` as a duplicate and keep the other (a hint saving would reject is accepted, so the object keeps a box), or for two objects keep both. Doing anything else ignores it. |
| `j` / `k` | Select the next / previous box. |
| `Delete` | Delete a drawn box, reject a model box. |
| `e` | Confirmed empty (rejects every box). |
| `z` | Undo. |
| `s`, `Enter` | Save & next. A box the auto-labelers flagged (`?`) needs a decision first: the save refuses it, on the page and on the server, however often it is pressed. |
| `Shift+R` | Reject the rest: every box still undecided (`reason: rest_rejected`; these count for the repeats history and spot memory). |
| `n` | Next without saving. |

Click a box to select it. Drag on the image to draw a new one.

### Export

`POST /v1/review/export` exports the **reviewed** frames. Its body:

```json
{"formats": ["yolo", "coco", "xanylabeling", "benchmark"], "out_dir": "optional",
 "holdout_cameras": [], "holdout_days": [], "holdout_fraction": 0.2,
 "benchmark_split": "all", "include_unreviewed": false,
 "labels": [], "include_empty": true}
```

- **`formats`** (or `format`): one name, a list, or `all`. Default: all formats.
- **`out_dir`:** a new or empty folder. Default: `<data root>/training/exports/<UTC time>`.
- **`labels`** (default empty = every class): export only the frames with a training box (accepted or
  added) of one of these classes. The other boxes of a kept frame stay labelled, and class ids still
  follow `collect.classes`. Every name must be in `collect.classes`; another one is a 400 that names
  it.
- **`include_empty`** (default true): also export the frames confirmed empty. With `labels` empty and
  `include_empty` false, every frame but the confirmed-empty ones is exported.
- The class filter applies before the split. `export.json` records it (`class_filter`) with the
  number of reviewed frames it left out (`filtered_out`). With `include_unreviewed`, the unreviewed
  X-AnyLabeling frames are filtered by the boxes a save would accept.
- **Frames left out, with a warning:** frames with a training label outside the class list, and frames
  whose image or sidecar is gone.
- **File names:** every frame is exported as `<scene>.<camera>.<date>.<stem>.jpg`. No part contains a
  `.`, so the name maps back to its frame.
- **Images** are hard-linked where the file system allows it, else copied.

```
<out>/
  images/{train,val}/<name>.jpg        YOLO and COCO share the images
  labels/{train,val}/<name>.txt        YOLO: `class cx cy w h`, normalised, clamped to 0..1, 6 decimals; empty file = no object
  classes.txt, data.yaml               class ids follow collect.classes; data.yaml is an Ultralytics dataset
  annotations/instances_{train,val}.json   COCO: categories 1..K in class order, image / annotation ids from 1,
                                       bbox [x, y, w, h] px (2 decimals), area = w * h, iscrowd 0,
                                       attributes.difficult on unsure boxes (left out of YOLO)
  hard_negatives.json                  rejected model / auto boxes (frame, label, bbox, origin, reason, reject_reason), without
                                       duplicate, too_small and dismissed (auto_dismissed, spot_dismissed, size_floor) ones
  ignore.json                          ignore regions per image, below, and in `parked` the frames whose
                                       training boxes all came from parked spots (for the training cap)
  frames.json                          `empty_confirmed` and `bulk_accepted`: image names, for a training run whichever
                                       formats were ticked (absent in older exports)
  xanylabeling/<name>.jpg + .json      X-AnyLabeling files (below)
  benchmark/images/<name>.jpg + manifest.json   a benchmark `dir:` set
  export.json                          the report, written last
```

<a id="ignore-regions"></a>**Ignore regions.** A real object not to train on, and no negative either: an
unsure box (`difficult`), a box rejected with `reject_reason: too_small`, or (with `collect.size_floor`
set) a training box under the size floor, below. YOLO has no way to say "skip this object", so a training
run decides what such a frame becomes (`training.ignore`, `docs/TRAINING.md` "Ignore regions"): by default
(`background`) a frame with a label of the run's classes trains with the object unlabelled, a frame of
nothing but such objects is left out (it is no "nothing here" frame, confirmed empty or not), and
validation never holds one; `drop-frame` leaves every such frame out of train and valid, as earlier versions
did. Counted in `dataset.json` (`ignore_frames`, `ignore_only`, `ignore_valid_day`, `ignore_kept`,
`ignore_boxes_background`) with a warning. It stays in the held-out set, where the box is `ignore`: a
detection on it counts neither way. `ignore.json` at the export's root carries them, in the numbers of a
YOLO label line, so no pixel size is needed to read it:

```json
{ "version": 1,
  "images": { "ir.drive.2026-10-22.20261022T010203004Z_abcd.jpg":
                [ { "label": "person", "cx": 0.11, "cy": 0.14, "w": 0.02, "h": 0.08 } ] } }
```

Keys are the image file names of `images/<split>/`, a frame without regions is not listed, and labels stay
as reviewed even outside the class list. Written with every export format. A folder without the file has no
regions, so an export from an older version still reads, with those objects unlabelled as before: export
again to use them. A damaged file stops the run rather than teach them as background.

<a id="size-floor"></a>**Size floor.** `collect.size_floor` (`{"px": 10, "input": 384}`; off by default)
says how small an object a model of that input size can learn: a training box whose short side at the
model's input, `min(w / W, h / H) * input` of the box clamped to the frame, is under `px` cannot be
learned (measured on 851 reviewed frames with fold models that never saw the camera day, at 384 px: under
8 px found 6% of the time, 8-12 px 37%, 12-24 px 83%, above that 94%). An export then writes such a box
(accepted or added, not already an ignore region) as an ignore region: out of the YOLO and the benchmark
ground truth, in `ignore.json` with `"size_floor": true` (and the floor at the file's top, `size_floor`),
`ignore: true` in the benchmark manifest, `attributes: {"difficult": true, "size_floor": true}` in COCO.
X-AnyLabeling files keep it as a plain box: an import would otherwise mark it unsure in the label file for
good. The label file never changes: the person stays "person, accepted", so another floor is just another
export. `export.json` records `size_floor` and counts `under_floor` (`train` and `val` per class, and
`frames_only`, frames left with no label but such regions; never "confirmed empty"). Older builds read such
an export as one with more unsure boxes. Held-out scores before and after a floor change do not compare:
the held-out set's ignore regions change, and with them its fingerprint (`heldout_fingerprint`). The
dataset check judges by the same rule (`docs/TRAINING.md`; its fold models train without it), and a
training run at a larger input than the floor's gives back the regions that are not under it there
(`docs/TRAINING.md`, Ignore regions). Accepted and added labels are never rewritten for the floor; the
Refine tab's one rule only hides *pending proposals* under the floor on unreviewed frames (`reason:
"size_floor"`, a dismissal like `auto_dismissed`, above), and its *Restore* gives back those a lower floor
or none no longer covers.

**Split.** Frames are split by whole camera days (camera and date folder), so frames of one camera on
one day never sit on both sides. That does not mean an unseen camera: other days of a held-out camera
train unless the camera is named in `holdout_cameras`.

- If `holdout_cameras` or `holdout_days` is given, a frame of a named camera or day goes to `val`, and
  every other frame to `train`.
- Otherwise a camera-day is held out when the first 8 bytes of SHA-256(`<camera>/<date>`), as a fraction
  of 2^64, fall below `holdout_fraction`. This is deterministic, and stable when days are added.
- Once there are two or more camera-days, at least one goes to each side.

**Benchmark set.** `manifest.json` gives each frame its training boxes as ground truth, and its ignore
regions (too-small rejects included) as `ignore` objects. `scored_labels`
is the class list, so predictions of other labels count neither way. The tags are:

- `<scene>`, `camera:<c>`, `date:<d>`, `split:<s>`, and `bucket:<b>` for each bucket;
- `hard-negative` for a frame confirmed empty.

`benchmark_split` (`all`, `val`, `train`) picks the frames. Run the set with
`blue-onyx-prism benchmark --dataset dir:<out>/benchmark`.

### X-AnyLabeling

X-AnyLabeling saves one LabelMe-style JSON per image, beside the image:

- **File keys:** `version`, `flags`, `shapes`, `imagePath`, `imageData`, `imageHeight`, `imageWidth`,
  `description`, `checked`.
- **Shape keys:** `label`, `score`, `points`, `group_id`, `description`, `difficult`, `shape_type`,
  `flags`, `attributes`, `kie_linking`.

**Export** writes each frame's training boxes as `rectangle` shapes with four corner points (top-left,
top-right, bottom-right, bottom-left), which is how X-AnyLabeling saves rectangles. Reviewed frames get
`checked: true`. With `include_unreviewed`, the queue's unreviewed frames are added with the boxes a save
would accept, and `checked: false`.

**Import** (`POST /v1/review/import` with `{"dir": "...", "treat_unchecked_as_reviewed": false}`) reads
every `*.json` in the folder. The frame comes from the `imagePath` file name, else the JSON's stem. A file
counts only when `checked` is true, unless unchecked files are treated as reviewed. Its rectangles
become the frame's complete set of training boxes:

- **Rectangles:** two-point or four-point rectangles, and polygons as their bounding box, are read.
  Other shapes are skipped with a warning.
- **Matching:** a rectangle at IoU >= 0.9 of an existing box replaces that box. A box of the same label
  is preferred. Its label and coordinates are taken, and the old label is kept in `model_label`.
- **New and deleted boxes:** any other rectangle becomes a `human` box. An existing box no rectangle
  matched is rejected (a model or auto box) or dropped (a human box).
- **Result:** the frame is saved as `reviewed`. With no rectangle it is `empty_confirmed`.
- **Refused files:** a file whose image size differs from the frame's (a resized copy), or whose labels
  are not in the class list, is an error for that file only.

### Safety

- **Frame references:** every `frame` parameter goes through `layout::parse_rel` / `layout::resolve`.
  The image and sidecar must be regular files inside the dataset folder, and the label file must not be
  a symlink. Anything else is a 404.
- **POST bodies** must be JSON (415 otherwise) and are capped: 256 KiB for labels, 16 KiB for export,
  import and classes (413).
- **Admin guard:** the routes sit behind sign-in (HTTPS, a session or the API token; 401 or a
  redirect to `/login` otherwise) and the cross-site check (403).
- **Folders:** `out_dir` and the import `dir` are checked like the other folder settings (no system
  folders), and must be under a browse root or picked in the native dialog. An `out_dir` that does not
  exist yet must be inside the browse roots or a picked folder.
- **Build and report:** an export is built in a hidden folder and appears under its name only once
  complete, so a failed one is never offered to training. The answer's `lost_frames` and
  `unreadable_folders` count reviewed frames left out because their image or sidecar is gone or a
  folder could not be read, and warnings name up to 20 of them.
- **An unavailable folder:** a folder under review that is not there, such as an unplugged drive, is
  not an empty dataset. `/v1/review/next` answers `folder_missing: true`, the routes that need the
  folder answer 503, and the Train strip says so. Nothing is rejected or pruned meanwhile.
- **Several writers:** `<dataset root>/.labels.lock` is an empty file that keeps the service and a
  `train autolabel` from the CLI out of each other's label writes, one write at a time. Where locks
  do not work, such as some network shares, writes carry on with one warning.

**Retention:** a reviewed frame, one with a human decision, or one with boxes waiting for review is never
pruned and does not count toward `collect.max_gb` (see [Disk](#disk)). A label file with nothing left to
look at is pruned with its frame.
Delete reviewed frames yourself if they are no longer wanted.
