# Dataset collector

The collector saves frames that Blue Iris sends, together with what the model saw and did. Later stages
use them: the review page, footage import, auto-labelling and training (plan, Phase 5). It is off by
default. Turn it on with `collect.enabled` or on the **Dataset** page (`/dataset`). Every setting
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

### `index.jsonl`

One JSON object per line, appended when a frame is saved:

```json
{"id": "...", "timestamp": "...", "scene": "ir", "camera": "driveway", "model": "IPcam-general", "buckets": ["suppressed"], "labels": [], "bytes": 415000}
```

- `bytes` is the image plus the sidecar.
- When frames are deleted, retention rewrites the day's index atomically.
- At startup, the service rebuilds the Dataset page's summary (counts by scene x bucket and per camera,
  bytes, the newest 8 frames per bucket) from the index files.

### `review-queue.jsonl`

The Dataset page's **Send to review** appends one line per ticked frame that is not queued yet:

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
  - It deletes only frame pairs in the layout above. It never deletes `review-queue.jsonl` or anything
    else. A frame in the review queue is pruned like any other until the review stage copies it out.
- **Low disk:** if free space under the folder drops below 1 GB (`retention::MIN_FREE_MB`, the same
  threshold as the low-space warning), or a write fails with a full disk or exhausted quota:
  - collection stops with a warning;
  - `dataset_collect_disabled` becomes 1 and frames count as `dataset_dropped_total{reason="disk"}`;
  - the Dataset page shows the reason.
  - It stays off until **Resume** is pressed on the page or the `collect` settings change.
- **Other write errors** (permissions, a read-only folder) drop the frame (`reason="error"`) with a
  throttled warning.

## Metrics

| Metric | Meaning |
|---|---|
| `blue_onyx_prism_dataset_frames_total{bucket}` | Frames saved per bucket (a frame counts in each bucket it was kept for). |
| `blue_onyx_prism_dataset_dropped_total{reason}` | `queue_full`, `rate`, `dup`, `disk`, `error`. |
| `blue_onyx_prism_dataset_bytes` | Image and sidecar bytes in the folder (from the index). |
| `blue_onyx_prism_dataset_collect_disabled` | 1 while collection is stopped for disk space. |

## How the later stages use it

- **Review page** (`/training`):
  - reads `review-queue.jsonl` first, then any frame by bucket via the index files;
  - shows `raw`, `suppressed` and `predictions` boxes from the sidecar;
  - sorts by uncertainty: `near_threshold`, `disagreement`, the verifier outcome;
  - writes YOLO `.txt` labels beside the frame (same stem).
  - Suppressed and `verify_rejected` night boxes marked wrong become hard negatives. Reviewed `empty`
    frames become negative images.
- **Auto-label** (offline, training env):
  - re-detects `empty` frames with the large labellers; a confident box is a candidate miss;
  - uses `below_threshold` and `verify.found` as cheap hints.
- **Dataset assembly:**
  - splits by `camera` and date folder, so held-out frames come from cameras and days the model did not
    train on;
  - balances by `scene`;
  - uses `dhash` to drop near duplicates across days.
- **Import:** imported footage uses the same layout and sidecar, with its own source and licence
  fields added (a later version).
