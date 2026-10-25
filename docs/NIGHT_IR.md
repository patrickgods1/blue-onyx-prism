# Night / IR detection

This page covers the features that reduce night and infrared false positives (rocks, plants, statues reported as
`person`), and what the benchmarks say about them.

## How a frame is classified

Every request gets a scene from a 64x64 sample of the decoded frame (`src/image_stats.rs`):

| Scene | Rule |
|---|---|
| `ir` | Monochrome: colour spread or tint-fit residual below 3/255. This catches gray IR frames and green- or white-balance-tinted ones, even when IR lighting makes them bright. |
| `night` | Colour frame with mean BT.601 luma below 50. |
| `day` | Everything else. |

The scene appears in the per-request debug log (`scene=`, `luma=`, `chroma=`). Night rules apply to `night` and `ir`.

## Settings

All of these are off by default. They apply live (no restart) and can be set globally or per model; the per-model
value wins. See the README config table for the exact keys.

- **Night thresholds:**
  - `night_confidence_threshold`: the minimum confidence for every label on night and IR frames.
  - `label_thresholds` / `night_label_thresholds`: per-label minimums, e.g. `{"person": 0.6}`.
- **Shape rules:**
  - `person_aspect: [min, max]` checks a person box's width / height; real people are tall.
  - `min_box_area_frac` drops tiny boxes.
- **`suppress_overlaps`:** drops a box that overlaps another label, e.g. a `person` box on a `potted plant`. Use this
  instead of Blue Iris "to cancel" labels, which cancel the whole alert.
- **`static_suppression`:** a box that sits in exactly the same place across many requests from one camera is treated
  as background and suppressed.
  - The camera is identified from the upload's file name.
  - A person who stands still for the whole window is suppressed too, so it stays opt-in.
- **`preprocess` / `night_preprocess`:** `none | grayscale | gamma:<g> | clahe[:clip,tiles]`. The night variant
  replaces `preprocess` on night and IR frames.
- **`night_threshold`** (per model): the same model answers night and IR frames with another confidence threshold, lower
  or higher than its day threshold or device profile (`night_confidence_threshold` can only raise it). On the Config
  page: Night model column, *same model, night threshold*. A request's `min_confidence` still wins.
- **`night_model`:** night and IR frames are handed, already decoded, to another configured model. If that model is not
  available, the frame runs on the original model.
  - Metrics: `night_model_routed_total` and `night_model_fallback_total{reason}`.
  - Suppressed boxes are counted in `detections_suppressed_total{reason}`.

The config page has a **Night / IR** fieldset and a **Night model** column on the Models card.

## Benchmark sets

| Set | What it is | Default |
|---|---|---|
| `exdark-night` | 160 low-light colour photos (ExDark), 458 persons | yes |
| `exdark-ir` | The same 160 photos converted to grayscale. An approximation of NIR, not real IR. | opt-in |
| `nir-night` | 204 real night IR frames from fixed camera traps (LILA BC, CDLA-Permissive): dog, cat, raccoon, car, other wildlife, and 80 hand-reviewed empty frames of vegetation, rocks and IR glare. Only 2 person boxes, because LILA removes most human images, so person is effectively scored as false positives only. | opt-in |

Compare variants with:

```
blue-onyx-prism benchmark --dataset nir-night --dataset exdark-ir --dataset exdark-night \
  --variants none,filters,pre:clahe,pre:gamma:0.6,night-pre:grayscale
```

`filters` uses the model's configured night rules. If none are configured, it uses the `night-default` preset:

- night threshold 0.5;
- night person threshold 0.6;
- person aspect between 0.2 and 0.9;
- drop a person box that overlaps a potted plant.

## Results (2026-10, Apple M1, `openvino:cpu`, threshold 0.5)

Person false positives per image, on frames that contain no person:

| Model | Variant | exdark-night | exdark-ir | nir-night | Person recall, exdark-night | ms p50 (640 px) |
|---|---|---|---|---|---|---|
| IPcam-general | none | 0.071 | 0.114 | 0.030 | 0.50 | 41 |
| IPcam-general | filters | 0.057 | 0.043 | 0.020 | 0.43 | 41 |
| yolo26s | none | 0.043 | 0.043 | 0.020 | — | 68 |
| yolo26s | filters | 0.014 | 0.000 | 0.010 | — | 67 |
| rt-detrv2-s | none | 0.086 | 0.114 | 0.050 | — | 112 |
| rt-detrv2-s | filters | 0.029 | 0.029 | 0.010 | — | 114 |

Person AP50 with no variant applied:

| Model | exdark-night | exdark-ir |
|---|---|---|
| IPcam-general | 0.686 | 0.647 |
| yolo26s | 0.633 | 0.616 |
| rt-detrv2-s | 0.695 | 0.674 |

What the results show:

- **The night filters are the most effective change.** They cut person false positives by 2–4x on every model, at
  roughly the same F1 at 0.5. They cost some person recall: IPcam-general drops from 0.50 to 0.43.
- **CLAHE and gamma on every frame make things worse.** Person AP falls 2–12 points and recall falls 3–13 points on
  ExDark. On nir-night they raise rt-detrv2-s AP but double its person false positives. CLAHE also adds about 15 ms on
  4 MP frames. Neither is a default. This matches published results: enhancing images before detection rarely helps.
- **Converting night frames to grayscale is neutral.**
- **Models:**
  - rt-detrv2-s has the best recall and AP, but the most false positives and 2.8x the latency.
  - yolo26s is the most precise.
  - IPcam-general is the fastest and nearly as good on person AP, but has no animal classes.
  - On ExDark, switching to yolo26s at night does not improve person AP; it trades recall for precision.

**Suggested starting point:**

- Keep the main model.
- Enable the night rules: `night_label_thresholds: {"person": 0.6}`, `person_aspect: [0.2, 0.9]`, and drop `person`
  overlapping `potted plant`.
- Enable `static_suppression` for cameras with recurring false positives.
- Leave preprocessing off.

What these numbers do not show:

- Real IR person accuracy. nir-night has only 2 persons, and exdark-ir is a grayscale approximation.
- Statistically solid differences. The false-positive counts are 0–14 boxes, so a difference of 0.014 is one box.
- The `night_model` switch end to end. The benchmark runs each model on its own, not with routing.

Benchmarking your own night snapshots (`--dataset dir:<path>`) is the next step.

## Verifier cascade (`verify`) (2026-10, Apple M1, `openvino:cpu`, threshold 0.5)

`verify: {"model": "yolo26s"}` re-checks the primary's night `person` boxes with a second model before replying.

How it works:

- **Crop mode (the default):** each candidate box is padded by 20% and sent as a crop.
- **Frame mode:** one full-frame request, with candidates matched to the verifier's boxes by IoU.
- **Fail-open:** if the verifier is missing, not ready, busy, past the deadline or answers with an error, the
  candidates are returned unverified.
- **Ordering:** verified replies may overtake other replies of the same model.
- **Metrics:** `verify_total{outcome}`, and `detections_suppressed_total{reason="verify"}`.

Person false positives per person-free image, and person recall:

| Model | Variant | FP exdark-night / exdark-ir / nir-night | Person recall exdark-night / ir |
|---|---|---|---|
| IPcam-general | none | 0.071 / 0.114 / 0.030 | 0.50 / 0.49 |
| IPcam-general | filters | 0.057 / 0.043 / 0.020 | 0.43 / 0.40 |
| IPcam-general | verify:yolo26s (crop, 0.5) | 0.043 / 0.000 / 0.000 | 0.33 / 0.29 |
| IPcam-general | verify, mode frame, min_confidence 0.3 | 0.014 / 0.043 / 0.010 | 0.44 / 0.42 |
| rt-detrv2-s | none | 0.086 / 0.114 / 0.050 | 0.67 / 0.65 |
| rt-detrv2-s | verify:yolo26s (crop, 0.5) | 0.014 / 0.000 / 0.005 | 0.49 / 0.43 |
| rt-detrv2-s | verify, mode frame, min_confidence 0.3 | 0.057 / 0.029 / 0.025 | 0.56 / 0.48 |

- **Crop mode at the defaults:** removes almost every night person false positive, but loses about a third of person
  recall, because yolo26s misses many people in tight crops. Each crop costs a full verifier inference: about 70 ms
  on the M1 CPU.
- **`mode: "frame"`, `min_confidence: 0.3`:** about as effective as the filters on false positives, with the same or
  better recall, and only one extra inference per night frame with candidates. Start here.
- **Small counts:** the false-positive counts are 0-10 boxes, so differences of one box are noise.

## Night model candidates (2026-10, Apple M1, `openvino:cpu`, threshold 0.5)

Five exportable models (Config page *Resources*, or `fetch --resource model:<name>`; nothing is exported
automatically) were added as `night_model` candidates and benchmarked on the same sets. IPcam-general was
rerun in the same session as an anchor; it reproduces the numbers above exactly.

| Model | Source | License | Classes | Input |
|---|---|---|---|---|
| `mdv6-rtdetr-c` | MegaDetector v6 Apache RT-DETR compact (Microsoft), `MDV6-apa-rtdetr-c.pth`, Zenodo record 15398270 | Apache-2.0 | animal, person, vehicle | 640 |
| `mdv6-yolov9-c` | MegaDetector v6 Ultralytics YOLOv9 compact, `MDV6-yolov9-c.pt`, same record | AGPL-3.0 | animal, person, vehicle | 640 |
| `mdv6-yolov10-c` | MegaDetector v6 Ultralytics YOLOv10 compact (NMS-free), `MDV6-yolov10-c.pt`, same record | AGPL-3.0 | animal, person, vehicle | 640 |
| `exdark-rfdetr-small` | RF-DETR small fine-tuned on ExDark (DetectionBench), `dronefreak/exdark-rfdetr-small` | Non-commercial / personal use (ExDark) | 12 ExDark classes | 512 |
| `exdark-rfdetr-nano` | RF-DETR nano fine-tuned on ExDark (DetectionBench), `dronefreak/exdark-rfdetr-nano` | Non-commercial / personal use (ExDark) | 12 ExDark classes | 384 |

How they are exported:

- **MegaDetector RT-DETR:** the checkpoint is built with the RT-DETRv2 modules of the pinned DEIMv2 source
  (`tool:deimv2-src`) and read with `torch.load(weights_only=True)`. Our export matches PytorchWildlife's own
  RT-DETRv2 code within 5e-6 in score and 7e-4 px in box on 16 real night images.
- **MegaDetector YOLO:** the existing Ultralytics recipe, with 3-class heads.
- **ExDark RF-DETR:** the `rfdetr` training checkpoints are loaded into transformers' `RfDetrForObjectDetection`.
  The weights fit with nothing missing or mismatched. This replaces `rfdetr`'s own `export()`, so the toolchain
  needs no new packages.

Every export is checked against PyTorch with ONNX Runtime, like the other recipes.

Label mapping:

- MegaDetector's `vehicle` is its own class (cars, trucks and other vehicles), not COCO's `car`. The benchmark
  scores it on car, truck and bus ground truth, as it does for IPcam-general.
- MegaDetector's `animal` is a class group. It is scored on every animal box: the COCO animals and the camera-trap
  wildlife labels (raccoon, fox, deer, ...). Wildlife boxes the set marks as ignore regions stay ignore regions.
  So on nir-night, MegaDetector's animal score covers 93 boxes (31 dogs, 30 cats, 32 raccoons), not only dog and
  cat.
- The ExDark models report COCO spellings: People becomes `person`, Motorbike `motorcycle`, Table
  `dining table`. They have no truck or other COCO classes.

The ExDark RF-DETRs lose most of their confidence in OpenVINO's f16. That is the CPU default on ARM: the best
person score drops from 0.885 to 0.50 on one image. They are in `F32_MODELS`, so they run in f32.

**Caution: OpenVINO's compile cache ignores the precision change.** A model compiled in f16 before it joined
`F32_MODELS` kept its f16 blob: person AP50 stayed at 0.21 until the cache was cleared. Clear `cache/` after
changing that list.

Person false positives per image (on frames with no person), person recall and AP50:

| Model | Variant | FP/img, exdark-night | FP/img, exdark-ir | FP/img, nir-night | Recall, exdark-night | Recall, exdark-ir | AP50, exdark-night | AP50, exdark-ir | ms p50, exdark / nir |
|---|---|---|---|---|---|---|---|---|---|
| IPcam-general | none | 0.071 | 0.114 | 0.030 | 0.50 | 0.49 | 0.686 | 0.647 | 40 / 55 |
| IPcam-general | filters | 0.057 | 0.043 | 0.020 | 0.43 | 0.40 | 0.476 | 0.388 | 40 / 55 |
| mdv6-rtdetr-c | none | 0.271 | 0.214 | 0.015 | 0.43 | 0.40 | 0.506 | 0.488 | 109 / 131 |
| mdv6-rtdetr-c | filters | 0.143 | 0.114 | 0.015 | 0.33 | 0.30 | 0.364 | 0.272 | 112 / 131 |
| mdv6-yolov9-c | none | 0.000 | 0.014 | 0.000 | 0.14 | 0.10 | 0.298 | 0.249 | 186 / 204 |
| mdv6-yolov9-c | filters | 0.000 | 0.014 | 0.000 | 0.12 | 0.07 | 0.173 | 0.073 | 189 / 204 |
| mdv6-yolov10-c | none | 0.043 | 0.043 | 0.000 | 0.19 | 0.13 | 0.318 | 0.276 | 34 / 47 |
| mdv6-yolov10-c | filters | 0.014 | 0.014 | 0.000 | 0.14 | 0.09 | 0.193 | 0.081 | 35 / 47 |
| exdark-rfdetr-small* | none | 0.000 | 0.014 | 0.000 | 0.71 | 0.71 | 0.863 | 0.847 | 289 / 307 |
| exdark-rfdetr-small* | filters | 0.000 | 0.000 | 0.000 | 0.59 | 0.55 | 0.621 | 0.540 | 286 / 301 |
| exdark-rfdetr-nano* | none | 0.000 | 0.000 | 0.000 | 0.63 | 0.60 | 0.804 | 0.772 | 158 / 169 |
| exdark-rfdetr-nano* | filters | 0.000 | 0.000 | 0.000 | 0.53 | 0.48 | 0.575 | 0.467 | 159 / 171 |

\* **The ExDark numbers on exdark-night and exdark-ir are optimistic.** DetectionBench trained these models on a
random 80/10/10 split of all of ExDark. The 160 exdark-night images come from the Roboflow test split and may be
in their training set. exdark-ir is the same images in grayscale. Only nir-night is a fair read for them.

There are 70 person-free frames in exdark-night and exdark-ir, and 202 in nir-night. At 70 frames, one box is
0.014.

nir-night, the real IR camera-trap set (AP50 / recall at 0.5):

| Model | animal (93 boxes) | dog (31) | cat (30) | vehicle or car (14) |
|---|---|---|---|---|
| IPcam-general | — | — | — | 0.100 / 0.07 |
| mdv6-rtdetr-c | 0.906 / 0.94 | — | — | 0.359 / 0.36 |
| mdv6-yolov9-c | 0.955 / 0.94 | — | — | 0.306 / 0.36 |
| mdv6-yolov10-c | 0.922 / 0.89 | — | — | 0.331 / 0.43 |
| exdark-rfdetr-small | — | 0.727 / 0.61 | 0.586 / 0.53 | 0.129 / 0.07 |
| exdark-rfdetr-nano | — | 0.532 / 0.45 | 0.406 / 0.50 | 0.164 / 0.14 |

What the results show:

- **MegaDetector is the IR animal and vehicle model.**
  - It finds 89–94% of the animals in real night IR frames, raccoons included, and puts no person box on the 202
    person-free frames (the RT-DETR: 3 boxes).
  - It is the best vehicle model on nir-night, but weak on ExDark street vehicles (AP50 0.08–0.14). Camera-trap
    viewpoints are not street scenes.
- **MegaDetector is a poor person model on colour night photos.**
  - The YOLO variants miss most ExDark persons (recall 0.10–0.19).
  - The RT-DETR has fair recall but the most false positives (0.27 per frame).
  - So it suits `night_model` on IR cameras that watch for animals, not as a person verifier on streets.
  - `mdv6-yolov10-c` is the cheapest at 34–47 ms.
- **The ExDark RF-DETRs are by far the best on ExDark, with almost no person false positives**, but that is partly
  memorised (see the note under the table).
  - On real IR they find only about half of the dogs and cats, and few cars.
  - They need f32 (289 ms for small, 158 ms for nano).
  - Nano costs 0.06 person AP50 on exdark-night for 1.8x the speed.
  - They are non-commercial, so they are a benchmark reference rather than a recommendation.

The person columns of nir-night are the person-free false positives only: its 2 person boxes are too few to score.

