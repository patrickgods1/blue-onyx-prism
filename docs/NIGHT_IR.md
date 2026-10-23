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
