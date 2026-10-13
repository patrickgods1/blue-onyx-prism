//! Accuracy scoring of a finished run: AP against real or pseudo ground truth (`score`), the
//! per-image TP/FP drill-down flags, frame-level ROC curves and the pseudo ground truth run
//! on a reference device. Pure over stored predictions apart from `pseudo_ground_truth`.

use super::*;

/// Score the run: overall accuracy (over the images with real ground truth, else those with
/// pseudo ground truth), per dataset and per tag; marks each stored prediction TP/FP and each
/// ground-truth object matched/missed in `per_image`.
pub(super) fn score(
    job: &Job,
    all: &[(&ImageSet, &images::SetImage)],
    per_image: &mut [ImageRun],
    eval_preds: &[Vec<PredBox>],
    threshold: f32,
    objective: threshold::Objective,
) -> (Option<AccuracyReport>, Option<String>) {
    let map = ClassMap::for_model(&job.classes);
    if map.is_empty() {
        return (
            None,
            Some("the model has none of the scored CCTV classes".to_string()),
        );
    }
    if !all.iter().any(|(_, i)| i.gt.is_some()) {
        return (
            None,
            Some("no ground truth (and no pseudo ground truth) for these images".to_string()),
        );
    }
    // Per image: the eval input (None = no ground truth).
    let mut evals: Vec<Option<EvalImage>> = Vec::with_capacity(all.len());
    // Per image: the classes its dataset annotates (frame samples).
    let mut fclasses: Vec<Vec<String>> = Vec::with_capacity(all.len());
    for (k, (set, img)) in all.iter().enumerate() {
        // `per_image` and `eval_preds` are parallel to `all`; a missing entry scores as
        // "no ground truth" rather than panicking.
        let (Some(gt), Some(preds), Some(run)) = (&img.gt, eval_preds.get(k), per_image.get_mut(k))
        else {
            evals.push(None);
            fclasses.push(Vec::new());
            continue;
        };
        let (eval, classes) = score_image(&map, set, gt, preds, run, threshold);
        fclasses.push(classes);
        evals.push(Some(eval));
    }
    let scored = Scored::new(all, evals, fclasses);
    let chosen = &scored.chosen;
    let relative = scored.relative;
    let overall = metrics::evaluate(&scored.subset(chosen), &map.classes, threshold);
    let all_classes = all_classes_ap(job, &map, &scored, eval_preds, threshold);
    let ground_truth = ground_truth_desc(all, relative);
    // Per dataset (each with its own kind of ground truth).
    let by_dataset = scored
        .dataset_groups()
        .into_iter()
        .map(|(s, ks)| {
            let sum = metrics::evaluate(&scored.subset(&ks), &map.classes, threshold);
            Breakdown::of(
                &s.id,
                &sum,
                !s.ground_truth.is_real(),
                scored.frame_roc(&ks, &map.classes, 0).auc,
            )
        })
        .collect();
    // Per tag, over the images of the overall score.
    let mut tags: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for &k in chosen {
        let Some((_, img)) = all.get(k) else { continue };
        for t in &img.tags {
            tags.entry(t.as_str()).or_default().push(k);
        }
    }
    let by_tag = tags
        .into_iter()
        .map(|(t, ks)| {
            let sum = metrics::evaluate(&scored.subset(&ks), &map.classes, threshold);
            Breakdown::of(
                t,
                &sum,
                relative,
                scored.frame_roc(&ks, &map.classes, 0).auc,
            )
        })
        .collect();
    let sweep = scored.threshold_sweep(&map.classes, threshold, objective);
    (
        Some(AccuracyReport {
            ground_truth,
            relative,
            classes: map.classes.clone(),
            threshold,
            overall,
            by_dataset,
            by_tag,
            all_classes,
            sweep: Some(sweep),
            frame_roc: Some(scored.frame_roc(chosen, &map.classes, 0)),
        }),
        None,
    )
}

/// Score one image with ground truth `gt`: flags its stored (production) predictions TP/FP
/// and its ground-truth objects matched/missed (drill-down), sets its counts at `threshold`.
/// Returns the image's eval input and the classes its dataset annotates.
pub(super) fn score_image(
    map: &ClassMap,
    set: &ImageSet,
    gt: &[GtBox],
    eval_preds: &[PredBox],
    run: &mut ImageRun,
    threshold: f32,
) -> (EvalImage, Vec<String>) {
    let (mapped, eval, allowed) = eval_input(map, set.scored_labels.as_deref(), gt, eval_preds);
    let classes = frame_classes(map, allowed.as_deref());
    let gt_list = eval.gt.clone();
    // Drill-down flags on the stored (production) predictions.
    let scored: Vec<(usize, PredBox)> = run
        .preds
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            map.pred_class(&p.label)
                .filter(|c| allowed.as_ref().is_none_or(|a| a.iter().any(|x| x == c)))
                .map(|c| {
                    (
                        i,
                        PredBox {
                            label: c.to_string(),
                            confidence: p.confidence,
                            bbox: p.bbox,
                        },
                    )
                })
        })
        .collect();
    let preds: Vec<PredBox> = scored.iter().map(|(_, p)| p.clone()).collect();
    let m = metrics::match_image(&preds, &gt_list, 0.5);
    for ((i, _), r) in scored.iter().zip(&m.pred) {
        if let Some(p) = run.preds.get_mut(*i) {
            p.tp = *r;
        }
    }
    let mut j = 0;
    run.gt_matched = mapped
        .iter()
        .map(|g| match g {
            Some(g) => {
                // One `gt_matched` entry per mapped ground-truth box, in order.
                let hit = m.gt_matched.get(j).copied().unwrap_or_default();
                j += 1;
                (!g.ignore).then_some(hit)
            }
            None => None,
        })
        .collect();
    run.counts = Some(metrics::image_counts(&eval, threshold).0);
    (eval, classes)
}

/// AP over every class of the model on the images of the overall score; None when the model
/// has no classes besides the scored ones.
pub(super) fn all_classes_ap(
    job: &Job,
    map: &ClassMap,
    scored: &Scored,
    eval_preds: &[Vec<PredBox>],
    threshold: f32,
) -> Option<AllClasses> {
    let all = scored.all;
    let all_map = ClassMap::all_classes(&job.classes);
    (all_map.classes.len() > map.classes.len()).then(|| {
        let imgs: Vec<EvalImage> = scored
            .chosen
            .iter()
            .filter_map(|&k| {
                let (set, img) = all.get(k)?;
                let gt = img.gt.as_ref()?;
                let preds = eval_preds.get(k)?;
                Some(eval_input(&all_map, set.scored_labels.as_deref(), gt, preds).1)
            })
            .collect();
        let s = metrics::evaluate(&imgs, &all_map.classes, threshold);
        AllClasses {
            classes: s.per_class.iter().filter(|c| c.gt > 0).count(),
            ap50: s.ap50,
            ap50_95: s.ap50_95,
        }
    })
}

/// [`AccuracyReport::ground_truth`]: "ground truth", or the pseudo ground truth references.
pub(super) fn ground_truth_desc(all: &[(&ImageSet, &images::SetImage)], relative: bool) -> String {
    if relative {
        let refs: Vec<String> = all
            .iter()
            .filter_map(|(s, _)| match &s.ground_truth {
                GroundTruth::Pseudo(r) => Some(r.clone()),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        format!("relative to {}, not ground truth", refs.join(", "))
    } else {
        "ground truth".to_string()
    }
}

/// The scored images of a run: evaluation inputs and frame samples per image (None / empty =
/// image not scored) and the images of the overall score.
pub(super) struct Scored<'a> {
    all: &'a [(&'a ImageSet, &'a images::SetImage)],
    evals: Vec<Option<EvalImage>>,
    /// Per image: the classes its dataset annotates.
    frame_classes: Vec<Vec<String>>,
    frames: Vec<Vec<roc::FrameSample>>,
    /// The images with real ground truth, else every scored image (pseudo ground truth).
    chosen: Vec<usize>,
    /// `chosen` is scored against pseudo ground truth.
    relative: bool,
}

impl<'a> Scored<'a> {
    fn new(
        all: &'a [(&'a ImageSet, &'a images::SetImage)],
        evals: Vec<Option<EvalImage>>,
        frame_classes: Vec<Vec<String>>,
    ) -> Self {
        let real: Vec<usize> = all
            .iter()
            .zip(&evals)
            .enumerate()
            .filter(|(_, ((s, _), e))| e.is_some() && s.ground_truth.is_real())
            .map(|(k, _)| k)
            .collect();
        let (chosen, relative) = if real.is_empty() {
            (
                evals
                    .iter()
                    .take(all.len())
                    .enumerate()
                    .filter(|(_, e)| e.is_some())
                    .map(|(k, _)| k)
                    .collect::<Vec<_>>(),
                true,
            )
        } else {
            (real, false)
        };
        let frames: Vec<Vec<roc::FrameSample>> = evals
            .iter()
            .zip(&frame_classes)
            .map(|(e, c)| e.as_ref().map_or(Vec::new(), |e| roc::frame_samples(e, c)))
            .collect();
        Self {
            all,
            evals,
            frame_classes,
            frames,
            chosen,
            relative,
        }
    }

    /// The eval inputs of the images `ks` (scored ones only).
    fn subset(&self, ks: &[usize]) -> Vec<EvalImage> {
        ks.iter()
            .filter_map(|&k| self.evals.get(k).cloned().flatten())
            .collect()
    }

    /// Frame ROC over the frame samples of the images `ks`.
    fn frame_roc(&self, ks: &[usize], classes: &[String], curve: usize) -> roc::FrameRoc {
        roc::FrameRoc::of(
            ks.iter().filter_map(|&k| self.frames.get(k)).flatten(),
            classes,
            curve,
        )
    }

    /// The scored images of each dataset, in first-seen order (datasets without any omitted).
    fn dataset_groups(&self) -> Vec<(&'a ImageSet, Vec<usize>)> {
        let all = self.all;
        let mut out = Vec::new();
        let mut seen: Vec<&str> = Vec::new();
        for (s, _) in all {
            if seen.contains(&s.id.as_str()) {
                continue;
            }
            seen.push(&s.id);
            let ks: Vec<usize> = all
                .iter()
                .zip(&self.evals)
                .enumerate()
                .filter(|(_, ((t, _), e))| t.id == s.id && e.is_some())
                .map(|(k, _)| k)
                .collect();
            if !ks.is_empty() {
                out.push((*s, ks));
            }
        }
        out
    }

    /// Threshold curves of the run: overall (over `chosen`), per dataset, per tag of
    /// [`threshold::SWEEP_TAGS`], for small objects and per class.
    fn threshold_sweep(
        &self,
        classes: &[String],
        configured: f32,
        objective: threshold::Objective,
    ) -> threshold::ThresholdSweep {
        use metrics::{SweepImage, SweepScope};
        let (all, chosen, relative) = (self.all, &self.chosen[..], self.relative);
        let curves = CurveMaker {
            matched: self
                .evals
                .iter()
                .zip(&self.frame_classes)
                .map(|(e, c)| e.as_ref().map(|e| SweepImage::of(e).with_frames(e, c)))
                .collect(),
            objectives: threshold::objectives(objective),
            classes,
            configured,
        };
        let mut by_dataset: Vec<threshold::Curve> = self
            .dataset_groups()
            .into_iter()
            .map(|(s, ks)| curves.curve(&s.id, &ks, SweepScope::All, !s.ground_truth.is_real()))
            .collect();
        let mut by_tag = Vec::new();
        for tag in threshold::SWEEP_TAGS {
            let ks: Vec<usize> = chosen
                .iter()
                .copied()
                .filter(|&k| {
                    all.get(k)
                        .is_some_and(|(_, i)| i.tags.iter().any(|t| t == tag))
                })
                .collect();
            if !ks.is_empty() {
                by_tag.push(curves.curve(tag, &ks, SweepScope::All, relative));
            }
        }
        let small = curves.curve(threshold::SMALL_KEY, chosen, SweepScope::Small, relative);
        if small.gt > 0 {
            by_tag.push(small);
        }
        let mut overall = curves.curve("overall", chosen, SweepScope::All, relative);
        let imgs = curves.images(chosen);
        overall.exact =
            metrics::Breakpoints::new(&imgs, SweepScope::All).decimated(threshold::EXACT_POINTS);
        let roc = curves.roc_of(&imgs, SweepScope::All, roc::ROC_POINTS);
        let mut per_class: Vec<threshold::Curve> = classes
            .iter()
            .map(|c| curves.curve(c, chosen, SweepScope::Class(c), relative))
            .filter(|c| c.gt > 0)
            .collect();
        // Group curves keep frame rates at the configured threshold and in their picks; their
        // 0.05 grids drop them (only the overall grid is charted and tabled with frame rates),
        // which keeps `benchmark.json` small.
        for c in by_dataset
            .iter_mut()
            .chain(by_tag.iter_mut())
            .chain(per_class.iter_mut())
        {
            for p in &mut c.points {
                p.frame = None;
            }
        }
        threshold::ThresholdSweep {
            configured,
            overall,
            by_dataset,
            by_tag,
            per_class,
            roc: Some(roc),
        }
    }
}

/// Builds the threshold curves of [`Scored::threshold_sweep`].
pub(super) struct CurveMaker<'c> {
    /// Per image: the sweep input (None = image not scored).
    matched: Vec<Option<metrics::SweepImage>>,
    objectives: Vec<threshold::Objective>,
    classes: &'c [String],
    configured: f32,
}

impl CurveMaker<'_> {
    /// The sweep inputs of the scored images among `ks`.
    fn images(&self, ks: &[usize]) -> Vec<&metrics::SweepImage> {
        ks.iter()
            .filter_map(|&k| self.matched.get(k).and_then(Option::as_ref))
            .collect()
    }

    fn roc_of(
        &self,
        imgs: &[&metrics::SweepImage],
        scope: metrics::SweepScope,
        curve: usize,
    ) -> roc::FrameRoc {
        let cls: Vec<String> = match scope {
            metrics::SweepScope::Class(c) => vec![c.to_string()],
            _ => self.classes.to_vec(),
        };
        roc::FrameRoc::of(imgs.iter().flat_map(|i| i.frames.iter()), &cls, curve)
    }

    /// The curve `key` over the images `ks`.
    fn curve(
        &self,
        key: &str,
        ks: &[usize],
        scope: metrics::SweepScope,
        relative: bool,
    ) -> threshold::Curve {
        let imgs = self.images(ks);
        let bp = metrics::Breakpoints::new(&imgs, scope);
        threshold::Curve {
            key: key.to_string(),
            images: imgs.len(),
            gt: bp.npos,
            relative,
            points: metrics::SWEEP_THRESHOLDS
                .iter()
                .map(|&t| bp.point_at(t))
                .collect(),
            configured: bp.point_at(self.configured),
            picks: self
                .objectives
                .iter()
                .filter_map(|o| threshold::exact_pick(&bp, *o))
                .collect(),
            exact: Vec::new(),
            roc_auc: match scope {
                metrics::SweepScope::Small => None,
                _ => self.roc_of(&imgs, scope, 0).auc,
            },
        }
    }
}

/// Fill the ground truth of every set without it with the detections of `reference` on
/// `device`: confidence >= [`PSEUDO_GT_CONFIDENCE`] are objects, down to
/// [`PSEUDO_IGNORE_CONFIDENCE`] ignore regions. Returns the reference description
/// ("rt-detrv2-x on openvino:cpu"), or None when every set already had ground truth.
pub(crate) fn pseudo_ground_truth(
    bench: &Bench,
    runtimes: &Mutex<Runtimes>,
    reference: &Job,
    device: Device,
    sets: &mut [ImageSet],
) -> Result<Option<String>> {
    let subset: Vec<ImageSet> = sets.iter().filter(|s| !s.annotated()).cloned().collect();
    if subset.is_empty() {
        return Ok(None);
    }
    let mut job = reference.clone();
    job.confidence_threshold = Some(PSEUDO_IGNORE_CONFIDENCE);
    job.object_filter = Some(Vec::new());
    let pb = Bench {
        sets: &subset,
        warmup: 0,
        repeat: 1,
        params: PostParams {
            confidence_threshold: PSEUDO_IGNORE_CONFIDENCE,
            nms_iou: bench.params.nms_iou,
        },
        object_filter: &[],
        cache_dir: bench.cache_dir,
        cancel: bench.cancel,
        progress: bench.progress,
        weights: bench.weights,
        threshold_objective: bench.threshold_objective,
        accuracy_metric: bench.accuracy_metric,
        // Every reference detection >= PSEUDO_IGNORE_CONFIDENCE becomes ground truth, not
        // only the 100 most confident (BM-10).
        max_stored_preds: usize::MAX,
    };
    let run = pb
        .run_exact(runtimes, &job, device)
        .with_context(|| format!("pseudo ground truth with {} on {device}", job.name))?;
    let desc = format!("{} on {device}", job.name);
    // `run.per_image` holds one entry per image of `subset`, in order.
    let mut runs = run.per_image.iter();
    for set in sets.iter_mut().filter(|s| !s.annotated()) {
        for img in &mut set.images {
            let r = runs
                .next()
                .context("pseudo ground truth run returned fewer images than requested")?;
            img.gt = Some(pseudo_gt_boxes(&r.preds));
        }
        set.ground_truth = GroundTruth::Pseudo(desc.clone());
    }
    Ok(Some(desc))
}

/// Pseudo ground truth from a reference run's predictions of one image: confidence >=
/// [`PSEUDO_GT_CONFIDENCE`] are objects, the rest ignore regions.
pub(super) fn pseudo_gt_boxes(preds: &[ImagePred]) -> Vec<GtBox> {
    preds
        .iter()
        .map(|p| GtBox {
            label: metrics::canonical_label(&p.label),
            bbox: p.bbox,
            ignore: p.confidence < PSEUDO_GT_CONFIDENCE,
            crowd: false,
        })
        .collect()
}

/// CPU devices whose detections serve as the agreement reference, in order of preference.
pub(crate) const REFERENCE_DEVICES: [Device; 2] = [
    Device::OPENVINO_CPU,
    Device {
        runtime: Runtime::Ort,
        target: Target::Cpu,
        index: None,
    },
];

/// The first runnable CPU device of `sel` (pseudo ground truth runs there unless
/// `benchmark.devices` limits it, see [`pseudo_gt_device`]).
pub(crate) fn reference_device(sel: &Selection) -> Option<Device> {
    REFERENCE_DEVICES
        .into_iter()
        .find(|d| sel.option(d).is_some_and(|o| o.runnable))
}

/// The device pseudo ground truth runs on. `allowed` (`benchmark.devices`; None or empty = no
/// limit) keeps it within the user's devices, so a GPU-only list never runs anything on a CPU:
/// a listed runnable [`REFERENCE_DEVICES`] entry, else the first listed runnable device in the
/// list's order. Without a limit, [`reference_device`]. None when nothing allowed can run.
pub(crate) fn pseudo_gt_device(sel: &Selection, allowed: Option<&[Device]>) -> Option<Device> {
    let allowed = match allowed {
        Some(a) if !a.is_empty() => a,
        _ => return reference_device(sel),
    };
    let runnable = |d: &Device| sel.option(d).is_some_and(|o| o.runnable);
    REFERENCE_DEVICES
        .into_iter()
        .find(|d| allowed.contains(d) && runnable(d))
        .or_else(|| allowed.iter().copied().find(|d| runnable(d)))
}

/// Why [`pseudo_gt_device`] found no device.
pub(crate) fn no_pseudo_gt_device(allowed: Option<&[Device]>) -> String {
    match allowed {
        Some(a) if !a.is_empty() => format!(
            "none of the benchmark devices ({}) can run the reference model for pseudo ground truth",
            a.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => "no CPU device can run the reference model for pseudo ground truth".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::run::{Sample, image_run};
    use super::*;

    /// A selection with these (spec, runnable) options.
    fn selection_of(opts: &[(&str, bool)]) -> Selection {
        let options: Vec<_> = opts
            .iter()
            .map(|&(s, runnable)| crate::backend::select::DeviceOption {
                spec: dev(s),
                label: s.to_string(),
                runnable,
                reason: (!runnable).then(|| "not runnable".to_string()),
                download: None,
            })
            .collect();
        let auto = options
            .iter()
            .filter(|o| o.runnable)
            .map(|o| o.spec)
            .collect();
        Selection {
            options,
            auto,
            auto_measured: false,
        }
    }

    fn dev(s: &str) -> Device {
        *spec::parse(s).unwrap().device().unwrap()
    }

    fn machine() -> Selection {
        selection_of(&[
            ("openvino:gpu.0", false),
            ("openvino:gpu.1", true),
            ("ort:directml:0", true),
            ("openvino:cpu", true),
            ("ort:cpu", true),
        ])
    }

    #[test]
    fn pseudo_gt_device_without_limit_is_cpu() {
        let sel = machine();
        assert_eq!(pseudo_gt_device(&sel, None), Some(Device::OPENVINO_CPU));
        assert_eq!(
            pseudo_gt_device(&sel, Some(&[])),
            Some(Device::OPENVINO_CPU)
        );
        // No runnable CPU and no limit: nothing (never a GPU by surprise).
        let gpu_only = selection_of(&[("openvino:gpu.1", true), ("openvino:cpu", false)]);
        assert_eq!(pseudo_gt_device(&gpu_only, None), None);
    }

    #[test]
    fn pseudo_gt_device_gpu_only_list_takes_first_runnable_listed() {
        let sel = machine();
        let list = [
            dev("openvino:gpu.0"),
            dev("ort:directml:0"),
            dev("openvino:gpu.1"),
        ];
        assert_eq!(
            pseudo_gt_device(&sel, Some(&list)),
            Some(dev("ort:directml:0"))
        );
        let list = [dev("openvino:gpu.1"), dev("ort:directml:0")];
        assert_eq!(
            pseudo_gt_device(&sel, Some(&list)),
            Some(dev("openvino:gpu.1"))
        );
    }

    #[test]
    fn pseudo_gt_device_prefers_a_listed_cpu() {
        let sel = machine();
        let list = [dev("openvino:gpu.1"), dev("ort:cpu")];
        assert_eq!(pseudo_gt_device(&sel, Some(&list)), Some(dev("ort:cpu")));
        let list = [dev("ort:cpu"), dev("openvino:cpu")];
        assert_eq!(
            pseudo_gt_device(&sel, Some(&list)),
            Some(Device::OPENVINO_CPU)
        );
        // A listed CPU that cannot run falls back to the listed GPU, not another CPU.
        let sel = selection_of(&[
            ("openvino:gpu.1", true),
            ("openvino:cpu", false),
            ("ort:cpu", true),
        ]);
        let list = [dev("openvino:cpu"), dev("openvino:gpu.1")];
        assert_eq!(
            pseudo_gt_device(&sel, Some(&list)),
            Some(dev("openvino:gpu.1"))
        );
    }

    #[test]
    fn pseudo_gt_device_nothing_runnable_is_none() {
        let sel = machine();
        let list = [dev("openvino:gpu.0"), dev("ort:cuda:0")];
        assert_eq!(pseudo_gt_device(&sel, Some(&list)), None);
        let msg = no_pseudo_gt_device(Some(&list));
        assert!(msg.contains("openvino:gpu.0, ort:cuda:0"), "{msg}");
        assert!(no_pseudo_gt_device(None).contains("CPU"));
    }

    /// A score over synthetic per-image predictions: ground truth, pseudo ground truth and the
    /// drill-down flags.
    #[test]
    fn scoring_real_and_pseudo_ground_truth() {
        let mut set = ImageSet::sample();
        set.id = "real".into();
        set.ground_truth = GroundTruth::Manifest;
        set.images[0].gt = Some(vec![
            GtBox {
                label: "dog".into(),
                bbox: [0.0, 0.0, 100.0, 100.0],
                ignore: false,
                crowd: false,
            },
            GtBox {
                label: "umbrella".into(),
                bbox: [0.0, 0.0, 10.0, 10.0],
                ignore: false,
                crowd: false,
            },
            GtBox {
                label: "car".into(),
                bbox: [300.0, 300.0, 400.0, 400.0],
                ignore: false,
                crowd: false,
            },
        ]);
        let mut pseudo = set.clone();
        pseudo.id = "pseudo".into();
        pseudo.ground_truth = GroundTruth::Pseudo("ref on openvino:cpu".into());
        let job = Job {
            name: "m".into(),
            path: "m.onnx".into(),
            family: ModelFamilyKind::Yolo5,
            classes: crate::model::classes::coco80(),
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
        };
        let pred = |l: &str, c: f32, b: [f32; 4]| PredBox {
            label: l.into(),
            confidence: c,
            bbox: b,
        };
        let eval = vec![
            pred("dog", 0.9, [1.0, 1.0, 100.0, 100.0]),
            pred("person", 0.2, [500.0, 0.0, 600.0, 100.0]),
        ];
        let stored = |e: &[PredBox]| {
            e.iter()
                .filter(|p| p.confidence >= 0.5)
                .map(|p| ImagePred {
                    label: p.label.clone(),
                    confidence: p.confidence,
                    bbox: p.bbox,
                    tp: None,
                })
                .collect::<Vec<_>>()
        };
        let all = vec![(&set, &set.images[0]), (&pseudo, &pseudo.images[0])];
        let mut per_image: Vec<ImageRun> = all
            .iter()
            .map(|(s, i)| ImageRun {
                set: s.id.clone(),
                file: i.file.clone(),
                total_ms: 1.0,
                infer_ms: 1.0,
                preds: stored(&eval),
                gt_matched: vec![],
                counts: None,
            })
            .collect();
        let (acc, note) = score(
            &job,
            &all,
            &mut per_image,
            &[eval.clone(), eval],
            0.5,
            Default::default(),
        );
        assert!(note.is_none());
        let acc = acc.unwrap();
        // Real ground truth wins for the overall score.
        assert!(!acc.relative);
        assert_eq!(acc.ground_truth, "ground truth");
        assert_eq!(acc.overall.images, 1);
        assert_eq!(
            (
                acc.overall.counts.tp,
                acc.overall.counts.fp,
                acc.overall.counts.fn_
            ),
            (1, 0, 1)
        );
        // dog AP 1, car AP 0 -> 0.5.
        assert!((acc.overall.ap50.unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(acc.by_dataset.len(), 2);
        assert!(acc.by_dataset[1].relative);
        assert!(acc.by_tag.iter().any(|t| t.key.starts_with("res:")));
        // Drill-down: the dog is a TP, the umbrella is not scored, the car is missed.
        assert_eq!(per_image[0].preds[0].tp, Some(true));
        assert_eq!(per_image[0].gt_matched, [Some(true), None, Some(false)]);

        // Pseudo only: relative, labelled with the reference.
        let all = vec![(&pseudo, &pseudo.images[0])];
        let mut pi = vec![per_image[1].clone()];
        let ev = vec![pred("dog", 0.9, [1.0, 1.0, 100.0, 100.0])];
        let (acc, _) = score(&job, &all, &mut pi, &[ev], 0.5, Default::default());
        let acc = acc.unwrap();
        assert!(acc.relative);
        assert_eq!(
            acc.ground_truth,
            "relative to ref on openvino:cpu, not ground truth"
        );

        // No scored classes / no ground truth.
        let pkg = Job {
            classes: vec!["package".into()],
            ..job.clone()
        };
        let (acc, note) = score(&pkg, &all, &mut pi, &[vec![]], 0.5, Default::default());
        assert!(acc.is_none() && note.unwrap().contains("CCTV"));
        let bare = ImageSet::sample();
        let all = vec![(&bare, &bare.images[0])];
        let (acc, note) = score(&job, &all, &mut pi, &[vec![]], 0.5, Default::default());
        assert!(acc.is_none() && note.unwrap().contains("no ground truth"));
    }

    #[test]
    fn scored_labels_limit_a_set() {
        // A vehicles-only set: the person prediction is neither TP nor FP, person ground truth
        // is not a miss.
        let mut set = ImageSet::sample();
        set.ground_truth = GroundTruth::Manifest;
        set.scored_labels = Some(vec!["car".into(), "motorbike".into()]);
        set.images[0].gt = Some(vec![
            GtBox {
                label: "car".into(),
                bbox: [0.0, 0.0, 100.0, 100.0],
                ignore: false,
                crowd: false,
            },
            GtBox {
                label: "person".into(),
                bbox: [200.0, 0.0, 300.0, 100.0],
                ignore: false,
                crowd: false,
            },
        ]);
        let job = Job {
            name: "m".into(),
            path: "m.onnx".into(),
            family: ModelFamilyKind::Yolo5,
            classes: crate::model::classes::coco80(),
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
        };
        let p = |l: &str, b: [f32; 4]| PredBox {
            label: l.into(),
            confidence: 0.9,
            bbox: b,
        };
        let eval = vec![
            p("car", [0.0, 0.0, 100.0, 100.0]),
            p("person", [500.0, 0.0, 600.0, 100.0]),
            // A low-confidence false positive: only in the threshold curve below 0.5.
            PredBox {
                confidence: 0.3,
                ..p("car", [700.0, 0.0, 800.0, 100.0])
            },
        ];
        let all = vec![(&set, &set.images[0])];
        let mut pi = vec![ImageRun {
            set: set.id.clone(),
            file: "f".into(),
            total_ms: 1.0,
            infer_ms: 1.0,
            preds: eval
                .iter()
                .map(|e| ImagePred {
                    label: e.label.clone(),
                    confidence: e.confidence,
                    bbox: e.bbox,
                    tp: None,
                })
                .collect(),
            gt_matched: vec![],
            counts: None,
        }];
        let (acc, _) = score(&job, &all, &mut pi, &[eval], 0.5, Default::default());
        let acc = acc.unwrap();
        assert_eq!(
            (
                acc.overall.counts.tp,
                acc.overall.counts.fp,
                acc.overall.counts.fn_
            ),
            (1, 0, 0)
        );
        assert_eq!(pi[0].preds[1].tp, None);
        assert_eq!(pi[0].gt_matched, [Some(true), None]);
        // All-class AP is reported for a COCO model (more classes than the CCTV ones).
        let all_c = acc.all_classes.unwrap();
        assert_eq!(all_c.classes, 1);
        assert!((all_c.ap50.unwrap() - 1.0).abs() < 1e-12);

        // Threshold sweep: the person prediction and person object are out of scope.
        let sw = acc.sweep.unwrap();
        assert_eq!(sw.configured, 0.5);
        let at = |t: f32| {
            let p = sw.overall.points.iter().find(|p| p.threshold == t).unwrap();
            (p.counts.tp, p.counts.fp, p.counts.fn_)
        };
        assert_eq!(at(0.25), (1, 1, 0));
        assert_eq!(at(0.5), (1, 0, 0));
        assert_eq!(at(0.95), (0, 0, 1));
        assert_eq!(sw.overall.gt, 1);
        assert_eq!(sw.overall.configured.counts, acc.overall.counts);
        assert_eq!(sw.by_dataset.len(), 1);
        let classes: Vec<&str> = sw.per_class.iter().map(|c| c.key.as_str()).collect();
        assert_eq!(classes, ["car"]);
        assert!(!sw.overall.exact.is_empty());
        // F1 1 for thresholds in (0.3, 0.9]: midpoint 0.6.
        let best = sw.overall.best(threshold::Objective::F1).unwrap();
        assert_eq!(best.exact_threshold, Some(0.9));
        assert_eq!(best.plateau, Some([0.3, 0.9]));
        assert_eq!(best.threshold, 0.6);
        assert_eq!(best.point.f1, Some(1.0));
        // Every standard objective has an exact pick (Youden too: the car frame is positive,
        // the motorcycle frame negative).
        assert_eq!(sw.overall.picks.len(), 4);
        // Frame samples only for the set's scored labels: person is not a sample at all.
        let roc = acc.frame_roc.as_ref().unwrap();
        let keys: Vec<(&str, usize, usize)> = roc
            .per_class
            .iter()
            .map(|c| (c.class.as_str(), c.positives, c.negatives))
            .collect();
        assert_eq!(keys, [("car", 1, 0), ("motorcycle", 0, 1)]);
        // One-sided per class: undefined, so no macro AUC.
        assert_eq!(roc.auc, None);
    }

    /// Frame-level ROC through scoring: an IPcam-style model (person + vehicle group), ignore
    /// regions, and a vehicles-only set.
    #[test]
    fn frame_roc_through_scoring() {
        let g = |l: &str, ignore: bool| GtBox {
            label: l.into(),
            bbox: [0.0, 0.0, 100.0, 100.0],
            ignore,
            crowd: false,
        };
        let p = |l: &str, c: f32| PredBox {
            label: l.into(),
            confidence: c,
            bbox: [0.0, 0.0, 100.0, 100.0],
        };
        let img = |file: &str, gt: Vec<GtBox>| images::SetImage {
            file: file.into(),
            path: None,
            width: 640,
            height: 480,
            tags: vec!["night".into()],
            gt: Some(gt),
        };
        let mut set = ImageSet::sample();
        set.id = "s".into();
        set.ground_truth = GroundTruth::Manifest;
        set.images = vec![
            img("a", vec![g("person", false), g("truck", false)]), // person+, vehicle+ (truck)
            img("b", vec![g("car", false)]),                       // person-, vehicle+
            img("c", vec![]),                                      // person-, vehicle-
            img("d", vec![g("person", true)]),                     // person excluded, vehicle-
        ];
        // Vehicles-only set: person is not annotated there, so never a negative.
        let mut cars = set.clone();
        cars.id = "cars".into();
        cars.scored_labels = Some(vec!["car".into(), "truck".into(), "bus".into()]);
        cars.images = vec![img("e", vec![g("bus", false)]), img("f", vec![])];
        let job = Job {
            name: "ipcam".into(),
            path: "m.onnx".into(),
            family: ModelFamilyKind::Yolo5,
            classes: vec!["person".into(), "vehicle".into(), "unknown".into()],
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
        };
        let eval = vec![
            vec![p("person", 0.9), p("vehicle", 0.4)],
            vec![p("person", 0.6), p("vehicle", 0.8)],
            vec![p("vehicle", 0.5)],
            vec![p("person", 0.95)],
            vec![p("person", 0.99), p("vehicle", 0.7)],
            vec![p("person", 0.99), p("vehicle", 0.2)],
        ];
        let all: Vec<(&ImageSet, &images::SetImage)> = set
            .images
            .iter()
            .map(|i| (&set, i))
            .chain(cars.images.iter().map(|i| (&cars, i)))
            .collect();
        let mut pi: Vec<ImageRun> = all
            .iter()
            .map(|(s, i)| ImageRun {
                set: s.id.clone(),
                file: i.file.clone(),
                total_ms: 1.0,
                infer_ms: 1.0,
                preds: vec![],
                gt_matched: vec![],
                counts: None,
            })
            .collect();
        let (acc, _) = score(
            &job,
            &all,
            &mut pi,
            &eval,
            0.5,
            threshold::Objective::Youden,
        );
        let acc = acc.unwrap();
        let roc = acc.frame_roc.clone().unwrap();
        // person: positives {a .9}; negatives {b .6, c 0}; d excluded (ignore only); the cars
        // set has no person samples -> AUC 1.
        let person = &roc.per_class[0];
        assert_eq!((person.positives, person.negatives), (1, 2));
        assert_eq!(person.auc, Some(1.0));
        // vehicle: positives {a .4, b .8, e .7}; negatives {c .5, d 0, f .2}:
        // pairs won: .4>{0,.2} 2, .8>all 3, .7>all 3 -> 8 / 9.
        let vehicle = &roc.per_class[1];
        assert_eq!((vehicle.positives, vehicle.negatives), (3, 3));
        assert!((vehicle.auc.unwrap() - 8.0 / 9.0).abs() < 1e-12);
        assert!((roc.auc.unwrap() - (1.0 + 8.0 / 9.0) / 2.0).abs() < 1e-12);
        assert!((acc.roc_auc().unwrap() - roc.auc.unwrap()).abs() < 1e-15);
        assert_eq!(
            acc.score(grade::AccuracyMetric::RocAuc),
            acc.roc_auc(),
            "the ROC AUC metric grades on the macro AUC"
        );
        // Per dataset and per tag.
        let s = acc.by_dataset.iter().find(|b| b.key == "s").unwrap();
        // s alone: person 1.0; vehicle positives {.4, .8} vs negatives {.5, 0}: 3 / 4.
        assert!((s.roc_auc.unwrap() - (1.0 + 0.75) / 2.0).abs() < 1e-12);
        assert!(
            acc.by_tag
                .iter()
                .any(|b| b.key == "night" && b.roc_auc.is_some())
        );
        // Sweep: pooled frame rates and a Youden pick; the curve ends at (1, 1).
        let sw = acc.sweep.as_ref().unwrap();
        let curve = &sw.roc.as_ref().unwrap().curve;
        assert_eq!(
            (curve.last().unwrap().tpr, curve.last().unwrap().fpr),
            (1.0, 1.0)
        );
        let f = sw.overall.configured.frame.unwrap();
        // At 0.5: positives a.person .9, b.vehicle .8, e.vehicle .7 alert, a.vehicle .4 does
        // not (3 of 4); negatives b.person .6, c.vehicle .5 alert (2 of 5).
        assert_eq!((f.tp, f.positives, f.fp, f.negatives), (3, 4, 2, 5));
        let y = sw.overall.best(threshold::Objective::Youden).unwrap();
        assert!(y.met);
        // Brute force over the 0.01 grid: no threshold beats the exact optimum.
        let bp_best = y.exact_score.unwrap();
        for k in 5..=100 {
            let t = k as f32 / 100.0;
            let j = sw
                .overall
                .points
                .iter()
                .find(|p| (p.threshold - t).abs() < 1e-6)
                .and_then(|p| p.frame?.youden());
            if let Some(j) = j {
                assert!(j <= bp_best + 1e-12, "{t}: {j} > {bp_best}");
            }
        }
        // Per class curves carry the class AUC.
        let pc = sw.per_class.iter().find(|c| c.key == "vehicle").unwrap();
        assert!((pc.roc_auc.unwrap() - 8.0 / 9.0).abs() < 1e-12);
        // Group grids carry no frame rates (size); their configured points and picks do.
        assert!(pc.points.iter().all(|p| p.frame.is_none()));
        assert!(pc.configured.frame.is_some());
        assert!(sw.overall.points.iter().all(|p| p.frame.is_some()));
    }

    /// The stored drill-down predictions are capped, but the pseudo ground truth run keeps
    /// every reference detection, so a crowded image's 101st object is still ground truth
    /// (BM-10).
    #[test]
    fn pseudo_ground_truth_keeps_every_reference_detection() {
        let set = ImageSet::sample();
        let img = &set.images[0];
        let n = MAX_STORED_PREDS + 50;
        // Confidences from 0.99 down to 0.30: some objects, the rest ignore regions.
        let predictions: Vec<Prediction> = (0..n)
            .map(|i| Prediction {
                x_min: i,
                y_min: 0,
                x_max: i + 10,
                y_max: 10,
                confidence: 0.99 - 0.69 * i as f32 / (n - 1) as f32,
                label: "person".into(),
            })
            .collect();
        let last = Sample {
            stages: [1.0; 4],
            predictions,
            eval: Vec::new(),
        };
        let (capped, _) = image_run(&set, img, &last, &[4.0], &[1.0], MAX_STORED_PREDS);
        assert_eq!(capped.preds.len(), MAX_STORED_PREDS);
        let (all, _) = image_run(&set, img, &last, &[4.0], &[1.0], usize::MAX);
        assert_eq!(all.preds.len(), n);
        let gt = pseudo_gt_boxes(&all.preds);
        assert_eq!(gt.len(), n);
        let objects = gt.iter().filter(|g| !g.ignore).count();
        let expected = all
            .preds
            .iter()
            .filter(|p| p.confidence >= PSEUDO_GT_CONFIDENCE)
            .count();
        assert_eq!(objects, expected);
        assert!(
            gt.iter().any(|g| g.ignore),
            "low confidences are ignore regions"
        );
        assert!(gt.iter().all(|g| g.label == "person"));
    }
}
