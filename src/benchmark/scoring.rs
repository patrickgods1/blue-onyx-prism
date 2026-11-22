//! Accuracy scoring of a finished run: AP against real or pseudo ground truth (`score`), the
//! per-image TP/FP drill-down flags, frame-level ROC curves and the pseudo ground truth run
//! on a reference device. Pure over stored predictions apart from `pseudo_ground_truth`.

use super::*;

/// Score the run: overall accuracy (over the images with real ground truth, else those with
/// pseudo ground truth), per dataset and per tag; marks each stored prediction TP/FP and each
/// ground-truth object matched/missed in `per_image`. Each held-out tag (`heldout:*`, the only
/// ones a training run reads them from; other tags would only grow `benchmark.json`) also gets
/// its per-class recall ([`Breakdown::per_class`]) over every class of the model the sets score:
/// a custom model's raccoon or fox is no CCTV class, so it is in no AP here, but it has its
/// recall per tag. A model without any CCTV class gets no accuracy at all; [`tag_recall_only`]
/// gives its held-out tags' recall.
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
    // B10: the pseudo ground truth of these sets is this model's own detections.
    let own = reference_sets(&job.name, all);
    let labelled = |s: &ImageSet| !own.contains(&s.id);
    if !all.iter().any(|(s, i)| i.gt.is_some() && labelled(s)) {
        return (
            None,
            Some(format!(
                "the labels of these images are its own detections (it is the pseudo-ground-truth \
                 reference of {}): it is not scored against itself",
                own.join(", ")
            )),
        );
    }
    // Per image: the scene the service computed (the scene splits and `scene:*` tags).
    let scenes: Vec<Option<crate::image_stats::Scene>> = (0..all.len())
        .map(|k| per_image.get(k).and_then(|r| r.scene))
        .collect();
    // Per image: the eval input (None = no ground truth).
    let mut evals: Vec<Option<EvalImage>> = Vec::with_capacity(all.len());
    // Per image: the classes its dataset annotates (frame samples).
    let mut fclasses: Vec<Vec<String>> = Vec::with_capacity(all.len());
    for (k, (set, img)) in all.iter().enumerate() {
        // `per_image` and `eval_preds` are parallel to `all`; a missing entry scores as
        // "no ground truth" rather than panicking.
        let (Some(gt), Some(preds), Some(run), true) = (
            &img.gt,
            eval_preds.get(k),
            per_image.get_mut(k),
            labelled(set),
        ) else {
            evals.push(None);
            fclasses.push(Vec::new());
            continue;
        };
        let (eval, classes) = score_image(&map, set, gt, preds, run, threshold);
        fclasses.push(classes);
        evals.push(Some(eval));
    }
    let scored = Scored::new(all, evals, fclasses, scenes);
    let chosen = &scored.chosen;
    let relative = scored.relative;
    let overall = metrics::evaluate(&scored.subset(chosen), &map.classes, threshold);
    // Every class of the model the sets score (raccoon, fox, ... besides the CCTV classes):
    // the all-classes AP and each tag's per-class recall.
    let wide = ClassMap::all_classes(&job.classes);
    let wide = (wide.classes.len() > map.classes.len()).then(|| {
        let evals = wide_evals(&wide, all, eval_preds);
        (wide, evals)
    });
    let all_classes = wide
        .as_ref()
        .map(|(m, evals)| all_classes_ap(m, evals, chosen, threshold));
    // The per-class recall of the images `ks`: over every class when the model has more than
    // the CCTV ones (the AP of a tag stays on the CCTV classes, the headline's basis).
    let class_recall = |ks: &[usize]| match &wide {
        Some((m, evals)) => {
            let imgs: Vec<EvalImage> = ks
                .iter()
                .filter_map(|&k| evals.get(k).cloned().flatten())
                .collect();
            metrics::class_recall(&imgs, &m.classes, threshold)
        }
        None => metrics::class_recall(&scored.subset(ks), &map.classes, threshold),
    };
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
    // Per tag, over the images of the overall score: the manifest's tags and the computed
    // scene's `scene:<day|night|ir>`.
    let mut tags: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for &k in chosen {
        let Some((_, img)) = all.get(k) else { continue };
        for t in &img.tags {
            tags.entry(t.clone()).or_default().push(k);
        }
        if let Some(sc) = scored.scenes.get(k).copied().flatten() {
            tags.entry(images::scene_tag(sc)).or_default().push(k);
        }
    }
    let by_tag = tags
        .iter()
        .map(|(t, ks)| (t.as_str(), ks.as_slice()))
        .map(|(t, ks)| {
            let sum = metrics::evaluate(&scored.subset(ks), &map.classes, threshold);
            Breakdown {
                per_class: if t.starts_with(HELDOUT_TAG) {
                    class_recall(ks)
                } else {
                    Vec::new()
                },
                ..Breakdown::of(t, &sum, relative, scored.frame_roc(ks, &map.classes, 0).auc)
            }
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

/// The sets of `all` whose pseudo ground truth is the detections of the model `name` (B10):
/// that model is never scored on them (it would score itself). Set ids, first-seen order.
pub(crate) fn reference_sets(name: &str, all: &[(&ImageSet, &images::SetImage)]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (s, _) in all {
        if let GroundTruth::Pseudo(r) = &s.ground_truth
            && is_reference(r, name)
            && !out.contains(&s.id)
        {
            out.push(s.id.clone());
        }
    }
    out
}

// The tag prefix whose breakdowns carry `Breakdown::per_class` (the held-out set's own tags).
use crate::training::dataset::HELDOUT_TAG;

/// Per image of `all`: its eval input in `map`'s label space (None = no ground truth).
fn wide_evals(
    map: &ClassMap,
    all: &[(&ImageSet, &images::SetImage)],
    eval_preds: &[Vec<PredBox>],
) -> Vec<Option<EvalImage>> {
    all.iter()
        .enumerate()
        .map(|(k, (set, img))| {
            let gt = img.gt.as_ref()?;
            let preds = eval_preds.get(k)?;
            Some(eval_input(map, set.scored_labels.as_deref(), gt, preds).1)
        })
        .collect()
}

/// The per-class recall of each held-out tag (`heldout:*`) of a run that [`score`] gave no
/// accuracy: a model without any CCTV class (a custom raccoon / opossum / fox model) still gets
/// what a training run reads, [`RunResult::tag_recall`]. Over the images with real ground truth
/// only, every class of the model the sets score; nothing else (no AP, no headline, no grade).
/// Empty without such images or classes.
pub(super) fn tag_recall_only(
    job: &Job,
    all: &[(&ImageSet, &images::SetImage)],
    eval_preds: &[Vec<PredBox>],
    threshold: f32,
) -> Vec<TagRecall> {
    let wide = ClassMap::all_classes(&job.classes);
    if wide.is_empty() {
        return Vec::new();
    }
    let evals = wide_evals(&wide, all, eval_preds);
    let mut tags: BTreeMap<&str, Vec<EvalImage>> = BTreeMap::new();
    for (k, (set, img)) in all.iter().enumerate() {
        let Some(Some(e)) = evals.get(k) else {
            continue;
        };
        if !set.ground_truth.is_real() {
            continue;
        }
        for t in img.tags.iter().filter(|t| t.starts_with(HELDOUT_TAG)) {
            tags.entry(t.as_str()).or_default().push(e.clone());
        }
    }
    tags.into_iter()
        .map(|(t, imgs)| TagRecall {
            key: t.to_string(),
            images: imgs.len(),
            per_class: metrics::class_recall(&imgs, &wide.classes, threshold),
        })
        .collect()
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
                            window: None,
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

/// AP over every class of the model (`all_map`, wider than the scored CCTV classes) on the
/// images `chosen` of the overall score, averaged like the overall AP (the classes with
/// [`metrics::MIN_CLASS_GT`] objects or more, recorded in its `ap_basis`). `evals`: per image of
/// the run, its eval input in `all_map`'s label space (None = no ground truth).
pub(super) fn all_classes_ap(
    all_map: &ClassMap,
    evals: &[Option<EvalImage>],
    chosen: &[usize],
    threshold: f32,
) -> AllClasses {
    let imgs: Vec<EvalImage> = chosen
        .iter()
        .filter_map(|&k| evals.get(k).cloned().flatten())
        .collect();
    let s = metrics::evaluate(&imgs, &all_map.classes, threshold);
    AllClasses {
        classes: s.per_class.iter().filter(|c| c.gt > 0).count(),
        ap50: s.ap50,
        ap50_95: s.ap50_95,
        ap_basis: s.ap_basis,
    }
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
    /// Per image: the scene the service computed (None in old results' re-scores).
    scenes: Vec<Option<crate::image_stats::Scene>>,
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
        scenes: Vec<Option<crate::image_stats::Scene>>,
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
            scenes,
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
        // Day / night / IR by the computed scene (the service's split), other tags as tagged.
        for tag in threshold::SWEEP_TAGS {
            let ks: Vec<usize> = chosen
                .iter()
                .copied()
                .filter(|&k| {
                    all.get(k).is_some_and(|(_, i)| {
                        images::in_split(&i.tags, self.scenes.get(k).copied().flatten(), tag)
                    })
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
                *p = p.without_frames();
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
                .map(|p| threshold::Pick { relative, ..p })
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
        on_image: None,
        weights: bench.weights,
        threshold_objective: bench.threshold_objective,
        accuracy_metric: bench.accuracy_metric,
        // Every reference detection >= PSEUDO_IGNORE_CONFIDENCE becomes ground truth, not
        // only the 100 most confident (BM-10).
        max_stored_preds: usize::MAX,
        // The reference model of a regular benchmark: cached like the run itself.
        no_cache: false,
        priority: bench.priority,
        // Only its detections are kept, so the time per frame is all that counts.
        read_ahead: super::ReadAhead::Always,
        warmup_rule: super::WarmupRule::DEFAULT,
        // Labels, not a measurement: nothing else to record.
        live_requests: None,
        video_decode: false,
        #[cfg(test)]
        slow_eval: None,
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
            variant: None,
            concurrency: None,
            device_thresholds: Default::default(),
        };
        let pred = |l: &str, c: f32, b: [f32; 4]| PredBox {
            label: l.into(),
            confidence: c,
            bbox: b,
            window: None,
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
                scene: None,
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

    /// B10: the reference model's detections are the labels of a pseudo-ground-truth set, so
    /// it is never scored on that set: alone, it gets no accuracy (a note says why); beside a
    /// set with real ground truth, the pseudo set is left out of every breakdown.
    #[test]
    fn the_reference_is_not_scored_against_its_own_labels() {
        let mut real = ImageSet::sample();
        real.id = "real".into();
        real.ground_truth = GroundTruth::Manifest;
        real.images[0].gt = Some(vec![GtBox {
            label: "dog".into(),
            bbox: [0.0, 0.0, 100.0, 100.0],
            ignore: false,
            crowd: false,
        }]);
        let mut pseudo = real.clone();
        pseudo.id = "pseudo".into();
        pseudo.ground_truth = GroundTruth::Pseudo("Ref-Model on openvino:cpu".into());
        let job = Job {
            name: "ref-model".into(),
            path: "m.onnx".into(),
            family: ModelFamilyKind::Yolo5,
            classes: crate::model::classes::coco80(),
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
            variant: None,
            concurrency: None,
            device_thresholds: Default::default(),
        };
        let eval = vec![PredBox {
            label: "dog".into(),
            confidence: 0.9,
            bbox: [1.0, 1.0, 100.0, 100.0],
            window: None,
        }];
        let run = |s: &ImageSet| ImageRun {
            set: s.id.clone(),
            file: s.images[0].file.clone(),
            total_ms: 1.0,
            infer_ms: 1.0,
            preds: vec![],
            gt_matched: vec![],
            counts: None,
            scene: None,
        };
        // Alone: nothing to score it against.
        let all = vec![(&pseudo, &pseudo.images[0])];
        let mut pi = vec![run(&pseudo)];
        let (acc, note) = score(
            &job,
            &all,
            &mut pi,
            std::slice::from_ref(&eval),
            0.5,
            Default::default(),
        );
        assert!(acc.is_none(), "{acc:?}");
        let note = note.unwrap();
        assert!(note.contains("own detections"), "{note}");
        assert_eq!(reference_sets(&job.name, &all), ["pseudo"]);
        // Another model is scored on it as before (relative).
        let other = Job {
            name: "other".into(),
            ..job.clone()
        };
        let (acc, _) = score(
            &other,
            &all,
            &mut pi,
            std::slice::from_ref(&eval),
            0.5,
            Default::default(),
        );
        assert!(acc.unwrap().relative);
        assert!(reference_sets(&other.name, &all).is_empty());
        // Beside real ground truth: the pseudo set is in no breakdown or curve.
        let all = vec![(&real, &real.images[0]), (&pseudo, &pseudo.images[0])];
        let mut pi = vec![run(&real), run(&pseudo)];
        let (acc, _) = score(
            &job,
            &all,
            &mut pi,
            &[eval.clone(), eval],
            0.5,
            Default::default(),
        );
        let acc = acc.unwrap();
        assert!(!acc.relative);
        let ids: Vec<&str> = acc.by_dataset.iter().map(|b| b.key.as_str()).collect();
        assert_eq!(ids, ["real"]);
        let sweep = acc.sweep.unwrap();
        assert!(sweep.by_dataset.iter().all(|c| c.key != "pseudo"));
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
            variant: None,
            concurrency: None,
            device_thresholds: Default::default(),
        };
        let p = |l: &str, b: [f32; 4]| PredBox {
            label: l.into(),
            confidence: 0.9,
            bbox: b,
            window: None,
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
            scene: None,
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
        // F1 1 for thresholds in (0.3, 0.9]: grid 0.31 ..= 0.90, the lower middle of 60.
        let best = sw.overall.best(threshold::Objective::F1).unwrap();
        assert_eq!(best.exact_threshold, Some(0.9));
        assert_eq!(best.plateau, Some([0.31, 0.9]));
        assert_eq!(best.threshold, 0.6);
        assert_eq!(best.point.f1, Some(1.0));
        // Every standard objective but Youden has a pick: one sample per frame, and the only
        // frame has a car (the pooled samples had its motorcycle class as a negative).
        assert_eq!(sw.overall.picks.len(), 3);
        assert!(sw.overall.picks.iter().all(|p| p.objective != "youden"));
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

    /// A custom model trained on the default review classes, on its held-out set (a folder set
    /// whose scored labels are the export's classes): every tag lists the recall of each class
    /// with ground truth, the wildlife classes the CCTV scoring leaves out included, while the
    /// AP of the tag and of the run stays on the CCTV classes. Datasets get no per-class list.
    #[test]
    fn tags_carry_recall_of_every_class_custom_animals_included() {
        let classes: Vec<String> = crate::collect::config::DEFAULT_CLASSES
            .map(String::from)
            .to_vec();
        let g = |l: &str, b: [f32; 4]| GtBox {
            label: l.into(),
            bbox: b,
            ignore: false,
            crowd: false,
        };
        let p = |l: &str, c: f32, b: [f32; 4]| PredBox {
            label: l.into(),
            confidence: c,
            bbox: b,
            window: None,
        };
        let mut set = ImageSet::sample();
        set.id = "heldout-m".into();
        set.ground_truth = GroundTruth::Manifest;
        set.scored_labels = Some(classes.clone());
        let tags = |scene: &str| -> Vec<String> {
            [
                format!("heldout:{scene}"),
                "heldout:all".into(),
                "heldout:cam:back".into(),
                format!("heldout:camscene:{scene}:back"),
            ]
            .to_vec()
        };
        let mut night = set.images[0].clone();
        night.tags = tags("night");
        night.gt = Some(vec![
            g("person", [0.0, 0.0, 100.0, 200.0]),
            g("raccoon", [300.0, 300.0, 360.0, 340.0]),
            g("fox", [500.0, 300.0, 580.0, 360.0]),
        ]);
        let mut ir = set.images[0].clone();
        ir.tags = tags("ir");
        ir.gt = Some(vec![
            g("raccoon", [0.0, 0.0, 60.0, 40.0]),
            g("opossum", [200.0, 200.0, 260.0, 240.0]),
            g("bird", [400.0, 0.0, 420.0, 20.0]),
        ]);
        set.images = vec![night, ir];
        let preds = vec![
            vec![
                p("person", 0.9, [0.0, 0.0, 100.0, 200.0]),
                p("raccoon", 0.8, [300.0, 300.0, 360.0, 340.0]),
                // The fox called a cat: a missed fox (and a false cat).
                p("cat", 0.7, [500.0, 300.0, 580.0, 360.0]),
            ],
            vec![
                p("opossum", 0.6, [200.0, 200.0, 260.0, 240.0]),
                p("bird", 0.9, [400.0, 0.0, 420.0, 20.0]),
                // Below the threshold: the raccoon is missed.
                p("raccoon", 0.2, [0.0, 0.0, 60.0, 40.0]),
            ],
        ];
        let job = Job {
            name: "custom".into(),
            path: "custom.onnx".into(),
            family: ModelFamilyKind::Yolo8,
            classes,
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
            variant: None,
            concurrency: None,
            device_thresholds: Default::default(),
        };
        let all: Vec<_> = set.images.iter().map(|i| (&set, i)).collect();
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
                scene: None,
            })
            .collect();
        let (acc, note) = score(&job, &all, &mut pi, &preds, 0.5, Default::default());
        assert!(note.is_none());
        let acc = acc.unwrap();
        let cells = |key: &str| -> BTreeMap<String, (usize, usize)> {
            acc.by_tag
                .iter()
                .find(|b| b.key == key)
                .unwrap()
                .per_class
                .iter()
                .map(|c| (c.class.clone(), (c.gt, c.missed)))
                .collect()
        };
        let want = |v: &[(&str, usize, usize)]| -> BTreeMap<String, (usize, usize)> {
            v.iter()
                .map(|&(c, gt, m)| (c.to_string(), (gt, m)))
                .collect()
        };
        let every = want(&[
            ("person", 1, 0),
            ("raccoon", 2, 1),
            ("fox", 1, 1),
            ("opossum", 1, 0),
            ("bird", 1, 0),
        ]);
        assert_eq!(cells("heldout:all"), every);
        assert_eq!(cells("heldout:cam:back"), every);
        assert_eq!(
            cells("heldout:night"),
            want(&[("person", 1, 0), ("raccoon", 1, 0), ("fox", 1, 1)])
        );
        assert_eq!(
            cells("heldout:camscene:ir:back"),
            want(&[("raccoon", 1, 1), ("opossum", 1, 0), ("bird", 1, 0)])
        );
        let racc = acc.by_tag[0]
            .per_class
            .iter()
            .find(|c| c.class == "raccoon")
            .cloned();
        assert!(racc.is_some_and(|c| c.recall() == Some(0.5)));
        // The AP of the run and of each tag stays on the CCTV classes.
        assert_eq!(
            acc.classes,
            [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "truck",
                "dog",
                "cat",
                "bird"
            ]
        );
        let tag = acc.by_tag.iter().find(|b| b.key == "heldout:all").unwrap();
        assert_eq!(tag.gt, 2, "person and bird: the CCTV classes with objects");
        assert!(acc.all_classes.is_some());
        assert!(acc.by_dataset.iter().all(|b| b.per_class.is_empty()));
        // The tag tables leave the per-camera held-out tags out.
        let shown: Vec<&str> = acc.shown_tags().iter().map(|b| b.key.as_str()).collect();
        assert!(shown.contains(&"heldout:all") && shown.contains(&"heldout:night"));
        assert!(
            shown.iter().all(|k| !k.starts_with("heldout:cam")),
            "{shown:?}"
        );
        assert!(acc.by_tag.iter().any(|b| b.key == "heldout:cam:back"));
        // Round trip, and a breakdown of an older version without the field parses.
        let j = serde_json::to_value(tag).unwrap();
        assert!(j["per_class"].as_array().is_some_and(|a| a.len() == 5));
        let back: Breakdown = serde_json::from_value(j).unwrap();
        assert_eq!(&back, tag);
        let mut old = serde_json::to_value(tag).unwrap();
        old.as_object_mut().unwrap().remove("per_class");
        let old: Breakdown = serde_json::from_value(old).unwrap();
        assert!(old.per_class.is_empty());
        assert!(
            !serde_json::to_string(&old).unwrap().contains("per_class"),
            "an empty list is not written"
        );
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
            window: None,
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
            variant: None,
            concurrency: None,
            device_thresholds: Default::default(),
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
                scene: None,
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
        // Brute force over the grid: no threshold beats the pick's best value (per frame).
        let bp_best = y.exact_score.unwrap();
        for k in 5..=100 {
            let t = k as f32 / 100.0;
            let j = sw
                .overall
                .points
                .iter()
                .find(|p| (p.threshold - t).abs() < 1e-6)
                .and_then(|p| p.per_frame?.youden());
            if let Some(j) = j {
                assert!(j <= bp_best + 1e-12, "{t}: {j} > {bp_best}");
            }
        }
        // Per class curves carry the class AUC.
        let pc = sw.per_class.iter().find(|c| c.key == "vehicle").unwrap();
        assert!((pc.roc_auc.unwrap() - 8.0 / 9.0).abs() < 1e-12);
        // Group grids carry no frame rates (size); their configured points and picks do.
        assert!(
            pc.points
                .iter()
                .all(|p| p.frame.is_none() && p.per_frame.is_none() && p.false_alerts.is_none())
        );
        assert!(pc.configured.frame.is_some());
        assert!(
            sw.overall
                .points
                .iter()
                .all(|p| p.frame.is_some() && p.per_frame.is_some())
        );
    }

    /// The computed scene splits the run: each image's `scene:<scene>` breakdown beside the
    /// manifest's tags, and the day / night / IR threshold curves by the computed scene, an
    /// image without one (an older result) by its tags.
    #[test]
    fn the_computed_scene_splits_the_run() {
        use crate::image_stats::Scene;
        let img = |file: &str| images::SetImage {
            file: file.into(),
            path: None,
            width: 640,
            height: 480,
            tags: vec!["night".into()],
            gt: Some(vec![GtBox {
                label: "person".into(),
                bbox: [0.0, 0.0, 100.0, 100.0],
                ignore: false,
                crowd: false,
            }]),
        };
        let mut set = ImageSet::sample();
        set.id = "s".into();
        set.ground_truth = GroundTruth::Manifest;
        set.images = vec![img("a"), img("b"), img("c"), img("d")];
        let job = job_of(&["person".into()]);
        let eval: Vec<Vec<PredBox>> = (0..4)
            .map(|_| {
                vec![PredBox {
                    label: "person".into(),
                    confidence: 0.9,
                    bbox: [0.0, 0.0, 100.0, 100.0],
                    window: None,
                }]
            })
            .collect();
        let all: Vec<(&ImageSet, &images::SetImage)> =
            set.images.iter().map(|i| (&set, i)).collect();
        let scenes = [Some(Scene::Day), Some(Scene::Night), Some(Scene::Ir), None];
        let mut pi: Vec<ImageRun> = all
            .iter()
            .zip(scenes)
            .map(|((s, i), scene)| ImageRun {
                set: s.id.clone(),
                file: i.file.clone(),
                total_ms: 1.0,
                infer_ms: 1.0,
                preds: vec![],
                gt_matched: vec![],
                counts: None,
                scene,
            })
            .collect();
        let (acc, _) = score(&job, &all, &mut pi, &eval, 0.5, Default::default());
        let acc = acc.unwrap();
        let tag = |k: &str| acc.by_tag.iter().find(|b| b.key == k).map(|b| b.images);
        assert_eq!(tag("night"), Some(4), "the manifest's tag stays");
        assert_eq!(
            (tag("scene:day"), tag("scene:night"), tag("scene:ir")),
            (Some(1), Some(1), Some(1))
        );
        let sw = acc.sweep.unwrap();
        let curve = |k: &str| sw.by_tag.iter().find(|c| c.key == k).map(|c| c.images);
        // Night: b (computed) and d (tagged, no computed scene); day a, IR c.
        assert_eq!(
            (curve("day"), curve("night"), curve("ir")),
            (Some(1), Some(2), Some(1))
        );
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
            scene: crate::image_stats::Scene::Day,
            routed: false,
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

    /// A held-out-like set of two night images on camera `back`, tagged like the held-out
    /// manifest plus a bare `date:` tag: image 1 person (found) and raccoon (found), image 2
    /// raccoon (missed) and fox (called a cat).
    fn heldout_pair(classes: &[String]) -> (ImageSet, Vec<Vec<PredBox>>) {
        let g = |l: &str, b: [f32; 4]| GtBox {
            label: l.into(),
            bbox: b,
            ignore: false,
            crowd: false,
        };
        let p = |l: &str, c: f32, b: [f32; 4]| PredBox {
            label: l.into(),
            confidence: c,
            bbox: b,
            window: None,
        };
        let mut set = ImageSet::sample();
        set.id = "heldout-m".into();
        set.ground_truth = GroundTruth::Manifest;
        set.scored_labels = Some(classes.to_vec());
        let tags: Vec<String> = [
            "heldout:night",
            "heldout:all",
            "heldout:cam:back",
            "heldout:camscene:night:back",
            "date:2026-11-01",
            "camera:back",
        ]
        .map(String::from)
        .to_vec();
        let mut one = set.images[0].clone();
        one.tags = tags.clone();
        one.gt = Some(vec![
            g("person", [0.0, 0.0, 100.0, 200.0]),
            g("raccoon", [300.0, 300.0, 360.0, 340.0]),
        ]);
        let mut two = set.images[0].clone();
        two.tags = tags;
        two.gt = Some(vec![
            g("raccoon", [0.0, 0.0, 60.0, 40.0]),
            g("fox", [500.0, 300.0, 580.0, 360.0]),
        ]);
        set.images = vec![one, two];
        let preds = vec![
            vec![
                p("person", 0.9, [0.0, 0.0, 100.0, 200.0]),
                p("raccoon", 0.8, [300.0, 300.0, 360.0, 340.0]),
            ],
            vec![p("cat", 0.7, [500.0, 300.0, 580.0, 360.0])],
        ];
        (set, preds)
    }

    fn job_of(classes: &[String]) -> Job {
        Job {
            name: "custom".into(),
            path: "custom.onnx".into(),
            family: ModelFamilyKind::Yolo8,
            classes: classes.to_vec(),
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
            variant: None,
            concurrency: None,
            device_thresholds: Default::default(),
        }
    }

    fn runs_of(set: &ImageSet) -> Vec<ImageRun> {
        set.images
            .iter()
            .map(|i| ImageRun {
                set: set.id.clone(),
                file: i.file.clone(),
                total_ms: 1.0,
                infer_ms: 1.0,
                preds: vec![],
                gt_matched: vec![],
                counts: None,
                scene: None,
            })
            .collect()
    }

    /// A model of CCTV classes only (no wider class map): the tags' per-class recall comes from
    /// the scored classes themselves, and only on the held-out tags.
    #[test]
    fn per_class_of_a_cctv_only_model_uses_the_scored_classes() {
        let classes: Vec<String> = ["person", "cat"].map(String::from).to_vec();
        let (set, preds) = heldout_pair(&classes);
        let all: Vec<_> = set.images.iter().map(|i| (&set, i)).collect();
        let job = job_of(&classes);
        assert_eq!(
            ClassMap::all_classes(&job.classes).classes.len(),
            ClassMap::for_model(&job.classes).classes.len(),
            "no wider map"
        );
        let (acc, _) = score(
            &job,
            &all,
            &mut runs_of(&set),
            &preds,
            0.5,
            Default::default(),
        );
        let acc = acc.unwrap();
        assert!(acc.all_classes.is_none());
        let tag = |k: &str| acc.by_tag.iter().find(|b| b.key == k).unwrap();
        // The raccoon and the fox are wildlife ignore regions here, not misses.
        assert_eq!(
            tag("heldout:all").per_class,
            [ClassRecall::new("person", 1, 0)]
        );
        assert_eq!(
            tag("heldout:camscene:night:back").per_class,
            [ClassRecall::new("person", 1, 0)]
        );
        // Not on other tags.
        assert!(tag("date:2026-11-01").per_class.is_empty());
        assert!(tag("camera:back").per_class.is_empty());
    }

    /// A model without any CCTV class: no accuracy, the same note, and the held-out tags'
    /// per-class recall from `tag_recall_only` (only on images with real ground truth, held-out
    /// tags only).
    #[test]
    fn a_model_without_cctv_classes_still_gets_heldout_recall() {
        let classes: Vec<String> = ["raccoon", "opossum", "skunk", "fox"]
            .map(String::from)
            .to_vec();
        let (set, preds) = heldout_pair(&classes);
        let all: Vec<_> = set.images.iter().map(|i| (&set, i)).collect();
        let job = job_of(&classes);
        let (acc, note) = score(
            &job,
            &all,
            &mut runs_of(&set),
            &preds,
            0.5,
            Default::default(),
        );
        assert!(acc.is_none());
        assert_eq!(
            note.as_deref(),
            Some("the model has none of the scored CCTV classes")
        );
        let tags = tag_recall_only(&job, &all, &preds, 0.5);
        let keys: Vec<&str> = tags.iter().map(|t| t.key.as_str()).collect();
        assert_eq!(
            keys,
            [
                "heldout:all",
                "heldout:cam:back",
                "heldout:camscene:night:back",
                "heldout:night"
            ]
        );
        let all_tag = &tags[0];
        assert_eq!(all_tag.images, 2);
        assert_eq!(
            all_tag.per_class,
            [
                ClassRecall::new("raccoon", 2, 1),
                ClassRecall::new("fox", 1, 1)
            ]
        );
        // Pseudo ground truth is no recall.
        let mut pseudo = set.clone();
        pseudo.ground_truth = GroundTruth::Pseudo("ref".into());
        let all: Vec<_> = pseudo.images.iter().map(|i| (&pseudo, i)).collect();
        assert!(tag_recall_only(&job, &all, &preds, 0.5).is_empty());
        // A model with no class at all: nothing.
        assert!(tag_recall_only(&job_of(&[]), &all, &preds, 0.5).is_empty());
    }
}
