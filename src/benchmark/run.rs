//! The benchmark run itself: `Bench` compiles a model on a device, warms it up and times the
//! per-image pipeline (decode -> preprocess -> infer -> postprocess) at the configured and the
//! evaluation confidence. Progress and cancellation are reported through `Progress`/`Cancelled`.
//! One image at a time on one thread; the runtimes mutex is held only while compiling.

use super::scoring::score;
use super::*;

/// What a benchmark is doing right now (for progress reporting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Compiling the model and creating the inference request.
    Loading,
    /// Untimed warm-up iterations.
    Warmup,
    /// Timed iterations over the images.
    Timing,
}

/// Progress of one device run, reported through [`Bench::progress`].
#[derive(Debug, Clone, Copy)]
pub struct Progress<'a> {
    pub model: &'a str,
    pub device: &'a str,
    pub phase: Phase,
    /// Iterations finished in this phase.
    pub done: usize,
    /// Iterations of this phase.
    pub total: usize,
    /// Image being processed (timing phase).
    pub image: &'a str,
}

/// Error returned when the cancel flag is set during a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("benchmark cancelled")]
pub struct Cancelled;

/// Whether `e` (or its cause chain) is a [`Cancelled`].
pub fn is_cancelled(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.downcast_ref::<Cancelled>().is_some())
}

/// Timing settings shared by every run of a benchmark.
pub struct Bench<'a> {
    /// Images to run (at least one image in total).
    pub sets: &'a [ImageSet],
    pub warmup: usize,
    /// Timed runs per image.
    pub repeat: usize,
    pub params: PostParams,
    pub object_filter: &'a [String],
    pub cache_dir: Option<&'a Path>,
    /// Checked before every iteration; a set flag ends the run with [`Cancelled`].
    pub cancel: Option<&'a AtomicBool>,
    /// Called at every phase change and after every iteration.
    pub progress: Option<&'a (dyn Fn(&Progress) + Sync)>,
    /// Called with each image's result as soon as its timed runs end (a cancelled run's
    /// finished images stay usable: the auto-labeler keeps them).
    pub on_image: Option<&'a (dyn Fn(&ImageRun) + Sync)>,
    pub weights: Weights,
    /// What the best confidence threshold optimizes (exact picks are computed per run).
    pub threshold_objective: threshold::Objective,
    /// What the accuracy grade is computed from.
    pub accuracy_metric: grade::AccuracyMetric,
    /// Stored predictions per image (most confident first; `MAX_STORED_PREDS`). The pseudo
    /// ground truth run keeps all of them (BM-10).
    pub max_stored_preds: usize,
}

impl std::fmt::Debug for Bench<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bench")
            .field("sets", &self.sets.len())
            .field("warmup", &self.warmup)
            .field("repeat", &self.repeat)
            .field("params", &self.params)
            .field("object_filter", &self.object_filter)
            .field("cache_dir", &self.cache_dir)
            .field("cancel", &self.cancel)
            .field("progress", &self.progress.is_some())
            .field("on_image", &self.on_image.is_some())
            .field("weights", &self.weights)
            .field("threshold_objective", &self.threshold_objective)
            .field("accuracy_metric", &self.accuracy_metric)
            .field("max_stored_preds", &self.max_stored_preds)
            .finish()
    }
}

/// One request's timings (decode, preprocess, infer, postprocess; ms).
pub(super) struct Sample {
    pub(super) stages: [f64; 4],
    pub(super) predictions: Vec<Prediction>,
    pub(super) eval: Vec<Prediction>,
}

/// Files directly inside `dir` (the compiled-blob cache is flat).
pub(super) fn cache_files(dir: &Path) -> std::collections::HashSet<PathBuf> {
    std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
        .unwrap_or_default()
}

pub(super) fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Lock the shared runtimes, ignoring poisoning (a panicked compile leaves them usable).
pub(super) fn lock(rt: &Mutex<Runtimes>) -> std::sync::MutexGuard<'_, Runtimes> {
    rt.lock_recover()
}

pub(super) fn to_pred(p: &Prediction) -> PredBox {
    PredBox {
        label: p.label.clone(),
        confidence: p.confidence,
        bbox: [
            p.x_min as f32,
            p.y_min as f32,
            p.x_max as f32,
            p.y_max as f32,
        ],
    }
}

impl<'a> Bench<'a> {
    /// Settings over `sets` with no cache dir, filter, cancel flag or progress callback.
    pub fn new(sets: &'a [ImageSet], warmup: usize, repeat: usize, params: PostParams) -> Self {
        Self {
            sets,
            warmup,
            repeat,
            params,
            object_filter: &[],
            cache_dir: None,
            cancel: None,
            progress: None,
            on_image: None,
            weights: Weights::default(),
            threshold_objective: threshold::Objective::default(),
            accuracy_metric: grade::AccuracyMetric::default(),
            max_stored_preds: MAX_STORED_PREDS,
        }
    }

    pub fn image_count(&self) -> usize {
        self.sets.iter().map(|s| s.images.len()).sum()
    }

    /// Dataset ids, comma separated.
    pub fn describe_sets(&self) -> String {
        self.sets
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(Cancelled.into());
        }
        Ok(())
    }

    fn report(
        &self,
        job: &Job,
        device: &str,
        phase: Phase,
        done: usize,
        total: usize,
        image: &str,
    ) {
        if let Some(cb) = self.progress {
            cb(&Progress {
                model: &job.name,
                device,
                phase,
                done,
                total,
                image,
            });
        }
    }

    /// Compile `job` on the device spec `device` and time it. The spec's candidates mirror the
    /// server: a GPU request may fall back to CPU (see [`RunResult::fell_back`]).
    pub fn run(&self, runtimes: &Mutex<Runtimes>, job: &Job, device: &str) -> Result<RunResult> {
        let spec = spec::parse(device)?;
        let candidates = lock(runtimes).plan(&spec, &job.path);
        self.run_candidates(runtimes, job, device, &candidates)
    }

    /// Compile `job` on exactly `device` (no fallback) and time it.
    pub fn run_exact(
        &self,
        runtimes: &Mutex<Runtimes>,
        job: &Job,
        device: Device,
    ) -> Result<RunResult> {
        let cand = Candidate {
            device,
            fell_back: false,
            note: None,
        };
        self.run_candidates(runtimes, job, &device.to_string(), &[cand])
    }

    fn run_candidates(
        &self,
        runtimes: &Mutex<Runtimes>,
        job: &Job,
        requested: &str,
        candidates: &[Candidate],
    ) -> Result<RunResult> {
        if self.image_count() == 0 {
            bail!("no benchmark images");
        }
        self.check_cancel()?;
        self.report(job, requested, Phase::Loading, 0, 1, "");
        let req = LoadRequest {
            path: job.path.clone(),
            requested: requested.to_string(),
            gpu_precision: job.gpu_precision.clone(),
        };
        let before = self.cache_dir.map(cache_files);
        let t = Instant::now();
        let label = format!("model {}", job.name);
        // Compile under the runtimes lock (workers compile through the same lock), create the
        // inference request outside it. A compile can not be interrupted: a cancel takes
        // effect after the current compile, before the next fallback candidate (BM-06).
        let loaded = crate::backend::try_candidates(&label, candidates, |cand| {
            self.check_cancel()?;
            let compiled = lock(runtimes).compile(cand, &req)?;
            compiled.into_backend()
        });
        // A cancel during the compiles is a cancel, not a load failure (the error of several
        // failed candidates no longer carries `Cancelled`).
        self.check_cancel()?;
        let backend = loaded.with_context(|| format!("loading {}", job.path.display()))?;
        let compile_ms = ms(t.elapsed());
        let cache = match (self.cache_dir, before) {
            (None, _) | (_, None) => "disabled".to_string(),
            (Some(dir), Some(before)) => {
                let after = cache_files(dir);
                if after.difference(&before).next().is_some() {
                    "miss".into()
                } else if after.is_empty() {
                    "unknown".into()
                } else {
                    "hit".into()
                }
            }
        };
        let verifier = match job.variant.as_ref().and_then(|v| v.verify.as_ref()) {
            Some(v) => Some(self.load_verifier(runtimes, &v.job, &backend.info().device.spec)?),
            None => None,
        };
        self.time(job, requested, backend, verifier, compile_ms, cache)
    }

    /// Compile the verifier of a `verify:<model>` variant on `device` (the spec the verified
    /// model loaded on; falls back like the service would).
    fn load_verifier(
        &self,
        runtimes: &Mutex<Runtimes>,
        vjob: &Job,
        device: &str,
    ) -> Result<Backend> {
        let spec = spec::parse(device)?;
        let candidates = lock(runtimes).plan(&spec, &vjob.path);
        let req = LoadRequest {
            path: vjob.path.clone(),
            requested: device.to_string(),
            gpu_precision: vjob.gpu_precision.clone(),
        };
        let label = format!("verifier {}", vjob.name);
        let loaded = crate::backend::try_candidates(&label, &candidates, |cand| {
            self.check_cancel()?;
            let compiled = lock(runtimes).compile(cand, &req)?;
            compiled.into_backend()
        });
        self.check_cancel()?;
        loaded.with_context(|| format!("loading the verifier {}", vjob.path.display()))
    }

    /// Warm up and time `backend` over every image, then score its detections.
    fn time(
        &self,
        job: &Job,
        requested: &str,
        backend: Backend,
        verifier: Option<Backend>,
        compile_ms: f64,
        cache: String,
    ) -> Result<RunResult> {
        let info = backend.info();
        let dev = info.device.clone();
        let (in_w, in_h) = info.input_size;
        let family =
            crate::model::make_family(job.family, &info.inputs, &info.outputs, job.classes.len())?;
        let threshold = job
            .confidence_threshold
            .unwrap_or(self.params.confidence_threshold);
        let mut pipe = Pipeline {
            pre: Preprocessor::for_family(in_w, in_h, family.as_ref()),
            family,
            backend,
            params: PostParams {
                confidence_threshold: threshold,
                nms_iou: self.params.nms_iou,
            },
            eval_params: PostParams {
                confidence_threshold: EVAL_CONFIDENCE.min(threshold),
                nms_iou: self.params.nms_iou,
            },
            classes: &job.classes,
            filter: self.object_filter_for(job),
            variant: job.variant.as_ref(),
            verifier: match (
                verifier,
                job.variant.as_ref().and_then(|v| v.verify.as_ref()),
            ) {
                (Some(b), Some(v)) => Some(Verifier::new(b, v, self.params)?),
                _ => None,
            },
        };

        let all: Vec<(&ImageSet, &images::SetImage)> = self
            .sets
            .iter()
            .flat_map(|s| s.images.iter().map(move |i| (s, i)))
            .collect();
        let warmup_ms = self.warm_up(job, requested, &all, &mut pipe)?;
        let repeat = self.repeat.max(1);
        let mut timed = self.timed_runs(job, requested, &all, &mut pipe, repeat)?;
        let loop_s = timed.loop_time.as_secs_f64();

        let stages_ms = timed.stage_stats();
        let (accuracy, accuracy_note) = score(
            job,
            &all,
            &mut timed.per_image,
            &timed.eval_preds,
            threshold,
            self.threshold_objective,
        );
        let accuracy_score = accuracy
            .as_ref()
            .and_then(|a| a.score(self.accuracy_metric));
        let relative = accuracy.as_ref().is_some_and(|a| a.relative);
        let grades = Grades::with_metric(
            self.accuracy_metric,
            accuracy_score,
            stages_ms.total.p50,
            self.weights,
            relative,
        );
        let first_dets = timed
            .per_image
            .first()
            .map(|r| r.preds.iter().map(Detection::from).collect())
            .unwrap_or_default();
        Ok(RunResult {
            model: job.name.clone(),
            path: job.path.display().to_string(),
            family: pipe.family.kind().to_string(),
            requested_device: requested.to_string(),
            device: dev.actual.clone(),
            device_name: dev.full_name.clone(),
            spec: dev.spec.clone(),
            fell_back: dev.fell_back,
            input: format!("{in_w}x{in_h}"),
            image: self.describe_sets(),
            image_size: match all.as_slice() {
                [(_, one)] => format!("{}x{}", one.width, one.height),
                _ => format!("{} images", all.len()),
            },
            images: all.len(),
            compile_ms,
            cache,
            warmup_iterations: self.warmup,
            warmup_ms,
            repeat,
            stages_ms,
            throughput_fps: if loop_s > 0.0 {
                timed.total_iters as f64 / loop_s
            } else {
                0.0
            },
            detections: first_dets,
            by_resolution: timed.by_resolution(),
            accuracy,
            accuracy_note,
            grades: Some(grades),
            agreement: None,
            eval_preds: {
                let map = ClassMap::for_model(&job.classes);
                timed.eval_preds.iter().map(|v| map.map_preds(v)).collect()
            },
            per_image: timed.per_image,
        })
    }

    /// The object filter of `job`'s production predictions: its own (an empty list = none),
    /// else the global one (empty = none).
    fn object_filter_for<'s>(&'s self, job: &'s Job) -> Option<&'s [String]> {
        match &job.object_filter {
            Some(f) if !f.is_empty() => Some(f),
            Some(_) => None,
            None => (!self.object_filter.is_empty()).then_some(self.object_filter),
        }
    }

    /// The untimed warm-up iterations on the first image; returns their wall time (ms).
    fn warm_up(
        &self,
        job: &Job,
        requested: &str,
        all: &[(&ImageSet, &images::SetImage)],
        pipe: &mut Pipeline,
    ) -> Result<f64> {
        let t = Instant::now();
        self.report(job, requested, Phase::Warmup, 0, self.warmup, "");
        if self.warmup > 0 {
            let (set, img) = all.first().context("no benchmark images")?;
            let first = img.bytes()?;
            for i in 0..self.warmup {
                self.check_cancel()?;
                pipe.run(&first, false, set.transform)
                    .context("warm-up inference")?;
                self.report(job, requested, Phase::Warmup, i + 1, self.warmup, "");
            }
        }
        Ok(ms(t.elapsed()))
    }

    /// `repeat` timed runs of every image; the last run of each image keeps its predictions
    /// (production and evaluation threshold).
    fn timed_runs(
        &self,
        job: &Job,
        requested: &str,
        all: &[(&ImageSet, &images::SetImage)],
        pipe: &mut Pipeline,
        repeat: usize,
    ) -> Result<Timings> {
        let total_iters = all.len().saturating_mul(repeat);
        let mut timed = Timings {
            samples: Default::default(),
            by_res: BTreeMap::new(),
            per_image: Vec::with_capacity(all.len()),
            eval_preds: Vec::with_capacity(all.len()),
            total_iters,
            loop_time: Duration::ZERO,
        };
        let mut done = 0;
        self.report(job, requested, Phase::Timing, 0, total_iters, "");
        for (set, img) in all {
            let bytes = img.bytes()?;
            // Not pre-sized by `repeat`: the CLI and service bound it, but stay safe regardless.
            let mut totals = Vec::new();
            let mut infers = Vec::new();
            let mut last: Option<Sample> = None;
            let bucket = img.resolution();
            for r in 0..repeat {
                self.check_cancel()?;
                let t = Instant::now();
                let sample = pipe
                    .run(&bytes, r + 1 == repeat, set.transform)
                    .with_context(|| format!("image {}", img.file))?;
                timed.loop_time += t.elapsed();
                let total = timed.record(bucket, &sample);
                totals.push(total);
                infers.push(sample.stages[2]);
                last = Some(sample);
                done += 1;
                self.report(job, requested, Phase::Timing, done, total_iters, &img.file);
            }
            timed.by_res.entry(bucket).or_default().1 += 1;
            let last = last.context("no timed run (repeat must be >= 1)")?;
            let (run, eval) = image_run(set, img, &last, &totals, &infers, self.max_stored_preds);
            if let Some(f) = self.on_image {
                f(&run);
            }
            timed.eval_preds.push(eval);
            timed.per_image.push(run);
        }
        Ok(timed)
    }
}

/// One compiled model with the production per-request stages (the code `worker.rs` runs).
pub(super) struct Pipeline<'j> {
    backend: Backend,
    family: Box<dyn crate::model::Family>,
    pre: Preprocessor,
    params: PostParams,
    eval_params: PostParams,
    classes: &'j [String],
    filter: Option<&'j [String]>,
    /// Enhancement and filter rules on top of the plain stages ([`super::variant`]).
    variant: Option<&'j PipelineVariant>,
    /// The verifier of a `verify:<model>` variant.
    verifier: Option<Verifier<'j>>,
}

/// The second model of a `verify:<model>` variant: the stages the verifier's worker runs on a
/// verifier check (`worker::Purpose::Verify`): its own enhancement for the frame's scene, the
/// resize, inference, and its raw detections at `min_confidence` (no night rules, no object
/// filter).
pub(super) struct Verifier<'j> {
    backend: Backend,
    family: Box<dyn crate::model::Family>,
    pre: Preprocessor,
    params: PostParams,
    classes: &'j [String],
    variant: &'j super::variant::VerifyVariant,
}

impl<'j> Verifier<'j> {
    fn new(
        backend: Backend,
        v: &'j super::variant::VerifyVariant,
        bench: PostParams,
    ) -> Result<Self> {
        let info = backend.info();
        let (in_w, in_h) = info.input_size;
        let family = crate::model::make_family(
            v.job.family,
            &info.inputs,
            &info.outputs,
            v.job.classes.len(),
        )?;
        let configured = v
            .job
            .confidence_threshold
            .unwrap_or(bench.confidence_threshold);
        Ok(Self {
            pre: Preprocessor::for_family(in_w, in_h, family.as_ref()),
            family,
            backend,
            params: PostParams {
                confidence_threshold: crate::worker::effective_threshold(
                    v.config.min_confidence,
                    configured,
                ),
                nms_iou: bench.nms_iou,
            },
            classes: &v.job.classes,
            variant: v,
        })
    }

    /// The verifier's predictions on `img` (a crop or the frame) of a frame of `scene`.
    fn detect(
        &mut self,
        mut img: crate::image::RgbImage,
        scene: crate::image_stats::Scene,
    ) -> Result<Vec<Prediction>> {
        crate::model::enhance::apply(&mut img, self.variant.modes.for_scene(scene));
        let (chw, ctx) = self.pre.run(&img.rgb, img.width, img.height)?;
        let extra = self.family.extra_inputs(&ctx);
        let outputs = self.backend.infer(chw, &extra)?;
        let dets = self.family.postprocess(&outputs, &ctx, &self.params)?;
        Ok(crate::model::to_predictions(
            &dets,
            &ctx,
            self.classes,
            None,
        ))
    }

    /// Check the candidates of `preds` ([`crate::verify::plan`]) on `frame` (the original,
    /// not enhanced) and drop the rejected ones from `preds` and from `eval` (the same boxes
    /// at the evaluation threshold; candidates are taken from `preds`, so evaluation boxes
    /// below the production threshold stay unverified).
    fn apply(
        &mut self,
        frame: crate::image::RgbImage,
        scene: crate::image_stats::Scene,
        preds: Vec<Prediction>,
        eval: &mut Vec<Prediction>,
    ) -> Result<Vec<Prediction>> {
        use crate::verify::VerifyMode;
        let cfg = &self.variant.config;
        let plan = crate::verify::plan(&preds, frame.width, frame.height, scene, cfg);
        if plan.is_empty() {
            return Ok(preds);
        }
        let found: Vec<Option<Vec<Prediction>>> = match cfg.mode {
            VerifyMode::Crop => {
                let mut out = Vec::with_capacity(plan.checks.len());
                for c in &plan.checks {
                    out.push(match c.crop {
                        Some(r) => Some(self.detect(crate::verify::crop(&frame, r), scene)?),
                        None => None,
                    });
                }
                out
            }
            VerifyMode::Frame => vec![Some(self.detect(frame, scene)?)],
        };
        let outcomes = crate::verify::outcomes(&plan, &found, &self.variant.config);
        let rejected: Vec<Prediction> = outcomes
            .iter()
            .filter(|(_, o)| *o == crate::verify::Outcome::Rejected)
            .filter_map(|(i, _)| preds.get(*i).cloned())
            .collect();
        eval.retain(|e| !rejected.contains(e));
        Ok(crate::verify::apply(preds, &outcomes).0)
    }
}

impl Pipeline<'_> {
    /// One request through the worker's stages, serially, with per-stage timings; with `eval`,
    /// the same outputs are post-processed again at the evaluation threshold (untimed, without
    /// the object filter). The set's `transform` (an image-set property, not a production
    /// stage) is applied after decode and is not part of any stage time.
    ///
    /// A variant adds what the service does with the same options: the scene statistics of the
    /// (transformed) frame and its enhancement count as preprocess time, its filter rules run
    /// on the family's detections (postprocess time) and on the evaluation detections too, so
    /// AP, false positives and the threshold sweep see the filtered boxes.
    fn run(
        &mut self,
        bytes: &[u8],
        eval: bool,
        transform: Option<images::ImageTransform>,
    ) -> Result<Sample> {
        let t = Instant::now();
        let mut img = crate::image::decode(bytes)?;
        let decode = ms(t.elapsed());
        if let Some(tr) = transform {
            tr.apply(&mut img);
        }
        // The verifier crops the original frame, as the service does (not timed: the service
        // keeps the decoded frame without a copy).
        let original = self.verifier.is_some().then(|| img.clone());

        let t = Instant::now();
        let scene = match self.variant {
            Some(v) => {
                let scene = crate::image_stats::ImageStats::compute(&img).scene();
                crate::model::enhance::apply(&mut img, v.modes().for_scene(scene));
                scene
            }
            None => crate::image_stats::Scene::Day,
        };
        let (chw, ctx) = self.pre.run(&img.rgb, img.width, img.height)?;
        let extra = self.family.extra_inputs(&ctx);
        let preprocess = ms(t.elapsed());

        let t = Instant::now();
        let outputs = self.backend.infer(chw, &extra)?;
        let infer = ms(t.elapsed());

        let rules = self.variant.and_then(|v| v.filters.as_ref());
        let filtered = |dets: Vec<crate::model::Detection>| match rules {
            Some(r) => {
                crate::model::filters::apply(dets, self.classes, ctx.orig_w, ctx.orig_h, scene, r).0
            }
            None => dets,
        };
        let t = Instant::now();
        let dets = filtered(self.family.postprocess(&outputs, &ctx, &self.params)?);
        let preds = crate::model::to_predictions(&dets, &ctx, self.classes, self.filter);
        let postprocess = ms(t.elapsed());
        let mut eval = if eval {
            let dets = filtered(self.family.postprocess(&outputs, &ctx, &self.eval_params)?);
            crate::model::to_predictions(&dets, &ctx, self.classes, None)
        } else {
            Vec::new()
        };
        // The verifier cascade counts as postprocess time (the service adds it to processMs).
        let (preds, postprocess) = match (self.verifier.as_mut(), original) {
            (Some(v), Some(frame)) => {
                let t = Instant::now();
                let preds = v.apply(frame, scene, preds, &mut eval)?;
                (preds, postprocess + ms(t.elapsed()))
            }
            _ => (preds, postprocess),
        };
        Ok(Sample {
            stages: [decode, preprocess, infer, postprocess],
            predictions: preds,
            eval,
        })
    }
}

/// What the timed runs of [`Bench::time`] collect.
pub(super) struct Timings {
    /// Per run: decode, preprocess, infer, postprocess, total (ms).
    samples: [Vec<f64>; 5],
    /// Per resolution bucket: pre (decode + preprocess), infer, post, total samples (ms) and
    /// the image count.
    by_res: BTreeMap<&'static str, ([Vec<f64>; 4], usize)>,
    per_image: Vec<ImageRun>,
    /// Per image: the last run's predictions at the evaluation threshold.
    eval_preds: Vec<Vec<PredBox>>,
    total_iters: usize,
    /// Wall time of the timed iterations (throughput).
    loop_time: Duration,
}

impl Timings {
    /// Add one run's stage timings (overall and to its resolution bucket); returns its total.
    fn record(&mut self, bucket: &'static str, sample: &Sample) -> f64 {
        let total: f64 = sample.stages.iter().sum();
        for (dst, v) in self.samples.iter_mut().zip(sample.stages) {
            dst.push(v);
        }
        self.samples[4].push(total);
        let e = self.by_res.entry(bucket).or_default();
        e.0[0].push(sample.stages[0] + sample.stages[1]);
        e.0[1].push(sample.stages[2]);
        e.0[2].push(sample.stages[3]);
        e.0[3].push(total);
        total
    }

    fn stage_stats(&self) -> StageStats {
        StageStats {
            decode: Stats::from_samples(&self.samples[0]),
            preprocess: Stats::from_samples(&self.samples[1]),
            infer: Stats::from_samples(&self.samples[2]),
            postprocess: Stats::from_samples(&self.samples[3]),
            total: Stats::from_samples(&self.samples[4]),
        }
    }

    /// Speed per resolution bucket with images, in [`images::RESOLUTION_BUCKETS`] order.
    fn by_resolution(&self) -> Vec<BucketSpeed> {
        images::RESOLUTION_BUCKETS
            .iter()
            .filter_map(|b| {
                let (s, n) = self.by_res.get(b)?;
                let t = Stats::from_samples(&s[3]);
                Some(BucketSpeed {
                    bucket: b.to_string(),
                    images: *n,
                    pre_p50: Stats::from_samples(&s[0]).p50,
                    infer_p50: Stats::from_samples(&s[1]).p50,
                    post_p50: Stats::from_samples(&s[2]).p50,
                    total_p50: t.p50,
                    total_p95: t.p95,
                })
            })
            .collect()
    }
}

/// The drill-down record of one image from its last run (stored predictions, most confident
/// first, at most `max_preds`; p50 timings) and that run's evaluation predictions.
pub(super) fn image_run(
    set: &ImageSet,
    img: &images::SetImage,
    last: &Sample,
    totals: &[f64],
    infers: &[f64],
    max_preds: usize,
) -> (ImageRun, Vec<PredBox>) {
    let mut preds: Vec<ImagePred> = last
        .predictions
        .iter()
        .map(|p| {
            let b = to_pred(p);
            ImagePred {
                label: b.label,
                confidence: b.confidence,
                bbox: b.bbox,
                tp: None,
            }
        })
        .collect();
    preds.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    preds.truncate(max_preds);
    let eval = last.eval.iter().map(to_pred).collect();
    let run = ImageRun {
        set: set.id.clone(),
        file: img.file.clone(),
        total_ms: Stats::from_samples(totals).p50,
        infer_ms: Stats::from_samples(infers).p50,
        preds,
        gt_matched: Vec::new(),
        counts: None,
    };
    (run, eval)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timings_record_overall_and_by_bucket() {
        let mut t = Timings {
            samples: Default::default(),
            by_res: BTreeMap::new(),
            per_image: Vec::new(),
            eval_preds: Vec::new(),
            total_iters: 3,
            loop_time: Duration::ZERO,
        };
        let sample = |stages: [f64; 4]| Sample {
            stages,
            predictions: Vec::new(),
            eval: Vec::new(),
        };
        assert_eq!(t.record("sd", &sample([1.0, 2.0, 10.0, 0.5])), 13.5);
        assert_eq!(t.record("sd", &sample([1.0, 4.0, 20.0, 0.5])), 25.5);
        assert_eq!(t.record("fhd", &sample([3.0, 3.0, 30.0, 1.0])), 37.0);
        t.by_res.entry("sd").or_default().1 += 1;
        t.by_res.entry("fhd").or_default().1 += 1;
        let s = t.stage_stats();
        assert_eq!(s.infer, Stats::from_samples(&[10.0, 20.0, 30.0]));
        assert_eq!(s.total, Stats::from_samples(&[13.5, 25.5, 37.0]));
        let b = t.by_resolution();
        // Bucket order, empty buckets omitted.
        assert_eq!(
            b.iter().map(|b| b.bucket.as_str()).collect::<Vec<_>>(),
            ["sd", "fhd"]
        );
        assert_eq!(b[0].images, 1);
        assert_eq!(b[0].pre_p50, 3.0); // decode + preprocess, nearest rank of [3, 5]
        assert_eq!(b[0].infer_p50, 10.0);
        assert_eq!(b[0].total_p95, 25.5);
        assert_eq!(b[1].post_p50, 1.0);
    }

    #[test]
    fn cancelled_is_detected_through_context() {
        let e = anyhow::Error::from(Cancelled).context("loading x");
        assert!(is_cancelled(&e));
        assert!(!is_cancelled(&anyhow::anyhow!("other")));
    }
}
