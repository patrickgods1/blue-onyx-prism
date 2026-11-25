//! The stages of one run ([`stages`], the driver behind [`super::run_training`]), one function
//! each, in order: the run's checks and its log, lock and busy marker, then [`setup_env`],
//! [`self_check`], either [`export_retry`] (an export-only run) or [`fetch_weights`],
//! [`assemble_dataset`], [`train_settings`] and [`train_attempts`] (the trainer, restarted on
//! running out of memory by [`super::attempt::next_attempt`]), and then [`Finish::run`] (install,
//! register, benchmark). Every stage reports through `report` and records the stage a failure
//! happens in through `current` ([`enter`]); both are passed apart, as `&mut`.

use super::attempt::{self, Ended, Tries};
use super::*;

/// What every stage reads: the job and the run's seams (never the report, which the stages take
/// apart, as `&mut`).
#[derive(Clone, Copy)]
struct Seams<'a> {
    job: &'a TrainJob,
    /// The recipe's base weights ([`TrainRecipe::base`]).
    base: &'static Resource,
    runner: &'a dyn CommandRunner,
    fetcher: &'a dyn Fetcher,
    cancel: &'a AtomicBool,
}

/// The scripts [`env::write_scripts`] wrote for the run.
struct Scripts {
    train: PathBuf,
    export: PathBuf,
}

/// The folders the trainer reads and writes.
struct Dirs {
    /// The prepared dataset (`<run>/data`).
    data: PathBuf,
    /// Where it writes the exported model (`<models>/.<name>.train-staging`).
    staging: PathBuf,
    /// Its checkpoints and scratch (`<run>/work`).
    work: PathBuf,
}

/// The arguments the training and the export-only commands share, in their order: the script,
/// the model (trainer, family, size, name), `--weights` when given, the dataset and input size,
/// `between` (the training's own settings), then where the script writes.
fn common_args(
    recipe: &TrainRecipe,
    name: &str,
    scripts: &Scripts,
    dirs: &Dirs,
    weights: Option<&Path>,
    between: Vec<OsString>,
) -> Vec<OsString> {
    let family = recipe.family;
    let mut args: Vec<OsString> = vec![
        scripts.train.clone().into_os_string(),
        "--trainer".into(),
        family.trainer().as_str().into(),
        "--family".into(),
        family.as_str().into(),
        "--size".into(),
        recipe.size.clone().into(),
        "--name".into(),
        name.into(),
    ];
    if let Some(w) = weights {
        args.push("--weights".into());
        args.push(w.as_os_str().to_owned());
    }
    args.extend([
        "--data".into(),
        dirs.data.clone().into_os_string(),
        "--imgsz".into(),
        recipe.imgsz.to_string().into(),
    ]);
    args.extend(between);
    args.extend([
        "--out-dir".into(),
        dirs.staging.clone().into_os_string(),
        "--work".into(),
        dirs.work.clone().into_os_string(),
        "--export-script".into(),
        scripts.export.clone().into_os_string(),
    ]);
    args
}

/// An empty `staging` folder: whatever an earlier attempt left there removed. The caller holds
/// its busy guard ([`crate::storage::Busy`]).
fn reset_staging(staging: &Path) -> Result<()> {
    manager::remove_any(staging)?;
    std::fs::create_dir_all(staging).with_context(|| format!("creating {}", staging.display()))
}

/// `<models>/.<name>.train-staging`.
fn staging_dir(models: &Path, name: &str) -> PathBuf {
    models.join(format!(".{name}.train-staging"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn stages(
    job: &TrainJob,
    held: Option<&TrainLock>,
    runner: &dyn CommandRunner,
    fetcher: &dyn Fetcher,
    hooks: &dyn Hooks,
    report: &mut dyn FnMut(TrainReport<'_>),
    cancel: &AtomicBool,
    current: &mut Option<TrainStage>,
) -> Result<RunOutcome> {
    let recipe = &job.request.recipe;
    recipe.validate()?;
    super::super::recipe::valid_model_name(&job.name)?;
    let paths = &job.paths;
    let models = job.install_dir();
    let family = recipe.family;
    let base = recipe.base()?;
    let s = Seams {
        job,
        base,
        runner,
        fetcher,
        cancel,
    };
    if models.join(format!("{}.onnx", job.name)).exists() {
        bail!(
            "{} already exists; choose another model name",
            models.join(format!("{}.onnx", job.name)).display()
        );
    }
    if hooks.config_has_model(&job.name) {
        bail!(
            "the config already has a model named {} (added after this run was queued); remove \
             that model first, or train again under another name",
            job.name
        );
    }
    if let Some(r) = &job.export_only {
        if !r.checkpoint.exists() {
            bail!(
                "the checkpoint {} is gone; train the model again",
                r.checkpoint.display()
            );
        }
        preflight::check_disk(vec![preflight::VolumeNeed {
            dir: models.to_path_buf(),
            bytes: super::super::retry::EXPORT_BYTES,
            advice: preflight::MODELS_ADVICE,
        }])?;
    }
    std::fs::create_dir_all(&job.run_dir)
        .with_context(|| format!("creating {}", job.run_dir.display()))?;
    let mut log = Log::open(&job.run_dir.join(LOG_FILE));
    log.line(&format!(
        "# {} {} on {} ({})",
        crate::long_version(),
        recipe.label(),
        job.accelerator.describe(),
        job.id
    ));
    // Taken here, or held by the task this run is part of (for the whole task).
    let _lock = match held {
        Some(l) if l.path() == paths.train_lock_file() => None,
        Some(l) => bail!(
            "the training lock held is {}, not this run's {}",
            l.path().display(),
            paths.train_lock_file().display()
        ),
        None => Some(TrainLock::try_acquire(&paths.train_lock_file())?),
    };
    // The run's liveness marker (`<run dir>.busy`, [`crate::storage::busy`]) for the whole run: a
    // queue that opens the folder, Remove with files and Retry export tell from it a run another
    // process trains from one a crash left ([`super::super::service::run_in_use`]). Its folder
    // stays when the run ends. A fold of a dataset check (`held`) is no queue run and holds none.
    let _alive = held
        .is_none()
        .then(|| crate::storage::Busy::hold(&job.run_dir));

    // The memory commit, before the run's first Python process (the environment's tools, the
    // self-check, the trainer). Each run of a queue (a seed, another model) is checked at its own
    // start; an export retry trains nothing and is not checked.
    if job.export_only.is_none() {
        super::super::commit::gate(
            &super::super::trainer_commit_need(family, job.workers),
            (job.headroom)().as_ref(),
            job.request.ignore_memory_check,
            &mut |l| {
                log.line(l);
                report(TrainReport::Line(l));
            },
        )?;
    }

    // 1. Environment.
    setup_env(&s, &mut log, report, current)?;
    check(cancel)?;

    // 2. Self-check.
    let scripts = self_check(&s, &mut log, report, current)?;
    check(cancel)?;
    let finish = Finish {
        job,
        base,
        hooks,
        cancel,
    };
    if let Some(retry) = &job.export_only {
        return export_retry(&s, retry, &scripts, &finish, &mut log, report, current);
    }

    // 3. Weights (and the public subset).
    let (wdir, public_dir) = fetch_weights(&s, report, current)?;

    // 4. Dataset.
    let (opts, prepared) = assemble_dataset(&s, &public_dir, &mut log, report, current)?;
    check(cancel)?;

    // 5. Train (the script also exports and checks the ONNX).
    let settings = train_settings(job, &prepared, &mut log, report, current);
    let staging = staging_dir(models, &job.name);
    // Held until this returns, then removed (also on the early errors that skip their own
    // cleanup): a crash leaves it to the storage sweep ([`crate::storage::sweep`]).
    let _staging = crate::storage::Busy::hold(&staging).remove_target_on_drop();
    reset_staging(&staging)?;
    let dirs = Dirs {
        data: job.run_dir.join("data"),
        staging,
        work: job.run_dir.join("work"),
    };
    let (finished_recipe, done) = train_attempts(
        &s, &settings, &scripts, &dirs, &wdir, &opts, &mut log, report,
    )?;
    check(cancel)?;
    finish.run(
        &finished_recipe,
        prepared,
        done,
        &dirs.staging,
        &mut log,
        report,
        current,
    )
}

/// Stage 1: the training environment (uv, Python, the lock's packages, the Visual C++ runtime),
/// its progress mapped onto the setup stage.
fn setup_env(
    s: &Seams<'_>,
    log: &mut Log,
    report: &mut dyn FnMut(TrainReport<'_>),
    current: &mut Option<TrainStage>,
) -> Result<()> {
    let job = s.job;
    enter(
        current,
        report,
        TrainStage::Setup,
        0,
        "checking the toolchain",
    );
    let ej = env::env_job(&job.paths, s.base, job.platform);
    env::prepare_train_env(
        &ej,
        job.accelerator.lane(),
        s.runner,
        s.fetcher,
        &mut |p| match p {
            Progress::Stage {
                stage,
                percent,
                detail,
            } => {
                // The export bar's share of these stages ends at its self-check (71%).
                let pct = u32::from(stage.overall(percent)) * 100 / 71;
                let d = format!("{}: {detail}", stage.label());
                report(TrainReport::Stage {
                    stage: TrainStage::Setup,
                    percent: u8::try_from(pct.min(100)).unwrap_or(100),
                    detail: &d,
                });
            }
            Progress::Line { line, .. } => {
                log.line(line);
                report(TrainReport::Line(line));
            }
        },
        s.cancel,
    )
    .map_err(|e| {
        if e.cancelled {
            anyhow!(Cancelled)
        } else {
            anyhow!(
                "{}{}",
                e.stage
                    .map(|s| format!("while {}: ", s.label()))
                    .unwrap_or_default(),
                e.message
            )
        }
    })?;
    Ok(())
}

/// Stage 2: the scripts written, then `train_model.py --self-check` (PyTorch imports and sees the
/// run's GPU), skipped when nothing it checks changed since the last one passed
/// ([`super::super::self_check`]; any doubt runs it).
fn self_check(
    s: &Seams<'_>,
    log: &mut Log,
    report: &mut dyn FnMut(TrainReport<'_>),
    current: &mut Option<TrainStage>,
) -> Result<Scripts> {
    let job = s.job;
    let paths = &job.paths;
    enter(
        current,
        report,
        TrainStage::SelfCheck,
        0,
        "importing PyTorch and the trainer",
    );
    let (train_script, export_script) = env::write_scripts(paths)?;
    let trainer = job.request.recipe.family.trainer().as_str();
    // Skipped when nothing it checks changed since the last check that passed; any doubt runs it.
    let check_marker = super::super::self_check::marker_path(paths);
    let check_key = super::super::self_check::key(trainer, &job.accelerator.device_arg());
    let check_fp =
        super::super::self_check::fingerprint(paths, job.platform, &job.accelerator, trainer);
    let skipped_since = check_fp.as_ref().and_then(|fp| {
        super::super::self_check::skip_since(
            &check_marker,
            &check_key,
            fp,
            std::time::SystemTime::now(),
        )
    });
    if let Some(at) = skipped_since {
        let l = format!(
            "self-check skipped (unchanged since {}: the same environment, scripts, GPU and \
             trainer passed it then)",
            crate::clock::display(std::time::UNIX_EPOCH + Duration::from_secs(at))
        );
        log.line(&format!("# {l}"));
        report(TrainReport::Line(&l));
        enter(current, report, TrainStage::SelfCheck, 100, &l);
        return Ok(Scripts {
            train: train_script,
            export: export_script,
        });
    }
    let lane = job.accelerator.lane();
    let cmd = env::train_python_cmd(
        paths,
        job.platform,
        Stage::SelfCheck,
        &paths.scripts,
        SELF_CHECK_TIMEOUT,
        vec![
            train_script.clone().into_os_string(),
            "--self-check".into(),
            "--trainer".into(),
            trainer.into(),
            // The test kernel runs on the run's GPU (another GPU may have other kernels).
            "--device".into(),
            job.accelerator.device_arg().into(),
        ],
    );
    let mut seen: Option<TrainEvent> = None;
    s.runner
        .run(
            &cmd,
            &mut |l| {
                log.line(l);
                report(TrainReport::Line(l));
                if let Some(ev @ TrainEvent::SelfCheck { .. }) = progress::parse_line(l) {
                    seen = Some(ev);
                }
            },
            s.cancel,
        )
        .map_err(|e| env::explain_train_failure(e, paths, job.platform, TRAINING_TASK))
        .context("the training environment's self-check failed")?;
    match seen {
        Some(TrainEvent::SelfCheck {
            cuda, mps, cuda_ok, ..
        }) => match &job.accelerator {
            Accelerator::Cuda { name, .. } if !cuda => bail!(
                "PyTorch does not see the NVIDIA GPU {name}: the driver may be too old for the \
                 PyTorch {} build; update the NVIDIA driver and try again",
                lane.as_str()
            ),
            Accelerator::Cuda {
                name,
                compute_capability,
                ..
            } if cuda_ok == Some(false) => bail!(
                "a test kernel failed on {name} (compute capability {compute_capability}): the \
                 PyTorch {} build has no kernels for it",
                lane.as_str()
            ),
            Accelerator::Mps { .. } if !mps => bail!(
                "PyTorch reports no Apple MPS device (macOS 14 or newer on Apple silicon is needed)"
            ),
            _ => {}
        },
        _ => bail!("the self-check printed no device report; the training script is damaged"),
    }
    if let Some(fp) = &check_fp
        && let Err(e) = super::super::self_check::record(
            &check_marker,
            &check_key,
            fp,
            std::time::SystemTime::now(),
        )
    {
        tracing::warn!("training {}: {e:#} (the next run checks again)", job.id);
    }
    enter(
        current,
        report,
        TrainStage::SelfCheck,
        100,
        "PyTorch sees the GPU",
    );
    Ok(Scripts {
        train: train_script,
        export: export_script,
    })
}

/// An export-only run ([`TrainJob::export_only`]) after its self-check: the failed run's best
/// checkpoint exported on its own dataset, then [`Finish::run`].
fn export_retry(
    s: &Seams<'_>,
    retry: &ExportRetry,
    scripts: &Scripts,
    finish: &Finish<'_>,
    log: &mut Log,
    report: &mut dyn FnMut(TrainReport<'_>),
    current: &mut Option<TrainStage>,
) -> Result<RunOutcome> {
    let job = s.job;
    let (paths, models) = (&job.paths, job.install_dir());
    let data_dir = job.run_dir.join("data");
    let prepared = dataset::read_prepared(&data_dir)?;
    let staging = staging_dir(models, &job.name);
    // Held until this returns, then removed: a crash leaves it to the storage sweep.
    let _staging = crate::storage::Busy::hold(&staging).remove_target_on_drop();
    reset_staging(&staging)?;
    enter(
        current,
        report,
        TrainStage::Train,
        0,
        "exporting the best checkpoint",
    );
    let dirs = Dirs {
        data: data_dir,
        staging,
        work: super::super::retry::work_dir(&job.run_dir),
    };
    let mut args = common_args(
        &job.request.recipe,
        &job.name,
        scripts,
        &dirs,
        None,
        Vec::new(),
    );
    args.push("--export-only".into());
    args.push(retry.checkpoint.clone().into_os_string());
    let cmd = env::train_python_cmd(
        paths,
        job.platform,
        Stage::Export,
        &job.run_dir,
        EXPORT_ONLY_TIMEOUT,
        args,
    );
    let mut done: Option<TrainEvent> = None;
    let res = s.runner.run(
        &cmd,
        &mut |l| {
            log.line(l);
            match progress::parse_line(l) {
                Some(ev @ TrainEvent::Done { .. }) => done = Some(ev),
                Some(TrainEvent::Phase { detail, .. }) => report(TrainReport::Stage {
                    stage: TrainStage::Train,
                    percent: 50,
                    detail: &detail,
                }),
                Some(_) => {}
                None => report(TrainReport::Line(l)),
            }
        },
        s.cancel,
    );
    if let Err(e) = res {
        let _ = manager::remove_any(&dirs.staging);
        let e = explain_full_disk(e, models);
        return Err(env::explain_train_failure(
            e,
            paths,
            job.platform,
            "the export retry",
        ));
    }
    check(s.cancel)?;
    finish.run(
        &job.request.recipe,
        prepared,
        done,
        &dirs.staging,
        log,
        report,
        current,
    )
}

/// Stage 3: the pinned base weights, and the public subset when the request mixes it in, each
/// downloaded unless installed. Returns their folders (weights, public subset).
fn fetch_weights(
    s: &Seams<'_>,
    report: &mut dyn FnMut(TrainReport<'_>),
    current: &mut Option<TrainStage>,
) -> Result<(PathBuf, PathBuf)> {
    let job = s.job;
    let wdir = job.paths.weights_dir(s.base);
    let mut downloads: Vec<(&'static Resource, PathBuf)> = vec![(s.base, wdir.clone())];
    let public_dir = super::super::public_subset_dir(&job.root);
    if job.request.include_public {
        let Some(public) = catalog::lila_night() else {
            bail!(
                "the public night subset is not pinned in this build (its manifest lists no \
                 image); train without it"
            );
        };
        super::super::check_public_download(&job.root, job.request.allow_large)?;
        downloads.push((public, public_dir.clone()));
    }
    let missing: Vec<_> = downloads
        .iter()
        .filter(|(r, d)| !manager::is_installed(r, d))
        .collect();
    if missing.is_empty() {
        enter(
            current,
            report,
            TrainStage::Weights,
            100,
            "already downloaded",
        );
        return Ok((wdir, public_dir));
    }
    let total: u64 = missing.iter().map(|(r, _)| r.size()).sum();
    let mut done = 0u64;
    for (r, dir) in missing {
        enter(
            current,
            report,
            TrainStage::Weights,
            manager::percent(done, total),
            r.title,
        );
        s.fetcher.fetch(
            r,
            dir,
            &mut |b, _| {
                let d = format!(
                    "{} {} of {} MB",
                    r.title,
                    (done + b) / 1_000_000,
                    total / 1_000_000
                );
                report(TrainReport::Stage {
                    stage: TrainStage::Weights,
                    percent: manager::percent(done + b, total),
                    detail: &d,
                });
            },
            s.cancel,
        )?;
        if !manager::is_installed(r, dir) {
            bail!(
                "{} is not complete in {} after the download",
                r.title,
                dir.display()
            );
        }
        done += r.size();
        check(s.cancel)?;
    }
    Ok((wdir, public_dir))
}

/// Stage 4: the request's exports (and the public subset) read, planned, checked against the
/// disk and written into `<run>/data` ([`dataset::write_reusing`], linking from an identical
/// dataset of `job.reuse_from`). Returns the options it was assembled with and what was written.
fn assemble_dataset(
    s: &Seams<'_>,
    public_dir: &Path,
    log: &mut Log,
    report: &mut dyn FnMut(TrainReport<'_>),
    current: &mut Option<TrainStage>,
) -> Result<(dataset::Options, Prepared)> {
    let job = s.job;
    let recipe = &job.request.recipe;
    enter(
        current,
        report,
        TrainStage::Dataset,
        0,
        "reading the exports",
    );
    if job.request.datasets.is_empty() {
        bail!("no dataset: choose at least one review export folder");
    }
    let sources = read_sources(
        &job.request.datasets,
        job.request.include_public.then_some(public_dir),
    )?;
    let opts = dataset::Options {
        // Train / valid images at what the trainer reads of them at most; the held-out frames
        // stay full size.
        prepared_short_side: Some(recipe.prepared_short_side(
            recipe.multi_scale_on(job.accelerator.memory_mb(), job.accelerator.target()),
        )),
        ..request_options(&job.options, &job.request)
    };
    let plan = preflight::planned(&sources, &opts)?;
    preflight::check_plan(&plan, &sources, &opts)?;
    let data_dir = job.run_dir.join("data");
    manager::remove_any(&data_dir)?;
    preflight::check_disk(preflight::run_needs(
        &plan,
        &sources,
        recipe,
        &job.run_dir,
        job.install_dir(),
    )?)?;
    if let Some(w) = preflight::gpu_warning(recipe, &job.accelerator) {
        log.line(&format!("gpu: {w}"));
        tracing::warn!("training {}: {w}", job.name);
    }
    let (prepared, reused) =
        dataset::write_reusing(&sources, &plan, &opts, &data_dir, &job.reuse_from, s.cancel)
            .map_err(|e| explain_full_disk(e, &job.run_dir))?;
    if let Some(l) = reused.describe() {
        log.line(&format!("dataset: {l}"));
        report(TrainReport::Line(&l));
    }
    if let Some(short) = opts.prepared_short_side {
        log.line(&format!(
            "dataset: train / valid images with a short side over {short} px written at {short} \
             px (what the trainer reads at most); held-out frames full size"
        ));
    }
    for w in &prepared.warnings {
        log.line(&format!("dataset: {w}"));
    }
    if let Some(t) = dataset::valid_groups_text(&prepared.valid_groups) {
        log.line(&format!("dataset: {t}"));
    }
    if let Some(b) = &prepared.balance_summary {
        log.line(&format!("dataset: balanced: {b}"));
    }
    report(TrainReport::Prepared(&prepared));
    let detail = format!(
        "{} training, {} validation, {} held-out images; classes {}",
        prepared.train_images,
        prepared.valid_images,
        prepared.heldout_images,
        prepared.classes.join(", ")
    );
    enter(current, report, TrainStage::Dataset, 100, &detail);
    if recipe.family.service_family() == ModelFamilyKind::Detr && prepared.classes.len() == 91 {
        bail!("91 classes look like a COCO category-id head to the service; add or drop a class");
    }
    Ok((opts, prepared))
}

/// What the training command trains with beyond the request's recipe ([`train_settings`]).
struct TrainSettings {
    /// The recipe with the patience the validation split's size calls for.
    recipe: TrainRecipe,
    /// The patience was doubled for a small validation split.
    patience_doubled: bool,
    /// RF-DETR's epochs of a lowered learning rate at a stop point (0 = none).
    anneal: u32,
    lr: Option<super::super::recipe::LearningRate>,
    /// The family's convergence rule (one for every trainer, `Convergence` in the script).
    rec: super::super::recipe::Recommended,
}

/// Stage 5's start: the patience, anneal and learning rate the dataset calls for, written to the
/// log (early stopping off is reported too).
fn train_settings(
    job: &TrainJob,
    prepared: &Prepared,
    log: &mut Log,
    report: &mut dyn FnMut(TrainReport<'_>),
    current: &mut Option<TrainStage>,
) -> TrainSettings {
    let recipe = &job.request.recipe;
    enter(
        current,
        report,
        TrainStage::Train,
        0,
        "starting the trainer",
    );
    // A small validation split: a recommended patience doubles (its metrics are noisy).
    let (patience, convergence_line) = recipe.convergence_patience(prepared.valid_images);
    let patience_doubled = patience != recipe.patience;
    let recipe = TrainRecipe {
        patience,
        ..recipe.clone()
    };
    // RF-DETR: at a stop point the learning rate is lowered over a few more epochs first.
    let anneal = recipe.anneal_epochs();
    let anneal_text = if anneal > 0 {
        format!(
            "; at a stop point the learning rate is lowered over {anneal} more epochs, then \
             training stops"
        )
    } else {
        String::new()
    };
    log.line(&format!("# {convergence_line}{anneal_text}"));
    if let Some(seed) = job.request.train_seed {
        log.line(&format!(
            "# training seed {seed} (the frames are the same for every seed)"
        ));
    }
    // RF-DETR on a small training split: a lower learning rate than rfdetr's default. Unique
    // frames: balancing repeats do not make a dataset larger.
    let lr = recipe.effective_lr(prepared.train_frames);
    log.line(&format!("# {}", recipe.lr_line(prepared.train_frames)));
    if !recipe.early_stopping() {
        let w = format!(
            "early stopping is off (patience {} of {} epochs): every epoch runs; the best one is \
             still exported",
            recipe.patience, recipe.epochs
        );
        log.line(&format!("# {w}"));
        report(TrainReport::Line(&w));
    }
    TrainSettings {
        rec: super::super::recipe::recommended(recipe.family),
        recipe,
        patience_doubled,
        anneal,
        lr,
    }
}

/// The training command's arguments at `batch` and `workers`: [`common_args`] with the base
/// weights and the training's settings, then the learning rate, the convergence stop, the stop
/// file and RF-DETR's multi-scale choice.
fn train_args(
    job: &TrainJob,
    t: &TrainSettings,
    scripts: &Scripts,
    dirs: &Dirs,
    weights: &Path,
    (batch, workers): (u32, u32),
) -> Vec<OsString> {
    let recipe = &t.recipe;
    let family = recipe.family;
    // RF-DETR's multi-scale training (`auto`: on when the estimate fits batch 4 with it).
    let multi_scale = recipe.multi_scale_on(job.accelerator.memory_mb(), job.accelerator.target());
    let between: Vec<OsString> = vec![
        "--epochs".into(),
        recipe.epochs.to_string().into(),
        "--batch".into(),
        batch.to_string().into(),
        "--device".into(),
        job.accelerator.device_arg().into(),
        "--workers".into(),
        workers.to_string().into(),
        "--seed".into(),
        job.request
            .trainer_seed(job.options.seed)
            .to_string()
            .into(),
        "--aug-dark".into(),
        recipe.augment.dark.to_string().into(),
        "--aug-noise".into(),
        recipe.augment.noise.to_string().into(),
        "--aug-gray".into(),
        recipe.augment.gray.to_string().into(),
    ];
    let mut args = common_args(recipe, &job.name, scripts, dirs, Some(weights), between);
    if let Some(lr) = t.lr {
        args.push("--lr".into());
        args.push(lr.lr.to_string().into());
        if let Some(e) = lr.encoder {
            args.push("--lr-encoder".into());
            args.push(e.to_string().into());
        }
    }
    // Only when set: the script resolves each trainer's own default.
    if let Some(s) = recipe.lr_schedule {
        args.push("--lr-scheduler".into());
        args.push(s.as_str().into());
    }
    if let Some(f) = recipe.lr_min_factor {
        args.push("--lr-min-factor".into());
        args.push(f.to_string().into());
    }
    if !recipe.select_metric.is_default() {
        args.push("--select-metric".into());
        args.push(recipe.select_metric.id().into());
    }
    // The convergence stop (one rule for every trainer, `Convergence` in the script).
    let rec = &t.rec;
    for (k, v) in [
        ("--patience", recipe.patience.to_string()),
        ("--min-epochs", rec.min_epochs.to_string()),
        ("--min-delta", rec.min_delta.to_string()),
        ("--smooth", rec.smooth.to_string()),
        ("--anneal", t.anneal.to_string()),
    ] {
        args.push(k.into());
        args.push(v.into());
    }
    args.push("--stop-file".into());
    args.push(stop_file(&job.run_dir).into_os_string());
    if family == TrainFamily::Rfdetr {
        args.push("--multi-scale".into());
        args.push(if multi_scale { "on" } else { "off" }.into());
    }
    args
}

/// The data loader workers of the first attempt: the automatic count ([`Workers::Auto`]: the
/// limit of the memory free now, then what earlier runs of this machine and key taught,
/// [`super::super::workers::learned`], never above that limit) or the configured one, each
/// logged. Returns (workers, the commit seen when choosing, the automatic limit).
fn first_workers(
    job: &TrainJob,
    recipe: &TrainRecipe,
    machine: &super::super::workers::Machine,
    log: &mut Log,
) -> (u32, Option<super::super::Commit>, Option<u32>) {
    match job.workers {
        Workers::Auto => {
            let c = super::super::workers_now(recipe.family.trainer());
            log.line(&format!("# {}", c.describe()));
            let (n, why) = super::super::workers::learned(
                machine,
                &super::super::workers::WorkerKey::of(recipe),
                &job.past_runs,
                c.workers,
            );
            let line = super::super::workers::sentence(n, &why, c.workers);
            log.line(&format!("# {line}"));
            tracing::info!("training {}: {line}", job.name);
            (n, c.commit, Some(c.workers))
        }
        Workers::Fixed(n) => {
            log.line(&format!("# {}", super::super::workers::forced_sentence(n)));
            (n, None, None)
        }
    }
}

/// Stage 5: the trainer, run until an attempt ends without a restart ([`attempt::next_attempt`]:
/// out of memory early, it restarts with fewer workers or a smaller batch, the user's Finish
/// request carried over), then its resources reported. Returns the recipe at the batch that
/// trained and the script's `done` report; a failure removes `staging` and says why.
#[allow(clippy::too_many_arguments)]
fn train_attempts(
    s: &Seams<'_>,
    t: &TrainSettings,
    scripts: &Scripts,
    dirs: &Dirs,
    wdir: &Path,
    opts: &dataset::Options,
    log: &mut Log,
    report: &mut dyn FnMut(TrainReport<'_>),
) -> Result<(TrainRecipe, Option<TrainEvent>)> {
    let job = s.job;
    let paths = &job.paths;
    let recipe = &t.recipe;
    let family = recipe.family;
    let base = s.base;
    let weights = match family.trainer() {
        catalog::Trainer::Transformers => wdir.to_path_buf(),
        _ => wdir.join(
            base.parts
                .first()
                .ok_or_else(|| anyhow!("{} has no file", base.id))?
                .file_name,
        ),
    };
    let machine = super::super::workers::Machine::now(&job.accelerator);
    let (workers, start_commit, limit) = first_workers(job, recipe, &machine, log);
    let loaders = super::super::worker_loaders(family.trainer());
    // The latest attempt that ran out of host memory (the fewest workers): its workers and
    // median speed.
    let mut host_oom_at: Option<(u32, Option<f64>)> = None;
    let mut at = Tries {
        batch: recipe.batch,
        workers,
        gpu_retries: 0,
        host_retries: 0,
    };
    let mut note: Option<String> = None;
    // A finish request of an earlier run does not end this one at its first epoch. Removed once:
    // one made during an attempt that is retried carries over to the retry (QA-103).
    let stop = stop_file(&job.run_dir);
    if let Err(e) = std::fs::remove_file(&stop)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        bail!("cannot remove the old stop request {}: {e}", stop.display());
    }
    let (res, done, usage, attempt) = loop {
        let mut cmd = env::train_python_cmd(
            paths,
            job.platform,
            Stage::Export,
            &job.run_dir,
            job.timeout,
            train_args(job, t, scripts, dirs, &weights, (at.batch, at.workers)),
        );
        if matches!(job.accelerator, Accelerator::Mps { .. }) {
            for (k, v) in MPS_WATERMARKS {
                // A value in the service's own environment wins (e.g. 0.0 = no limit).
                if std::env::var_os(k).is_none() {
                    cmd.env.push((k.to_string(), v.into()));
                }
            }
        }
        let started = Instant::now();
        let mut tp = TrainProgress {
            note: note.clone(),
            patience: recipe.patience,
            min_delta: t.rec.min_delta_f64(),
            min_epochs: t.rec.min_epochs,
            smooth: t.rec.smooth,
            batch_size: Some(at.batch),
            imgsz: Some(recipe.imgsz),
            settings: Some(progress::RunSettings {
                lr: t.lr.map(|l| l.lr),
                lr_encoder: t.lr.and_then(|l| l.encoder),
                lr_schedule: recipe.effective_lr_schedule(),
                lr_min_factor: recipe.effective_lr_min_factor(),
                select_metric: recipe.select_metric,
                epochs: recipe.epochs,
                patience: recipe.patience,
                patience_doubled: t.patience_doubled,
                batch: recipe.batch,
                workers: Some(at.workers),
                loaders: Some(loaders),
                seed: job.request.train_seed,
                scene: Some(job.request.scene.as_str().to_string()),
                gpu: Some(job.accelerator.name().to_string()),
                ignore: Some(opts.ignore),
                negative_share: Some(opts.negative_share),
                static_cap: Some(
                    opts.static_cap
                        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX)),
                ),
            }),
            ..TrainProgress::default()
        };
        let mut done: Option<TrainEvent> = None;
        // Commit, RAM and page-ins from here to the export (the baseline is read first).
        let sampler = super::super::workers::sampler::Sampler::start();
        let res = s.runner.run(
            &cmd,
            &mut |l| {
                log.line(l);
                match progress::parse_line(l) {
                    Some(ev @ TrainEvent::Done { .. }) => done = Some(ev),
                    Some(ev) => {
                        tp.apply(&ev, started.elapsed().as_secs_f64());
                        if tp.phase.is_some() {
                            sampler.stop();
                        }
                        let pct = super::super::estimate::to_u64(tp.fraction() * 100.0).min(100);
                        let d = tp.describe();
                        report(TrainReport::Stage {
                            stage: TrainStage::Train,
                            percent: u8::try_from(pct).unwrap_or(100),
                            detail: &d,
                        });
                        report(TrainReport::Progress(&tp));
                    }
                    None => report(TrainReport::Line(l)),
                }
            },
            s.cancel,
        );
        let usage = sampler.finish();
        let speed = (
            super::super::workers::median_metric(&tp.history, "img_s"),
            super::super::workers::median_metric(&tp.history, "data_wait"),
        );
        let lines = res.as_ref().err().and_then(|e| {
            e.downcast_ref::<crate::resources::export::ChildFailed>()
                .map(|c| c.lines.as_slice())
        });
        let end = Ended {
            host_oom: lines.is_some_and(|l| host_out_of_memory_line(l).is_some()),
            gpu_oom: lines.is_some_and(|l| out_of_memory_line(l).is_some()),
            cancelled: s.cancel.load(Ordering::Relaxed),
            keep: keeps_checkpoints(&tp, family.trainer(), &dirs.work),
            epochs: tp.history.len(),
        };
        if end.host_oom {
            // Each restart halves the workers: the latest such attempt has the fewest.
            host_oom_at = Some((at.workers, speed.0));
        }
        if let Some(retry) = attempt::next_attempt(&end, &at) {
            at = retry.next(at);
            let msg = retry.message();
            tracing::warn!("training {}: {msg}", job.id);
            log.line(&format!("# {msg}"));
            report(TrainReport::Line(&msg));
            if let Some(b) = retry.new_batch() {
                report(TrainReport::Batch(b));
            }
            report(TrainReport::Stage {
                stage: TrainStage::Train,
                percent: 0,
                detail: &msg,
            });
            // The stop file lives under `work`, which the retry starts without.
            let finish = std::fs::read(&stop).ok();
            let _ = manager::remove_any(&dirs.work);
            reset_staging(&dirs.staging)?;
            if let Some(request) = finish {
                // QA-103: the user's Finish now holds for the retry, which stops after its first
                // epoch and exports it, rather than training to the end.
                stop.parent()
                    .map_or(Ok(()), std::fs::create_dir_all)
                    .and_then(|()| std::fs::write(&stop, request))
                    .with_context(|| format!("keeping the stop request {}", stop.display()))?;
                let line = "finish requested: the retry stops after its first epoch";
                log.line(&format!("# {line}"));
                report(TrainReport::Line(line));
            }
            note = Some(msg);
            continue;
        }
        break (res, done, usage, speed);
    };
    let workers = at.workers;
    let resources = super::super::workers::RunResources {
        workers,
        loaders,
        forced: matches!(job.workers, Workers::Fixed(_)),
        limit,
        machine,
        img_s: attempt.0,
        data_wait: attempt.1,
        commit_start: usage.commit_start,
        commit_peak: usage.commit_peak,
        commit_limit_at_peak: usage.commit_limit_at_peak,
        ram_start: usage.ram_start,
        ram_min: usage.ram_min,
        page_ins_peak: usage.page_ins_peak,
        samples: usage.samples,
        host_oom: host_oom_at.map(|(n, _)| n),
        // The measured attempt's own speed is `img_s`.
        host_oom_img_s: host_oom_at
            .filter(|(n, _)| *n != workers)
            .and_then(|(_, v)| v),
    };
    log.line(&format!("# {}", resources.describe()));
    report(TrainReport::Resources(&resources));
    let finished_recipe = TrainRecipe {
        batch: at.batch,
        ..recipe.clone()
    };
    if let Err(e) = res {
        let _ = manager::remove_any(&dirs.staging);
        if super::super::import::disk::is_disk_full(&e) {
            return Err(explain_full_disk(e, &job.run_dir));
        }
        if let Some(why) = e
            .downcast_ref::<crate::resources::export::ChildFailed>()
            .and_then(|c| {
                let host = HostMemory {
                    windows: job.platform.os == "windows",
                    workers,
                    commit: start_commit,
                };
                classify_train_failure(&c.lines, &finished_recipe, &host)
            })
        {
            return Err(e.context(why));
        }
        return Err(env::explain_train_failure(
            e,
            paths,
            job.platform,
            TRAINING_TASK,
        ));
    }
    Ok((finished_recipe, done))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dirs() -> (Scripts, Dirs) {
        let p = |s: &str| PathBuf::from("r").join(s);
        (
            Scripts {
                train: p("train_model.py"),
                export: p("export_model.py"),
            },
            Dirs {
                data: p("data"),
                staging: p("staging"),
                work: p("work"),
            },
        )
    }

    fn text(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|a| a.to_string_lossy().replace('\\', "/"))
            .collect()
    }

    #[test]
    fn both_commands_share_their_start_and_end() {
        let recipe = TrainRecipe::new(TrainFamily::Yolo26, "n").unwrap();
        let (scripts, dirs) = dirs();
        let export = common_args(&recipe, "m", &scripts, &dirs, None, Vec::new());
        assert_eq!(
            text(&export),
            [
                "r/train_model.py",
                "--trainer",
                "ultralytics",
                "--family",
                "yolo26",
                "--size",
                "n",
                "--name",
                "m",
                "--data",
                "r/data",
                "--imgsz",
                &recipe.imgsz.to_string(),
                "--out-dir",
                "r/staging",
                "--work",
                "r/work",
                "--export-script",
                "r/export_model.py",
            ]
        );
        let train = common_args(
            &recipe,
            "m",
            &scripts,
            &dirs,
            Some(Path::new("w.pt")),
            vec!["--epochs".into(), "9".into()],
        );
        let train = text(&train);
        // The weights after the name, the training's own settings after the input size.
        assert_eq!(train[9..11], ["--weights", "w.pt"]);
        assert_eq!(train[15..17], ["--epochs", "9"]);
        let mut without: Vec<String> = train.clone();
        without.drain(15..17);
        without.drain(9..11);
        assert_eq!(without, text(&export));
    }

    #[test]
    fn staging_is_reset_to_an_empty_folder() {
        let t = crate::test_dirs::TempDir::new("job-staging");
        let staging = staging_dir(&t, "night");
        assert_eq!(staging, t.join(".night.train-staging"));
        reset_staging(&staging).unwrap();
        assert!(staging.is_dir());
        std::fs::write(staging.join("left.onnx"), b"x").unwrap();
        reset_staging(&staging).unwrap();
        assert_eq!(std::fs::read_dir(&staging).unwrap().count(), 0);
    }
}
