//! The benchmark at the end of a run: the trained model on its held-out frames (a `dir:` set)
//! and optional built-in sets, through the regular [`BenchmarkService`] (so the results land in
//! `benchmark.json` and its history like any other run), summarized for the Train tab.
//! It runs on the chosen devices one after the other (`training.benchmark_devices`, the Train
//! form), else on the one device the model would load on, never on every device of the machine
//! (a GTX 1070 Ti + UHD 630 box hard-hung within seconds of that sweep starting, twice).
//! A cancelled run waits for its own benchmark to stop (up to [`STOP_WAIT`]) before it returns,
//! so the queue never starts its next GPU job beside a model compile that cannot be interrupted.
//!
//! Per-scene AP comes from the benchmark's per-tag breakdowns of the held-out manifest's own tags
//! ([`dataset::heldout_tags`]: `heldout:day|night|ir|all|in-scope`), which are exact AP over
//! those images and never mix in the built-in sets run beside it (their bare `day` / `night` tags
//! would). Held-out sets of older versions only carry the bare scene tags: they are read when the
//! held-out set was the only dataset of the run, and give no `in_scope`. The Review queue's random
//! picks among the held-out frames (`heldout:random`) are read the same way into
//! [`BenchSummary::random`].
//!
//! Recall per class ([`BenchSummary::recall_cells`]) comes from the same tags' per-class recall
//! (`Breakdown::per_class`, every class the held-out set scores, custom animals included): per
//! scene (`heldout:<scene>`, `heldout:all`), per camera (`heldout:cam:<camera>`) and per camera in
//! a scene (`heldout:camscene:<scene>:<camera>`), only when the held-out set's own tags are there
//! ([`cells_of`]). [`measured_recall`] folds the runs' cells into the one table the Review queue
//! steers by ([`Measured`]; its rule is documented there): per scene the newest finished run that
//! trained on the scene, over every scene the newest that trained on all of them.
//!
//! The current default model then runs on the same sets and devices ([`baseline_skip`] says when
//! it cannot), so the Use card compares like with like ([`compare_text`]). Its results stay in
//! the run's folder: a set of a few hundred frames must not replace the default model's saved
//! benchmark, its threshold advice or its `auto` device order. A run of a comparison group whose
//! baseline inputs equal those of a baseline already measured for another run of the group (the
//! same default model file, held-out frames, devices, settings and build: [`baseline_key`]) takes
//! that result over instead of benchmarking again ([`reusable_baseline`]; the run's log says
//! "baseline reused from <run>"), as the seeds of one group do.

use super::dataset;
use super::file_id::FileId;
use super::job::{Baseline, BenchSummary, CameraSceneRecall, RecallCells, SceneAp, SubsetScore};
use crate::backend::DeviceSpec;
use crate::benchmark::report::ModelResult;
use crate::benchmark::service::{BenchmarkService, RunContext, RunRequest, RunState};
use crate::benchmark::{Breakdown, ClassRecall};
use crate::collect::labels::recall::{self, Measured};
use crate::config::DatasetRef;
use crate::resources::export::Cancelled;
use crate::sync::LockExt;
use anyhow::{Result, anyhow, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Longest wait for another benchmark to finish before this one starts.
pub const BUSY_WAIT: Duration = Duration::from_secs(60 * 60);
/// Longest wait for a cancelled benchmark of this run to stop before the run returns: a model
/// compile in progress cannot be interrupted (minutes for a large model on an OpenVINO GPU), and
/// the queue must not start its next GPU job beside it (one GPU job at a time, audit RA-037).
/// Bounded, so a compile that hangs cannot hold a cancel for good.
pub const STOP_WAIT: Duration = Duration::from_secs(10 * 60);
/// How often that wait says it is still waiting.
const STOP_LOG_EVERY: Duration = Duration::from_secs(30);

/// Wait while the benchmark that started at `ours` runs (checked every `poll`). `cancel` cancels
/// it, then waits for it to stop, up to `stop_wait` (logged every `log_every`), before answering
/// [`Cancelled`]: the queue's next job must not start beside a benchmark that is still compiling.
fn wait_own(
    svc: &BenchmarkService,
    ours: Option<Instant>,
    cancel: &AtomicBool,
    stop_wait: Duration,
    poll: Duration,
    log_every: Duration,
) -> Result<()> {
    let running = || {
        svc.running()
            .is_some_and(|r| ours.is_some() && r.started == ours)
    };
    while running() {
        if cancel.load(Ordering::Relaxed) {
            svc.cancel();
            let started = Instant::now();
            let mut logged = Instant::now();
            while running() {
                if started.elapsed() >= stop_wait {
                    tracing::warn!(
                        "the cancelled benchmark has not stopped after {} s (a model compile \
                         cannot be interrupted); the run ends anyway",
                        stop_wait.as_secs()
                    );
                    break;
                }
                if logged.elapsed() >= log_every {
                    tracing::info!(
                        "waiting for the cancelled benchmark to stop ({} s; a model compile \
                         cannot be interrupted)",
                        started.elapsed().as_secs()
                    );
                    logged = Instant::now();
                }
                std::thread::sleep(poll);
            }
            return Err(anyhow!(Cancelled));
        }
        std::thread::sleep(poll);
    }
    Ok(())
}

/// The datasets of the run: the held-out set, then `extra` built-in ids.
pub fn datasets(name: &str, heldout: Option<&Path>, extra: &[String]) -> Vec<DatasetRef> {
    let mut v = Vec::new();
    if let Some(h) = heldout {
        v.push(DatasetRef::Dir {
            dir: h.to_path_buf(),
            gt: None,
            name: Some(format!("heldout-{name}")),
        });
    }
    v.extend(extra.iter().map(|e| DatasetRef::Id(e.clone())));
    v
}

/// The devices of the run: `chosen` when given, else the one device the model `name` would load
/// on (its configured device, `auto` resolved to its first pick on this machine).
fn devices_for(c: &mut RunContext, name: &str, chosen: &[String]) -> Result<Vec<String>> {
    if !chosen.is_empty() {
        return Ok(chosen.to_vec());
    }
    let m = crate::benchmark::find_model(&c.config, name)
        .ok_or_else(|| anyhow!("'{name}' is not in the config"))?;
    if let DeviceSpec::Device(d) = c.config.device_spec_for(m)? {
        return Ok(vec![d.to_string()]);
    }
    let path = c.config.data_path(&m.path);
    // The service reuses these runtimes (the CLI has none until now).
    let rt = crate::benchmark::service::run_runtimes(c)?;
    c.runtimes = Some(rt.clone());
    let pick = rt
        .lock_recover()
        .selection(Some(&path))
        .auto_pick()
        .map(|o| o.spec.to_string());
    pick.map(|d| vec![d])
        .ok_or_else(|| anyhow!("no device can run {name} on this machine"))
}

/// Benchmark `name` on `sets` on `devices` one after the other (see the module docs). `ctx`
/// builds the run context when the service is free (the config must already list the model).
/// Waits for a running benchmark up to [`BUSY_WAIT`]; `cancel` stops the wait and the run.
pub fn run(
    svc: &Arc<BenchmarkService>,
    ctx: &dyn Fn() -> RunContext,
    name: &str,
    sets: Vec<DatasetRef>,
    devices: &[String],
    cancel: &AtomicBool,
) -> Result<Option<BenchSummary>> {
    if sets.is_empty() {
        return Ok(None);
    }
    let mut c = ctx();
    let devices = devices_for(&mut c, name, devices)?;
    let mut req = RunRequest::defaults(&c.config.benchmark);
    req.models = vec![name.to_string()];
    let fingerprint = sets.iter().find_map(|s| match s {
        DatasetRef::Dir { dir, .. } => dataset::read_heldout_fingerprint(dir),
        _ => None,
    });
    req.datasets = sets;
    req.devices = devices;
    req.reference_model = None;
    // The held-out scores of a trained model, not its latency under load: no burst test (the
    // Benchmark page runs one when asked).
    req.burst.enabled = false;
    req.validate()?;
    let started = Instant::now();
    let mut pending = Some(c);
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!(Cancelled));
        }
        let c = match pending.take() {
            Some(c) => c,
            None => ctx(),
        };
        match svc.start(c, req.clone()) {
            Ok(()) => break,
            Err(_) if started.elapsed() < BUSY_WAIT => {
                std::thread::sleep(Duration::from_secs(2));
            }
            Err(_) => bail!(
                "another benchmark kept running for {} min; benchmark the model from the \
                 Benchmark page",
                BUSY_WAIT.as_secs() / 60
            ),
        }
    }
    // This run's benchmark, told from any later one by its start.
    let ours = svc.running().and_then(|r| r.started);
    wait_own(
        svc,
        ours,
        cancel,
        STOP_WAIT,
        Duration::from_millis(500),
        STOP_LOG_EVERY,
    )?;
    let st = svc.status();
    if st.state == RunState::Failed {
        bail!(
            "the benchmark failed: {}",
            st.error.unwrap_or_else(|| "unknown error".into())
        );
    }
    let m = st
        .models
        .iter()
        .rfind(|m| m.model == name)
        .ok_or_else(|| anyhow!("the benchmark finished without a result for {name}"))?;
    if let Some(e) = &m.error {
        bail!("benchmarking {name} failed: {e}");
    }
    Ok(Some(BenchSummary {
        heldout_fingerprint: fingerprint,
        ..summarize(m)
    }))
}

/// Why the current default cannot be compared on a run's frames; None = it can. `default` is the
/// current default's name, `loaded` whether it is loaded, `default_classes` the classes it
/// reports (None = could not be read: the run itself will tell).
pub fn baseline_skip(
    default: Option<&str>,
    new: &str,
    new_classes: &[String],
    default_classes: Option<&[String]>,
    loaded: bool,
) -> Option<String> {
    use super::autolabel::catalog::same_class;
    let Some(default) = default else {
        return Some("no model is enabled as the default".to_string());
    };
    if crate::registry::normalize_name(default) == crate::registry::normalize_name(new) {
        return Some(format!(
            "{new} is the default model itself, so there is nothing to compare it with"
        ));
    }
    if !loaded {
        return Some(format!(
            "{default} is not loaded, so it could not run; once it is, benchmark it on the \
             Benchmark page"
        ));
    }
    if let Some(classes) = default_classes.filter(|c| !c.is_empty())
        && !new_classes.is_empty()
        && !classes
            .iter()
            .any(|c| new_classes.iter().any(|n| same_class(c, n)))
    {
        return Some(format!(
            "{default} reports none of the classes of {new} ({}), so the same frames say nothing \
             about it",
            new_classes.join(", ")
        ));
    }
    None
}

/// Format of [`baseline_key`]: a change of what it covers bumps it, so older keys never match.
pub const BASELINE_KEY_FORMAT: u32 = 1;

/// One dataset of a baseline as [`baseline_key`] sees it: what its result depends on, never its
/// display name (`heldout-<new model>` differs from run to run).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SetIdentity {
    /// A built-in set: pinned with its hashes in this build's catalog.
    Builtin(String),
    /// A folder (the held-out frames): SHA-256 of its manifest and class file, its held-out
    /// fingerprint, the size of every image and the identity of a ground-truth file.
    Dir {
        manifest_sha256: Option<String>,
        classes_sha256: Option<String>,
        heldout_fingerprint: Option<String>,
        images: Vec<(String, u64)>,
        gt: Option<FileId>,
    },
}

/// Everything a baseline measurement depends on: equal inputs measure the same thing, so a result
/// of equal inputs is reused instead of benchmarking again ([`reusable_baseline`]).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct BaselineInputs {
    /// The default model's name (normalized) and its config entry as configured.
    pub model: String,
    pub entry: serde_json::Value,
    /// The model file and its class file, with their identity.
    pub files: Vec<(PathBuf, FileId)>,
    /// The config without its model list (registering a trained model only adds to that list;
    /// any other change, a threshold or a benchmark setting, makes another key).
    pub config: serde_json::Value,
    pub sets: Vec<SetIdentity>,
    /// The devices it runs on, resolved ([`resolve_devices`]).
    pub devices: Vec<String>,
    /// This build ([`crate::long_version`]): another build may score differently.
    pub build: String,
}

/// Hex SHA-256 of `inputs` (with [`BASELINE_KEY_FORMAT`]). Pure.
pub fn baseline_key(inputs: &BaselineInputs) -> String {
    let bytes = serde_json::to_vec(&(BASELINE_KEY_FORMAT, inputs)).unwrap_or_default();
    crate::resources::manager::sha256_hex(&bytes)
}

/// The identity of the dataset `set` (see [`SetIdentity`]); None when a folder's files cannot be
/// read (then nothing is reused).
pub fn set_identity(set: &DatasetRef) -> Option<SetIdentity> {
    let sha = |p: &Path| {
        std::fs::read(p)
            .ok()
            .map(|b| crate::resources::manager::sha256_hex(&b))
    };
    match set {
        DatasetRef::Id(id) => Some(SetIdentity::Builtin(id.clone())),
        DatasetRef::Dir { dir, gt, .. } => {
            let mut images = Vec::new();
            for e in std::fs::read_dir(dir.join("images")).ok()? {
                let p = e.ok()?.path();
                // Through the file, never the directory entry (stale for a hard link on NTFS).
                let len = std::fs::metadata(&p).ok()?.len();
                images.push((p.file_name()?.to_string_lossy().into_owned(), len));
            }
            images.sort();
            let gt = match gt {
                Some(g) => Some(FileId::of(g)?),
                None => None,
            };
            Some(SetIdentity::Dir {
                manifest_sha256: sha(&dir.join("manifest.json")),
                classes_sha256: sha(&dir.join("classes.txt")),
                heldout_fingerprint: dataset::read_heldout_fingerprint(dir),
                images,
                gt,
            })
        }
    }
}

/// The inputs of a baseline of the default model `default` on `sets` and the resolved `devices`
/// under `config`; None when anything cannot be read (the model is not configured, its file has
/// no identity, a set's files do not read): then it is measured as before.
pub fn baseline_inputs(
    config: &crate::config::Config,
    default: &str,
    sets: &[DatasetRef],
    devices: Vec<String>,
) -> Option<BaselineInputs> {
    let m = crate::benchmark::find_model(config, default)?;
    let mut files = vec![{
        let p = config.data_path(&m.path);
        let id = FileId::of(&p)?;
        (p, id)
    }];
    if let Some(c) = &m.classes {
        let p = config.data_path(c);
        let id = FileId::of(&p)?;
        files.push((p, id));
    }
    let mut cfg = serde_json::to_value(config).ok()?;
    cfg.as_object_mut()?.remove("models");
    Some(BaselineInputs {
        model: crate::registry::normalize_name(default),
        entry: serde_json::to_value(m).ok()?,
        files,
        config: cfg,
        sets: sets.iter().map(set_identity).collect::<Option<_>>()?,
        devices,
        build: crate::long_version(),
    })
}

/// The devices a benchmark of `name` runs on: `chosen`, else the one device it would load on (as
/// [`run`] resolves them).
pub fn resolve_devices(c: &mut RunContext, name: &str, chosen: &[String]) -> Result<Vec<String>> {
    devices_for(c, name, chosen)
}

/// A baseline measured for another run of the comparison group of the run in `run_dir`, with the
/// same `key` and a result (never a note, a failure or a result without a device), marked
/// [`Baseline::reused_from`] that run (or the run it was reused from in turn). The first such run
/// in queue order. None: measure it. Pure over `records`.
pub fn reusable_baseline(
    records: &[super::service::RunRecord],
    run_dir: &Path,
    key: &str,
) -> Option<Baseline> {
    let me = records.iter().find(|r| r.run_dir == run_dir)?;
    let group = me.request.compare_group.as_deref()?;
    records
        .iter()
        .filter(|r| r.id != me.id && r.request.compare_group.as_deref() == Some(group))
        .find_map(|r| {
            let b = r.baseline.as_ref()?;
            let usable = b.key.as_deref() == Some(key)
                && b.note.is_none()
                && b.summary.as_ref().is_some_and(|s| s.device.is_some());
            usable.then(|| Baseline {
                reused_from: Some(b.reused_from.clone().unwrap_or_else(|| r.id.clone())),
                ..b.clone()
            })
        })
}

/// The chip of a benchmark that includes the model's own training frames; None without any.
pub fn trained_chip(share: f64) -> Option<String> {
    (share.is_finite() && share > 0.0).then(|| {
        let p = (share * 100.0).round().max(1.0);
        format!("{p:.0}% of these frames are its own training frames")
    })
}

/// `finds 91% of people, 84% of vehicles, false alerts 3 on 120 empty frames, person alerts 2 on
/// 210 frames without a person (threshold 0.50)`.
fn finds_text(b: &BenchSummary) -> String {
    let pct = |v: Option<f64>| {
        v.filter(|v| v.is_finite())
            .map(|v| format!("{:.0}%", v * 100.0))
    };
    let mut finds = Vec::new();
    if let Some(p) = pct(b.person_recall) {
        finds.push(format!("{p} of people"));
    }
    if let Some(p) = pct(b.vehicle_recall) {
        finds.push(format!("{p} of vehicles"));
    }
    let mut s = if finds.is_empty() {
        "finds n/a".to_string()
    } else {
        format!("finds {}", finds.join(", "))
    };
    match (b.empty_frame_alerts, b.empty_frames.filter(|e| *e > 0)) {
        (Some(a), Some(e)) => s.push_str(&format!(", false alerts {a} on {e} empty frames")),
        _ => s.push_str(", false alerts n/a (no empty frames)"),
    }
    // Frames with a parked car are not empty: false person alerts show here.
    if let (Some(a), Some(f)) = (
        b.person_false_alerts,
        b.person_free_frames.filter(|f| *f > 0),
    ) {
        s.push_str(&format!(
            ", person alerts {a} on {f} frames without a person"
        ));
    }
    if let Some(t) = b.threshold {
        s.push_str(&format!(" (threshold {t:.2})"));
    }
    s
}

/// The line of the run row and the Use card: the new model and the current default on the same
/// frames, or why they were not compared. Empty when no comparison was tried.
pub fn compare_text(new: Option<&BenchSummary>, baseline: Option<&Baseline>) -> String {
    let (Some(new), Some(b)) = (new, baseline) else {
        return String::new();
    };
    let Some(cur) = &b.summary else {
        let why = b.note.as_deref().unwrap_or("no result");
        return format!("Not compared with the current default model: {why}.");
    };
    let frames = if new.datasets.len() > 1 {
        format!("{} frames ({})", new.images, new.datasets.join(" + "))
    } else {
        format!("{} held-out frames", new.images)
    };
    let mut s = format!(
        "On {frames}: new {}; current ({}) {}.",
        finds_text(new),
        b.model,
        finds_text(cur)
    );
    if let Some(c) = cur.trained_on_share.and_then(trained_chip) {
        s.push_str(&format!(" For {}, {c}, so it has an unfair lead.", b.model));
    }
    s
}

/// The Train tab's view of a benchmark result: the recommended device (else the first that
/// ran), its p50 time and AP50 (over every class of the model when the set annotates others),
/// and the alerting figures of its overall score (person / vehicle recall, false alerts on empty
/// frames, false person alerts on frames without a person) at the benchmark's configured
/// threshold. Like the overall AP, those cover every dataset of the run (the held-out set and any
/// built-in set run beside it). Each AP carries the classes it averages (`ap_basis`).
pub fn summarize(m: &ModelResult) -> BenchSummary {
    let dev = m
        .devices
        .iter()
        .find(|d| d.ok() && m.recommended.as_deref() == Some(d.device.as_str()))
        .or_else(|| m.devices.iter().find(|d| d.ok()));
    let acc = dev
        .and_then(|d| d.run.as_ref())
        .and_then(|r| r.accuracy.as_ref());
    let (ap50, ap50_95, ap_basis) = match acc {
        Some(a) => match &a.all_classes {
            Some(all) if all.ap50.is_some() => (all.ap50, all.ap50_95, all.ap_basis.clone()),
            _ => (
                a.overall.ap50,
                a.overall.ap50_95,
                a.overall.ap_basis.clone(),
            ),
        },
        None => (None, None, None),
    };
    let tags: &[Breakdown] = acc.map(|a| a.by_tag.as_slice()).unwrap_or_default();
    let ap = |b: &Breakdown| SceneAp {
        ap50: b.ap50,
        ap50_95: b.ap50_95,
        images: b.images,
        ap_basis: b.ap_basis.clone(),
    };
    let tagged = |key: &str| tags.iter().find(|b| b.key == key).map(ap);
    let mut by_scene = std::collections::BTreeMap::new();
    let mut in_scope = None;
    let mut recall_cells = RecallCells::default();
    if tagged(dataset::HELDOUT_ALL).is_some() {
        for s in ["day", "night", "ir", "all"] {
            if let Some(a) = tagged(&format!("{}{s}", dataset::HELDOUT_TAG)) {
                by_scene.insert(s.to_string(), a);
            }
        }
        in_scope = tagged(dataset::HELDOUT_IN_SCOPE);
        recall_cells = cells_of(
            tags.iter()
                .filter(|b| !b.relative)
                .map(|b| (b.key.as_str(), b.per_class.as_slice())),
        );
    } else if acc.is_none()
        && let Some(r) = dev.and_then(|d| d.run.as_ref())
        && r.tag_recall.iter().any(|t| t.key == dataset::HELDOUT_ALL)
    {
        // No accuracy (a model without any CCTV class): its held-out tags' recall only.
        recall_cells = cells_of(
            r.tag_recall
                .iter()
                .map(|t| (t.key.as_str(), t.per_class.as_slice())),
        );
    } else if m.datasets.len() == 1
        && let Some(a) = acc
    {
        // An older held-out set alone: its bare scene tags are its own.
        for s in ["day", "night", "ir"] {
            if let Some(b) = tagged(s) {
                by_scene.insert(s.to_string(), b);
            }
        }
        by_scene.insert(
            "all".into(),
            SceneAp {
                ap50: a.overall.ap50,
                ap50_95: a.overall.ap50_95,
                images: a.overall.images,
                ap_basis: a.overall.ap_basis.clone(),
            },
        );
    }
    // The Review queue's random picks among the held-out frames: their own breakdown, so the
    // same AP as the scene columns, and the recall over every scored class at the threshold.
    let random = tags
        .iter()
        .find(|b| b.key == dataset::HELDOUT_RANDOM)
        .filter(|b| b.images > 0)
        .map(|b| SubsetScore {
            images: b.images,
            gt: b.gt,
            ap50: b.ap50,
            ap50_95: b.ap50_95,
            recall: b.recall,
            ap_basis: b.ap_basis.clone(),
        });
    // Alerting at the configured threshold, from the overall score (scored CCTV classes).
    let o = acc.map(|a| &a.overall);
    let person = o.and_then(|o| o.class_alert("person"));
    BenchSummary {
        datasets: m.datasets.clone(),
        device: dev.map(|d| d.device.clone()),
        ms_p50: dev.and_then(|d| d.total_p50()),
        ap50,
        ap50_95,
        images: m.images,
        threshold: acc.map(|a| a.threshold),
        person_recall: o.and_then(|o| o.person_recall),
        vehicle_recall: o.and_then(|o| o.vehicle_recall),
        empty_frames: o.and_then(|o| o.empty_frames),
        empty_frame_alerts: o.and_then(|o| o.empty_frame_alerts),
        by_scene,
        in_scope,
        heldout_fingerprint: None,
        trained_on_share: m.trained_on_share,
        random,
        ap_basis,
        person_free_frames: person.map(|p| p.frames_without),
        person_false_alerts: person.map(|p| p.alerts),
        recall_cells,
    }
}

/// The recall cells of a run ([`RecallCells`]) from the per-class recall of the held-out
/// manifest's own tags, `(tag, cells)`: `heldout:<scene>` (and `heldout:all`),
/// `heldout:cam:<camera>`, `heldout:camscene:<scene>:<camera>` ([`dataset::cell_tag`]). Never
/// from bare tags: a built-in set run beside the held-out frames has its own `night` or `camera:`
/// images. The caller leaves out breakdowns against pseudo ground truth.
fn cells_of<'a>(tags: impl IntoIterator<Item = (&'a str, &'a [ClassRecall])>) -> RecallCells {
    let mut c = RecallCells::default();
    for (key, cells) in tags.into_iter().filter(|(_, c)| !c.is_empty()) {
        let classes = cells.to_vec();
        if let Some(cell) = dataset::cell_tag(key) {
            match cell {
                dataset::CellTag::Camera(camera) => {
                    c.by_camera.insert(camera.to_string(), classes);
                }
                dataset::CellTag::CameraScene { scene, camera } => {
                    c.by_camera_scene.push(CameraSceneRecall {
                        camera: camera.to_string(),
                        scene: scene.to_string(),
                        classes,
                    });
                }
            }
        } else if let Some(scene) = key.strip_prefix(dataset::HELDOUT_TAG)
            && matches!(scene, "day" | "night" | "ir" | "all")
        {
            c.by_scene.insert(scene.to_string(), classes);
        }
    }
    c
}

/// The measured recall the Review queue steers by, from the training runs `records` (pure; the
/// rule of [`Measured`]):
///
/// - Only finished runs (`done`) with their model's output (its classes) whose benchmark has a
///   held-out fingerprint (the held-out frames were chosen from the review folder, before the
///   scene filter) and recall cells count, newest first (by `finished`, else `created`; equal
///   stamps: the later record).
/// - Per scene (`day` / `night` / `ir`): the newest such run whose scene filter keeps the scene
///   and that has cells for it, with its camera-in-scene cells. A night model also has day
///   frames among its held-out frames, but it never trained on them: its day recall says
///   nothing about what the day model misses.
/// - Over every scene (per class and per camera): the newest run whose filter keeps every scene;
///   none without one (a scene-filtered model never trained on some of those frames).
/// - Each slice carries its run's model classes and renames ([`recall::Run`]).
///
/// None when no run measured anything.
pub fn measured_recall(records: &[super::service::RunRecord]) -> Option<Measured> {
    use super::service::RunState;
    let mut runs: Vec<(usize, &super::service::RunRecord, &RecallCells)> = records
        .iter()
        .enumerate()
        .filter(|(_, r)| r.state == RunState::Done)
        .filter_map(|(i, r)| {
            let b = r.benchmark.as_ref()?;
            b.heldout_fingerprint.as_ref()?;
            r.output.as_ref()?;
            (!b.recall_cells.is_empty()).then_some((i, r, &b.recall_cells))
        })
        .collect();
    // What the run's model can detect: its classes (after the renames) and the renames.
    let run_of = |r: &super::service::RunRecord| recall::Run {
        id: r.id.clone(),
        classes: r
            .output
            .as_ref()
            .map(|o| o.classes.clone())
            .unwrap_or_default(),
        rename: r.request.rename.clone(),
    };
    // RFC 3339 UTC stamps (`service::now`): their text order is their time order.
    fn stamp(r: &super::service::RunRecord) -> &str {
        r.finished.as_deref().unwrap_or(&r.created)
    }
    runs.sort_by(|a, b| stamp(b.1).cmp(stamp(a.1)).then(b.0.cmp(&a.0)));
    let has = |c: &RecallCells, s: &str| c.by_scene.get(s).is_some_and(|v| !v.is_empty());
    let mut m = Measured::default();
    for s in recall::SCENES {
        let Some((_, r, c)) = runs
            .iter()
            .find(|(_, r, c)| r.request.scene.keeps(Some(s)) && has(c, s))
        else {
            continue;
        };
        let cams = c
            .by_camera_scene
            .iter()
            .filter(|x| x.scene == s)
            .map(|x| (x.camera.as_str(), x.classes.as_slice()));
        let classes = c.by_scene.get(s).map(Vec::as_slice).unwrap_or_default();
        m.scenes
            .insert(s.to_string(), recall::Slice::new(run_of(r), classes, cams));
    }
    let overall = runs
        .iter()
        .find(|(_, r, c)| r.request.scene.is_all() && has(c, "all"));
    if let Some((_, r, c)) = overall {
        let classes = c.by_scene.get("all").map(Vec::as_slice).unwrap_or_default();
        let cams = c.by_camera.iter().map(|(k, v)| (k.as_str(), v.as_slice()));
        m.overall = Some(recall::Slice::new(run_of(r), classes, cams));
    }
    (!m.is_empty()).then_some(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with(device: Option<&str>) -> RunContext {
        let mut config = crate::config::Config::default();
        config.models = vec![crate::config::ModelConfig {
            name: Some("custom-1".into()),
            path: "models/custom-1.onnx".into(),
            device: device.map(str::to_string),
            ..Default::default()
        }];
        RunContext {
            config,
            results_path: "benchmark.json".into(),
            runtimes: None,
            loaded: Default::default(),
            resources: None,
            profile: None,
            metrics: None,
        }
    }

    /// A fake benchmark that ignores the cancel for a while (a model compile), then stops.
    fn compiling(svc: &Arc<BenchmarkService>, busy: Duration) -> Option<Instant> {
        let req = RunRequest::defaults(&crate::config::Config::default().benchmark);
        svc.start_job(req, move |h| {
            let t = Instant::now();
            while t.elapsed() < busy || !h.is_cancelled() {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(anyhow!(Cancelled))
        })
        .unwrap();
        svc.running().and_then(|r| r.started)
    }

    #[test]
    fn a_cancelled_benchmark_is_waited_for_until_it_stopped() {
        let svc = BenchmarkService::new();
        let ours = compiling(&svc, Duration::from_millis(300));
        assert!(ours.is_some());
        let cancel = AtomicBool::new(true);
        let ms = Duration::from_millis;
        let t = Instant::now();
        let e = wait_own(&svc, ours, &cancel, ms(10_000), ms(5), ms(50)).unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some());
        assert!(t.elapsed() >= ms(250), "returned before the compile ended");
        assert!(
            !svc.is_running(),
            "the benchmark had stopped when the run went on"
        );
    }

    #[test]
    fn the_wait_for_a_cancelled_benchmark_is_bounded() {
        let svc = BenchmarkService::new();
        let ours = compiling(&svc, Duration::from_secs(5));
        let cancel = AtomicBool::new(true);
        let ms = Duration::from_millis;
        let t = Instant::now();
        let e = wait_own(&svc, ours, &cancel, ms(100), ms(5), ms(20)).unwrap_err();
        assert!(e.downcast_ref::<Cancelled>().is_some());
        assert!(t.elapsed() < Duration::from_secs(3));
        assert!(svc.is_running(), "past the bound the run ends anyway");
        svc.wait_idle(Duration::from_secs(10));
    }

    #[test]
    fn only_its_own_benchmark_is_waited_for() {
        let svc = BenchmarkService::new();
        let ms = Duration::from_millis;
        // Not ours (another start), or ours already over: no wait, no cancel.
        let other = compiling(&svc, ms(0));
        let cancel = AtomicBool::new(false);
        let t = Instant::now();
        wait_own(&svc, Some(Instant::now()), &cancel, ms(100), ms(5), ms(20)).unwrap();
        wait_own(&svc, None, &cancel, ms(100), ms(5), ms(20)).unwrap();
        assert!(t.elapsed() < Duration::from_secs(1));
        assert!(svc.is_running(), "another run's benchmark is left alone");
        assert!(other.is_some());
        svc.cancel();
        assert!(svc.wait_idle(Duration::from_secs(10)));
        // Ours, not cancelled: waited for until it ends.
        let ours = compiling(&svc, ms(0));
        let stopper = {
            let svc = svc.clone();
            std::thread::spawn(move || {
                std::thread::sleep(ms(100));
                svc.cancel();
            })
        };
        wait_own(&svc, ours, &cancel, ms(100), ms(5), ms(20)).unwrap();
        assert!(!svc.is_running());
        stopper.join().unwrap();
    }

    #[test]
    fn devices_are_the_chosen_ones_else_the_models_own() {
        // Chosen devices win, in order.
        let mut c = ctx_with(Some("openvino:cpu"));
        let chosen = ["ort:cuda:0".to_string(), "openvino:gpu.0".to_string()];
        assert_eq!(devices_for(&mut c, "custom-1", &chosen).unwrap(), chosen);
        // None chosen: the model's own device, one only, without starting any runtime.
        assert_eq!(
            devices_for(&mut c, "custom-1", &[]).unwrap(),
            ["openvino:cpu"]
        );
        assert!(c.runtimes.is_none());
        assert!(devices_for(&mut c, "other", &[]).is_err());
    }

    #[test]
    fn baseline_skip_reasons() {
        let new = vec!["person".to_string(), "car".to_string()];
        let skip = |d, c, l| baseline_skip(d, "new-model", &new, c, l);
        let coco = ["person".to_string(), "dog".to_string()];
        assert_eq!(skip(Some("IPcam"), Some(&coco[..]), true), None);
        // An unreadable class list does not stop it: the run will tell.
        assert_eq!(skip(Some("IPcam"), None, true), None);
        assert_eq!(skip(Some("IPcam"), Some(&[]), true), None);
        assert!(skip(None, None, true).unwrap().contains("no model"));
        let e = skip(Some("New-Model.onnx"), None, true).unwrap();
        assert!(e.contains("default model itself"), "{e}");
        let e = skip(Some("IPcam"), Some(&coco[..]), false).unwrap();
        assert!(
            e.contains("IPcam is not loaded") && e.contains("Benchmark page"),
            "{e}"
        );
        let dogs = ["dog".to_string()];
        let e = skip(Some("IPcam"), Some(&dogs[..]), true).unwrap();
        assert!(
            e.contains("none of the classes") && e.contains("person, car"),
            "{e}"
        );
        // Same class under two spellings: shared.
        let cycles = ["motorbike".to_string()];
        let m = ["motorcycle".to_string()];
        assert_eq!(
            baseline_skip(Some("a"), "b", &m, Some(&cycles[..]), true),
            None
        );
    }

    fn summary(person: f64, alerts: usize) -> BenchSummary {
        BenchSummary {
            datasets: vec!["heldout-x".into()],
            images: 251,
            threshold: Some(0.5),
            person_recall: Some(person),
            vehicle_recall: Some(0.8),
            empty_frames: Some(120),
            empty_frame_alerts: Some(alerts),
            ..BenchSummary::default()
        }
    }

    #[test]
    fn compare_text_sets_both_models_side_by_side() {
        let new = summary(0.91, 3);
        let base = Baseline {
            model: "IPcam".into(),
            summary: Some(summary(0.72, 9)),
            note: None,
            ..Baseline::default()
        };
        let t = compare_text(Some(&new), Some(&base));
        assert_eq!(
            t,
            "On 251 held-out frames: new finds 91% of people, 80% of vehicles, false alerts 3 on \
             120 empty frames (threshold 0.50); current (IPcam) finds 72% of people, 80% of \
             vehicles, false alerts 9 on 120 empty frames (threshold 0.50)."
        );
        // A default that trained on these frames has a lead the reader must see.
        let mut unfair = base.clone();
        unfair.summary.as_mut().unwrap().trained_on_share = Some(0.25);
        assert!(compare_text(Some(&new), Some(&unfair)).contains("25% of these frames"));
        // Skipped, or never tried.
        let skipped = Baseline {
            model: "IPcam".into(),
            summary: None,
            note: Some("IPcam is not loaded".into()),
            ..Baseline::default()
        };
        assert_eq!(
            compare_text(Some(&new), Some(&skipped)),
            "Not compared with the current default model: IPcam is not loaded."
        );
        assert_eq!(compare_text(Some(&new), None), "");
        assert_eq!(compare_text(None, Some(&base)), "");
        // With a built-in set beside the held-out frames the words say so.
        let mut more = new.clone();
        more.datasets.push("nir-night".into());
        assert!(compare_text(Some(&more), Some(&base)).contains("frames (heldout-x + nir-night)"));
    }

    #[test]
    fn the_chip_only_shows_a_share_above_zero() {
        assert_eq!(trained_chip(0.0), None);
        assert_eq!(trained_chip(f64::NAN), None);
        assert_eq!(
            trained_chip(0.625).as_deref(),
            Some("63% of these frames are its own training frames")
        );
        assert!(
            trained_chip(0.001).unwrap().starts_with("1%"),
            "never rounds away"
        );
        let mut m = result(vec![], false);
        m.trained_on_share = Some(0.5);
        assert_eq!(summarize(&m).trained_on_share, Some(0.5));
    }

    #[test]
    fn datasets_put_the_heldout_set_first() {
        let d = datasets(
            "night-yolo26n-x",
            Some(Path::new("/r/data/heldout")),
            &["nir-night".into()],
        );
        assert_eq!(d.len(), 2);
        assert_eq!(
            d[0],
            DatasetRef::Dir {
                dir: "/r/data/heldout".into(),
                gt: None,
                name: Some("heldout-night-yolo26n-x".into())
            }
        );
        assert_eq!(d[1], DatasetRef::Id("nir-night".into()));
        assert!(datasets("x", None, &[]).is_empty());
    }

    fn breakdown(key: &str, images: usize, ap: f64) -> Breakdown {
        Breakdown {
            key: key.into(),
            images,
            gt: images,
            precision: None,
            recall: None,
            f1: None,
            ap50: Some(ap + 0.2),
            ap50_95: Some(ap),
            relative: false,
            roc_auc: None,
            ap_basis: None,
            per_class: vec![],
        }
    }

    fn cr(class: &str, gt: usize, missed: usize) -> ClassRecall {
        ClassRecall::new(class, gt, missed)
    }

    /// [`breakdown`] with per-class recall.
    fn cells(key: &str, per_class: Vec<ClassRecall>) -> Breakdown {
        Breakdown {
            per_class,
            ..breakdown(key, 10, 0.5)
        }
    }

    /// A trained model's result on its held-out set (+ `extra` datasets) with `tags`.
    fn result(tags: Vec<Breakdown>, extra: bool) -> ModelResult {
        use crate::benchmark::report::test_support::{dev, model, scene};
        let mut d = dev("openvino:cpu", 12.0, scene());
        let overall = crate::benchmark::metrics::Summary {
            images: 30,
            ap50: Some(0.7),
            ap50_95: Some(0.4),
            person_recall: Some(0.39),
            vehicle_recall: Some(0.67),
            empty_frames: Some(130),
            empty_frame_alerts: Some(4),
            ..Default::default()
        };
        d.run.as_mut().unwrap().accuracy = Some(crate::benchmark::AccuracyReport {
            ground_truth: "ground truth".into(),
            relative: false,
            classes: vec!["person".into()],
            threshold: 0.5,
            overall,
            by_dataset: vec![],
            by_tag: tags,
            all_classes: None,
            sweep: None,
            frame_roc: None,
        });
        let mut m = model(vec![d], Some("openvino:cpu"));
        m.datasets = vec!["heldout-x".into()];
        if extra {
            m.datasets.push("nir-night".into());
        }
        m
    }

    /// False person alerts and the AP's classes reach the summary and the Use card's words.
    #[test]
    fn person_alerts_and_the_ap_basis_reach_the_summary_and_the_text() {
        use crate::benchmark::metrics::{ApBasis, ClassAlerts};
        let basis = ApBasis {
            min_gt: 20,
            classes: vec!["car".into(), "person".into()],
            fell_back: false,
        };
        let mut m = result(vec![], false);
        let acc = m.devices[0]
            .run
            .as_mut()
            .unwrap()
            .accuracy
            .as_mut()
            .unwrap();
        acc.overall.ap_basis = Some(basis.clone());
        acc.overall.class_alerts = vec![
            ClassAlerts {
                class: "person".into(),
                frames_without: 210,
                alerts: 3,
            },
            ClassAlerts {
                class: "car".into(),
                frames_without: 5,
                alerts: 1,
            },
        ];
        let b = summarize(&m);
        assert_eq!(b.ap_basis.as_ref(), Some(&basis));
        assert_eq!(
            (b.person_free_frames, b.person_false_alerts),
            (Some(210), Some(3))
        );
        let j = serde_json::to_value(&b).unwrap();
        assert_eq!(j["person_false_alerts"], 3);
        assert_eq!(j["ap_basis"]["classes"][1], "person");
        // The all-classes AP brings its own basis.
        let all_basis = ApBasis {
            classes: vec!["car".into()],
            ..basis.clone()
        };
        m.devices[0]
            .run
            .as_mut()
            .unwrap()
            .accuracy
            .as_mut()
            .unwrap()
            .all_classes = Some(crate::benchmark::AllClasses {
            classes: 3,
            ap50: Some(0.5),
            ap50_95: Some(0.3),
            ap_basis: Some(all_basis.clone()),
        });
        assert_eq!(summarize(&m).ap_basis, Some(all_basis));
        // The words: after the empty-frame alerts, before the threshold.
        let mut s = summary(0.91, 3);
        s.person_free_frames = Some(210);
        s.person_false_alerts = Some(2);
        assert_eq!(
            finds_text(&s),
            "finds 91% of people, 80% of vehicles, false alerts 3 on 120 empty frames, person \
             alerts 2 on 210 frames without a person (threshold 0.50)"
        );
        // No frame without a person: nothing said.
        s.person_free_frames = Some(0);
        assert!(!finds_text(&s).contains("person alerts"));
        // A record of an older version loads without them.
        let old: BenchSummary = serde_json::from_str(
            r#"{"datasets":[],"device":null,"ms_p50":null,"ap50":0.5,"ap50_95":null,"images":3,
               "by_scene":{"day":{"ap50":0.5,"ap50_95":0.3,"images":3}},
               "random":{"images":1,"gt":1,"ap50":null,"ap50_95":null,"recall":null}}"#,
        )
        .unwrap();
        assert_eq!((old.ap_basis, old.person_free_frames), (None, None));
        assert_eq!(old.by_scene["day"].ap_basis, None);
        assert_eq!(old.random.unwrap().ap_basis, None);
    }

    #[test]
    fn summarize_reads_ap_per_scene_from_the_heldout_tags() {
        // Bare tags mix in a built-in set run beside; the `heldout:` ones do not.
        let tags = vec![
            breakdown("day", 50, 0.9),
            breakdown("night", 210, 0.1),
            breakdown("heldout:all", 30, 0.45),
            breakdown("heldout:day", 10, 0.6),
            breakdown("heldout:night", 12, 0.3),
            breakdown("heldout:ir", 8, 0.35),
            breakdown("heldout:in-scope", 20, 0.32),
            breakdown("res:sd", 30, 0.45),
        ];
        let b = summarize(&result(tags, true));
        assert_eq!(b.by_scene.len(), 4);
        let day = &b.by_scene["day"];
        assert_eq!(
            (day.ap50_95, day.ap50, day.images),
            (Some(0.6), Some(0.8), 10)
        );
        assert_eq!(b.by_scene["night"].ap50_95, Some(0.3));
        assert_eq!(b.by_scene["ir"].images, 8);
        assert_eq!(b.by_scene["all"].ap50_95, Some(0.45));
        let s = b.in_scope.unwrap();
        assert_eq!((s.ap50_95, s.images), (Some(0.32), 20));
        assert_eq!(b.ms_p50, Some(12.0));
        assert_eq!(
            b.heldout_fingerprint, None,
            "set by run() from dataset.json"
        );
        // Serialized only when present (old records stay as they were).
        let j = serde_json::to_value(BenchSummary::default()).unwrap();
        assert!(j.get("by_scene").is_none() && j.get("in_scope").is_none());
        let back: BenchSummary = serde_json::from_str(
            r#"{"datasets":[],"device":null,"ms_p50":null,"ap50":null,"ap50_95":null,"images":0}"#,
        )
        .unwrap();
        assert!(back.by_scene.is_empty());
    }

    #[test]
    fn summarize_reads_the_random_picks_from_their_heldout_tag() {
        let mut random = breakdown("heldout:random", 60, 0.38);
        random.gt = 85;
        random.recall = Some(0.71);
        let tags = vec![
            breakdown("heldout:all", 300, 0.45),
            breakdown("heldout:day", 300, 0.45),
            // A bare `random` tag (the Dataset tab's benchmark set) beside it is not this one.
            breakdown("random", 90, 0.9),
            random,
        ];
        let b = summarize(&result(tags, true));
        assert_eq!(
            b.random,
            Some(SubsetScore {
                images: 60,
                gt: 85,
                ap50: Some(0.38 + 0.2),
                ap50_95: Some(0.38),
                recall: Some(0.71),
                ap_basis: None,
            })
        );
        let j = serde_json::to_value(&b).unwrap();
        assert_eq!(j["random"]["images"], 60);
        assert_eq!(j["random"]["ap50_95"], 0.38);
        let back: BenchSummary = serde_json::from_value(j).unwrap();
        assert_eq!(back.random, b.random);
        // No random pick among the held-out frames: none, and not serialized.
        let b = summarize(&result(vec![breakdown("heldout:all", 300, 0.45)], false));
        assert_eq!(b.random, None);
        assert!(serde_json::to_value(&b).unwrap().get("random").is_none());
        let old: BenchSummary = serde_json::from_str(
            r#"{"datasets":[],"device":null,"ms_p50":null,"ap50":0.5,"ap50_95":null,"images":3}"#,
        )
        .unwrap();
        assert_eq!(old.random, None);
    }

    #[test]
    fn summarize_carries_the_alerting_figures_and_threshold() {
        let b = summarize(&result(vec![], false));
        assert_eq!(b.threshold, Some(0.5));
        assert_eq!(
            (b.person_recall, b.vehicle_recall),
            (Some(0.39), Some(0.67))
        );
        assert_eq!((b.empty_frames, b.empty_frame_alerts), (Some(130), Some(4)));
        let j = serde_json::to_value(&b).unwrap();
        assert_eq!(j["empty_frame_alerts"], 4);
        assert_eq!(j["threshold"], 0.5);
        // Without accuracy (failed or unscored run): none, and not serialized.
        let mut m = result(vec![], false);
        m.devices[0].run.as_mut().unwrap().accuracy = None;
        let b = summarize(&m);
        assert_eq!(
            (b.threshold, b.person_recall, b.empty_frames),
            (None, None, None)
        );
        let j = serde_json::to_value(&b).unwrap();
        for k in [
            "threshold",
            "person_recall",
            "vehicle_recall",
            "empty_frames",
            "empty_frame_alerts",
        ] {
            assert!(j.get(k).is_none(), "{k}");
        }
        // A record of an older version loads without them.
        let old: BenchSummary = serde_json::from_str(
            r#"{"datasets":[],"device":null,"ms_p50":null,"ap50":0.5,"ap50_95":null,"images":3}"#,
        )
        .unwrap();
        assert_eq!((old.threshold, old.vehicle_recall), (None, None));
    }

    #[test]
    fn older_heldout_sets_use_the_bare_tags_only_when_alone() {
        let tags = vec![breakdown("day", 10, 0.6), breakdown("ir", 8, 0.35)];
        let b = summarize(&result(tags.clone(), false));
        assert_eq!(b.by_scene["day"].ap50_95, Some(0.6));
        assert_eq!(b.by_scene["ir"].ap50_95, Some(0.35));
        assert_eq!(b.by_scene["all"].ap50_95, Some(0.4));
        assert!(!b.by_scene.contains_key("night"));
        assert_eq!(b.in_scope, None);
        // Beside a built-in set the bare tags are not the held-out set's.
        assert!(summarize(&result(tags, true)).by_scene.is_empty());
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-bench-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A held-out set in `dir`: two images, a manifest, a class file.
    fn heldout(dir: &Path) {
        std::fs::create_dir_all(dir.join("images")).unwrap();
        std::fs::write(dir.join("images/s0-a.jpg"), b"aaaa").unwrap();
        std::fs::write(dir.join("images/s0-b.jpg"), b"bbbbbb").unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            br#"{"id":"heldout","heldout_fingerprint":"f1","images":[]}"#,
        )
        .unwrap();
        std::fs::write(dir.join("classes.txt"), "person\n").unwrap();
    }

    fn config_with(root: &Path) -> crate::config::Config {
        std::fs::create_dir_all(root.join("models")).unwrap();
        std::fs::write(root.join("models/base.onnx"), b"onnx-bytes").unwrap();
        crate::config::Config {
            models: vec![crate::config::ModelConfig {
                name: Some("base".into()),
                path: root.join("models/base.onnx"),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn key_of(c: &crate::config::Config, sets: &[DatasetRef], devices: &[&str]) -> String {
        let d = devices.iter().map(|d| d.to_string()).collect();
        baseline_key(&baseline_inputs(c, "base", sets, d).unwrap())
    }

    #[test]
    fn the_baseline_key_follows_everything_the_measurement_depends_on() {
        let root = tmp("key");
        let (h1, h2) = (root.join("run1/heldout"), root.join("run2/heldout"));
        heldout(&h1);
        heldout(&h2);
        let c = config_with(&root);
        // Two runs' copies of the same frames, under their own names: one key.
        let set = |dir: &Path, name: &str| DatasetRef::Dir {
            dir: dir.to_path_buf(),
            gt: None,
            name: Some(format!("heldout-{name}")),
        };
        let k = key_of(&c, &[set(&h1, "seed0")], &["openvino:gpu.0"]);
        assert_eq!(k, key_of(&c, &[set(&h2, "seed1")], &["openvino:gpu.0"]));
        // Registering another model (the trained one) does not change it.
        let mut more = c.clone();
        more.models.push(crate::config::ModelConfig {
            name: Some("night-yolo26n-s1".into()),
            path: root.join("models/other.onnx"),
            ..Default::default()
        });
        assert_eq!(k, key_of(&more, &[set(&h1, "a")], &["openvino:gpu.0"]));
        // Another device, set, extra set or held-out content: another key.
        assert_ne!(k, key_of(&c, &[set(&h1, "a")], &["openvino:cpu"]));
        assert_ne!(
            k,
            key_of(
                &c,
                &[set(&h1, "a"), DatasetRef::Id("nir-night".into())],
                &["openvino:gpu.0"]
            )
        );
        std::fs::write(h2.join("images/s0-b.jpg"), b"bbbbbbb").unwrap();
        assert_ne!(
            k,
            key_of(&c, &[set(&h2, "a")], &["openvino:gpu.0"]),
            "an image"
        );
        heldout(&h2);
        std::fs::write(
            h2.join("manifest.json"),
            br#"{"id":"heldout","images":[1]}"#,
        )
        .unwrap();
        assert_ne!(
            k,
            key_of(&c, &[set(&h2, "a")], &["openvino:gpu.0"]),
            "the manifest"
        );
        heldout(&h2);
        std::fs::write(h2.join("classes.txt"), "person\ncar\n").unwrap();
        assert_ne!(
            k,
            key_of(&c, &[set(&h2, "a")], &["openvino:gpu.0"]),
            "the classes"
        );
        // The default's settings, a global setting, or its file: another key.
        let mut t = c.clone();
        t.models[0].confidence_threshold = Some(0.3);
        assert_ne!(
            k,
            key_of(&t, &[set(&h1, "a")], &["openvino:gpu.0"]),
            "its threshold"
        );
        let mut g = c.clone();
        g.nms_iou += 0.05;
        assert_ne!(
            k,
            key_of(&g, &[set(&h1, "a")], &["openvino:gpu.0"]),
            "a global"
        );
        std::fs::write(root.join("models/base.onnx"), b"onnx-bytes-retrained").unwrap();
        assert_ne!(
            k,
            key_of(&c, &[set(&h1, "a")], &["openvino:gpu.0"]),
            "a new file"
        );
        // Nothing to tell: no key.
        let sets = [set(&root.join("gone"), "a")];
        assert!(baseline_inputs(&c, "base", &sets, vec![]).is_none());
        assert!(baseline_inputs(&c, "missing", &[set(&h1, "a")], vec![]).is_none());
        std::fs::remove_file(root.join("models/base.onnx")).unwrap();
        assert!(baseline_inputs(&c, "base", &[set(&h1, "a")], vec![]).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    fn rec(
        id: &str,
        group: Option<&str>,
        baseline: Option<Baseline>,
    ) -> super::super::service::RunRecord {
        use super::super::job::TrainRequest;
        use super::super::recipe::TrainRecipe;
        use crate::resources::catalog::TrainFamily;
        super::super::service::RunRecord {
            id: id.into(),
            name: id.into(),
            created: String::new(),
            request: TrainRequest {
                recipe: TrainRecipe::new(TrainFamily::Yolo26, "n").unwrap(),
                datasets: Vec::new(),
                include_public: false,
                scene: Default::default(),
                classes: None,
                rename: Default::default(),
                benchmark_sets: Vec::new(),
                benchmark_devices: Vec::new(),
                allow_large: false,
                install_dir: None,
                gpu: None,
                excluded: Default::default(),
                balance: Default::default(),
                compare_group: group.map(str::to_string),
                train_seed: None,
                ignore_memory_check: false,
            },
            device: String::new(),
            state: super::super::service::RunState::Done,
            stage: None,
            step: 0,
            steps: 8,
            percent: 0,
            detail: String::new(),
            progress: None,
            log: Vec::new(),
            dataset: None,
            output: None,
            registered: None,
            benchmark: None,
            benchmark_error: None,
            baseline,
            error: None,
            finished: None,
            run_dir: PathBuf::from(format!("/training/{id}")),
            retrying: false,
            resources: None,
        }
    }

    fn measured(key: &str) -> Baseline {
        Baseline {
            model: "base".into(),
            summary: Some(BenchSummary {
                ap50: Some(0.5793),
                device: Some("openvino:gpu.0".into()),
                ..BenchSummary::default()
            }),
            key: Some(key.into()),
            ..Baseline::default()
        }
    }

    #[test]
    fn a_baseline_is_reused_only_within_the_group_and_with_an_equal_key() {
        let me = Path::new("/training/s2");
        let runs = vec![
            rec("other", Some("cmp-2"), Some(measured("k"))),
            rec("s0", Some("cmp-1"), Some(measured("k"))),
            rec("s1", Some("cmp-1"), None),
            rec("s2", Some("cmp-1"), None),
        ];
        let b = reusable_baseline(&runs, me, "k").unwrap();
        assert_eq!(
            b.reused_from.as_deref(),
            Some("s0"),
            "the group's, not the other group's"
        );
        assert_eq!(b.summary.as_ref().unwrap().ap50, Some(0.5793));
        assert_eq!(b.key.as_deref(), Some("k"));
        // Another key, no group, a note, no result or no device: measured again.
        assert!(reusable_baseline(&runs, me, "other").is_none());
        let mut lone = runs.clone();
        lone[3].request.compare_group = None;
        assert!(reusable_baseline(&lone, me, "k").is_none());
        for spoil in [
            |b: &mut Baseline| b.note = Some("not loaded".into()),
            |b: &mut Baseline| b.summary = None,
            |b: &mut Baseline| b.summary.as_mut().unwrap().device = None,
            |b: &mut Baseline| b.key = None,
        ] {
            let mut r = runs.clone();
            spoil(r[1].baseline.as_mut().unwrap());
            assert!(reusable_baseline(&r, me, "k").is_none());
        }
        // A reused one passes on where it came from.
        let mut chain = runs.clone();
        chain[1].baseline = None;
        chain[2].baseline = Some(Baseline {
            reused_from: Some("s-first".into()),
            ..measured("k")
        });
        assert_eq!(
            reusable_baseline(&chain, me, "k")
                .unwrap()
                .reused_from
                .as_deref(),
            Some("s-first")
        );
        // Never its own.
        let mut own = runs;
        own[3].baseline = Some(measured("k"));
        own[1].baseline = None;
        assert!(reusable_baseline(&own, me, "k").is_none());
    }

    #[test]
    fn summarize_reads_recall_cells_from_the_heldout_tags_only() {
        let tags = vec![
            // Bare tags of a built-in set run beside: never cells.
            cells("night", vec![cr("person", 500, 100)]),
            cells("camera:back", vec![cr("person", 500, 100)]),
            cells(
                "heldout:all",
                vec![cr("person", 40, 4), cr("raccoon", 9, 5)],
            ),
            cells("heldout:night", vec![cr("raccoon", 9, 5)]),
            cells("heldout:day", vec![cr("person", 40, 4)]),
            cells("heldout:in-scope", vec![cr("raccoon", 9, 5)]),
            cells("heldout:random", vec![cr("person", 4, 1)]),
            cells("heldout:cam:back:yard", vec![cr("raccoon", 9, 5)]),
            cells(
                "heldout:camscene:night:back:yard",
                vec![cr("raccoon", 9, 5)],
            ),
            cells("heldout:cam:front", vec![cr("person", 40, 4)]),
            cells("heldout:camscene:day:front", vec![cr("person", 40, 4)]),
        ];
        let b = summarize(&result(tags.clone(), true));
        let c = &b.recall_cells;
        assert_eq!(
            c.by_scene.keys().collect::<Vec<_>>(),
            ["all", "day", "night"]
        );
        assert_eq!(c.by_scene["night"], [cr("raccoon", 9, 5)]);
        assert_eq!(
            c.by_scene["all"],
            [cr("person", 40, 4), cr("raccoon", 9, 5)]
        );
        assert_eq!(
            c.by_camera.keys().collect::<Vec<_>>(),
            ["back:yard", "front"]
        );
        assert_eq!(
            c.by_camera_scene,
            [
                CameraSceneRecall {
                    camera: "back:yard".into(),
                    scene: "night".into(),
                    classes: vec![cr("raccoon", 9, 5)],
                },
                CameraSceneRecall {
                    camera: "front".into(),
                    scene: "day".into(),
                    classes: vec![cr("person", 40, 4)],
                },
            ]
        );
        // The JSON shape, and a round trip.
        let j = serde_json::to_value(&b).unwrap();
        let cell = &j["recall_cells"]["by_scene"]["night"][0];
        assert_eq!(cell["class"], "raccoon");
        assert_eq!(
            (cell["gt"].as_u64(), cell["missed"].as_u64()),
            (Some(9), Some(5))
        );
        assert_eq!(j["recall_cells"]["by_camera"]["back:yard"][0]["gt"], 9);
        assert_eq!(j["recall_cells"]["by_camera_scene"][1]["camera"], "front");
        assert_eq!(j["recall_cells"]["by_camera_scene"][1]["scene"], "day");
        let back: BenchSummary = serde_json::from_value(j).unwrap();
        assert_eq!(back.recall_cells, b.recall_cells);
        // Without the held-out set's own tags: no cells, whatever the bare tags say, and
        // nothing written.
        let bare: Vec<Breakdown> = tags
            .into_iter()
            .filter(|t| !t.key.starts_with("heldout:"))
            .collect();
        let b = summarize(&result(bare, false));
        assert!(b.recall_cells.is_empty());
        assert!(
            serde_json::to_value(&b)
                .unwrap()
                .get("recall_cells")
                .is_none()
        );
        // A run scored against pseudo ground truth (every breakdown of it relative, as the
        // benchmark writes them): no cells.
        let mut m = result(
            ["heldout:all", "heldout:night", "heldout:cam:back"]
                .map(|k| Breakdown {
                    relative: true,
                    ..cells(k, vec![cr("raccoon", 9, 5)])
                })
                .to_vec(),
            false,
        );
        m.devices[0]
            .run
            .as_mut()
            .unwrap()
            .accuracy
            .as_mut()
            .unwrap()
            .relative = true;
        let b = summarize(&m);
        assert!(b.recall_cells.is_empty());
        assert!(b.by_scene.contains_key("all"), "its AP per scene stays");
    }

    /// A model without any CCTV class has no accuracy report: its cells come from the run's
    /// `tag_recall`, held-out tags only, and nothing else is summarized from them.
    #[test]
    fn summarize_reads_the_tag_recall_of_a_run_without_accuracy() {
        use crate::benchmark::TagRecall;
        let mut m = result(vec![], false);
        let run = m.devices[0].run.as_mut().unwrap();
        run.accuracy = None;
        run.accuracy_note = Some("the model has none of the scored CCTV classes".into());
        let tag = |k: &str, v: Vec<ClassRecall>| TagRecall {
            key: k.into(),
            images: 3,
            per_class: v,
        };
        run.tag_recall = vec![
            tag("heldout:all", vec![cr("raccoon", 9, 5), cr("fox", 2, 2)]),
            tag("heldout:night", vec![cr("raccoon", 9, 5)]),
            tag("heldout:cam:back", vec![cr("fox", 2, 2)]),
            tag("heldout:camscene:night:back", vec![cr("raccoon", 9, 5)]),
        ];
        let b = summarize(&m);
        let c = &b.recall_cells;
        assert_eq!(c.by_scene["all"], [cr("raccoon", 9, 5), cr("fox", 2, 2)]);
        assert_eq!(c.by_scene["night"], [cr("raccoon", 9, 5)]);
        assert_eq!(c.by_camera["back"], [cr("fox", 2, 2)]);
        assert_eq!(c.by_camera_scene.len(), 1);
        assert!(b.by_scene.is_empty() && b.ap50.is_none());
        // Without the held-out set's `heldout:all`: nothing.
        m.devices[0].run.as_mut().unwrap().tag_recall.remove(0);
        assert!(summarize(&m).recall_cells.is_empty());
    }

    /// Cells per camera: (camera, cells).
    type CamCells<'a> = Vec<(&'a str, Vec<ClassRecall>)>;

    /// A finished run of filter `scene` with a fingerprint and cells: per scene `scenes`
    /// ((scene, cells, camera-in-scene cells)), over all held-out frames `all`, per camera
    /// `cams`.
    fn cell_run(
        id: &str,
        finished: &str,
        scene: dataset::SceneFilter,
        scenes: &[(&str, Vec<ClassRecall>, CamCells<'_>)],
        all: Vec<ClassRecall>,
        cams: CamCells<'_>,
    ) -> super::super::service::RunRecord {
        let mut r = rec(id, None, None);
        r.request.scene = scene;
        r.finished = Some(finished.into());
        r.output = Some(super::super::job::TrainOutput {
            name: id.into(),
            onnx: PathBuf::from(format!("/models/{id}.onnx")),
            yaml: PathBuf::from(format!("/models/{id}.yaml")),
            provenance: PathBuf::new(),
            family: crate::model::ModelFamilyKind::Yolo26,
            classes: vec!["person".into(), "raccoon".into()],
            license: String::new(),
            heldout: None,
            device: String::new(),
            train_seconds: 0.0,
        });
        let mut c = RecallCells::default();
        for (s, classes, by_cam) in scenes {
            c.by_scene.insert(s.to_string(), classes.clone());
            for (cam, v) in by_cam {
                c.by_camera_scene.push(CameraSceneRecall {
                    camera: cam.to_string(),
                    scene: s.to_string(),
                    classes: v.clone(),
                });
            }
        }
        if !all.is_empty() {
            c.by_scene.insert("all".into(), all);
        }
        c.by_camera = cams.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        r.benchmark = Some(BenchSummary {
            heldout_fingerprint: Some("fp".into()),
            recall_cells: c,
            ..BenchSummary::default()
        });
        r
    }

    #[test]
    fn measured_recall_takes_the_newest_run_per_scene_that_trained_on_it() {
        use dataset::SceneFilter;
        let general = cell_run(
            "general",
            "2026-11-01T10:00:00Z",
            SceneFilter::ALL_SCENES,
            &[
                (
                    "day",
                    vec![cr("person", 100, 10)],
                    vec![("front", vec![cr("person", 60, 6)])],
                ),
                (
                    "night",
                    vec![cr("raccoon", 30, 15)],
                    vec![("back", vec![cr("raccoon", 30, 15)])],
                ),
            ],
            vec![cr("person", 100, 10), cr("raccoon", 30, 15)],
            vec![
                ("front", vec![cr("person", 60, 6)]),
                ("back", vec![cr("raccoon", 30, 15)]),
            ],
        );
        // Newer, a night model: its night cells win; its day cells (frames it never trained
        // on) and its overall cells do not.
        let night = cell_run(
            "night",
            "2026-11-05T10:00:00Z",
            SceneFilter::NIGHT_IR,
            &[
                ("day", vec![cr("person", 100, 90)], vec![]),
                (
                    "night",
                    vec![cr("raccoon", 30, 3)],
                    vec![("back", vec![cr("raccoon", 30, 3)])],
                ),
            ],
            vec![cr("person", 100, 90), cr("raccoon", 30, 3)],
            vec![("back", vec![cr("raccoon", 30, 3)])],
        );
        // Newest, but without a held-out fingerprint (an older version's run): skipped.
        let mut old = cell_run(
            "no-fingerprint",
            "2026-11-09T10:00:00Z",
            SceneFilter::ALL_SCENES,
            &[("day", vec![cr("person", 10, 10)], vec![])],
            vec![cr("person", 10, 10)],
            vec![],
        );
        old.benchmark.as_mut().unwrap().heldout_fingerprint = None;
        // Newest, failed: skipped.
        let mut failed = cell_run(
            "failed",
            "2026-11-10T10:00:00Z",
            SceneFilter::ALL_SCENES,
            &[("day", vec![cr("person", 10, 10)], vec![])],
            vec![cr("person", 10, 10)],
            vec![],
        );
        failed.state = super::super::service::RunState::Failed;
        let runs = vec![night.clone(), general.clone(), old, failed];
        let m = measured_recall(&runs).unwrap();
        assert_eq!(m.scenes["day"].run.id, "general");
        assert_eq!(m.scenes["night"].run.id, "night");
        assert!(!m.scenes.contains_key("ir"), "no run measured IR");
        assert_eq!(
            m.scenes["night"].camera_class("back", "raccoon"),
            recall::Cell { gt: 30, missed: 3 }
        );
        assert_eq!(
            m.scenes["day"].camera_class("front", "person"),
            recall::Cell { gt: 60, missed: 6 }
        );
        let o = m.overall.as_ref().unwrap();
        assert_eq!(
            o.run.id, "general",
            "the newest run that trained on every scene"
        );
        // The slices know what their model can detect.
        assert_eq!(o.run.classes, ["person", "raccoon"]);
        assert_eq!(
            m.miss("back", "night", "fox").map(|x| x.class),
            Some(recall::InRun::Absent)
        );
        assert_eq!(o.class("raccoon"), recall::Cell { gt: 30, missed: 15 });
        assert_eq!(
            m.class_recall_on_camera("back", "raccoon").unwrap().cell,
            recall::Cell { gt: 30, missed: 15 }
        );
        // The order of the records does not matter, the stamps do.
        assert_eq!(
            measured_recall(&[general.clone(), night.clone()]),
            Some(m.clone())
        );
        // Only the night model: no overall slice (it never trained on the day frames among its
        // held-out frames), and its day cells are not the day scene's either.
        let m = measured_recall(std::slice::from_ref(&night)).unwrap();
        assert!(m.overall.is_none());
        assert!(!m.scenes.contains_key("day"));
        assert_eq!(m.scenes["night"].run.id, "night");
        assert!(m.miss("back", "day", "person").is_none());
        // A run without its model's output (its classes) is not used.
        let mut no_output = general.clone();
        no_output.output = None;
        assert_eq!(measured_recall(&[no_output]), None);
        // No runs, or none measured: None.
        assert_eq!(measured_recall(&[]), None);
        let mut bare = rec("bare", None, None);
        bare.benchmark = Some(BenchSummary {
            heldout_fingerprint: Some("fp".into()),
            ..BenchSummary::default()
        });
        assert_eq!(measured_recall(&[bare]), None);
    }

    /// Equal stamps: the later record wins (a run of the same second queued after).
    #[test]
    fn measured_recall_breaks_equal_stamps_by_record_order() {
        use dataset::SceneFilter;
        let at = "2026-11-01T10:00:00Z";
        let one = |id: &str, missed: usize| {
            cell_run(
                id,
                at,
                SceneFilter::ALL_SCENES,
                &[("day", vec![cr("person", 40, missed)], vec![])],
                vec![cr("person", 40, missed)],
                vec![],
            )
        };
        let (a, b) = (one("first", 4), one("second", 8));
        let m = measured_recall(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(m.scenes["day"].run.id, "second");
        assert_eq!(m.overall.as_ref().unwrap().run.id, "second");
        let m = measured_recall(&[b, a]).unwrap();
        assert_eq!(m.scenes["day"].run.id, "first");
        // `finished` missing: `created` decides.
        let mut c = one("created-later", 1);
        c.finished = None;
        c.created = "2026-11-02T00:00:00Z".into();
        let m = measured_recall(&[c, one("finished-earlier", 2)]).unwrap();
        assert_eq!(m.scenes["day"].run.id, "created-later");
    }
}
