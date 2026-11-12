//! The pre-flight report: what a run would train on and whether it is worth starting, from the same
//! plan the run itself makes ([`planned`], used by the job, [`super::preview_plan`] and through it
//! the Train tab and `train prepare --dry-run`).
//!
//! - [`advice`]: warnings a plan alone cannot give (a class with few boxes, few held-out frames, a
//!   scene with few frames), in [`dataset::Plan::warnings`] so they reach the run row and `train.log`;
//! - [`refusals`]: data a run must not start on, with the next action;
//! - [`report`]: boxes per class and scene, the splits, the bulk-accepted share, frames dropped;
//! - [`run_needs`], [`combine`], [`refusal`]: disk needed against disk free, the volumes and the free
//!   space kept ([`disk::margin`]) as for an import;
//! - [`gpu_need`], [`gpu_warning`]: the run's estimated GPU memory against what the card has free
//!   now ([`crate::backend::vram`]), naming the service's models loaded on it; a warning, never a
//!   refusal, and none when the card cannot be read.
//!
//! Everything here is advice over a pure plan, but for reading file sizes, [`disk::space`] and the
//! card's free memory.

use super::capability::Accelerator;
use super::dataset::{self, FrameRef, Options, Plan, SceneCounts, Source};
use super::import::disk::{self, Space};
use super::recipe::TrainRecipe;
use crate::backend::vram;
use crate::resources::catalog::{self, Trainer};
use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Training boxes under which a class gets a warning: fine-tuning from pretrained weights needs
/// dozens of examples of a class before its AP means much, and 50 is also the fewest frames a
/// validation split wants ([`dataset::MIN_VALID`]). No refusal: a rare animal may be all there is.
pub const MIN_CLASS_BOXES: usize = 50;
/// Held-out frames under which the benchmark gets a warning: one frame is then 10% of the score, so
/// differences between runs are noise. No refusal: a rough number still beats none.
pub const MIN_HELDOUT_FRAMES: usize = 10;
/// Training frames under which a scene gets a warning: the same number as
/// [`MIN_CLASS_BOXES`], for the same reason; a few night frames among many day frames teach little.
pub const MIN_SCENE_FRAMES: usize = 50;
/// Disk the run's work folder takes, as a multiple of the base weights. Measured on 2-epoch runs:
/// YOLO26n 16 MB of 5.5 MB (best + last), RT-DETRv2 S 77 MB of 81 MB, RF-DETR nano 1.2 GB of 366 MB
/// (the mapped base, `last.ckpt` with the optimizer, the best checkpoints).
pub const WORK_WEIGHT_FACTOR: u64 = 4;
/// Disk per frame beyond its image: the label file, its share of the COCO JSON and the lists.
pub const LABEL_BYTES_PER_FRAME: u64 = 8 * 1024;

/// [`dataset::plan`] with the warnings of [`advice`]: what every run and preview goes by.
pub fn planned(sources: &[Source], opts: &Options) -> Result<Plan> {
    let mut plan = dataset::plan(sources, opts)?;
    let more = advice(&plan, sources);
    plan.warnings.extend(more);
    Ok(plan)
}

fn frame(sources: &[Source], r: FrameRef) -> Option<&dataset::Frame> {
    sources.get(r.0)?.frames.get(r.1)
}

/// Training frames per scene.
fn scene_frames(plan: &Plan, sources: &[Source]) -> SceneCounts {
    let mut c = SceneCounts::default();
    for f in plan.train.iter().filter_map(|(r, _)| frame(sources, *r)) {
        *scene_slot(&mut c, f.scene.as_deref()) += 1;
    }
    c
}

fn scene_slot<'a>(c: &'a mut SceneCounts, scene: Option<&str>) -> &'a mut usize {
    match scene {
        Some("day") => &mut c.day,
        Some("night") => &mut c.night,
        Some("ir") => &mut c.ir,
        _ => &mut c.unknown,
    }
}

/// The warnings of a usable plan, each with the next action (see the constants). A class without
/// any training box has its own warning in the plan.
pub fn advice(plan: &Plan, sources: &[Source]) -> Vec<String> {
    let mut out = Vec::new();
    for c in &plan.classes {
        let n = plan.stats.train_boxes.get(c).copied().unwrap_or(0);
        if n > 0 && n < MIN_CLASS_BOXES {
            let capped = dataset::static_cap_clause(&plan.stats, c)
                .map(|clause| format!(" Also, {clause}."))
                .unwrap_or_default();
            out.push(format!(
                "Only {n} training box(es) of '{c}': the model will find it poorly. Review more \
                 frames with it, or leave the class out.{capped}"
            ));
        }
    }
    let held = plan.heldout.len();
    if held > 0 && held < MIN_HELDOUT_FRAMES {
        out.push(format!(
            "Only {held} held-out frame(s): the benchmark of this run is a rough guess, one frame \
             moves it a lot. Review and export frames from more cameras or days."
        ));
    }
    let scenes = scene_frames(plan, sources);
    let all = [
        ("day", scenes.day),
        ("night", scenes.night),
        ("IR", scenes.ir),
    ];
    for (name, n) in all {
        if n == 0 || n >= MIN_SCENE_FRAMES {
            continue;
        }
        let others: Vec<&str> = all
            .iter()
            .filter(|(o, m)| *o != name && *m >= MIN_SCENE_FRAMES)
            .map(|(o, _)| *o)
            .collect();
        out.push(if others.is_empty() {
            format!("Only {n} {name} frame(s) to train on: review more of them.")
        } else {
            format!(
                "Only {n} {name} frame(s): review more, or train {} only.",
                others.join(" + ")
            )
        });
    }
    out
}

/// Why a run must not start on `plan`, each with the next action. A plan with no training box at
/// all never gets here ([`dataset::plan`] refuses it). Held-out frames are what the run's result is
/// measured on, so no frame held out means no way to tell whether the model is any good.
/// `training.heldout_share` 0 opts out of that and stays a warning.
pub fn refusals(plan: &Plan, sources: &[Source], opts: &Options) -> Vec<String> {
    let mut out = Vec::new();
    if plan.heldout.is_empty() && opts.heldout_share > 0.0 {
        let own: usize = sources
            .iter()
            .filter(|s| s.public.is_none())
            .map(|s| s.frames.len())
            .sum();
        out.push(if own == 0 {
            "No frame of your own to hold out, so this run could not be measured on your cameras: \
             add a Dataset tab export."
                .to_string()
        } else {
            format!(
                "No frame can be held out of the {own} frame(s) of your own, so this run could not \
                 be measured: review more frames and export again."
            )
        });
    }
    out
}

/// Refuse a run on `plan` ([`refusals`]).
pub fn check_plan(plan: &Plan, sources: &[Source], opts: &Options) -> Result<()> {
    let r = refusals(plan, sources, opts);
    if r.is_empty() {
        Ok(())
    } else {
        bail!("{}", r.join(" "))
    }
}

/// One class of a [`Report`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClassLine {
    pub name: String,
    pub train: usize,
    pub valid: usize,
    pub heldout: usize,
    /// Training boxes per scene.
    pub by_scene: SceneCounts,
}

/// What a run of a plan would train on ([`report`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    pub classes: Vec<ClassLine>,
    /// Training frames per scene.
    pub scenes: SceneCounts,
    pub train_frames: usize,
    /// Training images per epoch (repeats counted).
    pub train_images: usize,
    pub valid_frames: usize,
    pub heldout_frames: usize,
    /// Training frames without an object, and those a reviewer confirmed or rejected a box on.
    pub negatives: usize,
    pub hard_negatives: usize,
    /// Training and validation frames of your own accepted in bulk, of `own_frames`; None when an
    /// export does not say (older versions).
    pub bulk_accepted: Option<usize>,
    pub own_frames: usize,
    /// Frames of the folder under review not reviewed yet (not in any export); None = unknown.
    pub unreviewed: Option<u64>,
    /// Frames left out for ignore regions (unsure, too small, under the size floor;
    /// [`dataset::Stats::ignore_frames`]).
    pub ignore_frames: usize,
    /// What the plan does with frames with ignore regions (`training.ignore`).
    pub ignore_mode: dataset::IgnoreMode,
    /// [`dataset::IgnoreMode::Background`]: training frames trained with an unlabelled object
    /// ([`dataset::Stats::ignore_kept`]) and their regions per class.
    pub ignore_kept: usize,
    pub ignore_boxes_background: BTreeMap<String, usize>,
    /// Of `ignore_frames` (background): frames with nothing but regions, and frames of camera days
    /// validated on.
    pub ignore_only: usize,
    pub ignore_valid_day: usize,
    /// Size floor regions given back as labels at the run's input, per label
    /// ([`dataset::Stats::floor_restored`]); the plan's warnings say the rest of the floor story
    /// (exports of other floors, an input smaller than the floor's or unknown).
    pub floor_restored: BTreeMap<String, usize>,
    pub parked_capped: usize,
    /// Frames showing nothing but objects at static spots left out over the static cap
    /// ([`dataset::Stats::static_capped`]; the spots in `dataset::Stats::static_caps`).
    pub static_capped: usize,
    /// Static spots of the train / valid pool ([`dataset::Stats::static_spots`]).
    pub static_spots: usize,
    pub duplicates_dropped: usize,
    pub scene_filtered: usize,
    /// Plain sentences beside the counts.
    pub notes: Vec<String>,
    /// The plan's warnings.
    pub warnings: Vec<String>,
    /// Why the run would not start ([`refusals`]).
    pub refusals: Vec<String>,
}

/// The [`Report`] of `plan`.
pub fn report(plan: &Plan, sources: &[Source], opts: &Options, unreviewed: Option<u64>) -> Report {
    // The frames as the plan read them (size floor regions given back at the run's input).
    let at_input = dataset::at_run_input(sources, plan, opts);
    let sources: &[Source] = &at_input;
    let class_at = |b: usize| plan.classes.get(b);
    let mut classes: BTreeMap<&str, ClassLine> = plan
        .classes
        .iter()
        .map(|c| {
            (
                c.as_str(),
                ClassLine {
                    name: c.clone(),
                    train: 0,
                    valid: 0,
                    heldout: 0,
                    by_scene: SceneCounts::default(),
                },
            )
        })
        .collect();
    let mut add = |r: FrameRef, slot: fn(&mut ClassLine) -> &mut usize, scenes: bool| {
        let Some(f) = frame(sources, r) else { return };
        for (i, _) in dataset::mapped_boxes(f, opts, &plan.classes) {
            if let Some(l) = class_at(i).and_then(|c| classes.get_mut(c.as_str())) {
                *slot(l) += 1;
                if scenes {
                    *scene_slot(&mut l.by_scene, f.scene.as_deref()) += 1;
                }
            }
        }
    };
    for (r, _) in &plan.train {
        add(*r, |l| &mut l.train, true);
    }
    let trained: BTreeSet<FrameRef> = plan.train.iter().map(|(r, _)| *r).collect();
    // Without a validation split of its own the trainer validates on its training frames.
    for r in plan.valid.iter().filter(|r| !trained.contains(r)) {
        add(*r, |l| &mut l.valid, false);
    }
    for r in &plan.heldout {
        add(*r, |l| &mut l.heldout, false);
    }
    let mut own = 0usize;
    let mut bulk = 0usize;
    let used: BTreeSet<FrameRef> = trained.iter().chain(&plan.valid).copied().collect();
    for r in &used {
        if !sources.get(r.0).is_some_and(|s| s.public.is_none()) {
            continue;
        }
        own += 1;
        bulk += usize::from(frame(sources, *r).is_some_and(|f| f.bulk_accepted));
    }
    let bulk_accepted = sources.iter().all(|s| s.notes_known).then_some(bulk);
    let s = &plan.stats;
    let mut notes = Vec::new();
    match bulk_accepted {
        Some(n) if n > 0 => notes.push(format!(
            "{n} of {own} frame(s) ({:.0}%) were accepted in bulk: nobody looked at them one by \
             one. Spot-check them on the Review tab if the result disappoints.",
            100.0 * n as f64 / own.max(1) as f64
        )),
        Some(_) => {}
        None => notes.push(
            "How many frames were accepted in bulk is not known: an export is from an older \
             version. Export again to see it."
                .into(),
        ),
    }
    if let Some(note) = dataset::static_cap_note(s, opts.static_cap) {
        notes.push(note);
    }
    if let Some(note) = dataset::static_cap_idle_note(&plan.classes, opts.static_cap) {
        notes.push(note);
    }
    if let Some(n) = unreviewed.filter(|n| *n > 0) {
        notes.push(format!(
            "{n} frame(s) in the folder under review are not reviewed yet, so they are in no \
             export: review them and export again to train on them."
        ));
    }
    Report {
        classes: plan
            .classes
            .iter()
            .filter_map(|c| classes.remove(c.as_str()))
            .collect(),
        scenes: scene_frames(plan, sources),
        train_frames: plan.train.len(),
        train_images: plan.train.iter().map(|(_, n)| *n as usize).sum(),
        valid_frames: plan.valid.len(),
        heldout_frames: plan.heldout.len(),
        negatives: s.train_negatives,
        hard_negatives: s.hard_negatives,
        bulk_accepted,
        own_frames: own,
        unreviewed,
        ignore_frames: s.ignore_frames,
        ignore_mode: opts.ignore,
        ignore_kept: s.ignore_kept,
        ignore_boxes_background: s.ignore_boxes_background.clone(),
        ignore_only: s.ignore_only,
        ignore_valid_day: s.ignore_valid_day,
        floor_restored: s.floor_restored.clone(),
        parked_capped: s.parked_capped,
        static_capped: s.static_capped,
        static_spots: s.static_spots,
        duplicates_dropped: s.duplicates_dropped,
        scene_filtered: s.scene_filtered,
        notes,
        warnings: plan.warnings.clone(),
        refusals: refusals(plan, sources, opts),
    }
}

/// [`Report`] as text (`train prepare --dry-run`): warnings come with the plan's text.
pub fn text(r: &Report) -> String {
    let mut out = format!(
        "pre-flight: {} training frames ({} images per epoch, {} without objects, {} of them \
         confirmed empty or with a rejected box), {} validation, {} held out\n",
        r.train_frames,
        r.train_images,
        r.negatives,
        r.hard_negatives,
        r.valid_frames,
        r.heldout_frames
    );
    let sc = &r.scenes;
    out.push_str(&format!(
        "training frames per scene: day {}, night {}, ir {}, unknown {}\n",
        sc.day, sc.night, sc.ir, sc.unknown
    ));
    out.push_str("boxes per class (train / validation / held out; train by day, night, ir):\n");
    for c in &r.classes {
        let s = &c.by_scene;
        out.push_str(&format!(
            "  {}: {} / {} / {}; {}, {}, {}\n",
            c.name, c.train, c.valid, c.heldout, s.day, s.night, s.ir
        ));
    }
    if r.ignore_kept > 0 {
        let per: Vec<String> = r
            .ignore_boxes_background
            .iter()
            .map(|(c, n)| format!("{c} {n}"))
            .collect();
        out.push_str(&format!(
            "trained with an unsure or too-small object as background: {} frame(s) ({})\n",
            r.ignore_kept,
            per.join(", ")
        ));
    }
    if !r.floor_restored.is_empty() {
        let per: Vec<String> = r
            .floor_restored
            .iter()
            .map(|(c, n)| format!("{c} {n}"))
            .collect();
        out.push_str(&format!(
            "under the export's size floor but not at this run's input, trained as labels: {}\n",
            per.join(", ")
        ));
    }
    let background = r.ignore_mode == dataset::IgnoreMode::Background;
    let (ignore_all, ignore_only, ignore_valid) = if background {
        (0, r.ignore_only, r.ignore_valid_day)
    } else {
        (r.ignore_frames, 0, 0)
    };
    let dropped = [
        (ignore_all, "with unsure or too-small boxes"),
        (
            ignore_only,
            "with nothing but unsure or too-small objects (no label to train)",
        ),
        (
            ignore_valid,
            "with unsure or too-small objects on camera days used for validation",
        ),
        (r.parked_capped, "over the parked-spot cap"),
        (
            r.static_capped,
            "showing nothing but parked objects, over the cap per static spot",
        ),
        (
            r.duplicates_dropped,
            "that repeat a frame of another export",
        ),
        (r.scene_filtered, "outside the scene filter"),
    ];
    for (n, why) in dropped.iter().filter(|(n, _)| *n > 0) {
        out.push_str(&format!("left out: {n} frame(s) {why}\n"));
    }
    for n in &r.notes {
        out.push_str(&format!("note: {n}\n"));
    }
    for x in &r.refusals {
        out.push_str(&format!("refused: {x}\n"));
    }
    out
}

/// Disk a run needs on one folder's volume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VolumeNeed {
    pub dir: PathBuf,
    pub bytes: u64,
    /// The next action when it does not fit.
    pub advice: &'static str,
}

/// [`VolumeNeed::advice`] of the training folder.
pub const TRAINING_ADVICE: &str = "free some space, or choose a Training folder on a bigger disk \
     (This machine on the Train tab, training.dir in the config)";
/// [`VolumeNeed::advice`] of the folder a model is installed in.
pub const MODELS_ADVICE: &str = "free some space, or install the model in a folder on a bigger disk (Advanced on the Train \
     tab, --install-dir)";
/// [`VolumeNeed::advice`] of the tools folder (environment, base weights).
pub const TOOLS_ADVICE: &str = "free some space, or set download_dir to a bigger disk";

/// Whether `file` can be hard-linked into `dir` (or its nearest existing ancestor): the same
/// volume, and a file system that has hard links (exFAT does not, so a same-volume dataset can still
/// be a full copy). Probed with a link that is removed at once; a folder that refuses the probe
/// says nothing about the volume, so it counts as linkable.
fn hard_links_work(file: &Path, dir: &Path) -> bool {
    if !disk::same_volume(file, dir) {
        return false;
    }
    let Some(parent) = dir
        .ancestors()
        .find(|p| !p.as_os_str().is_empty() && p.is_dir())
    else {
        return false;
    };
    let probe = parent.join(format!(".bop-link-probe-{}", uuid::Uuid::new_v4()));
    match std::fs::hard_link(file, &probe) {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(e) => e.kind() == std::io::ErrorKind::PermissionDenied,
    }
}

/// Bytes of the plan's frames that cannot be hard-linked into `run_dir`
/// ([`dataset::write`] then copies them), each frame once.
fn copied_bytes(plan: &Plan, sources: &[Source], run_dir: &Path) -> u64 {
    let frames: BTreeSet<FrameRef> = plan
        .train
        .iter()
        .map(|(r, _)| *r)
        .chain(plan.valid.iter().copied())
        .chain(plan.heldout.iter().copied())
        .collect();
    let mut linked: BTreeMap<usize, bool> = BTreeMap::new();
    let mut bytes = 0u64;
    for r in frames {
        let Some(f) = frame(sources, r) else { continue };
        if *linked
            .entry(r.0)
            .or_insert_with(|| hard_links_work(&f.image, run_dir))
        {
            continue;
        }
        bytes = bytes.saturating_add(std::fs::metadata(&f.image).map_or(0, |m| m.len()));
    }
    bytes
}

/// What one run needs: under its run folder the frames that cannot be linked, their labels, and
/// the work folder ([`WORK_WEIGHT_FACTOR`]; RF-DETR with night augmentation also writes an extra
/// copy of up to every training image); under `models` the ONNX, no larger than the base weights.
pub fn run_needs(
    plan: &Plan,
    sources: &[Source],
    recipe: &TrainRecipe,
    run_dir: &Path,
    models: &Path,
) -> Result<Vec<VolumeNeed>> {
    let weights = recipe.base()?.size();
    let frames = plan.train.len() + plan.valid.len() + plan.heldout.len();
    let mut bytes = copied_bytes(plan, sources, run_dir)
        .saturating_add(frames as u64 * LABEL_BYTES_PER_FRAME)
        .saturating_add(weights.saturating_mul(WORK_WEIGHT_FACTOR));
    let a = &recipe.augment;
    if recipe.family.trainer() == Trainer::RfDetr && (a.dark > 0.0 || a.noise > 0.0 || a.gray > 0.0)
    {
        let images: u64 = plan
            .train
            .iter()
            .filter_map(|(r, _)| frame(sources, *r))
            .map(|f| std::fs::metadata(&f.image).map_or(0, |m| m.len()))
            .sum();
        bytes = bytes.saturating_add(images);
    }
    Ok(vec![
        VolumeNeed {
            dir: run_dir.to_path_buf(),
            bytes,
            advice: TRAINING_ADVICE,
        },
        VolumeNeed {
            dir: models.to_path_buf(),
            bytes: weights,
            advice: MODELS_ADVICE,
        },
    ])
}

/// `needs` per volume, in order, each with the first folder and advice of its volume.
pub fn combine(needs: Vec<VolumeNeed>) -> Vec<VolumeNeed> {
    let mut out: Vec<VolumeNeed> = Vec::new();
    for n in needs.into_iter().filter(|n| n.bytes > 0) {
        match out.iter_mut().find(|o| disk::same_volume(&o.dir, &n.dir)) {
            Some(o) => o.bytes = o.bytes.saturating_add(n.bytes),
            None => out.push(n),
        }
    }
    out
}

/// One volume of the report: what a run needs and what is there.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DiskLine {
    pub dir: PathBuf,
    pub need: u64,
    pub free: Option<u64>,
    /// The free space a run leaves on the volume ([`disk::margin`]).
    pub keep: Option<u64>,
    /// "about 3.1 GB needed under X, 13.2 GB free".
    pub text: String,
}

fn size(b: u64) -> String {
    catalog::format_size(b)
}

/// The [`DiskLine`] of a need, given its volume's space.
pub fn disk_line(n: &VolumeNeed, s: Option<Space>) -> DiskLine {
    DiskLine {
        dir: n.dir.clone(),
        need: n.bytes,
        free: s.map(|s| s.available),
        keep: s.map(|s| s.margin()),
        text: format!(
            "about {} needed under {}, {} free",
            size(n.bytes),
            n.dir.display(),
            s.map_or_else(|| "unknown".to_string(), |s| size(s.available))
        ),
    }
}

/// Why the run does not fit, or None: each need plus the volume's [`disk::margin`] must be free. A
/// volume whose space is unknown never refuses. Pure.
pub fn refusal(needs: &[(VolumeNeed, Option<Space>)]) -> Option<String> {
    let lines: Vec<String> = needs
        .iter()
        .filter_map(|(n, s)| {
            let s = s.filter(|s| n.bytes.saturating_add(s.margin()) > s.available)?;
            Some(format!(
                "Not enough disk space: the run needs about {} under {}, and {} is free (it keeps \
                 {} free so the disk never fills up): {}.",
                size(n.bytes),
                n.dir.display(),
                size(s.available),
                size(s.margin()),
                n.advice
            ))
        })
        .collect();
    (!lines.is_empty()).then(|| lines.join(" "))
}

/// What a run of `recipe` needs on `accel`: the Train tab's estimate at the recipe's batch and
/// size ([`super::estimate`], RF-DETR multi-scale as the run decides it). None when the family
/// and size have no estimate there.
pub fn gpu_need(recipe: &TrainRecipe, accel: &Accelerator) -> Option<vram::JobNeed> {
    let multi_scale = recipe.multi_scale_on(accel.memory_mb());
    let est = super::estimate::estimate_with(
        recipe.family,
        &recipe.size,
        recipe.imgsz,
        recipe.batch,
        1,
        1,
        accel.target(),
        multi_scale,
    )?;
    Some(vram::JobNeed {
        what: "this training run".into(),
        need_mb: est.vram_mb,
        detail: format!(
            "{} {} at batch {} and {} px{}, the Train tab's estimate",
            recipe.family.title(),
            recipe.size,
            recipe.batch,
            recipe.imgsz,
            if multi_scale { " multi-scale" } else { "" }
        ),
        smaller: "a smaller batch or size".into(),
    })
}

/// The warning before a run of `recipe` on `accel` when its card is short of free memory now
/// ([`vram::check_cuda_job`]); None when it fits, when the card cannot be read, and on Apple
/// silicon (unified memory: the system decides).
pub fn gpu_warning(recipe: &TrainRecipe, accel: &Accelerator) -> Option<String> {
    match accel {
        Accelerator::Cuda { ordinal, .. } => {
            vram::check_cuda_job(*ordinal, &gpu_need(recipe, accel)?)
        }
        Accelerator::Mps { .. } => None,
    }
}

#[cfg(test)]
thread_local! {
    /// A test that means to look at the real volume. Off, so that a job test's outcome does not
    /// depend on how full the machine's disk is; [`refusal`] is tested on its own.
    pub(crate) static REAL_DISK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`refusal`] on the volumes as they are now.
pub fn check_disk(needs: Vec<VolumeNeed>) -> Result<()> {
    #[cfg(test)]
    if !REAL_DISK.with(std::cell::Cell::get) {
        return Ok(());
    }
    let with_space: Vec<(VolumeNeed, Option<Space>)> = combine(needs)
        .into_iter()
        .map(|n| {
            let s = disk::space(&n.dir);
            (n, s)
        })
        .collect();
    match refusal(&with_space) {
        Some(m) => Err(anyhow!(m)),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests;
