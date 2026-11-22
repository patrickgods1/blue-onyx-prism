//! "Your cameras" (dataset id [`ID`]): the reviewed frames of the folder under review
//! (`collect.review_dir`, default `collect.dir`) as a benchmark set, read live, so models are
//! compared and tuned on the exact Blue Iris uploads the service answers.
//!
//! - **Selection** ([`load`]): exactly the rules of the Review tab's export, in memory, never an
//!   export on disk: [`export::reviewed_snapshot`] (reviewed frames only, so reopened and skipped
//!   frames are out; labels outside `collect.classes` and frames with pending boxes left out with a
//!   warning) and [`export::benchmark_manifest`] (ignore regions, [`crate::collect::labels::LabelBox::is_ignore`],
//!   and the boxes under `collect.size_floor` as `ignore`; a frame confirmed empty carries the
//!   `hard-negative` tag; scored on `collect.classes`). Every split is in: tags `<scene>`,
//!   `camera:<c>`, `date:<d>`, `split:<s>` (the default split plus the days `heldout-days.json`
//!   lists, read without its lock and never grown), `bucket:<b>`. Each image carries its camera day
//!   as its cross-validation group ([`group_of`]).
//! - **Read-only**: no lock is taken and nothing is written to the review folder. A label file that
//!   changes while it is read is left out and counted ([`Survey::changed`]).
//! - **Availability**: a review folder that cannot be read is [`Load::Missing`] (the Review
//!   routes' `folder_missing`), never an empty set; a folder without a reviewed frame is
//!   [`Load::Empty`]. A run skips the set with a note in both cases. Few frames are allowed:
//!   under [`MIN_CAMERA_DAYS`] camera days or [`MIN_OBJECTS`] labelled objects, the run warns that
//!   the figures are low confidence ([`Survey::low_confidence`]).
//! - **Per-model exclusion** ([`seen`]): the frames a trained model trained or validated on are
//!   left out of that model's run on this set ([`for_model`]); the result records how many
//!   ([`crate::benchmark::report::LeftOut`]).
//! - **Ground-truth snapshot** ([`snapshot`], `benchmark-gt.json` beside `benchmark.json`): the labels
//!   a run scored against; every re-score of saved results reads them, so results stay put while
//!   labels change, and [`snapshot::changes`] / [`rescore`] tell and catch up.

pub mod live;
pub mod rescore;
pub mod seen;
pub mod snapshot;
#[cfg(test)]
pub(crate) mod tests;

use super::images::{GroundTruth, ImageSet, SetKind};
use crate::collect::labels::SizeFloor;
use crate::collect::labels::export::{self, ClassFilter, ExportFrame, Group, Split, SplitConfig};
use crate::config::Config;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The dataset id of the set.
pub const ID: &str = "cameras";
/// Its title on the pages.
pub const TITLE: &str = "Your cameras";
/// Fewer camera days than this: the figures are low confidence (the plan's rule, shared with the
/// threshold pick of a later phase).
pub const MIN_CAMERA_DAYS: usize = 4;
/// Fewer labelled objects than this: the figures are low confidence.
pub const MIN_OBJECTS: usize = 50;

/// Where the set comes from: the folder under review and what the export reads of the config.
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    /// The folder under review (`collect.review_dir`, default `collect.dir`).
    pub dir: PathBuf,
    /// `collect.classes`: the scored classes; a training label outside them leaves its frame out.
    pub classes: Vec<String>,
    /// `collect.size_floor`.
    pub size_floor: Option<SizeFloor>,
    /// `collect.hidden_classes`: not read by the selection (the export keeps their boxes), only
    /// to find the Review tab's index of the folder ([`live`]).
    pub hidden: Vec<String>,
    /// Why the folder cannot be used now ([`crate::collect::CollectConfig::review_folder_problem`]).
    pub problem: Option<String>,
}

impl Source {
    /// The set's source in `config`.
    pub fn of(config: &Config) -> Self {
        let dir = config.collect.resolved_review_dir(&config.data_root());
        Self {
            problem: config.collect.review_folder_problem(&dir),
            classes: config.collect.classes.clone(),
            size_floor: config.collect.size_floor,
            hidden: config.collect.hidden_classes.clone(),
            dir,
        }
    }

    /// A key that changes when anything the selection reads of the config does.
    pub fn key(&self) -> String {
        format!(
            "{}|{:?}|{:?}|{:?}",
            self.dir.display(),
            self.classes,
            self.size_floor,
            self.hidden
        )
    }
}

/// What the set holds, for the Run tab's line and a run's notes.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Survey {
    pub frames: usize,
    pub cameras: usize,
    /// Camera days (the cross-validation groups).
    pub days: usize,
    /// Labelled objects (no ignore region) of the scored classes.
    pub objects: usize,
    /// Frames confirmed empty (hard negatives).
    pub empty: usize,
    /// Frames per scene folder.
    pub day: usize,
    pub night: usize,
    pub ir: usize,
    /// Smallest and largest frame (width, height), by pixel count (ties: the narrower first).
    pub smallest: Option<(u32, u32)>,
    pub largest: Option<(u32, u32)>,
    /// Distinct frame sizes.
    pub sizes: usize,
    /// Reviewed frames whose label file changed while it was read: left out this time.
    pub changed: usize,
    /// Reviewed frames whose image or sidecar is gone (the walk's warnings name the frames left
    /// out for unknown labels or pending boxes).
    pub lost: usize,
}

impl Survey {
    fn of(frames: &[ExportFrame], set: &ImageSet) -> Self {
        let mut s = Self {
            frames: frames.len(),
            ..Self::default()
        };
        let mut cameras = BTreeSet::new();
        let mut sizes = BTreeSet::new();
        let mut days = BTreeSet::new();
        for f in frames {
            cameras.insert(f.rel.camera.as_str());
            days.insert((f.rel.camera.as_str(), f.rel.date.as_str()));
            match f.rel.scene.as_str() {
                "night" => s.night += 1,
                "ir" => s.ir += 1,
                _ => s.day += 1,
            }
            s.empty += usize::from(f.labels.empty_confirmed);
            let px = |d: (u32, u32)| (u64::from(d.0) * u64::from(d.1), d.0);
            let d = (f.width, f.height);
            sizes.insert(d);
            if s.smallest.is_none_or(|m| px(d) < px(m)) {
                s.smallest = Some(d);
            }
            if s.largest.is_none_or(|m| px(d) > px(m)) {
                s.largest = Some(d);
            }
        }
        s.objects = set
            .images
            .iter()
            .flat_map(|i| i.gt.iter().flatten())
            .filter(|b| !b.ignore)
            .count();
        s.cameras = cameras.len();
        s.sizes = sizes.len();
        s.days = days.len();
        s
    }

    /// Too little to trust a difference: under [`MIN_CAMERA_DAYS`] camera days or [`MIN_OBJECTS`]
    /// labelled objects.
    pub fn low_confidence(&self) -> bool {
        self.days < MIN_CAMERA_DAYS || self.objects < MIN_OBJECTS
    }

    /// The frame sizes: "1920x1080" when they are all one size, "1520x2688 and 2688x1520" for two,
    /// else "1280x720 to 3840x2160 (5 sizes)".
    pub fn resolution_range(&self) -> String {
        let name = |(w, h): (u32, u32)| format!("{w}x{h}");
        match (self.smallest, self.largest) {
            (Some(a), Some(b)) if a == b => name(a),
            (Some(a), Some(b)) if self.sizes <= 2 => format!("{} and {}", name(a), name(b)),
            (Some(a), Some(b)) => format!("{} to {} ({} sizes)", name(a), name(b), self.sizes),
            _ => String::new(),
        }
    }

    /// The Run tab's line: frames, cameras, days, resolution range, scenes.
    pub fn line(&self) -> String {
        let mut parts = vec![
            plural(self.frames, "frame"),
            plural(self.cameras, "camera"),
            plural(self.days, "camera day"),
        ];
        let range = self.resolution_range();
        if !range.is_empty() {
            parts.push(range);
        }
        parts.push(format!(
            "day {} \u{b7} night {} \u{b7} IR {}",
            self.day, self.night, self.ir
        ));
        parts.join(" \u{b7} ")
    }

    /// The run's warning when the set is small ([`Self::low_confidence`]); None otherwise.
    pub fn warning(&self) -> Option<String> {
        self.low_confidence().then(|| {
            format!(
                "{TITLE} has {}, {} and {}: too few for a confident comparison (fewer than \
                 {MIN_CAMERA_DAYS} camera days or {MIN_OBJECTS} labelled objects), so treat its \
                 figures as low confidence. Review more frames on the Train tab's Review step",
                plural(self.frames, "frame"),
                plural(self.days, "camera day"),
                plural(self.objects, "labelled object"),
            )
        })
    }
}

fn plural(n: usize, what: &str) -> String {
    if n == 1 {
        format!("1 {what}")
    } else {
        format!("{n} {what}s")
    }
}

/// The set, loaded.
#[derive(Debug, Clone)]
pub struct Loaded {
    pub set: ImageSet,
    /// Every frame of the selection (before a run's image limit): file name (the export's flat
    /// name) -> label file revision. The snapshot keeps it, so a frame reviewed later is "new".
    pub reviewed: BTreeMap<String, String>,
    pub survey: Survey,
    /// What the walk could not use, in the export's words.
    pub warnings: Vec<String>,
}

/// What [`load`] found.
#[derive(Debug, Clone)]
pub enum Load {
    Ready(Box<Loaded>),
    /// The folder under review cannot be read (`folder_missing`): why.
    Missing(String),
    /// No reviewed frame (the walk's warnings beside).
    Empty(Vec<String>),
}

impl Load {
    /// The note of a run that skips the set; None when it is ready.
    pub fn skip_note(&self) -> Option<String> {
        match self {
            Load::Ready(_) => None,
            Load::Missing(why) => Some(format!(
                "{TITLE} was skipped (folder_missing): {}",
                why.trim_end_matches('.')
            )),
            Load::Empty(_) => Some(format!(
                "{TITLE} was skipped: the folder under review has no reviewed frame yet. Review \
                 frames on the Train tab's Review step to benchmark on your own cameras"
            )),
        }
    }
}

/// The camera day of the tags of an image (`camera:<c>`, `date:<d>`) as `<camera>/<date>`: its
/// cross-validation group. None for an image without both (built-in sets: folded by image).
pub fn group_of(tags: &[String]) -> Option<String> {
    let find = |p: &str| tags.iter().find_map(|t| t.strip_prefix(p));
    Some(format!("{}/{}", find("camera:")?, find("date:")?))
}

/// The camera days `heldout-days.json` of `dir` lists; a list that does not read is a warning and
/// none (the split tags then follow the hash split alone; nothing is moved aside or written).
fn listed_days(dir: &Path, warnings: &mut Vec<String>) -> BTreeSet<Group> {
    match crate::collect::labels::heldout::load(dir) {
        Ok(f) => f.days.into_iter().map(|d| (d.camera, d.date)).collect(),
        Err(e) => {
            warnings.push(format!(
                "{e:#}; the split: tags of {TITLE} follow the hash split alone"
            ));
            BTreeSet::new()
        }
    }
}

/// The export's split of `frames`, read-only: the default split plus the listed held-out days
/// ([`crate::collect::labels::heldout::decide`] without growing the list).
fn apply_split(dir: &Path, frames: &mut [ExportFrame], warnings: &mut Vec<String>) {
    let groups: BTreeSet<Group> = frames
        .iter()
        .map(|f| (f.rel.camera.clone(), f.rel.date.clone()))
        .collect();
    let listed = listed_days(dir, warnings);
    let splits =
        crate::collect::labels::heldout::decide(&groups, &SplitConfig::default(), &listed, None)
            .splits;
    for f in frames {
        f.split = splits
            .get(&(f.rel.camera.clone(), f.rel.date.clone()))
            .copied()
            .unwrap_or(Split::Train);
    }
}

/// The selected frames of `src`, with the export's split and size floor, and the selection's
/// revisions and warnings; Err(why) when the folder cannot be read.
pub(crate) fn select(src: &Source) -> Result<(Vec<ExportFrame>, export::Selection), String> {
    if let Some(why) = &src.problem {
        return Err(why.clone());
    }
    let mut sel = export::reviewed_snapshot(
        &src.dir,
        &src.classes,
        &SplitConfig::default(),
        &ClassFilter::default(),
    );
    let mut frames = std::mem::take(&mut sel.frames);
    apply_split(&src.dir, &mut frames, &mut sel.warnings);
    for f in &mut frames {
        f.floor = src.size_floor;
    }
    Ok((frames, sel))
}

/// The set of `frames` (the export's benchmark manifest, every split).
fn set_of(src: &Source, frames: &[ExportFrame]) -> ImageSet {
    let refs: Vec<&ExportFrame> = frames.iter().collect();
    let manifest = export::benchmark_manifest(&refs, &src.classes, ID);
    let images = manifest
        .images
        .iter()
        .zip(frames)
        .map(|(i, f)| super::images::manifest_image(i, f.image.clone()))
        .collect();
    ImageSet {
        id: ID.to_string(),
        title: TITLE.to_string(),
        kind: SetKind::Dir,
        dir: Some(src.dir.clone()),
        license: manifest.license,
        attribution: String::new(),
        source: String::new(),
        ground_truth: GroundTruth::Manifest,
        scored_labels: manifest.scored_labels,
        transform: None,
        images,
    }
}

/// Load the set from `src` (see the module docs). Reads only.
pub fn load(src: &Source) -> Load {
    let (frames, sel) = match select(src) {
        Ok(v) => v,
        Err(why) => return Load::Missing(why),
    };
    let mut warnings = sel.warnings;
    if sel.changed > 0 {
        warnings.push(format!(
            "{} of {TITLE}'s reviewed frames changed while they were read (saved during the \
             walk) and were left out this time",
            sel.changed
        ));
    }
    if frames.is_empty() {
        return Load::Empty(warnings);
    }
    let set = set_of(src, &frames);
    let mut survey = Survey::of(&frames, &set);
    survey.changed = sel.changed;
    survey.lost = sel.lost_frames;
    Load::Ready(Box::new(Loaded {
        set,
        reviewed: sel.revisions,
        survey,
        warnings,
    }))
}

/// `set` (a run's copy of the cameras set) without the frames `seen` says the model trained or
/// validated on, and what was left out; `set` itself when nothing is.
pub fn for_model(set: &ImageSet, seen: &seen::Seen) -> (ImageSet, Option<super::report::LeftOut>) {
    let mut out = set.clone();
    let (mut trained, mut validated) = (0usize, 0usize);
    out.images.retain(|i| match seen.saw(&i.file) {
        Some(seen::Saw::Trained) => {
            trained += 1;
            false
        }
        Some(seen::Saw::Validated) => {
            validated += 1;
            false
        }
        None => true,
    });
    let left = match seen {
        seen::Seen::NoRecord => None,
        seen::Seen::Unknown(why) => Some(super::report::LeftOut {
            set: ID.to_string(),
            frames: 0,
            trained: 0,
            validated: 0,
            unknown: Some(why.clone()),
        }),
        seen::Seen::Frames { .. } => Some(super::report::LeftOut {
            set: ID.to_string(),
            frames: trained + validated,
            trained,
            validated,
            unknown: None,
        }),
    };
    (out, left)
}

/// The sets a run sends the config model `m` through: `sets` with "Your cameras" left without the
/// frames `m` trained or validated on ([`for_model`]), and what was left out; (None, None) when
/// `sets` holds no cameras set (the run uses `sets` as they are).
pub fn sets_for_model(
    config: &Config,
    m: &crate::config::ModelConfig,
    sets: &[ImageSet],
) -> (Option<Vec<ImageSet>>, Option<super::report::LeftOut>) {
    let Some(cams) = sets.iter().find(|s| s.id == ID) else {
        return (None, None);
    };
    let seen = seen::of_model(config, m, cams.images.iter().map(|i| i.file.as_str()));
    if seen == seen::Seen::NoRecord {
        return (None, None);
    }
    let (own, left) = for_model(cams, &seen);
    // A set left without a frame is out of the model's run (an empty list: nothing to run).
    let out = sets
        .iter()
        .filter_map(|s| {
            if s.id != ID {
                Some(s.clone())
            } else {
                (!own.images.is_empty()).then(|| own.clone())
            }
        })
        .collect();
    (Some(out), left)
}

/// The failure of a model whose run has no image left once "Your cameras" left out what it saw.
pub fn nothing_left(left: Option<&super::report::LeftOut>) -> String {
    format!(
        "every frame of {TITLE} in this run is one it trained or validated on ({}), and no other dataset was chosen: nothing is left to score it on fairly",
        left.map_or_else(String::new, super::report::LeftOut::why)
    )
}

/// Write the ground-truth snapshot of a run's `sets` beside `results_path` when they hold "Your
/// cameras" (`loaded`: what the run's resolution kept of it). Called before the run's first save of
/// `benchmark.json`, whose copy of the set has no labels.
pub fn write_snapshot(
    results_path: &Path,
    sets: &[ImageSet],
    loaded: Option<&super::images::CamerasLoaded>,
) -> anyhow::Result<()> {
    let (Some(set), Some(l)) = (sets.iter().find(|s| s.id == ID), loaded) else {
        return Ok(());
    };
    snapshot::GtSnapshot::of(&l.source, set, &l.reviewed).save(&snapshot::path_beside(results_path))
}
