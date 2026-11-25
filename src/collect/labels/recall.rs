//! Measured recall of the newest trained models on their held-out frames, as one pure table the
//! Review queue can steer by ([`Measured`]; built from `runs.json` by
//! [`crate::training::bench::measured_recall`], cached by the server's `review::steer`).
//!
//! Cells. Every cell holds integer counts: `gt` held-out objects of a class and `missed` of
//! them (not found at the benchmark's threshold, IoU 0.5), so sums are exact in any order. A
//! [`Slice`] is one run's cells over a set of held-out frames: per class, and per camera and
//! class, with the run itself ([`Run`]: its id, its model's classes and its renames).
//! [`Measured::scenes`] holds one slice per scene (`day` / `night` / `ir`; the frames of that
//! scene, with per-camera cells from `heldout:camscene:<scene>:<camera>`), each from the newest
//! run that trained on the scene; [`Measured::overall`] one over every held-out frame
//! (`heldout:all`, per-camera cells from `heldout:cam:<camera>`), only from a run that trained on
//! every scene (a scene-filtered model never trained on some of those frames).
//!
//! Classes. A review class is first mapped through the run's renames (`fox -> animal`: the model
//! learnt foxes as `animal`; renamed to nothing: [`InRun::Dropped`]), then found among the run's
//! model classes ([`InRun::Absent`] when it is not one: the model cannot detect it at all, which
//! is no miss rate but the most a class can need). Names compare in the benchmark's canonical
//! spelling ([`crate::benchmark::metrics::canonical_label`]: lowercase, `_` as a space, VOC
//! spellings as COCO's), else through [`crate::model::coco::same_class`] (a review class
//! `motorbike` finds the model's `motorcycle`). Cells are stored in that canonical spelling.
//!
//! Shrinkage. A cell of 3 objects says little, so a miss rate is shrunk toward its parent:
//! `m = (missed + PRIOR_GT * m_parent) / (gt + PRIOR_GT)`, an empirical-Bayes mean with a prior
//! worth [`PRIOR_GT`] objects. The chain ([`Measured::miss`]): the overall miss rate (every class
//! on every held-out frame; without an overall slice the scene's own), the scene's (every class
//! in the scene), the class in the scene, then the camera's class in the scene. A cell of a class
//! of the model with no objects is its parent's rate; one with many moves toward its own.
//! [`PRIOR_GT`] is the benchmark's [`MIN_CLASS_GT`] (20): with fewer objects one object moves a
//! class by more than 5 points, which is where the benchmark stops averaging a class and where a
//! cell starts to carry half of its own weight here.

use crate::benchmark::metrics::{ClassRecall, MIN_CLASS_GT, canonical_label};
use crate::model::coco::same_class;
use std::collections::BTreeMap;

/// Objects a shrinkage prior is worth, and the "well supported" rule: a class measured on fewer
/// held-out objects of a camera is not trusted on its own ([`CameraRecall::well_supported`]).
pub const PRIOR_GT: usize = MIN_CLASS_GT;

/// The scenes a [`Measured`] may hold a slice for: the dataset's scene folders.
pub use crate::collect::layout::SCENES;

/// Held-out objects of one class (or of every class) and how many of them were missed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cell {
    pub gt: usize,
    pub missed: usize,
}

impl Cell {
    /// The cell of a benchmark recall cell (`missed` clamped to `gt`).
    pub fn of(c: &ClassRecall) -> Self {
        Self {
            gt: c.gt,
            missed: c.missed.min(c.gt),
        }
    }

    pub fn add(&mut self, o: Cell) {
        self.gt += o.gt;
        self.missed += o.missed;
    }

    /// `missed / gt`; None without objects.
    pub fn rate(self) -> Option<f64> {
        (self.gt > 0).then(|| self.missed as f64 / self.gt as f64)
    }

    /// The miss rate shrunk toward `parent` ([`PRIOR_GT`], see the module docs): `parent`
    /// itself without objects.
    pub fn shrunk(self, parent: f64) -> f64 {
        (self.missed as f64 + PRIOR_GT as f64 * parent) / (self.gt + PRIOR_GT) as f64
    }
}

/// The canonical spelling a class is stored and looked up in.
fn canon(class: &str) -> String {
    canonical_label(class)
}

/// One class under two spellings (see the module docs).
fn same(a: &str, b: &str) -> bool {
    canon(a) == canon(b) || same_class(a, b)
}

/// The cell of model class `class` in `classes` (zero when it has none).
fn lookup(classes: &BTreeMap<String, Cell>, class: &str) -> Cell {
    classes
        .get(&canon(class))
        .or_else(|| {
            classes
                .iter()
                .find(|(k, _)| same_class(k, class))
                .map(|(_, c)| c)
        })
        .copied()
        .unwrap_or_default()
}

/// Cells keyed by canonical class; cells of one class under two spellings add up.
fn class_map(cells: &[ClassRecall]) -> BTreeMap<String, Cell> {
    let mut m: BTreeMap<String, Cell> = BTreeMap::new();
    for c in cells.iter().filter(|c| c.gt > 0) {
        m.entry(canon(&c.class)).or_default().add(Cell::of(c));
    }
    m
}

/// The run a slice came from: what its model can detect.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Run {
    /// `RunRecord::id`.
    pub id: String,
    /// The model's classes (`TrainOutput::classes`, after the renames).
    pub classes: Vec<String>,
    /// The run's renames (`TrainRequest::rename`): review label -> the model's class; an empty
    /// target dropped the label.
    pub rename: BTreeMap<String, String>,
}

/// What a review class is to a run's model ([`Run::resolve`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InRun {
    /// A class of the model, `class` in the model's spelling; `renamed`: the run's renames made
    /// the review class into it (`fox` learnt as `animal`).
    Class { class: String, renamed: bool },
    /// The run renamed the review class to nothing: the model never learnt it.
    Dropped,
    /// Not a class of the model: it cannot detect it at all.
    Absent,
}

impl InRun {
    /// The model's class, when it has one.
    pub fn class(&self) -> Option<&str> {
        match self {
            InRun::Class { class, .. } => Some(class),
            _ => None,
        }
    }
}

impl Run {
    /// What review class `class` is to this run's model: through the renames, then among its
    /// classes (see the module docs).
    pub fn resolve(&self, class: &str) -> InRun {
        let (name, renamed) = match self.rename.iter().find(|(from, _)| same(from, class)) {
            Some((_, to)) if to.trim().is_empty() => return InRun::Dropped,
            Some((_, to)) => (to.trim(), true),
            None => (class, false),
        };
        match self.classes.iter().find(|c| same(c, name)) {
            Some(c) => InRun::Class {
                class: c.clone(),
                renamed,
            },
            None => InRun::Absent,
        }
    }
}

/// One run's cells over a set of held-out frames (a scene's, or all of them).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Slice {
    /// The run the cells came from.
    pub run: Run,
    /// Canonical class -> cell.
    pub classes: BTreeMap<String, Cell>,
    /// Camera -> canonical class -> cell, on the same frames.
    pub cameras: BTreeMap<String, BTreeMap<String, Cell>>,
}

impl Slice {
    /// The slice of `run`: its per-class cells and per-camera per-class cells (cells without
    /// objects are dropped).
    pub fn new<'a>(
        run: Run,
        classes: &[ClassRecall],
        cameras: impl IntoIterator<Item = (&'a str, &'a [ClassRecall])>,
    ) -> Self {
        let mut cams: BTreeMap<String, BTreeMap<String, Cell>> = BTreeMap::new();
        for (camera, cells) in cameras {
            let m = class_map(cells);
            if !m.is_empty() {
                let e = cams.entry(camera.to_string()).or_default();
                for (k, c) in m {
                    e.entry(k).or_default().add(c);
                }
            }
        }
        Self {
            run,
            classes: class_map(classes),
            cameras: cams,
        }
    }

    /// Every class together.
    pub fn total(&self) -> Cell {
        let mut t = Cell::default();
        for c in self.classes.values() {
            t.add(*c);
        }
        t
    }

    /// The cell of the model's class `class` (zero without objects; no rename applied: see
    /// [`Run::resolve`]).
    pub fn class(&self, class: &str) -> Cell {
        lookup(&self.classes, class)
    }

    /// The cell of the model's class `class` on `camera` (zero without objects).
    pub fn camera_class(&self, camera: &str, class: &str) -> Cell {
        self.cameras
            .get(camera)
            .map(|m| lookup(m, class))
            .unwrap_or_default()
    }

    fn is_empty(&self) -> bool {
        self.total().gt == 0
    }
}

/// The measured recall the Review queue steers by (see the module docs).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Measured {
    /// Scene (`day` / `night` / `ir`) -> the cells of the newest run that trained on it.
    pub scenes: BTreeMap<String, Slice>,
    /// The cells over every held-out frame, of the newest run that trained on every scene; None
    /// without such a run.
    pub overall: Option<Slice>,
}

/// A shrunk miss rate of a camera's class in a scene ([`Measured::miss`]), with the raw counts it
/// came from so a page can explain it.
#[derive(Debug, Clone, PartialEq)]
pub struct Miss {
    /// What the review class is to the scene's model. Not [`InRun::Class`]: `rate` is None and
    /// the class cells are zero (the model cannot find it: no rate, the most need).
    pub class: InRun,
    /// The shrunk miss rate, 0..=1; None when the model has no such class.
    pub rate: Option<f64>,
    /// The camera's class in the scene (raw; zero without held-out objects).
    pub cell: Cell,
    /// The class in the scene, every camera (raw).
    pub class_cell: Cell,
    /// Every class in the scene (raw).
    pub scene_cell: Cell,
    /// The run the scene's cells came from.
    pub run: String,
}

/// A camera's shrunk recall of one class over every held-out scene
/// ([`Measured::class_recall_on_camera`]).
#[derive(Debug, Clone, PartialEq)]
pub struct CameraRecall {
    /// What the review class is to the overall slice's model (as [`Miss::class`]).
    pub class: InRun,
    /// `1 - ` the shrunk miss rate, 0..=1; None when the model has no such class.
    pub recall: Option<f64>,
    /// The raw cell: the camera's held-out objects of the class and how many were missed.
    pub cell: Cell,
    /// The run it came from.
    pub run: String,
}

impl CameraRecall {
    /// A class of the model measured on at least [`PRIOR_GT`] held-out objects of the camera:
    /// enough to rank the camera's classes by it.
    pub fn well_supported(&self) -> bool {
        self.recall.is_some() && self.cell.gt >= PRIOR_GT
    }
}

impl Measured {
    /// Nothing measured.
    pub fn is_empty(&self) -> bool {
        self.scenes.values().all(Slice::is_empty)
            && self.overall.as_ref().is_none_or(Slice::is_empty)
    }

    /// The overall miss rate (every class on every held-out frame of the overall slice); None
    /// without one.
    pub fn overall_rate(&self) -> Option<f64> {
        self.overall.as_ref().and_then(|o| o.total().rate())
    }

    /// The shrunk miss rate of review class `class` on `camera` in `scene` (see the module docs):
    /// the camera's cell in the scene, shrunk toward the class in the scene, toward the scene,
    /// toward the overall rate (the scene's own without an overall slice). A class of the model
    /// without held-out objects (or a camera without any of them) gets its parent's rate; a class
    /// the model does not have gets none ([`Miss::class`]). None when the scene has no slice of
    /// its own (no run that trained on it measured it, or a frame without a scene): another
    /// scene's model never stands in.
    pub fn miss(&self, camera: &str, scene: &str, class: &str) -> Option<Miss> {
        let slice = self.scenes.get(scene).filter(|s| !s.is_empty())?;
        let total = slice.total();
        let root = self.overall_rate().or_else(|| total.rate())?;
        let resolved = slice.run.resolve(class);
        let (rate, cell, class_cell) = match resolved.class() {
            Some(c) => {
                let class_cell = slice.class(c);
                let cell = slice.camera_class(camera, c);
                let rate = cell.shrunk(class_cell.shrunk(total.shrunk(root)));
                (Some(rate), cell, class_cell)
            }
            None => (None, Cell::default(), Cell::default()),
        };
        Some(Miss {
            class: resolved,
            rate,
            cell,
            class_cell,
            scene_cell: total,
            run: slice.run.id.clone(),
        })
    }

    /// The shrunk recall of review class `class` on `camera` over every held-out scene (the
    /// overall slice): the camera's cell shrunk toward the class's, toward the overall rate. A
    /// camera without held-out objects of the class gets the class's; a class the model does not
    /// have gets none ([`CameraRecall::class`]). None without an overall slice; check
    /// [`CameraRecall::well_supported`] before ranking by it.
    pub fn class_recall_on_camera(&self, camera: &str, class: &str) -> Option<CameraRecall> {
        let o = self.overall.as_ref()?;
        let root = o.total().rate()?;
        let resolved = o.run.resolve(class);
        let (recall, cell) = match resolved.class() {
            Some(c) => {
                let cell = o.camera_class(camera, c);
                (Some(1.0 - cell.shrunk(o.class(c).shrunk(root))), cell)
            }
            None => (None, Cell::default()),
        };
        Some(CameraRecall {
            class: resolved,
            recall,
            cell,
            run: o.run.id.clone(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn cr(class: &str, gt: usize, missed: usize) -> ClassRecall {
        ClassRecall::new(class, gt, missed)
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    /// A run `id` whose model has `classes`, renaming `rename` (from, to).
    fn run(id: &str, classes: &[&str], rename: &[(&str, &str)]) -> Run {
        Run {
            id: id.into(),
            classes: classes.iter().map(|c| c.to_string()).collect(),
            rename: rename
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        }
    }

    const MODEL: [&str; 3] = ["person", "raccoon", "skunk"];

    /// Night: person 40 (4 missed), raccoon 10 (6 missed); camera `back` holds 5 raccoons (4
    /// missed). Overall: person 100 (8), raccoon 20 (8); `back` 10 raccoons (6). Both models
    /// have person, raccoon and skunk (no held-out skunk).
    fn table() -> Measured {
        let night = Slice::new(
            run("run-night", &MODEL, &[]),
            &[cr("person", 40, 4), cr("Raccoon", 10, 6)],
            [("back", &[cr("raccoon", 5, 4)][..])],
        );
        let overall = Slice::new(
            run("run-all", &MODEL, &[]),
            &[cr("person", 100, 8), cr("raccoon", 20, 8)],
            [
                ("back", &[cr("raccoon", 10, 6)][..]),
                ("side:2", &[cr("person", 30, 3)][..]),
            ],
        );
        Measured {
            scenes: [("night".to_string(), night)].into_iter().collect(),
            overall: Some(overall),
        }
    }

    fn detected(c: &str) -> InRun {
        InRun::Class {
            class: c.into(),
            renamed: false,
        }
    }

    #[test]
    fn shrinking_follows_the_chain() {
        let m = table();
        let p = PRIOR_GT as f64;
        let root = 16.0 / 120.0;
        close(m.overall_rate().unwrap(), root);
        let scene = (10.0 + p * root) / (50.0 + p);
        let class = (6.0 + p * scene) / (10.0 + p);
        let cam = (4.0 + p * class) / (5.0 + p);
        let x = m.miss("back", "night", "raccoon").unwrap();
        close(x.rate.unwrap(), cam);
        assert_eq!(x.class, detected("raccoon"));
        assert_eq!(x.cell, Cell { gt: 5, missed: 4 });
        assert_eq!(x.class_cell, Cell { gt: 10, missed: 6 });
        assert_eq!(x.scene_cell, Cell { gt: 50, missed: 10 });
        assert_eq!(x.run, "run-night");
        // A camera without held-out raccoons: the class's rate in the scene.
        let y = m.miss("front", "night", "RACCOON").unwrap();
        close(y.rate.unwrap(), class);
        assert_eq!(y.cell, Cell::default());
        // A class of the model never seen held out: the scene's rate.
        let s = m.miss("back", "night", "skunk").unwrap();
        assert_eq!(s.class, detected("skunk"));
        close(s.rate.unwrap(), scene);
    }

    /// A class the model does not have is no rate (never the parent's), and says so.
    #[test]
    fn a_class_outside_the_model_has_no_rate() {
        let m = table();
        let x = m.miss("back", "night", "fox").unwrap();
        assert_eq!(x.class, InRun::Absent);
        assert_eq!(x.rate, None);
        assert_eq!((x.cell, x.class_cell), (Cell::default(), Cell::default()));
        assert_eq!(x.scene_cell, Cell { gt: 50, missed: 10 });
        let r = m.class_recall_on_camera("back", "fox").unwrap();
        assert_eq!((r.class.clone(), r.recall), (InRun::Absent, None));
        assert!(!r.well_supported());
    }

    /// Renames: a renamed class is looked up under its new name (and says so); one renamed to
    /// nothing is dropped, never given the parent's rate.
    #[test]
    fn renames_map_the_review_class() {
        let night = Slice::new(
            run(
                "merged",
                &["person", "animal"],
                &[("fox", "animal"), ("Raccoon", "animal"), ("skunk", " ")],
            ),
            &[cr("person", 40, 4), cr("animal", 12, 3)],
            [("back", &[cr("animal", 8, 2)][..])],
        );
        let m = Measured {
            scenes: [("night".to_string(), night)].into_iter().collect(),
            overall: None,
        };
        let fox = m.miss("back", "night", "fox").unwrap();
        assert_eq!(
            fox.class,
            InRun::Class {
                class: "animal".into(),
                renamed: true
            }
        );
        assert_eq!(fox.cell, Cell { gt: 8, missed: 2 });
        assert_eq!(fox.class_cell, Cell { gt: 12, missed: 3 });
        assert!(fox.rate.is_some());
        // The rename's key matches across spellings.
        assert_eq!(
            m.miss("back", "night", "raccoon").unwrap().class_cell,
            Cell { gt: 12, missed: 3 }
        );
        let skunk = m.miss("back", "night", "skunk").unwrap();
        assert_eq!((skunk.class, skunk.rate), (InRun::Dropped, None));
        // `animal` itself is a class of the model, not renamed.
        assert_eq!(
            m.miss("back", "night", "animal").unwrap().class,
            InRun::Class {
                class: "animal".into(),
                renamed: false
            }
        );
        // Without an overall slice the scene's own rate is the root.
        let p = PRIOR_GT as f64;
        let root = 7.0 / 52.0;
        let class = (3.0 + p * root) / (12.0 + p);
        close(
            m.miss("front", "night", "fox").unwrap().rate.unwrap(),
            class,
        );
        assert!(m.class_recall_on_camera("back", "fox").is_none());
    }

    /// A scene without its own slice is None: another scene's model never stands in.
    #[test]
    fn a_scene_without_its_slice_is_none() {
        let m = table();
        assert!(m.miss("back", "day", "raccoon").is_none());
        assert!(m.miss("back", "", "raccoon").is_none());
    }

    #[test]
    fn class_recall_on_a_camera() {
        let m = table();
        let p = PRIOR_GT as f64;
        let root = 16.0 / 120.0;
        let class = (8.0 + p * root) / (20.0 + p);
        let r = m.class_recall_on_camera("back", "raccoon").unwrap();
        close(r.recall.unwrap(), 1.0 - (6.0 + p * class) / (10.0 + p));
        assert_eq!(r.cell, Cell { gt: 10, missed: 6 });
        assert_eq!(r.run, "run-all");
        assert!(!r.well_supported(), "10 objects < PRIOR_GT");
        // A camera name with a ':' is just a name.
        let s = m.class_recall_on_camera("side:2", "person").unwrap();
        assert_eq!(s.cell, Cell { gt: 30, missed: 3 });
        assert!(s.well_supported());
        // No held-out person on `back`: the class's recall, not supported.
        let b = m.class_recall_on_camera("back", "person").unwrap();
        close(b.recall.unwrap(), 1.0 - (8.0 + p * root) / (100.0 + p));
        assert_eq!(b.cell, Cell::default());
        assert!(!b.well_supported());
    }

    #[test]
    fn nothing_measured_is_none() {
        let m = Measured::default();
        assert!(m.is_empty());
        assert!(m.miss("back", "night", "person").is_none());
        assert!(m.class_recall_on_camera("back", "person").is_none());
        // Cells without objects are no measurement either.
        let empty = Measured {
            scenes: [(
                "day".to_string(),
                Slice::new(run("r", &MODEL, &[]), &[cr("person", 0, 0)], []),
            )]
            .into_iter()
            .collect(),
            overall: None,
        };
        assert!(empty.is_empty());
        assert!(empty.miss("a", "day", "person").is_none());
    }

    #[test]
    fn classes_match_across_spellings() {
        let s = Slice::new(
            run("r", &["motorcycle", "person"], &[]),
            &[
                cr("motorbike", 4, 1),
                cr("motorcycle", 6, 1),
                cr("Pedestrian", 3, 0),
            ],
            [],
        );
        // VOC spellings are stored as COCO's and add up.
        assert_eq!(s.class("motorcycle"), Cell { gt: 10, missed: 2 });
        assert_eq!(s.class("Motorbike"), Cell { gt: 10, missed: 2 });
        assert_eq!(s.class("person"), Cell { gt: 3, missed: 0 });
        assert_eq!(s.class("people"), Cell { gt: 3, missed: 0 });
        assert_eq!(s.class("fox"), Cell::default());
        assert_eq!(s.total(), Cell { gt: 13, missed: 2 });
        assert_eq!(s.run.resolve("motorbike"), detected("motorcycle"));
        assert_eq!(s.run.resolve("people"), detected("person"));
        assert_eq!(s.run.resolve("fox"), InRun::Absent);
    }

    /// Integer cells: the same totals in any order.
    #[test]
    fn cells_add_in_any_order() {
        let cells = [cr("a", 7, 3), cr("b", 11, 0), cr("a", 2, 2), cr("c", 5, 1)];
        let mut rev = cells.clone();
        rev.reverse();
        let r = || run("r", &["a", "b", "c"], &[]);
        assert_eq!(
            Slice::new(r(), &cells, []).classes,
            Slice::new(r(), &rev, []).classes
        );
        assert_eq!(
            Slice::new(r(), &cells, []).class("a"),
            Cell { gt: 9, missed: 5 }
        );
        close(Cell { gt: 0, missed: 0 }.shrunk(0.3), 0.3);
        assert_eq!(
            Cell::of(&ClassRecall {
                missed: 9,
                ..cr("x", 4, 0)
            })
            .missed,
            4
        );
    }
}
