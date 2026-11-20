//! Measured recall of the newest trained models on their held-out frames, as one pure table the
//! Review queue can steer by ([`Measured`]; built from `runs.json` by
//! [`crate::training::bench::measured_recall`], cached by the server's `review::steer`).
//!
//! Cells. Every cell holds integer counts: `gt` held-out objects of a class and `missed` of
//! them (not found at the benchmark's threshold, IoU 0.5), so sums are exact in any order. A
//! [`Slice`] is one run's cells over a set of held-out frames: per class, and per camera and
//! class. [`Measured::scenes`] holds one slice per scene (`day` / `night` / `ir`; the frames of
//! that scene, with per-camera cells from `heldout:camscene:<scene>:<camera>`), each from the
//! newest run that trained on the scene; [`Measured::overall`] one over every held-out frame
//! (`heldout:all`, per-camera cells from `heldout:cam:<camera>`).
//!
//! Classes are stored in the benchmark's canonical spelling
//! ([`crate::benchmark::metrics::canonical_label`]: lowercase, `_` as a space, VOC spellings as
//! COCO's) and looked up the same way, else through [`crate::model::coco::same_class`] (a review
//! class `motorbike` finds the model's `motorcycle`).
//!
//! Shrinkage. A cell of 3 objects says little, so a miss rate is shrunk toward its parent:
//! `m = (missed + PRIOR_GT * m_parent) / (gt + PRIOR_GT)`, an empirical-Bayes mean with a prior
//! worth [`PRIOR_GT`] objects. The chain ([`Measured::miss`]): the overall miss rate (every class
//! on every held-out frame), the scene's (every class in the scene), the class in the scene, then
//! the camera's class in the scene. A cell with no objects is its parent's rate; one with many
//! moves toward its own. [`PRIOR_GT`] is the benchmark's [`MIN_CLASS_GT`] (20): with fewer
//! objects one object moves a class by more than 5 points, which is where the benchmark stops
//! averaging a class and where a cell starts to carry half of its own weight here.

use crate::benchmark::metrics::{ClassRecall, MIN_CLASS_GT, canonical_label};
use crate::model::coco::same_class;
use std::collections::BTreeMap;

/// Objects a shrinkage prior is worth, and the "well supported" rule: a class measured on fewer
/// held-out objects of a camera is not trusted on its own ([`CameraRecall::well_supported`]).
pub const PRIOR_GT: usize = MIN_CLASS_GT;

/// The scenes a [`Measured`] may hold a slice for.
pub const SCENES: [&str; 3] = ["day", "night", "ir"];

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

/// The cell of `class` in `classes` (zero when it has none).
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

/// One run's cells over a set of held-out frames (a scene's, or all of them).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Slice {
    /// The run the cells came from (`RunRecord::id`).
    pub run: String,
    /// Canonical class -> cell.
    pub classes: BTreeMap<String, Cell>,
    /// Camera -> canonical class -> cell, on the same frames.
    pub cameras: BTreeMap<String, BTreeMap<String, Cell>>,
}

impl Slice {
    /// The slice of run `run`: its per-class cells and per-camera per-class cells (cells without
    /// objects are dropped).
    pub fn new<'a>(
        run: &str,
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
            run: run.to_string(),
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

    /// The cell of `class` (zero without objects).
    pub fn class(&self, class: &str) -> Cell {
        lookup(&self.classes, class)
    }

    /// The cell of `class` on `camera` (zero without objects).
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
    /// The cells over every held-out frame: of the newest run that trained on every scene, else
    /// of the newest run with any.
    pub overall: Option<Slice>,
}

/// A shrunk miss rate of a camera's class in a scene ([`Measured::miss`]), with the raw counts it
/// came from so a page can explain it.
#[derive(Debug, Clone, PartialEq)]
pub struct Miss {
    /// The shrunk miss rate, 0..=1.
    pub rate: f64,
    /// The camera's class in the scene (raw; zero without held-out objects).
    pub cell: Cell,
    /// The class in the scene, every camera (raw).
    pub class_cell: Cell,
    /// Every class in the scene (raw).
    pub scene_cell: Cell,
    /// The run the scene's cells came from.
    pub run: String,
    /// The scene had its own slice; false: no run that trained on the scene measured it, the
    /// overall slice stood in (every held-out frame, per-camera cells over every scene).
    pub scene_measured: bool,
}

/// A camera's shrunk recall of one class over every held-out scene
/// ([`Measured::class_recall_on_camera`]).
#[derive(Debug, Clone, PartialEq)]
pub struct CameraRecall {
    /// `1 - ` the shrunk miss rate, 0..=1.
    pub recall: f64,
    /// The raw cell: the camera's held-out objects of the class and how many were missed.
    pub cell: Cell,
    /// The run it came from.
    pub run: String,
}

impl CameraRecall {
    /// Measured on at least [`PRIOR_GT`] held-out objects of the camera: enough to rank the
    /// camera's classes by it.
    pub fn well_supported(&self) -> bool {
        self.cell.gt >= PRIOR_GT
    }
}

impl Measured {
    /// Nothing measured.
    pub fn is_empty(&self) -> bool {
        self.scenes.values().all(Slice::is_empty)
            && self.overall.as_ref().is_none_or(Slice::is_empty)
    }

    /// The root of the shrinkage chain: the overall miss rate (every class on every held-out
    /// frame), else that of the scene slices pooled. None when nothing was measured.
    pub fn overall_rate(&self) -> Option<f64> {
        if let Some(r) = self.overall.as_ref().and_then(|o| o.total().rate()) {
            return Some(r);
        }
        let mut t = Cell::default();
        for s in self.scenes.values() {
            t.add(s.total());
        }
        t.rate()
    }

    /// The shrunk miss rate of `class` on `camera` in `scene` (see the module docs): the camera's
    /// cell in the scene, shrunk toward the class in the scene, shrunk toward the scene, shrunk
    /// toward the overall rate. A scene without its own slice (no run that trained on it
    /// measured it, or a frame without a scene) uses the overall slice: the class over every
    /// held-out frame, then the camera's class over every scene. A class or camera without
    /// held-out objects gets its parent's rate (raw cells zero). None when nothing was measured.
    pub fn miss(&self, camera: &str, scene: &str, class: &str) -> Option<Miss> {
        let root = self.overall_rate()?;
        let (slice, parent, scene_measured) = match self.scenes.get(scene) {
            Some(s) if !s.is_empty() => (s, s.total().shrunk(root), true),
            _ => (self.overall.as_ref()?, root, false),
        };
        let class_cell = slice.class(class);
        let cell = slice.camera_class(camera, class);
        Some(Miss {
            rate: cell.shrunk(class_cell.shrunk(parent)),
            cell,
            class_cell,
            scene_cell: slice.total(),
            run: slice.run.clone(),
            scene_measured,
        })
    }

    /// The shrunk recall of `class` on `camera` over every held-out scene (the overall slice):
    /// the camera's cell shrunk toward the class's, shrunk toward the overall rate. None when the
    /// camera has no held-out object of the class (or nothing was measured); check
    /// [`CameraRecall::well_supported`] before ranking by it.
    pub fn class_recall_on_camera(&self, camera: &str, class: &str) -> Option<CameraRecall> {
        let o = self.overall.as_ref()?;
        let cell = o.camera_class(camera, class);
        if cell.gt == 0 {
            return None;
        }
        let root = o.total().rate()?;
        Some(CameraRecall {
            recall: 1.0 - cell.shrunk(o.class(class).shrunk(root)),
            cell,
            run: o.run.clone(),
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

    /// Night: person 40 (4 missed), raccoon 10 (6 missed); camera `back` holds 5 raccoons (4
    /// missed). Overall: person 100 (8), raccoon 20 (8); `back` 10 raccoons (6).
    fn table() -> Measured {
        let night = Slice::new(
            "run-night",
            &[cr("person", 40, 4), cr("Raccoon", 10, 6)],
            [("back", &[cr("raccoon", 5, 4)][..])],
        );
        let overall = Slice::new(
            "run-all",
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
        close(x.rate, cam);
        assert_eq!(x.cell, Cell { gt: 5, missed: 4 });
        assert_eq!(x.class_cell, Cell { gt: 10, missed: 6 });
        assert_eq!(x.scene_cell, Cell { gt: 50, missed: 10 });
        assert_eq!(x.run, "run-night");
        assert!(x.scene_measured);
        // A camera without held-out raccoons: the class's rate in the scene.
        let y = m.miss("front", "night", "RACCOON").unwrap();
        close(y.rate, class);
        assert_eq!(y.cell, Cell::default());
        // A class never measured: the scene's rate.
        close(m.miss("back", "night", "fox").unwrap().rate, scene);
    }

    #[test]
    fn a_scene_without_its_slice_uses_the_overall_one() {
        let m = table();
        let p = PRIOR_GT as f64;
        let root = 16.0 / 120.0;
        let class = (8.0 + p * root) / (20.0 + p);
        let cam = (6.0 + p * class) / (10.0 + p);
        let x = m.miss("back", "day", "raccoon").unwrap();
        close(x.rate, cam);
        assert!(!x.scene_measured);
        assert_eq!(x.run, "run-all");
        // A camera name with a ':' is just a name.
        let s = m.miss("side:2", "", "person").unwrap();
        assert_eq!(s.cell, Cell { gt: 30, missed: 3 });
    }

    #[test]
    fn class_recall_on_a_camera() {
        let m = table();
        let p = PRIOR_GT as f64;
        let root = 16.0 / 120.0;
        let class = (8.0 + p * root) / (20.0 + p);
        let r = m.class_recall_on_camera("back", "raccoon").unwrap();
        close(r.recall, 1.0 - (6.0 + p * class) / (10.0 + p));
        assert_eq!(r.cell, Cell { gt: 10, missed: 6 });
        assert_eq!(r.run, "run-all");
        assert!(!r.well_supported(), "10 objects < PRIOR_GT");
        let s = m.class_recall_on_camera("side:2", "person").unwrap();
        assert!(s.well_supported());
        assert!(m.class_recall_on_camera("back", "person").is_none());
        assert!(m.class_recall_on_camera("nowhere", "raccoon").is_none());
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
                Slice::new("r", &[cr("person", 0, 0)], []),
            )]
            .into_iter()
            .collect(),
            overall: None,
        };
        assert!(empty.is_empty());
        assert!(empty.miss("a", "day", "person").is_none());
        // Scene slices alone pool into the root.
        let scenes_only = Measured {
            scenes: [(
                "ir".to_string(),
                Slice::new("r", &[cr("person", 10, 5)], []),
            )]
            .into_iter()
            .collect(),
            overall: None,
        };
        close(scenes_only.overall_rate().unwrap(), 0.5);
        let x = scenes_only.miss("a", "ir", "person").unwrap();
        close(x.rate, 0.5);
        // No overall slice: an unmeasured scene has nothing to stand in.
        assert!(scenes_only.miss("a", "day", "person").is_none());
    }

    #[test]
    fn classes_match_across_spellings() {
        let s = Slice::new(
            "r",
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
    }

    /// Integer cells: the same totals in any order.
    #[test]
    fn cells_add_in_any_order() {
        let cells = [cr("a", 7, 3), cr("b", 11, 0), cr("a", 2, 2), cr("c", 5, 1)];
        let mut rev = cells.clone();
        rev.reverse();
        assert_eq!(
            Slice::new("r", &cells, []).classes,
            Slice::new("r", &rev, []).classes
        );
        assert_eq!(
            Slice::new("r", &cells, []).class("a"),
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
