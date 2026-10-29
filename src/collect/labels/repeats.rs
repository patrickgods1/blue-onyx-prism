//! Repeat detections of fixed cameras: the same false positive (a railing, a plant) or the same
//! parked car at one spot, frame after frame. Pure parts: which boxes a spot matches
//! ([`Spot::matches`]: a pending model / auto box of the same object class, [`same_object`], at
//! IoU >= [`Spot::min_iou`]) and whether a frame holds only such decided repeats
//! ([`repeats_only`]). IO: the per-camera history of the reviewer's decisions ([`history`], the
//! Review page's "rejected here before" hint; cached per folder and camera, [`invalidate`] after
//! every write of the folder's label files). The bulk edits are [`super::bulk::bulk_spot`] and
//! [`super::bulk::bulk_accept_repeats`].

use super::{BoxState, LabelBox, LabelFile, RejectReason, Source, Status};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// IoU at or above which "reject this spot everywhere" takes a box.
pub const REJECT_SPOT_IOU: f32 = 0.6;
/// IoU at or above which "accept this spot everywhere" takes a box (stricter: accepting a
/// different object is worse than leaving it pending).
pub const ACCEPT_SPOT_IOU: f32 = 0.8;
/// The provenance (`reason`) a spot edit gives a box that had none.
pub const REPEAT_SPOT_REASON: &str = "repeat_spot";
/// Most boxes [`history`] returns per camera (newest frames first).
pub const HISTORY_CAP: usize = 2000;
/// How long a cached history is used without a write of the folder (label files written by
/// other processes, such as an auto-label run, show up after this).
pub const HISTORY_TTL: Duration = Duration::from_secs(60);

/// Vehicle classes: a box of one may stand for an object another names (the Review page's
/// `VEHICLES`, `assets/review.js`).
pub const VEHICLES: [&str; 4] = ["car", "truck", "bus", "van"];

/// Two labels that can name one object: the same class, or two vehicle classes.
pub fn same_object(a: &str, b: &str) -> bool {
    a == b || (VEHICLES.contains(&a) && VEHICLES.contains(&b))
}

/// What a spot edit does to the boxes it matches.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Rejected, with the reviewer's reason if any.
    Reject(Option<RejectReason>),
    /// Accepted (the same label only, at [`ACCEPT_SPOT_IOU`]).
    Accept,
}

/// A box of one camera's frames to find again on its other frames.
#[derive(Debug, Clone, PartialEq)]
pub struct Spot {
    pub camera: String,
    pub label: String,
    /// `[x0, y0, x1, y1]` in pixels of a `width` x `height` frame.
    pub bbox: [f32; 4],
    pub width: u32,
    pub height: u32,
    pub min_iou: f32,
}

impl Spot {
    /// Whether `b` is a box this spot takes for `verdict`: a pending model or auto box at IoU >=
    /// `min_iou`, of the same object class (a reject) or the same label (an accept).
    pub fn matches(&self, b: &LabelBox, verdict: &Verdict) -> bool {
        let label_ok = match verdict {
            Verdict::Reject(_) => same_object(&b.label, &self.label),
            Verdict::Accept => b.label == self.label,
        };
        b.source != Source::Human
            && b.state == BoxState::Pending
            && label_ok
            && super::iou(&b.bbox, &self.bbox) >= self.min_iou
    }

    /// Boxes of `f` [`Self::matches`] takes; only on an unreviewed frame.
    pub fn count(&self, f: &LabelFile, verdict: &Verdict) -> usize {
        if f.status != Status::Unreviewed {
            return 0;
        }
        f.boxes.iter().filter(|b| self.matches(b, verdict)).count()
    }

    /// Apply `verdict` to the boxes of `f` [`Self::matches`] takes, mark the frame
    /// ([`LabelFile::repeat_spot`]) and return how many; only on an unreviewed frame. A box keeps
    /// its own provenance (`reason`) and gets [`REPEAT_SPOT_REASON`] when it had none.
    pub fn apply(&self, f: &mut LabelFile, verdict: &Verdict) -> usize {
        if f.status != Status::Unreviewed {
            return 0;
        }
        let mut n = 0;
        for b in f.boxes.iter_mut().filter(|b| self.matches(b, verdict)) {
            match verdict {
                Verdict::Reject(why) => {
                    b.state = BoxState::Rejected;
                    b.difficult = false;
                    b.reject_reason = why.clone();
                }
                Verdict::Accept => b.state = BoxState::Accepted,
            }
            if b.reason.is_none() {
                b.reason = Some(REPEAT_SPOT_REASON.into());
            }
            n += 1;
        }
        if n > 0 {
            f.repeat_spot = true;
        }
        n
    }
}

/// A frame whose boxes a spot edit settled: unreviewed, touched by a spot edit
/// ([`LabelFile::repeat_spot`]), no pending box left. Sorted last in the queue; the "accept
/// repeat-only frames" bulk save takes these only.
pub fn repeats_only(f: &LabelFile) -> bool {
    f.status == Status::Unreviewed
        && f.repeat_spot
        && !f.boxes.iter().any(|b| b.state == BoxState::Pending)
}

/// One earlier decision at a camera ([`history`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Decision {
    /// The frame (`<scene>/<camera>/<date>/<stem>.jpg`): the page leaves out the frame it shows.
    pub frame: String,
    pub label: String,
    pub bbox: [f32; 4],
    /// `accepted` or `rejected`.
    pub state: BoxState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reject_reason: Option<RejectReason>,
    pub width: u32,
    pub height: u32,
}

/// The reviewer's decisions on one camera's frames ([`history`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct History {
    pub camera: String,
    pub decisions: Vec<Decision>,
    /// More decisions than [`HISTORY_CAP`] (the oldest frames are left out).
    pub truncated: bool,
}

/// Whether a decided box counts as the reviewer's own look at that spot: a model or auto box
/// accepted or rejected, not by a rule over many frames (the low-score and unwanted-class
/// rejects, an earlier spot edit; counting those would make a spot look more certain than any
/// look at it was).
fn counts(b: &LabelBox) -> bool {
    b.source != Source::Human
        && matches!(b.state, BoxState::Accepted | BoxState::Rejected)
        && !matches!(
            b.reason.as_deref(),
            Some(
                super::bulk::LOW_SCORE_REASON
                    | super::bulk::OTHER_CLASS_REASON
                    | REPEAT_SPOT_REASON
            )
        )
}

/// Scan `dir` for the decisions of `camera` ([`counts`]), newest frames first, at most `cap`.
/// A frame whose sidecar does not read (its size is unknown) is left out.
pub fn scan(dir: &Path, camera: &str, cap: usize) -> History {
    let mut files: Vec<(String, PathBuf)> = super::label_files(dir)
        .into_iter()
        .filter(|(rel, _)| rel.split('/').nth(1) == Some(camera))
        .collect();
    // `<scene>/<camera>/<date>/<stem>`: newest date first, then the newest stem (a UTC time).
    files.sort_by(|a, b| {
        let key = |r: &str| {
            let mut p = r.split('/').skip(2);
            (
                p.next().unwrap_or("").to_string(),
                p.next().unwrap_or("").to_string(),
            )
        };
        key(&b.0).cmp(&key(&a.0))
    });
    let mut h = History {
        camera: camera.to_string(),
        ..History::default()
    };
    'files: for (rel, path) in files {
        let Some(l) = super::read(&path).ok().flatten() else {
            continue;
        };
        if !l.boxes.iter().any(counts) {
            continue;
        }
        let Some(rec) = super::Frame::resolve(dir, &rel).and_then(|f| f.record().ok()) else {
            continue;
        };
        for b in l.boxes.iter().filter(|b| counts(b)) {
            if h.decisions.len() >= cap {
                h.truncated = true;
                break 'files;
            }
            h.decisions.push(Decision {
                frame: rel.clone(),
                label: b.label.clone(),
                bbox: b.bbox,
                state: b.state,
                reject_reason: b.reject_reason.clone(),
                width: rec.width,
                height: rec.height,
            });
        }
    }
    h
}

type Cache = HashMap<(PathBuf, String), (Instant, Arc<History>)>;

fn cache() -> &'static Mutex<Cache> {
    static CACHE: std::sync::OnceLock<Mutex<Cache>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Scans running ([`history`]): per scan id, its folder and whether a label file under it was
/// written meanwhile (then the scan is not cached).
fn scanning() -> &'static Mutex<HashMap<u64, (PathBuf, bool)>> {
    static SCANNING: std::sync::OnceLock<Mutex<HashMap<u64, (PathBuf, bool)>>> =
        std::sync::OnceLock::new();
    SCANNING.get_or_init(|| Mutex::new(HashMap::new()))
}

/// [`scan`] of `camera` under `dir`, cached until a label file under `dir` is written
/// ([`written`], [`invalidate`]) or for [`HISTORY_TTL`] (writes by other processes).
pub fn history(dir: &Path, camera: &str) -> Arc<History> {
    static IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    // Keyed by the canonical folder: the frames' paths ([`super::Frame::resolve`]) are canonical.
    let root = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let key = (root.clone(), camera.to_string());
    if let Ok(c) = cache().lock()
        && let Some((at, h)) = c.get(&key)
        && at.elapsed() < HISTORY_TTL
    {
        return h.clone();
    }
    let id = IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Ok(mut s) = scanning().lock() {
        s.insert(id, (root, false));
    }
    let h = Arc::new(scan(dir, camera, HISTORY_CAP));
    let overtaken = scanning()
        .lock()
        .ok()
        .and_then(|mut s| s.remove(&id))
        .is_none_or(|(_, dirty)| dirty);
    if !overtaken && let Ok(mut c) = cache().lock() {
        c.insert(key, (Instant::now(), h.clone()));
    }
    h
}

/// Drop the cached histories of `dir`.
pub fn invalidate(dir: &Path) {
    let root = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    if let Ok(mut c) = cache().lock() {
        c.retain(|(d, _), _| *d != root);
    }
}

/// A label file at `path` was written: drop the cached histories of the folders holding it, and
/// keep a scan of them running now from being cached (every label write goes through here,
/// [`super::write`] and friends).
pub fn written(path: &Path) {
    let canon = path.canonicalize().ok();
    let under = |d: &Path| path.starts_with(d) || canon.as_ref().is_some_and(|c| c.starts_with(d));
    if let Ok(mut s) = scanning().lock() {
        for (d, dirty) in s.values_mut() {
            *dirty |= under(d);
        }
    }
    if let Ok(mut c) = cache().lock()
        && !c.is_empty()
    {
        c.retain(|(d, _), _| !under(d));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::tests_support::auto_box;
    use super::*;

    fn spot(min_iou: f32) -> Spot {
        Spot {
            camera: "patio".into(),
            label: "car".into(),
            bbox: [10.0, 10.0, 50.0, 50.0], // auto_box's place
            width: 640,
            height: 480,
            min_iou,
        }
    }

    #[test]
    fn a_spot_takes_pending_model_and_auto_boxes_of_the_object_at_the_place() {
        let s = spot(REJECT_SPOT_IOU);
        let rej = Verdict::Reject(None);
        let mut b = auto_box("truck");
        assert!(s.matches(&b, &rej), "a vehicle pair for a reject");
        assert!(
            !s.matches(&b, &Verdict::Accept),
            "the same label only for an accept"
        );
        b.label = "dog".into();
        assert!(!s.matches(&b, &rej), "another object class");
        b.label = "car".into();
        b.bbox = [10.0, 10.0, 50.0, 33.0]; // IoU 0.575
        assert!(super::super::iou(&b.bbox, &s.bbox) < 0.6);
        assert!(!s.matches(&b, &rej), "IoU below the minimum");
        b.bbox = s.bbox;
        b.state = BoxState::Accepted;
        assert!(!s.matches(&b, &rej), "already decided");
        b.state = BoxState::Pending;
        b.source = Source::Human;
        assert!(!s.matches(&b, &rej), "a drawn box");
        b.source = Source::Model;
        assert!(s.matches(&b, &rej));
        // An accept needs 0.8: a box at ~0.7 is left alone.
        let a = spot(ACCEPT_SPOT_IOU);
        b.bbox = [10.0, 10.0, 50.0, 40.0]; // IoU 0.75
        assert!(!a.matches(&b, &Verdict::Accept));
        assert!(s.matches(&b, &rej));
    }

    #[test]
    fn apply_rejects_marks_the_frame_and_keeps_provenance() {
        let s = spot(REJECT_SPOT_IOU);
        let mut f = LabelFile::default();
        let mut model = auto_box("car");
        model.source = Source::Model;
        model.reason = Some("night_threshold".into());
        model.difficult = true;
        f.boxes = vec![auto_box("car"), model, auto_box("dog")];
        let why = Verdict::Reject(Some(RejectReason::StaticObject));
        assert_eq!(s.count(&f, &why), 2);
        assert!(!repeats_only(&f));
        assert_eq!(s.apply(&mut f, &why), 2);
        assert!(f.repeat_spot);
        assert_eq!(f.boxes[0].reason.as_deref(), Some(REPEAT_SPOT_REASON));
        assert_eq!(f.boxes[1].reason.as_deref(), Some("night_threshold"));
        assert!(!f.boxes[1].difficult);
        assert_eq!(f.boxes[1].reject_reason, Some(RejectReason::StaticObject));
        assert_eq!(f.boxes[2].state, BoxState::Pending);
        assert!(!repeats_only(&f), "the dog is still pending");
        f.boxes[2].state = BoxState::Rejected;
        assert!(repeats_only(&f));
        f.status = Status::Reviewed;
        assert!(!repeats_only(&f));
        assert_eq!(s.count(&f, &why), 0, "reviewed frames are left alone");
    }

    #[test]
    fn the_history_lists_one_cameras_looked_at_decisions_newest_first() {
        use super::super::bulk::test_support::{labels_of, put_frame_at};
        let dir = std::env::temp_dir().join(format!("bop-history-{}", uuid::Uuid::new_v4()));
        let decided = |label: &str, state: BoxState, reason: Option<&str>| {
            let mut b = auto_box(label);
            b.state = state;
            b.reason = reason.map(str::to_string);
            b
        };
        let mut why = decided("car", BoxState::Rejected, None);
        why.reject_reason = Some(RejectReason::Reflection);
        put_frame_at(
            &dir,
            "cam",
            "20261021T010000000Z_a",
            (64, 48),
            None,
            vec![
                why,
                decided("car", BoxState::Pending, None),
                decided("dog", BoxState::Rejected, Some("low_score")),
            ],
        );
        let newer = put_frame_at(
            &dir,
            "cam",
            "20261021T020000000Z_b",
            (64, 48),
            None,
            vec![decided("car", BoxState::Accepted, Some("night_threshold"))],
        );
        put_frame_at(
            &dir,
            "drive",
            "20261021T030000000Z_c",
            (64, 48),
            None,
            vec![decided("car", BoxState::Rejected, None)],
        );
        let h = scan(&dir, "cam", HISTORY_CAP);
        let got: Vec<(&str, BoxState)> = h
            .decisions
            .iter()
            .map(|d| (d.label.as_str(), d.state))
            .collect();
        assert_eq!(
            got,
            [("car", BoxState::Accepted), ("car", BoxState::Rejected)],
            "newest first, rules and pending left out"
        );
        assert_eq!(h.decisions[1].reject_reason, Some(RejectReason::Reflection));
        assert_eq!((h.decisions[0].width, h.decisions[0].height), (64, 48));
        assert_eq!(h.decisions[0].frame, newer);
        assert!(!h.truncated);
        let capped = scan(&dir, "cam", 1);
        assert_eq!((capped.decisions.len(), capped.truncated), (1, true));
        assert_eq!(scan(&dir, "drive", HISTORY_CAP).decisions.len(), 1);
        // Cached: a change behind its back (another process) is not seen...
        assert_eq!(history(&dir, "cam").decisions.len(), 2);
        let frame = super::super::Frame::resolve(&dir, &newer).unwrap();
        let mut l = labels_of(&dir, &newer);
        l.boxes.push(decided("person", BoxState::Rejected, None));
        std::fs::write(&frame.labels, serde_json::to_vec(&l).unwrap()).unwrap();
        assert_eq!(history(&dir, "cam").decisions.len(), 2, "cached");
        invalidate(&dir);
        assert_eq!(history(&dir, "cam").decisions.len(), 3);
        // ...a write through this module drops it at once.
        l.boxes.push(decided("bird", BoxState::Rejected, None));
        super::super::write(&frame.labels, &l).unwrap();
        assert_eq!(history(&dir, "cam").decisions.len(), 4);
        invalidate(&dir);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn decisions_by_rules_do_not_count_as_a_look() {
        let mut b = auto_box("car");
        assert!(!counts(&b), "pending");
        b.state = BoxState::Rejected;
        assert!(counts(&b));
        for r in ["low_score", "other_class", REPEAT_SPOT_REASON] {
            b.reason = Some(r.into());
            assert!(!counts(&b), "{r}");
        }
        b.reason = Some("night_threshold".into());
        assert!(
            counts(&b),
            "a suppression reason is provenance, the decision was the reviewer's"
        );
        b.source = Source::Human;
        b.state = BoxState::Added;
        assert!(!counts(&b));
    }
}
