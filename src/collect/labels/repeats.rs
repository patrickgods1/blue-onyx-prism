//! Repeat detections of fixed cameras: the same false positive (a railing, a plant) or the same
//! parked car at one spot, frame after frame. Pure parts: which boxes a spot matches
//! ([`Spot::matches`]: a pending model / auto box of the same object class, [`same_object`], at
//! IoU >= [`Spot::min_iou`]) and whether a frame holds only such decided repeats
//! ([`repeats_only`]). IO: the per-camera history of the reviewer's decisions ([`history`], the
//! Review tab's "rejected here before" hint; cached per folder and camera, [`invalidate`] after
//! every write of the folder's label files). The bulk edits are [`super::bulk::bulk_spot`] and
//! [`super::bulk::bulk_accept_repeats`]; what a labelers run does with spots is in [`super::spots`].

use super::super::layout::Walker;
use super::{BoxState, LabelBox, LabelFile, RejectReason, Source, Status};
use crate::sync::LockExt;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
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
/// Earlier decisions at a place that make it a spot (the Review tab's `REPEAT_MIN`).
pub const REPEAT_MIN: usize = 2;
/// How long a cached history is used without a write of the folder (label files written by
/// other processes, such as an auto-label run, show up after this).
pub const HISTORY_TTL: Duration = Duration::from_secs(60);

/// Vehicle classes: a box of one may stand for an object another names (the Review tab's
/// `VEHICLES`, `assets/review-rules.js`).
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
                Verdict::Accept => {
                    b.state = BoxState::Accepted;
                    // The look that flag remembers did not cover this box.
                    f.checked_before_reopen = false;
                }
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
/// ([`LabelFile::repeat_spot`]), no pending box left, no dismissed box
/// ([`LabelBox::is_dismissed`]: a save claims the frame is complete, and nobody saw those), and no
/// accepted box the VLM said no to (a spot edit or a parked spot took it, nobody looked at this
/// frame, and the parked car may have left: [`super::spots`]'s rule, also for the page's "accept
/// this spot"). Sorted last in the queue; the "accept repeat-only frames" bulk save takes these
/// only.
pub fn repeats_only(f: &LabelFile) -> bool {
    f.status == Status::Unreviewed
        && f.repeat_spot
        && !f.boxes.iter().any(|b| {
            b.state == BoxState::Pending
                || b.is_dismissed()
                || (b.state == BoxState::Accepted
                    && b.auto.vlm.as_ref().is_some_and(super::VlmCheck::says_no))
        })
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
/// rejects, a labelers run's dismissals, its parked accepts, an earlier spot edit; counting those
/// would make a spot look more certain than any look at it was). A duplicate reject is no verdict on the spot either: another box stays on the
/// object there (a real parked car must not read "rejected here N times"), and neither is a
/// too-small reject: a real object, only too small to label. Twin of the page's `lookedAt`.
fn counts(b: &LabelBox) -> bool {
    b.source != Source::Human
        && matches!(b.state, BoxState::Accepted | BoxState::Rejected)
        && !matches!(
            b.reject_reason,
            Some(RejectReason::Duplicate | RejectReason::TooSmall)
        )
        && !super::bulk::is_dismissal(b.reason.as_deref())
        && !matches!(
            b.reason.as_deref(),
            Some(
                super::bulk::LOW_SCORE_REASON
                    | super::bulk::OTHER_CLASS_REASON
                    | REPEAT_SPOT_REASON
                    | super::spots::PARKED_REASON
            )
        )
}

/// Scan `dir` for the decisions of `camera` ([`counts`]), newest frames first, at most `cap`.
/// A frame whose sidecar does not read (its size is unknown) is left out.
pub fn scan(dir: &Path, camera: &str, cap: usize) -> History {
    let mut files = super::camera_label_files(dir, camera);
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
    let mut walker = Walker::new(dir);
    'files: for (rel, path) in files {
        let Some(l) = super::read(&path).ok().flatten() else {
            continue;
        };
        if !l.boxes.iter().any(counts) {
            continue;
        }
        let Some(rec) = super::Frame::resolve_in(&mut walker, &rel).and_then(|f| f.record().ok())
        else {
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

/// The canonical folder and the camera of a history.
type Key = (PathBuf, String);
type Cache = HashMap<Key, (Instant, Arc<History>)>;

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

/// One running scan and those waiting for it ([`history_with`]).
#[derive(Default)]
struct Flight {
    /// `None` while it runs; then its history, `None` inside when the scan panicked.
    done: Mutex<Option<Option<Arc<History>>>>,
    finished: Condvar,
}

impl Flight {
    /// Wait for the scan; its history, unless it panicked.
    fn wait(&self) -> Option<Arc<History>> {
        let mut done = self.done.lock_recover();
        loop {
            if let Some(h) = done.as_ref() {
                return h.clone();
            }
            done = self.finished.wait(done).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// The running scan of each folder and camera.
fn flights() -> &'static Mutex<HashMap<Key, Arc<Flight>>> {
    static FLIGHTS: std::sync::OnceLock<Mutex<HashMap<Key, Arc<Flight>>>> =
        std::sync::OnceLock::new();
    FLIGHTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Ends a scan when dropped, panic included, so one failure wedges neither the waiters nor the
/// next [`written`].
struct Landing {
    id: u64,
    key: Key,
    flight: Arc<Flight>,
    history: Option<Arc<History>>,
}

impl Drop for Landing {
    fn drop(&mut self) {
        scanning().lock_recover().remove(&self.id);
        // In this order: a caller that finds no flight finds the cache already filled.
        flights().lock_recover().remove(&self.key);
        *self.flight.done.lock_recover() = Some(self.history.take());
        self.flight.finished.notify_all();
    }
}

/// [`scan`] of `camera` under `dir`, cached until a label file under `dir` is written
/// ([`written`], [`invalidate`]) or for [`HISTORY_TTL`] (writes by other processes). One scan per
/// folder and camera at a time: a second caller waits for the running one.
pub fn history(dir: &Path, camera: &str) -> Arc<History> {
    // Keyed by the canonical folder: the frames' paths ([`super::Frame::resolve`]) are canonical.
    let root = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    history_with((root, camera.to_string()), &|| {
        scan(dir, camera, HISTORY_CAP)
    })
}

/// The cached history of `key`, else `scan`'s, one scan at a time.
///
/// A waiter whose scan was overtaken by a write ([`written`], not cached) may have been handed
/// something older than what it came for, so it asks again; the scan it then meets started after
/// it arrived, which is as good as its own, so it takes that answer whatever happens to it: under
/// a steady stream of writes the waiters do not run one scan each, one after the other.
fn history_with(key: Key, scan: &dyn Fn() -> History) -> Arc<History> {
    static IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut started_since_arrival = false;
    loop {
        if let Ok(c) = cache().lock()
            && let Some((at, h)) = c.get(&key)
            && at.elapsed() < HISTORY_TTL
        {
            return h.clone();
        }
        let flight = {
            let mut running = flights().lock_recover();
            match running.get(&key) {
                Some(f) => Ok(f.clone()),
                None => {
                    let f = Arc::new(Flight::default());
                    running.insert(key.clone(), f.clone());
                    Err(f)
                }
            }
        };
        match flight {
            Ok(joined) => {
                let answer = joined.wait();
                if started_since_arrival && let Some(h) = answer {
                    return h;
                }
                started_since_arrival = true;
            }
            Err(flight) => {
                let id = IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                scanning().lock_recover().insert(id, (key.0.clone(), false));
                let mut landing = Landing {
                    id,
                    key: key.clone(),
                    flight,
                    history: None,
                };
                let h = Arc::new(scan());
                // The cache lock first: a [`written`] past the check below cannot get to the cache
                // before this entry is in it, and it drops the entry then.
                let mut c = cache().lock_recover();
                let overtaken = scanning()
                    .lock_recover()
                    .remove(&id)
                    .is_none_or(|(_, dirty)| dirty);
                if !overtaken {
                    c.insert(key, (Instant::now(), h.clone()));
                }
                drop(c);
                landing.history = Some(h.clone());
                return h;
            }
        }
    }
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
        // A reviewer looking at the frame chose to reject it: that teaches the spot.
        b.reason = Some(super::super::REST_REASON.into());
        assert!(counts(&b), "reject the rest");
        b.reason = Some("night_threshold".into());
        assert!(
            counts(&b),
            "a suppression reason is provenance, the decision was the reviewer's"
        );
        b.reject_reason = Some(RejectReason::Reflection);
        assert!(counts(&b), "a reason of the reviewer's");
        b.reject_reason = Some(RejectReason::Duplicate);
        assert!(!counts(&b), "a duplicate says nothing about the spot");
        b.reject_reason = Some(RejectReason::TooSmall);
        assert!(!counts(&b), "too small is no verdict on the spot either");
        b.reject_reason = None;
        b.state = BoxState::Accepted;
        b.difficult = true;
        assert!(counts(&b), "an unsure accept is still the reviewer's look");
        b.difficult = false;
        b.source = Source::Human;
        b.state = BoxState::Added;
        assert!(!counts(&b));
    }

    fn dismissed() -> LabelBox {
        LabelBox {
            state: BoxState::Rejected,
            reason: Some(super::super::bulk::DISMISSED_REASON.into()),
            ..super::super::tests_support::auto_box("car")
        }
    }

    #[test]
    fn a_dismissed_box_teaches_the_spot_nothing() {
        assert!(!counts(&dismissed()));
        // Once a reviewer rejects it the reason is gone, and it counts.
        let mut judged = dismissed();
        judged.reason = None;
        assert!(counts(&judged));
    }

    #[test]
    fn the_rules_of_a_run_teach_the_spot_nothing_either() {
        let mut b = dismissed();
        b.reason = Some(super::super::bulk::SPOT_DISMISSED_REASON.into());
        assert!(!counts(&b), "spot_dismissed");
        b.reason = Some(super::super::spots::PARKED_REASON.into());
        b.state = BoxState::Accepted;
        assert!(!counts(&b), "a parked accept is the rule's");
        b.reason = None;
        assert!(counts(&b), "\"Real, parked\" itself is a reviewer's look");
    }

    /// `assets/<name>`, read at run time: `include_str!` would rebuild this test binary on a page
    /// edit.
    fn page_script(name: &str) -> String {
        let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
        std::fs::read_to_string(assets.join(name)).unwrap()
    }

    #[test]
    fn the_page_shares_the_spot_constants() {
        let js = page_script("review.js");
        assert!(js.contains(&format!("REPEAT_MIN = {REPEAT_MIN}")));
        assert!(js.contains(&format!("REPEAT_IOU = {REJECT_SPOT_IOU},")));
        assert!(js.contains(&format!("ACCEPT_SPOT_IOU = {ACCEPT_SPOT_IOU};")));
        let rules = page_script("review-rules.js");
        for reason in [
            super::super::bulk::DISMISSED_REASON,
            super::super::bulk::SPOT_DISMISSED_REASON,
            super::super::bulk::SIZE_FLOOR_REASON,
            super::super::spots::PARKED_REASON,
            super::super::stacks::DUPLICATE_REASON,
        ] {
            assert!(rules.contains(&format!("'{reason}'")), "{reason}");
        }
    }

    /// The thresholds and the vehicle list of the page's rules, by their text: this runs where
    /// `tests/it/js_rules.rs` skips for want of `node`, which compares the values too.
    #[test]
    fn the_page_shares_the_stack_thresholds_and_the_vehicles() {
        use super::super::stacks::{NESTED_SHARE, OVERLAP_IOU, STACK_IOU};
        let rules = page_script("review-rules.js");
        for line in [
            format!(
                "var STACK_IOU = {STACK_IOU}, OVERLAP_IOU = {OVERLAP_IOU}, \
                 NESTED_SHARE = {NESTED_SHARE};"
            ),
            format!(
                "var VEHICLES = [{}];",
                VEHICLES.map(|v| format!("'{v}'")).join(", ")
            ),
        ] {
            assert!(rules.contains(&line), "{line}");
        }
    }

    #[test]
    fn a_frame_with_a_dismissed_box_is_not_repeat_only() {
        let mut f = LabelFile {
            repeat_spot: true,
            boxes: vec![LabelBox {
                state: BoxState::Rejected,
                ..super::super::tests_support::auto_box("dog")
            }],
            ..LabelFile::default()
        };
        assert!(repeats_only(&f));
        f.boxes.push(dismissed());
        assert!(!repeats_only(&f), "a bulk save would pass over it unseen");
    }

    #[test]
    fn a_spot_edit_leaves_a_dismissed_box_alone() {
        let d = dismissed();
        let spot = Spot {
            camera: "c".into(),
            label: "car".into(),
            bbox: d.bbox,
            width: 640,
            height: 480,
            min_iou: REJECT_SPOT_IOU,
        };
        assert!(!spot.matches(&d, &Verdict::Reject(None)));
        assert!(!spot.matches(&d, &Verdict::Accept));
    }

    /// Test of [`history_with`]: a scan that says it started and waits to be let go.
    struct Gated {
        scans: std::sync::atomic::AtomicUsize,
        started: Mutex<std::sync::mpsc::Sender<usize>>,
        go: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl Gated {
        fn scan(&self) -> History {
            use std::sync::atomic::Ordering::SeqCst;
            let n = self.scans.fetch_add(1, SeqCst) + 1;
            self.started.lock().unwrap().send(n).unwrap();
            self.go.lock().unwrap().recv().unwrap();
            History {
                camera: format!("scan {n}"),
                ..History::default()
            }
        }
    }

    type Gate = (
        Arc<Gated>,
        std::sync::mpsc::Receiver<usize>,
        std::sync::mpsc::Sender<()>,
    );

    fn gated() -> Gate {
        let (started_tx, started) = std::sync::mpsc::channel();
        let (go, go_rx) = std::sync::mpsc::channel();
        let g = Gated {
            scans: std::sync::atomic::AtomicUsize::new(0),
            started: Mutex::new(started_tx),
            go: Mutex::new(go_rx),
        };
        (Arc::new(g), started, go)
    }

    fn fresh_key() -> Key {
        let root = PathBuf::from(format!("/bop-flight-{}", uuid::Uuid::new_v4()));
        (root, "cam".to_string())
    }

    fn call(key: &Key, g: &Arc<Gated>) -> std::thread::JoinHandle<Arc<History>> {
        let (key, g) = (key.clone(), g.clone());
        std::thread::spawn(move || history_with(key, &|| g.scan()))
    }

    /// Wait until `n` callers wait for the running scan of `key`.
    fn wait_for_waiters(key: &Key, n: usize) {
        let until = Instant::now() + Duration::from_secs(30);
        loop {
            // The list and the scan's own landing hold one each.
            let held = flights()
                .lock_recover()
                .get(key)
                .map_or(0, |f| Arc::strong_count(f).saturating_sub(2));
            if held == n {
                return;
            }
            assert!(Instant::now() < until, "{held} waiting, not {n}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn scans_left(key: &Key) -> bool {
        flights().lock_recover().contains_key(key)
            || scanning().lock_recover().values().any(|(d, _)| *d == key.0)
    }

    #[test]
    fn a_second_caller_waits_for_the_running_scan() {
        let (g, started, go) = gated();
        let key = fresh_key();
        let lead = call(&key, &g);
        started.recv().unwrap();
        let waiters: Vec<_> = (0..4).map(|_| call(&key, &g)).collect();
        wait_for_waiters(&key, 4);
        go.send(()).unwrap();
        let first = lead.join().unwrap();
        assert_eq!(first.camera, "scan 1");
        for w in waiters {
            assert!(Arc::ptr_eq(&w.join().unwrap(), &first), "the same scan");
        }
        assert_eq!(g.scans.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(!scans_left(&key));
        // Another camera of the folder is its own scan.
        let other = (key.0.clone(), "drive".to_string());
        let lead = call(&other, &g);
        started.recv().unwrap();
        go.send(()).unwrap();
        assert_eq!(lead.join().unwrap().camera, "scan 2");
    }

    #[test]
    fn a_panic_in_a_scan_wedges_nobody() {
        let (g, started, go) = gated();
        let key = fresh_key();
        let (let_go, wait) = std::sync::mpsc::channel::<()>();
        let panicking = {
            let (key, started) = (key.clone(), g.started.lock().unwrap().clone());
            std::thread::spawn(move || {
                history_with(key, &|| {
                    started.send(0).unwrap();
                    wait.recv().unwrap();
                    panic!("a scan fails");
                })
            })
        };
        assert_eq!(started.recv().unwrap(), 0);
        let waiter = call(&key, &g);
        wait_for_waiters(&key, 1);
        let_go.send(()).unwrap();
        assert!(panicking.join().is_err());
        // The waiter takes over.
        assert_eq!(started.recv().unwrap(), 1);
        go.send(()).unwrap();
        assert_eq!(waiter.join().unwrap().camera, "scan 1");
        assert!(!scans_left(&key));
    }

    #[test]
    fn a_waiter_of_an_overtaken_scan_does_not_get_the_old_answer() {
        let (g, started, go) = gated();
        let key = fresh_key();
        let lead = call(&key, &g);
        started.recv().unwrap();
        let waiter = call(&key, &g);
        wait_for_waiters(&key, 1);
        written(&key.0.join("ir/cam/2026-10-21/a.labels.json"));
        go.send(()).unwrap();
        assert_eq!(lead.join().unwrap().camera, "scan 1", "its caller's answer");
        // Not cached: the waiter scans again, after the write.
        started.recv().unwrap();
        go.send(()).unwrap();
        assert_eq!(waiter.join().unwrap().camera, "scan 2");
        // That one saw no write: it is cached.
        let cached = history_with(key, &|| panic!("cached"));
        assert_eq!(cached.camera, "scan 2");
    }

    #[test]
    fn under_steady_writes_a_waiter_takes_the_scan_that_started_after_it() {
        let (g, started, go) = gated();
        let key = fresh_key();
        let lead = call(&key, &g);
        started.recv().unwrap();
        let waiters: Vec<_> = (0..3).map(|_| call(&key, &g)).collect();
        wait_for_waiters(&key, 3);
        let label = key.0.join("ir/cam/2026-10-21/a.labels.json");
        written(&label);
        go.send(()).unwrap();
        lead.join().unwrap();
        // One of them scans again; the others join it, and it too is overtaken.
        started.recv().unwrap();
        wait_for_waiters(&key, 2);
        written(&label);
        go.send(()).unwrap();
        for w in waiters {
            assert_eq!(w.join().unwrap().camera, "scan 2");
        }
        assert_eq!(g.scans.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn history_lists_only_the_folders_of_its_camera() {
        let dir = std::env::temp_dir().join(format!("bop-repeats-{}", uuid::Uuid::new_v4()));
        let rejected = LabelBox {
            state: BoxState::Rejected,
            ..auto_box("car")
        };
        let frame = |camera, stem, b: &LabelBox| {
            use super::super::bulk::test_support::put_frame_at;
            put_frame_at(&dir, camera, stem, (64, 48), None, vec![b.clone()]);
        };
        frame("cam", "20261021T010000000Z_a", &rejected);
        frame("drive", "20261021T010000000Z_b", &rejected);
        assert_eq!(scan(&dir, "cam", HISTORY_CAP).decisions.len(), 1);
        assert!(scan(&dir, "..", HISTORY_CAP).decisions.is_empty());
        assert!(scan(&dir, "nobody", HISTORY_CAP).decisions.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
