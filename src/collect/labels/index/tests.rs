//! The index's tests, and the hooks other modules' tests use to make a small folder take an index
//! ([`force`], [`unforce`], [`fast_timing`], [`hold`]). Tests run side by side and the index is
//! one per process, so a forced folder is its own group: its index is independent of every other
//! test's ([`super::same_group`]).
//!
//! The test that matters is [`equivalence`]: for a seeded folder holding every case `queue::build`
//! knows, and a grid of filters and `after` values, the index's JSON equals the walk's, whole,
//! after a build, after writes through the real writers, after direct edits and a scan, and after
//! a full pass.

use super::Timing;
use crate::sync::LockExt as _;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::Duration;

/// A held build: the walk stops before its next frame until the [`Hold`] goes.
#[derive(Debug, Default)]
struct Gate {
    /// Still held.
    held: Mutex<bool>,
    /// Rung when the hold goes, and when a walk reaches it.
    rung: Condvar,
    /// Walks that reached it while held.
    reached: AtomicU64,
    /// Frames walked before it holds.
    after: u64,
}

/// What a test asked of one folder.
#[derive(Debug, Clone)]
struct Forced {
    group: u64,
    timing: Option<Timing>,
    gate: Option<Arc<Gate>>,
    /// The walk panics at its frame number this many.
    panic_at: Option<u64>,
    /// Frames walked, every walk of the folder.
    frames: Arc<AtomicU64>,
    /// How far the walks' clock is ahead of the real one.
    ahead: Option<Duration>,
    /// Whether the last query of the folder was answered by the index.
    served: bool,
}

static FORCED: LazyLock<Mutex<HashMap<PathBuf, Forced>>> = LazyLock::new(Mutex::default);
static GROUPS: AtomicU64 = AtomicU64::new(1);

/// The forced entry of `dir`, as given or canonical. A leaf lock.
fn with<T>(dir: &Path, f: impl FnOnce(&mut Forced) -> T) -> Option<T> {
    let mut forced = FORCED.lock_recover();
    if forced.contains_key(dir) {
        return forced.get_mut(dir).map(f);
    }
    let canon = dir.canonicalize().ok()?;
    forced
        .iter_mut()
        .find(|(k, _)| k.canonicalize().ok().as_deref() == Some(&canon))
        .map(|(_, v)| f(v))
}

/// The times a test index uses unless a test changes them: real, except that a query waits for
/// the build as long as it takes, so a forced answer is complete and the same on a slow machine.
fn settled() -> Timing {
    Timing {
        wait: Duration::from_secs(30),
        ..Timing::REAL
    }
}

/// Make `dir` take an index whatever its size (a folder of a few frames too), with times that
/// make every answer complete: a query waits for the build. Its own group: other tests' indexes
/// are not touched. Calling it again keeps what was set.
pub(crate) fn force(dir: &Path) {
    let mut forced = FORCED.lock_recover();
    forced.entry(dir.to_path_buf()).or_insert_with(|| Forced {
        group: GROUPS.fetch_add(1, Ordering::Relaxed),
        timing: Some(settled()),
        gate: None,
        panic_at: None,
        frames: Arc::default(),
        ahead: None,
        served: false,
    });
}

/// `dir` walks again; its index, if any, is dropped.
pub(crate) fn unforce(dir: &Path) {
    let canon = dir.canonicalize().ok();
    FORCED
        .lock_recover()
        .retain(|k, _| k != dir && (canon.is_none() || k.canonicalize().ok() != canon));
    let mut hub = super::HUB.lock_recover();
    hub.clear_where(|i| i.dir == dir);
    hub.refused.retain(|r| r.dir != dir);
}

/// [`force`], and the background passes come every 25 ms (a stale day after 1 ms), so a test sees
/// a scan within a moment. A query still waits for the build. Call before the folder's first query.
pub(crate) fn fast_timing(dir: &Path) {
    force(dir);
    set_timing(dir, |t| {
        t.refresh = Duration::from_millis(25);
        t.pause = Duration::from_millis(1);
    });
}

/// Change the times of `dir`'s next index (not one running now).
pub(super) fn set_timing(dir: &Path, change: impl FnOnce(&mut Timing)) {
    force(dir);
    with(dir, |f| {
        let mut t = f.timing.unwrap_or_else(settled);
        change(&mut t);
        f.timing = Some(t);
    });
}

/// Put `dirs` in one group, as the real folders are: a query of one drops the index of another.
pub(super) fn force_together(dirs: &[&Path]) {
    let group = GROUPS.fetch_add(1, Ordering::Relaxed);
    for dir in dirs {
        force(dir);
        with(dir, |f| f.group = group);
    }
}

/// A held build of `dir`'s index: every walk stops before its next frame until this goes, and a
/// query answers at once (with `indexing`) instead of waiting.
#[derive(Debug)]
#[must_use = "the hold ends when this is dropped"]
pub(crate) struct Hold {
    dir: PathBuf,
    gate: Arc<Gate>,
}

impl Hold {
    /// Wait until a walk is stopped at the hold (at most 10 s): true when it is.
    pub(crate) fn reached(&self) -> bool {
        let held = self.gate.held.lock_recover();
        let (_held, timeout) = self
            .gate
            .rung
            .wait_timeout_while(held, Duration::from_secs(10), |_| {
                self.gate.reached.load(Ordering::Acquire) == 0
            })
            .unwrap_or_else(|e| e.into_inner());
        !timeout.timed_out()
    }

    /// Let the walk go on.
    pub(crate) fn release(self) {}
}

impl Drop for Hold {
    fn drop(&mut self) {
        *self.gate.held.lock_recover() = false;
        self.gate.rung.notify_all();
        with(&self.dir, |f| {
            if f.gate.as_ref().is_some_and(|g| Arc::ptr_eq(g, &self.gate)) {
                f.gate = None;
            }
        });
    }
}

/// Hold the walks of `dir` (forced) before their next frame. Take it before the first query.
pub(crate) fn hold(dir: &Path) -> Hold {
    hold_after(dir, 0)
}

/// [`hold`] once `frames` more frames are walked.
pub(super) fn hold_after(dir: &Path, frames: u64) -> Hold {
    force(dir);
    let walked = with(dir, |f| f.frames.load(Ordering::Acquire)).unwrap_or(0);
    let gate = Arc::new(Gate {
        held: Mutex::new(true),
        after: walked + frames,
        ..Gate::default()
    });
    with(dir, |f| f.gate = Some(gate.clone()));
    Hold {
        dir: dir.to_path_buf(),
        gate,
    }
}

/// Make the walk of `dir` panic at its frame number `at` (from 0, across its walks).
pub(super) fn panic_at(dir: &Path, at: u64) {
    force(dir);
    with(dir, |f| f.panic_at = Some(at));
}

/// Walks of `dir` stat as if it were `ahead` later: nothing written now is racy then.
pub(super) fn set_clock_ahead(dir: &Path, ahead: Duration) {
    force(dir);
    with(dir, |f| f.ahead = Some(ahead));
}

/// See [`set_clock_ahead`].
pub(super) fn clock_ahead(dir: &Path) -> Option<Duration> {
    with(dir, |f| f.ahead).flatten()
}

/// Frames the walks of `dir` read so far.
pub(super) fn frames_walked(dir: &Path) -> u64 {
    with(dir, |f| f.frames.load(Ordering::Acquire)).unwrap_or(0)
}

/// Whether the last `/next` of `dir` was answered by its index (false: walked, or not forced).
pub(crate) fn served(dir: &Path) -> bool {
    with(dir, |f| f.served).unwrap_or(false)
}

/// See [`served`].
pub(super) fn note_served(dir: &Path, by_index: bool) {
    with(dir, |f| f.served = by_index);
}

/// Whether a test asked for an index of `dir`.
pub(super) fn forced(dir: &Path) -> bool {
    with(dir, |_| ()).is_some()
}

/// The times a test set for `dir`.
pub(super) fn timing(dir: &Path) -> Option<Timing> {
    with(dir, |f| f.timing).flatten()
}

/// Whether a walk of `dir` is held: a query does not wait.
pub(super) fn held(dir: &Path) -> bool {
    with(dir, |f| f.gate.is_some()).unwrap_or(false)
}

/// Whether `a` and `b` share one index.
pub(super) fn same_group(a: &Path, b: &Path) -> bool {
    let (ga, gb) = (with(a, |f| f.group), with(b, |f| f.group));
    ga.is_some() && ga == gb
}

/// Called by a walk before each frame of `dir`.
pub(super) fn walk_point(dir: &Path) {
    let Some((gate, panic_at, frames)) =
        with(dir, |f| (f.gate.clone(), f.panic_at, f.frames.clone()))
    else {
        return;
    };
    let n = frames.fetch_add(1, Ordering::AcqRel);
    assert!(
        panic_at != Some(n),
        "test panic in the walk of {}",
        dir.display()
    );
    if let Some(gate) = gate.filter(|g| n >= g.after) {
        let mut held = gate.held.lock_recover();
        if *held {
            gate.reached.fetch_add(1, Ordering::AcqRel);
            gate.rung.notify_all();
        }
        while *held {
            held = gate.rung.wait(held).unwrap_or_else(|e| e.into_inner());
        }
    }
}

#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod cases {
    use super::super::super::bulk::{self, Wanted};
    use super::super::super::overlaps::{Suggestion, Verdict};
    use super::super::super::queue::{self, AutoFilter, Filters, QueueSource, StatusFilter};
    use super::super::super::refine;
    use super::super::super::{
        AutoInfo, AutoRun, BoxState, Frame, LABELS_SUFFIX, LabelBox, LabelFile, Origin,
        RejectReason, Source, Status, Triage, VlmCheck,
    };
    use super::super::Indexing;
    use super::super::refresh::Pass;
    use super::super::{Answer, DIRTY_MAX, HUB, Index};
    use super::*;
    use crate::collect::layout::{self, INDEX_FILE, REVIEW_QUEUE_FILE};
    use crate::collect::record::{
        Bucket, Det, IndexLine, Record, Suppressed, VerifyCandidate, VerifyInfo, empty_record,
    };
    use crate::sync::RwLockExt as _;
    use std::collections::BTreeMap;
    use std::sync::RwLock;
    use std::time::Instant;

    /// A folder of its own, removed with its index when the test ends, failed or not.
    struct Dir(PathBuf);

    impl Dir {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("bop-index-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&dir).unwrap();
            force(&dir);
            Self(dir)
        }
    }

    impl std::ops::Deref for Dir {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            unforce(&self.0);
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The index of `dir` now.
    fn index_of(dir: &Path) -> Option<Arc<Index>> {
        HUB.lock_recover()
            .current
            .iter()
            .find(|c| c.index.dir == dir)
            .map(|c| c.index.clone())
    }

    /// xorshift: the same folder for the same seed.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
        }

        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            usize::try_from(self.0 % n as u64).unwrap()
        }

        fn one_in(&mut self, n: usize) -> bool {
            self.below(n) == 0
        }

        fn pick<T: Clone>(&mut self, xs: &[T]) -> T {
            xs[self.below(xs.len())].clone()
        }

        fn score(&mut self) -> f32 {
            self.below(100) as f32 / 100.0
        }
    }

    const SCENES: [&str; 3] = ["day", "night", "ir"];
    const CAMERAS: [&str; 3] = ["gate", "yard", "porch"];
    const DATES: [&str; 3] = ["2026-10-20", "2026-10-21", "2026-10-22"];
    /// `motorbike` maps onto `motorcycle`; `plant` is of no wanted class.
    const CLASSES: [&str; 6] = ["person", "car", "dog", "truck", "motorbike", "plant"];
    /// Places that stack (0, 1), nest (0, 2) and stand apart (3, 4).
    const PLACES: [[f32; 4]; 5] = [
        [0.0, 0.0, 40.0, 40.0],
        [1.0, 1.0, 40.0, 40.0],
        [5.0, 5.0, 25.0, 25.0],
        [100.0, 100.0, 140.0, 150.0],
        [200.0, 10.0, 260.0, 90.0],
    ];

    fn wanted_a() -> (Vec<String>, Vec<String>) {
        let classes = ["person", "car", "dog", "truck", "motorcycle"];
        (classes.map(String::from).to_vec(), Vec::new())
    }

    fn wanted_b() -> (Vec<String>, Vec<String>) {
        (
            ["person", "car", "dog"].map(String::from).to_vec(),
            vec!["dog".to_string()],
        )
    }

    fn det(rng: &mut Rng) -> Det {
        Det {
            label: rng.pick(&CLASSES).into(),
            score: rng.score(),
            bbox: rng.pick(&PLACES),
        }
    }

    fn sidecar(rng: &mut Rng, scene: &str, camera: &str, stem: &str) -> Record {
        let mut r = empty_record();
        r.id = stem.into();
        r.scene = scene.into();
        r.camera = camera.into();
        // Few values: ties that only the path breaks.
        r.timestamp = format!("2026-10-21T00:00:0{}Z", rng.below(4));
        r.threshold = 0.5;
        r.model = format!("m{}", rng.below(3));
        let n = rng.below(4);
        r.buckets = (0..n).map(|_| rng.pick(&Bucket::ALL)).collect();
        if rng.one_in(8) {
            r.matched = vec![Bucket::Disagreement];
        }
        r.predictions = (0..rng.below(3)).map(|_| det(rng)).collect();
        r.raw = r.predictions.clone();
        r.raw.extend((0..rng.below(2)).map(|_| det(rng)));
        r.below_threshold = (0..rng.below(2)).map(|_| det(rng)).collect();
        r.suppressed = (0..rng.below(3))
            .map(|_| {
                let d = det(rng);
                Suppressed {
                    label: d.label,
                    score: d.score,
                    bbox: d.bbox,
                    reason: "night_threshold".into(),
                }
            })
            .collect();
        if rng.one_in(6) {
            r.verify = Some(VerifyInfo {
                candidates: vec![VerifyCandidate {
                    label: "person".into(),
                    score: 0.6,
                    bbox: PLACES[0],
                    outcome: rng.pick(&["rejected", "kept"]).into(),
                }],
                ..VerifyInfo::default()
            });
        }
        r
    }

    fn a_box(rng: &mut Rng) -> LabelBox {
        let source = rng.pick(&[Source::Model, Source::Auto, Source::Auto, Source::Human]);
        let state = rng.pick(&[
            BoxState::Pending,
            BoxState::Pending,
            BoxState::Pending,
            BoxState::Accepted,
            BoxState::Rejected,
            BoxState::Rejected,
            BoxState::Added,
        ]);
        let mut b = LabelBox {
            label: rng.pick(&CLASSES).into(),
            bbox: rng.pick(&PLACES),
            source,
            score: Some(rng.score()),
            state,
            origin: (source == Source::Model).then(|| {
                rng.pick(&[
                    Origin::Prediction,
                    Origin::Suppressed,
                    Origin::BelowThreshold,
                ])
            }),
            reason: None,
            model_label: rng.one_in(10).then(|| "motorbike".into()),
            orig_bbox: rng.one_in(12).then_some(PLACES[3]),
            difficult: rng.one_in(10),
            reject_reason: None,
            auto: AutoInfo::default(),
        };
        if state == BoxState::Rejected {
            b.reason = rng
                .pick(&[
                    None,
                    None,
                    None,
                    Some(bulk::DISMISSED_REASON),
                    Some(bulk::SPOT_DISMISSED_REASON),
                    Some(bulk::SIZE_FLOOR_REASON),
                    Some(bulk::LOW_SCORE_REASON),
                    Some(bulk::OTHER_CLASS_REASON),
                ])
                .map(String::from);
            b.reject_reason = rng.pick(&[
                None,
                None,
                Some(RejectReason::Duplicate),
                Some(RejectReason::TooSmall),
                Some(RejectReason::FalsePositive),
                Some(RejectReason::Other("newer".into())),
            ]);
        }
        if source != Source::Human {
            b.auto.triage = rng.pick(&[None, Some(Triage::Review), Some(Triage::Accept)]);
            for name in ["gdino", "rfdetr", "m1"] {
                if rng.one_in(2) {
                    b.auto.labelers.insert(name.into(), rng.score());
                }
            }
            b.auto.likely = rng.one_in(4);
            b.auto.audit = rng.one_in(3);
            b.auto.auto_label = rng.one_in(8).then(|| "car".into());
            if rng.one_in(5) {
                let ok = rng.one_in(2);
                b.auto.vlm = Some(VlmCheck {
                    model: "qwen".into(),
                    answer: if ok { "yes" } else { "no" }.into(),
                    expected: "yes".into(),
                    ok,
                });
            }
        }
        b
    }

    fn a_label_file(rng: &mut Rng) -> LabelFile {
        let mut boxes: Vec<LabelBox> = (0..rng.below(6)).map(|_| a_box(rng)).collect();
        let mut suggestions = Vec::new();
        if !boxes.is_empty() && rng.one_in(2) {
            // A pair a suggestion is about: one object, a box nested in another.
            boxes[0].bbox = PLACES[0];
            let mut twin = boxes[0].clone();
            twin.bbox = rng.pick(&[PLACES[1], PLACES[2]]);
            twin.state = BoxState::Pending;
            twin.orig_bbox = None;
            twin.model_label = None;
            twin.source = Source::Auto;
            boxes.insert(1, twin);
            boxes[0].state = rng.pick(&[BoxState::Pending, BoxState::Accepted]);
        }
        if boxes.len() >= 2 && rng.one_in(2) {
            let verdict = rng.pick(&[
                Verdict::DropA,
                Verdict::DropB,
                Verdict::TwoObjects,
                Verdict::Unclear,
            ]);
            let low = rng.one_in(4);
            suggestions.push(Suggestion {
                a: boxes[0].bbox,
                b: boxes[1].bbox,
                verdict,
                why: "x".into(),
                kept_score: (!low).then(|| rng.score()),
                other_score: (!low).then(|| rng.score()),
                low_confidence: low,
                weak: rng.one_in(4),
            });
        }
        // From a generator of their own, so the folders of the other cases stay as they were.
        let mut own = Rng::new(rng.0 ^ 0x5eed);
        let refine = if own.one_in(2) {
            suggestions_for(&mut own, &boxes)
        } else {
            Vec::new()
        };
        let status = rng.pick(&[
            Status::Unreviewed,
            Status::Unreviewed,
            Status::Reviewed,
            Status::Reviewed,
            Status::Skipped,
        ]);
        let repeat_spot = rng.one_in(4);
        if repeat_spot && status == Status::Unreviewed && rng.one_in(2) {
            // Spot edits settled every box: a repeat-only frame.
            for b in &mut boxes {
                b.state = BoxState::Accepted;
                b.auto.vlm = None;
            }
        }
        LabelFile {
            status,
            empty_confirmed: rng.one_in(8),
            boxes,
            reviewed_at: (status != Status::Unreviewed || rng.one_in(6))
                .then(|| "2026-10-22T00:00:00Z".into()),
            disagreement: rng
                .one_in(6)
                .then(|| rng.pick(&["model m: 1 new box", "sam3: 2 boxes"]).into()),
            auto_run: rng.one_in(3).then(|| AutoRun {
                at: "2026-10-22T00:00:00Z".into(),
                labeler: "gdino + rfdetr".into(),
                classes: Vec::new(),
            }),
            repeat_spot,
            overlaps_checked: rng.one_in(8),
            overlap_suggestions: suggestions,
            refine,
            ..LabelFile::default()
        }
    }

    /// Suggestions of the dataset check on `boxes`: every kind, on a box or at a place, some with
    /// the target's label, some answered, some stale, some of an unwanted class.
    fn suggestions_for(rng: &mut Rng, boxes: &[LabelBox]) -> Vec<refine::Suggestion> {
        let mut out = Vec::new();
        for _ in 0..1 + rng.below(3) {
            let kind = rng.pick(&[
                refine::Kind::Add,
                refine::Kind::Add,
                refine::Kind::Tighten,
                refine::Kind::Tighten,
                refine::Kind::Relabel,
                refine::Kind::Relabel,
                refine::Kind::Relabel,
                refine::Kind::Verify,
                refine::Kind::Verify,
            ]);
            let on = (!boxes.is_empty() && !rng.one_in(6)).then(|| rng.pick(boxes));
            let target = on.as_ref().map_or_else(|| rng.pick(&PLACES), |b| b.bbox);
            let add = kind == refine::Kind::Add;
            out.push(refine::Suggestion {
                kind,
                target: (!add).then_some(target),
                target_label: (!add && rng.one_in(2)).then(|| {
                    on.as_ref()
                        .map_or_else(|| "car".into(), |b| b.label.clone())
                }),
                bbox: match kind {
                    refine::Kind::Add => Some(rng.pick(&PLACES)),
                    refine::Kind::Tighten => Some(rng.pick(&PLACES)),
                    _ => None,
                },
                label: matches!(kind, refine::Kind::Add | refine::Kind::Relabel)
                    .then(|| rng.pick(&CLASSES).into()),
                score: (!rng.one_in(5)).then(|| rng.score()),
                why: "x".into(),
                spot: None,
                answer: rng
                    .one_in(5)
                    .then(|| rng.pick(&[refine::Answer::Dismissed, refine::Answer::Kept])),
                run: "r1".into(),
            });
        }
        out
    }

    /// Label files every folder has, whatever the seed: what a random one rarely is.
    fn anchors() -> Vec<LabelFile> {
        let b = |state: BoxState, audit: bool| LabelBox {
            label: "car".into(),
            bbox: PLACES[3],
            source: Source::Auto,
            score: Some(0.3),
            state,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: AutoInfo {
                audit,
                ..AutoInfo::default()
            },
        };
        let reviewed = LabelFile {
            status: Status::Reviewed,
            reviewed_at: Some("2026-10-22T00:00:00Z".into()),
            boxes: vec![
                b(BoxState::Rejected, true),
                LabelBox {
                    bbox: PLACES[4],
                    ..b(BoxState::Accepted, true)
                },
                LabelBox {
                    bbox: PLACES[0],
                    ..b(BoxState::Rejected, false)
                },
            ],
            ..LabelFile::default()
        };
        let mut agreed = b(BoxState::Pending, false);
        agreed.auto.labelers = [("gdino".to_string(), 0.4), ("rfdetr".to_string(), 0.5)].into();
        let both = LabelFile {
            boxes: vec![
                agreed.clone(),
                LabelBox {
                    bbox: PLACES[4],
                    ..agreed
                },
            ],
            ..LabelFile::default()
        };
        let mut sure = b(BoxState::Pending, false);
        sure.auto.triage = Some(Triage::Accept);
        let accept = LabelFile {
            boxes: vec![sure],
            ..LabelFile::default()
        };
        // One live suggestion of every kind, and a frame where only a verify is.
        let s = |kind: refine::Kind, target: Option<[f32; 4]>, bbox, label: Option<&str>| {
            refine::Suggestion {
                kind,
                target,
                target_label: None,
                bbox,
                label: label.map(String::from),
                score: Some(0.6),
                why: "x".into(),
                spot: None,
                answer: None,
                run: "r1".into(),
            }
        };
        let checked = LabelFile {
            boxes: vec![
                b(BoxState::Accepted, false),
                LabelBox {
                    label: "person".into(),
                    bbox: PLACES[4],
                    ..b(BoxState::Accepted, false)
                },
            ],
            refine: vec![
                s(refine::Kind::Add, None, Some(PLACES[0]), Some("dog")),
                s(
                    refine::Kind::Tighten,
                    Some(PLACES[3]),
                    Some(PLACES[1]),
                    None,
                ),
                s(refine::Kind::Relabel, Some(PLACES[4]), None, Some("dog")),
                s(refine::Kind::Verify, Some(PLACES[3]), None, None),
            ],
            ..reviewed.clone()
        };
        let verify_only = LabelFile {
            refine: vec![s(refine::Kind::Verify, Some(PLACES[3]), None, None)],
            ..checked.clone()
        };
        vec![
            reviewed.clone(),
            reviewed,
            both,
            accept.clone(),
            accept,
            // Copies: strength ties only the timestamp and the path break.
            checked.clone(),
            checked.clone(),
            checked,
            verify_only,
        ]
    }

    fn review_line(rel: &str, scene: &str, camera: &str, buckets: &[Bucket]) -> String {
        serde_json::json!({
            "version": 1, "frame": rel, "sidecar": rel.replace(".jpg", ".json"), "id": "x",
            "buckets": buckets, "scene": scene, "camera": camera,
            "queued_at": "2026-10-21T00:00:00Z",
        })
        .to_string()
    }

    fn index_line(r: &Record) -> String {
        serde_json::to_string(&IndexLine::of(r, 1)).unwrap()
    }

    /// One frame on disk with its sidecar (both, unless a case takes one away), and maybe a label
    /// file; its sidecar.
    fn put(dir: &Path, rng: &mut Rng, rel: &str, anchor: Option<&LabelFile>) -> Record {
        let r = layout::parse_rel(rel).unwrap();
        let day = dir.join(&r.scene).join(&r.camera).join(&r.date);
        std::fs::create_dir_all(&day).unwrap();
        let rec = sidecar(rng, &r.scene, &r.camera, &r.stem);
        let jpg = day.join(format!("{}.jpg", r.stem));
        let json = day.join(format!("{}.json", r.stem));
        let labels = day.join(format!("{}{LABELS_SUFFIX}", r.stem));
        std::fs::write(&jpg, b"x").unwrap();
        std::fs::write(&json, serde_json::to_vec(&rec).unwrap()).unwrap();
        if let Some(f) = anchor {
            std::fs::write(&labels, serde_json::to_vec(f).unwrap()).unwrap();
            return rec;
        }
        match rng.below(20) {
            0..=4 => {}
            5 => std::fs::write(&labels, b"{ not json").unwrap(),
            6 => std::fs::write(&json, b"{ not json").unwrap(),
            7 => std::fs::remove_file(&jpg).unwrap(),
            8 => {
                std::fs::remove_file(&json).unwrap();
                std::fs::write(&labels, b"{}").unwrap();
            }
            _ => {
                let f = a_label_file(rng);
                std::fs::write(&labels, serde_json::to_vec(&f).unwrap()).unwrap();
            }
        }
        rec
    }

    fn rel_of(i: usize, rng: &mut Rng) -> String {
        let (scene, camera, date) = (rng.pick(&SCENES), rng.pick(&CAMERAS), rng.pick(&DATES));
        format!("{scene}/{camera}/{date}/20261021T{i:06}000Z_f{i}.jpg")
    }

    /// A folder of `n` frames with every case the walk meets (see the module doc), `foreign`:
    /// with index lines naming frames of another folder. The frames it made.
    fn generate(dir: &Path, seed: u64, n: usize, foreign: bool) -> Vec<String> {
        let mut rng = Rng::new(seed);
        let mut rels = Vec::new();
        let mut index: BTreeMap<PathBuf, String> = BTreeMap::new();
        let mut queue: Vec<String> = Vec::new();
        let anchors = anchors();
        for i in 0..n + anchors.len() {
            let rel = rel_of(i, &mut rng);
            let rec = put(dir, &mut rng, &rel, i.checked_sub(n).map(|a| &anchors[a]));
            let r = layout::parse_rel(&rel).unwrap();
            let day = dir.join(&r.scene).join(&r.camera).join(&r.date);
            if !rng.one_in(10) {
                let text = index.entry(day.join(INDEX_FILE)).or_default();
                text.push_str(&index_line(&rec));
                text.push('\n');
            }
            if rng.one_in(2) {
                // A review line may disagree with the path it names, and repeat.
                let (scene, camera) = match rng.below(6) {
                    0 => ("day", "elsewhere"),
                    1 => (r.scene.as_str(), "elsewhere"),
                    _ => (r.scene.as_str(), r.camera.as_str()),
                };
                let buckets: Vec<Bucket> =
                    (0..rng.below(3)).map(|_| rng.pick(&Bucket::ALL)).collect();
                queue.push(review_line(&rel, scene, camera, &buckets));
                if rng.one_in(10) {
                    queue.push(review_line(&rel, "night", "again", &[]));
                }
            }
            rels.push(rel);
        }
        // Damaged lines, a day folder with no index file, a review line for a frame long gone and
        // one naming a sidecar.
        if let Some((path, text)) = index.iter_mut().next() {
            text.push_str("{ damaged\n");
            let _ = path;
        }
        if let Some(path) = index.keys().nth(1).cloned() {
            index.remove(&path);
        }
        if foreign {
            // Lines naming a frame of another camera folder, existing or not.
            if let Some(text) = index.values_mut().next() {
                let mut rec = empty_record();
                let other = rels.iter().find(|r| r.contains("/porch/")).cloned();
                if let Some(other) = other {
                    let o = layout::parse_rel(&other).unwrap();
                    rec.id = o.stem.clone();
                    rec.scene = o.scene.clone();
                    rec.camera = "porch".into();
                    text.push_str(&index_line(&rec));
                    text.push('\n');
                }
                rec.id = "20261021T999999000Z_nowhere".into();
                rec.camera = "nowhere".into();
                text.push_str(&index_line(&rec));
                text.push('\n');
            }
        }
        queue.push(review_line(
            "ir/gate/2026-01-01/20260101T000000000Z_gone.jpg",
            "ir",
            "lost-camera",
            &[Bucket::Positive],
        ));
        queue.push(review_line(
            &rels[1].replace(".jpg", ".json"),
            "ir",
            "sidecar-line",
            &[],
        ));
        queue.push("{ damaged".into());
        for i in (1..queue.len()).rev() {
            queue.swap(i, rng.below(i + 1));
        }
        for (path, text) in index {
            std::fs::write(path, text).unwrap();
        }
        let text: String = queue.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(dir.join(REVIEW_QUEUE_FILE), text).unwrap();
        rels
    }

    /// The filters the comparison runs: every source x status x auto filter, and each other
    /// filter on its own, with `wanted`.
    fn grid(wanted: &(Vec<String>, Vec<String>)) -> Vec<Filters> {
        let autos = [
            None,
            Some(AutoFilter::Review),
            Some(AutoFilter::Accept),
            Some(AutoFilter::Nothing),
            Some(AutoFilter::Never),
            Some(AutoFilter::Repeats),
            Some(AutoFilter::Overlaps),
            Some(AutoFilter::Both),
            Some(AutoFilter::Dismissed),
            Some(AutoFilter::Refine),
        ];
        let statuses = [
            StatusFilter::Is(Status::Unreviewed),
            StatusFilter::Is(Status::Reviewed),
            StatusFilter::Is(Status::Skipped),
            StatusFilter::Any,
        ];
        let base = Filters {
            wanted: Some(wanted.clone()),
            ..Filters::default()
        };
        let mut out = Vec::new();
        for source in [QueueSource::Queue, QueueSource::All] {
            for status in statuses {
                if source == QueueSource::Queue
                    && matches!(status, StatusFilter::Is(Status::Reviewed | Status::Skipped))
                {
                    continue;
                }
                for auto in autos {
                    out.push(Filters {
                        source,
                        status,
                        auto,
                        ..base.clone()
                    });
                }
            }
        }
        let all = Filters {
            source: QueueSource::All,
            status: StatusFilter::Any,
            ..base
        };
        for b in Bucket::ALL {
            out.push(Filters {
                bucket: Some(b),
                ..all.clone()
            });
        }
        for scene in SCENES {
            out.push(Filters {
                scene: Some(scene.into()),
                ..all.clone()
            });
        }
        for camera in ["gate", "elsewhere", "lost-camera", "nope"] {
            out.push(Filters {
                camera: Some(camera.into()),
                ..all.clone()
            });
        }
        for labels in [&["person"][..], &["car", "dog"], &["zzz"], &["plant"]] {
            out.push(Filters {
                label: labels.iter().map(|l| (*l).to_string()).collect(),
                ..all.clone()
            });
        }
        out.push(Filters {
            source: QueueSource::Queue,
            camera: Some("elsewhere".into()),
            scene: Some("day".into()),
            label: ["person".to_string()].into(),
            auto: Some(AutoFilter::Overlaps),
            ..all.clone()
        });
        for kind in REFINE_KINDS {
            out.push(Filters {
                auto: Some(AutoFilter::Refine),
                refine_kind: Some(kind),
                ..all.clone()
            });
        }
        out.push(Filters {
            status: StatusFilter::Is(Status::Unreviewed),
            camera: Some("gate".into()),
            auto: Some(AutoFilter::Refine),
            refine_kind: Some(refine::Kind::Verify),
            ..all.clone()
        });
        out
    }

    const REFINE_KINDS: [refine::Kind; 4] = [
        refine::Kind::Add,
        refine::Kind::Tighten,
        refine::Kind::Relabel,
        refine::Kind::Verify,
    ];

    fn json(a: &Answer) -> serde_json::Value {
        serde_json::to_value(a).unwrap()
    }

    /// The index's answer, which must be complete.
    fn from_index(dir: &Path, f: &Filters, after: Option<&str>) -> Answer {
        let a = super::super::answer(dir, f, after).expect("the folder is indexed");
        assert_eq!(
            a.indexing, None,
            "{f:?}: a forced answer waits for the build"
        );
        a
    }

    /// For every filter of `grid(wanted)` and several `after`s, the index answers what the walk
    /// does, compared whole. `rels`: frames to try as `after` too.
    fn same_answers(dir: &Path, wanted: &(Vec<String>, Vec<String>), rels: &[String], round: &str) {
        let filters = grid(wanted);
        // The walks are the slow part: on threads, as the route's queries come.
        crate::parallel::map_all(filters.len(), 8, &|i| {
            let f = &filters[i];
            let built = queue::build(dir, f);
            let damaged = layout::damaged_lines(dir, f.source == QueueSource::All);
            let listed: Vec<&str> = built.items.iter().map(|i| i.frame.as_str()).collect();
            let mut afters: Vec<Option<String>> = vec![None, Some("junk".into())];
            for at in [
                0,
                1,
                listed.len() / 2,
                listed.len().saturating_sub(2),
                listed.len().saturating_sub(1),
            ] {
                afters.extend(listed.get(at).map(|r| Some((*r).to_string())));
            }
            // Frames not listed: of other filters, gone, unreadable.
            afters.extend(
                rels.iter()
                    .filter(|r| !listed.contains(&r.as_str()))
                    .take(2)
                    .map(|r| Some(r.clone())),
            );
            afters.dedup();
            for after in afters {
                let want = Answer::of(built.clone(), damaged, after.as_deref());
                let got = from_index(dir, f, after.as_deref());
                assert_eq!(json(&got), json(&want), "{round}: {f:?} after {after:?}");
            }
        });
    }

    /// The folder holds what the comparison is meant to cover: every auto filter lists frames,
    /// there are unreadable frames, gone frames whose camera still counts, ties only the path
    /// breaks, queued and unqueued frames, every status, notes and frames of several pages.
    fn covers_every_case(dir: &Path) {
        let all = all_any();
        let built = queue::build(dir, &all);
        assert!(built.items.len() > 60, "{}", built.items.len());
        assert!(built.unreadable >= 3, "{}", built.unreadable);
        assert!(built.cameras.iter().any(|c| c == "lost-camera"));
        assert!(built.cameras.iter().any(|c| c == "elsewhere"));
        for status in [Status::Unreviewed, Status::Reviewed, Status::Skipped] {
            assert!(built.items.iter().any(|i| i.status == status), "{status:?}");
        }
        assert!(built.items.iter().any(|i| i.queued) && built.items.iter().any(|i| !i.queued));
        assert!(built.items.iter().any(|i| i.uncertainty.relabel.is_some()));
        assert!(built.items.iter().any(|i| i.uncertainty.likely_boxes > 0));
        assert!(built.items.iter().any(|i| i.uncertainty.repeats_only));
        assert!(built.items.iter().any(|i| i.uncertainty.disagreement));
        assert!(built.items.iter().any(|i| i.uncertainty.margin.is_none()));
        assert!(built.counts.rejected_boxes > 0 && built.counts.empty_confirmed > 0);
        assert!(
            built.counts.audit_accepted > 0 && built.counts.audit_rejected > 0,
            "{:?}",
            built.counts
        );
        assert!(!built.counts.per_class.is_empty());
        let ties = built.items.windows(2).any(|w| {
            w[0].uncertainty.key() == w[1].uncertainty.key() && w[0].timestamp == w[1].timestamp
        });
        assert!(ties, "no tie for the path to break");
        for auto in [
            AutoFilter::Review,
            AutoFilter::Accept,
            AutoFilter::Nothing,
            AutoFilter::Never,
            AutoFilter::Repeats,
            AutoFilter::Overlaps,
            AutoFilter::Both,
            AutoFilter::Dismissed,
            AutoFilter::Refine,
        ] {
            let f = Filters {
                auto: Some(auto),
                ..all.clone()
            };
            let items = queue::build(dir, &f).items;
            assert!(!items.is_empty(), "nothing for auto={auto:?}");
            if auto == AutoFilter::Overlaps {
                assert!(items.iter().any(|i| i.uncertainty.overlap_strength > 0));
            }
            if auto == AutoFilter::Both {
                assert!(items.iter().any(|i| i.uncertainty.agreed_boxes > 1));
            }
            if auto == AutoFilter::Refine {
                assert!(items.iter().all(|i| i.uncertainty.refine_open > 0));
                assert!(items.iter().any(|i| i.uncertainty.refine_open > 1));
                // Unreviewed, reviewed and skipped-but-stale frames: a skipped one is never listed.
                assert!(items.iter().any(|i| i.status == Status::Unreviewed));
                assert!(items.iter().all(|i| i.status != Status::Skipped));
                // Strength ties for the path to break.
                assert!(
                    items.windows(2).any(|w| {
                        w[0].uncertainty.refine_strength == w[1].uncertainty.refine_strength
                    }),
                    "{:?}",
                    items
                        .iter()
                        .map(|i| (&i.frame, i.uncertainty.refine_strength))
                        .collect::<Vec<_>>()
                );
            }
        }
        for kind in REFINE_KINDS {
            let f = Filters {
                auto: Some(AutoFilter::Refine),
                refine_kind: Some(kind),
                ..all.clone()
            };
            assert!(
                queue::build(dir, &f).items.len() >= 2,
                "too few for refine_kind={kind:?}"
            );
        }
        assert!(layout::damaged_lines(dir, true) >= 2);
    }

    /// Next after next through the whole queue: the index's order is the walk's, ties included.
    fn same_order(dir: &Path, f: &Filters, round: &str) {
        let built = queue::build(dir, f);
        let mut after: Option<String> = None;
        for (at, want) in built.items.iter().enumerate() {
            let got = from_index(dir, f, after.as_deref());
            let item = got
                .item
                .unwrap_or_else(|| panic!("{round}: nothing at {at}"));
            assert_eq!(item.frame, want.frame, "{round}: place {at} of {f:?}");
            assert_eq!(got.position, Some(at + 1), "{round}");
            after = Some(item.frame);
        }
        let last = from_index(dir, f, after.as_deref());
        assert_eq!(last.wrapped, !built.items.is_empty(), "{round}: wraps");
    }

    /// Frames that exist now.
    fn existing(dir: &Path, rels: &[String]) -> Vec<String> {
        rels.iter()
            .filter(|r| Frame::resolve(dir, r).is_some())
            .cloned()
            .collect()
    }

    /// Writes through the real writers: label edits, new label files, the bulk actions, review
    /// lines, a new frame and retention.
    fn real_writes(
        dir: &Path,
        rng: &mut Rng,
        rels: &mut Vec<String>,
        wanted: &(Vec<String>, Vec<String>),
    ) {
        let live = existing(dir, rels);
        for _ in 0..12 {
            let rel = rng.pick(&live);
            let frame = Frame::resolve(dir, &rel).unwrap();
            let fresh = a_label_file(rng);
            let _ = super::super::super::update(&frame.labels, |old| {
                Ok(Some(match old {
                    Some(mut l) if !fresh.boxes.is_empty() => {
                        l.status = fresh.status;
                        l.boxes.extend(fresh.boxes);
                        l.disagreement = fresh.disagreement;
                        l
                    }
                    _ => fresh,
                }))
            });
        }
        let rel = rng.pick(&live);
        let frame = Frame::resolve(dir, &rel).unwrap();
        super::super::super::write(&frame.labels, &a_label_file(rng)).unwrap();
        let w = Wanted {
            classes: &wanted.0,
            hidden: &wanted.1,
        };
        refine_writes(dir, rng, &live, &w);
        let all = Filters {
            source: QueueSource::All,
            status: StatusFilter::Any,
            wanted: Some(wanted.clone()),
            ..Filters::default()
        };
        bulk::bulk_reject_below(dir, &all, 0.4, usize::MAX, false, &|_| true);
        bulk::bulk_reject_stacked(dir, &all, &w, usize::MAX, false, &|_| true);
        bulk::bulk_accept(
            dir,
            &all,
            &wanted.0,
            5,
            "2026-10-23T00:00:00Z",
            false,
            &|_| true,
        );
        let unqueued: Vec<String> = live
            .iter()
            .filter(|r| Frame::resolve(dir, r).is_some_and(|f| f.record().is_ok()))
            .filter(|r| {
                !layout::read_review(&dir.join(REVIEW_QUEUE_FILE))
                    .iter()
                    .any(|l| &l.frame == *r)
            })
            .take(3)
            .cloned()
            .collect();
        layout::append_review(dir, &unqueued).unwrap();
        // A new frame, in a day the rows have and in a new day, queued and labelled.
        for (i, date) in [(9001, "2026-10-21"), (9002, "2026-11-01")] {
            let rel = format!("night/gate/{date}/20261021T{i:06}000Z_f{i}.jpg");
            let r = layout::parse_rel(&rel).unwrap();
            let day = dir.join(&r.scene).join(&r.camera).join(&r.date);
            std::fs::create_dir_all(&day).unwrap();
            let rec = sidecar(rng, "night", "gate", &r.stem);
            std::fs::write(day.join(format!("{}.jpg", r.stem)), b"x").unwrap();
            std::fs::write(
                day.join(format!("{}.json", r.stem)),
                serde_json::to_vec(&rec).unwrap(),
            )
            .unwrap();
            layout::append_review(dir, std::slice::from_ref(&rel)).unwrap();
            let frame = Frame::resolve(dir, &rel).unwrap();
            super::super::super::write(&frame.labels, &a_label_file(rng)).unwrap();
            rels.push(rel);
        }
        // Retention: the unprotected frames go, through `unless_protected` and `rewrite_index`.
        crate::retention::prune_dataset(dir, 0, 1).unwrap();
    }

    /// What the dataset check and a reviewer do to suggestions, through the label writer: a check
    /// writes some (only `refine` changes), a reviewer dismisses one, accepts one (a box changes,
    /// or a verify is answered) and a rerun takes the unanswered ones away.
    fn refine_writes(dir: &Path, rng: &mut Rng, live: &[String], w: &Wanted<'_>) {
        let kinds = [
            refine::Kind::Add,
            refine::Kind::Tighten,
            refine::Kind::Relabel,
            refine::Kind::Verify,
        ];
        let labelled: Vec<Frame> = live
            .iter()
            .filter_map(|r| Frame::resolve(dir, r))
            .filter(|f| f.labels().is_ok_and(|l| l.is_some()))
            .collect();
        for (n, frame) in labelled.iter().take(16).enumerate() {
            let found = {
                let l = frame.labels().unwrap().unwrap();
                suggestions_for(rng, &l.boxes)
                    .into_iter()
                    .map(|s| refine::Suggestion { answer: None, ..s })
                    .collect::<Vec<_>>()
            };
            let _ = super::super::super::update(&frame.labels, |old| {
                let Some(mut l) = old else { return Ok(None) };
                let before = l.clone();
                match n % 4 {
                    // A check writes what it found.
                    0 | 1 => {
                        if !refine::apply(&mut l, found, &kinds).changed() {
                            return Ok(None);
                        }
                    }
                    // A reviewer answers no.
                    2 => {
                        let Some(i) = refine::live(&l, Some(w)).first().map(|x| x.index) else {
                            return Ok(None);
                        };
                        assert!(refine::dismiss(&mut l, i));
                    }
                    // A reviewer answers yes.
                    _ => {
                        let Some(i) = refine::live(&l, Some(w)).last().map(|x| x.index) else {
                            return Ok(None);
                        };
                        assert!(refine::accept(&mut l, i, Some(w)));
                    }
                }
                assert_ne!(l, before);
                Ok(Some(l))
            });
        }
        // A rerun that finds nothing: the unanswered suggestions go, the answered ones stay.
        for frame in labelled.iter().skip(16).take(4) {
            let _ = super::super::super::update(&frame.labels, |old| {
                let Some(mut l) = old else { return Ok(None) };
                Ok(refine::apply(&mut l, Vec::new(), &kinds)
                    .changed()
                    .then_some(l))
            });
        }
    }

    /// Edits no writer of this process tells the index about: they show after a scan.
    fn direct_edits(dir: &Path, rng: &mut Rng, rels: &mut Vec<String>) {
        let live = existing(dir, rels);
        // A label file replaced by a rename, as another process writes them.
        for _ in 0..4 {
            let rel = rng.pick(&live);
            let frame = Frame::resolve(dir, &rel).unwrap();
            let tmp = frame.labels.with_extension("tmp");
            std::fs::write(&tmp, serde_json::to_vec(&a_label_file(rng)).unwrap()).unwrap();
            std::fs::rename(&tmp, &frame.labels).unwrap();
        }
        // A frame deleted by hand, and a new one the collector wrote (index line appended).
        let gone = rng.pick(&live);
        let frame = Frame::resolve(dir, &gone).unwrap();
        std::fs::remove_file(&frame.jpg).unwrap();
        let rel = "day/yard/2026-10-20/20261021T009100000Z_f9100.jpg".to_string();
        let r = layout::parse_rel(&rel).unwrap();
        let day = dir.join(&r.scene).join(&r.camera).join(&r.date);
        std::fs::create_dir_all(&day).unwrap();
        let rec = sidecar(rng, "day", "yard", &r.stem);
        std::fs::write(day.join(format!("{}.jpg", r.stem)), b"x").unwrap();
        std::fs::write(
            day.join(format!("{}.json", r.stem)),
            serde_json::to_vec(&rec).unwrap(),
        )
        .unwrap();
        let mut text = std::fs::read_to_string(day.join(INDEX_FILE)).unwrap_or_default();
        text.push_str(&index_line(&rec));
        text.push_str("\n{ damaged too\n");
        std::fs::write(day.join(INDEX_FILE), text).unwrap();
        rels.push(rel.clone());
        // A review line another process appended.
        let mut q = std::fs::read_to_string(dir.join(REVIEW_QUEUE_FILE)).unwrap();
        q.push_str(&review_line(&rel, "day", "elsewhere", &[Bucket::Empty]));
        q.push('\n');
        std::fs::write(dir.join(REVIEW_QUEUE_FILE), q).unwrap();
    }

    /// Edits in place: no folder time moves, only a walk of everything sees them.
    fn in_place_edits(dir: &Path, rng: &mut Rng, rels: &[String]) {
        let live = existing(dir, rels);
        for _ in 0..4 {
            let rel = rng.pick(&live);
            let frame = Frame::resolve(dir, &rel).unwrap();
            std::fs::write(
                &frame.labels,
                serde_json::to_vec(&a_label_file(rng)).unwrap(),
            )
            .unwrap();
            let rec = sidecar(rng, &frame.rel.scene, &frame.rel.camera, &frame.rel.stem);
            std::fs::write(&frame.sidecar, serde_json::to_vec(&rec).unwrap()).unwrap();
        }
    }

    /// Fewer review lines: the rows that lost theirs go back to their index line.
    fn shrink_queue(dir: &Path) {
        let path = dir.join(REVIEW_QUEUE_FILE);
        let text = std::fs::read_to_string(&path).unwrap();
        let kept: String = text.lines().step_by(2).map(|l| format!("{l}\n")).collect();
        std::fs::write(&path, kept).unwrap();
    }

    fn all_any() -> Filters {
        Filters {
            source: QueueSource::All,
            status: StatusFilter::Any,
            wanted: Some(wanted_a()),
            ..Filters::default()
        }
    }

    /// The test that matters: see the module doc.
    #[test]
    fn equivalence() {
        for seed in [1_u64, 2, 3] {
            let dir = Dir::new();
            set_clock_ahead(&dir, Duration::from_secs(3600));
            let foreign = seed == 3;
            let mut rels = generate(&dir, seed, 120, foreign);
            let mut rng = Rng::new(seed ^ 0xabc);
            let (a, b) = (wanted_a(), wanted_b());
            covers_every_case(&dir);
            same_answers(&dir, &a, &rels, &format!("seed {seed}, built"));
            same_order(&dir, &all_any(), &format!("seed {seed}, built"));
            let refine_all = Filters {
                auto: Some(AutoFilter::Refine),
                ..all_any()
            };
            same_order(&dir, &refine_all, &format!("seed {seed}, built, refine"));
            assert!(index_of(&dir).is_some());

            real_writes(&dir, &mut rng, &mut rels, &a);
            same_answers(&dir, &a, &rels, &format!("seed {seed}, real writes"));
            same_order(
                &dir,
                &refine_all,
                &format!("seed {seed}, real writes, refine"),
            );

            // Another class list: another index, built anew.
            let before = index_of(&dir).unwrap();
            same_answers(&dir, &b, &rels, &format!("seed {seed}, class change"));
            assert!(!Arc::ptr_eq(&before, &index_of(&dir).unwrap()));
            assert!(before.cancelled());
            // Back: built anew before the edits below, so that only a scan can see them.
            from_index(&dir, &all_any(), None);

            direct_edits(&dir, &mut rng, &mut rels);
            index_of(&dir).unwrap().pass(Pass::Scan, &|| {});
            same_answers(&dir, &a, &rels, &format!("seed {seed}, scan"));

            shrink_queue(&dir);
            index_of(&dir).unwrap().pass(Pass::Scan, &|| {});
            same_answers(&dir, &a, &rels, &format!("seed {seed}, queue shrunk"));

            in_place_edits(&dir, &mut rng, &rels);
            index_of(&dir).unwrap().pass(Pass::Full, &|| {});
            same_answers(&dir, &a, &rels, &format!("seed {seed}, full pass"));
            same_order(&dir, &all_any(), &format!("seed {seed}, full pass"));
        }
    }

    /// The candidates of `dir` by day folder of their path.
    fn by_day(dir: &Path) -> BTreeMap<String, usize> {
        let mut days = BTreeMap::new();
        for c in queue::candidates(dir, QueueSource::All) {
            *days.entry(queue::day_of(&c.rel).to_string()).or_default() += 1;
        }
        days
    }

    fn walk(dir: &Path, f: &Filters, after: Option<&str>) -> serde_json::Value {
        json(&Answer::walk(dir, f, after))
    }

    /// Make `day` look changed to a scan, also where folder times are coarse: its index file
    /// grows by a blank line, which no reader counts.
    fn change_day(day: &Path) {
        let path = day.join(INDEX_FILE);
        let mut text = std::fs::read_to_string(&path).unwrap_or_default();
        text.push('\n');
        std::fs::write(&path, text).unwrap();
    }

    /// A frame whose files all read, in `dir`.
    fn readable(dir: &Path, rels: &[String]) -> Frame {
        rels.iter()
            .filter_map(|r| Frame::resolve(dir, r))
            .find(|f| f.record().is_ok() && f.labels().is_ok())
            .unwrap()
    }

    #[test]
    fn a_write_during_a_walk_survives_its_swap() {
        for kind in [Pass::Scan, Pass::Full] {
            let dir = Dir::new();
            set_clock_ahead(&dir, Duration::from_secs(3600));
            let rels = generate(&dir, 7, 40, false);
            let f = all_any();
            from_index(&dir, &f, None);
            let frame = readable(&dir, &rels);
            // The scan walks the frame's day: its folder changed.
            change_day(frame.jpg.parent().unwrap());
            let index = index_of(&dir).unwrap();
            let marked = LabelFile {
                status: Status::Skipped,
                disagreement: Some("written during the walk".into()),
                boxes: vec![LabelBox {
                    label: "truck".into(),
                    ..super::super::super::tests_support::auto_box("truck")
                }],
                ..LabelFile::default()
            };
            let ran = std::sync::atomic::AtomicBool::new(false);
            // Between the walk, which read the old file, and its swap: write, then ask.
            index.pass(kind, &|| {
                super::super::super::write(&frame.labels, &marked).unwrap();
                assert_eq!(
                    json(&from_index(&dir, &f, None)),
                    walk(&dir, &f, None),
                    "{kind:?}"
                );
                ran.store(true, Ordering::Release);
            });
            assert!(ran.load(Ordering::Acquire));
            assert_eq!(
                json(&from_index(&dir, &f, None)),
                walk(&dir, &f, None),
                "{kind:?}, after"
            );
            let truck = Filters {
                label: ["truck".to_string()].into(),
                ..f.clone()
            };
            assert_eq!(
                json(&from_index(&dir, &truck, None)),
                walk(&dir, &truck, None)
            );
        }
    }

    #[test]
    fn a_scan_walks_only_what_changed() {
        let dir = Dir::new();
        set_clock_ahead(&dir, Duration::from_secs(3600));
        let rels = generate(&dir, 11, 60, false);
        let f = all_any();
        from_index(&dir, &f, None);
        let index = index_of(&dir).unwrap();
        let walked = frames_walked(&dir);
        index.pass(Pass::Scan, &|| {});
        assert_eq!(frames_walked(&dir), walked, "nothing changed");
        // A new frame in one day folder, and its index line.
        let frame = readable(&dir, &rels);
        let day = frame.jpg.parent().unwrap().to_path_buf();
        let stem = "20261021T777777000Z_new";
        let mut rec = empty_record();
        rec.id = stem.into();
        rec.scene = frame.rel.scene.clone();
        rec.camera = frame.rel.camera.clone();
        rec.timestamp = "2026-10-21T00:00:09Z".into();
        std::fs::write(day.join(format!("{stem}.jpg")), b"x").unwrap();
        std::fs::write(
            day.join(format!("{stem}.json")),
            serde_json::to_vec(&rec).unwrap(),
        )
        .unwrap();
        let mut text = std::fs::read_to_string(day.join(INDEX_FILE)).unwrap_or_default();
        text.push_str(&index_line(&rec));
        text.push('\n');
        std::fs::write(day.join(INDEX_FILE), text).unwrap();
        let key = format!(
            "{}/{}/{}",
            frame.rel.scene, frame.rel.camera, frame.rel.date
        );
        let walked = frames_walked(&dir);
        index.pass(Pass::Scan, &|| {});
        assert_eq!(
            frames_walked(&dir) - walked,
            by_day(&dir)[&key] as u64,
            "only {key}"
        );
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
        // A full pass walks everything.
        let walked = frames_walked(&dir);
        index.pass(Pass::Full, &|| {});
        let all = queue::candidates(&dir, QueueSource::All).len() as u64;
        assert_eq!(frames_walked(&dir) - walked, all);
    }

    #[test]
    fn a_foreign_index_line_makes_every_scan_a_full_pass() {
        let dir = Dir::new();
        set_clock_ahead(&dir, Duration::from_secs(3600));
        let rels = generate(&dir, 5, 40, true);
        let f = all_any();
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
        let index = index_of(&dir).unwrap();
        assert!(index.state.read_recover().foreign);
        // One day changes, but a line of another day may name a frame of it: all is walked. The
        // day changed holds no foreign line itself, so only the index's own mark can say so.
        let own_lines_only = |day: &Path| {
            let names: Vec<String> = day
                .iter()
                .rev()
                .skip(1)
                .take(2)
                .map(|c| c.to_string_lossy().into_owned())
                .collect();
            layout::read_index(&day.join(INDEX_FILE))
                .iter()
                .all(|l| names == [l.camera.clone(), l.scene.clone()])
        };
        let frame = rels
            .iter()
            .filter_map(|r| Frame::resolve(&dir, r))
            .find(|f| {
                f.record().is_ok() && f.labels().is_ok() && own_lines_only(f.jpg.parent().unwrap())
            })
            .expect("a day with no foreign line");
        change_day(frame.jpg.parent().unwrap());
        let walked = frames_walked(&dir);
        index.pass(Pass::Scan, &|| {});
        let all = queue::candidates(&dir, QueueSource::All).len() as u64;
        assert_eq!(frames_walked(&dir) - walked, all);
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
    }

    #[test]
    fn a_query_during_the_build_answers_from_the_rows_so_far_and_moves_on() {
        let dir = Dir::new();
        // Under 32 frames: one thread walks, a day at a time, newest first.
        generate(&dir, 13, 24, false);
        let days = by_day(&dir);
        let date = |k: &str| k.rsplit('/').next().unwrap().to_string();
        let (first, n) = days
            .iter()
            .max_by_key(|(k, _)| (date(k), (*k).clone()))
            .map(|(k, n)| (k.clone(), *n))
            .unwrap();
        let total: usize = days.values().sum();
        let f = all_any();
        let hold = hold_after(&dir, n as u64);
        let started = Instant::now();
        let _ = super::super::answer(&dir, &f, None).unwrap();
        assert!(hold.reached(), "the build stops after the first day");
        let a = super::super::answer(&dir, &f, None).unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "a held build does not hold queries"
        );
        assert_eq!(
            a.indexing,
            Some(Indexing {
                done: n as u64,
                total: total as u64
            })
        );
        let v = json(&a);
        assert_eq!(
            v["indexing"],
            serde_json::json!({"done": n, "total": total})
        );
        // Only that day's frames, and Next goes round them.
        let item = a.item.clone().expect("the first day has a frame to show");
        assert!(item.frame.starts_with(&first), "{} of {first}", item.frame);
        let mut after = item.frame.clone();
        for _ in 0..a.total + 1 {
            let next = super::super::answer(&dir, &f, Some(&after)).unwrap();
            let shown = next.item.expect("Next always shows a frame");
            assert!(shown.frame.starts_with(&first));
            after = shown.frame;
        }
        // A frame the rows do not have yet: the first again.
        let elsewhere = super::super::answer(&dir, &f, Some("ir/gate/2026-10-20/x.jpg")).unwrap();
        assert_eq!(elsewhere.position, Some(1));
        drop(hold);
        let done = from_index(&dir, &f, None);
        assert_eq!(json(&done), walk(&dir, &f, None));
        assert!(json(&done).get("indexing").is_none());
    }

    #[test]
    fn with_nothing_walked_yet_an_answer_says_how_much_is_to_come() {
        let dir = Dir::new();
        generate(&dir, 17, 20, false);
        let total = queue::candidates(&dir, QueueSource::All).len() as u64;
        let f = all_any();
        let hold = hold(&dir);
        // The first answer, which may come before the thread read a line, knows the total.
        let first = super::super::answer(&dir, &f, None).unwrap();
        assert_eq!(first.indexing, Some(Indexing { done: 0, total }));
        assert!(hold.reached());
        let a = super::super::answer(&dir, &f, None).unwrap();
        assert_eq!(a.indexing, Some(Indexing { done: 0, total }));
        assert!(a.item.is_none() && a.total == 0);
        hold.release();
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
    }

    #[test]
    fn an_idle_index_goes_with_its_thread() {
        let dir = Dir::new();
        generate(&dir, 19, 10, false);
        // Longer than a build of these 20 frames takes while the other tests load the machine: an
        // index whose build outlasts it is idle before its first answer, which then walks.
        set_timing(&dir, |t| t.idle = Duration::from_secs(1));
        let f = all_any();
        from_index(&dir, &f, None);
        let first = index_of(&dir).unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        while index_of(&dir).is_some() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(index_of(&dir).is_none(), "dropped when idle");
        assert!(first.cancelled());
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
        assert!(!Arc::ptr_eq(&first, &index_of(&dir).unwrap()), "a new one");
    }

    #[test]
    fn over_the_ceiling_the_folder_walks_until_another_folder_is_chosen() {
        let (dir, other) = (Dir::new(), Dir::new());
        generate(&dir, 23, 20, false);
        generate(&other, 29, 10, false);
        force_together(&[&dir, &other]);
        set_timing(&dir, |t| t.ceiling = 1);
        let f = all_any();
        assert!(super::super::answer(&dir, &f, None).is_none(), "walk");
        assert!(index_of(&dir).is_none());
        let walked = frames_walked(&dir);
        assert!(super::super::answer(&dir, &f, None).is_none());
        assert_eq!(frames_walked(&dir), walked, "not tried again");
        // Another folder, then this one again: it is tried again.
        assert!(super::super::answer(&other, &f, None).is_some());
        set_timing(&dir, |t| t.ceiling = INDEX_CEILING);
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
        assert!(index_of(&other).is_none(), "one folder at a time");
    }

    const INDEX_CEILING: usize = super::super::INDEX_CEILING_BYTES;

    #[test]
    fn a_panic_in_the_build_leaves_the_folder_walking() {
        let dir = Dir::new();
        generate(&dir, 31, 20, false);
        panic_at(&dir, 2);
        let f = all_any();
        let started = Instant::now();
        assert!(super::super::answer(&dir, &f, None).is_none());
        assert!(started.elapsed() < Duration::from_secs(10), "no hang");
        assert!(index_of(&dir).is_none());
        let walked = frames_walked(&dir);
        assert!(
            super::super::answer(&dir, &f, None).is_none(),
            "walks for now"
        );
        assert_eq!(frames_walked(&dir), walked);
        let hub = HUB.lock_recover();
        let refused = hub.refused.iter().find(|r| r.dir == *dir).unwrap();
        assert!(refused.until.is_some(), "for a while, not for good");
    }

    #[test]
    fn a_folder_switch_drops_the_other_index() {
        let (a, b) = (Dir::new(), Dir::new());
        generate(&a, 37, 10, false);
        generate(&b, 41, 10, false);
        force_together(&[&a, &b]);
        let f = all_any();
        assert_eq!(json(&from_index(&a, &f, None)), walk(&a, &f, None));
        let first = index_of(&a).unwrap();
        assert_eq!(json(&from_index(&b, &f, None)), walk(&b, &f, None));
        assert!(first.cancelled() && index_of(&a).is_none());
        assert_eq!(json(&from_index(&a, &f, None)), walk(&a, &f, None));
        assert!(index_of(&b).is_none());
    }

    #[test]
    fn a_folder_that_went_drops_its_index() {
        let dir = Dir::new();
        generate(&dir, 43, 10, false);
        let f = all_any();
        from_index(&dir, &f, None);
        std::fs::remove_dir_all(&*dir).unwrap();
        assert!(super::super::answer(&dir, &f, None).is_none());
        assert!(index_of(&dir).is_none());
    }

    #[test]
    fn more_writes_than_the_dirty_set_holds_go_stale_and_are_walked() {
        let dir = Dir::new();
        let rels = generate(&dir, 47, 40, false);
        let f = all_any();
        from_index(&dir, &f, None);
        let frame = readable(&dir, &rels);
        let day = frame.labels.parent().unwrap().to_path_buf();
        let index = index_of(&dir).unwrap();
        // The thread cannot start a pass while this is held. A stale day wakes it, and a pass
        // reads what is pending (the queue file is new, so racy) and takes the stale days: the
        // log below would be what is left, not what the writes made of it.
        let walking = index.walk.lock_recover();
        // As a bulk action would: more label files than the set takes, one of them real.
        for i in 0..DIRTY_MAX + 50 {
            super::super::written(&day.join(format!("20261021T{i:06}999Z_x{LABELS_SUFFIX}")));
        }
        super::super::super::write(
            &frame.labels,
            &LabelFile {
                status: Status::Skipped,
                ..LabelFile::default()
            },
        )
        .unwrap();
        {
            let hub = HUB.lock_recover();
            let c = hub.current.iter().find(|c| c.index.dir == *dir).unwrap();
            assert_eq!(c.log.pending.rels.len(), DIRTY_MAX);
            assert!(c.log.stale.contains(queue::day_of(&format!(
                "{}/{}/{}/x.jpg",
                frame.rel.scene, frame.rel.camera, frame.rel.date
            ))));
        }
        drop(walking);
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
        let skipped = Filters {
            status: StatusFilter::Is(Status::Skipped),
            ..f.clone()
        };
        assert_eq!(
            json(&from_index(&dir, &skipped, None)),
            walk(&dir, &skipped, None)
        );
    }

    #[test]
    fn a_frame_retention_took_unseen_is_never_shown() {
        let dir = Dir::new();
        generate(&dir, 53, 40, false);
        let f = all_any();
        let first = from_index(&dir, &f, None);
        let shown: Vec<String> = first
            .item
            .iter()
            .map(|i| i.frame.clone())
            .chain(first.upcoming.iter().take(2).cloned())
            .collect();
        for rel in &shown {
            // No label file goes, so no writer tells the index.
            std::fs::remove_file(Frame::resolve(&dir, rel).unwrap().jpg).unwrap();
        }
        let a = from_index(&dir, &f, None);
        let now: Vec<&String> = a.item.iter().map(|i| &i.frame).chain(&a.upcoming).collect();
        assert!(now.iter().all(|r| !shown.contains(r)), "{now:?} {shown:?}");
        // Marked gone, so the counts are the walk's again too.
        assert_eq!(json(&a), walk(&dir, &f, None));
    }

    #[test]
    fn labels_of_other_folders_and_other_files_are_ignored() {
        let dir = Dir::new();
        generate(&dir, 59, 10, false);
        let log = super::super::Log::new(&dir);
        let canon = dir.canonicalize().unwrap();
        let label = |root: &Path| {
            root.join("ir/gate/2026-10-21")
                .join(format!("s{LABELS_SUFFIX}"))
        };
        assert_eq!(
            log.rel_of(&label(&dir)).as_deref(),
            Some("ir/gate/2026-10-21/s.jpg")
        );
        assert_eq!(
            log.rel_of(&label(&canon)).as_deref(),
            Some("ir/gate/2026-10-21/s.jpg")
        );
        assert_eq!(
            log.rel_of(&label(&std::env::temp_dir().join("elsewhere"))),
            None
        );
        assert_eq!(
            log.rel_of(
                &dir.join("ir/gate/2026-10-21/deeper")
                    .join(format!("s{LABELS_SUFFIX}"))
            ),
            None
        );
        assert_eq!(log.rel_of(&dir.join("ir/gate/2026-10-21/s.json")), None);
        assert_eq!(log.rel_of(&dir.join(format!("s{LABELS_SUFFIX}"))), None);
    }

    /// A label write that changes only the suggestions (a check's, then a reviewer's answer) moves
    /// the frame into `auto=refine` and out again, as the walk sees it.
    #[test]
    fn a_write_of_suggestions_alone_moves_a_frame_in_and_out_of_auto_refine() {
        let dir = Dir::new();
        set_clock_ahead(&dir, Duration::from_secs(3600));
        let rels = generate(&dir, 89, 20, false);
        let frame = readable(&dir, &rels);
        let car = LabelBox {
            bbox: PLACES[3],
            state: BoxState::Accepted,
            ..super::super::super::tests_support::auto_box("car")
        };
        let plain = LabelFile {
            status: Status::Reviewed,
            reviewed_at: Some("2026-10-22T00:00:00Z".into()),
            boxes: vec![car],
            ..LabelFile::default()
        };
        super::super::super::write(&frame.labels, &plain).unwrap();
        let mut f = all_any();
        f.auto = Some(AutoFilter::Refine);
        // How many frames the index lists, the walk's answer checked whole.
        let listed = |f: &Filters| {
            let a = from_index(&dir, f, None);
            assert_eq!(json(&a), walk(&dir, f, None), "{f:?}");
            a.total
        };
        let (before, verify_before, add_before) = (
            listed(&f),
            listed(&Filters {
                refine_kind: Some(refine::Kind::Verify),
                ..f.clone()
            }),
            listed(&Filters {
                refine_kind: Some(refine::Kind::Add),
                ..f.clone()
            }),
        );
        let verify = refine::Suggestion {
            kind: refine::Kind::Verify,
            target: Some(PLACES[3]),
            target_label: Some("car".into()),
            bbox: None,
            label: None,
            score: None,
            why: "no fold model found it".into(),
            spot: None,
            answer: None,
            run: "r1".into(),
        };
        let wrote = super::super::super::update(&frame.labels, |old| {
            let mut l = old.unwrap();
            let done = refine::apply(&mut l, vec![verify.clone()], &[refine::Kind::Verify]);
            assert_eq!(done.written, 1);
            Ok(Some(l))
        })
        .unwrap();
        assert!(wrote);
        assert_eq!(listed(&f), before + 1, "a check's suggestions alone");
        let verify_only = Filters {
            refine_kind: Some(refine::Kind::Verify),
            ..f.clone()
        };
        assert_eq!(listed(&verify_only), verify_before + 1);
        let adds = Filters {
            refine_kind: Some(refine::Kind::Add),
            ..f.clone()
        };
        assert_eq!(listed(&adds), add_before);
        super::super::super::update(&frame.labels, |old| {
            let mut l = old.unwrap();
            assert!(refine::accept(&mut l, 0, None));
            Ok(Some(l))
        })
        .unwrap();
        assert_eq!(listed(&f), before, "answered: kept");
        assert_eq!(listed(&verify_only), verify_before);
        assert!(served(&dir));
    }

    #[test]
    fn rows_keep_their_size() {
        use super::super::row::{Cand, Part, Row, Side};
        // A field that grows the row grows a 300,000 frame index by 300 kB a byte: say so here.
        assert_eq!(size_of::<Cand>(), 12);
        assert_eq!(size_of::<Side>(), 20);
        assert_eq!(size_of::<Part>(), 36);
        assert_eq!(size_of::<Row>(), 80);
    }

    #[test]
    fn big_counts_do_not_saturate() {
        let dir = Dir::new();
        let rel = "ir/gate/2026-10-21/20261021T000001000Z_many.jpg";
        let day = dir.join("ir/gate/2026-10-21");
        std::fs::create_dir_all(&day).unwrap();
        let mut rec = empty_record();
        rec.id = "20261021T000001000Z_many".into();
        rec.scene = "ir".into();
        rec.camera = "gate".into();
        rec.suppressed = (0..300)
            .map(|i| Suppressed {
                label: "person".into(),
                score: 0.4,
                bbox: [i as f32, 0.0, i as f32 + 1.0, 1.0],
                reason: "x".into(),
            })
            .collect();
        std::fs::write(day.join("20261021T000001000Z_many.jpg"), b"x").unwrap();
        std::fs::write(
            day.join("20261021T000001000Z_many.json"),
            serde_json::to_vec(&rec).unwrap(),
        )
        .unwrap();
        let mut b = super::super::super::tests_support::auto_box("person");
        b.auto.triage = Some(Triage::Review);
        let boxes: Vec<LabelBox> = (0..600)
            .map(|i| LabelBox {
                bbox: [i as f32 * 3.0, 0.0, i as f32 * 3.0 + 2.0, 2.0],
                ..b.clone()
            })
            .collect();
        let f = LabelFile {
            boxes,
            ..LabelFile::default()
        };
        let path = day.join(format!("20261021T000001000Z_many{LABELS_SUFFIX}"));
        std::fs::write(&path, serde_json::to_vec(&f).unwrap()).unwrap();
        layout::append_review(&dir, &[rel.to_string()]).unwrap();
        let q = Filters::default();
        let a = from_index(&dir, &q, None);
        assert_eq!(a.item.as_ref().unwrap().uncertainty.auto_review, 600);
        assert_eq!(a.item.as_ref().unwrap().uncertainty.suppressed, 300);
        assert_eq!(json(&a), walk(&dir, &q, None));
        // Reviewed: 600 hard negatives, and as many training boxes.
        let mut f = f;
        f.status = Status::Reviewed;
        for (i, b) in f.boxes.iter_mut().enumerate() {
            b.state = if i % 2 == 0 {
                BoxState::Rejected
            } else {
                BoxState::Accepted
            };
            b.auto.triage = None;
        }
        super::super::super::write(&path, &f).unwrap();
        let any = Filters {
            status: StatusFilter::Any,
            ..q
        };
        let a = from_index(&dir, &any, None);
        assert_eq!(a.counts.rejected_boxes, 300);
        assert_eq!(a.counts.per_class["person"], 300);
        assert_eq!(json(&a), walk(&dir, &any, None));
    }

    /// The design budget: a 300,000 frame folder of the shape a long-running service collects
    /// (10 cameras, a year of days, an index line per frame, a third with a label file, a few
    /// re-label notes) stays under 64 MB.
    #[test]
    fn a_300k_frame_index_fits_its_budget() {
        use super::super::row::{Derive, Encoded, Encoder, Names, Shard};
        let wanted = wanted_a();
        let derive = Derive::new(Some(&wanted));
        let names = RwLock::new(Names::default());
        let encoder = Encoder {
            derive: &derive,
            names: &names,
        };
        let mut rng = Rng::new(300);
        let mut state = super::super::State::default();
        let (cameras, days, per_day) = (10, 365, 300_000 / (10 * 365) + 1);
        let mut frames = 0;
        for camera in 0..cameras {
            for d in 0..days {
                let date = format!("2026-{:02}-{:02}", 1 + d / 28 % 12, 1 + d % 28);
                let key = format!("night/cam{camera}/{date}");
                let mut shard = Shard::new(key);
                for i in 0..per_day {
                    if frames == 300_000 {
                        break;
                    }
                    frames += 1;
                    let stem = format!("20260101T{i:06}123Z_{:08x}", rng.below(1 << 30));
                    let mut rec = empty_record();
                    rec.model = "IPcam-general".into();
                    rec.timestamp = "2026-01-01T00:00:00.123Z".into();
                    rec.predictions = vec![det(&mut rng)];
                    rec.raw = rec.predictions.clone();
                    let label = rng.one_in(3).then(|| a_label_file(&mut rng));
                    let (part, mut note) = derive.part(label.as_ref(), &names);
                    if !rng.one_in(100) {
                        note = None;
                    }
                    let c = queue::Candidate {
                        rel: String::new(),
                        scene: "night".into(),
                        camera: format!("cam{camera}"),
                        buckets: vec![rng.pick(&Bucket::ALL)],
                        queued: rng.one_in(5),
                    };
                    shard.push(
                        &stem,
                        false,
                        c.queued,
                        encoder.cand(&c),
                        Encoded {
                            gone: false,
                            side: Some(encoder.side(&rec)),
                            part,
                            note,
                        },
                    );
                }
                shard.shrink();
                state.shards.insert(shard.day.clone(), shard);
            }
        }
        assert_eq!(frames, 300_000);
        let bytes = state.bytes() + names.read_recover().bytes();
        eprintln!("300k frames: {} bytes, {} a frame", bytes, bytes / 300_000);
        assert!(bytes < 64 * 1024 * 1024, "{bytes} bytes");
    }

    #[test]
    fn writes_during_the_first_build_are_in_the_answer_after_it() {
        let dir = Dir::new();
        let rels = generate(&dir, 61, 24, false);
        let days = by_day(&dir);
        let date = |k: &str| k.rsplit('/').next().unwrap().to_string();
        let (first, n) = days
            .iter()
            .max_by_key(|(k, _)| (date(k), (*k).clone()))
            .map(|(k, n)| (k.clone(), *n))
            .unwrap();
        let f = all_any();
        let hold = hold_after(&dir, n as u64);
        let _ = super::super::answer(&dir, &f, None).unwrap();
        assert!(hold.reached());
        // One frame of the day already in, one of a day still to come.
        let frames: Vec<Frame> = existing(&dir, &rels)
            .iter()
            .filter_map(|r| Frame::resolve(&dir, r))
            .filter(|fr| fr.labels().is_ok() && fr.record().is_ok())
            .collect();
        let inside = frames
            .iter()
            .find(|fr| fr.jpg.to_string_lossy().contains(&first));
        let outside = frames
            .iter()
            .find(|fr| !fr.jpg.to_string_lossy().contains(&first));
        for frame in inside.into_iter().chain(outside) {
            super::super::super::write(
                &frame.labels,
                &LabelFile {
                    status: Status::Skipped,
                    disagreement: Some("during the build".into()),
                    ..LabelFile::default()
                },
            )
            .unwrap();
        }
        let during = super::super::answer(&dir, &f, None).unwrap();
        assert!(during.indexing.is_some());
        hold.release();
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
    }

    #[test]
    fn stale_days_show_as_indexing_until_walked() {
        let dir = Dir::new();
        // Nothing racy: only the stale day is walked.
        set_clock_ahead(&dir, Duration::from_secs(3600));
        let rels = generate(&dir, 67, 20, false);
        let f = all_any();
        from_index(&dir, &f, None);
        let frame = readable(&dir, &rels);
        let day = frame.jpg.parent().unwrap().to_path_buf();
        let hold = hold(&dir);
        super::super::day_written(&day);
        assert!(hold.reached(), "the thread walks a stale day at once");
        let a = super::super::answer(&dir, &f, None).unwrap();
        let rows = queue::candidates(&dir, QueueSource::All).len() as u64;
        let i = a.indexing.unwrap();
        assert_eq!(i.total, rows);
        assert_eq!(
            i.done,
            rows - by_day(&dir)[&format!(
                "{}/{}/{}",
                frame.rel.scene, frame.rel.camera, frame.rel.date
            )] as u64
        );
        drop(hold);
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
    }

    #[test]
    fn a_rewritten_index_file_is_walked_again() {
        let dir = Dir::new();
        let rels = generate(&dir, 71, 30, false);
        let f = all_any();
        let a = from_index(&dir, &f, None);
        // A frame not shown goes, and retention rewrites its day's index without it.
        let shown: Vec<&String> = a.item.iter().map(|i| &i.frame).chain(&a.upcoming).collect();
        let frame = rels
            .iter()
            .filter(|r| !shown.contains(r))
            .filter_map(|r| Frame::resolve(&dir, r))
            .find(|fr| fr.record().is_ok() && fr.labels().is_ok_and(|l| l.is_none()))
            .unwrap();
        std::fs::remove_file(&frame.jpg).unwrap();
        layout::rewrite_index(frame.jpg.parent().unwrap()).unwrap();
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
    }

    #[test]
    fn the_thread_finds_what_changed_on_its_own() {
        let dir = Dir::new();
        fast_timing(&dir);
        let rels = generate(&dir, 73, 20, false);
        let f = all_any();
        from_index(&dir, &f, None);
        // Another process saves a label file: a rename, so its folder's time moves.
        let frame = readable(&dir, &rels);
        let tmp = frame.labels.with_extension("tmp");
        let skipped = LabelFile {
            status: Status::Skipped,
            ..LabelFile::default()
        };
        std::fs::write(&tmp, serde_json::to_vec(&skipped).unwrap()).unwrap();
        std::fs::rename(&tmp, &frame.labels).unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        let want = walk(&dir, &f, None);
        while json(&from_index(&dir, &f, None)) != want && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(json(&from_index(&dir, &f, None)), want);
    }

    /// `indexing` means frames to come: never `done == total`. That showed as "Indexing 20000 of
    /// 20000" for one poll at the end of a build, and while stale days hold no row.
    #[test]
    fn indexing_never_says_all_of_them_are_done() {
        let dir = Dir::new();
        set_clock_ahead(&dir, Duration::from_secs(3600));
        let rels = generate(&dir, 79, 20, false);
        let f = all_any();
        from_index(&dir, &f, None);
        let index = index_of(&dir).unwrap();
        let rows = index.state.read_recover().rows();
        // The thread cannot start a pass while this is held: the states below stay put.
        let walking = index.walk.lock_recover();
        // The end of a build: every day in, only the last swap left.
        index.building.store(true, Ordering::Release);
        index.done.store(rows, Ordering::Release);
        index.total.store(rows, Ordering::Release);
        assert_eq!(index.indexing(), None);
        index.done.store(rows - 1, Ordering::Release);
        assert_eq!(
            index.indexing(),
            Some(Indexing {
                done: rows - 1,
                total: rows
            })
        );
        index.building.store(false, Ordering::Release);
        // A stale day with no row: a label file of a frame no line names.
        let stale = |day: &str| {
            let mut hub = HUB.lock_recover();
            let c = hub.of(&index).unwrap();
            c.log.stale.clear();
            c.log.stale.insert(day.to_string());
        };
        stale("ir/nowhere/2026-10-21");
        assert_eq!(index.indexing(), None);
        // A stale day with rows: those are behind.
        let frame = readable(&dir, &rels);
        let key = format!(
            "{}/{}/{}",
            frame.rel.scene, frame.rel.camera, frame.rel.date
        );
        stale(&key);
        let n = index.state.read_recover().shards[&key].rows.len() as u64;
        assert_eq!(
            index.indexing(),
            Some(Indexing {
                done: rows - n,
                total: rows
            })
        );
        drop(walking);
        assert_eq!(json(&from_index(&dir, &f, None)), walk(&dir, &f, None));
        assert!(served(&dir));
    }

    #[test]
    fn a_label_file_of_a_frame_no_line_names_leaves_no_indexing_behind() {
        let dir = Dir::new();
        generate(&dir, 83, 20, false);
        let f = all_any();
        from_index(&dir, &f, None);
        // A frame on disk that no index line or review line names, labelled.
        let day = dir.join("ir/gate/2026-10-21");
        std::fs::create_dir_all(&day).unwrap();
        let stem = "20261021T555555000Z_stray";
        let mut rec = empty_record();
        rec.id = stem.into();
        std::fs::write(day.join(format!("{stem}.jpg")), b"x").unwrap();
        std::fs::write(
            day.join(format!("{stem}.json")),
            serde_json::to_vec(&rec).unwrap(),
        )
        .unwrap();
        let rel = format!("ir/gate/2026-10-21/{stem}.jpg");
        let frame = Frame::resolve(&dir, &rel).unwrap();
        super::super::super::write(&frame.labels, &LabelFile::default()).unwrap();
        let a = from_index(&dir, &f, None);
        assert_eq!(json(&a), walk(&dir, &f, None));
        assert!(json(&a).get("indexing").is_none());
    }
}
