//! Spot memory of fixed cameras, for labelers runs: the places a reviewer rejected again and again,
//! and the real objects a reviewer marked as parked.
//!
//! **Rejected spots** have no store of their own: they are read from the reviewer's decisions
//! ([`super::repeats::scan`]). A spot is at least [`REPEAT_MIN`] frames rejected at one place
//! ([`CameraMemory::rejections_at`]), within [`REJECTED_WINDOW_DAYS`] before the frame, with no
//! accepted box there (mixed spots are not spots). **Parked spots** live in
//! `<review folder>/spots/<camera>.json` ([`CameraSpots`]), because the history is capped and forgets;
//! `layout::day_dirs`, and so retention and the label walks, never go there (a scene folder is
//! the only way in). `moved_at` on that file ends the camera's past: every decision before it stops
//! counting ([`CameraMemory::new`]).
//!
//! IO here is the store ([`read`], [`update`], under its own lock, never held with the label write
//! lock, which is not reentrant) and [`Snapshot::take`]: a run reads once, at its start, because
//! [`super::repeats::history`] is invalidated by every label write the run makes. Only the service
//! writes the store. What a run does with a match is [`crate::training::autolabel::triage`]'s.

use super::repeats::{
    self, ACCEPT_SPOT_IOU, Decision, History, REJECT_SPOT_IOU, REPEAT_MIN, Spot, Verdict,
    same_object,
};
use super::{BoxState, LabelBox, iou, read_capped};
use crate::collect::layout;
use anyhow::{Context, Result, bail};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The folder of the store under the folder under review.
pub const SPOTS_DIR: &str = "spots";
/// Version of a store file.
pub const SPOTS_VERSION: u32 = 1;
/// Rejections older than this many days before a frame's date do not hide its boxes.
pub const REJECTED_WINDOW_DAYS: i64 = 30;
/// The reason of a box a run accepted at a parked spot, kept as evidence.
pub const PARKED_REASON: &str = "parked_spot";
/// Most entries, live or retired, of one frame size in a store file.
pub const MAX_PARKED: usize = 500;
/// Largest store file read (bytes).
const MAX_FILE_BYTES: u64 = 1024 * 1024;

/// A real object a reviewer marked as parked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parked {
    pub label: String,
    /// `[x0, y0, x1, y1]` in pixels of the frame size the entry is filed under.
    pub bbox: [f32; 4],
    /// The date of the frame it was marked on (`YYYY-MM-DD`).
    pub from: String,
    /// When it stopped counting (`YYYY-MM-DD`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retired: Option<String>,
}

/// One store file: a camera's parked spots per frame size, and when it last moved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraSpots {
    pub version: u32,
    /// When the camera moved (RFC 3339, UTC): nothing seen before it counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moved_at: Option<String>,
    /// `<width>x<height>` -> entries.
    #[serde(default)]
    pub parked: BTreeMap<String, Vec<Parked>>,
}

impl Default for CameraSpots {
    fn default() -> Self {
        Self {
            version: SPOTS_VERSION,
            moved_at: None,
            parked: BTreeMap::new(),
        }
    }
}

fn size_key(width: u32, height: u32) -> String {
    format!("{width}x{height}")
}

impl CameraSpots {
    /// The entries still counting for a `width` x `height` frame.
    fn live(&self, width: u32, height: u32) -> impl Iterator<Item = &Parked> {
        self.parked
            .get(&size_key(width, height))
            .into_iter()
            .flatten()
            .filter(|p| p.retired.is_none())
    }

    /// Entries still counting, every frame size.
    pub fn live_count(&self) -> usize {
        self.parked
            .values()
            .flatten()
            .filter(|p| p.retired.is_none())
            .count()
    }
}

fn store_path(dir: &Path, camera: &str) -> Result<PathBuf> {
    if !layout::camera_ok(camera) {
        bail!(
            "'{}' is not a camera name of the dataset",
            camera.escape_debug()
        );
    }
    Ok(dir.join(SPOTS_DIR).join(format!("{camera}.json")))
}

/// The store of `camera` under `dir`; empty when it has no file. A damaged file or one of a newer
/// version is an error: going on without it would forget a camera move, and a write would lose it.
pub fn read(dir: &Path, camera: &str) -> Result<CameraSpots> {
    let path = store_path(dir, camera)?;
    if !path.exists() {
        return Ok(CameraSpots::default());
    }
    let bytes = read_capped(&path, MAX_FILE_BYTES)?;
    let spots: CameraSpots = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "{} is damaged; repair or delete it (deleting forgets this camera's parked spots)",
            path.display()
        )
    })?;
    if spots.version > SPOTS_VERSION {
        bail!(
            "{} has version {}, newer than this build understands ({SPOTS_VERSION}); update Blue \
             Onyx Prism",
            path.display(),
            spots.version
        );
    }
    Ok(spots)
}

/// Serializes the store's writes within the process. Never taken with the label write lock.
static STORE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Read-modify-write of `camera`'s store under its lock: `edit` gets the file, and the file is
/// written atomically when `edit` changed it.
pub fn update<T>(
    dir: &Path,
    camera: &str,
    edit: impl FnOnce(&mut CameraSpots) -> Result<T>,
) -> Result<T> {
    let _held = STORE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let before = read(dir, camera)?;
    let mut spots = before.clone();
    let out = edit(&mut spots)?;
    if spots != before {
        let path = store_path(dir, camera)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut text = serde_json::to_vec_pretty(&spots)?;
        text.push(b'\n');
        layout::write_atomic(&path, &text)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(out)
}

/// A place on one camera's frames of one size ([`add_parked`], [`retire`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Place<'a> {
    pub camera: &'a str,
    pub width: u32,
    pub height: u32,
    pub label: &'a str,
    pub bbox: [f32; 4],
}

/// Add a parked spot, marked on a frame of date `from`. Returns the entry and whether it is new: a
/// live entry of that label at IoU >= [`ACCEPT_SPOT_IOU`] is that spot already (and keeps its date).
pub fn add_parked(dir: &Path, at: &Place<'_>, from: &str) -> Result<(Parked, bool)> {
    update(dir, at.camera, |s| {
        let list = s.parked.entry(size_key(at.width, at.height)).or_default();
        let same = |p: &&Parked| {
            p.retired.is_none() && p.label == at.label && iou(&p.bbox, &at.bbox) >= ACCEPT_SPOT_IOU
        };
        if let Some(p) = list.iter().find(same) {
            return Ok((p.clone(), false));
        }
        if list.len() >= MAX_PARKED {
            bail!(
                "{} already has {MAX_PARKED} parked spots at this frame size; use Camera moved: \
                 forget spots on the Review tab, or delete old entries from its file",
                at.camera
            );
        }
        let p = Parked {
            label: at.label.to_string(),
            bbox: at.bbox,
            from: from.to_string(),
            retired: None,
        };
        list.push(p.clone());
        Ok((p, true))
    })
}

/// Retire the live parked spot closest to `at` (same label, IoU >= [`ACCEPT_SPOT_IOU`]) on date
/// `today`; false when there is none.
pub fn retire(dir: &Path, at: &Place<'_>, today: &str) -> Result<bool> {
    update(dir, at.camera, |s| {
        let list = s.parked.get_mut(&size_key(at.width, at.height));
        let closest = list.and_then(|l| {
            l.iter_mut()
                .filter(|p| p.retired.is_none() && p.label == at.label)
                .map(|p| (iou(&p.bbox, &at.bbox), p))
                .filter(|(v, _)| *v >= ACCEPT_SPOT_IOU)
                .max_by(|a, b| a.0.total_cmp(&b.0))
        });
        Ok(closest
            .map(|(_, p)| p.retired = Some(today.to_string()))
            .is_some())
    })
}

/// The camera moved at `now` (RFC 3339, UTC): every live parked spot retires and `moved_at` is set.
/// Returns how many spots retired.
pub fn camera_moved(dir: &Path, camera: &str, now: &str) -> Result<usize> {
    let today = now.get(..10).unwrap_or(now).to_string();
    update(dir, camera, |s| {
        let mut n = 0;
        for p in s.parked.values_mut().flatten() {
            if p.retired.is_none() {
                p.retired = Some(today.clone());
                n += 1;
            }
        }
        s.moved_at = Some(now.to_string());
        Ok(n)
    })
}

fn parse_date(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()
}

/// Whether a frame `rel` was taken after `moved_at` (RFC 3339, UTC). On the day of the move only a
/// stem that gives the time can say (`20261105T102030123Z_<id>`); without it the frame counts as
/// before: a rejection of the old view must not hide boxes of the new one.
fn after_move(rel: &layout::Rel, moved_at: &str) -> bool {
    let Some(day) = moved_at.get(..10) else {
        return false;
    };
    match rel.date.as_str().cmp(day) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => {
            let moved: String = moved_at
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == 'T')
                .take(18)
                .collect();
            let stamp = rel.stem.get(..18).filter(|s| {
                s.as_bytes().get(8) == Some(&b'T')
                    && s.bytes()
                        .enumerate()
                        .all(|(i, c)| i == 8 || c.is_ascii_digit())
            });
            stamp.is_some_and(|s| s >= moved.as_str())
        }
    }
}

/// A rejected spot as the Review tab lists it ([`CameraMemory::rejected_spots`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RejectedSpot {
    pub label: String,
    pub bbox: [f32; 4],
    pub width: u32,
    pub height: u32,
    /// Frames rejected there, in the window.
    pub rejections: usize,
    /// The newest of them (`YYYY-MM-DD`).
    pub last: String,
}

/// One earlier decision, ready to match.
#[derive(Debug, Clone)]
struct Seen {
    frame: String,
    date: NaiveDate,
    verdict: Verdict,
    spot: Spot,
}

impl Seen {
    fn of(d: &Decision, camera: &str, moved_at: Option<&str>) -> Option<Self> {
        let rel = layout::parse_rel(&d.frame)?;
        if moved_at.is_some_and(|m| !after_move(&rel, m)) {
            return None;
        }
        Some(Self {
            frame: d.frame.clone(),
            date: parse_date(&rel.date)?,
            verdict: match d.state {
                BoxState::Accepted => Verdict::Accept,
                _ => Verdict::Reject(d.reject_reason.clone()),
            },
            spot: Spot {
                camera: camera.to_string(),
                label: d.label.clone(),
                bbox: d.bbox,
                width: d.width,
                height: d.height,
                min_iou: REJECT_SPOT_IOU,
            },
        })
    }

    fn accepted(&self) -> bool {
        self.verdict == Verdict::Accept
    }
}

/// What one camera's spots are at a point in time: pure, so that a run holds it for its length.
#[derive(Debug, Clone, Default)]
pub struct CameraMemory {
    camera: String,
    store: CameraSpots,
    seen: Vec<Seen>,
}

impl CameraMemory {
    /// From `history` ([`repeats::scan`]) and the camera's store; decisions before
    /// [`CameraSpots::moved_at`] are left out, accepted ones with them.
    pub fn new(history: &History, store: CameraSpots) -> Self {
        let moved = store.moved_at.clone();
        let seen = history
            .decisions
            .iter()
            .filter_map(|d| Seen::of(d, &history.camera, moved.as_deref()))
            .collect();
        Self {
            camera: history.camera.clone(),
            store,
            seen,
        }
    }

    pub fn store(&self) -> &CameraSpots {
        &self.store
    }

    /// The frames (other than `frame`, of its `width` x `height`) on which the reviewer rejected the
    /// place of `b`, a pending box ([`Spot::matches`] at [`REJECT_SPOT_IOU`]), when that makes it a
    /// spot, else 0. It takes at least [`REPEAT_MIN`] such frames, dated at most
    /// [`REJECTED_WINDOW_DAYS`] before `frame` and not after it. An accepted box there of the same
    /// object, of any date since a move, means something real comes here, so it is no spot. Frames
    /// are counted, not boxes: two boxes rejected on one frame are one look.
    pub fn rejections_at(&self, b: &LabelBox, frame: &str, width: u32, height: u32) -> usize {
        let Some(date) = layout::parse_rel(frame).and_then(|r| parse_date(&r.date)) else {
            return 0;
        };
        let mut frames = BTreeSet::new();
        for s in self.seen.iter().filter(|s| {
            s.frame != frame
                && (s.spot.width, s.spot.height) == (width, height)
                && s.spot.matches(b, &Verdict::Reject(None))
        }) {
            if s.accepted() {
                return 0;
            }
            let age = (date - s.date).num_days();
            if (0..=REJECTED_WINDOW_DAYS).contains(&age) {
                frames.insert(s.frame.as_str());
            }
        }
        if frames.len() >= REPEAT_MIN {
            frames.len()
        } else {
            0
        }
    }

    fn parked_spot(&self, p: &Parked, width: u32, height: u32, min_iou: f32) -> Spot {
        Spot {
            camera: self.camera.clone(),
            label: p.label.clone(),
            bbox: p.bbox,
            width,
            height,
            min_iou,
        }
    }

    /// The parked spot that takes `b`, a pending box of a `width` x `height` frame: the exact label
    /// at IoU >= [`ACCEPT_SPOT_IOU`] ([`Spot::matches`], [`Verdict::Accept`]).
    pub fn parked_at(&self, b: &LabelBox, width: u32, height: u32) -> Option<&Parked> {
        self.store.live(width, height).find(|p| {
            self.parked_spot(p, width, height, ACCEPT_SPOT_IOU)
                .matches(b, &Verdict::Accept)
        })
    }

    /// Whether a parked spot of the same object sits at `b`'s place, at the looser IoU of a rejected
    /// spot: a real object is known to be there, so `b` is not dismissed for the place.
    pub fn parked_near(&self, b: &LabelBox, width: u32, height: u32) -> bool {
        self.store.live(width, height).any(|p| {
            self.parked_spot(p, width, height, REJECT_SPOT_IOU)
                .matches(b, &Verdict::Reject(None))
        })
    }

    /// The camera's rejected spots as of `today`, for the page: rejected decisions in the window
    /// grouped around the newest at a place, without mixed places and parked ones. A view of the
    /// same rule as [`Self::rejections_at`], which compares each new box with every decision.
    pub fn rejected_spots(&self, today: NaiveDate) -> Vec<RejectedSpot> {
        struct Group<'a> {
            seed: &'a Spot,
            last: NaiveDate,
            frames: BTreeSet<&'a str>,
        }
        let here = |a: &Spot, b: &Spot| {
            (a.width, a.height) == (b.width, b.height)
                && same_object(&a.label, &b.label)
                && iou(&a.bbox, &b.bbox) >= REJECT_SPOT_IOU
        };
        let mut rejected: Vec<&Seen> = self
            .seen
            .iter()
            .filter(|s| !s.accepted())
            .filter(|s| (0..=REJECTED_WINDOW_DAYS).contains(&(today - s.date).num_days()))
            .collect();
        rejected.sort_by_key(|s| std::cmp::Reverse(s.date));
        let mut groups: Vec<Group<'_>> = Vec::new();
        for s in rejected {
            match groups.iter_mut().find(|g| here(g.seed, &s.spot)) {
                Some(g) => {
                    g.frames.insert(&s.frame);
                }
                None => groups.push(Group {
                    seed: &s.spot,
                    last: s.date,
                    frames: BTreeSet::from([s.frame.as_str()]),
                }),
            }
        }
        groups
            .into_iter()
            .filter(|g| g.frames.len() >= REPEAT_MIN)
            .filter(|g| {
                !self
                    .seen
                    .iter()
                    .any(|s| s.accepted() && here(g.seed, &s.spot))
            })
            .filter(|g| {
                !self.store.live(g.seed.width, g.seed.height).any(|p| {
                    same_object(&p.label, &g.seed.label)
                        && iou(&p.bbox, &g.seed.bbox) >= REJECT_SPOT_IOU
                })
            })
            .map(|g| RejectedSpot {
                label: g.seed.label.clone(),
                bbox: g.seed.bbox,
                width: g.seed.width,
                height: g.seed.height,
                rejections: g.frames.len(),
                last: g.last.format("%Y-%m-%d").to_string(),
            })
            .collect()
    }
}

/// The spots of a run: read once, at its start. A label write makes
/// [`repeats::history`] scan again, so a per-frame look would cost a scan per frame; and a run must
/// not learn from what it writes.
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    cameras: BTreeMap<String, CameraMemory>,
    /// Cameras whose store could not be read: they get no spots this run.
    pub warnings: Vec<String>,
}

impl Snapshot {
    /// The spots of every camera of `frames` under `dir`.
    pub fn take<'a>(dir: &Path, frames: impl Iterator<Item = &'a str>) -> Self {
        Self::take_with(
            frames,
            |camera| repeats::scan(dir, camera, repeats::HISTORY_CAP),
            |camera| read(dir, camera),
        )
    }

    fn take_with<'a>(
        frames: impl Iterator<Item = &'a str>,
        mut scan: impl FnMut(&str) -> History,
        mut store: impl FnMut(&str) -> Result<CameraSpots>,
    ) -> Self {
        let cameras: BTreeSet<String> = frames
            .filter_map(|f| layout::parse_rel(f).map(|r| r.camera))
            .collect();
        let mut out = Self::default();
        for camera in cameras {
            match store(&camera) {
                Ok(s) => {
                    let memory = CameraMemory::new(&scan(&camera), s);
                    out.cameras.insert(camera, memory);
                }
                Err(e) => out.warnings.push(format!(
                    "{camera}: no spot memory this run, its boxes all stay visible: {e:#}"
                )),
            }
        }
        out
    }

    /// The memory of `frame`'s camera.
    pub fn of(&self, frame: &str) -> Option<&CameraMemory> {
        self.cameras.get(&layout::parse_rel(frame)?.camera)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::tests_support::auto_box;
    use super::*;

    const AT: [f32; 4] = [10.0, 10.0, 50.0, 50.0]; // auto_box's place

    fn temp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-spots-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn decision(day: &str, n: u32, label: &str, state: BoxState) -> Decision {
        Decision {
            frame: format!("day/cam/{day}/20261001T00000000{n}Z_a.jpg"),
            label: label.into(),
            bbox: AT,
            state,
            reject_reason: None,
            width: 640,
            height: 480,
        }
    }

    fn memory(decisions: Vec<Decision>, store: CameraSpots) -> CameraMemory {
        let h = History {
            camera: "cam".into(),
            decisions,
            truncated: false,
        };
        CameraMemory::new(&h, store)
    }

    fn rejected(days: &[&str]) -> Vec<Decision> {
        days.iter()
            .enumerate()
            .map(|(i, d)| decision(d, u32::try_from(i).unwrap(), "car", BoxState::Rejected))
            .collect()
    }

    const FRAME: &str = "day/cam/2026-10-20/20261020T010203004Z_abcd.jpg";

    fn hits(m: &CameraMemory, label: &str) -> usize {
        m.rejections_at(&auto_box(label), FRAME, 640, 480)
    }

    #[test]
    fn a_spot_takes_repeat_min_frames_at_one_place() {
        let store = CameraSpots::default;
        assert_eq!(hits(&memory(rejected(&["2026-10-10"]), store()), "car"), 0);
        let m = memory(rejected(&["2026-10-10", "2026-10-11"]), store());
        assert_eq!(hits(&m, "car"), 2);
        assert_eq!(hits(&m, "truck"), 2, "another vehicle at the place");
        assert_eq!(hits(&m, "dog"), 0, "another object");
        // Two boxes on one frame are one look.
        let one = rejected(&["2026-10-10", "2026-10-10"]);
        let same_frame = vec![one[0].clone(), one[0].clone()];
        assert_eq!(hits(&memory(same_frame, store()), "car"), 0);
        // Another frame size, and the frame's own decisions, say nothing.
        let mut other = rejected(&["2026-10-10", "2026-10-11"]);
        other[1].width = 1280;
        assert_eq!(hits(&memory(other, store()), "car"), 0);
        let mut own = rejected(&["2026-10-10", "2026-10-11"]);
        own[1].frame = FRAME.into();
        assert_eq!(hits(&memory(own, store()), "car"), 0);
        // Past the place.
        let m = memory(rejected(&["2026-10-10", "2026-10-11"]), store());
        let mut off = auto_box("car");
        off.bbox = [10.0, 10.0, 50.0, 33.0]; // IoU 0.575
        assert_eq!(m.rejections_at(&off, FRAME, 640, 480), 0);
    }

    #[test]
    fn only_rejections_of_the_thirty_days_before_the_frame_count() {
        let store = CameraSpots::default;
        // 30 days before 2026-10-20 is 2026-09-20.
        let edge = memory(rejected(&["2026-09-20", "2026-09-21"]), store());
        assert_eq!(hits(&edge, "car"), 2, "day 30 is in");
        let past = memory(rejected(&["2026-09-19", "2026-09-21"]), store());
        assert_eq!(hits(&past, "car"), 0, "day 31 is out");
        let same_day = memory(rejected(&["2026-10-20", "2026-10-20"]), store());
        assert_eq!(hits(&same_day, "car"), 2, "the frame's own date is in");
        let later = memory(rejected(&["2026-10-21", "2026-10-22"]), store());
        assert_eq!(hits(&later, "car"), 0, "a rejection after the frame is not");
    }

    #[test]
    fn a_mixed_place_is_no_spot_and_an_accept_counts_at_any_age() {
        let store = CameraSpots::default;
        let mut d = rejected(&["2026-10-10", "2026-10-11", "2026-10-12"]);
        d.push(decision("2026-10-13", 9, "car", BoxState::Accepted));
        assert_eq!(hits(&memory(d.clone(), store()), "car"), 0);
        // A truck accepted there is the same object.
        d.pop();
        d.push(decision("2026-10-13", 9, "truck", BoxState::Accepted));
        assert_eq!(hits(&memory(d.clone(), store()), "car"), 0);
        // Older than the window, and later than the frame: still something real comes here.
        d.pop();
        d.push(decision("2026-01-01", 9, "car", BoxState::Accepted));
        assert_eq!(hits(&memory(d.clone(), store()), "car"), 0);
        d.pop();
        d.push(decision("2026-12-01", 9, "car", BoxState::Accepted));
        assert_eq!(hits(&memory(d.clone(), store()), "car"), 0);
        // Another object accepted there is no mix.
        d.pop();
        d.push(decision("2026-10-13", 9, "dog", BoxState::Accepted));
        assert_eq!(hits(&memory(d, store()), "car"), 3);
    }

    #[test]
    fn a_camera_move_ends_the_past_exactly() {
        let store = |moved: &str| CameraSpots {
            moved_at: Some(moved.into()),
            ..CameraSpots::default()
        };
        let d = rejected(&["2026-10-10", "2026-10-11"]);
        let m = memory(d.clone(), store("2026-10-12T00:00:00.000Z"));
        assert_eq!(hits(&m, "car"), 0, "both before the move");
        let m = memory(d.clone(), store("2026-10-11T00:00:00.000Z"));
        assert_eq!(hits(&m, "car"), 0, "one before: under the minimum");
        let m = memory(d.clone(), store("2026-10-01T00:00:00.000Z"));
        assert_eq!(hits(&m, "car"), 2, "both after it");
        // An accepted box from before the move no longer mixes the spot.
        let mut mixed = rejected(&["2026-10-10", "2026-10-11"]);
        mixed.push(decision("2026-10-02", 9, "car", BoxState::Accepted));
        let m = memory(mixed, store("2026-10-05T00:00:00.000Z"));
        assert_eq!(hits(&m, "car"), 2);
        // On the day of the move the stem's time decides, and no time means before.
        let rel =
            |stem: &str| layout::parse_rel(&format!("day/cam/2026-10-05/{stem}.jpg")).unwrap();
        let moved = "2026-10-05T12:00:00.000Z";
        assert!(!after_move(&rel("20261005T115959999Z_a"), moved));
        assert!(after_move(&rel("20261005T120000000Z_a"), moved));
        assert!(!after_move(&rel("cam-0001"), moved));
    }

    #[test]
    fn parked_needs_the_exact_label_and_the_stricter_iou() {
        let mut store = CameraSpots::default();
        store.parked.insert(
            size_key(640, 480),
            vec![Parked {
                label: "car".into(),
                bbox: AT,
                from: "2026-10-01".into(),
                retired: None,
            }],
        );
        let m = memory(Vec::new(), store.clone());
        assert!(m.parked_at(&auto_box("car"), 640, 480).is_some());
        assert!(m.parked_at(&auto_box("truck"), 640, 480).is_none());
        assert!(m.parked_near(&auto_box("truck"), 640, 480), "same object");
        assert!(m.parked_at(&auto_box("car"), 1280, 480).is_none());
        let mut loose = auto_box("car");
        loose.bbox = [10.0, 10.0, 50.0, 40.0]; // IoU 0.75
        assert!(m.parked_at(&loose, 640, 480).is_none());
        assert!(m.parked_near(&loose, 640, 480));
        store.parked.values_mut().flatten().for_each(|p| {
            p.retired = Some("2026-10-02".into());
        });
        let m = memory(Vec::new(), store);
        assert!(m.parked_at(&auto_box("car"), 640, 480).is_none());
        assert!(!m.parked_near(&auto_box("car"), 640, 480));
    }

    fn place(label: &'static str, bbox: [f32; 4]) -> Place<'static> {
        Place {
            camera: "cam",
            width: 640,
            height: 480,
            label,
            bbox,
        }
    }

    #[test]
    fn the_store_round_trips_dedups_and_retires() {
        let dir = temp();
        assert_eq!(read(&dir, "cam").unwrap(), CameraSpots::default());
        let (p, new) = add_parked(&dir, &place("car", AT), "2026-10-01").unwrap();
        assert!(new);
        // The same spot again keeps its date.
        let (again, new) = add_parked(&dir, &place("car", AT), "2026-10-09").unwrap();
        assert!(!new);
        assert_eq!(again, p);
        // Another label at the place is another spot.
        assert!(
            add_parked(&dir, &place("truck", AT), "2026-10-09")
                .unwrap()
                .1
        );
        let on_disk: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("spots/cam.json")).unwrap()).unwrap();
        assert_eq!(on_disk["parked"]["640x480"][0]["from"], "2026-10-01");
        assert!(on_disk["parked"]["640x480"][0].get("retired").is_none());
        assert_eq!(read(&dir, "cam").unwrap().live_count(), 2);
        // Retired: not found again, and marking it again is a new spot from the new date.
        assert!(!retire(&dir, &place("bus", AT), "2026-10-10").unwrap());
        assert!(retire(&dir, &place("car", AT), "2026-10-10").unwrap());
        assert!(!retire(&dir, &place("car", AT), "2026-10-10").unwrap());
        assert_eq!(read(&dir, "cam").unwrap().live_count(), 1);
        let (back, new) = add_parked(&dir, &place("car", AT), "2026-10-11").unwrap();
        assert!(new);
        assert_eq!(back.from, "2026-10-11");
        // The move retires what is live and remembers when.
        assert_eq!(
            camera_moved(&dir, "cam", "2026-10-12T01:02:03.004Z").unwrap(),
            2
        );
        let s = read(&dir, "cam").unwrap();
        assert_eq!(s.live_count(), 0);
        assert_eq!(s.moved_at.as_deref(), Some("2026-10-12T01:02:03.004Z"));
        assert_eq!(s.parked["640x480"].len(), 3, "kept, retired");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_damaged_or_newer_store_is_refused_and_left_alone() {
        let dir = temp();
        let path = dir.join("spots/cam.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{ not json").unwrap();
        assert!(read(&dir, "cam").is_err());
        assert!(add_parked(&dir, &place("car", AT), "2026-10-01").is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not json");
        std::fs::write(&path, r#"{"version": 99}"#).unwrap();
        assert!(read(&dir, "cam").unwrap_err().to_string().contains("newer"));
        assert!(read(&dir, "../up").is_err(), "a camera name is checked");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_writes_lose_nothing() {
        let dir = temp();
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let dir = dir.clone();
                std::thread::spawn(move || {
                    let x = 100.0 * i as f32;
                    let at = Place {
                        bbox: [x, 0.0, x + 50.0, 50.0],
                        ..place("car", AT)
                    };
                    add_parked(&dir, &at, "2026-10-01").unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(read(&dir, "cam").unwrap().live_count(), 16);
        let leftovers = std::fs::read_dir(dir.join("spots")).unwrap().count();
        assert_eq!(leftovers, 1, "no temporary file stays");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn neither_the_label_walks_nor_retention_go_into_the_store() {
        let dir = temp();
        add_parked(&dir, &place("car", AT), "2026-10-01").unwrap();
        // Shaped like the layout on purpose: only a scene folder is walked.
        let fake = dir.join("spots/cam/2026-10-01");
        std::fs::create_dir_all(&fake).unwrap();
        std::fs::write(fake.join("20261001T000000000Z_a.jpg"), b"x").unwrap();
        std::fs::write(fake.join("20261001T000000000Z_a.labels.json"), b"{}").unwrap();
        assert!(layout::day_dirs(&dir).is_empty());
        assert!(super::super::label_files(&dir).is_empty());
        let r = crate::retention::prune_dataset(&dir, 1, 1).unwrap();
        assert_eq!((r.deleted, r.remaining), (0, 0));
        assert!(dir.join("spots/cam.json").is_file());
        assert!(fake.join("20261001T000000000Z_a.jpg").is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every reader of a dismissal, for both reasons.
    #[test]
    fn readers_treat_both_dismissals_alike() {
        use super::super::bulk::{DISMISSED_REASON, SPOT_DISMISSED_REASON};
        use super::super::queue::AutoFilter;
        use super::super::{
            LabelFile, RejectReason, Status, Triage, auto_accept_eligible, normalize,
        };
        let classes = ["car".to_string()];
        let dismissed = |reason: &str| LabelBox {
            state: BoxState::Rejected,
            reason: Some(reason.into()),
            ..auto_box("car")
        };
        let answers = |reason: &str| {
            let b = dismissed(reason);
            let mut sure = auto_box("car");
            sure.bbox = [100.0, 100.0, 200.0, 200.0];
            sure.auto.triage = Some(Triage::Accept);
            let f = LabelFile {
                repeat_spot: true,
                boxes: vec![b.clone(), sure],
                ..LabelFile::default()
            };
            let mut no_pending = f.clone();
            no_pending.boxes.pop();
            let saved = |mut b: LabelBox, status| {
                b.source = super::super::Source::Auto;
                let f = LabelFile {
                    status,
                    boxes: vec![b],
                    ..LabelFile::default()
                };
                let n = normalize(f, &classes, 640, 480, "t").unwrap();
                (n.boxes[0].state, n.boxes[0].reason.clone())
            };
            let mut with_reason = b.clone();
            with_reason.reject_reason = Some(RejectReason::Reflection);
            let mut accepted = b.clone();
            accepted.state = BoxState::Accepted;
            (
                b.is_dismissed(),
                f.hard_negatives().count(),
                auto_accept_eligible(&f),
                repeats::repeats_only(&no_pending),
                AutoFilter::Dismissed.matches(Some(&f)),
                saved(b.clone(), Status::Reviewed).1.is_some(),
                saved(with_reason, Status::Reviewed),
                saved(accepted, Status::Reviewed),
            )
        };
        let a = answers(DISMISSED_REASON);
        assert_eq!(a, answers(SPOT_DISMISSED_REASON));
        assert_eq!(
            a,
            (
                true,
                0,
                false,
                false,
                true,
                true,
                (BoxState::Rejected, None),
                (BoxState::Accepted, None),
            )
        );
    }

    #[test]
    fn a_save_ends_a_parked_mark_only_on_a_box_taken_back() {
        use super::super::{LabelFile, ParkedMark, normalize};
        let save = |state| {
            let mut b = auto_box("car");
            b.state = state;
            b.reason = Some(PARKED_REASON.into());
            b.auto.parked = Some(ParkedMark {
                since: "2026-10-01".into(),
                bbox: AT,
            });
            let f = LabelFile {
                boxes: vec![b],
                ..LabelFile::default()
            };
            normalize(f, &["car".to_string()], 640, 480, "t")
                .unwrap()
                .boxes
                .remove(0)
        };
        let kept = save(BoxState::Accepted);
        assert_eq!(kept.reason.as_deref(), Some(PARKED_REASON));
        assert!(kept.auto.parked.is_some());
        let back = save(BoxState::Rejected);
        assert_eq!((back.reason, back.auto.parked), (None, None));
    }

    #[test]
    fn a_snapshot_reads_each_camera_once_and_skips_an_unreadable_store() {
        let frames = [
            "day/cam/2026-10-20/a.jpg",
            "night/cam/2026-10-21/b.jpg",
            "day/cam/2026-10-22/c.jpg",
            "day/drive/2026-10-20/d.jpg",
            "day/broken/2026-10-20/e.jpg",
            "not a frame",
        ];
        let (mut scans, mut reads) = (Vec::new(), 0);
        let s = Snapshot::take_with(
            frames.iter().copied(),
            |c| {
                scans.push(c.to_string());
                History::default()
            },
            |c| {
                reads += 1;
                if c == "broken" {
                    bail!("damaged");
                }
                Ok(CameraSpots::default())
            },
        );
        assert_eq!(scans, ["cam", "drive"], "one scan per camera per run");
        assert_eq!(reads, 3);
        assert!(s.of(frames[1]).is_some());
        assert!(s.of(frames[4]).is_none(), "no store, no spots");
        assert_eq!(s.warnings.len(), 1);
        assert!(s.warnings[0].contains("broken"));
    }

    #[test]
    fn the_page_lists_rejected_spots_without_mixed_or_parked_ones() {
        let today = NaiveDate::from_ymd_opt(2026, 10, 20).unwrap();
        let m = memory(
            rejected(&["2026-10-10", "2026-10-11"]),
            CameraSpots::default(),
        );
        let list = m.rejected_spots(today);
        assert_eq!(list.len(), 1);
        assert_eq!(
            (list[0].rejections, list[0].last.as_str()),
            (2, "2026-10-11")
        );
        // Out of the window by today's date.
        let later = NaiveDate::from_ymd_opt(2026, 12, 20).unwrap();
        assert!(m.rejected_spots(later).is_empty());
        let mut d = rejected(&["2026-10-10", "2026-10-11"]);
        d.push(decision("2026-10-12", 7, "car", BoxState::Accepted));
        assert!(
            memory(d, CameraSpots::default())
                .rejected_spots(today)
                .is_empty()
        );
        let mut store = CameraSpots::default();
        store.parked.insert(
            size_key(640, 480),
            vec![Parked {
                label: "truck".into(),
                bbox: AT,
                from: "2026-10-01".into(),
                retired: None,
            }],
        );
        let d = rejected(&["2026-10-10", "2026-10-11"]);
        assert!(memory(d, store).rejected_spots(today).is_empty());
    }
}
