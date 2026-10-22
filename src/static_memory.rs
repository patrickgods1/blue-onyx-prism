//! Static-box memory: a detection that appears at the same place in many requests of one
//! camera is background (a rock or a plant read as "person") and is suppressed.
//!
//! Per key (one camera, see [`camera_key`]) the memory keeps tracks: the boxes of one label seen
//! in consecutive requests, one box per request, every pair with IoU >= `iou` (the box never
//! moved). A new box joins the matching track with the most boxes; it is suppressed when that
//! track already holds boxes from >= `hits` earlier requests within `window_minutes`. A request
//! without a matching box releases the track, so a box must be seen in every request of the
//! key to stay static. Boxes older than the window are forgotten.
//!
//! Bounded: at most [`MAX_KEYS`] keys (least recently used evicted), [`MAX_TRACKS_PER_KEY`]
//! tracks per key and [`MAX_BOXES_PER_TRACK`] boxes per track. The clock is the `now` passed
//! to [`StaticMemory::apply`] (tests pass synthetic instants). Single-threaded: the worker owns
//! one per model; it survives live-settings swaps and starts empty with a new worker.
//!
//! Risk: a person standing perfectly still for `hits` requests is suppressed too; the feature
//! is off by default.

use crate::model::Detection;
use crate::model::filters::label_of;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Most keys (cameras) remembered.
pub const MAX_KEYS: usize = 256;
/// Most tracks per key.
pub const MAX_TRACKS_PER_KEY: usize = 64;
/// Most boxes kept per track (also the largest accepted `hits`).
pub const MAX_BOXES_PER_TRACK: usize = 64;

/// `static_suppression` in the config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StaticSuppression {
    pub enabled: bool,
    /// IoU every pair of a track's boxes must reach (the box did not move).
    #[serde(serialize_with = "crate::config::thr::ser_clean")]
    pub iou: f32,
    /// Earlier requests with the same box before it is suppressed.
    pub hits: u32,
    /// Boxes older than this are forgotten.
    pub window_minutes: u32,
    /// Labels it applies to (case-insensitive).
    pub labels: Vec<String>,
}

impl Default for StaticSuppression {
    fn default() -> Self {
        Self {
            enabled: false,
            iou: 0.85,
            hits: 5,
            window_minutes: 30,
            labels: vec!["person".to_string()],
        }
    }
}

impl StaticSuppression {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub fn window(&self) -> Duration {
        Duration::from_secs(u64::from(self.window_minutes).saturating_mul(60))
    }

    /// Range checks, errors naming `field` (`static_suppression` or
    /// `models: 'x' static_suppression`).
    pub fn validate(&self, field: &str) -> anyhow::Result<()> {
        crate::config::check_threshold(&format!("{field}.iou"), self.iou)?;
        if self.iou <= 0.0 {
            anyhow::bail!("{field}.iou: must be above 0 (0.85 is the default), got 0");
        }
        let max_hits = u32::try_from(MAX_BOXES_PER_TRACK).unwrap_or(u32::MAX);
        if !(1..=max_hits).contains(&self.hits) {
            anyhow::bail!(
                "{field}.hits: must be between 1 and {max_hits}, got {}; set the number of \
                 earlier requests a box must repeat in (5 is the default)",
                self.hits
            );
        }
        if !(1..=1440).contains(&self.window_minutes) {
            anyhow::bail!(
                "{field}.window_minutes: must be between 1 and 1440, got {}",
                self.window_minutes
            );
        }
        if self.labels.iter().any(|l| l.trim().is_empty()) {
            anyhow::bail!("{field}.labels: contains an empty label; remove it");
        }
        Ok(())
    }
}

/// The memory key of a request: the camera part of `image_name` when it has one, else
/// `<model>|<w>x<h>`. The camera part is the file name up to its first `.`, without trailing
/// digits, `_` and `-` (`Driveway.20240101_120000.jpg` and `Driveway_0042.jpg` -> `driveway`),
/// so per-request file names of one camera share a key. Generic upload names (`image.jpg`,
/// `blob`, ...) carry no camera and fall back to the model and frame size.
pub fn camera_key(model: &str, image_name: &str, w: u32, h: u32) -> String {
    let file = image_name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default()
        .trim();
    let stem = file.split('.').next().unwrap_or_default();
    let camera = stem
        .trim_end_matches(|c: char| c.is_ascii_digit() || c == '_' || c == '-' || c == ' ')
        .to_ascii_lowercase();
    let generic = matches!(
        camera.as_str(),
        "" | "image" | "img" | "blob" | "file" | "upload" | "snapshot" | "frame" | "photo"
    );
    if generic {
        format!("{model}|{w}x{h}")
    } else {
        format!("{model}|cam:{camera}")
    }
}

#[derive(Debug, Clone)]
struct Seen {
    at: Instant,
    det: Detection,
}

#[derive(Debug)]
struct Track {
    label: String,
    seen: VecDeque<Seen>,
}

impl Track {
    fn last_at(&self) -> Option<Instant> {
        self.seen.back().map(|s| s.at)
    }
}

#[derive(Debug, Default)]
struct KeyState {
    tracks: Vec<Track>,
    /// Use counter of the last request (LRU over keys).
    used: u64,
}

/// The memory of one model (see the module docs).
#[derive(Debug, Default)]
pub struct StaticMemory {
    keys: HashMap<String, KeyState>,
    counter: u64,
}

impl StaticMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keys remembered.
    pub fn keys(&self) -> usize {
        self.keys.len()
    }

    /// Tracks remembered under `key`.
    pub fn tracks(&self, key: &str) -> usize {
        self.keys.get(key).map_or(0, |k| k.tracks.len())
    }

    /// Record the detections of one request under `key` at `now` and drop the static ones
    /// (labels resolved with `class_names`). Returns the kept detections in input order and the
    /// number suppressed. Disabled: returns `dets` unchanged and forgets everything.
    pub fn apply(
        &mut self,
        key: &str,
        dets: Vec<Detection>,
        class_names: &[String],
        cfg: &StaticSuppression,
        now: Instant,
    ) -> (Vec<Detection>, u32) {
        if !cfg.enabled {
            if !self.keys.is_empty() {
                self.keys.clear();
            }
            return (dets, 0);
        }
        self.counter = self.counter.wrapping_add(1);
        if !self.keys.contains_key(key)
            && self.keys.len() >= MAX_KEYS
            && let Some(oldest) = self
                .keys
                .iter()
                .min_by_key(|(_, s)| s.used)
                .map(|(k, _)| k.clone())
        {
            self.keys.remove(&oldest);
        }
        let state = self.keys.entry(key.to_string()).or_default();
        state.used = self.counter;
        let window = cfg.window();
        for t in &mut state.tracks {
            while t
                .seen
                .front()
                .is_some_and(|s| now.saturating_duration_since(s.at) > window)
            {
                t.seen.pop_front();
            }
        }
        state.tracks.retain(|t| !t.seen.is_empty());

        let hits = usize::try_from(cfg.hits).unwrap_or(usize::MAX);
        let mut matched = vec![false; state.tracks.len()];
        let mut keep = Vec::with_capacity(dets.len());
        let mut suppressed = 0u32;
        let mut new_tracks: Vec<Track> = Vec::new();
        for d in dets {
            let label = label_of(class_names, d.class_id);
            if !d.is_finite() || !cfg.labels.iter().any(|l| l.eq_ignore_ascii_case(&label)) {
                keep.push(d);
                continue;
            }
            // The unmatched track of this label that `d` fits with every box, longest first.
            let best = state
                .tracks
                .iter()
                .enumerate()
                .filter(|(i, t)| {
                    !matched.get(*i).copied().unwrap_or(true)
                        && t.label.eq_ignore_ascii_case(&label)
                        && t.seen
                            .iter()
                            .all(|s| crate::model::nms::iou(&s.det, &d) >= cfg.iou)
                })
                .max_by_key(|(_, t)| t.seen.len())
                .map(|(i, _)| i);
            let seen = Seen {
                at: now,
                det: d.clone(),
            };
            match best.and_then(|i| state.tracks.get_mut(i).map(|t| (i, t))) {
                Some((i, t)) => {
                    if let Some(m) = matched.get_mut(i) {
                        *m = true;
                    }
                    if t.seen.len() >= hits {
                        suppressed = suppressed.saturating_add(1);
                    } else {
                        keep.push(d);
                    }
                    t.seen.push_back(seen);
                    while t.seen.len() > MAX_BOXES_PER_TRACK {
                        t.seen.pop_front();
                    }
                }
                None => {
                    keep.push(d);
                    new_tracks.push(Track {
                        label: label.into_owned(),
                        seen: VecDeque::from([seen]),
                    });
                }
            }
        }
        // A track without a box in this request is released.
        let mut i = 0usize;
        state.tracks.retain(|_| {
            let m = matched.get(i).copied().unwrap_or(false);
            i += 1;
            m
        });
        state.tracks.extend(new_tracks);
        while state.tracks.len() > MAX_TRACKS_PER_KEY {
            // Drop the least recently seen track (ties: the shortest).
            let Some(victim) = state
                .tracks
                .iter()
                .enumerate()
                .min_by_key(|(_, t)| (t.last_at(), t.seen.len()))
                .map(|(i, _)| i)
            else {
                break;
            };
            state.tracks.remove(victim);
        }
        (keep, suppressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["person".into(), "car".into()]
    }

    fn det(x: f32, class_id: usize) -> Detection {
        Detection {
            x1: x,
            y1: 100.0,
            x2: x + 50.0,
            y2: 200.0,
            score: 0.8,
            class_id,
        }
    }

    fn on() -> StaticSuppression {
        StaticSuppression {
            enabled: true,
            hits: 3,
            ..Default::default()
        }
    }

    #[test]
    fn repeated_box_is_suppressed_after_hits() {
        let mut m = StaticMemory::new();
        let t0 = Instant::now();
        let cfg = on();
        for i in 0..3u64 {
            let (out, n) = m.apply(
                "k",
                vec![det(10.0, 0)],
                &names(),
                &cfg,
                t0 + Duration::from_secs(i),
            );
            assert_eq!((out.len(), n), (1, 0), "request {i}");
        }
        // Seen in 3 earlier requests: suppressed now and while it stays.
        for i in 3..6u64 {
            let (out, n) = m.apply(
                "k",
                vec![det(10.5, 0), det(500.0, 1)],
                &names(),
                &cfg,
                t0 + Duration::from_secs(i),
            );
            assert_eq!(n, 1);
            assert_eq!(out, vec![det(500.0, 1)], "the car is not a static label");
        }
    }

    #[test]
    fn moving_box_is_kept() {
        let mut m = StaticMemory::new();
        let t0 = Instant::now();
        let cfg = on();
        // Drifts 4 px per request: each consecutive pair matches (IoU ~0.85+) but the first
        // and the latest do not, so the track never holds 3 mutually overlapping boxes.
        for i in 0..10u32 {
            let x = 10.0 + 4.0 * i as f32;
            let (out, n) = m.apply(
                "k",
                vec![det(x, 0)],
                &names(),
                &cfg,
                t0 + Duration::from_secs(u64::from(i)),
            );
            assert_eq!((out.len(), n), (1, 0), "request {i}");
        }
        // A person walking across: never matched.
        let mut m = StaticMemory::new();
        for i in 0..10u32 {
            let (_, n) = m.apply("k", vec![det(60.0 * i as f32, 0)], &names(), &cfg, t0);
            assert_eq!(n, 0);
        }
    }

    #[test]
    fn a_request_without_the_box_releases_it() {
        let mut m = StaticMemory::new();
        let t0 = Instant::now();
        let cfg = on();
        for _ in 0..4 {
            m.apply("k", vec![det(10.0, 0)], &names(), &cfg, t0);
        }
        assert_eq!(m.apply("k", vec![det(10.0, 0)], &names(), &cfg, t0).1, 1);
        m.apply("k", vec![], &names(), &cfg, t0);
        assert_eq!(m.tracks("k"), 0);
        assert_eq!(m.apply("k", vec![det(10.0, 0)], &names(), &cfg, t0).1, 0);
        // Other keys are separate cameras.
        for _ in 0..3 {
            m.apply("a", vec![det(10.0, 0)], &names(), &cfg, t0);
        }
        assert_eq!(m.apply("b", vec![det(10.0, 0)], &names(), &cfg, t0).1, 0);
    }

    #[test]
    fn old_boxes_expire() {
        let mut m = StaticMemory::new();
        let t0 = Instant::now();
        let cfg = StaticSuppression {
            window_minutes: 1,
            ..on()
        };
        let at = |s: u64| t0 + Duration::from_secs(s);
        for s in [0, 20, 40] {
            m.apply("k", vec![det(10.0, 0)], &names(), &cfg, at(s));
        }
        // At 70 s the box of 0 s is out of the window: only 2 earlier hits remain.
        assert_eq!(
            m.apply("k", vec![det(10.0, 0)], &names(), &cfg, at(70)).1,
            0
        );
        // At 75 s: 20, 40, 70 are within the minute.
        assert_eq!(
            m.apply("k", vec![det(10.0, 0)], &names(), &cfg, at(75)).1,
            1
        );
        // Long gap: everything expired.
        assert_eq!(
            m.apply("k", vec![det(10.0, 0)], &names(), &cfg, at(1000)).1,
            0
        );
    }

    #[test]
    fn two_static_boxes_and_disabled() {
        let mut m = StaticMemory::new();
        let t0 = Instant::now();
        let cfg = on();
        let both = || vec![det(10.0, 0), det(300.0, 0)];
        for _ in 0..3 {
            m.apply("k", both(), &names(), &cfg, t0);
        }
        let (out, n) = m.apply("k", both(), &names(), &cfg, t0);
        assert_eq!((out.len(), n), (0, 2));
        let off = StaticSuppression::default();
        let (out, n) = m.apply("k", both(), &names(), &off, t0);
        assert_eq!((out.len(), n), (2, 0));
        assert_eq!(m.keys(), 0, "disabling forgets");
    }

    #[test]
    fn memory_is_bounded() {
        let mut m = StaticMemory::new();
        let t0 = Instant::now();
        let cfg = on();
        for k in 0..(MAX_KEYS + 10) {
            m.apply(&format!("k{k}"), vec![det(10.0, 0)], &names(), &cfg, t0);
        }
        assert_eq!(m.keys(), MAX_KEYS);
        assert_eq!(m.tracks("k0"), 0, "least recently used key evicted");
        assert_eq!(m.tracks(&format!("k{}", MAX_KEYS + 9)), 1);
        let many: Vec<Detection> = (0..(MAX_TRACKS_PER_KEY + 20))
            .map(|i| Detection {
                x1: 0.0,
                y1: 100.0 * i as f32,
                x2: 50.0,
                y2: 100.0 * i as f32 + 50.0,
                score: 0.8,
                class_id: 0,
            })
            .collect();
        m.apply("many", many, &names(), &cfg, t0);
        assert_eq!(m.tracks("many"), MAX_TRACKS_PER_KEY);
        // Boxes per track are capped too.
        let cfg = StaticSuppression {
            hits: u32::try_from(MAX_BOXES_PER_TRACK).unwrap(),
            ..on()
        };
        let mut m = StaticMemory::new();
        for _ in 0..(MAX_BOXES_PER_TRACK * 2) {
            m.apply("k", vec![det(10.0, 0)], &names(), &cfg, t0);
        }
        assert_eq!(m.keys["k"].tracks[0].seen.len(), MAX_BOXES_PER_TRACK);
        assert_eq!(m.apply("k", vec![det(10.0, 0)], &names(), &cfg, t0).1, 1);
    }

    #[test]
    fn keys_from_image_names() {
        assert_eq!(
            camera_key("m", "Driveway.20240101_120000.jpg", 640, 480),
            "m|cam:driveway"
        );
        assert_eq!(
            camera_key("m", "Driveway_0042.jpg", 640, 480),
            "m|cam:driveway"
        );
        assert_eq!(
            camera_key("m", "C:\\bi\\Front-Door.jpg", 1, 1),
            "m|cam:front-door"
        );
        assert_eq!(camera_key("m", "image.jpg", 640, 480), "m|640x480");
        assert_eq!(camera_key("m", "", 1920, 1080), "m|1920x1080");
        assert_eq!(camera_key("m", "12345.jpg", 10, 20), "m|10x20");
    }

    #[test]
    fn config_defaults_and_validation() {
        let d: StaticSuppression = serde_json::from_str(r#"{"enabled":true}"#).unwrap();
        assert!(d.enabled && d.hits == 5 && d.iou == 0.85 && d.window_minutes == 30);
        assert_eq!(d.labels, vec!["person"]);
        assert!(StaticSuppression::default().is_default());
        assert!(d.validate("static_suppression").is_ok());
        let bad = |f: fn(&mut StaticSuppression)| {
            let mut s = StaticSuppression::default();
            f(&mut s);
            s.validate("static_suppression").unwrap_err().to_string()
        };
        assert!(bad(|s| s.iou = 1.5).contains("static_suppression.iou"));
        assert!(bad(|s| s.iou = 0.0).contains("static_suppression.iou"));
        assert!(bad(|s| s.hits = 0).contains("hits"));
        assert!(bad(|s| s.hits = 1000).contains("hits"));
        assert!(bad(|s| s.window_minutes = 0).contains("window_minutes"));
        assert!(bad(|s| s.labels = vec![" ".into()]).contains("labels"));
    }
}
