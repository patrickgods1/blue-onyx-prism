//! How long the review walks take on a big folder, and what a compact index of it would weigh.
//! A measurement, not a check (`docs/PRODUCTION_REVIEW.md`, "Review walk measurements"): it prints
//! tables and does nothing unless `BOP_WALK_FRAMES` is set, so `--ignored` runs in CI skip it. A
//! debug build says nothing about these costs, so use `--release`:
//!
//! ```text
//! BOP_WALK_FRAMES=10000,50000 cargo test --release --test it review_walk -- --ignored --nocapture
//! ```
//!
//! - `BOP_WALK_FRAMES`: sizes, comma separated.
//! - `BOP_WALK_ROOT`: the folder to build the synthetic datasets in (default the temp folder); each
//!   is removed when done. Depth matters: `canonicalize` looks at every path component.
//! - `BOP_WALK_DATA`: use this one folder and keep it: it is generated when it has no
//!   `review-queue.jsonl`, so a later process can time it with the file cache dropped.
//! - `BOP_WALK_ONLY`: `generate` stops after generating; a walk's name (see [`walks`]) runs that
//!   walk once and stops, for a cold number per walk.
//!
//! The pass called `first` is not disk-cold: the files were just written, so the OS file cache
//! holds them. The frames are made up, from the real structs written the way the service writes
//! them; `FrameMix` states what a frame holds.

use blue_onyx_prism::backend::memory::process_memory;
use blue_onyx_prism::collect::labels::bulk::{self, Wanted};
use blue_onyx_prism::collect::labels::overlaps::{Suggestion, Verdict};
use blue_onyx_prism::collect::labels::queue::{
    self, AutoFilter, Candidate, Filters, QueueSource, StatusFilter,
};
use blue_onyx_prism::collect::labels::{
    self, AutoInfo, AutoRun, BoxState, LabelBox, LabelFile, MAX_LABEL_FILE_BYTES,
    MAX_SIDECAR_BYTES, Origin, Source, Status, Triage, VlmCheck, stacks,
};
use blue_onyx_prism::collect::layout::{self, ReviewLine, Summary};
use blue_onyx_prism::collect::record::{
    Bucket, Det, ImageInfo, IndexLine, RECORD_VERSION, Record, Stats, Suppressed, VerifyCandidate,
    VerifyInfo,
};
use serde::Deserialize;
use serde::de::IgnoredAny;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const CAMERAS: [&str; 8] = [
    "driveway",
    "back_yard",
    "front_door",
    "garage",
    "side_gate",
    "street",
    "porch",
    "alley",
];
const DAYS: u64 = 30;
const CLASSES: [&str; 8] = [
    "person",
    "car",
    "truck",
    "dog",
    "cat",
    "bicycle",
    "motorcycle",
    "bird",
];

/// What a frame holds, in percent, for the tables to state.
struct FrameMix;

impl FrameMix {
    /// Frames with a detection in the reply; the others are `empty` frames.
    const DETECTED: u64 = 80;
    /// Frames with a label file (an auto-label run went over them).
    const LABELLED: u64 = 85;
    /// Of the label files: reviewed, then skipped; the rest wait for review.
    const REVIEWED: u64 = 15;
    const SKIPPED: u64 = 5;
    /// Boxes carrying the auto-labeler's evidence.
    const EVIDENCE: u64 = 70;
    /// Label files with a stacked pair, half of them with an overlap suggestion.
    const OVERLAPPING: u64 = 6;
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "a made-up score, under 1000 thousandths"
    )]
    fn score(&mut self, low: f32) -> f32 {
        let at = self.below(1000) as f32 / 1000.0;
        ((low + (1.0 - low) * at) * 1000.0).round() / 1000.0
    }
    fn pick<'a>(&mut self, list: &[&'a str]) -> &'a str {
        let i = usize::try_from(self.below(list.len() as u64)).unwrap();
        list[i]
    }
    #[allow(clippy::cast_precision_loss, reason = "pixels of a 1080p frame")]
    fn bbox(&mut self) -> [f32; 4] {
        let (x, y) = (self.below(1500) as f32, self.below(800) as f32);
        let (w, h) = (20 + self.below(300), 20 + self.below(250));
        [x + 0.3, y + 0.7, x + w as f32 + 0.1, y + h as f32 + 0.4]
    }
}

fn det(rng: &mut Rng, low: f32) -> Det {
    Det {
        label: rng.pick(&CLASSES).to_string(),
        score: rng.score(low),
        bbox: rng.bbox(),
    }
}

/// A sidecar as the collector writes it.
fn record(rng: &mut Rng, id: &str, scene: &str, camera: &str, timestamp: String) -> Record {
    let raw: Vec<Det> = if rng.chance(FrameMix::DETECTED) {
        (0..1 + rng.below(3)).map(|_| det(rng, 0.5)).collect()
    } else {
        Vec::new()
    };
    let suppressed: Vec<Suppressed> = raw
        .iter()
        .filter(|_| rng.chance(10))
        .map(|d| Suppressed {
            label: d.label.clone(),
            score: d.score,
            bbox: d.bbox,
            reason: "aspect".into(),
        })
        .collect();
    let mut buckets = vec![if raw.is_empty() {
        Bucket::Empty
    } else {
        Bucket::Positive
    }];
    if !suppressed.is_empty() {
        buckets.push(Bucket::Suppressed);
    }
    let below: Vec<Det> = (0..rng.below(3)).map(|_| det(rng, 0.4)).collect();
    if !below.is_empty() {
        buckets.push(Bucket::NearThreshold);
    }
    let verify = (scene != "day" && !raw.is_empty() && rng.chance(30)).then(|| VerifyInfo {
        verifier: "rfdetr-large".into(),
        mode: "crop".into(),
        failure: None,
        candidates: raw
            .iter()
            .map(|d| VerifyCandidate {
                label: d.label.clone(),
                score: d.score,
                bbox: d.bbox,
                outcome: "kept".into(),
            })
            .collect(),
        found: Vec::new(),
        min_confidence: 0.4,
        iou: 0.5,
    });
    Record {
        version: RECORD_VERSION,
        id: id.to_string(),
        request_id: id.rsplit('_').next().unwrap().to_string(),
        timestamp,
        model: "IPcam-general".into(),
        device: "openvino:gpu".into(),
        routed_from: None,
        image_name: format!("{camera}.20261022_123456789.jpg"),
        camera: camera.to_string(),
        camera_key: format!("IPcam-general|cam:{camera}"),
        scene: scene.to_string(),
        stats: Stats {
            mean_luma: 83.21,
            chroma_spread: 11.4,
            tint_spread: 9.87,
            is_ir: scene == "ir",
            is_dark: scene != "day",
        },
        width: 1920,
        height: 1080,
        image: ImageInfo {
            file: format!("{id}.jpg"),
            bytes: 187_233,
            original: true,
        },
        threshold: 0.5,
        near_margin: Some(0.1),
        predictions: raw.clone(),
        raw,
        below_threshold: below,
        suppressed,
        verify,
        matched: buckets.clone(),
        buckets,
        dhash: format!("{:016x}", rng.next()),
        source: None,
        provenance: None,
    }
}

/// An auto box with the evidence a labelers run leaves.
fn evidence(rng: &mut Rng) -> AutoInfo {
    let single = rng.chance(40);
    let mut labelers = BTreeMap::from([("gdino".to_string(), rng.score(0.2))]);
    if !single {
        labelers.insert("rfdetr".to_string(), rng.score(0.2));
    }
    AutoInfo {
        agree: Some(!single),
        triage: Some(if !single && rng.chance(50) {
            Triage::Accept
        } else {
            Triage::Review
        }),
        why: if single {
            vec!["single labeler".into(), "low score".into()]
        } else {
            Vec::new()
        },
        labelers,
        vlm: Some(VlmCheck {
            model: "qwen3-vl-2b".into(),
            answer: "yes".into(),
            expected: "yes".into(),
            ok: rng.chance(85),
        }),
        likely: rng.chance(25),
        ..AutoInfo::default()
    }
}

fn label_box(rng: &mut Rng, status: Status) -> LabelBox {
    let auto = if rng.chance(FrameMix::EVIDENCE) {
        evidence(rng)
    } else {
        AutoInfo::default()
    };
    let det = det(rng, 0.2);
    let decided = rng.chance(75);
    LabelBox {
        label: det.label,
        bbox: det.bbox,
        source: if auto.is_empty() {
            Source::Model
        } else {
            Source::Auto
        },
        score: Some(det.score),
        state: match status {
            Status::Unreviewed => BoxState::Pending,
            _ if decided => BoxState::Accepted,
            _ => BoxState::Rejected,
        },
        origin: auto.is_empty().then_some(Origin::Prediction),
        reason: None,
        model_label: None,
        orig_bbox: None,
        difficult: false,
        reject_reason: None,
        auto,
    }
}

/// A label file as an auto-label run, then maybe a reviewer, leave it.
fn label_file(rng: &mut Rng) -> LabelFile {
    let roll = rng.below(100);
    let status = if roll < FrameMix::REVIEWED {
        Status::Reviewed
    } else if roll < FrameMix::REVIEWED + FrameMix::SKIPPED {
        Status::Skipped
    } else {
        Status::Unreviewed
    };
    let mut boxes: Vec<LabelBox> = (0..1 + rng.below(6))
        .map(|_| label_box(rng, status))
        .collect();
    let mut overlap_suggestions = Vec::new();
    if rng.chance(FrameMix::OVERLAPPING) {
        let mut twin = boxes[0].clone();
        twin.bbox[2] -= 2.0;
        twin.bbox[3] -= 1.0;
        if rng.chance(50) {
            overlap_suggestions.push(Suggestion {
                a: boxes[0].bbox,
                b: twin.bbox,
                verdict: Verdict::DropB,
                why: "RF-DETR L on the zoomed pair kept one box".into(),
                kept_score: Some(rng.score(0.5)),
                other_score: Some(0.0),
                low_confidence: false,
                weak: false,
            });
        }
        boxes.push(twin);
    }
    LabelFile {
        version: 1,
        status,
        empty_confirmed: false,
        reviewed_at: (status != Status::Unreviewed).then(|| "2026-10-29T10:11:12Z".into()),
        classes_version: "6f1ed002ab5595b9".into(),
        disagreement: None,
        auto_run: Some(AutoRun {
            at: "2026-10-28T02:03:04Z".into(),
            labeler: "mm-gdino-tiny + rfdetr-large + qwen3-vl-2b".into(),
            classes: Vec::new(),
        }),
        bulk_accepted: false,
        repeat_spot: false,
        overlaps_checked: false,
        checked_before_reopen: false,
        overlap_suggestions,
        boxes,
    }
}

/// What [`generate`] wrote.
#[derive(Default)]
struct Written {
    sidecar_bytes: u64,
    label_bytes: u64,
    label_files: u64,
    index_bytes: u64,
    folders: usize,
}

/// `frames` frames over [`CAMERAS`], [`DAYS`] days and the three scenes, all in the review queue
/// (the worst case of the default queue). The image is a stub: no walk reads it.
fn generate(dir: &Path, frames: u64) -> Written {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut out = Written::default();
    let mut index: BTreeMap<PathBuf, String> = BTreeMap::new();
    let mut review = String::new();
    for _ in 0..frames {
        let camera = rng.pick(&CAMERAS);
        let scene = match rng.below(10) {
            0..5 => "day",
            5..8 => "night",
            _ => "ir",
        };
        let day = 1 + rng.below(DAYS);
        let (h, m, s, ms) = (rng.below(24), rng.below(60), rng.below(60), rng.below(1000));
        let date = format!("2026-10-{day:02}");
        let id = format!(
            "202610{day:02}T{h:02}{m:02}{s:02}{ms:03}Z_{:08x}",
            rng.next() & 0xFFFF_FFFF
        );
        let timestamp = format!("{date}T{h:02}:{m:02}:{s:02}.{ms:03}Z");
        let folder = dir.join(scene).join(camera).join(&date);
        let rec = record(&mut rng, &id, scene, camera, timestamp);
        if !index.contains_key(&folder) {
            std::fs::create_dir_all(&folder).unwrap();
        }
        std::fs::write(folder.join(format!("{id}.jpg")), [0xFF, 0xD8, 0xFF, 0xD9]).unwrap();
        let json = serde_json::to_vec_pretty(&rec).unwrap();
        out.sidecar_bytes += json.len() as u64;
        std::fs::write(folder.join(format!("{id}.json")), json).unwrap();
        if rng.chance(FrameMix::LABELLED) {
            let mut text = serde_json::to_vec_pretty(&label_file(&mut rng)).unwrap();
            text.push(b'\n');
            out.label_bytes += text.len() as u64;
            out.label_files += 1;
            std::fs::write(folder.join(format!("{id}{}", labels::LABELS_SUFFIX)), text).unwrap();
        }
        let line = serde_json::to_string(&IndexLine::of(&rec, 187_233)).unwrap();
        writeln!(index.entry(folder).or_default(), "{line}").unwrap();
        let queued = ReviewLine {
            version: 1,
            frame: format!("{scene}/{camera}/{date}/{id}.jpg"),
            sidecar: format!("{scene}/{camera}/{date}/{id}.json"),
            id,
            buckets: rec.buckets,
            scene: scene.to_string(),
            camera: camera.to_string(),
            queued_at: rec.timestamp,
        };
        writeln!(review, "{}", serde_json::to_string(&queued).unwrap()).unwrap();
    }
    out.folders = index.len();
    for (folder, text) in index {
        out.index_bytes += text.len() as u64;
        std::fs::write(folder.join(layout::INDEX_FILE), text).unwrap();
    }
    std::fs::write(dir.join(layout::REVIEW_QUEUE_FILE), review).unwrap();
    out
}

fn class_list() -> Vec<String> {
    CLASSES.map(String::from).to_vec()
}

fn wanted_filters(f: Filters) -> Filters {
    Filters {
        wanted: Some((class_list(), Vec::new())),
        ..f
    }
}

/// `/v1/review/next`'s work with `f`: the queue, then the damaged-line count.
fn next(dir: &Path, f: &Filters) -> usize {
    let built = queue::build(dir, f);
    let damaged = layout::damaged_lines(dir, f.source == QueueSource::All);
    assert_eq!(damaged, 0);
    built.items.len()
}

type Walk<'a> = (&'static str, Box<dyn Fn() -> usize + 'a>);

/// Every walk measured, by name.
fn walks(dir: &Path) -> Vec<Walk<'_>> {
    let default = wanted_filters(Filters::default());
    let every = wanted_filters(Filters {
        source: QueueSource::All,
        status: StatusFilter::Any,
        ..Filters::default()
    });
    let overlaps = Filters {
        auto: Some(AutoFilter::Overlaps),
        ..every.clone()
    };
    let one_camera = Filters {
        camera: Some(CAMERAS[0].to_string()),
        ..default.clone()
    };
    vec![
        ("build", Box::new(move || next(dir, &default))),
        ("build_all", Box::new(move || next(dir, &every))),
        ("build_overlaps", Box::new(move || next(dir, &overlaps))),
        ("build_one_camera", Box::new(move || next(dir, &one_camera))),
        ("tally", Box::new(move || queue::tally(dir).frames as usize)),
        ("scan", Box::new(move || Summary::scan(dir).frames as usize)),
        (
            "bulk",
            Box::new(move || {
                let classes = class_list();
                let wanted = Wanted {
                    classes: &classes,
                    hidden: &[],
                };
                let f = Filters::default();
                let done = bulk::bulk_reject_stacked(dir, &f, &wanted, usize::MAX, true, &|_| true);
                done.overlaps_left
            }),
        ),
        (
            "candidates",
            Box::new(move || queue::candidates(dir, QueueSource::All).len()),
        ),
        (
            "label_files",
            Box::new(move || labels::label_files(dir).len()),
        ),
        ("split", Box::new(move || print_split(dir))),
    ]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn timed<T>(f: impl FnOnce() -> T) -> (Duration, T) {
    let t = Instant::now();
    let out = f();
    (t.elapsed(), out)
}

/// The middle of three passes.
fn warm(mut f: impl FnMut() -> Duration) -> Duration {
    let mut passes = [f(), f(), f()];
    passes.sort();
    passes[1]
}

/// The per-frame work of [`queue::build`], split by what it costs.
#[derive(Default)]
struct Split {
    candidates: Duration,
    parse_rel: Duration,
    lstat: Duration,
    canonicalize: Duration,
    read: Duration,
    parse_sidecar: Duration,
    parse_labels: Duration,
}

/// The file work of [`queue::build`] one call at a time, in its order: `Frame::resolve` (a path
/// check, then `layout::resolve` for the image and for the sidecar, a label file lstat), then the
/// sidecar, then the label file. The timers cost about 40 ns each here, 15 per frame.
fn split(dir: &Path, source: QueueSource) -> (Split, u64) {
    let mut s = Split::default();
    let (t, candidates) = timed(|| queue::candidates(dir, source));
    s.candidates = t;
    let mut n = 0;
    macro_rules! tick {
        ($field:ident, $e:expr) => {{
            let t = Instant::now();
            let out = $e;
            s.$field += t.elapsed();
            out
        }};
    }
    for c in candidates {
        let r = tick!(parse_rel, layout::parse_rel(&c.rel)).unwrap();
        for ext in ["jpg", "json"] {
            let rel = if ext == "jpg" { r.jpg() } else { r.json() };
            tick!(parse_rel, layout::parse_rel(&rel)).unwrap();
            let p = r.to_path(dir, ext);
            tick!(lstat, std::fs::symlink_metadata(&p)).unwrap();
            tick!(canonicalize, dir.canonicalize()).unwrap();
            tick!(canonicalize, p.canonicalize()).unwrap();
        }
        let folder = dir.join(&r.scene).join(&r.camera).join(&r.date);
        let label_path = folder.join(format!("{}{}", r.stem, labels::LABELS_SUFFIX));
        let exists = tick!(lstat, std::fs::symlink_metadata(&label_path)).is_ok();
        let side = r.to_path(dir, "json");
        let bytes = tick!(read, labels::read_capped(&side, MAX_SIDECAR_BYTES)).unwrap();
        let rec: Record = tick!(parse_sidecar, serde_json::from_slice(&bytes)).unwrap();
        std::hint::black_box(&rec);
        tick!(lstat, std::fs::symlink_metadata(&label_path)).ok();
        if exists {
            let bytes =
                tick!(read, labels::read_capped(&label_path, MAX_LABEL_FILE_BYTES)).unwrap();
            let l: LabelFile = tick!(parse_labels, serde_json::from_slice(&bytes)).unwrap();
            std::hint::black_box(&l);
        }
        n += 1;
    }
    (s, n)
}

/// The sidecar fields [`queue::build`] reads; serde still scans the others.
#[derive(Deserialize)]
struct ThinRecord {
    timestamp: String,
    model: String,
    threshold: f32,
    raw: Vec<ThinDet>,
    #[serde(default)]
    below_threshold: Vec<ThinDet>,
    suppressed: Vec<IgnoredAny>,
    predictions: Vec<ThinDet>,
    #[serde(default)]
    verify: Option<ThinVerify>,
    buckets: Vec<Bucket>,
    #[serde(default)]
    matched: Vec<Bucket>,
}

#[derive(Deserialize)]
struct ThinDet {
    label: String,
    score: f32,
}

#[derive(Deserialize)]
struct ThinVerify {
    candidates: Vec<ThinCandidate>,
}

#[derive(Deserialize)]
struct ThinCandidate {
    outcome: String,
}

/// One candidate fix to the per-frame file work.
#[derive(Clone, Copy)]
struct Fix {
    name: &'static str,
    /// Canonicalize the root once and each day folder once, instead of four times a frame.
    one_canonicalize: bool,
    /// No lstat at all, a stat of the image instead: this only shows what those calls cost, the
    /// lstats are what refuses a symlink.
    no_lstat: bool,
    thin_sidecar: bool,
    threads: usize,
}

/// A [`queue::build`] without its sort and counts, changed by `fix`, over `candidates`.
fn file_work(dir: &Path, candidates: &[Candidate], fix: Fix) -> usize {
    let root = dir.canonicalize().unwrap();
    let chunk = candidates.len().div_ceil(fix.threads).max(1);
    let root = &root;
    std::thread::scope(|scope| {
        let handles: Vec<_> = candidates
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let mut days: HashMap<String, PathBuf> = HashMap::new();
                    let mut done = 0;
                    for c in part {
                        done += usize::from(one_frame(dir, root, c, fix, &mut days));
                    }
                    done
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

fn one_frame(
    dir: &Path,
    root: &Path,
    c: &Candidate,
    fix: Fix,
    days: &mut HashMap<String, PathBuf>,
) -> bool {
    let Some(r) = layout::parse_rel(&c.rel) else {
        return false;
    };
    let side = r.to_path(dir, "json");
    let jpg = r.to_path(dir, "jpg");
    if !fix.no_lstat {
        for p in [&jpg, &side] {
            if std::fs::symlink_metadata(p).is_err() {
                return false;
            }
        }
    } else if std::fs::metadata(&jpg).is_err() {
        // The walk must still drop a frame whose image is gone.
        return false;
    }
    let folder = if fix.one_canonicalize {
        let day = r.day_dir();
        days.entry(day)
            .or_insert_with_key(|d| dir.join(d).canonicalize().unwrap())
            .clone()
    } else {
        for p in [&jpg, &side] {
            let (a, b) = (dir.canonicalize().unwrap(), p.canonicalize().unwrap());
            assert!(b.starts_with(&a));
        }
        jpg.parent().unwrap().to_path_buf()
    };
    assert!(folder.starts_with(root));
    let bytes =
        labels::read_capped(&folder.join(format!("{}.json", r.stem)), MAX_SIDECAR_BYTES).unwrap();
    if fix.thin_sidecar {
        let rec: ThinRecord = serde_json::from_slice(&bytes).unwrap();
        std::hint::black_box((
            &rec.timestamp,
            &rec.model,
            rec.threshold,
            rec.raw.first().map(|d| (&d.label, d.score)),
            rec.below_threshold.len(),
            rec.suppressed.len(),
            rec.predictions.len(),
            rec.verify
                .as_ref()
                .and_then(|v| v.candidates.first().map(|c| &c.outcome)),
            rec.buckets.len() + rec.matched.len(),
        ));
    } else {
        let rec: Record = serde_json::from_slice(&bytes).unwrap();
        std::hint::black_box(&rec);
    }
    let label_path = folder.join(format!("{}{}", r.stem, labels::LABELS_SUFFIX));
    if fix.no_lstat {
        if let Ok(bytes) = labels::read_capped(&label_path, MAX_LABEL_FILE_BYTES) {
            let l: LabelFile = serde_json::from_slice(&bytes).unwrap();
            std::hint::black_box(&l);
        }
    } else {
        let l = labels::read(&label_path).unwrap();
        std::hint::black_box(&l);
    }
    true
}

/// One frame of a compact index: everything [`queue::build`], its filters except the two that
/// need boxes, [`queue::sort_items`] and the counts read. Text lives in an arena, classes as ids.
#[derive(Clone, Copy, Default)]
#[allow(
    dead_code,
    reason = "the size is the point; only the query reads some fields"
)]
struct Row {
    ts_ms: u64,
    stem_at: u32,
    classes_at: u32,
    boxes_at: u32,
    suggestions_at: u32,
    camera: u16,
    date: u16,
    model: u16,
    /// queued, has a label file, empty confirmed, auto run, evidence, auto review, auto accept,
    /// repeats only, dismissed box, overlaps checked, verifier disagreement, relabel note.
    flags: u16,
    stem_len: u8,
    scene: u8,
    status: u8,
    buckets: u8,
    /// Hundredths; 255 = no box.
    margin: u8,
    suppressed: u8,
    auto_review: u8,
    likely: u8,
    /// Distinct classes ([`queue::frame_labels`]), then the training boxes' classes.
    classes: u8,
    training: u8,
    rejected: u8,
    audit_accepted: u8,
    audit_rejected: u8,
    boxes: u8,
    suggestions: u8,
}

/// What `auto=overlaps`, `auto=both` and `overlaps::live` read of a box, for any class list.
#[derive(Clone, Copy, Default)]
#[allow(dead_code, reason = "the size is the point")]
struct BoxRow {
    bbox: [f32; 4],
    label: u16,
    /// The spelling a relabel left (`overlaps::guarded`).
    model_label: u16,
    /// State, source, moved (`orig_bbox`).
    flags: u8,
    labelers: u8,
}

/// What `overlaps::live` and `strength` read of a suggestion.
#[derive(Clone, Copy, Default)]
#[allow(dead_code, reason = "the size is the point")]
struct SuggestionRow {
    a: [f32; 4],
    b: [f32; 4],
    kept_score: f32,
    other_score: f32,
    /// Verdict, low confidence, weak.
    flags: u8,
}

#[derive(Default)]
struct Compact {
    rows: Vec<Row>,
    stems: Vec<u8>,
    classes: Vec<u16>,
    /// Boxes of the frames with two or more live boxes, or an agreed box.
    boxes: Vec<BoxRow>,
    all_boxes: u64,
    suggestions: Vec<SuggestionRow>,
    /// `LabelFile::disagreement`: the page shows it, the sort only tests it.
    notes: Vec<u8>,
    names: HashMap<String, u16>,
}

fn intern(names: &mut HashMap<String, u16>, name: &str) -> u16 {
    let next = u16::try_from(names.len()).unwrap();
    *names.entry(name.to_string()).or_insert(next)
}

fn small(n: usize) -> u8 {
    u8::try_from(n).unwrap_or(u8::MAX)
}

impl Compact {
    fn add(&mut self, c: &Candidate, rec: &Record, l: Option<&LabelFile>) {
        let r = layout::parse_rel(&c.rel).unwrap();
        let classes = queue::frame_labels(rec, l);
        let classes_at = u32::try_from(self.classes.len()).unwrap();
        for name in &classes {
            let id = intern(&mut self.names, name);
            self.classes.push(id);
        }
        let u = queue::Uncertainty::of(rec).with_labels(l);
        let mut row = Row {
            ts_ms: 0,
            stem_at: u32::try_from(self.stems.len()).unwrap(),
            classes_at,
            boxes_at: u32::try_from(self.boxes.len()).unwrap(),
            suggestions_at: u32::try_from(self.suggestions.len()).unwrap(),
            camera: intern(&mut self.names, &c.camera),
            date: intern(&mut self.names, &r.date),
            model: intern(&mut self.names, &rec.model),
            stem_len: small(r.stem.len()),
            buckets: c.buckets.iter().fold(0, |m, b| m | 1 << (*b as u8)),
            margin: u.margin.map_or(255, |m| small((m * 100.0) as usize)),
            suppressed: small(u.suppressed),
            auto_review: small(u.auto_review),
            likely: small(u.likely_boxes as usize),
            classes: small(classes.len()),
            flags: u16::from(c.queued) | u16::from(u.disagreement) << 1,
            ..Row::default()
        };
        self.stems.extend_from_slice(r.stem.as_bytes());
        if let Some(l) = l {
            row.status = l.status as u8;
            row.flags |= 1 << 2;
            row.flags |= u16::from(queue::AutoFilter::Accept.matches(Some(l))) << 3;
            row.flags |= u16::from(queue::AutoFilter::Repeats.matches(Some(l))) << 4;
            row.flags |= u16::from(queue::AutoFilter::Dismissed.matches(Some(l))) << 5;
            row.flags |= u16::from(l.auto_run.is_some()) << 6;
            row.flags |= u16::from(l.boxes.iter().any(|b| !b.auto.is_empty())) << 7;
            for b in l.training() {
                let id = intern(&mut self.names, &b.label);
                self.classes.push(id);
                row.training = row.training.saturating_add(1);
            }
            row.rejected = small(l.hard_negatives().count());
            if let Some(note) = &l.disagreement {
                self.notes.extend_from_slice(note.as_bytes());
            }
            self.all_boxes += l.boxes.len() as u64;
            let live = l
                .boxes
                .iter()
                .filter(|b| b.state != BoxState::Rejected)
                .count();
            let agreed = l.boxes.iter().any(|b| b.auto.labelers.len() >= 2);
            if live >= 2 || agreed {
                for b in &l.boxes {
                    let label = intern(&mut self.names, &b.label);
                    self.boxes.push(BoxRow {
                        bbox: b.bbox,
                        label,
                        model_label: 0,
                        flags: b.state as u8,
                        labelers: small(b.auto.labelers.len()),
                    });
                    row.boxes = row.boxes.saturating_add(1);
                }
            }
            for s in &l.overlap_suggestions {
                self.suggestions.push(SuggestionRow {
                    a: s.a,
                    b: s.b,
                    kept_score: s.kept_score.unwrap_or(0.0),
                    other_score: s.other_score.unwrap_or(0.0),
                    flags: s.verdict as u8,
                });
                row.suggestions = row.suggestions.saturating_add(1);
            }
        }
        self.rows.push(row);
    }

    /// The bytes of the frame rows, of the text and class arenas, and of the path lookup (a save
    /// finds its row by hash: `HashMap<u64, u32>`, its capacity at the usual load).
    fn core_bytes(&self) -> usize {
        let lookup = self.rows.len() * 8 / 7 * (size_of::<(u64, u32)>() + 1);
        self.rows.len() * size_of::<Row>()
            + self.stems.len()
            + self.classes.len() * size_of::<u16>()
            + self.notes.len()
            + self.names.keys().map(|k| k.len() + 48).sum::<usize>()
            + lookup
    }

    fn box_bytes(&self) -> usize {
        self.boxes.len() * size_of::<BoxRow>()
    }

    fn suggestion_bytes(&self) -> usize {
        self.suggestions.len() * size_of::<SuggestionRow>()
    }

    /// What `/v1/review/next` would do with the index: filter, count per class, sort.
    fn query(&self, status: u8) -> usize {
        let mut per_class: HashMap<u16, u64> = HashMap::new();
        let mut hits: Vec<(u16, u8, u8, u32)> = Vec::with_capacity(self.rows.len());
        for (i, row) in self.rows.iter().enumerate() {
            let at = row.classes_at as usize + usize::from(row.classes);
            for id in &self.classes[at..at + usize::from(row.training)] {
                *per_class.entry(*id).or_default() += 1;
            }
            if row.status == status {
                let key = u16::from(row.likely) << 8 | u16::from(row.auto_review);
                hits.push((key, row.margin, row.suppressed, u32::try_from(i).unwrap()));
            }
        }
        hits.sort_unstable();
        std::hint::black_box(per_class.len());
        hits.len()
    }
}

fn compact(dir: &Path) -> Compact {
    let mut index = Compact::default();
    for c in queue::candidates(dir, QueueSource::All) {
        let frame = labels::Frame::resolve(dir, &c.rel).unwrap();
        let rec = frame.record().unwrap();
        let l = frame.labels().unwrap();
        index.add(&c, &rec, l.as_ref());
    }
    index.rows.shrink_to_fit();
    index.stems.shrink_to_fit();
    index.classes.shrink_to_fit();
    index.boxes.shrink_to_fit();
    index.suggestions.shrink_to_fit();
    index
}

/// The table of [`split`].
#[allow(clippy::cast_precision_loss, reason = "a printed ratio")]
fn print_split(dir: &Path) -> usize {
    let (s, n) = split(dir, QueueSource::All);
    let parts = [
        ("candidates (jsonl, dedupe)", s.candidates),
        ("parse_rel x3", s.parse_rel),
        ("lstat x4", s.lstat),
        ("canonicalize x4", s.canonicalize),
        ("open + read x2", s.read),
        ("parse sidecar", s.parse_sidecar),
        ("parse label file", s.parse_labels),
    ];
    let total: Duration = parts.iter().map(|p| p.1).sum();
    println!(
        "\nSplit of the file work, {n} frames: {:.0} ms\n",
        ms(total)
    );
    println!("| part | ms | share | us/frame |\n|---|---|---|---|");
    for (name, d) in parts {
        println!(
            "| {name} | {:.0} | {:.1}% | {:.1} |",
            ms(d),
            100.0 * d.as_secs_f64() / total.as_secs_f64(),
            ms(d) * 1000.0 / n as f64,
        );
    }
    println!();
    usize::try_from(n).unwrap()
}

#[allow(clippy::cast_precision_loss, reason = "a printed ratio")]
fn report(dir: &Path, frames: u64, written: Option<&Written>) {
    let per_frame = |d: Duration| ms(d) / frames as f64;
    println!("\n## {frames} frames in {}\n", dir.display());
    if let Some(w) = written {
        println!(
            "{} day folders; sidecar {} bytes, label file {} bytes on {} frames, index line {} \
             bytes (means)\n",
            w.folders,
            w.sidecar_bytes / frames,
            w.label_bytes / w.label_files.max(1),
            w.label_files,
            w.index_bytes / frames,
        );
    }
    println!("| walk | first ms | warm ms | warm ms/frame | frames |\n|---|---|---|---|---|");
    for (name, walk) in walks(dir).iter().filter(|(n, _)| *n != "split") {
        let (first, n) = timed(walk);
        let rest = warm(|| timed(walk).0);
        println!(
            "| {name} | {:.0} | {:.0} | {:.3} | {n} |",
            ms(first),
            ms(rest),
            per_frame(rest)
        );
    }

    print_split(dir);

    let candidates = queue::candidates(dir, QueueSource::All);
    let base = Fix {
        name: "as today",
        one_canonicalize: false,
        no_lstat: false,
        thin_sidecar: false,
        threads: 1,
    };
    let once = Fix {
        name: "one canonicalize per day folder",
        one_canonicalize: true,
        ..base
    };
    let bare = Fix {
        name: "that, no lstat (a cost, not a fix)",
        no_lstat: true,
        ..once
    };
    let thin = Fix {
        name: "that fix + thin sidecar struct",
        thin_sidecar: true,
        ..once
    };
    let cores = std::thread::available_parallelism().map_or(2, usize::from);
    let threads = Fix {
        name: "that + all cores but one",
        threads: (cores - 1).max(1),
        ..thin
    };
    let (mut before, mut rows) = (0.0, String::new());
    for fix in [base, once, bare, thin, threads] {
        let t = warm(|| timed(|| file_work(dir, &candidates, fix)).0);
        if fix.name == base.name {
            before = ms(t);
        }
        writeln!(
            rows,
            "| {} | {:.0} | {:.3} | {:.0}% |",
            fix.name,
            ms(t),
            per_frame(t),
            100.0 * ms(t) / before
        )
        .unwrap();
    }
    println!("\nFile work with one fix at a time (no sort, no counts)\n");
    println!("| variant | warm ms | ms/frame | of today |\n|---|---|---|---|\n{rows}");

    let every = wanted_filters(Filters {
        source: QueueSource::All,
        status: StatusFilter::Any,
        ..Filters::default()
    });
    let mut items = queue::build(dir, &every).items;
    let mut rng = Rng(7);
    let sort = warm(|| {
        for i in (1..items.len()).rev() {
            let j = usize::try_from(rng.below(i as u64 + 1)).unwrap();
            items.swap(i, j);
        }
        timed(|| {
            let mut copy = items.clone();
            queue::sort_items(&mut copy);
        })
        .0
    });
    println!("sort_items on a clone, shuffled: {:.1} ms", ms(sort));

    let (load, index) = timed(|| compact(dir));
    let rows = index.rows.len();
    let core = index.core_bytes();
    let boxes = index.box_bytes();
    let suggestions = index.suggestion_bytes();
    let total = core + boxes + suggestions;
    println!(
        "\nCompact index ({rows} frames): Row {} bytes, BoxRow {}, SuggestionRow {}; {} boxes of \
         {} kept ({:.1}/frame), {} suggestions; {} distinct names; built in {:.0} ms\n",
        size_of::<Row>(),
        size_of::<BoxRow>(),
        size_of::<SuggestionRow>(),
        index.boxes.len(),
        index.all_boxes,
        index.boxes.len() as f64 / rows as f64,
        index.suggestions.len(),
        index.names.len(),
        ms(load),
    );
    println!("| part | bytes/frame | MB here | MB at 200k frames |\n|---|---|---|---|");
    let at = |bytes: usize| {
        let per = bytes as f64 / rows as f64;
        format!(
            "{per:.0} | {:.1} | {:.1}",
            bytes as f64 / 1e6,
            per * 200_000.0 / 1e6
        )
    };
    println!("| rows, stems, classes, names, lookup | {} |", at(core));
    println!("| boxes (overlaps, both) | {} |", at(boxes));
    println!("| suggestions (strength) | {} |", at(suggestions));
    println!("| total | {} |", at(total));
    let query = warm(|| {
        timed(|| {
            std::hint::black_box(index.query(0));
        })
        .0
    });
    println!(
        "\nThe same query on the index (filter, count, sort; the index in memory): {:.2} ms",
        ms(query)
    );
    if let Some(m) = process_memory() {
        println!(
            "Process now: resident {:.0} MB, footprint {:?} MB",
            m.resident as f64 / 1e6,
            m.footprint.map(|b| b / 1_000_000)
        );
    }
}

/// [`stacks`] on one frame, so the [`FrameMix`] stated above is known to reach the overlap filter.
#[test]
fn the_synthetic_frames_reach_the_overlap_filter() {
    let mut rng = Rng(11);
    let files: Vec<LabelFile> = (0..500).map(|_| label_file(&mut rng)).collect();
    let n = files.iter().filter(|f| stacks::needs_look(f, None)).count();
    assert!(n > 5, "{n} of 500 frames need a look");
    assert!(files.iter().any(|f| f.auto_run.is_some()));
}

#[test]
#[ignore = "a measurement: set BOP_WALK_FRAMES, see the file's first lines"]
fn review_walks_on_a_big_folder() {
    let Ok(sizes) = std::env::var("BOP_WALK_FRAMES") else {
        println!("skipped: BOP_WALK_FRAMES is not set");
        return;
    };
    let only = std::env::var("BOP_WALK_ONLY").ok();
    println!(
        "frames: {}% detected, {}% with a label file ({}% reviewed, {}% skipped), {}% of boxes with \
         evidence, {}% with a stacked pair",
        FrameMix::DETECTED,
        FrameMix::LABELLED,
        FrameMix::REVIEWED,
        FrameMix::SKIPPED,
        FrameMix::EVIDENCE,
        FrameMix::OVERLAPPING,
    );
    for frames in sizes.split(',').map(|s| s.trim().parse::<u64>().unwrap()) {
        if let Ok(data) = std::env::var("BOP_WALK_DATA") {
            let dir = PathBuf::from(data);
            let written = (!dir.join(layout::REVIEW_QUEUE_FILE).exists()).then(|| {
                std::fs::create_dir_all(&dir).unwrap();
                let (t, written) = timed(|| generate(&dir, frames));
                println!("generated {frames} frames in {:.1} s", t.as_secs_f64());
                written
            });
            match only.as_deref() {
                Some("generate") => {}
                Some(name) => {
                    let list = walks(&dir);
                    let (_, walk) = list.iter().find(|(n, _)| *n == name).unwrap();
                    let (t, n) = timed(walk);
                    println!("{name}: {:.0} ms, {n} frames", ms(t));
                }
                None => report(&dir, frames, written.as_ref()),
            }
            continue;
        }
        let root = std::env::var_os("BOP_WALK_ROOT").map_or_else(std::env::temp_dir, PathBuf::from);
        let dir = root.join(format!("bop-walk-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let (t, written) = timed(|| generate(&dir, frames));
        println!("generated {frames} frames in {:.1} s", t.as_secs_f64());
        report(&dir, frames, Some(&written));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
