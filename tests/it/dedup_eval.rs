//! Offline evaluation of the near-duplicate filters on a real review folder: today's whole-frame
//! dHash ([`dhash`]) against the block test of [`blockdiff`]. A measurement, not a check: it does
//! nothing unless `BOP_DEDUP_EVAL_DIR` is set, so `--ignored` runs in CI pass without it. It only
//! reads the folder (no lock, no index, no write); everything goes to `BOP_DEDUP_EVAL_OUT`. Use
//! `--release`:
//!
//! ```text
//! BOP_DEDUP_EVAL_DIR=D:\dataset BOP_DEDUP_EVAL_OUT=<dir> \
//!     cargo test --release --test it dedup_eval -- --ignored --nocapture
//! ```
//!
//! - `BOP_DEDUP_EVAL_MAX_FRAMES` (default 40000): past it, whole camera days are sampled by a hash
//!   of their name until the count is reached (deterministic).
//! - `BOP_DEDUP_EVAL_DUMP`: also write the thumbnails, frames and pairs ([`dump`]) for trying
//!   variants outside Rust.
//!
//! **Frames**: reviewed (`reviewed_at` set), not reopened (no `disagreement`), no pending box, with a
//! sidecar and an image that decodes at the sidecar's size. Each has an *origin*: `collected` (saved
//! by the collector) or `import:<id>` (frames of imported footage; their times are footage times, so
//! they pair only with frames of the same import). **Pairs**: two frames of one origin, camera and
//! scene at most [`WINDOW_MS`] apart, of one size; the later one is the new frame, the earlier the
//! reference, as in the collector. A pair is *different* when a training box (accepted or added, no
//! ignore region) of either frame has no box of the same object ([`repeats::same_object`]) at an IoU
//! of [`MATCH_IOU`] or more in the other (training or ignore boxes: ignore regions are neutral),
//! stratified by the largest such box's share of the frame; else it is a *duplicate*.
//! *Missed-object rate* =
//! P(called duplicate | different), *kept-repeat rate* = P(called different | duplicate), with
//! [`BOOT`] bootstrap resamples of camera days for 95% intervals; every table is given for all
//! pairs, collected pairs and imported pairs. The replay runs the collector's keep logic (window,
//! [`KEEP`] references = earlier kept frames) per origin and camera in time order.
//!
//! Outputs: `report.json`, `report.md`, `misses.txt` (the worst missed objects of blockdiff and of
//! today's dHash), `repeats.txt` (blockdiff's worst kept repeats).

use blue_onyx_prism::collect::blockdiff::{self, NoiseMap, Params, Residual, Thumb, Verdict};
use blue_onyx_prism::collect::dhash;
use blue_onyx_prism::collect::labels::{self, BoxState, Frame, LabelFile, Status, repeats};
use blue_onyx_prism::image_stats::Scene;
use blue_onyx_prism::parallel;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The collector's dedup window ([`blue_onyx_prism::collect::store::DEDUP_WINDOW`]).
const WINDOW_MS: i64 = 10 * 60 * 1000;
/// References kept per camera ([`blue_onyx_prism::collect::store::DEDUP_KEEP`]).
const KEEP: usize = 32;
/// Same object at this IoU.
const MATCH_IOU: f32 = 0.5;
/// An unmatched box overlapping a same-object box of the other frame at this IoU moved (a shift),
/// rather than appeared.
const SHIFT_IOU: f32 = 0.1;
/// Bootstrap resamples.
const BOOT: usize = 1000;
/// Today's `collect.dedup_hamming`.
const DHASH_TODAY: u32 = 6;
/// Size buckets: upper bounds of the unmatched box's share of the frame.
const BUCKETS: [(&str, f64); 3] = [
    ("<0.05%", 0.0005),
    ("0.05-0.5%", 0.005),
    (">0.5%", f64::MAX),
];
/// The sweep: `z_cell`, one `d_min` for every scene, the smallest group, texture masking.
const Z_CELLS: [f32; 4] = [4.0, 5.0, 6.0, 8.0];
const D_MINS: [f32; 4] = [2.0, 4.0, 6.0, 8.0];
const MIN_COMPONENTS: [usize; 3] = [1, 2, 3];
const TEXTURE_BETAS: [f32; 4] = [0.0, 0.25, 0.5, 1.0];
/// The texture-masked variant reported beside the defaults.
const TEXTURE_VARIANT: f32 = 0.5;
/// The best sweep point minimises `MISS_WEIGHT * missed + kept` over all pairs: a lost object
/// costs more than a repeat to review.
const MISS_WEIGHT: f64 = 2.0;
/// Frames and pairs timed on one thread.
const TIME_FRAMES: usize = 60;
const TIME_PAIRS: usize = 2000;
/// The collector's rule (`collect.dedup_mode` `both`): dHash <= 6 and [`Params::COLLECTOR`] on
/// the same reference.
const COLLECTOR_BOTH: &str = "both: dhash<=6 + COLLECTOR";
/// Lines per section of the contact sheets.
const SHEET: usize = 20;

type Bx = (String, [f32; 4]);
type Point = (f32, f32, usize, f32);

struct Info {
    rel: String,
    jpg: PathBuf,
    camera: String,
    scene: Scene,
    date: String,
    t_ms: i64,
    w: u32,
    h: u32,
    train: Vec<Bx>,
    ignore: Vec<Bx>,
    /// `collected` or `import:<id>`.
    origin: String,
}

impl Info {
    fn imported(&self) -> bool {
        self.origin != "collected"
    }
}

struct Decoded {
    hash: u64,
    thumb: Thumb,
}

#[derive(Clone)]
enum Kind {
    Different {
        share: f64,
        class: String,
        shifted: bool,
    },
    Duplicate,
}

struct Pair {
    /// Reference (earlier), new (later).
    i: usize,
    j: usize,
    kind: Kind,
    hamming: u32,
    verdict: Verdict,
    /// [`TEXTURE_VARIANT`] at otherwise default parameters calls the pair different.
    textured: bool,
    /// [`Params::COLLECTOR`] (the collector's block test) calls the pair different.
    collector: bool,
    /// Sweep point `k` calls the pair different.
    sweep: Vec<bool>,
}

fn scene_of(s: &str) -> Option<Scene> {
    match s {
        "day" => Some(Scene::Day),
        "night" => Some(Scene::Night),
        "ir" => Some(Scene::Ir),
        _ => None,
    }
}

fn bucket_of(share: f64) -> &'static str {
    BUCKETS
        .iter()
        .find(|(_, hi)| share < *hi)
        .map_or(">0.5%", |(n, _)| n)
}

fn sweep_points() -> Vec<Point> {
    let mut v = Vec::new();
    for &b in &TEXTURE_BETAS {
        for &z in &Z_CELLS {
            for &d in &D_MINS {
                for &m in &MIN_COMPONENTS {
                    v.push((z, d, m, b));
                }
            }
        }
    }
    v
}

fn sweep_params((z, d, m, b): Point) -> Params {
    Params {
        z_cell: z,
        d_min_dark: d,
        d_min_day: d,
        min_component: m,
        texture_beta: b,
        ..Params::default()
    }
}

fn sweep_name((z, d, m, b): Point) -> String {
    format!("z{z}_d{d}_c{m}_t{b}")
}

/// Why a frame was left out, or the frame.
fn load(dir: &Path, rel: &str) -> Result<Info, &'static str> {
    let frame = Frame::resolve(dir, rel).ok_or("unresolved")?;
    let lf: LabelFile = labels::read(&frame.labels)
        .map_err(|_| "label file unreadable")?
        .ok_or("no label file")?;
    if lf.status != Status::Reviewed || lf.reviewed_at.is_none() {
        return Err("not reviewed");
    }
    if lf.disagreement.is_some() {
        return Err("reopened");
    }
    if lf.boxes.iter().any(|b| b.state == BoxState::Pending) {
        return Err("pending box");
    }
    let rec = frame.record().map_err(|_| "sidecar unreadable")?;
    let t_ms = chrono::DateTime::parse_from_rfc3339(&rec.timestamp)
        .map_err(|_| "bad timestamp")?
        .timestamp_millis();
    let scene = scene_of(&frame.rel.scene).ok_or("unknown scene")?;
    let lower = |b: &labels::LabelBox| (b.label.to_lowercase(), b.bbox);
    Ok(Info {
        rel: rel.to_string(),
        jpg: frame.jpg.clone(),
        camera: frame.rel.camera.clone(),
        scene,
        date: frame.rel.date.clone(),
        t_ms,
        w: rec.width,
        h: rec.height,
        train: lf
            .training()
            .filter(|b| !b.is_ignore())
            .map(lower)
            .collect(),
        ignore: lf
            .boxes
            .iter()
            .filter(|b| b.is_ignore())
            .map(lower)
            .collect(),
        origin: rec.source.unwrap_or_else(|| "collected".into()),
    })
}

fn decode(info: &Info) -> Result<Decoded, &'static str> {
    let bytes = std::fs::read(&info.jpg).map_err(|_| "image unreadable")?;
    let (luma, w, h) = dhash::luma_of_encoded(&bytes).map_err(|_| "image does not decode")?;
    if (w, h) != (info.w as usize, info.h as usize) {
        return Err("image size differs from the sidecar");
    }
    Ok(Decoded {
        hash: dhash::of_luma(&luma, w, h),
        thumb: Thumb::of_luma(&luma, w, h).ok_or("empty image")?,
    })
}

/// `b` has a box of the same object at IoU >= [`MATCH_IOU`] in `other` (training or ignore).
fn matched(b: &Bx, other: &Info) -> bool {
    other
        .train
        .iter()
        .chain(&other.ignore)
        .any(|o| repeats::same_object(&o.0, &b.0) && labels::iou(&o.1, &b.1) >= MATCH_IOU)
}

/// The largest training box of `a` with no match in `b`: (share of the frame, class, shifted).
fn largest_unmatched(a: &Info, b: &Info) -> Option<(f64, String, bool)> {
    let area = f64::from(a.w) * f64::from(a.h);
    a.train
        .iter()
        .filter(|x| !matched(x, b))
        .map(|x| {
            let bw = f64::from((x.1[2] - x.1[0]).max(0.0));
            let bh = f64::from((x.1[3] - x.1[1]).max(0.0));
            let shifted = b
                .train
                .iter()
                .any(|o| repeats::same_object(&o.0, &x.0) && labels::iou(&o.1, &x.1) >= SHIFT_IOU);
            (bw * bh / area.max(1.0), x.0.clone(), shifted)
        })
        .max_by(|p, q| p.0.total_cmp(&q.0))
}

fn kind_of(a: &Info, b: &Info) -> Kind {
    let both = [largest_unmatched(a, b), largest_unmatched(b, a)];
    match both
        .into_iter()
        .flatten()
        .max_by(|p, q| p.0.total_cmp(&q.0))
    {
        Some((share, class, shifted)) => Kind::Different {
            share,
            class,
            shifted,
        },
        None => Kind::Duplicate,
    }
}

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A rate with its count and bootstrap interval (resampled clusters).
#[derive(Clone, Copy, Default)]
struct Rate {
    k: u64,
    n: u64,
    lo: f64,
    hi: f64,
}

impl Rate {
    fn rate(&self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.k as f64 / self.n as f64
        }
    }

    fn json(&self) -> Value {
        json!({"k": self.k, "n": self.n, "rate": nan_null(self.rate()),
            "ci95": [nan_null(self.lo), nan_null(self.hi)]})
    }

    fn md(&self) -> String {
        if self.n == 0 {
            return "-".into();
        }
        format!(
            "{:.1}% [{:.1}, {:.1}] ({}/{})",
            100.0 * self.rate(),
            100.0 * self.lo,
            100.0 * self.hi,
            self.k,
            self.n
        )
    }
}

fn nan_null(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

/// The rate of `hit` over `items`, clustered by `cluster` for the bootstrap.
fn rate_of<'a>(
    items: impl Iterator<Item = &'a Pair>,
    cluster: impl Fn(&Pair) -> String,
    hit: impl Fn(&Pair) -> bool,
    seed: u64,
) -> Rate {
    let mut by: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for p in items {
        let e = by.entry(cluster(p)).or_default();
        e.1 += 1;
        if hit(p) {
            e.0 += 1;
        }
    }
    let cl: Vec<(u64, u64)> = by.into_values().collect();
    let (k, n) = cl.iter().fold((0, 0), |a, c| (a.0 + c.0, a.1 + c.1));
    let mut r = Rate {
        k,
        n,
        lo: f64::NAN,
        hi: f64::NAN,
    };
    if cl.is_empty() {
        return r;
    }
    let mut rng = Rng(seed);
    let m = cl.len() as u64;
    let mut rates: Vec<f64> = (0..BOOT)
        .filter_map(|_| {
            let (mut bk, mut bn) = (0u64, 0u64);
            for _ in 0..m {
                let c = cl[(rng.next() % m) as usize];
                bk += c.0;
                bn += c.1;
            }
            (bn > 0).then(|| bk as f64 / bn as f64)
        })
        .collect();
    rates.sort_by(f64::total_cmp);
    if !rates.is_empty() {
        let at =
            |q: f64| rates[((q * (rates.len() - 1) as f64).round() as usize).min(rates.len() - 1)];
        r.lo = at(0.025);
        r.hi = at(0.975);
    }
    r
}

fn quantiles(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let at = |q: f64| v[((q * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
    (at(0.5), at(0.95))
}

/// One point of the sweep, over all pairs.
struct SweepRow {
    /// Its index in [`Pair::sweep`].
    b: usize,
    pt: Point,
    missed: Rate,
    kept: Rate,
    by_subset: Value,
}

/// One way of calling a pair a duplicate.
struct Method {
    name: String,
    dup: Box<dyn Fn(&Pair) -> bool + Sync>,
}

/// The pairs a table covers.
#[derive(Clone, Copy)]
enum Subset {
    All,
    Collected,
    Imported,
}

impl Subset {
    const EVERY: [Subset; 3] = [Subset::All, Subset::Collected, Subset::Imported];

    fn name(self) -> &'static str {
        match self {
            Subset::All => "all",
            Subset::Collected => "collected",
            Subset::Imported => "imported",
        }
    }

    fn has(self, imported: bool) -> bool {
        match self {
            Subset::All => true,
            Subset::Collected => !imported,
            Subset::Imported => imported,
        }
    }
}

#[derive(Default)]
struct ReplayCount {
    frames: u64,
    kept: u64,
    dropped: u64,
    dropped_with_box: u64,
    dropped_with_unmatched: u64,
}

impl ReplayCount {
    fn add(&mut self, kept: bool, info: &Info, reference: Option<&Info>) {
        self.frames += 1;
        if kept {
            self.kept += 1;
            return;
        }
        self.dropped += 1;
        if !info.train.is_empty() {
            self.dropped_with_box += 1;
        }
        if reference.is_some_and(|r| largest_unmatched(info, r).is_some()) {
            self.dropped_with_unmatched += 1;
        }
    }

    fn json(&self) -> Value {
        json!({"frames": self.frames, "kept": self.kept, "dropped": self.dropped,
            "dropped_with_training_box": self.dropped_with_box,
            "dropped_with_object_missing_in_its_reference": self.dropped_with_unmatched})
    }
}

/// `BOP_DEDUP_EVAL_DUMP`: the thumbnails (`thumbs.bin`, 9,216 bytes per frame in `frames.json`
/// order), the frames and the pairs, for trying variants of the test outside Rust.
fn dump(out: &Path, infos: &[Info], decs: &[Decoded], pairs: &[Pair]) {
    let mut bin = Vec::new();
    let frames: Vec<Value> = infos
        .iter()
        .zip(decs)
        .map(|(f, d)| {
            bin.extend_from_slice(d.thumb.cells());
            json!({"rel": f.rel, "jpg": f.jpg.display().to_string(), "camera": f.camera,
                "scene": f.scene.as_str(), "date": f.date, "t_ms": f.t_ms, "w": f.w, "h": f.h,
                "cols": d.thumb.cols(), "rows": d.thumb.rows(), "dhash": dhash::hex(d.hash),
                "train": f.train, "ignore": f.ignore, "imported": f.imported(),
                "origin": f.origin})
        })
        .collect();
    let pairs: Vec<Value> = pairs
        .iter()
        .map(|p| {
            let (kind, share, class, shifted) = match &p.kind {
                Kind::Different {
                    share,
                    class,
                    shifted,
                } => ("different", *share, class.as_str(), *shifted),
                Kind::Duplicate => ("duplicate", 0.0, "", false),
            };
            json!({"i": p.i, "j": p.j, "kind": kind, "share": share, "class": class,
                "shifted": shifted, "hamming": p.hamming})
        })
        .collect();
    std::fs::write(out.join("thumbs.bin"), &bin).expect("write thumbs.bin");
    std::fs::write(
        out.join("frames.json"),
        serde_json::to_vec(&frames).expect("serialize"),
    )
    .expect("write frames.json");
    std::fs::write(
        out.join("pairs.json"),
        serde_json::to_vec(&pairs).expect("serialize"),
    )
    .expect("write pairs.json");
}

fn join_names(methods: &[Method]) -> String {
    methods
        .iter()
        .map(|m| m.name.as_str())
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
#[ignore = "an offline measurement on a real review folder: set BOP_DEDUP_EVAL_DIR"]
fn dedup_eval() {
    let Some(dir) = std::env::var_os("BOP_DEDUP_EVAL_DIR").map(PathBuf::from) else {
        eprintln!("dedup_eval: BOP_DEDUP_EVAL_DIR is not set, nothing to do");
        return;
    };
    let out = std::env::var_os("BOP_DEDUP_EVAL_OUT").map_or_else(
        || std::env::temp_dir().join("bop-dedup-eval"),
        PathBuf::from,
    );
    std::fs::create_dir_all(&out).expect("create the output folder");
    let max_frames: usize = std::env::var("BOP_DEDUP_EVAL_MAX_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(40_000);
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .saturating_sub(1)
        .max(1);
    let started = Instant::now();

    // Frames.
    let scan = labels::scan_labels(&dir);
    let mut rels: Vec<String> = scan.files.into_iter().map(|(rel, _)| rel).collect();
    let scanned = rels.len();
    let mut sampled = false;
    if rels.len() > max_frames {
        let day = |r: &str| r.rsplit_once('/').map_or(r, |(d, _)| d).to_string();
        let mut days: BTreeMap<String, usize> = BTreeMap::new();
        for r in &rels {
            *days.entry(day(r)).or_default() += 1;
        }
        let mut order: Vec<(u64, String, usize)> =
            days.into_iter().map(|(d, n)| (fnv(&d), d, n)).collect();
        order.sort();
        let (mut keep, mut total) = (std::collections::HashSet::new(), 0usize);
        for (_, d, n) in order {
            if total >= max_frames {
                break;
            }
            total += n;
            keep.insert(d);
        }
        rels.retain(|r| keep.contains(&day(r)));
        sampled = true;
    }
    let loaded = parallel::map_all(rels.len(), threads, &|i| load(&dir, &rels[i]));
    let mut excluded: BTreeMap<&str, u64> = BTreeMap::new();
    let mut infos: Vec<Info> = Vec::new();
    for r in loaded {
        match r {
            Ok(i) => infos.push(i),
            Err(why) => *excluded.entry(why).or_default() += 1,
        }
    }
    let t_load = started.elapsed();
    let decoded = parallel::map_all(infos.len(), threads, &|i| decode(&infos[i]));
    let mut frames: Vec<(Info, Decoded)> = Vec::new();
    for (info, d) in infos.into_iter().zip(decoded) {
        match d {
            Ok(d) => frames.push((info, d)),
            Err(why) => *excluded.entry(why).or_default() += 1,
        }
    }
    frames.sort_by(|a, b| {
        (&a.0.origin, &a.0.camera, a.0.t_ms, &a.0.rel).cmp(&(
            &b.0.origin,
            &b.0.camera,
            b.0.t_ms,
            &b.0.rel,
        ))
    });
    let (infos, decs): (Vec<Info>, Vec<Decoded>) = frames.into_iter().unzip();
    let t_decode = started.elapsed();
    eprintln!(
        "dedup_eval: {} label files, {} frames used, read {:.0?}, decoded {:.0?}",
        scanned,
        infos.len(),
        t_load,
        t_decode - t_load
    );

    // Pairs.
    let mut groups: BTreeMap<(&str, &str, &str), Vec<usize>> = BTreeMap::new();
    for (i, f) in infos.iter().enumerate() {
        groups
            .entry((f.origin.as_str(), f.camera.as_str(), f.scene.as_str()))
            .or_default()
            .push(i);
    }
    let mut idx_pairs: Vec<(usize, usize)> = Vec::new();
    for g in groups.values() {
        for (b, &j) in g.iter().enumerate() {
            for &i in g[..b].iter().rev() {
                if infos[j].t_ms - infos[i].t_ms > WINDOW_MS {
                    break;
                }
                if (infos[i].w, infos[i].h) == (infos[j].w, infos[j].h) {
                    idx_pairs.push((i, j));
                }
            }
        }
    }
    let points = sweep_points();
    let textured = Params {
        texture_beta: TEXTURE_VARIANT,
        ..Params::default()
    };
    let pairs: Vec<Pair> = parallel::map_all(idx_pairs.len(), threads, &|k| {
        let (i, j) = idx_pairs[k];
        let scene = infos[j].scene;
        let res = Residual::of(&decs[j].thumb, &decs[i].thumb);
        let call = |p: &Params| {
            res.as_ref()
                .map_or(Verdict::MISMATCH, |r| r.verdict(scene, None, p))
        };
        Pair {
            i,
            j,
            kind: kind_of(&infos[i], &infos[j]),
            hamming: dhash::hamming(decs[i].hash, decs[j].hash),
            verdict: call(&Params::default()),
            textured: call(&textured).different,
            collector: call(&Params::COLLECTOR).different,
            sweep: points
                .iter()
                .map(|&pt| call(&sweep_params(pt)).different)
                .collect(),
        }
    });
    let t_pairs = started.elapsed();
    eprintln!(
        "dedup_eval: {} pairs compared in {:.0?}",
        pairs.len(),
        t_pairs - t_decode
    );
    if std::env::var_os("BOP_DEDUP_EVAL_DUMP").is_some() {
        dump(&out, &infos, &decs, &pairs);
    }

    let scene_of_pair = |p: &Pair| infos[p.j].scene;
    let camday = |p: &Pair| {
        format!(
            "{}/{}/{}",
            infos[p.j].origin, infos[p.j].camera, infos[p.j].date
        )
    };
    let is_diff = |p: &Pair| matches!(p.kind, Kind::Different { .. });
    let imported_pair = |p: &Pair| infos[p.j].imported();

    // Sweep.
    let mut sweep_rows = Vec::new();
    for (b, &pt) in points.iter().enumerate() {
        let dup = |p: &Pair| !p.sweep[b];
        let missed = rate_of(pairs.iter().filter(|p| is_diff(p)), camday, dup, 1);
        let kept = rate_of(pairs.iter().filter(|p| !is_diff(p)), camday, |p| !dup(p), 2);
        let mut by = serde_json::Map::new();
        for sub in Subset::EVERY {
            let mut per_scene = serde_json::Map::new();
            for s in [Scene::Day, Scene::Ir, Scene::Night] {
                let inside = |p: &&Pair| sub.has(imported_pair(p)) && scene_of_pair(p) == s;
                let (mk, mn) = pairs
                    .iter()
                    .filter(|p| inside(p) && is_diff(p))
                    .fold((0u64, 0u64), |a, p| (a.0 + u64::from(dup(p)), a.1 + 1));
                let (dk, dn) = pairs
                    .iter()
                    .filter(|p| inside(p) && !is_diff(p))
                    .fold((0u64, 0u64), |a, p| (a.0 + u64::from(!dup(p)), a.1 + 1));
                per_scene.insert(
                    s.as_str().into(),
                    json!({"missed": [mk, mn], "kept": [dk, dn]}),
                );
            }
            by.insert(sub.name().into(), Value::Object(per_scene));
        }
        sweep_rows.push(SweepRow {
            b,
            pt,
            missed,
            kept,
            by_subset: Value::Object(by),
        });
    }
    let score = |m: &Rate, k: &Rate| {
        let (m, k) = (m.rate(), k.rate());
        MISS_WEIGHT * if m.is_finite() { m } else { 0.0 } + if k.is_finite() { k } else { 0.0 }
    };
    let best = sweep_rows
        .iter()
        .min_by(|a, b| score(&a.missed, &a.kept).total_cmp(&score(&b.missed, &b.kept)))
        .map(|r| (r.b, r.pt));

    // Methods.
    let mut methods: Vec<Method> = Vec::new();
    for t in [DHASH_TODAY, 4, 2] {
        methods.push(Method {
            name: format!("dhash<={t}"),
            dup: Box::new(move |p: &Pair| p.hamming <= t),
        });
    }
    methods.push(Method {
        name: "blockdiff".into(),
        dup: Box::new(|p: &Pair| !p.verdict.different),
    });
    methods.push(Method {
        name: format!("blockdiff texture {TEXTURE_VARIANT}"),
        dup: Box::new(|p: &Pair| !p.textured),
    });
    methods.push(Method {
        name: format!("dhash<={DHASH_TODAY} confirmed by blockdiff"),
        dup: Box::new(|p: &Pair| p.hamming <= DHASH_TODAY && !p.verdict.different),
    });
    methods.push(Method {
        name: "blockdiff COLLECTOR".into(),
        dup: Box::new(|p: &Pair| !p.collector),
    });
    methods.push(Method {
        name: COLLECTOR_BOTH.into(),
        dup: Box::new(|p: &Pair| p.hamming <= DHASH_TODAY && !p.collector),
    });
    if let Some((b, pt)) = best {
        methods.push(Method {
            name: format!("best sweep {}", sweep_name(pt)),
            dup: Box::new(move |p: &Pair| !p.sweep[b]),
        });
        methods.push(Method {
            name: format!("dhash<={DHASH_TODAY} confirmed by best sweep"),
            dup: Box::new(move |p: &Pair| p.hamming <= DHASH_TODAY && !p.sweep[b]),
        });
    }
    let best_params = best.map_or_else(Params::default, |(_, pt)| sweep_params(pt));

    // Strata.
    let scenes = [Scene::Day, Scene::Night, Scene::Ir];
    let classes: Vec<String> = {
        let mut c: Vec<String> = pairs
            .iter()
            .filter_map(|p| match &p.kind {
                Kind::Different { class, .. } => Some(class.clone()),
                Kind::Duplicate => None,
            })
            .collect();
        c.sort();
        c.dedup();
        c
    };
    let cameras: Vec<String> = {
        let mut c: Vec<String> = infos.iter().map(|f| f.camera.clone()).collect();
        c.sort();
        c.dedup();
        c
    };
    let mut subsets_json = serde_json::Map::new();
    // (subset, method, scene, column, rate)
    let mut headline: Vec<(&str, String, String, String, Rate)> = Vec::new();
    for sub in Subset::EVERY {
        let mut method_json = serde_json::Map::new();
        for m in &methods {
            let dup = &m.dup;
            let missed_where = |f: &dyn Fn(&Pair) -> bool| {
                rate_of(
                    pairs
                        .iter()
                        .filter(|p| sub.has(imported_pair(p)) && is_diff(p) && f(p)),
                    camday,
                    |p| dup(p),
                    3,
                )
            };
            let kept_where = |f: &dyn Fn(&Pair) -> bool| {
                rate_of(
                    pairs
                        .iter()
                        .filter(|p| sub.has(imported_pair(p)) && !is_diff(p) && f(p)),
                    camday,
                    |p| !dup(p),
                    4,
                )
            };
            let mut push = |scene: &str, col: &str, r: Rate| {
                headline.push((sub.name(), m.name.clone(), scene.into(), col.into(), r));
            };
            let mut by_scene = serde_json::Map::new();
            for &s in &scenes {
                let mut buckets = serde_json::Map::new();
                for (bn, _) in BUCKETS {
                    let r = missed_where(&|p: &Pair| {
                        scene_of_pair(p) == s
                            && matches!(&p.kind, Kind::Different { share, .. } if bucket_of(*share) == bn)
                    });
                    push(s.as_str(), bn, r);
                    buckets.insert(bn.into(), r.json());
                }
                let missed = missed_where(&|p: &Pair| scene_of_pair(p) == s);
                let missed_new = missed_where(&|p: &Pair| {
                    scene_of_pair(p) == s
                        && matches!(&p.kind, Kind::Different { shifted: false, .. })
                });
                let kept = kept_where(&|p: &Pair| scene_of_pair(p) == s);
                push(s.as_str(), "all", missed);
                push(s.as_str(), "new", missed_new);
                push(s.as_str(), "kept", kept);
                by_scene.insert(
                    s.as_str().into(),
                    json!({"missed": missed.json(), "missed_new_objects": missed_new.json(),
                        "missed_by_size": buckets, "kept_repeats": kept.json()}),
                );
            }
            let mut by_class = serde_json::Map::new();
            for c in &classes {
                let r = missed_where(
                    &|p: &Pair| matches!(&p.kind, Kind::Different { class, .. } if class == c),
                );
                if r.n > 0 {
                    push("all", &format!("class {c}"), r);
                }
                by_class.insert(c.clone(), r.json());
            }
            let mut by_camera = serde_json::Map::new();
            for c in &cameras {
                let missed = missed_where(&|p: &Pair| &infos[p.j].camera == c);
                let kept = kept_where(&|p: &Pair| &infos[p.j].camera == c);
                push("all", &format!("camera {c} missed"), missed);
                push("all", &format!("camera {c} kept"), kept);
                by_camera.insert(
                    c.clone(),
                    json!({"missed": missed.json(), "kept_repeats": kept.json()}),
                );
            }
            let all_missed = missed_where(&|_| true);
            let all_new =
                missed_where(&|p: &Pair| matches!(&p.kind, Kind::Different { shifted: false, .. }));
            let all_kept = kept_where(&|_| true);
            push("all", "all", all_missed);
            push("all", "new", all_new);
            push("all", "kept", all_kept);
            method_json.insert(
                m.name.clone(),
                json!({"missed": all_missed.json(), "missed_new_objects": all_new.json(),
                    "kept_repeats": all_kept.json(), "by_scene": by_scene, "by_class": by_class,
                    "by_camera": by_camera}),
            );
        }
        subsets_json.insert(sub.name().into(), Value::Object(method_json));
    }

    // Replay per origin and camera in time order (infos are sorted that way).
    let mut replay: BTreeMap<String, BTreeMap<String, ReplayCount>> = BTreeMap::new();
    let mut count = |f: &Info, method: &str, kept: bool, reference: Option<&Info>| {
        let kind = if f.imported() {
            "imported"
        } else {
            "collected"
        };
        for scope in [
            format!("{kind}/{}", f.scene.as_str()),
            format!("{kind}/all"),
            "all".to_string(),
        ] {
            replay
                .entry(scope)
                .or_default()
                .entry(method.to_string())
                .or_default()
                .add(kept, f, reference);
        }
    };
    let textured_name = format!("blockdiff texture {TEXTURE_VARIANT}");
    let veto_name = format!("dhash<={DHASH_TODAY} confirmed by blockdiff");
    let best_veto_name = format!("dhash<={DHASH_TODAY} confirmed by best sweep");
    let best_name = best.map_or_else(String::new, |(_, pt)| {
        format!("best sweep {}", sweep_name(pt))
    });
    let mut start = 0;
    while start < infos.len() {
        let key = (&infos[start].origin, &infos[start].camera);
        let end = infos[start..]
            .iter()
            .position(|f| (&f.origin, &f.camera) != key)
            .map_or(infos.len(), |e| start + e);
        let mut dh: VecDeque<usize> = VecDeque::new();
        let mut veto: VecDeque<usize> = VecDeque::new();
        let mut veto_best: VecDeque<usize> = VecDeque::new();
        let mut veto_collector: VecDeque<usize> = VecDeque::new();
        let mut bd: HashMap<Scene, VecDeque<usize>> = HashMap::new();
        let mut bt: HashMap<Scene, VecDeque<usize>> = HashMap::new();
        let mut bb: HashMap<Scene, VecDeque<usize>> = HashMap::new();
        let mut bs: HashMap<Scene, VecDeque<usize>> = HashMap::new();
        let mut maps: HashMap<Scene, NoiseMap> = HashMap::new();
        for (j, f) in infos.iter().enumerate().take(end).skip(start) {
            let fresh = |refs: &mut VecDeque<usize>| {
                refs.retain(|&r| f.t_ms - infos[r].t_ms <= WINDOW_MS);
            };
            let push = |refs: &mut VecDeque<usize>| {
                refs.push_back(j);
                while refs.len() > KEEP {
                    refs.pop_front();
                }
            };
            let block_dup = |refs: &VecDeque<usize>, p: &Params| {
                blockdiff::duplicate_of(
                    &decs[j].thumb,
                    refs.iter().rev().map(|&r| &decs[r].thumb),
                    f.scene,
                    None,
                    p,
                )
                .map(|k| refs[refs.len() - 1 - k])
            };
            // (a) dHash, references per camera.
            fresh(&mut dh);
            let m = dh
                .iter()
                .copied()
                .find(|&r| dhash::hamming(decs[r].hash, decs[j].hash) <= DHASH_TODAY);
            count(f, "dhash<=6", m.is_none(), m.map(|r| &infos[r]));
            if m.is_none() {
                push(&mut dh);
            }
            // dHash, a drop confirmed by blockdiff on the same reference.
            for (refs, params, name) in [
                (&mut veto, Params::default(), veto_name.as_str()),
                (&mut veto_best, best_params, best_veto_name.as_str()),
                (&mut veto_collector, Params::COLLECTOR, COLLECTOR_BOTH),
            ] {
                fresh(refs);
                let m = refs.iter().rev().copied().find(|&r| {
                    dhash::hamming(decs[r].hash, decs[j].hash) <= DHASH_TODAY
                        && !blockdiff::verdict_with(
                            &decs[j].thumb,
                            &decs[r].thumb,
                            f.scene,
                            None,
                            &params,
                        )
                        .different
                });
                count(f, name, m.is_none(), m.map(|r| &infos[r]));
                if m.is_none() {
                    push(refs);
                }
            }
            // (c) blockdiff, references per camera and scene, newest first.
            for (refs, params, name) in [
                (
                    bd.entry(f.scene).or_default(),
                    Params::default(),
                    "blockdiff",
                ),
                (
                    bt.entry(f.scene).or_default(),
                    textured,
                    textured_name.as_str(),
                ),
                (
                    bb.entry(f.scene).or_default(),
                    best_params,
                    best_name.as_str(),
                ),
            ] {
                fresh(refs);
                let m = block_dup(refs, &params);
                count(f, name, m.is_none(), m.map(|r| &infos[r]));
                if m.is_none() {
                    push(refs);
                }
            }
            // (c') blockdiff with the camera's noise map, fed by the best-matching reference.
            let refs = bs.entry(f.scene).or_default();
            fresh(refs);
            let map = maps
                .entry(f.scene)
                .or_insert_with(|| NoiseMap::for_thumb(&decs[j].thumb));
            let mut matched_ref = None;
            let mut best_ref: Option<(usize, Residual)> = None;
            for &r in refs.iter().rev() {
                let Some(res) = Residual::of(&decs[j].thumb, &decs[r].thumb) else {
                    continue;
                };
                let v = res.verdict(f.scene, Some(&*map), &Params::default());
                if !v.different {
                    matched_ref = Some(r);
                    best_ref = Some((0, res));
                    break;
                }
                if best_ref.as_ref().is_none_or(|(c, _)| v.changed_cells < *c) {
                    best_ref = Some((v.changed_cells, res));
                }
            }
            if let Some((_, res)) = &best_ref {
                blockdiff::update_noise(map, res);
            }
            count(
                f,
                "blockdiff+sigma_c",
                matched_ref.is_none(),
                matched_ref.map(|r| &infos[r]),
            );
            if matched_ref.is_none() {
                push(refs);
            }
        }
        start = end;
    }

    // Timings, one thread.
    let step = (infos.len() / TIME_FRAMES).max(1);
    let (mut t_luma, mut t_thumb, mut t_hash) = (Vec::new(), Vec::new(), Vec::new());
    for f in infos.iter().step_by(step).take(TIME_FRAMES) {
        let Ok(bytes) = std::fs::read(&f.jpg) else {
            continue;
        };
        let t0 = Instant::now();
        let Ok((luma, w, h)) = dhash::luma_of_encoded(&bytes) else {
            continue;
        };
        let t1 = Instant::now();
        let thumb = Thumb::of_luma(&luma, w, h);
        let t2 = Instant::now();
        let hash = dhash::of_luma(&luma, w, h);
        let t3 = Instant::now();
        std::hint::black_box((thumb, hash));
        t_luma.push((t1 - t0).as_secs_f64() * 1e3);
        t_thumb.push((t2 - t1).as_secs_f64() * 1e3);
        t_hash.push((t3 - t2).as_secs_f64() * 1e3);
    }
    let mut t_cmp = Vec::new();
    let pstep = (pairs.len() / TIME_PAIRS).max(1);
    for p in pairs.iter().step_by(pstep).take(TIME_PAIRS) {
        let t0 = Instant::now();
        let v = blockdiff::verdict(&decs[p.j].thumb, &decs[p.i].thumb, infos[p.j].scene, None);
        t_cmp.push(t0.elapsed().as_secs_f64() * 1e6);
        std::hint::black_box(v);
    }
    let (luma_med, luma_p95) = quantiles(t_luma);
    let (thumb_med, thumb_p95) = quantiles(t_thumb);
    let (hash_med, hash_p95) = quantiles(t_hash);
    let (cmp_med, cmp_p95) = quantiles(t_cmp);

    // Contact sheets.
    let share_of = |p: &Pair| match &p.kind {
        Kind::Different { share, .. } => *share,
        Kind::Duplicate => 0.0,
    };
    let line = |p: &Pair| {
        let what = match &p.kind {
            Kind::Different {
                share,
                class,
                shifted,
            } => format!(
                "{class} {:.3}%{}",
                100.0 * share,
                if *shifted { " shifted" } else { "" }
            ),
            Kind::Duplicate => "duplicate".into(),
        };
        format!(
            "{}\t{}\t{}\t{what}; hamming {}; changed {}, largest {}, max_z {:.1}\t{}\t{}\n",
            infos[p.j].origin,
            infos[p.j].scene,
            infos[p.j].camera,
            p.hamming,
            p.verdict.changed_cells,
            p.verdict.largest_blob,
            p.verdict.max_z,
            infos[p.i].jpg.display(),
            infos[p.j].jpg.display()
        )
    };
    let header = "# origin\tscene\tcamera\twhat\treference (earlier)\tnew (later)\n";
    let section = |title: &str, dup: &dyn Fn(&Pair) -> bool| -> String {
        let mut v: Vec<&Pair> = pairs.iter().filter(|p| is_diff(p) && dup(p)).collect();
        v.sort_by(|a, b| share_of(b).total_cmp(&share_of(a)));
        let mut s = format!(
            "## {title}: {} missed, the largest objects first\n",
            v.len()
        );
        s.push_str(header);
        for p in v.iter().take(SHEET) {
            s.push_str(&line(p));
        }
        s
    };
    let misses = [
        section("blockdiff (defaults)", &|p| !p.verdict.different),
        section(&format!("blockdiff texture {TEXTURE_VARIANT}"), &|p| {
            !p.textured
        }),
        section(&best_name, &|p| best.is_some_and(|(b, _)| !p.sweep[b])),
        section(&format!("dhash<={DHASH_TODAY}"), &|p| {
            p.hamming <= DHASH_TODAY
        }),
    ]
    .join("\n");
    std::fs::write(out.join("misses.txt"), misses).expect("write misses.txt");
    let mut kept: Vec<&Pair> = pairs
        .iter()
        .filter(|p| !is_diff(p) && p.verdict.different)
        .collect();
    kept.sort_by(|a, b| {
        (b.verdict.largest_blob, b.verdict.max_z)
            .partial_cmp(&(a.verdict.largest_blob, a.verdict.max_z))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let repeats_txt: String = std::iter::once(format!(
        "## blockdiff (defaults): {} kept repeats, the largest groups first\n{header}",
        kept.len()
    ))
    .chain(kept.iter().take(SHEET).map(|p| line(p)))
    .collect();
    std::fs::write(out.join("repeats.txt"), repeats_txt).expect("write repeats.txt");

    // Dataset counts.
    let mut frames_by: BTreeMap<String, u64> = BTreeMap::new();
    for f in &infos {
        let kind = if f.imported() {
            "imported"
        } else {
            "collected"
        };
        *frames_by
            .entry(format!("{kind}/{}", f.scene.as_str()))
            .or_default() += 1;
    }
    let mut pair_counts: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for p in &pairs {
        let key = match &p.kind {
            Kind::Different { share, .. } => format!("different {}", bucket_of(*share)),
            Kind::Duplicate => "duplicate".into(),
        };
        let kind = if imported_pair(p) {
            "imported"
        } else {
            "collected"
        };
        *pair_counts
            .entry(format!("{kind}/{}", scene_of_pair(p).as_str()))
            .or_default()
            .entry(key)
            .or_default() += 1;
    }
    let imported = infos.iter().filter(|f| f.imported()).count();
    let camera_days = {
        let mut d: Vec<String> = infos
            .iter()
            .map(|f| format!("{}/{}/{}", f.origin, f.camera, f.date))
            .collect();
        d.sort();
        d.dedup();
        d.len()
    };

    let sweep_json: Vec<Value> = sweep_rows
        .iter()
        .map(|r| {
            json!({"point": sweep_name(r.pt), "z_cell": r.pt.0, "d_min": r.pt.1,
                "min_component": r.pt.2, "texture_beta": r.pt.3, "missed": r.missed.json(),
                "kept_repeats": r.kept.json(), "by_subset": r.by_subset,
                "best": best.is_some_and(|x| x.0 == r.b)})
        })
        .collect();
    let mut replay_json = serde_json::Map::new();
    for (scope, ms) in &replay {
        let mut o = serde_json::Map::new();
        for (m, c) in ms {
            o.insert(m.clone(), c.json());
        }
        replay_json.insert(scope.clone(), Value::Object(o));
    }
    let report = json!({
        "dataset": {"dir": dir.display().to_string(), "label_files": scanned, "sampled": sampled,
            "max_frames": max_frames, "frames_used": infos.len(), "frames_by": frames_by,
            "imported_frames": imported, "camera_days": camera_days, "cameras": cameras,
            "excluded": excluded},
        "pairs": {"window_min": WINDOW_MS / 60_000, "total": pairs.len(), "by": pair_counts},
        "methods": subsets_json,
        "sweep": sweep_json,
        "best_rule": format!("min {MISS_WEIGHT} * missed + kept over all pairs"),
        "replay": replay_json,
        "timings": {"threads_for_bulk": threads,
            "decode_luma_ms": {"median": luma_med, "p95": luma_p95},
            "thumb_from_luma_ms": {"median": thumb_med, "p95": thumb_p95},
            "dhash_from_luma_ms": {"median": hash_med, "p95": hash_p95},
            "compare_per_reference_us": {"median": cmp_med, "p95": cmp_p95},
            "total_s": started.elapsed().as_secs_f64()},
    });
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&report).expect("serialize"),
    )
    .expect("write report.json");

    // Markdown.
    let get = |sub: &str, m: &str, s: &str, col: &str| {
        headline
            .iter()
            .find(|h| h.0 == sub && h.1 == m && h.2 == s && h.3 == col)
            .map_or_else(|| "-".to_string(), |h| h.4.md())
    };
    let mut md = String::new();
    let _ = writeln!(md, "# Near-duplicate filter evaluation\n");
    let _ = writeln!(
        md,
        "Folder `{}`: {} label files, {} frames used ({} imported), {} camera days{}. Excluded: {:?}.\n",
        dir.display(),
        scanned,
        infos.len(),
        imported,
        camera_days,
        if sampled {
            ", sampled by camera day"
        } else {
            ""
        },
        excluded
    );
    let _ = writeln!(md, "Frames by origin and scene: {frames_by:?}.\n");
    let _ = writeln!(md, "## Pairs (same origin, camera and scene, <= 10 min)\n");
    let _ = writeln!(
        md,
        "| origin/scene | duplicate | different <0.05% | 0.05-0.5% | >0.5% |"
    );
    let _ = writeln!(md, "|---|---|---|---|---|");
    for (s, c) in &pair_counts {
        let g = |k: &str| c.get(k).copied().unwrap_or(0);
        let _ = writeln!(
            md,
            "| {s} | {} | {} | {} | {} |",
            g("duplicate"),
            g("different <0.05%"),
            g("different 0.05-0.5%"),
            g("different >0.5%")
        );
    }
    for sub in Subset::EVERY {
        let sn = sub.name();
        let _ = writeln!(
            md,
            "\n## {sn} pairs: missed-object rate P(called duplicate | different) and kept-repeat rate P(called different | duplicate)\n"
        );
        let _ = writeln!(
            md,
            "Bootstrap 95% intervals over camera days, counts k/n. \"new\" = different pairs whose unmatched box overlaps no box of its class in the other frame (appeared, not moved).\n"
        );
        let _ = writeln!(
            md,
            "| method | scene | missed <0.05% | missed 0.05-0.5% | missed >0.5% | missed all | missed new | kept repeats |"
        );
        let _ = writeln!(md, "|---|---|---|---|---|---|---|---|");
        for m in &methods {
            for s in ["day", "night", "ir", "all"] {
                let g = |col: &str| get(sn, &m.name, s, col);
                if s == "night" && g("kept") == "-" && g("all") == "-" {
                    continue;
                }
                let _ = writeln!(
                    md,
                    "| {} | {s} | {} | {} | {} | {} | {} | {} |",
                    m.name,
                    g(BUCKETS[0].0),
                    g(BUCKETS[1].0),
                    g(BUCKETS[2].0),
                    g("all"),
                    g("new"),
                    g("kept")
                );
            }
        }
    }
    let _ = writeln!(md, "\n## Missed-object rate by class (all pairs)\n");
    let _ = writeln!(md, "| class | {} |", join_names(&methods));
    let _ = writeln!(md, "|---|{}", "---|".repeat(methods.len()));
    for c in &classes {
        let cells: Vec<String> = methods
            .iter()
            .map(|m| get("all", &m.name, "all", &format!("class {c}")))
            .collect();
        let _ = writeln!(md, "| {c} | {} |", cells.join(" | "));
    }
    let _ = writeln!(md, "\n## By camera, all pairs (missed / kept repeats)\n");
    let _ = writeln!(md, "| camera | {} |", join_names(&methods));
    let _ = writeln!(md, "|---|{}", "---|".repeat(methods.len()));
    for c in &cameras {
        let cells: Vec<String> = methods
            .iter()
            .map(|m| {
                format!(
                    "{} / {}",
                    get("all", &m.name, "all", &format!("camera {c} missed")),
                    get("all", &m.name, "all", &format!("camera {c} kept"))
                )
            })
            .collect();
        let _ = writeln!(md, "| {c} | {} |", cells.join(" | "));
    }
    let _ = writeln!(
        md,
        "\n## Sweep (one d_min for every scene, all pairs; best = min {MISS_WEIGHT} missed + kept; collected and imported: missed k/n, kept k/n)\n"
    );
    let _ = writeln!(
        md,
        "| point | missed | kept repeats | score | collected | imported |"
    );
    let _ = writeln!(md, "|---|---|---|---|---|---|");
    let mut rows: Vec<&SweepRow> = sweep_rows.iter().collect();
    rows.sort_by(|a, b| score(&a.missed, &a.kept).total_cmp(&score(&b.missed, &b.kept)));
    let sub_cell = |r: &SweepRow, sub: &str| {
        let (mut mk, mut mn, mut dk, mut dn) = (0, 0, 0, 0);
        if let Some(o) = r.by_subset.get(sub).and_then(Value::as_object) {
            for v in o.values() {
                let n = |k: &str, i: usize| v[k][i].as_u64().unwrap_or(0);
                mk += n("missed", 0);
                mn += n("missed", 1);
                dk += n("kept", 0);
                dn += n("kept", 1);
            }
        }
        format!("{mk}/{mn}, {dk}/{dn}")
    };
    for r in rows {
        let mark = if best.is_some_and(|x| x.0 == r.b) {
            " (best)"
        } else {
            ""
        };
        let _ = writeln!(
            md,
            "| {}{mark} | {} | {} | {:.3} | {} | {} |",
            sweep_name(r.pt),
            r.missed.md(),
            r.kept.md(),
            score(&r.missed, &r.kept),
            sub_cell(r, "collected"),
            sub_cell(r, "imported")
        );
    }
    let _ = writeln!(
        md,
        "\n## Replay of the keep logic (window 10 min, 32 references = earlier kept frames)\n"
    );
    let _ = writeln!(
        md,
        "| scope | method | frames | kept | dropped | dropped with a training box | dropped with an object its reference lacks |"
    );
    let _ = writeln!(md, "|---|---|---|---|---|---|---|");
    for (scope, ms) in &replay {
        for (m, c) in ms {
            let _ = writeln!(
                md,
                "| {scope} | {m} | {} | {} | {} | {} | {} |",
                c.frames, c.kept, c.dropped, c.dropped_with_box, c.dropped_with_unmatched
            );
        }
    }
    let _ = writeln!(md, "\n## Timings (one thread)\n");
    let _ = writeln!(md, "| step | median | p95 |");
    let _ = writeln!(md, "|---|---|---|");
    let _ = writeln!(md, "| JPEG to luma | {luma_med:.2} ms | {luma_p95:.2} ms |");
    let _ = writeln!(
        md,
        "| thumbnail from luma | {thumb_med:.2} ms | {thumb_p95:.2} ms |"
    );
    let _ = writeln!(
        md,
        "| dHash from luma | {hash_med:.2} ms | {hash_p95:.2} ms |"
    );
    let _ = writeln!(
        md,
        "| blockdiff compare per reference | {cmp_med:.0} us | {cmp_p95:.0} us |"
    );
    let _ = writeln!(
        md,
        "\nLimits: the folder only holds frames the collector kept, so collected pairs already passed today's dHash test unless a bucket skipped it; dHash's missed rate on them is low by construction and its kept-repeat rate high. Imported footage was selected by scene change, dHash and SigLIP instead. Labels are the reviewer's: an object labelled on one frame and not on its neighbour reads as different, and a \"duplicate\" pair (same labels) can differ in rain, insects, light and foliage."
    );
    std::fs::write(out.join("report.md"), &md).expect("write report.md");

    // Summary.
    eprintln!("dedup_eval: {} pairs, wrote {}", pairs.len(), out.display());
    for h in headline
        .iter()
        .filter(|h| h.2 != "all" || matches!(h.3.as_str(), "all" | "new" | "kept"))
        .filter(|h| matches!(h.3.as_str(), "all" | "kept"))
    {
        eprintln!(
            "  {:<9} {:<36} {:<5} {:<5} {}",
            h.0,
            h.1,
            h.2,
            h.3,
            h.4.md()
        );
    }
    eprintln!(
        "  timings: luma {luma_med:.1} ms, thumb {thumb_med:.2} ms, compare {cmp_med:.0} us (median); total {:.0?}",
        started.elapsed()
    );
}
