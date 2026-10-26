//! Disk space of an import: the estimate of what it writes ([`estimate`]), the free-space check
//! before it is queued and before each writing stage ([`check`]), the guard while it runs
//! ([`Guard`]: stops the helpers before the disk fills up) and the errors of a full disk from
//! Rust or from a helper's output ([`is_disk_full`]), all turned into one actionable message.
//!
//! What an import writes:
//! - the work folder (`<training dir>/imports/<id>`): the URL download, the candidate frames
//!   (JPEG quality 92 at most [`CANDIDATE_MAX_SIDE`] px on the long side), then the kept video
//!   frames again at full resolution (removed one by one once copied to the dataset);
//! - the dataset folder: the kept frames at full resolution (a copy of each kept image).
//!
//! JPEG sizes are estimated per pixel, conservatively for camera footage: [`FULL_BYTES_PER_PIXEL`]
//! at full resolution (1080p: about 1 MB a frame) and [`CANDIDATE_BYTES_PER_PIXEL`] for the
//! downscaled candidates (a downscaled frame has more detail per pixel; 640x360: about 185 kB).
//! Each volume keeps [`margin`] free: 5 % of its size, at least 1 GiB, at most 5 GiB.

use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Long side of the candidate frames (px). The analysis needs far less: dHash reads 9x8, the
/// scene stats a 64x64 sample, SigLIP 2 224x224.
pub const CANDIDATE_MAX_SIDE: u32 = 640;
/// Estimated JPEG (quality 92) bytes per pixel of a full-resolution camera frame (0.5, as a
/// fraction: numerator, denominator).
pub const FULL_BYTES_PER_PIXEL: (u64, u64) = (1, 2);
/// Estimated JPEG (quality 92) bytes per pixel of a downscaled candidate frame (0.8).
pub const CANDIDATE_BYTES_PER_PIXEL: (u64, u64) = (4, 5);
/// Source size assumed before a video is read (1080p; a URL is downloaded at most 1080 high).
pub const ASSUMED_SOURCE: (u32, u32) = (1920, 1080);
/// Bytes of a JPEG beyond its pixels (headers, tables).
const JPEG_OVERHEAD: u64 = 2_000;
const GIB: u64 = 1 << 30;
/// Least free space kept on a volume.
pub const MIN_MARGIN: u64 = GIB;
/// Most free space kept on a volume (5 % of a large disk would refuse most imports).
pub const MAX_MARGIN: u64 = 5 * GIB;
/// Share of a volume kept free (within [`MIN_MARGIN`]..[`MAX_MARGIN`]): 1/20 = 5 %.
pub const MARGIN_DIVISOR: u64 = 20;
/// While the import runs it stops below 1/2 of [`margin`] (the estimate may be off, other
/// programs write too; the import never fills the disk).
pub const GUARD_DIVISOR: u64 = 2;

/// `(w, h)` scaled to at most `max_side` on the long side (aspect kept, at least 1 px); unchanged
/// when it fits or `max_side` is 0. The same as `scaled_size` in `import_footage.py`. Pure.
pub fn scaled_size(w: u32, h: u32, max_side: u32) -> (u32, u32) {
    if max_side == 0 || w.max(h) <= max_side {
        return (w, h);
    }
    // round(a * max_side / b), exactly; b > max_side >= the result, so it fits a u32.
    let scale = |a: u32, b: u32| -> u32 {
        let (a, b, m) = (u64::from(a), u64::from(b), u64::from(max_side));
        u32::try_from((a * m + b / 2) / b)
            .unwrap_or(max_side)
            .max(1)
    };
    if w >= h {
        (max_side, scale(h, w))
    } else {
        (scale(w, h), max_side)
    }
}

fn jpeg_bytes(w: u32, h: u32, (num, den): (u64, u64)) -> u64 {
    let px = u64::from(w) * u64::from(h);
    (px * num).div_ceil(den) + JPEG_OVERHEAD
}

/// Estimated size of a full-resolution frame of a `w`x`h` video. Pure.
pub fn full_frame_bytes(w: u32, h: u32) -> u64 {
    jpeg_bytes(w, h, FULL_BYTES_PER_PIXEL)
}

/// Estimated size of a candidate frame of a `w`x`h` video (downscaled to
/// [`CANDIDATE_MAX_SIDE`]; full resolution, so [`full_frame_bytes`], when it is that small). Pure.
pub fn candidate_frame_bytes(w: u32, h: u32) -> u64 {
    let (cw, ch) = scaled_size(w, h, CANDIDATE_MAX_SIDE);
    if (cw, ch) == (w, h) {
        full_frame_bytes(w, h)
    } else {
        jpeg_bytes(cw, ch, CANDIDATE_BYTES_PER_PIXEL)
    }
}

/// What an import is expected to write, per folder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Need {
    /// Into the work folder.
    pub work: u64,
    /// Into the dataset folder.
    pub dataset: u64,
}

/// What [`estimate`] counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// Candidate frames extracted from the videos (the sum of their quotas).
    pub video_candidates: usize,
    /// Images that are candidates (read in place, copied when kept).
    pub image_candidates: usize,
    /// Frames the import keeps.
    pub frames: usize,
    /// The videos' size (`None`: [`ASSUMED_SOURCE`]).
    pub source: Option<(u32, u32)>,
    /// Mean size of the images.
    pub image_bytes: u64,
    /// The URL download still to come.
    pub download: u64,
}

/// The estimate of [`Plan`]: work = download + video candidates x [`candidate_frame_bytes`] +
/// the kept video frames x [`full_frame_bytes`] (extracted again at full resolution);
/// dataset = the kept frames x the larger of a full-resolution video frame and a mean image.
/// Kept frames are at most `frames` and at most the candidates. Pure.
pub fn estimate(p: &Plan) -> Need {
    let (w, h) = p.source.unwrap_or(ASSUMED_SOURCE);
    let full = full_frame_bytes(w, h);
    let n = |x: usize| x as u64;
    let kept_video = p.frames.min(p.video_candidates);
    let kept = p.frames.min(p.video_candidates + p.image_candidates);
    let per_kept = match (p.video_candidates > 0, p.image_candidates > 0) {
        (true, true) => full.max(p.image_bytes),
        (true, false) => full,
        (false, true) => p.image_bytes,
        (false, false) => 0,
    };
    Need {
        work: p
            .download
            .saturating_add(n(p.video_candidates).saturating_mul(candidate_frame_bytes(w, h)))
            .saturating_add(n(kept_video).saturating_mul(full)),
        dataset: n(kept).saturating_mul(per_kept),
    }
}

/// Free and total bytes of a volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Space {
    pub available: u64,
    pub total: u64,
}

impl Space {
    /// The space this import keeps free on the volume.
    pub fn margin(&self) -> u64 {
        margin(self.total)
    }
    /// Below this while running, the guard stops the import.
    pub fn floor(&self) -> u64 {
        self.margin() / GUARD_DIVISOR
    }
}

/// Free space kept on a volume of `total` bytes: 1/[`MARGIN_DIVISOR`] of it within
/// [`MIN_MARGIN`]..[`MAX_MARGIN`]. Pure.
pub fn margin(total: u64) -> u64 {
    (total / MARGIN_DIVISOR).clamp(MIN_MARGIN, MAX_MARGIN)
}

fn existing(path: &Path) -> Option<&Path> {
    path.ancestors()
        .find(|p| !p.as_os_str().is_empty() && p.exists())
}

/// The space of the volume holding `path` (its nearest existing ancestor; `None` when unknown).
pub fn space(path: &Path) -> Option<Space> {
    let s = fs4::statvfs(existing(path)?).ok()?;
    Some(Space {
        available: s.available_space(),
        total: s.total_space(),
    })
}

/// Whether `a` and `b` (or their nearest existing ancestors) are on the same volume.
pub fn same_volume(a: &Path, b: &Path) -> bool {
    let (Some(a), Some(b)) = (existing(a), existing(b)) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (std::fs::metadata(a), std::fs::metadata(b)) {
            (Ok(x), Ok(y)) => x.dev() == y.dev(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        // The drive or UNC share (volumes mounted in folders are rare).
        let root = |p: &Path| {
            std::path::absolute(p)
                .ok()
                .and_then(|p| p.components().next().map(|c| c.as_os_str().to_owned()))
                .map(|c| c.to_string_lossy().to_ascii_lowercase())
        };
        root(a).is_some() && root(a) == root(b)
    }
}

fn size(b: u64) -> String {
    crate::resources::catalog::format_size(b)
}

/// The advice of the messages, by folder.
const WORK_ADVICE: &str = "lower Candidates per video (--max-candidates), sample less often (a \
     larger One frame every / --every, a lower Max frames per second / --max-fps), or set \
     training.dir (the work folder) to a bigger disk";
const DATASET_ADVICE: &str = "keep fewer frames (Frames to keep, --frames) or save them to a \
     bigger disk (Save frames to, --out-dir)";

/// What the import refuses, if anything, for `need` given the volumes' space: `None` when it
/// fits with each volume's [`margin`] left free. `same` = both folders on one volume. Pure.
pub fn refusal(
    need: Need,
    work: (&Path, Option<Space>),
    dataset: (&Path, Option<Space>),
    same: bool,
) -> Option<String> {
    let short = |n: u64, s: Option<Space>| s.filter(|s| n.saturating_add(s.margin()) > s.available);
    let line = |n: u64, dir: &Path, s: Space, advice: &str| {
        format!(
            "the import needs about {} under {}, {} free (it keeps {} free): {advice}",
            size(n),
            dir.display(),
            size(s.available),
            size(s.margin())
        )
    };
    if same {
        let n = need.work.saturating_add(need.dataset);
        let s = short(n, work.1)?;
        return Some(line(
            n,
            work.0,
            s,
            &format!(
                "{}, {}",
                WORK_ADVICE.replacen(", or set", ", set", 1),
                DATASET_ADVICE.replacen("keep", "or keep", 1)
            ),
        ));
    }
    let a = short(need.work, work.1).map(|s| line(need.work, work.0, s, WORK_ADVICE));
    let b =
        short(need.dataset, dataset.1).map(|s| line(need.dataset, dataset.0, s, DATASET_ADVICE));
    match (a, b) {
        (Some(a), Some(b)) => Some(format!("{a}; and {b}")),
        (a, b) => a.or(b),
    }
}

/// Refuse `need` when the volumes of `work` and `dataset` lack the space ([`refusal`]); fine
/// when the space cannot be determined.
pub fn check(need: Need, work: &Path, dataset: &Path) -> Result<()> {
    match refusal(
        need,
        (work, space(work)),
        (dataset, space(dataset)),
        same_volume(work, dataset),
    ) {
        Some(m) => Err(anyhow!(m)),
        None => Ok(()),
    }
}

/// Text in a helper's output that means the disk is full (lowercase).
const FULL_MARKERS: [&str; 6] = [
    "no space left on device",
    "[errno 28]",
    "enospc",
    "not enough space on the disk",
    "disk quota exceeded",
    "there is not enough space",
];

/// Whether `e` (any cause in its chain) is a full disk: an I/O error of kind `StorageFull` /
/// `QuotaExceeded` (ENOSPC, EDQUOT, Windows ERROR_DISK_FULL / ERROR_HANDLE_DISK_FULL), or a
/// helper whose output says so (`import_footage.py` prints "no space left on device").
pub fn is_disk_full(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        if let Some(io) = c.downcast_ref::<std::io::Error>()
            && crate::collect::store::is_disk_full(io)
        {
            return true;
        }
        let has = |t: &str| {
            let t = t.to_ascii_lowercase();
            FULL_MARKERS.iter().any(|m| t.contains(m))
        };
        if let Some(cf) = c.downcast_ref::<crate::resources::export::ChildFailed>()
            && cf.lines.iter().any(|l| has(l))
        {
            return true;
        }
        has(&c.to_string())
    })
}

/// The import's message for a full disk (or nearly full, `guard`): what happened, where, and
/// what to do. `work` / `dataset` with their space now.
pub fn full_message(
    guard: bool,
    work: (&Path, Option<Space>),
    dataset: (&Path, Option<Space>),
    same: bool,
) -> String {
    // The fuller volume (relative to what it keeps free).
    let short = |s: Option<Space>| s.map_or(0, |s| s.margin().saturating_sub(s.available));
    let (dir, s, advice) = if same {
        (
            work.0,
            work.1,
            format!("{WORK_ADVICE}; {DATASET_ADVICE}").replacen(", or set", ", set", 1),
        )
    } else if dataset.1.is_some() && short(dataset.1) > short(work.1) {
        (dataset.0, dataset.1, DATASET_ADVICE.to_string())
    } else {
        (work.0, work.1, WORK_ADVICE.to_string())
    };
    let free = s.map_or_else(|| "unknown".into(), |s| size(s.available));
    if guard {
        format!(
            "the import stopped before the disk filled up: {free} free under {}, below the {} it \
             keeps free while running; free some space, {advice}",
            dir.display(),
            size(s.map_or(0, |s| s.floor())),
        )
    } else {
        format!(
            "the import stopped: the disk under {} is full ({free} free); free some space, \
             {advice}",
            dir.display()
        )
    }
}

/// Why the guard stopped an import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trip {
    /// The space fell below a volume's floor.
    Low,
    /// A write failed: the disk is full.
    Full,
}

/// The running free-space guard of one import. Its [`Guard::stop`] flag is the cancel flag of
/// every helper and pool of the job (the task's own cancel is mirrored into it), so a trip
/// kills the running helpers through the existing cancel path.
#[derive(Debug)]
pub struct Guard {
    pub stop: AtomicBool,
    work: PathBuf,
    dataset: PathBuf,
    tripped: Mutex<Option<Trip>>,
    /// What the import's error adds (frames written before it stopped).
    notes: Mutex<Vec<String>>,
    /// Space source (tests replace it).
    probe: fn(&Path) -> Option<Space>,
}

impl Guard {
    pub fn new(work: &Path, dataset: &Path) -> Self {
        Self::with_probe(work, dataset, space)
    }

    pub fn with_probe(work: &Path, dataset: &Path, probe: fn(&Path) -> Option<Space>) -> Self {
        Self {
            stop: AtomicBool::new(false),
            work: work.to_path_buf(),
            dataset: dataset.to_path_buf(),
            tripped: Mutex::new(None),
            notes: Mutex::new(Vec::new()),
            probe,
        }
    }

    fn check_dir(&self, dir: &Path, extra: u64) -> bool {
        if self.tripped().is_some() {
            return false;
        }
        match (self.probe)(dir) {
            Some(s) if s.available < s.floor().saturating_add(extra) => {
                self.trip(Trip::Low);
                false
            }
            _ => true,
        }
    }

    /// Whether the work folder's volume is above its floor; trips (and stops) when not.
    pub fn check_work(&self) -> bool {
        self.check_dir(&self.work, 0)
    }

    /// [`Guard::check_work`] with `extra` bytes still to be written (a download's rest).
    pub fn check_work_for(&self, extra: u64) -> bool {
        self.check_dir(&self.work, extra)
    }

    /// Whether the dataset folder's volume is above its floor; trips (and stops) when not.
    pub fn check_dataset(&self) -> bool {
        self.check_dir(&self.dataset, 0)
    }

    /// Stop the import for `why` (the first reason is kept).
    pub fn trip(&self, why: Trip) {
        if let Ok(mut t) = self.tripped.lock() {
            t.get_or_insert(why);
        }
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn tripped(&self) -> Option<Trip> {
        self.tripped.lock().ok().and_then(|t| *t)
    }

    /// Add `note` to the import's error.
    pub fn note(&self, note: String) {
        if let Ok(mut n) = self.notes.lock() {
            n.push(note);
        }
    }

    pub fn notes(&self) -> Vec<String> {
        self.notes.lock().map(|n| n.clone()).unwrap_or_default()
    }

    /// The import's error for a trip, with the space now.
    pub fn message(&self, why: Trip) -> String {
        full_message(
            why == Trip::Low,
            (&self.work, (self.probe)(&self.work)),
            (&self.dataset, (self.probe)(&self.dataset)),
            same_volume(&self.work, &self.dataset),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn sizes_scale_like_the_helper() {
        assert_eq!(scaled_size(1920, 1080, 640), (640, 360));
        assert_eq!(scaled_size(1080, 1920, 640), (360, 640));
        assert_eq!(scaled_size(3840, 2160, 640), (640, 360));
        assert_eq!(scaled_size(2592, 1944, 640), (640, 480));
        assert_eq!(scaled_size(640, 480, 640), (640, 480));
        assert_eq!(scaled_size(1920, 1080, 0), (1920, 1080));
        assert_eq!(scaled_size(5000, 3, 640), (640, 1));
        // 1080p: about 1 MB at full resolution, about 186 kB as a candidate.
        assert_eq!(full_frame_bytes(1920, 1080), 1_036_800 + 2_000);
        assert_eq!(candidate_frame_bytes(1920, 1080), 184_320 + 2_000);
        // A small video is not scaled: its candidate is the full frame.
        assert_eq!(candidate_frame_bytes(640, 360), full_frame_bytes(640, 360));
    }

    #[test]
    fn the_estimate_counts_candidates_reextracts_and_kept_frames() {
        let p = Plan {
            video_candidates: 20_000,
            image_candidates: 0,
            frames: 200,
            source: None,
            image_bytes: 0,
            download: 0,
        };
        let n = estimate(&p);
        assert_eq!(n.work, 20_000 * 186_320 + 200 * 1_038_800);
        assert_eq!(n.dataset, 200 * 1_038_800);
        // Before: 20 000 full frames in the work folder.
        assert!(n.work * 5 < 20_000 * full_frame_bytes(1920, 1080));
        // 4K: candidates stay small, the kept frames grow.
        let k = estimate(&Plan {
            source: Some((3840, 2160)),
            ..p
        });
        assert_eq!(k.work, 20_000 * 186_320 + 200 * (4_147_200 + 2_000));
        // Images only: nothing in the work folder, the copies at their mean size.
        let i = estimate(&Plan {
            video_candidates: 0,
            image_candidates: 50,
            frames: 200,
            source: None,
            image_bytes: 300_000,
            download: 7,
        });
        assert_eq!(
            i,
            Need {
                work: 7,
                dataset: 50 * 300_000
            }
        );
        // Mixed: the larger per-frame size.
        let m = estimate(&Plan {
            video_candidates: 10,
            image_candidates: 10,
            frames: 15,
            source: None,
            image_bytes: 3_000_000,
            download: 0,
        });
        assert_eq!(m.dataset, 15 * 3_000_000);
        assert_eq!(m.work, 10 * 186_320 + 10 * 1_038_800);
    }

    #[test]
    fn margins_are_five_percent_within_one_and_five_gib() {
        assert_eq!(margin(10 * GB), GIB);
        assert_eq!(margin(60 * GIB), 3 * GIB);
        assert_eq!(margin(4_000 * GB), 5 * GIB);
        let s = Space {
            available: 0,
            total: 60 * GIB,
        };
        assert_eq!(s.floor(), 3 * GIB / 2);
    }

    #[test]
    fn the_preflight_refuses_with_an_actionable_message() {
        let need = Need {
            work: 18_200_000_000,
            dataset: 0,
        };
        let s = Space {
            available: 12_900_000_000,
            total: 500 * GB,
        };
        let w = Path::new("/data/training/imports/x");
        let d = Path::new("/data/dataset");
        let m = refusal(need, (w, Some(s)), (d, Some(s)), false).unwrap();
        assert_eq!(
            m,
            "the import needs about 18.2 GB under /data/training/imports/x, 12.9 GB free (it keeps \
             5.4 GB free): lower Candidates per video (--max-candidates), sample less often (a \
             larger One frame every / --every, a lower Max frames per second / --max-fps), or set \
             training.dir (the work folder) to a bigger disk"
        );
        // Enough with the margin left: fine; unknown space: fine.
        let big = Space {
            available: 30 * GB,
            total: 500 * GB,
        };
        assert!(refusal(need, (w, Some(big)), (d, Some(big)), false).is_none());
        assert!(refusal(need, (w, None), (d, None), false).is_none());
        // On one volume the two add up.
        let both = Need {
            work: 15 * GB,
            dataset: 10 * GB,
        };
        let m = refusal(both, (w, Some(big)), (d, Some(big)), true).unwrap();
        assert!(
            m.starts_with("the import needs about 25.0 GB under /data/training"),
            "{m}"
        );
        assert!(
            m.contains("set training.dir") && m.contains("--out-dir"),
            "{m}"
        );
        assert!(refusal(both, (w, Some(big)), (d, Some(big)), false).is_none());
        // The dataset volume alone.
        let small = Space {
            available: 2 * GB,
            total: 20 * GB,
        };
        let m = refusal(both, (w, Some(big)), (d, Some(small)), false).unwrap();
        assert!(
            m.starts_with("the import needs about 10.0 GB under /data/dataset"),
            "{m}"
        );
        assert!(m.ends_with("(Save frames to, --out-dir)"), "{m}");
    }

    #[test]
    fn full_disks_are_recognized_in_rust_and_helper_errors() {
        let io = anyhow::Error::from(std::io::Error::from(std::io::ErrorKind::StorageFull))
            .context("writing x.jpg");
        assert!(is_disk_full(&io));
        #[cfg(unix)]
        assert!(is_disk_full(&anyhow::Error::from(
            std::io::Error::from_raw_os_error(28)
        )));
        #[cfg(windows)]
        assert!(is_disk_full(&anyhow::Error::from(
            std::io::Error::from_raw_os_error(112)
        )));
        let child = anyhow::Error::from(crate::resources::export::ChildFailed {
            program: "python".into(),
            status: "exit code: 1".into(),
            lines: vec![
                "[import_footage] error: no space left on device writing f.jpg ([Errno 28] No \
                 space left on device)"
                    .into(),
            ],
        })
        .context("the helper failed");
        assert!(is_disk_full(&child));
        let win = anyhow!("OSError: [WinError 112] There is not enough space on the disk");
        assert!(is_disk_full(&win));
        assert!(!is_disk_full(&anyhow!("cannot open the video")));
        assert!(!is_disk_full(&anyhow::Error::from(std::io::Error::from(
            std::io::ErrorKind::NotFound
        ))));
    }

    fn low(_: &Path) -> Option<Space> {
        Some(Space {
            available: 100_000_000,
            total: 100 * GB,
        })
    }
    fn plenty(_: &Path) -> Option<Space> {
        Some(Space {
            available: 50 * GB,
            total: 100 * GB,
        })
    }

    #[test]
    fn the_guard_trips_below_the_floor_and_stops() {
        let g = Guard::with_probe(Path::new("/w"), Path::new("/d"), plenty);
        assert!(g.check_work() && g.check_dataset());
        assert!(!g.check_work_for(49 * GB), "a large download's rest");
        assert_eq!(g.tripped(), Some(Trip::Low));
        let g = Guard::with_probe(Path::new("/w"), Path::new("/d"), low);
        assert!(!g.stop.load(Ordering::SeqCst));
        assert!(!g.check_work());
        assert!(g.stop.load(Ordering::SeqCst));
        assert_eq!(g.tripped(), Some(Trip::Low));
        g.trip(Trip::Full);
        assert_eq!(g.tripped(), Some(Trip::Low), "the first reason stays");
        let m = g.message(Trip::Low);
        assert!(
            m.starts_with(
                "the import stopped before the disk filled up: 100 MB free under /w, below the \
                 2.5 GB it keeps free while running; free some space, lower Candidates per video"
            ),
            "{m}"
        );
        let m = g.message(Trip::Full);
        assert!(
            m.starts_with("the import stopped: the disk under /w is full (100 MB free); free some space, lower Candidates per video"),
            "{m}"
        );
    }

    #[test]
    fn the_space_of_a_real_folder_is_known() {
        let d = std::env::temp_dir();
        let s = space(&d.join("not/yet/there")).unwrap();
        assert!(s.total > 0 && s.available <= s.total);
        assert!(same_volume(&d, &d.join("x/y")));
    }
}
