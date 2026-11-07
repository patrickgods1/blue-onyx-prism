//! What the index reads from disk besides frames: when the day folders, their `index.jsonl` and
//! `review-queue.jsonl` last changed ([`DayStat`], [`FileStat`]), and their lines. A day folder's
//! stat is taken before its index file is read, so a change in between shows as a change at the
//! next scan rather than getting lost.

use super::super::super::layout::{self, ReviewLine};
use super::super::super::record::IndexLine;
use super::super::queue::{self, Candidate};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// A folder or file whose time is this close to the scan may still change within that time unseen
/// (file times are coarse), so it is checked again at the next scan.
const RACY_WINDOW: Duration = Duration::from_secs(2);

/// A file as a stat sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FileStat {
    pub len: u64,
    pub modified: Option<SystemTime>,
}

impl FileStat {
    pub(super) fn of(path: &Path) -> Option<Self> {
        let m = std::fs::metadata(path).ok()?;
        Some(Self {
            len: m.len(),
            modified: m.modified().ok(),
        })
    }
}

fn racy(time: Option<SystemTime>, now: SystemTime) -> bool {
    time.is_none_or(|t| now.duration_since(t).map_or(true, |age| age < RACY_WINDOW))
}

/// How a day folder looked, and the damaged lines of its index file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DayStat {
    pub folder: Option<SystemTime>,
    pub index: Option<FileStat>,
    /// Changed so close to the scan that a later change may not show: walk it again next time.
    pub racy: bool,
    pub damaged: u64,
}

impl DayStat {
    pub(super) fn of(day: &Path, now: SystemTime) -> Self {
        let folder = std::fs::metadata(day).ok().and_then(|m| m.modified().ok());
        let index = FileStat::of(&day.join(layout::INDEX_FILE));
        Self {
            folder,
            index,
            racy: racy(folder, now) || index.is_some_and(|i| racy(i.modified, now)),
            damaged: 0,
        }
    }

    /// Whether the folder must be walked again, this being how it looks now.
    pub(super) fn changed_from(&self, old: &Self) -> bool {
        old.racy || self.folder != old.folder || self.index != old.index
    }
}

/// How `review-queue.jsonl` looked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct QueueStat {
    pub file: Option<FileStat>,
    pub racy: bool,
    pub damaged: u64,
}

impl QueueStat {
    pub(super) fn of(dir: &Path, now: SystemTime) -> Self {
        let file = FileStat::of(&dir.join(layout::REVIEW_QUEUE_FILE));
        Self {
            file,
            racy: file.is_some_and(|f| racy(f.modified, now)),
            damaged: 0,
        }
    }

    pub(super) fn changed_from(&self, old: &Self) -> bool {
        old.racy || self.file != old.file
    }
}

/// The review queue, read.
#[derive(Debug, Clone, Default)]
pub(super) struct QueueRead {
    pub stat: QueueStat,
    pub lines: Vec<ReviewLine>,
}

/// `dir`'s review queue, read at `now`.
pub(super) fn read_queue(dir: &Path, now: SystemTime) -> QueueRead {
    let mut stat = QueueStat::of(dir, now);
    let read = layout::read_lines::<ReviewLine>(&dir.join(layout::REVIEW_QUEUE_FILE));
    stat.damaged = read.damaged as u64;
    QueueRead {
        stat,
        lines: read.lines,
    }
}

/// `<scene>/<camera>/<date>` of a day folder.
pub(super) fn day_key(day: &Path) -> Option<String> {
    let mut parts = day.iter().rev().map(std::ffi::OsStr::to_str);
    let (date, camera, scene) = (parts.next()??, parts.next()??, parts.next()??);
    Some(format!("{scene}/{camera}/{date}"))
}

/// One day folder, read.
#[derive(Debug)]
pub(super) struct DayRead {
    pub key: String,
    pub stat: DayStat,
    pub lines: Vec<IndexLine>,
}

impl DayRead {
    fn of(folder: &Path, key: String, now: SystemTime) -> Self {
        let mut stat = DayStat::of(folder, now);
        let read = layout::read_lines::<IndexLine>(&folder.join(layout::INDEX_FILE));
        stat.damaged = read.damaged as u64;
        Self {
            key,
            stat,
            lines: read.lines,
        }
    }

    /// A line naming a frame of another folder (a folder renamed by hand, say): such a frame's
    /// row is not this day's, and a walk of this day alone cannot be exact.
    pub(super) fn foreign(&self) -> bool {
        let mut parts = self.key.split('/');
        let (scene, camera) = (parts.next(), parts.next());
        self.lines
            .iter()
            .any(|l| Some(l.scene.as_str()) != scene || Some(l.camera.as_str()) != camera)
    }

    fn date(&self) -> &str {
        self.key.rsplit('/').next().unwrap_or_default()
    }
}

/// The folder of the day `key` under `dir`, when it is one [`layout::day_dirs`] lists.
pub(super) fn read_day(dir: &Path, key: &str, now: SystemTime) -> Option<DayRead> {
    layout::parse_rel(&format!("{key}/x.jpg"))?;
    let folder = key.split('/').fold(dir.to_path_buf(), |p, c| p.join(c));
    let meta = std::fs::symlink_metadata(&folder).ok()?;
    meta.is_dir()
        .then(|| DayRead::of(&folder, key.to_string(), now))
}

/// The day folders of `dir` and how they look now.
pub(super) fn stat_days(dir: &Path, now: SystemTime) -> Vec<(String, DayStat)> {
    layout::day_dirs(dir)
        .iter()
        .filter_map(|d| Some((day_key(d)?, DayStat::of(d, now))))
        .collect()
}

/// The lines `queue` and `days` as [`queue::candidates`] with [`queue::QueueSource::All`] would list them.
pub(super) fn candidates_of(
    queue: Vec<ReviewLine>,
    days: Vec<DayRead>,
) -> (Vec<Candidate>, Vec<Stats>) {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    queue::review_candidates(queue, &mut seen, &mut out);
    let mut stats = Vec::with_capacity(days.len());
    for day in days {
        let foreign = day.foreign();
        let date = day.date().to_string();
        queue::index_candidates(day.lines, &date, &mut seen, &mut out);
        stats.push(Stats {
            key: day.key,
            stat: day.stat,
            foreign,
        });
    }
    (out, stats)
}

/// What is kept of a [`DayRead`] once its lines are candidates.
#[derive(Debug)]
pub(super) struct Stats {
    pub key: String,
    pub stat: DayStat,
    pub foreign: bool,
}

/// Every day folder of `dir`, read in [`layout::day_dirs`]' order.
pub(super) fn read_all_days(dir: &Path, now: SystemTime) -> Vec<DayRead> {
    let folders: Vec<PathBuf> = layout::day_dirs(dir);
    let reads = crate::parallel::map_all(
        folders.len(),
        crate::parallel::walk_threads_over(folders.len()),
        &|i| {
            let folder = folders.get(i)?;
            Some(DayRead::of(folder, day_key(folder)?, now))
        },
    );
    reads.into_iter().flatten().collect()
}
