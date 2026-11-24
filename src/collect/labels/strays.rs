//! Images in the folder under review that no frame of the dataset layout names ("strays"), for the
//! Review tab's Rescan folder (`POST /v1/review/rescan`): a `.jpg` copied into a day folder with no
//! `index.jsonl` or `review-queue.jsonl` line, a `.png` there, or an image at the wrong depth (in
//! the folder itself, a scene folder or a camera folder). Neither the review index nor the walk
//! ([`super::queue::candidates`]) can list them; Import is how outside images enter the layout.
//!
//! Read only: nothing here writes, moves or deletes a file. What is looked at:
//!
//! - every day folder of [`layout::day_dirs_checked`] (symlinked and badly named folders skipped as
//!   everywhere), listed on [`crate::parallel::map_all`]'s threads. An image there is a regular
//!   file (a symlink is not followed, and not counted) with an extension of
//!   [`IMAGE_EXTS`], in any case. It is known when its stem has a line naming that day folder: a
//!   review-queue line, or an index line of any day folder (a line naming another folder than its
//!   own, as a foreign index has, counts for the folder it names). A `.JPG` of a known stem is
//!   known where the file system finds it as `<stem>.jpg` (no exact `<stem>.jpg` beside it): there
//!   [`super::Frame::resolve`] opens it;
//! - images directly in the folder, a scene folder (`day`, `night`, `ir`) or a camera folder (one
//!   with a camera name the layout takes): always strays, no line can name them. No other folder
//!   is walked: an export's `images/`, `refine/`, `spots/` and whatever else a user keeps there are
//!   left alone.
//!
//! A folder that cannot be listed is counted in [`Strays::unreadable`] (the scene and camera
//! folders as [`layout::day_dirs_checked`] reports them), never an error. Memory: the queue's
//! frames, and per day folder being listed its index ids and image names; the strays' paths
//! until they are counted. The sample is the first [`SAMPLE`] paths in sorted order, with `/`
//! separators on every OS.

use crate::collect::layout::{self, INDEX_FILE, REVIEW_QUEUE_FILE, SCENES};
use crate::training::import::source::IMAGE_EXTS;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Most stray paths an answer names.
pub const SAMPLE: usize = 5;

/// What [`find`] found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Strays {
    /// Images no frame of the layout names.
    pub count: u64,
    /// The first [`SAMPLE`] of them in sorted order, relative to the folder, `/`-separated.
    pub sample: Vec<String>,
    /// Folders that could not be listed.
    pub unreadable: u64,
}

/// The strays of the folder `dir` (see the module's first lines).
pub fn find(dir: &Path) -> Strays {
    let queued = queued_by_day(dir);
    let (days, unlisted) = layout::day_dirs_checked(dir);
    let mut unreadable = unlisted.len() as u64;
    let mut strays: Vec<String> = Vec::new();
    unreadable += wrong_depth(dir, &mut strays);
    let threads = crate::parallel::walk_threads_over(days.len());
    // A UI walk like the review index's: its spawned threads run below normal priority.
    let spawned = crate::parallel::Spawned::Background;
    let found = crate::parallel::map_all_at(days.len(), threads, spawned, &|i| {
        days.get(i)
            .map(|day| in_day(dir, day, &queued))
            .unwrap_or_default()
    });
    let mut foreign: HashSet<String> = HashSet::new();
    for day in found {
        unreadable += u64::from(day.unreadable);
        strays.extend(day.strays);
        foreign.extend(day.foreign);
    }
    if !foreign.is_empty() {
        strays.retain(|rel| !foreign.contains(rel));
    }
    strays.sort_unstable();
    Strays {
        count: strays.len() as u64,
        sample: strays.into_iter().take(SAMPLE).collect(),
        unreadable,
    }
}

/// The stems of the review queue's frames, by `<scene>/<camera>/<date>`.
fn queued_by_day(dir: &Path) -> HashMap<String, HashSet<String>> {
    let mut by_day: HashMap<String, HashSet<String>> = HashMap::new();
    for line in layout::read_review(&dir.join(REVIEW_QUEUE_FILE)) {
        if let Some(r) = layout::parse_rel(&line.frame).filter(|_| line.frame.ends_with(".jpg")) {
            by_day.entry(r.day_dir()).or_default().insert(r.stem);
        }
    }
    by_day
}

/// Whether `name` is an image by its extension, in any case.
fn is_image(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// The names of the images in `folder` (regular files only, symlinks not followed); None when it
/// exists and cannot be listed.
fn images_in(folder: &Path) -> Option<Vec<String>> {
    let rd = match std::fs::read_dir(folder) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(Vec::new()),
        Err(_) => return None,
    };
    Some(
        rd.flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| is_image(n))
            .collect(),
    )
}

/// The real folders in `folder` (symlinks skipped), by name; none when it cannot be listed
/// ([`layout::day_dirs_checked`] counts that).
fn folders_in(folder: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    rd.flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .collect()
}

/// Add the images directly in `dir`, its scene folders and their camera folders to `strays`;
/// 1 when `dir` itself cannot be listed, else 0.
fn wrong_depth(dir: &Path, strays: &mut Vec<String>) -> u64 {
    let Some(top) = images_in(dir) else {
        return 1;
    };
    strays.extend(top);
    for scene in SCENES {
        let scene_dir = dir.join(scene);
        // A symlinked scene folder is not walked, as `day_dirs_checked` skips it.
        if !std::fs::symlink_metadata(&scene_dir).is_ok_and(|m| m.file_type().is_dir()) {
            continue;
        }
        strays.extend(
            images_in(&scene_dir)
                .unwrap_or_default()
                .into_iter()
                .map(|n| format!("{scene}/{n}")),
        );
        for camera in folders_in(&scene_dir) {
            // A camera name the layout takes: the folders `day_dirs_checked` looks in.
            if layout::parse_rel(&format!("{scene}/{camera}/2000-01-01/a.jpg")).is_none() {
                continue;
            }
            strays.extend(
                images_in(&scene_dir.join(&camera))
                    .unwrap_or_default()
                    .into_iter()
                    .map(|n| format!("{scene}/{camera}/{n}")),
            );
        }
    }
    0
}

/// What one day folder holds.
#[derive(Debug, Default)]
struct DayFound {
    strays: Vec<String>,
    /// Frames its index lines name in other day folders.
    foreign: Vec<String>,
    unreadable: u8,
}

/// The strays of the day folder `day` of `dir`.
fn in_day(dir: &Path, day: &Path, queued: &HashMap<String, HashSet<String>>) -> DayFound {
    let key = day
        .strip_prefix(dir)
        .unwrap_or(day)
        .iter()
        .map(|p| p.to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    let Some(images) = images_in(day) else {
        return DayFound {
            unreadable: 1,
            ..DayFound::default()
        };
    };
    if images.is_empty() {
        return DayFound::default();
    }
    let date = day.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let mut ids: HashSet<String> = HashSet::new();
    let mut foreign = Vec::new();
    for line in layout::read_index(&day.join(INDEX_FILE)) {
        let rel = format!("{}/{}/{date}/{}.jpg", line.scene, line.camera, line.id);
        let Some(r) = layout::parse_rel(&rel) else {
            continue;
        };
        if r.day_dir() == key {
            ids.insert(r.stem);
        } else {
            foreign.push(rel);
        }
    }
    let queue = queued.get(&key);
    let known = |stem: &str| ids.contains(stem) || queue.is_some_and(|q| q.contains(stem));
    let exact: HashSet<&str> = images
        .iter()
        .filter_map(|n| n.strip_suffix(".jpg"))
        .collect();
    let strays = images
        .iter()
        .filter(|name| {
            if let Some(stem) = name.strip_suffix(".jpg") {
                return !known(stem);
            }
            let path = Path::new(name.as_str());
            let jpg = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("jpg"));
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                return true;
            };
            // `x.JPG` is the frame `x` where `x.jpg` opens it (a file system that ignores case).
            let opens = jpg
                && known(stem)
                && !exact.contains(stem)
                && std::fs::symlink_metadata(day.join(format!("{stem}.jpg")))
                    .is_ok_and(|m| m.file_type().is_file());
            !opens
        })
        .map(|name| format!("{key}/{name}"))
        .collect();
    DayFound {
        strays,
        foreign,
        unreadable: 0,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::collect::layout::ReviewLine;
    use crate::collect::record::IndexLine;
    use std::path::PathBuf;

    /// A folder of its own, removed when the test ends.
    struct Dir(PathBuf);

    impl Dir {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("bop-strays-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn put(&self, rel: &str) {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"x").unwrap();
        }

        /// An index line for `id` in the day folder `day` (`scene/camera/date`), naming
        /// `scene`/`camera`.
        fn index(&self, day: &str, scene: &str, camera: &str, id: &str) {
            let line = IndexLine {
                id: id.into(),
                timestamp: "2026-11-20T00:00:00Z".into(),
                scene: scene.into(),
                camera: camera.into(),
                model: "m".into(),
                buckets: Vec::new(),
                labels: Vec::new(),
                bytes: 1,
            };
            let path = self.0.join(day).join(INDEX_FILE);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut text = std::fs::read_to_string(&path).unwrap_or_default();
            text.push_str(&serde_json::to_string(&line).unwrap());
            text.push('\n');
            std::fs::write(path, text).unwrap();
        }

        fn queue(&self, frame: &str) {
            let r = layout::parse_rel(frame).unwrap();
            let line = ReviewLine {
                version: layout::REVIEW_VERSION,
                frame: frame.into(),
                sidecar: frame.replace(".jpg", ".json"),
                id: r.stem,
                buckets: Vec::new(),
                scene: r.scene,
                camera: r.camera,
                queued_at: "2026-11-20T00:00:00Z".into(),
            };
            let path = self.0.join(REVIEW_QUEUE_FILE);
            let mut text = std::fs::read_to_string(&path).unwrap_or_default();
            text.push_str(&serde_json::to_string(&line).unwrap());
            text.push('\n');
            std::fs::write(path, text).unwrap();
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                // A test may have made a folder unlistable.
                for day in layout::day_dirs(&self.0) {
                    let _ = std::fs::set_permissions(day, std::fs::Permissions::from_mode(0o755));
                }
            }
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const DAY: &str = "day/drive/2026-11-20";

    #[test]
    fn an_empty_or_missing_folder_has_none() {
        let d = Dir::new();
        assert_eq!(find(&d.0), Strays::default());
        assert_eq!(find(&d.0.join("gone")), Strays::default());
    }

    #[test]
    fn indexed_and_queued_frames_are_no_strays_and_the_rest_are_in_sorted_order() {
        let d = Dir::new();
        // Indexed, with its sidecar and label file beside it (not images).
        d.put(&format!("{DAY}/20261120T000000000Z_a.jpg"));
        d.put(&format!("{DAY}/20261120T000000000Z_a.json"));
        d.put(&format!("{DAY}/20261120T000000000Z_a.labels.json"));
        d.index(DAY, "day", "drive", "20261120T000000000Z_a");
        // Queued only: its line in the review queue, none in the index.
        d.put(&format!("{DAY}/20261120T000000000Z_q.jpg"));
        d.queue(&format!("{DAY}/20261120T000000000Z_q.jpg"));
        // Strays: no line, another image type, wrong depths.
        for rel in [
            format!("{DAY}/copied.jpg"),
            format!("{DAY}/photo.png"),
            "top.jpg".to_string(),
            "night/scene-level.jpeg".to_string(),
            "ir/gate/camera-level.bmp".to_string(),
        ] {
            d.put(&rel);
        }
        // Not walked: other subfolders, other files.
        d.put("export/images/x.jpg");
        d.put("refine/a.png");
        d.put(&format!("{DAY}/sub/deeper.jpg"));
        d.put("day/Not A Camera/y.jpg");
        d.put("day/drive/not-a-date/z.jpg");
        d.put(&format!("{DAY}/notes.txt"));
        let s = find(&d.0);
        assert_eq!(s.count, 5, "{s:?}");
        assert_eq!(
            s.sample,
            [
                "day/drive/2026-11-20/copied.jpg",
                "day/drive/2026-11-20/photo.png",
                "ir/gate/camera-level.bmp",
                "night/scene-level.jpeg",
                "top.jpg",
            ]
        );
        assert_eq!(s.unreadable, 0);
    }

    #[test]
    fn extensions_are_read_in_any_case() {
        let d = Dir::new();
        for name in ["a.PNG", "b.Jpeg", "c.BMP", "d.jpG", "e.gif", "f"] {
            d.put(&format!("{DAY}/{name}"));
        }
        let s = find(&d.0);
        assert_eq!(s.count, 4, "{s:?}");
        assert!(s.sample.iter().all(|p| !p.ends_with(".gif")), "{s:?}");
    }

    #[test]
    fn an_upper_case_jpg_of_a_known_frame_is_a_stray_only_where_it_does_not_open() {
        let d = Dir::new();
        d.put(&format!("{DAY}/20261120T000000000Z_u.JPG"));
        d.index(DAY, "day", "drive", "20261120T000000000Z_u");
        let opens =
            std::fs::symlink_metadata(d.0.join(DAY).join("20261120T000000000Z_u.jpg")).is_ok();
        let s = find(&d.0);
        assert_eq!(s.count, u64::from(!opens), "{s:?}");
    }

    #[test]
    fn more_than_the_sample_are_counted_and_the_first_named() {
        let d = Dir::new();
        for i in 0..12 {
            d.put(&format!("{DAY}/s{i:02}.jpg"));
        }
        let s = find(&d.0);
        assert_eq!(s.count, 12);
        assert_eq!(s.sample.len(), SAMPLE);
        assert_eq!(s.sample[0], format!("{DAY}/s00.jpg"));
        assert_eq!(s.sample[4], format!("{DAY}/s04.jpg"));
    }

    #[test]
    fn a_line_of_another_day_folder_names_the_frame_it_points_at() {
        let d = Dir::new();
        d.put(&format!("{DAY}/20261120T000000000Z_f.jpg"));
        // The line is in the yard's index, and names the drive's frame.
        d.index(
            "day/yard/2026-11-20",
            "day",
            "drive",
            "20261120T000000000Z_f",
        );
        // A frame of the yard no line names: still a stray.
        d.put("day/yard/2026-11-20/lost.jpg");
        let s = find(&d.0);
        assert_eq!(s.sample, ["day/yard/2026-11-20/lost.jpg"], "{s:?}");
    }

    #[test]
    fn a_frame_line_of_another_day_does_not_cover_a_same_named_image_here() {
        let d = Dir::new();
        d.put(&format!("{DAY}/20261120T000000000Z_x.jpg"));
        d.index(
            "day/yard/2026-11-20",
            "day",
            "yard",
            "20261120T000000000Z_x",
        );
        d.queue("ir/drive/2026-11-20/20261120T000000000Z_x.jpg");
        assert_eq!(find(&d.0).count, 1);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_files_and_folders_are_not_followed() {
        let d = Dir::new();
        let outside = Dir::new();
        outside.put("real.jpg");
        outside.put("2026-11-21/inside.jpg");
        std::fs::create_dir_all(d.0.join(DAY)).unwrap();
        std::os::unix::fs::symlink(outside.0.join("real.jpg"), d.0.join(DAY).join("link.jpg"))
            .unwrap();
        std::os::unix::fs::symlink(outside.0.join("real.jpg"), d.0.join("top-link.jpg")).unwrap();
        // A symlinked day folder and a symlinked scene folder.
        std::os::unix::fs::symlink(
            outside.0.join("2026-11-21"),
            d.0.join("day/drive/2026-11-21"),
        )
        .unwrap();
        std::os::unix::fs::symlink(outside.0.join("2026-11-21"), d.0.join("night")).unwrap();
        let s = find(&d.0);
        assert_eq!(s, Strays::default());
    }

    #[cfg(unix)]
    #[test]
    fn a_folder_that_cannot_be_listed_is_counted_not_an_error() {
        use std::os::unix::fs::PermissionsExt as _;
        let d = Dir::new();
        d.put(&format!("{DAY}/hidden.jpg"));
        d.put("day/drive/2026-11-21/seen.jpg");
        let day = d.0.join(DAY);
        std::fs::set_permissions(&day, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::read_dir(&day).is_ok() {
            // Running as root: permissions do not stop a listing.
            return;
        }
        let s = find(&d.0);
        assert_eq!(s.unreadable, 1, "{s:?}");
        assert_eq!(s.sample, ["day/drive/2026-11-21/seen.jpg"]);
    }
}
