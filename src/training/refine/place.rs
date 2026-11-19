//! Where a dataset check's fold folders go ([`FoldRoot`]). Each fold exports its frames from the
//! folder under review and trains on that export: with the fold folders in the work folder
//! (`<training dir>/refine/<id>`) on another volume than the folder under review, every fold copied
//! its frames across volumes. When the training folder is on another volume, the folder under
//! review's volume allows hard links and has room ([`need`], the import's free-space margin kept),
//! the fold folders go to `<parent of the folder under review>/.blue-onyx-prism-work/refine/<id>`
//! instead ([`crate::storage::sweep::near_dataset_base`]: never inside the dataset or review
//! folder), where the export hard-links the frames; anything else keeps today's place.
//!
//! That folder is held busy like the work folder ([`crate::storage::Busy`]) and named in the work
//! folder before it is made ([`crate::storage::sweep::ELSEWHERE_FILE`]), so a sweep after a crash
//! removes it either way (its own marker under the near base, or through the work folder that
//! names it). It is removed when the check ends; when that fails, its marker stays and the next
//! sweep removes it. The fold folders hold only rebuildable files (exports, prepared datasets,
//! checkpoints, throwaway models), never a label.

use crate::storage::Busy;
use crate::storage::sweep::{ELSEWHERE_FILE, near_dataset_base};
use crate::training::import::disk::{self, Space};
use crate::training::preflight;
use std::path::{Path, PathBuf};

/// Bytes the fold folders may need on their volume: every selected image twice (the export's
/// copies where links fail, the downscaled training images, RF-DETR's night-augmented copies), the
/// checkpoints ([`preflight::WORK_WEIGHT_FACTOR`] x the base weights, one fold at a time) and the
/// label files. An upper bound: the fold's own disk check still runs ([`preflight::check_disk`]).
pub fn need(image_bytes: u64, weights: u64, frames: usize) -> u64 {
    image_bytes
        .saturating_mul(2)
        .saturating_add(weights.saturating_mul(preflight::WORK_WEIGHT_FACTOR))
        .saturating_add((frames as u64).saturating_mul(preflight::LABEL_BYTES_PER_FRAME))
}

/// What decides the place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Facts {
    /// The work folder is on the volume of the folder under review already.
    pub work_on_dataset_volume: bool,
    /// The near folder would be on the volume of the folder under review (not a mount point
    /// between them).
    pub near_on_dataset_volume: bool,
    /// A frame of the folder under review can be hard-linked there.
    pub links: bool,
    pub space: Option<Space>,
}

/// Whether the fold folders go near the dataset: only when it helps (another volume than the work
/// folder's, links there) and fits (`need` with the margin left). Pure.
pub fn near(f: &Facts, need: u64) -> bool {
    !f.work_on_dataset_volume
        && f.near_on_dataset_volume
        && f.links
        && f.space
            .is_some_and(|s| s.available >= need.saturating_add(s.margin()))
}

/// The folder the fold folders of a check go in, removed (when near the dataset) on drop.
#[derive(Debug)]
pub struct FoldRoot {
    pub dir: PathBuf,
    /// Near the dataset: its hold and the near base (removed when empty).
    near: Option<(Busy, PathBuf)>,
}

impl FoldRoot {
    /// Today's place: the work folder itself.
    pub fn in_work(work: &Path) -> Self {
        Self {
            dir: work.to_path_buf(),
            near: None,
        }
    }

    /// The fold folders of check `id` (work folder `work`) for frames of `review_dir`: near the
    /// dataset when [`near`] says so (`sample`: one selected frame, to probe the links; `need`:
    /// [`need`]), else [`FoldRoot::in_work`]. `dataset_dirs`: the dataset and review folders, which
    /// the near folder must lie outside of.
    pub fn choose(
        work: &Path,
        id: &str,
        review_dir: &Path,
        dataset_dirs: &[PathBuf],
        sample: Option<&Path>,
        need: u64,
    ) -> Self {
        let Some(base) = near_dataset_base(review_dir, dataset_dirs) else {
            return Self::in_work(work);
        };
        let facts = Facts {
            work_on_dataset_volume: disk::same_volume(work, review_dir),
            near_on_dataset_volume: disk::same_volume(&base, review_dir),
            links: sample.is_some_and(|f| preflight::hard_links_work(f, &base)),
            space: disk::space(&base),
        };
        if !near(&facts, need) {
            return Self::in_work(work);
        }
        let dir = base.join("refine").join(id);
        // Named before it exists: a crash from here on leaves it to the sweep.
        let named = std::fs::create_dir_all(work).and_then(|()| {
            std::fs::write(work.join(ELSEWHERE_FILE), dir.to_string_lossy().as_bytes())
        });
        if let Err(e) = named {
            tracing::warn!(
                "dataset check: could not name its fold folder in {} ({e}); the folds go there",
                work.display()
            );
            return Self::in_work(work);
        }
        let busy = Busy::hold(&dir);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!(
                "dataset check: could not create {} ({e}); the folds go to {}",
                dir.display(),
                work.display()
            );
            drop(busy);
            let _ = std::fs::remove_file(work.join(ELSEWHERE_FILE));
            remove_empty(&base);
            return Self::in_work(work);
        }
        tracing::info!(
            "dataset check: fold folders in {} (the volume of {}: its exports hard-link the \
             frames)",
            dir.display(),
            review_dir.display()
        );
        Self {
            dir,
            near: Some((busy, base)),
        }
    }

    /// The fold folders are near the dataset.
    pub fn is_near(&self) -> bool {
        self.near.is_some()
    }
}

/// Remove `base/refine` and `base` when empty.
fn remove_empty(base: &Path) {
    let _ = std::fs::remove_dir(base.join("refine"));
    let _ = std::fs::remove_dir(base);
}

impl Drop for FoldRoot {
    fn drop(&mut self) {
        let Some((busy, base)) = self.near.take() else {
            return;
        };
        match crate::resources::manager::remove_any(&self.dir) {
            Ok(()) => {
                drop(busy);
                remove_empty(&base);
            }
            Err(e) => {
                tracing::warn!(
                    "dataset check: could not remove its fold folder {} ({e:#}); the next storage \
                     sweep removes it",
                    self.dir.display()
                );
                busy.abandon();
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::storage::busy::{self, State};

    const GB: u64 = 1_000_000_000;

    fn facts(free: u64) -> Facts {
        Facts {
            work_on_dataset_volume: false,
            near_on_dataset_volume: true,
            links: true,
            space: Some(Space {
                available: free,
                total: 240 * GB,
            }),
        }
    }

    #[test]
    fn the_folds_go_near_the_dataset_only_when_it_helps_and_fits() {
        let need = need(GB, 120_000_000, 2000);
        assert_eq!(
            need,
            2 * GB + 5 * 120_000_000 + 2000 * preflight::LABEL_BYTES_PER_FRAME
        );
        // Another volume with room: near.
        assert!(near(&facts(90 * GB), need));
        // The work folder is on that volume already, a mount point between, no links there, or
        // no room once the margin is kept: today's place.
        let mut f = facts(90 * GB);
        f.work_on_dataset_volume = true;
        assert!(!near(&f, need));
        let mut f = facts(90 * GB);
        f.near_on_dataset_volume = false;
        assert!(!near(&f, need));
        let mut f = facts(90 * GB);
        f.links = false;
        assert!(!near(&f, need));
        let tight = facts(need + 1);
        assert!(!near(&tight, need), "the margin is kept");
        let mut f = facts(90 * GB);
        f.space = None;
        assert!(!near(&f, need), "unknown space");
    }

    #[test]
    fn a_near_root_is_held_and_removed_and_an_abandoned_hold_reads_as_crashed() {
        let d = std::env::temp_dir().join(format!("bop-fold-root-{}", uuid::Uuid::new_v4()));
        let work = d.join("training/refine/r1");
        let review = d.join("dataset");
        std::fs::create_dir_all(&review).unwrap();
        // Same volume (a temp folder): today's place.
        let r = FoldRoot::choose(&work, "r1", &review, std::slice::from_ref(&review), None, 0);
        assert!(!r.is_near());
        assert_eq!(r.dir, work);
        drop(r);
        // The near place itself, as `choose` makes it on another volume.
        let base = near_dataset_base(&review, std::slice::from_ref(&review)).unwrap();
        assert_eq!(base, d.join(crate::storage::sweep::NEAR_DATASET_DIR));
        let dir = base.join("refine/r1");
        let root = FoldRoot {
            dir: dir.clone(),
            near: Some((Busy::hold(&dir), base.clone())),
        };
        std::fs::create_dir_all(dir.join("fold-1/export")).unwrap();
        std::fs::write(dir.join("fold-1/export/a.jpg"), b"x").unwrap();
        assert!(matches!(busy::probe(&dir), State::Live));
        drop(root);
        assert!(
            !dir.exists() && !base.exists(),
            "removed with its empty parents"
        );
        assert!(matches!(busy::probe(&dir), State::Ended));
        // A removal that fails keeps the marker, unlocked: the sweep sees a crashed job.
        std::fs::create_dir_all(&dir).unwrap();
        let held = Busy::hold(&dir);
        held.abandon();
        assert!(matches!(busy::probe(&dir), State::Crashed(_)));
        let _ = std::fs::remove_dir_all(&d);
    }
}
