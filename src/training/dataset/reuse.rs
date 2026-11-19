//! Reusing the image files of an identical prepared dataset: the seeds of a comparison group
//! assemble the same plan, and writing it again copied or re-encoded every image (930 MB, about
//! 30 s per seed with the dataset on another volume than the training folder). [`super::write`]
//! records every image it wrote in [`PREPARED_FILES_FILE`] (written last, atomically) with the
//! plan's fingerprint ([`plan_fingerprint`]); a later run given that folder as a donor
//! ([`super::write_reusing`]) hard-links a donor image instead of writing it only when:
//!
//! - the donor's record has this format and the same plan fingerprint: the same options, classes,
//!   frames with the same labels and splits, and this build (the writer's code);
//! - the image comes from the same source path, whose identity ([`FileId`]: size and last-write
//!   time) is what it was when the donor wrote it;
//! - the donor's file still has the identity it had when written (nothing changed it since, the
//!   trainer included), and keeps it once linked (checked again after the link);
//! - it is no JPEG without its end marker ([`super::jpeg_without_end`]: the one kind of image a
//!   trainer writes back; such a file is a copy of its own in every dataset).
//!
//! Anything else, a link that fails (another volume, a file system without hard links) included,
//! writes the image from its source as before, so the dataset is the same bytes either way.
//! Nothing writes into a prepared image in place: [`super::write`] creates every file under a new
//! name (a link, a copy, or a temporary file renamed over), a run's folder is only ever replaced or
//! removed as a whole, and removing a run's folder (Remove) only drops its names, never another
//! run's.

use super::super::file_id::FileId;
use super::{FrameRef, Options, Plan, Source};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The record of a prepared dataset's image files, in the dataset folder.
pub const PREPARED_FILES_FILE: &str = "prepared_files.json";
/// Format of [`PREPARED_FILES_FILE`] and of what [`plan_fingerprint`] covers.
pub const PREPARED_FILES_FORMAT: u32 = 1;

/// One image the writer put into a dataset folder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedFile {
    pub source: PathBuf,
    /// The source's identity when it was written.
    pub source_id: FileId,
    /// The written file's identity right after it was written.
    pub written: FileId,
    /// Its size when it was written downscaled; None = linked or copied as it is.
    pub resized: Option<(u32, u32)>,
}

/// [`PREPARED_FILES_FILE`]: the plan's fingerprint and every image, by its path relative to the
/// dataset folder (`images/<name>`, `heldout/images/<name>`, `/`-separated).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PreparedFiles {
    pub format: u32,
    pub plan: String,
    pub files: BTreeMap<String, PreparedFile>,
}

/// Hex SHA-256 of what decides the bytes of every image and label of a dataset written from
/// `plan` (see the module docs): the options (the prepared image size included), the plan, every
/// frame it refers to (path, boxes,
/// ignore regions, scene, camera, day, flags), in plan order, and this build. `sources` are the
/// frames as the writer reads them ([`super::at_run_input`]).
pub fn plan_fingerprint(sources: &[Source], plan: &Plan, opts: &Options) -> String {
    let refs = plan
        .train
        .iter()
        .map(|(r, _)| *r)
        .chain(plan.valid.iter().copied())
        .chain(plan.heldout.iter().copied());
    let frames: Vec<(FrameRef, Option<&super::Frame>)> = refs
        .map(|r| (r, sources.get(r.0).and_then(|s| s.frames.get(r.1))))
        .collect();
    let bytes = serde_json::to_vec(&(
        PREPARED_FILES_FORMAT,
        crate::long_version(),
        opts,
        // Not serialized with the options (`dataset.json` states it apart), and it decides
        // which images are downscaled and to what.
        opts.prepared_short_side,
        plan,
        frames,
    ))
    .unwrap_or_default();
    crate::resources::manager::sha256_hex(&bytes)
}

/// A dataset folder whose record matches the plan being written.
#[derive(Debug, Clone)]
pub struct Donor {
    pub dir: PathBuf,
    record: PreparedFiles,
}

/// The first of `donors` (dataset folders) whose record reads, has this format and the plan
/// fingerprint `plan`. None: nothing to reuse.
pub fn find_donor(donors: &[PathBuf], plan: &str) -> Option<Donor> {
    donors.iter().find_map(|dir| {
        let bytes = std::fs::read(dir.join(PREPARED_FILES_FILE)).ok()?;
        let record: PreparedFiles = serde_json::from_slice(&bytes).ok()?;
        (record.format == PREPARED_FILES_FORMAT && record.plan == plan).then(|| Donor {
            dir: dir.clone(),
            record,
        })
    })
}

impl Donor {
    /// Hard-link the donor's `rel` to `dst` when it may stand for writing `src` there (see the
    /// module docs). Some(the size it was downscaled to, if it was) once linked; None: write it.
    pub fn link(&self, rel: &str, src: &Path, dst: &Path) -> Option<Option<(u32, u32)>> {
        let e = self.record.files.get(rel)?;
        if e.source != src || FileId::of(src)? != e.source_id {
            return None;
        }
        let from = self.dir.join(rel);
        if FileId::of(&from)? != e.written || super::jpeg_without_end(&from) {
            return None;
        }
        std::fs::hard_link(&from, dst).ok()?;
        // Replaced between the check and the link: not the file recorded.
        if FileId::of(dst) != Some(e.written) {
            let _ = std::fs::remove_file(dst);
            return None;
        }
        Some(e.resized)
    }
}

/// The record of the dataset being written.
#[derive(Debug, Default)]
pub struct Recorder {
    files: BTreeMap<String, PreparedFile>,
}

impl Recorder {
    /// `dst` (at `rel`) was written from `src`, whose identity was `before` when its writing
    /// started. Not recorded when the source changed meanwhile or an identity cannot be read
    /// (that image is then never reused).
    pub fn add(
        &mut self,
        rel: String,
        src: &Path,
        before: Option<FileId>,
        dst: &Path,
        resized: Option<(u32, u32)>,
    ) {
        let (Some(before), Some(after), Some(written)) = (before, FileId::of(src), FileId::of(dst))
        else {
            return;
        };
        if before != after {
            return;
        }
        self.files.insert(
            rel,
            PreparedFile {
                source: src.to_path_buf(),
                source_id: before,
                written,
                resized,
            },
        );
    }

    /// Write the record into `out` (last, through a temporary file and a rename: its presence
    /// says the folder is whole).
    pub fn write(self, out: &Path, plan: String) -> Result<()> {
        let rec = PreparedFiles {
            format: PREPARED_FILES_FORMAT,
            plan,
            files: self.files,
        };
        let path = out.join(PREPARED_FILES_FILE);
        crate::collect::layout::write_atomic(&path, &serde_json::to_vec(&rec)?)
            .with_context(|| format!("writing {}", path.display()))
    }
}
