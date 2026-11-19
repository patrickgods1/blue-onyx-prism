//! When a training run may skip its separate self-check process (`train_model.py --self-check`,
//! about 13 s of PyTorch imports before the training process imports them all again): only when
//! nothing it checked can have changed since the last check that passed. A passing check records
//! a [`Fingerprint`] of what it ran on in `<train env>/.prism-self-check.json` ([`MARKER_FILE`],
//! written atomically, one entry per trainer and device, [`key`]); a run whose fingerprint equals
//! a recorded one younger than [`MAX_AGE`] skips the check and says so in its log ("self-check
//! skipped (unchanged since ...)").
//!
//! The fingerprint covers the environment's marker (its bytes and file identity: a rebuild
//! rewrites it, and `uv venv --clear` deletes this file with the rest of the environment), the
//! environment's Python executable, the embedded training and export scripts (SHA-256), the
//! check's arguments (trainer, `--device`), the GPU as described (name, compute capability,
//! memory, PyTorch lane), the platform and this build. Any doubt runs the check: no environment
//! marker, a file that cannot be read, a missing, damaged or newer marker file, another
//! fingerprint, an entry older than [`MAX_AGE`] (a driver can change under an unchanged
//! environment) or dated in the future. A training process that fails afterwards forgets the
//! entry ([`forget`]), so the next run checks again. Nothing here changes what a run trains on
//! or how: the check only decides whether a run starts at all.

use super::capability::Accelerator;
use super::file_id::FileId;
use crate::resources::catalog::Platform;
use crate::resources::export::train::TrainPaths;
use crate::resources::manager;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// The record of passed checks, inside the training environment.
pub const MARKER_FILE: &str = ".prism-self-check.json";
/// Format of [`MARKER_FILE`]; a file of another format is ignored (the check runs) and replaced.
pub const MARKER_FORMAT: u32 = 1;
/// Longest a passed check stands for: a GPU driver can be updated under an unchanged environment,
/// and a check costs seconds against hours of training.
pub const MAX_AGE: Duration = Duration::from_secs(12 * 3600);

/// `<train env>/.prism-self-check.json`.
pub fn marker_path(paths: &TrainPaths) -> PathBuf {
    paths.export.env.join(MARKER_FILE)
}

/// The entry of a check of `trainer` on `device` (`--device`, e.g. `cuda:0`).
pub fn key(trainer: &str, device: &str) -> String {
    format!("train:{trainer}:{device}")
}

/// What a passed check ran on (see the module docs). Compared as a whole.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    /// SHA-256 of the environment's marker file and its identity.
    pub env_marker_sha256: String,
    pub env_marker: FileId,
    pub python: FileId,
    /// SHA-256 of the embedded `train_model.py` and `export_model.py`.
    pub train_script_sha256: String,
    pub export_script_sha256: String,
    pub trainer: String,
    pub device: String,
    /// [`Accelerator::describe`].
    pub accelerator: String,
    pub platform: String,
    pub build: String,
}

/// The fingerprint of a check of `trainer` on `accelerator` now; None when it cannot be told (no
/// environment marker or Python): then the check runs.
pub fn fingerprint(
    paths: &TrainPaths,
    platform: Platform,
    accelerator: &Accelerator,
    trainer: &str,
) -> Option<Fingerprint> {
    let marker = paths.export.env.join(crate::resources::export::ENV_MARKER);
    let bytes = std::fs::read(&marker).ok()?;
    Some(Fingerprint {
        env_marker_sha256: manager::sha256_hex(&bytes),
        env_marker: FileId::of(&marker)?,
        python: FileId::of(&paths.export.env_python(platform.os == "windows"))?,
        train_script_sha256: manager::sha256_hex(
            crate::resources::export::train::TRAIN_SCRIPT.as_bytes(),
        ),
        export_script_sha256: manager::sha256_hex(
            crate::resources::export::EXPORT_SCRIPT.as_bytes(),
        ),
        trainer: trainer.to_string(),
        device: accelerator.device_arg(),
        accelerator: accelerator.describe(),
        platform: format!("{}/{}", platform.os, platform.arch),
        build: crate::long_version(),
    })
}

/// One passed check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Passed {
    fingerprint: Fingerprint,
    /// When it passed, seconds since the Unix epoch.
    at: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
struct Marker {
    format: u32,
    #[serde(default)]
    checks: BTreeMap<String, Passed>,
}

fn unix_s(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn parse(bytes: &[u8]) -> Option<Marker> {
    serde_json::from_slice::<Marker>(bytes)
        .ok()
        .filter(|m| m.format == MARKER_FORMAT)
}

/// When the check `key` last passed with exactly `fp`, if that stands at `now` (see the module
/// docs); None = run the check. `marker` is the marker file's bytes (None: no file). Pure.
pub fn passed_at(
    marker: Option<&[u8]>,
    key: &str,
    fp: &Fingerprint,
    now: SystemTime,
) -> Option<u64> {
    let m = parse(marker?)?;
    let p = m.checks.get(key)?;
    if p.fingerprint != *fp {
        return None;
    }
    let now = unix_s(now);
    // Dated in the future (a clock set back): no telling how old it is.
    (p.at <= now && now - p.at <= MAX_AGE.as_secs()).then_some(p.at)
}

/// [`passed_at`] read from the marker file at `path`.
pub fn skip_since(path: &Path, key: &str, fp: &Fingerprint, now: SystemTime) -> Option<u64> {
    let bytes = std::fs::read(path).ok();
    passed_at(bytes.as_deref(), key, fp, now)
}

/// `marker` (the file's bytes, None: no file) with `key` recorded as passed with `fp` at `now`;
/// a damaged or other-format file starts over. Pure.
fn with_passed(marker: Option<&[u8]>, key: &str, fp: &Fingerprint, now: SystemTime) -> Marker {
    let mut m = marker.and_then(parse).unwrap_or(Marker {
        format: MARKER_FORMAT,
        checks: BTreeMap::new(),
    });
    m.checks.insert(
        key.to_string(),
        Passed {
            fingerprint: fp.clone(),
            at: unix_s(now),
        },
    );
    m
}

/// Record a passed check of `key` with `fp` in the marker file at `path` (atomically: a temporary
/// file and a rename). Only the training holding the training lock writes it.
pub fn record(path: &Path, key: &str, fp: &Fingerprint, now: SystemTime) -> Result<()> {
    let old = std::fs::read(path).ok();
    let m = with_passed(old.as_deref(), key, fp, now);
    let bytes = serde_json::to_vec_pretty(&m)?;
    crate::collect::layout::write_atomic(path, &bytes)
        .with_context(|| format!("writing {}", path.display()))
}

/// Drop `key` from the marker file at `path` (a training process failed after a skipped or
/// passed check: the next run checks again). A file that does not read is removed.
pub fn forget(path: &Path, key: &str) {
    let Ok(old) = std::fs::read(path) else {
        return;
    };
    let r = match parse(&old) {
        Some(mut m) => {
            if m.checks.remove(key).is_none() {
                return;
            }
            serde_json::to_vec_pretty(&m)
                .map_err(std::io::Error::other)
                .and_then(|b| crate::collect::layout::write_atomic(path, &b))
        }
        None => std::fs::remove_file(path),
    };
    if let Err(e) = r {
        tracing::warn!(
            "could not update {} ({e}); removing it so the next run checks again",
            path.display()
        );
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn fp() -> Fingerprint {
        Fingerprint {
            env_marker_sha256: "aa".into(),
            env_marker: FileId {
                len: 10,
                mtime_ns: 1,
            },
            python: FileId {
                len: 20,
                mtime_ns: 2,
            },
            train_script_sha256: "bb".into(),
            export_script_sha256: "cc".into(),
            trainer: "ultralytics".into(),
            device: "cuda:0".into(),
            accelerator: "GTX".into(),
            platform: "windows/x86_64".into(),
            build: "0.1".into(),
        }
    }

    fn at(s: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(s)
    }

    #[test]
    fn an_unchanged_fresh_check_is_skipped_and_any_doubt_runs_it() {
        let k = key("ultralytics", "cuda:0");
        let t0 = 1_000_000;
        let file = serde_json::to_vec(&with_passed(None, &k, &fp(), at(t0))).unwrap();
        let skip =
            |bytes: Option<&[u8]>, f: &Fingerprint, now: u64| passed_at(bytes, &k, f, at(now));
        assert_eq!(
            skip(Some(&file), &fp(), t0 + 60),
            Some(t0),
            "unchanged: skipped"
        );
        // A changed script hash, environment marker or Python: run.
        let mut f = fp();
        f.train_script_sha256 = "other".into();
        assert_eq!(skip(Some(&file), &f, t0 + 60), None, "script changed");
        let mut f = fp();
        f.export_script_sha256 = "other".into();
        assert_eq!(
            skip(Some(&file), &f, t0 + 60),
            None,
            "export script changed"
        );
        let mut f = fp();
        f.env_marker_sha256 = "other".into();
        assert_eq!(
            skip(Some(&file), &f, t0 + 60),
            None,
            "env marker bytes changed"
        );
        let mut f = fp();
        f.env_marker.mtime_ns = 9;
        assert_eq!(
            skip(Some(&file), &f, t0 + 60),
            None,
            "env rebuilt (marker rewritten)"
        );
        let mut f = fp();
        f.python.len = 21;
        assert_eq!(skip(Some(&file), &f, t0 + 60), None, "python replaced");
        let mut f = fp();
        f.accelerator = "RTX".into();
        assert_eq!(skip(Some(&file), &f, t0 + 60), None, "another GPU");
        let mut f = fp();
        f.build = "0.2".into();
        assert_eq!(skip(Some(&file), &f, t0 + 60), None, "another build");
        // Missing, damaged or other-format marker: run.
        assert_eq!(skip(None, &fp(), t0 + 60), None, "no marker");
        assert_eq!(skip(Some(b"{not json"), &fp(), t0 + 60), None, "corrupt");
        assert_eq!(skip(Some(b""), &fp(), t0 + 60), None, "empty");
        let mut newer = with_passed(None, &k, &fp(), at(t0));
        newer.format = MARKER_FORMAT + 1;
        let newer = serde_json::to_vec(&newer).unwrap();
        assert_eq!(skip(Some(&newer), &fp(), t0 + 60), None, "another format");
        // Another trainer or device has its own entry.
        assert_eq!(
            passed_at(Some(&file), &key("rfdetr", "cuda:0"), &fp(), at(t0 + 60)),
            None
        );
        assert_eq!(
            passed_at(
                Some(&file),
                &key("ultralytics", "cuda:1"),
                &fp(),
                at(t0 + 60)
            ),
            None
        );
        // Too old, or dated in the future: run.
        let max = MAX_AGE.as_secs();
        assert_eq!(skip(Some(&file), &fp(), t0 + max), Some(t0), "at the limit");
        assert_eq!(skip(Some(&file), &fp(), t0 + max + 1), None, "too old");
        assert_eq!(skip(Some(&file), &fp(), t0 - 1), None, "in the future");
    }

    #[test]
    fn record_and_forget_keep_the_other_entries_and_repair_a_damaged_file() {
        let d = std::env::temp_dir().join(format!("bop-self-check-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join(MARKER_FILE);
        let (a, b) = (key("ultralytics", "cuda:0"), key("rfdetr", "cuda:0"));
        let now = SystemTime::now();
        // A damaged file is replaced by the first record.
        std::fs::write(&p, b"garbage").unwrap();
        record(&p, &a, &fp(), now).unwrap();
        record(&p, &b, &fp(), now).unwrap();
        assert!(skip_since(&p, &a, &fp(), now).is_some());
        assert!(skip_since(&p, &b, &fp(), now).is_some());
        forget(&p, &a);
        assert_eq!(skip_since(&p, &a, &fp(), now), None, "forgotten");
        assert!(skip_since(&p, &b, &fp(), now).is_some(), "the other stays");
        // Forgetting in a damaged file removes it.
        std::fs::write(&p, b"garbage").unwrap();
        forget(&p, &b);
        assert!(!p.exists());
        forget(&p, &b);
        // No temporary files left.
        let left: Vec<_> = std::fs::read_dir(&d).unwrap().flatten().collect();
        assert!(left.is_empty(), "{left:?}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
