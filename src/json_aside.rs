//! Setting a damaged JSON store aside: the one policy every store of the data root follows when
//! its file cannot be used (audit 2026-11, phase 7: QA-126). Training's `runs.json` /
//! `tasks.json` ([`crate::training::task`], tag `corrupt`) and the benchmark's `benchmark.json`,
//! `benchmark-history.json` and `benchmark-preds.json` ([`crate::benchmark::report`], tag `bad`)
//! use these primitives; each keeps its own contract around them (training never fails a load,
//! the benchmark refuses the save).
//!
//! The policy:
//!
//! - **A file that cannot be read stays where it is.** A failed read says nothing about the
//!   content: antivirus or a backup can hold a file for a moment on Windows, and a file another
//!   account wrote can be unreadable for this one. [`read_retrying`] / [`open_retrying`] try
//!   [`READ_TRIES`] times [`READ_RETRY`] apart; "not found" is `None` at once. The caller then
//!   writes nothing over the file.
//! - **A file that was read and does not parse is moved aside** ([`move_or_copy_aside`]) to
//!   `<name>.<tag>-<YYYYmmdd-HHMMSS>` ([`aside_path`]; `-1`, `-2`, ... only when that name is
//!   taken, so an aside never replaces an earlier one). When the rename fails (Windows: another
//!   process has the file open without delete sharing) a copy will do, the original stays and
//!   the next save replaces it; a copy is skipped when an earlier aside of the same tag already
//!   holds the same bytes ([`Aside::AlreadyCopied`]), so a file held for a while does not leave a
//!   new copy on every load.
//! - **What a load drops is kept before a save can lose it:** [`keep_copy`] (bytes the caller
//!   chose, e.g. the records this build cannot read) and [`copy_aside`] (the whole file, streamed),
//!   both deduplicated by content against the existing asides of the tag and written through a
//!   temporary file of their own ([`crate::fsx`]). A caller that cannot keep the copy saves
//!   nothing.
//!
//! Asides are user data: nothing deletes them (the storage sweep matches neither tag); the logs
//! tell the user to restore one by hand. Std + [`crate::fsx`] + [`crate::clock`] (for the
//! stamp in the name) only.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Reads (or opens) of a store before it counts as unreadable.
pub const READ_TRIES: u32 = 3;
/// Wait between those tries.
pub const READ_RETRY: Duration = Duration::from_millis(50);

/// `f` tried [`READ_TRIES`] times [`READ_RETRY`] apart while it fails with anything but "not
/// found"; "not found" is `Ok(None)`.
fn retrying<T>(mut f: impl FnMut() -> std::io::Result<T>) -> std::io::Result<Option<T>> {
    let mut tries = 1;
    loop {
        match f() {
            Ok(v) => return Ok(Some(v)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) if tries < READ_TRIES => {
                tries += 1;
                std::thread::sleep(READ_RETRY);
            }
            Err(e) => return Err(e),
        }
    }
}

/// The bytes of `path`; `None` when it does not exist. Any other failure is tried again
/// ([`READ_TRIES`]) before it is returned: the file is then unreadable, not damaged.
pub fn read_retrying(path: &Path) -> std::io::Result<Option<Vec<u8>>> {
    retrying(|| std::fs::read(path))
}

/// `path` opened for reading (for a store parsed as a stream); `None` when it does not exist.
/// Retried as [`read_retrying`]. A folder in the file's place is an error here on every system
/// (Unix opens one, and only the first read would fail).
pub fn open_retrying(path: &Path) -> std::io::Result<Option<std::fs::File>> {
    retrying(|| {
        let f = std::fs::File::open(path)?;
        if f.metadata()?.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::IsADirectory,
                "a folder is in the file's place",
            ));
        }
        Ok(f)
    })
}

/// `<name>.<tag>-` : the start of every aside name of `path` with `tag`.
fn aside_prefix(path: &Path, tag: &str) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!("{name}.{tag}-")
}

/// `<name>.<tag>-<YYYYmmdd-HHMMSS>` (wall time of the configured zone) beside `path`, not in use
/// yet: `-1`, `-2`, ... is added only when the plain name is taken (two asides in one second,
/// or the hour a DST fall-back repeats).
pub fn aside_path(path: &Path, tag: &str) -> PathBuf {
    let base = format!(
        "{}{}",
        aside_prefix(path, tag),
        crate::clock::name_stamp(std::time::SystemTime::now())
    );
    (0u64..)
        .map(|i| match i {
            0 => path.with_file_name(&base),
            _ => path.with_file_name(format!("{base}-{i}")),
        })
        .find(|p| std::fs::symlink_metadata(p).is_err())
        .unwrap_or_else(|| path.with_file_name(base))
}

/// The existing asides of `path` with `tag` (regular files only).
fn asides(path: &Path, tag: &str) -> Vec<PathBuf> {
    let prefix = aside_prefix(path, tag);
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    rd.flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix))
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .map(|e| e.path())
        .collect()
}

/// Whether the files `a` and `b` hold the same bytes (streamed; any read error is "no").
fn same_file_content(a: &Path, b: &Path) -> bool {
    let (Ok(ma), Ok(mb)) = (std::fs::metadata(a), std::fs::metadata(b)) else {
        return false;
    };
    if ma.len() != mb.len() {
        return false;
    }
    let (Ok(fa), Ok(fb)) = (std::fs::File::open(a), std::fs::File::open(b)) else {
        return false;
    };
    let (mut ra, mut rb) = (std::io::BufReader::new(fa), std::io::BufReader::new(fb));
    let mut ba = [0u8; 8192];
    let mut bb = [0u8; 8192];
    loop {
        let Ok(n) = ra.read(&mut ba) else {
            return false;
        };
        if n == 0 {
            // Same length, all of `a` matched: `b` must be at its end too.
            return matches!(rb.read(&mut bb[..1]), Ok(0));
        }
        let (Some(want), Some(got)) = (ba.get(..n), bb.get_mut(..n)) else {
            return false;
        };
        if rb.read_exact(got).is_err() || want != got {
            return false;
        }
    }
}

/// Whether `file` holds exactly `bytes` (length first, so a large aside is read only when it
/// could match).
fn holds(file: &Path, bytes: &[u8]) -> bool {
    std::fs::metadata(file).is_ok_and(|m| u64::try_from(bytes.len()).is_ok_and(|n| n == m.len()))
        && std::fs::read(file).is_ok_and(|b| b == bytes)
}

/// Keep `bytes` (what a load of `path` could not use) in an aside of `tag`, unless one already
/// holds exactly these bytes. Returns the aside and whether it was written now. Written through
/// a temporary file of its own ([`crate::fsx::write_atomic`]): a crash leaves no half copy.
pub fn keep_copy(path: &Path, bytes: &[u8], tag: &str) -> std::io::Result<(PathBuf, bool)> {
    if let Some(found) = asides(path, tag).into_iter().find(|a| holds(a, bytes)) {
        return Ok((found, false));
    }
    let aside = aside_path(path, tag);
    crate::fsx::write_atomic(&aside, bytes)?;
    Ok((aside, true))
}

/// Copy the whole file `path` to an aside of `tag`, streamed, unless an aside already holds the
/// same bytes. Returns the aside and whether it was written now. The copy goes to a temporary
/// file of its own first ([`crate::fsx::temp_path`]), synced, then renamed to its free name.
pub fn copy_aside(path: &Path, tag: &str) -> std::io::Result<(PathBuf, bool)> {
    if let Some(found) = asides(path, tag)
        .into_iter()
        .find(|a| same_file_content(path, a))
    {
        return Ok((found, false));
    }
    let aside = aside_path(path, tag);
    let tmp = crate::fsx::temp_path(&aside);
    let copied = std::fs::copy(path, &tmp)
        .and_then(|_| std::fs::OpenOptions::new().write(true).open(&tmp))
        .and_then(|f| f.sync_all())
        .and_then(|()| {
            crate::fsx::rename_retrying(
                &tmp,
                &aside,
                crate::fsx::RENAME_ATTEMPTS,
                crate::fsx::RENAME_PAUSE,
            )
        });
    if let Err(e) = copied {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok((aside, true))
}

/// How [`move_or_copy_aside`] set a file aside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aside {
    /// Renamed: the file is no longer at its path.
    Moved,
    /// The rename failed and a copy was written: the original stays (the next save replaces it).
    Copied,
    /// The rename failed and an earlier aside already holds the same bytes: nothing written,
    /// the original stays.
    AlreadyCopied,
}

impl Aside {
    /// "moved to" / "copied to" / "already copied to", for a log line naming the aside.
    pub fn verb(self) -> &'static str {
        match self {
            Aside::Moved => "moved to",
            Aside::Copied => "copied to",
            Aside::AlreadyCopied => "already copied to",
        }
    }
}

/// Set the damaged file `path` aside: renamed to [`aside_path`], else copied ([`copy_aside`],
/// deduplicated). Errors when neither works (both errors in the message); the file is then
/// still only at its path and must not be written over.
pub fn move_or_copy_aside(path: &Path, tag: &str) -> std::io::Result<(PathBuf, Aside)> {
    move_or_copy_aside_with(path, tag, &|from, to| std::fs::rename(from, to))
}

/// [`move_or_copy_aside`] with the rename step given (tests make it fail, as Windows does for a
/// file another process holds).
pub fn move_or_copy_aside_with(
    path: &Path,
    tag: &str,
    rename: &dyn Fn(&Path, &Path) -> std::io::Result<()>,
) -> std::io::Result<(PathBuf, Aside)> {
    let aside = aside_path(path, tag);
    let moved = match rename(path, &aside) {
        Ok(()) => return Ok((aside, Aside::Moved)),
        Err(e) => e,
    };
    match copy_aside(path, tag) {
        Ok((p, true)) => Ok((p, Aside::Copied)),
        Ok((p, false)) => Ok((p, Aside::AlreadyCopied)),
        Err(c) => Err(std::io::Error::new(
            moved.kind(),
            format!(
                "renaming it to {} failed ({moved}), and copying it there failed too ({c})",
                aside.display()
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_dirs::TempDir;

    fn listed(path: &Path, tag: &str) -> Vec<PathBuf> {
        let mut v = asides(path, tag);
        v.sort();
        v
    }

    #[test]
    fn missing_is_none_and_a_folder_is_unreadable() {
        let dir = TempDir::new("json-aside-read");
        let p = dir.join("store.json");
        assert!(read_retrying(&p).unwrap().is_none());
        assert!(open_retrying(&p).unwrap().is_none());
        std::fs::write(&p, b"[1]").unwrap();
        assert_eq!(read_retrying(&p).unwrap().unwrap(), b"[1]");
        let mut s = String::new();
        open_retrying(&p)
            .unwrap()
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        assert_eq!(s, "[1]");
        std::fs::remove_file(&p).unwrap();
        std::fs::create_dir(&p).unwrap();
        assert!(read_retrying(&p).is_err());
        assert!(open_retrying(&p).is_err());
    }

    /// QA-057: two asides in the same second get two names; the first keeps the plain one.
    #[test]
    fn a_second_aside_in_the_same_second_gets_its_own_name() {
        let dir = TempDir::new("json-aside-names");
        let p = dir.join("benchmark.json");
        std::fs::write(&p, b"one").unwrap();
        let (a, how) = move_or_copy_aside(&p, "bad").unwrap();
        assert_eq!(how, Aside::Moved);
        std::fs::write(&p, b"two").unwrap();
        let (b, _) = move_or_copy_aside(&p, "bad").unwrap();
        assert_ne!(a, b);
        assert_eq!(std::fs::read(&a).unwrap(), b"one");
        assert_eq!(std::fs::read(&b).unwrap(), b"two");
        let plain = a.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(plain.len(), "benchmark.json.bad-".len() + 15, "{plain}");
    }

    /// QA-058 / QA-187: a rename that always fails leaves the original, writes one copy, and a
    /// second try adds none.
    #[test]
    fn a_failed_rename_copies_once() {
        let dir = TempDir::new("json-aside-copy");
        let p = dir.join("tasks.json");
        std::fs::write(&p, b"damaged").unwrap();
        let refuse = |_: &Path, _: &Path| -> std::io::Result<()> {
            Err(std::io::Error::other("held by another process"))
        };
        let (a, how) = move_or_copy_aside_with(&p, "corrupt", &refuse).unwrap();
        assert_eq!(how, Aside::Copied);
        assert_eq!(how.verb(), "copied to");
        assert_eq!(std::fs::read(&p).unwrap(), b"damaged", "the original stays");
        assert_eq!(std::fs::read(&a).unwrap(), b"damaged");
        let (b, how) = move_or_copy_aside_with(&p, "corrupt", &refuse).unwrap();
        assert_eq!((b.clone(), how), (a.clone(), Aside::AlreadyCopied));
        assert_eq!(listed(&p, "corrupt"), [a]);
        // Other bytes are another copy.
        std::fs::write(&p, b"damaged again").unwrap();
        let (c, how) = move_or_copy_aside_with(&p, "corrupt", &refuse).unwrap();
        assert_eq!(how, Aside::Copied);
        assert_eq!(std::fs::read(&c).unwrap(), b"damaged again");
        assert_eq!(listed(&p, "corrupt").len(), 2);
        // No temporary file is left.
        assert!(
            std::fs::read_dir(dir)
                .unwrap()
                .flatten()
                .all(|e| !crate::fsx::is_temp_name(&e.file_name().to_string_lossy()))
        );
    }

    #[test]
    fn keep_copy_and_copy_aside_dedupe_by_content_per_tag() {
        let dir = TempDir::new("json-aside-keep");
        let p = dir.join("runs.json");
        std::fs::write(&p, b"whole file").unwrap();
        let (a, new) = keep_copy(&p, b"[{\"id\":\"x\"}]", "corrupt").unwrap();
        assert!(new);
        let (b, new) = keep_copy(&p, b"[{\"id\":\"x\"}]", "corrupt").unwrap();
        assert!(!new);
        assert_eq!(a, b);
        // The whole file is other bytes: a copy of its own; then deduplicated.
        let (c, new) = copy_aside(&p, "corrupt").unwrap();
        assert!(new);
        assert_eq!(std::fs::read(&c).unwrap(), b"whole file");
        assert!(!copy_aside(&p, "corrupt").unwrap().1);
        // Another tag never matches this one's asides.
        assert!(copy_aside(&p, "bad").unwrap().1);
        assert_eq!(listed(&p, "corrupt").len(), 2);
        assert_eq!(listed(&p, "bad").len(), 1);
        assert!(p.is_file(), "copying never moves the file");
    }

    #[test]
    fn same_content_compares_every_byte() {
        let dir = TempDir::new("json-aside-same");
        let a = dir.join("a");
        let b = dir.join("b");
        let big: Vec<u8> = (0..20_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&a, &big).unwrap();
        std::fs::write(&b, &big).unwrap();
        assert!(same_file_content(&a, &b));
        let mut other = big.clone();
        *other.last_mut().unwrap() ^= 1;
        std::fs::write(&b, &other).unwrap();
        assert!(!same_file_content(&a, &b));
        std::fs::write(&b, &big[..100]).unwrap();
        assert!(!same_file_content(&a, &b));
        assert!(!same_file_content(&a, &dir.join("missing")));
    }
}
