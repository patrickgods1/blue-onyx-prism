//! A file's identity as the reuse checks compare it: its size and last-write time, read through
//! the file itself ([`std::fs::metadata`], never a directory listing: on NTFS a hard link's
//! directory entry can carry a stale size and time). A change of either means the bytes may have
//! changed; equal values are taken as "unchanged" by the callers that skip work on them (the
//! training self-check, a seed's dataset links, the baseline of a comparison group), which all
//! fall back to doing the work when the identity cannot be read.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Size and last-write time (nanoseconds since the Unix epoch) of a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileId {
    pub len: u64,
    pub mtime_ns: u64,
}

impl FileId {
    /// The identity of the file at `path`; None when it is not a file or its time cannot be read
    /// (a file system without one): the caller then cannot tell, and redoes the work.
    pub fn of(path: &Path) -> Option<Self> {
        let m = std::fs::metadata(path)
            .ok()
            .filter(std::fs::Metadata::is_file)?;
        Some(Self {
            len: m.len(),
            mtime_ns: unix_ns(m.modified().ok()?)?,
        })
    }
}

/// `t` in nanoseconds since the Unix epoch; None before it or past `u64` (the year 2554).
pub fn unix_ns(t: SystemTime) -> Option<u64> {
    u64::try_from(t.duration_since(UNIX_EPOCH).ok()?.as_nanos()).ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn a_file_has_an_identity_that_follows_its_bytes() {
        let d = std::env::temp_dir().join(format!("bop-file-id-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        let f = d.join("a.bin");
        std::fs::write(&f, b"abc").unwrap();
        let a = FileId::of(&f).unwrap();
        assert_eq!(a.len, 3);
        assert_eq!(FileId::of(&f), Some(a), "stable while untouched");
        std::fs::write(&f, b"abcd").unwrap();
        assert_ne!(FileId::of(&f), Some(a), "a write changes it");
        assert_eq!(FileId::of(&d), None, "a folder is no file");
        assert_eq!(FileId::of(&d.join("gone")), None);
        let _ = std::fs::remove_dir_all(&d);
    }
}
