//! Test-only scratch folders: [`TempDir`] is a fresh folder under the OS temp dir that is removed
//! on drop, also when the test panics (keep them with `BOP_KEEP_TEST_DIRS=1`, as `tests/common`'s
//! `temp_dir` does), so no test leaves a folder behind.

use std::path::{Path, PathBuf};

/// A scratch folder removed when dropped.
#[derive(Debug)]
pub(crate) struct TempDir(PathBuf);

impl TempDir {
    /// A new, empty folder `bop-<tag>-<uuid>`.
    #[allow(clippy::unwrap_used, reason = "test helper")]
    pub(crate) fn new(tag: &str) -> Self {
        let dir = Self::unmade(tag);
        std::fs::create_dir_all(&dir.0).unwrap();
        dir
    }

    /// The same, but the folder is not created (for a test that checks nothing exists yet);
    /// whatever the test creates there is removed on drop.
    pub(crate) fn unmade(tag: &str) -> Self {
        Self(std::env::temp_dir().join(format!("bop-{tag}-{}", uuid::Uuid::new_v4())))
    }
}

impl std::ops::Deref for TempDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if std::env::var_os("BOP_KEEP_TEST_DIRS").is_none() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dropping removes the folder and what is in it.
    #[test]
    fn drop_removes_the_folder() {
        let path = {
            let d = TempDir::new("guard");
            std::fs::write(d.join("f"), b"x").unwrap();
            assert!(d.is_dir());
            d.to_path_buf()
        };
        assert!(!path.exists());
    }
}
