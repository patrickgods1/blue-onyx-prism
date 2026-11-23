//! File-system helpers shared by every writer of the data root (audit 2026-11, Phase 0):
//! the one atomic write, the one file-lock wait and the one rule for on-disk names built from
//! user input. Std + fs4 + uuid only (a leaf: nothing else of the crate).
//!
//! - [`write_atomic`] / [`write_atomic_with`] write `<dir>/.<name>.<uuid>.tmp` (the shape
//!   [`is_temp_name`] recognises, so a sweep can remove one a crash left; [`temp_path`] makes
//!   it), opened with `create_new` (two writers never share a temporary file), synced
//!   (`sync_all`), then renamed over the target, so a reader sees the old file or the new one,
//!   never a mix. On Unix the target's mode (permission bits) is copied onto the temporary file
//!   before a byte is written, so a `chmod 0640` an installer set survives the save, and the
//!   folder is synced after the rename (best effort: the rename itself is then on the disk). On
//!   Windows nothing is copied: the file's DACL is inherited from its folder (the temporary file
//!   is made in the target's folder, so it gets what a new file there gets, e.g. `auth/`'s
//!   protected DACL), and the read-only attribute cannot carry over because a rename onto a
//!   read-only file fails (as it did before). [`Options::private`] instead makes the temporary
//!   file 0600 from its creation (secrets). Any error removes the temporary file.
//! - [`rename_retrying`] (and [`Options::rename_attempts`]) retries a rename while Windows
//!   reports a sharing violation (an antivirus or the indexer holding the file for a moment).
//! - [`lock_wait`] / [`lock_wait_with`] take an fs4 lock, polling until a deadline (fixed or
//!   doubling pauses, [`Pauses`]), optionally stopped early by the caller (a cancel flag).
//!   Callers keep their own lock-file names, open flags, error texts and policies.
//! - [`is_windows_reserved`] tells a Windows reserved device name (`CON`, `NUL`, `COM1`, ...),
//!   which opens the device instead of a file when used as a file or folder name.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The temporary file [`write_atomic`] writes beside `path`: `<dir>/.<name>.<uuid>.tmp` (`uuid`
/// as 32 lowercase hex digits). Unique per call.
pub fn temp_path(path: &Path) -> PathBuf {
    let dir = parent_of(path);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".into());
    dir.join(format!(".{name}.{}.tmp", uuid::Uuid::new_v4().simple()))
}

/// Whether `file_name` has exactly the shape [`temp_path`] makes: `.`, a non-empty name, `.`,
/// 32 hex digits, `.tmp`.
pub fn is_temp_name(file_name: &str) -> bool {
    let Some(rest) = file_name.strip_prefix('.') else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".tmp") else {
        return false;
    };
    match rest.rsplit_once('.') {
        Some((name, id)) => {
            !name.is_empty() && id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit())
        }
        None => false,
    }
}

/// The folder of `path` (`.` for a bare name).
fn parent_of(path: &Path) -> &Path {
    match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    }
}

/// [`Options::before_rename`]: a step on the complete temporary file (its path).
pub type Prepare<'a> = dyn Fn(&Path) -> std::io::Result<()> + 'a;

/// How [`write_atomic_with`] writes. `Options::default()` is [`write_atomic`].
#[derive(Clone, Copy, Default)]
pub struct Options<'a> {
    /// Create the target's folder (and its parents) first when it is missing.
    pub create_parent: bool,
    /// A secret: the temporary file is created 0600 on Unix (never readable by others, not even
    /// for a moment), and the target's mode is not copied. On Windows the folder's DACL protects
    /// it (as for every file).
    pub private: bool,
    /// How often the rename is tried while Windows reports a sharing violation, `rename_pause`
    /// apart ([`rename_retrying`]); 0 or 1 = once.
    pub rename_attempts: u32,
    /// Pause between rename attempts.
    pub rename_pause: Duration,
    /// Run on the complete, synced temporary file just before the rename (e.g. hand it to the
    /// folder's owner when running as root). An error stops the write.
    pub before_rename: Option<&'a Prepare<'a>>,
}

impl std::fmt::Debug for Options<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Options")
            .field("create_parent", &self.create_parent)
            .field("private", &self.private)
            .field("rename_attempts", &self.rename_attempts)
            .field("rename_pause", &self.rename_pause)
            .field("before_rename", &self.before_rename.is_some())
            .finish()
    }
}

/// Which part of [`write_atomic_with`] failed, so callers keep their own error texts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Creating the target's folder ([`Options::create_parent`]).
    CreateParent,
    /// Creating the temporary file (or setting its permissions).
    Create,
    /// Writing or syncing the temporary file.
    Write,
    /// [`Options::before_rename`].
    Prepare,
    /// Renaming the temporary file over the target.
    Rename,
}

/// A failed [`write_atomic_with`]: the step, the temporary file it used (already removed) and the
/// OS error. Displays as the OS error alone; callers add what they were writing.
#[derive(Debug)]
pub struct WriteError {
    pub step: Step,
    pub temp: PathBuf,
    pub source: std::io::Error,
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.source.fmt(f)
    }
}

impl std::error::Error for WriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.source()
    }
}

impl From<WriteError> for std::io::Error {
    fn from(e: WriteError) -> Self {
        e.source
    }
}

/// Write `bytes` to `path` through a synced temporary file in the same folder and a rename, so a
/// reader never sees a half-written file (see the module docs). The folder must exist.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    write_atomic_with(path, bytes, &Options::default()).map_err(std::io::Error::from)
}

/// [`write_atomic`] with [`Options`].
pub fn write_atomic_with(path: &Path, bytes: &[u8], opts: &Options<'_>) -> Result<(), WriteError> {
    let tmp = temp_path(path);
    let fail = |step: Step, source: std::io::Error| WriteError {
        step,
        temp: tmp.clone(),
        source,
    };
    if opts.create_parent {
        std::fs::create_dir_all(parent_of(path)).map_err(|e| fail(Step::CreateParent, e))?;
    }
    let mut open = std::fs::OpenOptions::new();
    open.write(true).create_new(true);
    #[cfg(unix)]
    if opts.private {
        use std::os::unix::fs::OpenOptionsExt;
        open.mode(0o600);
    }
    let mut f = open.open(&tmp).map_err(|e| fail(Step::Create, e))?;
    // From here on the temporary file exists: every failure removes it.
    let result = (|| {
        if !opts.private {
            keep_permissions(&f, path).map_err(|e| (Step::Create, e))?;
        }
        f.write_all(bytes)
            .and_then(|()| f.sync_all())
            .map_err(|e| (Step::Write, e))?;
        drop(f);
        if let Some(prepare) = opts.before_rename {
            prepare(&tmp).map_err(|e| (Step::Prepare, e))?;
        }
        rename_retrying(&tmp, path, opts.rename_attempts, opts.rename_pause)
            .map_err(|e| (Step::Rename, e))
    })();
    if let Err((step, e)) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(fail(step, e));
    }
    sync_dir(parent_of(path));
    Ok(())
}

/// Give the new file `f` the permission bits of the existing regular file `target` (Unix: the
/// mode's rwx bits, never setuid/setgid/sticky). Nothing when there is no target, and on
/// Windows (see the module docs).
#[cfg(unix)]
fn keep_permissions(f: &std::fs::File, target: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(target) {
        Ok(m) if m.is_file() => f.set_permissions(std::fs::Permissions::from_mode(
            m.permissions().mode() & 0o777,
        )),
        _ => Ok(()),
    }
}

#[cfg(not(unix))]
fn keep_permissions(_f: &std::fs::File, _target: &Path) -> std::io::Result<()> {
    Ok(())
}

/// Sync the folder `dir`, so a rename in it is on the disk (Unix; best effort: an error only
/// means the rename may need the next sync). Windows has no folder sync through std.
#[cfg(unix)]
fn sync_dir(dir: &Path) {
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) {}

/// [`Options::rename_attempts`] of writers that ride out a Windows sharing violation:
/// [`RENAME_ATTEMPTS`] x [`RENAME_PAUSE`], one second at most.
pub const RENAME_ATTEMPTS: u32 = 20;
/// Pause between those attempts.
pub const RENAME_PAUSE: Duration = Duration::from_millis(50);

/// `rename(from, to)`, tried `attempts` times `pause` apart while the error is a Windows sharing
/// violation ([`is_sharing_violation`]); any other error at once. Works for folders too: a
/// folder whose new files a virus scanner or the search indexer still holds open is refused with
/// ERROR_ACCESS_DENIED for a moment after they were written.
pub fn rename_retrying(
    from: &Path,
    to: &Path,
    attempts: u32,
    pause: Duration,
) -> std::io::Result<()> {
    let mut left = attempts.max(1);
    loop {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) if left > 1 && is_sharing_violation(&e) => {
                left -= 1;
                std::thread::sleep(pause);
            }
            Err(e) => return Err(e),
        }
    }
}

/// Whether `e` is Windows saying another process has the file open: ERROR_ACCESS_DENIED (5, a
/// file open without delete sharing), ERROR_SHARING_VIOLATION (32), ERROR_LOCK_VIOLATION (33).
/// Never on other systems, where a rename over an open file works.
pub fn is_sharing_violation(e: &std::io::Error) -> bool {
    cfg!(windows) && matches!(e.raw_os_error(), Some(5 | 32 | 33))
}

// ---------------------------------------------------------------------------------------------
// File locks
// ---------------------------------------------------------------------------------------------

/// Which lock [`lock_wait`] takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockKind {
    /// `try_lock_shared`.
    Shared,
    /// `try_lock` (exclusive).
    Exclusive,
}

/// The pauses between lock attempts: `first`, doubling up to `max` (equal = a fixed pause).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pauses {
    next: Duration,
    max: Duration,
}

impl Pauses {
    /// The same pause every time.
    pub fn fixed(pause: Duration) -> Self {
        Self {
            next: pause,
            max: pause,
        }
    }

    /// `first`, then twice the last one, never more than `max`.
    pub fn doubling(first: Duration, max: Duration) -> Self {
        Self {
            next: first.min(max),
            max,
        }
    }

    /// The pause to take now.
    fn take(&mut self) -> Duration {
        let now = self.next;
        self.next = now.saturating_mul(2).min(self.max);
        now
    }
}

/// How [`lock_wait_with`] ended (an OS error that makes the lock unusable is its `Err`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waited {
    /// The lock is held by `file` now.
    Held,
    /// Someone else still held it when the deadline passed.
    TimedOut,
    /// The caller's `keep_waiting` said stop.
    Stopped,
}

/// The deadline `wait` from now; None (no deadline) when that is past what `Instant` holds.
pub fn deadline_after(wait: Duration) -> Option<Instant> {
    Instant::now().checked_add(wait)
}

/// Take an fs4 lock on `file`, retrying every `poll` until `deadline` passes. `Ok(true)` = held,
/// `Ok(false)` = still held by someone else at the deadline, `Err` = the lock call cannot work
/// on this file (callers decide their fallback).
pub fn lock_wait(
    file: &std::fs::File,
    kind: LockKind,
    deadline: Instant,
    poll: Duration,
) -> std::io::Result<bool> {
    let waited = lock_wait_with(file, kind, Some(deadline), Pauses::fixed(poll), || true)?;
    Ok(waited == Waited::Held)
}

/// Take an fs4 lock on `file`. While another holder has it: `keep_waiting()` is asked first
/// (false = [`Waited::Stopped`]; the place for a cancel flag or a one-time "waiting" message),
/// then a passed `deadline` (None = none) gives [`Waited::TimedOut`], else it sleeps the next of
/// `pauses` and tries again. The first try is immediate. `Err`: the lock call failed with
/// something other than "held by another" (callers decide their fallback).
pub fn lock_wait_with(
    file: &std::fs::File,
    kind: LockKind,
    deadline: Option<Instant>,
    mut pauses: Pauses,
    mut keep_waiting: impl FnMut() -> bool,
) -> std::io::Result<Waited> {
    loop {
        let r = match kind {
            LockKind::Shared => fs4::FileExt::try_lock_shared(file),
            LockKind::Exclusive => fs4::FileExt::try_lock(file),
        };
        match r {
            Ok(()) => return Ok(Waited::Held),
            Err(fs4::TryLockError::WouldBlock) => {}
            Err(fs4::TryLockError::Error(e)) => return Err(e),
        }
        if !keep_waiting() {
            return Ok(Waited::Stopped);
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Ok(Waited::TimedOut);
        }
        std::thread::sleep(pauses.take());
    }
}

// ---------------------------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------------------------

/// Windows reserved device names: `CON.jpg` (any extension, any case) opens the device. `stem`
/// is a file or folder name; only the part before its first `.` counts (`CON`, `PRN`, `AUX`,
/// `NUL`, `COM0`-`COM9`, `LPT0`-`LPT9`).
pub fn is_windows_reserved(stem: &str) -> bool {
    let base = stem.split('.').next().unwrap_or(stem).to_ascii_uppercase();
    match base.as_str() {
        "CON" | "PRN" | "AUX" | "NUL" => true,
        _ => {
            let b = base.as_bytes();
            b.len() == 4
                && (base.starts_with("COM") || base.starts_with("LPT"))
                && b.get(3).is_some_and(u8::is_ascii_digit)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch() -> Scratch {
        let d = std::env::temp_dir().join(format!("bop_fsx_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut n: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        n.sort();
        n
    }

    #[test]
    fn temp_names_round_trip() {
        for target in ["c.json", "runs.json", "a.b.c", ".hidden", "x"] {
            let t = temp_path(&Path::new("/some/dir").join(target));
            assert_eq!(t.parent(), Some(Path::new("/some/dir")));
            let name = t.file_name().unwrap().to_str().unwrap();
            assert!(is_temp_name(name), "{name}");
            assert!(name.starts_with(&format!(".{target}.")), "{name}");
        }
        assert_ne!(temp_path(Path::new("a")), temp_path(Path::new("a")));
        assert_eq!(temp_path(Path::new("bare")).parent(), Some(Path::new(".")));
        let id = "0123456789abcdef0123456789ABCDEF";
        assert!(is_temp_name(&format!(".c.json.{id}.tmp")));
        for other in [
            "c.json.tmp",
            ".c.json.tmp",
            &format!("..{id}.tmp"),
            &format!(".c.{id}"),
            &format!("c.{id}.tmp"),
            ".c.0123.tmp",
            &format!(".c.{}.tmp", "g".repeat(32)),
            &format!(".c.{id}0.tmp"),
            ".token.12345678-1234-1234-1234-123456789abc.tmp",
            ".c.json.123.4.tmp",
            ".c.tmp-1-2",
        ] {
            assert!(!is_temp_name(other), "{other}");
        }
    }

    #[test]
    fn writes_and_leaves_no_temp_file() {
        let s = scratch();
        let p = s.0.join("a.json");
        write_atomic(&p, b"one").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"one");
        write_atomic(&p, b"two, longer").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"two, longer");
        assert_eq!(names(&s.0), ["a.json"]);
    }

    #[test]
    fn a_failed_rename_leaves_no_temp_file() {
        let s = scratch();
        // The target is a non-empty folder: the rename fails.
        let target = s.0.join("sub");
        std::fs::create_dir_all(target.join("x")).unwrap();
        let e = write_atomic_with(&target, b"x", &Options::default()).unwrap_err();
        assert_eq!(e.step, Step::Rename);
        assert!(!e.temp.exists());
        assert_eq!(names(&s.0), ["sub"]);
        assert!(write_atomic(&target, b"x").is_err());
        assert_eq!(names(&s.0), ["sub"]);
    }

    #[test]
    fn a_failed_prepare_leaves_the_old_file_and_no_temp_file() {
        let s = scratch();
        let p = s.0.join("a");
        std::fs::write(&p, "old").unwrap();
        let refuse = |_: &Path| -> std::io::Result<()> { Err(std::io::Error::other("no")) };
        let opts = Options {
            before_rename: Some(&refuse),
            ..Options::default()
        };
        let e = write_atomic_with(&p, b"new", &opts).unwrap_err();
        assert_eq!(e.step, Step::Prepare);
        assert_eq!(e.to_string(), "no");
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "old");
        assert_eq!(names(&s.0), ["a"]);
        // The hook sees the complete temporary file.
        let seen = std::sync::Mutex::new(Vec::new());
        let look = |t: &Path| -> std::io::Result<()> {
            seen.lock().unwrap().push(std::fs::read(t)?);
            Ok(())
        };
        let opts = Options {
            before_rename: Some(&look),
            ..Options::default()
        };
        write_atomic_with(&p, b"new", &opts).unwrap();
        assert_eq!(*seen.lock().unwrap(), [b"new".to_vec()]);
    }

    #[test]
    fn a_missing_folder_is_created_only_when_asked() {
        let s = scratch();
        let p = s.0.join("x").join("y").join("f");
        let e = write_atomic_with(&p, b"1", &Options::default()).unwrap_err();
        assert_eq!(e.step, Step::Create);
        assert!(!s.0.join("x").exists());
        let opts = Options {
            create_parent: true,
            ..Options::default()
        };
        write_atomic_with(&p, b"1", &opts).unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"1");
    }

    #[cfg(unix)]
    #[test]
    fn the_targets_mode_is_kept() {
        use std::os::unix::fs::PermissionsExt;
        let s = scratch();
        let p = s.0.join("c.json");
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o7777;
        for m in [0o640, 0o600, 0o644, 0o400] {
            std::fs::write(&p, "old").unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(m)).unwrap();
            write_atomic(&p, b"new").unwrap();
            assert_eq!(mode(&p), m, "{m:o}");
            assert_eq!(std::fs::read(&p).unwrap(), b"new");
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o600)).unwrap();
            std::fs::remove_file(&p).unwrap();
        }
        // Special bits are never copied.
        std::fs::write(&p, "old").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o1640)).unwrap();
        write_atomic(&p, b"new").unwrap();
        assert_eq!(mode(&p), 0o640);
    }

    #[cfg(unix)]
    #[test]
    fn a_private_file_is_0600_from_its_first_byte() {
        use std::os::unix::fs::PermissionsExt;
        let s = scratch();
        let p = s.0.join("secret");
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let seen = std::sync::Mutex::new(None);
        let look = |t: &Path| -> std::io::Result<()> {
            *seen.lock().unwrap() = Some(mode(t));
            Ok(())
        };
        let opts = Options {
            private: true,
            before_rename: Some(&look),
            ..Options::default()
        };
        write_atomic_with(&p, b"s3cret", &opts).unwrap();
        assert_eq!(mode(&p), 0o600);
        assert_eq!(*seen.lock().unwrap(), Some(0o600));
        // A wider existing file does not widen the new one.
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
        write_atomic_with(&p, b"s3cret2", &opts).unwrap();
        assert_eq!(mode(&p), 0o600);
        assert_eq!(names(&s.0), ["secret"]);
    }

    /// Two writers of different sizes, 200 writes each, and a reader: every read is one whole
    /// input, never a mix, and no temporary file stays.
    #[test]
    fn concurrent_writers_never_mix() {
        let s = scratch();
        let p = Arc::new(s.0.join("shared.json"));
        let inputs: Arc<Vec<Vec<u8>>> = Arc::new(vec![
            format!("[{}]", vec!["1"; 3000].join(",")).into_bytes(),
            format!("[{}]", vec!["22"; 50].join(",")).into_bytes(),
        ]);
        write_atomic(&p, inputs.first().unwrap()).unwrap();
        let done = Arc::new(AtomicBool::new(false));
        let reader = {
            let (p, inputs, done) = (p.clone(), inputs.clone(), done.clone());
            std::thread::spawn(move || {
                let mut reads = 0u32;
                loop {
                    let last = done.load(Ordering::Relaxed);
                    match std::fs::read(&*p) {
                        Ok(got) => {
                            assert!(inputs.contains(&got), "a mixed file of {} bytes", got.len());
                            let _: serde_json::Value = serde_json::from_slice(&got).unwrap();
                            reads += 1;
                        }
                        // Windows refuses an open while a rename replaces the file.
                        Err(e)
                            if cfg!(windows)
                                && e.kind() == std::io::ErrorKind::PermissionDenied => {}
                        Err(e) => panic!("{e}"),
                    }
                    if last {
                        break reads;
                    }
                }
            })
        };
        let writers: Vec<_> = (0..2)
            .map(|i| {
                let (p, inputs) = (p.clone(), inputs.clone());
                std::thread::spawn(move || {
                    // A reader or the other writer can hold the target for a moment on Windows.
                    let opts = Options {
                        rename_attempts: RENAME_ATTEMPTS,
                        rename_pause: Duration::from_millis(1),
                        ..Options::default()
                    };
                    for _ in 0..200 {
                        write_atomic_with(&p, inputs.get(i).unwrap(), &opts).unwrap();
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        done.store(true, Ordering::Relaxed);
        let reads = reader.join().unwrap();
        assert!(cfg!(windows) || reads > 0);
        assert!(inputs.contains(&std::fs::read(&*p).unwrap()));
        assert_eq!(names(&s.0), ["shared.json"]);
    }

    #[test]
    fn rename_retrying_retries_only_sharing_violations() {
        let s = scratch();
        let started = Instant::now();
        let e = rename_retrying(
            &s.0.join("missing"),
            &s.0.join("b"),
            20,
            Duration::from_millis(50),
        )
        .unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "not retried"
        );
        let err = |code| std::io::Error::from_raw_os_error(code);
        assert_eq!(is_sharing_violation(&err(32)), cfg!(windows));
        assert_eq!(is_sharing_violation(&err(5)), cfg!(windows));
        assert!(!is_sharing_violation(&std::io::Error::other("x")));
    }

    fn lock_file(dir: &Path) -> std::fs::File {
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(dir.join(".lock"))
            .unwrap()
    }

    #[test]
    fn lock_wait_takes_a_free_lock_at_once() {
        let s = scratch();
        let f = lock_file(&s.0);
        let started = Instant::now();
        let far = Instant::now() + Duration::from_secs(60);
        assert!(lock_wait(&f, LockKind::Exclusive, far, Duration::from_secs(10)).unwrap());
        assert!(started.elapsed() < Duration::from_secs(5));
        // Held: another handle cannot have it, shared or exclusive.
        let other = lock_file(&s.0);
        for kind in [LockKind::Shared, LockKind::Exclusive] {
            assert!(!lock_wait(&other, kind, Instant::now(), Duration::ZERO).unwrap());
        }
        fs4::FileExt::unlock(&f).unwrap();
        // Shared locks share.
        assert!(lock_wait(&f, LockKind::Shared, Instant::now(), Duration::ZERO).unwrap());
        assert!(lock_wait(&other, LockKind::Shared, Instant::now(), Duration::ZERO).unwrap());
    }

    #[test]
    fn lock_wait_times_out_while_held() {
        let s = scratch();
        let held = lock_file(&s.0);
        fs4::FileExt::try_lock(&held).unwrap();
        let f = lock_file(&s.0);
        let started = Instant::now();
        let deadline = started + Duration::from_millis(80);
        assert!(!lock_wait(&f, LockKind::Exclusive, deadline, Duration::from_millis(5)).unwrap());
        assert!(started.elapsed() >= Duration::from_millis(80));
        let w = lock_wait_with(
            &f,
            LockKind::Shared,
            deadline_after(Duration::ZERO),
            Pauses::fixed(Duration::ZERO),
            || true,
        )
        .unwrap();
        assert_eq!(w, Waited::TimedOut);
    }

    #[test]
    fn lock_wait_gets_a_lock_released_meanwhile() {
        let s = scratch();
        let held = lock_file(&s.0);
        fs4::FileExt::try_lock(&held).unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            fs4::FileExt::unlock(&held).unwrap();
        });
        let f = lock_file(&s.0);
        let mut asked = 0;
        let w = lock_wait_with(
            &f,
            LockKind::Exclusive,
            deadline_after(Duration::from_secs(30)),
            Pauses::doubling(Duration::from_micros(100), Duration::from_millis(5)),
            || {
                asked += 1;
                true
            },
        )
        .unwrap();
        assert_eq!(w, Waited::Held);
        assert!(asked > 1, "{asked}");
        release.join().unwrap();
    }

    #[test]
    fn lock_wait_stops_when_asked() {
        let s = scratch();
        let held = lock_file(&s.0);
        fs4::FileExt::try_lock(&held).unwrap();
        let f = lock_file(&s.0);
        let mut asked = 0;
        let w = lock_wait_with(
            &f,
            LockKind::Exclusive,
            None,
            Pauses::fixed(Duration::from_millis(1)),
            || {
                asked += 1;
                asked < 3
            },
        )
        .unwrap();
        assert_eq!(w, Waited::Stopped);
        assert_eq!(asked, 3);
    }

    #[test]
    fn pauses_double_up_to_their_ceiling() {
        let mut p = Pauses::doubling(Duration::from_millis(1), Duration::from_millis(5));
        let got: Vec<u128> = (0..5).map(|_| p.take().as_millis()).collect();
        assert_eq!(got, [1, 2, 4, 5, 5]);
        let mut p = Pauses::fixed(Duration::from_millis(3));
        assert_eq!(
            (p.take(), p.take()),
            (Duration::from_millis(3), Duration::from_millis(3))
        );
        let mut p = Pauses::doubling(Duration::from_secs(9), Duration::from_secs(1));
        assert_eq!(p.take(), Duration::from_secs(1));
        assert!(deadline_after(Duration::MAX).is_none());
    }

    /// The cases image.rs's `safe_stem` relied on (moved here with the rule).
    #[test]
    fn windows_reserved_names() {
        for bad in [
            "CON",
            "con",
            "Nul",
            "Nul.png",
            "aux",
            "PRN.x.jpg",
            "COM1",
            "lpt9",
            "COM0",
            "con.",
            "con..jpg",
            "LPT0.txt",
        ] {
            assert!(is_windows_reserved(bad), "{bad:?}");
        }
        for ok in [
            "CONSOLE", "COM10", "LPTX", "con_od", "NULL", "", "x.con", "COM", "cam1", "acon",
        ] {
            assert!(!is_windows_reserved(ok), "{ok:?}");
        }
    }
}
