//! The one cross-process transaction on the config file (audit 2026-11, QA-016): every
//! load-edit-save of the config, by the service ([`crate::config_store::ConfigStore::update`])
//! and by the commands that edit it while it runs (`add-model`, `download-models
//! --add-to-config`, `train` registering its model, `train autolabel` / `train import
//! --review-folder` through `FileConfigWriter`, `benchmark --apply` / `--apply-threshold`, the
//! start's `resolve_config` and default-model seed), holds [`lock`], so no writer saves a copy
//! it read before another writer's save (a lost update, silent on both sides).
//!
//! - **The lock** is an fs4 exclusive lock on a separate zero-byte file beside the config,
//!   `.<config file name>.lock` ([`lock_path`]); never on the config itself (on Windows that
//!   would block the save's rename and every reader). It is never deleted (a holder keeps its
//!   lock on an unlinked file while the next taker would lock a new one). Readers take no lock:
//!   every save is an atomic rename ([`Config::save`]). Commands wait for it up to
//!   [`LOCK_WAIT`] (another holder keeps it for milliseconds), the service's page saves up to
//!   [`STORE_LOCK_WAIT`] (they hold the store's writer mutex meanwhile), then [`LockBusy`],
//!   which names the lock file. A lock file that cannot be opened or locked (a read-only folder,
//!   a file system without locks) is gone without: the write goes on unlocked, with one warning
//!   per folder; one that exists but is not writable for this account (created by another one)
//!   is locked through a read-only handle.
//! - **[`edit_file`]** is the transaction of the commands: lock, load the file (a later format
//!   is refused there, nothing written), edit, normalize, validate when changed, save only when
//!   the saved text changed or there is no file yet ([`Edited::changed`]), release.
//!
//! Lock order: `server::AppState::apply_lock` -> `ConfigStore`'s `writer` -> this lock. Nothing
//! done under this lock (an [`edit_file`] closure included) may call a writer of a
//! `ConfigStore` or [`edit_file`] again: fs4 locks are per open file, so a second take in the
//! same thread waits for itself until its deadline.

use crate::config::Config;
use crate::sync::LockExt;
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

/// How long a command waits for another writer's hold of the config lock: holders keep it for
/// one load-edit-save (milliseconds; a page save's edit plus its `fsync`), so this is only
/// reached by a process stuck while holding it.
pub const LOCK_WAIT: Duration = Duration::from_secs(30);

/// How long a save of the service ([`crate::config_store::ConfigStore::update`]) waits for a
/// command's hold: short, because the store's writer mutex is held meanwhile and every other
/// page save queues behind it.
pub const STORE_LOCK_WAIT: Duration = Duration::from_secs(3);

/// The longest pause between two tries of the lock (doubling from 1 ms).
const LOCK_POLL: Duration = Duration::from_millis(25);

/// Folders whose lock file could not be used (warned once each).
static NO_FILE_LOCK: Mutex<Option<HashSet<PathBuf>>> = Mutex::new(None);

/// The lock file of the config at `config`: `.<file name>.lock` in its folder.
pub fn lock_path(config: &Path) -> PathBuf {
    let name = config
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config".into());
    folder_of(config).join(format!(".{name}.lock"))
}

/// The folder of `path` (`.` for a bare name).
fn folder_of(path: &Path) -> &Path {
    match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    }
}

/// Another process held the config lock past the wait: nothing was written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockBusy {
    /// The config file.
    pub config: PathBuf,
    /// Its lock file ([`lock_path`]).
    pub lock: PathBuf,
    /// How long this writer waited.
    pub waited: Duration,
}

impl std::fmt::Display for LockBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the config file {} was not changed: its lock {} stayed held for {:.1} s by another \
             blue-onyx-prism process writing the config (the service saving a page, or a \
             command such as add-model, train or benchmark --apply). Try again in a moment; a \
             process that holds it stuck releases it when it ends",
            self.config.display(),
            self.lock.display(),
            self.waited.as_secs_f32()
        )
    }
}

impl std::error::Error for LockBusy {}

/// A hold of the config lock ([`lock`]); released on drop.
#[derive(Debug)]
pub struct ConfigLock {
    /// None: going on without a lock (see the module docs).
    file: Option<std::fs::File>,
}

impl ConfigLock {
    /// Whether the file lock is held (false: the write goes on unlocked).
    pub fn is_held(&self) -> bool {
        self.file.is_some()
    }
}

impl Drop for ConfigLock {
    fn drop(&mut self) {
        // Explicitly: a child process spawned now would share the open file until it execs.
        if let Some(f) = self.file.take()
            && let Err(e) = fs4::FileExt::unlock(&f)
        {
            tracing::debug!("releasing the config lock: {e}");
        }
    }
}

/// Take the lock of the config file `config`, waiting at most `wait` for another holder (see
/// the module docs). Errors with [`LockBusy`] when it stays held; a lock file that cannot be
/// used gives an unlocked hold ([`ConfigLock::is_held`] false) and one warning per folder.
pub fn lock(config: &Path, wait: Duration) -> Result<ConfigLock> {
    let path = lock_path(config);
    let dir = folder_of(config);
    let without = |why: String| {
        // A missing folder has no config to protect (the save fails on its own).
        if !dir.is_dir() {
            return ConfigLock { file: None };
        }
        let mut warned = NO_FILE_LOCK.lock_recover();
        if warned
            .get_or_insert_with(HashSet::new)
            .insert(dir.to_path_buf())
        {
            crate::logbuf::early(
                tracing::Level::WARN,
                format!(
                    "the config lock {} {why}, so saves of the config in {} are not \
                     coordinated with other blue-onyx-prism processes: a command editing the \
                     config (add-model, train, benchmark --apply) while the service saves a \
                     page could undo one of the two. Make the folder writable for this account",
                    path.display(),
                    dir.display()
                ),
            );
        }
        ConfigLock { file: None }
    };
    let file = match std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
    {
        Ok(f) => f,
        // Created by another account: a read-only handle locks it as well.
        Err(e) => match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => return Ok(without(format!("cannot be opened ({e})"))),
        },
    };
    let waited = crate::fsx::lock_wait_with(
        &file,
        crate::fsx::LockKind::Exclusive,
        crate::fsx::deadline_after(wait),
        crate::fsx::Pauses::doubling(Duration::from_millis(1), LOCK_POLL),
        || true,
    );
    match waited {
        Ok(crate::fsx::Waited::Held) => Ok(ConfigLock { file: Some(file) }),
        Ok(crate::fsx::Waited::TimedOut | crate::fsx::Waited::Stopped) => {
            Err(anyhow::Error::new(LockBusy {
                config: config.to_path_buf(),
                lock: path,
                waited: wait,
            }))
        }
        Err(e) => Ok(without(format!("cannot be locked ({e})"))),
    }
}

/// What [`edit_file`] does when the config file does not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Missing {
    /// Edit the default config and write it ([`Config::load_or_default`]).
    Default,
    /// Fail ([`Config::load`]'s error).
    Error,
}

/// What [`edit_file`] did.
#[derive(Debug)]
pub struct Edited<R> {
    /// What the edit returned.
    pub value: R,
    /// The file was written: the edit changed the saved text, or there was no file yet. False:
    /// the file was left untouched (bytes and modification time).
    pub changed: bool,
    /// The config after the edit (normalized), as it is in the file.
    pub config: Config,
}

/// Edit the config file at `path` under its lock ([`lock`], waiting at most [`LOCK_WAIT`]):
/// load it (or the default, see [`Missing`]; a file of a later format is refused and left as it
/// is), run `edit`, normalize, validate when the saved text changed, and save only then or when
/// there is no file yet. An error from any step writes nothing.
pub fn edit_file<R>(
    path: &Path,
    missing: Missing,
    edit: impl FnOnce(&mut Config) -> Result<R>,
) -> Result<Edited<R>> {
    edit_file_within(path, missing, LOCK_WAIT, edit)
}

/// [`edit_file`] waiting at most `wait` for the lock.
pub fn edit_file_within<R>(
    path: &Path,
    missing: Missing,
    wait: Duration,
    edit: impl FnOnce(&mut Config) -> Result<R>,
) -> Result<Edited<R>> {
    let _held = lock(path, wait)?;
    let existed = path.exists();
    let before = match missing {
        Missing::Default => Config::load_or_default(path)?,
        Missing::Error => Config::load(path)?,
    };
    let mut config = before.clone();
    let value = edit(&mut config)?;
    config.normalize();
    let changed = !existed || config.saved_text()? != before.saved_text()?;
    if changed {
        config.validate().with_context(|| {
            format!(
                "the change would leave an invalid value in the config file {}, so it was not \
                 changed",
                path.display()
            )
        })?;
        config.save(path)?;
    }
    Ok(Edited {
        value,
        changed,
        config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CONFIG_FORMAT, ModelConfig};
    use crate::config_store::ConfigStore;
    use std::sync::Arc;
    use std::sync::mpsc;

    /// How long a test waits for a step another thread must reach (a timeout only turns a hang
    /// into a failure).
    const STEP: Duration = Duration::from_secs(20);

    /// `cfg.json` in a new temporary folder, removed with the guard.
    struct Tmp(PathBuf);

    impl Tmp {
        fn path(&self) -> PathBuf {
            self.0.join("cfg.json")
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tmp() -> Tmp {
        let d = std::env::temp_dir().join(format!("bop_cfgfile_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    fn model(name: &str) -> ModelConfig {
        ModelConfig {
            name: Some(name.into()),
            path: format!("models/{name}.onnx").into(),
            ..Default::default()
        }
    }

    fn names(c: &Config) -> Vec<String> {
        c.models.iter().map(|m| m.effective_name()).collect()
    }

    /// The lock taken through another open file, as another process would hold it.
    fn hold(config: &Path) -> std::fs::File {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path(config))
            .unwrap();
        fs4::FileExt::try_lock(&f).unwrap();
        f
    }

    #[test]
    fn the_lock_file_sits_beside_the_config() {
        let dir = Path::new("some").join("dir");
        assert_eq!(
            lock_path(&dir.join("blue_onyx_prism_config.json")),
            dir.join(".blue_onyx_prism_config.json.lock")
        );
        assert_eq!(
            lock_path(Path::new("c.json")),
            Path::new(".").join(".c.json.lock")
        );
        assert!(!crate::fsx::is_temp_name(".c.json.lock"), "never swept");
    }

    /// An edit is saved, the lock file stays (zero bytes, never deleted), and a missing file
    /// with [`Missing::Default`] starts from the default config.
    #[test]
    fn an_edit_is_saved_and_the_lock_file_stays() {
        let t = tmp();
        let path = t.path();
        let e = edit_file(&path, Missing::Default, |c| {
            c.port = 4000;
            Ok("done")
        })
        .unwrap();
        assert_eq!((e.value, e.changed, e.config.port), ("done", true, 4000));
        assert_eq!(Config::load(&path).unwrap().port, 4000);
        let lock = std::fs::metadata(lock_path(&path)).unwrap();
        assert_eq!(lock.len(), 0, "the lock file holds nothing");
        // Released: another handle takes it at once.
        drop(hold(&path));
    }

    /// QA-018: [`Missing::Error`] refuses a missing file and creates nothing.
    #[test]
    fn a_missing_file_is_an_error_when_asked() {
        let t = tmp();
        let path = t.path();
        let mut ran = false;
        let err = edit_file(&path, Missing::Error, |_| {
            ran = true;
            Ok(())
        })
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("could not read the config file"),
            "{err:#}"
        );
        assert!(!ran, "the edit never ran");
        assert!(!path.exists());
    }

    /// QA-017: an edit that changes nothing leaves the file alone (bytes and modification
    /// time), also a hand-formatted one, and says so.
    #[test]
    fn an_unchanged_edit_writes_nothing() {
        let t = tmp();
        let path = t.path();
        let text = "{ \"port\": 4000,\n\n  \"models_dir\": \"models\" }";
        std::fs::write(&path, text).unwrap();
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        let e = edit_file(&path, Missing::Default, |c| {
            c.port = 4000;
            Ok(())
        })
        .unwrap();
        assert!(!e.changed);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), mtime);
    }

    /// An edit that would leave an invalid value is refused; the file is unchanged.
    #[test]
    fn an_invalid_edit_is_refused() {
        let t = tmp();
        let path = t.path();
        Config::default().save(&path).unwrap();
        let before = std::fs::read(&path).unwrap();
        let err = edit_file(&path, Missing::Error, |c| {
            c.models = vec![model("twin"), model("twin")];
            Ok(())
        })
        .unwrap_err();
        let err = format!("{err:#}");
        assert!(err.contains("duplicate model name"), "{err}");
        assert!(err.contains("was not changed"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    /// QA-018: a file a later version wrote is refused and left byte-identical; the edit never
    /// runs.
    #[test]
    fn a_config_of_a_later_format_is_left_byte_identical() {
        let t = tmp();
        let path = t.path();
        let text = format!(r#"{{"format": {}, "port": 4000}}"#, CONFIG_FORMAT + 1);
        std::fs::write(&path, &text).unwrap();
        for missing in [Missing::Default, Missing::Error] {
            let mut ran = false;
            let err = edit_file(&path, missing, |c| {
                ran = true;
                c.port = 5000;
                Ok(())
            })
            .unwrap_err();
            assert!(
                format!("{err:#}").contains("written by a later version"),
                "{err:#}"
            );
            assert!(!ran);
            assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        }
    }

    /// QA-018: a lock another process holds past the wait is an error that names the lock
    /// file; the config bytes are unchanged and the edit never ran.
    #[test]
    fn a_lock_held_past_the_wait_is_an_error_naming_it() {
        let t = tmp();
        let path = t.path();
        Config::default().save(&path).unwrap();
        let before = std::fs::read(&path).unwrap();
        let held = hold(&path);
        let mut ran = false;
        let err = edit_file_within(&path, Missing::Error, Duration::from_millis(100), |c| {
            ran = true;
            c.port = 4000;
            Ok(())
        })
        .unwrap_err();
        let busy = err.downcast_ref::<LockBusy>().expect("LockBusy");
        assert_eq!(busy.lock, lock_path(&path));
        let msg = format!("{err:#}");
        assert!(
            msg.contains(&lock_path(&path).display().to_string()),
            "{msg}"
        );
        assert!(msg.contains("was not changed"), "{msg}");
        assert!(!ran);
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(held);
        edit_file_within(&path, Missing::Error, STEP, |c| {
            c.port = 4000;
            Ok(())
        })
        .unwrap();
        assert_eq!(Config::load(&path).unwrap().port, 4000);
    }

    /// QA-018: two writers at once both keep their change: the second waits for the first's
    /// save and then edits what it wrote.
    #[test]
    fn two_writers_at_once_both_keep_their_change() {
        let t = tmp();
        let path = t.path();
        Config::default().save(&path).unwrap();
        let (inside_tx, inside) = mpsc::channel();
        let (go, go_rx) = mpsc::channel::<()>();
        let first = {
            let path = path.clone();
            std::thread::spawn(move || {
                edit_file(&path, Missing::Error, |c| {
                    c.models.push(model("from-first"));
                    inside_tx.send(()).unwrap();
                    go_rx.recv_timeout(STEP).unwrap();
                    Ok(())
                })
                .unwrap()
            })
        };
        inside
            .recv_timeout(STEP)
            .expect("the first is inside its edit");
        let (done_tx, done) = mpsc::channel();
        let second = {
            let path = path.clone();
            std::thread::spawn(move || {
                let e = edit_file(&path, Missing::Error, |c| {
                    c.port = 4000;
                    Ok(())
                })
                .unwrap();
                done_tx.send(()).unwrap();
                e
            })
        };
        assert!(
            done.recv_timeout(Duration::from_millis(300)).is_err(),
            "the second saved while the first held its copy"
        );
        go.send(()).unwrap();
        assert!(first.join().unwrap().changed);
        let e = second.join().unwrap();
        assert_eq!(
            names(&e.config),
            ["from-first"],
            "the second edited the first's save"
        );
        let c = Config::load(&path).unwrap();
        assert_eq!((c.port, names(&c)), (4000, vec!["from-first".to_string()]));
    }

    /// QA-013, the mirrored case: a command has loaded the file (inside its edit) when a page
    /// save of the service starts; the save waits for the command's, then starts from it, so
    /// both changes stay on disk and in the service.
    #[test]
    fn a_page_save_during_a_command_edit_keeps_both() {
        let t = tmp();
        let path = t.path();
        Config::default().save(&path).unwrap();
        let s = Arc::new(ConfigStore::new(Config::load(&path).unwrap(), path.clone()));
        let (done_tx, done) = mpsc::channel();
        let mut page = None;
        edit_file(&path, Missing::Error, |c| {
            c.models.push(model("from-command"));
            let s = s.clone();
            let done_tx = done_tx.clone();
            page = Some(std::thread::spawn(move || {
                let up = s.update("config page", |c| {
                    c.port = 4000;
                    Ok(())
                });
                done_tx.send(()).unwrap();
                up
            }));
            assert!(
                done.recv_timeout(Duration::from_millis(300)).is_err(),
                "the page save ran while the command held its copy"
            );
            Ok(())
        })
        .unwrap();
        page.unwrap().join().unwrap().unwrap();
        let c = Config::load(&path).unwrap();
        assert_eq!(
            (c.port, names(&c)),
            (4000, vec!["from-command".to_string()])
        );
        s.check_disk();
        let r = s.read();
        assert_eq!(
            (r.port, names(&r)),
            (4000, vec!["from-command".to_string()])
        );
    }

    /// A lock file that cannot be opened (here: a folder in its place) does not stop the edit.
    #[test]
    fn an_unusable_lock_file_goes_on_unlocked() {
        let t = tmp();
        let path = t.path();
        std::fs::create_dir(lock_path(&path)).unwrap();
        edit_file(&path, Missing::Default, |c| {
            c.port = 4000;
            Ok(())
        })
        .unwrap();
        assert_eq!(Config::load(&path).unwrap().port, 4000);
    }

    /// The other process of [`two_processes_never_lose_each_others_edits`]; idle in a normal run.
    #[test]
    fn config_lock_child() {
        if let (Ok(path), Ok(tag), Ok(n)) = (
            std::env::var("BOP_TEST_CONFIG_LOCK_FILE"),
            std::env::var("BOP_TEST_CONFIG_LOCK_TAG"),
            std::env::var("BOP_TEST_CONFIG_LOCK_EDITS"),
        ) {
            for i in 0..n.parse::<usize>().unwrap() {
                edit_file(Path::new(&path), Missing::Error, |c| {
                    c.models.push(model(&format!("{tag}-{i}")));
                    Ok(())
                })
                .unwrap();
            }
        }
    }

    /// Two processes editing one config file at once lose none of each other's edits.
    #[test]
    fn two_processes_never_lose_each_others_edits() {
        const EDITS: usize = 25;
        let t = tmp();
        let path = t.path();
        Config::default().save(&path).unwrap();
        let children: Vec<_> = ["a", "b"]
            .iter()
            .map(|tag| {
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["config_file::tests::config_lock_child", "--exact"])
                    .env("BOP_TEST_CONFIG_LOCK_FILE", &path)
                    .env("BOP_TEST_CONFIG_LOCK_TAG", tag)
                    .env("BOP_TEST_CONFIG_LOCK_EDITS", EDITS.to_string())
                    .stdout(std::process::Stdio::null())
                    .spawn()
                    .unwrap()
            })
            .collect();
        for mut c in children {
            assert!(c.wait().unwrap().success());
        }
        let models = Config::load(&path).unwrap().models.len();
        assert_eq!(models, 2 * EDITS, "edits of the other process were lost");
    }
}
