//! The one in-memory config of the service with a revision number and a short change log.
//!
//! Every change goes through [`ConfigStore::update`] (web UI forms, Benchmark apply, Resources
//! "Add to config", the log level, YOLO26 export) or is picked up from the file
//! ([`ConfigStore::check_disk`]: the CLI or an editor changed it) or a restart
//! ([`ConfigStore::adopt`]). Each change bumps the revision and is logged with its source and a
//! summary, which the pages poll through `GET /v1/config`.
//!
//! The store is shared by every registry generation of the process (restarts keep the revision
//! and the log).
//!
//! Readers never wait (RR-C1): the state readers see (config, revision, generation, change log,
//! file error) is an immutable `Published` value behind an [`ArcSwap`]; [`ConfigStore::read`]
//! and the other readers are one atomic load, and a [`ConfigRef`] is one consistent state. A
//! writer builds the next state beside the current one and publishes it after the file is
//! written, so a request handler reading the config while a save syncs the file to disk gets
//! the previous config at once instead of waiting for the `fsync`.
//!
//! Lock order: readers take no lock. Writers ([`ConfigStore::update`], [`ConfigStore::adopt`],
//! [`ConfigStore::check_disk`]) serialize on `writer`, the store's only lock, held across the
//! external-edit check, the edit, the file write (its `fsync`) and the publish. When a writer
//! also holds `server::AppState::apply_lock` (`save_config`), that lock is taken first:
//! `apply_lock` -> `writer`, never the reverse (nothing done under `writer`, an edit closure
//! included, may take `apply_lock` or call a writer of this store: the mutex is not reentrant).
//! [`ConfigStore::check_disk`] only tries `writer`: the polls that call it never queue behind a
//! save.
//!
//! Edits on disk: each generation runs a disk watch ([`spawn_disk_watch`], every
//! [`DISK_POLL`]) that calls `server::sync_config`, which applies a changed file exactly like a
//! Save (live settings swap, per-model reload, runtime-level fields pending until a restart).
//! The pages and `GET /v1/config` sync as well. A file that does not parse or validate is not
//! adopted: the running config stays, the reason is logged once per content and shown as
//! `fileError` in `GET /v1/config`. A file caught mid-write (in-place rewrite) is retried for
//! [`SETTLE`] before it counts as broken.

use crate::config::Config;
use crate::config_merge::{FieldChange, describe_changes};
use crate::sync::LockExt;
use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use serde::Serialize;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Changes kept for `GET /v1/config`.
pub const CHANGE_LOG_LEN: usize = 50;

/// Source of changes found in the file.
pub const SOURCE_DISK: &str = "file changed on disk";

/// How often the disk watch looks at the config file (one `stat` when it did not change).
pub const DISK_POLL: std::time::Duration = std::time::Duration::from_secs(2);

/// A file modified this recently that ends before its JSON does may still be being written (an
/// editor truncates and rewrites in place): it is read again on the next poll before being
/// reported.
pub const SETTLE: std::time::Duration = std::time::Duration::from_secs(3);

/// Run `tick` every `interval` on a thread named `config-watch` until `cancel` is cancelled or
/// `tick` returns false (its state is gone). The runner passes `server::sync_config` of the
/// generation, so a file edited on disk applies without any page open.
pub fn spawn_disk_watch(
    interval: std::time::Duration,
    cancel: tokio_util::sync::CancellationToken,
    mut tick: impl FnMut() -> bool + Send + 'static,
) -> Option<std::thread::JoinHandle<()>> {
    let step = interval.min(std::time::Duration::from_millis(250));
    std::thread::Builder::new()
        .name("config-watch".into())
        .spawn(move || {
            loop {
                let next = std::time::Instant::now() + interval;
                while std::time::Instant::now() < next {
                    if cancel.is_cancelled() {
                        return;
                    }
                    std::thread::sleep(step);
                }
                if cancel.is_cancelled() || !tick() {
                    return;
                }
            }
        })
        .map_err(|e| {
            tracing::warn!(
                "could not start the config file watch ({e}); edits on disk apply when a page \
                 is opened or the service restarts"
            )
        })
        .ok()
}

/// One logged change.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangeRecord {
    pub revision: u64,
    /// RFC 3339.
    pub time: String,
    /// "config page", "benchmark apply", "file changed on disk", ...
    pub source: String,
    /// "IPcam-general device → openvino:cpu, threshold → 0.35".
    pub summary: String,
    pub fields: Vec<FieldChange>,
    /// Labels of the fields of this change that apply without a restart (live settings, the
    /// default model, the log level; see [`crate::config_merge::applied_live`]).
    pub live: Vec<String>,
}

/// File identity used to notice changes made by other processes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DiskStamp {
    modified: Option<SystemTime>,
    len: u64,
    hash: u64,
}

fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    b.hash(&mut h);
    h.finish()
}

fn stat(path: &Path) -> Option<(Option<SystemTime>, u64)> {
    let md = std::fs::metadata(path).ok()?;
    Some((md.modified().ok(), md.len()))
}

fn stamp_of(path: &Path, bytes: &[u8]) -> Option<DiskStamp> {
    let (modified, len) = stat(path)?;
    Some(DiskStamp {
        modified,
        len,
        hash: hash_bytes(bytes),
    })
}

/// `bytes` end before the JSON does (empty, or cut off): what a reader sees while an editor
/// rewrites the file in place. Validation errors and other syntax errors are never this.
fn looks_incomplete(bytes: &[u8]) -> bool {
    bytes.iter().all(u8::is_ascii_whitespace)
        || serde_json::from_slice::<serde::de::IgnoredAny>(bytes)
            .err()
            .is_some_and(|e| e.is_eof())
}

/// The warning (and `fileError`) for a config file edited on disk that cannot be adopted.
fn rejected_message(path: &Path, e: &anyhow::Error) -> String {
    format!(
        "config file {} changed on disk but was not applied: {e:#}. The service keeps running \
         with the previous config. Fix the file (valid JSON; values as the Config page \
         accepts them) and save it again: it is re-read within {} s. A save from the web pages \
         replaces the file with the running config.",
        path.display(),
        DISK_POLL.as_secs()
    )
}

/// What readers see: one consistent state, never changed in place; every change publishes a new
/// one ([`ConfigStore::publish`]).
#[derive(Debug, Clone)]
struct Published {
    config: Config,
    revision: u64,
    generation: u64,
    changes: VecDeque<ChangeRecord>,
    disk_error: Option<String>,
    /// +1 per publish: anything a reader sees changed (config, revision, generation, change
    /// log, file error).
    serial: u64,
}

/// What only writers touch (under [`ConfigStore::writer`]).
#[derive(Debug)]
struct Writer {
    /// The file as this store last read or wrote it.
    disk: Option<DiskStamp>,
}

/// Read access to the current config: a shared handle on one published state, so it never
/// blocks a writer, a writer never blocks it, and it can be held across an `.await`. It derefs
/// to the [`Config`]; the store state it was taken from is read with the associated functions
/// ([`ConfigRef::revision`], ...), named like `Arc::ptr_eq` so none shadows a `Config` item.
pub struct ConfigRef(Arc<Published>);

impl std::fmt::Debug for ConfigRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ConfigRef").field(&self.0.config).finish()
    }
}

impl std::ops::Deref for ConfigRef {
    type Target = Config;
    fn deref(&self) -> &Config {
        &self.0.config
    }
}

impl ConfigRef {
    /// Revision of this config.
    pub fn revision(this: &Self) -> u64 {
        this.0.revision
    }

    /// Registry generation when this was published.
    pub fn generation(this: &Self) -> u64 {
        this.0.generation
    }

    /// Publish counter: equal for two handles exactly when they hold the same published state
    /// (config, revision, generation, change log, file error), so a cache keyed by it is never
    /// stale.
    pub fn serial(this: &Self) -> u64 {
        this.0.serial
    }

    /// The last `n` changes, oldest first.
    pub fn changes(this: &Self, n: usize) -> Vec<ChangeRecord> {
        let skip = this.0.changes.len().saturating_sub(n);
        this.0.changes.iter().skip(skip).cloned().collect()
    }

    /// Why the file on disk could not be used, if so.
    pub fn disk_error(this: &Self) -> Option<String> {
        this.0.disk_error.clone()
    }
}

/// Outcome of [`ConfigStore::update`].
#[derive(Debug)]
pub struct Updated<R> {
    /// What the edit returned.
    pub value: R,
    /// Revision after the update.
    pub revision: u64,
    /// The config changed (and the revision was bumped).
    pub changed: bool,
    /// The config before the edit.
    pub before: Config,
    /// The logged change, when something changed.
    pub record: Option<ChangeRecord>,
}

/// Test hook: runs inside [`ConfigStore::update`] with `writer` held, right where the file is
/// written and synced to disk.
#[cfg(test)]
pub(crate) type WriteHook = Arc<dyn Fn() + Send + Sync>;

pub struct ConfigStore {
    path: PathBuf,
    /// Process start in Unix milliseconds: revisions restart at 1 in a new process.
    epoch: u64,
    /// What readers see (replaced whole, only with `writer` held).
    current: ArcSwap<Published>,
    /// Serializes the writers (see the module doc for the lock order).
    writer: Mutex<Writer>,
    #[cfg(test)]
    write_hook: Mutex<Option<WriteHook>>,
}

impl std::fmt::Debug for ConfigStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigStore")
            .field("path", &self.path)
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

/// Make the process-wide decode ceiling follow a config change (`max_image_pixels` applies at
/// once). Only set when the value changed, so unrelated saves never touch the global.
fn apply_max_pixels(before: &Config, after: &Config) {
    if before.max_image_pixels != after.max_image_pixels {
        crate::image::set_max_pixels(after.max_image_pixels);
    }
}

impl ConfigStore {
    /// A store holding `config` (normalized, see [`Config::normalize`]) for the file at `path`.
    pub fn new(mut config: Config, path: PathBuf) -> Self {
        config.normalize();
        let disk = std::fs::read(&path).ok().and_then(|b| stamp_of(&path, &b));
        let epoch = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .and_then(|d| u64::try_from(d.as_millis()).ok())
            .unwrap_or(0);
        Self {
            path,
            epoch,
            current: ArcSwap::from_pointee(Published {
                config,
                revision: 1,
                generation: 1,
                changes: VecDeque::new(),
                disk_error: None,
                serial: 1,
            }),
            writer: Mutex::new(Writer { disk }),
            #[cfg(test)]
            write_hook: Mutex::new(None),
        }
    }

    /// Run `hook` inside every later [`Self::update`] that writes the file, with the writer lock
    /// held, just before the write and its `fsync` (None removes it).
    #[cfg(test)]
    pub(crate) fn set_write_hook(&self, hook: Option<WriteHook>) {
        *self.write_hook.lock_recover() = hook;
    }

    #[cfg(test)]
    fn run_write_hook(&self) {
        let hook = self.write_hook.lock_recover().clone();
        if let Some(h) = hook {
            h();
        }
    }

    #[cfg(not(test))]
    fn run_write_hook(&self) {}

    /// What an edit from `before` to `new` writes: `new`, but a one-off flag of this run
    /// ([`Config::one_off_flags`]) stays in memory, or any save from a page would make it
    /// permanent. An edit that changes such a flag is a real change and is written. With no
    /// readable file there is nothing to keep (a file that does not parse gets replaced).
    fn for_the_file(&self, before: &Config, new: &Config) -> Config {
        let mut out = new.clone();
        if let Some(file) = Config::parse_file(&self.path) {
            let mut flags = file.one_off_flags();
            let edits = before.one_off_flags().into_iter().zip(new.one_off_flags());
            for (flag, (was, now)) in flags.iter_mut().zip(edits) {
                if was != now {
                    *flag = now;
                }
            }
            out.set_one_off_flags(flags);
        }
        out
    }

    /// Publish the current state changed by `edit` (a copy: readers holding the old one keep
    /// it). `_held` proves the caller holds `writer`, so no two publishes race.
    fn publish<R>(&self, _held: &Writer, edit: impl FnOnce(&mut Published) -> R) -> R {
        let mut next = Published::clone(&self.current.load());
        let out = edit(&mut next);
        next.serial += 1;
        self.current.store(Arc::new(next));
        out
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The current config. Never waits, also not while a save writes the file; the handle is
    /// one consistent state (see [`ConfigRef`]).
    pub fn read(&self) -> ConfigRef {
        ConfigRef(self.current.load_full())
    }

    /// Revision and a copy of the config, consistent with each other.
    pub fn snapshot(&self) -> (u64, Config) {
        let g = self.current.load();
        (g.revision, g.config.clone())
    }

    pub fn revision(&self) -> u64 {
        self.current.load().revision
    }

    /// Registry generation (1 for the first, +1 per restart).
    pub fn generation(&self) -> u64 {
        self.current.load().generation
    }

    /// The last `n` changes, oldest first.
    pub fn changes(&self, n: usize) -> Vec<ChangeRecord> {
        ConfigRef::changes(&self.read(), n)
    }

    /// Why the file on disk could not be used (it is not valid JSON, ...), if so.
    pub fn disk_error(&self) -> Option<String> {
        self.current.load().disk_error.clone()
    }

    /// Bump the revision and log the change from `before` to `p.config`, if any.
    fn record(p: &mut Published, source: &str, before: &Config) -> Option<ChangeRecord> {
        if *before == p.config {
            return None;
        }
        p.revision += 1;
        let (fields, summary) = describe_changes(before, &p.config);
        let rec = ChangeRecord {
            revision: p.revision,
            time: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            source: source.to_string(),
            summary,
            live: crate::config_merge::applied_live(&fields),
            fields,
        };
        p.changes.push_back(rec.clone());
        while p.changes.len() > CHANGE_LOG_LEN {
            p.changes.pop_front();
        }
        Some(rec)
    }

    /// Reload the file when another process changed it (`w` is the held writer lock) and
    /// publish what it changed. `verify`: read and hash the file even when its length and mtime
    /// are unchanged. An edit within the same mtime tick (1-2 s on FAT/exFAT, some SMB and bind
    /// mounts, HFS+) that keeps the length (`4000` -> `4001`) is only seen that way (CO-08).
    fn sync_disk(&self, w: &mut Writer, verify: bool) -> Option<ChangeRecord> {
        let (modified, len) = stat(&self.path)?;
        if !verify
            && let Some(d) = &w.disk
            && d.modified == modified
            && d.len == len
        {
            return None;
        }
        let bytes = std::fs::read(&self.path).ok()?;
        let stamp = stamp_of(&self.path, &bytes)?;
        if w.disk.as_ref().is_some_and(|d| d.hash == stamp.hash) {
            w.disk = Some(stamp);
            return None;
        }
        let parsed = std::str::from_utf8(&bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .and_then(|t| serde_json::from_str::<Config>(t).map_err(Into::into))
            .and_then(|mut c| {
                c.normalize();
                // A file that fails validation is not adopted (the in-memory config stays).
                c.validate()?;
                Ok(c)
            });
        let settling = !verify
            && parsed.is_err()
            && looks_incomplete(&bytes)
            && modified
                .and_then(|m| SystemTime::now().duration_since(m).ok())
                .is_some_and(|age| age < SETTLE);
        if settling {
            // Possibly half written: leave the stamp so the next poll reads it again.
            tracing::debug!(
                "config file {} changed but does not parse yet; reading it again shortly",
                self.path.display()
            );
            return None;
        }
        w.disk = Some(stamp);
        let cur = self.current.load_full();
        match parsed {
            Ok(c) => {
                if c == cur.config {
                    if cur.disk_error.is_some() {
                        self.publish(w, |p| p.disk_error = None);
                    }
                    return None;
                }
                apply_max_pixels(&cur.config, &c);
                let rec = self.publish(w, |p| {
                    p.disk_error = None;
                    let before = std::mem::replace(&mut p.config, c);
                    Self::record(p, SOURCE_DISK, &before)
                });
                if let Some(r) = &rec {
                    tracing::info!(
                        revision = r.revision,
                        "config file changed on disk ({}): {}",
                        self.path.display(),
                        r.summary
                    );
                }
                rec
            }
            Err(e) => {
                let msg = rejected_message(&self.path, &e);
                tracing::warn!("{msg}");
                if cur.disk_error.as_deref() != Some(msg.as_str()) {
                    self.publish(w, |p| p.disk_error = Some(msg));
                }
                None
            }
        }
    }

    /// Pick up changes other processes made to the file (cheap when it did not change: one
    /// `stat`; an edit that keeps both the length and the mtime tick is only seen by the next
    /// [`Self::update`], which always hashes). Returns the logged change.
    ///
    /// Never waits for a writer: while one holds the lock (a save writing the file) this
    /// returns None at once. That writer hashes the file itself before it writes, and the next
    /// poll (the disk watch, a page) looks again.
    pub fn check_disk(&self) -> Option<ChangeRecord> {
        let mut w = self.writer.try_lock_recover()?;
        let now = stat(&self.path);
        match (&w.disk, now) {
            (_, None) => None,
            (Some(d), Some((m, l))) if d.modified == m && d.len == l => None,
            _ => self.sync_disk(&mut w, false),
        }
    }

    /// Apply `edit` to a copy of the current config (after picking up external file changes),
    /// write the file and make it current. The revision is bumped and the change logged when
    /// the config changed; the file is written when it changed or does not exist yet. On error
    /// the config stays (a change picked up from the file before the edit is kept).
    ///
    /// Readers keep the previous config until the file is written and read back; `edit` runs
    /// with the writer lock held (see the module doc).
    pub fn update<R>(
        &self,
        source: &str,
        edit: impl FnOnce(&mut Config) -> Result<R>,
    ) -> Result<Updated<R>> {
        let mut w = self.writer.lock_recover();
        // Always hash here (rare, and about to overwrite the file): the stat-only shortcut of
        // the poll can miss an external edit, which this write would then destroy.
        self.sync_disk(&mut w, true);
        let cur = self.current.load_full();
        let before = cur.config.clone();
        let mut new = before.clone();
        let value = edit(&mut new)?;
        new.normalize();
        let changed = new != before;
        if changed {
            new.validate()?;
        }
        let wrote = changed || !self.path.exists();
        if wrote {
            self.run_write_hook();
            self.for_the_file(&before, &new).save(&self.path)?;
            let bytes = std::fs::read(&self.path)
                .with_context(|| format!("reading back {}", self.path.display()))?;
            w.disk = stamp_of(&self.path, &bytes);
        }
        let (revision, record) = if changed || (wrote && cur.disk_error.is_some()) {
            apply_max_pixels(&before, &new);
            self.publish(&w, |p| {
                if wrote {
                    p.disk_error = None;
                }
                p.config = new;
                let rec = Self::record(p, source, &before);
                (p.revision, rec)
            })
        } else {
            (cur.revision, None)
        };
        drop(w);
        if let Some(r) = &record {
            tracing::info!(
                revision = r.revision,
                source,
                "config changed: {}",
                r.summary
            );
        }
        Ok(Updated {
            value,
            revision,
            changed,
            before,
            record,
        })
    }

    /// Start a new registry generation with `config` (a restart reloaded the file or restored
    /// the previous config). The in-memory config becomes `config` when one is given and the
    /// revision is still `based_on` (the revision the generation's config was read at); a
    /// difference is logged under `source`. A config saved while the generation was being built
    /// (a newer revision) is kept: the runner applies it to the new generation.
    pub fn adopt(
        &self,
        config: Option<Config>,
        source: &str,
        based_on: u64,
    ) -> Option<ChangeRecord> {
        let mut w = self.writer.lock_recover();
        let revision = self.current.load().revision;
        let Some(mut config) = config else {
            self.publish(&w, |p| p.generation += 1);
            return None;
        };
        if revision != based_on {
            tracing::info!(
                revision,
                based_on,
                "the config was saved while the server restarted; keeping the saved config ({source} not adopted)"
            );
            self.publish(&w, |p| p.generation += 1);
            return None;
        }
        config.normalize();
        if let Ok(bytes) = std::fs::read(&self.path) {
            w.disk = stamp_of(&self.path, &bytes);
        }
        self.publish(&w, |p| {
            p.generation += 1;
            let before = std::mem::replace(&mut p.config, config);
            Self::record(p, source, &before)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    /// `cfg.json` in a new temporary folder, removed with the guard.
    struct Tmp(PathBuf);

    impl std::ops::Deref for Tmp {
        type Target = PathBuf;
        fn deref(&self) -> &PathBuf {
            &self.0
        }
    }

    impl AsRef<Path> for Tmp {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            if let Some(d) = self.0.parent() {
                let _ = std::fs::remove_dir_all(d);
            }
        }
    }

    fn tmp() -> Tmp {
        let d = std::env::temp_dir().join(format!("bo_store_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d.join("cfg.json"))
    }

    fn cfg() -> Config {
        Config {
            models: vec![ModelConfig {
                name: Some("IPcam-general".into()),
                path: "models/IPcam-general.onnx".into(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// How long a test waits for a step another thread must reach (never a sleep: a timeout
    /// only turns a hang into a failure).
    const STEP: std::time::Duration = std::time::Duration::from_secs(20);

    /// A write hook that parks the writing thread inside the file write until released:
    /// `entered` receives one message per write that reached the hook.
    struct Park {
        entered: std::sync::mpsc::Receiver<()>,
        release: std::sync::mpsc::Sender<()>,
    }

    fn park(s: &ConfigStore) -> Park {
        let (entered_tx, entered) = std::sync::mpsc::channel();
        let (release, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Mutex::new(release_rx);
        s.set_write_hook(Some(Arc::new(move || {
            entered_tx.send(()).unwrap();
            release_rx
                .lock_recover()
                .recv_timeout(STEP)
                .expect("released");
        })));
        Park { entered, release }
    }

    /// RR-C1: while a save is inside its file write (`fsync` included), with the writer lock
    /// held, every reader answers at once with the previous config, and the poll's disk check
    /// does not queue behind the writer. Once the save returns, readers see the new config.
    #[test]
    fn readers_are_not_blocked_while_a_save_writes() {
        let path = tmp();
        let s = Arc::new(ConfigStore::new(cfg(), path.clone()));
        s.update("test", |_| Ok(())).unwrap();
        let p = park(&s);
        let writer = {
            let s = s.clone();
            std::thread::spawn(move || {
                s.update("config page", |c| {
                    c.port = 4000;
                    Ok(())
                })
                .unwrap()
            })
        };
        p.entered
            .recv_timeout(STEP)
            .expect("the save reached its write");
        // The writer is parked in the write, holding the lock.
        assert!(
            s.writer.try_lock().is_err(),
            "writer lock held across the write"
        );
        let (done_tx, done) = std::sync::mpsc::channel();
        let reader = {
            let s = s.clone();
            std::thread::spawn(move || {
                let r = s.read();
                let seen = (
                    r.port,
                    ConfigRef::revision(&r),
                    s.snapshot().0,
                    s.revision(),
                    s.generation(),
                    s.changes(10).len(),
                    s.disk_error(),
                    s.check_disk().is_none(),
                );
                done_tx.send(seen).unwrap();
            })
        };
        let seen = done
            .recv_timeout(STEP)
            .expect("a reader waited for the save's write");
        assert_eq!(
            seen,
            (crate::DEFAULT_PORT, 1, 1, 1, 1, 0, None, true),
            "readers see the previous config while the save writes"
        );
        reader.join().unwrap();
        p.release.send(()).unwrap();
        let up = writer.join().unwrap();
        assert_eq!(up.revision, 2);
        let r = s.read();
        assert_eq!((r.port, ConfigRef::revision(&r)), (4000, 2));
        assert_eq!(Config::load(&path).unwrap().port, 4000);
        // A handle taken before the save still holds the old state, consistently.
        drop(r);
    }

    /// Saves still serialize: a second save that starts while the first is inside its write
    /// runs its edit only after the first published, so it starts from the first's result and
    /// neither edit is lost.
    #[test]
    fn concurrent_saves_serialize() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let path = tmp();
        let s = Arc::new(ConfigStore::new(cfg(), path.clone()));
        s.update("test", |_| Ok(())).unwrap();
        let p = park(&s);
        let first = {
            let s = s.clone();
            std::thread::spawn(move || {
                s.update("first", |c| {
                    c.port = 4000;
                    Ok(())
                })
                .unwrap()
            })
        };
        p.entered
            .recv_timeout(STEP)
            .expect("the first save reached its write");
        let released = Arc::new(AtomicBool::new(false));
        let (started_tx, started) = std::sync::mpsc::channel();
        let second = {
            let (s, released) = (s.clone(), released.clone());
            std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                s.update("second", |c| {
                    // Runs only once the first save let go of the writer lock.
                    assert!(
                        released.load(Ordering::SeqCst),
                        "edit ran during another save"
                    );
                    assert_eq!(c.port, 4000, "the edit starts from the first save");
                    c.nms_iou = 0.45;
                    Ok(())
                })
                .unwrap()
            })
        };
        started.recv_timeout(STEP).unwrap();
        // The second save cannot have taken the lock: the first still holds it.
        assert!(s.writer.try_lock().is_err());
        assert!(p.entered.try_recv().is_err(), "two saves inside the write");
        released.store(true, Ordering::SeqCst);
        p.release.send(()).unwrap();
        assert_eq!(first.join().unwrap().revision, 2);
        p.entered
            .recv_timeout(STEP)
            .expect("the second save writes after");
        p.release.send(()).unwrap();
        assert_eq!(second.join().unwrap().revision, 3);
        let on_disk = Config::load(&path).unwrap();
        assert_eq!((on_disk.port, on_disk.nms_iou), (4000, 0.45));
        let sources: Vec<String> = s.changes(10).into_iter().map(|c| c.source).collect();
        assert_eq!(sources, ["first", "second"]);
    }

    /// The external-edit check still runs under the writer lock before every save: an edit on
    /// disk is adopted before the save writes (never overwritten), and a file that does not
    /// validate is refused, reported to readers, and replaced by the save with the running
    /// config. A disk check while a save writes leaves the file to that save.
    #[test]
    fn saves_still_check_the_file_first() {
        let path = tmp();
        let s = Arc::new(ConfigStore::new(cfg(), path.clone()));
        s.update("test", |_| Ok(())).unwrap();
        let serial = ConfigRef::serial(&s.read());
        // An editor sets the port; a save of another field keeps it.
        let mut other = Config::load(&path).unwrap();
        other.port = 4321;
        other.save(&path).unwrap();
        s.update("config page", |c| {
            c.nms_iou = 0.45;
            Ok(())
        })
        .unwrap();
        let now = Config::load(&path).unwrap();
        assert_eq!((now.port, now.nms_iou), (4321, 0.45));
        assert!(ConfigRef::serial(&s.read()) > serial);
        // A value out of range on disk: refused, readers see why, the config stays.
        let text = std::fs::read_to_string(&path).unwrap().replace(
            "\"worker_queue_size\": 0",
            "\"worker_queue_size\": 1000000000",
        );
        std::fs::write(&path, text).unwrap();
        assert!(s.check_disk().is_none());
        let err = ConfigRef::disk_error(&s.read()).expect("refusal published");
        assert!(err.contains("worker_queue_size"), "{err}");
        assert_eq!(s.read().worker_queue_size, 0);
        // The next save writes the running config over the broken file and clears the error.
        s.update("config page", |c| {
            c.port = 4322;
            Ok(())
        })
        .unwrap();
        assert_eq!(s.disk_error(), None);
        let now = Config::load(&path).unwrap();
        assert_eq!((now.port, now.worker_queue_size), (4322, 0));
        // A file edit seen by a poll during a save: the poll leaves it to the save, which
        // hashes the file first, so the edit made before the save's write survives it.
        let mut other = Config::load(&path).unwrap();
        other.nms_iou = 0.4;
        other.save(&path).unwrap();
        let p = park(&s);
        let writer = {
            let s = s.clone();
            std::thread::spawn(move || {
                s.update("config page", |c| {
                    c.port = 4323;
                    Ok(())
                })
                .unwrap()
            })
        };
        p.entered.recv_timeout(STEP).unwrap();
        assert!(s.check_disk().is_none(), "a poll never waits for the save");
        p.release.send(()).unwrap();
        writer.join().unwrap();
        let now = Config::load(&path).unwrap();
        assert_eq!((now.port, now.nms_iou), (4323, 0.4));
        let sources: Vec<String> = s.changes(10).into_iter().map(|c| c.source).collect();
        assert_eq!(
            sources,
            [
                SOURCE_DISK,
                "config page",
                "config page",
                SOURCE_DISK,
                "config page"
            ]
        );
    }

    /// Every publish changes the serial; a write that changes nothing publishes nothing, so a
    /// cache keyed by the serial stays valid.
    #[test]
    fn serial_changes_with_every_published_state() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        let serial = || ConfigRef::serial(&s.read());
        let a = serial();
        s.update("test", |_| Ok(())).unwrap();
        assert_eq!(
            serial(),
            a,
            "a first write of the same config publishes nothing"
        );
        s.update("test", |c| {
            c.port = 4000;
            Ok(())
        })
        .unwrap();
        let b = serial();
        assert!(b > a);
        s.adopt(None, "restart", 2);
        let c = serial();
        assert!(c > b, "a new generation is a new state");
        std::fs::write(&path, "{ not json").unwrap();
        s.check_disk();
        assert!(serial() > c, "a file error is a new state");
        let d = serial();
        s.check_disk();
        assert_eq!(serial(), d, "the same error again publishes nothing");
    }

    #[test]
    fn update_bumps_revision_and_logs() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        assert_eq!(s.revision(), 1);
        // No-op edit: file written once (did not exist), no bump.
        let u = s.update("test", |_| Ok(())).unwrap();
        assert!(!u.changed);
        assert_eq!(u.revision, 1);
        assert!(path.exists());
        let u = s
            .update("benchmark apply", |c| {
                c.models[0].device = Some("openvino:cpu".into());
                c.models[0].confidence_threshold = Some(0.3449 + 0.0052);
                Ok(())
            })
            .unwrap();
        assert!(u.changed);
        assert_eq!(u.revision, 2);
        let log = s.changes(10);
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].source, "benchmark apply");
        assert_eq!(
            log[0].summary,
            "IPcam-general device \u{2192} openvino:cpu, threshold \u{2192} 0.35"
        );
        // Stored rounded, in memory and in the file.
        assert_eq!(s.read().models[0].confidence_threshold, Some(0.35));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"confidence_threshold\": 0.35"), "{text}");
        // Our own write is not reported as an external change.
        assert!(s.check_disk().is_none());
        // A failing edit changes nothing.
        assert!(
            s.update("x", |c| -> Result<()> {
                c.port = 1;
                anyhow::bail!("no")
            })
            .is_err()
        );
        assert_eq!(s.revision(), 2);
        assert_eq!(s.read().port, crate::DEFAULT_PORT);
    }

    #[test]
    fn external_file_changes_are_detected() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        s.update("test", |_| Ok(())).unwrap();
        assert!(s.check_disk().is_none());
        // Another process (the CLI, an editor) rewrites the file.
        let mut other = Config::load(&path).unwrap();
        other.port = 4321;
        // Same length and possibly the same mtime tick: the content hash catches it.
        std::thread::sleep(std::time::Duration::from_millis(20));
        other.save(&path).unwrap();
        let rec = s.check_disk().expect("change detected");
        assert_eq!(rec.source, SOURCE_DISK);
        assert_eq!(rec.revision, 2);
        assert!(
            rec.summary.contains("Port \u{2192} 4321"),
            "{}",
            rec.summary
        );
        assert_eq!(s.read().port, 4321);
        assert!(s.check_disk().is_none());
        // A broken file is reported and ignored.
        std::fs::write(&path, "{ not json").unwrap();
        assert!(s.check_disk().is_none());
        assert!(s.disk_error().is_some());
        assert_eq!(s.read().port, 4321);
        // An update after an external change starts from the file's content.
        let mut other = s.read().clone();
        other.nms_iou = 0.45;
        other.save(&path).unwrap();
        s.update("config page", |c| {
            c.port = 4000;
            Ok(())
        })
        .unwrap();
        let now = Config::load(&path).unwrap();
        assert_eq!((now.port, now.nms_iou), (4000, 0.45));
        let log = s.changes(10);
        assert_eq!(
            log.iter().map(|c| c.source.as_str()).collect::<Vec<_>>(),
            [SOURCE_DISK, SOURCE_DISK, "config page"]
        );
        assert_eq!(s.revision(), 4);
    }

    /// CO-08: an external edit that keeps the length and the mtime (coarse-mtime filesystems)
    /// is missed by the stat-only poll but picked up by `update`, never overwritten.
    #[test]
    fn same_length_same_mtime_edit_survives_an_update() {
        let path = tmp();
        let s = ConfigStore::new(
            Config {
                port: 4000,
                ..cfg()
            },
            path.clone(),
        );
        s.update("test", |_| Ok(())).unwrap();
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        // An editor changes 4000 -> 4001 within the same mtime tick.
        let text = std::fs::read_to_string(&path)
            .unwrap()
            .replace("\"port\": 4000", "\"port\": 4001");
        std::fs::write(&path, &text).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(mtime)
            .unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), len);
        assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), mtime);
        // The cheap poll does not see it (by design)...
        assert!(s.check_disk().is_none());
        assert_eq!(s.read().port, 4000);
        // ...but a UI save starts from the edited file instead of overwriting it.
        s.update("config page", |c| {
            c.nms_iou = 0.45;
            Ok(())
        })
        .unwrap();
        let now = Config::load(&path).unwrap();
        assert_eq!((now.port, now.nms_iou), (4001, 0.45));
        assert_eq!(s.read().port, 4001);
        let sources: Vec<String> = s.changes(10).into_iter().map(|c| c.source).collect();
        assert_eq!(sources, [SOURCE_DISK, "config page"]);
    }

    #[test]
    fn invalid_values_are_not_adopted() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        s.update("test", |_| Ok(())).unwrap();
        // A hand edit with an out-of-range value: reported, the in-memory config stays.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let text = std::fs::read_to_string(&path).unwrap().replace(
            "\"worker_queue_size\": 0",
            "\"worker_queue_size\": 1000000000",
        );
        std::fs::write(&path, text).unwrap();
        assert!(s.check_disk().is_none());
        let err = s.disk_error().expect("disk error recorded");
        assert!(err.contains("worker_queue_size"), "{err}");
        assert_eq!(s.read().worker_queue_size, 0);
        assert_eq!(s.revision(), 1);
        // An edit that makes the config invalid is refused and changes nothing.
        let path2 = tmp();
        let s = ConfigStore::new(cfg(), path2.clone());
        assert!(
            s.update("x", |c| {
                c.nms_iou = 3.0;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(s.read().nms_iou, 0.5);
    }

    /// A run with `--allow-remote-admin` (or the other one-off flags) keeps them in memory, and
    /// a save from a page must not make them permanent.
    #[test]
    fn one_off_flags_stay_out_of_every_save() {
        let path = tmp();
        let file = cfg();
        file.save(&path).unwrap();
        let merged = Config {
            allow_remote_admin: true,
            force_cpu: true,
            save_ref_image: true,
            ..file
        };
        let s = ConfigStore::new(merged, path.clone());
        let flags = |c: &Config| (c.allow_remote_admin, c.force_cpu, c.save_ref_image);

        s.update("page", |c| {
            c.port = 4000;
            Ok(())
        })
        .unwrap();
        let on_disk = Config::load(&path).unwrap();
        assert_eq!(on_disk.port, 4000);
        assert_eq!(flags(&on_disk), (false, false, false));
        // This run keeps them, and the watch sees no change in the file.
        assert_eq!(flags(&s.read()), (true, true, true));
        assert!(s.check_disk().is_none());
        assert_eq!(flags(&s.read()), (true, true, true));

        // Changing one by hand is a real change: that one is saved, the others stay out.
        s.update("page", |c| {
            c.force_cpu = false;
            Ok(())
        })
        .unwrap();
        s.update("page", |c| {
            c.force_cpu = true;
            Ok(())
        })
        .unwrap();
        assert_eq!(flags(&Config::load(&path).unwrap()), (false, true, false));
    }

    /// Flags the file itself holds are no one-off: a save keeps them.
    #[test]
    fn flags_in_the_file_survive_a_save() {
        let path = tmp();
        let file = Config {
            allow_remote_admin: true,
            save_ref_image: true,
            ..cfg()
        };
        file.save(&path).unwrap();
        let s = ConfigStore::new(file, path.clone());
        s.update("page", |c| {
            c.port = 4000;
            Ok(())
        })
        .unwrap();
        let on_disk = Config::load(&path).unwrap();
        assert!(on_disk.allow_remote_admin && on_disk.save_ref_image);
    }

    #[test]
    fn adopt_on_restart() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        assert_eq!(s.generation(), 1);
        // Resource restart: same config, new generation, no bump.
        assert!(s.adopt(None, "restart", 1).is_none());
        assert_eq!((s.generation(), s.revision()), (2, 1));
        let mut c = cfg();
        c.port = 5000;
        let rec = s
            .adopt(Some(c), "restart (config file reloaded)", 1)
            .unwrap();
        assert_eq!(rec.revision, 2);
        assert_eq!(s.generation(), 3);
        assert_eq!(s.read().port, 5000);
    }

    #[test]
    fn adopt_keeps_a_config_saved_during_the_restart() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        // The restart read the config at revision 1; a save lands while it builds.
        let based_on = s.revision();
        s.update("test", |c| {
            c.port = 6000;
            Ok(())
        })
        .unwrap();
        let mut stale = cfg();
        stale.port = 5000;
        assert!(
            s.adopt(Some(stale), "restart (config file reloaded)", based_on)
                .is_none()
        );
        assert_eq!(s.read().port, 6000);
        assert_eq!(s.generation(), 2);
    }

    /// An editor that truncates and rewrites in place: a poll between the two sees an empty or
    /// cut-off file, which is read again instead of being reported; once written, it applies.
    #[test]
    fn half_written_file_is_read_again() {
        let path = tmp();
        let s = ConfigStore::new(cfg(), path.clone());
        s.update("test", |_| Ok(())).unwrap();
        let mut edited = s.read().clone();
        edited.port = 4321;
        let full = serde_json::to_string_pretty(&edited).unwrap();
        for partial in ["", "  ", &full[..full.len() / 2]] {
            std::fs::write(&path, partial).unwrap();
            assert!(s.check_disk().is_none());
            assert_eq!(s.disk_error(), None, "{partial:?} reported while settling");
        }
        std::fs::write(&path, &full).unwrap();
        let rec = s.check_disk().expect("the finished write applies");
        assert_eq!(rec.source, SOURCE_DISK);
        assert_eq!(s.read().port, 4321);

        // Still cut off after SETTLE: reported, with what failed and what to do.
        std::fs::write(&path, &full[..full.len() / 2]).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(SystemTime::now() - SETTLE - std::time::Duration::from_secs(1))
            .unwrap();
        assert!(s.check_disk().is_none());
        let err = s.disk_error().expect("reported after SETTLE");
        assert!(err.contains("changed on disk but was not applied"), "{err}");
        assert!(err.contains("EOF while parsing"), "{err}");
        assert!(
            err.contains("keeps running with the previous config"),
            "{err}"
        );
        assert!(err.contains("re-read within 2 s"), "{err}");
        assert_eq!(s.read().port, 4321);
    }

    #[test]
    fn incomplete_json_is_told_apart_from_broken_json() {
        assert!(looks_incomplete(b""));
        assert!(looks_incomplete(b" \n"));
        assert!(looks_incomplete(b"{\"port\": 32"));
        assert!(looks_incomplete(b"{\"models\": ["));
        assert!(!looks_incomplete(b"{ not json"));
        assert!(!looks_incomplete(b"{\"port\": 1}"));
        assert!(!looks_incomplete(b"{\"port\": 1}}"));
    }

    /// The per-model fields of the night / verifier work are part of the compared config: an
    /// edit on disk to them alone is a change.
    #[test]
    fn edits_to_new_model_fields_on_disk_are_changes() {
        let path = tmp();
        let mut c = cfg();
        c.models.push(ModelConfig {
            name: Some("verifier".into()),
            path: "models/verifier.onnx".into(),
            ..Default::default()
        });
        let s = ConfigStore::new(c, path.clone());
        s.update("test", |_| Ok(())).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        v["models"][1]["enabled"] = false.into();
        v["models"][0]["night_confidence_threshold"] = 0.2.into();
        v["models"][0]["verify"] = serde_json::json!({"model": "verifier"});
        std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        let rec = s.check_disk().expect("change detected");
        assert_eq!(rec.source, SOURCE_DISK);
        let keys: Vec<&str> = rec.fields.iter().map(|f| f.key.as_str()).collect();
        for k in ["enabled", "night_confidence_threshold", "verify"] {
            assert!(
                keys.iter().any(|x| x.contains(k)),
                "{k} missing from {keys:?}"
            );
        }
        let now = s.read();
        assert!(!now.models[1].enabled);
        assert_eq!(now.models[0].verify.as_ref().unwrap().model, "verifier");
    }

    #[test]
    fn disk_watch_ticks_until_cancelled_or_done() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        let ms = std::time::Duration::from_millis;
        let n = Arc::new(AtomicU32::new(0));
        let cancel = tokio_util::sync::CancellationToken::new();
        let counter = n.clone();
        let h = spawn_disk_watch(ms(10), cancel.clone(), move || {
            counter.fetch_add(1, Ordering::SeqCst);
            true
        })
        .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while n.load(Ordering::SeqCst) < 3 {
            assert!(std::time::Instant::now() < deadline, "watch did not tick");
            std::thread::sleep(ms(5));
        }
        cancel.cancel();
        h.join().unwrap();
        let after = n.load(Ordering::SeqCst);
        std::thread::sleep(ms(50));
        assert_eq!(n.load(Ordering::SeqCst), after, "ticked after cancel");
        // A tick returning false (its state is gone) ends the thread.
        let h =
            spawn_disk_watch(ms(1), tokio_util::sync::CancellationToken::new(), || false).unwrap();
        h.join().unwrap();
    }
}
