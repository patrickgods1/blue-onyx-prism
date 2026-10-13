//! The download worker: `worker_loop` (queue, backoff, bounded retries, panic containment), the
//! HTTPS-only client, `run_job` (download, verify, extract, swap into place), stale-file cleanup
//! and the SHA-256 helpers. Nothing downloaded is executed here.

use super::*;

pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub(super) fn worker_loop(shared: &Shared, rx: &Receiver<Msg>) {
    let _exit = WorkerExit(shared);
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            *shared.dead_reason.lock_recover() = Some(format!("creating its async runtime: {e}"));
            return;
        }
    };
    let client = match http_client() {
        Ok(c) => c,
        Err(e) => {
            *shared.dead_reason.lock_recover() = Some(format!("{e:#}"));
            return;
        }
    };
    let mut queue: VecDeque<(Job, u32)> = VecDeque::new();
    // (due, job, attempts so far)
    let mut retries: Vec<(Instant, Job, u32)> = Vec::new();
    loop {
        // Collect messages; block only when nothing is runnable.
        let msg = if queue.is_empty() {
            let next_due = retries.iter().map(|(d, _, _)| *d).min();
            match next_due {
                Some(d) => rx
                    .recv_timeout(d.saturating_duration_since(Instant::now()))
                    .ok(),
                None => rx.recv().ok().or(Some(Msg::Stop)),
            }
        } else {
            rx.try_recv().ok()
        };
        match msg {
            Some(Msg::Stop) => return,
            Some(Msg::Job(job)) => {
                let key = job.key();
                retries.retain(|(_, j, _)| j.key() != key);
                if !queue.iter().any(|(j, _)| j.key() == key) {
                    queue.push_back((*job, 0));
                }
                continue;
            }
            None => {}
        }
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        let now = Instant::now();
        let (due, later): (Vec<_>, Vec<_>) = retries.drain(..).partition(|(d, _, _)| *d <= now);
        retries = later;
        for (_, job, attempts) in due {
            shared.set(&job, State::Queued);
            queue.push_back((job, attempts));
        }
        let Some((job, attempts)) = queue.pop_front() else {
            continue;
        };
        #[cfg(test)]
        if job.resource.id == TEST_PANIC_ID {
            panic!("test: download worker panics");
        }
        match run_job(shared, &client, &rt, &job) {
            Ok(()) => {
                shared.set(&job, State::Installed);
                shared.emit(Event::Installed {
                    key: job.key(),
                    id: job.resource.id,
                    target: job.target.clone(),
                });
            }
            Err(e) => {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                let attempts = attempts + 1;
                crate::metrics::DOWNLOADS
                    .failures
                    .fetch_add(1, Ordering::Relaxed);
                let mut error = format!("{e:#}");
                let retryable = retryable(&e);
                let exhausted = shared.opts.max_attempts.is_some_and(|max| attempts >= max);
                let retry = shared.opts.retry && retryable && !exhausted;
                if shared.opts.retry && retryable && exhausted {
                    error = format!(
                        "{error} (gave up after {attempts} attempts; check the network \
                         connection, proxy and firewall, then press Download on the Config page \
                         or run `{}`)",
                        fetch_command(job.resource)
                    );
                }
                let delay = (shared.opts.backoff)(attempts);
                let retry_at = retry.then(|| SystemTime::now() + delay);
                tracing::warn!(
                    "download of {} failed (attempt {attempts}): {error}{}",
                    job.resource.id,
                    if retry {
                        format!("; retrying in {} s", delay.as_secs())
                    } else {
                        String::new()
                    }
                );
                shared.set(
                    &job,
                    State::Failed {
                        error: error.clone(),
                        retry_at: retry_at.map(|t| {
                            t.duration_since(SystemTime::UNIX_EPOCH)
                                .map_or(0, |d| d.as_secs())
                        }),
                        attempts,
                    },
                );
                shared.emit(Event::Failed {
                    key: job.key(),
                    id: job.resource.id,
                    error,
                    retry_at,
                });
                if retry {
                    retries.push((Instant::now() + delay, job, attempts));
                }
            }
        }
    }
}

/// Resource id whose job panics the worker (tests of RS-04 only).
#[cfg(test)]
const TEST_PANIC_ID: &str = "model:test-worker-panic";

/// `http://` to the loopback address is accepted only in the crate's own test build (the
/// fixture server of the unit tests) or after [`allow_loopback_http_for_tests`]; otherwise the
/// client is HTTPS-only.
static LOOPBACK_HTTP_ALLOWED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(cfg!(test));

fn loopback_http_allowed() -> bool {
    LOOPBACK_HTTP_ALLOWED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Accept plain `http://127.0.0.1:*` / `http://localhost:*` download URLs for the rest of the
/// process. Only for integration tests with a loopback fixture server; no config or CLI path
/// reaches it. Must be called before the first download.
pub fn allow_loopback_http_for_tests() {
    LOOPBACK_HTTP_ALLOWED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// The download client: HTTPS only (plus loopback `http://` under `cfg(test)`), the same
/// rule applied to every redirect target, at most [`MAX_REDIRECTS`] hops.
fn http_client() -> Result<reqwest::Client> {
    let policy = reqwest::redirect::Policy::custom(|attempt| {
        // `previous` holds the initial URL and every hop so far.
        if attempt.previous().len() > MAX_REDIRECTS {
            return attempt.error(RedirectRefused::TooMany);
        }
        let next = attempt.url().to_string();
        if url_allowed(&next) {
            attempt.follow()
        } else {
            attempt.error(RedirectRefused::Scheme(next))
        }
    });
    reqwest::Client::builder()
        .user_agent(concat!("blue-onyx-prism/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(60))
        .https_only(!loopback_http_allowed())
        .redirect(policy)
        .build()
        .context("creating HTTP client")
}

/// Why a redirect was not followed (wrapped in a `reqwest::Error` that `is_redirect()`).
#[derive(Debug, thiserror::Error)]
enum RedirectRefused {
    #[error(
        "redirect to {0} refused: only https URLs are allowed (a changed server or an \
         intercepting proxy); update Blue Onyx Prism or report it"
    )]
    Scheme(String),
    #[error("too many redirects (more than {MAX_REDIRECTS})")]
    TooMany,
}

/// `https://`, or `http://` to the loopback address when [`loopback_http_allowed`].
fn url_allowed(url: &str) -> bool {
    url.starts_with("https://")
        || (loopback_http_allowed()
            && (url.starts_with("http://127.0.0.1:") || url.starts_with("http://localhost:")))
}

fn check_url(url: &str) -> Result<()> {
    if url_allowed(url) {
        Ok(())
    } else {
        Err(anyhow!(UrlRefused(url.to_string())))
    }
}

/// Install one job (steps 1-6 of the module docs).
fn run_job(
    shared: &Shared,
    client: &reqwest::Client,
    rt: &tokio::runtime::Runtime,
    job: &Job,
) -> Result<()> {
    let res = job.resource;
    let local = job.local_archives.is_some();
    if !local && is_installed(res, &job.target) {
        return Ok(());
    }
    let _lock = DownloadLock::acquire(&shared.opts.root, shared.opts.lock_wait, &shared.stop)?;
    if !local && is_installed(res, &job.target) {
        return Ok(());
    }
    let total = res.size();
    let mut done = 0u64;
    let mut last_report = Instant::now() - Duration::from_secs(1);
    let mut progress = |bytes: u64, force: bool| {
        if force || last_report.elapsed() >= Duration::from_millis(100) {
            last_report = Instant::now();
            shared.set(job, State::Downloading { bytes, total });
        }
    };

    // Model files: verified, then renamed into place next to the target.
    if res.parts.iter().all(|p| p.layout == Layout::File) {
        std::fs::create_dir_all(&job.target)
            .with_context(|| format!("creating {}", job.target.display()))?;
        let mut files = Vec::new();
        for part in res.parts {
            let dest = job.target.join(part.file_name);
            if file_matches(&dest, part.size, part.sha256, &shared.stop)? {
                done += part.size;
            } else {
                let partial = job.target.join(format!("{}.partial", part.file_name));
                rt.block_on(download_part(client, part, &partial, &shared.stop, |b| {
                    progress(done + b, false)
                }))?;
                shared.set(job, State::Verifying);
                verify(&partial, part, &shared.stop)?;
                std::fs::rename(&partial, &dest)
                    .with_context(|| format!("renaming to {}", dest.display()))?;
                done += part.size;
                progress(done, true);
            }
            files.push(part.file_name.to_string());
        }
        write_manifest(
            &manifest_path(res, &job.target),
            &Manifest {
                id: res.id.to_string(),
                version: job.version().to_string(),
                sha256: res.parts.iter().map(|p| p.sha256.to_string()).collect(),
                files,
            },
        )?;
        return Ok(());
    }

    // Archives: download (or take local ones), extract into staging, swap into place.
    let downloads = shared.opts.root.join(DOWNLOADS_DIR);
    let mut archives: Vec<PathBuf> = Vec::new();
    match &job.local_archives {
        Some(local) => {
            if local.len() != res.parts.len() {
                bail!(
                    "{} needs {} archive(s), got {}",
                    res.id,
                    res.parts.len(),
                    local.len()
                );
            }
            archives.extend(local.iter().cloned());
        }
        None => {
            std::fs::create_dir_all(&downloads)
                .with_context(|| format!("creating {}", downloads.display()))?;
            for part in res.parts {
                let file = downloads.join(part.file_name);
                if !file_matches(&file, part.size, part.sha256, &shared.stop)? {
                    let partial = downloads.join(format!("{}.partial", part.file_name));
                    rt.block_on(download_part(client, part, &partial, &shared.stop, |b| {
                        progress(done + b, false)
                    }))?;
                    shared.set(job, State::Verifying);
                    verify(&partial, part, &shared.stop)?;
                    std::fs::rename(&partial, &file)
                        .with_context(|| format!("renaming to {}", file.display()))?;
                }
                done += part.size;
                progress(done, true);
                archives.push(file);
            }
        }
    }

    shared.set(job, State::Extracting);
    let platform = res
        .platform
        .unwrap_or_else(crate::resources::catalog::Platform::current);
    let target = Target {
        os: platform.os,
        arch: platform.arch,
    };
    let staging = sibling(&job.target, "staging")?;
    remove_any(&staging)?;
    let result = (|| -> Result<BTreeSet<String>> {
        let mut got = BTreeSet::new();
        for (part, archive) in res.parts.iter().zip(&archives) {
            let kind = match &job.local_archives {
                Some(_) => archive_kind_of(archive).or(part.archive),
                None => part.archive,
            }
            .with_context(|| format!("{} is not an archive", archive.display()))?;
            if shared.stop.load(Ordering::Relaxed) {
                bail!("cancelled before extracting {}", archive.display());
            }
            got.extend(extract::extract_part_until(
                archive,
                kind,
                part.layout,
                res.flavor(),
                target,
                &staging,
                &shared.stop,
            )?);
        }
        extract::finalize(res, job.version(), target, &staging, &mut got)?;
        let manifest = Manifest {
            id: res.id.to_string(),
            version: job.version().to_string(),
            sha256: if local {
                Vec::new()
            } else {
                res.parts.iter().map(|p| p.sha256.to_string()).collect()
            },
            files: got.iter().cloned().collect(),
        };
        write_manifest(&staging.join(MANIFEST_FILE), &manifest)?;
        Ok(got)
    })();
    if let Err(e) = result {
        let _ = remove_any(&staging);
        return Err(e);
    }
    put_in_place(res, &staging, &job.target)?;
    if !local && !job.keep_downloads {
        for a in &archives {
            let _ = std::fs::remove_file(a);
        }
    }
    if job.activate_ort
        && let (Some(flavor), Some(parent)) = (res.flavor(), job.target.parent())
    {
        crate::backend::libs::write_active_flavor(parent, flavor.as_str())
            .with_context(|| format!("marking {flavor} active in {}", parent.display()))?;
    }
    Ok(())
}

fn archive_kind_of(p: &Path) -> Option<ArchiveKind> {
    let n = p.to_string_lossy().to_ascii_lowercase();
    if n.ends_with(".zip") || n.ends_with(".nupkg") || n.ends_with(".whl") {
        Some(ArchiveKind::Zip)
    } else if n.ends_with(".tgz") || n.ends_with(".tar.gz") {
        Some(ArchiveKind::TarGz)
    } else {
        None
    }
}

/// `.partial` files not touched for this long are leftovers of a download nobody resumed.
pub(crate) const STALE_PARTIAL_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// Staging / swap-aside directories (`.<name>.staging`, `.<name>.old-<pid>`) not touched for
/// this long belong to an interrupted install (a live one is rewritten continuously and the
/// download lock keeps a second process out).
pub(crate) const STALE_STAGING_AGE: Duration = Duration::from_secs(60 * 60);
/// How deep below the download root [`cleanup_stale`] looks.
const CLEANUP_MAX_DEPTH: usize = 4;

/// Startup cleanup under `root`: delete `*.partial` files older than [`STALE_PARTIAL_AGE`] and
/// stale staging directories ([`STALE_STAGING_AGE`]) left by an interrupted install. Only plain
/// files and directories are removed (symlinks are skipped); failures are ignored. Logs what
/// was removed at info. Returns the number of entries removed.
pub(crate) fn cleanup_stale(root: &Path, now: SystemTime) -> usize {
    let mut removed = 0;
    cleanup_dir(root, now, 0, &mut removed);
    if removed > 0 {
        tracing::info!(
            "download cleanup: removed {removed} stale partial file(s) / staging dir(s) under {}",
            root.display()
        );
    }
    removed
}

fn is_staging_name(name: &str) -> bool {
    name.starts_with('.') && (name.contains(".staging") || name.contains(".old-"))
}

fn older_than(meta: &std::fs::Metadata, now: SystemTime, age: Duration) -> bool {
    meta.modified()
        .ok()
        .and_then(|m| now.duration_since(m).ok())
        .is_some_and(|d| d >= age)
}

fn cleanup_dir(dir: &Path, now: SystemTime, depth: usize, removed: &mut usize) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in rd.flatten() {
        let Ok(ft) = ent.file_type() else { continue };
        let path = ent.path();
        let name = ent.file_name().to_string_lossy().into_owned();
        let Ok(meta) = ent.metadata() else { continue };
        if ft.is_file() && name.ends_with(".partial") {
            if older_than(&meta, now, STALE_PARTIAL_AGE) && std::fs::remove_file(&path).is_ok() {
                tracing::info!("removed stale partial download {}", path.display());
                *removed += 1;
            }
        } else if ft.is_dir() {
            if is_staging_name(&name) {
                if older_than(&meta, now, STALE_STAGING_AGE) && remove_any(&path).is_ok() {
                    tracing::info!("removed stale staging directory {}", path.display());
                    *removed += 1;
                }
            } else if depth < CLEANUP_MAX_DEPTH {
                cleanup_dir(&path, now, depth + 1, removed);
            }
        }
    }
}

/// `<parent>/.<name>.<what>` next to `dir` (same filesystem, so renames are atomic).
fn sibling(dir: &Path, what: &str) -> Result<PathBuf> {
    let name = dir
        .file_name()
        .with_context(|| format!("{} has no directory name", dir.display()))?
        .to_string_lossy();
    let parent = dir.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    Ok(parent.join(format!(".{name}.{what}")))
}

/// Remove `p` whether it is a file, a directory tree or a symlink; absent is fine.
pub(crate) fn remove_any(p: &Path) -> Result<()> {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => {
            std::fs::remove_dir_all(p).with_context(|| format!("removing {}", p.display()))
        }
        Ok(_) => std::fs::remove_file(p).with_context(|| format!("removing {}", p.display())),
        Err(_) => Ok(()),
    }
}

/// Move a finished `staging` directory to `target`:
/// - `target` absent: one rename.
/// - `target` is a previous install of the same resource (its manifest says so): rename it
///   aside, rename staging in, delete the old one (restored if the second rename fails).
/// - otherwise (a directory we do not own exclusively, e.g. a legacy flat
///   `onnxruntime/` that also holds per-flavor installs, or a user `--dir`): merge file by
///   file, first removing ONNX Runtime files of a previous install so flavors never mix.
fn put_in_place(res: &Resource, staging: &Path, target: &Path) -> Result<()> {
    let in_use_hint = |e: std::io::Error| {
        anyhow!(e).context(format!(
            "replacing {} (if a running process has its libraries loaded, stop it or restart \
             to finish the update)",
            target.display()
        ))
    };
    if std::fs::symlink_metadata(target).is_err() {
        return std::fs::rename(staging, target).map_err(in_use_hint);
    }
    let owned = read_manifest(&target.join(MANIFEST_FILE)).is_some_and(|m| m.id == res.id);
    if owned {
        let old = sibling(target, &format!("old-{}", std::process::id()))?;
        remove_any(&old)?;
        std::fs::rename(target, &old).map_err(in_use_hint)?;
        if let Err(e) = std::fs::rename(staging, target) {
            let _ = std::fs::rename(&old, target);
            return Err(in_use_hint(e));
        }
        if let Err(e) = remove_any(&old) {
            tracing::warn!(
                "could not remove the previous install {}: {e:#}",
                old.display()
            );
        }
        return Ok(());
    }
    if let ResourceKind::OnnxRuntime(_) = res.kind {
        let windows = res.platform.is_some_and(|p| p.os == "windows");
        if let Ok(rd) = std::fs::read_dir(target) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let is_file = e.file_type().is_ok_and(|t| t.is_file() || t.is_symlink());
                if is_file && extract::is_ort_file(&name, windows) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
    }
    merge_dir(staging, target)?;
    remove_any(staging)
}

fn merge_dir(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to).with_context(|| format!("creating {}", to.display()))?;
    for e in std::fs::read_dir(from).with_context(|| format!("reading {}", from.display()))? {
        let e = e?;
        let src = e.path();
        let dst = to.join(e.file_name());
        let ty = e.file_type()?;
        if ty.is_dir() {
            merge_dir(&src, &dst)?;
        } else {
            if std::fs::symlink_metadata(&dst).is_ok() {
                remove_any(&dst)?;
            }
            std::fs::rename(&src, &dst).with_context(|| format!("moving {}", dst.display()))?;
        }
    }
    Ok(())
}

/// Lowercase hex SHA-256 of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Lowercase hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    sha256_file_until(path, &AtomicBool::new(false))
}

/// [`sha256_file`], giving up with "cancelled" as soon as `stop` is set (checked per MiB).
pub(crate) fn sha256_file_until(path: &Path, stop: &AtomicBool) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        if stop.load(Ordering::Relaxed) {
            bail!("cancelled while hashing {}", path.display());
        }
        let n = f
            .read(&mut buf)
            .with_context(|| format!("reading {}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(
            buf.get(..n)
                .context("read returned more bytes than the buffer holds")?,
        );
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `path` exists with exactly `size` bytes and the pinned hash.
fn file_matches(path: &Path, size: u64, sha256: &str, stop: &AtomicBool) -> Result<bool> {
    match std::fs::metadata(path) {
        Ok(m) if m.is_file() && m.len() == size => Ok(sha256_file_until(path, stop)? == sha256),
        _ => Ok(false),
    }
}

/// Check size and hash of a finished download; delete it on mismatch.
fn verify(path: &Path, part: &crate::resources::catalog::Part, stop: &AtomicBool) -> Result<()> {
    let len = std::fs::metadata(path)
        .with_context(|| format!("reading {}", path.display()))?
        .len();
    if len != part.size {
        let _ = std::fs::remove_file(path);
        bail!(
            "size mismatch for {}: expected {} bytes, got {len}; the file was deleted",
            part.file_name,
            part.size
        );
    }
    let actual = sha256_file_until(path, stop)?;
    if actual != part.sha256 {
        let _ = std::fs::remove_file(path);
        return Err(anyhow!(HashMismatch {
            file: part.file_name.to_string(),
            expected: part.sha256.to_string(),
            actual,
        }));
    }
    Ok(())
}

/// Total size from a `Content-Range: bytes a-b/total` header, when stated.
fn content_range(resp: &reqwest::Response) -> Option<(u64, u64)> {
    let v = resp
        .headers()
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()?;
    let v = v.strip_prefix("bytes ")?;
    let (range, total) = v.split_once('/')?;
    let start = range.split('-').next()?.parse::<u64>().ok()?;
    let total = total.parse::<u64>().ok()?;
    Some((start, total))
}

fn size_mismatch(
    part: &crate::resources::catalog::Part,
    offered: impl std::fmt::Display,
) -> anyhow::Error {
    anyhow!(SizeMismatch {
        file: part.file_name.to_string(),
        expected: part.size,
        offered: offered.to_string(),
    })
}

/// Stream `part` into `partial`, resuming from its current length with a `Range` request.
/// `progress` gets the bytes of this part on disk.
///
/// The response is checked against the pinned size before and while streaming: a
/// `Content-Length` (or `Content-Range` total) that disagrees with it, or a body that keeps
/// coming past it, is a permanent [`SizeMismatch`] (the `.partial` is deleted). A body that
/// stops short is an interrupted transfer, resumed on the next attempt; the whole file is
/// hashed by [`verify`] afterwards, resumed or not.
async fn download_part(
    client: &reqwest::Client,
    part: &crate::resources::catalog::Part,
    partial: &Path,
    stop: &AtomicBool,
    mut progress: impl FnMut(u64),
) -> Result<()> {
    use futures_util::StreamExt;
    check_url(part.url)?;
    let mut have = std::fs::metadata(partial).map_or(0, |m| m.len());
    if have > part.size {
        std::fs::remove_file(partial).ok();
        have = 0;
    }
    if have == part.size {
        return Ok(());
    }
    let mut req = client.get(part.url);
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let resp = req
        .send()
        .await
        .with_context(|| format!("GET {}", part.url))?;
    if !url_allowed(resp.url().as_str()) {
        bail!(UrlRefused(resp.url().to_string()));
    }
    let status = resp.status();
    let append = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        // Only append when the server resumed exactly where we are, in a file of the
        // pinned size.
        let (start, total) = content_range(&resp)
            .with_context(|| format!("GET {}: 206 without a usable Content-Range", part.url))?;
        if total != part.size {
            std::fs::remove_file(partial).ok();
            return Err(size_mismatch(part, total));
        }
        if start != have {
            bail!(
                "GET {}: server resumed at {start}, expected {have}",
                part.url
            );
        }
        true
    } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        std::fs::remove_file(partial).ok();
        bail!("GET {}: range not satisfiable; restarting", part.url);
    } else {
        resp.error_for_status_ref()
            .with_context(|| format!("GET {}", part.url))?;
        false
    };
    if !append {
        have = 0;
    }
    // The announced body must be exactly what is still missing (when it is announced at all:
    // chunked responses have no Content-Length and are checked by the running count).
    if let Some(len) = resp.content_length()
        && len != part.size - have
    {
        std::fs::remove_file(partial).ok();
        return Err(size_mismatch(part, have + len));
    }
    if let Some(parent) = partial.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(partial)
        .with_context(|| format!("opening {}", partial.display()))?;
    progress(have);
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if stop.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let chunk = chunk.with_context(|| format!("downloading {}", part.url))?;
        have = have.saturating_add(chunk.len() as u64);
        if have > part.size {
            drop(f);
            std::fs::remove_file(partial).ok();
            return Err(size_mismatch(part, format!(">{}", part.size)));
        }
        f.write_all(&chunk)
            .with_context(|| format!("writing {}", partial.display()))?;
        crate::metrics::DOWNLOADS
            .bytes
            .fetch_add(chunk.len() as u64, Ordering::Relaxed);
        progress(have);
    }
    f.flush()?;
    if have != part.size {
        bail!(
            "download of {} interrupted at {have} of {} bytes (will resume)",
            part.url,
            part.size
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelFamilyKind;
    use crate::resources::catalog::{Part, Platform, Provides};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;

    #[cfg(unix)]
    fn age_file(p: &Path, by: Duration) {
        let f = std::fs::OpenOptions::new().write(true).open(p).unwrap();
        f.set_modified(SystemTime::now() - by).unwrap();
    }

    /// Unix only: setting a directory's mtime needs `File::open` on a directory.
    #[cfg(unix)]
    #[test]
    fn cleanup_removes_stale_partials_and_staging_only() {
        let root = tmp("cleanup");
        let dl = root.join(".downloads");
        let models = root.join("models");
        std::fs::create_dir_all(&dl).unwrap();
        std::fs::create_dir_all(models.join(".yolo.staging/inner")).unwrap();
        std::fs::create_dir_all(models.join(".yolo.old-123")).unwrap();
        std::fs::create_dir_all(models.join(".fresh.staging")).unwrap();
        std::fs::write(models.join(".yolo.staging/inner/f"), b"x").unwrap();
        let old = dl.join("a.zip.partial");
        let fresh = dl.join("b.zip.partial");
        let other = dl.join("c.zip");
        for p in [&old, &fresh, &other] {
            std::fs::write(p, b"x").unwrap();
        }
        age_file(&old, STALE_PARTIAL_AGE + Duration::from_secs(60));
        age_file(&other, STALE_PARTIAL_AGE * 2);
        for d in [".yolo.staging", ".yolo.old-123"] {
            let f = std::fs::File::open(models.join(d)).unwrap();
            f.set_modified(SystemTime::now() - STALE_STAGING_AGE * 2)
                .unwrap();
        }
        assert_eq!(cleanup_stale(&root, SystemTime::now()), 3);
        assert!(!old.exists());
        assert!(fresh.exists() && other.exists());
        assert!(!models.join(".yolo.staging").exists());
        assert!(!models.join(".yolo.old-123").exists());
        assert!(models.join(".fresh.staging").exists());
        assert_eq!(cleanup_stale(&root, SystemTime::now()), 0);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn backoff_schedule() {
        let s: Vec<u64> = (1..=7).map(|a| backoff(a).as_secs()).collect();
        assert_eq!(s, [30, 60, 120, 240, 300, 300, 300]);
        assert_eq!(backoff(0).as_secs(), 30);
        assert_eq!(backoff(u32::MAX), BACKOFF_MAX);
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-mgr-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    fn sha_hex(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        Sha256::digest(data)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// How the fixture server misbehaves.
    #[derive(Default, Clone)]
    struct Mode {
        /// Cut the first full response off after half the body (to exercise resume).
        cut_first: bool,
        /// Answer the first request with a 302 to this location.
        redirect_first: Option<String>,
        /// Serve a ranged request from this body instead (a file that changed upstream).
        ranged_body: Option<Vec<u8>>,
        /// Send this many extra bytes after the body, announced in `Content-Length`.
        extra_announced: usize,
        /// Send this many extra bytes after the body, not announced.
        extra_silent: usize,
        /// No `Content-Length` at all (body until the connection closes), so only the running
        /// byte count can catch an oversize body.
        no_length: bool,
    }

    /// Minimal HTTP/1.1 file server: honors `Range: bytes=N-`, misbehaves per [`Mode`].
    /// Counts requests and ranged requests.
    struct Server {
        url: String,
        requests: Arc<AtomicUsize>,
        ranged: Arc<AtomicUsize>,
    }

    fn serve(body: Vec<u8>, cut_first: bool) -> Server {
        serve_with(
            body,
            Mode {
                cut_first,
                ..Mode::default()
            },
        )
    }

    fn serve_with(body: Vec<u8>, mode: Mode) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/file.bin", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let ranged = Arc::new(AtomicUsize::new(0));
        let (rq, rg) = (Arc::clone(&requests), Arc::clone(&ranged));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = Vec::new();
                let mut byte = [0u8; 1];
                while !buf.ends_with(b"\r\n\r\n") {
                    if s.read(&mut byte).map_or(true, |n| n == 0) {
                        break;
                    }
                    buf.push(byte[0]);
                }
                let head = String::from_utf8_lossy(&buf).to_ascii_lowercase();
                let n = rq.fetch_add(1, Ordering::SeqCst);
                if n == 0
                    && let Some(loc) = &mode.redirect_first
                {
                    let _ = write!(
                        s,
                        "HTTP/1.1 302 Found\r\nLocation: {loc}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    let _ = s.flush();
                    continue;
                }
                let start = head
                    .lines()
                    .find_map(|l| l.strip_prefix("range: bytes="))
                    .and_then(|r| r.trim().trim_end_matches('-').parse::<usize>().ok());
                let (status, slice, extra) = match start {
                    Some(st) => {
                        rg.fetch_add(1, Ordering::SeqCst);
                        let src = mode.ranged_body.as_deref().unwrap_or(&body);
                        (
                            "206 Partial Content",
                            &src[st.min(src.len())..],
                            format!(
                                "Content-Range: bytes {st}-{}/{}\r\n",
                                src.len().saturating_sub(1),
                                src.len()
                            ),
                        )
                    }
                    None => ("200 OK", &body[..], String::new()),
                };
                let length = if mode.no_length {
                    String::new()
                } else {
                    format!("Content-Length: {}\r\n", slice.len() + mode.extra_announced)
                };
                let _ = write!(
                    s,
                    "HTTP/1.1 {status}\r\n{length}{extra}Connection: close\r\n\r\n"
                );
                let send = if mode.cut_first && n == 0 {
                    &slice[..slice.len() / 2]
                } else {
                    slice
                };
                let _ = s.write_all(send);
                let pad = vec![0u8; mode.extra_announced + mode.extra_silent];
                let _ = s.write_all(&pad);
                let _ = s.flush();
            }
        });
        Server {
            url,
            requests,
            ranged,
        }
    }

    fn model_resource(url: &str, data: &[u8], sha: Option<String>) -> &'static Resource {
        let part = Part {
            url: leak(url.to_string()),
            sha256: leak(sha.unwrap_or_else(|| sha_hex(data))),
            size: data.len() as u64,
            file_name: "test-model.onnx",
            archive: None,
            layout: Layout::File,
        };
        Box::leak(Box::new(Resource {
            id: "model:test-model",
            kind: ResourceKind::Model,
            version: "1",
            platform: None,
            parts: Box::leak(vec![part].into_boxed_slice()),
            dest: "models",
            provides: Provides::Model {
                name: "test-model",
                family: ModelFamilyKind::Yolo5,
            },
            title: "Model test-model",
            description: "test",
            license: "MIT",
        }))
    }

    fn body() -> Vec<u8> {
        (0..300_000u32).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn download_resumes_after_an_interrupted_transfer() {
        let data = body();
        let server = serve(data.clone(), true);
        let res = model_resource(&server.url, &data, None);
        let root = tmp("resume");
        let opts = ManagerOptions {
            backoff: |_| Duration::from_millis(10),
            retry: true,
            ..ManagerOptions::new(&root)
        };
        let m = Manager::start(opts);
        let events = m.subscribe();
        let key = m.enqueue(Job::into_dir(res, root.join("models")));
        // First attempt fails half way (retry scheduled), the second resumes with Range.
        let first = events.recv_timeout(Duration::from_secs(20)).unwrap();
        assert!(
            matches!(
                first,
                Event::Failed {
                    retry_at: Some(_),
                    ..
                }
            ),
            "{first:?}"
        );
        let second = events.recv_timeout(Duration::from_secs(20)).unwrap();
        assert!(matches!(second, Event::Installed { .. }), "{second:?}");
        assert_eq!(m.status(&key).unwrap().state, State::Installed);
        assert_eq!(
            server.ranged.load(Ordering::SeqCst),
            1,
            "resumed with Range"
        );
        assert_eq!(
            std::fs::read(root.join("models/test-model.onnx")).unwrap(),
            data
        );
        assert!(!root.join("models/test-model.onnx.partial").exists());
        assert!(is_installed(res, &root.join("models")));
        // Manifest round trip.
        let man = read_manifest(&manifest_path(res, &root.join("models"))).unwrap();
        assert_eq!(man.id, "model:test-model");
        assert_eq!(man.files, ["test-model.onnx"]);
        assert_eq!(man.sha256, [res.parts[0].sha256]);
        // Installed: queueing again does nothing.
        let before = server.requests.load(Ordering::SeqCst);
        m.enqueue(Job::into_dir(res, root.join("models")));
        m.shutdown();
        assert_eq!(server.requests.load(Ordering::SeqCst), before);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn hash_mismatch_deletes_the_file_and_is_not_retried() {
        let data = body();
        let server = serve(data.clone(), false);
        let res = model_resource(&server.url, &data, Some("0".repeat(64)));
        let root = tmp("hash");
        let opts = ManagerOptions {
            retry: true,
            ..ManagerOptions::new(&root)
        };
        let m = Manager::start(opts);
        let key = m.enqueue(Job::into_dir(res, root.join("models")));
        let st = m.wait(std::slice::from_ref(&key), Some(Duration::from_secs(20)));
        match &st[0].state {
            State::Failed {
                error, retry_at, ..
            } => {
                assert!(error.contains("hash mismatch"), "{error}");
                assert_eq!(*retry_at, None, "no automatic retry");
            }
            other => panic!("{other:?}"),
        }
        let models = root.join("models");
        assert!(!models.join("test-model.onnx").exists());
        assert!(!models.join("test-model.onnx.partial").exists());
        assert!(!is_installed(res, &models));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn lock_contention_and_install_all() {
        let root = tmp("lock");
        let held = DownloadLock::try_acquire(&root)
            .unwrap()
            .expect("free lock");
        assert!(
            DownloadLock::try_acquire(&root).unwrap().is_none(),
            "a second handle must not get the lock"
        );
        let stop = AtomicBool::new(false);
        let err =
            DownloadLock::acquire(&root, Some(Duration::from_millis(300)), &stop).unwrap_err();
        assert!(err.downcast_ref::<LockBusy>().is_some());

        // A job waits for the lock and fails the attempt when it stays busy.
        let data = body();
        let server = serve(data.clone(), false);
        let res = model_resource(&server.url, &data, None);
        let opts = ManagerOptions {
            lock_wait: Some(Duration::from_millis(300)),
            ..ManagerOptions::new(&root)
        };
        let err = install_all(opts.clone(), vec![Job::into_dir(res, root.join("m"))]).unwrap_err();
        assert!(format!("{err:#}").contains("download lock"), "{err:#}");
        assert_eq!(server.requests.load(Ordering::SeqCst), 0);
        drop(held);
        let st = install_all(opts, vec![Job::into_dir(res, root.join("m"))]).unwrap();
        assert_eq!(st[0].state, State::Installed);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn https_only() {
        assert!(check_url("https://example.com/x").is_ok());
        assert!(check_url("http://127.0.0.1:8080/x").is_ok());
        let e = check_url("http://example.com/x").unwrap_err();
        assert!(!retryable(&e));
        assert!(retryable(&anyhow!("connection reset")));
    }

    /// Run one job without retries; the final state.
    fn run_once(res: &'static Resource, root: &Path) -> State {
        let m = Manager::start(ManagerOptions {
            retry: true,
            backoff: |_| Duration::from_millis(10),
            ..ManagerOptions::new(root)
        });
        let key = m.enqueue(Job::into_dir(res, root.join("models")));
        let st = m.wait(std::slice::from_ref(&key), Some(Duration::from_secs(20)));
        m.shutdown();
        st[0].state.clone()
    }

    fn failed_permanently(state: &State, text: &str) {
        match state {
            State::Failed {
                error, retry_at, ..
            } => {
                assert!(error.contains(text), "expected {text:?} in: {error}");
                assert_eq!(*retry_at, None, "must not be retried: {error}");
            }
            other => panic!("{other:?}"),
        }
    }

    fn nothing_left(root: &Path) {
        let models = root.join("models");
        assert!(!models.join("test-model.onnx").exists());
        assert!(!models.join("test-model.onnx.partial").exists());
    }

    /// RS-01: a redirect to a non-https host is refused (permanently); one to the loopback
    /// test server is followed (test build only).
    #[test]
    fn redirects_off_https_are_refused() {
        let data = body();
        let server = serve_with(
            data.clone(),
            Mode {
                redirect_first: Some("http://example.com/file.bin".into()),
                ..Mode::default()
            },
        );
        let res = model_resource(&server.url, &data, None);
        let root = tmp("redirect-http");
        failed_permanently(&run_once(res, &root), "redirect");
        nothing_left(&root);
        assert_eq!(server.requests.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(&root);

        // Loopback http is allowed in this build; the final URL is checked too.
        let target = serve(data.clone(), false);
        let front = serve_with(
            Vec::new(),
            Mode {
                redirect_first: Some(target.url.clone()),
                ..Mode::default()
            },
        );
        let res = model_resource(&front.url, &data, None);
        let root = tmp("redirect-ok");
        assert_eq!(run_once(res, &root), State::Installed);
        assert_eq!(target.requests.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(&root);
        assert!(url_allowed("https://x/y") && !url_allowed("http://x/y"));
        assert!(!url_allowed("ftp://127.0.0.1:1/x"));
    }

    /// RS-05: a `Content-Length` larger than the pin fails before streaming; a body without
    /// a length that keeps coming past the pin fails during; neither is retried, nothing
    /// stays on disk.
    #[test]
    fn oversize_bodies_fail_permanently() {
        let data = body();
        for mode in [
            Mode {
                extra_announced: 100,
                ..Mode::default()
            },
            Mode {
                extra_silent: 100,
                no_length: true,
                ..Mode::default()
            },
        ] {
            let server = serve_with(data.clone(), mode);
            let res = model_resource(&server.url, &data, None);
            let root = tmp("oversize");
            failed_permanently(&run_once(res, &root), "size mismatch");
            nothing_left(&root);
            assert_eq!(server.requests.load(Ordering::SeqCst), 1, "no retry");
            let _ = std::fs::remove_dir_all(&root);
        }
        // A pin smaller than the file: Content-Length catches it before any byte is read.
        let server = serve(data.clone(), false);
        let mut part = *model_resource(&server.url, &data, None)
            .parts
            .first()
            .unwrap();
        part.size = 10;
        let res: &'static Resource = Box::leak(Box::new(Resource {
            parts: Box::leak(vec![part].into_boxed_slice()),
            ..*model_resource(&server.url, &data, None)
        }));
        let root = tmp("oversize-pin");
        failed_permanently(&run_once(res, &root), "size mismatch");
        nothing_left(&root);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// RS-15: a resumed download is hashed as a whole; when the upstream file changed between
    /// the two halves the stitched file is rejected and deleted (a 206 with another total is
    /// a size mismatch; the same size with other bytes is a hash mismatch).
    #[test]
    fn resumed_download_is_hashed_as_a_whole() {
        let data = body();
        let mut other = data.clone();
        for b in other.iter_mut().skip(data.len() / 2 + 7).take(64) {
            *b ^= 0xff;
        }
        let server = serve_with(
            data.clone(),
            Mode {
                cut_first: true,
                ranged_body: Some(other),
                ..Mode::default()
            },
        );
        let res = model_resource(&server.url, &data, None);
        let root = tmp("resume-hash");
        failed_permanently(&run_once(res, &root), "hash mismatch");
        nothing_left(&root);
        assert_eq!(server.ranged.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(&root);

        // Another total in the Content-Range: refused before streaming, partial removed.
        let mut longer = data.clone();
        longer.extend_from_slice(&[1u8; 1000]);
        let server = serve_with(
            data.clone(),
            Mode {
                cut_first: true,
                ranged_body: Some(longer),
                ..Mode::default()
            },
        );
        let res = model_resource(&server.url, &data, None);
        let root = tmp("resume-size");
        failed_permanently(&run_once(res, &root), "size mismatch");
        nothing_left(&root);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// RS-04: a worker that dies (here: panics on a job) fails every job it will never run,
    /// later enqueues fail at once, and `wait(None)` returns instead of hanging.
    #[test]
    fn dead_worker_fails_jobs_instead_of_leaving_them_queued() {
        let data = body();
        let root = tmp("dead");
        let panicking: &'static Resource = Box::leak(Box::new(Resource {
            id: TEST_PANIC_ID,
            ..*model_resource("http://127.0.0.1:9/x", &data, None)
        }));
        let other = model_resource("http://127.0.0.1:9/y", &data, None);
        let m = Manager::start(ManagerOptions {
            retry: true,
            ..ManagerOptions::new(&root)
        });
        let events = m.subscribe();
        let k1 = m.enqueue(Job::into_dir(panicking, root.join("a")));
        let k2 = m.enqueue(Job::into_dir(other, root.join("b")));
        let st = m.wait(&[k1.clone(), k2.clone()], None);
        assert_eq!(st.len(), 2);
        for s in &st {
            match &s.state {
                State::Failed {
                    error, retry_at, ..
                } => {
                    assert!(error.contains("download worker stopped"), "{error}");
                    assert!(error.contains("panicked"), "{error}");
                    assert_eq!(*retry_at, None);
                }
                other => panic!("{other:?}"),
            }
        }
        assert!(!m.is_alive());
        let ev = events.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(ev, Event::Failed { retry_at: None, .. }), "{ev:?}");
        // A later job fails at once.
        let k3 = m.enqueue(Job::into_dir(other, root.join("c")));
        assert!(matches!(
            m.status(&k3).unwrap().state,
            State::Failed { retry_at: None, .. }
        ));
        m.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// RS-06: with `max_attempts` the service stops retrying an unreachable URL and names the
    /// manual command.
    #[test]
    fn retries_are_bounded_by_max_attempts() {
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let data = body();
        let res = model_resource(&format!("http://127.0.0.1:{port}/gone"), &data, None);
        let root = tmp("bounded");
        let m = Manager::start(ManagerOptions {
            retry: true,
            backoff: |_| Duration::from_millis(10),
            max_attempts: Some(3),
            ..ManagerOptions::new(&root)
        });
        let events = m.subscribe();
        let key = m.enqueue(Job::into_dir(res, root.join("models")));
        let st = m.wait(std::slice::from_ref(&key), Some(Duration::from_secs(30)));
        match &st[0].state {
            State::Failed {
                error,
                retry_at,
                attempts,
            } => {
                assert_eq!(*retry_at, None);
                assert_eq!(*attempts, 3);
                assert!(error.contains("gave up after 3 attempts"), "{error}");
                assert!(
                    error.contains("blue-onyx-prism fetch --resource model:test-model"),
                    "{error}"
                );
            }
            other => panic!("{other:?}"),
        }
        // `set` runs before `emit`, so `wait` can return before the last event is delivered:
        // receive the three attempts with a deadline instead of draining what is there now.
        let mut retried = Vec::new();
        while retried.len() < 3 {
            match events.recv_timeout(Duration::from_secs(10)) {
                Ok(ev) => retried.push(ev),
                Err(e) => panic!("{e}: {retried:?}"),
            }
        }
        assert!(
            events.recv_timeout(Duration::from_millis(200)).is_err(),
            "more than 3 attempts were reported"
        );
        // Queuing it again (the Download button) starts a new series.
        m.enqueue(Job::into_dir(res, root.join("models")));
        assert!(!m.status(&key).unwrap().state.is_final());
        m.shutdown();
        let _ = std::fs::remove_dir_all(&root);
    }

    /// RS-10: hashing stops as soon as the stop flag is set.
    #[test]
    fn hashing_stops_on_the_stop_flag() {
        let root = tmp("hash-stop");
        let f = root.join("big.bin");
        std::fs::write(&f, vec![7u8; 3 << 20]).unwrap();
        let stop = AtomicBool::new(false);
        assert_eq!(
            sha256_file_until(&f, &stop).unwrap(),
            sha256_file(&f).unwrap()
        );
        stop.store(true, Ordering::SeqCst);
        let e = sha256_file_until(&f, &stop).unwrap_err();
        assert!(format!("{e:#}").contains("cancelled"), "{e:#}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fetch_command_adds_allow_large_for_large_resources() {
        let data = body();
        let small = model_resource("https://x/y", &data, None);
        assert_eq!(
            fetch_command(small),
            "blue-onyx-prism fetch --resource model:test-model"
        );
        let mut part = small.parts[0];
        part.size = crate::resources::catalog::LARGE_DOWNLOAD_BYTES + 1;
        let large: &'static Resource = Box::leak(Box::new(Resource {
            parts: Box::leak(vec![part].into_boxed_slice()),
            ..*small
        }));
        assert!(fetch_command(large).ends_with(" --allow-large"));
    }

    #[test]
    fn content_range_parsing() {
        let r = |v: &str| {
            let resp = axum::http::Response::builder()
                .header("Content-Range", v)
                .body("")
                .unwrap();
            content_range(&reqwest::Response::from(resp))
        };
        assert_eq!(r("bytes 100-299/300"), Some((100, 300)));
        assert_eq!(r("bytes 0-0/*"), None);
        assert_eq!(r("items 1-2/3"), None);
        assert_eq!(r("bytes x-2/3"), None);
    }

    /// An ONNX Runtime archive from local files: extracted into staging, swapped into place,
    /// manifest + flavor.txt written, marked active; a second install replaces the first.
    #[test]
    fn local_archive_install_and_swap() {
        let root = tmp("swap");
        let archive = root.join("ort.tgz");
        let make = |content: &[u8]| {
            let f = std::fs::File::create(&archive).unwrap();
            let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
            let mut tar = tar::Builder::new(gz);
            let mut h = tar::Header::new_gnu();
            h.set_size(content.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            tar.append_data(
                &mut h,
                "onnxruntime-linux-x64-1.24.4/lib/libonnxruntime.so",
                content,
            )
            .unwrap();
            tar.into_inner().unwrap().finish().unwrap();
        };
        let res = crate::resources::catalog::onnxruntime(
            "linux",
            "x86_64",
            crate::resources::Flavor::Cpu,
        )
        .unwrap();
        assert_eq!(res.platform, Some(Platform::LINUX_X64));
        let job = || Job {
            local_archives: Some(vec![archive.clone()]),
            activate_ort: true,
            ..Job::new(res, &root)
        };
        make(b"v1");
        install_all(ManagerOptions::new(&root), vec![job()]).unwrap();
        let dir = root.join("onnxruntime/cpu");
        assert_eq!(std::fs::read(dir.join("libonnxruntime.so")).unwrap(), b"v1");
        assert_eq!(
            std::fs::read_to_string(dir.join("flavor.txt"))
                .unwrap()
                .trim(),
            "cpu"
        );
        let man = read_manifest(&dir.join(MANIFEST_FILE)).unwrap();
        assert_eq!(man.id, "onnxruntime-cpu");
        assert!(man.files.contains(&"libonnxruntime.so".to_string()));
        assert_eq!(
            crate::backend::libs::read_active_flavor(&root.join("onnxruntime")).as_deref(),
            Some("cpu")
        );
        make(b"v2");
        install_all(ManagerOptions::new(&root), vec![job()]).unwrap();
        assert_eq!(std::fs::read(dir.join("libonnxruntime.so")).unwrap(), b"v2");
        let leftovers: Vec<_> = std::fs::read_dir(root.join("onnxruntime"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with('.'))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
