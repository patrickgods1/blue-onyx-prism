//! SAM 3's gated download, for the Auto-label tab's panel. Admin only.
//!
//! - `GET /v1/autolabel/sam3`: the state ([`Sam3State`]), where the token comes from (its source,
//!   never its value), the login command, the size and the progress. No network.
//! - `POST /v1/autolabel/sam3/download` is the user's "I have access", so it is also the opt-in to
//!   the 3.4 GB. In order: files already there at their pinned sizes are queued straight away, with
//!   no token and no large-download gate, so the manager adopts the folder by hash without a request;
//!   otherwise one request for the pinned `config.json` answers `needs_login` or `needs_access` at
//!   once, then free space is checked, then the download is queued.
//! - `POST /v1/autolabel/sam3/remove`: refused while a run uses the folder.
//!
//! A refusal is remembered per data root and token, without keeping the token (only its hash), so
//! the panel can still show step 3 after a reload and a new login clears it.

use super::{AppState, blocking};
use crate::config::Config;
use crate::resources::catalog::{self, Gated, Resource};
use crate::resources::export::train::TrainPaths;
use crate::resources::hf_token::{self, Access, HfToken, Lookup, Refusal};
use crate::resources::manager::{self, Job, Manager, State as JobState};
use crate::sync::LockExt;
use crate::training::pyenv;
use crate::training::task::{TaskKind, TaskRecord};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Where SAM 3 stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Sam3State {
    NotRequested,
    NeedsLogin,
    TokenUnreadable,
    NeedsAccess,
    Checking,
    Downloading,
    Installed,
    Failed,
}

/// The answer of all three routes. Holds the token's source, never its value.
#[derive(Debug, Serialize)]
pub(super) struct Sam3Status {
    success: bool,
    state: Sam3State,
    message: String,
    /// Where the token in use comes from; None without one.
    token_source: Option<&'static str>,
    /// `HF_TOKEN` hides a login on disk, so signing in again changes nothing here.
    token_env_overrides: bool,
    command: String,
    command_note: String,
    size: u64,
    size_text: String,
    /// Percent while downloading.
    progress: Option<u8>,
    /// Every file is here at its pinned size: no login needed.
    parts_on_disk: bool,
    /// An auto-label run uses the folder: Remove is refused.
    in_use: bool,
    repo_url: &'static str,
    license: &'static str,
    /// macOS: SAM 3 was never measured on an Apple GPU.
    apple: bool,
}

/// What the routes act on. The real one is SAM 3 on Hugging Face; tests give it a fixture.
struct Target {
    res: &'static Resource,
    gated: &'static Gated,
    /// The host that gets the token.
    host: String,
    lookup: Box<dyn Fn() -> Lookup + Send + Sync>,
}

impl Target {
    fn real(root: PathBuf) -> Option<Arc<Self>> {
        let res = hf_token::sam3_resource()?;
        Some(Arc::new(Self {
            res,
            gated: catalog::gated(res.id)?,
            host: hf_token::HF_HOST.to_string(),
            lookup: Box::new(move || hf_token::lookup(&root)),
        }))
    }
}

/// Last refusal per data root: the refusal and the hash of the token it was for.
type Refused = BTreeMap<PathBuf, (Refusal, Option<String>)>;
static REFUSED: Mutex<Refused> = Mutex::new(BTreeMap::new());

fn remember_refusal(root: &Path, refusal: Refusal, token: Option<&HfToken>) {
    let hash = token.map(HfToken::fingerprint);
    REFUSED
        .lock_recover()
        .insert(root.to_path_buf(), (refusal, hash));
}

fn clear_refusal(root: &Path) {
    REFUSED.lock_recover().remove(root);
}

/// The remembered refusal, if it was for the token in use now.
fn last_refusal(root: &Path, token: Option<&HfToken>) -> Option<Refusal> {
    let hash = token.map(HfToken::fingerprint);
    REFUSED
        .lock_recover()
        .get(root)
        .filter(|(_, was)| *was == hash)
        .map(|(r, _)| *r)
}

/// What [`state_of`] decides from.
struct Facts<'a> {
    installed: bool,
    parts_on_disk: bool,
    job: Option<&'a JobState>,
    lookup: &'a Lookup,
    refused: Option<Refusal>,
}

/// A job that is doing something now, not waiting for a retry.
fn running(s: &JobState) -> bool {
    matches!(
        s,
        JobState::Queued
            | JobState::Downloading { .. }
            | JobState::Verifying
            | JobState::Extracting
    )
}

/// The state and its sentence. A folder on disk needs no login, so the token is only looked at
/// when there is nothing to adopt.
fn state_of(f: &Facts<'_>, gated: &Gated) -> (Sam3State, String) {
    if f.installed {
        return (Sam3State::Installed, "SAM 3 is installed.".to_string());
    }
    match f.job {
        Some(JobState::Verifying) => return checking(),
        Some(JobState::Queued) if f.parts_on_disk => return checking(),
        Some(s @ JobState::Failed { retry_at: None, .. }) => {
            return (Sam3State::Failed, s.describe());
        }
        Some(s) if !matches!(s, JobState::Installed) => {
            return (Sam3State::Downloading, s.describe());
        }
        _ => {}
    }
    if f.parts_on_disk {
        return (
            Sam3State::NotRequested,
            "The files are already in this folder. The button checks them by hash; nothing is \
             downloaded."
                .to_string(),
        );
    }
    if let Some(problem) = f.lookup.problem() {
        return (Sam3State::TokenUnreadable, problem);
    }
    let source = f.lookup.token().map(HfToken::source);
    match f.refused {
        Some(r) => (
            match r {
                Refusal::NeedsLogin => Sam3State::NeedsLogin,
                Refusal::NeedsAccess => Sam3State::NeedsAccess,
            },
            hf_token::refusal_message(r, gated, source),
        ),
        None => (Sam3State::NotRequested, String::new()),
    }
}

fn checking() -> (Sam3State, String) {
    (
        Sam3State::Checking,
        "Checking the files against their pinned hashes; this reads all 3.4 GB and can take minutes.".to_string(),
    )
}

/// A queued or running auto-label run with SAM 3 reads the checkpoint.
fn run_uses_folder(tasks: &[TaskRecord], res: &Resource) -> bool {
    tasks.iter().any(|t| {
        t.kind == TaskKind::Autolabel && t.state.is_busy() && t.uses.iter().any(|u| u == res.id)
    })
}

fn folder_of(cfg: &Config, res: &Resource) -> PathBuf {
    pyenv::weights_dir(&TrainPaths::for_config(cfg), res)
}

fn job_status(state: &AppState, key: &str) -> Option<JobState> {
    let manager = state.resources.as_ref()?.provisioner.manager()?;
    manager.status(key).map(|s| s.state)
}

/// The panel's view now. Reads the disk (blocking).
fn view(state: &AppState, t: &Target, tasks: &[TaskRecord]) -> Sam3Status {
    let cfg = state.config_read().clone();
    let root = cfg.data_root();
    let dir = folder_of(&cfg, t.res);
    let job = job_status(state, &Job::into_dir(t.res, dir.clone()).key());
    let lookup = (t.lookup)();
    let parts_on_disk = hf_token::parts_present(t.res, &dir);
    let facts = Facts {
        installed: manager::is_installed(t.res, &dir),
        parts_on_disk,
        job: job.as_ref(),
        lookup: &lookup,
        refused: last_refusal(&root, lookup.token()),
    };
    let (status, message) = state_of(&facts, t.gated);
    let progress = match &job {
        Some(JobState::Downloading { bytes, total }) if status == Sam3State::Downloading => {
            Some(manager::percent(*bytes, *total))
        }
        _ => None,
    };
    let command = hf_token::login_command_for(&state.config_path);
    Sam3Status {
        success: true,
        state: status,
        message,
        token_source: lookup.token().map(|t| t.source().label()),
        token_env_overrides: hf_token::env_overrides_login(&root),
        command: command.command,
        command_note: command.note,
        size: t.res.size(),
        size_text: catalog::format_size(t.res.size()),
        progress,
        parts_on_disk,
        in_use: run_uses_folder(tasks, t.res),
        repo_url: t.gated.repo_url,
        license: t.gated.license,
        apple: std::env::consts::OS == "macos",
    }
}

async fn view_now(state: &Arc<AppState>, t: &Arc<Target>) -> Sam3Status {
    let (state, t) = (state.clone(), t.clone());
    blocking(move || {
        let tasks = super::training::tasks(&state);
        view(&state, &t, &tasks)
    })
    .await
}

fn reply(code: StatusCode, mut body: Sam3Status) -> Response {
    body.success = code.is_success();
    (code, Json(body)).into_response()
}

/// A failure of the route itself, shown as `failed` over the state of the moment.
fn failure(code: StatusCode, mut body: Sam3Status, message: String) -> Response {
    tracing::warn!("SAM 3 download not started: {message}");
    body.state = Sam3State::Failed;
    body.message = message;
    reply(code, body)
}

fn plain(code: StatusCode, message: &str) -> Response {
    (
        code,
        Json(serde_json::json!({"success": false, "message": message})),
    )
        .into_response()
}

fn no_pin() -> Response {
    plain(
        StatusCode::INTERNAL_SERVER_ERROR,
        "this build has no SAM 3 pin; update Blue Onyx Prism",
    )
}

/// `GET /v1/autolabel/sam3`.
pub(super) async fn status(State(state): State<Arc<AppState>>) -> Response {
    let root = state.config_read().data_root();
    let Some(t) = Target::real(root) else {
        return no_pin();
    };
    let body = view_now(&state, &t).await;
    reply(StatusCode::OK, body)
}

/// `POST /v1/autolabel/sam3/download`.
pub(super) async fn download(State(state): State<Arc<AppState>>) -> Response {
    let root = state.config_read().data_root();
    let Some(t) = Target::real(root) else {
        return no_pin();
    };
    download_with(state, t).await
}

/// Bytes still to fetch into `dir`: a file at its pinned size counts as there (the manager hashes
/// it), a `.partial` as far as it got.
fn missing_bytes(res: &Resource, dir: &Path) -> u64 {
    let len = |p: PathBuf| std::fs::metadata(p).map_or(0, |m| m.len());
    res.parts
        .iter()
        .map(|p| {
            if len(dir.join(p.file_name)) == p.size {
                0
            } else {
                p.size
                    .saturating_sub(len(dir.join(format!("{}.partial", p.file_name))))
            }
        })
        .sum()
}

/// Free bytes on the volume `dir` will be on (it may not exist yet). None: unknown.
fn free_bytes(dir: &Path) -> Option<u64> {
    let existing = dir.ancestors().find(|p| p.exists())?;
    fs4::available_space(existing).ok()
}

/// What is wrong with fetching `missing` bytes into `dir` with `free` available.
fn space_problem(missing: u64, free: Option<u64>, dir: &Path) -> Option<String> {
    let free = free?;
    (free < missing).then(|| {
        format!(
            "not enough disk space: SAM 3 needs {} more in {} and {} is free. Free some space or \
             move the data folder (download_dir), then try again",
            catalog::format_size(missing),
            dir.display(),
            catalog::format_size(free)
        )
    })
}

async fn download_with(state: Arc<AppState>, t: Arc<Target>) -> Response {
    let cfg = state.config_read().clone();
    let Some(ctx) = state.resources.as_ref() else {
        return plain(
            StatusCode::SERVICE_UNAVAILABLE,
            "this server has no download manager; run the service",
        );
    };
    let provisioner = ctx.provisioner.clone();
    let (manager, installed, present, dir): (Arc<Manager>, bool, bool, PathBuf) = {
        let (cfg, res) = (cfg.clone(), t.res);
        blocking(move || {
            let dir = folder_of(&cfg, res);
            (
                provisioner.manager_for(&cfg),
                manager::is_installed(res, &dir),
                hf_token::parts_present(res, &dir),
                dir,
            )
        })
        .await
    };
    let job = Job::into_dir(t.res, dir.clone());
    let key = job.key();
    let busy = manager.status(&key).is_some_and(|s| running(&s.state));
    if installed || busy {
        return reply(StatusCode::OK, view_now(&state, &t).await);
    }
    if !present {
        let root = cfg.data_root();
        let lookup = (t.lookup)();
        if lookup.problem().is_some() {
            return reply(StatusCode::CONFLICT, view_now(&state, &t).await);
        }
        let Some(url) = hf_token::probe_url(t.res) else {
            return no_pin();
        };
        let client = match manager::http_client() {
            Ok(c) => c,
            Err(e) => {
                let body = view_now(&state, &t).await;
                return failure(StatusCode::INTERNAL_SERVER_ERROR, body, format!("{e:#}"));
            }
        };
        match hf_token::probe(&client, url, lookup.token(), &t.host).await {
            Ok(Access::Granted) => clear_refusal(&root),
            Ok(Access::Refused(r)) => {
                remember_refusal(&root, r, lookup.token());
                return reply(StatusCode::CONFLICT, view_now(&state, &t).await);
            }
            Err(e) => {
                let body = view_now(&state, &t).await;
                return failure(
                    StatusCode::BAD_GATEWAY,
                    body,
                    format!("could not reach Hugging Face: {e:#}"),
                );
            }
        }
        let (res, target) = (t.res, dir.clone());
        let problem = blocking(move || {
            space_problem(missing_bytes(res, &target), free_bytes(&target), &target)
        })
        .await;
        if let Some(problem) = problem {
            let body = view_now(&state, &t).await;
            return failure(StatusCode::INSUFFICIENT_STORAGE, body, problem);
        }
    }
    // A remembered "installed" from before the files went, or an old failure, would make the
    // enqueue a no-op or leave its text on the panel.
    manager.forget(&key);
    manager.enqueue(job);
    reply(StatusCode::OK, view_now(&state, &t).await)
}

/// `POST /v1/autolabel/sam3/remove`.
pub(super) async fn remove(State(state): State<Arc<AppState>>) -> Response {
    let root = state.config_read().data_root();
    let Some(t) = Target::real(root) else {
        return no_pin();
    };
    blocking(move || {
        let tasks = super::training::tasks(&state);
        remove_with(&state, &t, &tasks)
    })
    .await
}

fn remove_with(state: &AppState, t: &Target, tasks: &[TaskRecord]) -> Response {
    let cfg = state.config_read().clone();
    let dir = folder_of(&cfg, t.res);
    let refuse = |code, message: String| {
        let mut body = view(state, t, tasks);
        body.message = message;
        reply(code, body)
    };
    if run_uses_folder(tasks, t.res) {
        return refuse(
            StatusCode::CONFLICT,
            "an auto-label run with SAM 3 is queued or running; wait for it or cancel it, then \
             remove"
                .to_string(),
        );
    }
    let key = Job::into_dir(t.res, dir.clone()).key();
    let manager = state
        .resources
        .as_ref()
        .and_then(|c| c.provisioner.manager());
    if manager
        .as_ref()
        .and_then(|m| m.status(&key))
        .is_some_and(|s| running(&s.state))
    {
        return refuse(
            StatusCode::CONFLICT,
            "SAM 3 is downloading or being checked; wait for it to finish, then remove".to_string(),
        );
    }
    if let Err(e) = manager::remove_any(&dir) {
        return refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e:#}; if another program has the files open, close it and try again"),
        );
    }
    if let Some(m) = manager {
        m.forget(&key);
    }
    clear_refusal(&cfg.data_root());
    let mut body = view(state, t, tasks);
    body.message = format!("Removed {}.", dir.display());
    reply(StatusCode::OK, body)
}

#[cfg(test)]
mod tests {
    use super::super::test_util::*;
    use super::*;
    use crate::resources::catalog::{Layout, Part, Provides, ResourceKind};
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    const TOKEN: &str = "hf_NotARealTokenOnlyForRouteTests";
    const GATED_401: &str = "401 Unauthorized\r\nX-Error-Code: GatedRepo";
    const GATED_403: &str = "403 Forbidden\r\nX-Error-Code: GatedRepo";

    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    /// Answers every request with `status` and the file named by the path; counts requests.
    fn server(
        status: &'static str,
        files: Vec<(&'static str, Vec<u8>)>,
    ) -> (String, Arc<AtomicUsize>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let seen = hits.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut head = Vec::new();
                let mut b = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    if s.read(&mut b).map_or(true, |n| n == 0) {
                        break;
                    }
                    head.push(b[0]);
                }
                seen.fetch_add(1, Ordering::SeqCst);
                let head = String::from_utf8_lossy(&head).to_string();
                let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
                let body = files
                    .iter()
                    .find(|(n, _)| path == format!("/{n}"))
                    .map_or(&[][..], |(_, b)| b.as_slice());
                let _ = write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(body);
            }
        });
        (base, hits)
    }

    const CONFIG: &[u8] = b"{\"model_type\":\"sam3\"}";
    const WEIGHTS: &[u8] = b"not really weights";

    /// The SAM 3 id (it is what marks a job gated) over files at `base`.
    fn fixture(base: &str) -> &'static Resource {
        let part = |name: &'static str, data: &[u8]| Part {
            url: leak(format!("{base}/{name}")),
            sha256: leak(manager::sha256_hex(data)),
            size: data.len() as u64,
            file_name: name,
            archive: None,
            layout: Layout::File,
        };
        let parts = vec![
            part("config.json", CONFIG),
            part("model.safetensors", WEIGHTS),
        ];
        Box::leak(Box::new(Resource {
            id: catalog::SAM3_ID,
            kind: ResourceKind::ExportModel,
            version: "hf-test",
            platform: None,
            parts: Box::leak(parts.into_boxed_slice()),
            dest: catalog::TRAIN_WEIGHTS_DEST,
            provides: Provides::Tool {
                name: "autolabel-sam3",
            },
            title: "SAM 3",
            description: "fixture",
            license: catalog::SAM_LICENSE,
        }))
    }

    fn target(res: &'static Resource, lookup: Lookup) -> Arc<Target> {
        Arc::new(Target {
            res,
            gated: catalog::gated(catalog::SAM3_ID).unwrap(),
            host: "127.0.0.1".to_string(),
            lookup: Box::new(move || lookup.clone()),
        })
    }

    fn found() -> Lookup {
        Lookup::Found(HfToken::new(TOKEN, hf_token::TokenSource::Saved).unwrap())
    }

    fn put_files(state: &AppState, res: &Resource) -> PathBuf {
        let dir = folder_of(&state.config_read(), res);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), CONFIG).unwrap();
        std::fs::write(dir.join("model.safetensors"), WEIGHTS).unwrap();
        dir
    }

    async fn body_of(resp: Response) -> (StatusCode, String, serde_json::Value) {
        let code = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let json = serde_json::from_str(&text).unwrap();
        (code, text, json)
    }

    /// Wait for the job of `res` to end and answer the view.
    async fn settled(state: &Arc<AppState>, t: &Arc<Target>) -> serde_json::Value {
        let dir = folder_of(&state.config_read(), t.res);
        let key = Job::into_dir(t.res, dir).key();
        let manager = state
            .resources
            .as_ref()
            .unwrap()
            .provisioner
            .manager()
            .unwrap();
        let keys = [key];
        tokio::task::spawn_blocking(move || manager.wait(&keys, Some(Duration::from_secs(20))))
            .await
            .unwrap();
        serde_json::to_value(view_now(state, t).await).unwrap()
    }

    #[test]
    fn state_follows_the_facts() {
        let g = catalog::gated(catalog::SAM3_ID).unwrap();
        let none = Lookup::Missing;
        let token = found();
        let bad = Lookup::Unreadable {
            source: hf_token::TokenSource::Saved,
            path: Some(PathBuf::from("t")),
            reason: "denied".into(),
        };
        let base = Facts {
            installed: false,
            parts_on_disk: false,
            job: None,
            lookup: &none,
            refused: None,
        };
        let of = |f: Facts<'_>| state_of(&f, g).0;
        let facts = || Facts { ..base_copy(&base) };
        assert_eq!(of(facts()), Sam3State::NotRequested);
        assert_eq!(
            of(Facts {
                lookup: &bad,
                ..facts()
            }),
            Sam3State::TokenUnreadable
        );
        assert_eq!(
            of(Facts {
                refused: Some(Refusal::NeedsLogin),
                ..facts()
            }),
            Sam3State::NeedsLogin
        );
        assert_eq!(
            of(Facts {
                refused: Some(Refusal::NeedsAccess),
                lookup: &token,
                ..facts()
            }),
            Sam3State::NeedsAccess
        );
        assert_eq!(
            of(Facts {
                job: Some(&JobState::Downloading { bytes: 1, total: 2 }),
                ..facts()
            }),
            Sam3State::Downloading
        );
        assert_eq!(
            of(Facts {
                job: Some(&JobState::Queued),
                ..facts()
            }),
            Sam3State::Downloading
        );
        // Hashing, whether the job has started on the file or still waits to.
        assert_eq!(
            of(Facts {
                job: Some(&JobState::Verifying),
                ..facts()
            }),
            Sam3State::Checking
        );
        assert_eq!(
            of(Facts {
                job: Some(&JobState::Queued),
                parts_on_disk: true,
                ..facts()
            }),
            Sam3State::Checking
        );
        let failed = JobState::Failed {
            error: "x".into(),
            retry_at: None,
            attempts: 1,
        };
        assert_eq!(
            of(Facts {
                job: Some(&failed),
                ..facts()
            }),
            Sam3State::Failed
        );
        // A retry is scheduled: still on its way.
        let retrying = JobState::Failed {
            error: "x".into(),
            retry_at: Some(u64::MAX),
            attempts: 1,
        };
        assert_eq!(
            of(Facts {
                job: Some(&retrying),
                ..facts()
            }),
            Sam3State::Downloading
        );
        // A folder on disk needs no login, whatever the token says.
        assert_eq!(
            of(Facts {
                parts_on_disk: true,
                lookup: &bad,
                refused: Some(Refusal::NeedsLogin),
                ..facts()
            }),
            Sam3State::NotRequested
        );
        assert_eq!(
            of(Facts {
                installed: true,
                job: Some(&failed),
                ..facts()
            }),
            Sam3State::Installed
        );
    }

    fn base_copy<'a>(f: &Facts<'a>) -> Facts<'a> {
        Facts {
            installed: f.installed,
            parts_on_disk: f.parts_on_disk,
            job: f.job,
            lookup: f.lookup,
            refused: f.refused,
        }
    }

    #[test]
    fn only_a_busy_autolabel_run_with_sam3_blocks_remove() {
        let res = fixture("http://127.0.0.1:9");
        let record = |kind: &str, state: &str, uses: &[&str]| -> TaskRecord {
            serde_json::from_value(serde_json::json!({
                "id": "t", "kind": kind, "title": "t", "created": "now", "state": state,
                "step": 0, "steps": 1, "percent": 0, "uses": uses,
            }))
            .unwrap()
        };
        let sam3 = [res.id];
        assert!(!run_uses_folder(&[], res));
        for state in ["queued", "running", "paused"] {
            assert!(
                run_uses_folder(&[record("autolabel", state, &sam3)], res),
                "{state}"
            );
            assert!(
                !run_uses_folder(&[record("autolabel", state, &[])], res),
                "a run with MM-Grounding-DINO: {state}"
            );
        }
        for state in ["done", "failed", "cancelled"] {
            assert!(
                !run_uses_folder(&[record("autolabel", state, &sam3)], res),
                "{state}"
            );
        }
        assert!(!run_uses_folder(
            &[
                record("import", "running", &[]),
                record("overlaps", "running", &[])
            ],
            res
        ));
    }

    #[test]
    fn missing_space_is_reported_and_partials_count() {
        let dir = std::env::temp_dir().join(format!("bop-sam3-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let res = fixture("https://example.invalid");
        let all = CONFIG.len() as u64 + WEIGHTS.len() as u64;
        assert_eq!(missing_bytes(res, &dir), all);
        std::fs::write(dir.join("config.json"), CONFIG).unwrap();
        std::fs::write(dir.join("model.safetensors.partial"), &WEIGHTS[..4]).unwrap();
        assert_eq!(missing_bytes(res, &dir), WEIGHTS.len() as u64 - 4);
        assert!(
            space_problem(100, Some(99), &dir)
                .unwrap()
                .contains("disk space")
        );
        assert_eq!(space_problem(100, Some(100), &dir), None);
        assert_eq!(space_problem(100, None, &dir), None);
        // Not created yet: the volume of what exists above it.
        assert!(free_bytes(&dir.join("a/b/c")).is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The folder on this machine: adopted by hash, with no token and not one request (neither
    /// the probe nor a file).
    #[tokio::test]
    async fn a_folder_on_disk_is_adopted_with_no_network() {
        let (state, root) = resources_state();
        let (base, hits) = server("200 OK", vec![]);
        let res = fixture(&base);
        let dir = put_files(&state, res);
        let t = target(res, Lookup::Missing);
        let (code, text, _) = body_of(download_with(state.clone(), t.clone()).await).await;
        assert_eq!(code, StatusCode::OK, "{text}");
        let v = settled(&state, &t).await;
        assert_eq!(v["state"], "installed", "{v}");
        assert_eq!(hits.load(Ordering::SeqCst), 0, "no request at all");
        assert!(manager::is_installed(res, &dir));
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn no_token_answers_needs_login_at_once_and_a_reload_keeps_it() {
        let (state, root) = resources_state();
        let (base, hits) = server(GATED_401, vec![]);
        let res = fixture(&base);
        let t = target(res, Lookup::Missing);
        let (code, _, v) = body_of(download_with(state.clone(), t.clone()).await).await;
        assert_eq!(code, StatusCode::CONFLICT);
        assert_eq!(v["state"], "needs_login", "{v}");
        assert_eq!(v["success"], false);
        assert!(v["message"].as_str().unwrap().contains("hf-login"), "{v}");
        assert!(v["command"].as_str().unwrap().contains("hf-login"), "{v}");
        assert_eq!(hits.load(Ordering::SeqCst), 1, "the probe only");
        // Nothing was queued.
        let m = state
            .resources
            .as_ref()
            .unwrap()
            .provisioner
            .manager()
            .unwrap();
        assert!(m.statuses().is_empty());
        let again = serde_json::to_value(view_now(&state, &t).await).unwrap();
        assert_eq!(again["state"], "needs_login");
        // A login clears it, with no request.
        let signed_in = target(res, found());
        let after = serde_json::to_value(view_now(&state, &signed_in).await).unwrap();
        assert_eq!(after["state"], "not_requested", "{after}");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        std::fs::remove_dir_all(&root).ok();
    }

    /// A token not approved: the state, and the value nowhere in the answer.
    #[tokio::test]
    async fn not_approved_answers_needs_access_and_never_shows_the_token() {
        let (state, root) = resources_state();
        let (base, _) = server(GATED_403, vec![]);
        let res = fixture(&base);
        let t = target(res, found());
        let (code, text, v) = body_of(download_with(state.clone(), t.clone()).await).await;
        assert_eq!(code, StatusCode::CONFLICT);
        assert_eq!(v["state"], "needs_access", "{v}");
        assert!(
            v["token_source"].as_str().unwrap().contains("hf-login"),
            "{v}"
        );
        assert!(v["message"].as_str().unwrap().contains(t.gated.repo_url));
        assert!(!text.contains(TOKEN), "{text}");
        let later = serde_json::to_string(&view_now(&state, &t).await).unwrap();
        assert!(!later.contains(TOKEN), "{later}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn an_unreadable_token_is_said_before_any_request() {
        let (state, root) = resources_state();
        let (base, hits) = server("200 OK", vec![]);
        let t = target(
            fixture(&base),
            Lookup::Unreadable {
                source: hf_token::TokenSource::Saved,
                path: Some(PathBuf::from("/x/token")),
                reason: "permission denied".into(),
            },
        );
        let (code, _, v) = body_of(download_with(state, t).await).await;
        assert_eq!(code, StatusCode::CONFLICT);
        assert_eq!(v["state"], "token_unreadable", "{v}");
        assert!(v["message"].as_str().unwrap().contains("permission denied"));
        assert_eq!(hits.load(Ordering::SeqCst), 0);
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn access_starts_the_download() {
        let (state, root) = resources_state();
        let (base, _) = server(
            "200 OK",
            vec![
                ("config.json", CONFIG.to_vec()),
                ("model.safetensors", WEIGHTS.to_vec()),
            ],
        );
        let res = fixture(&base);
        let t = target(res, found());
        let (code, text, _) = body_of(download_with(state.clone(), t.clone()).await).await;
        assert_eq!(code, StatusCode::OK, "{text}");
        let v = settled(&state, &t).await;
        assert_eq!(v["state"], "installed", "{v}");
        assert_eq!(v["parts_on_disk"], true);
        let dir = folder_of(&state.config_read(), res);
        assert_eq!(dir, root.join("tools/train-weights/sam3"));
        assert_eq!(std::fs::read(dir.join("config.json")).unwrap(), CONFIG);
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn an_unreachable_host_is_a_plain_failure() {
        let (state, root) = resources_state();
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let t = target(fixture(&format!("http://127.0.0.1:{port}")), found());
        let (code, text, v) = body_of(download_with(state, t).await).await;
        assert_eq!(code, StatusCode::BAD_GATEWAY);
        assert_eq!(v["state"], "failed", "{v}");
        assert!(!text.contains(TOKEN), "{text}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn without_a_manager_the_download_is_refused() {
        let state = test_state();
        let t = target(fixture("http://127.0.0.1:9"), Lookup::Missing);
        let (code, _, v) = body_of(download_with(state, t).await).await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["success"], false);
    }

    #[tokio::test]
    async fn remove_refuses_a_run_then_clears_folder_and_status() {
        let (state, root) = resources_state();
        let (base, _) = server("200 OK", vec![]);
        let res = fixture(&base);
        let dir = put_files(&state, res);
        let t = target(res, Lookup::Missing);
        body_of(download_with(state.clone(), t.clone()).await).await;
        assert_eq!(settled(&state, &t).await["state"], "installed");
        let key = Job::into_dir(res, dir.clone()).key();

        let busy: TaskRecord = serde_json::from_value(serde_json::json!({
            "id": "t", "kind": "autolabel", "title": "t", "created": "now", "state": "running",
            "step": 0, "steps": 1, "percent": 0, "uses": [res.id],
        }))
        .unwrap();
        let (code, _, v) = body_of(remove_with(&state, &t, &[busy])).await;
        assert_eq!(code, StatusCode::CONFLICT);
        assert_eq!(v["in_use"], true);
        assert!(dir.join("model.safetensors").exists());

        let (code, _, v) = body_of(remove_with(&state, &t, &[])).await;
        assert_eq!(code, StatusCode::OK, "{v}");
        assert_eq!(v["state"], "not_requested", "{v}");
        assert!(!dir.exists());
        let m = state
            .resources
            .as_ref()
            .unwrap()
            .provisioner
            .manager()
            .unwrap();
        assert!(m.status(&key).is_none(), "forgotten, so it can come back");
        std::fs::remove_dir_all(&root).ok();
    }

    /// The real routes: admin only, the real pin described, nothing fetched by a GET.
    #[tokio::test]
    async fn the_status_route_describes_the_pin() {
        let (state, root) = resources_state();
        let (code, text) = get_text(&state, "/v1/autolabel/sam3").await;
        assert_eq!(code, StatusCode::OK, "{text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["state"], "not_requested", "{v}");
        assert_eq!(v["license"], "SAM License");
        assert_eq!(v["repo_url"], "https://huggingface.co/facebook/sam3");
        assert!(v["size"].as_u64().unwrap() > 3_000_000_000);
        assert!(v["command"].as_str().unwrap().ends_with("hf-login"), "{v}");
        assert!(
            v["command"].as_str().unwrap().contains("cfg.json"),
            "the service's config: {v}"
        );

        let (code, _, _) = call(
            &state,
            anonymous(from_peer(
                Request::get("/v1/autolabel/sam3")
                    .body(Body::empty())
                    .unwrap(),
                REMOTE,
            )),
        )
        .await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);
        std::fs::remove_dir_all(&root).ok();
    }
}
