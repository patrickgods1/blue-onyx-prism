//! The diagnostics bundle a user attaches to a bug report: one zip, built by
//! `blue-onyx-prism diagnostics [--out <dir>]` and by the admin route `GET /v1/diagnostics`.
//!
//! What it holds (names inside the zip): `README.txt` (this list, for the reader), `system.json`
//! (version, git sha, build date, OS, CPU, memory), `devices.json` (the `/v1/devices` answer; the
//! CLI without a running service: the detected hardware only), `resources.json` (installed,
//! needed and available runtimes and models), `config.json` (the running or file config) and
//! `config-file.json` (the file as it is on disk, also when it does not parse), `logs/` (the
//! newest [`LOG_TAIL_BYTES`] of the log files, and `recent.txt`, the in-memory log of the
//! running service), `metrics.txt` (Prometheus text), `benchmark-summary.json` (per model and
//! device: the speed and grades of `benchmark.json`, without its per-image data) and
//! `problems.txt` (each part that could not be collected, and why).
//!
//! What it never holds: camera frames, label files, datasets, model weights, and the secrets.
//! The admin token, the Hugging Face token and everything in the private `auth` folder (the
//! accounts' password hashes in `users.json`, `sessions.json`, `devices.json`, the setup code,
//! the HTTPS key) live in their own files and are never read into a part; as a second line every
//! part is scrubbed of their values ([`known_secrets`]) before it is written. Camera names and file paths do stay (they are in the config and the logs, and
//! support needs them); the README's troubleshooting section says so.

use crate::config::Config;
use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};

/// How much of the log files goes in: the newest bytes, across the daily files.
pub const LOG_TAIL_BYTES: u64 = 5 * 1024 * 1024;

/// What replaces a secret's value in every part.
pub const REDACTED: &str = "[redacted]";

/// Secrets shorter than this are not scrubbed (a value that short would match ordinary text;
/// no real token is).
const MIN_SECRET_LEN: usize = 8;

/// The parts only a running service has (the CLI fills what it can).
#[derive(Debug, Default)]
pub struct Live {
    /// The `/v1/devices` answer.
    pub devices: Option<serde_json::Value>,
    /// The `/v1/resources` answer (the CLI builds it without the download manager).
    pub resources: Option<serde_json::Value>,
    /// Prometheus text.
    pub metrics: Option<String>,
    /// The in-memory log, one line per event.
    pub recent_logs: Option<String>,
    /// Why a part is missing (e.g. "the service is not running").
    pub notes: Vec<String>,
}

/// The values to scrub: the admin token (in `auth/` and where earlier versions kept it), the
/// token `hf-login` saved, the Hugging Face token this process would use (`HF_TOKEN`, the `hf`
/// CLI's file), the setup code (as stored and as printed), the password hashes of `users.json`
/// and every token-like string of it and of `sessions.json` / `devices.json` ([`token_like`]),
/// and the lines of the HTTPS key (generated, or the config's own `tls_key_path`). Never logged,
/// never returned.
pub fn known_secrets(config: &Config) -> Vec<String> {
    let root = config.data_root();
    let auth = crate::private_dir::path(&root);
    let mut out = Vec::new();
    let file = |p: PathBuf| {
        std::fs::read_to_string(p)
            .ok()
            .map(|s| s.trim().to_string())
    };
    out.extend(file(crate::admin_token::path(config)));
    out.extend(file(crate::admin_token::old_path(config)));
    out.extend(file(crate::resources::hf_token::saved_token_path(&root)));
    if let Some(t) = crate::resources::hf_token::lookup(&root).token() {
        out.push(t.secret().to_string());
    }
    if let Some(code) = file(auth.join(crate::accounts::setup::FILE_NAME)) {
        let grouped = code
            .as_bytes()
            .chunks(4)
            .map(|c| String::from_utf8_lossy(c).into_owned())
            .collect::<Vec<_>>()
            .join("-");
        out.push(grouped);
        out.push(code);
    }
    for name in [
        crate::accounts::FILE_NAME,
        crate::accounts::sessions::FILE_NAME,
        crate::accounts::devices::FILE_NAME,
    ] {
        if let Some(v) = file(auth.join(name)).and_then(|t| serde_json::from_str(&t).ok()) {
            token_like(&v, &mut out);
        }
    }
    let key = crate::tls::Source::of(config).paths().1;
    if let Some(pem) = file(key) {
        let body: Vec<&str> = pem
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with("-----"))
            .collect();
        out.push(body.concat());
        out.extend(body.iter().map(|l| (*l).to_string()));
    }
    out.retain(|s| s.len() >= MIN_SECRET_LEN);
    // Longest first, so a secret that holds another is replaced whole.
    out.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    out.dedup();
    out
}

/// Strings of at least this length without whitespace count as token-like in the auth files:
/// longer than any username (32), shorter than a hex SHA-256 (64) or a PHC string.
const TOKEN_LIKE_LEN: usize = 40;

/// Every token-like string (keys and values) in `v`, and the `$`-separated parts of a PHC
/// string (its salt and hash) of 16 characters or more.
fn token_like(v: &serde_json::Value, out: &mut Vec<String>) {
    let mut push = |s: &str| {
        if s.len() >= TOKEN_LIKE_LEN && !s.chars().any(char::is_whitespace) {
            out.push(s.to_string());
            if s.starts_with('$') {
                out.extend(s.split('$').filter(|p| p.len() >= 16).map(str::to_string));
            }
        }
    };
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        match v {
            serde_json::Value::String(s) => push(s),
            serde_json::Value::Array(a) => stack.extend(a),
            serde_json::Value::Object(o) => {
                for (k, v) in o {
                    push(k);
                    stack.push(v);
                }
            }
            _ => {}
        }
    }
}

/// `text` with every secret replaced by [`REDACTED`].
pub fn scrub(text: &str, secrets: &[String]) -> String {
    secrets
        .iter()
        .fold(text.to_string(), |t, s| t.replace(s.as_str(), REDACTED))
}

/// The newest log files under `dir` (`blue_onyx_prism.log.<date>`, newest first) with at most
/// `budget` bytes in all: whole files while they fit, then the tail of the next one (cut at a
/// line start). (name, text) pairs, newest first.
pub fn log_tail(dir: &Path, budget: u64) -> Vec<(String, String)> {
    let prefix = crate::cli::log_files::PREFIX;
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix))
        .collect();
    // The date suffix sorts by age.
    names.sort_by(|a, b| b.cmp(a));
    let mut left = budget;
    let mut out = Vec::new();
    for name in names {
        if left == 0 {
            break;
        }
        let Some((bytes, cut)) = read_tail(&dir.join(&name), left) else {
            continue;
        };
        left = left.saturating_sub(bytes.len() as u64);
        if !bytes.is_empty() {
            out.push((name, String::from_utf8_lossy(&bytes).into_owned()));
        }
        if cut {
            // The budget ended inside this file: older ones do not fit.
            break;
        }
    }
    out
}

/// The last `max` bytes of `path`, starting after a newline when cut; and whether it was cut.
fn read_tail(path: &Path, max: u64) -> Option<(Vec<u8>, bool)> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    let start = len.saturating_sub(max);
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = Vec::new();
    f.take(max).read_to_end(&mut buf).ok()?;
    if start > 0
        && let Some(nl) = buf.iter().position(|&b| b == b'\n')
    {
        buf.drain(..=nl);
    }
    Some((buf, start > 0))
}

/// Version, OS and machine facts.
pub fn system_json() -> serde_json::Value {
    serde_json::json!({
        "version": crate::VERSION,
        "longVersion": crate::long_version(),
        "gitSha": crate::GIT_SHA,
        "buildDate": crate::BUILD_DATE,
        "os": crate::system_info::os_description(),
        "arch": std::env::consts::ARCH,
        "family": std::env::consts::OS,
        "cpu": crate::system_info::cpu_name(),
        "memoryGb": crate::system_info::total_memory_gb(),
        "generatedAt": chrono::Utc::now().to_rfc3339(),
    })
}

/// `benchmark.json` beside `config_path` reduced to what support reads: per model the
/// recommendation and threshold, per device the median time, throughput, grades or error. None
/// when there are no results.
pub fn benchmark_summary(config_path: &Path) -> Result<Option<serde_json::Value>> {
    let path = crate::benchmark::results_path(config_path);
    let Some(r) = crate::benchmark::BenchmarkResults::load(&path)? else {
        return Ok(None);
    };
    let models: Vec<serde_json::Value> = r
        .models
        .iter()
        .map(|m| {
            let devices: Vec<serde_json::Value> = m
                .devices
                .iter()
                .map(|d| {
                    serde_json::json!({
                        "device": d.device,
                        "label": d.label,
                        "error": d.error,
                        "fellBack": d.run.as_ref().map(|r| r.fell_back),
                        "totalP50Ms": d.total_p50(),
                        "throughputFps": d.run.as_ref().map(|r| r.throughput_fps),
                        "compileMs": d.run.as_ref().map(|r| r.compile_ms),
                        "grades": d.grades(),
                    })
                })
                .collect();
            serde_json::json!({
                "model": m.model,
                "family": m.family,
                "timestamp": m.timestamp,
                "datasets": m.datasets,
                "images": m.images,
                "configured": m.configured,
                "recommended": m.recommended,
                "recommendation": m.recommendation,
                "error": m.error,
                "threshold": m.threshold.as_ref().and_then(|t| t.apply_value()),
                "devices": devices,
            })
        })
        .collect();
    Ok(Some(serde_json::json!({
        "version": r.version,
        "timestamp": r.timestamp,
        "hardware": r.hardware,
        "runtimes": r.runtimes,
        "models": models,
    })))
}

/// The bundle as zip bytes for `config` (the running or file config) read from `config_path`.
pub fn build(config: &Config, config_path: &Path, live: Live) -> Result<Vec<u8>> {
    let secrets = known_secrets(config);
    let mut parts: Vec<(String, String)> = Vec::new();
    let mut problems: Vec<String> = live.notes.clone();
    let json = |v: &serde_json::Value| serde_json::to_string_pretty(v).unwrap_or_default();

    parts.push(("README.txt".into(), readme()));
    parts.push(("system.json".into(), json(&system_json())));
    match &live.devices {
        Some(d) => parts.push(("devices.json".into(), json(d))),
        None => parts.push((
            "devices.json".into(),
            json(&serde_json::json!({
                "note": "no running service answered, so this is the detected hardware only \
                         (no runtimes, no device options)",
                "hardware": crate::backend::detect::hardware(),
            })),
        )),
    }
    let resources = live.resources.clone().unwrap_or_else(|| {
        serde_json::json!({
            "resources": crate::resources::status::rows(None, config),
            "localModels": crate::resources::status::local_models(None, config),
        })
    });
    parts.push(("resources.json".into(), json(&resources)));
    match config.saved_text() {
        Ok(t) => parts.push(("config.json".into(), t)),
        Err(e) => problems.push(format!("config.json: {e:#}")),
    }
    match std::fs::read(config_path) {
        Ok(b) => parts.push((
            "config-file.json".into(),
            String::from_utf8_lossy(&b).into_owned(),
        )),
        Err(e) => problems.push(format!(
            "config-file.json: reading {}: {e}",
            config_path.display()
        )),
    }
    match config.log_path.as_deref().map(crate::resolve_path) {
        Some(dir) => {
            let logs = log_tail(&dir, LOG_TAIL_BYTES);
            if logs.is_empty() {
                problems.push(format!("logs: no log files in {}", dir.display()));
            }
            for (name, text) in logs {
                parts.push((format!("logs/{name}"), text));
            }
        }
        None => problems.push(
            "logs: no log_path is configured (the log goes to the console), so only the \
             running service's recent log is included"
                .into(),
        ),
    }
    if let Some(recent) = &live.recent_logs {
        parts.push(("logs/recent.txt".into(), recent.clone()));
    }
    match &live.metrics {
        Some(m) => parts.push(("metrics.txt".into(), m.clone())),
        None => problems.push("metrics.txt: no running service answered".into()),
    }
    match benchmark_summary(config_path) {
        Ok(Some(b)) => parts.push(("benchmark-summary.json".into(), json(&b))),
        Ok(None) => problems.push("benchmark-summary.json: no benchmark results yet".into()),
        Err(e) => problems.push(format!("benchmark-summary.json: {e:#}")),
    }
    if !problems.is_empty() {
        parts.push(("problems.txt".into(), problems.join("\n") + "\n"));
    }

    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    let mut options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    if let Some(t) = zip_time(chrono::Utc::now()) {
        options = options.last_modified_time(t);
    }
    for (name, text) in parts {
        zip.start_file(name.as_str(), options)
            .with_context(|| format!("adding {name} to the diagnostics zip"))?;
        zip.write_all(scrub(&text, &secrets).as_bytes())
            .with_context(|| format!("writing {name} into the diagnostics zip"))?;
    }
    Ok(zip
        .finish()
        .context("finishing the diagnostics zip")?
        .into_inner())
}

/// `now` as a zip entry time (UTC; zip times have no zone), None outside 1980..=2107.
fn zip_time(now: chrono::DateTime<chrono::Utc>) -> Option<zip::DateTime> {
    use chrono::{Datelike, Timelike};
    let small = |v: u32| u8::try_from(v).ok();
    zip::DateTime::from_date_and_time(
        u16::try_from(now.year()).ok()?,
        small(now.month())?,
        small(now.day())?,
        small(now.hour())?,
        small(now.minute())?,
        small(now.second())?,
    )
    .ok()
}

fn readme() -> String {
    format!(
        "Blue Onyx Prism diagnostics, {}\n\n\
         Attach this zip to a bug report. It holds: the version and OS (system.json), devices \
         and runtimes (devices.json), resources (resources.json), the config (config.json, \
         config-file.json), the newest {} MB of the log files (logs/), metrics (metrics.txt) and \
         a benchmark summary (benchmark-summary.json). problems.txt lists what could not be \
         collected.\n\n\
         It never holds camera frames, labels, datasets, model weights, the admin token or a \
         Hugging Face token. Camera names and file paths are included: look through it before \
         you share it publicly.\n",
        crate::long_version(),
        LOG_TAIL_BYTES / (1024 * 1024),
    )
}

/// File name of a bundle made now: `blue-onyx-prism-diagnostics-<UTC time>.zip`.
pub fn file_name() -> String {
    format!(
        "blue-onyx-prism-diagnostics-{}.zip",
        chrono::Utc::now().format("%Y%m%d-%H%M%S")
    )
}

/// What a running service on this machine answers for the live parts, at `port` on loopback
/// (open routes only: `/v1/devices`, `/prometheus`); None for each part it does not answer.
async fn fetch_live(port: u16) -> Live {
    let mut live = Live::default();
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .no_proxy()
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            live.notes.push(format!("could not ask the service: {e}"));
            return live;
        }
    };
    let base = format!("http://127.0.0.1:{port}");
    match client.get(format!("{base}/v1/devices")).send().await {
        Ok(r) if r.status().is_success() => live.devices = r.json().await.ok(),
        Ok(r) => live
            .notes
            .push(format!("devices: the service answered HTTP {}", r.status())),
        Err(e) => live.notes.push(format!(
            "devices, metrics: no service answered on {base} ({e}); start it for the live \
             parts, or use the Diagnostics link of its web page"
        )),
    }
    if live.devices.is_some() {
        match client.get(format!("{base}/prometheus")).send().await {
            Ok(r) if r.status().is_success() => live.metrics = r.text().await.ok(),
            Ok(r) => live
                .notes
                .push(format!("metrics: the service answered HTTP {}", r.status())),
            Err(e) => live.notes.push(format!("metrics: {e}")),
        }
    }
    live
}

/// `blue-onyx-prism diagnostics`: build the bundle for the config at `config_path` (defaults
/// when the file does not exist or does not parse; the raw file is included either way), ask
/// a service running on this machine for the live parts, and write the zip into `out` (default:
/// the current directory). Returns the zip's path.
pub fn command(config_path: &Path, out: Option<&Path>) -> Result<PathBuf> {
    let mut notes = Vec::new();
    let config = if config_path.exists() {
        Config::load(config_path).unwrap_or_else(|e| {
            notes.push(format!(
                "config: {e:#}; the defaults are used for the rest (config-file.json is the \
                 file as it is)"
            ));
            Config::default()
        })
    } else {
        notes.push(format!(
            "config: there is no config file at {}; defaults are used (pass --config <file> \
             before the subcommand for the service's config)",
            config_path.display()
        ));
        Config::default()
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building a runtime to ask the service")?;
    let mut live = rt.block_on(fetch_live(config.port));
    live.notes.splice(0..0, notes);
    let bytes = build(&config, config_path, live)?;
    let dir = match out {
        Some(d) => d.to_path_buf(),
        None => std::env::current_dir().context("finding the current directory")?,
    };
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating {} for the diagnostics zip", dir.display()))?;
    let path = dir.join(file_name());
    std::fs::write(&path, bytes).with_context(|| {
        format!(
            "writing {}; choose a writable folder with --out <dir>",
            path.display()
        )
    })?;
    Ok(path)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Read;

    /// A fresh folder, removed on drop (also when the test fails).
    pub(crate) struct Scratch(pub(crate) PathBuf);

    impl Scratch {
        pub(crate) fn new() -> Self {
            let d = std::env::temp_dir().join(format!("bop-diag-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Every entry of a zip: (name, text).
    pub(crate) fn entries(bytes: &[u8]) -> Vec<(String, String)> {
        let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        (0..z.len())
            .map(|i| {
                let mut f = z.by_index(i).unwrap();
                let mut s = String::new();
                f.read_to_string(&mut s).unwrap();
                (f.name().unwrap().to_string(), s)
            })
            .collect()
    }

    pub(crate) const ADMIN: &str = "admintoken-PLANTED-4f1c9e";
    pub(crate) const HF: &str = "hf_PLANTEDtokenvalue123456";
    /// An admin token where earlier versions kept it (the data root).
    pub(crate) const OLD_ADMIN: &str = "oldadmintoken-PLANTED-77aa01";
    pub(crate) const SETUP: &str = "PLANTEDSETUPCODEQRST";
    pub(crate) const SETUP_GROUPED: &str = "PLAN-TEDS-ETUP-CODE-QRST";
    pub(crate) const PHC: &str = "$argon2id$v=19$m=19456,t=2,p=1$UExBTlRFRHNhbHRzYWx0MTI$UExBTlRFRGhhc2hoYXNoaGFzaGhhc2hoYXNoaGFzaDE";
    pub(crate) const PHC_SALT: &str = "UExBTlRFRHNhbHRzYWx0MTI";
    pub(crate) const PHC_HASH: &str = "UExBTlRFRGhhc2hoYXNoaGFzaGhhc2hoYXNoaGFzaDE";
    pub(crate) const SESSION: &str =
        "5e55105e55105e55105e55105e55105e55105e55105e55105e55105e55105e55";
    pub(crate) const DEVICE: &str =
        "dec1ced0dec1ced0dec1ced0dec1ced0dec1ced0dec1ced0dec1ced0dec1ced0";
    pub(crate) const KEY_LINE: &str =
        "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgPLANTEDkeyPLANTED";

    /// Every value [`planted`] hides, for the assertions.
    pub(crate) fn planted_secrets() -> Vec<&'static str> {
        vec![
            ADMIN,
            HF,
            OLD_ADMIN,
            SETUP,
            SETUP_GROUPED,
            PHC,
            PHC_SALT,
            PHC_HASH,
            SESSION,
            DEVICE,
            KEY_LINE,
        ]
    }

    /// A data root with every secret file planted (both token files, an old admin token, the
    /// auth folder's accounts, sessions, devices, setup code and HTTPS key), a log dir with two
    /// days of logs (one line of today's quoting each secret), a frame and a label file in a
    /// dataset folder, and a config file naming it all.
    pub(crate) fn planted(dir: &Path) -> (Config, PathBuf) {
        let config = Config {
            download_dir: Some(dir.to_path_buf()),
            log_path: Some(dir.join("logs")),
            ..Default::default()
        };
        let auth = crate::private_dir::ensure(dir).unwrap();
        std::fs::write(crate::admin_token::path(&config), format!("{ADMIN}\n")).unwrap();
        std::fs::write(crate::admin_token::old_path(&config), OLD_ADMIN).unwrap();
        std::fs::write(auth.join("setup-code"), format!("{SETUP}\n")).unwrap();
        std::fs::write(
            auth.join("users.json"),
            serde_json::json!({"format": 1, "users": [{"name": "alice", "hash": PHC,
                "created": 1, "password_changed": 1, "generation": 1}]})
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            auth.join("sessions.json"),
            serde_json::json!({"format": 1, "sessions": {SESSION: {"user": "alice"}}}).to_string(),
        )
        .unwrap();
        std::fs::write(
            auth.join("devices.json"),
            serde_json::json!({"format": 1, "devices": [{"id": DEVICE, "user": "alice"}]})
                .to_string(),
        )
        .unwrap();
        let tls = crate::private_dir::ensure_child(&auth, "tls").unwrap();
        std::fs::write(
            tls.join("key.pem"),
            format!("-----BEGIN PRIVATE KEY-----\n{KEY_LINE}\nAB==\n-----END PRIVATE KEY-----\n"),
        )
        .unwrap();
        let hf = crate::resources::hf_token::saved_token_path(dir);
        std::fs::create_dir_all(hf.parent().unwrap()).unwrap();
        std::fs::write(&hf, HF).unwrap();
        let logs = dir.join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        std::fs::write(
            logs.join("blue_onyx_prism.log.2026-11-06"),
            "old day line\n",
        )
        .unwrap();
        std::fs::write(
            logs.join("blue_onyx_prism.log.2026-11-07"),
            format!(
                "INFO camera FrontDoor\nWARN leaked {ADMIN} and {HF}\nWARN more {}\n",
                planted_secrets().join(" ")
            ),
        )
        .unwrap();
        std::fs::write(logs.join("other.txt"), "not a log").unwrap();
        let ds = dir.join("dataset").join("day").join("FrontDoor");
        std::fs::create_dir_all(&ds).unwrap();
        std::fs::write(ds.join("frame.jpg"), b"JPEGFRAMEBYTES").unwrap();
        std::fs::write(ds.join("frame.labels.json"), "{\"LABELFILE\":1}").unwrap();
        let path = dir.join("cfg.json");
        config.save(&path).unwrap();
        (config, path)
    }

    #[test]
    fn the_bundle_never_holds_the_tokens_frames_or_labels() {
        let dir = Scratch::new();
        let (config, path) = planted(&dir.0);
        let live = Live {
            metrics: Some(format!("bop_up 1\n# {ADMIN}\n")),
            recent_logs: Some(format!("INFO x {HF}\n")),
            ..Default::default()
        };
        let bytes = build(&config, &path, live).unwrap();
        let all = entries(&bytes);
        let names: Vec<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
        for want in [
            "README.txt",
            "system.json",
            "devices.json",
            "resources.json",
            "config.json",
            "config-file.json",
            "logs/blue_onyx_prism.log.2026-11-07",
            "logs/blue_onyx_prism.log.2026-11-06",
            "logs/recent.txt",
            "metrics.txt",
        ] {
            assert!(names.contains(&want), "{want} missing from {names:?}");
        }
        assert!(!names.iter().any(|n| n.contains("other.txt")), "{names:?}");
        let mut hidden = planted_secrets();
        hidden.extend(["JPEGFRAMEBYTES", "LABELFILE"]);
        for (name, text) in &all {
            for secret in &hidden {
                assert!(!text.contains(secret), "{name} holds {secret}");
            }
        }
        // The raw zip bytes too (names, comments).
        let raw = String::from_utf8_lossy(&bytes);
        for secret in planted_secrets() {
            assert!(!raw.contains(secret), "{secret}");
        }
        // No part is one of the auth folder's files.
        assert!(
            !names.iter().any(
                |n| ["users", "sessions", "setup-code", "key.pem", "admin-token"]
                    .iter()
                    .any(|f| n.contains(f))
            ),
            "{names:?}"
        );
        let log = &all
            .iter()
            .find(|(n, _)| n == "logs/blue_onyx_prism.log.2026-11-07")
            .unwrap()
            .1;
        assert!(log.contains("camera FrontDoor"), "camera names stay");
        assert!(log.contains(&format!("leaked {REDACTED} and {REDACTED}")));
        let system: serde_json::Value =
            serde_json::from_str(&all.iter().find(|(n, _)| n == "system.json").unwrap().1).unwrap();
        assert_eq!(system["version"], crate::VERSION);
        assert_eq!(system["longVersion"], crate::long_version());
    }

    #[test]
    fn log_tail_keeps_the_newest_bytes_from_a_line_start() {
        let dir = Scratch::new();
        let d = &dir.0;
        std::fs::write(d.join("blue_onyx_prism.log.2026-11-05"), "aaaa\n").unwrap();
        std::fs::write(d.join("blue_onyx_prism.log.2026-11-06"), "b1\nb2\nb3\n").unwrap();
        std::fs::write(d.join("blue_onyx_prism.log.2026-11-07"), "c1\nc2\n").unwrap();
        let all = log_tail(d, 1000);
        let names: Vec<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "blue_onyx_prism.log.2026-11-07",
                "blue_onyx_prism.log.2026-11-06",
                "blue_onyx_prism.log.2026-11-05"
            ]
        );
        // 6 bytes of today, then 4 of yesterday's 9: cut to the line after the cut.
        let some = log_tail(d, 10);
        assert_eq!(
            some,
            vec![
                (
                    "blue_onyx_prism.log.2026-11-07".to_string(),
                    "c1\nc2\n".to_string()
                ),
                (
                    "blue_onyx_prism.log.2026-11-06".to_string(),
                    "b3\n".to_string()
                ),
            ]
        );
        assert!(log_tail(&d.join("missing"), 10).is_empty());
    }

    #[test]
    fn scrub_replaces_every_occurrence() {
        let s = vec!["secretvalue1".to_string()];
        assert_eq!(
            scrub("a secretvalue1 b secretvalue1", &s),
            format!("a {REDACTED} b {REDACTED}")
        );
        assert_eq!(scrub("nothing", &[]), "nothing");
    }

    #[test]
    fn benchmark_summary_drops_per_image_data() {
        let dir = Scratch::new();
        let path = dir.0.join("cfg.json");
        assert!(benchmark_summary(&path).unwrap().is_none());
        std::fs::write(
            crate::benchmark::results_path(&path),
            serde_json::json!({
                "version": "1",
                "timestamp": "t",
                "models": [{
                    "model": "m",
                    "recommended": "ort:cpu",
                    "recommendation": "use ort:cpu",
                    "devices": [{"device": "ort:cpu", "label": "CPU", "error": "boom"}],
                }],
            })
            .to_string(),
        )
        .unwrap();
        let s = benchmark_summary(&path).unwrap().unwrap();
        assert_eq!(s["models"][0]["recommended"], "ort:cpu");
        assert_eq!(s["models"][0]["devices"][0]["error"], "boom");
        assert!(s["models"][0]["devices"][0]["totalP50Ms"].is_null());
    }

    /// The CLI path writes a zip with a dated name into `--out`, for a config whose file does
    /// not exist (problems.txt says so).
    #[test]
    fn command_writes_the_zip_into_out() {
        let dir = Scratch::new();
        let out = dir.0.join("out");
        let cfg = dir.0.join("missing.json");
        let path = command(&cfg, Some(&out)).unwrap();
        assert_eq!(path.parent(), Some(out.as_path()));
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("blue-onyx-prism-diagnostics-") && name.ends_with(".zip"));
        let all = entries(&std::fs::read(&path).unwrap());
        let problems = &all.iter().find(|(n, _)| n == "problems.txt").unwrap().1;
        assert!(problems.contains("there is no config file"), "{problems}");
    }
}
