//! Applying a changed config to the running service: live settings or a per-model reload.

use super::AppState;
use crate::config::Config;
use crate::sync::LockExt;
use std::sync::Arc;
use tracing::{info, warn};

/// What [`apply_config`] did.
#[derive(Debug, Default)]
pub(crate) struct Applied {
    pub live: crate::registry::LiveApplied,
    /// Per-model reloads, starts and stops.
    pub reloads: Vec<crate::registry::Reload>,
    /// Downloads queued for a new device ("ONNX Runtime CUDA ...").
    pub queued: Vec<String>,
    /// Models whose change could not be applied (they stay pending).
    pub errors: Vec<String>,
}

impl Applied {
    /// "Reloading IPcam-general on OpenVINO CPU; stopping yolo26s. Downloading ...: ..."
    pub fn describe(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        let r: Vec<String> = self
            .reloads
            .iter()
            .filter(|r| !matches!(r, crate::registry::Reload::Unchanged { .. }))
            .map(crate::registry::Reload::describe)
            .collect();
        if !r.is_empty() {
            let mut t = r.join("; ");
            if let Some(c) = t.get_mut(0..1) {
                c.make_ascii_uppercase();
            }
            parts.push(format!("{t}."));
        }
        if !self.queued.is_empty() {
            parts.push(format!(
                "Downloading {} for the new device setting; the model serves on what it can \
                 meanwhile and switches when the download is installed (progress on the home \
                 and Config pages).",
                self.queued.join(", ")
            ));
        }
        if !self.errors.is_empty() {
            parts.push(format!("Not applied: {}.", self.errors.join("; ")));
        }
        (!parts.is_empty()).then(|| parts.join(" "))
    }
}

/// Apply the current config to this generation as far as it can without a restart, and
/// record the result as what runs ([`AppState::running`]):
/// - per-model changes ([`crate::config_merge::model_reloads`]: enabled/disabled, added,
///   removed, device, path, family, ...) reload, start or stop that model's worker only
///   ([`ModelRegistry::reload_model`]); a device that needs a download queues it first and the
///   model waits for it (or serves on what it can meanwhile);
/// - the live settings (thresholds, object filters, timeout, image saving, NMS) and the
///   default model swap at once ([`ModelRegistry::apply_live`]).
///
/// Runtime-level globals keep their running values until a restart. Idempotent; one at a
/// time.
pub(crate) fn apply_config(state: &AppState) -> Applied {
    let held = state.apply_lock.lock_recover();
    apply_config_locked(state, &held)
}

/// [`apply_config`] for a caller that holds `state.apply_lock` (the guard proves it): a save
/// writes and applies under one lock, so no other applier (a page's config sync, the runner
/// after a restart) applies its edit in between and leaves its message without what it
/// reloaded.
pub(crate) fn apply_config_locked(
    state: &AppState,
    _held: &std::sync::MutexGuard<'_, ()>,
) -> Applied {
    let (revision, current) = state.config.snapshot();
    let running = state.running();
    let mut next = crate::config_merge::runtime_overlay(&running, &current);
    let names = crate::config_merge::model_reloads(&running, &next);
    let mut out = Applied::default();
    if !names.is_empty() && state.registry.can_reload() {
        let provision = state.resources.as_ref().map(|ctx| {
            // A device the dropdown marks "will download" is downloaded even with
            // `auto_download` off (the user picked it).
            let device_changed = names.iter().any(|n| {
                let key = crate::registry::normalize_name(n);
                let dev = |c: &Config| {
                    c.models
                        .iter()
                        .find(|m| crate::registry::normalize_name(&m.effective_name()) == key)
                        .filter(|m| m.enabled)
                        .map(|m| c.device_for(m))
                };
                dev(&next).is_some() && dev(&running) != dev(&next)
            });
            if device_changed {
                out.queued = crate::resources::actions::queue_needs(ctx, &next);
                if !out.queued.is_empty() {
                    info!("device change needs downloads: {}", out.queued.join(", "));
                }
            }
            let plan = Arc::new(ctx.provisioner.prepare(&next, ctx.ort_loaded));
            state.registry.set_provision(plan.clone());
            plan
        });
        for name in &names {
            match state
                .registry
                .reload_model(name, &next, provision.as_deref())
            {
                Ok(r) => out.reloads.push(r),
                Err(e) => {
                    warn!(model = %name, "not reloaded: {e:#}");
                    out.errors.push(format!("{name}: {e:#}"));
                    crate::config_merge::keep_model_from(&mut next, &running, name);
                }
            }
        }
    } else {
        // Registries built from handles cannot spawn workers: the changes stay pending.
        for name in &names {
            crate::config_merge::keep_model_from(&mut next, &running, name);
        }
    }
    out.live = state.registry.apply_live(&next);
    state.running.store(Arc::new(next));
    state
        .applied_revision
        .store(revision, std::sync::atomic::Ordering::Release);
    out
}

/// Reload `names` as they run now (their files or runtimes changed, not the config), e.g. the
/// models that failed for a missing file once it is installed. One at a time with
/// [`apply_config`].
pub(crate) fn reload_models(state: &AppState, names: &[String]) -> Vec<crate::registry::Reload> {
    let _one = state.apply_lock.lock_recover();
    let running = state.running();
    names
        .iter()
        .filter_map(|n| match state.registry.reload_model(n, &running, None) {
            Ok(r) => Some(r),
            Err(e) => {
                warn!(model = %n, "not reloaded: {e:#}");
                None
            }
        })
        .collect()
}

/// Pick up a config file changed on disk (or a change another writer made, e.g. an export
/// that added its model) and apply it ([`apply_config`]).
pub(crate) fn sync_config(state: &AppState) {
    state.config.check_disk();
    if state.config.revision()
        != state
            .applied_revision
            .load(std::sync::atomic::Ordering::Acquire)
    {
        apply_config(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::server::test_util::*;

    use crate::cli::LogReloadHandle;
    use crate::config::{Config, LogLevel};

    use axum::http::{StatusCode, header};

    use super::super::config_page::ConfigView;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use std::time::Duration;

    /// The main form as a browser posts it from a page rendered from `c`, with the user's
    /// `edits`.
    fn server_form_body(c: &Config, edits: &[(&str, &str)]) -> String {
        let view = serde_json::to_value(ConfigView::from_config(c)).unwrap();
        let mut pairs: Vec<(String, String)> = Vec::new();
        for (k, v) in view.as_object().unwrap() {
            match v {
                serde_json::Value::Bool(true) => pairs.push((k.clone(), "on".into())),
                serde_json::Value::String(s) => pairs.push((k.clone(), s.clone())),
                _ => {}
            }
        }
        for (k, v) in edits {
            pairs.retain(|(x, _)| x != k);
            pairs.push((k.to_string(), v.to_string()));
        }
        pairs.push(("download_settings".into(), "1".into()));
        pairs
            .iter()
            .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
            .collect::<Vec<_>>()
            .join("&")
    }

    fn post_json(uri: &str, body: &str) -> Request<Body> {
        Request::post(uri)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .header(header::ACCEPT, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn first_model(state: &AppState) -> (Option<String>, Option<f32>) {
        let c = state.config_read();
        (c.models[0].device.clone(), c.models[0].confidence_threshold)
    }

    /// Last write wins: a Config page opened before a Benchmark apply and saved afterwards
    /// writes what it shows (no 409, no merge); the change log records both writes. The page
    /// no longer carries a `base`, and one posted by an old cached page is ignored.
    #[tokio::test]
    async fn stale_config_page_last_write_wins() {
        let state = test_state();
        state.config.update("test", |_| Ok(())).unwrap();
        let (_, page) = get_text(&state, "/config").await;
        assert!(!page.contains("name=\"base\""), "{page}");
        assert!(!page.contains("resolve_all"), "{page}");
        let stale = state.config_read().clone();
        let (s, _, b) = call(
            &state,
            post_form(
                "/v1/benchmark/apply",
                "model=ipcam-general&device=openvino:cpu",
                false,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        let (s, _, b) = call(
            &state,
            post_form(
                "/v1/benchmark/apply-threshold",
                "model=ipcam-general&threshold=0.35",
                false,
            ),
        )
        .await;
        let text = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK, "{text}");
        // Rounded in the API answer (was 0.3499999940395355).
        assert!(text.contains("\"to\":0.35"), "{text}");
        assert!(!text.contains("0.34999"), "{text}");
        let applied = (Some("openvino:cpu".to_string()), Some(0.35));
        assert_eq!(first_model(&state), applied);

        // The stale main form (its models JSON predates the apply) overwrites the model entry.
        let body = server_form_body(&stale, &[("port", "4000")]);
        let body = format!("{body}&base=%7B%7D&base_revision=1");
        let (s, _, b) = call(&state, post_json("/config", &body)).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(s, StatusCode::OK, "{j}");
        assert_eq!(j["success"], true);
        assert_eq!(state.config_read().port, 4000);
        assert_eq!(first_model(&state), (None, None));
        let file = Config::load(&state.config_path).unwrap();
        assert_eq!(file, *state.config_read());
        let last = state.config.changes(1).pop().unwrap();
        assert_eq!(last.source, "config page");
        assert!(
            last.summary.contains("Port \u{2192} 4000"),
            "{}",
            last.summary
        );
        assert!(
            last.summary.contains("device \u{2192} global"),
            "{}",
            last.summary
        );

        // The Models card writes its rows as posted.
        let evil = urlencode("<b>evil</b>");
        let card = format!(
            "enabled=ipcam-general&enabled={evil}&default_model=ipcam-general&device.ipcam-general=\
             ort:cpu&device.{evil}=&threshold.ipcam-general=&threshold.{evil}=0.6&action=save"
        );
        let (s, _, b) = call(&state, post_json("/config/models", &card)).await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        assert_eq!(first_model(&state), (Some("ort:cpu".to_string()), None));
        assert_eq!(
            state.config_read().models[1].confidence_threshold,
            Some(0.6)
        );
        let last = state.config.changes(1).pop().unwrap();
        assert_eq!(last.source, "config page (Models card)");
    }

    #[tokio::test]
    async fn forms_apply_as_submitted() {
        let state = test_state();
        let (s, _, _) = call(
            &state,
            post_form(
                "/config/models",
                "enabled=ipcam-general&device.ipcam-general=openvino:cpu&threshold.ipcam-general=0.456&action=save",
                true,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(
            first_model(&state),
            (Some("openvino:cpu".to_string()), Some(0.46))
        );
    }

    /// The Models card's Night model select: set, rejected when it would chain, cleared; the
    /// page lists the other model as an option.
    #[tokio::test]
    async fn models_card_sets_the_night_model() {
        let state = test_state();
        let evil = urlencode("<b>evil</b>");
        let form =
            format!("enabled=ipcam-general&enabled={evil}&night.ipcam-general={evil}&action=save");
        let (s, _, b) = call(&state, post_json("/config/models", &form)).await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        assert_eq!(
            state.config_read().models[0].night_model.as_deref(),
            Some("<b>evil</b>")
        );
        // Live: no reload, nothing pending.
        let last = state.config.changes(1).pop().unwrap();
        assert!(last.summary.contains("night model"), "{}", last.summary);
        let (_, page) = get_text(&state, "/config").await;
        assert!(page.contains("name=\"night.ipcam-general\""), "{page}");
        assert!(
            page.contains("<option value=\"&#60;b&#62;evil&#60;/b&#62;\" selected>"),
            "{page}"
        );
        // A chain is rejected and keeps the config.
        let before = state.config_read().clone();
        let form =
            format!("enabled=ipcam-general&enabled={evil}&night.{evil}=ipcam-general&action=save");
        let (s, _, b) = call(&state, post_json("/config/models", &form)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(
            j["message"].as_str().unwrap().contains("do not chain"),
            "{j}"
        );
        assert_eq!(*state.config_read(), before);
        // "" clears it.
        let form = format!("enabled=ipcam-general&enabled={evil}&night.ipcam-general=&action=save");
        let (s, _, _) = call(&state, post_json("/config/models", &form)).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(state.config_read().models[0].night_model, None);
    }

    /// SV-14: the Models card rejects a per-model threshold outside 0..=1 (the browser's
    /// `max="1"` is not enforced by curl or a hand-typed value) and keeps the config.
    #[tokio::test]
    async fn models_card_rejects_out_of_range_threshold() {
        let state = test_state();
        let before = state.config_read().clone();
        for bad in ["1.5", "-1", "NaN", "abc"] {
            let form = format!(
                "enabled=ipcam-general&threshold.ipcam-general={}&action=save",
                urlencode(bad)
            );
            let (s, _, b) = call(&state, post_json("/config/models", &form)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}");
            let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
            let msg = j["message"].as_str().unwrap();
            assert!(msg.contains("threshold for 'ipcam-general'"), "{msg}");
            assert!(msg.contains("between 0 and 1"), "{msg}");
            assert_eq!(*state.config_read(), before, "{bad}");
        }
        for ok in ["0", "1"] {
            let form = format!("enabled=ipcam-general&threshold.ipcam-general={ok}&action=save");
            let (s, _, b) = call(&state, post_json("/config/models", &form)).await;
            assert_eq!(s, StatusCode::OK, "{ok}: {}", String::from_utf8_lossy(&b));
        }
    }

    #[tokio::test]
    async fn config_endpoint_etag_pending_and_disk_changes() {
        let state = test_state();
        let get = |etag: Option<&str>| {
            let mut r = Request::get("/v1/config");
            if let Some(e) = etag {
                r = r.header(header::IF_NONE_MATCH, e);
            }
            r.body(Body::empty()).unwrap()
        };
        let (s, h, b) = call(&state, get(None)).await;
        assert_eq!(s, StatusCode::OK);
        let etag = h[header::ETAG].to_str().unwrap().to_string();
        assert!(etag.starts_with("W/\"1-"), "{etag}");
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(j["revision"], 1);
        assert_eq!(j["restartNeeded"], false);
        assert_eq!(j["models"][0]["name"], "ipcam-general");
        assert_eq!(j["models"][0]["running"]["present"], true);
        assert_eq!(j["models"][0]["running"]["threshold"], 0.5);
        assert_eq!(j["models"][0]["default"], true);
        assert!(j["forms"]["server"].get("base").is_none());
        assert_eq!(
            j["forms"]["server"]["view"]["port"],
            crate::DEFAULT_PORT.to_string()
        );
        // Unchanged: 304.
        let (s, h, b) = call(&state, get(Some(&etag))).await;
        assert_eq!(s, StatusCode::NOT_MODIFIED);
        assert!(b.is_empty());
        assert_eq!(h[header::ETAG].to_str().unwrap(), etag);

        // A change: new revision, applied live (no restart), the change log entry.
        let (s, _, _) = call(
            &state,
            post_form(
                "/v1/benchmark/apply-threshold",
                "model=ipcam-general&threshold=0.35",
                false,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let (s, h, b) = call(&state, get(Some(&etag))).await;
        assert_eq!(s, StatusCode::OK);
        assert_ne!(h[header::ETAG].to_str().unwrap(), etag);
        let text = String::from_utf8(b).unwrap();
        assert!(text.contains("\"threshold\":0.35"), "{text}");
        assert!(!text.contains("0.34999"), "{text}");
        let j: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(j["revision"], 2);
        assert_eq!(j["restartNeeded"], false);
        assert_eq!(j["pendingCount"], 0);
        assert_eq!(j["models"][0]["restartNeeded"], false);
        // Running: what the worker uses, already the new threshold.
        assert_eq!(j["models"][0]["running"]["threshold"], 0.35);
        assert_eq!(j["models"][1]["restartNeeded"], false);
        assert_eq!(j["changes"][0]["source"], "benchmark apply-threshold");
        assert_eq!(
            j["changes"][0]["summary"],
            "ipcam-general threshold \u{2192} 0.35"
        );
        assert_eq!(j["changes"][0]["live"][0], "ipcam-general \u{b7} threshold");
        // The config file says 0.35, too.
        let file = std::fs::read_to_string(&state.config_path).unwrap();
        assert!(file.contains("\"confidence_threshold\": 0.35"), "{file}");
        let (_, home) = get_text(&state, "/").await;
        assert!(!home.contains("title=\"threshold"), "{home}");
        let (_, cfg) = get_text(&state, "/config").await;
        assert!(cfg.contains("value=\"0.35\" placeholder=\"0.50\""), "{cfg}");

        // Another process edits the file (the CLI, an editor): picked up on the next poll.
        std::thread::sleep(Duration::from_millis(20));
        let mut other = Config::load(&state.config_path).unwrap();
        other.port = 4321;
        other.save(&state.config_path).unwrap();
        let (_, _, b) = call(&state, get(None)).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(j["revision"], 3);
        assert_eq!(j["changes"][1]["source"], "file changed on disk");
        assert_eq!(j["changes"][1]["summary"], "Port \u{2192} 4321");
        assert_eq!(
            j["pending"]["process"][0],
            "Port: 32168 \u{2192} 4321 (needs a process restart)"
        );
        assert_eq!(j["pendingCount"], 1);
        assert_eq!(state.config_read().port, 4321);
    }

    #[tokio::test]
    async fn log_level_writer_bumps_the_revision() {
        let mut st = Arc::try_unwrap(test_state()).ok().unwrap();
        st.log_reload = Some(LogReloadHandle::detached(LogLevel::Info));
        let state = Arc::new(st);
        let (s, _, _) = call(&state, post_form("/config/loglevel", "level=debug", false)).await;
        assert_eq!(s, StatusCode::OK);
        let last = state.config.changes(1).pop().unwrap();
        assert_eq!((last.source.as_str(), last.revision), ("log level", 2));
        assert_eq!(last.summary, "Log level \u{2192} debug");
        // The log level applies now: no restart needed for it.
        let (_, _, b) = call(
            &state,
            Request::get("/v1/config").body(Body::empty()).unwrap(),
        )
        .await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(j["restartNeeded"], false);
    }
}
