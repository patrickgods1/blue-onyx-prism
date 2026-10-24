//! Benchmark apply endpoints: write measured per-model devices (`/v1/benchmark/apply`),
//! searched confidence thresholds (`/v1/benchmark/apply-threshold`) and per-device confidence
//! profiles with devices (`/v1/benchmark/apply-profiles`) into the config, and the exact
//! threshold search over the stored predictions (`/v1/benchmark/threshold-search`). Every
//! config write goes through one `save_config` call; file I/O and search run in
//! `server::blocking`.

use super::{ApiError, form_field, form_models, json_error, truthy};
use crate::benchmark::{BenchmarkResults, results_path};
use crate::config::Config;
use crate::config_form::{
    DeviceChange, ProfileChange, ThresholdChange, apply_device_thresholds, apply_model_devices,
    apply_model_thresholds,
};
use crate::server::{AppState, save_config, schedule_restart};
use axum::Json;
use axum::extract::rejection::FormRejection;
use axum::extract::{Form, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

/// `POST /v1/benchmark/apply`: set per-model devices in the config file.
/// - `model=<name>&device=<spec>`: one model ("" or `global` = follow the global device);
/// - `all=1` (optionally limited by repeated `model`): every recommendation of the saved results.
///
/// The changed models reload on their new devices at once (per-model reload); `restart=1`
/// restarts the server when a change could not be applied that way. Answers JSON with the
/// changes, `previous` (model -> its device before, a spec or `global`: what Undo posts
/// back as `device`), `restarting` and `needsRestart` (a change waits for a restart); 400 on
/// bad input, 404 when `all` finds no recommendation.
pub(in crate::server) async fn apply(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || apply_blocking(&state, &form)).await
}

/// [`apply`] off the runtime threads: it parses the results, writes the config and reloads
/// the changed models (SV-06).
fn apply_blocking(state: &Arc<AppState>, form: &[(String, String)]) -> Response {
    let restart = form_field(form, "restart").is_some_and(truthy);
    let picks = match device_picks(state, form) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let mut changes: Vec<DeviceChange> = Vec::new();
    let saved = save_config(state, "benchmark apply", |c| {
        changes = apply_model_devices(c, &picks)?;
        Ok(())
    });
    if let Err(e) = saved {
        warn!("benchmark apply refused: {e:#}");
        return json_error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    let (message, restarting, needs_restart) = device_apply_outcome(state, &changes, restart);
    info!("benchmark apply: {message}");
    // For Undo: POST `model=<m>&device=<previous>` (`global` = follow the global device).
    let previous: serde_json::Map<String, serde_json::Value> = changes
        .iter()
        .map(|c| {
            (
                c.model.clone(),
                c.from.clone().unwrap_or_else(|| "global".into()).into(),
            )
        })
        .collect();
    Json(serde_json::json!({
        "success": true,
        "message": message,
        "changes": changes,
        "previous": previous,
        "restarting": restarting,
        "needsRestart": needs_restart,
    }))
    .into_response()
}

/// The saved results of the apply actions: 404 when there are none yet, 500 when they cannot
/// be read.
fn results_for_apply(state: &AppState) -> Result<BenchmarkResults, ApiError> {
    match BenchmarkResults::load(&results_path(&state.config_path)) {
        Ok(Some(r)) => Ok(r),
        Ok(None) => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "no benchmark results yet; run a benchmark first",
        )),
        Err(e) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e:#}"),
        )),
    }
}

/// The (model, device) pairs of an apply form: `all=1` takes the saved recommendations of
/// the configured models (limited to the form's models, if any), else one `model` + `device`.
fn device_picks(
    state: &AppState,
    form: &[(String, String)],
) -> Result<Vec<(String, Option<String>)>, ApiError> {
    let models = form_models(form);
    if !form_field(form, "all").is_some_and(truthy) {
        return match (models.as_slice(), form_field(form, "device")) {
            ([model], Some(device)) => Ok(vec![(model.clone(), Some(device.to_string()))]),
            _ => Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "give `model` and `device`, or `all=1` to apply every recommendation",
            )),
        };
    }
    let results = results_for_apply(state)?;
    let only = (!models.is_empty()).then_some(models.as_slice());
    let recs = results.recommendations(only);
    if recs.is_empty() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "the saved results recommend no device for these models",
        ));
    }
    // Results may name models that were removed from the config since: skip those.
    let cfg = state.config_read().clone();
    Ok(recs
        .into_iter()
        .filter(|(m, _)| crate::benchmark::find_model(&cfg, m).is_some())
        .map(|(m, d)| (m, Some(d)))
        .collect())
}

/// The answer to saved device `changes`, whether the server restarts (scheduled here when
/// `restart` is asked and a change could not be applied by a per-model reload) and whether a
/// change needs a restart to take effect.
fn device_apply_outcome(
    state: &Arc<AppState>,
    changes: &[DeviceChange],
    restart: bool,
) -> (String, bool, bool) {
    let mut message = if changes.is_empty() {
        "No change: the models already use these devices.".to_string()
    } else {
        format!(
            "Saved {}: {}.",
            state.config_path.display(),
            changes
                .iter()
                .map(DeviceChange::describe)
                .collect::<Vec<_>>()
                .join("; ")
        )
    };
    let (rest, restarting, unapplied) = device_outcome(state, changes, restart);
    message.push_str(&rest);
    (message, restarting, unapplied)
}

/// What happens with saved device `changes` (the text after "Saved ...: <changes>."), whether
/// the server restarts and whether a change needs a restart ([`device_apply_outcome`]).
fn device_outcome(
    state: &Arc<AppState>,
    changes: &[DeviceChange],
    restart: bool,
) -> (String, bool, bool) {
    let mut message = String::new();
    if state.config_read().force_cpu {
        message.push_str(" Note: Force CPU is on and overrides per-model devices.");
    }
    // `save_config` reloaded the changed models (per-model reload); a change that could not
    // be applied that way stays pending and needs a restart (`restart=1` does it now).
    let pending = crate::config_merge::pending(&state.running(), &state.config_read());
    let unapplied = changes
        .iter()
        .any(|c| !pending.for_model(&c.model).is_empty());
    let restarting = restart && unapplied;
    if restarting {
        message.push_str(" Restarting the server to load the models on their new devices.");
        schedule_restart(state);
    } else if unapplied {
        message.push_str(" Restart the server to apply.");
    } else if changes
        .iter()
        .any(|c| state.registry.by_name(&c.model).is_some())
    {
        message.push_str(" Reloading the models on their new devices (no restart needed).");
    } else if !changes.is_empty() {
        message.push_str(" Used when the model is enabled (no restart needed).");
    }
    (message, restarting, unapplied)
}

/// The saved results and stored predictions, or a 400 telling to run a benchmark first.
fn search_data(
    state: &AppState,
) -> Result<(BenchmarkResults, Arc<crate::benchmark::search::StoredPreds>), ApiError> {
    use crate::benchmark::search::{StoredPreds, preds_path};
    let none = || {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "no stored predictions: run a benchmark first (it stores the predictions the \
             threshold search needs)",
        )
    };
    let results = match BenchmarkResults::load(&results_path(&state.config_path)) {
        Ok(Some(r)) => r,
        Ok(None) => return Err(none()),
        Err(e) => {
            return Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{e:#}"),
            ));
        }
    };
    match StoredPreds::load_cached(&preds_path(&state.config_path)) {
        Ok(Some(p)) if !p.runs.is_empty() => Ok((results, p)),
        Ok(_) => Err(none()),
        Err(e) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e:#}"),
        )),
    }
}

/// Normalized model name -> its effective confidence threshold in the config (2 decimals).
fn current_thresholds(cfg: &Config) -> HashMap<String, f64> {
    cfg.models
        .iter()
        .map(|m| {
            (
                crate::registry::normalize_name(&m.effective_name()),
                crate::config::round_threshold_f64(
                    m.confidence_threshold.unwrap_or(cfg.confidence_threshold),
                ),
            )
        })
        .collect()
}

/// `GET /v1/benchmark/threshold-search`: the searchable models (with their stored devices and
/// recommended device), datasets, tags and classes, the configured objective and thresholds.
/// With nothing to search yet (no results or no stored predictions) it answers 200 with
/// `available: false` and the message, so the page's empty state is not a console error.
pub(in crate::server) async fn search_options(State(state): State<Arc<AppState>>) -> Response {
    crate::server::blocking(move || {
        let (results, preds) = match search_data(&state) {
            Ok(d) => d,
            Err(e) if e.status == StatusCode::BAD_REQUEST => {
                return Json(serde_json::json!({
                    "success": true,
                    "available": false,
                    "message": e.message,
                }))
                .into_response();
            }
            Err(e) => return e.into_response(),
        };
        let cfg = state.config_read().clone();
        Json(serde_json::json!({
            "success": true,
            "available": true,
            "options": crate::benchmark::search::options(&results, &preds),
            "objective": cfg.benchmark.threshold_objective,
            "thresholds": current_thresholds(&cfg),
        }))
        .into_response()
    })
    .await
}

/// `POST /v1/benchmark/threshold-search` (urlencoded): `model` (repeated or comma separated;
/// `all` or none = every model), `device` (none = recommended), `dataset`, `tag`, `class`
/// (repeated), `objective` (`f1`, `f2`, `precision:<p>`, `recall:<r>`; default the configured
/// one), `iou` (0.5). Answers per model the best threshold, P/R/F1/F2/FP per image at the best
/// and the configured thresholds, the exact curve and the 0.05 grid, per-dataset and per-class
/// bests. 400 on bad input, when nothing is stored ("run a benchmark first") or nothing
/// matches.
pub(in crate::server) async fn threshold_search(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || threshold_search_blocking(&state, &form)).await
}

/// [`threshold_search`] off the runtime threads: it parses the results and the stored
/// predictions and runs the exhaustive search (SV-06, BQ-04).
fn threshold_search_blocking(state: &AppState, form: &[(String, String)]) -> Response {
    let cfg = state.config_read().clone();
    let req = match crate::benchmark::search::SearchRequest::from_form(
        form,
        cfg.benchmark.threshold_objective,
    ) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let (results, preds) = match search_data(state) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let current = current_thresholds(&cfg);
    let lookup = |m: &str| {
        current.get(&crate::registry::normalize_name(m)).map(|t| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a confidence threshold (0..1); f32 is the precision the models use"
            )]
            let t = *t as f32;
            t
        })
    };
    match crate::benchmark::search::search(&results, &preds, &req, &lookup) {
        Ok(out) if out.results.is_empty() => json_error(
            StatusCode::BAD_REQUEST,
            format!("nothing to search: {}", out.errors.join("; ")),
        ),
        Ok(out) => {
            let csv = crate::benchmark::search::to_csv(&out);
            Json(serde_json::json!({
                "success": true,
                "request": req,
                "results": out.results,
                "errors": out.errors,
                "elapsedMs": out.elapsed_ms,
                "csv": csv,
                "note": BLUE_IRIS_THRESHOLD_NOTE,
            }))
            .into_response()
        }
        Err(e) => json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

/// Shown with threshold changes: how the server threshold relates to Blue Iris's settings.
pub(super) const BLUE_IRIS_THRESHOLD_NOTE: &str = "Blue Iris may send its own min_confidence \
     with each request (it then replaces this threshold for that request), and each camera's \
     minimum confidence filters the returned objects again.";

/// `POST /v1/benchmark/apply-threshold`: set per-model `confidence_threshold` in the config.
/// - `model=<name>&threshold=<0..1>`: one model (repeat the pair for several);
///   `threshold=global` clears the model's own threshold (the Undo of a first apply);
/// - `model=<name>&threshold=<0..1>&device=<spec>`: a `device` right after a pair writes that
///   device's confidence profile instead (`threshold=global` or `none` removes it); the
///   model's own threshold is left alone (`device=` empty or `global`: the model's threshold);
/// - `all=1` (optionally limited by repeated `model`): every best threshold of the saved results.
///
/// Thresholds and profiles apply live (no restart; `restart=1` is accepted and ignored,
/// `restarting` and `needsRestart` are always false). Answers JSON with the changes,
/// `profiles` (the profile changes), `previous` (model -> its own threshold before, `0.40` or
/// `global`: what Undo posts back as `threshold`) and `undo` (form pairs that, posted back,
/// restore the previous values, profiles included); 400 on bad input, 404 when `all` finds no
/// best threshold.
pub(in crate::server) async fn apply_threshold(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || apply_threshold_blocking(&state, &form)).await
}

/// [`apply_threshold`] off the runtime threads: it parses the results and writes the config
/// (SV-06).
fn apply_threshold_blocking(state: &Arc<AppState>, form: &[(String, String)]) -> Response {
    let picks = match threshold_picks(state, form) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let mut changes: Vec<ThresholdChange> = Vec::new();
    let mut profiles: Vec<ProfileChange> = Vec::new();
    let mut profile_undo: Vec<(String, String, String)> = Vec::new();
    if let Err(e) = save_config(state, "benchmark apply-threshold", |c| {
        let before = c.clone();
        let set: Vec<(String, f32)> = picks
            .iter()
            .filter(|p| p.device.is_none())
            .filter_map(|p| Some((p.model.clone(), p.threshold?)))
            .collect();
        let clear: Vec<String> = picks
            .iter()
            .filter(|p| p.device.is_none() && p.threshold.is_none())
            .map(|p| p.model.clone())
            .collect();
        let entries: Vec<(String, String, Option<f32>)> = picks
            .iter()
            .filter_map(|p| Some((p.model.clone(), p.device.clone()?, p.threshold)))
            .collect();
        changes = apply_model_thresholds(c, &set)?;
        changes.extend(clear_model_thresholds(c, &clear)?);
        profiles = apply_device_thresholds(c, &entries)?;
        profile_undo = profiles
            .iter()
            .map(|p| {
                let (device, value) = previous_profile(&before, p);
                (p.model.clone(), device, value)
            })
            .collect();
        Ok(())
    }) {
        warn!("benchmark threshold apply refused: {e:#}");
        return json_error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    let described: Vec<String> = changes
        .iter()
        .map(ThresholdChange::describe)
        .chain(profiles.iter().map(ProfileChange::describe))
        .collect();
    let mut message = if described.is_empty() {
        "No change: the models already use these thresholds.".to_string()
    } else {
        format!(
            "Saved {}: {}.",
            state.config_path.display(),
            described.join("; ")
        )
    };
    // Thresholds apply live (`save_config` swapped the workers' settings): no restart.
    let restarting = false;
    if !described.is_empty() {
        message.push_str(" Applied now: the next detections use it.");
    }
    message.push(' ');
    message.push_str(BLUE_IRIS_THRESHOLD_NOTE);
    info!("benchmark threshold apply: {message}");
    // For Undo: POST `model=<m>&threshold=<previous>` (`global` = use the global threshold).
    let previous: serde_json::Map<String, serde_json::Value> = changes
        .iter()
        .map(|c| {
            (
                c.model.clone(),
                c.from
                    .map_or_else(|| "global".to_string(), |t| format!("{t:.2}"))
                    .into(),
            )
        })
        .collect();
    let mut undo: Vec<[String; 2]> = Vec::new();
    for c in &changes {
        undo.push(["model".into(), c.model.clone()]);
        undo.push([
            "threshold".into(),
            c.from
                .map_or_else(|| "global".to_string(), |t| format!("{t:.2}")),
        ]);
    }
    for (model, device, value) in profile_undo {
        undo.push(["model".into(), model]);
        undo.push(["threshold".into(), value]);
        undo.push(["device".into(), device]);
    }
    Json(serde_json::json!({
        "success": true,
        "message": message,
        "changes": changes,
        "profiles": profiles,
        "previous": previous,
        "undo": undo,
        "restarting": restarting,
        "needsRestart": false,
    }))
    .into_response()
}

/// What restores the profile `c` changed: the key the config held it under (the same device
/// may have been spelled with an index, `openvino:gpu.0`) and its value (`0.30`), or `c`'s
/// device and `none` when there was no profile. `before` is the config before the change.
fn previous_profile(before: &Config, c: &ProfileChange) -> (String, String) {
    crate::benchmark::find_model(before, &c.model)
        .and_then(|m| crate::config::lookup_device_threshold_entry(&m.device_thresholds, &c.device))
        .map_or_else(
            || (c.device.clone(), "none".to_string()),
            |(k, t)| (k.to_string(), format!("{t:.2}")),
        )
}

/// Clear the per-model `confidence_threshold` of the named models (they use the global one
/// again): the Undo of a threshold apply on a model that had none. Every name is checked
/// first; on error `config` is unchanged. Returns the entries that changed.
fn clear_model_thresholds(
    config: &mut Config,
    names: &[String],
) -> anyhow::Result<Vec<ThresholdChange>> {
    use crate::registry::normalize_name;
    let global = config.confidence_threshold;
    for n in names {
        if !config
            .models
            .iter()
            .any(|m| normalize_name(&m.effective_name()) == normalize_name(n))
        {
            anyhow::bail!("'{n}' is not one of the configured models");
        }
    }
    let mut changes = Vec::new();
    for n in names {
        let key = normalize_name(n);
        for m in config
            .models
            .iter_mut()
            .filter(|m| normalize_name(&m.effective_name()) == key)
        {
            if let Some(old) = m.confidence_threshold.take() {
                changes.push(ThresholdChange {
                    model: m.effective_name(),
                    from: Some(old),
                    to: global,
                    global,
                });
            }
        }
    }
    Ok(changes)
}

/// One pick of an apply-threshold form: `threshold` None clears the model's own threshold, or
/// removes the profile when `device` is given.
struct ThresholdPick {
    model: String,
    threshold: Option<f32>,
    /// Write this device's confidence profile instead of the model's threshold.
    device: Option<String>,
}

/// The picks of an apply-threshold form: `all=1` takes the saved best thresholds of the
/// configured models (limited to the form's models, if any), else the `model` / `threshold`
/// pairs (`threshold=global`: None, the model's own threshold is cleared), each optionally
/// followed by a `device` (a profile for that device; `global` / `none` removes it).
fn threshold_picks(
    state: &AppState,
    form: &[(String, String)],
) -> Result<Vec<ThresholdPick>, ApiError> {
    let models = form_models(form);
    if form_field(form, "all").is_some_and(truthy) {
        let results = results_for_apply(state)?;
        let only = (!models.is_empty()).then_some(models.as_slice());
        let cfg = state.config_read().clone();
        let picks: Vec<ThresholdPick> = results
            .thresholds(only)
            .into_iter()
            .filter(|(m, _)| crate::benchmark::find_model(&cfg, m).is_some())
            .map(|(model, t)| ThresholdPick {
                model,
                threshold: Some(t),
                device: None,
            })
            .collect();
        if picks.is_empty() {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "the saved results have no best threshold for these models",
            ));
        }
        return Ok(picks);
    }
    let usage = || {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "give `model` and `threshold` (repeated in pairs, each optionally followed by \
             `device` for a confidence profile), or `all=1` to apply every best threshold",
        )
    };
    let parse = |t: &str, profile: bool| -> Result<Option<f32>, ApiError> {
        if t.eq_ignore_ascii_case("global") || (profile && t.eq_ignore_ascii_case("none")) {
            return Ok(None);
        }
        t.parse::<f32>().map(Some).map_err(|_| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("threshold: '{t}' is not a number"),
            )
        })
    };
    if !form.iter().any(|(k, _)| k == "device") {
        // The pairs as before: `model`s and `threshold`s matched in order.
        let values: Vec<&str> = form
            .iter()
            .filter(|(k, _)| k == "threshold")
            .map(|(_, v)| v.trim())
            .collect();
        if models.is_empty() || models.len() != values.len() {
            return Err(usage());
        }
        return models
            .iter()
            .zip(values)
            .map(|(m, t)| {
                Ok(ThresholdPick {
                    model: m.clone(),
                    threshold: parse(t, false)?,
                    device: None,
                })
            })
            .collect();
    }
    // With profiles: each `model` starts a pick; its `threshold` and optional `device` follow.
    let mut groups: Vec<(String, Option<&str>, Option<&str>)> = Vec::new();
    for (k, v) in form {
        let v = v.trim();
        match k.as_str() {
            "model" if !v.is_empty() => groups.push((v.to_string(), None, None)),
            "threshold" => match groups.last_mut() {
                Some(g) if g.1.is_none() => g.1 = Some(v),
                _ => return Err(usage()),
            },
            "device" => match groups.last_mut() {
                Some(g) if g.1.is_some() && g.2.is_none() => g.2 = Some(v),
                _ => return Err(usage()),
            },
            _ => {}
        }
    }
    if groups.is_empty() {
        return Err(usage());
    }
    groups
        .into_iter()
        .map(|(model, t, device)| {
            let t = t.ok_or_else(usage)?;
            // `device=` empty or `global`: the model's own threshold, as without `device`.
            let device = device
                .filter(|d| !d.is_empty() && !d.eq_ignore_ascii_case("global"))
                .map(str::to_string);
            Ok(ThresholdPick {
                threshold: parse(t, device.is_some())?,
                model,
                device,
            })
        })
        .collect()
}

/// `POST /v1/benchmark/apply-profiles` (urlencoded): write per-device confidence profiles
/// ([`crate::config::ModelConfig::device_thresholds`]) and per-model devices in one
/// `save_config` (one revision, one Undo):
/// - `all=1` (optionally limited by repeated `model`): every measured device's best threshold
///   of every configured model of the saved results;
/// - `use=<device>|<model>`: set the model's device (`global` = follow the global device);
/// - `profile=<device>|<0.xx|none>|<model>`: one profile (`none` removes it); split on the first
///   two `|`, so the model name may contain `|`.
///
/// Any mix and repetition is allowed; devices are applied first, a later profile entry for the
/// same model and device wins. `restart=1` restarts the server when a device change cannot be
/// applied by a per-model reload. Answers `{success, message, changes: {devices, profiles},
/// undo: [[key, value], ...], needsRestart, restarting}`: `undo` posted back to this endpoint
/// restores the previous config. 400 on bad input (nothing changes), 404 when `all` finds no
/// results or nothing to apply.
pub(in crate::server) async fn apply_profiles(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || apply_profiles_blocking(&state, &form)).await
}

/// What an apply-profiles form asks for: device changes, then profile entries.
#[derive(Debug, Default, PartialEq)]
struct ProfileForm {
    uses: Vec<(String, Option<String>)>,
    profiles: Vec<(String, String, Option<f32>)>,
}

/// Parse an apply-profiles form (see [`apply_profiles`]); `all=1` reads the saved results.
fn profile_form(state: &AppState, form: &[(String, String)]) -> Result<ProfileForm, ApiError> {
    let bad = |m: String| ApiError::new(StatusCode::BAD_REQUEST, m);
    let mut out = ProfileForm::default();
    for (k, v) in form {
        match k.as_str() {
            "use" => {
                let Some((device, model)) = v.split_once('|') else {
                    return Err(bad(format!(
                        "use: '{v}' is not <device>|<model> (e.g. ort:coreml|IPcam-general)"
                    )));
                };
                let model = model.trim();
                if model.is_empty() {
                    return Err(bad(format!("use: '{v}' names no model")));
                }
                out.uses
                    .push((model.to_string(), Some(device.trim().to_string())));
            }
            "profile" => {
                let mut parts = v.splitn(3, '|');
                let (Some(device), Some(value), Some(model)) =
                    (parts.next(), parts.next(), parts.next())
                else {
                    return Err(bad(format!(
                        "profile: '{v}' is not <device>|<0.xx or none>|<model> \
                         (e.g. ort:coreml|0.35|IPcam-general)"
                    )));
                };
                let (device, value, model) = (device.trim(), value.trim(), model.trim());
                if model.is_empty() {
                    return Err(bad(format!("profile: '{v}' names no model")));
                }
                let value = if value.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(value.parse::<f32>().map_err(|_| {
                        bad(format!(
                            "profile: '{value}' is not a number (or `none` to remove the profile)"
                        ))
                    })?)
                };
                out.profiles
                    .push((model.to_string(), device.to_string(), value));
            }
            _ => {}
        }
    }
    if form_field(form, "all").is_some_and(truthy) {
        let models = form_models(form);
        let results = results_for_apply(state)?;
        let only = (!models.is_empty()).then_some(models.as_slice());
        let cfg = state.config_read().clone();
        // Results may name models that were removed from the config since: skip those.
        let all: Vec<(String, String, Option<f32>)> = results
            .device_thresholds(only)
            .into_iter()
            .filter(|(m, _, _)| crate::benchmark::find_model(&cfg, m).is_some())
            .map(|(m, d, t)| (m, d, Some(t)))
            .collect();
        if all.is_empty() && out.uses.is_empty() && out.profiles.is_empty() {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "the saved results have no per-device best threshold for these models",
            ));
        }
        // Explicit entries come after, so they win over the measured ones.
        out.profiles.splice(0..0, all);
    } else if out.uses.is_empty() && out.profiles.is_empty() {
        return Err(bad(
            "give `all=1`, `use=<device>|<model>` or `profile=<device>|<0.xx or none>|<model>`"
                .into(),
        ));
    }
    Ok(out)
}

/// [`apply_profiles`] off the runtime threads: it parses the results, writes the config once
/// and reloads the models whose device changed (SV-06).
fn apply_profiles_blocking(state: &Arc<AppState>, form: &[(String, String)]) -> Response {
    let restart = form_field(form, "restart").is_some_and(truthy);
    let asked = match profile_form(state, form) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let mut devices: Vec<DeviceChange> = Vec::new();
    let mut profiles: Vec<ProfileChange> = Vec::new();
    let mut undo: Vec<[String; 2]> = Vec::new();
    let saved = save_config(state, "benchmark apply-profiles", |c| {
        let before = c.clone();
        devices = apply_model_devices(c, &asked.uses)?;
        profiles = apply_device_thresholds(c, &asked.profiles)?;
        undo = devices
            .iter()
            .map(|d| {
                let from = d.from.clone().unwrap_or_else(|| "global".into());
                ["use".to_string(), format!("{from}|{}", d.model)]
            })
            .chain(profiles.iter().map(|p| {
                let (device, value) = previous_profile(&before, p);
                [
                    "profile".to_string(),
                    format!("{device}|{value}|{}", p.model),
                ]
            }))
            .collect();
        Ok(())
    });
    if let Err(e) = saved {
        warn!("benchmark apply-profiles refused: {e:#}");
        return json_error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    let described: Vec<String> = devices
        .iter()
        .map(DeviceChange::describe)
        .chain(profiles.iter().map(ProfileChange::describe))
        .collect();
    let mut message = if described.is_empty() {
        "No change: the config already has these devices and confidence profiles.".to_string()
    } else {
        format!(
            "Saved {}: {}.",
            state.config_path.display(),
            described.join("; ")
        )
    };
    let (mut restarting, mut needs_restart) = (false, false);
    if !devices.is_empty() {
        let (rest, r, n) = device_outcome(state, &devices, restart);
        message.push_str(&rest);
        (restarting, needs_restart) = (r, n);
    }
    if !profiles.is_empty() {
        message.push_str(
            " Confidence profiles apply now: the next detections on those devices use them. ",
        );
        message.push_str(BLUE_IRIS_THRESHOLD_NOTE);
    }
    info!("benchmark apply-profiles: {message}");
    Json(serde_json::json!({
        "success": true,
        "message": message,
        "changes": {"devices": devices, "profiles": profiles},
        "undo": undo,
        "needsRestart": needs_restart,
        "restarting": restarting,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::super::tests::{call, form_body, get, post, profile_results, state};
    use super::*;
    use crate::server::router;
    use std::time::Duration;
    use tower::ServiceExt;
    #[tokio::test]
    async fn apply_one_and_all() {
        let st = state();
        let cfg_path = st.config_path.clone();

        // `all` without results: 404.
        let (s, _) = call(&st, post("/v1/benchmark/apply", "all=1")).await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        // One model.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/apply", "model=DFINE-S&device=ort%3Acoreml"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"][0]["model"], "dfine-s");
        assert_eq!(v["changes"][0]["to"], "ort:coreml");
        assert_eq!(v["restarting"], false);
        // dfine-s is disabled: nothing to reload or restart.
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("Used when the model is enabled"),
            "{v}"
        );
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[1].device.as_deref(), Some("ort:coreml"));
        assert_eq!(*st.config_read(), saved);

        // Invalid input changes nothing.
        for body in [
            "model=dfine-s&device=vulkan",
            "model=zzz&device=ort%3Acpu",
            "model=dfine-s",
            "",
        ] {
            let (s, _) = call(&st, post("/v1/benchmark/apply", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(Config::load(&cfg_path).unwrap(), saved, "{body}");
        }

        // Every recommendation of the saved results; models no longer configured are skipped.
        let mk = |name: &str, rec: Option<&str>| {
            let mut m = crate::benchmark::ModelResult::failed(name, "", String::new());
            m.error = None;
            m.recommended = rec.map(str::to_string);
            m
        };
        let results = BenchmarkResults::new(
            Default::default(),
            Default::default(),
            vec![
                mk("ipcam-general", Some("openvino:cpu")),
                mk("dfine-s", Some("ort:coreml")),
                mk("removed", Some("ort:cpu")),
                mk("none", None),
            ],
        );
        results.save(&results_path(&cfg_path)).unwrap();
        let (s, v) = call(&st, post("/v1/benchmark/apply", "all=1&restart=1")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        // dfine-s already had ort:coreml: one change.
        assert_eq!(v["changes"].as_array().unwrap().len(), 1, "{v}");
        assert_eq!(v["changes"][0]["model"], "ipcam-general");
        assert_eq!(v["restarting"], true);
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[0].device.as_deref(), Some("openvino:cpu"));
        tokio::time::timeout(Duration::from_secs(5), st.restart.cancelled())
            .await
            .expect("restart requested");

        // The config page shows the per-model Device select and the recommendation.
        let resp = router(st.clone()).oneshot(get("/config")).await.unwrap();
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(html.contains("name=\"device.ipcam-general\""), "{html}");
        // The global option ("" = follow the global device; the config page may add data
        // attributes to it).
        assert!(
            html.split("<option value=\"\"").skip(1).any(|o| o
                .split_once('>')
                .is_some_and(|(_, t)| t.starts_with("Global (auto"))),
            "{html}"
        );
        // Configured but not loaded on it (the test workers never load): restart to use.
        assert!(
            html.contains("benchmark: openvino:cpu (configured; restart to use)"),
            "{html}"
        );
        assert!(
            html.contains("benchmark: ort:coreml (configured; restart to use)"),
            "{html}"
        );
    }

    #[tokio::test]
    async fn apply_threshold_one_and_all() {
        let st = state();
        let cfg_path = st.config_path.clone();
        let (s, _) = call(&st, post("/v1/benchmark/apply-threshold", "all=1")).await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        // One model.
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/apply-threshold",
                "model=DFINE-S&threshold=0.35",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"][0]["model"], "dfine-s");
        assert_eq!(v["changes"][0]["to"].as_f64().unwrap() as f32, 0.35);
        let msg = v["message"].as_str().unwrap();
        assert!(msg.contains("0.50 (global) -> 0.35"), "{msg}");
        assert!(msg.contains("min_confidence"), "{msg}");
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[1].confidence_threshold, Some(0.35));
        assert_eq!(saved.models[0].confidence_threshold, None);
        assert_eq!(*st.config_read(), saved);
        // The status view shows the effective thresholds.
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(
            v["config"]["models"][1]["confidenceThreshold"]
                .as_f64()
                .unwrap() as f32,
            0.35
        );
        assert_eq!(v["config"]["models"][0]["ownThreshold"], false);

        // Invalid input changes nothing.
        for body in [
            "model=dfine-s&threshold=0",
            "model=dfine-s&threshold=1.5",
            "model=dfine-s&threshold=abc",
            "model=zzz&threshold=0.3",
            "model=dfine-s",
            "",
        ] {
            let (s, _) = call(&st, post("/v1/benchmark/apply-threshold", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(Config::load(&cfg_path).unwrap(), saved, "{body}");
        }

        // Every best threshold of the saved results; unconfigured models are skipped, a model
        // without advice is left alone.
        let mk = |name: &str, t: Option<f32>| {
            let mut m = crate::benchmark::ModelResult::failed(name, "", String::new());
            m.error = None;
            m.threshold = t.map(crate::benchmark::threshold::test_advice);
            m
        };
        let results = BenchmarkResults::new(
            Default::default(),
            Default::default(),
            vec![
                mk("ipcam-general", Some(0.42)),
                mk("dfine-s", Some(0.35)),
                mk("removed", Some(0.3)),
                mk("none", None),
            ],
        );
        results.save(&results_path(&cfg_path)).unwrap();
        let (s, v) = call(
            &st,
            post("/v1/benchmark/apply-threshold", "all=1&restart=1"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"].as_array().unwrap().len(), 1, "{v}");
        assert_eq!(v["changes"][0]["model"], "ipcam-general");
        // Thresholds apply live: no restart even when asked for.
        assert_eq!(v["restarting"], false);
        assert!(
            v["message"].as_str().unwrap().contains("Applied now"),
            "{v}"
        );
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[0].confidence_threshold, Some(0.42));
        // Devices are untouched by the threshold apply.
        assert_eq!(saved.models[0].device, None);
        assert!(!st.restart.is_cancelled());
        // Several pairs at once (the search page's Apply all).
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/apply-threshold",
                "model=ipcam-general&threshold=0.3&model=dfine-s&threshold=0.25",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"].as_array().unwrap().len(), 2);
        let (s, _) = call(
            &st,
            post(
                "/v1/benchmark/apply-threshold",
                "model=ipcam-general&threshold=0.3&model=dfine-s",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[1].confidence_threshold, Some(0.25));
        // Back to the results' values for the check below.
        let (s, _) = call(&st, post("/v1/benchmark/apply-threshold", "all=1")).await;
        assert_eq!(s, StatusCode::OK);
        // Limited to one model: nothing left to change.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/apply-threshold", "all=1&model=dfine-s"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(v["changes"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn run_tab_browses_for_a_folder() {
        let st = state();
        let (s, page) = call(&st, get("/benchmark")).await;
        assert_eq!(s, StatusCode::OK);
        let page = page.as_str().unwrap();
        assert!(page.contains("data-browse=\"folder\" data-target=\"add-folder\""));
        assert!(page.contains("<dialog id=\"browse-dialog\">"));
        assert!(page.contains("<script src=\"/static/browse.js?v="));
        // The low-priority box is a check label; device ids keep their line.
        assert!(
            page.contains("<label class=\"check\"><input type=\"checkbox\" name=\"low_priority\"")
        );
        assert!(page.contains("class=\"bench-settings\""));
    }

    #[tokio::test]
    async fn threshold_search_endpoint() {
        let st = state();
        let cfg_path = st.config_path.clone();
        // Nothing stored yet: the options GET answers 200 `available: false`, the search POST
        // 400; both tell to run a benchmark.
        for (req, status) in [
            (get("/v1/benchmark/threshold-search"), StatusCode::OK),
            (
                post("/v1/benchmark/threshold-search", "model=all"),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let (s, v) = call(&st, req).await;
            assert_eq!(s, status, "{v}");
            assert!(
                v["message"]
                    .as_str()
                    .unwrap()
                    .contains("run a benchmark first"),
                "{v}"
            );
            if status == StatusCode::OK {
                assert_eq!(v["available"], false, "{v}");
                assert!(v.get("options").is_none(), "{v}");
            }
        }
        let (results, preds) = crate::benchmark::search::tests::fixture();
        results.save(&results_path(&cfg_path)).unwrap();
        // Results but no stored predictions (an older version's results): still 400 / unavailable.
        let (s, _) = call(&st, post("/v1/benchmark/threshold-search", "")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, v) = call(&st, get("/v1/benchmark/threshold-search")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["available"], false, "{v}");
        preds
            .save(&crate::benchmark::search::preds_path(&cfg_path))
            .unwrap();

        let (s, v) = call(&st, get("/v1/benchmark/threshold-search")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["available"], true, "{v}");
        assert_eq!(v["options"]["models"][0]["model"], "m");
        assert_eq!(
            v["options"]["models"][0]["devices"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(v["options"]["models"][0]["recommended"], "ort:coreml");
        assert_eq!(v["options"]["tags"], serde_json::json!(["day", "night"]));
        assert_eq!(v["objective"], "f1");

        // A valid search (default: every model, recommended device, f1).
        let (s, v) = call(&st, post("/v1/benchmark/threshold-search", "")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let r = &v["results"][0];
        assert_eq!(r["device"], "ort:coreml");
        assert_eq!(r["best"]["threshold"].as_f64().unwrap() as f32, 0.35);
        assert_eq!(r["grid"].as_array().unwrap().len(), 19);
        assert!(v["csv"].as_str().unwrap().starts_with("model,device"));
        assert!(v["elapsedMs"].as_f64().unwrap() >= 0.0);

        // Filters.
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/threshold-search",
                "model=m&device=openvino%3Acpu&tag=night&class=person&objective=recall%3A0.5&iou=0.5",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let r = &v["results"][0];
        assert_eq!(r["device"], "openvino:cpu");
        assert_eq!((r["images"].as_u64(), r["gt"].as_u64()), (Some(1), Some(1)));
        assert_eq!(r["objective"], "recall:0.5");
        assert_eq!(r["classes"], serde_json::json!(["person"]));

        // Frame-level objective from stored data: Youden's J. Frames (real set): a person+ .85,
        // a car+ .4, b person+ .6, b car- (no car prediction: 0). J = 1 for thresholds up to
        // 0.4 -> plateau [0.05, 0.4], midpoint 0.22.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/threshold-search", "objective=youden"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let r = &v["results"][0];
        assert_eq!(r["objective"], "youden");
        assert_eq!(r["best"]["threshold"].as_f64().unwrap() as f32, 0.22);
        assert_eq!(r["best"]["exact_threshold"].as_f64().unwrap() as f32, 0.4);
        assert_eq!(r["best"]["point"]["frame"]["tpr"], 1.0);
        assert_eq!(r["best"]["point"]["frame"]["fpr"], 0.0);
        assert_eq!(r["roc"]["auc"], 1.0);
        assert!(r["roc"]["curve"].as_array().unwrap().len() >= 2);
        assert!(v["csv"].as_str().unwrap().contains("roc_auc"));
        let (s, v) = call(
            &st,
            post("/v1/benchmark/threshold-search", "objective=fpr%3A0.05"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["results"][0]["objective"], "fpr:0.05");

        // Bad input / nothing matching: 400 with a message.
        for body in [
            "objective=f9",
            "iou=2",
            "model=zzz",
            "tag=fog",
            "device=ort%3Acpu",
        ] {
            let (s, v) = call(&st, post("/v1/benchmark/threshold-search", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert!(v["message"].is_string());
        }
    }
    #[tokio::test]
    async fn apply_profiles_all_use_and_undo() {
        let st = state();
        let cfg_path = st.config_path.clone();
        let url = "/v1/benchmark/apply-profiles";
        // `all` without results: 404; nothing asked: 400.
        let (s, _) = call(&st, post(url, "all=1")).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        let (s, v) = call(&st, post(url, "")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        profile_results(Default::default())
            .save(&results_path(&cfg_path))
            .unwrap();
        // The config as it is before (saved once so the file and memory agree).
        st.config.update("test", |_| Ok(())).unwrap();
        let original = st.config_read().clone();
        let rev = st.config.snapshot().0;

        // Every measured device of every configured model, in one revision; `removed` (not
        // configured) is skipped, the failed GPU has no best.
        let (s, v) = call(&st, post(url, "all=1")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(st.config.snapshot().0, rev + 1, "one save");
        let p = v["changes"]["profiles"].as_array().unwrap();
        let got: Vec<(&str, &str, f64)> = p
            .iter()
            .map(|c| {
                assert_eq!(c["from"], serde_json::Value::Null);
                (
                    c["model"].as_str().unwrap(),
                    c["device"].as_str().unwrap(),
                    c["to"].as_f64().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                ("ipcam-general", "openvino:cpu", 0.4),
                ("ipcam-general", "ort:cpu", 0.33),
                ("dfine-s", "ort:coreml", 0.3)
            ]
        );
        assert_eq!(v["changes"]["devices"], serde_json::json!([]));
        assert_eq!(
            (&v["needsRestart"], &v["restarting"]),
            (&serde_json::json!(false), &serde_json::json!(false))
        );
        assert!(v["message"].as_str().unwrap().contains("apply now"), "{v}");
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(*st.config_read(), saved);
        assert_eq!(
            saved.models[0].device_thresholds,
            [
                ("openvino:cpu".to_string(), 0.4),
                ("ort:cpu".to_string(), 0.33)
            ]
            .into()
        );
        assert_eq!(
            saved.models[1].device_thresholds,
            [("ort:coreml".to_string(), 0.3)].into()
        );
        // Thresholds and devices untouched.
        assert_eq!(saved.models[0].confidence_threshold, None);
        assert_eq!(saved.models[0].device, None);
        assert_eq!(
            v["undo"],
            serde_json::json!([
                ["profile", "openvino:cpu|none|ipcam-general"],
                ["profile", "ort:cpu|none|ipcam-general"],
                ["profile", "ort:coreml|none|dfine-s"]
            ])
        );
        // Again: nothing to change, no new revision.
        let (s, v2) = call(&st, post(url, "all=1&model=dfine-s")).await;
        assert_eq!(s, StatusCode::OK, "{v2}");
        assert_eq!(v2["changes"]["profiles"], serde_json::json!([]));
        assert_eq!(st.config.snapshot().0, rev + 1);
        // Undo restores the config exactly.
        let (s, u) = call(&st, post(url, &form_body(&v["undo"]))).await;
        assert_eq!(s, StatusCode::OK, "{u}");
        assert_eq!(Config::load(&cfg_path).unwrap(), original);
        assert_eq!(*st.config_read(), original);

        // `use` and explicit profiles (another spelling of a device already profiled under its
        // index; the model name after the second `|` is taken whole).
        st.config
            .update("test", |c| {
                c.models[0]
                    .device_thresholds
                    .insert("openvino:gpu.0".into(), 0.2);
                c.models[1].device = Some("ort:coreml".into());
                Ok(())
            })
            .unwrap();
        let before = st.config_read().clone();
        let body = "use=ort%3Acpu%7Cipcam-general&use=global%7Cdfine-s\
                    &profile=GPU%7C0.45%7Cipcam-general&profile=ort%3Acpu%7C0.3%7Cdfine-s";
        let (s, v) = call(&st, post(url, body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let d = &v["changes"]["devices"];
        assert_eq!(
            (&d[0]["model"], &d[0]["to"]),
            (
                &serde_json::json!("ipcam-general"),
                &serde_json::json!("ort:cpu")
            )
        );
        assert_eq!(
            (&d[1]["model"], &d[1]["to"]),
            (&serde_json::json!("dfine-s"), &serde_json::Value::Null)
        );
        // ipcam-general is enabled: its device change needs a restart (the test workers never
        // reload); not asked to restart.
        assert_eq!(v["needsRestart"], true, "{v}");
        assert_eq!(v["restarting"], false);
        assert!(!st.restart.is_cancelled());
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[0].device.as_deref(), Some("ort:cpu"));
        assert_eq!(saved.models[1].device, None);
        assert_eq!(
            saved.models[0].device_thresholds,
            [("openvino:gpu".to_string(), 0.45)].into()
        );
        assert_eq!(
            saved.models[1].device_thresholds,
            [("ort:cpu".to_string(), 0.3)].into()
        );
        assert_eq!(
            v["undo"],
            serde_json::json!([
                ["use", "global|ipcam-general"],
                ["use", "ort:coreml|dfine-s"],
                ["profile", "openvino:gpu.0|0.20|ipcam-general"],
                ["profile", "ort:cpu|none|dfine-s"]
            ])
        );
        let (s, u) = call(&st, post(url, &form_body(&v["undo"]))).await;
        assert_eq!(s, StatusCode::OK, "{u}");
        assert_eq!(Config::load(&cfg_path).unwrap(), before);
        assert_eq!(*st.config_read(), before);

        // Bad input: 400, nothing saved.
        let rev = st.config.snapshot().0;
        for body in [
            "use=ort%3Acpu",
            "use=ort%3Acpu%7C",
            "use=vulkan%7Cipcam-general",
            "use=ort%3Acpu%7Czzz",
            "profile=ort%3Acpu%7C0.3",
            "profile=ort%3Acpu%7Cabc%7Cipcam-general",
            "profile=auto%7C0.3%7Cipcam-general",
            "profile=ort%3Acpu%7C1.5%7Cipcam-general",
            "profile=ort%3Acpu%7C0%7Cipcam-general",
            // The model is `ipcam-general|x` (split on the first two `|` only).
            "profile=ort%3Acpu%7C0.3%7Cipcam-general%7Cx",
            // One bad entry refuses the good ones too.
            "profile=ort%3Acpu%7C0.3%7Cipcam-general&use=vulkan%7Cdfine-s",
            "model=ipcam-general",
        ] {
            let (s, v) = call(&st, post(url, body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert!(v["message"].is_string(), "{body}");
            assert_eq!(Config::load(&cfg_path).unwrap(), before, "{body}");
            assert_eq!(st.config.snapshot().0, rev, "{body}");
        }
        let (_, v) = call(
            &st,
            post(url, "profile=ort%3Acpu%7C0.3%7Cipcam-general%7Cx"),
        )
        .await;
        assert!(
            v["message"].as_str().unwrap().contains("'ipcam-general|x'"),
            "{v}"
        );
    }

    #[tokio::test]
    async fn apply_threshold_with_device_writes_a_profile() {
        let st = state();
        let cfg_path = st.config_path.clone();
        let url = "/v1/benchmark/apply-threshold";
        st.config.update("test", |_| Ok(())).unwrap();
        let original = st.config_read().clone();
        let (s, v) = call(
            &st,
            post(url, "model=dfine-s&threshold=0.3&device=ORT%3ACoreML"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"], serde_json::json!([]));
        let p = &v["profiles"][0];
        assert_eq!(
            (&p["model"], &p["device"]),
            (
                &serde_json::json!("dfine-s"),
                &serde_json::json!("ort:coreml")
            )
        );
        assert_eq!(
            (&p["from"], &p["to"]),
            (&serde_json::Value::Null, &serde_json::json!(0.3))
        );
        assert_eq!(
            (&v["needsRestart"], &v["restarting"]),
            (&serde_json::json!(false), &serde_json::json!(false))
        );
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("dfine-s on ort:coreml: none -> 0.30"),
            "{v}"
        );
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(
            saved.models[1].device_thresholds,
            [("ort:coreml".to_string(), 0.3)].into()
        );
        assert_eq!(saved.models[1].confidence_threshold, None);
        assert_eq!(
            v["undo"],
            serde_json::json!([
                ["model", "dfine-s"],
                ["threshold", "none"],
                ["device", "ort:coreml"]
            ])
        );
        // Undo removes it again.
        let (s, u) = call(&st, post(url, &form_body(&v["undo"]))).await;
        assert_eq!(s, StatusCode::OK, "{u}");
        assert_eq!(Config::load(&cfg_path).unwrap(), original);

        // A model threshold and a profile in one post; `device=global` is the model threshold.
        let (s, v) = call(
            &st,
            post(
                url,
                "model=ipcam-general&threshold=0.3&device=global&model=dfine-s&threshold=0.25&device=ort%3Acpu",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"].as_array().unwrap().len(), 1);
        assert_eq!(v["changes"][0]["model"], "ipcam-general");
        assert_eq!(v["profiles"].as_array().unwrap().len(), 1);
        assert_eq!(v["previous"]["ipcam-general"], "global");
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[0].confidence_threshold, Some(0.3));
        assert_eq!(
            saved.models[1].device_thresholds,
            [("ort:cpu".to_string(), 0.25)].into()
        );
        let (s, _) = call(&st, post(url, &form_body(&v["undo"]))).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(Config::load(&cfg_path).unwrap(), original);

        // Bad input changes nothing.
        for body in [
            "device=ort%3Acpu&model=dfine-s&threshold=0.3",
            "model=dfine-s&device=ort%3Acpu",
            "model=dfine-s&threshold=0.3&device=auto",
            "model=dfine-s&threshold=0.3&device=vulkan",
            "model=dfine-s&threshold=abc&device=ort%3Acpu",
            "model=dfine-s&threshold=0.3&device=ort%3Acpu&device=ort%3Acoreml",
            "model=zzz&threshold=0.3&device=ort%3Acpu",
        ] {
            let (s, v) = call(&st, post(url, body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert_eq!(Config::load(&cfg_path).unwrap(), original, "{body}");
        }
    }
}
