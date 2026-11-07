//! The safe default switch of the Models tab's Use card: `GET /v1/training/switch-preview`
//! (what changes, the settings copied, the suggested threshold, the comparison with the current
//! default) and `POST /training/revert` (also on Home). The pure part is
//! [`crate::training::switch`]; `POST /training/use` ([`super::training`]) does the switch.

use super::train_area::MODELS;
use super::training::{field, records, view};
use super::{AppState, blocking, devices_snapshot, render};
use crate::benchmark::{BenchmarkResults, configured_device, find_model, results_path};
use crate::config::{Config, ModelConfig};
use crate::training::autolabel::catalog::{ModelClasses, model_classes};
use crate::training::job::TrainOutput;
use crate::training::switch::{self, Side, Suggestion, SwitchDiff};
use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A model's name for the registry's map of loaded devices.
fn key(name: &str) -> String {
    crate::registry::normalize_name(name)
}

/// The device `m` runs on: the loaded one, else what the config resolves it to, else its spec.
fn device_of(state: &AppState, cfg: &Config, m: &ModelConfig) -> String {
    if let Some(d) = super::benchmark::loaded_devices(state).remove(&key(&m.effective_name())) {
        return d;
    }
    let snap = devices_snapshot(state);
    configured_device(cfg, m, &snap.model_selection(state, cfg, m))
        .unwrap_or_else(|| cfg.device_for(m))
}

/// What the page's threshold line shows.
pub(super) fn suggestion_for(
    state: &AppState,
    cfg: &Config,
    m: &ModelConfig,
) -> Result<Suggestion, String> {
    let name = m.effective_name();
    let device = device_of(state, cfg, m);
    let results = BenchmarkResults::load(&results_path(&state.config_path))
        .map_err(|e| format!("the saved benchmark results do not read: {e:#}"))?
        .ok_or_else(|| format!("{name} has no benchmark yet"))?;
    let r = results
        .models
        .iter()
        .find(|r| key(&r.model) == key(&name))
        .ok_or_else(|| format!("{name} has no benchmark yet"))?;
    switch::suggest(r, &name, Some(&device))
}

/// `GET /v1/training/switch-preview` answer.
#[derive(Debug, Serialize)]
pub(super) struct Preview {
    /// The new model is the default already: only its threshold can change.
    already_default: bool,
    #[serde(flatten)]
    diff: SwitchDiff,
    /// The threshold line, or why there is none.
    suggestion: Option<Suggestion>,
    no_suggestion: Option<String>,
    /// The new model and the current default on the same frames.
    compare: String,
    /// A share of the benchmark above that is its own training frames.
    trained_on: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct PreviewQuery {
    id: String,
}

/// The classes of `name` for one side of the diff, from `all` (or `fallback` for a trained
/// model whose class file cannot be read).
fn classes_of(
    all: &[ModelClasses],
    name: &str,
    fallback: &[String],
) -> (Vec<String>, Option<String>) {
    match all.iter().find(|m| key(&m.name) == key(name)) {
        Some(m) if !m.classes.is_empty() => (m.classes.clone(), None),
        Some(m) if fallback.is_empty() => (Vec::new(), m.error.clone()),
        _ => (fallback.to_vec(), None),
    }
}

pub(super) fn preview_of(
    state: &AppState,
    rec: &crate::training::service::RunRecord,
) -> anyhow::Result<Preview> {
    let out: &TrainOutput = rec
        .output
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("run {} has no installed model", rec.id))?;
    let mut cfg = state.config_read().clone();
    let from = switch::current_default(&cfg);
    if find_model(&cfg, &out.name).is_none() {
        crate::training::cli::add_trained_model(&mut cfg, out)?;
    }
    let new = find_model(&cfg, &out.name)
        .ok_or_else(|| anyhow::anyhow!("no configured model named {}", out.name))?
        .clone();
    let cur = from
        .as_deref()
        .and_then(|f| find_model(&cfg, f))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no model is enabled as the default"))?;
    let classes = model_classes(&cfg);
    let (cur_classes, cur_unknown) = classes_of(&classes, &cur.effective_name(), &[]);
    let (new_classes, new_unknown) = classes_of(&classes, &out.name, &out.classes);
    let cur_device = device_of(state, &cfg, &cur);
    let new_device = device_of(state, &cfg, &new);
    let diff = switch::diff(
        &cfg,
        &cur,
        &new,
        Side {
            classes: &cur_classes,
            unknown: cur_unknown.as_deref(),
            device: &cur_device,
        },
        Side {
            classes: &new_classes,
            unknown: new_unknown.as_deref(),
            device: &new_device,
        },
    );
    let (suggestion, no_suggestion) = match suggestion_for(state, &cfg, &new) {
        Ok(s) => (Some(s), None),
        Err(e) => (None, Some(e)),
    };
    let bench = rec.benchmark.as_ref();
    Ok(Preview {
        already_default: key(&cur.effective_name()) == key(&out.name),
        diff,
        suggestion,
        no_suggestion,
        compare: crate::training::bench::compare_text(bench, rec.baseline.as_ref()),
        trained_on: bench
            .and_then(|b| b.trained_on_share)
            .and_then(crate::training::bench::trained_chip),
    })
}

/// `GET /v1/training/switch-preview?id=<run>`: what "Use as default" would change, nothing
/// saved. 404 for an unknown run or one without a model.
pub(super) async fn preview(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PreviewQuery>,
) -> Response {
    blocking(move || {
        let rec = records(&state).into_iter().find(|r| r.id == q.id);
        let answer = rec
            .ok_or_else(|| anyhow::anyhow!("no run {}", q.id))
            .and_then(|r| preview_of(&state, &r));
        match answer {
            Ok(p) => Json(p).into_response(),
            Err(e) => (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!(
                        "The switch cannot be previewed: {e:#}. Reload the page; if the run was \
                         removed, pick another."
                    )
                })),
            )
                .into_response(),
        }
    })
    .await
}

/// `POST /training/revert`: undo the last default switch (`forget=1`: drop a record that cannot
/// be used). Answers with the Models tab, from Home too: that is where the message shows.
pub(super) async fn revert(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = super::dataset::parse_form(&body);
        let forget = field(&f, "forget").is_some();
        let mut message = String::new();
        let saved = super::save_config(&state, "default switch", |c| {
            if forget {
                c.previous_default = None;
                message = "The record of the last switch was dropped.".to_string();
            } else {
                let at = chrono::Utc::now().to_rfc3339();
                message = switch::revert(c, &at)?.message();
            }
            Ok(())
        });
        match saved {
            Ok(m) => render(&view(&state, MODELS, Some(format!("{message} {m}")), None)),
            Err(e) => render(&view(
                &state,
                MODELS,
                None,
                Some(format!("Not reverted: {e:#}.")),
            )),
        }
    })
    .await
}
