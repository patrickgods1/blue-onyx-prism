//! The Benchmark pages: `GET /benchmark` (307 to `/benchmark/compare` when results exist, else
//! `/benchmark/run`; the query is kept) and one page per step, each `templates/benchmark/<step>.html`
//! with the shared shell (`templates/benchmark/shell.html`, [`Shell`]): `/benchmark/run` (the
//! form: images, models, devices, options), `/benchmark/compare`, `/benchmark/tune`,
//! `/benchmark/history`, `/benchmark/images`. Everything but the Run form is drawn by the page
//! scripts (`assets/bench-*.js`) from the JSON endpoints (`/v1/benchmark/compare`, `/tune`,
//! `/tune/images`, `/history`, `/images`); a page polls `GET /v1/benchmark?only=run` for the
//! run's progress and the Re-score banner.
//!
//! Old links (`/benchmark#tab-tune`, `#m-<model>`, ...) arrive here with their fragment, which a
//! 307 keeps: `bench-shell.js` maps them to the new addresses (`BenchCore.mapHash`).

use super::super::{AppState, DevicesSnapshot, devices_snapshot, render};
use super::{LIVE_WARNING, cameras, configured_device, loaded_devices};
use crate::benchmark::images::{self, ImageSet};
use crate::benchmark::service::default_models;
use crate::benchmark::{BenchmarkResults, results_path};
use crate::config::Config;
use askama::Template;
use axum::extract::{RawQuery, State};
use axum::response::{IntoResponse, Redirect, Response};
use std::collections::HashMap;
use std::sync::Arc;

/// The shell of every Benchmark page.
pub(super) struct Shell {
    /// `run`, `compare`, `tune`, `history` or `images`.
    pub(super) step: &'static str,
    /// A results file exists (Compare and Tune have something to show).
    pub(super) has_results: bool,
    /// Results exist and this is not the Run step: the stepper's tick on Run.
    pub(super) run_done: bool,
    /// The saved goal, as Compare names it ("F2 · person, car · budget 1 s").
    pub(super) goal_summary: String,
    pub(super) goal_optimize: String,
    pub(super) goal_tune: String,
    /// Comma separated; empty = every class.
    pub(super) goal_classes: String,
    /// Milliseconds, empty without a budget.
    pub(super) goal_budget: String,
    /// `benchmark.weights`' accuracy share (the grade's weight).
    pub(super) accuracy_weight: String,
    /// The Goal dialog's class choices before a Compare view names them: `collect.classes`
    /// with the goal's own.
    pub(super) class_choices: String,
    pub(super) force_cpu: bool,
    /// A run that never ended, shown once ([`crate::benchmark::service::breadcrumb`]).
    pub(super) interrupted: Option<String>,
}

impl Shell {
    fn of(state: &AppState, cfg: &Config, step: &'static str) -> Self {
        let g = &cfg.benchmark.goal;
        let has_results = super::file_stamp(&results_path(&state.config_path)).is_some();
        let mut choices: Vec<String> = cfg.collect.classes.clone();
        for c in &g.classes {
            if !choices.iter().any(|x| x.eq_ignore_ascii_case(c)) {
                choices.push(c.clone());
            }
        }
        Self {
            step,
            has_results,
            run_done: has_results && step != "run",
            goal_summary: g.describe(),
            goal_optimize: g.optimize_for.to_string(),
            goal_tune: g.tune_for.to_string(),
            goal_classes: g.classes.join(","),
            goal_budget: g
                .speed_budget_ms
                .map(|b| format!("{b}"))
                .unwrap_or_default(),
            accuracy_weight: format!("{:.2}", cfg.benchmark.weights.accuracy_share()),
            class_choices: choices.join(","),
            force_cpu: cfg.force_cpu,
            interrupted: state.benchmark.take_interrupted(&cfg.data_root()),
        }
    }
}

/// `GET /benchmark`: the step that has something to show (see the module docs).
pub(in crate::server) async fn index(
    State(state): State<Arc<AppState>>,
    RawQuery(query): RawQuery,
) -> Response {
    let has = super::file_stamp(&results_path(&state.config_path)).is_some();
    let to = if has {
        "/benchmark/compare"
    } else {
        "/benchmark/run"
    };
    match query.filter(|q| !q.is_empty()) {
        Some(q) => Redirect::temporary(&format!("{to}?{q}")).into_response(),
        None => Redirect::temporary(to).into_response(),
    }
}

#[derive(Template)]
#[template(path = "benchmark/compare.html")]
struct ComparePage {
    nav: &'static str,
    version: &'static str,
    shell: Shell,
}

#[derive(Template)]
#[template(path = "benchmark/tune.html")]
struct TunePage {
    nav: &'static str,
    version: &'static str,
    shell: Shell,
    blue_iris_note: &'static str,
}

#[derive(Template)]
#[template(path = "benchmark/history.html")]
struct HistoryPage {
    nav: &'static str,
    version: &'static str,
    shell: Shell,
}

#[derive(Template)]
#[template(path = "benchmark/images.html")]
struct ImagesPage {
    nav: &'static str,
    version: &'static str,
    shell: Shell,
}

/// `GET /benchmark/compare`.
pub(in crate::server) async fn compare_page(State(state): State<Arc<AppState>>) -> Response {
    super::super::blocking(move || {
        let cfg = state.config_read().clone();
        render(&ComparePage {
            nav: "benchmark",
            version: crate::VERSION,
            shell: Shell::of(&state, &cfg, "compare"),
        })
    })
    .await
}

/// `GET /benchmark/tune`.
pub(in crate::server) async fn tune_page(State(state): State<Arc<AppState>>) -> Response {
    super::super::blocking(move || {
        let cfg = state.config_read().clone();
        render(&TunePage {
            nav: "benchmark",
            version: crate::VERSION,
            shell: Shell::of(&state, &cfg, "tune"),
            blue_iris_note: super::BLUE_IRIS_THRESHOLD_NOTE,
        })
    })
    .await
}

/// `GET /benchmark/history`.
pub(in crate::server) async fn history_page(State(state): State<Arc<AppState>>) -> Response {
    super::super::blocking(move || {
        let cfg = state.config_read().clone();
        render(&HistoryPage {
            nav: "benchmark",
            version: crate::VERSION,
            shell: Shell::of(&state, &cfg, "history"),
        })
    })
    .await
}

/// `GET /benchmark/images`.
pub(in crate::server) async fn images_page(State(state): State<Arc<AppState>>) -> Response {
    super::super::blocking(move || {
        let cfg = state.config_read().clone();
        render(&ImagesPage {
            nav: "benchmark",
            version: crate::VERSION,
            shell: Shell::of(&state, &cfg, "images"),
        })
    })
    .await
}

// ---- the Run form ------------------------------------------------------------------------------

pub(super) struct ModelOption {
    pub(super) name: String,
    pub(super) family: String,
    pub(super) enabled: bool,
    pub(super) exists: bool,
    pub(super) checked: bool,
    pub(super) is_default: bool,
    /// `{"<device>": <ms per image>}` of its last run per device (the run-time estimate).
    pub(super) timing: String,
}

pub(super) struct DeviceOptionRow {
    pub(super) spec: String,
    pub(super) label: String,
    pub(super) runnable: bool,
    pub(super) checked: bool,
    /// Counts toward the "several GPUs" confirmation.
    pub(super) gpu: bool,
    /// A CPU device: the reference other devices are checked against.
    pub(super) cpu: bool,
    /// A model runs on it now.
    pub(super) in_use: bool,
    pub(super) reason: Option<String>,
}

/// One dataset row of the form.
pub(super) struct DatasetRow {
    /// Value posted as `dataset` (`coco-cctv`, `sample`, `dir:/path`).
    pub(super) value: String,
    pub(super) title: String,
    /// What it tests, in a few words (a folder: its path).
    pub(super) tests: String,
    /// "120 images · 45 MB".
    pub(super) size: String,
    /// Images it holds now (0 = unknown; the run-time estimate).
    pub(super) images: usize,
    pub(super) state_text: String,
    /// The state chip's class: `ok`, `same`, `warn`, `bad`.
    pub(super) chip: &'static str,
    pub(super) checked: bool,
    /// Resource id to download (built-in sets not fully installed).
    pub(super) download_id: Option<String>,
    /// A user folder (removable from the defaults).
    pub(super) folder: bool,
    pub(super) selectable: bool,
}

#[derive(Template)]
#[template(path = "benchmark/run.html")]
struct RunPage {
    nav: &'static str,
    version: &'static str,
    shell: Shell,
    models: Vec<ModelOption>,
    devices: Vec<DeviceOptionRow>,
    datasets: Vec<DatasetRow>,
    reference_models: Vec<(String, bool)>,
    repeat: usize,
    warmup: usize,
    low_priority: bool,
    served: bool,
    max_images: usize,
    warning: &'static str,
    gpu_warning: &'static str,
}

/// Per normalized model name: its last run's p50 per device (ms), from the run history (small;
/// seeded from the results only when there is no history file yet).
fn timings(state: &AppState) -> HashMap<String, Vec<(String, f64)>> {
    let path = results_path(&state.config_path);
    let hist = crate::benchmark::history::history_path(&path);
    let history = if hist.is_file() {
        crate::benchmark::history::History::load_or_seed(&hist, None)
    } else {
        crate::benchmark::history::History::load_or_seed(
            &hist,
            BenchmarkResults::load_or_warn(&path).as_ref(),
        )
    };
    let mut out = HashMap::new();
    for (name, entries) in &history.models {
        let mut per: Vec<(String, f64)> = Vec::new();
        for e in entries.iter().rev() {
            for d in &e.devices {
                if let Some(ms) = d.total_p50_ms.filter(|v| *v > 0.0)
                    && !per
                        .iter()
                        .any(|(k, _)| crate::config::same_device(k, &d.device))
                {
                    per.push((d.device.clone(), ms));
                }
            }
        }
        out.insert(crate::registry::normalize_name(name), per);
    }
    out
}

pub(super) fn model_options(
    c: &Config,
    timing: &HashMap<String, Vec<(String, f64)>>,
) -> Vec<ModelOption> {
    let defaults = if c.benchmark.models.is_empty() {
        default_models(c)
    } else {
        c.benchmark.models.clone()
    };
    let key = crate::registry::normalize_name;
    let default_model = crate::training::switch::current_default(c).map(|n| key(&n));
    c.models
        .iter()
        .map(|m| {
            let exists = c.data_path(&m.path).is_file();
            let name = m.effective_name();
            let t: serde_json::Map<String, serde_json::Value> = timing
                .get(&key(&name))
                .map(|v| {
                    v.iter()
                        .map(|(d, ms)| (d.clone(), serde_json::json!((ms * 10.0).round() / 10.0)))
                        .collect()
                })
                .unwrap_or_default();
            ModelOption {
                checked: exists && defaults.iter().any(|d| key(d) == key(&name)),
                is_default: default_model.as_deref() == Some(key(&name).as_str()),
                timing: serde_json::Value::Object(t).to_string(),
                name,
                family: m.family.to_string(),
                enabled: m.enabled,
                exists,
            }
        })
        .collect()
}

/// The devices the enabled models run on now (loaded), else would run on, in config order.
fn devices_in_use(state: &AppState, snap: &DevicesSnapshot, c: &Config) -> Vec<String> {
    let loaded = loaded_devices(state);
    c.enabled_models()
        .filter_map(|m| {
            let name = crate::registry::normalize_name(&m.effective_name());
            loaded
                .get(&name)
                .cloned()
                .or_else(|| configured_device(c, m, &snap.model_selection(state, c, m)))
        })
        .collect()
}

/// Every device option, runnable first (downloadable ones count as not runnable here: the
/// benchmark only measures what is installed). The saved devices are ticked, else `in_use`, the
/// devices the models run on, else the first runnable option: a sweep of every device has hung
/// a machine, so that takes ticking them by hand. Any of those that cannot run here is left out
/// (a saved `openvino:gpu.0` whose iGPU was switched off must not leave the page with nothing
/// ticked).
pub(super) fn device_rows(
    snap: &DevicesSnapshot,
    c: &Config,
    in_use: &[String],
) -> Vec<DeviceOptionRow> {
    let (ok, bad): (Vec<_>, Vec<_>) = snap.selection.options.iter().partition(|o| o.runnable);
    let runnable = |list: &[String]| -> Vec<String> {
        list.iter()
            .filter_map(|d| crate::backend::spec::parse(d).ok().map(|s| s.to_string()))
            .filter(|d| ok.iter().any(|o| o.spec.to_string() == *d))
            .collect()
    };
    let used = runnable(in_use);
    let chosen = [runnable(&c.benchmark.devices), used.clone()]
        .into_iter()
        .find(|l| !l.is_empty())
        .or_else(|| ok.first().map(|o| vec![o.spec.to_string()]))
        .unwrap_or_default();
    ok.iter()
        .copied()
        .chain(bad)
        .map(|o| {
            let spec = o.spec.to_string();
            DeviceOptionRow {
                checked: o.runnable && chosen.contains(&spec),
                gpu: o.spec.is_gpu_like(),
                cpu: o.spec.is_cpu(),
                in_use: used.contains(&spec),
                label: o.label.clone(),
                runnable: o.runnable,
                reason: (!o.runnable).then(|| match &o.download {
                    Some(d) => format!("not installed ({})", d.summary),
                    None => o.reason.clone().unwrap_or_else(|| "not runnable".into()),
                }),
                spec,
            }
        })
        .collect()
}

/// What a built-in set tests, in a few words.
fn tests_of(m: &images::Manifest) -> String {
    let known = match m.id.as_str() {
        "coco-cctv" => "Street scenes by day: people, vehicles, animals",
        "bmd45-cctv" => "Real daytime CCTV traffic: vehicles only",
        "exdark-night" => "Low light and twilight: people, vehicles, animals",
        "exdark-ir" => "A grayscale stand-in for night IR",
        "nir-night" => "Real night IR camera traps: mostly false alerts",
        _ => "",
    };
    if !known.is_empty() {
        return known.to_string();
    }
    // A set added later: the first sentence of its description.
    let first = m.description.split(". ").next().unwrap_or_default();
    if m.has_ground_truth() {
        first.to_string()
    } else {
        format!("{first}; no labels")
    }
}

pub(super) fn dataset_rows(state: &AppState, c: &Config) -> Vec<DatasetRow> {
    let selected: Vec<String> = c.benchmark.datasets.iter().map(|d| d.label()).collect();
    let is_sel = |id: &str| selected.iter().any(|s| s.eq_ignore_ascii_case(id));
    let root = c.data_root();
    let rows_by_id: HashMap<String, crate::resources::status::ResourceRow> =
        crate::resources::status::rows(state.resources.as_ref(), c)
            .into_iter()
            .filter(|r| r.kind == "bench-images")
            .map(|r| (r.id.to_string(), r))
            .collect();
    // Your cameras first: the headline set.
    let mut out = vec![cameras::dataset_row(
        c,
        is_sel(crate::benchmark::cameras::ID),
    )];
    for m in images::builtin_manifests() {
        let (set, missing) = ImageSet::from_manifest(m, &images::builtin_dir(&root, &m.id));
        let rid = format!("{}{}", crate::resources::catalog::BENCH_ID_PREFIX, m.id);
        let busy = rows_by_id.get(&rid).filter(|r| r.busy);
        let (chip, text) = match busy {
            Some(r) => ("warn", r.state_text.clone()),
            None if missing == 0 => ("same", "downloaded".to_string()),
            None if !set.images.is_empty() => (
                "warn",
                format!("{} of {} images", set.images.len(), m.images.len()),
            ),
            None => ("same", String::new()),
        };
        out.push(DatasetRow {
            value: m.id.clone(),
            title: m.title.clone(),
            tests: tests_of(m),
            size: format!(
                "{} images \u{b7} {}",
                m.images.len(),
                crate::resources::catalog::format_size(m.size())
            ),
            images: m.images.len(),
            chip,
            state_text: text,
            checked: is_sel(&m.id),
            download_id: (missing > 0 && busy.is_none()).then_some(rid),
            folder: false,
            selectable: true,
        });
    }
    for id in images::KNOWN_SET_IDS {
        if images::builtin_manifest(id).is_none() {
            out.push(DatasetRow {
                value: id.to_string(),
                title: id.to_string(),
                tests: "curated set; not included in this build yet".into(),
                size: String::new(),
                images: 0,
                chip: "same",
                state_text: "not in this build".into(),
                checked: is_sel(id),
                download_id: None,
                folder: false,
                selectable: true,
            });
        }
    }
    out.push(DatasetRow {
        value: images::SAMPLE_ID.into(),
        title: "Embedded sample".into(),
        tests: "Checks that a model runs, offline; no labels".into(),
        size: "1 image".into(),
        images: 1,
        chip: "same",
        state_text: "built in".into(),
        checked: is_sel(images::SAMPLE_ID),
        download_id: None,
        folder: false,
        selectable: true,
    });
    for d in &c.benchmark.datasets {
        if let Some((dir, _, _)) = d.dir() {
            let abs = c.data_path(&dir);
            let ok = abs.is_dir();
            out.push(DatasetRow {
                value: d.label(),
                title: d.label(),
                tests: abs.display().to_string(),
                size: String::new(),
                images: 0,
                chip: if ok { "same" } else { "bad" },
                state_text: if ok {
                    "folder on the server".into()
                } else {
                    "folder not found".into()
                },
                checked: true,
                download_id: None,
                folder: true,
                selectable: ok,
            });
        }
    }
    out
}

/// `GET /benchmark/run`.
pub(in crate::server) async fn run_page(State(state): State<Arc<AppState>>) -> Response {
    super::super::blocking(move || {
        let snap = devices_snapshot(&state);
        let cfg = state.config_read().clone();
        let b = &cfg.benchmark;
        let timing = timings(&state);
        render(&RunPage {
            nav: "benchmark",
            version: crate::VERSION,
            shell: Shell::of(&state, &cfg, "run"),
            models: model_options(&cfg, &timing),
            devices: device_rows(&snap, &cfg, &devices_in_use(&state, &snap, &cfg)),
            datasets: dataset_rows(&state, &cfg),
            reference_models: cfg
                .models
                .iter()
                .filter(|m| cfg.data_path(&m.path).is_file())
                .map(|m| {
                    let n = m.effective_name();
                    let sel = b.reference_model.as_deref() == Some(n.as_str());
                    (n, sel)
                })
                .collect(),
            repeat: b.repeat_per_image.max(1),
            warmup: b.warmup,
            low_priority: b.low_priority,
            served: b.served,
            max_images: b.max_images_per_dataset,
            warning: LIVE_WARNING,
            gpu_warning: crate::benchmark::service::SEVERAL_GPUS_WARNING,
        })
    })
    .await
}
