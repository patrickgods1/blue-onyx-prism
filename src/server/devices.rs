//! The device snapshot (hardware + runtimes) and `GET /v1/devices`.

use super::{AppState, blocking, live};
use crate::backend::detect::HardwareInfo;
use crate::backend::select::{OpenVinoProbe, OrtProbe, RuntimeProbe, Selection, select};
use crate::backend::spec::{self, DeviceSpec};
use crate::config::{Config, ModelConfig};
use crate::setup_onnxruntime::InstalledOrt;
use axum::Json;
use axum::extract::State;
use std::sync::Arc;

/// "Force CPU is on — overrides device auto (would use ort:coreml)" when `force_cpu` is set.
pub(super) fn force_cpu_note(cfg: &Config, snap: &DevicesSnapshot) -> Option<String> {
    if !cfg.force_cpu {
        return None;
    }
    let would = match spec::parse(&cfg.device) {
        Ok(DeviceSpec::Auto) => snap
            .selection
            .auto_pick()
            .map_or_else(|| "nothing runnable".to_string(), |o| o.spec.to_string()),
        Ok(d) => d.with_default_index(cfg.gpu_index).to_string(),
        Err(_) => cfg.device.trim().to_string(),
    };
    Some(format!(
        "Force CPU is on \u{2014} overrides device {} (would use {would}). Uncheck Force CPU on \
         the Config page to use it.",
        cfg.device.trim()
    ))
}

/// Hardware, runtimes and device options as seen by this process.
pub(super) struct DevicesSnapshot {
    pub(super) hardware: HardwareInfo,
    /// Global options and `auto` ranking (any model, with an `.onnx`).
    pub(super) selection: Selection,
    /// What the options were built from (per-model selections, [`Self::model_selection`]).
    pub(super) probe: RuntimeProbe,
    pub(super) openvino_version: String,
    pub(super) openvino_devices: Vec<String>,
    pub(super) onnxruntime: Option<InstalledOrt>,
}

impl DevicesSnapshot {
    /// The selection model `m` loads with: see [`model_selection`].
    pub(super) fn model_selection(
        &self,
        state: &AppState,
        cfg: &Config,
        m: &ModelConfig,
    ) -> Selection {
        model_selection(state, &self.hardware, &self.probe, cfg, m)
    }

    /// Display text of the device `auto` resolves to.
    pub(super) fn auto_pick_label(&self) -> String {
        match self.selection.auto_pick() {
            Some(o) => format!("{} ({})", o.spec, o.label),
            None => "nothing runnable".to_string(),
        }
    }

    pub(super) fn onnxruntime_label(&self) -> String {
        match &self.onnxruntime {
            Some(o) => format!("{} ({})", o.version, o.flavor),
            None => "not installed (run setup-onnxruntime)".to_string(),
        }
    }
}

/// The selection model `m` loads with (after a restart, as the registry builds it): its own
/// `.onnx` check, and `auto` ordered by the model's benchmark speed order from the current
/// `benchmark.json` (cached by file stamp, see `live::bench_summary`).
pub(super) fn model_selection(
    state: &AppState,
    hw: &HardwareInfo,
    probe: &RuntimeProbe,
    cfg: &Config,
    m: &ModelConfig,
) -> Selection {
    let path = cfg.data_path(&m.path);
    let mut sel = select(
        hw,
        probe,
        crate::backend::select::onnx_path_for(&path).is_some(),
    );
    let bench = live::bench_summary(&crate::benchmark::results_path(&state.config_path));
    let key = crate::registry::normalize_name(&m.effective_name());
    if let Some(order) = (*bench)
        .as_ref()
        .and_then(|b| b.models.get(&key))
        .and_then(|f| f.auto_order.as_deref())
    {
        sel.apply_measured_order(order);
    }
    sel
}

/// Snapshot of the process's devices. Uses the shared runtimes when they are free (workers hold
/// the lock while compiling a model; waiting would stall the page), else rebuilds the options from
/// the OpenVINO facts the registry captured at startup.
pub(super) fn devices_snapshot(state: &AppState) -> DevicesSnapshot {
    let info = &state.registry.runtime_info;
    let live = state.registry.runtimes().and_then(|rt| {
        let rt = rt.try_lock().ok()?;
        Some((
            rt.hardware().clone(),
            rt.selection(None),
            rt.probe().clone(),
        ))
    });
    let live_probe = live.as_ref().map(|(_, _, p)| p.clone());
    let (hardware, selection, probe) = live.unwrap_or_else(|| {
        let hardware = crate::backend::detect::hardware().clone();
        let probe = if info.openvino_version.is_empty() && info.available_devices.is_empty() {
            RuntimeProbe {
                openvino: OpenVinoProbe::unavailable(crate::backend::OV_UNAVAILABLE),
                ort: OrtProbe::not_in_build(),
            }
        } else {
            RuntimeProbe::openvino_only(&info.available_devices)
        };
        let selection = select(&hardware, &probe, true);
        (hardware, selection, probe)
    });
    // Options a download would make runnable ("downloadable").
    let mut selection = selection;
    {
        let cfg = state.config_read();
        let root = crate::resources::download_root(&cfg);
        let installed = crate::resources::detect_installed(&cfg, &root, live_probe);
        crate::resources::resolve::annotate_selection(
            &mut selection,
            &cfg,
            &hardware,
            &installed,
            true,
        );
    }
    // The install the ONNX Runtime loader picks (explicit dir, ORT_DYLIB_PATH, active flavor,
    // legacy flat dir, ...), searched under `download_dir` like the loader does.
    let ort_opts = state.config_read().ort_options();
    let onnxruntime = ort_opts
        .lookup()
        .library
        .and_then(|lib| lib.parent().map(std::path::Path::to_path_buf))
        .and_then(|dir| crate::setup_onnxruntime::installed(&dir));
    DevicesSnapshot {
        hardware,
        selection,
        probe,
        openvino_version: info.openvino_version.clone(),
        openvino_devices: info.available_devices.clone(),
        onnxruntime,
    }
}

/// The hardware profile of the running generation and its decode + preprocess threads, else
/// (registries without runtimes) the profile the current config gives.
pub(super) fn profile_view(state: &AppState) -> (crate::profile::Profile, usize) {
    if let Some((p, n)) = state.registry.profile() {
        return (p.clone(), n);
    }
    let cfg = state.config_read();
    let p = cfg.profile();
    let n = crate::registry::preprocess_threads(&cfg, &p, &[]);
    (p, n)
}

/// `GET /v1/devices`: detected hardware, runtimes and every device option with `auto`'s pick.
pub(super) async fn devices_json(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    blocking(move || {
        let snap = devices_snapshot(&state);
        let (profile, threads) = profile_view(&state);
        Json(serde_json::json!({
            "success": true,
            "hardware": snap.hardware,
            "runtimes": {
                "openvino": {
                    "version": snap.openvino_version,
                    "devices": snap.openvino_devices,
                },
                "onnxruntime": snap.onnxruntime.as_ref().map(|o| serde_json::json!({
                    "version": o.version,
                    "flavor": o.flavor,
                })),
            },
            "selection": snap.selection,
            "autoPick": snap.selection.auto_pick().map(|o| o.spec.to_string()),
            "profile": profile.to_json(threads),
        }))
    })
    .await
}

#[cfg(test)]
mod tests {

    use crate::server::test_util::*;

    use axum::http::{StatusCode, header};

    use axum::body::Body;
    use axum::http::Request;

    #[tokio::test]
    async fn devices_endpoint_and_pages() {
        let state = test_state();
        let (s, h, b) = call(
            &state,
            Request::get("/v1/devices").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            h[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("application/json")
        );
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["success"], true);
        assert!(v["hardware"]["gpus"].is_array());
        assert_eq!(v["runtimes"]["openvino"]["version"], "2026.0.0-test");
        assert_eq!(v["runtimes"]["openvino"]["devices"][0], "CPU");
        let opts = v["selection"]["options"].as_array().unwrap();
        let cpu = opts
            .iter()
            .find(|o| o["spec"] == "openvino:cpu")
            .expect("cpu option");
        assert_eq!(cpu["runnable"], true);
        assert!(opts.iter().any(|o| o["spec"] == "ort:cpu"));
        assert!(v["selection"]["auto"].is_array());
        assert_eq!(v["autoPick"], v["selection"]["auto"][0]);

        let (_, cfg) = get_text(&state, "/config").await;
        assert!(
            cfg.contains("<select name=\"device\" id=\"device\">"),
            "{cfg}"
        );
        assert!(cfg.contains("<option value=\"auto\""));
        assert!(cfg.contains("currently:"));
        let (_, home) = get_text(&state, "/").await;
        assert!(
            home.contains("ONNX Runtime") && home.contains("GPUs") && home.contains("Auto picks")
        );
    }

    #[tokio::test]
    async fn devices_report_the_hardware_profile() {
        let json = |b: &[u8]| serde_json::from_slice::<serde_json::Value>(b).unwrap();
        let state = test_state();
        let req = || Request::get("/v1/devices").body(Body::empty()).unwrap();
        let (s, _, b) = call(&state, req()).await;
        assert_eq!(s, StatusCode::OK);
        let p = &json(&b)["profile"];
        assert!(
            ["low", "mid", "high"].contains(&p["class"].as_str().unwrap()),
            "{p}"
        );
        assert_eq!(p["forced"], false);
        assert!(p["cores"].as_u64().unwrap() >= 1);
        assert!(p["preprocessThreads"].as_u64().unwrap() >= 1);
        assert!(p["inflight"].as_u64().unwrap() >= 1);
        assert!(p["gpuClass"].is_string());
        assert!(!p["recommendedSizes"].as_array().unwrap().is_empty());
        assert!(!p["reasons"].as_array().unwrap().is_empty());
        let (_, home) = get_text(&state, "/").await;
        assert!(home.contains("Hardware profile"), "{home}");

        let state = test_state_with(|c| {
            c.hardware_profile = crate::profile::HardwareProfile::High;
            c.preprocess_threads = 5;
        });
        let (_, _, b) = call(&state, req()).await;
        let p = &json(&b)["profile"];
        assert_eq!(p["class"], "high");
        assert_eq!(p["forced"], true);
        assert_eq!(p["preprocessThreads"], 5);
        assert_eq!(p["recommendedSizes"], serde_json::json!(["m", "l", "x"]));
        let (_, home) = get_text(&state, "/").await;
        assert!(home.contains("high-spec"), "{home}");
    }

    #[tokio::test]
    async fn force_cpu_note_on_home_and_config() {
        let state = test_state();
        state
            .config
            .update("test", |c| {
                c.force_cpu = true;
                c.device = "auto".into();
                Ok(())
            })
            .unwrap();
        let (_, home) = get_text(&state, "/").await;
        assert!(
            home.contains("Force CPU is on \u{2014} overrides device auto (would use "),
            "{home}"
        );
        let (_, cfg) = get_text(&state, "/config").await;
        assert!(cfg.contains("<select name=\"device\" id=\"device\" disabled>"));
        assert!(cfg.contains("Overridden by Force CPU: Force CPU is on"));
    }
}
