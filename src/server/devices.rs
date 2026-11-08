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
            Some(o) => format!("{} \u{b7} {}", o.spec, o.label),
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

/// Hardware and runtime probe of this generation: the shared runtimes' own when they are free,
/// else (a worker holds the lock while compiling; waiting would stall the page) the same facts
/// captured when the registry started ([`crate::registry::StartupDevices`]). None only for
/// registries without runtimes.
pub(super) fn runtime_facts(state: &AppState) -> Option<(HardwareInfo, RuntimeProbe)> {
    let reg = &state.registry;
    reg.runtimes()
        .and_then(|rt| {
            let rt = rt.try_lock().ok()?;
            Some((rt.hardware().clone(), rt.probe().clone()))
        })
        .or_else(|| {
            reg.startup_devices()
                .map(|s| (s.hardware.clone(), s.probe.clone()))
        })
}

/// Snapshot of the process's devices, from [`runtime_facts`], else (registries without
/// runtimes) rebuilt from the OpenVINO facts the registry captured at startup. The global
/// selection is `Runtimes::selection(None)`'s: any model, an `.onnx` assumed, no measured order
/// (per-model orders come from [`DevicesSnapshot::model_selection`]).
pub(super) fn devices_snapshot(state: &AppState) -> DevicesSnapshot {
    let info = &state.registry.runtime_info;
    let facts = runtime_facts(state);
    let live_probe = facts.as_ref().map(|(_, p)| p.clone());
    let (hardware, probe) = facts.unwrap_or_else(|| {
        let hardware = crate::backend::detect::hardware().clone();
        let probe = if info.openvino_version.is_empty() && info.available_devices.is_empty() {
            RuntimeProbe {
                openvino: OpenVinoProbe::unavailable(crate::backend::OV_UNAVAILABLE),
                ort: OrtProbe::not_in_build(),
            }
        } else {
            RuntimeProbe::openvino_only(&info.available_devices)
        };
        (hardware, probe)
    });
    let mut selection = select(&hardware, &probe, true);
    // Options a download would make runnable ("downloadable").
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

/// `gpuMemory` of `GET /v1/devices`: every adapter with a live reading
/// ([`crate::backend::vram::VramReading`], `index` = `hardware.gpus[].index`) and the loaded
/// models holding memory on it (`models`: `{model, device, adapter, heldMb}`). Empty where no
/// adapter can be read (macOS: unified memory).
pub(super) fn gpu_memory_json(
    readings: &[crate::backend::vram::VramReading],
    holders: &[crate::backend::vram::Holder],
) -> serde_json::Value {
    readings
        .iter()
        .map(|r| {
            let mut v = serde_json::to_value(r).unwrap_or_default();
            if let Some(o) = v.as_object_mut() {
                o.insert("usedMb".into(), r.used_mb().into());
                o.insert("text".into(), r.text().into());
                o.insert(
                    "models".into(),
                    serde_json::to_value(
                        holders
                            .iter()
                            .filter(|h| h.adapter == r.index)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap_or_default(),
                );
            }
            v
        })
        .collect()
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
            "gpuMemory": gpu_memory_json(
                &crate::backend::vram::readings(),
                &crate::backend::vram::holders(),
            ),
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
        assert!(v["gpuMemory"].is_array());
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
            home.contains("ONNX Runtime") && home.contains("GPUs") && home.contains("Auto device")
        );
    }

    #[test]
    fn gpu_memory_lists_readings_with_their_models() {
        use crate::backend::vram::{Holder, MemoryKind, Source, VramReading};
        let r = VramReading {
            index: 1,
            name: "NVIDIA GeForce GTX 1070 Ti".into(),
            vendor: crate::backend::detect::GpuVendor::Nvidia,
            kind: MemoryKind::Dedicated,
            total_mb: 8192,
            free_mb: 6144,
            source: Source::Nvml,
        };
        let h = |adapter| Holder {
            model: "IPcam-general".into(),
            device: "ort:cuda:0".into(),
            adapter,
            held_mb: Some(410),
        };
        let v = super::gpu_memory_json(&[r], &[h(1), h(0)]);
        assert_eq!(v.as_array().unwrap().len(), 1);
        let g = &v[0];
        assert_eq!(g["index"], 1);
        assert_eq!(g["totalMb"], 8192);
        assert_eq!(g["freeMb"], 6144);
        assert_eq!(g["usedMb"], 2048);
        assert_eq!(g["kind"], "dedicated");
        assert_eq!(g["text"], "6.0 GB free of 8.0 GB");
        assert_eq!(g["models"].as_array().unwrap().len(), 1, "{g}");
        assert_eq!(g["models"][0]["heldMb"], 410);
        assert_eq!(super::gpu_memory_json(&[], &[]), serde_json::json!([]));
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

    /// Apple silicon with the CoreML flavor loaded, as a generation records it at startup.
    fn coreml_startup() -> crate::registry::StartupDevices {
        use crate::backend::select::{EpStatus, OpenVinoProbe, OrtProbe, RuntimeProbe};
        use crate::backend::spec::Target;
        crate::registry::StartupDevices {
            hardware: crate::backend::HardwareInfo::new("macos", "aarch64", Vec::new()),
            probe: RuntimeProbe {
                openvino: OpenVinoProbe::with_device_names(&["CPU".to_string()]),
                ort: OrtProbe::installed(
                    "coreml",
                    vec![
                        EpStatus::usable(Target::CoreMl),
                        EpStatus::usable(Target::Cpu),
                    ],
                )
                .with_version("1.24.4"),
            },
            onnxruntime_version: Some("1.24.4".into()),
        }
    }

    /// While a worker compiles it holds the runtimes lock: `/v1/devices`, Home and Config answer
    /// from the probe the generation recorded at startup (ONNX Runtime options runnable, `auto`
    /// on CoreML), never the OpenVINO-only guess that called every `ort:*` option unavailable.
    #[tokio::test]
    async fn devices_answer_from_the_startup_probe_while_a_worker_compiles() {
        use std::sync::{Arc, mpsc};
        let dir = std::env::temp_dir().join(format!("bo_srv_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let startup = coreml_startup();
        // The live runtimes (a fake) differ from the startup facts, so each answer shows which
        // one it came from.
        let rt = crate::backend::Runtimes::fake(crate::backend::fake::FakeScript::new(1, (64, 64)));
        let registry = crate::registry::ModelRegistry::idle_for_test(rt, startup.clone());
        let config = crate::config::Config {
            download_dir: Some(dir.clone()),
            ..Default::default()
        };
        let state = Arc::new(crate::server::AppState::new(
            Arc::new(registry),
            Arc::new(crate::metrics::Metrics::new("0.0.0")),
            config,
            dir.join("cfg.json"),
        ));

        // A "compiling worker": holds the runtimes lock until told to stop.
        let runtimes = state.registry.runtimes().unwrap().clone();
        let (locked_tx, locked_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let holder = std::thread::spawn(move || {
            let _held = runtimes.lock().unwrap();
            locked_tx.send(()).unwrap();
            let _ = release_rx.recv();
        });
        locked_rx.recv().unwrap();
        assert!(state.registry.runtimes().unwrap().try_lock().is_err());

        let snap = super::devices_snapshot(&state);
        assert_eq!(snap.probe, startup.probe);
        assert_eq!(snap.hardware, startup.hardware);
        let coreml = snap
            .selection
            .options
            .iter()
            .find(|o| o.spec.to_string() == "ort:coreml")
            .expect("ort:coreml option");
        assert!(coreml.runnable, "{coreml:?}");
        assert_eq!(
            snap.selection.auto_pick().map(|o| o.spec.to_string()),
            Some("ort:coreml".to_string())
        );
        assert!(
            snap.selection
                .options
                .iter()
                .filter(|o| o.spec.to_string().starts_with("ort:"))
                .all(|o| o.runnable),
            "{:?}",
            snap.selection.options
        );
        assert!(snap.auto_pick_label().starts_with("ort:coreml"));
        assert_eq!(
            super::super::pages::ort_version(&state.registry).as_deref(),
            Some("1.24.4")
        );

        let (s, _, b) = call(
            &state,
            Request::get("/v1/devices").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["autoPick"], "ort:coreml", "{v}");
        assert_eq!(v["hardware"]["os"], "macos");
        let (_, home) = get_text(&state, "/").await;
        assert!(home.contains("ort:coreml"), "{home}");
        let (_, cfg) = get_text(&state, "/config").await;
        assert!(cfg.contains("Auto \u{2014} currently: ort:coreml"), "{cfg}");
        assert!(!cfg.contains("ort:coreml \u{2014} not available"), "{cfg}");

        // Free again: the live runtimes answer (the fake has no ONNX Runtime).
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        let snap = super::devices_snapshot(&state);
        assert_eq!(snap.probe, crate::backend::select::RuntimeProbe::default());
        assert_eq!(super::super::pages::ort_version(&state.registry), None);
        let _ = std::fs::remove_dir_all(&dir);
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
