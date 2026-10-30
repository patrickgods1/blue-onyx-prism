//! The benchmark at the end of a run: the trained model on its held-out frames (a `dir:` set)
//! and optional built-in sets, through the regular [`BenchmarkService`] (so the results land in
//! `benchmark.json` and its history like any other run), summarized for the Training page.
//! It runs on the chosen devices one after the other (`training.benchmark_devices`, the Train
//! form), else on the one device the model would load on, never on every device of the machine
//! (a GTX 1070 Ti + UHD 630 box hard-hung within seconds of that sweep starting, twice).
//!
//! Per-scene AP comes from the benchmark's per-tag breakdowns of the held-out manifest's own tags
//! ([`dataset::heldout_tags`]: `heldout:day|night|ir|all|in-scope`), which are exact AP over
//! those images and never mix in the built-in sets run beside it (their bare `day` / `night` tags
//! would). Held-out sets of older versions only carry the bare scene tags: they are read when the
//! held-out set was the only dataset of the run, and give no `in_scope`.

use super::dataset;
use super::job::{BenchSummary, SceneAp};
use crate::backend::DeviceSpec;
use crate::benchmark::Breakdown;
use crate::benchmark::report::ModelResult;
use crate::benchmark::service::{BenchmarkService, RunContext, RunRequest, RunState};
use crate::config::DatasetRef;
use crate::resources::export::Cancelled;
use crate::sync::LockExt;
use anyhow::{Result, anyhow, bail};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Longest wait for another benchmark to finish before this one starts.
pub const BUSY_WAIT: Duration = Duration::from_secs(60 * 60);

/// The datasets of the run: the held-out set, then `extra` built-in ids.
pub fn datasets(name: &str, heldout: Option<&Path>, extra: &[String]) -> Vec<DatasetRef> {
    let mut v = Vec::new();
    if let Some(h) = heldout {
        v.push(DatasetRef::Dir {
            dir: h.to_path_buf(),
            gt: None,
            name: Some(format!("heldout-{name}")),
        });
    }
    v.extend(extra.iter().map(|e| DatasetRef::Id(e.clone())));
    v
}

/// The devices of the run: `chosen` when given, else the one device the model `name` would load
/// on (its configured device, `auto` resolved to its first pick on this machine).
fn devices_for(c: &mut RunContext, name: &str, chosen: &[String]) -> Result<Vec<String>> {
    if !chosen.is_empty() {
        return Ok(chosen.to_vec());
    }
    let m = crate::benchmark::find_model(&c.config, name)
        .ok_or_else(|| anyhow!("'{name}' is not in the config"))?;
    if let DeviceSpec::Device(d) = c.config.device_spec_for(m)? {
        return Ok(vec![d.to_string()]);
    }
    let path = c.config.data_path(&m.path);
    // The service reuses these runtimes (the CLI has none until now).
    let rt = crate::benchmark::service::run_runtimes(c)?;
    c.runtimes = Some(rt.clone());
    let pick = rt
        .lock_recover()
        .selection(Some(&path))
        .auto_pick()
        .map(|o| o.spec.to_string());
    pick.map(|d| vec![d])
        .ok_or_else(|| anyhow!("no device can run {name} on this machine"))
}

/// Benchmark `name` on `sets` on `devices` one after the other (see the module docs). `ctx`
/// builds the run context when the service is free (the config must already list the model).
/// Waits for a running benchmark up to [`BUSY_WAIT`]; `cancel` stops the wait and the run.
pub fn run(
    svc: &Arc<BenchmarkService>,
    ctx: &dyn Fn() -> RunContext,
    name: &str,
    sets: Vec<DatasetRef>,
    devices: &[String],
    cancel: &AtomicBool,
) -> Result<Option<BenchSummary>> {
    if sets.is_empty() {
        return Ok(None);
    }
    let mut c = ctx();
    let devices = devices_for(&mut c, name, devices)?;
    let mut req = RunRequest::defaults(&c.config.benchmark);
    req.models = vec![name.to_string()];
    let fingerprint = sets.iter().find_map(|s| match s {
        DatasetRef::Dir { dir, .. } => dataset::read_heldout_fingerprint(dir),
        _ => None,
    });
    req.datasets = sets;
    req.devices = devices;
    req.reference_model = None;
    req.validate()?;
    let started = Instant::now();
    let mut pending = Some(c);
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!(Cancelled));
        }
        let c = match pending.take() {
            Some(c) => c,
            None => ctx(),
        };
        match svc.start(c, req.clone()) {
            Ok(()) => break,
            Err(_) if started.elapsed() < BUSY_WAIT => {
                std::thread::sleep(Duration::from_secs(2));
            }
            Err(_) => bail!(
                "another benchmark kept running for {} min; benchmark the model from the \
                 Benchmark page",
                BUSY_WAIT.as_secs() / 60
            ),
        }
    }
    while svc.is_running() {
        if cancel.load(Ordering::Relaxed) {
            svc.cancel();
            svc.wait_idle(Duration::from_secs(30));
            return Err(anyhow!(Cancelled));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let st = svc.status();
    if st.state == RunState::Failed {
        bail!(
            "the benchmark failed: {}",
            st.error.unwrap_or_else(|| "unknown error".into())
        );
    }
    let m = st
        .models
        .iter()
        .rfind(|m| m.model == name)
        .ok_or_else(|| anyhow!("the benchmark finished without a result for {name}"))?;
    if let Some(e) = &m.error {
        bail!("benchmarking {name} failed: {e}");
    }
    Ok(Some(BenchSummary {
        heldout_fingerprint: fingerprint,
        ..summarize(m)
    }))
}

/// The Training page's view of a benchmark result: the recommended device (else the first that
/// ran), its p50 time and AP50 (over every class of the model when the set annotates others).
pub fn summarize(m: &ModelResult) -> BenchSummary {
    let dev = m
        .devices
        .iter()
        .find(|d| d.ok() && m.recommended.as_deref() == Some(d.device.as_str()))
        .or_else(|| m.devices.iter().find(|d| d.ok()));
    let acc = dev
        .and_then(|d| d.run.as_ref())
        .and_then(|r| r.accuracy.as_ref());
    let (ap50, ap50_95) = match acc {
        Some(a) => match &a.all_classes {
            Some(all) if all.ap50.is_some() => (all.ap50, all.ap50_95),
            _ => (a.overall.ap50, a.overall.ap50_95),
        },
        None => (None, None),
    };
    let tags: &[Breakdown] = acc.map(|a| a.by_tag.as_slice()).unwrap_or_default();
    let ap = |b: &Breakdown| SceneAp {
        ap50: b.ap50,
        ap50_95: b.ap50_95,
        images: b.images,
    };
    let tagged = |key: &str| tags.iter().find(|b| b.key == key).map(ap);
    let mut by_scene = std::collections::BTreeMap::new();
    let mut in_scope = None;
    if tagged(dataset::HELDOUT_ALL).is_some() {
        for s in ["day", "night", "ir", "all"] {
            if let Some(a) = tagged(&format!("{}{s}", dataset::HELDOUT_TAG)) {
                by_scene.insert(s.to_string(), a);
            }
        }
        in_scope = tagged(dataset::HELDOUT_IN_SCOPE);
    } else if m.datasets.len() == 1
        && let Some(a) = acc
    {
        // An older held-out set alone: its bare scene tags are its own.
        for s in ["day", "night", "ir"] {
            if let Some(b) = tagged(s) {
                by_scene.insert(s.to_string(), b);
            }
        }
        by_scene.insert(
            "all".into(),
            SceneAp {
                ap50: a.overall.ap50,
                ap50_95: a.overall.ap50_95,
                images: a.overall.images,
            },
        );
    }
    BenchSummary {
        datasets: m.datasets.clone(),
        device: dev.map(|d| d.device.clone()),
        ms_p50: dev.and_then(|d| d.total_p50()),
        ap50,
        ap50_95,
        images: m.images,
        by_scene,
        in_scope,
        heldout_fingerprint: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with(device: Option<&str>) -> RunContext {
        let mut config = crate::config::Config::default();
        config.models = vec![crate::config::ModelConfig {
            name: Some("custom-1".into()),
            path: "models/custom-1.onnx".into(),
            device: device.map(str::to_string),
            ..Default::default()
        }];
        RunContext {
            config,
            results_path: "benchmark.json".into(),
            runtimes: None,
            loaded: Default::default(),
            resources: None,
        }
    }

    #[test]
    fn devices_are_the_chosen_ones_else_the_models_own() {
        // Chosen devices win, in order.
        let mut c = ctx_with(Some("openvino:cpu"));
        let chosen = ["ort:cuda:0".to_string(), "openvino:gpu.0".to_string()];
        assert_eq!(devices_for(&mut c, "custom-1", &chosen).unwrap(), chosen);
        // None chosen: the model's own device, one only, without starting any runtime.
        assert_eq!(
            devices_for(&mut c, "custom-1", &[]).unwrap(),
            ["openvino:cpu"]
        );
        assert!(c.runtimes.is_none());
        assert!(devices_for(&mut c, "other", &[]).is_err());
    }

    #[test]
    fn datasets_put_the_heldout_set_first() {
        let d = datasets(
            "night-yolo26n-x",
            Some(Path::new("/r/data/heldout")),
            &["nir-night".into()],
        );
        assert_eq!(d.len(), 2);
        assert_eq!(
            d[0],
            DatasetRef::Dir {
                dir: "/r/data/heldout".into(),
                gt: None,
                name: Some("heldout-night-yolo26n-x".into())
            }
        );
        assert_eq!(d[1], DatasetRef::Id("nir-night".into()));
        assert!(datasets("x", None, &[]).is_empty());
    }

    fn breakdown(key: &str, images: usize, ap: f64) -> Breakdown {
        Breakdown {
            key: key.into(),
            images,
            gt: images,
            precision: None,
            recall: None,
            f1: None,
            ap50: Some(ap + 0.2),
            ap50_95: Some(ap),
            relative: false,
            roc_auc: None,
        }
    }

    /// A trained model's result on its held-out set (+ `extra` datasets) with `tags`.
    fn result(tags: Vec<Breakdown>, extra: bool) -> ModelResult {
        use crate::benchmark::report::test_support::{dev, model, scene};
        let mut d = dev("openvino:cpu", 12.0, scene());
        let overall = crate::benchmark::metrics::Summary {
            images: 30,
            ap50: Some(0.7),
            ap50_95: Some(0.4),
            ..Default::default()
        };
        d.run.as_mut().unwrap().accuracy = Some(crate::benchmark::AccuracyReport {
            ground_truth: "ground truth".into(),
            relative: false,
            classes: vec!["person".into()],
            threshold: 0.5,
            overall,
            by_dataset: vec![],
            by_tag: tags,
            all_classes: None,
            sweep: None,
            frame_roc: None,
        });
        let mut m = model(vec![d], Some("openvino:cpu"));
        m.datasets = vec!["heldout-x".into()];
        if extra {
            m.datasets.push("nir-night".into());
        }
        m
    }

    #[test]
    fn summarize_reads_ap_per_scene_from_the_heldout_tags() {
        // Bare tags mix in a built-in set run beside; the `heldout:` ones do not.
        let tags = vec![
            breakdown("day", 50, 0.9),
            breakdown("night", 210, 0.1),
            breakdown("heldout:all", 30, 0.45),
            breakdown("heldout:day", 10, 0.6),
            breakdown("heldout:night", 12, 0.3),
            breakdown("heldout:ir", 8, 0.35),
            breakdown("heldout:in-scope", 20, 0.32),
            breakdown("res:sd", 30, 0.45),
        ];
        let b = summarize(&result(tags, true));
        assert_eq!(b.by_scene.len(), 4);
        let day = b.by_scene["day"];
        assert_eq!(
            (day.ap50_95, day.ap50, day.images),
            (Some(0.6), Some(0.8), 10)
        );
        assert_eq!(b.by_scene["night"].ap50_95, Some(0.3));
        assert_eq!(b.by_scene["ir"].images, 8);
        assert_eq!(b.by_scene["all"].ap50_95, Some(0.45));
        let s = b.in_scope.unwrap();
        assert_eq!((s.ap50_95, s.images), (Some(0.32), 20));
        assert_eq!(b.ms_p50, Some(12.0));
        assert_eq!(
            b.heldout_fingerprint, None,
            "set by run() from dataset.json"
        );
        // Serialized only when present (old records stay as they were).
        let j = serde_json::to_value(BenchSummary::default()).unwrap();
        assert!(j.get("by_scene").is_none() && j.get("in_scope").is_none());
        let back: BenchSummary = serde_json::from_str(
            r#"{"datasets":[],"device":null,"ms_p50":null,"ap50":null,"ap50_95":null,"images":0}"#,
        )
        .unwrap();
        assert!(back.by_scene.is_empty());
    }

    #[test]
    fn older_heldout_sets_use_the_bare_tags_only_when_alone() {
        let tags = vec![breakdown("day", 10, 0.6), breakdown("ir", 8, 0.35)];
        let b = summarize(&result(tags.clone(), false));
        assert_eq!(b.by_scene["day"].ap50_95, Some(0.6));
        assert_eq!(b.by_scene["ir"].ap50_95, Some(0.35));
        assert_eq!(b.by_scene["all"].ap50_95, Some(0.4));
        assert!(!b.by_scene.contains_key("night"));
        assert_eq!(b.in_scope, None);
        // Beside a built-in set the bare tags are not the held-out set's.
        assert!(summarize(&result(tags, true)).by_scene.is_empty());
    }
}
