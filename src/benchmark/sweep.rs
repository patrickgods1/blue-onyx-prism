//! Device sweep: run one model on every chosen device, compare each device with a CPU reference
//! and finish the `ModelResult` (recommendation, verdicts). `partial_summary` is the live view
//! while the sweep is still running.

use super::run::lock;
use super::*;

/// Which devices a sweep runs.
#[derive(Debug, Clone, Default)]
pub struct SweepOptions {
    /// Exactly these devices (an option that cannot run for the model becomes a failed row);
    /// None = every runnable option of the model's selection.
    pub devices: Option<Vec<Device>>,
    /// The device the model is configured to run on (kept when within the noise margin of the
    /// best; marked in the UI).
    pub configured: Option<String>,
    /// What the best confidence threshold optimizes.
    pub threshold_objective: threshold::Objective,
}

/// The per-image details of a device run that [`ModelResult::summary`] drops.
type RunDetails = (
    Vec<ImageRun>,
    Vec<Vec<PredBox>>,
    Option<threshold::ThresholdSweep>,
);

/// The partial result of a sweep over the filled `slots`, finished ([`ModelResult::finish`] on
/// the full runs) and returned in its [`ModelResult::summary`] form, without cloning any
/// per-image data (BM-11): the runs move into `result` for `finish`, their details are moved
/// aside while the summary is cloned, then everything moves back into `slots`.
pub(super) fn partial_summary(
    result: &mut ModelResult,
    slots: &mut [Option<DeviceResult>],
) -> ModelResult {
    let filled: Vec<usize> = (0..slots.len())
        .filter(|&i| slots.get(i).is_some_and(Option::is_some))
        .collect();
    result.devices = slots.iter_mut().filter_map(Option::take).collect();
    result.finish();
    let details: Vec<Option<RunDetails>> = result
        .devices
        .iter_mut()
        .map(|d| {
            d.run.as_mut().map(|r| {
                (
                    std::mem::take(&mut r.per_image),
                    std::mem::take(&mut r.eval_preds),
                    r.accuracy.as_mut().and_then(|a| a.sweep.take()),
                )
            })
        })
        .collect();
    let summary = result.clone();
    for (d, det) in result.devices.iter_mut().zip(details) {
        if let (Some(r), Some((per_image, eval_preds, sweep))) = (d.run.as_mut(), det) {
            r.per_image = per_image;
            r.eval_preds = eval_preds;
            if let Some(a) = r.accuracy.as_mut() {
                a.sweep = sweep;
            }
        }
    }
    for (i, d) in filled.into_iter().zip(result.devices.drain(..)) {
        if let Some(slot) = slots.get_mut(i) {
            *slot = Some(d);
        }
    }
    summary
}

/// Run `job` on every chosen device, compare detections with the CPU reference device (OpenVINO
/// CPU, else ONNX Runtime CPU, whichever ran) and recommend a device. The reference CPU devices
/// run first so each later result can be judged as it arrives; the result lists devices in the
/// selection's order. `on_device` sees the partial result after each device (live progress), in
/// its [`ModelResult::summary`] form (no per-image details: BM-11).
///
/// Errors only with [`Cancelled`]; device failures are recorded in the result.
pub fn sweep(
    bench: &Bench,
    runtimes: &Mutex<Runtimes>,
    job: &Job,
    opts: &SweepOptions,
    on_device: &mut dyn FnMut(&ModelResult),
) -> Result<ModelResult> {
    let selection = lock(runtimes).selection(Some(&job.path));
    let mut result = ModelResult::new(job, bench, opts.configured.clone());
    result.threshold_objective = opts.threshold_objective;

    // (selection order, device, label, why it cannot run).
    let mut planned: Vec<(usize, Device, String, Option<String>)> = Vec::new();
    match &opts.devices {
        None => {
            for (i, o) in selection.options.iter().enumerate() {
                if o.runnable {
                    planned.push((i, o.spec, o.label.clone(), None));
                } else {
                    result.skipped.push(format!(
                        "{}: {}",
                        o.spec,
                        o.reason.as_deref().unwrap_or("not runnable")
                    ));
                }
            }
        }
        Some(devices) => {
            for d in devices {
                match selection
                    .options
                    .iter()
                    .enumerate()
                    .find(|(_, o)| o.spec == *d)
                {
                    Some((i, o)) => {
                        let why = (!o.runnable)
                            .then(|| o.reason.clone().unwrap_or_else(|| "not runnable".into()));
                        planned.push((i, o.spec, o.label.clone(), why));
                    }
                    None => planned.push((
                        usize::MAX,
                        *d,
                        d.to_string(),
                        Some("not a device option on this machine".into()),
                    )),
                }
            }
        }
    }
    planned.sort_by_key(|p| p.0);
    // Run order: the reference candidates first.
    let mut order: Vec<usize> = (0..planned.len()).collect();
    order.sort_by_key(|&k| {
        REFERENCE_DEVICES
            .iter()
            .position(|r| planned.get(k).is_some_and(|p| *r == p.1))
            .unwrap_or(REFERENCE_DEVICES.len())
    });

    let mut slots: Vec<Option<DeviceResult>> = vec![None; planned.len()];
    for k in order {
        // `order` is a permutation of 0..planned.len() and `slots` has the same length.
        let (Some((_, device, label, why)), true) = (planned.get(k), k < slots.len()) else {
            continue;
        };
        let entry = match why {
            Some(reason) => DeviceResult::failed(device, label, reason.clone()),
            None => match bench.run_exact(runtimes, job, *device) {
                Ok(run) => DeviceResult::ran(device, label, run),
                Err(e) if is_cancelled(&e) => return Err(e),
                Err(e) => {
                    tracing::warn!(model = %job.name, device = %device, "benchmark run failed: {e:#}");
                    DeviceResult::failed(device, label, format!("{e:#}"))
                }
            },
        };
        if let Some(slot) = slots.get_mut(k) {
            *slot = Some(entry);
        }
        let partial = partial_summary(&mut result, &mut slots);
        on_device(&partial);
    }
    result.devices = slots.into_iter().flatten().collect();
    result.finish();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_without_runtimes_records_failures() {
        // No OpenVINO and no ONNX Runtime: every explicitly requested device fails with its
        // reason, nothing is recommended, and the sweep itself succeeds.
        let dir = std::env::temp_dir().join(format!("bop-sweep-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.onnx");
        std::fs::write(&model, b"not a model").unwrap();
        let rt = Runtimes::new_with(
            &crate::backend::CoreOptions {
                openvino_dir: Some(dir.join("no-openvino")),
                ..Default::default()
            },
            &crate::backend::OrtOptions {
                onnxruntime_dir: Some(dir.join("no-ort")),
                default_dir: Some(dir.join("no-ort")),
                cuda_libs_dir: None,
            },
        );
        if rt.openvino().is_some() || rt.has_ort() {
            // A system-wide runtime was found; this test is about the failure path only.
            return;
        }
        let rt = Mutex::new(rt);
        let job = Job {
            name: "m".into(),
            path: model,
            family: ModelFamilyKind::Yolo5,
            classes: vec!["a".into()],
            device: "auto".into(),
            gpu_precision: None,
            confidence_threshold: None,
            object_filter: None,
        };
        let params = PostParams {
            confidence_threshold: 0.5,
            nms_iou: 0.5,
        };
        let sets = [ImageSet::sample()];
        let bench = Bench::new(&sets, 0, 1, params);
        let opts = SweepOptions {
            devices: Some(REFERENCE_DEVICES.to_vec()),
            configured: None,
            ..Default::default()
        };
        let mut seen = 0;
        let r = sweep(&bench, &rt, &job, &opts, &mut |_| seen += 1).unwrap();
        assert_eq!(seen, 2);
        assert_eq!(r.devices.len(), 2);
        assert!(
            r.devices
                .iter()
                .all(|d| d.error.is_some() && d.run.is_none())
        );
        assert_eq!(r.recommended, None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The live partial view of a sweep equals the summary of the finished partial result,
    /// and the slots keep their per-image details for the final result (BM-11).
    #[test]
    fn partial_summary_matches_the_summary_and_keeps_details() {
        let run = |spec: &str, conf: f32| {
            let mut r: RunResult = serde_json::from_value(serde_json::json!({
                "model": "m",
                "device": spec,
                "spec": spec,
                "detections": [{"label": "dog", "confidence": conf,
                    "x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}],
            }))
            .unwrap();
            r.stages_ms.total.p50 = 10.0;
            r.per_image = vec![ImageRun {
                set: "s".into(),
                file: "a.jpg".into(),
                total_ms: 10.0,
                infer_ms: 5.0,
                preds: Vec::new(),
                gt_matched: Vec::new(),
                counts: None,
            }];
            r.eval_preds = vec![vec![PredBox {
                label: "dog".into(),
                confidence: conf,
                bbox: [0.0, 0.0, 10.0, 10.0],
            }]];
            r
        };
        let dev = |spec: &str, conf: f32| {
            let d = spec::parse(spec).unwrap();
            DeviceResult::ran(d.device().unwrap(), spec, run(spec, conf))
        };
        let mut result = ModelResult::failed("m", "m.onnx", String::new());
        result.error = None;
        let mut slots = vec![
            Some(dev("ort:cpu", 0.8)),
            None,
            Some(dev("openvino:cpu", 0.9)),
        ];
        let before = slots.clone();
        // What the sweep did before: finish a full clone, then the caller summarized it.
        let mut old = result.clone();
        old.devices = slots.iter().flatten().cloned().collect();
        old.finish();
        let expected = old.summary();

        let got = partial_summary(&mut result, &mut slots);
        assert_eq!(got, expected);
        assert!(
            got.devices
                .iter()
                .all(|d| d.run.as_ref().unwrap().per_image.is_empty())
        );
        // The slots are where they were, with their details (and the finished agreement).
        assert!(slots[1].is_none());
        for (s, b) in slots.iter().zip(&before) {
            let (Some(s), Some(b)) = (s, b) else { continue };
            let (r, rb) = (s.run.as_ref().unwrap(), b.run.as_ref().unwrap());
            assert_eq!(
                (&r.per_image, &r.eval_preds),
                (&rb.per_image, &rb.eval_preds)
            );
            assert!(r.agreement.is_some());
        }
        assert!(result.devices.is_empty());
    }
}
