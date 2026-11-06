//! Load plan: the ordered devices a model is tried on, and the loop that tries them.
//!
//! Pure (no runtime libraries): [`plan_for`] turns a [`DeviceSpec`] plus the machine's
//! [`Selection`] into candidates, and [`try_candidates`] runs a caller-supplied attempt
//! (compile + create request + warm-up in the worker) on each until one succeeds.
//!
//! Semantics:
//! - `auto` -> the `select` ranking ([`auto_candidates`]; reordered by the model's benchmark
//!   speed order when it has one, `Selection::apply_measured_order`): the first candidate is not
//!   a fallback, every later one (down to CPU) is. A machine whose ranking is CPU only loads CPU as
//!   a regular pick, with a note when a GPU was found but nothing GPU-class can run.
//! - `openvino:gpu[.N]` -> GPU, then CPU as a fallback. When OpenVINO does not list the requested
//!   GPU (no GPU at all for `gpu`, no `GPU.N` for `gpu.N`), only CPU is tried, with a note, and it
//!   counts as a fallback (unchanged from the old `DeviceSelection`).
//! - `openvino:cpu` / `openvino:npu` -> just that device (plus ONNX Runtime CPU as a fallback
//!   when OpenVINO cannot run it, e.g. the OpenVINO runtime is missing).
//! - `ort:*` -> that device (with a note when the detected hardware has no such index), then
//!   the best other CPU option ([`best_cpu`]: OpenVINO CPU, else ONNX Runtime CPU) as a fallback.
//!
//! "No GPU" fallbacks also use [`best_cpu`], so a legacy `GPU` config on a machine with only
//! ONNX Runtime lands on `ort:cpu`.

use super::detect::HardwareInfo;
use super::select::{RuntimeProbe, Selection, select};
use super::spec::{Device, DeviceSpec, Runtime, Target};
use anyhow::Result;
use tracing::warn;

/// One device to try.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub device: Device,
    /// Reported as `DeviceInfo::fell_back` (", fallback" in `executionProvider`) when this
    /// candidate is the one that loads.
    pub fell_back: bool,
    /// Why the plan already deviates from the request before anything was tried (logged).
    pub note: Option<String>,
}

impl Candidate {
    fn new(device: Device, fell_back: bool) -> Self {
        Self {
            device,
            fell_back,
            note: None,
        }
    }
}

/// Why a device candidate failed to load, as far as trying it again is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadFailure {
    /// The device has no memory for it now; something else may free some.
    Memory,
    /// The device stopped answering or was reset; it may come back.
    DeviceLost,
    /// Anything else: a model that does not compile there or a missing runtime fails the same
    /// way every time.
    Other,
}

impl LoadFailure {
    /// Classify the error of a compile or a warm-up by its text: the runtimes report these
    /// conditions only in their messages (OpenCL, CUDA and DXGI names), and neither the
    /// backend's own context nor a model path holds any of the phrases below.
    pub fn of(e: &anyhow::Error) -> Self {
        Self::of_text(&format!("{e:#}"))
    }

    /// [`Self::of`] on the text of an error.
    pub fn of_text(text: &str) -> Self {
        const MEMORY: &[&str] = &[
            "out of memory",
            "out_of_memory",
            "outofmemory",
            "cl_out_of_resources",
            "cl_out_of_host_memory",
            "cl_mem_object_allocation_failure",
            "out of gpu resources",
            "exceeded max size of memory object allocation",
            "failed to allocate",
            "allocation failed",
            "bad_alloc",
            "bad allocation",
            "not enough memory",
            "insufficient memory",
            "8007000e",
        ];
        const DEVICE_LOST: &[&str] = &[
            "device lost",
            "device_lost",
            "device removed",
            "device_removed",
            "device hung",
            "device_hung",
            "device_reset",
            "device instance has been suspended",
            "may have been reset or lost",
            "the device driver may have hung",
            "cl_device_not_available",
            "cuda_error_unknown",
            "cuda failure 999",
            "launch_timeout",
            "fallen off the bus",
            "887a0005",
            "887a0006",
            "887a0007",
            "887a0020",
        ];
        let text = text.to_ascii_lowercase();
        let has = |list: &[&str]| list.iter().any(|p| text.contains(p));
        if has(MEMORY) {
            Self::Memory
        } else if has(DEVICE_LOST) {
            Self::DeviceLost
        } else {
            Self::Other
        }
    }

    /// True when the same candidate may work later.
    pub fn retryable(self) -> bool {
        self != Self::Other
    }
}

/// A candidate better than the device a model loaded on, which failed for a reason that may
/// pass ([`LoadFailure::retryable`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Better {
    pub candidate: Candidate,
    pub failure: LoadFailure,
    /// The error, for the log.
    pub reason: String,
}

impl Better {
    /// `candidate`'s failure `e`, when it is worth another try.
    pub fn of(candidate: &Candidate, e: &anyhow::Error) -> Option<Self> {
        let failure = LoadFailure::of(e);
        failure.retryable().then(|| Self {
            candidate: candidate.clone(),
            failure,
            reason: format!("{e:#}"),
        })
    }
}

/// True when `now` comes after `before` in `candidates`: the model runs on a lesser device than
/// it did. Specs are canonical (`DeviceInfo::spec`); one that is not listed compares as false.
pub fn lesser_device(candidates: &[Candidate], before: &str, now: &str) -> bool {
    let at = |spec: &str| candidates.iter().position(|c| c.device.to_string() == spec);
    matches!((at(before), at(now)), (Some(b), Some(n)) if n > b)
}

/// Ordered candidates for `spec` on a machine described by `sel` (from [`select`]).
pub fn plan_for(
    spec: &DeviceSpec,
    sel: &Selection,
    available_ov_devices: &[String],
) -> Vec<Candidate> {
    match spec {
        DeviceSpec::Auto => auto_candidates(sel),
        DeviceSpec::Device(d) => explicit_candidates(*d, sel, available_ov_devices),
    }
}

/// Ordered candidates for `spec` knowing only the device names OpenVINO lists
/// (`["CPU", "GPU.0", ..]`): `auto` is ranked as if no other runtime or GPU were present, which
/// for OpenVINO with an Intel GPU is `openvino:gpu`, then `openvino:cpu`.
pub fn plan_candidates(spec: &DeviceSpec, available_ov_devices: &[String]) -> Vec<Candidate> {
    let sel = select(
        &HardwareInfo::default(),
        &RuntimeProbe::openvino_only(available_ov_devices),
        true,
    );
    plan_for(spec, &sel, available_ov_devices)
}

fn ort_cpu() -> Device {
    Device {
        runtime: Runtime::Ort,
        target: Target::Cpu,
        index: None,
    }
}

fn runnable(sel: &Selection, d: &Device) -> bool {
    sel.option(d).is_some_and(|o| o.runnable)
}

/// The best CPU option (`force_cpu`, "no GPU" fallbacks): OpenVINO CPU when it can run, else
/// ONNX Runtime CPU when it can, else OpenVINO CPU (so the error names the missing runtime).
pub fn best_cpu(sel: &Selection) -> Device {
    [Device::OPENVINO_CPU, ort_cpu()]
        .into_iter()
        .find(|d| runnable(sel, d))
        .unwrap_or(Device::OPENVINO_CPU)
}

/// CPU fallback for an explicit `device`: the first runnable CPU option other than `device`,
/// else OpenVINO CPU unless that is `device` itself.
fn fallback_cpu(sel: &Selection, device: Device) -> Option<Device> {
    [Device::OPENVINO_CPU, ort_cpu()]
        .into_iter()
        .filter(|d| *d != device)
        .find(|d| runnable(sel, d))
        .or_else(|| (device != Device::OPENVINO_CPU).then_some(Device::OPENVINO_CPU))
}

/// The `auto` ranking as candidates. Without anything runnable, OpenVINO CPU is still tried so
/// the error names the missing runtime.
pub fn auto_candidates(sel: &Selection) -> Vec<Candidate> {
    if sel.auto.is_empty() {
        return vec![Candidate {
            note: Some(
                "no runnable device found, because no inference runtime loaded (OpenVINO and \
                 ONNX Runtime missing or failed to initialize); trying OpenVINO CPU so its error \
                 is shown; run `setup-openvino` (or `setup-onnxruntime`) and check `list-devices`"
                    .to_string(),
            ),
            ..Candidate::new(Device::OPENVINO_CPU, false)
        }];
    }
    let mut out: Vec<Candidate> = sel
        .auto
        .iter()
        .enumerate()
        .map(|(i, d)| Candidate::new(*d, i > 0))
        .collect();
    if let [only] = out.as_mut_slice()
        && only.device.is_cpu()
    {
        let blocked: Vec<String> = sel
            .options
            .iter()
            .filter(|o| !o.runnable && o.spec.is_gpu_like())
            .map(|o| {
                format!(
                    "{}: {}",
                    o.spec,
                    o.reason.as_deref().unwrap_or("not runnable")
                )
            })
            .collect();
        if !blocked.is_empty() {
            only.note = Some(format!(
                "no GPU option can run ({}); using {}; fix the reason shown for the GPU (or \
                 download what `list-devices` offers) to use it",
                blocked.join("; "),
                only.device
            ));
        }
    }
    out
}

/// An explicit device, then its fallback.
fn explicit_candidates(
    device: Device,
    sel: &Selection,
    available_ov_devices: &[String],
) -> Vec<Candidate> {
    let fallback = |d: Device| Candidate::new(d, true);
    match (device.runtime, device.target) {
        (Runtime::OpenVino, Target::Gpu) => {
            let cpu = best_cpu(sel);
            if ov_gpu_listed(available_ov_devices, device.index) {
                vec![Candidate::new(device, false), fallback(cpu)]
            } else {
                let requested = device.openvino_device().unwrap_or_default();
                let using = if cpu == Device::OPENVINO_CPU {
                    "CPU".to_string()
                } else {
                    cpu.to_string()
                };
                let cause = if available_ov_devices.iter().any(|d| is_ov_gpu(d)) {
                    "the GPU index is probably wrong"
                } else {
                    "OpenVINO sees no GPU (Intel graphics driver / compute runtime missing?)"
                };
                vec![Candidate {
                    note: Some(format!(
                        "requested device {requested} not available (devices: {available_ov_devices:?}); \
                         {cause}; using {using}; set `device` to a listed device or run `list-devices`"
                    )),
                    ..fallback(cpu)
                }]
            }
        }
        (Runtime::OpenVino, _) => {
            let mut out = vec![Candidate::new(device, false)];
            if !runnable(sel, &device)
                && let Some(cpu) = fallback_cpu(sel, device).filter(|d| runnable(sel, d))
            {
                out.push(fallback(cpu));
            }
            out
        }
        (Runtime::Ort, _) => {
            let mut out = vec![Candidate {
                note: missing_index_note(device, sel),
                ..Candidate::new(device, false)
            }];
            out.extend(fallback_cpu(sel, device).map(fallback));
            out
        }
    }
}

/// An OpenVINO GPU device name: `GPU` or `GPU.N`.
fn is_ov_gpu(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper == "GPU" || upper.starts_with("GPU.")
}

/// Whether OpenVINO lists the GPU `index` asks for: any GPU for a bare `openvino:gpu`; `GPU.N`
/// for `openvino:gpu.N`, or a bare `GPU` for N = 0 when OpenVINO lists no numbered GPU (one GPU
/// is listed as `GPU`, which is `GPU.0`).
fn ov_gpu_listed(available: &[String], index: Option<u32>) -> bool {
    let gpus: Vec<&String> = available.iter().filter(|d| is_ov_gpu(d)).collect();
    match index {
        None => !gpus.is_empty(),
        Some(n) => {
            let want = format!("GPU.{n}");
            gpus.iter().any(|d| d.eq_ignore_ascii_case(&want))
                || (n == 0
                    && gpus.iter().any(|d| d.eq_ignore_ascii_case("GPU"))
                    && !gpus.iter().any(|d| d.contains('.')))
        }
    }
}

/// For an explicit ONNX Runtime device with an index (`ort:cuda:3`): a note when the detected
/// hardware lists devices of that kind but none with that index. The device is still tried
/// (detection can miss a GPU), the note says why it will most likely fail.
fn missing_index_note(device: Device, sel: &Selection) -> Option<String> {
    let n = device.index?;
    let have: Vec<String> = sel
        .options
        .iter()
        .filter(|o| {
            o.spec.runtime == device.runtime
                && o.spec.target == device.target
                && o.spec.index.is_some()
        })
        .map(|o| o.spec.to_string())
        .collect();
    if have.is_empty() || have.contains(&device.to_string()) {
        return None;
    }
    Some(format!(
        "no {} device with index {n} (have {}: {}); the index is probably wrong; trying it anyway, \
         set `device` to a listed one or run `list-devices`",
        device.target.as_str(),
        have.len(),
        have.join(", ")
    ))
}

/// Run `attempt` on each candidate in order and return the first success. Failures are logged
/// (`warn!`, prefixed with `label`) and, when every candidate fails, listed in the error.
pub fn try_candidates<T>(
    label: &str,
    candidates: &[Candidate],
    mut attempt: impl FnMut(&Candidate) -> Result<T>,
) -> Result<T> {
    let mut failures: Vec<(String, anyhow::Error)> = Vec::new();
    for (i, cand) in candidates.iter().enumerate() {
        if let Some(note) = &cand.note {
            warn!("{label}: {note}");
        }
        match attempt(cand) {
            Ok(v) => return Ok(v),
            Err(e) => {
                match candidates.get(i + 1) {
                    Some(next) => warn!(
                        "{label}: {} failed: {e:#}; trying {}",
                        cand.device, next.device
                    ),
                    None if candidates.len() > 1 => warn!("{label}: {} failed: {e:#}", cand.device),
                    None => {}
                }
                failures.push((cand.device.to_string(), e));
            }
        }
    }
    match failures.as_slice() {
        [] => anyhow::bail!("{label}: no device to try"),
        // A single candidate keeps its own error chain.
        [_] => Err(failures
            .into_iter()
            .map(|(_, e)| e)
            .next()
            .unwrap_or_else(|| anyhow::anyhow!("{label}: no device to try"))),
        _ => {
            let list: Vec<String> = failures
                .iter()
                .map(|(dev, e)| format!("{dev}: {e:#}"))
                .collect();
            anyhow::bail!("every device failed ({})", list.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::spec::parse;

    fn devs(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// (spec, fell_back) pairs of a plan.
    fn plan(spec: &str, available: &[&str]) -> Vec<(String, bool)> {
        plan_candidates(&parse(spec).unwrap(), &devs(available))
            .into_iter()
            .map(|c| (c.device.to_string(), c.fell_back))
            .collect()
    }

    fn s(v: &[(&str, bool)]) -> Vec<(String, bool)> {
        v.iter().map(|(d, f)| (d.to_string(), *f)).collect()
    }

    /// Texts the runtimes report (or close to them), and what each means for a retry.
    #[test]
    fn load_failures_classified() {
        use LoadFailure::{DeviceLost, Memory, Other};
        for (text, want) in [
            // OpenVINO GPU plugin.
            ("[GPU] out of GPU resources", Memory),
            (
                "[GPU] clEnqueueMapBuffer, error code: -5 CL_OUT_OF_RESOURCES",
                Memory,
            ),
            ("error code: -4 CL_MEM_OBJECT_ALLOCATION_FAILURE", Memory),
            ("error code: -6 CL_OUT_OF_HOST_MEMORY", Memory),
            (
                "[GPU] Exceeded max size of memory object allocation: requested 3 GB",
                Memory,
            ),
            ("std::bad_alloc", Memory),
            ("bad allocation", Memory),
            // ONNX Runtime CUDA / DirectML.
            ("CUDA failure 2: out of memory ; GPU=0", Memory),
            ("CUDA_ERROR_OUT_OF_MEMORY", Memory),
            (
                "Failed to allocate memory for requested buffer of size 123",
                Memory,
            ),
            (
                "DmlExecutionProvider: The parameter is incorrect, 0x8007000E",
                Memory,
            ),
            ("Not enough memory resources are available", Memory),
            ("kIOGPUCommandBufferCallbackErrorOutOfMemory", Memory),
            // A device that went away.
            ("DXGI_ERROR_DEVICE_REMOVED", DeviceLost),
            ("DXGI_ERROR_DEVICE_HUNG", DeviceLost),
            ("DXGI_ERROR_DEVICE_RESET", DeviceLost),
            ("HRESULT 0x887A0005", DeviceLost),
            ("0x887A0006", DeviceLost),
            ("0x887A0007", DeviceLost),
            ("0x887A0020", DeviceLost),
            (
                "The GPU device instance has been suspended. Use GetDeviceRemovedReason",
                DeviceLost,
            ),
            ("CUDA_ERROR_UNKNOWN", DeviceLost),
            ("CUDA failure 999: unknown error", DeviceLost),
            ("CUDA_ERROR_LAUNCH_TIMEOUT", DeviceLost),
            ("GPU has fallen off the bus", DeviceLost),
            ("CL_DEVICE_NOT_AVAILABLE", DeviceLost),
            (
                "model m.xml on openvino:gpu: starting inference failed; the GPU/device may have been reset or lost; retry",
                DeviceLost,
            ),
            (
                "warm-up inference on ort:cuda did not finish within 180 s; the device driver may have hung or the GPU is not available yet",
                DeviceLost,
            ),
            // Fails the same way every time.
            ("[GPU] Unsupported operation: Foo", Other),
            ("Exception from src/core/shape_inference: wrong rank", Other),
            (
                "OpenVINO is not available (library not found); its runtime libraries are probably missing or of another version; run `setup-openvino`",
                Other,
            ),
            (
                "ONNX Runtime is not available (x); its library is probably not installed",
                Other,
            ),
            ("no CUDA-capable device is detected", Other),
            ("the model's outputs do not fit family `yolo8`", Other),
            ("model file not found: /models/out of order.onnx", Other),
            ("memory pattern disabled for this session", Other),
            ("device openvino:npu is not available", Other),
            ("", Other),
        ] {
            assert_eq!(LoadFailure::of_text(text), want, "{text}");
        }
        assert!(Memory.retryable());
        assert!(DeviceLost.retryable());
        assert!(!Other.retryable());
        // Memory wins when an error names both.
        assert_eq!(
            LoadFailure::of_text("CUDA_ERROR_OUT_OF_MEMORY after device lost"),
            Memory
        );
    }

    #[test]
    fn better_candidates_are_the_ones_worth_another_try() {
        let cand = Candidate::new(Device::OPENVINO_CPU, false);
        let oom = anyhow::anyhow!("CL_OUT_OF_RESOURCES").context("compiling");
        let b = Better::of(&cand, &oom).unwrap();
        assert_eq!(b.failure, LoadFailure::Memory);
        assert_eq!(b.candidate, cand);
        assert_eq!(b.reason, "compiling: CL_OUT_OF_RESOURCES");
        assert_eq!(Better::of(&cand, &anyhow::anyhow!("no such op")), None);
    }

    #[test]
    fn lesser_device_follows_the_candidate_order() {
        let cands = plan_candidates(&parse("GPU").unwrap(), &devs(&["CPU", "GPU"]));
        assert!(lesser_device(&cands, "openvino:gpu", "openvino:cpu"));
        assert!(!lesser_device(&cands, "openvino:cpu", "openvino:gpu"));
        assert!(!lesser_device(&cands, "openvino:gpu", "openvino:gpu"));
        assert!(
            !lesser_device(&cands, "ort:cuda", "openvino:cpu"),
            "unlisted"
        );
    }

    #[test]
    fn gpu_with_and_without_gpu_present() {
        let with_gpu = ["CPU", "GPU"];
        assert_eq!(
            plan("GPU", &with_gpu),
            s(&[("openvino:gpu", false), ("openvino:cpu", true)])
        );
        assert_eq!(
            plan("gpu.1", &["CPU", "GPU.0", "GPU.1"]),
            s(&[("openvino:gpu.1", false), ("openvino:cpu", true)])
        );
        // No GPU listed: only CPU, and it counts as a fallback (old DeviceSelection behavior).
        assert_eq!(plan("GPU", &["CPU"]), s(&[("openvino:cpu", true)]));
        let c = plan_candidates(&parse("GPU").unwrap(), &devs(&["CPU"]));
        assert!(c[0].note.as_deref().unwrap().contains("GPU not available"));
        assert_eq!(plan("GPU.1", &[]), s(&[("openvino:cpu", true)]));
    }

    #[test]
    fn gpu_index_must_be_listed() {
        // GPU.3 on a machine listing only GPU.0: straight to the CPU, with a note.
        assert_eq!(
            plan("gpu.3", &["CPU", "GPU.0"]),
            s(&[("openvino:cpu", true)])
        );
        let c = plan_candidates(&parse("gpu.3").unwrap(), &devs(&["CPU", "GPU.0"]));
        let note = c[0].note.as_deref().unwrap();
        assert!(
            note.contains("GPU.3 not available") && note.contains("index is probably wrong"),
            "{note}"
        );
        // One GPU listed as bare "GPU" is GPU.0.
        assert_eq!(
            plan("gpu.0", &["CPU", "GPU"]),
            s(&[("openvino:gpu.0", false), ("openvino:cpu", true)])
        );
        assert_eq!(plan("gpu.1", &["CPU", "GPU"]), s(&[("openvino:cpu", true)]));
        assert_eq!(
            plan("gpu.1", &["CPU", "gpu.0", "GPU.1"]),
            s(&[("openvino:gpu.1", false), ("openvino:cpu", true)])
        );
        assert!(ov_gpu_listed(&devs(&["GPU.0"]), None));
        assert!(!ov_gpu_listed(&devs(&["CPU", "NPU"]), None));
        assert!(!ov_gpu_listed(&devs(&["GPUX"]), Some(0)));
    }

    #[test]
    fn ort_index_not_detected_gets_a_note() {
        use crate::backend::detect::{GpuAdapter, GpuVendor};
        use crate::backend::select::{EpStatus, OrtProbe};
        let nvidia = GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: "NVIDIA GeForce RTX 3060".into(),
            vram_mb: 12288,
            index: 0,
            discrete: true,
            cuda: None,
        };
        let hw = HardwareInfo::new("linux", "x86_64", vec![nvidia]);
        let ov = devs(&["CPU"]);
        let mut probe = RuntimeProbe::openvino_only(&ov);
        probe.ort = OrtProbe::installed(
            "gpu",
            vec![
                EpStatus::usable(Target::Cuda),
                EpStatus::usable(Target::Cpu),
            ],
        );
        let sel = select(&hw, &probe, true);
        let c = plan_for(&parse("ort:cuda:3").unwrap(), &sel, &ov);
        assert_eq!(c[0].device.to_string(), "ort:cuda:3");
        let note = c[0].note.as_deref().unwrap();
        assert!(
            note.contains("no cuda device with index 3 (have 1: ort:cuda:0)"),
            "{note}"
        );
        assert!(
            plan_for(&parse("ort:cuda:0").unwrap(), &sel, &ov)[0]
                .note
                .is_none()
        );
        assert!(
            plan_for(&parse("ort:cuda").unwrap(), &sel, &ov)[0]
                .note
                .is_none()
        );
        // Nothing detected of that kind: no note (nothing to compare with).
        assert!(
            plan_for(&parse("ort:directml:5").unwrap(), &sel, &ov)[0]
                .note
                .is_none()
        );
    }

    #[test]
    fn cpu_npu_auto_and_ort() {
        assert_eq!(plan("CPU", &["CPU", "GPU"]), s(&[("openvino:cpu", false)]));
        assert_eq!(
            plan("openvino:npu", &["CPU", "NPU"]),
            s(&[("openvino:npu", false)])
        );
        assert_eq!(
            plan("auto", &["CPU", "GPU"]),
            plan("openvino:gpu", &["CPU", "GPU"])
        );
        // CPU-only machine: CPU is auto's regular pick, not a fallback, and there is no GPU to
        // complain about.
        assert_eq!(plan("auto", &["CPU"]), s(&[("openvino:cpu", false)]));
        let c = plan_candidates(&parse("auto").unwrap(), &devs(&["CPU"]));
        assert!(c[0].note.is_none());
        // Without OpenVINO devices nothing is runnable; CPU is still tried, with a note.
        let c = plan_candidates(&parse("auto").unwrap(), &[]);
        assert_eq!(c.len(), 1);
        assert_eq!((c[0].device, c[0].fell_back), (Device::OPENVINO_CPU, false));
        assert!(c[0].note.is_some());
        assert_eq!(
            plan("ort:cuda:1", &["CPU"]),
            s(&[("ort:cuda:1", false), ("openvino:cpu", true)])
        );
        assert_eq!(
            plan("ort:cpu", &["CPU"]),
            s(&[("ort:cpu", false), ("openvino:cpu", true)])
        );
        for c in plan_candidates(&parse("ort:coreml").unwrap(), &devs(&["CPU"])) {
            assert!(c.note.is_none());
        }
    }

    #[test]
    fn auto_follows_the_selection() {
        use crate::backend::detect::{GpuAdapter, GpuVendor};
        use crate::backend::select::{EpStatus, OrtProbe};
        let nvidia = GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: "NVIDIA GeForce RTX 3060".into(),
            vram_mb: 12288,
            index: 0,
            discrete: true,
            cuda: None,
        };
        let hw = HardwareInfo::new("linux", "x86_64", vec![nvidia]);
        let ov = devs(&["CPU", "GPU"]);
        let mut probe = RuntimeProbe::openvino_only(&ov);
        probe.ort = OrtProbe::installed(
            "gpu",
            vec![
                EpStatus::usable(Target::Cuda),
                EpStatus::usable(Target::Cpu),
            ],
        );
        let sel = select(&hw, &probe, true);
        let got: Vec<(String, bool)> = plan_for(&DeviceSpec::Auto, &sel, &ov)
            .into_iter()
            .map(|c| (c.device.to_string(), c.fell_back))
            .collect();
        assert_eq!(
            got,
            s(&[
                ("ort:cuda:0", false),
                ("openvino:gpu", true),
                ("openvino:cpu", true)
            ])
        );
        // Explicit specs ignore the ranking.
        assert_eq!(
            plan_for(&parse("GPU").unwrap(), &sel, &ov),
            plan_candidates(&parse("GPU").unwrap(), &ov)
        );

        // NVIDIA found but CUDA unusable and no OpenVINO GPU: CPU with a note naming CUDA.
        probe.ort =
            OrtProbe::installed("gpu", vec![EpStatus::broken(Target::Cuda, "cudnn missing")]);
        let cpu_only = devs(&["CPU"]);
        probe.openvino = crate::backend::select::OpenVinoProbe::with_device_names(&cpu_only);
        let c = auto_candidates(&select(&hw, &probe, true));
        assert_eq!(c.len(), 1);
        assert!(!c[0].fell_back);
        let note = c[0].note.as_deref().unwrap();
        assert!(note.contains("ort:cuda:0: cudnn missing"), "{note}");
    }

    #[test]
    fn tries_in_order_until_success() {
        let cands = plan_candidates(&parse("GPU").unwrap(), &devs(&["CPU", "GPU"]));
        let mut tried = Vec::new();
        let got = try_candidates("m", &cands, |c| {
            tried.push(c.device.to_string());
            if c.device.is_cpu() {
                Ok(c.fell_back)
            } else {
                anyhow::bail!("compile failed")
            }
        })
        .unwrap();
        assert!(got, "CPU after a GPU failure is a fallback");
        assert_eq!(tried, ["openvino:gpu", "openvino:cpu"]);

        let mut n = 0;
        let first = try_candidates("m", &cands, |c| {
            n += 1;
            Ok(c.device)
        })
        .unwrap();
        assert_eq!((first, n), (Device::OPENVINO_GPU, 1));
    }

    #[test]
    fn error_lists_every_failure() {
        let cands = plan_candidates(&parse("ort:cuda").unwrap(), &devs(&["CPU"]));
        let err = try_candidates::<()>("m", &cands, |c| anyhow::bail!("boom on {}", c.device))
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("ort:cuda: boom on ort:cuda"), "{msg}");
        assert!(msg.contains("openvino:cpu: boom on openvino:cpu"), "{msg}");

        // One candidate: its error is returned as is.
        let cands = plan_candidates(&parse("CPU").unwrap(), &devs(&["CPU"]));
        let err = try_candidates::<()>("m", &cands, |_| {
            Err(anyhow::anyhow!("inner").context("outer"))
        })
        .unwrap_err();
        assert_eq!(format!("{err:#}"), "outer: inner");

        assert!(try_candidates::<()>("m", &[], |_| Ok(())).is_err());
    }

    #[test]
    fn ort_aware_cpu_fallbacks() {
        use crate::backend::detect::{GpuAdapter, GpuVendor};
        use crate::backend::select::{EpStatus, OpenVinoProbe, OrtProbe};
        let apple = GpuAdapter {
            vendor: GpuVendor::Apple,
            name: "Apple M1 GPU".into(),
            vram_mb: 0,
            index: 0,
            discrete: false,
            cuda: None,
        };
        let hw = HardwareInfo::new("macos", "aarch64", vec![apple]);
        let ort = OrtProbe::installed(
            "coreml",
            vec![
                EpStatus::usable(Target::Cpu),
                EpStatus::usable(Target::CoreMl),
            ],
        );
        let names = |sel: &Selection, spec: &str, ov: &[&str]| -> Vec<(String, bool)> {
            plan_for(&parse(spec).unwrap(), sel, &devs(ov))
                .into_iter()
                .map(|c| (c.device.to_string(), c.fell_back))
                .collect()
        };

        // OpenVINO and ONNX Runtime both present.
        let both = select(
            &hw,
            &RuntimeProbe {
                openvino: OpenVinoProbe::with_device_names(&devs(&["CPU"])),
                ort: ort.clone(),
            },
            true,
        );
        assert_eq!(best_cpu(&both), Device::OPENVINO_CPU);
        assert_eq!(
            names(&both, "auto", &["CPU"]),
            s(&[("ort:coreml", false), ("openvino:cpu", true)])
        );
        assert_eq!(
            names(&both, "ort:coreml", &["CPU"]),
            s(&[("ort:coreml", false), ("openvino:cpu", true)])
        );
        assert_eq!(
            names(&both, "ort:cpu", &["CPU"]),
            s(&[("ort:cpu", false), ("openvino:cpu", true)])
        );
        assert_eq!(names(&both, "CPU", &["CPU"]), s(&[("openvino:cpu", false)]));

        // ONNX Runtime only: CPU options and "no GPU" fallbacks land on ort:cpu.
        let ort_only = select(
            &hw,
            &RuntimeProbe {
                openvino: OpenVinoProbe::unavailable("missing"),
                ort,
            },
            true,
        );
        assert_eq!(
            best_cpu(&ort_only),
            parse("ort:cpu").unwrap().device().copied().unwrap()
        );
        assert_eq!(
            names(&ort_only, "auto", &[]),
            s(&[("ort:coreml", false), ("ort:cpu", true)])
        );
        let legacy_gpu = plan_for(&parse("GPU").unwrap(), &ort_only, &[]);
        assert_eq!(legacy_gpu.len(), 1);
        assert_eq!(legacy_gpu[0].device.to_string(), "ort:cpu");
        assert!(legacy_gpu[0].fell_back);
        assert!(
            legacy_gpu[0]
                .note
                .as_deref()
                .unwrap()
                .contains("using ort:cpu")
        );
        assert_eq!(
            names(&ort_only, "CPU", &[]),
            s(&[("openvino:cpu", false), ("ort:cpu", true)])
        );
        assert_eq!(
            names(&ort_only, "ort:cuda", &[]),
            s(&[("ort:cuda", false), ("ort:cpu", true)])
        );
        // ort:cpu itself: OpenVINO CPU is still tried so the error names it.
        assert_eq!(
            names(&ort_only, "ort:cpu", &[]),
            s(&[("ort:cpu", false), ("openvino:cpu", true)])
        );
        assert!(ort_only.any_gpu_runnable());
    }
}
