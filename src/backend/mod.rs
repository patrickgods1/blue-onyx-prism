//! Inference backends behind one runtime-neutral interface.
//!
//! Two runtimes: OpenVINO (always built) and ONNX Runtime ([`ort`], Cargo feature `onnxruntime`,
//! on by default). [`Backend`] and [`Compiled`] have one variant per runtime (docs/PLAN.md,
//! "Multi-runtime backend"). [`detect`] finds the GPUs without any runtime library, [`select`]
//! ranks the device options for `auto`, and [`Runtimes`] initializes each runtime once at startup
//! (a runtime that fails to initialize only marks its options unavailable).
//!
//! Usage (from `registry`/`worker`/benchmark):
//!
//! ```ignore
//! let mut rt = Runtimes::new_with(&CoreOptions { cache_dir, intra_threads, openvino_dir },
//!                                 &OrtOptions { onnxruntime_dir });
//! rt.require_any()?;                                   // fails when no runtime initialized
//! let req = LoadRequest { path, requested: "GPU".into(), gpu_precision: None };
//! // Whole plan (GPU, then CPU) in one call, without warm-up:
//! let mut backend: Backend = rt.load(&spec::parse("auto")?, &req)?;
//! // Or per candidate, compiling under the `Runtimes` lock and creating the request elsewhere:
//! for candidate in rt.plan(&spec, &req.path) { .. }       // `auto` = hardware (or benchmark) ranking
//! let compiled: Compiled = rt.compile(&candidate, &req)?;   // `Compiled` is Send
//! let mut backend = compiled.into_backend()?;             // creates the InferRequest
//! let outs: Vec<NamedOutput> = backend.infer(&chw_f32, &extra_inputs)?;
//! // Pipelined (worker): `compile_tuned` with a profile's `DeviceTuning` gives several slots;
//! let seq = backend.submit(slot, &chw_f32, &extra_inputs)?; // infer_async / session thread
//! let done: Completion = backend.completions().recv()?;      // select on it with other work
//! let outs = backend.collect(done.slot)?;                     // ...postprocess...
//! backend.recycle(done.slot, outs);                           // buffers back to the slot
//! ```

pub mod cpu_threads;
pub mod detect;
pub mod device;
#[cfg(test)]
pub mod fake;
#[cfg(windows)]
pub mod job;
pub mod libs;
pub mod memory;
pub mod onnx_rewrite;
#[cfg(feature = "onnxruntime")]
pub mod ort;
pub mod plan;
pub mod priority;
pub mod select;
pub mod spec;

pub use detect::{GpuAdapter, GpuVendor, HardwareInfo};
pub use device::DeviceInfo;
pub use plan::{Candidate, auto_candidates, best_cpu, plan_candidates, plan_for, try_candidates};
pub use select::{DeviceOption, RuntimeProbe, Selection};
pub use spec::{DeviceSpec, Runtime};

use crate::model::{ExtraInput, NamedOutput, PortSpec};
use crate::profile::{DeviceClass, DeviceTuning, PerfHint};
use anyhow::Result;
use spec::Device;
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Options applied once per runtime.
#[derive(Debug, Clone, Default)]
pub struct CoreOptions {
    /// Compiled-model cache directory; None disables caching.
    pub cache_dir: Option<PathBuf>,
    /// CPU inference threads (0 = all physical cores but one, see [`cpu_threads`]).
    pub intra_threads: usize,
    /// Explicit OpenVINO runtime directory (config `openvino_dir`).
    pub openvino_dir: Option<PathBuf>,
}

/// Options for the ONNX Runtime library lookup (see `libs::find_onnxruntime`).
#[derive(Debug, Clone, Default)]
pub struct OrtOptions {
    /// Config `onnxruntime_dir`: a directory holding the library, or the library file. None =
    /// `ORT_DYLIB_PATH`, then `<exe_dir>/onnxruntime`.
    pub onnxruntime_dir: Option<PathBuf>,
    /// Install root searched after the explicit dir and `ORT_DYLIB_PATH` (active flavor, legacy
    /// flat layout, per-flavor dirs). None = `<exe_dir>/onnxruntime`; config `download_dir` sets
    /// it to `<download_dir>/onnxruntime`.
    pub default_dir: Option<PathBuf>,
    /// `onnxruntime/cuda-libs` (`nvidia-cuda-libs`): preloaded before the CUDA provider is used.
    pub cuda_libs_dir: Option<PathBuf>,
}

impl OrtOptions {
    /// Locate the ONNX Runtime library these options select (the same lookup the runtime
    /// loads from, so status pages report the library actually in use).
    pub fn lookup(&self) -> libs::OrtLookup {
        match &self.default_dir {
            Some(d) => libs::find_onnxruntime_with(self.onnxruntime_dir.as_deref(), d),
            None => libs::find_onnxruntime(self.onnxruntime_dir.as_deref()),
        }
    }
}

/// What to load. The device comes separately as a [`DeviceSpec`] / [`Candidate`].
#[derive(Debug, Clone)]
pub struct LoadRequest {
    pub path: PathBuf,
    /// Device as configured ("GPU", "GPU.1", "openvino:cpu", "auto", ...), reported as
    /// [`DeviceInfo::requested`].
    pub requested: String,
    /// "f16" (default on GPU) or "f32".
    pub gpu_precision: Option<String>,
}

/// Runtime-neutral facts about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model file this was loaded from (used for error context).
    pub path: PathBuf,
    pub inputs: Vec<PortSpec>,
    pub outputs: Vec<PortSpec>,
    pub device: DeviceInfo,
    /// Name of the image input tensor (e.g. "images" / "input").
    pub image_input: String,
    /// (width, height) of the image input.
    pub input_size: (u32, u32),
    /// Wall time spent in read+compile, for logging/stats.
    pub compile_ms: u64,
}

/// Facts about the available runtimes, captured once at startup.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RuntimeInfo {
    /// Empty when OpenVINO could not be initialized.
    pub openvino_version: String,
    /// OpenVINO device names ("CPU", "GPU.0", ...).
    pub available_devices: Vec<String>,
    pub has_gpu: bool,
}

/// Owns the OpenVINO `Core`. One per process; models are loaded sequentially through it.
pub struct OvCore {
    core: openvino::Core,
    opts: CoreOptions,
    available: Vec<String>,
}

impl fmt::Debug for OvCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OvCore")
            .field("opts", &self.opts)
            .field("available", &self.available)
            .finish_non_exhaustive()
    }
}

impl OvCore {
    pub fn new(opts: &CoreOptions) -> Result<Self> {
        libs::prepare_environment(opts.openvino_dir.as_deref());
        let core = openvino::Core::new().map_err(|e| {
            anyhow::anyhow!("OpenVINO Core::new failed: {e:?}. {}", libs::diagnostics())
        })?;
        let available = core
            .available_devices()
            .map(|v| v.iter().map(|d| d.to_string()).collect())
            .unwrap_or_default();
        let mut me = Self {
            core,
            opts: opts.clone(),
            available,
        };
        me.apply_core_properties()?;
        Ok(me)
    }

    pub fn available_devices(&self) -> &[String] {
        &self.available
    }

    pub fn has_gpu(&self) -> bool {
        self.available.iter().any(|d| d.starts_with("GPU"))
    }

    pub fn openvino_version(&self) -> String {
        let v = openvino::version();
        format!("{} ({})", v.build_number, v.description)
    }

    fn apply_core_properties(&mut self) -> Result<()> {
        device::apply_core_properties(&mut self.core, &self.opts, &self.available)
    }

    /// Read, (reshape to static), compile on exactly `cand.device` with `tuning`'s performance
    /// hint and introspect ports.
    pub fn compile(
        &mut self,
        cand: &Candidate,
        req: &LoadRequest,
        tuning: &DeviceTuning,
    ) -> Result<OvCompiled> {
        device::load_model(&mut self.core, &self.available, cand, req, tuning)
    }

    /// Human readable device description (FULL_DEVICE_NAME) for a device string.
    pub fn device_full_name(&self, device: &str) -> String {
        device::full_device_name(&self.core, device)
    }

    /// Devices with FULL_DEVICE_NAME and, for GPUs, DEVICE_TYPE ("integrated"/"discrete").
    pub fn probe(&self) -> select::OpenVinoProbe {
        let devices = self
            .available
            .iter()
            .map(|name| {
                let dt = openvino::DeviceType::from(name.as_str());
                let device_type = name.starts_with("GPU").then(|| {
                    self.core
                        .get_property(&dt, &openvino::PropertyKey::Other("DEVICE_TYPE".into()))
                        .ok()
                });
                select::OvDevice {
                    name: name.clone(),
                    full_name: self.device_full_name(name),
                    device_type: device_type.flatten().map(|t| t.trim().to_string()),
                }
            })
            .collect();
        select::OpenVinoProbe {
            error: None,
            devices,
        }
    }

    pub fn inner_mut(&mut self) -> &mut openvino::Core {
        &mut self.core
    }
}

/// An OpenVINO compiled model plus its metadata. `CompiledModel` is `Send`, so this can be moved
/// into a worker thread; the `InferRequest`s are created there ([`OvBackend::new`]).
pub struct OvCompiled {
    pub(crate) compiled: openvino::CompiledModel,
    pub info: ModelInfo,
    /// Hint it was compiled with and the in-flight count wanted.
    pub(crate) tuning: DeviceTuning,
    /// `NUM_STREAMS` of the compiled model (None when not readable).
    pub(crate) streams: Option<usize>,
}

/// A finished asynchronous inference: which slot, its sequence number (from
/// [`Backend::submit`]) and when it finished. Stale ones (e.g. of a synchronous
/// [`Backend::infer`]) carry an older sequence number and are ignored by the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completion {
    pub slot: usize,
    pub seq: u64,
    pub at: std::time::Instant,
}

/// Per-worker OpenVINO inference state: one `InferRequest` per slot, all from one compiled
/// model, each with a completion callback feeding [`Backend::completions`].
pub struct OvBackend {
    pub(crate) info: ModelInfo,
    /// Dropped (after `Drop` aborted what is in flight) before the compiled model.
    pub(crate) slots: Vec<device::OvSlot>,
    pub(crate) done_rx: crossbeam_channel::Receiver<Completion>,
    pub(crate) tuning: DeviceTuning,
    pub(crate) streams: Option<usize>,
    /// Kept alive for as long as its infer requests.
    pub(crate) _compiled: openvino::CompiledModel,
}

impl fmt::Debug for OvCompiled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OvCompiled")
            .field("info", &self.info)
            .field("tuning", &self.tuning)
            .field("streams", &self.streams)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for OvBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OvBackend")
            .field("info", &self.info)
            .field("slots", &self.slots.len())
            .field("tuning", &self.tuning)
            .field("streams", &self.streams)
            .finish_non_exhaustive()
    }
}

impl OvBackend {
    pub fn new(model: OvCompiled) -> Result<Self> {
        device::create_backend(model)
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Run one inference. `chw` must be `3 * w * h` f32 values in CHW order.
    pub fn infer(&mut self, chw: &[f32], extra: &[ExtraInput]) -> Result<Vec<NamedOutput>> {
        let mut outs = Vec::new();
        self.infer_into(chw, extra, &mut outs)?;
        Ok(outs)
    }

    /// [`Self::infer`] into `outs`, reusing its buffers (slot 0, synchronously).
    pub fn infer_into(
        &mut self,
        chw: &[f32],
        extra: &[ExtraInput],
        outs: &mut Vec<NamedOutput>,
    ) -> Result<()> {
        // Completions of earlier synchronous calls are never consumed; drop them.
        while self.done_rx.try_recv().is_ok() {}
        device::recycle(&mut self.slots, 0, std::mem::take(outs));
        device::submit(&self.info, &mut self.slots, 0, chw, extra)?;
        *outs = device::collect(&self.info, &mut self.slots, 0)?;
        Ok(())
    }
}

impl Drop for OvBackend {
    fn drop(&mut self) {
        // Never free a request (or its callback context) with an inference in flight.
        device::abort(&mut self.slots);
    }
}

/// A compiled model of any runtime, before its inference request exists. `Send`.
#[derive(Debug)]
pub enum Compiled {
    OpenVino(OvCompiled),
    #[cfg(feature = "onnxruntime")]
    Ort(self::ort::OrtCompiled),
    /// Scripted test backend (see [`fake`]).
    #[cfg(test)]
    Fake(fake::FakeCompiled),
}

impl Compiled {
    pub fn info(&self) -> &ModelInfo {
        match self {
            Compiled::OpenVino(c) => &c.info,
            #[cfg(feature = "onnxruntime")]
            Compiled::Ort(c) => &c.info,
            #[cfg(test)]
            Compiled::Fake(c) => &c.info,
        }
    }

    /// Create the per-thread inference state. Call on the thread that will run inference.
    pub fn into_backend(self) -> Result<Backend> {
        match self {
            Compiled::OpenVino(c) => Ok(Backend::OpenVino(OvBackend::new(c)?)),
            #[cfg(feature = "onnxruntime")]
            Compiled::Ort(c) => Ok(Backend::Ort(c.into_backend()?)),
            #[cfg(test)]
            Compiled::Fake(c) => Ok(Backend::Fake(c.into_backend())),
        }
    }
}

/// Inference state of one model on one worker thread: [`Self::slots`] inferences can be in
/// flight at once ([`Self::submit`] / [`Self::collect`]); [`Self::infer`] runs one
/// synchronously on slot 0.
#[derive(Debug)]
pub enum Backend {
    OpenVino(OvBackend),
    #[cfg(feature = "onnxruntime")]
    Ort(self::ort::OrtBackend),
    /// Scripted test backend (see [`fake`]).
    #[cfg(test)]
    Fake(fake::FakeBackend),
}

impl Backend {
    pub fn info(&self) -> &ModelInfo {
        match self {
            Backend::OpenVino(b) => b.info(),
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.info(),
            #[cfg(test)]
            Backend::Fake(b) => b.info(),
        }
    }

    /// Number of slots (concurrent inferences): OpenVINO infer requests, ONNX Runtime sessions.
    pub fn slots(&self) -> usize {
        match self {
            Backend::OpenVino(b) => b.slots.len(),
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.slots(),
            #[cfg(test)]
            Backend::Fake(b) => b.slots(),
        }
    }

    /// Hint the model was compiled with and `NUM_STREAMS` (OpenVINO; None for ONNX Runtime or
    /// when not readable).
    pub fn hint_and_streams(&self) -> (PerfHint, Option<usize>) {
        match self {
            Backend::OpenVino(b) => (b.tuning.hint, b.streams),
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(_) => (PerfHint::Latency, None),
            #[cfg(test)]
            Backend::Fake(_) => (PerfHint::Latency, None),
        }
    }

    /// Start an inference on the free slot `slot`. `chw` (`3 * w * h` f32 values in CHW
    /// order) and `extra` are copied, so they can be reused at once. Returns the sequence
    /// number its [`Completion`] carries.
    pub fn submit(&mut self, slot: usize, chw: &[f32], extra: &[ExtraInput]) -> Result<u64> {
        match self {
            Backend::OpenVino(b) => device::submit(&b.info, &mut b.slots, slot, chw, extra),
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.submit(slot, chw, extra),
            #[cfg(test)]
            Backend::Fake(b) => b.submit(slot, chw, extra),
        }
    }

    /// Completions of submitted inferences (select on it; then [`Self::collect`] the slot).
    pub fn completions(&self) -> &crossbeam_channel::Receiver<Completion> {
        match self {
            Backend::OpenVino(b) => &b.done_rx,
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.completions(),
            #[cfg(test)]
            Backend::Fake(b) => b.completions(),
        }
    }

    /// True while `slot` has an inference in flight (submitted, not collected).
    pub fn is_busy(&self, slot: usize) -> bool {
        match self {
            Backend::OpenVino(b) => device::is_busy(&b.slots, slot),
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.is_busy(slot),
            #[cfg(test)]
            Backend::Fake(b) => b.is_busy(slot),
        }
    }

    /// The outputs of `slot`'s inference, waiting for it when it has not finished. The buffers
    /// belong to the slot: hand them back with [`Self::recycle`] so the next inference reuses
    /// them.
    pub fn collect(&mut self, slot: usize) -> Result<Vec<NamedOutput>> {
        match self {
            Backend::OpenVino(b) => device::collect(&b.info, &mut b.slots, slot),
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.collect(slot),
            #[cfg(test)]
            Backend::Fake(b) => b.collect(slot),
        }
    }

    /// Return output buffers lent by [`Self::collect`] to `slot`.
    pub fn recycle(&mut self, slot: usize, outs: Vec<NamedOutput>) {
        match self {
            Backend::OpenVino(b) => device::recycle(&mut b.slots, slot, outs),
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.recycle(slot, outs),
            #[cfg(test)]
            Backend::Fake(b) => b.recycle(slot, outs),
        }
    }

    /// False when a slot can no longer run inferences and the model must be reloaded (an ONNX
    /// Runtime session thread died or hung; see [`SessionDead`]). OpenVINO is always healthy
    /// here: its errors are per request.
    pub fn is_healthy(&self) -> bool {
        match self {
            Backend::OpenVino(_) => true,
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.is_healthy(),
            #[cfg(test)]
            Backend::Fake(b) => b.is_healthy(),
        }
    }

    /// Stop every inference in flight (OpenVINO: cancel and wait; ONNX Runtime: wait at most
    /// `ort::ABORT_TIMEOUT`, a running session cannot be interrupted; a hung one is marked dead).
    /// Their results are discarded.
    pub fn abort(&mut self) {
        match self {
            Backend::OpenVino(b) => device::abort(&mut b.slots),
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.abort(),
            #[cfg(test)]
            Backend::Fake(b) => b.abort(),
        }
    }

    /// Run one inference synchronously (slot 0: submit + collect). `chw` must be `3 * w * h`
    /// f32 values in CHW order.
    pub fn infer(&mut self, chw: &[f32], extra: &[ExtraInput]) -> Result<Vec<NamedOutput>> {
        let mut outs = Vec::new();
        self.infer_into(chw, extra, &mut outs)?;
        Ok(outs)
    }

    /// [`Self::infer`] into `outs`, reusing its buffers (one entry per model output).
    pub fn infer_into(
        &mut self,
        chw: &[f32],
        extra: &[ExtraInput],
        outs: &mut Vec<NamedOutput>,
    ) -> Result<()> {
        match self {
            Backend::OpenVino(b) => b.infer_into(chw, extra, outs),
            #[cfg(feature = "onnxruntime")]
            Backend::Ort(b) => b.infer_into(chw, extra, outs),
            #[cfg(test)]
            Backend::Fake(b) => b.infer_into(chw, extra, outs),
        }
    }
}

/// A backend slot whose runtime thread is gone: an ONNX Runtime session thread panicked,
/// exited, or stayed hung in `Session::run` past `ort::ABORT_TIMEOUT` (BE-10). The slot answers
/// every later `submit` with this error; only reloading the model brings it back. Detect it
/// with [`is_session_dead`] (it survives `anyhow` context) or poll [`Backend::is_healthy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionDead {
    pub slot: usize,
    /// What happened, e.g. "session thread 0 panicked during inference (...)".
    pub reason: String,
}

impl std::fmt::Display for SessionDead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}; the inference runtime is in an unknown state; reload the model (the worker \
             does this automatically) or restart the service",
            self.reason
        )
    }
}

impl std::error::Error for SessionDead {}

/// Whether `e` (anywhere in its chain) is a [`SessionDead`]: the model must be reloaded.
pub fn is_session_dead(e: &anyhow::Error) -> bool {
    e.chain().any(|c| c.is::<SessionDead>())
}

/// Reason given to ONNX Runtime options in a build without the `onnxruntime` feature.
pub const ORT_UNAVAILABLE: &str =
    "ONNX Runtime support is not compiled into this build (Cargo feature `onnxruntime`)";

/// Short reason for ONNX Runtime options when the library is not installed (the full lookup is
/// in [`Runtimes::onnxruntime_error`]).
pub const ORT_NOT_INSTALLED: &str = "ONNX Runtime library not found (run `setup-onnxruntime`)";

/// Short reason shown for OpenVINO options when the runtime failed to initialize (the full
/// error is in [`Runtimes::openvino_error`]).
pub const OV_UNAVAILABLE: &str = "OpenVINO runtime not loadable (run `setup-openvino`)";

/// Every runtime this process can use. Shared behind a mutex; models compile one at a time.
#[derive(Debug)]
pub struct Runtimes {
    ov: Option<OvCore>,
    /// Why OpenVINO could not be initialized (`{e:#}`).
    ov_error: Option<String>,
    #[cfg(feature = "onnxruntime")]
    ort: Option<self::ort::OrtRuntime>,
    /// Why ONNX Runtime could not be initialized (`{e:#}`).
    ort_error: Option<String>,
    /// Detected once per process ([`detect::hardware`]).
    hw: HardwareInfo,
    /// What each runtime offers, captured at construction.
    probe: RuntimeProbe,
    /// Measured `auto` order per model path (benchmark speed order, fastest first), see
    /// [`Self::set_measured_orders`].
    measured: HashMap<PathBuf, Vec<Device>>,
    /// Test runtime: every plan is one `openvino:cpu` candidate compiled by this script.
    #[cfg(test)]
    fake: Option<fake::FakeScript>,
}

impl Runtimes {
    /// Initialize every runtime and probe its devices, looking for ONNX Runtime in
    /// `ORT_DYLIB_PATH` and `<exe_dir>/onnxruntime`. See [`Self::new_with`].
    pub fn new(opts: &CoreOptions) -> Self {
        Self::new_with(opts, &OrtOptions::default())
    }

    /// Initialize every runtime and probe its devices. Failures are kept (see
    /// [`Self::require_any`]), not returned: a runtime that fails only marks its options
    /// unavailable.
    pub fn new_with(opts: &CoreOptions, ort_opts: &OrtOptions) -> Self {
        let hw = detect::hardware().clone();
        tracing::info!(
            "CPU inference threads: {}",
            cpu_threads::describe(opts.intra_threads)
        );
        let (ov, ov_error) = match OvCore::new(opts) {
            Ok(core) => (Some(core), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        };
        let openvino = match &ov {
            Some(core) => core.probe(),
            None => select::OpenVinoProbe::unavailable(OV_UNAVAILABLE),
        };

        #[cfg(feature = "onnxruntime")]
        let (ort, ort_error, ort_probe) = {
            let lookup = ort_opts.lookup();
            match &lookup.library {
                None => (
                    None,
                    Some(lookup.not_found_message()),
                    select::OrtProbe::unavailable(ORT_NOT_INSTALLED),
                ),
                Some(path) => match self::ort::OrtRuntime::new(
                    path,
                    opts,
                    &hw,
                    ort_opts.cuda_libs_dir.as_deref(),
                ) {
                    Ok(rt) => {
                        let probe = rt.probe().clone();
                        (Some(rt), None, probe)
                    }
                    Err(e) => {
                        let msg = format!("{e:#}");
                        tracing::warn!(
                            "ONNX Runtime unavailable: {msg}; the installed library is \
                             probably damaged or of an unsupported version; models fall back to \
                             OpenVINO; run `setup-onnxruntime` to reinstall it"
                        );
                        (
                            None,
                            Some(msg.clone()),
                            select::OrtProbe::unavailable(format!(
                                "ONNX Runtime failed to load: {msg}"
                            )),
                        )
                    }
                },
            }
        };
        #[cfg(not(feature = "onnxruntime"))]
        let (ort_error, ort_probe) = {
            let _ = ort_opts;
            (
                Some(ORT_UNAVAILABLE.to_string()),
                select::OrtProbe::not_in_build(),
            )
        };
        if let Some(e) = &ort_error {
            tracing::debug!("ONNX Runtime not initialized: {e}");
        }

        let probe = RuntimeProbe {
            openvino,
            ort: ort_probe,
        };
        tracing::debug!(?hw, ?probe, "runtimes probed");
        Self {
            ov,
            ov_error,
            #[cfg(feature = "onnxruntime")]
            ort,
            ort_error,
            hw,
            probe,
            measured: HashMap::new(),
            #[cfg(test)]
            fake: None,
        }
    }

    /// A runtime set without OpenVINO or ONNX Runtime whose every plan is one `openvino:cpu`
    /// candidate that "compiles" into a [`fake::FakeBackend`] played by `script`.
    #[cfg(test)]
    pub(crate) fn fake(script: fake::FakeScript) -> Self {
        Self {
            ov: None,
            ov_error: Some("fake runtime".into()),
            #[cfg(feature = "onnxruntime")]
            ort: None,
            ort_error: Some("fake runtime".into()),
            hw: HardwareInfo::default(),
            probe: RuntimeProbe::default(),
            measured: HashMap::new(),
            fake: Some(script),
        }
    }

    pub fn hardware(&self) -> &HardwareInfo {
        &self.hw
    }

    pub fn probe(&self) -> &RuntimeProbe {
        &self.probe
    }

    /// Device options and the `auto` ranking for `model` (None: any model, assuming an `.onnx`
    /// exists for ONNX Runtime). A model with a measured order ([`Self::set_measured_orders`])
    /// has its `auto` ranking reordered by it.
    pub fn selection(&self, model: Option<&Path>) -> Selection {
        let has_onnx = model.is_none_or(|m| select::onnx_path_for(m).is_some());
        let mut sel = select::select(&self.hw, &self.probe, has_onnx);
        if let Some(order) = model.and_then(|m| self.measured.get(m)) {
            sel.apply_measured_order(order);
        }
        sel
    }

    /// Per-model measured `auto` orders (model path as passed to [`Self::plan`] -> devices,
    /// fastest first; from `benchmark.json`, see `benchmark::auto_orders`). Replaces any earlier
    /// orders; explicit device specs are unaffected.
    pub fn set_measured_orders(&mut self, orders: HashMap<PathBuf, Vec<Device>>) {
        self.measured = orders;
    }

    /// The measured `auto` orders set with [`Self::set_measured_orders`].
    pub fn measured_orders(&self) -> &HashMap<PathBuf, Vec<Device>> {
        &self.measured
    }

    /// Ok when at least one runtime is usable; otherwise the initialization errors of every
    /// runtime, under the context "initializing OpenVINO" (the primary runtime).
    pub fn require_any(&self) -> Result<()> {
        if self.ov.is_some() || self.has_ort() {
            return Ok(());
        }
        let err = self
            .ov_error
            .as_deref()
            .unwrap_or("OpenVINO was not initialized");
        match &self.ort_error {
            Some(ort) if cfg!(feature = "onnxruntime") => Err(anyhow::anyhow!(
                "{err}; ONNX Runtime is not available either: {ort}"
            )
            .context("initializing OpenVINO")),
            _ => Err(anyhow::anyhow!("{err}").context("initializing OpenVINO")),
        }
    }

    /// ONNX Runtime was loaded.
    pub fn has_ort(&self) -> bool {
        #[cfg(feature = "onnxruntime")]
        {
            self.ort.is_some()
        }
        #[cfg(not(feature = "onnxruntime"))]
        {
            false
        }
    }

    #[cfg(feature = "onnxruntime")]
    pub fn onnxruntime(&self) -> Option<&self::ort::OrtRuntime> {
        self.ort.as_ref()
    }

    /// Version of the loaded ONNX Runtime library ("1.24.4"); None when not loaded.
    pub fn onnxruntime_version(&self) -> Option<String> {
        #[cfg(feature = "onnxruntime")]
        {
            self.ort.as_ref().map(|r| r.version().to_string())
        }
        #[cfg(not(feature = "onnxruntime"))]
        {
            None
        }
    }

    /// Why ONNX Runtime could not be initialized (lookup details or load error).
    pub fn onnxruntime_error(&self) -> Option<&str> {
        self.ort_error.as_deref()
    }

    /// The device `force_cpu` uses: OpenVINO CPU, or ONNX Runtime CPU when OpenVINO is missing.
    pub fn best_cpu(&self) -> DeviceSpec {
        DeviceSpec::Device(best_cpu(&self.selection(None)))
    }

    pub fn openvino(&self) -> Option<&OvCore> {
        self.ov.as_ref()
    }

    pub fn openvino_mut(&mut self) -> Option<&mut OvCore> {
        self.ov.as_mut()
    }

    pub fn openvino_error(&self) -> Option<&str> {
        self.ov_error.as_deref()
    }

    /// OpenVINO device names, empty without OpenVINO.
    pub fn available_ov_devices(&self) -> &[String] {
        self.ov
            .as_ref()
            .map(OvCore::available_devices)
            .unwrap_or(&[])
    }

    /// Runtime facts for the API: `has_gpu` (`canUseGPU`) is true when any GPU-class option of
    /// any runtime can run.
    pub fn info(&self) -> RuntimeInfo {
        let has_gpu = self.selection(None).any_gpu_runnable();
        match &self.ov {
            Some(core) => RuntimeInfo {
                openvino_version: core.openvino_version(),
                available_devices: core.available_devices().to_vec(),
                has_gpu,
            },
            None => RuntimeInfo {
                has_gpu,
                ..RuntimeInfo::default()
            },
        }
    }

    /// Candidates for `spec` and the model at `model` on this machine, in the order they should
    /// be tried.
    pub fn plan(&self, spec: &DeviceSpec, model: &Path) -> Vec<Candidate> {
        #[cfg(test)]
        if self.fake.is_some() {
            return vec![Candidate {
                device: Device::new(Runtime::OpenVino, spec::Target::Cpu, None)
                    .unwrap_or_else(|e| panic!("{e:#}")),
                fell_back: false,
                note: None,
            }];
        }
        match spec {
            DeviceSpec::Auto => auto_candidates(&self.selection(Some(model))),
            DeviceSpec::Device(_) => plan_for(
                spec,
                &self.selection(Some(model)),
                self.available_ov_devices(),
            ),
        }
    }

    /// The first runnable load candidate of `spec` for `model` (where the model is headed) and
    /// its option label ("ONNX Runtime CoreML (Apple M1)"); None when nothing can run.
    pub fn first_runnable(&self, spec: &DeviceSpec, model: &Path) -> Option<(Device, String)> {
        let sel = self.selection(Some(model));
        self.plan(spec, model).into_iter().find_map(|c| {
            sel.option(&c.device)
                .filter(|o| o.runnable)
                .map(|o| (c.device, o.label.clone()))
        })
    }

    /// Compile on exactly `cand.device` (no fallback) for serial use: LATENCY hint, one infer
    /// request / session ([`DeviceTuning::SERIAL`]).
    pub fn compile(&mut self, cand: &Candidate, req: &LoadRequest) -> Result<Compiled> {
        self.compile_tuned(cand, req, &DeviceTuning::SERIAL)
    }

    /// Compile on exactly `cand.device` (no fallback) with `tuning`: OpenVINO's performance
    /// hint and in-flight count (clamped to `NUM_STREAMS` when asked), ONNX Runtime's sessions.
    pub fn compile_tuned(
        &mut self,
        cand: &Candidate,
        req: &LoadRequest,
        tuning: &DeviceTuning,
    ) -> Result<Compiled> {
        #[cfg(test)]
        if let Some(script) = &self.fake {
            let device = cand.device.to_string();
            return Ok(Compiled::Fake(script.compile(req, &device, cand.fell_back)));
        }
        match cand.device.runtime {
            Runtime::OpenVino => {
                let Some(core) = self.ov.as_mut() else {
                    anyhow::bail!(
                        "OpenVINO is not available ({}); its runtime libraries are probably \
                         missing or of another version; run `setup-openvino` or pick an `ort:*` \
                         device",
                        self.ov_error.as_deref().unwrap_or("not initialized")
                    );
                };
                Ok(Compiled::OpenVino(core.compile(cand, req, tuning)?))
            }
            #[cfg(feature = "onnxruntime")]
            Runtime::Ort => {
                let Some(rt) = self.ort.as_ref() else {
                    anyhow::bail!(
                        "ONNX Runtime is not available ({}); its library is probably not \
                         installed; run `setup-onnxruntime` or pick an `openvino:*` device",
                        self.ort_error.as_deref().unwrap_or("not initialized")
                    );
                };
                Ok(Compiled::Ort(rt.compile(cand, req, tuning.ort_sessions)?))
            }
            #[cfg(not(feature = "onnxruntime"))]
            Runtime::Ort => {
                let _ = tuning;
                anyhow::bail!("{ORT_UNAVAILABLE} ({})", cand.device)
            }
        }
    }

    /// The concurrency class of `device` ([`crate::profile::Profile::tuning_for`]): an
    /// OpenVINO GPU is discrete when OpenVINO reports `DEVICE_TYPE=discrete` for the GPU the
    /// spec compiles on (a bare `GPU` resolves like `load_model`, see `ov_gpu_for`).
    pub fn device_class(&self, device: &Device) -> DeviceClass {
        match (device.runtime, device.target) {
            (Runtime::OpenVino, spec::Target::Gpu) => {
                if ov_gpu_for(&self.probe.openvino.devices, device.index)
                    .is_some_and(|d| d.is_discrete())
                {
                    DeviceClass::DiscreteGpu
                } else {
                    DeviceClass::IntegratedGpu
                }
            }
            (Runtime::OpenVino, spec::Target::Npu) => DeviceClass::Npu,
            (Runtime::OpenVino, _) => DeviceClass::Cpu,
            (Runtime::Ort, t) => DeviceClass::Ort {
                gpu: t != spec::Target::Cpu,
            },
        }
    }

    /// Resolve `spec` and load the first candidate that compiles and creates its request
    /// (no warm-up; the worker additionally warms up each candidate).
    pub fn load(&mut self, spec: &DeviceSpec, req: &LoadRequest) -> Result<Backend> {
        let candidates = self.plan(spec, &req.path);
        let label = format!("model {}", req.path.display());
        try_candidates(&label, &candidates, |cand| {
            self.compile(cand, req)?.into_backend()
        })
    }
}

/// The probed OpenVINO GPU `openvino:gpu[.N]` compiles on: bare `GPU` via
/// [`select::bare_gpu_device`] (the Intel preference of `load_model`), `GPU.N` by name (`GPU`
/// for N = 0 when only one unnumbered GPU is listed). None when it is not listed.
fn ov_gpu_for(devices: &[select::OvDevice], index: Option<u32>) -> Option<&select::OvDevice> {
    match index {
        None => select::bare_gpu_device(devices),
        Some(n) => {
            let want = format!("GPU.{n}");
            devices
                .iter()
                .find(|d| d.name.eq_ignore_ascii_case(&want))
                .or_else(|| {
                    (n == 0)
                        .then(|| devices.iter().find(|d| d.name == "GPU"))
                        .flatten()
                })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ovd(name: &str, full: &str, ty: &str) -> select::OvDevice {
        select::OvDevice {
            name: name.into(),
            full_name: full.into(),
            device_type: Some(ty.into()),
        }
    }

    #[test]
    fn session_dead_is_detectable_through_context() {
        let e: anyhow::Error = SessionDead {
            slot: 1,
            reason: "session thread 1 exited".into(),
        }
        .into();
        let e = e.context("model m.onnx on ort:cuda");
        assert!(is_session_dead(&e));
        let msg = format!("{e:#}");
        assert!(
            msg.contains("session thread 1 exited") && msg.contains("reload"),
            "{msg}"
        );
        assert!(!is_session_dead(&anyhow::anyhow!("inference failed")));
    }

    #[test]
    fn device_class_follows_the_compiled_gpu() {
        // OpenCL lists the NVIDIA card first; a bare GPU compiles on the Intel iGPU (BE-09).
        let devs = [
            ovd("CPU", "Intel(R) Core(TM) i5-8500", ""),
            ovd("GPU.0", "NVIDIA GeForce RTX 3060 (dGPU)", "discrete"),
            ovd("GPU.1", "Intel(R) UHD Graphics 630 (iGPU)", "integrated"),
        ];
        assert_eq!(ov_gpu_for(&devs, None).unwrap().name, "GPU.1");
        assert!(!ov_gpu_for(&devs, None).unwrap().is_discrete());
        assert!(ov_gpu_for(&devs, Some(0)).unwrap().is_discrete());
        assert_eq!(ov_gpu_for(&devs, Some(1)).unwrap().name, "GPU.1");
        assert!(ov_gpu_for(&devs, Some(2)).is_none());
        let single = [ovd("GPU", "Intel(R) Arc(TM) A770 (dGPU)", "discrete")];
        assert_eq!(ov_gpu_for(&single, Some(0)).unwrap().name, "GPU");
        assert!(ov_gpu_for(&single, Some(1)).is_none());
    }
}
