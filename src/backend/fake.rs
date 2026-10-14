//! Scriptable in-process backend for tests (`cfg(test)` only): `Backend::Fake` /
//! `Compiled::Fake`, loaded through `Runtimes::fake` exactly like a real runtime, so
//! `worker::run` can be driven without OpenVINO or ONNX Runtime.
//!
//! Each `submit` takes the next [`Step`] of the shared [`FakeScript`] (a FIFO queue) and every
//! backend call is appended to the script's call log ([`Call`]). Completions are delivered on
//! the backend's channel like the real runtimes do (from another thread, after the step's delay).

use super::{Completion, DeviceInfo, LoadRequest, ModelInfo, SessionDead};
use crate::model::{ExtraInput, NamedOutput, OutputBuf, PortElem, PortSpec};
use crate::sync::LockExt;
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Longest a [`Step::Hang`] blocks `collect` when the test never releases it (keeps a broken
/// test from hanging the suite).
const HANG_LIMIT: Duration = Duration::from_secs(30);

/// What one submitted inference does.
#[derive(Debug, Clone)]
pub(crate) enum Step {
    /// Completes after the delay; `collect` returns these outputs.
    Ok(Vec<NamedOutput>, Duration),
    /// Completes at once; `collect` returns this error (a per-request runtime error).
    Fail(String),
    /// Never completes until [`FakeScript::release`]; `collect` blocks until then (at most
    /// [`HANG_LIMIT`]). `abort` drops it like a cancelled request.
    Hang,
    /// The runtime thread panicked during the inference (what the ONNX Runtime backend reports
    /// for a panicking session thread): completes at once, `collect` returns [`SessionDead`] and
    /// the slot stays dead (`is_healthy` false, later submits fail).
    Panic,
}

impl Step {
    /// A YOLO26 end-to-end output `[1, N, 6]` with rows `x1, y1, x2, y2, score, class` (model
    /// input pixels), completing after `delay`.
    pub(crate) fn detections(rows: &[[f32; 6]], delay: Duration) -> Self {
        Step::Ok(vec![yolo26_output(rows)], delay)
    }
}

/// The `[1, N, 6]` output tensor of a YOLO26 model holding `rows`.
pub(crate) fn yolo26_output(rows: &[[f32; 6]]) -> NamedOutput {
    NamedOutput {
        name: "output0".into(),
        shape: vec![1, rows.len(), 6],
        data: OutputBuf::F32(rows.iter().flatten().copied().collect()),
    }
}

/// One backend call, in the order it happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Call {
    /// Synchronous `infer` (the worker's timed warm-up on slot 0).
    Infer,
    /// `submit` of a warm-up of the other slots (does not consume a step).
    WarmupSubmit {
        slot: usize,
    },
    /// `submit` of a request: it took the next step.
    Submit {
        slot: usize,
        seq: u64,
    },
    Collect {
        slot: usize,
    },
    Abort,
}

#[derive(Debug, Default)]
struct ScriptState {
    steps: VecDeque<Step>,
    calls: Vec<Call>,
    /// Set by [`FakeScript::release`]: what hung inferences return.
    released: Option<Result<Vec<NamedOutput>, String>>,
    /// Hung inferences waiting for a release: their completion channel, slot and sequence.
    hung: Vec<(Sender<Completion>, usize, u64)>,
}

#[derive(Debug, Default)]
struct Shared {
    state: Mutex<ScriptState>,
    cv: Condvar,
}

/// The test's handle on the fake: steps to play, the call log, hung inferences to release.
/// Cloning shares the script.
#[derive(Debug, Clone)]
pub(crate) struct FakeScript {
    shared: Arc<Shared>,
    /// Concurrent inferences ([`super::Backend::slots`]).
    pub(crate) slots: usize,
    /// (width, height) of the image input.
    pub(crate) input_size: (u32, u32),
    /// Warm-up submits never complete (a hung device at load time).
    pub(crate) warmup_hangs: bool,
}

impl FakeScript {
    /// A YOLO26-shaped model (`images [1,3,h,w]` -> `output0 [1,300,6]`) with `slots` slots.
    pub(crate) fn new(slots: usize, input_size: (u32, u32)) -> Self {
        Self {
            shared: Arc::default(),
            slots: slots.max(1),
            input_size,
            warmup_hangs: false,
        }
    }

    /// Queue the behavior of the next submitted inferences, in order.
    pub(crate) fn push(&self, steps: impl IntoIterator<Item = Step>) {
        self.shared.state.lock_recover().steps.extend(steps);
    }

    /// Every backend call so far.
    pub(crate) fn calls(&self) -> Vec<Call> {
        self.shared.state.lock_recover().calls.clone()
    }

    /// The request submits so far (warm-up excluded).
    pub(crate) fn submits(&self) -> usize {
        self.calls()
            .iter()
            .filter(|c| matches!(c, Call::Submit { .. }))
            .count()
    }

    /// Steps not taken yet.
    pub(crate) fn remaining(&self) -> usize {
        self.shared.state.lock_recover().steps.len()
    }

    /// Let every hung inference (now and later) finish with `outcome`: their completions are
    /// sent and a blocked `collect` returns.
    pub(crate) fn release(&self, outcome: Result<Vec<NamedOutput>, String>) {
        let mut st = self.shared.state.lock_recover();
        st.released = Some(outcome);
        for (tx, slot, seq) in st.hung.drain(..) {
            let _ = tx.send(Completion {
                slot,
                seq,
                at: Instant::now(),
            });
        }
        self.shared.cv.notify_all();
    }

    fn log(&self, call: Call) {
        self.shared.state.lock_recover().calls.push(call);
    }

    fn ports(&self) -> (Vec<PortSpec>, Vec<PortSpec>) {
        let (w, h) = self.input_size;
        let inputs = vec![PortSpec {
            name: "images".into(),
            shape: vec![1, 3, i64::from(h), i64::from(w)],
            elem: PortElem::F32,
        }];
        let outputs = vec![PortSpec {
            name: "output0".into(),
            shape: vec![1, 300, 6],
            elem: PortElem::F32,
        }];
        (inputs, outputs)
    }

    /// "Compile" for `req` on `device` (what `Runtimes::compile_tuned` returns in a fake runtime).
    pub(crate) fn compile(&self, req: &LoadRequest, device: &str, fell_back: bool) -> FakeCompiled {
        let (inputs, outputs) = self.ports();
        FakeCompiled {
            info: ModelInfo {
                path: req.path.clone(),
                inputs,
                outputs,
                device: DeviceInfo {
                    requested: req.requested.clone(),
                    actual: "CPU".into(),
                    full_name: "Fake CPU".into(),
                    fell_back,
                    runtime: super::Runtime::OpenVino,
                    spec: device.to_string(),
                },
                image_input: "images".into(),
                input_size: self.input_size,
                compile_ms: 0,
            },
            script: self.clone(),
        }
    }
}

/// A fake compiled model (`Send`, like the real ones).
#[derive(Debug)]
pub struct FakeCompiled {
    pub(crate) info: ModelInfo,
    script: FakeScript,
}

impl FakeCompiled {
    pub(crate) fn into_backend(self) -> FakeBackend {
        let (done_tx, done_rx) = crossbeam_channel::unbounded();
        let slots = self.script.slots;
        FakeBackend {
            info: self.info,
            slots: (0..slots).map(|_| Slot::Idle).collect(),
            // The worker warms every slot up through `submit` (slot 0 first, timed).
            warmup_left: slots,
            script: self.script,
            done_tx,
            done_rx,
            seq: 0,
        }
    }
}

#[derive(Debug)]
enum Slot {
    Idle,
    Busy { step: Step, ready_at: Instant },
    Dead(String),
}

/// The per-worker fake inference state.
#[derive(Debug)]
pub struct FakeBackend {
    info: ModelInfo,
    script: FakeScript,
    slots: Vec<Slot>,
    /// Submits still owed to the warm-up of slots 1.. (they do not consume steps).
    warmup_left: usize,
    done_tx: Sender<Completion>,
    done_rx: Receiver<Completion>,
    seq: u64,
}

impl FakeBackend {
    pub(crate) fn info(&self) -> &ModelInfo {
        &self.info
    }

    pub(crate) fn slots(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn completions(&self) -> &Receiver<Completion> {
        &self.done_rx
    }

    pub(crate) fn is_busy(&self, slot: usize) -> bool {
        matches!(self.slots.get(slot), Some(Slot::Busy { .. }))
    }

    pub(crate) fn is_healthy(&self) -> bool {
        !self.slots.iter().any(|s| matches!(s, Slot::Dead(_)))
    }

    pub(crate) fn submit(
        &mut self,
        slot: usize,
        _chw: &[f32],
        _extra: &[ExtraInput],
    ) -> Result<u64> {
        match self.slots.get(slot) {
            None => anyhow::bail!("fake: no slot {slot}"),
            Some(Slot::Busy { .. }) => anyhow::bail!("fake: slot {slot} is busy"),
            Some(Slot::Dead(reason)) => {
                return Err(SessionDead {
                    slot,
                    reason: reason.clone(),
                }
                .into());
            }
            Some(Slot::Idle) => {}
        }
        self.seq += 1;
        let seq = self.seq;
        let step = if self.warmup_left > 0 {
            self.warmup_left -= 1;
            self.script.log(Call::WarmupSubmit { slot });
            if self.script.warmup_hangs {
                Step::Hang
            } else {
                Step::Ok(vec![yolo26_output(&[])], Duration::ZERO)
            }
        } else {
            let mut st = self.script.shared.state.lock_recover();
            st.calls.push(Call::Submit { slot, seq });
            let step = st
                .steps
                .pop_front()
                .unwrap_or_else(|| Step::Ok(vec![yolo26_output(&[])], Duration::ZERO));
            if matches!(step, Step::Hang) {
                if st.released.is_some() {
                    // Released before it was submitted: completes at once.
                    let _ = self.done_tx.send(Completion {
                        slot,
                        seq,
                        at: Instant::now(),
                    });
                } else {
                    st.hung.push((self.done_tx.clone(), slot, seq));
                }
            }
            step
        };
        let delay = match &step {
            Step::Ok(_, d) => *d,
            Step::Fail(_) | Step::Hang | Step::Panic => Duration::ZERO,
        };
        if !matches!(step, Step::Hang) {
            let tx = self.done_tx.clone();
            let send = move || {
                std::thread::sleep(delay);
                let _ = tx.send(Completion {
                    slot,
                    seq,
                    at: Instant::now(),
                });
            };
            if delay.is_zero() {
                send();
            } else {
                std::thread::spawn(send);
            }
        }
        if let Some(s) = self.slots.get_mut(slot) {
            *s = Slot::Busy {
                step,
                ready_at: Instant::now() + delay,
            };
        }
        Ok(seq)
    }

    pub(crate) fn collect(&mut self, slot: usize) -> Result<Vec<NamedOutput>> {
        self.script.log(Call::Collect { slot });
        let Some(cell) = self.slots.get_mut(slot) else {
            anyhow::bail!("fake: no slot {slot}");
        };
        let taken = std::mem::replace(cell, Slot::Idle);
        let (step, ready_at) = match taken {
            Slot::Busy { step, ready_at, .. } => (step, ready_at),
            Slot::Dead(reason) => {
                *cell = Slot::Dead(reason.clone());
                return Err(SessionDead { slot, reason }.into());
            }
            Slot::Idle => anyhow::bail!("fake: no inference in flight on slot {slot}"),
        };
        match step {
            Step::Ok(outs, _) => {
                std::thread::sleep(ready_at.saturating_duration_since(Instant::now()));
                Ok(outs)
            }
            Step::Fail(msg) => Err(anyhow::anyhow!(msg)),
            Step::Panic => {
                let reason = format!("session thread {slot} panicked during inference (fake)");
                *cell = Slot::Dead(reason.clone());
                Err(SessionDead { slot, reason }.into())
            }
            Step::Hang => {
                let deadline = Instant::now() + HANG_LIMIT;
                let mut st = self.script.shared.state.lock_recover();
                while st.released.is_none() {
                    let left = deadline.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        anyhow::bail!("fake: hung inference never released");
                    }
                    st = self
                        .script
                        .shared
                        .cv
                        .wait_timeout(st, left)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
                st.released
                    .clone()
                    .unwrap_or_else(|| Ok(Vec::new()))
                    .map_err(|e| anyhow::anyhow!(e))
            }
        }
    }

    pub(crate) fn recycle(&mut self, _slot: usize, _outs: Vec<NamedOutput>) {}

    /// Cancel everything in flight (a hung inference is dropped, like a cancelled request).
    pub(crate) fn abort(&mut self) {
        self.script.log(Call::Abort);
        let mut st = self.script.shared.state.lock_recover();
        st.hung.clear();
        drop(st);
        for s in &mut self.slots {
            if matches!(s, Slot::Busy { .. }) {
                *s = Slot::Idle;
            }
        }
    }

    pub(crate) fn infer_into(
        &mut self,
        _chw: &[f32],
        _extra: &[ExtraInput],
        outs: &mut Vec<NamedOutput>,
    ) -> Result<()> {
        self.script.log(Call::Infer);
        *outs = vec![yolo26_output(&[])];
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn backend(slots: usize) -> (FakeScript, FakeBackend) {
        let script = FakeScript::new(slots, (32, 32));
        let req = LoadRequest {
            path: PathBuf::from("fake.onnx"),
            requested: "auto".into(),
            gpu_precision: None,
        };
        let b = script.compile(&req, "openvino:cpu", false).into_backend();
        (script, b)
    }

    #[test]
    fn steps_play_in_order_with_completions() {
        let (script, mut b) = backend(2);
        // The warm-up submits (one per slot) do not consume a step.
        b.submit(0, &[], &[]).unwrap();
        b.collect(0).unwrap();
        b.submit(1, &[], &[]).unwrap();
        b.collect(1).unwrap();
        // The worker drops warm-up completions the same way.
        while b.completions().try_recv().is_ok() {}
        script.push([
            Step::detections(&[[1.0, 2.0, 3.0, 4.0, 0.9, 0.0]], Duration::from_millis(5)),
            Step::Fail("boom".into()),
            Step::Panic,
        ]);
        let s1 = b.submit(0, &[], &[]).unwrap();
        let c = b
            .completions()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        assert_eq!((c.slot, c.seq), (0, s1));
        let outs = b.collect(0).unwrap();
        assert_eq!(outs[0].shape, vec![1, 1, 6]);
        b.submit(0, &[], &[]).unwrap();
        assert_eq!(format!("{:#}", b.collect(0).unwrap_err()), "boom");
        b.submit(1, &[], &[]).unwrap();
        let e = b.collect(1).unwrap_err();
        assert!(super::super::is_session_dead(&e));
        assert!(!b.is_healthy());
        assert!(b.submit(1, &[], &[]).is_err());
        assert_eq!(script.submits(), 3);
        assert_eq!(script.remaining(), 0);
    }

    #[test]
    fn hang_blocks_until_released() {
        let (script, mut b) = backend(1);
        // Warm-up first (it never hangs unless `warmup_hangs`).
        b.submit(0, &[], &[]).unwrap();
        b.collect(0).unwrap();
        while b.completions().try_recv().is_ok() {}
        script.push([Step::Hang]);
        let seq = b.submit(0, &[], &[]).unwrap();
        assert!(b.completions().try_recv().is_err());
        let s2 = script.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            s2.release(Err("device lost".into()));
        });
        let c = b
            .completions()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        assert_eq!(c.seq, seq);
        assert_eq!(format!("{:#}", b.collect(0).unwrap_err()), "device lost");
        t.join().unwrap();
        b.abort();
        assert_eq!(script.calls().last(), Some(&Call::Abort));
    }
}
