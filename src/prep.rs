//! Process-wide decode + preprocess pool (docs/REVIEW_PLAN.md, phase 2.1).
//!
//! JPEG decode and letterbox take ~10-20 ms of CPU per 1080p frame; done on the model's worker
//! thread they would sit between two inferences. Instead every worker sends a [`PrepJob`] to
//! one shared pool of `preprocess_threads` threads (owned by the registry generation) and keeps
//! inferring while the pool prepares the next frames. Each pool thread keeps one
//! [`Preprocessor`] per distinct model geometry ([`PrepSpec`]), and the CHW tensors come from a
//! [`TensorPool`] so steady state does not allocate 4.9 MB per request.
//!
//! Between decode and letterbox the pool applies the request's optional enhancement
//! ([`crate::model::enhance`], chosen per request from the frame's scene); the scene statistics
//! are always taken from the original frame.

use crate::image::RgbImage;
use crate::image_stats::ImageStats;
use crate::model::enhance::{self, PreprocessModes};
use crate::model::preprocess::Preprocessor;
use crate::model::{Family, Normalization, PreprocessCtx, ResizeMode};
use crate::sync::LockExt;
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Buffers kept per tensor length; more are freed on return.
const MAX_POOLED_PER_LEN: usize = 32;
/// Queued jobs per pool thread before `submit` blocks (workers keep their own look-ahead far
/// below this).
const QUEUE_PER_THREAD: usize = 16;

/// What [`Preprocessor::for_family`] needs: the model input size, resize mode and
/// normalization (as bit patterns so specs can key a map).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrepSpec {
    pub input_w: u32,
    pub input_h: u32,
    pub mode: ResizeMode,
    norm: Option<[u32; 6]>,
}

impl PrepSpec {
    pub fn new(
        input_w: u32,
        input_h: u32,
        mode: ResizeMode,
        normalization: Option<Normalization>,
    ) -> Self {
        let norm = normalization.map(|n| {
            [
                n.mean[0].to_bits(),
                n.mean[1].to_bits(),
                n.mean[2].to_bits(),
                n.std[0].to_bits(),
                n.std[1].to_bits(),
                n.std[2].to_bits(),
            ]
        });
        Self {
            input_w,
            input_h,
            mode,
            norm,
        }
    }

    /// The spec of a model with input `(w, h)` of `family`.
    pub fn for_family(input_w: u32, input_h: u32, family: &dyn Family) -> Self {
        Self::new(
            input_w,
            input_h,
            family.resize_mode(),
            family.normalization(),
        )
    }

    pub fn normalization(&self) -> Option<Normalization> {
        self.norm.map(|b| Normalization {
            mean: [
                f32::from_bits(b[0]),
                f32::from_bits(b[1]),
                f32::from_bits(b[2]),
            ],
            std: [
                f32::from_bits(b[3]),
                f32::from_bits(b[4]),
                f32::from_bits(b[5]),
            ],
        })
    }

    /// A preprocessor for this spec.
    pub fn preprocessor(&self) -> Preprocessor {
        Preprocessor::new(self.input_w, self.input_h, self.mode)
            .with_normalization(self.normalization())
    }

    /// `3 * input_w * input_h`.
    pub fn tensor_len(&self) -> usize {
        3 * self.input_w as usize * self.input_h as usize
    }
}

/// Reusable `f32` tensor buffers keyed by length.
#[derive(Debug, Default)]
pub struct TensorPool {
    free: Mutex<HashMap<usize, Vec<Vec<f32>>>>,
}

impl TensorPool {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// A buffer with capacity for `len` values (contents unspecified, length 0 or `len`),
    /// returned to the pool when the [`PooledBuf`] drops.
    pub fn take(self: &Arc<Self>, len: usize) -> PooledBuf {
        let buf = self
            .free
            .lock_recover()
            .get_mut(&len)
            .and_then(Vec::pop)
            .unwrap_or_else(|| Vec::with_capacity(len));
        PooledBuf {
            buf,
            len,
            pool: Arc::clone(self),
        }
    }

    /// Buffers currently free for `len`.
    pub fn free_count(&self, len: usize) -> usize {
        self.free.lock_recover().get(&len).map_or(0, Vec::len)
    }

    fn give_back(&self, len: usize, buf: Vec<f32>) {
        let mut free = self.free.lock_recover();
        let list = free.entry(len).or_default();
        if list.len() < MAX_POOLED_PER_LEN {
            list.push(buf);
        }
    }
}

/// A tensor buffer from a [`TensorPool`]; goes back to it on drop.
#[derive(Debug)]
pub struct PooledBuf {
    buf: Vec<f32>,
    len: usize,
    pool: Arc<TensorPool>,
}

impl PooledBuf {
    pub fn as_slice(&self) -> &[f32] {
        &self.buf
    }

    pub fn vec_mut(&mut self) -> &mut Vec<f32> {
        &mut self.buf
    }
}

impl std::ops::Deref for PooledBuf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.buf
    }
}

impl Drop for PooledBuf {
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.buf);
        if buf.capacity() >= self.len {
            self.pool.give_back(self.len, buf);
        }
    }
}

/// A decoded and preprocessed request.
#[derive(Debug)]
pub struct Prepared {
    /// CHW `f32` input tensor, `3 * input_w * input_h` values.
    pub chw: PooledBuf,
    pub ctx: PreprocessCtx,
    /// The decoded image as received (before any enhancement), kept only when the job asked
    /// for it (`save_image_path`, forwarding to another model).
    pub img: Option<RgbImage>,
    /// Scene statistics of the decoded image (a strided sample, see [`crate::image_stats`]).
    pub stats: crate::image_stats::ImageStats,
    pub decode: Duration,
    /// Enhancement (when any) plus letterbox / stretch into the tensor.
    pub preprocess: Duration,
}

/// One request for the pool. The reply goes to a bounded(1) channel owned by the worker.
#[derive(Debug)]
pub struct PrepJob {
    pub image: bytes::Bytes,
    pub spec: PrepSpec,
    /// Keep the decoded image in [`Prepared::img`].
    pub keep_image: bool,
    /// Enhancement modes of this request (from the model's live settings when the request was
    /// accepted; per request, so a config change applies to the next request). The pool picks
    /// `night_preprocess` or `preprocess` from the decoded frame's scene.
    pub preprocess: PreprocessModes,
    pub reply: Sender<Result<Prepared>>,
}

/// The pool: `threads` OS threads fed by one bounded channel.
pub struct PrepPool {
    tx: Option<Sender<PrepJob>>,
    threads: Vec<JoinHandle<()>>,
    tensors: Arc<TensorPool>,
}

impl std::fmt::Debug for PrepPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrepPool")
            .field("threads", &self.threads.len())
            .finish()
    }
}

impl PrepPool {
    /// Start `threads` (at least 1) pool threads.
    pub fn new(threads: usize) -> Arc<Self> {
        let n = threads.max(1);
        let (tx, rx) = crossbeam_channel::bounded::<PrepJob>(n * QUEUE_PER_THREAD);
        let tensors = TensorPool::new();
        let threads = (0..n)
            .filter_map(|i| {
                let rx = rx.clone();
                let tensors = tensors.clone();
                std::thread::Builder::new()
                    .name(format!("prep-{i}"))
                    .spawn(move || pool_thread(rx, tensors))
                    .map_err(|e| warn!("could not spawn preprocess thread {i}: {e}"))
                    .ok()
            })
            .collect::<Vec<_>>();
        debug!(threads = threads.len(), "preprocess pool started");
        Arc::new(Self {
            tx: Some(tx),
            threads,
            tensors,
        })
    }

    /// Number of pool threads.
    pub fn threads(&self) -> usize {
        self.threads.len()
    }

    /// The pool's tensor buffers.
    pub fn tensors(&self) -> &Arc<TensorPool> {
        &self.tensors
    }

    /// Queue `job` (blocks while the pool queue is full). Fails when no pool thread runs.
    pub fn submit(&self, job: PrepJob) -> Result<()> {
        if self.threads.is_empty() {
            anyhow::bail!("no preprocess thread is running");
        }
        self.tx
            .as_ref()
            .context("preprocess pool stopped")?
            .send(job)
            .map_err(|_| anyhow::anyhow!("preprocess pool stopped"))
    }

    /// Decode + preprocess `image` on the calling thread (what a pool thread does): scene
    /// statistics of the decoded frame, then [`PrepPool::prepare_image`].
    pub fn prepare_now(
        pre: &mut Preprocessor,
        tensors: &Arc<TensorPool>,
        image: &[u8],
        spec: &PrepSpec,
        keep_image: bool,
        modes: PreprocessModes,
    ) -> Result<Prepared> {
        let t = Instant::now();
        let img = crate::image::decode(image)?;
        let stats = ImageStats::compute(&img);
        let decode = t.elapsed();
        let mut p = Self::prepare_image(pre, tensors, img, stats, spec, keep_image, modes)?;
        p.decode = decode;
        Ok(p)
    }

    /// Preprocess an already decoded `img` whose statistics are `stats`: the enhancement of
    /// `modes` for `stats.scene()` (on a copy when the original is kept), then the letterbox /
    /// stretch into a pooled tensor. [`Prepared::decode`] is zero.
    pub fn prepare_image(
        pre: &mut Preprocessor,
        tensors: &Arc<TensorPool>,
        img: RgbImage,
        stats: ImageStats,
        spec: &PrepSpec,
        keep_image: bool,
        modes: PreprocessModes,
    ) -> Result<Prepared> {
        let t = Instant::now();
        let mode = modes.for_scene(stats.scene());
        let mut img = img;
        let mut enhanced = None;
        if !mode.is_none() {
            if keep_image {
                let mut e = img.clone();
                enhance::apply(&mut e, mode);
                enhanced = Some(e);
            } else {
                enhance::apply(&mut img, mode);
            }
        }
        let input = enhanced.as_ref().unwrap_or(&img);
        let mut chw = tensors.take(spec.tensor_len());
        let ctx = pre.run_into(&input.rgb, input.width, input.height, chw.vec_mut())?;
        let preprocess = t.elapsed();
        Ok(Prepared {
            chw,
            ctx,
            img: keep_image.then_some(img),
            stats,
            decode: Duration::ZERO,
            preprocess,
        })
    }
}

impl Drop for PrepPool {
    fn drop(&mut self) {
        // Disconnect the queue; every thread exits after its current job.
        self.tx = None;
        for t in self.threads.drain(..) {
            if t.join().is_err() {
                warn!("preprocess thread panicked");
            }
        }
    }
}

fn pool_thread(rx: Receiver<PrepJob>, tensors: Arc<TensorPool>) {
    let mut pres: HashMap<PrepSpec, Preprocessor> = HashMap::new();
    while let Ok(job) = rx.recv() {
        let PrepJob {
            image,
            spec,
            keep_image,
            preprocess,
            reply,
        } = job;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let pre = pres.entry(spec).or_insert_with(|| spec.preprocessor());
            PrepPool::prepare_now(pre, &tensors, &image, &spec, keep_image, preprocess)
        }));
        let result = match result {
            Ok(r) => r,
            Err(_) => {
                // The preprocessor may be half-updated: start that geometry afresh.
                pres.remove(&spec);
                Err(anyhow::anyhow!("preprocessing panicked"))
            }
        };
        // The worker may have gone (shutdown); nothing to do then.
        let _ = reply.try_send(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jpeg(w: u32, h: u32) -> Vec<u8> {
        let img = RgbImage {
            width: w,
            height: h,
            rgb: (0..w * h * 3).map(|i| (i % 251) as u8).collect(),
        };
        crate::image::encode_jpeg(&img, 90).unwrap()
    }

    fn job(image: Vec<u8>, spec: PrepSpec, keep: bool) -> (PrepJob, Receiver<Result<Prepared>>) {
        let (tx, rx) = crossbeam_channel::bounded(1);
        (
            PrepJob {
                image: image.into(),
                spec,
                keep_image: keep,
                preprocess: PreprocessModes::default(),
                reply: tx,
            },
            rx,
        )
    }

    #[test]
    fn pool_prepares_like_the_preprocessor() {
        let pool = PrepPool::new(2);
        assert_eq!(pool.threads(), 2);
        let spec = PrepSpec::new(64, 64, ResizeMode::Letterbox, None);
        let bytes = jpeg(160, 90);
        let (j, rx) = job(bytes.clone(), spec, true);
        pool.submit(j).unwrap();
        let p = rx.recv().unwrap().unwrap();
        let img = crate::image::decode(&bytes).unwrap();
        let mut pre = spec.preprocessor();
        let (want, ctx) = pre.run(&img.rgb, img.width, img.height).unwrap();
        assert_eq!(p.ctx, ctx);
        assert_eq!(&*p.chw, want);
        assert_eq!(p.img.as_ref().map(|i| (i.width, i.height)), Some((160, 90)));

        // Without `keep_image` the decoded image is dropped on the pool thread.
        let (j, rx) = job(bytes, spec, false);
        pool.submit(j).unwrap();
        assert!(rx.recv().unwrap().unwrap().img.is_none());
    }

    #[test]
    fn enhancement_follows_the_scene_and_keeps_the_original() {
        use crate::model::enhance::Preprocess;
        let spec = PrepSpec::new(32, 32, ResizeMode::Stretch, None);
        let tensors = TensorPool::new();
        let mut pre = spec.preprocessor();
        // A monochrome (IR) frame and a colour (day) frame, both mid-bright.
        let ir = RgbImage {
            width: 32,
            height: 32,
            rgb: (0..32 * 32)
                .flat_map(|i| [(i % 200) as u8 + 30; 3])
                .collect(),
        };
        // Hues that do not follow brightness (a constant offset or gain would be tinted IR).
        let mut day = ir.clone();
        for (i, p) in day.rgb.chunks_mut(3).enumerate() {
            p[1] = ((i * 7) % 200) as u8 + 30;
            p[2] = ((i * 13) % 200) as u8 + 30;
        }
        let night_gamma = PreprocessModes {
            preprocess: Preprocess::None,
            night_preprocess: Some(Preprocess::Gamma(0.5)),
        };
        let tensor = |img: &RgbImage, mode: Preprocess, pre: &mut Preprocessor| {
            let mut e = img.clone();
            crate::model::enhance::apply(&mut e, mode);
            pre.run(&e.rgb, e.width, e.height).unwrap().0.to_vec()
        };
        for (img, want_mode) in [(&ir, Preprocess::Gamma(0.5)), (&day, Preprocess::None)] {
            let stats = ImageStats::compute(img);
            for keep in [false, true] {
                let p = PrepPool::prepare_image(
                    &mut pre,
                    &tensors,
                    img.clone(),
                    stats,
                    &spec,
                    keep,
                    night_gamma,
                )
                .unwrap();
                assert_eq!(&*p.chw, tensor(img, want_mode, &mut pre), "{want_mode}");
                assert_eq!(p.stats, stats);
                // The kept image is the frame as received.
                assert_eq!(p.img.as_ref(), keep.then_some(img));
            }
        }
        assert_eq!(
            ImageStats::compute(&ir).scene(),
            crate::image_stats::Scene::Ir
        );
        assert_eq!(
            ImageStats::compute(&day).scene(),
            crate::image_stats::Scene::Day
        );
    }

    #[test]
    fn bad_images_are_errors_and_the_pool_keeps_going() {
        let pool = PrepPool::new(1);
        let spec = PrepSpec::new(32, 32, ResizeMode::Stretch, Some(Normalization::IMAGENET));
        let (j, rx) = job(b"not an image".to_vec(), spec, false);
        pool.submit(j).unwrap();
        assert!(rx.recv().unwrap().is_err());
        let (j, rx) = job(jpeg(40, 40), spec, false);
        pool.submit(j).unwrap();
        let p = rx.recv().unwrap().unwrap();
        assert_eq!(p.chw.len(), 3 * 32 * 32);
        assert_eq!(spec.normalization(), Some(Normalization::IMAGENET));
    }

    #[test]
    fn tensors_are_recycled() {
        let pool = PrepPool::new(1);
        let spec = PrepSpec::new(48, 48, ResizeMode::Letterbox, None);
        let len = spec.tensor_len();
        let tensors = pool.tensors().clone();
        let (j, rx) = job(jpeg(64, 64), spec, false);
        pool.submit(j).unwrap();
        let p = rx.recv().unwrap().unwrap();
        let ptr = p.chw.as_ptr();
        assert_eq!(tensors.free_count(len), 0);
        drop(p);
        assert_eq!(tensors.free_count(len), 1, "returned on drop");
        let (j, rx) = job(jpeg(64, 64), spec, false);
        pool.submit(j).unwrap();
        let p = rx.recv().unwrap().unwrap();
        assert_eq!(p.chw.as_ptr(), ptr, "same allocation reused");
    }

    #[test]
    fn dropping_the_pool_stops_its_threads() {
        let pool = PrepPool::new(3);
        let spec = PrepSpec::new(16, 16, ResizeMode::Letterbox, None);
        let (j, rx) = job(jpeg(20, 20), spec, false);
        pool.submit(j).unwrap();
        drop(pool); // joins: the queued job still completes or its reply is dropped
        let _ = rx.recv();
        let zero = PrepPool::new(0);
        assert_eq!(zero.threads(), 1, "at least one thread");
    }
}
