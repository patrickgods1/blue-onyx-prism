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
//!
//! Each pool thread decodes into its own reusable RGB buffer ([`Scratch`], at most one per
//! thread, never more than [`MAX_SCRATCH_BYTES`]): a 2688x1520 frame is 12.3 MB that would
//! otherwise be allocated, page-faulted and freed per request. The buffer never leaves its
//! thread. A frame that outlives the request (kept for `save_image_path`, the night model
//! switch or the verifier) is decoded into a new buffer of its own, as before, so nothing kept
//! can alias the reused one.
//!
//! Frames another worker already decoded ([`PrepInput::Decoded`]: the night model switch and
//! verifier checks) are prepared here too, never on the receiving model's inference thread.
//!
//! Nothing here is held for a model that is gone (audit 2026-11, backend-3 / backend-4): the pool
//! drops a tensor length's free buffers once no request took one for [`IDLE_DROP`] (a model
//! switched or reloaded to another input size), keeps at most [`MAX_POOLED_BYTES`] of free
//! tensors in all (or [`MIN_POOLED_FRAMES`] of the largest live length, if more), and each pool
//! thread drops a geometry's [`Preprocessor`] unused for [`IDLE_DROP`]. Both run when the pool is
//! used (a take, a give-back, a job), never on a timer; a dropped entry is rebuilt on demand.
//! [`PooledBuf`] holds the pool weakly, so a buffer that outlives its pool (a detached ONNX
//! Runtime session thread) is freed on drop and pins only itself.

use crate::image::RgbImage;
use crate::image_stats::ImageStats;
use crate::model::enhance::{self, PreprocessModes};
use crate::model::preprocess::Preprocessor;
use crate::model::{Family, Normalization, PreprocessCtx, ResizeMode};
use crate::sync::LockExt;
use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Buffers kept per tensor length; more are freed on return.
const MAX_POOLED_PER_LEN: usize = 32;
/// A tensor length nobody took for this long loses its free buffers, and a pool thread drops a
/// [`Preprocessor`] unused this long. Blue Iris sends frames every few seconds per camera, so a
/// model in use never gets near it; a model that was replaced (another input size after a live
/// reload or a default-model switch) stops pinning its tensors and resize buffers 10 minutes
/// later, and coming back costs one allocation.
pub const IDLE_DROP: Duration = Duration::from_secs(10 * 60);
/// Free tensor bytes the pool keeps in all (64 MiB): 13 tensors of a 640x640 model (4.9 MB each),
/// above what the target box keeps in flight (each worker's slots plus its look-ahead of up to 4:
/// 4 to 8 tensors for a LATENCY integrated GPU), or three of a 1280x1280 model (19.7 MB). A
/// buffer returned above the cap evicts one of the length taken longest ago, else is freed.
pub const MAX_POOLED_BYTES: usize = 64 << 20;
/// The cap is at least this many tensors of the largest length in use, so a model with a huge
/// input (up to the 256 MiB tensor `check_input_size` allows) still reuses its buffers.
pub const MIN_POOLED_FRAMES: usize = 2;
/// Queued jobs per pool thread before `submit` blocks (workers keep their own look-ahead far
/// below this).
const QUEUE_PER_THREAD: usize = 16;
/// Largest decode buffer a pool thread keeps between requests ([`Scratch`]): a 4K frame
/// (3840 x 2160 x 3 = 24.9 MB) fits. A larger frame's buffer is freed after its request, so one
/// huge upload does not pin its memory on the thread for good.
pub const MAX_SCRATCH_BYTES: usize = 32 * 1024 * 1024;

/// The reusable RGB buffer of one pool thread: decode target of requests whose frame is not
/// kept, and the copy an enhancement works on when the frame is kept. Owned by the thread
/// (never shared, never handed to a request), so there is at most one per thread.
#[derive(Debug)]
pub struct Scratch {
    rgb: Vec<u8>,
    /// Largest capacity kept ([`MAX_SCRATCH_BYTES`]; smaller in tests).
    limit: usize,
}

impl Default for Scratch {
    fn default() -> Self {
        Self::with_limit(MAX_SCRATCH_BYTES)
    }
}

impl Scratch {
    /// A scratch that keeps buffers of at most `limit` bytes.
    pub fn with_limit(limit: usize) -> Self {
        Self {
            rgb: Vec::new(),
            limit,
        }
    }

    /// The buffer (empty when none is kept); give it back with [`Self::put`].
    fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.rgb)
    }

    /// Keep `buf` for the next request when it fits the limit and is at least as large as
    /// the one kept (one buffer at most: the other is freed).
    fn put(&mut self, buf: Vec<u8>) {
        if buf.capacity() <= self.limit && buf.capacity() >= self.rgb.capacity() {
            self.rgb = buf;
        }
    }

    /// Capacity of the kept buffer (0: none).
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.rgb.capacity()
    }

    /// Address of the kept buffer, for tests that check a kept frame never aliases it.
    #[cfg(test)]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.rgb.as_ptr()
    }
}

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

/// The time source of a [`TensorPool`] (a fake one in tests).
type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// Bytes a tensor buffer holds.
fn bytes_of(buf: &Vec<f32>) -> usize {
    buf.capacity().saturating_mul(std::mem::size_of::<f32>())
}

/// The free buffers of one tensor length.
#[derive(Debug)]
struct LenList {
    bufs: Vec<Vec<f32>>,
    /// The latest [`TensorPool::take`] of this length.
    last_take: Instant,
}

/// Every length's free buffers and their byte total.
#[derive(Debug, Default)]
struct FreeLists {
    lists: HashMap<usize, LenList>,
    /// Sum of [`bytes_of`] over every list's buffers.
    bytes: usize,
    /// Live [`LengthHold`]s per length.
    users: HashMap<usize, usize>,
}

impl FreeLists {
    /// Remove every length not taken for [`IDLE_DROP`] at `now` (but `keep`), with its buffers,
    /// which are moved to `out`. True when a buffer was among them.
    fn drop_idle(&mut self, now: Instant, keep: Option<usize>, out: &mut Vec<Vec<f32>>) -> bool {
        let before = out.len();
        let idle: Vec<usize> = self
            .lists
            .iter()
            .filter(|(len, l)| {
                Some(**len) != keep && now.saturating_duration_since(l.last_take) >= IDLE_DROP
            })
            .map(|(len, _)| *len)
            .collect();
        for len in idle {
            if let Some(l) = self.lists.remove(&len) {
                self.remove_all(l.bufs, out);
            }
        }
        out.len() > before
    }

    /// Move `bufs` to `out`, uncounted.
    fn remove_all(&mut self, bufs: Vec<Vec<f32>>, out: &mut Vec<Vec<f32>>) {
        for b in bufs {
            self.bytes = self.bytes.saturating_sub(bytes_of(&b));
            out.push(b);
        }
    }

    /// The byte cap: [`MAX_POOLED_BYTES`], or [`MIN_POOLED_FRAMES`] tensors of the largest length
    /// with a list (taken in the last [`IDLE_DROP`]), if more.
    fn cap(&self) -> usize {
        let largest = self.lists.keys().copied().max().unwrap_or(0);
        MAX_POOLED_BYTES.max(
            largest
                .saturating_mul(std::mem::size_of::<f32>())
                .saturating_mul(MIN_POOLED_FRAMES),
        )
    }
}

/// Reusable `f32` tensor buffers keyed by length, bounded by idle time and total bytes (see the
/// module docs).
pub struct TensorPool {
    free: Mutex<FreeLists>,
    clock: Clock,
}

impl std::fmt::Debug for TensorPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let free = self.free.lock_recover();
        f.debug_struct("TensorPool")
            .field("lengths", &free.lists.len())
            .field("bytes", &free.bytes)
            .finish_non_exhaustive()
    }
}

impl TensorPool {
    pub fn new() -> Arc<Self> {
        Self::with_clock(Instant::now)
    }

    /// A pool that reads the time from `clock` (tests: a fake clock for exact idle times).
    pub fn with_clock(clock: impl Fn() -> Instant + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self {
            free: Mutex::new(FreeLists::default()),
            clock: Arc::new(clock),
        })
    }

    /// A buffer with capacity for `len` values (contents unspecified, length 0 or `len`),
    /// returned to the pool when the [`PooledBuf`] drops. Drops the lists of other lengths idle
    /// for [`IDLE_DROP`] first.
    pub fn take(self: &Arc<Self>, len: usize) -> PooledBuf {
        let now = (self.clock)();
        let mut dropped = Vec::new();
        let (buf, idle) = {
            let mut free = self.free.lock_recover();
            let idle = free.drop_idle(now, Some(len), &mut dropped);
            let FreeLists { lists, bytes, .. } = &mut *free;
            let list = lists.entry(len).or_insert_with(|| LenList {
                bufs: Vec::new(),
                last_take: now,
            });
            list.last_take = now;
            let buf = list.bufs.pop();
            if let Some(b) = &buf {
                *bytes = bytes.saturating_sub(bytes_of(b));
            }
            (buf, idle)
        };
        released(dropped, idle);
        PooledBuf {
            buf: buf.unwrap_or_else(|| Vec::with_capacity(len)),
            len,
            pool: Arc::downgrade(self),
        }
    }

    /// Buffers currently free for `len`.
    #[cfg(test)]
    pub(crate) fn free_count(&self, len: usize) -> usize {
        self.free
            .lock_recover()
            .lists
            .get(&len)
            .map_or(0, |l| l.bufs.len())
    }

    /// Bytes of free buffers the pool holds now.
    pub fn pooled_bytes(&self) -> usize {
        self.free.lock_recover().bytes
    }

    /// Tensor lengths the pool tracks now (taken in the last [`IDLE_DROP`]).
    pub fn lengths(&self) -> usize {
        self.free.lock_recover().lists.len()
    }

    /// Mark `len` as used by a loaded model until the returned hold drops (the worker's engine
    /// keeps one). When the last hold of a length drops (the model was retired, reloaded at
    /// another size, or its generation ended), that length's free buffers are freed at once,
    /// instead of [`IDLE_DROP`] after its last take, and a trim is asked for.
    pub fn hold_length(self: &Arc<Self>, len: usize) -> LengthHold {
        let mut free = self.free.lock_recover();
        let n = free.users.entry(len).or_insert(0);
        *n = n.saturating_add(1);
        LengthHold {
            pool: Arc::downgrade(self),
            len,
        }
    }

    /// The last hold of `len` dropped: free its list.
    fn release_length(&self, len: usize) {
        let mut dropped = Vec::new();
        {
            let mut free = self.free.lock_recover();
            let last = match free.users.get_mut(&len) {
                Some(n) if *n > 1 => {
                    *n -= 1;
                    false
                }
                Some(_) => true,
                None => false,
            };
            if !last {
                return;
            }
            free.users.remove(&len);
            if let Some(l) = free.lists.remove(&len) {
                free.remove_all(l.bufs, &mut dropped);
            }
        }
        let any = !dropped.is_empty();
        released(dropped, any);
    }

    /// Keep `buf` for the next [`Self::take`] of `len` when that length is still tracked (taken
    /// in the last [`IDLE_DROP`]), has fewer than [`MAX_POOLED_PER_LEN`] free and the byte cap
    /// allows (evicting buffers of lengths taken longer ago first); else free it.
    fn give_back(&self, len: usize, buf: Vec<f32>) {
        let now = (self.clock)();
        let mut dropped = Vec::new();
        let idle = {
            let mut free = self.free.lock_recover();
            let idle = free.drop_idle(now, None, &mut dropped);
            let cap = free.cap();
            let size = bytes_of(&buf);
            match free.lists.get(&len).map(|l| (l.bufs.len(), l.last_take)) {
                Some((n, last_take)) if n < MAX_POOLED_PER_LEN => {
                    // Make room from lengths taken longest ago (before this one), one buffer at a
                    // time.
                    while free.bytes.saturating_add(size) > cap {
                        let older = free
                            .lists
                            .iter()
                            .filter(|(l, list)| {
                                **l != len && list.last_take < last_take && !list.bufs.is_empty()
                            })
                            .min_by_key(|(_, list)| list.last_take)
                            .map(|(l, _)| *l);
                        let Some(b) = older
                            .and_then(|l| free.lists.get_mut(&l))
                            .and_then(|list| list.bufs.pop())
                        else {
                            break;
                        };
                        free.bytes = free.bytes.saturating_sub(bytes_of(&b));
                        dropped.push(b);
                    }
                    if free.bytes.saturating_add(size) <= cap {
                        free.bytes = free.bytes.saturating_add(size);
                        if let Some(list) = free.lists.get_mut(&len) {
                            list.bufs.push(buf);
                        }
                    } else {
                        dropped.push(buf);
                    }
                }
                // Full, or the length went idle while this buffer was out.
                _ => dropped.push(buf),
            }
            idle
        };
        released(dropped, idle);
    }
}

/// Free `dropped` (outside the pool's lock) and, when a length went idle with buffers (`trim`),
/// ask for an allocator trim: that memory is otherwise kept by the heap.
fn released(dropped: Vec<Vec<f32>>, trim: bool) {
    drop(dropped);
    if trim {
        crate::backend::memory::trim_soon("prep pool: tensors of an unused input size");
    }
}

/// A loaded model's use of one tensor length ([`TensorPool::hold_length`]); holds the pool
/// weakly, like [`PooledBuf`].
#[derive(Debug)]
pub struct LengthHold {
    pool: std::sync::Weak<TensorPool>,
    len: usize,
}

impl Drop for LengthHold {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            pool.release_length(self.len);
        }
    }
}

/// A tensor buffer from a [`TensorPool`]; goes back to it on drop while the pool lives (it holds
/// the pool weakly: a buffer kept by a detached session thread pins only itself).
#[derive(Debug)]
pub struct PooledBuf {
    buf: Vec<f32>,
    len: usize,
    pool: Weak<TensorPool>,
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
        if buf.capacity() >= self.len
            && let Some(pool) = self.pool.upgrade()
        {
            pool.give_back(self.len, buf);
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

/// What a [`PrepJob`] prepares.
#[derive(Debug)]
pub enum PrepInput {
    /// The upload's encoded bytes (decoded on the pool thread).
    Encoded(bytes::Bytes),
    /// A frame another worker decoded ([`crate::worker::Decoded`]), its scene statistics (the
    /// sender's) and the sender's decode time (reported as this request's decode).
    Decoded {
        img: RgbImage,
        stats: ImageStats,
        decode: Duration,
    },
}

/// One request for the pool. The reply goes to a bounded(1) channel owned by the worker.
#[derive(Debug)]
pub struct PrepJob {
    pub input: PrepInput,
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
    /// Jobs the pool threads finished (any outcome). Only tests read it ([`Self::jobs_done`]).
    #[cfg_attr(
        not(test),
        allow(
            dead_code,
            reason = "the pool threads count every job; tests read the count"
        )
    )]
    done: Arc<AtomicU64>,
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
        let done = Arc::new(AtomicU64::new(0));
        let threads = (0..n)
            .filter_map(|i| {
                let rx = rx.clone();
                let tensors = tensors.clone();
                let done = done.clone();
                std::thread::Builder::new()
                    .name(format!("prep-{i}"))
                    .spawn(move || pool_thread(rx, tensors, done))
                    .map_err(|e| warn!("could not spawn preprocess thread {i}: {e}"))
                    .ok()
            })
            .collect::<Vec<_>>();
        debug!(threads = threads.len(), "preprocess pool started");
        Arc::new(Self {
            tx: Some(tx),
            threads,
            tensors,
            done,
        })
    }

    /// Jobs the pool threads have finished so far (prepared or failed).
    #[cfg(test)]
    pub(crate) fn jobs_done(&self) -> u64 {
        self.done.load(Ordering::Acquire)
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

    /// Decode + preprocess `image` on the calling thread (what a pool thread does, without a
    /// reused buffer): scene statistics of the decoded frame, then [`PrepPool::prepare_image`].
    #[cfg(test)]
    pub(crate) fn prepare_now(
        pre: &mut Preprocessor,
        tensors: &Arc<TensorPool>,
        image: &[u8],
        spec: &PrepSpec,
        keep_image: bool,
        modes: PreprocessModes,
    ) -> Result<Prepared> {
        let mut scratch = Scratch::default();
        Self::prepare_encoded(pre, tensors, &mut scratch, image, spec, keep_image, modes)
    }

    /// Prepare `input` (what a pool thread does), decoding into `scratch` unless the frame is
    /// kept. A [`PrepInput::Decoded`] frame reports the sender's decode time.
    pub fn prepare(
        pre: &mut Preprocessor,
        tensors: &Arc<TensorPool>,
        scratch: &mut Scratch,
        input: PrepInput,
        spec: &PrepSpec,
        keep_image: bool,
        modes: PreprocessModes,
    ) -> Result<Prepared> {
        match input {
            PrepInput::Encoded(bytes) => {
                Self::prepare_encoded(pre, tensors, scratch, &bytes, spec, keep_image, modes)
            }
            PrepInput::Decoded { img, stats, decode } => {
                let mut p = Self::prepare_frame(
                    pre, tensors, scratch, img, stats, spec, keep_image, modes,
                )?;
                p.decode = decode;
                Ok(p)
            }
        }
    }

    /// Decode `image` (into `scratch`'s buffer unless the frame is kept: a kept frame outlives
    /// the request and gets its own), take its statistics, then [`Self::prepare_frame`].
    fn prepare_encoded(
        pre: &mut Preprocessor,
        tensors: &Arc<TensorPool>,
        scratch: &mut Scratch,
        image: &[u8],
        spec: &PrepSpec,
        keep_image: bool,
        modes: PreprocessModes,
    ) -> Result<Prepared> {
        let t = Instant::now();
        let mut rgb = if keep_image {
            Vec::new()
        } else {
            scratch.take()
        };
        let (width, height) = match crate::image::decode_into(image, &mut rgb) {
            Ok(size) => size,
            Err(e) => {
                if !keep_image {
                    scratch.put(rgb);
                }
                return Err(e);
            }
        };
        let img = RgbImage { width, height, rgb };
        let stats = ImageStats::compute(&img);
        let decode = t.elapsed();
        let mut p =
            Self::prepare_frame(pre, tensors, scratch, img, stats, spec, keep_image, modes)?;
        p.decode = decode;
        Ok(p)
    }

    /// Preprocess an already decoded `img` whose statistics are `stats`: the enhancement of
    /// `modes` for `stats.scene()` (on a copy when the original is kept), then the letterbox /
    /// stretch into a pooled tensor. [`Prepared::decode`] is zero. Tests only; the pool threads
    /// call [`Self::prepare`].
    #[cfg(test)]
    pub(crate) fn prepare_image(
        pre: &mut Preprocessor,
        tensors: &Arc<TensorPool>,
        img: RgbImage,
        stats: ImageStats,
        spec: &PrepSpec,
        keep_image: bool,
        modes: PreprocessModes,
    ) -> Result<Prepared> {
        let mut scratch = Scratch::default();
        Self::prepare_frame(
            pre,
            tensors,
            &mut scratch,
            img,
            stats,
            spec,
            keep_image,
            modes,
        )
    }

    /// Preprocess an already decoded `img` whose statistics are `stats` (the enhancement of
    /// `modes` for its scene, then the letterbox / stretch into a pooled tensor) with `scratch`:
    /// the enhanced copy of a kept frame is made in its buffer, and a frame that is not kept goes
    /// back to it afterwards.
    #[allow(
        clippy::too_many_arguments,
        reason = "the preprocessing state of a pool thread plus one request's inputs"
    )]
    fn prepare_frame(
        pre: &mut Preprocessor,
        tensors: &Arc<TensorPool>,
        scratch: &mut Scratch,
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
                // The kept frame stays as received: enhance a copy, in the thread's buffer.
                let mut rgb = scratch.take();
                rgb.clear();
                rgb.extend_from_slice(&img.rgb);
                let mut e = RgbImage {
                    width: img.width,
                    height: img.height,
                    rgb,
                };
                enhance::apply(&mut e, mode);
                enhanced = Some(e);
            } else {
                enhance::apply(&mut img, mode);
            }
        }
        let input = enhanced.as_ref().unwrap_or(&img);
        let mut chw = tensors.take(spec.tensor_len());
        let ctx = pre.run_into(&input.rgb, input.width, input.height, chw.vec_mut());
        if let Some(e) = enhanced {
            scratch.put(e.rgb);
        }
        let img = if keep_image {
            Some(img)
        } else {
            scratch.put(img.rgb);
            None
        };
        let ctx = ctx?;
        let preprocess = t.elapsed();
        Ok(Prepared {
            chw,
            ctx,
            img,
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

/// One pool thread's [`Preprocessor`]s, one per geometry, each dropped once unused for
/// [`IDLE_DROP`] (checked when the next job arrives).
#[derive(Debug, Default)]
struct Preprocessors {
    map: HashMap<PrepSpec, (Preprocessor, Instant)>,
}

impl Preprocessors {
    /// The preprocessor of `spec`, used at `now` (made when missing); the others unused for
    /// [`IDLE_DROP`] are dropped first (an allocator trim is asked for when any was).
    fn get(&mut self, spec: PrepSpec, now: Instant) -> &mut Preprocessor {
        let before = self.map.len();
        self.map
            .retain(|s, (_, used)| *s == spec || now.saturating_duration_since(*used) < IDLE_DROP);
        if self.map.len() < before {
            crate::backend::memory::trim_soon("prep pool: preprocessor of an unused input size");
        }
        let entry = self
            .map
            .entry(spec)
            .or_insert_with(|| (spec.preprocessor(), now));
        entry.1 = now;
        &mut entry.0
    }

    fn remove(&mut self, spec: &PrepSpec) {
        self.map.remove(spec);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.len()
    }
}

fn pool_thread(rx: Receiver<PrepJob>, tensors: Arc<TensorPool>, done: Arc<AtomicU64>) {
    let mut pres = Preprocessors::default();
    let mut scratch = Scratch::default();
    while let Ok(job) = rx.recv() {
        let PrepJob {
            input,
            spec,
            keep_image,
            preprocess,
            reply,
        } = job;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let pre = pres.get(spec, Instant::now());
            PrepPool::prepare(
                pre,
                &tensors,
                &mut scratch,
                input,
                &spec,
                keep_image,
                preprocess,
            )
        }));
        let result = match result {
            Ok(r) => r,
            Err(_) => {
                // The preprocessor may be half-updated: start that geometry afresh. The scratch
                // buffer only ever holds pixels of the next decode, so it stays.
                pres.remove(&spec);
                Err(anyhow::anyhow!("preprocessing panicked"))
            }
        };
        done.fetch_add(1, Ordering::AcqRel);
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
                input: PrepInput::Encoded(image.into()),
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

    /// Frames that are not kept are decoded into the thread's one buffer, again and again; a
    /// kept frame never aliases it (and leaves it in place); a frame over the limit is not kept
    /// by the thread.
    #[test]
    fn scratch_is_reused_bounded_and_never_kept() {
        let spec = PrepSpec::new(32, 32, ResizeMode::Letterbox, None);
        let tensors = TensorPool::new();
        let mut pre = spec.preprocessor();
        let small = jpeg(40, 30);
        let frame_len = 40 * 30 * 3;
        let mut scratch = Scratch::with_limit(frame_len);
        let run = |scratch: &mut Scratch, pre: &mut Preprocessor, bytes: &[u8], keep: bool| {
            PrepPool::prepare(
                pre,
                &tensors,
                scratch,
                PrepInput::Encoded(bytes.to_vec().into()),
                &spec,
                keep,
                PreprocessModes::default(),
            )
            .unwrap()
        };
        let want = PrepPool::prepare_now(
            &mut pre,
            &tensors,
            &small,
            &spec,
            true,
            PreprocessModes::default(),
        )
        .unwrap();
        let first = run(&mut scratch, &mut pre, &small, false);
        assert!(first.img.is_none());
        assert_eq!(*first.chw, *want.chw, "same tensor as a fresh decode");
        assert_eq!(scratch.capacity(), frame_len);
        let ptr = scratch.as_ptr();
        let again = run(&mut scratch, &mut pre, &small, false);
        assert_eq!(*again.chw, *want.chw);
        assert_eq!(scratch.as_ptr(), ptr, "the same buffer, reused");

        // A kept frame owns a buffer of its own; the thread's stays where it was.
        let kept = run(&mut scratch, &mut pre, &small, true);
        let img = kept.img.as_ref().unwrap();
        assert_ne!(
            img.rgb.as_ptr(),
            ptr,
            "a kept frame never aliases the scratch"
        );
        assert_eq!(img, want.img.as_ref().unwrap());
        assert_eq!(scratch.as_ptr(), ptr);

        // A larger frame than the limit: decoded, then its buffer is freed, not kept.
        let big = jpeg(64, 48);
        let p = run(&mut scratch, &mut pre, &big, false);
        assert_eq!(p.ctx.orig_w, 64);
        assert!(scratch.capacity() <= frame_len, "{}", scratch.capacity());

        // A bad image hands the buffer back too.
        let mut scratch = Scratch::with_limit(MAX_SCRATCH_BYTES);
        run(&mut scratch, &mut pre, &small, false);
        let ptr = scratch.as_ptr();
        assert!(
            PrepPool::prepare(
                &mut pre,
                &tensors,
                &mut scratch,
                PrepInput::Encoded(b"not an image".to_vec().into()),
                &spec,
                false,
                PreprocessModes::default(),
            )
            .is_err()
        );
        assert_eq!(scratch.as_ptr(), ptr);
    }

    /// A frame another worker decoded is prepared by a pool thread like an upload, with the
    /// sender's decode time.
    #[test]
    fn decoded_frames_are_prepared_on_the_pool() {
        let pool = PrepPool::new(1);
        let spec = PrepSpec::new(64, 64, ResizeMode::Letterbox, None);
        let bytes = jpeg(160, 90);
        let img = crate::image::decode(&bytes).unwrap();
        let stats = ImageStats::compute(&img);
        for keep in [false, true] {
            let (tx, rx) = crossbeam_channel::bounded(1);
            let before = pool.jobs_done();
            pool.submit(PrepJob {
                input: PrepInput::Decoded {
                    img: img.clone(),
                    stats,
                    decode: Duration::from_millis(7),
                },
                spec,
                keep_image: keep,
                preprocess: PreprocessModes::default(),
                reply: tx,
            })
            .unwrap();
            let p = rx.recv().unwrap().unwrap();
            assert_eq!(pool.jobs_done(), before + 1);
            let mut pre = spec.preprocessor();
            let (want, ctx) = pre.run(&img.rgb, 160, 90).unwrap();
            assert_eq!(&*p.chw, want);
            assert_eq!(p.ctx, ctx);
            assert_eq!(p.decode, Duration::from_millis(7));
            assert_eq!(p.stats, stats);
            assert_eq!(p.img.as_ref(), keep.then_some(&img));
        }
    }

    /// A clock tests move by hand.
    #[derive(Clone)]
    struct FakeClock(Arc<Mutex<Instant>>);

    impl FakeClock {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Instant::now())))
        }
        fn now(&self) -> Instant {
            *self.0.lock().unwrap()
        }
        fn advance(&self, by: Duration) {
            *self.0.lock().unwrap() += by;
        }
    }

    fn fake_pool() -> (Arc<TensorPool>, FakeClock) {
        let clock = FakeClock::new();
        let c = clock.clone();
        (TensorPool::with_clock(move || c.now()), clock)
    }

    /// Take `n` buffers of `len` at once, then give them all back.
    fn cycle(pool: &Arc<TensorPool>, len: usize, n: usize) {
        let bufs: Vec<PooledBuf> = (0..n).map(|_| pool.take(len)).collect();
        drop(bufs);
    }

    const F32: usize = std::mem::size_of::<f32>();

    #[test]
    fn a_length_nobody_takes_for_the_idle_time_loses_its_buffers() {
        let (pool, clock) = fake_pool();
        let (a, b) = (100, 200);
        cycle(&pool, a, 3);
        cycle(&pool, b, 2);
        assert_eq!((pool.free_count(a), pool.free_count(b)), (3, 2));
        assert_eq!(pool.pooled_bytes(), (3 * a + 2 * b) * F32);
        assert_eq!(pool.lengths(), 2);

        // Just under the idle time: `a` stays, while `b` is in use.
        clock.advance(IDLE_DROP - Duration::from_secs(1));
        cycle(&pool, b, 1);
        assert_eq!(pool.free_count(a), 3);
        // The idle time reached: the next use of the pool drops `a`'s list.
        clock.advance(Duration::from_secs(1));
        cycle(&pool, b, 1);
        assert_eq!((pool.free_count(a), pool.free_count(b)), (0, 2));
        assert_eq!(pool.pooled_bytes(), 2 * b * F32);
        assert_eq!(pool.lengths(), 1);

        // A give-back alone drops idle lengths too (no take of another length needed).
        let held = pool.take(b);
        clock.advance(IDLE_DROP);
        drop(held);
        assert_eq!(
            (pool.free_count(b), pool.pooled_bytes(), pool.lengths()),
            (0, 0, 0)
        );

        // A buffer that was out while its length went idle is freed, not pooled again.
        let out = pool.take(a);
        clock.advance(IDLE_DROP + Duration::from_secs(5));
        cycle(&pool, b, 1);
        drop(out);
        assert_eq!(pool.free_count(a), 0);
        assert_eq!(pool.pooled_bytes(), b * F32);

        // Coming back costs one allocation; the length is pooled again.
        cycle(&pool, a, 1);
        assert_eq!(pool.free_count(a), 1);
    }

    #[test]
    fn the_byte_cap_holds_and_evicts_the_length_taken_longest_ago() {
        let (pool, clock) = fake_pool();
        // 4 MiB tensors (capacity only: nothing is touched): the 64 MiB cap holds 16.
        let l = (4 << 20) / F32;
        cycle(&pool, l, 20);
        assert_eq!(pool.free_count(l), MAX_POOLED_BYTES / (l * F32));
        assert_eq!(pool.pooled_bytes(), MAX_POOLED_BYTES);
        // A length taken later takes the room of the older one, one buffer each.
        clock.advance(Duration::from_secs(1));
        let m = l - 3;
        cycle(&pool, m, 4);
        assert_eq!(pool.free_count(m), 4);
        assert_eq!(pool.free_count(l), 16 - 4);
        assert!(
            pool.pooled_bytes() <= MAX_POOLED_BYTES,
            "{}",
            pool.pooled_bytes()
        );
        // Lengths taken at the same time never evict each other: `l` (taken again now) fills the
        // room left and its 13th return is freed, `m` keeps its 4.
        let held: Vec<PooledBuf> = (0..13).map(|_| pool.take(l)).collect();
        assert_eq!(pool.free_count(l), 0);
        drop(held);
        assert_eq!((pool.free_count(l), pool.free_count(m)), (12, 4));
        assert!(
            pool.pooled_bytes() <= MAX_POOLED_BYTES,
            "{}",
            pool.pooled_bytes()
        );
        assert_eq!(
            pool.pooled_bytes(),
            pool.free_count(l) * l * F32 + pool.free_count(m) * m * F32
        );

        // A model with a huge input still keeps two of its tensors (36 MiB each, cap 72 MiB).
        let (big, _clock) = fake_pool();
        let h = (36 << 20) / F32;
        cycle(&big, h, 3);
        assert_eq!(big.free_count(h), MIN_POOLED_FRAMES);
        assert_eq!(big.pooled_bytes(), MIN_POOLED_FRAMES * h * F32);
    }

    /// The last hold of a length frees its buffers at once; a second holder (a replacement
    /// loading beside the old worker) keeps them; other lengths stay.
    #[test]
    fn the_last_hold_of_a_length_frees_its_buffers_at_once() {
        let (pool, _clock) = fake_pool();
        let old = pool.hold_length(10);
        let other = pool.hold_length(20);
        cycle(&pool, 10, 2);
        cycle(&pool, 20, 3);
        let replacement = pool.hold_length(10);
        drop(old);
        assert_eq!((pool.free_count(10), pool.free_count(20)), (2, 3));
        drop(replacement);
        assert_eq!((pool.free_count(10), pool.free_count(20)), (0, 3));
        assert_eq!(pool.pooled_bytes(), 3 * 20 * F32);
        // A buffer of the released length still out is freed on return, not pooled.
        let out = pool.take(20);
        drop(other);
        assert_eq!(pool.free_count(20), 0);
        drop(out);
        assert_eq!((pool.pooled_bytes(), pool.free_count(20)), (0, 0));
        // A hold that outlives its pool is harmless.
        let late = pool.hold_length(30);
        drop(pool);
        drop(late);
    }

    /// The buffer holds the pool weakly: returned while the pool lives, freed (no panic, nothing
    /// kept) once it is gone.
    #[test]
    fn a_buffer_outliving_its_pool_is_freed_and_pins_nothing() {
        let pool = TensorPool::new();
        let b = pool.take(10);
        assert_eq!(
            Arc::strong_count(&pool),
            1,
            "a buffer keeps no strong reference"
        );
        drop(b);
        assert_eq!(pool.free_count(10), 1, "returned while the pool lives");
        let kept = pool.take(10);
        let weak = Arc::downgrade(&pool);
        drop(pool);
        assert!(
            weak.upgrade().is_none(),
            "the pool is gone with a buffer still out"
        );
        assert_eq!(kept.as_slice().len(), 0);
        drop(kept);
    }

    #[test]
    fn preprocessors_unused_for_the_idle_time_are_dropped() {
        let mut pres = Preprocessors::default();
        let a = PrepSpec::new(32, 32, ResizeMode::Letterbox, None);
        let b = PrepSpec::new(64, 48, ResizeMode::Stretch, None);
        let t0 = Instant::now();
        pres.get(a, t0);
        pres.get(b, t0 + Duration::from_secs(5 * 60));
        assert_eq!(pres.len(), 2);
        // `b` 5 minutes unused, `a` in use: both stay.
        pres.get(a, t0 + IDLE_DROP);
        assert_eq!(pres.len(), 2);
        // `b` unused for the idle time when the next job comes: dropped.
        pres.get(a, t0 + Duration::from_secs(5 * 60) + IDLE_DROP);
        assert_eq!(pres.len(), 1);
        assert!(pres.map.contains_key(&a));
        // The spec asked for is never the one dropped, however long it was unused.
        pres.get(a, t0 + 10 * IDLE_DROP);
        assert_eq!(pres.len(), 1);
        // A dropped geometry is rebuilt on demand and prepares as before.
        let img = RgbImage {
            width: 80,
            height: 60,
            rgb: vec![90; 80 * 60 * 3],
        };
        let want = b.preprocessor().run(&img.rgb, 80, 60).unwrap().0.to_vec();
        let got = pres
            .get(b, t0 + 10 * IDLE_DROP)
            .run(&img.rgb, 80, 60)
            .unwrap()
            .0
            .to_vec();
        assert_eq!(got, want);
        assert_eq!(pres.len(), 2);
        pres.remove(&b);
        assert_eq!(pres.len(), 1);
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
