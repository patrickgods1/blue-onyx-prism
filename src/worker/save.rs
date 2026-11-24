//! `save_image_path` off the detection path: after a request is answered, its frame and
//! predictions go to one writer thread (`image-save`) that draws the boxes, encodes the
//! annotated JPEG and writes it ([`crate::image::save_annotated`], through a temporary file and a
//! rename, so a stop or crash mid-write never leaves a truncated JPEG); the reply never waits for
//! it, so `analysisRoundTripMs` holds no encode. The collector's pattern: a bounded queue fed by
//! `try_send`; a full queue drops the save, counted, with a warning at most once per
//! [`crate::metrics::LOG_THROTTLE_INTERVAL`]. A failed write (a folder the service cannot write,
//! a full disk, a share that is gone) is warned at most once per that interval too, by the
//! thread's own throttle, with the error of the save that gets through and the count held back:
//! such a write fails at once, so the queue never fills and every frame would log.
//!
//! [`QUEUE`] bounds the memory: each queued save holds its decoded frame (12.3 MB for 2688x1520).
//! The thread keeps the normal priority: a lowered one could starve under CPU inference and
//! drop every save.
//!
//! Stop: [`flush`] (called by `ModelRegistry::shutdown` once its workers are gone, so their last
//! saves are queued) closes the queue and waits, until a deadline, for the thread to write what
//! is queued and end; what is still queued then is logged as lost, never silently. The next save
//! after a flush starts a new thread (the Windows service restarts in process).

use crate::api::Prediction;
use crate::image::RgbImage;
use crate::metrics::LogThrottle;
use crate::sync::LockExt;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{debug, error, warn};

/// Saves waiting for the writer. Four 2688x1520 frames are about 50 MB; a camera system that
/// triggers faster than one thread encodes loses saves instead of growing memory.
pub(super) const QUEUE: usize = 4;

/// One answered request's save.
#[derive(Debug)]
pub(super) struct SaveJob {
    pub(super) model: String,
    pub(super) dir: PathBuf,
    pub(super) image_name: String,
    /// The decoded frame as received (owned: it outlives the request).
    pub(super) img: RgbImage,
    pub(super) predictions: Vec<Prediction>,
    /// `save_ref_image`: also write the unannotated frame.
    pub(super) save_ref: bool,
    /// The upload's encoded bytes, written as the reference when they are a JPEG (None: the
    /// frame is encoded instead).
    pub(super) upload: Option<bytes::Bytes>,
}

/// What [`Saver::submit`] did with a save.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Submitted {
    Queued,
    /// The queue was full (or the writer gone); `warned` when this drop logged the warning.
    Dropped {
        warned: bool,
    },
}

/// The writer's queue.
#[derive(Debug)]
pub(crate) struct Saver {
    tx: Sender<SaveJob>,
    dropped: AtomicU64,
    log: LogThrottle,
    clock: Instant,
    /// Saves queued and not written yet (the writer counts down after each write).
    queued: Arc<AtomicUsize>,
}

impl Saver {
    /// A saver with a queue of `capacity` and the receiving end (no thread).
    pub(super) fn unstarted(capacity: usize) -> (Self, Receiver<SaveJob>) {
        let (tx, rx) = crossbeam_channel::bounded(capacity);
        let saver = Self {
            tx,
            dropped: AtomicU64::new(0),
            log: LogThrottle::default(),
            clock: Instant::now(),
            queued: Arc::default(),
        };
        (saver, rx)
    }

    /// Queue `job` (never blocks).
    pub(super) fn submit(&self, job: SaveJob) -> Submitted {
        let model = job.model.clone();
        // Counted before the send: the writer may write it (and count down) at once.
        self.queued.fetch_add(1, Ordering::AcqRel);
        match self.tx.try_send(job) {
            Ok(()) => Submitted::Queued,
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.queued.fetch_sub(1, Ordering::AcqRel);
                self.dropped.fetch_add(1, Ordering::Relaxed);
                let now = u64::try_from(self.clock.elapsed().as_millis()).unwrap_or(u64::MAX);
                let held_back = self.log.allow(now);
                if let Some(held_back) = held_back {
                    warn!(
                        model = %model,
                        queue = QUEUE,
                        similar_since_last_warning = held_back,
                        "annotated image not saved: the image-save queue is full (images arrive \
                         faster than they are encoded and written); the detection was answered; \
                         check the disk under save_image_path, or turn save_image_path off"
                    );
                }
                Submitted::Dropped {
                    warned: held_back.is_some(),
                }
            }
        }
    }

    /// Saves dropped so far.
    #[cfg(test)]
    pub(super) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// A [`Saver`] with its `image-save` thread.
#[derive(Debug)]
struct Writer {
    saver: Saver,
    thread: JoinHandle<()>,
}

/// How [`flush`] ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flushed {
    /// Every queued save was written and the thread ended (or none was running).
    Drained,
    /// The deadline passed first: this many saves were still queued or being written (lost
    /// when the process exits; the thread is left to finish on its own).
    Left(usize),
}

/// How often [`Writer::finish`] looks at the thread.
const FLUSH_POLL: Duration = Duration::from_millis(10);

impl Writer {
    /// A saver whose `image-save` thread writes every queued save (None: no thread).
    fn start(capacity: usize) -> Option<Self> {
        let (saver, rx) = Saver::unstarted(capacity);
        let queued = saver.queued.clone();
        let spawned = std::thread::Builder::new()
            .name("image-save".into())
            .spawn(move || {
                let failures = LogThrottle::default();
                let clock = Instant::now();
                while let Ok(job) = rx.recv() {
                    write_logged(job, &failures, clock);
                    queued.fetch_sub(1, Ordering::AcqRel);
                }
            });
        match spawned {
            Ok(thread) => Some(Self { saver, thread }),
            Err(e) => {
                warn!(
                    "could not start the image-save thread ({e}); save_image_path images are \
                     not written; the system is out of threads or memory; restart the service"
                );
                None
            }
        }
    }

    /// Close the queue and wait, until `deadline`, for the thread to write what is queued.
    fn finish(self, deadline: Instant) -> Flushed {
        let Self { saver, thread } = self;
        let queued = saver.queued.clone();
        // The thread's `recv` ends once the queue is empty and this, the only sender, is gone.
        drop(saver);
        while !thread.is_finished() && Instant::now() < deadline {
            std::thread::sleep(FLUSH_POLL);
        }
        if !thread.is_finished() {
            let left = queued.load(Ordering::Acquire);
            warn!(
                left,
                "{left} annotated image(s) for save_image_path were not written before the stop \
                 deadline (the disk is slow or full); they are lost; check the disk under \
                 save_image_path"
            );
            return Flushed::Left(left);
        }
        if thread.join().is_err() {
            error!("the image-save thread panicked; queued save_image_path images were lost");
        } else {
            debug!("image-save thread stopped; every queued image was written");
        }
        Flushed::Drained
    }
}

/// The process-wide writer.
#[derive(Debug)]
enum Global {
    /// Not started yet, or flushed: the next save starts it.
    Idle,
    Running(Writer),
    /// Its thread could not start (warned once); saves are dropped until the next flush.
    Failed,
}

static GLOBAL: Mutex<Global> = Mutex::new(Global::Idle);

/// Queue `job` on `to`, else on the process-wide saver (started on first use).
pub(super) fn submit(to: Option<&Saver>, job: SaveJob) -> Submitted {
    if let Some(s) = to {
        return s.submit(job);
    }
    let mut g = GLOBAL.lock_recover();
    if matches!(*g, Global::Idle) {
        *g = Writer::start(QUEUE).map_or(Global::Failed, Global::Running);
    }
    match &*g {
        Global::Running(w) => w.saver.submit(job),
        Global::Idle | Global::Failed => Submitted::Dropped { warned: false },
    }
}

/// Stop the process-wide saver: write what is queued, until `deadline` (see the module doc).
/// Saves submitted meanwhile go to a new writer.
pub(crate) fn flush(deadline: Instant) -> Flushed {
    let taken = std::mem::replace(&mut *GLOBAL.lock_recover(), Global::Idle);
    match taken {
        Global::Running(w) => w.finish(deadline),
        Global::Idle | Global::Failed => Flushed::Drained,
    }
}

/// Draw, encode and write one save (a warning on every failure; the writer thread uses
/// [`write_logged`]). Tests write what they took off a queue with it.
#[cfg(test)]
pub(super) fn write(job: SaveJob) {
    write_logged(job, &LogThrottle::new(Duration::ZERO), Instant::now());
}

/// Draw, encode and write one save; a failure is warned when `failures` lets it through (on the
/// millisecond clock that started at `clock`), with the count of failures held back since the
/// last warning. `Some(held back)` when this save failed and warned. The file name: the
/// request's image name, or a random one for an unnamed upload (`image.jpg`, Blue Iris'
/// default).
pub(super) fn write_logged(job: SaveJob, failures: &LogThrottle, clock: Instant) -> Option<u64> {
    let SaveJob {
        model,
        dir,
        image_name,
        img,
        predictions,
        save_ref,
        upload,
    } = job;
    let file_name = if image_name.is_empty() || image_name == "image.jpg" {
        format!("{}.jpg", uuid::Uuid::new_v4())
    } else {
        image_name
    };
    if let Err(e) = crate::image::save_annotated(
        &dir,
        &file_name,
        &img,
        &predictions,
        save_ref,
        upload.as_deref(),
    ) {
        let now = u64::try_from(clock.elapsed().as_millis()).unwrap_or(u64::MAX);
        let held_back = failures.allow(now)?;
        warn!(
            model = %model,
            similar_since_last_warning = held_back,
            "could not save annotated image: {e:#}; check that save_image_path exists, is \
             writable by the service and has free space, or turn save_image_path off"
        );
        return Some(held_back);
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn job(dir: &std::path::Path) -> SaveJob {
        SaveJob {
            model: "m".into(),
            dir: dir.to_path_buf(),
            image_name: "cam1.jpg".into(),
            img: RgbImage {
                width: 8,
                height: 8,
                rgb: vec![100; 8 * 8 * 3],
            },
            predictions: Vec::new(),
            save_ref: false,
            upload: None,
        }
    }

    /// A full queue drops the save, counted; only the first drop of an interval warns.
    #[test]
    fn a_full_queue_drops_with_one_warning() {
        let dir = crate::test_dirs::TempDir::unmade("save-test");
        let (saver, rx) = Saver::unstarted(1);
        assert_eq!(saver.submit(job(&dir)), Submitted::Queued);
        assert_eq!(saver.submit(job(&dir)), Submitted::Dropped { warned: true });
        assert_eq!(
            saver.submit(job(&dir)),
            Submitted::Dropped { warned: false },
            "rate-limited"
        );
        assert_eq!(saver.dropped(), 2);
        // The queued one is written by whoever drains the queue.
        write(rx.try_recv().unwrap());
        assert!(dir.join("cam1_od.jpg").is_file());
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            ["cam1_od.jpg"],
            "written in place by a rename, no temporary file"
        );
        assert!(rx.try_recv().is_err(), "the dropped ones are gone");
        // A writer that is gone drops too.
        drop(rx);
        assert!(matches!(saver.submit(job(&dir)), Submitted::Dropped { .. }));
    }

    fn files_in(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .map(|d| {
                d.map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    /// RA-075: a stop writes what is queued and ends the thread (before, the thread was
    /// detached and queued saves were lost without a word).
    #[test]
    fn finish_writes_the_queue_and_ends_the_thread() {
        let dir = crate::test_dirs::TempDir::unmade("save-flush");
        let w = Writer::start(4).unwrap();
        for name in ["a.jpg", "b.jpg", "c.jpg"] {
            let j = SaveJob {
                image_name: name.into(),
                ..job(&dir)
            };
            assert_eq!(w.saver.submit(j), Submitted::Queued);
        }
        let queued = w.saver.queued.clone();
        assert_eq!(
            w.finish(Instant::now() + Duration::from_secs(30)),
            Flushed::Drained
        );
        assert_eq!(files_in(&dir), ["a_od.jpg", "b_od.jpg", "c_od.jpg"]);
        assert_eq!(queued.load(Ordering::Acquire), 0);
    }

    /// Past the deadline the saves still queued are reported (and logged), and the thread is
    /// left to finish on its own.
    #[test]
    fn finish_past_the_deadline_reports_what_is_left() {
        let dir = crate::test_dirs::TempDir::unmade("save-late");
        let w = Writer::start(4).unwrap();
        // Large frames: encoding them takes long enough that none is done at once.
        let big = || RgbImage {
            width: 2000,
            height: 2000,
            rgb: vec![90; 2000 * 2000 * 3],
        };
        for name in ["a.jpg", "b.jpg", "c.jpg"] {
            let j = SaveJob {
                image_name: name.into(),
                img: big(),
                ..job(&dir)
            };
            assert_eq!(w.saver.submit(j), Submitted::Queued);
        }
        let queued = w.saver.queued.clone();
        let Flushed::Left(n) = w.finish(Instant::now()) else {
            panic!("drained at once");
        };
        assert!((1..=3).contains(&n), "{n}");
        // The thread goes on writing them.
        let t = Instant::now();
        while queued.load(Ordering::Acquire) > 0 {
            assert!(t.elapsed() < Duration::from_secs(60), "never written");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// The process-wide saver starts again after a flush (the Windows service restarts in
    /// process).
    #[test]
    fn the_global_saver_starts_again_after_a_flush() {
        let dir = crate::test_dirs::TempDir::unmade("save-global");
        let deadline = || Instant::now() + Duration::from_secs(30);
        // Another test's registry shutdown may flush the global saver at the same time, and
        // then writes this test's save itself: wait for the file rather than for this flush.
        let written = |want: &[&str]| {
            let t = Instant::now();
            while files_in(&dir) != want {
                assert!(
                    t.elapsed() < Duration::from_secs(30),
                    "{:?}",
                    files_in(&dir)
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        assert_eq!(submit(None, job(&dir)), Submitted::Queued);
        assert_eq!(flush(deadline()), Flushed::Drained);
        written(&["cam1_od.jpg"]);
        let again = SaveJob {
            image_name: "cam2.jpg".into(),
            ..job(&dir)
        };
        assert_eq!(submit(None, again), Submitted::Queued, "a new writer");
        assert_eq!(flush(deadline()), Flushed::Drained);
        written(&["cam1_od.jpg", "cam2_od.jpg"]);
    }

    /// RA-110: a save that fails at once (here: its folder is a file) warns once per interval,
    /// not per frame, and the next warning counts the ones held back.
    #[test]
    fn failed_writes_warn_once_per_interval() {
        let root = crate::test_dirs::TempDir::new("save-fail");
        let dir = root.join("blocker");
        std::fs::write(&dir, b"not a folder").unwrap();
        let throttle = LogThrottle::default();
        let clock = Instant::now();
        let warned: Vec<Option<u64>> = (0..10)
            .map(|_| write_logged(job(&dir), &throttle, clock))
            .collect();
        assert_eq!(warned.first(), Some(&Some(0)), "{warned:?}");
        assert!(warned.iter().skip(1).all(Option::is_none), "{warned:?}");
        assert_eq!(throttle.held_back(), 9);
        // A write that works logs nothing and holds nothing back.
        let ok = crate::test_dirs::TempDir::unmade("save-ok");
        assert_eq!(write_logged(job(&ok), &throttle, clock), None);
        assert_eq!(throttle.held_back(), 9);
        // Past the interval (a clock started that long ago), the next failure reports the 9.
        let earlier = Instant::now()
            .checked_sub(crate::metrics::LOG_THROTTLE_INTERVAL * 2)
            .unwrap();
        assert_eq!(write_logged(job(&dir), &throttle, earlier), Some(9));
    }
}
