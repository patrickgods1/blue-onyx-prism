//! `save_image_path` off the detection path: after a request is answered, its frame and
//! predictions go to one writer thread (`image-save`) that draws the boxes, encodes the
//! annotated JPEG and writes it ([`crate::image::save_annotated`]); the reply never waits for
//! it, so `analysisRoundTripMs` holds no encode. The collector's pattern: a bounded queue fed by
//! `try_send`; a full queue drops the save, counted, with a warning at most once per
//! [`crate::metrics::LOG_THROTTLE_INTERVAL`].
//!
//! [`QUEUE`] bounds the memory: each queued save holds its decoded frame (12.3 MB for 2688x1520).
//! The thread keeps the normal priority: a lowered one could starve under CPU inference and
//! drop every save.

use crate::api::Prediction;
use crate::image::RgbImage;
use crate::metrics::LogThrottle;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::warn;

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
        };
        (saver, rx)
    }

    /// A saver whose `image-save` thread writes every queued save (None: no thread).
    fn spawn(capacity: usize) -> Option<Self> {
        let (saver, rx) = Self::unstarted(capacity);
        let spawned = std::thread::Builder::new()
            .name("image-save".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    write(job);
                }
            });
        match spawned {
            Ok(_) => Some(saver),
            Err(e) => {
                warn!(
                    "could not start the image-save thread ({e}); save_image_path images are \
                     not written; restart the service"
                );
                None
            }
        }
    }

    /// Queue `job` (never blocks).
    pub(super) fn submit(&self, job: SaveJob) -> Submitted {
        let model = job.model.clone();
        match self.tx.try_send(job) {
            Ok(()) => Submitted::Queued,
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
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

/// The process-wide saver, started on first use (None: its thread could not start).
fn global() -> Option<&'static Saver> {
    static SAVER: OnceLock<Option<Saver>> = OnceLock::new();
    SAVER.get_or_init(|| Saver::spawn(QUEUE)).as_ref()
}

/// Queue `job` on `to`, else on the process-wide saver.
pub(super) fn submit(to: Option<&Saver>, job: SaveJob) -> Submitted {
    match to {
        Some(s) => s.submit(job),
        None => match global() {
            Some(s) => s.submit(job),
            None => Submitted::Dropped { warned: false },
        },
    }
}

/// Draw, encode and write one save (a warning on failure). The file name: the request's image
/// name, or a random one for an unnamed upload (`image.jpg`, Blue Iris' default).
pub(super) fn write(job: SaveJob) {
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
        warn!(model = %model, "could not save annotated image: {e:#}");
    }
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
        let dir = std::env::temp_dir().join(format!("bo_save_test_{}", uuid::Uuid::new_v4()));
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
        assert!(rx.try_recv().is_err(), "the dropped ones are gone");
        // A writer that is gone drops too.
        drop(rx);
        assert!(matches!(saver.submit(job(&dir)), Submitted::Dropped { .. }));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
