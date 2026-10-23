//! Request bookkeeping of the run loop: where each request is between intake and reply
//! ([`Stage`], [`Job`]), the inference in flight per slot ([`InFlight`]) and the reply order
//! ([`Done`]). Invariant: requests leave `pending` in FIFO order and are answered in the order
//! they entered `order`, so replies keep the request order. Pure data: no I/O, no backend calls.

use super::{LiveSettings, Reply};
use crate::api::VisionDetectionResponse;
use crate::backend::Completion;
use crate::model::PreprocessCtx;
use crate::prep::Prepared;
use anyhow::Result;
use crossbeam_channel::Receiver;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Where a request is between intake and reply.
pub(super) enum Stage {
    /// Waiting for the prep pool.
    Prep(Receiver<Result<Prepared>>),
    /// Decoded and preprocessed, waiting for a free slot.
    Ready(Prepared),
    /// Already answered (dropped or failed); replied in order.
    Answer(VisionDetectionResponse),
}

/// Who asked and when.
pub(super) struct Meta {
    pub(super) reply: Reply,
    pub(super) enqueued: Instant,
    pub(super) image_name: String,
    pub(super) min_confidence: f32,
    /// The live settings when the request arrived (used for its whole life).
    pub(super) live: Arc<LiveSettings>,
    /// Short id in every log line of this request (see [`super::new_request_id`]).
    pub(super) request_id: String,
    /// What is left of the timeout of a request forwarded by another worker, counted from
    /// `enqueued` (None: this worker's own request timeout).
    pub(super) budget: Option<Duration>,
    /// May still be forwarded to the night model (false once tried, and for requests that
    /// were forwarded here: those are never forwarded again).
    pub(super) forwardable: bool,
    /// The model that forwarded this request here (None: it came from the HTTP layer).
    pub(super) routed_from: Option<String>,
}

impl Meta {
    /// The request timeout of this request: the live one, capped by a forwarded budget.
    pub(super) fn timeout(&self) -> Duration {
        let own = self.live.request_timeout;
        self.budget.map_or(own, |b| b.min(own))
    }
}

pub(super) struct Job {
    pub(super) meta: Meta,
    pub(super) stage: Stage,
}

/// A request whose inference was submitted into a slot.
pub(super) struct InFlight {
    pub(super) meta: Meta,
    pub(super) ctx: PreprocessCtx,
    pub(super) img: Option<crate::image::RgbImage>,
    pub(super) stats: crate::image_stats::ImageStats,
    pub(super) decode: Duration,
    pub(super) preprocess: Duration,
    pub(super) seq: u64,
    pub(super) submitted: Instant,
    /// When the backend reported it finished.
    pub(super) done: Option<Instant>,
    /// No completion arrived within the request timeout: collected under the stuck watchdog.
    pub(super) overdue: bool,
}

/// Replies in request order: a slot to collect, or an answer decided earlier.
pub(super) enum Done {
    Slot(usize),
    Answer(Box<(Meta, VisionDetectionResponse)>),
}

/// The worker's requests between intake and reply. Requests leave `pending` in FIFO order and
/// are answered in the order they entered `order`, so replies keep the request order.
#[derive(Default)]
pub(super) struct Pipeline {
    pub(super) pending: VecDeque<Job>,
    pub(super) order: VecDeque<Done>,
    pub(super) slots: Vec<Option<InFlight>>,
    /// Round-robin start for the next free slot.
    pub(super) next: usize,
}

impl Pipeline {
    pub(super) fn resize(&mut self, slots: usize) {
        self.slots.resize_with(slots.max(1), || None);
    }

    pub(super) fn free_slot(&mut self) -> Option<usize> {
        let n = self.slots.len();
        let i = (0..n)
            .map(|k| (self.next + k) % n)
            .find(|&i| self.slots.get(i).is_some_and(Option::is_none))?;
        self.next = (i + 1) % n;
        Some(i)
    }

    pub(super) fn busy(&self) -> bool {
        self.slots.iter().any(Option::is_some)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.order.is_empty() && !self.busy()
    }

    /// Record a backend completion (ignores stale ones, e.g. of the warm-up).
    pub(super) fn complete(&mut self, c: Completion) -> bool {
        match self.slots.get_mut(c.slot) {
            Some(Some(f)) if f.seq == c.seq && f.done.is_none() => {
                f.done = Some(c.at);
                true
            }
            _ => false,
        }
    }
}

impl InFlight {
    pub(super) fn min_confidence(&self) -> f32 {
        self.meta.min_confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_flight(seq: u64) -> InFlight {
        let (reply, _rx) = tokio::sync::oneshot::channel();
        InFlight {
            meta: Meta {
                reply: Reply::Client(reply),
                enqueued: Instant::now(),
                image_name: String::new(),
                min_confidence: 0.0,
                live: Arc::new(LiveSettings::default()),
                request_id: String::new(),
                budget: None,
                forwardable: true,
                routed_from: None,
            },
            ctx: PreprocessCtx {
                orig_w: 1,
                orig_h: 1,
                input_w: 1,
                input_h: 1,
                mode: crate::model::ResizeMode::Letterbox,
                scale: 1.0,
                pad_x: 0.0,
                pad_y: 0.0,
            },
            img: None,
            stats: crate::image_stats::ImageStats::default(),
            decode: Duration::ZERO,
            preprocess: Duration::ZERO,
            seq,
            submitted: Instant::now(),
            done: None,
            overdue: false,
        }
    }

    #[test]
    fn pipeline_slots_round_robin_and_completions() {
        let mut p = Pipeline::default();
        p.resize(0);
        assert_eq!(p.slots.len(), 1, "at least one slot");
        p.resize(3);
        assert!(p.is_empty() && !p.busy());
        assert_eq!(p.free_slot(), Some(0));
        p.slots[0] = Some(in_flight(1));
        assert_eq!(p.free_slot(), Some(1));
        p.slots[1] = Some(in_flight(1));
        assert_eq!(p.free_slot(), Some(2));
        p.slots[2] = Some(in_flight(4));
        assert_eq!(p.free_slot(), None);
        assert!(p.busy() && !p.is_empty());
        let at = Instant::now();
        // Stale (older sequence number) and unknown slots are ignored.
        assert!(!p.complete(Completion {
            slot: 2,
            seq: 3,
            at
        }));
        assert!(!p.complete(Completion {
            slot: 9,
            seq: 1,
            at
        }));
        assert!(p.complete(Completion {
            slot: 2,
            seq: 4,
            at
        }));
        assert!(
            !p.complete(Completion {
                slot: 2,
                seq: 4,
                at
            }),
            "once"
        );
        assert_eq!(p.slots[2].as_ref().unwrap().done, Some(at));
        p.slots[0] = None;
        assert_eq!(p.free_slot(), Some(0));
    }
}
