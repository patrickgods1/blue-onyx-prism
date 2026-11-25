//! Whether a failed training attempt restarts, and how ([`next_attempt`], pure): the
//! out-of-memory rules of a run (CLAUDE.md "Protection"). Out of system memory (commit): the
//! next attempt has half the data loader workers and the same batch (the worker count does not
//! change the result), at most [`OOM_RETRIES`] times and while there are workers to halve. Out of
//! GPU memory: half the batch (down to 1), at most [`OOM_RETRIES`] times and only before
//! [`OOM_RETRY_EPOCHS`] finished epochs (later, the batch had fitted). Neither restarts a run
//! whose checkpoints are worth more than a restart ([`super::keeps_checkpoints`]) or one that was
//! cancelled. A system memory failure is judged first: a run out of both restarts with fewer
//! workers.

use super::{OOM_RETRIES, OOM_RETRY_EPOCHS};

/// Where the attempts of a run stand: what the next one runs with, and the restarts so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Tries {
    pub batch: u32,
    pub workers: u32,
    /// Restarts for running out of GPU memory.
    pub gpu_retries: u32,
    /// Restarts for running out of system memory.
    pub host_retries: u32,
}

/// How an attempt ended, as far as restarting goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Ended {
    /// Its output names an out-of-system-memory failure ([`super::host_out_of_memory_line`]).
    pub host_oom: bool,
    /// Its output names an out-of-GPU-memory failure ([`super::out_of_memory_line`]).
    pub gpu_oom: bool,
    pub cancelled: bool,
    /// Its checkpoints are kept rather than restarted over ([`super::keeps_checkpoints`]).
    pub keep: bool,
    /// Epochs it finished.
    pub epochs: usize,
}

/// A restart of a failed attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Retry {
    /// Out of system memory: `from` workers become `to`, the batch stays.
    FewerWorkers { from: u32, to: u32 },
    /// Out of GPU memory: the batch `from` becomes `to`.
    SmallerBatch { from: u32, to: u32 },
}

/// The restart after an attempt that ended as `end` with `at`, or None (the run fails).
pub(super) fn next_attempt(end: &Ended, at: &Tries) -> Option<Retry> {
    if end.host_oom
        && !end.cancelled
        && !end.keep
        && at.host_retries < OOM_RETRIES
        && at.workers > 0
    {
        Some(Retry::FewerWorkers {
            from: at.workers,
            to: at.workers / 2,
        })
    } else if end.gpu_oom
        && !end.cancelled
        && at.gpu_retries < OOM_RETRIES
        && !end.keep
        && at.batch > 1
        && end.epochs < OOM_RETRY_EPOCHS
    {
        Some(Retry::SmallerBatch {
            from: at.batch,
            to: (at.batch / 2).max(1),
        })
    } else {
        None
    }
}

impl Retry {
    /// The run log's, the report's and the progress note's sentence.
    pub fn message(&self) -> String {
        match *self {
            Retry::FewerWorkers { from, to } => format!(
                "out of system memory with {}, retrying with {to}",
                super::super::workers_text(from)
            ),
            Retry::SmallerBatch { from, to } => {
                format!("out of GPU memory at batch {from}, retrying with batch {to}")
            }
        }
    }

    /// The batch the restart trains at, when it changes ([`super::TrainReport::Batch`]).
    pub fn new_batch(&self) -> Option<u32> {
        match *self {
            Retry::FewerWorkers { .. } => None,
            Retry::SmallerBatch { to, .. } => Some(to),
        }
    }

    /// `at` after this restart.
    pub fn next(&self, at: Tries) -> Tries {
        match *self {
            Retry::FewerWorkers { to, .. } => Tries {
                workers: to,
                host_retries: at.host_retries + 1,
                ..at
            },
            Retry::SmallerBatch { to, .. } => Tries {
                batch: to,
                gpu_retries: at.gpu_retries + 1,
                ..at
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: Tries = Tries {
        batch: 16,
        workers: 4,
        gpu_retries: 0,
        host_retries: 0,
    };

    fn host() -> Ended {
        Ended {
            host_oom: true,
            ..Ended::default()
        }
    }

    fn gpu(epochs: usize) -> Ended {
        Ended {
            gpu_oom: true,
            epochs,
            ..Ended::default()
        }
    }

    #[test]
    fn the_rules_of_a_restart() {
        let workers = |from, to| Some(Retry::FewerWorkers { from, to });
        let batch = |from, to| Some(Retry::SmallerBatch { from, to });
        let at = |batch, workers, gpu_retries, host_retries| Tries {
            batch,
            workers,
            gpu_retries,
            host_retries,
        };
        let cancelled = |e: Ended| Ended {
            cancelled: true,
            ..e
        };
        let kept = |e: Ended| Ended { keep: true, ..e };
        let table: [(&str, Ended, Tries, Option<Retry>); 22] = [
            ("no memory failure", Ended::default(), START, None),
            (
                "no memory failure, epochs",
                Ended {
                    epochs: 1,
                    ..Ended::default()
                },
                START,
                None,
            ),
            // System memory: half the workers, the batch stays.
            ("host: halves the workers", host(), START, workers(4, 2)),
            (
                "host: 1 worker to 0",
                host(),
                at(16, 1, 0, 0),
                workers(1, 0),
            ),
            (
                "host: 3 workers to 1",
                host(),
                at(16, 3, 0, 2),
                workers(3, 1),
            ),
            ("host: no worker left", host(), at(16, 0, 0, 0), None),
            ("host: retries used up", host(), at(16, 4, 0, 3), None),
            (
                "host: GPU retries do not count",
                host(),
                at(16, 4, 3, 0),
                workers(4, 2),
            ),
            ("host: cancelled", cancelled(host()), START, None),
            ("host: checkpoints kept", kept(host()), START, None),
            // Epochs alone do not stop it: the kept checkpoints do.
            (
                "host: after epochs",
                Ended {
                    epochs: 5,
                    ..host()
                },
                START,
                workers(4, 2),
            ),
            // GPU memory: half the batch, only early.
            ("gpu: halves the batch", gpu(0), START, batch(16, 8)),
            ("gpu: 3 to 1", gpu(0), at(3, 4, 0, 0), batch(3, 1)),
            ("gpu: 2 to 1", gpu(2), at(2, 4, 2, 0), batch(2, 1)),
            ("gpu: batch 1", gpu(0), at(1, 4, 0, 0), None),
            ("gpu: retries used up", gpu(0), at(16, 4, 3, 0), None),
            ("gpu: 3 epochs finished", gpu(OOM_RETRY_EPOCHS), START, None),
            ("gpu: cancelled", cancelled(gpu(0)), START, None),
            ("gpu: checkpoints kept", kept(gpu(0)), START, None),
            // Both: fewer workers first; with no worker to halve, the batch.
            (
                "both: workers first",
                Ended {
                    gpu_oom: true,
                    ..host()
                },
                START,
                workers(4, 2),
            ),
            (
                "both, no worker: the batch",
                Ended {
                    gpu_oom: true,
                    ..host()
                },
                at(16, 0, 0, 0),
                batch(16, 8),
            ),
            (
                "both, host retries used up: the batch",
                Ended {
                    gpu_oom: true,
                    ..host()
                },
                at(16, 4, 0, 3),
                batch(16, 8),
            ),
        ];
        for (name, end, tries, want) in table {
            assert_eq!(next_attempt(&end, &tries), want, "{name}");
        }
    }

    #[test]
    fn a_restart_moves_its_own_counter_and_says_what_it_does() {
        let r = next_attempt(&host(), &START).unwrap();
        assert_eq!(
            r.message(),
            "out of system memory with 4 data loader workers, retrying with 2"
        );
        assert_eq!(r.new_batch(), None);
        assert_eq!(
            r.next(START),
            Tries {
                workers: 2,
                host_retries: 1,
                ..START
            }
        );
        let r = next_attempt(
            &host(),
            &Tries {
                workers: 1,
                ..START
            },
        )
        .unwrap();
        assert_eq!(
            r.message(),
            "out of system memory with 1 data loader worker, retrying with 0"
        );
        let r = next_attempt(&gpu(1), &START).unwrap();
        assert_eq!(
            r.message(),
            "out of GPU memory at batch 16, retrying with batch 8"
        );
        assert_eq!(r.new_batch(), Some(8));
        assert_eq!(
            r.next(START),
            Tries {
                batch: 8,
                gpu_retries: 1,
                ..START
            }
        );
    }

    #[test]
    fn restarts_stop_after_the_last_retry() {
        // Out of GPU memory every time from batch 64: 32, 16, 8, then the run fails.
        let mut at = Tries { batch: 64, ..START };
        let mut batches = Vec::new();
        while let Some(r) = next_attempt(&gpu(0), &at) {
            at = r.next(at);
            batches.push(at.batch);
        }
        assert_eq!(batches, [32, 16, 8]);
        assert_eq!(at.gpu_retries, OOM_RETRIES);
        // Out of system memory every time from 8 workers: 4, 2, 1, then the run fails.
        let mut at = Tries {
            workers: 8,
            ..START
        };
        let mut workers = Vec::new();
        while let Some(r) = next_attempt(&host(), &at) {
            at = r.next(at);
            workers.push(at.workers);
        }
        assert_eq!(workers, [4, 2, 1]);
        assert_eq!(at.batch, START.batch);
    }
}
