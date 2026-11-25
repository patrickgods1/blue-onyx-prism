//! The shared core of saving a run's results into `benchmark.json`, used by the CLI
//! (`benchmark`, one save of the whole run) and the service (one save per finished model,
//! BM-11): from the results loaded for the merge to the file written.
//!
//! [`commit`] does, in this order: the warning for a merge that drops earlier results, the
//! history seeds from the file before this save, the merge itself (the caller's rule), the
//! "Your cameras" snapshot stamp, the regrade of every kept model like this run's, and the write.
//! The callers keep what differs between them: the results lock ([`super::results_io::lock`]),
//! [`super::report::BenchmarkResults::load_for_merge`] and its warnings, whether and when the
//! snapshot is written ([`super::cameras::write_snapshot`]), the history record
//! ([`super::history::record`]), the stored predictions ([`super::search::save_beside`]) and
//! timing logs.

use anyhow::Result;
use std::path::Path;
use std::time::{Duration, Instant};

use super::BenchmarkResults;
use super::cameras::snapshot::SnapshotId;
use super::grade::{AccuracyMetric, Weights};
use super::history::HistoryEntry;

/// What [`commit`] saved, and what the caller still needs.
#[derive(Debug)]
pub(crate) struct Committed {
    /// The results as saved (with this run's evaluation predictions, which are not serialized).
    pub merged: BenchmarkResults,
    /// History entries of the models in the file before this save, seeding the history of
    /// models that have none yet ([`super::history::seed_entries`]).
    pub seeds: Vec<HistoryEntry>,
    /// The merge dropped the earlier results ([`BenchmarkResults::merge_warning`]).
    pub merge_warning: Option<String>,
    /// How long the write of `benchmark.json` took (the rest of [`commit`] is the merge).
    pub write: Duration,
}

/// Merge `new` into `prev` (the file at `path` as loaded under the results lock) with `merge`,
/// stamp the snapshot `snapshot` (written before), regrade every model on `grade` and save the
/// result to `path`. The caller holds the results lock.
pub(crate) fn commit(
    path: &Path,
    prev: Option<BenchmarkResults>,
    new: BenchmarkResults,
    merge: impl FnOnce(Option<BenchmarkResults>, BenchmarkResults) -> BenchmarkResults,
    snapshot: Option<&SnapshotId>,
    grade: (AccuracyMetric, Weights),
) -> Result<Committed> {
    let merge_warning = BenchmarkResults::merge_warning(prev.as_ref(), &new);
    // The results before this save seed the history of models that have none yet.
    let seeds = prev
        .as_ref()
        .map(super::history::seed_entries)
        .unwrap_or_default();
    let mut merged = merge(prev, new);
    if let Some(id) = snapshot {
        super::cameras::snapshot::stamp(&mut merged, id);
    }
    // Models kept from earlier runs are graded like this run's.
    let (metric, weights) = grade;
    merged.regrade(metric, weights);
    let t = Instant::now();
    merged.save(path)?;
    Ok(Committed {
        merged,
        seeds,
        merge_warning,
        write: t.elapsed(),
    })
}
