//! Side jobs: work a person started that runs beside the one-at-a-time training queue and must
//! not wait behind it (a dataset export, an X-AnyLabeling import), registered while it works so
//! the Jobs tab of the Train area ([`super::train_jobs`]) lists it with its step, percent and time
//! like a queue job. The work itself still runs where it always did (inside its request, on
//! [`super::blocking`]); a browser that leaves the page does not stop it, and the Jobs tab shows
//! it until it ends.
//!
//! One [`SideJobs`] per process: [`super::AppState::side_jobs`], which the runner hands every
//! registry generation (like the benchmark runner), so a restart neither forgets a running export
//! nor shows a stale one. Memory only, no disk: the Jobs tab asks every few seconds.
//!
//! A job registers through [`SideJobs::start`], whose guard ([`SideJob`]) removes it when dropped,
//! so an error or a panic never leaves a ghost row. Reports are cheap (one short lock); each one
//! also feeds the job's [`JobClock`], so a step's time left is estimated as for a queue job
//! ([`crate::training::eta`]).
//!
//! The mapping of an export's and an import's own progress ([`export_report`],
//! [`import_report`]) is here too: steps, their share of the whole, the detail line.

use super::train_area::DATASET;
use crate::collect::labels::export::ExportStep;
use crate::sync::LockExt;
use crate::training::eta::{JobClock, JobTiming};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What kind of side job: its name in the list and where it lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SideKind {
    /// `POST /v1/review/export`.
    DatasetExport,
    /// `POST /v1/review/import`.
    XanyImport,
}

impl SideKind {
    /// The kind column of the Jobs tab.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::DatasetExport => "Dataset export",
            Self::XanyImport => "X-AnyLabeling import",
        }
    }

    /// The Train tab with the job's form.
    pub(crate) const fn tab(self) -> &'static str {
        DATASET
    }

    /// The form's card on that tab (an `id` in `templates/train/dataset.html`).
    pub(crate) const fn anchor(self) -> &'static str {
        match self {
            Self::DatasetExport => "export",
            Self::XanyImport => "xany-import",
        }
    }
}

/// One progress report of a side job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Report {
    /// The step's name, lower case: `writing files`.
    pub(crate) stage: &'static str,
    /// 1-based.
    pub(crate) step: usize,
    pub(crate) steps: usize,
    /// Of this step: what its time left is measured from.
    pub(crate) step_percent: u8,
    /// Of the whole job.
    pub(crate) percent: u8,
    /// `1200 of 5000 files written`.
    pub(crate) detail: String,
}

#[derive(Debug)]
struct Entry {
    kind: SideKind,
    title: String,
    report: Option<Report>,
    clock: JobClock,
}

/// What the Jobs tab shows of a side job at one moment ([`SideJobs::views`]).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SideView {
    /// `side-<n>`: never the id of a run or a task.
    pub(crate) id: String,
    pub(crate) kind: SideKind,
    pub(crate) title: String,
    /// None before the job's first report.
    pub(crate) report: Option<Report>,
    pub(crate) timing: JobTiming,
}

/// The side jobs of the process (see the module doc).
#[derive(Debug, Default)]
pub(crate) struct SideJobs {
    next: AtomicU64,
    jobs: Mutex<BTreeMap<u64, Entry>>,
    /// Every report, in order (tests: what a job said while it ran).
    #[cfg(test)]
    seen: Mutex<Vec<(SideKind, Report)>>,
}

impl SideJobs {
    /// Register a job of `kind` titled `title`, listed until the guard is dropped.
    pub(crate) fn start(self: &Arc<Self>, kind: SideKind, title: String) -> SideJob {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        self.jobs.lock_recover().insert(
            id,
            Entry {
                kind,
                title,
                report: None,
                clock: JobClock::new(Instant::now()),
            },
        );
        SideJob {
            jobs: Arc::clone(self),
            id,
        }
    }

    /// The running side jobs at `now`, oldest first.
    pub(crate) fn views(&self, now: Instant) -> Vec<SideView> {
        self.jobs
            .lock_recover()
            .iter()
            .map(|(id, e)| SideView {
                id: format!("side-{id}"),
                kind: e.kind,
                title: e.title.clone(),
                report: e.report.clone(),
                timing: e.clock.timing(now),
            })
            .collect()
    }

    fn report(&self, id: u64, r: Report) {
        let mut jobs = self.jobs.lock_recover();
        let Some(e) = jobs.get_mut(&id) else {
            return;
        };
        e.clock.report(Instant::now(), r.step, r.step_percent);
        #[cfg(test)]
        self.seen.lock_recover().push((e.kind, r.clone()));
        e.report = Some(r);
    }

    /// Every report so far, in order.
    #[cfg(test)]
    pub(crate) fn seen(&self) -> Vec<(SideKind, Report)> {
        self.seen.lock_recover().clone()
    }
}

/// A registered side job: report through it; dropping it removes the job from the list.
#[derive(Debug)]
pub(crate) struct SideJob {
    jobs: Arc<SideJobs>,
    id: u64,
}

impl SideJob {
    pub(crate) fn report(&self, r: Report) {
        self.jobs.report(self.id, r);
    }
}

impl Drop for SideJob {
    fn drop(&mut self) {
        self.jobs.jobs.lock_recover().remove(&self.id);
    }
}

/// `done` of `of` in whole percent, 100 for nothing to do.
fn percent_of(done: usize, of: usize) -> u8 {
    if of == 0 {
        return 100;
    }
    u8::try_from(done.min(of).saturating_mul(100) / of).unwrap_or(100)
}

/// The share of the whole each export step takes, in percent: (start, end). Reading the label
/// files and placing the images are what a large export spends its time on; the publish is a
/// rename (a move per file into a folder that existed).
const EXPORT_BANDS: [(u8, u8); 3] = [(0, 25), (25, 97), (97, 100)];

/// The whole job's percent at `step_percent` of export step `step` (1-based).
fn export_overall(step: usize, step_percent: u8) -> u8 {
    let (lo, hi) = EXPORT_BANDS
        .get(step.saturating_sub(1))
        .copied()
        .unwrap_or((100, 100));
    let within = u16::from(hi - lo) * u16::from(step_percent.min(100)) / 100;
    lo.saturating_add(u8::try_from(within).unwrap_or(0))
}

/// The side-job report of an export at `s`.
pub(crate) fn export_report(s: ExportStep) -> Report {
    let (stage, step, step_percent, detail) = match s {
        ExportStep::Listing => (
            "selecting frames",
            1,
            0,
            "listing the label files".to_string(),
        ),
        ExportStep::Reading { done, of } => (
            "selecting frames",
            1,
            percent_of(done, of),
            format!("{done} of {of} label files read"),
        ),
        ExportStep::Queue => (
            "selecting frames",
            1,
            100,
            "reading the review queue's unreviewed frames".to_string(),
        ),
        ExportStep::Writing { done, of } => (
            "writing files",
            2,
            percent_of(done, of),
            format!("{done} of {of} files written"),
        ),
        ExportStep::Publishing => (
            "publishing",
            3,
            0,
            "moving the export into place".to_string(),
        ),
    };
    Report {
        stage,
        step,
        steps: EXPORT_BANDS.len(),
        step_percent,
        percent: export_overall(step, step_percent),
        detail,
    }
}

/// The side-job report of an import that looked at `done` of `of` files.
pub(crate) fn import_report(done: usize, of: usize) -> Report {
    let p = percent_of(done, of);
    Report {
        stage: "importing",
        step: 1,
        steps: 1,
        step_percent: p,
        percent: p,
        detail: format!("{done} of {of} files"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_is_listed_while_its_guard_lives_and_never_after() {
        let jobs = Arc::new(SideJobs::default());
        let a = jobs.start(SideKind::DatasetExport, "yolo to D:/x".into());
        let b = jobs.start(SideKind::XanyImport, "From D:/y".into());
        a.report(export_report(ExportStep::Writing { done: 5, of: 10 }));
        let v = jobs.views(Instant::now());
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].id, "side-1");
        assert_eq!(v[1].id, "side-2");
        assert_eq!(
            v[0].report.as_ref().unwrap().detail,
            "5 of 10 files written"
        );
        assert!(v[1].report.is_none());
        drop(a);
        let v = jobs.views(Instant::now());
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].kind, SideKind::XanyImport);
        // A panic in the work drops the guard too.
        let held = Arc::clone(&jobs);
        let r = std::panic::catch_unwind(move || {
            let _g = held.start(SideKind::DatasetExport, "boom".into());
            panic!("the export failed");
        });
        assert!(r.is_err());
        assert_eq!(jobs.views(Instant::now()).len(), 1);
        drop(b);
        assert!(jobs.views(Instant::now()).is_empty());
        // Ids are never reused.
        let c = jobs.start(SideKind::XanyImport, "again".into());
        assert_eq!(jobs.views(Instant::now())[0].id, "side-4");
        drop(c);
    }

    #[test]
    fn an_export_moves_through_its_steps_and_its_percent_only_grows() {
        let steps = [
            ExportStep::Listing,
            ExportStep::Reading { done: 0, of: 200 },
            ExportStep::Reading { done: 100, of: 200 },
            ExportStep::Reading { done: 200, of: 200 },
            ExportStep::Queue,
            ExportStep::Writing { done: 0, of: 50 },
            ExportStep::Writing { done: 25, of: 50 },
            ExportStep::Writing { done: 50, of: 50 },
            ExportStep::Publishing,
        ];
        let r: Vec<Report> = steps.into_iter().map(export_report).collect();
        let pct: Vec<u8> = r.iter().map(|r| r.percent).collect();
        assert_eq!(pct, [0, 0, 12, 25, 25, 25, 61, 97, 97]);
        assert!(r.iter().all(|r| r.steps == 3));
        assert_eq!(r[2].stage, "selecting frames");
        assert_eq!(r[2].detail, "100 of 200 label files read");
        assert_eq!(r[6].step, 2);
        assert_eq!(r[6].step_percent, 50);
        assert_eq!(r[8].stage, "publishing");
        // Nothing to write is a finished step, not a stuck one.
        assert_eq!(
            export_report(ExportStep::Writing { done: 0, of: 0 }).step_percent,
            100
        );
    }

    #[test]
    fn an_import_counts_its_files() {
        let r = import_report(3, 12);
        assert_eq!((r.step, r.steps, r.percent), (1, 1, 25));
        assert_eq!(r.detail, "3 of 12 files");
        assert_eq!(import_report(0, 0).percent, 100);
    }
}
