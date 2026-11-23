//! The run's one stop timeline (audit 2026-11, RA-009): what the first stop request starts at
//! once, what is waited for afterwards and for how long, and what the forced-exit paths do
//! before `std::process::exit`. The budget table is in the [`super`] module docs.
//!
//! - [`Timeline::begin`] (t0, idempotent, never blocks): cancels the benchmark (no wait) and
//!   calls the three "stop for exit" hooks, the training queue, the provisioner (model export and
//!   downloads) and the collector, each on its own thread with the deadline `t0 + JOBS_STOP`. They
//!   are signalled together and never one after another. [`arm`] calls it from a runtime task the
//!   moment the run's `shutdown` token is cancelled, wherever the run loop is blocked then (a
//!   takeover wait, a registry start).
//! - [`Timeline::finish`] then waits, each step against the same t0: the HTTP stop
//!   (`t0 + SERVER_STOP_TIMEOUT`), the benchmark (`t0 + BENCHMARK_STOP`; a thread still busy is
//!   abandoned with a warning naming its device, and its breadcrumb is kept on purpose: it is
//!   the only record of a device that hung), the step-1 threads (`t0 + JOBS_STOP + JOIN_SLACK`;
//!   a hook that overruns is abandoned and named), a forced sessions flush, and last the
//!   generation's retire (registry shutdown, bounded by the registry itself).
//! - The forced exits (second Ctrl-C, [`super::SHUTDOWN_DEADLINE`], the Windows service giving
//!   up, a console window closing) call [`before_forced_exit`]: it kills the child trees this
//!   process recorded in `.children.json` ([`crate::children`]) of every data root the run used,
//!   proven by PID + start time + exe, then flushes the file log ([`crate::cli::flush_logs`]).
//! - [`sweep_orphans_once`] runs [`crate::children::Registry::sweep`] once per data root and
//!   process, before the run spawns the retention sweeper (whose storage sweep would otherwise
//!   take a dead job's unlocked work folder while its orphaned child still writes into it).

use crate::children::{Entry, OsProbe, Probe, Registry};
use crate::sync::LockExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

/// What the timeline stops; [`ProcessTargets`] is the real one, tests use fakes.
pub(crate) trait StopTargets: Send + Sync + 'static {
    /// The training queue's stop for exit (returns by `deadline`).
    fn stop_training(&self, deadline: Instant);
    /// The provisioner's stop for exit: model export and the download thread.
    fn stop_provisioner(&self, deadline: Instant);
    /// The collector's stop for exit.
    fn stop_collector(&self, deadline: Instant);
    /// Ask a running benchmark to stop; never waits.
    fn cancel_benchmark(&self);
    /// Wait until no benchmark runs, at most until `deadline`; false when one still does.
    fn wait_benchmark(&self, deadline: Instant) -> bool;
    /// Write sessions and device cookies now (forced).
    fn flush_auth(&self);
}

/// The warning for a benchmark thread still busy after its stop wait (`waited`): it names the
/// device and how long the run had been going, and says the breadcrumb stays.
pub(crate) fn abandoned_benchmark(
    running: Option<&crate::benchmark::service::Running>,
    waited: Duration,
) -> String {
    let device = running
        .and_then(|r| r.progress.as_ref())
        .map(|p| p.device.as_str())
        .filter(|d| !d.is_empty())
        .unwrap_or("its device");
    let age = running
        .and_then(|r| r.started)
        .map(|s| format!(" (the run started {} s ago)", s.elapsed().as_secs()))
        .unwrap_or_default();
    format!(
        "benchmark thread abandoned on {device} after {} s: still compiling or hung{age}. Its \
         breadcrumb stays, so the next start names {device}; if the machine hangs again, leave \
         that device out of benchmarks",
        waited.as_secs()
    )
}

/// The three step-1 hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Hook {
    Training,
    Provisioner,
    Collector,
}

impl Hook {
    const ALL: [Hook; 3] = [Hook::Training, Hook::Provisioner, Hook::Collector];

    /// What it stops (logs).
    pub(crate) fn name(self) -> &'static str {
        match self {
            Hook::Training => "training queue",
            Hook::Provisioner => "model export and downloads",
            Hook::Collector => "dataset collector",
        }
    }

    fn run(self, targets: &dyn StopTargets, deadline: Instant) {
        match self {
            Hook::Training => targets.stop_training(deadline),
            Hook::Provisioner => targets.stop_provisioner(deadline),
            Hook::Collector => targets.stop_collector(deadline),
        }
    }
}

/// One step-1 hook on its thread.
struct Job {
    hook: Hook,
    join: Option<std::thread::JoinHandle<()>>,
}

struct Begun {
    t0: Instant,
    jobs: Vec<Job>,
}

/// What [`Timeline::finish`] saw (logged; asserted in tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Report {
    /// Step-1 hooks still running at their join deadline (left behind).
    pub(crate) abandoned: Vec<Hook>,
    /// Whether the benchmark was idle by its deadline.
    pub(crate) benchmark_idle: bool,
    /// From t0 to the end of the retire.
    pub(crate) took: Duration,
}

/// The stop of one run (see the module docs).
pub(crate) struct Timeline {
    targets: Arc<dyn StopTargets>,
    begun: Mutex<Option<Begun>>,
    /// Cancelled when the run ends, so the [`arm`] task never outlives it (it holds the targets).
    done: CancellationToken,
    jobs_stop: Duration,
    benchmark_stop: Duration,
    join_slack: Duration,
}

impl Timeline {
    pub(crate) fn new(targets: Arc<dyn StopTargets>) -> Arc<Self> {
        Self::with_budget(
            targets,
            super::JOBS_STOP,
            super::BENCHMARK_STOP,
            super::JOIN_SLACK,
        )
    }

    fn with_budget(
        targets: Arc<dyn StopTargets>,
        jobs_stop: Duration,
        benchmark_stop: Duration,
        join_slack: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            targets,
            begun: Mutex::new(None),
            done: CancellationToken::new(),
            jobs_stop,
            benchmark_stop,
            join_slack,
        })
    }

    /// Start the stop now (t0) unless it started already; returns t0. Never blocks: the step-1
    /// hooks run on their own threads (a thread the OS refuses runs its hook inline, warned).
    pub(crate) fn begin(&self) -> Instant {
        let mut begun = self.begun.lock_recover();
        if let Some(b) = begun.as_ref() {
            return b.t0;
        }
        let t0 = Instant::now();
        let deadline = t0 + self.jobs_stop;
        let mut jobs = Vec::with_capacity(Hook::ALL.len());
        for hook in Hook::ALL {
            let targets = self.targets.clone();
            let join = std::thread::Builder::new()
                .name(format!("stop {}", hook.name()))
                .spawn(move || hook.run(&*targets, deadline))
                .map_err(|e| {
                    warn!(
                        "could not start a thread to stop the {} ({e}); stopping it on this one, \
                         so the stop takes longer",
                        hook.name()
                    );
                })
                .ok();
            jobs.push(Job { hook, join });
        }
        self.targets.cancel_benchmark();
        for job in jobs.iter().filter(|j| j.join.is_none()) {
            job.hook.run(&*self.targets, deadline);
        }
        *begun = Some(Begun { t0, jobs });
        t0
    }

    /// Run the waits of the stop in order (see the module docs), starting it first when nobody
    /// did: `http(deadline)` stops the HTTP server by `deadline`, `retire` stops the last
    /// generation. Call once, at the end of the run.
    pub(crate) fn finish(&self, http: impl FnOnce(Instant), retire: impl FnOnce()) -> Report {
        let t0 = self.begin();
        self.done.cancel();
        http(t0 + super::SERVER_STOP_TIMEOUT);
        let benchmark_idle = self.targets.wait_benchmark(t0 + self.benchmark_stop);
        let jobs = self
            .begun
            .lock_recover()
            .as_mut()
            .map(|b| std::mem::take(&mut b.jobs))
            .unwrap_or_default();
        let abandoned = join_by(jobs, t0 + self.jobs_stop + self.join_slack);
        for hook in &abandoned {
            warn!(
                "stopping the {} did not finish within {} s of the stop request; the stop goes on \
                 without it (what it was doing is in the lines before)",
                hook.name(),
                (self.jobs_stop + self.join_slack).as_secs()
            );
        }
        self.targets.flush_auth();
        retire();
        let took = t0.elapsed();
        debug!(took_ms = took.as_millis(), "stop timeline finished");
        Report {
            abandoned,
            benchmark_idle,
            took,
        }
    }
}

/// Join `jobs` until `deadline`; the hooks still running then (their threads are detached).
fn join_by(jobs: Vec<Job>, deadline: Instant) -> Vec<Hook> {
    let mut left = Vec::new();
    for job in jobs {
        let Some(join) = job.join else { continue };
        while !join.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if join.is_finished() {
            if join.join().is_err() {
                warn!("stopping the {} panicked", job.hook.name());
            }
        } else {
            left.push(job.hook);
        }
    }
    left
}

/// Start `timeline` on `rt` as soon as `shutdown` is cancelled (t0 is the stop request, not the
/// moment the run loop notices it). The task ends with the run either way.
pub(crate) fn arm(
    rt: &tokio::runtime::Runtime,
    timeline: &Arc<Timeline>,
    shutdown: &CancellationToken,
) {
    let (timeline, shutdown) = (timeline.clone(), shutdown.clone());
    let done = timeline.done.clone();
    rt.spawn(async move {
        tokio::select! {
            () = shutdown.cancelled() => {
                timeline.begin();
            }
            () = done.cancelled() => {}
        }
    });
}

/// The real targets of one run.
pub(crate) struct ProcessTargets {
    pub(crate) provisioner: Arc<crate::resources::provision::Provisioner>,
    pub(crate) benchmark: Arc<crate::benchmark::service::BenchmarkService>,
    pub(crate) auth: Arc<crate::accounts::Auth>,
}

impl StopTargets for ProcessTargets {
    fn stop_training(&self, deadline: Instant) {
        crate::server::stop_training_for_exit(deadline);
    }
    fn stop_provisioner(&self, deadline: Instant) {
        self.provisioner.stop_for_exit(deadline);
    }
    fn stop_collector(&self, deadline: Instant) {
        crate::collect::global().stop_for_exit(deadline);
    }
    fn cancel_benchmark(&self) {
        if self.benchmark.cancel() {
            info!("benchmark cancelled: the service is stopping");
        }
    }
    fn wait_benchmark(&self, deadline: Instant) -> bool {
        // Cancels, waits, and on a timeout marks the run's breadcrumb as stopped-while-running
        // and warns with the device (`BenchmarkService::stop_for_exit`).
        self.benchmark.stop_for_exit(deadline)
    }
    fn flush_auth(&self) {
        self.auth.flush(crate::accounts::now(), true);
    }
}

/// The data roots this process's runs used (where their children are recorded).
static DATA_ROOTS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Remember `root` for [`before_forced_exit`].
pub(crate) fn remember_data_root(root: &Path) {
    let mut roots = DATA_ROOTS.lock_recover();
    if !roots.iter().any(|r| r == root) {
        roots.push(root.to_path_buf());
    }
}

/// Data roots already swept by [`sweep_orphans_once`] in this process.
static SWEPT: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Kill the children that earlier, now dead, processes left in `data_root`'s `.children.json`
/// ([`Registry::sweep`]), once per data root and process, and log what happened. Call it before
/// anything that trusts a job's lock or busy marker (the retention sweeper's storage sweep).
pub(crate) fn sweep_orphans_once(data_root: &Path) {
    {
        let mut swept = SWEPT.lock_recover();
        if swept.iter().any(|r| r == data_root) {
            return;
        }
        swept.push(data_root.to_path_buf());
    }
    let report = Registry::new(data_root).sweep();
    if let Some(e) = &report.error {
        warn!(
            "the orphaned-process check of {} could not run: {e}. A training, export or \
             auto-label process an earlier run left may still be running; end it by hand if so",
            data_root.display()
        );
    }
    if !report.killed.is_empty() {
        info!(
            killed = report.killed.len(),
            forgotten = report.forgotten,
            "ended processes an earlier run left behind (listed above)"
        );
    } else if report.forgotten > 0 {
        debug!(
            forgotten = report.forgotten,
            "dropped child-process entries of earlier runs (those processes are gone)"
        );
    }
}

/// The entries among `entries` that this process (`me_pid`, started at `me_start`) recorded and
/// that are still the same processes (PID, start time and exe match): what a forced exit kills.
pub(crate) fn own_live(
    entries: Vec<Entry>,
    me_pid: u32,
    me_start: Option<u64>,
    probe: &impl Probe,
) -> Vec<Entry> {
    let Some(me_start) = me_start else {
        return Vec::new();
    };
    entries
        .into_iter()
        .filter(|e| e.owner_pid == me_pid && e.owner_start_time == me_start)
        .filter(|e| probe.start_time(e.pid) == Some(e.start_time))
        .filter(|e| probe.exe_path(e.pid).as_deref() == Some(e.exe.as_path()))
        .collect()
}

/// Kill the child trees this process recorded (every data root its runs used), each proven to be
/// the recorded process first. Returns how many were killed. For the forced-exit paths, where no
/// job's own cancel runs (`std::process::exit` runs no destructors). On Windows the Job Objects
/// end the trees at exit anyway; this ends the recorded processes a moment earlier.
pub fn kill_own_children() -> usize {
    let roots = DATA_ROOTS.lock_recover().clone();
    kill_own_in(&roots)
}

/// [`kill_own_children`] over `roots`.
fn kill_own_in(roots: &[PathBuf]) -> usize {
    let me = std::process::id();
    let me_start = crate::backend::process::start_time(me);
    let mut killed = 0;
    for root in roots {
        let entries = match Registry::new(root).entries() {
            Ok(e) => e,
            Err(e) => {
                warn!(
                    "could not read the child-process list in {}: {e:#}",
                    root.display()
                );
                continue;
            }
        };
        for e in own_live(entries, me, me_start, &OsProbe) {
            // The child leads its own group (`process::prepare`): the tree, then the PID.
            let group = crate::backend::process::kill_group(e.pid);
            let one = crate::backend::process::kill_pid(e.pid);
            match group.and(one) {
                Ok(()) => {
                    killed += 1;
                    warn!(
                        "ended {} process {} ({}) on the forced exit",
                        e.kind,
                        e.pid,
                        e.exe.display()
                    );
                }
                Err(err) => warn!(
                    "could not end {} process {} ({}): {err}. End it by hand",
                    e.kind,
                    e.pid,
                    e.exe.display()
                ),
            }
        }
    }
    killed
}

/// What every forced exit does before `std::process::exit`: kill this process's recorded child
/// trees ([`kill_own_children`]), then write the file log's tail ([`crate::cli::flush_logs`]).
pub fn before_forced_exit() {
    kill_own_children();
    crate::cli::flush_logs();
}

#[cfg(test)]
mod tests;
