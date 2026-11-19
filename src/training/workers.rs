//! Learned data loader worker counts (docs/TRAINING.md, "Learned worker count").
//!
//! [`super::workers_for`] gives a safe start from the cores and the memory free when a run starts,
//! not the best count: on a GTX 1070 Ti / i5-8600K box it allowed 4, and 4 were no faster than 2
//! and then ran out of host memory. So every run records what it used ([`RunResources`] on its
//! `runs.json` record: the count, the machine, the commit and RAM it took, sampled by [`sampler`],
//! and the median speed and data wait of its epochs), and the next run of the same machine and
//! recipe gets [`learned`] (pure) from those records. There is no state file of its own.
//!
//! The rules of [`learned`], for runs of one [`Machine`] and one [`WorkerKey`] (trainer, family,
//! size, input size, batch):
//! - no such run: the limit (today's count);
//! - the speed per count is that of the latest run with it; going up from the fewest workers, a
//!   count is the best only when it is at least [`FASTER_BY`] faster than the best below it;
//! - a measured count above the best (no faster): the best;
//! - the best is the highest measured count and its run waited on data more than
//!   [`DATA_WAIT_SHARE`] of its training time and left memory for one more worker
//!   ([`one_more_fits`]): one more; else the best;
//! - a run that ran out of host memory with N workers caps the count at N - 1 (for that machine:
//!   another [`Machine`] starts over);
//! - never above the limit when the run starts. `training.workers` always wins (the job never asks
//!   here then), and such forced runs still count as measurements.

pub mod sampler;

use super::capability::Accelerator;
use super::recipe::TrainRecipe;
use super::service::RunRecord;
use crate::resources::catalog::{TrainFamily, Trainer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Bound::{Excluded, Unbounded};

/// Data wait (the share of the training time spent waiting for the next batch, `data_wait` of the
/// epochs) above which a run is called data-bound, so one more worker may help. The measured box
/// waited 35% with 0 workers and 15% with 2.
pub const DATA_WAIT_SHARE: f64 = 0.10;
/// How much faster (images per second) more workers must be to be worth their memory: repeats of
/// one setting varied by about 4% in `img_s` (docs/TRAINING.md, the M1 table).
pub const FASTER_BY: f64 = 0.05;

/// The machine a measurement holds for: another CPU, core count, RAM size or GPU starts over
/// (that includes forgetting a host out-of-memory cap).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Machine {
    /// CPU brand string ([`crate::system_info::cpu_name`]).
    pub cpu: String,
    /// Logical cores.
    pub cores: u32,
    /// Installed RAM, rounded to whole GB.
    pub ram_gb: u32,
    /// The training GPU's name and memory (MB; on a Mac the MPS share).
    pub gpu: String,
    pub vram_mb: u64,
}

impl Machine {
    /// This machine, training on `accel`.
    pub fn now(accel: &Accelerator) -> Self {
        let cores = std::thread::available_parallelism().map_or(0, std::num::NonZero::get);
        Self {
            cpu: crate::system_info::cpu_name(),
            cores: u32::try_from(cores).unwrap_or(u32::MAX),
            // Rounded: the usable RAM (15.9 GB of 16) moves a little with firmware reserves.
            ram_gb: crate::system_info::total_memory_gb().map_or(0, |g| {
                u32::try_from(super::estimate::to_u64(g)).unwrap_or(u32::MAX)
            }),
            gpu: accel.name().to_string(),
            vram_mb: accel.memory_mb(),
        }
    }
}

/// What a learned count is for: a run of another trainer, family, size, input size or batch learns
/// nothing from this one (the data loading cost and the GPU's pace both change with them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerKey {
    pub trainer: Trainer,
    pub family: TrainFamily,
    pub size: String,
    pub imgsz: u32,
    pub batch: u32,
}

impl WorkerKey {
    /// The key of a run of `recipe` (its batch is the one that trained: a record's recipe keeps
    /// the batch an out-of-GPU-memory restart ended with).
    pub fn of(recipe: &TrainRecipe) -> Self {
        Self {
            trainer: recipe.family.trainer(),
            family: recipe.family,
            size: recipe.size.clone(),
            imgsz: recipe.imgsz,
            batch: recipe.batch,
        }
    }
}

/// What a run used, on its record (`RunRecord::resources`; None for runs from before this was
/// recorded and for runs that never reached the training stage). Measured over the attempt that
/// finished (an out-of-host-memory restart starts the measurement over), from the start of the
/// training stage to the export, every [`sampler::SAMPLE_EVERY`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RunResources {
    /// Data loader workers (`--workers`) of that attempt.
    pub workers: u32,
    /// Loaders keeping workers alive at once ([`super::worker_loaders`]): the run had
    /// `workers x loaders` worker processes.
    pub loaders: u32,
    /// `training.workers` set the count.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub forced: bool,
    /// The automatic limit ([`super::workers_for`]) when the run started; None when forced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    pub machine: Machine,
    /// Median `img_s` and `data_wait` over that attempt's epochs; None before a finished epoch
    /// (or from a trainer that does not report it).
    pub img_s: Option<f64>,
    pub data_wait: Option<f64>,
    /// System commit in use at the start (bytes), its peak, and the commit limit at the peak
    /// (Windows only).
    pub commit_start: Option<u64>,
    pub commit_peak: Option<u64>,
    pub commit_limit_at_peak: Option<u64>,
    /// RAM available at the start and the least seen (bytes; Windows and Linux).
    pub ram_start: Option<u64>,
    pub ram_min: Option<u64>,
    /// Most pages read in per second (hard faults; Windows and Linux).
    pub page_ins_peak: Option<f64>,
    /// Readings taken.
    pub samples: u32,
    /// The run ran out of host memory with this many workers (the fewest of its attempts that
    /// did: each restart halves them).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_oom: Option<u32>,
    /// That attempt's median `img_s`, when it finished an epoch before failing and it was not the
    /// attempt measured above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_oom_img_s: Option<f64>,
}

impl RunResources {
    /// Worker processes of the run (`workers x loaders`).
    pub fn processes(&self) -> u64 {
        u64::from(self.workers) * u64::from(self.loaders.max(1))
    }

    /// One worker's measured share of commit and of RAM (bytes): the run's growth over its
    /// baseline (the start) shared out over its worker processes, times the loaders (one worker
    /// more is that many processes). The growth includes the training process, so this errs
    /// high. None without worker processes or a reading.
    pub fn worker_share(&self) -> (Option<u64>, Option<u64>) {
        let n = self.processes();
        let per = |grew: Option<u64>| {
            grew.filter(|_| n > 0)
                .map(|g| (g / n).saturating_mul(u64::from(self.loaders.max(1))))
        };
        let commit = per(self
            .commit_peak
            .zip(self.commit_start)
            .map(|(p, s)| p.saturating_sub(s)));
        let ram = per(self
            .ram_start
            .zip(self.ram_min)
            .map(|(s, m)| s.saturating_sub(m)));
        (commit, ram)
    }

    /// The log line: "resources: 2 data loader workers (4 processes), commit 17.9 GB at the start,
    /// peak 31.4 of 32.4 GB, RAM 7.6 GB free at the start, 2.06 GB at least, page-ins up to
    /// 2.8k/s, 11.4 img/s, waited on data 15%"; the parts with no reading left out.
    pub fn describe(&self) -> String {
        let gib = super::gib;
        let mut parts = vec![format!(
            "{} ({} processes)",
            super::workers_text(self.workers),
            self.processes()
        )];
        if let (Some(s), Some(p), Some(l)) = (
            self.commit_start,
            self.commit_peak,
            self.commit_limit_at_peak,
        ) {
            parts.push(format!(
                "commit {:.1} GB at the start, peak {:.1} of {:.1} GB",
                gib(s),
                gib(p),
                gib(l)
            ));
        }
        if let (Some(s), Some(m)) = (self.ram_start, self.ram_min) {
            parts.push(format!(
                "RAM {:.1} GB free at the start, {:.2} GB at least",
                gib(s),
                gib(m)
            ));
        }
        if let Some(p) = self.page_ins_peak {
            parts.push(format!("page-ins up to {}/s", per_second(p)));
        }
        if let Some(v) = self.img_s {
            parts.push(format!("{v:.1} img/s"));
        }
        if let Some(w) = self.data_wait {
            parts.push(format!("waited on data {}", pct(w)));
        }
        if let Some(n) = self.host_oom {
            parts.push(format!(
                "out of system memory with {}",
                super::workers_text(n)
            ));
        }
        format!("resources: {}", parts.join(", "))
    }
}

/// "850", "2.8k", "12k".
fn per_second(v: f64) -> String {
    if v >= 10_000.0 {
        format!("{:.0}k", v / 1000.0)
    } else if v >= 1000.0 {
        format!("{:.1}k", v / 1000.0)
    } else {
        format!("{v:.0}")
    }
}

/// "15%".
fn pct(share: f64) -> String {
    format!("{:.0}%", share * 100.0)
}

/// The median of `metric` over `history`'s epochs (finite values only); None without one.
pub fn median_metric(history: &[super::progress::EpochPoint], metric: &str) -> Option<f64> {
    let mut v: Vec<f64> = history
        .iter()
        .filter_map(|p| p.metrics.get(metric).copied())
        .filter(|x| x.is_finite())
        .collect();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    let upper = *v.get(mid)?;
    Some(if v.len() % 2 == 1 {
        upper
    } else {
        (v.get(mid.checked_sub(1)?)? + upper) / 2.0
    })
}

/// One measured worker count.
#[derive(Debug, Clone, Copy)]
struct Point<'a> {
    img_s: f64,
    data_wait: Option<f64>,
    /// The run it came from (None: the failed attempt of a restarted run, whose memory was not
    /// kept).
    run: Option<&'a RunResources>,
}

/// Whether the memory `run` left covers one more worker: its least available RAM less
/// [`super::RAM_RESERVE_BYTES`], and its commit limit less its peak, each at least one worker's
/// measured share ([`RunResources::worker_share`]; a run without workers uses
/// [`super::WORKER_RAM_BYTES`] / [`super::WORKER_COMMIT_BYTES`] per loader). A figure the OS does
/// not give (no commit limit on Linux) is not checked.
pub fn one_more_fits(run: &RunResources) -> bool {
    let loaders = u64::from(run.loaders.max(1));
    let (commit_share, ram_share) = run.worker_share();
    let commit_need = commit_share.unwrap_or(super::WORKER_COMMIT_BYTES * loaders);
    let ram_need = ram_share.unwrap_or(super::WORKER_RAM_BYTES * loaders);
    let ram_ok = run
        .ram_min
        .is_none_or(|m| m.saturating_sub(super::RAM_RESERVE_BYTES) >= ram_need);
    let commit_ok = run
        .commit_limit_at_peak
        .zip(run.commit_peak)
        .is_none_or(|(l, p)| l.saturating_sub(p) >= commit_need);
    ram_ok && commit_ok
}

/// "4", "3 and 4", "2, 3 and 4".
fn counts_text(v: &[u32]) -> String {
    match v {
        [] => String::new(),
        [one] => one.to_string(),
        [rest @ .., last] => format!(
            "{} and {last}",
            rest.iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The data loader workers a run of `key` on `machine` gets, from the earlier runs `past` (any
/// order; runs of other machines or keys and runs without [`RunResources`] are ignored), and why
/// (the module docs give the rules). `safe_limit` is [`super::workers_for`] now.
pub fn learned(
    machine: &Machine,
    key: &WorkerKey,
    past: &[RunRecord],
    safe_limit: u32,
) -> (u32, String) {
    let mut runs: Vec<(&str, &RunResources)> = past
        .iter()
        .filter(|r| WorkerKey::of(&r.request.recipe) == *key)
        .filter_map(|r| Some((r.created.as_str(), r.resources.as_ref()?)))
        .filter(|(_, res)| res.machine == *machine)
        .collect();
    if runs.is_empty() {
        return (safe_limit, "nothing learned on this machine yet".into());
    }
    runs.sort_by(|a, b| a.0.cmp(b.0));
    // A host out-of-memory failure with N workers: at most N - 1 (the fewest such N).
    let oom = runs.iter().filter_map(|(_, r)| r.host_oom).min();
    let cap = oom.map_or(u32::MAX, |n| n.saturating_sub(1));
    let oom_clause = |n: u32| {
        format!(
            "{n} ran out of system memory, so at most {}",
            n.saturating_sub(1)
        )
    };
    // Speed per count: the latest run with it wins.
    let mut speeds: BTreeMap<u32, Point<'_>> = BTreeMap::new();
    for (_, r) in &runs {
        if let (Some(n), Some(v)) = (r.host_oom, r.host_oom_img_s)
            && n != r.workers
            && v.is_finite()
            && v > 0.0
        {
            speeds.insert(
                n,
                Point {
                    img_s: v,
                    data_wait: None,
                    run: None,
                },
            );
        }
        if let Some(v) = r.img_s.filter(|v| v.is_finite() && *v > 0.0) {
            speeds.insert(
                r.workers,
                Point {
                    img_s: v,
                    data_wait: r.data_wait,
                    run: Some(r),
                },
            );
        }
    }
    // The best count: the fewest workers, raised only by a count at least FASTER_BY faster.
    let mut best: Option<(u32, Point<'_>)> = None;
    for (&n, &p) in speeds.range(..=cap) {
        match best {
            None => best = Some((n, p)),
            Some((_, b)) if p.img_s >= b.img_s * (1.0 + FASTER_BY) => best = Some((n, p)),
            Some(_) => {}
        }
    }
    // A count above the best that was faster is above the cap (else it would be the best): the
    // memory, not the speed, stopped it, and the cap is said.
    let capped;
    let (count, mut why) = match best {
        None => match oom {
            Some(_) => {
                capped = true;
                (cap, String::new())
            }
            None => {
                return (
                    safe_limit,
                    "nothing measured on this machine yet (no finished epoch)".into(),
                );
            }
        },
        Some((b, p)) => {
            let above = speeds.range((Excluded(b), Unbounded)).count();
            let slower: Vec<u32> = speeds
                .range((Excluded(b), Unbounded))
                .filter(|(_, q)| q.img_s < p.img_s * (1.0 + FASTER_BY))
                .map(|(n, _)| *n)
                .collect();
            capped = slower.len() < above;
            if !slower.is_empty() {
                // More workers gave nothing: stay.
                (
                    b,
                    format!(
                        "{} {} no faster than {b}",
                        counts_text(&slower),
                        if slower.len() == 1 { "was" } else { "were" }
                    ),
                )
            } else {
                // The highest count measured within the cap: one more if it waited on data.
                match p.data_wait {
                    Some(w) if w > DATA_WAIT_SHARE => {
                        if p.run.is_some_and(one_more_fits) {
                            (
                                b.saturating_add(1),
                                format!(
                                    "{b} waited on data {} of the time and left memory for one \
                                     more",
                                    pct(w)
                                ),
                            )
                        } else {
                            (
                                b,
                                format!(
                                    "{b} waited on data {} of the time, but left too little \
                                     memory for one more",
                                    pct(w)
                                ),
                            )
                        }
                    }
                    Some(w) => (b, format!("{b} waited on data only {} of the time", pct(w))),
                    None => (b, format!("{b} reported no data wait")),
                }
            }
        }
    };
    if (capped || count > cap)
        && let Some(n) = oom
    {
        if !why.is_empty() {
            why.push_str("; ");
        }
        why.push_str(&oom_clause(n));
    }
    let count = count.min(cap).min(safe_limit);
    (count, format!("learned on this machine: {why}"))
}

/// The log line and the pre-flight sentence: "data loader workers: 2 (learned on this machine: 4
/// was no faster than 2; limits allow 4)".
pub fn sentence(workers: u32, why: &str, limit: u32) -> String {
    format!("data loader workers: {workers} ({why}; limits allow {limit})")
}

/// The sentence of a forced count: "data loader workers: 3 (training.workers)".
pub fn forced_sentence(workers: u32) -> String {
    format!("data loader workers: {workers} (training.workers)")
}

/// What a run of `recipe` on `accel` would get now: `forced` (`training.workers`) as it is, else
/// [`learned`] from `past` under [`super::workers_now`]. Returns the count and
/// [`sentence`] / [`forced_sentence`].
pub fn choice_now(
    recipe: &TrainRecipe,
    accel: &Accelerator,
    forced: Option<u32>,
    past: &[RunRecord],
) -> (u32, String) {
    if let Some(n) = forced {
        return (n, forced_sentence(n));
    }
    let limit = super::workers_now(recipe.family.trainer()).workers;
    let (n, why) = learned(&Machine::now(accel), &WorkerKey::of(recipe), past, limit);
    (n, sentence(n, &why, limit))
}

/// One learned count per recipe key with measurements on this machine (`GET /v1/training`'s
/// `workers_learned`): `{family, size, imgsz, batch, workers, learned, limit, why, text}`, oldest
/// key first.
/// `forced` (`training.workers`) replaces the count and the text.
pub fn learned_now(
    accel: &Accelerator,
    forced: Option<u32>,
    past: &[RunRecord],
) -> Vec<serde_json::Value> {
    let machine = Machine::now(accel);
    let mut keys: Vec<WorkerKey> = Vec::new();
    let mut ordered: Vec<&RunRecord> = past
        .iter()
        .filter(|r| r.resources.as_ref().is_some_and(|x| x.machine == machine))
        .collect();
    ordered.sort_by(|a, b| a.created.cmp(&b.created));
    for r in ordered {
        let k = WorkerKey::of(&r.request.recipe);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    let mut limits: BTreeMap<&'static str, u32> = BTreeMap::new();
    keys.into_iter()
        .map(|k| {
            let limit = *limits
                .entry(k.trainer.as_str())
                .or_insert_with(|| super::workers_now(k.trainer).workers);
            let (n, why) = learned(&machine, &k, past, limit);
            let (workers, text) = match forced {
                Some(f) => (f, forced_sentence(f)),
                None => (n, sentence(n, &why, limit)),
            };
            serde_json::json!({
                "family": k.family,
                "size": k.size,
                "imgsz": k.imgsz,
                "batch": k.batch,
                "workers": workers,
                "learned": n,
                "limit": limit,
                "why": why,
                "text": text,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
