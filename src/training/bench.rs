//! The benchmark at the end of a run: the trained model on its held-out frames (a `dir:` set)
//! and optional built-in sets, through the regular [`BenchmarkService`] (so the results land in
//! `benchmark.json` and its history like any other run), summarized for the Training page.

use super::job::BenchSummary;
use crate::benchmark::report::ModelResult;
use crate::benchmark::service::{BenchmarkService, RunContext, RunRequest, RunState};
use crate::config::DatasetRef;
use crate::resources::export::Cancelled;
use anyhow::{Result, anyhow, bail};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Longest wait for another benchmark to finish before this one starts.
pub const BUSY_WAIT: Duration = Duration::from_secs(60 * 60);

/// The datasets of the run: the held-out set, then `extra` built-in ids.
pub fn datasets(name: &str, heldout: Option<&Path>, extra: &[String]) -> Vec<DatasetRef> {
    let mut v = Vec::new();
    if let Some(h) = heldout {
        v.push(DatasetRef::Dir {
            dir: h.to_path_buf(),
            gt: None,
            name: Some(format!("heldout-{name}")),
        });
    }
    v.extend(extra.iter().map(|e| DatasetRef::Id(e.clone())));
    v
}

/// Benchmark `name` on `sets` (see the module docs). `ctx` builds the run context when the
/// service is free (the config must already list the model). Waits for a running benchmark up to
/// [`BUSY_WAIT`]; `cancel` stops the wait and the run.
pub fn run(
    svc: &Arc<BenchmarkService>,
    ctx: &dyn Fn() -> RunContext,
    name: &str,
    sets: Vec<DatasetRef>,
    cancel: &AtomicBool,
) -> Result<Option<BenchSummary>> {
    if sets.is_empty() {
        return Ok(None);
    }
    let c = ctx();
    let mut req = RunRequest::defaults(&c.config.benchmark);
    req.models = vec![name.to_string()];
    req.datasets = sets;
    req.devices = Vec::new();
    req.reference_model = None;
    req.validate()?;
    let started = Instant::now();
    let mut pending = Some(c);
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!(Cancelled));
        }
        let c = match pending.take() {
            Some(c) => c,
            None => ctx(),
        };
        match svc.start(c, req.clone()) {
            Ok(()) => break,
            Err(_) if started.elapsed() < BUSY_WAIT => {
                std::thread::sleep(Duration::from_secs(2));
            }
            Err(_) => bail!(
                "another benchmark kept running for {} min; benchmark the model from the \
                 Benchmark page",
                BUSY_WAIT.as_secs() / 60
            ),
        }
    }
    while svc.is_running() {
        if cancel.load(Ordering::Relaxed) {
            svc.cancel();
            svc.wait_idle(Duration::from_secs(30));
            return Err(anyhow!(Cancelled));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let st = svc.status();
    if st.state == RunState::Failed {
        bail!(
            "the benchmark failed: {}",
            st.error.unwrap_or_else(|| "unknown error".into())
        );
    }
    let m = st
        .models
        .iter()
        .rfind(|m| m.model == name)
        .ok_or_else(|| anyhow!("the benchmark finished without a result for {name}"))?;
    if let Some(e) = &m.error {
        bail!("benchmarking {name} failed: {e}");
    }
    Ok(Some(summarize(m)))
}

/// The Training page's view of a benchmark result: the recommended device (else the first that
/// ran), its p50 time and AP50 (over every class of the model when the set annotates others).
pub fn summarize(m: &ModelResult) -> BenchSummary {
    let dev = m
        .devices
        .iter()
        .find(|d| d.ok() && m.recommended.as_deref() == Some(d.device.as_str()))
        .or_else(|| m.devices.iter().find(|d| d.ok()));
    let acc = dev
        .and_then(|d| d.run.as_ref())
        .and_then(|r| r.accuracy.as_ref());
    let (ap50, ap50_95) = match acc {
        Some(a) => match &a.all_classes {
            Some(all) if all.ap50.is_some() => (all.ap50, all.ap50_95),
            _ => (a.overall.ap50, a.overall.ap50_95),
        },
        None => (None, None),
    };
    BenchSummary {
        datasets: m.datasets.clone(),
        device: dev.map(|d| d.device.clone()),
        ms_p50: dev.and_then(|d| d.total_p50()),
        ap50,
        ap50_95,
        images: m.images,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datasets_put_the_heldout_set_first() {
        let d = datasets(
            "night-yolo26n-x",
            Some(Path::new("/r/data/heldout")),
            &["nir-night".into()],
        );
        assert_eq!(d.len(), 2);
        assert_eq!(
            d[0],
            DatasetRef::Dir {
                dir: "/r/data/heldout".into(),
                gt: None,
                name: Some("heldout-night-yolo26n-x".into())
            }
        );
        assert_eq!(d[1], DatasetRef::Id("nir-night".into()));
        assert!(datasets("x", None, &[]).is_empty());
    }
}
