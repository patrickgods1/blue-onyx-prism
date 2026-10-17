//! How much a benchmark result can be trusted ([`trust`]: High / Medium / Low with the reasons
//! that lowered it) and which results are the best trade-offs between speed and accuracy
//! ([`pareto`]).
//!
//! Trust rules (the level is the lowest any rule allows):
//! - labelled objects scored on the device: fewer than [`TRUST_LOW_OBJECTS`] = low, fewer than
//!   [`TRUST_MEDIUM_OBJECTS`] = medium;
//! - labels from a reference model rather than people: at most medium;
//! - fewer than [`TRUST_MIN_REPEAT`] timed runs per image: at most medium (speed is noisier);
//! - the device fell back to another one, or did not run: low.
//!
//! Without accuracy (no ground truth, no reference model) only the speed rules apply, with a
//! "no accuracy data" reason.

use crate::benchmark::report::{DeviceResult, ModelResult};
use serde::{Deserialize, Serialize};

/// Fewer labelled objects than this: low trust.
pub const TRUST_LOW_OBJECTS: usize = 100;
/// Fewer labelled objects than this: medium trust.
pub const TRUST_MEDIUM_OBJECTS: usize = 300;
/// Fewer timed runs per image than this: at most medium trust.
pub const TRUST_MIN_REPEAT: usize = 2;

/// Trust level, ordered low < medium < high.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrustLevel {
    Low,
    Medium,
    High,
}

/// How much a result can be trusted, and why not more.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trust {
    pub level: TrustLevel,
    pub reasons: Vec<String>,
}

/// The facts the trust rules look at.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrustInput {
    /// Labelled objects scored (None: no accuracy).
    pub objects: Option<usize>,
    /// Labels came from a reference model (named, when known).
    pub relative: bool,
    pub reference: Option<String>,
    /// Timed runs per image.
    pub repeat: usize,
    /// `(ran on, requested)` when the device fell back.
    pub fell_back: Option<(String, String)>,
    /// Why the device did not run.
    pub error: Option<String>,
}

/// The trust of a result with the facts `i`.
pub fn trust(i: &TrustInput) -> Trust {
    let mut level = TrustLevel::High;
    let mut reasons = Vec::new();
    let mut lower = |to: TrustLevel, why: String, reasons: &mut Vec<String>| {
        level = level.min(to);
        reasons.push(why);
    };
    if let Some(e) = &i.error {
        lower(TrustLevel::Low, format!("did not run: {e}"), &mut reasons);
        return Trust { level, reasons };
    }
    if let Some((ran, wanted)) = &i.fell_back {
        lower(
            TrustLevel::Low,
            format!("ran on {ran} instead of {wanted}"),
            &mut reasons,
        );
    }
    match i.objects {
        None => reasons.push("no accuracy data (no ground truth or reference model)".into()),
        Some(n) if n < TRUST_LOW_OBJECTS => lower(
            TrustLevel::Low,
            format!("only {n} labelled objects (aim for {TRUST_MEDIUM_OBJECTS}+)"),
            &mut reasons,
        ),
        Some(n) if n < TRUST_MEDIUM_OBJECTS => lower(
            TrustLevel::Medium,
            format!("{n} labelled objects (aim for {TRUST_MEDIUM_OBJECTS}+)"),
            &mut reasons,
        ),
        Some(_) => {}
    }
    if i.objects.is_some() && i.relative {
        let by = i
            .reference
            .as_deref()
            .map(|r| format!(" ({r})"))
            .unwrap_or_default();
        lower(
            TrustLevel::Medium,
            format!("labels come from a reference model{by}, not from people"),
            &mut reasons,
        );
    }
    if i.repeat < TRUST_MIN_REPEAT {
        lower(
            TrustLevel::Medium,
            format!(
                "{} timed run per image (speed is noisier; use {TRUST_MIN_REPEAT} or more)",
                i.repeat.max(1)
            ),
            &mut reasons,
        );
    }
    Trust { level, reasons }
}

/// The reference model named in an accuracy report's "relative to <model> on <device>".
fn reference_name(desc: &str) -> Option<String> {
    super::reference_of_desc(desc)
}

/// The trust facts of device row `d` of `m`.
pub fn device_input(m: &ModelResult, d: &DeviceResult) -> TrustInput {
    let Some(run) = &d.run else {
        return TrustInput {
            error: Some(d.error.clone().unwrap_or_else(|| "no result".into())),
            ..Default::default()
        };
    };
    let acc = run.accuracy.as_ref();
    TrustInput {
        objects: acc.map(|a| a.overall.counts.tp + a.overall.counts.fn_),
        relative: acc.is_some_and(|a| a.relative),
        reference: acc.and_then(|a| reference_name(&a.ground_truth)),
        repeat: if run.repeat > 0 { run.repeat } else { m.repeat },
        fell_back: run.fell_back.then(|| {
            let ran = if run.spec.is_empty() {
                run.device.clone()
            } else {
                run.spec.clone()
            };
            (ran, d.device.clone())
        }),
        error: None,
    }
}

/// The trust of device row `d` of `m`.
pub fn device_trust(m: &ModelResult, d: &DeviceResult) -> Trust {
    trust(&device_input(m, d))
}

/// The trust of model `m`: of its recommended device, else of the first scored device, else
/// of the first device that ran, else of the first device (low without any).
pub fn model_trust(m: &ModelResult) -> Trust {
    let d = m
        .recommended
        .as_deref()
        .and_then(|r| m.devices.iter().find(|d| d.device == r))
        .or_else(|| {
            m.devices
                .iter()
                .find(|d| d.ok() && d.run.as_ref().is_some_and(|r| r.accuracy.is_some()))
        })
        .or_else(|| m.devices.iter().find(|d| d.ok()))
        .or_else(|| m.devices.first());
    match d {
        Some(d) => device_trust(m, d),
        None => Trust {
            level: TrustLevel::Low,
            reasons: vec![match &m.error {
                Some(e) => format!("did not run: {e}"),
                None => "no device ran".into(),
            }],
        },
    }
}

/// Best trade-offs among `(p50 ms, accuracy score)` rows: a row is one when no other row is at
/// least as fast and at least as accurate, and strictly better in one of the two. Rows without
/// a score are never best trade-offs while any row has one; when none has, the fastest rows
/// are. A non-finite p50 counts as infinitely slow.
pub fn pareto(rows: &[(f64, Option<f64>)]) -> Vec<bool> {
    let speed = |p: f64| if p.is_finite() { p } else { f64::INFINITY };
    let any_scored = rows.iter().any(|r| r.1.is_some_and(f64::is_finite));
    rows.iter()
        .enumerate()
        .map(|(i, &(pi, ai))| {
            let pi = speed(pi);
            let ai = ai.filter(|a| a.is_finite());
            if any_scored && ai.is_none() {
                return false;
            }
            let ai = ai.unwrap_or(f64::NEG_INFINITY);
            !rows.iter().enumerate().any(|(j, &(pj, aj))| {
                if i == j {
                    return false;
                }
                let pj = speed(pj);
                let aj = aj.filter(|a| a.is_finite()).unwrap_or(f64::NEG_INFINITY);
                pj <= pi && aj >= ai && (pj < pi || aj > ai)
            })
        })
        .collect()
}
