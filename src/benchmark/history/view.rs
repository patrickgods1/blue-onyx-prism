//! The history-derived fields of `GET /v1/benchmark`: the newest run id, the noise rules, and
//! per saved model whether it is from the newest run, its trust and what changed since the
//! baseline; per device its runtime and trust. Pure: works on the JSON of the saved results.

use super::compare::{SE_FACTOR, SPEED_ABS_MS, SPEED_REL};
use super::trust::{TRUST_LOW_OBJECTS, TRUST_MEDIUM_OBJECTS, device_trust, model_trust};
use super::{History, HistoryEntry, compare_with_history, runtime_of};
use crate::benchmark::report::BenchmarkResults;
use crate::benchmark::threshold::PLATEAU;

/// The newest `run_id` over `results.models` (by time; None when no model has one).
pub fn latest_run_id(results: &BenchmarkResults) -> Option<String> {
    results
        .models
        .iter()
        .filter_map(|m| m.run_id.clone())
        .max_by(|a, b| time_cmp(a, b))
}

/// Order two RFC 3339 times by the instant they name (strings that do not parse sort first).
pub fn time_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    let p = |s: &str| chrono::DateTime::parse_from_rfc3339(s).ok();
    match (p(a), p(b)) {
        (Some(x), Some(y)) => x.cmp(&y),
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, None) => a.cmp(b),
    }
}

/// The verdict rules, for the page's glossary.
pub fn noise_json() -> serde_json::Value {
    serde_json::json!({
        "speed_rel": SPEED_REL,
        "speed_abs_ms": SPEED_ABS_MS,
        "se_factor": SE_FACTOR,
        "plateau_points": PLATEAU * 100.0,
        "trust_low_objects": TRUST_LOW_OBJECTS,
        "trust_medium_objects": TRUST_MEDIUM_OBJECTS,
    })
}

/// Add to `value` (the JSON of `results`, models in the same order) per model `this_run`,
/// `trust` and `changes` (against `history`), and per device `runtime` and `trust`.
pub fn annotate(value: &mut serde_json::Value, results: &BenchmarkResults, history: &History) {
    let latest = latest_run_id(results);
    let Some(models) = value
        .get_mut("models")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };
    for (m, v) in results.models.iter().zip(models.iter_mut()) {
        let Some(obj) = v.as_object_mut() else {
            continue;
        };
        let this_run = latest.is_some() && m.run_id == latest;
        obj.insert("this_run".into(), this_run.into());
        obj.insert("run_id".into(), serde_json::json!(m.run_id));
        obj.insert(
            "trust".into(),
            serde_json::to_value(model_trust(m)).unwrap_or_default(),
        );
        let changes = super::recordable(m).then(|| {
            let current = HistoryEntry::of(m, results);
            compare_with_history(history.entries(&m.model), &current)
        });
        obj.insert(
            "changes".into(),
            serde_json::to_value(changes).unwrap_or_default(),
        );
        if let Some(devs) = obj
            .get_mut("devices")
            .and_then(serde_json::Value::as_array_mut)
        {
            for (d, dv) in m.devices.iter().zip(devs.iter_mut()) {
                if let Some(o) = dv.as_object_mut() {
                    o.insert("runtime".into(), runtime_of(&d.device).into());
                    o.insert(
                        "trust".into(),
                        serde_json::to_value(device_trust(m, d)).unwrap_or_default(),
                    );
                }
            }
        }
    }
}
