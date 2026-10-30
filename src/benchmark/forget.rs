//! Drop a removed model's benchmark results: its entry in `benchmark.json` (and the image sets
//! only it used), its stored predictions in `benchmark-preds.json` and its history in
//! `benchmark-history.json`, so the Benchmark page stops listing a model that is gone.

use super::history::{self, History};
use super::report::BenchmarkResults;
use super::search::StoredPreds;
use crate::registry::normalize_name;
use anyhow::Result;
use std::path::Path;

/// What [`forget_model`] dropped.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Forgotten {
    pub results: bool,
    pub preds: usize,
    pub history: usize,
}

impl Forgotten {
    pub fn any(&self) -> bool {
        self.results || self.preds > 0 || self.history > 0
    }
}

/// Drop model `name` (normalized match) from the results file `results_file` and the
/// predictions and history files beside it. Files without the model are not rewritten; a
/// missing file is nothing to drop.
pub fn forget_model(results_file: &Path, name: &str) -> Result<Forgotten> {
    let key = normalize_name(name);
    let same = |m: &str| normalize_name(m) == key;
    let mut out = Forgotten::default();

    if let Some(mut r) = BenchmarkResults::load(results_file)? {
        let before = r.models.len();
        r.models.retain(|m| !same(&m.model));
        if r.models.len() != before {
            let models = &r.models;
            r.image_sets
                .retain(|s| models.iter().any(|m| m.datasets.contains(&s.id)));
            r.save(results_file)?;
            out.results = true;
        }
    }

    let preds_file = results_file.with_file_name(super::search::PREDS_FILE);
    if let Some(mut p) = StoredPreds::load(&preds_file)? {
        let before = p.runs.len();
        p.runs.retain(|r| !same(&r.model));
        out.preds = before - p.runs.len();
        if out.preds > 0 {
            p.save(&preds_file)?;
        }
    }

    let history_file = history::history_path(results_file);
    if history_file.is_file() {
        let (n, _) = History::update(&history_file, None, |h| {
            let before: usize = h.models.values().map(Vec::len).sum();
            h.models.retain(|k, _| !same(k));
            Ok(before - h.models.values().map(Vec::len).sum::<usize>())
        })?;
        out.history = n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forgets_the_model_everywhere_and_keeps_the_others() {
        let dir = std::env::temp_dir().join(format!("bop-forget-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let results = dir.join("benchmark.json");
        let model = |name: &str, sets: &[&str]| {
            let mut m = super::super::report::ModelResult::failed(name, "m.onnx", "x".into());
            m.datasets = sets.iter().map(|s| s.to_string()).collect();
            m
        };
        let mut r = BenchmarkResults::new(
            Default::default(),
            Default::default(),
            vec![
                model("keep", &["a"]),
                model("Gone-1", &["a", "heldout-gone-1"]),
            ],
        );
        r.image_sets = ["a", "heldout-gone-1"]
            .iter()
            .map(|id| {
                serde_json::from_value(serde_json::json!({"id": id, "kind": "dir", "ground_truth": {"kind": "manifest"}, "images": []}))
                    .unwrap()
            })
            .collect();
        r.save(&results).unwrap();
        let preds = dir.join(super::super::search::PREDS_FILE);
        std::fs::write(
            &preds,
            serde_json::json!({"version": "1", "runs": [
                {"model": "keep", "device": "ort:cpu", "configured": 0.5, "eval_confidence": 0.05},
                {"model": "gone-1", "device": "ort:cpu", "configured": 0.5, "eval_confidence": 0.05},
            ]})
            .to_string(),
        )
        .unwrap();

        let f = forget_model(&results, "gone-1").unwrap();
        assert!(f.results);
        assert_eq!(f.preds, 1);
        let r = BenchmarkResults::load(&results).unwrap().unwrap();
        let names: Vec<&str> = r.models.iter().map(|m| m.model.as_str()).collect();
        assert_eq!(names, ["keep"]);
        let sets: Vec<&str> = r.image_sets.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(sets, ["a"]);
        let p = StoredPreds::load(&preds).unwrap().unwrap();
        assert_eq!(p.runs.len(), 1);
        // Nothing left to drop: nothing rewritten.
        assert!(!forget_model(&results, "gone-1").unwrap().any());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
