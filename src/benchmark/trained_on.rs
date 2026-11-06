//! How much of a benchmark set a model trained on: a score on its own training frames says
//! little about the frames it will see next. Frames are compared by identity
//! (`<scene>/<camera>/<date>/<stem>.jpg`, [`frame_of_file`]) with the list of the run named by
//! the model's `<model>.provenance.json` ([`trained_frames`]). Results of older versions get the
//! share when they are served ([`backfill`]), from the image names they stored.

use super::find_model;
use super::report::BenchmarkResults;
use crate::config::{Config, ModelConfig};
use crate::training::autolabel::{frame_of_file, trained_frames};
use std::collections::BTreeSet;

/// The share of `files` (benchmark image names) that are in `trained`, 0..=1; None for no files.
/// An image that is no frame of a dataset folder (a public image) counts as not trained on.
pub fn share<'a>(
    files: impl IntoIterator<Item = &'a str>,
    trained: &BTreeSet<String>,
) -> Option<f64> {
    let (mut total, mut hits) = (0usize, 0usize);
    for f in files {
        total += 1;
        if frame_of_file(f).is_some_and(|k| trained.contains(&k)) {
            hits += 1;
        }
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "frame counts of a benchmark set, far below 2^52"
    )]
    (total > 0).then(|| hits as f64 / total as f64)
}

/// [`share`] for model `m`, plus a warning when its record exists but cannot be read. None and no
/// warning for a model without a training record (a catalog or plain local model): nothing here
/// says what it trained on.
pub fn of_model<'a>(
    config: &Config,
    m: &ModelConfig,
    files: impl IntoIterator<Item = &'a str>,
) -> (Option<f64>, Option<String>) {
    let mut unused = Vec::new();
    match trained_frames(config, m, &mut unused) {
        // No record gives an empty list and a warning; a run with no frames does not exist.
        Ok(trained) if trained.is_empty() => (None, None),
        Ok(trained) => (share(files, &trained), None),
        Err(e) => (
            None,
            Some(format!(
                "could not tell whether {} trained on these frames: {e:#}",
                m.effective_name()
            )),
        ),
    }
}

/// Fill `trained_on_share` of the saved models that lack it, from the image names of their sets
/// in `results`. The file is not written: the next run of a model stores its own.
pub fn backfill(results: &mut BenchmarkResults, config: &Config) {
    let sets = std::mem::take(&mut results.image_sets);
    for r in &mut results.models {
        if r.trained_on_share.is_some() {
            continue;
        }
        let Some(m) = find_model(config, &r.model) else {
            continue;
        };
        let files = sets
            .iter()
            .filter(|s| r.datasets.contains(&s.id))
            .flat_map(|s| s.images.iter())
            .map(|i| i.file.as_str());
        r.trained_on_share = of_model(config, m, files).0;
    }
    results.image_sets = sets;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    const A: &str = "s0-day.back.2026-10-25.20261025T010203004Z_aa.jpg";
    const B: &str = "s1-night.back.2026-10-25.20261025T010203005Z_bb.jpg";
    const C: &str = "day.front.2026-10-26.20261026T010203006Z_cc.jpg";

    fn trained(names: &[&str]) -> BTreeSet<String> {
        names.iter().filter_map(|n| frame_of_file(n)).collect()
    }

    #[test]
    fn share_counts_frames_by_identity() {
        // None, some, all; a held-out copy and an export's flat name of one frame are one frame.
        let t = trained(&[A]);
        assert_eq!(share([B, C], &t), Some(0.0));
        assert_eq!(share([A, B, C, "coco/000001.jpg"], &t), Some(0.25));
        assert_eq!(
            share(["day.back.2026-10-25.20261025T010203004Z_aa.jpg"], &t),
            Some(1.0)
        );
        assert_eq!(share([A], &t), Some(1.0));
        assert_eq!(
            share(["x/day/back/2026-10-25/20261025T010203004Z_aa.jpg"], &t),
            Some(1.0),
            "a path inside a dataset folder"
        );
        assert_eq!(share([], &t), None);
    }

    /// A model with a provenance record of a run listing `frames`.
    struct Fixture {
        root: PathBuf,
        config: Config,
        model: ModelConfig,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn fixture(run: Option<&str>, frames: &[&str]) -> Fixture {
        let root = std::env::temp_dir().join(format!("bop-trained-on-{}", uuid::Uuid::new_v4()));
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let config = Config {
            download_dir: Some(root.clone()),
            ..Config::default()
        };
        let model = ModelConfig {
            name: Some("custom".into()),
            path: models.join("custom.onnx"),
            ..Default::default()
        };
        if let Some(run) = run {
            std::fs::write(
                models.join("custom.provenance.json"),
                format!(r#"{{"run_id":"{run}"}}"#),
            )
            .unwrap();
        }
        if !frames.is_empty() {
            let dir = crate::training::training_dir(&config)
                .join(run.unwrap())
                .join("data");
            std::fs::create_dir_all(&dir).unwrap();
            let lines: Vec<String> = frames.iter().map(|f| format!("./images/{f}")).collect();
            write(&dir.join("train.txt"), &lines.join("\n"));
        }
        Fixture {
            root,
            config,
            model,
        }
    }

    fn write(p: &Path, text: &str) {
        std::fs::write(p, text).unwrap();
    }

    #[test]
    fn a_model_shares_what_its_run_listed() {
        let f = fixture(Some("run-1"), &[A, B]);
        let of = |files: &[&'static str]| of_model(&f.config, &f.model, files.iter().copied());
        assert_eq!(of(&[C]), (Some(0.0), None));
        assert_eq!(of(&[A, C]), (Some(0.5), None));
        assert_eq!(of(&[A, B]), (Some(1.0), None));
    }

    #[test]
    fn missing_provenance_is_unknown_and_a_broken_one_warns() {
        // A catalog model: no record, nothing to say.
        let f = fixture(None, &[]);
        assert_eq!(of_model(&f.config, &f.model, [A]), (None, None));
        // A record whose run folder is gone: unknown, and the benchmark says so.
        let f = fixture(Some("gone"), &[]);
        let (share, warning) = of_model(&f.config, &f.model, [A]);
        assert_eq!(share, None);
        let w = warning.unwrap();
        assert!(w.contains("custom") && w.contains("gone"), "{w}");
    }

    #[test]
    fn saved_results_of_older_versions_are_backfilled() {
        use crate::benchmark::report::test_support::model;
        use crate::benchmark::report::{ImageInfo, ImageSetInfo};
        let mut f = fixture(Some("run-1"), &[A]);
        f.config.models = vec![f.model.clone()];
        let info = |file: &str| ImageInfo {
            file: file.into(),
            width: 1,
            height: 1,
            tags: vec![],
            objects: None,
        };
        let mut m = model(vec![], None);
        m.model = "custom".into();
        m.datasets = vec!["export".into()];
        let mut other = m.clone();
        other.model = "not-configured".into();
        let mut r = BenchmarkResults::new(Default::default(), Default::default(), vec![m, other]);
        r.image_sets = vec![ImageSetInfo {
            id: "export".into(),
            title: String::new(),
            kind: crate::benchmark::images::SetKind::Dir,
            dir: None,
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: crate::benchmark::images::GroundTruth::None,
            scored_labels: None,
            images: vec![info(A), info(C)],
        }];
        backfill(&mut r, &f.config);
        assert_eq!(r.models[0].trained_on_share, Some(0.5));
        assert_eq!(r.models[1].trained_on_share, None);
        assert_eq!(r.image_sets.len(), 1, "the sets stay");
    }
}
