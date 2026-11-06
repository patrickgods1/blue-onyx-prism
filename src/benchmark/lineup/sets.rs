//! The image sets models can be compared on: every dataset of the saved results, every list of
//! datasets a model ran as a whole, and the held-out frames of a set that carries a `split:val`
//! tag. Each set knows its exact images, so "the same images" means the same files, and
//! [`default_set`] picks the one to open with.

use super::super::images::SetKind;
use crate::benchmark::report::{BenchmarkResults, ImageInfo, ImageSetInfo};
use serde::Serialize;

/// The tag of an export's held-out frames (`split:train` are the others).
pub const HELD_OUT_TAG: &str = "split:val";
/// The name prefix of a training run's held-out set (`training::bench::datasets`).
const HELD_OUT_PREFIX: &str = "heldout-";

/// What a set is, for the owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SetRole {
    /// Own frames kept out of training.
    HeldOut,
    /// Own frames.
    OwnCamera,
    /// Public images.
    Public,
}

impl SetRole {
    pub fn what(self) -> &'static str {
        match self {
            SetRole::HeldOut => "Your own frames, kept out of training",
            SetRole::OwnCamera => "Your own camera frames",
            SetRole::Public => "Public images",
        }
    }

    /// The word for its pictures.
    pub fn unit(self) -> &'static str {
        match self {
            SetRole::Public => "images",
            SetRole::HeldOut | SetRole::OwnCamera => "frames",
        }
    }
}

/// One scored image of a set.
#[derive(Debug, Clone, Copy)]
pub struct SetImage<'a> {
    pub set: &'a ImageSetInfo,
    pub info: &'a ImageInfo,
}

/// A set of images to compare on.
#[derive(Debug, Clone)]
pub struct CompareSet<'a> {
    /// `<dataset>`, `<dataset>+<dataset>` or `<dataset>#split:val`.
    pub id: String,
    pub title: String,
    pub role: SetRole,
    /// Its datasets, as the run form names them.
    pub datasets: Vec<&'a ImageSetInfo>,
    /// The images with labels, in dataset order.
    pub images: Vec<SetImage<'a>>,
}

fn scored<'a>(set: &'a ImageSetInfo, tag: Option<&str>) -> Vec<SetImage<'a>> {
    set.images
        .iter()
        .filter(|i| i.objects.is_some())
        .filter(|i| tag.is_none_or(|t| i.tags.iter().any(|x| x == t)))
        .map(|info| SetImage { set, info })
        .collect()
}

fn role_of(set: &ImageSetInfo) -> SetRole {
    if set.id.starts_with(HELD_OUT_PREFIX) {
        SetRole::HeldOut
    } else if set.kind == SetKind::Dir {
        SetRole::OwnCamera
    } else {
        SetRole::Public
    }
}

fn title_of(set: &ImageSetInfo) -> String {
    if set.title.is_empty() {
        set.id.clone()
    } else {
        set.title.clone()
    }
}

/// Every set of `results` with labelled images: each dataset, its held-out frames, and each list
/// of several datasets a model ran.
pub fn compare_sets(results: &BenchmarkResults) -> Vec<CompareSet<'_>> {
    let mut out: Vec<CompareSet<'_>> = Vec::new();
    for set in &results.image_sets {
        let images = scored(set, None);
        if images.is_empty() {
            continue;
        }
        let role = role_of(set);
        let held = scored(set, Some(HELD_OUT_TAG));
        if !held.is_empty() && held.len() < images.len() {
            out.push(CompareSet {
                id: format!("{}#{HELD_OUT_TAG}", set.id),
                title: format!("Held-out frames of {}", title_of(set)),
                role: SetRole::HeldOut,
                datasets: vec![set],
                images: held,
            });
        }
        out.push(CompareSet {
            id: set.id.clone(),
            title: title_of(set),
            role,
            datasets: vec![set],
            images,
        });
    }
    for m in &results.models {
        if m.datasets.len() < 2 {
            continue;
        }
        let mut ids: Vec<&String> = m.datasets.iter().collect();
        ids.sort();
        ids.dedup();
        let id = ids.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("+");
        if out.iter().any(|s| s.id == id) {
            continue;
        }
        let sets: Vec<&ImageSetInfo> = ids
            .iter()
            .filter_map(|i| results.image_sets.iter().find(|s| s.id == **i))
            .collect();
        if sets.len() != ids.len() {
            continue; // a dataset of the list is no longer recorded
        }
        let images: Vec<SetImage<'_>> = sets.iter().flat_map(|s| scored(s, None)).collect();
        if images.is_empty() {
            continue;
        }
        out.push(CompareSet {
            title: sets
                .iter()
                .map(|s| title_of(s))
                .collect::<Vec<_>>()
                .join(" + "),
            role: if sets.iter().all(|s| role_of(s) == SetRole::Public) {
                SetRole::Public
            } else {
                SetRole::OwnCamera
            },
            id,
            datasets: sets,
            images,
        });
    }
    out
}

/// What [`default_set`] weighs of a set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub id: String,
    /// Models scored on it.
    pub models: usize,
    pub role: SetRole,
    /// Models scored on it that trained on none of its frames.
    pub fair_models: usize,
    pub images: usize,
}

/// The order of [`default_set`], larger first.
pub fn rank(c: &Candidate) -> (usize, usize, u8, usize) {
    let role = match c.role {
        SetRole::HeldOut => 2,
        SetRole::OwnCamera => 1,
        SetRole::Public => 0,
    };
    (c.models, c.fair_models, role, c.images)
}

/// The set to open with: the one the most models share; among those, the one where the most
/// models are judged fairly, held-out frames, then own frames, then the most images. None when no
/// model is scored on any set.
pub fn default_set(candidates: &[Candidate]) -> Option<&str> {
    candidates
        .iter()
        .filter(|c| c.models > 0)
        // The first of equals: `max_by_key` would answer the last.
        .rev()
        .max_by_key(|c| rank(c))
        .map(|c| c.id.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, models: usize, role: SetRole, fair: usize, images: usize) -> Candidate {
        Candidate {
            id: id.into(),
            models,
            role,
            fair_models: fair,
            images,
        }
    }

    #[test]
    fn most_models_win() {
        let c = [
            cand("public", 3, SetRole::Public, 3, 500),
            cand("own", 2, SetRole::HeldOut, 2, 800),
        ];
        assert_eq!(default_set(&c), Some("public"));
    }

    #[test]
    fn among_equals_held_out_then_own_camera_then_size() {
        let c = [
            cand("public", 2, SetRole::Public, 2, 900),
            cand("own", 2, SetRole::OwnCamera, 2, 300),
            cand("held", 2, SetRole::HeldOut, 2, 23),
        ];
        assert_eq!(default_set(&c), Some("held"));
        assert_eq!(default_set(&c[..2]), Some("own"));
        let s = [
            cand("small", 2, SetRole::Public, 2, 10),
            cand("big", 2, SetRole::Public, 2, 20),
        ];
        assert_eq!(default_set(&s), Some("big"));
    }

    #[test]
    fn a_set_with_training_frames_gives_way_to_a_clean_one() {
        // The real case: 251 frames with custom's training frames, 23 clean ones of the same export.
        let c = [
            cand("all", 2, SetRole::OwnCamera, 1, 251),
            cand("val", 2, SetRole::HeldOut, 2, 23),
        ];
        assert_eq!(default_set(&c), Some("val"));
        // The clean set wins even when the contaminated one is called held-out.
        let c = [
            cand("a", 2, SetRole::HeldOut, 1, 100),
            cand("b", 2, SetRole::Public, 2, 100),
        ];
        assert_eq!(default_set(&c), Some("b"));
    }

    #[test]
    fn no_scored_model_no_default() {
        assert_eq!(default_set(&[]), None);
        assert_eq!(default_set(&[cand("a", 0, SetRole::HeldOut, 0, 9)]), None);
    }

    #[test]
    fn ties_keep_the_first() {
        let c = [
            cand("a", 2, SetRole::Public, 2, 10),
            cand("b", 2, SetRole::Public, 2, 10),
        ];
        assert_eq!(default_set(&c), Some("a"));
    }

    fn info(file: &str, tags: &[&str], labelled: bool) -> ImageInfo {
        ImageInfo {
            file: file.into(),
            width: 10,
            height: 10,
            tags: tags.iter().map(ToString::to_string).collect(),
            objects: labelled.then(Vec::new),
        }
    }

    fn set(id: &str, kind: SetKind, images: Vec<ImageInfo>) -> ImageSetInfo {
        ImageSetInfo {
            id: id.into(),
            title: format!("{id} title"),
            kind,
            dir: None,
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: crate::benchmark::images::GroundTruth::Yolo,
            scored_labels: None,
            images,
        }
    }

    #[test]
    fn sets_are_datasets_their_held_out_frames_and_the_lists_models_ran() {
        let mut r = super::super::tests::results();
        r.image_sets = vec![
            set(
                "dir:/export",
                SetKind::Dir,
                vec![
                    info("a.jpg", &["split:train"], true),
                    info("b.jpg", &["split:val"], true),
                    info("c.jpg", &[], false),
                ],
            ),
            set(
                "heldout-x",
                SetKind::Dir,
                vec![info("d.jpg", &["split:val"], true)],
            ),
            set("coco", SetKind::Builtin, vec![info("e.jpg", &[], true)]),
            set("sample", SetKind::Sample, vec![info("f.jpg", &[], false)]),
        ];
        r.models.push(super::super::tests::model(
            "m",
            &["coco", "dir:/export"],
            Vec::new(),
        ));
        let s = compare_sets(&r);
        let ids: Vec<&str> = s.iter().map(|x| x.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "dir:/export#split:val",
                "dir:/export",
                "heldout-x",
                "coco",
                "coco+dir:/export"
            ],
            "the label-free sample is left out; a set wholly held out has no slice"
        );
        assert_eq!(s[0].role, SetRole::HeldOut);
        assert_eq!(s[0].images.len(), 1);
        assert_eq!(s[0].title, "Held-out frames of dir:/export title");
        assert_eq!(s[1].role, SetRole::OwnCamera);
        assert_eq!(s[1].images.len(), 2, "only labelled images");
        assert_eq!(s[2].role, SetRole::HeldOut);
        assert_eq!(s[3].role, SetRole::Public);
        assert_eq!(s[4].images.len(), 3);
        assert_eq!(s[4].title, "coco title + dir:/export title");
    }
}
