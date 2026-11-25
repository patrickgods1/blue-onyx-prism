//! The shared label space of the combos on a [`CompareSet`] and the lookups that rescore a stored
//! run on it: car, truck and bus are one `vehicle`, the classes every model of the set can output,
//! a run's images by (dataset, file), the evaluation input of one image. Pure.

use super::sets::{CompareSet, SetImage};
use crate::benchmark::RunResult;
use crate::benchmark::eval_input;
use crate::benchmark::metrics::{ClassMap, EvalImage, GtBox, PredBox, is_vehicle_label};
use crate::benchmark::report::DeviceResult;
use crate::benchmark::search::StoredRun;
use std::collections::HashMap;

/// A model label in the shared label space.
pub fn fold(label: &str) -> &str {
    if is_vehicle_label(label) {
        "vehicle"
    } else {
        label
    }
}

/// The scored classes of a stored run in the shared label space.
pub fn folded_classes(run: &StoredRun) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in ClassMap::for_model(&run.classes).classes {
        let f = fold(&c).to_string();
        if !out.contains(&f) {
            out.push(f);
        }
    }
    out
}

/// The classes every list holds, in the first list's order.
pub fn common_classes(lists: &[Vec<String>]) -> Vec<String> {
    let Some(first) = lists.first() else {
        return Vec::new();
    };
    first
        .iter()
        .filter(|c| lists.iter().all(|l| l.contains(c)))
        .cloned()
        .collect()
}

/// (dataset, file) -> position in a stored run.
pub type Index<'r> = HashMap<(&'r str, &'r str), usize>;

pub fn index_of(run: &StoredRun) -> Index<'_> {
    run.images
        .iter()
        .enumerate()
        .map(|(k, i)| ((i.set.as_str(), i.file.as_str()), k))
        .collect()
}

/// How many images of `set` the indexed run scored.
pub fn covered(set: &CompareSet<'_>, index: &Index<'_>) -> usize {
    set.images
        .iter()
        .filter(|i| index.contains_key(&(i.set.id.as_str(), i.info.file.as_str())))
        .count()
}

pub(crate) fn eval_of(img: &SetImage<'_>, map: &ClassMap, run: &StoredRun, k: usize) -> EvalImage {
    let gt: &[GtBox] = img.info.objects.as_deref().unwrap_or_default();
    let preds: Vec<PredBox> = run
        .preds(k)
        .into_iter()
        .map(|p| PredBox {
            label: fold(&p.label).to_string(),
            ..p
        })
        .collect();
    eval_input(map, img.set.scored_labels.as_deref(), gt, &preds).1
}

/// The run of a device that ran itself (no fallback to another device).
pub fn usable(d: &DeviceResult) -> Option<&RunResult> {
    d.run.as_ref().filter(|r| !r.fell_back)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding_merges_only_the_vehicle_group() {
        assert_eq!(fold("car"), "vehicle");
        assert_eq!(fold("truck"), "vehicle");
        assert_eq!(fold("vehicle"), "vehicle");
        assert_eq!(fold("person"), "person");
        assert_eq!(fold("motorcycle"), "motorcycle");
    }

    #[test]
    fn common_classes_keep_the_first_order() {
        let l = |v: &[&str]| v.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert_eq!(
            common_classes(&[l(&["person", "vehicle", "dog"]), l(&["dog", "person"])]),
            ["person", "dog"]
        );
        assert!(common_classes(&[]).is_empty());
    }
}
