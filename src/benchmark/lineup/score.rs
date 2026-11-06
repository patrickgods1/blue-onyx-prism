//! The figures of one model on one [`CompareSet`], rescored from its stored predictions so every
//! model answers for the same objects: one label space shared by the models of the set (car, truck
//! and bus are one `vehicle`, classes not all of them can output are left out), the same ground
//! truth, IoU 0.5, each model at its own confidence. Per image timings give the speed on exactly
//! these images. Pure.

use super::sets::{CompareSet, SetImage};
use crate::benchmark::metrics::{
    AlertStats, ClassMap, Counts, EvalImage, GtBox, PredBox, image_counts, is_vehicle_label,
};
use crate::benchmark::report::DeviceResult;
use crate::benchmark::search::StoredRun;
use crate::benchmark::{ImageRun, RunResult, eval_input};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// One table cell. `n` is what the value rests on: objects for finds, detections for correct,
/// empty frames for false alerts, images for speed. No value = no data, a dash on the page.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct Cell {
    pub value: Option<f64>,
    pub n: usize,
    /// Best of its column, past the noise ([`super::verdict::mark_best`]).
    pub best: bool,
}

/// A figure over the whole set and per scene.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct Scenes {
    pub all: Cell,
    pub day: Cell,
    pub night: Cell,
    pub ir: Cell,
}

pub const SCENES: [&str; 3] = ["day", "night", "ir"];

impl Scenes {
    pub fn get_mut(&mut self, scene: &str) -> Option<&mut Cell> {
        match scene {
            "all" => Some(&mut self.all),
            "day" => Some(&mut self.day),
            "night" => Some(&mut self.night),
            "ir" => Some(&mut self.ir),
            _ => None,
        }
    }
}

/// Finds and Correct of one kind of object.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct Objects {
    pub finds: Scenes,
    pub correct: Scenes,
}

/// The accuracy cells of a model at one confidence.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Scored {
    /// `all`, and `person` / `vehicle` where the set has them.
    pub objects: BTreeMap<String, Objects>,
    pub false_alerts: Cell,
    /// False alerts counted (out of `false_alerts.n` empty frames).
    pub alerts: usize,
}

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

/// An image ready to score, with the tags that split it by scene.
#[derive(Debug)]
pub struct Scorable<'a> {
    pub tags: &'a [String],
    pub eval: EvalImage,
}

/// Every image of `set` in the shared label space `map`, with `run`'s predictions.
pub fn scorable<'a>(
    set: &CompareSet<'a>,
    map: &ClassMap,
    run: &StoredRun,
    index: &Index<'_>,
) -> Vec<Scorable<'a>> {
    set.images
        .iter()
        .filter_map(|img| {
            let k = *index.get(&(img.set.id.as_str(), img.info.file.as_str()))?;
            Some(Scorable {
                tags: &img.info.tags,
                eval: eval_of(img, map, run, k),
            })
        })
        .collect()
}

fn eval_of(img: &SetImage<'_>, map: &ClassMap, run: &StoredRun, k: usize) -> EvalImage {
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

/// The one class of `img`.
fn only(img: &EvalImage, class: &str) -> EvalImage {
    EvalImage {
        gt: img
            .gt
            .iter()
            .filter(|g| g.label == class)
            .cloned()
            .collect(),
        preds: img
            .preds
            .iter()
            .filter(|p| p.label == class)
            .cloned()
            .collect(),
    }
}

fn counts_of<'e>(images: impl Iterator<Item = &'e EvalImage>, t: f32) -> Counts {
    let mut c = Counts::default();
    for i in images {
        c.add(image_counts(i, t).0);
    }
    c
}

fn finds_cell(c: Counts) -> Cell {
    Cell {
        value: c.recall(),
        n: c.tp + c.fn_,
        best: false,
    }
}

fn correct_cell(c: Counts) -> Cell {
    Cell {
        value: c.precision(),
        n: c.tp + c.fp,
        best: false,
    }
}

fn objects_of(images: &[(&[String], &EvalImage)], t: f32) -> Objects {
    let scene = |tag: Option<&str>| {
        counts_of(
            images
                .iter()
                .filter(|(tags, _)| tag.is_none_or(|x| tags.iter().any(|y| y == x)))
                .map(|(_, e)| *e),
            t,
        )
    };
    let (all, day, night, ir) = (
        scene(None),
        scene(Some("day")),
        scene(Some("night")),
        scene(Some("ir")),
    );
    Objects {
        finds: Scenes {
            all: finds_cell(all),
            day: finds_cell(day),
            night: finds_cell(night),
            ir: finds_cell(ir),
        },
        correct: Scenes {
            all: correct_cell(all),
            day: correct_cell(day),
            night: correct_cell(night),
            ir: correct_cell(ir),
        },
    }
}

/// The accuracy cells of `images` at confidence `t`. `classes` is the shared label space.
pub fn score(images: &[Scorable<'_>], classes: &[String], t: f32) -> Scored {
    let mut objects = BTreeMap::new();
    let all: Vec<(&[String], &EvalImage)> = images.iter().map(|i| (i.tags, &i.eval)).collect();
    objects.insert("all".to_string(), objects_of(&all, t));
    for class in ["person", "vehicle"] {
        if !classes.iter().any(|c| c == class) {
            continue;
        }
        let one: Vec<EvalImage> = images.iter().map(|i| only(&i.eval, class)).collect();
        if !one.iter().any(|e| e.gt.iter().any(|g| !g.ignore)) {
            continue;
        }
        let refs: Vec<(&[String], &EvalImage)> =
            images.iter().zip(&one).map(|(i, e)| (i.tags, e)).collect();
        objects.insert(class.to_string(), objects_of(&refs, t));
    }
    let evals: Vec<EvalImage> = images.iter().map(|i| i.eval.clone()).collect();
    let a = AlertStats::of(&evals, classes, t);
    #[allow(
        clippy::cast_precision_loss,
        reason = "frame counts of a benchmark set, far below 2^52"
    )]
    let rate = (a.empty_frames > 0).then(|| a.empty_frame_alerts as f64 / a.empty_frames as f64);
    Scored {
        objects,
        false_alerts: Cell {
            value: rate,
            n: a.empty_frames,
            best: false,
        },
        alerts: a.empty_frame_alerts,
    }
}

/// The median of `v`.
fn median(v: &mut [f64]) -> Option<f64> {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => v.get(n / 2).copied(),
        _ => Some((v.get(n / 2 - 1)? + v.get(n / 2)?) / 2.0),
    }
}

/// The p50 time per image on `set`: over its own images when the run kept their timings, else
/// the run's overall figure (`n` 0: it is not of these images alone).
pub fn speed(set: &CompareSet<'_>, run: &RunResult) -> Cell {
    let by: HashMap<(&str, &str), &ImageRun> = run
        .per_image
        .iter()
        .map(|i| ((i.set.as_str(), i.file.as_str()), i))
        .collect();
    let mut times: Vec<f64> = set
        .images
        .iter()
        .filter_map(|i| by.get(&(i.set.id.as_str(), i.info.file.as_str())))
        .map(|i| i.total_ms)
        .filter(|t| t.is_finite())
        .collect();
    let n = times.len();
    match median(&mut times) {
        Some(v) => Cell {
            value: Some(v),
            n,
            best: false,
        },
        None => Cell {
            value: Some(run.stages_ms.total.p50).filter(|v| v.is_finite()),
            n: 0,
            best: false,
        },
    }
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

    #[test]
    fn median_of_odd_and_even() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn no_object_no_finds_no_empty_frame_no_rate() {
        // Dashes, not zeros: nothing to find, no detection, no empty frame.
        let s = score(&[], &["person".to_string()], 0.5);
        let all = &s.objects["all"];
        assert_eq!(all.finds.all.value, None);
        assert_eq!(all.correct.all.value, None);
        assert_eq!(s.false_alerts.value, None);
        assert!(!s.objects.contains_key("person"));
    }

    #[test]
    fn an_empty_frame_alone_gives_a_rate_of_zero_not_a_dash() {
        let img = Scorable {
            tags: &[],
            eval: EvalImage::default(),
        };
        let s = score(&[img], &["person".to_string()], 0.5);
        assert_eq!(s.false_alerts.value, Some(0.0));
        assert_eq!(s.false_alerts.n, 1);
    }
}
