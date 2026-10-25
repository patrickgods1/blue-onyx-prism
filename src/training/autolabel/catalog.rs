//! The class catalog behind the Review page's class picker (`GET /v1/review/class-catalog`):
//! every class name worth offering (the review list `collect.classes`, the auto-labeler's
//! default phrases, its vehicle and animal names, the 80 COCO classes RF-DETR knows, and the
//! classes of every configured model), each with its category, its prompt phrases, whether
//! RF-DETR can propose it ([`coco_class`], the Rust twin of `scripts/autolabel.py`'s
//! `coco_class`) and which models emit it; plus [`relevant_classes`], the review classes the
//! enabled models actually report (their object filters applied). Pure apart from
//! [`model_classes`] and [`ClassCatalog::for_config`], which read the models' class files.

use super::{ANIMALS, Category, DEFAULT_SYNONYMS, VEHICLES, synonyms};
use crate::config::Config;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// The 80 COCO class names (RF-DETR L's label space), as `scripts/autolabel.py` spells them.
pub const COCO_NAMES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Other spellings of COCO classes (`scripts/autolabel.py` `ALIASES`; the VOC names of the
/// bundled `coco_classes.yaml` among them).
pub const COCO_ALIASES: [(&str, &str); 9] = [
    ("motorbike", "motorcycle"),
    ("people", "person"),
    ("pedestrian", "person"),
    ("bike", "bicycle"),
    ("sofa", "couch"),
    ("tvmonitor", "tv"),
    ("aeroplane", "airplane"),
    ("pottedplant", "potted plant"),
    ("diningtable", "dining table"),
];

/// The COCO name of a class (aliases applied, case-insensitive), or None: RF-DETR L can propose
/// it. Mirrors `coco_class` of `scripts/autolabel.py`.
pub fn coco_class(label: &str) -> Option<&'static str> {
    let l = label.to_lowercase();
    let n = COCO_ALIASES
        .iter()
        .find(|(a, _)| *a == l)
        .map_or(l.as_str(), |(_, c)| c);
    COCO_NAMES.iter().copied().find(|c| *c == n)
}

/// A configured model's classes, as the catalog answers them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelClasses {
    /// The model's effective name.
    pub name: String,
    pub enabled: bool,
    /// Its class names in head order (empty when they could not be read).
    pub classes: Vec<String>,
    /// The model's own `object_filter`: None = the global one, empty = every class.
    pub object_filter: Option<Vec<String>>,
    /// Why its class names could not be read.
    pub error: Option<String>,
}

/// The classes of every configured model (disabled ones too), read from their class files
/// (`classes`, else `<model>.yaml`, else COCO-80, like the registry). A model whose file and
/// class file are both missing gets an error instead of COCO-80 (its names are not known
/// until it is downloaded).
pub fn model_classes(config: &Config) -> Vec<ModelClasses> {
    config
        .models
        .iter()
        .map(|m| {
            let path = config.data_path(&m.path);
            let classes_path = m.classes.as_deref().map(|c| config.data_path(c));
            let res = if classes_path.is_none()
                && !path.is_file()
                && !path.with_extension("yaml").is_file()
            {
                Err(format!(
                    "{} is not there yet (not downloaded or exported); its classes are known \
                     once it is",
                    path.display()
                ))
            } else {
                crate::model::classes::resolve_class_names(&path, classes_path.as_deref())
                    .map_err(|e| format!("{e:#}"))
            };
            let (classes, error) = match res {
                Ok(c) => (c, None),
                Err(e) => (Vec::new(), Some(e)),
            };
            ModelClasses {
                name: m.effective_name(),
                enabled: m.enabled,
                classes,
                object_filter: m.object_filter.clone(),
                error,
            }
        })
        .collect()
}

fn contains_ci(list: &[String], name: &str) -> bool {
    list.iter().any(|x| x.eq_ignore_ascii_case(name))
}

/// One class under two spellings: equal ignoring case, or the same COCO class through the
/// aliases (a VOC-spelled model's `motorbike` is the list's `motorcycle`).
fn same_class(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b) || coco_class(a).is_some_and(|c| coco_class(b) == Some(c))
}

fn contains_same(list: &[String], name: &str) -> bool {
    list.iter().any(|x| same_class(x, name))
}

/// The review classes the enabled models report: per enabled model its effective filter (its
/// own `object_filter`, else `global_filter`; empty = every class of the model) limited to the
/// model's classes (a filter name the model does not have never matches; with unreadable class
/// names the filter is taken as is), united over the models, then intersected with
/// `review_classes` (case-insensitive), in `review_classes` order and spelling.
pub fn relevant_classes(
    review_classes: &[String],
    global_filter: &[String],
    models: &[ModelClasses],
) -> Vec<String> {
    let mut emitted: Vec<String> = Vec::new();
    for m in models.iter().filter(|m| m.enabled) {
        let filter = m.object_filter.as_deref().unwrap_or(global_filter);
        if filter.is_empty() {
            emitted.extend(m.classes.iter().cloned());
        } else if m.error.is_some() && m.classes.is_empty() {
            emitted.extend(filter.iter().cloned());
        } else {
            emitted.extend(
                filter
                    .iter()
                    .filter(|f| contains_ci(&m.classes, f))
                    .cloned(),
            );
        }
    }
    review_classes
        .iter()
        .filter(|c| contains_same(&emitted, c))
        .cloned()
        .collect()
}

/// One class of the catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CatalogEntry {
    pub name: String,
    pub category: Category,
    /// In `collect.classes`.
    pub in_list: bool,
    /// In `collect.hidden_classes` (kept, hidden from the Review pickers).
    pub hidden: bool,
    /// RF-DETR L (COCO) can propose it; otherwise MM-Grounding-DINO alone, always `review`.
    pub coco: bool,
    /// Configured models that have this class.
    pub models: Vec<String>,
    /// MM-Grounding-DINO prompt phrases ([`synonyms`]).
    pub phrases: Vec<String>,
}

fn category_rank(c: Category) -> u8 {
    match c {
        Category::Person => 0,
        Category::Vehicle => 1,
        Category::Animal => 2,
        Category::Object => 3,
    }
}

/// The catalog entries: every name of `review_classes`, `hidden`, the default phrase keys,
/// [`VEHICLES`], [`ANIMALS`], [`COCO_NAMES`] and the models' classes, once each
/// (case-insensitive; the first spelling wins, `review_classes` first). Order: `review_classes`
/// as listed, then the rest by category (person, vehicle, animal, object), alphabetical
/// (case-insensitive) within.
pub fn class_catalog(
    review_classes: &[String],
    hidden: &[String],
    overrides: &BTreeMap<String, Vec<String>>,
    models: &[ModelClasses],
) -> Vec<CatalogEntry> {
    let mut seen = BTreeSet::new();
    let mut names: Vec<String> = Vec::new();
    let mut add = |names: &mut Vec<String>, n: &str| {
        let n = n.trim();
        // Alias spellings of one COCO class (`pottedplant`, `potted plant`) are one entry.
        let key = coco_class(n).map_or_else(|| n.to_lowercase(), str::to_string);
        if !n.is_empty() && seen.insert(key) {
            names.push(n.to_string());
        }
    };
    for n in review_classes {
        add(&mut names, n);
    }
    let listed = names.len();
    for n in hidden {
        add(&mut names, n);
    }
    for n in DEFAULT_SYNONYMS
        .iter()
        .map(|(k, _)| *k)
        .chain(VEHICLES)
        .chain(ANIMALS)
        .chain(COCO_NAMES)
    {
        add(&mut names, n);
    }
    for m in models {
        for n in &m.classes {
            add(&mut names, n);
        }
    }
    if let Some(rest) = names.get_mut(listed..) {
        rest.sort_by(|a, b| {
            category_rank(Category::of(a))
                .cmp(&category_rank(Category::of(b)))
                .then_with(|| a.to_lowercase().cmp(&b.to_lowercase()))
                .then_with(|| a.cmp(b))
        });
    }
    let phrases = synonyms(&names, overrides);
    names
        .iter()
        .map(|n| CatalogEntry {
            name: n.clone(),
            category: Category::of(n),
            in_list: contains_ci(review_classes, n),
            hidden: contains_ci(hidden, n),
            coco: coco_class(n).is_some(),
            models: models
                .iter()
                .filter(|m| contains_same(&m.classes, n))
                .map(|m| m.name.clone())
                .collect(),
            phrases: phrases.get(n).cloned().unwrap_or_else(|| vec![n.clone()]),
        })
        .collect()
}

/// `GET /v1/review/class-catalog` (see docs/TRAINING.md "Choosing classes").
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClassCatalog {
    /// `collect.classes`, in order.
    pub classes: Vec<String>,
    /// `collect.hidden_classes`.
    pub hidden: Vec<String>,
    /// [`relevant_classes`].
    pub relevant: Vec<String>,
    /// The global `object_filter`.
    pub global_object_filter: Vec<String>,
    pub models: Vec<ModelClasses>,
    pub entries: Vec<CatalogEntry>,
}

impl ClassCatalog {
    /// From the parts (pure).
    pub fn build(config: &Config, models: Vec<ModelClasses>) -> Self {
        let c = &config.collect;
        Self {
            relevant: relevant_classes(&c.classes, &config.object_filter, &models),
            entries: class_catalog(
                &c.classes,
                &c.hidden_classes,
                &config.training.synonyms,
                &models,
            ),
            classes: c.classes.clone(),
            hidden: c.hidden_classes.clone(),
            global_object_filter: config.object_filter.clone(),
            models,
        }
    }

    /// Of `config`, reading the models' class files.
    pub fn for_config(config: &Config) -> Self {
        Self::build(config, model_classes(config))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    const SCRIPT: &str = include_str!("../../../scripts/autolabel.py");

    /// The `{...}` literal after `NAME = ` in the script.
    fn py_literal(name: &str) -> &'static str {
        let start = SCRIPT
            .find(&format!("\n{name} = {{"))
            .unwrap_or_else(|| panic!("{name} not in autolabel.py"));
        let rest = &SCRIPT[start..];
        let open = rest.find('{').unwrap();
        let close = rest.find('}').unwrap();
        &rest[open + 1..close]
    }

    /// The double-quoted strings of a Python literal.
    fn strings(lit: &str) -> Vec<String> {
        lit.split('"')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn the_alias_map_and_coco_names_match_the_script() {
        let s = strings(py_literal("ALIASES"));
        assert_eq!(s.len() % 2, 0);
        let py: BTreeMap<String, String> =
            s.chunks(2).map(|p| (p[0].clone(), p[1].clone())).collect();
        let rs: BTreeMap<String, String> = COCO_ALIASES
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string()))
            .collect();
        assert_eq!(py, rs);
        let py: BTreeSet<String> = strings(py_literal("COCO_NAMES")).into_iter().collect();
        let rs: BTreeSet<String> = COCO_NAMES.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(py.len(), 80);
        assert_eq!(py, rs);
    }

    #[test]
    fn coco_class_applies_aliases() {
        assert_eq!(coco_class("motorbike"), Some("motorcycle"));
        assert_eq!(coco_class("Person"), Some("person"));
        assert_eq!(coco_class("TVMonitor"), Some("tv"));
        assert_eq!(coco_class("raccoon"), None);
        assert_eq!(coco_class("vehicle"), None);
        // The bundled (VOC-spelled) COCO-80 list: every name maps onto a COCO name.
        let unmapped: Vec<String> = crate::model::classes::coco80()
            .into_iter()
            .filter(|n| coco_class(n).is_none())
            .collect();
        assert!(unmapped.is_empty(), "{unmapped:?}");
    }

    #[test]
    fn alias_spellings_are_one_class() {
        let voc = ModelClasses {
            name: "voc".into(),
            enabled: true,
            classes: s(&["motorbike", "pottedplant", "person"]),
            object_filter: None,
            error: None,
        };
        let review = s(&["person", "motorcycle", "raccoon"]);
        assert_eq!(
            relevant_classes(&review, &[], std::slice::from_ref(&voc)),
            ["person", "motorcycle"]
        );
        let cat = class_catalog(&review, &[], &BTreeMap::new(), &[voc]);
        let moto: Vec<_> = cat
            .iter()
            .filter(|e| coco_class(&e.name) == Some("motorcycle"))
            .collect();
        assert_eq!(moto.len(), 1);
        assert_eq!(
            (moto[0].name.as_str(), moto[0].models.as_slice()),
            ("motorcycle", &["voc".to_string()][..])
        );
        assert_eq!(
            cat.iter()
                .filter(|e| coco_class(&e.name) == Some("potted plant"))
                .count(),
            1
        );
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    fn model(name: &str, enabled: bool, classes: &[&str], filter: Option<&[&str]>) -> ModelClasses {
        ModelClasses {
            name: name.into(),
            enabled,
            classes: s(classes),
            object_filter: filter.map(s),
            error: None,
        }
    }

    #[test]
    fn relevant_classes_follow_the_enabled_models_filters() {
        let review = s(&["Person", "car", "dog", "raccoon", "bird"]);
        let general = model("general", true, &["person", "vehicle", "dog"], None);
        let coco = model(
            "coco",
            true,
            &["person", "car", "dog", "bird"],
            Some(&["car"]),
        );
        let off = model("off", false, &["raccoon"], None);
        // Global filter empty: general = all its classes; coco = car only; off ignored.
        assert_eq!(
            relevant_classes(&review, &[], &[general.clone(), coco.clone(), off.clone()]),
            ["Person", "car", "dog"],
            "review order and spelling"
        );
        // Global filter applies to general only; a name the model lacks never matches.
        assert_eq!(
            relevant_classes(&review, &s(&["PERSON", "bird"]), &[general.clone(), coco]),
            ["Person", "car"]
        );
        // An empty own filter = every class, even with a global filter.
        let all = model("all", true, &["bird"], Some(&[]));
        assert_eq!(relevant_classes(&review, &s(&["person"]), &[all]), ["bird"]);
        // Unreadable class names: the filter as is; without a filter nothing.
        let mut broken = model("broken", true, &[], Some(&["raccoon"]));
        broken.error = Some("x".into());
        assert_eq!(
            relevant_classes(&review, &[], &[broken.clone()]),
            ["raccoon"]
        );
        broken.object_filter = None;
        assert!(relevant_classes(&review, &[], &[broken]).is_empty());
        assert!(relevant_classes(&review, &[], &[off]).is_empty());
    }

    #[test]
    fn the_catalog_lists_review_classes_first_then_by_category() {
        let review = s(&["raccoon", "Person", "package"]);
        let hidden = s(&["package"]);
        let over: BTreeMap<String, Vec<String>> =
            [("package".to_string(), s(&["package", "parcel"]))].into();
        let models = [
            model("general", true, &["person", "vehicle", "Dog"], None),
            model("coco", false, &["person", "car"], None),
        ];
        let e = class_catalog(&review, &hidden, &over, &models);
        let names: Vec<&str> = e.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(&names[..3], ["raccoon", "Person", "package"]);
        // Once each, case-insensitively.
        let lower: BTreeSet<String> = names.iter().map(|n| n.to_lowercase()).collect();
        assert_eq!(lower.len(), names.len());
        assert!(!names.contains(&"person"));
        assert!(
            names.contains(&"dog") && !names.contains(&"Dog"),
            "first spelling wins"
        );
        // The rest: vehicles, then animals, then objects, alphabetical within.
        let rest = &e[3..];
        let ranks: Vec<u8> = rest.iter().map(|x| category_rank(x.category)).collect();
        assert!(ranks.windows(2).all(|w| w[0] <= w[1]), "{ranks:?}");
        assert_eq!(rest[0].category, Category::Vehicle);
        assert_eq!(rest[0].name, "airplane");
        let objects: Vec<&str> = rest
            .iter()
            .filter(|x| x.category == Category::Object)
            .map(|x| x.name.as_str())
            .collect();
        assert_eq!(objects[0], "apple");
        assert!(objects.contains(&"vehicle") && objects.contains(&"toothbrush"));
        let by = |n: &str| e.iter().find(|x| x.name == n).unwrap();
        let p = by("package");
        assert!(p.in_list && p.hidden && !p.coco);
        assert_eq!(p.phrases, ["package", "parcel"]);
        assert_eq!(p.category, Category::Object);
        let person = by("Person");
        assert!(person.in_list && !person.hidden && person.coco);
        assert_eq!(person.models, ["general", "coco"]);
        assert_eq!(person.phrases, ["person", "pedestrian"]);
        let r = by("raccoon");
        assert!(!r.coco && r.models.is_empty());
        assert_eq!(r.category, Category::Animal);
        let car = by("car");
        assert!(!car.in_list && car.coco);
        assert_eq!(car.phrases, ["car", "van", "pickup truck"]);
        assert_eq!(car.models, ["coco"]);
        assert_eq!(by("vehicle").models, ["general"]);
        assert_eq!(by("hair drier").phrases, ["hair drier"]);
    }

    #[test]
    fn the_catalog_of_a_config_reads_model_classes() {
        let dir = std::env::temp_dir().join(format!("bop-catalog-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.onnx"), b"x").unwrap();
        std::fs::write(dir.join("m.yaml"), "names: [person, raccoon]\n").unwrap();
        let mut c = Config {
            download_dir: Some(dir.clone()),
            ..Config::default()
        };
        c.models = vec![
            crate::config::ModelConfig {
                name: Some("m".into()),
                path: dir.join("m.onnx"),
                ..crate::config::ModelConfig::default()
            },
            crate::config::ModelConfig {
                name: Some("gone".into()),
                path: dir.join("gone.onnx"),
                ..crate::config::ModelConfig::default()
            },
            crate::config::ModelConfig {
                name: Some("badclasses".into()),
                path: dir.join("m.onnx"),
                classes: Some(dir.join("nope.yaml")),
                ..crate::config::ModelConfig::default()
            },
        ];
        c.collect.hidden_classes = vec!["bird".into()];
        let cat = ClassCatalog::for_config(&c);
        assert_eq!(cat.models[0].classes, ["person", "raccoon"]);
        assert_eq!(cat.models[0].error, None);
        assert!(
            cat.models[1]
                .error
                .as_deref()
                .unwrap()
                .contains("not there yet")
        );
        assert!(cat.models[2].error.is_some() && cat.models[2].classes.is_empty());
        assert_eq!(cat.relevant, ["person", "raccoon"]);
        assert_eq!(cat.hidden, ["bird"]);
        assert_eq!(cat.classes, c.collect.classes);
        let v = serde_json::to_value(&cat).unwrap();
        for k in [
            "classes",
            "hidden",
            "relevant",
            "global_object_filter",
            "models",
            "entries",
        ] {
            assert!(v.get(k).is_some(), "{k}");
        }
        assert_eq!(v["models"][0]["object_filter"], serde_json::Value::Null);
        assert_eq!(v["models"][0]["error"], serde_json::Value::Null);
        let e = &v["entries"][0];
        assert_eq!(e["name"], "person");
        assert_eq!(e["category"], "person");
        for k in ["in_list", "hidden", "coco", "models", "phrases"] {
            assert!(e.get(k).is_some(), "{k}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
