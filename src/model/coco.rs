//! The COCO class tables and the name mapping built on them, std only. One home so `collect`
//! and `training::autolabel` share them without depending on each other. The Rust twin of
//! `scripts/autolabel.py` (`COCO_NAMES`, `ALIASES`, `coco_class`).

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

/// One class under two spellings: equal ignoring case, or the same COCO class through the
/// aliases (a VOC-spelled model's `motorbike` is the list's `motorcycle`). Also maps a detection
/// label onto the class list ([`map_label`]).
pub fn same_class(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b) || coco_class(a).is_some_and(|c| coco_class(b) == Some(c))
}

/// The class of `classes` a detection `label` is: the same name, else the same name ignoring
/// case, else the same COCO class through the aliases (a VOC-spelled model's `motorbike`,
/// `tvmonitor`, `pottedplant`, `aeroplane`, `sofa`, `diningtable` are the list's `motorcycle`,
/// `tv`, `potted plant`, `airplane`, `couch`, `dining table`). None: not a class of the run.
pub fn map_label<'a>(label: &str, classes: &'a [String]) -> Option<&'a str> {
    let l = label.trim();
    classes
        .iter()
        .find(|c| c.as_str() == l)
        .or_else(|| classes.iter().find(|c| c.eq_ignore_ascii_case(l)))
        .or_else(|| classes.iter().find(|c| same_class(c, l)))
        .map(String::as_str)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    const AUTOLABEL: &str = include_str!("../../scripts/autolabel.py");
    const TRAIN: &str = include_str!("../../scripts/train_model.py");
    const EXPORT: &str = include_str!("../../scripts/export_model.py");
    const COCO_YAML: &str = include_str!("../../assets/coco_classes.yaml");

    /// The `open ... close` literal after `NAME = ` at the start of a line in `script`.
    fn py_literal<'a>(script: &'a str, name: &str, open: char, close: char) -> &'a str {
        let start = script
            .find(&format!("\n{name} = {open}"))
            .unwrap_or_else(|| panic!("{name} not in the script"));
        let rest = &script[start..];
        let o = rest.find(open).unwrap();
        let c = rest.find(close).unwrap();
        &rest[o + 1..c]
    }

    /// The double-quoted strings of a Python literal, in order.
    fn strings(lit: &str) -> Vec<String> {
        lit.split('"')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect()
    }

    fn pairs(lit: &str) -> BTreeMap<String, String> {
        let s = strings(lit);
        assert_eq!(s.len() % 2, 0);
        s.chunks(2).map(|p| (p[0].clone(), p[1].clone())).collect()
    }

    fn rust_aliases() -> BTreeMap<String, String> {
        COCO_ALIASES
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string()))
            .collect()
    }

    #[test]
    fn the_alias_map_and_coco_names_match_the_script() {
        let py = pairs(py_literal(AUTOLABEL, "ALIASES", '{', '}'));
        assert_eq!(py, rust_aliases());
        let py: BTreeSet<String> = strings(py_literal(AUTOLABEL, "COCO_NAMES", '{', '}'))
            .into_iter()
            .collect();
        let rs: BTreeSet<String> = COCO_NAMES.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(py.len(), 80);
        assert_eq!(py, rs);
    }

    /// `train_model.py` maps dataset labels onto the base model's head rows, so it may know
    /// spellings the service does not: its table must be this one plus exactly `table`. Any
    /// other difference is drift.
    #[test]
    fn the_training_alias_map_is_this_one_plus_table() {
        let py = pairs(py_literal(TRAIN, "COCO_ALIASES", '{', '}'));
        let mut expected = rust_aliases();
        expected.insert("table".into(), "dining table".into());
        assert_eq!(py, expected);
    }

    #[test]
    fn the_export_script_and_the_bundled_yaml_list_the_coco_names_in_head_order() {
        let export = strings(py_literal(EXPORT, "COCO80", '[', ']'));
        assert_eq!(export, COCO_NAMES);
        // The bundled list is VOC-spelled (`motorbike`): each entry is the COCO name at its
        // index through the aliases.
        let yaml = crate::model::classes::parse_class_names(COCO_YAML).unwrap();
        assert_eq!(yaml.len(), 80);
        for (i, n) in yaml.iter().enumerate() {
            assert_eq!(coco_class(n), Some(COCO_NAMES[i]), "entry {i}: {n}");
        }
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
    fn same_class_follows_aliases_and_case() {
        assert!(same_class("Motorbike", "motorcycle"));
        assert!(same_class("CAR", "car"));
        assert!(!same_class("car", "truck"));
        assert!(same_class("raccoon", "RACCOON"));
        assert!(!same_class("raccoon", "bear"));
    }
}
