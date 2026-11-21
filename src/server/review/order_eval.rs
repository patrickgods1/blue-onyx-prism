//! A measurement, not a check: the Review tab's default order on a real review folder before the
//! need model ([`queue::build_legacy`]: event rank as a strict tier, then the balance of class,
//! scene and camera) and after it ([`queue::build`] with a [`Steer`]: no spot table, then the
//! folder's own), tallied over the first 100 and 500 frames served. Does nothing unless
//! `BOP_ORDER_EVAL_DIR` names the folder, so the `--ignored` runs of CI pass it by. Only reads:
//! `queue::build` and the spot table's walk ([`super::super::refine_spots::spot_table`]) write
//! nothing, and no review index is made.
//!
//! ```text
//! BOP_ORDER_EVAL_DIR=D:\dataset BOP_ORDER_EVAL_CONFIG=<config json> BOP_ORDER_EVAL_RUNS=<runs.json> \
//!   BOP_ORDER_EVAL_OUT=<folder> cargo test order_eval -- --ignored --nocapture
//! ```
//!
//! `BOP_ORDER_EVAL_CONFIG` gives `collect.classes` and `collect.hidden_classes` (else the default
//! class list), `BOP_ORDER_EVAL_RUNS` the recall table (else counts only), `BOP_ORDER_EVAL_OUT`
//! where `report.md` goes (else the temp folder). Per frame the class, the scene and the cues are
//! those of the steered need, so both orders are tallied alike.

#![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::print_stdout)]

use crate::collect::labels::need::{self, RecallNeed, Steer};
use crate::collect::labels::queue::{self, Filters, Item, Order, Queue};
use crate::collect::labels::{Frame, recommend, statics};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The median of three runs of `f`, and its last answer.
fn median_of_3<T>(mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..3 {
        let t = Instant::now();
        last = Some(f());
        times.push(t.elapsed());
    }
    times.sort();
    (times[1], last.unwrap())
}

/// `collect.classes` and `collect.hidden_classes` of the config at `path`.
fn class_list(path: Option<&Path>) -> (Vec<String>, Vec<String>) {
    let strings = |v: &serde_json::Value| -> Vec<String> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    let Some(v) = path
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
    else {
        return (
            crate::collect::config::CollectConfig::default().classes,
            Vec::new(),
        );
    };
    (
        strings(&v["collect"]["classes"]),
        strings(&v["collect"]["hidden_classes"]),
    )
}

/// What one served frame is, for the tallies.
#[derive(Debug, Clone, Default)]
struct Seen {
    camera: String,
    scene: String,
    class: String,
    need: Option<RecallNeed>,
    /// Candidates of an animal class, of one under the floor, and missed ones.
    animal: bool,
    rare_animal: bool,
    person: bool,
    missed_boxes: usize,
    spot_boxes: usize,
}

/// The per-frame facts of the frames of `q` (the steered order): its need and candidates.
fn facts_of(dir: &Path, q: &Queue, f: &Filters, spots: &statics::Table) -> HashMap<String, Seen> {
    let (classes, hidden) = f.wanted.clone().unwrap_or_default();
    let w = crate::collect::labels::bulk::Wanted {
        classes: &classes,
        hidden: &hidden,
    };
    let floor: BTreeMap<String, f64> = q
        .need
        .as_ref()
        .map(|c| {
            c.boxes
                .iter()
                .map(|(k, s)| (k.clone(), s.values().sum::<f64>()))
                .collect()
        })
        .unwrap_or_default();
    q.items
        .iter()
        .map(|i| {
            let boxes = Frame::resolve(dir, &i.frame)
                .and_then(|fr| {
                    let rec = fr.record().ok()?;
                    let l = fr.labels().ok()?;
                    Some(match l {
                        Some(l) => (need::of_file(&l, Some(&w)), rec.width, rec.height),
                        None => (need::of_reply(&rec, Some(&w)), rec.width, rec.height),
                    })
                })
                .unwrap_or_default();
            let (boxes, width, height) = boxes;
            let name = |k: u16| classes.get(usize::from(k)).cloned().unwrap_or_default();
            let animal = |k: u16| {
                crate::training::autolabel::Category::of(&name(k))
                    == crate::training::autolabel::Category::Animal
            };
            let rare =
                |k: u16| floor.get(&name(k)).copied().unwrap_or(0.0) < need::CLASS_FLOOR as f64;
            let at_spot = |b: &need::CellBox| {
                statics::normalised(&b.bbox.map(f32::from), width, height)
                    .is_some_and(|nb| spots.spot_at(&i.camera, &name(b.class), &nb).is_some())
            };
            let n = i.uncertainty.need.clone();
            (
                i.frame.clone(),
                Seen {
                    camera: i.camera.clone(),
                    scene: i.scene.clone(),
                    class: n
                        .as_ref()
                        .and_then(|n| n.class.clone())
                        .unwrap_or_else(|| "(empty)".into()),
                    animal: boxes.iter().any(|b| animal(b.class)),
                    rare_animal: boxes.iter().any(|b| animal(b.class) && rare(b.class)),
                    person: boxes.iter().any(|b| name(b.class) == "person"),
                    missed_boxes: boxes.iter().filter(|b| b.has(need::MISSED)).count(),
                    spot_boxes: boxes.iter().filter(|b| at_spot(b)).count(),
                    need: n,
                },
            )
        })
        .collect()
}

/// The tallies of the first `n` frames of `items`.
fn tally(items: &[Item], n: usize, facts: &HashMap<String, Seen>, out: &mut String, title: &str) {
    let first: Vec<&Seen> = items
        .iter()
        .take(n)
        .filter_map(|i| facts.get(&i.frame))
        .collect();
    let mut cells: BTreeMap<(String, String, String), usize> = BTreeMap::new();
    for s in &first {
        *cells
            .entry((s.camera.clone(), s.class.clone(), s.scene.clone()))
            .or_default() += 1;
    }
    let count = |p: &dyn Fn(&Seen) -> bool| first.iter().filter(|s| p(s)).count();
    let sum = |p: &dyn Fn(&Seen) -> usize| first.iter().map(|s| p(s)).sum::<usize>();
    let _ = writeln!(out, "\n#### {title}: first {n}\n");
    let _ = writeln!(
        out,
        "| frames | animal | rare animal (< {} boxes) | IR/night person | missed-cue boxes | gap frames | parked-spot frames | spot boxes | expanded visits |",
        need::CLASS_FLOOR
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|");
    let _ = writeln!(
        out,
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
        first.len(),
        count(&|s| s.animal),
        count(&|s| s.rare_animal),
        count(&|s| s.person && (s.scene == "ir" || s.scene == "night")),
        sum(&|s| s.missed_boxes),
        count(&|s| s.need.as_ref().is_some_and(|n| n.gap)),
        count(&|s| s.spot_boxes > 0),
        sum(&|s| s.spot_boxes),
        count(&|s| s.need.as_ref().is_some_and(|n| n.expanded)),
    );
    let _ = writeln!(
        out,
        "\n| camera | top class | scene | frames |\n|---|---|---|---|"
    );
    let mut rows: Vec<_> = cells.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for ((camera, class, scene), n) in rows {
        let _ = writeln!(out, "| {camera} | {class} | {scene} | {n} |");
    }
}

/// Every fifth slot (from the decided count's phase) is a random pick while both sides last.
fn random_slots_hold(q: &Queue, upto: usize) -> (usize, usize) {
    let decided = q.counts.reviewed + q.counts.skipped;
    let pool = q.items.iter().filter(|i| i.random_pick).count();
    let rest = q.items.len() - pool;
    let (mut used_pool, mut used_rest, mut checked, mut wrong) = (0, 0, 0, 0);
    for (slot, item) in q.items.iter().enumerate().take(upto) {
        if used_pool < pool && used_rest < rest {
            let random =
                (decided + slot as u64) % recommend::RANDOM_EVERY == recommend::RANDOM_EVERY - 1;
            checked += 1;
            if item.random_pick != random || item.random_pick != recommend::in_pool(&item.frame) {
                wrong += 1;
            }
        }
        if item.random_pick {
            used_pool += 1;
        } else {
            used_rest += 1;
        }
    }
    (checked, wrong)
}

#[test]
#[ignore = "a measurement on a real review folder: set BOP_ORDER_EVAL_DIR, see the file's first lines"]
fn the_recommended_order_before_and_after_the_need_model() {
    let Ok(dir) = std::env::var("BOP_ORDER_EVAL_DIR") else {
        println!("skipped: BOP_ORDER_EVAL_DIR is not set");
        return;
    };
    let dir = PathBuf::from(dir);
    let config = std::env::var("BOP_ORDER_EVAL_CONFIG")
        .ok()
        .map(PathBuf::from);
    let wanted = class_list(config.as_deref());
    let recall = std::env::var("BOP_ORDER_EVAL_RUNS").ok().and_then(|p| {
        crate::training::bench::measured_recall(&crate::training::service::load_bare_records(
            Path::new(&p),
        ))
        .map(Arc::new)
    });
    let out_dir = std::env::var("BOP_ORDER_EVAL_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("order-eval"));
    // The Review tab's default view.
    let base = Filters {
        order: Order::Recommended,
        wanted: Some(wanted.clone()),
        ..Filters::default()
    };
    let t = Instant::now();
    let spots = Arc::new(super::super::refine_spots::spot_table(
        &dir, &wanted.0, &wanted.1,
    ));
    let spot_ms = t.elapsed();
    let empty = Arc::new(Steer {
        recall: recall.clone(),
        ..Steer::default()
    });
    let real = Arc::new(Steer {
        recall: recall.clone(),
        spots: spots.clone(),
        heldout_listed: Arc::default(),
    });
    let with = |s: &Arc<Steer>| Filters {
        steer: Some(s.clone()),
        ..base.clone()
    };
    let (old_ms, old) = median_of_3(|| queue::build_legacy(&dir, &base));
    let (new_empty_ms, new_empty) = median_of_3(|| queue::build(&dir, &with(&empty)));
    let (new_ms, new) = median_of_3(|| queue::build(&dir, &with(&real)));
    let facts = facts_of(&dir, &new, &with(&real), &spots);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# Recommended order before and after the need model\n\nFolder `{}`, {} frames listed \
         (default view), {} reviewed, {} skipped. Classes {:?} (hidden {:?}). Recall table: {}. \
         Parked spots: {} on {} cameras (built in {:.0} ms).",
        dir.display(),
        new.items.len(),
        new.counts.reviewed,
        new.counts.skipped,
        wanted.0,
        wanted.1,
        recall.as_ref().map_or_else(
            || "none (no run with per-class held-out recall)".to_string(),
            |m| format!(
                "{} scene slice(s), overall {}",
                m.scenes.len(),
                m.overall.is_some()
            )
        ),
        spots.len(),
        spots.cameras.len(),
        spot_ms.as_secs_f64() * 1000.0
    );
    if let Some(c) = &new.need {
        let _ = writeln!(out, "\nReviewed boxes per class (floor {}):", c.floor);
        for (class, scenes) in &c.boxes {
            let _ = writeln!(out, "- {class}: {scenes:?}");
        }
        let _ = writeln!(out, "- empty frames: {:?}", c.empty);
    }
    let _ = writeln!(
        out,
        "\n## Walk time (`queue::build`, median of 3)\n\n| order | ms |\n|---|---|\n| before (legacy) | {:.0} |\n| need, no spot table | {:.0} |\n| need, spot table | {:.0} |",
        old_ms.as_secs_f64() * 1000.0,
        new_empty_ms.as_secs_f64() * 1000.0,
        new_ms.as_secs_f64() * 1000.0
    );
    assert_eq!(old.items.len(), new.items.len(), "the same frames listed");
    for (title, q) in [
        ("Before (legacy)", &old),
        ("Need, no spot table", &new_empty),
        ("Need, spot table", &new),
    ] {
        let _ = writeln!(out, "\n### {title}");
        for n in [100, 500] {
            tally(&q.items, n, &facts, &mut out, title);
        }
        let all = |p: &dyn Fn(&RecallNeed) -> bool| {
            q.items
                .iter()
                .filter(|i| {
                    facts
                        .get(&i.frame)
                        .and_then(|s| s.need.as_ref())
                        .is_some_and(p)
                })
                .count()
        };
        let _ = writeln!(
            out,
            "\nWhole list ({} frames): {} with an expanded visit, {} with a gap, {} at a parked \
             spot, {} of event rank > 0.",
            q.items.len(),
            all(&|n| n.expanded),
            all(&|n| n.gap),
            all(&|n| n.spot),
            q.items.iter().filter(|i| i.event_rank > Some(0)).count()
        );
        let (checked, wrong) = random_slots_hold(q, 500);
        let _ = writeln!(
            out,
            "\nRandom slots in the first 500: {checked} checked, {wrong} out of place; {} random picks.",
            q.items.iter().take(500).filter(|i| i.random_pick).count()
        );
        if title != "Before (legacy)" {
            assert_eq!(wrong, 0, "{title}: every fifth slot is a random pick");
        }
    }
    // The first 30 served of the steered order, with their notes.
    let _ = writeln!(
        out,
        "\n## First 30 served (need, spot table)\n\n| # | frame | rank | random | need | visit | class | cues |\n|---|---|---|---|---|---|---|---|"
    );
    for (i, item) in new.items.iter().take(30).enumerate() {
        let n = item.uncertainty.need.clone().unwrap_or_default();
        let cues: Vec<&str> = [
            (n.below_floor, "floor"),
            (n.catch_up, "catch-up"),
            (n.missed, "missed"),
            (n.gap, "gap"),
            (n.spot, "spot"),
            (n.new_region, "new place"),
            (n.new_day, "new day"),
            (n.expanded, "expanded"),
            (n.model_lacks, "model lacks"),
        ]
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, s)| *s)
        .collect();
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            i + 1,
            item.frame,
            item.event_rank.unwrap_or(0),
            item.random_pick,
            n.score,
            n.visit_score,
            n.class.unwrap_or_else(|| "(empty)".into()),
            cues.join(", ")
        );
    }
    // What the first five frames' needs are made of: their candidates (class, evidence, flags).
    let _ = writeln!(
        out,
        "\n## Candidates of the first five\n\nflags: 1 training, 2 missed, 4 strong, 8 gap\n"
    );
    let (classes, hidden) = wanted.clone();
    let w = crate::collect::labels::bulk::Wanted {
        classes: &classes,
        hidden: &hidden,
    };
    for item in new.items.iter().take(5) {
        let Some(fr) = Frame::resolve(&dir, &item.frame) else {
            continue;
        };
        let boxes = match fr.labels().ok().flatten() {
            Some(l) => need::of_file(&l, Some(&w)),
            None => fr
                .record()
                .map(|r| need::of_reply(&r, Some(&w)))
                .unwrap_or_default(),
        };
        let list: Vec<String> = boxes
            .iter()
            .map(|b| {
                format!(
                    "{} e{} f{}",
                    classes
                        .get(usize::from(b.class))
                        .map_or("?", String::as_str),
                    b.evidence,
                    b.flags
                )
            })
            .collect();
        let _ = writeln!(out, "- {}: {}", item.frame, list.join("; "));
    }
    std::fs::create_dir_all(&out_dir).unwrap();
    std::fs::write(out_dir.join("report.md"), &out).unwrap();
    println!("{out}");
    println!("written: {}", out_dir.join("report.md").display());
}
