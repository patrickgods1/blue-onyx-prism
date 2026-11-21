//! A measurement, not a check: the Review tab's default order on a real review folder before the
//! need model ([`queue::build_legacy`]: event rank as a strict tier, then the balance of class,
//! scene and camera) and after it ([`queue::build`] with a [`Steer`]: no spot table, then the
//! folder's own, then with held-out steering: the folder's `heldout-days.json`, read only, and the
//! hash split), tallied over the first 100 and 500 frames served, with how many are on held-out
//! days by camera and by class. Does nothing unless `BOP_ORDER_EVAL_DIR` names the folder, so the
//! `--ignored` runs of CI pass it by. Only reads: `queue::build`, the spot table's walk
//! ([`queue::reviewed_obs`], the service's path for a folder without an index) and
//! [`crate::collect::labels::heldout::load`] write nothing, and no review index is made.
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
    /// On a held-out day (listed or hashed).
    heldout: bool,
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
                    heldout: false,
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
        "| frames | animal | rare animal (< {} boxes) | IR/night person | missed-cue boxes | gap frames | parked-spot frames | spot boxes | expanded visits | spot-led | hard-negative |",
        need::CLASS_FLOOR
    );
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|");
    let _ = writeln!(
        out,
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
        first.len(),
        count(&|s| s.animal),
        count(&|s| s.rare_animal),
        count(&|s| s.person && (s.scene == "ir" || s.scene == "night")),
        sum(&|s| s.missed_boxes),
        count(&|s| s.need.as_ref().is_some_and(|n| n.gap)),
        count(&|s| s.spot_boxes > 0),
        sum(&|s| s.spot_boxes),
        count(&|s| s.need.as_ref().is_some_and(|n| n.expanded)),
        count(&|s| s.need.as_ref().is_some_and(|n| n.spot)),
        count(&|s| s.need.as_ref().is_some_and(|n| n.hard_negative > 0.0)),
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

/// How many of the first `n` frames of `items` are on held-out days, overall, by camera and by top
/// class (`frames held-out/all`), and how many the held-out gain raised on each split.
fn heldout_tally(
    items: &[Item],
    n: usize,
    facts: &HashMap<String, Seen>,
    out: &mut String,
    title: &str,
) {
    let first: Vec<&Seen> = items
        .iter()
        .take(n)
        .filter_map(|i| facts.get(&i.frame))
        .collect();
    let held = first.iter().filter(|s| s.heldout).count();
    let gain = |day: bool| {
        first
            .iter()
            .filter(|s| s.heldout == day)
            .filter(|s| s.need.as_ref().is_some_and(|n| n.heldout_gain))
            .count()
    };
    let _ = writeln!(
        out,
        "\n#### {title}: held-out days in the first {n}\n\n{held} of {} on held-out days; \
         raised by the held-out gain: {} on held-out days, {} on training days.\n",
        first.len(),
        gain(true),
        gain(false)
    );
    for (what, by_class) in [("camera", false), ("top class", true)] {
        let mut by: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for s in &first {
            let key = if by_class { &s.class } else { &s.camera };
            let e = by.entry(key.clone()).or_default();
            e.1 += 1;
            if s.heldout {
                e.0 += 1;
            }
        }
        let _ = writeln!(out, "| {what} | held-out / frames |\n|---|---|");
        let mut rows: Vec<_> = by.into_iter().collect();
        rows.sort_by(|a, b| (b.1).1.cmp(&(a.1).1).then_with(|| a.0.cmp(&b.0)));
        for (k, (h, all)) in rows {
            let _ = writeln!(out, "| {k} | {h} / {all} |");
        }
        let _ = writeln!(out);
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
    let spots = Arc::new(crate::collect::labels::statics::Table::build(
        queue::reviewed_obs(
            &dir,
            queue::candidates(&dir, queue::QueueSource::All),
            &crate::collect::labels::bulk::Wanted {
                classes: &wanted.0,
                hidden: &wanted.1,
            },
        ),
    ));
    let spot_ms = t.elapsed();
    let empty = Arc::new(Steer {
        recall: recall.clone(),
        ..Steer::default()
    });
    let real = Arc::new(Steer {
        recall: recall.clone(),
        spots: spots.clone(),
        heldout_listed: None,
    });
    // Held-out steering: the folder's listed days (read only), as the server's steer.
    let listed = match crate::collect::labels::heldout::load(&dir) {
        Ok(f) => f.keys(),
        Err(e) => {
            println!("heldout-days.json does not read ({e:#}): the hash split's days alone");
            Default::default()
        }
    };
    let listed_days = listed.len();
    let held = Arc::new(Steer {
        heldout_listed: Some(Arc::new(listed)),
        ..(*real).clone()
    });
    let with = |s: &Arc<Steer>| Filters {
        steer: Some(s.clone()),
        ..base.clone()
    };
    let (old_ms, old) = median_of_3(|| queue::build_legacy(&dir, &base));
    let (new_empty_ms, new_empty) = median_of_3(|| queue::build(&dir, &with(&empty)));
    let (new_ms, new) = median_of_3(|| queue::build(&dir, &with(&real)));
    let (held_ms, steered) = median_of_3(|| queue::build(&dir, &with(&held)));
    // Whether a frame is on a held-out day, by the steered need's one test.
    let held_need = need::Need::new(
        Some(crate::collect::labels::bulk::Wanted {
            classes: &wanted.0,
            hidden: &wanted.1,
        }),
        Some(&held),
    );
    let on_heldout =
        |frame: &str| held_need.heldout_day(frame.rsplit_once('/').map_or("", |(d, _)| d));
    let mut facts = facts_of(&dir, &new, &with(&real), &spots);
    for (frame, s) in &mut facts {
        s.heldout = on_heldout(frame);
    }
    let mut held_facts = facts_of(&dir, &steered, &with(&held), &spots);
    for (frame, s) in &mut held_facts {
        s.heldout = on_heldout(frame);
    }
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
        "\n## Walk time (`queue::build`, median of 3)\n\n| order | ms |\n|---|---|\n| before (legacy) | {:.0} |\n| need, no spot table | {:.0} |\n| need, spot table | {:.0} |\n| need, spot table, held-out steering | {:.0} |",
        old_ms.as_secs_f64() * 1000.0,
        new_empty_ms.as_secs_f64() * 1000.0,
        new_ms.as_secs_f64() * 1000.0,
        held_ms.as_secs_f64() * 1000.0
    );
    let _ = writeln!(
        out,
        "\nHeld-out days: {listed_days} listed in heldout-days.json, plus the hash split's; {} of \
         {} listed frames on held-out days.",
        held_facts.values().filter(|s| s.heldout).count(),
        held_facts.len()
    );
    assert_eq!(old.items.len(), new.items.len(), "the same frames listed");
    assert_eq!(
        steered.items.len(),
        new.items.len(),
        "the same frames listed"
    );
    for (title, q, facts) in [
        ("Before (legacy)", &old, &facts),
        ("Need, no spot table", &new_empty, &facts),
        ("Need, spot table", &new, &facts),
        ("Need, spot table, held-out steering", &steered, &held_facts),
    ] {
        let _ = writeln!(out, "\n### {title}");
        for n in [100, 500] {
            tally(&q.items, n, facts, &mut out, title);
            heldout_tally(&q.items, n, facts, &mut out, title);
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
        "\n## First 30 served (need, spot table, held-out steering)\n\n| # | frame | rank | random | need | visit | class | cues |\n|---|---|---|---|---|---|---|---|"
    );
    for (i, item) in steered.items.iter().take(30).enumerate() {
        let n = item.uncertainty.need.clone().unwrap_or_default();
        let cues: Vec<&str> = [
            (n.heldout_day, "held-out day"),
            (n.heldout_gain, "held-out gain"),
            (n.below_floor, "floor"),
            (n.catch_up, "catch-up"),
            (n.missed, "missed"),
            (n.gap, "gap"),
            (n.spot, "spot"),
            (n.new_region, "new place"),
            (n.new_day, "new day"),
            (n.expanded, "expanded"),
            (n.model_lacks, "model lacks"),
            (n.hard_negative > 0.0, "hard negative"),
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
    for item in steered.items.iter().take(5) {
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
    // What the review index would hold of the folder's need candidates: every frame of every
    // source, a row of 108 bytes plus 12 bytes a candidate.
    let all = queue::candidates(&dir, queue::QueueSource::All);
    let (mut frames, mut cands) = (0usize, 0usize);
    for c in &all {
        let Some(fr) = Frame::resolve(&dir, &c.rel) else {
            continue;
        };
        let n = match (fr.labels().ok().flatten(), fr.record().ok()) {
            (Some(l), _) => need::of_file(&l, Some(&w)).len(),
            (None, Some(r)) => need::of_reply(&r, Some(&w)).len(),
            (None, None) => continue,
        };
        frames += 1;
        cands += n;
    }
    let per = cands as f64 / frames.max(1) as f64;
    let _ = writeln!(
        out,
        "\n## Index need candidates\n\n{frames} frames, {cands} candidates: {per:.2} a frame, \
         {:.1} bytes a frame in the arena (rows 108 bytes)",
        per * 12.0
    );
    std::fs::create_dir_all(&out_dir).unwrap();
    std::fs::write(out_dir.join("report.md"), &out).unwrap();
    println!("{out}");
    println!("written: {}", out_dir.join("report.md").display());
}
