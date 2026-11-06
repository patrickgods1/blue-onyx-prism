//! The judgement of a comparison table, pure: when one value beats another (the run-vs-run noise
//! rules of [`crate::benchmark::history`], applied between models), which cells of a column are
//! marked best ([`mark_best`]), and the one plain sentence above the table ([`headline`]).

use crate::benchmark::history::compare::{SPEED_ABS_MS, SPEED_REL, proportion_noise};
use serde::Serialize;

/// Under this many objects behind a figure, no sentence calls a winner: the noise floor of one
/// object in ten is already ten points.
pub const MIN_OBJECTS_TO_CALL: usize = 10;

/// `a` beats `b` past the noise of both: each is a share of its own `n` objects, detections or
/// frames, and the larger noise of the two decides. Higher is better. An `n` of 0 never counts.
pub fn clearly_higher(a: (f64, usize), b: (f64, usize)) -> bool {
    if a.1 == 0 || b.1 == 0 {
        return false;
    }
    a.0 - b.0 > proportion_noise(a.0, a.1).max(proportion_noise(b.0, b.1))
}

/// `fast` beats `slow` (ms) past the noise of a speed: `max(SPEED_REL, SPEED_ABS_MS)`, the percent
/// of the slower value, so "5% faster" reads the same from both sides.
pub fn clearly_faster(fast: f64, slow: f64) -> bool {
    slow - fast >= (SPEED_REL * slow).max(SPEED_ABS_MS)
}

/// How a column orders its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    /// A share, higher is better (finds, correct).
    Share,
    /// A share, lower is better (false alerts).
    Fewer,
    /// Milliseconds, lower is better.
    Time,
}

fn beats(column: Column, a: (f64, usize), b: (f64, usize)) -> bool {
    match column {
        Column::Share => clearly_higher(a, b),
        Column::Fewer => clearly_higher(b, a),
        Column::Time => clearly_faster(a.0, b.0),
    }
}

/// Which rows of a column are marked best: the top value and any value not clearly behind it,
/// among the rows that may win (`values[i]` Some). Nothing is marked when fewer than two rows
/// compete or every competitor is inside the noise of the top: nobody is better then.
pub fn mark_best(column: Column, values: &[Option<(f64, usize)>]) -> Vec<bool> {
    let mut marks = vec![false; values.len()];
    let present: Vec<(usize, (f64, usize))> = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|v| (i, v)))
        .filter(|(_, v)| v.0.is_finite())
        .collect();
    if present.len() < 2 {
        return marks;
    }
    let higher = column == Column::Share;
    let top = present
        .iter()
        .map(|p| p.1)
        .reduce(|a, b| if (b.0 > a.0) == higher { b } else { a })
        .unwrap_or((0.0, 0));
    let level: Vec<usize> = present
        .iter()
        .filter(|(_, v)| !beats(column, top, *v))
        .map(|(i, _)| *i)
        .collect();
    if level.len() < present.len() {
        for i in level {
            if let Some(m) = marks.get_mut(i) {
                *m = true;
            }
        }
    }
    marks
}

/// One model's figures behind [`headline`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Contender {
    pub model: String,
    /// Share of the frames it trained on; above 0 it is named but cannot win.
    pub trained_on: Option<f64>,
    /// (recall, objects) of people, vehicles and everything scored.
    pub people: Option<(f64, usize)>,
    pub vehicles: Option<(f64, usize)>,
    pub all: Option<(f64, usize)>,
    /// (precision, detections) over everything scored.
    pub correct: Option<(f64, usize)>,
}

impl Contender {
    fn fair(&self) -> bool {
        self.trained_on.is_none_or(|s| s <= 0.0)
    }
}

/// What the sentence says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Call {
    /// One model finds clearly more.
    Clear,
    /// Every gap is inside the noise.
    Level,
    /// Too few objects to judge.
    TooFew,
    /// Fewer than two models can be judged here.
    Alone,
}

/// The plain sentence above a table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Headline {
    pub call: Call,
    pub text: String,
}

fn pct(v: f64) -> String {
    format!("{:.0}%", v * 100.0)
}

/// "a", "a and b", "a, b and c".
fn names(v: &[&str]) -> String {
    match v {
        [] => String::new(),
        [one] => (*one).to_string(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// The trained-on models, for "left out" wording.
fn left_out(rows: &[Contender], unit: &str) -> String {
    let v: Vec<String> = rows
        .iter()
        .filter(|c| !c.fair())
        .map(|c| {
            format!(
                "{} trained on {} of these {unit}",
                c.model,
                pct(c.trained_on.unwrap_or(0.0))
            )
        })
        .collect();
    v.join("; ")
}

/// One sentence about `rows` on `count` images (`unit`: "frames" or "images"): which model finds
/// more, by people, then vehicles, then everything scored, or why that cannot be said. A model
/// that trained on any of these frames cannot win. Honest about the cost: a winner clearly less
/// precise than a model it beats says so.
pub fn headline(rows: &[Contender], count: usize, unit: &str) -> Headline {
    let fair: Vec<&Contender> = rows.iter().filter(|c| c.fair()).collect();
    let on_these = format!("on these {count} {unit}");
    if fair.len() < 2 {
        let text = match (rows.len(), fair.first()) {
            (0, _) => format!("No model is scored {on_these}."),
            (1, _) => format!(
                "Only {} is scored {on_these}, so there is nothing to compare it with.",
                rows.first().map_or("", |c| c.model.as_str())
            ),
            (_, Some(c)) => format!(
                "Only {} can be judged {on_these}: {}.",
                c.model,
                left_out(rows, unit)
            ),
            (_, None) => format!(
                "No model can be judged {on_these}: {}.",
                left_out(rows, unit)
            ),
        };
        return Headline {
            call: Call::Alone,
            text,
        };
    }
    type Pick = fn(&Contender) -> Option<(f64, usize)>;
    let metrics: [(&str, Pick); 3] = [
        ("people", |c| c.people),
        ("vehicles", |c| c.vehicles),
        ("objects", |c| c.all),
    ];
    let mut level: Vec<&str> = Vec::new();
    let mut judged = false;
    let mut counted: Vec<usize> = Vec::new();
    for (word, pick) in metrics {
        let mut v: Vec<(&Contender, (f64, usize))> = fair
            .iter()
            .filter_map(|c| pick(c).map(|p| (*c, p)))
            .filter(|(_, p)| p.1 >= MIN_OBJECTS_TO_CALL)
            .collect();
        if v.len() < 2 {
            continue;
        }
        // The same objects under another name say nothing new.
        if word == "objects" && !counted.iter().all(|n| v.iter().all(|(_, p)| p.1 != *n)) {
            continue;
        }
        counted.extend(v.iter().map(|(_, p)| p.1));
        judged = true;
        v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let Some((top, tv)) = v.first().copied() else {
            continue;
        };
        let (beaten, even): (Vec<_>, Vec<_>) =
            v.iter().skip(1).partition(|(_, p)| clearly_higher(tv, *p));
        if beaten.is_empty() {
            level.push(word);
            continue;
        }
        let beaten_names: Vec<&str> = beaten.iter().map(|(c, _)| c.model.as_str()).collect();
        let lead = if level.is_empty() {
            String::new()
        } else {
            format!("Level on {}; ", level.join(" and "))
        };
        let mut text = format!(
            "{lead}{} finds more {word} than {} {on_these} ({} of {} {word}",
            top.model,
            names(&beaten_names),
            pct(tv.0),
            tv.1
        );
        if let [(_, p)] = beaten.as_slice() {
            text.push_str(&format!(" against {}", pct(p.0)));
        }
        text.push(')');
        if !even.is_empty() {
            let even_names: Vec<&str> = even.iter().map(|(c, _)| c.model.as_str()).collect();
            text.push_str(&format!(", and is level with {}", names(&even_names)));
        }
        // The price of finding more.
        let dearer = beaten.iter().find_map(|(c, _)| {
            let (mine, theirs) = (top.correct?, c.correct?);
            clearly_higher(theirs, mine).then_some((mine, theirs))
        });
        if let Some((mine, theirs)) = dearer {
            text.push_str(&format!(
                ", but fewer of its detections are right ({} against {})",
                pct(mine.0),
                pct(theirs.0)
            ));
        }
        text.push('.');
        if rows.len() > fair.len() {
            text.push_str(&format!(" Left out: {}.", left_out(rows, unit)));
        }
        return Headline {
            call: Call::Clear,
            text,
        };
    }
    if !judged {
        let most = fair
            .iter()
            .filter_map(|c| c.all.map(|p| p.1))
            .max()
            .unwrap_or(0);
        return Headline {
            call: Call::TooFew,
            text: format!(
                "Too few images to call: {most} objects {on_these}, and a verdict needs at least \
                 {MIN_OBJECTS_TO_CALL}."
            ),
        };
    }
    let mut text = format!(
        "No clear difference in finding {} {on_these}: the gaps are inside the noise.",
        names(&level)
    );
    text.push_str(&more_often_right(&fair));
    Headline {
        call: Call::Level,
        text,
    }
}

/// " a's detections are right more often (90% against 40%).", when one model clearly beats all the
/// others on that: finding the same objects with fewer wrong boxes is a real difference.
fn more_often_right(fair: &[&Contender]) -> String {
    let mut v: Vec<(&Contender, (f64, usize))> =
        fair.iter().filter_map(|c| Some((*c, c.correct?))).collect();
    v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    match v.as_slice() {
        [(top, tv), (_, next), rest @ ..]
            if v.len() == fair.len()
                && clearly_higher(*tv, *next)
                && rest.iter().all(|(_, p)| clearly_higher(*tv, *p)) =>
        {
            format!(
                " {}'s detections are right more often ({} against {}).",
                top.model,
                pct(tv.0),
                pct(next.0)
            )
        }
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_counts_only_past_two_standard_errors() {
        // 80% of 100: noise 2 * sqrt(.8 * .2 / 100) = 8 points.
        assert!(
            !clearly_higher((0.88, 100), (0.80, 100)),
            "exactly the noise"
        );
        assert!(clearly_higher((0.89, 100), (0.80, 100)));
        // One object in ten is the floor: 100% against 90% of 10 is not past it.
        assert!(!clearly_higher((1.0, 10), (0.9, 10)));
        assert!(clearly_higher((1.0, 10), (0.7, 10)));
        // No objects, no verdict.
        assert!(!clearly_higher((1.0, 0), (0.0, 0)));
    }

    #[test]
    fn the_larger_noise_of_the_two_decides() {
        // 99% of 1000 has noise ~1.3 points, 90% of 20 has 13.4.
        assert!(!clearly_higher((0.99, 1000), (0.90, 20)));
    }

    #[test]
    fn speed_counts_at_five_percent_and_half_a_millisecond() {
        assert!(clearly_faster(90.0, 100.0));
        assert!(!clearly_faster(96.0, 100.0), "4% is noise");
        assert!(clearly_faster(95.0, 100.0), "5% counts");
        // Fast models: 10% of 4 ms is under 0.5 ms.
        assert!(!clearly_faster(3.7, 4.0));
        assert!(clearly_faster(3.5, 4.0));
    }

    #[test]
    fn best_is_marked_only_past_the_noise() {
        let s = |v: f64| Some((v, 100));
        // Clear winner.
        assert_eq!(
            mark_best(Column::Share, &[s(0.95), s(0.5), s(0.6)]),
            [true, false, false]
        );
        // The second is inside the noise of the top: both lead, the third is behind.
        assert_eq!(
            mark_best(Column::Share, &[s(0.95), s(0.93), s(0.5)]),
            [true, true, false]
        );
        // Everyone is level: nobody is better.
        assert_eq!(
            mark_best(Column::Share, &[s(0.95), s(0.93)]),
            [false, false]
        );
        // One value: nothing to beat; a missing value is no competitor.
        assert_eq!(mark_best(Column::Share, &[s(0.95), None]), [false, false]);
    }

    #[test]
    fn lower_is_better_for_false_alerts_and_time() {
        let s = |v: f64| Some((v, 100));
        assert_eq!(mark_best(Column::Fewer, &[s(0.05), s(0.5)]), [true, false]);
        let t = |v: f64| Some((v, 1));
        assert_eq!(mark_best(Column::Time, &[t(50.0), t(80.0)]), [true, false]);
        assert_eq!(mark_best(Column::Time, &[t(79.0), t(80.0)]), [false, false]);
    }

    #[test]
    fn a_row_held_out_of_the_race_is_not_marked() {
        // A trained-on row passes None: the best of the rest wins, however good it looks.
        let s = |v: f64| Some((v, 100));
        assert_eq!(
            mark_best(Column::Share, &[None, s(0.5), s(0.8)]),
            [false, false, true]
        );
    }

    fn c(model: &str, people: f64, n: usize, correct: f64) -> Contender {
        Contender {
            model: model.into(),
            trained_on: Some(0.0),
            people: Some((people, n)),
            all: Some((people, n)),
            correct: Some((correct, n)),
            ..Contender::default()
        }
    }

    #[test]
    fn the_sentence_names_the_winner() {
        let h = headline(
            &[c("rfdetr-nano", 0.55, 18, 0.9), c("custom", 1.0, 18, 0.9)],
            23,
            "frames",
        );
        assert_eq!(h.call, Call::Clear);
        assert_eq!(
            h.text,
            "custom finds more people than rfdetr-nano on these 23 frames \
             (100% of 18 people against 55%)."
        );
    }

    #[test]
    fn the_sentence_says_what_finding_more_costs() {
        let h = headline(
            &[c("a", 0.9, 100, 0.3), c("b", 0.5, 100, 0.95)],
            200,
            "images",
        );
        assert!(
            h.text
                .ends_with("but fewer of its detections are right (30% against 95%)."),
            "{}",
            h.text
        );
    }

    #[test]
    fn a_gap_inside_the_noise_is_no_verdict() {
        let h = headline(
            &[c("a", 0.95, 18, 0.9), c("b", 0.90, 18, 0.9)],
            23,
            "frames",
        );
        assert_eq!(h.call, Call::Level);
        assert!(h.text.contains("inside the noise"), "{}", h.text);
        assert!(!h.text.contains("finds more"));
    }

    #[test]
    fn level_on_finding_can_still_differ_on_being_right() {
        // The real 23 held-out frames: both find nearly every person, one with many wrong boxes.
        let h = headline(
            &[c("nano", 0.94, 18, 0.38), c("custom", 1.0, 18, 1.0)],
            23,
            "frames",
        );
        assert_eq!(h.call, Call::Level);
        assert_eq!(
            h.text,
            "No clear difference in finding people on these 23 frames: the gaps are inside the \
             noise. custom's detections are right more often (100% against 38%)."
        );
    }

    #[test]
    fn too_few_objects_is_too_few_to_call() {
        let h = headline(&[c("a", 1.0, 4, 1.0), c("b", 0.0, 4, 1.0)], 6, "frames");
        assert_eq!(h.call, Call::TooFew);
        assert!(h.text.starts_with("Too few images to call"), "{}", h.text);
    }

    #[test]
    fn a_model_on_its_training_frames_cannot_win() {
        let mut trained = c("custom", 1.0, 170, 0.98);
        trained.trained_on = Some(0.59);
        let h = headline(&[trained.clone(), c("nano", 0.7, 170, 0.3)], 251, "frames");
        assert_eq!(h.call, Call::Alone);
        assert_eq!(
            h.text,
            "Only nano can be judged on these 251 frames: custom trained on 59% of these frames."
        );
        // With two others the race goes on, and the sentence says who is left out.
        let h = headline(
            &[trained, c("nano", 0.7, 170, 0.3), c("yolo", 0.2, 170, 0.3)],
            251,
            "frames",
        );
        assert_eq!(h.call, Call::Clear);
        assert!(
            h.text.starts_with("nano finds more people than yolo"),
            "{}",
            h.text
        );
        assert!(
            h.text
                .ends_with("Left out: custom trained on 59% of these frames.")
        );
    }

    #[test]
    fn a_tie_on_people_falls_through_to_vehicles() {
        let mut a = c("a", 0.9, 50, 0.9);
        let mut b = c("b", 0.9, 50, 0.9);
        a.vehicles = Some((0.9, 40));
        b.vehicles = Some((0.3, 40));
        let h = headline(&[a, b], 80, "frames");
        assert_eq!(h.call, Call::Clear);
        assert!(
            h.text.starts_with("Level on people; a finds more vehicles"),
            "{}",
            h.text
        );
    }

    #[test]
    fn three_models_name_who_is_beaten_and_who_is_level() {
        let h = headline(
            &[
                c("a", 0.9, 100, 0.9),
                c("b", 0.88, 100, 0.9),
                c("c", 0.5, 100, 0.9),
            ],
            120,
            "images",
        );
        assert!(
            h.text.starts_with("a finds more people than c on"),
            "{}",
            h.text
        );
        assert!(h.text.contains("and is level with b"), "{}", h.text);
    }

    #[test]
    fn nothing_to_compare_is_said_plainly() {
        assert_eq!(headline(&[], 5, "images").call, Call::Alone);
        let one = headline(&[c("a", 1.0, 50, 1.0)], 5, "images");
        assert_eq!(one.call, Call::Alone);
        assert!(one.text.contains("nothing to compare"), "{}", one.text);
    }
}
