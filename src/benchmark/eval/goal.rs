//! The benchmark goal (`benchmark.goal` in the config): what Compare ranks by, what Tune picks
//! the confidence for, which classes count and how slow a combo may be.
//!
//! - `optimize_for` ([`Optimize`]): a threshold objective ([`Objective`]: `f1` (default), `f2`,
//!   `f05`, `recall_at_fa:<N>`, `precision:<p>`, `recall:<r>`, `youden`, `fpr:<x>`), scored at
//!   each combo's cross-validated confidence pick, or a rank-only metric that needs no
//!   threshold: `ap50`, `ap50_95`, `roc_auc` (one sample per frame) or `grade` (the weighted A-F
//!   overall grade of [`crate::benchmark::grade`]).
//! - `tune_for`: the threshold objective the confidence is picked for when `optimize_for` is
//!   rank-only (default `f1`); ignored otherwise ([`Goal::tuning`]).
//! - `classes`: the counted classes; empty = every scored class the set labels.
//! - `speed_budget_ms`: none (default) or the p95 latency a combo may take (burst p95 when a
//!   burst ran, else one-at-a-time p95); slower combos rank after the others as "too slow".
//!
//! Older config files have `threshold_objective` / `accuracy_metric` only: [`Goal::from_legacy`]
//! turns them into a goal (`accuracy_metric: roc_auc` -> `optimize_for: roc_auc` tuned for the
//! objective, else `optimize_for` = the objective). The config keeps writing both legacy keys
//! beside `goal` ([`Goal::legacy`]) so an older binary reading the file keeps working; `goal`
//! wins when both are present.

use crate::benchmark::grade::AccuracyMetric;
use crate::benchmark::threshold::Objective;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// What a comparison optimizes ([`Goal::optimize_for`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Optimize {
    /// A threshold objective, scored at the cross-validated pick.
    Threshold(Objective),
    /// AP@0.5 over the counted classes (the [`crate::benchmark::metrics::MIN_CLASS_GT`] rule).
    Ap50,
    /// AP@0.5:0.95 over the counted classes.
    Ap5095,
    /// ROC AUC over one sample per frame (positive with a counted object; score = the frame's
    /// highest counted-class confidence).
    RocAuc,
    /// The weighted A-F overall grade (accuracy score from AP50 and small-object recall, speed
    /// from the p50; `benchmark.weights`).
    Grade,
}

impl Default for Optimize {
    fn default() -> Self {
        Optimize::Threshold(Objective::F1)
    }
}

impl Optimize {
    /// The threshold objective, None for a rank-only goal.
    pub fn objective(&self) -> Option<Objective> {
        match self {
            Optimize::Threshold(o) => Some(*o),
            _ => None,
        }
    }

    /// Ranks without a threshold.
    pub fn rank_only(&self) -> bool {
        self.objective().is_none()
    }

    /// Short column name: "F1", "F2", "Recall @ P≥90%", "AP50", ...
    pub fn short(&self) -> String {
        match self {
            Optimize::Threshold(o) => objective_short(*o),
            Optimize::Ap50 => "AP50".into(),
            Optimize::Ap5095 => "AP50-95".into(),
            Optimize::RocAuc => "ROC AUC".into(),
            Optimize::Grade => "Grade".into(),
        }
    }

    /// Human-readable, as [`Objective::describe`].
    pub fn describe(&self) -> String {
        match self {
            Optimize::Threshold(o) => o.describe(),
            Optimize::Ap50 => "AP50 (box level, no threshold)".into(),
            Optimize::Ap5095 => "AP50-95 (box level, no threshold)".into(),
            Optimize::RocAuc => "ROC AUC (one sample per frame, no threshold)".into(),
            Optimize::Grade => "overall A-F grade (accuracy and speed)".into(),
        }
    }
}

/// The column name of a threshold objective's maximized value.
pub fn objective_short(o: Objective) -> String {
    let pct = |v: f64| {
        let s = format!("{:.1}", v * 100.0);
        s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
    };
    let num = |v: f64| {
        let s = format!("{v:.1}");
        s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
    };
    match o {
        Objective::F1 => "F1".into(),
        Objective::F2 => "F2".into(),
        Objective::F05 => "F0.5".into(),
        Objective::Precision(p) => format!("Recall @ P\u{2265}{}%", pct(p)),
        Objective::Recall(r) => format!("Precision @ R\u{2265}{}%", pct(r)),
        Objective::Youden => "Youden's J".into(),
        Objective::Fpr(x) => format!("Frame TPR @ FPR\u{2264}{}%", pct(x)),
        Objective::RecallAtFa(n) => format!("Recall @ \u{2264}{} FA/100", num(n)),
    }
}

impl fmt::Display for Optimize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Optimize::Threshold(o) => write!(f, "{o}"),
            Optimize::Ap50 => f.write_str("ap50"),
            Optimize::Ap5095 => f.write_str("ap50_95"),
            Optimize::RocAuc => f.write_str("roc_auc"),
            Optimize::Grade => f.write_str("grade"),
        }
    }
}

impl FromStr for Optimize {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim().to_ascii_lowercase().replace('-', "_");
        match t.as_str() {
            "ap50" | "ap" | "map50" => Ok(Optimize::Ap50),
            "ap50_95" | "ap5095" | "map" | "map50_95" => Ok(Optimize::Ap5095),
            "roc_auc" | "auc" | "rocauc" | "roc" => Ok(Optimize::RocAuc),
            "grade" => Ok(Optimize::Grade),
            _ => s
                .parse::<Objective>()
                .map(Optimize::Threshold)
                .map_err(|e| {
                    format!(
                        "{e}; or a rank-only goal: ap50, ap50_95, roc_auc or grade (optimize_for \
                     '{s}')"
                    )
                }),
        }
    }
}

impl TryFrom<String> for Optimize {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        s.parse()
    }
}

impl From<Optimize> for String {
    fn from(o: Optimize) -> String {
        o.to_string()
    }
}

/// The benchmark goal (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Goal {
    pub optimize_for: Optimize,
    /// The threshold objective of a rank-only `optimize_for` (default `f1`).
    pub tune_for: Objective,
    /// Counted classes; empty = every scored class the set labels.
    pub classes: Vec<String>,
    /// p95 latency budget in ms; None = no budget.
    pub speed_budget_ms: Option<f64>,
}

impl Goal {
    /// The objective the confidence is picked for: `optimize_for` when it is one, else
    /// `tune_for`.
    pub fn tuning(&self) -> Objective {
        self.optimize_for.objective().unwrap_or(self.tune_for)
    }

    /// The goal of a config without `goal` (see the module docs).
    pub fn from_legacy(objective: Objective, metric: AccuracyMetric) -> Self {
        match metric {
            AccuracyMetric::RocAuc => Self {
                optimize_for: Optimize::RocAuc,
                tune_for: objective,
                ..Self::default()
            },
            AccuracyMetric::Ap50 => Self {
                optimize_for: Optimize::Threshold(objective),
                ..Self::default()
            },
        }
    }

    /// `(threshold_objective, accuracy_metric)` written beside `goal` for older binaries: the
    /// tuning objective, and `roc_auc` for a ROC AUC goal (else `ap50`, what an older binary
    /// grades with).
    pub fn legacy(&self) -> (Objective, AccuracyMetric) {
        let metric = if self.optimize_for == Optimize::RocAuc {
            AccuracyMetric::RocAuc
        } else {
            AccuracyMetric::Ap50
        };
        (self.tuning(), metric)
    }

    /// The goal with what an older form or flag sets (`--threshold-objective`,
    /// `--accuracy-metric`, the page's objective and metric selects), keeping the classes and
    /// the budget: `roc_auc` makes the goal ROC AUC; `ap50` turns a ROC AUC goal back into its
    /// threshold objective; an objective becomes `optimize_for`, or `tune_for` of a rank-only
    /// goal.
    #[must_use]
    pub fn with_legacy(
        &self,
        objective: Option<Objective>,
        metric: Option<AccuracyMetric>,
    ) -> Self {
        let mut g = self.clone();
        match metric {
            Some(AccuracyMetric::RocAuc) => {
                g.tune_for = objective.unwrap_or_else(|| self.tuning());
                g.optimize_for = Optimize::RocAuc;
                return g;
            }
            Some(AccuracyMetric::Ap50) if g.optimize_for == Optimize::RocAuc => {
                g.optimize_for = Optimize::Threshold(objective.unwrap_or(g.tune_for));
                return g;
            }
            _ => {}
        }
        if let Some(o) = objective {
            if g.optimize_for.rank_only() {
                g.tune_for = o;
            } else {
                g.optimize_for = Optimize::Threshold(o);
            }
        }
        g
    }

    /// Checks what serde cannot: a budget is a positive, finite number of ms.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(b) = self.speed_budget_ms
            && !(b.is_finite() && b > 0.0)
        {
            return Err(format!(
                "speed_budget_ms: {b} is not a positive number of milliseconds (null = no budget)"
            ));
        }
        if self.classes.iter().any(|c| c.trim().is_empty()) {
            return Err("classes: an empty class name".into());
        }
        Ok(())
    }

    /// The goal with what a request tries (`goal=`, `tune_for=`, `classes=`, `budget=`; empty
    /// or absent = the saved value; `classes=all` and `budget=none` clear).
    pub fn try_with(
        &self,
        optimize_for: Option<&str>,
        tune_for: Option<&str>,
        classes: Option<&str>,
        budget: Option<&str>,
    ) -> Result<Self, String> {
        let mut g = self.clone();
        if let Some(o) = optimize_for.map(str::trim).filter(|s| !s.is_empty()) {
            g.optimize_for = o.parse()?;
        }
        if let Some(t) = tune_for.map(str::trim).filter(|s| !s.is_empty()) {
            g.tune_for = t.parse().map_err(|e: String| format!("tune_for: {e}"))?;
        }
        if let Some(c) = classes.map(str::trim).filter(|s| !s.is_empty()) {
            g.classes = parse_classes(c);
        }
        if let Some(b) = budget.map(str::trim).filter(|s| !s.is_empty()) {
            g.speed_budget_ms = parse_budget(b)?;
        }
        g.validate()?;
        Ok(g)
    }

    /// "F2 · person, car · budget 1 s".
    pub fn describe(&self) -> String {
        let mut parts = vec![self.optimize_for.short()];
        if self.optimize_for.rank_only() {
            parts.push(format!("tuned for {}", objective_short(self.tune_for)));
        }
        if !self.classes.is_empty() {
            parts.push(self.classes.join(", "));
        }
        if let Some(b) = self.speed_budget_ms {
            parts.push(format!("budget {}", ms_text(b)));
        }
        parts.join(" \u{b7} ")
    }

    /// A stable text of every field (cache keys).
    pub fn key(&self) -> String {
        format!(
            "{}|{}|{}|{:?}",
            self.optimize_for,
            self.tune_for,
            self.classes.join(","),
            self.speed_budget_ms
        )
    }
}

/// `person, car` / `person,car` -> the trimmed, non-empty, deduplicated names; `all` = none.
pub fn parse_classes(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in s.split(',').map(str::trim).filter(|c| !c.is_empty()) {
        if c.eq_ignore_ascii_case("all") {
            return Vec::new();
        }
        let c = c.to_ascii_lowercase();
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// `250` / `250ms` / `1.5s` -> ms; `none` / `off` / `0` -> no budget.
pub fn parse_budget(s: &str) -> Result<Option<f64>, String> {
    let t = s.trim().to_ascii_lowercase();
    if matches!(t.as_str(), "none" | "off" | "null" | "0" | "") {
        return Ok(None);
    }
    let (num, scale) = if let Some(v) = t.strip_suffix("ms") {
        (v, 1.0)
    } else if let Some(v) = t.strip_suffix('s') {
        (v, 1000.0)
    } else {
        (t.as_str(), 1.0)
    };
    let v: f64 = num
        .trim()
        .parse()
        .map_err(|_| format!("speed budget '{s}': not a number of ms (e.g. 250 or 1.5s)"))?;
    let ms = v * scale;
    if !(ms.is_finite() && ms > 0.0) {
        return Err(format!("speed budget '{s}': must be above 0 ms"));
    }
    Ok(Some(ms))
}

/// "250 ms", "1.5 s".
pub fn ms_text(ms: f64) -> String {
    if ms >= 1000.0 {
        let s = format!("{:.1}", ms / 1000.0);
        format!("{} s", s.strip_suffix(".0").unwrap_or(&s))
    } else {
        format!("{ms:.0} ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimize_for_parses_every_goal_and_round_trips() {
        for (text, want) in [
            ("f1", Optimize::Threshold(Objective::F1)),
            ("f2", Optimize::Threshold(Objective::F2)),
            ("f05", Optimize::Threshold(Objective::F05)),
            (
                "recall_at_fa:5",
                Optimize::Threshold(Objective::RecallAtFa(5.0)),
            ),
            (
                "precision:0.9",
                Optimize::Threshold(Objective::Precision(0.9)),
            ),
            ("recall:0.8", Optimize::Threshold(Objective::Recall(0.8))),
            ("youden", Optimize::Threshold(Objective::Youden)),
            ("fpr:0.05", Optimize::Threshold(Objective::Fpr(0.05))),
            ("ap50", Optimize::Ap50),
            ("ap50_95", Optimize::Ap5095),
            ("AP50-95", Optimize::Ap5095),
            ("roc_auc", Optimize::RocAuc),
            ("grade", Optimize::Grade),
        ] {
            let o: Optimize = text.parse().unwrap();
            assert_eq!(o, want, "{text}");
            assert_eq!(o.to_string().parse::<Optimize>().unwrap(), o);
        }
        let e = "f3".parse::<Optimize>().unwrap_err();
        assert!(e.contains("ap50_95"), "{e}");
        assert!(Optimize::Ap50.rank_only() && !Optimize::default().rank_only());
    }

    #[test]
    fn legacy_keys_map_onto_the_goal_both_ways() {
        let g = Goal::from_legacy(Objective::Youden, AccuracyMetric::RocAuc);
        assert_eq!(g.optimize_for, Optimize::RocAuc);
        assert_eq!(g.tune_for, Objective::Youden);
        assert_eq!(g.tuning(), Objective::Youden);
        assert_eq!(g.legacy(), (Objective::Youden, AccuracyMetric::RocAuc));
        let g = Goal::from_legacy(Objective::Precision(0.9), AccuracyMetric::Ap50);
        assert_eq!(
            g.optimize_for,
            Optimize::Threshold(Objective::Precision(0.9))
        );
        assert_eq!(
            g.legacy(),
            (Objective::Precision(0.9), AccuracyMetric::Ap50)
        );
        // A goal older binaries cannot express writes its tuning objective and ap50.
        let g = Goal {
            optimize_for: Optimize::Ap5095,
            tune_for: Objective::F2,
            ..Goal::default()
        };
        assert_eq!(g.legacy(), (Objective::F2, AccuracyMetric::Ap50));
        // An old flag keeps the classes and the budget.
        let base = Goal {
            classes: vec!["person".into()],
            speed_budget_ms: Some(500.0),
            ..Goal::default()
        };
        let g = base.with_legacy(Some(Objective::F2), Some(AccuracyMetric::Ap50));
        assert_eq!(g.optimize_for, Optimize::Threshold(Objective::F2));
        assert_eq!(g.classes, ["person"]);
        assert_eq!(g.speed_budget_ms, Some(500.0));
        // roc_auc and back.
        let g = g.with_legacy(None, Some(AccuracyMetric::RocAuc));
        assert_eq!(
            (g.optimize_for, g.tune_for),
            (Optimize::RocAuc, Objective::F2)
        );
        let g = g.with_legacy(None, Some(AccuracyMetric::Ap50));
        assert_eq!(g.optimize_for, Optimize::Threshold(Objective::F2));
        // On a rank-only goal an objective is what it is tuned for.
        let g = Goal {
            optimize_for: Optimize::Ap5095,
            ..base
        }
        .with_legacy(Some(Objective::Youden), Some(AccuracyMetric::Ap50));
        assert_eq!(
            (g.optimize_for, g.tune_for),
            (Optimize::Ap5095, Objective::Youden)
        );
    }

    #[test]
    fn goal_json_and_tries() {
        let g: Goal = serde_json::from_str(
            r#"{"optimize_for":"ap50","tune_for":"f2","classes":["person"],"speed_budget_ms":250}"#,
        )
        .unwrap();
        assert_eq!(g.optimize_for, Optimize::Ap50);
        assert_eq!(g.tuning(), Objective::F2);
        assert_eq!(
            g.describe(),
            "AP50 \u{b7} tuned for F2 \u{b7} person \u{b7} budget 250 ms"
        );
        let back: Goal = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back, g);
        let d: Goal = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Goal::default());
        assert!(serde_json::from_str::<Goal>(r#"{"optimize_for":"f3"}"#).is_err());
        let t = g
            .try_with(Some("f2"), None, Some("Person, car,person"), Some("1.5s"))
            .unwrap();
        assert_eq!(t.optimize_for, Optimize::Threshold(Objective::F2));
        assert_eq!(t.classes, ["person", "car"]);
        assert_eq!(t.speed_budget_ms, Some(1500.0));
        let t = g.try_with(None, None, Some("all"), Some("none")).unwrap();
        assert!(t.classes.is_empty() && t.speed_budget_ms.is_none());
        assert!(g.try_with(None, None, None, Some("-3")).is_err());
        assert!(g.try_with(None, Some("ap50"), None, None).is_err());
        assert!(
            Goal {
                speed_budget_ms: Some(f64::NAN),
                ..Goal::default()
            }
            .validate()
            .is_err()
        );
        assert_eq!(ms_text(1000.0), "1 s");
        assert_eq!(ms_text(420.4), "420 ms");
    }
}
