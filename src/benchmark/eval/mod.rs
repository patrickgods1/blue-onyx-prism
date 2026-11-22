//! The evaluation core: one goal, cross-validated confidence picks, intervals, paired
//! comparisons, rank groups and the speed budget. Pure: it reads the saved results and the
//! stored predictions handed in and computes; the server ([`crate::server`]'s
//! `/v1/benchmark/compare`, `/tune`, `/tune/images`, `/goal`) and the CLI render it.
//!
//! - [`goal`]: `benchmark.goal` (`optimize_for`, `tune_for`, `classes`, `speed_budget_ms`) and
//!   its migration from `threshold_objective` / `accuracy_metric`;
//! - [`tally`]: per-image counts on the 0.01 grid (the definitions of every count);
//! - [`pick`]: the grid pick with targets judged on their 95% Wilson bound;
//! - [`cv`]: cross-validation over camera days (or single images): folds, out-of-fold counts;
//! - [`stats`]: Wilson intervals, the seeded bootstrap, percentile intervals, paired
//!   differences;
//! - [`rankfree`]: AP50, AP50-95 and frame ROC AUC with group weights;
//! - [`assess`]: one combo on one set judged for the goal;
//! - [`rank`]: the order and rank groups;
//! - [`compare`]: the Compare view (rows, headline, breakdowns);
//! - [`tune`]: the Tune view of one combo (grid rows, pick, saved value) and its example frames.

pub mod assess;
pub mod compare;
pub mod cv;
pub mod goal;
pub mod pick;
pub mod rank;
pub mod rankfree;
pub mod stats;
pub mod tally;
pub mod tune;

#[cfg(test)]
mod simulation;
#[cfg(test)]
pub(crate) mod tests;

pub use goal::{Goal, Optimize};
