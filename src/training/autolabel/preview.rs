//! What an auto-label run would do, before it runs (`POST /training/autolabel/preview`,
//! `train autolabel --dry-run`): how many frames match, how many it takes after `limit`, how many
//! the labelers' marker or a model's training leaves out, the split by review status, a rough time
//! and what to know first. The frames come from [`super::AutolabelJob::preview`], the walk of the
//! run itself, so the two cannot disagree. No gate applies (counting frames needs no GPU, download
//! or login) and nothing is written. The texts live here so the pages and the CLI say the same.

use super::job::REOPENED_WARNING;
use super::{AutolabelRequest, GdinoSize, MAX_LIMIT};
use crate::collect::labels::Status;
use crate::training::estimate::format_duration;
use serde::Serialize;

/// SAM 3 per frame before its phrases, measured on [`RATE_HARDWARE`].
pub const SAM3_BASE_SECONDS: f64 = 2.0;
/// SAM 3 per frame per prompt phrase (one `forward` each), measured on [`RATE_HARDWARE`].
pub const SAM3_PHRASE_SECONDS: f64 = 0.2;
/// MM-Grounding-DINO per frame, measured on [`RATE_HARDWARE`].
pub const GDINO_SECONDS: f64 = 0.55;
/// Where the rates come from: every estimate names it, since another GPU differs a lot.
pub const RATE_HARDWARE: &str = "a GTX 1070 Ti";

/// Frames by review status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct StatusSplit {
    pub unreviewed: usize,
    pub reviewed: usize,
    pub skipped: usize,
}

impl StatusSplit {
    pub const fn add(&mut self, s: Status) {
        match s {
            Status::Unreviewed => self.unreviewed += 1,
            Status::Reviewed => self.reviewed += 1,
            Status::Skipped => self.skipped += 1,
        }
    }

    pub const fn total(&self) -> usize {
        self.unreviewed + self.reviewed + self.skipped
    }

    /// `809 reviewed, 12 skipped`: the statuses that have frames.
    fn text(&self) -> String {
        [
            (self.unreviewed, "unreviewed"),
            (self.reviewed, "reviewed"),
            (self.skipped, "skipped"),
        ]
        .iter()
        .filter(|(n, _)| *n > 0)
        .map(|(n, name)| format!("{n} {name}"))
        .collect::<Vec<_>>()
        .join(", ")
    }
}

/// The rough time of a run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Estimate {
    pub seconds_per_frame: f64,
    pub total_seconds: f64,
    /// What the rates were measured on ([`RATE_HARDWARE`]).
    pub hardware: &'static str,
    /// The time leaves out the VLM check, which this run has.
    pub before_vlm: bool,
    /// `about 3.5 h on a GTX 1070 Ti (...)`.
    pub text: String,
}

/// The rough time of `frames` frames of a run of `r` with `phrases` prompt phrases. None when no
/// rate was measured: a run with a model, or RF-DETR L alone.
pub fn estimate(r: &AutolabelRequest, phrases: usize, frames: usize) -> Option<Estimate> {
    if r.with_model.is_some() {
        return None;
    }
    let (seconds_per_frame, rate) = if r.sam3 {
        let s = SAM3_PHRASE_SECONDS.mul_add(phrases as f64, SAM3_BASE_SECONDS);
        (
            s,
            format!(
                "SAM 3 at {SAM3_BASE_SECONDS} s + {SAM3_PHRASE_SECONDS} s x {phrases} phrases = \
                 {s:.1} s per frame"
            ),
        )
    } else {
        let size = r.gdino?;
        let other = if size == GdinoSize::Tiny {
            ""
        } else {
            ", measured with tiny: this size is slower"
        };
        (
            GDINO_SECONDS,
            format!("MM-Grounding-DINO at {GDINO_SECONDS} s per frame{other}"),
        )
    };
    let total_seconds = seconds_per_frame * frames as f64;
    let before_vlm = r.vlm.is_some();
    let vlm = if before_vlm {
        ", before the VLM check (most of a frame's time, about 1 s per box)"
    } else {
        ""
    };
    Some(Estimate {
        seconds_per_frame,
        total_seconds,
        hardware: RATE_HARDWARE,
        before_vlm,
        text: format!(
            "about {} on {RATE_HARDWARE} ({rate}){vlm}",
            format_duration(total_seconds)
        ),
    })
}

/// What the walk over the frames found ([`super::AutolabelJob::preview`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counts {
    pub matched: StatusSplit,
    pub taken: StatusSplit,
    pub skipped_done: usize,
    pub skipped_trained: usize,
    pub unreadable: u64,
}

/// `POST /training/autolabel/preview`, `train autolabel --dry-run`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AutolabelPreview {
    /// The folder under review.
    pub dir: String,
    /// The filter as the run's first stage shows it.
    pub filter: String,
    /// What the frames' marker is compared to (`mm-gdino-tiny + rfdetr-large + ...`).
    pub labeler: String,
    pub limit: usize,
    /// Frames of the filter (a `missing_only` run: those it can look at), whatever `limit` and the
    /// marker do to them.
    pub matched: usize,
    pub matched_by_status: StatusSplit,
    /// Frames the run labels: the first `limit` of those it does not leave out.
    pub taken: usize,
    pub taken_by_status: StatusSplit,
    /// Matching frames left out: these labelers looked at them already (`again=1` to redo).
    pub skipped_done: usize,
    /// `missing_only`: matching frames left out because the model trained on them.
    pub skipped_trained: usize,
    /// Frames to label that `limit` leaves for another run.
    pub over_limit: usize,
    /// Frames whose sidecar or label file does not read.
    pub unreadable: u64,
    /// Distinct prompt phrases of the open-vocabulary labeler.
    pub phrases: usize,
    pub estimate: Option<Estimate>,
    /// The counts in a sentence or two.
    pub summary: String,
    /// What to know before running this, most important first.
    pub warnings: Vec<String>,
    /// The frames [`Self::taken`] counts, in run order.
    #[serde(skip)]
    pub frames: Vec<String>,
}

const fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

impl AutolabelPreview {
    /// The answer for `counts` of a run of `r`; `notes` (a model's training record) end the
    /// warnings.
    pub fn new(
        r: &AutolabelRequest,
        dir: String,
        labeler: String,
        phrases: usize,
        counts: &Counts,
        frames: Vec<String>,
        notes: Vec<String>,
    ) -> Self {
        let matched = counts.matched.total();
        let taken = counts.taken.total();
        let left = matched - counts.skipped_done - counts.skipped_trained;
        let mut p = Self {
            dir,
            filter: r.frames.describe(),
            labeler,
            limit: r.frames.limit,
            matched,
            matched_by_status: counts.matched,
            taken,
            taken_by_status: counts.taken,
            skipped_done: counts.skipped_done,
            skipped_trained: counts.skipped_trained,
            over_limit: left - taken,
            unreadable: counts.unreadable,
            phrases,
            estimate: estimate(r, phrases, taken),
            summary: String::new(),
            warnings: Vec::new(),
            frames,
        };
        p.summary = p.summary_text(r);
        p.warnings = p.warnings_for(r);
        p.warnings.extend(notes);
        p
    }

    fn summary_text(&self, r: &AutolabelRequest) -> String {
        if self.matched == 0 {
            return format!("No frame matches {} in {}.", self.filter, self.dir);
        }
        let mut s = format!(
            "{} frame{} match{} ({}).",
            self.matched,
            plural(self.matched),
            if self.matched == 1 { "es" } else { "" },
            self.matched_by_status.text()
        );
        if self.skipped_done > 0 {
            s.push_str(&format!(
                " {} left out: these labelers already looked at them.",
                self.skipped_done
            ));
        }
        if self.skipped_trained > 0 {
            s.push_str(&format!(
                " {} left out: the model trained on them.",
                self.skipped_trained
            ));
        }
        if self.taken == 0 {
            return s;
        }
        s.push_str(&format!(
            " This run takes {} ({}).",
            self.taken,
            self.taken_by_status.text()
        ));
        match &self.estimate {
            Some(e) => s.push_str(&format!(" Time: {}.", e.text)),
            None if r.with_model.is_some() => {
                s.push_str(" No time estimate: it depends on the model and its device.");
            }
            None => s.push_str(" No time estimate: no rate was measured for these labelers."),
        }
        s
    }

    fn warnings_for(&self, r: &AutolabelRequest) -> Vec<String> {
        let mut w = Vec::new();
        if self.over_limit > 0 {
            let left = self.taken + self.over_limit;
            let mut text = format!(
                "Partial pass: limit {} takes {} of the {left} frames left to label, so {} stay \
                 out of this run. Raise Limit to {} to take {}",
                self.limit,
                self.taken,
                self.over_limit,
                left.min(MAX_LIMIT),
                if left > MAX_LIMIT {
                    "the most one run can"
                } else {
                    "them all"
                }
            );
            if r.frames.again() || r.missing_only {
                text.push_str("; running it again would look at the same frames first.");
            } else {
                text.push_str(", or run it again for the next ones.");
            }
            w.push(text);
        }
        let reviewed = self.taken_by_status.reviewed;
        if reviewed > 0 {
            w.push(format!(
                "{reviewed} reviewed frame{} in this run may reopen: {REOPENED_WARNING}.",
                plural(reviewed)
            ));
        }
        if self.taken == 0 && self.skipped_done > 0 {
            w.push(format!(
                "Nothing to label: {} looked at {} frame{} for these classes already. Add again=1 \
                 to the filter, or tick the box for frames already labelled.",
                self.labeler,
                self.skipped_done,
                plural(self.skipped_done)
            ));
        }
        if self.taken == 0 && self.skipped_trained > 0 {
            w.push(format!(
                "Nothing to check: the model trained on {} frame{}.",
                self.skipped_trained,
                plural(self.skipped_trained)
            ));
        }
        if self.unreadable > 0 {
            w.push(format!(
                "{} frame{} could not be read and {} left out.",
                self.unreadable,
                if self.unreadable == 1 { "" } else { "s" },
                if self.unreadable == 1 { "is" } else { "are" }
            ));
        }
        w
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::{FrameFilter, VlmSize};
    use super::*;

    fn request() -> AutolabelRequest {
        AutolabelRequest::new(FrameFilter::default())
    }

    fn sam3() -> AutolabelRequest {
        let mut r = request();
        r.sam3 = true;
        r.gdino = None;
        r
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn preview(r: &AutolabelRequest, c: &Counts) -> AutolabelPreview {
        AutolabelPreview::new(r, "d".into(), "l".into(), 9, c, vec![], vec![])
    }

    fn unreviewed(n: usize) -> StatusSplit {
        StatusSplit {
            unreviewed: n,
            ..StatusSplit::default()
        }
    }

    fn reviewed(n: usize) -> StatusSplit {
        StatusSplit {
            reviewed: n,
            ..StatusSplit::default()
        }
    }

    #[test]
    fn the_estimate_uses_the_measured_rates_and_names_the_hardware() {
        let e = estimate(&sam3(), 20, 100).unwrap();
        assert!(near(e.seconds_per_frame, 6.0), "2 s + 0.2 s x 20: {e:?}");
        assert!(near(e.total_seconds, 600.0), "{e:?}");
        assert!(e.text.contains("a GTX 1070 Ti"), "{}", e.text);
        assert!(e.text.contains("20 phrases"), "{}", e.text);
        assert!(e.text.contains("before the VLM check"), "{}", e.text);
        assert!(e.before_vlm);

        let mut r = request();
        r.vlm = None;
        let e = estimate(&r, 20, 1000).unwrap();
        assert!(near(e.seconds_per_frame, 0.55), "phrases do not count");
        assert!(near(e.total_seconds, 550.0), "{e:?}");
        assert!(!e.before_vlm && !e.text.contains("VLM"), "{}", e.text);
        assert!(!e.text.contains("slower"), "{}", e.text);

        r.vlm = Some(VlmSize::Qwen2b);
        r.gdino = Some(GdinoSize::Large);
        let e = estimate(&r, 20, 1).unwrap();
        assert!(e.text.contains("measured with tiny"), "{}", e.text);
    }

    #[test]
    fn no_measured_rate_means_no_estimate() {
        let mut r = request();
        r.gdino = None;
        assert_eq!(estimate(&r, 20, 10), None, "RF-DETR L alone");
        let mut m = request();
        m.with_model = Some("mine".into());
        assert_eq!(estimate(&m, 20, 10), None);
        let c = Counts {
            matched: unreviewed(1),
            taken: unreviewed(1),
            ..Counts::default()
        };
        let p = preview(&m, &c);
        assert!(p.summary.contains("No time estimate"), "{}", p.summary);
        assert!(p.summary.starts_with("1 frame matches"), "{}", p.summary);
    }

    #[test]
    fn a_limit_below_the_matches_is_called_a_partial_pass() {
        let c = Counts {
            matched: reviewed(809),
            taken: reviewed(200),
            ..Counts::default()
        };
        let r = request();
        let p = preview(&r, &c);
        assert_eq!((p.matched, p.taken, p.over_limit), (809, 200, 609));
        assert_eq!(p.limit, 200, "the limit is reported, never changed");
        assert!(
            p.warnings[0].starts_with("Partial pass: limit 200"),
            "{p:?}"
        );
        assert!(p.warnings[0].contains("Raise Limit to 809"), "{p:?}");
        assert!(p.warnings[0].contains("next ones"), "{p:?}");
        assert!(p.warnings[1].contains("200 reviewed frames"), "{p:?}");
        assert!(p.warnings[1].contains("leave exports"), "{p:?}");

        // With again=1 a rerun starts from the same frames.
        let again = AutolabelRequest::new(FrameFilter::parse("again=1", "any").unwrap());
        let p = preview(&again, &c);
        assert!(p.warnings[0].contains("same frames first"), "{p:?}");

        // Nothing over the limit: no warning about it.
        let c = Counts {
            matched: reviewed(150),
            taken: reviewed(100),
            skipped_done: 50,
            ..Counts::default()
        };
        let p = preview(&r, &c);
        assert_eq!(p.over_limit, 0);
        assert!(!p.warnings.iter().any(|w| w.contains("Partial")), "{p:?}");
    }

    #[test]
    fn more_frames_than_one_run_can_take_are_said_so() {
        let c = Counts {
            matched: reviewed(MAX_LIMIT + 5),
            taken: reviewed(MAX_LIMIT),
            ..Counts::default()
        };
        let p = preview(&request(), &c);
        assert!(p.warnings[0].contains("Raise Limit to 20000"), "{p:?}");
        assert!(p.warnings[0].contains("the most one run can"), "{p:?}");
    }

    #[test]
    fn unreviewed_frames_carry_no_reopen_warning_and_notes_stay() {
        let c = Counts {
            matched: unreviewed(5),
            taken: unreviewed(5),
            ..Counts::default()
        };
        let note = vec!["no training record".to_string()];
        let p = AutolabelPreview::new(&request(), "d".into(), "l".into(), 9, &c, vec![], note);
        assert_eq!(p.warnings, ["no training record"]);
    }

    #[test]
    fn nothing_to_do_says_why() {
        let r = request();
        let p = AutolabelPreview::new(
            &r,
            "D:/data".into(),
            "l".into(),
            9,
            &Counts::default(),
            vec![],
            vec![],
        );
        assert!(p.summary.starts_with("No frame matches"), "{}", p.summary);
        assert!(p.summary.contains("D:/data"), "{}", p.summary);

        let c = Counts {
            matched: reviewed(3),
            skipped_done: 3,
            unreadable: 2,
            ..Counts::default()
        };
        let p = preview(&r, &c);
        assert_eq!((p.taken, p.over_limit), (0, 0));
        assert!(!p.summary.contains("Time:"), "{}", p.summary);
        assert!(p.warnings[0].contains("again=1"), "{p:?}");
        assert!(
            p.warnings[1].contains("2 frames could not be read"),
            "{p:?}"
        );

        let c = Counts {
            matched: reviewed(1),
            skipped_trained: 1,
            ..Counts::default()
        };
        let p = preview(&r, &c);
        assert!(p.warnings[0].contains("trained on 1 frame."), "{p:?}");
    }
}
