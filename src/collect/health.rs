//! What the Dataset and Home pages and `GET /v1/dataset` say about the collector, so an
//! unwritable folder never looks like a quiet camera: counts since this process started, the last
//! frame and the last error in [`Health`], and the sentences of [`note`] and [`home_line`]. Pure
//! but for [`Counts::now`], which reads [`METRICS`].

use super::Status;
use super::metrics::{Dropped, METRICS};
use serde::Serialize;

/// An error of the collector, with when it happened.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LastError {
    pub message: String,
    /// UTC, RFC 3339.
    pub at: String,
}

/// Frames not saved for one reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DropCount {
    /// `dataset_dropped_total{reason}`.
    pub reason: &'static str,
    /// The reason in plain words.
    pub plain: &'static str,
    pub count: u64,
}

/// The collector's counts since this process started.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Counts {
    /// Detection requests received, collection on or off (no Test page uploads).
    pub requests_seen: u64,
    /// Answered requests handed to the collector while it was on.
    pub frames_offered: u64,
    pub frames_saved: u64,
    /// Frames that matched no enabled bucket.
    pub no_bucket: u64,
    /// Every reason, zeros included.
    pub dropped: Vec<DropCount>,
}

impl Counts {
    pub fn now() -> Self {
        Self {
            requests_seen: METRICS.requests(),
            frames_offered: METRICS.offered(),
            frames_saved: METRICS.saved(),
            no_bucket: METRICS.no_bucket(),
            dropped: Dropped::ALL
                .into_iter()
                .map(|d| DropCount {
                    reason: d.as_str(),
                    plain: d.plain(),
                    count: METRICS.dropped(d),
                })
                .collect(),
        }
    }

    /// The most frequent reason a frame was not saved.
    fn top_drop(&self) -> Option<&DropCount> {
        self.dropped
            .iter()
            .filter(|d| d.count > 0)
            .max_by_key(|d| d.count)
    }
}

fn images(n: u64) -> String {
    if n == 1 {
        "1 image has".to_string()
    } else {
        format!("{n} images have")
    }
}

/// The one sentence that explains an empty dataset, None when there is nothing to explain:
/// collection is off, stopped for disk space (the stop is shown on its own), or saving frames.
pub fn note(
    enabled: bool,
    stopped: bool,
    c: &Counts,
    last_error: Option<&LastError>,
) -> Option<String> {
    if !enabled || stopped || c.frames_saved > 0 {
        return None;
    }
    if c.requests_seen == 0 {
        return Some("Blue Iris has not sent any images since this service started.".to_string());
    }
    if c.frames_offered == 0 {
        return Some(format!(
            "{} come in since this service started, but none has reached the collector since it \
             was turned on. If this stays, check on the Home page that a model is ready.",
            images(c.requests_seen)
        ));
    }
    let why = if let Some(e) = last_error {
        format!("The last error: {}", e.message)
    } else if let Some(d) = c.top_drop() {
        format!("The main reason: {} ({}).", d.plain, d.count)
    } else if c.no_bucket > 0 {
        "None of them matched a bucket that is turned on.".to_string()
    } else {
        "They may still be waiting to be saved.".to_string()
    };
    Some(format!(
        "{} reached the collector since this service started, and no frame was saved. {why}",
        images(c.frames_offered)
    ))
}

/// Home's status line: collecting or not, and the frames so far.
pub fn home_line(enabled: bool, stopped: bool, frames: u64) -> String {
    let state = match (enabled, stopped) {
        (true, true) => "stopped",
        (true, false) => "collecting",
        (false, _) => "not collecting",
    };
    let unit = if frames == 1 { "frame" } else { "frames" };
    format!("{state}, {frames} {unit} so far")
}

/// The newest frame's time: the last one saved by this process, else the newest in the folder.
fn last_frame_at(s: &Status) -> Option<String> {
    s.last_saved_at.clone().or_else(|| {
        s.summary
            .by_camera
            .values()
            .map(|c| c.newest.as_str())
            .max()
            .filter(|t| !t.is_empty())
            .map(str::to_string)
    })
}

/// The `collection` object of `GET /v1/dataset` and the Dataset page's Status card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Health {
    #[serde(flatten)]
    pub counts: Counts,
    /// UTC, RFC 3339.
    pub last_frame_at: Option<String>,
    pub last_error: Option<String>,
    /// UTC, RFC 3339.
    pub last_error_at: Option<String>,
    /// [`note`].
    pub note: Option<String>,
}

impl Health {
    pub fn of(enabled: bool, s: &Status, counts: Counts) -> Self {
        Self {
            note: note(
                enabled,
                s.disabled.is_some(),
                &counts,
                s.last_error.as_ref(),
            ),
            last_frame_at: last_frame_at(s),
            last_error: s.last_error.as_ref().map(|e| e.message.clone()),
            last_error_at: s.last_error.as_ref().map(|e| e.at.clone()),
            counts,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn counts(requests: u64, offered: u64, saved: u64) -> Counts {
        Counts {
            requests_seen: requests,
            frames_offered: offered,
            frames_saved: saved,
            ..Counts::default()
        }
    }

    fn drop_of(d: Dropped, count: u64) -> DropCount {
        DropCount {
            reason: d.as_str(),
            plain: d.plain(),
            count,
        }
    }

    #[test]
    fn no_requests_says_so_in_the_wanted_words() {
        assert_eq!(
            note(true, false, &counts(0, 0, 0), None).unwrap(),
            "Blue Iris has not sent any images since this service started."
        );
    }

    #[test]
    fn off_stopped_or_saving_needs_no_sentence() {
        assert_eq!(note(false, false, &counts(0, 0, 0), None), None);
        assert_eq!(note(true, true, &counts(5, 5, 0), None), None);
        assert_eq!(note(true, false, &counts(5, 5, 1), None), None);
    }

    #[test]
    fn nothing_saved_after_requests_names_the_cause() {
        let e = LastError {
            message: "could not save a frame under /x: Permission denied".into(),
            at: "2026-11-05T10:00:00.000Z".into(),
        };
        let n = note(true, false, &counts(7, 7, 0), Some(&e)).unwrap();
        assert!(n.starts_with("7 images have reached the collector"), "{n}");
        assert!(n.contains("no frame was saved"), "{n}");
        assert!(n.contains("Permission denied"), "{n}");

        let mut c = counts(1, 1, 0);
        c.dropped = vec![drop_of(Dropped::Rate, 1), drop_of(Dropped::Dup, 0)];
        let n = note(true, false, &c, None).unwrap();
        assert!(n.starts_with("1 image has reached"), "{n}");
        assert!(n.contains(Dropped::Rate.plain()), "{n}");

        let mut c = counts(3, 3, 0);
        c.no_bucket = 3;
        assert!(
            note(true, false, &c, None)
                .unwrap()
                .contains("matched a bucket")
        );
    }

    #[test]
    fn requests_that_never_reached_the_collector_are_not_blamed_on_blue_iris() {
        let n = note(true, false, &counts(4, 0, 0), None).unwrap();
        assert!(n.starts_with("4 images have come in"), "{n}");
        assert!(n.contains("model is ready"), "{n}");
    }

    #[test]
    fn the_most_frequent_drop_leads() {
        let mut c = counts(9, 9, 0);
        c.dropped = vec![drop_of(Dropped::Rate, 2), drop_of(Dropped::Error, 7)];
        assert_eq!(c.top_drop().unwrap().reason, "error");
    }

    #[test]
    fn home_line_states() {
        assert_eq!(home_line(true, false, 12), "collecting, 12 frames so far");
        assert_eq!(home_line(false, false, 1), "not collecting, 1 frame so far");
        assert_eq!(home_line(true, true, 0), "stopped, 0 frames so far");
    }

    #[test]
    fn health_prefers_this_process_then_the_folder_for_the_last_frame() {
        let mut s = Status::default();
        assert_eq!(Health::of(true, &s, counts(0, 0, 0)).last_frame_at, None);
        s.summary.by_camera.insert(
            "a".into(),
            super::super::layout::CameraTotals {
                frames: 1,
                bytes: 1,
                newest: "2026-01-01T00:00:00.000Z".into(),
            },
        );
        s.summary.by_camera.insert(
            "b".into(),
            super::super::layout::CameraTotals {
                frames: 1,
                bytes: 1,
                newest: "2026-02-01T00:00:00.000Z".into(),
            },
        );
        assert_eq!(
            Health::of(true, &s, counts(0, 0, 0)).last_frame_at.unwrap(),
            "2026-02-01T00:00:00.000Z"
        );
        s.last_saved_at = Some("2026-03-01T00:00:00.000Z".into());
        s.last_error = Some(LastError {
            message: "m".into(),
            at: "t".into(),
        });
        let h = Health::of(true, &s, counts(1, 1, 1));
        assert_eq!(h.last_frame_at.unwrap(), "2026-03-01T00:00:00.000Z");
        assert_eq!(
            (h.last_error.as_deref(), h.last_error_at.as_deref()),
            (Some("m"), Some("t"))
        );
    }
}
