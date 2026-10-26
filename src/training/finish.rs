//! The steps after the last task of an import chain ([`Finish`]): run by an auto-label job
//! that follows an import, or by the import itself when nothing follows it, only when it ended
//! Done. In order: reject the pending auto boxes scoring below a minimum
//! ([`bulk::bulk_reject_below`]), accept the frames the triage then settles
//! ([`bulk::bulk_accept`]), point the Review page at the import's folder
//! ([`crate::collect::config::set_review_dir`]). Reject and accept are scoped to the import's
//! frames (sidecar `source: import:<id>`). Each step reports one line; a failing step is a
//! `warning: ...` line, never a failed task.

use super::autolabel::FrameFilter;
use super::autolabel::models::ConfigWriter;
use super::task::TaskOutcome;
use crate::collect::labels::Frame;
use crate::collect::labels::bulk::{self, BulkOutcome};
use crate::collect::labels::queue::{Filters, QueueSource};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Most frames one step changes.
const STEP_LIMIT: usize = super::autolabel::MAX_LIMIT;

/// The steps after a chain (see the module docs).
#[derive(Clone, Default)]
pub struct Finish {
    /// Reject pending auto boxes scoring below this (0.01..=1) on the unreviewed frames.
    pub reject_below: Option<f32>,
    /// Save the frames whose boxes are all `auto: accept` as reviewed.
    pub accept_auto: bool,
    /// Point the Review page (`collect.review_dir`) at this folder.
    pub review_folder: Option<PathBuf>,
    /// Saves the folder switch (None: the switch is a warning).
    pub config_writer: Option<Arc<dyn ConfigWriter>>,
}

impl std::fmt::Debug for Finish {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Finish")
            .field("reject_below", &self.reject_below)
            .field("accept_auto", &self.accept_auto)
            .field("review_folder", &self.review_folder)
            .field("config_writer", &self.config_writer.is_some())
            .finish()
    }
}

impl Finish {
    /// No step.
    pub fn is_empty(&self) -> bool {
        self.reject_below.is_none() && !self.accept_auto && self.review_folder.is_none()
    }

    /// Run the steps on `dataset_dir` (the frames of import `import_id` only, when given;
    /// `classes` = `collect.classes`, the accept's class check); returns one line per step.
    pub fn apply(
        &self,
        dataset_dir: &Path,
        import_id: Option<&str>,
        classes: &[String],
    ) -> Vec<String> {
        let mut lines = Vec::new();
        let scope = import_id.map(|id| FrameFilter {
            query: [("import".to_string(), id.to_string())].into(),
            limit: STEP_LIMIT,
        });
        let keep = |rel: &str| match &scope {
            None => true,
            Some(f) => Frame::resolve(dataset_dir, rel)
                .and_then(|fr| fr.record().ok())
                .is_some_and(|r| f.accepts_source(r.source.as_deref())),
        };
        let filters = Filters {
            source: QueueSource::All,
            ..Filters::default()
        };
        let of = import_id.map_or_else(String::new, |id| format!(" of import {id}"));
        if let Some(min) = self.reject_below {
            let o = bulk::bulk_reject_below(dataset_dir, &filters, min, STEP_LIMIT, false, &keep);
            lines.push(format!(
                "rejected {} auto box(es) scoring below {min} on {} frame(s){of}",
                o.boxes, o.done_count
            ));
            lines.extend(failures("rejecting low-scoring boxes", &o));
        }
        if self.accept_auto {
            let now = crate::collect::utc_rfc3339(std::time::SystemTime::now());
            let o = bulk::bulk_accept(
                dataset_dir,
                &filters,
                classes,
                STEP_LIMIT,
                &now,
                false,
                &keep,
            );
            lines.push(format!(
                "accepted {} frame(s){of} whose boxes were all auto: accept ({} boxes; saved as \
                 reviewed)",
                o.done_count, o.boxes
            ));
            lines.extend(failures("accepting frames", &o));
        }
        if let Some(dir) = &self.review_folder {
            lines.push(self.switch_folder(dir));
        }
        for l in &lines {
            tracing::info!("after the chain: {l}");
        }
        lines
    }

    /// The folder step's line.
    fn switch_folder(&self, dir: &Path) -> String {
        let Some(w) = &self.config_writer else {
            return format!(
                "warning: the Review page was not switched to {} (no config to save); choose the \
                 folder on the Review page (Folder under review)",
                dir.display()
            );
        };
        let r =
            w.update(&mut |c| crate::collect::config::set_review_dir(c, Some(dir.to_path_buf())));
        match r {
            Ok(_) => format!("the Review page now shows {}", dir.display()),
            Err(e) => format!(
                "warning: the Review page was not switched to {}: {e:#}; choose the folder on \
                 the Review page (Folder under review)",
                dir.display()
            ),
        }
    }
}

/// Add the steps' `lines` to a task's outcome: the summary gets `; then: <lines>`, the result
/// a `finish` list, and its `warnings` the warning lines (without the `warning: ` prefix).
pub fn append(out: &mut TaskOutcome, lines: Vec<String>) {
    if lines.is_empty() {
        return;
    }
    out.summary.push_str("; then: ");
    out.summary.push_str(&lines.join("; "));
    if !out.result.is_object() {
        out.result = serde_json::json!({});
    }
    if let Some(o) = out.result.as_object_mut() {
        let warnings = o.entry("warnings").or_insert_with(|| serde_json::json!([]));
        if let Some(w) = warnings.as_array_mut() {
            w.extend(
                lines
                    .iter()
                    .filter_map(|l| l.strip_prefix("warning: "))
                    .map(|l| serde_json::Value::String(l.to_string())),
            );
        }
        o.insert("finish".into(), serde_json::json!(lines));
    }
}

/// The warning line of a step's failed frames (none when all went through).
fn failures(step: &str, o: &BulkOutcome) -> Option<String> {
    (o.failed_count > 0).then(|| {
        format!(
            "warning: {step}: {} frame(s) not changed (review them by hand): {}",
            o.failed_count,
            o.failed.join("; ")
        )
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::collect::labels::Triage;
    use crate::collect::labels::bulk::test_support::{auto_box, labels_of, put_frame};
    use crate::config::Config;
    use std::sync::Mutex;

    /// Records the folder it saved; fails when `fail`.
    struct Writer {
        saved: Mutex<Option<PathBuf>>,
        fail: bool,
    }

    impl ConfigWriter for Writer {
        fn update(
            &self,
            edit: &mut dyn FnMut(&mut Config) -> anyhow::Result<()>,
        ) -> anyhow::Result<(Config, String)> {
            if self.fail {
                anyhow::bail!("the config file is read-only");
            }
            let mut c = Config::default();
            edit(&mut c)?;
            *self.saved.lock().unwrap() = c.collect.review_dir.clone();
            Ok((c, "saved".into()))
        }
    }

    #[test]
    fn steps_run_in_order_scoped_to_the_import_and_failures_are_warnings() {
        let dir = std::env::temp_dir().join(format!("bop-finish-{}", uuid::Uuid::new_v4()));
        let mine = put_frame(
            &dir,
            "a",
            Some("import:x"),
            vec![
                auto_box("person", 0.9, Triage::Accept),
                auto_box("car", 0.2, Triage::Review),
            ],
        );
        let theirs = put_frame(
            &dir,
            "b",
            Some("import:y"),
            vec![
                auto_box("person", 0.9, Triage::Accept),
                auto_box("car", 0.2, Triage::Review),
            ],
        );
        assert!(Finish::default().is_empty());
        assert!(Finish::default().apply(&dir, Some("x"), &[]).is_empty());
        let w = Arc::new(Writer {
            saved: Mutex::new(None),
            fail: false,
        });
        let f = Finish {
            reject_below: Some(0.3),
            accept_auto: true,
            review_folder: Some(dir.clone()),
            config_writer: Some(w.clone()),
        };
        assert!(!f.is_empty());
        assert!(format!("{f:?}").contains("config_writer: true"));
        let classes = vec!["person".to_string(), "car".into()];
        let lines = f.apply(&dir, Some("x"), &classes);
        assert_eq!(
            lines,
            [
                "rejected 1 auto box(es) scoring below 0.3 on 1 frame(s) of import x".to_string(),
                "accepted 1 frame(s) of import x whose boxes were all auto: accept (1 boxes; \
                 saved as reviewed)"
                    .into(),
                format!("the Review page now shows {}", dir.display()),
            ]
        );
        assert_eq!(
            labels_of(&dir, &mine).status,
            crate::collect::labels::Status::Reviewed
        );
        let t = labels_of(&dir, &theirs);
        assert_eq!(
            t.status,
            crate::collect::labels::Status::Unreviewed,
            "another import's frame"
        );
        assert!(
            t.boxes
                .iter()
                .all(|b| b.state == crate::collect::labels::BoxState::Pending)
        );
        assert_eq!(*w.saved.lock().unwrap(), Some(dir.clone()));

        // The folder switch fails or has no writer: a warning line.
        let mut g = Finish {
            review_folder: Some(dir.clone()),
            config_writer: Some(Arc::new(Writer {
                saved: Mutex::new(None),
                fail: true,
            })),
            ..Finish::default()
        };
        let l = g.apply(&dir, None, &classes);
        assert_eq!(l.len(), 1);
        assert!(
            l[0].starts_with("warning: the Review page was not switched"),
            "{l:?}"
        );
        assert!(l[0].contains("read-only"), "{l:?}");
        g.config_writer = None;
        assert!(g.apply(&dir, None, &classes)[0].starts_with("warning: "));

        // Without a scope, every frame: the other import's frame now.
        let h = Finish {
            reject_below: Some(0.3),
            ..Finish::default()
        };
        assert_eq!(
            h.apply(&dir, None, &classes),
            ["rejected 1 auto box(es) scoring below 0.3 on 1 frame(s)"]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lines_are_appended_to_the_outcome() {
        let mut out = TaskOutcome {
            summary: "2 frames labelled".into(),
            result: serde_json::json!({ "frames": 2, "warnings": ["w0"] }),
        };
        append(&mut out, vec![]);
        assert_eq!(out.summary, "2 frames labelled");
        append(
            &mut out,
            vec!["accepted 2 frame(s)".into(), "warning: not switched".into()],
        );
        assert_eq!(
            out.summary,
            "2 frames labelled; then: accepted 2 frame(s); warning: not switched"
        );
        assert_eq!(
            out.result["warnings"],
            serde_json::json!(["w0", "not switched"])
        );
        assert_eq!(out.result["finish"].as_array().unwrap().len(), 2);
        assert_eq!(out.result["frames"], 2);
    }
}
