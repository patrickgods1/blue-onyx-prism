//! "Your cameras" on the Benchmark pages ([`crate::benchmark::cameras`]): its row on the Run step
//! (first; the line of frames, cameras, days, sizes and scenes, or why it cannot be ticked),
//! `labelsChanged` of `GET /v1/benchmark` (frames whose labels changed since the snapshot a run
//! scored against), and `POST /v1/benchmark/rescore`. The counts come from the background walk of
//! [`crate::benchmark::cameras::live`]: a poll never walks the review folder.

use super::pages::DatasetRow;
use super::{AppState, json_error, raw};
use crate::benchmark::cameras::live::{self, Found};
use crate::benchmark::cameras::snapshot::{self, GtSnapshot};
use crate::benchmark::cameras::{ID, Source, TITLE};
use crate::benchmark::results_path;
use crate::config::Config;
use crate::sync::LockExt;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::value::RawValue;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long the page waits for the first walk of the folder before it shows "counting".
const PAGE_WAIT: Duration = Duration::from_millis(1500);

/// The Run step's row of the set: the line of frames, cameras, days, sizes and scenes, or why it
/// cannot be ticked ("review frames first", a missing folder).
pub(super) fn dataset_row(c: &Config, checked: bool) -> DatasetRow {
    let src = Source::of(c);
    let mut current = live::peek(&src);
    let deadline = Instant::now() + PAGE_WAIT;
    while current.is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
        current = live::peek(&src);
    }
    let base = DatasetRow {
        value: ID.to_string(),
        title: TITLE.to_string(),
        tests: String::new(),
        size: String::new(),
        images: 0,
        state_text: String::new(),
        chip: "same",
        checked,
        download_id: None,
        folder: false,
        selectable: true,
    };
    match current.as_deref().map(|c| &c.found) {
        Some(Found::Ready { survey, .. }) => DatasetRow {
            tests: survey.line(),
            size: format!("{} labelled objects", survey.objects),
            images: survey.frames,
            state_text: if survey.low_confidence() {
                "low confidence".into()
            } else {
                "recommended".into()
            },
            chip: if survey.low_confidence() {
                "warn"
            } else {
                "ok"
            },
            ..base
        },
        Some(Found::Missing(why)) => DatasetRow {
            tests: why.clone(),
            state_text: "folder missing".into(),
            chip: "bad",
            checked: false,
            selectable: false,
            ..base
        },
        Some(Found::Empty) => DatasetRow {
            tests: "No reviewed frame yet: review frames on the Train area's Review step to \
                    benchmark on your own cameras"
                .into(),
            state_text: "no reviewed frame".into(),
            checked: false,
            selectable: false,
            ..base
        },
        None => DatasetRow {
            tests: format!(
                "Counting the reviewed frames of {} (reload for the numbers)",
                src.dir.display()
            ),
            state_text: "counting".into(),
            ..base
        },
    }
}

/// The last `labelsChanged` and what it was computed from (the snapshot's stamp, the walk's time,
/// the source): the snapshot (MBs of labels) is parsed only when one of them changes.
static LAST: Mutex<Option<(String, Arc<RawValue>)>> = Mutex::new(None);

/// The snapshot identity the saved results recorded for Your cameras and its frame count, read
/// without loading the results (None: no such set).
fn recorded_snapshot(results: &std::path::Path) -> Option<(Option<snapshot::SnapshotId>, usize)> {
    #[derive(serde::Deserialize)]
    struct Set {
        id: String,
        #[serde(default)]
        snapshot: Option<snapshot::SnapshotId>,
        #[serde(default)]
        images: Vec<serde::de::IgnoredAny>,
    }
    #[derive(serde::Deserialize)]
    struct Head {
        #[serde(default)]
        image_sets: Vec<Set>,
    }
    let bytes = std::fs::read(results).ok()?;
    let head: Head = serde_json::from_slice(&bytes).ok()?;
    head.image_sets
        .into_iter()
        .find(|s| s.id == ID)
        .map(|s| (s.snapshot, s.images.len()))
}

/// `labelsChanged` of `GET /v1/benchmark`: `null` without a snapshot; else `{frames, changed,
/// left, joined, unknown}` ([`snapshot::Changes`]) from the last walk, `{frames: null, computing:
/// true}` before the first one, `{frames: null, folderMissing: true, message}` when the folder
/// cannot be read. When the snapshot is not the one the results were saved with
/// ([`snapshot::mismatch`]): `stale: true` and the `message`, `frames` at least the set's frames,
/// so the page offers a Re-score. Never waits for a walk.
pub(super) fn labels_changed(state: &AppState, c: &Config) -> Arc<RawValue> {
    let results = results_path(&state.config_path);
    let gt = snapshot::path_beside(&results);
    let stamp = super::file_stamp(&gt);
    if stamp.is_none() {
        return RawValue::NULL.to_owned().into();
    }
    let src = Source::of(c);
    let current = live::peek(&src);
    let key = format!(
        "{}|{stamp:?}|{:?}|{}|{:?}",
        gt.display(),
        super::file_stamp(&results),
        src.key(),
        current.as_ref().map(|c| c.at)
    );
    if let Some((k, v)) = LAST.lock_recover().as_ref()
        && *k == key
    {
        return v.clone();
    }
    let value = match current.as_deref().map(|c| &c.found) {
        None => serde_json::json!({ "frames": null, "computing": true }),
        Some(Found::Missing(why)) => {
            serde_json::json!({ "frames": null, "folderMissing": true, "message": why })
        }
        Some(found) => match GtSnapshot::load_with_id(&gt) {
            Ok(Some((snap, id))) => {
                let empty = std::collections::BTreeMap::new();
                let now = match found {
                    Found::Ready { reviewed, .. } => reviewed,
                    _ => &empty,
                };
                let mut changes = snapshot::changes(&snap, now);
                let stale = recorded_snapshot(&results).and_then(|(recorded, frames)| {
                    snapshot::mismatch(recorded.as_ref(), &id).map(|why| (why, frames))
                });
                if let Some((_, frames)) = &stale {
                    changes.frames = changes.frames.max(*frames);
                }
                let mut v = serde_json::to_value(changes).unwrap_or_default();
                if let (Some((why, _)), Some(o)) = (stale, v.as_object_mut()) {
                    o.insert("stale".into(), true.into());
                    o.insert("message".into(), why.into());
                }
                v
            }
            Ok(None) => serde_json::Value::Null,
            Err(e) => serde_json::json!({ "frames": null, "message": format!("{e:#}") }),
        },
    };
    let value: Arc<RawValue> = raw(&value).into();
    *LAST.lock_recover() = Some((key, value.clone()));
    value
}

/// `POST /v1/benchmark/rescore`: the stored predictions of Your cameras scored against the
/// labels now, no inference ([`crate::benchmark::cameras::rescore`]); the snapshot is replaced.
/// Under the results lock as a Re-score (a run's start answers 409 meanwhile). 409 while a
/// benchmark runs (checked under the lock) and when the snapshot is of another folder or class
/// list (run the benchmark again); 503 `folder_missing` when the review folder cannot be read.
pub(in crate::server) async fn rescore(State(state): State<Arc<AppState>>) -> Response {
    crate::server::blocking(move || {
        let path = results_path(&state.config_path);
        let held = match crate::benchmark::results_io::lock(
            &path,
            crate::benchmark::results_io::Holder::Rescore,
        ) {
            Ok(h) => h,
            Err(e) => {
                return json_error(StatusCode::CONFLICT, format!("Not re-scored: {e:#}"));
            }
        };
        if state.benchmark.is_running() {
            return json_error(
                StatusCode::CONFLICT,
                "a benchmark is running; re-score once it has finished".into(),
            );
        }
        let cfg = state.config_read().clone();
        let src = Source::of(&cfg);
        if let Some(why) = src.problem {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "success": false, "folder_missing": true, "message": why,
                })),
            )
                .into_response();
        }
        let preds = crate::benchmark::search::preds_path(&state.config_path);
        let done = crate::benchmark::cameras::rescore::rescore_locked(&held, &cfg, &path, &preds);
        drop(held);
        match done {
            Ok(r) => {
                tracing::info!(
                    runs = r.runs,
                    frames = r.frames,
                    changed = r.changes.changed,
                    left = r.changes.left,
                    unknown = r.unknown.len(),
                    need_run = r.need_run,
                    "benchmark: {TITLE} re-scored against the current labels"
                );
                // The page's next poll walks again for the new counts.
                live::walk(&src);
                Json(serde_json::json!({ "success": true, "rescored": r })).into_response()
            }
            Err(e) => json_error(
                if e.downcast_ref::<crate::benchmark::cameras::rescore::SourceChanged>()
                    .is_some()
                {
                    StatusCode::CONFLICT
                } else {
                    StatusCode::UNPROCESSABLE_ENTITY
                },
                format!("Not re-scored: {e:#}"),
            ),
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::super::tests::{call, get, post, state};
    use super::*;
    use crate::benchmark::cameras::tests::{
        flat, labels_path, lb, put, ready, reviewed, scored_results,
    };
    use crate::benchmark::images::CamerasLoaded;
    use crate::collect::labels::{self, BoxState};

    /// Two reviewed frames of the state's dataset folder, a scored run of them saved with its
    /// snapshot; returns the dataset folder.
    fn seeded(st: &Arc<AppState>) -> std::path::PathBuf {
        let cfg = st.config_read().clone();
        let dir = cfg.collect.resolved_review_dir(&cfg.data_root());
        let person = lb("person", [4.0, 4.0, 40.0, 40.0], BoxState::Accepted);
        put(
            &dir,
            "day/front/2026-11-01/a",
            &reviewed(vec![person.clone()]),
        );
        put(&dir, "day/front/2026-11-02/b", &reviewed(vec![person]));
        let src = Source::of(&cfg);
        let l = ready(crate::benchmark::cameras::load(&src));
        let (results, m) = scored_results(&l.set);
        let path = results_path(&st.config_path);
        let id = crate::benchmark::cameras::write_snapshot(
            &path,
            std::slice::from_ref(&l.set),
            Some(&CamerasLoaded {
                source: src.clone(),
                reviewed: l.reviewed.clone(),
            }),
            None,
        )
        .unwrap()
        .unwrap();
        let mut results = results;
        crate::benchmark::cameras::snapshot::stamp(&mut results, &id);
        results.save(&path).unwrap();
        crate::benchmark::search::StoredPreds::merge(None, &results, &[m])
            .save(&crate::benchmark::search::preds_path(&st.config_path))
            .unwrap();
        live::walk(&src);
        dir
    }

    fn st_with_classes() -> Arc<AppState> {
        let st = state();
        st.config
            .update("test", |c| {
                c.collect.classes = vec!["person".into(), "car".into()];
                Ok(())
            })
            .unwrap();
        st
    }

    #[tokio::test]
    async fn labels_changed_counts_edits_and_a_rescore_catches_up() {
        let st = st_with_classes();
        let dir = seeded(&st);
        let (s, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["labelsChanged"]["frames"], 0, "{}", v["labelsChanged"]);
        // An edit through the label write path, seen by the next walk.
        labels::update(&labels_path(&dir, "day/front/2026-11-02/b"), |f| {
            let mut f = f.unwrap();
            f.boxes
                .push(lb("person", [44.0, 4.0, 60.0, 40.0], BoxState::Accepted));
            Ok(Some(f))
        })
        .unwrap();
        live::walk(&Source::of(&st.config_read()));
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        let c = &v["labelsChanged"];
        assert_eq!(
            (c["frames"].as_u64(), c["changed"].as_u64()),
            (Some(1), Some(1)),
            "{c}"
        );
        let (s, v) = call(&st, post("/v1/benchmark/rescore", "")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["rescored"]["runs"], 1, "{v}");
        assert_eq!(v["rescored"]["changes"]["changed"], 1, "{v}");
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(v["labelsChanged"]["frames"], 0, "{}", v["labelsChanged"]);
        // The folder gone: refused as folder_missing, never an empty set.
        st.config
            .update("test", |c| {
                c.collect.review_dir = Some(dir.join("unplugged"));
                Ok(())
            })
            .unwrap();
        let (s, v) = call(&st, post("/v1/benchmark/rescore", "")).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{v}");
        assert_eq!(v["folder_missing"], true);
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(
            v["labelsChanged"]["folderMissing"], true,
            "{}",
            v["labelsChanged"]
        );
    }

    /// D2: a run does not start while a Re-score holds the saved results (and a Re-score checks
    /// for a run under the same lock).
    #[tokio::test]
    async fn a_start_is_refused_while_a_rescore_holds_the_results() {
        use crate::benchmark::results_io::{Holder, lock};
        let st = st_with_classes();
        seeded(&st);
        let held = lock(&results_path(&st.config_path), Holder::Rescore).unwrap();
        let (s, v) = call(&st, post("/v1/benchmark", "dataset=sample")).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert!(
            v["message"]
                .as_str()
                .unwrap_or_default()
                .contains("re-scored"),
            "{v}"
        );
        assert!(!st.benchmark.is_running());
        drop(held);
    }

    /// D3: results whose snapshot was replaced without them (an interrupted save) are reported
    /// stale through `labelsChanged`, so the page offers a Re-score, which clears it.
    #[tokio::test]
    async fn a_snapshot_the_results_were_not_saved_with_is_reported_stale() {
        let st = st_with_classes();
        seeded(&st);
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert!(
            v["labelsChanged"].get("stale").is_none(),
            "{}",
            v["labelsChanged"]
        );
        let gt = snapshot::path_beside(&results_path(&st.config_path));
        let mut snap = GtSnapshot::load(&gt).unwrap().unwrap();
        snap.created = "2030-01-01T00:00:00Z".into();
        snap.save(&gt).unwrap();
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        let c = &v["labelsChanged"];
        assert_eq!(c["stale"], true, "{c}");
        assert_eq!(c["frames"], 2, "{c}");
        assert!(c["message"].as_str().unwrap().contains("Re-score"), "{c}");
        let (s, v) = call(&st, post("/v1/benchmark/rescore", "")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert!(
            v["labelsChanged"].get("stale").is_none(),
            "{}",
            v["labelsChanged"]
        );
    }

    #[tokio::test]
    async fn the_image_route_serves_only_frames_of_the_snapshot() {
        let st = st_with_classes();
        let dir = seeded(&st);
        let url = |file: &str| format!("/v1/benchmark/image?set={ID}&file={file}");
        let (s, _) = call(&st, get(&url(&flat("day/front/2026-11-01/a")))).await;
        assert_eq!(s, StatusCode::OK);
        // A frame of the folder no run scored, a traversal, a nested path, another file.
        put(
            &dir,
            "day/front/2026-11-03/c",
            &reviewed(vec![lb(
                "person",
                [4.0, 4.0, 40.0, 40.0],
                BoxState::Accepted,
            )]),
        );
        for bad in [
            flat("day/front/2026-11-03/c"),
            "..%2F..%2Fcfg.json".to_string(),
            "day%2Ffront%2F2026-11-01%2Fa.jpg".to_string(),
            "day.front.2026-11-01.a.labels.json".to_string(),
        ] {
            let (s, v) = call(&st, get(&url(&bad))).await;
            assert_eq!(s, StatusCode::NOT_FOUND, "{bad}: {v}");
        }
        // A frame of the results that is no longer in the snapshot is not served either.
        let gt = snapshot::path_beside(&results_path(&st.config_path));
        let mut snap = GtSnapshot::load(&gt).unwrap().unwrap();
        snap.frames
            .retain(|f| f.file != flat("day/front/2026-11-01/a"));
        snap.save(&gt).unwrap();
        let (s, _) = call(&st, get(&url(&flat("day/front/2026-11-01/a")))).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        let (s, _) = call(&st, get(&url(&flat("day/front/2026-11-02/b")))).await;
        assert_eq!(s, StatusCode::OK);
    }

    #[tokio::test]
    async fn the_run_step_lists_your_cameras_first() {
        let st = st_with_classes();
        seeded(&st);
        let (s, v) = call(&st, get("/benchmark/run")).await;
        assert_eq!(s, StatusCode::OK);
        let html = v.as_str().unwrap();
        let cams = html
            .find("name=\"dataset\" value=\"cameras\"")
            .expect("listed");
        let coco = html.find("name=\"dataset\" value=\"coco-cctv\"").unwrap();
        assert!(cams < coco, "first");
        assert!(
            html.contains("name=\"dataset\" value=\"cameras\" data-images=\"2\" checked"),
            "selected by default, with its frame count for the estimate"
        );
        assert!(
            html.contains("2 frames \u{b7} 1 camera \u{b7} 2 camera days"),
            "the line"
        );
        // No reviewed frame: shown, not selectable, with what to do.
        let st = state();
        let (_, v) = call(&st, get("/benchmark/run")).await;
        let html = v.as_str().unwrap();
        assert!(
            html.contains("name=\"dataset\" value=\"cameras\" data-images=\"0\" disabled"),
            "disabled"
        );
        assert!(html.contains("No reviewed frame yet"));
    }
}
