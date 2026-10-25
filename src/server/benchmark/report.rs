//! `POST /v1/benchmark/report`: save the standalone HTML report of the saved results
//! (`benchmark.json` beside the config, shown against the config now, the same view as the page
//! and the CLI's `--report`) into a folder the admin chose, as
//! `benchmark-report-<UTC time>.html`. The folder must pass [`crate::server::browse::allowed_path`]
//! (inside the browsable folders or natively picked); the file is written atomically. The results
//! themselves never move: `auto` device ordering reads them beside the config.

use super::{devices_snapshot, json_error, models_now};
use crate::benchmark::{BenchmarkResults, results_path};
use crate::server::AppState;
use axum::Json;
use axum::extract::rejection::FormRejection;
use axum::extract::{Form, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

/// `POST /v1/benchmark/report` (urlencoded `dir`): 200 `{success, path, message}`; 400 without a
/// folder or with a file, 403/404 for a folder outside the browsable folders or missing, 404 when
/// there are no saved results, 500 when the file cannot be written.
pub(in crate::server) async fn save_report(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    let dir = form
        .iter()
        .find(|(k, _)| k == "dir")
        .map(|(_, v)| v.trim().to_string())
        .unwrap_or_default();
    crate::server::blocking(move || save_blocking(&state, &dir)).await
}

/// [`save_report`] off the runtime threads: it parses the results and writes the file.
fn save_blocking(state: &AppState, raw: &str) -> Response {
    if raw.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "No folder given: choose the folder to save the report in with Browse...".into(),
        );
    }
    let cfg = state.config_read().clone();
    let dir = match crate::server::browse::allowed_path(&cfg, raw) {
        Ok(d) => d,
        Err(e) => {
            let (status, message) = e.into_parts();
            return json_error(
                status,
                format!(
                    "The report was not saved: {message}. Choose the folder with Browse... (or \
                     type a folder inside the browsable folders)."
                ),
            );
        }
    };
    if !dir.is_dir() {
        return json_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The report was not saved: {raw} is a file, not a folder. Use Browse... to choose \
                 a folder."
            ),
        );
    }
    let results = results_path(&state.config_path);
    if !results.exists() {
        return json_error(
            StatusCode::NOT_FOUND,
            format!(
                "No benchmark results to report yet ({} does not exist): run a benchmark in \
                 step 1 Run first.",
                results.display()
            ),
        );
    }
    let Some(mut r) = BenchmarkResults::load_or_warn(&results) else {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "The report was not saved: {} could not be read (see the log); run the benchmark \
                 again to replace it.",
                results.display()
            ),
        );
    };
    // Shown against the config now, like the page and the CLI's --report.
    let snap = devices_snapshot(state);
    let (_, now_cfg) = state.config.snapshot();
    let now = models_now(state, &now_cfg, &snap);
    let preds = crate::benchmark::search::StoredPreds::load_cached(
        &crate::benchmark::search::preds_path(&state.config_path),
    )
    .map_err(|e| tracing::warn!("{e:#}"))
    .ok()
    .flatten();
    crate::benchmark::current::rebase(&mut r, preds.as_deref(), &now);
    let path = report_path(&dir, chrono::Utc::now());
    let html = crate::benchmark::export::html(&r);
    if let Err(e) = crate::collect::layout::write_atomic(&path, html.as_bytes()) {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "The report was not saved: writing {} failed: {e}. Check that the folder is \
                 writable or choose another one.",
                path.display()
            ),
        );
    }
    info!(path = %path.display(), "benchmark report saved from the web UI");
    Json(serde_json::json!({
        "success": true,
        "path": path.display().to_string(),
        "message": format!("Report saved to {}.", path.display()),
    }))
    .into_response()
}

/// `<dir>/benchmark-report-<YYYYMMDD-HHMMSS>Z.html`.
fn report_path(dir: &Path, at: chrono::DateTime<chrono::Utc>) -> PathBuf {
    dir.join(format!(
        "benchmark-report-{}.html",
        at.format("%Y%m%d-%H%M%SZ")
    ))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{call, post, state};
    use super::*;
    use crate::server::save_config;

    fn enc(p: &Path) -> String {
        crate::server::encode_uri_component(&p.display().to_string())
    }

    /// The state's folder as the data root (a browsable root).
    fn rooted() -> (Arc<AppState>, PathBuf) {
        let st = state();
        let dir = st.config_path.parent().unwrap().canonicalize().unwrap();
        let d = dir.clone();
        save_config(&st, "test", move |c| {
            c.download_dir = Some(d);
            Ok(())
        })
        .unwrap();
        (st, dir)
    }

    #[test]
    fn report_file_name_is_utc_stamped() {
        use chrono::TimeZone;
        let at = chrono::Utc.with_ymd_and_hms(2026, 10, 25, 9, 3, 7).unwrap();
        assert_eq!(
            report_path(Path::new("out"), at),
            Path::new("out").join("benchmark-report-20261025-090307Z.html")
        );
    }

    #[tokio::test]
    async fn saves_the_report_into_an_allowed_folder() {
        let (st, dir) = rooted();
        let out = dir.join("reports");
        std::fs::create_dir_all(&out).unwrap();

        // No results yet: a clear 404.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/report", &format!("dir={}", enc(&out))),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{v}");
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("No benchmark results"),
            "{v}"
        );

        let (results, _) = crate::benchmark::search::tests::fixture();
        results.save(&results_path(&st.config_path)).unwrap();
        let (s, v) = call(
            &st,
            post("/v1/benchmark/report", &format!("dir={}", enc(&out))),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["success"], true);
        let path = PathBuf::from(v["path"].as_str().unwrap());
        assert_eq!(path.parent(), Some(out.as_path()));
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with("benchmark-report-") && name.ends_with("Z.html"),
            "{name}"
        );
        let html = std::fs::read_to_string(&path).unwrap();
        assert!(html.contains("Blue Onyx Prism benchmark report"));
        // Nothing else (no temporary file) is left in the folder.
        assert_eq!(std::fs::read_dir(&out).unwrap().count(), 1);

        // No folder, or a file instead of a folder.
        let (s, _) = call(&st, post("/v1/benchmark/report", "dir=")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, v) = call(
            &st,
            post("/v1/benchmark/report", &format!("dir={}", enc(&path))),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert!(
            v["message"].as_str().unwrap().contains("not a folder"),
            "{v}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Outside every browsable root: refused. (Unix only: on Windows every drive is a root.)
    #[cfg(unix)]
    #[tokio::test]
    async fn refuses_a_folder_outside_the_roots() {
        let (st, dir) = rooted();
        let (results, _) = crate::benchmark::search::tests::fixture();
        results.save(&results_path(&st.config_path)).unwrap();
        let outside =
            std::env::temp_dir().join(format!("bo_bench_out_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&outside).unwrap();
        let (s, v) = call(
            &st,
            post("/v1/benchmark/report", &format!("dir={}", enc(&outside))),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{v}");
        let m = v["message"].as_str().unwrap();
        assert!(
            m.contains("outside the browsable folders") && m.contains("Browse..."),
            "{m}"
        );
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&outside);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
