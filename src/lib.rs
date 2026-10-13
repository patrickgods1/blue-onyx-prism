//! Blue Onyx Prism (formerly Blue Onyx OpenVINO): a Blue Iris / CodeProject.AI compatible
//! object detection service on native OpenVINO, with ONNX Runtime as a second runtime.
//!
//! Module map (see CLAUDE.md):
//! - [`api`]       wire structs for the CodeProject.AI compatible JSON API
//! - [`config`]    JSON config next to the executable, merged with CLI flags
//! - [`config_form`] applying the web UI's forms (config page, models, devices, thresholds) to a `Config`
//! - [`config_store`] the live config with a revision and change log (every writer goes through it)
//! - [`config_merge`] field-level config view: change summaries, pending restart, 3-way form merge
//! - [`backend`]   runtimes (OpenVINO today), device specs, load plans, inference backends
//! - [`model`]     model families (yolo26, yolo5, yolo8, rtdetr): pre/post-processing
//! - [`prep`]      process-wide decode + preprocess pool shared by the workers
//! - [`profile`]   hardware profile (low/mid/high): defaults of the concurrency settings
//! - [`registry`]  loads all configured models and owns their worker threads
//! - [`worker`]    per-model inference thread
//! - [`startup`]   model readiness state shared with the HTTP layer
//! - [`server`]    axum HTTP server
//! - [`runner`]    server run loop (generations, restart) shared by the CLI and the service
//! - [`metrics`]   request/latency counters and Prometheus rendering
//! - [`logbuf`]    in-memory ring of recent log events for the web UI's Logs page
//! - [`resources`] downloadable runtimes/models: pinned catalog and the needs resolver
//! - [`benchmark`] per-model device benchmark (CLI, `/v1/benchmark`), results and recommendations
//! - [`sync`]      poison-recovering `Mutex` / `RwLock` helpers (`lock_recover`, `read_recover`, `write_recover`)

//! # Lint policy
//!
//! Non-test code warns on every panicking or truncating construct so that a new `unwrap`, index
//! or `as` cast must be justified at the site with `#[allow(..., reason = "...")]`; CI runs
//! clippy with `-D warnings`, so a warning is a failure. Tests (`cfg(test)`) are exempt.
//! `unsafe` is only permitted in [`backend`] (FFI into OpenVINO, ONNX Runtime, DXGI, CUDA and
//! `libloading`), every block with a `// SAFETY:` comment.
#![cfg_attr(
    not(test),
    warn(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::large_futures,
        clippy::unused_async,
        clippy::dbg_macro,
        clippy::undocumented_unsafe_blocks,
        missing_debug_implementations
    )
)]
#![warn(unsafe_op_in_unsafe_fn)]
#![deny(unsafe_code)]

pub mod api;
#[allow(
    unsafe_code,
    reason = "FFI into OpenVINO, ONNX Runtime, DXGI, CUDA and libloading"
)]
pub mod backend;
pub mod benchmark;
pub mod cli;
pub mod config;
pub mod config_form;
pub mod config_merge;
pub mod config_store;
pub mod download;
pub mod image;
pub mod logbuf;
pub mod metrics;
pub mod model;
pub mod prep;
pub mod profile;
pub mod registry;
pub mod resources;
pub mod runner;
pub mod server;
pub mod setup_onnxruntime;
pub mod setup_openvino;
pub mod startup;
pub mod sync;
pub mod system_info;
pub mod update;
pub mod worker;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const MODULE_ID: &str = "ObjectDetectionPrism";
pub const MODULE_NAME: &str = "Object Detection (Prism)";
pub const PROCESSED_BY: &str = "BlueOnyxPrism";
pub const DEFAULT_PORT: u16 = 32168;

/// Directory containing the running executable (falls back to cwd).
pub fn exe_dir() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
}

/// Resolve a possibly relative path against the executable directory.
pub fn resolve_path(p: &std::path::Path) -> std::path::PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        exe_dir().join(p)
    }
}
