//! The integration tests, one binary: linking fifteen of them cost about half of a `cargo test
//! --no-run` after a one-line edit (`docs/PRODUCTION_REVIEW.md`, "Build times"). They share a
//! process, so a test here must not need the process to itself: `tests/pipeline.rs` installs its
//! own global tracing subscriber and stays a binary of its own for that reason.
//!
//! One file's tests: `cargo test --test it js_rules`. Fixtures: `tests/common/mod.rs`.

#[path = "../common/mod.rs"]
mod common;

mod backend_cache;
mod backend_cpu;
mod backend_ort;
mod config_disk_http;
mod enabled_models_http;
mod families_http;
mod formats_020;
mod integration_http;
mod js_poll;
mod js_review_touch;
mod js_rules;
mod js_viewport;
mod live_config;
mod multi_model_http;
mod postprocess;
mod provision;
mod resources;
mod review_walk;
mod select;
mod template_inputs;
