//! Benchmark: compile each model, then time the production per-request pipeline
//! (decode -> preprocess -> infer -> postprocess, as in `worker.rs`) over N repeats.
//! Thin CLI over [`blue_onyx_prism::benchmark`]; also available as `blue-onyx-prism benchmark`.
//!
//! ```text
//! blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --device CPU --repeat 20
//! blue-onyx-prism-benchmark --model models/yolo26s.xml --compare-cpu      # GPU vs CPU + confidence diff
//! blue-onyx-prism-benchmark --json                                         # every enabled model in the config
//! blue-onyx-prism-benchmark --all-devices --apply                          # pick the best device per model
//! ```

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
        missing_debug_implementations
    )
)]
#![deny(unsafe_code)]

use blue_onyx_prism::benchmark::cli::{BenchArgs, main as bench_main};
use clap::Parser;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "blue-onyx-prism-benchmark",
    version = blue_onyx_prism::cli::long_version_str(),
    about = "Benchmark Blue Onyx Prism models with the production pre/post-processing pipeline"
)]
struct Cli {
    #[command(flatten)]
    args: BenchArgs,
}

fn main() -> ExitCode {
    bench_main(Cli::parse().args, None)
}
