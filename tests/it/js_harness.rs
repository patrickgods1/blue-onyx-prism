//! The one `node` runner of the `js_*` tests (QA-268): whether `node` runs here, and running a
//! driver script in a folder of its own.
//!
//! - [`have_node`]: a missing `node` is a skip (the test says so and passes), unless
//!   `BOP_REQUIRE_NODE=1` (CI), which makes it a failure: a skip must not read as a pass there.
//! - [`run_node`]: the driver and its input files go in a `common::temp_dir` guard (removed when
//!   the run ends, panics included; kept with `BOP_KEEP_TEST_DIRS=1`); a run that exits non-zero
//!   fails the test with node's stdout and stderr. [`run_node_ok`] also wants `ok` as the last
//!   line the driver prints; [`run_node_json`] reads stdout as JSON.
#![allow(clippy::unwrap_used)]

use serde_json::Value;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::{Command, Output};

/// Whether `node` runs here. Missing: a skip, or with `BOP_REQUIRE_NODE=1` a failure.
pub fn have_node() -> bool {
    let ok = Command::new("node")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        assert!(
            std::env::var("BOP_REQUIRE_NODE").as_deref() != Ok("1"),
            "BOP_REQUIRE_NODE=1 but `node` does not run: install Node or unset the variable"
        );
        println!("skipped: no node");
    }
    ok
}

/// `assets/<name>` of this checkout.
pub fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(name)
}

/// Run the JavaScript `driver` with `node`. `process.argv[2..]` are `args`, then the path of each
/// of `files` (name, contents), written beside the driver. Panics with node's stdout and stderr
/// when it exits non-zero.
pub fn run_node(driver: &str, args: &[&OsStr], files: &[(&str, &[u8])]) -> Output {
    let dir = crate::common::temp_dir("js");
    let script = dir.join("driver.js");
    std::fs::write(&script, driver).unwrap();
    let mut cmd = Command::new("node");
    cmd.arg(&script).args(args);
    for (name, contents) in files {
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        cmd.arg(path);
    }
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "node failed ({}): {}{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// [`run_node`] for a driver that ends by printing `ok`: its last line must be `ok`, so a driver
/// that stopped early (an `await` that never settled, a forgotten call) does not pass.
pub fn run_node_ok(driver: &str, args: &[&OsStr]) -> Output {
    let out = run_node(driver, args, &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        stdout.lines().map(str::trim).rfind(|l| !l.is_empty()),
        Some("ok"),
        "the driver did not end with `ok`: {stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// [`run_node`] for a driver that prints one JSON value: that value.
pub fn run_node_json(driver: &str, args: &[&OsStr], files: &[(&str, &[u8])]) -> Value {
    let out = run_node(driver, args, files);
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "node printed no JSON ({e}): {}",
            String::from_utf8_lossy(&out.stdout)
        )
    })
}
