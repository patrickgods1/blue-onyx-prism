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

/// The message of a caught panic.
fn panic_text(e: &(dyn std::any::Any + Send)) -> String {
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default()
}

/// The harness itself (QA-147): a CommonJS driver whose `await` never settles exits 0 once node's
/// event loop is empty, so only the final `ok` line tells it from one that ran to the end;
/// [`run_node_ok`] fails it, and one that printed something after `ok`.
#[test]
fn a_driver_that_stops_early_fails() {
    if !have_node() {
        return;
    }
    let hung = "(async () => { await new Promise(() => {}); console.log('ok'); })();";
    let out = run_node(hung, &[], &[]);
    assert!(
        out.status.success(),
        "node exits 0 on a promise that never settles"
    );
    let Err(caught) = std::panic::catch_unwind(|| run_node_ok(hung, &[])) else {
        panic!("a driver that never got to `ok` passed");
    };
    let text = panic_text(caught.as_ref());
    assert!(text.contains("did not end with `ok`"), "{text}");
    let late =
        "(async () => { console.log('ok'); await Promise.resolve(); console.log('{}'); })();";
    assert!(
        std::panic::catch_unwind(|| run_node_ok(late, &[])).is_err(),
        "`ok` must be the last line"
    );
    run_node_ok(
        "(async () => { await Promise.resolve(); console.log('ok'); })();",
        &[],
    );
}
