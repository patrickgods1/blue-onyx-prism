//! Registry of child processes the service started (`<data root>/.children.json`; audit 2026-11,
//! orphan sweep). Every long-running child (export, training, auto-label, import, refine) is
//! recorded with its PID, start time and executable when it starts and forgotten when it is
//! reaped; a start after a crash kills only recorded entries whose PID, start time and exe all
//! still match ([`crate::backend::process::start_time`] / `exe_path`), so a reused PID is never
//! touched.
