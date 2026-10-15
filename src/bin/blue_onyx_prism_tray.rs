//! `blue-onyx-prism-tray`: the Windows tray / macOS menu bar app of Blue Onyx Prism
//! ([`blue_onyx_prism::tray`]). Shows the background service's status and starts, stops or
//! restarts it (elevated); on Linux it prints how to control the service and exits 2.
#![cfg_attr(windows, windows_subsystem = "windows")]

use clap::Parser as _;

fn main() -> std::process::ExitCode {
    blue_onyx_prism::tray::run(blue_onyx_prism::tray::TrayArgs::parse())
}
