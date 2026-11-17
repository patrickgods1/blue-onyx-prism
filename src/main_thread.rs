//! Runs a binary's `main` body on a thread with a larger stack ([`run`]).
//!
//! Windows gives the process's main thread a 1 MiB stack (the PE header's default), and a debug
//! build of this program needs more: clap's derived `Cli::command()` alone builds every
//! subcommand's arguments in a few huge unoptimized frames (measured in a debug build: about
//! 1,031 KiB for `blue-onyx-prism`, 839 KiB of it under `Command::augment_subcommands`, 583 KiB
//! under `train`), so a debug `blue-onyx-prism --version` overflowed before printing anything.
//! Server startup then fits in 1.5 MiB. Release builds never overflowed; they use the same thread so
//! both builds run the same way. The thread is named `main`, so panic messages and logs read as
//! before; a panic is re-raised on the real main thread (the panic hook has already reported it),
//! so the process exits with Rust's panic code as it did. Not for code that must stay on the real
//! main thread: the tray's event loop (macOS) and the Windows service's control dispatcher.

/// Stack of the thread [`run`] starts: a Linux main thread's default, over five times what a
/// debug build was measured to need (address space only; pages are committed as they are used).
pub const STACK_BYTES: usize = 8 * 1024 * 1024;

/// Runs `body` on a thread named `main` with a [`STACK_BYTES`] stack and returns its result. A
/// panic in `body` continues on the calling thread (`resume_unwind`, without a second report). If
/// the thread cannot be started at all (out of memory), says so and exits with code 1.
pub fn run<T, F>(body: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let spawned = std::thread::Builder::new()
        .name("main".into())
        .stack_size(STACK_BYTES)
        .spawn(body);
    match spawned {
        Ok(handle) => match handle.join() {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        },
        Err(e) => {
            eprintln!(
                "error: could not start the main thread with a {} MiB stack: {e}",
                STACK_BYTES / (1024 * 1024)
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Touches about `kib` KiB of stack, one 64 KiB frame per level, so the optimizer cannot
    /// fold it away.
    #[inline(never)]
    fn deep(kib: usize) -> u8 {
        let mut frame = [0u8; 64 * 1024];
        frame[kib % frame.len()] = 1;
        let below = if kib > 64 { deep(kib - 64) } else { 0 };
        std::hint::black_box(&mut frame);
        frame[0].wrapping_add(below)
    }

    #[test]
    fn body_runs_on_a_main_named_thread_with_the_big_stack() {
        let name = run(|| std::thread::current().name().map(str::to_string));
        assert_eq!(name.as_deref(), Some("main"));
        // Well past the 1 MiB of a Windows main thread and the 2 MiB of a default Rust thread.
        let _ = run(|| deep(6 * 1024));
        const { assert!(STACK_BYTES >= 8 * 1024 * 1024) };
    }

    #[test]
    fn the_cli_parser_builds_on_the_thread() {
        // What overflowed a debug build's 1 MiB Windows main thread (about 1,031 KiB measured).
        let subcommands = run(|| {
            use clap::CommandFactory;
            crate::cli::Cli::command().get_subcommands().count()
        });
        assert!(subcommands > 10);
    }

    #[test]
    fn a_panic_in_the_body_reaches_the_caller() {
        let caught = std::panic::catch_unwind(|| run(|| -> u8 { panic!("boom") }));
        let payload = caught.expect_err("the panic must cross the join");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom"));
    }
}
