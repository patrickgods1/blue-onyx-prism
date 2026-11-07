//! The process-wide panic hook every binary installs first thing in `main` ([`install`]).
//!
//! A panic on any thread is written through `tracing` at error level (target `panic`): the
//! thread's name, the message, the source location, and a backtrace when `RUST_BACKTRACE` (or
//! `RUST_LIB_BACKTRACE`) asks for one. Then the previous hook runs (by default Rust's own, which
//! prints to stderr), so nothing that worked before is lost. Without this, a panic on a worker or
//! server thread of the Windows service (stderr discarded) or of a daemon whose stderr nobody
//! reads left no trace in the log file or the event log.

use std::cell::Cell;
use std::sync::Once;

/// What the hook logs about one panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanicReport {
    /// The thread's name, or `<unnamed>` (as Rust's own hook prints it).
    pub thread: String,
    pub message: String,
    /// `file:line:column`, when the panic carries one.
    pub location: Option<String>,
    /// The captured backtrace (only when `RUST_BACKTRACE` / `RUST_LIB_BACKTRACE` enable it).
    pub backtrace: Option<String>,
}

impl PanicReport {
    /// The report of the panic `info` on the current thread.
    pub fn of(info: &std::panic::PanicHookInfo<'_>) -> Self {
        let payload = info.payload();
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".to_string());
        let backtrace = std::backtrace::Backtrace::capture();
        Self {
            thread: std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_string(),
            message,
            location: info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column())),
            backtrace: (backtrace.status() == std::backtrace::BacktraceStatus::Captured)
                .then(|| backtrace.to_string()),
        }
    }

    /// One line in the words of Rust's own hook: `thread 'x' panicked at file:1:2: message`.
    pub fn summary(&self) -> String {
        match &self.location {
            Some(at) => format!(
                "thread '{}' panicked at {at}: {}",
                self.thread, self.message
            ),
            None => format!("thread '{}' panicked: {}", self.thread, self.message),
        }
    }
}

thread_local! {
    /// Set while this thread's hook logs: a panic inside the logging (a subscriber that panics)
    /// goes straight to the previous hook instead of recursing.
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

/// Log `report` at error level (target `panic`).
pub fn log(report: &PanicReport) {
    match &report.backtrace {
        Some(bt) => tracing::error!(
            target: "panic",
            thread = %report.thread,
            location = report.location.as_deref().unwrap_or("unknown"),
            backtrace = %bt,
            "{}. This is a bug: please report it with the log",
            report.summary()
        ),
        None => tracing::error!(
            target: "panic",
            thread = %report.thread,
            location = report.location.as_deref().unwrap_or("unknown"),
            "{}. This is a bug: please report it with the log (set RUST_BACKTRACE=1 for a \
             backtrace)",
            report.summary()
        ),
    }
}

/// Install the hook (once per process; later calls do nothing). Call it before anything else
/// in `main`: events logged before the subscriber is set up go nowhere, but the previous hook
/// still prints them to stderr.
pub fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let reentered = IN_HOOK.with(|f| f.replace(true));
            if !reentered {
                log(&PanicReport::of(info));
                IN_HOOK.with(|f| f.set(false));
            }
            previous(info);
        }));
    });
}

/// Sends a value to the thread's owner when the thread ends, normally ([`ExitSignal::finish`])
/// or by a panic (the `on_unwind` value, sent from `Drop` while unwinding). The Windows service
/// waits on such a channel: without the unwind case a panicking server thread left the service
/// reporting Running with no HTTP server.
#[derive(Debug)]
pub struct ExitSignal<T> {
    tx: Option<std::sync::mpsc::Sender<T>>,
    on_unwind: Option<T>,
}

impl<T> ExitSignal<T> {
    pub fn new(tx: std::sync::mpsc::Sender<T>, on_unwind: T) -> Self {
        Self {
            tx: Some(tx),
            on_unwind: Some(on_unwind),
        }
    }

    /// The thread is done: send `value` (and nothing on drop).
    pub fn finish(mut self, value: T) {
        self.on_unwind = None;
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(value);
        }
    }
}

impl<T> Drop for ExitSignal<T> {
    fn drop(&mut self) {
        if let (Some(tx), Some(v)) = (self.tx.take(), self.on_unwind.take()) {
            let _ = tx.send(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_signal_is_sent_once_on_return_and_on_panic() {
        let (tx, rx) = std::sync::mpsc::channel::<&str>();
        let signal = ExitSignal::new(tx.clone(), "panicked");
        std::thread::spawn(move || signal.finish("returned"))
            .join()
            .unwrap();
        assert_eq!(rx.recv().unwrap(), "returned");
        assert!(rx.try_recv().is_err(), "sent once");

        let signal = ExitSignal::new(tx, "panicked");
        let joined = std::thread::spawn(move || {
            let _signal = signal;
            panic!("server thread failure");
        })
        .join();
        assert!(joined.is_err());
        assert_eq!(rx.recv().unwrap(), "panicked");
        assert!(rx.try_recv().is_err(), "sent once");
    }
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::layer::SubscriberExt;

    /// Collects every event's message and fields as text.
    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<String>>>);

    struct Visit<'a>(&'a mut String);

    impl tracing::field::Visit for Visit<'_> {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0.push_str(&format!(" {}={:?}", field.name(), value));
        }
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Capture {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut s = format!("{} {}", event.metadata().level(), event.metadata().target());
            event.record(&mut Visit(&mut s));
            self.0.lock().unwrap().push(s);
        }
    }

    /// A panic on a spawned, named thread reaches the subscriber of that thread at error
    /// level with its thread, message and location, and the previous hook still runs (the
    /// join sees the panic as before).
    #[test]
    fn a_panic_on_a_spawned_thread_is_logged() {
        install();
        install(); // a second call changes nothing
        let capture = Capture::default();
        let sub = tracing_subscriber::registry().with(capture.clone());
        let joined = std::thread::Builder::new()
            .name("panic-test-worker".into())
            .spawn(move || {
                tracing::subscriber::with_default(sub, || {
                    // Other tests' threads use the hook's callsites with no subscriber of
                    // theirs; their cached "never" must not hide this one's events.
                    tracing::callsite::rebuild_interest_cache();
                    panic!("synthetic failure 42");
                })
            })
            .unwrap()
            .join();
        assert!(joined.is_err(), "the thread still panics");
        let events = capture.0.lock().unwrap().clone();
        let line = events
            .iter()
            .find(|e| e.contains("synthetic failure 42"))
            .unwrap_or_else(|| panic!("panic not logged: {events:?}"));
        // Windows: `src\panic.rs`, and the field's Debug quoting doubles the backslash.
        let line = line.replace("\\\\", "/").replace('\\', "/");
        assert!(line.starts_with("ERROR panic"), "{line}");
        assert!(line.contains("thread=panic-test-worker"), "{line}");
        assert!(
            line.contains("thread 'panic-test-worker' panicked at src/panic.rs:"),
            "{line}"
        );
        assert!(line.contains("location=\"src/panic.rs:"), "{line}");
    }

    #[test]
    fn summary_matches_the_std_wording() {
        let r = PanicReport {
            thread: "server".into(),
            message: "boom".into(),
            location: Some("src/x.rs:1:2".into()),
            backtrace: None,
        };
        assert_eq!(
            r.summary(),
            "thread 'server' panicked at src/x.rs:1:2: boom"
        );
        let r = PanicReport {
            location: None,
            ..r
        };
        assert_eq!(r.summary(), "thread 'server' panicked: boom");
    }
}
