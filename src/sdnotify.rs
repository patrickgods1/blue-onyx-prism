//! systemd service notification (`Type=notify`, watchdog) without libsystemd: datagrams to
//! `$NOTIFY_SOCKET`. Everything is a no-op on Windows or when the variable is unset.

use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Send one `sd_notify` state line (e.g. `READY=1`, `WATCHDOG=1`, `STOPPING=1`). Returns true
/// when it was written; false without `$NOTIFY_SOCKET`, on non-Unix targets, or on error.
pub fn notify(state: &str) -> bool {
    #[cfg(unix)]
    {
        match std::env::var_os("NOTIFY_SOCKET") {
            Some(p) if !p.is_empty() => notify_to(&p, state),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = state;
        false
    }
}

#[cfg(unix)]
fn notify_to(path: &std::ffi::OsStr, state: &str) -> bool {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::net::UnixDatagram;
    let bytes = path.as_bytes();
    let Ok(sock) = UnixDatagram::unbound() else {
        return false;
    };
    if let Some(name) = bytes.strip_prefix(b"@") {
        // Abstract socket: `@name` means a leading NUL.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            use std::os::linux::net::SocketAddrExt;
            use std::os::unix::net::SocketAddr;
            SocketAddr::from_abstract_name(name)
                .and_then(|a| sock.send_to_addr(state.as_bytes(), &a))
                .is_ok()
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            let _ = (name, &sock);
            false
        }
    } else {
        sock.send_to(state.as_bytes(), path).is_ok()
    }
}

/// Watchdog ping interval: half of `WATCHDOG_USEC` (systemd's recommendation), or None when no
/// watchdog is set or `WATCHDOG_PID` names another process.
pub fn watchdog_interval() -> Option<Duration> {
    interval_from(
        std::env::var("WATCHDOG_USEC").ok().as_deref(),
        std::env::var("WATCHDOG_PID").ok().as_deref(),
        std::process::id(),
    )
}

fn interval_from(usec: Option<&str>, pid: Option<&str>, me: u32) -> Option<Duration> {
    let usec: u64 = usec?.trim().parse().ok()?;
    if usec == 0 {
        return None;
    }
    if let Some(p) = pid
        && p.trim().parse::<u32>().ok() != Some(me)
    {
        return None;
    }
    Some(Duration::from_micros(usec / 2))
}

/// Start the watchdog thread when systemd asked for one. While `ready()` is true it sends
/// `WATCHDOG=1` every interval; when it is false it stays silent so systemd kills a hung
/// process. On `shutdown` it sends `STOPPING=1` and exits. Does nothing without a watchdog.
pub fn spawn_watchdog(ready: impl Fn() -> bool + Send + 'static, shutdown: CancellationToken) {
    let Some(interval) = watchdog_interval() else {
        return;
    };
    spawn_with(interval, ready, shutdown, notify);
}

fn spawn_with(
    interval: Duration,
    ready: impl Fn() -> bool + Send + 'static,
    shutdown: CancellationToken,
    send: impl Fn(&str) -> bool + Send + 'static,
) -> Option<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("sd-watchdog".into())
        .spawn(move || {
            let step = Duration::from_millis(100).min(interval);
            let mut waited = Duration::ZERO;
            loop {
                if shutdown.is_cancelled() {
                    send("STOPPING=1");
                    return;
                }
                if waited >= interval {
                    waited = Duration::ZERO;
                    if ready() {
                        send("WATCHDOG=1");
                    }
                }
                std::thread::sleep(step);
                waited += step;
            }
        })
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn interval_parsing() {
        assert_eq!(
            interval_from(Some("120000000"), None, 5),
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            interval_from(Some("120000000"), Some("5"), 5),
            Some(Duration::from_secs(60))
        );
        assert_eq!(interval_from(Some("120000000"), Some("6"), 5), None);
        assert_eq!(interval_from(Some("0"), None, 5), None);
        assert_eq!(interval_from(Some("abc"), None, 5), None);
        assert_eq!(interval_from(None, None, 5), None);
    }

    #[test]
    fn watchdog_pings_only_while_ready_then_stops() {
        let sent = Arc::new(Mutex::new(Vec::<String>::new()));
        let ready = Arc::new(AtomicBool::new(false));
        let token = CancellationToken::new();
        let s = sent.clone();
        let r = ready.clone();
        let h = spawn_with(
            Duration::from_millis(100),
            move || r.load(Ordering::SeqCst),
            token.clone(),
            move |m| {
                s.lock().unwrap().push(m.to_string());
                true
            },
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(450));
        assert!(sent.lock().unwrap().is_empty(), "no ping while not ready");
        ready.store(true, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(450));
        token.cancel();
        h.join().unwrap();
        let got = sent.lock().unwrap().clone();
        assert!(got.iter().any(|m| m == "WATCHDOG=1"));
        assert_eq!(got.last().map(String::as_str), Some("STOPPING=1"));
    }

    #[cfg(unix)]
    #[test]
    fn notify_writes_to_socket() {
        use std::os::unix::net::UnixDatagram;
        let dir = std::env::temp_dir().join(format!("bo_sdn_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("n.sock");
        let rx = UnixDatagram::bind(&path).unwrap();
        rx.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        assert!(notify_to(path.as_os_str(), "READY=1"));
        let mut buf = [0u8; 64];
        let n = rx.recv(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"READY=1");
        assert!(!notify_to(dir.join("missing").as_os_str(), "READY=1"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
