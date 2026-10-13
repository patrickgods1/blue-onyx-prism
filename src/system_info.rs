//! Best-effort host information for logging and the status page.

/// CPU brand string.
pub fn cpu_name() -> String {
    cpu_name_impl()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_name_impl() -> String {
    let cpuid = raw_cpuid::CpuId::new();
    cpuid
        .get_processor_brand_string()
        .map(|b| b.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Unknown CPU".to_string())
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn cpu_name_impl() -> String {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            && out.status.success()
            && let Ok(s) = String::from_utf8(out.stdout)
            && !s.trim().is_empty()
        {
            return s.trim().to_string();
        }
        return "Apple Silicon".to_string();
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
            for key in ["model name", "Model", "Hardware"] {
                if let Some(v) = text
                    .lines()
                    .find(|l| l.starts_with(key))
                    .and_then(|l| l.split_once(':'))
                    .map(|(_, v)| v.trim().to_string())
                    .filter(|v| !v.is_empty())
                {
                    return v;
                }
            }
        }
    }
    #[allow(unreachable_code)]
    std::env::consts::ARCH.to_string()
}

/// Total physical memory in GiB, if it can be determined. Probed once per process and cached
/// (the hardware profile and the start-up log line both ask).
pub fn total_memory_gb() -> Option<f64> {
    static BYTES: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    BYTES
        .get_or_init(total_memory_bytes)
        .map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0))
}

#[cfg(target_os = "linux")]
fn total_memory_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = text
        .lines()
        .find(|l| l.starts_with("MemTotal:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kb * 1024)
}

#[cfg(target_os = "macos")]
fn total_memory_bytes() -> Option<u64> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

/// Longest wait for the Windows memory query (CIM can hang on a broken WMI provider).
#[cfg(windows)]
const MEMORY_QUERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Windows: ask CIM through the system PowerShell, by absolute path (never the search path)
/// and with [`MEMORY_QUERY_TIMEOUT`]; a hung or missing PowerShell gives `None` (logged).
/// `GlobalMemoryStatusEx` would need `unsafe`, which the project keeps in `src/backend/`.
#[cfg(windows)]
fn total_memory_bytes() -> Option<u64> {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
    let exe = std::path::Path::new(&root)
        .join("System32")
        .join("WindowsPowerShell")
        .join("v1.0")
        .join("powershell.exe");
    let mut cmd = std::process::Command::new(&exe);
    cmd.args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
    ]);
    match output_within(cmd, MEMORY_QUERY_TIMEOUT) {
        Ok(Some(out)) => String::from_utf8(out).ok()?.trim().parse().ok(),
        Ok(None) => {
            tracing::warn!(
                "the memory query ({}) did not answer within {} s; the hardware profile is \
                 classified without the RAM size (set hardware_profile to low/mid/high to \
                 force it)",
                exe.display(),
                MEMORY_QUERY_TIMEOUT.as_secs()
            );
            None
        }
        Err(e) => {
            tracing::warn!(
                "could not run {} to read the RAM size: {e}; the hardware profile is \
                 classified without it (set hardware_profile to low/mid/high to force it)",
                exe.display()
            );
            None
        }
    }
}

/// Run `cmd` and return its stdout, or `Ok(None)` (the child is killed) when it does not
/// exit within `timeout`. For commands with small output (the pipe cannot fill up).
#[cfg_attr(not(windows), allow(dead_code))]
fn output_within(
    mut cmd: std::process::Command,
    timeout: std::time::Duration,
) -> std::io::Result<Option<Vec<u8>>> {
    use std::io::Read;
    use std::process::Stdio;
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if child.try_wait()?.is_some() {
            let mut out = Vec::new();
            if let Some(mut s) = child.stdout.take() {
                s.read_to_end(&mut out)?;
            }
            return Ok(Some(out));
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn total_memory_bytes() -> Option<u64> {
    None
}

/// e.g. "windows x86_64".
pub fn os_description() -> String {
    format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
}

#[cfg(test)]
mod tests {
    #[test]
    fn nonempty() {
        assert!(!super::cpu_name().is_empty());
        assert!(!super::os_description().is_empty());
        let _ = super::total_memory_gb();
        // Cached: the same answer twice.
        assert_eq!(super::total_memory_gb(), super::total_memory_gb());
    }

    #[cfg(unix)]
    #[test]
    fn output_within_returns_stdout_or_times_out() {
        use std::time::Duration;
        let mut cmd = std::process::Command::new("/bin/sh");
        cmd.args(["-c", "echo 42"]);
        let out = super::output_within(cmd, Duration::from_secs(10))
            .unwrap()
            .unwrap();
        assert_eq!(String::from_utf8(out).unwrap().trim(), "42");
        let mut cmd = std::process::Command::new("/bin/sh");
        cmd.args(["-c", "sleep 5"]);
        let t = std::time::Instant::now();
        assert!(
            super::output_within(cmd, Duration::from_millis(200))
                .unwrap()
                .is_none()
        );
        assert!(t.elapsed() < Duration::from_secs(4));
        assert!(
            super::output_within(
                std::process::Command::new("/nonexistent/prog"),
                Duration::from_secs(1)
            )
            .is_err()
        );
    }
}
