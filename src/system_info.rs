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

/// The system commit limit and what is free of it now (Windows only; `None` elsewhere or when the
/// query fails). Not cached: it changes with every program started.
pub fn commit() -> Option<crate::backend::memory::Commit> {
    crate::backend::memory::commit_bytes()
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

/// Environment variable Windows PowerShell 5.1 finds its modules through.
pub const PS_MODULE_PATH: &str = "PSModulePath";

/// Windows PowerShell 5.1: `System32\WindowsPowerShell\v1.0\powershell.exe` under
/// `SystemRoot` (`C:\Windows` when unset), an absolute path, so no `powershell.exe` on the
/// search path runs instead. Built on every OS (tests).
pub fn windows_powershell_exe() -> std::path::PathBuf {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
    std::path::Path::new(&root)
        .join("System32")
        .join("WindowsPowerShell")
        .join("v1.0")
        .join("powershell.exe")
}

/// The one way the product starts Windows PowerShell 5.1 (`program`, usually
/// [`windows_powershell_exe`]): a `Command` without [`PS_MODULE_PATH`] in the child's
/// environment.
///
/// Why: a process started from a PowerShell 7 terminal inherits 7's `PSModulePath`, whose
/// PowerShell 7 module folders come first. Windows PowerShell then autoloads 7's copies of its
/// own modules and fails ("The 'Get-Acl' command was found in the module
/// 'Microsoft.PowerShell.Security', but the module could not be loaded"; any autoloaded
/// cmdlet can fail so, such as the `Add-Type`, `Start-Process` and `Get-CimInstance` the
/// product's scripts use). Without the variable, 5.1 builds its own
/// default. Seen in CI, where every step runs in PowerShell 7; the service, the console exe and
/// the tray hit it whenever they were started from a PowerShell 7 terminal. Only for Windows
/// PowerShell: PowerShell 7 (`pwsh`) children need 7's module path.
pub fn windows_powershell_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    without_ps_module_path(&mut cmd);
    cmd
}

/// Removes [`PS_MODULE_PATH`] from `cmd`'s environment, also when it was set on `cmd` itself
/// (a later `env_remove` wins over an earlier `env`). Case-insensitive on Windows.
fn without_ps_module_path(cmd: &mut std::process::Command) -> &mut std::process::Command {
    cmd.env_remove(PS_MODULE_PATH)
}

/// Longest wait for the Windows memory query (CIM can hang on a broken WMI provider).
#[cfg(windows)]
const MEMORY_QUERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Windows: `GlobalMemoryStatusEx` (`backend::memory`, no process spawned); only when that
/// fails, the PowerShell fallback [`total_memory_bytes_powershell`].
#[cfg(windows)]
fn total_memory_bytes() -> Option<u64> {
    crate::backend::memory::total_physical_bytes().or_else(|| {
        tracing::debug!("GlobalMemoryStatusEx failed; asking CIM through PowerShell");
        total_memory_bytes_powershell()
    })
}

/// Windows fallback: ask CIM through the system PowerShell, by absolute path (never the search
/// path) and with [`MEMORY_QUERY_TIMEOUT`]; a hung or missing PowerShell gives `None` (logged).
#[cfg(windows)]
fn total_memory_bytes_powershell() -> Option<u64> {
    let exe = windows_powershell_exe();
    match output_within(memory_query_command(&exe), MEMORY_QUERY_TIMEOUT) {
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

/// The CIM query of `total_memory_bytes_powershell` (`Get-CimInstance` is module-autoloaded).
#[cfg_attr(not(windows), allow(dead_code))]
fn memory_query_command(exe: &std::path::Path) -> std::process::Command {
    let mut cmd = windows_powershell_command(exe);
    cmd.args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
    ]);
    cmd
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
pub(crate) mod tests {
    #[test]
    fn nonempty() {
        assert!(!super::cpu_name().is_empty());
        assert!(!super::os_description().is_empty());
        let _ = super::total_memory_gb();
        // Cached: the same answer twice.
        assert_eq!(super::total_memory_gb(), super::total_memory_gb());
    }

    /// `(key, None)` in `get_envs()` = removed from the child's environment.
    pub(crate) fn removes_ps_module_path(cmd: &std::process::Command) -> bool {
        cmd.get_envs()
            .any(|(k, v)| k.eq_ignore_ascii_case(super::PS_MODULE_PATH) && v.is_none())
    }

    #[test]
    fn windows_powershell_children_get_no_ps_module_path() {
        let exe = super::windows_powershell_exe();
        assert!(exe.ends_with("powershell.exe"), "{}", exe.display());
        let cmd = super::windows_powershell_command(&exe);
        assert!(removes_ps_module_path(&cmd));
        assert_eq!(cmd.get_program(), exe.as_os_str());
        // Set on the command itself first: the removal still wins.
        let mut cmd = std::process::Command::new(&exe);
        cmd.env(super::PS_MODULE_PATH, "bogus");
        super::without_ps_module_path(&mut cmd);
        assert!(removes_ps_module_path(&cmd));
        assert!(removes_ps_module_path(&super::memory_query_command(&exe)));
    }

    /// The real thing: a bogus module path (a folder with a broken
    /// `Microsoft.PowerShell.Security` first, as PowerShell 7's is for 5.1) set for the child,
    /// then removed by the helper, and an autoloaded cmdlet still resolves.
    #[cfg(windows)]
    #[test]
    fn windows_powershell_loads_its_modules_despite_a_foreign_module_path() {
        let dir = std::env::temp_dir().join(format!("bop-psmodulepath-{}", std::process::id()));
        let module = dir.join("Microsoft.PowerShell.Security");
        std::fs::create_dir_all(&module).unwrap();
        std::fs::write(
            module.join("Microsoft.PowerShell.Security.psm1"),
            "throw 'bogus module'\n",
        )
        .unwrap();
        let mut cmd = std::process::Command::new(super::windows_powershell_exe());
        cmd.env(super::PS_MODULE_PATH, &dir);
        super::without_ps_module_path(&mut cmd);
        let out = cmd
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$ErrorActionPreference = 'Stop'; Get-Command Get-Acl | Out-Null; \
                 $null = Get-Acl -LiteralPath $PSHOME; 'ok'",
            ])
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let text = String::from_utf8_lossy(&out.stdout);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{text}\n{err}");
        assert_eq!(text.trim(), "ok", "{err}");
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
