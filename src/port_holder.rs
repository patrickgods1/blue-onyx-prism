//! Which process listens on a TCP port, for the message of a bind failure ("port 32168 is held
//! by blue-onyx-prism.exe (PID 4120)").
//!
//! Asks the OS tools rather than the socket tables (no `unsafe` outside `backend/`): Windows
//! `netstat -ano` then `tasklist`, elsewhere `lsof`, then `ss` on Linux. Their output is parsed
//! by pure functions tested on every OS. The answer is best effort: a tool that is missing, slow
//! or not allowed to see another account's processes gives [`Lookup::Unknown`], with the reason.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How long one tool may run.
const TOOL_TIMEOUT: Duration = Duration::from_secs(5);

/// A process listening on the port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Holder {
    pub pid: u32,
    /// Executable or command name, when the tools tell it.
    pub name: Option<String>,
}

/// What [`lookup`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lookup {
    Held(Holder),
    /// The tools ran and no listener was on the port (it may have been freed meanwhile).
    Free,
    /// The holder could not be found out; the reason says why.
    Unknown(String),
}

impl Lookup {
    /// One sentence for a log line or error message.
    pub fn describe(&self, port: u16) -> String {
        match self {
            Lookup::Held(Holder {
                pid,
                name: Some(name),
            }) => format!("port {port} is held by {name} (PID {pid})"),
            Lookup::Held(Holder { pid, name: None }) => {
                format!("port {port} is held by the process with PID {pid}")
            }
            Lookup::Free => format!(
                "no process was listening on port {port} when it was checked (it may have been \
                 freed since, or the address is not usable on this machine)"
            ),
            Lookup::Unknown(why) => {
                format!("which process holds port {port} could not be found out ({why})")
            }
        }
    }
}

/// The process listening on TCP `port` on this machine.
pub fn lookup(port: u16) -> Lookup {
    #[cfg(windows)]
    {
        lookup_windows(port)
    }
    #[cfg(not(windows))]
    {
        lookup_unix(port)
    }
}

#[cfg(windows)]
fn lookup_windows(port: u16) -> Lookup {
    let netstat = match run("netstat", &["-ano"]) {
        Ok(out) => out,
        Err(e) => return Lookup::Unknown(format!("netstat: {e}")),
    };
    let Some(pid) = parse_netstat(&netstat, port) else {
        return Lookup::Free;
    };
    let filter = format!("PID eq {pid}");
    let name = run("tasklist", &["/FI", &filter, "/FO", "CSV", "/NH"])
        .ok()
        .and_then(|out| parse_tasklist(&out, pid));
    Lookup::Held(Holder { pid, name })
}

#[cfg(not(windows))]
fn lookup_unix(port: u16) -> Lookup {
    let spec = format!("-iTCP:{port}");
    let lsof = run("lsof", &["-nP", &spec, "-sTCP:LISTEN", "-Fpc"]);
    match &lsof {
        Ok(out) => {
            if let Some(h) = parse_lsof(out) {
                return Lookup::Held(h);
            }
        }
        Err(_) if cfg!(target_os = "linux") => {}
        // lsof exits 1 when it finds nothing; `run` reports that with empty output.
        Err(e) => return Lookup::Unknown(format!("lsof: {e}")),
    }
    if cfg!(target_os = "linux") {
        let filter = format!("sport = :{port}");
        match run("ss", &["-ltnpH", &filter]) {
            Ok(out) => return parse_ss(&out),
            Err(e) => {
                let lsof = lsof.err().unwrap_or_default();
                return Lookup::Unknown(format!("ss: {e}; lsof: {lsof}"));
            }
        }
    }
    Lookup::Free
}

/// Run `program args`, at most [`TOOL_TIMEOUT`]; stdout as text. A non-zero exit with output
/// still counts (lsof exits 1 when nothing matches).
fn run(program: &str, args: &[&str]) -> Result<String, String> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        /// No console window for a tool started by the tray or a console-less process.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not start it: {e}"))?;
    let deadline = Instant::now() + TOOL_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("no answer within {} s", TOOL_TIMEOUT.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => return Err(format!("waiting for it failed: {e}")),
        }
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("reading its output failed: {e}"))?;
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The port of a `host:port` / `[v6]:port` address column.
fn port_of(addr: &str) -> Option<u16> {
    addr.rsplit_once(':').and_then(|(_, p)| p.parse().ok())
}

/// The PID listening on `port` in `netstat -ano` output. The state column is translated on
/// non-English Windows, so a listener is told by its foreign address (`0.0.0.0:0`, `[::]:0`,
/// port 0) instead of the word LISTENING.
pub fn parse_netstat(out: &str, port: u16) -> Option<u32> {
    out.lines().find_map(|line| {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let [proto, local, foreign, .., pid] = cols.as_slice() else {
            return None;
        };
        if !proto.eq_ignore_ascii_case("TCP") {
            return None;
        }
        (port_of(local) == Some(port) && port_of(foreign) == Some(0))
            .then(|| pid.parse().ok())
            .flatten()
    })
}

/// The image name of `pid` in `tasklist /FO CSV /NH` output
/// (`"blue-onyx-prism.exe","4120","Services","0","81,232 K"`). Anything else, e.g. the
/// translated "no tasks" line, gives None.
pub fn parse_tasklist(out: &str, pid: u32) -> Option<String> {
    out.lines().find_map(|line| {
        let fields: Vec<&str> = line.split("\",\"").collect();
        let name = fields.first()?.trim().trim_start_matches('"');
        let listed: u32 = fields.get(1)?.trim_matches('"').parse().ok()?;
        (listed == pid && !name.is_empty()).then(|| name.to_string())
    })
}

/// The first process of `lsof -F pc` output (`p<pid>` then `c<command>` lines).
pub fn parse_lsof(out: &str) -> Option<Holder> {
    let mut holder: Option<Holder> = None;
    for line in out.lines() {
        if let Some(pid) = line.strip_prefix('p') {
            if holder.is_some() {
                break;
            }
            holder = pid
                .trim()
                .parse()
                .ok()
                .map(|pid| Holder { pid, name: None });
        } else if let (Some(c), Some(h)) = (line.strip_prefix('c'), holder.as_mut()) {
            h.name = Some(c.trim().to_string()).filter(|c| !c.is_empty());
        }
    }
    holder
}

/// `ss -ltnpH` output: a line means a listener; its `users:(("name",pid=N,fd=M))` names the
/// process when this account may see it.
pub fn parse_ss(out: &str) -> Lookup {
    let Some(line) = out.lines().find(|l| !l.trim().is_empty()) else {
        return Lookup::Free;
    };
    let Some(users) = line.split_once("users:((").map(|(_, u)| u) else {
        return Lookup::Unknown(
            "a process listens there but this account may not see which; run the same check \
             as an administrator (ss -ltnp)"
                .to_string(),
        );
    };
    let name = users
        .split_once('"')
        .and_then(|(_, rest)| rest.split_once('"'))
        .map(|(n, _)| n.to_string());
    let pid = users
        .split_once("pid=")
        .and_then(|(_, rest)| rest.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|p| p.parse().ok());
    match pid {
        Some(pid) => Lookup::Held(Holder { pid, name }),
        None => Lookup::Unknown(format!("unexpected ss output: {line}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NETSTAT: &str = "\r
Active Connections\r
\r
  Proto  Local Address          Foreign Address        State           PID\r
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1032\r
  TCP    127.0.0.1:32168        127.0.0.1:50211        ESTABLISHED     5000\r
  TCP    0.0.0.0:32168          0.0.0.0:0              ABH\u{d6}REN         4120\r
  TCP    [::]:32168             [::]:0                 ABH\u{d6}REN         4120\r
  UDP    0.0.0.0:32168          *:*                                    77\r
";

    #[test]
    fn netstat_finds_the_listener_in_any_language() {
        assert_eq!(parse_netstat(NETSTAT, 32168), Some(4120));
        assert_eq!(parse_netstat(NETSTAT, 135), Some(1032));
        assert_eq!(parse_netstat(NETSTAT, 80), None);
        // An established connection on the port alone is no listener.
        let only_conn = "  TCP    127.0.0.1:9000   127.0.0.1:50211   ESTABLISHED   5000";
        assert_eq!(parse_netstat(only_conn, 9000), None);
        let v6 = "  TCP    [::1]:9000   [::]:0   LISTENING   12";
        assert_eq!(parse_netstat(v6, 9000), Some(12));
    }

    #[test]
    fn tasklist_names_the_pid() {
        let out = "\"blue-onyx-prism.exe\",\"4120\",\"Services\",\"0\",\"81,232 K\"\r\n";
        assert_eq!(
            parse_tasklist(out, 4120).as_deref(),
            Some("blue-onyx-prism.exe")
        );
        assert_eq!(parse_tasklist(out, 4121), None);
        assert_eq!(
            parse_tasklist(
                "INFORMATION: Es werden keine Aufgaben mit den angegebenen Kriterien ausgef\u{fc}hrt.",
                4120
            ),
            None
        );
    }

    #[test]
    fn lsof_gives_the_first_process() {
        let out = "p812\nccodeproject\nf9\np900\ncother\n";
        assert_eq!(
            parse_lsof(out),
            Some(Holder {
                pid: 812,
                name: Some("codeproject".into())
            })
        );
        assert_eq!(parse_lsof(""), None);
        assert_eq!(parse_lsof("p7\n"), Some(Holder { pid: 7, name: None }));
    }

    #[test]
    fn ss_with_and_without_process_info() {
        let out =
            "LISTEN 0 4096 0.0.0.0:32168 0.0.0.0:* users:((\"blue-onyx-prism\",pid=1234,fd=9))\n";
        assert_eq!(
            parse_ss(out),
            Lookup::Held(Holder {
                pid: 1234,
                name: Some("blue-onyx-prism".into())
            })
        );
        assert_eq!(parse_ss(""), Lookup::Free);
        assert!(matches!(
            parse_ss("LISTEN 0 4096 0.0.0.0:32168 0.0.0.0:*\n"),
            Lookup::Unknown(_)
        ));
    }

    #[test]
    fn descriptions_name_port_and_process() {
        let held = Lookup::Held(Holder {
            pid: 4120,
            name: Some("CodeProject.AI.Server.exe".into()),
        });
        assert_eq!(
            held.describe(32168),
            "port 32168 is held by CodeProject.AI.Server.exe (PID 4120)"
        );
        assert!(
            Lookup::Unknown("netstat: x".into())
                .describe(1)
                .contains("could not be found out (netstat: x)")
        );
    }

    /// The real lookup on this machine finds the listener this test opens (where the tools
    /// exist and may see this process; otherwise it must at least not claim another holder).
    #[test]
    fn lookup_finds_a_listener_of_this_process() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        match lookup(port) {
            Lookup::Held(h) => assert_eq!(h.pid, std::process::id()),
            Lookup::Unknown(_) | Lookup::Free => {}
        }
    }
}
