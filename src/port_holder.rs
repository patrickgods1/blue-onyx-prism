//! Which process listens on a TCP port, for the message of a bind failure ("port 32168 is held
//! by blue-onyx-prism.exe (PID 4120)").
//!
//! Asks the OS tools rather than the socket tables (no `unsafe` outside `backend/`): Windows
//! `netstat -ano` then `tasklist` (both by their `System32` path), elsewhere `lsof`, then `ss` on
//! Linux. They run through [`crate::command`], which reads their output while they run (`netstat
//! -ano` on a busy host prints far more than a pipe holds). Their output is parsed by pure
//! functions tested on every OS. The answer is best effort: a tool that is missing, slow or not
//! allowed to see another account's processes gives [`Lookup::Unknown`], with the reason.

use std::ffi::OsStr;
use std::process::Command;
use std::time::Duration;

/// How long one tool may run.
const TOOL_TIMEOUT: Duration = Duration::from_secs(5);

/// Bytes of a tool's output read: `netstat -ano` lists every socket of the machine (about 80
/// bytes a line), so 1 MiB holds some 13 000 of them; the other tools print one line or a few.
const TOOL_OUTPUT_LIMIT: usize = 1024 * 1024;

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
    let netstat = crate::command::system32("netstat.exe");
    let netstat = match run(&netstat, &["-ano"], TOOL_TIMEOUT) {
        Ok(out) => out,
        Err(e) => return Lookup::Unknown(format!("netstat: {e}")),
    };
    let pid = match netstat_pid(&netstat, port) {
        Ok(pid) => pid,
        Err(lookup) => return lookup,
    };
    let filter = format!("PID eq {pid}");
    let tasklist = crate::command::system32("tasklist.exe");
    let name = run(
        &tasklist,
        &["/FI", &filter, "/FO", "CSV", "/NH"],
        TOOL_TIMEOUT,
    )
    .ok()
    .and_then(|out| parse_tasklist(&out.text, pid));
    Lookup::Held(Holder { pid, name })
}

/// The listener's PID in a `netstat -ano` run, else what to answer: [`Lookup::Free`], or
/// [`Lookup::Unknown`] when the output was cut off before the listener could have been listed.
#[cfg_attr(
    not(windows),
    allow(dead_code, reason = "Windows only; tested everywhere")
)]
fn netstat_pid(out: &Ran, port: u16) -> Result<u32, Lookup> {
    match parse_netstat(&out.text, port) {
        Some(pid) => Ok(pid),
        None if out.truncated => Err(Lookup::Unknown(format!(
            "netstat listed more than {} KiB of sockets and the port was not among the first; \
             run `netstat -ano | findstr :{port}` to see the holder",
            TOOL_OUTPUT_LIMIT / 1024
        ))),
        None => Err(Lookup::Free),
    }
}

#[cfg(not(windows))]
fn lookup_unix(port: u16) -> Lookup {
    let spec = format!("-iTCP:{port}");
    let lsof = run(
        "lsof",
        &["-nP", &spec, "-sTCP:LISTEN", "-Fpc"],
        TOOL_TIMEOUT,
    );
    match &lsof {
        // Without `ss` to ask, an empty lsof answer is only "free" when it could see every
        // account's sockets (QA-056).
        Ok(out) if !cfg!(target_os = "linux") => {
            return lsof_lookup(&out.text, crate::command::euid(), port);
        }
        Ok(out) => {
            if let Some(h) = parse_lsof(&out.text) {
                return Lookup::Held(h);
            }
        }
        Err(_) if cfg!(target_os = "linux") => {}
        // lsof exits 1 when it finds nothing; `run` reports that with empty output.
        Err(e) => return Lookup::Unknown(format!("lsof: {e}")),
    }
    if cfg!(target_os = "linux") {
        let filter = format!("sport = :{port}");
        match run("ss", &["-ltnpH", &filter], TOOL_TIMEOUT) {
            Ok(out) => return parse_ss(&out.text),
            Err(e) => {
                let lsof = lsof.err().unwrap_or_default();
                return Lookup::Unknown(format!("ss: {e}; lsof: {lsof}"));
            }
        }
    }
    Lookup::Free
}

/// A tool's stdout as text, and whether it was cut at [`TOOL_OUTPUT_LIMIT`].
#[derive(Debug, Clone, PartialEq, Eq)]
struct Ran {
    text: String,
    truncated: bool,
}

/// Run `program args` for at most `timeout` ([`TOOL_TIMEOUT`] in the lookups); stdout as text,
/// stderr dropped. A non-zero exit with output still counts (lsof exits 1 when nothing matches).
fn run(program: impl AsRef<OsStr>, args: &[&str], timeout: Duration) -> Result<Ran, String> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    let out = crate::command::run_capped(cmd, timeout, None, TOOL_OUTPUT_LIMIT)
        .map_err(|e| format!("could not start it: {e}"))?;
    match out.ended {
        crate::command::Ended::Exited => Ok(Ran {
            text: out.stdout_text(),
            truncated: out.stdout_truncated,
        }),
        crate::command::Ended::TimedOut | crate::command::Ended::Abandoned => {
            Err(format!("no answer within {} s", timeout.as_secs()))
        }
    }
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

/// What an `lsof -nP -iTCP:<port> -sTCP:LISTEN -Fpc` run says where no other tool is asked
/// (macOS and the other non-Linux Unixes). lsof run by an account other than root lists only
/// that account's processes and prints nothing for another's listener, so an empty answer is
/// [`Lookup::Free`] only for root (`euid` 0) and else [`Lookup::Unknown`] with the command to
/// run as root (QA-056; Linux asks `ss`, which tells an unseen listener apart, instead).
#[cfg_attr(
    any(windows, target_os = "linux"),
    allow(dead_code, reason = "macOS and other Unixes only; tested everywhere")
)]
fn lsof_lookup(out: &str, euid: Option<u32>, port: u16) -> Lookup {
    if let Some(h) = parse_lsof(out) {
        return Lookup::Held(h);
    }
    if euid == Some(0) {
        return Lookup::Free;
    }
    Lookup::Unknown(format!(
        "no process of this account listens there, and a process of another account may hold \
         it, which this account cannot see; run `sudo lsof -nP -iTCP:{port} -sTCP:LISTEN`"
    ))
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

    // QA-056
    #[test]
    fn an_empty_lsof_answer_is_free_only_for_root() {
        let held = Lookup::Held(Holder {
            pid: 812,
            name: Some("codeproject".into()),
        });
        for euid in [Some(0), Some(501), None] {
            assert_eq!(lsof_lookup("p812\nccodeproject\n", euid, 32168), held);
        }
        assert_eq!(lsof_lookup("", Some(0), 32168), Lookup::Free);
        for euid in [Some(501), None] {
            let Lookup::Unknown(why) = lsof_lookup("", euid, 32168) else {
                panic!("{euid:?}: an account that is not root cannot see every listener");
            };
            assert!(
                why.contains("sudo lsof -nP -iTCP:32168 -sTCP:LISTEN"),
                "{why}"
            );
        }
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
    /// netstat always exists on Windows and lists every account's sockets, so there it must
    /// find it (it used to time out on a full pipe).
    #[test]
    fn lookup_finds_a_listener_of_this_process() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        match lookup(port) {
            Lookup::Held(h) => assert_eq!(h.pid, std::process::id()),
            other if cfg!(windows) => panic!("{other:?}"),
            Lookup::Unknown(_) | Lookup::Free => {}
        }
    }

    /// A tool printing more than a pipe buffer (here 3 MB) answers with the first
    /// TOOL_OUTPUT_LIMIT bytes, marked cut; a silent one times out with the old text; a missing
    /// one cannot start.
    #[cfg(unix)]
    #[test]
    fn run_reads_big_output_and_keeps_its_texts() {
        let t = std::time::Instant::now();
        let out = run(
            "/bin/sh",
            &[
                "-c",
                "yes 'TCP 0.0.0.0:1 0.0.0.0:0 LISTENING 1' | head -c 3000000; exit 1",
            ],
            Duration::from_secs(20),
        )
        .unwrap();
        assert!(t.elapsed() < Duration::from_secs(15));
        assert_eq!(out.text.len(), TOOL_OUTPUT_LIMIT);
        assert!(out.truncated);
        assert_eq!(parse_netstat(&out.text, 1), Some(1));
        let small = run("/bin/sh", &["-c", "echo p7"], Duration::from_secs(10)).unwrap();
        assert_eq!(
            small,
            Ran {
                text: "p7\n".into(),
                truncated: false
            }
        );
        assert_eq!(
            run("/bin/sh", &["-c", "sleep 30"], Duration::from_secs(1)),
            Err("no answer within 1 s".to_string())
        );
        let e = run("/nonexistent/netstat", &[], Duration::from_secs(1)).unwrap_err();
        assert!(e.starts_with("could not start it: "), "{e}");
    }

    /// A listener missing from a cut-off netstat listing is unknown, never "free".
    #[test]
    fn cut_netstat_output_is_not_free() {
        let ran = |text: &str, truncated| Ran {
            text: text.into(),
            truncated,
        };
        assert_eq!(netstat_pid(&ran(NETSTAT, false), 32168), Ok(4120));
        assert_eq!(netstat_pid(&ran(NETSTAT, true), 32168), Ok(4120));
        assert_eq!(netstat_pid(&ran(NETSTAT, false), 80), Err(Lookup::Free));
        let Err(Lookup::Unknown(why)) = netstat_pid(&ran(NETSTAT, true), 80) else {
            panic!("a cut listing gave an answer");
        };
        assert!(why.contains("findstr :80"), "{why}");
    }
}
