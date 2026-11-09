//! `blue-onyx-prism user add|remove|list|reset-password|unlock` and `setup-code`: the account
//! commands, run on the server as the service's account (`--config` names the service's
//! config, which says where the data root is).
//!
//! Passwords never come from argv or the environment (shells keep history, `ps` shows argv).
//! On a terminal they are read without echo and asked twice; when stdin is not a terminal
//! (a script, `docker exec -i`) one line is read from stdin, its line ending removed and nothing
//! else (spaces are part of a password). Run as root on Unix, every file written is handed to
//! the data root's owner ([`crate::private_dir`]), so the service can still read it.

use super::{Store, password, setup};
use crate::cli::UserCommand;
use crate::config::Config;
use anyhow::{Context, Result, bail};
use std::io::{BufRead, Write};
use std::time::Duration;

/// How long a command waits for a hashing slot (the service may be busy with sign-ins).
const PERMIT_WAIT: Duration = Duration::from_secs(60);

/// Where a new password comes from.
pub trait Passwords {
    /// The new password for the account `name`.
    fn new_password(&mut self, name: &str) -> Result<String>;
}

/// The terminal without echo (asked twice), or one line of stdin when it is not a terminal.
#[derive(Debug, Default)]
pub struct Terminal;

impl Passwords for Terminal {
    fn new_password(&mut self, name: &str) -> Result<String> {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            return read_line_password(&mut std::io::stdin().lock());
        }
        eprintln!(
            "New password for {name}: {} to {} characters, a passphrase of four or more words \
             works well. Nothing is shown while you type.",
            password::MIN_LEN,
            password::MAX_LEN
        );
        let first = rpassword::prompt_password("Password: ").context("reading the password")?;
        let again =
            rpassword::prompt_password("Same password again: ").context("reading the password")?;
        if first != again {
            bail!("the two passwords differ, so nothing was changed; run the command again");
        }
        Ok(first)
    }
}

/// One line of `input` per password (a script's stdin, or a test's).
#[derive(Debug)]
pub struct Lines<R>(pub R);

impl<R: BufRead> Passwords for Lines<R> {
    fn new_password(&mut self, _name: &str) -> Result<String> {
        read_line_password(&mut self.0)
    }
}

/// One line of `input` with its line ending (`\n` or `\r\n`) removed and nothing else.
pub fn read_line_password(input: &mut dyn BufRead) -> Result<String> {
    let mut line = String::new();
    let n = input
        .read_line(&mut line)
        .context("reading the password from stdin (one line, UTF-8)")?;
    if n == 0 {
        bail!(
            "no password on stdin, so nothing was changed; run the command in a terminal, or \
             pipe the password in as one line"
        );
    }
    if let Some(l) = line.strip_suffix('\n') {
        line.truncate(l.len());
    }
    if let Some(l) = line.strip_suffix('\r') {
        line.truncate(l.len());
    }
    Ok(line)
}

fn permit() -> Result<password::Permit<'static>> {
    password::acquire_for_command(PERMIT_WAIT)
}

/// Run `cmd` on the accounts of `config`'s data root; what it says goes to `out`.
pub fn user(
    config: &Config,
    cmd: &UserCommand,
    passwords: &mut dyn Passwords,
    out: &mut dyn Write,
) -> Result<()> {
    let store = Store::open(&config.data_root());
    let now = super::now();
    match cmd {
        UserCommand::Add { name } => {
            let name = super::normalize_name(name)?;
            // Refuse a taken name (or an unreadable file) before asking for a password.
            if let Some(users) = store.load()?
                && users.get(&name).is_some()
            {
                return Err(super::Refused::NameTaken.into());
            }
            let pw = passwords.new_password(&name)?;
            let user = store.add(&name, &pw, &permit()?, now)?;
            writeln!(
                out,
                "Added the account {}. It signs in to the web UI at https://<this machine>:{}/.",
                user.name, config.https_port
            )?;
        }
        UserCommand::Remove { name } => {
            store.remove(name)?;
            writeln!(
                out,
                "Removed the account {}; its sessions end.",
                name.to_ascii_lowercase()
            )?;
        }
        UserCommand::List => list(&store, out)?,
        UserCommand::ResetPassword { name } => {
            let name = super::normalize_name(name).map_err(|_| super::Refused::UnknownName)?;
            match store.load()? {
                None => return Err(super::Refused::NoAccounts.into()),
                Some(u) if u.get(&name).is_none() => {
                    return Err(super::Refused::UnknownName.into());
                }
                Some(_) => {}
            }
            let pw = passwords.new_password(&name)?;
            let user = store.set_password(&name, &pw, &permit()?, now)?;
            writeln!(
                out,
                "Set a new password for {}; its sessions end, and its failed attempts were \
                 cleared (it is unlocked).",
                user.name
            )?;
        }
        UserCommand::Unlock { name } => {
            let user = store.unlock(name)?;
            writeln!(
                out,
                "Unlocked {}: its failed attempts were cleared; its sessions stay.",
                user.name
            )?;
        }
    }
    Ok(())
}

/// `user list`: names, dates, failures and the disabled flag; never hashes.
fn list(store: &Store, out: &mut dyn Write) -> Result<()> {
    let Some(users) = store.load()? else {
        writeln!(
            out,
            "No accounts yet. Create the first with `blue-onyx-prism user add <name>`, or get \
             the setup code for the web UI with `blue-onyx-prism setup-code`."
        )?;
        return Ok(());
    };
    if users.users.is_empty() {
        writeln!(
            out,
            "{} lists no accounts, so nobody can sign in; add one with `blue-onyx-prism user \
             add <name>`.",
            store.path().display()
        )?;
        return Ok(());
    }
    writeln!(
        out,
        "{:<32}  {:<20}  {:<20}  {:>8}  DISABLED",
        "NAME", "CREATED", "PASSWORD CHANGED", "FAILURES"
    )?;
    for u in &users.users {
        writeln!(
            out,
            "{:<32}  {:<20}  {:<20}  {:>8}  {}",
            u.name,
            super::rfc3339(u.created),
            super::rfc3339(u.password_changed),
            u.failures,
            if u.disabled { "yes" } else { "no" }
        )?;
    }
    Ok(())
}

/// `setup-code`: the code (made when needed) and how to reach the web UI, or why there is none.
pub fn setup_code(config: &Config) -> Result<String> {
    let root = config.data_root();
    let code = match setup::ensure_code(&root)? {
        setup::Ensured::Created(c) | setup::Ensured::Existing(c) => c,
        setup::Ensured::AccountsExist => bail!(
            "accounts exist already, so there is no setup code: sign in with one of them, add \
             another with `blue-onyx-prism user add <name>`, or reset a forgotten password on \
             this machine with `blue-onyx-prism user reset-password <name>`"
        ),
    };
    let mut text = format!(
        "Setup code: {}\n\nIt creates the first account of the web UI, once: enter it there \
         with a username and a password (15 to 128 characters). On this machine, \
         `blue-onyx-prism user add <name>` does the same without it.\n\n",
        code.grouped()
    );
    match crate::tls::fingerprint_report(config) {
        Ok(report) => text.push_str(&report),
        Err(e) => text.push_str(&format!(
            "The HTTPS certificate could not be read ({e:#}); `blue-onyx-prism \
             tls-fingerprint` shows it once that is fixed.\n"
        )),
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(std::path::PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn config() -> (Scratch, Config) {
        let d = std::env::temp_dir().join(format!("bo_usercmd_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        let config = Config {
            download_dir: Some(d.clone()),
            ..Default::default()
        };
        (Scratch(d), config)
    }

    fn run(config: &Config, cmd: UserCommand, stdin: &str) -> Result<String> {
        let mut out = Vec::new();
        let mut pw = Lines(std::io::Cursor::new(stdin.as_bytes().to_vec()));
        user(config, &cmd, &mut pw, &mut out)?;
        Ok(String::from_utf8(out).unwrap())
    }

    const PW: &str = "correct horse battery staple violet";

    #[test]
    fn stdin_lines_lose_only_their_line_ending() {
        let mut r = std::io::Cursor::new(b"  two spaces  \r\nnext\nlast".to_vec());
        assert_eq!(read_line_password(&mut r).unwrap(), "  two spaces  ");
        assert_eq!(read_line_password(&mut r).unwrap(), "next");
        assert_eq!(read_line_password(&mut r).unwrap(), "last");
        assert!(read_line_password(&mut r).is_err(), "end of input");
    }

    #[test]
    fn user_commands_through_stdin() {
        let (_d, config) = config();
        let said = run(&config, UserCommand::List, "").unwrap();
        assert!(said.contains("No accounts yet"), "{said}");
        let said = run(
            &config,
            UserCommand::Add {
                name: "Alice".into(),
            },
            &format!("{PW}\n"),
        )
        .unwrap();
        assert!(said.contains("Added the account alice"), "{said}");
        assert!(!said.contains(PW));
        let store = Store::open(&config.data_root());
        let users = store.load().unwrap().unwrap();
        let alice = users.get("alice").unwrap();
        let p = password::acquire(Duration::from_secs(60)).unwrap();
        assert!(password::verify(&alice.hash, PW, &p).unwrap());
        drop(p);

        // A taken name is refused before any password is read; a bad password is refused.
        let e = run(
            &config,
            UserCommand::Add {
                name: "alice".into(),
            },
            "",
        )
        .unwrap_err();
        assert!(format!("{e:#}").contains("exists already"), "{e:#}");
        let e = run(&config, UserCommand::Add { name: "bob".into() }, "short\n").unwrap_err();
        let msg = format!("{e:#}");
        assert!(
            msg.contains("at least 15") && !msg.contains("short"),
            "{msg}"
        );
        assert_eq!(msg.matches("at least 15").count(), 1, "said once: {msg}");
        let e = run(&config, UserCommand::Add { name: "bob".into() }, "").unwrap_err();
        assert!(format!("{e:#}").contains("no password on stdin"), "{e:#}");
        run(
            &config,
            UserCommand::Add { name: "bob".into() },
            "another fine passphrase here\r\n",
        )
        .unwrap();

        let list = run(&config, UserCommand::List, "").unwrap();
        assert!(list.contains("alice") && list.contains("bob"), "{list}");
        assert!(!list.contains("$argon2"), "never hashes: {list}");

        run(&config, UserCommand::Unlock { name: "bob".into() }, "").unwrap();
        let before = store
            .load()
            .unwrap()
            .unwrap()
            .get("bob")
            .unwrap()
            .generation;
        let said = run(
            &config,
            UserCommand::ResetPassword { name: "BOB".into() },
            "a brand new passphrase for him\n",
        )
        .unwrap();
        assert!(said.contains("sessions end"), "{said}");
        let bob = store.load().unwrap().unwrap().get("bob").unwrap().clone();
        assert_eq!(bob.generation, before + 1);
        let e = run(
            &config,
            UserCommand::ResetPassword {
                name: "carol".into(),
            },
            "",
        )
        .unwrap_err();
        assert!(
            format!("{e:#}").contains("no account has that name"),
            "{e:#}"
        );

        run(&config, UserCommand::Remove { name: "bob".into() }, "").unwrap();
        let e = run(
            &config,
            UserCommand::Remove {
                name: "alice".into(),
            },
            "",
        )
        .unwrap_err();
        assert!(format!("{e:#}").contains("last account"), "{e:#}");
    }

    #[test]
    fn setup_code_prints_a_code_until_an_account_exists() {
        let (_d, config) = config();
        let text = setup_code(&config).unwrap();
        let code = setup::read_code(&config.data_root()).unwrap().unwrap();
        assert!(text.contains(&code.grouped()), "{text}");
        assert!(text.contains("SHA-256 fingerprint"), "{text}");
        assert!(text.contains(&format!("https://localhost:{}/", config.https_port)));
        // Asked again: the same code.
        assert!(setup_code(&config).unwrap().contains(&code.grouped()));
        run(
            &config,
            UserCommand::Add {
                name: "alice".into(),
            },
            &format!("{PW}\n"),
        )
        .unwrap();
        let e = format!("{:#}", setup_code(&config).unwrap_err());
        assert!(e.contains("accounts exist already"), "{e}");
    }
}
