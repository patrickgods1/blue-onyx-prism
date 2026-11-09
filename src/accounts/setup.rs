//! The one-time setup code that creates the first account: `<data root>/auth/setup-code`.
//!
//! While there is no `users.json`, whoever can read this file (the service's account and
//! administrators, through `blue-onyx-prism setup-code` on the server) can create the first
//! account with it; afterwards it is gone. The code is 20 characters of the RFC 4648 base32
//! alphabet (`A-Z`, `2-7`, no padding): 100 bits from the OS ([`crate::secret`]). It is printed
//! in groups of four (`ABCD-EFGH-...`), and a typed code is compared ignoring case, spaces and
//! dashes, in time that depends on neither value (both are hashed with SHA-256 and the digests
//! compared with no early exit, like [`crate::admin_token::AdminToken::matches`]).
//!
//! It is made only while no `users.json` exists ([`ensure_code`]), never when that file cannot
//! be read (fail closed: a damaged accounts file must not reopen setup). [`create_first`]
//! re-checks both under `auth/.lock`, so of concurrent attempts exactly one creates an account.
//! [`SetupCode`] has no `Display` or `Serialize` and its `Debug` hides the value.

use super::{Refused, Store, User, normalize_name, password};
use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::Path;

/// The code file's name in `auth/`.
pub const FILE_NAME: &str = "setup-code";
/// Characters of a code.
pub const CODE_LEN: usize = 20;
/// The RFC 4648 base32 alphabet.
const ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

/// A well-formed setup code (see the module docs).
#[derive(Clone, PartialEq, Eq)]
pub struct SetupCode(String);

impl std::fmt::Debug for SetupCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SetupCode(..)")
    }
}

impl SetupCode {
    /// The code as stored (20 characters), for `setup-code` to print. Nothing else should.
    pub fn reveal(&self) -> &str {
        &self.0
    }

    /// The code in groups of four with dashes, easier to read out and type.
    pub fn grouped(&self) -> String {
        self.0
            .as_bytes()
            .chunks(4)
            .map(|c| String::from_utf8_lossy(c).into_owned())
            .collect::<Vec<_>>()
            .join("-")
    }

    /// Whether `presented` is this code, ignoring case, whitespace and dashes; constant time in
    /// both values.
    pub fn matches(&self, presented: &str) -> bool {
        use sha2::{Digest, Sha256};
        let typed: String = presented
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '-')
            .map(|c| c.to_ascii_uppercase())
            .collect();
        let ours = Sha256::digest(self.0.as_bytes());
        let theirs = Sha256::digest(typed.as_bytes());
        let diff = ours
            .iter()
            .zip(theirs.iter())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b));
        std::hint::black_box(diff) == 0
    }
}

/// `bytes` in RFC 4648 base32 without padding, cut to `len` characters.
fn base32(bytes: &[u8], len: usize) -> String {
    let mut out = String::with_capacity(len);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &b in bytes {
        acc = (acc << 8) | u32::from(b);
        bits += 8;
        while bits >= 5 && out.len() < len {
            bits -= 5;
            let i = usize::try_from((acc >> bits) & 31).unwrap_or(0);
            out.push(char::from(*ALPHABET.get(i).unwrap_or(&b'A')));
        }
        acc &= (1 << bits) - 1;
    }
    out
}

/// A new random code (100 bits: 13 bytes, of which the first 100 bits are used).
pub fn generate() -> SetupCode {
    SetupCode(base32(&crate::secret::random_bytes(13), CODE_LEN))
}

/// Whether `text` is a stored code: exactly [`CODE_LEN`] characters of the alphabet.
fn well_formed(text: &str) -> bool {
    text.len() == CODE_LEN && text.bytes().all(|b| ALPHABET.contains(&b))
}

/// The code in `auth`: `None` when there is no file. A file that is not a code is an error
/// (never quoted).
fn read_in(auth: &Path) -> Result<Option<SetupCode>> {
    let path = auth.join(FILE_NAME);
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => bail!(
            "could not read the setup code in {} ({e}); run this as the service's account",
            path.display()
        ),
    };
    let mut bytes = Vec::new();
    file.take(256)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading the setup code in {}", path.display()))?;
    let text = String::from_utf8_lossy(bytes.trim_ascii()).into_owned();
    if !well_formed(&text) {
        bail!(
            "{} does not hold a setup code (it was probably edited); delete it and run \
             `blue-onyx-prism setup-code` for a new one",
            path.display()
        );
    }
    Ok(Some(SetupCode(text)))
}

/// Delete the code file in `auth` (the caller holds the lock); true when there was one.
pub(crate) fn remove_code_file(auth: &Path) -> Result<bool> {
    let path = auth.join(FILE_NAME);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e).with_context(|| {
            format!(
                "deleting the setup code {}; delete it by hand (accounts exist, so it is of no \
                 use any more)",
                path.display()
            )
        }),
    }
}

/// What [`ensure_code`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ensured {
    /// Made now.
    Created(SetupCode),
    /// The code made earlier, still valid.
    Existing(SetupCode),
    /// Accounts exist: there is no code (a leftover file was deleted).
    AccountsExist,
}

/// The setup code of `data_root`, made when there is none and no accounts exist. A
/// `users.json` that cannot be read is an error and no code is made.
pub fn ensure_code(data_root: &Path) -> Result<Ensured> {
    Store::open(data_root).locked(|auth, current| {
        if current.is_some() {
            remove_code_file(auth)?;
            return Ok((None, Ensured::AccountsExist));
        }
        if let Some(code) = read_in(auth)? {
            return Ok((None, Ensured::Existing(code)));
        }
        let code = generate();
        crate::private_dir::write_file(auth, FILE_NAME, &format!("{}\n", code.reveal()))?;
        Ok((None, Ensured::Created(code)))
    })
}

/// The setup code of `data_root`, `None` when there is none.
pub fn read_code(data_root: &Path) -> Result<Option<SetupCode>> {
    read_in(&crate::private_dir::path(data_root))
}

/// Delete the setup code of `data_root` (under the lock); true when there was one.
pub fn clear_code(data_root: &Path) -> Result<bool> {
    Store::open(data_root).locked(|auth, _| Ok((None, remove_code_file(auth)?)))
}

/// Why [`create_first`] made no account.
#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    /// Accounts exist already: sign in instead.
    #[error("an account exists already, so setup is closed: sign in instead")]
    AccountsExist,
    /// There is no setup code (none was made, or it was used).
    #[error("there is no setup code: run `blue-onyx-prism setup-code` on the server to get one")]
    NoCode,
    /// The code does not match.
    #[error(
        "that is not the setup code: run `blue-onyx-prism setup-code` on the server and type \
         what it prints"
    )]
    WrongCode,
    /// The name or password was refused.
    #[error(transparent)]
    Refused(#[from] Refused),
    /// Something else went wrong (the message says what).
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Create the first account with the setup code: the code is checked, the password checked and
/// hashed (with `permit`, outside the lock), then under `auth/.lock` it is checked again that no
/// accounts exist and that the code matches, the account is written and the code deleted. Of
/// concurrent calls exactly one succeeds.
pub fn create_first(
    data_root: &Path,
    code: &str,
    name: &str,
    new_password: &str,
    permit: &password::Permit<'_>,
    now: u64,
) -> Result<User, SetupError> {
    let store = Store::open(data_root);
    // Cheap checks first, so a wrong code costs no hashing.
    let gate = |current: &Option<super::Users>, stored: Option<SetupCode>| {
        if current.is_some() {
            return Err(SetupError::AccountsExist);
        }
        match stored {
            None => Err(SetupError::NoCode),
            Some(c) if !c.matches(code) => Err(SetupError::WrongCode),
            Some(_) => Ok(()),
        }
    };
    gate(&store.load()?, read_code(data_root)?)?;
    let name = normalize_name(name)?;
    password::check(new_password, &name).map_err(Refused::from)?;
    let hash = password::hash(new_password, permit)?;
    store.locked(|auth, current| {
        if let Err(e) = gate(&current, read_in(auth)?) {
            return Ok((None, Err(e)));
        }
        let user = User {
            name,
            hash,
            created: now,
            password_changed: now,
            generation: 1,
            failures: 0,
            last_failure: None,
            disabled: false,
        };
        let users = super::Users {
            format: super::FORMAT,
            users: vec![user.clone()],
        };
        // The account first: a failure after it leaves a code nobody can use, never setup
        // open with an account in place.
        super::write_users(auth, &users)?;
        remove_code_file(auth)?;
        Ok((None, Ok(user)))
    })?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct Scratch(std::path::PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch() -> Scratch {
        let d = std::env::temp_dir().join(format!("bo_setup_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }

    const PW: &str = "correct horse battery staple violet";

    fn permit() -> password::Permit<'static> {
        password::acquire(Duration::from_secs(60)).unwrap()
    }

    #[test]
    fn codes_are_twenty_base32_characters() {
        for _ in 0..50 {
            let c = generate();
            assert_eq!(c.reveal().len(), CODE_LEN);
            assert!(well_formed(c.reveal()), "{}", c.reveal());
            let g = c.grouped();
            assert_eq!(g.len(), CODE_LEN + 4);
            assert_eq!(g.matches('-').count(), 4);
        }
        assert_ne!(generate(), generate());
        // RFC 4648 test vectors, whole 5-bit groups only (a code never needs padding):
        // "fooba" -> MZXW6YTB, "foobar" -> MZXW6YTBOI====== (its first 9 characters).
        assert_eq!(base32(b"fooba", 8), "MZXW6YTB");
        assert_eq!(base32(b"foobar", 9), "MZXW6YTBO");
        assert_eq!(base32(&[0xff; 13], CODE_LEN), "7".repeat(CODE_LEN));
        let c = generate();
        assert!(!format!("{c:?}").contains(c.reveal()));
    }

    #[test]
    fn matches_ignores_case_spaces_and_dashes_only() {
        let c = SetupCode("ABCDEFGHIJKLMNOPQRST".into());
        assert!(c.matches("ABCDEFGHIJKLMNOPQRST"));
        assert!(c.matches("abcdefghijklmnopqrst"));
        assert!(c.matches("ABCD-EFGH-IJKL-MNOP-QRST"));
        assert!(c.matches(" abcd efgh\tijkl mnop qrst \n"));
        assert!(!c.matches("ABCDEFGHIJKLMNOPQRS"));
        assert!(!c.matches("ABCDEFGHIJKLMNOPQRSTU"));
        assert!(!c.matches("ABCDEFGHIJKLMNOPQRS_T"));
        assert!(!c.matches(""));
    }

    #[test]
    fn a_code_is_made_only_without_accounts() {
        let d = scratch();
        let first = match ensure_code(&d.0).unwrap() {
            Ensured::Created(c) => c,
            other => panic!("{other:?}"),
        };
        assert_eq!(read_code(&d.0).unwrap(), Some(first.clone()));
        assert_eq!(ensure_code(&d.0).unwrap(), Ensured::Existing(first));
        assert!(clear_code(&d.0).unwrap());
        assert!(!clear_code(&d.0).unwrap());
        assert_eq!(read_code(&d.0).unwrap(), None);
        // With an account there is none, and a leftover goes.
        assert!(matches!(ensure_code(&d.0).unwrap(), Ensured::Created(_)));
        Store::open(&d.0).add("alice", PW, &permit(), 1).unwrap();
        assert_eq!(read_code(&d.0).unwrap(), None, "the first add removes it");
        std::fs::write(d.0.join("auth").join(FILE_NAME), "ABCDEFGHIJKLMNOPQRST").unwrap();
        assert_eq!(ensure_code(&d.0).unwrap(), Ensured::AccountsExist);
        assert_eq!(read_code(&d.0).unwrap(), None);
    }

    #[test]
    fn no_code_when_the_accounts_file_is_damaged() {
        let d = scratch();
        let auth = crate::private_dir::ensure(&d.0).unwrap();
        std::fs::write(auth.join(super::super::FILE_NAME), "{not json").unwrap();
        assert!(ensure_code(&d.0).is_err());
        assert!(!auth.join(FILE_NAME).exists());
        assert_eq!(
            std::fs::read_to_string(auth.join(super::super::FILE_NAME)).unwrap(),
            "{not json"
        );
        // A damaged code file is an error, never quoted.
        std::fs::remove_file(auth.join(super::super::FILE_NAME)).unwrap();
        std::fs::write(auth.join(FILE_NAME), "hello-there").unwrap();
        let e = format!("{:#}", read_code(&d.0).unwrap_err());
        assert!(!e.contains("hello-there"), "{e}");
        assert!(ensure_code(&d.0).is_err());
    }

    #[test]
    fn create_first_checks_the_code_and_closes_setup() {
        let d = scratch();
        let p = permit();
        assert!(matches!(
            create_first(&d.0, "x", "alice", PW, &p, 1),
            Err(SetupError::NoCode)
        ));
        let Ensured::Created(code) = ensure_code(&d.0).unwrap() else {
            panic!("no code");
        };
        assert!(matches!(
            create_first(&d.0, "AAAAAAAAAAAAAAAAAAAA", "alice", PW, &p, 1),
            Err(SetupError::WrongCode)
        ));
        assert!(matches!(
            create_first(&d.0, code.reveal(), "Bad Name", PW, &p, 1),
            Err(SetupError::Refused(Refused::BadName))
        ));
        assert!(matches!(
            create_first(&d.0, code.reveal(), "alice", "short", &p, 1),
            Err(SetupError::Refused(Refused::Password(_)))
        ));
        let typed = code.grouped().to_lowercase();
        let user = create_first(&d.0, &typed, "Alice", PW, &p, 7).unwrap();
        assert_eq!(user.name, "alice");
        assert_eq!(user.created, 7);
        assert_eq!(read_code(&d.0).unwrap(), None);
        assert!(matches!(
            create_first(&d.0, code.reveal(), "bob", PW, &p, 8),
            Err(SetupError::AccountsExist)
        ));
        let users = Store::open(&d.0).load().unwrap().unwrap();
        assert_eq!(users.users.len(), 1);
    }

    #[test]
    fn of_concurrent_setups_exactly_one_wins() {
        let d = scratch();
        let Ensured::Created(code) = ensure_code(&d.0).unwrap() else {
            panic!("no code");
        };
        let results: Vec<bool> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|i| {
                    let root = d.0.clone();
                    let code = code.reveal().to_string();
                    s.spawn(move || {
                        // A semaphore of its own per thread, so all of them hash at once.
                        let sem = password::Semaphore::new(1);
                        let p = sem.acquire(Duration::ZERO).unwrap();
                        create_first(&root, &code, &format!("user{i}"), PW, &p, 1).is_ok()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(results.iter().filter(|ok| **ok).count(), 1, "{results:?}");
        let users = Store::open(&d.0).load().unwrap().unwrap();
        assert_eq!(users.users.len(), 1);
        assert_eq!(read_code(&d.0).unwrap(), None);
        let left: Vec<String> = std::fs::read_dir(d.0.join("auth"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(left.is_empty(), "{left:?}");
    }
}
