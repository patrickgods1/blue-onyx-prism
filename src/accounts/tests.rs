//! The accounts store: fail-closed loading, the account operations and the lock.

use super::*;
use std::time::Duration;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch() -> Scratch {
    let d = std::env::temp_dir().join(format!("bo_accounts_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    Scratch(d)
}

const PW: &str = "correct horse battery staple violet";
const PW2: &str = "another fine passphrase here";

fn permit() -> password::Permit<'static> {
    password::acquire(Duration::from_secs(60)).unwrap()
}

fn refused(e: &anyhow::Error) -> Option<&Refused> {
    e.downcast_ref::<Refused>()
}

/// A store with alice (and `more`) in it.
fn with_users(d: &Scratch, more: &[&str]) -> Store {
    let s = Store::open(&d.0);
    let p = permit();
    s.add("alice", PW, &p, 100).unwrap();
    for n in more {
        s.add(n, PW2, &p, 100).unwrap();
    }
    s
}

fn temp_files(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect()
}

#[test]
fn names_are_lowercased_and_checked() {
    assert_eq!(normalize_name("Alice").unwrap(), "alice");
    assert_eq!(normalize_name("a").unwrap(), "a");
    assert_eq!(normalize_name("a.b_c-9").unwrap(), "a.b_c-9");
    assert_eq!(normalize_name(&"x".repeat(32)).unwrap(), "x".repeat(32));
    for bad in [
        "",
        ".hidden",
        "-dash",
        "with space",
        "a/b",
        "\u{212A}elvin",
        "caf\u{e9}",
        &"x".repeat(33),
    ] {
        assert_eq!(normalize_name(bad), Err(Refused::BadName), "{bad:?}");
    }
    assert!(!Refused::BadName.to_string().contains("hidden"));
}

#[test]
fn a_missing_file_is_none_and_nothing_else_is() {
    let d = scratch();
    let s = Store::open(&d.0);
    assert_eq!(s.load().unwrap(), None);
    let auth = crate::private_dir::ensure(&d.0).unwrap();
    let path = s.path();
    let good = {
        let users = Users {
            format: FORMAT,
            users: vec![],
        };
        serde_json::to_string(&users).unwrap()
    };
    std::fs::write(&path, &good).unwrap();
    assert_eq!(
        s.load().unwrap().unwrap().users.len(),
        0,
        "an empty list is no setup mode"
    );
    let hash = "$argon2id$v=19$m=64,t=1,p=1$c2FsdHNhbHRzYWx0c2FsdA$aGFzaA";
    for (bad, says) in [
        ("garbage".to_string(), "damaged"),
        ("{\"format\":1,\"users\":[".to_string(), "cut short"),
        (String::new(), "damaged"),
        ("{\"format\":2,\"users\":[]}".to_string(), "newer version"),
        ("{\"format\":0,\"users\":[]}".to_string(), "damaged"),
        (
            format!(
                "{{\"format\":1,\"users\":[{{\"name\":\"Bad Name\",\"hash\":\"{hash}\",\
                 \"created\":1,\"password_changed\":1,\"generation\":1}}]}}"
            ),
            "invalid name",
        ),
        (
            format!(
                "{{\"format\":1,\"users\":[{{\"name\":\"a\",\"hash\":\"{hash}\",\"created\":1,\
                 \"password_changed\":1,\"generation\":1}},{{\"name\":\"a\",\"hash\":\"{hash}\",\
                 \"created\":1,\"password_changed\":1,\"generation\":\"{hash}\"}}]}}"
            ),
            "damaged",
        ),
    ] {
        std::fs::write(&path, &bad).unwrap();
        let e = format!("{:#}", s.load().unwrap_err());
        assert!(e.contains(says), "{bad:?}: {e}");
        assert!(!e.contains(hash), "never quotes the file: {e}");
        // Every change refuses too, and the file stays as it was.
        let p = permit();
        assert!(s.add("bob", PW2, &p, 1).is_err());
        assert!(s.unlock("a").is_err());
        assert!(s.record_failure("a", 1).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), bad);
        assert!(super::setup::ensure_code(&d.0).is_err());
        assert!(!auth.join(super::setup::FILE_NAME).exists());
    }
    // A directory in its place, what an unreadable file looks like on every OS.
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert!(s.load().is_err());
}

#[test]
fn add_and_remove_never_the_last_account() {
    let d = scratch();
    let s = Store::open(&d.0);
    let p = permit();
    assert_eq!(
        refused(&s.remove("alice").unwrap_err()),
        Some(&Refused::NoAccounts)
    );
    let alice = s.add("Alice", PW, &p, 100).unwrap();
    assert_eq!(alice.name, "alice");
    assert_eq!(alice.generation, 1);
    assert_eq!((alice.created, alice.password_changed), (100, 100));
    assert!(
        !format!("{alice:?}").contains("argon2"),
        "Debug hides the hash"
    );
    assert_eq!(
        refused(&s.add("ALICE", PW2, &p, 1).unwrap_err()),
        Some(&Refused::NameTaken)
    );
    assert!(matches!(
        refused(&s.add("bob", "too short", &p, 1).unwrap_err()),
        Some(Refused::Password(password::Refusal::TooShort(9)))
    ));
    assert!(matches!(
        refused(
            &s.add("bobby", "bobby likes long passwords", &p, 1)
                .unwrap_err()
        ),
        Some(Refused::Password(password::Refusal::ContainsName))
    ));
    s.add("bob", PW2, &p, 200).unwrap();
    assert_eq!(
        refused(&s.remove("carol").unwrap_err()),
        Some(&Refused::UnknownName)
    );
    s.remove("BOB").unwrap();
    assert_eq!(
        refused(&s.remove("alice").unwrap_err()),
        Some(&Refused::LastAccount)
    );
    let users = s.load().unwrap().unwrap();
    assert_eq!(users.users.len(), 1);
    assert!(password::verify(&users.users[0].hash, PW, &p).unwrap());
    assert!(temp_files(&s.dir()).is_empty());
}

#[test]
fn set_password_bumps_the_generation_and_clears_failures() {
    let d = scratch();
    let s = with_users(&d, &[]);
    for t in 0..lockout::DISABLE_AT {
        s.record_failure("alice", 1_000 + u64::from(t)).unwrap();
    }
    let before = s.load().unwrap().unwrap().get("alice").unwrap().clone();
    assert!(before.disabled);
    assert_eq!(before.failures, lockout::DISABLE_AT);
    let p = permit();
    let after = s.set_password("alice", PW2, &p, 5_000).unwrap();
    assert_eq!(after.generation, before.generation + 1);
    assert_eq!(after.password_changed, 5_000);
    assert_eq!(after.created, before.created);
    assert_eq!(after.lockout(), lockout::State::default());
    assert!(password::verify(&after.hash, PW2, &p).unwrap());
    assert!(!password::verify(&after.hash, PW, &p).unwrap());
    // The policy holds here too, and an unknown name is refused before hashing.
    assert!(matches!(
        refused(
            &s.set_password("alice", "1234567890123456", &p, 1)
                .unwrap_err()
        ),
        Some(Refused::Password(password::Refusal::Common))
    ));
    assert_eq!(
        refused(&s.set_password("ghost", PW, &p, 1).unwrap_err()),
        Some(&Refused::UnknownName)
    );
    assert_eq!(s.bump_generation("alice").unwrap(), after.generation + 1);
}

#[test]
fn unlock_clears_failures_but_keeps_the_generation() {
    let d = scratch();
    let s = with_users(&d, &[]);
    for t in 0..12 {
        s.record_failure("alice", 50 + t).unwrap();
    }
    let u = s.load().unwrap().unwrap().get("alice").unwrap().clone();
    assert_eq!(u.failures, 12);
    assert_eq!(u.last_failure, Some(61));
    assert_eq!(
        lockout::decide(&u.lockout(), 61),
        lockout::Decision::Wait(4)
    );
    let unlocked = s.unlock("alice").unwrap();
    assert_eq!(unlocked.lockout(), lockout::State::default());
    assert_eq!(unlocked.generation, u.generation);
    assert_eq!(unlocked.hash, u.hash);
    // Failures of unknown names are not stored.
    assert_eq!(s.record_failure("ghost", 1).unwrap(), None);
    assert_eq!(s.record_failure("Bad Name", 1).unwrap(), None);
    s.record_failure("alice", 1).unwrap();
    s.record_success("alice").unwrap();
    assert_eq!(
        s.load().unwrap().unwrap().get("alice").unwrap().lockout(),
        lockout::State::default()
    );
}

#[test]
fn a_success_writes_only_when_there_was_something_to_clear() {
    let d = scratch();
    let s = with_users(&d, &[]);
    let path = s.path();
    let stamp = |p: &Path| {
        let m = std::fs::metadata(p).unwrap();
        (m.modified().unwrap(), m.len())
    };
    let before = std::fs::read_to_string(&path).unwrap();
    let st = stamp(&path);
    s.record_success("alice").unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    assert_eq!(stamp(&path), st);
}

/// Several threads and two `Store` instances (as two processes) change one file at once:
/// nothing is lost, and no temporary file stays.
#[test]
fn concurrent_writers_lose_no_update() {
    let d = scratch();
    let s = with_users(&d, &[]);
    let other = Store::open(&d.0);
    const THREADS: usize = 6;
    const EACH: u32 = 5;
    std::thread::scope(|scope| {
        for t in 0..THREADS {
            let store = if t % 2 == 0 { s.clone() } else { other.clone() };
            scope.spawn(move || {
                // Its own semaphore, so the threads really overlap.
                let sem = password::Semaphore::new(1);
                let p = sem.acquire(Duration::ZERO).unwrap();
                store.add(&format!("user{t}"), PW2, &p, 1).unwrap();
                for i in 0..EACH {
                    store.record_failure("alice", u64::from(i)).unwrap();
                }
            });
        }
    });
    let users = s.load().unwrap().unwrap();
    assert_eq!(users.users.len(), THREADS + 1);
    let expected = u32::try_from(THREADS).unwrap() * EACH;
    assert_eq!(users.get("alice").unwrap().failures, expected);
    assert!(temp_files(&s.dir()).is_empty());
}

#[test]
fn the_first_add_removes_the_setup_code() {
    let d = scratch();
    let setup::Ensured::Created(_) = setup::ensure_code(&d.0).unwrap() else {
        panic!("no code");
    };
    with_users(&d, &[]);
    assert_eq!(setup::read_code(&d.0).unwrap(), None);
}

#[test]
fn the_cache_reads_again_only_after_a_change() {
    let d = scratch();
    let cache = Cached::new(&d.0);
    assert_eq!(*cache.get(), Snapshot::Missing);
    let s = with_users(&d, &[]);
    let first = cache.get();
    let Snapshot::Users(u) = &*first else {
        panic!("{first:?}");
    };
    assert_eq!(u.users.len(), 1);
    assert!(
        Arc::ptr_eq(&first, &cache.get()),
        "unchanged: the same snapshot"
    );
    // A change of the same length right away (failures 0 -> 1 is one more byte; 1 -> 2 is not).
    s.record_failure("alice", 7).unwrap();
    let one = cache.get();
    s.record_failure("alice", 7).unwrap();
    let two = cache.get();
    let Snapshot::Users(u) = &*two else {
        panic!("{two:?}");
    };
    assert_eq!(u.get("alice").unwrap().failures, 2, "{one:?}");
    std::fs::write(s.path(), "garbage").unwrap();
    assert!(matches!(&*cache.get(), Snapshot::Broken(m) if m.contains("damaged")));
    std::fs::remove_file(s.path()).unwrap();
    assert_eq!(*cache.get(), Snapshot::Missing);
    cache.invalidate();
    assert_eq!(*cache.get(), Snapshot::Missing);
}

#[test]
fn times_print_as_rfc3339() {
    assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
    assert_eq!(rfc3339(1_762_560_000), "2025-11-08T00:00:00Z");
    assert!(now() > 1_762_560_000);
}

#[cfg(unix)]
#[test]
fn files_are_private_on_unix() {
    use std::os::unix::fs::PermissionsExt;
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    let d = scratch();
    setup::ensure_code(&d.0).unwrap();
    let s = with_users(&d, &["bob"]);
    assert_eq!(mode(&s.dir()), 0o700);
    assert_eq!(mode(&s.path()), 0o600);
    setup::clear_code(&d.0).unwrap();
    let d2 = scratch();
    setup::ensure_code(&d2.0).unwrap();
    assert_eq!(
        mode(&crate::private_dir::path(&d2.0).join(setup::FILE_NAME)),
        0o600
    );
}
