//! Passwords of the web UI's accounts: the policy a new one must meet, and Argon2id hashing.
//!
//! **Policy** (NIST SP 800-63B-4, 3.1.1.2, for a single-factor password): the password is
//! NFC-normalized ([`nfc`]) and then counted in Unicode code points, 15 ..= 128 of them. There are
//! no composition rules (no "one digit, one symbol"). Refused: an entry of the common-password
//! list ([`super::blocklist`], any case), a password that contains the account's name (any case,
//! for names of 3 characters or more) or the product's name (`blueonyx`, `prism`, any case). The
//! messages say what is wrong and what to do; none repeats the password.
//!
//! **Hashing** (OWASP Password Storage cheat sheet): Argon2id v1.3, m = 19456 KiB, t = 2, p = 1,
//! a 16-byte salt from the OS (`getrandom`), a 32-byte output, stored as a PHC string
//! (`$argon2id$v=19$m=19456,t=2,p=1$<salt>$<hash>`). [`verify`] reads the parameters from the
//! stored string, so a later change of [`PRODUCTION`] keeps old hashes working. It refuses a
//! string that is not Argon2id v1.3 or whose cost is below the OWASP floor (m = 7168 KiB, t = 1),
//! or absurdly above it: `users.json` is ours, so such a string was edited in, and checking a
//! password against it would make guessing cheap (or, above the ceiling, let one sign-in
//! attempt exhaust the server's memory). Unit tests hash with [`TEST_PARAMS`] instead (cheap),
//! and the floor is lowered to them under `cfg(test)` only.
//!
//! **Concurrency:** each hash costs 19 MiB and tens of milliseconds of one core, so a burst of
//! sign-ins could exhaust the machine. Every [`hash`], [`verify`] and [`dummy_verify`] takes a
//! [`Permit`] of the process-wide semaphore ([`acquire`]: `max(1, min(4, physical cores / 2))`
//! permits); a caller that gets none within its wait answers "busy" (the web UI: 503 with
//! `Retry-After`). Taking it is the caller's job, so the type system makes sure it happens.

use anyhow::{Context, Result, anyhow, bail};
use argon2::password_hash::{PasswordHasher, PasswordVerifier, phc::PasswordHash};
use argon2::{Algorithm, Argon2, Params, Version};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Fewest code points a password has (after NFC).
pub const MIN_LEN: usize = 15;
/// Most code points a password has (after NFC).
pub const MAX_LEN: usize = 128;
/// A name shorter than this is not looked for inside the password (any password would contain
/// a one- or two-letter name somewhere by chance).
pub const NAME_MIN_CHECKED: usize = 3;
/// Names of the product a password must not contain (lowercase).
pub const PRODUCT_WORDS: [&str; 2] = ["blueonyx", "prism"];

/// Argon2id memory cost in KiB (OWASP: 19 MiB).
pub const M_COST: u32 = 19_456;
/// Argon2id passes (OWASP: 2).
pub const T_COST: u32 = 2;
/// Argon2id lanes (OWASP: 1).
pub const P_COST: u32 = 1;
/// Salt length in bytes.
pub const SALT_LEN: usize = 16;
/// Output length in bytes.
pub const OUTPUT_LEN: usize = 32;
/// The OWASP floor [`verify`] accepts: m = 7168 KiB with t = 5 is OWASP's cheapest
/// recommendation, and no recommendation goes below t = 1 at any memory.
pub const MIN_M_COST: u32 = 7_168;
/// See [`MIN_M_COST`].
pub const MIN_T_COST: u32 = 1;
/// The ceiling [`verify`] accepts (see the module docs): 256 MiB, 16 passes, 8 lanes.
pub const MAX_M_COST: u32 = 262_144;
/// See [`MAX_M_COST`].
pub const MAX_T_COST: u32 = 16;
/// See [`MAX_M_COST`].
pub const MAX_P_COST: u32 = 8;

/// The parameters new hashes use outside unit tests: (m KiB, t, p).
pub const PRODUCTION: (u32, u32, u32) = (M_COST, T_COST, P_COST);
/// What unit tests hash with (and what the floor drops to under `cfg(test)`): cheap.
pub const TEST_PARAMS: (u32, u32, u32) = (64, 1, 1);

/// The parameters [`hash`] uses.
const fn active_params() -> (u32, u32, u32) {
    if cfg!(test) { TEST_PARAMS } else { PRODUCTION }
}

/// The lowest (m, t) [`verify`] accepts.
const fn floor() -> (u32, u32) {
    if cfg!(test) {
        (TEST_PARAMS.0, TEST_PARAMS.1)
    } else {
        (MIN_M_COST, MIN_T_COST)
    }
}

/// `text` in Unicode Normalization Form C (composed): "é" typed as one code point or as "e" plus
/// a combining accent is the same password.
pub fn nfc(text: &str) -> String {
    icu_normalizer::ComposingNormalizerBorrowed::new_nfc()
        .normalize(text)
        .into_owned()
}

/// Code points of `password` after NFC: what the length rules count.
pub fn length(password: &str) -> usize {
    nfc(password).chars().count()
}

/// Why a new password was refused. The message (`Display`) never holds the password.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    #[error(
        "the password has {0} characters and needs at least {MIN_LEN}: use a passphrase of four \
         or more words"
    )]
    TooShort(usize),
    #[error("the password has {0} characters and may have at most {MAX_LEN}: use a shorter one")]
    TooLong(usize),
    #[error(
        "that password is on a list of commonly used passwords: choose another, for example a \
         passphrase of four or more random words"
    )]
    Common,
    #[error("the password contains the account's name: choose one without it")]
    ContainsName,
    #[error("the password contains the product's name (blueonyx or prism): choose one without it")]
    ContainsProduct,
}

/// Check a new password for the account `name` against the policy (see the module docs).
pub fn check(password: &str, name: &str) -> Result<(), Refusal> {
    let n = length(password);
    if n < MIN_LEN {
        return Err(Refusal::TooShort(n));
    }
    if n > MAX_LEN {
        return Err(Refusal::TooLong(n));
    }
    let folded = super::blocklist::fold(password);
    let name = name.to_lowercase();
    if name.chars().count() >= NAME_MIN_CHECKED && folded.contains(&name) {
        return Err(Refusal::ContainsName);
    }
    if PRODUCT_WORDS.iter().any(|w| folded.contains(w)) {
        return Err(Refusal::ContainsProduct);
    }
    if super::blocklist::contains(password) {
        return Err(Refusal::Common);
    }
    Ok(())
}

/// The Argon2id hasher of (m, t, p).
fn hasher(params: (u32, u32, u32)) -> Result<Argon2<'static>> {
    let (m, t, p) = params;
    let params = Params::new(m, t, p, Some(OUTPUT_LEN))
        .map_err(|e| anyhow!("invalid Argon2 parameters m={m}, t={t}, p={p}: {e}"))?;
    Ok(Argon2::new(Algorithm::Argon2id, Version::V0x13, params))
}

/// The PHC string of `password` (NFC first) with a fresh salt, at the parameters new hashes
/// use. The caller checked the policy ([`check`]) and holds a permit.
pub fn hash(password: &str, permit: &Permit<'_>) -> Result<String> {
    hash_with(password, active_params(), permit)
}

/// [`hash`] at `params` (m KiB, t, p).
pub fn hash_with(password: &str, params: (u32, u32, u32), _permit: &Permit<'_>) -> Result<String> {
    let salt = argon2::password_hash::try_generate_salt()
        .map_err(|e| anyhow!("the operating system gave no random salt ({e}); try again"))?;
    let hash = hasher(params)?
        .hash_password_with_salt(nfc(password).as_bytes(), &salt)
        .map_err(|e| anyhow!("hashing the password failed: {e}"))?;
    Ok(hash.to_string())
}

/// The stored string, parsed and checked (see the module docs): what [`verify`] may use.
fn stored(phc: &str) -> Result<PasswordHash> {
    let hash = PasswordHash::new(phc).map_err(|e| {
        anyhow!("the stored password hash is not a PHC string ({e}); reset the password")
    })?;
    if hash.algorithm.as_str() != Algorithm::Argon2id.as_str() {
        bail!(
            "the stored password hash uses {}, not argon2id; reset the password",
            hash.algorithm.as_str()
        );
    }
    if hash.version != Some(u32::from(Version::V0x13)) {
        bail!("the stored password hash is not Argon2 version 1.3; reset the password");
    }
    let params = Params::try_from(&hash)
        .map_err(|e| anyhow!("the stored password hash has bad parameters ({e}); reset it"))?;
    let (m, t, p) = (params.m_cost(), params.t_cost(), params.p_cost());
    let (min_m, min_t) = floor();
    if m < min_m || t < min_t {
        bail!(
            "the stored password hash is weaker than allowed (m={m}, t={t}; at least \
             m={min_m}, t={min_t}), so it is not used; reset the password"
        );
    }
    if m > MAX_M_COST || t > MAX_T_COST || p > MAX_P_COST {
        bail!(
            "the stored password hash asks for more than allowed (m={m}, t={t}, p={p}; at most \
             m={MAX_M_COST}, t={MAX_T_COST}, p={MAX_P_COST}), so it is not used; reset the \
             password"
        );
    }
    let salt_len = hash.salt.as_ref().map_or(0, |s| s.len());
    if salt_len < SALT_LEN {
        bail!("the stored password hash has a salt under {SALT_LEN} bytes; reset the password");
    }
    if hash.hash.is_none() {
        bail!("the stored password hash has no hash part; reset the password");
    }
    Ok(hash)
}

/// Whether `password` (NFC first) matches the stored PHC string `phc`, at the parameters in it.
/// `Err`: the stored string cannot be used (see the module docs); the caller logs it and refuses.
/// A password longer than [`MAX_LEN`] never matched any policy, so it is `false` without hashing.
pub fn verify(phc: &str, password: &str, _permit: &Permit<'_>) -> Result<bool> {
    let hash = stored(phc)?;
    if password.chars().count() > MAX_LEN.saturating_mul(4) || length(password) > MAX_LEN {
        return Ok(false);
    }
    match Argon2::default().verify_password(nfc(password).as_bytes(), &hash) {
        Ok(()) => Ok(true),
        Err(argon2::password_hash::Error::PasswordInvalid) => Ok(false),
        Err(e) => Err(anyhow!("checking the password failed: {e}")),
    }
}

/// The work of one [`verify`] for a name that has no account, so the answer takes as long as a
/// wrong password for a real one: a fixed hash of a random value, made once (at the parameters
/// new hashes use), checked against `password`. Always false.
pub fn dummy_verify(password: &str, permit: &Permit<'_>) -> bool {
    static DUMMY: OnceLock<Option<String>> = OnceLock::new();
    let dummy = DUMMY.get_or_init(|| hash(&crate::secret::random_hex(32), permit).ok());
    if let Some(phc) = dummy {
        let _ = std::hint::black_box(verify(phc, password, permit));
    }
    false
}

// ---------------------------------------------------------------------------------------------
// The hashing semaphore
// ---------------------------------------------------------------------------------------------

/// A counting semaphore (std only: a mutex and a condition variable).
#[derive(Debug)]
pub struct Semaphore {
    free: Mutex<usize>,
    freed: Condvar,
    total: usize,
}

/// One permit to hash, given back on drop.
#[derive(Debug)]
pub struct Permit<'a> {
    sem: &'a Semaphore,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut free = self.sem.free.lock().unwrap_or_else(|e| e.into_inner());
        *free = free.saturating_add(1).min(self.sem.total);
        self.sem.freed.notify_one();
    }
}

impl Semaphore {
    /// `total` permits (at least one).
    pub fn new(total: usize) -> Self {
        let total = total.max(1);
        Self {
            free: Mutex::new(total),
            freed: Condvar::new(),
            total,
        }
    }

    /// How many permits there are.
    pub fn total(&self) -> usize {
        self.total
    }

    /// A permit, waiting at most `timeout` (blocking); `None` when none came free in time.
    pub fn acquire(&self, timeout: Duration) -> Option<Permit<'_>> {
        let deadline = Instant::now().checked_add(timeout);
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        while *free == 0 {
            let left = match deadline {
                Some(d) => d.checked_duration_since(Instant::now())?,
                None => Duration::from_secs(3600),
            };
            if left.is_zero() {
                return None;
            }
            free = self
                .freed
                .wait_timeout(free, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        *free -= 1;
        Some(Permit { sem: self })
    }
}

/// Permits for `physical` cores: `max(1, min(4, physical / 2))`, so hashing never takes more
/// than half the machine (and at most four 19 MiB buffers at once).
pub fn permits_for(physical: usize) -> usize {
    (physical / 2).clamp(1, 4)
}

/// The process-wide hashing semaphore.
pub fn semaphore() -> &'static Semaphore {
    static SEM: OnceLock<Semaphore> = OnceLock::new();
    SEM.get_or_init(|| Semaphore::new(permits_for(num_cpus::get_physical())))
}

/// A permit of the process-wide semaphore, waiting at most `timeout`; `None`: busy, answer so.
pub fn acquire(timeout: Duration) -> Option<Permit<'static>> {
    semaphore().acquire(timeout)
}

/// A permit for a command-line tool, which may wait longer than a web request: an error after
/// `timeout` (the service is hashing on every permit; try again).
pub fn acquire_for_command(timeout: Duration) -> Result<Permit<'static>> {
    acquire(timeout).context(
        "every password-hashing slot stayed busy, so nothing was changed; try again in a moment",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn permit() -> Permit<'static> {
        acquire(Duration::from_secs(60)).unwrap()
    }

    #[test]
    fn nfc_makes_composed_and_decomposed_equal() {
        let composed = "caf\u{e9}";
        let decomposed = "cafe\u{301}";
        assert_ne!(composed, decomposed);
        assert_eq!(nfc(composed), nfc(decomposed));
        assert_eq!(length(composed), 4);
        assert_eq!(length(decomposed), 4);
        assert_eq!(decomposed.chars().count(), 5);
    }

    #[test]
    fn lengths_count_code_points() {
        // An emoji is one code point (4 UTF-8 bytes), a CJK character one (3 bytes).
        assert_eq!(length("\u{1F600}"), 1);
        assert_eq!(length("\u{732B}\u{72AC}"), 2);
        let fifteen_emoji = "\u{1F600}".repeat(15);
        assert_eq!(fifteen_emoji.len(), 60);
        assert_eq!(check(&fifteen_emoji, "x"), Ok(()));
        let cjk = "\u{732B}\u{72AC}\u{9CE5}\u{9B5A}\u{82B1}".repeat(3);
        assert_eq!(check(&cjk, "x"), Ok(()));
        assert_eq!(
            check(&"\u{732B}".repeat(14), "x"),
            Err(Refusal::TooShort(14))
        );
    }

    #[test]
    fn length_bounds() {
        let w = |n: usize| "kx7mq".repeat(n / 5) + &"z".repeat(n % 5);
        assert_eq!(check(&w(14), "bob"), Err(Refusal::TooShort(14)));
        assert_eq!(check(&w(15), "bob"), Ok(()));
        assert_eq!(check(&w(128), "bob"), Ok(()));
        assert_eq!(check(&w(129), "bob"), Err(Refusal::TooLong(129)));
        // A decomposed accent counts once.
        let p = format!("{}e\u{301}", w(14));
        assert_eq!(check(&p, "bob"), Ok(()));
    }

    #[test]
    fn refusals_name_the_problem_and_never_the_password() {
        let secret = "Tr0ub4dor-and-something";
        for (pw, name, want) in [
            ("1234567890123456", "alice", Refusal::Common),
            ("QWERTYUIOPASDFGHJKL", "alice", Refusal::Common),
            ("my name is ALICE and more", "alice", Refusal::ContainsName),
            (
                "this is my Prism password ok",
                "alice",
                Refusal::ContainsProduct,
            ),
            (
                "blueonyxrocks forever and ever",
                "x",
                Refusal::ContainsProduct,
            ),
            (
                "BlueOnyx rocks forever and ever",
                "x",
                Refusal::ContainsProduct,
            ),
        ] {
            assert_eq!(check(pw, name), Err(want), "{pw}");
        }
        // A two-letter name is not looked for.
        assert_eq!(check("my name is BO and more words", "bo"), Ok(()));
        for r in [
            Refusal::TooShort(3),
            Refusal::TooLong(200),
            Refusal::Common,
            Refusal::ContainsName,
            Refusal::ContainsProduct,
        ] {
            let m = r.to_string();
            assert!(!m.contains(secret), "{m}");
            assert!(m.contains(": "), "says what to do: {m}");
        }
        assert!(
            Refusal::TooShort(3)
                .to_string()
                .contains("four or more words")
        );
    }

    #[test]
    fn hashing_round_trip_with_test_params() {
        let p = permit();
        let phc = hash("correct horse battery staple violet", &p).unwrap();
        assert!(phc.starts_with("$argon2id$v=19$m=64,t=1,p=1$"), "{phc}");
        assert!(verify(&phc, "correct horse battery staple violet", &p).unwrap());
        assert!(!verify(&phc, "correct horse battery staple violeT", &p).unwrap());
        assert!(!verify(&phc, "", &p).unwrap());
        // NFC on both sides.
        let phc = hash("caf\u{e9} au lait every morning", &p).unwrap();
        assert!(verify(&phc, "cafe\u{301} au lait every morning", &p).unwrap());
        // Two hashes of one password differ (salt).
        let a = hash("correct horse battery staple violet", &p).unwrap();
        assert_ne!(a, hash("correct horse battery staple violet", &p).unwrap());
        // Far too long never matches and costs nothing.
        assert!(!verify(&phc, &"x".repeat(10_000), &p).unwrap());
    }

    /// The one production-parameter hash of the suite (with its verify).
    #[test]
    fn production_hashes_carry_argon2id_and_the_owasp_constants() {
        assert_eq!(PRODUCTION, (19_456, 2, 1));
        assert_eq!((MIN_M_COST, MIN_T_COST), (7_168, 1));
        assert_eq!(SALT_LEN, 16);
        let p = permit();
        let phc = hash_with("correct horse battery staple violet", PRODUCTION, &p).unwrap();
        assert!(phc.starts_with("$argon2id$v=19$m=19456,t=2,p=1$"), "{phc}");
        let parsed = PasswordHash::new(&phc).unwrap();
        assert_eq!(parsed.salt.unwrap().len(), SALT_LEN);
        assert_eq!(parsed.hash.unwrap().len(), OUTPUT_LEN);
        assert!(verify(&phc, "correct horse battery staple violet", &p).unwrap());
    }

    #[test]
    fn weak_odd_or_foreign_hashes_are_refused() {
        let p = permit();
        // Below the (test) floor: m = 32 < 64.
        let weak = hash_with("correct horse battery staple violet", (32, 1, 1), &p).unwrap();
        let e = format!(
            "{:#}",
            verify(&weak, "correct horse battery staple violet", &p).unwrap_err()
        );
        assert!(e.contains("weaker than allowed"), "{e}");
        // Above the ceiling.
        let good = hash("correct horse battery staple violet", &p).unwrap();
        let huge = good.replace("m=64,", "m=999999,");
        assert!(verify(&huge, "x", &p).is_err());
        let wide = good.replace("p=1$", "p=9$");
        assert!(verify(&wide, "x", &p).is_err());
        // Another algorithm or version, or no hash at all.
        assert!(verify(&good.replace("argon2id", "argon2i"), "x", &p).is_err());
        assert!(verify(&good.replace("v=19", "v=16"), "x", &p).is_err());
        assert!(verify("", "x", &p).is_err());
        assert!(verify("plain text", "x", &p).is_err());
        let no_hash = good.rsplit_once('$').unwrap().0.to_string();
        assert!(verify(&no_hash, "x", &p).is_err());
        // The floor outside tests is OWASP's.
        assert!(MIN_M_COST > TEST_PARAMS.0);
    }

    #[test]
    fn dummy_verify_is_false_and_repeatable() {
        let p = permit();
        assert!(!dummy_verify("correct horse battery staple violet", &p));
        assert!(!dummy_verify("", &p));
    }

    #[test]
    fn permits_are_bounded_and_time_out() {
        assert_eq!(permits_for(0), 1);
        assert_eq!(permits_for(1), 1);
        assert_eq!(permits_for(2), 1);
        assert_eq!(permits_for(6), 3);
        assert_eq!(permits_for(8), 4);
        assert_eq!(permits_for(64), 4);
        assert!((1..=4).contains(&semaphore().total()));

        let sem = Semaphore::new(2);
        let a = sem.acquire(Duration::ZERO).unwrap();
        let b = sem.acquire(Duration::ZERO).unwrap();
        let started = Instant::now();
        assert!(sem.acquire(Duration::from_millis(50)).is_none());
        assert!(started.elapsed() >= Duration::from_millis(50));
        drop(a);
        let c = sem.acquire(Duration::ZERO).unwrap();
        // A waiter is woken by a release.
        std::thread::scope(|s| {
            let waiter = s.spawn(|| sem.acquire(Duration::from_secs(10)).is_some());
            std::thread::sleep(Duration::from_millis(20));
            drop(b);
            assert!(waiter.join().unwrap());
        });
        drop(c);
        let all: Vec<_> = (0..2)
            .map(|_| sem.acquire(Duration::ZERO).unwrap())
            .collect();
        assert_eq!(all.len(), 2);
        assert!(sem.acquire(Duration::ZERO).is_none());
    }
}
