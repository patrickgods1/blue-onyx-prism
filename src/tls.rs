//! The HTTPS listener's certificate: a self-signed one this service generates and keeps, or the
//! config's own files (`tls_cert_path` / `tls_key_path`), behind a rustls certificate resolver
//! that picks up a changed file without a restart.
//!
//! # The generated certificate
//!
//! An ECDSA P-256 leaf, valid two years (within Apple's 825-day limit) from one day before it was
//! made (clock skew between machines), EKU serverAuth, not a CA, CN `Blue Onyx Prism
//! (<hostname>)`. Its names: `localhost`, `127.0.0.1`, `::1`, the hostname and
//! `<hostname>.local` when they are valid DNS names, and the addresses this machine had when it
//! was made ([`Identity::of_this_machine`]). Kept in `<data root>/auth/tls/{cert,key}.pem`
//! ([`crate::private_dir`]: 0700/0600 on Unix, a protected DACL on Windows).
//!
//! It is **kept stable**: browsers remember the exception the user granted per certificate, so a
//! new one means a new warning. It is made only when there is none; never again because an
//! address changed (the warning names the address, not this certificate); again only when fewer
//! than [`RENEW_BEFORE`] of its validity are left (a warning from [`WARN_BEFORE`] on), or by
//! `blue-onyx-prism tls-regenerate`.
//!
//! A pair that cannot be read or parsed at start (a damaged file, a key that does not match the
//! certificate) is replaced by a new one, with a warning. That is safe for a self-signed
//! certificate: nobody trusts it through a CA or a pin that a new one would break; what trusts it
//! is the exception each browser stored, and a new certificate costs one more warning there. The
//! alternative, an HTTPS listener that stays down until someone deletes the files, is worse.
//! Changes found while running are not replaced: the file may be mid-write by `tls-regenerate`,
//! and the certificate in memory keeps serving until a good pair is on disk.
//!
//! # Own files
//!
//! Used as given (the chain allowed, leaf first; the key as PKCS#8, PKCS#1 or SEC1), never
//! written or regenerated. A pair that does not load is an error: the HTTPS listener stays down
//! with a message (HTTP keeps working) and the next attempt reads the files again. Expiry within
//! [`RENEW_BEFORE`] is a warning, once a day.
//!
//! # Reload
//!
//! [`Resolver`] holds the current [`CertifiedKey`] in an `ArcSwap`. On a handshake, at most once
//! every [`CHECK_EVERY`], it stats the files (cheap, on the handshake's thread); a change of
//! modification time or length, or a renewal that is due, is handled on a thread of its own, and
//! the handshakes meanwhile get the certificate in memory. So `tls-regenerate` from the CLI
//! applies to the running service without a restart.
//!
//! Crypto: rustls with the aws-lc-rs provider, passed explicitly ([`provider`]); the process-wide
//! default is never relied on. TLS 1.2 and 1.3 (rustls' defaults), ALPN `http/1.1` only, no
//! client certificates.

use crate::config::Config;
use anyhow::{Context, Result, bail};
use arc_swap::ArcSwap;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::sign::CertifiedKey;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tracing::{info, warn};

/// Folder of the generated pair under `<data root>/auth`.
pub const DIR_NAME: &str = "tls";
/// The generated certificate's file.
pub const CERT_FILE: &str = "cert.pem";
/// The generated key's file.
pub const KEY_FILE: &str = "key.pem";

const DAY: Duration = Duration::from_secs(24 * 60 * 60);
/// Validity of a generated certificate: two years (Apple accepts at most 825 days).
pub const VALIDITY: Duration = Duration::from_secs(730 * 24 * 60 * 60);
/// A generated certificate starts this long before it was made, for clients whose clock is
/// behind.
pub const BACKDATE: Duration = DAY;
/// A generated certificate with less validity than this left is replaced.
pub const RENEW_BEFORE: Duration = Duration::from_secs(30 * 24 * 60 * 60);
/// From this much validity left a generated certificate's renewal is announced in the log.
pub const WARN_BEFORE: Duration = Duration::from_secs(60 * 24 * 60 * 60);
/// Most often a handshake checks the files.
pub const CHECK_EVERY: Duration = Duration::from_secs(10);

/// The crypto provider of every TLS config here: aws-lc-rs, the one rustls is built with.
pub fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::aws_lc_rs::default_provider())
}

/// Where the HTTPS certificate comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Generated and kept under `<data_root>/auth/tls/`.
    Generated { data_root: PathBuf },
    /// The config's `tls_cert_path` / `tls_key_path`, resolved.
    Files { cert: PathBuf, key: PathBuf },
}

impl Source {
    /// The source `config` names: its own files (relative ones under the data root) when both
    /// are set, else the generated pair of its data root.
    pub fn of(config: &Config) -> Self {
        match (&config.tls_cert_path, &config.tls_key_path) {
            (Some(cert), Some(key)) => Self::Files {
                cert: config.data_path(cert),
                key: config.data_path(key),
            },
            _ => Self::Generated {
                data_root: config.data_root(),
            },
        }
    }

    /// The certificate and key files.
    pub fn paths(&self) -> (PathBuf, PathBuf) {
        match self {
            Self::Generated { data_root } => {
                let dir = generated_dir(data_root);
                (dir.join(CERT_FILE), dir.join(KEY_FILE))
            }
            Self::Files { cert, key } => (cert.clone(), key.clone()),
        }
    }

    fn generated(&self) -> bool {
        matches!(self, Self::Generated { .. })
    }
}

/// `<data_root>/auth/tls`.
pub fn generated_dir(data_root: &Path) -> PathBuf {
    crate::private_dir::path(data_root).join(DIR_NAME)
}

/// What a new certificate names besides `localhost`, `127.0.0.1` and `::1`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Identity {
    /// The computer's name, as the OS reports it (checked when used).
    pub hostname: Option<String>,
    /// Addresses of this machine (loopback and unspecified ones are left out when used).
    pub ips: Vec<IpAddr>,
}

impl Identity {
    /// This machine now: [`hostname`] and [`local_ips`].
    pub fn of_this_machine() -> Self {
        Self {
            hostname: hostname(),
            ips: local_ips(),
        }
    }
}

/// The computer's name: `COMPUTERNAME` on Windows, `/proc/sys/kernel/hostname` on Linux, `scutil
/// --get LocalHostName` on macOS (the Bonjour name), else `HOSTNAME`. None when nothing answers.
pub fn hostname() -> Option<String> {
    let clean = |s: String| {
        let s = s.trim().to_string();
        (!s.is_empty()).then_some(s)
    };
    let found = if cfg!(windows) {
        std::env::var("COMPUTERNAME").ok()
    } else if cfg!(target_os = "linux") {
        std::fs::read_to_string("/proc/sys/kernel/hostname").ok()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("/usr/sbin/scutil")
            .args(["--get", "LocalHostName"])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
    } else {
        None
    };
    found
        .and_then(clean)
        .or_else(|| std::env::var("HOSTNAME").ok().and_then(clean))
}

/// The addresses this machine sends from to the IPv4 and the IPv6 internet: a UDP socket
/// "connected" to a documentation address (192.0.2.1, 2001:db8::1) takes the source address of
/// the route there. `connect` on UDP sends nothing. A machine without such a route gives none.
pub fn local_ips() -> Vec<IpAddr> {
    let probe = |bind: &str, to: &str| -> Option<IpAddr> {
        let s = std::net::UdpSocket::bind(bind).ok()?;
        s.connect(to).ok()?;
        let ip = s.local_addr().ok()?.ip();
        (!ip.is_unspecified() && !ip.is_loopback()).then_some(ip)
    };
    [
        probe("0.0.0.0:0", "192.0.2.1:9"),
        probe("[::]:0", "[2001:db8::1]:9"),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Whether `name` is a DNS name a certificate can hold: labels of 1-63 letters, digits and
/// hyphens, not starting or ending with a hyphen, at most 253 characters, not all digits (an
/// address is not a name).
pub fn valid_dns_name(name: &str) -> bool {
    let label = |l: &str| {
        (1..=63).contains(&l.len())
            && l.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            && !l.starts_with('-')
            && !l.ends_with('-')
    };
    !name.is_empty()
        && name.len() <= 253
        && name.split('.').all(label)
        && !name.bytes().all(|b| b.is_ascii_digit() || b == b'.')
}

/// The DNS names of a new certificate: `localhost`, then `hostname` (lower case) and
/// `<first label>.local` when valid ([`valid_dns_name`]); nothing that is not.
pub fn dns_names(hostname: Option<&str>) -> Vec<String> {
    let mut out = vec!["localhost".to_string()];
    if let Some(h) = hostname.map(|h| h.trim().trim_end_matches('.').to_ascii_lowercase())
        && valid_dns_name(&h)
    {
        let first = h.split('.').next().unwrap_or(&h).to_string();
        for name in [h.clone(), format!("{first}.local")] {
            if !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

/// The IP addresses of a new certificate: `127.0.0.1`, `::1`, then `ips` without loopback,
/// unspecified or repeated ones (IPv4-mapped IPv6 as IPv4).
pub fn ip_names(ips: &[IpAddr]) -> Vec<IpAddr> {
    let mut out = vec![
        IpAddr::from([127, 0, 0, 1]),
        IpAddr::from(std::net::Ipv6Addr::LOCALHOST),
    ];
    for ip in ips.iter().map(|ip| ip.to_canonical()) {
        if !ip.is_loopback() && !ip.is_unspecified() && !out.contains(&ip) {
            out.push(ip);
        }
    }
    out
}

/// A new certificate and its key, PEM.
#[derive(Clone)]
pub struct Generated {
    pub cert_pem: String,
    pub key_pem: String,
}

impl std::fmt::Debug for Generated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generated").finish_non_exhaustive()
    }
}

/// Generate a certificate for `identity` made at `now` (see the module docs).
pub fn generate(identity: &Identity, now: SystemTime) -> Result<Generated> {
    use rcgen::{
        CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa, KeyPair,
        KeyUsagePurpose, SanType,
    };
    let at = |t: SystemTime| -> Result<time::OffsetDateTime> {
        let t = time::OffsetDateTime::from(t);
        t.replace_nanosecond(0)
            .context("a certificate date out of range")
    };
    let mut params = CertificateParams::default();
    params.not_before = at(now.checked_sub(BACKDATE).unwrap_or(now))?;
    params.not_after = at(now
        .checked_add(VALIDITY)
        .context("the clock is out of range")?)?;
    let mut dn = DistinguishedName::new();
    let who = identity
        .hostname
        .as_deref()
        .map(str::trim)
        .filter(|h| !h.is_empty())
        .unwrap_or("this computer");
    dn.push(DnType::CommonName, format!("Blue Onyx Prism ({who})"));
    params.distinguished_name = dn;
    params.is_ca = IsCa::ExplicitNoCa;
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let mut sans = Vec::new();
    for name in dns_names(identity.hostname.as_deref()) {
        // `dns_names` keeps ASCII names only, which always convert; skip one that would not.
        if let Ok(n) = rcgen::string::Ia5String::try_from(name) {
            sans.push(SanType::DnsName(n));
        }
    }
    sans.extend(ip_names(&identity.ips).into_iter().map(SanType::IpAddress));
    params.subject_alt_names = sans;
    let key = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
        .context("generating the HTTPS key")?;
    let cert = params
        .self_signed(&key)
        .context("signing the HTTPS certificate")?;
    Ok(Generated {
        cert_pem: cert.pem(),
        key_pem: key.serialize_pem(),
    })
}

/// SHA-256 of a DER certificate as browsers show it: upper-case hex pairs joined by colons.
pub fn fingerprint_sha256(cert_der: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(cert_der)
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(":")
}

/// One DER element at the start of `input`: (tag, contents, rest). Single-byte tags and
/// definite lengths of up to four bytes, which is all an X.509 certificate's outer layers use.
fn der_element(input: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let (&tag, rest) = input.split_first()?;
    let (&first, rest) = rest.split_first()?;
    let (len, rest) = if first < 0x80 {
        (usize::from(first), rest)
    } else {
        let n = usize::from(first & 0x7f);
        if n == 0 || n > 4 || rest.len() < n {
            return None;
        }
        let (bytes, rest) = rest.split_at(n);
        let len = bytes
            .iter()
            .fold(0usize, |acc, b| (acc << 8) | usize::from(*b));
        (len, rest)
    };
    if rest.len() < len {
        return None;
    }
    let (value, rest) = rest.split_at(len);
    Some((tag, value, rest))
}

/// Days from 1970-01-01 to the civil date (proleptic Gregorian; Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// An X.509 `Time` (UTCTime `YYMMDDHHMMSSZ`, GeneralizedTime `YYYYMMDDHHMMSSZ`).
fn der_time(tag: u8, value: &[u8]) -> Option<SystemTime> {
    let text = std::str::from_utf8(value).ok()?;
    let digits = text.strip_suffix('Z')?;
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let num =
        |from: usize, len: usize| -> Option<i64> { digits.get(from..from + len)?.parse().ok() };
    let (year, at) = match (tag, digits.len()) {
        (0x17, 12) => {
            let yy = num(0, 2)?;
            (if yy >= 50 { 1900 + yy } else { 2000 + yy }, 2)
        }
        (0x18, 14) => (num(0, 4)?, 4),
        _ => return None,
    };
    let (mo, d, h, mi, s) = (
        num(at, 2)?,
        num(at + 2, 2)?,
        num(at + 4, 2)?,
        num(at + 6, 2)?,
        num(at + 8, 2)?,
    );
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 60 {
        return None;
    }
    let secs = days_from_civil(year, mo, d) * 86_400 + h * 3600 + mi * 60 + s;
    let secs = u64::try_from(secs).ok()?;
    SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(secs))
}

/// The validity (not before, not after) of a DER certificate; None when it is not one.
pub fn validity(cert_der: &[u8]) -> Option<(SystemTime, SystemTime)> {
    const SEQUENCE: u8 = 0x30;
    let (tag, cert, _) = der_element(cert_der)?;
    if tag != SEQUENCE {
        return None;
    }
    let (tag, tbs, _) = der_element(cert)?;
    if tag != SEQUENCE {
        return None;
    }
    let (mut tag, _, mut rest) = der_element(tbs)?;
    // [0] EXPLICIT version, absent in a v1 certificate.
    if tag == 0xa0 {
        (tag, _, rest) = der_element(rest)?;
    }
    // serialNumber INTEGER, then signature AlgorithmIdentifier, issuer Name, validity.
    if tag != 0x02 {
        return None;
    }
    let (_, _, rest) = der_element(rest)?;
    let (_, _, rest) = der_element(rest)?;
    let (tag, validity, _) = der_element(rest)?;
    if tag != SEQUENCE {
        return None;
    }
    let (t1, v1, rest) = der_element(validity)?;
    let (t2, v2, _) = der_element(rest)?;
    Some((der_time(t1, v1)?, der_time(t2, v2)?))
}

/// Modification time and length of the files a [`Loaded`] came from: a change of either is a new
/// file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    cert: (Option<SystemTime>, u64),
    key: (Option<SystemTime>, u64),
}

fn stamp_of(cert: &Path, key: &Path) -> Option<Stamp> {
    let one = |p: &Path| {
        std::fs::metadata(p)
            .ok()
            .map(|m| (m.modified().ok(), m.len()))
    };
    Some(Stamp {
        cert: one(cert)?,
        key: one(key)?,
    })
}

/// A certificate in use: the rustls key, its fingerprint and expiry, and the stamp of its files.
#[derive(Debug)]
pub struct Loaded {
    pub key: Arc<CertifiedKey>,
    /// [`fingerprint_sha256`] of the leaf.
    pub fingerprint: String,
    /// The leaf's not-after; None when it could not be read.
    pub not_after: Option<SystemTime>,
    stamp: Option<Stamp>,
}

/// Why a pair on disk is not usable.
#[derive(Debug)]
enum PairProblem {
    /// One of the files does not exist.
    Missing,
    /// It exists but does not load: the message names the file and the cause, never the key.
    Bad(String),
}

/// Read and parse the pair at `cert` / `key`.
fn read_pair(cert: &Path, key: &Path) -> std::result::Result<Loaded, PairProblem> {
    let read = |p: &Path, what: &str| match std::fs::read(p) {
        Ok(b) => Ok(b),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(PairProblem::Missing),
        Err(e) => Err(PairProblem::Bad(format!(
            "the {what} {} could not be read ({e})",
            p.display()
        ))),
    };
    // Stamp first: a file replaced while it is read shows up as a change at the next check.
    let stamp = stamp_of(cert, key);
    let cert_pem = read(cert, "certificate")?;
    let key_pem = read(key, "key")?;
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(&cert_pem)
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| {
            PairProblem::Bad(format!(
                "the certificate {} is not PEM ({e})",
                cert.display()
            ))
        })?;
    let Some(leaf) = chain.first().cloned() else {
        return Err(PairProblem::Bad(format!(
            "the certificate {} holds no CERTIFICATE block",
            cert.display()
        )));
    };
    // The key's own error could quote its bytes: only say what kind of problem it is.
    let key_der = PrivateKeyDer::from_pem_slice(&key_pem).map_err(|e| {
        let why = match e {
            rustls::pki_types::pem::Error::NoItemsFound => "no PRIVATE KEY block in it",
            _ => "it is not a PEM private key",
        };
        PairProblem::Bad(format!("the key {} does not load: {why}", key.display()))
    })?;
    let certified = CertifiedKey::from_der(chain, key_der, &provider()).map_err(|e| {
        PairProblem::Bad(format!(
            "the key {} and the certificate {} do not go together ({e})",
            key.display(),
            cert.display()
        ))
    })?;
    Ok(Loaded {
        key: Arc::new(certified),
        fingerprint: fingerprint_sha256(&leaf),
        not_after: validity(&leaf).map(|(_, end)| end),
        stamp,
    })
}

/// What the expiry of a certificate calls for at `now`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expiry {
    Fine,
    /// Within [`WARN_BEFORE`] (generated) or [`RENEW_BEFORE`] (own files): days left.
    Soon(u64),
    /// A generated one within [`RENEW_BEFORE`]: make a new one.
    Renew,
    /// An own certificate past its end.
    Expired,
}

/// [`Expiry`] of a certificate ending at `not_after` (None: unknown) at `now`.
pub fn expiry(not_after: Option<SystemTime>, now: SystemTime, generated: bool) -> Expiry {
    let Some(end) = not_after else {
        return Expiry::Fine;
    };
    let left = end.duration_since(now).unwrap_or(Duration::ZERO);
    let days = left.as_secs() / DAY.as_secs();
    match (generated, left) {
        (true, l) if l < RENEW_BEFORE => Expiry::Renew,
        (true, l) if l < WARN_BEFORE => Expiry::Soon(days),
        (false, l) if l.is_zero() => Expiry::Expired,
        (false, l) if l < RENEW_BEFORE => Expiry::Soon(days),
        _ => Expiry::Fine,
    }
}

/// Write a new pair under `<data_root>/auth/tls` (key first: a reader that sees the new
/// certificate finds its key) and load it.
fn write_generated(data_root: &Path, identity: &Identity, now: SystemTime) -> Result<Loaded> {
    let auth = crate::private_dir::ensure(data_root)?;
    let dir = crate::private_dir::ensure_child(&auth, DIR_NAME)?;
    let made = generate(identity, now)?;
    crate::private_dir::write_file(&dir, KEY_FILE, &made.key_pem)?;
    crate::private_dir::write_file(&dir, CERT_FILE, &made.cert_pem)?;
    read_pair(&dir.join(CERT_FILE), &dir.join(KEY_FILE)).map_err(|p| match p {
        PairProblem::Missing => anyhow::anyhow!(
            "the certificate written to {} is gone again; check what else writes there",
            dir.display()
        ),
        PairProblem::Bad(why) => anyhow::anyhow!("{why}"),
    })
}

/// Who loads a generated pair, which decides what a load may change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The service: a damaged pair is replaced, one due for renewal renewed (module docs).
    Serve,
    /// `tls-fingerprint`: makes a pair when there is none, and otherwise never changes the
    /// certificate it reports (a damaged pair is an error).
    Report,
}

/// The certificate of `source` at `now`: the generated pair (made when missing; replaced when
/// damaged or due for renewal as `mode` allows), or the own files.
pub fn load(
    source: &Source,
    identity: &dyn Fn() -> Identity,
    now: SystemTime,
    mode: Mode,
) -> Result<Loaded> {
    let (cert, key) = source.paths();
    match source {
        Source::Files { .. } => {
            let loaded = read_pair(&cert, &key).map_err(|p| own_files_error(&cert, &key, p))?;
            note_expiry(&loaded, source, now);
            Ok(loaded)
        }
        Source::Generated { data_root } => {
            // The folders first: their protection is checked on every start.
            let auth = crate::private_dir::ensure(data_root)?;
            crate::private_dir::ensure_child(&auth, DIR_NAME)?;
            let loaded = match read_pair(&cert, &key) {
                Ok(l) => l,
                Err(PairProblem::Missing) => {
                    let l = write_generated(data_root, &identity(), now)?;
                    info!(
                        cert = %cert.display(),
                        fingerprint = %l.fingerprint,
                        "created a self-signed HTTPS certificate; browsers warn about it once, \
                         compare the fingerprint they show with this one before accepting it"
                    );
                    return Ok(l);
                }
                Err(PairProblem::Bad(why)) => match mode {
                    Mode::Report => bail!(
                        "{why}. Run `blue-onyx-prism tls-regenerate` to replace the pair (browsers \
                         then ask once more)"
                    ),
                    Mode::Serve => {
                        warn!(
                            "{why}; replacing it with a new self-signed certificate, so browsers \
                             that accepted the old one ask once more"
                        );
                        return write_generated(data_root, &identity(), now);
                    }
                },
            };
            if mode == Mode::Serve && expiry(loaded.not_after, now, true) == Expiry::Renew {
                let l = write_generated(data_root, &identity(), now)?;
                info!(
                    fingerprint = %l.fingerprint,
                    "the HTTPS certificate had less than {} days left; made a new one, so \
                     browsers ask once more",
                    RENEW_BEFORE.as_secs() / DAY.as_secs()
                );
                return Ok(l);
            }
            note_expiry(&loaded, source, now);
            Ok(loaded)
        }
    }
}

/// The error of own files that do not load: what failed and what to do.
fn own_files_error(cert: &Path, key: &Path, p: PairProblem) -> anyhow::Error {
    let why = match p {
        PairProblem::Missing => format!(
            "tls_cert_path {} or tls_key_path {} does not exist",
            cert.display(),
            key.display()
        ),
        PairProblem::Bad(why) => why,
    };
    anyhow::anyhow!(
        "{why}. The HTTPS listener stays down (plain HTTP keeps working) until the pair loads; \
         fix the files (PEM: the certificate chain, leaf first, and its private key) or remove \
         tls_cert_path and tls_key_path from the config file to use a generated certificate. \
         The files are read again every 30 s"
    )
}

/// Log what the expiry of `loaded` calls for (renewal is the caller's).
fn note_expiry(loaded: &Loaded, source: &Source, now: SystemTime) {
    match expiry(loaded.not_after, now, source.generated()) {
        Expiry::Fine | Expiry::Renew => {}
        Expiry::Soon(days) if source.generated() => warn!(
            days_left = days,
            "the generated HTTPS certificate expires soon; it is renewed automatically when {} \
             days are left (browsers then ask once more), or now with `blue-onyx-prism \
             tls-regenerate`",
            RENEW_BEFORE.as_secs() / DAY.as_secs()
        ),
        Expiry::Soon(days) => warn!(
            days_left = days,
            cert = %source.paths().0.display(),
            "the HTTPS certificate in tls_cert_path expires soon; replace the files (the \
             running service picks up the change at its next HTTPS connection)"
        ),
        Expiry::Expired => warn!(
            cert = %source.paths().0.display(),
            "the HTTPS certificate in tls_cert_path has expired, so browsers refuse it; replace \
             the files (the running service picks up the change at its next HTTPS connection)"
        ),
    }
}

/// Replace the generated pair of `data_root` with a new one (`tls-regenerate`).
pub fn regenerate(data_root: &Path, identity: &Identity, now: SystemTime) -> Result<Loaded> {
    write_generated(data_root, identity, now)
}

/// A clock: the tests move it.
pub type Clock = Arc<dyn Fn() -> SystemTime + Send + Sync>;
/// What a new certificate names, asked when one is made.
pub type IdentityFn = Arc<dyn Fn() -> Identity + Send + Sync>;

/// The rustls certificate resolver of the HTTPS listener (see the module docs, Reload).
#[derive(Clone)]
pub struct Resolver {
    inner: Arc<Inner>,
}

struct Inner {
    source: Source,
    identity: IdentityFn,
    clock: Clock,
    current: ArcSwap<Loaded>,
    /// When a handshake next checks the files.
    next_check: Mutex<Instant>,
    check_every: Duration,
    /// A reload or renewal is running on its thread.
    busy: AtomicBool,
    /// Day (since 1970) of the last expiry note while running: one a day.
    noted_day: Mutex<Option<u64>>,
    /// Stamp of the last pair that failed to load while running: one warning per change.
    failed: Mutex<Option<Stamp>>,
}

impl std::fmt::Debug for Resolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resolver")
            .field("source", &self.inner.source)
            .field("fingerprint", &self.fingerprint())
            .finish_non_exhaustive()
    }
}

impl Resolver {
    /// Load `source` now (blocking file IO: call it off the async threads) for this machine
    /// and the system clock.
    pub fn new(source: Source) -> Result<Self> {
        Self::with(
            source,
            Arc::new(Identity::of_this_machine),
            Arc::new(SystemTime::now),
            CHECK_EVERY,
        )
    }

    /// [`Resolver::new`] with its parts given.
    pub fn with(
        source: Source,
        identity: IdentityFn,
        clock: Clock,
        check_every: Duration,
    ) -> Result<Self> {
        let loaded = load(&source, identity.as_ref(), clock(), Mode::Serve)?;
        Ok(Self {
            inner: Arc::new(Inner {
                source,
                identity,
                clock,
                current: ArcSwap::from_pointee(loaded),
                next_check: Mutex::new(Instant::now() + check_every),
                check_every,
                busy: AtomicBool::new(false),
                noted_day: Mutex::new(None),
                failed: Mutex::new(None),
            }),
        })
    }

    /// The fingerprint of the certificate in use.
    pub fn fingerprint(&self) -> String {
        self.inner.current.load().fingerprint.clone()
    }

    /// The certificate in use.
    pub fn current(&self) -> Arc<Loaded> {
        self.inner.current.load_full()
    }

    /// Where the certificate comes from.
    pub fn source(&self) -> &Source {
        &self.inner.source
    }

    /// Whether the files changed or a renewal is due: a cheap stat, then the work on a thread
    /// of its own. At most once per `check_every`.
    fn maybe_check(&self) {
        use crate::sync::LockExt;
        let now = Instant::now();
        {
            let mut next = self.inner.next_check.lock_recover();
            if now < *next {
                return;
            }
            *next = now + self.inner.check_every;
        }
        if !self.needs_refresh() || self.inner.busy.swap(true, Ordering::AcqRel) {
            return;
        }
        let me = self.clone();
        let spawned = std::thread::Builder::new()
            .name("tls-reload".into())
            .spawn(move || {
                me.refresh();
                me.inner.busy.store(false, Ordering::Release);
            });
        if let Err(e) = spawned {
            self.inner.busy.store(false, Ordering::Release);
            warn!("could not start the HTTPS certificate check: {e}");
        }
    }

    /// Whether [`Resolver::refresh`] has something to do: a changed stamp, a renewal or the
    /// day's expiry note.
    fn needs_refresh(&self) -> bool {
        use crate::sync::LockExt;
        let cur = self.inner.current.load();
        let (cert, key) = self.inner.source.paths();
        let stamp = stamp_of(&cert, &key);
        if stamp != cur.stamp && stamp != *self.inner.failed.lock_recover() {
            return true;
        }
        let now = (self.inner.clock)();
        match expiry(cur.not_after, now, self.inner.source.generated()) {
            Expiry::Fine => false,
            Expiry::Renew => true,
            Expiry::Soon(_) | Expiry::Expired => {
                *self.inner.noted_day.lock_recover() != Some(day_of(now))
            }
        }
    }

    /// Reload a changed pair, renew a generated one that is due, note an expiry once a day
    /// (blocking; [`Resolver::maybe_check`] runs it on its own thread, the tests directly).
    pub fn refresh(&self) {
        use crate::sync::LockExt;
        let now = (self.inner.clock)();
        let source = &self.inner.source;
        let (cert, key) = source.paths();
        let cur = self.inner.current.load_full();
        let stamp = stamp_of(&cert, &key);
        if stamp != cur.stamp {
            match read_pair(&cert, &key) {
                Ok(l) => {
                    info!(
                        fingerprint = %l.fingerprint,
                        cert = %cert.display(),
                        "the HTTPS certificate file changed; serving the new certificate"
                    );
                    *self.inner.failed.lock_recover() = None;
                    self.inner.current.store(Arc::new(l));
                }
                Err(p) => {
                    let mut failed = self.inner.failed.lock_recover();
                    if *failed != stamp {
                        let why = match p {
                            PairProblem::Missing => format!("{} is gone", cert.display()),
                            PairProblem::Bad(why) => why,
                        };
                        warn!(
                            "the HTTPS certificate files changed but do not load: {why}; \
                             keeping the certificate in use (fingerprint {}) until a good \
                             pair is in place",
                            cur.fingerprint
                        );
                        *failed = stamp;
                    }
                }
            }
        }
        let cur = self.inner.current.load_full();
        match (source, expiry(cur.not_after, now, source.generated())) {
            (_, Expiry::Fine) => {}
            (Source::Generated { data_root }, Expiry::Renew) => {
                match write_generated(data_root, &(self.inner.identity)(), now) {
                    Ok(l) => {
                        info!(
                            fingerprint = %l.fingerprint,
                            "renewed the HTTPS certificate ({} days or less were left); browsers \
                             ask once more",
                            RENEW_BEFORE.as_secs() / DAY.as_secs()
                        );
                        self.inner.current.store(Arc::new(l));
                    }
                    Err(e) => {
                        let mut noted = self.inner.noted_day.lock_recover();
                        if *noted != Some(day_of(now)) {
                            warn!(
                                "could not renew the HTTPS certificate: {e:#}; it keeps serving \
                                 until it expires. Check that the service may write in {}",
                                generated_dir(data_root).display()
                            );
                            *noted = Some(day_of(now));
                        }
                    }
                }
            }
            _ => {
                let mut noted = self.inner.noted_day.lock_recover();
                if *noted != Some(day_of(now)) {
                    note_expiry(&cur, source, now);
                    *noted = Some(day_of(now));
                }
            }
        }
    }
}

/// Whole days since 1970 at `t`.
fn day_of(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        / DAY.as_secs()
}

impl rustls::server::ResolvesServerCert for Resolver {
    fn resolve(&self, _hello: rustls::server::ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        self.maybe_check();
        Some(self.inner.current.load().key.clone())
    }
}

/// The server config of the HTTPS listener: aws-lc-rs, TLS 1.2 and 1.3, no client certificates,
/// ALPN `http/1.1` only, certificates from `resolver`.
pub fn server_config(resolver: &Resolver) -> Result<Arc<rustls::ServerConfig>> {
    let mut config = rustls::ServerConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
        .context("the TLS provider offers no protocol version")?
        .with_no_client_auth()
        .with_cert_resolver(Arc::new(resolver.clone()));
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Arc::new(config))
}

/// Fingerprint and URLs `tls-fingerprint` prints for `config`: the certificate is made when
/// there is none (generated source), never replaced.
pub fn fingerprint_report(config: &Config) -> Result<String> {
    let source = Source::of(config);
    let loaded = match &source {
        Source::Files { cert, key } => {
            read_pair(cert, key).map_err(|p| own_files_error(cert, key, p))?
        }
        Source::Generated { .. } => load(
            &source,
            &Identity::of_this_machine,
            SystemTime::now(),
            Mode::Report,
        )?,
    };
    Ok(report_text(
        &loaded,
        &source,
        config.https_port,
        &local_ips(),
        SystemTime::now(),
    ))
}

/// The text of [`fingerprint_report`].
fn report_text(
    loaded: &Loaded,
    source: &Source,
    port: u16,
    ips: &[IpAddr],
    now: SystemTime,
) -> String {
    let mut out = format!("SHA-256 fingerprint: {}\n", loaded.fingerprint);
    out.push_str(&format!("Certificate: {}\n", source.paths().0.display()));
    if let Some(end) = loaded.not_after {
        let days = end
            .duration_since(now)
            .map_or(0, |d| d.as_secs() / DAY.as_secs());
        out.push_str(&format!("Valid for {days} more days\n"));
    }
    out.push_str("Open the web UI at:\n");
    out.push_str(&format!("  https://localhost:{port}/\n"));
    for ip in ips.iter().filter(|ip| !ip.is_loopback()) {
        out.push_str(&format!(
            "  https://{}/\n",
            std::net::SocketAddr::new(*ip, port)
        ));
    }
    out.push_str(
        "The browser warns about a self-signed certificate once: compare the fingerprint it \
         shows (certificate details, SHA-256) with the one above before accepting it.\n",
    );
    out
}

/// `tls-regenerate` for `config`: refused with own files, else a new pair. The text to print.
pub fn regenerate_report(config: &Config) -> Result<String> {
    if let Source::Files { cert, .. } = Source::of(config) {
        bail!(
            "tls_cert_path and tls_key_path are set in the config file ({}), so this service \
             uses those files and never writes them; replace them yourself, or remove both keys \
             to use a generated certificate",
            cert.display()
        );
    }
    let loaded = regenerate(
        &config.data_root(),
        &Identity::of_this_machine(),
        SystemTime::now(),
    )?;
    Ok(format!(
        "Made a new self-signed certificate.\nSHA-256 fingerprint: {}\nA running service \
         switches to it within seconds of its next HTTPS connection; browsers that accepted the \
         old one ask once more.\n",
        loaded.fingerprint
    ))
}

#[cfg(test)]
pub(crate) mod tests;
