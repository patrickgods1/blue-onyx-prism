//! Random secrets from the operating system: the admin token ([`crate::admin_token`]), the
//! setup code ([`crate::accounts::setup`]), session and device-cookie ids
//! (`accounts::opaque`) and new accounts' first generation; plus [`digest_eq`], the one
//! constant-time comparison of a presented secret with a kept one.
//!
//! `uuid` is this crate's source of randomness (`getrandom`, no userspace generator: the build
//! has neither `fast-rng` nor `rng-rand`). One v4 UUID holds 122 random bits, so every 32 bytes
//! handed out are the SHA-256 of three fresh UUIDs (366 bits): no output byte carries less than a
//! full byte of OS entropy, and nothing is kept between calls.

use sha2::{Digest, Sha256};

/// UUIDs hashed into each 32-byte block (3 x 122 = 366 random bits).
const UUIDS_PER_BLOCK: usize = 3;

/// `n` random bytes.
pub fn random_bytes(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut seed = Sha256::new();
        for _ in 0..UUIDS_PER_BLOCK {
            seed.update(uuid::Uuid::new_v4().as_bytes());
        }
        let block = seed.finalize();
        let want = (n - out.len()).min(block.len());
        out.extend(block.iter().take(want));
    }
    out
}

/// `bytes` random bytes as lowercase hex (`2 * bytes` characters).
pub fn random_hex(bytes: usize) -> String {
    hex(&random_bytes(bytes))
}

/// `bytes` as lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// Whether `a` and `b` are the same bytes, in time that depends on neither: both are hashed
/// (SHA-256, so lengths do not show either) and the digests compared with no early exit.
pub fn digest_eq(a: &[u8], b: &[u8]) -> bool {
    let ours = Sha256::digest(a);
    let theirs = Sha256::digest(b);
    let diff = ours
        .iter()
        .zip(theirs.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(diff) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_and_alphabet() {
        for n in [0, 1, 13, 31, 32, 33, 64, 100] {
            assert_eq!(random_bytes(n).len(), n);
            let h = random_hex(n);
            assert_eq!(h.len(), 2 * n);
            assert!(
                h.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            );
        }
        assert_eq!(hex(&[0, 0xab, 0x0f]), "00ab0f");
    }

    #[test]
    fn digest_eq_compares_whole_values() {
        assert!(digest_eq(b"", b""));
        assert!(digest_eq(b"abc", b"abc"));
        assert!(!digest_eq(b"abc", b"abd"));
        assert!(!digest_eq(b"abc", b"abcd"), "a prefix is not equal");
        assert!(!digest_eq(b"abc", b""));
        assert!(!digest_eq(b"ABC", b"abc"), "case counts");
    }

    #[test]
    fn values_differ() {
        let a = random_hex(32);
        let b = random_hex(32);
        assert_ne!(a, b);
        // Blocks of one call differ too (each from fresh UUIDs).
        let long = random_bytes(64);
        assert_ne!(long[..32], long[32..]);
    }
}
