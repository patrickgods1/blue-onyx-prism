//! FNV-1a 64: the one copy of the fixed hash the collector and the review labels use where std's
//! hasher would not do (it may change between Rust releases): the collector's sampling
//! ([`super::store::sample`]), the random-pick pool ([`super::labels::recommend::pool_hash`]) and
//! the static-spot frame sample ([`super::labels::statics::fnv1a`]). Each of those is pinned to
//! fixed values by its own test; they need not agree with one another, only with themselves over
//! time. Std-only.

/// The FNV-1a 64 offset basis (the hash of no bytes).
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// The FNV-1a 64 prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 of `bytes`.
pub fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    fnv1a64_from(FNV_OFFSET, bytes)
}

/// FNV-1a 64 continued from the state `h` (the hash of what came before) over `bytes`: hashing in
/// pieces gives the hash of the whole.
pub fn fnv1a64_from(h: u64, bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes
        .into_iter()
        .fold(h, |h, b| (h ^ u64::from(b)).wrapping_mul(FNV_PRIME))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_values() {
        assert_eq!(fnv1a64([]), 14_695_981_039_346_656_037);
        assert_eq!(fnv1a64(*b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(*b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn pieces_hash_as_the_whole() {
        let whole = fnv1a64("salt/day/cam/x.jpg".bytes());
        let first = fnv1a64("salt/".bytes());
        assert_eq!(fnv1a64_from(first, "day/cam/x.jpg".bytes()), whole);
    }
}
