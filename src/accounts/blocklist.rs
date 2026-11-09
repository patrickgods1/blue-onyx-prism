//! The list of common passwords a new password must not be (NIST SP 800-63B-4, 3.1.1.2).
//!
//! `assets/password-blocklist.txt` is embedded: the entries of 15 to 128 characters (shorter ones
//! are refused by length anyway) of two SecLists lists (MIT, credited in
//! `THIRD_PARTY_LICENSES.md`), NFC-normalized, lowercased, deduplicated and sorted by byte, one
//! per line after a header line that names the source, its commit and its license. The header is
//! never an entry. A lookup is a binary search over the lines with the password NFC-normalized
//! and lowercased the same way, so the check ignores case.

use std::sync::OnceLock;

/// The embedded file (see the module docs).
const TEXT: &str = include_str!("../../assets/password-blocklist.txt");

/// The entries, sorted (the file's lines after the header), split once.
fn entries() -> &'static [&'static str] {
    static ENTRIES: OnceLock<Vec<&'static str>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        TEXT.lines()
            .skip(1)
            .map(|l| l.strip_suffix('\r').unwrap_or(l))
            .filter(|l| !l.is_empty())
            .collect()
    })
}

/// How many entries the list has.
pub fn len() -> usize {
    entries().len()
}

/// `password` as the list stores it: NFC, lowercased, NFC again (lowercasing can undo NFC in
/// rare cases).
pub fn fold(password: &str) -> String {
    let nfc = super::password::nfc(password);
    super::password::nfc(&nfc.to_lowercase())
}

/// Whether `password` (any case, any normalization) is on the list.
pub fn contains(password: &str) -> bool {
    entries().binary_search(&fold(password).as_str()).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_is_sorted_unique_folded_and_in_range() {
        let e = entries();
        assert!(e.len() > 5_000, "{} entries", e.len());
        assert!(e.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
        for w in e {
            assert_eq!(&fold(w), w, "folded");
            let n = w.chars().count();
            assert!((15..=128).contains(&n), "{w:?} has {n} characters");
        }
        let header = TEXT.lines().next().unwrap();
        assert!(header.starts_with("# "), "{header}");
        assert!(header.contains("SecLists") && header.contains("MIT"));
        assert!(!contains(header), "the header is no entry");
    }

    #[test]
    fn lookups_ignore_case() {
        let e = entries();
        let first = e[0];
        assert!(contains(first));
        assert!(contains(&first.to_uppercase()));
        assert!(contains("1234567890123456"), "a well-known one");
        assert!(contains("QWERTYUIOPASDFGHJKL"));
        assert!(!contains("correct horse battery staple violet"));
    }
}
