//! Small text formatters shared by the training jobs' progress lines and summaries (import,
//! auto-label, overlaps, the dataset check). Pure, std only; [`crate::training::import::preview`]
//! re-exports [`thousands`] for its older callers.

/// `n` with thousands separators (`1,500`).
pub fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousands_separators() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1500), "1,500");
        assert_eq!(thousands(20_000), "20,000");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }
}
