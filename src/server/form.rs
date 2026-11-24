//! One reading of on / off form fields for every route (QA-094).
//!
//! A ticked HTML checkbox posts `on` (or its `value`), an unticked one posts nothing; scripts and
//! hidden inputs post `1` / `0`, `true` / `false`, `yes` / `no`. [`truthy`] reads all of them the
//! same way on every route, so `confirm_large=no` never confirms a large download on one route
//! while refusing it on another. [`checked`] is a field that is present and truthy: an absent,
//! empty or false-like field is off. The Refine tab's `allow_large` stays stricter on purpose
//! (only `1|true|on|yes`; anything else is refused, `server::refine`).

/// Whether a form value means on: anything but empty, `0`, `false`, `off` or `no` (any case,
/// surrounding blanks ignored).
pub(super) fn truthy(v: &str) -> bool {
    !matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "off" | "no"
    )
}

/// Whether the first `k` field of `f` is present and [`truthy`].
pub(super) fn checked(f: &[(String, String)], k: &str) -> bool {
    f.iter()
        .find(|(n, _)| n == k)
        .is_some_and(|(_, v)| truthy(v))
}

/// [`checked`] for a form read into a map (one value per name).
pub(super) fn checked_in(f: &std::collections::HashMap<String, String>, k: &str) -> bool {
    f.get(k).is_some_and(|v| truthy(v))
}

/// The values every route reads as off (route tests post each one).
#[cfg(test)]
pub(super) const FALSE_LIKE: [&str; 9] =
    ["", " ", "0", "false", "FALSE", "off", "Off", "no", " No "];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn false_like_values_are_off_everywhere_else_on() {
        for v in FALSE_LIKE {
            assert!(!truthy(v), "{v:?}");
        }
        for v in ["1", "on", "true", "yes", "YES", "x"] {
            assert!(truthy(v), "{v:?}");
        }
    }

    #[test]
    fn checked_needs_the_field_present_and_truthy() {
        let f = |k: &str, v: &str| vec![(k.to_string(), v.to_string())];
        assert!(checked(&f("a", "on"), "a"));
        assert!(!checked(&f("a", "no"), "a"));
        assert!(!checked(&f("b", "on"), "a"));
        let mut m = std::collections::HashMap::new();
        assert!(!checked_in(&m, "a"));
        m.insert("a".to_string(), "0".to_string());
        assert!(!checked_in(&m, "a"));
        m.insert("a".to_string(), "1".to_string());
        assert!(checked_in(&m, "a"));
    }
}
