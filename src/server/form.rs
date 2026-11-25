//! One reading of on / off form fields for every route (QA-094).
//!
//! A ticked HTML checkbox posts `on` (or its `value`), an unticked one posts nothing; scripts and
//! hidden inputs post `1` / `0`, `true` / `false`, `yes` / `no`. [`truthy`] reads all of them the
//! same way on every route, so `confirm_large=no` never confirms a large download on one route
//! while refusing it on another. [`checked`] is a field that is present and truthy: an absent,
//! empty or false-like field is off. The Refine tab's `allow_large` stays stricter on purpose
//! (only `1|true|on|yes`; anything else is refused, `server::refine`).
//!
//! The fields themselves (QA-133): [`parse`] reads an urlencoded body (or query string) into its
//! fields in order, repeated names kept, as `axum::Form<Vec<(String, String)>>` does. Then one
//! name per meaning: [`value`] is the first field trimmed, empty kept (present and empty means
//! something: unset); [`field`] the first field trimmed and non-empty (empty = absent); [`all`]
//! every such field trimmed and non-empty; [`num`] a [`field`] parsed as a number.

/// The fields of an `application/x-www-form-urlencoded` body in order, repeated names kept (the
/// page's frame checkboxes and model ticks need them): `&`-separated, `+` is a space, `%` takes
/// two hex digits only (anything else is kept as typed), invalid UTF-8 becomes U+FFFD.
pub(super) fn parse(body: &[u8]) -> Vec<(String, String)> {
    form_urlencoded::parse(body)
        .map(|(k, v)| (k.into_owned(), v.into_owned()))
        .collect()
}

/// The first `k` field of `f`, trimmed; an empty one is kept (`Some("")`: present and empty).
pub(super) fn value<'a>(f: &'a [(String, String)], k: &str) -> Option<&'a str> {
    f.iter().find(|(n, _)| n == k).map(|(_, v)| v.trim())
}

/// The first `k` field of `f`, trimmed; None when absent or empty.
pub(super) fn field<'a>(f: &'a [(String, String)], k: &str) -> Option<&'a str> {
    value(f, k).filter(|v| !v.is_empty())
}

/// Every `k` field of `f` in order, trimmed, the empty ones left out.
pub(super) fn all<'a>(f: &'a [(String, String)], k: &str) -> Vec<&'a str> {
    f.iter()
        .filter(|(n, _)| n == k)
        .map(|(_, v)| v.trim())
        .filter(|v| !v.is_empty())
        .collect()
}

/// The [`field`] `k` as a number; None when absent or empty, an error naming `what` when it does
/// not parse (`what: 'x' is not a number`).
pub(super) fn num<T: std::str::FromStr>(
    f: &[(String, String)],
    k: &str,
    what: &str,
) -> anyhow::Result<Option<T>> {
    field(f, k)
        .map(|v| {
            v.parse::<T>()
                .map_err(|_| anyhow::anyhow!("{what}: '{v}' is not a number"))
        })
        .transpose()
}

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
    fn form_parsing_keeps_repeated_keys_and_decodes() {
        let f = parse(b"frame=ir%2Fa%2F2026-10-22%2Fx.jpg&frame=b+c&enabled=on&bad=%zz&k");
        assert_eq!(f[0], ("frame".into(), "ir/a/2026-10-22/x.jpg".into()));
        assert_eq!(f[1].1, "b c");
        assert_eq!(f[3].1, "%zz");
        assert_eq!(f[4], ("k".into(), String::new()));
    }

    /// q-extras [17]: `%` takes two hex digits only; `from_str_radix` alone read `%+5` as 0x05.
    #[test]
    fn a_percent_without_two_hex_digits_is_kept_literally() {
        let f = parse(b"a=%+5&b=%-1x&c=%41%4");
        assert_eq!(f[0].1, "% 5");
        assert_eq!(f[1].1, "%-1x");
        assert_eq!(f[2].1, "A%4");
    }

    /// What the hand-written decoder this replaced gave, case by case (QA-133).
    #[test]
    fn parsing_matches_the_old_decoder() {
        let pairs = |b: &[u8]| -> Vec<(String, String)> { parse(b) };
        let p = |k: &str, v: &str| (k.to_string(), v.to_string());
        assert_eq!(pairs(b""), vec![]);
        assert_eq!(pairs(b"&&a=1&&"), vec![p("a", "1")]);
        assert_eq!(pairs(b"a=b=c"), vec![p("a", "b=c")]);
        assert_eq!(pairs(b"a="), vec![p("a", "")]);
        assert_eq!(pairs(b"=v"), vec![p("", "v")]);
        assert_eq!(pairs(b"a+b%3D=x%26y"), vec![p("a b=", "x&y")]);
        assert_eq!(pairs(b"n=%C3%A9t%C3%A9"), vec![p("n", "\u{e9}t\u{e9}")]);
        assert_eq!(pairs("n=\u{e9}".as_bytes()), vec![p("n", "\u{e9}")]);
        assert_eq!(pairs(b"n=%FF"), vec![p("n", "\u{fffd}")]);
        assert_eq!(pairs(b"n=%2b%2B"), vec![p("n", "++")]);
        assert_eq!(pairs(b"a=1;b=2"), vec![p("a", "1;b=2")]);
    }

    #[test]
    fn value_field_all_and_num_differ_only_on_empty() {
        let f = parse(b"a=+x+&a=y&e=&e=z&n=+7+&bad=seven");
        assert_eq!(value(&f, "a"), Some("x"));
        assert_eq!(field(&f, "a"), Some("x"));
        // The first `e` is present and empty: `value` keeps it, `field` reads it as absent.
        assert_eq!(value(&f, "e"), Some(""));
        assert_eq!(field(&f, "e"), None);
        assert_eq!(value(&f, "missing"), None);
        assert_eq!(all(&f, "a"), vec!["x", "y"]);
        assert_eq!(all(&f, "e"), vec!["z"]);
        assert_eq!(num::<u32>(&f, "n", "count").unwrap(), Some(7));
        assert_eq!(num::<u32>(&f, "e", "count").unwrap(), None);
        assert_eq!(num::<u32>(&f, "missing", "count").unwrap(), None);
        let e = num::<u32>(&f, "bad", "count").unwrap_err();
        assert_eq!(e.to_string(), "count: 'seven' is not a number");
    }

    #[test]
    fn truthy_table() {
        for (v, on) in [
            ("", false),
            ("  ", false),
            ("0", false),
            (" 0 ", false),
            ("false", false),
            ("False", false),
            ("off", false),
            ("OFF", false),
            ("no", false),
            ("No", false),
            ("1", true),
            ("on", true),
            ("ON", true),
            ("true", true),
            ("yes", true),
            ("2", true),
            ("00", true),
            ("nope", true),
            ("x", true),
        ] {
            assert_eq!(truthy(v), on, "{v:?}");
        }
    }

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
