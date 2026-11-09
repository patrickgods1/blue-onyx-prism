//! The web UI's cookies (pure): their names, the `Set-Cookie` values that set and clear them,
//! and the one parser of a `Cookie` header.
//!
//! Both cookies carry the `__Host-` prefix: a browser only accepts such a cookie from a secure
//! origin, with `Secure`, `Path=/` and no `Domain`, so a cookie planted over plain HTTP (by
//! anyone on the LAN able to answer for this host's port 80 or the plain port) can never carry
//! the name. [`value`] therefore matches the exact name only, case included: `bop_session` or
//! `__host-bop_session` is someone else's cookie. Sessions: [`super::sessions`]; devices:
//! [`super::devices`].

use super::devices::{self, DeviceId};
use super::sessions::SessionId;

/// The session cookie.
pub const SESSION: &str = "__Host-bop_session";
/// The device cookie.
pub const DEVICE: &str = "__Host-bop_device";
/// The admin-token cookie of earlier versions (plain HTTP, no prefix).
pub const LEGACY_ADMIN: &str = "bop_admin";
/// Attributes of both cookies.
const ATTRIBUTES: &str = "Secure; HttpOnly; SameSite=Strict; Path=/";

/// `Set-Cookie` of a session: `max_age_secs` is what is left of it
/// ([`super::sessions::Valid::expires_in`] at least 1, or [`super::sessions::ABSOLUTE_SECS`] for
/// a new one).
pub fn session_set_cookie(id: &SessionId, max_age_secs: u64) -> String {
    format!(
        "{SESSION}={}; {ATTRIBUTES}; Max-Age={max_age_secs}",
        id.reveal()
    )
}

/// `Set-Cookie` of a device cookie, kept [`devices::MAX_AGE_SECS`].
pub fn device_set_cookie(id: &DeviceId) -> String {
    format!(
        "{DEVICE}={}; {ATTRIBUTES}; Max-Age={}",
        id.reveal(),
        devices::MAX_AGE_SECS
    )
}

/// `Set-Cookie` that removes the cookie `name` ([`SESSION`] or [`DEVICE`]).
pub fn cleared(name: &str) -> String {
    format!("{name}=; {ATTRIBUTES}; Max-Age=0")
}

/// `Set-Cookie` that removes the earlier versions' `bop_admin` cookie (set without `Secure`, so
/// cleared without it too).
pub fn cleared_legacy_admin() -> String {
    format!("{LEGACY_ADMIN}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
}

/// The value of the cookie named exactly `name` in one `Cookie` header's bytes: the first
/// occurrence wins, an empty value is `None`. Bytes, so a value that is not text is a wrong one
/// rather than none.
pub fn value<'a>(header: &'a [u8], name: &str) -> Option<&'a [u8]> {
    header
        .split(|b| *b == b';')
        .find_map(|pair| {
            let pair = pair.trim_ascii();
            let (key, value) = pair.split_at_checked(name.len())?;
            (key == name.as_bytes())
                .then(|| value.strip_prefix(b"="))
                .flatten()
        })
        .filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_cookie_strings_are_exact() {
        let id = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let s = SessionId::from_presented(id.as_bytes()).unwrap();
        assert_eq!(
            session_set_cookie(&s, 43_200),
            format!(
                "__Host-bop_session={id}; Secure; HttpOnly; SameSite=Strict; Path=/; \
                 Max-Age=43200"
            )
        );
        let d = DeviceId::from_presented(id.as_bytes()).unwrap();
        assert_eq!(
            device_set_cookie(&d),
            format!(
                "__Host-bop_device={id}; Secure; HttpOnly; SameSite=Strict; Path=/; \
                 Max-Age=15552000"
            )
        );
        assert_eq!(
            cleared(SESSION),
            "__Host-bop_session=; Secure; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
        );
        assert_eq!(
            cleared(DEVICE),
            "__Host-bop_device=; Secure; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
        );
        assert_eq!(
            cleared_legacy_admin(),
            "bop_admin=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
        );
    }

    #[test]
    fn only_the_exact_name_matches() {
        fn v(h: &[u8]) -> Option<&[u8]> {
            value(h, SESSION)
        }
        assert_eq!(v(b"__Host-bop_session=abc"), Some(&b"abc"[..]));
        assert_eq!(v(b"a=1;  __Host-bop_session=abc ; b=2"), Some(&b"abc"[..]));
        assert_eq!(v(b"a=1;__Host-bop_session=abc"), Some(&b"abc"[..]));
        // Prefix and case variants are other cookies.
        assert_eq!(v(b"bop_session=abc"), None);
        assert_eq!(v(b"__host-bop_session=abc"), None);
        assert_eq!(v(b"__HOST-bop_session=abc"), None);
        assert_eq!(v(b"__Host-BOP_SESSION=abc"), None);
        assert_eq!(v(b"x__Host-bop_session=abc"), None);
        assert_eq!(v(b"__Host-bop_sessionx=abc"), None);
        assert_eq!(v(b"__Host-bop_session_old=abc"), None);
        assert_eq!(v(b"__Host-bop_session =abc"), None);
        // A planted variant before the real one does not hide it.
        assert_eq!(
            v(b"bop_session=evil; __host-bop_session=evil; __Host-bop_session=good"),
            Some(&b"good"[..])
        );
        // Duplicates: the first wins, also when it is empty.
        assert_eq!(
            v(b"__Host-bop_session=one; __Host-bop_session=two"),
            Some(&b"one"[..])
        );
        assert_eq!(v(b"__Host-bop_session=; __Host-bop_session=two"), None);
        // Empty and missing.
        assert_eq!(v(b""), None);
        assert_eq!(v(b";;"), None);
        assert_eq!(v(b"__Host-bop_session"), None);
        assert_eq!(v(b"__Host-bop_session="), None);
        // Bytes that are not text come back as they are.
        assert_eq!(
            v(b"__Host-bop_session=\xff\xfe; x=\xc3"),
            Some(&b"\xff\xfe"[..])
        );
        assert_eq!(value(b"__Host-bop_device=d1", DEVICE), Some(&b"d1"[..]));
        assert_eq!(value(b"__Host-bop_device=d1", SESSION), None);
        assert_eq!(value(b"bop_admin=t", LEGACY_ADMIN), Some(&b"t"[..]));
    }
}
