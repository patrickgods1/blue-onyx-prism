//! The network rule (pure): which client addresses this server answers at all, before any route,
//! credential or body is looked at.
//!
//! - [`allowed`]: loopback, the RFC 1918 ranges, IPv4 link-local, IPv6 unique-local and
//!   link-local, plus the config file's `allowed_networks` (e.g. a VPN such as Tailscale's
//!   100.64/10). Everything else is refused: public addresses, NAT64 `64:ff9b::/96` (it maps public
//!   IPv4), the deprecated site-local `fec0::/10`, CGNAT 100.64/10 unless listed, and, even when a
//!   listed network covers them, the unspecified, multicast and broadcast addresses. An IPv4
//!   address mapped into IPv6 (`::ffff:a.b.c.d`) counts as that IPv4 address.
//! - [`forwarding_problem`]: the forwarding headers (`guard::PROXY_HEADERS`) are deny-only. When
//!   one names an address outside the rule, or holds an entry this cannot read (`unknown`, an
//!   obfuscated `_token`, garbage), the request is refused. A header never grants anything: a
//!   private address in it does not make a request local, and any client can send one.
//!
//! The router applies both to every request (`guard::network_rule`) and the accept loop applies
//! [`allowed`] to every connection (`guard::admit`).

use super::guard::PROXY_HEADERS;
use axum::http::HeaderMap;
use ipnet::IpNet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Whether this server answers a client at `ip`, with `extra` the parsed `allowed_networks`.
pub(crate) fn allowed(ip: IpAddr, extra: &[IpNet]) -> bool {
    let ip = ip.to_canonical();
    if ip.is_unspecified() || ip.is_multicast() || ip == IpAddr::V4(Ipv4Addr::BROADCAST) {
        return false;
    }
    local(ip) || extra.iter().any(|net| net.contains(&ip))
}

/// The built-in local ranges, for a canonical address.
fn local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            // 127/8, 10/8, 172.16/12, 192.168/16, 169.254/16.
            v4.is_loopback() || v4.is_private() || v4.is_link_local()
        }
        IpAddr::V6(v6) => v6.is_loopback() || unique_local(v6) || link_local(v6),
    }
}

/// `fc00::/7`.
fn unique_local(ip: Ipv6Addr) -> bool {
    ip.segments()[0] & 0xfe00 == 0xfc00
}

/// `fe80::/10`.
fn link_local(ip: Ipv6Addr) -> bool {
    ip.segments()[0] & 0xffc0 == 0xfe80
}

/// What is wrong with a forwarding header. The value itself is never kept: it is the client's
/// text and goes into no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Problem {
    /// It names this address, outside the rule.
    Outside(IpAddr),
    /// It says `unknown`.
    Unknown,
    /// It holds an obfuscated identifier (`_token`).
    Obfuscated,
    /// It holds something that is no address.
    Unreadable,
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Problem::Outside(_) => write!(f, "names a client outside the local network"),
            Problem::Unknown => write!(f, "names an unknown client"),
            Problem::Obfuscated => write!(f, "names a hidden (obfuscated) client"),
            Problem::Unreadable => write!(f, "holds a client address that cannot be read"),
        }
    }
}

/// The first forwarding header (its name, from `guard::PROXY_HEADERS`) that refuses the request,
/// and why; `None` when every address in every occurrence of every such header is [`allowed`].
pub(crate) fn forwarding_problem(
    headers: &HeaderMap,
    extra: &[IpNet],
) -> Option<(&'static str, Problem)> {
    for name in PROXY_HEADERS {
        for value in headers.get_all(name) {
            let Ok(text) = value.to_str() else {
                return Some((name, Problem::Unreadable));
            };
            let nodes = if name == "forwarded" {
                forwarded_nodes(text)
            } else {
                list_nodes(text)
            };
            let problem = match nodes {
                Err(p) => Some(p),
                Ok(ips) => ips
                    .into_iter()
                    .find(|ip| !allowed(*ip, extra))
                    .map(|ip| Problem::Outside(ip.to_canonical())),
            };
            if let Some(p) = problem {
                return Some((name, p));
            }
        }
    }
    None
}

/// The addresses of an `X-Forwarded-For` / `X-Real-IP` value: a comma list of nodes, every one
/// of which must be an address (an empty entry is unreadable too).
fn list_nodes(text: &str) -> Result<Vec<IpAddr>, Problem> {
    text.split(',').map(|n| node(n.trim())).collect()
}

/// The `for=` addresses of a `Forwarded` value (RFC 7239): comma-separated elements of
/// `;`-separated `key=value` pairs, keys in any case, values a token or a quoted string. Empty
/// list elements are allowed (the RFC's list syntax); a pair without `=` or an empty `for` is
/// unreadable. Other keys (`by`, `proto`, `host`) are not looked at.
fn forwarded_nodes(text: &str) -> Result<Vec<IpAddr>, Problem> {
    let mut out = Vec::new();
    for pair in split_outside_quotes(text)? {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (key, value) = pair.split_once('=').ok_or(Problem::Unreadable)?;
        if !key.trim().eq_ignore_ascii_case("for") {
            continue;
        }
        out.push(node(&unquote(value.trim())?)?);
    }
    Ok(out)
}

/// `text` split at every `,` and `;` that is not inside a quoted string. An unterminated quote
/// is unreadable.
fn split_outside_quotes(text: &str) -> Result<Vec<&str>, Problem> {
    let mut parts = Vec::new();
    let (mut start, mut quoted, mut escaped) = (0, false, false);
    for (i, c) in text.char_indices() {
        if escaped {
            escaped = false;
        } else if quoted && c == '\\' {
            escaped = true;
        } else if c == '"' {
            quoted = !quoted;
        } else if !quoted && (c == ',' || c == ';') {
            parts.push(&text[start..i]);
            start = i + 1;
        }
    }
    if quoted {
        return Err(Problem::Unreadable);
    }
    parts.push(&text[start..]);
    Ok(parts)
}

/// A token, or the content of a quoted string with its `\` escapes undone.
fn unquote(value: &str) -> Result<String, Problem> {
    let Some(inner) = value.strip_prefix('"') else {
        return if value.contains('"') {
            Err(Problem::Unreadable)
        } else {
            Ok(value.to_string())
        };
    };
    let inner = inner.strip_suffix('"').ok_or(Problem::Unreadable)?;
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => out.push(chars.next().ok_or(Problem::Unreadable)?),
            '"' => return Err(Problem::Unreadable),
            c => out.push(c),
        }
    }
    Ok(out)
}

/// One node: `a.b.c.d`, `a.b.c.d:port`, `v6`, `[v6]` or `[v6]:port` (a `%zone` of an IPv6
/// address is dropped), `unknown` in any case, or an obfuscated `_token`.
fn node(text: &str) -> Result<IpAddr, Problem> {
    if text.eq_ignore_ascii_case("unknown") {
        return Err(Problem::Unknown);
    }
    if text.starts_with('_') {
        return Err(Problem::Obfuscated);
    }
    if let Ok(ip) = text.parse::<IpAddr>() {
        return Ok(ip);
    }
    let port_ok = |p: &str| !p.is_empty() && p.len() <= 5 && p.bytes().all(|b| b.is_ascii_digit());
    if let Some(rest) = text.strip_prefix('[') {
        let (inside, after) = rest.split_once(']').ok_or(Problem::Unreadable)?;
        if !after.is_empty() && !after.strip_prefix(':').is_some_and(port_ok) {
            return Err(Problem::Unreadable);
        }
        return v6_with_zone(inside).map(IpAddr::V6);
    }
    if let Some((host, port)) = text.split_once(':')
        && !port.contains(':')
    {
        let v4 = host.parse::<Ipv4Addr>().map_err(|_| Problem::Unreadable)?;
        return if port_ok(port) {
            Ok(IpAddr::V4(v4))
        } else {
            Err(Problem::Unreadable)
        };
    }
    v6_with_zone(text).map(IpAddr::V6)
}

/// An IPv6 address, with an optional non-empty `%zone` that is dropped.
fn v6_with_zone(text: &str) -> Result<Ipv6Addr, Problem> {
    let (addr, zone) = text.split_once('%').unwrap_or((text, "x"));
    if zone.is_empty() {
        return Err(Problem::Unreadable);
    }
    addr.parse().map_err(|_| Problem::Unreadable)
}

/// Why a request is refused ([`refusal`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The request has no peer address (it did not come through the listener).
    NoPeer,
    /// The peer is outside the rule.
    Peer(IpAddr),
    /// The peer is inside, but a forwarding header (named) refuses the request.
    Header {
        peer: IpAddr,
        header: &'static str,
        problem: Problem,
    },
}

/// The rule for one request: its peer, then its forwarding headers. `None` passes.
pub(crate) fn refusal(
    peer: Option<IpAddr>,
    headers: &HeaderMap,
    extra: &[IpNet],
) -> Option<Refusal> {
    let Some(peer) = peer else {
        return Some(Refusal::NoPeer);
    };
    if !allowed(peer, extra) {
        return Some(Refusal::Peer(peer.to_canonical()));
    }
    forwarding_problem(headers, extra).map(|(header, problem)| Refusal::Header {
        peer: peer.to_canonical(),
        header,
        problem,
    })
}

/// What the refused client is told: what happened and what would let it in. Names the peer
/// address (the client knows it anyway) and the header, never a header's value.
pub(crate) fn refusal_message(r: Refusal) -> String {
    const FIX: &str = "If this client should be served (a VPN such as Tailscale, or another \
                       network of your own), add its network to \"allowed_networks\" in the \
                       config file on the server, for example \"100.64.0.0/10\"; it applies at \
                       once.";
    match r {
        Refusal::NoPeer => format!(
            "Refused: this server only answers its local network, and this request has no \
             client address to check. {FIX}"
        ),
        Refusal::Peer(ip) => format!(
            "Refused: this server only answers its local network (this machine, private and \
             link-local addresses), and your address ({ip}) is outside it. {FIX}"
        ),
        Refusal::Header {
            peer,
            header,
            problem,
        } => format!(
            "Refused: this server only answers its local network, and this request (from \
             {peer}) has a {header} header that {problem}. A proxy in front of this server must \
             forward clients of the local network only. {FIX}"
        ),
    }
}

#[cfg(test)]
mod router_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    fn nets(list: &[&str]) -> Vec<IpNet> {
        list.iter().map(|n| n.parse().unwrap()).collect()
    }

    #[test]
    fn the_local_ranges_are_allowed_and_nothing_else() {
        for a in [
            "127.0.0.1",
            "127.255.255.254",
            "10.0.0.0",
            "10.255.255.255",
            "172.16.0.0",
            "172.20.1.2",
            "172.31.255.255",
            "192.168.0.0",
            "192.168.1.2",
            "192.168.255.255",
            "169.254.0.1",
            "169.254.255.254",
            "::1",
            "fe80::1",
            "febf:ffff::1",
            "fc00::1",
            "fd00::1",
            "fdff:ffff::1",
            "::ffff:192.168.1.2",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.5",
        ] {
            assert!(allowed(ip(a), &[]), "{a} should be allowed");
        }
        for a in [
            "8.8.8.8",
            "1.1.1.1",
            "203.0.113.5",
            "9.255.255.255",
            "11.0.0.0",
            "172.15.255.255",
            "172.32.0.0",
            "192.167.255.255",
            "192.169.0.0",
            "169.253.255.255",
            "169.255.0.0",
            "100.64.0.1",
            "100.127.255.254",
            "0.0.0.0",
            "255.255.255.255",
            "224.0.0.1",
            "239.255.255.250",
            "::",
            "::ffff:8.8.8.8",
            "::ffff:0.0.0.0",
            // IPv4-compatible (deprecated), not mapped: not taken for 127.0.0.1.
            "::127.0.0.1",
            "64:ff9b::808:808",
            "64:ff9b::c0a8:102",
            "2001:db8::1",
            "2606:4700::1111",
            "fec0::1",
            "fe7f:ffff::1",
            "fec0:ffff::1",
            "fbff:ffff::1",
            "fe00::1",
            "ff02::1",
            "ff05::1:3",
        ] {
            assert!(!allowed(ip(a), &[]), "{a} should be refused");
        }
    }

    #[test]
    fn listed_networks_add_to_the_rule() {
        let tailscale = nets(&["100.64.0.0/10", "fd7a:115c:a1e0::/48"]);
        assert!(!allowed(ip("100.64.0.1"), &[]));
        assert!(allowed(ip("100.64.0.1"), &tailscale));
        assert!(allowed(ip("::ffff:100.100.1.2"), &tailscale));
        assert!(allowed(ip("100.127.255.255"), &tailscale));
        assert!(!allowed(ip("100.128.0.0"), &tailscale));
        assert!(!allowed(ip("8.8.8.8"), &tailscale));
        // A single address (`/32`) and an IPv6 one.
        assert!(allowed(ip("203.0.113.5"), &nets(&["203.0.113.5/32"])));
        assert!(!allowed(ip("203.0.113.6"), &nets(&["203.0.113.5/32"])));
        assert!(allowed(ip("2001:db8::1"), &nets(&["2001:db8::/32"])));
        // A listed network never opens the unspecified, multicast or broadcast addresses.
        let wide = nets(&["0.0.0.0/1", "128.0.0.0/1", "::/1", "8000::/1"]);
        assert!(allowed(ip("8.8.8.8"), &wide));
        for a in ["0.0.0.0", "255.255.255.255", "224.0.0.1", "::", "ff02::1"] {
            assert!(!allowed(ip(a), &wide), "{a}");
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    fn check(pairs: &[(&str, &str)]) -> Option<(&'static str, Problem)> {
        forwarding_problem(&headers(pairs), &[])
    }

    #[test]
    fn private_forwarded_addresses_pass() {
        assert_eq!(check(&[]), None);
        for (name, value) in [
            ("x-forwarded-for", "192.168.1.9"),
            ("X-Forwarded-For", "192.168.1.9, 10.0.0.1 ,127.0.0.1"),
            ("x-forwarded-for", "192.168.1.9:51234"),
            ("x-forwarded-for", "fd00::1"),
            ("x-forwarded-for", "[fd00::1]:443"),
            ("x-forwarded-for", "[fe80::1%eth0]"),
            ("x-forwarded-for", "fe80::1%25"),
            ("x-forwarded-for", "::ffff:192.168.1.9"),
            ("x-real-ip", "10.1.2.3"),
            ("forwarded", "for=192.168.1.9"),
            ("Forwarded", "For=192.168.1.9;proto=https;by=10.0.0.1"),
            ("forwarded", "for=\"[fd00::1]:4711\""),
            ("forwarded", "for=\"192.168.1.9:80\", for=10.0.0.2"),
            ("forwarded", "FOR=\"10.0.0.2\";host=\"a,b;c\""),
            ("forwarded", "proto=https;host=example.test"),
            ("forwarded", "for=10.0.0.2, , for=10.0.0.3"),
            ("forwarded", "for=\"\\1\\0.0.0.4\""),
        ] {
            assert_eq!(check(&[(name, value)]), None, "{name}: {value}");
        }
        // A listed network passes in a header too.
        let h = headers(&[("x-forwarded-for", "100.64.0.9")]);
        assert_eq!(forwarding_problem(&h, &nets(&["100.64.0.0/10"])), None);
        assert!(forwarding_problem(&h, &[]).is_some());
    }

    #[test]
    fn public_or_unreadable_forwarded_addresses_refuse() {
        let outside = |a: &str| Problem::Outside(ip(a));
        for (pairs, want) in [
            (
                vec![("x-forwarded-for", "8.8.8.8")],
                ("x-forwarded-for", outside("8.8.8.8")),
            ),
            (
                vec![("x-forwarded-for", "192.168.1.9, 8.8.8.8")],
                ("x-forwarded-for", outside("8.8.8.8")),
            ),
            (
                vec![("x-forwarded-for", "8.8.8.8:443")],
                ("x-forwarded-for", outside("8.8.8.8")),
            ),
            // The second occurrence of a header counts as much as the first.
            (
                vec![
                    ("x-forwarded-for", "192.168.1.9"),
                    ("x-forwarded-for", "203.0.113.5"),
                ],
                ("x-forwarded-for", outside("203.0.113.5")),
            ),
            (
                vec![("x-forwarded-for", "[2001:db8::1]:80")],
                ("x-forwarded-for", outside("2001:db8::1")),
            ),
            (
                vec![("x-forwarded-for", "::ffff:8.8.8.8")],
                ("x-forwarded-for", outside("8.8.8.8")),
            ),
            (
                vec![("x-forwarded-for", "64:ff9b::808:808")],
                ("x-forwarded-for", outside("64:ff9b::808:808")),
            ),
            (
                vec![("x-forwarded-for", "unknown")],
                ("x-forwarded-for", Problem::Unknown),
            ),
            (
                vec![("x-forwarded-for", "10.0.0.1,")],
                ("x-forwarded-for", Problem::Unreadable),
            ),
            (
                vec![("x-forwarded-for", "")],
                ("x-forwarded-for", Problem::Unreadable),
            ),
            (
                vec![("x-forwarded-for", "localhost")],
                ("x-forwarded-for", Problem::Unreadable),
            ),
            (
                vec![("x-forwarded-for", "10.0.0.1:port")],
                ("x-forwarded-for", Problem::Unreadable),
            ),
            (
                vec![("x-forwarded-for", "[fd00::1]x")],
                ("x-forwarded-for", Problem::Unreadable),
            ),
            (
                vec![("x-forwarded-for", "[fd00::1")],
                ("x-forwarded-for", Problem::Unreadable),
            ),
            (
                vec![("x-forwarded-for", "fe80::1%")],
                ("x-forwarded-for", Problem::Unreadable),
            ),
            (
                vec![("x-forwarded-for", "10.0.0.1:123456")],
                ("x-forwarded-for", Problem::Unreadable),
            ),
            (
                vec![("x-real-ip", "8.8.4.4")],
                ("x-real-ip", outside("8.8.4.4")),
            ),
            (
                vec![("x-real-ip", "garbage")],
                ("x-real-ip", Problem::Unreadable),
            ),
            (
                vec![("forwarded", "for=8.8.8.8")],
                ("forwarded", outside("8.8.8.8")),
            ),
            (
                vec![("Forwarded", "FoR=\"[2001:db8:cafe::17]:4711\"")],
                ("forwarded", outside("2001:db8:cafe::17")),
            ),
            (
                vec![("forwarded", "for=192.168.1.9;proto=http, for=198.51.100.17")],
                ("forwarded", outside("198.51.100.17")),
            ),
            (
                vec![("forwarded", "proto=https;for=10.0.0.1;for=8.8.8.8")],
                ("forwarded", outside("8.8.8.8")),
            ),
            (
                vec![("forwarded", "for=unknown")],
                ("forwarded", Problem::Unknown),
            ),
            (
                vec![("forwarded", "for=\"UNKNOWN\"")],
                ("forwarded", Problem::Unknown),
            ),
            (
                vec![("forwarded", "for=_hidden")],
                ("forwarded", Problem::Obfuscated),
            ),
            (
                vec![("forwarded", "for=\"_SEVKISEK\"")],
                ("forwarded", Problem::Obfuscated),
            ),
            (
                vec![("forwarded", "for=\"[fd00::1]:_obfport\"")],
                ("forwarded", Problem::Unreadable),
            ),
            (
                vec![("forwarded", "for=")],
                ("forwarded", Problem::Unreadable),
            ),
            (
                vec![("forwarded", "for=\"10.0.0.1")],
                ("forwarded", Problem::Unreadable),
            ),
            (
                vec![("forwarded", "for=10.0.0.1\"x\"")],
                ("forwarded", Problem::Unreadable),
            ),
            (
                vec![("forwarded", "garbage")],
                ("forwarded", Problem::Unreadable),
            ),
            (
                vec![("forwarded", "for=10.0.0.1"), ("forwarded", "for=8.8.8.8")],
                ("forwarded", outside("8.8.8.8")),
            ),
            // A private header does not cover a public one of another name.
            (
                vec![("x-real-ip", "10.0.0.1"), ("forwarded", "for=8.8.8.8")],
                ("forwarded", outside("8.8.8.8")),
            ),
        ] {
            assert_eq!(check(&pairs), Some(want), "{pairs:?}");
        }
        // A value that is not text.
        let mut h = HeaderMap::new();
        h.insert(
            "x-forwarded-for",
            HeaderValue::from_bytes(b"10.0.0.1\xff").unwrap(),
        );
        assert_eq!(
            forwarding_problem(&h, &[]),
            Some(("x-forwarded-for", Problem::Unreadable))
        );
    }

    #[test]
    fn a_request_needs_a_peer_inside_and_headers_that_agree() {
        let none = HeaderMap::new();
        assert_eq!(refusal(None, &none, &[]), Some(Refusal::NoPeer));
        assert_eq!(refusal(Some(ip("192.168.1.2")), &none, &[]), None);
        assert_eq!(
            refusal(Some(ip("::ffff:203.0.113.5")), &none, &[]),
            Some(Refusal::Peer(ip("203.0.113.5")))
        );
        let h = headers(&[("x-forwarded-for", "8.8.8.8")]);
        assert_eq!(
            refusal(Some(ip("127.0.0.1")), &h, &[]),
            Some(Refusal::Header {
                peer: ip("127.0.0.1"),
                header: "x-forwarded-for",
                problem: Problem::Outside(ip("8.8.8.8")),
            })
        );
        // A private header never lets an outside peer in.
        let h = headers(&[("x-forwarded-for", "192.168.1.9")]);
        assert_eq!(
            refusal(Some(ip("203.0.113.5")), &h, &[]),
            Some(Refusal::Peer(ip("203.0.113.5")))
        );
        assert_eq!(refusal(Some(ip("127.0.0.1")), &h, &[]), None);
    }

    #[test]
    fn refusals_name_the_peer_and_the_fix_but_no_header_value() {
        for r in [
            Refusal::NoPeer,
            Refusal::Peer(ip("203.0.113.5")),
            Refusal::Header {
                peer: ip("127.0.0.1"),
                header: "forwarded",
                problem: Problem::Outside(ip("8.8.8.8")),
            },
        ] {
            let m = refusal_message(r);
            assert!(
                m.contains("allowed_networks") && m.contains("local network"),
                "{m}"
            );
            assert!(!m.contains("8.8.8.8"), "{m}");
        }
        assert!(refusal_message(Refusal::Peer(ip("203.0.113.5"))).contains("203.0.113.5"));
        let m = refusal_message(Refusal::Header {
            peer: ip("127.0.0.1"),
            header: "forwarded",
            problem: Problem::Unknown,
        });
        assert!(
            m.contains("127.0.0.1") && m.contains("forwarded header"),
            "{m}"
        );
    }

    #[test]
    fn problems_never_repeat_the_value() {
        for p in [
            Problem::Outside(ip("8.8.8.8")),
            Problem::Unknown,
            Problem::Obfuscated,
            Problem::Unreadable,
        ] {
            assert!(!p.to_string().contains("8.8"), "{p}");
        }
    }
}
