//! Tests of the HTTPS certificate: what a generated one holds (checked by rustls handshakes with
//! the certificate as the only root, and by its DER), when it is kept, replaced or renewed, own
//! files, and the resolver's reload.

use super::*;
use std::sync::atomic::AtomicU64;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch() -> Scratch {
    let d = std::env::temp_dir().join(format!("bo_tls_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    Scratch(d)
}

/// What the tests' certificates name: never the host's real name or addresses.
fn identity() -> Identity {
    Identity {
        hostname: Some("Prism-Box".into()),
        ips: vec![
            "192.168.1.50".parse().unwrap(),
            "fd00::50".parse().unwrap(),
            "127.0.0.1".parse().unwrap(),
            "::ffff:192.168.1.50".parse().unwrap(),
        ],
    }
}

fn identity_fn() -> IdentityFn {
    Arc::new(identity)
}

fn der_of(pem: &str) -> CertificateDer<'static> {
    CertificateDer::from_pem_slice(pem.as_bytes()).unwrap()
}

fn secs(t: SystemTime) -> u64 {
    t.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
}

/// A movable clock.
#[derive(Clone)]
struct TestClock(Arc<AtomicU64>);

impl TestClock {
    fn at(t: SystemTime) -> Self {
        Self(Arc::new(AtomicU64::new(secs(t))))
    }
    fn set(&self, t: SystemTime) {
        self.0.store(secs(t), Ordering::SeqCst);
    }
    fn clock(&self) -> Clock {
        let c = self.0.clone();
        Arc::new(move || SystemTime::UNIX_EPOCH + Duration::from_secs(c.load(Ordering::SeqCst)))
    }
}

fn generated_source(root: &Scratch) -> Source {
    Source::Generated {
        data_root: root.0.clone(),
    }
}

/// A TLS handshake over an in-memory pipe: the server serves `key`, the client trusts only
/// `root` and asks for `name`. The negotiated ALPN protocol, or the client's error.
async fn handshake(
    key: Arc<CertifiedKey>,
    root: &CertificateDer<'static>,
    name: &str,
    alpn: &[&[u8]],
) -> std::result::Result<Option<Vec<u8>>, String> {
    #[derive(Debug)]
    struct Fixed(Arc<CertifiedKey>);
    impl rustls::server::ResolvesServerCert for Fixed {
        fn resolve(&self, _: rustls::server::ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
            Some(self.0.clone())
        }
    }
    let mut server = rustls::ServerConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_cert_resolver(Arc::new(Fixed(key)));
    server.alpn_protocols = vec![b"http/1.1".to_vec()];
    let client = client_config(root, alpn);
    let (a, b) = tokio::io::duplex(64 * 1024);
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server));
    let server_side = tokio::spawn(async move { acceptor.accept(a).await.map(|_| ()) });
    let connector = tokio_rustls::TlsConnector::from(client);
    let sn = rustls::pki_types::ServerName::try_from(name.to_string()).unwrap();
    let got = connector.connect(sn, b).await;
    let _ = server_side.await;
    got.map(|s| s.get_ref().1.alpn_protocol().map(<[u8]>::to_vec))
        .map_err(|e| e.to_string())
}

/// A client config trusting `root` alone, offering `alpn`.
pub(crate) fn client_config(
    root: &CertificateDer<'static>,
    alpn: &[&[u8]],
) -> Arc<rustls::ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();
    roots.add(root.clone()).unwrap();
    let mut c = rustls::ClientConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(roots)
        .with_no_client_auth();
    c.alpn_protocols = alpn.iter().map(|p| p.to_vec()).collect();
    Arc::new(c)
}

fn certified(made: &Generated) -> Arc<CertifiedKey> {
    Arc::new(
        CertifiedKey::from_der(
            vec![der_of(&made.cert_pem)],
            PrivateKeyDer::from_pem_slice(made.key_pem.as_bytes()).unwrap(),
            &provider(),
        )
        .unwrap(),
    )
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[tokio::test]
async fn a_generated_certificate_serves_every_name_it_holds_and_no_other() {
    let made = generate(&identity(), SystemTime::now()).unwrap();
    let root = der_of(&made.cert_pem);
    let key = certified(&made);
    for name in [
        "localhost",
        "127.0.0.1",
        "::1",
        "prism-box",
        "prism-box.local",
        "192.168.1.50",
        "fd00::50",
    ] {
        let alpn = handshake(key.clone(), &root, name, &[b"http/1.1"]).await;
        assert_eq!(alpn, Ok(Some(b"http/1.1".to_vec())), "{name}");
    }
    for name in ["example.com", "192.168.1.51", "prism-box.example"] {
        let e = handshake(key.clone(), &root, name, &[]).await.unwrap_err();
        assert!(e.contains("not valid for name"), "{name}: {e}");
    }
    // A client that offers only HTTP/2 is refused: the listener speaks HTTP/1.1.
    assert!(
        handshake(key.clone(), &root, "localhost", &[b"h2"])
            .await
            .is_err()
    );
}

/// The handshake above would also pass a CA certificate if webpki took one as a server's: it
/// does not, so the passing handshakes show the generated one is not a CA.
#[tokio::test]
async fn a_ca_certificate_fails_the_same_handshake() {
    let key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
    let mut params = rcgen::CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let cert = params.self_signed(&key).unwrap();
    let made = Generated {
        cert_pem: cert.pem(),
        key_pem: key.serialize_pem(),
    };
    let e = handshake(certified(&made), &der_of(&made.cert_pem), "localhost", &[])
        .await
        .unwrap_err();
    assert!(e.contains("CaUsedAsEndEntity"), "{e}");
}

#[test]
fn a_generated_certificate_has_its_validity_usage_and_name() {
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000);
    let made = generate(&identity(), now).unwrap();
    let der = der_of(&made.cert_pem);
    let (start, end) = validity(&der).unwrap();
    assert_eq!(secs(start), 1_800_000_000 - 86_400, "a day back");
    assert_eq!(secs(end), 1_800_000_000 + 730 * 86_400, "two years");
    // EKU serverAuth (1.3.6.1.5.5.7.3.1), and basicConstraints (2.5.29.19), critical, with an
    // empty value: cA false, written out (IsCa::ExplicitNoCa).
    assert!(contains(
        &der,
        &[0x06, 0x08, 0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x01]
    ));
    assert!(contains(
        &der,
        &[
            0x06, 0x03, 0x55, 0x1d, 0x13, 0x01, 0x01, 0xff, 0x04, 0x02, 0x30, 0x00
        ]
    ));
    // ECDSA P-256 key (prime256v1 1.2.840.10045.3.1.7) signed with ECDSA SHA-256.
    assert!(contains(
        &der,
        &[0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07]
    ));
    assert!(contains(
        &der,
        &[0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02]
    ));
    assert!(contains(&der, b"Blue Onyx Prism (Prism-Box)"));
    // Two certificates never share a key, and so neither a serial (taken from the key).
    let other = generate(&identity(), now).unwrap();
    assert_ne!(other.key_pem, made.key_pem);
    assert_ne!(other.cert_pem, made.cert_pem);
    // No host name: still a certificate, for localhost and the addresses.
    let bare = generate(&Identity::default(), now).unwrap();
    assert!(contains(
        &der_of(&bare.cert_pem),
        b"Blue Onyx Prism (this computer)"
    ));
}

#[test]
fn validity_reads_utc_and_generalized_times_and_refuses_garbage() {
    // From 2050 on X.509 writes GeneralizedTime.
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(2_540_000_000); // 2050-06
    let der = der_of(&generate(&Identity::default(), now).unwrap().cert_pem);
    let (start, end) = validity(&der).unwrap();
    assert_eq!(secs(start), 2_540_000_000 - 86_400);
    assert_eq!(secs(end), 2_540_000_000 + 730 * 86_400);
    assert_eq!(validity(b""), None);
    assert_eq!(validity(&[0x30, 0x03, 0x02, 0x01]), None);
    assert_eq!(validity(&der[..der.len() / 2]), None);
    assert_eq!(
        der_time(0x17, b"991231235959Z").map(secs),
        Some(946_684_799)
    );
    assert_eq!(
        der_time(0x17, b"000101000000Z").map(secs),
        Some(946_684_800)
    );
    assert_eq!(der_time(0x18, b"19700101000000Z").map(secs), Some(0));
    assert_eq!(der_time(0x17, b"001301000000Z"), None, "month 13");
    assert_eq!(der_time(0x17, b"0001010000Z"), None, "too short");
}

#[test]
fn names_keep_only_valid_dns_labels() {
    assert_eq!(
        dns_names(Some("Prism-Box")),
        ["localhost", "prism-box", "prism-box.local"]
    );
    assert_eq!(
        dns_names(Some("nas.example.com.")),
        ["localhost", "nas.example.com", "nas.local"]
    );
    assert_eq!(dns_names(Some("box.local")), ["localhost", "box.local"]);
    for bad in [
        "",
        " ",
        "under_score",
        "-dash",
        "dash-",
        "1234",
        "10.0.0.1",
        "a..b",
        "caf\u{e9}",
        &"x".repeat(64),
    ] {
        assert_eq!(dns_names(Some(bad)), ["localhost"], "{bad:?}");
    }
    assert_eq!(dns_names(None), ["localhost"]);
    assert_eq!(
        ip_names(&identity().ips),
        [
            "127.0.0.1".parse::<IpAddr>().unwrap(),
            "::1".parse().unwrap(),
            "192.168.1.50".parse().unwrap(),
            "fd00::50".parse().unwrap(),
        ]
    );
}

#[test]
fn fingerprint_is_upper_case_hex_pairs_joined_by_colons() {
    // SHA-256 of nothing.
    assert_eq!(
        fingerprint_sha256(b""),
        "E3:B0:C4:42:98:FC:1C:14:9A:FB:F4:C8:99:6F:B9:24:27:AE:41:E4:64:9B:93:4C:A4:95:99:1B:\
         78:52:B8:55"
    );
    let f = fingerprint_sha256(&der_of(
        &generate(&identity(), SystemTime::now()).unwrap().cert_pem,
    ));
    assert_eq!(f.len(), 32 * 3 - 1);
    assert!(f.split(':').all(|p| {
        p.len() == 2
            && p.bytes()
                .all(|b| b.is_ascii_digit() || (b'A'..=b'F').contains(&b))
    }));
}

#[test]
fn the_generated_pair_is_made_once_and_kept() {
    let root = scratch();
    let source = generated_source(&root);
    let now = SystemTime::now();
    let first = load(&source, &identity, now, Mode::Serve).unwrap();
    let (cert, key) = source.paths();
    assert!(cert.is_file() && key.is_file());
    assert_eq!(cert, root.0.join("auth").join("tls").join("cert.pem"));
    // Another identity (new addresses, a renamed computer) does not make a new one.
    let moved = || Identity {
        hostname: Some("renamed".into()),
        ips: vec!["10.9.9.9".parse().unwrap()],
    };
    let again = load(&source, &moved, now, Mode::Serve).unwrap();
    assert_eq!(again.fingerprint, first.fingerprint);
    let reported = load(&source, &moved, now, Mode::Report).unwrap();
    assert_eq!(reported.fingerprint, first.fingerprint);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&key), 0o600);
        assert_eq!(mode(key.parent().unwrap()), 0o700);
        assert_eq!(mode(&root.0.join("auth")), 0o700);
    }
}

/// A damaged pair: the service replaces it (with a warning), `tls-fingerprint` refuses and
/// leaves it.
#[test]
fn a_damaged_pair_is_replaced_by_the_service_only() {
    let root = scratch();
    let source = generated_source(&root);
    let now = SystemTime::now();
    let first = load(&source, &identity, now, Mode::Serve).unwrap();
    let (cert, key) = source.paths();
    for (path, garbage) in [
        (&cert, "not a certificate"),
        (
            &key,
            "-----BEGIN PRIVATE KEY-----\nSECRETSECRET\n-----END PRIVATE KEY-----\n",
        ),
    ] {
        let kept = std::fs::read(path).unwrap();
        std::fs::write(path, garbage).unwrap();
        let e = format!(
            "{:#}",
            load(&source, &identity, now, Mode::Report).unwrap_err()
        );
        assert!(e.contains("tls-regenerate"), "{e}");
        assert!(!e.contains("SECRET"), "never quotes the key: {e}");
        assert_eq!(std::fs::read_to_string(path).unwrap(), garbage);
        let fresh = load(&source, &identity, now, Mode::Serve).unwrap();
        assert_ne!(fresh.fingerprint, first.fingerprint);
        std::fs::write(path, kept).unwrap();
    }
    // A key of another certificate does not go with it: also damaged.
    let other = generate(&identity(), now).unwrap();
    std::fs::write(&key, &other.key_pem).unwrap();
    let e = format!(
        "{:#}",
        load(&source, &identity, now, Mode::Report).unwrap_err()
    );
    assert!(e.contains("do not go together"), "{e}");
    let fixed = load(&source, &identity, now, Mode::Serve).unwrap();
    assert!(validity(&fixed.key.cert[0]).is_some());
}

#[test]
fn expiry_rules() {
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000);
    let left = |days: u64| Some(now + Duration::from_secs(days * 86_400));
    assert_eq!(expiry(left(400), now, true), Expiry::Fine);
    assert_eq!(expiry(left(60), now, true), Expiry::Fine);
    assert_eq!(expiry(left(59), now, true), Expiry::Soon(59));
    assert_eq!(expiry(left(30), now, true), Expiry::Soon(30));
    assert_eq!(expiry(left(29), now, true), Expiry::Renew);
    assert_eq!(expiry(Some(now - DAY), now, true), Expiry::Renew);
    assert_eq!(expiry(left(45), now, false), Expiry::Fine);
    assert_eq!(expiry(left(29), now, false), Expiry::Soon(29));
    assert_eq!(expiry(Some(now - DAY), now, false), Expiry::Expired);
    assert_eq!(expiry(None, now, true), Expiry::Fine);
}

/// Under 30 days left at start: a new certificate; 45 days left: kept (a note only).
#[test]
fn a_generated_certificate_near_its_end_is_renewed_at_start() {
    let root = scratch();
    let source = generated_source(&root);
    let now = SystemTime::now();
    let made_at = |left_days: u64| now - VALIDITY + Duration::from_secs(left_days * 86_400);
    regenerate(&root.0, &identity(), made_at(45)).unwrap();
    let kept = load(&source, &identity, now, Mode::Serve).unwrap();
    assert_eq!(
        load(&source, &identity, now, Mode::Serve)
            .unwrap()
            .fingerprint,
        kept.fingerprint
    );
    regenerate(&root.0, &identity(), made_at(20)).unwrap();
    let old = load(&source, &identity, now, Mode::Report).unwrap();
    let renewed = load(&source, &identity, now, Mode::Serve).unwrap();
    assert_ne!(renewed.fingerprint, old.fingerprint);
    let end = secs(renewed.not_after.unwrap());
    assert!(
        end.abs_diff(secs(now) + VALIDITY.as_secs()) <= 2,
        "a full two years again"
    );
}

/// The resolver serves what the CLI writes while it runs, renews by its clock, and keeps the
/// certificate in use when the files on disk break.
#[test]
fn the_resolver_follows_the_files_and_renews_by_its_clock() {
    let root = scratch();
    let source = generated_source(&root);
    let clock = TestClock::at(SystemTime::now());
    let r = Resolver::with(source.clone(), identity_fn(), clock.clock(), Duration::ZERO).unwrap();
    let first = r.fingerprint();
    assert_eq!(r.current().key.cert.len(), 1);

    // `tls-regenerate` in another process.
    let written = regenerate(&root.0, &identity(), SystemTime::now()).unwrap();
    assert_ne!(written.fingerprint, first);
    assert!(r.needs_refresh());
    r.refresh();
    assert_eq!(r.fingerprint(), written.fingerprint);
    assert!(!r.needs_refresh());

    // A broken file while running: the certificate in memory stays, and no new one is made.
    let (cert, _) = source.paths();
    std::fs::write(&cert, "half written").unwrap();
    r.refresh();
    assert_eq!(r.fingerprint(), written.fingerprint);
    assert_eq!(std::fs::read_to_string(&cert).unwrap(), "half written");
    assert!(
        !r.needs_refresh(),
        "the same broken file is not tried again"
    );
    regenerate(&root.0, &identity(), SystemTime::now()).unwrap();
    r.refresh();
    let repaired = r.fingerprint();
    assert_ne!(repaired, written.fingerprint);

    // 20 days before the end: renewed, on disk and in use.
    clock.set(SystemTime::now() + VALIDITY - Duration::from_secs(20 * 86_400));
    assert!(r.needs_refresh());
    r.refresh();
    assert_ne!(r.fingerprint(), repaired);
    let on_disk = load(&source, &identity, SystemTime::now(), Mode::Report).unwrap();
    assert_eq!(on_disk.fingerprint, r.fingerprint());
}

/// A handshake is what triggers the check: the next handshakes get the new certificate.
#[tokio::test]
async fn a_handshake_picks_up_a_new_certificate() {
    let root = scratch();
    let r = Resolver::with(
        generated_source(&root),
        identity_fn(),
        Arc::new(SystemTime::now),
        Duration::ZERO,
    )
    .unwrap();
    let config = server_config(&r).unwrap();
    assert_eq!(config.alpn_protocols, [b"http/1.1".to_vec()]);
    let written = regenerate(&root.0, &identity(), SystemTime::now()).unwrap();
    let trust = written.key.cert[0].clone();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (a, b) = tokio::io::duplex(64 * 1024);
        let acceptor = tokio_rustls::TlsAcceptor::from(config.clone());
        let server = tokio::spawn(async move { acceptor.accept(a).await.map(|_| ()) });
        let connector = tokio_rustls::TlsConnector::from(client_config(&trust, &[]));
        let ok = connector
            .connect(
                rustls::pki_types::ServerName::try_from("localhost").unwrap(),
                b,
            )
            .await
            .is_ok();
        let _ = server.await;
        if ok {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the new certificate was never served"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(r.fingerprint(), written.fingerprint);
}

/// Own files: used as given, chain included, never written, never renewed, and a pair that
/// does not load is an error naming what to do.
#[test]
fn own_files_are_used_as_given_and_never_written() {
    let root = scratch();
    let now = SystemTime::now();
    // Ten days left: a generated one would be renewed.
    let made = generate(
        &identity(),
        now - VALIDITY + Duration::from_secs(10 * 86_400),
    )
    .unwrap();
    let ca = generate(&Identity::default(), now).unwrap();
    let cert = root.0.join("mine.crt");
    let key = root.0.join("mine.key");
    std::fs::write(&cert, format!("{}{}", made.cert_pem, ca.cert_pem)).unwrap();
    std::fs::write(&key, &made.key_pem).unwrap();
    let source = Source::Files {
        cert: cert.clone(),
        key: key.clone(),
    };
    let before = (std::fs::read(&cert).unwrap(), std::fs::read(&key).unwrap());
    let stamp = stamp_of(&cert, &key);
    let clock = TestClock::at(now);
    let r = Resolver::with(source.clone(), identity_fn(), clock.clock(), Duration::ZERO).unwrap();
    assert_eq!(r.current().key.cert.len(), 2, "the chain is served");
    assert_eq!(r.fingerprint(), fingerprint_sha256(&der_of(&made.cert_pem)));
    clock.set(now + Duration::from_secs(400 * 86_400));
    r.refresh();
    r.refresh();
    assert_eq!(r.fingerprint(), fingerprint_sha256(&der_of(&made.cert_pem)));
    assert_eq!(
        (std::fs::read(&cert).unwrap(), std::fs::read(&key).unwrap()),
        before
    );
    assert_eq!(stamp_of(&cert, &key), stamp);
    assert!(!root.0.join("auth").exists(), "nothing generated");

    std::fs::remove_file(&key).unwrap();
    let e = format!("{:#}", Resolver::new(source).unwrap_err());
    assert!(e.contains("tls_key_path"), "{e}");
    assert!(e.contains("stays down"), "{e}");
    assert!(!root.0.join("auth").exists(), "nothing generated");
}

#[test]
fn the_cli_reports_and_refuses_to_regenerate_own_files() {
    let root = scratch();
    let mut config = Config {
        download_dir: Some(root.0.clone()),
        https_port: 8443,
        ..Config::default()
    };
    let text = fingerprint_report(&config).unwrap();
    let source = Source::of(&config);
    let loaded = load(&source, &identity, SystemTime::now(), Mode::Report).unwrap();
    assert!(text.contains(&loaded.fingerprint), "{text}");
    assert!(text.contains("https://localhost:8443/"), "{text}");
    let report = report_text(
        &loaded,
        &source,
        8443,
        &["192.168.1.50".parse().unwrap(), "fd00::50".parse().unwrap()],
        SystemTime::now(),
    );
    assert!(report.contains("https://192.168.1.50:8443/"), "{report}");
    assert!(report.contains("https://[fd00::50]:8443/"), "{report}");
    assert!(report.contains("Valid for 72"), "{report}");

    let made = regenerate_report(&config).unwrap();
    let after = load(&source, &identity, SystemTime::now(), Mode::Report).unwrap();
    assert_ne!(after.fingerprint, loaded.fingerprint);
    assert!(made.contains(&after.fingerprint), "{made}");

    config.tls_cert_path = Some("mine.crt".into());
    config.tls_key_path = Some("mine.key".into());
    assert_eq!(
        Source::of(&config),
        Source::Files {
            cert: root.0.join("mine.crt"),
            key: root.0.join("mine.key"),
        }
    );
    let e = regenerate_report(&config).unwrap_err().to_string();
    assert!(e.contains("never writes them"), "{e}");
    assert!(
        fingerprint_report(&config).is_err(),
        "the files do not exist"
    );
}
