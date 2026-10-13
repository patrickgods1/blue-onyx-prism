//! Hostile-input tests for the archive drivers: hand-crafted zip and tar.gz files (escaping
//! paths, links, lying size fields, truncation, duplicate / long / non-UTF-8 names) and a
//! seeded pseudo-random byte soup. Every case must return without panicking, write nothing
//! outside the destination, leave no `.partial` file, and (when it succeeds) write only the
//! whitelisted names it reports.

use super::test_util::*;
use super::*;

/// Layout of one test: `root/outside` must stay empty, `root/dest-N` receives extractions.
struct Sandbox {
    root: TempDir,
    runs: usize,
}

impl Sandbox {
    fn new(tag: &str) -> Self {
        let root = TempDir::new(tag);
        std::fs::create_dir_all(root.join("outside")).unwrap();
        std::fs::create_dir_all(root.join("archives")).unwrap();
        Self { root, runs: 0 }
    }

    fn archive(&self, name: &str) -> std::path::PathBuf {
        self.root.join("archives").join(name)
    }

    /// Extract `archive` into a fresh `dest-N` and check the invariants; returns the result.
    fn run(
        &mut self,
        archive: &Path,
        kind: ArchiveKind,
        layout: Layout,
        limits: Limits,
    ) -> Result<BTreeSet<String>> {
        self.runs += 1;
        let dest = self.root.join(format!("dest-{}", self.runs));
        let flavor = (layout == Layout::OrtRelease).then_some(Flavor::Cpu);
        let r = extract_part_with(archive, kind, layout, flavor, LINUX, &dest, limits);
        self.check(&dest, r.as_ref().ok(), limits);
        r
    }

    fn check(&self, dest: &Path, got: Option<&BTreeSet<String>>, limits: Limits) {
        let mut files = Vec::new();
        walk(&self.root, &self.root, &mut files);
        for (rel, _) in &files {
            assert!(!rel.ends_with(".partial"), "left behind: {rel}");
            assert!(!rel.starts_with("outside/"), "written outside: {rel}");
            assert!(
                rel.starts_with("archives/") || rel.starts_with("dest-"),
                "unexpected file {rel}"
            );
        }
        let mut in_dest = Vec::new();
        walk(dest, dest, &mut in_dest);
        for (rel, len) in &in_dest {
            assert!(*len <= limits.entry_bytes, "{rel}: {len} bytes");
        }
        if let Some(got) = got {
            for (rel, _) in &in_dest {
                assert!(got.contains(rel), "{rel} written but not reported: {got:?}");
            }
        }
    }
}

/// Every non-directory under `dir` as (path relative to `base` with `/`, size); symlinks are
/// listed (size 0), never followed.
fn walk(base: &Path, dir: &Path, out: &mut Vec<(String, u64)>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd {
        let e = e.unwrap();
        let meta = std::fs::symlink_metadata(e.path()).unwrap();
        if meta.is_dir() {
            walk(base, &e.path(), out);
            continue;
        }
        let rel = e.path().strip_prefix(base).unwrap().to_path_buf();
        let rel = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        let len = if meta.file_type().is_symlink() {
            0
        } else {
            meta.len()
        };
        out.push((rel, len));
    }
}

/// Deterministic xorshift64 generator for the byte soup.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// One raw tar entry; names and link targets are written byte-for-byte into the header (the
/// tar builder refuses `..` and absolute names), with a GNU long-name record when the name is
/// longer than the 100-byte header field.
enum Raw<'a> {
    File(&'a [u8], &'a [u8]),
    Symlink(&'a [u8], &'a [u8]),
    Hardlink(&'a [u8], &'a [u8]),
}

fn raw_tgz(path: &Path, entries: &[Raw<'_>]) {
    let f = std::fs::File::create(path).unwrap();
    let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
    let mut tar = tar::Builder::new(gz);
    let header = |name: &[u8], ty: tar::EntryType, size: u64, link: &[u8]| {
        let mut h = tar::Header::new_gnu();
        {
            let g = h.as_gnu_mut().unwrap();
            let n = name.len().min(g.name.len());
            g.name[..n].copy_from_slice(&name[..n]);
            g.linkname[..link.len()].copy_from_slice(link);
        }
        h.set_entry_type(ty);
        h.set_size(size);
        h.set_mode(0o644);
        h.set_cksum();
        h
    };
    for e in entries {
        let (name, ty, data, link): (&[u8], _, &[u8], &[u8]) = match e {
            Raw::File(n, d) => (n, tar::EntryType::Regular, d, b""),
            Raw::Symlink(n, t) => (n, tar::EntryType::Symlink, b"", t),
            Raw::Hardlink(n, t) => (n, tar::EntryType::Link, b"", t),
        };
        if name.len() > 100 {
            let mut long = name.to_vec();
            long.push(0);
            let h = header(
                b"././@LongLink",
                tar::EntryType::GNULongName,
                long.len() as u64,
                b"",
            );
            tar.append(&h, long.as_slice()).unwrap();
        }
        let h = header(name, ty, data.len() as u64, link);
        tar.append(&h, data).unwrap();
    }
    tar.into_inner().unwrap().finish().unwrap();
}

/// A zip whose entry names are written byte-for-byte: each entry is stored under a unique
/// placeholder of the same length (the writer normalises or refuses hostile names) and the
/// name is patched into the central directory and the local header afterwards. A non-ASCII
/// name gets the UTF-8 flag, as a real writer would set it.
fn raw_zip(path: &Path, entries: &[(&[u8], &[u8])]) {
    {
        let f = std::fs::File::create(path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        for (i, (name, data)) in entries.iter().enumerate() {
            let placeholder: String = (0..name.len())
                .map(|k| {
                    if k == 0 {
                        (b'a' + i as u8) as char
                    } else {
                        'q'
                    }
                })
                .collect();
            z.start_file(placeholder, zip::write::SimpleFileOptions::default())
                .unwrap();
            z.write_all(data).unwrap();
        }
        z.finish().unwrap();
    }
    let mut b = std::fs::read(path).unwrap();
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as usize;
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let eocd = b
        .windows(4)
        .rposition(|w| w == [0x50, 0x4b, 0x05, 0x06])
        .unwrap();
    let mut at = u32_at(&b, eocd + 16) as usize;
    for (name, _) in entries {
        assert_eq!(u32_at(&b, at), 0x0201_4b50, "central directory record");
        let name_len = u16_at(&b, at + 28);
        let skip = name_len + u16_at(&b, at + 30) + u16_at(&b, at + 32);
        let local = u32_at(&b, at + 42) as usize;
        assert_eq!(name_len, name.len());
        b[at + 46..at + 46 + name_len].copy_from_slice(name);
        b[local + 30..local + 30 + name_len].copy_from_slice(name);
        if !name.is_ascii() {
            // General purpose flag bit 11 (UTF-8): bit 3 of the flag's high byte.
            b[at + 9] |= 0x08;
            b[local + 7] |= 0x08;
        }
        at += 46 + skip;
    }
    std::fs::write(path, b).unwrap();
}

/// Overwrite the uncompressed-size fields (central directory and local header) of every entry.
fn zip_claim_size(path: &Path, size: u32) {
    let mut b = std::fs::read(path).unwrap();
    let mut i = 0;
    while i + 4 <= b.len() {
        match &b[i..i + 4] {
            [0x50, 0x4b, 0x01, 0x02] => b[i + 24..i + 28].copy_from_slice(&size.to_le_bytes()),
            [0x50, 0x4b, 0x03, 0x04] => b[i + 22..i + 26].copy_from_slice(&size.to_le_bytes()),
            _ => {}
        }
        i += 1;
    }
    std::fs::write(path, b).unwrap();
}

const WHEEL_LIB: &[u8] = b"nvidia/cudnn/lib/libcudnn.so.9";
const ORT_LIB: &[u8] = b"ort-1/lib/libonnxruntime.so";

fn small() -> Limits {
    Limits {
        entry_bytes: 64 << 10,
        total_bytes: 256 << 10,
    }
}

#[test]
fn zip_escaping_names_are_refused() {
    let mut sb = Sandbox::new("fuzz-zip-names");
    for (name, why) in [
        (&b"nvidia/../../outside/libx.so.1"[..], "`..`"),
        (b"../outside/x", "`..`"),
        (b"/outside/libcudnn.so.9", "absolute path"),
        (b"\\outside\\libcudnn.so.9", "absolute path"),
        (b"C:/Windows/evil.dll", "drive letter"),
        (b"c:\\nvidia\\cudnn\\lib\\libcudnn.so.9", "drive letter"),
    ] {
        let a = sb.archive("names.whl");
        raw_zip(&a, &[(WHEEL_LIB, b"ok"), (name, b"evil")]);
        refused(
            sb.run(&a, ArchiveKind::Zip, Layout::NvidiaWheel, small()),
            why,
        );
    }
}

/// Names the whitelist cannot map (`.`, empty, a bare directory, `./`) are skipped silently.
#[test]
fn zip_dot_and_empty_names_are_skipped() {
    let mut sb = Sandbox::new("fuzz-zip-dot");
    for name in [&b"."[..], b"./", b"nvidia/./.", b"nvidia//lib/", b"x"] {
        let a = sb.archive("dot.whl");
        raw_zip(&a, &[(name, b"data"), (WHEEL_LIB, b"ok")]);
        let got = sb
            .run(&a, ArchiveKind::Zip, Layout::NvidiaWheel, small())
            .unwrap_or_else(|e| panic!("{:?}: {e:#}", String::from_utf8_lossy(name)));
        assert_eq!(got, set(&["libcudnn.so.9"]));
    }
    // A zero-length name.
    let a = sb.archive("empty.whl");
    {
        let f = std::fs::File::create(&a).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = zip::write::SimpleFileOptions::default();
        if z.start_file("", opts).is_ok() {
            z.write_all(b"data").unwrap();
        }
        z.start_file(String::from_utf8_lossy(WHEEL_LIB), opts)
            .unwrap();
        z.write_all(b"ok").unwrap();
        z.finish().unwrap();
    }
    let got = sb
        .run(&a, ArchiveKind::Zip, Layout::NvidiaWheel, small())
        .unwrap();
    assert_eq!(got, set(&["libcudnn.so.9"]));
}

/// Two entries with the same name: whatever the reader decides, one file with the content of
/// one of them, or an error; never both, never a leftover.
#[test]
fn zip_duplicate_names() {
    let mut sb = Sandbox::new("fuzz-zip-dup");
    let a = sb.archive("dup.whl");
    raw_zip(&a, &[(WHEEL_LIB, b"first"), (WHEEL_LIB, b"second")]);
    let dest_n = sb.runs + 1;
    if let Ok(got) = sb.run(&a, ArchiveKind::Zip, Layout::NvidiaWheel, small()) {
        assert_eq!(got, set(&["libcudnn.so.9"]));
        let data =
            std::fs::read(sb.root.join(format!("dest-{dest_n}")).join("libcudnn.so.9")).unwrap();
        assert!(data == b"first" || data == b"second", "{data:?}");
    }
}

/// The uncompressed size in the zip headers is never trusted: a huge claim does not allocate
/// or pass the ceiling, a tiny claim does not let more bytes through. Boundaries of both
/// ceilings: exactly the limit passes, one byte less fails.
#[test]
fn zip_size_claims_and_ceiling_boundaries() {
    let mut sb = Sandbox::new("fuzz-zip-size");
    let data = XorShift(7).bytes(10_000);
    let limits = |entry: u64, total: u64| Limits {
        entry_bytes: entry,
        total_bytes: total,
    };
    for claim in [u32::MAX - 1, 1 << 31, 1, 0] {
        let a = sb.archive("claim.whl");
        raw_zip(&a, &[(WHEEL_LIB, &data)]);
        zip_claim_size(&a, claim);
        // Under a ceiling below the real size, never Ok.
        assert!(
            sb.run(
                &a,
                ArchiveKind::Zip,
                Layout::NvidiaWheel,
                limits(4_000, 1 << 20)
            )
            .is_err(),
            "claim {claim}"
        );
        // With room, either the real bytes or an error (a lying header may fail the CRC
        // or size check of the reader); `check` bounds the written size.
        let _ = sb.run(
            &a,
            ArchiveKind::Zip,
            Layout::NvidiaWheel,
            limits(1 << 20, 1 << 20),
        );
    }
    let a = sb.archive("exact.whl");
    let names: [&[u8]; 2] = [WHEEL_LIB, b"nvidia/cublas/lib/libcublas.so.12"];
    raw_zip(&a, &[(names[0], &data), (names[1], &data)]);
    let n = data.len() as u64;
    let run = |sb: &mut Sandbox, l| sb.run(&a, ArchiveKind::Zip, Layout::NvidiaWheel, l);
    assert_eq!(run(&mut sb, limits(n, 2 * n)).unwrap().len(), 2);
    refused(run(&mut sb, limits(n - 1, 2 * n)), "per file");
    refused(run(&mut sb, limits(n, 2 * n - 1)), "archive total");
}

#[test]
fn truncated_archives_fail_cleanly() {
    let mut sb = Sandbox::new("fuzz-trunc");
    let data = XorShift(11).bytes(40_000);
    let zip = sb.archive("full.whl");
    raw_zip(&zip, &[(WHEEL_LIB, &data)]);
    let tgz = sb.archive("full.tgz");
    raw_tgz(&tgz, &[Raw::File(b"ort-1/lib/libonnxruntime.so.1", &data)]);
    for (full, kind, layout) in [
        (&zip, ArchiveKind::Zip, Layout::NvidiaWheel),
        (&tgz, ArchiveKind::TarGz, Layout::OrtRelease),
    ] {
        let bytes = std::fs::read(full).unwrap();
        let len = bytes.len();
        for cut in [
            0,
            1,
            10,
            30,
            512,
            len / 3,
            len / 2,
            len - 30,
            len - 22,
            len - 1,
        ] {
            let a = sb.archive("cut");
            std::fs::write(&a, &bytes[..cut]).unwrap();
            let r = sb.run(&a, kind, layout, small());
            // A cut inside the data always fails; a cut in a trailer may still yield the
            // complete file, never a short one.
            if cut <= len / 2 {
                assert!(r.is_err(), "{layout:?} cut at {cut}/{len} accepted");
            } else if let Ok(got) = r {
                assert!(got.len() <= 1, "{got:?}");
                let dest = sb.root.join(format!("dest-{}", sb.runs));
                for f in got {
                    assert_eq!(std::fs::read(dest.join(f)).unwrap(), data);
                }
            }
        }
    }
}

/// Links out of the destination, symlinks with a file of the same name after them, and a
/// hard link to a path outside: no link escapes and nothing lands in `outside/`.
#[test]
fn tar_links_never_escape() {
    let mut sb = Sandbox::new("fuzz-links");
    let a = sb.archive("links.tgz");
    raw_tgz(
        &a,
        &[
            Raw::Symlink(ORT_LIB, b"../outside/x"),
            Raw::File(ORT_LIB, b"lib"),
            Raw::Symlink(b"ort-1/lib/libonnxruntime.so.1", b"../../../outside"),
            Raw::File(b"ort-1/lib/libonnxruntime.so.1/inner", b"x"),
            Raw::Hardlink(
                b"ort-1/lib/libonnxruntime_providers_shared.so",
                b"../outside/secret",
            ),
            Raw::Hardlink(b"ort-1/lib/libonnxruntime.so.2", b"/etc/passwd"),
            Raw::Symlink(b"ort-1/lib/libonnxruntime.so.3", b"C:\\outside\\x"),
        ],
    );
    let got = sb
        .run(&a, ArchiveKind::TarGz, Layout::OrtRelease, small())
        .unwrap();
    assert_eq!(got, set(&["libonnxruntime.so"]));
    let dest = sb.root.join(format!("dest-{}", sb.runs));
    let meta = std::fs::symlink_metadata(dest.join("libonnxruntime.so")).unwrap();
    assert!(meta.file_type().is_file(), "written as a regular file");
    assert_eq!(
        std::fs::read(dest.join("libonnxruntime.so")).unwrap(),
        b"lib"
    );
    assert_eq!(
        std::fs::read_dir(sb.root.join("outside")).unwrap().count(),
        0
    );
    // A `..` entry path in a GNU long name is refused like a short one.
    let mut long = b"ort-1/".to_vec();
    long.extend(std::iter::repeat_n(b'd', 120));
    long.extend_from_slice(b"/../../../outside/libonnxruntime.so");
    let a = sb.archive("long-dotdot.tgz");
    raw_tgz(&a, &[Raw::File(&long, b"x")]);
    refused(
        sb.run(&a, ArchiveKind::TarGz, Layout::OrtRelease, small()),
        "`..`",
    );
}

/// A name of exactly 100 bytes (no NUL in the header field) and a GNU long name both map
/// through the whitelist like any other.
#[test]
fn tar_100_byte_and_gnu_long_names() {
    let mut sb = Sandbox::new("fuzz-long");
    let name = |top_len: usize| {
        let mut n = vec![b't'; top_len];
        n.extend_from_slice(b"/lib/libonnxruntime.so.1");
        n
    };
    let exact = name(100 - b"/lib/libonnxruntime.so.1".len());
    assert_eq!(exact.len(), 100);
    let long = name(300);
    for n in [&exact, &long] {
        let a = sb.archive("long.tgz");
        raw_tgz(&a, &[Raw::File(n, b"lib")]);
        let got = sb
            .run(&a, ArchiveKind::TarGz, Layout::OrtRelease, small())
            .unwrap();
        assert_eq!(got, set(&["libonnxruntime.so.1"]), "{} bytes", n.len());
    }
}

/// Non-UTF-8 names. Tar on Unix: the lossy name is not whitelisted, so the entry is skipped;
/// on Windows the tar reader cannot decode it and the error names the archive (RS-13). Zip:
/// a name flagged UTF-8 that is not either fails or is decoded as CP437 (a flat name).
#[test]
fn non_utf8_names() {
    let mut sb = Sandbox::new("fuzz-utf8");
    let a = sb.archive("bad-name.tgz");
    raw_tgz(
        &a,
        &[
            Raw::File(b"ort-1/lib/libonnxruntime\xff.so", b"x"),
            Raw::File(b"ort-1/lib/libonnxruntime.so", b"lib"),
        ],
    );
    let r = sb.run(&a, ArchiveKind::TarGz, Layout::OrtRelease, small());
    if cfg!(windows) {
        let err = format!("{:#}", r.unwrap_err());
        assert!(err.contains("reading an entry name in"), "{err}");
        assert!(err.contains("bad-name.tgz"), "{err}");
    } else {
        assert_eq!(r.unwrap(), set(&["libonnxruntime.so"]));
    }
    let a = sb.archive("bad-name.whl");
    raw_zip(
        &a,
        &[
            (b"nvidia/cudnn/lib/libcudnn\xff\xfe.so.9", b"x"),
            (WHEEL_LIB, b"ok"),
        ],
    );
    // The zip reader may fall back to CP437 for the bad name, which then matches the
    // whitelist like any library name: still one flat component inside the destination.
    match sb.run(&a, ArchiveKind::Zip, Layout::NvidiaWheel, small()) {
        Ok(got) => {
            assert!(got.contains("libcudnn.so.9"), "{got:?}");
            assert!(got.iter().all(|n| !n.contains(['/', '\\', ':'])), "{got:?}");
        }
        Err(e) => assert!(format!("{e:#}").contains("zip"), "{e:#}"),
    }
}

/// 200 seeded iterations each of: random bytes as zip and tar.gz, random bytes gzipped (so
/// the tar parser sees them), and valid archives with a few random bytes flipped.
#[test]
fn byte_soup_never_panics_or_escapes() {
    let mut sb = Sandbox::new("fuzz-soup");
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let payload = rng.bytes(3_000);
    let valid_zip = sb.archive("valid.whl");
    raw_zip(
        &valid_zip,
        &[
            (WHEEL_LIB, &payload),
            (b"nvidia/cublas/lib/libcublas.so.12", b"cublas"),
        ],
    );
    let valid_tgz = sb.archive("valid.tgz");
    raw_tgz(
        &valid_tgz,
        &[
            Raw::File(b"ort-1/lib/libonnxruntime.so.1", &payload),
            Raw::Symlink(ORT_LIB, b"libonnxruntime.so.1"),
        ],
    );
    let valid = [
        std::fs::read(&valid_zip).unwrap(),
        std::fs::read(&valid_tgz).unwrap(),
    ];
    let a = sb.archive("soup");
    for i in 0..200 {
        let len = rng.below(4_096);
        let soup = rng.bytes(len);
        let gz = {
            let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            e.write_all(&soup).unwrap();
            e.finish().unwrap()
        };
        let mut mutated = valid[i % 2].clone();
        for _ in 0..1 + rng.below(8) {
            let at = rng.below(mutated.len());
            mutated[at] = rng.next() as u8;
        }
        for (bytes, kind, layout) in [
            (&soup, ArchiveKind::Zip, Layout::NvidiaWheel),
            (&soup, ArchiveKind::TarGz, Layout::OrtRelease),
            (&gz, ArchiveKind::TarGz, Layout::OrtRelease),
            (
                &mutated,
                if i % 2 == 0 {
                    ArchiveKind::Zip
                } else {
                    ArchiveKind::TarGz
                },
                if i % 2 == 0 {
                    Layout::NvidiaWheel
                } else {
                    Layout::OrtRelease
                },
            ),
        ] {
            std::fs::write(&a, bytes).unwrap();
            let _ = sb.run(&a, kind, layout, small());
        }
        // Keep the sandbox small: drop this iteration's destinations.
        for n in sb.runs - 3..=sb.runs {
            let _ = std::fs::remove_dir_all(sb.root.join(format!("dest-{n}")));
        }
    }
    assert_eq!(
        std::fs::read_dir(sb.root.join("outside")).unwrap().count(),
        0
    );
}
