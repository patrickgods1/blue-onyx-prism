//! Test helpers for the extraction tests: a self-removing temp dir and builders for crafted
//! zip / tar.gz archives (including names the archive writers refuse, patched in afterwards).

use super::*;

/// A fresh directory under the OS temp dir, removed on drop (also when a test panics).
pub(super) struct TempDir(std::path::PathBuf);

impl TempDir {
    pub(super) fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("bop-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl std::ops::Deref for TempDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// Write a zip (a wheel) with `entries` (name, bytes).
pub(super) fn zip_with(path: &Path, entries: &[(&str, &[u8])]) {
    let f = std::fs::File::create(path).unwrap();
    let mut z = zip::ZipWriter::new(f);
    for (name, data) in entries {
        z.start_file(*name, zip::write::SimpleFileOptions::default())
            .unwrap();
        z.write_all(data).unwrap();
    }
    z.finish().unwrap();
}

/// A tar.gz built from `entries`: (path, kind, data or link target, mode).
pub(super) enum TarEntry<'a> {
    File(&'a str, &'a [u8], u32),
    Symlink(&'a str, &'a str),
    Hardlink(&'a str, &'a str),
}

pub(super) fn tgz_with(path: &Path, entries: &[TarEntry<'_>]) {
    let f = std::fs::File::create(path).unwrap();
    let gz = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
    let mut tar = tar::Builder::new(gz);
    for e in entries {
        let mut h = tar::Header::new_gnu();
        match e {
            TarEntry::File(name, data, mode) => {
                h.set_size(data.len() as u64);
                h.set_mode(*mode);
                // `set_path` refuses `..` and absolute names; a crafted archive has
                // them in the raw header, so write the bytes directly.
                let field = &mut h.as_gnu_mut().unwrap().name;
                field[..name.len()].copy_from_slice(name.as_bytes());
                h.set_cksum();
                tar.append(&h, *data).unwrap();
            }
            TarEntry::Symlink(name, target) => {
                h.set_entry_type(tar::EntryType::Symlink);
                h.set_size(0);
                h.set_mode(0o777);
                tar.append_link(&mut h, name, target).unwrap();
            }
            TarEntry::Hardlink(name, target) => {
                h.set_entry_type(tar::EntryType::Link);
                h.set_size(0);
                h.set_mode(0o644);
                tar.append_link(&mut h, name, target).unwrap();
            }
        }
    }
    tar.into_inner().unwrap().finish().unwrap();
}

/// The zip writer normalises `..` and absolute names away, so a crafted name is written
/// with a same-length placeholder (`..` -> `ZZ`, leading `/` -> `Z`) and patched into
/// the local header and the central directory afterwards (the CRC covers data only).
pub(super) fn zip_with_modes(path: &Path, entries: &[(&str, &[u8], Option<u32>)]) {
    let placeholder = |n: &str| {
        n.replace("..", "ZZ")
            .replacen('/', "Z", usize::from(n.starts_with('/')))
    };
    {
        let f = std::fs::File::create(path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        for (name, data, mode) in entries {
            let mut o = zip::write::SimpleFileOptions::default();
            if let Some(m) = mode {
                // `unix_permissions` masks to 0o777; the raw attributes keep the file
                // type and setuid/setgid bits.
                o = o.system(zip::System::Unix).external_attributes(m << 16);
            }
            z.start_file(placeholder(name), o).unwrap();
            z.write_all(data).unwrap();
        }
        z.finish().unwrap();
    }
    let mut bytes = std::fs::read(path).unwrap();
    for (name, _, _) in entries {
        let from = placeholder(name);
        if from == *name {
            continue;
        }
        let mut at = 0;
        while let Some(i) = bytes[at..]
            .windows(from.len())
            .position(|w| w == from.as_bytes())
        {
            bytes[at + i..at + i + from.len()].copy_from_slice(name.as_bytes());
            at += i + from.len();
        }
    }
    std::fs::write(path, bytes).unwrap();
}

pub(super) const LINUX: Target<'static> = Target {
    os: "linux",
    arch: "x86_64",
};

pub(super) fn ort_tgz(archive: &Path, dest: &Path, limits: Limits) -> Result<BTreeSet<String>> {
    extract_part_with(
        archive,
        ArchiveKind::TarGz,
        Layout::OrtRelease,
        Some(Flavor::Cpu),
        LINUX,
        dest,
        limits,
    )
}

pub(super) fn wheel_zip(archive: &Path, dest: &Path, limits: Limits) -> Result<BTreeSet<String>> {
    extract_part_with(
        archive,
        ArchiveKind::Zip,
        Layout::NvidiaWheel,
        None,
        LINUX,
        dest,
        limits,
    )
}

pub(super) fn refused(r: Result<BTreeSet<String>>, why: &str) {
    let err = r.expect_err("the archive must be refused");
    let text = format!("{err:#}");
    assert!(text.contains(why), "expected {why:?} in: {text}");
}
