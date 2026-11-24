//! File names of a prepared dataset's images (`images/`, `heldout/images/`) and labels
//! (`labels/`, `heldout/labels/`): the one rule the writer ([`super::write_reusing`]) and every
//! reader of a run's lists (the dataset check's fold-safety check, the quick check's `Seen`,
//! auto-label's trained frames) share.
//!
//! An image is named `s<N>-<stem>.<ext>`: `N` is the index of its source in the run's sources,
//! `<stem>.<ext>` its file name with every byte but ASCII letters, digits, `-`, `_` and `.` made
//! `_` (one `_` per character). A name is at most [`MAX_NAME_BYTES`]; only the stem is cut to fit,
//! so the extension (what trainers such as Ultralytics pick images by) always stays. A name taken
//! by an earlier image gets `-2`, `-3`, ... before its extension (the stem cut further to fit).
//! The label file is `<the same stem>.txt` ([`ImageName::label`]), never the stem of a cut
//! string, so two images never share a label file.
//!
//! Datasets written before (QA-040) cut the whole name at 120 bytes, extension included:
//! [`legacy_spelling`] gives that spelling exactly, so readers match both. Short names (every
//! name of at most [`MAX_NAME_BYTES`]) are spelled the same by both rules.

use std::collections::BTreeSet;

/// Most bytes of a prepared image name (well under every file system's 255, with room for the
/// trainer's own cache files beside it).
pub const MAX_NAME_BYTES: usize = 120;

/// A prepared image's name and its label file's stem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageName {
    /// `s<N>-<stem>[-k].<ext>`, the name in `images/` and the lists.
    pub image: String,
    /// The image name without `.<ext>`: the label file is `<stem>.txt`.
    pub stem: String,
}

impl ImageName {
    /// The label file's name, `<stem>.txt`.
    pub fn label(&self) -> String {
        format!("{}.txt", self.stem)
    }
}

/// The dataset's character rule: ASCII letters, digits, `-`, `_` and `.` stay, any other
/// character becomes one `_`. The result is ASCII, so it can be cut at any byte.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `file` (sanitized) split into its stem and its extension with the dot (`".jpg"`, or `""` for
/// none). The extension is what follows the last `.` that is not the first byte, as
/// [`std::path::Path::extension`] reads it.
fn split_ext(file: &str) -> (&str, &str) {
    match file.rfind('.') {
        Some(i) if i > 0 => file.split_at(i),
        _ => (file, ""),
    }
}

/// `prefix` + `stem` cut to fit + `suffix` + `ext`, at most [`MAX_NAME_BYTES`] (unless the parts
/// that are never cut alone exceed it).
fn fitted(prefix: &str, stem: &str, suffix: &str, ext: &str) -> String {
    let room = MAX_NAME_BYTES.saturating_sub(prefix.len() + suffix.len() + ext.len());
    // ASCII (sanitized), so every byte index is a char boundary.
    let cut = stem.get(..room.min(stem.len())).unwrap_or(stem);
    format!("{prefix}{cut}{suffix}{ext}")
}

/// The name of source `source`'s image `file` (its file name), the first of `s<N>-<stem>.<ext>`,
/// `s<N>-<stem>-2.<ext>`, ... not in `used`, which it is added to (see the module docs).
pub fn image_name(source: usize, file: &str, used: &mut BTreeSet<String>) -> ImageName {
    let clean = sanitize(file);
    let (stem, ext) = split_ext(&clean);
    let prefix = format!("s{source}-");
    let mut k = 1usize;
    loop {
        let suffix = if k == 1 {
            String::new()
        } else {
            format!("-{k}")
        };
        let image = fitted(&prefix, stem, &suffix, ext);
        if used.insert(image.clone()) {
            let stem = image.strip_suffix(ext).unwrap_or(&image).to_string();
            return ImageName { image, stem };
        }
        k += 1;
    }
}

/// The name [`image_name`] gives source `source`'s image `file` when no earlier image took it.
pub fn spelling(source: usize, file: &str) -> String {
    image_name(source, file, &mut BTreeSet::new()).image
}

/// The name a dataset written before QA-040 gave source `source`'s image `flat` (with no
/// collision): `s<N>-<flat>` by the same character rule, the whole name cut at 120 bytes, the
/// extension included. Equal to [`spelling`] for names that were never cut.
pub fn legacy_spelling(source: usize, flat: &str) -> String {
    let mut s = sanitize(&format!("s{source}-{flat}"));
    s.truncate(MAX_NAME_BYTES);
    s
}

/// The source index of a dataset image name (`s<digits>-`) and the rest; None when it carries
/// no such prefix (a public image named otherwise, a hand-made name).
pub fn split_source(name: &str) -> Option<(usize, &str)> {
    let (p, rest) = name.split_once('-')?;
    let digits = p.strip_prefix('s')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((digits.parse().ok()?, rest))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// The flat name of a frame of a 64-byte camera with a 64-byte stem (the longest the
    /// collector writes): 150 bytes, 153 with `s0-`.
    fn longest() -> String {
        format!("night.{}.2026-11-01.{}.jpg", "c".repeat(64), "x".repeat(64))
    }

    #[test]
    fn short_names_are_spelled_as_before() {
        let mut used = BTreeSet::new();
        let a = image_name(0, "day.my cam.2026-10-01.x.jpg", &mut used);
        assert_eq!(a.image, "s0-day.my_cam.2026-10-01.x.jpg");
        assert_eq!(a.label(), "s0-day.my_cam.2026-10-01.x.txt");
        // Taken twice: `-2` before the extension, as before.
        let b = image_name(0, "day.my cam.2026-10-01.x.jpg", &mut used);
        assert_eq!(b.image, "s0-day.my_cam.2026-10-01.x-2.jpg");
        assert_eq!(b.label(), "s0-day.my_cam.2026-10-01.x-2.txt");
        assert_eq!(
            image_name(12, "é.png", &mut used).image,
            "s12-_.png",
            "one _ per character"
        );
        for f in ["a.jpg", "day.cam.2026-10-01.x y.JPG", "noext"] {
            assert_eq!(spelling(3, f), legacy_spelling(3, f), "{f}");
        }
    }

    // QA-040: the cut keeps the extension; the label is the image's own stem.
    #[test]
    fn a_long_name_keeps_its_extension_and_its_label_stem() {
        let flat = longest();
        assert_eq!(flat.len(), 150);
        let n = image_name(0, &flat, &mut BTreeSet::new());
        assert_eq!(n.image.len(), MAX_NAME_BYTES);
        assert!(n.image.ends_with(".jpg"), "{}", n.image);
        assert_eq!(format!("{}.jpg", n.stem), n.image);
        assert_eq!(n.label(), format!("{}.txt", n.stem));
        assert!(flat.starts_with(n.stem.strip_prefix("s0-").unwrap()));
    }

    // QA-040 (verifier extra): two frames sharing their first 117 bytes get distinct image and
    // label names, both with the real extension, both within the limit.
    #[test]
    fn names_sharing_their_first_117_bytes_stay_distinct() {
        let a = longest();
        let b = a.replace("xxx.jpg", "xxy.jpg");
        assert_eq!(a.get(..117), b.get(..117));
        let mut used = BTreeSet::new();
        let na = image_name(0, &a, &mut used);
        let nb = image_name(0, &b, &mut used);
        assert_ne!(na.image, nb.image);
        assert_ne!(na.label(), nb.label());
        for n in [&na, &nb] {
            assert!(
                n.image.ends_with(".jpg") && n.image.len() <= MAX_NAME_BYTES,
                "{n:?}"
            );
        }
        assert!(nb.image.ends_with("-2.jpg"), "{}", nb.image);
        // A tenth, with a two-digit suffix, still fits.
        for _ in 0..8 {
            image_name(0, &a, &mut used);
        }
        let n10 = image_name(0, &a, &mut used);
        assert!(n10.image.ends_with("-11.jpg"), "{}", n10.image);
        assert_eq!(n10.image.len(), MAX_NAME_BYTES);
    }

    // QA-136: the parse of the prefix is the writer's inverse.
    #[test]
    fn split_source_reads_back_what_image_name_wrote() {
        for (i, f) in [
            (0, "a.jpg".to_string()),
            (12, longest()),
            (7, "x y.png".into()),
        ] {
            let n = image_name(i, &f, &mut BTreeSet::new());
            let (s, rest) = split_source(&n.image).unwrap();
            assert_eq!(s, i);
            assert_eq!(format!("s{i}-{rest}"), n.image);
        }
        assert_eq!(split_source("s12-a.jpg"), Some((12, "a.jpg")));
        assert_eq!(split_source("sun.jpg"), None);
        assert_eq!(split_source("s-a.jpg"), None);
        assert_eq!(split_source("sx1-a.jpg"), None);
        assert_eq!(split_source("frame_17.jpg"), None);
    }

    // QA-136: the old writer's spelling, pinned as a literal; the source index is the real one
    // (the refine check used to cut with `s0-` whatever the source).
    #[test]
    fn legacy_spelling_is_the_old_whole_name_cut() {
        let flat = longest();
        let old = format!("s0-night.{}.2026-11-01.{}", "c".repeat(64), "x".repeat(35));
        assert_eq!(old.len(), 120);
        assert_eq!(legacy_spelling(0, &flat), old);
        assert!(!old.ends_with(".jpg"));
        // Source 10: one byte more of prefix, one byte less of the name.
        let ten = legacy_spelling(10, &flat);
        assert_eq!(ten.len(), 120);
        assert_eq!(ten.strip_prefix("s10-").unwrap(), &old[3..119]);
        assert_ne!(legacy_spelling(10, &flat), spelling(10, &flat));
    }
}
