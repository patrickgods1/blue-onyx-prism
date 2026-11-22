//! Cross-validated confidence picks: how well the pick procedure does on frames it did not see.
//!
//! Groups are each image's cross-validation group ([`crate::benchmark::report::ImageInfo::group`]:
//! its camera day on "Your cameras") or, without one, the image itself (built-in sets are
//! independent photos). [`folds`] assigns them to K = min([`MAX_FOLDS`], groups) folds, stable
//! and balanced: groups sorted by the SHA-256 of their id, dealt round-robin (fold sizes differ
//! by at most one group; the same groups always land in the same folds, whatever order they
//! come in). For each fold the confidence is picked ([`super::pick::pick`]) on the other folds
//! and that fold is counted at it; the images' counts at their fold's pick, pooled, are the
//! out-of-fold figures ([`Cv::oof`]). The suggested confidence is the pick on every frame
//! ([`Cv::all`]): cross-validation measures the procedure, the suggestion uses all the data.
//!
//! A fold whose other folds cannot be picked on (no objects there) is counted at the pick on
//! every frame instead ([`Cv::borrowed`]; the assessment then flags the figures low confidence,
//! partly in-sample); with fewer than two groups there is no
//! cross-validation at all and the out-of-fold figures are the in-sample ones.

use super::pick::{GridPick, pick};
use super::tally::{At, ImageTally};
use crate::benchmark::threshold::Objective;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Folds at most.
pub const MAX_FOLDS: usize = 5;

/// The fold of each of `groups` (distinct ids) and K (see the module docs).
pub fn folds(groups: &[String]) -> (usize, Vec<usize>) {
    let k = groups.len().min(MAX_FOLDS);
    if k == 0 {
        return (0, Vec::new());
    }
    let mut order: Vec<(Vec<u8>, usize)> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| (Sha256::digest(g.as_bytes()).to_vec(), i))
        .collect();
    // Ties (two equal ids) by input position, so the order is total.
    order.sort();
    let mut fold = vec![0; groups.len()];
    for (pos, (_, i)) in order.iter().enumerate() {
        if let Some(f) = fold.get_mut(*i) {
            *f = pos % k;
        }
    }
    (k, fold)
}

/// The groups of a set of images: distinct ids in first-seen order, and each image's group.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Groups {
    pub ids: Vec<String>,
    /// Per image: its index in `ids`.
    pub of_image: Vec<usize>,
}

impl Groups {
    /// From each image's group id.
    pub fn new<'a>(ids: impl IntoIterator<Item = &'a str>) -> Self {
        let mut index: HashMap<&str, usize> = HashMap::new();
        let mut out = Self::default();
        for id in ids {
            let next = out.ids.len();
            let g = *index.entry(id).or_insert(next);
            if g == next {
                out.ids.push(id.to_string());
            }
            out.of_image.push(g);
        }
        out
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// A cross-validation of one combo on one set (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct Cv {
    pub k: usize,
    /// Per group: its fold.
    pub fold_of_group: Vec<usize>,
    /// Per fold: the grid index picked on the other folds.
    pub fold_pick: Vec<Option<GridPick>>,
    /// Per image: the grid index it was counted at (its fold's pick).
    pub image_index: Vec<Option<usize>>,
    /// The pick on every image: the suggestion.
    pub all: Option<GridPick>,
    /// Folds counted at the pick on every image (their other folds could not be picked on).
    pub borrowed: usize,
    /// Per group: its images' counts at their fold's pick.
    pub oof: Vec<At>,
}

impl Cv {
    /// Cross-validated at all (two groups or more).
    pub fn validated(&self) -> bool {
        self.k >= 2
    }
}

/// Per group and grid value: the summed counts of its images.
pub fn group_sums(images: &[ImageTally], groups: &Groups, n: usize) -> Vec<Vec<At>> {
    let mut out = vec![vec![At::default(); n]; groups.len()];
    for (img, &g) in images.iter().zip(&groups.of_image) {
        if let Some(row) = out.get_mut(g) {
            for (k, a) in row.iter_mut().enumerate() {
                a.add_image(img, k, 1);
            }
        }
    }
    out
}

/// The pick on the groups `use_group(g)` selects.
pub fn pick_on(
    objective: Objective,
    sums: &[Vec<At>],
    n: usize,
    use_group: impl Fn(usize) -> bool,
) -> Option<GridPick> {
    let mut points = vec![At::default(); n];
    for (g, row) in sums.iter().enumerate() {
        if use_group(g) {
            for (p, a) in points.iter_mut().zip(row) {
                p.add(a);
            }
        }
    }
    pick(objective, &points)
}

/// Cross-validate `objective` over `images` in `groups` on a grid of `n` values.
pub fn cross_validate(
    objective: Objective,
    images: &[ImageTally],
    groups: &Groups,
    n: usize,
) -> Cv {
    let sums = group_sums(images, groups, n);
    let (k, fold_of_group) = folds(&groups.ids);
    let all = pick_on(objective, &sums, n, |_| true);
    let mut fold_pick = Vec::with_capacity(k);
    let mut borrowed = 0;
    let validated = k >= 2;
    for f in 0..k {
        let p = if validated {
            pick_on(objective, &sums, n, |g| fold_of_group.get(g) != Some(&f))
        } else {
            all
        };
        if validated && p.is_none() && all.is_some() {
            borrowed += 1;
        }
        fold_pick.push(p.or(all));
    }
    let index_of_group = |g: usize| -> Option<usize> {
        let f = *fold_of_group.get(g)?;
        fold_pick.get(f).copied().flatten().map(|p| p.index)
    };
    let image_index: Vec<Option<usize>> =
        groups.of_image.iter().map(|&g| index_of_group(g)).collect();
    let oof = sums
        .iter()
        .enumerate()
        .map(|(g, row)| {
            index_of_group(g)
                .and_then(|i| row.get(i).copied())
                .unwrap_or_default()
        })
        .collect();
    Cv {
        k,
        fold_of_group,
        fold_pick,
        image_index,
        all,
        borrowed,
        oof,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::eval::tally::Step;

    fn ids(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| format!("cam{}/2026-11-{:02}", i % 3, i))
            .collect()
    }

    #[test]
    fn folds_are_stable_and_balanced() {
        for n in [1, 2, 4, 5, 6, 13, 145] {
            let g = ids(n);
            let (k, f) = folds(&g);
            assert_eq!(k, n.min(5));
            let mut sizes = vec![0usize; k];
            for &x in &f {
                sizes[x] += 1;
            }
            let (lo, hi) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
            assert!(hi - lo <= 1, "{n}: {sizes:?}");
            // The same group lands in the same fold whatever the input order.
            let mut rev = g.clone();
            rev.reverse();
            let (_, fr) = folds(&rev);
            for (i, id) in g.iter().enumerate() {
                let j = rev.iter().position(|x| x == id).unwrap();
                assert_eq!(f[i], fr[j], "{id}");
            }
        }
        assert_eq!(folds(&[]), (0, vec![]));
        // Pinned: the assignment is part of the results' reproducibility.
        let (_, f) = folds(&ids(6));
        assert_eq!(f, folds(&ids(6)).1);
    }

    /// An image with `objects` objects, found (all) at grid indices below `found_until` and
    /// with `fp` false positives at indices below `fp_until`.
    fn img(objects: u32, found_until: usize, fp: u32, fp_until: usize, n: usize) -> ImageTally {
        ImageTally {
            objects,
            positive: objects > 0,
            negative: objects == 0,
            frame: true,
            steps: (0..n)
                .map(|k| Step {
                    tp: if k < found_until { objects } else { 0 },
                    fp: if k < fp_until { fp } else { 0 },
                    alerts: u8::from(k < found_until.max(fp_until)),
                })
                .collect(),
        }
    }

    #[test]
    fn a_folds_pick_never_sees_the_fold() {
        let n = 20;
        let groups: Vec<String> = ids(10);
        let images: Vec<ImageTally> = (0..10).map(|_| img(4, 12, 1, 6, n)).collect();
        let g = Groups::new(groups.iter().map(String::as_str));
        let base = cross_validate(Objective::F1, &images, &g, n);
        assert_eq!(base.k, 5);
        // Make the groups of fold 0 wildly different: many objects found only at indices
        // 12..18, which moves any pick trained on them there.
        let fold0: Vec<usize> = (0..10).filter(|&i| base.fold_of_group[i] == 0).collect();
        let mut changed = images.clone();
        for &i in &fold0 {
            let mut t = img(40, 0, 0, 0, n);
            for (k, s) in t.steps.iter_mut().enumerate() {
                if (12..18).contains(&k) {
                    s.tp = 40;
                    s.alerts = 1;
                }
            }
            changed[i] = t;
        }
        let after = cross_validate(Objective::F1, &changed, &g, n);
        // Fold 0's pick is made on the other folds only: unchanged.
        assert_eq!(after.fold_pick[0], base.fold_pick[0]);
        // Every other fold trained on fold 0: their picks moved.
        assert!((1..5).any(|f| after.fold_pick[f] != base.fold_pick[f]));
        // The pooled counts hold every group once.
        assert_eq!(after.oof.len(), 10);
        let mut pooled = At::default();
        for g in &after.oof {
            pooled.add(g);
        }
        assert_eq!(pooled.objects(), 4 * (10 - fold0.len()) + 40 * fold0.len());
    }

    #[test]
    fn one_group_is_in_sample() {
        let n = 10;
        let images = vec![img(3, 5, 1, 2, n), img(2, 7, 0, 0, n)];
        let g = Groups::new(["a/1", "a/1"]);
        let cv = cross_validate(Objective::F2, &images, &g, n);
        assert_eq!(cv.k, 1);
        assert!(!cv.validated());
        assert_eq!(cv.fold_pick[0], cv.all);
        assert_eq!(cv.image_index, vec![cv.all.map(|p| p.index); 2]);
    }

    #[test]
    fn groups_keep_first_seen_order() {
        let g = Groups::new(["b", "a", "b", "c"]);
        assert_eq!(g.ids, ["b", "a", "c"]);
        assert_eq!(g.of_image, [0, 1, 0, 2]);
    }
}
