//! Folds of the dataset check ([`assign`]): whole (camera, day) groups, so a fold model never trains
//! on a frame of the camera day it is asked about (frames of one camera on one day are near copies).
//! Pure and deterministic.

use anyhow::{Result, bail};
use std::collections::BTreeMap;

/// A (camera, day) group, as the export names it ([`crate::collect::labels::export::Group`]).
pub type Group = (String, String);

/// The rank of `key` under `seed`: the first 8 bytes of SHA-256(`seed:key`), for seeded ties.
fn rank(seed: u64, key: &str) -> u64 {
    use sha2::Digest;
    let d = sha2::Sha256::digest(format!("{seed}:{key}").as_bytes());
    let mut b = [0u8; 8];
    for (o, i) in b.iter_mut().zip(d.iter()) {
        *o = *i;
    }
    u64::from_be_bytes(b)
}

/// How many folds `k` folds over `groups` groups become: at most one per group.
pub fn effective_folds(k: usize, groups: usize) -> usize {
    k.min(groups)
}

/// The fold (0-based) of every group of `groups` (group -> its frames): the largest group first
/// (ties in a seeded order), each into the fold with the fewest frames so far (ties: the lowest
/// fold). `k` is lowered to the number of groups, so no fold is empty. Fewer than 2 groups are
/// refused: a fold model would have nothing of its own to train on.
pub fn assign(
    groups: &BTreeMap<Group, usize>,
    k: usize,
    seed: u64,
) -> Result<BTreeMap<Group, usize>> {
    if groups.len() < 2 {
        bail!(
            "the dataset check needs reviewed frames of at least 2 camera days (one is held out \
             while the others train), and there {}: review frames of another camera or day first",
            if groups.len() == 1 {
                "is 1".to_string()
            } else {
                "are none".to_string()
            }
        );
    }
    if k < 2 {
        bail!("the dataset check needs at least 2 folds, got {k}");
    }
    let k = effective_folds(k, groups.len());
    let mut order: Vec<(&Group, usize, u64)> = groups
        .iter()
        .map(|(g, n)| (g, *n, rank(seed, &format!("{}/{}", g.0, g.1))))
        .collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(b.0)));
    let mut sizes = vec![0usize; k];
    let mut out = BTreeMap::new();
    for (g, n, _) in order {
        let (fold, _) = sizes
            .iter()
            .enumerate()
            .min_by_key(|(i, s)| (**s, *i))
            .unwrap_or((0, &0));
        if let Some(s) = sizes.get_mut(fold) {
            *s += n;
        }
        out.insert(g.clone(), fold);
    }
    Ok(out)
}

/// Frames per fold of `folds` over `groups` (`k` entries).
pub fn fold_sizes(
    groups: &BTreeMap<Group, usize>,
    folds: &BTreeMap<Group, usize>,
    k: usize,
) -> Vec<usize> {
    let mut v = vec![0usize; k];
    for (g, f) in folds {
        if let Some(s) = v.get_mut(*f) {
            *s += groups.get(g).copied().unwrap_or(0);
        }
    }
    v
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn groups(sizes: &[(&str, &str, usize)]) -> BTreeMap<Group, usize> {
        sizes
            .iter()
            .map(|(c, d, n)| ((c.to_string(), d.to_string()), *n))
            .collect()
    }

    fn many(n: usize) -> BTreeMap<Group, usize> {
        (0..n)
            .map(|i| {
                (
                    (format!("cam{}", i % 5), format!("2026-10-{:02}", i / 5 + 1)),
                    (i * 37) % 23 + 1,
                )
            })
            .collect()
    }

    #[test]
    fn every_group_is_in_exactly_one_fold_and_no_fold_is_empty() {
        for n in [2, 3, 7, 30] {
            for k in [2, 4, 8] {
                let g = many(n);
                let f = assign(&g, k, 0).unwrap();
                assert_eq!(f.len(), g.len());
                assert!(f.keys().eq(g.keys()), "every group once");
                let k = effective_folds(k, n);
                assert!(f.values().all(|&v| v < k));
                let sizes = fold_sizes(&g, &f, k);
                assert!(sizes.iter().all(|&s| s > 0), "{sizes:?}");
                assert_eq!(sizes.iter().sum::<usize>(), g.values().sum::<usize>());
            }
        }
    }

    #[test]
    fn it_is_deterministic_for_a_seed() {
        let g = many(30);
        assert_eq!(assign(&g, 4, 7).unwrap(), assign(&g, 4, 7).unwrap());
        // Ties are broken by the seed: groups of one size land elsewhere for another seed.
        let same = groups(&[
            ("a", "d1", 5),
            ("b", "d1", 5),
            ("c", "d1", 5),
            ("d", "d1", 5),
            ("e", "d1", 5),
            ("f", "d1", 5),
        ]);
        let seeds: Vec<_> = (0..8).map(|s| assign(&same, 3, s).unwrap()).collect();
        assert!(seeds.iter().any(|f| *f != seeds[0]), "a seed changes ties");
    }

    #[test]
    fn folds_are_balanced_within_the_largest_group() {
        let g = many(40);
        let k = 4;
        let f = assign(&g, k, 0).unwrap();
        let sizes = fold_sizes(&g, &f, k);
        let largest = *g.values().max().unwrap();
        let (lo, hi) = (*sizes.iter().min().unwrap(), *sizes.iter().max().unwrap());
        assert!(hi - lo <= largest, "{sizes:?}, largest group {largest}");
        // A dominant group gets a fold of its own; the rest share the others.
        let g = groups(&[
            ("big", "d1", 100),
            ("a", "d1", 10),
            ("b", "d1", 10),
            ("c", "d1", 10),
        ]);
        let f = assign(&g, 2, 0).unwrap();
        let big = f[&("big".to_string(), "d1".to_string())];
        assert!(
            f.iter()
                .filter(|(g, _)| g.0 != "big")
                .all(|(_, &v)| v != big)
        );
    }

    #[test]
    fn k_is_lowered_for_few_groups_and_one_group_is_refused() {
        let g = groups(&[("a", "d1", 3), ("b", "d1", 9)]);
        let f = assign(&g, 8, 0).unwrap();
        assert_eq!(effective_folds(8, 2), 2);
        let folds: std::collections::BTreeSet<usize> = f.values().copied().collect();
        assert_eq!(folds.len(), 2);
        let e = assign(&groups(&[("a", "d1", 30)]), 4, 0)
            .unwrap_err()
            .to_string();
        assert!(e.contains("at least 2 camera days"), "{e}");
        assert!(assign(&BTreeMap::new(), 4, 0).is_err());
        assert!(assign(&g, 1, 0).is_err());
    }
}
