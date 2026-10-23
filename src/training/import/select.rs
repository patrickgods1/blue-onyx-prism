//! Frame selection of an import (pure): the FFmpeg `select` expression that keeps scene changes
//! at a bounded rate ([`select_expr`]), near-duplicate removal by dHash ([`dedup`]), the
//! per-scene quotas ([`quotas`]) and k-center greedy selection over embeddings
//! ([`k_center`], cosine distance) or dHashes ([`k_center_hamming`]).

use std::collections::BTreeMap;

/// The FFmpeg `select` filter expression of an extraction: the first frame; then a frame whose
/// scene-change score is above `scene` and at least `1 / max_fps` seconds after the last kept
/// one; and in any case one frame every `every_s` seconds (so a still scene is sampled too).
/// `every_s <= 0` turns the periodic frame off; `max_fps <= 0` removes the rate limit.
///
/// `prev_selected_t` is NAN before the first selection, and every comparison with NAN is false,
/// hence the explicit first-frame term.
pub fn select_expr(scene: f32, max_fps: f32, every_s: f32) -> String {
    let scene = fmt_num(f64::from(scene.clamp(0.0, 1.0)));
    let change = if max_fps > 0.0 {
        format!(
            "gt(scene,{scene})*gte(t-prev_selected_t,{})",
            fmt_num(1.0 / f64::from(max_fps))
        )
    } else {
        format!("gt(scene,{scene})")
    };
    let periodic = if every_s > 0.0 {
        format!("+gte(t-prev_selected_t,{})", fmt_num(f64::from(every_s)))
    } else {
        String::new()
    };
    format!("if(isnan(prev_selected_t),1,{change}{periodic})")
}

/// A number for an FFmpeg expression: at most 4 decimals, no trailing zeros, never exponent form.
fn fmt_num(v: f64) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-" {
        "0".into()
    } else {
        s.to_string()
    }
}

/// Indices of `frames` (dHash, group, time in seconds) kept when dropping near duplicates: in
/// order, a frame is dropped when its dHash is within `max_hamming` bits of a frame kept before
/// it of the same group (camera) at most `window_s` seconds apart; `window_s <= 0` compares with
/// every kept frame of the group. The window is what the collector does too (10 minutes there):
/// a fixed camera's frames are all near duplicates by dHash (a person crossing the view flips a
/// few of the 64 bits at most), so without it a long clip would keep a single frame.
pub fn dedup(frames: &[(u64, &str, f64)], max_hamming: u32, window_s: f64) -> Vec<usize> {
    let mut kept: Vec<usize> = Vec::new();
    let mut by_group: BTreeMap<&str, Vec<(u64, f64)>> = BTreeMap::new();
    for (i, (h, group, t)) in frames.iter().enumerate() {
        let seen = by_group.entry(group).or_default();
        if seen.iter().any(|(k, kt)| {
            (window_s <= 0.0 || (t - kt).abs() <= window_s)
                && crate::collect::dhash::hamming(*k, *h) <= max_hamming
        }) {
            continue;
        }
        seen.push((*h, *t));
        kept.push(i);
    }
    kept
}

/// How many of `total` frames each group gets when `available` frames are in each: an equal
/// share per group, capped by what the group has, the rest handed out again to the groups that
/// still have frames (water filling), larger remainders first. Never more than `available`.
pub fn quotas(available: &BTreeMap<String, usize>, total: usize) -> BTreeMap<String, usize> {
    let mut out: BTreeMap<String, usize> = available.keys().map(|k| (k.clone(), 0)).collect();
    let mut left = total.min(available.values().sum());
    while left > 0 {
        let open: Vec<&String> = available
            .iter()
            .filter(|(k, n)| out.get(*k).copied().unwrap_or(0) < **n)
            .map(|(k, _)| k)
            .collect();
        if open.is_empty() {
            break;
        }
        let share = (left / open.len()).max(1);
        // Fill the groups with the most frames left first, so the 1-frame round-robin of the
        // last remainder goes to the groups that can take it.
        let mut order = open.clone();
        order.sort_by_key(|k| {
            let room = available.get(*k).copied().unwrap_or(0) - out.get(*k).copied().unwrap_or(0);
            (std::cmp::Reverse(room), (*k).clone())
        });
        for k in order {
            if left == 0 {
                break;
            }
            let have = out.get(k).copied().unwrap_or(0);
            let room = available.get(k).copied().unwrap_or(0) - have;
            let take = share.min(room).min(left);
            out.insert(k.clone(), have + take);
            left -= take;
        }
    }
    out
}

/// Cosine distance of two L2-normalized vectors (`1 - a.b`, clamped to 0..2).
fn cos_dist(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    (1.0 - dot).clamp(0.0, 2.0)
}

/// Generic k-center greedy (Gonzalez): start from `first`, then repeatedly take the point
/// farthest from everything taken (ties: the lowest index). Exact: every distance is computed.
fn k_center_by(n: usize, k: usize, first: usize, dist: impl Fn(usize, usize) -> f64) -> Vec<usize> {
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let first = first.min(n - 1);
    let mut chosen = vec![first];
    let mut near: Vec<f64> = (0..n).map(|i| dist(i, first)).collect();
    while chosen.len() < k.min(n) {
        let mut best: Option<(usize, f64)> = None;
        for (i, d) in near.iter().enumerate() {
            if chosen.contains(&i) {
                continue;
            }
            if best.is_none_or(|(_, b)| *d > b) {
                best = Some((i, *d));
            }
        }
        let Some((pick, _)) = best else { break };
        chosen.push(pick);
        for (i, d) in near.iter_mut().enumerate() {
            let x = dist(i, pick);
            if x < *d {
                *d = x;
            }
        }
    }
    chosen
}

/// The point closest to the mean of `vectors` (the k-center start: a typical frame rather than
/// an outlier).
fn medoid_start(vectors: &[&[f32]]) -> usize {
    let Some(dim) = vectors.first().map(|v| v.len()) else {
        return 0;
    };
    let mut mean = vec![0f64; dim];
    for v in vectors {
        for (m, x) in mean.iter_mut().zip(v.iter()) {
            *m += f64::from(*x);
        }
    }
    let n = vectors.len() as f64;
    for m in &mut mean {
        *m /= n;
    }
    let mut best = (0usize, f64::INFINITY);
    for (i, v) in vectors.iter().enumerate() {
        let d: f64 = v
            .iter()
            .zip(&mean)
            .map(|(x, m)| (f64::from(*x) - m).powi(2))
            .sum();
        if d < best.1 {
            best = (i, d);
        }
    }
    best.0
}

/// `k` of `vectors` (L2-normalized embeddings) by k-center greedy on the cosine distance,
/// starting at the frame closest to the mean. Returns indices into `vectors`, in pick order.
pub fn k_center(vectors: &[&[f32]], k: usize) -> Vec<usize> {
    let first = medoid_start(vectors);
    k_center_by(vectors.len(), k, first, |i, j| {
        match (vectors.get(i), vectors.get(j)) {
            (Some(a), Some(b)) => cos_dist(a, b),
            _ => 0.0,
        }
    })
}

/// `k` of `hashes` by k-center greedy on the dHash Hamming distance (bits / 64), starting at
/// the first frame: the selection without embeddings.
pub fn k_center_hamming(hashes: &[u64], k: usize) -> Vec<usize> {
    k_center_by(hashes.len(), k, 0, |i, j| {
        match (hashes.get(i), hashes.get(j)) {
            (Some(a), Some(b)) => f64::from(crate::collect::dhash::hamming(*a, *b)) / 64.0,
            _ => 0.0,
        }
    })
}

/// Pick `total` of the candidates, balanced by `group` (the scene): each group's quota
/// ([`quotas`]; or one group with everything when `balance` is off) filled by k-center within the
/// group, using `embedding(i)` when every candidate has one, else the dHash. Returns candidate
/// indices sorted ascending.
pub fn pick(
    groups: &[&str],
    hashes: &[u64],
    embeddings: Option<&[Vec<f32>]>,
    total: usize,
    balance: bool,
) -> Vec<usize> {
    let mut members: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        let key = if balance {
            (*g).to_string()
        } else {
            String::new()
        };
        members.entry(key).or_default().push(i);
    }
    let avail: BTreeMap<String, usize> =
        members.iter().map(|(k, v)| (k.clone(), v.len())).collect();
    let q = quotas(&avail, total);
    let mut out = Vec::new();
    for (g, idx) in &members {
        let k = q.get(g).copied().unwrap_or(0);
        let picked: Vec<usize> = match embeddings {
            Some(e) if e.len() == hashes.len() => {
                let vs: Vec<&[f32]> = idx
                    .iter()
                    .filter_map(|i| e.get(*i).map(Vec::as_slice))
                    .collect();
                k_center(&vs, k)
            }
            _ => {
                let hs: Vec<u64> = idx.iter().filter_map(|i| hashes.get(*i).copied()).collect();
                k_center_hamming(&hs, k)
            }
        };
        out.extend(picked.into_iter().filter_map(|p| idx.get(p).copied()));
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn select_expression_keeps_scene_changes_at_a_bounded_rate() {
        assert_eq!(
            select_expr(0.3, 2.0, 10.0),
            "if(isnan(prev_selected_t),1,gt(scene,0.3)*gte(t-prev_selected_t,0.5)+gte(t-prev_selected_t,10))"
        );
        assert_eq!(
            select_expr(0.25, 0.0, 0.0),
            "if(isnan(prev_selected_t),1,gt(scene,0.25))"
        );
        assert_eq!(
            select_expr(2.0, 3.0, 2.5),
            "if(isnan(prev_selected_t),1,gt(scene,1)*gte(t-prev_selected_t,0.3333)+gte(t-prev_selected_t,2.5))"
        );
        // Never exponent notation (FFmpeg reads `1e+05` differently).
        let tiny = select_expr(0.3, 1e-5, 1e-6);
        assert!(!tiny.contains("e+") && !tiny.contains("e-"), "{tiny}");
    }

    #[test]
    fn dedup_drops_near_duplicates_per_camera() {
        let a = 0u64;
        let b = 0b111111u64; // 6 bits from a: a duplicate at <= 6
        let c = 0b1111111u64; // 7 bits from a, 1 from b
        let far = u64::MAX;
        let f = |h: u64, g: &'static str, t: f64| (h, g, t);
        let kept = dedup(
            &[
                f(a, "x", 0.0),
                f(b, "x", 1.0),
                f(c, "x", 2.0),
                f(far, "x", 3.0),
                f(a, "y", 4.0),
            ],
            6,
            0.0,
        );
        // c is within 6 of b, but b was dropped: compared with kept frames only (a: 7 bits).
        assert_eq!(kept, [0, 2, 3, 4]);
        assert_eq!(dedup(&[f(a, "x", 0.0), f(b, "x", 1.0)], 5, 0.0), [0, 1]);
        assert!(dedup(&[], 6, 10.0).is_empty());
        // The window: the same picture 30 s later is kept, 5 s later dropped.
        assert_eq!(
            dedup(
                &[
                    f(a, "x", 0.0),
                    f(a, "x", 5.0),
                    f(a, "x", 30.0),
                    f(a, "x", 33.0)
                ],
                6,
                10.0
            ),
            [0, 2]
        );
    }

    #[test]
    fn quotas_fill_equal_shares_then_redistribute() {
        let m = |v: &[(&str, usize)]| -> BTreeMap<String, usize> {
            v.iter().map(|(k, n)| ((*k).to_string(), *n)).collect()
        };
        assert_eq!(
            quotas(&m(&[("day", 100), ("ir", 3), ("night", 10)]), 30),
            m(&[("day", 17), ("ir", 3), ("night", 10)])
        );
        assert_eq!(
            quotas(&m(&[("day", 100), ("ir", 100)]), 31),
            m(&[("day", 16), ("ir", 15)])
        );
        assert_eq!(
            quotas(&m(&[("day", 2), ("ir", 1)]), 10),
            m(&[("day", 2), ("ir", 1)])
        );
        assert_eq!(quotas(&m(&[("day", 5)]), 0), m(&[("day", 0)]));
        let q = quotas(&m(&[("a", 7), ("b", 7), ("c", 7)]), 20);
        assert_eq!(q.values().sum::<usize>(), 20);
        assert!(q.values().all(|v| *v >= 6));
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn k_center_covers_the_clusters_before_repeating_one() {
        // Three tight clusters on the axes; ten points near x, three near y, two near z.
        let mut pts = Vec::new();
        for i in 0..10 {
            pts.push(unit(&[1.0, 0.01 * i as f32, 0.0]));
        }
        for i in 0..3 {
            pts.push(unit(&[0.01 * i as f32, 1.0, 0.0]));
        }
        for i in 0..2 {
            pts.push(unit(&[0.0, 0.01 * i as f32, 1.0]));
        }
        let refs: Vec<&[f32]> = pts.iter().map(Vec::as_slice).collect();
        let picked = k_center(&refs, 3);
        let cluster = |i: usize| {
            if i < 10 {
                0
            } else if i < 13 {
                1
            } else {
                2
            }
        };
        let mut cs: Vec<usize> = picked.iter().map(|i| cluster(*i)).collect();
        cs.sort_unstable();
        assert_eq!(cs, [0, 1, 2], "{picked:?}");
        // Starts from the medoid (the big x cluster) and is deterministic.
        assert_eq!(cluster(picked[0]), 0);
        assert_eq!(k_center(&refs, 3), picked);
        assert_eq!(k_center(&refs, 100).len(), pts.len());
        assert!(k_center(&[], 3).is_empty());
        // Every pick is distinct.
        let mut all = k_center(&refs, 15);
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 15);
    }

    #[test]
    fn k_center_on_dhashes_and_balanced_pick() {
        let hashes = [0u64, 1, 3, u64::MAX, u64::MAX - 1, 0xFFFF_0000_0000_0000];
        let p = k_center_hamming(&hashes, 3);
        assert_eq!(p[0], 0);
        assert_eq!(p[1], 3, "the farthest from 0 first");
        assert_eq!(p[2], 5);
        let groups = ["day", "day", "day", "day", "ir", "ir"];
        let picked = pick(&groups, &hashes, None, 4, true);
        assert_eq!(picked.len(), 4);
        assert_eq!(picked.iter().filter(|i| **i >= 4).count(), 2, "{picked:?}");
        let unbalanced = pick(&groups, &hashes, None, 2, false);
        assert_eq!(unbalanced, [0, 3]);
        // With embeddings: k-center on them.
        let e: Vec<Vec<f32>> = (0..6)
            .map(|i| unit(&[1.0, if i == 2 { 5.0 } else { 0.01 * i as f32 }]))
            .collect();
        let picked = pick(&groups, &hashes, Some(&e), 3, false);
        assert!(picked.contains(&2), "the outlier is covered: {picked:?}");
    }
}
