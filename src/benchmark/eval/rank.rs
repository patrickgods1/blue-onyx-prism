//! Ranks and rank groups of the combos on one set of frames.
//!
//! Order, in blocks: rows within the speed budget with the target met (or no target); then rows
//! over the budget ("too slow"); then rows whose target is not met on its bound ("unmet",
//! whatever their speed); then rows that cannot be judged fairly (trained on these frames, or
//! no metric). Within a block: the goal metric (higher is better for every goal), then the p95
//! (faster first), then the input order.
//!
//! Rank groups: the first row is rank 1 and leads its group. Each next row of the same block is
//! compared with the block's group leaders in rank order and joins the first one it is not
//! clearly worse than (the paired bootstrap difference's 95% interval does not lie below 0,
//! [`super::stats::Difference`]), sharing its rank; clearly worse than every leader, it starts a
//! new group with itself as leader and its position as rank. So a row with a wide interval
//! shares rank 1 with the best row when it is not clearly worse than it, even after a row that
//! is (it never lands in a lower group than the evidence says). The first row of each block
//! starts a new group too. Each row also keeps its difference from the best row
//! ([`Place::vs_best`]), which the headline's "not clearly different" reads.

use super::stats::Difference;

/// Which block a row is in (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Block {
    Ok,
    TooSlow,
    Unmet,
    Unfair,
}

/// One row to rank.
#[derive(Debug, Clone, Copy)]
pub struct Rankable<'a> {
    pub metric: Option<f64>,
    pub resampled: &'a [Option<f64>],
    pub too_slow: bool,
    /// The target is not met on its bound.
    pub unmet: bool,
    /// Shown, never ranked among the fair rows.
    pub unfair: bool,
    pub p95_ms: Option<f64>,
}

impl Rankable<'_> {
    pub fn block(&self) -> Block {
        if self.unfair || self.metric.is_none() {
            Block::Unfair
        } else if self.unmet {
            Block::Unmet
        } else if self.too_slow {
            Block::TooSlow
        } else {
            Block::Ok
        }
    }
}

/// A row's place.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Place {
    /// 1-based position in the order.
    pub position: usize,
    /// Its rank group's rank (the leader's position).
    pub rank: usize,
    /// The input index of its group's leader.
    pub leader: usize,
    pub block: Block,
    /// Its paired difference from its group's leader, or for a row that starts a new group,
    /// from the leader of the group before (None for the first row of a block).
    pub vs_leader: Option<Difference>,
    /// Its paired difference from the best row (the first of the order) when it is in the same
    /// block; None for the best row itself.
    pub vs_best: Option<Difference>,
}

/// The order (input indices, best first) and each input row's place.
pub fn rank(rows: &[Rankable<'_>]) -> (Vec<usize>, Vec<Place>) {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let key = |i: usize| rows.get(i).copied();
    order.sort_by(|&a, &b| {
        let (Some(x), Some(y)) = (key(a), key(b)) else {
            return a.cmp(&b);
        };
        x.block()
            .cmp(&y.block())
            .then_with(|| {
                y.metric
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&x.metric.unwrap_or(f64::NEG_INFINITY))
            })
            .then_with(|| {
                x.p95_ms
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&y.p95_ms.unwrap_or(f64::INFINITY))
            })
            .then_with(|| a.cmp(&b))
    });
    let mut places = vec![
        Place {
            position: 0,
            rank: 0,
            leader: 0,
            block: Block::Unfair,
            vs_leader: None,
            vs_best: None,
        };
        rows.len()
    ];
    let diff = |a: Rankable<'_>, b: Rankable<'_>| -> Option<Difference> {
        Some(Difference::of(
            a.metric?,
            b.metric?,
            a.resampled,
            b.resampled,
        ))
    };
    let best = order.first().and_then(|&b| key(b).map(|r| (b, r)));
    // The leaders of the current block's groups, in rank order: (input index, rank).
    let mut leaders: Vec<(usize, usize)> = Vec::new();
    let mut block_of_leaders: Option<Block> = None;
    for (pos, &i) in order.iter().enumerate() {
        let Some(r) = key(i) else { continue };
        let position = pos + 1;
        let block = r.block();
        if block_of_leaders != Some(block) {
            leaders.clear();
            block_of_leaders = Some(block);
        }
        // The first leader it is not clearly worse than (a difference that cannot be taken is
        // no evidence either way: it joins).
        let joined = leaders.iter().find_map(|&(l, rank)| {
            let d = key(l).and_then(|lr| diff(r, lr));
            (!d.is_some_and(|d| d.clearly_below())).then_some((l, rank, d))
        });
        let (l, rank, vs) = match joined {
            Some(x) => x,
            None => {
                let before = leaders
                    .last()
                    .and_then(|&(l, _)| key(l))
                    .and_then(|lr| diff(r, lr));
                leaders.push((i, position));
                (i, position, before)
            }
        };
        let vs_best = best
            .filter(|(b, br)| *b != i && br.block() == block)
            .and_then(|(_, br)| diff(r, br));
        if let Some(p) = places.get_mut(i) {
            *p = Place {
                position,
                rank,
                leader: l,
                block,
                vs_leader: vs,
                vs_best,
            };
        }
    }
    (order, places)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draws(centre: f64, spread: f64) -> Vec<Option<f64>> {
        (0..200)
            .map(|i| Some(centre + spread * (f64::from(i % 21) - 10.0) / 10.0))
            .collect()
    }

    #[test]
    fn rank_groups_follow_the_paired_difference() {
        let a = draws(0.80, 0.01);
        let b = draws(0.795, 0.03); // not clearly worse than a
        let c = draws(0.60, 0.01); // clearly worse than a
        let d = draws(0.598, 0.02); // not clearly worse than c
        let slow = draws(0.95, 0.01);
        let row = |m: f64, r: &[Option<f64>]| -> (f64, Vec<Option<f64>>) { (m, r.to_vec()) };
        let data = [
            row(0.60, &c),
            row(0.80, &a),
            row(0.95, &slow),
            row(0.795, &b),
            row(0.598, &d),
        ];
        let mut rows: Vec<Rankable<'_>> = data
            .iter()
            .map(|(m, r)| Rankable {
                metric: Some(*m),
                resampled: r,
                too_slow: false,
                unmet: false,
                unfair: false,
                p95_ms: Some(100.0),
            })
            .collect();
        rows[2].too_slow = true;
        let (order, places) = rank(&rows);
        assert_eq!(order, [1, 3, 0, 4, 2]);
        let ranks: Vec<usize> = order.iter().map(|&i| places[i].rank).collect();
        // a and b share rank 1; c is clearly worse: rank 3 with d; the slow row after them.
        assert_eq!(ranks, [1, 1, 3, 3, 5]);
        assert_eq!(places[4].leader, 0);
        assert_eq!(places[2].block, Block::TooSlow);
        assert!(places[0].vs_leader.unwrap().clearly_below());
        assert!(places[1].vs_leader.is_none());
    }

    /// S4, the reviewers' case: A 0.80 ± 0.005, B 0.70 ± 0.005, C 0.69 with very wide resamples.
    /// B is clearly worse than A: rank 2. C is not clearly worse than A, the first leader in
    /// rank order, so it shares rank 1 (comparing it with B's group alone put it at rank 2).
    #[test]
    fn a_row_joins_the_first_leader_it_is_not_clearly_worse_than() {
        let a = draws(0.80, 0.005);
        let b = draws(0.70, 0.005);
        let c = draws(0.69, 0.5);
        let rows: Vec<Rankable<'_>> = [(0.80, &a), (0.70, &b), (0.69, &c)]
            .iter()
            .map(|(m, r)| Rankable {
                metric: Some(*m),
                resampled: r,
                too_slow: false,
                unmet: false,
                unfair: false,
                p95_ms: Some(100.0),
            })
            .collect();
        let (order, places) = rank(&rows);
        assert_eq!(order, [0, 1, 2]);
        assert_eq!(places[0].rank, 1);
        assert_eq!(places[1].rank, 2, "B is clearly worse than A");
        assert_eq!(places[2].rank, 1, "C is not clearly worse than A");
        assert_eq!(places[2].leader, 0);
        assert!(places[1].vs_best.unwrap().clearly_below());
        assert!(!places[2].vs_best.unwrap().clearly_below());
        assert!(places[0].vs_best.is_none());
    }

    #[test]
    fn unmet_and_unfair_rows_come_last() {
        let r = draws(0.5, 0.01);
        let base = Rankable {
            metric: Some(0.5),
            resampled: &r,
            too_slow: false,
            unmet: false,
            unfair: false,
            p95_ms: None,
        };
        let rows = [
            Rankable {
                unmet: true,
                metric: Some(0.99),
                ..base
            },
            Rankable {
                metric: None,
                ..base
            },
            base,
            Rankable {
                too_slow: true,
                unmet: true,
                ..base
            },
        ];
        let (order, places) = rank(&rows);
        assert_eq!(order, [2, 0, 3, 1]);
        assert_eq!(places[0].block, Block::Unmet);
        assert_eq!(places[3].block, Block::Unmet);
        assert_eq!(places[1].block, Block::Unfair);
        // A new block starts a new group.
        assert_eq!(places[0].rank, 2);
    }
}
