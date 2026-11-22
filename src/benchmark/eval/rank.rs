//! Ranks and rank groups of the combos on one set of frames.
//!
//! Order, in blocks: rows within the speed budget with the target met (or no target); then rows
//! over the budget ("too slow"); then rows whose target is not met on its bound ("unmet",
//! whatever their speed); then rows that cannot be judged fairly (trained on these frames, or
//! no metric). Within a block: the goal metric (higher is better for every goal), then the p95
//! (faster first), then the input order.
//!
//! Rank groups: the first row is rank 1 and leads its group; each next row of the same block
//! shares the leader's rank unless it is clearly worse than the leader (the paired bootstrap
//! difference's 95% interval lies below 0, [`super::stats::Difference`]), in which case it
//! starts a new group with itself as leader and its position as rank. The first row of each
//! block starts a new group too.

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
    /// Its paired difference from the leader it was compared with: its group's leader, or for a
    /// row that starts a new group, the leader of the group before (None for the first row of a
    /// block).
    pub vs_leader: Option<Difference>,
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
        };
        rows.len()
    ];
    let mut leader: Option<(usize, usize)> = None; // (input index, rank)
    for (pos, &i) in order.iter().enumerate() {
        let Some(r) = key(i) else { continue };
        let position = pos + 1;
        let block = r.block();
        let same_block = leader
            .and_then(|(l, _)| key(l))
            .is_some_and(|l| l.block() == block);
        let vs = match leader {
            Some((l, _)) if same_block => key(l).and_then(|lr| {
                Some(Difference::of(
                    r.metric?,
                    lr.metric?,
                    r.resampled,
                    lr.resampled,
                ))
            }),
            _ => None,
        };
        let starts = !same_block || vs.is_some_and(|d| d.clearly_below());
        let (l, rank) = match leader {
            Some(x) if !starts => x,
            _ => (i, position),
        };
        leader = Some((l, rank));
        if let Some(p) = places.get_mut(i) {
            *p = Place {
                position,
                rank,
                leader: l,
                block,
                vs_leader: vs,
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
