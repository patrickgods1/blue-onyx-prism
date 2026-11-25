//! Step 6 of [`super::plan`]: the validation split, as whole camera days ([`pick_valid_groups`];
//! docs/TRAINING.md "Dataset"). Frames of one camera day are near-duplicates, so validation on
//! them cannot see overfitting. A class of [`VALID_CLASS_FLOORS`] gets more days until validation
//! holds that many of its boxes (a noisy checkpoint choice otherwise). Pure.

use super::{
    FrameRef, Options, Source, UNKNOWN, class_boxes, frame_key, group_key, seeded_rank, share_of,
};
use std::collections::BTreeMap;

/// Salt of the validation split's seeded ranks.
const VALID_SALT: u64 = 0x5eed;
/// Fewest camera days (with the held-out ones gone) a validation split by camera day needs.
const MIN_VALID_GROUPS: usize = 3;
/// Fewest validation boxes wanted of a class the run trains: whole camera days are added for it
/// (the third pass of [`whole_groups`]). A run validated on 34 person boxes chose its checkpoint
/// on few boxes: one box found or missed was 3% of the person recall there; at 80 it is 1.25%.
/// Person only: the class every Blue Iris camera is about, and the one with reviewed days to
/// spare.
pub const VALID_CLASS_FLOORS: &[(&str, usize)] = &[("person", 80)];

/// A camera day of the validation candidates.
struct ValidGroup {
    key: String,
    frames: Vec<FrameRef>,
    /// Boxes per run class.
    boxes: Vec<usize>,
    /// Its frames with ignore regions: no validation frame, and out of training when the day
    /// validates ([`ValidSplit::dropped`]).
    held: Vec<FrameRef>,
    /// Boxes per run class of `held`: training loses them with the day.
    lost: Vec<usize>,
}

/// What [`whole_groups`] picked.
struct Picked {
    frames: Vec<FrameRef>,
    /// Sorted group keys.
    keys: Vec<String>,
    /// The frames with ignore regions of those groups ([`ValidSplit::dropped`]).
    held: Vec<FrameRef>,
    /// [`ValidSplit::notes`].
    notes: Vec<String>,
}

/// What [`pick_valid_groups`] chose.
pub(super) struct ValidSplit {
    /// Sorted.
    pub(super) frames: Vec<FrameRef>,
    /// Sorted `camera/day` keys; empty after the fallback to the frame split.
    pub(super) groups: Vec<String>,
    /// Why validation fell back to the frame split, when the user can do something about it.
    pub(super) warning: Option<String>,
    /// Frames of the pool with ignore regions on the camera days chosen (sorted): never validation
    /// frames (validation cannot ignore a region), and the day is validation's, so they leave
    /// training too. Empty after the frame split, where such frames simply train.
    pub(super) dropped: Vec<FrameRef>,
    /// Plain warnings beside the split: a class of [`VALID_CLASS_FLOORS`] short of its floor.
    pub(super) notes: Vec<String>,
}

/// Step 6 of [`super::plan`]: the validation split, for the trainer's checkpoint choice.
///
/// Candidates: the review frames of `pool` when there are any, else the pool. The size is
/// `val_share` of them, at least `min_valid` (at most a quarter of them): a share of the whole
/// pool, public frames included, once took every review frame, so a class only the review frames
/// have was never trained.
///
/// Whole `camera/day` groups ([`whole_groups`]), so that validation is on camera days the model
/// never trained on: frames of one camera day are near-duplicates, and validation on them cannot
/// see overfitting. When that does not work out, the frames are picked one by one by seeded rank
/// (never a frame holding the last training box of a class), with a warning; public-only pools
/// do that silently, nothing about them can be improved.
///
/// A frame of `pool` with ignore regions (there with [`super::IgnoreMode::Background`]) is never a
/// candidate: the trainer's validation cannot ignore a region. It goes with its camera day: when
/// the day validates, it leaves training too ([`ValidSplit::dropped`]); after the frame split it
/// trains.
pub(super) fn pick_valid_groups(
    sources: &[Source],
    pool: &[FrameRef],
    opts: &Options,
    classes: &[String],
) -> ValidSplit {
    let frame = |r: FrameRef| sources.get(r.0).and_then(|s| s.frames.get(r.1));
    // A frame with ignore regions (in the pool with `IgnoreMode::Background`) never validates.
    let regions = |r: &FrameRef| frame(*r).is_some_and(|f| !f.ignore.is_empty());
    let eligible: Vec<FrameRef> = pool.iter().copied().filter(|r| !regions(r)).collect();
    let mut cands: Vec<FrameRef> = eligible
        .iter()
        .copied()
        .filter(|r| sources.get(r.0).is_some_and(|s| s.public.is_none()))
        .collect();
    let public_only = cands.is_empty();
    // Own frames in the pool, every one with ignore regions: validation falls to the public ones.
    let own_all_regions = public_only
        && !eligible.is_empty()
        && pool
            .iter()
            .any(|r| sources.get(r.0).is_some_and(|s| s.public.is_none()));
    if public_only {
        cands = eligible;
    }
    let n_val = if cands.len() >= 2 {
        let wanted = share_of(cands.len(), opts.val_share).max(opts.min_valid.min(cands.len() / 4));
        wanted.clamp(1, cands.len() - 1)
    } else {
        0
    };
    let boxes_of = |r: FrameRef| {
        frame(r).map_or_else(|| vec![0; classes.len()], |f| class_boxes(f, opts, classes))
    };
    let mut pool_boxes = vec![0usize; classes.len()];
    for &r in pool {
        for (t, n) in pool_boxes.iter_mut().zip(boxes_of(r)) {
            *t += n;
        }
    }
    let by_groups = if public_only || n_val == 0 {
        Err(None)
    } else {
        let mut groups: BTreeMap<String, ValidGroup> = BTreeMap::new();
        for &r in &cands {
            let Some(f) = frame(r) else { continue };
            let key = group_key(f);
            let g = groups.entry(key.clone()).or_insert_with(|| ValidGroup {
                key,
                frames: Vec::new(),
                boxes: vec![0; classes.len()],
                held: Vec::new(),
                lost: vec![0; classes.len()],
            });
            g.frames.push(r);
            for (t, n) in g.boxes.iter_mut().zip(boxes_of(r)) {
                *t += n;
            }
        }
        // Frames with regions go with their day (a day of nothing but them is no group: it
        // validates nothing, so it trains).
        for r in pool.iter().copied().filter(|r| regions(r)) {
            let Some(g) = frame(r).and_then(|f| groups.get_mut(&group_key(f))) else {
                continue;
            };
            g.held.push(r);
            for (t, n) in g.lost.iter_mut().zip(boxes_of(r)) {
                *t += n;
            }
        }
        whole_groups(groups, opts.seed, n_val, cands.len(), &pool_boxes, classes).map_err(Some)
    };
    let mut dropped = Vec::new();
    let mut notes = Vec::new();
    let (mut frames, groups, warning) = match by_groups {
        Ok(p) => {
            dropped = p.held;
            notes = p.notes;
            (p.frames, p.keys, None)
        }
        Err(warning) => {
            cands.sort_by_cached_key(|&r| {
                let key = sources
                    .get(r.0)
                    .zip(frame(r))
                    .map(|(s, f)| frame_key(s, f))
                    .unwrap_or_default();
                seeded_rank(opts.seed ^ VALID_SALT, &key)
            });
            let mut left = pool_boxes.clone();
            let mut valid: Vec<FrameRef> = Vec::new();
            for r in cands {
                if valid.len() >= n_val {
                    break;
                }
                let need = boxes_of(r);
                if need.iter().zip(&left).any(|(n, l)| *n > 0 && l <= n) {
                    continue;
                }
                for (l, n) in left.iter_mut().zip(&need) {
                    *l -= n;
                }
                valid.push(r);
            }
            // No camera day is added for a floor here: the warning says why validation is frame
            // by frame, the floor's note what it holds (nothing for public frames or a pool too
            // small to validate, where no review can change it or the plan says so).
            if !public_only && n_val > 0 {
                let mut in_valid = vec![0usize; classes.len()];
                for &r in &valid {
                    for (v, n) in in_valid.iter_mut().zip(boxes_of(r)) {
                        *v += n;
                    }
                }
                notes = floor_notes(classes, &in_valid, &pool_boxes, FRAME_SPLIT_SHORT);
            }
            (valid, Vec::new(), warning)
        }
    };
    frames.sort_unstable();
    dropped.sort_unstable();
    let warning = warning.or_else(|| {
        own_all_regions.then(|| {
            "every training frame of your own shows an unsure or too-small object (an ignore \
             region), and validation cannot ignore one: the trainer validates on the public frames \
             only, so its best epoch is picked for them, not your cameras; review frames without \
             such objects, or turn the size floor down"
                .to_string()
        })
    });
    ValidSplit {
        frames,
        groups,
        warning,
        dropped,
        notes,
    }
}

/// Why a floor is short after the third pass of [`whole_groups`] (`{plural}`: the class's).
const WHOLE_DAYS_SHORT: &str = "the remaining camera days with {plural} would take more than half \
                                of a class's training boxes or past a third of the frames";
/// Why a floor is short after the frame split.
const FRAME_SPLIT_SHORT: &str = "validation is split frame by frame (see the warning on it), so \
                                 no camera day was added for it";

/// The plural of a class name in a sentence ("days with people").
fn plural(class: &str) -> String {
    match class {
        "person" => "people".to_string(),
        c => format!("{c}s"),
    }
}

/// The notes of the [`VALID_CLASS_FLOORS`] classes of the run that validation holds fewer boxes of
/// than wanted (`in_valid` per run class), each with `why` and the next action. A class without
/// any box in the pool (`pool_boxes`) gets none: the plan warns about it apart.
fn floor_notes(
    classes: &[String],
    in_valid: &[usize],
    pool_boxes: &[usize],
    why: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    for (class, floor) in VALID_CLASS_FLOORS {
        let Some(i) = classes.iter().position(|c| c == class) else {
            continue;
        };
        let have = in_valid.get(i).copied().unwrap_or(0);
        if have >= *floor || pool_boxes.get(i).copied().unwrap_or(0) == 0 {
            continue;
        }
        let plural = plural(class);
        out.push(format!(
            "validation has {have} {class} boxes, {floor} wanted: {}; review more days with \
             {plural}",
            why.replace("{plural}", &plural)
        ));
    }
    out
}

/// Pick whole camera days for validation: `n_val` frames wanted out of `cands`, `pool_boxes` the
/// boxes per class of the whole pool. Returns the frames and sorted group keys, or the warning
/// for [`pick_valid_groups`]'s fallback.
///
/// Groups go by seeded rank, those adding a class not yet in validation first. A group is
/// skipped when training would then lose more than half of a class's boxes (to validation, or with
/// the group's frames with ignore regions, [`ValidGroup::lost`]: a rare class stays in training,
/// and so does its last box). Groups rarely add up to `n_val`: the first pass
/// accepts a group while the total stays within 1.5 x `n_val`, the second, when that fell short,
/// within `max(n_val, cands / 4)`. That cap is below the first pass on pools under 300 frames,
/// where one whole day may then pass a quarter of the pool (at most 37.5%): better than a split
/// that cannot see overfitting. Frames tagged with neither camera nor day are no group: they may
/// be from any camera day, so they stay in training.
///
/// A third pass, once that split stands, serves [`VALID_CLASS_FLOORS`]: while a floor class of
/// the run (`classes`) is short, the next group with its boxes in the same seeded rank is taken,
/// under the same half rule, while the total stays within a third of `cands` (training keeps its
/// size). A floor still short gets a note ([`floor_notes`]).
fn whole_groups(
    mut by_key: BTreeMap<String, ValidGroup>,
    seed: u64,
    n_val: usize,
    cands: usize,
    pool_boxes: &[usize],
    classes: &[String],
) -> Result<Picked, String> {
    const FRAME_SPLIT: &str = "validation is split frame by frame, so it shares camera days with \
                               training and its mAP reads optimistic";
    by_key.remove(&format!("{UNKNOWN}/{UNKNOWN}"));
    if by_key.is_empty() {
        return Err(format!(
            "no frame has a camera or day tag: {FRAME_SPLIT}; train from a Dataset tab export, \
             or tag the frames (camera: and date: tags in manifest.json)"
        ));
    }
    if by_key.len() < MIN_VALID_GROUPS {
        return Err(format!(
            "only {} camera day(s) are left after the held-out split ({}), at least \
             {MIN_VALID_GROUPS} are needed to validate on whole days: {FRAME_SPLIT}; review \
             frames from more cameras or days",
            by_key.len(),
            by_key.keys().cloned().collect::<Vec<_>>().join(", ")
        ));
    }
    let mut groups: Vec<ValidGroup> = by_key.into_values().collect();
    groups.sort_by_cached_key(|g| (seeded_rank(seed ^ VALID_SALT, &g.key), g.key.clone()));
    let count = groups.len();
    let cap = n_val.max(cands / 4);
    let mut frames: Vec<FrameRef> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    let mut held: Vec<FrameRef> = Vec::new();
    let mut in_valid = vec![0usize; pool_boxes.len()];
    // Boxes training loses: validation's and those of the chosen days' frames with regions.
    let mut gone = vec![0usize; pool_boxes.len()];
    // Training keeps at least half of every class's boxes ([`ValidGroup::lost`] counted).
    let keeps_half = |g: &ValidGroup, gone: &[usize]| {
        g.boxes
            .iter()
            .zip(&g.lost)
            .zip(gone)
            .zip(pool_boxes)
            .all(|(((n, l), v), all)| n + l == 0 || 2 * (v + n + l) <= *all)
    };
    for limit in [n_val * 3 / 2, cap] {
        while frames.len() < n_val {
            let fits =
                |g: &ValidGroup| frames.len() + g.frames.len() <= limit && keeps_half(g, &gone);
            let adds_class = |g: &ValidGroup| {
                g.boxes
                    .iter()
                    .zip(&in_valid)
                    .any(|(n, v)| *n > 0 && *v == 0)
            };
            let next = groups
                .iter()
                .position(|g| fits(g) && adds_class(g))
                .or_else(|| groups.iter().position(fits));
            let Some(i) = next else { break };
            let g = groups.remove(i);
            for (v, n) in in_valid.iter_mut().zip(&g.boxes) {
                *v += n;
            }
            for ((v, n), l) in gone.iter_mut().zip(&g.boxes).zip(&g.lost) {
                *v += n + l;
            }
            frames.extend(g.frames);
            held.extend(g.held);
            keys.push(g.key);
        }
    }
    if frames.len() * 2 < n_val {
        return Err(format!(
            "whole camera days reach only {} of the {n_val} validation frames wanted (of {count} \
             days, the others are too large or hold too many of a class's boxes): {FRAME_SPLIT}; \
             review frames from more cameras or days",
            frames.len()
        ));
    }
    if in_valid.iter().all(|n| *n == 0) {
        return Err(format!(
            "no camera day with objects can go to validation (of {count} days, those are too \
             large or hold too many of a class's boxes): {FRAME_SPLIT}; review frames from more \
             cameras or days"
        ));
    }
    // Third pass: the floors.
    let third = cands / 3;
    for (class, floor) in VALID_CLASS_FLOORS {
        let Some(ci) = classes.iter().position(|c| c == class) else {
            continue;
        };
        while in_valid.get(ci).is_some_and(|n| n < floor) {
            let next = groups.iter().position(|g| {
                g.boxes.get(ci).is_some_and(|n| *n > 0)
                    && frames.len() + g.frames.len() <= third
                    && keeps_half(g, &gone)
            });
            let Some(i) = next else { break };
            let g = groups.remove(i);
            for (v, n) in in_valid.iter_mut().zip(&g.boxes) {
                *v += n;
            }
            for ((v, n), l) in gone.iter_mut().zip(&g.boxes).zip(&g.lost) {
                *v += n + l;
            }
            frames.extend(g.frames);
            held.extend(g.held);
            keys.push(g.key);
        }
    }
    let notes = floor_notes(classes, &in_valid, pool_boxes, WHOLE_DAYS_SHORT);
    keys.sort_unstable();
    Ok(Picked {
        frames,
        keys,
        held,
        notes,
    })
}

/// Where validation came from, for the logs (None after the frame split).
pub fn valid_groups_text(groups: &[String]) -> Option<String> {
    (!groups.is_empty()).then(|| {
        format!(
            "validation on {} camera day(s) kept out of training: {}",
            groups.len(),
            groups.join(", ")
        )
    })
}

#[cfg(test)]
mod tests;
