//! What a trained model saw of the review folder: the frames of its training run's `train.txt`
//! and `valid.txt` ([`crate::training::autolabel::seen_frames`], by frame identity
//! `<scene>/<camera>/<date>/<stem>.jpg`, the identity of [`super::super::trained_on`]), plus the
//! frames its run's record of image sources names behind a renamed or cut list name
//! ([`crate::training::refine::quick::seen_of`], when that record reads). Those frames are left
//! out of the model's scores on "Your cameras" ([`super::for_model`]) and out of the Compare view
//! of every model listed beside it ([`crate::benchmark::combos`]).

use crate::collect::labels::export::{self, ExportFrame, Split};
use crate::config::{Config, ModelConfig};
use crate::training::autolabel::frame_of_file;
use std::collections::BTreeSet;

/// How a model saw a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Saw {
    Trained,
    Validated,
}

/// What a model saw of the review folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Seen {
    /// No training record (`<model>.provenance.json`): a catalog or plain local model, which sees
    /// every frame.
    NoRecord,
    /// The frames of its run's lists, by identity.
    Frames {
        trained: BTreeSet<String>,
        validated: BTreeSet<String>,
    },
    /// A record whose lists do not read: what it saw cannot be told (why). Scored on every frame,
    /// never "best" on this set.
    Unknown(String),
}

impl Seen {
    /// How the model saw the set image `file` (a flat export name), if at all.
    pub fn saw(&self, file: &str) -> Option<Saw> {
        let Seen::Frames { trained, validated } = self else {
            return None;
        };
        let id = frame_of_file(file)?;
        if trained.contains(&id) {
            Some(Saw::Trained)
        } else if validated.contains(&id) {
            Some(Saw::Validated)
        } else {
            None
        }
    }

    /// The model's fairness on the set cannot be told.
    pub fn unknown(&self) -> bool {
        matches!(self, Seen::Unknown(_))
    }
}

/// The training record beside the model file of `m`.
fn record_of(config: &Config, m: &ModelConfig) -> std::path::PathBuf {
    let path = config.data_path(&m.path);
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!("{stem}.provenance.json"))
}

/// What `m` saw (see the module docs); `files` are the set's image names, matched against the
/// run's record of image sources.
pub fn of_model<'a>(
    config: &Config,
    m: &ModelConfig,
    files: impl IntoIterator<Item = &'a str>,
) -> Seen {
    if !record_of(config, m).is_file() {
        return Seen::NoRecord;
    }
    let s = match crate::training::autolabel::seen_frames(config, m, &mut Vec::new()) {
        Ok(s) => s,
        Err(e) => {
            return Seen::Unknown(format!(
                "could not tell which of these frames {} trained or validated on: {e:#}",
                m.effective_name()
            ));
        }
    };
    let (mut trained, mut validated) = (s.train, s.valid);
    // A renamed or cut list name: the run's record of image sources names the frame.
    if let Ok(q) = crate::training::refine::quick::seen_of(config, m)
        && q.by_source
    {
        for f in files {
            let Some(rel) = export::parse_flat_name(f) else {
                continue;
            };
            let probe = ExportFrame {
                image: std::path::PathBuf::new(),
                name: f.to_string(),
                width: 0,
                height: 0,
                buckets: Vec::new(),
                split: Split::Train,
                labels: Default::default(),
                floor: None,
                rel: rel.clone(),
            };
            match q.saw(&probe) {
                Some(crate::training::refine::quick::Saw::Trained) => {
                    trained.insert(rel.jpg());
                }
                Some(crate::training::refine::quick::Saw::Validated) => {
                    validated.insert(rel.jpg());
                }
                None => {}
            }
        }
    }
    Seen::Frames { trained, validated }
}
