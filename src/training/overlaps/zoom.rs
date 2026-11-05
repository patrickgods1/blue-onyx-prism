//! Stage 3 of the overlap check: `scripts/autolabel.py --zoom-pairs` runs RF-DETR L on the crops of
//! [`super::select`], one flushed JSONL line per finished crop (`@overlaps` events carry the
//! progress). The answers ([`read_results`]) are detections only; every decision is the service's
//! ([`crate::collect::labels::overlaps`]).

use super::select::{Crop, Selected};
use crate::collect::labels::overlaps::{Det, ZOOM_FLOOR, ZOOM_PAD};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::Path;

/// Event tag of the script in this mode.
pub const TAG: &str = "overlaps";

/// One frame of the script's input.
#[derive(Debug, Serialize)]
struct FrameLine<'a> {
    frame: &'a str,
    path: &'a Path,
    crops: &'a [Crop],
}

/// One crop of the script's output.
#[derive(Debug, Deserialize)]
struct ZoomLine {
    id: usize,
    #[serde(default)]
    boxes: Option<Vec<Det>>,
}

/// What a crop answered.
#[derive(Debug, Clone, PartialEq)]
pub enum CropResult {
    /// Its detections, in frame coordinates.
    Boxes(Vec<Det>),
    /// The frame could not be read: there is no zoom answer.
    Unreadable,
}

/// The script's input: one line per frame with crops.
pub fn crop_list(frames: &[Selected]) -> Result<String> {
    let mut text = String::new();
    for f in frames.iter().filter(|f| !f.crops.is_empty()) {
        text.push_str(&serde_json::to_string(&FrameLine {
            frame: &f.rel,
            path: &f.jpg,
            crops: &f.crops,
        })?);
        text.push('\n');
    }
    Ok(text)
}

/// The script's arguments after its path.
pub fn args(
    list: &Path,
    out: &Path,
    weights: &Path,
    classes: &[String],
    device: &str,
) -> Result<Vec<OsString>> {
    Ok(vec![
        "--zoom-pairs".into(),
        list.as_os_str().to_owned(),
        "--out".into(),
        out.as_os_str().to_owned(),
        "--rfdetr".into(),
        weights.as_os_str().to_owned(),
        "--classes".into(),
        serde_json::to_string(classes)?.into(),
        "--device".into(),
        device.into(),
        "--rfdetr-threshold".into(),
        ZOOM_FLOOR.to_string().into(),
        "--zoom-pad".into(),
        ZOOM_PAD.to_string().into(),
    ])
}

/// The finished crops of `path` by id. `partial`: the run stopped (a cancel, a failure), so the file
/// may be missing or end in a line cut short; that line is left out. Any other line that does not
/// parse is a warning: its crop has no answer, and a rerun does it again.
pub fn read_results(
    path: &Path,
    partial: bool,
    warnings: &mut Vec<String>,
) -> Result<BTreeMap<usize, CropResult>> {
    let text = match std::fs::read_to_string(path) {
        Err(e) if partial && e.kind() == std::io::ErrorKind::NotFound => String::new(),
        t => t.with_context(|| format!("reading {}", path.display()))?,
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut out = BTreeMap::new();
    for (i, line) in lines.iter().enumerate() {
        match serde_json::from_str::<ZoomLine>(line) {
            Ok(z) => {
                out.insert(
                    z.id,
                    z.boxes.map_or(CropResult::Unreadable, CropResult::Boxes),
                );
            }
            Err(_) if partial && i + 1 == lines.len() => {}
            Err(e) => warnings.push(format!("a zoom result line does not parse: {e}")),
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn results_read_what_finished_and_a_cut_line_is_no_error() {
        let d = std::env::temp_dir().join(format!("bop-zoom-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("zoom.jsonl");
        let mut warnings = Vec::new();
        assert!(read_results(&p, true, &mut warnings).unwrap().is_empty());
        assert!(read_results(&p, false, &mut warnings).is_err(), "no file");
        std::fs::write(
            &p,
            "{\"id\":0,\"frame\":\"f\",\"boxes\":[{\"label\":\"car\",\"bbox\":[0,0,9,9],\"score\":0.7}]}\n\
             {\"id\":1,\"frame\":\"f\",\"error\":\"broken\"}\n\
             {\"id\":2,\"fr",
        )
        .unwrap();
        let got = read_results(&p, true, &mut warnings).unwrap();
        assert_eq!(got.len(), 2);
        assert!(matches!(&got[&0], CropResult::Boxes(b) if b.len() == 1 && b[0].score == 0.7));
        assert_eq!(got[&1], CropResult::Unreadable);
        assert!(warnings.is_empty());
        // A bad line in a finished run: a warning.
        read_results(&p, false, &mut warnings).unwrap();
        assert_eq!(warnings.len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }
}
