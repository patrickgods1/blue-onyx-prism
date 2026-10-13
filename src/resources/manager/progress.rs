//! CLI progress: [`cli_progress`] draws one indicatif bar per job from [`Status`] updates.

use super::*;

/// Progress hook drawing one indicatif bar per job (CLI).
pub fn cli_progress() -> ProgressFn {
    let multi = indicatif::MultiProgress::new();
    let bars: Mutex<BTreeMap<String, indicatif::ProgressBar>> = Mutex::new(BTreeMap::new());
    let style = indicatif::ProgressStyle::with_template(
        "{msg:40} [{bar:30}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
    )
    .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
    .progress_chars("=> ");
    Arc::new(move |s: &Status| {
        let mut bars = bars.lock_recover();
        let bar = bars.entry(s.key.clone()).or_insert_with(|| {
            let b = multi.add(indicatif::ProgressBar::new(s.size));
            b.set_style(style.clone());
            b.set_message(s.id.to_string());
            b
        });
        match &s.state {
            State::Queued => {}
            State::Downloading { bytes, total } => {
                bar.set_length(*total);
                bar.set_position(*bytes);
            }
            State::Verifying => bar.set_message(format!("{} (verifying)", s.id)),
            State::Extracting => bar.set_message(format!("{} (extracting)", s.id)),
            State::Installed => {
                bar.set_message(format!("{} installed", s.id));
                bar.finish();
            }
            State::Failed { .. } => bar.abandon_with_message(format!("{} failed", s.id)),
        }
    })
}
