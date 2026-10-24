//! The import's stage machine ([`ImportJob`], run on the training queue as a
//! [`Task`]): check the source -> set up the training environment (when videos, a URL or the
//! SigLIP selection need it) -> download (the URL with yt-dlp, pinned footage and the SigLIP 2
//! checkpoint through the download manager) -> extract frames (`import_footage.py extract`, one
//! run per video, several at a time, each video's share of the candidate budget as its
//! `--max-frames`) -> analyse (decode, dHash, scene on all cores but one, then the dedup; in
//! Rust) -> select (SigLIP embeddings
//! from `import_footage.py embed`, k-center per scene) -> write (the dataset layout, the review
//! queue, `imports.jsonl`).

use super::parallel;
use super::select;
use super::source::{self, ImportSource};
use super::write::{self, ImportLine};
use super::{ImportRequest, MAX_CANDIDATES_TOTAL, MAX_URL_BYTES, Selection};
use crate::collect::record::Provenance;
use crate::image_stats::ImageStats;
use crate::resources::catalog::{self, Platform, Resource};
use crate::resources::export::CommandRunner;
use crate::resources::export::train::TrainPaths;
use crate::training::capability::Accelerator;
use crate::training::pyenv::{self, check, stage};
use crate::training::task::{Task, TaskContext, TaskKind, TaskOutcome, TaskReport};
use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, SystemTime};

/// Stages of an import.
pub const STAGES: [&str; 7] = [
    "checking the source",
    "setting up the training environment",
    "downloading",
    "extracting frames",
    "analysing frames",
    "selecting frames",
    "writing frames",
];

/// A planned import ([`super::plan_import`]).
#[derive(Debug, Clone)]
pub struct ImportJob {
    pub id: String,
    pub request: ImportRequest,
    /// The licence the user confirmed (or the pinned dataset's, filled in at the download).
    pub license: String,
    /// When it was confirmed.
    pub confirmed_at: String,
    /// The dataset folder (`collect.dir`).
    pub dataset_dir: PathBuf,
    /// `<training dir>/imports/<id>`: downloads and extracted candidates (deleted at the end).
    pub work: PathBuf,
    pub paths: TrainPaths,
    pub platform: Platform,
    pub accelerator: Option<Accelerator>,
    pub root: PathBuf,
    pub timeout: Duration,
    /// Videos, a URL or the SigLIP selection: the training environment is used.
    pub needs_python: bool,
}

/// A frame before the selection.
#[derive(Debug, Clone)]
struct Candidate {
    file: PathBuf,
    origin: PathBuf,
    image_name: String,
    camera: String,
    base: SystemTime,
    offset: Option<f64>,
    kind: &'static str,
}

/// A video to extract and where its frames go.
#[derive(Debug, Clone)]
struct Video {
    path: PathBuf,
    camera: String,
    base: SystemTime,
    kind: &'static str,
}

/// What one extraction found.
#[derive(Debug, Default)]
struct Extracted {
    /// Frame files (in the video's output folder) and their time in seconds.
    found: Vec<(String, f64)>,
    /// Stopped at its `--max-frames`.
    capped: bool,
    /// `hw:<type>` or `sw` (the helper's `start` event; `sw` after a hardware fallback).
    decoder: Option<String>,
}

/// The report of a finished import.
#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub import_id: String,
    pub source: String,
    pub license: String,
    pub videos: usize,
    pub images: usize,
    /// Frames out of the extraction (and the images).
    pub candidates: usize,
    /// After the dHash dedup.
    pub unique: usize,
    pub frames: usize,
    pub scenes: BTreeMap<String, usize>,
    pub cameras: Vec<String>,
    pub selection: String,
    pub dataset_dir: PathBuf,
    /// The first written frames (`<scene>/<camera>/<date>/<stem>.jpg`).
    pub sample: Vec<String>,
    pub warnings: Vec<String>,
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// When a file was recorded: the time in a Blue Iris or MEVA name, else its modification time.
fn base_time(p: &Path) -> SystemTime {
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    source::blue_iris_time(&name)
        .or_else(|| source::meva_time(&name))
        .map(SystemTime::from)
        .or_else(|| mtime(p))
        .unwrap_or_else(SystemTime::now)
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

impl ImportJob {
    fn step(&self, report: &mut dyn FnMut(TaskReport<'_>), n: usize, pct: u8, detail: &str) {
        let name = STAGES.get(n.saturating_sub(1)).copied().unwrap_or("");
        stage(report, name, n, STAGES.len(), pct, detail);
    }

    fn device(&self) -> String {
        self.accelerator
            .as_ref()
            .map_or_else(|| "cpu".to_string(), Accelerator::device_arg)
    }

    fn siglip(&self) -> Result<&'static Resource> {
        catalog::autolabel_weights("siglip2-base")
            .ok_or_else(|| anyhow!("SigLIP 2 is not pinned in this build"))
    }

    fn provenance(&self, c: &Candidate, url: Option<&(String, serde_json::Value)>) -> Provenance {
        let dataset = match &self.request.source {
            ImportSource::Dataset(d) => catalog::import_dataset(d).map(|r| r.id.to_string()),
            _ => None,
        };
        Provenance {
            import_id: self.id.clone(),
            kind: if url.is_some() {
                "url".into()
            } else if dataset.is_some() {
                "dataset".into()
            } else {
                c.kind.into()
            },
            path: (url.is_none()).then(|| c.origin.display().to_string()),
            url: url.map(|(u, _)| u.clone()),
            attribution: dataset
                .as_ref()
                .map(|_| catalog::MEVA_ATTRIBUTION.to_string()),
            dataset,
            license: self.license.clone(),
            license_confirmed: !matches!(self.request.source, ImportSource::Dataset(_)),
            confirmed_at: (!matches!(self.request.source, ImportSource::Dataset(_)))
                .then(|| self.confirmed_at.clone()),
            site: url.map(|(_, s)| s.clone()),
            time_offset_s: c.offset.map(|o| (o * 1000.0).round() / 1000.0),
        }
    }

    /// Extract the frames of `video` into `out` (`import_footage.py extract`), at most `quota`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the per-video inputs of one helper run"
    )]
    fn extract_one(
        &self,
        runner: &dyn CommandRunner,
        script: &OsString,
        expr: &str,
        video: &Path,
        out: &Path,
        quota: usize,
        line: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<Extracted> {
        let mut ex = Extracted::default();
        let text = |ev: &serde_json::Value, k: &str| {
            ev.get(k)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        };
        pyenv::run_script(
            runner,
            &self.paths,
            self.platform,
            &self.work,
            self.timeout,
            vec![
                script.clone(),
                "extract".into(),
                "--video".into(),
                video.as_os_str().to_owned(),
                "--out".into(),
                out.as_os_str().to_owned(),
                "--select".into(),
                expr.into(),
                "--max-frames".into(),
                quota.to_string().into(),
                "--hwaccel".into(),
                self.request.hwaccel.as_str().into(),
            ],
            "import",
            &mut |ev, _| match ev.get("event").and_then(serde_json::Value::as_str) {
                Some("frame") => {
                    if let (Some(f), Some(t)) = (
                        text(ev, "file"),
                        ev.get("t").and_then(serde_json::Value::as_f64),
                    ) {
                        ex.found.push((f, t));
                    }
                }
                Some("start" | "decoder") => {
                    if let Some(d) = text(ev, "decoder") {
                        ex.decoder = Some(d);
                    }
                }
                Some("done") => {
                    ex.capped = ev.get("capped").and_then(serde_json::Value::as_bool) == Some(true);
                }
                _ => {}
            },
            &mut |rep| {
                if let TaskReport::Line(l) = rep {
                    line(l);
                }
            },
            cancel,
        )?;
        Ok(ex)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one linear stage machine reads best in one place"
    )]
    fn stages(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<TaskOutcome> {
        let r = &self.request;
        let mut warnings: Vec<String> = Vec::new();
        let mut license = self.license.clone();

        // 1. The source.
        self.step(report, 1, 0, &r.source.describe());
        let mut videos: Vec<Video> = Vec::new();
        let mut images: Vec<PathBuf> = Vec::new();
        match &r.source {
            ImportSource::Path(p) => {
                let d = source::discover(p)?;
                warnings.extend(d.warnings());
                for v in d.videos {
                    videos.push(Video {
                        camera: source::camera_of(&v, r.camera.as_deref()),
                        base: base_time(&v),
                        path: v,
                        kind: "video",
                    });
                }
                images = d.images;
            }
            ImportSource::Url(u) => {
                source::check_url(u)?;
            }
            ImportSource::Dataset(_) => {}
        }
        self.step(report, 1, 100, &r.source.describe());
        check(cancel)?;

        // 2. The environment.
        let _lock =
            crate::resources::export::train::TrainLock::try_acquire(&self.paths.train_lock_file())?;
        let scripts = if self.needs_python {
            let accel = self.accelerator.as_ref().ok_or_else(|| {
                anyhow!("no NVIDIA GPU or Apple silicon: the training environment cannot run here")
            })?;
            pyenv::setup(
                &self.paths,
                self.platform,
                accel,
                ctx.runner,
                ctx.fetcher,
                report,
                (STAGES[1], 2, STAGES.len()),
                cancel,
            )?;
            let (import, _) = crate::resources::export::train::write_task_scripts(&self.paths)?;
            self.step(report, 2, 95, "checking PyAV and yt-dlp");
            let mut seen = None;
            pyenv::run_script(
                ctx.runner,
                &self.paths,
                self.platform,
                &self.paths.scripts,
                Duration::from_secs(10 * 60),
                vec![import.clone().into_os_string(), "--self-check".into()],
                "import",
                &mut |ev, _| {
                    if ev.get("event").and_then(serde_json::Value::as_str) == Some("self_check") {
                        seen = Some(ev.clone());
                    }
                },
                report,
                cancel,
            )
            .context("the import helpers' self-check failed")?;
            let Some(ev) = seen else {
                bail!("the import helpers' self-check printed no report; the script is damaged");
            };
            let hw: Vec<&str> = ev
                .get("hwdevices")
                .and_then(serde_json::Value::as_array)
                .map(|a| a.iter().filter_map(serde_json::Value::as_str).collect())
                .unwrap_or_default();
            report(TaskReport::Line(&format!(
                "PyAV {} (FFmpeg {}), yt-dlp {}; hardware decoders: {}; video decoding: {}",
                ev.get("av")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("?"),
                ev.get("ffmpeg_license")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("?"),
                ev.get("yt_dlp")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("?"),
                if hw.is_empty() {
                    "none".to_string()
                } else {
                    hw.join(", ")
                },
                r.hwaccel.as_str()
            )));
            Some(import)
        } else {
            self.step(report, 2, 100, "not needed");
            None
        };
        check(cancel)?;
        let script = || -> Result<OsString> {
            scripts
                .clone()
                .map(PathBuf::into_os_string)
                .ok_or_else(|| anyhow!("internal: no import script"))
        };

        // 3. Downloads.
        let mut downloads: Vec<(&'static Resource, PathBuf)> = Vec::new();
        if r.selection == Selection::Siglip {
            let s = self.siglip()?;
            downloads.push((s, pyenv::weights_dir(&self.paths, s)));
        }
        let mut dataset_res = None;
        if let ImportSource::Dataset(d) = &r.source {
            let res = catalog::import_dataset(d).ok_or_else(|| anyhow!("unknown dataset {d}"))?;
            downloads.push((res, pyenv::dataset_dir(&self.root, res)));
            dataset_res = Some(res);
            license = res.license.to_string();
        }
        pyenv::check_large(&downloads, r.allow_large)?;
        pyenv::fetch_missing(
            ctx.fetcher,
            &downloads,
            report,
            (STAGES[2], 3, STAGES.len()),
            cancel,
        )?;
        if let Some(res) = dataset_res {
            let dir = pyenv::dataset_dir(&self.root, res);
            for p in res.parts {
                let v = dir.join(p.file_name);
                videos.push(Video {
                    camera: source::camera_of(&v, r.camera.as_deref()),
                    base: base_time(&v),
                    path: v,
                    kind: "video",
                });
            }
        }
        let mut url_info: Option<(String, serde_json::Value)> = None;
        if let ImportSource::Url(u) = &r.source {
            self.step(report, 3, 50, "downloading the video with yt-dlp");
            let dl = self.work.join("download");
            let mut fetched: Option<serde_json::Value> = None;
            pyenv::run_script(
                ctx.runner,
                &self.paths,
                self.platform,
                &self.work,
                self.timeout,
                vec![
                    script()?,
                    "fetch-url".into(),
                    "--url".into(),
                    u.into(),
                    "--out".into(),
                    dl.clone().into_os_string(),
                    "--max-bytes".into(),
                    MAX_URL_BYTES.to_string().into(),
                    "--max-height".into(),
                    "1080".into(),
                ],
                "import",
                &mut |ev, _| {
                    if ev.get("event").and_then(serde_json::Value::as_str) == Some("fetched") {
                        fetched = Some(ev.clone());
                    }
                },
                report,
                cancel,
            )
            .context("downloading the video")?;
            let ev = fetched.ok_or_else(|| anyhow!("yt-dlp reported no downloaded file"))?;
            let file = PathBuf::from(
                ev.get("file")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default(),
            );
            if !file.starts_with(&dl) || !file.is_file() {
                bail!("yt-dlp wrote no video into {}", dl.display());
            }
            let host = u
                .split("://")
                .nth(1)
                .and_then(|x| x.split(['/', '?', '#', ':']).next())
                .unwrap_or("web")
                .trim_start_matches("www.")
                .to_string();
            let camera = source::camera_of(&file, r.camera.as_deref().or(Some(&host)));
            let mut site = ev.clone();
            if let Some(o) = site.as_object_mut() {
                o.remove("event");
                o.remove("file");
            }
            if let Some(l) = site
                .get("license")
                .and_then(serde_json::Value::as_str)
                .filter(|l| {
                    let (a, b) = (l.to_ascii_lowercase(), license.to_ascii_lowercase());
                    !l.is_empty() && !a.contains(&b) && !b.contains(&a)
                })
            {
                warnings.push(format!(
                    "the site reports the licence \"{l}\"; you confirmed \"{license}\" (both are \
                     in the provenance)"
                ));
            }
            url_info = Some((u.clone(), site));
            videos.push(Video {
                path: file,
                camera,
                base: SystemTime::now(),
                kind: "video",
            });
        }
        self.step(report, 3, 100, "done");
        check(cancel)?;

        // 4. Extract: the candidate budget shared fairly by the cameras, their videos and the
        // images, then the videos `workers` at a time.
        let cams: Vec<&str> = videos.iter().map(|v| v.camera.as_str()).collect();
        let budget = select::budget(&cams, images.len(), MAX_CANDIDATES_TOTAL, r.max_candidates);
        warnings.extend(budget.warnings(MAX_CANDIDATES_TOTAL));
        let used_images: Vec<&PathBuf> = select::stride(images.len(), budget.images)
            .into_iter()
            .filter_map(|i| images.get(i))
            .collect();
        let todo: Vec<(usize, usize)> = budget
            .per_video
            .iter()
            .enumerate()
            .filter(|(_, q)| **q > 0)
            .map(|(i, q)| (i, *q))
            .collect();
        let mut cands: Vec<Candidate> = Vec::new();
        if !todo.is_empty() {
            let script = script()?;
            let expr = r.select_expr();
            let total = todo.len();
            let workers = parallel::extract_workers(r.workers, total);
            let names: Vec<String> = todo
                .iter()
                .map(|(i, _)| {
                    videos
                        .get(*i)
                        .map(|v| file_name(&v.path))
                        .unwrap_or_default()
                })
                .collect();
            self.step(
                report,
                4,
                0,
                &format!(
                    "{total} video(s), {workers} at a time, up to {} frames each",
                    budget.max_per_video()
                ),
            );
            let runner = ctx.runner;
            let work = |k: usize, line: &mut dyn FnMut(&str)| -> Result<Extracted> {
                let (i, quota) = todo
                    .get(k)
                    .copied()
                    .ok_or_else(|| anyhow!("internal: no video {k}"))?;
                let v = videos
                    .get(i)
                    .ok_or_else(|| anyhow!("internal: no video {i}"))?;
                let name = file_name(&v.path);
                let out = self.work.join("frames").join(format!("v{i:04}"));
                let ex = self.extract_one(
                    runner,
                    &script,
                    &expr,
                    &v.path,
                    &out,
                    quota,
                    &mut |l| line(&format!("{name}: {l}")),
                    cancel,
                )?;
                line(&format!(
                    "{name}: {} candidate frames ({})",
                    ex.found.len(),
                    ex.decoder.as_deref().unwrap_or("decoder not reported")
                ));
                Ok(ex)
            };
            let mut running: Vec<usize> = Vec::new();
            let mut done = 0usize;
            let results = parallel::run_with_lines(
                total,
                workers,
                &work,
                &mut |e| {
                    match e {
                        parallel::Event::Line(_, l) => {
                            report(TaskReport::Line(l));
                            return;
                        }
                        parallel::Event::Started(k) => running.push(k),
                        parallel::Event::Finished(k) => {
                            running.retain(|x| *x != k);
                            done += 1;
                        }
                    }
                    let reading: Vec<&str> = running
                        .iter()
                        .filter_map(|k| names.get(*k).map(String::as_str))
                        .collect();
                    let mut detail = format!("videos {done}/{total} done");
                    if !reading.is_empty() {
                        detail.push_str("; reading ");
                        detail.push_str(&reading.join(", "));
                    }
                    let pct = u8::try_from(done * 100 / total.max(1)).unwrap_or(100);
                    self.step(report, 4, pct, &detail);
                },
                cancel,
            );
            // In video order: the dedup keeps the first of near duplicates.
            for (k, res) in results.into_iter().enumerate() {
                let Some(((i, quota), res)) = todo.get(k).copied().zip(res) else {
                    continue;
                };
                let Some(v) = videos.get(i) else { continue };
                let ex = match res {
                    Ok(ex) => ex,
                    Err(e) => {
                        if e.downcast_ref::<crate::resources::export::Cancelled>()
                            .is_some()
                        {
                            return Err(e);
                        }
                        warnings.push(format!("{}: not read ({e:#})", file_name(&v.path)));
                        continue;
                    }
                };
                if ex.capped {
                    warnings.push(format!(
                        "{}: stopped after {quota} frames (its share of the {MAX_CANDIDATES_TOTAL} \
                         candidate frames, at most {} per video); the rest of the video was not \
                         read",
                        file_name(&v.path),
                        r.max_candidates
                    ));
                }
                let out = self.work.join("frames").join(format!("v{i:04}"));
                for (f, t) in ex.found {
                    if f.contains(['/', '\\']) || f.contains("..") {
                        continue;
                    }
                    cands.push(Candidate {
                        file: out.join(f),
                        origin: v.path.clone(),
                        image_name: file_name(&v.path),
                        camera: v.camera.clone(),
                        base: v.base,
                        offset: Some(t),
                        kind: v.kind,
                    });
                }
            }
            check(cancel)?;
        }
        for img in used_images {
            cands.push(Candidate {
                file: img.clone(),
                origin: img.clone(),
                image_name: file_name(img),
                camera: source::camera_of(img, r.camera.as_deref()),
                base: base_time(img),
                offset: None,
                kind: "image",
            });
        }
        // A guard only: the budget keeps the candidates within the total.
        if cands.len() > MAX_CANDIDATES_TOTAL {
            warnings.push(format!(
                "{} candidate frames; only the first {MAX_CANDIDATES_TOTAL} are considered",
                cands.len()
            ));
            cands.truncate(MAX_CANDIDATES_TOTAL);
        }
        self.step(report, 4, 100, &format!("{} candidate frames", cands.len()));
        if cands.is_empty() {
            bail!(
                "no frames to import{}",
                if warnings.is_empty() {
                    String::new()
                } else {
                    format!(": {}", warnings.join("; "))
                }
            );
        }
        check(cancel)?;

        // 5. Analyse: decode, dHash, scene (in parallel); dedup per camera.
        let n = cands.len();
        let threads = parallel::analyse_threads(n);
        self.step(report, 5, 0, &format!("0 of {n} ({threads} threads)"));
        let analyse = |i: usize| -> Result<(u64, ImageStats, u32, u32)> {
            let c = cands
                .get(i)
                .ok_or_else(|| anyhow!("internal: no candidate {i}"))?;
            let b = std::fs::read(&c.file)?;
            let img = crate::image::decode(&b)?;
            Ok((
                crate::collect::dhash::of_rgb(&img),
                ImageStats::compute(&img),
                img.width,
                img.height,
            ))
        };
        let analysed = parallel::map(
            n,
            threads,
            &analyse,
            &mut |done| {
                let pct = u8::try_from(done * 100 / n.max(1)).unwrap_or(100);
                self.step(
                    report,
                    5,
                    pct,
                    &format!("{done} of {n} ({threads} threads)"),
                );
            },
            Duration::from_millis(250),
            cancel,
        );
        check(cancel)?;
        let analysed = analysed.ok_or_else(|| anyhow!("internal: the analysis stopped early"))?;
        let mut info: Vec<Option<(u64, ImageStats, u32, u32)>> = Vec::with_capacity(n);
        for (c, a) in cands.iter().zip(analysed) {
            match a {
                Ok(x) => info.push(Some(x)),
                Err(e) => {
                    warnings.push(format!("{}: not decoded ({e:#})", c.file.display()));
                    info.push(None);
                }
            }
        }
        let ok: Vec<usize> = (0..n)
            .filter(|i| info.get(*i).is_some_and(Option::is_some))
            .collect();
        let keys: Vec<(u64, &str, f64)> = ok
            .iter()
            .filter_map(|i| {
                let h = info.get(*i)?.as_ref()?.0;
                let c = cands.get(*i)?;
                let t = write::frame_time(c.base, c.offset)
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map_or(0.0, |d| d.as_secs_f64());
                Some((h, c.camera.as_str(), t))
            })
            .collect();
        let unique: Vec<usize> = select::dedup(&keys, r.dedup_hamming, f64::from(r.dedup_window_s))
            .into_iter()
            .filter_map(|k| ok.get(k).copied())
            .collect();
        self.step(
            report,
            5,
            100,
            &format!(
                "{} unique of {n} (dHash within {} bits in {} s)",
                unique.len(),
                r.dedup_hamming,
                r.dedup_window_s
            ),
        );
        check(cancel)?;

        // 6. Select.
        let groups: Vec<String> = unique
            .iter()
            .filter_map(|i| {
                info.get(*i)?
                    .as_ref()
                    .map(|x| x.1.scene().as_str().to_string())
            })
            .collect();
        let hashes: Vec<u64> = unique
            .iter()
            .filter_map(|i| info.get(*i)?.as_ref().map(|x| x.0))
            .collect();
        let embeddings = if r.selection == Selection::Siglip && unique.len() > r.frames {
            self.step(report, 6, 0, "SigLIP 2 embeddings");
            let list = self.work.join("candidates.txt");
            let text: String = unique
                .iter()
                .filter_map(|i| cands.get(*i))
                .map(|c| format!("{}\n", c.file.display()))
                .collect();
            std::fs::create_dir_all(&self.work)?;
            std::fs::write(&list, text)?;
            let emb = self.work.join("embeddings.f32");
            let s = self.siglip()?;
            let mut dim = None;
            let total = unique.len();
            pyenv::run_script(
                ctx.runner,
                &self.paths,
                self.platform,
                &self.work,
                self.timeout,
                vec![
                    script()?,
                    "embed".into(),
                    "--list".into(),
                    list.into_os_string(),
                    "--model".into(),
                    pyenv::weights_dir(&self.paths, s).into_os_string(),
                    "--out".into(),
                    emb.clone().into_os_string(),
                    "--device".into(),
                    self.device().into(),
                ],
                "import",
                &mut |ev, report| match ev.get("event").and_then(serde_json::Value::as_str) {
                    Some("progress") => {
                        let done = ev
                            .get("done")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        let pct =
                            u8::try_from(done * 90 / u64::try_from(total.max(1)).unwrap_or(1))
                                .unwrap_or(90);
                        stage(
                            report,
                            STAGES[5],
                            6,
                            STAGES.len(),
                            pct,
                            &format!("SigLIP 2 embeddings {done} of {total}"),
                        );
                    }
                    Some("done") => dim = ev.get("dim").and_then(serde_json::Value::as_u64),
                    _ => {}
                },
                report,
                cancel,
            )
            .context("computing the SigLIP 2 embeddings")?;
            Some(read_embeddings(&emb, total, dim)?)
        } else {
            None
        };
        let group_refs: Vec<&str> = groups.iter().map(String::as_str).collect();
        let picked = select::pick(
            &group_refs,
            &hashes,
            embeddings.as_deref(),
            r.frames,
            r.balance,
        );
        let chosen: Vec<usize> = picked
            .iter()
            .filter_map(|p| unique.get(*p).copied())
            .collect();
        self.step(
            report,
            6,
            100,
            &format!(
                "{} of {} frames ({})",
                chosen.len(),
                unique.len(),
                if embeddings.is_some() {
                    "SigLIP 2 k-center"
                } else {
                    "dHash k-center"
                }
            ),
        );
        check(cancel)?;

        // 7. Write.
        let frames: Vec<write::Frame> = chosen
            .iter()
            .filter_map(|i| {
                let c = cands.get(*i)?;
                let (h, stats, w, hh) = info.get(*i)?.as_ref()?;
                Some(write::Frame {
                    file: c.file.clone(),
                    image_name: c.image_name.clone(),
                    camera: c.camera.clone(),
                    time: write::frame_time(c.base, c.offset),
                    width: *w,
                    height: *hh,
                    stats: *stats,
                    dhash: *h,
                    provenance: Provenance {
                        license: license.clone(),
                        ..self.provenance(c, url_info.as_ref())
                    },
                })
            })
            .collect();
        self.step(report, 7, 0, &format!("{} frames", frames.len()));
        let (rels, skipped) = write::write_frames(&self.dataset_dir, &self.id, &frames)?;
        if skipped > 0 {
            warnings.push(format!(
                "{skipped} frame(s) were imported before (same camera, time and picture) and \
                 skipped"
            ));
        }
        crate::collect::layout::append_review(&self.dataset_dir, &rels)
            .context("queueing the imported frames for review")?;
        let mut scenes: BTreeMap<String, usize> = BTreeMap::new();
        let mut cameras: Vec<String> = Vec::new();
        for f in frames.iter().filter(|f| {
            rels.iter()
                .any(|r| r.contains(&crate::collect::utc_file_ts(f.time)) && r.contains(&f.camera))
        }) {
            *scenes
                .entry(f.stats.scene().as_str().to_string())
                .or_default() += 1;
            if !cameras.contains(&f.camera) {
                cameras.push(f.camera.clone());
            }
        }
        write::append_import(
            &self.dataset_dir,
            &ImportLine {
                id: self.id.clone(),
                created: crate::collect::utc_rfc3339(SystemTime::now()),
                source: r.source.describe(),
                kind: match &r.source {
                    ImportSource::Path(_) => "path",
                    ImportSource::Url(_) => "url",
                    ImportSource::Dataset(_) => "dataset",
                }
                .into(),
                license: license.clone(),
                attribution: dataset_res.map(|_| catalog::MEVA_ATTRIBUTION.to_string()),
                frames: rels.len(),
                cameras: cameras.clone(),
                scenes: scenes.clone(),
            },
        )?;
        crate::collect::global().rescan_if_running(&self.dataset_dir);
        if !r.keep_work {
            let _ = crate::resources::manager::remove_any(&self.work);
        }
        self.step(report, 7, 100, &format!("{} frames", rels.len()));
        let rep = ImportReport {
            import_id: self.id.clone(),
            source: r.source.describe(),
            license,
            videos: videos.len(),
            images: images.len(),
            candidates: n,
            unique: unique.len(),
            frames: rels.len(),
            scenes,
            cameras,
            selection: if embeddings.is_some() {
                Selection::Siglip.as_str().into()
            } else {
                Selection::Dhash.as_str().into()
            },
            dataset_dir: self.dataset_dir.clone(),
            sample: rels.iter().take(20).cloned().collect(),
            warnings,
        };
        let summary = format!(
            "{} frames from {} ({} candidates, {} unique; {}) in {}; queued for review",
            rep.frames,
            rep.source,
            rep.candidates,
            rep.unique,
            rep.scenes
                .iter()
                .map(|(k, v)| format!("{v} {k}"))
                .collect::<Vec<_>>()
                .join(", "),
            self.dataset_dir.display()
        );
        Ok(TaskOutcome {
            summary,
            result: serde_json::to_value(&rep)?,
        })
    }
}

/// Read `n` L2-normalized float32 rows (little endian) of `dim` values from `path`.
pub fn read_embeddings(path: &Path, n: usize, dim: Option<u64>) -> Result<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let floats = bytes.len() / 4;
    let d = dim
        .and_then(|d| usize::try_from(d).ok())
        .unwrap_or(floats / n);
    if d == 0 || floats != n * d || bytes.len() % 4 != 0 {
        bail!(
            "{}: {} bytes are not {n} embeddings of {d} floats",
            path.display(),
            bytes.len()
        );
    }
    let all: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok(all.chunks(d).map(<[f32]>::to_vec).collect())
}

impl Task for ImportJob {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> TaskKind {
        TaskKind::Import
    }
    fn title(&self) -> String {
        format!(
            "import of {} ({} frames, {})",
            self.request.source.describe(),
            self.request.frames,
            self.request.selection.as_str()
        )
    }
    fn steps(&self) -> usize {
        STAGES.len()
    }
    fn run(
        &self,
        ctx: TaskContext<'_>,
        report: &mut dyn FnMut(TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> Result<TaskOutcome> {
        let r = self.stages(ctx, report, cancel);
        if r.is_err() && !self.request.keep_work {
            let _ = crate::resources::manager::remove_any(&self.work);
        }
        r
    }
}

#[cfg(test)]
mod tests;
