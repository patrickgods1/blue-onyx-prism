//! The import stage machine over fakes (no network, no Python): the export's [`Fake`] for uv and
//! downloads, plus a fake `import_footage.py` that "extracts" synthetic JPEGs, "downloads" a
//! URL and writes embeddings.

use super::*;
use crate::backend::detect::{HardwareInfo, apple_gpu};
use crate::config::Config;
use crate::resources::export::fake::{Fake, arg_after};
use crate::resources::export::{Cmd, CommandRunner, Fetcher};
use crate::training::capability::{self, Capability};
use crate::training::import::{ImportRequest, plan_import};
use crate::training::service::{RunState, TrainingService};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tmp(tag: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!("bop-import-job-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

/// A synthetic frame: pattern `i` (distinct dHashes), gray (IR) or coloured (day).
fn jpeg(i: usize, gray: bool) -> Vec<u8> {
    let (w, h) = (48u32, 32u32);
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            // A different stripe frequency and phase per pattern.
            let f = 1 + (i % 7) as u32;
            let v = (((x * f + y * (i as u32 % 5) + (i as u32) * 13) % 32) * 8) as u8;
            if gray {
                rgb.extend([v, v, v]);
            } else {
                rgb.extend([v, 255 - v, v / 3 + 40]);
            }
        }
    }
    crate::image::encode_jpeg(
        &crate::image::RgbImage {
            width: w,
            height: h,
            rgb,
        },
        92,
    )
    .unwrap()
}

#[derive(Default)]
struct ImportFake {
    inner: Fake,
    /// Frames each extraction writes; the first `gray` of them gray.
    frames: usize,
    gray: usize,
    /// Every frame of a video is the same picture (dedup test).
    duplicates: bool,
    cmds: Mutex<Vec<Cmd>>,
}

impl ImportFake {
    fn new(frames: usize, gray: usize) -> Self {
        Self {
            inner: Fake::new(),
            frames,
            gray,
            ..Default::default()
        }
    }

    fn script_cmds(&self, sub: &str) -> Vec<Cmd> {
        self.cmds
            .lock()
            .unwrap()
            .iter()
            .filter(|c| c.args.iter().any(|a| a == sub))
            .cloned()
            .collect()
    }
}

impl CommandRunner for ImportFake {
    fn run(&self, cmd: &Cmd, on_line: &mut dyn FnMut(&str), cancel: &AtomicBool) -> Result<()> {
        let d = cmd.display();
        if !d.contains("import_footage.py") {
            return self.inner.run(cmd, on_line, cancel);
        }
        self.cmds.lock().unwrap().push(cmd.clone());
        if cmd.args.iter().any(|a| a == "--self-check") {
            on_line(
                r#"@import {"event":"self_check","av":"18.1.0","yt_dlp":"2026.08.19","ffmpeg_license":"LGPL version 3 or later"}"#,
            );
        } else if cmd.args.iter().any(|a| a == "extract") {
            let out = arg_after(cmd, "--out").unwrap();
            std::fs::create_dir_all(&out).unwrap();
            for i in 0..self.frames {
                let name = format!("f_{i:06}.jpg");
                let pat = if self.duplicates { 0 } else { i };
                std::fs::write(out.join(&name), jpeg(pat, i < self.gray)).unwrap();
                on_line(&format!(
                    "@import {{\"event\":\"frame\",\"file\":\"{name}\",\"t\":{}.5,\"w\":48,\"h\":32}}",
                    i * 10
                ));
            }
            on_line("[import_footage] some log line");
            on_line(&format!(
                "@import {{\"event\":\"done\",\"frames\":{},\"capped\":false}}",
                self.frames
            ));
        } else if cmd.args.iter().any(|a| a == "fetch-url") {
            let out = arg_after(cmd, "--out").unwrap();
            std::fs::create_dir_all(&out).unwrap();
            let f = out.join("video.mp4");
            std::fs::write(&f, b"video").unwrap();
            on_line(&format!(
                "@import {}",
                serde_json::json!({"event": "fetched", "file": f, "size": 5, "title": "A street",
                    "uploader": "someone", "license": "Creative Commons Attribution license (reuse allowed)",
                    "webpage_url": "https://example.org/v/1", "extractor": "generic"})
            ));
        } else if cmd.args.iter().any(|a| a == "embed") {
            let list = arg_after(cmd, "--list").unwrap();
            let out = arg_after(cmd, "--out").unwrap();
            let n = std::fs::read_to_string(&list).unwrap().lines().count();
            let mut bytes = Vec::new();
            for i in 0..n {
                // Two tight clusters plus one outlier at the end.
                let v: [f32; 4] = if i + 1 == n {
                    [0.0, 0.0, 0.0, 1.0]
                } else if i % 2 == 0 {
                    [1.0, 0.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0, 0.0]
                };
                for x in v {
                    bytes.extend(x.to_le_bytes());
                }
            }
            std::fs::write(&out, bytes).unwrap();
            on_line(&format!(
                "@import {{\"event\":\"progress\",\"done\":{n},\"images\":{n}}}"
            ));
            on_line(&format!(
                "@import {{\"event\":\"done\",\"images\":{n},\"dim\":4}}"
            ));
        }
        Ok(())
    }
}

impl Fetcher for ImportFake {
    fn fetch(
        &self,
        res: &'static Resource,
        dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.inner.fetch(res, dir, progress, cancel)
    }
}

fn mac() -> Capability {
    capability::capability(
        &HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]),
        Some(16.0),
    )
}

fn config(root: &Path) -> Config {
    Config {
        download_dir: Some(root.to_path_buf()),
        allow_large_downloads: true,
        ..Config::default()
    }
}

fn run(job: &ImportJob, fake: &ImportFake) -> (Result<TaskOutcome>, Vec<String>) {
    let cancel = AtomicBool::new(false);
    let mut stages = Vec::new();
    let r = job.run(
        TaskContext {
            runner: fake,
            fetcher: fake,
        },
        &mut |p| {
            if let TaskReport::Stage { stage, .. } = p
                && stages.last().is_none_or(|s: &String| s != stage)
            {
                stages.push(stage.to_string());
            }
        },
        &cancel,
    );
    (r, stages)
}

fn records(dir: &Path) -> Vec<crate::collect::Record> {
    crate::collect::labels::queue::candidates(
        dir,
        crate::collect::labels::queue::QueueSource::Queue,
    )
    .iter()
    .map(|c| {
        crate::collect::labels::Frame::resolve(dir, &c.rel)
            .unwrap()
            .record()
            .unwrap()
    })
    .collect()
}

#[test]
fn a_folder_import_extracts_dedups_selects_and_writes_reviewable_frames() {
    let t = tmp("folder");
    let root = &t.0;
    let clips = root.join("BlueIris");
    std::fs::create_dir_all(clips.join("Alerts")).unwrap();
    std::fs::write(clips.join("Driveway.20240315_083012.mp4"), b"v").unwrap();
    std::fs::write(clips.join("Driveway.20240315_090000.bvr"), b"v").unwrap();
    std::fs::write(
        clips.join("Alerts/Porch.20240316_220000.123.jpg"),
        jpeg(3, true),
    )
    .unwrap();
    let cfg = config(root);
    let mut req = ImportRequest::new(ImportSource::Path(clips.clone()));
    req.license_confirmed = true;
    req.frames = 4;
    let job = plan_import(&cfg, req, &mac()).unwrap();
    let fake = ImportFake::new(8, 3);
    let (r, stages) = run(&job, &fake);
    let out = r.unwrap();
    assert_eq!(stages, STAGES.map(String::from).to_vec());
    let rep: serde_json::Value = out.result;
    assert_eq!(rep["videos"], 1);
    assert_eq!(rep["images"], 1);
    assert_eq!(rep["candidates"], 9);
    assert_eq!(rep["frames"], 4);
    assert_eq!(rep["selection"], "siglip");
    assert!(
        rep["warnings"][0].as_str().unwrap().contains(".bvr"),
        "{rep}"
    );
    // The select expression and the frame cap went to the extraction.
    let ex = fake.script_cmds("extract");
    assert_eq!(ex.len(), 1);
    assert!(
        arg_after(&ex[0], "--select")
            .unwrap()
            .to_string_lossy()
            .contains("gt(scene,0.01)")
    );
    assert_eq!(fake.script_cmds("embed").len(), 1);
    // Frames in the dataset, queued for review, with provenance.
    let dir = root.join("dataset");
    let recs = records(&dir);
    assert_eq!(recs.len(), 4);
    for rec in &recs {
        assert_eq!(
            rec.source.as_deref(),
            Some(format!("import:{}", job.id).as_str())
        );
        let p = rec.provenance.as_ref().unwrap();
        assert_eq!(p.license, "own footage");
        assert!(p.license_confirmed && p.confirmed_at.is_some());
        assert!(rec.predictions.is_empty());
    }
    // Both scenes are represented (balanced), cameras from the Blue Iris names.
    let scenes: std::collections::BTreeSet<&str> = recs.iter().map(|r| r.scene.as_str()).collect();
    assert!(
        scenes.contains("ir") && scenes.contains("day"),
        "{scenes:?}"
    );
    let cams: std::collections::BTreeSet<&str> = recs.iter().map(|r| r.camera.as_str()).collect();
    assert!(
        cams.iter().all(|c| *c == "driveway" || *c == "porch"),
        "{cams:?}"
    );
    let video_frame = recs
        .iter()
        .find(|r| r.camera == "driveway")
        .expect("a video frame");
    let p = video_frame.provenance.as_ref().unwrap();
    assert!(p.time_offset_s.unwrap() >= 0.5);
    assert!(
        video_frame.timestamp.starts_with("2024-03-15T08:"),
        "{}",
        video_frame.timestamp
    );
    let imports = write::read_imports(&dir);
    assert_eq!(imports.len(), 1);
    assert_eq!(imports[0].frames, 4);
    // The work folder is gone.
    assert!(!job.work.exists());
}

#[test]
fn duplicates_are_dropped_and_dhash_selection_runs_without_embeddings() {
    let t = tmp("dup");
    let root = &t.0;
    std::fs::write(root.join("clip.mp4"), b"v").unwrap();
    let mut req = ImportRequest::new(ImportSource::Path(root.join("clip.mp4")));
    req.license = Some("CC BY 4.0".into());
    req.license_confirmed = true;
    req.selection = Selection::Dhash;
    // Any time apart (the fake's frames are 10 s apart, the default window).
    req.dedup_window_s = 0.0;
    let job = plan_import(&config(root), req, &mac()).unwrap();
    let mut fake = ImportFake::new(6, 0);
    fake.duplicates = true;
    let (r, _) = run(&job, &fake);
    let rep = r.unwrap().result;
    assert_eq!(rep["candidates"], 6);
    assert_eq!(rep["unique"], 1, "{rep}");
    assert_eq!(rep["frames"], 1);
    assert_eq!(rep["selection"], "dhash");
    assert!(fake.script_cmds("embed").is_empty());
    assert_eq!(
        records(&root.join("dataset"))[0]
            .provenance
            .as_ref()
            .unwrap()
            .license,
        "CC BY 4.0"
    );
}

#[test]
fn an_image_folder_with_dhash_selection_needs_no_python() {
    let t = tmp("images");
    let root = &t.0;
    std::fs::create_dir_all(root.join("snaps")).unwrap();
    for i in 0..5 {
        std::fs::write(
            root.join(format!("snaps/Gate.2024010{i}_120000.jpg")),
            jpeg(i, i % 2 == 0),
        )
        .unwrap();
    }
    let mut req = ImportRequest::new(ImportSource::Path(root.join("snaps")));
    req.license_confirmed = true;
    req.selection = Selection::Dhash;
    req.frames = 3;
    let mut cfg = config(root);
    cfg.allow_large_downloads = false;
    let none = capability::capability(&HardwareInfo::new("linux", "x86_64", vec![]), None);
    let job = plan_import(&cfg, req, &none).unwrap();
    let fake = ImportFake::new(0, 0);
    let (r, _) = run(&job, &fake);
    assert_eq!(r.unwrap().result["frames"], 3);
    assert_eq!(fake.inner.runs.load(Ordering::SeqCst), 0, "no process ran");
    assert!(
        fake.inner.fetched.lock().unwrap().is_empty(),
        "nothing downloaded"
    );
    let recs = records(&root.join("dataset"));
    assert!(recs.iter().all(|r| r.camera == "gate"));
    assert!(
        recs.iter()
            .all(|r| r.provenance.as_ref().unwrap().kind == "image")
    );
}

#[test]
fn a_url_import_records_the_site_and_the_confirmed_licence() {
    let t = tmp("url");
    let root = &t.0;
    let mut req = ImportRequest::new(ImportSource::Url("https://www.example.org/v/1".into()));
    req.license = Some("CC BY 4.0".into());
    req.license_confirmed = true;
    req.selection = Selection::Dhash;
    req.frames = 2;
    let job = plan_import(&config(root), req, &mac()).unwrap();
    let fake = ImportFake::new(4, 0);
    let (r, _) = run(&job, &fake);
    let out = r.unwrap();
    assert_eq!(out.result["frames"], 2);
    assert!(
        out.result["warnings"][0]
            .as_str()
            .unwrap()
            .contains("the site reports the licence"),
        "{}",
        out.result
    );
    let rec = &records(&root.join("dataset"))[0];
    let p = rec.provenance.as_ref().unwrap();
    assert_eq!(p.kind, "url");
    assert_eq!(p.url.as_deref(), Some("https://www.example.org/v/1"));
    assert_eq!(p.site.as_ref().unwrap()["title"], "A street");
    assert_eq!(p.license, "CC BY 4.0");
    assert!(p.path.is_none());
    assert_eq!(rec.camera, "example_org");
    let fetch = fake.script_cmds("fetch-url");
    assert_eq!(fetch.len(), 1);
    assert!(fetch[0].display().contains("--max-height 1080"));
}

#[test]
fn pinned_footage_is_downloaded_and_attributed() {
    let t = tmp("meva");
    let root = &t.0;
    let mut req = ImportRequest::new(ImportSource::Dataset("meva-sample".into()));
    req.selection = Selection::Dhash;
    req.frames = 5;
    let job = plan_import(&config(root), req, &mac()).unwrap();
    let fake = ImportFake::new(2, 1);
    let (r, _) = run(&job, &fake);
    let out = r.unwrap();
    assert_eq!(out.result["license"], catalog::MEVA_LICENSE);
    assert_eq!(out.result["videos"], 6);
    assert!(
        fake.inner
            .fetched
            .lock()
            .unwrap()
            .contains(&"import-data:meva-sample".to_string())
    );
    let recs = records(&root.join("dataset"));
    assert_eq!(recs.len(), 5);
    let p = recs[0].provenance.as_ref().unwrap();
    assert_eq!(p.kind, "dataset");
    assert_eq!(p.dataset.as_deref(), Some("import-data:meva-sample"));
    assert!(p.attribution.as_deref().unwrap().contains("CC BY 4.0"));
    assert!(
        !p.license_confirmed,
        "nothing to confirm for pinned footage"
    );
    assert!(recs.iter().all(|r| r.camera.starts_with("meva-")));
}

#[test]
fn imports_run_on_the_training_queue_and_can_be_cancelled_while_queued() {
    let t = tmp("queue");
    let root = &t.0;
    std::fs::create_dir_all(root.join("snaps")).unwrap();
    for i in 0..3 {
        std::fs::write(root.join(format!("snaps/s{i}.jpg")), jpeg(i, false)).unwrap();
    }
    let mut req = ImportRequest::new(ImportSource::Path(root.join("snaps")));
    req.license_confirmed = true;
    req.selection = Selection::Dhash;
    let cfg = config(root);
    let fake = Arc::new(ImportFake::new(0, 0));
    let svc = TrainingService::start(
        fake.clone(),
        fake.clone(),
        Arc::new(crate::training::job::NoHooks),
        Some(root.join("training/runs.json")),
    );
    let j1 = plan_import(&cfg, req.clone(), &mac()).unwrap();
    let id = svc.enqueue_task(Box::new(j1)).unwrap();
    let rec = svc.wait_task(&id, Duration::from_secs(30)).unwrap();
    assert_eq!(rec.state, RunState::Done, "{}", rec.describe());
    assert!(
        rec.summary.as_deref().unwrap().contains("3 frames"),
        "{rec:?}"
    );
    assert_eq!(rec.steps, STAGES.len());
    // Saved beside runs.json.
    let saved = crate::training::task::load_records(&root.join("training/tasks.json"));
    assert_eq!(saved.len(), 1);
    assert!(svc.remove(&id).is_ok());
    assert!(svc.tasks().is_empty());
    svc.shutdown();
}
